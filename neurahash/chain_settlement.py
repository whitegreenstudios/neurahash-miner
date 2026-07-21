"""
neurahash/chain_settlement.py — settle pool rewards on the neura_l1 chain ledger (#5).

THE PROBLEM. The pool's payouts were an in-process Python dict the operator could rewrite, then a
coordinator-owned off-chain emission (HeightEmission) — the MONEY AUTHORITY lived with the operator, not
the chain. Issue #5: route accepted-work rewards through the neura_l1 chain ledger with signed receipts +
consensus-enforced emission cap, and reconstruct balances from the chain, not a dict.

THE FIX. Each settled height becomes a COORDINATOR-SIGNED neura_l1 Block whose reward is minted pro-rata
to the verified-work contributors via the SAME audited State.apply_block path the chain's consensus uses —
so the hard cap (MAX_SUPPLY) is enforced by chain code, balances are the neura_l1 State (reconstructed by
REPLAYING the signed blocks, never a mutable dict), and every settlement is replay-protected (height +
prev_hash chain) and signed. Anyone can replay the blocks and recompute identical balances + re-check the
signatures (verify()).

HONEST SCOPE. This routes settlement through the neura_l1 CHAIN LEDGER (State + apply_block + the MAX_SUPPLY
cap + signed blocks). It does NOT yet run the FULL PoUW consensus (validate_block / fork choice / networked
validators) — folding the pool fully into neura_l1 consensus is the coordinator-decentralization (L3)
milestone. The per-height reward AMOUNT follows the caller's schedule (the pool's reward_at); the chain
enforces the hard cap + the signed, replayable settlement.
"""
from __future__ import annotations

import hashlib
import json

from neura_l1.block_state import (State, Block, BlockHeader, tx_root,
                                  GENESIS_PREV_HASH, GENESIS_VERSION, MAX_SUPPLY)
from neura_l1.signing import (gen_account, account_from_key, sign_bytes, recover_bytes,
                              verify_only_account)


def _beacon(prev_hash):
    return hashlib.sha256(str(prev_hash).encode()).hexdigest()


def _quorum_hash(quorum):
    """Deterministic sha256 of a quorum bundle -- the value folded into BlockHeader.quorum_hash so the
    coordinator signature (over hdr.hash()) COVERS the exact attached quorum. sort_keys canonical JSON =>
    every node recomputes the identical digest from the block's own quorum field."""
    return hashlib.sha256(
        json.dumps(quorum, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verify_block_quorum(rec, hdr, quorum_roster, quorum_m):
    """(B8-1 enforce) INDEPENDENTLY re-verify that block `rec` carries an M-of-N staked-validator
    quorum authorizing ITS OWN mint -- recipient (= the sole contributor), the header reward, and the
    header height. Returns (ok, reason). This is what makes the QUORUM (not the single coordinator
    signature) the trust root: a malicious coordinator cannot forge validator signatures, cannot
    inflate the header reward (the validators signed a specific amount -> the rebuilt payload no longer
    recovers them), and in enforce mode cannot omit the quorum (a block with no/invalid quorum is
    rejected here). `quorum_roster`, when given, PINS the eligible validator set: the signers are
    re-recovered and counted only if they are in this set -- so the coordinator cannot self-select a
    shill roster (the residual that the proof's self-claimed roster otherwise leaves; the on-chain
    binding of this pinned set is B8-2). `quorum_m` overrides the required threshold (else the proof's
    own n_required, itself floor-checked to a strict majority by collect_settlement_signatures)."""
    from neurahash import diloco_settlement as _ds        # lazy: keep default replay import-free + acyclic
    q = rec.get("quorum")
    if not isinstance(q, dict) or not q:
        return False, "no quorum proof on block"
    # (B8-1) if the coordinator-signed header binds a quorum_hash, the attached quorum MUST match it -- this
    # is what makes the signature COVER the quorum, so a served-block-list mutator cannot swap in a different
    # bundle after signing. Empty quorum_hash = a pre-binding block; the (recipient,amount,height) + pinned
    # M-of-N below still bind the mint.
    hh = getattr(hdr, "quorum_hash", "")
    if hh and _quorum_hash(q) != hh:
        return False, "quorum does not match the signed header quorum_hash"
    # ENFORCE REQUIRES A PINNED ROSTER (fail-closed): verifying against the proof's OWN self-stated
    # roster would let a malicious coordinator name a shill validator set -- the exact residual that
    # pinning closes -- so enforce without a pinned set is REFUSED, not silently degraded to
    # audit-grade. (Observe-only auditing of a self-stated roster stays in
    # diloco_settlement.audit_settlement_block_quorums, which never rejects a block.)
    if quorum_roster is None:
        return False, "enforce requires a pinned validator roster (quorum_roster)"
    contribs = rec.get("contributors", {}) or {}
    recipient = next(iter(contribs)) if len(contribs) == 1 else None
    # PINNED roster: re-recover signers over THIS block's decision, count only pinned members.
    # A {addr: stake} roster (B8-2/B8-4, e.g. staked_roster over on-chain bonds) authorizes on a strict
    # majority of STAKE -- the Sybil-by-headcount defense; a bare address list/set stays count-weighted
    # (byte-identical to today). Do NOT flatten a dict: list(dict) would drop the stake and silently
    # downgrade the authoritative path to count-majority (the gap the on-chain B8-4 test surfaced).
    roster_arg = quorum_roster if isinstance(quorum_roster, dict) else list(quorum_roster)
    try:
        v = _ds.collect_settlement_signatures(
            recipient=recipient, amount=hdr.reward, height=hdr.height,
            delta_cid=q.get("delta_cid", ""), roster=roster_arg,
            attestations=q.get("sigs", []), m=quorum_m)
    except Exception as e:
        return False, f"quorum verify error ({type(e).__name__})"
    if not v.get("approved"):
        return False, f"pinned-roster quorum not met ({v.get('n_signers')}/{v.get('n_required')})"
    return True, "ok"


def _replay_blocks(blocks, coord_address, genesis_checkpoint,
                   *, require_quorum=False, quorum_roster=None, quorum_m=None):
    """Replay a signed block chain from genesis into a fresh State, checking every block's
    coordinator signature against `coord_address` (a plain string -- no `Account`/private key
    required) and the prev_hash chain. Module-level free function so replay does not require a
    live `ChainSettlement` instance: a read-only verifier (e.g. the wallet app's chain_sync
    client, which never holds the coordinator's key) can call this directly with a publicly
    fetched block list plus a pinned public address. Returns (state, ok, reason): ok is False
    (with a reason) if any block's coordinator signature fails, the prev_hash chain is broken, or
    a block does not apply cleanly.

    (B8-1 authoritative quorum) When `require_quorum=True`, EACH block must ALSO carry a valid
    M-of-N staked-validator quorum authorizing its own mint (see `_verify_block_quorum`); a block
    without one, or whose quorum does not bind to the block's (recipient, reward, height), is
    REJECTED -- making the quorum, not the lone coordinator signature, the trust root. Default
    `require_quorum=False` => byte-identical to the single-coordinator replay above (the quorum
    field, if present, is ignored exactly as before)."""
    st = State(model_checkpoint=genesis_checkpoint)
    head = GENESIS_PREV_HASH
    for i, rec in enumerate(blocks):
        hdr = BlockHeader.from_dict(rec["header"])
        if hdr.prev_hash != head:
            return st, False, f"block {i}: broken chain (prev_hash != head)"
        try:
            signer = recover_bytes(hdr.hash().encode(), rec["sig"])
        except Exception as e:
            return st, False, f"block {i}: bad signature ({e})"
        if signer.lower() != str(coord_address).lower():
            return st, False, f"block {i}: signed by {signer}, not the coordinator"
        if require_quorum:
            ok_q, reason_q = _verify_block_quorum(rec, hdr, quorum_roster, quorum_m)
            if not ok_q:
                return st, False, f"block {i}: quorum enforce failed ({reason_q})"
        try:
            st = st.apply_block(Block(header=hdr, txs=[], pouw={}), rec["contributors"])
        except Exception as e:
            return st, False, f"block {i}: does not apply ({e})"
        head = hdr.hash()
    return st, True, "ok"


class ChainSettlement:
    """A coordinator-signed neura_l1 settlement chain for pool rewards. The neura_l1 State is the balance
    authority; balances reconstruct by replaying the signed blocks. One per coordinator."""

    def __init__(self, coordinator_account=None, genesis_checkpoint="pool-genesis"):
        self.coord = coordinator_account or gen_account()
        self.genesis_checkpoint = genesis_checkpoint
        self.state = State(model_checkpoint=genesis_checkpoint)
        self.blocks = []                       # list of {header, contributors, sig} — the signed chain
        self.head = GENESIS_PREV_HASH          # prev_hash for the next block
        # (DiLoCo GAP2 restart-replay guard) The set of delta_cids ALREADY rewarded through this chain.
        # The ONLY durable, per-CID idempotency key for the external DiLoCo reward path: the merge poller's
        # in-memory dedup (_diloco_seen) is rebuilt EMPTY on every coordinator restart, so after a routine
        # restart the poller re-reads the persisted VPS manifest and re-queues every contribution as "new";
        # without this the reward path would re-settle (double-mint) the SAME delta_cid on every restart
        # (unbounded under merge=audit + reward=enforce, where the trunk never moves so the gain re-measures
        # positive forever). Persisted alongside the signed blocks (to_state/from_state) so it survives a
        # restart with the SAME durability as the blocks it guards. Populated only by mark_paid(), which the
        # reward path calls in `enforce` after a successful settle — so it stays EMPTY (and to_state stays
        # byte-identical to today) whenever the DiLoCo reward is off/audit.
        self.paid_cids = set()

    # ----------------------------- settle -----------------------------
    def settle(self, height, reward_pot, ex_by, timestamp, *, quorum=None):
        """Settle one completed height: mint `reward_pot` NRH pro-rata to the verified-work weights
        `ex_by` ({address: weight}) as a coordinator-signed neura_l1 block applied via State.apply_block
        (cap-enforced). Returns {address: credited_amount} (the chain balance delta). A height with no
        positive-weight contributors mints nothing.

        (B8-1) `quorum` optionally embeds the M-of-N staked-settlement proof (diloco_settlement.
        settlement_block_proof) for THIS mint in the block record, so a replayer can independently re-check
        WHO authorized it. DEFAULT None => the block record is BYTE-IDENTICAL to today (no 'quorum' key). The
        field is NON-AUTHORITATIVE for chain acceptance: `_replay_blocks` still governs on the single
        coordinator signature; making the on-chain quorum load-bearing is a later B8 increment."""
        contributors = {a: float(w) for a, w in ex_by.items() if float(w) > 0}
        ck = self.state.model_checkpoint
        hdr = BlockHeader(
            height=int(height), prev_hash=self.head, beacon=_beacon(self.head),
            parent_checkpoint=ck, new_checkpoint=ck,
            work_score=float(sum(contributors.values())),
            proposer=self.coord.address, reward=float(max(0.0, reward_pot)),
            tx_root=tx_root([]), timestamp=int(timestamp), protocol_version=GENESIS_VERSION)
        if quorum is not None:                                  # (B8-1) bind the quorum INTO the signed hash
            hdr.quorum_hash = _quorum_hash(quorum)             # so the signature below COVERS it (no strip/swap)
        sig = sign_bytes(self.coord, hdr.hash().encode())       # signed receipt over the block hash (+quorum)
        before = {a: self.state.balance(a) for a in contributors}
        self.state = self.state.apply_block(Block(header=hdr, txs=[], pouw={}), contributors)
        self.head = hdr.hash()
        rec = {"header": hdr.to_dict(), "contributors": contributors, "sig": sig}
        if quorum is not None:                                  # (B8-1) durable, independently-verifiable M-of-N
            rec["quorum"] = quorum                              # proof; absent => block byte-identical to today.
        # (B8-1) `quorum` is embedded so a replayer re-verifies the M-of-N. sha256(quorum) is ALSO folded into
        # hdr.quorum_hash ABOVE (before signing), so the coordinator signature now COVERS the quorum: a
        # served-block-list mutator that strips or swaps it either invalidates the signature or fails the
        # _verify_block_quorum header-binding check under enforce. The rec["quorum"] bytes stay the bundle the
        # verifier re-recovers the M-of-N over.
        self.blocks.append(rec)
        return {a: round(self.state.balance(a) - before.get(a, 0.0), 12) for a in contributors}

    # ----------------------------- reads -----------------------------
    def balance(self, address):
        return self.state.balance(address)

    def minted(self):
        return self.state.minted

    def height(self):
        return self.blocks[-1]["header"]["height"] if self.blocks else 0

    # ------------------- DiLoCo reward idempotency (restart-replay guard) -------------------
    def already_paid(self, cid):
        """True iff `cid` was ALREADY rewarded through this settlement chain (see self.paid_cids). The
        DiLoCo reward decision consults this BEFORE chain.settle so a re-queued/replayed contribution
        cannot double-mint the same delta across a coordinator restart. A falsy cid is never 'paid'
        (there is nothing to dedup on)."""
        return bool(cid) and cid in self.paid_cids

    def mark_paid(self, cid):
        """Record `cid` as rewarded so a later restart (which rebuilds the poller's in-memory dedup empty)
        or a duplicate manifest record cannot re-pay it. Idempotent; a falsy cid is ignored. Travels with
        the blocks in to_state/from_state, so it is exactly as durable as the mint it guards."""
        if cid:
            self.paid_cids.add(cid)

    # ----------------------------- replay + verify -----------------------------
    def _replay(self):
        """Replay the signed block chain from genesis into a fresh State, against this settlement's
        own coordinator address. Thin instance wrapper over the module-level free function
        `_replay_blocks` (see below), which does the actual work without needing a live
        `ChainSettlement`/`Account` -- that split is what lets a read-only verifier (no private key)
        replay a chain it only has the coordinator's public address for. Returns (state, ok, reason):
        ok is False (with a reason) if any block's coordinator signature fails, the prev_hash chain is
        broken, or a block does not apply cleanly."""
        return _replay_blocks(self.blocks, self.coord.address, self.genesis_checkpoint)

    def verify(self):
        """Re-check the whole settlement chain: every block's coordinator signature, the prev_hash links,
        clean application, AND that the replayed balances equal the live state. A tampered reward/amount
        or a forged block fails here — this is what makes the chain-settled balances trustworthy to
        audit. Returns (ok, reason)."""
        st, ok, reason = self._replay()
        if not ok:
            return False, reason
        live = self.state.bal
        for a in set(live) | set(st.bal):
            if abs(st.balance(a) - live.get(a, 0.0)) > 1e-9:
                return False, f"balance mismatch for {a}: replay {st.balance(a)} != live {live.get(a, 0.0)}"
        if abs(st.minted - self.state.minted) > 1e-9:
            return False, "minted mismatch on replay"
        return True, "ok"

    # ----------------------------- public export (read-only verifiers) -----------------------------
    def public_state(self):
        """Serialize the PUBLIC-SAFE view of the settlement chain for third-party verifiers (e.g.
        the wallet app's chain_sync client) -- the coordinator's public address (never the private
        key), the genesis checkpoint, and the signed block list. Deliberately omits `coord_key`
        (private key material, to_state()-only, see the docstring there) and `paid_cids` (internal
        DiLoCo restart-replay bookkeeping -- not needed to reconstruct balances) to minimize the
        exported surface. Pair with the module-level `verify_public_chain()` on the reader side."""
        return {"coord_address": self.coord.address, "genesis_checkpoint": self.genesis_checkpoint,
                "blocks": [dict(b) for b in self.blocks]}

    # ----------------------------- persistence (coordinator resume) -----------------------------
    def to_state(self, include_key=True):
        """Serialize for the coordinator checkpoint.

        include_key=True (default): the coordinator's signing key (hex) is included so the signed chain
        stays verifiable against the key that produced it across a restart — the LEGACY layout, identical
        to before F10.

        include_key=False (F10 key-scrub): the private key is OMITTED (only the public `coord_address` is
        emitted), so the chain remains replay-verifiable but the settlement key never rides inside the
        checkpoint. The key is exported once to a separate encrypted keyfile; a failover node decrypts it
        to restore signing (see neurahash.coord_checkpoint)."""
        d = ({"coord_key": self.coord.key.hex()} if include_key
             else {"coord_address": self.coord.address})
        d.update({"genesis_checkpoint": self.genesis_checkpoint,
                  "blocks": [dict(b) for b in self.blocks]})
        # (DiLoCo GAP2) Persist the paid-cid index ONLY when non-empty so a coordinator running with the
        # DiLoCo reward off/audit (paid_cids always empty) serializes a blob BYTE-IDENTICAL to today.
        # from_state() defaults an absent field to an empty set (back-compat with pre-guard checkpoints).
        if self.paid_cids:
            d["paid_cids"] = sorted(self.paid_cids)
        return d

    @classmethod
    def from_state(cls, state, coord_key=None, sign_disabled_reason=None):
        """Rebuild from to_state() and REPLAY the signed chain to reconstruct balances (so a tampered
        checkpoint fails the integrity check here, not silently). Raises ValueError on a broken chain.

        F10 key handling mirrors SignedPoolLedger.from_state: a supplied `coord_key` (from the decrypted
        sidecar keyfile) or an embedded `coord_key` (legacy/scrub-OFF) restores full block signing; a
        SCRUBBED state with neither yields a VERIFY-ONLY chain that replays/serves but cannot mint a new
        settlement block (any sign attempt raises `sign_disabled_reason`)."""
        embedded = state.get("coord_key")
        key = coord_key if coord_key is not None else embedded
        if key is not None:
            account = account_from_key(key)
            addr = state.get("coord_address")
            if addr and account.address.lower() != str(addr).lower():
                raise ValueError(
                    f"coord key/address mismatch: the supplied signing key resolves to {account.address}, "
                    f"but this settlement chain belongs to {addr} (wrong keyfile for this checkpoint)")
        else:
            addr = state.get("coord_address")
            if not addr:
                raise ValueError("scrubbed settlement chain has neither an embedded coord_key nor a "
                                 "coord_address and no key was supplied; cannot restore")
            account = verify_only_account(addr, sign_disabled_reason)
        cs = cls(account, genesis_checkpoint=state.get("genesis_checkpoint", "pool-genesis"))
        cs.blocks = [dict(b) for b in state["blocks"]]
        st, ok, reason = cs._replay()
        if not ok:
            raise ValueError(f"restored settlement chain failed integrity check: {reason}")
        cs.state = st
        cs.head = BlockHeader.from_dict(cs.blocks[-1]["header"]).hash() if cs.blocks else GENESIS_PREV_HASH
        cs.paid_cids = set(state.get("paid_cids", []))     # back-compat: absent field -> no CID yet paid
        return cs


def verify_public_chain(export, *, expected_coord_address=None,
                        require_quorum=False, quorum_roster=None, quorum_m=None):
    """Verify a `public_state()` export with no private key and no live `ChainSettlement`
    instance -- the read-only verifier path (e.g. the wallet app's `chain_sync.ChainSyncClient`).

    Anti-spoofing design (this is the important part -- see the scoping doc's Risk #2): when
    `expected_coord_address` is supplied, the chain is replayed against THAT address only.
    `export["coord_address"]` is NEVER consulted for the trust decision in that case -- it is
    informational only, because an attacker who controls the export (a compromised/malicious
    registry, or a leaked publish token) fully controls that field too. A chain that is internally
    self-consistent (every signature checks out, every prev_hash link is intact) but signed by a
    DIFFERENT key than the pinned one -- e.g. an attacker publishing their own throwaway-signed
    chain to the same registry slot -- fails here even though `export["coord_address"]` would
    otherwise claim to be the real coordinator. When `expected_coord_address` is omitted, this
    falls back to trusting `export["coord_address"]` itself and marks the result "unpinned" in
    `reason`, so callers can tell a pinned-and-verified chain apart from a merely
    self-consistent one.

    Returns `(ok, reason, balances, minted, height)`. `balances` is `{address: float}` and
    `minted` a float; both are empty/zero when `ok` is False.

    Known limitation, BY DESIGN, not a bug (scoping doc Risk #1): a strict PREFIX of a genuine
    chain -- i.e. the real tail withheld by whoever served this export -- replays and verifies
    cleanly here, just with a lower balance/height than the true chain. Signature + link replay
    alone cannot detect withholding; it only proves that everything IN the export is genuine and
    self-consistent, not that nothing was left out. Callers that need withholding/rollback
    detection must track height against an independent local ratchet themselves (see
    `wallet_app/backend/chain_sync.py`'s height-ratchet)."""
    if not isinstance(export, dict):
        return False, "malformed export: not a dict", {}, 0.0, 0
    blocks = export.get("blocks")
    if not isinstance(blocks, list):
        return False, "malformed export: missing or non-list 'blocks'", {}, 0.0, 0
    genesis_checkpoint = export.get("genesis_checkpoint", "pool-genesis")
    pinned = expected_coord_address is not None
    coord_address = expected_coord_address if pinned else export.get("coord_address")
    if not coord_address:
        return False, "malformed export: no 'coord_address' in export and none pinned", {}, 0.0, 0
    st, ok, reason = _replay_blocks(blocks, coord_address, genesis_checkpoint,
                                    require_quorum=require_quorum, quorum_roster=quorum_roster,
                                    quorum_m=quorum_m)
    if not ok:
        return False, reason, {}, 0.0, 0
    height = int(blocks[-1]["header"]["height"]) if blocks else 0
    if not pinned:
        reason = "ok (UNPINNED: trusted the export's own coord_address, not independently verified)"
    elif require_quorum:
        reason = "ok (quorum-enforced: every block M-of-N staked-validator authorized)"
    return True, reason, dict(st.bal), float(st.minted), height


# re-export for tests/callers that want the chain's hard cap
CHAIN_MAX_SUPPLY = MAX_SUPPLY
