"""
neurahash/diloco_round_authority.py -- B8-3 Stage B (LOCAL proof-of-mechanism): a GENUINELY
two-node-agreed round-authority DRIVE + legitimate failover, built on the commit-reveal
primitives already in `neurahash/diloco_clock.py`.

WHAT THIS FIXES. Today ONE coordinator unilaterally advances the clock -- `r += 1` at
sharded_pool_node.py:5813 -- and drives every round: a single point of trust AND of failure. The
D4 clock (diloco_clock.py) shipped the trustless PRIMITIVE that removes the say-so in principle
(commit-reveal `agree_round` -> a deterministic, un-grindable proposer ORDER any node re-derives),
but its Stage-B `enforce` deliberately RAISES `NotImplementedError` rather than "silently fake a
decentralized clock": it had no DRIVE LOOP, no FAILOVER, and no second-candidate transport. This
module supplies exactly those three missing pieces, scoped as a DEFAULT-OFF LOCAL proof:

  * DRIVE LOOP  -- N rounds are driven across >=2 candidate coordinators. Each round every candidate
    COMMITS (make_commit) -> broadcasts its commit_envelope -> REVEALS (reveal_of) -> and EACH node
    INDEPENDENTLY calls `diloco_clock.agree_round` over the shared commit/reveal set. The agreed
    `order[0]` is the elected proposer; it produces that round's settlement block
    (`ChainSettlement.settle`) and the others VERIFY it.
  * FAILOVER    -- if the elected proposer does not produce within a bounded number of ticks, the
    NEXT signer in the AGREED order takes over. This is LEGITIMATE, not a unilateral seizure: the
    order came from the commit-reveal beacon (every node re-derives the identical permutation), so
    "who is next" is not any one node's decision.
  * FAILURE DETECTION (this build) -- the TRIGGER for that failover is now itself a CONSENSUS output,
    not a caller-supplied `stalled` flag. A candidate may only assume the elected primary's slot after
    an M-of-N QUORUM of candidates have each SIGNED a primary-timeout / view-change attestation for
    that (round, failed_proposer, view) -- `sign_timeout` / `verify_timeout_quorum`, reusing the exact
    `guardian_halt.GuardianSet` distinct-signer M-of-N the D3 settlement quorum uses. This is a
    VIEW-CHANGE: view 0 = order[0] proposes; on a PROVEN timeout, view 1 = order[1], etc. Fork-safety:
    a secondary cannot forge M-of-N signatures, so if the primary was actually producing (honest
    candidates saw its proposal and did NOT sign a timeout), no quorum forms and the out-of-view
    proposal is REJECTED in `Candidate.accept` BEFORE any state mutation (`agreed_failover=True`).
  * TRANSPORT   -- a minimal in-process `LoopbackBus` stands in for the networked `ClockMesh`
    (diloco_clock_net.py) so the two candidate roles can exchange envelopes in one process. The
    prompt allows a loopback transport for the proof; the production seam is still
    `diloco_clock._remote_clock_envelopes` / `diloco_clock_net.ClockMesh`.

IS IT GENUINELY AGREED (not a relabeled single-node clock)? YES. Both candidates hold DISTINCT
secp256k1 identity keys; each seals its own secret nonce BEFORE seeing the other's (commit precedes
reveal); the round seed mixes BOTH nonces (`agree_seed`), and the order is `H(seed|signer)` per
signer (`agree_order`) -- so neither node can bias the order toward itself, and BOTH independently
compute the byte-identical `(round, seed, order)`. The elected proposer AUTHENTICATES itself with
its identity key (a signed proposer-claim bound to the block hash); a node driving a round it was
NOT elected for recovers to the wrong identity for that slot and is REJECTED. A candidate that
commits TWO different digests at one round is an EQUIVOCATION -- `agree_round` drops it from the
beacon (it lands in disputes["commit_equivocations"]) and it can never be the proposer.

DEFAULT-OFF, BYTE-IDENTICAL. The whole drive is gated behind `NEURAHASH_DILOCO_ROUND_AUTHORITY`
(default off). Unset, `advance_round` returns the plain single-node `r + 1` and never touches the
commit-reveal machinery -- so a live coordinator with the flag off behaves exactly as today. This
module is NOT wired into sharded_pool_node (no import there); the live-hook is design-only, mirroring
the clock's Stage-B seam.

HONEST SCOPE (what is proven LOCALLY vs what production still needs):
  * PROVEN here: the round NUMBER + proposer ORDER are decided by a real 2-key commit-reveal
    agreement, not one node's `r += 1`; a stalled primary fails over to the next AGREED signer with
    no gap and no fork -- and the failover TRIGGER is a genuine M-of-N candidate-signed timeout QUORUM,
    so a unilateral seizure of a producing primary's slot is provably REJECTED; an out-of-turn
    proposer is rejected; an equivocator is excluded; and a legitimately-elected takeover's blocks
    still satisfy today's B8-1 authoritative quorum (`verify_public_chain(require_quorum=True, ...)`).
  * PARTITION residual (2-node roster): with n_candidates=2 and the strict-majority M=2, a timeout
    quorum CANNOT form once one node is down (only 1 signer remains) -- fork-SAFE (no takeover without
    both) at the cost of LIVENESS (no failover). Partition-tolerant AGREED failover needs
    n_candidates >= 3 (M=2-of-3: the two surviving candidates form the quorum). This is the standard
    BFT liveness-vs-safety split, not a defect.
  * NOT solved here (same residuals the D-series/B8 docs name): (1) #45 CROSS-ARCH DETERMINISM --
    if a REMOTE node drives, its honest ~1-ULP cross-vendor numerics must be distinguishable from a
    fraud before a takeover can be slashable; (2) a REAL NETWORK TRANSPORT for a second always-on
    candidate coordinator (here a loopback bus); (3) an ON-CHAIN VALIDATOR SET -- the settlement
    BLOCK SIGNING key is still the single coordinator role key (the identity that travels in
    coord_checkpoint.pt and that a legitimate failover node holds); B8-3 decentralizes WHO DRIVES
    the round, while B8-1's M-of-N quorum is what actually gates the MINT. This is a
    proof-of-mechanism, NOT production trustlessness.

Pure stdlib + `neura_l1.signing` + `neura_l1.block_state` + `neurahash.diloco_clock` +
`neurahash.chain_settlement` + `neurahash.diloco_settlement`. NO torch, NO import of
sharded_pool_node. ASCII-only output (the coordinator console is cp1252).
"""
import copy
import json
import os
from collections import defaultdict

from neura_l1.signing import gen_account, sign_bytes, recover_bytes
from neura_l1.block_state import State, Block, BlockHeader
from neurahash.chain_settlement import ChainSettlement
from neurahash.guardian_halt import GuardianSet
from neurahash import diloco_clock as _dc
from neurahash import diloco_settlement as _ds

# --------------------------------------------------------------------------- flag
FLAG = "NEURAHASH_DILOCO_ROUND_AUTHORITY"        # default OFF -> single-node r += 1, byte-identical
RA_TAG = "neurahash-round-authority/v1"           # domain tag for the proposer self-authentication claim
RA_TIMEOUT_TAG = "neurahash-round-authority-timeout/v1"  # DISTINCT domain tag for a primary-timeout / view-change vote
_TRUTHY = ("1", "on", "true", "enforce", "enabled", "yes")


def round_authority_enabled(env=None):
    """Read NEURAHASH_DILOCO_ROUND_AUTHORITY. Returns True only for an explicit truthy value; ANY other
    / absent value fails SAFE to False (byte-identical single-node clock). Mirrors diloco_clock.clock_mode's
    fail-safe parse."""
    v = (os.environ if env is None else env).get(FLAG, "off")
    return str(v).strip().lower() in _TRUTHY


# --------------------------------------------------------------------------- proposer self-authentication
def _norm(addr):
    return str(addr).lower()


def proposer_claim_payload(round_id, proposer, block_hash):
    """Canonical bytes the elected proposer SIGNS with its IDENTITY key to authenticate that it -- and not
    some other node -- produced this round's block. Binds (round, proposer, block_hash) so the claim cannot
    be replayed onto a different round or a different block (sorted-keys JSON, ASCII)."""
    return json.dumps({"w": RA_TAG, "round": int(round_id), "proposer": _norm(proposer),
                       "block_hash": str(block_hash)}, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_proposer_claim(identity_account, round_id, block_hash):
    """The elected proposer's signed claim over its round + block, using the repo's real secp256k1
    `sign_bytes`. `proposer` is the account address but is NEVER trusted -- `recover_proposer` re-recovers it."""
    return {"round": int(round_id), "proposer": identity_account.address, "block_hash": str(block_hash),
            "sig": sign_bytes(identity_account, proposer_claim_payload(round_id, identity_account.address, block_hash))}


def recover_proposer(claim):
    """(ok, recovered_addr) for a proposer claim: rebuild the canonical payload from the claim's OWN fields and
    re-recover the signer. A tampered field (edited round/block_hash/proposer) recovers a DIFFERENT address than
    the claimed `proposer` -> (False, _). Never raises."""
    if not isinstance(claim, dict):
        return False, None
    try:
        payload = proposer_claim_payload(claim["round"], claim["proposer"], claim["block_hash"])
        rec = recover_bytes(payload, claim["sig"])
    except Exception:
        return False, None
    claimed = claim.get("proposer")
    if not isinstance(claimed, str) or _norm(rec) != _norm(claimed):
        return False, rec
    return True, rec


# --------------------------------------------------------------------------- AGREED failure-detection (view-change)
# The failover TRIGGER is a CONSENSUS output, not one node's say-so. Today's `stalled` down-set was
# caller-supplied/simulated, so a secondary could unilaterally declare the primary down while it was
# producing -> two proposers -> fork. These three functions make the trigger un-forgeable: a candidate may
# only assume the elected primary's slot after an M-of-N QUORUM of candidates have each SIGNED a
# primary-timeout / view-change attestation for that (round, failed_proposer, view). A secondary cannot
# forge M-of-N signatures, so if the primary was actually producing (honest candidates saw its proposal and
# did NOT sign a timeout), no quorum forms and the out-of-view proposal is rejected in `Candidate.accept`.
def _sig_of(att):
    """Raw signature hex out of a timeout attestation dict (from `sign_timeout`) or a bare hex string."""
    if isinstance(att, dict):
        return att.get("sig")
    return att


def timeout_payload(round_id, failed_proposer, view):
    """Canonical bytes a candidate SIGNS to attest 'the view-`view` proposer `failed_proposer` did not
    produce this round'. Binds (round, failed_proposer, view) under a DISTINCT domain tag (RA_TIMEOUT_TAG)
    so a timeout signature can NEVER be reinterpreted as a proposer-claim (RA_TAG) or a settlement vote
    (diloco_settlement.SETTLE_TAG), and a timeout bound to one (round, view) cannot be replayed onto another
    round/view/proposer (sorted-keys JSON, ASCII -- the coordinator console is cp1252). Mirrors
    diloco_settlement.settlement_payload / diloco_clock.clock_commit_payload."""
    return json.dumps({"w": RA_TIMEOUT_TAG, "round": int(round_id),
                       "failed_proposer": _norm(failed_proposer), "view": int(view)},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_timeout(candidate, round_id, failed_proposer, view):
    """ONE candidate-coordinator's SIGNED primary-timeout / view-change attestation over the canonical
    timeout message, using the repo's real secp256k1 `sign_bytes` with the candidate's IDENTITY key. A
    candidate emits this when it OBSERVES no valid proposal from the current-view proposer within the bounded
    window. `signer` is the address but is NEVER trusted -- `verify_timeout_quorum` RE-RECOVERS it. Accepts a
    `Candidate` (uses `.identity`) or a bare signing account."""
    acct = getattr(candidate, "identity", candidate)
    return {"round": int(round_id), "failed_proposer": _norm(failed_proposer), "view": int(view),
            "signer": acct.address, "sig": sign_bytes(acct, timeout_payload(round_id, failed_proposer, view))}


def verify_timeout_quorum(attestations, roster, m, round_id, failed_proposer, view):
    """Re-recover the DISTINCT on-roster candidate signers of a primary-timeout / view-change over the
    canonical (round, failed_proposer, view) message and decide whether an M-of-N view-change quorum formed
    -- the fork-safety gate. REUSES `guardian_halt.GuardianSet.distinct_signers` (the SAME distinct-signer
    k-of-m the D3 settlement quorum uses via `diloco_settlement.collect_settlement_signatures`), so ONLY
    signature-valid attestations whose RECOVERED signer is IN the candidate roster count, one vote per
    distinct signer; a forged/edited field, an off-roster key, a duplicate identity, or a signature over a
    DIFFERENT (round, proposer, view) is DROPPED.

    `m` MUST be a STRICT MAJORITY of the roster (`1 <= m <= N and m*2 > N`) -- so a minority, and in
    particular a lone secondary, can NEVER forge a quorum (the fork-prevention property). A
    degenerate/minority `m`, an empty roster, or malformed input fails CLOSED (approved=False), never raises.

    Returns {approved, n_signers, n_required, roster_n, signers, round, failed_proposer, view}."""
    members = [str(a) for a in (roster or [])]
    N = len({_norm(a) for a in members})
    try:
        m = int(m)
    except (TypeError, ValueError):
        m = 0
    out = {"approved": False, "n_signers": 0, "n_required": int(m), "roster_n": int(N), "signers": [],
           "round": int(round_id), "failed_proposer": _norm(failed_proposer), "view": int(view)}
    if N == 0 or not (1 <= m <= N and m * 2 > N):
        out["reason"] = "empty roster or threshold is not a strict majority of the candidate set"
        return out
    payload = timeout_payload(round_id, failed_proposer, view)
    sigs = [s for s in (_sig_of(a) for a in (attestations or [])) if s is not None]
    try:
        gs = GuardianSet(members, m)                          # distinct-signer k-of-m over the candidate roster (reused)
        signers = gs.distinct_signers(payload, sigs)          # distinct on-roster signers over THIS exact decision
    except Exception:
        return out
    n = len(signers)
    out.update(approved=bool(n >= m), n_signers=int(n), signers=sorted(signers))
    return out


# --------------------------------------------------------------------------- election policy (deterministic)
def elect_proposer(order, unavailable=frozenset()):
    """The proposer for a round given the AGREED order + a set of known-down signers: the FIRST signer in
    `order` that is not unavailable. This is the entire failover rule -- pure and deterministic, so every node
    that holds the same agreed order and observes the same down-set elects the identical proposer. `order` came
    from commit-reveal (agree_round), so this is NOT a unilateral claim. Returns None if every signer is down."""
    un = {_norm(s) for s in (unavailable or ())}
    for ident in (order or []):
        if _norm(ident) not in un:
            return ident
    return None


# --------------------------------------------------------------------------- in-process transport (proof only)
class LoopbackBus:
    """Minimal in-process broadcast transport standing in for the networked `ClockMesh` (diloco_clock_net) for
    a LOCAL proof. Every candidate PUBLISHES its public commit envelope / reveal here; every candidate READS the
    SAME shared set back, so each node runs `agree_round` over identical inputs and independently reaches the
    identical (round, seed, order). NO TRUST: `agree_round` re-verifies every reveal against its commit, so a
    forged/withheld/equivocating envelope on the bus is caught by agreement, not by the bus."""

    def __init__(self):
        self._commits = defaultdict(list)     # round -> [commit_env]
        self._reveals = defaultdict(list)     # round -> [reveal]

    def publish_commit(self, round_id, env):
        self._commits[int(round_id)].append(dict(env))

    def publish_reveal(self, round_id, rev):
        self._reveals[int(round_id)].append(dict(rev))

    def commits(self, round_id):
        return [dict(x) for x in self._commits[int(round_id)]]

    def reveals(self, round_id):
        return [dict(x) for x in self._reveals[int(round_id)]]


# --------------------------------------------------------------------------- a candidate coordinator role
class Candidate:
    """One candidate-coordinator: a DISTINCT identity key (for commit-reveal + proposer authentication) plus a
    REPLICA of the canonical settlement chain (its own `ChainSettlement`, sharing the coordinator role signing
    key + genesis with its peers -- the key a legitimate failover node holds). It seals its own secret nonce
    each round, agrees the round independently, produces the block when elected, and verifies + applies peers'
    blocks otherwise. Replicas stay byte-identical across nodes (same accepted block sequence) -> no fork."""

    def __init__(self, identity_account, coord_account, *, genesis_checkpoint):
        self.identity = identity_account                    # distinct per candidate (commit-reveal + claim)
        self.coord = coord_account                          # SHARED coordinator role key (block signer)
        self.chain = ChainSettlement(coordinator_account=coord_account, genesis_checkpoint=genesis_checkpoint)
        self._sealed = {}                                   # round -> sealed commit (holds the secret nonce)
        self.last_agreement = None                          # this node's independently-computed agree_round dict

    # ---- commit-reveal phases ----
    def make_round_commit(self, round_id):
        """Seal a SECRET-nonce commit for `round_id` and return the PUBLIC commit envelope (digest, no nonce)."""
        sealed = _dc.make_commit(round_id, self.identity.address)
        self._sealed[int(round_id)] = sealed
        return _dc.commit_envelope(sealed)

    def round_reveal(self, round_id):
        """The reveal (nonce) for a round previously committed by this candidate."""
        return _dc.reveal_of(self._sealed[int(round_id)])

    def agree(self, round_id, bus):
        """INDEPENDENTLY agree the round: read the shared commit/reveal set off the bus and call the pure
        `agree_round`. Stores + returns {round, seed, order, valid_signers, n_valid, disputes}. Two honest
        candidates over the same bus set produce identical results (that is assert #1)."""
        self.last_agreement = _dc.agree_round(bus.commits(round_id), bus.reveals(round_id), round_id)
        return self.last_agreement

    # ---- primary-timeout attestation (AGREED failure-detection) ----
    def sign_timeout(self, round_id, failed_proposer, view):
        """This candidate's SIGNED primary-timeout / view-change attestation (delegates to the module-level
        `sign_timeout`): emitted when it OBSERVES no valid proposal from the view-`view` proposer."""
        return sign_timeout(self, round_id, failed_proposer, view)

    # ---- block production (when elected) ----
    def propose(self, round_id, *, height, reward, recipient, quorum, timestamp, view=None, timeout_quorums=None):
        """Produce THIS round's settlement block on this candidate's chain replica (single recipient, so the
        B8-1 quorum's recipient-binding holds), embed the M-of-N quorum, and sign a proposer-claim with the
        IDENTITY key. Returns the broadcast envelope {round, proposer, block, claim}.

        AGREED-failover path: when `view` is given, the envelope ALSO carries the view index and
        `timeout_quorums` (a {prior_view: [timeout attestations]} map) -- the proof that EACH prior view timed
        out, which a view>0 proposer MUST attach and every peer RE-VERIFIES in `accept`. When `view` is None
        (legacy path) the envelope omits both and acceptance uses the plain elected-order rule
        (byte-identical to before)."""
        self.chain.settle(int(height), float(reward), {recipient: 1.0}, int(timestamp), quorum=quorum)
        rec = self.chain.blocks[-1]
        block_hash = BlockHeader.from_dict(rec["header"]).hash()
        claim = sign_proposer_claim(self.identity, round_id, block_hash)
        env = {"round": int(round_id), "proposer": self.identity.address,
               "block": copy.deepcopy(rec), "claim": claim}
        if view is not None:
            env["view"] = int(view)
            env["timeout_quorums"] = {int(k): list(v) for k, v in (timeout_quorums or {}).items()}
        return env

    # ---- block acceptance (when a peer) ----
    def accept(self, envelope, order, unavailable=frozenset(), *, timeout_m=None, roster=None):
        """Verify + apply a peer's proposed block to this replica. Returns (ok, reason). REJECTS unless:
          (1) the proposer-claim signature re-recovers to the signer it claims;
          (2) that signer is EXACTLY the legitimately-elected proposer for this round;
          (3) the claim is bound to THIS block (block_hash matches);
          (4) the block extends our head (no gap / no fork) and carries a valid coordinator signature.
        Only then is the block appended and the head/state advanced -- so every node applies the identical
        accepted sequence. The (2) election check has TWO modes:

          * AGREED-failover / view-change (envelope carries `view`): the expected proposer is the view-v
            ELECTED node `order[view]`, and a takeover at view>0 must carry a VALID M-of-N timeout quorum
            (`verify_timeout_quorum`) for EACH prior view 0..view-1 -- otherwise REJECT "unilateral seizure"
            BEFORE any state mutation. The caller-supplied down-set is NOT consulted here (that was the fork
            hole): the trigger is the signed quorum, not one node's claim.
          * legacy (no `view`): the expected proposer is `elect_proposer(order, unavailable)` -- the shipped
            behaviour, kept byte-identical so the original proof tests are unchanged."""
        ok, who = recover_proposer(envelope.get("claim"))
        if not ok:
            return False, "proposer claim signature invalid"
        if "view" in envelope:
            try:
                view = int(envelope["view"])
            except (TypeError, ValueError):
                return False, "malformed view"
            if not (0 <= view < len(order)):
                return False, f"view {view} out of range for the agreed order of {len(order)}"
            if _norm(who) != _norm(order[view]):
                return False, f"out-of-turn: view-{view} proposer {who} is not the elected {order[view]}"
            if view > 0:                                       # a takeover must PROVE every prior view timed out
                rost = list(roster if roster is not None else order)
                nn = len({_norm(a) for a in rost})
                m = int(timeout_m) if timeout_m is not None else (nn // 2 + 1)
                r_id = int(envelope["round"])
                quorums = envelope.get("timeout_quorums") or {}
                for v in range(view):
                    atts = quorums.get(v, quorums.get(str(v)))
                    vr = verify_timeout_quorum(atts, rost, m, r_id, order[v], v)
                    if not vr["approved"]:
                        return False, (f"no view-change proof for view {v} (unilateral seizure: "
                                       f"{vr['n_signers']}/{vr['n_required']} timeout signers)")
        else:
            expected = elect_proposer(order, unavailable)
            if expected is None or _norm(who) != _norm(expected):
                return False, f"out-of-turn: proposer {who} is not the elected {expected}"
        rec = envelope.get("block") or {}
        try:
            hdr = BlockHeader.from_dict(rec["header"])
        except Exception as e:
            return False, f"malformed block header ({type(e).__name__})"
        if str(envelope["claim"]["block_hash"]) != hdr.hash():
            return False, "proposer claim does not bind this block"
        # (networked hardening) tie the envelope's round (the round the timeout-quorum payload above was
        # verified for) to the block's HEIGHT and the proposer-claim's round, so a networked peer cannot
        # attach a genuine round-r takeover quorum to a height!=r block. Honest producers settle at
        # height==round==claim.round (propose height=r), so this rejects only a spliced/mismatched envelope
        # -- and it runs BEFORE any state mutation below.
        if not (int(envelope.get("round", -2)) == int(hdr.height) == int(envelope["claim"].get("round", -3))):
            return False, "round mismatch (envelope round != header height != claim round)"
        if hdr.prev_hash != self.chain.head:
            return False, "block does not extend our head (gap/fork)"
        try:
            signer = recover_bytes(hdr.hash().encode(), rec["sig"])
        except Exception as e:
            return False, f"bad coordinator signature ({type(e).__name__})"
        if _norm(signer) != _norm(self.chain.coord.address):
            return False, f"block signed by {signer}, not the coordinator role key"
        try:
            self.chain.state = self.chain.state.apply_block(
                Block(header=hdr, txs=[], pouw={}), rec["contributors"])
        except Exception as e:
            return False, f"block does not apply ({type(e).__name__})"
        self.chain.blocks.append(dict(rec))
        self.chain.head = hdr.hash()
        return True, "ok"

    # ---- read helpers ----
    def block_hashes(self):
        """The ordered block-hash list of this replica -- two replicas equal here means NO FORK."""
        return [BlockHeader.from_dict(b["header"]).hash() for b in self.chain.blocks]

    def public_export(self):
        return self.chain.public_state()


# --------------------------------------------------------------------------- the drive
class RoundAuthority:
    """Drives N rounds across >=2 candidate coordinators using commit-reveal agreement + legitimate failover.
    The canonical settlement chain is REPLICATED on every candidate (each holds its own `ChainSettlement` with
    the shared coordinator role key); the commit-reveal beacon decides WHO extends it each round."""

    def __init__(self, candidates, validators, *, quorum_m, grace_ticks=3, timeout_m=None):
        if len(candidates) < 2:
            raise ValueError("RoundAuthority needs >= 2 candidate coordinators (B8-3 is two-node-agreed)")
        self.candidates = list(candidates)
        self.by_identity = {_norm(c.identity.address): c for c in self.candidates}
        self.validators = list(validators)
        self.roster = [v.address for v in self.validators]
        self.quorum_m = int(quorum_m)
        self.grace_ticks = int(grace_ticks)                 # bounded ticks a stalled proposer gets before failover
        # AGREED failure-detection threshold: an M-of-N of the CANDIDATE coordinators must SIGN a
        # primary-timeout before a view change is authorized. Default = STRICT MAJORITY of the candidate set,
        # so a minority (and in particular a lone secondary) can never seize the primary's slot (fork-safety).
        n_cand = len(self.candidates)
        self.timeout_m = int(timeout_m) if timeout_m is not None else (n_cand // 2 + 1)
        self.bus = LoopbackBus()
        self._miner_seq = 0
        self._pending = {}                                  # round -> {agreements, order} between begin/finish

    def _fresh_miner(self):
        self._miner_seq += 1
        return gen_account().address

    def _build_quorum(self, *, recipient, amount, height, delta_cid):
        """An M-of-N staked-validator quorum authorizing THIS exact mint (same shape as
        tests/test_authoritative_quorum). Independent of WHO proposes -> a failover takeover still needs it."""
        atts = [_ds.sign_settlement(self.validators[i], recipient=recipient, amount=amount,
                                    height=height, delta_cid=delta_cid) for i in range(self.quorum_m)]
        return _ds.settlement_block_proof(recipient=recipient, amount=amount, height=height,
                                          delta_cid=delta_cid, roster=self.roster,
                                          attestations=atts, m=self.quorum_m)

    def begin_round(self, round_id, *, extra_commit_envs=None, extra_reveals=None):
        """PHASES 1-3 of a round: COMMIT (each candidate seals a secret nonce, broadcasts its public commit
        envelope) -> REVEAL (each broadcasts its nonce) -> AGREE (EACH node INDEPENDENTLY runs agree_round over
        the shared bus set). Returns {round, order, agreements}. Split out from `finish_round` so a caller can
        learn the AGREED order BEFORE deciding who is down this round -- WITHOUT re-committing (a second commit
        for the same round from one signer would itself be an equivocation).

          extra_commit_envs -- extra public commit envelopes to place on the bus (used to inject an EQUIVOCATION:
                               two different digests from one signer at this round).
          extra_reveals     -- extra reveals to place on the bus alongside them.
        """
        r = int(round_id)
        for c in self.candidates:
            self.bus.publish_commit(r, c.make_round_commit(r))
        for env in (extra_commit_envs or []):               # optional equivocation / adversarial injection
            self.bus.publish_commit(r, env)
        for c in self.candidates:
            self.bus.publish_reveal(r, c.round_reveal(r))
        for rev in (extra_reveals or []):
            self.bus.publish_reveal(r, rev)
        agreements = {_norm(c.identity.address): c.agree(r, self.bus) for c in self.candidates}
        order = agreements[_norm(self.candidates[0].identity.address)]["order"]
        self._pending[r] = {"agreements": agreements, "order": order}
        return {"round": r, "order": order, "agreements": agreements}

    def finish_round(self, round_id, *, stalled=frozenset(), reward=10.0, agreed_failover=False):
        """PHASES 4-6 of a round (requires a prior `begin_round`): ELECT + bounded-tick FAILOVER down the AGREED
        order -> the elected proposer PRODUCES the block (with the B8-1 quorum) on its replica and broadcasts ->
        every OTHER candidate VERIFIES the election + applies. Returns the full per-round record.

          stalled          -- identities that will NOT produce this round (a genuine-stall injection; a
                              simulated crash). In BOTH modes this is only the INJECTION of a real stall (the
                              proposer does not produce); it is the failover TRIGGER only in the legacy mode.
          agreed_failover  -- when False (DEFAULT, byte-identical to the shipped proof): `stalled` directly
                              drives failover to the next AGREED signer (the trigger is the caller-supplied
                              down-set -- NOT fork-safe for a real two-node failover, which is the gap this
                              build closes). When True: the takeover TRIGGER is an M-of-N primary-timeout
                              QUORUM the candidates SIGN (`_finish_viewchange`) -- a secondary cannot forge it,
                              so a producing primary cannot be unseated.
        """
        r = int(round_id)
        if r not in self._pending:
            raise RuntimeError(f"finish_round({r}) without begin_round({r})")
        pend = self._pending.pop(r)
        order, agreements = pend["order"], pend["agreements"]
        stalled = {_norm(s) for s in (stalled or ())}
        if agreed_failover:
            return self._finish_viewchange(r, order, agreements, stalled, reward)

        # PHASE 4 -- ELECT + bounded-tick FAILOVER down the AGREED order (LEGACY: caller-supplied down-set).
        elected, ticks = None, 0
        for ident in order:
            if _norm(ident) in stalled:
                ticks += self.grace_ticks                   # waited the full budget; the primary never produced
                continue                                    # -> legitimate failover to the next AGREED signer
            elected = ident
            ticks += 1                                       # produced on its first tick
            break
        if elected is None:
            raise RuntimeError(f"round {r}: no available proposer in the agreed order {order}")

        # PHASE 5 -- PRODUCE: the elected proposer settles the block (with quorum) on its replica + broadcasts.
        proposer = self.by_identity[_norm(elected)]
        recipient = self._fresh_miner()
        delta_cid = f"ra-round-{r}"
        quorum = self._build_quorum(recipient=recipient, amount=reward, height=r, delta_cid=delta_cid)
        envelope = proposer.propose(r, height=r, reward=reward, recipient=recipient,
                                    quorum=quorum, timestamp=1000 + r)

        # PHASE 6 -- VERIFY + APPLY: every OTHER candidate independently validates the election + applies.
        for c in self.candidates:
            if c is proposer:
                continue
            ok, reason = c.accept(envelope, order, stalled)
            if not ok:
                raise RuntimeError(f"round {r}: peer {c.identity.address} rejected the elected proposer: {reason}")

        return {"round": r, "order": order, "agreements": agreements, "elected": elected,
                "ticks": ticks, "recipient": recipient, "envelope": envelope}

    def _finish_viewchange(self, r, order, agreements, stalled, reward):
        """AGREED-failover drive (the B8-3 failure-DETECTION fix): walk views 0,1,2,... The current-view
        proposer is `order[view]`. If it OBSERVABLY does not produce (a genuine stall injected via `stalled`),
        every OTHER agreed candidate that is itself up SIGNS a primary-timeout attestation for
        (r, order[view], view); the takeover to view+1 is authorized ONLY when those attestations reach the
        M-of-N timeout quorum (`verify_timeout_quorum`). A MINORITY cannot advance the view -> the round HOLDS
        (a liveness stall), it never forks. The elected view-v proposer attaches the accumulated per-prior-view
        quorums; every peer RE-VERIFIES them in `accept`, so a takeover with no/insufficient quorum is rejected
        as a unilateral seizure. Roster for the quorum = the agreed candidate set (`order` itself)."""
        roster = list(order)                                  # timeout quorum roster = the agreed candidate set
        order_set = {_norm(o) for o in order}
        timeout_quorums, view, ticks, elected = {}, 0, 0, None
        while view < len(order):
            proposer_id = order[view]
            if _norm(proposer_id) in stalled:
                # OBSERVE the stall -> every up, on-roster peer SIGNS a primary-timeout attestation for this view.
                observers = [c for c in self.candidates
                             if _norm(c.identity.address) in order_set
                             and _norm(c.identity.address) != _norm(proposer_id)
                             and _norm(c.identity.address) not in stalled]
                atts = [c.sign_timeout(r, proposer_id, view) for c in observers]
                verdict = verify_timeout_quorum(atts, roster, self.timeout_m, r, proposer_id, view)
                ticks += self.grace_ticks                     # waited the full budget; the proposer never produced
                if not verdict["approved"]:                   # MINORITY -> no view change; chain holds, no fork
                    raise RuntimeError(
                        f"round {r}: view-{view} proposer {proposer_id} stalled but the timeout quorum did NOT "
                        f"form ({verdict['n_signers']}/{verdict['n_required']} of {verdict['roster_n']} "
                        f"candidates); no view change authorized -- chain holds (liveness needs >= "
                        f"{self.timeout_m} up peers besides the stalled proposer)")
                timeout_quorums[view] = atts                  # the M-of-N proof that authorizes view+1
                view += 1
                continue
            elected = proposer_id                             # this view's proposer produced
            ticks += 1
            break
        if elected is None:
            raise RuntimeError(f"round {r}: every candidate in the agreed order {order} is stalled")

        proposer = self.by_identity[_norm(elected)]
        recipient = self._fresh_miner()
        delta_cid = f"ra-round-{r}"
        quorum = self._build_quorum(recipient=recipient, amount=reward, height=r, delta_cid=delta_cid)
        envelope = proposer.propose(r, height=r, reward=reward, recipient=recipient, quorum=quorum,
                                    timestamp=1000 + r, view=view, timeout_quorums=timeout_quorums)

        for c in self.candidates:                             # every OTHER candidate RE-VERIFIES the quorum chain
            if c is proposer:
                continue
            ok, reason = c.accept(envelope, order, timeout_m=self.timeout_m, roster=roster)
            if not ok:
                raise RuntimeError(f"round {r}: peer {c.identity.address} rejected the view-{view} proposer: {reason}")

        return {"round": r, "order": order, "agreements": agreements, "elected": elected, "view": view,
                "ticks": ticks, "recipient": recipient, "envelope": envelope,
                "timeout_quorums": timeout_quorums, "timeout_m": self.timeout_m, "roster": roster}

    def drive_round(self, round_id, *, stalled=frozenset(), reward=10.0, agreed_failover=False,
                    extra_commit_envs=None, extra_reveals=None):
        """Drive ONE round end to end (begin_round + finish_round). Returns the full per-round record: the
        per-node agreements, the agreed order, the elected proposer, the failover ticks, and the block envelope.
        `agreed_failover=True` routes through the M-of-N primary-timeout view-change trigger."""
        self.begin_round(round_id, extra_commit_envs=extra_commit_envs, extra_reveals=extra_reveals)
        return self.finish_round(round_id, stalled=stalled, reward=reward, agreed_failover=agreed_failover)

    def run(self, n_rounds, *, stall_plan=None, start_round=1, reward=10.0, agreed_failover=False):
        """Drive `n_rounds` consecutive rounds. `stall_plan` maps a round number -> a set of identities that are
        down that round (for the failover proof). Returns the list of per-round records."""
        stall_plan = stall_plan or {}
        return [self.drive_round(r, stalled=stall_plan.get(r, frozenset()), reward=reward,
                                 agreed_failover=agreed_failover)
                for r in range(start_round, start_round + int(n_rounds))]


# --------------------------------------------------------------------------- construction helper
def build_round_authority(*, n_candidates=2, n_validators=5, quorum_m=3, grace_ticks=3,
                          timeout_m=None, genesis_checkpoint="ra-genesis"):
    """Build a RoundAuthority with `n_candidates` candidate coordinators (DISTINCT identity keys), one SHARED
    coordinator role signing key, and an `n_validators`-strong staked-validator set for the B8-1 quorum. Each
    candidate holds its own replica of the settlement chain. `timeout_m` is the M-of-N primary-timeout
    view-change threshold over the candidate set (default = strict majority); partition-tolerant AGREED
    failover needs n_candidates >= 3 (a 2-node roster with a strict-majority M=2 cannot form a timeout quorum
    once one node is down -> fork-safe but not live)."""
    coord = gen_account()                                    # the shared coordinator role key (block signer)
    identities = [gen_account() for _ in range(n_candidates)]
    validators = [gen_account() for _ in range(n_validators)]
    candidates = [Candidate(idn, coord, genesis_checkpoint=genesis_checkpoint) for idn in identities]
    ra = RoundAuthority(candidates, validators, quorum_m=quorum_m, grace_ticks=grace_ticks, timeout_m=timeout_m)
    ra.coord = coord                                         # exposed for verify_public_chain pinning in tests
    return ra


# --------------------------------------------------------------------------- policy hook (default-off passthrough)
def advance_round(authority, single_node_r, *, env=None, **drive_kw):
    """The round-advance POLICY a live coordinator's loop would call in place of `r += 1` -- factored as a pure
    hook so the whole off/on behaviour is testable away from the live loop (mirrors
    diloco_clock.elected_round_authority).

      off (DEFAULT) -> return (single_node_r + 1, None). The RoundAuthority is NOT consulted and NO commit-reveal
                       runs -> byte-identical to today's single-node `r += 1`.
      on            -> drive the next round via the two-node commit-reveal agreement; return (r, drive_record).

    Unlike diloco_clock's enforce (which RAISES because it had no drive/transport), this hook can actually run
    the drive -- but only LOCALLY (loopback transport, shared coordinator role key). The production residuals
    (#45 determinism, a real network transport, an on-chain validator set) are unchanged; see the module header.
    """
    if not round_authority_enabled(env):
        return int(single_node_r) + 1, None                  # single-node clock governs, unchanged
    r = int(single_node_r) + 1
    return r, authority.drive_round(r, **drive_kw)
