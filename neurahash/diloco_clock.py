"""
neurahash/diloco_clock.py -- D4 (Stage A): a commit-reveal ROUND-CLOCK shadow, audit-first
(docs/DECENTRALIZE_D_SERIES.md D4).

Today the round number/order is one plain Python variable advanced by the coordinator alone --
`r += 1` at sharded_pool_node.py:5813 -- with NO external agreement about which round is being
decided or in what participant order. That is one process's say-so over the clock every fleet
member trains against. This module supplies the trustless primitive that REPLACES that say-so in
principle: a commit-reveal round beacon in which each participant first COMMITS a hiding digest
H(round || nonce || signer), later REVEALS its nonce, and a PURE function agrees a deterministic
round seed + participant order from the revealed set -- identical on every node, un-grindable
(the nonce is fixed before anyone sees another's, so no participant can bias the beacon toward an
order it prefers).

STAGE A (this build) is AUDIT-ONLY joint-reporting, exactly the D-series `audit` posture
(DECENTRALIZE_D_SERIES.md conventions, mirroring B12-1 NEURAHASH_ELECTED_PROPOSER=off|audit):
participants ALSO emit SIGNED commitments in shadow, and an auditor checks that the single-node
clock's claimed order matches the commit-reveal outcome -- LOG ONLY. It NEVER advances the round;
`r += 1` still governs. `off` (default) does not even build the shadow, so it is byte-identical to
today.

STAGE B (enforce -- taking `r` FROM the elected commit-reveal clock) is DESIGN-ONLY here and is
NOT implemented; see STAGE_B_DESIGN. It is blocked on (a) a real transport for a second
candidate-coordinator role that exists nowhere yet, and (b) #45 cross-arch determinism (else an
honest cross-arch proposer is indistinguishable from a fraud). `elected_round_authority` refuses
to fake it: for off/audit it returns the single node's `r` unchanged; for enforce it raises.

WHY it reuses, and what it deliberately does not:
  * The CRYPTO is reused unchanged -- `neura_l1.signing.sign_bytes` / `recover_bytes` (the same
    real secp256k1 ecrecover the block layer and D1's `neurahash/diloco_committee.py` use). We do
    NOT reimplement signing.
  * The SHADOW-AUDITOR SHAPE mirrors the shipped B12-3b standalone trustless auditor
    (`neura_l1/pool_sequencer.py:178` replay_and_verify): replay the signed governance records,
    re-recover every signature (a tampered field recovers a different address and fails), and treat
    two signed commitments by one signer at one round with DIFFERENT digests as an EQUIVOCATION
    (pool_sequencer.py:208, `neura_l1.signing.EquivocationProof` for blocks). An order MISMATCH
    (single-node vs commit-reveal) is the calibration signal joint-reporting exists to collect, so
    -- like a B12-3b round GAP -- it is REPORTED, not treated as an integrity failure.
  * The COMMIT SCHEMA is clock-specific and NEW (round||nonce||signer), because the block-layer
    `neura_l1.signing.Commitment` binds a full block header (height/prev_hash/beacon/checkpoints)
    -- the wrong shape for a bare round beacon. We reuse the signing primitive, not that payload,
    exactly as D1 signs a NEW DiLoCo attestation payload rather than a block Commitment.

HONEST SCOPE. This removes the single-node clock as a matter of PROTOCOL (any node re-derives the
same seed+order from the public reveal set), NOT yet as a matter of MACHINE (no second
candidate-coordinator transport exists -- Stage B) and NOT the coordinator's control of who is in
the staked participant set (on-chain staking, B8). DETERMINISM caveat: the seed/order are
byte-reproducible across nodes by construction (sha256 over sorted canonical bytes); the enforce
step that would let a REMOTE node drive the round inherits the #45 cross-arch gate exactly as the
D-series doc states.

Pure stdlib (`hashlib`, `json`, `os`, `secrets`) + `neura_l1.signing`. NO torch, NO import of
sharded_pool_node (this build does not touch it; the Stage-B hook is design-only).
"""
import hashlib
import json
import os
import secrets

from neura_l1.signing import sign_bytes, recover_bytes

DOMAIN = b"neurahash-diloco-clock/v1"       # domain separation: a clock commit can never be reinterpreted
CLOCK_TAG = "neurahash-diloco-clock"        # governance-record kind + signed-payload tag
NONCE_BYTES = 32                            # 256-bit hiding nonce
CLOCK_MODES = ("off", "audit")             # functional Stage A modes; "enforce" is Stage B (design-only)


# --------------------------------------------------------------------------- flag / helpers
def clock_mode(env=None):
    """Read NEURAHASH_DILOCO_CLOCK. Returns 'off' (DEFAULT), 'audit', or 'enforce' verbatim; any other/
    absent value fails SAFE to 'off' (byte-identical to today). Mirrors the B12-1 flag parse; the
    caller decides what enforce means (elected_round_authority refuses it -- Stage B)."""
    v = (os.environ if env is None else env).get("NEURAHASH_DILOCO_CLOCK", "off")
    v = str(v).strip().lower()
    return v if v in ("off", "audit", "enforce") else "off"


def _norm_signer(signer):
    """Canonical signer key: lowercased string, so 0x-checksummed and lowercase addresses agree (the
    digest, the seed set, and every dispute bucket use this)."""
    return str(signer).lower()


def new_nonce():
    """A fresh 256-bit hiding nonce (os.urandom-seeded) as hex. A participant keeps this SECRET until
    its reveal; publishing the digest first is what makes the beacon un-grindable."""
    return secrets.token_hex(NONCE_BYTES)


# --------------------------------------------------------------------------- commit-reveal core
def commit_digest(round_id, signer, nonce_hex):
    """The hiding commitment a participant broadcasts at commit time: H(DOMAIN | round | nonce | signer).
    Binding (nonce fixes the digest) + hiding (256-bit nonce) -- a committer cannot later reveal a
    DIFFERENT nonce that opens the same digest, and no one learns the nonce from the digest. `nonce_hex`
    is hex; a malformed nonce raises (callers guard)."""
    h = hashlib.sha256()
    h.update(DOMAIN)
    h.update(b"|")
    h.update(int(round_id).to_bytes(8, "big", signed=False))
    h.update(b"|")
    h.update(bytes.fromhex(str(nonce_hex)))
    h.update(b"|")
    h.update(_norm_signer(signer).encode("utf-8"))
    return h.hexdigest()


def make_commit(round_id, signer, nonce_hex=None):
    """A participant's SEALED commit for `round_id`: pick a secret nonce (or use the supplied one) and
    return {round, signer, digest, nonce}. The participant holds this; it publishes `commit_envelope`
    (digest, NO nonce) in the commit phase and `reveal_of` (nonce) in the reveal phase."""
    if nonce_hex is None:
        nonce_hex = new_nonce()
    nonce_hex = str(nonce_hex)
    return {"round": int(round_id), "signer": _norm_signer(signer),
            "digest": commit_digest(round_id, signer, nonce_hex), "nonce": nonce_hex}


def commit_envelope(sealed):
    """The PUBLIC commit broadcast in the commit phase -- {round, signer, digest}, nonce omitted (still
    hidden)."""
    return {"round": int(sealed["round"]), "signer": _norm_signer(sealed["signer"]),
            "digest": str(sealed["digest"])}


def reveal_of(sealed):
    """The reveal broadcast in the reveal phase -- {round, signer, nonce}."""
    return {"round": int(sealed["round"]), "signer": _norm_signer(sealed["signer"]),
            "nonce": str(sealed["nonce"])}


def verify_reveal(commit_env, reveal):
    """True iff `reveal` correctly opens `commit_env`: same round, same signer, and
    H(round||nonce||signer) recomputed from the revealed nonce equals the committed digest. A WRONG
    reveal (a nonce that does not hash to the committed digest, or a mismatched round/signer) -> False.
    Never raises (a malformed nonce/field returns False), so one bad reveal cannot crash agreement."""
    try:
        if int(commit_env["round"]) != int(reveal["round"]):
            return False
        if _norm_signer(commit_env["signer"]) != _norm_signer(reveal["signer"]):
            return False
        recomputed = commit_digest(reveal["round"], reveal["signer"], reveal["nonce"])
    except Exception:
        return False
    return recomputed == str(commit_env["digest"])


# --------------------------------------------------------------------------- deterministic seed + order
def agree_seed(reveals, round_id):
    """The round SEED derived from a set of (assumed-valid) reveals: sha256 over DOMAIN|round| the
    SORTED, DE-DUPLICATED {(signer, nonce)} set. Deterministic over the reveal SET -- independent of the
    order reveals arrive in and of duplicates -- so every node computes the byte-identical seed. Binding
    `round_id` prevents replaying one round's reveal set as another's beacon."""
    items = sorted({(_norm_signer(r["signer"]), str(r["nonce"])) for r in reveals})
    h = hashlib.sha256()
    h.update(DOMAIN)
    h.update(b"|seed|")
    h.update(int(round_id).to_bytes(8, "big", signed=False))
    for signer, nonce in items:
        h.update(b"|")
        h.update(signer.encode("utf-8"))
        h.update(b":")
        h.update(bytes.fromhex(nonce))
    return h.hexdigest()


def _order_key(seed_hex, signer):
    """A participant's deterministic sort key for the agreed order: H(seed | signer). Because `seed` is
    unpredictable until every nonce is revealed, no participant can pre-position itself in the order."""
    return hashlib.sha256((str(seed_hex) + "|" + _norm_signer(signer)).encode("utf-8")).hexdigest()


def agree_order(reveals, round_id, seed=None):
    """The agreed participant ORDER for the round: the distinct valid signers sorted by _order_key(seed,
    signer). Deterministic over the reveal set (same seed => same permutation on every node). `seed`
    defaults to agree_seed(reveals, round_id)."""
    if seed is None:
        seed = agree_seed(reveals, round_id)
    signers = sorted({_norm_signer(r["signer"]) for r in reveals})
    return sorted(signers, key=lambda s: _order_key(seed, s))


def agree_round(commit_envs, reveals, round_id):
    """PURE round-agreement over a commit set and its reveals -- the trustless replacement for the
    single-node `r += 1`. Matches each reveal to its commit by signer, verifies it, and derives a
    deterministic seed + participant order from the VALID revealed set only.

    DISPUTES ARE DOCUMENTED, never silently resolved (they are the audit's whole value):
      * rejected            -- a reveal whose nonce does not open its commit (a wrong reveal);
      * missing             -- a signer that committed but never validly revealed (withheld reveal);
      * unsolicited         -- a reveal with no matching commit (revealed without committing);
      * commit_equivocations-- a signer that committed TWO different digests for this round (its digest
                               is untrusted and it is excluded from the beacon -- the nothing-at-stake
                               fault, mirroring B12-3b two-at-one-height);
      * wrong_round         -- a commit/reveal whose round != round_id (routed out, not counted).

    Returns {round, seed, order, valid_signers, n_valid, disputes}. Identical valid reveal SET =>
    identical seed + order on every node."""
    round_id = int(round_id)
    disputes = {"rejected": [], "missing": [], "unsolicited": [],
                "commit_equivocations": [], "wrong_round": []}

    # ---- commit map (detect two-different-digests-at-one-round equivocation) ----
    commit_map, equiv = {}, set()
    for c in commit_envs:
        try:
            if int(c["round"]) != round_id:
                disputes["wrong_round"].append(_norm_signer(c.get("signer")))
                continue
            s, d = _norm_signer(c["signer"]), str(c["digest"])
        except Exception:
            continue
        if s in commit_map and commit_map[s] != d:
            equiv.add(s)                                    # committed two distinct digests -> untrusted
        else:
            commit_map.setdefault(s, d)
    for s in equiv:
        disputes["commit_equivocations"].append(s)
        commit_map.pop(s, None)                             # cannot trust which digest -> exclude entirely

    # ---- match + verify reveals against the (non-equivocating) commit map ----
    valid = {}                                              # signer -> revealed nonce
    for rv in reveals:
        try:
            if int(rv["round"]) != round_id:
                disputes["wrong_round"].append(_norm_signer(rv.get("signer")))
                continue
            s = _norm_signer(rv["signer"])
        except Exception:
            continue
        if s not in commit_map:
            disputes["unsolicited"].append(s)               # revealed without a matching commit
            continue
        if s in valid:
            continue                                        # first valid reveal per signer wins
        env = {"round": round_id, "signer": s, "digest": commit_map[s]}
        if verify_reveal(env, rv):
            valid[s] = str(rv["nonce"])
        else:
            disputes["rejected"].append(s)                  # wrong reveal
    disputes["missing"] = sorted(set(commit_map) - set(valid))

    # de-dup + sort every bucket for a deterministic, tamper-evident record
    for k in ("rejected", "unsolicited", "commit_equivocations", "wrong_round"):
        disputes[k] = sorted(set(disputes[k]))

    valid_reveals = [{"round": round_id, "signer": s, "nonce": n} for s, n in sorted(valid.items())]
    seed = agree_seed(valid_reveals, round_id)
    order = agree_order(valid_reveals, round_id, seed=seed)
    return {"round": round_id, "seed": seed, "order": order,
            "valid_signers": sorted(valid), "n_valid": len(valid), "disputes": disputes}


# --------------------------------------------------------------------------- signed shadow (audit mode)
def clock_commit_payload(round_id, signer, digest):
    """Canonical bytes a participant SIGNS in the audit shadow (sorted-keys JSON; mirrors
    diloco_committee.diloco_attestation_payload). Binds the signature to (round, signer, committed
    digest) so a signed commitment cannot be replayed onto another round or edited after signing."""
    return json.dumps({"w": CLOCK_TAG, "round": int(round_id), "signer": _norm_signer(signer),
                       "digest": str(digest)}, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_clock_commit(account, round_id, digest):
    """One participant's SIGNED clock commitment (the audit-mode shadow), over the canonical payload,
    using the repo's real secp256k1 `sign_bytes`. Returns {round, signer, digest, sig}; `signer` is the
    account address but is NEVER trusted -- verify_clock_commit RE-RECOVERS it."""
    payload = clock_commit_payload(round_id, account.address, digest)
    return {"round": int(round_id), "signer": account.address, "digest": str(digest),
            "sig": sign_bytes(account, payload)}


def make_shadow_commit(account, round_id, nonce_hex=None):
    """Convenience for an audit-mode participant: returns (sealed, signed) -- the sealed commit-reveal
    record it keeps AND the signed shadow commitment it emits. Both bind the SAME digest under the
    account's address."""
    sealed = make_commit(round_id, account.address, nonce_hex=nonce_hex)
    signed = sign_clock_commit(account, round_id, sealed["digest"])
    return sealed, signed


def verify_clock_commit(signed):
    """(ok, recovered_addr): re-recover the signer over the canonical payload rebuilt from the signed
    commitment's own fields. A tampered field (an edited round/digest) recovers a DIFFERENT address than
    the claimed `signer` -> (False, _). `signer` is never trusted, always re-derived. Mirrors
    diloco_committee.verify_diloco_attestation."""
    if not isinstance(signed, dict):
        return False, None
    try:
        payload = clock_commit_payload(signed["round"], signed["signer"], signed["digest"])
        rec = recover_bytes(payload, signed["sig"])
    except Exception:
        return False, None
    claimed = signed.get("signer")
    if not isinstance(claimed, str) or rec.lower() != claimed.lower():
        return False, rec
    return True, rec


def clock_equivocation(signed_a, signed_b):
    """Deterministically check a clock EQUIVOCATION (the nothing-at-stake fault): two SIGNED commitments
    that (i) both verify, (ii) recover to the SAME signer, (iii) are for the SAME round, and (iv) carry
    DIFFERENT digests -> the signer committed two distinct beacons at one round. Returns (is_equiv,
    signer/reason). Mirrors neura_l1.signing.verify_equivocation for blocks (a slash is Stage-B/enforce,
    calibration-gated -- here it is only reported)."""
    okA, recA = verify_clock_commit(signed_a)
    if not okA:
        return False, "envelope A invalid signature"
    okB, recB = verify_clock_commit(signed_b)
    if not okB:
        return False, "envelope B invalid signature"
    if recA.lower() != recB.lower():
        return False, f"different signers ({recA} vs {recB}) - not equivocation"
    if int(signed_a["round"]) != int(signed_b["round"]):
        return False, "different rounds - not equivocation"
    if str(signed_a["digest"]) == str(signed_b["digest"]):
        return False, "identical commitment (duplicate, not equivocation)"
    return True, recA


# --------------------------------------------------------------------------- shadow audit (Stage A, LOG-only)
def shadow_audit(round_id, claimed_order, commit_envs, reveals, *, mode="audit"):
    """STAGE A joint-reporting: derive the commit-reveal agreed order for `round_id` and compare it to
    the single node's CLAIMED order. Returns a tamper-evident, ASCII-safe governance record
    {kind, round, mode, match, clean, claimed_order, agreed_order, seed, n_valid, disputes}:
      * match  -- did the single-node claimed order EQUAL the commit-reveal agreed order? (False flags a
                  divergence -- the calibration signal enforce would need);
      * clean  -- were there zero disputes (no rejected/missing/unsolicited/equivocation/wrong-round)?
    LOG ONLY: this returns a record; it does NOT advance the round (`r += 1` still governs). An observer
    must never change control flow (the D-series audit invariant)."""
    agreed = agree_round(commit_envs, reveals, round_id)
    claimed = [_norm_signer(s) for s in (claimed_order or [])]
    disputes = agreed["disputes"]
    clean = not any(disputes[k] for k in disputes)
    return {"kind": CLOCK_TAG, "round": int(round_id), "mode": str(mode),
            "match": bool(claimed == agreed["order"]), "clean": bool(clean),
            "claimed_order": claimed, "agreed_order": agreed["order"],
            "seed": agreed["seed"], "n_valid": int(agreed["n_valid"]), "disputes": disputes}


def audit_clock_records(records, *, expected_signers=None):
    """STANDALONE TRUSTLESS AUDITOR (mirrors B12-3b `neura_l1/pool_sequencer.py:178`): replay a mixed log
    of signed clock commitments and shadow-audit records and verify it with NO trust in the coordinator.
    A record is a SIGNED COMMITMENT if it has a `sig` field, else a shadow-audit record if its `kind` is
    CLOCK_TAG (anything else is ignored). Checks:
      (1) every signed commitment's signature re-recovers to its stated signer (and, if
          `expected_signers` is given, into that staked set) -- a tampered field fails;
      (2) two signed commitments by one signer at one round with DIFFERENT digests are a clock
          EQUIVOCATION (hard-fail: `ok` False);
      (3) any shadow-audit record with match == False is a divergence -- REPORTED, not an `ok` failure
          (like a B12-3b round gap: joint-reporting exists to surface it, and slashing is Stage B).
    Returns (ok, report). `ok` = no sig_failures and no equivocations (INTEGRITY); `mismatches` is the
    soft calibration signal. Any node holding the log reaches the identical verdict."""
    allow = None if expected_signers is None else {_norm_signer(a) for a in expected_signers}
    report = {"n_signed": 0, "n_audits": 0, "sig_failures": [], "off_roster": [],
              "equivocations": [], "mismatches": []}
    seen = {}                                              # (signer, round) -> digest
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if "sig" in rec:                                   # a signed clock commitment
            report["n_signed"] += 1
            ok, who = verify_clock_commit(rec)
            if not ok:
                report["sig_failures"].append(rec.get("round"))
                continue
            key = (who.lower(), int(rec["round"]))
            if allow is not None and who.lower() not in allow:
                report["off_roster"].append(who)           # not a staked participant -> excluded
                continue
            if key in seen and seen[key] != str(rec["digest"]):
                report["equivocations"].append({"signer": who, "round": int(rec["round"])})
            else:
                seen.setdefault(key, str(rec["digest"]))
        elif rec.get("kind") == CLOCK_TAG:                 # a shadow-audit record
            report["n_audits"] += 1
            if rec.get("match") is False:
                report["mismatches"].append(int(rec.get("round", 0)))
    # de-dup for a deterministic report
    report["sig_failures"] = sorted(x for x in report["sig_failures"] if x is not None)
    report["off_roster"] = sorted(set(report["off_roster"]))
    report["mismatches"] = sorted(set(report["mismatches"]))
    report["ok"] = not (report["sig_failures"] or report["equivocations"])
    return report["ok"], report


# --------------------------------------------------------------------------- round-authority policy (flag)
def elected_round_authority(single_node_r, agreed, mode):
    """The round-advance policy the (design-only) Stage-B hook beside sharded_pool_node.py:5813 would
    apply -- factored out as a pure function so the whole off/audit/enforce policy is unit-testable away
    from the live loop, exactly as D1 factors `gated_accept`.

      off / audit -> return `single_node_r` UNCHANGED. Observe-only: the plain `r += 1` governs the
                     round; the commit-reveal outcome is only logged (shadow_audit). ZERO control-flow
                     change -> byte-identical to today.
      enforce     -> Stage B, NOT BUILT. Taking `r` from the elected commit-reveal clock is blocked on
                     #45 (an honest cross-arch proposer must be distinguishable from a fraud) and on a
                     real second candidate-coordinator transport that exists nowhere yet. Raises
                     NotImplementedError rather than silently faking a decentralized clock. See
                     STAGE_B_DESIGN."""
    if mode == "enforce":
        raise NotImplementedError(
            "NEURAHASH_DILOCO_CLOCK=enforce is Stage B (design-only): the elected commit-reveal clock "
            "DRIVING the round is blocked on #45 (cross-arch determinism) and a second "
            "candidate-coordinator transport that does not exist yet. Stage A ships off|audit only. "
            "See neurahash.diloco_clock.STAGE_B_DESIGN.")
    return int(single_node_r)                              # off / audit: single-node clock governs, unchanged


# --------------------------------------------------------------------------- LIVE seam (design-only hook)
def _remote_clock_envelopes(coord, round_id):
    """Gather the round's commit/reveal envelopes from the fleet's second candidate-coordinator(s).
    (B8-3a) Uses the transport INJECTED as `coord.clock_envelope_transport` -- a callable
    `fn(round_id) -> (commit_envs, reveals)` the coordinator installs when a second candidate-coordinator
    peer is wired over the gossip mesh (mirrors D1's injected `coord.diloco_verify_transport`). ABSENT =>
    ([], []) => no peer => `shadow_audit_live` reports a no-peer record and `r += 1` governs unchanged
    (byte-identical to a single-coordinator fleet). A transport hiccup never breaks the round (guarded);
    the returned envelopes are UNTRUSTED -- agree_round re-verifies every reveal against its commit."""
    fn = getattr(coord, "clock_envelope_transport", None)
    if fn is None:
        return [], []
    try:
        commit_envs, reveals = fn(int(round_id))
        # (B8-3a F2, opus) bound the returned sets so a hostile/buggy transport cannot flood agree_round or the
        # governance record. (F1, opus -- DONE in increment 2: the transport delivers SIGNED commit envelopes
        # and `shadow_audit_live` re-recovers each signer via verify_clock_commit + drops forged/off-roster
        # commits BEFORE agree_round; the live gossip transport MUST therefore emit signed commits.)
        cap = 4096
        return list(commit_envs or [])[:cap], list(reveals or [])[:cap]
    except Exception:
        return [], []                                       # a transport hiccup must never break the round


def shadow_audit_live(coord, round_id, claimed_order, *, local_commits=None, local_reveals=None,
                      mode="audit", allowed=None):
    """LIVE audit adapter the default-off `audit`-mode clock hook calls each round: gather the round's
    commit/reveal envelopes from a second candidate-coordinator (via the injected transport) + the local
    node, and shadow-audit the single node's claimed order. LOG ONLY (never touches `r`).

    (B8-3a increment 2 / F1, opus) The REMOTE commits arrive SIGNED ({round, signer, digest, sig}); each is
    AUTHENTICATED here via `verify_clock_commit` (re-recovers the signer over the canonical payload) and is
    kept ONLY if the signature is valid (recovered == claimed signer) and, when `allowed` is given, the
    recovered signer is in that staked candidate-coordinator roster. A forged / unsigned / off-roster remote
    commit is DROPPED before `agree_round`, so a hostile transport cannot inject fabricated signers into the
    agreed seed/order. The LOCAL commits are this node's own (trusted, passed as bare envelopes). With no
    authenticated remote commit (no transport, or none survive auth) the audit reports a NO-PEER record
    (match=None) and the plain `r += 1` governs unchanged."""
    remote_signed, remote_reveals = _remote_clock_envelopes(coord, round_id)
    claimed = [_norm_signer(s) for s in (claimed_order or [])]
    allow = None if allowed is None else {_norm_signer(a) for a in allowed}
    remote_commits = []
    for c in (remote_signed or []):
        ok, rec = verify_clock_commit(c)                    # re-recover the signer; drop bad/forged sigs
        if not ok:
            continue
        if allow is not None and _norm_signer(rec) not in allow:
            continue                                        # off the staked candidate-coordinator roster
        remote_commits.append(commit_envelope(c))
    if not remote_commits:
        return {"kind": CLOCK_TAG, "round": int(round_id), "mode": str(mode), "match": None,
                "reason": "no-peer: no authenticated second-coordinator commit for this round",
                "claimed_order": claimed, "agreed_order": [], "n_valid": 0}
    commits = list(local_commits or []) + remote_commits
    reveals = list(local_reveals or []) + list(remote_reveals or [])
    return shadow_audit(round_id, claimed_order, commits, reveals, mode=mode)


# --------------------------------------------------------------------------- Stage B design (documentation)
STAGE_B_DESIGN = """\
STAGE B -- enforce (DESIGN-ONLY; NOT built in this module).

GOAL: the round number/order is taken FROM the elected commit-reveal clock (this module's
agree_round), not from the coordinator's plain `r += 1` (sharded_pool_node.py:5813). An elected
node would DRIVE the round: broadcast the commit phase, collect reveals, publish the agreed seed +
order, and advance `r` only on a quorum of revealed commitments.

WHY IT IS NOT BUILDABLE NOW (docs/DECENTRALIZE_D_SERIES.md D4(f)):
  (1) TRANSPORT. The audit shadow needs, and enforce doubly needs, a SECOND candidate-coordinator
      role -- a node other than today's single coordinator that also runs the round loop and
      exchanges commit/reveal envelopes. No such transport exists (the live loop has exactly one
      coordinator). `_remote_clock_envelopes` is the exact seam that role would fill; until it does,
      shadow_audit_live returns NO-PEER and `r += 1` is untouched.
  (2) #45 CROSS-ARCH DETERMINISM. If a REMOTE node drives the round, its honest cross-vendor
      numerics (a 5090 vs a 4060, ~1 ULP apart -- memory gpu-testbed-cross-arch,
      safetensors-mmap-recompute-determinism) must be distinguishable from an actual fraud. Without
      #45, an honest cross-arch proposer is indistinguishable from a cheater, so enforce cannot slash
      safely. Every D-series enforce is #45-gated; this is no exception.

SECURITY INVARIANTS enforce must preserve (all hold in Stage A already):
  * QUORUM UNFORGEABLE BY ONE NODE. The seed mixes every participant's nonce; one node cannot bias
    it. Two signed commitments at one round with different digests are a clock EQUIVOCATION
    (clock_equivocation) -- the Stage-B slash, mirroring B12-3.
  * DEFAULT-OFF BYTE-IDENTICAL. off/audit never change `r` (elected_round_authority returns it
    unchanged); only enforce would, and only once (1)+(2) land.

DOC CAVEAT (from D4 recon): B12-5's tracker is cited as #82, but D4 recon reports GitHub #82 is
UNRELATED -- confirm the real tracker before scheduling Stage B.
"""
