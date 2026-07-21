"""
neurahash/diloco_committee.py -- D1: an M-of-N SIGNED-ATTESTATION committee for the DiLoCo merge
ACCEPT verdict (docs/DECENTRALIZE_D_SERIES.md D1).

Today the coordinator ALONE decides whether a contributed DiLoCo trunk delta is merged (and therefore
whether its author is rewarded): the sole held-out gate `neurahash/diloco_merge.py:apply_delta_gated`
runs once, in one process, and its `accepted` bool is trusted (sharded_pool_node.py:5353). That is one
party's say-so over money and model state. This module turns that ONE verdict into an M-of-N quorum: N
STAKED verifiers each INDEPENDENTLY re-run the SAME deterministic held-out gate on the SAME candidate
delta + the coordinator's post-round-average trunk snapshot, and each SIGNS its accept/reject + measured
held-out gain with its OWN secp256k1 key. The candidate is accepted iff >= M signature-valid attestations
from DISTINCT recovered signers IN THE STAKED ROSTER vote accept -- so no single node's "accepted" (not
even the coordinator's) can mint acceptance or a reward.

WHY it reuses, and what it deliberately does not:
  * The CRYPTO is reused unchanged -- `neura_l1.signing.sign_bytes` / `recover_bytes` (the same real
    secp256k1 ecrecover the block layer uses). We do NOT reimplement signing.
  * The QUORUM SHAPE mirrors the shipped trunk committee exactly (neurahash_torch/trunk_committee.py:
    `aggregate_trunk_attestations` + neurahash_torch/trunk_verify_net.py:`collect_committee`): count only
    signature-valid attestations, ONE per distinct RECOVERED signer, drop any whose recovered signer is
    not in the `allowed`/`selected` staked set (so a coordinator cannot pad the quorum with its own
    off-roster keys -- the B7-2c staked-signer binding), strict-majority M, slash the losing side.
  * The ATTESTATION SCHEMA is DiLoCo-specific and NEW, because the verdict is different: the trunk
    committee signs a `{cosine, norm, verdict}` recompute check; a DiLoCo verifier signs a
    `{delta_hash, base_round, accepted, gain}` HELD-OUT check. The signed bytes must honestly describe
    what was verified, so we cannot reuse the trunk payload. The SOLE apply decision itself is reused as
    the per-verifier check: `diloco_merge.apply_delta_gated(..., apply=False)`.

SEAM (unit-testable without sockets, transport-ready for the live wire -- the same pluggable-transport
pattern as B7-1's `recompute_fn` / B7-2's `collect_committee` verifier callables):
  * `verify_diloco_candidate(...)` accepts EITHER `verifier_accounts` (N accounts that run the held-out
    gate IN THIS PROCESS and sign -- proves the quorum math + staked binding with no network) OR
    `verifier_fns` (N zero-arg callables each returning a signed attestation or None -- real sockets in
    production, fixture peers in tests).
  * `verify_candidate_live(...)` is the LIVE coordinator adapter the default-off hook calls. It selects
    the staked verifier subset (reused `neura_l1.incentives.must_audit`) and gathers their signed
    held-out attestations. TODAY it returns a NO-QUORUM verdict: the DiLoCo-verify WIRE VERB + worker
    handler are the named D1 increment (docs/DECENTRALIZE_D_SERIES.md D1(b)/(f)) and are not yet
    deployed, so `_live_verifier_fns` yields zero callables. With the quorum-FALLBACK idiom (mirroring
    the shipped trunk-NET committee, sharded_pool_node.py:5161-5165), no quorum => the single-node gate
    governs => `enforce` is byte-identical to today until the wire lands. When it lands, ONLY
    `_live_verifier_fns` changes.

HONEST SCOPE. Like the trunk committee, this removes the single VERDICT (now M-of-N signed and
re-derivable by any node from public inputs -- the candidate delta is CID-fetchable, the eval corpus and
seed are deterministically self-built), NOT yet the single MACHINE (that is the wire increment) and not a
coordinator that also controls STAKED cohort membership (that is on-chain staking, B8 -- the same honest
residual `neurahash_torch/trunk_verify_net.py` documents). DETERMINISM caveat inherited from the held-out
gate: the accept/gain is reproducible across nodes only insofar as the held-out eval is (cross-vendor
bit-exactness is #45); every `enforce` here is gated on #45 exactly as the D-series doc states.

Pure numpy + stdlib + `neura_l1.signing` + the numpy-only `neurahash.diloco_merge` gate: NO new torch
dependency (apply_delta_gated is numpy-only), and NO import of sharded_pool_node (the live coordinator is
passed in, so there is no circular import -- the wire increment injects the transport).
"""
import hashlib
import json
import time

import numpy as np

from neura_l1.signing import sign_bytes, recover_bytes
from neurahash.diloco_merge import apply_delta_gated

ATTEST_TAG = "neurahash-diloco-committee"


# --------------------------------------------------------------------------- content binding
def delta_hash_np(delta_np):
    """Deterministic content hash of a candidate delta dict {key: ndarray}. Sorted keys + each array's
    shape + float32 bytes, so two nodes hashing the same delta get the SAME digest (it binds an
    attestation to the exact candidate). Not a security primitive on its own -- the signature is; this
    just names the candidate an attestation is about."""
    h = hashlib.sha256()
    for k in sorted(delta_np):
        a = np.ascontiguousarray(delta_np[k], dtype=np.float32)
        h.update(str(k).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def diloco_committee_beacon(base_round, round_beacon_hex, delta_hash_hex):
    """Post-commit beacon binding an attestation to (the base round, that round's trunk content via the
    round beacon, the COMMITTED candidate delta bytes). The verifier's verdict material is fixed only
    AFTER the candidate is committed, so a verifier cannot pre-grind a verdict, and an attestation cannot
    be replayed onto another round/candidate (its beacon differs). Mirrors
    trunk_committee.trunk_committee_beacon; in the live wire this is also the `must_audit` selection
    salt that chooses which staked nodes verify."""
    h = hashlib.sha256()
    h.update(int(base_round).to_bytes(8, "big", signed=True))
    h.update(str(round_beacon_hex).encode("utf-8"))
    h.update(str(delta_hash_hex).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- signed attestation
def diloco_attestation_payload(delta_hash, base_round, beacon_hex, accepted, gain):
    """Canonical bytes a verifier SIGNS (sorted-keys JSON, full binding; mirrors
    trunk_attestation_payload). `gain` is rounded to a FIXED precision so two same-eval verifiers produce
    BYTE-IDENTICAL payloads (determinism -- the verdict can later be a consensus artifact)."""
    return json.dumps({
        "w": ATTEST_TAG, "delta_hash": str(delta_hash), "base_round": int(base_round),
        "beacon": str(beacon_hex), "accepted": bool(accepted), "gain": round(float(gain), 6),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_diloco_attestation(account, delta_hash, base_round, beacon_hex, accepted, gain):
    """One verifier's signed attestation dict over the canonical payload, using the repo's real
    secp256k1 `sign_bytes`. `accepted`/`gain` are this verifier's OWN held-out verdict (the caller
    computed them via the SAME apply_delta_gated gate). Returns a JSON-safe dict."""
    payload = diloco_attestation_payload(delta_hash, base_round, beacon_hex, accepted, gain)
    return {"delta_hash": str(delta_hash), "base_round": int(base_round), "beacon": str(beacon_hex),
            "accepted": bool(accepted), "gain": round(float(gain), 6),
            "verifier_addr": account.address, "sig": sign_bytes(account, payload)}


def verify_diloco_attestation(att):
    """(ok, recovered_addr): re-recover the signer over the canonical payload rebuilt from the
    attestation's own fields. A tampered field (a flipped `accepted`, an edited `gain`) recovers a
    DIFFERENT address than the claimed `verifier_addr` -> (False, _), so it cannot count toward the
    quorum. `verifier_addr` is NEVER trusted, always re-derived. Mirrors verify_trunk_attestation."""
    if not isinstance(att, dict):
        return False, None
    try:
        payload = diloco_attestation_payload(att["delta_hash"], att["base_round"], att["beacon"],
                                             att["accepted"], att["gain"])
        rec = recover_bytes(payload, att["sig"])
    except Exception:
        return False, None
    claimed = att.get("verifier_addr")
    if not isinstance(claimed, str) or rec.lower() != claimed.lower():
        return False, rec
    return True, rec


# --------------------------------------------------------------------------- per-verifier held-out gate
def held_out_verdict(base_trunk_np, delta_np, eval_fn, *, outer=0.7, margin=0.0):
    """ONE verifier's INDEPENDENT held-out verdict on the candidate, computed by the SOLE apply decision
    itself -- `diloco_merge.apply_delta_gated` with apply=False so it NEVER mutates the shared base
    (safe to run N verifiers over the same `base_trunk_np`). Returns (accepted, gain) where
    gain = base_val - merged_val (held-out loss DROP; > 0 == the delta helps). A candidate that does not
    lower the held-out loss is rejected, exactly as the sole gate rejects it -- so on honest,
    deterministic input the committee is parity with the sole gate by construction."""
    v = apply_delta_gated(base_trunk_np, delta_np, eval_fn, outer=outer, margin=margin, apply=False)
    bv, mv = v.get("base_val"), v.get("merged_val")
    gain = (float(bv) - float(mv)) if (bv is not None and mv is not None) else 0.0
    return bool(v["accepted"]), gain


def build_diloco_attestation(account, *, delta_hash, base_round, beacon_hex, base_trunk_np, delta_np,
                             eval_fn, outer=0.7, margin=0.0, flip=False):
    """ONE verifier's signed attestation, computed from ITS OWN independent held-out gate -- the exact
    unit a REMOTE verifier node runs when the wire lands (fetch the candidate by CID, self-build the
    held-out eval, run apply_delta_gated, sign with ITS key; the coordinator never signs for a verifier).
    `flip` is TEST-ONLY: a lying/lazy verifier that reports the opposite verdict. Mirrors
    trunk_committee.build_trunk_attestation."""
    accepted, gain = held_out_verdict(base_trunk_np, delta_np, eval_fn, outer=outer, margin=margin)
    if flip:
        accepted = not accepted
    return sign_diloco_attestation(account, delta_hash, base_round, beacon_hex, accepted, gain)


# --------------------------------------------------------------------------- M-of-N aggregation
def _attestation_stake_weights(allowed):
    """Stake weights for the committee vote when the staked-signer set is supplied as a {addr: stake}
    mapping: returns a canonical {lower_addr: bonded_stake} (case-duplicates folded, negatives clamped to
    0). Returns None -- i.e. fall back to the byte-identical COUNT vote -- when `allowed` is absent, a bare
    address iterable, or a dict whose values are not plain numbers (so a {admitted_id: signer_addr} map is
    NEVER misread as stake). The keys are the SIGNER addresses the recovered vote is matched against."""
    if not isinstance(allowed, dict):
        return None
    out = {}
    for a, s in allowed.items():
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            return None                                       # not a stake mapping -> unweighted (count)
        k = str(a).lower()
        out[k] = out.get(k, 0.0) + max(0.0, float(s))
    return out


def aggregate_diloco_attestations(attestations, m, *, allowed=None, boundary=0.0, slash_tolerance=0.0):
    """Aggregate signed DiLoCo attestations into an M-of-N verdict -- the shared core of the in-process
    and networked committees, mirroring trunk_committee.aggregate_trunk_attestations. Counts only
    signature-valid attestations, ONE per distinct RECOVERED (authenticated) verifier (a duplicated /
    forged identity cannot double-vote -- `verifier_addr` is never trusted, always re-derived); a
    verifier whose vote disagrees with the M-of-N result is slashable (the quorum.py "slash the losing
    side" rule). Returns (accepted, valid, slashable, n_accept) where `valid` is the de-duplicated
    signature-valid subset.

    `allowed` (STAKED-SIGNER BINDING, B7-2c): when provided, the permitted signer set (case-insensitive) --
    an attestation whose RECOVERED signer is not in it is DROPPED, so a vote from outside the staked roster
    (e.g. a coordinator's own off-roster Sybil key) cannot count. It may be EITHER a bare iterable of
    addresses (the vote is a strict majority of COUNT -- byte-identical to before) OR a {addr: stake}
    mapping (the vote becomes a strict majority of bonded STAKE: the ACCEPTING signers' summed stake must
    exceed half the selected roster's total stake, so N shell keys each bonding MIN_STAKE cannot out-vote a
    few validators bonding far more -- the same Sybil-by-headcount fix D3 applies at settlement). A
    non-numeric-valued dict is treated as unweighted. `m` must be >= 1 (a zero threshold would accept an
    empty committee); with a stake mapping the accept threshold is FIXED at a strict stake majority
    regardless of m, so a minority-by-stake can never accept."""
    if m < 1:
        raise ValueError(f"committee threshold m={m} must be >= 1")
    weights = _attestation_stake_weights(allowed)             # {addr: stake} => stake-weighted; else count
    allow = None if allowed is None else {str(a).lower() for a in allowed}
    valid, seen, n_accept, accept_stake = [], set(), 0, 0.0
    for att in attestations:
        ok, rec = verify_diloco_attestation(att)
        if not ok or rec.lower() in seen or (allow is not None and rec.lower() not in allow):
            continue
        seen.add(rec.lower())
        valid.append(att)
        if att["accepted"]:
            n_accept += 1
            if weights is not None:
                accept_stake += weights.get(rec.lower(), 0.0)
    if weights is not None:                                   # strict majority OF STAKE (Sybil-resistant)
        total_stake = sum(weights.values())
        accepted = total_stake > 0.0 and accept_stake * 2.0 > total_stake
    else:
        accepted = n_accept >= m                             # strict majority of COUNT (byte-identical)
    # (#4 determinism, opus-review) a valid verifier whose vote disagrees with the M-of-N result is
    # slashable ONLY when its OWN signed gain is clearly (beyond slash_tolerance) on its side of the accept
    # boundary (gain == margin): a NEAR-boundary disagreement is fp / cross-arch (#45) noise, not malice, so
    # slashing it would punish an honest verifier. slash_tolerance <= 0 keeps the exact pre-#4 behavior.
    slashable = sorted({att["verifier_addr"] for att in valid
                        if bool(att["accepted"]) != accepted
                        and (slash_tolerance <= 0.0
                             or abs(float(att.get("gain", 0.0)) - float(boundary)) > float(slash_tolerance))})
    return accepted, valid, slashable, n_accept


def verify_diloco_candidate(*, delta_np, base_round, round_beacon_hex, base_trunk_np, eval_fn, m,
                            verifier_accounts=None, verifier_fns=None, selected=None,
                            outer=0.7, margin=0.0, dishonest=(), delta_hash=None, slash_tolerance=0.0):
    """Run an M-of-N signed-attestation committee on a DiLoCo merge candidate and return the quorum
    verdict. Exactly ONE verifier-injection mode must be given (the same pluggable seam as the trunk
    committee):

      verifier_accounts : N signing accounts; each runs the held-out gate IN THIS PROCESS over
                          (base_trunk_np, delta_np, eval_fn) and signs. Proves the quorum + binding math
                          with no sockets.
      verifier_fns      : N zero-arg callables, each returning that verifier's signed attestation dict
                          or None if unreachable (a raised exception is treated as None). Networked
                          sockets in production; fixture peers that REALLY run the gate + sign in tests.

    selected : the STAKED-SIGNER set the quorum is bound to (from the admission roster). When given, only
               attestations whose RECOVERED signer is in it count -- a coordinator cannot reach M with its
               own off-roster keys. A bare address set votes by COUNT (byte-identical); a {addr: stake} map
               votes by a strict majority of bonded STAKE (see aggregate_diloco_attestations). dishonest :
               TEST-ONLY set of verifier addresses whose verdict is flipped (accounts mode only).

    Returns {accepted, n_accept, n_total, quorum, quorum_met, selected_n, per_verifier, slashable,
    beacon, delta_hash}: `n_total` = valid distinct on-roster attestations counted, `quorum` = m,
    `quorum_met` = did >= m verifiers actually respond (so a caller can fall back to the single-node gate
    when the cohort was unreachable, never rejecting an honest candidate for a network gap)."""
    if (verifier_accounts is None) == (verifier_fns is None):
        raise ValueError("provide exactly one of verifier_accounts or verifier_fns")
    if selected is not None:                                 # bound quorum: forbid a degenerate m
        _n = len({str(s).lower() for s in selected})         # (mirrors trunk_verify_net.collect_committee)
        if not (m >= 1 and m * 2 > _n):
            raise ValueError(f"committee threshold m={m} must be >= 1 and a strict majority of the "
                             f"staked set N={_n} (m*2 > N)")
    if delta_hash is None:
        delta_hash = delta_hash_np(delta_np)
    beacon = diloco_committee_beacon(base_round, round_beacon_hex, delta_hash)
    dishonest = {str(a).lower() for a in dishonest}

    atts = []
    if verifier_accounts is not None:
        for acct in verifier_accounts:
            atts.append(build_diloco_attestation(
                acct, delta_hash=delta_hash, base_round=base_round, beacon_hex=beacon,
                base_trunk_np=base_trunk_np, delta_np=delta_np, eval_fn=eval_fn, outer=outer,
                margin=margin, flip=(acct.address.lower() in dishonest)))
    else:
        for fn in verifier_fns:
            try:
                a = fn()
            except Exception:
                a = None
            if isinstance(a, dict) and a.get("beacon") == beacon:   # bind to THIS request (anti-replay)
                atts.append(a)

    accepted, valid, slashable, n_accept = aggregate_diloco_attestations(
        atts, m, allowed=selected, boundary=margin, slash_tolerance=slash_tolerance)
    selected_n = len({str(s).lower() for s in selected}) if selected is not None else len(valid)
    per_verifier = [{"address": a["verifier_addr"], "accepted": bool(a["accepted"]), "gain": a["gain"],
                     "sig": a["sig"]} for a in valid]
    return {"accepted": bool(accepted), "n_accept": int(n_accept), "n_total": int(len(valid)),
            "quorum": int(m), "quorum_met": bool(len(valid) >= m), "selected_n": int(selected_n),
            "per_verifier": per_verifier, "slashable": list(slashable), "beacon": beacon,
            "delta_hash": delta_hash}


# --------------------------------------------------------------------------- final accept decision (hook)
def gated_accept(single_ok, committee_v, mode, *, min_cohort=1, strict=False):
    """The FINAL accept decision the sharded_pool_node hook applies, given the single-node held-out gate
    (`single_ok`), the committee verdict, and the flag mode. This is the whole enforce/audit/off policy,
    factored out as a pure function so it is unit-testable away from the live round loop.

      off / audit -> `single_ok` UNCHANGED (audit is observe-only: it logs the committee-vs-single
                     agreement, it never flips the merge/reward decision -- an observer must not change
                     control flow, matching the trunk-committee audit posture). `strict` is INERT in these
                     observe-only modes: they never withhold.
      enforce     -> quorum-FALLBACK VETO (mirrors the shipped trunk-NET idiom, sharded_pool_node.py:
                     5161-5165): a RESPONDED quorum (quorum_met) that does NOT accept VETOES the merge
                     (=> not merged, not rewarded); if no quorum responded, or the quorum concurs, the
                     single-node gate governs. The committee can therefore REMOVE the single point of
                     trust that rubber-stamps a bad delta, but never rejects an honest candidate merely
                     because its staked verifiers were unreachable.

    strict (default False -- safety over liveness; affects ONLY the enforce no-quorum fallback): a genuinely
    retired coordinator must be ABLE to choose SAFETY over LIVENESS. With `strict=True` AND `mode ==
    "enforce"`, the two fallback branches that otherwise defer to the single node -- a DEGENERATE cohort
    (selected_n < min_cohort) and NO responded quorum (its staked verifiers partitioned / DoS'd) -- WITHHOLD
    acceptance (return False) instead of rubber-stamping the single-node decision under the banner of
    enforce. The responded-quorum VETO branch is unchanged (still False); a responded quorum that CONCURS
    still lets the single-node gate govern (strict never loosens, exactly as enforce never does). Thus
    `strict=False` is BYTE-IDENTICAL to today, and strict is inert in off/audit. Wiring strict to a call-site
    flag is a later activation step (deliberately out of scope for this mechanism increment).

    NOTE (deliberate, and why it is correct-by-construction here): enforce only ever TIGHTENS acceptance
    (True -> False on a quorum veto -- or, under strict, when no real quorum is available), never loosens it
    (False -> True). That is exactly what a fraud-gate needs, and it side-steps DiLoCo's IN-PLACE apply: the
    live gate has already merged into a local trunk copy iff single_ok; a veto sets accepted False so that
    copy is discarded (never written back), with no revert bookkeeping. The opposite direction (a quorum
    ACCEPTING a candidate the single node rejected) requires re-running the merge against the exact
    post-average snapshot and is deferred to the wire increment that owns that snapshot
    (docs/DECENTRALIZE_D_SERIES.md D1(f))."""
    if mode != "enforce":
        return bool(single_ok)                       # off / audit: observe-only, single-node decides (strict inert)
    # strict enforce WITHHOLDS (safety over liveness) when no real quorum is available; default enforce
    # FALLS BACK to the single-node gate (liveness over safety). The veto/concur branches are shared.
    fallback = False if strict else bool(single_ok)
    # (#5 min-cohort, opus-review) a DEGENERATE cohort is not authoritative: with too few staked verifiers
    # selected, m = majority(selected) lets a single always-reject node veto EVERY merge (a 1-node cohort =>
    # m=1). Require a minimum selected cohort before the committee may override the single-node gate; below
    # it, fall back (default) or WITHHOLD (strict) -- never veto on a cohort too small to be a real quorum.
    if int(committee_v.get("selected_n", min_cohort)) < int(min_cohort):   # absent => assume it meets the
        return fallback                                                    # floor (a real verdict always sets it)
    if committee_v.get("quorum_met") and not committee_v.get("accepted"):
        return False                                 # responded quorum of a real cohort vetoes (strict-independent)
    if not committee_v.get("quorum_met"):
        return fallback                              # NO quorum responded -> fall back (default) / WITHHOLD (strict)
    return bool(single_ok)                           # quorum responded and CONCURS -> single-node governs (never loosens)


def committee_verdict_record(round_id, rec, single_v, committee_v, mode):
    """A tamper-evident governance record (re-derivable, appended to the signed pool ledger): the
    single-node held-out verdict, the M-of-N committee result, whether they AGREE, and the per-verifier
    signed votes -- the calibration data needed before flipping enforce. Mirrors
    trunk_committee.committee_verdict_record. ASCII-only fields (the coordinator console is cp1252); the
    free-text contributor name is deliberately NOT echoed."""
    single_ok = bool(single_v.get("accepted")) if isinstance(single_v, dict) else bool(single_v)
    return {"kind": "diloco-committee", "round": int(round_id), "mode": str(mode),
            "delta_cid": (rec or {}).get("delta_cid"),
            "base_round": (rec or {}).get("base_round"),
            "single_ok": single_ok, "committee_accepted": bool(committee_v.get("accepted")),
            "agreement": (single_ok == bool(committee_v.get("accepted"))),
            "n_accept": int(committee_v.get("n_accept", 0)),
            "n_total": int(committee_v.get("n_total", 0)),
            "quorum": int(committee_v.get("quorum", 0)),
            "quorum_met": bool(committee_v.get("quorum_met", False)),
            "selected_n": int(committee_v.get("selected_n", 0)),
            "slashable": list(committee_v.get("slashable", [])),
            "delta_hash": committee_v.get("delta_hash"),
            "votes": [{"verifier": p["address"], "accepted": bool(p["accepted"]), "gain": p["gain"]}
                      for p in committee_v.get("per_verifier", [])]}


# --------------------------------------------------------------------------- live coordinator adapter
def _select_staked_verifiers(coord, signer_of, beacon, sample_prob, pushing=None):
    """The staked verifier subset selected to verify this candidate, deterministic + publicly
    recomputable via `neura_l1.incentives.must_audit` seeded by the POST-COMMIT committee beacon (a
    verifier cannot pre-grind selection). Only admission-roster members (`signer_of` keys) are eligible,
    exactly as the trunk networked committee restricts its pool (sharded_pool_node.py:3288). Returns a
    sorted list of connected, staked node ids. Empty when there is no roster / no live cohort.

    (ASYNC RE-PARTITION) `pushing` is an optional set/dict of coordinator addresses whose socket is
    currently owned by an OFF-round-loop trunk+experts push thread. Such a worker is EXCLUDED from the
    verifier/validator pool: its socket has a single writer (a verify/settle request here would be a
    SECOND writer racing the push daemon), and it does NOT yet hold this round's model (an attestation
    from it would vote over a model it never received, counting toward the M-of-N quorum). `pushing` is
    {}/None when the async-repush lane is off -> byte-identical (no exclusion), mirroring the trunk
    `committee_verifier_pool` and storage-audit filters."""
    if not signer_of:
        return []
    try:
        from neura_l1.incentives import must_audit
    except Exception:
        return []
    conns = getattr(coord, "conns", {}) or {}
    _pushing = pushing or ()
    pool = [v for v in signer_of                          # staked AND connected AND not off-loop pushing
            if str(v) in conns and str(v) not in _pushing]
    return sorted(v for v in pool if must_audit(beacon, str(v).lower(), sample_prob, salt="diloco-audit"))


def _live_verifier_fns(coord, selected, rec, delta_np, beacon, psk, timeout, *,
                       base_round=None, round_beacon_hex=None, base_trunk_np=None,
                       delta_hash=None, outer=0.7, margin=0.0):
    """Produce one zero-arg callable per selected verifier, each dialing that node for a signed DiLoCo
    held-out attestation via the coordinator-INJECTED transport `coord.diloco_verify_transport`.

    THIS is the seam the D1 wire increment fills (D1(b)/(f)). The live coordinator installs
    `diloco_verify_transport = fn(verifier_id, request, timeout) -> attestation_dict | None` -- a
    PSK-framed `diloco_verify_request`/`diloco_verify_resp` socket round-trip that MIRRORS the shipped
    trunk-NET committee (sharded_pool_node.py `_serve_net_verify` / `_networked_committee`): the worker
    receives/fetches the candidate, self-builds the held-out eval against the coordinator's post-average
    trunk snapshot, runs `apply_delta_gated`, and signs the verdict with ITS OWN staked key via
    `build_diloco_attestation` -- the coordinator never signs for a verifier.

    Kept PURE (no socket import; the transport is INJECTED on `coord`) so this module stays
    sharded_pool_node-free and unit-testable with an in-process transport (tests/test_diloco_verify_wire.py).
    When no transport is installed -- the default until the coordinator wires it, and on any node without a
    live staked roster -- this returns [] so the quorum is (correctly) not met and `enforce` falls back to
    the single-node gate (`gated_accept`), byte-identical to today. Each returned callable is best-effort:
    a transport error or a non-dict reply becomes None (an unreachable verifier), never an exception into
    the round loop -- `verify_diloco_candidate` additionally binds each reply to THIS request's beacon."""
    transport = getattr(coord, "diloco_verify_transport", None)
    if transport is None or not selected:
        return []
    request = {
        "kind": "diloco_verify_request",
        "delta_cid": (rec or {}).get("delta_cid"),
        "delta_hash": delta_hash if delta_hash is not None else delta_hash_np(delta_np),
        "base_round": (int(base_round) if base_round is not None else None),
        "round_beacon": (str(round_beacon_hex) if round_beacon_hex is not None else None),
        "beacon": beacon,
        "submitted_delta": delta_np,
        "base_trunk": base_trunk_np,
        "outer": float(outer), "margin": float(margin),
    }
    fns = []
    for vid in selected:
        def _one(_vid=vid):
            try:
                att = transport(_vid, request, timeout)
            except Exception:                             # unreachable / transport error -> no vote
                return None
            return att if isinstance(att, dict) else None
        fns.append(_one)
    return fns


def verify_candidate_live(coord, rec, delta_np, *, base_round, round_beacon_hex, base_trunk_np, eval_fn,
                          single_ok, signer_of=None, psk=None, sample_prob=1.0, outer=0.7, margin=0.0,
                          timeout=5.0, slash_tolerance=0.0, pushing=None):
    """LIVE coordinator adapter the default-off sharded_pool_node hook calls: select the staked verifier
    subset, gather their signed held-out attestations, and return the M-of-N quorum verdict (bound to the
    staked roster). Best-effort: never raises into the round loop is the CALLER's contract, but this is
    written to be exception-light.

    TODAY returns a NO-QUORUM verdict (see `_live_verifier_fns`: the wire verb/handler are the D1
    increment), so with the quorum-fallback policy in `gated_accept`, enforce is byte-identical to today
    until the wire lands.

    HARDEST SUB-PROBLEM (docs D1(f)): a verifier must eval against the coordinator's EXACT
    post-this-round-average trunk snapshot. The caller MUST pass `base_trunk_np` captured BEFORE the live
    apply_delta_gated mutates it (sharded_pool_node.py:5344, before :5353). It is unused while there are
    no verifiers, but the parameter fixes the contract so the wire increment inherits no latent bug."""
    delta_h = delta_hash_np(delta_np)
    beacon = diloco_committee_beacon(base_round, round_beacon_hex, delta_h)
    selected = _select_staked_verifiers(coord, signer_of, beacon, sample_prob, pushing=pushing)
    fns = _live_verifier_fns(coord, selected, rec, delta_np, beacon, psk, timeout,
                             base_round=base_round, round_beacon_hex=round_beacon_hex,
                             base_trunk_np=base_trunk_np, delta_hash=delta_h, outer=outer, margin=margin)
    # (#2 liveness, opus-review) gather the verifier attestations CONCURRENTLY within one shared deadline,
    # so N slow / unreachable staked verifiers cost ~one timeout total (not N*timeout) and never serialize
    # inside the single-threaded coordinator round loop; then feed the collected votes to the shared
    # aggregator as trivial callables (mirrors the trunk-NET committee: fan-out + bounded gather + tally).
    atts = _gather_live_attestations(fns, timeout)
    # strict majority of the SELECTED staked cohort (scales with the live roster); >= 1 floor when the
    # roster/cohort is empty so aggregation's m>=1 invariant holds and an empty committee never accepts.
    m = (len(selected) // 2 + 1) if selected else 1
    allowed = [signer_of[s] for s in selected] if (signer_of and selected) else None
    v = verify_diloco_candidate(delta_np=delta_np, base_round=base_round,
                                round_beacon_hex=round_beacon_hex, base_trunk_np=base_trunk_np,
                                eval_fn=eval_fn, m=m, verifier_fns=[(lambda _a=_a: _a) for _a in atts],
                                selected=allowed, outer=outer, margin=margin, delta_hash=delta_h,
                                slash_tolerance=slash_tolerance)
    v["single_ok"] = bool(single_ok)
    v["selected_n"] = len(selected)
    return v


def _gather_live_attestations(fns, timeout):
    """(#2 liveness, opus-review) Invoke the per-verifier transport callables CONCURRENTLY and collect their
    signed attestations within ONE shared wall-clock deadline. Each callable is already bounded by its own
    socket send/recv timeout; running them on a small thread pool means N slow / unreachable staked verifiers
    cost ~one `timeout` total instead of N*timeout, so the single-threaded coordinator round loop is never
    serialized on the committee (the liveness the shipped trunk-NET committee gets from its fan-out +
    single-shared-deadline gather). Stragglers past the deadline are abandoned (shutdown wait=False) and
    simply do not vote -- the quorum tolerates a missing verifier. Pure stdlib; NO socket import here (the
    injected callables own the I/O), so the module stays sharded_pool_node-free and unit-testable."""
    if not fns:
        return []
    import concurrent.futures as _cf
    ex = _cf.ThreadPoolExecutor(max_workers=min(len(fns), 32))
    try:
        futs = [ex.submit(_fn) for _fn in fns]
        out, deadline = [], time.time() + max(0.1, float(timeout))
        for fut in futs:
            try:
                a = fut.result(timeout=max(0.0, deadline - time.time()))
            except Exception:                        # TimeoutError, or a callable that raised -> no vote
                a = None
            if isinstance(a, dict):
                out.append(a)
        return out
    finally:
        ex.shutdown(wait=False)                      # never block the round loop waiting on stragglers
