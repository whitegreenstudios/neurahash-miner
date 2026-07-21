"""
neura_l1/pool_sequencer.py — B12-1: shadow election + CPU-average audit, the first (observe-only)
increment toward retiring the single trusted pool coordinator (the XL B1 lift).

Today one trusted coordinator (sharded_pool_node.py) drives every pool training round. B12 replaces that
single clock with a rotating, ELECTED, staked proposer chosen by the fleet — REUSING the L1's existing
beacon-election + tie-break primitives (pouw_gate.beacon_for / leader_tag, the consensus.select_leader
form) rather than inventing new consensus.

THIS increment is pure OBSERVATION, mirroring B7-1's audit-first rollout and #39 TRUSTVERIFY: the live
coordinator keeps driving each round byte-for-byte unchanged, while these pure functions re-derive, per
round, (a) WHO the elected staked proposer WOULD be and (b) the DiLoCo trunk-average on CPU from the
already-collected, content-addressed submitted delta bytes — and the caller logs the result to the
tamper-evident pool_led.governance sink. Nothing here mutates state, slashes, mints, or sends traffic.

Two load-bearing design choices, both flagged by the B12 red-team:
  * The election seed is the PRIOR round's committed beacon (prev_hash semantics), NOT the current
    mutable trunk content — so a proposer cannot grind THIS round's trunk to change who leads THIS round.
  * The trunk-average is np.mean over the content-addressed submitted delta BYTES (vendor-deterministic),
    NEVER a GPU recompute — so it can later become a consensus equality. The fuzzy GPU recompute stays a
    per-worker admission gate only. Cross-vendor determinism is still UNPROVEN and gates all ENFORCEMENT;
    this module only collects the agreement/liveness distribution that calibration needs.

Pure: stdlib + numpy + pouw_gate (which is stdlib/hashlib). No torch, no I/O, no networking.
"""
import hashlib

import numpy as np

from neura_l1.pouw_gate import beacon_for, leader_tag

# Election seed for the very first round (no prior committed beacon yet). A fixed, public constant so the
# genesis election is itself publicly recomputable.
GENESIS_SEED = hashlib.sha256(b"neurahash-pool-sequencer-genesis").hexdigest()


def _addr_root(addr):
    """leader_tag's first arg must be HEX-decodable (it does bytes.fromhex on a str); a raw 0x-address is
    not valid hex. Bind a candidate by the sha256 hex digest of its address so the tag is well-defined for
    ANY address string and an address can't be crafted to break the election."""
    return hashlib.sha256(str(addr).encode()).hexdigest()


def election_beacon(prev_commit_hash, candidate):
    """Per-round, per-candidate election beacon = pouw_gate.beacon_for(prev_commit_hash, candidate).
    `prev_commit_hash` is the PRIOR round's committed beacon/commitment hash (prev_hash semantics): it is
    already fixed before this round runs, so the election cannot be ground by this round's mutable trunk."""
    return beacon_for(prev_commit_hash, str(candidate))


def elect_proposer(roster, prev_commit_hash, stake_of=None):
    """Deterministically elect the round proposer from `roster` (the staked-admission addresses), mirroring
    consensus.select_leader: rank by (-bounded_stake, ungrindable leader_tag) and take the minimum; with no
    stake function, rank by the tag alone. Returns (winner, ranked_list); (None, []) for an empty roster.
    Pure + publicly recomputable: any node with the same roster + prev_commit_hash reproduces the ordering."""
    roster = sorted({str(a) for a in roster})
    if not roster:
        return None, []
    if stake_of is not None:
        from neura_l1.consensus import saturating_weight    # lazy: keep the hot path stdlib-only

        def keyfn(a):
            tag = leader_tag(_addr_root(a), election_beacon(prev_commit_hash, a))
            return (-saturating_weight(float(stake_of(a))), tag)
    else:
        def keyfn(a):
            return leader_tag(_addr_root(a), election_beacon(prev_commit_hash, a))
    ranked = sorted(roster, key=keyfn)
    return ranked[0], ranked


def cpu_trunk_average(good_deltas, trunk_keys):
    """The DiLoCo outer-loop average the live coordinator applies (sharded_pool_node.py:1269-1273),
    recomputed on CPU as np.mean over the ALREADY-COLLECTED submitted trunk-delta bytes — NOT a GPU
    recompute. np.mean over identical content-addressed bytes is vendor-deterministic, so this is the
    quantity that can become a consensus equality. `good_deltas` maps addr -> a submission dict carrying
    ["trunk_delta"][k]; returns {k: averaged ndarray} over `trunk_keys`. Does not mutate its inputs."""
    keys = list(trunk_keys)
    addrs = list(good_deltas)
    return {k: np.mean([np.asarray(good_deltas[a]["trunk_delta"][k]) for a in addrs], axis=0) for k in keys}


def average_is_reproducible(good_deltas, trunk_keys):
    """True iff cpu_trunk_average is bit-for-bit identical across two independent CPU computations over the
    same submitted bytes — the determinism heartbeat this increment logs. A regression sentinel: if the
    average ever drifts off pure numpy into a vendor-dependent path, this flips to False. (It is NOT a
    cross-vendor proof — that needs the B12-2/3 multi-machine soak; it asserts the average is derivable
    from content-addressed bytes alone, the property a consensus equality will rest on.)"""
    if not good_deltas:
        return False
    a = cpu_trunk_average(good_deltas, trunk_keys)
    b = cpu_trunk_average(good_deltas, trunk_keys)
    return all(k in b and np.array_equal(a[k], b[k]) for k in a)


def takeover_decision(ranked, last_seen, current_round, *, stale_after=2):
    """(B12-2) The deterministic, publicly-recomputable FAILOVER decision over the elected proposer ranking.
    Given `ranked` (the elect_proposer ordering) and `last_seen` ({proposer: the round it was last live /
    responsive}), a proposer is FRESH iff `current_round - last_seen[proposer] <= stale_after` (never-seen =
    stale). The successor is the highest-ranked FRESH proposer; `would_take_over` is True when the top-ranked
    leader is stale and the lease falls to a runner-up. Every watcher holding the same ranking + heartbeat
    log picks the SAME successor — this is p2p_node.note_failed_winner's skip-to-runner-up made publicly
    checkable, so liveness failover needs no trusted referee. Returns a dict (no side effects)."""
    ranked = list(ranked)
    if not ranked:
        return {"leader": None, "leader_last_seen": None, "leader_stale": True,
                "successor": None, "would_take_over": False}

    def fresh(p):
        ls = last_seen.get(p)
        return ls is not None and (int(current_round) - int(ls)) <= int(stale_after)

    leader = ranked[0]
    successor = next((p for p in ranked if fresh(p)), None)
    return {"leader": leader, "leader_last_seen": last_seen.get(leader),
            "leader_stale": not fresh(leader), "successor": successor,
            "would_take_over": (successor is not None and successor != leader)}


def proposer_liveness_record(round_id, mode, heartbeat_proposer, beacon, decision, n_tracked):
    """(B12-2) Governance record for the heartbeat + failover shadow. `heartbeat_proposer` is the node that
    actually drove this round (the single coordinator today) stamping its liveness BOUND to the round
    `beacon` (fresh per round, never a static id, so a heartbeat can't be replayed forward). `decision` is
    `takeover_decision` over the ELECTED ranking — in this single-process phase the standby only RECORDS
    'would_take_over'; no real failover happens (that is B12-5). The data measures whether the rotation
    would have liveness: how often the top-ranked elected leader is offline and the lease would fall to a
    runner-up."""
    decision = decision or {}
    return {"type": "proposer_liveness", "round": int(round_id), "mode": str(mode),
            "heartbeat": str(heartbeat_proposer), "beacon": str(beacon),
            "leader": (None if decision.get("leader") is None else str(decision["leader"])),
            "successor": (None if decision.get("successor") is None else str(decision["successor"])),
            "leader_stale": bool(decision.get("leader_stale")),
            "would_take_over": bool(decision.get("would_take_over")),
            "n_tracked": int(n_tracked)}


def delta_set_root(delta_hashes):
    """(B12-3) Deterministic root over the round's committed {address: trunk_delta_hash} set = sha256 of the
    sorted 'addr=hash' entries. Publicly recomputable by anyone holding the same accepted-delta set, so the
    commitment's merkle_root binds EXACTLY which contributions the round committed to (no silent add/drop /
    censorship of an accepted delta is detectable). Order-independent."""
    items = "|".join(f"{str(a)}={str(h)}" for a, h in sorted((delta_hashes or {}).items()))
    return hashlib.sha256(items.encode()).hexdigest()


def build_signed_round_commitment(account, *, height, prev_commit_id, beacon, parent_checkpoint,
                                  new_checkpoint, delta_hashes, work_score):
    """(B12-3) Build + sign the round's consensus Commitment (neura_l1.signing.Commitment): the proposer
    commits (height, prev=prior commit id, beacon, parent/new checkpoint, work_score, merkle_root over the
    accepted-delta set) and signs it with its OWN key. Returns (signed_envelope, commit_id). Reuses the
    chain's real secp256k1 sign path, so the SAME equivocation detection applies — two distinct signed
    commitments by one signer at one height form a verify_equivocation-checkable EquivocationProof (the
    slash is B12-3-enforce, calibration-gated). This makes each pool round a verifiable, chained, signed
    artifact: the thing B12-5 signs to DRIVE and B12-6 settles."""
    from neura_l1.signing import Commitment, sign_commitment
    c = Commitment(height=int(height), prev_hash=str(prev_commit_id), proposer=account.address,
                   beacon=str(beacon), parent_checkpoint=str(parent_checkpoint),
                   new_checkpoint=str(new_checkpoint), work_score=float(work_score),
                   merkle_root=delta_set_root(delta_hashes), tx_root="")
    return sign_commitment(c, account), c.commit_id()


def round_commitment_record(round_id, mode, signed, commit_id, prev_commit_id, n_deltas):
    """(B12-3) Governance record for the round-commitment shadow: the signed commitment envelope + its id
    and the prior id (the chain link), so any node can replay the commitment chain and re-verify every
    signature. Audit-only: signed + logged, NEVER slashed (the equivocation-slash is enforce, gated on the
    cross-machine determinism soak — an honest cross-arch new_checkpoint can differ)."""
    sc = (signed or {}).get("commitment", {})
    return {"type": "round_commitment", "round": int(round_id), "mode": str(mode),
            "commit_id": str(commit_id), "prev_commit_id": str(prev_commit_id),
            "proposer": str((signed or {}).get("signer", "")), "n_deltas": int(n_deltas),
            "new_checkpoint": str(sc.get("new_checkpoint", "")),
            "work_score": float(sc.get("work_score", 0.0) or 0.0), "signed": signed}


def verify_commitment_chain(records, *, expected_proposer=None, genesis_seed=GENESIS_SEED):
    """(B12-3b) STANDALONE TRUSTLESS AUDITOR. Replay the `round_commitment` governance records and verify the
    chain is self-consistent with NO trust in the coordinator: for each round (in round order)
      (1) the signature re-recovers to the stated proposer (and, if `expected_proposer` is given, to that
          registered key) — a tampered signed field recovers a different address and fails;
      (2) the recomputed canonical commit_id equals the logged commit_id (no swapped id);
      (3) prev_commit_id chains exactly to the prior round's commit_id (first links to `genesis_seed`) — a
          fork or a silent re-link is caught.
    A DUPLICATE round (two commitments at one height) is exactly an equivocation and HARD-fails. A round-
    number GAP is reported in `round_gaps` but does NOT fail `ok`: a gap is indistinguishable from the log
    alone between a maliciously-dropped round and a legitimately-skipped one (a round whose commitment hook
    was off / raised), so it is a soft signal to investigate, not a proof of fraud.
    Any node holding the signed, hash-chained governance log can run this and reach the IDENTICAL verdict.
    Returns (ok, report). HONEST LIMIT: this proves the commitment CHAIN integrity + authorship; fully
    re-deriving WHO was elected each round additionally needs the published per-round roster (the on-chain
    validator set, B12-4) — the log does not yet carry it, so election re-derivation is out of scope here."""
    from neura_l1.signing import verify_commitment, Commitment

    def _rnd(rec):                                         # defensive: a malformed round must not crash the audit
        try:
            return int(rec.get("round", 0))
        except (TypeError, ValueError):
            return 0

    recs = sorted((r for r in records if isinstance(r, dict) and r.get("type") == "round_commitment"),
                  key=_rnd)
    report = {"n": len(recs), "sig_failures": [], "id_mismatches": [], "chain_breaks": [],
              "duplicate_rounds": [], "round_gaps": []}
    prev_id, prev_round, seen = None, None, set()
    for rec in recs:
        rnd = _rnd(rec)
        if rnd in seen:                                   # two commitments at one height = equivocation
            report["duplicate_rounds"].append(rnd)
            continue                                      # never advances the chain
        seen.add(rnd)
        if prev_round is not None and rnd != prev_round + 1:
            report["round_gaps"].append(rnd)              # soft: maybe a dropped round, maybe a legit skip
        signed = rec.get("signed") or {}
        ok, _who = verify_commitment(signed, expected_proposer)
        if not ok:
            report["sig_failures"].append(rnd)
        try:
            if Commitment.from_dict(signed["commitment"]).commit_id() != rec.get("commit_id"):
                report["id_mismatches"].append(rnd)
        except Exception:
            report["id_mismatches"].append(rnd)
        expected_prev = prev_id if prev_id is not None else genesis_seed
        if rec.get("prev_commit_id") != expected_prev:
            report["chain_breaks"].append(rnd)
        prev_id, prev_round = rec.get("commit_id"), rnd
    report["ok"] = not (report["sig_failures"] or report["id_mismatches"] or report["chain_breaks"]
                        or report["duplicate_rounds"])     # gaps are a SOFT signal, not an ok-failure
    return report["ok"], report


def roster_root(roster_with_stake):
    """(B12-4) Deterministic root over the round's published staked validator set = sha256 of the sorted
    'addr=stake' entries. Publicly recomputable; binds EXACTLY which validators (+ weights) the round's
    election ran over, so a silently added / dropped / reweighted validator changes the root and roster
    CHURN between rounds is detectable. Order-independent."""
    items = "|".join(f"{str(a)}={float(s):.8g}" for a, s in sorted((roster_with_stake or {}).items()))
    return hashlib.sha256(items.encode()).hexdigest()


def validator_set_record(round_id, mode, roster_with_stake, election_seed):
    """(B12-4) Governance record PUBLISHING the round's staked validator set (the election roster + stake
    weights) and the election SEED used — so the proposer election becomes publicly RE-DERIVABLE end-to-end,
    closing the B12-3b residual (the commitment chain proved integrity + authorship, but WHO should have
    been elected needed the roster, which the log didn't carry). `election_seed` is the prior round's
    committed beacon (prev-beacon semantics: fixed before this round, so it can't be ground). Audit-first:
    published + logged, NEVER enforced. HONEST LIMIT: the roster is still admission-LOCAL coordinator state,
    so this makes SELECTION verifiable + roster churn auditable, but a malicious coordinator can still bias
    WHO it admits — the on-chain validator set (the deploy-gated end state) removes that residual."""
    roster_with_stake = {str(a): float(s) for a, s in (roster_with_stake or {}).items()}
    return {"type": "validator_set", "round": int(round_id), "mode": str(mode),
            "roster": roster_with_stake, "roster_root": roster_root(roster_with_stake),
            "election_seed": str(election_seed), "n": len(roster_with_stake)}


def verify_proposer_elections(records, *, genesis_seed=GENESIS_SEED):
    """(B12-4) STANDALONE ELECTION AUDITOR. Using the PUBLISHED per-round `validator_set` records, the
    `elected_proposer_audit` records, and the `round_commitment` chain, RE-DERIVE who should have been
    elected each round and verify the coordinator's claim — closing the B12-3b residual (election was
    unverifiable without the roster). For each round that published a roster:
      (1) re-run elect_proposer(published roster, published election_seed [, published stakes]) and require
          the winner == the `elected_proposer_audit.elected` the coordinator CLAIMED — a claimed proposer
          that is NOT the deterministic winner over the published roster is election fraud (HARD-fails ok);
      (2) the published election_seed must equal the PRIOR round's COMMITTED beacon (round_commitment.beacon;
          the first audited round -> genesis_seed) — so the seed is chained to committed content and cannot
          be ground (HARD-fails ok); when the prior beacon isn't in the log the check is skipped, not failed;
      (3) roster CHURN (roster_root changed vs the prior audited round) is reported as a SOFT signal.
    A round with an election claim but NO published roster is reported in `missing_roster` (unverifiable).
    Returns (ok, report); any node holding the published log reaches the identical verdict.
    HONEST LIMIT: this proves the election ran honestly OVER THE PUBLISHED ROSTER + an ungrindable seed; it
    cannot prove the roster itself wasn't biased at ADMISSION — that needs the on-chain validator set."""
    def _rnd(rec):
        try:
            return int(rec.get("round", 0))
        except (TypeError, ValueError):
            return 0

    vsets, claims, beacons = {}, {}, {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        t = rec.get("type")
        if t == "validator_set":
            vsets[_rnd(rec)] = rec
        elif t == "elected_proposer_audit":
            claims[_rnd(rec)] = rec
        elif t == "round_commitment":                       # the committed beacon the NEXT seed must equal
            beacons[_rnd(rec)] = ((rec.get("signed") or {}).get("commitment", {}) or {}).get("beacon")

    report = {"n": len(claims), "verified": [], "election_mismatches": [], "seed_chain_breaks": [],
              "roster_churn": [], "missing_roster": []}
    prev_root = None
    for rr in sorted(claims):
        vs = vsets.get(rr)
        if vs is None:
            report["missing_roster"].append(rr)
            continue
        roster = vs.get("roster") or {}
        staked = any(float(s) > 0.0 for s in roster.values())
        stake_of = ((lambda a, _r=roster: float(_r.get(a, 0.0))) if staked else None)
        winner, _ranked = elect_proposer(list(roster.keys()), vs.get("election_seed", ""), stake_of)
        claimed = claims[rr].get("elected")
        (report["verified"] if winner == claimed else report["election_mismatches"]).append(rr)
        # (2) seed must chain to the prior committed beacon; ONLY round 0 expects genesis. When the prior
        # beacon isn't in the (possibly partial / post-resume) log, the check is SKIPPED, not failed.
        expected_seed = genesis_seed if rr == 0 else beacons.get(rr - 1)
        if expected_seed is not None and vs.get("election_seed") != expected_seed:
            report["seed_chain_breaks"].append(rr)
        # (3) roster churn (soft)
        if prev_root is not None and vs.get("roster_root") != prev_root:
            report["roster_churn"].append(rr)
        prev_root = vs.get("roster_root")

    report["ok"] = not (report["election_mismatches"] or report["seed_chain_breaks"])   # churn/missing soft
    return report["ok"], report


def elected_proposer_record(r, elected, actual, ranked, *, elected_is_live, avg_reproducible, roster_size):
    """The tamper-evident governance record for one round's shadow election + CPU-average audit. In this
    single-coordinator phase the elected proposer is a SHADOW — it does NOT drive the round — so the record
    builds the rotation distribution + liveness evidence (does the deterministic election land on a LIVE
    staked node each round?) and the average-determinism heartbeat that B12-2/B12-3 need before any
    enforcement. `actual` is today's sole coordinator; `proposer_agrees` is expected False until B12-5 lets
    the elected proposer actually drive. `ranked` is bounded to 8 entries (like TRUNK_COMMITTEE_MAX_WORKERS)."""
    return {"type": "elected_proposer_audit", "round": int(r),
            "elected": (None if elected is None else str(elected)),
            "actual": str(actual), "proposer_agrees": (elected == actual),
            "elected_is_live": bool(elected_is_live), "avg_reproducible": bool(avg_reproducible),
            "roster_size": int(roster_size), "ranked": [str(a) for a in ranked[:8]]}
