"""
neurahash/diloco_settlement.py -- D3: M-of-N STAKED-VALIDATOR-SET settlement authorization
(docs/DECENTRALIZE_D_SERIES.md D3).

Today one key authorizes every mint. `neurahash/chain_settlement.py:ChainSettlement.settle` (:58) turns a
completed height into a reward block signed by ONE coordinator account (:71); anyone can replay the chain
and re-check that single signature (verify:127), but WHO may authorize a mint is one party's key. That is
the last money-authority single point of trust after the merge verdict was decentralized in D1
(neurahash/diloco_committee.py). This module turns that ONE settlement signature into an M-of-N quorum:
each reward DECISION (recipient, amount, height, delta_cid) is signed by DISTINCT members of the STAKED
validator roster over a canonical reward-block message, and the mint is authorized iff a STRICT MAJORITY
of that on-roster set signs the EXACT decision -- so no single key (not even the coordinator's) can mint.

WHY it reuses, and what it deliberately does not:
  * The k-of-m THRESHOLD primitive is reused unchanged -- `neurahash/guardian_halt.py:GuardianSet`
    (`distinct_signers`:58 recovers each signature via the repo's real secp256k1 `recover_bytes_scheme`
    and counts ONLY distinct roster members, so one key signing twice counts ONCE and an off-roster key
    counts ZERO; `meets_threshold`:72). We do NOT reimplement multisig or secp256k1.
  * The CRYPTO is reused unchanged -- `neura_l1.signing.sign_bytes` / `recover_bytes` (the same
    ecrecover the block layer and the D1 committee use).
  * The STAKED ROSTER derives from the shipped staking primitives -- `State.stake_of` (block_state.py:503)
    filtered by `MIN_STAKE` (block_state.py:49), exactly the economic Sybil deterrent `consensus.is_eligible`
    (:85) already enforces for proposer eligibility. `neura_l1.pool_sequencer.roster_root` (B12-4, :232)
    commits WHICH validators + weights the quorum ran over, so roster churn is auditable.
  * The SETTLEMENT LEDGER is reused unchanged -- `ChainSettlement.settle` mints via the SAME audited
    `State.apply_block` path with the SAME MAX_SUPPLY cap. This module is a NEW PRE-CHECK ahead of that
    call; it NEVER edits chain_settlement.py and NEVER touches the single-key signing/replay path. In
    `off`/`audit` the mint is byte-identical to today; only `enforce` gates it on the quorum.

HONEST SCOPE (docs D3(f)). This is the multisig MECHANISM: threshold verification bound to the staked
roster over a canonical reward-block message. A genuinely ON-CHAIN validator ROSTER (publicly recomputable
from on-chain stake, not an admission-LOCAL coordinator list) is DESIGN-ONLY and closes B12-4/#81 -- the
same honest residual `pool_sequencer.validator_set_record` documents. The roster this module is handed is
still coordinator-admitted state; binding it to on-chain stake so any node recomputes the identical set is
the hardest sub-problem and is NOT built here.

#45 GATE (docs D3, security invariant 5). `enforce` -- requiring the M-of-N quorum before a mint -- is
RESEARCH-gated on the cross-vendor determinism run (#45, memory enforcement-gate-45) and MUST stay OFF
until it lands: under fp drift an honest cross-arch validator's independently-derived reward decision can
differ by ~1 ULP, so its signature would recover over a slightly-different canonical message and be
dropped, starving an honest settlement of quorum. Until #45, run `audit` only (observe + log the
would-require verdict, ZERO ledger effect). The DEFAULT is `off` (byte-identical to today).

CONSERVATION (docs D3, security invariant 4). D3 changes WHO authorizes settlement, never the emission
math: when the quorum permits, the mint is the SAME `ChainSettlement.settle` call, so ledger conservation
and the MAX_SUPPLY cap hold identically to the single-key path (asserted in tests/test_diloco_settlement.py).

Pure stdlib + `neura_l1.signing` + `neurahash.guardian_halt.GuardianSet` + (optionally, injected)
`neurahash.chain_settlement.ChainSettlement`. No torch, no import of sharded_pool_node (the live
coordinator is injected, so there is no circular import -- the integration phase wires the hook).
"""
import hashlib
import json

from neura_l1.signing import sign_bytes
from neurahash.guardian_halt import GuardianSet

try:                                            # the staking floor (reused, not redefined); fall back to
    from neura_l1.block_state import MIN_STAKE  # the shipped value if the import surface ever moves.
except Exception:                               # pragma: no cover - defensive
    MIN_STAKE = 10.0

try:                                            # B12-4 roster commitment (audit-record only; optional).
    from neura_l1.pool_sequencer import roster_root as _roster_root
except Exception:                               # pragma: no cover - defensive
    _roster_root = None

SETTLE_TAG = "neurahash-diloco-settlement"
MODES = ("off", "audit", "enforce")


# --------------------------------------------------------------------------- canonical reward-block message
def settlement_payload(recipient, amount, height, delta_cid):
    """The canonical reward-block message each validator SIGNS (sorted-keys JSON, full binding). It names
    the WHOLE reward decision -- recipient, minted amount, settled height, and the contribution (delta_cid)
    it pays for -- so a signature authorizes THAT decision and nothing else: a validator that signed a
    different amount/recipient/height/cid recovers a DIFFERENT address over THIS message and cannot count
    (the anti-substitution binding). `amount` is rounded to the SAME 12 dp `ChainSettlement.settle` uses
    (chain_settlement.py:76) so two honest validators computing the same decision produce BYTE-IDENTICAL
    bytes -- the determinism the #45 gate is about. ASCII-only (the coordinator console is cp1252)."""
    return json.dumps({
        "w": SETTLE_TAG,
        "recipient": str(recipient),
        "amount": round(float(amount), 12),
        "height": int(height),
        "delta_cid": str(delta_cid or ""),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_digest(recipient, amount, height, delta_cid):
    """sha256 of the canonical reward-block message -- a short, ASCII, log-safe name for the exact decision
    the quorum signed (put in the audit record so any node can re-derive and match it)."""
    return hashlib.sha256(settlement_payload(recipient, amount, height, delta_cid)).hexdigest()


# --------------------------------------------------------------------------- one validator's signed vote
def sign_settlement(account, *, recipient, amount, height, delta_cid):
    """ONE staked validator's signed attestation over the canonical reward-block message, using the repo's
    real secp256k1 `sign_bytes`. Returns a JSON-safe dict; the claimed fields are convenience/audit only
    and are NEVER trusted -- aggregation re-recovers the signer from the signature over the canonical
    message rebuilt from the coordinator's decision (mirrors diloco_committee.sign_diloco_attestation)."""
    payload = settlement_payload(recipient, amount, height, delta_cid)
    return {"recipient": str(recipient), "amount": round(float(amount), 12), "height": int(height),
            "delta_cid": str(delta_cid or ""), "validator_addr": account.address,
            "sig": sign_bytes(account, payload)}


def _sig_of(att):
    """The raw signature hex out of an attestation dict (from sign_settlement) or a bare hex string."""
    if isinstance(att, dict):
        return att.get("sig")
    return att


# --------------------------------------------------------------------------- staked roster derivation
def staked_roster(stake_source, *, min_stake=MIN_STAKE, candidates=None):
    """The set of validator ADDRESSES eligible to authorize settlement = those with active bonded stake
    >= `min_stake`, the SAME economic gate `consensus.is_eligible` (:85) applies to proposers. Returns a
    sorted list of addresses (the roster the M-of-N quorum is bound to).

    `stake_source` may be EITHER a neura_l1 `State` (uses `State.stake_of`, block_state.py:503) OR a plain
    `{address: stake}` mapping -- the latter is the "shadow roster" the D-series doc allows until the
    on-chain validator set exists (D3(b)/(f)). `candidates` optionally restricts which addresses are
    considered (e.g. the registered validator keys `State.key_of` binds); default = every address that has
    posted stake. Deduplicated case-insensitively (the recovered-signer check is case-insensitive too)."""
    if hasattr(stake_source, "stake_of"):
        stake_of = stake_source.stake_of
        pool = candidates if candidates is not None else list(getattr(stake_source, "staked", {}) or {})
    else:
        m = {str(a): float(s) for a, s in dict(stake_source or {}).items()}
        stake_of = lambda a: m.get(str(a), 0.0)               # noqa: E731 - tiny local adapter
        pool = candidates if candidates is not None else list(m)
    seen, roster = set(), []
    for a in pool:
        if float(stake_of(a)) >= float(min_stake) and str(a).lower() not in seen:
            seen.add(str(a).lower())
            roster.append(str(a))
    return sorted(roster, key=str.lower)


def _roster_members_and_stakes(roster):
    """Normalize a roster given as a list of addresses OR a {addr: stake} mapping into (members_list,
    stake_map_for_roster_root)."""
    if isinstance(roster, dict):
        members = list(roster)
        stakes = {str(a): float(s) for a, s in roster.items()}
    else:
        members = list(roster or [])
        stakes = {str(a): 1.0 for a in members}               # unweighted commitment when only addrs known
    return members, stakes


def _stake_weight_map(stakes):
    """Canonical {lower_addr: bonded_stake} for the stake-weighted authorization test -- fold the roster's
    stake map onto lowercased keys (the case `GuardianSet.distinct_signers` returns), summing any
    case-duplicates and clamping negatives to 0 (bonded stake is non-negative). A signer's weight is looked
    up from and summed into the SAME map used for the total, so the strict-majority test is self-consistent."""
    out = {}
    for a, s in (stakes or {}).items():
        k = str(a).lower()
        try:
            w = max(0.0, float(s))
        except (TypeError, ValueError):
            w = 0.0
        out[k] = out.get(k, 0.0) + w
    return out


def _stake_majority(signer_stake, total_stake):
    """Strict-majority-OF-STAKE test: authorize IFF the signers' summed bonded stake is strictly more than
    half the roster's total bonded stake -- the stake analogue of the count guard `n_signers*2 > N`. A
    minority-by-stake (<= 50%) can NEVER pass, and the threshold is NOT configurable below a strict majority
    (the strongest form of the degenerate-threshold guard). total_stake <= 0 => nothing authorizes
    (fail-closed: no bonded stake, no authority)."""
    return float(total_stake) > 0.0 and float(signer_stake) * 2.0 > float(total_stake)


# --------------------------------------------------------------------------- M-of-N aggregation (the mechanism)
def collect_settlement_signatures(*, recipient, amount, height, delta_cid, roster, attestations, m=None):
    """Run the M-of-N staked-validator multisig over one reward decision and return the authorization
    verdict -- the core D3 mechanism. Reuses `GuardianSet` for the threshold count so exactly the shipped
    distinct-signer semantics apply: only signature-valid attestations whose RECOVERED signer is IN the
    staked roster count, one vote per distinct signer, everything else (off-roster key, forged/edited
    field, duplicate identity, signature over a DIFFERENT decision) is DROPPED.

    roster       : the staked roster the quorum is bound to -- a bare list of addresses (COUNT-weighted) OR
                   a {addr: stake} mapping (STAKE-weighted; see the authorization note below), typically
                   `staked_roster(...)`. An empty roster authorizes nothing.
    attestations : signed-settlement dicts (from `sign_settlement`) or bare signature hex strings.
    m            : required distinct on-roster signers for the COUNT path. Default = STRICT MAJORITY
                   `N//2 + 1`. A provided m MUST be a strict majority of N (`m*2 > N`) and `1 <= m <= N` --
                   a plurality/degenerate threshold that would let a minority (or one key) mint is REJECTED
                   with ValueError, so a coordinator cannot pick m=1 over a bound roster to make a single
                   vote decisive (mirrors diloco_committee's strict-majority guard).

    AUTHORIZATION. With a bare address roster the mint is authorized iff `n_signers >= n_required` (strict
    majority of COUNT -- today's live path, byte-identical). With a {addr: stake} roster it is authorized iff
    the signers' summed bonded stake is a strict majority of the roster's TOTAL bonded stake (_stake_majority):
    N shell keys each bonding MIN_STAKE cannot out-authorize a few validators bonding far more -- the
    Sybil-by-headcount hole the coordinator-retirement quorum otherwise carries. The stake threshold is fixed
    at a strict majority (never sub-majority), the stake analogue of the m*2>N count guard.

    Returns {approved, n_signers, n_required, roster_n, signers, n_submitted, n_dropped, roster_root,
    recipient, amount, height, delta_cid, payload_sha256} (+ weighted, total_stake, signer_stake when the
    roster carries stake). `approved` is `n_signers >= n_required` (count roster) or a strict majority of
    bonded STAKE (stake roster)."""
    members, stakes = _roster_members_and_stakes(roster)
    N = len({str(a).lower() for a in members})
    if m is None:
        m = (N // 2) + 1                                       # strict-majority default
    m = int(m)
    if N == 0:
        raise ValueError("staked roster is empty; no settlement authorization possible")
    if not (1 <= m <= N and m * 2 > N):
        raise ValueError(f"settlement threshold m={m} must be a strict majority of the staked roster "
                         f"N={N} (1 <= m <= N and m*2 > N)")

    payload = settlement_payload(recipient, amount, height, delta_cid)
    sigs = [s for s in (_sig_of(a) for a in (attestations or [])) if s is not None]
    gs = GuardianSet(members, m)                               # k-of-m over the staked roster (reused)
    signers = gs.distinct_signers(payload, sigs)               # distinct roster members over THIS decision
    n_signers = len(signers)

    # AUTHORIZATION THRESHOLD. When the roster is a {addr: stake} mapping carrying REAL bonded stake, the
    # decision is authorized on a strict majority OF STAKE (_stake_majority): the summed bonded stake of the
    # distinct on-roster signers must exceed half the roster's TOTAL bonded stake -- so N shell keys each
    # bonding MIN_STAKE cannot out-authorize a few validators bonding orders of magnitude more (the
    # Sybil-by-headcount hole). The stake threshold is FIXED at a strict majority (not configurable below it),
    # the stake analogue of the m*2>N count guard above -- a minority-by-stake can never authorize. When the
    # roster is a bare address list (today's live path -- staked_roster returns a list), fall back to the
    # BYTE-IDENTICAL strict-majority-of-COUNT (n_signers >= m): with no per-address weights the two are the
    # same test, so the live count-based path is unchanged.
    weighted = isinstance(roster, dict)
    if weighted:
        wmap = _stake_weight_map(stakes)
        total_stake = sum(wmap.values())
        signer_stake = sum(wmap.get(s, 0.0) for s in signers)
        approved = _stake_majority(signer_stake, total_stake)
    else:
        approved = n_signers >= m

    rroot = _roster_root(stakes) if _roster_root is not None else None
    result = {
        "approved": bool(approved),
        "n_signers": int(n_signers),
        "n_required": int(m),
        "roster_n": int(N),
        "signers": sorted(signers),
        "n_submitted": int(len(attestations or [])),
        "n_dropped": int(len(attestations or []) - n_signers),
        "roster_root": rroot,
        "recipient": str(recipient),
        "amount": round(float(amount), 12),
        "height": int(height),
        "delta_cid": str(delta_cid or ""),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if weighted:                                              # stake-weighted verdict exposes the tally used
        result["weighted"] = True
        result["total_stake"] = round(float(total_stake), 12)
        result["signer_stake"] = round(float(signer_stake), 12)
    return result


# --------------------------------------------------------------------------- flag policy (off/audit/enforce)
def settle_allowed(mode, verdict):
    """The whole off/audit/enforce policy, factored as a pure function so it is unit-testable away from the
    live settlement call site (mirrors diloco_committee.gated_accept). Returns whether the mint may proceed:

      off / audit -> True ALWAYS. Observe-only: the existing single-key `ChainSettlement.settle` runs
                     exactly as today; `audit` additionally LOGS the would-require verdict, but the multisig
                     has ZERO ledger effect (it never blocks or changes a mint).
      enforce     -> require the quorum: authorize the mint IFF `verdict['approved']` (a strict majority of
                     DISTINCT staked validators signed THIS exact decision). A decision the staked set did
                     not authorize does NOT settle -> no mint. #45-GATED: keep OFF until cross-machine
                     determinism lands (see the module header), else an honest cross-arch validator's ~1-ULP
                     decision difference drops its signature and starves an honest settlement of quorum."""
    if mode not in MODES:
        raise ValueError(f"unknown settlement mode {mode!r}; expected one of {MODES}")
    if mode != "enforce":
        return True                                           # off / audit: single-key path governs
    return bool(verdict.get("approved"))                      # enforce: require the M-of-N quorum


# --------------------------------------------------------------------------- audit governance record
def settlement_verdict_record(round_id, mode, verdict):
    """A tamper-evident, re-derivable governance record (appended to the signed pool ledger by the
    integration hook) capturing WHO authorized -- or would authorize -- a settlement: the mode, the bound
    roster size + commitment, the required vs actual distinct staked signers, and the exact decision digest.
    This is the calibration data an operator reviews under `audit` BEFORE ever flipping `enforce` (which is
    itself #45-gated). ASCII-only fields (the coordinator console is cp1252); mirrors
    diloco_committee.committee_verdict_record."""
    return {"kind": "diloco-settlement", "round": int(round_id), "mode": str(mode),
            "approved": bool(verdict.get("approved")),
            "n_signers": int(verdict.get("n_signers", 0)),
            "n_required": int(verdict.get("n_required", 0)),
            "roster_n": int(verdict.get("roster_n", 0)),
            "n_submitted": int(verdict.get("n_submitted", 0)),
            "n_dropped": int(verdict.get("n_dropped", 0)),
            "roster_root": verdict.get("roster_root"),
            "recipient": verdict.get("recipient"),
            "amount": verdict.get("amount"),
            "height": verdict.get("height"),
            "delta_cid": verdict.get("delta_cid"),
            "payload_sha256": verdict.get("payload_sha256"),
            "signers": list(verdict.get("signers", []))}


# --------------------------------------------------------------------------- gated settle (NEW pre-check wrapper)
def settle_with_multisig(chain, *, recipient, amount, height, delta_cid, roster, attestations, timestamp,
                         mode="off", m=None, ex_by=None):
    """Compose the D3 pre-check with the EXISTING single-key settlement: run the M-of-N staked-validator
    multisig, then invoke `chain.settle(...)` iff the flag permits. This is the NEW pre-check the D-series
    doc specifies (D3(c)); it reuses `neurahash.chain_settlement.ChainSettlement` UNMODIFIED (the injected
    `chain`), and the live integration phase wires these SAME two calls -- `collect_settlement_signatures`
    + `settle_allowed` -- around the existing settle call site in sharded_pool_node.py (not touched here).

    Returns (credited, verdict, record):
      off    -> pre-check computed but IGNORED; `chain.settle` runs UNCONDITIONALLY (byte-identical to today).
      audit  -> pre-check computed + `record` built for logging; `chain.settle` STILL runs unconditionally
                (observe-only -- the multisig has zero ledger effect).
      enforce-> `chain.settle` runs ONLY if the verdict is APPROVED (strict-majority distinct staked signers
                over THIS exact decision); otherwise NO settle and NO mint ({} credited). #45-GATED: keep OFF.

    The mint itself is the recipient credited `amount` at `height` via the single-key path (default
    `ex_by = {recipient: 1.0}` -- the per-delta DiLoCo reward; pass an explicit `ex_by` for a multi-
    contributor split). The multisig authorizes the DECISION; the emission math + MAX_SUPPLY cap are
    unchanged (`ChainSettlement.settle` -> `State.apply_block`)."""
    verdict = collect_settlement_signatures(recipient=recipient, amount=amount, height=height,
                                            delta_cid=delta_cid, roster=roster,
                                            attestations=attestations, m=m)
    record = settlement_verdict_record(height, mode, verdict)
    if not settle_allowed(mode, verdict):
        return {}, verdict, record                            # enforce + not approved: blocked, zero mint
    credited = chain.settle(int(height), float(amount), (ex_by or {recipient: 1.0}), int(timestamp))
    return credited, verdict, record


# --------------------------------------------------------------------------- B8-1: quorum proof embedded in the block
# D3 (above) gathers an M-of-N of signed attestations but only as a COORDINATOR-SIDE pre-check: the verdict goes
# to the audit log, while the on-chain block `ChainSettlement.settle` writes still carries ONE coordinator
# signature (D3's honest scope says it "NEVER edits chain_settlement.py"). B8-1 is the increment that DOES cross
# that line: it EMBEDS the raw quorum in the block the mint creates, so any third party replaying the chain can
# re-recover the distinct staked signers over THIS reward decision and re-check the M-of-N itself -- lifting the
# D3 "pre-check only" ceiling with NO second coordinator and NO external chain (docs/B8_COORDINATOR_RETIREMENT.md
# increment B8-1). It stays audit-first: the embedded proof is DURABLE + INDEPENDENTLY VERIFIABLE, but chain
# ACCEPTANCE still rests on the coordinator signature checked in `chain_settlement._replay_blocks` -- making the
# on-chain quorum load-bearing is a later B8 increment (and #45-gated, same as D3 enforce).
def settlement_block_proof(*, recipient, amount, height, delta_cid, roster, attestations, m=None):
    """Build the JSON-safe quorum proof to embed in ONE settlement block: {recipient, amount, height,
    delta_cid, sigs, roster, n_required, payload_sha256}. `sigs` are the RAW signature hexes of the gathered
    attestations (a replayer re-recovers the signer over the rebuilt payload -- the claimed fields are
    convenience/binding only and are NEVER trusted on verify). Raises via `collect_settlement_signatures` on a
    degenerate roster/threshold (a proof over an empty roster authorizes nothing) -- callers guard and embed
    None on failure. Note (F3 residual, same as D3): this records the quorum over the STATED roster; binding
    that roster to on-chain stake so any node recomputes the identical set is a later B8 increment (B8-2)."""
    sigs = [s for s in (_sig_of(a) for a in (attestations or [])) if s is not None]
    members, _stakes = _roster_members_and_stakes(roster)
    v = collect_settlement_signatures(recipient=recipient, amount=amount, height=height,
                                      delta_cid=delta_cid, roster=roster, attestations=sigs, m=m)
    return {"recipient": str(recipient), "amount": round(float(amount), 12), "height": int(height),
            "delta_cid": str(delta_cid or ""), "sigs": list(sigs), "roster": [str(a) for a in members],
            "n_required": int(v["n_required"]), "payload_sha256": v["payload_sha256"]}


def verify_settlement_block_proof(proof, *, recipient=None, amount=None, height=None):
    """INDEPENDENTLY re-verify a block's embedded quorum proof -- the read-only verifier path (no private key,
    no live coordinator). Rebuilds the canonical payload from the proof's OWN decision fields, re-recovers the
    DISTINCT on-roster signers over it, and re-checks the M-of-N, so a forged/edited proof (wrong roster,
    spliced signatures, altered amount, sub-majority threshold) fails here. When the block's minted identity
    (recipient = its sole contributor, amount = header reward, height = header height) is supplied, the proof is
    ALSO BOUND to it, so a genuine proof for mint A cannot be attached to block B. Returns
    (ok, reason, n_signers, n_required). NEVER raises: a malformed/degenerate proof returns (False, ...). Scope
    (same F3 residual as D3): this proves a strict majority of the STATED roster signed THIS exact decision; it
    does not prove the roster is the legitimate on-chain staked set (that binding is B8-2)."""
    try:
        if not isinstance(proof, dict):
            return False, "malformed proof (not a dict)", 0, 0
        if recipient is not None and str(recipient).lower() != str(proof.get("recipient", "")).lower():
            return False, "proof recipient != block", 0, 0
        if amount is not None and abs(float(amount) - float(proof.get("amount", -1.0))) > 1e-9:
            return False, "proof amount != block", 0, 0
        if height is not None and int(height) != int(proof.get("height", -1)):
            return False, "proof height != block", 0, 0
        v = collect_settlement_signatures(recipient=proof.get("recipient"), amount=proof.get("amount", 0.0),
                                          height=proof.get("height", 0), delta_cid=proof.get("delta_cid", ""),
                                          roster=proof.get("roster", []), attestations=proof.get("sigs", []),
                                          m=proof.get("n_required"))
        return (bool(v["approved"]), ("ok" if v["approved"] else "quorum-not-met"),
                int(v["n_signers"]), int(v["n_required"]))
    except Exception as e:                                    # degenerate roster/threshold, bad types, etc.
        return False, f"proof verify error: {type(e).__name__}", 0, 0


def audit_settlement_block_quorums(blocks):
    """(observe-only) For each settled block carrying an embedded 'quorum' proof, INDEPENDENTLY verify the
    M-of-N over the block's OWN minted identity (recipient = the sole contributor, amount = the header reward,
    height = the header height). Audit-shadow read: it NEVER rejects a block (chain acceptance still rests on
    the coordinator signature in chain_settlement._replay_blocks) -- it is the calibration data an operator
    reviews before B8 ever makes the on-chain quorum load-bearing. Returns a list of
    {height, ok, reason, n_signers, n_required} for the blocks that carry a proof."""
    out = []
    for rec in (blocks or []):
        q = rec.get("quorum") if isinstance(rec, dict) else None
        if not q:
            continue
        hdr = rec.get("header", {}) or {}
        contribs = rec.get("contributors", {}) or {}
        recipient = next(iter(contribs)) if len(contribs) == 1 else None
        ok, reason, ns, nr = verify_settlement_block_proof(
            q, recipient=recipient, amount=hdr.get("reward"), height=hdr.get("height"))
        out.append({"height": hdr.get("height"), "ok": bool(ok), "reason": reason,
                    "n_signers": int(ns), "n_required": int(nr)})
    return out


# --------------------------------------------------------------------------- B8-2: roster bound to on-chain stake
def roster_source_select(mode, admission_signers, stake_source):
    """(B8-2) Choose the effective committee signer roster by SOURCE, so the eligible-signer set can be bound to
    ON-CHAIN bonded stake instead of the coordinator-admitted admission map -- closing the F3 residual that D1,
    D3 and B8-1 all carry (the roster is publicly recomputable from stake, not one party's admission list).
    `admission_signers` = {admitted_addr: attestation_signer_addr} (or None); `stake_source` = a neura_l1 State
    (uses .staked/.stake_of) or a {addr: stake} map -- staked_roster's input. Eligibility binds on the ADMITTED
    address (the validator identity that bonds stake). Returns (effective_signers, report):
      off     -> admission_signers unchanged (byte-identical to today).
      audit   -> admission_signers unchanged (the quorum is NOT altered); the caller LOGS `report`.
      enforce -> the INTERSECTION: only admitted signers whose admitted address also holds >= MIN_STAKE on
                 chain. If on-chain stake is EMPTY (the default today -- ChainSettlement mints with txs=[], so
                 State.staked stays {}), the intersection is empty and the committee has no eligible signers ->
                 the quorum starves and fail-closes (D3 withholds, D1 falls back to single-node), the correct
                 end-state until a stake-population path exists (B8-4). Also gated on #45 and the sha256(quorum)
                 -in-header prerequisite before it can be load-bearing.
    `report` = {mode, admission_n, staked_n, intersection_n, dropped_for_no_stake}."""
    adm = dict(admission_signers or {})
    staked = {str(a).lower() for a in staked_roster(stake_source)} if stake_source is not None else set()
    inter = {a: s for a, s in adm.items() if str(a).lower() in staked}
    report = {"mode": str(mode), "admission_n": len(adm), "staked_n": len(staked),
              "intersection_n": len(inter), "dropped_for_no_stake": len(adm) - len(inter)}
    return (inter if mode == "enforce" else adm), report
