"""
neurahash/glm_round_authority.py -- STAGE 4 (PLAN_BITCOIN_STYLE_PAYOUTS_2026-11-19): a rotating block
producer for the GLM lane. DEFAULT-OFF (flag NEURAHASH_GLM_ROUND_AUTHORITY), build + local test only.

It promotes the B8-3 / B12 pieces (diloco_round_authority.py: signed proposer claim, M-of-N view-change
timeout quorum, deterministic election) into a lane-level round authority that does NOT depend on one
coordinator process:

  1. SCHEDULE. A deterministic, stake/roster-WEIGHTED proposer order per height:
     order(h) = weighted sampling without replacement, integer-only, seeded by
     sha256(tag | ns | genesis_seed | epoch_beacon(h) | h). No float math, so every node on every OS
     computes the identical order. The seed is fixed by the roster file / an external beacon, never drawn
     by a coordinator, and does not depend on block contents, so a proposer cannot grind the next slot.
     View v at height h is proposed by order(h)[v % n].
  2. BLOCKS. The view-v proposer signs a header {ns, height, view, prev, proposer, payload_sha,
     timeouts_sha, ts} (domain "glm-ra-header"); block_hash = sha256(canonical header). It publishes the
     header announcement first, then the full block, over relay names (transport-agnostic: any object with
     put(name, bytes) / get_name(name) -> bytes|None, which is the shape of glm_committee.RelayStore).
  3. VOTING, two phases per (height, view) -- the Tendermint / PBFT pattern. A node votes only in its
     current view. PREPARE (domain "glm-ra-prepare"): at most one block per view, subject to the LOCK RULE
     below. A strict weight majority of prepares for one block in one view is a PREPARED CERTIFICATE PC(v, B).
     ENDORSE (domain "glm-ra-endorse", the commit vote): on seeing PC(v, B) a node locks on (v, B) and
     endorses B in view v. A block with endorsements from a strict weight majority IN ONE VIEW is FINAL.
     LOCK RULE: a node locked on (lv, LB) at height h prepares block X in a later view only if X == LB, or the
     view's proposal carries a PC for X from a view >= lv (a newer certificate unlocks it). An unlocked node
     prepares any valid proposal. Votes may change across views under this rule; that is NOT equivocation.
     Two prepares or two endorsements by one signer in ONE view still are.
  4. SUCCESSION (pacemaker). When a node's current view v has not finalised by its local deadline it signs a
     timeout (domain "glm-ra-timeout") bound to (ns, height, view, prev, failed proposer), even if it already
     voted in v, attaching (unsigned, self-verifying) its lock certificate; after that it casts no more votes
     in v. A strict weight majority of view-v timeouts is a TIMEOUT CERTIFICATE TC(v); on seeing it a node
     enters view v+1 with a fresh timer. The view-v proposer needs TC(u) for EVERY u < v (a lone successor
     cannot forge that) and must RE-PROPOSE the block of the highest PC it knows (signed "reprop", domain
     "glm-ra-reprop", carrying the block, the PC and the TCs); only if it knows no PC may it propose a fresh
     block. Safety + liveness argument: docs/research/STAGE4_ROTATING_PROPOSER_2026-11-19.md.
  5. FORK CHOICE walks from genesis; at each height, among valid blocks whose prev is the chosen parent, it
     picks the max of (is final, best single-view endorsement weight, -view, -block_hash). An endorser that endorsed
     two blocks in ONE view counts for neither in that view. A node opens height h+1 only once a block at h is
     final, and re-publishes that decision (fin/<h>/<addr>) so a node that missed endorsements catches up.
     A node restarted with the same key must use state_path (last-sign state, written before signing leaves
     the node); without it a node is crash-STOP only.
  6. AUDITOR. Self-proving evidence usable for stage-3 slashing (verify_evidence): proposer equivocation (two
     signed headers, or two signed proposals, for one height+view), prepare equivocation and endorsement
     equivocation (two blocks, one height+view), and LOCK VIOLATION (an endorsement of B in view v plus a
     prepare of X != B in a later view whose own signed justification view is < v). WITHHELD blocks (a signed
     header whose block never reached the slot) are ATTESTED, not self-proving; missed slots (TC formed, no
     proposal) are liveness records only. A timeout after a vote in the same view is legal under rule 4.

Payouts: canonical_balances() replays ONLY the fork-choice chain. Payload format (interface for stage 0/2):
  {"mints": [{"recipient": addr, "amount": float, "delta_cid": str, "proposal_sha": str (glm_committee
  proposal digest), "votes": [glm_committee votes]}]}; a node may pass payload_check(payload) -> bool, e.g. a
  wrapper over glm_committee.decide, so a block paying an unapproved mint is invalid at every node.

Pure stdlib + neura_l1.signing. No torch, no network in this module (transport is injected).
ASCII-only output.
"""
import functools
import hashlib
import json
import os
import threading
import time
from collections import defaultdict

from neura_l1.signing import sign_bytes, recover_bytes

FLAG = "NEURAHASH_GLM_ROUND_AUTHORITY"
VERSION = 2                      # v2 (2026-11-19): two-phase votes per view + lock rule (liveness fix)
GENESIS_PREV = "0" * 64
SCHED_TAG = "glm-ra-schedule/v1"
DOM_HEADER = "glm-ra-header"
DOM_ENDORSE = "glm-ra-endorse"
DOM_TIMEOUT = "glm-ra-timeout"
DOM_PREPARE = "glm-ra-prepare"
DOM_REPROP = "glm-ra-reprop"
NAME_PREFIX = "glm-ra"
_TRUTHY = ("1", "on", "true", "yes", "enabled", "shadow")
_FORBIDDEN_ROSTER_KEYS = ("key", "priv", "private", "secret", "seed_phrase", "mnemonic")


def round_authority_on(environ=None):
    """True only for an explicit truthy NEURAHASH_GLM_ROUND_AUTHORITY; anything else fails safe to False."""
    v = (os.environ if environ is None else environ).get(FLAG, "")
    return str(v).strip().lower() in _TRUTHY


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def _n(a):
    return str(a).lower()


# --------------------------------------------------------------------------- roster
class Roster:
    """{address: integer weight}. Weight = stake (stage 3) or 1 per seat. Addresses are compared lower-case."""

    def __init__(self, weights, *, genesis_seed="", epoch_len=0, epoch_beacons=None):
        w = {}
        for a, x in dict(weights).items():
            if isinstance(x, bool) or int(x) != x:
                raise ValueError("roster weight must be a positive integer")
            x = int(x)
            if x <= 0:
                raise ValueError("roster weight must be a positive integer")
            if _n(a) in w:
                raise ValueError("duplicate roster address")
            w[_n(a)] = x
        if not w:
            raise ValueError("empty roster")
        self.w = w
        self.addrs = sorted(w)
        self.total = sum(w.values())
        self.genesis_seed = str(genesis_seed)
        self.epoch_len = int(epoch_len)
        self.epoch_beacons = {int(k): str(v) for k, v in (epoch_beacons or {}).items()}

    def weight(self, a):
        return self.w.get(_n(a), 0)

    def majority(self, signers):
        """Strict weight majority over DISTINCT on-roster signers."""
        s = sum(self.weight(a) for a in {_n(x) for x in signers} if a in self.w)
        return s * 2 > self.total

    def weight_of(self, signers):
        return sum(self.weight(a) for a in {_n(x) for x in signers} if a in self.w)

    def seed(self, height):
        ep = (int(height) // self.epoch_len) if self.epoch_len > 0 else 0
        beacon = self.epoch_beacons.get(ep, "")
        return sha(("%s|%s|%d|%s|%d" % (SCHED_TAG, self.genesis_seed, ep, beacon, int(height))).encode("ascii"))

    def order(self, ns, height):
        """Weighted sampling without replacement, integers only (cross-platform identical)."""
        seed = sha(("%s|%s" % (ns, self.seed(height))).encode("ascii"))
        remaining = list(self.addrs)
        out = []
        i = 0
        while remaining:
            W = sum(self.w[a] for a in remaining)
            r = int(sha(("%s|%d" % (seed, i)).encode("ascii")), 16) % W
            acc = 0
            for a in remaining:
                acc += self.w[a]
                if r < acc:
                    out.append(a)
                    remaining.remove(a)
                    break
            i += 1
        return out

    def proposer(self, ns, height, view):
        o = self.order(ns, height)
        return o[int(view) % len(o)]

    def to_json(self):
        return {"weights": dict(self.w), "genesis_seed": self.genesis_seed, "epoch_len": self.epoch_len,
                "epoch_beacons": {str(k): v for k, v in self.epoch_beacons.items()}}


def load_roster(path):
    """Roster JSON {"weights": {addr: int}, "genesis_seed": str, ...}. Refuses files carrying key material."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    low = json.dumps(d).lower()
    for k in _FORBIDDEN_ROSTER_KEYS:
        if '"%s' % k in low:
            raise ValueError("roster file must hold public addresses only (found %r)" % k)
    return Roster(d["weights"], genesis_seed=d.get("genesis_seed", ""), epoch_len=d.get("epoch_len", 0),
                  epoch_beacons=d.get("epoch_beacons"))


# --------------------------------------------------------------------------- signed objects
def _sign(acct, obj, domain):
    return sign_bytes(acct, canonical(obj), domain=domain)


@functools.lru_cache(maxsize=65536)
def _recover_cached(data, sig, domain):
    try:
        return _n(recover_bytes(data, sig, domain=domain))
    except Exception:                                   # noqa: BLE001 -- malformed sig -> no signer
        return None


def _recover(obj, sig, domain):
    if not isinstance(sig, str):
        return None
    return _recover_cached(canonical(obj), sig, domain)


def timeout_body(ns, height, view, prev, failed):
    return {"w": "timeout", "v": VERSION, "ns": str(ns), "height": int(height), "view": int(view),
            "prev": str(prev), "failed": _n(failed)}


def sign_timeout(acct, ns, height, view, prev, failed):
    b = timeout_body(ns, height, view, prev, failed)
    return dict(b, signer=_n(acct.address), sig=_sign(acct, b, DOM_TIMEOUT))


def timeout_signer(t):
    """Recovered signer of a timeout attestation, or None if malformed / forged."""
    try:
        b = timeout_body(t["ns"], t["height"], t["view"], t["prev"], t["failed"])
    except Exception:                                   # noqa: BLE001
        return None
    s = _recover(b, t.get("sig"), DOM_TIMEOUT)
    return s if s and s == _n(t.get("signer", "")) else None


def endorse_body(ns, height, view, block_hash):
    return {"w": "endorse", "v": VERSION, "ns": str(ns), "height": int(height), "view": int(view),
            "block_hash": str(block_hash)}


def sign_endorsement(acct, ns, height, view, block_hash):
    b = endorse_body(ns, height, view, block_hash)
    return dict(b, endorser=_n(acct.address), sig=_sign(acct, b, DOM_ENDORSE))


def endorsement_signer(e):
    try:
        b = endorse_body(e["ns"], e["height"], e["view"], e["block_hash"])
    except Exception:                                   # noqa: BLE001
        return None
    s = _recover(b, e.get("sig"), DOM_ENDORSE)
    return s if s and s == _n(e.get("endorser", "")) else None


_TO_KEYS = ("w", "v", "ns", "height", "view", "prev", "failed", "signer", "sig")


def bare_timeout(t):
    """A timeout without its unsigned attachments (the lock certificate), as it goes into certificates."""
    return {k: t[k] for k in _TO_KEYS if k in t}


def prepare_body(ns, height, view, block_hash, just):
    """just = the view of the certificate the signer relies on (its own lock, or the proposal's PC); -1 if
    none. Signed so that preparing against one's own lock is self-proving (lock_violation)."""
    return {"w": "prepare", "v": VERSION, "ns": str(ns), "height": int(height), "view": int(view),
            "block_hash": str(block_hash), "just": int(just)}


def sign_prepare(acct, ns, height, view, block_hash, just=-1):
    b = prepare_body(ns, height, view, block_hash, just)
    return dict(b, signer=_n(acct.address), sig=_sign(acct, b, DOM_PREPARE))


def prepare_signer(p):
    try:
        b = prepare_body(p["ns"], p["height"], p["view"], p["block_hash"], p["just"])
    except Exception:                                   # noqa: BLE001
        return None
    s = _recover(b, p.get("sig"), DOM_PREPARE)
    return s if s and s == _n(p.get("signer", "")) else None


def pc_signers(prepares, ns, height, view, bh):
    """Distinct valid signers of prepare(ns, height, view, bh) among `prepares` (untrusted input)."""
    out = set()
    for p in prepares or []:
        if not isinstance(p, dict):
            continue
        s = prepare_signer(p)
        if s and p["ns"] == str(ns) and int(p["height"]) == int(height) and int(p["view"]) == int(view) \
                and p["block_hash"] == bh:
            out.add(s)
    return out


def _cert_by_view(cert, roster, ns, height, prev):
    by_view = defaultdict(set)
    for t in cert:
        s = timeout_signer(t)
        if s is None or t["ns"] != ns or int(t["height"]) != height or t["prev"] != prev:
            continue
        if _n(t["failed"]) != roster.proposer(ns, height, int(t["view"])):
            continue
        by_view[int(t["view"])].add(s)
    return by_view


def _timeouts_sha(cert):
    keyed = sorted(cert or [], key=lambda t: (int(t.get("view", 0)), _n(t.get("signer", ""))))
    return sha(canonical(keyed)) if keyed else ""


def make_block(acct, ns, height, view, prev, payload, timeout_cert=(), ts=0):
    cert = sorted([bare_timeout(t) for t in (timeout_cert or [])],
                  key=lambda t: (int(t.get("view", 0)), _n(t.get("signer", ""))))
    header = {"w": "header", "v": VERSION, "ns": str(ns), "height": int(height), "view": int(view),
              "prev": str(prev), "proposer": _n(acct.address), "payload_sha": sha(canonical(payload)),
              "timeouts_sha": _timeouts_sha(cert), "ts": int(ts)}
    return {"header": header, "sig": _sign(acct, header, DOM_HEADER), "payload": payload, "timeout_cert": cert}


def block_hash(block_or_header):
    h = block_or_header.get("header", block_or_header)
    return sha(canonical(h))


def header_signer(signed):
    """signed = {"header": ..., "sig": ...} (a block or a bare announcement)."""
    try:
        h = signed["header"]
    except Exception:                                   # noqa: BLE001
        return None
    s = _recover(h, signed.get("sig"), DOM_HEADER)
    return s if s and s == _n(h.get("proposer", "")) else None


def validate_block(block, roster, *, payload_check=None):
    """(ok, reason). Stateless: signature, schedule, payload hash, timeout certificate for skipped views,
    optional payload_check. Parent linkage is decided by fork choice, not here. Never raises: relay objects
    are untrusted input, so any malformed field is a rejection, not an exception."""
    try:
        return _validate_block(block, roster, payload_check=payload_check)
    except Exception:                                   # noqa: BLE001 -- untrusted input
        return False, "malformed block"


def _validate_block(block, roster, *, payload_check=None):
    try:
        h = block["header"]
        ns, height, view, prev = h["ns"], int(h["height"]), int(h["view"]), h["prev"]
    except Exception:                                   # noqa: BLE001
        return False, "malformed block"
    if int(h.get("v", -1)) != VERSION or h.get("w") != "header":
        return False, "unknown header version"
    if height < 1 or view < 0:
        return False, "bad height/view"
    signer = header_signer(block)
    if signer is None:
        return False, "bad proposer signature"
    if signer != roster.proposer(ns, height, view):
        return False, "proposer not scheduled for this height/view"
    if sha(canonical(block.get("payload"))) != h.get("payload_sha"):
        return False, "payload hash mismatch"
    cert = block.get("timeout_cert") or []
    if _timeouts_sha(cert) != h.get("timeouts_sha", ""):
        return False, "timeout certificate hash mismatch"
    by_view = _cert_by_view(cert, roster, ns, height, prev)
    for u in range(view):
        if not roster.majority(by_view.get(u, ())):
            return False, "missing timeout quorum for skipped view %d" % u
    if payload_check is not None:
        try:
            if not payload_check(block.get("payload")):
                return False, "payload rejected by payload_check"
        except Exception:                               # noqa: BLE001
            return False, "payload_check raised"
    return True, "ok"


# --------------------------------------------------------------------------- re-proposals + lock certificates
def reprop_body(ns, height, view, prev, bh, pc_view, proposer, timeouts_sha):
    return {"w": "reprop", "v": VERSION, "ns": str(ns), "height": int(height), "view": int(view),
            "prev": str(prev), "block_hash": str(bh), "pc_view": int(pc_view), "proposer": _n(proposer),
            "timeouts_sha": str(timeouts_sha)}


def make_reprop(acct, ns, height, view, prev, block, pc_view, pc, timeout_cert):
    """The view-`view` proposer re-proposes an EXISTING block (from an earlier view) that holds the highest
    prepared certificate it knows. The block keeps its hash; only the vote view changes."""
    cert = sorted([bare_timeout(t) for t in (timeout_cert or [])],
                  key=lambda t: (int(t.get("view", 0)), _n(t.get("signer", ""))))
    pcs = sorted([dict(p) for p in pc], key=lambda p: _n(p.get("signer", "")))
    body = reprop_body(ns, height, view, prev, block_hash(block), pc_view, acct.address, _timeouts_sha(cert))
    return {"rp": body, "sig": _sign(acct, body, DOM_REPROP), "block": block, "pc": pcs, "timeout_cert": cert}


def reprop_signer(r):
    try:
        b = r["rp"]
        body = reprop_body(b["ns"], b["height"], b["view"], b["prev"], b["block_hash"], b["pc_view"],
                           b["proposer"], b["timeouts_sha"])
    except Exception:                                   # noqa: BLE001
        return None
    if body != b:
        return None
    s = _recover(body, r.get("sig"), DOM_REPROP)
    return s if s and s == _n(b["proposer"]) else None


def validate_reprop(r, roster, *, payload_check=None):
    """(ok, reason). Never raises (untrusted relay input)."""
    try:
        return _validate_reprop(r, roster, payload_check=payload_check)
    except Exception:                                   # noqa: BLE001
        return False, "malformed reprop"


def _validate_reprop(r, roster, *, payload_check=None):
    b = r["rp"]
    if b.get("w") != "reprop" or int(b.get("v", -1)) != VERSION:
        return False, "unknown reprop version"
    ns, height, view, prev, bh = b["ns"], int(b["height"]), int(b["view"]), b["prev"], b["block_hash"]
    pcv = int(b["pc_view"])
    if height < 1 or view < 1 or not 0 <= pcv < view:
        return False, "bad height/view"
    s = reprop_signer(r)
    if s is None:
        return False, "bad reprop signature"
    if s != roster.proposer(ns, height, view):
        return False, "proposer not scheduled for this height/view"
    blk = r["block"]
    ok, _ = validate_block(blk, roster, payload_check=payload_check)
    if not ok:
        return False, "re-proposed block invalid"
    hh = blk["header"]
    if block_hash(blk) != bh or hh["ns"] != ns or int(hh["height"]) != height or hh["prev"] != prev \
            or int(hh["view"]) > pcv:
        return False, "re-proposed block does not match the reprop"
    if not roster.majority(pc_signers(r.get("pc"), ns, height, pcv, bh)):
        return False, "missing prepared certificate"
    cert = r.get("timeout_cert") or []
    if _timeouts_sha(cert) != b["timeouts_sha"]:
        return False, "timeout certificate hash mismatch"
    by_view = _cert_by_view(cert, roster, ns, height, prev)
    for u in range(view):
        if not roster.majority(by_view.get(u, ())):
            return False, "missing timeout quorum for skipped view %d" % u
    return True, "ok"


def lock_certificate(lc, roster, ns, height, prev, *, payload_check=None):
    """Verify a lock certificate {view, block, pc} attached to a timeout. -> (view, hash, block, pc) or None."""
    try:
        lv, blk, pc = int(lc["view"]), lc["block"], list(lc["pc"])
        ok, _ = validate_block(blk, roster, payload_check=payload_check)
        hh = blk["header"]
        if not ok or hh["ns"] != str(ns) or int(hh["height"]) != int(height) or hh["prev"] != prev \
                or int(hh["view"]) > lv:
            return None
        bh = block_hash(blk)
        if not roster.majority(pc_signers(pc, ns, height, lv, bh)):
            return None
        return lv, bh, blk, pc
    except Exception:                                   # noqa: BLE001
        return None


# --------------------------------------------------------------------------- fork choice + payouts
def fork_choice(blocks, endorsements, roster, *, payload_check=None, ns=None, start_prev=GENESIS_PREV,
                start_height=1):
    """Deterministic canonical chain from any set of blocks + endorsements (order of inputs is irrelevant).
    Returns a list of blocks, heights 1..k. Rule per height: among valid children of the chosen parent,
    max (is final, best single-view endorsement weight, -view, -hash). "Final" counts every signer exactly as
    is_final does, so a final block is never out-ranked (at most one exists per height, see the module doc). An endorser that endorsed two blocks in ONE view
    at one height counts for neither in that view (a vote changed in a LATER view under the lock rule is not
    equivocation and counts in its own view). ns: keep only blocks of this namespace. Every namespace shares
    GENESIS_PREV, so without it a valid height-1 block of ANOTHER campaign would be a candidate child."""
    valid = {}
    for b in blocks:
        ok, _ = validate_block(b, roster, payload_check=payload_check)
        if ok and (ns is None or b["header"]["ns"] == str(ns)):
            valid[block_hash(b)] = b
    votes = defaultdict(lambda: defaultdict(set))         # block_hash -> view -> {endorser}
    per_hv = defaultdict(lambda: defaultdict(set))        # (height, view) -> endorser -> {hash}
    for e in endorsements:
        s = endorsement_signer(e)
        if s is None or s not in roster.w:
            continue
        bh = e["block_hash"]
        b = valid.get(bh)
        if b is None or int(b["header"]["height"]) != int(e["height"]) or b["header"]["ns"] != e["ns"]:
            continue
        ev = int(e["view"])
        if ev < int(b["header"]["view"]):                 # a vote cannot precede its proposal
            continue
        votes[bh][ev].add(s)
        per_hv[(int(e["height"]), ev)][s].add(bh)
    children = defaultdict(list)                          # (height, prev) -> [block]
    for b in valid.values():
        children[(int(b["header"]["height"]), b["header"]["prev"])].append(b)
    chain = []
    parent = start_prev                                   # (start_prev, start_height): extend a known final
    height = int(start_height)                            # prefix instead of re-walking from genesis
    while True:
        kids = children.get((height, parent))
        if not kids:
            break

        def key(b, height=height):
            bh = block_hash(b)
            w, final = 0, False
            for u, ss in votes[bh].items():
                final = final or roster.majority(ss)          # same test as is_final: FINAL always wins
                bad = {x for x, hs in per_hv[(height, u)].items() if len(hs) > 1}
                w = max(w, roster.weight_of(ss - bad))
            return (final, w, -int(b["header"]["view"]), -int(bh, 16))

        best = max(kids, key=key)
        chain.append(best)
        parent = block_hash(best)
        height += 1
    return chain


def _view_endorsers(block, endorsements):
    """{vote view: {endorser}} of valid endorsements of this block (view >= the block's own view)."""
    bh = block_hash(block)
    h = block["header"]
    out = defaultdict(set)
    for e in endorsements:
        if not isinstance(e, dict) or e.get("block_hash") != bh:
            continue
        s = endorsement_signer(e)
        if s and e["ns"] == h["ns"] and int(e["height"]) == int(h["height"]) and int(e["view"]) >= int(h["view"]):
            out[int(e["view"])].add(s)
    return out


def endorsement_weight(block, endorsements, roster):
    """Best single-view endorsement weight."""
    return max([roster.weight_of(ss) for ss in _view_endorsers(block, endorsements).values()] or [0])


def is_final(block, endorsements, roster):
    """FINAL = endorsements from a strict weight majority in ONE view."""
    return any(roster.majority(ss) for ss in _view_endorsers(block, endorsements).values())


def canonical_balances(chain):
    """Replay payouts along the canonical chain only. A delta_cid is paid at most once (first occurrence)."""
    bal = defaultdict(float)
    paid = set()
    for b in chain:
        for m in (b.get("payload") or {}).get("mints", []):
            cid = m.get("delta_cid") or ""
            if cid and cid in paid:
                continue
            if cid:
                paid.add(cid)
            bal[_n(m["recipient"])] += float(m["amount"])
    return {a: round(v, 9) for a, v in sorted(bal.items())}


# --------------------------------------------------------------------------- relay names + transports
def hdr_name(ns, h, v):
    return "%s/%s/hdr/%d/%d" % (NAME_PREFIX, ns, int(h), int(v))


def blk_name(ns, h, v):
    return "%s/%s/blk/%d/%d" % (NAME_PREFIX, ns, int(h), int(v))


def end_name(ns, h, addr, view=0):
    return "%s/%s/end/%d/%d/%s" % (NAME_PREFIX, ns, int(h), int(view), _n(addr))


def prep_name(ns, h, v, addr):
    return "%s/%s/prep/%d/%d/%s" % (NAME_PREFIX, ns, int(h), int(v), _n(addr))


def fin_name(ns, h, addr):
    return "%s/%s/fin/%d/%s" % (NAME_PREFIX, ns, int(h), _n(addr))


def rp_name(ns, h, v):
    return "%s/%s/rp/%d/%d" % (NAME_PREFIX, ns, int(h), int(v))


def to_name(ns, h, v, addr):
    return "%s/%s/to/%d/%d/%s" % (NAME_PREFIX, ns, int(h), int(v), _n(addr))


def ev_name(ns, digest):
    return "%s/%s/ev/%s" % (NAME_PREFIX, ns, digest)


class InMemoryRelay:
    """Test fake with the RelayStore shape. Keeps every value ever written per name (history) so an
    overwrite (an equivocation attempt) stays visible to the auditor, as recorded observations would."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hist = defaultdict(list)

    def put(self, name, body):
        with self._lock:
            self._hist[name].append(bytes(body))
        return sha(body)

    def get_name(self, name):
        with self._lock:
            h = self._hist.get(name)
            return h[-1] if h else None

    def history(self, name=None):
        with self._lock:
            if name is not None:
                return list(self._hist.get(name, []))
            return {k: list(v) for k, v in self._hist.items()}


class DirRelay:
    """File-backed fake for multi-PROCESS local drills (still no network). Each put appends a new file under
    <root>/<name>/NNNNNN; get_name returns the newest. Writes are atomic (tmp + rename)."""

    def __init__(self, root):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, name):
        safe = name.replace("/", os.sep)
        return os.path.join(self.root, safe)

    def put(self, name, body):
        d = self._dir(name)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, ".tmp-%d-%d" % (os.getpid(), threading.get_ident()))
        with open(tmp, "wb") as f:
            f.write(body)
        for _ in range(1000):
            n = len([x for x in os.listdir(d) if not x.startswith(".")])
            dst = os.path.join(d, "%06d" % n)
            try:
                os.rename(tmp, dst)            # fails if dst exists on Windows -> retry next index
                return sha(body)
            except OSError:
                continue
        raise OSError("DirRelay: could not place %s" % name)

    def _files(self, name):
        d = self._dir(name)
        try:
            return sorted(x for x in os.listdir(d) if not x.startswith("."))
        except OSError:
            return []

    def get_name(self, name):
        fs = self._files(name)
        if not fs:
            return None
        with open(os.path.join(self._dir(name), fs[-1]), "rb") as f:
            return f.read()

    def history(self, name):
        out = []
        for x in self._files(name):
            with open(os.path.join(self._dir(name), x), "rb") as f:
                out.append(f.read())
        return out


def _load(body):
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:                                   # noqa: BLE001
        return None


# --------------------------------------------------------------------------- node
class Node:
    """One candidate proposer / validator. Drive with tick(now). Holds only its own key; everything else is
    read from the relay. payload_fn(height, prev) -> payload when this node proposes a FRESH block.

    Per height h it runs views 0, 1, ... (rule 3/4 of the module docstring): it votes only in its current view
    `cv`, enters view cv+1 when it sees TC(cv), times out cv after `deadline_s` without finality, and opens
    height h+1 once a block at h is final. All signed objects it has ever seen are kept (an overwrite on the
    relay cannot make it forget a vote or a certificate)."""

    def __init__(self, acct, roster, ns, relay, *, deadline_s=30.0, payload_fn=None, payload_check=None,
                 log=None, crash_after_header_at=None, state_path=None, attest_only=False):
        self.acct = acct
        self.me = _n(acct.address)
        self.roster, self.ns, self.relay = roster, str(ns), relay
        self.deadline = float(deadline_s)
        self.payload_fn = payload_fn or (lambda h, prev: {"mints": []})
        self.payload_check = payload_check
        self.log = log or (lambda s: None)
        self.crash_after_header_at = crash_after_header_at
        self.state_path = state_path          # last-sign state (votes, locks, proposals); None = crash-stop only
        # attest_only (2026-11-26, step 4 stage S1, default False = unchanged): the node votes, times out and
        # finalizes as usual but NEVER proposes or re-proposes, even in its own slot (its view then times out and
        # succession moves on). The S1 shadow attester (neurahash/glm_shadow_attester.py) is this role.
        self.attest_only = bool(attest_only)
        self.blocks = {}                      # hash -> valid block
        self.endorsements = {}                # (endorser, view, hash) -> endorsement
        self.prepares = {}                    # (signer, height, view, hash) -> prepare
        self.reprops = {}                     # (height, view, body digest) -> valid reprop
        self.tos = {}                         # (height, view, signer, prev) -> timeout (may carry "lock")
        self.endorsed_at = {}                 # height -> {view: hash I endorsed}
        self.prepared_at = {}                 # height -> {view: hash I prepared}
        self.timed_out = defaultdict(set)     # height -> {views I timed out}
        self.locks = {}                       # height -> {"view", "hash", "block", "pc"}
        self.proposed = set()                 # (height, view)
        self.height_start = None              # start time of the CURRENT view's timer
        self.h = 1                            # height being decided
        self.cv = 0                           # current view at self.h
        self.head_height = 0
        self.chain = []
        self.final_head = GENESIS_PREV        # hash of the highest final block (the prev for self.h)
        self._final_h = 0
        self.height_opened = {}
        self.max_voted = {}                   # height -> highest view I prepared/endorsed in (votes only go up)
        self._load_state()

    # -- last-sign state: written BEFORE anything signed leaves the node, so a restart with the same key
    #    never forgets a lock or signs a second, different vote/proposal for a slot it already used
    def _load_state(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        with open(self.state_path, "r", encoding="ascii") as f:
            st = json.load(f)
        if st.get("ns") != self.ns or _n(st.get("me", "")) != self.me or int(st.get("v", -1)) != VERSION:
            raise ValueError("state file belongs to another node/namespace/version: %s" % self.state_path)
        hk = lambda d: {int(k): v for k, v in d.items()}                      # noqa: E731
        self.prepared_at = {h: hk(d) for h, d in hk(st["prepared_at"]).items()}
        self.endorsed_at = {h: hk(d) for h, d in hk(st["endorsed_at"]).items()}
        self.timed_out = defaultdict(set, {h: set(vs) for h, vs in hk(st["timed_out"]).items()})
        self.locks = hk(st["locks"])
        self.max_voted = hk(st["max_voted"])
        self.proposed = {tuple(x) for x in st["proposed"]}

    def _save_state(self):
        if not self.state_path:
            return
        keep = lambda d: {str(h): x for h, x in d.items() if h >= self.h - 1}  # noqa: E731
        st = {"v": VERSION, "ns": self.ns, "me": self.me,
              "prepared_at": {k: {str(u): x for u, x in d.items()} for k, d in keep(self.prepared_at).items()},
              "endorsed_at": {k: {str(u): x for u, x in d.items()} for k, d in keep(self.endorsed_at).items()},
              "timed_out": {k: sorted(x) for k, x in keep(self.timed_out).items()},
              "locks": keep(self.locks), "max_voted": keep(self.max_voted),
              "proposed": sorted([h, v] for (h, v) in self.proposed if h >= self.h - 1)}
        d = os.path.dirname(os.path.abspath(self.state_path))
        os.makedirs(d, exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="ascii") as f:
            f.write(json.dumps(st, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    # -- reads
    def _poll(self, h, max_view):
        for v in range(max_view + 1):
            b = _load(self.relay.get_name(blk_name(self.ns, h, v)))
            if isinstance(b, dict) and isinstance(b.get("header"), dict) and b["header"].get("ns") == self.ns:
                ok, _ = validate_block(b, self.roster, payload_check=self.payload_check)
                if ok and int(b["header"]["height"]) == h:
                    self.blocks[block_hash(b)] = b
            r = _load(self.relay.get_name(rp_name(self.ns, h, v)))
            if isinstance(r, dict) and isinstance(r.get("rp"), dict) and r["rp"].get("ns") == self.ns:
                ok, _ = validate_reprop(r, self.roster, payload_check=self.payload_check)
                if ok and int(r["rp"]["height"]) == h and int(r["rp"]["view"]) == v:
                    self.reprops[(h, v, sha(canonical(r["rp"])))] = r
                    self.blocks[r["rp"]["block_hash"]] = r["block"]
            for a in (self.roster.addrs if v == 0 else ()):
                f = _load(self.relay.get_name(fin_name(self.ns, h, a)))
                if not (isinstance(f, dict) and isinstance(f.get("block"), dict)
                        and isinstance(f.get("endorsements"), list)):
                    continue
                fb = f["block"]
                ok, _ = validate_block(fb, self.roster, payload_check=self.payload_check)
                if not ok or fb["header"]["ns"] != self.ns or int(fb["header"]["height"]) != h:
                    continue
                self.blocks[block_hash(fb)] = fb
                for e in f["endorsements"][:len(self.roster.addrs)]:
                    s_ = endorsement_signer(e) if isinstance(e, dict) else None
                    if s_ and s_ in self.roster.w and e["ns"] == self.ns and int(e["height"]) == h:
                        self.endorsements.setdefault((s_, int(e["view"]), e["block_hash"]), e)
            for a in self.roster.addrs:
                p = _load(self.relay.get_name(prep_name(self.ns, h, v, a)))
                if isinstance(p, dict) and prepare_signer(p) == a and p["ns"] == self.ns \
                        and int(p["height"]) == h and int(p["view"]) == v:
                    self.prepares.setdefault((a, h, v, p["block_hash"]), p)
                e = _load(self.relay.get_name(end_name(self.ns, h, a, v)))
                if isinstance(e, dict) and endorsement_signer(e) == a and e["ns"] == self.ns \
                        and int(e["height"]) == h and int(e["view"]) == v:
                    self.endorsements.setdefault((a, v, e["block_hash"]), e)
                t = _load(self.relay.get_name(to_name(self.ns, h, v, a)))
                if isinstance(t, dict) and timeout_signer(t) == a and t["ns"] == self.ns \
                        and int(t["height"]) == h and int(t["view"]) == v \
                        and _n(t["failed"]) == self.roster.proposer(self.ns, h, v):
                    self.tos.setdefault((h, v, a, t["prev"]), t)

    def _to_signers(self, h, v, prev):
        return {a for (hh, vv, a, pp) in self.tos if hh == h and vv == v and pp == prev}

    def _recompute(self, now):
        # final blocks never change (safety), so only the tail above the final head is re-derived
        fh = self._final_h
        blocks = [b for b in self.blocks.values() if int(b["header"]["height"]) > fh]
        ends = [e for e in self.endorsements.values() if int(e["height"]) > fh]
        tail = fork_choice(blocks, ends, self.roster, payload_check=self.payload_check, ns=self.ns,
                           start_prev=self.final_head, start_height=fh + 1)
        self.chain = self.chain[:fh] + tail
        self.head_height = len(self.chain)
        for i in range(self._final_h, len(self.chain)):
            if is_final(self.chain[i], ends, self.roster):
                fh = i + 1
        if fh > self._final_h:
            for k in range(self._final_h + 1, fh + 1):
                self._publish_finality(self.chain[k - 1], ends)
            self._final_h = fh
            self.final_head = block_hash(self.chain[fh - 1])
            self.h, self.cv, self.height_start = fh + 1, 0, now
            for d in (self.prepares, self.tos, self.reprops):           # vote state of decided heights
                for k in [k for k in d if (k[1] if d is self.prepares else k[0]) < self.h]:
                    del d[k]
            self.height_opened.setdefault(self.h, now)
            self.log("[glm-ra] %s final_head_height=%d" % (self.me[:10], fh))

    def _publish_finality(self, b, ends):
        """Catch-up gossip (Tendermint LastCommit): re-publish the block and the single-view endorsement
        majority that made it final, so a node that missed some endorsements (e.g. a byzantine endorser showed
        its vote to others only) still learns the decision after its peers moved to the next height."""
        for u, ss in sorted(_view_endorsers(b, ends).items()):
            if self.roster.majority(ss):
                bh = block_hash(b)
                es = [self.endorsements[(a, u, bh)] for a in sorted(ss) if (a, u, bh) in self.endorsements]
                self._put(fin_name(self.ns, int(b["header"]["height"]), self.me), {"block": b, "endorsements": es})
                return

    def final_height(self):
        return self._final_h

    def head_hash(self):
        return block_hash(self.chain[-1]) if self.chain else GENESIS_PREV

    def _advance_views(self, now):
        """Pacemaker: TC(cv) seen -> enter cv+1 with a fresh timer (repeat while certificates exist)."""
        while self.roster.majority(self._to_signers(self.h, self.cv, self.final_head)):
            self.cv += 1
            self.height_start = now
            self._poll(self.h, self.cv + 1)

    # -- writes
    def _put(self, name, obj):
        self.relay.put(name, canonical(obj))

    def _timeout(self, h, v, prev):
        t = sign_timeout(self.acct, self.ns, h, v, prev, self.roster.proposer(self.ns, h, v))
        lock = self.locks.get(h)
        if lock is not None:                          # unsigned, self-verifying attachment
            t = dict(t, lock={"view": lock["view"], "block": lock["block"], "pc": lock["pc"]})
        self.timed_out[h].add(v)
        self._save_state()
        self._put(to_name(self.ns, h, v, self.me), t)
        self.tos.setdefault((h, v, self.me, prev), t)

    def _proposals(self, h, v, prev):
        """Valid proposals for (h, v) extending prev: [(target hash, pc_view or None)], deterministic order."""
        out = []
        for bh, b in self.blocks.items():
            hd = b["header"]
            if int(hd["height"]) == h and int(hd["view"]) == v and hd["prev"] == prev:
                out.append((bh, None))
        for (hh, vv, _), r in self.reprops.items():
            if hh == h and vv == v and r["rp"]["prev"] == prev:
                out.append((r["rp"]["block_hash"], int(r["rp"]["pc_view"])))
        return sorted(set(out), key=lambda c: (c[0], -1 if c[1] is None else c[1]))

    def _vote(self, h, v, prev):
        lock = self.locks.get(h)
        if v < self.max_voted.get(h, -1):             # never vote in a view below one I already voted in
            return
        if v not in self.prepared_at.get(h, {}):
            for bh, pcv in self._proposals(h, v, prev):
                # LOCK RULE: unlocked, or the locked block itself, or a certificate at least as new as the lock
                if lock is None or bh == lock["hash"] or (pcv is not None and pcv >= lock["view"]):
                    just = max(-1 if pcv is None else pcv,
                               lock["view"] if (lock is not None and bh == lock["hash"]) else -1)
                    pr = sign_prepare(self.acct, self.ns, h, v, bh, just)
                    self.prepared_at.setdefault(h, {})[v] = bh
                    self.max_voted[h] = v
                    self._save_state()
                    self._put(prep_name(self.ns, h, v, self.me), pr)
                    self.prepares.setdefault((self.me, h, v, bh), pr)
                    break
        if v not in self.endorsed_at.get(h, {}):
            groups = defaultdict(set)
            for (s_, hh, vv, bh) in self.prepares:
                if hh == h and vv == v:
                    groups[bh].add(s_)
            for bh in sorted(groups):
                b = self.blocks.get(bh)
                if b is None or b["header"]["prev"] != prev or not self.roster.majority(groups[bh]):
                    continue
                pc = [self.prepares[(s_, h, v, bh)] for s_ in sorted(groups[bh])]
                self.locks[h] = {"view": v, "hash": bh, "block": b, "pc": pc}       # PC(v, B) -> lock
                e = sign_endorsement(self.acct, self.ns, h, v, bh)
                self.endorsed_at.setdefault(h, {})[v] = bh
                self.max_voted[h] = v
                self._save_state()
                self._put(end_name(self.ns, h, self.me, v), e)
                self.endorsements.setdefault((self.me, v, bh), e)
                break

    def _highest_pc(self, h, v, prev):
        """Highest prepared certificate at height h below view v, from prepares seen and from the lock
        certificates attached to timeouts. -> (view, hash, block, pc) or None."""
        cands = []
        groups = defaultdict(set)
        for (s_, hh, vv, bh) in self.prepares:
            if hh == h and vv < v:
                groups[(vv, bh)].add(s_)
        for (vv, bh), ss in groups.items():
            b = self.blocks.get(bh)
            if b is not None and b["header"]["prev"] == prev and self.roster.majority(ss):
                cands.append((vv, bh, b, [self.prepares[(s_, h, vv, bh)] for s_ in sorted(ss)]))
        for (hh, vv, _, pp), t in self.tos.items():
            if hh == h and pp == prev and isinstance(t.get("lock"), dict):
                lc = lock_certificate(t["lock"], self.roster, self.ns, h, prev, payload_check=self.payload_check)
                if lc is not None and lc[0] < v:
                    self.blocks.setdefault(lc[1], lc[2])
                    cands.append(lc)
        if not cands:
            return None
        return max(cands, key=lambda c: (c[0], -int(c[1], 16)))

    def _proposal_cert(self, h, v, prev):
        """Timeouts proving TC(u) for every u < v, or None if some view still lacks a quorum."""
        cert = []
        for u in range(v):
            ts = [t for (hh, uu, a, pp), t in self.tos.items() if hh == h and uu == u and pp == prev]
            if not self.roster.majority(timeout_signer(t) for t in ts):
                return None
            cert.extend(ts)
        return cert

    def _maybe_propose(self, h, v, prev, now):
        if self.attest_only:
            return
        if self.roster.proposer(self.ns, h, v) != self.me or (h, v) in self.proposed:
            return
        cert = self._proposal_cert(h, v, prev)
        if cert is None:
            return
        best = self._highest_pc(h, v, prev)
        self.proposed.add((h, v))
        self._save_state()                            # a restart never signs a second proposal for (h, v)
        if best is not None:                          # MUST re-propose the highest certified block
            lv, bh, b, pc = best
            r = make_reprop(self.acct, self.ns, h, v, prev, b, lv, pc, cert)
            self._put(rp_name(self.ns, h, v), r)
            self.reprops[(h, v, sha(canonical(r["rp"])))] = r
            self.log("[glm-ra] %s re-proposed h=%d view=%d block_view=%d pc_view=%d"
                     % (self.me[:10], h, v, int(b["header"]["view"]), lv))
            return
        blk = make_block(self.acct, self.ns, h, v, prev, self.payload_fn(h, prev), cert, ts=int(now))
        self._put(hdr_name(self.ns, h, v), {"header": blk["header"], "sig": blk["sig"]})
        if self.crash_after_header_at is not None and h == int(self.crash_after_header_at):
            raise SystemExit("simulated crash after header announce at height %d" % h)
        self._put(blk_name(self.ns, h, v), blk)
        self.blocks[block_hash(blk)] = blk
        self.log("[glm-ra] %s proposed h=%d view=%d" % (self.me[:10], h, v))

    # -- one step
    def tick(self, now):
        if self.height_start is None:
            self.height_start = now
            self.height_opened.setdefault(1, now)
        self._poll(self.h, self.cv + 1)
        self._recompute(now)
        self._advance_views(now)
        h, v, prev = self.h, self.cv, self.final_head
        self._maybe_propose(h, v, prev, now)
        if v not in self.timed_out[h]:
            self._vote(h, v, prev)
            self._recompute(now)
            if self.h != h:
                return
            if now - self.height_start >= self.deadline:
                self._timeout(h, v, prev)

    def balances(self):
        return canonical_balances(self.chain)


# --------------------------------------------------------------------------- auditor
def _hist(relay, name):
    if hasattr(relay, "history"):
        return relay.history(name)
    b = relay.get_name(name)
    return [b] if b else []


def audit(relay, roster, ns, *, max_height, max_view=8, now=None, height_starts=None, deadline_s=30.0,
          grace_s=0.0, extra_observations=()):
    """Scan relay history (+ any headers nodes forwarded) and return a list of evidence dicts.
    height_starts: {height: time the height opened} used for withheld/missed judgments (None -> clockless)."""
    ev = []
    headers = defaultdict(dict)          # (h, v) -> block_hash -> signed header
    reps = defaultdict(dict)             # (h, v) -> body digest -> {"rp", "sig"}
    ends = defaultdict(dict)             # (h, v, endorser) -> block_hash -> endorsement
    preps = defaultdict(dict)            # (h, v, signer) -> block_hash -> prepare
    tos = defaultdict(dict)              # (h, v, signer) -> timeout
    for obs in extra_observations:
        if "header" in obs and header_signer(obs):
            hh = obs["header"]
            headers[(int(hh["height"]), int(hh["view"]))][block_hash(obs)] = {"header": hh, "sig": obs["sig"]}
    for h in range(1, int(max_height) + 1):
        for v in range(max_view + 1):
            for nm in (hdr_name(ns, h, v), blk_name(ns, h, v)):
                for body in _hist(relay, nm):
                    o = _load(body)
                    if isinstance(o, dict) and header_signer(o) and int(o["header"]["height"]) == h \
                            and int(o["header"]["view"]) == v and o["header"]["ns"] == ns:
                        headers[(h, v)][block_hash(o)] = {"header": o["header"], "sig": o["sig"]}
            for body in _hist(relay, rp_name(ns, h, v)):
                o = _load(body)
                if isinstance(o, dict) and reprop_signer(o) and o["rp"]["ns"] == ns \
                        and int(o["rp"]["height"]) == h and int(o["rp"]["view"]) == v:
                    reps[(h, v)][sha(canonical(o["rp"]))] = {"rp": o["rp"], "sig": o["sig"]}
            for a in roster.addrs:
                for body in _hist(relay, to_name(ns, h, v, a)):
                    t = _load(body)
                    if isinstance(t, dict) and timeout_signer(t) == a:
                        tos[(h, v, a)] = bare_timeout(t)
                for body in _hist(relay, end_name(ns, h, a, v)):
                    e = _load(body)
                    if isinstance(e, dict) and endorsement_signer(e) == a and e["ns"] == ns \
                            and int(e["height"]) == h and int(e["view"]) == v:
                        ends[(h, v, a)][e["block_hash"]] = e
                for body in _hist(relay, prep_name(ns, h, v, a)):
                    p = _load(body)
                    if isinstance(p, dict) and prepare_signer(p) == a and p["ns"] == ns \
                            and int(p["height"]) == h and int(p["view"]) == v:
                        preps[(h, v, a)][p["block_hash"]] = p
    # (a) proposer equivocation: two headers, or two proposals (reprop/reprop, reprop/header), one slot
    for (h, v), byhash in sorted(headers.items()):
        if len(byhash) > 1:
            hs = sorted(byhash)
            ev.append({"kind": "proposer_equivocation", "ns": ns, "height": h, "view": v,
                       "offender": byhash[hs[0]]["header"]["proposer"], "a": byhash[hs[0]], "b": byhash[hs[1]]})
    for (h, v), bydig in sorted(reps.items()):
        objs = [bydig[k] for k in sorted(bydig)] + [headers[(h, v)][k] for k in sorted(headers.get((h, v), {}))]
        if len(objs) > 1:
            ev.append({"kind": "reproposal_equivocation", "ns": ns, "height": h, "view": v,
                       "offender": objs[0]["rp"]["proposer"], "a": objs[0], "b": objs[1]})
    # (b) prepare / endorsement equivocation (two blocks, one signer, one height+view)
    for kind, table in (("endorsement_equivocation", ends), ("prepare_equivocation", preps)):
        for (h, v, a), byhash in sorted(table.items()):
            if len(byhash) > 1:
                hs = sorted(byhash)
                ev.append({"kind": kind, "ns": ns, "height": h, "view": v, "offender": a,
                           "a": byhash[hs[0]], "b": byhash[hs[1]]})
    # (c) lock violation: endorsed B in view v, later prepared X != B citing a certificate older than v
    for (h, v, a), byhash in sorted(ends.items()):
        for bh, e in sorted(byhash.items()):
            for (h2, v1, a2), pb in sorted(preps.items()):
                if h2 != h or a2 != a or v1 <= v:
                    continue
                for xh, p in sorted(pb.items()):
                    if xh != bh and int(p["just"]) < v:
                        ev.append({"kind": "lock_violation", "ns": ns, "height": h, "offender": a,
                                   "endorsement": e, "prepare": p})
    # (d) withheld, (e) missed slot
    published_at = defaultdict(set)      # (h, v) -> hashes of blocks actually published at the slot
    for (h, v) in list(headers):
        for body in _hist(relay, blk_name(ns, h, v)):
            o = _load(body)
            if isinstance(o, dict) and "header" in o and header_signer(o) \
                    and sha(canonical(o.get("payload"))) == o["header"].get("payload_sha"):
                published_at[(h, v)].add(block_hash(o))
    if height_starts is None:
        # no clock: a signed header whose block never appeared while a LATER view at that height (or the next
        # height) did publish a proposal is withheld -- the round moved on without it
        for (h, v), byhash in sorted(headers.items()):
            later = any(published_at.get((h, u)) or reps.get((h, u)) for u in range(v + 1, max_view + 1)) or \
                any(published_at.get((h + 1, u)) for u in range(max_view + 1))
            if not later:
                continue
            for bh, sh in byhash.items():
                if bh not in published_at.get((h, v), set()):
                    ev.append({"kind": "withheld_block", "ns": ns, "height": h, "view": v,
                               "offender": sh["header"]["proposer"], "signed_header": sh,
                               "attested_at": now, "self_proving": False})
    if height_starts is not None and now is not None:
        for h in range(1, int(max_height) + 1):
            hs0 = height_starts.get(h)
            if hs0 is None:
                continue
            for v in range(max_view + 1):
                due = hs0 + (v + 1) * float(deadline_s) + float(grace_s)
                if now < due:
                    continue
                for bh, sh in headers.get((h, v), {}).items():
                    if bh not in published_at.get((h, v), set()):
                        ev.append({"kind": "withheld_block", "ns": ns, "height": h, "view": v,
                                   "offender": sh["header"]["proposer"], "signed_header": sh,
                                   "attested_at": now, "self_proving": False})
                signers = [a for a in roster.addrs if (h, v, a) in tos]
                if roster.majority(signers) and not headers.get((h, v)) and not reps.get((h, v)):
                    ev.append({"kind": "missed_slot", "ns": ns, "height": h, "view": v,
                               "offender": roster.proposer(ns, h, v), "timeout_signers": signers,
                               "self_proving": False})
    return ev


def _proposal_slot(o):
    """(signer, ns, height, view, signed-body digest) of a signed header or reprop, or None."""
    if isinstance(o.get("rp"), dict):
        s, b = reprop_signer(o), o["rp"]
    else:
        s, b = header_signer(o), o["header"]
    if not s:
        return None
    return s, b["ns"], int(b["height"]), int(b["view"]), sha(canonical(b))


def verify_evidence(e, roster):
    """(ok, reason) for the SELF-PROVING kinds; anyone can check this offline (stage-3 slashing input).
    withheld_block / missed_slot return (False, 'attested ...'): they need an auditor quorum, not one party."""
    k = e.get("kind")
    try:
        if k == "proposer_equivocation":
            a, b = e["a"], e["b"]
            sa, sb = header_signer(a), header_signer(b)
            ha, hb = a["header"], b["header"]
            if not sa or sa != sb or sa != _n(e["offender"]) or sa not in roster.w:
                return False, "signers do not match offender"
            if (ha["ns"], int(ha["height"]), int(ha["view"])) != (hb["ns"], int(hb["height"]), int(hb["view"])):
                return False, "different slots"
            if block_hash(a) == block_hash(b):
                return False, "same block"
            return True, "ok"
        if k == "reproposal_equivocation":
            pa, pb = _proposal_slot(e["a"]), _proposal_slot(e["b"])
            if not isinstance(e["a"].get("rp"), dict) or pa is None or pb is None:
                return False, "not a signed reprop pair"
            if pa[0] != pb[0] or pa[0] != _n(e["offender"]) or pa[0] not in roster.w:
                return False, "signers do not match offender"
            if pa[1:4] != pb[1:4]:
                return False, "different slots"
            if pa[4] == pb[4]:
                return False, "same proposal"
            return True, "ok"
        if k in ("endorsement_equivocation", "prepare_equivocation"):
            a, b = e["a"], e["b"]
            f = endorsement_signer if k == "endorsement_equivocation" else prepare_signer
            sa, sb = f(a), f(b)
            if not sa or sa != sb or sa != _n(e["offender"]) or sa not in roster.w:
                return False, "signers do not match offender"
            if (a["ns"], int(a["height"]), int(a["view"])) != (b["ns"], int(b["height"]), int(b["view"])) \
                    or a["block_hash"] == b["block_hash"]:
                return False, "not conflicting (votes in different views may differ under the lock rule)"
            return True, "ok"
        if k == "lock_violation":
            en, p = e["endorsement"], e["prepare"]
            se, sp = endorsement_signer(en), prepare_signer(p)
            if not se or se != sp or se != _n(e["offender"]) or se not in roster.w:
                return False, "signers do not match offender"
            if (en["ns"], int(en["height"])) != (p["ns"], int(p["height"])):
                return False, "different heights"
            if not (int(p["view"]) > int(en["view"]) and p["block_hash"] != en["block_hash"]
                    and int(p["just"]) < int(en["view"])):
                return False, "not a lock violation"
            return True, "ok"
        if k == "timeout_endorse_conflict":
            return False, "not a fault under the v2 lock rule: a timeout may follow a vote in the same view"
        if k in ("withheld_block", "missed_slot"):
            return False, "attested evidence: needs an auditor quorum"
    except Exception:                                   # noqa: BLE001
        return False, "malformed evidence"
    return False, "unknown kind"


# --------------------------------------------------------------------------- coordinator shadow hook
def coordinator_shadow_record(record_path, *, recipient, amount, height, cid, credited):
    """Called by the GLM coordinator ONLY when the flag is on (fenced at the call site). Shadow mode: appends
    the settled mint as a candidate payload line to a local JSONL that a round-authority node can propose
    from. No network, no signing, no change to the coordinator's own settlement chain."""
    if not credited:
        return False
    rec = {"recipient": _n(recipient), "amount": float(amount), "height": int(height),
           "delta_cid": str(cid or ""), "ts": int(time.time())}
    d = os.path.dirname(os.path.abspath(record_path))
    os.makedirs(d, exist_ok=True)
    with open(record_path, "a", encoding="ascii") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return True
