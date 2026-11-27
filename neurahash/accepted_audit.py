"""TAMPER-EVIDENT accepted records for the GLM lane (2026-11-25, default OFF via NEURAHASH_SD_PUBLISH_AUDIT).
Doc: docs/research/PAYOUT_QUORUM_STEP3_2026-11-24.md, section "Relay-only re-derivation".

Every accepted record  sharddiloco/glm/accepted/<campaign>/r<event>  keeps ALL its existing fields (so the current
readers, tools/glm_verifier_shadow.py and tools/glm_payout_rederive.py, are unaffected) and gains ONE additive key,
"audit":

    {"v": 1, "domain": "glm-accepted-audit", "campaign": str, "event": int, "coord_address": "0x..",
     "prev_hash": hex64 | None,           # record_hash of r<event-1> as the coordinator published it
     "reset": {"reason": str} (optional),  # REQUIRED whenever prev_hash is None (campaign start, unreadable prev)
     "boot": {"ce": float} (optional),     # first record after a coordinator (re)boot: the boot held-out CE
     "prev_ce": float,                     # the CE this event's held-out gain is measured from (heldout_gain =
                                           #   max(0, prev_ce - heldout_ce)); = boot.ce on a boot record
     "gains": [{"miner", "cid", "accepted", "gain"}] (optional),  # the per-row probe gains attribute_minted split by
     "gains_truncated": int (optional),
     "chain_name": str (optional),         # the named object the live public payout chain is PUT under
     "sig": "0x.."}                        # coordinator secp256k1 signature over record_hash, domain DOMAIN

record_hash(rec) = sha256(canonical(rec with audit.sig removed)), so the signature and the successor's prev_hash
cover every field of the record, old and new. A reader holding the coordinator ADDRESS (obtained out of band, the
same address that signs the payout blocks) detects, each with a distinct reason (verify_chain):
    reordered       the record stored under r<k> says it is event != k
    campaign        the signed record names a different campaign
    bad_signature   no / a forged signature (does not recover to the pinned coordinator address)
    overwritten     a validly signed record whose hash is not what its successor attested (prev_hash)
    dropped         an event missing between the first audited event and the tip
    reset_unmarked  an audited record with no prev_hash and no reset marker (a silent restart of the chain)
    unaudited       a record without the audit block after the audit chain started
Limitation (stated, not hidden): a coordinator that re-signs EVERY record from some event to the tip produces a
consistent chain; detecting that needs an external anchor of the head hash (e.g. the OTS anchor of the payout
chain), not this module.

Pure module: no sockets, no torch. Reuses glm_committee.canonical and neura_l1 secp256k1 sign/recover.
"""
from __future__ import annotations

import copy
import hashlib
import math

from neurahash import glm_committee as gc

DOMAIN = "glm-accepted-audit"
VERSION = 1
AUDIT_KEY = "audit"
MAX_GAINS = 64                  # per record; more rows -> accepted rows only, then truncated (flagged)
MAX_AUDIT_BYTES = 16384         # bound on the canonical audit block (the relay stores small named records)
MAX_TIP_SLACK = 1024            # a pointer/tip may run at most this many events past the newest SIGNED record
MAX_DROPPED_LISTED = 4096       # per-event "dropped" entries listed; the rest are counted (dropped_unlisted)
MAX_DEPTH = 64                  # nesting bound on any record a reader handles (real records nest < 8)

A_REORDERED = "reordered"
A_CAMPAIGN = "campaign"
A_BAD_SIG = "bad_signature"
A_OVERWRITTEN = "overwritten"
A_DROPPED = "dropped"
A_RESET = "reset_unmarked"
A_UNAUDITED = "unaudited"
REASONS = (A_REORDERED, A_CAMPAIGN, A_BAD_SIG, A_OVERWRITTEN, A_DROPPED, A_RESET, A_UNAUDITED)


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _strip_sig(rec):
    out = dict(rec)
    a = out.get(AUDIT_KEY)
    if isinstance(a, dict) and "sig" in a:
        a = dict(a)
        a.pop("sig", None)
        out[AUDIT_KEY] = a
    return out


def record_hash(rec):
    """sha256 hex of the canonical record with audit.sig removed (an unaudited record hashes whole)."""
    return hashlib.sha256(gc.canonical(_strip_sig(rec))).hexdigest()


def link_claims(rec, legacy_keys=()):
    """Review 3 fix #23: [(label, hex)] = what `rec` claims as its predecessor's hash, in the order a hash-chain
    link check must read them. The publish-audit chain stores the link UNDER the audit block (audit.prev_hash,
    signed); a top-level `legacy_keys` field comes after it. A signed reset (audit.prev_hash None with a reset
    marker) claims no link and yields nothing for the audit key. tools/glm_verifier_checks.ChainTracker read only
    the top-level keys, so its chain_break link check never ran on an audited record."""
    out = []
    if not isinstance(rec, dict):
        return out
    a = rec.get(AUDIT_KEY)
    if isinstance(a, dict) and a.get("prev_hash") is not None:
        out.append(("audit.prev_hash", str(a.get("prev_hash")).lower()))
    for k in legacy_keys or ():
        if k in rec:
            out.append((k, str(rec[k] or "").lower()))
    return out


def link_matches(want, prev_rec, known=()):
    """True when the claimed predecessor hash `want` is the record_hash of `prev_rec` (the value audit.prev_hash
    carries) or one of `known` (a caller's other accepted encodings, e.g. the manifest sha256 of the stored bytes).
    A record that cannot be hashed matches nothing but `known`."""
    w = str(want or "").lower()
    if w and w in {str(k or "").lower() for k in known}:
        return True
    try:
        return bool(w) and isinstance(prev_rec, dict) and w == record_hash(prev_rec)
    except Exception:                                    # noqa: BLE001 -- untrusted bytes: not a record
        return False


def gains_rows(per_expert, cid_by_miner=None):
    """The per-row gate gains attribute_minted split the held-out gain by, in per_expert ORDER (the float sum order
    matters for an exact re-derivation). Returns (rows, n_dropped)."""
    cid_by_miner = cid_by_miner or {}
    rows = []
    for pe in per_expert or ():
        if not isinstance(pe, dict):
            continue
        rows.append({"miner": str(pe.get("miner")), "cid": cid_by_miner.get(pe.get("miner")),
                     "accepted": bool(pe.get("accepted")), "gain": _finite(pe.get("gain", 0.0))})
    if len(rows) <= MAX_GAINS:
        return rows, 0
    acc = [r for r in rows if r["accepted"]]       # rejected rows carry no share; drop them first
    if len(acc) <= MAX_GAINS:
        return acc, len(rows) - len(acc)
    return acc[:MAX_GAINS], len(rows) - MAX_GAINS


def build_audit(*, campaign, event, coord_address, prev_hash, prev_ce, reset=None, boot_ce=None, gains=None,
                gains_dropped=0, chain_name=None):
    a = {"v": VERSION, "domain": DOMAIN, "campaign": str(campaign or ""), "event": int(event),
         "coord_address": str(coord_address or "").lower(), "prev_hash": prev_hash,
         "prev_ce": _finite(prev_ce)}
    if reset:
        a["reset"] = {"reason": str(reset)[:80]}
    if boot_ce is not None:
        a["boot"] = {"ce": _finite(boot_ce)}
    if gains is not None:
        a["gains"] = list(gains)
        if gains_dropped:
            a["gains_truncated"] = int(gains_dropped)
    if chain_name:
        a["chain_name"] = str(chain_name)[:128]
    if len(gc.canonical(a)) > MAX_AUDIT_BYTES:      # bounded object: drop the gains, say so
        n = len(a.pop("gains", []) or [])
        a["gains_truncated"] = int(a.get("gains_truncated", 0)) + n
    return a


def seal(record, account, audit):
    """A COPY of `record` with `audit` attached and signed by `account` (None -> unsigned, carried for (b)/(c)
    only; a reader reports it bad_signature). The record's own fields are never changed."""
    out = copy.deepcopy(record)
    out[AUDIT_KEY] = dict(audit)
    if account is not None:
        from neura_l1.signing import sign_bytes
        out[AUDIT_KEY]["sig"] = sign_bytes(account, record_hash(out).encode("ascii"), domain=DOMAIN)
    return out


def signer_of(rec):
    """Lower-case address recovered from the record's audit signature, or None."""
    a = rec.get(AUDIT_KEY) if isinstance(rec, dict) else None
    if not isinstance(a, dict) or not a.get("sig"):
        return None
    try:
        from neura_l1.signing import recover_bytes
        return str(recover_bytes(record_hash(rec).encode("ascii"), a["sig"], domain=DOMAIN)).lower()
    except Exception:                                    # noqa: BLE001 -- untrusted bytes
        return None


def depth_ok(obj, max_depth=MAX_DEPTH):
    """True when `obj` nests at most `max_depth` containers deep. Iterative (never recurses), so a hostile record
    cannot raise RecursionError here; used before hashing / canonicalizing untrusted records."""
    stack = [(obj, 1)]
    while stack:
        o, d = stack.pop()
        if isinstance(o, dict):
            if d > max_depth:
                return False
            stack.extend((v, d + 1) for v in o.values())
        elif isinstance(o, (list, tuple)):
            if d > max_depth:
                return False
            stack.extend((v, d + 1) for v in o)
    return True


def verify_chain(records, campaign, coord_address, tip_event=None, check=None):
    """PURE. `records` = {stored_event(int): record} (as fetched from r<event> names). Returns
      {"ok": bool, "events": {e: {"status": "ok"|"bad"|"unaudited_pre", "reason": code|"", "detail": str}},
       "verified": [events whose audit block is signed, linked and trustworthy], "reasons": {code: n},
       "first_audited": e|None, "tip": e|None}
    Records before the first audited one are pre-deployment ("unaudited_pre", not an error).

    BOUNDED (review fix 2026-11-25, finding #5): the work is O(number of records), never O(an event span someone
    else chose). "dropped" events are looked for only between the first and the newest record SIGNED by the pinned
    coordinator (an unsigned record anyone can PUT at r<huge> never widens the span); `tip_event` (the unsigned
    pointer) extends it by at most MAX_TIP_SLACK ("tip_clamped" is set when it asked for more); at most
    MAX_DROPPED_LISTED missing events get their own entry, the rest are counted in reasons and "dropped_unlisted".
    A record nested deeper than MAX_DEPTH (or one that cannot be hashed) is treated as malformed. `check()`
    (optional) is called once per record and may raise to abort (the miner's CPU/memory budget)."""
    coord = str(coord_address or "").lower()
    camp = str(campaign or "")
    evs = {}
    for k, r in (records or {}).items():
        try:
            e = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(r, dict) and not depth_ok(r):
            r = "malformed: nested deeper than %d" % MAX_DEPTH
        evs[e] = r
    out = {"ok": True, "events": {}, "verified": [], "reasons": {}, "first_audited": None, "tip": None}

    def _set(e, status, reason="", detail=""):
        out["events"][e] = {"status": status, "reason": reason, "detail": detail}
        if status == "bad":
            out["ok"] = False
            out["reasons"][reason] = out["reasons"].get(reason, 0) + 1

    hashes = {}
    for e in sorted(evs):
        r = evs[e]
        if not isinstance(r, dict):
            continue
        if check is not None:
            check()
        try:
            hashes[e] = record_hash(r)
        except Exception:                                    # noqa: BLE001 -- untrusted bytes: not a record
            evs[e] = "malformed: cannot be hashed"
    audited = sorted(e for e, r in evs.items() if isinstance(r, dict) and isinstance(r.get(AUDIT_KEY), dict))
    if not audited:
        for e in sorted(evs):
            _set(e, "unaudited_pre")
        return out
    first = audited[0]
    out["first_audited"] = first
    signer = {}
    for e in audited:
        if check is not None:
            check()
        signer[e] = signer_of(evs[e])
    signed = [e for e in audited if signer[e] is not None and signer[e] == coord
              and str(evs[e][AUDIT_KEY].get("coord_address", "")).lower() == coord]
    try:
        tip = int(tip_event) if tip_event is not None else None
    except (TypeError, ValueError):
        tip = None
    top = None                                              # the span "dropped" is judged over: signed records only
    if signed:
        top = signed[-1]
        if tip is not None and tip > top:
            if tip > top + MAX_TIP_SLACK:
                out["tip_clamped"] = tip
            top = min(tip, top + MAX_TIP_SLACK)
    out["tip"] = max(max(evs), top if top is not None else max(evs))
    missing = []                                            # events in [first signed, top] with no record at all
    n_missing = 0
    if signed:
        lo = signed[0]
        prev = lo - 1
        for e in sorted(x for x in evs if lo <= x <= top) + [top + 1]:
            gap = e - prev - 1
            if gap > 0:
                n_missing += gap
                room = MAX_DROPPED_LISTED - len(missing)
                if room > 0:
                    missing.extend(range(prev + 1, prev + 1 + min(gap, room)))
            prev = e
    sig_ok = {}
    for e in sorted(set(evs) | set(missing)):
        r = evs.get(e)
        if e < first:
            if r is not None:
                _set(e, "unaudited_pre")
            continue
        if not isinstance(r, dict):
            if r is None:
                _set(e, "bad", A_DROPPED, "no record at r%d (audited chain runs %d..%d)" % (e, first, top))
            else:
                _set(e, "bad", A_DROPPED, "r%d is not a usable record (%s)" % (e, str(r)[:60]))
            continue
        a = r.get(AUDIT_KEY)
        if not isinstance(a, dict):
            _set(e, "bad", A_UNAUDITED, "r%d has no audit block after the audit chain started at r%d" % (e, first))
            continue
        try:
            rev, aev = int(r.get("event", e)), int(a.get("event", -1))
        except (TypeError, ValueError):
            rev, aev = -1, -1
        if rev != e or aev != e:
            _set(e, "bad", A_REORDERED, "stored at r%d but says event %s (audit %s)" % (e, str(r.get("event"))[:24],
                                                                                       str(a.get("event"))[:24]))
            continue
        if str(a.get("campaign")) != camp:
            _set(e, "bad", A_CAMPAIGN, "signed for campaign %r" % str(a.get("campaign"))[:24])
            continue
        who = signer.get(e)
        if who is None or who != coord or str(a.get("coord_address", "")).lower() != coord:
            _set(e, "bad", A_BAD_SIG, "signature recovers to %s, not %s" % (who, coord))
            continue
        sig_ok[e] = True
        if a.get("prev_hash") is None and not (isinstance(a.get("reset"), dict) and a["reset"].get("reason")):
            _set(e, "bad", A_RESET, "r%d starts a new chain (no prev_hash) without a reset marker" % e)
            continue
        # a prev_hash that does not match r<e-1> blames the PREDECESSOR (its bytes changed): second pass below
        _set(e, "ok")
    unlisted = n_missing - len(missing)
    if unlisted > 0:
        out["ok"] = False
        out["reasons"][A_DROPPED] = out["reasons"].get(A_DROPPED, 0) + unlisted
        out["dropped_unlisted"] = unlisted
    # second pass: successor attestations. r<e> is overwritten when r<e+1> is validly signed and its prev_hash
    # is not r<e>'s hash.
    for e in sorted(sig_ok):
        nxt = evs.get(e + 1)
        if e + 1 not in sig_ok or not isinstance(nxt, dict):
            continue
        ph = nxt[AUDIT_KEY].get("prev_hash")
        if ph is not None and ph != hashes.get(e):
            if out["events"].get(e, {}).get("status") == "ok":
                _set(e, "bad", A_OVERWRITTEN, "r%d hashes %s.. but r%d attests %s.." % (e, str(hashes.get(e))[:12], e + 1,
                                                                                          str(ph)[:12]))
    # a first audited record that links to an unaudited predecessor must link to its bytes too
    if first - 1 in evs and first in sig_ok:
        # review 4 (R4-A2 / C4-F6): the predecessor may be a non-dict (a 2-byte `[]` PUT) or unhashable, so it has
        # no entry in `hashes`. Never index it: a missing hash reads as overwritten / unverifiable, never a raise.
        ph = evs[first][AUDIT_KEY].get("prev_hash")
        hp = hashes.get(first - 1)
        if ph is not None and ph != hp and out["events"].get(first - 1, {}).get("status") == "unaudited_pre":
            seen = (str(hp)[:12] + "..") if hp is not None else "<not a record: %s>" % str(evs.get(first - 1))[:24]
            _set(first - 1, "bad", A_OVERWRITTEN, "pre-audit r%d hashes %s but r%d attests %s.."
                 % (first - 1, seen, first, str(ph)[:12]))
    out["verified"] = sorted(e for e, v in out["events"].items() if v["status"] == "ok")
    return out
