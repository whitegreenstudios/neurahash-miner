"""
signed_put -- per-prefix write POLICY for the relay + the coordinator-SIGNED POINTER.

Design: docs/research/RELAY_SIGNED_WRITES_DESIGN_2026-11-26.md. DEFAULT-OFF: nothing imports this
module unless the relay is started with CONTENT_PREFIX_POLICY (tools/content_store.py) or a client
opts into signing / verifying the pointer. Pure: no network, no clock unless one is passed, no I/O.

WHY
---
The relay (tools/content_store.py) already HAS signed PUT (neurahash/store_auth.py, nh-store-put/1|2):
a write signed by a secp256k1 key is attributed to the RECOVERED address, and a name the address
creates is OWNED by it. What it lacks is a rule saying WHICH address may write WHICH name. Without
that, anyone holding the shared write token (or a free key) can still create the names honest parties
will write later -- co-sign names, accepted records, heldout-epoch objects, the pointer -- because
"first writer owns it" rewards whoever is first. This module adds:

  1. A PrefixPolicy: longest-prefix rules over relay names, each one of
       operator : only a pinned operator (coordinator) address may write; token writes are refused;
       self     : the name must EMBED the signer's own address at a fixed path segment
                  (e.g. sharddiloco/glm/cosign/<chain>/<index>/<addr>[~suffix]); token writes refused;
       signed   : any valid signer (normal first-claim ownership), token writes refused.
     A name no rule matches is left to the relay's legacy rules unchanged (migration path).
     A permitted operator / self write may TAKE OVER an existing entry (the policy has proven the
     writer is the rightful one), which is how today's ownerless squats are reclaimed.

  2. A signed pointer: the coordinator adds a `psig` object to the pointer dict it already publishes.
     Old readers ignore the extra key; new readers verify it against the PINNED coordinator address and
     keep a high-water mark (PointerGuard) so a replayed older pointer is refused. This is checked
     CLIENT-side, so a compromised relay still cannot forge the pointer; the relay-side policy is the
     denial-of-service layer (it stops the overwrite from happening at all).
"""

import hashlib
import json
import re

ALLOWED_MODES = ("operator", "self", "signed")
_ADDR_RE = re.compile(r"^0x[0-9a-f]{40}$")
POINTER_ALG = "nh-glm-pointer/1"
_HEX_RE = re.compile(r"^[0-9a-f]*$")
_FIELD_RE = re.compile(r"^[\x20-\x7e]{0,256}$")          # printable ASCII, no newline: one message field


class PolicyError(ValueError):
    """A policy document that must not be served (fail closed at startup)."""


class PrefixRule(object):
    __slots__ = ("prefix", "mode", "addr_segment", "pattern", "max_bytes", "max_names")

    def __init__(self, prefix, mode, addr_segment=None, pattern=None, max_bytes=0, max_names=0):
        if not isinstance(prefix, str) or not prefix or not all(32 <= ord(c) < 127 for c in prefix):
            raise PolicyError("rule prefix must be non-empty printable ASCII: %r" % (prefix,))
        if mode not in ALLOWED_MODES:
            raise PolicyError("rule %r: mode must be one of %s" % (prefix, ALLOWED_MODES))
        if mode == "self":
            if not isinstance(addr_segment, int) or isinstance(addr_segment, bool) or addr_segment < 0:
                raise PolicyError("rule %r: mode self needs addr_segment >= 0" % prefix)
            if addr_segment < len(prefix.rstrip("/").split("/")):
                raise PolicyError("rule %r: addr_segment %d falls inside the fixed prefix" % (prefix, addr_segment))
        elif addr_segment is not None:
            raise PolicyError("rule %r: addr_segment only applies to mode self" % prefix)
        self.prefix, self.mode, self.addr_segment = prefix, mode, addr_segment
        self.pattern = re.compile(pattern) if pattern else None
        self.max_bytes, self.max_names = int(max_bytes or 0), int(max_names or 0)
        if self.max_bytes < 0 or self.max_names < 0:
            raise PolicyError("rule %r: limits must be >= 0" % prefix)
        if mode != "operator" and self.max_names <= 0:
            # free keys can create names under a self/signed prefix: it needs its own bound (review F4)
            raise PolicyError("rule %r: mode %s needs max_names > 0" % (prefix, mode))


class PrefixPolicy(object):
    """Parsed CONTENT_PREFIX_POLICY. `operators` are lowercase 0x addresses (the pinned coordinator
    key(s); more than one only for a planned key rotation)."""

    def __init__(self, operators, rules):
        ops = []
        for a in operators or ():
            a = str(a).strip().lower()
            if not _ADDR_RE.match(a):
                raise PolicyError("operator address malformed: %r" % (a,))
            ops.append(a)
        self.operators = frozenset(ops)
        self.rules = sorted(rules, key=lambda r: len(r.prefix), reverse=True)   # longest prefix first
        seen = set()
        for r in self.rules:
            if r.prefix in seen:
                raise PolicyError("duplicate rule prefix %r" % r.prefix)
            seen.add(r.prefix)
            if r.mode == "operator" and not self.operators:
                raise PolicyError("rule %r is operator-only but no operator address is pinned" % r.prefix)

    def rule_for(self, name):
        if not name:
            return None
        for r in self.rules:
            if name.startswith(r.prefix):
                return r
        return None

    def is_operator(self, writer):
        return isinstance(writer, str) and writer.lower() in self.operators


def parse_policy(text):
    """JSON document -> PrefixPolicy. Shape:
        {"operators": ["0x..."],
         "rules": [{"prefix": "sharddiloco/glm/accepted/", "mode": "operator", "max_bytes": 1048576},
                   {"prefix": "sharddiloco/glm/cosign/", "mode": "self", "addr_segment": 5,
                    "max_bytes": 65536, "max_names": 200000}]}
    Unknown keys are an error (a typo must not silently drop a protection)."""
    try:
        doc = json.loads(text)
    except Exception as e:
        raise PolicyError("policy is not JSON: %s" % e.__class__.__name__)
    if not isinstance(doc, dict) or set(doc) - {"operators", "rules", "comment"}:
        raise PolicyError("policy must be an object with keys operators, rules[, comment]")
    rules = []
    for raw in doc.get("rules") or []:
        if not isinstance(raw, dict):
            raise PolicyError("each rule must be an object")
        extra = set(raw) - {"prefix", "mode", "addr_segment", "pattern", "max_bytes", "max_names", "comment"}
        if extra:
            raise PolicyError("rule has unknown keys: %s" % sorted(extra))
        rules.append(PrefixRule(raw.get("prefix"), raw.get("mode"), raw.get("addr_segment"),
                                raw.get("pattern"), raw.get("max_bytes", 0), raw.get("max_names", 0)))
    if not rules:
        raise PolicyError("policy has no rules")
    return PrefixPolicy(doc.get("operators") or [], rules)


def embedded_address(name, segment):
    """The address a `self` name embeds at path `segment`, lowercase, or None. A trailing '~<suffix>'
    on that segment (payout_select.cosign_alt_name) is not part of the address."""
    parts = str(name).split("/")
    if segment >= len(parts):
        return None
    seg = parts[segment].split("~", 1)[0]
    return seg if _ADDR_RE.match(seg) else None


def decide(policy, name, writer, signed, nbytes=None):
    """The policy verdict for one named PUT, BEFORE its body is read.

    `writer` is the relay's recovered signer address when `signed` is True; for a token / anonymous
    writer `signed` is False (whatever tier it is). Returns
        None                         -- no rule matches: legacy rules decide (unchanged);
        (True, rule, takeover)       -- permitted; `takeover` lets the write replace an entry owned by
                                        someone else (or ownerless) because the policy proved the
                                        writer is the rightful one;
        (False, status, reason)      -- refused with HTTP `status`."""
    rule = policy.rule_for(name)
    if rule is None:
        return None
    if not signed:
        return False, 403, "signature required for %s" % rule.prefix
    w = str(writer).lower()
    if rule.pattern is not None and not rule.pattern.fullmatch(name):
        return False, 403, "name does not match the pattern for %s" % rule.prefix
    if rule.max_bytes and nbytes is not None and nbytes > rule.max_bytes:
        return False, 413, "body too large for %s: %d > %d" % (rule.prefix, nbytes, rule.max_bytes)
    if rule.mode == "operator":
        if w not in policy.operators:
            return False, 403, "%s is operator-only" % rule.prefix
        return True, rule, True
    if rule.mode == "self":
        emb = embedded_address(name, rule.addr_segment)
        if emb is None:
            return False, 403, "name must embed the signer address at segment %d" % rule.addr_segment
        if emb != w:
            return False, 403, "name embeds %s, signer is %s" % (emb, w)
        return True, rule, True
    return True, rule, False                            # mode "signed": normal first-claim ownership


# --------------------------------------------------------------------------- signed pointer
def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _field(v):
    s = "" if v is None else str(v)
    if not _FIELD_RE.match(s):
        raise ValueError("pointer field not printable ASCII <= 256: %r" % (s[:40],))
    return s


def pointer_campaign(ptr):
    """The campaign a pointer names: `campaign`, else the live GLM lane's key `campaign_id`
    (tools/sharddiloco_glm_coordinator._pointer_extra writes N.CAMPAIGN_POINTER_KEY = "campaign_id").
    Client-side addition 2026-11-26: without it every real GLM pointer signs/verifies with an empty
    campaign and PointerGuard treats a campaign switch (event back to 1) as a rollback."""
    c = ptr.get("campaign")
    return ptr.get("campaign_id") if c is None else c


def pointer_body_sha(ptr):
    """sha256 of the canonical JSON of the pointer WITHOUT its `psig`: binds every field, including
    the ones readers act on today (rounds, domains, next_up, done)."""
    body = {k: v for k, v in ptr.items() if k != "psig"}
    return hashlib.sha256(_canon(body)).hexdigest()


def pointer_signing_bytes(ptr, chain_tip, store_id=""):
    """The exact bytes the coordinator signs:
        nh-glm-pointer/1\\n<store-id>\\n<campaign>\\n<event>\\n<model_root>\\n<chain_tip>\\n<body sha256>\\n
    The explicit fields let a reader see what it is trusting; the body hash covers the rest."""
    event = ptr.get("event", ptr.get("round"))
    if not isinstance(event, int) or isinstance(event, bool) or event < 0:
        raise ValueError("pointer event must be an int >= 0")
    pa = ptr.get("published_at")
    if not isinstance(pa, int) or isinstance(pa, bool) or pa < 0:
        raise ValueError("pointer published_at must be an int >= 0 (epoch ms): the anti-rollback order needs it")
    tip = _field(chain_tip).lower()
    if not _HEX_RE.match(tip):
        raise ValueError("chain_tip must be hex or empty")
    return ("%s\n%s\n%s\n%d\n%s\n%s\n%s\n" % (
        POINTER_ALG, _field(store_id), _field(pointer_campaign(ptr)), event, _field(ptr.get("model_root")),
        tip, pointer_body_sha(ptr))).encode("ascii")


def sign_pointer(account, ptr, chain_tip="", store_id=""):
    """COORDINATOR side: return a COPY of `ptr` with a `psig` object added. `account` is a
    neura_l1.signing account (the coordinator key). Readers that do not know psig ignore it."""
    from neura_l1 import signing
    body = {k: v for k, v in ptr.items() if k != "psig"}
    data = pointer_signing_bytes(body, chain_tip, store_id)
    out = dict(body)
    out["psig"] = {"alg": POINTER_ALG, "addr": account.address.lower(), "chain_tip": str(chain_tip).lower(),
                   "store": str(store_id), "sig": signing.sign_bytes(account, data)}
    return out


def verify_pointer(ptr, expected_addrs, store_id=""):
    """READER side: (True, info) or (False, reason). `expected_addrs` = the pinned coordinator
    address(es). The signer is RECOVERED, never taken from psig.addr; the store id is the READER's
    configured one (a pointer signed for another relay does not verify here)."""
    if not isinstance(ptr, dict):
        return False, "pointer is not an object"
    ps = ptr.get("psig")
    if not isinstance(ps, dict):
        return False, "unsigned pointer"
    if ps.get("alg") != POINTER_ALG:
        return False, "unsupported pointer alg %r" % (ps.get("alg"),)
    pinned = {str(a).lower() for a in (expected_addrs or ())}
    if not pinned:
        return False, "no pinned coordinator address"
    if str(ps.get("store", "")) != str(store_id):
        return False, "pointer signed for store %r, not %r" % (ps.get("store"), store_id)
    try:
        data = pointer_signing_bytes(ptr, ps.get("chain_tip", ""), store_id)
    except (ValueError, TypeError) as e:
        return False, "malformed pointer: %s" % e
    from neura_l1 import signing
    try:
        rec = signing.recover_bytes(data, ps.get("sig") or "").lower()
    except Exception as e:
        return False, "signature recovery failed: %s" % e.__class__.__name__
    if rec not in pinned:
        return False, "pointer signed by %s, not the pinned coordinator" % rec
    event = ptr.get("event", ptr.get("round"))
    return True, {"signer": rec, "campaign": pointer_campaign(ptr), "event": event,
                  "model_root": ptr.get("model_root"), "chain_tip": ps.get("chain_tip", ""),
                  "published_at": ptr.get("published_at")}


class PointerGuard(object):
    """Client-side anti-rollback for VERIFIED pointers. A replayed older pointer is still validly
    signed, so signature checks alone cannot refuse it; this keeps the newest accepted one.

    Order: (published_at, event) within a campaign; a pointer for a different campaign is accepted
    only with a strictly newer published_at than the last accepted pointer. A refused pointer leaves
    the last good one in place (`current`), so an attacker who overwrites or replays the pointer
    cannot make a running reader lose the live campaign (R4-A1)."""

    def __init__(self, expected_addrs, store_id="", max_age_ms=None, floor=None):
        """`max_age_ms`: refuse a pointer published longer ago than this (needs `now_ms` in offer); it
        bounds what a COLD reader (fresh install, lost state) can be fed. `floor`: a previously accepted
        info dict to resume from (persist `current` across restarts)."""
        self.expected, self.store_id = tuple(expected_addrs), store_id
        self.max_age_ms = max_age_ms
        self.current = dict(floor) if floor else None   # info dict of the newest accepted pointer

    def offer(self, ptr, now_ms=None):
        ok, info = verify_pointer(ptr, self.expected, self.store_id)
        if not ok:
            return False, info
        cur = self.current
        pa = info["published_at"]                       # an int: pointer_signing_bytes requires it
        if self.max_age_ms is not None:
            if now_ms is None:
                return False, "max_age_ms set but no now_ms given"
            if now_ms - pa > self.max_age_ms:
                return False, "stale pointer: published %d ms ago" % (now_ms - pa)
        if cur is not None:
            cpa = cur["published_at"]
            if info["campaign"] == cur["campaign"]:
                if (pa, info["event"]) < (cpa, cur["event"]):
                    return False, "rollback: older pointer for campaign %s" % info["campaign"]
                if info["event"] < cur["event"]:
                    return False, "rollback: event %d < %d" % (info["event"], cur["event"])
            elif pa <= cpa:
                return False, "campaign switch with a non-newer published_at"
        self.current = dict(info)
        return True, self.current
