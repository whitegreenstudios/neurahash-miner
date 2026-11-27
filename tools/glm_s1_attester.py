#!/usr/bin/env python
"""STAGE S1 SHADOW-ATTESTATION RUNNER for the live GLM lane (step 4, build item 6; owner 2026-11-26: "build the S1
runner and start it on both PCs", 4060 = "Light attester"). Design: docs/research/STEP4_ROTATING_PROPOSER_DESIGN_
2026-11-26.md sec. 2.1-2.2 and 5.3 (S1 row).

Every validator independently re-checks each coordinator event and PUBLISHES ONE SIGNED WOULD-BE ATTESTATION per
event. Nothing is enforced: the coordinator never reads these names and never waits for them.

MODES
  --mode full   tools/glm_verifier_shadow.ShadowCore (CPU fp32 re-score: gate d, gain, paired z, drift cap, slot /
                model roots, omissions, chain breaks) plus the model-free checks below. Writes the SAME ShadowCore
                rows to --out as glm_verifier_shadow.py did (output continuity: --resume works across the swap).
  --mode light  no model, no torch, no CE re-score (stays well under 2 GB RAM). Checks what needs no model state:
                the coordinator's publish-audit signature + hash chain (neurahash/accepted_audit), the model-root /
                frontier link (a record with no accept must keep the previous model_root, one with an accept must
                change it), chain breaks (glm_verifier_checks.ChainTracker: rewritten/removed records, pointer), and
                omission CANDIDATES (OmissionTracker; signatures recovered model-free, lineage from PUBLISHED slot
                roots). The gain / z decision is "abstain (light)". Payout arithmetic of settled blocks is NOT
                re-checked here: the separate phase-A payout verifier (tools/glm_miner_payout_verify.py) does that.

THE ATTESTATION (one per coordinator event, float-free, signed by the validator's own secp256k1 key with
neura_l1.signing.sign_bytes over glm_round_authority.canonical(body), domain "glm-s1-attest"):
  {"w": "nrh-glm-s1-event-attestation/1", "campaign", "event", "mode", "coord": {decision, gain_pico, margin_pico,
   model_root, veto_reason}, "verdict": agree|disagree|abstain, "reason", "checks": [{"k","v","r"}],
   "measured": {...full mode only...}, "code", "ts", "tolerance", "attester_addr", "sig"}
  verdict = disagree if ANY check disagrees; else agree if the decision check agrees (full) / nothing to re-score
  and every model-free check agrees; else abstain (with the deciding reason). A disagree is a FINDING, never hidden.

PUBLICATION. Relay name  sharddiloco/glm/attest/<campaign16hex>/<event>/<0xsigneraddr>  (relay prefix policy rule,
mode "self", addr_segment 5), written with a store-bound nh-store-put/2 signed PUT (neurahash/store_auth, store id
neurahash-relay-sg1). The published bytes are EXACTLY the line appended to the local JSONL (without the newline).
A failed publish is logged and queued (pending file) and retried every loop; it is never fatal.
RETENTION: after publishing event e the runner deletes ITS OWN name for event e - --retain-events (signed
nh-store-release/1 DELETE); the local JSONL keeps everything. No relay-side sweeper is needed.

Resume-safe (<state>: last attested event + pending queue), flushing logger, <status>.json heartbeat. Keys are
loaded only inside this process and are never printed: --key-file holds either {"key": "<hex>"} (the 5090 roster
co-signer, tools/glm_verifier_node.load_key format) or a bare hex key (the miner wallet, glm_miner_key format).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glm_verifier_shadow as S                                   # noqa: E402  (numpy + stdlib at import)
from neurahash import glm_round_authority as RA                  # noqa: E402
from neurahash import glm_shadow_attester as A                   # noqa: E402

RELAY = S.RELAY
STORE_ID = "neurahash-relay-sg1"
COORD_ADDRESS = "0x2165534ae1c161b6a8523f37f39ce4cf45bf7b86"      # the pinned coordinator settlement address (public)
NAME_ROOT = "sharddiloco/glm/attest"
NAME_RE = re.compile(r"sharddiloco/glm/attest/[0-9a-f]{16}/[0-9]{1,12}/0x[0-9a-f]{40}")
ADDR_SEGMENT = 5
BODY_TAG = "nrh-glm-s1-event-attestation/1"
DOMAIN = A.ATTEST_DOMAIN
MAX_LINE = 16384                                                  # well under the relay rule's max_bytes 65536
REL_ALG, REL_DOMAIN = "nh-store-release/1", "nh-store-release"
AGREE, ABSTAIN, DISAGREE = A.AGREE, A.ABSTAIN, A.DISAGREE


# ============================================================================ identity
def load_account(path):
    """The validator's signing account. Never logs or returns key material beyond the account object."""
    from neura_l1.signing import account_from_key
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    key = raw
    if raw.startswith("{"):
        key = str(json.loads(raw)["key"]).strip()
    return account_from_key(key)


def attest_name(campaign, event, addr):
    name = "%s/%s/%d/%s" % (NAME_ROOT, str(campaign).lower(), int(event), str(addr).lower())
    if not NAME_RE.fullmatch(name):
        raise ValueError("attestation name does not match the relay pattern: %s" % name)
    return name


def name_address(name):
    """The signer address a relay "self" rule reads from `name` (segment 5), lower-case, or None."""
    parts = str(name).split("/")
    return parts[ADDR_SEGMENT].lower() if len(parts) > ADDR_SEGMENT else None


def line_bytes(att):
    """The exact bytes published AND written (plus "\n") to the local JSONL."""
    return json.dumps(att, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sign_body(acct, body):
    return dict(body, sig=RA._sign(acct, body, DOMAIN))


def signer(att):
    """Recovered lower-case signer iff it equals att["attester_addr"], else None."""
    try:
        body = {k: v for k, v in att.items() if k != "sig"}
        s = RA._recover(body, att.get("sig"), DOMAIN)
    except Exception:                                             # noqa: BLE001
        return None
    return s if s and s == str(att.get("attester_addr", "")).lower() else None


def verify_published(name, body_bytes):
    """(ok, why) for one relay object: parses, signature recovers to attester_addr, and the name embeds that address."""
    try:
        att = json.loads(body_bytes.decode("ascii"))
    except Exception as e:                                        # noqa: BLE001
        return False, "unparsable: %s" % type(e).__name__
    s = signer(att)
    if s is None:
        return False, "bad signature"
    if name_address(name) != s:
        return False, "name address %s != signer %s" % (name_address(name), s)
    return True, s


def code_commit():
    c = os.environ.get("NEURAHASH_CODE_COMMIT", "").strip()
    if c:
        return c[:12]
    p = os.path.join(_REPO, "CODE_COMMIT")
    if os.path.isfile(p):
        with open(p, "r", encoding="ascii") as f:
            return f.read().strip()[:12]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, timeout=10,
                             encoding="utf-8", errors="replace", creationflags=getattr(subprocess,
                                                                                     "CREATE_NO_WINDOW", 0))
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()[:12]
    except Exception:                                             # noqa: BLE001
        pass
    with open(os.path.abspath(__file__), "rb") as f:
        return "file:" + hashlib.sha256(f.read()).hexdigest()[:12]


# ============================================================================ relay: tee (GET) + signed publisher
class TeeRelay(S.ReadOnlyRelay):
    """The GET-only relay client, remembering the accepted records it fetched (so the model-free checks use the very
    bytes ShadowCore judged, without a second download)."""

    def __init__(self, *a, keep=8, **kw):
        super().__init__(*a, **kw)
        self.records, self._keep = {}, int(keep)

    def _remember(self, name, rec):
        if isinstance(rec, dict) and str(name).startswith("sharddiloco/glm/accepted/"):
            self.records[name] = rec
            while len(self.records) > self._keep:
                self.records.pop(next(iter(self.records)))
        return rec

    def json_by_name(self, name):
        return self._remember(name, super().json_by_name(name))

    def json_by_name_safe(self, name, max_bytes=None):
        return self._remember(name, super().json_by_name_safe(name, max_bytes))


class RelayPublisher:
    """Signed writes only to the attester's own names. put() -> HTTP status (201 ok); never raises for HTTP."""

    def __init__(self, acct, url=RELAY, store_id=STORE_ID, timeout=60, opener=None):
        from neurahash.glm_committee import assert_wan_url
        self.acct, self.url = acct, assert_wan_url(str(url).rstrip("/"))
        self.store_id, self.timeout = store_id, float(timeout)
        self._open = opener or urllib.request.urlopen
        self._last_ts = 0

    def _call(self, req):
        try:
            with self._open(req, timeout=self.timeout) as r:
                return int(getattr(r, "status", 200) or 200), r.read()[:300]
        except urllib.error.HTTPError as e:
            return int(e.code), e.read()[:300]

    def put(self, name, data):
        from neurahash import store_auth
        sha = hashlib.sha256(data).hexdigest()
        ts = max(int(time.time()), self._last_ts + 1)
        self._last_ts = ts
        h = store_auth.sign_put(self.acct, sha, name=name, ts=ts, store_id=self.store_id)
        h["Content-Type"] = "application/octet-stream"
        return self._call(urllib.request.Request(self.url + "/o/" + sha, data=data, method="PUT", headers=h))

    def delete(self, name):
        from neura_l1 import signing
        ts = int(time.time())
        msg = ("%s\n%s\n%s\n%s\n%s\n%d\n" % (REL_ALG, self.store_id, "delete", name, "", ts)).encode("ascii")
        h = {"X-Sig-Alg": REL_ALG, "X-Sig-Addr": self.acct.address, "X-Sig-Ts": str(ts),
             "X-Sig": signing.sign_bytes(self.acct, msg, domain=REL_DOMAIN), "X-Sig-Store": self.store_id}
        return self._call(urllib.request.Request(self.url + "/n/" + name, method="DELETE", headers=h))

    def get_named(self, name):
        """(status, bytes) of the object currently bound to `name` (readback check)."""
        req = urllib.request.Request(self.url + "/n/" + name.replace("/", "%2F"), method="GET")
        st, meta = self._call(req)
        if st != 200:
            return st, b""
        try:
            sha = json.loads(meta.decode("utf-8"))["sha256"]
        except Exception:                                         # noqa: BLE001
            return 0, b""
        with self._open(urllib.request.Request(self.url + "/o/" + sha, method="GET"), timeout=self.timeout) as r:
            return 200, r.read()


# ============================================================================ model-free checks
class AuditChain:
    """Incremental publish-audit check of r<e> against the pinned coordinator address and r<e-1>."""

    def __init__(self, campaign, coord_address=COORD_ADDRESS):
        self.campaign, self.coord = str(campaign), str(coord_address).lower()
        self.prev = None                                          # (event, record)

    def seed(self, event, rec):
        self.prev = (int(event), rec) if isinstance(rec, dict) else None

    def check(self, e, rec):
        from neurahash import accepted_audit as AA
        prev, self.prev = self.prev, (int(e), rec)
        if not isinstance(rec, dict):
            return ABSTAIN, "record_unavailable"
        a = rec.get(AA.AUDIT_KEY)
        if not isinstance(a, dict):
            return ABSTAIN, "unaudited"
        if int(rec.get("event", -1)) != int(e) or int(a.get("event", -1)) != int(e):
            return DISAGREE, "audit_reordered"
        if str(a.get("campaign")) != self.campaign:
            return DISAGREE, "audit_campaign"
        if AA.signer_of(rec) != self.coord:
            return DISAGREE, "audit_bad_signature"
        if a.get("prev_hash") is None:
            return (AGREE, "audit_signed_reset") if isinstance(a.get("reset"), dict) else (DISAGREE, "audit_reset_unmarked")
        if prev is None or prev[0] != int(e) - 1 or not isinstance(prev[1], dict):
            return AGREE, "audit_signed_link_unseeded"
        if not AA.link_matches(a.get("prev_hash"), prev[1]):
            return DISAGREE, "audit_prev_hash_mismatch"
        return AGREE, "audit_signed_and_linked"


def root_link(prev_rec, rec):
    """Model-root / frontier link: no accept -> model_root unchanged; an accept -> model_root changed."""
    if not isinstance(prev_rec, dict) or not isinstance(rec, dict):
        return ABSTAIN, "root_link_unseeded"
    pr, r = prev_rec.get("model_root"), rec.get("model_root")
    if not r or not pr:
        return ABSTAIN, "root_link_no_root"
    acc = bool(rec.get("accepted"))
    if acc and r == pr:
        return DISAGREE, "root_unchanged_after_accept"
    if not acc and r != pr:
        return DISAGREE, "root_changed_without_accept"
    return AGREE, "root_link_ok"


# ============================================================================ per-event attestation
# audit.check reasons reached only AFTER AA.signer_of(rec) == the pinned coordinator: the record bytes (and so its
# signed judged.gate_decision) are the coordinator's, whatever the hash link says.
_SIG_OK_REASONS = ("audit_signed_reset", "audit_reset_unmarked", "audit_signed_link_unseeded",
                   "audit_prev_hash_mismatch", "audit_signed_and_linked")


def _signature_verified(extra_checks):
    return any(c[0] == "audit" and c[2] in _SIG_OK_REASONS for c in extra_checks)


def _gate_check(row, tol):
    v, r = A.classify_gate(row, tol)
    sdec = (row.get("coordinator") or {}).get("signed_decision")
    if v == DISAGREE and sdec in A.NOT_RECHECKABLE:              # owner 2026-11-26: abstain, never disagree
        return ABSTAIN, A.NOT_RECHECKABLE[sdec]
    if v == DISAGREE:
        reason = str(row.get("disagreement") or "").lower()
        cdec = (row.get("coordinator") or {}).get("decision")
        if cdec == "veto-after-margin" and "held-out" in reason:
            return ABSTAIN, "heldout_tol_not_rechecked"           # design 2.2: held-out tol not re-checked here
    return v, r


def _measured(row):
    c, v = row.get("coordinator") or {}, row.get("verifier") or {}
    z = A._num(v.get("z"))
    zp = A._num(c.get("z"))                                       # the SIGNED z; key only when published
    return dict({"coord_z_micro": int(round(zp * 1e6))} if zp is not None else {},
                **{"coord_gain_pico": A.pico(c.get("gain")), "attester_gain_pico": A.pico(v.get("gain")),
            "margin_pico": A.pico(c.get("margin")), "attester_decision": v.get("decision"),
            "z_micro": None if z is None else int(round(z * 1e6)), "drift": v.get("drift"),
            "attester_base_val_pico": A.pico(v.get("base_val")), "coord_base_val_pico": A.pico(c.get("base_val")),
            "probe_mode": row.get("probe_mode"), "gate_s": row.get("gate_s")})


def build_attestation(acct, campaign, event, rec, rows, extra_checks, mode, tol, commit, late=(), now=None):
    """Signed attestation for one event. `rows` = the ShadowCore rows of this event (full; [] in light),
    `extra_checks` = [(kind, verdict, reason)] from the model-free checks."""
    ev = S.parse_event(rec) if isinstance(rec, dict) else None
    cdec, cgain = S.coordinator_decision(ev) if ev is not None else ("none", None)
    checks, measured = [], None
    sg = A.signed_gate(rec) if isinstance(rec, dict) else None   # coordinator NEURAHASH_SD_SIGN_GATE (else None)
    sg_ok = sg is not None and _signature_verified(extra_checks)
    gate_rows = [r for r in rows if A.row_kind(r) == "gate"]
    for r in rows:
        kind = A.row_kind(r)
        if kind == "gate":
            r = A.with_signed_gate(r, sg if sg_ok else None)    # compare against the SIGNED z only when verified
            checks.append(("gate",) + _gate_check(r, tol))
            if "error" not in r:
                measured = _measured(r)
        else:
            k, v, why = A.classify(r, tol)
            if v is not None:
                checks.append((k, v, why))
        for sub in ("root_check", "ce_check"):
            sc = r.get(sub)
            if isinstance(sc, dict) and sc.get("class") in A._CLASS_MAP:
                checks.append((sub.split("_")[0],) + A._CLASS_MAP[sc["class"]])
    if mode == "light" and ev is not None:
        checks.append(("gate", ABSTAIN, "light_no_rescore"))
    elif mode == "full" and ev is not None and not gate_rows:
        checks.append(("gate", ABSTAIN, "gate_row_missing"))
    if sg is not None and not sg_ok:
        checks.append(("signed_gate", ABSTAIN, "signed_gate_unverified"))
    elif sg_ok:
        checks += A.check_signed_gate(sg, (rec.get("judged") or {}).get("accepted"), tol)
    checks += list(extra_checks)
    dis = [c for c in checks if c[1] == DISAGREE]
    gate = [c for c in checks if c[0] == "gate"]
    if dis:
        verdict, reason = DISAGREE, "%s:%s" % (dis[0][0], dis[0][2])
    elif gate and gate[0][1] != AGREE:
        verdict, reason = ABSTAIN, "%s:%s" % (gate[0][0], gate[0][2])
    elif not gate and any(c[1] == ABSTAIN for c in checks):
        ab = [c for c in checks if c[1] == ABSTAIN][0]
        verdict, reason = ABSTAIN, "%s:%s" % (ab[0], ab[2])
    else:
        verdict, reason = AGREE, ("gate:" + gate[0][2]) if gate else "no_decision_all_checks_agree"
    body = {"w": BODY_TAG, "campaign": str(campaign), "event": int(event), "mode": mode,
            "coord": {"decision": cdec, "gain_pico": A.pico(cgain),
                      "margin_pico": A.pico(ev["margin"]) if ev is not None else None,
                      "model_root": (rec or {}).get("model_root") if isinstance(rec, dict) else None,
                      "veto_reason": ev["veto_reason"] if ev is not None else None,
                      "record": ev["name"] if ev is not None else None,
                      "accepted_rows": len((rec or {}).get("accepted") or []) if isinstance(rec, dict) else None},
            "verdict": verdict, "reason": reason,
            "checks": [{"k": k, "v": v, "r": r} for k, v, r in checks],
            "code": str(commit), "ts": int(time.time() if now is None else now), "tolerance": tol.to_json(),
            "attester_addr": str(acct.address).lower()}
    if sg is not None:                                            # the proposer's signed gate, float-free
        body["coord"]["signed_gate"] = {"decision": sg["decision"], "verified": bool(sg_ok),
                                        "z_micro": None if sg["z"] is None else int(round(sg["z"] * 1e6)),
                                        "z_min_micro": None if sg["z_min"] is None else int(round(sg["z_min"] * 1e6)),
                                        "gain_pico": A.pico(sg["gain"]), "margin_pico": A.pico(sg["margin"]),
                                        "base_val_pico": A.pico(sg["base_val"]),
                                        "merged_val_pico": A.pico(sg["merged_val"])}
    if measured is not None:
        body["measured"] = measured
    if late:
        body["late"] = list(late)[:16]
    att = sign_body(acct, body)
    if len(line_bytes(att)) > MAX_LINE:                           # never exceed the relay rule: drop detail, re-sign
        body.pop("late", None)
        body["checks"] = body["checks"][:24]
        att = sign_body(acct, body)
    return att


# ============================================================================ state + output
class Output:
    """Local JSONL + state (last attested, pending publishes, own published events) + status heartbeat."""

    def __init__(self, out_dir, campaign, addr, log):
        os.makedirs(out_dir, exist_ok=True)
        tag = "%s_%s" % (campaign[:8], addr[2:10])
        self.jsonl = os.path.join(out_dir, "attest_%s.jsonl" % tag)
        self.state_path = os.path.join(out_dir, "attest_%s.state.json" % tag)
        self.status_path = os.path.join(out_dir, "attest_%s.status.json" % tag)
        self.log = log
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        except (OSError, ValueError):
            self.state = {}
        self.state.setdefault("last_attested", None)
        self.state.setdefault("pending", [])                      # [[event, name]] not yet published
        self.state.setdefault("published", [])                    # own events currently on the relay (retention)
        self.state.setdefault("to_delete", [])
        self.counts = dict(self.state.get("counts") or {"agree": 0, "disagree": 0, "abstain": 0, "published": 0,
                                                        "publish_fail": 0, "deleted": 0})

    def save(self):
        self.state["counts"] = self.counts
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, sort_keys=True)
        os.replace(tmp, self.state_path)

    def append(self, att):
        with open(self.jsonl, "ab") as f:
            f.write(line_bytes(att) + b"\n")
            f.flush()
            os.fsync(f.fileno())

    def line_for(self, event):
        """Local line bytes of the newest attestation for `event` (retries publish exactly what was written)."""
        hit = None
        with open(self.jsonl, "rb") as f:
            for ln in f:
                ln = ln.rstrip(b"\r\n")
                if ln and b'"event":%d,' % int(event) in ln:
                    hit = ln
        return hit

    def status(self, extra):
        st = {"updated": time.time(), "last_attested": self.state["last_attested"], "counts": self.counts,
              "pending": len(self.state["pending"]), "published_on_relay": len(self.state["published"])}
        st.update(extra or {})
        tmp = self.status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, sort_keys=True, default=str)
        os.replace(tmp, self.status_path)


class Attester:
    """Glue: build + write + publish + retention. Publishing never raises out of here."""

    def __init__(self, acct, campaign, mode, out, publisher, tol, commit, retain, log):
        self.acct, self.campaign, self.mode, self.out = acct, str(campaign), mode, out
        self.addr = str(acct.address).lower()
        self.pub, self.tol, self.commit, self.retain, self.log = publisher, tol, commit, int(retain), log
        self.late = []

    def attest(self, event, rec, rows, extra_checks):
        att = build_attestation(self.acct, self.campaign, event, rec, rows, extra_checks, self.mode, self.tol,
                                self.commit, late=self.late)
        self.late = []
        self.out.append(att)
        self.out.counts[att["verdict"]] = self.out.counts.get(att["verdict"], 0) + 1
        self.out.state["last_attested"] = int(event)
        self.out.state["pending"].append([int(event), attest_name(self.campaign, event, self.addr)])
        self.out.save()
        lvl = "FINDING " if att["verdict"] == DISAGREE else ""
        self.log("[s1] %sevent %d %s (%s) coord=%s checks=%s" % (lvl, event, att["verdict"], att["reason"],
                                                                 att["coord"]["decision"],
                                                                 ",".join("%s=%s" % (c["k"], c["v"])
                                                                          for c in att["checks"])))
        self.flush_publish()
        return att

    def flush_publish(self):
        if self.pub is None:
            return
        keep = []
        for event, name in list(self.out.state["pending"]):
            try:
                data = self.out.line_for(event)
                if data is None:
                    self.log("[s1] publish: no local line for event %d (dropped from queue)" % event)
                    continue
                st, msg = self.pub.put(name, data)
            except Exception as e:                                # noqa: BLE001 -- never fatal
                st, msg = 0, ("%s: %s" % (type(e).__name__, e)).encode()[:200]
            if st in (200, 201):
                self.out.counts["published"] = self.out.counts.get("published", 0) + 1
                pubd = self.out.state["published"]
                if int(event) not in pubd:
                    pubd.append(int(event))
                old = [x for x in pubd if x <= int(event) - self.retain]
                for x in old:
                    pubd.remove(x)
                    self.out.state["to_delete"].append([x, attest_name(self.campaign, x, self.addr)])
            else:
                self.out.counts["publish_fail"] = self.out.counts.get("publish_fail", 0) + 1
                self.log("[s1] publish event %d failed (%s %s); will retry" % (event, st, msg[:120]))
                keep.append([event, name])
        self.out.state["pending"] = keep
        dkeep = []
        for x, name in list(self.out.state["to_delete"]):
            try:
                st, msg = self.pub.delete(name)
            except Exception as e:                                # noqa: BLE001
                st, msg = 0, str(e).encode()[:100]
            if st in (200, 204, 404):
                self.out.counts["deleted"] = self.out.counts.get("deleted", 0) + (0 if st == 404 else 1)
            else:
                self.log("[s1] retention delete of event %d failed (%s %s); will retry" % (x, st, msg[:80]))
                dkeep.append([x, name])
        self.out.state["to_delete"] = dkeep[-512:]
        self.out.save()


# ============================================================================ full mode
class EventCollector:
    """Passed to ShadowCore as `attester=`: buffers its rows per event (ShadowCore hands it a copy)."""

    def __init__(self):
        self.rows = {}

    def observe(self, row):
        e = row.get("event", row.get("at_event"))
        try:
            e = int(e)
        except (TypeError, ValueError):
            return
        self.rows.setdefault(e, []).append(row)


def full_flush(core, collector, relay, att, audit, start_event, prev_rec):
    """Attest every buffered event < core.next_event. Returns the newest record seen (for the root link)."""
    done = sorted(e for e in collector.rows if e < core.next_event)
    for e in done:
        rows = collector.rows.pop(e)
        last = att.out.state["last_attested"]
        if e < start_event:
            continue
        if last is not None and e <= last:                        # late rows for an attested event: carry
            for r in rows:
                k, v, why = A.classify(r, att.tol)
                if v is not None:
                    att.late.append({"at": e, "k": k, "v": v, "r": why})
            continue
        rec = relay.records.get(S.accepted_name(core.campaign, e))
        extra = [("audit",) + audit.check(e, rec), ("root_link",) + root_link(prev_rec, rec)]
        att.attest(e, rec, rows, extra)
        prev_rec = rec
    return prev_rec


def build_full_core(a, relay, collector, log):
    """glm_verifier_shadow.main's construction, unchanged numbers (same flags), with the collector attached."""
    import numpy as np
    guard = S.MemGuard(a.mem_floor_gib, log=log).start()
    with open(a.probe_pool, "rb") as f:
        pool_sha = hashlib.sha256(f.read()).hexdigest()
    if a.probe_pool_sha256 and pool_sha != a.probe_pool_sha256:
        raise SystemExit("probe pool sha256 %s.. != pinned" % pool_sha[:12])
    pool = np.load(a.probe_pool)
    probe = S.ProbeSource(pool, 0, a.probe_size, "full_pool")
    ev, host, args, N, G = S.build_real(a, log=log)
    log("[s1] after build: rss %.2f GiB, min available %.2f GiB" % (guard.peak_rss_gib(), guard.min_avail_gib()))
    import sharddiloco_glm_coordinator as C
    drift = None
    if a.drift_cap and a.drift_cap > 0:
        import glm_expert_drift as GD
        why = C._drift_cap_selfcheck(GD, a.drift_cap)
        if why is not None:
            raise SystemExit("drift self-check: %s" % why)
        drift = C.DriftCap(a.drift_cap, a.drift_base_dir, GD)
    why = C._paired_z_selfcheck(a.paired_z)
    if why is not None:
        raise SystemExit("paired-z self-check: %s" % why)
    import glm_verifier_checks as K
    from neurahash import diloco_merge as dm
    extra = dict(check_root=a.check_root, N=N, omission_window=a.omission_window, chain_check=a.chain_check)
    if a.omission_window > 0:
        extra.update(sig_ok=K.make_sig_ok(C, N), lineage_ok=K.make_lineage_ok(C),
                     omission_max_stale=dm.max_staleness(), omission_delta_ok=K.make_delta_ok(C, G, relay, host))
    core = S.ShadowCore(ev, relay, probe, a.campaign, a.out, outer=a.outer, margin=a.margin, paired_z=a.paired_z,
                        drift=drift, slot_root=N.slot_root, rescore_from=a.rescore_from, log=log, memguard=guard,
                        attester=collector, **extra)
    return core, guard


def run_full(a, att, log, stop):
    relay = TeeRelay(a.relay)
    collector = EventCollector()
    a.rescore_from = S.resume_point(a.out, a.rescore_from)
    core, guard = build_full_core(a, relay, collector, log)
    start = a.rescore_from if att.out.state["last_attested"] is None else max(a.rescore_from,
                                                                              att.out.state["last_attested"] + 1)
    log("[s1] full mode: replay 1..%d, re-score + attest from %d" % (a.rescore_from - 1, start))
    t0 = time.time()
    if a.rescore_from > 1:
        core.step(upto=a.rescore_from - 1)
    collector.rows.clear()
    log("[s1] replayed 1..%d in %.0fs (rss peak %.2f GiB)" % (a.rescore_from - 1, time.time() - t0,
                                                               guard.peak_rss_gib()))
    audit = AuditChain(a.campaign, a.coord_address)
    prev = relay.json_by_name_safe(S.accepted_name(a.campaign, core.next_event - 1)) if core.next_event > 1 else None
    audit.seed(core.next_event - 1, prev)
    while True:
        before = core.next_event
        try:
            core.step(upto=core.next_event)                       # one event at a time: attest as soon as judged
        except Exception as ex:                                   # noqa: BLE001 -- logged, retried next pass
            log("[s1] core.step error %s: %s" % (type(ex).__name__, str(ex)[:300]))
        prev = full_flush(core, collector, relay, att, audit, start, prev)
        att.flush_publish()
        st = core.status()
        att.out.status({"mode": "full", "addr": att.addr, "core": st, "rss_gib": round(guard.peak_rss_gib(), 2),
                        "min_avail_gib": guard.min_avail_gib()})
        if stop.is_set() and core.next_event == before:
            break
        if core.next_event == before:
            stop.wait(a.poll_s)
    return 0


# ============================================================================ light mode
class LightCore:
    """Model-free walk over the published records (no torch is ever imported on this path)."""

    def __init__(self, relay, campaign, coord_address, omission_window, log):
        import glm_verifier_checks as K
        self.K, self.relay, self.campaign, self.log = K, relay, str(campaign), log
        self.audit = AuditChain(campaign, coord_address)
        self.chain = K.ChainTracker(self.campaign)
        self.next_event, self.prev_rec, self.late = None, None, []
        self.slot_hist = {}
        self.omit = None
        if omission_window > 0:
            import sharddiloco_glm_coordinator as C                # numpy-only; checked: no torch
            import sharddiloco_glm_contributor as N
            lin = K.make_lineage_ok(C)
            self.omit = K.OmissionTracker(self.campaign, omission_window, relay.json_by_name_safe,
                                          K.make_sig_ok(C, N),
                                          lambda body, coord, be, cur: lin(self.slot_hist, coord, be, cur,
                                                                           body.get("base_slot_root")))

    def start(self, start_event):
        self.relay.refresh()
        names = self.relay.names()
        pub = S.published_events(names, self.campaign)
        head = pub[-1] if pub else 0
        self.next_event = int(start_event if start_event is not None else max(1, head))
        if self.next_event > 1:
            self.prev_rec = self.relay.json_by_name_safe(S.accepted_name(self.campaign, self.next_event - 1))
            self.audit.seed(self.next_event - 1, self.prev_rec)
        if self.omit is not None:                                 # only contributions that appear after start
            self.omit.observe(names, head)
            self.omit.judged.update(self.omit.first_seen)
        return self.next_event

    def _note_roots(self, e, rec):
        for k, root in (rec.get("slot_roots") or {}).items():
            try:
                L, E = (int(x) for x in str(k).split("_"))
            except ValueError:
                continue
            h = self.slot_hist.setdefault((L, E), [])
            if not h or h[-1][1] != root:
                h.append((int(e), root))

    def step(self, attest_fn):
        """Attest every new published event: attest_fn(event, record|None, extra_checks, late_rows). -> count."""
        K = self.K
        self.relay.refresh()
        names = self.relay.names()
        pub = S.published_events(names, self.campaign)
        head = pub[-1] if pub else 0
        for r in self.chain.observe_manifest(names, self.next_event - 1, lambda ev: S.accepted_name(self.campaign, ev)):
            self.late.append(r)
        if self.omit is not None:
            self.omit.observe(names, head)
        n = 0
        for e in [x for x in pub if x >= self.next_event]:
            if e != self.next_event:
                self.log("[s1] light: event gap, expected r%d, next is r%d (waiting)" % (self.next_event, e))
                break
            nm = S.accepted_name(self.campaign, e)
            v = names.get(nm)
            sha = str(v["sha256"] if isinstance(v, dict) else v)
            try:
                rec = self.relay.json_by_name_safe(nm)
                bad = K.record_shape_error(rec)
            except Exception as ex:                               # noqa: BLE001
                rec, bad = None, "unreadable: %s" % type(ex).__name__
            rows = []
            if bad is None:
                rows += self.chain.observe_record(e, rec, sha)
                self._note_roots(e, rec)
                if self.omit is not None:
                    self.omit.note_judged(rec)
                    rows += self.omit.evaluate(e)
            extra = [("audit",) + self.audit.check(e, rec if bad is None else None),
                     ("root_link",) + root_link(self.prev_rec, rec if bad is None else None)]
            if bad is not None:
                extra.append(("record", ABSTAIN, "record_unusable"))
            for r in rows:                                        # light: omissions are candidates only
                if r.get("class") == "omitted":
                    extra.append(("omission", ABSTAIN, "omission_candidate_light"))
                else:
                    k, vv, why = A.classify(r, A.Tolerance())
                    if vv is not None:
                        extra.append((k, vv, why))
            late, self.late = self.late, []
            attest_fn(e, rec if bad is None else None, extra, late)
            self.prev_rec = rec if bad is None else self.prev_rec
            self.next_event = e + 1
            n += 1
        return n


def run_light(a, att, log, stop):
    relay = TeeRelay(a.relay)
    lc = LightCore(relay, a.campaign, a.coord_address, a.omission_window, log)
    last = att.out.state["last_attested"]
    first = lc.start(last + 1 if last is not None else a.start_event)
    log("[s1] light mode: attesting from event %d (no model; torch loaded=%s)" % (first, "torch" in sys.modules))

    def attest_fn(e, rec, extra, late):
        for r in late:
            k, v, why = A.classify(r, att.tol)
            if v is not None:
                att.late.append({"at": r.get("event", r.get("at_event")), "k": k, "v": v, "r": why})
        att.attest(e, rec, [], extra)
    while True:
        try:
            n = lc.step(attest_fn)
        except Exception as ex:                                   # noqa: BLE001 -- logged, retried next pass
            log("[s1] light step error %s: %s" % (type(ex).__name__, str(ex)[:300]))
            n = 0
        att.flush_publish()
        att.out.status({"mode": "light", "addr": att.addr, "next_event": lc.next_event, "rss_gib": _rss_gib(),
                        "torch_loaded": "torch" in sys.modules})
        if stop.is_set():
            break
        if n == 0:
            stop.wait(a.poll_s)
    return 0


def _rss_gib():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2 ** 30, 3)
    except Exception:                                             # noqa: BLE001
        return None


# ============================================================================ main
def make_logger(path):
    lock = threading.Lock()
    f = open(path, "a", encoding="utf-8", buffering=1) if path else None

    def log(m):
        line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), m)
        with lock:
            print(line, flush=True)
            if f is not None:
                f.write(line + "\n")
                f.flush()
    return log


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="S1 shadow attester (publishes signed would-be attestations; "
                                             "enforces nothing).")
    ap.add_argument("--mode", choices=("full", "light"), required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--key-file", required=True, help="validator key: {\"key\": hex} JSON or a bare hex file")
    ap.add_argument("--attest-dir", required=True, help="local JSONL / state / status / log")
    ap.add_argument("--relay", default=RELAY)
    ap.add_argument("--store-id", default=STORE_ID)
    ap.add_argument("--coord-address", default=COORD_ADDRESS)
    ap.add_argument("--no-publish", action="store_true", help="local JSONL only")
    ap.add_argument("--retain-events", type=int, default=500, help="delete own relay names older than this")
    ap.add_argument("--start-event", type=int, default=None, help="light, first run: first event (default newest)")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--omission-window", type=int, default=0)
    # ---- full mode: the glm_verifier_shadow flags (same defaults) ----
    ap.add_argument("--out", default=None, help="full: ShadowCore rows JSONL (the verifier's --out, for continuity)")
    ap.add_argument("--rescore-from", type=int, default=1)
    ap.add_argument("--probe-full-pool", action="store_true")
    ap.add_argument("--probe-size", type=int, default=256)
    ap.add_argument("--probe-pool", default="D:/glm_daily/coord/ids_daily_probe.npy")
    ap.add_argument("--probe-pool-sha256", default="28bba0c05cdc8a556f656678b560b511dc7acd3c06fa74832c8193dbe67b3704")
    ap.add_argument("--outer", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=1e-5)
    ap.add_argument("--paired-z", type=float, default=3.0)
    ap.add_argument("--drift-cap", type=float, default=0.02)
    ap.add_argument("--drift-base-dir", default="E:/_glm_layer_experts")
    ap.add_argument("--shard-dir", default="D:/hf_models/GLM-4.7-Flash-bf16_shards_100mb")
    ap.add_argument("--config-dir", default="D:/hf_models/GLM-4.7-Flash-bf16")
    ap.add_argument("--coord-data-dir", default="D:/glm_daily/coord")
    ap.add_argument("--domains", default="daily,daily")
    ap.add_argument("--pieces", default="0-23")
    ap.add_argument("--max-active-slots", type=int, default=120)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--mem-floor-gib", type=float, default=8.0)
    ap.add_argument("--lazy-fp32", action="store_true")
    ap.add_argument("--layer-major", action="store_true")
    ap.add_argument("--check-root", action="store_true")
    ap.add_argument("--chain-check", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "full":
        if not a.out or not a.probe_full_pool:
            ap.error("--mode full needs --out and --probe-full-pool (the live verifier's configuration)")
    if not re.fullmatch(r"[0-9a-f]{16}", a.campaign):
        ap.error("--campaign must be 16 lower-case hex")
    return a


PUBLIC_LIGHT_ONLY_MSG = ("full mode re-scores contributions and needs the private coordinator code; the public miner "
                         "ships light mode only (run --mode light, see docs/RUN_A_VALIDATOR.md)")
OMISSION_NEEDS_COORD_MSG = ("--omission-window > 0 recovers contribution signatures with the private coordinator code; "
                            "the public miner ships light mode without it (leave --omission-window at 0)")


def coordinator_available():
    """True when tools/sharddiloco_glm_coordinator.py is importable (the private repo). The public miner does not
    ship it, so full mode and light-mode omission candidates refuse early instead of dying on an ImportError."""
    import importlib.util
    return importlib.util.find_spec("sharddiloco_glm_coordinator") is not None


def main(argv=None):
    a = parse_args(argv)
    if a.mode == "full" and not coordinator_available():
        print("[s1] REFUSING full mode: " + PUBLIC_LIGHT_ONLY_MSG, flush=True)
        return 2
    if a.mode == "light" and a.omission_window > 0 and not coordinator_available():
        print("[s1] REFUSING: " + OMISSION_NEEDS_COORD_MSG, flush=True)
        return 2
    if a.mode == "full" and os.environ.get("CUDA_VISIBLE_DEVICES", None) != "-1":
        print("[s1] REFUSING full mode: set CUDA_VISIBLE_DEVICES=-1 (the GPU belongs to the miners)", flush=True)
        return 2
    os.makedirs(a.attest_dir, exist_ok=True)
    acct = load_account(a.key_file)
    addr = str(acct.address).lower()
    log = make_logger(os.path.join(a.attest_dir, "attest_%s_%s.log" % (a.campaign[:8], addr[2:10])))
    out = Output(a.attest_dir, a.campaign, addr, log)
    pub = None if a.no_publish else RelayPublisher(acct, a.relay, a.store_id)
    commit = code_commit()
    att = Attester(acct, a.campaign, a.mode, out, pub, A.Tolerance.from_env(), commit, a.retain_events, log)
    log("[s1] start mode=%s attester=%s campaign=%s code=%s publish=%s retain=%d last_attested=%s"
        % (a.mode, addr, a.campaign, commit, pub is not None, a.retain_events, out.state["last_attested"]))
    stop = threading.Event()
    if not a.follow:
        stop.set()                                                # one pass (tests / offline)
    try:
        return run_full(a, att, log, stop) if a.mode == "full" else run_light(a, att, log, stop)
    finally:
        out.save()


if __name__ == "__main__":
    sys.exit(main())
