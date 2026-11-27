"""
neurahash/glm_shadow_attester.py -- STAGE S1 SHADOW ATTESTER for the GLM lane (step 4, build-plan item 6).
Design: docs/research/STEP4_ROTATING_PROPOSER_DESIGN_2026-11-26.md sec. 2.2 (tolerance rule) and 5.3 (S1).

DEFAULT OFF. Nothing on the live path imports this module. It is constructed only by from_env() under
NEURAHASH_STEP4_S1_ATTEST=1 (and publishes to the relay only with NEURAHASH_STEP4_S1_PUBLISH=1 AND an injected
publisher); tools/glm_verifier_shadow.ShadowCore calls it only when given attester=... explicitly. With the flags
unset ShadowCore's decisions and the bytes it writes are unchanged (tests/test_glm_shadow_attester.py proves it).

WHAT IT DOES. ShadowCore (the step-2 verifier shadow, `--check-root --chain-check` in the live run) re-scores each
coordinator decision on the CPU and emits one row per check. The attester turns every row into a SIGNED verdict
on the coordinator's decision -- the would-be attestation of a stage-4 attester in "attest-only" mode:

  agree     the re-check reproduces the decision AND the numbers lie within tolerance. For a gate row the
            attester also signs a stage-4 PREPARE (glm_round_authority.sign_prepare, attest-only: it never
            proposes) over the would-be step digest, so the same objects can feed a finality count in stage S2.
  abstain   not attributable: the decision is within tolerance of a threshold (ambiguous), the numbers are
            outside tolerance although the decision agrees, the probe rows differ (independent probe), the
            check was skipped / errored, or the coordinator vetoed for a reason not re-checkable from public
            records (canary latch, held-out tol). An abstain is NEVER a fault.
  disagree  the decision flips FAR from every threshold (or a slot/model root differs -- roots are exact),
            or a published check (CE, omission, chain) failed beyond its own tolerance.

THE TOLERANCE RULE (design 2.2, rule 2, adapted to S1 where the live coordinator is the proposer):
  gain_p = coordinator's published gain (base_val - merged_val), gain_a = this attester's, margin = published.
  num_ok = |gain_a - gain_p| <= tau_gate.
  near   = min(|gain_p - margin|, |gain_a - margin|) <= tau_gate, or (z-veto involved and |z_a - z_min| <= tau_z).
  decisions equal  -> agree if num_ok, else abstain("gain_outside_tau").
  decisions differ -> abstain("near_threshold") if near, else disagree.
  So a decision near a threshold is classified abstain/ambiguous, never disagree, and bit-exact CPU agreement is
  NOT assumed.
  SIGNED z (2026-11-26, coordinator NEURAHASH_SD_SIGN_GATE=1, judged.gate_decision v1, used only when the record's
  publish-audit signature recovers to the pinned coordinator): z_p = the proposer's signed z. Decisions equal ->
  additionally abstain("z_outside_tau") if |z_a - z_p| > tau_z. Decisions differ -> near also when
  |z_p - z_min| <= tau_z. Held-out tol / canary / judge vetoes abstain, never disagree (owner 2026-11-26).
  check_signed_gate() checks the proposer's own numbers for internal consistency (light mode).

DEFAULTS, derived from a measurement (E:/rsft_runs/cpu_determinism_1126/runs_summary.json, 2026-11-26; partial,
the 6-thread run was still finishing; the same gate replayed on the 5090-host CPU vs the 4060-host coordinator,
MKL_CBWR=AVX2 did NOT make it bit-exact):
  |d gain|      max 4.15e-7 (n=2 vs coordinator; 3.0e-7 vs the live 20-thread verifier)
  |d raw CE|    max 3.15e-6 (base_val / merged_val individually; mostly cancels in the gain)
  |d z|         max 0.0034
  held-out CE   1.3e-7 vs published (n=1)
  decisions     agreed in every measured case.
  TAU_GATE_DEFAULT = 1e-6 : 2.4x the max measured gain diff, and margin/10 at the live margin 1e-5 (design rule
                            "tau_gate <= margin/10"). If later measurements exceed it, the excess shows up as
                            abstain("gain_outside_tau"), never as a false disagree.
  TAU_Z_DEFAULT    = 0.01  : ~3x the max measured |dz|.
  TAU_CE_DEFAULT   = 2e-5  : the pinned tools/glm_verifier_checks.CE_TOL_DEFAULT (~6x raw-CE diff, ~150x CE diff);
                            the CE check itself classifies ce_agree / ce_disagree with it.
  Env overrides: NEURAHASH_STEP4_TAU_GATE, NEURAHASH_STEP4_TAU_Z (floats > 0; tau = 0 is refused, because
  bit-exact cross-machine agreement was measured NOT to hold).

PUBLICATION. One relay name per (event, check) under the attester's OWN prefix, in relay "self" mode
(neurahash/signed_put.py: the name embeds the signer address at segment 5):
    sharddiloco/glm/s1att/<campaign>/<event>/<address>~<kind>
written by the injected publisher (e.g. tools/sharddiloco_harness.ContentLane with account=<attester key> and
sign_shadow=True: a store_auth-signed PUT). Without a publisher (the default) verdicts go only to the local JSONL.
The attestation body is float-free (integer pico-nats) and signed under domain "glm-s1-attest".
Pure stdlib + neura_l1.signing; no torch; no network of its own.
"""
import hashlib
import json
import math
import os

from neurahash import glm_round_authority as RA

ATTEST_ENV = "NEURAHASH_STEP4_S1_ATTEST"
PUBLISH_ENV = "NEURAHASH_STEP4_S1_PUBLISH"
TAU_GATE_ENV = "NEURAHASH_STEP4_TAU_GATE"
TAU_Z_ENV = "NEURAHASH_STEP4_TAU_Z"
TAU_GATE_DEFAULT = 1e-6
TAU_Z_DEFAULT = 0.01
TAU_CE_DEFAULT = 2e-5
TOLERANCE_VERSION = "s1-tol/1"
ATTEST_DOMAIN = "glm-s1-attest"
BODY_TAG = "nrh-glm-s1-attestation/1"
NAME_ROOT = "sharddiloco/glm/s1att"
ADDR_SEGMENT = 5                          # signed_put "self" rule: sharddiloco/glm/s1att/<camp>/<event>/<addr>
PICO = 10 ** 12
AGREE, ABSTAIN, DISAGREE = "agree", "abstain", "disagree"
_COMPARABLE_PROBES = ("full_pool", "exact")   # the attester scored the SAME rows as the coordinator


def _on(name, environ=None):
    return str((os.environ if environ is None else environ).get(name, "")).strip() == "1"


def attest_on(environ=None):
    return _on(ATTEST_ENV, environ)


def publish_on(environ=None):
    return _on(PUBLISH_ENV, environ)


class Tolerance:
    """The S1 tolerance constants (protocol-versioned). tau must be > 0: bit-exact CPU agreement does not hold."""

    def __init__(self, tau_gate=TAU_GATE_DEFAULT, tau_z=TAU_Z_DEFAULT, tau_ce=TAU_CE_DEFAULT):
        for n, v in (("tau_gate", tau_gate), ("tau_z", tau_z), ("tau_ce", tau_ce)):
            v = float(v)
            if not math.isfinite(v) or v <= 0:
                raise ValueError("%s must be finite and > 0 (cross-machine CPU replay is not bit-exact)" % n)
        self.tau_gate, self.tau_z, self.tau_ce = float(tau_gate), float(tau_z), float(tau_ce)

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        return cls(float(env.get(TAU_GATE_ENV) or TAU_GATE_DEFAULT), float(env.get(TAU_Z_ENV) or TAU_Z_DEFAULT))

    def to_json(self):
        return {"v": TOLERANCE_VERSION, "tau_gate_pico": pico(self.tau_gate), "tau_z_micro": int(round(self.tau_z * 1e6)),
                "tau_ce_pico": pico(self.tau_ce)}


def pico(x):
    """Float -> integer pico-units (deterministic round-half-even), None stays None."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return int(round(x * PICO))


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ------------------------------------------------------------------------------------------------- classification
def classify_gate(row, tol):
    """(verdict, reason) for one ShadowCore gate row (the per-event re-score row)."""
    if "error" in row:
        return ABSTAIN, "rescore_error"
    cls = row.get("class")
    if cls in ("canary_halt", "coord_only_veto"):
        return ABSTAIN, cls                                   # not re-checkable from public records
    c, v = row.get("coordinator") or {}, row.get("verifier") or {}
    same = bool(row.get("agree"))
    gp, ga, margin = _num(c.get("gain")), _num(v.get("gain")), _num(c.get("margin"))
    if row.get("probe_mode") not in _COMPARABLE_PROBES:
        return (AGREE, "decision_only_independent_probe") if same else (ABSTAIN, "probe_rows_differ")
    if gp is None or ga is None:
        return (AGREE, "decision_only_no_numbers") if same else (ABSTAIN, "no_published_numbers")
    num_ok = abs(ga - gp) <= tol.tau_gate
    za_, zp_ = _num(v.get("z")), _num(c.get("z"))             # signed z (coordinator NEURAHASH_SD_SIGN_GATE)
    z_ok = za_ is None or zp_ is None or abs(za_ - zp_) <= tol.tau_z
    if same:
        if not num_ok:
            return ABSTAIN, "gain_outside_tau"
        if not z_ok:
            return ABSTAIN, "z_outside_tau"
        return (AGREE, "within_tau_signed_z") if (za_ is not None and zp_ is not None) else (AGREE, "within_tau")
    near = margin is not None and min(abs(gp - margin), abs(ga - margin)) <= tol.tau_gate
    za, zmin = _num(v.get("z")), _num(v.get("z_min"))
    zinvolved = "z-veto" in (str(c.get("decision")), str(v.get("decision")),
                             SIGNED_TO_SHADOW.get(str(c.get("signed_decision")), ""))
    if zinvolved and za is not None and zmin is not None and abs(za - zmin) <= tol.tau_z:
        near = True
    zp = _num(c.get("z"))                                     # the proposer's SIGNED z (None if not published)
    zminp = _num(c.get("z_min"))
    if zinvolved and zp is not None and zminp is not None and abs(zp - zminp) <= tol.tau_z:
        near = True
    return (ABSTAIN, "near_threshold") if near else (DISAGREE, "decision_flip_far_from_threshold")



# ------------------------------------------------------------------------------------------------- the SIGNED gate
# The coordinator (NEURAHASH_SD_SIGN_GATE=1, 2026-11-26 owner "make the proposer sign z") puts judged.gate_decision
# {"v": 1, decision, margin, z_min, z_mode, rows[{z, base_val, merged_val, gain, ...}], heldout{...}, probe{...}}
# into the accepted record, covered by the publish-audit signature. Absent (flag off) -> None and nothing changes.
SIGNED_GATE_KEY = "gate_decision"
SIGNED_GATE_V = 1
# signed label -> the ShadowCore decision vocabulary (tools/glm_verifier_shadow.verifier_decision / compare)
SIGNED_TO_SHADOW = {"accept": "accept", "margin_reject": "reject-margin", "z_veto": "z-veto",
                    "drift_cap": "drift-cap", "heldout_tol": "veto-after-margin",
                    "canary_halt": "veto-after-margin", "judge_reject": "veto-after-margin"}
NOT_RECHECKABLE = {"heldout_tol": "heldout_tol_not_rechecked", "canary_halt": "canary_halt",
                   "judge_reject": "judge_not_rechecked"}     # owner 2026-11-26: these abstain, never disagree


def signed_gate(rec):
    """Normalised signed gate decision of one accepted record, or None (absent, unknown version, malformed)."""
    j = rec.get("judged") if isinstance(rec, dict) else None
    g = j.get(SIGNED_GATE_KEY) if isinstance(j, dict) else None
    if not isinstance(g, dict) or g.get("v") != SIGNED_GATE_V or not isinstance(g.get("decision"), str):
        return None
    rows = g.get("rows") if isinstance(g.get("rows"), list) else []
    r = rows[0] if rows and isinstance(rows[0], dict) else {}
    h = g.get("heldout") if isinstance(g.get("heldout"), dict) else {}
    return {"decision": g["decision"], "margin": _num(g.get("margin")), "z_min": _num(g.get("z_min")),
            "z_mode": str(g.get("z_mode") or ""), "z": _num(r.get("z")), "z_ok": r.get("z_ok"),
            "margin_accepted": r.get("margin_accepted"), "gain": _num(r.get("gain")),
            "base_val": _num(r.get("base_val")), "merged_val": _num(r.get("merged_val")),
            "ce_before": _num(h.get("ce_before")), "ce_candidate": _num(h.get("ce_candidate")),
            "ce_after": _num(h.get("ce_after")), "merge_tol": _num(h.get("merge_tol"))}


def with_signed_gate(row, sg):
    """A COPY of a ShadowCore gate row whose coordinator side carries the signed z / z_min / label."""
    if sg is None or not isinstance(row, dict):
        return row
    c = dict(row.get("coordinator") or {})
    c.update(z=sg["z"], z_min=sg["z_min"], signed_decision=sg["decision"])
    if c.get("margin") is None:
        c["margin"] = sg["margin"]
    return dict(row, coordinator=c)


def check_signed_gate(sg, judged_accepted, tol):
    """Model-free internal consistency of the proposer's own signed numbers -> [(kind, verdict, reason)].
    The coordinator decides from exactly these floats, so a contradiction is a signed self-contradiction
    (disagree). A strict margin comparison within tau_gate of its threshold abstains (float rounding of
    base - merged vs merged < base - margin). Decisions not re-checkable from public records (held-out tol,
    canary, judge) never disagree on their own numbers."""
    if sg is None:
        return []
    out, d = [], sg["decision"]
    if d not in SIGNED_TO_SHADOW and d != "no_row":
        return [("signed_gate", ABSTAIN, "signed_label_unknown")]
    if judged_accepted is not None and (d == "accept") != bool(judged_accepted):
        out.append(("signed_gate", DISAGREE, "signed_decision_contradicts_record"))
    bv, mv, g, m = sg["base_val"], sg["merged_val"], sg["gain"], sg["margin"]
    if bv is not None and mv is not None and g is not None and abs((bv - mv) - g) > tol.tau_gate:
        out.append(("signed_gate", DISAGREE, "signed_gain_not_base_minus_merged"))
    if g is not None and m is not None and d in ("accept", "margin_reject", "z_veto"):
        passed = g > m
        if abs(g - m) <= tol.tau_gate:
            out.append(("signed_gate", ABSTAIN, "signed_margin_near_threshold"))
        elif passed != (d != "margin_reject"):
            out.append(("signed_gate", DISAGREE, "signed_margin_contradicts_decision"))
    z, zmin = sg["z"], sg["z_min"]
    if sg["z_mode"] == "veto" and zmin is not None and d in ("accept", "z_veto"):
        sig = bool(sg["z_ok"]) and z is not None and z >= zmin
        if sig != (d == "accept"):
            out.append(("signed_gate", DISAGREE, "signed_z_contradicts_decision"))
    elif d == "z_veto" and sg["z_mode"] != "veto":
        out.append(("signed_gate", DISAGREE, "z_veto_without_z_gate"))
    if d in NOT_RECHECKABLE:
        out.append(("signed_gate", ABSTAIN, NOT_RECHECKABLE[d]))
    if not out:
        out.append(("signed_gate", AGREE, "signed_gate_consistent"))
    return out


_CLASS_MAP = {                                               # non-gate ShadowCore rows (already classified there)
    "ce_agree": (AGREE, "ce_within_tol"), "ce_disagree": (DISAGREE, "ce_outside_2tol"),
    "ce_skipped": (ABSTAIN, "ce_skipped"),
    "root_agree": (AGREE, "roots_exact"), "root_disagree": (DISAGREE, "root_mismatch_exact"),
    "root_skipped": (ABSTAIN, "root_skipped"), "root_unverifiable": (ABSTAIN, "root_unverifiable"),
    "omitted": (DISAGREE, "eligible_record_omitted"), "chain_break": (DISAGREE, "audit_chain_break"),
    "campaign_rotated": (ABSTAIN, "campaign_rotated"), "walk_paused": (ABSTAIN, "walk_paused"),
    "event_error": (ABSTAIN, "event_error"),
}


def row_kind(row):
    k = row.get("kind")
    if k:
        return str(k)
    c = str(row.get("class") or "")
    if c.startswith("ce_"):
        return "ce"
    if c.startswith("root_"):
        return "root"
    return "gate"


def classify(row, tol):
    """(kind, verdict, reason) for any ShadowCore row; verdict None = not an attestable row."""
    kind = row_kind(row)
    if kind == "gate":
        v, r = classify_gate(row, tol)
        return kind, v, r
    m = _CLASS_MAP.get(row.get("class"))
    return kind, (m[0] if m else None), (m[1] if m else None)


# ------------------------------------------------------------------------------------------------- signed objects
def attestation_name(campaign, event, address, kind):
    return "%s/%s/%d/%s~%s" % (NAME_ROOT, campaign, int(event), str(address).lower(), kind)


def would_be_step_digest(campaign, row):
    """The digest a stage-4 prepare signs in S1: the coordinator's decision for this event as published."""
    c = row.get("coordinator") or {}
    body = {"campaign": str(campaign), "event": int(row.get("event", row.get("at_event", 0)) or 0),
            "record": row.get("record"), "delta_cid": row.get("delta_cid"), "decision": c.get("decision"),
            "gain_pico": pico(c.get("gain")), "margin_pico": pico(c.get("margin"))}
    return RA.sha(RA.canonical(body))


def attestation_body(campaign, row, kind, verdict, reason, tol, attester_addr):
    c, v = row.get("coordinator") or {}, row.get("verifier") or {}
    ev = row.get("event", row.get("at_event"))
    return {"w": BODY_TAG, "ns": "glm-lane|%s" % campaign, "campaign": str(campaign), "event": int(ev or 0),
            "kind": kind, "record": row.get("record"), "delta_cid": row.get("delta_cid"),
            "coord": {"decision": c.get("decision"), "gain_pico": pico(c.get("gain")),
                      "margin_pico": pico(c.get("margin"))},
            "attester": {"decision": v.get("decision"), "gain_pico": pico(v.get("gain"))},
            "class": row.get("class"), "verdict": verdict, "reason": reason, "tolerance": tol.to_json(),
            "attester_addr": str(attester_addr).lower()}


def sign_attestation(acct, body):
    return dict(body, sig=RA._sign(acct, body, ATTEST_DOMAIN))


def attestation_signer(att):
    """Recovered signer (lower-case) iff it equals the body's attester_addr, else None. The optional "prepare"
    (a self-signed stage-4 object, checked with glm_round_authority.prepare_signer) is not part of the body."""
    try:
        body = {k: v for k, v in att.items() if k not in ("sig", "prepare")}
        s = RA._recover(body, att.get("sig"), ATTEST_DOMAIN)
    except Exception:                                       # noqa: BLE001
        return None
    return s if s and s == str(att.get("attester_addr", "")).lower() else None


# ------------------------------------------------------------------------------------------------- the attester
class ShadowAttester:
    """Consume ShadowCore rows (observe(row)); write one signed verdict per attestable row to `out_path` (JSONL)
    and, only if a publisher was injected, put it on the relay under the attester's own prefix.

    acct      : the attester's signing account (neura_l1.signing / eth_account). Its address is the prefix.
    publisher : None (default: local JSONL only) or an object with put_blob(body_bytes, name=...) -- a signed
                ContentLane. The attester never builds a network client itself.
    Never raises into the caller's walk: a failure is counted and logged."""

    def __init__(self, acct, campaign, out_path, *, tolerance=None, publisher=None, log=print):
        self.acct, self.campaign, self.out_path = acct, str(campaign), out_path
        self.addr = str(acct.address).lower()
        self.tol = tolerance or Tolerance()
        self.publisher, self.log = publisher, log
        self.stats = {"agree": 0, "abstain": 0, "disagree": 0, "skipped": 0, "published": 0, "errors": 0}

    def make(self, row):
        """-> signed attestation dict, or None for a non-attestable row. Pure apart from signing."""
        kind, verdict, reason = classify(row, self.tol)
        if verdict is None:
            return None
        body = attestation_body(self.campaign, row, kind, verdict, reason, self.tol, self.addr)
        att = sign_attestation(self.acct, body)
        if kind == "gate" and verdict == AGREE:
            att = dict(att, prepare=RA.sign_prepare(self.acct, body["ns"], body["event"], 0,
                                                    would_be_step_digest(self.campaign, row)))
        return att

    def observe(self, row):
        try:
            att = self.make(row)
            if att is None:
                self.stats["skipped"] += 1
                return None
            self.stats[att["verdict"]] += 1
            line = json.dumps(att, sort_keys=True)
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if self.publisher is not None:
                self.publisher.put_blob(line.encode("utf-8"),
                                        name=attestation_name(self.campaign, att["event"], self.addr, att["kind"]))
                self.stats["published"] += 1
            return att
        except Exception as e:                              # noqa: BLE001 -- a shadow never stops the walk
            self.stats["errors"] += 1
            try:
                self.log("[s1-attest] ERROR %s: %s" % (type(e).__name__, e))
            except Exception:                               # noqa: BLE001
                pass
            return None


def from_env(acct, campaign, out_path, *, publisher=None, environ=None, log=print):
    """The ONLY constructor the tools use. None unless NEURAHASH_STEP4_S1_ATTEST=1. The publisher is dropped
    unless NEURAHASH_STEP4_S1_PUBLISH=1 as well (a second, separate switch for relay writes)."""
    if not attest_on(environ):
        return None
    pub = publisher if publish_on(environ) else None
    return ShadowAttester(acct, campaign, out_path, tolerance=Tolerance.from_env(environ), publisher=pub, log=log)


def digest_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
