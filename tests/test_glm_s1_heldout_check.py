"""S1 validator x the coordinator's #145 HELD-OUT TOLERANCE (owner 2026-11-27: "add the held-out tolerance recheck
to the validator"). Public-repo port: PART A + the flag-off byte-identity pin only (PART B, the full-mode
re-measurement, needs the private coordinator code). CPU only, fakes only: no model, no network, no relay writes.
  CUDA_VISIBLE_DEVICES=-1 C:/Python313/python.exe -m pytest tests/test_glm_s1_heldout_check.py -q

Pins:
  PART A (model-free, light + full): the signed held-out rule on the proposer's own numbers. heldout_tol needs
          ce_candidate > ce_floor + merge_tol, ce_floor <= ce_before, ce_after == ce_before; accept needs
          ce_candidate <= ce_floor + merge_tol and ce_after == ce_candidate. Near the threshold -> abstain;
          missing fields -> no extra check; canary / judge stay abstain.
  PART B (full mode, --heldout-recheck, default OFF): ShadowCore measures held-out CE of the pre-merge state and of
          the candidate on the epoch's rows; the attester re-applies the rule. A ce_before mismatch is the
          positive control failing -> abstain heldout_rows_unmatched, never disagree. Flag off -> attestation bytes
          identical to the code before this change (golden hashes generated from HEAD c8b0634).
"""
import copy
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eth_account import Account  # noqa: E402

import glm_s1_attester as R  # noqa: E402
import glm_verifier_shadow as S  # noqa: E402
from neurahash import glm_shadow_attester as A  # noqa: E402

CAMP = "c0ffee0000000001"
TOL = A.Tolerance(tau_gate=3e-6, tau_z=0.1)             # s1-tol/1 recommendation; tau_ce default 2e-5
SIG_OK = [("audit", A.AGREE, "audit_signed_and_linked")]
COORD = Account.from_key(hashlib.sha256(b"heldout-recheck-coordinator").digest())
ACCT = Account.from_key(hashlib.sha256(b"golden-heldout|0").digest())


def _gd(decision, cb=8.5, cc=8.5005, ca=None, fl=None, mt=2e-4, gain=7.379e-5, accepted=None):
    """judged.gate_decision v1 as sharddiloco_glm_coordinator._gate_decision_fields writes it (branch phasec-live)."""
    fl = cb if fl is None else fl
    if ca is None:
        ca = cb if decision == "heldout_tol" else (cc if cc is not None else cb)
    return {"v": 1, "miner": "glm-m", "decision": decision, "margin": 1e-5, "z_min": 3.0, "z_mode": "veto",
            "rows": [{"expert": 3, "accepted": decision == "accept" if accepted is None else accepted,
                      "decision": decision, "base_val": 5.0, "merged_val": 5.0 - gain, "gain": gain,
                      "margin_accepted": True, "z": 7.15, "z_ok": True}],
            "heldout": {"ce_before": cb, "ce_candidate": cc, "ce_after": ca, "ce_floor": fl, "merge_tol": mt},
            "probe": {"threads": 6, "fp32": True, "full_pool": True, "outer": 0.7}}


def _rec(e, gd=None, accepted=None, epoch=None, root=None):
    acc = (gd or {}).get("decision") == "accept" if accepted is None else accepted
    j = {"name": "cg/%s/r%d/glm-m.%d" % (CAMP, e - 1, e), "accepted": acc, "miner": "glm-m",
         "gate": {"margin": 1e-5, "rows": [{"base_val": 5.0, "merged_val": 5.0 - 7.379e-5}]}}
    if gd is not None:
        j["gate_decision"] = gd
    r = {"event": e, "model_root": "ab" * 32, "accepted": [], "judged": j, "heldout_ce": 8.5}
    if epoch is not None:
        r["heldout_epoch"] = {"epoch": epoch, "root": root or ""}
    return r


def _row(cdec, vdec, agree, cls, dis=None, hr=None):
    r = {"event": 618, "record": "cg/x/r0/glm-m.618", "delta_cid": "cid-618", "probe_mode": "full_pool",
         "agree": agree, "class": cls, "disagreement": dis,
         "coordinator": {"decision": cdec, "gain": 7.379e-5, "margin": 1e-5, "base_val": 5.0},
         "verifier": {"decision": vdec, "gain": 7.349e-5, "z": 7.15, "z_min": 3.0, "base_val": 5.0,
                      "drift": "on"}, "gate_s": 1.0}
    if hr is not None:
        r["heldout_recheck"] = hr
    return r


VETO_ROW = dict(cdec="veto-after-margin", vdec="accept", agree=False, cls="disagree",
                dis="coordinator veto-after-margin; verifier accepted")


def _sgc(gd):
    return A.check_signed_gate(A.signed_gate(_rec(5, gd)), gd["rows"][0]["accepted"], TOL)


def _ck(att, k):
    return [(c["v"], c["r"]) for c in att["checks"] if c["k"] == k]


# ============================================================================ PART A: the signed held-out rule
def test_signed_gate_parses_ce_floor():
    sg = A.signed_gate(_rec(5, _gd("heldout_tol", fl=8.49)))
    assert sg["ce_floor"] == 8.49 and sg["merge_tol"] == 2e-4 and sg["ce_candidate"] == 8.5005


def test_heldout_tol_consistent_agrees():
    # candidate 8.5005 > floor 8.5 + tol 2e-4 (far: 3e-4 past it), ce_after = ce_before (rolled back)
    assert _sgc(_gd("heldout_tol", accepted=False)) == [("signed_gate", A.AGREE, "signed_heldout_rule_consistent")]
    # the ratchet floor below ce_before is legal (floor = min(prev, best))
    assert _sgc(_gd("heldout_tol", cc=8.5, fl=8.49, accepted=False))[0][1] == A.AGREE


def test_heldout_tol_with_candidate_within_tolerance_disagrees():
    out = _sgc(_gd("heldout_tol", cc=8.5001, accepted=False))           # 1e-4 BELOW the threshold 8.5002
    assert out == [("signed_gate", A.DISAGREE, "signed_heldout_veto_within_tolerance")]


def test_heldout_near_threshold_abstains():
    for cc in (8.5002 + 1e-5, 8.5002 - 1e-5):                           # |d| = 1e-5 <= tau_ce 2e-5
        out = _sgc(_gd("heldout_tol", cc=cc, accepted=False))
        assert out == [("signed_gate", A.ABSTAIN, "signed_heldout_near_threshold")], cc
    out = _sgc(_gd("accept", cc=8.5002 + 1e-5))
    assert ("signed_gate", A.ABSTAIN, "signed_heldout_near_threshold") in out


def test_accept_past_tolerance_disagrees_and_consistent_accept_agrees():
    out = _sgc(_gd("accept", cc=8.5009))
    assert ("signed_gate", A.DISAGREE, "signed_heldout_accept_past_tolerance") in out
    assert _sgc(_gd("accept", cc=8.4990)) == [("signed_gate", A.AGREE, "signed_heldout_rule_consistent")]


def test_signed_held_out_contradictions():
    cases = [(_gd("heldout_tol", ca=8.5005, accepted=False), "signed_heldout_rollback_ce_after_moved"),
             (_gd("accept", cc=8.499, ca=8.5), "signed_heldout_accept_ce_after_not_candidate"),
             (_gd("heldout_tol", fl=8.6, cc=8.9, accepted=False), "signed_heldout_floor_above_before")]
    for gd, why in cases:
        assert ("signed_gate", A.DISAGREE, why) in _sgc(gd), why


def test_missing_fields_add_no_check():
    for k in ("ce_floor", "merge_tol", "ce_before", "ce_candidate"):
        gd = _gd("heldout_tol", accepted=False)
        gd["heldout"][k] = None
        assert _sgc(gd) == [("signed_gate", A.ABSTAIN, "heldout_tol_not_rechecked")], k     # the old behaviour
        gd = _gd("accept", cc=8.499)
        gd["heldout"][k] = None
        assert _sgc(gd) == [("signed_gate", A.AGREE, "signed_gate_consistent")], k
    gd = _gd("heldout_tol", accepted=False)
    del gd["heldout"]["ce_floor"]                                         # older v1 records: key absent
    assert _sgc(gd) == [("signed_gate", A.ABSTAIN, "heldout_tol_not_rechecked")]
    for lab, why in (("canary_halt", "canary_halt"), ("judge_reject", "judge_not_rechecked")):
        assert _sgc(_gd(lab, accepted=False)) == [("signed_gate", A.ABSTAIN, why)]


def test_light_mode_reports_the_signed_heldout_numbers():
    att = R.build_attestation(ACCT, CAMP, 5, _rec(5, _gd("heldout_tol", accepted=False)), [], list(SIG_OK),
                              "light", TOL, "c", now=1)
    assert _ck(att, "signed_gate") == [(A.AGREE, "signed_heldout_rule_consistent")]
    h = att["coord"]["signed_gate"]["heldout"]
    assert h == {"ce_before_pico": 8500000000000, "ce_candidate_pico": 8500500000000, "ce_after_pico": 8500000000000,
                 "ce_floor_pico": 8500000000000, "merge_tol_pico": 200000000}
    assert att["verdict"] == A.ABSTAIN and att["reason"] == "gate:light_no_rescore"
    json.dumps(att)


def test_event_618_shape_without_gate_decision_still_abstains():
    """Event 618 of 508ed8c1ec9a2f9a: no gate_decision (signed gate came mid-run), the probe gate would accept
    (coord gain 7.379e-5, validator 7.349e-5, margin 1e-5, z 7.15), the coordinator vetoed on held-out tol."""
    row = _row("veto-after-margin", "accept", False, "disagree",
               "coordinator veto-after-margin (held-out tol: the merged result moved held-out CE past the "
               "tolerance); verifier accepted")
    att = R.build_attestation(ACCT, CAMP, 618, _rec(618, None, accepted=False), [row], list(SIG_OK), "full", TOL,
                              "c", now=1)
    assert _ck(att, "gate") == [(A.ABSTAIN, "heldout_tol_not_rechecked")] and att["verdict"] == A.ABSTAIN
    assert not _ck(att, "heldout") and "heldout_recheck" not in att["measured"]
    # a stray measurement on an UNSIGNED record is ignored (nothing to compare it with)
    row2 = dict(row, heldout_recheck={"status": "measured", "ce_before": 8.5, "ce_candidate": 8.4})
    att2 = R.build_attestation(ACCT, CAMP, 618, _rec(618, None, accepted=False), [row2], list(SIG_OK), "full", TOL,
                               "c", now=1)
    assert _ck(att2, "gate") == [(A.ABSTAIN, "heldout_tol_not_rechecked")] and not _ck(att2, "heldout")


# ============================================================================ flag off: byte-identical
GOLDEN = {  # sha256(line_bytes) from HEAD c8b0634 (before this change), same inputs, now=1, code "c"
    "full_accept": "a41ec1a6c089819cff8c5e17e414afff7f629868e7ec6391df84a5489b8fda16",
    "full_618": "f944cdac3c960d11ce8871368b7d0cd1b78a62ab1c1c8f49206146acdb7e5e03",
    "light": "9ebeec7b504e0565a1acf51972eefe1ee067699cb5a8f7e3f634ce0e1c9a4ae9",
}


def test_flag_off_and_no_gate_decision_is_byte_identical_to_before():
    cases = {"full_accept": (_rec(618, None, accepted=True), [_row("accept", "accept", True, "agree")], "full"),
             "full_618": (_rec(618, None, accepted=False),
                          [_row("veto-after-margin", "accept", False, "disagree",
                                "coordinator veto-after-margin (held-out tol: the merged result moved held-out CE "
                                "past the tolerance); verifier accepted")], "full"),
             "light": (_rec(618, None, accepted=False), [], "light")}
    for k, (rec, rows, mode) in cases.items():
        att = R.build_attestation(ACCT, CAMP, 618, rec, rows, list(SIG_OK), mode, TOL, "c", now=1)
        assert hashlib.sha256(R.line_bytes(att)).hexdigest() == GOLDEN[k], k


