"""UPDATE CAP (owner-approved 2026-10-23) -- the contributor scales what it PUBLISHES so that every expert
matrix's |D|_F/|W|_F <= NEURAHASH_GLM_UPDATE_CAP, direction preserved. Default OFF.

Why: offline gate replays (scratchpad/GATE_REPLAY_2026-10-22.md sec. 8) measured rejected deltas at
0.7*|D|/|W| ~42% median of the expert weight norm; the two ever-accepted deltas were 5% and 8%.

CPU/numpy only: the contributor imports torch-free; _G() pulls torch in, but no CUDA call is made.
Each test that touches apply_update_cap fails on the pre-change file (the symbols do not exist at
HEAD 83777ee), and test_control_uncapped_payload_fails_the_cap shows the old publish (the raw payload)
fails the very assertion the capped one passes."""
import inspect
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_HERE, _REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_harness as H                       # noqa: E402
import sharddiloco_glm_contributor as N               # noqa: E402

CAP = 0.114
I, HD, R = 48, 64, 16


def _resident(seed=0):
    rng = np.random.default_rng(seed)
    return {"gate": (rng.standard_normal((I, HD)) * 0.02).astype(np.float32),
            "up": (rng.standard_normal((I, HD)) * 0.02).astype(np.float32),
            "down": (rng.standard_normal((HD, I)) * 0.02).astype(np.float32)}


def _lora(seed=1, mag=0.03):
    rng = np.random.default_rng(seed)
    return {"lora_A_gu": (rng.standard_normal((R, HD)) * mag).astype(np.float32),
            "lora_B_gu": (rng.standard_normal((2 * I, R)) * mag).astype(np.float32),
            "lora_A_d": (rng.standard_normal((R, I)) * mag).astype(np.float32),
            "lora_B_d": (rng.standard_normal((HD, R)) * mag).astype(np.float32),
            "lora_scale": np.asarray([2.0], dtype=np.float32)}


def _coordinator_dense(payload):
    """Exactly the coordinator's path (sharddiloco_glm_coordinator.py ~2610): unpack the fp16 wire bytes,
    shape-validate the factors against the resident dims, materialise the dense delta."""
    G = N._G()
    wire = H.unpack_arrays(H.pack_arrays(payload, np.float16))
    ok, why = G.validate_lora_factors(wire, I, HD)
    assert ok, why
    return G.materialize_from_lora(wire)


def _ratios(dense, res):
    return {k: float(np.linalg.norm(dense[k].astype(np.float64)) /
                     np.linalg.norm(res[k].astype(np.float64))) for k in ("gate", "up", "down")}


def _cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_env_default_is_off_and_malformed_fails_fast():
    assert N.update_cap_from_env({}) is None
    assert N.update_cap_from_env({N.UPDATE_CAP_ENV: "  "}) is None
    assert N.update_cap_from_env({N.UPDATE_CAP_ENV: "0.114"}) == pytest.approx(0.114)
    for bad in ("abc", "0", "-1", "nan", "inf"):
        with pytest.raises(ValueError):
            N.update_cap_from_env({N.UPDATE_CAP_ENV: bad})


def test_publish_paths_are_gated_on_the_env():
    """Cap OFF -> neither publish path calls the cap, so put_delta sees the old object. Both the async
    and the sync path cap BEFORE their put_delta; layer claims + cap refuse to start."""
    src = inspect.getsource(N._run_async)
    assert "_update_cap = update_cap_from_env()" in src
    assert src.count("cap_for_publish(") == 1
    i = src.index("cap_for_publish(")
    assert "if _update_cap is not None and not args.garbage:" in src[max(0, i - 600):i]
    assert i < src.index("ecid = lane.put_delta(payload)")
    assert "layer claims are enabled" in src
    msrc = inspect.getsource(N.main)
    j = msrc.index("cap_for_publish(")
    assert "if _sync_cap is not None and not args.garbage:" in msrc[max(0, j - 300):j]
    assert j < msrc.index("ecid = lane.put_delta(payload)")


def test_cap_for_publish_counts_refusals_and_exits():
    res, lo = _resident(), _lora()
    lo["lora_B_gu"][0, 0] = np.nan
    lines = []
    out, n = N.cap_for_publish(CAP, lo, res, 1, 2, "m", lines.append, 0)
    assert out is None and n == 1 and "UPDATE-CAP REFUSED (L1,E2)" in lines[-1]
    out, n = N.cap_for_publish(CAP, lo, res, 1, 2, "m", lines.append, n)
    assert out is None and n == 2
    with pytest.raises(SystemExit):
        N.cap_for_publish(CAP, lo, res, 1, 2, "m", lines.append, n)
    out, n = N.cap_for_publish(CAP, _lora(), res, 1, 2, "m", lines.append, 2)   # success resets
    assert out is not None and n == 0 and "UPDATE-CAP coordinate (1,2)" in lines[-1]


def test_control_uncapped_payload_fails_the_cap():
    res, lo = _resident(), _lora()
    r = _ratios(_coordinator_dense(lo), res)
    assert max(r.values()) > CAP, r          # the pre-change publish ships this: over the cap


def test_cap_on_scales_every_matrix_under_cap_direction_preserved():
    res, lo = _resident(), _lora()
    raw = _coordinator_dense(lo)
    capped, info = N.apply_update_cap(lo, res, CAP)
    got = _coordinator_dense(capped)
    r = _ratios(got, res)
    assert all(v <= CAP for v in r.values()), r
    assert max(r.values()) > 0.9 * CAP                # shrunk TO the cap, not far below it
    for k in ("gate", "up", "down"):
        assert _cos(raw[k], got[k]) == pytest.approx(1.0, abs=1e-5)
        assert info["after"][k] == pytest.approx(r[k], rel=1e-9)
    # one uniform scale: every matrix shrank by the same factor
    ks = [r[k] / _ratios(raw, res)[k] for k in ("gate", "up", "down")]
    assert max(ks) - min(ks) < 2e-3 and ks[0] == pytest.approx(info["scale"], rel=2e-3)
    # only the B factors changed; A and the scale are byte-identical; the input is not mutated
    for key in ("lora_A_gu", "lora_A_d", "lora_scale"):
        assert np.array_equal(capped[key], lo[key])
    assert np.array_equal(lo["lora_B_gu"], _lora()["lora_B_gu"])
    line = N.update_cap_line("m", 3, 7, info)
    assert "UPDATE-CAP coordinate (3,7): ratios gate/up/down before " in line and "(scale " in line


def test_delta_already_under_cap_is_untouched():
    res, lo = _resident(), _lora(mag=0.005)
    assert max(_ratios(_coordinator_dense(lo), res).values()) < CAP
    out, info = N.apply_update_cap(lo, res, CAP)
    assert out is lo and info["scale"] == 1.0
    assert H.pack_arrays(out, np.float16) == H.pack_arrays(lo, np.float16)


def test_dense_payload_is_capped_too():
    res = _resident()
    rng = np.random.default_rng(5)
    dense = {k: (v * 0.0 + rng.standard_normal(v.shape) * 0.01).astype(np.float32) for k, v in res.items()}
    out, info = N.apply_update_cap(dense, res, CAP)
    wire = H.unpack_arrays(H.pack_arrays(out, np.float16))
    assert all(v <= CAP for v in _ratios(wire, res).values())
    assert all(_cos(dense[k], wire[k]) == pytest.approx(1.0, abs=1e-5) for k in dense)


@pytest.mark.parametrize("where", ["delta", "resident_nan", "resident_zero"])
def test_non_finite_or_zero_norm_refuses(where):
    res, lo = _resident(), _lora()
    if where == "delta":
        lo["lora_B_d"][0, 0] = np.inf
    elif where == "resident_nan":
        res["up"][0, 0] = np.nan
    else:
        res["down"][:] = 0.0
    with pytest.raises(N.UpdateCapRefused):
        N.apply_update_cap(lo, res, CAP)


def test_capped_record_is_self_consistent_cid():
    """expert_cid is the content hash of the packed CAPPED payload, so what the coordinator fetches by
    cid reproduces the capped D (the publish computes put_delta AFTER the cap)."""
    res, lo = _resident(), _lora()
    capped, _ = N.apply_update_cap(lo, res, CAP)
    blob = H.pack_arrays(capped, np.float16)
    assert blob != H.pack_arrays(lo, np.float16)
    np.testing.assert_allclose(_coordinator_dense(capped)["down"],
                               N._G().materialize_from_lora(H.unpack_arrays(blob))["down"])
