"""shardDiLoCo GLM lane -- guards for the two NEW real-model lane processes
(tools/sharddiloco_glm_contributor.py + tools/sharddiloco_glm_coordinator.py, design
docs/research/SHARDDILOCO_GLM_WAN_PLAN.md sec 2).

Those two files carry a REAL glm4_moe_lite expert over the SAME lane that passed the 2026-07-21
two-box WAN run (tag de33f6). This module guards the four properties that make that safe, and does
it WITHOUT loading torch or any model, so the suite stays fast and hermetic:

  1. a GLM-shaped expert delta survives the lane wire ......... test_glm_delta_wire_roundtrip
  2. a record signed with the WRONG key never moves a weight .. test_wrong_key_record_is_rejected
     (and the raw HMAC check itself) ......................... test_wrong_key_signature_does_not_verify
  3. the coordinator is DEFAULT-OFF .......................... test_coordinator_refuses_without_flag
  4. the GLM lane cannot clobber the toy lane's objects ...... test_glm_lane_names_are_disjoint
     and a contributor's local replay of the coordinator's
     merge is bit-identical to the coordinator's .............. test_apply_accepted_replicates_merge

Run: C:/Python313/python.exe -m pytest tests/test_sharddiloco_glm_lane.py -q
"""
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neurahash.diloco_merge as dm                                   # noqa: E402
import sharddiloco_harness as H                                       # noqa: E402
import sharddiloco_glm_contributor as N

# Three tests below exercise the COORDINATOR half of the lane (apply_delta_gated / shard_merge_round
# / the default-off master gate). The public miner's neurahash/diloco_merge.py is a contributor-only
# subset that deliberately omits that machinery, so on a public checkout these cannot run -- and a
# test suite that ImportErrors on a fresh clone is worse than one that says why it skipped.
coordinator_only = pytest.mark.skipif(
    not hasattr(dm, "apply_delta_gated"),
    reason="coordinator-side merge machinery absent; this checkout ships the contributor subset")

# A separate axis: some tests SPAWN tools/sharddiloco_glm_coordinator.py. Once the trust root was
# published the merge machinery became importable everywhere, so the check above stopped
# distinguishing them and the spawn test failed on checkouts that (correctly) do not ship the
# training-coordinator script. Gate on the actual dependency -- the file -- not on a proxy for it.
_COORD_SCRIPT = os.path.join(_TOOLS, "sharddiloco_glm_coordinator.py")
coordinator_script_only = pytest.mark.skipif(
    not os.path.exists(_COORD_SCRIPT),
    reason="tools/sharddiloco_glm_coordinator.py is not in this checkout (training-coordinator role)")
                               # noqa: E402

# GLM-4.7-Flash routed-expert canonical shapes, scaled down 32x so the test is instant. The real
# unit is {gate:[I,H], up:[I,H], down:[H,I]} with H=hidden_size=2048, I=moe_intermediate_size=1536
# (piece_loader.py:369-373); only the RANK/ORDER/keys matter to the wire, not the magnitude.
_HID, _INT = 64, 48


def _glm_shaped_delta(seed=0, scale=1e-3):
    rng = np.random.default_rng(seed)
    return {"gate": (rng.standard_normal((_INT, _HID)) * scale).astype(np.float32),
            "up": (rng.standard_normal((_INT, _HID)) * scale).astype(np.float32),
            "down": (rng.standard_normal((_HID, _INT)) * scale).astype(np.float32)}


# ================================================================================ 1. the wire
def test_glm_delta_wire_roundtrip():
    """A GLM {gate,up,down} delta must survive pack_arrays/unpack_arrays -- the SAME wire the toy lane
    uses (plan sec 2: the gap is the model layer, not the wire layer, so pack_arrays needs no change).
    fp16 is the deliberate transport dtype (D1: OpenDiLoCo FP16 transfer)."""
    d = _glm_shaped_delta(seed=1)
    body = H.pack_arrays(d, np.float16)
    back = H.unpack_arrays(body)

    assert sorted(back) == ["down", "gate", "up"]                 # keys preserved
    for k in d:
        assert back[k].shape == d[k].shape, k                     # shapes preserved
        assert back[k].dtype == np.float64, k                     # unpack widens fp16 -> float64
        # the ONLY loss is the intended fp16 squeeze (relative, since values are ~1e-3)
        assert np.allclose(back[k], d[k], rtol=1e-3, atol=1e-7), k
    # the wire really is fp16, and it is a true content address (deterministic bytes)
    assert len(body) - len(body) % 2 >= 2 * sum(a.size for a in d.values())
    assert H.pack_arrays(d, np.float16) == body
    assert H.cid_of(body) == H.cid_of(H.pack_arrays(d, np.float16))
    # a non-GLM-shaped delta is NOT silently accepted by the merge gate
    assert np.shape(back["down"]) != np.shape(back["gate"])


# ============================================================================ 2. signed identity
def test_wrong_key_signature_does_not_verify():
    good, bad = b"0" * 16, b"1" * 16
    cid = H.cid_of(H.pack_arrays(_glm_shaped_delta(seed=2), np.float16))
    sig = H.sign(good, cid, 3, "miner0")
    assert H.verify(good, sig, cid, 3, "miner0") is True
    assert H.verify(bad, sig, cid, 3, "miner0") is False           # wrong key
    assert H.verify(good, sig, cid, 4, "miner0") is False          # replayed into another round
    assert H.verify(good, sig, cid, 3, "miner1") is False          # another miner's name


@coordinator_only
def test_wrong_key_record_is_rejected():
    """The property that matters: a record whose signature does not verify must never touch a canonical
    weight. This is the coordinator's exact call shape -- verify_ok is computed from the roster key and
    handed to shard_merge_round, which drops it BEFORE the gate (diloco_merge.py:844)."""
    roster_key, attacker_key = b"a" * 16, b"b" * 16
    expert0 = _glm_shaped_delta(seed=3, scale=1.0)
    before = {k: v.copy() for k, v in expert0.items()}
    delta = _glm_shaped_delta(seed=4, scale=1.0)
    cid = H.cid_of(H.pack_arrays(delta, np.float16))
    sig = H.sign(attacker_key, cid, 0, "miner0")                   # signed with the WRONG key

    calls = []

    def eval_expert(e, cand, pX, pY):
        calls.append(e)
        return 0.0                                                 # would ACCEPT anything, if reached

    probe = dm.SecretRotatedProbe({0: (np.zeros((4, 2)), np.zeros((4, 2)))}, seed=0, size=4)
    meter = dm.FlopMeter(1.0)
    contribs = [dict(miner="miner0", expert=0, verify_ok=H.verify(roster_key, sig, cid, 0, "miner0"),
                     base_round=0, trunk_delta={}, expert_delta=delta, train_flops=1.0)]
    assert contribs[0]["verify_ok"] is False
    res = dm.shard_merge_round({}, [expert0], contribs, eval_expert, probe, meter, 0, outer=0.7)

    assert res["accepts"] == 0 and res["rejects"] == 1
    assert calls == [], "a wrongly-signed delta must never even be evaluated"
    for k in before:
        assert np.array_equal(expert0[k], before[k]), "canonical weights moved on a bad signature"


# ========================================================================== 3. default-off gate
@coordinator_script_only
def test_coordinator_refuses_without_flag():
    """The GLM coordinator inherits the toy coordinator's master gate: with NEURAHASH_SHARDDILOCO unset
    it must refuse (rc 3) before touching the lane or loading any model."""
    env = dict(os.environ)
    env.pop("NEURAHASH_SHARDDILOCO", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = _REPO + os.pathsep + _TOOLS + os.pathsep + env.get("PYTHONPATH", "")
    script = os.path.join(_TOOLS, "sharddiloco_glm_coordinator.py")
    p = subprocess.run([sys.executable, script], cwd=_REPO, env=env, capture_output=True, timeout=300)
    out = (p.stdout + p.stderr).decode("utf-8", "replace")
    assert p.returncode == 3, "expected rc=3 (default-off refusal), got %d: %s" % (p.returncode, out)
    assert "REFUSING to run" in out and "NEURAHASH_SHARDDILOCO" in out

    env["NEURAHASH_SHARDDILOCO"] = "1"                             # flag ON -> it gets past the gate
    p2 = subprocess.run([sys.executable, script, "--url", "http://127.0.0.1:9"], cwd=_REPO, env=env,
                        capture_output=True, timeout=300)
    out2 = (p2.stdout + p2.stderr).decode("utf-8", "replace")
    assert "REFUSING to run" not in out2
    assert p2.returncode == 4 and "cannot reach content store" in out2   # died on the dead lane instead


# ================================================================= 4. lane isolation + replication
def test_glm_lane_names_are_disjoint():
    """The GLM run shares the live content store with the PASSING de33f6 toy artifacts, so its pointer
    and contribution names must not collide (plan risk 7). use_glm_lane_names() must retarget the
    harness pointer BY RUNTIME ASSIGNMENT -- tools/sharddiloco_harness.py itself is never edited."""
    original = H.POINTER_NAME
    try:
        assert N.GLM_POINTER_NAME != original
        assert N.contrib_prefix(3) != H.contrib_prefix(3)
        assert N.contrib_name(3, "miner0") != H.contrib_name(3, "miner0")
        assert not N.accepted_name(3).startswith(H.contrib_prefix(3))
        N.use_glm_lane_names()
        assert H.POINTER_NAME == N.GLM_POINTER_NAME
    finally:
        H.POINTER_NAME = original


class _FakeHost:
    """Minimal GlmExpertLaneHost stand-in: slot weights as float32 numpy, exactly what read_slot /
    write_slot expose (tools/sharddiloco_glm_expert.py:245-266) but with no torch model behind them."""

    def __init__(self, slots_params):
        self._p = [{k: v.astype(np.float32) for k, v in d.items()} for d in slots_params]
        self.slots = [(1, i) for i in range(len(self._p))]

    def read_slot(self, i):
        return {k: v.copy() for k, v in self._p[i].items()}

    def write_slot(self, i, d):
        self._p[i] = {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}


class _FakeLane:
    def __init__(self, blobs):
        self._b = blobs

    def get_delta(self, cid):
        return H.unpack_arrays(self._b[cid])


@coordinator_only
def test_apply_accepted_replicates_merge():
    """The GLM lane replaces the toy lane's 'pull the whole canonical state' with 'replay the accepted
    deltas locally' (plan sec 2b: a float64 GLM state blob is 75.5 MB, over content_store's 32 MiB cap).
    That is only safe if the contributor's replay is BIT-IDENTICAL to the coordinator's merge, which is
    what the pointer's model_root fingerprint asserts every round. Guard it here."""
    expert0 = _glm_shaped_delta(seed=5, scale=1.0)
    delta = _glm_shaped_delta(seed=6, scale=1.0)
    outer = 0.7
    body = H.pack_arrays(delta, np.float16)
    cid = H.cid_of(body)
    wire = H.unpack_arrays(body)                     # what the coordinator actually gates/applies

    # --- coordinator side: apply_delta_gated on an accepting eval, then land it on the model ---
    coord = _FakeHost([expert0])
    canonical = coord.read_slot(0)
    seq = iter([1.0, 0.5])                           # base_val, merged_val -> accept
    verdict = dm.apply_delta_gated(canonical, wire, lambda _p: next(seq), outer=outer, margin=0.0)
    assert verdict["accepted"] is True
    coord.write_slot(0, canonical)

    # --- contributor side: apply_accepted replaying the coordinator's ACCEPTED record ---
    node = _FakeHost([expert0])
    n = N.apply_accepted(node, _FakeLane({cid: body}),
                         dict(round=0, accepted=[dict(miner="miner0", slot=0, cid=cid, outer=outer)]))
    assert n == 1

    for k in ("gate", "up", "down"):
        assert np.array_equal(node.read_slot(0)[k], coord.read_slot(0)[k]), \
            "contributor replay diverged from the coordinator merge on key %s" % k
    assert N.model_root(node) == N.model_root(coord)

    # a REJECTED delta (empty accepted list) must leave the replica exactly where it was
    node2 = _FakeHost([expert0])
    assert N.apply_accepted(node2, _FakeLane({cid: body}), dict(round=1, accepted=[])) == 0
    assert N.model_root(node2) != N.model_root(coord)
    for k in ("gate", "up", "down"):
        assert np.array_equal(node2.read_slot(0)[k], expert0[k].astype(np.float32))


# ================================================================= 5. F1 probe-pool secrecy on disk
def test_prep_probe_heldout_go_to_coord_only_dir(tmp_path):
    """F1: tools/glm_wan_prep_data.build_domain must write train+val to the MINER-facing dir and
    probe+heldout to the COORDINATOR-ONLY dir, so a miner's data dir never holds the secret gate
    pool (one `rsync <data-dir> pod:` then cannot defeat the gate)."""
    import importlib
    prep = importlib.import_module("glm_wan_prep_data")
    miner, coord = tmp_path / "miner", tmp_path / "coord"
    miner.mkdir()
    coord.mkdir()
    seq = 4
    n_tokens = sum(n for _, n in prep.SPLITS) * seq
    fake_tok = lambda text, add_special_tokens=False: {"input_ids": list(range(n_tokens))}  # noqa: E731
    _arr, written, _used = prep.build_domain(fake_tok, "x", seq, str(miner), str(coord), "code",
                                             vocab_size=n_tokens + 1)
    where = {name: os.path.dirname(path) for name, _shape, path in written}
    assert where["train"] == str(miner) and where["val"] == str(miner)
    assert where["probe"] == str(coord) and where["heldout"] == str(coord)
    # the MINER-facing dir on disk must contain NO probe/heldout file at all
    miner_files = os.listdir(str(miner))
    assert not any(("probe" in f or "heldout" in f) for f in miner_files), miner_files
    assert prep.COORD_ONLY == frozenset({"probe", "heldout"})


# ============================================================================ 6. F8 LoRA shape gate
def test_lora_shape_gate_rejects_mismatch():
    """F8: validate_lora_factors accepts well-formed factors for the resident dims and REJECTS a
    rank/dim mismatch (or a non-LoRA payload) BEFORE the coordinator's float64 outer product
    materialises attacker-controlled factors."""
    import sharddiloco_glm_expert as G
    I, Hd, r = 48, 64, 16
    ref = {"gate": np.zeros((I, Hd), np.float32), "up": np.zeros((I, Hd), np.float32),
           "down": np.zeros((Hd, I), np.float32)}
    good = G.garbage_lora(ref, r=r)
    ok, why = G.validate_lora_factors(good, I, Hd)
    assert ok, why
    ok2, why2 = G.validate_lora_factors(good, I + 8, Hd)          # wrong resident dims
    assert not ok2 and "shape" in why2.lower()
    ok3, _ = G.validate_lora_factors({"gate": np.zeros((I, Hd), np.float32)}, I, Hd)  # not LoRA
    assert not ok3
    big = G.garbage_lora(ref, r=1024)                            # oversized rank
    ok4, why4 = G.validate_lora_factors(big, I, Hd, max_rank=512)
    assert not ok4 and "rank" in why4.lower()


# ================================================================ 7. F2 no-key defense: local re-gate
def test_apply_accepted_regate_rejects_regressing_delta():
    """F2 defense-in-depth: apply_accepted RE-GATES each fetched delta on a local held-out metric and
    UNFOLDS one that regresses it (leaving the replica untouched), while folding one that improves it.
    A forged accepted record on the UNSIGNED shared-token lane therefore cannot push an ungated delta
    into a replica -- the core no-key protection when no coordinator pubkey is pinned yet."""
    expert0 = _glm_shaped_delta(seed=11, scale=1.0)
    body = H.pack_arrays(_glm_shaped_delta(seed=12, scale=1.0), np.float16)
    cid = H.cid_of(body)
    rec = dict(round=0, accepted=[dict(miner="m", slot=0, cid=cid, outer=0.7)])

    node = _FakeHost([expert0])                                  # ce_fn reports a REGRESSION -> reject
    seq = iter([1.0, 2.0])
    rejected = []
    n = N.apply_accepted(node, _FakeLane({cid: body}), rec, ce_fn=lambda h: next(seq), tol=0.0,
                         rejected=rejected)
    assert n == 0 and len(rejected) == 1
    for k in ("gate", "up", "down"):
        assert np.array_equal(node.read_slot(0)[k], expert0[k].astype(np.float32)), k

    node2 = _FakeHost([expert0])                                 # ce_fn reports an IMPROVEMENT -> fold
    seq2 = iter([1.0, 0.5])
    n2 = N.apply_accepted(node2, _FakeLane({cid: body}), rec, ce_fn=lambda h: next(seq2), tol=0.0)
    assert n2 == 1
    assert not np.array_equal(node2.read_slot(0)["gate"], expert0["gate"].astype(np.float32))


def test_apply_accepted_regate_rejects_shape_mismatch():
    """F2/F8 defense-in-depth on the contributor: a fetched delta whose shape does not match the slot
    is skipped, not broadcast into the weights."""
    expert0 = _glm_shaped_delta(seed=13, scale=1.0)

    class _BadLane:
        def get_delta(self, cid):
            return {"gate": np.zeros((_INT + 1, _HID), np.float64),   # wrong first dim
                    "up": np.zeros((_INT, _HID), np.float64),
                    "down": np.zeros((_HID, _INT), np.float64)}

    node = _FakeHost([expert0])
    rejected = []
    n = N.apply_accepted(node, _BadLane(),
                         dict(round=0, accepted=[dict(slot=0, cid="x", outer=0.7)]), rejected=rejected)
    assert n == 0 and rejected and rejected[0].get("reason") == "shape-mismatch"
    for k in ("gate", "up", "down"):
        assert np.array_equal(node.read_slot(0)[k], expert0[k].astype(np.float32))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
