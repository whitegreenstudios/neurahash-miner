"""Trunk-OPTIONAL model build for the GradCast 8 GB layer-claim miner.

WHY (measured 2026-07-29, safetensors headers): a layer dose is slabs 1.125 GiB + fp32 Dgu/Ddn
2.250 GiB and reads ZERO trunk tensors (glm_grad_cache.inner_step consumes gu_all/dn_all/act_fn
only), yet build_node_model always loaded the 4.024 GiB / 659-tensor trunk: 9.180 GiB total, OOM
on a 7.996 GiB 4060. Trunk-free is 5.156 GiB and fits.

THE PRICE, and what half these tests guard: the F2 own-slot re-gate (heldout_ce = a full trunk
forward; the local check that catches a forged/poisoned accepted record for THIS node's own
coordinate) is impossible without the trunk. The contract under test:
  * DEFAULT PATH BYTE-IDENTICAL: include_trunk defaults True; the summary gains no key; the
    re-gate closure make_regate_ce returns is the same lambda the five bind sites used inline.
  * TRUNK-FREE IS OPT-IN AND LOUD: the trunk stays on meta (zero storage; forward RAISES),
    make_regate_ce returns the REGATE_UNAVAILABLE sentinel (never a silent None), and
    apply_accepted logs SECURITY WAIVED on every own-slot fold it performs unjudged.
  * FAIL-FAST WALL: build_node_model refuses every forward-needing combination at config time.

Run: C:/Python313/python.exe -m pytest tests/test_glm_trunk_free.py -q
"""
import os
import shutil
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import piece_loader as PL  # noqa: E402  (bare import = the module identity the contributor uses)
import sharddiloco_glm_contributor as N  # noqa: E402
import sharddiloco_harness as H  # noqa: E402
from sharddiloco_glm_expert import build_tiny_glm, heldout_ce, markov_dataset, make_transition  # noqa: E402


# --------------------------------------------------------------------------- tiny real shard set
def _write_unfused_shards(model, cfg, out_dir):
    """Same shard writer as tests/test_trunk_reduction.py: split a tiny Glm4MoeLite state_dict into
    trunk.safetensors + experts_0.safetensors with a v1 manifest, so build_partial_model reads a
    REAL (if tiny) shard set rather than a mock of itself."""
    import json
    from safetensors.torch import save_file
    os.makedirs(os.path.join(out_dir, "pieces"), exist_ok=True)
    I = cfg.moe_intermediate_size
    trunk, experts, le = {}, {}, []
    for k, v in model.state_dict().items():
        if ".mlp.experts." in k and (k.endswith(".gate_up_proj") or k.endswith(".down_proj")):
            L = int(k.split(".layers.")[1].split(".")[0])
            if k.endswith(".gate_up_proj"):
                for E in range(v.shape[0]):
                    experts["model.layers.%d.mlp.experts.%d.gate_proj.weight" % (L, E)] = v[E, :I].clone()
                    experts["model.layers.%d.mlp.experts.%d.up_proj.weight" % (L, E)] = v[E, I:].clone()
                    le.append((L, E))
            else:
                for E in range(v.shape[0]):
                    experts["model.layers.%d.mlp.experts.%d.down_proj.weight" % (L, E)] = v[E].clone()
        else:
            trunk[k] = v.clone()
    save_file(trunk, os.path.join(out_dir, "pieces", "trunk.safetensors"))
    save_file(experts, os.path.join(out_dir, "pieces", "experts_0.safetensors"))
    man = {"version": 1, "model_dir": out_dir, "shard_gb": 0.001, "n_pieces": 2,
           "n_experts": len(le), "model_root": "x" * 64,
           "pieces": [{"piece": "trunk", "experts": "trunk", "n_keys": len(trunk),
                       "nbytes": 0, "sha256": "0" * 64, "cid": None},
                      {"piece": "experts_0", "experts": [[L, E] for (L, E) in sorted(set(le))],
                       "n_keys": len(experts), "nbytes": 0, "sha256": "0" * 64, "cid": None}]}
    with open(os.path.join(out_dir, "model_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    cfg.save_pretrained(out_dir)


@pytest.fixture(scope="module")
def tiny_shards(tmp_path_factory):
    """(source model, cfg, shard_dir, probe). The SOURCE model is the ground truth the shards were
    written from -- an independent path from either loader branch."""
    model, cfg = build_tiny_glm(seed=5, vocab=24, hidden=64, inter=128, moe_inter=48,
                                layers=3, n_experts=2, topk=1)
    model.eval()
    d = str(tmp_path_factory.mktemp("tf_shard"))
    _write_unfused_shards(model, cfg, d)
    P = make_transition(24, seed=7, peak=12)
    probe = markov_dataset(24, 16, 64, seed=999, transition=P)
    return model, cfg, d, probe


# ============================================================ 1. DEFAULT PATH: unchanged and alive
def test_include_trunk_defaults_true():
    import inspect
    assert inspect.signature(PL.build_partial_model).parameters["include_trunk"].default is True


def test_default_build_loads_trunk_and_regate_is_alive(tiny_shards):
    """Criterion 'default path unchanged', asserted not assumed: flag unset -> the trunk LOADS
    (zero meta params, forward works) and make_regate_ce hands out a WORKING closure that calls
    G.heldout_ce -- the F2 re-gate is alive."""
    _src, _cfg, d, probe = tiny_shards
    m, s = PL.build_partial_model(d, [0], device="cpu", dtype=torch.float32)
    assert s["meta_params_left"] == 0
    for key in ("include_trunk", "n_trunk_meta", "n_nontrunk_meta"):
        assert key not in s                      # default summary shape byte-identical
    assert not N.model_is_trunk_free(m)
    assert all(not p.is_meta for p in m.parameters())
    assert np.isfinite(heldout_ce(m, probe))     # the re-gate's forward path works

    calls = []

    class _G:
        @staticmethod
        def heldout_ce(model, val):
            calls.append(model)
            return 1.23

    host = types.SimpleNamespace(model=m)
    fn = N.make_regate_ce(_G, m, np.zeros((4, 8)), miner="t", log=None)
    assert callable(fn) and not isinstance(fn, N.RegateUnavailable)
    assert fn(host) == 1.23 and calls[0] is m
    # empty val split -> None, exactly as the old inline expression behaved
    assert N.make_regate_ce(_G, m, [], miner="t", log=None) is None


def test_trunk_free_flag_default_off():
    assert N.trunk_free_enabled({}) is False
    assert N.trunk_free_enabled({N.TRUNK_FREE_FLAG: "0"}) is False
    assert N.trunk_free_enabled({N.TRUNK_FREE_FLAG: "off"}) is False
    assert N.trunk_free_enabled({N.TRUNK_FREE_FLAG: "1"}) is True


# ============================================================ 2. TRUNK-FREE: gone, real, and loud
def test_trunk_free_leaves_trunk_on_meta_slabs_bit_identical(tiny_shards):
    """Trunk params on meta (zero storage), fused slabs REAL and bit-identical to the SOURCE model
    the shards were written from (ground truth from a different path than the loader)."""
    src, _cfg, d, _probe = tiny_shards
    m, s = PL.build_partial_model(d, [0], device="cpu", dtype=torch.float32, include_trunk=False)
    assert s["include_trunk"] is False and s["n_nontrunk_meta"] == 0
    assert N.model_is_trunk_free(m)
    meta = [n for n, p in m.named_parameters() if p.is_meta]
    real = [n for n, p in m.named_parameters() if not p.is_meta]
    assert len(meta) == s["n_trunk_meta"] > 0
    # every REAL param is a fused expert slab; everything else stayed meta
    assert real and all(".mlp.experts." in n and n.endswith(("gate_up_proj", "down_proj"))
                        for n in real)
    sd = src.state_dict()
    for L in s["resident_layers"]:
        gu = m.model.layers[L].mlp.experts.gate_up_proj
        dn = m.model.layers[L].mlp.experts.down_proj
        assert not gu.is_meta and not dn.is_meta
        assert torch.equal(gu.float(), sd["model.layers.%d.mlp.experts.gate_up_proj" % L].float())
        assert torch.equal(dn.float(), sd["model.layers.%d.mlp.experts.down_proj" % L].float())


def test_trunk_free_builds_without_trunk_file(tiny_shards, tmp_path):
    """THE 8 GB point: a trunk-free node never needs pieces/trunk.safetensors on disk at all.
    The default path must still REQUIRE it (no accidental waiver)."""
    _src, _cfg, d, _probe = tiny_shards
    d2 = str(tmp_path / "no_trunk")
    shutil.copytree(d, d2)
    os.remove(os.path.join(d2, "pieces", "trunk.safetensors"))
    m, s = PL.build_partial_model(d2, [0], device="cpu", dtype=torch.float32, include_trunk=False)
    assert N.model_is_trunk_free(m) and s["n_nontrunk_meta"] == 0
    with pytest.raises(FileNotFoundError):
        PL.build_partial_model(d2, [0], device="cpu", dtype=torch.float32)


def test_trunk_free_forward_raises_loud(tiny_shards):
    """A trunk-free model must never SILENTLY pretend it can forward (the re-gate would look
    healthy while checking nothing). Meta tensors make any forward raise, by construction."""
    _src, _cfg, d, _probe = tiny_shards
    m, _s = PL.build_partial_model(d, [0], device="cpu", dtype=torch.float32, include_trunk=False)
    with pytest.raises((RuntimeError, NotImplementedError)):
        m(input_ids=torch.zeros((1, 8), dtype=torch.long))


def test_trunk_free_with_trunk_quant_is_refused(tiny_shards):
    _src, _cfg, d, _probe = tiny_shards
    with pytest.raises(ValueError, match="contradictory"):
        PL.build_partial_model(d, [0], device="cpu", include_trunk=False, trunk_quant="nf4")


# ==================================================== 3. THE RE-GATE: fail closed, LOUD, never quiet
def test_make_regate_ce_trunk_free_returns_sentinel_and_logs():
    """Never a silent None for a trunk-free model: the sentinel is returned, the forfeited property
    is named in the log, and CALLING the sentinel is a hard error."""
    m = types.SimpleNamespace(_nh_trunk_free=True)
    lines = []
    fn = N.make_regate_ce(object(), m, np.zeros((4, 8)), miner="tf", log=lines.append)
    assert isinstance(fn, N.RegateUnavailable)
    assert fn is N.REGATE_UNAVAILABLE
    assert any("re-gate DISABLED" in ln and "FORFEITED" in ln for ln in lines), lines
    with pytest.raises(RuntimeError):
        fn(None)


def _slot_arrays(seed, scale=1.0):
    g = np.random.default_rng(seed)
    return {"gate": (g.standard_normal((6, 8)) * scale).astype(np.float32),
            "up": (g.standard_normal((6, 8)) * scale).astype(np.float32),
            "down": (g.standard_normal((8, 6)) * scale).astype(np.float32)}


class _FakeHost:
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


def _two_slot_record():
    d0, d1 = _slot_arrays(3, 0.1), _slot_arrays(4, 0.1)
    b0, b1 = H.pack_arrays(d0, np.float16), H.pack_arrays(d1, np.float16)
    c0, c1 = H.cid_of(b0), H.cid_of(b1)
    rec = dict(round=0, accepted=[dict(miner="m", slot=0, cid=c0, outer=0.7),
                                  dict(miner="x", slot=1, cid=c1, outer=0.7)])
    return rec, _FakeLane({c0: b0, c1: b1})


def test_apply_accepted_sentinel_folds_loudly_never_silently():
    """The heart of the fail-closed contract: under REGATE_UNAVAILABLE the own-slot fold (1) still
    happens BIT-IDENTICALLY to the ce_fn=None replay (refusing would fail replica_root_ok and
    freeze the frontier), (2) logs SECURITY WAIVED for the OWN slot and only the own slot, and
    (3) ce_fn=None itself stays silent (the legacy contract is untouched)."""
    e0, e1 = _slot_arrays(1), _slot_arrays(2)
    rec, lane = _two_slot_record()

    host = _FakeHost([e0, e1])
    lines = []
    n = N.apply_accepted(host, lane, rec, log=lines.append, ce_fn=N.REGATE_UNAVAILABLE, own_slot=0)
    assert n == 2
    waived = [ln for ln in lines if "SECURITY WAIVED" in ln]
    assert len(waived) == 1 and "slot 0" in waived[0], lines
    assert "CANNOT be detected" in waived[0]
    assert not np.array_equal(host.read_slot(0)["gate"], e0["gate"])       # own row DID fold

    host_none = _FakeHost([e0, e1])
    none_lines = []
    assert N.apply_accepted(host_none, lane, rec, log=none_lines.append, ce_fn=None,
                            own_slot=0) == 2
    for s_ in (0, 1):
        for k in ("gate", "up", "down"):
            assert np.array_equal(host.read_slot(s_)[k], host_none.read_slot(s_)[k]), (s_, k)
    assert not any("SECURITY WAIVED" in ln for ln in none_lines)           # None stays silent


def test_apply_accepted_working_ce_fn_still_gates():
    """The default path's re-gate is untouched by the sentinel plumbing: a regressing own-slot
    delta is still UNFOLDED and reported."""
    e0, e1 = _slot_arrays(1), _slot_arrays(2)
    rec, lane = _two_slot_record()
    host = _FakeHost([e0, e1])
    seq = iter([1.0, 9.9])                                  # base_ce, new_ce -> regression
    rejected = []
    n = N.apply_accepted(host, lane, rec, ce_fn=lambda h: next(seq), tol=0.0,
                         rejected=rejected, own_slot=0)
    assert n == 1 and len(rejected) == 1                    # own rejected, cross-domain folded
    for k in ("gate", "up", "down"):
        assert np.array_equal(host.read_slot(0)[k], e0[k])  # own slot restored exactly


# ================================================== 4. build_node_model: the fail-fast config wall
def test_build_node_model_refuses_forward_needing_combos(monkeypatch):
    monkeypatch.setenv(N.TRUNK_FREE_FLAG, "1")
    args = types.SimpleNamespace(mode="tiny", threads=1, device="cpu", claim_by="hash")
    with pytest.raises(SystemExit, match="only for the GLM layer-claim lane"):
        N.build_node_model(args)
    args.mode = "glm"
    monkeypatch.delenv(N.LAYER_CLAIM_FLAG, raising=False)
    with pytest.raises(SystemExit, match=N.LAYER_CLAIM_FLAG):
        N.build_node_model(args)
    monkeypatch.setenv(N.LAYER_CLAIM_FLAG, "1")
    args.claim_by = "affinity"
    with pytest.raises(SystemExit, match="affinity"):
        N.build_node_model(args)


def test_trunk_free_sync_lane_exit_code_reserved():
    """RC_TRUNK_FREE_SYNC exists, is distinct from every other RC_*, and main()'s sync branch
    refuses a trunk-free base with it (source-level assert: the guard sits between the async
    dispatch and the sync loop, so a v1 lane can never reach a forward with a meta trunk)."""
    rcs = {v for k, v in vars(N).items() if k.startswith("RC_")}
    assert N.RC_TRUNK_FREE_SYNC == 12 and len(rcs) == len(
        [k for k in vars(N) if k.startswith("RC_")])
