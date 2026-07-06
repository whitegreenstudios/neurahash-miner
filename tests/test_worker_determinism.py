"""
test_worker_determinism.py — the GOLDEN-DELTA gate for the public NeuraHash miner.

The worker's frozen-expert trunk step and the coordinator's recompute-verify MUST stay byte-for-byte
reproducible: recompute-verify accepts a delta only if cosine(submitted, recompute) >= VERIFY_COS=0.92.
If this path drifts, honest miners are SILENTLY false-rejected. This test pins
`neurahash.worker_core._recompute_trunk_delta` to an exact golden `trunk_delta_hash` over a FIXED tiny
CPU scenario, so a public miner build can prove its recompute path produces the EXACT delta the pool
expects (the same value the private coordinator recomputes and compares against).

Fully hermetic: no corpus files, no GPU, CPU-only (never contends with a GPU), deterministic inputs.

Run: python -m pytest tests/test_worker_determinism.py -q
"""
import numpy as np
import torch


# ------------------------- the FIXED scenario (identical every run) -------------------------
_ARCH = dict(d_model=64, n_head=4, n_layers=2, d_ff=256, n_experts=8, block_size=64)  # the `tiny` MoE rung
_VOCAB = 32
_HOSTED = [0]
_E = 8
_NL = 2
_BLOCK = 64
_SEED = 777
_H = 3
_LR = 1e-3
_BATCH = 4
_MODEL_SEED = 1234
_CORPUS_LEN = 4096

# GOLDEN HASH — recorded from _recompute_trunk_delta on the fixed scenario. A DIFFERENT hash on your
# machine means your build's recompute path diverges from the pool's => your honest work would be
# false-rejected (cosine<0.92). Measured on CPU (wire_mode=off): delta L2 ~0.5040, 27 trunk keys.
_GOLDEN_TRUNK_DELTA_HASH = "74a0c65edb6d3cc113d2d830994e4ede8b46e59498aef4101c53c11824c2bfd4"


def _fixed_corpus():
    """A FIXED 1-D token tensor (LCG-generated, deterministic, no file I/O), long enough for get_batch."""
    idx = np.arange(_CORPUS_LEN, dtype=np.int64)
    toks = (idx * 1103515245 + 12345) % _VOCAB
    return torch.from_numpy(toks.astype(np.int64))


def _build_fixed_inputs(build_pool_model, trunk_keys):
    """Build the FIXED (trunk_np, full_state, shard_data) under a pinned model seed so every run starts
    from byte-identical weights and data."""
    torch.manual_seed(_MODEL_SEED)
    m = build_pool_model(dict(_ARCH), _VOCAB, list(_HOSTED), "cpu", load_base=False)
    sd = m.state_dict()
    trunk_np = {k: sd[k].detach().cpu().float().numpy().copy() for k in trunk_keys(sd)}
    full_state = {k: v.detach().cpu().float().clone() for k, v in sd.items()}  # frozen-expert reference (raw fp32)
    shard_data = _fixed_corpus()
    return trunk_np, full_state, shard_data


def _compute_golden(recompute_fn, trunk_delta_hash, build_pool_model, trunk_keys):
    """Run the REAL recompute path on the fixed scenario and return trunk_delta_hash(delta)."""
    trunk_np, full_state, shard_data = _build_fixed_inputs(build_pool_model, trunk_keys)
    delta = recompute_fn(dict(_ARCH), _VOCAB, list(_HOSTED), full_state, trunk_np, shard_data,
                         _SEED, _H, _LR, _BATCH, _BLOCK, _NL, _E, "cpu", wire_mode="off")
    return trunk_delta_hash(delta), delta


def test_worker_core_recompute_is_reproducible():
    """The recompute path is reproducible run-to-run (self-consistency) and non-trivial (real work)."""
    from neurahash.worker_core import (_recompute_trunk_delta, trunk_delta_hash,
                                        build_pool_model, trunk_keys)
    h1, delta = _compute_golden(_recompute_trunk_delta, trunk_delta_hash, build_pool_model, trunk_keys)
    h2, _ = _compute_golden(_recompute_trunk_delta, trunk_delta_hash, build_pool_model, trunk_keys)
    assert h1 == h2, f"recompute is not reproducible run-to-run: {h1} != {h2}"
    l2 = float(np.linalg.norm(np.concatenate([np.asarray(v).ravel() for v in delta.values()])))
    assert l2 > 0.0, "recompute produced an all-zero delta (scenario is not exercising training)"


def test_worker_core_recompute_matches_golden():
    """The public build's recompute hash must equal the pinned golden. A different hash = a determinism
    regression: this build's honest work would be false-rejected by the pool (cosine<0.92)."""
    from neurahash.worker_core import (_recompute_trunk_delta, trunk_delta_hash,
                                        build_pool_model, trunk_keys)
    h, _ = _compute_golden(_recompute_trunk_delta, trunk_delta_hash, build_pool_model, trunk_keys)
    assert h == _GOLDEN_TRUNK_DELTA_HASH, (
        f"DETERMINISM MISMATCH: worker_core recompute hash {h} != pinned golden "
        f"{_GOLDEN_TRUNK_DELTA_HASH}. This build's trunk-delta bytes differ from the pool's reference; "
        f"your honest work would be rejected. Check torch/BLAS versions and the determinism pins.")
