"""Determinism gate for the all-outbound miner's trunk-delta codec (neurahash/delta_codec.py).

The public miner (tools/run_miner.py -> tools/diloco_contributor.py) trains the trunk and PUBLISHES a
top-k+int8 compressed delta. That compressed artifact must be BIT-IDENTICAL across honest miners and
across builds, or the coordinator's merge gate can silently reject an honest contribution (or, worse,
accept a subtly-different one). This gate pins the codec on a fixed, fully synthetic, CPU-only tensor:

  1. compress -> decompress round-trips the KEPT top-k entries BIT-EXACTLY (to the codec's own
     deterministic int8 dequantization), with zeros everywhere else and the key-set preserved;
  2. the compressed bytes hash to a pinned GOLDEN sha256 -- a drift in numpy/zlib DEFLATE or the codec
     itself trips this immediately (mirrors tests/test_worker_determinism.py's golden-hash discipline).

No GPU, no torch, no corpus files: numpy + the pure codec only.
"""
import hashlib

import numpy as np

from neurahash import delta_codec

# Fixed top-k fraction for the gate (exercises real selection + tie handling, well under keep-all).
_FRAC = 0.1

# Pinned golden sha256 of compress_delta(_fixed_delta(), _FRAC). Computed with numpy 2.2.6 on
# C:/Python313. A mismatch means the compressed-delta bytes drifted (codec change, or a numpy/zlib
# DEFLATE change) -- investigate before shipping; do NOT loosen this to make it pass.
_GOLDEN_SHA256 = "60f950f7485866a752a4c44ce7b1af622a489760661b7f5b5d458fe7835c0645"


def _fixed_delta():
    """A deterministic synthetic {name: float32 ndarray} trunk delta -- fixed seed, fixed shapes."""
    rng = np.random.default_rng(0)
    return {
        "trunk.block0.attn.weight": rng.standard_normal((64, 48), dtype=np.float32),
        "trunk.block0.mlp.bias":    rng.standard_normal((96,),    dtype=np.float32),
    }


def test_delta_codec_roundtrip_keeps_topk_bit_exact():
    """decompress(compress(delta)) reconstructs the kept top-k entries BIT-EXACTLY (to the codec's own
    int8 dequantization) and zeros elsewhere, preserving the key-set."""
    delta = _fixed_delta()
    out = delta_codec.decompress_delta(delta_codec.compress_delta(delta, topk_fraction=_FRAC))

    assert set(out) == set(delta), "compressed delta must round-trip the exact key-set"
    for name, arr in delta.items():
        a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
        idx, q, scale = delta_codec.quantize_topk(a, _FRAC)          # the codec's own top-k selection
        expected = np.zeros(a.size, dtype=np.float32)
        expected[idx] = q.astype(np.float32) * np.float32(scale)     # deterministic int8 dequantization
        expected = expected.reshape(arr.shape)

        got = out[name]
        assert got.dtype == np.float32
        assert got.shape == arr.shape
        # full-array bit-exactness (covers the kept entries AND the zeros)
        assert np.array_equal(got, expected), f"{name}: reconstruction not bit-exact"
        # explicit: the KEPT top-k positions reconstruct to the exact dequantized values
        assert np.array_equal(got.reshape(-1)[idx], expected.reshape(-1)[idx]), \
            f"{name}: kept top-k entries not bit-exact"
        # everything outside the kept set is exactly zero
        mask = np.ones(a.size, dtype=bool)
        mask[idx] = False
        assert not np.any(got.reshape(-1)[mask]), f"{name}: non-kept entries must be zero"


def test_delta_codec_compressed_bytes_match_golden():
    """The compressed payload hashes to the pinned golden sha256 (cross-build determinism gate)."""
    payload = delta_codec.compress_delta(_fixed_delta(), topk_fraction=_FRAC)
    # deterministic within this build: recompute and confirm identical before checking the golden
    assert payload == delta_codec.compress_delta(_fixed_delta(), topk_fraction=_FRAC)
    assert hashlib.sha256(payload).hexdigest() == _GOLDEN_SHA256
