"""neurahash/delta_codec.py — communication-efficient trunk-delta compression for DiLoCo-over-IPFS.

WHY. tools/diloco_contributor.py trains the trunk and, today, publishes the FULL fp32 trunk delta
(a ~2.4 GB .npz for the dense Qwen3-0.6B trunk). Two real-world problems (RunPod test 2026-07-10):
(1) a NAT'd contributor cannot SERVE a big file inbound, but CAN PUSH a small one outbound;
(2) large-file transfers truncate on flaky links. The fix is to make the contributor's output small
enough to push through NAT. This module compresses a {name: tensor} trunk delta to a <10 MB payload.

SCHEME (per tensor, top-k + int8):
  * keep only the TOP-K entries by |magnitude| (k = topk_fraction * numel, >=1);
  * quantize the kept values to int8 with a per-tensor scale = max(|kept|)/127;
  * store the sorted flat indices GAP-encoded (np.diff), the int8 values, the float32 scale, and the
    tensor shape, DEFLATE-packed via np.savez_compressed. Gap-encoding makes the int32 index stream
    mostly-zero in its high bytes, so DEFLATE squeezes it toward ~2-3 bytes/entry; with the default
    topk this lands the whole 600M-param trunk delta well under the 10 MB ceiling.

SAFETY. The payload is a plain-numeric npz: decompress loads it with allow_pickle=False, so a hostile
artifact cannot execute code (same discipline as neurahash/diloco_merge.fetch_delta). Decompress
returns {name: float32 ndarray} of the ORIGINAL shapes (zeros except the kept entries).

ERROR FEEDBACK. compress_delta(..., return_residual=True) also returns {name: residual} where
residual = delta - decompressed. A caller running multiple rounds can ACCUMULATE that residual into
the next round's delta so the information dropped by top-k is not lost forever (standard DiLoCo/
gradient-compression error feedback). A single round doesn't need it; the hook is here for the loop.
"""
import io
import json

import numpy as np

MAGIC = "NHQ8"
VERSION = 1
# Default top-k FRACTION used when compression is turned ON (NEURAHASH_DELTA_TOPK overrides in the
# contributor). 0.003 keeps 0.3% of each tensor's entries: for the 600M-param Qwen trunk that is
# ~1.8M entries at ~3 bytes each -> ~5-9 MB, under the 10 MB push-through-NAT ceiling.
DEFAULT_TOPK = 0.003
_MANIFEST_KEY = "__manifest__"
# Streaming block size for the memory-bounded top-k. EVERY transient the compressor makes is O(this)
# or O(k), never O(numel) -- see quantize_topk's peak-alloc arithmetic. 1<<22 = 4,194,304 elements =
# 16.8 MiB as fp32; a fragmented address space that cannot serve a 594 MiB contiguous block can still
# serve this. (A smaller value trades more passes for a smaller peak; nothing about correctness.)
_CHUNK = 1 << 22


def _to_numpy(t):
    """Accept a numpy array OR a torch tensor; return a contiguous CPU float32 numpy array (a view, not
    a copy, when the source is already a contiguous CPU float32 tensor/array — no duplicate of a big
    trunk tensor on the hot compress path)."""
    if isinstance(t, np.ndarray):
        return np.ascontiguousarray(t, dtype=np.float32)
    if hasattr(t, "detach"):                                    # torch tensor (avoid a hard import)
        try:
            import torch
            if isinstance(t, torch.Tensor):
                return np.ascontiguousarray(t.detach().to("cpu", torch.float32).numpy())
        except ImportError:
            pass
    return np.ascontiguousarray(np.asarray(t, dtype=np.float32))


def _kth_largest_abs_key(a, k):
    """EXACT k-th-largest magnitude of `a`, found in O(_CHUNK) peak memory by a two-pass 16-bit radix
    select streamed in blocks — no full-size |a| copy, no np.partition, no n-length index array.

    Magnitudes are compared as their raw uint32 bit pattern: for finite non-negative float32 that
    pattern is order-preserving (bigger |value| <-> bigger key; +0.0 and -0.0 both map to key 0; a
    NaN, were one present, sorts high — the same end np.partition/np.sort put it). Equal magnitude
    <-> equal key, so a key tie IS a |value| tie. Pass A histograms the HIGH 16 bits of every key to
    find the bucket H that straddles rank k; pass B histograms the LOW 16 bits of the keys inside H
    to pin the exact key. Requires 0 < k < a.size. Returns (T, n_gt, need_eq):
      T       uint32 key of the k-th-largest |value| (t = T.view(float32));
      n_gt    count(|value| > t)  (always < k);
      need_eq k - n_gt >= 1 — how many |value| == t entries are kept to reach EXACTLY k.
    """
    n = a.size
    histA = np.zeros(1 << 16, dtype=np.int64)                       # 512 KiB, reused across chunks
    for s in range(0, n, _CHUNK):
        key = np.abs(a[s:min(s + _CHUNK, n)]).view(np.uint32)       # chunk fp32 copy + zero-copy view
        histA += np.bincount(key >> np.uint32(16), minlength=1 << 16)
    cntA = np.cumsum(histA[::-1])[::-1]                             # cntA[b] = count(high16 >= b)
    H = int(np.count_nonzero(cntA >= k)) - 1                        # largest b with count(>=b) >= k
    n_above = int(cntA[H]) - int(histA[H])                          # elems in strictly-higher buckets
    kk2 = k - n_above                                              # rank of the k-th WITHIN bucket H
    histB = np.zeros(1 << 16, dtype=np.int64)
    Hk = np.uint32(H)
    for s in range(0, n, _CHUNK):
        key = np.abs(a[s:min(s + _CHUNK, n)]).view(np.uint32)
        lo = key[(key >> np.uint32(16)) == Hk] & np.uint32(0xFFFF)  # low 16 bits of the bucket-H keys
        if lo.size:
            histB += np.bincount(lo, minlength=1 << 16)
    cntB = np.cumsum(histB[::-1])[::-1]
    L = int(np.count_nonzero(cntB >= kk2)) - 1
    n_above_B = int(cntB[L]) - int(histB[L])                        # bucket-H elems with low16 > L
    T = np.uint32((H << 16) | L)
    return T, n_above + n_above_B, kk2 - n_above_B


def quantize_topk(arr, topk_fraction):
    """Top-k-by-magnitude int8 quantization of ONE tensor's flattened values.
    Returns (sorted_flat_indices int64, int8_values, scale float). k = max(1, round(fraction*numel)),
    capped at numel. The per-tensor scale is max(|kept|)/127 so the largest kept value maps to +/-127;
    an all-zero delta yields scale=1.0 and zero values (never a divide-by-zero).

    Selection is a MEMORY-BOUNDED exact GLOBAL top-k (semantics identical to the original
    np.argpartition path — the same k largest-|value| entries over the whole flat array, NOT
    per-block). It never materializes a second full-size array: the k-th-largest magnitude t is
    found by a streamed radix histogram (_kth_largest_abs_key), then survivors are gathered by
    scanning |value| block-by-block and appending only the ~k surviving int indices.

    Peak transient allocations (n=155,582,464, k=round(0.003*n)=466,747, _CHUNK=1<<22):
      OLD  np.argpartition int64 permutation       n*8      = 1.16 GiB  (the alloc that OOM'd 2026-07-11)
      OLD  np.partition fp32 threshold copy         n*4      = 594  MiB  (the interim fix's largest alloc)
      OLD  boolean (|a| >= t) mask                  n*1      = 148  MiB
      NEW  per-block |a| fp32 copy                  _CHUNK*4 = 16.8 MiB  (largest single alloc)
      NEW  per-block uint32 radix digit             _CHUNK*4 = 16.8 MiB
      NEW  per-block (key > T) bool mask            _CHUNK*1 =  4.2 MiB
      NEW  surviving indices (n_gt + need_eq = k)   k*8      =  3.6 MiB  (the OUTPUT, O(k) not O(n))
    Every NEW transient is O(_CHUNK) or O(k); NONE is O(n). The largest single request drops from
    1.16 GiB to 16.8 MiB (~1x chunk; ~2x chunk if two blocks are briefly co-resident), so a host
    whose fragmented address space cannot serve one 594 MiB contiguous block still compresses the
    155.6M-element trunk delta. (The k>=n keep-all branch is inherently O(n): its output IS all n
    indices — but that is the value the caller asked for, not a reducible working buffer, and it
    never fires at the miner op-point where topk_fraction<<1.)

    TIE RULE (unchanged): entries whose |value| is strictly greater than the k-th-largest magnitude t
    are ALL kept; the remaining need_eq slots are filled from the tie group (|value| == t) taking the
    LOWEST flat indices first. This yields the SAME |value| multiset as argpartition's top-k (the
    strictly-greater set is unique; every remaining kept entry has magnitude exactly t) — what any
    top-k selection must produce. Determinism is total: the result depends only on (a, k)."""
    a = _to_numpy(arr).reshape(-1)
    n = a.size
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int8), 1.0
    k = min(max(1, int(round(topk_fraction * n))), n)
    if k >= n:
        idx = np.arange(n, dtype=np.int64)                          # keep-all: output is inherently O(n)
    else:
        T, n_gt, need_eq = _kth_largest_abs_key(a, k)               # streamed; O(_CHUNK) peak
        gt_parts, eq_parts, eq_left = [], [], need_eq
        for s in range(0, n, _CHUNK):
            e = min(s + _CHUNK, n)
            key = np.abs(a[s:e]).view(np.uint32)                    # chunk fp32 copy + zero-copy view
            g = np.flatnonzero(key > T)                             # strictly-greater survivors (always kept)
            if g.size:
                gt_parts.append(g + s)
            if eq_left:                                             # fill tie slots, lowest flat index first
                eqp = np.flatnonzero(key == T)[:eq_left]
                if eqp.size:
                    eq_parts.append(eqp + s)
                    eq_left -= int(eqp.size)
        parts = gt_parts + eq_parts
        idx = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        idx = idx.astype(np.int64, copy=False)
        idx.sort()                                                  # union of gt+ties -> ascending (in place)
    vals = a[idx]
    max_abs = float(np.max(np.abs(vals))) if vals.size else 0.0
    if max_abs == 0.0:
        return idx.astype(np.int64, copy=False), np.zeros(idx.size, np.int8), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.round(vals / scale), -127, 127).astype(np.int8)
    return idx.astype(np.int64, copy=False), q, float(scale)


def compress_delta(delta, topk_fraction=DEFAULT_TOPK, *, return_residual=False):
    """Compress a trunk-delta dict {name: tensor(numpy|torch)} to a compact bytes payload (see module
    docstring for the scheme). EVERY key in `delta` is represented (a zero delta stores nnz=0), so the
    key-set round-trips exactly. If return_residual=True, also returns {name: residual ndarray} with
    residual = delta - decompressed (ERROR-FEEDBACK hook)."""
    manifest = {"magic": MAGIC, "version": VERSION, "topk": float(topk_fraction), "tensors": []}
    arrays = {}
    scales = []
    residual = {} if return_residual else None
    for i, name in enumerate(delta):
        a = _to_numpy(delta[name])
        shape = list(a.shape)
        idx, q, scale = quantize_topk(a, topk_fraction)
        if idx.size and int(idx[-1]) >= 2 ** 31:
            raise ValueError(f"tensor {name!r} has >2^31 elements; index encoding overflows")
        gaps = np.diff(idx, prepend=np.int64(0))                    # sorted -> nonneg; cumsum inverts
        # narrowest unsigned width that holds every gap: uint16 for the common dense case (~2 bytes,
        # which DEFLATE squeezes further), uint32 only when a tensor has a >65535-element unselected run.
        gap_dtype = np.uint16 if (gaps.size == 0 or int(gaps.max()) < 2 ** 16) else np.uint32
        arrays[f"g{i}"] = gaps.astype(gap_dtype)
        arrays[f"v{i}"] = q
        scales.append(scale)
        manifest["tensors"].append({"name": name, "shape": shape, "nnz": int(idx.size)})
        if return_residual:
            recon = np.zeros(a.size, dtype=np.float32)
            if idx.size:
                recon[idx] = q.astype(np.float32) * np.float32(scale)
            residual[name] = (a.reshape(-1) - recon).reshape(a.shape)
    arrays["_scales"] = np.asarray(scales, dtype=np.float32)        # one array, not one per tensor
    mbytes = json.dumps(manifest).encode("utf-8")
    arrays[_MANIFEST_KEY] = np.frombuffer(mbytes, dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    payload = buf.getvalue()
    return (payload, residual) if return_residual else payload


def decompress_delta(payload):
    """Inverse of compress_delta. Returns {name: float32 ndarray} of the ORIGINAL shapes (zeros except
    the kept top-k entries, each = int8_value * scale). Loaded with allow_pickle=False so a hostile
    artifact cannot execute code."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        manifest = json.loads(z[_MANIFEST_KEY].tobytes().decode("utf-8"))
        if manifest.get("magic") != MAGIC:
            raise ValueError("bad magic in compressed delta")
        if int(manifest.get("version", 0)) != VERSION:
            raise ValueError(f"unsupported compressed-delta version {manifest.get('version')}")
        scales = z["_scales"]
        out = {}
        for i, meta in enumerate(manifest["tensors"]):
            name = meta["name"]
            shape = tuple(meta["shape"])
            nnz = int(meta["nnz"])
            n = int(np.prod(shape, dtype=np.int64)) if shape else 1
            flat = np.zeros(n, dtype=np.float32)
            if nnz:
                idx = np.cumsum(z[f"g{i}"].astype(np.int64))
                flat[idx] = z[f"v{i}"].astype(np.float32) * np.float32(scales[i])
            out[name] = flat.reshape(shape)
    return out


def save_compressed_delta(path, delta, topk_fraction=DEFAULT_TOPK):
    """Compress `delta` and write it to `path`. Returns the payload size in bytes."""
    payload = compress_delta(delta, topk_fraction)
    with open(path, "wb") as f:
        f.write(payload)
    return len(payload)


def load_compressed_delta(path):
    """Read + decompress a compressed delta file. Returns {name: float32 ndarray}."""
    with open(path, "rb") as f:
        return decompress_delta(f.read())
