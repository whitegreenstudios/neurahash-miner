"""
neurahash/wire_compress.py — capability-gated COMPRESSED WIRE for the sharded pool (#92 follow-up).

WHY: a thin-pipe miner (a Colab T4 behind a ~25 Mbps VPS relay) is DROPPED every session at the
medium rung because the per-round payloads are raw fp32 numpy — the full trunk goes DOWN to every
worker each round, a reassign additionally pushes experts+trunk, and the worker's result delta comes
back equally large. At the medium rung (d_model=512, 8 layers) that is ~90 MB one-way per round, which
exceeds the 60 s bounded-send budget (#92) through a ~25 Mbps pipe: the worker joins, mines 0-2 rounds,
and the session dies ~90-140 s into the big send, forever. Shrinking those arrays 2x (fp16) or 4x
(int8) puts them back inside the budget.

THE CORRECTNESS INVARIANT (the pool verifies work by RECOMPUTE — the coordinator replays a worker's
trajectory from the same start state and compares deltas with cosine >= VERIFY_COS + a norm band). If a
worker trains from a QUANTIZED trunk, the coordinator MUST recompute from the SAME dequantized values,
or the honest delta diverges and the honest worker is rejected. This module makes that exact by two
properties, both relied on at the call sites in sharded_pool_node.py:

  (P1) LOSSLESS TRANSPORT of the quantized values. fp16 arrays and (int8 array + fp32 scale) pairs cross
       safe_codec byte-for-byte (safe_codec ships numeric ndarrays via .tobytes(); we add float16/int8 to
       its dtype allowlist). So the exact quantized values the SENDER produced are what the RECEIVER sees.

  (P2) DETERMINISTIC dequantization. dequantize_state() is a pure function of the wire bytes: fp16 is
       float16->float32 (IEEE, deterministic); int8 is int8->float32 then *scale (same inputs => same
       IEEE result on both sides). Therefore

           dequantize_state(quantize_state(x, mode))            # what the SENDER keeps as its reference
             ==  (bit-identical)                                # SAME float32 arrays on both machines
           dequantize_state(wire_received)                      # what the RECEIVER reconstructs

       This is the reference-equality the coordinator relies on: it quantizes the trunk ONCE, keeps
       `dequantize_state(quantize_state(trunk))` as the canonical start-state for that worker's recompute,
       sends the quantized form, and the worker reconstructs the IDENTICAL start-state. The recompute
       therefore starts from exactly the worker's view -> honest cosine ~1.0, unchanged.

HASH / BEACON / WORK-SIG consequences (each traced in sharded_pool_node.py; summarized here):
  * Round beacon  H(round || trunk_state_hash(trunk_np))  is derived by the coordinator from the CANONICAL
    fp32 trunk and the worker ECHOES it (it never re-derives the beacon from received trunk bytes — see
    run_worker: `beacon = msg.get("beacon", "")`). So the beacon stays a SINGLE value over canonical fp32,
    sent identically to capable and legacy workers alike; mixed fleets echo the same token. No per-worker
    beacon is needed, and the beacon representation is UNAFFECTED by compression.
  * Work signature over (addr, round, beacon, trunk_delta_hash(delta)) and the DeltaReplayCache both hash
    the delta with trunk_delta_hash. Both sides hash the DEQUANTIZED delta (== bit-identical by P2, and
    always float32 so the dtype byte in the hash is stable): the worker downcasts its fp32 delta to the
    wire dtype, computes its reference = dequantize_state(that wire form), and hashes/signs THAT; the
    coordinator receives the wire form, computes the identical reference, hashes it -> the signature
    matches. A delta swapped in transit still fails (the received bytes hash differently).

MODES:
  * "fp16" — each tensor -> numpy float16 ndarray (2x). ~1e-3 relative error, far inside the cosine
    tolerance (VERIFY_COS 0.92 vs honest ~1.0).
  * "int8" — each tensor -> {"q": int8 ndarray, "s": float32 per-tensor absmax scale} (4x). The absmax
    scale pattern is the one proven in glm52_fleet/pipeline_compress_test.py (`int8()`), per-tensor so a
    tensor's own dynamic range sets its step; a zero/tiny tensor uses a clamped scale so 0/0 never occurs.

Everything here is PURE (no torch, no network, no env). The DEFAULT-OFF / capability-gated rollout lives
at the call sites: an old worker never advertises `wire_compress`, so the coordinator sends it raw fp32
and verifies it against raw fp32 exactly as before; an old coordinator never sets NEURAHASH_WIRE_COMPRESS
and the wire is byte-identical to today.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "WIRE_CAP", "WIRE_MODES", "normalize_mode",
    "quantize_state", "dequantize_state", "quantize_array", "dequantize_array",
    "wire_nbytes",
]

# The hello capability key a compression-aware worker advertises (additive; absent == old/unaware
# client, mirrors storage_wire.STORAGE_CAP). The value is the list of modes the worker can DEQUANTIZE.
WIRE_CAP = "wire_compress"
WIRE_MODES = ("fp16", "int8")

# Tag stored in the wire dict so dequantize_state is self-describing (a received frame carries its own
# mode; the receiver never has to be told out-of-band which codec produced it). Reserved key that cannot
# collide with a tensor name (tensor names are dotted module paths like "blocks.0.attn.qkv.weight").
_MODE_KEY = "$wire"
# int8 clamp so a zero / near-zero tensor gets a finite, nonzero scale (0/0 -> nan otherwise). Matches
# the clamp(min=1e-8) in glm52_fleet/pipeline_compress_test.py's int8().
_ABSMAX_FLOOR = 1e-8


def normalize_mode(mode):
    """Normalize an env/hello string to a supported mode or 'off'. Unknown/empty -> 'off' (fail-safe:
    an unrecognized value never silently corrupts the wire — it just stays uncompressed)."""
    m = (str(mode) if mode is not None else "").strip().lower()
    if m in ("", "0", "off", "false", "no", "none"):
        return "off"
    return m if m in WIRE_MODES else "off"


# --------------------------------------------------------------------------- per-array codec
def quantize_array(a, mode):
    """One fp32 (or fp32-castable) ndarray -> its wire form for `mode`.
      fp16 : a float16 ndarray (safe_codec ships it byte-exact).
      int8 : {"q": int8 ndarray, "s": float32 scalar absmax/127 scale}. Per-tensor absmax; a zero/tiny
             tensor gets the clamped floor scale so q is all-zeros and dequant returns ~zeros (no nan).
    """
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    if mode == "fp16":
        return a.astype(np.float16)
    if mode == "int8":
        # per-tensor absmax scale: s = max(|a|) / 127. Stored as a PLAIN Python float (safe_codec ships
        # scalars via JSON $v and rejects numpy scalar types); np.float32(s) below makes the compute
        # deterministic on both sides from the SAME float value, so the dequant is bit-identical (P2).
        amax = float(np.abs(a).max()) if a.size else 0.0
        s = float(np.float32(max(amax, _ABSMAX_FLOOR) / 127.0))   # round the scale to float32 precision once
        # round-half-to-even (numpy default) then clamp into int8 range; identical on both machines.
        q = np.clip(np.rint(a / np.float32(s)), -127, 127).astype(np.int8)
        return {"q": np.ascontiguousarray(q), "s": s}
    raise ValueError(f"unknown wire mode: {mode!r}")


def dequantize_array(w, mode):
    """Inverse of quantize_array: wire form -> a float32 ndarray. Deterministic (a pure function of the
    wire bytes), so the value is bit-identical to the one the sender kept as its reference (P2)."""
    if mode == "fp16":
        return np.ascontiguousarray(np.asarray(w, dtype=np.float16).astype(np.float32))
    if mode == "int8":
        q = np.asarray(w["q"], dtype=np.int8).astype(np.float32)
        s = np.float32(w["s"])
        return np.ascontiguousarray(q * s)
    raise ValueError(f"unknown wire mode: {mode!r}")


# --------------------------------------------------------------------------- state-dict codec
def quantize_state(state, mode):
    """A {name: fp32 ndarray} dict -> a self-describing wire dict {name: wireform, _MODE_KEY: mode}.
    mode 'off' (or unknown) returns the arrays untouched as float32 (a caller that passes 'off' gets the
    byte-identical raw payload — the call sites only reach here when compression is actually on, but this
    keeps the helper total)."""
    m = normalize_mode(mode)
    if m == "off":
        return {k: np.ascontiguousarray(np.asarray(v, dtype=np.float32)) for k, v in state.items()}
    out = {_MODE_KEY: m}
    for k, v in state.items():
        out[k] = quantize_array(v, m)
    return out


def dequantize_state(wire):
    """A wire dict from quantize_state -> {name: fp32 ndarray}. Reads the embedded _MODE_KEY so it is
    self-describing; a plain {name: ndarray} dict with no tag (an uncompressed / legacy payload) is
    returned as float32 unchanged. Bit-identical to what the sender kept as reference (P2)."""
    if not isinstance(wire, dict):
        raise ValueError("wire payload must be a dict")
    if _MODE_KEY not in wire:                       # untagged -> raw fp32 (legacy / off), pass through
        return {k: np.ascontiguousarray(np.asarray(v, dtype=np.float32)) for k, v in wire.items()}
    m = normalize_mode(wire[_MODE_KEY])
    if m == "off":
        raise ValueError("tagged wire dict carries an invalid mode")
    return {k: dequantize_array(v, m) for k, v in wire.items() if k != _MODE_KEY}


def wire_nbytes(wire):
    """Total transmitted array bytes of a wire dict (for size measurement / logging). Counts int8 q + the
    4-byte scale per tensor, or the float16 buffer, or a raw float32 buffer — mirrors what safe_codec puts
    on the wire (the small JSON metadata is not counted; it is negligible next to the buffers)."""
    n = 0
    for k, v in wire.items():
        if k == _MODE_KEY:
            continue
        if isinstance(v, dict) and "q" in v:                       # int8 pair
            n += int(np.asarray(v["q"]).nbytes) + int(np.float32(v["s"]).nbytes)
        else:
            n += int(np.asarray(v).nbytes)
    return n
