"""
Safe wire codec — replaces pickle on the network to remove the remote-code-execution
risk the audit flagged.

pickle.loads can execute arbitrary code from a malicious peer. This codec instead
serializes only an explicit allowlist of types — dict / list / tuple / str / int /
float / bool / None / numpy arrays (numeric dtypes only) / the MoELM model — as
JSON metadata plus raw little-endian array buffers. Decoding constructs ONLY those
types, so a hostile frame cannot run code. An optional pre-shared HMAC key
authenticates and integrity-checks every frame.

Frame: [4-byte meta-len][meta json][ (8-byte len + bytes) per array buffer ].
On the wire each frame is length-prefixed, optionally preceded by a 32-byte HMAC.
"""

import json
import os
import struct
import hmac
import hashlib
import numpy as np

from .model import MoELM

_ALLOWED_DTYPES = {"float64", "float32", "float16", "int64", "int32", "int16", "int8", "uint8", "bool"}
# float16 / int8 carry the compressed wire (neurahash/wire_compress.py, #92 follow-up): a fp16 trunk/delta
# ships as a float16 ndarray and an int8-quantized tensor as an int8 ndarray + a float32 scale. Both are
# just numeric ndarrays through this codec (byte-exact via .tobytes()), so a capable peer reconstructs the
# EXACT quantized values the sender produced — the property the recompute-verify reference-equality needs.
_MAX_DEPTH = 64                 # nesting cap -> no recursion-crash DoS
# Allocation guards (#18). The defaults suit toy/MoE pools; a DENSE base (Rung-2) pushes the whole model
# per round (0.6B->600M elems, 1.7B->1.7B elems), so a single-machine dense run raises _MAX_ELEMS via
# NEURAHASH_MAX_ELEMS. Keep the modest defaults on a networked pool — a huge cap off-loopback is a real
# memory-amplification DoS surface.
_MAX_PARAMS = int(os.environ.get("NEURAHASH_MAX_PARAMS", "200000000"))   # cap reconstructed model size
_MAX_NODES = int(os.environ.get("NEURAHASH_MAX_NODES", "5000000"))       # total decoded nodes (#18)
_MAX_ELEMS = int(os.environ.get("NEURAHASH_MAX_ELEMS", "400000000"))     # total array elements (#18)
from .model import V as _VOCAB


def _moelm_param_estimate(hp):
    C, Demb, H, Do, E = hp["C"], hp["Demb"], hp["H"], hp["Do"], hp["E"]
    Din = C * Demb
    return _VOCAB * Demb + Din * E + E * (Din * H + H + H * Do + Do) + Do * _VOCAB


# ---------------------------- encode ----------------------------
def _enc(o, buffers):
    if isinstance(o, np.ndarray):
        if o.dtype.name not in _ALLOWED_DTYPES:
            raise ValueError(f"disallowed array dtype: {o.dtype.name}")
        arr = np.ascontiguousarray(o)
        buffers.append(arr.tobytes())
        return {"$nd": len(buffers) - 1, "shape": list(arr.shape), "dtype": arr.dtype.name}
    if isinstance(o, MoELM):
        return {"$moelm": {"hp": {"C": o.C, "Demb": o.Demb, "H": o.H, "Do": o.Do,
                                  "E": o.E},
                           "p": _enc({k: v for k, v in o.p.items()}, buffers)}}
    if isinstance(o, dict):
        return {"$d": {str(k): _enc(v, buffers) for k, v in o.items()}}
    if isinstance(o, tuple):
        return {"$t": [_enc(v, buffers) for v in o]}
    if isinstance(o, list):
        return {"$l": [_enc(v, buffers) for v in o]}
    if isinstance(o, (str, int, float, bool)) or o is None:
        return {"$v": o}
    raise ValueError(f"unserializable type: {type(o)}")


def encode_msg(obj, key=None):
    buffers = []
    meta = _enc(obj, buffers)
    meta_bytes = json.dumps(meta).encode("utf-8")
    out = bytearray(struct.pack("!I", len(meta_bytes)))
    out += meta_bytes
    for b in buffers:
        out += struct.pack("!Q", len(b)) + b
    frame = bytes(out)
    if key is not None:
        frame = hmac.new(key, frame, hashlib.sha256).digest() + frame
    return frame


# ---------------------------- decode ----------------------------
def _dec(node, buffers, depth=0, budget=None):
    if depth > _MAX_DEPTH:
        raise ValueError("max nesting depth exceeded")
    if budget is not None:                       # NODE-COUNT cap: a small frame can encode millions of
        budget["nodes"] += 1                     # tiny nodes that each expand to a Python object — bound
        if budget["nodes"] > _MAX_NODES:         # the total so a tiny frame can't amplify into an OOM (#18)
            raise ValueError("node budget exceeded (allocation guard)")
    if not isinstance(node, dict) or len(node) == 0:
        raise ValueError("malformed node")
    if "$v" in node:
        return node["$v"]
    if "$nd" in node:
        dtype = node.get("dtype")
        shape = node.get("shape")
        idx = node.get("$nd")
        if dtype not in _ALLOWED_DTYPES:
            raise ValueError(f"disallowed array dtype: {dtype}")
        if (not isinstance(shape, list) or not isinstance(idx, int) or
                not (0 <= idx < len(buffers))):
            raise ValueError("malformed $nd metadata")
        if any((not isinstance(d, int)) or d < 0 for d in shape):
            raise ValueError("invalid array shape")
        buf = buffers[idx]
        arr = np.frombuffer(buf, dtype=dtype)
        count = 1
        for d in shape:
            count *= d
        if count != arr.size:                    # shape must match buffer exactly
            raise ValueError("shape/buffer size mismatch")
        if budget is not None:                    # ELEMENT cap: bound total decoded array elements (#18)
            budget["elems"] += int(count)
            if budget["elems"] > _MAX_ELEMS:
                raise ValueError("element budget exceeded (allocation guard)")
        return arr.reshape(shape).copy()
    if "$d" in node:
        return {k: _dec(v, buffers, depth + 1, budget) for k, v in node["$d"].items()}
    if "$t" in node:
        return tuple(_dec(v, buffers, depth + 1, budget) for v in node["$t"])
    if "$l" in node:
        return [_dec(v, buffers, depth + 1, budget) for v in node["$l"]]
    if "$moelm" in node:
        hp = node["$moelm"]["hp"]
        for f in ("C", "Demb", "H", "Do", "E"):
            if not isinstance(hp.get(f), int) or not (1 <= hp[f] <= 100000):
                raise ValueError(f"invalid MoELM hyperparameter {f}")
        if _moelm_param_estimate(hp) > _MAX_PARAMS:
            raise ValueError("MoELM too large to reconstruct (allocation guard)")
        m = MoELM(C=hp["C"], Demb=hp["Demb"], H=hp["H"], Do=hp["Do"], n_experts=hp["E"])
        m.set_params(_dec(node["$moelm"]["p"], buffers, depth + 1, budget))
        return m
    raise ValueError(f"unknown tag: {list(node)[0]}")


def decode_msg(data, key=None):
    # authenticate FIRST (before parsing any untrusted JSON) when a key is set
    if key is not None:
        if len(data) < 32:
            raise ValueError("frame too short for HMAC")
        mac, data = data[:32], data[32:]
        if not hmac.compare_digest(mac, hmac.new(key, data, hashlib.sha256).digest()):
            raise ValueError("HMAC authentication failed")
    if len(data) < 4:
        raise ValueError("frame too short")
    (meta_len,) = struct.unpack("!I", data[:4])
    if meta_len > len(data) - 4:
        raise ValueError("meta length exceeds frame")
    try:
        meta = json.loads(data[4:4 + meta_len].decode("utf-8"))
    except (ValueError, RecursionError) as e:
        raise ValueError(f"malformed frame: {e}")
    buffers, off = [], 4 + meta_len
    while off + 8 <= len(data):
        (blen,) = struct.unpack("!Q", data[off:off + 8])
        off += 8
        if blen > len(data) - off:
            raise ValueError("buffer length exceeds frame")
        buffers.append(data[off:off + blen])
        off += blen
    try:
        return _dec(meta, buffers, budget={"nodes": 0, "elems": 0})
    except RecursionError:
        raise ValueError("nesting too deep")
