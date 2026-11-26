r"""h1_wire_delta -- an EXACT, lossless delta codec for the h1 two-box runner's per-step bf16 slices.

Every H=1 crossing, each node PUTs its OWNED slice (post-AdamW weights, bf16, one safetensors blob) and
GETs the peer's. Node A's uplink carries ~13.6 Mbit/s per connection, so its 28.1 MB slice costs ~16.5 s
a step. Consecutive slices barely differ (53-55% of the bf16 values bit-identical, most others 1-2 ULP),
so this module ships the difference and the receiver rebuilds the next slice BIT-IDENTICALLY from the
previous one. No value is rounded, clipped or re-encoded: training math is untouched.

MEASURED on the live arm's own slices (ns glm47h1e5arms0_r2, fixtures A s0326-s0328, B s0328-s0329):
    node A 28,140,920 B -> 4.36 MB (6.5x) | node B 31,029,072 B -> 4.50 MB (6.9x)
    encode 0.75 s + self-check 0.35 s on the sender, decode 0.35 s on the receiver, single-threaded.
  The ordered (sign-magnitude) int8 ulp-delta of the prototype (anchor_diag_0925/compress_now.py)
  gives 5.52 / 5.75 MB on the same pairs.

THE CODE, per value, u0 = base bf16 bits and u1 = target bf16 bits (uint16):
  d = (u1 - u0) mod 2^16, read as int16. The MODULAR difference, not the prototype's sign-magnitude
      ORDERED difference, because it is a bijection on raw bits: u1 = (u0 + d) mod 2^16 for EVERY pattern
      (+-0, every NaN payload, +-Inf, subnormals, max finite), so -0 needs no special case and no float
      semantics are involved anywhere. Measured it is also smaller: its sign is the sign of the change in
      MAGNITUDE (the ordered one is the change in value), and escapes are identical in number (54,237 vs
      54,237 on A s0326->s0327: every zero-crossing is a large jump in both codes).
  order: the values are coded in a STABLE sort by the base value's exponent field (bits 7-14). An
      AdamW step moves every weight by roughly lr, so the ULP count of a change is set by the weight's
      exponent: sorting by it groups like statistics for the compressor (-19% bytes vs source order:
      A 5.42 -> 4.36 MB). The order depends only on the base, which the decoder holds and has proven by
      sha256, and a stable sort's output is unique -- both sides derive the same permutation.
      (Adding sign, tensor class or lora A/B to the key changed the size by <= 0.8%.)
  stream: one int8 per value, -128 = escape; then, in coding order, the full int16 d of each escaped
      value (|d| > 127, or d = -128). Compressed with xz (lzma preset 1, CRC64); preset 6 is ~6% smaller
      but 9x slower, zlib-9 is larger and 5x slower. The prototype's escape INDEX array is dropped: the
      decoder finds escapes as the -128 bytes (-1.6%).

PAYLOAD (little-endian; struct _HEAD, then the xz stream, then a digest):
  magic 8 B = MAGIC | version u16 | sha256(base) 32 B | sha256(target) 32 B | target length u64 |
  safetensors header length u64 (8-byte prefix + JSON) | n values u64 | n escapes u64 |
  xz stream length u64 | xz stream | sha256 of every byte before it (32 B).
  MAGIC's first 8 bytes read as a safetensors header length are ~1.9e18, so no real slice starts with
  it; the CR/LF/^Z bytes make a text-mode copy of a payload fail loudly (the PNG trick).

GUARANTEES
  encode_delta raises DeltaNotApplicable -- the caller sends the FULL slice -- unless both blobs are
  all-BF16 safetensors with byte-identical headers (so identical names, dtypes, shapes, offsets and
  length) whose tensors tile the data region exactly, AND the payload is strictly smaller than the full
  blob, AND a decode of the payload (its self-check, ~0.35 s) rebuilds the target byte-for-byte. Any
  other encoder fault is also a DeltaNotApplicable: it can cost a full slice, never the run.
  decode_delta returns ONLY a blob whose sha256 equals the target sha256 recorded by the encoder, after
  proving magic, version, the payload's own digest and length, and sha256(base) == the recorded base.
  Anything else -- or any exception inside the decoder -- raises DeltaMismatch; no partial or unverified
  bytes ever leave it. Decompression is bounded to the exact expected size and a 64 MiB memlimit.

numpy + stdlib only (no torch). Tests: scratchpad/test_h1_wire_delta.py.

===========================================================================================================
INTEGRATION NOTE -- scratchpad/glm47_h1x_chain.py. NOT wired in: the live arm (glm47h1e5arms0_r2) runs
that file unchanged. Line numbers are for its sha256 fcc6f2eb9b42...3154874 (3878 lines). NEXT run only.

(0) Version agreement. Ship this file beside the runner on BOTH boxes (the 4060 runs its own copy) and add
      init_meta["wire_codec"] = {"name": "h1wd", "version": VERSION, "keyframe_every": ANCHOR_GC["every"],
                                 "file_sha256": sha256_file(h1_wire_delta.__file__)}
    next to init_meta["anchor_gc"] (:1717). step0_exchange (:1990) VOIDs any key that differs or exists
    on one side only, so a box with no codec, another version or an edited copy fails at step 0, not at
    its first delta. The hash covers every byte, so ship byte-identical copies: a CRLF/LF-converted copy
    also VOIDs at step 0 (loud and cheap, by design).

(1) State: ONE dict PER NODE PER ATTEMPT -- a local made in _train (:1561) just before the step loop (beside
    gc_start, :1753) and passed to fetch_peer_slice as a new argument; never a module global, because the
    selftests drive BOTH nodes through fetch_peer_slice in one process (:3224-3249) and role_train also
    runs in-process (:3546). Never checkpointed:
      WIRE = {"tx": None, "rx": None}
    tx = the last FULL bf16 blob this node SHIPPED: the bytes slice_bytes (:696) returned at the per-step
         PUT, whether they went out whole or as a delta; set only after anchor.put returned.
    rx = the last FULL bf16 blob this node LOADED: what it passed to load_slice (:711), set only after
         load_slice returned. A resume re-runs step0_exchange in a fresh namespace <run_id>_rN (:1615,
         :1718) and loads its local checkpoint only after (:1743), so both start None: the first crossing
         after init OR any --resume-from is always a FULL slice.

(2) Names on the anchor. <ns>/<node>/s<NNNN> still always holds a FULL bf16 safetensors slice; a delta goes
    under <ns>/<node>/d<NNNN> (its payload starts with MAGIC; its base is the SAME node's slice of step
    NNNN-1 as shipped). Exactly one of s/d per (node, step), so s-names become SPARSE (first crossing,
    keyframes, refusals): anything that follows the arm by s-names (e.g. anchor_diag_0925/compress_now.py,
    which needs two consecutive ones) must read d-names too. A name entry's sha256 is the sha256 of the
    object PUT (the payload, for d), so fetch_verified's G3 check (:877) needs no change.
    sync_log (:1808): KEEP own_sha / own_bytes / peer_sha / peer_bytes meaning "the anchor OBJECT this node
    PUT / GOT" -- on a d-step, the PAYLOAD. The registered verdict glm47_h1e5_verdict.py matches them across
    the nodes (V2, :919-921) and against the anchor's /o/<sha12> request lines (anchor_expectations
    :937-957 -> V3, :1023); a full slice's sha never crosses the anchor on a d-step, so logging it there
    would read as missing PUTs/GETs and VOID the arm. ADD own_kind / peer_kind ("s", "d", "d+replay") and
    own_full_sha / own_full_bytes / peer_full_sha / peer_full_bytes for the full bf16 slice; on a d-step its
    sha256 is peek(payload)["target_sha256"], which decode_delta proves against the rebuilt bytes.

(3) Sender -- the per-step PUT, :1803-1804. After  blob, n_up = slice_bytes(surface, owned, a.wire_dtype):
      wire, kind = blob, "s"
      if a.wire_dtype == "bf16" and WIRE["tx"] is not None and step % ANCHOR_GC["every"] != 0:
          try:
              wire, kind = encode_delta(WIRE["tx"], blob), "d"
          except DeltaNotApplicable as e:
              log("  wire: FULL slice at step %d (%s)" % (step, e))
      sha = anchor.put(wire, "%s/%s/%s%04d" % (ns, plan.node, kind, step))    # own_sha: the OBJECT, as today
      WIRE["tx"] = blob
    and at :1808-1813 log own_bytes = len(wire) (not len(blob)), own_kind = kind, own_full_sha =
    sha256_bytes(blob), own_full_bytes = len(blob). A FULL slice goes out on: the first crossing of the
    attempt (tx None), every keyframe step (step % ANCHOR_GC["every"] == 0, see (6)), and every
    DeltaNotApplicable. B's anchor_gc_before_put (:1801) gates this PUT exactly as today, whatever its
    name. --wire-dtype fp32 never deltas.

(4) Receiver -- fetch_peer_slice (:2022), now given WIRE. Wait on [s-name, d-name, void, stop] (:2026;
    wait_names :853 reads ONE manifest per poll, so the extra name is free). pblob = fetch_verified(...) as
    today (:2031).
      s-name: full = pblob; is_delta(pblob) there is a sender bug -> Abort.
      d-name: full = decode_delta(WIRE["rx"], pblob)  (rx None, or not is_delta(pblob), = DeltaMismatch)
    then load_slice(surface, full, peer_names, log) exactly as today (:2032) and WIRE["rx"] = full. Return
    psha and len(pblob) exactly as today (the OBJECT; V2 compares them with the sender's own_sha /
    own_bytes), plus the kind, sha256_bytes(full) and len(full) for the new peer_* fields. A DeltaMismatch
    is NOT a G3 miss (the payload matched its name, so re-fetching it cannot help): log it LOUDLY, then
    (5). Never pass load_slice a blob that did not come out of fetch_verified or decode_delta.

(5) Fallback on DeltaMismatch: rebuild the full slice FROM THE ANCHOR, read-only, no handshake with the
    sender. Take ONE manifest (anchor.manifest(), retried with backoff on a transport error for at most
    fetch_verified's transport_timeout_s (:877), then Abort -- a dead anchor must not hang the receiver)
    and walk back j = step-1, step-2, ... for at most ANCHOR_GC["every"] + 1 steps: if <ns>/<peer>/s<j>
    is in it, fetch_verified that and stop; else if <ns>/<peer>/d<j> is, fetch_verified it and go on;
    neither -> Abort (the chain is broken). Not anchor.resolve: it returns None for a 404 AND for a failed poll
    (wan_gradient_chain_ckpt.py:408-430), so a network blip would read as a broken chain. Then decode
    forward from that full slice through every collected delta up to and including this step's payload
    (kind "d+replay"). Each decode proves its base and target sha256, so the result IS the sender's slice;
    a DeltaMismatch here means the SENDER's chain is bad -> Abort (the step loop's except, :1846-1862,
    publishes the void marker). In a correct run this path never fires (tx and rx hold the same bytes by
    lockstep): if it fires every step, rx is not being updated -- a bug, not noise.

(6) Rolling anchor GC. The normal path never reads a base from the anchor (rx is in RAM); only (5) needs
    one: the newest FULL slice at or before step-1 plus every delta after it. Keyframes at
    step % ANCHOR_GC["every"] == 0 keep that reachable by construction: a pass runs only after a step p
    with p % every == 0 (anchor_gc_due :2215) and drops names below p - keep_steps (:2226), and step p is
    itself a full slice, so no pass can cut a live chain and no chain is longer than `every` steps. With
    --no-anchor-gc nothing is dropped. MANDATORY: extend the GC script's pattern (:2129) to
      r"^(?:A|B)/(?:s|d|gc_s)(\d{4,})$"
    and cover it in the gc selftest -- without it d-names are NEVER dropped, the anchor gains ~4.4 MB per
    node per step, and a PUT into a full anchor dies silently. anchor_gc_peak_bytes (:2239) still counts
    full slices: an over-estimate, safe as is.

(7) Untouched: step0_exchange's fp32 init, the terminal fp32 `final` exchange (:1877-1887), checkpoints
    (WIRE is not saved). Cost with one keyframe per 25 steps: A averages ~5.31 MB/step (5.3x), B ~5.56 MB
    (5.6x); the sender adds ~1.1 s of CPU, the receiver ~0.35 s.
    Selftest to add (MemAnchor, :810): ~60 crossings with ONE WIRE per simulated node, asserting the exact
    name sequence (s0001, d0002..d0024, s0025, d0026, ... -- so it cannot pass without shipping a delta),
    one forced rx corruption to prove (5) rebuilds the slice, and a GC pass that drops d-names.
"""
import hashlib
import json
import lzma
import struct

import numpy as np

__all__ = ["MAGIC", "VERSION", "DeltaNotApplicable", "DeltaMismatch", "encode_delta", "decode_delta",
           "is_delta", "peek"]

MAGIC = b"\x89H1WD\r\n\x1a"
VERSION = 1
# magic | version | sha256(base) | sha256(target) | target length | safetensors header length |
# n values | n escapes | xz stream length
_HEAD = struct.Struct("<8sH32s32sQQQQQ")
_DIGEST = 32                    # trailing sha256 over every payload byte before it
_ESC = -128                     # int8 escape marker; the value's full int16 delta follows the int8 stream
_PRESET = 1                     # xz preset: measured best size per second on real slices
_MEMLIMIT = 64 << 20            # xz decoder memory cap (preset 1 needs ~1 MiB of dictionary)


class DeltaNotApplicable(Exception):
    """encode_delta refused: send the FULL slice instead."""


class DeltaMismatch(Exception):
    """decode_delta refused: nothing trustworthy was rebuilt; fetch the FULL slice instead."""


def is_delta(blob):
    """True iff `blob` starts with MAGIC (a payload of this codec, not a safetensors slice)."""
    return bytes(blob[:len(MAGIC)]) == MAGIC


def peek(payload):
    """The payload's header fields, after checking magic, version, length and its own digest. Proves
    NOTHING about the base or the target -- only decode_delta does."""
    f = _read_head(_as_bytes(payload))
    return {"version": VERSION, "base_sha256": f["base"].hex(), "target_sha256": f["target"].hex(),
            "target_len": f["target_len"], "header_len": f["header_len"], "n_values": f["n_values"],
            "n_escapes": f["n_escapes"], "payload_len": len(payload)}


def encode_delta(prev_blob, next_blob):
    """The payload that rebuilds `next_blob` exactly from `prev_blob`, or DeltaNotApplicable."""
    try:
        return _encode(_as_bytes(prev_blob), _as_bytes(next_blob))
    except DeltaNotApplicable:
        raise
    except Exception as e:                                               # noqa: BLE001
        raise DeltaNotApplicable("encoder error, send the full slice: %r" % (e,)) from e


def decode_delta(prev_blob, payload):
    """`next_blob`, byte-identical and sha256-verified, or DeltaMismatch. Never partial, never unverified."""
    try:
        return _decode(_as_bytes(prev_blob), _as_bytes(payload))
    except DeltaMismatch:
        raise
    except Exception as e:                                               # noqa: BLE001
        raise DeltaMismatch("undecodable payload: %r" % (e,)) from e


# ------------------------------------------------------------------------------------------------ internals
def _as_bytes(x):
    if isinstance(x, bytes):
        return x
    if isinstance(x, (bytearray, memoryview)):
        return bytes(x)
    raise TypeError("expected bytes, got %s" % type(x).__name__)


def _is_uint(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _st_layout(blob):
    """(header_len, n_values) of an all-BF16 safetensors blob whose tensors tile its data region exactly
    (the format's own rule: no holes, no overlaps); ValueError(reason) otherwise."""
    if len(blob) < 8:
        raise ValueError("%d B is shorter than the 8-byte header length" % len(blob))
    n = int.from_bytes(blob[:8], "little")
    if n < 2 or n > len(blob) - 8:
        raise ValueError("header length %d does not fit a %d B blob" % (n, len(blob)))
    try:
        hdr = json.loads(blob[8:8 + n].decode("utf-8"))
    except ValueError as e:                     # UnicodeDecodeError and JSONDecodeError are ValueErrors
        raise ValueError("header is not UTF-8 JSON: %s" % str(e)[:80])
    if not isinstance(hdr, dict):
        raise ValueError("header is not a JSON object")
    data_len = len(blob) - 8 - n
    spans = []
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        if not isinstance(v, dict):
            raise ValueError("tensor %r: entry is not an object" % k)
        if v.get("dtype") != "BF16":
            raise ValueError("tensor %r is %r, not BF16" % (k, v.get("dtype")))
        shape, offs = v.get("shape"), v.get("data_offsets")
        if (not isinstance(shape, list) or not all(_is_uint(x) for x in shape) or not isinstance(offs, list)
                or len(offs) != 2 or not all(_is_uint(x) for x in offs)):
            raise ValueError("tensor %r: malformed shape or data_offsets" % k)
        numel = 1
        for x in shape:
            numel *= x
        s, e = offs
        if not s <= e <= data_len or e - s != 2 * numel:
            raise ValueError("tensor %r spans [%d, %d) of a %d B data region for %d values"
                             % (k, s, e, data_len, numel))
        spans.append((s, e))
    if not spans:
        raise ValueError("no tensors")
    pos = 0
    for s, e in sorted(spans):
        if s != pos:
            raise ValueError("data region has a hole or an overlap at byte %d" % pos)
        pos = e
    if pos != data_len:
        raise ValueError("tensors cover %d of %d data bytes" % (pos, data_len))
    return 8 + n, data_len // 2


def _order(u0):
    """The coding order: a STABLE sort of the base values by their bf16 exponent field (bits 7-14).
    Depends on the base alone; a stable sort's result is unique, so encoder and decoder agree."""
    return np.argsort(((u0 >> 7) & 0xFF).astype(np.uint8), kind="stable")


def _encode(prev_blob, next_blob):
    try:
        hl, n = _st_layout(next_blob)
    except ValueError as e:
        raise DeltaNotApplicable("the new slice is not an all-BF16 safetensors blob: %s" % e)
    if len(prev_blob) != len(next_blob) or prev_blob[:hl] != next_blob[:hl]:
        raise DeltaNotApplicable("the safetensors headers differ (names, dtypes, shapes or offsets)")
    payload = _encode_payload(prev_blob, next_blob, hl, n)
    if len(payload) >= len(next_blob):
        raise DeltaNotApplicable("the delta (%d B) is not smaller than the full slice (%d B)"
                                 % (len(payload), len(next_blob)))
    try:                                        # self-check: an encoder fault costs a full slice, not a VOID
        back = decode_delta(prev_blob, payload)
    except DeltaMismatch as e:
        raise DeltaNotApplicable("self-check failed, send the full slice: %s" % e)
    if back != next_blob:
        raise DeltaNotApplicable("self-check rebuilt different bytes, send the full slice")
    return payload


def _encode_payload(prev_blob, next_blob, hl, n):
    u0 = np.frombuffer(prev_blob, dtype="<u2", count=n, offset=hl)
    u1 = np.frombuffer(next_blob, dtype="<u2", count=n, offset=hl)
    ds = np.subtract(u1, u0, dtype=np.uint16).view(np.int16)[_order(u0)]    # modular delta, coding order
    esc = (ds < -127) | (ds > 127)
    stream = (np.where(esc, np.int16(_ESC), ds).astype(np.int8).tobytes()
              + ds[esc].astype("<i2").tobytes())
    comp = lzma.compress(stream, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=_PRESET)
    body = _HEAD.pack(MAGIC, VERSION, hashlib.sha256(prev_blob).digest(), hashlib.sha256(next_blob).digest(),
                      len(next_blob), hl, n, int(np.count_nonzero(esc)), len(comp)) + comp
    return body + hashlib.sha256(body).digest()


def _read_head(payload):
    if len(payload) < _HEAD.size + _DIGEST:
        raise DeltaMismatch("payload truncated: %d B" % len(payload))
    magic, ver, base, target, tlen, hl, n, n_esc, clen = _HEAD.unpack_from(payload, 0)
    if magic != MAGIC:
        raise DeltaMismatch("not an h1 wire delta (bad magic)")
    if ver != VERSION:
        raise DeltaMismatch("h1 wire delta version %d, this codec reads %d" % (ver, VERSION))
    if len(payload) != _HEAD.size + clen + _DIGEST:
        raise DeltaMismatch("payload is %d B but its header implies %d B"
                            % (len(payload), _HEAD.size + clen + _DIGEST))
    if hashlib.sha256(memoryview(payload)[:-_DIGEST]).digest() != payload[-_DIGEST:]:
        raise DeltaMismatch("payload digest mismatch: corrupted in transit or at rest")
    return {"base": base, "target": target, "target_len": tlen, "header_len": hl, "n_values": n,
            "n_escapes": n_esc, "stream_len": clen}


def _decode(prev_blob, payload):
    f = _read_head(payload)
    if hashlib.sha256(prev_blob).digest() != f["base"]:
        raise DeltaMismatch("base mismatch: the delta was made against a different slice than the one held here")
    hl, n, n_esc = f["header_len"], f["n_values"], f["n_escapes"]
    if (f["target_len"] != len(prev_blob) or not 8 < hl <= len(prev_blob) or len(prev_blob) - hl != 2 * n
            or n_esc > n):
        raise DeltaMismatch("payload layout does not fit the base")
    want = n + 2 * n_esc
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=_MEMLIMIT)
    stream = dec.decompress(memoryview(payload)[_HEAD.size:_HEAD.size + f["stream_len"]], max_length=want + 1)
    if len(stream) != want or not dec.eof or dec.unused_data:
        raise DeltaMismatch("xz stream gave %d B (eof %s), expected exactly %d B" % (len(stream), dec.eof, want))
    a8 = np.frombuffer(stream, dtype=np.int8, count=n)
    esc = a8 == _ESC
    if int(np.count_nonzero(esc)) != n_esc:
        raise DeltaMismatch("stream has %d escapes, header says %d" % (int(np.count_nonzero(esc)), n_esc))
    ds = a8.astype(np.int16)
    ds[esc] = np.frombuffer(stream, dtype="<i2", count=n_esc, offset=n)
    vals = _rebuild(np.frombuffer(prev_blob, dtype="<u2", count=n, offset=hl), ds)
    out = prev_blob[:hl] + vals.astype("<u2", copy=False).tobytes()
    if len(out) != f["target_len"] or hashlib.sha256(out).digest() != f["target"]:
        raise DeltaMismatch("rebuilt slice sha256 != the payload's target sha256: refusing it")
    return out


def _rebuild(u0, ds):
    """Target values: undo the coding order, then add the modular delta (uint16 wraps mod 2^16). A
    separate function so the positive-control test can corrupt its output."""
    d = np.empty(ds.shape, dtype=np.int16)
    d[_order(u0)] = ds
    return np.add(u0, d.view(np.uint16), dtype=np.uint16)
