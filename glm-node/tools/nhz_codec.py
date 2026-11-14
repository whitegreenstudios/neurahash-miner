"""NHZ1: a LOSSLESS compressed container for the GLM expert packs. stdlib + numpy only (no torch).

WHY
---
GLM-4.7-Flash training streams every MoE layer's frozen routed experts from disk twice per optimizer
step: 46 packs x 1.125 GiB of bf16 (`layer_N.safetensors`), 103.5 GiB per step per node. The 5090 box's
pack drive (E:) gives ~1 GB/s under that load, and only ~20 GB of page cache is free, so most bytes are
re-read every step. Real miners (8-16 GB cards, 16-32 GB RAM, consumer SSDs) have the same problem.
Fewer bytes on disk means fewer bytes read per miss AND more layers held by the same page cache, because
the OS caches the COMPRESSED file.

WHAT IS COMPRESSIBLE (measured on real layer_10, 2026-09-24, bench_primitives.py)
---------------------------------------------------------------------------------
A bf16 word is [sign | 8 exponent bits | 7 mantissa bits]. Stored little-endian, byte 1 holds the sign
plus the top 7 exponent bits and byte 0 the lowest exponent bit plus the mantissa. Entropy: high byte
2.707 of 8 bits, low byte 7.970 (incompressible), and the high byte's order-1 conditional entropy is also
2.707 -- adjacent weights are independent, so no context model can beat order 0. The bound for "code the
high byte, store the low byte" is 16 / (2.707 + 8) = 1.494x.

THE CODEC -- per chunk of CHUNK_WORDS words; chunks are independent, so they decode in parallel threads
  high-byte plane -> raw deflate with Z_HUFFMAN_ONLY: 2.745 bits/word, 1.4% above the entropy. zlib's LZ77
                     match search only HURTS on this data (level 1: 3.555 bits; level 6: 3.286 bits at
                     6 MB/s encode). lzma (3.44 bits, 60 MB/s decode) and bz2 (3.13, 39 MB/s) lose too.
  low-byte plane  -> stored raw
  decode          -> inflate (zlib releases the GIL) + interleave (a numpy strided copy, GIL released)
                     + crc32 of the decoded bytes (GIL released)
BF16 is the only dtype that gets the byte split; every other dtype, and any chunk whose high plane does
not shrink, is stored raw -- still chunked, still checksummed.

FILE LAYOUT
  [0:32)  fixed prefix  <8s magic | u32 version | u32 header_len | u64 file_size | u32 header_crc32 | u32 0>
  [32:32+header_len)   header JSON (UTF-8)
  zero padding up to data_start (a multiple of 4096)
  chunk payloads, back to back, in table order:  MODE_HILO = [raw deflate of the high bytes][low bytes]
                                                 MODE_RAW  = [the original bytes]
  file_size == the end of the last payload; anything else is a named error.
HEADER JSON: format/version, codec parameters, chunk_bytes, the SOURCE (basename, size, its safetensors
header bytes base64 + sha256, whole-file sha256 -- enough to rebuild the source byte-for-byte), and per
tensor in source data order: name, dtype, shape, nbytes, st_data_offsets, sha256 of the original bytes,
and the chunk table [file_offset, stored_len, out_len, crc32, mode].

INTEGRITY
  * header: magic, version, its own crc32, a recorded file size (truncation is caught before any data
    is read), and a structural validation of every chunk entry (contiguous, in bounds, sizes consistent);
  * every chunk: crc32 over its DECODED bytes -- which are exactly the original bytes -- seeded with
    crc32(b"nhz1|<tensor index>|<chunk index>"), so a flipped bit in either plane, a decode bug, a torn
    write, or two swapped table entries all fail AT THAT CHUNK;
  * every tensor: sha256 of the original bytes, checked on request (verify="sha"), or on the first decode
    of a file in this process (verify="first").
Every failure is an NhzError subclass with the file, tensor and chunk in its message; nothing continues.
"""
import base64
import hashlib
import json
import os
import struct
import sys
import threading
import time
import zlib

import numpy as np

FORMAT = "nhz1"
VERSION = 1
MAGIC = b"NHZ1\r\n\x1a\n"                  # the \r\n / \x1a / \n bytes catch text-mode and truncating copies
_PREFIX = struct.Struct("<8sIIQII")
PREFIX_LEN = _PREFIX.size                  # 32
DATA_ALIGN = 4096
CHUNK_WORDS = 1 << 20                      # 1 Mi bf16 words = 2 MiB of original bytes per chunk
CHUNK_BYTES_MAX = 64 << 20                 # a header may not ask a decode thread for a bigger staging buffer
HEADER_MAX = 64 << 20
ST_HEADER_MAX = 100 << 20
ZLIB_LEVEL = 1                             # irrelevant under Z_HUFFMAN_ONLY; recorded for completeness
ZLIB_STRATEGY = zlib.Z_HUFFMAN_ONLY
ZLIB_WBITS = -15                           # raw deflate: integrity is the chunk crc32 over DECODED bytes
MODE_RAW = 0
MODE_HILO = 1
HILO_DTYPES = ("BF16",)
ITEMSIZE = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
            "F32": 4, "I64": 8, "U64": 8, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1}
VERIFY_MODES = ("crc", "first", "sha")
THREADS_MAX = 32


# ============================================================================ named errors
class NhzError(Exception):
    """Base class: something about an nhz file's CONTENT is wrong. Never caught-and-continued here."""


class NhzFormatError(NhzError, ValueError):
    """Not an nhz1 file, a corrupt/inconsistent header, a trailing byte, or an unknown tensor name."""


class NhzVersionError(NhzFormatError):
    """The magic is right but the format version is not one this module can read."""


class NhzTruncatedError(NhzError, EOFError):
    """The file is shorter than its own header says (or a read came back short)."""


class NhzChunkError(NhzError):
    """A chunk failed to inflate, inflated to the wrong length, or its crc32 does not match."""


class NhzShaMismatch(NhzError):
    """A decoded tensor's sha256 is not the one recorded from the source bytes."""


class NhzMissingError(NhzError, FileNotFoundError):
    """An nhz file that was asked for does not exist."""


class NhzStaleError(NhzError):
    """The nhz file does not describe the raw safetensors pack next to it (size or header differ)."""


def _bn(path):
    return os.path.basename(str(path))


def chunk_seed(ti, ci):
    """The crc32 start value of chunk ci of tensor ti: makes every chunk's checksum POSITION-dependent."""
    return zlib.crc32(b"nhz1|%d|%d" % (int(ti), int(ci)))


def file_stamp(path):
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


# ============================================================================ safetensors side
def read_st_header(path):
    """(header dict, header bytes, data_start) of a safetensors file, with the reader's own checks."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise EOFError("%s: file is shorter than a safetensors header" % _bn(path))
        n = struct.unpack("<Q", raw)[0]
        if not 0 < n <= ST_HEADER_MAX:
            raise ValueError("%s: safetensors header length %d out of range" % (_bn(path), n))
        hb = fh.read(n)
        if len(hb) != n:
            raise EOFError("%s: safetensors header is truncated" % _bn(path))
    return json.loads(hb.decode("utf-8")), hb, 8 + n


def source_identity(st_path):
    """(size, sha256 of the safetensors header bytes) -- what an nhz file records about its source."""
    _h, hb, _b = read_st_header(st_path)
    return os.path.getsize(st_path), hashlib.sha256(hb).hexdigest()


# ============================================================================ the header
class TensorEntry(object):
    __slots__ = ("index", "name", "dtype", "shape", "nbytes", "st_offsets", "sha256", "chunks")

    def __repr__(self):
        return "TensorEntry(%r, %s, %s, %d bytes, %d chunks)" % (self.name, self.dtype, self.shape,
                                                                 self.nbytes, len(self.chunks))


class NhzHeader(object):
    """A parsed, VALIDATED nhz1 header. Tensors are in source data order (`_st_read`'s copy order)."""

    def __init__(self, path, meta, header_len, file_size, data_start, stamp):
        self.path = path
        self.meta = meta
        self.header_len = header_len
        self.file_size = file_size
        self.data_start = data_start
        self.stamp = stamp
        self.version = meta["version"]
        self.chunk_bytes = meta["chunk_bytes"]
        self.source = meta.get("source") or {}
        self.tensors = []
        self.by_name = {}

    def entry(self, name):
        e = self.by_name.get(name)
        if e is None:
            raise NhzFormatError("%s: no tensor %r in this file (it has %s)"
                                 % (_bn(self.path), name, sorted(self.by_name)))
        return e

    def order(self, names):
        """`names` in source data order."""
        return sorted(names, key=lambda k: self.entry(k).st_offsets[0])

    def layout(self, names, align=DATA_ALIGN):
        """([(name, dtype, shape, buffer_offset, nbytes)] in source data order, total bytes): each tensor
        starts at an `align`-aligned offset of ONE host buffer -- the placement `_st_read_into` uses."""
        out, at = [], 0
        for name in self.order(names):
            e = self.by_name[name]
            at = -(-at // align) * align
            out.append((name, e.dtype, list(e.shape), at, e.nbytes))
            at += e.nbytes
        return out, at

    def decoded_bytes(self, names=None):
        return sum(self.entry(n).nbytes for n in (names if names is not None else self.by_name))

    def stored_bytes(self, names=None):
        return sum(c[1] for n in (names if names is not None else self.by_name) for c in self.entry(n).chunks)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _validate(path, meta, header_len, file_size, stamp):
    """Structural validation: after this, no chunk entry can point outside its tensor or the file."""
    bn = _bn(path)

    def bad(msg):
        raise NhzFormatError("%s: invalid nhz1 header: %s" % (bn, msg))
    if not isinstance(meta, dict):
        bad("not a JSON object")
    if meta.get("format") != FORMAT:
        bad("format %r, want %r" % (meta.get("format"), FORMAT))
    if meta.get("version") != VERSION:
        raise NhzVersionError("%s: header version %r, this reader knows %d" % (bn, meta.get("version"), VERSION))
    cb, ds = meta.get("chunk_bytes"), meta.get("data_start")
    if not (_is_int(cb) and 0 < cb <= CHUNK_BYTES_MAX and cb % 2 == 0):
        bad("chunk_bytes %r (want an even number up to %d)" % (cb, CHUNK_BYTES_MAX))
    if not (_is_int(ds) and ds % DATA_ALIGN == 0 and ds >= PREFIX_LEN + header_len):
        bad("data_start %r (header ends at %d)" % (ds, PREFIX_LEN + header_len))
    tens = meta.get("tensors")
    if not isinstance(tens, list):
        bad("no tensor list")
    h = NhzHeader(path, meta, header_len, file_size, ds, stamp)
    at = ds
    for ti, t in enumerate(tens):
        if not isinstance(t, dict):
            bad("tensor %d is not an object" % ti)
        name, dt, shape, nb = t.get("name"), t.get("dtype"), t.get("shape"), t.get("nbytes")
        if not isinstance(name, str) or name in h.by_name or name == "__metadata__":
            bad("tensor %d name %r" % (ti, name))
        if not isinstance(dt, str) or dt not in ITEMSIZE:      # type first: a list is unhashable
            bad("tensor %r dtype %r" % (name, dt))
        if not (isinstance(shape, list) and len(shape) <= 16 and all(_is_int(s) and 0 <= s < (1 << 48) for s in shape)):
            bad("tensor %r shape (rank %s)" % (name, len(shape) if isinstance(shape, list) else type(shape).__name__))
        want = ITEMSIZE[dt]
        for s in shape:
            want *= s
        if nb != want:
            bad("tensor %r nbytes %r != prod(shape) * itemsize = %d" % (name, nb, want))
        sha = t.get("sha256")
        if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
            bad("tensor %r sha256 %r" % (name, sha))
        so = t.get("st_data_offsets")
        if not (isinstance(so, list) and len(so) == 2 and all(_is_int(x) for x in so) and so[1] - so[0] == nb):
            bad("tensor %r st_data_offsets %r" % (name, so))
        chunks = t.get("chunks")
        if not isinstance(chunks, list):
            bad("tensor %r chunk table" % name)
        total = 0
        rows = []
        for ci, c in enumerate(chunks):
            if not (isinstance(c, list) and len(c) == 5 and all(_is_int(x) for x in c)):
                bad("tensor %r chunk %d entry %r" % (name, ci, c))
            off, stored, out_len, crc, mode = c
            if off != at:
                bad("tensor %r chunk %d at offset %d, expected %d (payloads are contiguous)" % (name, ci, off, at))
            if not 0 < out_len <= cb:
                bad("tensor %r chunk %d decodes to %d bytes (chunk_bytes %d)" % (name, ci, out_len, cb))
            if ci < len(chunks) - 1 and out_len != cb:
                bad("tensor %r chunk %d is short (%d of %d) but not the last" % (name, ci, out_len, cb))
            if not 0 <= crc <= 0xFFFFFFFF:
                bad("tensor %r chunk %d crc %r" % (name, ci, crc))
            if mode == MODE_RAW:
                if stored != out_len:
                    bad("tensor %r raw chunk %d stores %d bytes for %d" % (name, ci, stored, out_len))
            elif mode == MODE_HILO:
                if dt not in HILO_DTYPES or out_len % 2:
                    bad("tensor %r chunk %d is hilo but dtype %s / %d bytes" % (name, ci, dt, out_len))
                if not 1 <= stored - out_len // 2 < out_len // 2:
                    # the encoder stores a chunk RAW unless its high stream is SMALLER than the plane, so a hilo
                    # chunk is always shorter than it decodes to -- which also bounds every staging buffer
                    bad("tensor %r hilo chunk %d stores %d bytes for %d words" % (name, ci, stored, out_len // 2))
            else:
                bad("tensor %r chunk %d mode %r" % (name, ci, mode))
            total += out_len
            at += stored
            rows.append((off, stored, out_len, crc, mode))
        if total != nb:
            bad("tensor %r chunks decode to %d bytes, want %d" % (name, total, nb))
        e = TensorEntry()
        e.index, e.name, e.dtype, e.shape, e.nbytes = ti, name, dt, list(shape), nb
        e.st_offsets, e.sha256, e.chunks = list(so), sha, rows
        h.tensors.append(e)
        h.by_name[name] = e
    if at != file_size or meta.get("file_size") != file_size:
        bad("the payloads end at %d, the JSON says %r, the prefix says %d" % (at, meta.get("file_size"), file_size))
    # The tensor table must BE the embedded source header: the same names, dtypes, shapes and data offsets, in
    # its data order. check_source() ties that embedded header to the raw pack on disk, so a table that renamed
    # or swapped tensors -- even one written with a valid header crc -- is refused here, not decoded.
    src = meta.get("source")
    if not (isinstance(src, dict) and isinstance(src.get("st_header_b64"), str)
            and isinstance(src.get("basename"), str) and _is_int(src.get("size"))
            and isinstance(src.get("st_header_sha256"), str)):
        bad("no valid embedded source record")
    try:
        sb = base64.b64decode(src["st_header_b64"].encode("ascii"), validate=True)
        st = json.loads(sb.decode("utf-8"))
        want = sorted(((k, v) for k, v in st.items() if k != "__metadata__"),
                      key=lambda kv: kv[1]["data_offsets"][0])
        exp = [(k, v["dtype"], list(v["shape"]), list(v["data_offsets"])) for k, v in want]
    except Exception as e:                                                # noqa: BLE001
        bad("the embedded source header does not parse (%s)" % type(e).__name__)
    if hashlib.sha256(sb).hexdigest() != src["st_header_sha256"]:
        bad("the embedded source header does not match its recorded sha256")
    if src.get("data_start") != 8 + len(sb):
        bad("source data_start %r, its header says %d" % (src.get("data_start"), 8 + len(sb)))
    got = [(e.name, e.dtype, e.shape, e.st_offsets) for e in h.tensors]
    if got != exp:
        bad("the tensor table does not match the embedded source header (%d vs %d tensors; first difference at "
            "%s)" % (len(got), len(exp), next((i for i, (g, x) in enumerate(zip(got, exp)) if g != x),
                                             min(len(got), len(exp)))))
    return h


def read_header(path):
    """Parse and validate an nhz1 file's header. Raises NhzMissingError / NhzFormatError / NhzVersionError /
    NhzTruncatedError -- a truncated file is caught HERE, from its recorded size, before any data read."""
    try:
        fh = open(path, "rb")
    except FileNotFoundError as e:
        raise NhzMissingError(e.errno, "%s: no such nhz file" % path) from None
    with fh:
        st = os.fstat(fh.fileno())
        size, stamp = st.st_size, (st.st_size, st.st_mtime_ns)
        pre = fh.read(PREFIX_LEN)
        if pre[:8] != MAGIC:
            if len(pre) < 8 and MAGIC.startswith(pre):
                raise NhzTruncatedError("%s: %d bytes -- truncated inside the magic" % (_bn(path), len(pre)))
            raise NhzFormatError("%s: not an nhz1 file (magic %r)" % (_bn(path), bytes(pre[:8])))
        if len(pre) < 12:
            raise NhzTruncatedError("%s: truncated inside the fixed prefix (%d bytes)" % (_bn(path), len(pre)))
        version = struct.unpack("<I", pre[8:12])[0]
        if version != VERSION:
            raise NhzVersionError("%s: format version %d, this reader knows %d" % (_bn(path), version, VERSION))
        if len(pre) < PREFIX_LEN:
            raise NhzTruncatedError("%s: truncated inside the fixed prefix (%d bytes)" % (_bn(path), len(pre)))
        _m, _v, hlen, fsize, hcrc, _r = _PREFIX.unpack(pre)
        if not 0 < hlen <= HEADER_MAX:
            raise NhzFormatError("%s: header length %d out of range" % (_bn(path), hlen))
        if size < fsize:
            raise NhzTruncatedError("%s: file is %d bytes but its header says %d -- truncated"
                                    % (_bn(path), size, fsize))
        if size > fsize:
            raise NhzFormatError("%s: file is %d bytes but its header says %d -- %d trailing bytes"
                                 % (_bn(path), size, fsize, size - fsize))
        hb = fh.read(hlen)
        if len(hb) != hlen:
            raise NhzTruncatedError("%s: header truncated (%d of %d bytes)" % (_bn(path), len(hb), hlen))
    if zlib.crc32(hb) != hcrc:
        raise NhzFormatError("%s: header checksum mismatch (crc32 %08x, recorded %08x)"
                             % (_bn(path), zlib.crc32(hb), hcrc))
    try:
        meta = json.loads(hb.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise NhzFormatError("%s: header is not JSON (%s)" % (_bn(path), type(e).__name__)) from None
    return _validate(path, meta, hlen, fsize, stamp)


def check_name(hdr, basename):
    """NhzStaleError unless this nhz was converted from a source file called `basename`. Every GLM pack has the
    SAME size and byte-identical header bytes (measured: 46 packs, one size, no metadata), so the name is what
    binds a file to its LAYER -- a layer_2.nhz copied to layer_3.nhz must be refused, not served as layer 3."""
    if hdr.source.get("basename") != basename:
        raise NhzStaleError("%s was converted from %r, not %r -- a file of another layer (reconvert it)"
                            % (_bn(hdr.path), hdr.source.get("basename"), basename))


def spot_check_content(hdr, st_path, fh=None):
    """NhzStaleError unless the raw pack's bytes still match what this nhz was converted from: for every tensor
    the MIDDLE chunk's byte range is plain-read from the raw pack and its position-seeded crc32 compared with the
    chunk table. One chunk per tensor (2 x 2 MiB for a GLM slab) catches a re-generated pack of the same shape;
    `convert_packs.py --reverify` audits every byte. Returns the number of bytes read."""
    base = hdr.source.get("data_start")
    n = 0
    own = fh is None
    f = open(st_path, "rb") if own else fh
    try:
        for e in hdr.tensors:
            if not e.chunks:
                continue
            ci = len(e.chunks) // 2
            off, stored, out_len, crc, mode = e.chunks[ci]
            f.seek(base + e.st_offsets[0] + ci * hdr.chunk_bytes)
            b = f.read(out_len)
            n += len(b)
            if len(b) != out_len or zlib.crc32(b, chunk_seed(e.index, ci)) != crc:
                raise NhzStaleError("%s: the raw pack's tensor %r chunk %d is not the one %s was converted from "
                                    "(crc32 differs) -- the pack was re-generated; reconvert it"
                                    % (_bn(st_path), e.name, ci, _bn(hdr.path)))
    finally:
        if own:
            f.close()
    return n


# ---------------------------------------------------------------- the binding record (layer_N.nhz.bind.json)
# WHY. A spot-check SAMPLES the raw pack (one chunk per tensor = 4 MiB of a 1.125 GiB GLM slab), so a pack
# re-generated only partly (some experts) would pass it and the codec would keep serving the OLD bytes. The
# binding record closes that without re-reading 1.2 GB per layer per launch: the converter writes it next to the
# .nhz after a VERIFIED conversion (or after a --reverify audit that re-hashed the whole raw pack) and it records
# both files' (size, mtime_ns) and the source sha256. At build, a raw pack or .nhz whose stamp differs from its
# record -- touched, re-generated, replaced, copied -- is REFUSED until `convert_packs.py --reverify` re-hashes it
# and re-binds it (content unchanged) or reconverts it. The .nhz itself stays content-deterministic.
BIND_SUFFIX = ".bind.json"


def bind_path(nhz_path):
    return str(nhz_path) + BIND_SUFFIX


def write_binding(nhz_path, st_path, how, raw_stamp=None, nhz_stamp=None):
    """Record that nhz_path is the verified encoding of st_path. Atomic. Only the converter (after a verified
    conversion or a full audit) should call this: it is a statement that was checked.

    raw_stamp: the raw pack's (size, mtime_ns) taken BEFORE the bytes that were verified were read -- the caller
    re-stats afterwards and refuses when they differ, so a pack changed mid-conversion can never be bound to the
    old bytes. None means "stat it now" (only for callers with nothing in flight, e.g. tests).
    st_path=None writes a RAW-FREE binding (raw_stamp null) for a box that holds only the .nhz: the caller must
    have decoded the whole file and matched its rebuilt sha256 against an INDEPENDENT manifest first.
    nhz_stamp: the .nhz stamp of the header the caller VERIFIED; a file replaced since is refused, never bound."""
    h = read_header(nhz_path)
    if nhz_stamp is not None and tuple(h.stamp) != tuple(nhz_stamp):
        raise NhzStaleError("%s changed while it was being verified (stamp %s, verified %s) -- not bound"
                            % (_bn(nhz_path), list(h.stamp), list(nhz_stamp)))
    if st_path is None:
        raw, rs = h.source.get("basename"), None
    else:
        raw, rs = _bn(st_path), list(raw_stamp) if raw_stamp is not None else list(file_stamp(st_path))
    rec = {"format": "nhz1-bind", "version": 1, "nhz": _bn(nhz_path), "nhz_stamp": list(h.stamp), "raw": raw,
           "raw_stamp": rs, "file_sha256": h.source.get("file_sha256"), "how": str(how),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = bind_path(nhz_path)
    with open(p + ".tmp", "w", encoding="ascii") as f:
        json.dump(rec, f)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(20):                  # Windows: a reader holding the record open blocks the replace
        try:
            os.replace(p + ".tmp", p)
            return rec
        except PermissionError:
            if attempt == 19:
                try:
                    os.remove(p + ".tmp")
                except OSError:
                    pass
                raise
            time.sleep(0.05 * (attempt + 1))
    return rec


def rebuilt_sha256(nhz_path, threads=4):
    """sha256 of the SOURCE file this nhz rebuilds to (8-byte length + its safetensors header + every tensor's
    decoded bytes in data order), streamed, nothing written. Every chunk crc32 and tensor sha256 is checked on
    the way. This is what a raw-free box compares with an independent manifest."""
    hdr = read_header(nhz_path)
    src = hdr.source
    if not src.get("tiles_exactly") or not src.get("file_sha256"):
        raise NhzFormatError("%s: the source's tensors did not tile its data region" % _bn(nhz_path))
    hb = base64.b64decode(src["st_header_b64"])
    h = hashlib.sha256()
    h.update(struct.pack("<Q", len(hb)))
    h.update(hb)
    for e in sorted(hdr.tensors, key=lambda x: x.st_offsets[0]):
        buf = np.empty(e.nbytes, dtype=np.uint8)
        decode_into(nhz_path, [(e.name, buf)], threads=threads, verify="sha", hdr=hdr)
        h.update(buf)
        del buf
    return h.hexdigest()


def read_binding(hdr):
    """The binding record of hdr's file, checked against it: NhzStaleError when absent, unreadable, or when the
    .nhz changed since it was bound (its stamp) or records another source."""
    p = bind_path(hdr.path)
    try:
        with open(p, "rb") as f:
            raw = f.read(1 << 16)
        rec = json.loads(raw.decode("ascii"))
    except FileNotFoundError:
        raise NhzStaleError("%s has no binding record (%s): it was not written by a verified conversion or audit "
                            "-- run convert_packs.py --reverify" % (_bn(hdr.path), _bn(p))) from None
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as e:
        raise NhzStaleError("%s: its binding record is unreadable (%s)" % (_bn(hdr.path), type(e).__name__)) from None
    rs = rec.get("raw_stamp") if isinstance(rec, dict) else None
    if not (isinstance(rec, dict) and rec.get("format") == "nhz1-bind" and rec.get("version") == 1
            and (rs is None or (isinstance(rs, list) and len(rs) == 2 and all(_is_int(x) for x in rs)))):
        raise NhzStaleError("%s: its binding record is not a valid nhz1-bind record" % _bn(hdr.path))
    if rec.get("nhz") != _bn(hdr.path) or rec.get("nhz_stamp") != list(hdr.stamp):
        raise NhzStaleError("%s changed since it was bound (stamp %s, bound %s): replaced or copied -- run "
                            "convert_packs.py --reverify" % (_bn(hdr.path), list(hdr.stamp), rec.get("nhz_stamp")))
    if rec.get("raw") != hdr.source.get("basename") or rec.get("file_sha256") != hdr.source.get("file_sha256"):
        raise NhzStaleError("%s: its binding record describes another source" % _bn(hdr.path))
    return rec


def check_unbound(hdr, basename):
    """With NO raw pack to compare against: the right layer's file, bound by a valid record, unchanged since.
    Returns the record (its file_sha256 is what an operator can compare with an independent manifest)."""
    check_name(hdr, basename)
    return read_binding(hdr)


def check_source(hdr, st_path, content=True, binding=True):
    """NhzStaleError unless the raw pack at st_path is the file this nhz was converted from: its NAME, its size
    and header bytes, (binding=True) both files unchanged since the verified conversion/audit recorded them, and
    (content=True) a crc spot-check of one chunk per tensor against the raw bytes. An unreadable raw pack is an
    NhzStaleError too. Returns the raw bytes the spot-check read."""
    check_name(hdr, _bn(st_path))
    try:
        size, hsha = source_identity(st_path)
    except (OSError, EOFError, ValueError, UnicodeDecodeError, RecursionError) as e:
        raise NhzStaleError("%s: the raw pack is unreadable (%s: %s)" % (st_path, type(e).__name__,
                                                                         str(e)[:120])) from None
    src = hdr.source
    if size != src.get("size") or hsha != src.get("st_header_sha256"):
        raise NhzStaleError("%s does not describe %s: source size %s / header sha256 %s..., the raw pack is "
                            "%d bytes / %s... -- reconvert it" % (_bn(hdr.path), st_path, src.get("size"),
                                                                  str(src.get("st_header_sha256"))[:16],
                                                                  size, hsha[:16]))
    if binding:
        rec = read_binding(hdr)
        now = list(file_stamp(st_path))
        if rec["raw_stamp"] is None:
            raise NhzStaleError("%s was bound WITHOUT a raw pack (a raw-free rebind), but %s is present now -- run "
                                "convert_packs.py --reverify to bind it to that pack" % (_bn(hdr.path), st_path))
        if now != rec["raw_stamp"]:
            raise NhzStaleError("%s changed since %s was bound to it (stamp %s, bound %s): touched, re-generated or "
                                "copied -- run convert_packs.py --reverify: it re-hashes the pack and re-binds it "
                                "when the content is unchanged, else reconverts" % (st_path, _bn(hdr.path), now,
                                                                                    rec["raw_stamp"]))
    if not content:
        return 0
    try:
        return spot_check_content(hdr, st_path)
    except OSError as e:
        raise NhzStaleError("%s: the raw pack is unreadable (%s)" % (st_path, type(e).__name__)) from None


# ============================================================================ decode
_SHA_DONE = set()                          # (normalised path, stamp, tensor) verified in THIS process
_SHA_LOCK = threading.Lock()


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def sha_verified(path, stamp, name):
    with _SHA_LOCK:
        return (_norm(path), stamp, name) in _SHA_DONE


def _readinto_exact(fh, mv, path, what):
    got, want = 0, len(mv)
    while got < want:
        k = fh.readinto(mv[got:])
        if not k:
            raise NhzTruncatedError("%s: short read for %s -- %d of %d bytes" % (_bn(path), what, got, want))
        got += k


def _decode_one(path, fh, smv, task):
    ti, ci, name, off, stored, out_len, crc, mode, dst = task
    what = "tensor %r chunk %d" % (name, ci)
    fh.seek(off)
    if mode == MODE_RAW:
        _readinto_exact(fh, dst, path, what)
    else:
        buf = smv[:stored]
        _readinto_exact(fh, buf, path, what)
        nw = out_len // 2
        hl = stored - nw
        d = zlib.decompressobj(ZLIB_WBITS)
        try:
            hi = d.decompress(buf[:hl], nw + 1)
        except zlib.error as e:
            raise NhzChunkError("%s: %s: the high-byte stream does not inflate (%s)" % (_bn(path), what, e)) from None
        if len(hi) != nw or not d.eof or d.unused_data or d.unconsumed_tail:
            raise NhzChunkError("%s: %s: the high-byte stream inflates to %d of %d bytes (end of stream: %s, "
                                "%d bytes left over)" % (_bn(path), what, len(hi), nw, d.eof,
                                                         len(d.unused_data) + len(d.unconsumed_tail)))
        out = np.frombuffer(dst, dtype=np.uint8).reshape(nw, 2)
        out[:, 1] = np.frombuffer(hi, dtype=np.uint8)
        out[:, 0] = np.frombuffer(buf, dtype=np.uint8, count=nw, offset=hl)
        del out
    got = zlib.crc32(dst, chunk_seed(ti, ci))
    if got != crc:
        raise NhzChunkError("%s: %s: crc32 of the decoded bytes %08x != recorded %08x -- corrupt chunk"
                            % (_bn(path), what, got, crc))


def _run_decode(path, tasks, threads, max_stored):
    """Each worker has its OWN handle and staging buffer and pulls chunks in file order; the first error
    stops every worker and is re-raised here, after all of them have joined."""
    errs, lock, nxt, stop = [], threading.Lock(), [0], [False]

    def worker():
        try:
            try:
                fh = open(path, "rb")
            except FileNotFoundError as e:           # removed after its header was read: still the NAMED error
                raise NhzMissingError(e.errno, "%s: no such nhz file (it disappeared mid-read)" % path) from None
            with fh:
                stage = bytearray(max(1, max_stored))
                smv = memoryview(stage)
                try:
                    while True:
                        with lock:
                            if stop[0] or nxt[0] >= len(tasks):
                                return
                            i = nxt[0]
                            nxt[0] += 1
                        _decode_one(path, fh, smv, tasks[i])
                finally:
                    smv.release()
        except BaseException as e:                                        # noqa: BLE001 -- re-raised below
            with lock:
                errs.append(e)
                stop[0] = True

    n = max(1, min(int(threads), len(tasks), THREADS_MAX))
    if n == 1:
        worker()
    else:
        started = []
        try:
            for i in range(n):
                t = threading.Thread(target=worker, name="st-nhz-%d" % i, daemon=True)
                t.start()
                started.append(t)
        except BaseException:                        # e.g. no thread could be created: stop and join the rest
            with lock:
                stop[0] = True
            for t in started:
                t.join()
            raise
        for t in started:
            t.join()
    if errs:
        raise errs[0]


def _sha_check(path, pairs, threads):
    """pairs = [(TensorEntry, memoryview)]: one hashing thread per tensor (hashlib releases the GIL)."""
    errs, lock = [], threading.Lock()

    def one(e, mv):
        try:
            got = hashlib.sha256(mv).hexdigest()
            if got != e.sha256:
                raise NhzShaMismatch("%s: tensor %r sha256 %s... != recorded %s... -- the decoded bytes are NOT "
                                     "the source's" % (_bn(path), e.name, got[:16], e.sha256[:16]))
        except BaseException as x:                                        # noqa: BLE001
            with lock:
                errs.append(x)
    if len(pairs) <= 1 or threads <= 1:
        for e, mv in pairs:
            one(e, mv)
    else:
        ts = [threading.Thread(target=one, args=p, name="st-nhz-sha-%d" % i, daemon=True)
              for i, p in enumerate(pairs)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    if errs:
        raise errs[0]


def decode_into(path, targets, threads=4, verify="crc", hdr=None):
    """Fill each (name, writable uint8 buffer of exactly nbytes) in `targets` with that tensor's ORIGINAL
    bytes. Every chunk's crc32 is always checked; verify="sha" also checks every tensor's sha256, and
    verify="first" does so the first time this process decodes that (file, stamp, tensor).
    Returns stats {"bytes_in", "bytes_out", "chunks", "sha_checked", "seconds"}."""
    if verify not in VERIFY_MODES:
        raise ValueError("verify=%r, want one of %s" % (verify, VERIFY_MODES))
    t0 = time.perf_counter()
    if hdr is None:
        hdr = read_header(path)
    tasks, pairs, views = [], [], []
    max_stored = bytes_in = bytes_out = 0
    todo = []
    try:
        for name, buf in targets:
            e = hdr.entry(name)
            mv = memoryview(buf)                # OUR view: a caller's memoryview is never released here
            views.append(mv)
            if mv.ndim != 1 or mv.format != "B":
                mv = mv.cast("B")               # a contiguous buffer of another format; TypeError otherwise
                views.append(mv)
            if mv.readonly or mv.nbytes != e.nbytes or mv.format != "B" or mv.ndim != 1:
                raise ValueError("%s: target for %r must be a writable 1-D byte buffer of %d bytes (got %d, "
                                 "readonly=%s, format %r)" % (_bn(path), name, e.nbytes, mv.nbytes, mv.readonly,
                                                              mv.format))
            at = 0
            for ci, (off, stored, out_len, crc, mode) in enumerate(e.chunks):
                sub = mv[at:at + out_len]
                views.append(sub)
                tasks.append((e.index, ci, name, off, stored, out_len, crc, mode, sub))
                at += out_len
                if mode == MODE_HILO:
                    max_stored = max(max_stored, stored)
                bytes_in += stored
            bytes_out += e.nbytes
            pairs.append((e, mv))
        tasks.sort(key=lambda t: t[3])          # file order: the disk sees one forward sweep per worker set
        _run_decode(path, tasks, threads, max_stored)
        todo = [(e, mv) for e, mv in pairs
                if verify == "sha" or (verify == "first" and not sha_verified(path, hdr.stamp, e.name))]
        if todo:
            _sha_check(path, todo, threads)
            with _SHA_LOCK:
                for e, _mv in todo:
                    _SHA_DONE.add((_norm(path), hdr.stamp, e.name))
    finally:
        for v in reversed(views):
            try:
                v.release()
            except BufferError:                 # something still exports it (a traceback frame): GC frees it
                pass
    return {"bytes_in": bytes_in, "bytes_out": bytes_out, "chunks": len(tasks), "sha_checked": len(todo),
            "seconds": time.perf_counter() - t0}


def decode_tensors(path, names=None, threads=4, verify="sha"):
    """{name: 1-D np.uint8 array of the original bytes} -- for tests, the converter and rebuilds."""
    hdr = read_header(path)
    names = [e.name for e in hdr.tensors] if names is None else list(names)
    out = {n: np.empty(hdr.entry(n).nbytes, dtype=np.uint8) for n in names}
    decode_into(path, [(n, out[n]) for n in names], threads=threads, verify=verify, hdr=hdr)
    return out


def rebuild_safetensors(nhz_path, out_path, threads=4):
    """Write the SOURCE safetensors file back, byte for byte, and check it against the recorded whole-file
    sha256. Returns that sha256. Refuses a file whose source tensors did not tile the data region."""
    hdr = read_header(nhz_path)
    src = hdr.source
    if not src.get("tiles_exactly") or not src.get("file_sha256"):
        raise NhzFormatError("%s: the source's tensors did not tile its data region; no byte-exact rebuild"
                             % _bn(nhz_path))
    hb = base64.b64decode(src["st_header_b64"])
    if hashlib.sha256(hb).hexdigest() != src["st_header_sha256"]:
        raise NhzShaMismatch("%s: recorded safetensors header does not match its sha256" % _bn(nhz_path))
    data = decode_tensors(nhz_path, threads=threads, verify="sha")
    h = hashlib.sha256()
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        for piece in (struct.pack("<Q", len(hb)), hb):
            f.write(piece)
            h.update(piece)
        for e in sorted(hdr.tensors, key=lambda x: x.st_offsets[0]):
            f.write(data[e.name])
            h.update(data[e.name])
    if h.hexdigest() != src["file_sha256"]:
        os.remove(tmp)
        raise NhzShaMismatch("%s: rebuilt file sha256 %s... != recorded %s..."
                             % (_bn(nhz_path), h.hexdigest()[:16], src["file_sha256"][:16]))
    os.replace(tmp, out_path)
    return h.hexdigest()


# ============================================================================ encode
def _parallel_map(fn, items, threads, name="nhz-enc"):
    """[fn(x) for x in items] on `threads` threads, in order; the first error is re-raised after all join."""
    out = [None] * len(items)
    errs, lock, nxt = [], threading.Lock(), [0]

    def worker():
        while True:
            with lock:
                if errs or nxt[0] >= len(items):
                    return
                i = nxt[0]
                nxt[0] += 1
            try:
                out[i] = fn(items[i])
            except BaseException as e:                                    # noqa: BLE001
                with lock:
                    errs.append(e)
                return
    n = max(1, min(int(threads), len(items), THREADS_MAX))
    if n == 1:
        worker()
    else:
        ts = [threading.Thread(target=worker, name="%s-%d" % (name, i), daemon=True) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    if errs:
        raise errs[0]
    return out


def _encode_chunk(args):
    """(mode, [payload parts], crc) of one chunk; the crc is over the ORIGINAL bytes, position-seeded."""
    ti, ci, raw, hilo, level, strategy = args
    crc = zlib.crc32(raw, chunk_seed(ti, ci))
    if hilo:
        pair = raw.reshape(-1, 2)
        hi = np.ascontiguousarray(pair[:, 1])
        co = zlib.compressobj(level, zlib.DEFLATED, ZLIB_WBITS, 9, strategy)
        hz = co.compress(hi) + co.flush()
        if len(hz) < len(hi):
            return MODE_HILO, [hz, np.ascontiguousarray(pair[:, 0]).tobytes()], crc
    return MODE_RAW, [raw.tobytes()], crc


def _header_json(meta):
    return json.dumps(meta, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def encode_file(src, dst, chunk_words=CHUNK_WORDS, threads=4, level=ZLIB_LEVEL, strategy=ZLIB_STRATEGY,
                block_chunks=32, log=None):
    """Encode the safetensors file `src` into the nhz1 file `dst` (written in place and fsync'd; the caller
    owns the .part + rename). Every crc and sha256 is taken from the SOURCE bytes as they were read.
    Returns a summary dict. Deterministic: the same source and parameters give the same bytes."""
    t0 = time.perf_counter()
    if _norm(src) == _norm(dst):
        raise ValueError("refusing to encode %s onto itself" % src)
    st_hdr, hb, base = read_st_header(src)
    src_size = os.path.getsize(src)
    names = sorted((k for k in st_hdr if k != "__metadata__"), key=lambda k: st_hdr[k]["data_offsets"][0])
    chunk_bytes = 2 * int(chunk_words)
    plan, at, tiles = [], 0, True
    for ti, k in enumerate(names):
        m = st_hdr[k]
        dt = m["dtype"]
        if not isinstance(dt, str) or dt not in ITEMSIZE:
            raise NhzFormatError("%s: tensor %r has dtype %r, which nhz1 does not know" % (_bn(src), k, dt))
        a, b = m["data_offsets"]
        want = ITEMSIZE[dt]
        for s in m["shape"]:
            want *= int(s)
        if b - a != want:
            raise NhzFormatError("%s: tensor %r spans %d bytes, its shape says %d" % (_bn(src), k, b - a, want))
        if base + b > src_size:
            raise EOFError("%s: tensor %r runs past the end of the file" % (_bn(src), k))
        if a != at:
            tiles = False
        at = b
        plan.append((ti, k, dt, list(m["shape"]), a, b))
    if base + at != src_size:
        tiles = False
    # ---- reserve room for the header: the same JSON with every number at its widest possible value
    wide = (1 << 53) - 1
    src_meta = {"basename": _bn(src), "size": src_size, "data_start": base, "tiles_exactly": tiles,
                "st_header_b64": base64.b64encode(hb).decode("ascii"),
                "st_header_sha256": hashlib.sha256(hb).hexdigest(), "file_sha256": "0" * 64}
    codec_meta = {"name": "bf16-hilo-deflate", "hilo_dtypes": list(HILO_DTYPES), "zlib_level": int(level),
                  "zlib_strategy": int(strategy), "zlib_wbits": ZLIB_WBITS,
                  "hilo_payload": "raw deflate of byte 1 (sign + 7 exponent bits) of every word, then byte 0 raw",
                  "crc": "crc32 of the chunk's decoded bytes, start value crc32(b'nhz1|<tensor>|<chunk>')"}

    def meta_for(tensors, file_size, data_start):
        return {"format": FORMAT, "version": VERSION, "chunk_bytes": chunk_bytes, "data_start": data_start,
                "file_size": file_size, "codec": codec_meta, "source": src_meta, "tensors": tensors}
    wide_t = []
    for ti, k, dt, shape, a, b in plan:
        nch = -(-(b - a) // chunk_bytes)
        wide_t.append({"name": k, "dtype": dt, "shape": shape, "nbytes": b - a, "st_data_offsets": [a, b],
                       "sha256": "0" * 64, "chunks": [[wide, wide, wide, 0xFFFFFFFF, 1]] * nch})
    reserve = -(-(PREFIX_LEN + len(_header_json(meta_for(wide_t, wide, wide))) + 64) // DATA_ALIGN) * DATA_ALIGN
    # ---- stream the source: read a block of chunks, compress it on `threads` threads, hash it meanwhile
    fsha = hashlib.sha256()
    fsha.update(struct.pack("<Q", len(hb)))
    fsha.update(hb)
    tensors, off = [], reserve
    per = {}
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(b"\0" * reserve)
        for ti, k, dt, shape, a, b in plan:
            nb = b - a
            tsha = hashlib.sha256()
            rows = []
            hilo = dt in HILO_DTYPES
            tin = tout = 0
            nch = -(-nb // chunk_bytes)
            fin.seek(base + a)
            for c0 in range(0, nch, block_chunks):
                c1 = min(nch, c0 + block_chunks)
                lo_b, hi_b = c0 * chunk_bytes, min(nb, c1 * chunk_bytes)
                blk = np.empty(hi_b - lo_b, dtype=np.uint8)
                _readinto_exact(fin, memoryview(blk), src, "tensor %r bytes %d..%d" % (k, lo_b, hi_b))
                jobs = [(ti, ci, blk[(ci - c0) * chunk_bytes:min(hi_b - lo_b, (ci - c0 + 1) * chunk_bytes)],
                         hilo, level, strategy) for ci in range(c0, c1)]

                herr = []

                def _hash(blk=blk, tsha=tsha, herr=herr):   # both digests IN ORDER, alongside the compression
                    try:
                        tsha.update(blk)
                        if tiles:
                            fsha.update(blk)
                    except BaseException as e:              # noqa: BLE001 -- re-raised below
                        herr.append(e)
                hasher = threading.Thread(target=_hash, name="nhz-enc-sha", daemon=True)
                hasher.start()
                try:
                    res = _parallel_map(_encode_chunk, jobs, threads)
                finally:
                    hasher.join()
                if herr:
                    raise herr[0]
                for (_ti, _ci, raw, _h, _l, _s), (mode, parts, crc) in zip(jobs, res):
                    stored = sum(len(p) for p in parts)
                    for p in parts:
                        fout.write(p)
                    rows.append([off, stored, len(raw), crc, mode])
                    off += stored
                    tin += len(raw)
                    tout += stored
                del blk, jobs, res
            tensors.append({"name": k, "dtype": dt, "shape": shape, "nbytes": nb, "st_data_offsets": [a, b],
                            "sha256": tsha.hexdigest(), "chunks": rows})
            per[k] = {"dtype": dt, "shape": shape, "bytes": tin, "stored": tout,
                      "ratio": round(tin / tout, 4) if tout else None, "sha256": tsha.hexdigest(),
                      "hilo_chunks": sum(1 for r in rows if r[4] == MODE_HILO), "chunks": len(rows)}
            if log is not None:
                log("[nhz] %s %-8s %s %s: %d -> %d bytes (%.4fx)" % (_bn(src), k, dt, shape, tin, tout,
                                                                   tin / tout if tout else 0.0))
        src_meta["file_sha256"] = fsha.hexdigest() if tiles else None
        meta = meta_for(tensors, off, reserve)
        js = _header_json(meta)
        if PREFIX_LEN + len(js) > reserve:
            raise RuntimeError("header of %d bytes does not fit the %d reserved -- a bug" % (len(js), reserve))
        fout.seek(0)
        fout.write(_PREFIX.pack(MAGIC, VERSION, len(js), off, zlib.crc32(js), 0))
        fout.write(js)
        fout.flush()
        os.fsync(fout.fileno())
    return {"src": src, "dst": dst, "source_bytes": src_size, "nhz_bytes": off,
            "ratio_file": round(src_size / off, 4), "data_bytes": sum(p["bytes"] for p in per.values()),
            "stored_bytes": sum(p["stored"] for p in per.values()), "header_reserved": reserve,
            "header_json_bytes": len(js), "file_sha256": src_meta["file_sha256"], "tensors": per,
            "chunk_words": int(chunk_words), "threads": int(threads), "seconds": round(time.perf_counter() - t0, 3)}


# ============================================================================ CLI
def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="nhz1 pack codec: info / verify / encode / rebuild")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info")
    p.add_argument("path")
    p = sub.add_parser("verify")
    p.add_argument("path")
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("encode")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("rebuild")
    p.add_argument("path")
    p.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "info":
        h = read_header(a.path)
        print(json.dumps({"file_size": h.file_size, "data_start": h.data_start, "chunk_bytes": h.chunk_bytes,
                          "source": {k: v for k, v in h.source.items() if k != "st_header_b64"},
                          "tensors": [{"name": e.name, "dtype": e.dtype, "shape": e.shape, "nbytes": e.nbytes,
                                       "stored": sum(c[1] for c in e.chunks), "chunks": len(e.chunks),
                                       "sha256": e.sha256} for e in h.tensors]}, indent=1))
    elif a.cmd == "verify":
        h = read_header(a.path)
        s = decode_into(a.path, [(e.name, np.empty(e.nbytes, dtype=np.uint8)) for e in h.tensors],
                        threads=a.threads, verify="sha", hdr=h)
        print("OK %s: %d tensors sha256-verified, %d chunks crc-verified, %.2fs" % (a.path, s["sha_checked"],
                                                                                    s["chunks"], s["seconds"]))
    elif a.cmd == "encode":
        print(json.dumps(encode_file(a.src, a.dst, threads=a.threads), indent=1))
    elif a.cmd == "rebuild":
        print("rebuilt %s sha256 %s" % (a.out, rebuild_safetensors(a.path, a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
