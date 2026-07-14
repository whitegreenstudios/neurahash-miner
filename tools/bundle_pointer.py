#!/usr/bin/env python
"""bundle_pointer.py -- content-addressed, hash-verified + signature-pinned distribution of the miner
training bundle (client-side reference implementation).

A consumer resolves ONE canonical pointer record carrying the bundle's sha256 (+ optional IPFS cid) and
an ordered seed list [ipfs, VPS content-store, HuggingFace], tries the seeds in order, VERIFIES the hash,
and rejects any mismatch. Trust lives in the HASH at the consumer -- never in the source -- so the seeds
are interchangeable and droppable: removing one only reduces AVAILABILITY, never correctness, and a
substituted body from ANY seed is rejected. That delivers "everyone can read it, nobody can edit it".

Two independent checks, in two functions:
  * resolve_bundle()          -- pins WHAT the bytes are (sha256 / CID hash-verify at the consumer).
  * verified_bundle_record()  -- pins WHO wrote the pointer (the coordinator's secp256k1 signature over
                                 the governance log, against an OUT-OF-BAND coordinator address). This
                                 rejects a FORGED pointer whose sha256 would otherwise self-verify against
                                 the attacker's own bytes.

Deps: Python stdlib + neura_l1.signing (public secp256k1 recover) for the signature check, and a LAZY
import of tools.ipfs_checkpoint only when an ipfs: seed with a cid is actually tried."""
import hashlib
import json
import os
import tempfile
import time
import urllib.request

BUNDLE_KIND = "bundle_canonical"
DEFAULT_VPS = "http://47.84.93.96:8710"          # VPS content-store; served BY sha256 at /o/<sha>
_GENESIS = "0" * 64                              # signed-log genesis head (matches the coordinator ledger)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def default_seeds(*, vps=DEFAULT_VPS, hf_repo=None, ipfs=True):
    """Ordered seed templates; `{sha}` is substituted at resolve time. IPFS first (the decentralized
    backbone that grows with the fleet via self-pinning), then the fast centralized convenience seeds
    (droppable -- integrity is the hash check, not the source)."""
    seeds = []
    if ipfs:
        seeds.append("ipfs:")
    if vps:
        seeds.append(vps.rstrip("/") + "/o/{sha}")
    if hf_repo:
        seeds.append("https://huggingface.co/datasets/%s/resolve/main/bundle_{sha}.zip" % hf_repo)
    return seeds


def bundle_record(zip_path, *, cid=None, seeds=None, vps=DEFAULT_VPS, hf_repo=None, ts=None):
    """Build a canonical `bundle_canonical` pointer record over `zip_path`. `cid` (an IPFS CIDv1, a
    DIFFERENT digest from the sha256) is optional and additive: when absent the ipfs: seed is omitted."""
    sha = sha256_file(zip_path)
    if seeds is None:
        seeds = default_seeds(vps=vps, hf_repo=hf_repo, ipfs=cid is not None)
    return {"kind": BUNDLE_KIND, "sha256": sha, "cid": cid,
            "size": int(os.path.getsize(zip_path)), "seeds": list(seeds),
            "ts": int(ts if ts is not None else time.time())}


def latest_bundle_record(governance_records):
    """Return the NEWEST bundle_canonical record (records are oldest-first), or None. Also accepts WRAPPED
    log entries ({"type":"governance","record":{...}}) so the same resolver works against the wrapped,
    signed governance log. Non-bundle governance records are ignored."""
    latest = None
    for e in governance_records or []:
        if not isinstance(e, dict):
            continue
        rec = e
        if e.get("type") == "governance" and isinstance(e.get("record"), dict):
            rec = e["record"]
        if rec.get("kind") == BUNDLE_KIND:
            latest = rec
    return latest


def _canon(obj):
    """Deterministic bytes for a signed-log entry body (sorted keys, no sig/hash) -- must match the
    coordinator's canonicalization exactly."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def verify_signed_log(log, expected_address, *, genesis=_GENESIS):
    """Verify a signed, hash-chained coordinator governance log against a PINNED signer address, with NO
    private key -- the read-only verifier a public miner runs to reject a FORGED bundle pointer. Every
    entry's prev-link, signature and hash must check out AND every recovered signer must equal
    `expected_address` (pinned OUT-OF-BAND; an address travelling inside the log is never trusted, since an
    attacker who controls the log controls that field too). A log signed by a different key than the pin
    is rejected even when internally self-consistent. Returns (ok, reason); malformed input is rejected
    (returns False), never raised, because this is fed UNTRUSTED bytes off the wire.

    Known limitation, BY DESIGN: signature + link replay cannot detect WITHHOLDING -- a genuine strict
    PREFIX of the log verifies clean and yields an OLDER genuine pointer. This guarantees AUTHENTICITY,
    not FRESHNESS; a consumer needing anti-rollback must ratchet on a monotonic ts/height it tracks."""
    from neura_l1.signing import recover_bytes_scheme         # public secp256k1 recover (crypto-agility)
    if not expected_address:
        return False, "no pinned address supplied"
    head = genesis
    for i, e in enumerate(log or []):
        if not isinstance(e, dict):
            return False, "entry %d: not a dict" % i
        if e.get("prev") != head:
            return False, "entry %d: broken chain (prev != head)" % i
        try:
            body = {k: e[k] for k in e if k not in ("sig", "hash")}
            bts = _canon(body)
            signer = recover_bytes_scheme(bts, e["sig"], e.get("scheme"))
        except Exception as ex:                               # noqa: BLE001 (malformed -> reject)
            return False, "entry %d: bad entry/signature (%s)" % (i, ex)
        if signer.lower() != str(expected_address).lower():
            return False, "entry %d: signed by %s, not the pinned address %s" % (i, signer, expected_address)
        h = hashlib.sha256(head.encode() + bts + e["sig"].encode()).hexdigest()
        if h != e.get("hash"):
            return False, "entry %d: hash mismatch (tampered)" % i
        head = h
    return True, "ok"


def verified_bundle_record(signed_log, expected_coord_address):
    """Return the NEWEST bundle_canonical pointer from a SIGNED governance `signed_log`, but ONLY after the
    log's signature chain verifies against the PINNED `expected_coord_address`.

    Why the signature matters even though resolve_bundle already hash-verifies the BYTES: the sha256 inside
    a pointer only proves "the bytes match THIS pointer" -- it says nothing about who wrote the pointer. An
    attacker who can write the registry slot can publish {sha256: <hash of their OWN malicious bundle>,
    seeds: [their seed]}; resolve_bundle would fetch it, hash it, match the attacker's sha, and ACCEPT. The
    pin closes that gap: trust the pointer only if the REAL coordinator signed it. `expected_coord_address`
    is REQUIRED and supplied OUT-OF-BAND. Raises on a forged / tampered / empty log. Pair with
    resolve_bundle: signature pins WHO wrote the pointer, hash pins WHAT the bytes are."""
    if not expected_coord_address:
        raise ValueError("verified_bundle_record: a pinned coordinator address is REQUIRED -- an "
                         "unpinned bundle pointer is worthless (the signature's only job is the pin; the "
                         "bytes are already self-verified by resolve_bundle's hash check)")
    ok, reason = verify_signed_log(signed_log, expected_coord_address)
    if not ok:
        raise RuntimeError("verified_bundle_record: signed log failed pinned verification "
                           "(forged or tampered pointer): %s" % reason)
    rec = latest_bundle_record(signed_log)
    if rec is None:
        raise RuntimeError("verified_bundle_record: no %s record in the verified log" % BUNDLE_KIND)
    return rec


def _http_get(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "neurahash-bundle/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _atomic_write(dest, data):
    d = os.path.dirname(os.path.abspath(dest)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".bundle_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def resolve_bundle(record, dest, *, ipfs_bin=None, timeout=300):
    """Resolve a `bundle_canonical` record to `dest`, trying its seeds in order and VERIFYING the hash; a
    seed serving wrong bytes is rejected and the next seed is tried. Returns the seed string that served
    it. Raises RuntimeError only when EVERY seed is exhausted (dest is NOT written on failure)."""
    sha = record["sha256"]
    cid = record.get("cid")
    errors = []
    for seed in record.get("seeds", []):
        try:
            if seed.startswith("ipfs:"):
                if not cid:
                    continue                                   # no cid -> IPFS seed unusable, skip it
                try:
                    from tools import ipfs_checkpoint          # lazy: only when an ipfs seed is tried
                except ImportError:
                    import ipfs_checkpoint                     # when tools/ is on sys.path directly
                ipfs_checkpoint.fetch(cid, dest, verify_cid=True, ipfs_bin=ipfs_bin, timeout=timeout)
                return seed
            url = seed.replace("{sha}", sha)
            data = _http_get(url, timeout)
            got = sha256_bytes(data)
            if got != sha:
                errors.append("%s: sha mismatch (%s..)" % (url, got[:12]))
                continue                                        # tampered/wrong bytes -> reject, next seed
            if record.get("size") is not None and len(data) != record["size"]:
                errors.append("%s: size mismatch (%d != %d)" % (url, len(data), record["size"]))
                continue
            _atomic_write(dest, data)
            return url
        except Exception as ex:                                 # noqa: BLE001 (any seed failure -> next seed)
            errors.append("%s: %s" % (seed, str(ex)[:120]))
            continue
    raise RuntimeError("resolve_bundle: all %d seed(s) failed for sha %s..: %s"
                       % (len(record.get("seeds", [])), sha[:12], " | ".join(errors)))
