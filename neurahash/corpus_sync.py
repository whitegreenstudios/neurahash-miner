"""
corpus_sync.py — fetch the pool's corpus BY CONTENT HASH from the anchor content store, so a joiner
trains on EXACTLY the coordinator's bytes instead of whatever its git checkout happens to hold.

THE BUG THIS KILLS (issue #86)
------------------------------
The pool admits a miner only if its local ``corpus_torch.corpus_sha()`` matches the coordinator's
(the hello handshake, ``sharded_pool_node`` ~line 556). That hash is over the raw bytes of
``corpus_data/*.txt``. Git's autocrlf can rewrite those bytes on checkout (observed live 2026-07-02:
``english.txt`` was 762,087 bytes on one Windows box and 758,699 on another — same commit) → the
handshake correctly REFUSED the join, and the only fix was to hand-copy the coordinator's file.

Fetching each corpus file by its sha256 from the content store makes that impossible: you ask for a
hash, you get exactly those bytes (verified locally before they land) or the sync fails loudly. The
store's canonical bytes are the coordinator's bytes, so after a sync your ``corpus_sha()`` matches by
construction.

DESIGN
------
Pure stdlib (``urllib``) — a miner mid-bootstrap may not have anything else yet. The flow:

  1. GET ``<store>/manifest`` -> ``{name -> {sha256, size}}``.
  2. Keep the entries whose name is ``corpus/<file>`` and map them to ``<target_dir>/<file>``.
  3. For each: if the local file already hashes to the manifest sha256, SKIP it (idempotent, no
     re-download). Otherwise GET ``<store>/o/<sha256>``, verify ``sha256(body) == manifest sha256``
     (a malicious/buggy store that returns the wrong bytes is REFUSED, never written), and write the
     file ATOMICALLY (tmp in the same dir + ``os.replace``) so a crash mid-write can't leave a
     half-written corpus that would poison the next ``corpus_sha()``.

FAILURE MODES (all typed, all leave the checkout's bytes untouched on failure)
  * store unreachable / bad HTTP           -> ``StoreUnreachable``
  * manifest malformed                     -> ``StoreUnreachable`` (can't trust the index)
  * object missing / 404 for a listed hash -> ``CorpusSyncError``
  * body hash != manifest hash (tamper)    -> ``HashMismatch``   (the file is NOT written)

The caller (``run_miner_client``) treats any ``CorpusSyncError`` as "warn loudly and fall back to the
checkout's bytes" — a store outage must never block joining. The sync is OPT-IN: it runs only when
``NEURAHASH_CONTENT_STORE`` is set (default unset == today's behavior, byte-for-byte).
"""

import hashlib
import json
import os
import urllib.error
import urllib.request

__all__ = [
    "CorpusSyncError", "StoreUnreachable", "HashMismatch",
    "store_url_from_env", "fetch_manifest", "sync_corpus", "SyncResult",
]

# Manifest entries the pool serves for the corpus are named "corpus/<basename>" (see the live anchor
# manifest). Only those are synced into the corpus dir; other named objects (checkpoints, task_data)
# are out of scope for corpus sync.
CORPUS_PREFIX = "corpus/"
_ENV_STORE = "NEURAHASH_CONTENT_STORE"
_DEFAULT_TIMEOUT = 30.0


class CorpusSyncError(Exception):
    """Base class for every corpus-sync failure. Catching this catches all of them."""


class StoreUnreachable(CorpusSyncError):
    """The content store could not be reached, or its manifest was unreadable/malformed. The caller
    should warn and fall back to the git checkout's bytes — an outage must not block joining."""


class HashMismatch(CorpusSyncError):
    """A downloaded object's sha256 did not match the hash the manifest promised (a corrupt or
    MALICIOUS store). The file is refused and NOT written — surfacing this is a security event, not a
    reason to silently trust the bytes."""


class SyncResult:
    """Outcome of a :func:`sync_corpus` run, for logging/inspection.

    Attributes:
      updated:  list of basenames whose bytes were downloaded and (re)written.
      skipped:  list of basenames already matching the manifest hash (no download).
      target:   the directory corpus files were synced into.
    """

    __slots__ = ("updated", "skipped", "target")

    def __init__(self, updated, skipped, target):
        self.updated = list(updated)
        self.skipped = list(skipped)
        self.target = target

    @property
    def changed(self):
        """True iff at least one file's bytes were rewritten (so the caller knows corpus_sha may have
        moved and is worth re-logging)."""
        return bool(self.updated)

    def __repr__(self):
        return (f"SyncResult(updated={self.updated!r}, skipped={self.skipped!r}, "
                f"target={self.target!r})")


def store_url_from_env():
    """Return the content-store base URL from ``NEURAHASH_CONTENT_STORE`` (trailing slash stripped),
    or ``None`` when it is unset/empty. ``None`` means "sync disabled — keep today's behavior"."""
    url = (os.environ.get(_ENV_STORE, "") or "").strip()
    return url.rstrip("/") or None


def _http_get(url, timeout):
    """GET ``url`` and return the raw body bytes. Any transport/HTTP error is normalized:
      * a 404 -> ``CorpusSyncError`` (a listed object is missing — a store bug, distinct from an
        outage so the caller can tell them apart if it wants),
      * anything else (connection refused, DNS, timeout, 5xx) -> ``StoreUnreachable``.
    """
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:                       # a response arrived, but non-2xx
        if e.code == 404:
            raise CorpusSyncError(f"{url}: 404 (object listed in manifest but not served)") from e
        raise StoreUnreachable(f"{url}: HTTP {e.code}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:  # no usable response
        raise StoreUnreachable(f"{url}: unreachable ({e})") from e


def fetch_manifest(store_url, timeout=_DEFAULT_TIMEOUT):
    """GET ``<store_url>/manifest`` and return the parsed ``{name -> {sha256, size}}`` dict.

    Raises ``StoreUnreachable`` if the store can't be reached OR the manifest isn't valid JSON of the
    expected shape (a manifest we can't parse is as useless as an outage — never guess)."""
    body = _http_get(f"{store_url}/manifest", timeout)
    try:
        manifest = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise StoreUnreachable(f"{store_url}/manifest: not valid JSON ({e})") from e
    if not isinstance(manifest, dict):
        raise StoreUnreachable(f"{store_url}/manifest: expected a JSON object, got {type(manifest).__name__}")
    return manifest


def _sha256_file(path):
    """sha256 of a file's bytes, or ``None`` if it doesn't exist. Streamed so a large corpus file
    doesn't have to be fully resident to compare against the manifest."""
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _corpus_entries(manifest):
    """Yield ``(basename, sha256hex)`` for every ``corpus/<basename>`` entry in the manifest, skipping
    malformed rows (a partial/garbled manifest yields fewer files, never a crash). ``..``/separators
    in a name are rejected so a malicious manifest can't write outside the target dir (path
    traversal)."""
    for name, meta in manifest.items():
        if not isinstance(name, str) or not name.startswith(CORPUS_PREFIX):
            continue
        base = name[len(CORPUS_PREFIX):]
        # reject traversal / nested paths — corpus files are flat basenames under the target dir
        if not base or base != os.path.basename(base) or base in (".", ".."):
            continue
        sha = (meta or {}).get("sha256") if isinstance(meta, dict) else None
        if not isinstance(sha, str) or len(sha) != 64:
            continue
        yield base, sha.lower()


def _atomic_write(path, data):
    """Write ``data`` to ``path`` atomically: a temp file in the SAME directory + ``os.replace`` (an
    atomic rename on the same filesystem). A crash can leave the ``.tmp`` behind but never a
    half-written corpus file, so ``corpus_sha()`` can never read a torn write."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def sync_corpus(store_url, target_dir, timeout=_DEFAULT_TIMEOUT, verbose=True):
    """Sync every ``corpus/<file>`` object from the content store into ``target_dir``, byte-exact.

    For each corpus entry in the manifest: skip it if the local file already matches the manifest
    sha256; otherwise download it, VERIFY ``sha256(body) == manifest hash`` before writing, and write
    atomically. Returns a :class:`SyncResult`.

    Raises:
      StoreUnreachable  the store/manifest could not be reached or parsed (fall back to the checkout).
      HashMismatch      a downloaded body's hash didn't match the manifest (tamper) — the offending
                        file is NOT written; already-synced files stay as they were.
      CorpusSyncError   a manifest-listed object 404'd.

    A partial manifest (fewer ``corpus/`` entries) simply syncs fewer files — not an error.
    """
    manifest = fetch_manifest(store_url, timeout=timeout)
    entries = list(_corpus_entries(manifest))
    updated, skipped = [], []
    if verbose and not entries:
        print(f"[corpus-sync] store manifest has no 'corpus/*' entries — nothing to sync "
              f"(falling back to the checkout's bytes).", flush=True)
    for base, want_sha in entries:
        dest = os.path.join(target_dir, base)
        have = _sha256_file(dest)
        if have == want_sha:
            skipped.append(base)
            if verbose:
                print(f"[corpus-sync] {base}: already byte-exact ({want_sha[:12]}…) — skip", flush=True)
            continue
        body = _http_get(f"{store_url}/o/{want_sha}", timeout)
        got = hashlib.sha256(body).hexdigest()
        if got != want_sha:
            raise HashMismatch(
                f"{base}: store served bytes hashing to {got[:16]}… but manifest promised "
                f"{want_sha[:16]}… — REFUSING to write (corrupt or malicious store).")
        _atomic_write(dest, body)
        updated.append(base)
        if verbose:
            was = "updated" if have else "created"
            print(f"[corpus-sync] {base}: {was} {len(body)} bytes, sha {want_sha[:12]}…", flush=True)
    return SyncResult(updated, skipped, target_dir)
