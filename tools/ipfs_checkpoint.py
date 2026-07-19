"""Quota-free checkpoint distribution over IPFS — the heavy-data half of the "VPS is just a tiny
tracker" design.

The pool coordinator PUBLISHES each durable checkpoint to IPFS (content-addressed: the CID *is* the
sha256-style hash of the bytes, so it is self-verifying) and records only the CID + round metadata in a
tiny tracker file. A joining/remote miner READS the tracker (a few hundred bytes over the VPS or any
mirror), then FETCHES the checkpoint by CID — either from its own IPFS daemon or, with no local daemon,
from any public HTTP gateway. The bulk bytes ride IPFS's free relay/DHT/hole-punch infrastructure, so
the metered VPS never carries them and no home port-forward is needed.

MEASURED 2026-07-03 (home box behind a port-restricted NAT, no port-forward): a 207 MB real checkpoint
published from the home node and re-fetched byte-exact from a public gateway in ~20 s (~10 MB/s); a
20 MB blob in 1.4 s (~14 MB/s) — faster than the VPS relay at its healthiest.

This module shells out to a `kubo` (`ipfs`) binary for publish (the node that HOSTS the bytes must run a
daemon), and uses plain HTTP gateways for fetch (a miner needs NO daemon to pull). Verification is by
CID: the fetched bytes are re-hashed and MUST reproduce the CID, so a malicious gateway cannot substitute
content. Pure stdlib on the fetch path; publish requires the ipfs binary + a running daemon.
"""
import datetime
import hashlib
import hmac
import http.client
import json
import os
import shutil
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

# Public HTTP gateways, fastest-first by the 2026-07-03 measurement. A miner tries them in order; any
# one serving the CID is sufficient (they are interchangeable — the CID pins the exact bytes).
DEFAULT_GATEWAYS = (
    "https://gateway.pinata.cloud/ipfs/{cid}",   # a Pinata-pinned CID is served here immediately (no DHT wait)
    "https://ipfs.filebase.io/ipfs/{cid}",       # a Filebase-pinned CID (the S3 backstop) is served here immediately
    "https://{cid}.ipfs.w3s.link",
    "https://{cid}.ipfs.dweb.link",
    "https://ipfs.io/ipfs/{cid}",
)

IPFS_BIN = os.environ.get("NEURAHASH_IPFS_BIN", "ipfs")   # path to the kubo binary on a hosting node
_UA = "Mozilla/5.0 (NeuraHash-pool ipfs_checkpoint) Gecko/20100101"   # gateways 403 the default urllib UA

# --------------------------------------------------------------------------- resilient PUT (WAN uploads)
# Incident this guards against (2026-07-09/10, docs/research/diloco-p0-kickoff-2026-07-10.md:22-24): a
# fleet contributor's ~38 MB trunk-delta PUT to the anchor VPS content_store died with a bare
# urllib.request.urlopen library-default socket timeout and NO retry, leaving the node publishing-stale
# for 19+ hours while it kept training. NEURAHASH_PUT_TIMEOUT / NEURAHASH_PUT_RETRIES tune the per-attempt
# timeout and total attempt budget for every large-payload PUT below (announce_pin, push_named_to_store,
# pin_file_to_pinata, and diloco_contributor.publish_delta's registry PUT). Small control-plane GETs
# (fetch, read_tracker*, known_pinners, _store_get_named) are untouched and keep their own timeouts.
PUT_TIMEOUT = float(os.environ.get("NEURAHASH_PUT_TIMEOUT", "180"))     # seconds per attempt
PUT_RETRIES = int(os.environ.get("NEURAHASH_PUT_RETRIES", "3"))         # total attempts (not extra retries)
_PUT_BACKOFF_S = (5, 15, 45)                                            # wait before attempt 2, 3, 4, ...
# transient (retryable) failure classes: timeouts, DNS/connection errors, and (via _PutHTTPError) 5xx.
# NOT retryable: urllib.error.HTTPError / _PutHTTPError carrying a 4xx status (auth/logic error) --
# handled explicitly in _put_retry below, never falls into this tuple's blanket retry.
_PUT_TRANSIENT_EXC = (TimeoutError, urllib.error.URLError, ConnectionError, OSError, http.client.HTTPException)


class _PutHTTPError(RuntimeError):
    """A completed (non-exception) HTTP response with a non-2xx status, raised by the http.client-based
    PUT paths (pin_file_to_pinata) so _put_retry can apply the same 4xx-never-retry / 5xx-retry rule it
    applies to urllib.error.HTTPError. Carries `.status` for that check."""
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _put_retry(attempt_fn, *, label, retries=None):
    """Call `attempt_fn()` (one full PUT/POST attempt; must raise on failure, return on success) up to
    `retries` times total, with exponential backoff between attempts. A 4xx response (bad auth, bad
    request -- not going to fix itself) surfaces on the FIRST attempt and is NEVER retried; a timeout,
    connection error, or 5xx is retried up to the budget. Logs one ASCII line per retry (attempt N/M,
    seconds waited, target) so a stalled publish is visible instead of silently stuck for hours."""
    attempts = max(1, PUT_RETRIES if retries is None else int(retries))
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return attempt_fn()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                raise                                     # auth/logic error -- never retry
            last_exc = e
        except _PutHTTPError as e:
            if 400 <= e.status < 500:
                raise                                      # auth/logic error -- never retry
            last_exc = e
        except _PUT_TRANSIENT_EXC as e:
            last_exc = e
        if attempt < attempts:
            wait = _PUT_BACKOFF_S[min(attempt - 1, len(_PUT_BACKOFF_S) - 1)]
            print(f"[ipfs_checkpoint] put retry {attempt}/{attempts} to {label}: "
                  f"waiting {wait}s after {type(last_exc).__name__}: {last_exc}")
            time.sleep(wait)
    raise last_exc


def publish(path, ipfs_bin=None, announce=True, pin=True, timeout=600, record_path=None):
    """Add `path` to the local IPFS node and return its CIDv1. The node must be running `ipfs daemon`
    for the content to be retrievable by others. `pin` runs `ipfs pin add <cid>` so the node KEEPS
    serving the checkpoint across garbage-collection (this is the self-hosted replacement for Pinata's
    pin — the "be your own Pinata" availability layer). `announce` provides the CID to the DHT so public
    gateways can locate it promptly (best-effort; the daemon also reprovides on its own cadence).
    `record_path` overrides where the published CID is recorded (see _record_local_pin)."""
    ipfs_bin = ipfs_bin or IPFS_BIN
    cid = subprocess.check_output(
        [ipfs_bin, "add", "-q", "--cid-version=1", path], timeout=timeout).decode().strip().splitlines()[-1]
    if pin:
        # keep the copy across GC so this node stays a durable seeder; best-effort — `ipfs add` already
        # pins by default on most kubo builds, so a failure here is not fatal to hosting the CID.
        try:
            subprocess.run([ipfs_bin, "pin", "add", cid],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            pass
        _record_local_pin(cid, record_path=record_path)
    if announce:
        # fire-and-forget: a slow DHT provide must never block the coordinator's round loop
        try:
            subprocess.Popen([ipfs_bin, "routing", "provide", cid],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
    return cid


# Durable published-CID record: every publish APPENDS the CID here (newest-last). Pruning derives its
# unpin set from `ipfs pin ls` GROUND TRUTH intersected with this record — the record identifies which
# pins are OURS (checkpoints, vs corpus shards / other tenants of the daemon that must never be
# touched) and its order defines "newest N kept"; pin-ls says what is actually still pinned. The
# record alone is NOT truth (#138: it used to be trimmed even when `ipfs pin rm` silently failed, so
# every failed unpin escaped all future prunes — 369 pins / 125.6 GB accumulated against a keep of 3).
# A CID now leaves the record only after its unpin is VERIFIED (or pin-ls shows it already gone).
_LOCAL_PIN_RECORD = os.environ.get(
    "NEURAHASH_IPFS_PIN_RECORD",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ipfs_local_pins.json"))


def _record_local_pin(cid, record_path=None):
    """Append `cid` (if new) to the local newest-last pin ledger. Best-effort — a write failure must not
    break publishing (the pin still exists; only pruning bookkeeping is affected)."""
    record_path = record_path or _LOCAL_PIN_RECORD
    try:
        cids = []
        if os.path.exists(record_path):
            with open(record_path) as f:
                cids = json.load(f)
        if cid in cids:                                        # keep it newest-last: drop the old position
            cids.remove(cid)
        cids.append(cid)
        tmp = record_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cids, f)
        os.replace(tmp, record_path)
    except (OSError, ValueError):
        pass


def _ipfs_pin_ls(ipfs_bin=None, timeout=120):
    """GROUND-TRUTH pin inventory: the recursive pins the local kubo daemon holds RIGHT NOW
    (`ipfs pin ls --type=recursive -q`). Returns a set of CID strings. Raises on any failure —
    callers must treat "cannot list pins" as "do not prune" rather than pruning blind."""
    ipfs_bin = ipfs_bin or IPFS_BIN
    p = subprocess.run([ipfs_bin, "pin", "ls", "--type=recursive", "-q"], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"`ipfs pin ls` failed (rc {p.returncode}): {(p.stderr or '').strip()[:300]}")
    return {ln.split()[0] for ln in (p.stdout or "").splitlines() if ln.strip()}


def prune_local_pins(keep=3, ipfs_bin=None, record_path=None):
    """Keep the newest `keep` published checkpoint pins; `ipfs pin rm` every OTHER pin the durable
    published-CID record attributes to a checkpoint publish. #138 rewrite — the unpin set is derived
    from `ipfs pin ls` GROUND TRUTH intersected with the record (NOT from in-process history), so
    checkpoint pins accumulated by PRIOR coordinator processes are pruned too, and a CID leaves the
    record only once its unpin is VERIFIED (rc 0 / already unpinned). Failed unpins stay recorded and
    are RETRIED next cycle, and every failure is LOGGED LOUDLY — silently forgetting them is how 369
    pins / 125.6 GB piled up against a keep of 3. Pins NOT in the record (corpus shards, other
    tenants) are never touched. Never raises; returns the CIDs actually unpinned."""
    ipfs_bin = ipfs_bin or IPFS_BIN
    record_path = record_path or _LOCAL_PIN_RECORD
    if not os.path.exists(record_path):
        return []
    try:
        with open(record_path) as f:
            cids = json.load(f)
        if not isinstance(cids, list):
            raise ValueError(f"pin record holds a {type(cids).__name__}, expected a list")
    except (OSError, ValueError) as e:
        print(f"[ipfs_checkpoint] pin prune SKIPPED: record {record_path} unreadable ({e})", flush=True)
        return []
    try:
        pinned = _ipfs_pin_ls(ipfs_bin=ipfs_bin)
    except (OSError, subprocess.SubprocessError, RuntimeError) as e:
        print(f"[ipfs_checkpoint] pin prune SKIPPED: cannot read `ipfs pin ls` ground truth "
              f"({type(e).__name__}: {e}) — refusing to prune blind", flush=True)
        return []
    keep_tail = set(cids[-keep:]) if keep > 0 else set()
    dropped, failed = [], set()
    for cid in cids:
        if cid in keep_tail or cid not in pinned:              # keeper, or already gone from the daemon
            continue
        try:
            p = subprocess.run([ipfs_bin, "pin", "rm", cid], capture_output=True,
                               encoding="utf-8", errors="replace", timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            failed.add(cid)
            print(f"[ipfs_checkpoint] UNPIN FAILED (kept in record for retry): {cid}: "
                  f"{type(e).__name__}: {e}", flush=True)
            continue
        err = (p.stderr or "").strip()
        if p.returncode == 0 or "not pinned" in err:           # unpinned (or a racer beat us to it)
            dropped.append(cid)
        else:
            failed.add(cid)
            print(f"[ipfs_checkpoint] UNPIN FAILED (rc {p.returncode}, kept in record for retry): "
                  f"{cid}: {err[:300]}", flush=True)
    # rewrite the record: the keep-tail plus every FAILED unpin (order preserved, newest-last) —
    # never drop a CID we have not verifiably unpinned; already-gone CIDs are bookkeeping catch-up.
    kept = [c for c in cids if c in keep_tail or c in failed]
    try:
        tmp = record_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(kept, f)
        os.replace(tmp, record_path)
    except OSError as e:
        print(f"[ipfs_checkpoint] pin record rewrite FAILED ({record_path}): {e}", flush=True)
    if dropped or failed:
        print(f"[ipfs_checkpoint] pin prune: unpinned {len(dropped)}, failed {len(failed)} "
              f"(keep {keep}, daemon holds {len(pinned)} recursive pins)", flush=True)
    return dropped


def ipfs_repo_gc(ipfs_bin=None, timeout=600):
    """Run `ipfs repo gc` to reclaim unpinned blocks. kubo NEVER garbage-collects on its own unless
    the daemon was started with --enable-gc (#138: RepoSize hit 125.6 GB vs a 10 GB StorageMax), so
    the publisher runs one bounded gc after each prune cycle that actually unpinned something.
    Best-effort: failures are LOGGED loudly and swallowed (a slow or failed gc must never break
    publishing). Returns the number of blocks removed, or -1 on failure."""
    ipfs_bin = ipfs_bin or IPFS_BIN
    try:
        p = subprocess.run([ipfs_bin, "repo", "gc", "-q"], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[ipfs_checkpoint] ipfs repo gc FAILED: {type(e).__name__}: {e}", flush=True)
        return -1
    if p.returncode != 0:
        print(f"[ipfs_checkpoint] ipfs repo gc FAILED (rc {p.returncode}): "
              f"{(p.stderr or '').strip()[:300]}", flush=True)
        return -1
    n = sum(1 for ln in (p.stdout or "").splitlines() if ln.strip())
    print(f"[ipfs_checkpoint] ipfs repo gc: removed {n} block(s)", flush=True)
    return n


# --------------------------------------------------------------------------- disk-floor preflight (#138)
IPFS_MIN_FREE_GB = float(os.environ.get("NEURAHASH_IPFS_MIN_FREE_GB", "10"))


def ipfs_repo_free_gb(path=None):
    """Free GB (binary, 2**30) on the drive holding the kubo repo (env IPFS_PATH, default ~/.ipfs).
    Walks up to the nearest EXISTING ancestor so a not-yet-initialized repo dir still resolves to its
    drive. Returns None when free space cannot be measured — callers FAIL OPEN on None (an
    unmeasurable disk must not stop publishing; a MEASURED low disk must)."""
    path = path or (os.environ.get("IPFS_PATH", "") or "").strip() or os.path.expanduser("~/.ipfs")
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free / (1 << 30)
    except OSError:
        return None


def publish_preflight_ok(min_free_gb=None):
    """DISK-FLOOR preflight (#138): True iff the kubo repo's drive has at least
    NEURAHASH_IPFS_MIN_FREE_GB (default 10) GB free. Below the floor it logs ONE loud line and
    returns False — the caller skips that publish (adding ~checkpoint-size bytes per cycle is what
    ran D: from 41 GB to 1.5 GB overnight) and should prune+gc instead so the repo can shrink back
    under the floor without an operator. Fails OPEN (True) when free space is unmeasurable."""
    floor = IPFS_MIN_FREE_GB if min_free_gb is None else float(min_free_gb)
    free = ipfs_repo_free_gb()
    if free is None or free >= floor:
        return True
    print(f"[ipfs_checkpoint] publish SKIPPED by disk floor: {free:.1f} GB free on the IPFS repo "
          f"drive < NEURAHASH_IPFS_MIN_FREE_GB={floor:g} — not adding bytes; prune+gc must catch up",
          flush=True)
    return False


def _ipfs_get(cid, dest, ipfs_bin=None, timeout=600):
    """Fetch `cid` to `dest` via the LOCAL ipfs daemon (`ipfs get`), pulling peer-to-peer over the
    swarm/relay from whichever node is seeding it. Returns True on success, False if no local ipfs is
    available or the get failed (caller then falls back to public gateways). `ipfs get -o` writes the
    raw file for a single-file CID."""
    ipfs_bin = ipfs_bin or IPFS_BIN
    try:
        proc = subprocess.run([ipfs_bin, "get", "-o", dest, cid],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        return proc.returncode == 0 and os.path.exists(dest)
    except (OSError, subprocess.SubprocessError):
        return False


def fetch(cid, dest, gateways=DEFAULT_GATEWAYS, timeout=180, verify_cid=True,
          pinners=None, ipfs_bin=None):
    """Fetch `cid` to `dest` over the first working public gateway (NO local daemon needed). Returns the
    source ("ipfs:<bin>" or the gateway URL) that served it. Raises if every source fails or (when
    verify_cid) the bytes don't reproduce the CID. Trust is in the CID, not the source — a substituted
    body is rejected.

    ADDITIVE peer-to-peer fast path: if `pinners` are known (live seeders from the registry) AND a local
    ipfs daemon is present, try `ipfs get <cid>` FIRST (pulls straight from the swarm/relay, no gateway
    round-trip); on any failure it falls through to the public-gateway path below UNCHANGED. With no
    pinners and no local ipfs, behaviour is byte-identical to gateway-only fetch."""
    if pinners:                                                # a live seeder is claimed -> try the swarm first
        if _ipfs_get(cid, dest, ipfs_bin=ipfs_bin, timeout=max(timeout, 600)):
            if not verify_cid:
                return f"ipfs:{ipfs_bin or IPFS_BIN}"
            with open(dest, "rb") as f:
                if _cid_matches(cid, f.read()):
                    return f"ipfs:{ipfs_bin or IPFS_BIN}"
            # daemon returned bytes that don't reproduce the CID (shouldn't happen) -> discard, use gateways
            try:
                os.remove(dest)
            except OSError:
                pass
    last = None
    for tmpl in gateways:
        url = tmpl.format(cid=cid)
        try:
            # some public gateways (ipfs.io) 403 the default `Python-urllib/x` UA as a bot — send a
            # normal UA. Trust is still in the CID (verified below), never the gateway.
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if verify_cid and not _cid_matches(cid, data):
                last = f"{url}: CID verification FAILED (body != {cid})"
                continue
            with open(dest, "wb") as f:
                f.write(data)
            return url
        except Exception as e:                                 # noqa: BLE001 — any gateway error -> try next
            last = f"{url}: {e!r}"
            continue
    raise RuntimeError(f"all gateways failed for {cid}; last: {last}")


def _cid_matches(cid, data):
    """Verify fetched bytes reproduce `cid` by re-adding them offline (`ipfs add -n`, no network) and
    comparing. VERSION-AWARE: a CIDv0 (base58 "Qm...") is re-added with --cid-version=0, everything else
    (a CIDv1, e.g. base32 "bafy...") with --cid-version=1 — re-adding under the wrong version yields a
    different CID and would hard-reject an HONEST fetch. Our own publish paths emit CIDv1 (publish() and
    pin_file_to_pinata(cid_version=1)), but Pinata defaults to CIDv0 if a future caller unsets cidVersion,
    so matching the version to the CID keeps verification correct for both. Falls back to True only if no
    ipfs binary is available (HTTPS + gateway trust), and says so via the env flag so an operator can
    require strict verification."""
    ipfs_bin = IPFS_BIN
    cid_version = "0" if cid.startswith("Qm") else "1"        # CIDv0 multihash "Qm..."; else treat as CIDv1
    try:
        proc = subprocess.run([ipfs_bin, "add", "-qn", f"--cid-version={cid_version}"],
                              input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
        got = proc.stdout.decode().strip().splitlines()[-1] if proc.stdout else ""
        return got == cid
    except (OSError, subprocess.SubprocessError):
        # no local ipfs to verify with: allow only if the operator hasn't demanded strict verification
        return os.environ.get("NEURAHASH_IPFS_STRICT", "") not in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- Pinata pinning service
# Public gateways don't PIN (content they haven't served recently can fall out of cache), so a checkpoint
# nobody fetched for a while may be slow to re-locate. A pinning service keeps an always-on hosted copy so
# a remote miner's fetch is fast and reliable regardless of whether the home node is up. Pinata's free tier
# hosts the bytes; we upload each checkpoint and prune old ones to stay lean. The JWT rides an HTTPS header
# (never the process command line), streamed so a 200 MB checkpoint never loads fully into RAM.
PINATA_HOST = "api.pinata.cloud"


def _pinata_jwt(jwt=None):
    """Resolve the Pinata JWT: explicit arg > PINATA_JWT env > PINATA_JWT_FILE (a path, keeps the secret
    out of env dumps / process listings). Returns '' if none configured."""
    jwt = (jwt or os.environ.get("PINATA_JWT", "")).strip()
    if not jwt:
        f = os.environ.get("PINATA_JWT_FILE", "").strip()
        if f and os.path.exists(f):
            with open(f) as fh:
                jwt = fh.read().strip()
    return jwt


def pin_file_to_pinata(path, jwt=None, name=None, cid_version=1, timeout=1800, retries=None):
    """Stream-upload `path` to Pinata (which HOSTS + pins it) and return the pinned CID. No local IPFS
    daemon needed on the publisher — Pinata becomes the always-on host. Streams the file in 1 MiB chunks
    (a big checkpoint never fully enters RAM) and passes the JWT as an HTTPS header (not the cmd line).
    `timeout` (default 1800s) is the generous per-attempt socket timeout for potentially 100s-of-MB
    checkpoints -- left unchanged rather than shrunk to NEURAHASH_PUT_TIMEOUT's 180s default, so this
    stays semantics-preserving for large real checkpoints. The upload is retried per NEURAHASH_PUT_RETRIES
    / `retries` (see _put_retry) on a timeout, connection error, or 5xx -- never on a 4xx (bad JWT/request
    surfaces immediately). This closes the 2026-07-10 incident: a contributor's ~38 MB delta upload died
    on ONE silent library-default timeout with no retry and stayed stuck for 19+ hours."""
    jwt = _pinata_jwt(jwt)
    if not jwt:
        raise RuntimeError("no Pinata JWT (set PINATA_JWT or PINATA_JWT_FILE)")
    name = name or os.path.basename(path)

    def _attempt():
        boundary = "----neurahash" + os.urandom(16).hex()

        def _field(fieldname, value):
            return (f'--{boundary}\r\nContent-Disposition: form-data; name="{fieldname}"\r\n\r\n'
                    f'{value}\r\n').encode()

        pre = _field("pinataOptions", json.dumps({"cidVersion": cid_version}))
        pre += _field("pinataMetadata", json.dumps({"name": name}))
        pre += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                f'Content-Type: application/octet-stream\r\n\r\n').encode()
        post = f'\r\n--{boundary}--\r\n'.encode()
        clen = len(pre) + os.path.getsize(path) + len(post)

        conn = http.client.HTTPSConnection(PINATA_HOST, timeout=timeout, context=ssl.create_default_context())
        try:
            conn.putrequest("POST", "/pinning/pinFileToIPFS", skip_host=False, skip_accept_encoding=True)
            conn.putheader("Authorization", f"Bearer {jwt}")
            conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            conn.putheader("Content-Length", str(clen))
            conn.endheaders()
            conn.send(pre)
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    conn.send(chunk)
            conn.send(post)
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise _PutHTTPError(resp.status, f"Pinata pin failed HTTP {resp.status}: {data[:300]!r}")
            return json.loads(data)["IpfsHash"]
        finally:
            conn.close()

    return _put_retry(_attempt, label=f"Pinata pinFileToIPFS {name}", retries=retries)


def _pinata_api(method, path, jwt, timeout=60):
    """Small JSON call to the Pinata API with the JWT as a header. Returns parsed JSON (or {} on 200
    with empty body). Raises on non-2xx."""
    conn = http.client.HTTPSConnection(PINATA_HOST, timeout=timeout, context=ssl.create_default_context())
    try:
        conn.putrequest(method, path, skip_accept_encoding=True)
        conn.putheader("Authorization", f"Bearer {jwt}")
        conn.endheaders()
        resp = conn.getresponse()
        body = resp.read()
        if not (200 <= resp.status < 300):
            raise RuntimeError(f"Pinata {method} {path} -> HTTP {resp.status}: {body[:300]!r}")
        # some endpoints (unpin) return a plain "OK" body, not JSON — a 2xx is success regardless.
        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return {"_raw": body.decode(errors="replace")}
    finally:
        conn.close()


def unpin_pinata(cid, jwt=None):
    """Remove a pin (frees free-tier storage). Best-effort — a already-gone pin is not an error."""
    jwt = _pinata_jwt(jwt)
    if not jwt:
        return
    try:
        _pinata_api("DELETE", f"/pinning/unpin/{cid}", jwt)
    except RuntimeError as e:
        if "HTTP 404" not in str(e):
            raise


def prune_pinata(keep=3, name_prefix="neurahash-ckpt", jwt=None):
    """Keep only the newest `keep` pins whose metadata name starts with `name_prefix`; unpin the rest.
    Stops the free tier filling with stale checkpoints. Returns the list of unpinned CIDs."""
    jwt = _pinata_jwt(jwt)
    if not jwt:
        return []
    got = _pinata_api("GET", "/data/pinList?status=pinned&pageLimit=100", jwt)
    rows = [r for r in got.get("rows", [])
            if (r.get("metadata") or {}).get("name", "").startswith(name_prefix)]
    rows.sort(key=lambda r: r.get("date_pinned") or "", reverse=True)   # newest first
    dropped = []
    for r in rows[keep:]:
        cid = r.get("ipfs_pin_hash")
        if cid:
            unpin_pinata(cid, jwt)
            dropped.append(cid)
    return dropped


# --------------------------------------------------------------------------- free pin-service backstop (Filebase / 4EVERLAND / any IPFS-Pinning-Service-API provider)
# A free commodity pin service keeps an always-on OFF-MACHINE copy of a checkpoint the home node is
# already seeding — the durability backstop for when the home daemon is offline. Unlike the Pinata path
# (which UPLOADS the bytes via a bespoke endpoint), this speaks the STANDARD IPFS Pinning Service API
# (POST /pins by CID + Bearer token), so it is VENDOR-AGNOSTIC: point NEURAHASH_PIN_SERVICE_URL at
# Filebase (default), 4EVERLAND, web3.storage, etc. and only the URL + token change. The service FETCHES
# the CID from IPFS (while the home node / Pinata seeds it) and pins it, so a later remote fetch by CID
# succeeds even with the home node down. Free tiers are small (Filebase: 5 GB, no credit card) so this is
# a BACKSTOP, not the primary host — old pins are pruned to stay under the cap.
PIN_SERVICE_URL_DEFAULT = "https://api.filebase.io/v1/ipfs"   # Filebase IPFS-Pinning-Service-API base


def _pin_service_cfg(url=None, token=None):
    """Resolve (base_url, token): explicit args > NEURAHASH_PIN_SERVICE_URL / _TOKEN env >
    NEURAHASH_PIN_SERVICE_TOKEN_FILE (a path — keeps the secret out of env dumps / process listings) >
    the Filebase default URL. Returns (url, token); token == '' means NO service is configured (the
    backstop is OFF, default). The token is created in the provider console (Filebase: an IPFS 'Access
    Token'); it never touches the command line."""
    url = (url or os.environ.get("NEURAHASH_PIN_SERVICE_URL", "") or PIN_SERVICE_URL_DEFAULT).strip().rstrip("/")
    token = (token or os.environ.get("NEURAHASH_PIN_SERVICE_TOKEN", "")).strip()
    if not token:
        f = os.environ.get("NEURAHASH_PIN_SERVICE_TOKEN_FILE", "").strip()
        if f and os.path.exists(f):
            with open(f) as fh:
                token = fh.read().strip()
    return url, token


def _pin_service_api(method, path, token, url=None, body=None, timeout=120):
    """One JSON call to a standard IPFS Pinning Service API. `path` (e.g. '/pins', '/pins?cid=...') is
    appended to the service base. Bearer token in the header (never the cmd line). Returns parsed JSON
    ({} on an empty 2xx body). Raises RuntimeError on non-2xx (message carries the status + body head)."""
    base, _ = _pin_service_cfg(url, token)
    parsed = urllib.parse.urlparse(base + path)
    rel = parsed.path + (("?" + parsed.query) if parsed.query else "")
    payload = json.dumps(body).encode() if body is not None else None
    conn = http.client.HTTPSConnection(parsed.netloc, timeout=timeout, context=ssl.create_default_context())
    try:
        conn.putrequest(method, rel, skip_accept_encoding=True)
        conn.putheader("Authorization", f"Bearer {token}")
        if payload is not None:
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(len(payload)))
        conn.endheaders()
        if payload is not None:
            conn.send(payload)
        resp = conn.getresponse()
        data = resp.read()
        if not (200 <= resp.status < 300):
            raise RuntimeError(f"pin-service {method} {parsed.path} -> HTTP {resp.status}: {data[:300]!r}")
        if not data.strip():
            return {}
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return {"_raw": data.decode(errors="replace")}
    finally:
        conn.close()


def pin_cid_to_service(cid, name=None, url=None, token=None, timeout=120):
    """Ask a free IPFS pinning service (Filebase by default) to pin an EXISTING `cid` the home node is
    seeding — the off-machine durability backstop. Standard `POST /pins {cid,name}` + Bearer token, so
    any compliant provider works. Returns the service `requestid` (or '' if no token is configured, i.e.
    the backstop is OFF). The service pins ASYNCHRONOUSLY (queued -> pinning -> pinned); poll
    `pin_service_status` to confirm it holds the CID before relying on it as a backstop."""
    _, token = _pin_service_cfg(url, token)
    if not token:
        return ""                                              # backstop off -> no-op, byte-for-byte as before
    res = _pin_service_api("POST", "/pins", token, url=url,
                           body={"cid": cid, "name": name or ("neurahash-" + str(cid))}, timeout=timeout)
    return res.get("requestid", "")


def pin_service_status(cid=None, requestid=None, url=None, token=None, timeout=60):
    """Return the pin status ('queued'|'pinning'|'pinned'|'failed'|'unknown') for a CID (GET /pins?cid=)
    or a requestid (GET /pins/<id>). '' if no token. Use it to confirm the backstop actually holds the
    CID (status 'pinned') before the home node goes offline."""
    _, token = _pin_service_cfg(url, token)
    if not token:
        return ""
    if requestid:
        return _pin_service_api("GET", f"/pins/{requestid}", token, url=url, timeout=timeout).get("status", "unknown")
    res = _pin_service_api("GET", f"/pins?cid={urllib.parse.quote(str(cid))}", token, url=url, timeout=timeout)
    rows = res.get("results") or res.get("rows") or []
    return rows[0].get("status", "unknown") if rows else "unknown"


def unpin_cid_from_service(requestid, url=None, token=None, timeout=60):
    """DELETE /pins/<requestid> — free the backstop's (small, free-tier) storage. Best-effort; an
    already-gone pin (404) is not an error."""
    _, token = _pin_service_cfg(url, token)
    if not token or not requestid:
        return
    try:
        _pin_service_api("DELETE", f"/pins/{requestid}", token, url=url, timeout=timeout)
    except RuntimeError as e:
        if "HTTP 404" not in str(e):
            raise


def prune_service(keep=3, name_prefix="neurahash-ckpt", url=None, token=None, timeout=120):
    """Keep only the newest `keep` service pins whose name starts with `name_prefix`; unpin the rest so
    the small free tier (Filebase 5 GB) does not fill with stale checkpoints. Returns the unpinned CIDs,
    newest-first by the API's `created` timestamp. No-op (returns []) if no service is configured."""
    _, token = _pin_service_cfg(url, token)
    if not token:
        return []
    res = _pin_service_api("GET", f"/pins?name={urllib.parse.quote(name_prefix)}&limit=1000",
                           token, url=url, timeout=timeout)
    rows = [r for r in (res.get("results") or res.get("rows") or [])
            if ((r.get("pin") or {}).get("name", "")).startswith(name_prefix)]
    rows.sort(key=lambda r: r.get("created") or "", reverse=True)   # newest first
    dropped = []
    for r in rows[keep:]:
        rid = r.get("requestid")
        if rid:
            unpin_cid_from_service(rid, url=url, token=token)
            dropped.append((r.get("pin") or {}).get("cid") or rid)
    return dropped


# --------------------------------------------------------------------------- Filebase S3 backstop (direct upload with the owner's S3 access key)
# Filebase also exposes an S3-compatible API (s3.filebase.io): a signed PUT stores the bytes AND pins
# them to IPFS, and the object's CID comes back on a HEAD as `x-amz-meta-cid`. This path UPLOADS the
# checkpoint DIRECTLY (unlike the pinning-service path, which pins a CID the home node must still be
# seeding), so the off-machine copy survives even if the home node dies the instant after publish.
# Signed with stdlib AWS SigV4 (no boto3). Creds live in a local file (default firebase/accesskey.txt,
# gitignored); the secret never touches the command line or a log.
S3_ENDPOINT_DEFAULT = "https://s3.filebase.io"
S3_REGION_DEFAULT = "us-east-1"   # Filebase signs SigV4 as us-east-1 (a creds-file 'region: auto' is NOT a signing region)


def read_s3_creds(path=None):
    """Read Filebase S3 credentials from a small creds file in the console's copy-paste format
    (`key: 'value'` lines: endpoint / accessKeyId / secretAccessKey / region). Path = arg >
    NEURAHASH_FILEBASE_KEYFILE env > <repo>/firebase/accesskey.txt. Returns {endpoint, access_key,
    secret_key, region} (region 'auto'/'' -> us-east-1), or None if no usable file / no secret."""
    path = path or os.environ.get("NEURAHASH_FILEBASE_KEYFILE", "") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "firebase", "accesskey.txt")
    if not os.path.exists(path):
        return None
    kv = {}
    with open(path) as fh:
        for line in fh:
            if ":" in line:
                k, _, v = line.partition(":")
                kv[k.strip()] = v.strip().strip("',\"")                # strip surrounding quotes/commas
    access, secret = kv.get("accessKeyId", ""), kv.get("secretAccessKey", "")
    if not access or not secret:
        return None
    region = kv.get("region", "") or ""
    return {"endpoint": (kv.get("endpoint", "") or S3_ENDPOINT_DEFAULT).rstrip("/"),
            "access_key": access, "secret_key": secret,
            "region": S3_REGION_DEFAULT if region in ("", "auto") else region}


def _sigv4_signing_key(secret, date, region, service):
    """AWS SigV4 signing key: HMAC-SHA256 chain over ('AWS4'+secret) -> date -> region -> service ->
    'aws4_request'. Validated against AWS's published derive-signing-key test vector in the tests."""
    def _h(key, msg):
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
    return _h(_h(_h(_h(("AWS4" + secret).encode("utf-8"), date), region), service), "aws4_request")


def _s3_auth(method, endpoint, bucket, key, creds, amz_date, payload_hash="UNSIGNED-PAYLOAD"):
    """Build the SigV4 Authorization + required headers for a path-style S3 request
    `<endpoint>/<bucket>/<key>`. `amz_date`='YYYYMMDDTHHMMSSZ'. Returns (host, canonical_uri, headers)."""
    host = endpoint.split("://", 1)[-1].split("/", 1)[0]
    date = amz_date[:8]
    canonical_uri = "/" + urllib.parse.quote(bucket, safe="") + "/" + urllib.parse.quote(key, safe="/")
    canonical_headers = f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join([method, canonical_uri, "", canonical_headers, signed_headers, payload_hash])
    scope = f"{date}/{creds['region']}/s3/aws4_request"
    string_to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope,
                                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()])
    signing_key = _sigv4_signing_key(creds["secret_key"], date, creds["region"], "s3")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (f"AWS4-HMAC-SHA256 Credential={creds['access_key']}/{scope}, "
                     f"SignedHeaders={signed_headers}, Signature={signature}")
    return host, canonical_uri, {"Host": host, "x-amz-date": amz_date,
                                 "x-amz-content-sha256": payload_hash, "Authorization": authorization}


def _amz_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def s3_object_cid(bucket, key, creds=None, timeout=60, amz_date=None):
    """HEAD `<bucket>/<key>` and return its IPFS CID from the `x-amz-meta-cid` response header (Filebase
    sets it once the object is pinned). '' if the header is absent (e.g. still pinning)."""
    creds = creds or read_s3_creds()
    amz_date = amz_date or _amz_now()
    empty = hashlib.sha256(b"").hexdigest()                        # HEAD has no body
    host, uri, headers = _s3_auth("HEAD", creds["endpoint"], bucket, key, creds, amz_date, payload_hash=empty)
    conn = http.client.HTTPSConnection(host, timeout=timeout, context=ssl.create_default_context())
    try:
        conn.putrequest("HEAD", uri, skip_host=True, skip_accept_encoding=True)
        for k, v in headers.items():
            conn.putheader(k, v)
        conn.endheaders()
        resp = conn.getresponse(); resp.read()
        if not (200 <= resp.status < 300):
            raise RuntimeError(f"Filebase S3 HEAD {uri} -> HTTP {resp.status}")
        return resp.getheader("x-amz-meta-cid", "") or ""
    finally:
        conn.close()


def pin_file_via_s3(path, bucket, key=None, creds=None, timeout=1800, amz_date=None):
    """Upload `path` to Filebase over S3 (SigV4-signed PUT, UNSIGNED-PAYLOAD, streamed in 1 MiB chunks so
    a big checkpoint never fully enters RAM), which stores + PINS it to IPFS, then HEAD the object to read
    its IPFS CID. Returns the CID (or '' if not yet reported). Raises on a non-2xx PUT. `creds` defaults
    to read_s3_creds(); `key` to the basename."""
    creds = creds or read_s3_creds()
    if not creds:
        raise RuntimeError("no Filebase S3 creds (firebase/accesskey.txt or NEURAHASH_FILEBASE_KEYFILE)")
    key = key or os.path.basename(path)
    amz_date = amz_date or _amz_now()
    host, uri, headers = _s3_auth("PUT", creds["endpoint"], bucket, key, creds, amz_date)
    headers["Content-Length"] = str(os.path.getsize(path))
    conn = http.client.HTTPSConnection(host, timeout=timeout, context=ssl.create_default_context())
    try:
        conn.putrequest("PUT", uri, skip_host=True, skip_accept_encoding=True)
        for k, v in headers.items():
            conn.putheader(k, v)
        conn.endheaders()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                conn.send(chunk)
        resp = conn.getresponse(); body = resp.read()
        if not (200 <= resp.status < 300):
            raise RuntimeError(f"Filebase S3 PUT {uri} -> HTTP {resp.status}: {body[:300]!r}")
    finally:
        conn.close()
    try:
        return s3_object_cid(bucket, key, creds=creds, timeout=60, amz_date=amz_date)
    except Exception:                                             # noqa: BLE001 — the bytes are stored; CID is a bonus
        return ""


# --------------------------------------------------------------------------- one-call publisher
def publish_checkpoint(ckpt_path, round_no, tracker_path, *, prefer="auto", pin_keep=3,
                       ipfs_bin=None, peers=None, ts=0, backstop="auto", pin_record=None):
    """Publish `ckpt_path` and (re)write the tiny tracker doc — the single call the coordinator makes.
    `prefer`: 'pinata' (upload to Pinata, always-on host), 'ipfs' (local daemon hosts), or 'auto'
    (Pinata if a JWT is configured, else local IPFS). `backstop` = off-machine durability copy: 's3'
    (direct upload to Filebase via S3, most robust — records the s3 pointer in the tracker), 'service'
    (pin the CID to an IPFS-Pinning-Service provider), 'off', or 'auto' (S3 iff NEURAHASH_FILEBASE_BUCKET
    + creds are set, else the pin-service iff a token is set, else nothing). Prunes old pins to
    `pin_keep`. Returns the published CID. Designed to run in a BACKGROUND thread — it does network I/O
    and must never block the coordinator's round loop; the caller swallows exceptions so a publish hiccup
    can't kill the pool. `pin_record` = durable published-CID record for the local-daemon path (#138):
    appended on every publish, consulted by the pin-ls-ground-truth prune, survives restarts."""
    name = f"neurahash-ckpt-r{round_no}"
    jwt = _pinata_jwt()
    use_pinata = prefer == "pinata" or (prefer == "auto" and jwt)
    if use_pinata:
        cid = pin_file_to_pinata(ckpt_path, jwt=jwt, name=name)
        try:
            prune_pinata(keep=pin_keep, jwt=jwt)
        except Exception as e:                                 # noqa: BLE001 — must not block publish, but LOUD (#138)
            print(f"[ipfs_checkpoint] Pinata pin prune FAILED (pins will accumulate): "
                  f"{type(e).__name__}: {e}", flush=True)
    else:
        cid = publish(ckpt_path, ipfs_bin=ipfs_bin, record_path=pin_record)
        try:                                                   # keep the local repo lean, same as Pinata prune
            if prune_local_pins(keep=pin_keep, ipfs_bin=ipfs_bin, record_path=pin_record):
                # kubo never GCs on its own — reclaim the just-unpinned blocks (bounded, logged, no raise)
                ipfs_repo_gc(ipfs_bin=ipfs_bin)
        except Exception as e:                                 # noqa: BLE001 — must not block publish, but LOUD (#138)
            print(f"[ipfs_checkpoint] pin prune FAILED (repo will grow until this is fixed): "
                  f"{type(e).__name__}: {e}", flush=True)
    # OFF-MACHINE DURABILITY BACKSTOP: keep a copy of the checkpoint off this machine so a remote fetch /
    # a coordinator resume still succeeds when the home node is offline. Best-effort — a backstop hiccup
    # must never block the round loop (the caller also swallows; this is defence-in-depth). OFF unless
    # configured. 's3' UPLOADS the bytes (survives the home node dying mid-pin); 'service' pins the CID
    # the home node is currently seeding.
    extra = {"ts": ts, "host": "pinata" if use_pinata else "ipfs"}
    fb_bucket = os.environ.get("NEURAHASH_FILEBASE_BUCKET", "").strip()
    if backstop == "s3" or (backstop == "auto" and fb_bucket and read_s3_creds()):
        try:
            s3_key = f"{name}.pt"
            s3_cid = pin_file_via_s3(ckpt_path, fb_bucket, key=s3_key)
            extra.update(s3_bucket=fb_bucket, s3_key=s3_key, s3_cid=s3_cid)   # pointer for a home-off resume
        except Exception:                                      # noqa: BLE001 — the backstop is best-effort
            pass
    elif backstop == "service" or (backstop == "auto" and _pin_service_cfg()[1]):
        try:
            pin_cid_to_service(cid, name=name)
            prune_service(keep=pin_keep, name_prefix="neurahash-ckpt")
        except Exception as e:                                 # noqa: BLE001 — the backstop is best-effort, but LOUD (#138)
            print(f"[ipfs_checkpoint] pin-service backstop pin/prune FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
    write_tracker(tracker_path, round_no, cid, peers=peers, extra=extra)
    return cid


# --------------------------------------------------------------------------- tiny tracker file
def write_tracker(path, round_no, checkpoint_cid, peers=None, extra=None):
    """Write the tiny tracker doc the VPS (or any mirror) serves: round + checkpoint CID + optional peer
    hints. A few hundred bytes — the ONLY thing the metered box moves. Atomic (temp + replace)."""
    doc = {"round": int(round_no), "checkpoint_cid": checkpoint_cid,
           "peers": list(peers or []), "ts": int(extra.pop("ts", 0)) if extra else 0}
    if extra:
        doc.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, path)
    return doc


def read_tracker(url_or_path, timeout=15):
    """Read a tracker doc from an HTTP(S) URL or a local path."""
    if url_or_path.startswith(("http://", "https://")):
        with urllib.request.urlopen(url_or_path, timeout=timeout) as r:
            return json.loads(r.read().decode())
    with open(url_or_path) as f:
        return json.load(f)


def _store_get_named(store_url, name, timeout=15):
    """GET the object registered under friendly `name` from a content store (manifest -> sha256 -> /o/<sha>).
    Returns the raw bytes, or None if the store is unreachable / the name is absent. This is how a fresh
    remote node reads a NAMED doc (e.g. the current 'tracker') without knowing its content hash up front."""
    store_url = (store_url or "").rstrip("/")
    if not store_url:
        return None
    try:
        with urllib.request.urlopen(f"{store_url}/manifest", timeout=timeout) as r:
            manifest = json.loads(r.read().decode())
        sha = (manifest.get(name) or {}).get("sha256")
        if not sha:
            return None
        with urllib.request.urlopen(f"{store_url}/o/{sha}", timeout=timeout) as r:
            return r.read()
    except Exception:                                          # noqa: BLE001 — store down / name absent -> None
        return None


def read_tracker_from_store(store_url, timeout=15):
    """Read the checkpoint tracker doc from a content store BY NAME ('tracker'). Unlike read_tracker (which
    takes a direct URL/path), this resolves the CURRENT tracker object through the store manifest — the
    path a fresh remote node uses to discover {round, checkpoint_cid, s3_cid, peers} with the home node
    offline. Returns the dict, or None if unavailable."""
    raw = _store_get_named(store_url, "tracker", timeout=timeout)
    if not raw:
        return None
    try:
        return json.loads(raw.decode())
    except (json.JSONDecodeError, ValueError):
        return None


def bootstrap_checkpoint(ckpt_file, store_url, *, ipfs_bin=None, timeout=900, max_age_s=3600):
    """COLD START: when a node has NO local checkpoint, fetch the current one from the off-machine copy so
    ANY host can resume the model with the home node offline. Reads the tracker from the content store
    (round + checkpoint_cid + optional Filebase s3_cid + known pinners), then fetches the checkpoint bytes
    to `ckpt_file`, CID-VERIFIED. Tries the most reliable source first — the Filebase-pinned copy (its own
    s3_cid, served by the authoritative ipfs.filebase.io gateway), then the local-publish CID over the
    fleet's live pinners (P2P) and public gateways. Returns {round, cid, source, pinners} on success, or
    None (no tracker / no CID / every source failed) so the caller falls back to its start-fresh path.
    NEVER raises."""
    trk = read_tracker_from_store(store_url, timeout=30)
    if not isinstance(trk, dict):
        return None
    seen = set()
    for cid in (trk.get("s3_cid") or "", trk.get("checkpoint_cid") or ""):
        if not cid or cid in seen:
            continue
        seen.add(cid)
        # fleet P2P only applies to the local-publish CID (announced to the registry); the s3 copy is
        # served by the Filebase gateway (already in DEFAULT_GATEWAYS), so it needs no pinner hints.
        pinners = known_pinners(store_url, cid, max_age_s=max_age_s) if cid == (trk.get("checkpoint_cid") or "") else []
        try:
            src = fetch(cid, ckpt_file, pinners=pinners, verify_cid=True, ipfs_bin=ipfs_bin, timeout=timeout)
            return {"round": trk.get("round", 0), "cid": cid, "source": src, "pinners": len(pinners)}
        except Exception:                                      # noqa: BLE001 — try the next source, then give up
            continue
    return None


# --------------------------------------------------------------------------- pinner registry (VPS content_store)
# The "be your own Pinata" availability layer: a node that pins a checkpoint ANNOUNCES itself to the VPS
# content_store under the friendly name `pinner-<peer_id>` (a tiny JSON record, same PUT shape as
# diloco_contributor.publish_delta). A cold fetcher reads /manifest, pulls each `pinner-*` record, and
# keeps only the LIVE ones for the CID it wants (records self-expire CLIENT-SIDE by their `ts`, so the VPS
# stays a dumb store — no server change, per the design's §7). The VPS carries only these ~100-byte records,
# never checkpoint bytes.
def announce_pin(registry_url, cid, peer_id, token=None, ts=None, timeout=None, retries=None):
    """PUT `{cid, peer_id, ts}` to the content_store under name `pinner-<peer_id>` so other nodes learn
    this peer is seeding `cid`. Reuses the exact PUT pattern of diloco_contributor.publish_delta
    (sha256-of-body path, X-Auth token, X-Name friendly name). `token` defaults to NEURAHASH_CONTENT_TOKEN.
    No-op (returns None) if `registry_url` is empty. Returns the record dict on success. `timeout`
    defaults to NEURAHASH_PUT_TIMEOUT (180s); the PUT is retried per NEURAHASH_PUT_RETRIES / `retries` on
    a timeout, connection error, or 5xx (see _put_retry) -- never on a 4xx."""
    registry_url = (registry_url or "").rstrip("/")
    if not registry_url:
        return None
    token = token if token is not None else os.environ.get("NEURAHASH_CONTENT_TOKEN", "")
    rec = {"cid": cid, "peer_id": peer_id, "ts": int(ts if ts is not None else time.time())}
    body = json.dumps(rec).encode()
    h = hashlib.sha256(body).hexdigest()
    req = urllib.request.Request(f"{registry_url}/o/{h}", data=body, method="PUT",
                                 headers={"X-Auth": token, "X-Name": f"pinner-{peer_id}"})
    t = PUT_TIMEOUT if timeout is None else timeout
    _put_retry(lambda: urllib.request.urlopen(req, timeout=t).read(),
               label=f"{registry_url}/o/{h} (pinner-{peer_id})", retries=retries)
    return rec


def push_named_to_store(store_url, name, data, token=None, timeout=None, retries=None):
    """PUT `data` (bytes or str) to a content store under friendly `name` so a fresh remote node can read
    it BY NAME (via _store_get_named / read_tracker_from_store). Same PUT shape as announce_pin
    (sha256-of-body path, X-Auth token, X-Name). The coordinator uses this to publish the tracker doc to
    the store on every checkpoint publish, so cold-start discovery doesn't depend on any external push
    script. `token` defaults to NEURAHASH_CONTENT_TOKEN. Returns the sha256, or None if store_url empty.
    `timeout` defaults to NEURAHASH_PUT_TIMEOUT (180s); the PUT is retried per NEURAHASH_PUT_RETRIES /
    `retries` on a timeout, connection error, or 5xx (see _put_retry) -- never on a 4xx."""
    store_url = (store_url or "").rstrip("/")
    if not store_url:
        return None
    if isinstance(data, str):
        data = data.encode()
    token = token if token is not None else os.environ.get("NEURAHASH_CONTENT_TOKEN", "")
    h = hashlib.sha256(data).hexdigest()
    req = urllib.request.Request(f"{store_url}/o/{h}", data=data, method="PUT",
                                 headers={"X-Auth": token, "X-Name": name})
    t = PUT_TIMEOUT if timeout is None else timeout
    _put_retry(lambda: urllib.request.urlopen(req, timeout=t).read(),
               label=f"{store_url}/o/{h} ({name})", retries=retries)
    return h


def known_pinners(registry_url, cid, max_age_s=3600, timeout=30):
    """Return the list of LIVE pinner records ({cid, peer_id, ts}) for `cid` from the content_store.
    Reads /manifest, fetches each `pinner-*` record by its sha256, keeps those whose `cid` matches and
    whose `ts` is within `max_age_s` of now (stale/expired seeders dropped client-side). Best-effort: a
    registry that is down or a record that won't parse yields [] / is skipped, never raises."""
    registry_url = (registry_url or "").rstrip("/")
    if not registry_url:
        return []
    try:
        with urllib.request.urlopen(f"{registry_url}/manifest", timeout=timeout) as r:
            manifest = json.loads(r.read().decode())
    except Exception:                                          # noqa: BLE001 — registry down -> no known pinners
        return []
    now = time.time()
    live = []
    for name, meta in manifest.items():
        if not name.startswith("pinner-"):
            continue
        sha = (meta or {}).get("sha256") if isinstance(meta, dict) else None
        if not sha:
            continue
        try:
            with urllib.request.urlopen(f"{registry_url}/o/{sha}", timeout=timeout) as r:
                rec = json.loads(r.read().decode())
        except Exception:                                      # noqa: BLE001 — skip an unreadable record
            continue
        if rec.get("cid") != cid:
            continue
        if max_age_s is not None and (now - float(rec.get("ts", 0))) > max_age_s:
            continue                                           # expired: this seeder hasn't re-announced
        live.append(rec)
    return live


def _ipfs_peer_id(ipfs_bin=None, timeout=15):
    """Return this node's IPFS peer id via `ipfs id -f '<id>'`, or '' if no daemon/binary. Best-effort."""
    ipfs_bin = ipfs_bin or IPFS_BIN
    try:
        out = subprocess.check_output([ipfs_bin, "id", "-f", "<id>"],
                                      stderr=subprocess.DEVNULL, timeout=timeout).decode().strip()
        return out.splitlines()[-1] if out else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# NEURAHASH_SEED=on turns a fetch-only miner into an opt-in SEEDER: after it has the checkpoint file, it
# `ipfs pin add`s the CID (so it keeps serving it) and announces itself to the registry. Default OFF and a
# no-op if there is no local ipfs daemon — so a plain fetch-only miner is completely unaffected.
def _seed_enabled():
    return (os.environ.get("NEURAHASH_SEED", "") or "").strip().lower() in ("1", "true", "yes", "on")


def seed_checkpoint(cid, registry_url=None, token=None, ipfs_bin=None, ckpt_path=None):
    """Opt-in miner seeding: iff NEURAHASH_SEED=on AND a local ipfs daemon exists, pin `cid` (adding
    `ckpt_path` to the local repo first if given, so a gateway-fetched file becomes locally served) and
    announce_pin to the registry. Returns the peer_id it announced under, or '' if it did nothing
    (seeding disabled, no daemon, or no peer id). Best-effort — never raises into a miner's loop.
    `registry_url`/`token` default to NEURAHASH_CONTENT_URL / NEURAHASH_CONTENT_TOKEN."""
    if not _seed_enabled():
        return ""
    ipfs_bin = ipfs_bin or IPFS_BIN
    peer_id = _ipfs_peer_id(ipfs_bin=ipfs_bin)
    if not peer_id:                                            # no local daemon -> silently stay fetch-only
        return ""
    try:
        if ckpt_path and os.path.exists(ckpt_path):
            # re-add the fetched bytes so THIS node hosts the CID (idempotent; add is content-addressed).
            publish(ckpt_path, ipfs_bin=ipfs_bin, announce=True, pin=True)
        else:
            subprocess.run([ipfs_bin, "pin", "add", cid],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
            _record_local_pin(cid)
    except (OSError, subprocess.SubprocessError):
        pass
    registry_url = registry_url if registry_url is not None else os.environ.get("NEURAHASH_CONTENT_URL", "")
    try:
        announce_pin(registry_url, cid, peer_id, token=token)
    except Exception:                                          # noqa: BLE001 — a registry hiccup must not matter
        pass
    return peer_id


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Publish/fetch a checkpoint over IPFS (quota-free).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish"); p.add_argument("path"); p.add_argument("--tracker")
    p.add_argument("--round", type=int, default=0)
    g = sub.add_parser("fetch"); g.add_argument("cid"); g.add_argument("dest")
    s = sub.add_parser("seed")      # opt-in miner seeding: pin a CID + announce (NEURAHASH_SEED=on)
    s.add_argument("cid"); s.add_argument("--path", default=None); s.add_argument("--registry", default=None)
    sub.add_parser("pinata-test")   # verify a configured Pinata JWT end-to-end (pin -> fetch -> unpin)
    sub.add_parser("service-test")  # verify a configured free pin service (Filebase etc.): auth + pin count
    sub.add_parser("filebase-test") # verify Filebase S3 creds + bucket end-to-end (upload -> pin -> CID)
    a = ap.parse_args()
    if a.cmd == "publish":
        t0 = time.time()
        cid = publish(a.path)
        print(f"CID {cid}  ({os.path.getsize(a.path)} bytes, add {time.time()-t0:.1f}s)")
        if a.tracker:
            write_tracker(a.tracker, a.round, cid)
            print(f"tracker -> {a.tracker}")
    elif a.cmd == "fetch":
        t0 = time.time()
        gw = fetch(a.cid, a.dest)
        print(f"fetched via {gw} in {time.time()-t0:.1f}s -> {a.dest}")
    elif a.cmd == "seed":
        if not _seed_enabled():
            raise SystemExit("seeding disabled — set NEURAHASH_SEED=on (and run a local ipfs daemon)")
        pid = seed_checkpoint(a.cid, registry_url=a.registry, ckpt_path=a.path)
        print(f"seeding {a.cid} as peer {pid}" if pid else "did nothing (no local ipfs daemon / no peer id)")
    elif a.cmd == "pinata-test":
        jwt = _pinata_jwt()
        if not jwt:
            raise SystemExit("no Pinata JWT — set PINATA_JWT or PINATA_JWT_FILE first")
        print(f"JWT found ({len(jwt)} chars). Pin -> fetch -> unpin round-trip:")
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pinata_selftest.bin")
        with open(tmp, "wb") as f:
            f.write(os.urandom(1_000_000))               # 1 MB unique blob
        want = hashlib.sha256(open(tmp, "rb").read()).hexdigest()[:16]
        try:
            t0 = time.time(); cid = pin_file_to_pinata(tmp, jwt=jwt, name="neurahash-ckpt-selftest")
            print(f"  [1/3] pinned to Pinata: {cid}  ({time.time()-t0:.1f}s)")
            time.sleep(3)
            back = tmp + ".back"
            t0 = time.time(); gw = fetch(cid, back, verify_cid=False, timeout=60)
            got = hashlib.sha256(open(back, "rb").read()).hexdigest()[:16]
            ok = "MATCH" if got == want else "MISMATCH"
            print(f"  [2/3] fetched back via {gw.split('//')[1].split('/')[0]}  ({time.time()-t0:.1f}s) — sha256 {ok}")
            unpin_pinata(cid, jwt)
            print(f"  [3/3] unpinned {cid}")
            os.remove(back)
            print("PINATA OK — ready to enable NEURAHASH_IPFS_PUBLISH=on" if ok == "MATCH"
                  else "PINATA reachable but integrity check failed — investigate")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    elif a.cmd == "service-test":
        url, token = _pin_service_cfg()
        if not token:
            raise SystemExit("no pin service token — set NEURAHASH_PIN_SERVICE_TOKEN or "
                             "NEURAHASH_PIN_SERVICE_TOKEN_FILE (default provider = Filebase)")
        print(f"pin service: {url}  (token {len(token)} chars). Verifying auth via GET /pins ...")
        res = _pin_service_api("GET", "/pins?limit=1", token)
        n = res.get("count", len(res.get("results") or res.get("rows") or []))
        print(f"  OK — authenticated; {n} pin(s) currently on the service.")
        print("PIN SERVICE OK — leave the token configured and publish_checkpoint mirrors each CID as a "
              "backstop (Filebase free tier = 5 GB; prune keeps newest 3).")
    elif a.cmd == "filebase-test":
        creds = read_s3_creds()
        if not creds:
            raise SystemExit("no Filebase S3 creds — put them in firebase/accesskey.txt (endpoint / "
                             "accessKeyId / secretAccessKey / region) or set NEURAHASH_FILEBASE_KEYFILE")
        bucket = os.environ.get("NEURAHASH_FILEBASE_BUCKET", "").strip()
        if not bucket:
            raise SystemExit("set NEURAHASH_FILEBASE_BUCKET to your Filebase IPFS bucket name "
                             "(create one bucket in the Filebase console first)")
        print(f"Filebase S3: {creds['endpoint']} region={creds['region']} bucket={bucket} "
              f"key={creds['access_key'][:6]}...  — PUT + pin a 64 KB probe:")
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_filebase_selftest.bin")
        with open(tmp, "wb") as f:
            f.write(os.urandom(65536))
        try:
            t0 = time.time(); cid = pin_file_via_s3(tmp, bucket, key="neurahash-filebase-selftest")
            print(f"  uploaded + pinned ({time.time()-t0:.1f}s) — CID {cid or '(pending)'}")
            print("FILEBASE OK — publish_checkpoint(backstop='auto') will now mirror each checkpoint to Filebase "
                  "(set NEURAHASH_FILEBASE_BUCKET + NEURAHASH_IPFS_PUBLISH=on on the coordinator)."
                  if cid else "FILEBASE upload OK — CID not reported yet (usually appears within seconds).")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
