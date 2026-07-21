#!/usr/bin/env python3
"""tools/self_update.py -- SIGNED, FAIL-CLOSED auto-update + SIGNED NETWORK MANIFEST for the miner.

WHY THIS EXISTS
    The public client is `git clone` + `pip install -r requirements.txt` with no version file
    and no self-update, so every code change means a manual re-clone / a forum ping. This module
    lets the operator push ONE signed release and have every running miner self-update on its next
    check -- WITHOUT ever running code that is not cryptographically signed by the project's
    release key. Auto-running pushed code on strangers' machines means a repo/mirror compromise
    could push malware to the whole fleet, so the ONE hard rule is:

        a miner NEVER checks out / runs code unless a manifest signed by the PINNED release key
        says to, and it never DOWNGRADES. On ANY doubt it stays on the code it already has.

    SECOND PURPOSE, ADDED 2026-07-21 (docs/MINER_MANIFEST_DESIGN.md): the same signed artifact now
    also carries the NETWORK'S EXPECTATIONS -- `config` (endpoints/protocol) and
    `min_client_version`. A joiner (issue #71) ran a client whose CODE needed three env vars while
    the DOCS named two, got an opaque HTTP 401, and had no channel to receive the fix once it was
    written. Code, config and expectations arrived through different channels and silently
    disagreed. One signed object collapses them, and makes the disagreement LOUD at startup instead
    of silent at first publish.

HOW IT WORKS (all steps fail CLOSED -- any failure => stay on the current version, keep mining)
    1. fetch a signed manifest (release.json) from HARD-CODED mirrors (never a url from the
       manifest, never a shell command from the manifest);
    2. VERIFY every fetched manifest's secp256k1 signature against a PINNED release public key
       baked into this file, using the repo's own signing lib (neura_l1.signing) -- no hand-rolled
       crypto. A manifest that fails is ignored SILENTLY: a host can WITHHOLD a manifest, it can
       never FORGE one, so a forged mirror cannot stop a good mirror from winning;
    3. among the VERIFIED ones take the HIGHEST version (`is_forward`); zero verified -> warn and
       keep the current code;
    4. only if manifest.version > local VERSION (a strict FORWARD move) do we `git fetch` +
       `git checkout <manifest.git_commit>` (the commit is validated to be a bare hex id -- no
       arbitrary refs);
    5. VERIFY HEAD == manifest.git_commit after checkout;
    6. `pip install -r requirements.txt` ONLY if requirements.txt changed;
    7. re-exec the miner on the new code;
    8. apply `config` as DEFAULTS ONLY from a strict ALLOWLIST (explicit env always wins), and
       compute the `min_client_version` publish block (refuse to PUBLISH, keep TRAINING).

    A bad signature, wrong key, unreachable mirror, malformed manifest, commit mismatch, or a
    downgrade attempt -> a clear WARNING is logged and control returns unchanged. It NEVER raises
    out to the caller and NEVER hard-crashes the miner.

OPT-OUT + RATE LIMIT
    Default ON. `NEURAHASH_AUTOUPDATE=0` (or the `--no-auto-update` flag, which sets it) fully
    disables it. The STARTUP check ALWAYS runs (short per-mirror timeout, fail-open to current
    code) -- a joiner who just restarted to pick up a fix must not be told "checked 4h ago". The
    6h rate limit continues to govern the IN-RUN PERIODIC check via a small JSON dotfile, so a
    crash-loop costs one bounded GET per restart and a run-forever miner still does not hammer
    GitHub.

Everything network/git/pip/re-exec is injectable (fetch_fn/git_fn/pip_fn/reexec_fn) so the whole
policy is unit-tested with NO real network, git, pip, or process replacement -- see
tests/test_self_update.py and tests/test_miner_manifest.py.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from neura_l1.signing import recover_bytes            # secp256k1 ecrecover (real crypto, reused)
from neurahash.canon import _canon                    # deterministic canonical bytes (sorted-key JSON)

# ===========================================================================================
#  PINNED RELEASE PUBLIC KEY
# ===========================================================================================
# REAL RELEASE KEY (pinned 2026-07-19). This is the secp256k1 / EIP-55 address whose signature over
# a release manifest this client will trust. The matching PRIVATE key was generated offline by the
# operator and is held off-machine (see SIGNING.md) -- it never lives in this repo. A repo/mirror
# compromise cannot forge an update because trust is anchored in this pinned constant, not the
# transport. To rotate: ship a new-pinned-key update signed with the CURRENT key (SIGNING.md).
# (The prior TEST key 0x19E7E376...aff2A derived from 0x11..11 remains ONLY in tests/.)
PINNED_RELEASE_PUBKEY = "0x5168F6cc4cc05bfd6d4714906d68e083c02dDC66"  # real offline release address
# ===========================================================================================

# The manifest is fetched from exactly these HARD-CODED mirrors. A url is NEVER read from the
# manifest itself (a compromised manifest cannot redirect the next fetch) and `config` cannot add
# one. Because the manifest is SIGNED, host choice is an AVAILABILITY decision, not a trust
# decision -- so we use several (docs/MINER_MANIFEST_DESIGN.md sec.1) and take the highest VERIFIED
# version across all of them. Deliberately NOT included: the coordinator (B8 exists to retire it;
# bootstrapping every miner off it re-centralises exactly what B8 removes) and IPNS (slow and
# unreliable for a mutable pointer -- IPFS stays where it is already good, the artifacts).
MANIFEST_URL = "https://raw.githubusercontent.com/whitegreenstudios/neurahash-miner/main/release.json"
HF_MANIFEST_URL = ("https://huggingface.co/datasets/whitegreenstudios888/neurahash-data/"
                   "resolve/main/release.json")
# The VPS content store. Plain HTTP by design (that box terminates no TLS) -- acceptable ONLY
# because the payload is signature-verified and this url is a compiled-in constant. Today the store
# serves /health, /manifest and /o/<sha256>; until it also serves /release.json this mirror simply
# 404s, which is a NON-EVENT (an unreachable mirror never blocks or crashes the miner).
VPS_MANIFEST_URL = "http://47.84.93.96:8710/release.json"

MIRRORS = (
    ("github-raw", MANIFEST_URL),
    ("huggingface", HF_MANIFEST_URL),
    ("vps-store", VPS_MANIFEST_URL),
)
# http:// is refused for anything not on this compiled list, so no config, manifest or environment
# value can downgrade the GitHub / HuggingFace fetch to cleartext.
_ALLOWED_HTTP_URLS = frozenset({VPS_MANIFEST_URL})

# Domain tag mixed into the signed bytes so a release-manifest signature can never be confused
# with any other signed object this project produces.
RELEASE_KIND = "neurahash-miner-release"

AUTOUPDATE_ENV = "NEURAHASH_AUTOUPDATE"          # "0"/"false"/"no"/"off" => disabled
STATE_ENV = "NEURAHASH_AUTOUPDATE_STATE"         # override the rate-limit dotfile path (tests/ops)
DEFAULT_RATE_LIMIT_S = 6 * 3600                  # at most one PERIODIC check per 6h
STARTUP_TIMEOUT_S = 6                            # per-mirror; bounds the cost of a restart loop
MAX_MANIFEST_BYTES = 1 << 20                     # a real manifest is ~300 bytes; refuse a flood
_FALSEY = {"0", "false", "no", "off", ""}

VERSION_FILE = "VERSION"
REQUIREMENTS_FILE = "requirements.txt"
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")  # a bare git commit id -- NOT an arbitrary ref


def log(msg):
    """One ASCII line, flushed -- safe on the Windows cp1252 console."""
    print(f"[self_update] {msg}", flush=True)


# ------------------------------------------------------------------ version parsing / ordering
def parse_version(s):
    """Parse a strict numeric dotted version ('MAJOR.MINOR.PATCH', 1-4 components) into a tuple of
    ints for ordering. Raises ValueError on anything non-numeric -- so a malformed/booby-trapped
    version string can never compare as 'newer' (fail closed at the caller)."""
    if not isinstance(s, str):
        raise ValueError(f"version must be a string, got {type(s).__name__}")
    parts = s.strip().split(".")
    if not (1 <= len(parts) <= 4) or not all(p.isdigit() for p in parts):
        raise ValueError(f"not a numeric dotted version: {s!r}")
    return tuple(int(p) for p in parts)


def is_forward(new_v, cur_v):
    """True iff new_v is STRICTLY greater than cur_v (a forward move). Equal or lower -> False,
    so a downgrade or a replay of the current version is never applied."""
    return parse_version(new_v) > parse_version(cur_v)


def read_local_version(repo_dir=REPO):
    """Read + parse the repo-root VERSION file. Raises on a missing/unparseable file."""
    with open(os.path.join(repo_dir, VERSION_FILE), "r", encoding="utf-8") as f:
        raw = f.read().strip()
    parse_version(raw)                 # validate now; raises if malformed
    return raw


# ------------------------------------------------------------------ manifest canon / verify
def canonical_manifest_bytes(manifest):
    """The exact bytes the release signature is computed over and recovered against. Built ONLY
    from the security-relevant fields (kind is a fixed constant; the self-declared 'signer' field,
    if any, is IGNORED). Any tampering with a signed field changes these bytes, so recovery yields
    a different address and verification fails.

    v2 BACKWARD COMPATIBILITY -- THE THING NOT TO BREAK. `min_client_version` and `config` are
    OPTIONAL and are added to the signed body ONLY when the manifest actually carries them. A v1
    manifest (the LIVE one: version / git_commit / published_ts / signature / signer, with neither
    optional field) therefore produces BYTE-IDENTICAL canonical bytes to before this change, so
    every already-signed manifest keeps verifying. Because the optional fields ARE signed when
    present, an attacker can neither ADD a `config` to a v1 manifest nor STRIP one from a v2
    manifest without invalidating the signature. (`_canon` sorts keys recursively, so the nested
    `config` object's bytes are deterministic too.)"""
    body = {
        "kind": RELEASE_KIND,
        "version": str(manifest["version"]),
        "git_commit": str(manifest["git_commit"]),
        "published_ts": int(manifest["published_ts"]),
    }
    # Truthiness, not `is not None`, deliberately: `"config": {}` / `"config": null` /
    # `"min_client_version": ""` all mean "this manifest declares nothing", and all must
    # canonicalise EXACTLY like a v1 manifest that omits the key. Keying on presence instead would
    # create a second, subtly different byte string for the same declared content.
    if manifest.get("min_client_version"):
        body["min_client_version"] = str(manifest["min_client_version"])
    if manifest.get("config"):
        body["config"] = manifest["config"]
    return _canon(body)


def verify_manifest(manifest, pubkey=PINNED_RELEASE_PUBKEY):
    """Verify a release manifest against the PINNED release public key. Returns (ok, reason).

    Rejects (fail closed):
      * a non-dict / missing required field (version, git_commit, published_ts, signature);
      * a missing / empty signature;
      * a git_commit that is not a bare hex commit id;
      * a `config` that is not a JSON object, or a `min_client_version` that is not a strict
        numeric dotted version (an unparseable one would make the publish gate undecidable --
        refusing the whole manifest is the fail-closed direction);
      * a signature that recovers to ANY address other than the pinned key (covers a tampered
        field, a wrong-key signature, and a garbage signature -- all recover to != pinned).
    On success returns (True, recovered_address)."""
    if not isinstance(manifest, dict):
        return False, "manifest is not a JSON object"
    for k in ("version", "git_commit", "published_ts", "signature"):
        if k not in manifest:
            return False, f"missing field: {k}"
    sig = manifest.get("signature")
    if not isinstance(sig, str) or not sig.strip():
        return False, "missing or empty signature"
    if not _COMMIT_RE.match(str(manifest["git_commit"])):
        return False, f"git_commit is not a bare hex commit id: {manifest['git_commit']!r}"
    try:
        # Parsed HERE, not only at the comparison site: an unparseable `version` on ONE mirror
        # would otherwise be accepted as "verified" and then poison the whole mirror loop / the
        # forward gate, killing the fleet's update channel depending on mirror ORDER.
        parse_version(str(manifest["version"]))
    except Exception as e:
        return False, f"version malformed: {e}"
    if manifest.get("config") is not None and not isinstance(manifest["config"], dict):
        return False, f"config is not a JSON object: {type(manifest['config']).__name__}"
    if manifest.get("min_client_version") is not None:
        try:
            parse_version(str(manifest["min_client_version"]))
        except Exception as e:
            return False, f"min_client_version malformed: {e}"
    try:
        data = canonical_manifest_bytes(manifest)
    except Exception as e:
        return False, f"manifest fields malformed: {e}"
    try:
        recovered = recover_bytes(data, sig)
    except Exception as e:
        return False, f"signature recovery failed: {e}"
    if recovered.lower() != str(pubkey).lower():
        return False, f"signature does not match pinned release key (recovered {recovered})"
    return True, recovered


# ------------------------------------------------------------------ fetch (stdlib only)
def _default_fetch(url, timeout=15):
    """Fetch the manifest text from a COMPILED-IN mirror url. HTTPS is required except for the
    hard-coded VPS mirror (the payload is signature-verified and the url can never come from the
    manifest, `config`, or the environment). Any network error propagates to the caller, which
    treats that mirror as a non-event.

    BOUNDED IN BOTH DIMENSIONS. A socket timeout only fires on an IDLE connection, so a hostile or
    broken mirror that dribbles one byte at a time would otherwise stall miner startup for as long
    as it likes (and an endless body would eat memory). So the body is read in chunks against a
    WALL-CLOCK deadline and a hard byte cap: a real manifest is a few hundred bytes."""
    u = str(url)
    low = u.lower()
    if not (low.startswith("https://") or (low.startswith("http://") and u in _ALLOWED_HTTP_URLS)):
        raise ValueError(f"refusing to fetch manifest over non-HTTPS url: {url!r}")
    req = urllib.request.Request(u, headers={"User-Agent": "neurahash-miner-selfupdate"})
    deadline = time.monotonic() + float(timeout)
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 (scheme checked above)
        chunks, total = [], 0
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"manifest fetch from {url!r} exceeded {timeout}s")
            # read1(): return after ONE underlying recv, so a dribbling peer cannot block us
            # inside a single call past the deadline check.
            b = resp.read1(65536)
            if not b:
                break
            total += len(b)
            if total > MAX_MANIFEST_BYTES:
                raise ValueError(f"manifest from {url!r} exceeds {MAX_MANIFEST_BYTES} bytes")
            chunks.append(b)
    return b"".join(chunks).decode("utf-8")


def _accepts_timeout(fn):
    """True if fn takes a `timeout` kwarg. Test fakes are usually `lambda url: ...`, so we must not
    force a kwarg they do not accept."""
    try:
        import inspect
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return "timeout" in sig.parameters


class ManifestFetch:
    """Outcome of one multi-mirror fetch. `manifest` is the highest-version VERIFIED manifest, or
    None. `tried` is [(mirror_name, url, status)] for EVERY mirror, so `--doctor` can name exactly
    which hosts were tried and what each one said."""

    def __init__(self, manifest=None, source=None, tried=None, any_parsed=False):
        self.manifest = manifest
        self.source = source
        self.tried = list(tried or [])
        self.any_parsed = any_parsed

    @property
    def ok(self):
        return self.manifest is not None

    def summary(self):
        return "; ".join(f"{n}: {s}" for n, _u, s in self.tried) or "no mirrors configured"

    def __repr__(self):
        return f"ManifestFetch(ok={self.ok}, source={self.source!r}, tried={self.tried!r})"


def fetch_best_manifest(mirrors=MIRRORS, pubkey=PINNED_RELEASE_PUBKEY, fetch_fn=None,
                        timeout=STARTUP_TIMEOUT_S, min_published_ts=0):
    """Fetch EVERY mirror, keep only manifests that VERIFY against the pinned key, and return the
    highest-version one as a ManifestFetch. Mirrors that are unreachable, serve garbage, or serve a
    FORGED manifest are ignored silently -- a valid mirror still wins. Never raises; on total
    failure the caller keeps the code and config it already has (fail-OPEN on AVAILABILITY,
    fail-CLOSED on CRYPTO).

    REPLAY FLOOR. `min_published_ts` rejects a manifest older than the newest one this client has
    already seen. Without it, an attacker who can answer one mirror while withholding the others
    can serve a GENUINE OLD signed manifest to roll `config` (and the min_client_version gate) back
    to superseded values -- the forward-only version gate protects the CODE, not the config."""
    fetch_fn = fetch_fn or _default_fetch
    tried, best, best_src, any_parsed = [], None, None, False
    for name, url in mirrors:
        try:
            text = fetch_fn(url, timeout=timeout) if _accepts_timeout(fetch_fn) else fetch_fn(url)
            manifest = json.loads(text)
            any_parsed = True
        except Exception as e:
            tried.append((name, url, f"unreachable ({type(e).__name__})"))
            continue
        ok, info = verify_manifest(manifest, pubkey)
        if not ok:
            tried.append((name, url, f"REJECTED ({info})"))
            continue
        try:
            published = int(manifest.get("published_ts") or 0)
        except Exception:
            published = 0
        if min_published_ts and published < int(min_published_ts):
            tried.append((name, url, f"REJECTED (replay: published_ts {published} < floor "
                                     f"{int(min_published_ts)})"))
            continue
        v = str(manifest.get("version"))
        try:
            better = best is None or is_forward(v, str(best["version"]))
        except Exception as e:
            tried.append((name, url, f"REJECTED (version unusable: {e})"))
            continue
        tried.append((name, url, f"valid v{v}"))
        if better:
            best, best_src = manifest, url
    return ManifestFetch(best, best_src, tried, any_parsed)


# ------------------------------------------------------ manifest `config` -> environment defaults
# No `@`: a url with embedded userinfo (http://user:pass@host/) is a CREDENTIAL, and the network is
# never allowed to hand one to this client. No whitespace of any kind, so a value can never smuggle
# a second header or line into a downstream consumer.
_URL_RE = re.compile(r"^https?://[A-Za-z0-9._~:/?#\[\]!$&'()*+,;=%-]{1,300}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{8,128}$")
# No `/` and no leading `-`/`.`: keeps this a bare version TAG, so it can never read as a
# filesystem path ("../../etc/passwd", "/etc/passwd") or as a command-line flag ("--publish-delta")
# to any future consumer that interpolates it.
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _v_url(v):
    return v.strip() if isinstance(v, str) and _URL_RE.match(v.strip()) else None


def _v_hex(v):
    return v.strip().lower() if isinstance(v, str) and _HEX_RE.match(v.strip()) else None


def _v_tag(v):
    return v.strip() if isinstance(v, str) and _TAG_RE.match(v.strip()) else None


def _v_bool(v):
    return ("1" if v else "0") if isinstance(v, bool) else None


# STRICT ALLOWLIST (docs/MINER_MANIFEST_DESIGN.md sec.2 + sec.5). ONLY these fields are read; every
# other key inside `config` is IGNORED rather than applied. Nothing here can name a filesystem
# path, a credential, or anything executable -- the values are endpoint urls, a hex digest, a
# version tag and a boolean, each SHAPE-VALIDATED before it ever reaches the environment.
CONFIG_ALLOWLIST = {
    "merge_url":    ("NEURAHASH_DILOCO_MERGE_URL", _v_url),
    "content_url":  ("NEURAHASH_CONTENT_URL", _v_url),
    # The CORPUS store is a DIFFERENT variable from content_url, and conflating them is an easy and
    # silent mistake: corpus_sync.store_url_from_env reads NEURAHASH_CONTENT_STORE
    # (neurahash/corpus_sync.py:102-106), while NEURAHASH_CONTENT_URL is the base/checkpoint tracker
    # hint. Shipping only content_url therefore looks correct and does NOT redirect the corpus fetch.
    # This entry is what lets a signed release point joiners at the HF-hosted corpus; it wins over
    # the coordinator's hello-advertised store, because the miner resolves
    # `store_url_from_env() or _store_url_from_hello(hello)` (run_miner_client.py:201).
    "corpus_store": ("NEURAHASH_CONTENT_STORE", _v_url),
    "corpus_sha":   ("NEURAHASH_CORPUS_SHA", _v_hex),
}
PROTOCOL_ALLOWLIST = {
    "signed_put":          ("NEURAHASH_SIGNED_PUT", _v_bool),
    "contrib_sig_version": ("NEURAHASH_CONTRIB_SIG_VERSION", _v_tag),
}


def apply_manifest_config(config, environ=None):
    """Apply a VERIFIED manifest's `config` as DEFAULTS ONLY. Returns (applied, ignored):
    `applied` is ["NAME=value", ...] for what THIS call actually set (so the banner can show it
    rather than having it happen invisibly), `ignored` is ["key (why)", ...].

    PRECEDENCE -- the same rule as run_miner.apply_zero_config_defaults: an explicitly set,
    non-empty environment variable ALWAYS wins, so an operator or a pod pinning its own values is
    never overridden by the network. Only allowlisted keys carrying a valid shape are applied;
    anything else (unknown key, wrong type, null, a path, a credential) is IGNORED, never applied.
    Never raises."""
    env = os.environ if environ is None else environ
    applied, ignored = [], []
    if not isinstance(config, dict):
        return applied, ["<config> (not a JSON object)"]

    def _one(name, validator, raw, label):
        val = validator(raw)
        if val is None:
            ignored.append(f"{label} (value rejected by the allowlist validator)")
            return
        if env.get(name, "").strip():
            ignored.append(f"{label} (explicit {name} in the environment wins)")
            return
        env[name] = val
        applied.append(f"{name}={val}")

    for key, raw in config.items():
        if key == "protocol":
            if not isinstance(raw, dict):
                ignored.append("protocol (not a JSON object)")
                continue
            for pkey, praw in raw.items():
                if pkey not in PROTOCOL_ALLOWLIST:
                    ignored.append(f"protocol.{pkey} (not on the allowlist)")
                    continue
                name, validator = PROTOCOL_ALLOWLIST[pkey]
                _one(name, validator, praw, f"protocol.{pkey}")
            continue
        if key not in CONFIG_ALLOWLIST:
            ignored.append(f"{key} (not on the allowlist)")
            continue
        name, validator = CONFIG_ALLOWLIST[key]
        _one(name, validator, raw, key)
    return applied, ignored


# ------------------------------------------------------------------ min_client_version publish gate
def publish_block_reason(manifest, local_version):
    """Return a human reason why this client must NOT publish, or None if it may.

    A client older than the network's `min_client_version` still TRAINS (a stranger must never get
    a crash for being out of date) but must not submit anything the network would reject. The
    reason is returned BY NAME so it shows up in the LIVE/LOCAL banner exactly like the other
    publish-mode reasons."""
    if not isinstance(manifest, dict):
        return None
    mcv = manifest.get("min_client_version")
    if mcv is None:
        return None
    try:
        need = parse_version(str(mcv))
    except Exception:
        return None                       # verify_manifest already rejects these; belt-and-braces
    if not local_version:
        return (f"client version unknown (no readable {VERSION_FILE}) but the signed network "
                f"manifest requires min_client_version {mcv} -- publishing refused; training "
                f"continues (run again to auto-update, or `git pull`)")
    try:
        have = parse_version(str(local_version))
    except Exception:
        return (f"local version {local_version!r} is unparseable but the signed network manifest "
                f"requires min_client_version {mcv} -- publishing refused; training continues "
                f"(run again to auto-update, or `git pull`)")
    if have < need:
        return (f"client v{local_version} is below the signed network manifest's "
                f"min_client_version {mcv} -- publishing refused so nothing is submitted that the "
                f"network would reject; training continues (run again to auto-update, or "
                f"`git pull`)")
    return None


# ------------------------------------------------------------------ git / pip / re-exec (default impls)
def _default_git(repo_dir, *args, timeout=180):
    """Run `git -C <repo> <args...>` with NO shell (list args) and utf-8 decoding. Returns
    (returncode, combined_output). Never runs anything from the manifest."""
    cmd = ["git", "-C", repo_dir, *args]
    p = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _default_pip(repo_dir, timeout=1800):
    """`<python> -m pip install -r requirements.txt` (the CURRENT interpreter). No shell."""
    req = os.path.join(repo_dir, REQUIREMENTS_FILE)
    cmd = [sys.executable, "-m", "pip", "install", "-r", req]
    p = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _default_reexec(argv):
    """Replace the current process with a fresh run of the miner on the NOW-checked-out code.
    Never returns."""
    os.execv(sys.executable, [sys.executable, *argv])


# ------------------------------------------------------------------ rate-limit state (dotfile)
def _state_path(repo_dir, override=None):
    if override:
        return override
    env = os.environ.get(STATE_ENV, "").strip()
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".neurahash_autoupdate.json")


def _load_state(path):
    """The whole state dotfile: {last_check, manifest_floor_ts, min_client_version}. Missing or
    corrupt -> {} (the client degrades to pre-state behaviour, never crashes)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_state(path, **updates):
    """READ-MODIFY-WRITE so writing one key never drops the others (the rate-limit stamp and the
    replay floor live in the same file and are written at different moments)."""
    d = _load_state(path)
    d.update(updates)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception as e:
        log(f"WARN: could not persist updater state to {path}: {e}")


def _load_last_check(path):
    try:
        return float(_load_state(path).get("last_check", 0.0))
    except Exception:
        return 0.0


def _save_last_check(path, ts):
    _save_state(path, last_check=float(ts))


# ------------------------------------------------------------------ result object
class UpdateResult:
    """Outcome of a check. `applied` is True only when a verified forward update was fully applied
    and a re-exec was requested (with the real reexec_fn the process is already gone; a test's fake
    reexec_fn lets this return). `action` is a short machine tag; `reason` is human detail.
    `manifest` is the VERIFIED manifest when one was obtained (None otherwise); `fetch` is the
    ManifestFetch describing every mirror that was tried."""

    def __init__(self, applied, action, reason="", local_version=None, target_version=None,
                 checked_out=None, pip_ran=False, manifest=None, fetch=None):
        self.applied = applied
        self.action = action
        self.reason = reason
        self.local_version = local_version
        self.target_version = target_version
        self.checked_out = checked_out
        self.pip_ran = pip_ran
        self.manifest = manifest
        self.fetch = fetch

    def __repr__(self):
        return (f"UpdateResult(applied={self.applied}, action={self.action!r}, "
                f"local={self.local_version}, target={self.target_version}, "
                f"checked_out={self.checked_out}, pip_ran={self.pip_ran}, reason={self.reason!r})")


def _env_enabled():
    """Auto-update is ON unless NEURAHASH_AUTOUPDATE is explicitly falsey."""
    return os.environ.get(AUTOUPDATE_ENV, "1").strip().lower() not in _FALSEY


def _sha256_file(path):
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


# ------------------------------------------------------------------ the orchestrator
def check_and_update(repo_dir=REPO, argv=None, *, manifest_url=None, mirrors=None,
                     pubkey=PINNED_RELEASE_PUBKEY, enabled=None, state_path=None,
                     rate_limit_s=DEFAULT_RATE_LIMIT_S, now=None, honor_rate_limit=True,
                     timeout=STARTUP_TIMEOUT_S, manifest=None,
                     fetch_fn=None, git_fn=None, pip_fn=None, reexec_fn=None):
    """Do at most one signed-update check and, if a VERIFIED forward release exists, apply it and
    re-exec. Returns an UpdateResult. FAIL CLOSED: any error is caught, logged as a warning, and
    the working tree is left untouched (control returns to the caller so the miner keeps running).

    `mirrors` (or the legacy single `manifest_url`) selects where to look; the default is the
    compiled MIRRORS list, and the BEST VERIFIED version across all of them wins. `manifest` lets a
    caller pass an ALREADY-VERIFIED manifest so a startup sync does not fetch twice.

    Injectables (real defaults if None): fetch_fn(url)->text, git_fn(repo,*args)->(rc,out),
    pip_fn(repo)->(rc,out), reexec_fn(argv)->NoReturn. Tests pass fakes so nothing real happens.
    """
    argv = list(argv if argv is not None else sys.argv)
    git_fn = git_fn or _default_git
    pip_fn = pip_fn or _default_pip
    reexec_fn = reexec_fn or _default_reexec
    now = time.time() if now is None else now
    if enabled is None:
        enabled = _env_enabled()
    if mirrors is None:
        mirrors = ((("manifest-url", manifest_url),) if manifest_url else MIRRORS)

    try:
        local_version = read_local_version(repo_dir)
    except Exception as e:
        log(f"WARN: cannot read local {VERSION_FILE} ({e}); skipping auto-update, staying put")
        return UpdateResult(False, "no-version-file", reason=str(e), manifest=manifest)

    if not enabled:
        return UpdateResult(False, "disabled", reason=f"{AUTOUPDATE_ENV} is off",
                            local_version=local_version, manifest=manifest)

    spath = _state_path(repo_dir, state_path)
    if honor_rate_limit:
        last = _load_last_check(spath)
        if now - last < rate_limit_s:
            return UpdateResult(False, "rate-limited",
                                reason=f"checked {int(now - last)}s ago (< {rate_limit_s}s)",
                                local_version=local_version, manifest=manifest)
    # record the attempt up-front so a crash/re-exec loop is throttled by the same rate limit.
    _save_last_check(spath, now)

    # 1+2+3) fetch EVERY mirror, keep only signature-VERIFIED manifests, take the highest version -
    fetch = None
    if manifest is not None:
        # A caller-supplied manifest is re-verified here, ALWAYS. sync_from_manifest only ever
        # passes one that already verified, but this function is public: nothing may reach the
        # `git checkout` below on any path that has not recovered the pinned key.
        ok, info = verify_manifest(manifest, pubkey)
        if not ok:
            log(f"WARN: supplied release manifest REJECTED ({info}); staying on v{local_version} "
                f"(never running unverified code)")
            return UpdateResult(False, "verify-failed", reason=info, local_version=local_version)
    if manifest is None:
        fetch = fetch_best_manifest(mirrors, pubkey, fetch_fn=fetch_fn, timeout=timeout)
        manifest = fetch.manifest
        if manifest is None:
            if not fetch.any_parsed:
                log(f"WARN: no release manifest reachable on any mirror ({fetch.summary()}); "
                    f"staying on v{local_version}")
                return UpdateResult(False, "fetch-failed", reason=fetch.summary(),
                                    local_version=local_version, fetch=fetch)
            log(f"WARN: release manifest REJECTED on every mirror ({fetch.summary()}); staying on "
                f"v{local_version} (never running unverified code)")
            return UpdateResult(False, "verify-failed", reason=fetch.summary(),
                                local_version=local_version, fetch=fetch)

    target_version = str(manifest["version"])
    commit = str(manifest["git_commit"])

    # 4) forward-only gate (no downgrade, no re-apply of the same version) --------------------
    try:
        forward = is_forward(target_version, local_version)
    except Exception as e:
        log(f"WARN: cannot compare versions ({e}); staying on v{local_version}")
        return UpdateResult(False, "version-parse-failed", reason=str(e),
                            local_version=local_version, target_version=target_version,
                            manifest=manifest, fetch=fetch)
    if not forward:
        return UpdateResult(False, "no-op-not-forward",
                            reason=f"manifest v{target_version} <= local v{local_version}",
                            local_version=local_version, target_version=target_version,
                            manifest=manifest, fetch=fetch)

    log(f"verified signed release v{target_version} (commit {commit[:12]}) > local v{local_version}; "
        f"applying update")

    # 5) apply: git fetch + checkout <pinned commit> (list-arg git, hex-validated commit) ------
    req_before = _sha256_file(os.path.join(repo_dir, REQUIREMENTS_FILE))
    try:
        rc, out = git_fn(repo_dir, "fetch", "--quiet", "origin")
        if rc != 0:
            log(f"WARN: `git fetch` failed (rc={rc}); staying on v{local_version}. {out.strip()[-200:]}")
            return UpdateResult(False, "git-fetch-failed", reason=out.strip()[-200:],
                                local_version=local_version, target_version=target_version,
                                manifest=manifest, fetch=fetch)
        rc, out = git_fn(repo_dir, "checkout", "--quiet", commit)
        if rc != 0:
            log(f"WARN: `git checkout {commit[:12]}` failed (rc={rc}); staying on v{local_version}. "
                f"{out.strip()[-200:]}")
            return UpdateResult(False, "git-checkout-failed", reason=out.strip()[-200:],
                                local_version=local_version, target_version=target_version,
                                manifest=manifest, fetch=fetch)
    except Exception as e:
        log(f"WARN: git error during update ({e}); staying on v{local_version}")
        return UpdateResult(False, "git-error", reason=str(e),
                            local_version=local_version, target_version=target_version,
                            manifest=manifest, fetch=fetch)

    # 6) VERIFY the tree is now exactly the signed commit -------------------------------------
    try:
        rc, head = git_fn(repo_dir, "rev-parse", "HEAD")
        head = head.strip()
    except Exception as e:
        rc, head = 1, ""
        log(f"WARN: could not read HEAD after checkout ({e})")
    if rc != 0 or head.lower() != commit.lower():
        log(f"WARN: post-checkout HEAD {head!r} != signed commit {commit!r}; "
            f"NOT re-exec'ing. Attempting to restore v{local_version}.")
        # best-effort restore so we do not strand the miner on a half-applied tree
        try:
            git_fn(repo_dir, "checkout", "--quiet", "-")
        except Exception:
            pass
        return UpdateResult(False, "head-mismatch",
                            reason=f"HEAD {head} != {commit}",
                            local_version=local_version, target_version=target_version,
                            manifest=manifest, fetch=fetch)

    # 7) pip install ONLY if requirements.txt actually changed --------------------------------
    pip_ran = False
    req_after = _sha256_file(os.path.join(repo_dir, REQUIREMENTS_FILE))
    if req_after and req_after != req_before:
        log("requirements.txt changed -- running pip install -r requirements.txt")
        try:
            prc, pout = pip_fn(repo_dir)
            pip_ran = True
            if prc != 0:
                log(f"WARN: pip install returned rc={prc}; continuing to re-exec the signed code "
                    f"anyway (deps may already be satisfied). {pout.strip()[-200:]}")
        except Exception as e:
            log(f"WARN: pip install error ({e}); continuing to re-exec the signed code anyway")

    # 8) re-exec onto the new code ------------------------------------------------------------
    log(f"update to v{target_version} applied; re-exec'ing miner on the new code")
    result = UpdateResult(True, "applied", reason=f"v{local_version} -> v{target_version}",
                          local_version=local_version, target_version=target_version,
                          checked_out=commit, pip_ran=pip_ran, manifest=manifest, fetch=fetch)
    reexec_fn(argv)          # real impl never returns; a test fake returns and we fall through
    return result


class SyncResult:
    """What one startup sync produced, for the launcher's banner, publish gate and --doctor."""

    def __init__(self, update=None, fetch=None, manifest=None, config_applied=None,
                 config_ignored=None, publish_block=None, local_version=None):
        self.update = update
        self.fetch = fetch
        self.manifest = manifest
        self.config_applied = list(config_applied or [])
        self.config_ignored = list(config_ignored or [])
        self.publish_block = publish_block
        self.local_version = local_version

    @property
    def manifest_version(self):
        return str(self.manifest.get("version")) if isinstance(self.manifest, dict) else None

    def mirrors_summary(self):
        return self.fetch.summary() if self.fetch is not None else "not checked"

    def __repr__(self):
        return (f"SyncResult(manifest_version={self.manifest_version}, "
                f"config_applied={self.config_applied}, publish_block={self.publish_block!r})")


def sync_from_manifest(repo_dir=REPO, argv=None, *, startup=True, mirrors=None,
                       pubkey=PINNED_RELEASE_PUBKEY, enabled=None, state_path=None,
                       rate_limit_s=DEFAULT_RATE_LIMIT_S, now=None,
                       timeout=STARTUP_TIMEOUT_S, environ=None,
                       fetch_fn=None, git_fn=None, pip_fn=None, reexec_fn=None):
    """The ONE call a launcher makes. Order is docs/MINER_MANIFEST_DESIGN.md sec.3:

      1. fetch + verify across ALL mirrors (best valid version wins);
      2. if that version is forward -> check out the signed commit, pip if needed, re-exec;
      3. apply `config` as DEFAULTS ONLY (explicit env always wins);
      4. compute the `min_client_version` publish block (refuse to publish, still train).

    `startup=True` bypasses the 6h rate limit -- a joiner who just restarted to pick up a fix must
    not be told "checked 4h ago"; the periodic in-run check keeps the limit. NEVER raises: any
    failure leaves the client exactly as it was, running the code and config it already had."""
    try:
        if enabled is None:
            enabled = _env_enabled()
        try:
            local_version = read_local_version(repo_dir)
        except Exception:
            local_version = None

        if not enabled:
            return SyncResult(local_version=local_version)

        spath = _state_path(repo_dir, state_path)
        st = _load_state(spath)
        try:
            floor = int(st.get("manifest_floor_ts") or 0)
        except Exception:
            floor = 0

        fetch = fetch_best_manifest(mirrors or MIRRORS, pubkey, fetch_fn=fetch_fn, timeout=timeout,
                                    min_published_ts=floor)
        if not fetch.ok:
            log(f"WARN: no VERIFIED network manifest ({fetch.summary()}); keeping the code and "
                f"config this client already has")
            # A withheld manifest must not silently LIFT a publish gate the network already
            # declared: the last known min_client_version keeps applying until a newer SIGNED
            # manifest says otherwise. (Still fail-open on availability -- training is unaffected.)
            remembered = publish_block_reason({"min_client_version": st.get("min_client_version")},
                                              local_version)
            if remembered:
                log("WARN: " + remembered)
            return SyncResult(fetch=fetch, publish_block=remembered, local_version=local_version)

        upd = check_and_update(repo_dir, argv, manifest=fetch.manifest, pubkey=pubkey,
                               enabled=enabled, state_path=state_path, rate_limit_s=rate_limit_s,
                               now=now, honor_rate_limit=not startup, timeout=timeout,
                               git_fn=git_fn, pip_fn=pip_fn, reexec_fn=reexec_fn)
        upd.fetch = fetch

        # raise the replay floor + remember the declared gate, so neither can be rolled back by a
        # genuine-but-superseded manifest served while the good mirrors are withheld.
        try:
            _save_state(spath, manifest_floor_ts=max(floor, int(fetch.manifest.get("published_ts") or 0)),
                        min_client_version=fetch.manifest.get("min_client_version"))
        except Exception as e:
            log(f"WARN: could not persist the manifest replay floor ({e})")

        applied, ignored = apply_manifest_config(fetch.manifest.get("config") or {}, environ)
        block = publish_block_reason(fetch.manifest, local_version)
        if block:
            log("WARN: " + block)
        return SyncResult(update=upd, fetch=fetch, manifest=fetch.manifest, config_applied=applied,
                          config_ignored=ignored, publish_block=block, local_version=local_version)
    except Exception as e:                # a miner must never crash for lack of infra
        log(f"WARN: manifest sync raised ({e}); keeping current code and config")
        return SyncResult()


def maybe_auto_update(argv=None):
    """Convenience entry for a launcher: run a fail-closed check with all real defaults. Swallows
    everything -- a launcher must never crash because of the updater."""
    try:
        return check_and_update(argv=argv)
    except Exception as e:                       # belt-and-suspenders: never escape to the miner
        log(f"WARN: auto-update check raised ({e}); staying on current version")
        return UpdateResult(False, "unexpected-error", reason=str(e))


if __name__ == "__main__":
    # Manual, one-shot check ignoring the rate limit (handy for operators testing a release).
    print(check_and_update(honor_rate_limit=False))
