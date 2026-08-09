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

OBSOLETE-FILE RECLAIM (task #119, added 2026-08-08) -- DEFAULT OFF
    A miner that has updated for months keeps files an old release installed and the current one no
    longer needs. Step 6b reclaims them, but ONLY by a positive ALLOWLIST: paths a PREVIOUS SIGNED
    manifest declared it shipped (`files`) which the CURRENT one no longer declares. It NEVER
    derives a deletion set from the filesystem -- a miner's directory legitimately holds wallet
    keystores, campaign state, logs, checkpoints and the operator's own files that no manifest ever
    mentions, and deleting those loses money with no recovery. With `NEURAHASH_UPDATE_RECLAIM`
    unset it is a DRY RUN that prints what it would remove and removes nothing; `=1` arms it, `=0`
    switches it off entirely. See the block above `reclaim_mode` for the five refusal rules.

Everything network/git/pip/re-exec is injectable (fetch_fn/git_fn/pip_fn/reexec_fn) so the whole
policy is unit-tested with NO real network, git, pip, or process replacement -- see
tests/test_self_update.py, tests/test_miner_manifest.py and tests/test_self_update_reclaim.py.
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
RECLAIM_ENV = "NEURAHASH_UPDATE_RECLAIM"         # obsolete-file reclaim: absent => DRY RUN (see below)
MANIFEST_FILES_KEY = "files"                     # v3 signed shipped-file list {relpath: sha256hex}
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
    # v3 (2026-08-08, task #119): the SHIPPED FILE LIST -- {relative/path: sha256hex}. Same
    # truthiness rule and the same backward-compatibility guarantee: a manifest that omits it (or
    # carries `{}` / `null`) canonicalises BYTE-IDENTICALLY to v1/v2, so every already-signed
    # manifest keeps verifying. It MUST be signed, because it is the input to file DELETION: an
    # unsigned "here is what the old release shipped" list would let anyone name a path to remove.
    if manifest.get(MANIFEST_FILES_KEY):
        body[MANIFEST_FILES_KEY] = manifest[MANIFEST_FILES_KEY]
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
    if manifest.get(MANIFEST_FILES_KEY) is not None and not isinstance(manifest[MANIFEST_FILES_KEY], dict):
        return False, (f"{MANIFEST_FILES_KEY} is not a JSON object: "
                       f"{type(manifest[MANIFEST_FILES_KEY]).__name__}")
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
    Never returns.

    WINDOWS TAKES A DIFFERENT PATH, and it is not cosmetic. The Windows CRT `exec` family builds the
    child's command line by joining the argument vector with SPACES and does not quote members that
    contain them. With the stock python.org all-users install at `C:\\Program Files\\Python311\\
    python.exe`, the child re-parses its own command line as argv[0]='C:\\Program',
    argv[1]='Files\\Python311\\python.exe' -- and Python then treats that argv[1] as the script to
    run, resolving it against the cwd:

        C:\\Program: can't open file 'C:\\Users\\...\\Files\\Python311\\python.exe':
        [Errno 2] No such file or directory

    Reported from a fresh clone on an RTX 3070 taking v3.6.0 -> v3.6.1 (issue #71, 2026-08-02).
    Impact: the update genuinely SUCCEEDS -- checkout and VERSION land on the new version -- and then
    the miner exits. Unattended, a miner silently stops at whichever ~6-hourly check first sees a new
    release, on any Windows box whose interpreter lives in a spaced path. That is the DEFAULT
    location for the all-users installer, so it is not an exotic setup. It stops rather than spins
    (check_and_update writes last_check up front, and the local version is no longer behind), but a
    miner that quietly stops overnight is indistinguishable from one that was never running.

    subprocess does Windows quoting correctly via list2cmdline; os.execv does not. `_default_pip`
    above already goes through subprocess.run with a list and was never affected -- this is the only
    exec path in the file."""
    if os.name == "nt":
        raise SystemExit(subprocess.run([sys.executable, *argv]).returncode)
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


# ===========================================================================================
#  OBSOLETE-FILE RECLAIM (task #119) -- DELETE ONLY BY SHIPPED ALLOWLIST, NEVER BY COMPLEMENT
# ===========================================================================================
# THE PROBLEM. A miner that has self-updated for months carries files that some PAST release
# installed and the CURRENT one no longer needs. They accumulate forever on every miner in the
# world.
#
# THE OBVIOUS IMPLEMENTATION IS CATASTROPHIC. "delete everything in the install dir that is not in
# the current manifest" is a COMPLEMENT rule, and a miner's directory legitimately contains files
# no manifest ever mentions: the WALLET KEYSTORE it is paid into (neurahash/wallet.py defaults to
# `miner_wallet.json`), `.neurahash_keys/` PQC secrets, `keystore/*.json`, campaign state
# (`_state_*`, `_poollive/`), logs, checkpoints, corpus caches, and the operator's own files.
# Deleting those destroys money and identity with NO recovery, and it does it on every machine at
# once, with a valid signature on it. So:
#
#     THE DELETION SET IS AN ALLOWLIST -- paths a PREVIOUS SHIPPED manifest is known to have
#     installed, which the CURRENT manifest no longer ships. It is NEVER derived from the
#     filesystem. A file that is in neither manifest is not a candidate and cannot become one.
#
# WHERE THE "PREVIOUS" LIST COMES FROM. Each verified manifest may carry a SIGNED `files` map
# {relative/path: sha256hex} (canonical_manifest_bytes above). After an update lands, the client
# persists a cumulative ledger of it in the updater state dotfile; the next update compares that
# ledger against the new manifest's map. The ledger is a LOCAL file and it is NOT signed, so every
# entry is re-validated at the deletion site below (a corrupted/tampered dotfile must not be able
# to name `../../` anything).
#
# KNOWN RESIDUAL, stated honestly (adversarial review finding F3): because the PREVIOUS half of the
# allowlist comes from that unsigned dotfile, "only a signed manifest can name a deletion" is true
# of the CURRENT half only. Whoever can write `~/.neurahash_autoupdate.json` (or set
# NEURAHASH_AUTOUPDATE_STATE) chooses candidates -- bounded by every rule below: safe relative path,
# not never-touch, inside the root, a regular file, a sha256 they must already know, under the
# mass-delete ceiling, and with the whole feature off by default. An attacker with write access to
# that dotfile generally has write access to the install directory anyway. Closing it properly means
# binding each ledger entry to the manifest signature that declared it; not done here.
#
# FIVE REFUSALS, EACH LOGGED BY NAME (silence must never read as success):
#   1. the path is not a safe RELATIVE path (absolute, drive-qualified, UNC, or contains `..`);
#   2. it resolves -- after symlink resolution -- outside the install root;
#   3. it matches the hard-coded NEVER-TOUCH list (wallet/keystore/state/log/checkpoint/corpus/
#      .git/venv). These should already be impossible by construction, because they were never in
#      any manifest. The list is DEFENCE IN DEPTH: this is the failure mode that loses money, and
#      the cost of a false refusal is one stale file, while the cost of a false delete is a wallet;
#   4. it is not a REGULAR FILE (a symlink, a directory, a device);
#   5. its sha256 does not match what the old manifest recorded shipping -- i.e. the operator has
#      edited it, or it is not the file we think it is. Leave it.
#
# DEFAULT IS NOT DELETION. With the knob unset this is a DRY RUN: it prints exactly what it would
# remove and removes nothing. Deleting requires an explicit opt-in (`NEURAHASH_UPDATE_RECLAIM=1`).
RECLAIM_MODE_OFF = "off"                 # do not even look
RECLAIM_MODE_DRY = "dry-run"             # look, report, delete NOTHING  <-- the default
RECLAIM_MODE_DELETE = "delete"           # actually unlink (explicit opt-in only)
_RECLAIM_ARMED = {"1", "true", "yes", "on", "delete", "apply", "reclaim", "enabled"}
_RECLAIM_DISABLED = {"0", "false", "no", "off", "never", "disabled"}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_WIN_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"]
    + [f"COM{i}" for i in range(1, 10)] + [f"LPT{i}" for i in range(1, 10)])
# A pass that wants to remove MOST of what was previously shipped is not a cleanup, it is an
# accident (a hand-written / partial `files` map -- adversarial review finding F2, measured: a map
# listing 1 of 5 shipped paths deleted the updater itself). Refuse the whole pass rather than
# half-destroy the install; small passes are still allowed by the absolute floor.
RECLAIM_MAX_FRACTION = 0.25
RECLAIM_MIN_ABSOLUTE = 5

# A path component equal to any of these makes the whole path untouchable. Lowercased compare.
NEVER_TOUCH_DIR_PARTS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "site-packages", "__pycache__",
    "wallet", "wallets", "keystore", "keystores", "key", "keys", "secret", "secrets",
    "state", "logs", "log", "checkpoint", "checkpoints", "ckpt", "ckpts", "runs", "artifacts",
    "corpus", "corpora", "corpus_cache", "cache", "data", "datasets", "_poollive",
})
# A path component STARTING with any of these (this repo's own experiment/state dir conventions).
NEVER_TOUCH_PART_PREFIXES = ("_state", "_poollive", ".neurahash", "_rl_", "_granite", "_rung",
                             "_xarch", "_olmoe")
# Exact basenames (lowercased) that must survive regardless of what any manifest says.
NEVER_TOUCH_NAMES = frozenset({
    "version", "requirements.txt", "release.json", ".env", ".gitignore", ".gitattributes",
    "miner_wallet.json", "wallet.json", "keystore.json", "identity.json",
})
NEVER_TOUCH_SUFFIXES = (".key", ".pem", ".p12", ".pfx", ".keystore", ".seed", ".mnemonic",
                        ".log", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".npy", ".npz",
                        ".db", ".sqlite", ".sqlite3", ".pid", ".sock", ".lock")
# Matched against the WHOLE lowercased relative path, not just the basename. Deliberately broad:
# a false refusal costs one stale file; a false delete costs the miner's wallet.
NEVER_TOUCH_SUBSTRINGS = ("wallet", "keystore", "privkey", "private_key", "secret", "mnemonic",
                          "passphrase", "credential", "seed_phrase", "token")


def reclaim_mode(environ=None, override=None):
    """Resolve the reclaim mode. Returns exactly one of RECLAIM_MODE_OFF/DRY/DELETE.

    UNSET => DRY RUN. Deletion is opt-in and nothing else: an operator who has never heard of this
    feature can never lose a byte to it, but they DO get told what it would have removed.

    Two rules the adversarial review (finding F6) forced:
      * an explicit `NEURAHASH_UPDATE_RECLAIM=0` is the operator's KILL SWITCH and no caller-side
        `override` may lift it -- a knob you can be overridden out of is not a kill switch;
      * an unrecognised `override` (`True`, "yes", a typo) falls back to the DRY RUN rather than
        being passed through as a mode string. Previously `reclaim=True` stringified to "true",
        matched nothing, and silently did nothing while reading as "armed" at the call site."""
    env = os.environ if environ is None else environ
    raw = str(env.get(RECLAIM_ENV, "")).strip().lower()
    env_off = raw in _RECLAIM_DISABLED
    if override:
        ov = str(override).strip().lower()
        if env_off:
            return RECLAIM_MODE_OFF
        return ov if ov in (RECLAIM_MODE_OFF, RECLAIM_MODE_DRY, RECLAIM_MODE_DELETE) else RECLAIM_MODE_DRY
    if raw in _RECLAIM_ARMED:
        return RECLAIM_MODE_DELETE
    if env_off:
        return RECLAIM_MODE_OFF
    return RECLAIM_MODE_DRY


def manifest_files(manifest):
    """The manifest's signed `files` map as {str: str}. Absent/not-an-object -> {}.

    Entries are NOT validated here on purpose -- a bad path must be refused at the DELETION site
    (`plan_reclaim`), which is the only place that can be bypassed by neither a caller nor a
    corrupted state file. Filtering here would move a security check away from the thing it
    protects."""
    if not isinstance(manifest, dict):
        return {}
    raw = manifest.get(MANIFEST_FILES_KEY)
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _normalise_relpath(rel):
    """(normalised_posix_relpath, None) or (None, reason). Refuses anything that is not a plain
    relative path INSIDE the tree: absolute, drive-qualified (`C:/..`), UNC (`//host/share`), any
    `..` component, a NUL byte, or an empty result.

    WINDOWS PATH EQUIVALENCE (adversarial review 2026-08-08, finding F1 -- MEASURED, not
    theoretical). Windows silently strips a trailing `.` or space from every path component and
    treats `name:stream` as an alternate data stream of `name`. So `.git./HEAD` opens `.git/HEAD`
    and `identity.json.` opens `identity.json`. Every name-based guard downstream (the never-touch
    list, the still-shipped exclusion) compares the STRING it was given, while `os.remove` acts on
    the RESOLVED file -- a one-character spelling change walked straight past the never-touch list
    and deleted `.git/HEAD`, `keys/node_id.json` and `identity.json` in a probe. Refusing these
    spellings here is the only place that closes the whole class at once: no legitimate shipped
    path needs a trailing dot, a trailing space, or a colon in a component."""
    if not isinstance(rel, str) or not rel.strip():
        return None, "path is empty or not a string"
    s = rel.strip().replace("\\", "/")
    if "\x00" in s:
        return None, "NUL byte in path"
    if s.startswith("//"):
        return None, "UNC path (starts with //)"
    if s.startswith("/"):
        return None, "absolute path (starts with /)"
    if _DRIVE_RE.match(s):
        return None, "drive-qualified absolute path"
    parts = [p for p in s.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None, "'..' traversal component"
    if not parts:
        return None, "path normalises to nothing"
    for p in parts:
        if p != p.rstrip(". "):
            return None, (f"component {p!r} ends in a dot or space (Windows strips it, so it "
                          f"aliases another file)")
        if ":" in p:
            return None, f"component {p!r} contains ':' (Windows alternate data stream)"
        if any(ord(c) < 32 for c in p):
            return None, f"component {p!r} contains a control character"
        if p.split(".")[0].upper() in _WIN_RESERVED_NAMES:
            return None, f"component {p!r} is a reserved Windows device name"
    return "/".join(parts), None


def _never_touch_reason(norm_rel):
    """Why this path is on the hard-coded never-touch list, or None."""
    low = norm_rel.lower()
    parts = low.split("/")
    for p in parts:
        if p in NEVER_TOUCH_DIR_PARTS:
            return f"never-touch path component {p!r}"
        for pre in NEVER_TOUCH_PART_PREFIXES:
            if p.startswith(pre):
                return f"never-touch component prefix {pre!r}"
    name = parts[-1]
    if name in NEVER_TOUCH_NAMES:
        return f"never-touch filename {name!r}"
    for suf in NEVER_TOUCH_SUFFIXES:
        if name.endswith(suf):
            return f"never-touch suffix {suf!r}"
    for sub in NEVER_TOUCH_SUBSTRINGS:
        if sub in low:
            return f"never-touch substring {sub!r}"
    return None


def _inside(root_real, target_real):
    """True iff target_real is root_real or lives under it (case-insensitive on Windows)."""
    r = os.path.normcase(os.path.abspath(root_real)).rstrip(os.sep)
    t = os.path.normcase(os.path.abspath(target_real))
    return t == r or t.startswith(r + os.sep)


class ReclaimAction:
    """One candidate and what happened to it. `verdict` is one of:
    'delete' (removed), 'would-delete' (dry run), 'refused' (a safety rule said no),
    'absent' (already gone), 'error' (removal failed). `reason` is ALWAYS populated."""

    def __init__(self, rel, verdict, reason, path=None):
        self.rel = rel
        self.verdict = verdict
        self.reason = reason
        self.path = path

    def __repr__(self):
        return f"ReclaimAction({self.rel!r}, {self.verdict!r}, {self.reason!r})"


class ReclaimReport:
    """Outcome of one reclaim pass. Every candidate appears in `actions` with a reason."""

    def __init__(self, mode, actions=None, candidates=None):
        self.mode = mode
        self.actions = list(actions or [])
        self.candidates = list(candidates or [])

    def _of(self, verdict):
        return [a.rel for a in self.actions if a.verdict == verdict]

    @property
    def deleted(self):
        return self._of("delete")

    @property
    def would_delete(self):
        return self._of("would-delete")

    @property
    def refused(self):
        return [(a.rel, a.reason) for a in self.actions if a.verdict in ("refused", "error")]

    @property
    def absent(self):
        return self._of("absent")

    def summary(self):
        return (f"mode={self.mode} candidates={len(self.candidates)} deleted={len(self.deleted)} "
                f"would_delete={len(self.would_delete)} refused={len(self.refused)} "
                f"already_absent={len(self.absent)}")

    def __repr__(self):
        return f"ReclaimReport({self.summary()})"


def plan_reclaim(install_root, prev_files, cur_files):
    """Decide, WITHOUT touching the filesystem's contents, what each candidate's fate is.

    candidates = keys(prev_files) - keys(cur_files). That is the whole allowlist. The filesystem is
    only ever ASKED ABOUT these paths; it is never enumerated, so a file that no manifest shipped
    is not a candidate and cannot become one. Returns (candidates, [ReclaimAction, ...]) where every
    action carries a non-empty reason."""
    prev_files = prev_files if isinstance(prev_files, dict) else {}
    cur_files = cur_files if isinstance(cur_files, dict) else {}
    root_real = os.path.realpath(os.path.abspath(install_root))
    cur_norm, cur_ids = set(), set()
    for k in cur_files:
        n, _why = _normalise_relpath(k)
        if not n:
            continue
        cur_norm.add(os.path.normcase(n))
        # IDENTITY, not spelling. A string set alone is defeated by every way two names can mean
        # one file: case on Windows/macOS, 8.3 short names, hard links, trailing dots. (st_dev,
        # st_ino) is what the filesystem itself considers the same file, so a candidate that
        # resolves onto a file the CURRENT release ships is caught however it was spelled.
        try:
            st = os.stat(os.path.join(root_real, *n.split("/")))
            cur_ids.add((st.st_dev, st.st_ino))
        except OSError:
            pass
    # THE ALLOWLIST: shipped-before minus shipped-now. An unsafe path cannot be normalised, so it
    # cannot be matched against the current list either -- it stays a candidate and gets refused
    # below by name, rather than being dropped silently.
    candidates = []
    for rel in sorted(str(k) for k in prev_files):
        norm, _why = _normalise_relpath(rel)
        if norm is not None and os.path.normcase(norm) in cur_norm:
            continue                                     # still shipped -- not a candidate at all
        candidates.append(rel)
    actions = []
    # F2 mass-delete guard: refuse the WHOLE pass rather than gut the install on a bad `files` map.
    ceiling = max(RECLAIM_MIN_ABSOLUTE, int(RECLAIM_MAX_FRACTION * len(prev_files)))
    if len(candidates) > ceiling:
        why = (f"REFUSING THE WHOLE PASS: {len(candidates)} of {len(prev_files)} previously shipped "
               f"paths would be removed (> {ceiling}); that is a partial/garbage `files` map, not a "
               f"cleanup")
        return candidates, [ReclaimAction(rel, "refused", why) for rel in candidates]
    for rel in candidates:
        norm, why = _normalise_relpath(rel)
        if norm is None:
            actions.append(ReclaimAction(rel, "refused", f"unsafe path: {why}"))
            continue
        why = _never_touch_reason(norm)
        if why:
            actions.append(ReclaimAction(rel, "refused", why))
            continue
        target = os.path.join(root_real, *norm.split("/"))
        # realpath BEFORE the containment test: a symlinked component that points out of the tree
        # would otherwise pass a pure-string check and delete somebody else's file.
        target_real = os.path.realpath(target)
        if not _inside(root_real, target_real):
            actions.append(ReclaimAction(rel, "refused",
                                         "resolves outside the install root (symlink or traversal)"))
            continue
        if os.path.islink(target):
            actions.append(ReclaimAction(rel, "refused", "is a symlink, not a regular file", target))
            continue
        if not os.path.exists(target):
            actions.append(ReclaimAction(rel, "absent", "already gone", target))
            continue
        if not os.path.isfile(target):
            actions.append(ReclaimAction(rel, "refused", "not a regular file (directory or device)",
                                         target))
            continue
        try:
            st = os.stat(target)
            same_as_current = (st.st_dev, st.st_ino) in cur_ids
        except OSError:
            same_as_current = False
        if same_as_current:
            actions.append(ReclaimAction(rel, "refused",
                                         "resolves onto a file the CURRENT manifest still ships",
                                         target))
            continue
        want = str(prev_files.get(rel, "")).strip().lower()
        if not _SHA256_RE.match(want):
            actions.append(ReclaimAction(rel, "refused",
                                         "old manifest recorded no usable sha256 for it", target))
            continue
        have = _sha256_file(target)
        if have is None:
            actions.append(ReclaimAction(rel, "refused", "could not be hashed (unreadable)", target))
            continue
        if have.lower() != want:
            actions.append(ReclaimAction(rel, "refused",
                                         f"content differs from the shipped file (on-disk "
                                         f"{have[:12]} != shipped {want[:12]}) -- user-modified",
                                         target))
            continue
        actions.append(ReclaimAction(rel, "would-delete", "shipped by an older release, dropped by "
                                                          "the current one, unmodified", target))
    return candidates, actions


def reclaim_obsolete_files(install_root, prev_files, cur_files, *, mode=None, environ=None,
                           log_fn=None):
    """Run one reclaim pass and return a ReclaimReport. NEVER raises.

    `mode` (RECLAIM_MODE_*) overrides the knob; otherwise `reclaim_mode(environ)` decides, and with
    the knob unset that is a DRY RUN. Every candidate is logged with the reason it was kept or
    removed -- a silent pass is indistinguishable from a broken one, so there is no silent pass."""
    log_fn = log_fn or log
    mode = reclaim_mode(environ, mode)
    if mode == RECLAIM_MODE_OFF:
        log_fn(f"[reclaim] disabled ({RECLAIM_ENV} is off); no obsolete-file check was made")
        return ReclaimReport(mode)
    # AN ABSENT LIST MEANS "UNDECLARED", NEVER "SHIPS NOTHING". Caught by
    # test_a_manifest_without_files_never_reclaims_anything, which deleted the WHOLE previous
    # release: every live manifest today is v1/v2 and carries no `files`, so without this guard the
    # first release after this feature ships would have emptied every miner's install directory of
    # everything the previous one installed -- including tools/self_update.py itself. Only a manifest
    # that explicitly declares what it ships may cause a deletion.
    if not cur_files:
        log_fn("[reclaim] the current manifest declares no shipped file list; NO-OP (an absent "
               "list means 'undeclared', never 'ships nothing')")
        return ReclaimReport(mode)
    try:
        candidates, actions = plan_reclaim(install_root, prev_files, cur_files)
    except Exception as e:
        log_fn(f"[reclaim] WARN: planning failed ({type(e).__name__}: {e}); nothing was deleted")
        return ReclaimReport(mode)
    if mode == RECLAIM_MODE_DELETE:
        for a in actions:
            if a.verdict != "would-delete":
                continue
            # TOCTOU (adversarial review finding F4, measured): plan_reclaim hashes ALL candidates
            # and only then returns, so for candidate #1 the hash->unlink gap spans the full read of
            # candidates #2..N. An operator edit landing in that window was destroyed. Re-hash
            # HERE, immediately before the unlink, so the window is as small as the OS allows.
            want = str(prev_files.get(a.rel, "")).strip().lower()
            have = _sha256_file(a.path)
            if have is None or have.lower() != want:
                a.verdict = "refused"
                a.reason = ("content changed between the check and the removal -- kept "
                            "(re-verified immediately before unlink)")
                continue
            try:
                os.remove(a.path)
                a.verdict, a.reason = "delete", "removed (obsolete, unmodified, inside the root)"
            except Exception as e:
                a.verdict, a.reason = "error", f"removal failed ({type(e).__name__}: {e})"
    report = ReclaimReport(mode, actions, candidates)
    if not candidates:
        log_fn(f"[reclaim] {mode}: no previous release file list to compare against yet "
               f"(0 candidates); nothing to do")
        return report
    for a in actions:
        log_fn(f"[reclaim] {a.verdict.upper():<12} {a.rel} -- {a.reason}")
    log_fn(f"[reclaim] {report.summary()}")
    if mode == RECLAIM_MODE_DRY and report.would_delete:
        log_fn(f"[reclaim] DRY RUN -- deleted nothing. Set {RECLAIM_ENV}=1 to actually reclaim "
               f"the {len(report.would_delete)} file(s) above.")
    return report


def _reclaim_after_update(repo_dir, state_path, manifest, *, mode=None, environ=None, log_fn=None):
    """Reclaim step wired into an applied update, plus the persisted shipped-file ledger.

    The ledger is CUMULATIVE ({**previous, **current}) so a miner that skips releases still knows
    what an older one installed; paths that were actually deleted drop out of it. Never raises: a
    failure here must never strand a miner mid-update."""
    log_fn = log_fn or log
    try:
        st = _load_state(state_path)
        prev = st.get("shipped_files")
        prev = {str(k): str(v) for k, v in prev.items()} if isinstance(prev, dict) else {}
        cur = manifest_files(manifest)
        report = reclaim_obsolete_files(repo_dir, prev, cur, mode=mode, environ=environ,
                                        log_fn=log_fn)
        ledger = dict(prev)
        ledger.update(cur)
        for rel in report.deleted:
            ledger.pop(rel, None)
        if ledger != prev:
            _save_state(state_path, shipped_files=ledger)
        return report
    except Exception as e:
        log_fn(f"[reclaim] WARN: reclaim step raised ({type(e).__name__}: {e}); nothing was deleted")
        return ReclaimReport(reclaim_mode(environ, mode))


# ------------------------------------------------------------------ the orchestrator
def check_and_update(repo_dir=REPO, argv=None, *, manifest_url=None, mirrors=None,
                     pubkey=PINNED_RELEASE_PUBKEY, enabled=None, state_path=None,
                     rate_limit_s=DEFAULT_RATE_LIMIT_S, now=None, honor_rate_limit=True,
                     timeout=STARTUP_TIMEOUT_S, manifest=None, reclaim=None, environ=None,
                     fetch_fn=None, git_fn=None, pip_fn=None, reexec_fn=None):
    """Do at most one signed-update check and, if a VERIFIED forward release exists, apply it and
    re-exec. Returns an UpdateResult. FAIL CLOSED: any error is caught, logged as a warning, and
    the working tree is left untouched (control returns to the caller so the miner keeps running).

    `mirrors` (or the legacy single `manifest_url`) selects where to look; the default is the
    compiled MIRRORS list, and the BEST VERIFIED version across all of them wins. `manifest` lets a
    caller pass an ALREADY-VERIFIED manifest so a startup sync does not fetch twice.

    `reclaim` overrides the obsolete-file reclaim mode (RECLAIM_MODE_*); unset means the
    NEURAHASH_UPDATE_RECLAIM knob decides, and an absent knob is a DRY RUN that deletes nothing.

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

    # 6b) reclaim files an OLDER release shipped that this one no longer does ------------------
    # Placed HERE deliberately: after HEAD is proven to equal the signed commit (so `manifest` is
    # the manifest for the tree now on disk) and BEFORE the re-exec (which never returns). Nothing
    # below the reclaim depends on it, and it can never fail the update -- it is best-effort.
    _reclaim_after_update(repo_dir, spath, manifest, mode=reclaim, environ=environ, log_fn=log)

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
                       timeout=STARTUP_TIMEOUT_S, environ=None, reclaim=None,
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
                               reclaim=reclaim, environ=environ,
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
