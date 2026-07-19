#!/usr/bin/env python3
"""tools/self_update.py -- SIGNED, FAIL-CLOSED auto-update for the public NeuraHash miner.

WHY THIS EXISTS
    The public client is `git clone` + `pip install -r requirements.txt` with no version file
    and no self-update, so every code change means a manual re-clone / a forum ping. This module
    lets the operator push ONE signed release and have every running miner self-update on its next
    check -- WITHOUT ever running code that is not cryptographically signed by the project's
    release key. Auto-running pushed code on strangers' machines means a repo/mirror compromise
    could push malware to the whole fleet, so the ONE hard rule is:

        a miner NEVER checks out / runs code unless a manifest signed by the PINNED release key
        says to, and it never DOWNGRADES. On ANY doubt it stays on the code it already has.

HOW IT WORKS (all steps fail CLOSED -- any failure => stay on the current version, keep mining)
    1. fetch a signed manifest (release.json) over HTTPS from a HARD-CODED url (never from the
       manifest, never a shell command from the manifest);
    2. VERIFY its secp256k1 signature against a PINNED release public key baked into this file,
       using the repo's own signing lib (neura_l1.signing) -- no hand-rolled crypto;
    3. only if verify OK AND manifest.version > local VERSION (a strict FORWARD move) do we
       `git fetch` + `git checkout <manifest.git_commit>` (the commit is validated to be a bare
       hex id -- no arbitrary refs);
    4. VERIFY HEAD == manifest.git_commit after checkout;
    5. `pip install -r requirements.txt` ONLY if requirements.txt changed;
    6. re-exec the miner on the new code.

    A bad signature, wrong key, unreachable manifest, malformed manifest, commit mismatch, or a
    downgrade attempt -> a clear WARNING is logged and control returns unchanged. It NEVER raises
    out to the caller and NEVER hard-crashes the miner.

OPT-OUT + RATE LIMIT
    Default ON. `NEURAHASH_AUTOUPDATE=0` (or the run_miner_client `--no-auto-update` flag, which
    sets it) fully disables it. The check is rate-limited (once / 6h by default) via a small JSON
    dotfile so a run-forever miner does not hammer GitHub.

Everything network/git/pip/re-exec is injectable (fetch_fn/git_fn/pip_fn/reexec_fn) so the whole
policy is unit-tested with NO real network, git, pip, or process replacement -- see
tests/test_self_update.py.
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
# (The prior TEST key 0x19E7E376...aff2A derived from 0x11..11 remains ONLY in tests/test_self_update.py.)
PINNED_RELEASE_PUBKEY = "0x5168F6cc4cc05bfd6d4714906d68e083c02dDC66"  # real offline release address
# ===========================================================================================

# The manifest is fetched from exactly this url -- a single hard-coded constant. It is NEVER
# read from the manifest itself (a compromised manifest cannot redirect the fetch).
MANIFEST_URL = "https://raw.githubusercontent.com/whitegreenstudios/neurahash-miner/main/release.json"

# Domain tag mixed into the signed bytes so a release-manifest signature can never be confused
# with any other signed object this project produces.
RELEASE_KIND = "neurahash-miner-release"

AUTOUPDATE_ENV = "NEURAHASH_AUTOUPDATE"          # "0"/"false"/"no"/"off" => disabled
STATE_ENV = "NEURAHASH_AUTOUPDATE_STATE"         # override the rate-limit dotfile path (tests/ops)
DEFAULT_RATE_LIMIT_S = 6 * 3600                  # at most one check per 6h
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
    if any, is IGNORED). Any tampering with version/git_commit/published_ts changes these bytes,
    so recovery yields a different address and verification fails."""
    body = {
        "kind": RELEASE_KIND,
        "version": str(manifest["version"]),
        "git_commit": str(manifest["git_commit"]),
        "published_ts": int(manifest["published_ts"]),
    }
    return _canon(body)


def verify_manifest(manifest, pubkey=PINNED_RELEASE_PUBKEY):
    """Verify a release manifest against the PINNED release public key. Returns (ok, reason).

    Rejects (fail closed):
      * a non-dict / missing required field (version, git_commit, published_ts, signature);
      * a missing / empty signature;
      * a git_commit that is not a bare hex commit id;
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
    """Fetch the manifest text over HTTPS from the hard-coded url. HTTPS is enforced; any network
    error propagates to the caller, which treats it as 'stay on current version'."""
    if not str(url).lower().startswith("https://"):
        raise ValueError(f"refusing to fetch manifest over non-HTTPS url: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "neurahash-miner-selfupdate"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 (https enforced above)
        return resp.read().decode("utf-8")


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


def _load_last_check(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("last_check", 0.0))
    except Exception:
        return 0.0


def _save_last_check(path, ts):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"last_check": float(ts)}, f)
    except Exception as e:
        log(f"WARN: could not persist rate-limit state to {path}: {e}")


# ------------------------------------------------------------------ result object
class UpdateResult:
    """Outcome of a check. `applied` is True only when a verified forward update was fully applied
    and a re-exec was requested (with the real reexec_fn the process is already gone; a test's fake
    reexec_fn lets this return). `action` is a short machine tag; `reason` is human detail."""

    def __init__(self, applied, action, reason="", local_version=None, target_version=None,
                 checked_out=None, pip_ran=False):
        self.applied = applied
        self.action = action
        self.reason = reason
        self.local_version = local_version
        self.target_version = target_version
        self.checked_out = checked_out
        self.pip_ran = pip_ran

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
def check_and_update(repo_dir=REPO, argv=None, *, manifest_url=MANIFEST_URL,
                     pubkey=PINNED_RELEASE_PUBKEY, enabled=None, state_path=None,
                     rate_limit_s=DEFAULT_RATE_LIMIT_S, now=None, honor_rate_limit=True,
                     fetch_fn=None, git_fn=None, pip_fn=None, reexec_fn=None):
    """Do at most one signed-update check and, if a VERIFIED forward release exists, apply it and
    re-exec. Returns an UpdateResult. FAIL CLOSED: any error is caught, logged as a warning, and
    the working tree is left untouched (control returns to the caller so the miner keeps running).

    Injectables (real defaults if None): fetch_fn(url)->text, git_fn(repo,*args)->(rc,out),
    pip_fn(repo)->(rc,out), reexec_fn(argv)->NoReturn. Tests pass fakes so nothing real happens.
    """
    argv = list(argv if argv is not None else sys.argv)
    fetch_fn = fetch_fn or _default_fetch
    git_fn = git_fn or _default_git
    pip_fn = pip_fn or _default_pip
    reexec_fn = reexec_fn or _default_reexec
    now = time.time() if now is None else now
    if enabled is None:
        enabled = _env_enabled()

    try:
        local_version = read_local_version(repo_dir)
    except Exception as e:
        log(f"WARN: cannot read local {VERSION_FILE} ({e}); skipping auto-update, staying put")
        return UpdateResult(False, "no-version-file", reason=str(e))

    if not enabled:
        return UpdateResult(False, "disabled", reason=f"{AUTOUPDATE_ENV} is off",
                            local_version=local_version)

    spath = _state_path(repo_dir, state_path)
    if honor_rate_limit:
        last = _load_last_check(spath)
        if now - last < rate_limit_s:
            return UpdateResult(False, "rate-limited",
                                reason=f"checked {int(now - last)}s ago (< {rate_limit_s}s)",
                                local_version=local_version)
    # record the attempt up-front so a crash/re-exec loop is throttled by the same rate limit.
    _save_last_check(spath, now)

    # 1) fetch --------------------------------------------------------------------------------
    try:
        text = fetch_fn(manifest_url)
        manifest = json.loads(text)
    except Exception as e:
        log(f"WARN: could not fetch/parse release manifest ({e}); staying on v{local_version}")
        return UpdateResult(False, "fetch-failed", reason=str(e), local_version=local_version)

    # 2) VERIFY signature against the pinned key ----------------------------------------------
    ok, info = verify_manifest(manifest, pubkey)
    if not ok:
        log(f"WARN: release manifest REJECTED ({info}); staying on v{local_version} "
            f"(never running unverified code)")
        return UpdateResult(False, "verify-failed", reason=info, local_version=local_version)

    target_version = str(manifest["version"])
    commit = str(manifest["git_commit"])

    # 3) forward-only gate (no downgrade, no re-apply of the same version) --------------------
    try:
        forward = is_forward(target_version, local_version)
    except Exception as e:
        log(f"WARN: cannot compare versions ({e}); staying on v{local_version}")
        return UpdateResult(False, "version-parse-failed", reason=str(e),
                            local_version=local_version, target_version=target_version)
    if not forward:
        return UpdateResult(False, "no-op-not-forward",
                            reason=f"manifest v{target_version} <= local v{local_version}",
                            local_version=local_version, target_version=target_version)

    log(f"verified signed release v{target_version} (commit {commit[:12]}) > local v{local_version}; "
        f"applying update")

    # 4) apply: git fetch + checkout <pinned commit> (list-arg git, hex-validated commit) ------
    req_before = _sha256_file(os.path.join(repo_dir, REQUIREMENTS_FILE))
    try:
        rc, out = git_fn(repo_dir, "fetch", "--quiet", "origin")
        if rc != 0:
            log(f"WARN: `git fetch` failed (rc={rc}); staying on v{local_version}. {out.strip()[-200:]}")
            return UpdateResult(False, "git-fetch-failed", reason=out.strip()[-200:],
                                local_version=local_version, target_version=target_version)
        rc, out = git_fn(repo_dir, "checkout", "--quiet", commit)
        if rc != 0:
            log(f"WARN: `git checkout {commit[:12]}` failed (rc={rc}); staying on v{local_version}. "
                f"{out.strip()[-200:]}")
            return UpdateResult(False, "git-checkout-failed", reason=out.strip()[-200:],
                                local_version=local_version, target_version=target_version)
    except Exception as e:
        log(f"WARN: git error during update ({e}); staying on v{local_version}")
        return UpdateResult(False, "git-error", reason=str(e),
                            local_version=local_version, target_version=target_version)

    # 5) VERIFY the tree is now exactly the signed commit -------------------------------------
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
                            local_version=local_version, target_version=target_version)

    # 6) pip install ONLY if requirements.txt actually changed --------------------------------
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

    # 7) re-exec onto the new code ------------------------------------------------------------
    log(f"update to v{target_version} applied; re-exec'ing miner on the new code")
    result = UpdateResult(True, "applied", reason=f"v{local_version} -> v{target_version}",
                          local_version=local_version, target_version=target_version,
                          checked_out=commit, pip_ran=pip_ran)
    reexec_fn(argv)          # real impl never returns; a test fake returns and we fall through
    return result


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
