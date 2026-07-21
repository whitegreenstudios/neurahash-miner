#!/usr/bin/env python3
"""tools/install_kubo.py -- SAFE, pinned, local auto-install of the kubo (`ipfs`) binary.

WHY THIS EXISTS: `tools/run_miner.py` publishes a trained delta by PINNING it to IPFS, which needs
either a Pinata JWT (the miner's own third-party account) or a local kubo binary. On a bare machine
neither exists, so `--doctor` reports `FAIL pinning backend` and the miner stays in LOCAL mode --
the last non-credential blocker to a zero-configuration join.

WHY IT IS WRITTEN THE WAY IT IS: this module downloads a binary and then EXECUTES it, on machines
that also hold miners' wallet keys. A naive "download and run" is a supply-chain backdoor, so every
decision here is a security decision:

  * PINNED DIGESTS. `RELEASES` maps (system, machine) -> the exact archive, its PUBLISHED SHA-512,
    and its PUBLISHED byte size, for ONE kubo version compiled into this source. The download is
    verified against those constants BEFORE anything is extracted. A mismatch deletes the download
    and installs NOTHING -- there is deliberately no "install anyway" path.
    (SHA-512, not SHA-256, because SHA-512 is what the kubo project actually PUBLISHES -- see
    KUBO_CHECKSUM_URL_FMT below. Pinning a digest we computed ourselves would pin nothing.)
  * OFFICIAL SOURCE ONLY. The URL is built from compiled-in constants and the platform key. It is
    never read from config, env, a manifest, or a command line -- so nothing a coordinator or an
    attacker-controlled file says can redirect the download.
  * NO SHELL. urllib + zipfile/tarfile + one `subprocess.run([...])` with a list argv. No
    `shell=True`, no `curl | sh`, and the archive's own `install.sh` is never extracted, let alone run.
  * LOCAL, NOT SYSTEM-WIDE. Everything lands in `<work_dir>/kubo/`. No PATH edits, no admin, no
    writes outside the miner's own work dir (the temp download dir is inside it too).
  * ZIP-SLIP REFUSED. Every member name in the archive is resolved against the destination first;
    if ANY escapes it (or is a tar symlink/device entry), the whole install is refused.
  * OPT-OUT. `NEURAHASH_AUTO_INSTALL_KUBO=0` (or run_miner's `--no-auto-install-kubo`), mirroring
    how `NEURAHASH_AUTOUPDATE` / `--no-auto-update` work in tools/self_update.py.
  * NEVER CRASHES THE MINER. `ensure_kubo()` catches everything and returns (None, reason). Offline,
    hash mismatch, unsupported platform, disk full -- all produce one clear log line and leave the
    miner in LOCAL mode. This client's core promise is that it never fails for lack of infra.

Everything with a side effect is injectable (fetch_fn / probe_fn) so the test suite exercises the
real verify/extract/refuse logic with NO network and NO downloads -- same pattern as
tests/test_self_update.py.
"""
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

# --------------------------------------------------------------------------- the pinned release
# ONE version, compiled in. Bumping it means replacing every digest below with the ones published
# for the new version at KUBO_CHECKSUM_URL_FMT -- never by re-hashing a file we downloaded.
KUBO_VERSION = "v0.42.0"

# Official kubo distribution (the ipfs.tech dist site; the same artifacts are mirrored as the
# ipfs/kubo GitHub release assets, byte sizes cross-checked against both).
KUBO_DIST_BASE = "https://dist.ipfs.tech/kubo/%s/" % KUBO_VERSION
# Where each SHA-512 below was READ FROM (kept here so a reviewer can re-verify every constant):
#   https://dist.ipfs.tech/kubo/v0.42.0/<archive>.sha512
KUBO_CHECKSUM_URL_FMT = KUBO_DIST_BASE + "%s.sha512"


class Release(object):
    """One pinned artifact: what to download, what it must hash to, and what to pull out of it."""

    __slots__ = ("archive", "sha512", "size", "member", "binary")

    def __init__(self, archive, sha512, size, member, binary):
        self.archive = archive          # file name under KUBO_DIST_BASE
        self.sha512 = sha512            # PUBLISHED digest of that file (lowercase hex)
        self.size = size                # PUBLISHED byte size of that file
        self.member = member            # the ONE member we extract (nothing else is written)
        self.binary = binary            # its installed file name

    @property
    def url(self):
        return KUBO_DIST_BASE + self.archive


# (platform.system().lower(), normalized machine) -> Release.
# Digests: `curl https://dist.ipfs.tech/kubo/v0.42.0/<archive>.sha512` (read 2026-07-21).
# Sizes: Content-Length from the same host, cross-checked against the GitHub release asset sizes.
RELEASES = {
    ("windows", "amd64"): Release(
        "kubo_v0.42.0_windows-amd64.zip",
        "5501a7745898e71326e1b85d8d231d79a3409147484ce2ca28da94c9272319e6"
        "c691e19a1ed74c0f0a7beb601fb203e10a6c242aaab1a5961c260b1b0d14c452",
        41339751, "kubo/ipfs.exe", "ipfs.exe"),
    ("linux", "amd64"): Release(
        "kubo_v0.42.0_linux-amd64.tar.gz",
        "054c38a0cf66f7d738e25085ad62cb3a42d03d4bac329b7dd25c1d71cf18e1ce"
        "87d55b1d1b705b04c65210dca9109973579e0eb1cd72f6341ecb3311d840d156",
        54571642, "kubo/ipfs", "ipfs"),
    ("linux", "arm64"): Release(
        "kubo_v0.42.0_linux-arm64.tar.gz",
        "5f4abb1a63e82bbdd0417517eb1c7bb5f64e95da2724f85d9762f640ddb9e6a5"
        "728bb86d60022ac367accf14248d80a0484cf7960392e15e540dfbf655974def",
        38212454, "kubo/ipfs", "ipfs"),
    ("darwin", "arm64"): Release(
        "kubo_v0.42.0_darwin-arm64.tar.gz",
        "5f863972f7edee0ac3f003d8b097366927e8d9f651fd5c74e1fda980f766dbf7"
        "af4b8a813b13400eb2bcf3a871a494a139c85f022e239b237581ad152259cd22",
        40640478, "kubo/ipfs", "ipfs"),
    ("darwin", "amd64"): Release(
        "kubo_v0.42.0_darwin-amd64.tar.gz",
        "090105ea166d4db85ff6a5f9a2e12c6efd451bd6ba15336aa3de5534ef48fa48"
        "706cd0911a40c071175e08131432158c07cac8060721185e50ab0dd46011c7bb",
        43781549, "kubo/ipfs", "ipfs"),
}

AUTO_INSTALL_ENV = "NEURAHASH_AUTO_INSTALL_KUBO"   # "0"/"false"/"no"/"off" => disabled
IPFS_BIN_ENV = "NEURAHASH_IPFS_BIN"                # what tools/ipfs_checkpoint.py reads
_FALSEY = {"0", "false", "no", "off", ""}          # same set as tools/self_update.py

INSTALL_SUBDIR = "kubo"                            # <work_dir>/kubo/ -- never anywhere else
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024              # refuse a flood; every pinned archive is <60MB
MIN_BINARY_BYTES = 8 * 1024 * 1024                 # real ipfs is ~40-90MB; a tiny file is not it
MAX_BINARY_BYTES = 512 * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 120
PROBE_TIMEOUT_S = 30


def log(msg):
    """One ASCII line, flushed -- safe on the Windows cp1252 console."""
    print("[install_kubo] %s" % msg, flush=True)


# --------------------------------------------------------------------------- platform resolution
def platform_key(system=None, machine=None):
    """Return the (system, machine) key into RELEASES, or None if this platform is not pinned.

    Unsupported is a normal, clean outcome (32-bit, riscv, freebsd, ...): the caller stays in LOCAL
    mode. We never guess a nearby build."""
    import platform as _platform
    system = (system if system is not None else _platform.system()).strip().lower()
    machine = (machine if machine is not None else _platform.machine()).strip().lower()
    machine = {"x86_64": "amd64", "amd64": "amd64",
               "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if machine is None:
        return None
    key = (system, machine)
    return key if key in RELEASES else None


def env_enabled(env=None):
    """Auto-install is ON unless NEURAHASH_AUTO_INSTALL_KUBO is explicitly falsey (mirrors
    self_update._env_enabled for NEURAHASH_AUTOUPDATE, so the two knobs behave identically)."""
    env = os.environ if env is None else env
    return env.get(AUTO_INSTALL_ENV, "1").strip().lower() not in _FALSEY


def install_dir(work_dir):
    return os.path.join(os.path.abspath(work_dir), INSTALL_SUBDIR)


def installed_binary(work_dir, key=None):
    """Path to a kubo binary ALREADY installed in this work dir, else None."""
    key = key or platform_key()
    names = ("ipfs.exe", "ipfs") if key is None else (RELEASES[key].binary,)
    for name in names:
        p = os.path.join(install_dir(work_dir), name)
        if os.path.isfile(p):
            return p
    return None


def find_kubo(work_dir=None, env=None, key=None):
    """The kubo binary this miner can use: an operator-provided/PATH one first (we never override
    an install the machine already has), then our own local install. None if neither exists."""
    env = os.environ if env is None else env
    onpath = shutil.which(env.get(IPFS_BIN_ENV, "ipfs") or "ipfs")
    if onpath:
        return onpath
    return installed_binary(work_dir, key) if work_dir else None


# --------------------------------------------------------------------------- download + verify
def _default_fetch(url, timeout=DOWNLOAD_TIMEOUT_S):
    """GET `url` and return its bytes. HTTPS only, hard byte cap, no redirect to another scheme.
    `url` always comes from the compiled-in table -- never from config, env, or argv."""
    import urllib.request
    if not url.startswith("https://"):
        raise ValueError("refusing a non-HTTPS kubo URL: %s" % url)
    req = urllib.request.Request(url, headers={"User-Agent": "neurahash-miner-install-kubo"})
    chunks, total = [], 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 (scheme checked above)
        if not resp.geturl().startswith("https://"):
            raise ValueError("refusing a non-HTTPS redirect target")
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("kubo archive exceeded the %d byte cap" % MAX_ARCHIVE_BYTES)
            chunks.append(chunk)
    return b"".join(chunks)


def verify_archive(data, release):
    """(ok, detail). Size FIRST (cheap), then the pinned SHA-512 over the exact bytes downloaded.
    Pure function: no filesystem, no network -- this is the whole trust decision in one place."""
    if len(data) != release.size:
        return False, ("size mismatch: got %d bytes, pinned %d" % (len(data), release.size))
    got = hashlib.sha512(data).hexdigest()
    if got != release.sha512:
        return False, ("sha512 MISMATCH: got %s..., pinned %s..." % (got[:16], release.sha512[:16]))
    return True, "sha512 %s... matches the pinned digest" % got[:16]


# --------------------------------------------------------------------------- extraction
def _escapes(dest_dir, name):
    """True if archive member `name` would be written outside `dest_dir` (zip-slip / tar-slip).
    Absolute paths, drive letters, and any `..` traversal all resolve out and are caught here."""
    if not name or name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":"):
        return True
    dest = os.path.realpath(dest_dir)
    target = os.path.realpath(os.path.join(dest, name.replace("\\", "/")))
    return not (target == dest or target.startswith(dest + os.sep))


def extract_binary(archive_path, dest_dir, release, min_binary_bytes=MIN_BINARY_BYTES):
    """Extract EXACTLY ONE member (`release.member`) into `dest_dir` and return its path.

    Every member name is validated against `dest_dir` before anything is written, and for tar
    archives non-regular entries (symlinks, hardlinks, devices) are refused outright -- one bad
    member fails the WHOLE install rather than being skipped. Nothing else in the archive is
    written to disk at all, so the vendored `install.sh` never even lands on the machine.
    Raises on any refusal; ensure_kubo() turns that into a clean LOCAL-mode reason."""
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, release.binary)

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for name in zf.namelist():
                if _escapes(dest_dir, name):
                    raise ValueError("refusing archive: member escapes the destination: %r" % name)
            if release.member not in zf.namelist():
                raise ValueError("archive does not contain %r" % release.member)
            with zf.open(release.member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            found = None
            for m in tf.getmembers():
                if _escapes(dest_dir, m.name):
                    raise ValueError("refusing archive: member escapes the destination: %r" % m.name)
                if not (m.isfile() or m.isdir()):
                    raise ValueError("refusing archive: non-regular member %r" % m.name)
                if m.name == release.member:
                    found = m
            if found is None:
                raise ValueError("archive does not contain %r" % release.member)
            src = tf.extractfile(found)
            if src is None:
                raise ValueError("could not read %r from the archive" % release.member)
            with src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    size = os.path.getsize(out_path)
    if not (min_binary_bytes <= size <= MAX_BINARY_BYTES):
        os.remove(out_path)
        raise ValueError("extracted binary size %d is outside the sane range [%d, %d]"
                         % (size, min_binary_bytes, MAX_BINARY_BYTES))
    if os.name != "nt":
        os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out_path


def _default_probe(binary):
    """Run the installed binary once, list argv, no shell, bounded time. (ok, detail)."""
    p = subprocess.run([binary, "--version"], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT_S)
    out = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
    line = out[0][:80] if out else ""
    return p.returncode == 0 and "ipfs" in line.lower(), (line or "rc=%d" % p.returncode)


# --------------------------------------------------------------------------- the orchestrator
def ensure_kubo(work_dir, *, enabled=None, fetch_fn=None, probe_fn=None, log_fn=log,
                key=None, min_binary_bytes=MIN_BINARY_BYTES, set_env=True):
    """Make a kubo binary available for this miner. Returns (path_or_None, reason).

    NEVER raises and never falls back to an unverified install. Order:
      1. opt-out honoured (no network touched at all);
      2. a kubo already on PATH / already installed here wins -- no download;
      3. unsupported platform -> clean failure;
      4. download the ONE pinned URL, verify size + pinned SHA-512 BEFORE extracting;
      5. extract exactly one member into <work_dir>/kubo/, refuse any escaping member;
      6. probe it once; a binary that will not run is removed, not reported as installed.
    On success `NEURAHASH_IPFS_BIN` is pointed at the install so the publish child (which reads it
    at import time in tools/ipfs_checkpoint.py) uses it without any PATH change."""
    enabled = env_enabled() if enabled is None else bool(enabled)
    if not enabled:
        return None, "auto-install disabled (%s=0)" % AUTO_INSTALL_ENV

    # Resolve the platform key FIRST so the "already installed here" probe looks for the right file
    # name, but check for an existing binary BEFORE rejecting an unpinned platform: a machine that
    # already has kubo on PATH is fine even where we have no build to offer it.
    key = key or platform_key()
    existing = find_kubo(work_dir, key=key)
    if existing:
        return existing, "already available: %s" % existing

    if key is None:
        import platform as _platform
        return None, ("unsupported platform %s/%s -- no pinned kubo build; install kubo manually "
                      "or set PINATA_JWT" % (_platform.system(), _platform.machine()))

    release = RELEASES[key]
    fetch_fn = fetch_fn or _default_fetch
    probe_fn = probe_fn or _default_probe
    tmp_dir = None
    try:
        log_fn("no pinning backend found -- fetching pinned kubo %s (%s, %.1f MB) from %s"
               % (KUBO_VERSION, "%s-%s" % key, release.size / 1e6, release.url))
        data = fetch_fn(release.url)

        ok, detail = verify_archive(data, release)
        if not ok:
            # The download is never written next to the install, and on a mismatch it is dropped
            # here without ever reaching disk. There is deliberately no "install anyway" branch.
            log_fn("REFUSED: %s -- installing nothing (kubo %s from %s)"
                   % (detail, KUBO_VERSION, release.url))
            return None, "archive rejected: %s" % detail
        log_fn("verified: %s" % detail)

        os.makedirs(os.path.abspath(work_dir), exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix=".kubo_dl_", dir=os.path.abspath(work_dir))
        archive_path = os.path.join(tmp_dir, release.archive)
        with open(archive_path, "wb") as f:
            f.write(data)

        binary = extract_binary(archive_path, install_dir(work_dir), release,
                                min_binary_bytes=min_binary_bytes)

        pok, pdetail = probe_fn(binary)
        if not pok:
            try:
                os.remove(binary)
            except OSError:
                pass
            log_fn("installed binary did not run (%s) -- removed" % pdetail)
            return None, "installed binary failed its version probe: %s" % pdetail

        if set_env:
            os.environ[IPFS_BIN_ENV] = binary
        log_fn("kubo %s installed at %s (%s)" % (KUBO_VERSION, binary, pdetail))
        return binary, "auto-installed kubo %s at %s" % (KUBO_VERSION, binary)
    except Exception as e:                      # offline, disk full, bad archive, anything
        log_fn("auto-install failed (%s: %s) -- staying in LOCAL mode" % (type(e).__name__, e))
        return None, "auto-install failed: %s: %s" % (type(e).__name__, e)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    work_dir = argv[0] if argv else os.path.join(os.path.expanduser("~"), ".neurahash")
    path, reason = ensure_kubo(work_dir)
    log(reason)
    return 0 if path else 1


if __name__ == "__main__":
    sys.exit(main())
