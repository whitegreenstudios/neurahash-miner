#!/usr/bin/env python3
"""tools/run_miner.py -- the turnkey, run-forever, ALL-OUTBOUND public miner client for NeuraHash.

Join with ONE command: any GPU (the owner's 4060, a rented pod, or a stranger's machine) runs this
and it loops forever -- fetch the latest base, train a contribution, PUBLISH a small compressed
delta OUTBOUND. Nothing ever listens on an inbound port, so it works behind any NAT.

This is a THIN wrapper over the already-working `tools/diloco_contributor.py contribute` path. It does
NOT reimplement training or the delta codec -- each iteration just shells out:

    python tools/diloco_contributor.py contribute <base> --out <wd>/contrib.pt --steps N \
        --seed <fresh> --device <dev> --name <wallet> --compress-delta \
        [--publish-delta --publish-compressed-delta]   # only when publish infra is configured

Publish infra = ALL THREE of: NEURAHASH_DILOCO_MERGE_URL (the coordinator's merge registry),
NEURAHASH_CONTENT_TOKEN (the registry write token -- sent as the `X-Auth` header on the registry
PUT; without it the store answers HTTP 401), and a pinning backend (PINATA_JWT / PINATA_JWT_FILE,
or a local kubo `ipfs` binary -- which this client will SAFELY auto-install into the work dir when
absent, see tools/install_kubo.py). If ANY is missing the miner runs in LOCAL mode: it still trains and
KEEPS the compressed delta on disk (so a stranger can smoke-test), and prints exactly which one is
missing -- it never crashes for lack of infra.

The base is resolved per iteration: an explicit --base-source (checkpoint path | IPFS CID | tracker
URL) wins; else a local base in the work dir; else a round-0 base is materialized from HuggingFace
via tools/make_base_from_hf.py (the cold-start, all-outbound fallback).

The PROVEN 8GB-safe training recipe is pinned into the child env (bf16 + grad-checkpoint + micro
batches + compressed-delta-only) -- see PROVEN_RECIPE below.
"""
import argparse
import hashlib
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DILOCO = os.path.join(REPO, "tools", "diloco_contributor.py")
MAKE_BASE = os.path.join(REPO, "tools", "make_base_from_hf.py")

# The SIGNED NETWORK MANIFEST (docs/MINER_MANIFEST_DESIGN.md). One signed artifact carries the code
# version, the network's `config` and its `min_client_version`, so code/config/expectations can no
# longer silently disagree (issue #71). Imported defensively: a miner must never fail to start
# because the updater module is unavailable.
try:
    import self_update as _su                          # tools/ on sys.path (python tools/run_miner.py)
except ImportError:                                    # imported as a package
    try:
        from tools import self_update as _su
    except ImportError:
        _su = None

# The SAFE pinned auto-installer for kubo (tools/install_kubo.py): on a bare machine neither Pinata
# nor a kubo binary exists, so the pinning backend -- the last non-credential blocker to a
# zero-config join -- is missing. Imported exactly as defensively as the updater: a miner must never
# fail to start because this module is unavailable, it just stays in LOCAL mode.
try:
    import install_kubo as _ik
except ImportError:
    try:
        from tools import install_kubo as _ik
    except ImportError:
        _ik = None

# Set by ensure_pinning_backend(): the one-line outcome of the auto-install attempt, so --doctor and
# the LOCAL-mode banner can report what actually happened instead of a bare "not found".
KUBO_AUTO_REASON = None

# Set by manifest_sync() when the signed manifest's `min_client_version` is ABOVE this client's
# VERSION: the miner must keep TRAINING but must not PUBLISH. publish_mode() reports it by name so
# it lands in the same LIVE/LOCAL banner line as every other publish-mode reason.
PUBLISH_BLOCK_REASON = None

# The PROVEN 8GB-safe fit (memory hf-piece-streaming-training / RunPod 2026-07-10). These are pushed
# into the child's environment on every iteration; the child (diloco_contributor / make_base) reads
# them. PYTHONIOENCODING pins UTF-8 so the child never trips the Windows cp1252 print trap.
PROVEN_RECIPE = {
    "NEURAHASH_CORPUS": "qwen",
    "NEURAHASH_TRAIN_DTYPE": "bf16",
    "NEURAHASH_GRAD_CHECKPOINT": "1",
    "NEURAHASH_EVAL_MICROBATCH": "4",
    "NEURAHASH_TRAIN_BATCH": "4",
    "NEURAHASH_DELTA_COMPRESS": "1",
    "NEURAHASH_SKIP_DELTA": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONIOENCODING": "utf-8",
}


def log(msg):
    """One ASCII line to stdout (flushed) -- cp1252 console safe."""
    print(f"[run_miner] {msg}", flush=True)


def _pinata_configured():
    """True if a Pinata JWT is reachable (env PINATA_JWT or a PINATA_JWT_FILE that exists). Mirrors
    ipfs_checkpoint._pinata_jwt WITHOUT reading/printing the secret itself."""
    if os.environ.get("PINATA_JWT", "").strip():
        return True
    f = os.environ.get("PINATA_JWT_FILE", "").strip()
    return bool(f and os.path.exists(f))


def _kubo_available(work_dir=None):
    """Path to a usable kubo/ipfs binary (a hosting node that can pin outbound), else None.

    A machine-provided binary on PATH wins; `work_dir` additionally makes a previous SAFE
    auto-install under `<work_dir>/kubo/` count, so a miner that installed kubo once never
    downloads it again. Truthy/falsey either way, so every existing call site is unchanged."""
    if _ik is not None:
        return _ik.find_kubo(work_dir)
    return shutil.which(os.environ.get("NEURAHASH_IPFS_BIN", "ipfs"))


def ensure_pinning_backend(work_dir):
    """Try ONCE per process to make a pinning backend exist, by auto-installing the pinned kubo
    build into `<work_dir>/kubo/` (tools/install_kubo.py: compiled-in URL, published SHA-512
    verified BEFORE extraction, one member extracted, no shell, no PATH change, opt-out via
    NEURAHASH_AUTO_INSTALL_KUBO=0).

    Skipped entirely when a backend already exists -- Pinata configured, or a kubo already on PATH
    or already installed here -- so the 41MB download happens at most once, on a machine that
    genuinely has no way to pin. NEVER raises: on any failure the miner simply stays in LOCAL mode,
    and the reason is recorded for --doctor. Returns the binary path or None."""
    global KUBO_AUTO_REASON
    if _ik is None:
        KUBO_AUTO_REASON = "installer module unavailable"
        return None
    if _pinata_configured():
        return None
    path = _ik.find_kubo(work_dir)
    if path is None:
        path, KUBO_AUTO_REASON = _ik.ensure_kubo(work_dir)   # points IPFS_BIN_ENV at it on success
    elif not os.environ.get(_ik.IPFS_BIN_ENV, "").strip():
        # A kubo installed by an EARLIER run of this client lives in the work dir, not on PATH.
        # Name it explicitly so both this process's checks and the publish child (which reads
        # NEURAHASH_IPFS_BIN at import time) use it -- still without touching PATH.
        os.environ[_ik.IPFS_BIN_ENV] = path
    return path


def publish_mode(work_dir=None):
    """Return (is_live, reason). LIVE needs THREE things: the merge registry URL, the registry WRITE
    TOKEN, and a pinning backend.

    The token is the easy one to miss (reported by a real joiner on issue #71): the registry PUT in
    `diloco_contributor.publish_delta` sends NEURAHASH_CONTENT_TOKEN as the `X-Auth` header, but this
    preflight used to check only the URL + pinning -- so an unset token sailed through as "LIVE" and
    then failed LATE with an opaque HTTP 401. Checking it here turns that into a named, actionable
    LOCAL-mode reason. We deliberately do NOT default the token here, even though `fleet/esh_worker.py`
    defaults its `--token` to the demo value: baking a shared write credential into the turnkey client
    makes every miner write with one secret, so rotating it breaks the whole fleet at once. Requiring
    it explicitly keeps per-joiner tokens possible. (Verified 2026-07-21: the store gates PUT on a
    single `CONTENT_TOKEN` -- `tools/content_store.py:184` -- so an unset token is a hard 401.)

    A `min_client_version` block from the SIGNED network manifest is checked FIRST and is reported
    the same way: this client is too old for what the network accepts, so it trains but publishes
    nothing (docs/MINER_MANIFEST_DESIGN.md sec.3 step 4)."""
    if PUBLISH_BLOCK_REASON:
        return False, PUBLISH_BLOCK_REASON
    merge_url = os.environ.get("NEURAHASH_DILOCO_MERGE_URL", "").strip()
    if not merge_url:
        return False, "NEURAHASH_DILOCO_MERGE_URL not set"
    if not os.environ.get("NEURAHASH_CONTENT_TOKEN", "").strip():
        return False, ("NEURAHASH_CONTENT_TOKEN not set -- the registry PUT sends it as the X-Auth "
                       "header; without it the content store rejects the publish with HTTP 401")
    if _pinata_configured():
        return True, "Pinata pinning"
    if _kubo_available(work_dir):
        return True, "local kubo pinning"
    reason = "no pinning backend (PINATA_JWT / local kubo)"
    if KUBO_AUTO_REASON:
        reason += " -- kubo auto-install: %s" % KUBO_AUTO_REASON
    return False, reason


def _identity(wallet):
    """Resolve the contributor id passed as diloco --name. A key PATH -> its file stem; else the raw
    name. Default: a hostname-derived id. Sanitized to a safe registry-slot token."""
    if not wallet:
        wallet = "miner-" + (socket.gethostname() or "unknown")
    if os.path.exists(wallet):
        wallet = os.path.splitext(os.path.basename(wallet))[0]
    safe = re.sub(r"[^0-9A-Za-z._-]", "-", wallet).strip("-") or "miner"
    return safe


def _signer_address():
    """GAP1 banner-only: if NEURAHASH_MINER_KEY points at an EXISTING key file, return its 0x address so
    the banner can show what the miner signs as. Pass-through only -- this wrapper NEVER signs and NEVER
    creates the key; the child (diloco_contributor.publish_delta / _miner_account) owns signing and
    first-run key creation. Returns None when unset, not-yet-created, or the crypto deps are unavailable
    (a miner must never crash for lack of a banner detail)."""
    key_path = os.environ.get("NEURAHASH_MINER_KEY", "").strip()
    if not key_path or not os.path.exists(key_path):
        return None
    try:
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from neura_l1.signing import account_from_key
        with open(key_path) as f:
            return account_from_key(f.read().strip()).address
    except Exception:
        return None


def _child_env(base_name):
    """os.environ + the PROVEN recipe + NEURAHASH_BASE. Existing secrets/URLs (PINATA_JWT,
    NEURAHASH_DILOCO_MERGE_URL, and the GAP1 wallet key NEURAHASH_MINER_KEY) are inherited untouched via
    os.environ.copy() -- so a key set in this process reaches the contribute child, which signs with it.
    We never read or print those secrets here."""
    env = os.environ.copy()
    env.update(PROVEN_RECIPE)
    env["NEURAHASH_BASE"] = base_name
    return env


def resolve_base(args, work_dir, env):
    """Return (base_ref, description). Priority: --base-source > local base in work dir > DECENTRALIZED
    fetch of the fleet's current base by CID (content-store tracker -> IPFS swarm/gateways) > cold-start
    build from HuggingFace. base_ref is whatever `diloco contribute` should receive as its source (its
    _resolve_ckpt handles a path / bare CID / tracker URL)."""
    if args.base_source:
        return args.base_source, f"base-source (given): {args.base_source}"
    local_base = os.path.join(work_dir, "base.pt")
    if os.path.exists(local_base):
        return local_base, f"local base in work dir: {local_base}"
    # DECENTRALIZED-FIRST (owner directive: download the model over the decentralized network, HF only as a
    # fallback). Before the HF cold-start, pull the fleet's CURRENT base checkpoint by CID -- the
    # content-store tracker -> IPFS swarm / public gateways, CID-VERIFIED. bootstrap_checkpoint NEVER raises
    # and returns None on any miss, so we drop through to HuggingFace only when NO decentralized base is
    # reachable (same fetch a manual `--base-source <CID>` uses, just tracker-auto-discovered).
    store = (os.environ.get("NEURAHASH_CONTENT_URL", "").strip()
             or os.environ.get("NEURAHASH_DILOCO_MERGE_URL", "").strip())
    if store:
        try:
            import ipfs_checkpoint as _ic                    # tools/ on sys.path (python tools/run_miner.py)
        except ImportError:
            from tools import ipfs_checkpoint as _ic         # imported as a package
        got = _ic.bootstrap_checkpoint(local_base, store)
        if got and os.path.exists(local_base):
            return local_base, (f"decentralized base via {got.get('source')} "
                                f"(round {got.get('round')}, cid {str(got.get('cid'))[:12]}..)")
        log("no decentralized base reachable (content-store tracker/CID) -- falling back to HuggingFace")
    log("cold-start fallback: no base found -- building a round-0 base from HuggingFace "
        "(all-outbound) via make_base_from_hf.py; this downloads weights on first run")
    cmd = [sys.executable, MAKE_BASE, local_base, "--device", "cpu", "--base", args.base]
    # inherit stdout/stderr so the (long, ~1.2GB) download shows live progress
    subprocess.run(cmd, env=env, check=True)
    return local_base, f"cold-start base built from HF: {local_base}"


def run_iteration(args, work_dir, name, seed, is_live):
    """One fetch->train->publish cycle. Raises on child failure so the caller can log + continue."""
    env = _child_env(args.base)
    base_ref, base_desc = resolve_base(args, work_dir, env)
    contrib_out = os.path.join(work_dir, "contrib.pt")

    cmd = [sys.executable, DILOCO, "contribute", base_ref,
           "--out", contrib_out, "--steps", str(args.steps), "--lr", str(args.lr),
           "--seed", str(seed), "--device", args.device, "--name", name, "--compress-delta"]
    if is_live:
        cmd += ["--publish-delta", "--publish-compressed-delta"]

    log(f"train: base={base_desc} steps={args.steps} seed={seed} device={args.device} name={name}")
    r = subprocess.run(cmd, env=env, capture_output=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-12:])
        raise RuntimeError(f"contribute exited {r.returncode}; tail:\n{tail}")

    held = re.search(r"held-out\s+([0-9.]+)\s*->\s*([0-9.]+)\s*\(([A-Za-z-]+)", out)
    delta = re.search(r"compressed delta:\s+(\S+)\s+\(([0-9.]+)\s*MB\)", out)
    cidm = re.search(r"published \+ registered trunk delta CID:\s+(\S+)", out)

    if held:
        vb, va, verdict = held.group(1), held.group(2), held.group(3)
        summary = f"held-out {vb} -> {va} ({verdict})"
    else:
        summary = "held-out (unparsed -- see child output)"
        log("WARN: could not parse the held-out line; child stdout tail:")
        for ln in out.strip().splitlines()[-8:]:
            log("  child> " + ln)

    if delta:
        dpath, dmb = delta.group(1), delta.group(2)
        real = ""
        if os.path.exists(dpath):
            real = f" ({os.path.getsize(dpath) / 1e6:.2f} MB on disk)"
        summary += f" | compressed delta {dmb} MB at {dpath}{real}"
    else:
        summary += " | compressed delta (unparsed)"

    if is_live:
        if cidm:
            summary += f" | PUBLISHED CID {cidm.group(1)}"
        else:
            summary += " | PUBLISH attempted (no CID parsed -- check child output)"
        log("iter done: " + summary)
    else:
        log("iter done: " + summary)
        log("publish infra not configured -- %s" % publish_mode(work_dir)[1])
        log("to go live set ALL THREE: NEURAHASH_DILOCO_MERGE_URL (merge registry), "
            "NEURAHASH_CONTENT_TOKEN (registry write token, sent as X-Auth), and a pinning backend "
            "(PINATA_JWT / PINATA_JWT_FILE or a local kubo `ipfs`); delta saved locally")


def apply_manifest_config(cfg):
    """Apply a VERIFIED manifest's `config` as DEFAULTS ONLY; returns ["NAME=value", ...] for what
    this call actually set. Deliberately a SEPARATE function from any hard-coded zero-config
    defaults so the two compose in one direction: explicit env > signed manifest `config` >
    hard-coded fallback. Delegates the allowlist + validation to self_update so there is exactly
    ONE definition of what the network is allowed to set.

    WARNING FOR FUTURE CALLERS: this takes a bare dict and performs NO signature check. The only
    legitimate source of `cfg` is `SyncResult.manifest["config"]` from a manifest that already
    recovered the pinned release key (which is why the startup path calls sync_from_manifest, not
    this). Never wire this to a CLI flag, a config file, or a coordinator response."""
    if _su is None or not cfg:
        return []
    applied, _ignored = _su.apply_manifest_config(cfg)
    return applied


def manifest_sync(argv=None, startup=True):
    """Step 1-4 of docs/MINER_MANIFEST_DESIGN.md sec.3, run BEFORE any env is read: fetch + verify
    the signed manifest across all mirrors, self-update if it names a forward version (this
    re-execs and does not return), apply `config` as defaults, and arm the `min_client_version`
    publish block. Returns the SyncResult (or None if the updater module is unavailable).

    NEVER raises and never blocks for long: unreachable mirrors are a non-event, a bad signature
    means we keep the code and config we already have. That is this client's core promise -- a
    miner must never fail to start for lack of infra."""
    global PUBLISH_BLOCK_REASON
    if _su is None:
        return None
    sync = _su.sync_from_manifest(REPO, argv=argv, startup=startup)
    PUBLISH_BLOCK_REASON = sync.publish_block
    return sync


# ------------------------------------------------------------------------------- --doctor preflight
def _check(name, ok, detail, remedy=""):
    return {"name": name, "ok": bool(ok), "detail": detail, "remedy": remedy}


def _doctor_registry_reachable(url, timeout=4):
    """HTTP HEAD the registry. ANY answer (including 404) proves reachability; only a transport
    failure/timeout is a FAIL. Short timeout so --doctor can never hang."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "neurahash-miner-doctor"})
        with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 (operator url)
            return True, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return True, f"HTTP {e.code} (reachable)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def doctor(work_dir, device="cuda", sync=None, head_fn=None):
    """The preflight table of docs/MINER_MANIFEST_DESIGN.md sec.4. One PASS/FAIL line per check
    plus a one-line remedy on failure. Returns (exit_code, checks) so it is usable from CI and a
    pod bootstrap. All network checks use short timeouts; nothing here can hang."""
    head_fn = head_fn or _doctor_registry_reachable
    checks = []

    # 1) client version vs the signed manifest's min_client_version -------------------------------
    local_v = None
    if _su is not None:
        try:
            local_v = _su.read_local_version(REPO)
        except Exception:
            local_v = None
    block = sync.publish_block if sync is not None else PUBLISH_BLOCK_REASON
    need = (sync.manifest or {}).get("min_client_version") if sync is not None else None
    checks.append(_check(
        "client version", not block,
        f"local v{local_v or '?'}; manifest min_client_version {need or '(none declared)'}",
        "run again to auto-update, or `git pull`"))

    # 2) manifest reachable + signature valid ------------------------------------------------------
    if sync is None:
        checks.append(_check("signed manifest", False, "updater module unavailable",
                             "reinstall the client (`git pull` + pip install -r requirements.txt)"))
    else:
        mirrors = sync.mirrors_summary()
        checks.append(_check(
            "signed manifest", sync.manifest is not None,
            (f"v{sync.manifest_version} from {sync.fetch.source}" if sync.manifest is not None
             else "no mirror served a manifest signed by the pinned release key"),
            f"mirrors tried -- {mirrors}"))

    # 3) signing key present / creatable ----------------------------------------------------------
    key_path = os.environ.get("NEURAHASH_MINER_KEY", "").strip()
    # the zero-config default the client would create on first publish
    default_key = os.path.join(work_dir, "miner_key.hex")
    probe = key_path or default_key
    if key_path and os.path.exists(key_path):
        checks.append(_check("signing key", True, f"present: {key_path}"))
    else:
        parent = os.path.dirname(os.path.abspath(probe)) or "."
        creatable = os.path.isdir(parent) and os.access(parent, os.W_OK)
        checks.append(_check("signing key", creatable,
                             f"not yet created; will be created at {probe}",
                             f"make {parent} writable, or set NEURAHASH_MINER_KEY to a writable path"))

    # 4) registry reachable ------------------------------------------------------------------------
    merge_url = os.environ.get("NEURAHASH_DILOCO_MERGE_URL", "").strip()
    if not merge_url:
        checks.append(_check("registry reachable", False, "NEURAHASH_DILOCO_MERGE_URL not set",
                             "set NEURAHASH_DILOCO_MERGE_URL, or let the signed manifest supply it"))
    else:
        ok, detail = head_fn(merge_url)
        checks.append(_check("registry reachable", ok, f"{merge_url} -- {detail}",
                             f"check network access to {merge_url}"))

    # 5) publish credential / auth path usable -----------------------------------------------------
    signed_put = os.environ.get("NEURAHASH_SIGNED_PUT", "").strip() in ("1", "true", "yes", "on")
    if signed_put:
        checks.append(_check("publish auth", True,
                             "signed-PUT path (per-miner key, no shared secret)"))
    else:
        has_token = bool(os.environ.get("NEURAHASH_CONTENT_TOKEN", "").strip())
        checks.append(_check("publish auth", has_token,
                             "token path: NEURAHASH_CONTENT_TOKEN " +
                             ("set (sent as X-Auth)" if has_token else "NOT set"),
                             "set NEURAHASH_CONTENT_TOKEN (the registry PUT sends it as X-Auth; "
                             "without it the store answers HTTP 401)"))

    # 6) pinning backend ---------------------------------------------------------------------------
    if _pinata_configured():
        checks.append(_check("pinning backend", True, "Pinata JWT configured"))
    else:
        # Reflect REALITY, including a kubo this client auto-installed into the work dir: main()
        # runs ensure_pinning_backend() BEFORE this, and that points NEURAHASH_IPFS_BIN at the
        # local install, so the same no-arg probe sees it. The doctor's job is to answer "can this
        # machine pin?", not "is there one on PATH?".
        kubo = _kubo_available()
        if kubo:
            checks.append(_check("pinning backend", True, "local kubo `ipfs`: %s" % kubo))
        else:
            detail = "neither Pinata nor kubo found"
            if KUBO_AUTO_REASON:
                detail += " -- kubo auto-install: %s" % KUBO_AUTO_REASON
            checks.append(_check("pinning backend", False, detail,
                                 "install kubo, or set PINATA_JWT -- or let this client auto-install "
                                 "a pinned, digest-verified kubo into the work dir (it is on by "
                                 "default; disabled here only by --no-auto-install-kubo / "
                                 "NEURAHASH_AUTO_INSTALL_KUBO=0)"))

    # 7) CUDA / device ------------------------------------------------------------------------------
    try:
        import torch
        if torch.cuda.is_available():
            checks.append(_check("device", True, f"CUDA available: {torch.cuda.get_device_name(0)}"))
        else:
            checks.append(_check("device", True,
                                 f"no CUDA -- requested device {device!r} falls back to CPU "
                                 f"(slow but correct)"))
    except Exception as e:
        checks.append(_check("device", False, f"torch unavailable ({type(e).__name__})",
                             "pip install -r requirements.txt"))

    print("=" * 70, flush=True)
    print("  NeuraHash miner -- preflight doctor", flush=True)
    print("=" * 70, flush=True)
    failed = 0
    for c in checks:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']:<18} {c['detail']}", flush=True)
        if not c["ok"]:
            failed += 1
            if c["remedy"]:
                print(f"        remedy: {c['remedy']}", flush=True)
    print("=" * 70, flush=True)
    print(f"  {len(checks) - failed}/{len(checks)} checks passed", flush=True)
    return (1 if failed else 0), checks


def main():
    ap = argparse.ArgumentParser(
        description="Turnkey run-forever ALL-OUTBOUND NeuraHash miner (thin wrapper over "
                    "tools/diloco_contributor.py contribute).")
    ap.add_argument("--wallet", default=None,
                    help="contributor identity (a name or a key path) passed as diloco --name; "
                         "default: a hostname-derived id")
    ap.add_argument("--base", default="qwen3-0.6b", help="base model key (NEURAHASH_BASE)")
    ap.add_argument("--steps", type=int, default=200, help="inner training steps per contribution")
    ap.add_argument("--lr", type=float, default=3e-4, help="contributor local-SGD learning rate")
    ap.add_argument("--device", default="cuda", help="training device: cuda (default) or cpu")
    ap.add_argument("--interval", type=int, default=30,
                    help="seconds to sleep between iterations (ignored with --once)")
    ap.add_argument("--once", action="store_true", help="run a single iteration then exit (testing)")
    ap.add_argument("--work-dir", default="D:/aiCrypto_work/run_miner",
                    help="scratch dir for the base + contribution + compressed delta")
    ap.add_argument("--base-source", default=None,
                    help="explicit base: a checkpoint path, IPFS CID, or tracker URL "
                         "(wins over any local/cold-start base)")
    ap.add_argument("--no-auto-update", action="store_true",
                    help="disable the signed self-update + network-manifest sync "
                         "(same as NEURAHASH_AUTOUPDATE=0)")
    ap.add_argument("--no-auto-install-kubo", action="store_true",
                    help="disable the pinned, digest-verified kubo auto-install into the work dir "
                         "(same as NEURAHASH_AUTO_INSTALL_KUBO=0)")
    ap.add_argument("--doctor", action="store_true",
                    help="run the preflight checks (PASS/FAIL + a remedy each) and exit non-zero "
                         "if any fail; makes no changes and starts no training")
    args = ap.parse_args()

    if args.no_auto_update:
        os.environ["NEURAHASH_AUTOUPDATE"] = "0"
    if args.no_auto_install_kubo:
        os.environ["NEURAHASH_AUTO_INSTALL_KUBO"] = "0"

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # STEP 1-4 (docs/MINER_MANIFEST_DESIGN.md sec.3), BEFORE any env is read: fetch + verify the
    # signed manifest on every mirror, self-update onto the signed commit if it is forward (that
    # re-execs), apply `config` as DEFAULTS, arm the min_client_version publish block.
    sync = manifest_sync(argv=sys.argv, startup=True)
    defaulted = list(sync.config_applied) if sync is not None else []
    # MERGE POINT: any hard-coded zero-config fallback (apply_zero_config_defaults) belongs HERE --
    # AFTER the signed manifest, so the precedence chain is explicit env > signed manifest config >
    # hard-coded fallback. Guarded so this file works with or without that function present.
    if "apply_zero_config_defaults" in globals():
        defaulted += globals()["apply_zero_config_defaults"](work_dir)

    # ZERO-CONFIG PINNING: AFTER the signed manifest has had its say (it may supply publish config)
    # and BEFORE anything reads the publish mode, so --doctor and the banner both see the result of
    # this attempt rather than a stale "not found". No-ops when a backend already exists.
    ensure_pinning_backend(work_dir)

    if args.doctor:
        code, _checks = doctor(work_dir, device=args.device, sync=sync)
        return code

    name = _identity(args.wallet)
    is_live, reason = publish_mode(work_dir)
    base_desc = args.base_source or f"(local/cold-start in {work_dir})"

    signer = _signer_address()
    key_set = bool(os.environ.get("NEURAHASH_MINER_KEY", "").strip())
    if signer:
        identity_line = f"  identity     : {name} (signed as {signer})"
    elif key_set:
        identity_line = f"  identity     : {name} (signing enabled: NEURAHASH_MINER_KEY set; wallet created on first publish)"
    else:
        identity_line = f"  identity     : {name} (UNSIGNED -- set NEURAHASH_MINER_KEY to sign contributions)"

    print("=" * 70, flush=True)
    print("  NeuraHash miner -- turnkey, run-forever, ALL-OUTBOUND", flush=True)
    print(identity_line, flush=True)
    print(f"  device       : {args.device}   base: {args.base}   steps/iter: {args.steps}", flush=True)
    print(f"  base source  : {base_desc}", flush=True)
    print(f"  publish mode : {'LIVE (' + reason + ')' if is_live else 'LOCAL (' + reason + ')'}", flush=True)
    print(f"  work dir     : {work_dir}", flush=True)
    print(f"  mode         : {'single iteration (--once)' if args.once else 'run forever'}", flush=True)
    if sync is not None:
        if sync.manifest is not None:
            print(f"  manifest     : signed v{sync.manifest_version} from {sync.fetch.source}",
                  flush=True)
        else:
            print(f"  manifest     : NONE verified -- {sync.mirrors_summary()}", flush=True)
    for line in defaulted:
        print(f"  defaulted    : {line}", flush=True)
    print("=" * 70, flush=True)
    if not is_live:
        log("LOCAL mode: deltas are trained + kept on disk, not published. "
            "Set NEURAHASH_DILOCO_MERGE_URL + PINATA_JWT (or run a local kubo daemon) to go LIVE.")

    # deterministic, per-wallet, per-iteration seed (NOT time-based): distinct base per wallet so
    # parallel miners are not the same computation, incremented each iteration so they keep differing.
    base_seed = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 100000
    i = 0
    while True:
        i += 1
        seed = base_seed + i
        log(f"===== iteration {i} (seed {seed}) =====")
        try:
            run_iteration(args, work_dir, name, seed, is_live)
        except KeyboardInterrupt:
            log("interrupted -- exiting")
            return 0
        except Exception as e:  # a miner must be resilient: log and keep going (unless --once)
            log(f"iteration {i} FAILED: {e}")
            if args.once:
                return 1
        if args.once:
            log("--once: single iteration complete, exiting")
            return 0
        log(f"sleeping {args.interval}s before next iteration")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log("interrupted -- exiting")
            return 0


if __name__ == "__main__":
    sys.exit(main())
