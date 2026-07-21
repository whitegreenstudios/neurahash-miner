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
or a local kubo `ipfs` binary). If ANY is missing the miner runs in LOCAL mode: it still trains and
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DILOCO = os.path.join(REPO, "tools", "diloco_contributor.py")
MAKE_BASE = os.path.join(REPO, "tools", "make_base_from_hf.py")

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


def _kubo_available():
    """True if a local kubo/ipfs binary is on PATH (a hosting node that can pin outbound)."""
    return shutil.which(os.environ.get("NEURAHASH_IPFS_BIN", "ipfs")) is not None


def publish_mode():
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
    single `CONTENT_TOKEN` -- `tools/content_store.py:184` -- so an unset token is a hard 401.)"""
    merge_url = os.environ.get("NEURAHASH_DILOCO_MERGE_URL", "").strip()
    if not merge_url:
        return False, "NEURAHASH_DILOCO_MERGE_URL not set"
    if not os.environ.get("NEURAHASH_CONTENT_TOKEN", "").strip():
        return False, ("NEURAHASH_CONTENT_TOKEN not set -- the registry PUT sends it as the X-Auth "
                       "header; without it the content store rejects the publish with HTTP 401")
    if _pinata_configured():
        return True, "Pinata pinning"
    if _kubo_available():
        return True, "local kubo pinning"
    return False, "no pinning backend (PINATA_JWT / local kubo)"


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
        log("publish infra not configured -- %s" % publish_mode()[1])
        log("to go live set ALL THREE: NEURAHASH_DILOCO_MERGE_URL (merge registry), "
            "NEURAHASH_CONTENT_TOKEN (registry write token, sent as X-Auth), and a pinning backend "
            "(PINATA_JWT / PINATA_JWT_FILE or a local kubo `ipfs`); delta saved locally")


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
    args = ap.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    name = _identity(args.wallet)
    is_live, reason = publish_mode()
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
