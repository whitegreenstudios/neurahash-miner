"""
run_miner_client.py — TURNKEY miner launcher for the DYNAMIC sharded training pool.

One command turns any machine with Python into a miner on a NeuraHash sharded pool
(`sharded_pool_node.py`). It:

  * picks the heaviest device available (CUDA GPU if present, else CPU),
  * (optionally) clones the repo + installs the minimal deps it needs (numpy + torch),
  * connects to the coordinator and trains its assigned expert shard, and
  * AUTO-RECONNECTS with capped exponential backoff when the tunnel/coordinator drops,
    keeping the SAME miner address so the pool re-partitions you back in seamlessly.

The actual training/verification loop lives in `sharded_pool_node.run_worker` — this file
is a thin, dependency-light supervisor around it so a newcomer never has to remember the
raw `python sharded_pool_node.py --worker --host … --port … --name …` invocation, and so a
flaky free-tunnel hiccup doesn't end the mining session.

  Quick start (deps already installed, repo already cloned):
    python run_miner_client.py --host <COORD_HOST> --port <COORD_PORT> --name 0xYOURADDR

  Cold start on a fresh box (let it fetch the code + deps for you):
    python run_miner_client.py --bootstrap --host <H> --port <P> --name 0xYOURADDR

  See MINING.md for the full walkthrough, the Colab/Kaggle join cell, and how an operator
  stands up a STABLE coordinator (docker/systemd + a persistent tunnel/port-forward).

HONESTY: this launcher only makes JOINING turnkey. It does NOT add economic security — a
faker is still merely rejected/unpaid by the pool today, not bonded/slashed (that wiring is
a separate, open blocker; see MEMORY + SECURITY.md). Do not treat this as a value-bearing
client.
"""

import argparse
import os
import random
import socket
import subprocess
import sys
import time

REPO_URL = "https://github.com/whitegreenstudios/neurahash.git"
DEFAULT_PORT = 7000

# A worker "session" that lasted at least this long counts as a REAL mining session for the failover
# preference (the door is remembered + tried first on reconnect). A shorter one — the link dropped
# during/right after the handshake, typical of a door that is listening but whose coordinator is dying —
# is treated as a failure so a broken door feeds the preferred-door decay instead of staying preferred.
_SESSION_MIN_SECONDS = 5.0


# --------------------------------------------------------------------------- bootstrap
def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def ensure_repo(target_dir):
    """If we're not already inside the repo (sharded_pool_node.py missing), clone it next
    to us and chdir in. Returns the directory we end up running from. No-op when the file
    is already importable from cwd."""
    if os.path.isfile(os.path.join(os.getcwd(), "sharded_pool_node.py")):
        return os.getcwd()
    dest = os.path.abspath(target_dir)
    if not os.path.isdir(os.path.join(dest, ".git")):
        print(f"[bootstrap] cloning {REPO_URL} -> {dest}", flush=True)
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, dest])
    os.chdir(dest)
    return dest


def _corpus_target_dir():
    """Where corpus files live for THIS run — the same resolution corpus_torch.corpus_sha() uses so a
    sync lands exactly where the handshake will read: NEURAHASH_CORPUS_DIR if set, else
    corpus_data/ at the repo root (the dir we run from after ensure_repo)."""
    env_dir = (os.environ.get("NEURAHASH_CORPUS_DIR", "") or "").strip()
    if env_dir:
        return env_dir
    return os.path.join(os.getcwd(), "corpus_data")


def _log_corpus_sha(prefix="[corpus-sync]"):
    """Print the local corpus content-hash so a user can eyeball-match it against the coordinator's
    (a mismatch is exactly what the pool's hello handshake rejects). Best-effort: never fatal."""
    try:
        from neurahash_torch.corpus_torch import corpus_sha
        sha = corpus_sha()
        mode = os.environ.get("NEURAHASH_CORPUS", "<toy>")
        print(f"{prefix} local corpus_sha = {sha}  (NEURAHASH_CORPUS={mode}) — this must match the "
              f"coordinator's or the pool will reject your work.", flush=True)
        return sha
    except Exception as e:                                   # pragma: no cover - defensive only
        print(f"{prefix} could not compute corpus_sha ({e!r})", flush=True)
        return None


def maybe_sync_corpus(force=False):
    """Fetch the corpus BY HASH from the content store so this joiner trains on EXACTLY the
    coordinator's bytes (issue #86 — kills the autocrlf byte-drift that makes a same-commit checkout
    fail the corpus handshake). OPT-IN and non-blocking:

      * runs only when NEURAHASH_CONTENT_STORE is set (env unset -> today's behavior, a no-op) OR when
        the caller passes force=True (the --sync-corpus flag),
      * on ANY store failure prints a LOUD warning and returns False so the caller falls back to the
        checkout's bytes — a store outage must never stop a miner from joining,
      * logs the resulting corpus_sha either way so the user can compare it to the coordinator's.

    Returns True iff a sync ran to completion (whether or not any file changed)."""
    from neurahash import corpus_sync
    store = corpus_sync.store_url_from_env()
    if not store:
        if force:
            print("[corpus-sync] --sync-corpus given but NEURAHASH_CONTENT_STORE is not set; nothing "
                  "to sync from. Set it to e.g. http://47.84.93.96:8710 (the anchor store).",
                  flush=True)
            _log_corpus_sha()
        return False
    target = _corpus_target_dir()
    print(f"[corpus-sync] syncing corpus_data/ from {store} -> {target} (fetch-by-hash, byte-exact)…",
          flush=True)
    try:
        res = corpus_sync.sync_corpus(store, target)
    except corpus_sync.HashMismatch as e:
        # a corrupt/malicious store served the wrong bytes: do NOT trust them. Fail loud, fall back.
        print("!" * 72, flush=True)
        print(f"[corpus-sync] SECURITY: {e}", flush=True)
        print("[corpus-sync] The content store returned bytes that did not match the promised hash. "
              "Ignoring the store and using the git checkout's bytes. If the coordinator rejects your "
              "corpus_sha, sync from a trusted store.", flush=True)
        print("!" * 72, flush=True)
        _log_corpus_sha()
        return False
    except corpus_sync.CorpusSyncError as e:
        print("!" * 72, flush=True)
        print(f"[corpus-sync] WARNING: could not sync from the content store: {e}", flush=True)
        print("[corpus-sync] Falling back to the git checkout's corpus bytes. On an autocrlf Windows "
              "box these may differ from the coordinator's and get your work rejected — retry with a "
              "reachable NEURAHASH_CONTENT_STORE, or hand-match corpus_data/ to the coordinator.",
              flush=True)
        print("!" * 72, flush=True)
        _log_corpus_sha()
        return False
    if res.changed:
        print(f"[corpus-sync] updated {res.updated} (skipped {len(res.skipped)} already byte-exact).",
              flush=True)
    else:
        print(f"[corpus-sync] corpus already byte-exact with the store ({len(res.skipped)} files).",
              flush=True)
    _log_corpus_sha()
    return True


def ensure_deps():
    """Install the runtime deps a sharded-pool worker needs to IMPORT and run: numpy + torch
    (compute) PLUS the signing/crypto stack the worker pulls in transitively via
    neura_l1.signing at module load (eth-account, web3, cryptography, dilithium-py, segno).
    Earlier this installed only numpy+torch — wrong: the very first import of sharded_pool_node
    -> neura_l1.signing fails with `No module named 'eth_account'` in a fresh venv. We still skip
    the heavy SERVING stack (transformers/accelerate/peft/flask) — a miner doesn't need it. torch
    is left untouched if already present (Colab/Kaggle ship it) so we don't fight the host CUDA;
    for a specific CUDA build, `pip install torch --index-url …` yourself first, then --bootstrap."""
    heavy = [m for m in ("numpy", "torch") if not _have(m)]
    # import-name -> pinned pip spec for the crypto/signing deps imported at worker load time
    crypto = {"eth_account": "eth-account==0.13.7", "web3": "web3==7.16.0",
              "cryptography": "cryptography>=42", "dilithium_py": "dilithium-py==1.4.0",
              "segno": "segno==1.6.6"}
    need = heavy + [spec for mod, spec in crypto.items() if not _have(mod)]
    if not need:
        return
    print(f"[bootstrap] pip installing {need} (this can take a few minutes for torch)…", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])


# --------------------------------------------------------------------------- device pick
def pick_device(requested=None):
    """auto -> cuda if a GPU is visible, else cpu. Honors an explicit --device."""
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def device_banner(device):
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{torch.cuda.get_device_name(0)} ({p.total_memory / 1e9:.0f} GB)"
    except Exception:
        pass
    return "CPU"


# --------------------------------------------------------------------------- reconnect loop
def _preflight(host, port, timeout=5.0):
    """Quick TCP reachability probe so a wrong host/port fails loudly with a hint instead of
    looping silently. Returns True if the coordinator port accepts a connection right now."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def doctor(host, port):
    """PREFLIGHT CHECKLIST: diagnose the common join failures BEFORE the reconnect loop, print a
    pass/fail line + a one-line fix hint per failure, and return True iff everything is healthy. The
    caller exits nonzero on a False so a broken setup fails loudly instead of looping silently. Checks:
      (a) torch import + version + CUDA availability + device name (catches the #1 trap: launched with
          the WRONG Python where torch is missing or CPU-only),
      (b) TCP reachability to host:port (reuses _preflight),
      (c) TLS pin well-formedness when NEURAHASH_TLS_PIN is set (via tls.normalize_fingerprint),
      (d) the LOCAL corpus content-hash (printed so it can be compared to the coordinator's)."""
    ok_all = True

    def _line(ok, label, detail, hint=""):
        nonlocal ok_all
        ok_all = ok_all and ok
        mark = "PASS" if ok else "FAIL"
        print(f" [{mark}] {label}: {detail}" + (f"\n        fix: {hint}" if (hint and not ok) else ""),
              flush=True)

    print("=" * 72)
    print(" NeuraHash miner DOCTOR — preflight checklist")
    print("=" * 72, flush=True)

    # (a) torch import + CUDA
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        dev = torch.cuda.get_device_name(0) if cuda else "CPU"
        _line(True, "torch", f"v{torch.__version__} | cuda_available={cuda} | device={dev}")
        if not cuda:
            print("        note: CUDA not available — you will mine on CPU (slow, but valid). If you "
                  "have a GPU, you are likely on the wrong Python/venv or a CPU-only torch build.",
                  flush=True)
    except Exception as e:
        _line(False, "torch", f"import failed: {e!r}",
              "install torch in THIS Python: `python -m pip install torch` (run "
              "`python -c \"import sys; print(sys.executable)\"` to confirm which interpreter this is)")

    # (b) TCP reachability
    if host:
        reachable = _preflight(host, port)
        _line(reachable, "coordinator TCP", f"{host}:{port} {'reachable' if reachable else 'NOT reachable'}",
              f"check the coordinator is up and {host}:{port} is correct + not firewalled "
              f"(or pass --connect <connect.json url> to auto-discover it)")
    else:
        _line(False, "coordinator TCP", "no --host given",
              "pass --host <coordinator> (or --connect <url> to auto-discover)")

    # (c) TLS pin well-formedness (only when set)
    pin = os.environ.get("NEURAHASH_TLS_PIN")
    if pin:
        try:
            from neurahash import tls
            norm = tls.normalize_fingerprint(pin)
            _line(True, "TLS pin", f"well-formed ({norm[:16]}…)")
        except Exception as e:
            _line(False, "TLS pin", f"NEURAHASH_TLS_PIN is malformed: {e!r}",
                  "set NEURAHASH_TLS_PIN to the coordinator's 64-hex sha256 cert fingerprint "
                  "(the coordinator prints it on startup)")
    else:
        print(" [skip] TLS pin: NEURAHASH_TLS_PIN not set (HMAC-only — fine on loopback/trusted nets)",
              flush=True)

    # (d) local corpus content-hash (printed for comparison with the coordinator's)
    try:
        from neurahash_torch.corpus_torch import corpus_sha
        sha = corpus_sha()
        _line(True, "corpus sha", f"{sha} (NEURAHASH_CORPUS={os.environ.get('NEURAHASH_CORPUS', '<toy>')})")
        print("        compare this to the coordinator's corpus_sha; a mismatch means the pool will "
              "reject your work (sync corpus_data/).", flush=True)
    except Exception as e:
        _line(False, "corpus sha", f"could not compute: {e!r}",
              "ensure you launched from inside the repo so neurahash_torch is importable")

    print("=" * 72)
    print(f" DOCTOR: {'all checks passed — ready to mine' if ok_all else 'one or more checks FAILED (see fixes above)'}")
    print("=" * 72, flush=True)
    return ok_all


def _run_worker_on(ep, global_pin, name, honest, psk, device, signer, run_worker):
    """Run one worker session against ONE endpoint `ep`, enforcing ep's TLS pin (its own per-endpoint
    pin if it carries one, else the global pin). run_worker reads the pin from NEURAHASH_TLS_PIN via
    tls.resolve_client_pin(), so we set that env var for the duration of THIS call and restore it after —
    keeping run_worker itself single-endpoint and byte-unchanged. Returns nothing; raises on disconnect."""
    from neurahash import endpoints as _ep
    pin = _ep.effective_pin(ep, global_pin)
    prev = os.environ.get("NEURAHASH_TLS_PIN")
    if pin:
        os.environ["NEURAHASH_TLS_PIN"] = pin
    elif "NEURAHASH_TLS_PIN" in os.environ:
        # this endpoint is explicitly plaintext (no pin, no global pin) — make sure a stale env pin from
        # a previous endpoint doesn't leak in and force TLS the coordinator isn't speaking.
        del os.environ["NEURAHASH_TLS_PIN"]
    try:
        run_worker(ep.host, ep.port, name, honest=honest, psk=psk, device=device, signer=signer)
    finally:
        if prev is None:
            os.environ.pop("NEURAHASH_TLS_PIN", None)
        else:
            os.environ["NEURAHASH_TLS_PIN"] = prev


def supervise(host, port=None, name=None, honest=True, device=None, psk=None,
              max_retries=None, base_backoff=2.0, max_backoff=60.0, signer=None, global_pin=None):
    """Run the sharded-pool worker, restarting it on disconnect with capped jittered backoff, ROTATING
    across a list of coordinator "doors" so a dead door fails the miner over to the next one. Keeps the
    same `name` (miner address) across reconnects + doors so the coordinator slots you back into the
    partition. `signer` (optional) is an external/hardware signer passed to the worker so it signs
    without a local key. Returns the number of disconnect cycles survived.

    `host` is EITHER (back-compat, single door) a host STRING used with `port` — byte-identical to the
    original single-endpoint supervisor — OR a neurahash.endpoints.FailoverRotator carrying the whole
    door list (then `port` is ignored). `global_pin` is the NEURAHASH_TLS_PIN fallback for door entries
    that carry no per-endpoint #pin.

    FAILOVER POLICY (see neurahash.endpoints.FailoverRotator):
      * each reconnect tries the doors in rotator.order() — the last door that gave a REAL session first
        (a takeover means that door is the one still up), then the operator's declared list order;
      * a door that REFUSES the connection, or gives a session that never really started, is ADVANCED
        past to the next door in the same pass (fast, no backoff between doors) — find a live door quick;
      * a door that gave a real session (admitted, then the link dropped) is remembered as preferred and
        we back off + jitter before reconnecting (a healthy long session resets the backoff);
      * a preferred door that keeps failing eventually DECAYS (endpoints.DEFAULT_PREF_DECAY) so a retired
        standby stops jumping the queue and the list-order primary is tried first again.

    Single-door (a bare host string, or a one-entry rotator) collapses this to exactly today's
    behaviour: order() is always [that one door], no skipping, same preflight/backoff cadence + return
    value (cycles)."""
    # imported lazily: only valid once ensure_repo/ensure_deps have run.
    from neurahash.worker_core import run_worker   # (miner-extraction Step 2) run_worker lives in worker_core
    from neurahash import endpoints as _ep

    # Back-compat: a host STRING (+ port) is the single-door path; a FailoverRotator carries the list.
    if isinstance(host, _ep.FailoverRotator):
        rotator = host
    else:
        rotator = _ep.FailoverRotator([_ep.Endpoint(host, int(port), None)])
        if global_pin is None:
            # (TLS regression fix) Legacy single-door callers relied on run_worker reading
            # NEURAHASH_TLS_PIN from the env itself — but _run_worker_on now OWNS that env var
            # per-endpoint and DELETES it when no pin resolves, which silently stripped an
            # operator-set pin and downgraded the dial to PLAINTEXT (live incident: the 5090
            # worker was RST'd by the TLS coordinator on every connect while legacy clients
            # admitted fine). Resolve the env pin here so a bare-host call pins exactly as it
            # did before the multi-door refactor.
            from neurahash import tls as _tls
            global_pin = _tls.resolve_client_pin()

    multi = len(rotator.endpoints) > 1
    cycles, backoff = 0, base_backoff
    while True:
        order = rotator.order()
        for idx, ep in enumerate(order):
            if not _preflight(ep.host, ep.port):
                # connect-refused: not a session. ADVANCE to the next door in THIS pass (no backoff
                # between doors) so a dead door fails us over fast; only once every door in the pass has
                # been tried do we back off + retry the whole list.
                rotator.note_failure(ep)
                cycles += 1
                if multi:
                    print(f"[miner {name}] door {ep} not reachable "
                          f"(door {idx + 1}/{len(order)}); trying next…", flush=True)
                continue

            print(f"[miner {name}] connecting to {ep.host}:{ep.port} | device {device} "
                  f"| honest={honest}" + (f"  [door {idx + 1}/{len(order)}]" if multi else ""),
                  flush=True)
            t0 = time.time()
            try:
                # run_worker blocks until the coordinator sends 'done' or the link drops.
                _run_worker_on(ep, global_pin, name, honest, psk, device, signer, run_worker)
                lived = time.time() - t0
                ended = f"session on {ep.host}:{ep.port} ended after {lived:.0f}s"
            except KeyboardInterrupt:
                raise
            except Exception as e:             # ConnectionError / OSError / transient errors
                lived = time.time() - t0
                ended = f"disconnected from {ep.host}:{ep.port} after {lived:.0f}s: {e!r}"
            cycles += 1
            print(f"[miner {name}] {ended}.", flush=True)
            if lived >= _SESSION_MIN_SECONDS:
                # A REAL session: this door gave us work -> remember it + try it FIRST on reconnect. Break
                # out to the backoff and reconnect (the pool re-partitions us back in on the same door).
                rotator.note_session(ep)
                if lived > max_backoff:        # a long, healthy session -> reset backoff (as before)
                    backoff = base_backoff
                break
            # A trivially-short "session" (link dropped during/right after the handshake — a door that is
            # listening but whose coordinator is dying) is NOT a real session: treat it as a failure that
            # feeds the preferred-door decay, and ADVANCE to the next door in this pass like a refusal.
            rotator.note_failure(ep)
            if multi:
                print(f"[miner {name}] (door {idx + 1}/{len(order)} gave no real session; trying next…)",
                      flush=True)
            continue
        else:
            # for-loop completed WITHOUT break: no door gave a real session this pass.
            if multi:
                print(f"[miner {name}] no door gave a session this pass.", flush=True)

        if max_retries is not None and cycles >= max_retries:
            print(f"[miner {name}] reached --max-retries={max_retries}; stopping.", flush=True)
            return cycles
        sleep = min(max_backoff, backoff) * (0.5 + random.random())   # jitter to avoid herd
        print(f"[miner {name}] reconnecting in {sleep:.1f}s … (Ctrl-C to quit)", flush=True)
        time.sleep(sleep)
        backoff = min(max_backoff, backoff * 2)     # grow toward the cap (reset above after a long session)


# --------------------------------------------------------------------------- cli
def build_parser():
    ap = argparse.ArgumentParser(
        description="Turnkey miner launcher for a NeuraHash dynamic sharded training pool.")
    ap.add_argument("--host", default=None,
                    help="coordinator host (e.g. 6.tcp.ngrok.io), OR a comma-separated FAILOVER list of "
                         "doors 'host:port,host:port,…' (a bare host uses --port). The miner tries them "
                         "in order, remembers the last door that gave a mining session and tries it first "
                         "on reconnect, and rolls to the next when one dies — so it survives any single "
                         "coordinator/door dropping. A future standby on its own cert can carry its own "
                         "TLS pin: 'host:port#sha256:<64hex>' (unpinned entries use NEURAHASH_TLS_PIN). "
                         "Env NEURAHASH_COORDS is the same list (this flag wins). A single bare host "
                         "behaves exactly as before.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="coordinator port (default port for bare hosts in the --host list)")
    ap.add_argument("--connect", default=None, metavar="URL",
                    help="AUTO-DISCOVER the coordinator from a published connect.json (e.g. "
                         "https://neoo.com/pool/connect.json) — no need to know the host/port. "
                         "Overrides --host/--port when the pool is online.")
    ap.add_argument("--name", default=None,
                    help="your miner address/id (default: a stable per-machine 0x… id)")
    ap.add_argument("--wallet", default=None, metavar="PATH",
                    help="path to a keystore (see `python -m neurahash.wallet new`). The miner is PAID "
                         "to THIS wallet's address: rewards accrue to it in the signed persisted ledger "
                         "and survive restart. Takes precedence over --name.")
    ap.add_argument("--wallet-password", default=None,
                    help="password for an ENCRYPTED --wallet keystore (omit for a plaintext keystore)")
    ap.add_argument("--signer-cmd", default=None, metavar="CMD",
                    help="HARDWARE-WALLET / external signer: a command that signs for the wallet "
                         "address so the private key never lives on this machine (e.g. "
                         "'python -m neurahash.wallet signer --keystore device.json'). REQUIRED for a "
                         "watch-only --wallet; with a keyed --wallet it overrides the local key.")
    ap.add_argument("--solo", action="store_true",
                    help="SOLO mining: spin up a local single-node coordinator on this machine and "
                         "mine into it, so the FULL block reward credits YOUR wallet (no pool fee). "
                         "Implies a local --host/--port unless you override them.")
    ap.add_argument("--experts", type=int, default=8,
                    help="(--solo only) initial expert count for the local coordinator (default 8)")
    ap.add_argument("--block-time", type=float, default=12.0,
                    help="(--solo only) seconds per block for the local coordinator (default 12)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu (default auto)")
    ap.add_argument("--psk", default=None,
                    help="pre-shared HMAC key (bytes); omit to use the pool default")
    ap.add_argument("--bootstrap", action="store_true",
                    help="clone the repo if needed and pip-install numpy+torch before joining; also "
                         "syncs corpus_data/ by hash when NEURAHASH_CONTENT_STORE is set (see "
                         "--sync-corpus)")
    ap.add_argument("--repo-dir", default="neurahash",
                    help="where to clone when --bootstrap is set and we're outside the repo")
    ap.add_argument("--sync-corpus", action="store_true",
                    help="fetch corpus_data/ BY CONTENT HASH from the content store in "
                         "NEURAHASH_CONTENT_STORE (e.g. http://47.84.93.96:8710) before joining, so "
                         "your corpus is byte-exact with the coordinator's (kills the autocrlf "
                         "line-ending drift that makes a same-commit checkout fail the pool's corpus "
                         "handshake). Requires NEURAHASH_CONTENT_STORE to be set; a store outage is a "
                         "loud warning + fallback to the checkout's bytes, never a hard stop. "
                         "--bootstrap runs this automatically when the env var is set.")
    ap.add_argument("--max-retries", type=int, default=None,
                    help="stop after this many disconnect cycles (default: retry forever)")
    ap.add_argument("--base-backoff", type=float, default=2.0,
                    help="initial reconnect backoff seconds (default 2)")
    ap.add_argument("--max-backoff", type=float, default=60.0,
                    help="cap on reconnect backoff seconds (default 60)")
    ap.add_argument("--cheat", action="store_true",
                    help="(testing only) submit fabricated work to exercise rejection")
    ap.add_argument("--doctor", action="store_true",
                    help="run a preflight checklist (torch/CUDA, coordinator reachability, TLS pin, "
                         "local corpus hash) and exit; nonzero exit on any failure")
    ap.add_argument("--no-auto-update", action="store_true",
                    help="disable the SIGNED, fail-closed auto-updater (default: ON; same as "
                         "NEURAHASH_AUTOUPDATE=0). The updater only ever checks out code cryptographically "
                         "SIGNED by the project release key, never downgrades, and on ANY failure stays on "
                         "the current version -- it never runs unsigned code. See tools/self_update.py + "
                         "SIGNING.md.")
    return ap


def resolve_connect(url, fallback_host, fallback_port):
    """Fetch a published connect.json and return the live coordinator (host, port). Falls back to
    (fallback_host, fallback_port) if the feed is unreachable, reports offline, or is stale (>5 min
    old) — so an offline pool never silently sends you to a dead address."""
    import json as _json
    import time as _time
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            c = _json.loads(r.read().decode())
    except Exception as e:                                   # pragma: no cover - network path
        print(f" [connect] could not fetch {url} ({e}); falling back to --host/--port", flush=True)
        return fallback_host, fallback_port
    age = _time.time() - int(c.get("updated_s", 0) or 0)
    if not (c.get("online") and c.get("host")) or age > 300:
        print(f" [connect] pool reports OFFLINE (online={c.get('online')}, {int(age)}s old) — "
              f"using --host/--port", flush=True)
        return fallback_host, fallback_port
    extra = "  · PSK required (operator must share NEURAHASH_PSK)" if c.get("psk_required") else ""
    print(f" [connect] discovered live coordinator {c['host']}:{c.get('port')} "
          f"({int(age)}s ago){extra}", flush=True)
    return c["host"], int(c.get("port", fallback_port))


def default_name():
    """A stable, non-colliding default miner id derived from the machine hostname, so two
    laptops don't both show up as the same address and get rejected as a duplicate."""
    host = socket.gethostname().lower().replace(" ", "-")[:16] or "miner"
    return f"0x{host}-{os.getpid() % 1000:03d}"


def _seed_worker_key(address, keyhex, key_dir, wallet_path):
    """Pre-seed .neurahash_keys/<addr>.key with the wallet's private key so the existing worker
    identity path (sharded_pool_node.load_or_create_key) signs as the wallet. Idempotent: a matching
    existing key is left untouched; a MISMATCH is a loud error (don't silently overwrite a different
    identity)."""
    os.makedirs(key_dir, exist_ok=True)
    path = os.path.join(key_dir, f"{str(address).replace(os.sep, '_')}.key")
    if os.path.exists(path):
        with open(path) as f:
            existing = f.read().strip().lower().removeprefix("0x")
        if existing != keyhex.lower().removeprefix("0x"):
            raise ValueError(
                f"{path} already holds a DIFFERENT key than --wallet {wallet_path}; refusing to "
                f"overwrite. Remove that file or pick a wallet whose address is {address}.")
    else:
        with open(path, "w") as f:
            f.write(keyhex)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def _verified_subprocess_signer(signer_cmd, address):
    """Build a SubprocessExternalSigner for `signer_cmd` and PROVE it controls `address` with a probe
    signature that must recover to it — so a misconfigured/wrong device is caught at startup, not when
    the first reward fails to land. Returns the signer."""
    from neura_l1.signing import recover_bytes
    from neurahash.wallet import SubprocessExternalSigner
    s = SubprocessExternalSigner(signer_cmd, address=address)
    probe = f"neurahash-signer-probe:{address}".encode()
    if recover_bytes(probe, s.sign_bytes(probe)).lower() != str(address).lower():
        raise ValueError(
            f"--signer-cmd does not control {address} (its probe signature did not recover to it). "
            f"Point it at the device/keystore that holds {address}'s key.")
    return s


def resolve_wallet_identity(wallet_path, password, signer_cmd=None, key_dir=".neurahash_keys"):
    """Resolve the miner's PAYOUT identity (the address the coordinator credits) AND how it signs,
    from a --wallet keystore. Returns (address, signer):
      * KEYED keystore -> pre-seed .neurahash_keys/<addr>.key with the wallet key and return
        (address, None) so the existing local-signing path is used. If --signer-cmd is ALSO given, the
        external device signs instead and (address, signer) is returned.
      * WATCH-ONLY / hardware keystore (no key on disk) -> REQUIRE --signer-cmd; return
        (address, signer). A clear error if no external signer is configured (you can't mine a
        keyless wallet without one).
    Validates the address shape (rejects a malformed/corrupt keystore)."""
    from neurahash.wallet import Wallet, normalize_address
    w = Wallet.load(wallet_path, password=password)
    if not w.is_valid():
        raise ValueError(f"{wallet_path}: keystore failed self-check (corrupt key)")
    address = normalize_address(w.address)
    if w.has_key:
        if signer_cmd:
            # a device is configured: sign via it and DON'T drop a plaintext copy of the key into
            # .neurahash_keys/ — the wallet keystore stays the only at-rest copy on this machine.
            return address, _verified_subprocess_signer(signer_cmd, address)
        _seed_worker_key(address, w.account.key.hex(), key_dir, wallet_path)
        return address, None
    if not signer_cmd:
        raise ValueError(
            f"{wallet_path} is a watch-only/hardware keystore (no private key on disk); supply "
            f"--signer-cmd '<external signer>' so the device can sign for {address}. Example: "
            f"--signer-cmd 'python -m neurahash.wallet signer --keystore device.json'.")
    return address, _verified_subprocess_signer(signer_cmd, address)


def resolve_wallet(wallet_path, password, key_dir=".neurahash_keys"):
    """Back-compat shim: resolve a KEYED --wallet to its address and pre-seed its signing key (the
    original behaviour). New code should use resolve_wallet_identity, which also returns the signer
    and supports watch-only/hardware keystores."""
    address, _signer = resolve_wallet_identity(wallet_path, password, signer_cmd=None, key_dir=key_dir)
    return address


def run_solo_coordinator(port, device, experts, state_dir, block_time=12.0):
    """Stand up a LOCAL single-node coordinator in a daemon thread so a solo miner can train into it
    and earn the FULL block reward (no pool fee beyond the protocol's). It is the exact same
    coordinator the pool runs — solo just means YOU are the only miner, so every settled height
    credits your wallet. Returns once the coordinator is accepting connections (port is listening).
    A persistent --state-dir means the solo balance survives a restart (resume-on-restart)."""
    import threading
    # (public-miner v1) --solo needs the FULL coordinator monolith (sharded_pool_node), which the public
    # miner build does NOT ship (it is the private core). This LAZY import fires only on the --solo path;
    # plain pool mining never reaches it. In a public build it raises ModuleNotFoundError, caught by the
    # caller's try/except (main) which reports "--solo: could not start local coordinator". Give an honest
    # message here first so the reason is clear.
    try:
        from sharded_pool_node import coordinator
    except ImportError as _e:
        raise RuntimeError(
            "--solo needs the full node package (sharded_pool_node, the coordinator), which the public "
            "neurahash-miner build does not include. Join an existing pool coordinator with "
            "--host <coordinator> instead, or install the full node to mine solo.") from _e

    def _run():
        try:
            coordinator(port, False, block_time, device, experts, state_dir=state_dir, resume=True)
        except Exception as e:                                  # pragma: no cover - background thread
            print(f"[solo] coordinator exited: {e!r}", flush=True)

    t = threading.Thread(target=_run, name="solo-coordinator", daemon=True)
    t.start()
    # wait for the local coordinator to bind + listen so the miner doesn't race it
    for _ in range(120):
        if _preflight("127.0.0.1", port, timeout=1.0):
            return t
        time.sleep(0.25)
    print(f"[solo] WARNING: local coordinator not listening on :{port} yet; the miner will retry.",
          flush=True)
    return t


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # SIGNED, FAIL-CLOSED AUTO-UPDATE (default ON). Before doing any work, check for a release
    # manifest SIGNED by the project's pinned release key; if a verified, forward (never downgrade)
    # release exists it is checked out and the miner re-execs onto it. On ANY failure (bad signature,
    # unreachable/tampered manifest, commit mismatch, downgrade) it logs a warning and continues on
    # the CURRENT already-verified code -- it never runs unsigned code and never crashes the miner.
    # Opt out with --no-auto-update or NEURAHASH_AUTOUPDATE=0. Skipped for the one-shot --doctor
    # preflight so a diagnostic never silently re-execs into a new version. Rate-limited (once/6h).
    if not args.no_auto_update and not args.doctor:
        try:
            from tools.self_update import maybe_auto_update
            maybe_auto_update(argv=sys.argv)                # returns unless it re-execs onto new code
        except Exception as _e:                             # a launcher must never crash on the updater
            print(f"[self_update] WARN: auto-update skipped ({_e}); continuing on current version",
                  flush=True)

    if args.connect:                                        # auto-discover the live coordinator
        args.host, args.port = resolve_connect(args.connect, args.host, args.port)

    if args.doctor:                                         # preflight checklist, then exit (no join loop)
        # a door LIST is allowed here too — probe the FIRST door (the doctor is a single-target smoke).
        from neurahash import endpoints as _endpoints
        _doors = _endpoints.resolve_endpoints(args.host, args.port)
        _dhost, _dport = (_doors[0].host, _doors[0].port) if _doors else (args.host, args.port)
        healthy = doctor(_dhost, _dport)
        sys.exit(0 if healthy else 1)

    # SOLO: target a LOCAL coordinator by default (the operator can still override --host/--port).
    if args.solo and not args.host:
        args.host = "127.0.0.1"

    if args.bootstrap:                                      # may chdir into the repo — do BEFORE wallet/solo
        ensure_repo(args.repo_dir)
        ensure_deps()

    # CORPUS-BY-HASH (issue #86): with NEURAHASH_CONTENT_STORE set, fetch corpus_data/ from the content
    # store so this joiner's bytes are byte-identical to the coordinator's (else autocrlf drift on a
    # fresh checkout fails the pool's corpus_sha handshake). Runs automatically under --bootstrap, or
    # any time via --sync-corpus. Non-blocking: a store outage warns and falls back to the checkout.
    if args.bootstrap or args.sync_corpus:
        maybe_sync_corpus(force=args.sync_corpus)

    # WALLET (the payout identity, takes precedence over --name): resolve the address the reward
    # accrues to and either pre-seed its local signing key OR wire an external/hardware signer
    # (--signer-cmd) so the key never lands on this machine.
    name = args.name or default_name()
    signer = None
    if args.wallet:
        try:
            name, signer = resolve_wallet_identity(args.wallet, args.wallet_password,
                                                   signer_cmd=args.signer_cmd)
        except Exception as e:
            parser.error(f"--wallet {args.wallet}: {e}")
    elif args.signer_cmd:
        # external signer WITHOUT a --wallet: ASK the device for its address, then verify it controls
        # it (probe must recover) — the payout identity IS the device's address.
        try:
            from neurahash.wallet import SubprocessExternalSigner
            name = SubprocessExternalSigner(args.signer_cmd).address    # query the device
            signer = _verified_subprocess_signer(args.signer_cmd, name)
        except Exception as e:
            parser.error(f"--signer-cmd: {e}")

    # DOOR LIST (multi-endpoint failover): resolve the coordinator door(s) from --host (a single host or
    # a comma-separated failover list) OR env NEURAHASH_COORDS (flag wins). --connect / --solo above set
    # args.host to a single host, so those paths resolve to exactly one door (unchanged). A single bare
    # host -> one Endpoint, so single-door mining is byte-identical to before.
    from neurahash import endpoints as _endpoints
    try:
        doors = _endpoints.resolve_endpoints(args.host, args.port)
    except ValueError as e:
        parser.error(f"--host/{_endpoints.COORDS_ENV}: {e}")
    if not doors:
        parser.error(f"provide --host <coordinator> (or a comma-separated door list, or set "
                     f"{_endpoints.COORDS_ENV}), or --connect <url> to auto-discover it, "
                     f"or --solo to mine into a local one")
    # the global TLS pin (NEURAHASH_TLS_PIN) is the fallback for door entries without their own #pin. A
    # malformed global pin fails loud here (resolve_client_pin normalizes/raises).
    from neurahash import tls
    try:
        global_pin = tls.resolve_client_pin()
    except Exception as e:
        parser.error(f"NEURAHASH_TLS_PIN is malformed: {e}")
    rotator = _endpoints.FailoverRotator(doors)

    # device + psk resolution
    device = pick_device(args.device)
    psk = args.psk.encode() if isinstance(args.psk, str) else args.psk
    if psk is None:
        # default to the pool's shared demo key so a newcomer needs no --psk (matches
        # sharded_pool_node / testnet_node PSK). Override with --psk on BOTH sides for privacy.
        try:
            from testnet_node import PSK
            psk = PSK
        except Exception:
            psk = b"neurahash-demo-psk"

    # SOLO: stand up the local single-node coordinator so the FULL block reward credits this wallet.
    if args.solo:
        try:
            solo_state_dir = os.environ.get("NEURAHASH_SOLO_STATE_DIR", "solo_state")
            print(f"[solo] starting a LOCAL coordinator on :{args.port} (state {solo_state_dir}) — "
                  f"you are the only miner, so every block credits {name} in full.", flush=True)
            run_solo_coordinator(args.port, device, args.experts, solo_state_dir,
                                 block_time=args.block_time)
        except Exception as e:
            parser.error(f"--solo: could not start local coordinator: {e}")

    print("=" * 72)
    print(" NeuraHash sharded-pool miner — earn by training a DISTINCT expert shard"
          + ("  [SOLO]" if args.solo else ""))
    print("=" * 72)
    if len(doors) == 1:
        print(f" coordinator : {doors[0].host}:{doors[0].port}" + ("  (local solo)" if args.solo else ""))
    else:
        print(f" coordinators: {len(doors)} doors (failover) — " + ", ".join(str(d) for d in doors))
        print(f"               primary tried first; rolls to next on any door failure")
    print(f" miner id    : {name}" + ("  (from --wallet)" if args.wallet else ""))
    if signer is not None:
        print(" signing     : EXTERNAL signer (--signer-cmd) — no private key on this machine")
    print(f" device      : {device_banner(device)}")
    print(f" auto-reconnect: on (backoff {args.base_backoff:.0f}-{args.max_backoff:.0f}s, "
          + ("forever)" if args.max_retries is None else f"max {args.max_retries} cycles)"))
    print("=" * 72, flush=True)

    try:
        if len(doors) == 1 and doors[0].pin is None:
            # SINGLE DOOR, no per-endpoint pin: call supervise the LEGACY way (host string + port),
            # passing the already-resolved global pin EXPLICITLY — supervise's per-endpoint pin
            # management deletes the env pin when none resolves, so relying on run_worker reading
            # the env itself silently downgraded this path to plaintext (TLS regression fix).
            supervise(doors[0].host, doors[0].port, name=name, honest=not args.cheat, device=device,
                      psk=psk, max_retries=args.max_retries, base_backoff=args.base_backoff,
                      max_backoff=args.max_backoff, signer=signer, global_pin=global_pin)
        else:
            # MULTI-DOOR (or a single door carrying its OWN #pin): hand supervise the rotator so it can
            # fail over between doors and apply each door's effective TLS pin.
            supervise(rotator, name=name, honest=not args.cheat, device=device, psk=psk,
                      max_retries=args.max_retries, base_backoff=args.base_backoff,
                      max_backoff=args.max_backoff, signer=signer, global_pin=global_pin)
    except KeyboardInterrupt:
        print(f"\n[miner {name}] stopped by user. bye.", flush=True)


if __name__ == "__main__":
    main()
