#!/usr/bin/env python3
"""run_glm_miner.py -- keep-alive supervisor for the NeuraHash GLM shardDiLoCo miner.

WHAT IT DOES (the "set it and forget it" wrapper):
  1. (optional, default ON) runs a signature-verified self-update once at startup so a fresh
     box picks up the latest signed release before it starts.
  2. launches tools/sharddiloco_glm_contributor.py with sane defaults + the known-good env.
  3. if the miner EXITS for ANY reason -- crash, OOM, a coordinator campaign switch (the client
     exits with "FATAL: campaign CHANGED" by design), or the box hiccuping -- it logs the exit and
     RELAUNCHES. A fresh launch re-reads the coordinator's pointer and re-latches whatever campaign
     is current, then reclaims its coordinate via its persisted walk cursor. (This is the exact
     recovery a hard GPU restart was verified to survive: the miner rejoins the running fleet on the
     same coordinate with no manual action.)
  4. backoff is exponential (5s -> capped) so a crash-loop cannot hammer the box, but a miner that
     ran a healthy stretch and then died restarts fast.

Portable: pure Python 3, no shell, works on Windows / Linux / macOS. Ctrl-C stops cleanly.

USAGE (from the repo root):
  python tools/fetch_glm_base.py --dest ~/glm_base --pieces 0-11     # one-time: download the base
  python tools/run_glm_miner.py                                          # then supervise the miner forever

  Pass extra miner flags after `--`, e.g. to claim a specific coordinate or set a VRAM cap:
  python tools/run_glm_miner.py --vram-cap-gb 6.5 -- --expert 1:3
Everything after `--` is handed to the contributor verbatim; the supervisor's own flags come before.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRIB = os.path.join(HERE, "sharddiloco_glm_contributor.py")
FETCH = os.path.join(HERE, "fetch_glm_base.py")
SELF_UPDATE = os.path.join(HERE, "self_update.py")


def _log(msg):
    # ASCII + flush: an unflushed supervisor log turns a diagnosable crash into a silent one.
    line = "[run_glm_miner %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _child_env(vram_cap_gb):
    """The known-good environment every trainer needs, layered over the caller's env."""
    env = dict(os.environ)
    # USE_TF trap: `import transformers` can pick up a stray TensorFlow and die on "numpy.core.umath
    # failed to import"; force the torch backend. PYTHONUNBUFFERED/IOENCODING keep child logs live+ASCII-safe.
    env.setdefault("USE_TF", "0")
    env.setdefault("USE_TORCH", "1")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if vram_cap_gb is not None:
        # Hard per-process GPU ceiling, set BEFORE the child's first CUDA call. Leave headroom for
        # anything else on a shared GPU; the miner pauses (not crashes) when memory is tight.
        env["NEURAHASH_VRAM_CAP_GB"] = str(vram_cap_gb)
    return env


def _base_present(shard_dir):
    try:
        return os.path.isdir(shard_dir) and any(os.scandir(shard_dir))
    except Exception:
        return False


def _run_once(cmd, env):
    """Launch the child, inherit its stdout/stderr (so the user sees the miner directly), wait, return rc."""
    proc = subprocess.Popen(cmd, env=env, cwd=HERE)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        _log("Ctrl-C -- stopping the miner and exiting.")
        try:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
        raise


def _maybe_self_update():
    if not os.path.isfile(SELF_UPDATE):
        return
    _log("checking for a newer signed release (self_update.py; signature-verified, fail-closed)...")
    try:
        # Never let an update hiccup stop mining: a failure here is logged and ignored.
        subprocess.run([sys.executable, SELF_UPDATE], cwd=HERE, timeout=600)
    except Exception as e:
        _log("self-update skipped (%s: %s) -- continuing with the installed version." % (type(e).__name__, e))


def main():
    ap = argparse.ArgumentParser(description="Keep-alive supervisor for the GLM shardDiLoCo miner.")
    ap.add_argument("--shard-dir", default=os.path.expanduser("~/glm_base"),
                    help="model pieces dir (default ~/glm_base; must be fetched once with fetch_glm_base.py)")
    ap.add_argument("--config-dir", default=None, help="config dir (default <shard-dir>/config)")
    ap.add_argument("--pieces", default="0-11", help="pieces to fetch if the base is missing (default 0-11 = one full layer)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vram-cap-gb", type=float, default=None, help="hard per-process GPU memory ceiling (GiB)")
    ap.add_argument("--no-self-update", action="store_true", help="do not check for a signed release at startup")
    ap.add_argument("--no-fetch", action="store_true", help="do not auto-run fetch_glm_base.py when the base dir is empty")
    ap.add_argument("--min-healthy-secs", type=float, default=120.0,
                    help="a run lasting at least this long resets the backoff (default 120s)")
    ap.add_argument("--max-backoff-secs", type=float, default=300.0, help="cap on the restart backoff (default 300s)")
    ap.add_argument("miner_args", nargs=argparse.REMAINDER,
                    help="extra args passed verbatim to the contributor (put them after `--`)")
    a = ap.parse_args()

    config_dir = a.config_dir or os.path.join(a.shard_dir, "config")
    # argparse REMAINDER keeps a leading "--"; drop it so the child sees clean flags.
    extra = list(a.miner_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if not os.path.isfile(CONTRIB):
        _log("FATAL: cannot find %s -- run this from inside the miner repo (tools/run_glm_miner.py)." % CONTRIB)
        return 2

    if not a.no_self_update:
        _maybe_self_update()

    if not _base_present(a.shard_dir):
        if a.no_fetch or not os.path.isfile(FETCH):
            _log("WARNING: base dir %s looks empty and auto-fetch is off. Run: python tools/fetch_glm_base.py "
                 "--dest %s --pieces %s" % (a.shard_dir, a.shard_dir, a.pieces))
        else:
            _log("base dir %s is empty -- fetching pieces %s once..." % (a.shard_dir, a.pieces))
            try:
                subprocess.run([sys.executable, FETCH, "--dest", a.shard_dir, "--pieces", a.pieces], cwd=HERE, check=False)
            except Exception as e:
                _log("fetch failed (%s: %s) -- will still try to start; the miner self-fetches data too." % (type(e).__name__, e))

    cmd = [sys.executable, CONTRIB, "--mode", "glm", "--device", a.device,
           "--shard-dir", a.shard_dir, "--config-dir", config_dir] + extra
    env = _child_env(a.vram_cap_gb)
    _log("supervising: %s" % " ".join(cmd))
    _log("restarts on ANY exit (crash / OOM / campaign switch); backoff up to %ss; Ctrl-C to stop." % a.max_backoff_secs)

    backoff = 5.0
    restarts = 0
    try:
        while True:
            started = time.time()
            rc = _run_once(cmd, env)
            ran = time.time() - started
            restarts += 1
            if ran >= a.min_healthy_secs:
                backoff = 5.0  # a healthy run that then died -> restart promptly
            _log("miner exited rc=%s after %.0fs (restart #%d) -- relaunching in %.0fs. A fresh launch re-latches "
                 "the current campaign and reclaims the coordinate." % (rc, ran, restarts, backoff))
            time.sleep(backoff)
            backoff = min(backoff * 2.0, a.max_backoff_secs)
    except KeyboardInterrupt:
        _log("supervisor stopped by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
