#!/usr/bin/env python3
"""glm-node self-check: is this machine ready to join the GLM-4.7-Flash training trial as a node?

Prints one PASS / WARN / FAIL / SKIP / INFO line per check and ends with a REPORT block you can paste
to the operator as-is (it contains no secrets, no hostname, and your public IP only masked).

    python glm-node/node_selfcheck.py                      # everything (GPU via nvidia-smi + a torch probe)
    python glm-node/node_selfcheck.py --no-gpu             # never touches CUDA (CI / a box without a GPU)
    python glm-node/node_selfcheck.py --disk-test          # also time reads from your expert packs
    python glm-node/node_selfcheck.py --json report.json   # machine-readable copy of the report
    python glm-node/node_selfcheck.py --quick              # skip the ~2 min golden check
    python glm-node/node_selfcheck.py --relay-write-test   # also upload 1 MiB once with YOUR token (uplink + token)

Transport paths are tested SEPARATELY, because a node falls back from one to the next at run time:
relay HTTP (GET /health + timed small GETs), STUN over UDP and TCP (binding request), TURN over TCP and
over UDP (a real allocation when your TURN credentials are set, released immediately; otherwise only the
server's challenge is checked), and the P2P sidecar (started locally, no network). The last lines say
which path this node WILL use and what it falls back to. It also runs the GOLDEN CHECK (the shared-average math on
this CPU must give the fleet's digest) and compares this PC's clock with the relay's.

Credentials are read from the environment only and never printed:
    NEURAHASH_CONTENT_TOKEN   your per-miner relay token (from the operator)
    NEURAHASH_TURN_USER       your TURN username         (from the operator; optional)
    NEURAHASH_TURN_PASSWORD   your TURN password         (from the operator; optional)

Standard library only (psutil is used when present). Python 3.10+. Exit code: 0 READY, 1 NOT READY.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import platform
import random
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER_DIR = os.path.join(HERE, "lib", "runner")
SIDECAR_EXE = os.path.join(RUNNER_DIR, "bin", "p2psidecar.exe")
MANIFEST = os.path.join(HERE, "MANIFEST.sha256")
PACKS_MANIFEST = os.path.join(HERE, "manifests", "packs_manifest.json")

DEFAULT_RELAY = "http://47.84.93.96:8710"
DEFAULT_TURN = "47.84.93.96:3478"

# What the fleet's own 8 GB node (an RTX 4060, Windows 11) runs with -- the numbers below are MEASURED there.
NEED = {
    "python": ((3, 10), (3, 13)),
    "vram_mib_min": 7600,            # an "8 GB" card reports ~8188 MiB; 7600 leaves room for odd SKUs
    "driver_min": 570,               # torch 2.9.1+cu128 needs a CUDA 12.8 capable driver
    "ram_gb_min": 16.0,
    "ram_gb_recommended": 32.0,
    "node_commit_gb": 18.5,          # what a node commits (RAM + pagefile) while it trains: a 16 GB PC fits only
    #                                  with a pagefile, so Windows' commit LIMIT must exceed this plus the desktop's
    "commit_headroom_gb": 4.0,       # the OS, the desktop and the launcher's own share of the commit
    "disk_gb_build": 75.0,           # 46 packs (55.6 GB) + trunk (5.7 GB) + one source shard in flight (<= 2.6 GB) + run
    "disk_gb_run": 10.0,             # after the build: round checkpoints (~0.75 GB x 2), archive, logs, init cache
    "trunk_bytes": 5668965432,
    "pack_bytes": 1207959728,
    "n_packs": 46,
    "config_sha256": "dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb",
    "corpus_sha256": "f9b00e91e50d9eea668e6e520d1b0ba5e1228fa5911de7cb4b4b35520fbf85da",
    "golden_all": "cb5d16ca18642b5a3f9dd4873d9288236a4be26e7a08f837691843c4b0fbdb77",
    "vram_cap_gib": 6.5,
    "clock_skew_warn_s": 60.0,         # the P2P rendezvous slot is time() // 120 on every node
    "part_mb": 37.0,                   # one coded round part on the wire
    "deadline_floor_s": 900.0,         # a round waits at least this long for a late part
}
PINS = {"torch": "2.9.1", "transformers": "5.8.1", "peft": "0.18.0", "numpy": "2.2.6", "safetensors": "0.7.0",
        "accelerate": "1.8.1", "tokenizers": "0.22.1", "huggingface_hub": "1.10.2", "psutil": "7.0.0",
        "cryptography": "46.0.3", "pyarrow": "22.0.0"}
REQUIRED_PKGS = ("torch", "transformers", "peft", "numpy", "safetensors", "accelerate", "cryptography")

RESULTS = []


def say(status, name, detail, need=""):
    RESULTS.append({"status": status, "check": name, "detail": detail, "need": need})
    line = "[%-4s] %-12s %s" % (status, name, detail)
    if need:
        line += "   (need: %s)" % need
    print(line, flush=True)


def human_gb(n):
    return "%.1f GB" % (n / 1e9)


def mask_ip(ip):
    return "x.x.x.x" if ip.count(".") == 3 else "(masked)"


# ------------------------------------------------------------------------------------------------ system
def check_python():
    v = sys.version_info
    lo, hi = NEED["python"]
    bits = 64 if sys.maxsize > 2 ** 32 else 32
    ok = lo <= (v.major, v.minor) <= hi and bits == 64
    say("PASS" if ok else "FAIL", "python", "%d.%d.%d (%d-bit) on %s %s" % (v.major, v.minor, v.micro, bits,
                                                                      platform.system(), platform.release()),
        "Python 3.10-3.13, 64-bit")


def pkg_version(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:                                                     # noqa: BLE001
        return None


def check_packages():
    missing = []
    for name, pin in PINS.items():
        got = pkg_version(name)
        if got is None:
            if name in REQUIRED_PKGS:
                missing.append(name)
                say("FAIL", "pkg:" + name, "not installed", "pip install -r glm-node/requirements-node.txt")
            else:
                say("WARN", "pkg:" + name, "not installed", "pinned %s" % pin)
            continue
        same = got.split("+")[0] == pin
        say("PASS" if same else "WARN", "pkg:" + name, got + ("" if same else "  (our nodes run %s)" % pin))
    tv = pkg_version("transformers")
    if tv:
        try:
            major, minor = (int(x) for x in tv.split(".")[:2])
        except ValueError:
            major, minor = 0, 0
        if (major, minor) < (5, 8):
            say("FAIL", "transformers", "%s has no glm4_moe_lite (GLM-4.7-Flash)" % tv, "transformers >= 5.8")
    return not missing


# ------------------------------------------------------------------------------------------------ GPU
def check_gpu(no_gpu):
    smi = shutil.which("nvidia-smi")
    info = None
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version,compute_cap,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, encoding="utf-8",
                                 errors="replace", timeout=30)
            rows = [r.strip() for r in (out.stdout or "").replace("\r", "").split("\n") if r.strip()]
            if out.returncode == 0 and rows:
                f = [x.strip() for x in rows[0].split(",")]
                info = {"name": f[0], "mib": int(float(f[1])), "driver": f[2], "cc": f[3] if len(f) > 3 else "?",
                        "count": len(rows), "used": int(float(f[4])) if len(f) > 4 and f[4].strip() else None}
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            say("WARN", "gpu", "nvidia-smi failed: %s" % type(e).__name__)
    if info is None:
        say("SKIP" if no_gpu else "FAIL", "gpu", "no NVIDIA GPU found (nvidia-smi missing or silent)",
            "an NVIDIA card with >= 8 GB VRAM and driver >= %d" % NEED["driver_min"])
        return False
    try:
        drv_major = int(info["driver"].split(".")[0])
    except ValueError:
        drv_major = 0
    ok_mem = info["mib"] >= NEED["vram_mib_min"]
    ok_drv = drv_major >= NEED["driver_min"]
    cuda = "?"
    try:
        import re
        hdr = subprocess.run([smi], capture_output=True, encoding="utf-8", errors="replace", timeout=30).stdout or ""
        m = re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", hdr)      # 570-series and 610-series headers
        cuda = m.group(1) if m else "?"
    except (OSError, subprocess.SubprocessError):
        pass
    detail = "%s, %d MiB, driver %s (CUDA up to %s), compute capability %s%s" % (
        info["name"], info["mib"], info["driver"], cuda, info["cc"],
        "" if info["count"] == 1 else " (%d GPUs; the node uses GPU 0)" % info["count"])
    say("PASS" if ok_mem and ok_drv else "FAIL", "gpu", detail,
        ">= 8 GB VRAM (RTX 3070 / 4060 class), driver >= %d" % NEED["driver_min"])
    if info.get("used") is not None:
        free_gib = (info["mib"] - info["used"]) / 1024.0
        say("PASS" if free_gib >= NEED["vram_cap_gib"] + 0.25 else "WARN", "vram-free",
            "%.1f GiB free now (%.1f GiB in use by other programs)" % (free_gib, info["used"] / 1024.0),
            "the node takes %.1f GiB: close GPU-heavy programs before a run" % NEED["vram_cap_gib"])
    if no_gpu:
        say("SKIP", "cuda-probe", "--no-gpu: torch.cuda is not touched")
        return ok_mem and ok_drv
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        say("WARN", "cuda-probe", "CUDA_VISIBLE_DEVICES=%r hides the GPU; NOT probing (a CUDA probe under -1 crashes "
            "some NVIDIA drivers ~80 s later). Unset it before training." % cvd)
        return ok_mem and ok_drv
    code = ("import torch,json\n"
            "ok=torch.cuda.is_available()\n"
            "r={'available':ok,'cuda':torch.version.cuda,'torch':torch.__version__}\n"
            "if ok:\n"
            "  p=torch.cuda.get_device_properties(0)\n"
            "  r.update(name=p.name,total=p.total_memory,cc='%d.%d'%(p.major,p.minor))\n"
            "  a=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16)\n"
            "  b=(a@a).float().abs().sum().item(); torch.cuda.synchronize(); r['matmul_ok']=b==b\n"
            "print('PROBE '+json.dumps(r))\n")
    try:
        t0 = time.time()
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8", errors="replace",
                             timeout=300)
        line = [x for x in (out.stdout or "").splitlines() if x.startswith("PROBE ")]
        if out.returncode != 0 or not line:
            tail = ((out.stderr or "") + (out.stdout or "")).strip().splitlines()[-1:] or ["no output"]
            say("FAIL", "cuda-probe", "torch CUDA probe exited %s: %s" % (out.returncode, tail[0][:200]))
            return False
        r = json.loads(line[0][6:])
        if not r.get("available"):
            say("FAIL", "cuda-probe", "torch %s cannot see the GPU (torch.cuda.is_available() is False; a CPU-only "
                "torch wheel?)" % r.get("torch"), "torch 2.9.1 from https://download.pytorch.org/whl/cu128")
            return False
        say("PASS" if r.get("matmul_ok") else "FAIL", "cuda-probe",
            "torch %s / CUDA %s sees %s (%.1f GiB, cc %s); bf16 matmul ok in a child process, %.1f s"
            % (r.get("torch"), r.get("cuda"), r.get("name"), r.get("total", 0) / 2 ** 30, r.get("cc"),
               time.time() - t0))
        return bool(r.get("matmul_ok")) and ok_mem and ok_drv
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        say("FAIL", "cuda-probe", "could not run the probe: %s" % type(e).__name__)
        return False


# ------------------------------------------------------------------------------------------------ RAM / disk
def ram_bytes():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total, vm.available
    except Exception:                                                     # noqa: BLE001
        pass
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.ullTotalPhys, m.ullAvailPhys
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                k, v = ln.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info["MemTotal"], info.get("MemAvailable", info.get("MemFree", 0))
    except (OSError, KeyError, ValueError):
        return None, None


def check_ram():
    total, avail = ram_bytes()
    if total is None:
        say("WARN", "ram", "could not read the RAM size")
        return True
    gb = total / 1e9
    status = "FAIL" if gb < NEED["ram_gb_min"] - 0.6 else ("WARN" if gb < NEED["ram_gb_recommended"] - 1.0 else "PASS")
    say(status, "ram", "%.1f GB total, %.1f GB available now" % (gb, avail / 1e9),
        "16 GB minimum, 32 GB recommended: the node pins 4 GiB of host buffers for read-ahead and the OS "
        "page cache of the 55.6 GB packs is free speed")
    return status != "FAIL"


def commit_bytes():
    """(commit limit, commit available now) on Windows via GlobalMemoryStatusEx: ullTotalPageFile is the system's
    COMMIT LIMIT (physical RAM + pagefile) and ullAvailPageFile what is still committable. (None, None) elsewhere."""
    if os.name != "nt":
        return None, None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.ullTotalPageFile, m.ullAvailPageFile
    except Exception:                                                     # noqa: BLE001
        pass
    return None, None


def commit_verdict(limit, avail):
    """-> (status, detail) for the commit LIMIT and the commit available now (bytes; None = unknown). A node commits
    ~NEED["node_commit_gb"]; the limit must hold that plus the desktop's headroom, or the node dies with Windows'
    'paging file is too small' (os error 1455) mid-run."""
    need = NEED["node_commit_gb"] + NEED["commit_headroom_gb"]
    if limit is None:
        return "INFO", "commit limit not readable on this OS (Linux overcommits; nothing to check)"
    lim, av = limit / 1e9, (avail / 1e9 if avail is not None else None)
    if lim < need:
        return "WARN", ("Windows commit limit %.1f GB (RAM + pagefile) < %.1f GB: a node commits ~%.1f GB, so set "
                        "the pagefile to 'System managed size' (or >= %.0f GB more)" % (lim, need,
                                                                                      NEED["node_commit_gb"],
                                                                                      need - lim + 0.5))
    if av is not None and av < NEED["node_commit_gb"]:
        return "WARN", ("only %.1f GB of the %.1f GB commit limit is free now; a node needs ~%.1f GB: close big "
                        "programs before the run" % (av, lim, NEED["node_commit_gb"]))
    return "PASS", "commit limit %.1f GB, %.1f GB free now (a node commits ~%.1f GB)" % (
        lim, av if av is not None else float("nan"), NEED["node_commit_gb"])


def check_commit():
    limit, avail = commit_bytes()
    status, detail = commit_verdict(limit, avail)
    say(status, "commit", detail, "" if status in ("PASS", "INFO") else
        "Settings > System > About > Advanced system settings > Performance > Advanced > Virtual memory: "
        "'Automatically manage paging file size' (a system-managed pagefile on the system drive)")
    return True


def check_disk(home, have_packs):
    os.makedirs(home, exist_ok=True)
    du = shutil.disk_usage(home)
    need = NEED["disk_gb_run"] if have_packs else NEED["disk_gb_build"]
    status = "PASS" if du.free / 1e9 >= need else "FAIL"
    say(status, "disk", "%s free on the drive of %s" % (human_gb(du.free), home),
        "%.0f GB free %s; an NVMe SSD (the node reads ~100-110 GB of packs per training step)"
        % (need, "while running (the packs are built)" if have_packs else "to build the packs and run"))
    return status == "PASS"


def check_disk_speed(packs, seconds=20.0, threads=4, block=8 << 20):
    files = [os.path.join(packs, "layer_%d.safetensors" % L) for L in range(1, NEED["n_packs"] + 1)]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        say("SKIP", "disk-speed", "no packs yet under %s (build them first, then re-run with --disk-test)" % packs)
        return
    order = files[:]
    random.Random(7).shuffle(order)
    total = [0]
    lock = threading.Lock()
    stop = time.time() + seconds

    def worker(k):
        buf_n = 0
        for f in order[k::threads] + order:
            if time.time() > stop:
                break
            try:
                with open(f, "rb", buffering=0) as fh:
                    while time.time() < stop:
                        b = fh.read(block)
                        if not b:
                            break
                        buf_n += len(b)
            except OSError:
                continue
        with lock:
            total[0] += buf_n

    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(k,)) for k in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = max(1e-6, time.time() - t0)
    gbs = total[0] / dt / 1e9
    est_lo, est_hi = 72 * 2 ** 30 / (gbs * 1e9), 92 * 2 ** 30 / (gbs * 1e9)
    status = "PASS" if gbs >= 1.5 else ("WARN" if gbs >= 0.5 else "FAIL")
    say(status, "disk-speed", "%.2f GB/s reading packs (%d threads, %.0f s, %s read)%s -> pack reads alone ~%.0f-%.0f s "
        "per training step" % (gbs, threads, dt, human_gb(total[0]),
                               " (much of it from the OS cache)" if gbs > 6 else "", est_lo, est_hi),
        ">= 1.5 GB/s for a useful node (our 4060's NVMe: ~45-55 s per step all-in)")


# ------------------------------------------------------------------------------------------------ files
def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def check_code():
    if not os.path.isfile(MANIFEST):
        say("FAIL", "code", "MANIFEST.sha256 missing -- this is not a complete glm-node checkout")
        return False
    bad, n = [], 0
    for ln in open(MANIFEST, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        want, rel = ln.split(None, 1)
        rel = rel.lstrip("*")
        p = os.path.join(HERE, *rel.split("/"))
        n += 1
        if not os.path.isfile(p) or sha256_file(p) != want:
            bad.append(rel)
    runner = os.path.join(RUNNER_DIR, "glm47_h1x_chain.py")
    rs = sha256_file(runner)[:16] if os.path.isfile(runner) else "MISSING"
    if bad:
        say("FAIL", "code", "%d of %d files differ from MANIFEST.sha256: %s%s" % (
            len(bad), n, ", ".join(bad[:4]), " ..." if len(bad) > 4 else ""),
            "a byte-exact checkout: re-clone, and keep glm-node/.gitattributes (it stops Git rewriting line endings)")
        return False
    say("PASS", "code", "%d files match MANIFEST.sha256 (runner %s)" % (n, rs))
    return True


def check_base(base):
    ok = True
    trunk = os.path.join(base, "pieces", "trunk.safetensors")
    cfg = os.path.join(base, "config", "config.json")
    if os.path.isfile(trunk) and os.path.getsize(trunk) == NEED["trunk_bytes"]:
        say("PASS", "base-trunk", "%s (%d bytes)" % (trunk, NEED["trunk_bytes"]))
    else:
        ok = False
        say("FAIL", "base-trunk", "%s %s" % (trunk, "has %d bytes" % os.path.getsize(trunk) if os.path.isfile(trunk)
                                            else "is missing"), "python glm-node/build_packs.py --base-only")
    if os.path.isfile(cfg):
        s = sha256_file(cfg)
        good = s == NEED["config_sha256"]
        ok &= good
        say("PASS" if good else "FAIL", "base-config", "config.json sha256 %s" % s[:16],
            "%s (the fleet compares it)" % NEED["config_sha256"][:16])
    else:
        ok = False
        say("FAIL", "base-config", "%s missing" % cfg, "python glm-node/build_packs.py --base-only")
    return ok


def check_packs(packs):
    present, wrong = 0, []
    for L in range(1, NEED["n_packs"] + 1):
        p = os.path.join(packs, "layer_%d.safetensors" % L)
        if os.path.isfile(p):
            if os.path.getsize(p) == NEED["pack_bytes"]:
                present += 1
            else:
                wrong.append(L)
    stamp = os.path.join(packs, "PACKS_VERIFIED.json")
    verified = False
    if os.path.isfile(stamp):
        try:
            st = json.load(open(stamp, encoding="utf-8"))
            verified = st.get("ok") is True and all(
                os.path.isfile(os.path.join(packs, k)) and
                abs(os.path.getmtime(os.path.join(packs, k)) - v["mtime"]) < 2.0
                for k, v in st.get("files", {}).items()) and len(st.get("files", {})) == NEED["n_packs"]
        except (OSError, ValueError, KeyError, TypeError):
            verified = False
    if present == NEED["n_packs"] and not wrong:
        say("PASS" if verified else "WARN", "packs", "46 of 46 present, sizes ok, sha256 %s" % (
            "verified (PACKS_VERIFIED.json)" if verified else "NOT verified yet"),
            "" if verified else "python glm-node/build_packs.py --verify-only")
        return True
    say("FAIL", "packs", "%d of 46 present%s under %s" % (present, ", wrong size: %s" % wrong if wrong else "", packs),
        "python glm-node/build_packs.py")
    return False


def check_corpus(corpus):
    if not corpus or not os.path.isfile(corpus):
        say("FAIL", "corpus", "%s missing" % corpus, "python glm-node/fetch_corpus.py")
        return False
    s = sha256_file(corpus)
    good = s == NEED["corpus_sha256"]
    say("PASS" if good else "FAIL", "corpus", "%s sha256 %s" % (os.path.basename(corpus), s[:16]),
        NEED["corpus_sha256"][:16])
    return good


# ------------------------------------------------------------------------------------------------ network
def check_relay(relay, n=5):
    url = relay.rstrip("/") + "/health"
    times, body, date_hdr = [], None, None
    for i in range(n):
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "glm-node-selfcheck"}),
                                        timeout=15) as r:
                raw = r.read(65536)
                if i == 0:
                    body = json.loads(raw.decode("utf-8", "replace"))
                    date_hdr = r.headers.get("Date")
            times.append(time.time() - t0)
        except (urllib.error.URLError, OSError, ValueError) as e:
            say("FAIL", "relay-http", "GET %s failed: %s" % (url, str(e)[:160]),
                "outbound HTTP to %s (the one path every node must have)" % relay)
            return False
    times.sort()
    ok = bool(body and body.get("ok") is True)
    ret = (body or {}).get("retention") or {}
    detail = "GET /health ok=%s, %d tries, median %.0f ms, max %.0f ms" % (
        (body or {}).get("ok"), n, times[len(times) // 2] * 1000, times[-1] * 1000)
    if body:
        detail += " | admission %s, store free %s, watermark %s" % (
            body.get("admission"), human_gb(body.get("free_bytes", 0)), ret.get("watermark", "?"))
    say("PASS" if ok else "FAIL", "relay-http", detail)
    if body and (ret.get("watermark") == "critical" or body.get("free_bytes", 1e12) < 10e9):
        say("WARN", "relay-space", "the relay store is nearly full (%s free, watermark %s): tell the operator before "
            "a run" % (human_gb(body.get("free_bytes", 0)), ret.get("watermark", "?")))
    if body and body.get("admission") not in (None, "open"):
        say("WARN", "relay-admit", "the relay currently admits writes from: %s (ask the operator before a run)"
            % body.get("admission"))
    if date_hdr:
        try:
            import email.utils
            skew = time.time() - email.utils.parsedate_to_datetime(date_hdr).timestamp()
            say("PASS" if abs(skew) < NEED["clock_skew_warn_s"] else "WARN", "clock",
                "this PC's clock is %+.0f s from the relay's" % skew,
                "within %d s (P2P peers meet in 2-minute slots): turn on automatic time sync"
                % NEED["clock_skew_warn_s"])
        except (TypeError, ValueError, OverflowError):
            pass
    return ok


def check_relay_write(relay, size=1 << 20):
    """OPT-IN: upload `size` random bytes ONCE with YOUR token (content-addressed: it cannot overwrite anything),
    read them back, and estimate the time one round part takes to upload."""
    tok = os.environ.get("NEURAHASH_CONTENT_TOKEN", "").strip()
    if not tok:
        say("SKIP", "relay-write", "NEURAHASH_CONTENT_TOKEN not set")
        return False
    data = os.urandom(size)
    sha = hashlib.sha256(data).hexdigest()
    url = relay.rstrip("/") + "/o/" + sha
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={"X-Auth": tok, "Content-Type": "application/octet-stream",
                                          "User-Agent": "glm-node-selfcheck"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            st = r.status
    except urllib.error.HTTPError as e:
        say("FAIL", "relay-write", "PUT refused: HTTP %d %s (token wrong, expired or not allowed to write?)"
            % (e.code, (e.read(300) or b"").decode("utf-8", "replace").strip()[:120]))
        return False
    except (urllib.error.URLError, OSError) as e:
        say("FAIL", "relay-write", "PUT failed: %s" % str(e)[:160])
        return False
    up = time.time() - t0
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "glm-node-selfcheck"}),
                                    timeout=120) as r:
            back = r.read()
    except (urllib.error.URLError, OSError) as e:
        say("FAIL", "relay-write", "uploaded (HTTP %d) but the read-back failed: %s" % (st, str(e)[:120]))
        return False
    mbs = size / 1e6 / max(up, 1e-6)
    part_s = NEED["part_mb"] / max(mbs, 1e-6)
    ok = hashlib.sha256(back).hexdigest() == sha
    say("PASS" if ok and part_s < NEED["deadline_floor_s"] / 3 else ("WARN" if ok else "FAIL"), "relay-write",
        "token accepted (HTTP %d), %d KiB up in %.1f s = %.2f MB/s -> one %.0f MB round part ~%.0f s; read-back %s"
        % (st, size // 1024, up, mbs, NEED["part_mb"], part_s, "identical" if ok else "DIFFERENT"),
        "a part must reach the relay well inside the %.0f s round deadline" % NEED["deadline_floor_s"])
    return ok


def check_golden(timeout=900):
    """The DiLoCo math (outer average + inner AdamW, numpy only) on THIS CPU must give the fleet's digest."""
    script = os.path.join(RUNNER_DIR, "diloco_golden_check.py")
    if not os.path.isfile(script):
        say("FAIL", "golden", "%s missing" % script)
        return False
    t0 = time.time()
    try:
        out = subprocess.run([sys.executable, "-B", script], capture_output=True, encoding="utf-8",
                             errors="replace", timeout=timeout, cwd=RUNNER_DIR)
    except (OSError, subprocess.SubprocessError) as e:
        say("FAIL", "golden", "could not run the golden check: %s" % type(e).__name__)
        return False
    got = None
    for ln in (out.stdout or "").splitlines():
        if ln.startswith("RESULT "):
            try:
                got = json.loads(ln[7:]).get("all")
            except ValueError:
                got = None
    ok = got == NEED["golden_all"]
    say("PASS" if ok else "FAIL", "golden", "digest %s %s the fleet's (%.0f s)" % (
        (got or "MISSING")[:16], "==" if ok else "!=", time.time() - t0),
        "" if ok else "this CPU / numpy computes the shared average differently: send the operator golden output")
    return ok


MAGIC = 0x2112A442


def _stun_msg(mtype, tid, attrs):
    body = b"".join(struct.pack("!HH", t, len(v)) + v + b"\0" * ((4 - len(v) % 4) % 4) for t, v in attrs)
    return struct.pack("!HHI", mtype, len(body), MAGIC) + tid + body


def _stun_attrs(data):
    out, i = {}, 20
    n = struct.unpack("!H", data[2:4])[0]
    end = min(len(data), 20 + n)
    while i + 4 <= end:
        t, ln = struct.unpack("!HH", data[i:i + 4])
        out.setdefault(t, data[i + 4:i + 4 + ln])
        i += 4 + ln + ((4 - ln % 4) % 4)
    return out


def _xor_addr(v, tid):
    if len(v) < 8:
        return None
    fam, port = v[1], struct.unpack("!H", v[2:4])[0] ^ (MAGIC >> 16)
    if fam == 1:
        ip = struct.unpack("!I", v[4:8])[0] ^ MAGIC
        return socket.inet_ntoa(struct.pack("!I", ip)), port
    return "ipv6", port


def _with_integrity(mtype, tid, attrs, key):
    """STUN message with MESSAGE-INTEGRITY (HMAC-SHA1, RFC 5389 sec. 15.4) over everything before it."""
    pre = _stun_msg(mtype, tid, attrs)
    body_len = len(pre) - 20 + 24                       # + the MESSAGE-INTEGRITY attribute itself
    pre = pre[:2] + struct.pack("!H", body_len) + pre[4:]
    mac = hmac.new(key, pre, hashlib.sha1).digest()
    return pre + struct.pack("!HH", 0x0008, 20) + mac


class _Conn:
    def __init__(self, host, port, proto, timeout):
        self.proto, self.addr = proto, (host, port)
        if proto == "udp":
            self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.s.settimeout(timeout)
        else:
            self.s = socket.create_connection((host, port), timeout=timeout)
            self.s.settimeout(timeout)

    def ask(self, msg, tries=3):
        if self.proto == "udp":
            last = None
            for _ in range(tries):
                self.s.sendto(msg, self.addr)
                try:
                    data, _ = self.s.recvfrom(4096)
                    return data
                except socket.timeout as e:
                    last = e
            raise last
        self.s.sendall(msg)
        head = self._read(20)
        n = struct.unpack("!H", head[2:4])[0]
        return head + self._read(n)

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            b = self.s.recv(n - len(buf))
            if not b:
                raise OSError("connection closed by the server")
            buf += b
        return buf

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def check_stun(host, port, proto, show_ip, timeout=5.0):
    name = "stun-" + proto
    try:
        c = _Conn(host, port, proto, timeout)
    except OSError as e:
        say("FAIL", name, "cannot reach %s:%d over %s: %s" % (host, port, proto.upper(), str(e)[:120]))
        return False, None
    try:
        tid = os.urandom(12)
        data = c.ask(_stun_msg(0x0001, tid, []))
        if len(data) < 20 or struct.unpack("!H", data[:2])[0] != 0x0101 or data[8:20] != tid:
            say("FAIL", name, "%s:%d answered, but not with a STUN binding success" % (host, port))
            return False, None
        a = _stun_attrs(data)
        mapped = _xor_addr(a[0x0020], tid) if 0x0020 in a else None
        shown = ("%s:%d" % (mapped[0] if show_ip else mask_ip(mapped[0]), mapped[1])) if mapped else "?"
        say("PASS", name, "%s:%d answered; your public address as seen there: %s" % (host, port, shown))
        return True, mapped
    except (OSError, socket.timeout, struct.error) as e:
        say("FAIL", name, "no STUN answer from %s:%d over %s (%s) -- a firewall blocking %s %d?" % (
            host, port, proto.upper(), type(e).__name__, proto.upper(), port))
        return False, None
    finally:
        c.close()


def check_turn(host, port, proto, user, password, timeout=6.0):
    """TURN Allocate (RFC 5766): unauthenticated first (expect 401 + REALM + NONCE), then -- with credentials --
    the authenticated Allocate, then Refresh(LIFETIME 0) to release it at once. -> 'ok' | 'reachable' | 'fail'."""
    name = "turn-" + proto
    try:
        c = _Conn(host, port, proto, timeout)
    except OSError as e:
        say("FAIL", name, "cannot reach %s:%d over %s: %s" % (host, port, proto.upper(), str(e)[:120]))
        return "fail"
    try:
        req_tr = (0x0019, struct.pack("!B3x", 17))             # REQUESTED-TRANSPORT = UDP (WebRTC TURN)
        tid = os.urandom(12)
        r = c.ask(_stun_msg(0x0003, tid, [req_tr]))
        mt = struct.unpack("!H", r[:2])[0]
        a = _stun_attrs(r)
        if mt != 0x0113 or 0x0014 not in a or 0x0015 not in a:
            say("FAIL", name, "%s:%d did not answer like a TURN server (message type 0x%04x)" % (host, port, mt))
            return "fail"
        realm, nonce = a[0x0014], a[0x0015]
        if not (user and password):
            say("SKIP", name, "server reachable over %s (answered the TURN challenge, realm %r); allocation NOT tested "
                "-- set NEURAHASH_TURN_USER and NEURAHASH_TURN_PASSWORD (from the operator)" % (
                    proto.upper(), realm.decode("utf-8", "replace")))
            return "reachable"
        key = hashlib.md5(b"%s:%s:%s" % (user.encode(), realm, password.encode())).digest()
        tid = os.urandom(12)
        attrs = [req_tr, (0x0006, user.encode()), (0x0014, realm), (0x0015, nonce)]
        r = c.ask(_with_integrity(0x0003, tid, attrs, key))
        mt = struct.unpack("!H", r[:2])[0]
        a = _stun_attrs(r)
        if mt == 0x0113:
            code = a.get(0x0009, b"\0\0\0\0")
            num = (code[2] & 7) * 100 + code[3] if len(code) >= 4 else 0
            reason = code[4:].decode("utf-8", "replace") if len(code) > 4 else ""
            say("FAIL", name, "allocation refused: %d %s%s" % (num, reason,
                " (wrong or expired TURN credentials?)" if num in (400, 401, 431, 437, 441) else ""))
            return "reachable"
        if mt != 0x0103:
            say("FAIL", name, "unexpected answer 0x%04x to the authenticated Allocate" % mt)
            return "reachable"
        relayed = _xor_addr(a.get(0x0016, b""), tid)
        life = struct.unpack("!I", a[0x000D])[0] if 0x000D in a and len(a[0x000D]) == 4 else 0
        # release it right away: Refresh with LIFETIME 0 (fresh transaction, same credentials)
        try:
            c.ask(_with_integrity(0x0004, os.urandom(12), [(0x000D, struct.pack("!I", 0)), (0x0006, user.encode()),
                                                             (0x0014, realm), (0x0015, nonce)], key), tries=1)
            released = "released"
        except (OSError, socket.timeout):
            released = "release not confirmed (it expires by itself)"
        say("PASS", name, "allocation OK over %s (relayed port %s, lifetime %d s), %s" % (
            proto.upper(), relayed[1] if relayed else "?", life, released))
        return "ok"
    except (OSError, socket.timeout, struct.error, KeyError) as e:
        say("FAIL", name, "no TURN answer from %s:%d over %s (%s)" % (host, port, proto.upper(), type(e).__name__))
        return "fail"
    finally:
        c.close()


def check_sidecar(home, timeout=30.0):
    exe = SIDECAR_EXE
    if os.name != "nt":
        alt = os.path.join(RUNNER_DIR, "bin", "p2psidecar")
        exe = alt if os.path.isfile(alt) else exe
        if not os.path.isfile(alt):
            say("WARN", "sidecar", "no Linux/macOS sidecar binary: build it with `python glm-node/sidecar/build_sidecar.py` "
                "(Go 1.27.1); without it this node runs relay-only (still trains)")
            return False
    if not os.path.isfile(exe):
        say("FAIL", "sidecar", "%s missing" % exe)
        return False
    sha = sha256_file(exe)
    work = os.path.join(home, "selfcheck_sidecar")
    os.makedirs(work, exist_ok=True)
    cfg = {"cmd": "config", "id": "selfcheck", "node": "selfcheck", "relay_url": "", "local_token": os.urandom(24).hex(),
           "ice_policy": "nohost", "stun": [], "log_level": "error"}
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000                           # CREATE_NO_WINDOW
    t0 = time.time()
    try:
        p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work,
                             **kw)
    except OSError as e:
        say("FAIL", "sidecar", "cannot start %s: %s (antivirus blocking an unsigned exe?)" % (exe, e))
        return False
    ready = {}
    try:
        p.stdin.write((json.dumps(cfg) + "\n").encode())
        p.stdin.flush()
        box = []

        def rd():
            box.append(p.stdout.readline())

        t = threading.Thread(target=rd, daemon=True)
        t.start()
        t.join(timeout)
        line = box[0] if box else b""
        try:
            ev = json.loads(line.decode("utf-8", "replace") or "{}")
        except ValueError:
            ev = {}
        if ev.get("ev") == "ready":
            ready = ev
        p.stdin.write(b'{"cmd":"shutdown"}\n')
        p.stdin.flush()
        p.wait(timeout=15)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    finally:
        if p.poll() is None:
            p.kill()
    if ready:
        say("PASS", "sidecar", "%s (sha256 %s) started and answered ready in %.1f s: %s" % (
            os.path.basename(exe), sha[:16], time.time() - t0, ready.get("version")))
        return True
    err = (p.stderr.read() if p.stderr else b"").decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
    say("FAIL", "sidecar", "did not answer ready within %.0f s (rc %s) %s" % (timeout, p.returncode, err[0][:160]))
    return False


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--home", default=os.environ.get("NEURAHASH_GLM_HOME", ""),
                    help="the node's data dir (packs/, base/, corpus/, runs/); env NEURAHASH_GLM_HOME")
    ap.add_argument("--packs", default="", help="expert packs dir (default <home>/packs)")
    ap.add_argument("--base", default="", help="base dir with pieces/trunk.safetensors + config/ (default <home>/base)")
    ap.add_argument("--corpus", default="", help="training corpus (default <home>/corpus/e5_scimc_train.jsonl)")
    ap.add_argument("--relay", default=DEFAULT_RELAY)
    ap.add_argument("--turn", default=DEFAULT_TURN, help="TURN/STUN server host:port")
    ap.add_argument("--no-gpu", action="store_true", help="never touch CUDA (no torch probe)")
    ap.add_argument("--no-network", action="store_true", help="skip relay / STUN / TURN")
    ap.add_argument("--disk-test", action="store_true", help="time ~20 s of reads from the packs")
    ap.add_argument("--show-ip", action="store_true", help="print your public IP unmasked (default: masked)")
    ap.add_argument("--json", default="", help="also write the report as JSON to this path")
    ap.add_argument("--quick", action="store_true", help="skip the golden check (~1-3 min, ~1.5 GB RAM)")
    ap.add_argument("--relay-write-test", action="store_true",
                    help="upload 1 MiB once with your token: proves the token and measures your uplink")
    a = ap.parse_args(argv)
    home = os.path.abspath(a.home or os.path.join(os.path.expanduser("~"), "glmnode"))
    packs = os.path.abspath(a.packs or os.path.join(home, "packs"))
    base = os.path.abspath(a.base or os.path.join(home, "base"))
    corpus = os.path.abspath(a.corpus or os.path.join(home, "corpus", "e5_scimc_train.jsonl"))
    host, _, port = a.turn.rpartition(":")
    port = int(port or 3478)
    user = os.environ.get("NEURAHASH_TURN_USER", "")
    password = os.environ.get("NEURAHASH_TURN_PASSWORD", "")

    print("GLM-NODE SELF-CHECK (glm-node %s) -- paste everything from here to the end to the operator" % _version())
    print("home %s | packs %s | base %s" % (_redact_home(home), _redact_home(packs), _redact_home(base)))
    code = check_code()                    # FIRST: every later check is meaningless on altered or rewritten files
    check_python()
    pk = check_packages()
    gpu = check_gpu(a.no_gpu)
    ram = check_ram()
    check_commit()
    have_packs = all(os.path.isfile(os.path.join(packs, "layer_%d.safetensors" % L)) for L in range(1, 47))
    disk = check_disk(home, have_packs)
    files = check_base(base)
    files = check_packs(packs) and files
    files = check_corpus(corpus) and files
    if a.disk_test:
        check_disk_speed(packs)
    tok = bool(os.environ.get("NEURAHASH_CONTENT_TOKEN", "").strip())
    say("PASS" if tok else "WARN", "relay-token", "NEURAHASH_CONTENT_TOKEN is %s (value never printed or tested here)"
        % ("set" if tok else "NOT set"), "" if tok else "your per-miner token from the operator, before a run")
    say("PASS" if (user and password) else "INFO", "turn-creds", "NEURAHASH_TURN_USER / _PASSWORD %s" % (
        "set" if (user and password) else "not set (P2P through TURN is then not possible; relay still works)"))
    relay = turn_tcp = turn_udp = "skip"
    stun_udp = stun_tcp = False
    if a.no_network:
        say("SKIP", "network", "--no-network: relay / STUN / TURN not tested")
    else:
        relay = "ok" if check_relay(a.relay) else "fail"
        stun_udp, mapped = check_stun(host, port, "udp", a.show_ip)
        stun_tcp, _ = check_stun(host, port, "tcp", a.show_ip)
        turn_tcp = check_turn(host, port, "tcp", user, password)
        turn_udp = check_turn(host, port, "udp", user, password)
        if a.relay_write_test:
            check_relay_write(a.relay)
    side = check_sidecar(home)
    if not a.quick:
        check_golden()
    say("INFO", "p2p-direct", "a direct (hole-punched) path needs a live peer: ICE tries it per peer at run time "
        "when the run card allows it (ice_policy nohost); the fleet's TURN profile uses relay-only ICE")

    # ---- the fallback ladder this node will actually use
    def word(x):
        return {"ok": "OK", "reachable": "reachable, no allocation test", "fail": "FAIL", "skip": "not tested"}.get(x, x)
    ladder = "P2P direct: UNTESTED (needs a peer) -> TURN TCP: %s -> TURN UDP: %s -> relay HTTP: %s" % (
        word(turn_tcp), word(turn_udp), word(relay))
    if relay != "ok":
        verdict = ("this node CANNOT train: the relay is the one path every node needs (uploads, names, commits and "
                   "every fallback go through it)" if relay == "fail" else "relay not tested (--no-network)")
    elif side and tok and turn_tcp == "ok":
        verdict = ("this node will fetch peers' parts via P2P over TURN TCP when that is faster, and fall back to the "
                   "relay automatically (uploads always go to the relay)")
    elif side and tok and turn_udp == "ok":
        verdict = ("this node can use P2P over TURN UDP (only if the run card lists a UDP TURN URL), else the relay; "
                   "the fleet's TURN profile is TCP-only, so expect relay")
    else:
        verdict = "this node will exchange through the relay only (P2P unavailable here); it trains normally"
    print()
    print("FALLBACK LADDER: " + ladder)
    print("=> " + verdict)
    fails = [r for r in RESULTS if r["status"] == "FAIL"]
    warns = [r for r in RESULTS if r["status"] == "WARN"]
    ready = not fails
    print("RESULT: %s (%d FAIL, %d WARN) -- python %s, %s" % ("READY" if ready else "NOT READY", len(fails), len(warns),
                                                               platform.python_version(), platform.platform(terse=True)))
    for r in fails:
        print("  FAIL %s: %s%s" % (r["check"], r["detail"], (" -> " + r["need"]) if r["need"] else ""))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"version": _version(), "ready": ready, "ladder": ladder, "verdict": verdict,
                       "results": RESULTS, "when": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    return 0 if ready else 1


def _version():
    try:
        return open(os.path.join(HERE, "VERSION_GLM_NODE"), encoding="utf-8").read().strip()
    except OSError:
        return "?"


def _redact_home(p):
    """Hide the user name in a home-relative path (the report gets pasted)."""
    h = os.path.expanduser("~")
    return ("~" + p[len(h):]) if p.lower().startswith(h.lower()) else p


if __name__ == "__main__":
    sys.exit(main())
