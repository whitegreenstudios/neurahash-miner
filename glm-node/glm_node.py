#!/usr/bin/env python3
"""glm-node launcher: run THIS machine as one node of a GLM-4.7-Flash DiLoCo training run.

    python glm-node/glm_node.py keygen                    # make your P2P signing key; send the PUBKEY line to the operator
    python glm-node/glm_node.py preflight --card CARD     # every check a launch needs; nothing is started
    python glm-node/glm_node.py run --card CARD           # start the node (it trains until the run ends)
    python glm-node/glm_node.py run --card CARD --rejoin  # re-enter the same run after a crash / reboot / network loss
    python glm-node/glm_node.py run --card CARD --relay-only   # never use P2P (the relay carries everything)
    python glm-node/glm_node.py status --card CARD        # what the node is doing (log tail, rounds, transport)
    python glm-node/glm_node.py logs --card CARD          # zip the diagnosis files to send to the operator

The RUN CARD is a small JSON file the operator sends you for one run (it holds NO secrets): the run's namespace and
attempt, your node letter, the exact training arguments every node must share, and the P2P settings. See
examples/run_card.example.json. Your credentials come only from environment variables (never from a file in this
folder, never from the card):
    NEURAHASH_CONTENT_TOKEN    your per-miner relay token            (required)
    NEURAHASH_TURN_USER        your TURN username                    (needed for P2P through TURN)
    NEURAHASH_TURN_PASSWORD    your TURN password                    (needed for P2P through TURN)
    NEURAHASH_P2P_SIGNAL_KEY   the run's P2P signalling key          (only if the card says signal_key: "env")
    NEURAHASH_NODE             your node letter                      (optional; overrides the card's "node")
    NEURAHASH_GLM_HOME         your data dir (packs/, base/, corpus/, runs/, keys/)

Why a launcher at all: a node that fails AT START (wrong file, wrong version, missing credential) can void the
whole run for everyone, so every check runs here first, and nothing starts until all of them pass. It also keeps
the training process in its own process group (Ctrl+C once = leave at the next round boundary, twice = leave now;
either is a clean leave the node can come back from with --rejoin), keeps Windows from sleeping while the node
runs, sets the VRAM cap from the card, and falls back to relay-only transport (--p2p force-relay) by itself when
P2P cannot work on this machine.

Python 3.10+, standard library only.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER_DIR = os.path.join(HERE, "lib", "runner")
RUNNER = os.path.join(RUNNER_DIR, "glm47_h1x_chain.py")
TRANSPORT = os.path.join(RUNNER_DIR, "p2p_transport.py")
SIDECAR_EXE = os.path.join(RUNNER_DIR, "bin", "p2psidecar.exe" if os.name == "nt" else "p2psidecar")
MANIFEST = os.path.join(HERE, "MANIFEST.sha256")
PACKS_MANIFEST = os.path.join(HERE, "manifests", "packs_manifest.json")

NODE_LETTERS = "ABCDEFGH"
CORPUS_SHA256 = "f9b00e91e50d9eea668e6e520d1b0ba5e1228fa5911de7cb4b4b35520fbf85da"
PINS = {"torch": "2.9.1", "transformers": "5.8.1", "peft": "0.18.0", "numpy": "2.2.6", "safetensors": "0.7.0",
        "accelerate": "1.8.1"}
# The runner's registered profiles bind the VRAM cap to the node LETTER (its PREREG-DL VOID-17): a node with any
# other cap is refused at its start -- and a refused start voids the attempt for everyone. In the engineering
# profiles ("test", and the Stage-0 runner's "trial": member-scoped outcomes, so one node's own failure evicts that
# node instead of stopping the run) the cap is free.
REGISTERED_PROFILES = ("arm", "arm_s2", "arm_r24", "smoke", "smoke_r24")   # runner DL_ARM/SMOKE_PROFILES
REGISTERED_VRAM_CAP = {"A": 9.5, "B": 6.5, "C": 9.5}
DEFAULT_VRAM_CAP = 6.5          # the fleet's 8 GB node: measured peak 5.6 GiB at this cap (more VRAM is not used)
VRAM_MARGIN_GIB = 1.25          # what the pack reader itself leaves an 8 GB card's desktop (glm_arc_eval.py)


def device_vram_cap(total_gib):
    """The cap for THIS card when neither the card nor --vram-cap names one: 6.5 GiB on any card with >= 7.75 GiB
    (the recipe's working set does not grow with the card), None below that (too small)."""
    return DEFAULT_VRAM_CAP if total_gib >= DEFAULT_VRAM_CAP + VRAM_MARGIN_GIB else None
# flags that belong to ONE machine; the card's argv_common must not carry them (the launcher adds them)
NODE_LOCAL_FLAGS = {"--node": 1, "--config-dir": 1, "--pack-dir": 1, "--data": 1, "--out-dir": 1, "--vram-cap": 1,
                    "--pin-layers": 1, "--p2p-config": 1, "--init-cache-shared": 1, "--rejoin": 0,
                    "--resume-from": 1}
FORBIDDEN_FLAGS = ("--selftest", "--plan", "--closer-dir", "--diloco-pause-at", "--scratch")
CONSOLE_LOG = "glm_node_console.log"


def say(msg):
    print("[glm-node] " + msg, flush=True)


def fail(msg, rc=2):
    say("REFUSED: " + msg)
    sys.exit(rc)


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def pkg_version(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:                                                     # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ paths / card
class Ctx(object):
    pass


def make_ctx(a):
    c = Ctx()
    c.home = os.path.abspath(a.home or os.environ.get("NEURAHASH_GLM_HOME") or
                             os.path.join(os.path.expanduser("~"), "glmnode"))
    c.packs = os.path.abspath(a.packs or os.path.join(c.home, "packs"))
    c.base = os.path.abspath(a.base or os.path.join(c.home, "base"))
    c.corpus = os.path.abspath(a.corpus or os.path.join(c.home, "corpus", "e5_scimc_train.jsonl"))
    c.keyfile = os.path.abspath(a.key_file or os.path.join(c.home, "keys", "p2p_sign.key"))
    c.init_cache = os.path.join(c.home, "init_cache")
    c.card = None
    if getattr(a, "card", None):
        c.card = load_card(a.card)
        c.node = (getattr(a, "node", "") or os.environ.get("NEURAHASH_NODE", "") or c.card.get("node") or "").upper()
        c.out = os.path.abspath(a.out or os.path.join(c.home, "runs", c.card["ns"], c.node or "X"))
    return c


def load_card(path):
    try:
        with open(path, encoding="utf-8") as f:
            card = json.load(f)
    except (OSError, ValueError) as e:
        fail("cannot read the run card %s: %s" % (path, e))
    need = ("card_version", "runner_sha256", "anchor", "ns", "attempt", "nodes", "argv_common")
    miss = [k for k in need if k not in card]
    if miss:
        fail("the run card lacks %s (ask the operator for a complete card)" % ", ".join(miss))
    if card["card_version"] != 1:
        fail("run card version %r; this launcher reads version 1" % card["card_version"])
    if not isinstance(card["argv_common"], list) or not all(isinstance(x, str) for x in card["argv_common"]):
        fail("run card: argv_common must be a list of strings")
    if not re.match(r"^https?://[A-Za-z0-9.\-]+(:[0-9]+)?/?$", str(card["anchor"])):
        fail("run card: anchor %r is not a plain http(s)://host:port URL" % card["anchor"])
    if not re.match(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,80}$", str(card["ns"])):
        fail("run card: ns %r has characters a namespace may not have" % card["ns"])
    return card


def argv_value(argv, flag):
    for i, x in enumerate(argv):
        if x == flag and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith(flag + "="):
            return x.split("=", 1)[1]
    return None


def check_argv_common(card, problems):
    av = card["argv_common"]
    if av and not av[0].startswith("--"):
        problems.append("argv_common must start with a --flag, not %r" % av[0])
    for i, x in enumerate(av):
        if re.search(r"[<>|&;`$\r\n]", x):
            problems.append("argv_common token %r looks like shell syntax (redirection / control); a run card holds "
                            "the runner's arguments only" % x)
        key = x.split("=", 1)[0]
        if key in NODE_LOCAL_FLAGS:
            problems.append("argv_common carries the node-local flag %s (the launcher sets it per machine)" % key)
        if key in FORBIDDEN_FLAGS or key == "--mode" and argv_value(av, "--mode") != "arm":
            problems.append("argv_common carries %s, which a node never runs with" % x)
    for flag, want in (("--mode", "arm"), ("--exchange", "diloco"), ("--ns", card["ns"]),
                       ("--attempt", str(card["attempt"])), ("--nodes", str(card["nodes"]))):
        got = argv_value(av, flag)
        if got != want:
            problems.append("argv_common %s is %r, the card says %r" % (flag, got, want))
    anch = argv_value(av, "--anchor")
    if anch is not None and anch.rstrip("/") != str(card["anchor"]).rstrip("/"):
        problems.append("argv_common --anchor %s differs from the card's anchor %s" % (anch, card["anchor"]))


def card_manifest_problem(card):
    """None, or why this checkout is not the glm-node build the card names. The run's step-0 agreement compares the
    runner's and p2p_transport's sha256, not the libraries beside them; a card that carries glm_node_manifest_sha256
    (the sha256 of MANIFEST.sha256, which lists every shipped file's sha256) pins every file of the package."""
    want = card.get("glm_node_manifest_sha256")
    if not want:
        return None
    have = sha256_file(MANIFEST) if os.path.isfile(MANIFEST) else None
    if have != want:
        return ("glm-node MANIFEST.sha256 %s != the run's %s: this checkout is another glm-node build than the other "
                "nodes run (check out the tag the operator named)" % ((have or "MISSING")[:16], str(want)[:16]))
    return None


def platform_key():
    """'windows/amd64', 'linux/amd64', 'darwin/arm64', ... (Go's GOOS/GOARCH names)."""
    import platform
    m = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(m, m)
    osn = "windows" if os.name == "nt" else ("darwin" if sys.platform == "darwin" else "linux")
    return "%s/%s" % (osn, arch)


def newest_round_ckpt(out, node):
    best = None
    if os.path.isdir(out):
        for n in os.listdir(out):
            m = re.match(r"^round_ckpt_%s_r(\d{4})\.pt$" % re.escape(node), n)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), os.path.join(out, n))
    return best


# ------------------------------------------------------------------------------------------------ preflight
def gpu_info():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        rows = [r.strip() for r in (out.stdout or "").replace("\r", "").split("\n") if r.strip()]
        if out.returncode != 0 or not rows:
            return None
        f = [x.strip() for x in rows[0].split(",")]
        used = None
        if len(f) > 3:
            try:
                used = int(float(f[3]))
            except ValueError:
                used = None
        return {"name": f[0], "mib": int(float(f[1])), "driver": f[2], "used_mib": used}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def relay_health(anchor, tries=3, wait=20):
    url = anchor.rstrip("/") + "/health"
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "glm-node"}),
                                        timeout=20) as r:
                body = json.loads(r.read(65536).decode("utf-8", "replace"))
            if body.get("ok") is True:
                return body, None
            last = "health says ok=%r" % body.get("ok")
        except Exception as e:                                            # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, str(e)[:120])
        if i + 1 < tries:
            time.sleep(wait)
    return None, last


def sidecar_pubkey(keyfile, create):
    """(pubkey hex or None, message) via `p2psidecar keygen` (it never prints the seed)."""
    if not os.path.isfile(SIDECAR_EXE):
        return None, "sidecar executable missing (%s)" % SIDECAR_EXE
    if not create and not os.path.isfile(keyfile):
        return None, "no signing key yet (%s): run `python glm-node/glm_node.py keygen` and send the PUBKEY " \
                     "line to the operator" % keyfile
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
    try:
        out = subprocess.run([SIDECAR_EXE, "keygen", keyfile], capture_output=True, encoding="utf-8",
                             errors="replace", timeout=60, **kw)
    except OSError as e:
        return None, "cannot run the sidecar (%s) -- blocked by antivirus? %s" % (SIDECAR_EXE, e)
    m = re.search(r"PUBKEY ([0-9a-f]{64})", out.stdout or "")
    if out.returncode != 0 or not m:
        return None, "sidecar keygen failed (rc %s): %s" % (out.returncode, (out.stderr or "").strip()[:160])
    return m.group(1), ("created" if "CREATED" in (out.stdout or "") else "exists")


def p2p_plan(c, a, problems, warns):
    """-> (p2p mode for the runner, config dict or None). The fleet's mode comes from the card's argv_common; this
    node can only ever step DOWN to force-relay (same transport file, so the run's agreement is unchanged)."""
    av = c.card["argv_common"]
    fleet_mode = argv_value(av, "--p2p") or "off"
    if fleet_mode in ("off", "force-relay"):
        return fleet_mode, None
    if a.relay_only:
        warns.append("--relay-only: this node uses --p2p force-relay (all fetches over the relay)")
        return "force-relay", None
    p = c.card.get("p2p") or {}
    why = []
    if not os.path.isfile(SIDECAR_EXE):
        why.append("no sidecar binary for this OS (%s)" % os.path.basename(SIDECAR_EXE))
    else:
        want = p.get("exe_sha256")
        if isinstance(want, dict):                       # {"windows/amd64": sha, "linux/amd64": sha, ...}
            want = want.get(platform_key())
        if want and sha256_file(SIDECAR_EXE) != want:
            why.append("the sidecar binary's sha256 differs from the card's exe_sha256 for %s" % platform_key())
    if pkg_version("cryptography") is None:
        why.append("python package cryptography missing (needed for signed signalling)")
    turn = p.get("turn") or []
    pol = p.get("ice_policy", "relay")
    user, pw = os.environ.get("NEURAHASH_TURN_USER", ""), os.environ.get("NEURAHASH_TURN_PASSWORD", "")
    if turn and not (user and pw):
        why.append("NEURAHASH_TURN_USER / NEURAHASH_TURN_PASSWORD not set")
    if pol == "relay" and not turn:
        why.append("ice_policy relay needs TURN URLs in the card")
    if p.get("signal_key") == "env" and not os.environ.get("NEURAHASH_P2P_SIGNAL_KEY", "").strip():
        why.append("the card wants NEURAHASH_P2P_SIGNAL_KEY and it is not set")
    peers = p.get("peer_pubkeys") or {}
    mine, msg = (None, "skipped")
    if not why:
        mine, msg = sidecar_pubkey(c.keyfile, create=False)
        if mine is None:
            why.append(msg)
        elif peers.get(c.node) != mine:
            why.append("your public key %s... is not the card's key for node %s (send the operator the PUBKEY line "
                       "from `glm_node.py keygen`)" % (mine[:12], c.node))
        missing = [x for x in NODE_LETTERS[:int(c.card["nodes"])] if x not in peers]
        if missing:
            why.append("the card has no P2P public key for node(s) %s" % ",".join(missing))
        badkeys = sorted(k for k, v in peers.items() if not re.match(r"^[0-9a-f]{64}$", str(v)))
        if badkeys:
            why.append("the card's P2P public key for node(s) %s is not 64 hex" % ",".join(badkeys))
    if why:
        warns.append("P2P is not usable here (%s) -> this node falls back to --p2p force-relay: it still trains, "
                     "every fetch goes over the relay" % "; ".join(why))
        return "force-relay", None
    sc = {"kind": "pion", "exe": "bin/" + os.path.basename(SIDECAR_EXE), "exe_sha256": sha256_file(SIDECAR_EXE),
          "ice_policy": pol, "turn": list(turn), "stun": list(p.get("stun") or []),
          "sign_key_file": c.keyfile, "peer_pubkeys": dict(peers), "allow_private_relay": False,
          "connect_slot_s": int(p.get("connect_slot_s", 120)), "connect_timeout_s": int(p.get("connect_timeout_s", 60))}
    if turn:
        sc["turn_user"] = user
        sc["turn_password_env"] = "NEURAHASH_TURN_PASSWORD"
    if p.get("signal_key") == "env":
        sc["signal_key_env"] = "NEURAHASH_P2P_SIGNAL_KEY"
    else:
        sc["signal_key_from"] = "relay_token"
    return fleet_mode, {"note": "generated by glm_node.py for node %s of %s (no secrets: they are read from the "
                                "environment and the key file named here)" % (c.node, c.card["ns"]), "sidecar": sc}


def p2p_dry_build(c, mode, cfg):
    """Build (never start) the transport exactly as the runner will, with the runner's own p2p_transport.py:
    load_config + config_problems + PionSidecar.from_config (exe sha256, secrets, keys). -> None, or the error the
    runner would have turned into a setup refusal (exit 2) at its start."""
    tmp = os.path.join(c.out, "p2p_config_node.check.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    added = RUNNER_DIR not in sys.path
    if added:
        sys.path.insert(0, RUNNER_DIR)
    try:
        import p2p_transport as PT
        if os.path.abspath(PT.__file__) != os.path.abspath(TRANSPORT):
            return "imported %s instead of the shipped p2p_transport.py" % PT.__file__
        env = dict(os.environ)
        PT.build_from_config(mode, PT.load_config(tmp), node=c.node, nodes=list(NODE_LETTERS[:int(c.card["nodes"])]),
                             relay_url=str(c.card["anchor"]).rstrip("/"),
                             relay_token=env.get("NEURAHASH_CONTENT_TOKEN", ""), base_dir=RUNNER_DIR,
                             work_dir=os.path.join(c.out, "p2p"), env=env)
        return None
    except Exception as e:                                                # noqa: BLE001
        return "%s: %s" % (type(e).__name__, str(e)[:300])
    finally:
        if added and RUNNER_DIR in sys.path:
            sys.path.remove(RUNNER_DIR)
        try:
            os.remove(tmp)
        except OSError:
            pass


def preflight(c, a, quiet=False):
    """-> (ok, problems, warns, plan dict). Read-only except creating dirs."""
    problems, warns, plan = [], [], {}
    card = c.card
    check_argv_common(card, problems)
    # node letter
    n = int(card["nodes"])
    if not c.node or c.node not in NODE_LETTERS[:n]:
        problems.append("node letter %r is not one of %s (set it with --node, NEURAHASH_NODE or the card's \"node\")"
                        % (c.node, NODE_LETTERS[:n]))
    elif c.node == "A" and not a.i_am_the_operator:
        problems.append("node A publishes the run's initial state; outside nodes take another letter")
    # code identity
    rs = sha256_file(RUNNER)
    if rs != card["runner_sha256"]:
        problems.append("runner sha256 %s != the run's %s: this node would be refused as a version skew (update "
                        "glm-node, or ask the operator which glm-node tag the run uses)" % (rs[:16], card["runner_sha256"][:16]))
    if card.get("p2p_transport_sha256") and sha256_file(TRANSPORT) != card["p2p_transport_sha256"]:
        problems.append("p2p_transport.py differs from the run's (version skew)")
    pm_problem = card_manifest_problem(card)
    if pm_problem:
        problems.append(pm_problem)
    if os.path.isfile(MANIFEST):
        bad = []
        for ln in open(MANIFEST, encoding="utf-8"):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            want, rel = ln.split(None, 1)
            p = os.path.join(HERE, *rel.lstrip("*").split("/"))
            if not os.path.isfile(p) or sha256_file(p) != want:
                bad.append(rel)
        if bad:
            problems.append("%d glm-node file(s) differ from MANIFEST.sha256 (%s): re-clone without changing them"
                            % (len(bad), ", ".join(bad[:3])))
    else:
        problems.append("MANIFEST.sha256 missing")
    # libraries (they are NOT in the run's version check, so a different torch would silently train differently)
    for name, pin in PINS.items():
        got = pkg_version(name)
        if got is None:
            problems.append("python package %s is not installed (pip install -r glm-node/requirements-node.txt)" % name)
        elif got.split("+")[0] != pin and not a.allow_version_mismatch:
            problems.append("%s %s installed, the fleet runs %s (install the pinned version, or pass "
                            "--allow-version-mismatch if the operator agrees)" % (name, got, pin))
    v = sys.version_info
    if not (3, 10) <= (v.major, v.minor) <= (3, 13):
        problems.append("Python %d.%d: use 3.10-3.13" % (v.major, v.minor))
    # GPU / VRAM cap
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        problems.append("CUDA_VISIBLE_DEVICES=%r hides the GPU: unset it" % cvd)
    g = gpu_info()
    prof = argv_value(card["argv_common"], "--diloco-profile") or "?"
    cap = a.vram_cap or card.get("vram_cap_gb")
    if prof in REGISTERED_PROFILES:
        want_cap = REGISTERED_VRAM_CAP.get(c.node)
        if want_cap is None:
            problems.append("profile %s has no registered VRAM cap for node %s" % (prof, c.node))
        elif cap is not None and abs(float(cap) - want_cap) > 0.005:
            problems.append("profile %s requires --vram-cap %.2f on node %s (got %s)" % (prof, want_cap, c.node, cap))
        cap = want_cap if cap is None else cap
    if g is None:
        problems.append("no NVIDIA GPU visible (nvidia-smi)")
        cap = float(cap if cap is not None else DEFAULT_VRAM_CAP)
    else:
        total = g["mib"] / 1024.0
        if cap is None:
            cap = device_vram_cap(total)              # NEVER the runner's own 23.0 GiB default
            if cap is None:
                problems.append("%s has %.1f GiB VRAM: a node needs an 8 GB card" % (g["name"], total))
                cap = DEFAULT_VRAM_CAP
        cap = float(cap)
        if cap > total - VRAM_MARGIN_GIB:
            problems.append("VRAM cap %.2f GiB does not fit %s (%.1f GiB; the node keeps >= %.2f GiB for the "
                            "desktop)" % (cap, g["name"], total, VRAM_MARGIN_GIB))
        if g.get("used_mib") is not None and cap > total - g["used_mib"] / 1024.0 - 0.25:
            warns.append("other programs already use %.1f GiB of VRAM (%.1f free); the node needs %.2f GiB -- close "
                         "GPU-heavy programs (games, video, browsers) before the run" % (
                             g["used_mib"] / 1024.0, total - g["used_mib"] / 1024.0, cap))
        plan["gpu"] = "%s %d MiB driver %s" % (g["name"], g["mib"], g["driver"])
    plan["vram_cap"] = cap
    plan["pin_layers"] = int(a.pin_layers if a.pin_layers is not None else card.get("pin_layers", 0))
    # files
    try:
        pm = json.load(open(PACKS_MANIFEST, encoding="utf-8"))
    except (OSError, ValueError):
        pm = None
        problems.append("manifests/packs_manifest.json missing or unreadable")
    if pm:
        for name, ent in pm["config_files"].items():
            p = os.path.join(c.base, "config", name)
            if not (os.path.isfile(p) and os.path.getsize(p) == ent["size"] and sha256_file(p) == ent["sha256"]):
                problems.append("base/config/%s missing or not the fleet's (python glm-node/build_packs.py --base-only)"
                                % name)
        for key in ("trunk", "model_manifest"):
            ent = pm["base"][key]
            p = os.path.join(c.base, *ent["local"].split("/"))
            if not (os.path.isfile(p) and os.path.getsize(p) == ent["size"]):
                problems.append("base/%s missing or the wrong size (python glm-node/build_packs.py --base-only)"
                                % ent["local"])
        stamp = os.path.join(c.packs, "PACKS_VERIFIED.json")
        try:
            st = json.load(open(stamp, encoding="utf-8"))
            files = st.get("files", {})
            okp = st.get("ok") is True and len(files) == 46 and all(
                os.path.isfile(os.path.join(c.packs, k)) and os.path.getsize(os.path.join(c.packs, k)) == v["size"]
                and abs(os.path.getmtime(os.path.join(c.packs, k)) - v["mtime"]) < 2.0 for k, v in files.items())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            okp = False
        if not okp:
            problems.append("packs under %s are not all present and verified (python glm-node/build_packs.py, or "
                            "--verify-only)" % c.packs)
    if not (os.path.isfile(c.corpus) and sha256_file(c.corpus) == CORPUS_SHA256):
        problems.append("corpus %s missing or not sha256 %s... (python glm-node/fetch_corpus.py)"
                        % (c.corpus, CORPUS_SHA256[:12]))
    # disk
    os.makedirs(c.out, exist_ok=True)
    free = shutil.disk_usage(c.out).free
    if free < 10e9:
        problems.append("only %.1f GB free where the node writes (%s): it needs >= 10 GB (round checkpoints, "
                        "archive, logs)" % (free / 1e9, c.out))
    # credentials (never printed)
    if not os.environ.get("NEURAHASH_CONTENT_TOKEN", "").strip():
        problems.append("NEURAHASH_CONTENT_TOKEN is not set: get your per-miner credentials from the operator "
                        "(the node never falls back to a shared token)")
    # an attempt this node already took part in
    ck = newest_round_ckpt(c.out, c.node) if c.node else None
    attempts = os.path.join(c.out, "ATTEMPTS.txt")
    if (ck or os.path.isfile(attempts)) and not a.rejoin:
        problems.append("this node already ran in %s (%s): to re-enter the SAME run use --rejoin"
                        % (c.out, "round checkpoint r%04d" % ck[0] if ck else "ATTEMPTS.txt"))
    if a.rejoin and not (ck or os.path.isfile(attempts)):
        warns.append("--rejoin but no earlier output of this node in %s: the runner will refuse it if this letter "
                     "never published in the attempt" % c.out)
    plan["resume"] = ck[1] if (a.rejoin and ck and not a.from_snapshot) else None
    # relay (the one path every node needs)
    if not a.skip_relay_check:
        body, err = relay_health(card["anchor"])
        if body is None:
            problems.append("the relay %s does not answer /health (%s): start later; a node cannot train without it"
                            % (card["anchor"], err))
        else:
            if body.get("admission") not in (None, "open"):
                warns.append("relay admission is %r right now" % body.get("admission"))
            plan["relay"] = "ok, free %.1f GB" % (body.get("free_bytes", 0) / 1e9)
    # P2P
    mode, cfg = p2p_plan(c, a, problems, warns)
    if cfg is not None and os.environ.get("NEURAHASH_CONTENT_TOKEN", "").strip():
        err = p2p_dry_build(c, mode, cfg)
        if err:
            warns.append("P2P setup would fail in the runner (%s) -> this node falls back to --p2p force-relay: it "
                         "still trains, every fetch goes over the relay" % err)
            mode, cfg = "force-relay", None
    plan["p2p_mode"], plan["p2p_cfg"] = mode, cfg
    if not quiet:
        for w in warns:
            say("WARN " + w)
        for p in problems:
            say("FAIL " + p)
    return not problems, problems, warns, plan


# ------------------------------------------------------------------------------------------------ launch
def child_env(c, plan):
    env = dict(os.environ)
    popped = sorted(k for k in env if k.upper().startswith("NEURAHASH_") and k.upper() not in (
        "NEURAHASH_CONTENT_TOKEN", "NEURAHASH_TURN_USER", "NEURAHASH_TURN_PASSWORD", "NEURAHASH_P2P_SIGNAL_KEY"))
    for k in popped:
        env.pop(k)
    for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(k, None)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "USE_TF": "0", "USE_TORCH": "1",
                "TRANSFORMERS_NO_TF": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "NEURAHASH_SD_SHARD_DIR": c.base, "NEURAHASH_SD_CONFIG_DIR": os.path.join(c.base, "config"),
                "NEURAHASH_SD_LAYER_PACK_DIR": c.packs, "NEURAHASH_ST_READ_THREADS": "4",
                "NEURAHASH_ST_PREFETCH": "1"})
    for k, v in (c.card.get("env") or {}).items():          # the operator may tune the reader, nothing else
        if re.match(r"^NEURAHASH_ST_[A-Z_]+$", k) and re.match(r"^[A-Za-z0-9_.:/\\-]{0,200}$", str(v)):
            env[k] = str(v)
    return env, popped


def build_argv(c, plan):
    av = list(c.card["argv_common"])
    if plan["p2p_mode"] == "force-relay":
        for i, x in enumerate(av):
            if x == "--p2p" and i + 1 < len(av):
                av[i + 1] = "force-relay"
            elif x.startswith("--p2p="):
                av[i] = "--p2p=force-relay"
    if argv_value(av, "--anchor") is None:
        av += ["--anchor", str(c.card["anchor"]).rstrip("/")]     # every node must talk to the same relay
    argv = [sys.executable, "-u", RUNNER] + av + [
        "--node", c.node, "--config-dir", os.path.join(c.base, "config"), "--pack-dir", c.packs, "--data", c.corpus,
        "--out-dir", c.out, "--vram-cap", "%g" % plan["vram_cap"]]
    if plan["pin_layers"]:
        argv += ["--pin-layers", str(plan["pin_layers"])]
    if argv_value(av, "--part-chunk-mib"):
        argv += ["--init-cache-shared", c.init_cache]
    if plan.get("p2p_cfg_path"):
        argv += ["--p2p-config", plan["p2p_cfg_path"]]
    if plan.get("rejoin"):
        argv += ["--rejoin"]
        if plan.get("resume"):
            argv += ["--resume-from", plan["resume"]]
    return argv


class NoSleep(object):
    """Windows: keep the PC awake while the node runs (a sleeping node can stop the whole run)."""
    def __enter__(self):
        if os.name == "nt":
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # CONTINUOUS|SYSTEM_REQUIRED
            except Exception:                                             # noqa: BLE001
                pass
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:                                             # noqa: BLE001
                pass


def kill_tree(p):
    if p.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            p.kill()
    for _ in range(30):
        if p.poll() is not None:
            return
        time.sleep(1)


EXIT_MEANING = {0: "the run completed", 2: "setup refused (see the FAIL/ABORT line above)",
                3: "VOID (the run's agreement or a registered check failed)",
                4: "a resumable stop or pause: re-enter later with --rejoin when the operator says so",
                5: "ABORT (transport or integrity failure, or a peer died)",
                7: "the relay refused this node's key (revoked, expired, outside its scope, or a quota): the run goes "
                   "on without this node; get a new key from the operator, then re-enter with --rejoin"}
NO_RELAUNCH = (0, 2, 3, 4, 7)       # --auto-rejoin never relaunches these (7: the same key would be refused again)


def cmd_run(c, a):
    ok, problems, warns, plan = preflight(c, a)
    if not ok:
        fail("%d preflight check(s) failed; nothing was started (a node that fails at start can void the run for "
             "everyone)" % len(problems))
    plan["rejoin"] = bool(a.rejoin)
    if plan["p2p_cfg"]:
        p = os.path.join(c.out, "p2p_config_node.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(plan["p2p_cfg"], f, indent=1)
        plan["p2p_cfg_path"] = p
    env, popped = child_env(c, plan)
    lock = os.path.join(c.out, "NODE.lock")
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip() or 0)
        except (OSError, ValueError):
            pid = 0
        if pid and _alive(pid):
            fail("another glm_node.py (pid %d) runs this node already (%s)" % (pid, lock))
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    say("node %s of %s (attempt %s) | p2p %s | vram cap %.2f GiB | out %s" % (
        c.node, c.card["ns"], c.card["attempt"], plan["p2p_mode"], plan["vram_cap"], c.out))
    if popped:
        say("cleared from the child's environment: %s" % ", ".join(popped))
    rc = None
    try:
        relaunches = 0
        while True:
            argv = build_argv(c, plan)
            say("argv: %s" % " ".join(argv[2:]))
            if a.dry_run:
                say("--dry-run: not started")
                return 0
            rc, stopped_by_user = launch_once(c, argv, env)
            say("the node exited with code %s: %s" % (rc, EXIT_MEANING.get(rc, "killed or crashed")))
            if stopped_by_user or rc in NO_RELAUNCH or not a.auto_rejoin or relaunches >= a.auto_rejoin:
                break
            relaunches += 1
            say("--auto-rejoin %d of %d: re-entering in 120 s (Ctrl+C to cancel), after the full preflight again"
                % (relaunches, a.auto_rejoin))
            time.sleep(120)
            body, err = relay_health(c.card["anchor"], tries=6, wait=60)
            if body is None:
                say("the relay is still unreachable (%s): not relaunching" % err)
                break
            a.rejoin = True
            ok2, problems2, _w2, plan = preflight(c, a)
            if not ok2:
                say("not relaunching: %d preflight check(s) fail now" % len(problems2))
                break
            plan["rejoin"] = True
            if plan["p2p_cfg"]:
                plan["p2p_cfg_path"] = os.path.join(c.out, "p2p_config_node.json")
                with open(plan["p2p_cfg_path"], "w", encoding="utf-8") as f:
                    json.dump(plan["p2p_cfg"], f, indent=1)
            say("re-entering from %s" % (plan["resume"] or "the run's newest snapshot"))
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass
    if rc == 7:
        say("set your NEW key in NEURAHASH_CONTENT_TOKEN, then re-enter this run: python glm-node/glm_node.py run "
            "--card %s --rejoin" % a.card)
    elif rc not in (0, 3):
        say("to re-enter this run: python glm-node/glm_node.py run --card %s --rejoin" % a.card)
    return rc if isinstance(rc, int) and rc >= 0 else 1


ROUND_LINE = re.compile(br"^\s*(?:\[[^\]]*\]\s*)?round\s+(\d+)\s+\|")


def launch_once(c, argv, env):
    """Run the node once. -> (exit code, stopped_by_user). Ctrl+C once: leave at the next ROUND BOUNDARY (after this
    node's next `round N |` line and its round checkpoint); Ctrl+C again: leave now. Either way the process tree is
    killed -- for the run a clean leave (no marker is published), and the node can come back with --rejoin."""
    kw = {"creationflags": 0x00000200} if os.name == "nt" else {"start_new_session": True}   # own process group
    logf = open(os.path.join(c.out, CONSOLE_LOG), "ab")
    state = {"boundary": False, "now": False, "round": None, "user": False}
    try:
        with NoSleep():
            p = subprocess.Popen(argv, env=env, cwd=RUNNER_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 **kw)

            def on_int(signum, frame):
                if state["boundary"]:
                    say("Ctrl+C again: stopping the node NOW (rejoin later with --rejoin)")
                    state["now"] = True
                    return
                state["boundary"], state["user"] = True, True
                first = newest_round_ckpt(c.out, c.node) is None
                say("Ctrl+C: the node will leave at the next round boundary (Ctrl+C again = leave now).%s" % (
                    " WARNING: no round has finished on this node yet -- leaving before the first round can void "
                    "the attempt; tell the operator." if first else ""))
            prev = signal.signal(signal.SIGINT, on_int)
            prev_b = signal.signal(signal.SIGBREAK, on_int) if hasattr(signal, "SIGBREAK") else None
            # The pipe is drained into the log file by one thread and shown on the console by ANOTHER, through a
            # bounded queue: a console that blocks (Windows QuickEdit selection pauses output) must never
            # back-pressure the training process. Lines the console cannot take are dropped from the screen, never
            # from the file.
            import queue
            shown = queue.Queue(maxsize=2000)

            def pump():
                for line in iter(p.stdout.readline, b""):
                    logf.write(line)
                    logf.flush()
                    m = ROUND_LINE.match(line)
                    if m:
                        state["round"] = int(m.group(1))
                    try:
                        shown.put_nowait(line)
                    except queue.Full:
                        pass
                shown.put(None)

            def show():
                while True:
                    line = shown.get()
                    if line is None:
                        return
                    try:
                        sys.stdout.write(line.decode("utf-8", "replace"))
                        sys.stdout.flush()
                    except (OSError, UnicodeError, ValueError):
                        pass
            t = threading.Thread(target=pump, daemon=True)
            t.start()
            threading.Thread(target=show, daemon=True).start()
            seen_round = None
            while p.poll() is None:
                time.sleep(1)
                if state["now"]:
                    kill_tree(p)
                    break
                if state["boundary"]:
                    if seen_round is None:
                        seen_round = state["round"] if state["round"] is not None else -1
                    elif state["round"] is not None and state["round"] != seen_round:
                        r = state["round"]
                        ck = os.path.join(c.out, "round_ckpt_%s_r%04d.pt" % (c.node, r))
                        say("round %d done: leaving after its checkpoint is written" % r)
                        _wait_stable(ck, 180, p)        # the LAST round writes no checkpoint: the node then exits
                        kill_tree(p)
                        break
            t.join(10)
            signal.signal(signal.SIGINT, prev)
            if prev_b is not None:
                signal.signal(signal.SIGBREAK, prev_b)
            return p.returncode, state["user"]
    finally:
        logf.close()


def _wait_stable(path, limit_s, proc=None):
    """Wait until `path` exists and its size has not changed for 5 s, the process exits, or limit_s passed."""
    t0, last, since = time.time(), None, time.time()
    while time.time() - t0 < limit_s:
        if proc is not None and proc.poll() is not None:
            return True
        size = os.path.getsize(path) if os.path.isfile(path) else None
        if size != last:
            last, since = size, time.time()
        elif size is not None and time.time() - since >= 5:
            return True
        time.sleep(1)
    return False


def _alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"], capture_output=True, encoding="utf-8",
                             errors="replace")
        return re.search(r"\b%d\b" % pid, out.stdout or "") is not None
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ------------------------------------------------------------------------------------------------ other commands
def cmd_preflight(c, a):
    ok, problems, warns, plan = preflight(c, a)
    say("plan: node %s | p2p %s | vram cap %.2f | gpu %s | relay %s" % (c.node, plan.get("p2p_mode"),
                                                                       plan.get("vram_cap", 0), plan.get("gpu"),
                                                                       plan.get("relay", "not checked")))
    say("PREFLIGHT %s (%d problem(s), %d warning(s))" % ("PASS" if ok else "FAIL", len(problems), len(warns)))
    return 0 if ok else 1


def cmd_keygen(c, a):
    pub, msg = sidecar_pubkey(c.keyfile, create=True)
    if pub is None:
        fail(msg)
    say("signing key %s (%s). Send the operator THIS line (it is public; the key file stays on this PC):" % (
        c.keyfile, msg))
    print("PUBKEY %s" % pub)
    return 0


def tail(path, n=40):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200000))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def cmd_status(c, a):
    log = os.path.join(c.out, "trainer_arm_%s.log" % c.node)
    lines = tail(log, 400)
    rounds = [ln for ln in lines if re.search(r"\bround\s+\d+\s+\|", ln)]
    marks = [ln for ln in lines if ln.lstrip().startswith(("==== ", "!!!!")) or "REJOIN" in ln or "LATE:" in ln]
    say("out %s" % c.out)
    say("log %s (%s)" % (log, "present" if lines else "missing"))
    ck = newest_round_ckpt(c.out, c.node)
    say("newest round checkpoint: %s" % (("r%04d " % ck[0]) + ck[1] if ck else "none"))
    for ln in rounds[-5:]:
        print("  " + ln[:220])
    for ln in marks[-8:]:
        print("  " + ln[:220])
    for name in ("xport.jsonl", "transport.jsonl"):
        p = os.path.join(c.out, name)
        if os.path.isfile(p):
            kinds = {}
            for ln in open(p, encoding="utf-8", errors="replace"):
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                k = "%s/%s" % (r.get("kind", r.get("op", "?")), r.get("result", r.get("status", r.get("outcome", "-"))))
                kinds[k] = kinds.get(k, 0) + 1
            say("%s: %s" % (name, ", ".join("%s %d" % kv for kv in sorted(kinds.items())[:12])))
    for ln in lines[-12:]:
        print("  " + ln[:220])
    return 0


def cmd_logs(c, a):
    names = ["trainer_arm_%s.log" % c.node, "TRAIN.json", "TRAIN_%s.json" % c.node, "ATTEMPTS.txt", "xport.jsonl",
             "transport.jsonl", CONSOLE_LOG, "p2p_config_node.json", os.path.join("p2p", "sidecar_events.jsonl"),
             os.path.join("p2p", "sidecar.stderr.log")]
    zp = os.path.join(c.home, "glmnode_logs_%s_%s_%s.zip" % (c.card["ns"], c.node, time.strftime("%Y%m%d_%H%M%S")))
    n = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for nm in names:
            p = os.path.join(c.out, nm)
            if not os.path.isfile(p):
                continue
            if nm == "p2p_config_node.json":                 # the TURN user name is per miner: send it redacted
                try:
                    doc = json.load(open(p, encoding="utf-8"))
                    if (doc.get("sidecar") or {}).get("turn_user"):
                        doc["sidecar"]["turn_user"] = "<set>"
                    z.writestr(nm, json.dumps(doc, indent=1))
                    n += 1
                except (OSError, ValueError):
                    pass
                continue
            z.write(p, nm.replace("\\", "/"))
            n += 1
        sc = os.path.join(c.home, "selfcheck.json")
        if os.path.isfile(sc):
            z.write(sc, "selfcheck.json")
            n += 1
    say("wrote %s (%d files; no checkpoints, no keys, no tokens). Send it to the operator." % (zp, n))
    return 0


def cmd_card(c, a):
    """OPERATOR helper: turn a node command line printed by the operator's kit into a run card for one slot."""
    raw = open(a.cmdline_file, encoding="utf-8").read().strip()
    toks = raw.replace('"', " ").split()
    try:
        start = next(i for i, t in enumerate(toks) if t == "--mode")
    except StopIteration:
        fail("no --mode in %s" % a.cmdline_file)
    av, i = [], start
    while i < len(toks):
        t = toks[i]
        if re.match(r"^(\d?>>?|<|\||&&?|;|2>&1)", t) or t == "disown":
            break                                    # the kit's shell tail: `>> log 2>&1 & disown`
        if t in NODE_LOCAL_FLAGS:
            i += 1 + NODE_LOCAL_FLAGS[t]
            continue
        av.append(t)
        i += 1
    card = {"card_version": 1, "runner_sha256": sha256_file(RUNNER), "p2p_transport_sha256": sha256_file(TRANSPORT),
            "anchor": argv_value(av, "--anchor") or a.anchor, "ns": argv_value(av, "--ns"),
            "attempt": int(argv_value(av, "--attempt") or 1), "nodes": int(argv_value(av, "--nodes") or 2),
            "node": a.node.upper(), "vram_cap_gb": a.vram_cap, "pin_layers": 0, "argv_common": av,
            "p2p": {"ice_policy": "relay", "turn": [a.turn] if a.turn else [], "stun": [], "signal_key": "relay_token",
                    "exe_sha256": {platform_key(): sha256_file(SIDECAR_EXE)} if os.path.isfile(SIDECAR_EXE) else {},
                    "peer_pubkeys": {}}}
    with open(a.out_card, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=1)
    say("wrote %s for node %s (%d common args). Fill in p2p.peer_pubkeys before sending it." % (
        a.out_card, card["node"], len(av)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="glm-node launcher (see the module docstring)")
    ap.add_argument("command", choices=("run", "preflight", "keygen", "status", "logs", "card"))
    ap.add_argument("--card", default="", help="the run card (JSON) from the operator")
    ap.add_argument("--home", default="")
    ap.add_argument("--packs", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default="", help="this node's output dir (default <home>/runs/<ns>/<node>)")
    ap.add_argument("--key-file", default="", help="P2P signing key (default <home>/keys/p2p_sign.key)")
    ap.add_argument("--node", default="", help="your node letter (else NEURAHASH_NODE, else the card)")
    ap.add_argument("--vram-cap", type=float, default=None, help="GiB (default: the card's, else 6.5)")
    ap.add_argument("--pin-layers", type=int, default=None)
    ap.add_argument("--rejoin", action="store_true", help="re-enter the same run (from the newest round checkpoint)")
    ap.add_argument("--from-snapshot", action="store_true", help="with --rejoin: ignore own checkpoints, rejoin from "
                                                                 "the run's newest snapshot (when away > ~30 rounds)")
    ap.add_argument("--relay-only", action="store_true", help="never use P2P on this node (--p2p force-relay)")
    ap.add_argument("--dry-run", action="store_true", help="run: print the plan and argv, start nothing")
    ap.add_argument("--auto-rejoin", type=int, default=0, metavar="N",
                    help="run: after a crash (not after your stop, a VOID, a setup refusal, a pause or a refused key) "
                         "re-enter the run with --rejoin, up to N times, 2 minutes apart, while the relay answers")
    ap.add_argument("--allow-version-mismatch", action="store_true")
    ap.add_argument("--skip-relay-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--i-am-the-operator", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cmdline-file", default="", help="card: a node command line from the operator's kit")
    ap.add_argument("--out-card", default="run_card.json")
    ap.add_argument("--anchor", default="http://47.84.93.96:8710")
    ap.add_argument("--turn", default="turn:47.84.93.96:3478?transport=tcp")
    a = ap.parse_args(argv)
    if a.command == "card":
        if not a.cmdline_file or not a.node:
            fail("card needs --cmdline-file and --node")
        return cmd_card(make_ctx(a), a)
    c = make_ctx(a)
    if a.command == "keygen":
        return cmd_keygen(c, a)
    if not c.card:
        fail("%s needs --card <run card from the operator>" % a.command)
    return {"run": cmd_run, "preflight": cmd_preflight, "status": cmd_status, "logs": cmd_logs}[a.command](c, a)


if __name__ == "__main__":
    sys.exit(main())
