"""glm_node.py: the run card -> runner argv contract, and the fall-back-to-relay rules. CPU only, no network.

The last test asks the REAL runner to parse the launcher's argv (`--plan`: reads config.json, loads nothing). It
needs the GLM-4.7-Flash config.json: set GLM_NODE_TEST_CONFIG_DIR, or have <home>/base/config from build_packs.py.

Run: python -m pytest glm-node/tests/test_glm_node_launcher.py -q
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import glm_node as GN  # noqa: E402

EXAMPLE = os.path.join(ROOT, "examples", "run_card.example.json")


def ctx(tmp_path, node="C"):
    c = GN.Ctx()
    c.home = str(tmp_path)
    c.packs, c.base = os.path.join(c.home, "packs"), os.path.join(c.home, "base")
    c.corpus = os.path.join(c.home, "corpus", "e5_scimc_train.jsonl")
    c.keyfile = os.path.join(c.home, "keys", "p2p_sign.key")
    c.init_cache = os.path.join(c.home, "init_cache")
    c.card = GN.load_card(EXAMPLE)
    c.node = node
    c.out = os.path.join(c.home, "runs", c.card["ns"], node)
    return c


class Args(object):
    relay_only = False


def test_the_example_card_is_valid():
    card = GN.load_card(EXAMPLE)
    probs = []
    GN.check_argv_common(card, probs)
    assert probs == [], probs
    assert card["runner_sha256"] == GN.sha256_file(GN.RUNNER), "the example card names another runner"


def test_node_local_and_foreign_flags_in_a_card_are_refused(tmp_path):
    card = json.load(open(EXAMPLE, encoding="utf-8"))
    card["argv_common"] = card["argv_common"] + ["--node", "B", "--selftest"]
    p = tmp_path / "card.json"
    p.write_text(json.dumps(card), encoding="utf-8")
    probs = []
    GN.check_argv_common(GN.load_card(str(p)), probs)
    assert any("--node" in x for x in probs) and any("--selftest" in x for x in probs), probs


def test_a_card_whose_ns_disagrees_with_its_argv_is_refused(tmp_path):
    card = json.load(open(EXAMPLE, encoding="utf-8"))
    card["ns"] = "another_run"
    p = tmp_path / "card.json"
    p.write_text(json.dumps(card), encoding="utf-8")
    probs = []
    GN.check_argv_common(GN.load_card(str(p)), probs)
    assert any("--ns" in x for x in probs), probs


def test_argv_gets_the_node_local_flags_and_the_relay_only_step_down(tmp_path):
    c = ctx(tmp_path)
    argv = GN.build_argv(c, {"p2p_mode": "force-relay", "vram_cap": 6.5, "pin_layers": 0})
    tail = argv[3:]
    assert GN.argv_value(tail, "--p2p") == "force-relay" and "auto" not in tail[tail.index("--p2p"):][:2]
    assert GN.argv_value(tail, "--node") == "C" and GN.argv_value(tail, "--vram-cap") == "6.5"
    assert GN.argv_value(tail, "--anchor") == "http://47.84.93.96:8710"
    assert GN.argv_value(tail, "--init-cache-shared") == c.init_cache
    assert "--p2p-config" not in tail and "--rejoin" not in tail
    for flag in GN.NODE_LOCAL_FLAGS:
        assert tail.count(flag) <= 1, flag


def test_rejoin_resumes_from_the_newest_round_checkpoint(tmp_path):
    c = ctx(tmp_path)
    os.makedirs(c.out)
    for r in (3, 11, 7):
        open(os.path.join(c.out, "round_ckpt_C_r%04d.pt" % r), "wb").close()
    open(os.path.join(c.out, "round_ckpt_B_r0099.pt"), "wb").close()          # another letter's: ignored
    best = GN.newest_round_ckpt(c.out, "C")
    assert best[0] == 11
    argv = GN.build_argv(c, {"p2p_mode": "auto", "vram_cap": 6.5, "pin_layers": 0, "rejoin": True,
                             "resume": best[1]})
    assert "--rejoin" in argv and GN.argv_value(argv, "--resume-from") == best[1]


def test_p2p_falls_back_to_relay_without_turn_credentials(tmp_path, monkeypatch):
    for k in ("NEURAHASH_TURN_USER", "NEURAHASH_TURN_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    c = ctx(tmp_path)
    warns = []
    mode, cfg = GN.p2p_plan(c, Args(), [], warns)
    assert mode == "force-relay" and cfg is None
    assert any("NEURAHASH_TURN_USER" in w for w in warns), warns


def test_relay_only_is_force_relay_never_off(tmp_path):
    """`--p2p off` would drop the transport from this node's init_meta while the fleet's has it: the run would
    refuse the node at step 0. The relay-only fallback must therefore be force-relay."""
    c = ctx(tmp_path)
    a = Args()
    a.relay_only = True
    mode, cfg = GN.p2p_plan(c, a, [], [])
    assert mode == "force-relay" and cfg is None


def test_the_childs_environment_is_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURAHASH_VRAM_CAP_GB", "7.4")              # a machine-wide leftover hid a bug once
    monkeypatch.setenv("NEURAHASH_CONTENT_TOKEN", "x" * 8)
    monkeypatch.setenv("PYTHONPATH", "C:/somewhere")
    c = ctx(tmp_path)
    env, popped = GN.child_env(c, {})
    assert "NEURAHASH_VRAM_CAP_GB" in popped and "NEURAHASH_VRAM_CAP_GB" not in env
    assert env["NEURAHASH_CONTENT_TOKEN"] == "x" * 8 and "PYTHONPATH" not in env
    assert env["NEURAHASH_SD_LAYER_PACK_DIR"] == c.packs and env["NEURAHASH_ST_READ_THREADS"] == "4"


def _config_dir():
    d = os.environ.get("GLM_NODE_TEST_CONFIG_DIR", "")
    if not d:
        home = os.environ.get("NEURAHASH_GLM_HOME") or os.path.join(os.path.expanduser("~"), "glmnode")
        d = os.path.join(home, "base", "config")
    return d if os.path.isfile(os.path.join(d, "config.json")) else None


def test_the_real_runner_accepts_the_launchers_argv(tmp_path):
    cfg = _config_dir()
    if cfg is None:
        pytest.skip("no GLM-4.7-Flash config.json (set GLM_NODE_TEST_CONFIG_DIR)")
    c = ctx(tmp_path)
    argv = GN.build_argv(c, {"p2p_mode": "force-relay", "vram_cap": 6.5, "pin_layers": 0})
    i = argv.index("--config-dir")
    argv[i + 1] = cfg
    env = dict(os.environ, PYTHONIOENCODING="utf-8", NEURAHASH_SD_CONFIG_DIR=cfg)
    out = subprocess.run(argv + ["--plan"], capture_output=True, encoding="utf-8", errors="replace", timeout=300,
                         env=env, cwd=GN.RUNNER_DIR)
    assert out.returncode == 0, (out.stdout[-1500:], out.stderr[-1500:])
    assert "GLM47-H1X PLAN | mode arm | node C" in out.stdout, out.stdout[:800]


FAKE_NODE = """
import os, sys, time
out = sys.argv[1]
print("[10-15 08:00:00]   round   1 | all-in | loss 1.0", flush=True)
time.sleep(4)
print("[10-15 08:00:04]   round   2 | all-in | loss 0.9", flush=True)
open(os.path.join(out, "round_ckpt_C_r0002.pt"), "wb").write(b"x" * 16)
time.sleep(90)
print("SHOULD NOT BE REACHED", flush=True)
"""


def _fake_run(tmp_path, presses):
    """Run launch_once on a fake node; `presses` = seconds after the first round line at which Ctrl+C is sent."""
    import signal
    import threading
    import time
    c = ctx(tmp_path)
    os.makedirs(c.out)
    script = tmp_path / "fake_node.py"
    script.write_text(FAKE_NODE, encoding="utf-8")
    log = os.path.join(c.out, GN.CONSOLE_LOG)

    def presser():
        t0 = time.time()
        while time.time() - t0 < 30:
            if os.path.isfile(log) and b"round   1" in open(log, "rb").read():
                break
            time.sleep(0.2)
        start = time.time()
        for at in presses:
            while time.time() - start < at:
                time.sleep(0.1)
            signal.raise_signal(signal.SIGINT)
    threading.Thread(target=presser, daemon=True).start()
    t0 = time.time()
    rc, user = GN.launch_once(c, [sys.executable, "-u", str(script), c.out], dict(os.environ))
    return rc, user, time.time() - t0, open(log, "rb").read()


def test_ctrl_c_once_leaves_at_the_next_round_boundary(tmp_path):
    rc, user, secs, out = _fake_run(tmp_path, [0.5])
    assert user and rc != 0, (rc, user)
    assert b"round   2" in out and b"SHOULD NOT BE REACHED" not in out
    assert 8 <= secs < 60, secs                         # waited for round 2 + a stable checkpoint, not 90 s


def test_ctrl_c_twice_leaves_at_once(tmp_path):
    rc, user, secs, out = _fake_run(tmp_path, [0.3, 1.0])
    assert user and rc != 0, (rc, user)
    assert b"round   2" not in out, "the second Ctrl+C must not wait for the round boundary"
    assert secs < 15, secs


def test_the_duplicate_launcher_lock_sees_a_live_process():
    """NODE.lock is only a guard if a live pid is recognised (two processes on one letter void the run)."""
    assert GN._alive(os.getpid()) is True
    assert GN._alive(2 ** 22 + 12345) is False


def _p2p_ready_ctx(tmp_path, monkeypatch):
    """A ctx whose card is P2P-complete: a real signing key from the shipped sidecar, TURN creds and a token in env."""
    if not os.path.isfile(GN.SIDECAR_EXE):
        pytest.skip("no sidecar binary for this OS")
    c = ctx(tmp_path)
    c.keyfile = str(tmp_path / "keys" / "p2p_sign.key")
    pub, msg = GN.sidecar_pubkey(c.keyfile, create=True)
    if pub is None:
        pytest.skip("the sidecar cannot run here (%s)" % msg)
    c.card["p2p"]["peer_pubkeys"] = {"A": "a" * 64, "B": "b" * 64, "C": pub}
    os.makedirs(c.out, exist_ok=True)
    for k, v in (("NEURAHASH_TURN_USER", "tester"), ("NEURAHASH_TURN_PASSWORD", "not-a-real-password"),
                 ("NEURAHASH_CONTENT_TOKEN", "not-a-real-token"),
                 ("NEURAHASH_P2P_SIGNAL_KEY", "0f" * 32)):     # the example card says signal_key "env" (per-run key)
        monkeypatch.setenv(k, v)
    return c


def test_a_complete_p2p_card_builds_in_the_runners_own_factory(tmp_path, monkeypatch):
    c = _p2p_ready_ctx(tmp_path, monkeypatch)
    warns = []
    mode, cfg = GN.p2p_plan(c, Args(), [], warns)
    assert mode == "auto" and cfg is not None, warns
    assert GN.p2p_dry_build(c, mode, cfg) is None


def test_what_would_abort_the_runner_steps_down_to_relay_instead(tmp_path, monkeypatch):
    """The runner's own transport factory rejects a bad exe sha256 / a missing secret (the Stage-0 runner then falls
    back to relay-only under --p2p auto, and refuses under --p2p required); the launcher must catch both in its dry
    build and run --p2p force-relay, so the step-down is explicit and logged before the node starts."""
    c = _p2p_ready_ctx(tmp_path, monkeypatch)
    mode, cfg = GN.p2p_plan(c, Args(), [], [])
    bad = json.loads(json.dumps(cfg))
    bad["sidecar"]["exe_sha256"] = "0" * 64
    assert "sha256" in (GN.p2p_dry_build(c, mode, bad) or ""), "a wrong exe sha256 was not caught"
    monkeypatch.delenv("NEURAHASH_TURN_PASSWORD")
    err = GN.p2p_dry_build(c, mode, cfg)
    assert err and "NEURAHASH_TURN_PASSWORD" in err, err


def test_the_relay_only_step_down_also_rewrites_the_equals_form(tmp_path):
    c = ctx(tmp_path)
    c.card["argv_common"] = [("--p2p=auto" if x == "--p2p" else x) for x in c.card["argv_common"]]
    c.card["argv_common"].remove("auto")
    argv = GN.build_argv(c, {"p2p_mode": "force-relay", "vram_cap": 6.5, "pin_layers": 0})
    assert "--p2p=force-relay" in argv and "--p2p=auto" not in argv


def test_the_logs_zip_redacts_the_turn_user(tmp_path):
    import zipfile

    class A(object):
        pass
    c = ctx(tmp_path)
    os.makedirs(c.out)
    with open(os.path.join(c.out, "p2p_config_node.json"), "w", encoding="utf-8") as f:
        json.dump({"sidecar": {"turn_user": "per-miner-user-123", "turn_password_env": "NEURAHASH_TURN_PASSWORD"}}, f)
    with open(os.path.join(c.out, "trainer_arm_C.log"), "w", encoding="utf-8") as f:
        f.write("round   1 | all-in\n")
    GN.cmd_logs(c, A())
    zp = [os.path.join(c.home, n) for n in os.listdir(c.home) if n.endswith(".zip")][0]
    with zipfile.ZipFile(zp) as z:
        body = z.read("p2p_config_node.json").decode("utf-8")
        assert "per-miner-user-123" not in body and "<set>" in body
        assert "trainer_arm_C.log" in z.namelist()


def test_a_card_can_pin_the_whole_package_build(tmp_path):
    """The step-0 agreement compares the runner / p2p_transport shas only; a card's glm_node_manifest_sha256 pins every
    other shipped file too (libraries included). Absent = no pin; the right sha passes; another sha is a problem."""
    card = GN.load_card(EXAMPLE)
    assert GN.card_manifest_problem(card) is None
    card["glm_node_manifest_sha256"] = GN.sha256_file(GN.MANIFEST)
    assert GN.card_manifest_problem(card) is None
    card["glm_node_manifest_sha256"] = "0" * 64
    p = GN.card_manifest_problem(card)
    assert p and "MANIFEST.sha256" in p and "another glm-node build" in p, p


def test_a_refused_key_exit_7_is_explained_and_never_auto_relaunched(tmp_path, monkeypatch):
    """Stage-0 runner: exit 7 = the relay refused THIS node's key. The same key would be refused again, so
    --auto-rejoin stops there; a crash (exit 1) is still relaunched."""
    assert 7 in GN.EXIT_MEANING and "key" in GN.EXIT_MEANING[7] and 7 in GN.NO_RELAUNCH and 1 not in GN.NO_RELAUNCH
    plan = {"p2p_mode": "force-relay", "p2p_cfg": None, "vram_cap": 6.5, "pin_layers": 0, "resume": None}
    monkeypatch.setattr(GN, "preflight", lambda c_, a_, quiet=False: (True, [], [], dict(plan)))
    monkeypatch.setattr(GN, "relay_health", lambda *a_, **k_: ({"ok": True}, None))
    monkeypatch.setattr(GN.time, "sleep", lambda s: None)

    class A(object):
        auto_rejoin, rejoin, dry_run, card = 3, False, False, "card.json"
    for rcs, want_calls in (([7], 1), ([1, 7], 2)):
        c = ctx(tmp_path / ("r%d" % len(rcs)))
        os.makedirs(c.out)
        calls = []

        def fake_launch(c_, argv, env, rcs=rcs, calls=calls):
            calls.append(argv)
            return rcs[len(calls) - 1], False
        monkeypatch.setattr(GN, "launch_once", fake_launch)
        a = A()
        assert GN.cmd_run(c, a) == 7 and len(calls) == want_calls, (rcs, len(calls))
