"""tools/run_glm_miner.py must stay runnable from a fresh clone (CPU-only: no GPU, no network).

WHY THIS EXISTS: run_glm_miner.py is the keep-alive supervisor a public miner (e.g. an external
contributor cloning `main`) launches and leaves running. If its CLI stops parsing, if the known-good
child environment stops being set, or if a missing sibling stops being reported cleanly, the miner is
dead on a stranger's box -- the 3.7.1 class of break. These checks are fast and touch no GPU and no
network.

Run: C:/Python313/python.exe -m pytest tests/test_run_glm_miner.py -q
"""
import importlib.util
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SCRIPT = os.path.join(_REPO, "tools", "run_glm_miner.py")


def _load_module():
    """Import tools/run_glm_miner.py by path. It only defines functions; main() is __main__-guarded,
    so importing it launches nothing."""
    spec = importlib.util.spec_from_file_location("run_glm_miner", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_help_exits_zero():
    """A fresh clone must be able to run `python tools/run_glm_miner.py --help` and get exit 0.
    Uses sys.executable so the check is portable: it is exactly the interpreter running this suite
    (C:/Python313/python.exe when the release gate runs it here; the miner's own python on a joiner)."""
    assert os.path.exists(_SCRIPT), "tools/run_glm_miner.py is not in this tree"
    r = subprocess.run([sys.executable, _SCRIPT, "--help"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 0, "run_glm_miner.py --help exited %s:\n%s" % (r.returncode, r.stdout)
    assert "run_glm_miner" in r.stdout or "supervisor" in r.stdout.lower()


def test_child_env_sets_torch_backend_and_encoding(monkeypatch):
    """The supervisor must hand the miner the known-good env: USE_TF=0 (avoids the stray-TensorFlow
    'numpy.core.umath failed to import' death) and an explicit PYTHONIOENCODING (a cp1252 console
    otherwise kills non-ASCII child log lines). Clear these first so we test the supervisor's own
    defaults, not whatever the ambient environment happens to carry."""
    for var in ("USE_TF", "USE_TORCH", "TRANSFORMERS_NO_TF", "PYTHONIOENCODING", "NEURAHASH_VRAM_CAP_GB"):
        monkeypatch.delenv(var, raising=False)
    mod = _load_module()
    env = mod._child_env(None)
    assert env["USE_TF"] == "0"
    assert env["USE_TORCH"] == "1"
    assert env.get("PYTHONIOENCODING")  # set to a real encoding, not empty/absent
    assert "NEURAHASH_VRAM_CAP_GB" not in env  # no cap requested -> none injected
    capped = mod._child_env(6.5)
    assert capped["NEURAHASH_VRAM_CAP_GB"] == "6.5"


def test_missing_contributor_returns_rc_2(monkeypatch):
    """If the contributor sibling is absent (wrong cwd, or a broken clone), the supervisor must fail
    fast with rc 2 rather than launch nothing silently. This branch is reached before any self-update,
    base fetch, or CUDA call, so the test needs no GPU and no network."""
    mod = _load_module()
    monkeypatch.setattr(mod, "CONTRIB", os.path.join(_REPO, "tools", "__no_such_contributor__.py"))
    monkeypatch.setattr(sys, "argv", ["run_glm_miner.py"])
    assert mod.main() == 2
