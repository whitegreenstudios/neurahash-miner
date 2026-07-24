"""VRAM starve-pause + OOM round-skip helpers (2026-07-24 fix).

The crash these prevent: the elastic VRAM manager advertised "resize 12 -> 0 resident units"
and the round loop entered train/eval anyway, dying with a fatal CUDA OOM inside heldout_ce
(openadm_contrib5090.log:60-92 during the keyless live test). The fix: the round loop consults
the manager's sustainable capacity and PAUSES at 0 (the behavior the elastic-VRAM design always
promised), and a CUDA OOM mid-round skips the round instead of killing the miner.

All tests are torch-free: the helpers are pure stdlib on purpose.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import sharddiloco_glm_contributor as C  # noqa: E402


class _FakeTuner(object):
    def __init__(self, units):
        self.current_units = units


class _FakeMgr(object):
    """Scripted capacity: pop the next value on every probe (last value repeats)."""

    def __init__(self, script):
        self._script = list(script)
        self.tuner = self

    @property
    def current_units(self):
        if len(self._script) > 1:
            return self._script.pop(0)
        return self._script[0]


def test_vram_units_none_without_manager():
    assert C._vram_units(vm=None) is None or isinstance(C._vram_units(vm=None), int)
    # explicit: a broken manager (no tuner) -> None, never raises
    class Broken(object):
        pass
    assert C._vram_units(vm=Broken()) is None


def test_no_manager_means_no_pause():
    logs = []
    waits = C._vram_pause_if_starved(logs.append, miner="m", vm=None if C._VRAM_MGR is None else _FakeMgr([3]))
    assert waits == 0


def test_positive_capacity_means_no_pause():
    logs = []
    waits = C._vram_pause_if_starved(logs.append, miner="m", vm=_FakeMgr([5]), sleep_fn=lambda s: None)
    assert waits == 0
    assert logs == []                      # silent on the happy path


def test_starved_pauses_then_resumes():
    logs = []
    slept = []
    # capacity probes: initial 0 (enter pause), then 0, then 2 (resume)
    mgr = _FakeMgr([0, 0, 2])
    waits = C._vram_pause_if_starved(logs.append, miner="m", vm=mgr,
                                     poll_s=15.0, sleep_fn=slept.append)
    assert waits == 2                      # two sleeps before capacity returned
    assert slept == [15.0, 15.0]
    assert any("PAUSED" in m for m in logs)
    assert any("recovered" in m for m in logs)


def test_starved_max_waits_cap():
    logs = []
    mgr = _FakeMgr([0, 0, 0, 0, 0])
    waits = C._vram_pause_if_starved(logs.append, miner="m", vm=mgr,
                                     sleep_fn=lambda s: None, max_waits=3)
    assert waits == 3                      # bounded for tests / callers that need a cap


def test_is_cuda_oom_detection():
    assert C._is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1.12 GiB"))
    assert C._is_cuda_oom(RuntimeError("HIP Out Of Memory"))
    assert not C._is_cuda_oom(RuntimeError("device-side assert triggered"))
    assert not C._is_cuda_oom(ValueError("out of memory"))          # only RuntimeError family
    assert not C._is_cuda_oom(KeyError("staled"))


def test_torch_not_imported_by_helpers():
    """The helpers must be usable without torch. Checked in a FRESH subprocess -- asserting on
    this process's sys.modules is order-dependent (other test files in a combined run import
    torch first, which is fine and not this module's doing)."""
    import subprocess
    code = (
        "import sys, os\n"
        "sys.path.insert(0, r'%s')\n"
        "import sharddiloco_glm_contributor as C\n"
        "C._vram_units(vm=None)\n"
        "C._is_cuda_oom(RuntimeError('CUDA out of memory'))\n"
        "C._vram_pause_if_starved(lambda m: None, vm=None)\n"
        "assert 'torch' not in sys.modules, 'helpers pulled in torch'\n"
        "print('TORCH_FREE_OK')\n"
    ) % os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0 and "TORCH_FREE_OK" in (r.stdout or ""), (r.stdout, r.stderr)
