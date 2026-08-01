"""The miner must SURVIVE an OOM and still make progress -- and must never trust an unverified cap.

Owner directive 2026-08-01: "make sure our miner must has a built in oom detection and verify like
the vram cap guard, so that it will never failed."

Detection and a cap already existed (tools/sharddiloco_glm_contributor.py: _is_cuda_oom :2355,
apply_vram_guard :2194). What did NOT exist, and what these tests pin down:

  1. BACKOFF. The old recovery freed the cache and retried the SAME batch/inner next round. For a
     DETERMINISTIC OOM that is an infinite skip loop -- the miner never crashes and never produces
     a contribution. Memory v332-oom-death-and-resume-verdict-2026-07-25 records that exact shape
     as 18 hours of downtime behind a self-heal that healed nothing. The property under test is
     therefore not "it survives" but "the next attempt DIFFERS from the one that just failed".

  2. CAP VERIFICATION. apply_vram_guard applied a fraction and printed a confident message without
     ever checking it took. Memory vram-cap-live-verified caveat 0: "the DiLoCo P0 smoke OOM'd
     twice while 'capped'". An unenforced cap is worse than none -- it spills to shared system RAM
     and hangs the host instead of failing the process.

No GPU and no torch install required: torch is injected into sys.modules as a fake, which also
guards the property that the helpers only ever READ sys.modules and never import torch themselves.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import sharddiloco_glm_contributor as C  # noqa: E402


class _Args(object):
    def __init__(self, batch=64, inner=8):
        self.batch, self.inner = batch, inner


@pytest.fixture(autouse=True)
def _clean_history():
    C._OOM_TRIED.clear()
    yield
    C._OOM_TRIED.clear()


# ----------------------------------------------------------------- 1. backoff

def test_backoff_reduces_batch_first():
    """Batch before inner: batch scales activation memory ~linearly and costs only throughput,
    whereas cutting inner steps cuts how much training actually happened."""
    a, logs = _Args(64, 8), []
    assert C._oom_backoff(a, logs.append, miner="m") is True
    assert a.batch == 32 and a.inner == 8
    assert any("BACKOFF" in m for m in logs)


def test_backoff_never_repeats_a_configuration_that_oomed():
    """THE core property. Retrying a config byte-identical to the one that just OOM'd is the
    infinite-skip-loop bug; every step must land on a configuration not previously attempted."""
    a, logs = _Args(64, 8), []
    seen = set()
    seen.add((a.batch, a.inner))
    for _ in range(64):
        if not C._oom_backoff(a, logs.append, miner="m"):
            break
        cfg = (a.batch, a.inner)
        assert cfg not in seen, "backoff re-proposed an already-failed config %r" % (cfg,)
        seen.add(cfg)
    assert len(seen) > 1


def test_backoff_terminates_at_the_floor_and_says_so():
    """It must not reduce forever, and hitting the floor must be LOUD -- a miner that quietly
    skips every round looks identical to a healthy idle one."""
    a, logs = _Args(64, 8), []
    for _ in range(200):
        if not C._oom_backoff(a, logs.append, miner="m"):
            break
    else:
        pytest.fail("backoff never terminated")
    assert a.batch == C._OOM_FLOOR_BATCH and a.inner == C._OOM_FLOOR_INNER
    assert any("AT THE FLOOR" in m for m in logs)
    assert any("NOT SILENT PROGRESS" in m for m in logs)


def test_backoff_at_floor_is_idempotent():
    """Repeated floor hits keep returning False instead of driving batch/inner below 1 (a zero or
    negative batch would raise something far more confusing than an OOM)."""
    a = _Args(1, 1)
    for _ in range(5):
        assert C._oom_backoff(a, lambda _m: None, miner="m") is False
    assert a.batch >= 1 and a.inner >= 1


# ------------------------------------------------- 2. VRAM cap verification

def _fake_torch(*, alloc_succeeds):
    """Minimal torch stand-in. `alloc_succeeds=True` simulates a build where the per-process
    fraction is NOT enforced -- the over-cap allocation goes through, which is the dangerous case."""
    t = types.ModuleType("torch")
    t.uint8 = "uint8"
    calls = {"fractions": []}

    def _empty(_n, dtype=None, device=None):
        if alloc_succeeds:
            return object()
        raise RuntimeError("CUDA out of memory. Tried to allocate 256.00 MiB")

    cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        is_available=lambda: True,
        current_device=lambda: 0,
        mem_get_info=lambda _i=0: (8 * 2 ** 30, 32 * 2 ** 30),
        set_per_process_memory_fraction=lambda f, i=0: calls["fractions"].append(f),
    )
    t.cuda = cuda
    t.empty = _empty
    t._calls = calls
    return t


def test_cap_probe_reports_NOT_enforced_when_allocation_succeeds(monkeypatch):
    """The failure this exists to catch: the ceiling is advisory, the alloc goes through, and the
    process would spill to shared system RAM instead of raising."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(alloc_succeeds=True))
    assert C._verify_vram_cap_enforced(0) is False


def test_cap_probe_reports_enforced_when_allocation_ooms(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(alloc_succeeds=False))
    assert C._verify_vram_cap_enforced(0) is True


def test_cap_probe_always_restores_a_full_fraction(monkeypatch):
    """The probe installs a deliberately tiny ceiling. If it ever failed to remove it, the real run
    would OOM at 64 MiB -- so the restore must happen on BOTH paths."""
    for succeeds in (True, False):
        fake = _fake_torch(alloc_succeeds=succeeds)
        monkeypatch.setitem(sys.modules, "torch", fake)
        C._verify_vram_cap_enforced(0)
        assert fake._calls["fractions"][-1] == 1.0, (
            "probe left a restrictive fraction installed (alloc_succeeds=%r)" % succeeds)


def test_cap_probe_propagates_non_oom_errors(monkeypatch):
    """A device-side assert is a real fault, not a verdict about the cap -- it must not be
    swallowed and reported as 'enforced'."""
    fake = _fake_torch(alloc_succeeds=False)

    def _boom(_n, dtype=None, device=None):
        raise RuntimeError("device-side assert triggered")

    fake.empty = _boom
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="device-side assert"):
        C._verify_vram_cap_enforced(0)


# --------------------------------------- 3. manager-free pause (the default path)

def test_pause_fallback_is_inert_without_torch(monkeypatch):
    """Must stay torch-free and must NOT hang when free VRAM is unknowable."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert C._pause_on_low_free_vram(lambda _m: None, miner="m") == 0


def test_pause_fallback_waits_while_starved_then_resumes(monkeypatch):
    """The behaviour the design promised but the default deployment never had: with no manager
    running, a starved card PAUSES instead of walking back into the same allocation."""
    free = [0.2, 0.3, 9.0]                       # GiB: starved, starved, recovered
    t = _fake_torch(alloc_succeeds=False)
    t.cuda.mem_get_info = lambda _i=0: (int(free.pop(0) * 2 ** 30), 32 * 2 ** 30)
    monkeypatch.setitem(sys.modules, "torch", t)
    monkeypatch.setenv("NEURAHASH_VRAM_MIN_FREE_GB", "1.0")
    slept, logs = [], []
    waits = C._pause_on_low_free_vram(logs.append, miner="m", poll_s=7.0,
                                      sleep_fn=slept.append, max_waits=5)
    assert waits == 2 and slept == [7.0, 7.0]
    assert any("PAUSED" in m for m in logs) and any("recovered" in m for m in logs)


def test_pause_fallback_is_bounded(monkeypatch):
    """A permanently starved card must not block a miner forever with no way out."""
    t = _fake_torch(alloc_succeeds=False)
    t.cuda.mem_get_info = lambda _i=0: (0, 32 * 2 ** 30)
    monkeypatch.setitem(sys.modules, "torch", t)
    monkeypatch.setenv("NEURAHASH_VRAM_MIN_FREE_GB", "1.0")
    assert C._pause_on_low_free_vram(lambda _m: None, miner="m",
                                     sleep_fn=lambda _s: None, max_waits=3) == 3
