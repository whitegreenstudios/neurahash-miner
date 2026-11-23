"""Windows self-update re-exec must release the GPU first (2026-11-22, v3.9.3 -> v3.9.4).

The stall these pin: on Windows the updater cannot replace the process, so it runs the new miner as a
CHILD and the parent waits. The parent was still inside the mining loop with the model resident, and
the child sat in its VRAM pause for ~17 h ("STILL PAUSED on VRAM after 4020 checks", 0.00 GiB free
vs a 0.17 GiB bar). The fix raises the re-exec as _ReexecAfterRelease, unwinds the loop, releases
CUDA, and only then re-execs.

Torch-free, like test_glm_vram_pause.py.
"""
import gc
import os
import sys
import weakref

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import self_update as SU  # noqa: E402
import sharddiloco_glm_contributor as C  # noqa: E402


class _Model(object):
    """Stands in for the resident model: only reachable from the mining loop's frame."""


def test_deferred_reexec_raises_on_windows(monkeypatch):
    monkeypatch.setattr(C.os, "name", "nt")
    with pytest.raises(C._ReexecAfterRelease) as ei:
        C._deferred_reexec(["miner.py", "--mode", "glm"])
    assert ei.value.argv == ["miner.py", "--mode", "glm"]
    assert not isinstance(ei.value, Exception), "an `except Exception` must not be able to swallow it"


def test_deferred_reexec_execs_directly_on_posix(monkeypatch):
    calls = []
    monkeypatch.setattr(C.os, "name", "posix")
    monkeypatch.setattr(SU, "_default_reexec", lambda argv: calls.append(list(argv)))
    C._deferred_reexec(["a", "b"])
    assert calls == [["a", "b"]]


def test_update_check_passes_the_deferred_reexec_and_does_not_swallow_it(monkeypatch):
    monkeypatch.setattr(C.os, "name", "nt")
    seen = {}

    def fake_check(reexec_fn=None):
        seen["fn"] = reexec_fn
        reexec_fn(["x"])                      # what check_and_update does after a verified update

    monkeypatch.setattr(SU, "check_and_update", fake_check)
    with pytest.raises(C._ReexecAfterRelease):
        C._maybe_self_update(log=lambda m: None)
    assert seen["fn"] is C._deferred_reexec


def test_defer_can_be_turned_off(monkeypatch):
    calls = []

    def fake_check(*a, **kw):
        calls.append((a, kw))

        class R(object):
            action = "rate-limited"
        return R()

    monkeypatch.setattr(SU, "check_and_update", fake_check)
    C._maybe_self_update(log=lambda m: None, defer_reexec=False)
    assert calls == [((), {})]


def test_the_mining_loop_call_site_is_plain(monkeypatch):
    """Tests elsewhere stub _maybe_self_update with `lambda log: ...`; the loop must not pass extra
    arguments, or every stubbed run dies with a TypeError (seen while building this fix)."""
    src = open(C.__file__, encoding="utf-8").read()
    assert "        _maybe_self_update(log)\n" in src


def test_entry_frees_the_loop_locals_before_reexec(monkeypatch):
    """THE property: when _default_reexec runs, nothing from the mining loop is still alive, and the
    CUDA release ran first. Fails if the re-exec happens inside the loop (the 3.9.4 behaviour) or
    inside the except block (the traceback would still hold the frame)."""
    monkeypatch.setattr(C.os, "name", "nt")
    ref = {}
    order = []

    def fake_main(argv=None):
        model = _Model()
        ref["w"] = weakref.ref(model)
        C._deferred_reexec(["miner.py", "--x"])      # raised from deep inside the loop
        return 0

    def fake_release(log=None):
        gc.collect()
        order.append(("release", ref["w"]() is None))

    def fake_reexec(argv):
        gc.collect()
        order.append(("reexec", ref["w"]() is None, list(argv)))
        return 0

    monkeypatch.setattr(C, "main", fake_main)
    monkeypatch.setattr(C, "_release_cuda_before_reexec", fake_release)
    monkeypatch.setattr(SU, "_default_reexec", fake_reexec)
    C._entry()
    assert order == [("release", True), ("reexec", True, ["miner.py", "--x"])], order


def test_entry_returns_main_rc_when_nothing_updates(monkeypatch):
    monkeypatch.setattr(C, "main", lambda argv=None: 7)
    assert C._entry() == 7


def test_release_without_torch_is_a_noop(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert C._release_cuda_before_reexec(log=lambda m: None) is None


def test_release_never_raises(monkeypatch):
    class _Boom(object):
        class cuda(object):
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def synchronize():
                raise RuntimeError("driver gone")

    msgs = []
    monkeypatch.setitem(sys.modules, "torch", _Boom)
    assert C._release_cuda_before_reexec(log=msgs.append) is None
    assert msgs and "failed" in msgs[0]
