"""v3.2.1 signed auto-update WIRE tests (torch-free).

The bug this guards against was not a broken updater -- tools/self_update.py was fully built and
tested -- it was that NOTHING in the GLM-only client ever CALLED it (found live 2026-07-24: the
4060 sat on v3.1.0 after v3.2.0 was signed). So besides the hook's behavior, these tests assert
the WIRING: main() and the async round loop actually invoke the hook.
"""
import inspect
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_glm_contributor as contrib                    # noqa: E402
import glm_rollout_worker as worker                              # noqa: E402


class _Result:
    def __init__(self, action):
        self.action = action


def test_hook_swallows_exceptions_and_returns_none():
    """The update machinery must NEVER kill mining: any Exception -> warn + None."""
    logs = []

    def boom():
        raise RuntimeError("mirror down")

    r = contrib._maybe_self_update(log=logs.append, _check=boom)
    assert r is None
    assert any("mining continues on current code" in m for m in logs)


def test_hook_returns_result_and_logs_only_interesting_actions():
    """Quiet on the every-round no-ops (rate-limited); loud on a real apply."""
    logs = []
    r = contrib._maybe_self_update(log=logs.append, _check=lambda: _Result("rate-limited"))
    assert r.action == "rate-limited" and logs == []
    r = contrib._maybe_self_update(log=logs.append, _check=lambda: _Result("applied"))
    assert r.action == "applied" and len(logs) == 1


def test_hook_does_not_swallow_systemexit():
    """SystemExit is the updater's own re-exec/exit path -- swallowing it would CANCEL the update."""
    def exits():
        raise SystemExit(0)

    try:
        contrib._maybe_self_update(log=lambda m: None, _check=exits)
    except SystemExit:
        return
    raise AssertionError("SystemExit was swallowed; the applied update would have been cancelled")


def test_wiring_contributor_startup_and_round_loop_call_the_hook():
    """THE regression this file exists for: the updater must actually be CALLED."""
    assert "_maybe_self_update" in inspect.getsource(contrib.main), \
        "startup auto-update call missing from contributor main()"
    assert "_maybe_self_update" in inspect.getsource(contrib._run_async), \
        "round-boundary auto-update call missing from the async loop"


def test_wiring_rollout_worker_startup_calls_the_updater():
    assert "check_and_update" in inspect.getsource(worker.main), \
        "startup auto-update call missing from rollout worker main()"
