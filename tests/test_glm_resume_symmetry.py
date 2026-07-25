"""CONTRIBUTOR resume symmetry: reach the coordinator's advertised root, or roll fully back.

Measured 2026-07-25: coordinator --resume alone broke the fleet -- contributors rebuilt from the
frozen base, could not reach the resumed root, and 1527 contributions were dropped
wrong-lineage-root (network earned ZERO). These tests pin the other half.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                              # noqa: E402
import sharddiloco_glm_contributor as N                         # noqa: E402


class FakeHost:
    """Slots hold one array each; folding record e sets the slot to e and the root to 'root<e>'."""

    def __init__(self):
        self.slots = [0]
        self._d = {"w": np.zeros(3)}
        self.writes = 0

    def read_slot(self, j):
        return self._d

    def write_slot(self, j, d):
        self._d = d
        self.writes += 1


class FakeLane:
    def __init__(self, events):
        self.names, self.recs = {}, {}
        for e in events:
            sha = "sha%d" % e
            self.names[N.accepted_name(e)] = {"sha256": sha}
            self.recs[sha] = {"event": e, "model_root": "root%d" % e}

    def manifest(self):
        return dict(self.names)

    def get_json(self, sha):
        return self.recs[sha]


def _fold_factory(state, ok_events=None):
    """Fold sets the current root to the record's advertised root (that IS the fold contract)."""
    def fold(host, lane, rec, regate, own_slot, log=None):
        if ok_events is not None and rec["event"] not in ok_events:
            return False, "lineage", []
        state["root"] = rec["model_root"]
        return True, "ok", []
    return fold


def test_noop_when_base_already_matches(monkeypatch):
    monkeypatch.setattr(N, "model_root", lambda h: "rootX")
    applied, reached = N.resume_to_root(FakeHost(), FakeLane([1, 2]), "rootX", lambda m: None)
    assert (applied, reached) == (0, True), "a matching base must not replay anything"


def test_reaches_target_and_stops_early(monkeypatch):
    """Must stop the moment the target is reached -- never fold past the coordinator's root."""
    state = {"root": "base"}
    monkeypatch.setattr(N, "model_root", lambda h: state["root"])
    monkeypatch.setattr(N, "_fold_accepted_checked", _fold_factory(state))
    host, lane = FakeHost(), FakeLane([1, 2, 3, 4, 5])
    applied, reached = N.resume_to_root(host, lane, "root3", lambda m: None)
    assert reached is True and applied == 3, "should fold 1,2,3 then stop"
    assert state["root"] == "root3"
    assert host.writes == 0, "a successful replay must not roll back"


def test_unreachable_target_rolls_fully_back(monkeypatch):
    """Fail-closed: an unreachable root must restore EVERY slot, not leave a half-folded base."""
    state = {"root": "base"}
    monkeypatch.setattr(N, "model_root", lambda h: state["root"])
    monkeypatch.setattr(N, "_fold_accepted_checked", _fold_factory(state))
    host, lane = FakeHost(), FakeLane([1, 2])
    applied, reached = N.resume_to_root(host, lane, "root-never", lambda m: None)
    assert (applied, reached) == (0, False)
    assert host.writes == len(host.slots), "every slot must be restored on failure"


def test_off_lineage_records_are_skipped_not_fatal(monkeypatch):
    """A dirty store lists other runs' records at the same events; those fail their own root check
    (already rolled back by the fold) and must be skipped, not abort the search."""
    state = {"root": "base"}
    monkeypatch.setattr(N, "model_root", lambda h: state["root"])
    monkeypatch.setattr(N, "_fold_accepted_checked", _fold_factory(state, ok_events={1, 3}))
    applied, reached = N.resume_to_root(FakeHost(), FakeLane([1, 2, 3]), "root3", lambda m: None)
    assert reached is True and applied == 2, "skips event 2, still reaches root3 via 1 and 3"


def test_no_records_at_all_is_fail_closed(monkeypatch):
    monkeypatch.setattr(N, "model_root", lambda h: "base")
    applied, reached = N.resume_to_root(FakeHost(), FakeLane([]), "rootZ", lambda m: None)
    assert (applied, reached) == (0, False)


def test_wired_into_the_async_loop():
    """The regression that matters: the replay must actually be CALLED at contributor startup."""
    import inspect
    src = inspect.getsource(N._run_async)
    assert "resume_to_root" in src, "contributor startup no longer attempts resume symmetry"
    assert "base MISMATCH" in src, "the mismatch diagnostic disappeared"
