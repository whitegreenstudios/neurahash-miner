"""Alpha 3.0 Objective 2 -- periodic corpus RE-SYNC in the GLM contributor (a RUNNING miner picks up
a freshly-published corpus WITHOUT a restart). Guards tools/sharddiloco_glm_contributor.py's ADDITIVE
alpha-3 helpers with NO network / GPU / model:

  * plan_data_resync            -- PURE change decision (manifest-sha gate + which files differ)
  * _data_resync_enabled        -- the NEURAHASH_GLM_DATA_RESYNC guard (flag-off == alpha-2.0)
  * glm_data_periodic_resync    -- flag-off no-op, cheap unchanged path, verified re-fetch, and the
                                   FAIL-CLOSED keep-the-old-corpus path (reuses glm_data_autosync)

The fake lane + recording getter mirror tests/test_glm_data_autosync.py (the W6 startup half).

Run: C:/Python313/python.exe -m pytest tests/test_glm_periodic_resync.py -q
"""
import hashlib
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_glm_contributor as N                                # noqa: E402


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class FakeLane:
    """ContentLane stand-in: a MUTABLE named record resolves name -> sha256 -> record dict, the same
    two-step _read_data_record uses. set_record swaps it (publish v2 mid-run). record=None => absent."""

    def __init__(self, record=None):
        self.record = record
        self.manifest_calls = 0

    def set_record(self, record):
        self.record = record

    def manifest(self):
        self.manifest_calls += 1
        return {} if self.record is None else {N.DATA_RECORD_NAME: {"sha256": "REC-CID"}}

    def get_json(self, cid):
        if cid == "REC-CID" and self.record is not None:
            return self.record
        raise KeyError(cid)


class RecordingGetter:
    """Injected in place of data_http_get (F2 signature). `table` maps full URL -> bytes (served) or an
    Exception (raised). STREAMS served bytes to dest_path and returns (sha256_hex, n). Records URLs."""

    def __init__(self, table=None):
        self.table = table or {}
        self.calls = []

    def __call__(self, url, timeout=60, expected_size=None, dest_path=None):
        self.calls.append(url)
        v = self.table.get(url)
        if v is None:
            raise OSError("no object at %s" % url)
        if isinstance(v, Exception):
            raise v
        with open(dest_path, "wb") as f:
            f.write(v)
        return _sha(v), len(v)


def _f(body):
    return {"sha256": _sha(body), "size": len(body)}


def _rec(manifest_sha, files, seeds=("http://seedA",)):
    return {"manifest_sha256": manifest_sha, "seeds": list(seeds), "files": files}


def _mklog():
    lines = []
    return lines, (lambda *a: lines.append(" ".join(str(x) for x in a)))


class TestPlanDataResync:
    """The PURE change decision -- no I/O, the cheap gate the async loop runs every round boundary."""

    def test_none_new_record_is_unchanged(self):
        prev = _rec("a" * 64, {"ids_x_train.npy": _f(b"x")})
        changed, old, new, files = N.plan_data_resync(prev, None)
        assert changed is False and files == [] and old == "a" * 64 and new is None

    def test_same_manifest_sha_unchanged(self):
        files = {"ids_x_train.npy": _f(b"x")}
        changed, old, new, cf = N.plan_data_resync(_rec("a" * 64, files), _rec("a" * 64, files))
        assert changed is False and old == new == "a" * 64 and cf == []

    def test_new_record_without_manifest_sha_is_unchanged(self):
        prev = _rec("a" * 64, {})
        changed, *_ = N.plan_data_resync(prev, {"files": {"ids_x_train.npy": _f(b"x")}})
        assert changed is False                       # a malformed record never disturbs the corpus

    def test_first_sync_prev_none_marks_all_files(self):
        new = _rec("b" * 64, {"ids_a_train.npy": _f(b"a"), "ids_b_train.npy": _f(b"b")})
        changed, old, new_sha, cf = N.plan_data_resync(None, new)
        assert changed is True and old is None and new_sha == "b" * 64
        assert cf == ["ids_a_train.npy", "ids_b_train.npy"]

    def test_changed_files_covers_rehash_add_and_remove(self):
        prev = _rec("a" * 64, {"ids_keep.npy": _f(b"same"), "ids_change.npy": _f(b"old"),
                               "ids_drop.npy": _f(b"gone")})
        new = _rec("c" * 64, {"ids_keep.npy": _f(b"same"),       # unchanged -> NOT listed
                              "ids_change.npy": _f(b"NEW"),       # re-hashed
                              "ids_add.npy": _f(b"added")})       # added
        changed, _old, _new, cf = N.plan_data_resync(prev, new)
        assert changed is True
        assert cf == sorted(["ids_change.npy", "ids_drop.npy", "ids_add.npy"])


class TestResyncEnabledGuard:
    # DEFAULT FLIPPED TO ON 2026-07-25 (owner directive): a miner that never picks up the daily
    # corpus trains forever on stale data. Opt-OUT semantics now: only an explicit off-value
    # disables it. The re-sync itself stays fail-closed (unverifiable manifest -> keep old corpus).
    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("YES", True), ("on", True), ("On ", True),
        ("0", False), ("false", False), ("off", False), ("no", False), ("OFF ", False),
        ("", True), ("maybe", True),
    ])
    def test_explicit_values(self, val, expected):
        assert N._data_resync_enabled({"NEURAHASH_GLM_DATA_RESYNC": val}) is expected

    def test_absent_is_ON(self):
        assert N._data_resync_enabled({}) is True, "resync must default ON"

    def test_reads_os_environ_by_default(self, monkeypatch):
        monkeypatch.setenv("NEURAHASH_GLM_DATA_RESYNC", "0")
        assert N._data_resync_enabled() is False
        monkeypatch.delenv("NEURAHASH_GLM_DATA_RESYNC", raising=False)
        assert N._data_resync_enabled() is True


class TestPeriodicResync:
    def test_flag_off_is_zero_io_noop(self, tmp_path):
        lane = FakeLane(_rec("a" * 64, {"ids_x_train.npy": _f(b"x")}))
        get = RecordingGetter()
        rec, refreshed = N.glm_data_periodic_resync(
            lane, str(tmp_path), None, log=(lambda *a: None), http_get=get,
            env={"NEURAHASH_GLM_DATA_RESYNC": "0"})     # flag now defaults ON -> opt out explicitly
        assert refreshed is False and rec is None
        assert lane.manifest_calls == 0 and get.calls == []      # never even read the record

    def test_unchanged_reads_record_but_never_fetches(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        files = {"ids_x_train.npy": _f(b"x")}
        prev = _rec("a" * 64, files)
        lane = FakeLane(_rec("a" * 64, files))                   # same manifest sha
        get = RecordingGetter()
        rec, refreshed = N.glm_data_periodic_resync(lane, str(tmp_path), prev, log=(lambda *a: None),
                                                    http_get=get, env={"NEURAHASH_GLM_DATA_RESYNC": "1"})
        assert refreshed is False and rec is prev
        assert lane.manifest_calls == 1                          # read the record...
        assert get.calls == []                                   # ...but no fetch (cheap gate)

    def test_changed_fetches_verifies_and_advances_baseline(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        prev = _rec("a" * 64, {"ids_code_train.npy": _f(b"OLD-CORPUS")})
        name, newbody = "ids_code_train.npy", b"FRESH-DAILY-CORPUS-v2"
        newrec = _rec("b" * 64, {name: _f(newbody)}, seeds=["http://seedA"])
        lane = FakeLane(newrec)
        url = "http://seedA/o/%s" % _sha(newbody)
        get = RecordingGetter({url: newbody})
        lines, log = _mklog()
        rec, refreshed = N.glm_data_periodic_resync(lane, str(tmp_path), prev, log=log,
                                                    http_get=get, env={"NEURAHASH_GLM_DATA_RESYNC": "1"})
        assert refreshed is True and rec is newrec               # baseline advanced to the new record
        assert get.calls == [url]                                # fetched the changed file
        assert (tmp_path / name).read_bytes() == newbody         # installed + sha-verified
        joined = "\n".join(lines)
        assert "corpus resync: manifest" in joined and "re-fetched 1 file(s)" in joined

    def test_fail_closed_keeps_old_verified_corpus_and_does_not_raise(self, tmp_path, monkeypatch):
        # The whole point of Objective 2's fail-closed rule: an unverifiable NEW corpus must NOT kill a
        # running miner nor replace its known-good data. Both seeds serve garbage -> autosync raises
        # SystemExit(rc9); the periodic step CATCHES it, keeps the old file, and returns not-refreshed.
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        name, old_body = "ids_code_train.npy", b"KNOWN-GOOD-OLD-CORPUS"
        (tmp_path / name).write_bytes(old_body)                  # the miner's current verified file
        prev = _rec("a" * 64, {name: _f(old_body)})
        newbody = b"the-fresh-v2-bytes-nobody-serves"
        newrec = _rec("b" * 64, {name: _f(newbody)}, seeds=["http://seedA", "http://seedB"])
        lane = FakeLane(newrec)
        sha = _sha(newbody)
        get = RecordingGetter({"http://seedA/o/%s" % sha: b"garbage-A",
                               "http://seedB/o/%s" % sha: b"garbage-B"})
        lines, log = _mklog()
        rec, refreshed = N.glm_data_periodic_resync(lane, str(tmp_path), prev, log=log,   # must NOT raise
                                                    http_get=get, env={"NEURAHASH_GLM_DATA_RESYNC": "1"})
        assert refreshed is False and rec is prev                # baseline NOT advanced -> retry later
        assert (tmp_path / name).read_bytes() == old_body        # old verified corpus intact
        assert get.calls == ["http://seedA/o/%s" % sha, "http://seedB/o/%s" % sha]   # both seeds tried
        assert "REFUSED" in "\n".join(lines)
