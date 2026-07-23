"""W6 corpus-over-WAN DOWNLOAD half -- contributor auto-fetch + sha256 verification (fail-closed).

Guards tools/sharddiloco_glm_contributor.glm_data_autosync WITHOUT network or GPU: a fake lane hands
back a crafted data record and a recording getter stands in for the HTTP fetch (injected exactly like
W5's uploaders), so every path -- local-match short-circuit, in-order multi-seed fetch, tamper
refetch, fail-closed rc9, opt-out, and the F1 filename hard-guard -- runs offline in tmp_path.

Run: C:/Python313/python.exe -m pytest tests/test_glm_data_autosync.py -q
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
    """Minimal ContentLane stand-in: a named record resolves name -> sha256 (manifest) -> bytes
    (get_json), the SAME two-step the real lane + read_pointer use. record=None => record absent."""

    def __init__(self, record):
        self._record = record
        self.manifest_calls = 0

    def manifest(self):
        self.manifest_calls += 1
        if self._record is None:
            return {}
        return {N.DATA_RECORD_NAME: {"sha256": "REC-CID"}}

    def get_json(self, cid):
        if cid == "REC-CID" and self._record is not None:
            return self._record
        raise KeyError(cid)


class RecordingGetter:
    """Injected in place of data_http_get (F2 signature). `table` maps a full URL -> bytes (served) or an
    Exception (raised, i.e. seed unusable); a URL absent from the table also raises. STREAMS the served
    bytes to dest_path (exactly like the real getter) and returns (sha256_hex, n_bytes). Records every
    call's URL."""

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


class _ChunkResponse:
    """A urllib-response stand-in for data_http_get's `opener` (F2). read(n) yields the body ONE preset
    chunk at a time and RECORDS how many reads were consumed (to prove EARLY STOP on an over-read), and
    getheader('Content-Length') exposes an optional length (to prove UP-FRONT rejection). Context-manager
    like the real response. content_length='__auto__' == the true summed length; None == header absent."""

    def __init__(self, chunks, content_length="__auto__"):
        self._chunks = list(chunks)
        self._i = 0
        self.reads = 0
        self._clen = (sum(len(c) for c in self._chunks) if content_length == "__auto__"
                      else content_length)

    def getheader(self, name, default=None):
        if str(name).lower() == "content-length":
            return None if self._clen is None else str(self._clen)
        return default

    def read(self, n=-1):
        self.reads += 1
        if self._i >= len(self._chunks):
            return b""
        c = self._chunks[self._i]
        self._i += 1
        return c

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _record(files, seeds):
    return {"manifest_sha256": "m" * 64, "seeds": seeds, "files": files}


def _mklog():
    lines = []
    return lines, (lambda *a: lines.append(" ".join(str(x) for x in a)))


class TestShardDiLoCoDataAutosync:
    def test_missing_file_fetched_from_first_seed(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        body = b"IDS-CODE-TRAIN-BYTES"
        name = "ids_code_train.npy"
        url = "http://seedA/o/%s" % _sha(body)
        rec = _record({name: {"sha256": _sha(body), "size": len(body)}},
                      ["http://seedA", "http://seedB"])
        get = RecordingGetter({url: body})
        _lines, log = _mklog()
        N.glm_data_autosync(FakeLane(rec), str(tmp_path), log=log, http_get=get)
        assert get.calls == [url]                                 # first seed only
        assert (tmp_path / name).read_bytes() == body             # sha-verified + written
        assert not list(tmp_path.glob("*.tmp*"))                  # atomic install left no tmp

    def test_local_match_zero_network(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        body = b"already-here-and-correct"
        name = "ids_gutenberg_val.npy"
        (tmp_path / name).write_bytes(body)
        rec = _record({name: {"sha256": _sha(body), "size": len(body)}}, ["http://seedA"])
        get = RecordingGetter()
        N.glm_data_autosync(FakeLane(rec), str(tmp_path), log=(lambda *a: None), http_get=get)
        assert get.calls == []                                    # zero network on a local match

    def test_local_tamper_refetched(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        good = b"the-verified-corpus-bytes"
        name = "ids_code_train.npy"
        (tmp_path / name).write_bytes(b"TAMPERED-local-copy")     # wrong bytes on disk
        url = "http://seedA/o/%s" % _sha(good)
        rec = _record({name: {"sha256": _sha(good), "size": len(good)}}, ["http://seedA"])
        get = RecordingGetter({url: good})
        N.glm_data_autosync(FakeLane(rec), str(tmp_path), log=(lambda *a: None), http_get=get)
        assert get.calls == [url]                                 # tamper forced a refetch
        assert (tmp_path / name).read_bytes() == good             # replaced with verified bytes

    def test_second_seed_wins_when_first_wrong(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        good = b"correct-bytes-only-on-seedB"
        name = "ids_code_val.npy"
        sha = _sha(good)
        rec = _record({name: {"sha256": sha, "size": len(good)}}, ["http://seedA", "http://seedB"])
        get = RecordingGetter({"http://seedA/o/%s" % sha: b"seedA-serves-GARBAGE",
                               "http://seedB/o/%s" % sha: good})
        N.glm_data_autosync(FakeLane(rec), str(tmp_path), log=(lambda *a: None), http_get=get)
        assert get.calls == ["http://seedA/o/%s" % sha, "http://seedB/o/%s" % sha]  # in order
        assert (tmp_path / name).read_bytes() == good

    def test_all_seeds_bad_exits_rc9_naming_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        good = b"never-served-correctly"
        name = "ids_code_train.npy"
        sha = _sha(good)
        rec = _record({name: {"sha256": sha, "size": len(good)}}, ["http://seedA", "http://seedB"])
        get = RecordingGetter({"http://seedA/o/%s" % sha: b"wrong1",
                               "http://seedB/o/%s" % sha: b"wrong2"})
        lines, log = _mklog()
        with pytest.raises(SystemExit) as ei:
            N.glm_data_autosync(FakeLane(rec), str(tmp_path), log=log, http_get=get)
        assert ei.value.code == N.RC_DATA_UNVERIFIED == 9
        assert len(get.calls) == 2                                # both seeds tried
        assert any(name in ln for ln in lines)                    # rc9 message names the file
        assert not (tmp_path / name).exists()                     # nothing unverified written

    def test_record_absent_noop(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        get = RecordingGetter()
        N.glm_data_autosync(FakeLane(None), str(tmp_path), log=(lambda *a: None), http_get=get)
        assert get.calls == []                                    # absent record -> no-op

    def test_env_optout_noop_zero_calls(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEURAHASH_GLM_DATA_AUTOSYNC", "0")
        body = b"would-be-fetched-if-on"
        name = "ids_code_train.npy"
        rec = _record({name: {"sha256": _sha(body), "size": len(body)}}, ["http://seedA"])
        get = RecordingGetter({"http://seedA/o/%s" % _sha(body): body})
        lane = FakeLane(rec)
        N.glm_data_autosync(lane, str(tmp_path), log=(lambda *a: None), http_get=get)
        assert get.calls == []                                    # opted out before any I/O
        assert lane.manifest_calls == 0                           # did not even read the record
        assert not (tmp_path / name).exists()

    @pytest.mark.parametrize("evil", ["ids_code_probe.npy", "ids_code_heldout.npy", "../evil"])
    def test_malicious_name_refuses_whole_record_zero_fetch(self, tmp_path, monkeypatch, evil):
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        # a legit file rides ALONGSIDE the evil one: the guard must refuse the WHOLE record, so even
        # the good file is not fetched (one poisoned name discards everything -- F1).
        good = b"legit"
        rec = _record({"ids_code_train.npy": {"sha256": _sha(good), "size": len(good)},
                       evil: {"sha256": "0" * 64, "size": 1}}, ["http://seedA"])
        get = RecordingGetter({"http://seedA/o/%s" % _sha(good): good})
        lines, log = _mklog()
        N.glm_data_autosync(FakeLane(rec), str(tmp_path), log=log, http_get=get)
        assert get.calls == []                                    # ZERO fetches
        assert any("SECURITY" in ln and evil in ln for ln in lines)  # refused loudly, naming it
        assert not list(tmp_path.iterdir())                       # nothing written at all

    @pytest.mark.parametrize("bad", [
        {"sha256": "a" * 64},                                     # size MISSING -> ceiling would be off
        {"sha256": "a" * 64, "size": 0},                          # non-positive
        {"sha256": "a" * 64, "size": True},                       # bool masquerading as int
        {"sha256": "a" * 64, "size": "12"},                       # non-int
        {"sha256": "a" * 63, "size": 12},                         # malformed sha (63 hex)
        "notadict",                                               # entry not a dict at all
    ])
    def test_entry_without_valid_sha_or_size_refuses_whole_record(self, tmp_path, monkeypatch, bad):
        # F2 companion: `size` IS the download ceiling -- an entry without a valid one would reopen
        # the declare-nothing-send-huge DoS, so the whole record is refused fail-closed (a legit W5
        # record always carries both fields).
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        good = b"legit"
        rec = _record({"ids_code_train.npy": {"sha256": _sha(good), "size": len(good)},
                       "ids_code_val.npy": bad}, ["http://seedA"])
        get = RecordingGetter({"http://seedA/o/%s" % _sha(good): good})
        lines, log = _mklog()
        N.glm_data_autosync(FakeLane(rec), str(tmp_path), log=log, http_get=get)
        assert get.calls == []                                    # ZERO fetches, even the good file
        assert any("SECURITY" in ln and "ids_code_val.npy" in ln for ln in lines)
        assert not list(tmp_path.iterdir())                       # nothing written at all


class TestShardDiLoCoDataFetchCeiling:
    """F2: the real data_http_get streams under a size+slack ceiling. These exercise its OWN abort logic
    through the injectable `opener` (a fake chunked response), with zero network."""

    def test_oversized_chunked_body_aborts_before_consuming_all(self, tmp_path):
        # ceiling = expected_size + 65536. A body delivered in 20 chunks of 20000 B (400000 B total) MUST
        # abort mid-stream as soon as the running total crosses the ceiling -- the recorder proves the
        # early stop (only ~4 chunks read, not all 20), so a forged giant body is never fully consumed.
        expected, chunk = 10, b"x" * 20000
        chunks = [chunk] * 20
        resp = _ChunkResponse(chunks, content_length=None)        # no Content-Length -> exercise read cap
        dest = str(tmp_path / "over.bin")
        with pytest.raises(ValueError):
            N.data_http_get("http://seed/o/x", timeout=1, expected_size=expected, dest_path=dest,
                            opener=lambda _u, _t: resp)
        assert resp.reads < len(chunks)                           # EARLY STOP: not all chunks consumed
        assert resp.reads <= ((expected + 65536) // len(chunk)) + 2   # ~4 reads, nowhere near 20

    def test_oversized_content_length_rejected_before_body_read(self, tmp_path):
        # a Content-Length header over the ceiling is rejected BEFORE a single body byte is read.
        resp = _ChunkResponse([b"whatever"], content_length=10_000_000)   # >> ceiling
        dest = str(tmp_path / "clen.bin")
        with pytest.raises(ValueError):
            N.data_http_get("http://seed/o/x", timeout=1, expected_size=10, dest_path=dest,
                            opener=lambda _u, _t: resp)
        assert resp.reads == 0                                    # rejected up-front, body never read
        assert not os.path.exists(dest)                          # ...and no temp file was even opened

    def test_within_ceiling_streams_and_hashes(self, tmp_path):
        # the happy path: a body inside the ceiling streams to dest and returns its true sha + length.
        body = b"small-and-legit" * 3
        resp = _ChunkResponse([body[:10], body[10:]])            # 2 chunks, Content-Length auto = real len
        dest = str(tmp_path / "ok.bin")
        got, n = N.data_http_get("http://seed/o/x", timeout=1, expected_size=len(body), dest_path=dest,
                                 opener=lambda _u, _t: resp)
        assert got == _sha(body) and n == len(body)
        assert (tmp_path / "ok.bin").read_bytes() == body

    def test_autosync_ceiling_abort_falls_to_next_seed(self, tmp_path, monkeypatch):
        # F2 at the autosync level: seedA aborts (over-ceiling -> ValueError, partial tmp), seedB serves the
        # real bytes; the file installs from seedB, both seeds are tried in order, and the partial is cleaned.
        monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
        good = b"the-correct-small-body"
        name = "ids_code_train.npy"
        sha = _sha(good)
        rec = _record({name: {"sha256": sha, "size": len(good)}}, ["http://seedA", "http://seedB"])
        urlA, urlB = "http://seedA/o/%s" % sha, "http://seedB/o/%s" % sha
        calls = []

        def http_get(url, timeout=60, expected_size=None, dest_path=None):
            calls.append(url)
            if url == urlA:
                with open(dest_path, "wb") as f:                 # partial write, then the ceiling abort
                    f.write(b"z" * 4096)
                raise ValueError("body over ceiling")
            with open(dest_path, "wb") as f:
                f.write(good)
            return _sha(good), len(good)

        N.glm_data_autosync(FakeLane(rec), str(tmp_path), log=(lambda *a: None), http_get=http_get)
        assert calls == [urlA, urlB]                              # tried in order, fell through the abort
        assert (tmp_path / name).read_bytes() == good             # installed from seedB
        assert not list(tmp_path.glob("*.tmp*"))                  # the aborted partial was cleaned up
