"""A relay refusal must never stop a miner (2026-11-12).

MEASURED 2026-11-11 (4060 joiner test, client 3.9.0): the relay answered a record PUT with HTTP 403
and the HTTPError escaped the contributor's main loop, so the miner process EXITED. These tests pin
the fix at the unit level: the classifier, the backoff, and the layer-claim publisher's partial
progress. The end-to-end checks drive the real `_run_async` loop on the tiny real GLM against an in-process
content-addressed store (private repo: the same tests live in test_sd_async_lane.py s10-s13).
"""
import io
import os
import socket
import sys
import urllib.error

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_HERE, _REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_harness as H                             # noqa: E402
import sharddiloco_glm_contributor as N                     # noqa: E402


def _http(code, body=b'{"ok":false,"error":"name outside the public namespace"}'):
    return urllib.error.HTTPError("http://relay/o/x", code, "refused", {}, io.BytesIO(body))


class TestClassifier:
    @pytest.mark.parametrize("code", [401, 403, 404, 409, 413, 429, 500, 503, 507])
    def test_every_http_status_is_a_refusal_with_the_store_reason(self, code):
        why = N.publish_failure_reason(_http(code))
        assert why.startswith("HTTP %d" % code)
        assert "public namespace" in why                    # the store's own reason reaches the log

    def test_transport_faults_are_refusals(self):
        for exc in (urllib.error.URLError("connection reset"), socket.timeout("timed out"),
                    ConnectionResetError(104, "reset"), OSError("network unreachable")):
            assert N.publish_failure_reason(exc) is not None, exc

    def test_programming_errors_are_not_hidden(self):
        for exc in (ValueError("bad payload"), KeyError("gate"), AttributeError("x"),
                    TypeError("y")):
            assert N.publish_failure_reason(exc) is None, exc

    def test_reason_is_bounded_and_one_line(self):
        why = N.publish_failure_reason(_http(403, b"A" * 5000 + b"\nsecond line"))
        assert "\n" not in why and len(why) < 260

    def test_unreadable_body_still_classifies(self):
        e = urllib.error.HTTPError("http://relay/o/x", 507, "full", {}, None)
        assert N.publish_failure_reason(e).startswith("HTTP 507")

    def test_wrapped_refusal_classifies_by_its_cause(self):
        pr = N.PublishRefused(_http(507), 2, 5, 100)
        assert N.publish_failure_reason(pr).startswith("HTTP 507")


class TestBackoff:
    def test_doubles_from_base_and_caps(self):
        b = N.PublishBackoff(base_s=60, cap_s=1800)
        waits = [b.failed(1000.0) for _ in range(8)]
        assert waits == [60, 120, 240, 480, 960, 1800, 1800, 1800]

    def test_blocked_counts_down_and_success_resets(self):
        b = N.PublishBackoff(base_s=60, cap_s=1800)
        assert b.blocked(0.0) == 0.0                        # a fresh miner publishes immediately
        b.failed(100.0)
        assert b.blocked(130.0) == pytest.approx(30.0)
        assert b.blocked(161.0) == 0.0
        b.failed(161.0)
        assert b.failures == 2 and b.blocked(161.0) == pytest.approx(120.0)
        b.succeeded()
        assert b.failures == 0 and b.blocked(161.0) == 0.0

    def test_module_defaults_are_read_at_construction(self, monkeypatch):
        monkeypatch.setattr(N, "PUBLISH_BACKOFF_BASE_S", 0.0)
        assert N.PublishBackoff().failed(5.0) == 0.0
        monkeypatch.undo()
        assert N.PublishBackoff().failed(5.0) == 60.0


class _RefusingLane:
    """Accepts deltas; refuses the `refuse_at`-th record PUT (0-based) with `exc`."""

    def __init__(self, refuse_at, exc, refuse_delta_at=None):
        self.refuse_at, self.exc, self.refuse_delta_at = refuse_at, exc, refuse_delta_at
        self.deltas, self.records, self._rec_calls, self._delta_calls = {}, {}, 0, 0

    def put_delta(self, d):
        k = self._delta_calls
        self._delta_calls += 1
        if self.refuse_delta_at is not None and k == self.refuse_delta_at:
            raise self.exc
        cid = "cid%d" % len(self.deltas)
        self.deltas[cid] = H.pack_arrays(d, np.float16)
        return cid

    def put_json_named(self, name, obj):
        k = self._rec_calls
        self._rec_calls += 1
        if k == self.refuse_at:
            raise self.exc
        self.records[name] = obj


class _Host:
    def __init__(self, slots):
        self.slots = list(slots)

    def index_of(self, L, E):
        t = (int(L), int(E))
        return self.slots.index(t) if t in self.slots else None


def _publish(lane, coords, publish_k=10):
    payload = {"%d_%d" % c: {k: np.zeros((5, 4), np.float32) for k in N.DELTA_SLOT_KEYS}
               for c in coords}
    orig = N.slot_root
    N.slot_root = lambda h, i: "root_%d" % int(i)
    try:
        return N.publish_layer_claim_records(lane, _Host(coords), payload, coords, "m0", 7, "groot",
                                             b"k" * 16, None, 1e9, 12, 0, publish_k,
                                             lambda *_a: None, address="0xabc")
    finally:
        N.slot_root = orig


class TestLayerPublishPartialProgress:
    COORDS = [(1, e) for e in range(4)]

    def test_record_refusal_keeps_published_records_and_never_reuses_a_name(self):
        lane = _RefusingLane(refuse_at=2, exc=_http(403))
        with pytest.raises(N.PublishRefused) as ei:
            _publish(lane, self.COORDS, publish_k=10)
        pr = ei.value
        assert pr.n_pub == 2 and len(lane.records) == 2      # the two that landed stay on the lane
        assert pr.publish_k == 13                            # 10, 11 used; 12 attempted -> skip to 13
        assert pr.total_bytes == sum(len(b) for b in list(lane.deltas.values())[:2])
        assert isinstance(pr.cause, urllib.error.HTTPError) and pr.cause.code == 403

    def test_delta_refusal_reuses_nothing_it_did_not_attempt(self):
        lane = _RefusingLane(refuse_at=None, exc=_http(507), refuse_delta_at=1)
        with pytest.raises(N.PublishRefused) as ei:
            _publish(lane, self.COORDS, publish_k=0)
        assert ei.value.n_pub == 1 and ei.value.publish_k == 1   # no record name was attempted

    def test_a_bug_in_the_publish_path_still_raises_as_itself(self):
        lane = _RefusingLane(refuse_at=0, exc=ValueError("not a refusal"))
        with pytest.raises(ValueError):
            _publish(lane, self.COORDS)

    def test_no_refusal_is_unchanged(self):
        lane = _RefusingLane(refuse_at=None, exc=None)
        n, pk, nbytes = _publish(lane, self.COORDS, publish_k=3)
        assert (n, pk, len(lane.records)) == (4, 7, 4)



class TestPermanentRefusals:
    """Review finding (2026-11-12): a refusal that waiting cannot fix must not make the miner train
    forever with nothing published. Mirrors UPDATE_CAP_MAX_REFUSALS."""

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 413])
    def test_permanent_codes(self, code):
        assert N.publish_failure_is_permanent(_http(code)) is True

    @pytest.mark.parametrize("code", [408, 425, 429, 500, 503, 507])
    def test_temporary_codes(self, code):
        assert N.publish_failure_is_permanent(_http(code)) is False

    def test_transport_faults_are_temporary(self):
        assert N.publish_failure_is_permanent(urllib.error.URLError("reset")) is False
        assert N.publish_failure_is_permanent(socket.timeout("t")) is False

    def test_exits_after_max_consecutive_permanent_refusals_with_the_reason(self):
        b = N.PublishBackoff(base_s=1, cap_s=10, permanent_max=3)
        b.failed(0.0, _http(403), "mX")
        b.failed(0.0, _http(403), "mX")
        with pytest.raises(SystemExit) as ei:
            b.failed(0.0, _http(403), "mX")
        msg = str(ei.value)
        assert "mX" in msg and "HTTP 403" in msg and "public namespace" in msg

    def test_a_temporary_fault_or_a_success_resets_the_permanent_count(self):
        b = N.PublishBackoff(base_s=1, cap_s=10, permanent_max=3)
        b.failed(0.0, _http(403)); b.failed(0.0, _http(403))
        b.failed(0.0, _http(503))                            # temporary: the streak is broken
        b.failed(0.0, _http(403)); b.failed(0.0, _http(403))
        b.succeeded()
        b.failed(0.0, _http(403)); b.failed(0.0, _http(403))
        assert b.permanent == 2                              # never reached 3 consecutive

    def test_quota_full_507_retries_forever(self):
        b = N.PublishBackoff(base_s=1, cap_s=10, permanent_max=3)
        for _ in range(50):
            b.failed(0.0, _http(507))
        assert b.permanent == 0


class TestBadStoreReply:
    def test_non_json_2xx_and_corrupt_gzip_are_transport_faults(self):
        import json
        import zlib
        try:
            json.loads("<html>captive portal</html>")
        except json.JSONDecodeError as e:
            assert N.publish_failure_reason(e).startswith("bad store reply")
        assert N.publish_failure_reason(zlib.error("bad header")).startswith("bad store reply")


class TestLayerRefusalReasonIsKept:
    def test_reason_survives_the_drained_http_body(self):
        lane = _RefusingLane(refuse_at=1, exc=_http(403))
        with pytest.raises(N.PublishRefused) as ei:
            _publish(lane, [(1, e) for e in range(3)])
        assert "public namespace" in ei.value.reason
        assert "public namespace" in N.publish_failure_reason(ei.value)   # second read: still there



# ============================================================ end to end through the real miner loop
import json                                                 # noqa: E402
import types                                                # noqa: E402

import neurahash.diloco_merge as dm                         # noqa: E402

_SLOTS = "1:0,1:1"


class _InProcStore:
    """A content-addressed store (cid == sha256 of the bytes, like H.ContentLane), in-process."""

    def __init__(self):
        self._blobs, self._names = {}, {}

    def put_blob(self, body, name=None):
        cid = H.cid_of(body)
        self._blobs[cid] = body
        if name:
            self._names[name] = cid
        return cid

    def get_blob(self, cid):
        return self._blobs[cid]

    def put_delta(self, delta, name=None):
        return self.put_blob(H.pack_arrays(delta, np.float16), name=name)

    def get_delta(self, cid):
        return H.unpack_arrays(self._blobs[cid])

    def put_json_named(self, name, obj):
        return self.put_blob(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode(), name=name)

    def get_json(self, cid, max_bytes=None):
        return json.loads(self._blobs[cid].decode())

    def read_pointer(self, man=None):
        if man is not None:
            e = man.get(N.GLM_POINTER_NAME)
            return self.get_json(e["sha256"]) if e else None
        cid = self._names.get(N.GLM_POINTER_NAME)
        return self.get_json(cid) if cid else None

    def health(self):
        return {"ok": True}

    def manifest(self):
        return {n: {"sha256": c} for n, c in self._names.items()}


class _RefusingStore(_InProcStore):
    """Refuses the first `n_refuse` contribution-record PUTs with `exc`, then behaves normally."""

    def __init__(self, exc, n_refuse=1):
        super().__init__()
        self.exc, self.n_refuse, self.refused = exc, n_refuse, 0

    def put_json_named(self, name, obj):
        if name.startswith("cg/r") and self.refused < self.n_refuse:
            self.refused += 1
            raise self.exc
        return super().put_json_named(name, obj)


@pytest.fixture(scope="module")
def _glm():
    import torch
    torch.set_num_threads(2)
    G = N._G()
    T = N.TINY
    model, cfg = G.build_tiny_glm(seed=T["seed"], vocab=T["vocab"], hidden=T["hidden"], inter=T["inter"],
                                  moe_inter=T["moe_inter"], layers=T["layers"], n_experts=T["n_experts"],
                                  topk=T["topk"])
    slots = N.parse_slots(_SLOTS)
    host = G.GlmExpertLaneHost(model, cfg, slots)
    snap = [host.read_slot(i) for i in range(len(slots))]
    return G, model, host, slots, T["seq"], snap


@pytest.fixture
def glm(_glm, monkeypatch):
    G, model, host, slots, seq, snap = _glm
    for i, d in enumerate(snap):                              # every test starts from the same base
        host.write_slot(i, d)
    monkeypatch.setenv("NEURAHASH_SD_ASYNC", "1")
    monkeypatch.delenv("NEURAHASH_SHARDDILOCO_MAX_STALENESS", raising=False)
    monkeypatch.setattr(N, "fresh_publish_k", lambda now=None: 0)   # 3.9.2: names counted from 0 here
    return G, model, host, slots, seq


def _genesis(store, host, slots):
    """The event-0 pointer a live coordinator publishes at boot (every slot at round 0)."""
    rounds = {"%d_%d" % (int(L), int(E)): 0 for (L, E) in slots}
    store.put_json_named(N.GLM_POINTER_NAME, dm.sd_pointer_encode(0, rounds, N.model_root(host), False))


def _args():
    return types.SimpleNamespace(poll=0.001, max_rounds=1, garbage=True, round_wait=1e9, inner=1,
                                 lora_r=8, lr=1e-3, batch=2, outer=0.7, wire="dense", data_dir=".",
                                 domains="daily")


def _run(glm, store, logs=None):
    G, model, host, slots, seq = glm
    _genesis(store, host, slots)
    sink = (lambda *a: logs.append(" ".join(str(x) for x in a))) if logs is not None else (lambda *_a: None)
    return N._run_async(_args(), store, host, model, None, G, b"k" * 16, 0, 1, 0, "tester",
                        [], [], seq, sink)


def _pubs(store):
    return sorted(nm for nm in store._names if nm.startswith("cg/r") and "/tester." in nm)


class TestThroughTheMinerLoop:
    def test_a_refused_record_put_backs_off_and_the_miner_keeps_mining(self, glm, monkeypatch):
        monkeypatch.setattr(N, "PUBLISH_BACKOFF_BASE_S", 0.0)
        store = _RefusingStore(_http(403), n_refuse=2)
        logs = []
        assert _run(glm, store, logs) == 0                    # 3.9.0 exited here on the first 403
        assert store.refused == 2
        assert sum("PUBLISH REFUSED (HTTP 403" in m for m in logs) == 2
        assert _pubs(store) == ["cg/r0/tester.2"]             # .0 and .1 were attempted: never reused

    def test_a_bug_in_the_publish_path_still_raises(self, glm):
        with pytest.raises(ValueError):
            _run(glm, _RefusingStore(ValueError("not a relay refusal")))

    def test_a_permanent_refusal_exits_loudly_instead_of_mining_for_nothing(self, glm, monkeypatch):
        monkeypatch.setattr(N, "PUBLISH_BACKOFF_BASE_S", 0.0)
        monkeypatch.setattr(N, "PUBLISH_PERMANENT_MAX", 3)
        store = _RefusingStore(_http(403), n_refuse=10 ** 9)
        with pytest.raises(SystemExit) as ei:
            _run(glm, store)
        assert store.refused == 3 and "HTTP 403" in str(ei.value)

    def test_while_backing_off_the_round_is_kept_and_published_after(self, glm, monkeypatch):
        monkeypatch.setattr(N, "PUBLISH_BACKOFF_BASE_S", 0.3)
        store = _RefusingStore(urllib.error.URLError("connection reset"), n_refuse=1)
        logs = []
        assert _run(glm, store, logs) == 0 and store.refused == 1
        assert any("PUBLISH BACKOFF" in m for m in logs)      # the blocked branch really ran
        assert _pubs(store) == ["cg/r0/tester.1"]

    def test_names_owned_by_another_writer_are_skipped_not_fatal(self, glm, monkeypatch):
        # 3.9.2 (measured 2026-11-13): 403 "name not owned by signer" on .0-.450 made 3.9.1 exit after 6.
        monkeypatch.setattr(N, "PUBLISH_PERMANENT_MAX", 3)
        monkeypatch.setattr(N, "NAME_TAKEN_RETRY_S", 0.0)
        store = _RefusingStore(_http(403, TestNameTaken.TAKEN), n_refuse=10)
        assert _run(glm, store) == 0 and store.refused == 10
        assert _pubs(store) == ["cg/r0/tester.10"]


class TestNameTaken:
    """3.9.2 (measured 2026-11-13): a restarted 4060 started at publish_k=0 while the coordinator sat on
    one base_event; names .0-.450 were owned by an earlier unsigned client, so every publish got 403
    "name not owned by signer" and 3.9.1 exited after 6, ten times in one night."""

    TAKEN = b'{"ok": false, "error": "name not owned by signer: cg/e2c4cb58fcfb41cc/r368/glm-c47c9334.4"}'

    def test_fresh_publish_k_starts_at_the_clock_not_zero(self):
        assert N.fresh_publish_k(now=1794451197.709) == 1794451197709
        assert N.fresh_publish_k() > 10 ** 12

    def test_a_restart_starts_above_every_name_the_previous_run_used(self):
        first = N.fresh_publish_k(now=1000.0)
        used_by_run1 = first + 10000                         # 10k publishes, far more than a real run
        assert N.fresh_publish_k(now=1000.0 + 11.0) > used_by_run1

    def test_classifies_name_taken(self):
        e = _http(403, self.TAKEN)
        why = N.publish_failure_reason(e)
        assert N.publish_failure_is_name_taken(e, why) is True
        pr = N.PublishRefused(_http(403, self.TAKEN), 1, 5, 10)
        assert N.publish_failure_is_name_taken(pr) is True

    def test_other_403s_are_not_name_taken(self):
        e = _http(403)                                       # "name outside the public namespace"
        assert N.publish_failure_is_name_taken(e, N.publish_failure_reason(e)) is False
        e = _http(409, self.TAKEN)
        assert N.publish_failure_is_name_taken(e, N.publish_failure_reason(e)) is False

    def test_name_taken_is_skipped_quickly_and_never_exits_early(self):
        b = N.PublishBackoff(base_s=60, cap_s=1800, permanent_max=3)
        for _ in range(N.NAME_TAKEN_MAX - 1):
            e = _http(403, self.TAKEN)
            assert b.failed(0.0, e, "mX", reason=N.publish_failure_reason(e)) == N.NAME_TAKEN_RETRY_S
        assert b.permanent == 0

    def test_a_long_run_of_name_taken_exits_at_exactly_the_limit(self):
        b = N.PublishBackoff(base_s=1, cap_s=10, permanent_max=3)
        n = 0
        with pytest.raises(SystemExit) as ei:
            for _ in range(N.NAME_TAKEN_MAX + 5):
                n += 1
                e = _http(403, self.TAKEN)
                b.failed(0.0, e, "mX", reason=N.publish_failure_reason(e))
        assert n == N.NAME_TAKEN_MAX and "duplicate miner" in str(ei.value)

    def test_name_taken_does_not_inflate_a_later_backoff(self):
        b = N.PublishBackoff(base_s=60, cap_s=1800, permanent_max=6)
        for _ in range(10):
            e = _http(403, self.TAKEN)
            b.failed(0.0, e, "mX", reason=N.publish_failure_reason(e))
        assert b.failed(0.0, _http(503)) == 60.0             # first REAL fault: base wait, not 1800

    def test_without_a_reason_name_taken_is_the_3_9_1_permanent_path(self):
        # The sync path passes no reason (its name has no .k suffix): 3.9.1 behaviour, exit after 3.
        b = N.PublishBackoff(base_s=1, cap_s=10, permanent_max=3)
        e = _http(403, self.TAKEN)
        N.publish_failure_reason(e)
        b.failed(0.0, e, "mX"); b.failed(0.0, e, "mX")
        with pytest.raises(SystemExit):
            b.failed(0.0, e, "mX")

    def test_the_same_error_classifies_the_same_twice(self):
        e = _http(403, self.TAKEN)                           # the body can be read only once
        first, second = N.publish_failure_reason(e), N.publish_failure_reason(e)
        assert first == second and "not owned by signer" in second
        assert N.publish_failure_is_name_taken(e, second) is True

    def test_success_resets_the_name_taken_streak(self):
        b = N.PublishBackoff(base_s=1, cap_s=10, permanent_max=3)
        for _ in range(5):
            e = _http(403, self.TAKEN)
            b.failed(0.0, e, "mX", reason=N.publish_failure_reason(e))
        b.succeeded()
        assert b.name_taken == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
