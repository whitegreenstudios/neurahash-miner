"""Fleet-hosted pipeline (tools/glm_pipe.py) -- codec, bus, stage routing, driver loop.

Model-free: the store is a FakeLane dict, stages are monkeypatched pumps; only the activation
codec uses real (CPU) torch tensors. The real-model path is exercised by the live smoke.
"""
import json
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glm_pipe as GP                                        # noqa: E402


class FakeLane:
    """Dict-backed content store: put_blob/get_blob/manifest with the LIVE store's manifest shape
    (name -> {'sha256': cid, 'size': n})."""

    def __init__(self):
        self.blobs, self.names = {}, {}
        self.lock = threading.Lock()

    def put_blob(self, body, name=None):
        import hashlib
        cid = hashlib.sha256(body).hexdigest()
        with self.lock:
            self.blobs[cid] = bytes(body)
            if name:
                self.names[name] = {"sha256": cid, "size": len(body)}
        return cid

    def get_blob(self, cid):
        with self.lock:
            return self.blobs[cid]

    def manifest(self):
        with self.lock:
            return dict(self.names)


def test_act_codec_roundtrip_bf16_and_f32():
    import torch
    for dt in (torch.bfloat16, torch.float32):
        t = torch.randn(1, 3, 8, dtype=torch.float32).to(dt)
        back = GP.act_decode(GP.act_encode(t))
        assert back.dtype == dt and list(back.shape) == [1, 3, 8]
        assert torch.equal(back, t)


def test_bus_send_recv_and_manifest_shape():
    lane = FakeLane()
    bus = GP.PipeBus(lane, poll_s=0.01)
    bus.send("sharddiloco/glm/pipe/r1/s1/p/in0", {"ids": [1, 2, 3]})
    body = bus.recv("sharddiloco/glm/pipe/r1/s1/p/in0", timeout=2.0)
    assert json.loads(body.decode()) == {"ids": [1, 2, 3]}
    try:
        bus.recv("sharddiloco/glm/pipe/r1/s1/p/in9", timeout=0.15)
    except TimeoutError:
        return
    raise AssertionError("recv on a missing name must time out")


def test_run_stage_routes_and_tracks_positions(monkeypatch):
    """Stage 0 gets ids (prefill 3 tokens, then a 1-token step); the fake stage_step must see the
    right position offsets, and the outputs must land on in1. done-advert evicts the cache."""
    import torch
    lane = FakeLane()
    bus = GP.PipeBus(lane, poll_s=0.01)
    calls = []

    def fake_stage_step(model, cfg, hidden, position_ids, cache, lo, hi, ids=None):
        calls.append({"pos0": int(position_ids[0, 0]), "n": int(position_ids.shape[1]),
                      "ids": None if ids is None else ids.tolist()})
        n = int(position_ids.shape[1])
        return torch.zeros(1, n, 4, dtype=torch.bfloat16)

    monkeypatch.setattr(GP, "stage_step", fake_stage_step)
    monkeypatch.setattr(GP, "new_cache", lambda cfg: object())

    bus.send("sharddiloco/glm/pipe/r1/sA/p/in0", {"ids": [5, 6, 7]})
    t = threading.Thread(target=GP.run_stage,
                         args=(lane, "r1", 0, None, None, 0, 2, "cpu"),
                         kwargs={"bus": bus, "idle_exit_s": 1.2})
    t.start()
    out = bus.recv("sharddiloco/glm/pipe/r1/sA/p/in1", timeout=3.0)
    assert GP.act_decode(out).shape == (1, 3, 4)
    bus.send("sharddiloco/glm/pipe/r1/sA/t0/in0", {"ids": [9]})
    out2 = bus.recv("sharddiloco/glm/pipe/r1/sA/t0/in1", timeout=3.0)
    assert GP.act_decode(out2).shape == (1, 1, 4)
    bus.send("sharddiloco/glm/pipe/r1/sA/done", {"done": True})
    t.join(timeout=5.0)
    assert not t.is_alive(), "stage did not idle-exit"
    assert calls[0] == {"pos0": 0, "n": 3, "ids": [[5, 6, 7]]}       # prefill at offset 0
    assert calls[1] == {"pos0": 3, "n": 1, "ids": [[9]]}             # decode step continues at 3


def test_driver_prefill_step_finish_against_fake_stage():
    """PipeDriver against a threaded fake single-stage pump: names line up, last-position slicing
    works, finish publishes the done-advert."""
    import torch
    lane = FakeLane()
    bus = GP.PipeBus(lane, poll_s=0.01)
    stop = []

    def pump():
        seen = set()
        while not stop:
            for name, rec in lane.manifest().items():
                if name in seen or not name.endswith("/in0"):
                    continue
                seen.add(name)
                body = lane.get_blob(rec["sha256"])
                n = len(json.loads(body.decode())["ids"])
                lane.put_blob(GP.act_encode(torch.full((1, n, 4), 2.0)),
                              name=name[:-1] + "1")
            time.sleep(0.01)

    t = threading.Thread(target=pump)
    t.start()
    try:
        drv = GP.PipeDriver(lane, "r2", 1, head_model=None, cfg=None, device="cpu",
                            bus=bus, timeout=5.0)
        stream = drv.new_stream()
        h = drv.prefill(stream, [1, 2, 3, 4])
        assert h.shape == (1, 1, 4)                          # last position only
        h2 = drv.step(stream, 0, 7)
        assert h2.shape == (1, 1, 4)
        drv.finish(stream)
        assert any(n.endswith("/done") for n in lane.manifest())
    finally:
        stop.append(1)
        t.join(timeout=2.0)
