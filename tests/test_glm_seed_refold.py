"""SEED REFOLD (3.9.3, extends D1): the FIRST accepted record that touches a resident coordinate MOVES it,
while this node's copy is still the untouched factory copy and the coordinator's pre-fold state is the
campaign's seed snapshot.

Incident (2026-11-13, campaign e2c4cb58fcfb41cc, the 8 GB 4060): record 369 moved 1_1 with outer 0.7.
seed + 0.7*delta = 0111393590.. (advertised), factory + 0.7*delta = 4dc50acd.. (local). The fold refused
(correct), D1 never re-seeds a MOVED coordinate (F3), and its re-seed needs the seed to carry the POST-fold
root, which it never can. The async frontier froze at 368 forever. Next instance: 1_21 at event 389.

The fix restores K, applies the hash-verified seed for K, re-folds the record's rows for K and keeps the
result ONLY if K's root equals the advertised root. Every test below that asserts a refusal also asserts the
paired positive control, so the file fails on the pre-3.9.3 code.

Run: C:/Python313/python.exe -m pytest tests/test_glm_seed_refold.py -q
"""
import hashlib
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_glm_contributor as N                               # noqa: E402

CAMPAIGN = "e2c4cb58fcfb41cc"
OWN = (1, 43)                    # the coordinate the 4060 trains
K = (1, 1)                       # resident, factory copy, seed differs, first record MOVES it
KKEY = "1_1"
SHAPE = (2, 3)
ORDER = ("down", "gate", "up")
SEED_EVENT = 101


def _bf16(seed, scale=0.01):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal(SHAPE).astype(np.float32) * scale)
    return (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def _pack_bf16(a):
    """The publisher's truncating fp32 -> bf16 pack (glm_seed_publish.pack_bf16), inlined so this file runs
    in the public repo too."""
    return (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> np.uint32(16)
            ).astype("<u2").tobytes()


class Host:
    """Real numpy slots, copying reads/writes, campaign-scoped (host_campaign_id reads campaign_id)."""

    def __init__(self, coords, campaign=CAMPAIGN):
        self.slots = [tuple(c) for c in coords]
        self.campaign_id = campaign
        self._w = {i: {k: np.zeros(SHAPE, np.float32) for k in ORDER} for i in range(len(self.slots))}

    def index_of(self, L, E):
        t = (int(L), int(E))
        return self.slots.index(t) if t in self.slots else None

    def read_slot(self, i):
        return {k: v.copy() for k, v in self._w[int(i)].items()}

    def write_slot(self, i, d):
        self._w[int(i)] = {k: np.array(v, dtype=np.float32, copy=True) for k, v in d.items()}


def _root_of(w):
    h = Host([K])
    h.write_slot(0, w)
    return N.slot_root(h, 0)


def _fold(base, delta, outer=0.7):
    """The fold's exact arithmetic (apply_accepted: cur + outer*d, fp32), through a host write."""
    h = Host([K])
    h.write_slot(0, base)
    cur = h.read_slot(0)
    h.write_slot(0, {k: cur[k] + outer * delta[k] for k in ORDER})
    return h.read_slot(0)


def _publish(weights):
    blob = b"".join(_pack_bf16(weights[k]) for k in ORDER)
    chunks = [blob[o:o + 16] for o in range(0, len(blob), 16)]
    cids = [hashlib.sha256(c).hexdigest() for c in chunks]
    entry = {"root": _root_of(weights), "dtype": "bf16", "order": list(ORDER),
             "shapes": {k: list(SHAPE) for k in ORDER}, "bytes": len(blob), "chunks": cids}
    return entry, dict(zip(cids, chunks))


class Lane:
    def __init__(self, records, seed_coords, blobs, deltas, seed_event=SEED_EVENT):
        self.names, self._recs = {}, {}
        for e, rec in sorted(records.items()):
            sha = "sha%d" % e
            self.names[N.accepted_name(e, CAMPAIGN)] = {"sha256": sha}
            self._recs[sha] = dict(rec, event=int(e))
        self._seed = {"campaign": CAMPAIGN, "coords": dict(seed_coords)}
        if seed_event is not None:
            self._seed["event"] = seed_event
        self._blobs = dict(blobs)
        self._deltas = dict(deltas)
        self.blob_reads = 0
        self.manifest_reads = 0
        self.manifest_error = None

    def manifest(self):
        return dict(self.names)

    def get_json(self, sha, max_bytes=None):
        return self._recs[sha]

    def get_json_named(self, name, man=None, max_bytes=None):
        assert name == N.seed_manifest_name(CAMPAIGN), name
        self.manifest_reads += 1
        if self.manifest_error is not None:
            raise self.manifest_error
        return self._seed

    def get_blob(self, cid, max_bytes=None):
        self.blob_reads += 1
        return self._blobs.get(cid)

    def get_delta(self, cid):
        return {k: v.copy() for k, v in self._deltas[cid].items()}


def _row(miner, coord, cid, outer=0.7):
    return dict(miner=miner, slot=0, layer=int(coord[0]), glm_expert=int(coord[1]), cid=cid,
                outer=float(outer), paid=0.0, token_weight=None)


FACTORY = {k: _bf16(i + 1) for i, k in enumerate(ORDER)}
SEED = {k: _bf16(i + 10) for i, k in enumerate(ORDER)}
OTHER_SEED = {k: _bf16(i + 50) for i, k in enumerate(ORDER)}
DELTA = {k: _bf16(i + 30, scale=0.1) for i, k in enumerate(ORDER)}


def _scenario(advertise_base=SEED, verified_first=False, seed_event=SEED_EVENT, event=369,
              manifest_has_k=True):
    """Frontier `event-1`; record `event` MOVES K (the 4060's own accepted delta) advertising
    root(advertise_base + 0.7*delta). verified_first: an earlier zero-row record advertises K's FACTORY
    root, so this process verifies the factory copy before the moving record arrives."""
    host = Host([OWN, K])
    host.write_slot(1, FACTORY)
    entry, blobs = _publish(SEED)
    want = _root_of(_fold(advertise_base, DELTA))
    recs = {event: dict(accepted=[_row("glm-c47c9334", K, "d369")], slot_roots={KKEY: want},
                        model_root="g%d" % event)}
    if verified_first:
        recs[event - 1] = dict(accepted=[], slot_roots={KKEY: _root_of(FACTORY)},
                               model_root="g%d" % (event - 1))
    seeds = {"1:1": entry} if manifest_has_k else {"1:27": entry}
    lane = Lane(recs, seeds, blobs, {"d369": DELTA}, seed_event=seed_event)
    start = event - (2 if verified_first else 1)
    return host, lane, want, start


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    for name in ("_LATCH_RESEED_TRIED", "_LATCH_VERIFIED_ROOTS"):
        getattr(N, name, {}).clear()
    monkeypatch.delenv("NEURAHASH_GLM_LATCH_RESEED", raising=False)
    yield
    for name in ("_LATCH_RESEED_TRIED", "_LATCH_VERIFIED_ROOTS"):
        getattr(N, name, {}).clear()


def _catch_up(host, lane, start, frontier=369):
    lines = []
    out = N.catch_up_accepted(host, lane, lane.manifest(), start, frontier, None, 0, "glm-c47c9334",
                              lines.append)
    return out, lines


def _k_equals(host, w):
    got = host.read_slot(1)
    return all(np.array_equal(got[k], w[k]) for k in ORDER)


def _assert_held(host, lines, at=369):
    assert any(("LINEAGE-SKIP: accepted record at event %d" % at) in m for m in lines), lines
    assert _k_equals(host, FACTORY), "K must be restored bit-exactly to the pre-call factory copy"


class TestA_TheLiveFreezeIsRepaired:
    def test_moving_first_record_folds_onto_the_seed(self):
        host, lane, want, start = _scenario()
        assert N.slot_root(host, 1) == _root_of(FACTORY) != _root_of(SEED)
        (last, applied, abort), lines = _catch_up(host, lane, start)
        assert abort is None
        assert (last, applied) == (369, True), lines
        assert N.slot_root(host, 1) == want
        assert _k_equals(host, _fold(SEED, DELTA))
        assert any("SEED REFOLD: coordinate 1_1" in m and "root now matches the record" in m
                   for m in lines), lines
        assert not any("LINEAGE-SKIP" in m for m in lines), lines
        assert lane.blob_reads == len(lane._seed["coords"]["1:1"]["chunks"])
        # the repaired root is now VERIFIED, so a later forged record can never revert it
        assert want in N._LATCH_VERIFIED_ROOTS[(CAMPAIGN, KKEY)]


class TestB_SeedPlusDeltaDiffers:
    def test_wrong_advertised_root_stays_fail_closed_and_restores_k(self):
        host, lane, want, start = _scenario(advertise_base=OTHER_SEED)
        (last, applied, abort), lines = _catch_up(host, lane, start)
        assert (last, applied, abort) == (start, False, None)
        _assert_held(host, lines)
        assert lane.blob_reads > 0, "the refold must have been ATTEMPTED (fails on pre-3.9.3 code)"
        assert any("SEED REFOLD 1_1" in m and "still differs" in m for m in lines), lines
        assert N._LATCH_RESEED_TRIED[("seed-refold", CAMPAIGN, KKEY, want)][0] == "refold-root-differs"


class TestC_VerifiedCopyIsNeverRefolded:
    def test_control_then_verified(self):
        # control: never verified -> refolds (fails on pre-3.9.3 code)
        host, lane, want, start = _scenario()
        (last, _, _), _ = _catch_up(host, lane, start)
        assert last == 369
        N._LATCH_RESEED_TRIED.clear()
        N._LATCH_VERIFIED_ROOTS.clear()
        # verified by an earlier record in this process -> no attempt at all
        host, lane, want, start = _scenario(verified_first=True)
        (last, applied, abort), lines = _catch_up(host, lane, start)
        assert (last, abort) == (368, None), lines
        _assert_held(host, lines)
        assert lane.manifest_reads == 0 and lane.blob_reads == 0
        assert not any("SEED REFOLD" in m for m in lines), lines


class TestD_KillSwitch:
    def test_control_then_d1_disabled_is_the_old_behaviour(self, monkeypatch):
        host, lane, want, start = _scenario()
        (last, _, _), _ = _catch_up(host, lane, start)
        assert last == 369, "control: D1 on repairs (fails on pre-3.9.3 code)"
        N._LATCH_RESEED_TRIED.clear()
        N._LATCH_VERIFIED_ROOTS.clear()
        monkeypatch.setenv("NEURAHASH_GLM_LATCH_RESEED", "0")
        host, lane, want, start = _scenario()
        (last, applied, abort), lines = _catch_up(host, lane, start)
        assert (last, applied, abort) == (start, False, None)
        _assert_held(host, lines)
        assert lane.manifest_reads == 0 and lane.blob_reads == 0
        assert not any("SEED REFOLD" in m or "latch re-seed" in m for m in lines), lines
        assert any("off-lineage: coordinate 1_1" in m for m in lines), lines


class TestE_ManifestUnavailable:
    def test_unreadable_manifest_is_retried_after_the_cooldown(self, monkeypatch):
        t = {"now": 1000.0}
        monkeypatch.setattr(N, "_latch_now", lambda: t["now"])
        host, lane, want, start = _scenario()
        lane.manifest_error = OSError("relay 502")
        (last, _, _), lines = _catch_up(host, lane, start)
        assert last == start
        _assert_held(host, lines)
        assert lane.manifest_reads == 1
        lane.manifest_error = None
        (last, _, _), lines = _catch_up(host, lane, start)       # inside the cooldown: no retry
        assert last == start and lane.manifest_reads == 1
        _assert_held(host, lines)
        t["now"] += N.LATCH_RESEED_RETRY_S + 1.0
        (last, _, _), lines = _catch_up(host, lane, start)       # after it: retried and repaired
        assert last == 369, lines
        assert N.slot_root(host, 1) == want

    def test_manifest_without_k_or_without_event_stays_fail_closed(self):
        for kw in (dict(manifest_has_k=False), dict(seed_event=None)):
            N._LATCH_RESEED_TRIED.clear()
            host, lane, want, start = _scenario(**kw)
            (last, applied, abort), lines = _catch_up(host, lane, start)
            assert (last, applied, abort) == (start, False, None), kw
            _assert_held(host, lines)
            assert lane.manifest_reads == 1 and lane.blob_reads == 0, kw

    def test_record_not_later_than_the_seed_is_never_refolded(self):
        host, lane, want, start = _scenario(seed_event=369)
        (last, _, _), lines = _catch_up(host, lane, start)
        assert last == start
        _assert_held(host, lines)
        assert lane.blob_reads == 0
        assert N._LATCH_RESEED_TRIED[("seed-refold", CAMPAIGN, KKEY, want)][0] == "seed-not-older"


class TestF_OncePerAdvertisedRoot:
    def test_a_refusal_is_attempted_once(self):
        host, lane, want, start = _scenario(advertise_base=OTHER_SEED)
        _catch_up(host, lane, start)
        n = lane.blob_reads
        assert n > 0, "first pass must attempt the refold (fails on pre-3.9.3 code)"
        m = lane.manifest_reads
        for _ in range(3):
            (last, _, _), lines = _catch_up(host, lane, start)
            assert last == start
            _assert_held(host, lines)
        assert lane.blob_reads == n and lane.manifest_reads == m
