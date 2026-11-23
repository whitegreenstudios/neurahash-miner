"""RESUME-SKIP (3.9.6): records the boot resume already folded out of order are not judged again in order.

Incident (2026-11-23, campaign 508ed8c1ec9a2f9a, the 8 GB 4060 on (L1,E50)). resume_to_root folded only
the records naming 1_50 (events 24..155) and reached the coordinator's root 054d4cae. The in-order
catch-up then met event 24, which advertises 1_50's root AS IT WAS THEN (9512eecc), called it
off-lineage, and held the frontier at 23. Every contribution went out as base_event=23 + root 054d4cae
and the coordinator dropped it (wrong-lineage-slot-root), terminally, every round.

Each refusal test below has a paired positive control, and test_the_live_freeze_is_repaired fails on the
pre-3.9.6 code (frontier held at 23).

Run: C:/Python313/python.exe -m pytest tests/test_glm_resume_skip_folded.py -q
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), os.path.join(os.path.dirname(_HERE), "tools"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_glm_contributor as N        # noqa: E402
import test_glm_seed_refold as S               # noqa: E402  (shared fakes: Host, Lane, _row, _fold, ...)

OWN = (1, 50)
OWNKEY = "1_50"
OTHER = (1, 7)                                 # resident, never resumed
OTHERKEY = "1_7"
def _root(w, coord=OWN):
    """slot_root hashes the coordinate too, so the root must be taken at OUR coordinate."""
    h = S.Host([coord])
    h.write_slot(0, w)
    return N.slot_root(h, 0)


W0 = S.FACTORY                                 # the coordinator registered 1_50 at this copy
D1 = S.DELTA                                   # accepted at event 155
D2 = {k: S._bf16(i + 70, scale=0.1) for i, k in enumerate(S.ORDER)}   # accepted later, at event 156
W1 = S._fold(W0, D1)
W2 = S._fold(W1, D2)


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    for name in ("_LATCH_RESEED_TRIED", "_LATCH_VERIFIED_ROOTS"):
        getattr(N, name, {}).clear()
    monkeypatch.delenv("NEURAHASH_GLM_LATCH_RESEED", raising=False)
    yield
    for name in ("_LATCH_RESEED_TRIED", "_LATCH_VERIFIED_ROOTS"):
        getattr(N, name, {}).clear()


def _reject(key, root):
    return dict(accepted=[], slot_roots={key: root}, model_root="g")


def _live_records():
    """The 1_50 history the relay actually shows: rejects advertising the registration root at 24 and
    26, the accept at 155, and an unrelated (non-resident) coordinate's record in between."""
    return {
        24: _reject(OWNKEY, _root(W0)),
        26: _reject(OWNKEY, _root(W0)),
        100: _reject("2_15", "c683a7d8" * 8),
        155: dict(accepted=[S._row("glm-7872f61C", OWN, "d155")], slot_roots={OWNKEY: _root(W1)},
                  model_root="g155"),
    }


def _contiguous(records, upto):
    """The relay holds EVERY event 1..head (the catch-up stops at a gap); fill the ones this test does
    not care about with records for a coordinate no host here holds."""
    out = {e: _reject("2_99", "9%063d" % e) for e in range(1, upto + 1)}
    out.update(records)
    return out


def _setup(records=None, extra_coords=()):
    records = _contiguous(records or _live_records(), 155)
    host = S.Host([OWN] + list(extra_coords))
    host.write_slot(0, W0)
    for j in range(1, len(host.slots)):
        host.write_slot(j, S.FACTORY)
    lane = S.Lane(records, {}, {}, {"d155": D1, "d156": D2}, seed_event=None)
    return host, lane


def _resume(host, lane):
    lines = []
    out = {}
    applied, reached = N.resume_to_root(host, lane, "coordinator-global-root", lines.append,
                                        own_coord=OWN, outcome=out)
    return reached, lines


def _catch_up(host, lane, start, frontier):
    lines = []
    out = N.catch_up_accepted(host, lane, lane.manifest(), start, frontier, None, 0, "glm-c47c9334",
                              lines.append)
    return out, lines


def _own_equals(host, w):
    got = host.read_slot(0)
    return all(np.array_equal(got[k], w[k]) for k in S.ORDER)


def test_the_live_freeze_is_repaired():
    host, lane = _setup()
    reached, rl = _resume(host, lane)
    assert reached, rl
    assert _own_equals(host, W1), "resume must leave 1_50 at the post-155 copy"
    (last, applied_any, rc), lines = _catch_up(host, lane, 0, 155)
    assert rc is None
    assert last == 155, "frontier must reach 155, not stick at 23 (the live freeze): %r" % (lines,)
    assert not any("LINEAGE-SKIP" in m for m in lines), lines
    assert any("RESUME-SKIP" in m and OWNKEY in m for m in lines), lines
    assert _own_equals(host, W1), "event 155's delta must NOT be folded a second time"


def test_later_records_for_the_coordinate_still_fold_and_verify():
    host, lane = _setup()
    assert _resume(host, lane)[0]
    _catch_up(host, lane, 0, 155)
    # a NEW accept for 1_50 lands after the resume: it must fold normally on top of W1
    lane.names[N.accepted_name(156, S.CAMPAIGN)] = {"sha256": "sha156"}
    lane._recs["sha156"] = dict(accepted=[S._row("glm-c47c9334", OWN, "d156")],
                                slot_roots={OWNKEY: _root(W2)}, model_root="g156", event=156)
    (last, applied_any, rc), lines = _catch_up(host, lane, 155, 156)
    assert last == 156 and rc is None, lines
    assert _own_equals(host, W2)


def test_a_wrong_root_above_the_resumed_event_is_still_refused():
    host, lane = _setup()
    assert _resume(host, lane)[0]
    _catch_up(host, lane, 0, 155)
    lane.names[N.accepted_name(156, S.CAMPAIGN)] = {"sha256": "sha156"}
    lane._recs["sha156"] = dict(accepted=[], slot_roots={OWNKEY: "f" * 64}, model_root="g156", event=156)
    (last, _a, _rc), lines = _catch_up(host, lane, 155, 156)
    assert last == 155, "the lineage guard must still hold above the resumed event"
    assert any("LINEAGE-SKIP: accepted record at event 156" in m for m in lines), lines


def test_a_coordinate_that_was_not_resumed_is_judged_as_before():
    recs = _live_records()
    recs[30] = _reject(OTHERKEY, "e" * 64)            # 1_7 is resident but never resumed: wrong root
    host, lane = _setup(recs, extra_coords=[OTHER])
    assert _resume(host, lane)[0]
    (last, _a, _rc), lines = _catch_up(host, lane, 0, 155)
    assert last == 29, lines                          # 24 and 26 skipped, refused at 30 as before
    assert any("LINEAGE-SKIP: accepted record at event 30" in m for m in lines), lines


def test_positive_control_the_same_record_passes_when_its_root_is_right():
    recs = _live_records()
    recs[30] = _reject(OTHERKEY, _root(S.FACTORY, OTHER))
    host, lane = _setup(recs, extra_coords=[OTHER])
    assert _resume(host, lane)[0]
    (last, _a, _rc), lines = _catch_up(host, lane, 0, 155)
    assert last == 155, lines


def test_without_a_resume_nothing_is_skipped():
    """No resume -> no record of out-of-order folds -> the old behaviour, byte for byte: event 24's
    root matches the untouched copy, 155 folds, frontier 155."""
    host, lane = _setup()
    (last, _a, _rc), lines = _catch_up(host, lane, 0, 155)
    assert last == 155 and _own_equals(host, W1), lines
    assert not any("RESUME-SKIP" in m for m in lines), lines


def test_an_unreached_resume_records_nothing():
    host, lane = _setup()
    host.write_slot(0, S.OTHER_SEED)                  # a copy no advertisement can be reached from
    reached, _ = _resume(host, lane)
    assert not reached
    assert all(np.array_equal(host.read_slot(0)[k], S.OTHER_SEED[k]) for k in S.ORDER), "rolled back"
    assert N.already_folded_by_resume(host, _live_records()[24], 24) is None


def test_helper_ignores_legacy_and_foreign_records():
    host, lane = _setup()
    assert _resume(host, lane)[0]
    assert N.already_folded_by_resume(host, {"model_root": "x"}, 24) is None          # no slot_roots
    assert N.already_folded_by_resume(host, _reject("2_15", "c" * 64), 24) is None    # not resident
    assert N.already_folded_by_resume(host, _reject(OWNKEY, "c" * 64), 156) is None   # above the fold
    assert N.already_folded_by_resume(host, _reject(OWNKEY, "c" * 64), 155) == [OWNKEY]


# ---------------------------------------------------------------- review findings (2026-11-23)
def test_a_forged_far_future_record_cannot_switch_off_checking():
    """Finding 1: resume folds a self-consistent forged accept at event 100000. The recorded event is
    above the coordinator's frontier, so it is distrusted: the genuine record at 156 is NOT skipped
    silently; the guard refuses it visibly, as it did before this change."""
    host, lane = _setup()
    forged = {k: S._bf16(i + 90, scale=0.1) for i, k in enumerate(S.ORDER)}
    lane._deltas["dforged"] = forged
    lane.names[N.accepted_name(100000, S.CAMPAIGN)] = {"sha256": "sha100000"}
    lane._recs["sha100000"] = dict(accepted=[S._row("attacker", OWN, "dforged")],
                                   slot_roots={OWNKEY: _root(S._fold(W1, forged))}, model_root="gx",
                                   event=100000)
    assert _resume(host, lane)[0]
    lane.names[N.accepted_name(156, S.CAMPAIGN)] = {"sha256": "sha156"}
    lane._recs["sha156"] = dict(accepted=[S._row("glm-c47c9334", OWN, "d156")],
                                slot_roots={OWNKEY: _root(W2)}, model_root="g156", event=156)
    (last, _a, _rc), lines = _catch_up(host, lane, 0, 156)
    assert not any("RESUME-SKIP" in m for m in lines), lines
    assert any("LINEAGE-SKIP" in m for m in lines), lines


def test_a_campaign_rebind_does_not_reuse_the_old_map():
    """Finding 2: the map is per campaign; after a rebind the old campaign's entry skips nothing."""
    host, lane = _setup()
    assert _resume(host, lane)[0]
    assert N.already_folded_by_resume(host, _reject(OWNKEY, "c" * 64), 10) == [OWNKEY]
    host.campaign_id = "ffffffffffffffff"
    assert N.already_folded_by_resume(host, _reject(OWNKEY, "c" * 64), 10) is None


def test_a_failed_later_resume_keeps_the_earlier_entry():
    """Finding 3: a later unreachable resume rolls the slots back to the call-start snapshot (still the
    earlier resumed copy), so the earlier map entry must survive too."""
    host, lane = _setup()
    assert _resume(host, lane)[0]
    lane._recs["sha155"] = dict(lane._recs["sha155"], slot_roots={OWNKEY: "a" * 64})
    for e in (24, 26):
        lane._recs["sha%d" % e] = dict(lane._recs["sha%d" % e], slot_roots={OWNKEY: "b" * 64})
    reached, _ = _resume(host, lane)
    assert not reached
    assert _own_equals(host, W1)
    assert N.already_folded_by_resume(host, _reject(OWNKEY, "c" * 64), 155, frontier=155) == [OWNKEY]
