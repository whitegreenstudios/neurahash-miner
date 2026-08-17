"""The miner must notice when the lane it is mining has gone DEAD -- and must not cry wolf.

THE INCIDENT. Our own 4060 mined a lane whose coordinator and judge were both gone for 4.27 days
(368,928 s). It trained, it published, and nothing on the other end scored or paid for any of it.
It never complained. A stranger running the public miner gets the same deal: the card spins, the
electricity bill runs, and nothing tells them the other end is dead.

WHY NOTHING CAUGHT IT. `async_should_abort_no_progress` is the only no-progress logic in the
client and its FIRST rule is `if not comparable: return False`. On a shard-claim lane the miner
holds a different slot set from the coordinator BY CONSTRUCTION, so `global_root_comparable` is
False forever, the guard is skipped forever, and nothing at all is watching the clock.
`test_the_old_guard_is_structurally_blind_to_this` pins that, so nobody "simplifies" the new
detector away on the grounds that the old one covers it.

WHAT CAN ACTUALLY BE OBSERVED. Nothing the coordinator publishes carries a wall clock -- not the
v2 pointer (v/event/rounds/model_root/done), not an accepted record -- so "how old is this?"
cannot be asked. What CAN be seen is that a live coordinator MOVES its monotonic counters: the
pointer `event` advances and new accepted/<campaign>/r<event> names appear. The detector watches
those two counters against the LOCAL clock.

THE THRESHOLD AND ITS TWO SIDES, both pinned below:
  * POSITIVE -- it fires. 3 h detects the real incident 34x sooner than never.
  * NEGATIVE -- it does NOT fire on a lane that is merely SLOW. The slowest legitimate cadence
    ever measured here is a ~660 s WAN pipeline step, and the coordinator's own idle-exit is
    600 s, so a live coordinator publishes something inside 600 s by its own configuration. The
    test drives 24 simulated hours at the 660 s cadence and asserts zero false alarms.

A miner that exits because the pool was briefly slow is a worse bug than the one being fixed, so
detection PAUSES and recovers by itself; `test_a_paused_miner_resumes_by_itself` pins that no
human restart is needed.

Torch-free where it can be, tmp-dir only, no network, no GPU, no real coordinator.

Run: C:/Python313/python.exe -m pytest tests/test_stale_lane_detector.py -q
"""
import os
import sys
import types

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neurahash.diloco_merge as dm                              # noqa: E402  (numpy-only)
import sharddiloco_glm_contributor as N                          # noqa: E402  (torch-free import)

HOUR = 3600.0
SLOW_STEP_S = 660.0          # the slowest legitimate cadence ever MEASURED on this project
REAL_OUTAGE_S = 4.27 * 24 * HOUR


def _live(threshold_s=None, t0=0.0, environ=None):
    d = N.LaneLiveness(threshold_s, now=t0, environ=environ if environ is not None else {})
    d.observe(t0, event=0, accepted_seen=0)          # first observation seeds, never fires
    return d


# ==================================================================== 1. THE THRESHOLD ITSELF
def test_the_default_threshold_is_three_hours():
    """Pinned, because both bounds below are stated relative to it."""
    assert N.DEFAULT_STALE_LANE_S == 3 * HOUR


def test_the_threshold_is_far_below_the_real_outage_and_far_above_the_slowest_real_round():
    """The justification, as arithmetic rather than prose. Detects 34x sooner than the incident;
    tolerates 16x the slowest legitimate round."""
    assert REAL_OUTAGE_S / N.DEFAULT_STALE_LANE_S > 30
    assert N.DEFAULT_STALE_LANE_S / SLOW_STEP_S > 15
    # ... and comfortably above the coordinator's OWN idle-exit deadline (600 s), so a coordinator
    # that is alive at all has published something long before we call the lane dead.
    assert N.DEFAULT_STALE_LANE_S > 10 * 600.0


# ==================================================================== 2. POSITIVE CONTROL
def test_it_fires_when_no_counter_moves_for_the_threshold():
    d = _live(t0=0.0)
    assert d.observe(N.DEFAULT_STALE_LANE_S - 1.0, event=0, accepted_seen=0) == "waiting"
    assert d.observe(N.DEFAULT_STALE_LANE_S, event=0, accepted_seen=0) == "stale"


def test_it_would_have_caught_the_real_incident_on_its_first_three_hours():
    """The 4.27-day lane, replayed: frozen counters, polled every 60 s. The detector must go stale
    once and STAY stale, not flicker."""
    d = _live(t0=0.0)
    states, t = [], 0.0
    while t < REAL_OUTAGE_S:
        t += 60.0
        states.append(d.observe(t, event=7, accepted_seen=3))
    first = states.index("stale")
    assert (first + 1) * 60.0 <= N.DEFAULT_STALE_LANE_S + 60.0
    assert set(states[first:]) == {"stale"}, "the alarm flickered instead of latching"


@pytest.mark.parametrize("event,accepted", [(9, None), (None, 4)])
def test_either_counter_alone_is_enough_to_go_stale(event, accepted):
    """A lane that publishes only one of the two counters is still watched."""
    d = N.LaneLiveness(None, now=0.0, environ={})
    d.observe(0.0, event=event, accepted_seen=accepted)
    assert d.observe(N.DEFAULT_STALE_LANE_S, event=event, accepted_seen=accepted) == "stale"


# ==================================================================== 3. NEGATIVE CONTROLS
def test_it_never_fires_on_a_slow_but_live_lane_over_a_full_day():
    """THE NEGATIVE CONTROL THAT MATTERS. 24 h at the slowest cadence ever measured here (660 s
    per WAN pipeline step), polled every 30 s. Zero false alarms, or this change is worse than the
    bug it fixes."""
    d = _live(t0=0.0)
    t, ev, next_move = 0.0, 0, SLOW_STEP_S
    seen = set()
    while t < 24 * HOUR:
        t += 30.0
        if t >= next_move:
            ev += 1
            next_move += SLOW_STEP_S
        seen.add(d.observe(t, event=ev, accepted_seen=ev))
    assert "stale" not in seen, "fired on a lane that was merely SLOW: %s" % sorted(seen)


def test_a_single_late_publish_just_under_the_threshold_does_not_fire():
    """The boundary from the safe side: a lane that took 2 h 59 min for one round is fine."""
    d = _live(t0=0.0)
    assert d.observe(N.DEFAULT_STALE_LANE_S - 60.0, event=0, accepted_seen=0) == "waiting"
    assert d.observe(N.DEFAULT_STALE_LANE_S - 59.0, event=1, accepted_seen=0) == "live"
    # the clock restarts from the move, not from process start
    assert d.observe(2 * N.DEFAULT_STALE_LANE_S - 120.0, event=1, accepted_seen=0) == "waiting"


def test_a_counter_that_goes_BACKWARDS_is_not_treated_as_proof_of_life():
    """A re-read of a stale mirror, or a coordinator restart republishing an older pointer, must
    not reset the staleness clock -- that would make the detector unfalsifiable."""
    d = _live(t0=0.0)
    d.observe(10.0, event=5, accepted_seen=5)
    assert d.observe(20.0, event=2, accepted_seen=1) == "waiting"
    assert d.observe(10.0 + N.DEFAULT_STALE_LANE_S, event=2, accepted_seen=1) == "stale"


# ------------------------------------------------- #160: the RESUMED-AT-THE-SAME-EVENT deadlock
# THE SECOND INCIDENT (2026-08-09 -> 2026-08-15, six days, one healthy 4060 earning nothing).
# The detector above recovers only when the coordinator advances PAST the miner's latched high-water
# mark. A coordinator that dies at event=579 and is later restarted RESUMES at the preserved
# frontier -- it republishes event=579 -- and the accepted-record count replays the same 579 names.
# For a miner that started DURING the outage, both counters are exactly what it latched, so nothing
# ever "moves": it stayed in the stale-lane pause forever while the lane was healthy. The event
# counter only advances when miners submit, and nothing submitted. Deadlock.
# The fix is a third counter that a republish ALWAYS moves: the pointer's publish stamp (#160,
# dm.SD_PTR_PUBLISHED_AT / dm.sd_publish_stamp).

def test_a_resumed_coordinator_republishing_the_SAME_event_unwedges_the_miner():
    """THE #160 REGRESSION, at the unit the defect lives in. Latch the frontier while the lane is
    dead, then hand back the IDENTICAL lane counters with a fresh publish stamp: that must read as
    proof of life, because only the coordinator writes that pointer."""
    d = N.LaneLiveness(None, now=0.0, environ={})
    d.observe(0.0, event=579, accepted_seen=579, published_at=1_000)      # latched during the outage
    assert d.observe(4 * HOUR, event=579, accepted_seen=579, published_at=1_000) == "stale"
    # The coordinator restarts and RESUMES: same event, same records, new write.
    assert d.observe(4 * HOUR + 60.0, event=579, accepted_seen=579,
                     published_at=2_000) == "recovered", \
        "a republished pointer at an unchanged event must count as proof of life"
    assert d.stale is False
    # ...and the detector is not disarmed by the recovery: a lane that then goes quiet still fires.
    assert d.observe(4 * HOUR + 60.0 + N.DEFAULT_STALE_LANE_S, event=579, accepted_seen=579,
                     published_at=2_000) == "stale"


def test_a_frozen_publish_stamp_does_NOT_keep_a_dead_lane_looking_alive():
    """The new field must not become a way to look alive forever. A stamp that is merely PRESENT and
    unchanging is a dead lane, exactly as before."""
    d = N.LaneLiveness(None, now=0.0, environ={})
    d.observe(0.0, event=5, accepted_seen=5, published_at=1_000)
    assert d.observe(N.DEFAULT_STALE_LANE_S - 1.0, event=5, accepted_seen=5,
                     published_at=1_000) == "waiting"
    assert d.observe(N.DEFAULT_STALE_LANE_S, event=5, accepted_seen=5, published_at=1_000) == "stale"


def test_a_BACKWARDS_publish_stamp_is_still_not_proof_of_life():
    """THE PROPERTY THE MONOTONIC RULE EXISTS TO PROTECT, extended to the new counter. An OLDER
    stamp can only come from an older pointer object -- a stale mirror or cached manifest re-read --
    so it must not reset the staleness clock. If any CHANGE counted, a store flapping between two
    objects would keep a dead lane 'alive' forever and the detector would be unfalsifiable."""
    d = N.LaneLiveness(None, now=0.0, environ={})
    d.observe(0.0, event=5, accepted_seen=5, published_at=9_000)
    assert d.observe(10.0, event=5, accepted_seen=5, published_at=1_000) == "waiting"
    assert d.observe(N.DEFAULT_STALE_LANE_S, event=5, accepted_seen=5,
                     published_at=1_000) == "stale"
    # every counter backwards at once is still not movement
    assert d.observe(N.DEFAULT_STALE_LANE_S + 1.0, event=2, accepted_seen=1,
                     published_at=500) == "stale"


def test_a_pointer_with_NO_publish_stamp_behaves_exactly_as_before():
    """Older coordinators publish no stamp. Passing None must reproduce today's behaviour STATE FOR
    STATE -- compared against a detector driven through the old two-argument call, not against a
    hand-written expectation."""
    old = N.LaneLiveness(None, now=0.0, environ={})
    new = N.LaneLiveness(None, now=0.0, environ={})
    seq = [(0.0, 0, 0), (60.0, 0, 0), (4 * HOUR, 0, 0), (4 * HOUR + 30.0, 1, 0),
           (4 * HOUR + 60.0, 1, 0), (8 * HOUR, 1, 0), (8 * HOUR + 30.0, 0, 0)]
    old_states = [old.observe(t, event=e, accepted_seen=a) for (t, e, a) in seq]
    new_states = [new.observe(t, event=e, accepted_seen=a, published_at=None) for (t, e, a) in seq]
    assert new_states == old_states
    assert old_states[2] == "stale" and old_states[3] == "recovered", old_states
    assert (new.high_event, new.high_accepted) == (old.high_event, old.high_accepted)


def test_garbage_counters_do_not_crash_and_do_not_count_as_movement():
    d = _live(t0=0.0)
    for bad in ("nope", None, [], {}):
        assert d.observe(10.0, event=bad, accepted_seen=bad) in ("waiting", "live")
    assert d.observe(N.DEFAULT_STALE_LANE_S, event="nope", accepted_seen=None) == "stale"


# ==================================================================== 4. RECOVERY, NO RESTART
def test_it_recovers_automatically_when_the_lane_comes_back():
    d = _live(t0=0.0)
    assert d.observe(4 * HOUR, event=0, accepted_seen=0) == "stale"
    assert d.observe(4 * HOUR + 30.0, event=1, accepted_seen=0) == "recovered"
    assert d.stale is False
    assert d.observe(4 * HOUR + 60.0, event=1, accepted_seen=0) == "waiting"
    # ...and it can go stale AGAIN afterwards: recovery is not a one-shot disarm.
    assert d.observe(4 * HOUR + 30.0 + N.DEFAULT_STALE_LANE_S, event=1, accepted_seen=0) == "stale"


def test_the_alarm_is_repeated_but_not_once_per_poll():
    """A 3-hour outage polled every 60 s must not print 180 banners; it must also not print one
    banner at hour 0 and then go quiet for a day."""
    d = _live(t0=0.0)
    d.observe(4 * HOUR, event=0, accepted_seen=0)
    shouts = [t for t in range(4 * 3600, 12 * 3600, 60) if d.should_shout(float(t))]
    assert shouts[0] == 4 * 3600
    assert 12 <= len(shouts) <= 20, len(shouts)          # ~1 per 30 min over 8 h


# ==================================================================== 5. THE KNOB
def test_the_threshold_is_configurable_and_defaults_to_the_justified_value():
    assert N.LaneLiveness(None, environ={}).threshold_s == N.DEFAULT_STALE_LANE_S
    assert N.LaneLiveness(None, environ={N.STALE_LANE_ENV: "900"}).threshold_s == 900.0
    assert N.LaneLiveness(1234.0, environ={}).threshold_s == 1234.0
    # fail-soft on a typo, exactly like every other NEURAHASH_* numeric knob
    assert N.LaneLiveness(None, environ={N.STALE_LANE_ENV: "abc"}).threshold_s == \
        N.DEFAULT_STALE_LANE_S


def test_zero_disables_the_detector_entirely():
    d = N.LaneLiveness(None, now=0.0, environ={N.STALE_LANE_ENV: "0"})
    d.observe(0.0, event=0, accepted_seen=0)
    assert d.enabled is False
    assert d.observe(30 * 24 * HOUR, event=0, accepted_seen=0) == "waiting"


# ==================================================================== 6. THE MESSAGE
def test_the_banner_is_unmistakable_and_pure_ascii():
    """cp1252 console: one non-ASCII byte turns the loud warning into a UnicodeEncodeError."""
    d = N.LaneLiveness(None, now=0.0, environ={})
    d.observe(0.0, event=7, accepted_seen=3)
    assert d.observe(4 * HOUR, event=7, accepted_seen=3) == "stale"
    text = "\n".join(d.stale_banner(4 * HOUR))
    text.encode("ascii")
    assert "!" * 78 in text
    assert "LOOKS DEAD" in text
    assert "PAUSED" in text
    assert "RESUMES BY ITSELF" in text and "do NOT restart" in text
    assert N.STALE_LANE_ENV in text, "the operator is not told which knob to turn"
    assert "240 min" in text, "the banner does not say HOW LONG nothing has moved"


# ==================================================================== 7. COUNTING THE LANE
def test_count_accepted_names_counts_only_this_campaign():
    a, b = "aa" * 8, "bb" * 8                       # campaign ids are lowercase hex (>=8 chars)
    man = {N.accepted_name(0, a): {}, N.accepted_name(1, a): {},
           N.accepted_name(0, b): {}, N.GLM_POINTER_NAME: {}}
    assert N.count_accepted_names(man, a) == 2
    assert N.count_accepted_names(man, b) == 1
    assert N.count_accepted_names({}, a) == 0
    # a legacy (unscoped) lane keeps counting its own flat family
    assert N.count_accepted_names({N.accepted_name(0): {}, N.accepted_name(1): {}}, None) == 2


def test_the_old_guard_is_structurally_blind_to_this():
    """WHY THE NEW DETECTOR IS NECESSARY (this passes on the old code too -- it documents the hole,
    it is not the regression). On a shard-claim lane `comparable` is False, so the existing
    no-progress guard returns False no matter how long nothing has moved."""
    assert N.async_should_abort_no_progress(
        "local-root", "pointer-root", False, REAL_OUTAGE_S, 300.0, comparable=False) is False


# ==================================================================== 8. WIRED INTO THE REAL LOOP
class _StopTick(BaseException):
    """Cuts _run_async at a chosen point. BaseException so the loop's `except Exception` paths
    cannot swallow it."""


class _Paused(BaseException):
    """Raised from the fake sleep the stale branch performs -- reaching it PROVES the pause."""


class _Args:
    def __init__(self, d):
        self.data_dir = d
        self.mode = "tiny"
        self.rows = 12
        self.seq = 16
        self.domains = 0
        self.max_rounds = 2
        self.poll = 0.01
        self.corpus_rotate_rounds = 0
        self.stale_lane_s = None


class _Lane:
    """A lane driven by a per-tick schedule of (event_counter, wall_clock). It also OWNS the fake
    clock: `now` is what time.time() returns, and each pointer read advances it to that tick's
    scheduled value. Driving the clock off the tick number (rather than off a list of consecutive
    time.time() calls) keeps the test independent of how many times the loop happens to read the
    clock -- otherwise an unrelated edit elsewhere in the loop silently re-times this test."""

    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.n = 0
        self.now = 0.0                                    # before the first tick

    def _cur(self):
        return self.schedule[min(self.n, len(self.schedule) - 1)]

    def manifest(self):
        ev, _t = self._cur()
        return {N.accepted_name(e): {"sha256": "c%d" % e} for e in range(ev)}

    def get_json(self, cid):
        raise KeyError(cid)

    def read_pointer(self, man=None, **kw):
        if man is None:
            return None                                   # the pre-loop probe
        ev, t = self._cur()
        self.now = float(t)
        self.n += 1
        return dm.sd_pointer_encode(event=ev, slot_rounds={"0_0": ev}, model_root="root-%d" % ev)


class _ResumedLane(_Lane):
    """#160: a lane whose schedule is (event, wall_clock, publish_stamp). The coordinator RESUMES at
    a preserved frontier, so `event` and the accepted-record count NEVER move -- only the stamp does.

    The stamp key is written as the LITERAL "published_at" instead of dm.SD_PTR_PUBLISHED_AT / the
    encoder's `published_at=` argument for one deliberate reason: this regression test must be
    RUNNABLE against the PRE-FIX tree, which has neither the constant nor the parameter, or it could
    never be shown RED -- and a regression test never shown red is not evidence. The literal is
    pinned to the shipped constant by
    tests/test_sd_async_helpers.py::test_the_publish_stamp_key_is_the_one_the_wire_uses."""

    def _cur(self):
        ev, t, _stamp = self.schedule[min(self.n, len(self.schedule) - 1)]
        return ev, t

    def _stamp(self):
        return self.schedule[min(self.n, len(self.schedule) - 1)][2]

    def read_pointer(self, man=None, **kw):
        if man is None:
            return None                                   # the pre-loop probe
        ev, t = self._cur()
        stamp = self._stamp()
        self.now = float(t)
        self.n += 1
        ptr = dm.sd_pointer_encode(event=ev, slot_rounds={"0_0": ev}, model_root="root-%d" % ev)
        return dict(ptr, published_at=stamp) if stamp is not None else ptr


def _sleep_until(limit):
    """A fake time.sleep that lets the first `limit - 1` pauses pass and RAISES on the next one, so a
    loop that keeps pausing terminates the test instead of hanging it forever."""
    calls = [0]

    def _sleep(_s):
        calls[0] += 1
        if calls[0] >= limit:
            raise _Paused()
    return _sleep


def _drive(tmp_path, monkeypatch, schedule, lane=None):
    """Run _run_async far enough to pass the stale-lane seam. Returns the exception the run ended
    with, plus the log lines. `lane` overrides the default _Lane(schedule) fixture (#160)."""
    d = str(tmp_path)
    for split in ("train", "val"):
        np.save(os.path.join(d, "ids_daily_%s.npy" % split),
                np.zeros((12, 16), dtype=np.int64))
    monkeypatch.setenv("NEURAHASH_GLM_DATA_RESYNC", "0")
    monkeypatch.setattr(N, "_maybe_self_update", lambda log: None)
    monkeypatch.setattr(N, "heartbeat_miner_card", lambda *a, **k: None)
    monkeypatch.setattr(N, "claim_all_coords", lambda args, slots: [(0, 0)])
    monkeypatch.setattr(N, "claim_walk_order",
                        lambda coords, identity=None, ranked=None: list(coords))
    monkeypatch.setattr(N, "layer_claim_enabled", lambda environ=None: False)
    monkeypatch.setattr(N, "make_regate_ce", lambda *a, **k: (lambda h: 0.0))
    _state = types.SimpleNamespace(path=None,
                                   restore_cooldown=lambda cooldown, event=None: (0, 0),
                                   restore_ladder=lambda ladder: (0, 0))
    monkeypatch.setattr(N, "ClaimState",
                        types.SimpleNamespace(for_args=lambda args, ident, log=None: _state))
    lane = _Lane(schedule) if lane is None else lane
    monkeypatch.setattr(N.time, "time", lambda: lane.now)
    # Reaching the catch-up means the tick got PAST the stale gate and is about to do real work.
    def _reached(*a, **k):
        raise _StopTick()
    monkeypatch.setattr(N, "catch_up_accepted", _reached)

    args = _Args(d)
    logs = []
    try:
        N._run_async(args, lane, types.SimpleNamespace(slots=[(0, 0)]),
                     types.SimpleNamespace(), None, types.SimpleNamespace(), b"k", 0, 0, 0, "t0",
                     N.node_ids(args, 0, "train"), N.node_ids(args, 0, "val"), 16,
                     (lambda *a: logs.append(" ".join(str(x) for x in a))))
    except BaseException as e:                                   # noqa: BLE001
        return e, logs
    return None, logs


def test_a_live_lane_reaches_the_training_path(tmp_path, monkeypatch):
    """NEGATIVE CONTROL AT LOOP LEVEL: only 60 s since the process started -> normal work."""
    monkeypatch.setattr(N.time, "sleep", lambda s: (_ for _ in ()).throw(_Paused()))
    end, logs = _drive(tmp_path, monkeypatch, [(5, 60.0)])
    assert isinstance(end, _StopTick), "a live lane must not be paused: %r" % (end,)
    assert not any("LOOKS DEAD" in l for l in logs)


def test_a_slow_lane_still_reaches_the_training_path(tmp_path, monkeypatch):
    """NEGATIVE CONTROL AT LOOP LEVEL, the one that matters: 660 s -- the slowest legitimate round
    ever measured here -- with a counter that did move. The miner must keep working."""
    monkeypatch.setattr(N.time, "sleep", lambda s: (_ for _ in ()).throw(_Paused()))
    end, logs = _drive(tmp_path, monkeypatch, [(6, SLOW_STEP_S)])
    assert isinstance(end, _StopTick), "paused a lane that was merely SLOW: %r" % (end,)
    assert not any("LOOKS DEAD" in l for l in logs)


def test_a_dead_lane_pauses_before_any_training(tmp_path, monkeypatch):
    """THE REGRESSION -- the old loop trains forever here. Frozen counters, 4 h of wall clock: the
    tick must reach the PAUSE and never the work path."""
    monkeypatch.setattr(N.time, "sleep", lambda s: (_ for _ in ()).throw(_Paused()))
    end, logs = _drive(tmp_path, monkeypatch, [(5, 4 * HOUR)])
    assert isinstance(end, _Paused), "the miner kept working on a dead lane: %r" % (end,)
    joined = "\n".join(logs)
    assert "LOOKS DEAD" in joined, joined[-2000:]
    assert "PAUSED" in joined
    assert "stale-lane detector ON" in joined


def test_a_paused_miner_resumes_by_itself(tmp_path, monkeypatch):
    """No human restart. The lane goes dead, then publishes again; the very next tick recovers and
    goes back to work."""
    monkeypatch.setattr(N.time, "sleep", lambda s: None)         # let the pause return, not raise
    end, logs = _drive(tmp_path, monkeypatch, [(5, 4 * HOUR), (9, 4 * HOUR + 60.0)])
    joined = "\n".join(logs)
    assert "LOOKS DEAD" in joined
    assert "LANE RECOVERED" in joined, joined[-2000:]
    assert isinstance(end, _StopTick), "did not resume real work after recovery: %r" % (end,)


def test_a_RESUMED_coordinator_at_the_SAME_event_gets_the_miner_back_to_work(tmp_path, monkeypatch):
    """THE #160 REGRESSION AT LOOP LEVEL -- red on the pre-fix tree, and the one that reproduces the
    six-day outage exactly.

    Tick 1: the miner latches the frontier of a lane whose coordinator has been gone 4 h -> PAUSE.
    Tick 2: the coordinator is back and RESUMES: it republishes the SAME event=579 with the SAME
    record count, and only the publish stamp is new. Every lane counter the pre-fix miner watched is
    byte-identical to what it latched, so the pre-fix loop pauses again, forever -- the event counter
    can only advance when a miner submits, and no miner ever will. The fixed loop must reach the work
    path (_StopTick) on that very tick, with no restart.

    The fake sleep raises on the SECOND pause, so 'still wedged' fails fast instead of hanging."""
    monkeypatch.setattr(N.time, "sleep", _sleep_until(2))
    lane = _ResumedLane([(579, 4 * HOUR, 1_000), (579, 4 * HOUR + 60.0, 2_000)])
    end, logs = _drive(tmp_path, monkeypatch, None, lane=lane)
    joined = "\n".join(logs)
    assert "LOOKS DEAD" in joined, "the miner never reached the wedged state: %s" % joined[-2000:]
    assert isinstance(end, _StopTick), (
        "WEDGED: the coordinator republished event=579 (frontier preserved) and the miner stayed in "
        "the stale-lane pause -- this is the six-day 4060 outage (#160): %r" % (end,))
    assert "LANE RECOVERED" in joined, joined[-2000:]


def test_a_resumed_lane_WITHOUT_a_publish_stamp_still_behaves_exactly_as_today(tmp_path,
                                                                               monkeypatch):
    """The other side of the same seam: against a pre-#160 coordinator (no stamp on the wire) the
    miner must behave EXACTLY as it does today -- it stays paused on a frozen lane and does NOT
    invent liveness out of the missing field."""
    monkeypatch.setattr(N.time, "sleep", _sleep_until(2))
    lane = _ResumedLane([(579, 4 * HOUR, None), (579, 4 * HOUR + 60.0, None)])
    end, logs = _drive(tmp_path, monkeypatch, None, lane=lane)
    assert isinstance(end, _Paused), (
        "an unstamped frozen lane must still read as dead: %r" % (end,))
    assert "LANE RECOVERED" not in "\n".join(logs)
