"""Restart durability of the GLM shard-claim walk: cooldown parks + the walk cursor survive a
process restart, and expire honestly.

WHY THIS FILE EXISTS -- MEASURED, do not re-derive
(docs/research/RUN5_CONCENTRATION_2026-07-27.md, memory run5-idle-and-register-gate):
the 5090 miner process started 28 times in 15.7 h. Both pieces of claim-walk state lived only in
RAM, so every start re-claimed the SAME wallet-hash head (L1,E12) -- pick_start_coord is a constant
for a given wallet -- with an EMPTY cooldown table. Each new process therefore re-walked coordinates
a previous one had already parked, re-parked them (125 catch-up stalls, 5.20 h), dropped into repair
mode with nothing left to train, and was killed by the supervisor for making no rounds. Net: 25 of 28
cycles trained (L1,E12) and nothing else; 38 coordinates were claimed and instantly re-parked; 53 of
60 claimable coordinates were never trained at all.

The guard below is the failure itself, not a proxy: process 1 parks a coordinate and lands past it,
process 2 is constructed FRESH against the same --data-dir, and must (a) still see the park and
(b) resume on the coordinate process 1 landed on rather than back at the head.

Two properties are load-bearing in the other direction, and each has its own test:
  * EXPIRY. A park is a wall-clock lease (900 s / 10 events). Restoring one whose deadline has passed
    would resurrect stale parks hours later and starve the miner of coordinates -- the exact opposite
    failure. And the event half is an ABSOLUTE count against a campaign's event counter, so it is
    meaningless against a counter that restarted (a resumed coordinator republishes genesis at
    event 0); it is kept only while the live counter is at or past the one it was measured against.
  * NEVER FATAL. A truncated / garbage / wrong-schema file is a WARN and a clean start. This state is
    an optimisation; crashing a miner over it would cost more than never having written it.

Run: C:/Python313/python.exe -m pytest tests/test_claim_state_persistence.py -q
"""
import argparse
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_glm_contributor as N                               # noqa: E402

IDENT = "0x5168F6aB0000000000000000000000000000DC66"
CLAIMABLE = [(1, e) for e in (2, 7, 12, 18, 31, 50)]


def _silent(*_a, **_k):
    pass


class _Clock:
    """Injectable monotonic-ish clock: CoordCooldown is pure except for this, which is what makes
    900 s cooldowns testable without sleeping."""

    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += float(dt)
        return self.t


def _args(data_dir, expert=None, slot=None):
    """The miner namespace resolve_claim actually reads. node_claimable_coords is monkeypatched per
    test, so no shard manifest is needed -- this file is about persistence, not claimability.
    tmp_path is absolute, which is what _claim_state_path requires (see the relative-dir test)."""
    return argparse.Namespace(data_dir=str(data_dir), expert=expert, slot=slot, mode="glm",
                              slots="1:0", domains="daily", shard_dir=None)


@pytest.fixture(autouse=True)
def _no_env_expert(monkeypatch):
    monkeypatch.delenv("NEURAHASH_SD_EXPERT", raising=False)


@pytest.fixture
def claimable(monkeypatch):
    monkeypatch.setattr(N, "node_claimable_coords", lambda _a: list(CLAIMABLE))
    return list(CLAIMABLE)


def _walk_positions(identity, coords):
    """(head, next_after_head, second_after_head) in THIS identity's real walk order -- the same
    order advance_claim walks, so the fixture cannot drift from the code under test."""
    order = N.claim_walk_order(coords, identity)
    k = order.index(N.pick_start_coord(coords, identity))
    return order[k], order[(k + 1) % len(order)], order[(k + 2) % len(order)]


# ============================================================ 1. the run-5 failure, end to end
class TestRestartDoesNotForget:
    """(a) the park survives, (b) the walk resumes past it. Both FAIL on the pre-change code."""

    def test_restart_keeps_the_park_and_resumes_past_it(self, tmp_path, claimable):
        head, parked, landed = _walk_positions(IDENT, claimable)
        clock, wall = _Clock(), _Clock(1_700_000_000.0)

        # ---- process 1: claims the head, plateaus, parks `parked`, lands on `landed` -------------
        args = _args(tmp_path)
        L, E, _i, src = N.resolve_claim(args, [], log=_silent, identity=IDENT)
        assert (L, E) == head and src == "wallet-hash (auto-spread)"
        cd1 = N.CoordCooldown(seconds=900.0, events=10, now=clock)
        cd1.park(parked, 30, "catch-up stall")              # exactly what advance_claim does on a bound
        st1 = N.ClaimState.for_args(args, IDENT, log=_silent)
        assert st1.save(cd1, landed, event=30, now=wall) is True
        assert os.path.isfile(st1.path), "the state file must land inside --data-dir"

        # ---- process 2: FRESH objects, same --data-dir, 60 s later ------------------------------
        clock2, wall2 = _Clock(), _Clock(wall.t + 60.0)     # a new process: its clock zero is its own
        args2 = _args(tmp_path)
        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=clock2)
        st2 = N.ClaimState.for_args(args2, IDENT, log=_silent)
        restored, dropped = st2.restore_cooldown(cd2, event=31, now=wall2)

        # (a) the park is still there, and still has ~840 s of its 900 s lease left
        assert (restored, dropped) == (1, 0)
        assert cd2.blocked(parked, 31) is True
        assert 830.0 <= cd2.left(parked, 31)[0] <= 845.0
        # (b) the walk resumes where process 1 left it, NOT at the constant wallet-hash head
        L2, E2, _i2, src2 = N.resolve_claim(args2, [], log=_silent, identity=IDENT)
        assert (L2, E2) == landed
        assert (L2, E2) != head, "restart re-claimed the head -- this is the run-5 amnesia"
        assert src2 == "resumed walk cursor"

    def test_a_second_advance_moves_the_saved_cursor(self, tmp_path, claimable):
        """The cursor is not write-once: the last landing wins, so a long-lived process's progress is
        what a restart inherits."""
        _head, first, second = _walk_positions(IDENT, claimable)
        args, wall = _args(tmp_path), _Clock(1_700_000_000.0)
        st = N.ClaimState.for_args(args, IDENT, log=_silent)
        cd = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        st.save(cd, first, event=1, now=wall)
        st.save(cd, second, event=2, now=wall)
        assert N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent).cursor() == second

    def test_a_pinned_expert_ignores_the_cursor(self, tmp_path, claimable):
        """--expert is an operator instruction; a stale state file must never override it."""
        _head, _parked, landed = _walk_positions(IDENT, claimable)
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(N.CoordCooldown(now=_Clock()), landed, event=1, now=_Clock(1_700_000_000.0))
        pinned = next(c for c in claimable if c != landed)
        L, E, _i, src = N.resolve_claim(_args(tmp_path, expert="%d:%d" % pinned), [],
                                        log=_silent, identity=IDENT)
        assert (L, E) == pinned and src == "--expert"

    def test_a_cursor_this_node_no_longer_holds_falls_back_to_the_hash(self, tmp_path, claimable,
                                                                       monkeypatch):
        """Re-pointed --pieces: the saved coordinate is not claimable here any more. Falling back to
        the head is the only safe answer -- claiming it would train an inert expert."""
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(N.CoordCooldown(now=_Clock()), (9, 63), event=1, now=_Clock(1_700_000_000.0))
        L, E, _i, src = N.resolve_claim(_args(tmp_path), [], log=_silent, identity=IDENT)
        assert (L, E) == N.pick_start_coord(claimable, IDENT) and src == "wallet-hash (auto-spread)"


# ============================================================ 2. expiry, both dimensions
class TestExpiry:
    def test_a_park_whose_deadline_passed_is_dropped_at_load(self, tmp_path):
        """A restart HOURS later must start clean, or every coordinate is parked and the miner
        starves -- the opposite of the failure this feature fixes."""
        wall = _Clock(1_700_000_000.0)
        cd1 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        cd1.park((1, 7), 30, "catch-up stall")
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(cd1, (1, 12), event=30, now=wall)

        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        restored, dropped = st.restore_cooldown(cd2, event=99, now=_Clock(wall.t + 3 * 3600.0))
        assert (restored, dropped) == (0, 1)
        assert cd2.blocked((1, 7), 99) is False, "a 3h-old 900s park was resurrected"

    def test_a_park_that_is_still_live_survives_partially_decayed(self, tmp_path):
        wall = _Clock(1_700_000_000.0)
        cd1 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        cd1.park((1, 7), 30, "register refused (no seat)")
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(cd1, (1, 12), event=30, now=wall)
        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert st.restore_cooldown(cd2, event=30, now=_Clock(wall.t + 300.0)) == (1, 0)
        assert 595.0 <= cd2.left((1, 7), 30)[0] <= 605.0
        assert cd2.reason((1, 7)).startswith("register refused")

    def test_the_event_half_is_dropped_when_the_counter_went_backwards(self, tmp_path):
        """A resumed coordinator republishes genesis at event 0. `until_event=40` restored against
        that counter would keep the coordinate parked for a whole new campaign, because blocked()
        holds while EITHER half is unelapsed. Scope it to the counter it was measured against."""
        wall = _Clock(1_700_000_000.0)
        cd1 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        cd1.park((1, 7), 30, "catch-up stall")               # -> until_event = 40
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(cd1, (1, 12), event=30, now=wall)

        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        st.restore_cooldown(cd2, event=0, now=_Clock(wall.t + 10.0))     # NEW campaign, counter reset
        assert cd2.left((1, 7), 0)[1] == 0, "an event deadline from a dead campaign was honoured"
        assert cd2.blocked((1, 7), 0) is True                            # wall clock still parks it
        cd2._now = _Clock(cd2._now() + 1000.0)                           # ... and expires it alone
        assert cd2.blocked((1, 7), 0) is False

    def test_the_event_half_is_kept_within_the_same_campaign(self, tmp_path):
        """Same counter, moved forward -> the deadline still means what it meant, so keep it: that
        half exists so a FAST lane cannot expire a park merely because 15 minutes passed."""
        wall = _Clock(1_700_000_000.0)
        cd1 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        cd1.park((1, 7), 30, "catch-up stall")
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(cd1, (1, 12), event=30, now=wall)
        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        st.restore_cooldown(cd2, event=33, now=_Clock(wall.t + 10.0))
        assert cd2.left((1, 7), 33)[1] == 7                      # 40 - 33
        cd2._now = _Clock(cd2._now() + 10_000.0)                 # seconds elapsed, events have not
        assert cd2.blocked((1, 7), 33) is True
        assert cd2.blocked((1, 7), 40) is False                  # both halves elapsed -> claimable

    def test_an_event_deadline_stamped_before_a_mid_process_reset_is_clamped(self, tmp_path):
        """The nastier reset: the coordinator restarts while the miner is still UP. The park was made
        at event 30 (until_event=40); the counter is then republished at genesis, so the save stamps
        at_event=2 next to the stale 40. On restart the live counter (5) is >= 2, so the campaign
        looks unchanged and the raw 40 would be honoured -- parking the coordinate until a brand-new
        campaign reaches event 40, which the 900 s lease can never override (blocked() holds while
        EITHER half is unelapsed). A deadline past at_event + events cannot have been stamped against
        at_event, so it is clamped."""
        wall = _Clock(1_700_000_000.0)
        cd1 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        cd1.park((1, 7), 30, "catch-up stall")                    # -> until_event = 40
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(cd1, (1, 12), event=2, now=wall)                  # coordinator restarted: counter = 2
        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert st.restore_cooldown(cd2, event=5, now=_Clock(wall.t + 1.0)) == (1, 0)
        assert cd2.left((1, 7), 5)[1] == 7                        # clamped to 2 + 10, not 40
        cd2._now = _Clock(cd2._now() + 100_000.0)                 # the lease expires...
        assert cd2.blocked((1, 7), 12) is False                   # ... and the coordinate comes back

    def test_an_unknown_live_counter_drops_the_event_half(self, tmp_path):
        """No pointer yet (the read is inside a try/except) -> we cannot prove the campaign, so the
        conservative half is the wall clock."""
        wall = _Clock(1_700_000_000.0)
        cd1 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        cd1.park((1, 7), 30, "catch-up stall")
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(cd1, (1, 12), event=30, now=wall)
        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        st.restore_cooldown(cd2, event=None, now=_Clock(wall.t + 10.0))
        assert cd2.left((1, 7), 0)[1] == 0

    def test_export_omits_a_park_that_already_elapsed(self):
        clock = _Clock()
        cd = N.CoordCooldown(seconds=900.0, events=10, now=clock)
        cd.park((1, 7), 30, "catch-up stall")
        cd.park((1, 8), 30, "catch-up stall")
        clock.tick(901.0)
        cd.park((1, 8), 30, "catch-up stall")                     # re-parked -> fresh 900 s
        got = cd.export(event=30)
        assert [(p["L"], p["E"]) for p in got["parked"]] == [(1, 8)]
        assert got["at_event"] == 30


# ============================================================ 3. never fatal
class TestCorruptStateStartsClean:
    @pytest.mark.parametrize("body", [
        '{"schema": 1, "coord": [1, 12], "coold',                 # truncated mid-write
        "",                                                       # zero-length
        "\x00\x00\x00\x00",                                       # garbage bytes
        '["not", "an", "object"]',                                # wrong top-level type
        '{"schema": 99, "coord": [1, 12]}',                       # a future schema
        '{"schema": 1, "coord": "not-a-coord"}',                  # right schema, wrong field type
    ])
    def test_unusable_file_is_a_warn_not_an_exception(self, tmp_path, claimable, body):
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        with open(st.path, "w", encoding="utf-8") as fh:
            fh.write(body)
        warned = []
        st2 = N.ClaimState.for_args(_args(tmp_path), IDENT, log=lambda m: warned.append(m))
        assert st2.cursor() is None
        cd = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert st2.restore_cooldown(cd, event=5) == (0, 0)
        # ... and the miner starts on the head instead of dying
        L, E, _i, src = N.resolve_claim(_args(tmp_path), [], log=_silent, identity=IDENT)
        assert (L, E) == N.pick_start_coord(claimable, IDENT) and src == "wallet-hash (auto-spread)"

    @pytest.mark.parametrize("cooldown_block", [
        [1, 2],                                                   # a list where an object belongs
        "x",                                                      # a string
        dict(at_event=30, parked=5),                              # parked is not iterable
        dict(at_event="soon", parked=[]),                         # at_event is not a number
        dict(at_event=30, parked=[dict(L=1, E=7, left_s=100.0, until_event="never")]),
        dict(at_event=30, parked=[7, None, "row"]),               # rows that are not objects
        dict(parked=[dict(L=1, E=7, left_s=float("nan"), until_event=40)]),
    ])
    def test_a_nested_type_error_starts_clean_instead_of_boot_looping(self, tmp_path,
                                                                      cooldown_block):
        """The file is JSON-valid and schema-correct, so load() waves it through -- an operator
        hand-editing it to clear a stuck park lands exactly here. A raise would traceback out of
        _run_async BEFORE the first save, so the bad file is never rewritten and every supervisor
        restart dies the same way: a permanent boot loop caused by an optimisation."""
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        with open(st.path, "w", encoding="utf-8") as fh:
            json.dump(dict(schema=N.CLAIM_STATE_SCHEMA, saved_wall=1_700_000_000.0, coord=[1, 12],
                           cooldown=cooldown_block), fh)
        cd = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        restored, _dropped = st.restore_cooldown(cd, event=30, now=_Clock(1_700_000_000.0))
        assert restored == 0
        assert st.cursor() == (1, 12)              # the readable half of the file still works

    def test_a_nan_cooldown_is_never_persisted_and_never_restored(self, tmp_path):
        """nan compares False against every clock, so a nan deadline is a park nothing can expire and
        no restart can clear. It must die at BOTH ends: export drops it, restore drops it."""
        cd = N.CoordCooldown(seconds=float("nan"), events=10, now=_Clock())
        cd.park((1, 7), 30, "catch-up stall")
        assert cd.blocked((1, 7), 999999) is True                  # the in-memory pathology...
        assert cd.export(event=30)["parked"] == []                 # ... is not written to disk
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        with open(st.path, "w", encoding="utf-8") as fh:
            json.dump(dict(schema=N.CLAIM_STATE_SCHEMA, saved_wall=1_700_000_000.0, coord=[1, 12],
                           cooldown=dict(at_event=30, parked=[
                               dict(L=1, E=7, reason="nan", left_s=float("nan"), until_event=40)])),
                      fh)
        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert st.restore_cooldown(cd2, event=30, now=_Clock(1_700_000_000.0)) == (0, 1)
        assert cd2.blocked((1, 7), 30) is False

    def test_a_malformed_park_row_is_skipped_not_raised(self, tmp_path):
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        with open(st.path, "w", encoding="utf-8") as fh:
            json.dump(dict(schema=N.CLAIM_STATE_SCHEMA, saved_wall=1_700_000_000.0, coord=[1, 12],
                           cooldown=dict(at_event=30, parked=[
                               dict(L="x", E=7, reason="bad", left_s=100.0, until_event=40),
                               dict(L=1, E=8, reason="good", left_s=100.0, until_event=40)])), fh)
        cd = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert st.restore_cooldown(cd, event=30, now=_Clock(1_700_000_000.0)) == (1, 1)
        assert cd.blocked((1, 8), 30) is True

    def test_an_unwritable_path_is_a_warn_not_an_exception(self, tmp_path):
        """A save that cannot land must never propagate: the run is worth more than the file."""
        args = _args(os.path.join(str(tmp_path), "nope"))
        st = N.ClaimState.for_args(args, IDENT, log=_silent)
        st.path = os.path.join(str(tmp_path), "a-directory-not-a-file")
        os.makedirs(st.path)
        warned = []
        st._log = lambda m: warned.append(m)
        assert st.save(N.CoordCooldown(now=_Clock()), (1, 12), event=1) is False
        assert warned and "could not save" in warned[0]

    def test_no_data_dir_is_a_no_op(self):
        st = N.ClaimState.for_args(argparse.Namespace(), IDENT, log=_silent)
        assert st.path is None
        assert st.cursor() is None
        assert st.save(N.CoordCooldown(now=_Clock()), (1, 12), event=1) is False

    @pytest.mark.parametrize("rel", [".", "miner", "./out/miner", ""])
    def test_a_relative_data_dir_disables_persistence_and_writes_nothing(self, tmp_path, claimable,
                                                                         monkeypatch, rel):
        """A relative --data-dir resolves against the launcher's CWD, and this state exists to be
        found by the NEXT process -- a supervisor respawn or a self-update re-exec makes no promise
        about CWD. Rather than scatter half-remembered state (and litter whatever directory the miner
        happened to start in), persistence is OFF; _run_async says so loudly. Guard against a
        regression that starts writing files next to the caller."""
        monkeypatch.chdir(tmp_path)
        st = N.ClaimState.for_args(_args(rel), IDENT, log=_silent)
        assert st.path is None
        assert st.save(N.CoordCooldown(now=_Clock()), (1, 12), event=1) is False
        assert os.listdir(str(tmp_path)) == []
        L, E, _i, src = N.resolve_claim(_args(rel), [], log=_silent, identity=IDENT)
        assert (L, E) == N.pick_start_coord(claimable, IDENT) and src == "wallet-hash (auto-spread)"


# ============================================================ 4. file hygiene
class TestFileHygiene:
    def test_two_identities_sharing_one_data_dir_do_not_collide(self, tmp_path, claimable):
        """--data-dir is shareable by construction: its files are named by DOMAIN, never by miner
        (_ids_path), so two miners on one box legitimately point at the same dir."""
        other = "0x361447E30000000000000000000000000000AAAA"
        wall = _Clock(1_700_000_000.0)
        a = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        b = N.ClaimState.for_args(_args(tmp_path), other, log=_silent)
        assert a.path != b.path
        cd_a = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        cd_a.park((1, 7), 5, "catch-up stall")
        a.save(cd_a, (1, 12), event=5, now=wall)
        b.save(N.CoordCooldown(now=_Clock()), (1, 50), event=5, now=wall)
        assert a.cursor() == (1, 12) and b.cursor() == (1, 50)
        cd_b = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert b.restore_cooldown(cd_b, event=5, now=_Clock(wall.t)) == (0, 0)

    def test_save_leaves_no_temp_file_and_rewrites_in_place(self, tmp_path):
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        wall = _Clock(1_700_000_000.0)
        st.save(N.CoordCooldown(now=_Clock()), (1, 12), event=1, now=wall)
        st.save(N.CoordCooldown(now=_Clock()), (1, 18), event=2, now=wall)
        names = sorted(os.listdir(str(tmp_path)))
        assert names == [os.path.basename(st.path)], names
        with open(st.path, "r", encoding="utf-8") as fh:
            assert json.load(fh)["coord"] == [1, 18]

    def test_an_unchanged_state_is_not_rewritten(self, tmp_path):
        """The repair-mode retry calls advance_claim (and therefore save) every iteration; when
        nothing changed the deadlines on disk are still correct, so the write is skipped."""
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        cd = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        cd.park((1, 7), 5, "catch-up stall")
        wall = _Clock(1_700_000_000.0)
        assert st.save(cd, (1, 12), event=5, now=wall) is True
        wall.tick(5.0)
        assert st.save(cd, (1, 12), event=5, now=wall) is False    # identical -> skipped
        cd.park((1, 8), 5, "catch-up stall")
        assert st.save(cd, (1, 12), event=5, now=wall) is True     # a new park -> written

    def test_parks_made_the_way_advance_claim_makes_them_round_trip(self, tmp_path, claimable):
        """advance_claim's only two park sites are register-refused and catch-up-abort; both call
        cooldown.park(cand, event, reason). Whatever it builds must survive verbatim."""
        clock, wall = _Clock(), _Clock(1_700_000_000.0)
        cd = N.CoordCooldown(seconds=900.0, events=10, now=clock)
        for c in claimable[1:]:
            cd.park(c, 12, "register refused (no seat under --max-active-slots)")
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(cd, claimable[0], event=12, now=wall)
        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert st.restore_cooldown(cd2, event=12, now=_Clock(wall.t)) == (len(claimable) - 1, 0)
        assert cd2.describe(claimable, 12) == cd.describe(claimable, 12)


# ============================================================ 5. the real advance path, end to end
class _StubHost:
    """Just enough GlmExpertLaneHost for advance_claim: register() hands back a local index."""

    def __init__(self, coords):
        self.slots = [tuple(c) for c in coords]

    def register(self, L, E):
        if (int(L), int(E)) not in self.slots:
            self.slots.append((int(L), int(E)))
        return self.slots.index((int(L), int(E)))


class TestRealAdvanceClaimIsWhatGetsPersisted:
    """The unit tests above exercise ClaimState directly. This one drives the REAL advance_claim --
    the only mutator of both the park table and the cursor, and therefore the only thing the save
    point in _run_async has to capture -- and then proves the restored parks actually change the
    next process's walk. That last assertion is the goal metric: a restarted miner must not re-try
    the wall the previous one already hit."""

    @staticmethod
    def _stall_on(coords, monkeypatch):
        """resume_to_root reports a catch-up abort for `coords` -- the run-5 'catch-up stall'."""
        stalling = {tuple(c) for c in coords}

        def _fake(host, lane, target_root, log, own_coord=None, outcome=None, **_kw):
            if tuple(own_coord) in stalling:
                (outcome if outcome is not None else {}).update(
                    aborted=True, reason="stall", elapsed_s=66.1)
                return 0, False
            (outcome if outcome is not None else {}).update(reason="empty", elapsed_s=0.1)
            return 0, True

        monkeypatch.setattr(N, "resume_to_root", _fake)

    def test_a_restart_re_walks_every_parked_coordinate_without_the_restore_and_none_with_it(
            self, tmp_path, claimable, monkeypatch):
        """Run 5 cycles 3-27, reproduced and then fixed, with the control INLINE so the two arms
        cannot drift apart: claim the head, plateau, walk the rest, park every one on a catch-up
        stall, get killed for making no rounds. The restarted process either re-walks all 5 (what
        happened: 38 claim-and-re-park cycles) or skips all 5 (what should happen)."""
        head, first, _second = _walk_positions(IDENT, claimable)
        self._stall_on(claimable, monkeypatch)                  # EVERY coordinate stalls its catch-up
        wall = _Clock(1_700_000_000.0)

        # ---- process 1: parks all 5 others, lands nowhere -> 1.5(b) repair mode ------------------
        cd1 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert N.advance_claim(_StubHost(claimable), None, claimable, head, IDENT, None, "root", 30,
                               cd1, _silent, "m0", plateau_rejects=3) is None
        parked = [c for c in claimable if c != head]
        assert all(cd1.blocked(c, 30) for c in parked) and len(parked) == 5
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        assert st.save(cd1, head, event=30, now=wall) is True   # the _run_async save point

        def _retried(cooldown):
            """Which coordinates the restarted walk actually pays a catch-up for. The wall is still
            up (every coordinate stalls), exactly as it was for run 5's next restart."""
            seen = []
            orig = N.resume_to_root

            def _spy(host, lane, target_root, log, own_coord=None, outcome=None, **kw):
                seen.append(tuple(own_coord))
                return orig(host, lane, target_root, log, own_coord=own_coord, outcome=outcome, **kw)

            monkeypatch.setattr(N, "resume_to_root", _spy)
            N.advance_claim(_StubHost(claimable), None, claimable, head, IDENT, None, "root", 31,
                            cooldown, _silent, "m0", plateau_rejects=3)
            monkeypatch.setattr(N, "resume_to_root", orig)
            return seen

        # ---- control: the pre-change restart -- fresh process, EMPTY table ----------------------
        amnesiac = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert sorted(_retried(amnesiac)) == sorted(parked), "control: it re-walks all 5"
        assert all(amnesiac.blocked(c, 31) for c in parked)   # ... and re-parks all 5, for nothing

        # ---- fixed: same restart, table restored from disk 60 s later ---------------------------
        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        assert st.restore_cooldown(cd2, event=31, now=_Clock(wall.t + 60.0)) == (5, 0)
        assert _retried(cd2) == [], ("a restarted miner re-tried coordinates the previous process "
                                     "had already parked -- the run-5 waste, 5 of 5 re-walked")
        assert cd2.blocked(first, 31) is True

    def test_register_refusals_are_parked_and_persisted_too(self, tmp_path, claimable, monkeypatch):
        """advance_claim's other park site: no seat under --max-active-slots. 38 coordinates were
        claimed-and-re-parked in run 5, so both reasons have to survive."""
        head, first, _second = _walk_positions(IDENT, claimable)
        self._stall_on([], monkeypatch)

        class _NoSeat(_StubHost):
            def register(self, L, E):
                if (int(L), int(E)) == first:
                    raise RuntimeError("no seat under --max-active-slots")
                return _StubHost.register(self, L, E)

        cd1 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        N.advance_claim(_NoSeat(claimable), None, claimable, head, IDENT, None, "root", 30, cd1,
                        _silent, "m0", plateau_rejects=3)
        assert "register refused" in (cd1.reason(first) or "")
        wall = _Clock(1_700_000_000.0)
        st = N.ClaimState.for_args(_args(tmp_path), IDENT, log=_silent)
        st.save(cd1, head, event=30, now=wall)
        cd2 = N.CoordCooldown(seconds=900.0, events=10, now=_Clock())
        st.restore_cooldown(cd2, event=30, now=_Clock(wall.t + 1.0))
        assert "register refused" in (cd2.reason(first) or "")
