"""Tests for the ALPHA-2 truly-decoupled async-lane primitives in neurahash/diloco_merge.py (#146, W1):
token_quality_weight, sd_pointer_encode/sd_pointer_decode (v1<->v2), and SlotClock.

All four are PURE (no I/O, no time), so they are tested directly. The load-bearing contracts are: the
weight formula matches the paper exactly and is zero-guarded; the v2 pointer round-trips AND stays
readable by a v1-only reader; a real v1 pointer (the {round, state_cid, done} shape actually published by
tools/sharddiloco_harness.py) normalizes correctly; malformed input raises a NAMED ValueError; and
SlotClock's global event stays strictly monotonic while per-slot rounds advance independently, with
fresh() delegating to staleness_ok (no re-implemented policy).

The tests live in one class named for the shardDiLoCo lane so the lane's usual selector
`-k "diloco or sharddiloco"` picks up this file (the module name alone does not contain the substring;
the class name is inherited as a keyword by every method, and pytest matches -k case-insensitively).
"""
import pytest

from neurahash import diloco_merge as dm


def _real_v1_pointer(rnd, state_cid, done=False):
    """The EXACT shape tools/sharddiloco_harness.py publish_pointer emits:
    dict(round=int(rnd), state_cid=state_cid, done=bool(done)) -- built here with the REAL field names so
    this test breaks loudly if the v1 wire ever drifts."""
    return dict(round=int(rnd), state_cid=state_cid, done=bool(done))


class TestShardDiLoCoAsyncHelpers:

    # -------------------------------------------------------------- token_quality_weight (paper Alg 2)

    def test_weight_exact_values(self):
        # weight = tokens * (tokens / steps): quantity x quality (tokens-per-step).
        assert dm.token_quality_weight(10, 100) == 1000.0   # 100 * (100/10) = 100 * 10
        assert dm.token_quality_weight(4, 8) == 16.0         # 8 * (8/4) = 8 * 2
        assert dm.token_quality_weight(1, 1) == 1.0          # 1 * (1/1)
        assert dm.token_quality_weight(100, 100) == 100.0    # 100 * (100/100) = 100 * 1
        # denser (same tokens, fewer steps) weighs strictly more than sparser.
        assert dm.token_quality_weight(2, 100) > dm.token_quality_weight(50, 100)

    def test_weight_is_float(self):
        assert isinstance(dm.token_quality_weight(4, 8), float)

    @pytest.mark.parametrize("steps,tokens", [
        (0, 100),     # zero steps (would divide-by-zero) -> guarded
        (-1, 100),    # negative steps
        (10, 0),      # zero tokens
        (10, -5),     # negative tokens
        (0, 0),       # both zero
        (-3, -3),     # both negative
    ])
    def test_weight_zero_guards(self, steps, tokens):
        assert dm.token_quality_weight(steps, tokens) == 0.0

    # -------------------------------------------------------------- pointer v2 encode/decode

    def test_v2_encode_shape_and_aliases(self):
        p = dm.sd_pointer_encode(event=7, slot_rounds={"1_0": 3, "1_1": 2}, model_root="root-abc",
                                 done=False)
        assert p["v"] == 2
        assert p["event"] == 7
        assert p["rounds"] == {"1_0": 3, "1_1": 2}
        assert p["model_root"] == "root-abc"
        assert p["done"] is False
        # v1 back-compat aliases: a v1-only reader looks at `round` (must be the monotonic event) and
        # `state_cid` (must be the model root). These MUST be present in a v2 pointer.
        assert p["round"] == 7
        assert p["state_cid"] == "root-abc"

    def test_v2_encode_decode_round_trip(self):
        p = dm.sd_pointer_encode(event=42, slot_rounds={"1_0": 5, "1_1": 4}, model_root="cid-xyz",
                                 done=True)
        d = dm.sd_pointer_decode(p)
        assert d == {
            "v": 2,
            "event": 42,
            "slot_rounds": {"1_0": 5, "1_1": 4},
            "model_root": "cid-xyz",
            "done": True,
        }

    def test_v2_encode_none_slot_rounds_is_empty_map(self):
        p = dm.sd_pointer_encode(event=1, slot_rounds=None, model_root="r", done=False)
        assert p["rounds"] == {}
        d = dm.sd_pointer_decode(p)
        # a v2 pointer with an empty map decodes to {} (NOT None); None is reserved for the v1 shape.
        assert d["slot_rounds"] == {}
        assert d["v"] == 2

    def test_v2_encode_coerces_ints(self):
        # event / rounds values arriving as strings (e.g. via JSON) are normalized to int on both sides.
        p = dm.sd_pointer_encode(event="9", slot_rounds={"1_0": "6"}, model_root="r")
        assert p["event"] == 9 and p["rounds"]["1_0"] == 6
        d = dm.sd_pointer_decode({"v": 2, "event": "9", "rounds": {"1_0": "6"}, "model_root": "r"})
        assert d["event"] == 9 and d["slot_rounds"]["1_0"] == 6

    def test_v2_pointer_is_readable_as_v1_monotonic_int(self):
        # Simulate a v1-only reader (knows ONLY `round` + `state_cid`) seeing a sequence of v2 pointers:
        # `round` must be a strictly increasing int so the old client still advances correctly.
        prev = -1
        for ev in (0, 1, 5, 100):
            p = dm.sd_pointer_encode(event=ev, slot_rounds={"1_0": ev}, model_root="r%d" % ev)
            v1_round = p["round"]           # the ONLY progress field a v1 reader consults
            assert isinstance(v1_round, int)
            assert v1_round > prev
            prev = v1_round
            assert p["state_cid"] == "r%d" % ev

    # -------------------------------------------------------------- pointer v1 normalization

    def test_v1_pointer_normalizes(self):
        v1 = _real_v1_pointer(11, "state-cid-11", done=True)
        d = dm.sd_pointer_decode(v1)
        assert d == {
            "v": 1,
            "event": 11,                    # v1 `round` -> normalized `event`
            "slot_rounds": None,            # no per-slot breakdown in the sync lane
            "model_root": "state-cid-11",   # v1 `state_cid` -> normalized `model_root`
            "done": True,
        }

    def test_v1_pointer_done_defaults_false_when_absent(self):
        d = dm.sd_pointer_decode({"round": 3, "state_cid": "s"})  # no `done` key
        assert d["done"] is False
        assert d["v"] == 1 and d["event"] == 3 and d["slot_rounds"] is None and d["model_root"] == "s"

    # -------------------------------------------------------------- malformed pointers

    def test_decode_non_dict_raises(self):
        with pytest.raises(ValueError):
            dm.sd_pointer_decode(None)
        with pytest.raises(ValueError):
            dm.sd_pointer_decode([1, 2, 3])

    def test_decode_v2_missing_key_names_it(self):
        # v2 (v==2) missing each required key -> ValueError naming that key.
        for missing in ("event", "rounds", "model_root"):
            bad = {"v": 2, "event": 1, "rounds": {}, "model_root": "r"}
            del bad[missing]
            with pytest.raises(ValueError) as ei:
                dm.sd_pointer_decode(bad)
            assert missing in str(ei.value)

    def test_decode_v1_missing_key_names_it(self):
        # v1 (no `v`) missing each required key -> ValueError naming that key.
        with pytest.raises(ValueError) as ei:
            dm.sd_pointer_decode({"state_cid": "s"})   # missing `round`
        assert "round" in str(ei.value)
        with pytest.raises(ValueError) as ei:
            dm.sd_pointer_decode({"round": 5})         # missing `state_cid`
        assert "state_cid" in str(ei.value)
        with pytest.raises(ValueError) as ei:
            dm.sd_pointer_decode({})                    # empty -> missing `round`
        assert "round" in str(ei.value)

    # -------------------------------------------------------------- SlotClock

    def test_slotclock_starts_at_zero(self):
        c = dm.SlotClock()
        assert c.event == 0
        assert c.slot_round("1_0") == 0
        assert c.slot_round(("1", "1")) == 0   # unseen slot -> 0

    def test_slotclock_event_strictly_monotonic_across_slots(self):
        c = dm.SlotClock()
        events = []
        # interleave advances across two slots; the GLOBAL event must strictly increase each time.
        for slot in ["a", "b", "a", "a", "b"]:
            ev, _ = c.advance(slot)
            events.append(ev)
        assert events == [1, 2, 3, 4, 5]
        assert c.event == 5
        assert all(events[i] < events[i + 1] for i in range(len(events) - 1))

    def test_slotclock_per_slot_rounds_independent(self):
        c = dm.SlotClock()
        # advance slot a three times, slot b once -> global event 4, but each slot's round is its own count.
        assert c.advance("a") == (1, 1)
        assert c.advance("a") == (2, 2)
        assert c.advance("b") == (3, 1)   # b's FIRST round even though global event is 3
        assert c.advance("a") == (4, 3)
        assert c.slot_round("a") == 3
        assert c.slot_round("b") == 1
        assert c.event == 4

    def test_slotclock_tuple_slot_ids(self):
        # slots are (layer, expert) tuples in the real lane (5090=slot(1,0), 4060=slot(1,1)).
        c = dm.SlotClock()
        assert c.advance((1, 0)) == (1, 1)
        assert c.advance((1, 1)) == (2, 1)
        assert c.advance((1, 0)) == (3, 2)
        assert c.slot_round((1, 0)) == 2
        assert c.slot_round((1, 1)) == 1

    def test_slotclock_age_and_clamp(self):
        c = dm.SlotClock()
        for _ in range(5):
            c.advance("a")           # event now 5
        assert c.event == 5
        assert c.age(3) == 2         # 5 - 3
        assert c.age(5) == 0         # equal -> 0
        assert c.age(9) == 0         # base ahead of current (race) -> clamp to 0, never negative
        assert c.age(0) == 5

    def test_slotclock_fresh_agrees_with_staleness_ok(self):
        c = dm.SlotClock()
        for _ in range(10):
            c.advance("a")           # event now 10
        base = 4                     # age = 10 - 4 = 6
        # fresh(base, max_stale) MUST equal staleness_ok(base, current_event, max_stale) for every policy.
        for max_stale in (None, 0, 1, 5, 6, 7, 100):
            assert c.fresh(base, max_stale) == dm.staleness_ok(base, c.event, max_stale)
        # boundary semantics: age 6 is fresh at max_stale 6, stale at 5.
        assert c.fresh(base, 6)[0] is True
        assert c.fresh(base, 5)[0] is False
        # unbounded / non-positive policy is always fresh (today's default behavior).
        assert c.fresh(base, None)[0] is True
        assert c.fresh(base, 0)[0] is True
