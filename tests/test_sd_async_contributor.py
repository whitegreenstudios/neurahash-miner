"""Alpha 2.0 non-blocking contributor cadence (#146, W3) -- PURE-logic guards.

These exercise the four decisions the async GLM shardDiLoCo contributor makes, factored out of
tools/sharddiloco_glm_contributor.py so they test with NO socket, NO GPU, NO real model (the async
loop itself, _run_async, is I/O-bound and proven on the real WAN pair per docs/ALPHA2_PLAN.md sec 4):

  1. mode selection is POINTER-DRIVEN ..................... test_mode_*
     (v1 pointer -> sync ALWAYS; v2 -> async; v2 + NEURAHASH_SD_ASYNC=0 -> sync fallback)
  2. non-blocking accepted-record catch-up scan .......... test_scan_*
  3. no-progress abort decision (rc6, not rc7 drift) ..... test_abort_* / test_*progress* / test_*root*
  4. publish-record field extension (base_event/etc.) .... test_contrib_record_*

The tests live inside a class whose name contains "ShardDiLoCo" so `-k "sharddiloco"` picks up this
file (the module name `test_sd_async_contributor` alone does not contain the substring -- same
convention as tests/test_sd_async_helpers.py / test_sd_async_coordinator.py).

Run: C:/Python313/python.exe -m pytest tests/test_sd_async_contributor.py -q
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neurahash.diloco_merge as dm                              # noqa: E402  (numpy-only, no torch)
import sharddiloco_glm_contributor as N                          # noqa: E402  (torch-free to import)


def _v2(event=5, model_root="root-a", rounds=None):
    """A v2 pointer as the coordinator publishes it (dm.sd_pointer_encode, W1)."""
    return dm.sd_pointer_encode(event=event, slot_rounds=rounds or {"1_0": 3, "1_1": 2},
                                model_root=model_root)


def _v1(rnd=5, state_cid="root-a", done=False):
    """A v1 pointer as an alpha-1.0 coordinator publishes it (harness publish_pointer shape)."""
    return {"round": rnd, "state_cid": state_cid, "done": done}


def _manifest(events):
    """A fake lane manifest: {accepted_name(e): entry} for each event e."""
    return {N.accepted_name(e): {"sha256": "sha-%d" % e} for e in events}


class TestShardDiLoCoAsyncContributor:
    # ------------------------------------------------------------ 1. mode selection (pointer-driven)
    def test_mode_v1_pointer_is_sync(self):
        # v1 lane -> sync, so a fresh public clone joins today's lanes byte-identically.
        assert N._select_async_mode(_v1(), env={}) is False

    def test_mode_v1_stays_sync_even_with_async_env_on(self):
        # POINTER version decides, never the env: NEURAHASH_SD_ASYNC=1 cannot force async onto v1.
        assert N._select_async_mode(_v1(), env={"NEURAHASH_SD_ASYNC": "1"}) is False

    def test_mode_v2_pointer_is_async_by_default(self):
        assert N._select_async_mode(_v2(), env={}) is True

    def test_mode_v2_explicit_optout_is_sync_fallback(self):
        for val in ("0", "false", "no", "off", "n", "OFF", "False", " No "):
            assert N._select_async_mode(_v2(), env={"NEURAHASH_SD_ASYNC": val}) is False, val

    def test_mode_v2_truthy_or_unset_stays_async(self):
        for val in ("1", "true", "yes", "on", ""):     # "" == unset -> still async on a v2 lane
            assert N._select_async_mode(_v2(), env={"NEURAHASH_SD_ASYNC": val}) is True, val

    # ------------------------------------------------------------- 2. accepted-record catch-up scan
    def test_scan_contiguous_from_zero(self):
        assert N.scan_accepted_events(_manifest([1, 2, 3]), last_applied=0) == [1, 2, 3]

    def test_scan_stops_at_first_gap(self):
        # 3 is missing -> the scan MUST stop at 2 and never fold 4/5 out of order.
        assert N.scan_accepted_events(_manifest([1, 2, 4, 5]), last_applied=0) == [1, 2]

    def test_scan_from_midpoint(self):
        assert N.scan_accepted_events(_manifest([3, 4, 5]), last_applied=2) == [3, 4, 5]

    def test_scan_none_visible_returns_empty(self):
        # last_applied already at 3, next event 4 not present -> nothing to apply, caller trains anyway.
        assert N.scan_accepted_events(_manifest([1, 2, 3]), last_applied=3) == []

    def test_scan_accepts_a_bare_name_set(self):
        names = {N.accepted_name(e) for e in (1, 2)}
        assert N.scan_accepted_events(names, last_applied=0) == [1, 2]

    def test_scan_is_bounded_by_max_scan(self):
        # a pathological long contiguous run is capped so a malformed manifest cannot loop forever.
        assert N.scan_accepted_events(_manifest(range(1, 50)), last_applied=0, max_scan=5) == \
            [1, 2, 3, 4, 5]

    # ------------------------------------------------------------------- 3. no-progress abort (rc6)
    def test_abort_on_mismatch_no_progress_and_elapsed(self):
        assert N.async_should_abort_no_progress("root-a", "root-b", False, 300.0, 300.0) is True

    def test_progress_resets_timer_no_abort(self):
        # any applied record this tick -> never abort, even with a mismatch and huge elapsed.
        assert N.async_should_abort_no_progress("root-a", "root-b", True, 99999.0, 300.0) is False

    def test_matching_roots_never_abort(self):
        # fully caught up -> not stuck (root mismatch is the ONLY thing that can trigger an abort).
        assert N.async_should_abort_no_progress("root-a", "root-a", False, 99999.0, 300.0) is False

    def test_no_abort_before_round_wait_elapses(self):
        assert N.async_should_abort_no_progress("root-a", "root-b", False, 299.9, 300.0) is False

    def test_empty_pointer_root_never_aborts(self):
        for empty in (None, "", 0):
            assert N.async_should_abort_no_progress("root-a", empty, False, 99999.0, 300.0) is False

    def test_mismatch_alone_is_not_an_abort(self):
        # The whole point of async: a root mismatch with progress still fresh is NORMAL, not rc7 drift.
        assert N.async_should_abort_no_progress("root-a", "root-b", False, 10.0, 300.0) is False

    # ------------------------------------------------------------- 4. publish-record field extension
    def test_contrib_record_carries_async_fields(self):
        rec = N.build_async_contrib_record(
            miner="miner0", i=0, L=1, E=0, base_event=7, base_root="root-xyz", expert_cid="cid-abc",
            sig="deadbeef", train_flops=1.2e9, delta_bytes=278731, steps=60, tokens=3840)
        for k in ("base_event", "base_root", "steps", "tokens"):
            assert k in rec, k
        assert rec["base_event"] == 7
        assert rec["base_root"] == "root-xyz"
        assert rec["steps"] == 60
        assert rec["tokens"] == 3840

    def test_contrib_record_is_superset_of_sync_record(self):
        # base_round is kept and aliased to base_event so a v1-shaped reader still finds a base height,
        # and every field the sync record carries is present (strict superset, like the v2 pointer).
        rec = N.build_async_contrib_record(
            miner="m", i=1, L=1, E=1, base_event=42, base_root="r", expert_cid="c", sig="s",
            train_flops=0.0, delta_bytes=10, steps=0, tokens=0)
        assert rec["base_round"] == 42 == rec["base_event"]
        for k in ("miner", "expert", "layer", "glm_expert", "expert_cid", "trunk_cid", "sig",
                  "train_flops", "trunk_bytes", "delta_bytes"):
            assert k in rec, k
        assert rec["expert"] == 1 and rec["layer"] == 1 and rec["glm_expert"] == 1
        assert rec["trunk_cid"] is None and rec["trunk_bytes"] == 0

    # ------------------------------------------------- 5. F-Q1: unique publish name per H-block (no drop)
    def test_async_publish_names_unique_per_publish_same_base_event(self):
        # THE F-Q1 fix: two H-blocks trained against the SAME base_event (coordinator merge lagging) must
        # publish to DISTINCT manifest names -- otherwise the 2nd atomically repoints (and silently drops)
        # the 1st. The per-publish counter suffix makes each name unique while base_event stays constant.
        n0 = N.async_publish_name(5, "miner0", 0)
        n1 = N.async_publish_name(5, "miner0", 1)
        assert n0 != n1 and n0 == "cg/r5/miner0.0" and n1 == "cg/r5/miner0.1"

    def test_two_publishes_both_survive_a_manifest_scan(self):
        # end-to-end: put both names in a fake manifest and let the coordinator's discovery scan it. BOTH
        # must be found (neither lost), each carrying base_event 5 -- i.e. no silent single-H-block drop.
        # importorskip: the coordinator is PRIVATE-ONLY -- this file also ships in the public miner
        # repo (byte-identical), where the cross-check auto-skips instead of erroring.
        C = pytest.importorskip("sharddiloco_glm_coordinator")
        n0, n1 = N.async_publish_name(5, "miner0", 0), N.async_publish_name(5, "miner0", 1)
        man = {n0: {"sha256": "a"}, n1: {"sha256": "b"}, N.accepted_name(4): {"sha256": "acc"}}
        got = C._collect_unprocessed(list(man.keys()), set(), C._parse_contrib_name)
        assert sorted(nm for nm, _e, _m in got) == sorted([n0, n1])   # BOTH survive, accepted-name excluded
        assert all(e == 5 for _nm, e, _m in got)                       # same authoritative base_event
