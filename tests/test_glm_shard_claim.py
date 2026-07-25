"""Shard Claim -- unlimited CLAIMABLE expert coordinates with a bounded ACTIVE set.

Design: docs/SHARD_CLAIM_DESIGN.md. Owner directive (memory public-testing-unlimited-slots-directive):
public testing, unlimited slots, anyone may join -- so a miner must be able to claim a GLM
(layer, expert) coordinate the coordinator has never seen, finish it, and move to the next.

This module guards the STEP 1 half of that: the slot registry inside GlmExpertLaneHost, plus the
claimability filter that decides which coordinates a node may legitimately be asked to host. Two
properties here are load-bearing and neither is obvious:

  1. INDEX STABILITY. The coordinator persists the merged slot as a bare int in every accepted
     record (sharddiloco_glm_coordinator.py:559) and a contributor replays weights back by that int
     (apply_accepted -> write_slot, sharddiloco_glm_contributor.py:570). So eviction must never
     compact or reuse an index, or every historical accepted record silently re-points at a
     DIFFERENT expert. test_evict_never_shifts_indices is the guard.
  2. CLAIMABILITY. piece_loader allocates a resident layer's fused params FULL WIDTH (all 64 expert
     rows) and fills only the resident ones, so a NON-resident row of a resident layer is writable
     and silently inert -- zero weights, router pinned to -inf (piece_loader.py:366-385). Measured
     2026-07-25. A miner claiming one trains forever and is gate-rejected forever with nothing in
     any log to explain it, so the claim has to be refused up front.

Run: C:/Python313/python.exe -m pytest tests/test_glm_shard_claim.py -q
"""
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import piece_loader as PL                                             # noqa: E402
import sharddiloco_glm_contributor as N                               # noqa: E402


# ================================================================== claimable_expert_ids (no torch)
class TestClaimableExpertIds:
    """The filter that separates real trainable coordinates from manifest artefacts. Pure dict/int
    work -- no model, no torch, so it runs in milliseconds."""

    @staticmethod
    def _manifest(pieces):
        return {"pieces": [{"piece": "experts_%d" % pid, "experts": exps}
                           for pid, exps in pieces.items()]}

    @staticmethod
    def _cfg(n_layers=47, dense=1):
        return types.SimpleNamespace(num_hidden_layers=n_layers, first_k_dense_replace=dense)

    def test_drops_the_mtp_layer_that_the_model_never_instantiates(self):
        """Layer == num_hidden_layers is the MTP/nextn layer: present in the shard manifest, never
        built by Glm4MoeLiteForCausalLM. Passing it through raised a naked IndexError deep inside
        read_slot (measured: 'index 47 is out of range')."""
        man = self._manifest({0: [[46, 3], [47, 0], [47, 1]]})
        assert PL.assigned_expert_ids(man, [0]) == {(46, 3), (47, 0), (47, 1)}   # unfiltered: unsafe
        assert PL.claimable_expert_ids(man, [0], self._cfg()) == [(46, 3)]       # filtered: safe

    def test_drops_the_dense_layer_which_has_no_routed_experts(self):
        man = self._manifest({0: [[0, 0], [1, 0]]})
        assert PL.claimable_expert_ids(man, [0], self._cfg(dense=1)) == [(1, 0)]

    def test_returns_a_sorted_list_not_a_set(self):
        """Callers build a POSITIONAL slot list from this; a set's iteration order is not stable, so
        two nodes could derive different slot orders from the same manifest."""
        man = self._manifest({0: [[3, 7], [1, 2], [3, 1], [1, 0]]})
        got = PL.claimable_expert_ids(man, [0], self._cfg())
        assert got == [(1, 0), (1, 2), (3, 1), (3, 7)]
        assert isinstance(got, list)

    def test_an_all_mtp_piece_yields_an_empty_set_so_callers_can_fail_loudly(self):
        """Measured on the live manifest: pieces 589-601 are 100% MTP. A node given one of those has
        no real experts at all, and used to boot completely clean and then train nothing."""
        man = self._manifest({589: [[47, 0], [47, 1], [47, 2]]})
        assert PL.claimable_expert_ids(man, [589], self._cfg()) == []


# ============================================================ the slot registry on a real tiny GLM
@pytest.fixture(scope="module")
def host():
    """One tiny REAL GLM (build_tiny_glm is a few seconds; per-test rebuilds are not affordable).
    TINY is layers=3 (so instantiated layer indices are 0,1,2) with n_experts=4.

    Claimable set is declared explicitly as layer 1 and 2 x experts 0..3, which is what a node
    holding those pieces would report. (1,0) and (1,1) start registered, mirroring today's live
    --slots 1:0,1:1; everything else must be reachable only by claiming it."""
    import torch
    torch.set_num_threads(2)
    G = N._G()
    T = N.TINY
    model, cfg = G.build_tiny_glm(seed=T["seed"], vocab=T["vocab"], hidden=T["hidden"],
                                  inter=T["inter"], moe_inter=T["moe_inter"], layers=T["layers"],
                                  n_experts=T["n_experts"], topk=T["topk"])
    claimable = [(L, E) for L in (1, 2) for E in range(T["n_experts"])]
    return G, model, cfg, claimable


@pytest.fixture
def h(host):
    """A fresh host per test over the shared model (registry state must not leak between tests)."""
    G, model, cfg, claimable = host
    return G.GlmExpertLaneHost(model, cfg, [(1, 0), (1, 1)], claimable=claimable)


class TestSlotRegistry:

    def test_startup_slots_are_registered_and_active(self, h):
        assert h.slots == [(1, 0), (1, 1)]
        assert h.index_of(1, 0) == 0 and h.index_of(1, 1) == 1
        assert h.active == {0, 1}
        assert h.index_of(1, 2) is None            # never declared -> not yet addressable

    def test_register_an_unseen_coordinate_appends_and_activates_it(self, h):
        """THE headline acceptance property: a coordinate the host has never seen becomes live
        without a restart."""
        idx = h.register(1, 2)
        assert idx == 2
        assert h.slots[2] == (1, 2)
        assert h.index_of(1, 2) == 2
        assert h.is_active(2)

    def test_register_is_idempotent(self, h):
        assert h.register(1, 0) == 0
        assert h.register(1, 0) == 0
        assert h.slots == [(1, 0), (1, 1)]         # no duplicate row appended

    def test_register_spans_layers_within_the_claimable_set(self, h):
        assert h.register(2, 3) == 2
        assert h.slots[2] == (2, 3)

    def test_evict_never_shifts_indices(self, h):
        """The corruption guard. Evicting a middle slot must not renumber anything: an accepted
        record persisted with slot=2 has to keep meaning the same expert forever."""
        h.register(1, 2)                                   # idx 2
        h.register(1, 3)                                   # idx 3
        before = list(h.slots)
        assert h.evict(2) is True
        assert h.slots == before                           # list itself untouched
        assert h.index_of(1, 2) == 2                       # mapping survives eviction
        assert h.index_of(1, 3) == 3                       # later slot did NOT slide down
        assert h.active == {0, 1, 3}
        assert h.register(2, 0) == 4                       # a new claim gets a FRESH index, not 2

    def test_evict_then_reclaim_resumes_on_the_same_index(self, h):
        idx = h.register(1, 2)
        assert h.evict(idx) is True
        assert h.is_active(idx) is False
        assert h.register(1, 2) == idx                      # same coordinate -> same index
        assert h.is_active(idx) is True

    def test_evicting_an_inactive_slot_is_a_no_op(self, h):
        assert h.evict(1) is True
        assert h.evict(1) is False

    def test_register_refuses_a_coordinate_this_node_cannot_host(self, h):
        """The inert-slot trap: writable, never routed, gate-rejected forever, silent. Refuse it."""
        with pytest.raises(ValueError, match=r"not hostable"):
            h.register(1, 99)
        with pytest.raises(ValueError, match=r"not hostable"):
            h.register(2, 99)
        assert h.index_of(1, 99) is None                    # and nothing was appended
        assert h.slots == [(1, 0), (1, 1)]

    def test_claimable_is_unchecked_when_not_supplied(self, host):
        """Tiny-model tests and single-piece deployments that pass no claimable set keep working."""
        G, model, cfg, _ = host
        free = G.GlmExpertLaneHost(model, cfg, [(1, 0)])
        assert free.is_claimable(1, 3) is True
        assert free.claimable_coords() is None
        assert free.register(1, 3) == 1


class TestMaxActiveCap:
    """--max-active-slots: a flash crowd must not be able to OOM the coordinator, and a refused
    claim must be an explicit signal, never a silent drop (a silently dropped claim looks exactly
    like a rejected delta from the miner's side)."""

    def test_cap_refuses_a_new_coordinate_loudly(self, host):
        G, model, cfg, claimable = host
        c = G.GlmExpertLaneHost(model, cfg, [(1, 0), (1, 1)], claimable=claimable, max_active=3)
        assert c.register(1, 2) == 2                         # 3rd active: fits
        with pytest.raises(RuntimeError, match=r"max_active_slots=3"):
            c.register(1, 3)                                 # 4th: refused, not dropped
        assert c.index_of(1, 3) is None

    def test_evicting_frees_a_seat(self, host):
        G, model, cfg, claimable = host
        c = G.GlmExpertLaneHost(model, cfg, [(1, 0), (1, 1)], claimable=claimable, max_active=2)
        with pytest.raises(RuntimeError):
            c.register(1, 2)
        c.evict(0)
        assert c.register(1, 2) == 2                         # seat freed -> claim admitted

    def test_re_admitting_an_evicted_slot_also_respects_the_cap(self, host):
        G, model, cfg, claimable = host
        c = G.GlmExpertLaneHost(model, cfg, [(1, 0), (1, 1)], claimable=claimable, max_active=2)
        c.evict(0)
        c.register(1, 2)                                     # takes the free seat -> active {1,2}
        with pytest.raises(RuntimeError, match=r"re-admit"):
            c.register(1, 0)                                 # known coordinate, but no seat left


class TestCanonicalExpertsWithHoles:
    """canonical_experts() is what makes the per-event cost O(active) rather than O(ever-seen).
    Evicted slots come back as None so the list stays POSITIONAL -- experts[e] must keep meaning
    slot e, because that is how the merge addresses them (neurahash/diloco_merge.py:931-941)."""

    def test_evicted_slots_are_none_and_active_slots_are_real(self, h):
        h.register(1, 2)
        h.evict(1)
        experts = h.canonical_experts()
        assert len(experts) == 3                             # positional over ALL registered slots
        assert experts[1] is None                            # the hole
        for i in (0, 2):
            assert set(experts[i]) == {"gate", "up", "down"}

    def test_begin_round_and_sync_tolerate_holes(self, h):
        """A round over a host with an evicted slot must not raise, and must not write the hole."""
        h.register(1, 2)
        h.evict(1)
        experts = h.canonical_experts()
        h.begin_round(experts)                               # used to crash on None.items()
        h.sync_from_canonical(experts)                        # must skip the hole, not write it
        assert h._base_slots[1] is None

    def test_sync_round_trips_active_slot_weights_bit_exactly(self, h):
        """Determinism is a product constraint here (memory safetensors-mmap-recompute-determinism),
        so a read->write cycle must not perturb a single bit."""
        import numpy as np
        h.register(1, 2)
        experts = h.canonical_experts()
        before = {k: v.copy() for k, v in experts[2].items()}
        h.sync_from_canonical(experts)
        after = h.read_slot(2)
        for k in before:
            assert np.array_equal(before[k], after[k]), k

    def test_a_written_delta_is_visible_on_a_newly_claimed_slot(self, h):
        """Registration is worthless if the freshly claimed slot is not actually writable."""
        import numpy as np
        idx = h.register(2, 1)
        d = h.read_slot(idx)
        d["gate"] = d["gate"] + np.float32(0.25)
        h.write_slot(idx, d)
        assert np.allclose(h.read_slot(idx)["gate"], d["gate"], atol=1e-6)


class TestSlotRootIsPerCoordinate:
    """N.slot_root is what makes registration/eviction possible: lineage must be judged per
    coordinate, because the global model_root is a function of the whole slot list."""

    def test_slot_root_differs_between_coordinates(self, h):
        assert N.slot_root(h, 0) != N.slot_root(h, 1)

    def test_slot_root_is_stable_and_tracks_its_own_weights(self, h):
        import numpy as np
        before = N.slot_root(h, 0)
        assert N.slot_root(h, 0) == before                 # pure function of the weights
        d = h.read_slot(0)
        d["gate"] = d["gate"] + np.float32(0.5)
        h.write_slot(0, d)
        assert N.slot_root(h, 0) != before                 # its own move is visible

    def test_moving_one_coordinate_does_not_change_anothers_root(self, h):
        """THE property the whole design rests on. Under the global model_root this is false, which
        is why admitting a new coordinate used to drop every in-flight miner as wrong-lineage-root."""
        import numpy as np
        other_before = N.slot_root(h, 1)
        global_before = N.model_root(h)
        d = h.read_slot(0)
        d["down"] = d["down"] + np.float32(0.25)
        h.write_slot(0, d)
        assert N.slot_root(h, 1) == other_before           # untouched coordinate: root holds
        assert N.model_root(h) != global_before            # ... but the GLOBAL root moved

    def test_registering_a_new_coordinate_moves_the_global_root_but_no_slot_root(self, h):
        """The regression this replaces. Registering (1,2) changes model_root -- so a miner training
        (1,0) against a perfectly valid base would have been rejected. Its slot_root is untouched."""
        g_before, s0_before = N.model_root(h), N.slot_root(h, 0)
        h.register(1, 2)
        assert N.model_root(h) != g_before                 # global root: broken by registration
        assert N.slot_root(h, 0) == s0_before              # per-coordinate root: unaffected

    def test_model_root_is_unchanged_by_the_refactor(self, h):
        """model_root was refactored to share _slot_digest_into with slot_root. Its output must stay
        byte-identical or every pointer the LIVE campaign has published stops validating. Recompute
        it here from the documented algorithm rather than trusting the refactor."""
        import hashlib
        import numpy as np
        want = hashlib.sha256()
        for i in range(len(h.slots)):
            d = h.read_slot(i)
            L, E = h.slots[i]
            want.update(("L%dE%d|" % (L, E)).encode())
            for k in sorted(d):
                want.update(k.encode())
                want.update(np.ascontiguousarray(d[k], dtype=np.float32).tobytes())
        assert N.model_root(h) == want.hexdigest()


_COORD = os.path.join(_TOOLS, "sharddiloco_glm_coordinator.py")
coordinator_only = pytest.mark.skipif(
    not os.path.exists(_COORD),
    reason="tools/sharddiloco_glm_coordinator.py is not in this checkout (training-coordinator role)")


@coordinator_only
class TestLineageResolution:
    """_expected_slot_root + _lineage_ok. Pure dict/string logic -- no model needed."""

    @staticmethod
    def _C():
        import sharddiloco_glm_coordinator as C
        return C

    def test_resolves_the_newest_root_at_or_before_the_base_event(self):
        C = self._C()
        hist = {}
        C._record_slot_root(hist, (1, 0), 0, "r0")
        C._record_slot_root(hist, (1, 0), 5, "r5")
        C._record_slot_root(hist, (1, 0), 9, "r9")
        assert C._expected_slot_root(hist, (1, 0), 0) == "r0"
        assert C._expected_slot_root(hist, (1, 0), 4) == "r0"
        assert C._expected_slot_root(hist, (1, 0), 5) == "r5"
        assert C._expected_slot_root(hist, (1, 0), 100) == "r9"

    def test_unknown_coordinate_resolves_to_none(self):
        C = self._C()
        assert C._expected_slot_root({}, (1, 0), 3) is None

    def test_a_coordinate_seeded_after_the_base_event_falls_back_to_its_seed(self):
        """A miner reads the pointer at event 7, then claims a coordinate the coordinator seeds at
        event 12. Nothing could have merged that coordinate before it was registered, so the seed IS
        the frozen-base root the miner hashed."""
        C = self._C()
        hist = {}
        C._record_slot_root(hist, (2, 3), 12, "seed")
        assert C._expected_slot_root(hist, (2, 3), 7) == "seed"

    def test_lineage_prefers_the_slot_root_when_present(self):
        C = self._C()
        rh = {0: "GLOBAL"}
        ok, why = C._lineage_ok(0, "STALE-GLOBAL", 0, rh, base_slot_root="S", want_slot_root="S")
        assert (ok, why) == (True, "ok")        # global mismatch is IRRELEVANT once we have a slot root

    def test_lineage_rejects_a_mismatched_slot_root(self):
        C = self._C()
        ok, why = C._lineage_ok(0, "G", 0, {0: "G"}, base_slot_root="MINE", want_slot_root="OURS")
        assert ok is False and why == "wrong-lineage-slot-root"

    def test_lineage_rejects_a_coordinate_we_do_not_know(self):
        C = self._C()
        ok, why = C._lineage_ok(0, "G", 0, {0: "G"}, base_slot_root="MINE", want_slot_root=None)
        assert ok is False and why == "unknown-coordinate"

    def test_old_miners_still_judged_on_the_global_root(self):
        """v3.3.2 sends no base_slot_root. It must keep working exactly as before."""
        C = self._C()
        assert C._lineage_ok(0, "G", 0, {0: "G"}) == (True, "ok")
        ok, why = C._lineage_ok(0, "WRONG", 0, {0: "G"})
        assert ok is False and why == "wrong-lineage-root"

    def test_event_bounds_still_apply_to_slot_root_records(self):
        """A per-coordinate root must not become a way to smuggle in a forged base height."""
        C = self._C()
        ok, why = C._lineage_ok(9, "G", 3, {0: "G"}, base_slot_root="S", want_slot_root="S")
        assert ok is False and why == "future-base-event"
        ok, why = C._lineage_ok(2, "G", 3, {0: "G"}, base_slot_root="S", want_slot_root="S")
        assert ok is False and why == "unknown-event"
        ok, why = C._lineage_ok("nope", "G", 3, {0: "G"}, base_slot_root="S", want_slot_root="S")
        assert ok is False and why == "bad-base-event"


@coordinator_only
class TestRecordCarriesTheSlotRoot:

    def test_async_record_includes_base_slot_root_and_stays_a_superset(self):
        rec = N.build_async_contrib_record("m", 3, 1, 2, 7, "GROOT", "cid", "sig", 1.0, 10, 5, 99,
                                           base_slot_root="SROOT")
        assert rec["base_slot_root"] == "SROOT"
        assert rec["base_root"] == "GROOT"          # still sent, for pre-Shard-Claim coordinators
        assert (rec["layer"], rec["glm_expert"], rec["expert"]) == (1, 2, 3)

    def test_omitting_the_slot_root_is_byte_identical_to_before(self):
        a = N.build_async_contrib_record("m", 0, 1, 0, 1, "G", "c", "s", 1.0, 2, 3, 4)
        assert "base_slot_root" not in a


class TestResolveClaim:
    """--expert L:E is the shard-claim address; --slot stays as a deprecated alias so v3.3.2 miners
    and every existing launch script keep working."""

    @staticmethod
    def _args(expert=None, slot=None, mode="tiny"):
        return types.SimpleNamespace(expert=expert, slot=slot, mode=mode, slots="1:0,1:1",
                                     domains="code,gutenberg", piece=0,
                                     shard_dir=None, config_dir=None)

    def test_expert_flag_claims_a_coordinate_outside_the_startup_list(self):
        """The point of the feature: a coordinate that is NOT in --slots becomes workable, and it is
        appended to the local list so the lane host can actually read and write it."""
        slots = N.parse_slots("1:0,1:1")
        L, E, i, src = N.resolve_claim(self._args(expert="2:7"), slots, log=lambda *a: None)
        assert (L, E) == (2, 7)
        assert i == 2 and slots[2] == (2, 7)
        assert src == "--expert"

    def test_expert_flag_reuses_the_index_of_an_already_listed_coordinate(self):
        slots = N.parse_slots("1:0,1:1")
        L, E, i, _ = N.resolve_claim(self._args(expert="1:1"), slots, log=lambda *a: None)
        assert (L, E, i) == (1, 1, 1)
        assert len(slots) == 2                             # nothing appended

    def test_slot_alias_still_resolves_positionally(self):
        slots = N.parse_slots("1:0,1:1")
        L, E, i, src = N.resolve_claim(self._args(slot=1), slots, log=lambda *a: None)
        assert (L, E, i) == (1, 1, 1)
        assert "deprecated" in src

    def test_expert_wins_over_slot_and_says_so(self):
        slots = N.parse_slots("1:0,1:1")
        said = []
        L, E, _, _ = N.resolve_claim(self._args(expert="1:0", slot=1), slots, log=said.append)
        assert (L, E) == (1, 0)
        assert any("--slot 1 ignored" in s for s in said)

    def test_out_of_range_slot_still_fails_loudly(self):
        slots = N.parse_slots("1:0,1:1")
        with pytest.raises(SystemExit, match=r"--slot 5 out of range"):
            N.resolve_claim(self._args(slot=5), slots, log=lambda *a: None)

    def test_malformed_expert_is_rejected(self):
        for bad in ("7", "a:b", "1:"):
            with pytest.raises(SystemExit, match=r"--expert must look like L:E"):
                N.parse_coord(bad)

    def test_claim_is_refused_when_the_piece_does_not_hold_the_coordinate(self, monkeypatch):
        """The inert-slot guard, at the CLI. Without this the miner trains a router-masked expert and
        is rejected forever with no explanation."""
        monkeypatch.setattr(N, "node_claimable_coords", lambda a: [(1, 0), (1, 1), (1, 2)])
        slots = N.parse_slots("1:0,1:1")
        with pytest.raises(SystemExit, match=r"REFUSING to claim \(L9,E9\)"):
            N.resolve_claim(self._args(expert="9:9", mode="glm"), slots, log=lambda *a: None)

    def test_an_all_mtp_piece_is_refused_with_its_own_message(self, monkeypatch):
        monkeypatch.setattr(N, "node_claimable_coords", lambda a: [])
        slots = N.parse_slots("1:0,1:1")
        with pytest.raises(SystemExit, match=r"holds NO real experts"):
            N.resolve_claim(self._args(expert="1:0", mode="glm"), slots, log=lambda *a: None)


class TestCoordDataSlot:
    """The domain the miner trains on and the domain the coordinator gates on must agree."""

    def test_reproduces_todays_live_mapping(self):
        """Live campaign is --slots 1:0,1:1 -> indices 0,1 -> domains code,gutenberg. Coordinate
        addressing must land on the SAME files or the frozen probe/heldout change meaning."""
        assert N.coord_data_slot(1, 0) == 0
        assert N.coord_data_slot(1, 1) == 1

    def test_is_independent_of_the_layer(self):
        """Two miners on the same expert index in different layers share a domain; what matters is that
        BOTH sides derive it from the coordinate, never from a registry index."""
        assert N.coord_data_slot(1, 3) == N.coord_data_slot(42, 3)


@coordinator_only
class TestAdmitCoordinate:
    """_admit_coordinate is the hinge: it turns a wire coordinate into a live slot, registering one the
    coordinator has never serviced. This is acceptance criterion (a)."""

    @staticmethod
    def _setup(host_slots=((1, 0), (1, 1)), max_active=None, claimable=None):
        import sharddiloco_glm_coordinator as C
        import neurahash.diloco_merge as dm
        G = N._G()
        T = N.TINY
        model, cfg = G.build_tiny_glm(seed=T["seed"], vocab=T["vocab"], hidden=T["hidden"],
                                      inter=T["inter"], moe_inter=T["moe_inter"], layers=T["layers"],
                                      n_experts=T["n_experts"], topk=T["topk"])
        if claimable is None:
            claimable = [(L, E) for L in (1, 2) for E in range(T["n_experts"])]
        slots = [tuple(s) for s in host_slots]
        host = G.GlmExpertLaneHost(model, cfg, slots, claimable=claimable, max_active=max_active)
        args = types.SimpleNamespace(mode="tiny", domains="code,gutenberg", probe_size=8)
        pools = {i: C._slot_probe_pool(args, slots[i]) for i in range(len(slots))}
        probe = dm.SecretRotatedProbe(pools, seed=7, size=8)
        clock = dm.SlotClock()
        srh = {}
        for i, le in enumerate(slots):
            C._record_slot_root(srh, le, 0, N.slot_root(host, i))
        return C, host, slots, probe, clock, srh, args

    @staticmethod
    def _rec(L, E, idx=0):
        return {"layer": L, "glm_expert": E, "expert": idx, "miner": "m1"}

    def test_an_unseen_coordinate_is_registered_and_gets_a_live_slot(self):
        C, host, slots, probe, clock, srh, args = self._setup()
        logs = []
        e = C._admit_coordinate(host, slots, self._rec(2, 3), "m1", args, probe, clock, srh, logs.append)
        assert e == 2                                       # a brand-new slot index
        assert host.slots[2] == (2, 3) and host.is_active(2)
        assert slots[2] == (2, 3)                           # coordinator's own list kept in step
        assert probe.has_pool(2)                            # gate pool materialized
        assert C._expected_slot_root(srh, (2, 3), 0) is not None   # lineage seeded
        assert any("REGISTER (L2,E3) -> slot 2" in s for s in logs)

    def test_registration_is_idempotent_across_contributions(self):
        C, host, slots, probe, clock, srh, args = self._setup()
        a = C._admit_coordinate(host, slots, self._rec(2, 3), "m1", args, probe, clock, srh, lambda *_: None)
        b = C._admit_coordinate(host, slots, self._rec(2, 3), "m2", args, probe, clock, srh, lambda *_: None)
        assert a == b == 2
        assert len(host.slots) == 3                         # not appended twice

    def test_a_known_coordinate_resolves_without_registering(self):
        C, host, slots, probe, clock, srh, args = self._setup()
        e = C._admit_coordinate(host, slots, self._rec(1, 1), "m1", args, probe, clock, srh, lambda *_: None)
        assert e == 1 and len(host.slots) == 2

    def test_the_wire_index_is_ignored_when_a_coordinate_is_present(self):
        """A miner's local index need not match ours -- that decoupling IS coordinate addressing. A
        record claiming (1,1) with a bogus index 99 must still land on OUR slot for (1,1)."""
        C, host, slots, probe, clock, srh, args = self._setup()
        e = C._admit_coordinate(host, slots, self._rec(1, 1, idx=99), "m1", args, probe, clock, srh,
                                lambda *_: None)
        assert e == 1

    def test_an_unhostable_coordinate_is_dropped_not_registered(self):
        C, host, slots, probe, clock, srh, args = self._setup()
        logs = []
        e = C._admit_coordinate(host, slots, self._rec(1, 99), "m1", args, probe, clock, srh, logs.append)
        assert e is None
        assert host.index_of(1, 99) is None
        assert any("not hostable" in s for s in logs)

    def test_over_capacity_defers_for_retry_instead_of_dropping(self):
        """A silently dropped claim is indistinguishable, miner-side, from a rejected delta."""
        C, host, slots, probe, clock, srh, args = self._setup(max_active=2)
        obj, logs = self._rec(2, 3), []
        e = C._admit_coordinate(host, slots, obj, "m1", args, probe, clock, srh, logs.append)
        assert e is None
        assert obj.get("_retry") is True                    # caller un-sees the record and retries
        assert any("DEFER" in s for s in logs)

    def test_a_freed_seat_admits_the_deferred_claim(self):
        C, host, slots, probe, clock, srh, args = self._setup(max_active=2)
        assert C._admit_coordinate(host, slots, self._rec(2, 3), "m", args, probe, clock, srh,
                                   lambda *_: None) is None
        host.evict(0)
        assert C._admit_coordinate(host, slots, self._rec(2, 3), "m", args, probe, clock, srh,
                                   lambda *_: None) == 2

    def test_a_v332_record_without_a_coordinate_still_resolves_positionally(self):
        C, host, slots, probe, clock, srh, args = self._setup()
        e = C._admit_coordinate(host, slots, {"expert": 1, "miner": "old"}, "old", args, probe, clock,
                                srh, lambda *_: None)
        assert e == 1

    def test_a_v332_record_naming_an_evicted_slot_is_dropped(self):
        C, host, slots, probe, clock, srh, args = self._setup()
        host.evict(1)
        e = C._admit_coordinate(host, slots, {"expert": 1, "miner": "old"}, "old", args, probe, clock,
                                srh, lambda *_: None)
        assert e is None

    def test_registered_slots_gate_pool_matches_the_coordinates_domain(self):
        """If the probe pool came from the registry index instead of the coordinate, the delta would be
        gated against a domain the miner never trained on -- a systematic silent reject."""
        C, host, slots, probe, clock, srh, args = self._setup()
        import numpy as np
        e = C._admit_coordinate(host, slots, self._rec(2, 1), "m1", args, probe, clock, srh, lambda *_: None)
        want = C._slot_probe_pool(args, (2, 1))[0]
        got = probe.ensure_pool(e, None)[0]                 # idempotent: returns the stored pool
        assert np.array_equal(got, want)


class TestGlobalRootComparable:
    """N.global_root_comparable is the gate that stopped shard claim from killing every miner at
    startup. The contributor compared the coordinator's GLOBAL model_root against its own; once the
    coordinator can REGISTER a coordinate we do not hold, that digest is unreachable BY CONSTRUCTION,
    so a perfectly healthy miner logged 'base MISMATCH', burned a full replay, and rolled back. The
    coordinator's slot set is readable straight off the v2 pointer: the `rounds` map is keyed 'L_E'
    per active slot (sharddiloco_glm_coordinator._slot_key)."""

    @staticmethod
    def _host(slots):
        return types.SimpleNamespace(slots=[tuple(s) for s in slots])

    def test_true_when_the_slot_sets_match(self):
        h = self._host([(1, 0), (1, 1)])
        assert N.global_root_comparable(h, {"slot_rounds": {"1_0": 4, "1_1": 2}}) is True
        assert N.global_root_comparable(h, {"rounds": {"1_1": 0, "1_0": 0}}) is True   # raw v2 pointer

    def test_false_when_the_coordinator_holds_a_coordinate_we_do_not(self):
        """The live shard-claim case: someone claimed (2,3), the coordinator registered it, our root
        can never equal theirs again."""
        h = self._host([(1, 0), (1, 1)])
        assert N.global_root_comparable(h, {"slot_rounds": {"1_0": 4, "1_1": 2, "2_3": 1}}) is False

    def test_false_when_we_hold_a_coordinate_the_coordinator_does_not(self):
        h = self._host([(1, 0), (1, 1), (2, 3)])
        assert N.global_root_comparable(h, {"slot_rounds": {"1_0": 4, "1_1": 2}}) is False

    def test_true_for_a_pre_v2_pointer_so_behaviour_is_unchanged(self):
        """A v1 pointer carries no per-slot map. The old comparison must still happen there, or this
        fix would silently disable the drift/resume detection every prior release relied on."""
        h = self._host([(1, 0)])
        assert N.global_root_comparable(h, {"model_root": "r", "event": 3}) is True
        assert N.global_root_comparable(h, {"slot_rounds": {}}) is True
        assert N.global_root_comparable(h, None) is True

    def test_pointer_slot_count_reports_what_the_log_line_claims(self):
        assert N.pointer_slot_count({"slot_rounds": {"1_0": 1, "2_3": 0}}) == 2
        assert N.pointer_slot_count({"rounds": {"1_0": 1}}) == 1
        assert N.pointer_slot_count({"model_root": "r"}) == 0

    def test_the_startup_check_is_actually_gated_on_it(self):
        """Wiring regression: the helper is worthless if _run_async still compares unconditionally."""
        import inspect
        src = inspect.getsource(N._run_async)
        assert "global_root_comparable" in src
        assert "own_coord=(L, E)" in src, "the resume call must target OUR coordinate"


class _CoordFakeHost:
    """One slot per registered coordinate; `_root` is what slot_root would return for it."""

    def __init__(self, coords):
        self.slots = [tuple(c) for c in coords]
        self._d = {"w": 0}
        self.writes = 0

    def index_of(self, L, E):
        t = (int(L), int(E))
        return self.slots.index(t) if t in self.slots else None

    def read_slot(self, j):
        return self._d

    def write_slot(self, j, d):
        self._d = d
        self.writes += 1


class _CoordFakeLane:
    """Accepted records: event e advertises slot_roots for the coordinate given in `adv[e]`."""

    def __init__(self, adv):
        self.names, self.recs = {}, {}
        for e, sr in sorted(adv.items()):
            sha = "sha%d" % e
            self.names[N.accepted_name(e)] = {"sha256": sha}
            self.recs[sha] = {"event": e, "model_root": "global%d" % e, "slot_roots": dict(sr)}

    def manifest(self):
        return dict(self.names)

    def get_json(self, sha):
        return self.recs[sha]


class TestResumeToOwnCoordinate:
    """resume_to_root(own_coord=(L,E)). Targeting the GLOBAL root on a shard-claim network means
    replaying every record and then rolling all of it back, because that digest includes coordinates
    we do not hold. The per-coordinate target is exactly what the coordinator's lineage guard checks
    (_lineage_ok base_slot_root)."""

    @staticmethod
    def _patch(monkeypatch, state):
        """Fold sets the per-coordinate root of whatever the record advertised; slot_root reads it."""
        monkeypatch.setattr(N, "model_root", lambda h: state["global"])

        def fold(host, lane, rec, regate, own_slot, log=None):
            for k, v in (rec.get("slot_roots") or {}).items():
                state[k] = v
            state["global"] = rec["model_root"]
            return True, "ok", []
        monkeypatch.setattr(N, "_fold_accepted_checked", fold)
        monkeypatch.setattr(N, "slot_root",
                            lambda h, i: state.get("%d_%d" % h.slots[i], "base"))

    def test_stops_on_the_per_coordinate_target_not_the_global_one(self, monkeypatch):
        state = {"global": "base"}
        self._patch(monkeypatch, state)
        host = _CoordFakeHost([(1, 0)])
        lane = _CoordFakeLane({1: {"1_0": "mine1"}, 2: {"2_3": "theirs"}, 3: {"1_0": "mine2"}})
        applied, reached = N.resume_to_root(host, lane, "unreachable-global", lambda m: None,
                                           own_coord=(1, 0))
        assert reached is True, "our coordinate WAS reproduced; the global root is irrelevant"
        assert applied == 3
        assert state["1_0"] == "mine2"                  # newest advertisement for us wins
        assert host.writes == 0, "a reached target must not roll back"

    def test_fails_closed_when_no_record_advertises_our_coordinate(self, monkeypatch):
        state = {"global": "base"}
        self._patch(monkeypatch, state)
        host = _CoordFakeHost([(1, 0)])
        lane = _CoordFakeLane({1: {"2_3": "theirs"}, 2: {"2_4": "theirs"}})
        applied, reached = N.resume_to_root(host, lane, "unreachable-global", lambda m: None,
                                           own_coord=(1, 0))
        assert (applied, reached) == (0, False)
        assert host.writes == len(host.slots), "fail-closed: every slot restored"

    def test_an_unregistered_own_coordinate_is_refused_without_replaying(self, monkeypatch):
        state = {"global": "base"}
        self._patch(monkeypatch, state)
        host = _CoordFakeHost([(1, 0)])
        said = []
        applied, reached = N.resume_to_root(host, _CoordFakeLane({1: {"1_0": "x"}}), "g", said.append,
                                           own_coord=(9, 9))
        assert (applied, reached) == (0, False)
        assert any("not registered locally" in s for s in said)

    def test_global_behaviour_is_unchanged_when_own_coord_is_none(self, monkeypatch):
        """The v3.3.2 path must stay byte-identical: stop the moment the GLOBAL root matches."""
        state = {"global": "base"}
        self._patch(monkeypatch, state)
        host = _CoordFakeHost([(1, 0)])
        lane = _CoordFakeLane({1: {"1_0": "a"}, 2: {"1_0": "b"}, 3: {"1_0": "c"}})
        applied, reached = N.resume_to_root(host, lane, "global2", lambda m: None)
        assert (applied, reached) == (2, True), "folds events 1,2 then stops on the global root"
        assert host.writes == 0


class TestBaseSlotsSurviveRegistration:
    """_base_slots is what eval_expert restores the model from after gating a candidate. begin_round
    sized it to the slot list as it was when the event began, so a coordinate REGISTERED mid-event
    (which is the whole feature) left it short -- and eval_expert then did
    write_slot(e, None) -> "TypeError: 'NoneType' object is not subscriptable" inside the merge, which
    apply_delta_gated does not wrap. The coordinator died mid-merge."""

    def test_register_after_begin_round_repairs_base_slots(self, h):
        h.begin_round(h.canonical_experts())
        assert len(h._base_slots) == 2
        idx = h.register(1, 2)                                  # the mid-event claim
        assert len(h._base_slots) == len(h.slots) == 3
        assert isinstance(h._base_slots[idx], dict), "must be a real snapshot, never None"
        assert set(h._base_slots[idx]) == {"gate", "up", "down"}

    def test_the_snapshot_is_the_slots_current_weights_and_is_a_copy(self, h):
        import numpy as np
        h.begin_round(h.canonical_experts())
        idx = h.register(2, 1)
        snap = h._base_slots[idx]
        assert np.array_equal(snap["gate"], h.read_slot(idx)["gate"])
        keep = {k: v.copy() for k, v in h.read_slot(idx).items()}
        d = h.read_slot(idx)
        d["gate"] = d["gate"] + np.float32(1.0)
        h.write_slot(idx, d)                                    # move the model
        assert not np.array_equal(snap["gate"], h.read_slot(idx)["gate"]), "snapshot must be a COPY"
        h.write_slot(idx, keep)                                 # shared module-scoped model: put it back

    def test_re_admitting_an_evicted_slot_refills_its_hole(self, h):
        """evict() sets the entry to None on purpose; re-registering must fill it again or the very
        next gate on that slot writes None."""
        h.register(1, 2)
        h.begin_round(h.canonical_experts())
        h.evict(2)
        assert h._base_slots[2] is None
        h.register(1, 2)
        assert isinstance(h._base_slots[2], dict)

    def test_eval_expert_raises_a_named_error_instead_of_a_typeerror(self, host):
        """Diagnosability: the failure must name the slot and the cause, and must be a RuntimeError
        subclass rather than the TypeError that used to come out of numpy indexing."""
        G, model, cfg, claimable = host
        c = G.GlmExpertLaneHost(model, cfg, [(1, 0), (1, 1)], claimable=claimable)
        meter = types.SimpleNamespace(add_verify=lambda n: None)
        ev = c.make_eval_expert(meter, 8)
        with pytest.raises(RuntimeError, match=r"slot 1 has no pre-round base snapshot"):
            ev(1, c.read_slot(1), [[0, 1]], None)               # begin_round never ran
        c.begin_round(c.canonical_experts())
        c.evict(1)
        with pytest.raises(RuntimeError, match=r"no pre-round base snapshot"):
            ev(1, c.read_slot(1), [[0, 1]], None)               # evicted -> hole
        with pytest.raises(G.MissingBaseSnapshot):               # the NAMED subclass, not a bare RuntimeError
            ev(1, c.read_slot(1), [[0, 1]], None)


@coordinator_only
class TestRecordBaseEventIsPoisonProof:
    """The lane's PUT token is a shared PUBLIC demo token, so any stranger can publish one
    contribution record. `int(r.get("base_event", ...))` was therefore a remote kill switch: null ->
    TypeError, "x" -> ValueError, {} -> TypeError, and neither caller sits inside a try/except (the
    async loop only wraps lane.manifest()). One field exited the coordinator process."""

    @staticmethod
    def _C():
        import sharddiloco_glm_coordinator as C
        return C

    def test_valid_integers_are_unchanged(self):
        C = self._C()
        assert C._record_base_event({"base_event": 7}) == 7
        assert C._record_base_event({"base_event": "7"}) == 7
        assert C._record_base_event({"base_round": 7}) == 7
        assert C._record_base_event({"_base_event": 3}) == 3
        assert C._record_base_event({}) == 0                     # no field at all -> the old default

    def test_every_malformed_shape_returns_none_and_never_raises(self):
        C = self._C()
        for bad in (None, "x", {}, [], object()):
            assert C._record_base_event({"base_event": bad}) is None, repr(bad)

    def test_a_bad_base_event_is_dropped_by_the_staleness_partition(self):
        """The caller must DROP it, not propagate. This is the frame that used to raise."""
        import neurahash.diloco_merge as dm
        C = self._C()
        clock = dm.SlotClock()
        recs = [{"base_event": 0, "_name": "good"}, {"base_event": None, "_name": "poison"}]
        logs = []
        fresh, staled = C._partition_by_staleness(clock, recs, None, C._record_base_event,
                                                  log=logs.append)
        assert [r["_name"] for r in fresh] == ["good"]
        assert staled == 1
        assert any("unusable base_event" in s for s in logs)

    def test_the_partition_signature_stays_back_compatible(self):
        """Existing callers/tests pass four positional args and no log."""
        import neurahash.diloco_merge as dm
        C = self._C()
        fresh, staled = C._partition_by_staleness(dm.SlotClock(), [{"be": 0}], None,
                                                  lambda r: r["be"])
        assert (len(fresh), staled) == (1, 0)


@coordinator_only
class TestLegacyMinerDiagnostic:
    """`wrong-lineage-root` named the symptom and hid the cause. A <=v3.3.2 miner sends no
    base_slot_root, so it is judged on the GLOBAL root -- which is unreachable for it the moment the
    coordinator registers a coordinate, because it hashed a shorter slot list. The fix is upgrading
    that miner, and the log line has to say so."""

    @staticmethod
    def _C():
        import sharddiloco_glm_coordinator as C
        return C

    def test_grown_slot_list_names_the_real_cause(self):
        C = self._C()
        ok, why = C._lineage_ok(0, "OLD", 0, {0: "NEW"}, slots_grew=True)
        assert ok is False and why == "legacy-miner-vs-dynamic-slots"

    def test_a_stable_slot_list_still_reports_wrong_lineage_root(self):
        """Dead-run leftovers and forged bases must keep their own honest reason."""
        C = self._C()
        ok, why = C._lineage_ok(0, "OLD", 0, {0: "NEW"}, slots_grew=False)
        assert ok is False and why == "wrong-lineage-root"
        assert C._lineage_ok(0, "OLD", 0, {0: "NEW"})[1] == "wrong-lineage-root"   # default

    def test_the_new_reason_never_masks_a_valid_record(self):
        C = self._C()
        assert C._lineage_ok(0, "G", 0, {0: "G"}, slots_grew=True) == (True, "ok")

    def test_a_shard_claim_miner_is_unaffected_by_the_flag(self):
        """A miner that sends base_slot_root never reaches the global fallback at all."""
        C = self._C()
        assert C._lineage_ok(0, "STALE", 0, {0: "G"}, base_slot_root="S", want_slot_root="S",
                             slots_grew=True) == (True, "ok")

    def test_the_flag_is_computed_from_the_slot_root_history_at_the_call_site(self):
        """Wiring: a coordinate's history is seeded at the event it is admitted, so a first entry
        later than the record's base event IS 'the slot list grew since then'."""
        C = self._C()
        hist = {}
        C._record_slot_root(hist, (1, 0), 0, "r0")
        assert any(h and int(h[0][0]) > 0 for h in hist.values()) is False
        C._record_slot_root(hist, (2, 3), 12, "seed")
        assert any(h and int(h[0][0]) > 7 for h in hist.values()) is True
        import inspect
        src = inspect.getsource(C.run_async_events)
        assert "slots_grew=_grew" in src, "the coordinator no longer passes the diagnostic through"


class TestClaimAndAdvance:
    """The sweep: claim -> work -> plateau -> release -> claim next. This is the owner's "finish one
    then store it and start the 2nd one" -- storing is a no-op because an accepted delta is already
    merged into the model, so the model IS the store."""

    def test_start_coord_is_deterministic_per_identity(self):
        c = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        a = N.pick_start_coord(c, "0xabc123")
        assert N.pick_start_coord(c, "0xabc123") == a       # same wallet -> same coordinate, always
        assert a in c

    def test_different_identities_spread_across_the_space(self):
        """No registry and no lock, so the only thing stopping every default-configured miner from
        piling onto coordinate 0 is that the start is hashed from the wallet address."""
        c = [(1, e) for e in range(5)]
        got = {N.pick_start_coord(c, "0x%040x" % k) for k in range(40)}
        assert len(got) >= 3                                 # spread, not all one bucket

    def test_start_coord_refuses_an_empty_claimable_set(self):
        with pytest.raises(SystemExit, match=r"no claimable coordinates"):
            N.pick_start_coord([], "0xabc")

    def test_next_claim_coord_cycles_in_order(self):
        c = [(1, 0), (1, 1), (1, 2)]
        assert N.next_claim_coord(c, (1, 0)) == (1, 1)
        assert N.next_claim_coord(c, (1, 1)) == (1, 2)
        assert N.next_claim_coord(c, (1, 2)) == (1, 0)       # wraps -> sweeps forever

    def test_next_claim_coord_is_none_when_there_is_nowhere_to_go(self):
        """One coordinate means advancing would just churn the data reload for nothing."""
        assert N.next_claim_coord([(1, 0)], (1, 0)) is None
        assert N.next_claim_coord([], (1, 0)) is None

    def test_next_claim_coord_recovers_when_current_is_not_in_the_set(self):
        assert N.next_claim_coord([(1, 0), (1, 1)], (9, 9)) == (1, 0)

    def test_record_touched_coord_reads_the_per_coordinate_root_map(self):
        """slot_roots names exactly the ONE coordinate that event merged, which is a precise "the
        coordinator processed MY expert" signal -- the top-level `slot` int is the coordinator's own
        registry index and means nothing to us."""
        rec = {"slot_roots": {"1_3": "abc"}}
        assert N.record_touched_coord(rec, (1, 3)) is True
        assert N.record_touched_coord(rec, (1, 4)) is False
        assert N.record_touched_coord({}, (1, 3)) is False

    def test_accepted_names_me_is_the_verdict_a_miner_could_not_see(self):
        """Before this, apply_accepted matched only item["slot"] and ignored `miner`, so "my delta
        lost", "another miner won" and "the record never arrived" were indistinguishable."""
        rec = {"accepted": [{"miner": "glm-aaaa", "slot": 0}, {"miner": "glm-bbbb", "slot": 1}]}
        assert N.accepted_names_me(rec, "glm-bbbb") is True
        assert N.accepted_names_me(rec, "glm-cccc") is False
        assert N.accepted_names_me({"accepted": []}, "glm-aaaa") is False
        assert N.accepted_names_me({}, "glm-aaaa") is False

    def test_auto_spread_when_no_coordinate_was_requested(self, monkeypatch):
        """A stranger runs the miner with no flags: it must land on a coordinate derived from its
        wallet, not default to index 0 like everyone else."""
        monkeypatch.delenv("NEURAHASH_SD_EXPERT", raising=False)
        claim = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        monkeypatch.setattr(N, "node_claimable_coords", lambda a: claim)
        args = types.SimpleNamespace(expert=None, slot=None, mode="glm", slots="1:0",
                                     domains="daily", piece=0, shard_dir="x", config_dir="x")
        slots = N.parse_slots("1:0")
        L, E, i, src = N.resolve_claim(args, slots, log=lambda *a: None, identity="0xdeadbeef")
        assert (L, E) == N.pick_start_coord(claim, "0xdeadbeef")
        assert "wallet-hash" in src
        assert (L, E) in slots                               # appended so the host can touch it

    def test_an_explicit_expert_flag_still_overrides_the_hash(self, monkeypatch):
        monkeypatch.setattr(N, "node_claimable_coords", lambda a: [(1, 0), (1, 4)])
        args = types.SimpleNamespace(expert="1:4", slot=None, mode="glm", slots="1:0",
                                     domains="daily", piece=0, shard_dir="x", config_dir="x")
        L, E, _, src = N.resolve_claim(args, N.parse_slots("1:0"), log=lambda *a: None,
                                       identity="0xdeadbeef")
        assert (L, E) == (1, 4) and src == "--expert"

    def test_claimability_probe_is_unchecked_not_fatal_on_a_partial_namespace(self):
        """The async lane's dirty-namespace path passes an incomplete args. Refusing to start because
        we could not find a manifest we never needed would be a self-inflicted outage."""
        assert N.node_claimable_coords(types.SimpleNamespace()) is None
        assert N.node_claimable_coords(types.SimpleNamespace(mode="tiny")) is None
        assert N.node_claimable_coords(types.SimpleNamespace(mode="glm")) is None

    def test_claim_all_coords_falls_back_to_the_slot_list(self):
        args = types.SimpleNamespace(mode="tiny")
        assert N.claim_all_coords(args, [(1, 0), (1, 1)]) == [(1, 0), (1, 1)]


class TestUnhostableLayerErrors:
    """Cross-LAYER access is the loud failure mode and it must stay loud AND named -- the old code
    died on a naked IndexError / AttributeError from inside read_slot with no hint about what was
    wrong or what the node actually holds."""

    def test_a_layer_beyond_the_model_names_the_mtp_case(self, host):
        G, model, cfg, _ = host
        c = G.GlmExpertLaneHost(model, cfg, [(N.TINY["layers"], 0)])   # layers=3 -> index 3 unbuilt
        with pytest.raises(IndexError, match=r"not instantiated"):
            c.read_slot(0)


# ==================================================================================================
# ADVERSARIAL PRE-FLIGHT FINDINGS F1-F7 (2026-07-25). Every test below was written from a reviewer's
# reproduction of a defect that the live ONE-MINER, --slot 0 topology happens to dodge, and each one
# goes RED on the code as it stood before that finding's fix. The SECOND miner is where shard claim
# breaks, so these are all two-coordinate / two-miner scenarios.
# ==================================================================================================
import numpy as np                                                    # noqa: E402
import sharddiloco_harness as _H                                      # noqa: E402

_LKEY = b"\x21" * 16


@pytest.fixture(scope="module")
def store_harness():
    """The REAL content-addressed in-process lane from tests/test_sd_async_lane.py (cid == sha256 of the
    stored bytes, with a manifest reveal schedule so "contributor speed" is deterministic and no sleep is
    involved). Imported INSIDE a fixture, not at module scope: that module imports the coordinator, and a
    public-miner checkout without it must skip these tests only -- never the whole file."""
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    return pytest.importorskip("test_sd_async_lane", reason="async lane harness not importable")


@pytest.fixture(scope="module")
def loop_model():
    """A SEPARATE tiny GLM for the loop-driving tests. They MERGE deltas, i.e. they move the model's
    weights; sharing the `host` fixture's model would leak those moves into the registry/root tests."""
    import torch
    torch.set_num_threads(2)
    G = N._G()
    T = N.TINY
    model, cfg = G.build_tiny_glm(seed=T["seed"], vocab=T["vocab"], hidden=T["hidden"],
                                  inter=T["inter"], moe_inter=T["moe_inter"], layers=T["layers"],
                                  n_experts=T["n_experts"], topk=T["topk"])
    return G, model, cfg, [(L, E) for L in (1, 2) for E in range(T["n_experts"])]


def _loop_env(loop_model, store_harness, start_slots, rounds, max_active=None):
    """A pristine coordinator context (fresh host + fresh store + fresh probe) over the loop-test model.

    margin=-1e9 makes the secret-probe gate ACCEPT and merge_tol=1e9 disables the #145 rollback: these
    tests prove WHERE work is addressed and WHICH seat is held, not the gate's discrimination (that is
    tests/test_sd_async_lane.py). A test that silently depended on a random delta improving CE would be
    flaky and would prove nothing either way."""
    import sharddiloco_glm_coordinator as C
    import neurahash.diloco_merge as dm
    G, model, cfg, claimable = loop_model
    args = types.SimpleNamespace(rounds=rounds, poll_timeout=5.0, mode="tiny", outer=0.7, margin=-1e9,
                                 merge_tol=1e9, probe_size=8, eval_chunk=64, threads=2,
                                 slots=",".join("%d:%d" % s for s in start_slots),
                                 domains="code,gutenberg")
    slots = [tuple(s) for s in start_slots]
    host = G.GlmExpertLaneHost(model, cfg, slots, claimable=claimable, max_active=max_active)
    store = store_harness._InProcStore()
    pools = {i: C._slot_probe_pool(args, slots[i]) for i in range(len(slots))}
    probe = dm.SecretRotatedProbe(pools, seed=1234, size=args.probe_size)
    seq = N.TINY["seq"]
    meter = dm.FlopMeter(G.glm_fwd_flops_per_example(cfg, seq))
    heldout = N.coord_secret_ids(args, 0, "heldout")[:64]
    return dict(C=C, G=G, model=model, cfg=cfg, host=host, store=store, probe=probe, meter=meter,
                eval_expert=host.make_eval_expert(meter, seq), heldout=heldout, slots=slots, args=args,
                init_ce=C._chunked_heldout_ce(G, model, heldout, args.eval_chunk),
                claimable=claimable)


def _publish_claim(env, coord, miner, base_event=0, wire_idx=0, host=None, seed=0):
    """Publish ONE contribution addressed by COORDINATE, the way a shard-claim miner does. `host` is the
    MINER's host (its own slot list, hence its own local index) -- deliberately not the coordinator's."""
    h = env["host"] if host is None else host
    idx = h.index_of(*coord)
    ref = h.read_slot(idx if idx is not None else 0)
    rng = np.random.default_rng(seed)
    payload = {k: (rng.standard_normal(v.shape) * 1e-3).astype(np.float32) for k, v in ref.items()}
    cid = env["store"].put_delta(payload)
    rec = N.build_async_contrib_record(
        miner, wire_idx, coord[0], coord[1], base_event, N.model_root(h), cid,
        _H.sign(_LKEY, cid, base_event, miner), 1e9,
        int(len(_H.pack_arrays(payload, np.float16))), 10, 160,
        base_slot_root=(N.slot_root(h, idx) if idx is not None else None))
    name = N.contrib_name(base_event, miner)
    env["store"].put_json_named(name, rec)
    env["store"].schedule_contrib(name)
    return name


def _drive_loop(env, miners):
    """Run the REAL async event loop over this env; `miners` are the roster names (each bound to _LKEY).
    A roster entry per miner is mandatory: open admission is DEFAULT ON, so an unrostered name makes the
    coordinator try to recover a secp256k1 address from an HMAC signature and the record dies in validate
    (rejected_gate) long before any merge."""
    logs = []
    env["C"].run_async_events(env["G"], env["model"], env["host"], env["store"], env["probe"],
                              env["meter"], env["eval_expert"], env["heldout"], env["slots"],
                              {m: {"key": _LKEY, "expert": 0} for m in miners}, env["args"],
                              env["init_ce"], lambda *a: logs.append(" ".join(str(x) for x in a)))
    return logs


@coordinator_only
class TestF1WorkIsMergedWhereItWasClaimed:
    """F1 (CRITICAL): work was ADMITTED by coordinate and MERGED by the miner's wire index.

    Every miner numbers its OWN slots, so two miners on different coordinates routinely publish the SAME
    positional index -- and with wallet-hash auto-spread plus --advance-after on by default, divergent
    index order is the NORMAL case, not an edge case. The DISCOVER loop resolved the true index via
    _admit_coordinate and never wrote it back, so _fetch_validate_contribs re-read the wire field and THAT
    is what indexed experts[e], probe.batch(e, rnd) and the accepted stamp: the delta was gated with
    another coordinate's secret pool, merged into that other coordinate, and stamped as that other
    coordinate so every replica folded it there too -- while slot_roots advertised the UNCHANGED root of
    the coordinate the miner actually claimed, so nothing detected it. The #145 rollback could not undo it
    either (pre_e snapshots the CLAIMED slot, not the corrupted one), and the miner was paid for it."""

    def test_the_merge_lands_on_the_coordinate_the_record_names(self, loop_model, store_harness):
        env = _loop_env(loop_model, store_harness, [(1, 0), (1, 1)], rounds=2)
        G, model, cfg, claimable = loop_model
        host = env["host"]
        # Two miners, two coordinates, and BOTH publish wire index 1 -- each is index 1 in that miner's
        # own two-slot list. The coordinator will assign 2 and 3.
        mh_a = G.GlmExpertLaneHost(model, cfg, [(1, 0)], claimable=claimable)
        mh_b = G.GlmExpertLaneHost(model, cfg, [(1, 0)], claimable=claimable)
        ia, ib = mh_a.register(2, 3), mh_b.register(2, 2)
        assert ia == ib == 1, "the collision this finding is about: same local index, different coordinate"
        root_11_before = N.slot_root(host, 1)          # the innocent bystander the wire index points at
        _publish_claim(env, (2, 3), "mA", wire_idx=ia, host=mh_a, seed=1)
        _publish_claim(env, (2, 2), "mB", wire_idx=ib, host=mh_b, seed=2)

        logs = _drive_loop(env, ["mA", "mB"])
        assert host.index_of(2, 3) == 2 and host.index_of(2, 2) == 3, "\n".join(logs[-10:])

        r1, r2 = env["store"].accepted(1), env["store"].accepted(2)
        assert r1 is not None and r2 is not None, "\n".join(logs[-12:])
        assert r1["accepted"] and r2["accepted"], "\n".join(logs[-12:])
        # 1. each event's accepted row names the coordinate its record CLAIMED
        assert (r1["accepted"][0]["layer"], r1["accepted"][0]["glm_expert"]) == (2, 3)
        assert (r2["accepted"][0]["layer"], r2["accepted"][0]["glm_expert"]) == (2, 2)
        # 2. the weights that MOVED are the claimed coordinates', and they match what was advertised
        assert N.slot_root(host, 2) == r1["slot_roots"]["2_3"]
        assert N.slot_root(host, 3) == r2["slot_roots"]["2_2"]
        # 3. and the coordinate the raw wire index pointed at was never touched by either delta
        assert N.slot_root(host, 1) == root_11_before, "a wire index moved an unclaimed coordinate"


@coordinator_only
class TestF4IdleEvictionFreesASeat:
    """F4 (partial): the active-slot cap was a ONE-WAY RATCHET. Nothing called host.evict except the
    gate-pool failure path, so --max-active-slots (default 16) became permanent: every further claim was
    DEFERRED forever, re-fetched and re-attempted on every poll, with nothing in the miner's log to
    explain it. Eviction is cheap and lossless -- an accepted delta is already merged into the weights
    (the model IS the store) and the index<->coordinate mapping survives forever by design."""

    def test_idle_events_knob_is_read_and_pure(self):
        import sharddiloco_glm_coordinator as C
        assert C._idle_evict_events({}) == 50                          # documented default
        assert C._idle_evict_events({"NEURAHASH_SD_IDLE_EVICT_EVENTS": "7"}) == 7
        assert C._idle_evict_events({"NEURAHASH_SD_IDLE_EVICT_EVENTS": "junk"}) == 50
        assert C._idle_evict_candidates({0, 1}, {0: 5, 1: 0}, 5, 0) == []          # <=0 -> OFF
        assert C._idle_evict_candidates({0, 1}, {0: 5, 1: 0}, 5, 5) == [1]         # only the idle one
        assert C._idle_evict_candidates({0, 1}, {0: 5, 1: 0}, 5, 5, pending={1}) == [], \
            "a slot with queued work must never be evicted -- validate would drop its own records"

    def test_a_coordinate_idle_for_the_configured_events_is_evicted(self, loop_model, store_harness,
                                                                   monkeypatch):
        """Slot 1 is declared at startup and never receives a contribution while slot 0 keeps working.
        After the configured number of events its seat is released; slot 0 keeps merging."""
        monkeypatch.setenv("NEURAHASH_SD_IDLE_EVICT_EVENTS", "2")
        env = _loop_env(loop_model, store_harness, [(1, 0), (1, 1)], rounds=3, max_active=2)
        host = env["host"]
        for k, m in enumerate(("e0", "e1", "e2")):
            _publish_claim(env, (1, 0), m, base_event=0, wire_idx=0, seed=10 + k)
        logs = _drive_loop(env, ["e0", "e1", "e2"])

        assert host.is_active(1) is False, "the idle seat was never released:\n" + "\n".join(logs[-12:])
        assert host.is_active(0) is True, "the WORKING slot must not be evicted"
        assert any("EVICT slot 1 (L1,E1)" in ln for ln in logs), "\n".join(logs[-12:])
        assert env["store"].accepted(3) is not None, "eviction must not stall the merge loop"
        assert host.register(1, 1) == 1, "re-admission resumes on the SAME index (weights kept)"


@coordinator_only
class TestF7ReAdmissionAfterAPartialRegistration:
    """F7: a re-admitted slot could end up with NO gate pool and NO lineage seed. host.register succeeds,
    _slot_probe_pool raises, and by then the coordinate is already in host._idx -- evict() releases the
    working state but KEEPS the mapping (deliberate: indices are never reused). The next record for it
    took the `known is not None` path and skipped BOTH seeds, permanently: _expected_slot_root -> None
    means every shard-claim record for it is dropped `unknown-coordinate` forever, and a <=v3.3.2 record
    that passes the global lineage check reaches probe.batch(e, rnd) -> bare KeyError
    (diloco_merge.py:948), uncaught, killing the coordinator process."""

    class _FlakyProbe:
        """Wraps the real probe and fails the FIRST ensure_pool, like a bad domain/data read would."""

        def __init__(self, inner):
            self.inner, self.fail_next = inner, True

        def has_pool(self, e):
            return self.inner.has_pool(e)

        def ensure_pool(self, e, pool):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("simulated probe-pool build failure")
            return self.inner.ensure_pool(e, pool)

        def batch(self, e, rnd):
            return self.inner.batch(e, rnd)

    @staticmethod
    def _rec():
        return {"layer": 2, "glm_expert": 3, "expert": 0, "miner": "m"}

    def test_a_coordinate_whose_first_pool_build_failed_is_fully_seeded_on_re_admission(self):
        C, host, slots, probe, clock, srh, args = TestAdmitCoordinate._setup()
        flaky = self._FlakyProbe(probe)
        logs = []

        assert C._admit_coordinate(host, slots, self._rec(), "m", args, flaky, clock, srh,
                                   logs.append) is None
        assert any("cannot build a gate pool" in s for s in logs)
        assert host.index_of(2, 3) == 2, "the index<->coordinate mapping survives eviction by design"
        assert flaky.has_pool(2) is False and C._expected_slot_root(srh, (2, 3), 0) is None

        e = C._admit_coordinate(host, slots, self._rec(), "m", args, flaky, clock, srh, logs.append)
        assert e == 2
        assert flaky.has_pool(2) is True, "re-admission must build the gate pool it failed to build"
        assert C._expected_slot_root(srh, (2, 3), 0) is not None, \
            "lineage root never seeded -> every record for this coordinate is unknown-coordinate forever"
        assert flaky.batch(2, 0) is not None, \
            "probe.batch would raise the bare KeyError that kills the coordinator process"
        assert slots == [(1, 0), (1, 1), (2, 3)] and len(host.slots) == 3, "no double append"


class _FoldHost:
    """A lane host with REAL numpy slot weights -- enough for apply_accepted / _fold_accepted_checked /
    slot_root / model_root without building a torch model, and every read/write COPIES so a rollback that
    only looks like a rollback cannot pass."""

    def __init__(self, coords, shape=(2, 3)):
        self.slots = [tuple(c) for c in coords]
        self._w = {i: {k: np.zeros(shape, np.float32) for k in ("gate", "up", "down")}
                   for i in range(len(self.slots))}

    def index_of(self, L, E):
        t = (int(L), int(E))
        return self.slots.index(t) if t in self.slots else None

    def read_slot(self, i):
        return {k: v.copy() for k, v in self._w[int(i)].items()}

    def write_slot(self, i, d):
        self._w[int(i)] = {k: np.array(v, dtype=np.float32, copy=True) for k, v in d.items()}


class _FoldLane:
    """get_delta by cid + an accepted-record manifest, mirroring the surface the fold path calls."""

    def __init__(self, deltas, records=None):
        self._deltas = {k: {kk: np.asarray(vv, dtype=np.float32) for kk, vv in v.items()}
                        for k, v in (deltas or {}).items()}
        self.names, self._recs = {}, {}
        for e, rec in sorted((records or {}).items()):
            sha = "sha%d" % e
            self.names[N.accepted_name(e)] = {"sha256": sha}
            self._recs[sha] = dict(rec, event=int(e))

    def get_delta(self, cid):
        return {k: v.copy() for k, v in self._deltas[cid].items()}

    def manifest(self):
        return dict(self.names)

    def get_json(self, sha):
        return self._recs[sha]


def _row(miner, coord, cid, slot=0, outer=0.7):
    return dict(miner=miner, slot=int(slot), layer=int(coord[0]), glm_expert=int(coord[1]), cid=cid,
                outer=float(outer), paid=0.0, token_weight=None)


def _fold_shape(shape=(2, 3), scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return {k: (rng.standard_normal(shape) * scale).astype(np.float32) for k in ("gate", "up", "down")}


def _root_after(coords, idx, delta, outer=0.7, pre=None):
    """The slot_root a HONEST coordinator would advertise for `coords[idx]` after folding `delta` into it
    (optionally after `pre` = [(idx, delta), ...] earlier folds), computed on a scratch host."""
    h = _FoldHost(coords)
    for j, d in (pre or []):
        cur = h.read_slot(j)
        h.write_slot(j, {k: cur[k] + outer * d[k] for k in cur})
    cur = h.read_slot(idx)
    h.write_slot(idx, {k: cur[k] + outer * delta[k] for k in cur})
    return N.slot_root(h, idx), h


class TestF2NotResidentIsNotPoison:
    """F2 (CRITICAL): a "not resident here" SKIP was classified as POISON, so every contributor rc8-exited.

    _resolve_accepted_slot returns None for a coordinate this node does not hold -- its own comment calls
    that "Normal on a shard-claim network ... Nothing to fold" -- but apply_accepted appended it to
    `rejected`, _fold_accepted_checked then reported `poison`, catch_up_accepted returned abort 8 and
    _run_async exited rc8 "forged/poisoned record" on the FIRST accepted record for ANOTHER miner's
    coordinate. On a network where miners deliberately hold different coordinates that is the very first
    record they see. The same call from the coordinator's _resume_from_lane stopped the resume replay at
    the first dynamically-registered coordinate and rolled the campaign back to the frozen base."""

    @staticmethod
    def _foreign():
        host = _FoldHost([(1, 0)])
        rec = dict(accepted=[_row("stranger", (2, 5), "cX", slot=7)], slot_roots={"2_5": "theirs"},
                   model_root="g1")
        lane = _FoldLane({"cX": _fold_shape(seed=3)}, records={1: rec})
        return host, lane, rec

    def test_a_record_for_a_coordinate_we_do_not_hold_folds_nothing_and_is_not_poison(self):
        host, lane, rec = self._foreign()
        before = N.model_root(host)
        ok, reason, rejected = N._fold_accepted_checked(host, lane, rec, lambda h: 1.0, 0,
                                                        log=lambda *_a: None)
        assert (ok, reason, rejected) == (True, "ok", []), "a coordinate we do not hold is not poison"
        assert N.model_root(host) == before, "nothing of ours can move -- the expert is not resident"

    def test_the_contributor_does_not_self_abort_rc8_on_it(self):
        host, lane, _rec = self._foreign()
        last_applied, applied_any, abort = N.catch_up_accepted(
            host, lane, lane.manifest(), 0, 1, lambda h: 1.0, 0, "me", lambda *_a: None)
        assert abort is None, "rc8 on a normal shard-claim record kills every contributor on the network"
        assert (last_applied, applied_any) == (1, True), "the frontier must still advance past it"

    def test_apply_accepted_reports_it_on_the_skipped_channel_never_rejected(self):
        host, lane, rec = self._foreign()
        rejected, skipped, folded = [], [], set()
        n = N.apply_accepted(host, lane, rec, rejected=rejected, skipped=skipped, folded_slots=folded)
        assert (n, rejected, folded) == (0, [], set())
        assert [s["reason"] for s in skipped] == ["unknown-coordinate"]

    def test_a_real_shape_mismatch_is_still_poison(self):
        """The fix must NARROW `rejected`, not empty it: a delta whose shapes do not match the resident
        slot is still refused and still reported as poison."""
        host = _FoldHost([(1, 0)])
        rec = dict(accepted=[_row("attacker", (1, 0), "bad")], slot_roots={"1_0": "x"}, model_root="g1")
        lane = _FoldLane({"bad": _fold_shape(shape=(5, 5), seed=4)}, records={1: rec})
        ok, reason, rejected = N._fold_accepted_checked(host, lane, rec, None, 0, log=lambda *_a: None)
        assert (ok, reason) == (False, "poison")
        assert [r["reason"] for r in rejected] == ["shape-mismatch"]


class TestF3EveryFoldedCoordinateMustBeAdvertised:
    """F3 (HIGH, security): replica_root_ok was bypassable with PUBLIC data.

    It returned True when `checked == 0` (none of the ADVERTISED coordinates are resident) and never
    checked that the coordinates it actually FOLDED were advertised at all. The lane is UNSIGNED and its
    PUT token is a shared PUBLIC demo token, so a forged accepted record whose delta targets a RESIDENT,
    NON-OWN coordinate -- advertising either a non-resident coordinate or a resident-but-untouched one
    (using its already-published root) -- folded with no verification whatsoever. The F2 local re-gate
    does not cover it: the target is not own_slot, so it is folded unconditionally on the coordinator's
    "signed" accept. Measured by the reviewer: weights 0.0 -> 9000.0."""

    _COORDS = [(1, 0), (1, 1)]                 # slot 0 = ours to train, slot 1 = resident, not ours

    def _attack(self, slot_roots):
        host = _FoldHost(self._COORDS)
        big = _fold_shape(scale=9e3, seed=5)
        rec = dict(accepted=[_row("attacker", (1, 1), "big", slot=1)],
                   slot_roots=dict(slot_roots), model_root="forged")
        lane = _FoldLane({"big": big}, records={1: rec})
        return host, lane, rec

    def test_a_folded_coordinate_absent_from_slot_roots_fails_closed(self):
        host, lane, rec = self._attack({"9_9": "not-even-resident"})
        ok, reason, _rej = N._fold_accepted_checked(host, lane, rec, None, 0, log=lambda *_a: None)
        assert (ok, reason) == (False, "lineage"), "an unadvertised fold must fail CLOSED"
        assert float(np.abs(host.read_slot(1)["gate"]).max()) == 0.0, "rollback did not happen"

    def test_advertising_a_resident_but_untouched_coordinate_does_not_launder_it(self):
        """The nastier variant: the forged record advertises a coordinate we DO hold, with its correct
        already-published root, so the old loop found checked == 1 and matched."""
        host = _FoldHost(self._COORDS)
        honest_root_of_slot0 = N.slot_root(host, 0)
        host, lane, rec = self._attack({"1_0": honest_root_of_slot0})
        ok, reason, _rej = N._fold_accepted_checked(host, lane, rec, None, 0, log=lambda *_a: None)
        assert (ok, reason) == (False, "lineage")
        assert float(np.abs(host.read_slot(1)["gate"]).max()) == 0.0
        assert N.slot_root(host, 0) == honest_root_of_slot0, "the bystander slot must be untouched too"

    def test_an_honest_cross_domain_record_still_folds(self):
        """Fail-closed must not mean fail-always: a record that advertises the coordinate it moves, with
        the root that fold produces, is still accepted -- that is the ordinary cross-domain replay."""
        d = _fold_shape(seed=6)
        want, _h = _root_after(self._COORDS, 1, d)
        host = _FoldHost(self._COORDS)
        rec = dict(accepted=[_row("peer", (1, 1), "c", slot=1)], slot_roots={"1_1": want},
                   model_root="g1")
        lane = _FoldLane({"c": d}, records={1: rec})
        ok, reason, _rej = N._fold_accepted_checked(host, lane, rec, None, 0, log=lambda *_a: None)
        assert (ok, reason) == (True, "ok")
        assert N.slot_root(host, 1) == want

    def test_replica_root_ok_is_unchanged_when_the_caller_tracks_nothing(self):
        """`folded` None -> the pre-F3 behaviour, so no other caller changes meaning."""
        host = _FoldHost(self._COORDS)
        rec = dict(slot_roots={"9_9": "x"})
        assert N.replica_root_ok(host, rec) is True
        assert N.replica_root_ok(host, rec, folded=set()) is True
        assert N.replica_root_ok(host, rec, folded={1}) is False


class TestF5aStreakOnlyCountsVerdictsOnOurOwnWork:
    """F5a: the sweep advanced on rejects it never earned. `reject_streak` incremented for ANY event whose
    slot_roots named our coordinate and whose accepted rows lacked our name -- including a co-claimant
    winning the same coordinate, our record being lineage/stale/validate-dropped, and records that PREDATE
    our first publish. A fresh miner joining a running campaign folds the WHOLE history in one catch-up
    pass, so the streak blew past advance_after (default 3) and it abandoned its coordinate before
    publishing once."""

    _HIST = [{"event": e, "slot_roots": {"1_3": "r%d" % e},
              "accepted": [{"miner": "someone-else", "slot": 0}]} for e in range(1, 11)]

    def test_history_folded_before_our_first_publish_is_never_a_verdict(self):
        assert all(N.record_touched_coord(r, (1, 3)) for r in self._HIST), "precondition: they name us"
        assert [N.event_judged_us(r, None) for r in self._HIST] == [False] * 10, \
            "10 historical rejects would blow past advance_after=3 before we ever published"

    def test_only_events_at_or_after_our_last_publish_can_judge_us(self):
        assert N.event_judged_us({"event": 3}, 5) is False        # committed before our delta existed
        assert N.event_judged_us({"event": 5}, 5) is True
        assert N.event_judged_us({"event": 7}, 5) is True
        assert [r["event"] for r in self._HIST if N.event_judged_us(r, 8)] == [8, 9, 10]

    def test_an_undateable_record_is_never_a_verdict(self):
        for bad in ({}, {"event": None}, {"event": "x"}, {"event": {}}):
            assert N.event_judged_us(bad, 0) is False, repr(bad)

    def test_the_sweep_is_actually_gated_on_it(self):
        """Wiring: the helper is worthless if the loop still counts every touching event."""
        import inspect
        src = inspect.getsource(N._run_async)
        assert "event_judged_us(_rec, last_pub_base_event)" in src
        assert "last_pub_base_event = int(base_event)" in src, "the publish must record its base_event"


class TestF5bAdvanceLandsOnTheCoordinatorsBase:
    """F5b: after host.register(*nxt) the new coordinate's local weights are the FROZEN BASE, and
    catch_up_accepted only scans (last_applied, frontier] -- so if anyone had already trained that
    coordinate, our base_slot_root could never match and EVERY later contribution was dropped
    `wrong-lineage-slot-root` forever, silently. The advance has to bring the new coordinate up to the
    coordinator's state, targeted per-coordinate (the global root is unreachable on a shard-claim
    network) and fail-closed."""

    def test_the_advance_replays_the_new_coordinates_history(self):
        """Functional: a coordinate whose history we never folded (it was not resident when those events
        happened) is brought to exactly the root the coordinator advertised for it."""
        coords = [(1, 0), (1, 1)]
        d0, d1 = _fold_shape(seed=7), _fold_shape(seed=8)
        r0, _h = _root_after(coords, 0, d0)
        r1, _h2 = _root_after(coords, 1, d1, pre=[(0, d0)])
        host = _FoldHost(coords)                                  # frozen base for BOTH coordinates
        lane = _FoldLane(
            {"c0": d0, "c1": d1},
            records={1: dict(accepted=[_row("m0", (1, 0), "c0", slot=0)], slot_roots={"1_0": r0},
                             model_root="g1"),
                     2: dict(accepted=[_row("m1", (1, 1), "c1", slot=1)], slot_roots={"1_1": r1},
                             model_root="g2")})
        applied, reached = N.resume_to_root(host, lane, "global-root-we-can-never-reach",
                                            lambda *_a: None, own_coord=(1, 1))
        assert (applied, reached) == (2, True)
        assert N.slot_root(host, 1) == r1, "the freshly claimed coordinate is on the coordinator's base"

    def test_the_advance_branch_is_actually_wired_to_it(self):
        """Wiring: the mechanism already existed (resume_to_root own_coord=) -- the defect was that the
        PLATEAU/advance branch never called it."""
        import inspect
        src = inspect.getsource(N._run_async)
        i_reg = src.index("host.register(*nxt)")
        i_res = src.index("resume_to_root(host, lane, pointer_root, log, own_coord=nxt)")
        assert i_res > i_reg, "the catch-up must run AFTER the new coordinate is registered"
        assert "last_pub_base_event = None" in src, "nothing of ours is in flight on the new coordinate"
    def test_the_replay_does_not_double_apply_what_we_already_folded(self):
        """The advance replays the WHOLE accepted history, including records we folded long ago. Those must
        not be applied twice, and they are not: re-folding one produces a root that does not match the
        `slot_roots` entry that record itself advertises for that coordinate, so _fold_accepted_checked
        rolls it back and the replay moves on. (That is the ADVERTISED-root check, which predates F3 --
        this test is the regression guard for the F5b replay being safe to run, not an F3 test.)"""
        coords = [(1, 0), (1, 1)]
        d0, d1 = _fold_shape(seed=7), _fold_shape(seed=8)
        r0, host = _root_after(coords, 0, d0)              # host = OUR base, with event 1 already folded
        r1, _h2 = _root_after(coords, 1, d1, pre=[(0, d0)])
        lane = _FoldLane(
            {"c0": d0, "c1": d1},
            records={1: dict(accepted=[_row("m0", (1, 0), "c0", slot=0)], slot_roots={"1_0": r0},
                             model_root="g1"),
                     2: dict(accepted=[_row("m1", (1, 1), "c1", slot=1)], slot_roots={"1_1": r1},
                             model_root="g2")})
        applied, reached = N.resume_to_root(host, lane, "global-root-we-can-never-reach",
                                            lambda *_a: None, own_coord=(1, 1))
        assert reached is True and applied == 1, "event 1 was already ours; only event 2 is new work"
        assert N.slot_root(host, 0) == r0, "an already-folded delta must NOT be applied a second time"
        assert N.slot_root(host, 1) == r1


class TestF6NoProgressGuardSkippedWhenSlotSetsDiffer:
    """F6: healthy shard-claim miners self-aborted rc6. The in-loop no-progress guard compares our GLOBAL
    model_root against the pointer's, and global_root_comparable gated only the STARTUP check. A
    shard-claim miner holds a different slot set BY CONSTRUCTION, so `local == pointer` is unsatisfiable
    and its only protection was folding a record within --round-wait (default 300 s): one quiet 5-minute
    window killed every miner on the network."""

    def test_incomparable_roots_never_abort(self):
        assert N.async_should_abort_no_progress("ours", "theirs", False, 99999.0, 300.0,
                                                comparable=False) is False

    def test_comparable_roots_keep_the_old_behaviour_exactly(self):
        # same five arguments the pre-F6 call site passed, plus the explicit True
        assert N.async_should_abort_no_progress("ours", "theirs", False, 300.0, 300.0) is True
        assert N.async_should_abort_no_progress("ours", "theirs", False, 300.0, 300.0,
                                                comparable=True) is True
        assert N.async_should_abort_no_progress("ours", "theirs", True, 99999.0, 300.0,
                                                comparable=True) is False
        assert N.async_should_abort_no_progress("same", "same", False, 99999.0, 300.0,
                                                comparable=True) is False
        assert N.async_should_abort_no_progress("ours", "", False, 99999.0, 300.0,
                                                comparable=True) is False

    def test_the_two_states_this_guard_sees_on_a_shard_claim_network(self):
        """The gate value is not hypothetical: it is global_root_comparable over the pointer's slot map."""
        ours = types.SimpleNamespace(slots=[(1, 0), (1, 1)])
        same = {"slot_rounds": {"1_0": 3, "1_1": 1}}
        grown = {"slot_rounds": {"1_0": 3, "1_1": 1, "2_3": 2}}      # someone claimed (2,3)
        assert N.async_should_abort_no_progress(
            "ours", "theirs", False, 99999.0, 300.0,
            comparable=N.global_root_comparable(ours, grown)) is False
        assert N.async_should_abort_no_progress(
            "ours", "theirs", False, 99999.0, 300.0,
            comparable=N.global_root_comparable(ours, same)) is True

    def test_the_in_loop_guard_is_actually_gated_on_it(self):
        import inspect
        src = inspect.getsource(N._run_async)
        assert "comparable=global_root_comparable(host, dec)" in src
