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
import argparse
import ipaddress
import json
import os
import sys
import threading
import time
import types
import urllib.parse

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


# ================================================== MULTI-PIECE residency (--pieces), no torch
# WHY: run 4 plateaued at held-out CE 6.51103 and then ran ~620 events over ~7.5 h with ZERO accepted
# merges. Not archaeology (760 of 769 events went to the two live miners) and not dead miners (both
# training at ~40 s/round). The cause was structural: --piece is ONE int, one piece covers 5
# coordinates, so the campaign's entire trainable universe was (L1,E0)..(L1,E4) and the miners cycled
# 129 PLATEAU -> RELEASE -> CLAIM transitions across the same five forever. Shard Claim worked; it had
# nowhere to go (scratchpad/FINDING_five_coordinate_ceiling.md). On the live manifest layer 1 is
# covered by pieces 0..12 = all 64 experts, and piece_loader allocates a resident layer's fused params
# FULL WIDTH either way -- so widening residency fills rows of tensors that already exist.
@pytest.fixture
def shard_dir(tmp_path):
    """A manifest + config shaped like the real one but scaled to the tiny model (layers 0..2 with
    layer 0 dense, 4 experts per layer), so the same fixtures drive both the pure filter and the real
    host registry. Piece 3 is 100% MTP (layer 3 == num_hidden_layers), mirroring live pieces 589-601.

    No piece FILES are written: node_claimable_coords reads the manifest as pure metadata
    (require_files=False), which is the whole point of doing the claim check before the 5.67 GB load."""
    pieces = {0: [[1, 0], [1, 1]], 1: [[1, 2], [1, 3]], 2: [[2, 0], [2, 1]], 3: [[3, 0], [3, 1]]}
    man = {"version": 1, "n_pieces": len(pieces) + 1,
           "pieces": [{"piece": "trunk", "experts": []}]
                     + [{"piece": "experts_%d" % p, "experts": e} for p, e in sorted(pieces.items())]}
    (tmp_path / "model_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 3, "first_k_dense_replace": 1}), encoding="utf-8")
    return str(tmp_path)


def _piece_args(shard_dir, pieces=None, piece=0, expert=None):
    """A --mode glm namespace carrying exactly the fields the piece/claim path reads."""
    return types.SimpleNamespace(mode="glm", shard_dir=shard_dir, config_dir=None,
                                 piece=piece, pieces=pieces, expert=expert, slot=None,
                                 slots="1:0,1:1", domains="code,gutenberg")


class TestParsePieces:
    """Parsing is where a silent failure would hurt most: degrading a typo to piece 0 would rebuild
    the exact five-coordinate ceiling this flag exists to remove, and nothing downstream would say so."""

    def test_a_single_id_is_a_one_element_list(self):
        assert N.parse_pieces("7") == [7]

    def test_a_comma_list_keeps_every_id(self):
        assert N.parse_pieces("0,1,2") == [0, 1, 2]

    def test_a_range_is_inclusive_on_both_ends(self):
        """'0-12' must be THIRTEEN pieces: on the live manifest that is exactly layer 1's 64 experts,
        and an exclusive end would leave it 59/64 with the missing rows silently inert."""
        assert N.parse_pieces("0-12") == list(range(13))
        assert len(N.parse_pieces("0-12")) == 13

    def test_overlapping_and_duplicate_entries_collapse_and_sort(self):
        assert N.parse_pieces("2,0-3,1,3") == [0, 1, 2, 3]
        assert N.parse_pieces("5-7,6-8") == [5, 6, 7, 8]

    def test_whitespace_is_tolerated(self):
        assert N.parse_pieces(" 0 - 2 , 5 ") == [0, 1, 2, 5]

    @pytest.mark.parametrize("bad", ["", "   ", "a", "0,,1", "1-", "-", "1-b", "0-2,", "3-1", "-1"])
    def test_every_malformed_value_fails_loudly_instead_of_selecting_piece_0(self, bad):
        with pytest.raises(SystemExit) as ex:
            N.parse_pieces(bad)
        assert "--pieces" in str(ex.value)


class TestNodePieceIds:
    """--pieces wins when given; --piece keeps working byte-for-byte otherwise."""

    def test_pieces_selects_the_union(self, shard_dir):
        assert N.node_piece_ids(_piece_args(shard_dir, pieces="0-2")) == [0, 1, 2]

    def test_piece_alone_is_unchanged(self, shard_dir):
        assert N.node_piece_ids(_piece_args(shard_dir, piece=1)) == [1]

    def test_pieces_overrides_piece(self, shard_dir):
        assert N.node_piece_ids(_piece_args(shard_dir, pieces="2", piece=0)) == [2]

    def test_an_args_namespace_without_pieces_at_all_still_works(self):
        """Callers predating this flag (and the async lane's dirty-namespace test) pass a partial
        namespace; getattr-defaulting is what keeps them on the old single-piece path."""
        args = types.SimpleNamespace(piece=4)
        assert N.node_piece_ids(args) == [4]

    def test_an_empty_pieces_string_falls_back_to_piece(self, shard_dir):
        """NEURAHASH_GLM_PIECES='' must mean 'unset', not 'parse the empty string and die'."""
        assert N.node_piece_ids(_piece_args(shard_dir, pieces="", piece=3)) == [3]


# ============================================ DEFAULT residency = FILL THE RESIDENT LAYER (no flags)
# WHY: --pieces removed the ceiling only for an operator who already knows to type it. A stranger who
# just runs the miner tomorrow got the SAME five coordinates that stalled run 4 -- 620 events, 7.5 h,
# zero accepted merges. So the default itself has to fill the layer. It is free: MEASURED 2026-07-26
# on the real model, --piece 0 = 5 coords / 2,764,301,056 params / 1.857 GiB, --pieces 0-11 = 60
# coords / 2,764,301,056 params / 1.859 GiB. Identical parameter count, +0.002 GiB. Crossing a layer
# is NOT free: experts_12 holds (1,60)..(1,63) PLUS (2,0), and that one coordinate materialises a
# second full-width MoE layer (+603,979,776 params, +1.126 GiB) -- hence "never straddle by default".
@pytest.fixture
def live_shape_shard_dir(tmp_path):
    """A manifest with the REAL manifest's SHAPE, small enough to be pure dict work: 64 experts per
    layer, 5 experts per piece, layer 0 dense, and the last layer the MTP/nextn one the model never
    instantiates. That reproduces exactly the geometry the finding turns on -- layer 1 is covered by
    the twelve clean pieces 0..11 plus the STRADDLER experts_12 = (1,60)..(1,63) + (2,0) -- so these
    tests assert the same boundary the live model does, without a 5.67 GB load.

    Layer 2 is present and complete so a straddling ANCHOR has somewhere real to straddle into, and
    piece 25 = (2,61)..(2,63) + (3,0),(3,1) straddles into the MTP layer, which must NOT count as a
    straddle (an unreal layer is never instantiated, so it costs nothing)."""
    coords = [[L, E] for L in (1, 2, 3) for E in range(64)]
    pieces = {p: coords[5 * p:5 * p + 5] for p in range((len(coords) + 4) // 5)}
    man = {"version": 1, "n_pieces": len(pieces) + 1,
           "pieces": [{"piece": "trunk", "experts": []}]
                     + [{"piece": "experts_%d" % p, "experts": e} for p, e in sorted(pieces.items())]}
    (tmp_path / "model_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 3, "first_k_dense_replace": 1}), encoding="utf-8")
    return str(tmp_path)


def _default_args(shard_dir, expert=None):
    """A --mode glm namespace with NEITHER piece flag set -- what a stranger's command line resolves
    to now that argparse defaults --piece to None instead of 0."""
    return _piece_args(shard_dir, pieces=None, piece=None, expert=expert)


def _manifest_and_cfg(shard_dir):
    return (PL.load_manifest(shard_dir, require_files=False),
            types.SimpleNamespace(num_hidden_layers=3, first_k_dense_replace=1))


class TestDefaultPieceIdsFillTheLayer:
    """The rule, stated as arithmetic: fill every piece whose experts lie ENTIRELY inside the layer(s)
    the anchor already makes resident, and exclude anything that would add a layer."""

    def test_the_default_is_every_same_layer_piece_and_excludes_the_straddler(
            self, live_shape_shard_dir):
        """The exact piece list, not a count: 0..11 is layer 1's first 60 experts, and experts_12 is
        left out because its (2,0) would materialise a whole second MoE layer for one coordinate."""
        man, cfg = _manifest_and_cfg(live_shape_shard_dir)
        ids, excluded = N.default_piece_ids(man, cfg, anchor=0)
        assert ids == list(range(12))
        assert excluded == [(12, [2])]

    def test_the_default_coordinate_count_is_sixty(self, live_shape_shard_dir):
        """60, not 5. The whole finding is the number: five coordinates is what a campaign exhausts
        in 21 merges and then grinds on for 7.5 h with nothing left to claim."""
        got = N.node_claimable_coords(_default_args(live_shape_shard_dir))
        assert len(got) == 60
        assert got[0] == (1, 0) and got[-1] == (1, 59)
        assert all(L == 1 for (L, _) in got)                 # one layer: the free one

    def test_the_default_needs_no_flags_at_all(self, live_shape_shard_dir):
        assert N.node_piece_ids(_default_args(live_shape_shard_dir)) == list(range(12))
        assert N.resolve_piece_selection(_default_args(live_shape_shard_dir))["source"] == "default"

    def test_a_manifest_whose_layer_needs_only_two_pieces_fills_exactly_those(self, shard_dir):
        """The rule is manifest-driven, not a hardcoded 0-11: on the tiny fixture layer 1 is pieces
        0 and 1, so the default is [0, 1] -- and piece 2 (layer 2) is neither filled nor reported as
        a straddler, because it shares no layer with the anchor at all."""
        assert N.node_piece_ids(_default_args(shard_dir)) == [0, 1]
        assert N.node_claimable_coords(_default_args(shard_dir)) == [(1, 0), (1, 1), (1, 2), (1, 3)]
        assert N.resolve_piece_selection(_default_args(shard_dir))["excluded"] == []


class TestExplicitFlagsStillWinUnchanged:
    """The no-silent-change guard. Someone whose launch script says --piece 0 today must get exactly
    today's five coordinates tomorrow; the new behaviour applies ONLY when neither flag was given."""

    def test_piece_zero_explicitly_is_still_exactly_five_coordinates(self, live_shape_shard_dir):
        args = _piece_args(live_shape_shard_dir, piece=0)
        assert N.node_piece_ids(args) == [0]
        assert N.node_claimable_coords(args) == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        assert N.resolve_piece_selection(args)["source"] == "--piece"

    def test_pieces_explicitly_still_wins(self, live_shape_shard_dir):
        args = _piece_args(live_shape_shard_dir, pieces="0-5")
        assert N.node_piece_ids(args) == list(range(6))
        assert len(N.node_claimable_coords(args)) == 30

    def test_pieces_may_name_the_straddler_on_purpose(self, live_shape_shard_dir):
        """Excluding the straddler is a DEFAULT, not a policy: 0-12 buys 65 coordinates for a second
        layer, and an operator who wants that pays for it explicitly (measured +1.126 GiB)."""
        args = _piece_args(live_shape_shard_dir, pieces="0-12")
        assert N.node_piece_ids(args) == list(range(13))
        got = N.node_claimable_coords(args)
        assert len(got) == 65 and (2, 0) in got

    def test_pieces_still_overrides_piece(self, live_shape_shard_dir):
        assert N.node_piece_ids(_piece_args(live_shape_shard_dir, pieces="7", piece=0)) == [7]


class TestStraddlingAnchor:
    """A straddling ANCHOR is not a special case -- it is the same economics. experts_12 already
    materialises BOTH layer 1 and layer 2 full-width just by loading, so filling both is still free,
    and refusing to would leave rows of tensors already paid for permanently untrainable."""

    def test_a_straddling_anchor_fills_both_of_its_layers(self, live_shape_shard_dir):
        man, cfg = _manifest_and_cfg(live_shape_shard_dir)
        ids, excluded = N.default_piece_ids(man, cfg, anchor=12)
        assert ids == list(range(26))                # every piece whose real layers are within {1,2}
        assert 12 in ids
        assert excluded == []                        # nothing can straddle OUT of the last real layer
        assert len(PL.claimable_expert_ids(man, ids, cfg)) == 128       # both layers, 64 each

    def test_a_piece_straddling_only_into_the_mtp_layer_is_not_a_straddler(
            self, live_shape_shard_dir):
        """Piece 25 is (2,61)..(2,63) + (3,0),(3,1). Layer 3 is the MTP/nextn layer that
        Glm4MoeLiteForCausalLM never instantiates, so including 25 adds no resident layer and no
        claimable coordinate beyond layer 2's -- it must be filled, not excluded."""
        man, cfg = _manifest_and_cfg(live_shape_shard_dir)
        ids, _ = N.default_piece_ids(man, cfg, anchor=12)
        assert 25 in ids
        assert all(L in (1, 2) for (L, _) in PL.claimable_expert_ids(man, ids, cfg))

    def test_an_anchor_with_no_real_layers_yields_only_itself(self, live_shape_shard_dir):
        """Pieces 26+ here (589-601 live) are 100% MTP. The default must NOT quietly substitute some
        other piece: resolve_claim's existing 'holds NO real experts' error is the right failure, and
        it only fires if we hand back the anchor the operator's manifest actually pointed at."""
        man, cfg = _manifest_and_cfg(live_shape_shard_dir)
        assert N.default_piece_ids(man, cfg, anchor=30) == ([30], [])


def _materialise_pieces(shard_dir, present):
    """Give a metadata-only fixture a REAL pieces/ dir holding the trunk plus exactly `present`.

    Contents are irrelevant: every disk check on this path -- ours and
    piece_loader.load_manifest's (piece_loader.py:178-181) -- is os.path.exists, so empty files
    reproduce a partial fetch exactly, at zero bytes and with no torch."""
    pdir = os.path.join(shard_dir, "pieces")
    os.makedirs(pdir, exist_ok=True)
    for nm in ["trunk"] + ["experts_%d" % int(p) for p in present]:
        open(os.path.join(pdir, nm + ".safetensors"), "wb").close()
    return shard_dir


# ============================ the DEFAULT is BEST-EFFORT over what was actually fetched (no torch)
# WHY: the layer-filling default asks for pieces 0-11, but the published quickstart tells a new joiner
# to fetch `--pieces 0`. MEASURED on the live fleet 2026-07-26: a 4060 holding only experts_0 stopped
# starting at all --
#   File "C:\Users\User\nu_4060\tools\piece_loader.py", line 181, in load_manifest
#     raise FileNotFoundError("piece file missing: %s" % fp)
#   FileNotFoundError: piece file missing: C:/Users/User/glm_base\pieces\experts_1.safetensors
# reached from build_node_model -> build_partial_model -> load_manifest(require_pieces=...). A node
# that CAN train 5 coordinates must not refuse to train because it cannot train 60, and every new
# joiner following the README hit this. An EXPLICIT selection is the opposite case: those pieces were
# named by a human, so a missing one still has to be fatal.
class TestTheDefaultDegradesToWhatIsOnDisk:
    """Best effort when nobody named the pieces; unchanged hard failure when somebody did."""

    def test_a_dir_holding_only_the_anchor_starts_instead_of_dying(self, live_shape_shard_dir):
        """RED then GREEN on the exact live crash, in one test: the set the pre-fix default handed the
        loader (0-11) still raises FileNotFoundError on experts_1 -- that is the bug, reproduced --
        while the resolved set now loads and yields the anchor's five coordinates."""
        sd = _materialise_pieces(live_shape_shard_dir, [0])
        args = _default_args(sd)
        with pytest.raises(FileNotFoundError) as ex:
            PL.load_manifest(sd, require_pieces=list(range(12)))      # what the old default asked for
        assert "experts_1.safetensors" in str(ex.value)
        assert N.node_piece_ids(args) == [0]
        PL.load_manifest(sd, require_pieces=N.node_piece_ids(args))   # the same call, now fine
        assert N.node_claimable_coords(args) == [(1, E) for E in range(5)]

    def test_a_partial_fetch_uses_exactly_the_pieces_present(self, live_shape_shard_dir):
        """Three fetched pieces = three resident pieces = 15 coordinates. Not 12 pieces (crash), not
        1 (throwing away two thirds of what the operator already paid to download)."""
        args = _default_args(_materialise_pieces(live_shape_shard_dir, [0, 1, 2]))
        assert N.node_piece_ids(args) == [0, 1, 2]
        assert len(N.node_claimable_coords(args)) == 15

    def test_the_startup_line_names_what_was_skipped_and_how_to_get_it(self, live_shape_shard_dir):
        """Silently training a smaller set is how run 4 burned 7.5 h at a plateau nobody could
        explain. The operator must see BOTH that they are below the layer's ceiling and the one
        command that fixes it."""
        sd = _materialise_pieces(live_shape_shard_dir, [0, 1, 2])
        txt = N.fmt_piece_selection(_default_args(sd))
        assert "using 3 of 12 piece(s)" in txt and "SKIPPED" in txt
        assert "fetch_glm_base.py --dest %s --pieces 3,4,5,6,7,8,9,10,11" % sd in txt
        assert N.resolve_piece_selection(_default_args(sd))["absent"] == list(range(3, 12))

    def test_an_explicit_pieces_range_still_fails_loudly_on_a_partial_dir(self, live_shape_shard_dir):
        """The operator NAMED 0-11. Quietly handing back 1 piece would train 5 coordinates while they
        believe they bought 60 -- worse than the crash, because nothing would ever say so."""
        sd = _materialise_pieces(live_shape_shard_dir, [0])
        assert N.node_piece_ids(_piece_args(sd, pieces="0-11")) == list(range(12))
        with pytest.raises(FileNotFoundError, match="experts_1.safetensors"):
            PL.load_manifest(sd, require_pieces=N.node_piece_ids(_piece_args(sd, pieces="0-11")))

    def test_an_explicit_single_piece_on_a_full_dir_is_still_exactly_five(self, live_shape_shard_dir):
        """--piece pins residency; the disk filter must not widen OR narrow an explicit selection."""
        sd = _materialise_pieces(live_shape_shard_dir, range(26))
        assert N.node_piece_ids(_piece_args(sd, piece=0)) == [0]
        assert len(N.node_claimable_coords(_piece_args(sd, piece=0))) == 5

    def test_a_fully_fetched_dir_still_gets_the_whole_layer(self, live_shape_shard_dir):
        """No regression: when every piece is there the default is the same 12 pieces / 60
        coordinates it resolved before this filter existed, with nothing reported skipped."""
        sd = _materialise_pieces(live_shape_shard_dir, range(26))
        args = _default_args(sd)
        assert N.node_piece_ids(args) == list(range(12))
        assert len(N.node_claimable_coords(args)) == 60
        assert N.resolve_piece_selection(args)["absent"] == []
        assert "SKIPPED" not in N.fmt_piece_selection(args)

    def test_a_metadata_only_dir_still_resolves_the_full_default(self, live_shape_shard_dir):
        """THE THIRD STATE. A shard dir with no pieces/ at all is the PRE-FETCH case
        load_manifest(require_files=False) exists for (piece_loader.py:146-150): a cold node asks what
        it SHOULD fetch before anything is on disk. Intersecting there would answer 'nothing'."""
        assert not os.path.isdir(os.path.join(live_shape_shard_dir, "pieces"))
        assert N.node_piece_ids(_default_args(live_shape_shard_dir)) == list(range(12))

    def test_not_even_the_anchor_present_is_fatal_and_says_how_to_fix_it(self, live_shape_shard_dir):
        """Empty intersection is not a degrade -- there is no expert to train at all. Failing with the
        dir AND the command beats booting a node that would train nothing and never say why."""
        sd = _materialise_pieces(live_shape_shard_dir, [])            # trunk only
        with pytest.raises(SystemExit) as ex:
            N.node_piece_ids(_default_args(sd))
        msg = str(ex.value)
        assert sd in msg and "fetch_glm_base.py --dest %s --pieces 0," % sd in msg

    def test_pieces_fetched_mid_run_do_not_widen_a_running_nodes_claims(self, live_shape_shard_dir):
        """The hazard this fix's own advice creates: an operator pastes the printed fetch command
        WITHOUT restarting. The model was loaded with piece 0 only, so a claimable set that grew to
        the new pieces would hand the miner rows that are writable but router-masked to -inf
        (piece_loader.py:366-385) -- train forever, rejected forever. Residency is frozen at first
        resolution instead; the log line tells the operator to restart."""
        sd = _materialise_pieces(live_shape_shard_dir, [0])
        args = _default_args(sd)
        assert N.node_piece_ids(args) == [0]
        _materialise_pieces(sd, [0, 1, 2])                   # fetched while the node is running
        assert N.node_piece_ids(args) == [0]
        assert "RESTART" in N.fmt_piece_selection(args)

    def test_both_roles_agree_on_the_degraded_set(self, live_shape_shard_dir):
        """LOCKSTEP still holds after the filter: a miner that degraded to piece 0 while the
        coordinator held 0-11 would be told 'not hostable here' for coordinates it does hold."""
        C = pytest.importorskip("sharddiloco_glm_coordinator")
        args = _default_args(_materialise_pieces(live_shape_shard_dir, [0, 1]))
        assert C.N.node_piece_ids(args) == N.node_piece_ids(args) == [0, 1]
        assert C.N.node_claimable_coords(args) == N.node_claimable_coords(args)


class TestBothRolesResolveTheSameDefault:
    """THE LOCKSTEP GUARD. A miner that auto-expanded to 60 coordinates while the coordinator hosted 5
    would simply be told `not hostable here` for 55 of them, which looks like a miner bug and is not.
    The two roles cannot drift because there is exactly one implementation and the coordinator imports
    it -- these assertions are what keep it that way if someone later adds a local copy."""

    def test_the_coordinator_uses_this_module_s_resolver_object(self):
        C = pytest.importorskip("sharddiloco_glm_coordinator")
        assert C.N is N
        assert C.N.resolve_piece_selection is N.resolve_piece_selection
        assert C.N.default_piece_ids is N.default_piece_ids

    def test_both_roles_resolve_the_same_ids_and_coordinates(self, live_shape_shard_dir):
        C = pytest.importorskip("sharddiloco_glm_coordinator")
        args = _default_args(live_shape_shard_dir)
        assert C.N.node_piece_ids(args) == N.node_piece_ids(args) == list(range(12))
        assert C.N.node_claimable_coords(args) == N.node_claimable_coords(args)
        assert len(C.N.node_claimable_coords(args)) == 60

    def test_the_coordinator_startup_line_is_built_from_the_shared_formatter(self):
        """The log line both roles print is one function, so 'pieces_here=' cannot say one thing on
        the coordinator and another on the miner."""
        C = pytest.importorskip("sharddiloco_glm_coordinator")
        assert C.N.fmt_piece_selection is N.fmt_piece_selection


class TestTheDefaultIsVisibleAndOverridable:
    """An operator must be able to SEE the choice and the excluded straddlers at startup. Run 4 is the
    price of the opposite: no log line anywhere said 'your trainable universe is five coordinates'."""

    def test_the_startup_line_names_the_pieces_the_count_and_the_excluded_straddler(
            self, live_shape_shard_dir):
        txt = N.fmt_piece_selection(_default_args(live_shape_shard_dir))
        assert "12 piece(s)" in txt
        assert "DEFAULT (no --pieces/--piece given)" in txt
        assert "excluded straddler(s) 12" in txt and "ADD layer(s) 2" in txt

    def test_an_explicit_selection_says_so_instead(self, live_shape_shard_dir):
        assert "explicit --piece 0" in N.fmt_piece_selection(_piece_args(live_shape_shard_dir,
                                                                         piece=0))
        assert "explicit --pieces 0-3" in N.fmt_piece_selection(
            _piece_args(live_shape_shard_dir, pieces="0-3"))

    def test_a_refusal_message_does_not_invent_a_flag_the_operator_never_passed(
            self, live_shape_shard_dir):
        """Under the default there is no --piece on the command line, so the error must not tell the
        reader to fix one. (2,0) is outside the default set here, which is the point of the message."""
        with pytest.raises(SystemExit, match=r"the DEFAULT resident set"):
            N.resolve_claim(_default_args(live_shape_shard_dir, expert="2:0"),
                            N.parse_slots("1:0,1:1"), log=lambda *a: None)


class TestTheArgparseWiringActuallyDefaultsToNone:
    """The resolver can only see 'no flag given' if argparse stops substituting 0. This is the wiring
    test: without it the whole default is dead code behind an always-present --piece 0."""

    def test_no_piece_flag_parses_to_none_on_both_roles(self, monkeypatch):
        for var in ("NEURAHASH_GLM_PIECE", "NEURAHASH_GLM_PIECES"):
            monkeypatch.delenv(var, raising=False)
        ap = argparse.ArgumentParser()
        N.add_common_args(ap)
        args = ap.parse_args([])
        assert args.piece is None and args.pieces is None

    def test_the_env_var_still_pins_a_single_piece(self, monkeypatch):
        """NEURAHASH_GLM_PIECE is the same request typed elsewhere, so it counts as EXPLICIT and must
        keep its old five-coordinate meaning rather than becoming an anchor to widen from."""
        monkeypatch.delenv("NEURAHASH_GLM_PIECES", raising=False)
        monkeypatch.setenv("NEURAHASH_GLM_PIECE", "3")
        ap = argparse.ArgumentParser()
        N.add_common_args(ap)
        assert ap.parse_args([]).piece == 3

    def test_an_explicit_piece_on_the_command_line_survives(self, monkeypatch):
        for var in ("NEURAHASH_GLM_PIECE", "NEURAHASH_GLM_PIECES"):
            monkeypatch.delenv(var, raising=False)
        ap = argparse.ArgumentParser()
        N.add_common_args(ap)
        assert ap.parse_args(["--piece", "0"]).piece == 0


class TestTheDefaultDegradesInsteadOfCrashing:
    """fmt_piece_selection is printed by BOTH roles at startup, and a broken manifest already has one
    loud owner (node_claimable_coords -> 'cannot determine this node's claimable coordinates'). Two
    different crashes for one cause is worse than one, so resolution falls back to the anchor."""

    def test_tiny_mode_keeps_the_single_anchor_piece(self):
        args = types.SimpleNamespace(mode="tiny", shard_dir=None, config_dir=None,
                                     piece=None, pieces=None)
        assert N.node_piece_ids(args) == [N.DEFAULT_ANCHOR_PIECE]

    def test_a_missing_manifest_falls_back_and_says_why(self, tmp_path):
        args = types.SimpleNamespace(mode="glm", shard_dir=str(tmp_path), config_dir=None,
                                     piece=None, pieces=None)
        assert N.node_piece_ids(args) == [0]
        assert "anchor piece 0 only" in N.fmt_piece_selection(args)

    def test_a_namespace_predating_both_flags_still_means_unchecked(self):
        """The async lane's dirty-namespace path passes a partial namespace; it cannot express a
        selection, and [] is what keeps claims UNCHECKED exactly as before."""
        assert N.node_piece_ids(types.SimpleNamespace(mode="glm", shard_dir="x")) == []


class TestClaimableUnionAcrossPieces:
    """The actual unlock: claimable_here must be the UNION of the selected pieces' coordinates."""

    def test_multi_piece_yields_the_union_and_its_exact_size(self, shard_dir):
        """Assert the NUMBER, not just non-emptiness: the whole finding is that the count was 5 when
        the operator believed the space was unlimited. Pieces 0..2 hold 2 coordinates each = 6."""
        got = N.node_claimable_coords(_piece_args(shard_dir, pieces="0-2"))
        assert got == [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]
        assert len(got) == 6 == sum(2 for _ in range(3))

    def test_a_comma_list_matches_the_equivalent_range(self, shard_dir):
        assert (N.node_claimable_coords(_piece_args(shard_dir, pieces="0,1,2"))
                == N.node_claimable_coords(_piece_args(shard_dir, pieces="0-2")))

    def test_single_piece_behaviour_is_byte_for_byte_what_it_was(self, shard_dir):
        """The no-silent-change guard: --piece N must return exactly what piece_loader reports for
        [N] -- the same call the pre-multi-piece code made."""
        cfg = types.SimpleNamespace(num_hidden_layers=3, first_k_dense_replace=1)
        man = PL.load_manifest(shard_dir, require_files=False)
        for pid in (0, 1, 2):
            assert (N.node_claimable_coords(_piece_args(shard_dir, piece=pid))
                    == PL.claimable_expert_ids(man, [pid], cfg))
        assert N.node_claimable_coords(_piece_args(shard_dir, piece=0)) == [(1, 0), (1, 1)]

    def test_the_mtp_filter_still_applies_across_a_union(self, shard_dir):
        """Piece 3 is 100% MTP. Widening residency must not smuggle unreal coordinates in -- handing
        one to the lane host raised a naked 'index 3 is out of range' deep inside read_slot."""
        assert N.node_claimable_coords(_piece_args(shard_dir, pieces="3")) == []
        assert N.node_claimable_coords(_piece_args(shard_dir, pieces="0-3")) == \
            N.node_claimable_coords(_piece_args(shard_dir, pieces="0-2"))

    def test_a_piece_id_absent_from_the_manifest_fails_loudly(self, shard_dir):
        with pytest.raises(SystemExit, match=r"cannot determine this node's claimable coordinates"):
            N.node_claimable_coords(_piece_args(shard_dir, pieces="0-9"))


class TestClaimabilityGuardAcrossPieces:
    """The guard must WIDEN with residency and not one millimetre further: a coordinate in a selected
    piece becomes claimable, a coordinate in a NON-selected piece stays refused."""

    def test_a_coordinate_in_a_newly_resident_piece_is_claimable(self, shard_dir):
        slots = N.parse_slots("1:0,1:1")
        L, E, i, src = N.resolve_claim(_piece_args(shard_dir, pieces="0-2", expert="1:3"),
                                       slots, log=lambda *a: None)
        assert (L, E) == (1, 3) and src == "--expert"
        assert slots[i] == (1, 3)

    def test_the_same_coordinate_is_refused_under_the_old_single_piece(self, shard_dir):
        """(1,3) lives in piece 1. Under --piece 0 it must still be refused -- this is the assertion
        that the guard was widened by RESIDENCY and not simply weakened."""
        with pytest.raises(SystemExit, match=r"REFUSING to claim \(L1,E3\)"):
            N.resolve_claim(_piece_args(shard_dir, piece=0, expert="1:3"),
                            N.parse_slots("1:0,1:1"), log=lambda *a: None)

    def test_a_coordinate_outside_every_selected_piece_is_still_refused(self, shard_dir):
        """(2,0) is in piece 2; selecting only 0-1 must not make it claimable."""
        with pytest.raises(SystemExit, match=r"REFUSING to claim \(L2,E0\)"):
            N.resolve_claim(_piece_args(shard_dir, pieces="0-1", expert="2:0"),
                            N.parse_slots("1:0,1:1"), log=lambda *a: None)

    def test_the_refusal_names_the_flag_the_operator_actually_passed(self, shard_dir):
        with pytest.raises(SystemExit, match=r"--pieces 0-1 does not hold it"):
            N.resolve_claim(_piece_args(shard_dir, pieces="0-1", expert="2:0"),
                            N.parse_slots("1:0,1:1"), log=lambda *a: None)

    def test_an_all_mtp_selection_still_gets_its_own_message(self, shard_dir):
        with pytest.raises(SystemExit, match=r"--pieces 3 holds NO real experts"):
            N.resolve_claim(_piece_args(shard_dir, pieces="3", expert="1:0"),
                            N.parse_slots("1:0,1:1"), log=lambda *a: None)

    def test_claim_all_coords_sweeps_the_whole_union(self, shard_dir):
        """The plateau-advance path must see the widened universe, or the miner still cycles the same
        five coordinates forever -- the measured symptom this change exists to fix."""
        got = N.claim_all_coords(_piece_args(shard_dir, pieces="0-2"), N.parse_slots("1:0,1:1"))
        assert len(got) == 6 and (1, 3) in got and (2, 1) in got


class TestTheUnionFeedsTheRealRegistry:
    """Claimable is only half the story -- the lane host must actually REGISTER the widened
    coordinates, because the host is what reads and writes weights. Uses the real tiny GLM."""

    def test_a_coordinate_from_a_newly_selected_piece_registers_and_activates(self, host, shard_dir):
        G, model, cfg, _ = host
        claimable = N.node_claimable_coords(_piece_args(shard_dir, pieces="0-2"))
        h = G.GlmExpertLaneHost(model, cfg, [(1, 0), (1, 1)], claimable=claimable)
        assert h.index_of(1, 3) is None                      # unseen at startup
        i = h.register(1, 3)
        assert i == 2 and h.index_of(1, 3) == 2

    def test_the_narrow_single_piece_host_still_refuses_it(self, host, shard_dir):
        """Same model, same code path, narrower residency -> refused. The registry guard tracks
        residency rather than having been loosened."""
        G, model, cfg, _ = host
        claimable = N.node_claimable_coords(_piece_args(shard_dir, piece=0))
        h = G.GlmExpertLaneHost(model, cfg, [(1, 0), (1, 1)], claimable=claimable)
        with pytest.raises(ValueError):
            h.register(1, 3)


class TestResidencyAssertion:
    """A piece id names `experts_<id>`, NOT a position in the manifest's piece list (which also holds
    the trunk). An off-by-one there does not raise anywhere: the layer comes back partially resident
    and the missing rows are writable-but-inert. Measured on tools/glm_router_domain_probe.py as a
    layer silently left 59/64. This assertion is what turns that into a startup failure."""

    def test_a_match_passes_and_returns_the_count(self):
        assert N.check_residency(64, [(1, e) for e in range(64)], list(range(13))) == 64

    def test_a_short_load_is_fatal_and_says_which_pieces(self):
        with pytest.raises(SystemExit) as ex:
            N.check_residency(59, [(1, e) for e in range(64)], list(range(13)))
        msg = str(ex.value)
        assert "residency mismatch" in msg and "59 resident" in msg and "64 claimable" in msg

    def test_an_over_long_load_is_equally_fatal(self):
        """Too MANY resident experts means the ids resolved to different pieces than requested --
        just as wrong, and it would quietly change what every node routes over (plan risk 5)."""
        with pytest.raises(SystemExit, match=r"residency mismatch"):
            N.check_residency(70, [(1, e) for e in range(64)], list(range(13)))

    def test_unchecked_residency_stays_a_no_op(self):
        """tiny mode / no manifest -> claimable is None -> no assertion, exactly as before."""
        assert N.check_residency(5, None, [0]) is None


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
        # V0.1: event 2 advertises 2_3, which cannot move OUR coordinate, so it is skipped instead of
        # folded-and-rolled-back. 2 folds, not 3 -- the reached target below is what this test asserts.
        assert applied == 2
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


def _publish_claim(env, coord, miner, base_event=0, wire_idx=0, host=None, seed=0, key=_LKEY,
                   slot_root=None):
    """Publish ONE contribution addressed by COORDINATE, the way a shard-claim miner does. `host` is the
    MINER's host (its own slot list, hence its own local index) -- deliberately not the coordinator's.

    `key` is the HMAC key the record is SIGNED with; pass anything other than _LKEY (or None for no
    signature at all) to publish exactly what an unauthenticated party can PUT on this shared-token lane.

    `slot_root` OVERRIDES base_slot_root, which is how a lineage-DEAD record is modelled: that field is
    UNSIGNED (the HMAC covers cid + base_event + miner only), so a dead run's leftover -- or anyone
    holding the public PUT token -- can carry a root this coordinator never produced."""
    h = env["host"] if host is None else host
    idx = h.index_of(*coord)
    ref = h.read_slot(idx if idx is not None else 0)
    rng = np.random.default_rng(seed)
    payload = {k: (rng.standard_normal(v.shape) * 1e-3).astype(np.float32) for k, v in ref.items()}
    cid = env["store"].put_delta(payload)
    rec = N.build_async_contrib_record(
        miner, wire_idx, coord[0], coord[1], base_event, N.model_root(h), cid,
        (_H.sign(key, cid, base_event, miner) if key is not None else None), 1e9,
        int(len(_H.pack_arrays(payload, np.float16))), 10, 160,
        base_slot_root=(slot_root if slot_root is not None
                        else (N.slot_root(h, idx) if idx is not None else None)))
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
        # V0.1: ONE fold, not two -- event 1 moves (1,0) and a slot root digests only its own slot, so
        # skipping it reaches (1,1)'s advertised root just the same. `reached` is the claim here.
        assert (applied, reached) == (1, True)
        assert N.slot_root(host, 1) == r1, "the freshly claimed coordinate is on the coordinator's base"

    def test_the_advance_branch_is_actually_wired_to_it(self):
        """Wiring: the mechanism already existed (resume_to_root own_coord=) -- the defect was that the
        PLATEAU/advance branch never called it.

        The register->catch-up pair moved into N.advance_claim with never-block V0 (the walk has to be
        able to try more than one candidate), so the ORDER is asserted there; _run_async is checked for
        calling it and for still clearing last_pub_base_event on the coordinate it lands on."""
        import inspect
        walk = inspect.getsource(N.advance_claim)
        i_reg = walk.index("host.register(*cand)")
        i_res = walk.index("resume_to_root(host, lane, pointer_root, log, own_coord=cand")
        assert i_res > i_reg, "the catch-up must run AFTER the new coordinate is registered"
        src = inspect.getsource(N._run_async)
        assert "advance_claim(host, lane, claim_coords, (L, E), claim_identity" in src
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


# ===================================================== v3.4.1 FIX A: discovery cost on a long-lived store
class _CountingLane:
    """The ContentLane surface the async loop's DISCOVER pass touches, with CALL COUNTERS.

    Counting is the whole point: on the live store (11,051 objects, 13.6 GB) `manifest()` was MEASURED at
    23.79 s against `get_json()` at 0.06 s, so "how many times per pass" is a first-order cost, not a
    detail. Nothing here fakes the loop -- run_async_events is driven for real."""

    def __init__(self, records):
        self.names, self._recs = {}, {}
        for name, rec in sorted(records.items()):
            sha = "sha-" + name.replace("/", "-")
            self.names[name] = {"sha256": sha}
            self._recs[sha] = rec
        self.manifest_calls = 0
        self.fetched = []                       # sha of every get_json, in order
        self.written = []                       # named writes (genesis + terminal pointer)

    def manifest(self):
        self.manifest_calls += 1
        return dict(self.names)

    def get_json(self, sha):
        self.fetched.append(sha)
        return dict(self._recs[sha])

    def put_json_named(self, name, obj):
        self.written.append(name)
        return "cid%d" % len(self.written)


class _RegistryHost(_FoldHost):
    """_FoldHost plus the slot-registry surface run_async_events touches BEFORE any merge (register /
    index_of / is_active / evict / active / claimable_coords). No torch, no GLM: these tests never let a
    slot become ready, so the merge path -- the only part that needs a real model -- is never entered."""

    def __init__(self, coords, shape=(2, 3)):
        _FoldHost.__init__(self, coords, shape=shape)
        self._shape = shape
        self.active = set(range(len(self.slots)))
        self.max_active = None

    def register(self, L, E):
        i = self.index_of(L, E)
        if i is None:
            self.slots.append((int(L), int(E)))
            i = len(self.slots) - 1
            self._w[i] = {k: np.zeros(self._shape, np.float32) for k in ("gate", "up", "down")}
        self.active.add(int(i))
        return int(i)

    def is_active(self, i):
        return int(i) in self.active

    def evict(self, i):
        had = int(i) in self.active
        self.active.discard(int(i))
        return had

    def claimable_coords(self):
        return None


class _PooledProbe:
    """Every startup slot already HAS its secret gate pool, so _admit_coordinate takes the
    known-coordinate path and never builds one (pool construction needs the real corpus)."""

    def __init__(self):
        self.ensure_calls = []

    def has_pool(self, i):
        return True

    def ensure_pool(self, i, pools):
        self.ensure_calls.append(int(i))


_A_COORD = (1, 0)                       # the ONE coordinate these tests declare at startup


def _contrib_rec(base_event, miner, coord=_A_COORD):
    """A contribution record as the wire carries it: addressed by COORDINATE, dated by base_event."""
    return dict(miner=miner, expert=0, layer=int(coord[0]), glm_expert=int(coord[1]),
                base_event=int(base_event), base_root="g0", base_slot_root="s0",
                expert_cid="c-%s" % miner, sig="00" * 32, steps=10, tokens=160,
                train_flops=1e9, delta_bytes=1024)


def _loop_args(rounds=1):
    a = types.SimpleNamespace()
    a.rounds = rounds
    a.poll_timeout = 5.0
    a.mode = "tiny"
    a.outer = 0.7
    a.margin = -1e9
    a.merge_tol = 1e9
    a.probe_size = 8
    a.eval_chunk = 64
    a.slots = "1:0"
    a.domains = "code,gutenberg"
    a.threads = 2
    a.max_active_slots = 8
    return a


@coordinator_only
class TestFixAManifestIsNotReReadEveryPass:
    """FIX A (v3.4.1). run_async_events called `lane.manifest()` at the TOP of every iteration. MEASURED
    2026-07-25 against the live store: manifest() = 23.79 s, get_json() = 0.06 s -- so the loop was capped
    near 2.5 iterations per MINUTE no matter what NEURAHASH_SD_POLL_S said (it logged poll=1.5s), it
    degrades linearly forever because the store never deletes, and a fresh --no-resume coordinator spent
    ~20 minutes without judging any of a healthy miner's 7 valid contributions. On a CLEAN store (manifest
    0.014 s) the same campaign accepted within ~2 minutes.

    Method: drive the REAL loop (the same function the live campaign runs) with a grace window nothing can
    satisfy, so exactly one record is DISCOVERED and stays PENDING -- the loop keeps spinning with work in
    hand, which is precisely the state in which re-reading the manifest is pure waste. Pass count comes
    from _collect_unprocessed, called once per pass past the manifest step."""

    @staticmethod
    def _env(monkeypatch, refresh="1e9", idle="0.3"):
        monkeypatch.setenv("NEURAHASH_SD_MANIFEST_REFRESH_S", refresh)
        monkeypatch.setenv("NEURAHASH_SD_IDLE_EXIT_S", idle)
        monkeypatch.setenv("NEURAHASH_SD_POLL_S", "0.001")
        monkeypatch.setenv("NEURAHASH_SD_GRACE_S", "300")     # nothing EVER becomes ready -> no merge
        monkeypatch.setenv("NEURAHASH_SD_QUORUM_K", "1")
        monkeypatch.setenv("NEURAHASH_SD_IDLE_EVICT_EVENTS", "0")
        monkeypatch.setenv("NEURAHASH_GLM_OPEN_ADMISSION", "0")   # no registry file read in a unit test
        monkeypatch.setenv("NEURAHASH_GLM_QUORUM", "0")
        monkeypatch.delenv("NEURAHASH_SHARDDILOCO_MAX_STALENESS", raising=False)

    @staticmethod
    def _drive(monkeypatch, records):
        import sharddiloco_glm_coordinator as C
        lane = _CountingLane(records)
        host = _RegistryHost([_A_COORD])
        passes = {"n": 0}
        _real = C._collect_unprocessed

        def _counting(*a, **kw):
            passes["n"] += 1
            return _real(*a, **kw)

        monkeypatch.setattr(C, "_collect_unprocessed", _counting)
        logs = []
        rc = C.run_async_events(None, None, host, lane, _PooledProbe(),
                                types.SimpleNamespace(verify=0.0, fwd=1.0), None, [], host.slots,
                                {}, _loop_args(), 1.0,
                                lambda *a: logs.append(" ".join(str(x) for x in a)))
        return rc, lane, passes["n"], logs

    def test_the_manifest_is_read_twice_not_once_per_pass(self, monkeypatch):
        """The claim, exactly: many passes, TWO manifest reads -- one at entry, one forced before the idle
        exit. Pre-fix this was one read PER PASS (manifest_calls == passes)."""
        self._env(monkeypatch)
        rc, lane, passes, logs = self._drive(monkeypatch, {N.contrib_name(0, "m1"): _contrib_rec(0, "m1")})
        assert rc == 0
        assert passes >= 5, "the loop must really have spun many times (got %d)" % passes
        assert lane.manifest_calls == 2, \
            "expected 1 entry read + 1 forced pre-exit read, got %d over %d passes" % (
                lane.manifest_calls, passes)
        assert any("ASYNC idle" in ln for ln in logs), "\n".join(logs[-5:])

    def test_the_idle_exit_is_never_taken_on_a_stale_manifest(self, monkeypatch):
        """The guard means "no new contribution EXISTS"; a manifest we chose not to re-read cannot support
        that claim. So the second read above is not incidental -- it happens BEFORE the idle exit, with
        pending work in hand and the refresh window (1e9 s) nowhere near elapsed."""
        self._env(monkeypatch)
        _rc, lane, _p, logs = self._drive(monkeypatch, {N.contrib_name(0, "m1"): _contrib_rec(0, "m1")})
        assert lane.manifest_calls == 2 and any("ASYNC idle" in ln for ln in logs)

    def test_a_zero_refresh_window_still_reads_it_every_pass(self, monkeypatch):
        """The knob is live in BOTH directions: NEURAHASH_SD_MANIFEST_REFRESH_S=0 restores the old
        read-every-pass cadence, so the caching is a policy, not a hard-coded skip."""
        self._env(monkeypatch, refresh="0")
        _rc, lane, passes, _logs = self._drive(monkeypatch,
                                               {N.contrib_name(0, "m1"): _contrib_rec(0, "m1")})
        assert lane.manifest_calls == passes >= 5

    def test_a_future_base_event_name_is_never_fetched(self, monkeypatch):
        """FIX A part 2. `_collect_unprocessed` returned EVERY unseen parseable name (3,915 of them on the
        live store) and the loop paid one get_json for each before _lineage_ok dropped it as
        `future-base-event` -- e.g. `LINEAGE-DROP cg/r1/glm-E2223497.4 base_event=1`. The NAME already
        carries the base event (N.CONTRIB_PREFIX_FMT), so cur_event=0 can reject cg/r7/... for free."""
        self._env(monkeypatch)
        recs = {N.contrib_name(0, "m1"): _contrib_rec(0, "m1"),
                N.contrib_name(7, "m2"): _contrib_rec(7, "m2")}
        _rc, lane, _p, logs = self._drive(monkeypatch, recs)
        assert lane.fetched == ["sha-cg-r0-m1"], \
            "the r7 record must never be fetched at event 0; fetched=%r" % (lane.fetched,)
        assert not any("future-base-event" in ln for ln in logs), "it should not even reach the gate"

    def test_the_skipped_name_is_not_remembered_and_is_picked_up_when_the_clock_reaches_it(self):
        """The skip must be a "not yet", not a rejection. Reproduces the loop's own discovery contract
        (it adds ONLY returned names to its run-long `seen` set) and shows the same record being collected
        once cur_event catches up -- so a record for base_event 5 is still usable at event 5."""
        import sharddiloco_glm_coordinator as C
        names = [N.contrib_name(0, "m1"), N.contrib_name(5, "m2")]
        seen = set()
        for name, _be, _m in C._collect_unprocessed(names, seen, C._parse_contrib_name, max_base_event=0):
            seen.add(name)                                   # exactly what run_async_events does
        assert seen == {N.contrib_name(0, "m1")}
        assert N.contrib_name(5, "m2") not in seen, "a future record must NOT be permanently ignored"
        later = C._collect_unprocessed(names, seen, C._parse_contrib_name, max_base_event=5)
        assert [n for n, _b, _m in later] == [N.contrib_name(5, "m2")]

    def test_the_prefilter_is_off_by_default_so_other_callers_do_not_change(self):
        import sharddiloco_glm_coordinator as C
        names = [N.contrib_name(0, "a"), N.contrib_name(9, "b"), "glm/pointer"]
        assert [b for _n, b, _m in C._collect_unprocessed(names, set(), C._parse_contrib_name)] == [0, 9]

    def test_the_loop_scopes_discovery_to_the_current_event_and_documents_the_knob(self):
        """Wiring: the helpers are worthless if the loop does not pass clock.event, and an operator who
        cannot see the refresh cadence in the startup line will spend another 20 minutes guessing."""
        import inspect
        import sharddiloco_glm_coordinator as C
        src = inspect.getsource(C.run_async_events)
        assert "max_base_event=clock.event" in src
        assert "NEURAHASH_SD_MANIFEST_REFRESH_S" in src, "the startup log must name the knob"
        assert C._manifest_refresh_s({}) == 30.0
        assert C._manifest_refresh_s({"NEURAHASH_SD_MANIFEST_REFRESH_S": "5"}) == 5.0


# ======================================================== v3.4.1 FIX B: the sweep order is per-identity
class TestFixBSweepOrderIsPerIdentity:
    """FIX B (v3.4.1). pick_start_coord spread miners by wallet hash, but next_claim_coord advanced
    EVERYONE by +1 through the same sorted list, so a one-off collision became permanent. OBSERVED LIVE
    2026-07-25: the 5090 (glm-ea20C873) swept 1:1 -> 1:2 -> 1:3 -> 1:4 into the 4060 (glm-361447E3)
    parked on 1:2; they shared 1:2 for events 12-15, every one a reject, held-out CE frozen at 7.76966,
    while the coordinates neither miner reached starved."""

    _C = [(1, e) for e in range(7)]
    _IDS = ["0x%040x" % k for k in range(40)]

    @classmethod
    def _walk(cls, identity, start, coords=None):
        """Follow next_claim_coord from `start` for len(coords)-1 steps -- one full sweep."""
        coords = cls._C if coords is None else coords
        out, cur = [], tuple(start)
        for _ in range(len(coords) - 1):
            cur = N.next_claim_coord(coords, cur, identity=identity)
            out.append(cur)
        return out

    def test_two_identities_walk_the_same_set_in_different_orders(self):
        walks = {tuple(self._walk(i, (1, 0))) for i in self._IDS}
        assert len(walks) >= 3, "40 identities produced only %d distinct sweeps" % len(walks)

    def test_a_collision_does_not_persist_the_way_it_did_live(self):
        """The measured failure, as an assertion. Legacy (+1, identity=None): two miners meeting on 1:2
        advance to the SAME next coordinate -- locked. Per-identity: the successors of 1:2 spread out."""
        assert N.next_claim_coord(self._C, (1, 2)) == N.next_claim_coord(self._C, (1, 2)) == (1, 3)
        nxt = {N.next_claim_coord(self._C, (1, 2), identity=i) for i in self._IDS}
        assert len(nxt) >= 3, "successors of a shared coordinate must spread, got %r" % (nxt,)

    def test_every_identity_still_sweeps_the_whole_claimable_set(self):
        """A different order must not mean a smaller orbit: the walk is a single cycle over a permutation,
        so from ANY start it visits every other coordinate exactly once before repeating."""
        for i in self._IDS[:12]:
            for start in (self._C[0], self._C[3], self._C[-1]):
                w = self._walk(i, start)
                assert len(set(w)) == len(w) == len(self._C) - 1, (i, start, w)
                assert set(w) | {tuple(start)} == set(self._C), (i, start, w)

    def test_the_walk_is_deterministic_per_identity(self):
        """A restarted miner must behave predictably -- no PID, no clock, no randomness in the order."""
        for i in self._IDS[:5]:
            assert self._walk(i, (1, 0)) == self._walk(i, (1, 0))
            assert N.claim_walk_order(self._C, i) == N.claim_walk_order(list(self._C), i)

    def test_it_never_returns_the_coordinate_we_are_on(self):
        """Advancing onto our own coordinate would reload the data shard and change nothing."""
        for i in self._IDS:
            for cur in self._C:
                assert N.next_claim_coord(self._C, cur, identity=i) != tuple(cur)

    def test_a_walk_order_is_a_permutation_not_a_subset(self):
        for i in self._IDS[:8]:
            assert sorted(N.claim_walk_order(self._C, i)) == sorted(tuple(c) for c in self._C)

    def test_no_identity_keeps_the_legacy_shared_sweep_exactly(self):
        """Backward compatibility is load-bearing: the 2-argument call is what every pre-3.4.1 caller and
        test uses, and it must stay the plain +1 cycle."""
        assert N.claim_walk_order(self._C) == self._C
        assert N.next_claim_coord(self._C, (1, 0)) == (1, 1)
        assert N.next_claim_coord([(1, 0)], (1, 0), identity="0xabc") is None
        assert N.next_claim_coord([], (1, 0), identity="0xabc") is None
        assert N.next_claim_coord([(1, 0), (1, 1)], (9, 9), identity=None) == (1, 0)

    def test_a_coordinate_outside_the_set_recovers_into_this_identitys_order(self):
        got = N.next_claim_coord(self._C, (9, 9), identity="0xabc")
        assert got == N.claim_walk_order(self._C, "0xabc")[0]

    def test_the_contributor_loop_advances_with_its_wallet_identity(self):
        """Wiring: a per-identity order that the sweep does not pass its identity to is decoration.

        Never-block V0 replaced the single next_claim_coord() step with advance_claim(), which walks
        the SAME claim_walk_order permutation (so it can skip a coordinate on cooldown instead of
        stopping at it). The property guarded is unchanged: the sweep passes its own identity, and
        that identity is the durable wallet address."""
        import inspect
        src = inspect.getsource(N._run_async)
        assert "advance_claim(host, lane, claim_coords, (L, E), claim_identity, claim_ranked" in src
        walk = inspect.getsource(N.advance_claim)
        assert "claim_walk_order(claim_coords, identity, ranked=ranked)" in walk
        assert "claim_identity = wallet.address if wallet is not None else" in src, \
            "the identity must be the DURABLE wallet address, so a restart walks the same order"


# ============================================================ the JOIN defaults a stranger inherits
class TestJoinDefaultsAreReachable:
    """Unlimited claimable coordinates are worthless if the client cannot reach the lane at all.

    Audited 2026-07-25 against the fresh public clone at v3.4.0: no flag is required=True, so the
    client STARTS for a stranger -- and then talks to nobody, because `--url` defaulted to
    http://127.0.0.1:8797 (the stranger's OWN box, closed port) and `--token` to "" (every PUT 401).
    The published Mine snippet was therefore uncopyable by anyone but the author, which contradicts
    the standing directive (memory public-testing-unlimited-slots-directive: public testing,
    UNLIMITED slots, anyone may join).

    Both values CAN ship as defaults because neither is a secret: the lane is the public anchor
    store and the token is its PUBLIC DEMO token, already the shipped `--token` default of the
    esh_worker client (public commit 1fdcd5a) and spam-open by design -- rotating it would break
    every running miner (memory content-token-is-public-demo-token-2026-07-21). These tests are the
    guard against a future session "hardening" either one back into a placeholder.
    """

    @staticmethod
    def _is_unreachable_host(host):
        """True for hosts that only ever resolve to the machine running the client."""
        if host.lower() in ("localhost", "localhost.localdomain", ""):
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False                     # a real hostname; DNS is not this test's business
        return ip.is_loopback or ip.is_unspecified

    @staticmethod
    def _join_defaults(monkeypatch):
        """Defaults straight from the module's OWN argparse setup, with the env cleared.

        argparse captures os.environ at add_argument() time, so clearing first is what makes this a
        test of the CODE's defaults rather than of whatever this machine exports."""
        for k in ("NEURAHASH_CONTENT_URL", "NEURAHASH_CONTENT_TOKEN"):
            monkeypatch.delenv(k, raising=False)
        return N.add_common_args(argparse.ArgumentParser()).parse_args([])

    def test_the_url_default_is_not_a_loopback_address(self, monkeypatch):
        url = self._join_defaults(monkeypatch).url
        assert url.startswith("http://") or url.startswith("https://"), url
        host = urllib.parse.urlsplit(url).hostname or ""
        assert not self._is_unreachable_host(host), (
            "--url default %r resolves to the miner's own box: a stranger copy-pasting the Mine "
            "snippet joins nothing" % url)

    def test_the_token_default_is_not_empty(self, monkeypatch):
        tok = self._join_defaults(monkeypatch).token
        assert tok.strip(), ("--token default is empty: every publish 401s. The lane token is "
                             "public by design, so the default can be the real one")
        assert len(tok.strip()) >= 16, tok

    def test_the_environment_still_wins_over_both_defaults(self, monkeypatch):
        """The live campaign and any private lane are driven by these env vars -- defaulting the
        public lane must not take that override away."""
        monkeypatch.setenv("NEURAHASH_CONTENT_URL", "http://10.0.0.5:8797")
        monkeypatch.setenv("NEURAHASH_CONTENT_TOKEN", "private-lane-token-0123456789")
        ns = N.add_common_args(argparse.ArgumentParser()).parse_args([])
        assert ns.url == "http://10.0.0.5:8797"
        assert ns.token == "private-lane-token-0123456789"
        cli = N.add_common_args(argparse.ArgumentParser()).parse_args(
            ["--url", "http://127.0.0.1:8797", "--token", "x"])
        assert (cli.url, cli.token) == ("http://127.0.0.1:8797", "x")    # explicit flags still win


# ============================================ v3.4.2 FIX A: registration is gated on a VERIFIED signature
@coordinator_only
class TestFixARegistrationRequiresAVerifiedSignature:
    """FIX A (SECURITY, 2026-07-25). `_admit_coordinate` ran inside the DISCOVER pass, so a coordinate was
    REGISTERED -- seat under --max-active-slots taken, secret gate pool built, ~75.5 MB of fp32 slot
    materialized, lineage root seeded -- BEFORE `_fetch_validate_contribs` had checked a single signature.

    Why that is reachable and not theoretical: the lane's PUT token is a deliberately PUBLIC demo token
    (memory content-token-is-public-demo-token-2026-07-21) and the client now DEFAULTS it, so "anyone holding
    the token" means any user of the client. One unsigned PUT per coordinate was enough to make the
    coordinator register every claimable coordinate it holds and exhaust the active set with records that can
    never merge -- honest claimants then DEFER forever. Resource occupation, not weight corruption, but
    trivially reachable. The fix: DISCOVER resolves the coordinate PURELY (_record_coordinate) and queues by
    it; registration happens in the merge path only after _verify_record_identity authenticates a record."""

    _CLAIM = (2, 3)                      # unseen at startup -> registering it IS the state change we watch
    _OTHER_KEY = b"\x77" * 16            # a genuine HMAC over the right message, with the WRONG key

    @pytest.fixture(autouse=True)
    def _fast_idle(self, monkeypatch):
        """These tests END with the loop waiting for work that will never come (that is the point: the
        unverified record must never merge), so the 600 s default idle guard would hang the suite. Set it
        INSIDE the class so the file stays fast standalone and never depends on the caller's environment."""
        monkeypatch.setenv("NEURAHASH_SD_IDLE_EXIT_S", "1")
        monkeypatch.setenv("NEURAHASH_SD_POLL_S", "0.05")

    def test_a_bad_signature_registers_nothing(self, loop_model, store_harness):
        """RED before the fix: host.index_of(2,3) == 1, probe.has_pool(1) True, len(host.slots) == 2."""
        env = _loop_env(loop_model, store_harness, [(1, 0)], rounds=1, max_active=8)
        host, G = env["host"], loop_model[0]
        mh = G.GlmExpertLaneHost(loop_model[1], loop_model[2], [(1, 0)], claimable=env["claimable"])
        mh.register(*self._CLAIM)
        _publish_claim(env, self._CLAIM, "attacker", wire_idx=1, host=mh, seed=3, key=self._OTHER_KEY)

        logs = _drive_loop(env, ["attacker"])           # rostered under _LKEY; the record is signed wrongly
        assert host.index_of(*self._CLAIM) is None, \
            "an unverified record REGISTERED a coordinate:\n" + "\n".join(logs[-12:])
        assert len(host.slots) == 1 and host.active == {0}, "a seat was taken by unauthenticated input"
        assert env["probe"].has_pool(1) is False, "a secret gate pool was built for an unverified record"
        assert env["store"].accepted(1) is None, "an unverified record committed an event"
        assert any("UNVERIFIED" in ln for ln in logs), \
            "the drop must be VISIBLE -- a silent drop is what made this look safe:\n" + "\n".join(logs[-8:])

    def test_an_unsigned_record_registers_nothing_either(self, loop_model, store_harness):
        """No `sig` field at all -- the cheapest possible PUT, and the one that used to cost a whole seat."""
        env = _loop_env(loop_model, store_harness, [(1, 0)], rounds=1, max_active=8)
        host, G = env["host"], loop_model[0]
        mh = G.GlmExpertLaneHost(loop_model[1], loop_model[2], [(1, 0)], claimable=env["claimable"])
        mh.register(*self._CLAIM)
        _publish_claim(env, self._CLAIM, "nosig", wire_idx=1, host=mh, seed=4, key=None)
        logs = _drive_loop(env, ["nosig"])
        assert host.index_of(*self._CLAIM) is None, "\n".join(logs[-12:])
        assert env["probe"].has_pool(1) is False

    def test_the_cap_cannot_be_exhausted_by_unsigned_records(self, loop_model, store_harness):
        """The squat, end to end: ONE free seat, three unsigned claims on DIFFERENT coordinates, then an
        honest signed claim. Pre-fix the first unsigned record took the seat and the honest miner was
        DEFERRED (`max_active_slots=2 reached; cannot admit`) with nothing ever merged."""
        env = _loop_env(loop_model, store_harness, [(1, 0)], rounds=1, max_active=2)
        host, G = env["host"], loop_model[0]
        mh = G.GlmExpertLaneHost(loop_model[1], loop_model[2], [(1, 0)], claimable=env["claimable"])
        for n, sq in enumerate(((2, 0), (2, 1), (2, 2))):
            mh.register(*sq)
            _publish_claim(env, sq, "squat%d" % n, wire_idx=1, host=mh, seed=20 + n, key=self._OTHER_KEY)
        mh.register(*self._CLAIM)
        _publish_claim(env, self._CLAIM, "honest", wire_idx=1, host=mh, seed=9)

        logs = _drive_loop(env, ["squat0", "squat1", "squat2", "honest"])
        assert [host.index_of(*c) for c in ((2, 0), (2, 1), (2, 2))] == [None, None, None], \
            "unsigned claims still occupy seats:\n" + "\n".join(logs[-14:])
        assert host.index_of(*self._CLAIM) == 1, \
            "the honest claimant did not get the only free seat:\n" + "\n".join(logs[-14:])
        rec = env["store"].accepted(1)
        assert rec is not None and rec["accepted"], "\n".join(logs[-14:])
        assert (rec["accepted"][0]["layer"], rec["accepted"][0]["glm_expert"]) == self._CLAIM

    def test_a_verified_record_still_registers_and_merges_on_OUR_index(self, loop_model, store_harness):
        """The complement, so the fix cannot pass by refusing everything: the SAME coordinate with a
        correctly signed record registers, gets its gate pool, and is merged into the COORDINATOR's own
        registry index -- never the miner's wire index (F1)."""
        env = _loop_env(loop_model, store_harness, [(1, 0)], rounds=1, max_active=8)
        host, G = env["host"], loop_model[0]
        mh = G.GlmExpertLaneHost(loop_model[1], loop_model[2], [(1, 0)], claimable=env["claimable"])
        mh.register(*self._CLAIM)
        _publish_claim(env, self._CLAIM, "honest", wire_idx=99, host=mh, seed=5)   # nonsense wire index

        logs = _drive_loop(env, ["honest"])
        idx = host.index_of(*self._CLAIM)
        assert idx == 1, "a VERIFIED claim was not registered:\n" + "\n".join(logs[-12:])
        assert host.is_active(idx) and env["probe"].has_pool(idx)
        rec = env["store"].accepted(1)
        assert rec is not None and rec["accepted"], "\n".join(logs[-12:])
        row = rec["accepted"][0]
        assert row["slot"] == idx, "merged on the WIRE index %r, not ours %r" % (row["slot"], idx)
        assert (row["layer"], row["glm_expert"]) == self._CLAIM
        assert rec["slot_roots"]["2_3"] == N.slot_root(host, idx)

    def test_an_over_capacity_honest_claim_is_deferred_and_does_not_starve_anyone(
            self, loop_model, store_harness):
        """The constraint the restructure had to preserve, now that DEFER happens with the record already
        popped off its queue: an over-capacity claim must be put BACK (retried), never silently dropped --
        miner-side a dropped claim is indistinguishable from a rejected delta. And because a deferred
        coordinate stays queued, it must not be re-selected forever: its arrival stamp is demoted so the
        coordinates that CAN merge still get their events."""
        env = _loop_env(loop_model, store_harness, [(1, 0)], rounds=3, max_active=1)   # zero free seats
        host, G = env["host"], loop_model[0]
        mh = G.GlmExpertLaneHost(loop_model[1], loop_model[2], [(1, 0)], claimable=env["claimable"])
        # The waiting claim is revealed FIRST (the harness reveals one contribution per poll), so it is in
        # hand for every pass -- exactly the state that used to spin the loop or starve the others.
        mh.register(*self._CLAIM)
        _publish_claim(env, self._CLAIM, "zz-waiting", wire_idx=1, host=mh, seed=7)   # honest, no seat free
        for n, m in enumerate(("h0", "h1", "h2")):
            _publish_claim(env, (1, 0), m, wire_idx=0, seed=40 + n)      # mergeable on the taken seat

        logs = _drive_loop(env, ["h0", "h1", "h2", "zz-waiting"])
        defers = [ln for ln in logs if "DEFER zz-waiting" in ln]
        assert len(defers) >= 2, ("the deferred claim was dropped instead of retried (%d DEFER lines):\n"
                                  % len(defers)) + "\n".join(logs[-14:])
        assert host.index_of(*self._CLAIM) is None, "the cap was breached"
        assert [env["store"].accepted(e) is not None for e in (1, 2, 3)] == [True] * 3, \
            "a deferred coordinate starved the one that could merge:\n" + "\n".join(logs[-14:])
        assert not any("zz-waiting" in (row.get("miner") or "")
                       for e in (1, 2, 3) for row in (env["store"].accepted(e)["accepted"] or [])), \
            "an unadmitted claim was paid"

    def test_discovery_itself_mutates_nothing(self):
        """_record_coordinate is the DISCOVER-side resolver and its whole value is that it is PURE: it must
        not register, append, activate or seed anything, for either record shape."""
        C, host, slots, probe, _clock, srh, _args = TestAdmitCoordinate._setup()
        before = (list(host.slots), set(host.active), list(slots), dict(srh))
        assert C._record_coordinate(host, slots, {"layer": 2, "glm_expert": 3}, "m",
                                    lambda *_: None) == (2, 3)
        assert C._record_coordinate(host, slots, {"expert": 1}, "old", lambda *_: None) == (1, 1)
        assert (list(host.slots), set(host.active), list(slots), dict(srh)) == before
        assert host.index_of(2, 3) is None and probe.has_pool(2) is False

    def test_the_wire_index_never_becomes_the_pending_key(self):
        """A record claiming (1,1) with a bogus wire index 99 must be filed under the COORDINATE, so two
        miners that both publish local index 1 cannot collide in the queue -- the F1 collision, moved one
        step earlier now that discovery is what keys the queue."""
        C, host, slots, _probe, _clock, _srh, _args = TestAdmitCoordinate._setup()
        a = C._record_coordinate(host, slots, {"layer": 1, "glm_expert": 1, "expert": 99}, "a",
                                 lambda *_: None)
        b = C._record_coordinate(host, slots, {"layer": 2, "glm_expert": 3, "expert": 99}, "b",
                                 lambda *_: None)
        assert a == (1, 1) and b == (2, 3) and a != b

    def test_identity_verification_precedes_registration_in_the_loop(self):
        """Wiring regression, and the only cheap guard against this hole coming back: in run_async_events
        the authentication call must appear BEFORE the registration call."""
        import inspect
        import sharddiloco_glm_coordinator as C
        src = inspect.getsource(C.run_async_events)
        assert "_record_coordinate(" in src, "DISCOVER must resolve the coordinate purely"
        i_verify, i_admit = src.find("_verify_record_identity("), src.find("_admit_coordinate(")
        assert i_verify > 0 and i_admit > 0, (i_verify, i_admit)
        assert i_verify < i_admit, "registration is back ahead of the signature check"

    def test_unverified_is_a_drop_not_a_verify_ok_false_merge(self):
        """_verify_record_identity's outcomes, which its two callers read differently: a rostered bad HMAC
        keeps the (False, miner) shape so _fetch_validate_contribs still hands it to the gate exactly as
        before, while a keyless failure is (False, None) = drop outright. run_async_events treats BOTH as
        "must not register". A missing sig must never raise: the lane's PUT token is public, so a rostered
        NAME with no `sig` field is a record a stranger can write."""
        import sharddiloco_glm_coordinator as C
        rec = {"sig": _H.sign(_LKEY, "cid", 0, "m1"), "expert_cid": "cid"}
        assert C._verify_record_identity(rec, "m1", 0, {"m1": {"key": _LKEY}}, None) == (True, "m1")
        assert C._verify_record_identity(rec, "m1", 0, {"m1": {"key": b"\x01" * 16}}, None) \
            == (False, "m1")
        assert C._verify_record_identity({}, "m1", 0, {"m1": {"key": _LKEY}}, None) == (False, "m1")
        assert C._verify_record_identity({}, "stranger", 0, {}, None) == (False, "stranger")


# ================================================ v3.4.2 FIX B: --domains is cross-checked between roles
@coordinator_only
class TestFixBDomainListsAreCrossChecked:
    """FIX B (2026-07-25). `N.coord_data_slot(L,E)` returns E on both sides, but the DOMAIN is then
    `doms[E % len(doms)]` and `--domains` was a PER-PROCESS flag with nothing cross-verifying it.

    Coordinator `code,gutenberg` with a miner on `code,gutenberg,web` and a claim on E=2 gives probe pool
    "code" against train shard "web": every delta is gated on text it never trained on and rejected
    SYSTEMATICALLY WITH NO ERROR ANYWHERE -- the silent-failure class this project keeps getting bitten by
    (memory pouw-verified-not-useful). The coordinator now publishes a digest of its effective list on the v2
    pointer and a contributor refuses to start on a mismatch; an absent digest (pre-Shard-Claim peer) is
    logged once and still starts. Comparison is on the MAPPING, not the spelling, so the live pair
    (coordinator `daily,daily`, contributor `daily`) keeps running."""

    @staticmethod
    def _args(domains):
        return types.SimpleNamespace(domains=domains)

    @staticmethod
    def _args_full(domains):
        return types.SimpleNamespace(domains=domains, data_dir="D", mode="glm")

    def test_the_digest_covers_content_and_order(self):
        d = N.domains_digest
        assert d(["code", "gutenberg"]) == d(["code", "gutenberg"])
        assert d(["code", "gutenberg"]) != d(["gutenberg", "code"]), \
            "ORDER decides doms[E % len(doms)] -- a reordered list is a DIFFERENT mapping"
        assert d(["code", "gutenberg"]) != d(["code", "gutenberg", "web"]), \
            "appending one domain renumbers every modulus"
        assert len(d(["daily"])) == 64

    def test_equal_mappings_spelled_differently_are_not_a_mismatch(self):
        """The live pair: `daily,daily` and `daily` resolve EVERY coordinate to the same domain, so refusing
        to start on it would break a provably safe running campaign."""
        assert N.domains_canonical(["daily", "daily"]) == ["daily"]
        assert N.domains_canonical(["a", "b", "a", "b"]) == ["a", "b"]
        assert N.domains_canonical(["code", "gutenberg", "web"]) == ["code", "gutenberg", "web"]
        assert N.domains_digest(["daily", "daily"]) == N.domains_digest(["daily"])
        for e in range(6):
            assert N._ids_path(self._args_full("daily,daily"), N.coord_data_slot(1, e), "train") \
                == N._ids_path(self._args_full("daily"), N.coord_data_slot(1, e), "train"), \
                "canonicalization must only ever merge lists that really do resolve identically"

    def test_whitespace_and_empty_entries_do_not_create_a_false_mismatch(self):
        assert N.domains_list(self._args(" code , gutenberg ,")) == ["code", "gutenberg"]
        assert N.domains_digest(N.domains_list(self._args(" code , gutenberg ,"))) \
            == N.domains_digest(N.domains_list(self._args("code,gutenberg")))

    def test_a_mismatch_is_reported_and_names_both_lists(self):
        """The exact trap from the finding: an extra domain on the miner side only."""
        ptr = dict(N.domains_pointer_fields(self._args("code,gutenberg")), v=2, event=0)
        msg = N.domains_mismatch(ptr, self._args("code,gutenberg,web"))
        assert msg is not None, "the coordinator/miner domain trap went undetected"
        assert "code,gutenberg]" in msg, msg
        assert "code,gutenberg,web]" in msg, msg
        assert "MISMATCH" in msg

    def test_a_reordered_list_is_also_a_mismatch(self):
        ptr = dict(N.domains_pointer_fields(self._args("code,gutenberg")), v=2)
        msg = N.domains_mismatch(ptr, self._args("gutenberg,code"))
        assert msg is not None and "gutenberg,code]" in msg, msg

    def test_matching_lists_pass_and_an_absent_digest_still_starts(self):
        ptr = dict(N.domains_pointer_fields(self._args("daily,daily")), v=2)
        assert N.domains_mismatch(ptr, self._args("daily,daily")) is None
        assert N.domains_mismatch(ptr, self._args("daily")) is None           # the LIVE pair
        # ADDITIVE: a pre-Shard-Claim coordinator publishes no digest -> never a hard fail.
        assert N.domains_mismatch({"v": 2, "event": 3, "model_root": "r"}, self._args("web")) is None
        assert N.domains_mismatch({"round": 0, "state_cid": "r"}, self._args("web")) is None
        assert N.domains_mismatch(None, self._args("web")) is None

    def test_the_coordinator_stamps_the_digest_on_every_v2_pointer(self):
        """Genesis, per-event and terminal pointers all have to carry it: a contributor joining MID-RUN reads
        whatever the current pointer is, not the genesis one."""
        import sharddiloco_glm_coordinator as C
        import neurahash.diloco_merge as dm
        args = self._args("code,gutenberg")
        meta = N.domains_pointer_fields(args)
        writes = {}

        class _Lane:
            def put_json_named(self, name, obj):
                writes[name] = obj
                return "cid"

        lane, clock = _Lane(), dm.SlotClock()
        C._publish_async_genesis(lane, [(1, 0)], "root", domains=meta)
        assert N.domains_mismatch(writes[N.GLM_POINTER_NAME], args) is None
        assert N.domains_mismatch(writes[N.GLM_POINTER_NAME], self._args("web")) is not None
        writes.clear()
        C._commit_accepted_and_advance(lane, clock, [(1, 0)], 0, {"accepted": []}, "root2", False,
                                       domains=meta)
        assert writes[N.GLM_POINTER_NAME]["domains_digest"] == meta["domains_digest"]
        assert N.domains_mismatch(writes[N.GLM_POINTER_NAME], self._args("web")) is not None
        # ...and it stays ADDITIVE: omitting it reproduces the previous pointer byte-for-byte.
        assert C._build_pointer(dm.SlotClock(), [(1, 0)], "r") \
            == dm.sd_pointer_encode(0, {"1_0": 0}, "r", False)

    def test_the_contributor_refuses_to_start_on_a_mismatch(self):
        """Wiring: the pure helper is worthless unless main() actually exits on it, and the coordinator
        actually publishes what the miner checks."""
        import inspect
        src = inspect.getsource(N.main)
        assert "domains_mismatch(ptr, args)" in src, "main must cross-check the FIRST pointer it reads"
        assert "RC_DOMAINS_MISMATCH" in src and N.RC_DOMAINS_MISMATCH != 0
        assert "domains_digest" in src, "the absent-digest case must be logged, not silently skipped"
        import sharddiloco_glm_coordinator as C
        assert "domains_pointer_fields" in inspect.getsource(C.run_async_events)


# ================================================ ESFT expert-affinity claim selection (--claim-by)
# WHY (measured elsewhere, see docs/research/MOE_POSTTRAIN_2026-07-25.md +
# docs/SHARD_CLAIM_DESIGN.md "Selecting the coordinate by AFFINITY"): pick_start_coord chooses which
# expert to train by HASHING THE WALLET ADDRESS and next_claim_coord advances along a per-identity
# permutation. Both are routing-BLIND, and routing-blind selection is the one variant published work
# has MEASURED to lose -- MoE-Sieve (arXiv:2603.24044) put random expert selection 2.5 percentage
# points behind router-guided at a matched budget; Mixtral (arXiv:2401.04088 sec 5) showed nobody gets
# to DECIDE what an expert specialises in ("we do not observe obvious patterns in the assignment of
# experts based on the topic"); Branch-Train-Merge (arXiv:2208.03306) showed random splits do not work
# at this exact 64-expert count. ESFT (arXiv:2407.01906) is the published alternative on a
# near-identical architecture: probe with a small forward-only sample, score every expert, train the
# top-scored ones.
class TestEsftSelectionRule:
    """The PURE half: ESFT's threshold rule, "the smallest top-scored set E_s^l with
    SUM_{i in E_s^l} R_i^l >= p", at the paper's verbatim thresholds ("The threshold p is set to 0.1
    for ESFT-Gate and 0.2 for ESFT-Token, respectively"). No torch, no model -- microseconds."""

    def test_the_published_thresholds_are_the_ones_we_ship(self):
        assert N.ESFT_P_GATE == 0.1 and N.ESFT_P_TOKEN == 0.2

    def test_returns_the_smallest_set_clearing_the_threshold(self):
        """SMALLEST is the whole point: the set is a training budget, so one coordinate too many is
        wasted GPU-hours and one too few misses the mass the threshold was calibrated for."""
        s = {(1, 0): 0.5, (1, 1): 0.3, (1, 2): 0.15, (1, 3): 0.05}
        assert N.esft_select(s, 0.1) == [(1, 0)]                  # 0.5 alone already clears 0.1
        assert N.esft_select(s, 0.5) == [(1, 0)]                  # >= is inclusive, so still one
        assert N.esft_select(s, 0.51) == [(1, 0), (1, 1)]         # exactly one more, never two
        assert N.esft_select(s, 0.95) == [(1, 0), (1, 1), (1, 2)]

    def test_it_is_descending_by_score_not_by_coordinate(self):
        s = {(1, 0): 0.01, (1, 3): 0.9, (1, 1): 0.09}
        assert N.esft_select(s, 0.5) == [(1, 3)]

    def test_degenerate_thresholds_do_not_invent_or_lose_coordinates(self):
        s = {(1, 0): 0.6, (1, 1): 0.4}
        assert N.esft_select(s, 0.0) == []                        # smallest set clearing 0 is nothing
        assert N.esft_select(s, 5.0) == [(1, 0), (1, 1)]          # unreachable p -> everything
        assert N.esft_select({}, 0.1) == []

    def test_ties_break_on_the_coordinate_so_two_nodes_agree(self):
        """Two miners probing the same model must select the same set, or one of them silently trains
        an expert the other believes it owns."""
        s = [((1, 2), 0.25), ((1, 0), 0.25), ((1, 1), 0.25), ((1, 3), 0.25)]
        assert N.esft_select(s, 0.3) == [(1, 0), (1, 1)]
        assert N.esft_select(list(reversed(s)), 0.3) == [(1, 0), (1, 1)]

    def test_the_threshold_is_applied_per_layer_not_pooled(self):
        """ESFT's scores carry the layer superscript (g_i^l, r_i^l) and sum to 1 WITHIN a layer, so
        pooling two candidate layers doubles the total and p=0.1 would clear on half the mass it
        describes. A node holding several pieces spans layers, so this is not hypothetical."""
        s = {(1, 0): 0.7, (1, 1): 0.3, (2, 0): 0.6, (2, 1): 0.4}
        assert N.esft_select_layers(s, 0.5) == [(1, 0), (2, 0)]   # one per layer
        assert N.esft_select(s, 0.5) == [(1, 0)]                  # pooled: layer 2 never considered
        assert N.esft_select_layers(s, 0.8) == [(1, 0), (1, 1), (2, 0), (2, 1)]


@pytest.fixture(scope="module")
def probe_host(host):
    """A host holding EVERY claimable coordinate (layers 1-2 x experts 0-3 on the tiny GLM), which is
    what main() builds before the probe runs, plus the tiny train split as the token sample. Module
    scope: build_tiny_glm is seconds and the probe must not mutate anything, which is exactly what
    test_the_probe_updates_no_parameter asserts."""
    G, model, cfg, claimable = host
    h = G.GlmExpertLaneHost(model, cfg, [(1, 0), (1, 1)], claimable=claimable)
    for c in claimable:
        if h.index_of(*c) is None:
            h.register(*c)
    return h, claimable, N.tiny_ids("train")


def _state_hash(model):
    """sha256 over the FULL state_dict (parameters AND buffers, in sorted key order). Buffers matter:
    the router's e_score_correction_bias is a buffer, and piece_loader writes -inf into it."""
    import hashlib
    hh = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        hh.update(k.encode())
        hh.update(v.detach().cpu().numpy().tobytes())
    return hh.hexdigest()


class TestEsftAffinityProbe:
    """The probe on a REAL tiny GLM (real router, real fused experts, real top-k). Forward passes only."""

    def test_the_probe_updates_no_parameter(self, probe_host):
        """THE LOAD-BEARING TEST. The base is FROZEN and every per-coordinate lineage root
        (slot_root -> base_slot_root -> _lineage_ok) is a hash over it, so a "probe" that trained even
        one expert by accident would change our root, get every later contribution dropped
        `wrong-lineage-slot-root` forever, and corrupt the weights the whole fleet gates against. Full
        state_dict hash, not a spot check."""
        h, claimable, ids = probe_host
        before = _state_hash(h.model)
        N.probe_expert_affinity(h, ids, coords=claimable, samples=8)
        assert _state_hash(h.model) == before, \
            "the affinity probe MUTATED the frozen base -- it must be forward-only"

    def test_no_grad_and_no_optimizer_are_left_behind(self, probe_host):
        """Belt and braces on the same property: no parameter may come back with a gradient (which
        would prove a backward ran), and no forward hook may survive the call (a leaked hook would
        keep accumulating during TRAINING forwards and slow every later round)."""
        h, claimable, ids = probe_host
        N.probe_expert_affinity(h, ids, coords=claimable, samples=8)
        assert all(p.grad is None for p in h.model.parameters())
        for L in (1, 2):
            mod = N._routed_experts_module(h, L)
            assert len(getattr(mod, "_forward_pre_hooks", {})) == 0, \
                "a probe hook leaked onto layer %d" % L

    def test_ranks_exactly_the_claimable_coordinates(self, probe_host):
        """No extras (a non-resident row is writable and silently inert -- claiming one trains forever
        and is rejected forever) and no omissions (an unranked coordinate is starved out of the sweep)."""
        h, claimable, ids = probe_host
        rep = N.probe_expert_affinity(h, ids, coords=claimable, samples=8)
        ranked = [c for (c, _g, _r) in rep["ranking"]]
        assert sorted(ranked) == sorted(tuple(c) for c in claimable)
        assert len(ranked) == len(set(ranked)) == len(claimable)
        assert sorted(rep["gate"]) == sorted(rep["token"]) == sorted(ranked)

    def test_the_ranking_is_ordered_highest_affinity_first(self, probe_host):
        h, claimable, ids = probe_host
        rep = N.probe_expert_affinity(h, ids, coords=claimable, samples=8)
        gates = [g for (_c, g, _r) in rep["ranking"]]
        assert gates == sorted(gates, reverse=True)

    def test_is_deterministic_for_a_fixed_model_and_sample(self, probe_host):
        """Bit-identical, not approximately equal: the ranking decides which coordinate a miner claims,
        so two runs of the same miner on the same base must agree or a restart silently re-claims."""
        h, claimable, ids = probe_host
        a = N.probe_expert_affinity(h, ids, coords=claimable, samples=8)
        b = N.probe_expert_affinity(h, ids, coords=claimable, samples=8)
        assert a["ranking"] == b["ranking"]
        assert all(a["gate"][c] == b["gate"][c] for c in a["gate"])
        assert all(a["token"][c] == b["token"][c] for c in a["token"])

    def test_both_esft_metrics_are_computed_and_normalised_per_layer(self, probe_host):
        """ESFT-Gate g_i^l and ESFT-Token r_i^l are DIFFERENT statistics (mean gate score vs. token
        selection ratio) and both must be reported -- the paper thresholds them separately. Each sums to
        ~1 within a layer, which is what makes p a fraction: the token ratio exactly (every token
        contributes 1/K per pick, K picks), the gate score once routed_scaling_factor is divided out."""
        h, claimable, ids = probe_host
        rep = N.probe_expert_affinity(h, ids, coords=claimable, samples=8)
        assert rep["topk"] == N.TINY["topk"] and rep["scaling"] == 1.8      # GLM scales by 1.8, not 1
        for L in (1, 2):
            gate_sum = sum(v for c, v in rep["gate"].items() if c[0] == L)
            tok_sum = sum(v for c, v in rep["token"].items() if c[0] == L)
            assert abs(tok_sum - 1.0) < 1e-9, (L, tok_sum)
            assert abs(gate_sum - 1.0) < 1e-6, (L, gate_sum)
        assert rep["gate"] != rep["token"], "the two metrics collapsed into one -- one is not computed"

    def test_the_p_threshold_selection_uses_both_metrics_at_their_own_thresholds(self, probe_host):
        h, claimable, ids = probe_host
        rep = N.probe_expert_affinity(h, ids, coords=claimable, samples=8)
        assert rep["select_gate"] == N.esft_select_layers(rep["gate"], N.ESFT_P_GATE)
        assert rep["select_token"] == N.esft_select_layers(rep["token"], N.ESFT_P_TOKEN)
        assert rep["select_gate"] and rep["select_token"]
        # SMALLEST-set property, re-checked on the real measured scores rather than on fixtures.
        for L in (1, 2):
            sel = [c for c in rep["select_gate"] if c[0] == L]
            assert sum(rep["gate"][c] for c in sel) >= N.ESFT_P_GATE
            assert sum(rep["gate"][c] for c in sel[:-1]) < N.ESFT_P_GATE

    def test_the_sample_size_is_a_bounded_parameter(self, probe_host):
        """ESFT's own anchor is 32 samples x 4096 tokens ~= 131K tokens, forward-only. The probe must
        never read the whole train split by default -- on the real lane that is the entire corpus."""
        h, claimable, ids = probe_host
        assert N.ESFT_PROBE_SAMPLES == 32
        rep = N.probe_expert_affinity(h, ids, coords=claimable, samples=4)
        assert rep["n_samples"] == 4 and rep["n_tokens"] == 4 * N.TINY["seq"]
        assert len(ids) == N.TINY["train_n"] > 4                 # i.e. it really did sample, not read all
        big = N.probe_expert_affinity(h, ids, coords=claimable, samples=len(ids) + 10)
        assert big["n_samples"] == len(ids)                      # clamps, never indexes past the end

    def test_it_reuses_the_model_it_was_given(self, probe_host):
        """A second model does not fit: the real base is 4.02 GiB of trunk plus 1.125 GiB per resident
        layer (memory glm-capacity-per-card). The probe therefore takes a HOST, never a path, and the
        object identity of the model must be unchanged afterwards."""
        h, claimable, ids = probe_host
        before = id(h.model)
        N.probe_expert_affinity(h, ids, coords=claimable, samples=4)
        assert id(h.model) is not None and id(h.model) == before
        import inspect
        sig = inspect.signature(N.probe_expert_affinity)
        assert list(sig.parameters)[:2] == ["host", "ids"]
        src = inspect.getsource(N.probe_expert_affinity)
        assert "build_tiny_glm" not in src and "build_node_model" not in src and "load_pieces" not in src

    def test_the_hook_target_is_the_module_the_router_actually_calls(self, probe_host):
        """host._fused() unwraps to `.base`, whose forward LoRAExperts NEVER calls (it reaches into
        base.gate_up_proj directly), so hooking there would silently never fire -- and a probe that
        measures nothing returns a ranking that is pure noise. Must be layers[L].mlp.experts, wrapper
        included."""
        h, _claimable, _ids = probe_host
        layer = h.model.model.layers[1]
        assert N._routed_experts_module(h, 1) is layer.mlp.experts
        G = N._G()
        LoRAExperts = G._lora_experts_cls()
        base = layer.mlp.experts
        try:
            layer.mlp.experts = LoRAExperts(base, {0: 0}, r=2, alpha=4)
            assert N._routed_experts_module(h, 1) is layer.mlp.experts       # the WRAPPER
            assert N._routed_experts_module(h, 1) is not h._fused(1)         # not the unwrapped base
        finally:
            layer.mlp.experts = base
        assert _state_hash(h.model)                                          # model still intact

    def test_an_unhostable_layer_still_fails_loudly(self, probe_host):
        h, _claimable, ids = probe_host
        with pytest.raises(IndexError, match=r"not instantiated"):
            N.probe_expert_affinity(h, ids, coords=[(N.TINY["layers"], 0)], samples=2)

    def test_a_useless_sample_or_empty_candidate_set_is_refused_up_front(self, probe_host):
        h, claimable, _ids = probe_host
        with pytest.raises(SystemExit, match=r"no candidate coordinates"):
            N.probe_expert_affinity(h, np.zeros((4, 8), dtype=np.int64), coords=[], samples=2)
        with pytest.raises(SystemExit, match=r"\[N,T\] token sample"):
            N.probe_expert_affinity(h, np.zeros((0, 8), dtype=np.int64), coords=claimable)


class TestClaimByFlag:
    """--claim-by {hash,affinity}. hash is the DEFAULT, so a miner that does not ask for affinity
    behaves exactly as v3.3.2 did; affinity claims the highest-affinity coordinate and advances on
    plateau to the next-highest instead of the next hash bucket."""

    @staticmethod
    def _args(**kw):
        a = dict(expert=None, slot=None, mode="glm", slots="1:0", domains="daily", piece=0,
                 shard_dir="x", config_dir="x", claim_by="hash")
        a.update(kw)
        return types.SimpleNamespace(**a)

    def test_hash_is_the_default_flag_value(self):
        ap = __import__("argparse").ArgumentParser()
        N.add_common_args(ap)      # --claim-by lives on the contributor parser; check main()'s default
        import inspect
        src = inspect.getsource(N.main)
        assert '"--claim-by"' in src and 'default=os.environ.get("NEURAHASH_SD_CLAIM_BY", "hash")' in src
        assert 'choices=("hash", "affinity")' in src

    def test_claim_by_hash_reproduces_todays_wallet_hash_choice(self, monkeypatch):
        """NO SILENT BEHAVIOUR CHANGE. The live campaign runs without this flag, so the default path
        must resolve to the same coordinate the wallet hash resolved to before it existed --
        resolve_claim does not even look at claim_by."""
        monkeypatch.delenv("NEURAHASH_SD_EXPERT", raising=False)
        claim = [(1, e) for e in range(5)]
        monkeypatch.setattr(N, "node_claimable_coords", lambda a: claim)
        wallet = "0x" + "ab" * 20
        expect = N.pick_start_coord(claim, wallet)
        for a in (self._args(), self._args(claim_by="affinity"), types.SimpleNamespace(
                expert=None, slot=None, mode="glm", slots="1:0", domains="daily", piece=0,
                shard_dir="x", config_dir="x")):                      # no claim_by attribute at all
            L, E, i, src = N.resolve_claim(a, N.parse_slots("1:0"), log=lambda *x: None,
                                           identity=wallet)
            assert (L, E) == expect and "wallet-hash" in src
        import inspect
        assert "claim_by" not in inspect.getsource(N.resolve_claim), \
            "the hash path must be untouched by the new flag"

    def test_affinity_claims_the_top_ranked_coordinate(self, probe_host, monkeypatch):
        """The wiring, with a KNOWN ranking so the assertion is about selection and not about which
        expert the tiny model happens to prefer."""
        h, claimable, ids = probe_host
        want = (2, 3)
        order = [want] + [c for c in claimable if c != want]
        monkeypatch.setattr(N, "probe_expert_affinity", lambda *a, **k: {
            "ranking": [(c, 1.0 - n * 0.1, 0.5) for n, c in enumerate(order)],
            "gate": {c: 1.0 - n * 0.1 for n, c in enumerate(order)}})
        monkeypatch.setattr(N, "claim_all_coords", lambda a, s: [tuple(c) for c in claimable])
        L, E, i, ranked = N.affinity_claim(self._args(claim_by="affinity"), h, ids, 1, 0,
                                           h.index_of(1, 0), log=lambda *x: None)
        assert (L, E) == want
        assert i == h.index_of(*want), "the claim must land on the LOCAL slot index for that coordinate"
        assert ranked[0] == want and sorted(ranked) == sorted(tuple(c) for c in claimable)

    def test_affinity_advances_to_the_next_highest_on_plateau(self, probe_host, monkeypatch):
        """Plateau = K consecutive gate rejects. Under affinity the release must drop to the
        NEXT-HIGHEST-affinity coordinate, not to the next wallet-hash bucket, and must still cycle so
        no coordinate is starved."""
        h, claimable, ids = probe_host
        order = [(2, 3), (1, 2), (1, 0), (2, 1), (1, 1), (1, 3), (2, 0), (2, 2)]
        assert sorted(order) == sorted(tuple(c) for c in claimable)
        assert N.next_claim_coord(claimable, (2, 3), identity="0xabc", ranked=order) == (1, 2)
        assert N.next_claim_coord(claimable, (1, 2), identity="0xabc", ranked=order) == (1, 0)
        assert N.next_claim_coord(claimable, order[-1], identity="0xabc", ranked=order) == order[0]
        assert N.claim_walk_order(claimable, "0xabc", ranked=order) == order
        # ...and the hash permutation is genuinely a DIFFERENT order, so this test would pass
        # vacuously if the ranking were being ignored.
        assert N.claim_walk_order(claimable, "0xabc") != order

    def test_the_walk_never_drops_a_claimable_coordinate(self):
        """A ranking that is stale (probed before a piece changed) must not starve the coordinates it
        does not mention, and must not smuggle in ones this node cannot host."""
        claim = [(1, 0), (1, 1), (1, 2)]
        got = N.claim_walk_order(claim, "0xabc", ranked=[(1, 2), (9, 9), (1, 2)])
        assert got == [(1, 2), (1, 0), (1, 1)]
        assert (9, 9) not in got and len(got) == len(set(got)) == 3

    def test_a_failed_probe_keeps_the_hash_claim_instead_of_stopping_the_miner(self, probe_host,
                                                                              monkeypatch):
        """Public testing, anyone may join: a probe that raises must degrade to today's behaviour, not
        kill a stranger's miner. ranked=None then also puts the sweep back on the hash permutation."""
        h, claimable, ids = probe_host

        def _boom(*a, **k):
            raise RuntimeError("probe exploded")

        monkeypatch.setattr(N, "probe_expert_affinity", _boom)
        monkeypatch.setattr(N, "node_claimable_coords", lambda a: [tuple(c) for c in claimable])
        said = []
        L, E, i, ranked = N.affinity_claim(self._args(claim_by="affinity"), h, ids, 1, 1,
                                           h.index_of(1, 1), miner="glm-x", log=said.append)
        assert (L, E, i) == (1, 1, h.index_of(1, 1)) and ranked is None
        assert any("affinity probe FAILED" in s for s in said), said

    def test_main_actually_calls_the_probe_and_reloads_the_data_shard(self):
        """A pure helper nobody calls is worthless. main() must (a) gate the probe on --claim-by
        affinity, (b) re-read the ids for the NEW coordinate -- the shard is
        doms[coord_data_slot(L,E) % len(doms)], so re-claiming without a reload trains one domain and
        self-gates on another (C6) -- and (c) hand the ranking to the async loop's plateau advance."""
        import inspect
        src = inspect.getsource(N.main)
        assert 'getattr(args, "claim_by", "hash")) == "affinity"' in src
        assert "affinity_claim(" in src
        i_claim = src.index("affinity_claim(")
        tail = src[i_claim:i_claim + 900]
        assert "node_ids(args, coord_data_slot(L, E), \"train\")" in tail
        assert "node_ids(args, coord_data_slot(L, E), \"val\")" in tail
        assert "claim_ranked=_claim_ranked" in src
        asrc = inspect.getsource(N._run_async)
        assert "ranked=claim_ranked" in asrc, "the plateau advance must follow the affinity order"
        assert "claim_ranked" in asrc.split("\n")[0] + asrc.split("\n")[1]


# ==================================================== CAMPAIGN SCOPING (cross-campaign replay, 07-25)
# MEASURED (scratchpad/FINDING_cross_campaign_replay.md): the never-deleting store held 11,229 objects
# from every campaign that ever ran under ONE flat namespace (cg/r<N>/<miner>), so a fresh coordinator
# discovered dead runs' records forever -- and because every campaign starts from the SAME pristine
# base, at genesis their base_root/base_slot_root were EQUAL, so those records were lineage-VALID.
# Identity glm-ea20C873 belongs to no live miner and was MINTED into runs 2, 3 and 4.
class TestCampaignIdIsAWellFormedNonce:

    def test_a_minted_id_is_hex_and_round_trips(self):
        cid = N.new_campaign_id()
        assert N.normalize_campaign_id(cid) == cid and len(cid) == 16
        assert N.new_campaign_id() != N.new_campaign_id(), "a campaign id is a NONCE, not a constant"

    def test_junk_is_normalized_to_none_so_it_never_reaches_a_name_or_a_hash_seed(self):
        """The pointer is UNSIGNED on a shared-token lane: anything we cannot validate is treated as NO
        campaign (fail-closed) rather than pasted into an object name or a digest."""
        for bad in (None, "", "  ", "not-hex", "../../etc", "r7", "ab/cd", "0123", "G" * 8, "f" * 65):
            assert N.normalize_campaign_id(bad) is None, bad
        assert N.normalize_campaign_id("  DEADBEEF12345678 ") == "deadbeef12345678"   # trimmed+lowered

    def test_the_id_can_never_be_read_as_a_round_number(self):
        """cg/<id>/ and the legacy cg/r<N>/ share a prefix, so an id beginning with 'r' would make the
        two shapes ambiguous. Hex-only settles it by construction."""
        assert N.normalize_campaign_id("r0000000") is None
        assert N.campaign_prefix("ab12cd34ab12cd34") == "cg/ab12cd34ab12cd34/"
        assert N.campaign_prefix(None) == "cg/"


class TestCampaignScopedNames:

    _CID = "ab12cd34ab12cd34"

    def test_records_live_under_the_campaign_prefix(self):
        assert N.contrib_prefix(7, self._CID) == "cg/%s/r7/" % self._CID
        assert N.contrib_name(7, "glm-abc", self._CID) == "cg/%s/r7/glm-abc" % self._CID
        assert N.async_publish_name(7, "glm-abc", 2, self._CID) == "cg/%s/r7/glm-abc.2" % self._CID

    def test_no_campaign_is_byte_identical_to_the_pre_campaign_wire(self):
        """Every name the LIVE lane has ever published is unscoped; the legacy shape must not move."""
        assert N.contrib_prefix(7) == "cg/r7/"
        assert N.contrib_name(7, "m") == "cg/r7/m"
        assert N.async_publish_name(7, "m", 2) == "cg/r7/m.2"

    @coordinator_only
    def test_both_name_shapes_parse_but_discovery_accepts_only_ours(self):
        """The parser used by DISCOVERY is campaign-scoped; the generic one only answers "is this a
        contribution record at all" (the manifest-visibility question)."""
        import sharddiloco_glm_coordinator as C
        ours = N.contrib_name(3, "m", self._CID)
        theirs = N.contrib_name(3, "m", "ff99ff99ff99ff99")
        legacy = N.contrib_name(3, "m")
        assert C._parse_contrib_name(ours) == C._parse_contrib_name(theirs) == (3, "m")
        assert C._parse_contrib_name(legacy) == (3, "m")
        assert C._parse_contrib_name("sharddiloco/glm/pointer") is None
        mine = C._contrib_name_parser(self._CID)
        assert mine(ours) == (3, "m")
        assert mine(theirs) is None, "another campaign's record must be invisible to discovery"
        assert mine(legacy) is None, "an unscoped record predates this campaign -- also invisible"
        # ... and symmetrically, a legacy (unscoped) coordinator does not pick up scoped records,
        # whose seeded roots it could not validate anyway.
        assert C._contrib_name_parser(None)(legacy) == (3, "m")
        assert C._contrib_name_parser(None)(ours) is None

    @coordinator_only
    def test_an_id_full_of_digits_still_parses(self):
        """Regression on the matcher itself: it is built by splitting the FORMAT on its %d. Substituting
        the id first and splitting on a literal digit would cut the pattern INSIDE the id -- and hex ids
        are mostly digits, so this would have failed on roughly every real campaign."""
        import sharddiloco_glm_coordinator as C
        for cid in ("0000000000000000", "0123456789abcdef", "00ff00ff00ff00ff", "9999999999999999"):
            name = N.contrib_name(12, "glm-x.3", cid)
            assert name == "cg/%s/r12/glm-x.3" % cid
            assert C._contrib_name_parser(cid)(name) == (12, "glm-x.3"), cid
            assert C._parse_contrib_name(name) == (12, "glm-x.3"), cid
            assert C._contrib_name_parser("aaaaaaaaaaaaaaaa")(name) is None, cid


class TestCampaignSeedsTheLineageRoot:
    """The half that actually stops the merge. Namespacing alone is an optimisation a replayer routes
    around by renaming; the ROOT is what the coordinator judges."""

    def test_two_campaigns_over_the_identical_base_share_no_root(self, h):
        a = h
        base_slot, base_model = N.slot_root(a, 0), N.model_root(a)
        N.bind_campaign_id(a, "aaaaaaaaaaaaaaaa")
        s_a, m_a = N.slot_root(a, 0), N.model_root(a)
        N.bind_campaign_id(a, "bbbbbbbbbbbbbbbb")
        s_b, m_b = N.slot_root(a, 0), N.model_root(a)
        assert len({base_slot, s_a, s_b}) == 3, "identical weights must NOT hash alike across campaigns"
        assert len({base_model, m_a, m_b}) == 3

    def test_an_unbound_host_reproduces_the_pre_campaign_digest_exactly(self, h):
        """Protects every root the live campaign has already published: no campaign -> nothing is fed
        into the digest, so model_root/slot_root are byte-identical to v3.4.1."""
        import hashlib
        import numpy as np
        assert N.host_campaign_id(h) is None
        want = hashlib.sha256()
        d = h.read_slot(0)
        L, E = h.slots[0]
        want.update(("L%dE%d|" % (L, E)).encode())
        for k in sorted(d):
            want.update(k.encode())
            want.update(np.ascontiguousarray(d[k], dtype=np.float32).tobytes())
        assert N.slot_root(h, 0) == want.hexdigest()
        N.bind_campaign_id(h, None)                       # explicit None is also "unscoped"
        assert N.slot_root(h, 0) == want.hexdigest()

    def test_the_seed_is_a_prefix_and_the_scope_rides_the_host(self, h):
        """A suffix could be appended by anyone holding the inner digest; a prefix scopes the whole
        hash. And the scope lives on the HOST, so two hosts in one process cannot leak into each
        other -- which is exactly what a coordinator + a foreign miner in one test are."""
        import hashlib
        N.bind_campaign_id(h, "abcdef0123456789")
        want = hashlib.sha256()
        want.update(b"campaign:abcdef0123456789|")
        N._slot_digest_into(want, h, 0)
        assert N.slot_root(h, 0) == want.hexdigest()
        assert N.host_campaign_id(h) == "abcdef0123456789"
        assert N.host_campaign_id(types.SimpleNamespace()) is None      # any host object is tolerated


@coordinator_only
class TestCampaignSurvivesACoordinatorRestart:
    """THE failure mode this must not introduce: a legitimate restart that MINTED a new id would orphan
    the campaign's own records (their names and their seeded roots both move), which is worse than the
    cross-campaign replay the id exists to prevent. So: mint once, persist, load on every resume."""

    @staticmethod
    def _args(tmp_path, **kw):
        a = types.SimpleNamespace(campaign=None, campaign_file=str(tmp_path / "glm_campaign.json"),
                                  coord_data_dir=str(tmp_path), no_resume=False, resume=False,
                                  url="http://lane-a:8710")
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def test_a_restart_resumes_the_same_id(self, tmp_path):
        import sharddiloco_glm_coordinator as C
        first, minted = C._campaign_id_for_run(self._args(tmp_path), environ={})
        assert minted is True and N.normalize_campaign_id(first) == first
        again, minted2 = C._campaign_id_for_run(self._args(tmp_path), environ={})
        assert (again, minted2) == (first, False), "a restart must NOT mint a new campaign"
        third, _ = C._campaign_id_for_run(self._args(tmp_path), environ={})
        assert third == first

    def test_no_resume_means_a_deliberately_fresh_campaign(self, tmp_path):
        """--no-resume already means "start from the frozen base"; a fresh base with a REUSED campaign
        id would put the new run back in the old run's namespace."""
        import sharddiloco_glm_coordinator as C
        first, _ = C._campaign_id_for_run(self._args(tmp_path), environ={})
        fresh, minted = C._campaign_id_for_run(self._args(tmp_path, no_resume=True), environ={})
        assert minted is True and fresh != first
        # ... and the new id is what the NEXT restart resumes.
        again, _ = C._campaign_id_for_run(self._args(tmp_path), environ={})
        assert again == fresh

    def test_a_pinned_id_wins_and_is_persisted(self, tmp_path):
        import sharddiloco_glm_coordinator as C
        C._campaign_id_for_run(self._args(tmp_path), environ={})
        pinned, minted = C._campaign_id_for_run(
            self._args(tmp_path, campaign="C0FFEE00C0FFEE00"), environ={})
        assert (pinned, minted) == ("c0ffee00c0ffee00", False)
        assert C._load_campaign_id(str(tmp_path / "glm_campaign.json")) == "c0ffee00c0ffee00"

    def test_a_fresh_run_on_ANOTHER_lane_does_not_clobber_this_lane(self, tmp_path):
        """The state file lives in --coord-data-dir, which every run on this box shares (the SECRET
        probe/heldout splits are there). A from-scratch A/B baseline (--no-resume) against a different
        --url must not overwrite the LIVE lane's id: the damage would only appear at the live
        coordinator's NEXT restart, as a rollback to the frozen base plus every miner dropping out."""
        import sharddiloco_glm_coordinator as C
        live, _ = C._campaign_id_for_run(self._args(tmp_path), environ={})
        other, minted = C._campaign_id_for_run(
            self._args(tmp_path, url="http://lane-b:8711", no_resume=True), environ={})
        assert minted is True and other != live
        again, minted2 = C._campaign_id_for_run(self._args(tmp_path), environ={})
        assert (again, minted2) == (live, False), "the baseline run clobbered the live lane's campaign"
        # both entries coexist, keyed by lane
        state = C._read_campaign_state(str(tmp_path / "glm_campaign.json"))
        assert len(state) == 2
        assert sorted(v["url"] for v in state.values()) == ["http://lane-a:8710", "http://lane-b:8711"]

    def test_an_unknown_lane_says_what_the_file_does_hold_before_minting(self, tmp_path):
        """A URL change (moved tunnel, new VPS) must not resume silently onto a fresh base with nothing
        said: name the lanes we DO have and the way to adopt one."""
        import sharddiloco_glm_coordinator as C
        live, _ = C._campaign_id_for_run(self._args(tmp_path), environ={})
        said = []
        fresh, minted = C._campaign_id_for_run(self._args(tmp_path, url="http://moved:9000"),
                                               log=said.append, environ={})
        assert minted is True and fresh != live
        assert any("no campaign persisted for lane" in s and "http://lane-a:8710" in s for s in said), \
            said
        assert any("--campaign <id> to adopt" in s for s in said), said

    def test_an_unusable_pinned_id_stops_the_run_instead_of_minting_a_different_one(self, tmp_path):
        import sharddiloco_glm_coordinator as C
        with pytest.raises(SystemExit):
            C._campaign_id_for_run(self._args(tmp_path, campaign="nonsense!"), environ={})

    def test_a_corrupt_or_missing_state_file_is_treated_as_no_campaign_not_as_a_crash(self, tmp_path):
        import sharddiloco_glm_coordinator as C
        p = tmp_path / "glm_campaign.json"
        assert C._load_campaign_id(str(p)) is None
        p.write_text("{not json", encoding="utf-8")
        assert C._load_campaign_id(str(p)) is None
        p.write_text('{"campaign_id": "zzzz"}', encoding="utf-8")
        assert C._load_campaign_id(str(p)) is None
        cid, minted = C._campaign_id_for_run(self._args(tmp_path), environ={})
        assert minted is True and C._load_campaign_id(str(p)) == cid

    def test_the_documented_opt_out_turns_scoping_off_entirely(self, tmp_path):
        import sharddiloco_glm_coordinator as C
        assert C._campaign_id_for_run(self._args(tmp_path),
                                      environ={"NEURAHASH_SD_CAMPAIGN_SCOPE": "0"}) == (None, False)
        assert not (tmp_path / "glm_campaign.json").exists(), "the opt-out must not mint anything"
        assert N.campaign_scope_on({}) is True                       # default ON
        assert N.campaign_scope_on({"NEURAHASH_SD_CAMPAIGN_SCOPE": "off"}) is False

    def test_main_binds_the_campaign_before_anything_hashes_a_root(self):
        """Ordering is load-bearing: the resume replay verifies THIS campaign's accepted records against
        campaign-seeded roots, and the genesis pointer publishes one. Binding after either would make a
        resumed campaign unable to verify its own history."""
        import inspect
        import sharddiloco_glm_coordinator as C
        src = inspect.getsource(C.main)
        i_bind = src.index("N.bind_campaign_id(host, campaign)")
        assert i_bind < src.index("_resume_from_lane("), "bind BEFORE the resume replay"
        assert i_bind < src.index("_publish_async_genesis("), "bind BEFORE the genesis pointer"
        assert "_campaign_id_for_run(args" in src


@coordinator_only
class TestDiscoveryIgnoresForeignCampaigns:
    """The namespace half, measured where it hurts: the live store's manifest is 11,229 objects and one
    lane.manifest() call takes 23.79 s. Scoping discovery is not only a correctness fix -- it is what
    stops a fresh run from FETCHING (one get_json each) every dead record it can see."""

    _MINE = "1111222233334444"
    _THEIRS = "aaaabbbbccccdddd"

    def _names(self):
        return ([N.contrib_name(0, "m%d" % i, self._THEIRS) for i in range(3)]
                + [N.contrib_name(0, "legacy")]
                + [N.contrib_name(0, "mine", self._MINE)])

    def test_only_this_campaigns_names_are_collected(self):
        import sharddiloco_glm_coordinator as C
        got = C._collect_unprocessed(self._names(), set(), C._contrib_name_parser(self._MINE),
                                     max_base_event=0)
        assert [n for n, _b, _m in got] == [N.contrib_name(0, "mine", self._MINE)]
        assert len(got) == 1, "1 of 5 names, and the other 4 are never even dated"

    def test_the_loop_never_fetches_a_foreign_record(self, monkeypatch):
        """End-to-end through the REAL loop: a foreign record must cost ZERO get_json calls. Pre-fix
        every one of them was fetched and only then dropped by the lineage guard."""
        import sharddiloco_glm_coordinator as C
        TestFixAManifestIsNotReReadEveryPass._env(monkeypatch)
        recs = {n: _contrib_rec(0, n.rsplit("/", 1)[1]) for n in self._names()}
        lane = _CountingLane(recs)
        host = _RegistryHost([_A_COORD])
        args = _loop_args()
        args.campaign_id = self._MINE
        logs = []
        rc = C.run_async_events(None, None, host, lane, _PooledProbe(),
                                types.SimpleNamespace(verify=0.0, fwd=1.0), None, [], host.slots,
                                {}, args, 1.0, lambda *a: logs.append(" ".join(str(x) for x in a)))
        assert rc == 0
        assert lane.fetched == ["sha-" + N.contrib_name(0, "mine", self._MINE).replace("/", "-")], \
            "expected exactly our own record to be fetched, got %r" % (lane.fetched,)
        assert any("campaign=%s" % self._MINE in ln for ln in logs), "the loop must log its campaign"

    def test_the_pointer_advertises_the_campaign_so_a_miner_can_latch_it(self):
        import neurahash.diloco_merge as dm
        import sharddiloco_glm_coordinator as C
        ptr = C._build_pointer(dm.SlotClock(), [(1, 0)], "ROOT",
                               domains={"domains": ["code"], "domains_digest": "dd"},
                               campaign=self._MINE)
        assert ptr[N.CAMPAIGN_POINTER_KEY] == self._MINE
        assert N.pointer_campaign_id(ptr) == self._MINE
        assert ptr["domains_digest"] == "dd" and ptr["v"] == 2      # additive, nothing displaced
        plain = C._build_pointer(dm.SlotClock(), [(1, 0)], "ROOT")
        assert N.CAMPAIGN_POINTER_KEY not in plain, "no campaign -> the pre-campaign pointer bytes"


class TestAMinerRefusesAnUnscopedLane:
    """FAIL-CLOSED, and it has to SAY WHY: a public miner operator reading one line must be able to act.
    Publishing into the shared namespace anyway is what let a dead campaign's delta be minted."""

    _CID = "1234abcd1234abcd"

    def test_a_pointer_without_a_campaign_is_refused_by_name(self):
        msg = N.campaign_refusal({"v": 2, "event": 0, "rounds": {}, "model_root": "R"}, environ={})
        assert msg and "REFUSING to publish" in msg
        assert "NEURAHASH_SD_CAMPAIGN_SCOPE=0" in msg, "the way out must be in the same line"
        assert "campaign" in msg.lower()

    def test_a_pointer_with_a_campaign_is_accepted_and_latched(self):
        ptr = {"v": 2, "event": 0, "rounds": {}, "model_root": "R", "campaign_id": self._CID}
        assert N.campaign_refusal(ptr, environ={}) is None
        assert N.pointer_campaign_id(ptr) == self._CID

    def test_a_malformed_campaign_id_is_refused_like_a_missing_one(self):
        """Fail-closed: an id we cannot validate must not be pasted into names or hash seeds."""
        ptr = {"v": 2, "event": 0, "rounds": {}, "model_root": "R", "campaign_id": "../etc/passwd"}
        assert N.pointer_campaign_id(ptr) is None
        assert N.campaign_refusal(ptr, environ={}) is not None

    def test_the_opt_out_re_enables_publishing_on_a_legacy_lane(self):
        env = {"NEURAHASH_SD_CAMPAIGN_SCOPE": "0"}
        assert N.campaign_refusal({"v": 2, "event": 0, "rounds": {}, "model_root": "R"}, environ=env) is None

    def test_a_v1_SYNC_lane_is_never_refused_because_it_cannot_carry_a_campaign(self):
        """A v1 pointer has no field to advertise a campaign in, and the coordinator makes scoping inert
        in sync mode for exactly that reason. Refusing there would stop a DEFAULT-configured miner from
        joining ANY legacy v1 lane -- so the refusal must sit AFTER the mode decision and fire only on
        the async path. (Regression: it was first written before _select_async_mode.)"""
        import inspect
        v1 = {"round": 3, "state_cid": "ROOT", "done": False}
        assert N._select_async_mode(v1, {}) is False
        src = inspect.getsource(N.main)
        i_mode = src.index("_mode_async = _select_async_mode(")
        i_ref = src.index("campaign_refusal(ptr)")
        assert i_mode < i_ref, "the refusal must not run before the mode is known"
        assert src[i_ref - 400:i_ref].count("if _mode_async:") == 1, \
            "the campaign block must be gated on the ASYNC path"

    def test_main_actually_refuses_and_binds_before_publishing(self):
        """A pure helper nobody calls is worthless (same reason as the affinity wiring test): main()
        must exit on the refusal, bind the id to the host, and both publish paths must name-scope."""
        import inspect
        src = inspect.getsource(N.main)
        assert "campaign_refusal(ptr)" in src
        assert "return RC_NO_CAMPAIGN" in src and N.RC_NO_CAMPAIGN == 11
        assert "bind_campaign_id(host, pointer_campaign_id(ptr))" in src
        i_bind = src.index("bind_campaign_id(host, pointer_campaign_id(ptr))")
        assert i_bind < src.index("_run_async("), "bind BEFORE the async cadence starts publishing"
        assert "contrib_name(rnd, miner, host_campaign_id(host))" in src, "sync publish must scope too"
        asrc = inspect.getsource(N._run_async)
        assert "async_publish_name(base_event, miner, publish_k, host_campaign_id(host))" in asrc

    def test_a_campaign_change_mid_flight_exits_instead_of_starving_silently(self):
        """If the coordinator restarts into a NEW campaign, an already-running miner would publish where
        nobody looks -- and an undiscovered record is never even DROPPED, so no log anywhere would say
        so. The loop must notice and exit. (Wiring assertion: driving _run_async needs a real model.)"""
        import inspect
        asrc = inspect.getsource(N._run_async)
        assert "pointer_campaign_id(ptr)" in asrc
        assert "campaign CHANGED on the lane" in asrc
        i_guard = asrc.index("_ptr_camp != host_campaign_id(host)")
        assert i_guard < asrc.index("async_publish_name("), "check BEFORE publishing, not after"
        assert "return RC_NO_CAMPAIGN" in asrc[i_guard:i_guard + 800]


# ============================== v3.4.2 SEAT SQUATTING: the lineage verdict is taken BEFORE a seat is given
def _goodput(logs):
    """The FINAL goodput counters exactly as an operator reads them out of the log."""
    return json.loads([ln for ln in logs if "goodput FINAL" in ln][-1].split("goodput FINAL ", 1)[1])


@coordinator_only
class TestTerminalLineageTaxonomy:
    """WHICH lineage verdicts may be acted on BEFORE a coordinate is registered.

    Getting this list wrong has teeth in both directions. Too narrow and the measured seat squat stays:
    a lineage-dead record registers its coordinate, holds one of --max-active-slots and DEFERs an honest
    claimant out. Too wide and a record that is merely EARLY (`future-base-event`) or UNDECIDED
    (`unknown-coordinate`) is destroyed instead of queued -- silent theft of a miner's GPU hours, which
    on this lane is indistinguishable, miner-side, from a rejected delta."""

    _ROOTS = {0: "G0"}

    @staticmethod
    def _C():
        import sharddiloco_glm_coordinator as C
        return C

    def _verdict(self, rec, coord=(1, 0), cur=0, root_hist=None, srh=None, host=None):
        return self._C()._terminal_lineage_verdict(
            rec, coord, rec.get("base_event"), cur,
            self._ROOTS if root_hist is None else root_hist, {} if srh is None else srh, host=host)

    # ---------------------------------------------------------------------------- TERMINAL (safe to drop)
    def test_a_wrong_slot_root_is_terminal(self):
        """The measured squat: the record asserts it trained against weights we never had for this
        coordinate. No later event can make that true."""
        assert self._verdict(dict(base_event=0, base_slot_root="THEIRS"),
                             srh={(1, 0): [(0, "OURS")]}) == (True, "wrong-lineage-slot-root")

    def test_a_wrong_global_root_is_terminal_for_a_pre_shard_claim_record(self):
        assert self._verdict(dict(base_event=0, base_root="THEIRS")) == (True, "wrong-lineage-root")

    def test_a_legacy_record_against_a_grown_slot_list_is_terminal_and_keeps_its_own_name(self):
        """`legacy-miner-vs-dynamic-slots` is the same drop with an honest cause attached (an old client,
        not a forgery). Still terminal: that client cannot retroactively send a base_slot_root."""
        assert self._verdict(dict(base_event=0, base_root="THEIRS"), cur=3,
                             srh={(1, 9): [(3, "seeded-later")]}) == (True,
                                                                      "legacy-miner-vs-dynamic-slots")

    def test_a_height_we_cannot_even_parse_is_terminal(self):
        assert self._verdict(dict(base_event=None, base_root="G0")) == (True, "bad-base-event")
        assert self._verdict(dict(base_event="nope", base_root="G0")) == (True, "bad-base-event")

    # ------------------------------------------------------------------- RETRYABLE (must stay queued)
    def test_a_future_base_event_is_retryable(self):
        """A "not yet", not a rejection: the same record is valid once the clock reaches it."""
        assert self._verdict(dict(base_event=5, base_slot_root="X")) == (False, "future-base-event")

    def test_an_undecidable_coordinate_is_retryable(self):
        """At this point the coordinate is usually NOT registered, so the slot-root comparison often
        cannot be decided at all. Undecided must never read as "dead"."""
        assert self._verdict(dict(base_event=0, base_slot_root="X")) == (False, "unknown-coordinate")

    def test_an_unknown_event_is_retryable_even_though_the_loop_cannot_reach_it(self):
        """Unreachable here (the clock starts at 0 and root_hist gains a contiguous entry per commit),
        so it is classified fail-safe rather than relied upon."""
        assert self._verdict(dict(base_event=1, base_slot_root="X"), cur=2,
                             root_hist={0: "G0", 2: "G2"}) == (False, "unknown-event")

    def test_a_valid_record_is_never_dead(self):
        assert self._verdict(dict(base_event=0, base_slot_root="OURS"),
                             srh={(1, 0): [(0, "OURS")]}) == (False, "ok")

    def test_every_reason_lineage_ok_can_emit_is_deliberately_classified(self):
        """A reason added to _lineage_ok later falls through as RETRYABLE -- fail-safe, but SILENTLY. Pin
        the whole set so the next author has to look at this classification on purpose."""
        import inspect
        import re
        C = self._C()
        emitted = set(re.findall(r'return False, "([a-z-]+)"', inspect.getsource(C._lineage_ok)))
        assert emitted == {"bad-base-event", "future-base-event", "unknown-event", "unknown-coordinate",
                           "wrong-lineage-slot-root", "legacy-miner-vs-dynamic-slots",
                           "wrong-lineage-root"}
        assert set(C._TERMINAL_LINEAGE_REASONS) < emitted
        assert emitted - set(C._TERMINAL_LINEAGE_REASONS) == {"future-base-event", "unknown-event",
                                                             "unknown-coordinate"}

    # --------------------------------------------------------------------- nothing may be REMEMBERED
    def test_a_lineage_rollback_revives_a_record_that_was_dead_before_it(self):
        """_resume_from_lane can roll back and REWRITE root_hist, so a memo of "terminally dead" would
        wrongly exclude records that are valid for the NEW lineage. The same record object must be
        re-judged from live state every time."""
        C = self._C()
        rec = dict(base_event=1, base_root="G1-theirs")
        rh = {0: "G0", 1: "G1-ours"}
        assert C._terminal_lineage_verdict(rec, (1, 0), 1, 1, rh, {}) == (True, "wrong-lineage-root")
        rh[1] = "G1-theirs"                       # what a rollback to that lineage leaves behind
        assert C._terminal_lineage_verdict(rec, (1, 0), 1, 1, rh, {}) == (False, "ok")

    def test_a_rollback_of_a_per_coordinate_root_revives_it_too(self):
        C = self._C()
        rec = dict(base_event=0, base_slot_root="THEIRS")
        srh = {(1, 0): [(0, "OURS")]}
        assert C._terminal_lineage_verdict(rec, (1, 0), 0, 0, {0: "G0"}, srh)[0] is True
        srh[(1, 0)] = [(0, "THEIRS")]
        assert C._terminal_lineage_verdict(rec, (1, 0), 0, 0, {0: "G0"}, srh) == (False, "ok")


@coordinator_only
class TestProspectiveSlotRoot:
    """_prospective_slot_root: OUR root for a coordinate we have NOT registered.

    Without it the FIRST record on an unseen coordinate is undecidable (`unknown-coordinate`), so a
    forged or dead-campaign root would still have to be given a seat before it could be judged -- which
    is exactly the deadlock. Its load-bearing property is therefore an EQUALITY: it must return what
    _admit_coordinate would seed, or it would call honest first-contact claims dead."""

    @staticmethod
    def _C():
        import sharddiloco_glm_coordinator as C
        return C

    _UNSEEN = (2, 3)                     # claimable on the `h` fixture's host, never registered

    def test_it_equals_the_root_the_registry_would_seed_on_registration(self, h):
        got = self._C()._prospective_slot_root(h, self._UNSEEN)
        assert got is not None
        idx = h.register(*self._UNSEEN)
        assert N.slot_root(h, idx) == got, "a first-contact claim would be dropped as wrong-lineage"

    def test_it_takes_no_seat_and_leaves_no_slot_behind(self, h):
        """The whole point: deciding lineage must not cost the thing we are protecting."""
        before_slots, before_active = list(h.slots), set(h.active)
        self._C()._prospective_slot_root(h, self._UNSEEN)
        assert h.slots == before_slots and h.active == before_active
        assert h.index_of(*self._UNSEEN) is None

    def test_the_campaign_seed_is_carried_across(self, h):
        """That seed is what separates two campaigns over one pristine base. A prospective root computed
        without it would declare every campaign-scoped record dead -- the whole lane, not just replays."""
        C = self._C()
        unscoped = C._prospective_slot_root(h, self._UNSEEN)
        N.bind_campaign_id(h, "aaaa1111aaaa1111")
        scoped = C._prospective_slot_root(h, self._UNSEEN)
        assert scoped != unscoped
        assert N.slot_root(h, h.register(*self._UNSEEN)) == scoped

    def test_an_unhostable_coordinate_is_undecidable_not_dead(self, h):
        """(9,9) is outside the claimable set; _admit_coordinate refuses it with no seat, so there is
        nothing here to decide -- and None must mean "undecidable", never "mismatch"."""
        assert self._C()._prospective_slot_root(h, (9, 9)) is None

    def test_a_host_that_is_not_a_lane_host_is_undecidable(self):
        """Fails soft on any duck-typed host (test fakes, future host classes) instead of raising inside
        the event loop."""
        assert self._C()._prospective_slot_root(_RegistryHost([(1, 0)]), (1, 5)) is None


@coordinator_only
class TestALineageDeadRecordTakesNoSeat:
    """MEASURED DEADLOCK (2026-07-25). `_admit_coordinate` ran BEFORE the lineage check, so an
    authenticated-but-lineage-DEAD record REGISTERED its coordinate and took one of --max-active-slots.
    With one free seat, two dead records and one honest miner, the run measured:

        REGISTER (L1,E1) -> slot 1 on first contribution from glm-deadcamp (active=2, ...)
        LINEAGE-DROP cg/r0/glm-deadcamp.0 base_event=0 (wrong-lineage-slot-root)
        DEFER glm-live5090: max_active_slots=2 reached; cannot admit (L1,E3) yet
        events=0 ... -> live miner merged: False | dead coordinates registered: [(1, 1)]

    and it could not self-heal: idle eviction is counted in EVENTS, and events were stuck at 0 precisely
    because the honest miner could not merge. The same shape was observed on the live lane, where foreign
    identity glm-E2223497 held 1 of 6 seats.

    The dead records here get every advantage: on the roster, correctly signed, and claiming coordinates
    this node really hosts. Only `base_slot_root` is a root we never produced -- which is free to forge,
    because that field is not covered by the signature."""

    _DEAD = ((2, 0), (2, 1))             # coordinates the dead records claim
    _LIVE = (2, 2)                       # the honest miner's coordinate
    _DEAD_ROOT = "de" * 32               # a per-coordinate root this coordinator never produced

    @pytest.fixture(autouse=True)
    def _fast_idle(self, monkeypatch):
        """Pre-fix this scenario ends with the loop waiting for work it will never be able to merge, so
        the 600 s default idle guard would hang the suite. Set INSIDE the class so the file stays fast
        standalone and never depends on the caller's environment."""
        monkeypatch.setenv("NEURAHASH_SD_IDLE_EXIT_S", "1")
        monkeypatch.setenv("NEURAHASH_SD_POLL_S", "0.05")

    @classmethod
    def _run(cls, loop_model, store_harness, rounds=1):
        """ONE free seat (start slot (1,0), --max-active-slots=2), two dead claims, then the honest one."""
        env = _loop_env(loop_model, store_harness, [(1, 0)], rounds=rounds, max_active=2)
        G = loop_model[0]
        mh = G.GlmExpertLaneHost(loop_model[1], loop_model[2], [(1, 0)], claimable=env["claimable"])
        for n, coord in enumerate(cls._DEAD):
            mh.register(*coord)
            _publish_claim(env, coord, "dead%d" % n, wire_idx=1, host=mh, seed=40 + n,
                           slot_root=cls._DEAD_ROOT)
        mh.register(*cls._LIVE)
        _publish_claim(env, cls._LIVE, "honest", wire_idx=1, host=mh, seed=9)
        return env, _drive_loop(env, ["dead0", "dead1", "honest"])

    def test_the_dead_coordinates_are_not_registered_and_the_honest_miner_merges(self, loop_model,
                                                                                store_harness):
        """THE load-bearing assertion, on the REGISTRY and the accepted record -- not on a log string."""
        env, logs = self._run(loop_model, store_harness)
        host = env["host"]
        assert [host.index_of(*c) for c in self._DEAD] == [None, None], \
            "a lineage-dead record still took a seat:\n" + "\n".join(logs[-14:])
        assert host.index_of(*self._LIVE) == 1, \
            "the honest claimant did not get the only free seat:\n" + "\n".join(logs[-14:])
        assert sorted(host.slots[i] for i in host.active) == [(1, 0), self._LIVE]
        rec = env["store"].accepted(1)
        assert rec is not None and rec["accepted"], \
            "the honest miner earned nothing:\n" + "\n".join(logs[-14:])
        assert (rec["accepted"][0]["layer"], rec["accepted"][0]["glm_expert"]) == self._LIVE
        assert N.accepted_names_me(rec, "honest") is True

    def test_nothing_is_deferred_because_no_seat_was_ever_squatted(self, loop_model, store_harness):
        """DEFER was the visible symptom on the live lane, and the state that froze idle eviction."""
        _env, logs = self._run(loop_model, store_harness)
        assert not [ln for ln in logs if "DEFER" in ln], \
            "an honest claim was still deferred:\n" + "\n".join(logs[-14:])

    def test_the_seat_gate_pool_belongs_to_the_coordinate_that_really_claimed(self, loop_model,
                                                                             store_harness):
        """The secondary cost: registration also builds the secret gate pool (keyed by the coordinate's
        DATA domain) and, once begin_round has run, copies the whole ~75.5 MB fp32 slot on the real model.
        Pre-fix slot 1's pool was built for the dead coordinate. ensure_pool is idempotent and returns the
        pool already present, so asking for it with a sentinel reads back what registration installed."""
        env, logs = self._run(loop_model, store_harness)
        import sharddiloco_glm_coordinator as C

        def _same(a, b):
            return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))

        want, dead = (C._slot_probe_pool(env["args"], self._LIVE),
                      C._slot_probe_pool(env["args"], self._DEAD[0]))
        assert not _same(want, dead), "the two coordinates share a data domain: this test proves nothing"
        assert env["probe"].has_pool(1), "\n".join(logs[-12:])
        # ensure_pool is idempotent and returns the pool already present, so a sentinel call READS BACK
        # what registration installed without changing it.
        assert _same(env["probe"].ensure_pool(1, ("sentinel",)), want), \
            "slot 1's gate pool belongs to another coordinate's data domain"

    def test_the_early_drops_are_counted_and_named_apart_from_gate_rejections(self, loop_model,
                                                                             store_harness):
        """A silent filter is indistinguishable from a broken lane. The drops are PRE-gate, so they must
        not land in processed/rejected_gate (which would either fake work or fake failure) -- they get
        their own counter, stay inside the existing `staled` accounting, and say "pre-register" in the
        log so they cannot be confused with the post-registration lineage pass."""
        env, logs = self._run(loop_model, store_harness)
        early = [ln for ln in logs if "LINEAGE-DROP(pre-register)" in ln]
        assert len(early) == 2, "\n".join(logs[-14:])
        assert all("wrong-lineage-slot-root" in ln for ln in early), early
        assert all(("(L%d,E%d)" % self._DEAD[i]) in early[i] for i in (0, 1)), early
        gp = _goodput(logs)
        assert gp["lineage_dead_pregate"] == 2, gp
        assert gp["staled"] >= 2, gp
        assert (gp["processed"], gp["accepted"], gp["rejected_gate"]) == (1, 1, 0), gp

    def test_the_counter_is_reported_even_when_it_never_fires(self, loop_model, store_harness):
        """A counter that only appears once it fires cannot be used to rule the drop OUT."""
        env = _loop_env(loop_model, store_harness, [(1, 0)], rounds=1, max_active=2)
        G = loop_model[0]
        mh = G.GlmExpertLaneHost(loop_model[1], loop_model[2], [(1, 0)], claimable=env["claimable"])
        mh.register(*self._LIVE)
        _publish_claim(env, self._LIVE, "honest", wire_idx=1, host=mh, seed=9)
        logs = _drive_loop(env, ["honest"])
        assert _goodput(logs)["lineage_dead_pregate"] == 0
        assert env["store"].accepted(1)["accepted"], "\n".join(logs[-12:])

    def test_neutralising_the_pre_register_gate_brings_the_deadlock_straight_back(self, loop_model,
                                                                                 store_harness,
                                                                                 monkeypatch):
        """RED WITNESS. The tests above would also pass on a coordinator that simply never sees these
        records, so pin the mechanism: with the pre-registration verdict forced to "alive" (the v3.4.1
        order, where registration ran first) the measured deadlock returns EXACTLY -- dead coordinate on
        the seat, honest claim deferred, nothing merged."""
        import sharddiloco_glm_coordinator as C
        monkeypatch.setattr(C, "_terminal_lineage_verdict", lambda *a, **k: (False, "ok"))
        env, logs = self._run(loop_model, store_harness)
        host = env["host"]
        assert host.index_of(*self._DEAD[0]) == 1, "\n".join(logs[-14:])
        assert host.index_of(*self._LIVE) is None
        assert [ln for ln in logs if "DEFER" in ln], "\n".join(logs[-14:])
        assert env["store"].accepted(1) is None, "the honest miner should have earned nothing here"


@coordinator_only
class TestRetryableRecordsStillMergeThroughTheLoop:
    """The other half of the fix, end-to-end on the real loop: a record the early check could not
    condemn must still be picked up and merged. Dropping these would be worse than the squat -- the
    squat costs a seat, this would silently burn a miner's GPU hours."""

    @pytest.fixture(autouse=True)
    def _fast_idle(self, monkeypatch):
        monkeypatch.setenv("NEURAHASH_SD_IDLE_EXIT_S", "1")
        monkeypatch.setenv("NEURAHASH_SD_POLL_S", "0.05")

    def test_a_future_base_event_record_is_merged_once_the_clock_reaches_it(self, loop_model,
                                                                           store_harness):
        """`future-base-event` is a "not yet": at event 0 the r1 name is not even fetched, and after the
        event-0 record commits event 1 the SAME record must be discovered, judged and merged. It also
        exercises the undecidable path -- (2,3) is unregistered when it arrives, so its verdict rests on
        the prospective root, and it is registered only once it has been judged alive."""
        env = _loop_env(loop_model, store_harness, [(1, 0)], rounds=2, max_active=8)
        G = loop_model[0]
        mh = G.GlmExpertLaneHost(loop_model[1], loop_model[2], [(1, 0)], claimable=env["claimable"])
        _publish_claim(env, (1, 0), "now", wire_idx=0, host=mh, seed=11)
        mh.register(2, 3)
        late = _publish_claim(env, (2, 3), "later", base_event=1, wire_idx=1, host=mh, seed=12)
        assert late.startswith("cg/r1/"), late

        logs = _drive_loop(env, ["now", "later"])
        rec2 = env["store"].accepted(2)
        assert rec2 is not None and rec2["accepted"], \
            "the deferred-discovery record never merged:\n" + "\n".join(logs[-16:])
        assert (rec2["accepted"][0]["layer"], rec2["accepted"][0]["glm_expert"]) == (2, 3)
        assert N.accepted_names_me(rec2, "later") is True
        assert not [ln for ln in logs if "LINEAGE-DROP" in ln and "later" in ln], \
            "\n".join([ln for ln in logs if "LINEAGE-DROP" in ln][:4])
        assert _goodput(logs)["lineage_dead_pregate"] == 0


# ==================================================================================================
# F8 (CRITICAL, 2026-07-26): the ADVANCE killed the miner. Live 4060, processes p61 and p62, both
# rc=1 -- three published rounds on (L1,E45), PLATEAU -> RELEASE -> CLAIM the next coordinate, then
#   File "tools/sharddiloco_glm_expert.py", line 680, in train_glm_expert_contribution
#     out.loss.backward()
#   RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
#
# MECHANISM (measured, not assumed). LoRAExperts.forward injects the per-expert LoRA only for experts
# that appear in this batch's `top_k_index`, and the trunk is frozen, so if the claimed expert gets no
# token the loss depends on NOTHING that requires grad and backward() refuses it. The 4060 runs
# --batch 4 x seq 32 = 128 token positions x top-4 of 64 experts = ~512 draws over the 60 experts its
# pieces make routable, so a below-average expert missing an entire step is routine -- and the advance
# is what lands a miner on a fresh, routing-BLIND coordinate (next_claim_coord walks a wallet-hash
# permutation). It is NOT the inert-row trap: the coordinate the 4060 advanced to is (L1,E23)
# (next_claim_coord over its 60 claimable coords for wallet 0x6217c0CB...), and a later process on the
# same box trained and published that exact coordinate four times (coordinator events 48-53).
# ==================================================================================================
class _MiniLane:
    """The ContentLane surface `_run_async` calls, in-process, plus a SCRIPTED coordinator: every
    contribution the miner publishes is answered, at the next event, with an accepted record that
    ADVERTISES the miner's coordinate and does not accept its delta. That is a gate rejection -- the
    exact input `reject_streak` counts -- so `--advance-after` fires for real instead of being poked.

    Deliberately not tests/test_sd_async_lane._InProcStore: that module imports the coordinator, and
    this file must stay runnable from a contributor-only (public miner) checkout."""

    def __init__(self, host):
        import hashlib
        import neurahash.diloco_merge as dm
        self._sha, self._dm, self._host = hashlib, dm, host
        self._blobs, self._names, self._n = {}, {}, 0
        self.event, self.published = 0, []
        self._pointer(N.model_root(host))

    def _put(self, obj):
        body = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
        cid = self._sha.sha256(body).hexdigest()
        self._blobs[cid] = body
        return cid

    def _pointer(self, root):
        rounds = {"%d_%d" % (L, E): 0 for (L, E) in self._host.slots}
        self._names[N.GLM_POINTER_NAME] = self._put(
            self._dm.sd_pointer_encode(self.event, rounds, root, False))

    # -- ContentLane surface -----------------------------------------------------------------
    def read_pointer(self):
        return self.get_json(self._names[N.GLM_POINTER_NAME])

    def manifest(self):
        return {n: {"sha256": c} for n, c in self._names.items()}

    def get_json(self, cid):
        return json.loads(self._blobs[cid].decode())

    def get_delta(self, cid):
        raise AssertionError("no accepted record in this scenario carries a delta to fold")

    def put_delta(self, payload, name=None):
        self._n += 1
        return "cid%032d" % self._n

    def put_json_named(self, name, obj):
        cid = self._put(obj)
        self._names[name] = cid
        if "glm_expert" in obj:                       # a contribution -> the coordinator rejects it
            self.published.append(obj)
            self.event += 1
            key = "%d_%d" % (int(obj["layer"]), int(obj["glm_expert"]))
            self._names[N.accepted_name(self.event)] = self._put(
                {"event": self.event, "accepted": [], "slot_roots": {key: obj["base_slot_root"]},
                 "model_root": obj["base_root"]})
            self._pointer(obj["base_root"])
        return cid


def _starve_training_steps(model, layer, expert, state):
    """Make the next `state[n]` TRAINING forwards route NOTHING to `expert`, by pinning its router
    bias to -inf for the duration of each of those forwards (eval forwards are left alone). A faithful
    stand-in for a cold expert on a small batch, and TRANSIENT by construction, so the upfront -inf
    refusal cannot mask the defect under test. Returns the two hook handles.

    Hooked on the MoE block, not on `gate`: Glm4MoeLiteTopkRouter.forward only produces the router
    logits, and `e_score_correction_bias` is read one level up in Glm4MoeLiteMoE.route_tokens_to_experts
    -- a gate-level post-hook restores the bias before the selection that reads it, and silently does
    nothing at all."""
    import torch
    moe = model.model.layers[layer].mlp
    gate = moe.gate
    orig = float(gate.e_score_correction_bias[expert])

    def _pre(mod, args):
        if state.get("armed") and mod.training and state["n"] > 0:
            state["n"] -= 1
            state["starved"] += 1
            with torch.no_grad():
                gate.e_score_correction_bias[expert] = float("-inf")
            state["on"] = True

    def _post(mod, args, out):
        if state.pop("on", False):
            with torch.no_grad():
                gate.e_score_correction_bias[expert] = orig
        return out

    return moe.register_forward_pre_hook(_pre), moe.register_forward_hook(_post)


@pytest.fixture(scope="module")
def advance_model():
    """A tiny GLM used ONLY by the advance tests, so a folded delta or a half-trained LoRA from any
    other test in this file cannot change what the router does here."""
    import torch
    torch.set_num_threads(2)
    G = N._G()
    T = N.TINY
    model, cfg = G.build_tiny_glm(seed=T["seed"], vocab=T["vocab"], hidden=T["hidden"],
                                  inter=T["inter"], moe_inter=T["moe_inter"], layers=T["layers"],
                                  n_experts=T["n_experts"], topk=T["topk"])
    return G, model, cfg


class TestF8AdvanceOntoAColdCoordinate:
    """The miner must SURVIVE a training step whose batch routed nothing to the coordinate it just
    claimed, and it must refuse -- loudly, by coordinate -- a coordinate it could never train."""

    MINER = "advancer"
    INNER = 3
    BATCH = 4

    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch):
        monkeypatch.setattr(N, "_maybe_self_update", lambda log: None)
        monkeypatch.delenv("NEURAHASH_GLM_DATA_RESYNC", raising=False)

    def _drive(self, advance_model, starve_steps):
        """One full run: train+publish on (1,0) -> plateau -> claim the next coordinate -> train it,
        with the first `starve_steps` training step(s) on the NEW coordinate routed away from it."""
        G, model, cfg = advance_model
        coords = [(1, 0), (1, 1), (1, 2), (1, 3)]
        host = G.GlmExpertLaneHost(model, cfg, list(coords), claimable=list(coords))
        args = types.SimpleNamespace(
            mode="tiny", max_rounds=2, poll=0.0, round_wait=1e9, advance_after=1, garbage=False,
            inner=self.INNER, lora_r=4, lr=1e-3, batch=self.BATCH, outer=0.7, wire="lora",
            data_dir=".")
        lane = _MiniLane(host)
        nxt = N.next_claim_coord(list(host.slots), (1, 0), identity=self.MINER)
        assert nxt is not None and tuple(nxt) != (1, 0)
        state = {"armed": False, "n": starve_steps, "starved": 0}
        logs = []

        def log(*a):
            line = " ".join(str(x) for x in a)
            logs.append(line)
            if "PLATEAU" in line:
                state["armed"] = True                 # starve only the freshly claimed coordinate

        h1, h2 = _starve_training_steps(model, nxt[0], nxt[1], state)
        try:
            rc = N._run_async(args, lane, host, model, cfg, G, b"k" * 16, 0, 1, 0, self.MINER,
                              N.tiny_ids("train", slot=0)[:64], N.tiny_ids("val", slot=0)[:8],
                              N.TINY["seq"], log)
        finally:
            h1.remove()
            h2.remove()
        return rc, lane, logs, state, tuple(nxt), host

    def test_a_step_that_routes_nothing_to_the_new_coordinate_does_not_kill_the_miner(
            self, advance_model):
        """RED before the fix: RuntimeError('element 0 of tensors does not require grad and does not
        have a grad_fn') escapes _run_async and the process dies rc=1, exactly as the live 4060 did."""
        rc, lane, logs, state, nxt, _host = self._drive(advance_model, starve_steps=1)
        assert state["starved"] == 1, "the scenario never starved a step -- it would prove nothing"
        assert rc == 0, "\n".join(logs[-8:])
        assert len(lane.published) == 2, [p.get("glm_expert") for p in lane.published]
        second = lane.published[1]
        assert (second["layer"], second["glm_expert"]) == nxt, "round 2 is on the CLAIMED coordinate"
        assert second["steps"] == self.INNER - 1, \
            "the published step count must exclude the step that had no gradient to apply"
        assert second["tokens"] == (self.INNER - 1) * self.BATCH * N.TINY["seq"], \
            "tokens must exclude the skipped step too -- reward is weighted by steps AND tokens"
        assert [ln for ln in logs if "routed NO token" in ln], "the skip has to be visible in the log"

    def test_the_advance_really_happened(self, advance_model):
        """Guard on the SCENARIO: without a real plateau -> release -> claim there is no regression
        test here at all, only a training-loop unit test wearing its name."""
        _rc, _lane, logs, _state, nxt, host = self._drive(advance_model, starve_steps=1)
        plateau = [ln for ln in logs if "PLATEAU on (L1,E0)" in ln]
        assert plateau and ("CLAIM (L%d,E%d)" % nxt) in plateau[0], logs[:6]
        assert host.index_of(*nxt) is not None, "the claimed coordinate has to be registered"

    def test_a_coordinate_that_can_never_route_is_refused_by_name_not_in_backward(self,
                                                                                  advance_model):
        """The inert-row trap (piece_loader pins a NON-resident row's router bias to -inf). A miner
        must refuse the round and say WHICH coordinate, instead of dying inside autograd."""
        import torch
        G, model, cfg = advance_model
        gate = model.model.layers[1].mlp.gate
        keep = float(gate.e_score_correction_bias[2])
        with torch.no_grad():
            gate.e_score_correction_bias[2] = float("-inf")
        try:
            with pytest.raises(G.UnroutableExpert) as ei:
                G.train_glm_expert_contribution(model, cfg, 1, 2, N.tiny_ids("train", slot=0)[:32],
                                                N.tiny_ids("val", slot=0)[:8], H=2, r=4, lr=1e-3,
                                                batch=4)
        finally:
            with torch.no_grad():
                gate.e_score_correction_bias[2] = keep
        assert "(L1,E2)" in str(ei.value) and "INERT" in str(ei.value)

    def test_the_optimizer_holds_exactly_the_trainable_parameters(self, advance_model):
        """The invariant the crash violated in effect: after ANY claim the parameters the optimizer
        steps are exactly the ones requiring grad, and there is at least one. Asserted on the REAL
        wrap the trainer builds, then re-asserted through a whole contribution."""
        G, model, cfg = advance_model
        was = [(p, p.requires_grad) for p in model.parameters()]
        LoRAExperts = G._lora_experts_cls()
        layer = model.model.layers[1]
        base = layer.mlp.experts
        try:
            for p in model.parameters():
                p.requires_grad_(False)
            le = LoRAExperts(base, {3: 0}, r=4, alpha=8)
            layer.mlp.experts = le
            mine = list(le.params_for(0))
            live = [p for p in model.parameters() if p.requires_grad]
            assert mine, "a claim with nothing trainable cannot produce a gradient"
            assert {id(p) for p in live} == {id(p) for p in mine}, \
                "the optimizer would step parameters the loss does not reach (or miss ones it does)"
        finally:
            layer.mlp.experts = base
            for p, flag in was:
                p.requires_grad_(flag)
        out = G.train_glm_expert_contribution(model, cfg, 1, 3, N.tiny_ids("train", slot=0)[:32],
                                              N.tiny_ids("val", slot=0)[:8], H=2, r=4, lr=1e-3,
                                              batch=4)
        assert (out["steps_trained"], out["steps_skipped"]) == (2, 0)

    def test_the_defensive_unwrap_can_actually_recognise_a_leaked_wrapper(self, advance_model):
        """_lora_experts_cls used to mint a NEW class object per call, so the documented unwrap
        (`while isinstance(base, LoRAExperts)`) could never match a wrapper left by an EARLIER call --
        i.e. by any leak that could really happen (memory v332-oom-death-and-resume-verdict)."""
        G = advance_model[0]
        assert G._lora_experts_cls() is G._lora_experts_cls()


# ==================================================================================================
# NEVER-BLOCK V0 (docs/NEVER_BLOCK_HANDOVER.md section 0-PRE + 7.1). THE MEASURED INCIDENT:
# 2026-07-25 the 5090 plateaued on (L1,E50), released, claimed (L1,E0) -- and BLOCKED 23 MINUTES.
# Process alive, "Responding: True", GPU 9%, nothing in any log. scratchpad/wan_miner5090.log:113 is
# the last line that process ever wrote:
#   [glm-contrib glm-1325009E] PLATEAU on (L1,E50) after 3 consecutive rejects -> RELEASE,
#   CLAIM (L1,E0) [local slot 0]. Sweeping: 60 coordinate(s) claimable here.
# There is NO `post-advance catch-up` line after it, while the three earlier advances in that same
# run all printed one -- so it entered resume_to_root and never came out of a SINGLE call. That is
# the refutation of the original "linear full-history replay" theory (0-PRE): the completed
# catch-ups had tried only 7-17 records each.
# ==================================================================================================
class _StallingLane(_MiniLane):
    """`_MiniLane` whose lane fetches BLOCK FOREVER while the miner is inside post-advance catch-up
    for one nominated coordinate -- the incident, reproduced in-process and without CUDA.

    The stall is armed by the caller flipping `in_resume`, so ONLY the catch-up path stalls: the
    ordinary per-iteration `catch_up_accepted` scan keeps working, exactly as it did live (the miner
    had been training and publishing happily for 40 rounds before this)."""

    def __init__(self, host, stall_coords):
        super().__init__(host)
        self.stall_coords = {tuple(c) for c in stall_coords}
        self.in_resume, self.resume_coord = False, None
        self.stalled_calls = 0
        self.never = threading.Event()          # NEVER set while the miner is running

    def _pointer(self, root):
        """A SHARD-CLAIM coordinator: it holds a coordinate we do NOT (`9_9`), so its global
        model_root is unreachable for us by construction. That is the live 60-coordinate
        configuration (scratchpad/wan_miner5090.log: "Sweeping: 60 coordinate(s) claimable here"),
        and it is what makes resume_to_root's `model_root(host) == target_root` early return
        (sharddiloco_glm_contributor.py:2525) unable to fire -- the runs that DID early-return had
        only 5 claimable coordinates and therefore the coordinator's exact slot set."""
        rounds = {"%d_%d" % (L, E): 0 for (L, E) in self._host.slots}
        rounds["9_9"] = 1
        self._names[N.GLM_POINTER_NAME] = self._put(self._dm.sd_pointer_encode(
            self.event, rounds, "coordinator-global-root-%d" % self.event, False))

    def release(self):
        """Let the abandoned worker threads die at teardown (they are daemons, but tidy is cheap)."""
        self.never.set()

    def _maybe_stall(self):
        if self.in_resume and tuple(self.resume_coord or ()) in self.stall_coords:
            self.stalled_calls += 1
            self.never.wait()                   # this call NEVER returns

    def manifest(self):
        self._maybe_stall()
        return super().manifest()

    def get_json(self, cid):
        self._maybe_stall()
        return super().get_json(cid)

    def get_delta(self, cid):
        self._maybe_stall()
        return super().get_delta(cid)


class TestV0NeverBlockOnCatchUp:
    """THE GOAL METRIC, stated as a bound: a catch-up network call that never returns must not stop
    the miner from starting a training round on ANOTHER coordinate.

    RED on the pre-fix code: `resume_to_root` calls `lane.manifest()` straight through, so
    `_run_async` never returns and this test has to be KILLED (measured: hung until a 180 s
    subprocess timeout, 0 rounds published after the plateau).
    GREEN after: the per-call deadline fires, the wall budget rolls back fail-closed, the coordinate
    goes on cooldown, the walk advances, and round 2 is published on a DIFFERENT coordinate."""

    MINER = "neverblock0"
    COORDS = [(1, 0), (1, 1), (1, 2), (1, 3)]

    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch):
        monkeypatch.setattr(N, "_maybe_self_update", lambda log: None)
        monkeypatch.delenv("NEURAHASH_GLM_DATA_RESYNC", raising=False)
        # Small budgets so the TEST is fast; the bound under test is "some finite budget", not 180 s.
        monkeypatch.setenv("NEURAHASH_SD_CATCHUP_BUDGET_S", "2.0")
        monkeypatch.setenv("NEURAHASH_SD_CATCHUP_CALL_TIMEOUT_S", "1.0")
        monkeypatch.setenv("NEURAHASH_SD_CATCHUP_STALL_S", "2.0")
        monkeypatch.setenv("NEURAHASH_SD_COORD_COOLDOWN_S", "600")
        monkeypatch.setenv("NEURAHASH_SD_COORD_COOLDOWN_EVENTS", "10")

    def _drive(self, advance_model, monkeypatch):
        G, model, cfg = advance_model
        host = G.GlmExpertLaneHost(model, cfg, list(self.COORDS), claimable=list(self.COORDS))
        args = types.SimpleNamespace(
            mode="tiny", max_rounds=2, poll=0.0, round_wait=1e9, advance_after=1, garbage=False,
            inner=2, lora_r=4, lr=1e-3, batch=4, outer=0.7, wire="lora", data_dir=".")
        blocked = N.next_claim_coord(list(self.COORDS), (1, 0), identity=self.MINER)
        assert blocked is not None and tuple(blocked) != (1, 0)
        lane = _StallingLane(host, [tuple(blocked)])
        # Arm the stall for exactly the window the miner is inside catch-up. The REAL resume_to_root
        # still runs -- this only tells the fake lane when the call is coming from it.
        real_resume = N.resume_to_root

        def _traced(h, ln, *a, **kw):
            lane.in_resume, lane.resume_coord = True, kw.get("own_coord")
            try:
                return real_resume(h, ln, *a, **kw)
            finally:
                lane.in_resume = False
        monkeypatch.setattr(N, "resume_to_root", _traced)
        logs = []
        t0 = time.monotonic()
        try:
            rc = N._run_async(args, lane, host, model, cfg, G, b"k" * 16, 0, 1, 0, self.MINER,
                              N.tiny_ids("train", slot=0)[:32], N.tiny_ids("val", slot=0)[:8],
                              N.TINY["seq"], logs.append)
        finally:
            lane.release()
        return rc, lane, logs, tuple(blocked), time.monotonic() - t0

    def test_a_catchup_call_that_never_returns_does_not_stop_the_next_training_round(
            self, advance_model, monkeypatch):
        rc, lane, logs, blocked, elapsed = self._drive(advance_model, monkeypatch)
        assert lane.stalled_calls >= 1, "the scenario never stalled a call -- it would prove nothing"
        assert rc == 0, "\n".join(logs[-10:])
        assert len(lane.published) == 2, [p.get("glm_expert") for p in lane.published]
        second = (lane.published[1]["layer"], lane.published[1]["glm_expert"])
        assert second != blocked, "the miner trained the coordinate whose catch-up hung"
        assert second != (1, 0), "the miner never actually advanced off its plateaued coordinate"
        assert elapsed < 120.0, "the whole run must be bounded, not merely finite (%.1fs)" % elapsed

    def test_the_blocked_coordinate_is_named_and_put_on_cooldown(self, advance_model, monkeypatch):
        """A miner that silently skips a coordinate is the same outage with nicer logs. The abort
        reason and the cooldown have to be visible."""
        _rc, _lane, logs, blocked, _el = self._drive(advance_model, monkeypatch)
        hit = [ln for ln in logs if "COOLDOWN" in ln and ("L%d,E%d" % blocked) in ln]
        assert hit, "\n".join(logs[-10:])
        assert any(("catch-up" in ln and ("budget" in ln or "timeout" in ln)) for ln in logs), \
            "the abort reason must say WHY, not just that it happened"


class _HangLane:
    """Every named call blocks until `release()`. The 23-minute incident, in one object."""

    def __init__(self, hang=("manifest",), inner=None):
        self.hang = set(hang)
        self.never = threading.Event()
        self.calls = []
        self._inner = inner or {}

    def release(self):
        self.never.set()

    def __getattr__(self, name):
        def _call(*a, **kw):
            self.calls.append(name)
            if name in self.hang:
                self.never.wait()
            v = self._inner.get(name)
            return v(*a, **kw) if callable(v) else v
        return _call


class _FakeClock:
    """Injected monotonic clock: the budget/stall rules are TIME rules, and a test that sleeps to
    exercise them is a slow flaky test that proves the same thing."""

    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t


class TestV0CatchUpBounds:
    """resume_to_root's three bounds, each in isolation. All of them exit through the SAME fail-closed
    rollback the function already had -- the point of V0 is that nothing about lineage safety changes,
    only that the wait is finite."""

    @staticmethod
    def _host():
        return _CoordFakeHost([(1, 0)])

    def test_a_manifest_that_never_returns_is_abandoned_not_awaited(self):
        lane = _HangLane(hang=("manifest",))
        out, said = {}, []
        try:
            t0 = time.monotonic()
            applied, reached = N.resume_to_root(self._host(), lane, "target", said.append,
                                                own_coord=(1, 0), budget_s=5.0,
                                                call_timeout_s=0.3, outcome=out)
            elapsed = time.monotonic() - t0
        finally:
            lane.release()
        assert (applied, reached) == (0, False)
        assert out["reason"] == "call-timeout" and out["aborted"] is True
        assert elapsed < 3.0, "the call was awaited, not bounded (%.1fs)" % elapsed
        assert any("BLOCKED reading the manifest" in s for s in said), said

    def test_a_record_fetch_that_never_returns_rolls_back_fail_closed(self):
        host = self._host()
        real = _CoordFakeLane({1: {"1_0": "mine1"}})
        lane = _HangLane(hang=("get_json",), inner={"manifest": real.manifest})
        out = []
        try:
            applied, reached = N.resume_to_root(host, lane, "target", lambda *_a: None,
                                                own_coord=(1, 0), budget_s=5.0, call_timeout_s=0.3,
                                                outcome=(o := {}))
            out.append(o)
        finally:
            lane.release()
        assert (applied, reached) == (0, False)
        assert out[0]["reason"] == "call-timeout"
        assert host.writes == len(host.slots), "fail-closed: every slot restored, exactly as before"

    def test_the_wall_budget_aborts_a_replay_that_is_moving_but_too_slow(self, monkeypatch):
        """Option A of design 1.2: some replays make progress forever. B_wall is the backstop.

        V0.1: the records must ADVERTISE our coordinate, or the applicability filter skips them for
        free and there is no slow replay left to bound. Each folds cleanly but never moves our slot
        root onto the advertised value, so the loop keeps going -- which is exactly "moving but too
        slow"."""
        clk = _FakeClock()
        state = {"global": "base"}
        TestResumeToOwnCoordinate._patch(monkeypatch, state)

        def slow_fold(*_a, **_kw):
            clk.t += 10.0                                   # 10 s of wall clock per record
            return True, "ok", []                           # folds, but moves nothing -> never reached
        monkeypatch.setattr(N, "_fold_accepted_checked", slow_fold)
        host = _CoordFakeHost([(1, 0)])
        lane = _CoordFakeLane({e: {"1_0": "mine%d" % e} for e in range(1, 9)})
        out = {}
        applied, reached = N.resume_to_root(host, lane, "unreachable", lambda *_a: None,
                                            own_coord=(1, 0), budget_s=25.0, stall_s=1e9,
                                            now=clk, outcome=out)
        assert (applied, reached) == (0, False)
        assert out["reason"] == "budget" and out["aborted"] is True
        assert out["records"] == 3, "aborted at the first check past 25 s, not after all 8 records"
        assert host.writes == len(host.slots), "fail-closed rollback, unchanged"

    def test_the_no_fold_stall_aborts_even_inside_the_wall_budget(self, monkeypatch):
        """Option B: a replay whose records all fail to fold is not progress, however long we wait."""
        clk = _FakeClock()
        monkeypatch.setattr(N, "model_root", lambda h: "ours")
        monkeypatch.setattr(N, "slot_root", lambda h, i: "ours")

        def dead_fold(*_a, **_kw):
            clk.t += 10.0
            return False, "lineage", []                     # rolled back inside; nothing folded
        monkeypatch.setattr(N, "_fold_accepted_checked", dead_fold)
        host = _CoordFakeHost([(1, 0)])
        lane = _CoordFakeLane({e: {"1_0": "mine%d" % e} for e in range(1, 9)})
        out = {}
        applied, reached = N.resume_to_root(host, lane, "unreachable", lambda *_a: None,
                                            own_coord=(1, 0), budget_s=1e9, stall_s=25.0,
                                            now=clk, outcome=out)
        assert (applied, reached) == (0, False)
        assert out["reason"] == "stall" and out["records"] == 0

    def test_a_healthy_replay_is_not_touched_by_any_of_the_bounds(self, monkeypatch):
        """The bounds must be invisible on the happy path, or they trade one outage for another."""
        state = {"global": "base"}
        TestResumeToOwnCoordinate._patch(monkeypatch, state)
        host = _CoordFakeHost([(1, 0)])
        lane = _CoordFakeLane({1: {"1_0": "mine1"}, 2: {"2_3": "theirs"}, 3: {"1_0": "mine2"}})
        out = {}
        applied, reached = N.resume_to_root(host, lane, "unreachable-global", lambda *_a: None,
                                            own_coord=(1, 0), outcome=out)
        assert (applied, reached) == (2, True)              # V0.1: event 2 (2_3) is not ours to fold
        assert out["reason"] == "ok" and out["aborted"] is False and host.writes == 0

    def test_the_deadline_also_covers_the_delta_fetch_inside_the_fold(self):
        """apply_accepted re-fetches every accepted delta by CID (`lane.get_delta`), so bounding only
        get_json/manifest would leave the longest fetch on the path unbounded. _DeadlineLane wraps the
        OBJECT for exactly that reason."""
        lane = _HangLane(hang=("get_delta",))
        bounded = N._DeadlineLane(lane, 0.3)
        try:
            with pytest.raises(N.CatchupTimeout):
                bounded.get_delta("cid")
        finally:
            lane.release()
        assert bounded.abandoned == 1

    def test_the_deadline_never_outlives_the_remaining_wall_budget(self):
        clk = _FakeClock()
        lane = _HangLane(hang=("manifest",))
        bounded = N._DeadlineLane(lane, 60.0, deadline=clk() + 5.0, now=clk)
        clk.t += 6.0                                        # budget already spent
        try:
            with pytest.raises(N.CatchupTimeout):
                bounded.manifest()
        finally:
            lane.release()
        assert lane.calls == [], "no call should even be started once the budget is gone"

    def test_the_live_lane_is_never_mutated_by_tightening_its_socket_timeout(self):
        """The main loop's own manifest scan shares the lane object; tightening it in place would
        change a call this function does not own."""
        lane = types.SimpleNamespace(timeout=30.0, retries=6, ping=lambda: "pong")
        bounded = N._DeadlineLane(lane, 5.0)
        assert bounded.ping() == "pong"
        assert (lane.timeout, lane.retries) == (30.0, 6), "the shared lane object was mutated"
        assert bounded._lane is not lane and bounded._lane.timeout == 5.0
        assert bounded._lane.retries == 2, "an abandoned call must die on its own, not retry 6 times"

    def test_the_knobs_are_env_tunable_and_fail_soft(self, monkeypatch):
        monkeypatch.setenv("NEURAHASH_SD_CATCHUP_BUDGET_S", "42.5")
        monkeypatch.setenv("NEURAHASH_SD_CATCHUP_CALL_TIMEOUT_S", "not-a-number")
        monkeypatch.delenv("NEURAHASH_SD_CATCHUP_STALL_S", raising=False)
        assert N._catchup_budget_s() == 42.5
        assert N._catchup_call_timeout_s() == N.CATCHUP_CALL_TIMEOUT_S, "garbage must not kill a miner"
        assert N._catchup_stall_s() == N.CATCHUP_STALL_S


class TestCoordCooldown:
    """15 minutes OR 10 events, whichever is LATER (design 1.4). Both halves have to elapse: a quiet
    lane must not expire a cooldown on wall clock alone, and a fast one must not expire it on events
    alone."""

    def _cd(self):
        clk = _FakeClock()
        return N.CoordCooldown(seconds=900.0, events=10, now=clk), clk

    def test_wall_clock_alone_does_not_release_it(self):
        cd, clk = self._cd()
        cd.park((1, 5), event=100, reason="catch-up budget")
        clk.t += 901.0
        assert cd.blocked((1, 5), 100) is True, "10 events have not passed"
        assert cd.blocked((1, 5), 110) is False

    def test_events_alone_do_not_release_it(self):
        cd, clk = self._cd()
        cd.park((1, 5), event=100, reason="catch-up stall")
        assert cd.blocked((1, 5), 999) is True, "15 minutes have not passed"
        clk.t += 900.0
        assert cd.blocked((1, 5), 999) is False

    def test_an_unparked_coordinate_is_never_blocked_and_the_reason_survives(self):
        cd, _clk = self._cd()
        assert cd.blocked((1, 6), 0) is False and cd.reason((1, 6)) is None
        cd.park((1, 6), event=3, reason="catch-up call-timeout")
        assert cd.reason((1, 6)) == "catch-up call-timeout"
        s, e = cd.left((1, 6), 3)
        assert (round(s), e) == (900, 10)

    def test_describe_names_every_still_parked_coordinate_for_the_heartbeat(self):
        cd, _clk = self._cd()
        cd.park((1, 1), event=0, reason="catch-up budget")
        lines = cd.describe([(1, 0), (1, 1)], 0)
        assert len(lines) == 1 and "(L1,E1)" in lines[0] and "catch-up budget" in lines[0]
        assert lines[0].isascii(), "cp1252 console: log lines stay ASCII"


class TestAdvanceClaimWalksPastBlockedCoordinates:
    """1.4's fall-through guarantee, as a unit: the walk parks what blocks it and keeps going, and
    reports None only when a WHOLE pass found nothing -- which is the 1.5 repair state, not a stall."""

    COORDS = [(1, 0), (1, 1), (1, 2), (1, 3)]

    class _Host:
        def __init__(self, coords, refuse=()):
            self.slots = [tuple(c) for c in coords]
            self.refuse = {tuple(c) for c in refuse}
            self.registered = []

        def index_of(self, L, E):
            t = (int(L), int(E))
            return self.slots.index(t) if t in self.slots else None

        def register(self, L, E):
            if (int(L), int(E)) in self.refuse:
                raise RuntimeError("max_active_slots=1 reached; cannot admit (L%d,E%d) yet" % (L, E))
            self.registered.append((int(L), int(E)))
            return self.index_of(L, E)

    def _walk(self, monkeypatch, outcomes, refuse=(), cooldown=None, logs=None):
        """outcomes: coord -> the `reason` resume_to_root should report for it."""
        host = self._Host(self.COORDS, refuse=refuse)

        def fake_resume(_h, _l, _root, _log, own_coord=None, outcome=None, **_kw):
            reason = outcomes.get(tuple(own_coord), "unreachable")
            if outcome is not None:
                outcome.update(reason=reason, elapsed_s=1.0, records=0,
                               aborted=reason in N._CATCHUP_ABORTED)
            return 0, reason == "ok"
        monkeypatch.setattr(N, "resume_to_root", fake_resume)
        cd = cooldown or N.CoordCooldown(seconds=900.0, events=10, now=_FakeClock())
        got = N.advance_claim(host, object(), list(self.COORDS), (1, 0), "wallet0", None, "root", 7,
                              cd, (logs if logs is not None else []).append, "m0")
        return got, host, cd

    def test_a_blocked_candidate_is_parked_and_the_walk_continues(self, monkeypatch):
        order = [c for c in N.claim_walk_order(self.COORDS, "wallet0") if c != (1, 0)]
        logs = []
        got, host, cd = self._walk(monkeypatch, {order[0]: "budget"}, logs=logs)
        assert got is not None and got[0] == order[1], "the walk stopped at the blocked coordinate"
        assert cd.blocked(order[0], 7) is True and "catch-up budget" in cd.reason(order[0])
        assert cd.blocked(order[1], 7) is False, "a coordinate we landed on must not be parked"
        assert any("COOLDOWN" in ln for ln in logs)

    def test_an_honest_unreachable_root_is_NOT_a_block(self, monkeypatch):
        """The pre-existing semantics: unreachable -> rolled back to the frozen base and we train
        anyway, contributions lineage-dropped. Parking that would be a behaviour change, not a fix."""
        order = [c for c in N.claim_walk_order(self.COORDS, "wallet0") if c != (1, 0)]
        got, _host, cd = self._walk(monkeypatch, {order[0]: "unreachable"})
        assert got[0] == order[0] and got[3] is False
        assert cd.blocked(order[0], 7) is False

    def test_a_refused_registration_is_parked_too_and_never_crashes_the_walk(self, monkeypatch):
        order = [c for c in N.claim_walk_order(self.COORDS, "wallet0") if c != (1, 0)]
        got, host, cd = self._walk(monkeypatch, {}, refuse=[order[0]])
        assert got[0] == order[1] and order[0] not in host.registered
        assert "register refused" in cd.reason(order[0])

    def test_every_candidate_blocked_reports_the_repair_state_rather_than_looping(self, monkeypatch):
        order = [c for c in N.claim_walk_order(self.COORDS, "wallet0") if c != (1, 0)]
        got, _host, cd = self._walk(monkeypatch, {c: "call-timeout" for c in order})
        assert got is None, "a full pass with nothing startable is the 1.5 repair state"
        assert all(cd.blocked(c, 7) for c in order), "every one of them has to be parked"

    def test_with_nothing_blocked_the_walk_lands_exactly_where_next_claim_coord_says(self,
                                                                                     monkeypatch):
        """Equivalence with the previous behaviour: when no coordinate is parked, advance_claim's
        first (and only) candidate is next_claim_coord's answer. Without this the never-block change
        could silently re-order the sweep, which is a different feature with its own measured
        history (the 5090/4060 permanent collision, next_claim_coord's docstring)."""
        got, _host, _cd = self._walk(monkeypatch, {})
        want = N.next_claim_coord(list(self.COORDS), (1, 0), identity="wallet0")
        assert got[0] == tuple(want)

    def test_a_single_claimable_coordinate_still_means_stay_put(self, monkeypatch):
        monkeypatch.setattr(N, "resume_to_root", lambda *a, **k: (0, True))
        cd = N.CoordCooldown(seconds=1.0, events=1, now=_FakeClock())
        assert N.advance_claim(self._Host([(1, 0)]), object(), [(1, 0)], (1, 0), "w", None, "r", 0,
                               cd, lambda *_a: None, "m0") is None

    def test_the_loop_refuses_to_train_while_every_coordinate_is_blocked(self):
        """1.5(b) over 1.5(a), stated in the source: training a base we KNOW is off-lineage is the
        ~900-rounds-paid-for-nothing failure with better uptime."""
        import inspect
        src = inspect.getsource(N._run_async)
        assert "REPAIR MODE" in src and "repair_since" in src
        i_rep = src.index("if repair_since is not None:")
        i_train = src.index("_vram_pause_if_starved(log, miner=miner)")
        assert i_rep < i_train, "the repair gate must come BEFORE the training step, or it is decoration"


class _CountingFoldHost(_FoldHost):
    """_FoldHost that counts read_slot, which is exactly the snapshot cost under test."""

    def __init__(self, coords, shape=(2, 3)):
        super().__init__(coords, shape=shape)
        self.reads = 0

    def read_slot(self, i):
        self.reads += 1
        return super().read_slot(i)


class TestCowRollbackIsExactAndCheap:
    """_fold_accepted_checked used to deep-copy EVERY resident slot before every fold so it could roll
    back an off-lineage record. At the 60-coordinate residency that is ~1.05 GiB per record (one
    coordinate's canonical {gate,up,down} triple is 18,874,493 B). Copy-on-write pays for the slots the
    record actually moves instead -- but ONLY if the restore stays byte-identical, because that
    rollback is the fail-closed guarantee that keeps un-gated weights out of the base."""

    SHAPE = (4, 5)

    def _loaded(self, n, cls=_FoldHost):
        """A host whose slots all hold DISTINCT non-zero weights, so a missed restore cannot hide in
        a field of zeros."""
        coords = [(1, e) for e in range(n)]
        host = cls(coords, shape=self.SHAPE)
        rng = np.random.default_rng(11)
        for j in range(n):
            host.write_slot(j, {k: rng.standard_normal(self.SHAPE).astype(np.float32)
                                for k in ("gate", "up", "down")})
        return host, coords

    @staticmethod
    def _snapshot(host, n):
        return {j: {k: v.copy() for k, v in host.read_slot(j).items()} for j in range(n)}

    @staticmethod
    def _offlineage(coord, cid="c0"):
        """A record that MOVES `coord` but advertises a root our replay cannot produce -> lineage fail
        -> rollback. This is the only path that rolls back, so it is the only one worth proving."""
        return dict(accepted=[_row("m", coord, cid, slot=0)],
                    slot_roots={"%d_%d" % coord: "a-root-our-fold-will-not-produce"},
                    model_root="g")

    def test_the_rollback_is_byte_identical_across_every_slot(self):
        host, coords = self._loaded(6)
        before = self._snapshot(host, len(coords))
        lane = _FoldLane({"c0": _fold_shape(self.SHAPE, seed=5)})
        ok, reason, _rej = N._fold_accepted_checked(host, lane, self._offlineage((1, 2)), None, -1,
                                                    log=None)
        assert (ok, reason) == (False, "lineage")
        for j in range(len(coords)):
            now = host.read_slot(j)
            for k in ("gate", "up", "down"):
                assert np.array_equal(now[k], before[j][k]), "slot %d key %s was not restored" % (j, k)
                assert now[k].dtype == before[j][k].dtype

    def test_that_assertion_is_load_bearing_the_fold_really_moved_the_slot(self, monkeypatch):
        """Positive control: with the restore disabled the same test FAILS, so a green run above is
        evidence of a working rollback and not of a fold that never wrote anything."""
        host, coords = self._loaded(6)
        before = self._snapshot(host, len(coords))
        monkeypatch.setattr(N._CowSlots, "rollback", lambda self: 0)
        lane = _FoldLane({"c0": _fold_shape(self.SHAPE, seed=5)})
        N._fold_accepted_checked(host, lane, self._offlineage((1, 2)), None, -1, log=None)
        assert not np.array_equal(host.read_slot(2)["gate"], before[2]["gate"])
        assert np.array_equal(host.read_slot(3)["gate"], before[3]["gate"]), \
            "only the moved slot should differ -- otherwise this control proves nothing"

    def test_the_snapshot_cost_no_longer_scales_with_residency(self):
        """The measured win: identical work on a 5-slot and a 60-slot host. Before this change the
        60-slot fold deep-copied 60 slots (60 x 18,874,493 B = 1.05 GiB on a real GLM); now it copies
        the one slot the record moves."""
        small, _c = self._loaded(5, cls=_CountingFoldHost)
        big, _c2 = self._loaded(60, cls=_CountingFoldHost)
        small.reads = big.reads = 0
        lane = _FoldLane({"c0": _fold_shape(self.SHAPE, seed=5)})
        for h in (small, big):
            N._fold_accepted_checked(h, lane, self._offlineage((1, 2)), None, -1, log=None)
        assert big.reads == small.reads, "the snapshot still scales with residency (%d vs %d)" % (
            big.reads, small.reads)
        assert big.reads <= 6, "one moved slot should cost a handful of reads, got %d" % big.reads

    def test_a_record_for_a_coordinate_we_do_not_hold_copies_nothing_at_all(self):
        """The common case on a shard-claim network: 59 of 60 records move somebody else's expert."""
        host, _c = self._loaded(60, cls=_CountingFoldHost)
        host.reads = 0
        rec = dict(accepted=[_row("stranger", (7, 7), "cX", slot=9)], slot_roots={"7_7": "theirs"},
                   model_root="g")
        ok, reason, _rej = N._fold_accepted_checked(host, _FoldLane({}), rec, None, -1, log=None)
        assert (ok, reason) == (True, "ok")
        assert host.reads == 0, "nothing resident moved, so nothing should have been copied"

    def test_a_fold_that_RAISES_part_way_through_is_also_rolled_back(self):
        """FAIL-CLOSED on a mid-fold failure. apply_accepted folds the accepted rows one at a time, so
        a lane fetch that errors (or, since never-block V0, exceeds its deadline) on row 2 used to
        leave row 1 in the base with no rollback whatsoever -- an unverified, off-lineage write. The
        exception must still reach the caller; only the weights are restored."""
        host, coords = self._loaded(4)
        before = self._snapshot(host, len(coords))
        d = _fold_shape(self.SHAPE, seed=5)

        class _HalfLane(_FoldLane):
            def get_delta(self, cid):
                if cid == "cBOOM":
                    raise N.CatchupTimeout("lane.get_delta did not return within 90.0s")
                return super().get_delta(cid)
        rec = dict(accepted=[_row("m", (1, 1), "c0", slot=0), _row("m", (1, 2), "cBOOM", slot=1)],
                   slot_roots={"1_1": "x", "1_2": "y"}, model_root="g")
        with pytest.raises(N.CatchupTimeout):
            N._fold_accepted_checked(host, _HalfLane({"c0": d}), rec, None, -1, log=None)
        for j in range(len(coords)):
            for k in ("gate", "up", "down"):
                assert np.array_equal(host.read_slot(j)[k], before[j][k]), \
                    "slot %d survived a mid-fold failure un-rolled-back" % j

    def test_a_clean_fold_still_keeps_its_result(self):
        """Guard the other direction: copy-on-write must not roll back a record that VALIDATES."""
        coords = [(1, 0), (1, 1)]
        d = _fold_shape(self.SHAPE, seed=9)
        host = _FoldHost(coords, shape=self.SHAPE)
        want, _h = None, None
        scratch = _FoldHost(coords, shape=self.SHAPE)
        cur = scratch.read_slot(1)
        scratch.write_slot(1, {k: cur[k] + 0.7 * d[k] for k in cur})
        want = N.slot_root(scratch, 1)
        rec = dict(accepted=[_row("m", (1, 1), "c0", slot=1)], slot_roots={"1_1": want},
                   model_root="g")
        ok, reason, _rej = N._fold_accepted_checked(host, _FoldLane({"c0": d}), rec, None, -1,
                                                    log=None)
        assert (ok, reason) == (True, "ok") and N.slot_root(host, 1) == want


class _CostLane(_CoordFakeLane):
    """_CoordFakeLane wearing the LIVE lane's MEASURED cost structure on an injected clock.

    Every number here is anchored, not invented:
      * MANIFEST_S -- lane.manifest() was measured at 23.79 s over an 11,051-object store (memory
        glm-lane-manifest-throughput-bound, quoted at sharddiloco_glm_coordinator.py:339); the live
        store is at 23,503 objects / 13.9 GB (scratchpad/wan_coord.log:4), so ~50 s.
      * GET_JSON_S -- 0.06 s, measured in the same pass (sharddiloco_glm_coordinator.py:340).
      * FOLD_S -- one _fold_accepted_checked: a 278,731 B delta fetch over WAN + the _CowSlots copy +
        the slot_root hash + (on a running miner) the rollback. 1.0 s is the conservative end.
    That is what makes the elapsed numbers below comparable to the 55.9-92.8 s measured live."""

    MANIFEST_S = 50.0
    GET_JSON_S = 0.06
    FOLD_S = 1.0

    def __init__(self, adv, clk):
        super().__init__(adv)
        self.clk = clk
        self.manifests = 0

    def manifest(self):
        self.manifests += 1
        self.clk.t += self.MANIFEST_S
        return super().manifest()

    def get_json(self, sha):
        self.clk.t += self.GET_JSON_S
        return super().get_json(sha)


class TestNothingToFoldIsNotAStall:
    """V0.1. MEASURED LIVE 2026-07-26, both miners, 12 of 12 advances:

        resume: ABORTED catch-up for 1_50 after 61.8s (stall, 0 record(s) folded, budget=180s
        stall=30s call-timeout=90s); rolled back to the frozen base
        COOLDOWN (L1,E50): catch-up stall after 61.8s -- parked for 900s / 10 event(s)

    Every one said `0 record(s) folded`, and not one said `call-timeout` -- so no single fetch was
    hung; the miner was paying ~50 s for a second whole-store manifest read plus the FULL 30 s stall
    window to discover that no accepted record has anything to do with the coordinate it just
    claimed. The coordinator advertises exactly ONE coordinate per accepted record
    (sharddiloco_glm_coordinator.py:1974), so `slot_roots` answers that question for 0.06 s a record
    -- V0's rule instead answered it by FOLDING each record and rolling it back, which on a running
    miner (whose base the main loop has already advanced) fails every time and so never resets the
    no-fold clock. Correct, complete and EMPTY was charged like BLOCKED, then parked for 900 s.

    The bounds themselves are unchanged and re-asserted below: applicable records that fail still
    stall, park and walk on; unreachable-but-non-empty still trains through and is lineage-dropped."""

    # 79 accepted records: the live coordinator was at base_event 70-79 when the aborts were logged
    # (scratchpad/wan_miner5090.log:296-312), and NONE of them advertises the coordinate we claim.
    N_RECORDS = 79
    OURS = (1, 50)

    def _empty_lane(self, clk):
        """Records for OTHER coordinates only -- the live `no record advertised it` case."""
        return _CostLane({e: {"1_%d" % (12 + e % 5): "theirs%d" % e}
                          for e in range(1, self.N_RECORDS + 1)}, clk)

    @staticmethod
    def _dead_fold(clk, monkeypatch):
        """A running miner's base has already folded the live records, so re-folding one fails
        `replica_root_ok` and rolls back -- costly, and it never resets the no-fold clock."""
        def fold(*_a, **_kw):
            clk.t += _CostLane.FOLD_S
            return False, "lineage", []
        monkeypatch.setattr(N, "_fold_accepted_checked", fold)
        return fold

    def test_a_coordinate_with_nothing_to_fold_completes_instead_of_stalling(self, monkeypatch):
        """THE GOAL METRIC. Same scenario that produced `61.8s (stall, 0 record(s) folded)` live."""
        clk = _FakeClock()
        monkeypatch.setattr(N, "model_root", lambda h: "ours")
        monkeypatch.setattr(N, "slot_root", lambda h, i: "ours")
        self._dead_fold(clk, monkeypatch)
        host = _CoordFakeHost([self.OURS])
        lane = self._empty_lane(clk)
        man = lane.manifest()                                # the caller's manifest, as live
        clk.t = 0.0                                          # time the catch-up, not the caller
        out, said = {}, []
        applied, reached = N.resume_to_root(host, lane, "target", said.append, own_coord=self.OURS,
                                            budget_s=180.0, stall_s=30.0, now=clk, outcome=out,
                                            man=man)
        elapsed = out["elapsed_s"]
        assert out["reason"] == "empty", "a catch-up with nothing to fold is COMPLETE, not stalled"
        assert out["aborted"] is False, "empty must never read as BLOCKED -> never a cooldown"
        assert (applied, reached) == (0, False)
        assert out["applicable"] == 0 and out["scanned"] == self.N_RECORDS, out
        # V0 spent MANIFEST_S + the whole 30 s stall window + one more record: >= 80 s. V0.1 pays
        # only 0.06 s a record to read slot_roots, and reuses the manifest the caller already has.
        assert elapsed < 6.0, "empty catch-up still slow (%.1fs)" % elapsed
        assert any("NOTHING TO FOLD" in s for s in said), said
        assert not any("ABORTED" in s for s in said), said

    def test_the_records_that_cannot_move_our_coordinate_are_never_folded(self, monkeypatch):
        """The cost fix itself: `slot_roots` is read BEFORE the fold, not discovered by attempting
        one. A fold here means a delta fetch + a slot copy + a hash + a rollback, for nothing."""
        clk = _FakeClock()
        folds = []
        monkeypatch.setattr(N, "model_root", lambda h: "ours")
        monkeypatch.setattr(N, "slot_root", lambda h, i: "ours")
        monkeypatch.setattr(N, "_fold_accepted_checked",
                            lambda *a, **k: (folds.append(a[2]), (False, "lineage", []))[1])
        lane = self._empty_lane(clk)
        N.resume_to_root(_CoordFakeHost([self.OURS]), lane, "target", lambda *_a: None,
                         own_coord=self.OURS, now=clk, man=lane.manifest())
        assert folds == [], "%d record(s) folded that could not move our coordinate" % len(folds)

    def test_a_legacy_record_carrying_no_slot_roots_is_still_folded(self, monkeypatch):
        """The filter keys on an ADVERTISEMENT. A pre-shard-claim record advertises no coordinate at
        all, so skipping it would silently change what the replay folds -- keep folding it."""
        clk, folds = _FakeClock(), []
        monkeypatch.setattr(N, "model_root", lambda h: "ours")
        monkeypatch.setattr(N, "slot_root", lambda h, i: "ours")
        monkeypatch.setattr(N, "_fold_accepted_checked",
                            lambda *a, **k: (folds.append(a[2]), (True, "ok", []))[1])
        lane = _CostLane({}, clk)
        lane.names[N.accepted_name(1)] = {"sha256": "shaL"}
        lane.recs["shaL"] = {"event": 1, "model_root": "g1"}          # no slot_roots at all
        N.resume_to_root(_CoordFakeHost([self.OURS]), lane, "target", lambda *_a: None,
                         own_coord=self.OURS, now=clk, man=lane.manifest())
        assert len(folds) == 1, "a record with no slot_roots must keep its pre-V0.1 handling"

    def test_the_global_root_path_keeps_V0s_stall_rule_exactly(self, monkeypatch):
        """own_coord=None has NO applicability filter -- every record is a candidate for the global
        root -- so its arming signal must stay "a record was scanned", not "a record was applicable"
        (which never becomes true there). Without this, V0.1 would silently delete the stall bound
        from the global path."""
        clk = _FakeClock()
        monkeypatch.setattr(N, "model_root", lambda h: "ours")
        self._dead_fold(clk, monkeypatch)
        lane = _CostLane({e: {"1_%d" % e: "theirs%d" % e} for e in range(1, 61)}, clk)
        out = {}
        applied, reached = N.resume_to_root(_CoordFakeHost([self.OURS]), lane, "target",
                                            lambda *_a: None, budget_s=180.0, stall_s=30.0,
                                            now=clk, outcome=out, man=lane.manifest())
        assert out["reason"] == "stall" and out["aborted"] is True, out
        assert (applied, reached) == (0, False)

    def test_applicable_records_that_fail_to_fold_STILL_stall_park_and_walk_on(self, monkeypatch):
        """V0's guarantee, un-regressed: when records DO name our coordinate and none of them folds,
        that is a real stall -- abort, park for 900 s, advance to the next claimable coordinate."""
        clk = _FakeClock()
        monkeypatch.setattr(N, "model_root", lambda h: "ours")
        monkeypatch.setattr(N, "slot_root", lambda h, i: "ours")
        self._dead_fold(clk, monkeypatch)
        ours_key = "%d_%d" % self.OURS
        lane = _CostLane({e: {ours_key: "mine%d" % e} for e in range(1, 61)}, clk)
        out = {}
        applied, reached = N.resume_to_root(_CoordFakeHost([self.OURS]), lane, "target",
                                            lambda *_a: None, own_coord=self.OURS, budget_s=180.0,
                                            stall_s=30.0, now=clk, outcome=out, man=lane.manifest())
        assert out["reason"] == "stall" and out["aborted"] is True, out
        assert (applied, reached) == (0, False) and out["applicable"] > 0

        cd = N.CoordCooldown(seconds=900.0, events=10, now=_FakeClock())
        host = TestAdvanceClaimWalksPastBlockedCoordinates._Host(TestAdvanceClaimWalksPastBlockedCoordinates.COORDS)
        order = [c for c in N.claim_walk_order(TestAdvanceClaimWalksPastBlockedCoordinates.COORDS, "wallet0")
                 if c != (1, 0)]

        def fake_resume(_h, _l, _root, _log, own_coord=None, outcome=None, **_kw):
            reason = "stall" if tuple(own_coord) == order[0] else "empty"
            if outcome is not None:
                outcome.update(reason=reason, elapsed_s=1.0, records=0,
                               aborted=reason in N._CATCHUP_ABORTED)
            return 0, False
        monkeypatch.setattr(N, "resume_to_root", fake_resume)
        got = N.advance_claim(host, object(), list(TestAdvanceClaimWalksPastBlockedCoordinates.COORDS), (1, 0), "wallet0",
                              None, "root", 7, cd, lambda *_a: None, "m0")
        assert cd.blocked(order[0], 7) is True, "a real stall must still park the coordinate"
        assert got is not None and got[0] == order[1], "and the walk must still move on"

    def test_an_empty_catch_up_is_never_parked_by_the_walk(self, monkeypatch):
        """The second half of the goal metric, at the caller: `empty` lands, trains, no cooldown."""
        cd = N.CoordCooldown(seconds=900.0, events=10, now=_FakeClock())
        host = TestAdvanceClaimWalksPastBlockedCoordinates._Host(TestAdvanceClaimWalksPastBlockedCoordinates.COORDS)
        order = [c for c in N.claim_walk_order(TestAdvanceClaimWalksPastBlockedCoordinates.COORDS, "wallet0")
                 if c != (1, 0)]
        logs = []

        def fake_resume(_h, _l, _root, _log, own_coord=None, outcome=None, **_kw):
            if outcome is not None:
                outcome.update(reason="empty", elapsed_s=0.9, records=0, aborted=False)
            return 0, False
        monkeypatch.setattr(N, "resume_to_root", fake_resume)
        got = N.advance_claim(host, object(), list(TestAdvanceClaimWalksPastBlockedCoordinates.COORDS), (1, 0), "wallet0",
                              None, "root", 7, cd, logs.append, "m0")
        assert got is not None and got[0] == order[0], "empty must not make the walk skip past it"
        assert cd.blocked(order[0], 7) is False, "a healthy empty coordinate must NOT be parked"
        assert not any("COOLDOWN" in ln for ln in logs), logs
        assert any("NOTHING TO FOLD" in ln for ln in logs), logs

    def test_an_unreachable_but_non_empty_coordinate_keeps_todays_semantics(self, monkeypatch):
        """Guarantee 4: records DO advertise our coordinate and DO fold, but our replay never
        reproduces the advertised root. That is the honest `unreachable` -- roll back to the frozen
        base, train anyway, be lineage-dropped. It must not become `empty` and must not be parked."""
        clk = _FakeClock()
        monkeypatch.setattr(N, "model_root", lambda h: "ours")
        monkeypatch.setattr(N, "slot_root", lambda h, i: "not-what-was-advertised")
        monkeypatch.setattr(N, "_fold_accepted_checked", lambda *_a, **_k: (True, "ok", []))
        ours_key = "%d_%d" % self.OURS
        lane = _CostLane({e: {ours_key: "mine%d" % e} for e in range(1, 5)}, clk)
        host = _CoordFakeHost([self.OURS])
        out, said = {}, []
        applied, reached = N.resume_to_root(host, lane, "target", said.append, own_coord=self.OURS,
                                            now=clk, outcome=out, man=lane.manifest())
        assert out["reason"] == "unreachable" and out["aborted"] is False, out
        assert (applied, reached) == (0, False) and out["applicable"] == 4
        assert host.writes == len(host.slots), "fail-closed rollback, exactly as before"
        assert any("lineage-dropped" in s for s in said), said

    def test_the_callers_manifest_is_reused_instead_of_read_a_second_time(self, monkeypatch):
        """The other half of the 55.9-92.8 s. The async loop reads a manifest every pass
        (sharddiloco_glm_contributor._run_async, `man = lane.manifest()`) seconds before it calls
        advance_claim, and a walk past N parked coordinates used to buy N more of them."""
        clk = _FakeClock()
        monkeypatch.setattr(N, "model_root", lambda h: "ours")
        monkeypatch.setattr(N, "slot_root", lambda h, i: "ours")
        self._dead_fold(clk, monkeypatch)
        lane = self._empty_lane(clk)
        man = lane.manifest()
        assert lane.manifests == 1
        N.resume_to_root(_CoordFakeHost([self.OURS]), lane, "target", lambda *_a: None,
                         own_coord=self.OURS, now=clk, man=man)
        assert lane.manifests == 1, "resume_to_root re-read a manifest it was handed"
        N.resume_to_root(_CoordFakeHost([self.OURS]), lane, "target", lambda *_a: None,
                         own_coord=self.OURS, now=clk)
        assert lane.manifests == 2, "man=None must still read one, exactly as before"

    def test_the_async_loop_hands_its_manifest_down_to_the_walk(self):
        """Wiring: the parameter is worthless if the live call sites do not pass it."""
        import inspect
        src = inspect.getsource(N._run_async)
        assert src.count("advance_claim(host, lane, claim_coords") == 2
        assert src.count("man=man)") == 2, "both advance_claim call sites must pass the manifest"
        walk = inspect.getsource(N.advance_claim)
        assert "man=man)" in walk, "advance_claim must hand it to resume_to_root"
