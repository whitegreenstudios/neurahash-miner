"""THE DOSE-BACKOFF LADDER -- shrink the dose on a rejected layer before ever giving it up.

WHY THIS EXISTS, AND WHY THE OBVIOUS DESIGN IS WRONG (measured, do not re-derive from intuition --
docs/research/LAYER_COMPOSITION_TRUEGRAD_2026-07-29.md):

  "Detect a bad layer, drop it, claim another" treats the wrong cause. Three layers dosed at the
  SAME relative drift from ONE multi-tap backward (identical batches) moved full-47 held-out CE by
  L1 -0.095165, L5 -0.054498, L2 +0.217781 (damage) -- which reads as "L2 is bad". But layers 1 and
  5 TOGETHER, two layers that each individually IMPROVE the model, returned -0.002736 against a
  linear prediction of -0.149663: 1.8% of their predicted joint gain. Good layers INTERFERE.

  The mechanism sets the policy. Cross-curvature interference is QUADRATIC in the dose while the
  gain is LINEAR, so halving the dose costs half the gain and a quarter of the interference --
  shrinking is the lever that recovers value. A drop-first miner instead burns GPU hours
  re-discovering, one layer at a time, that every layer is "bad in company".

  The defect that made this impossible: LayerClaimTrainer.train_layer(L, ...) took the layer index
  and IGNORED it for dosing -- one campaign scalar, `self.target_rho`, for every layer. A per-layer
  dose was inexpressible, so abandonment was the only available response to a rejection.

WHAT EACH GROUP GUARDS
  1. PER-LAYER DOSE, FAIL-CLOSED. A silently-ignored dose override yields a run that reports it
     tested rho/10 while it tested rho -- a confident WRONG measurement, the class of defect that
     has already voided published numbers here (memory cross-campaign-record-replay).
  2. THE LADDER DESCENDS, and the assertion is on the rho ACTUALLY handed to the dose call, never
     on a log line.
  3. THE RUNG SURVIVES A RESTART. Run 5's 5090 miner started 28 times in 15.7 h; an in-RAM ladder
     would rarely reach its second rung, and every restart would re-inflict the largest dose.
  4. AN ACCEPT RESETS IT, so a layer that recovers is not permanently penalised.
  5. FLAG OFF IS INERT. The owner's no-regression contract.

All fakes, no GLM, no GPU. Run:
  C:/Python313/python.exe -m pytest tests/test_glm_dose_ladder.py -q
"""
import json
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_HERE, _REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glm_grad_cache as G                                  # noqa: E402  (stdlib-only at import)
import sharddiloco_glm_contributor as N                     # noqa: E402

CAMPAIGN_RHO = 6.0e-2                     # the campaign dose these tests configure


def _act(x):
    import torch
    return torch.nn.functional.silu(x)


def _env(**over):
    env = {G.FLAG: "1", "NEURAHASH_SD_LAYER_RHO": "6.0e-2"}
    env.update(over)
    return env


def _build(tmp_path, **over):
    """The REAL startup constructor -- the malformed-config tests must hit the real parse."""
    return N._maybe_build_layer_trainer(types.SimpleNamespace(data_dir=str(tmp_path)),
                                        environ=_env(**over), log=lambda *_a: None)


class _RecordingGC:
    """Records the rho `train_layer` actually hands to the dose call.

    Deliberately NOT a re-implemented trainer: it captures the ONE argument under test and returns
    the minimum shape train_layer reads. A fake that is more capable than production is how a suite
    stays green while production 404s, so the REAL bisection is exercised against the ladder in
    TestTheLadderAgainstTheRealDose below -- that test uses no fake at all."""

    CacheVerifyError = G.CacheVerifyError

    def __init__(self):
        self.rhos = []

    def iter_cache(self, path, torch_mod=None, **exp):
        return [(0, {"h": None})]

    def train_layer_dose(self, units, gu_all, dn_all, I, act_fn, target_rho, **kw):
        self.rhos.append(float(target_rho))
        return {"delta": (None, None), "target_rho": float(target_rho), "lr": 1.0,
                "achieved_rho": float(target_rho), "rel_err": 0.0, "n_units": len(units),
                "wall_s": 0.0, "applied": True}


class _Slab:
    """Stands in for gate_up[E, 2I, H] -- train_layer only reads .shape[0] off it."""
    shape = (3,)


def _recording(tmp_path, **over):
    t = _build(tmp_path, **over)
    t._GC = _RecordingGC()
    return t


def _dose(t, L=1):
    """One training attempt; returns the rho that reached the dose call."""
    t.train_layer(L, _Slab(), None, 5, _act, cache_path="/nowhere", torch_mod=None)
    return t._GC.rhos[-1]


# ============================================================ 1. the per-layer dose override
class TestPerLayerDoseOverride:
    def test_a_layer_without_an_override_gets_the_campaign_dose(self, tmp_path):
        t = _build(tmp_path)
        assert t.base_rho(0) == pytest.approx(CAMPAIGN_RHO)
        assert t.base_rho(47) == pytest.approx(CAMPAIGN_RHO)

    def test_a_named_layer_gets_its_own_dose(self, tmp_path):
        t = _build(tmp_path, NEURAHASH_SD_LAYER_RHO_MAP="1:0.02,5:1.5e-3")
        assert t.base_rho(1) == pytest.approx(0.02)
        assert t.base_rho(5) == pytest.approx(1.5e-3)
        assert t.base_rho(2) == pytest.approx(CAMPAIGN_RHO)      # unnamed -> campaign dose

    def test_the_override_reaches_the_DOSE_CALL_not_just_the_accessor(self, tmp_path):
        t = _recording(tmp_path, NEURAHASH_SD_LAYER_RHO_MAP="5:1.5e-3")
        assert _dose(t, L=5) == pytest.approx(1.5e-3)
        assert _dose(t, L=2) == pytest.approx(CAMPAIGN_RHO)

    @pytest.mark.parametrize("bad", [
        "1", "1;2", "banana", "1:banana", "x:0.01", "1:0.01,1:0.02", "-1:0.01",
        "1:0", "1:-0.01", "1:nan", "1:inf", "1:0.01,2:", "1:0.01,:0.02", "1:0.01,3",
    ])
    def test_a_malformed_map_RAISES_instead_of_silently_using_the_global_rho(self, tmp_path, bad):
        with pytest.raises(N.DoseConfigError):
            _build(tmp_path, NEURAHASH_SD_LAYER_RHO_MAP=bad)

    def test_the_raise_happens_at_CONSTRUCTION_before_any_training(self, tmp_path):
        """Fail at boot, not at round 47: a miner that has already published deltas at the wrong
        dose cannot un-publish them."""
        with pytest.raises(N.DoseConfigError, match="NEURAHASH_SD_LAYER_RHO_MAP"):
            _build(tmp_path, NEURAHASH_SD_LAYER_RHO_MAP="1:banana")

    def test_an_override_the_campaign_gate_would_refuse_is_refused_here_too(self):
        """One owner for 'is this a legal drift target': dose_spec_from_config. The per-layer map
        must not be a second, laxer door into the same number."""
        with pytest.raises(N.DoseConfigError):
            N.parse_layer_rho_map("3:-1e-3", GC=G)
        assert N.parse_layer_rho_map("3:1e-3", GC=G) == {3: 1e-3}

    def test_an_empty_or_unset_map_is_simply_no_overrides(self):
        for raw in (None, "", "   ", ","):
            assert N.parse_layer_rho_map(raw, GC=G) == {}


# ============================================================ 2. the ladder descends
class TestTheDoseLadderDescends:
    def test_two_rejections_walk_the_dose_down_rho_then_third_then_tenth(self, tmp_path):
        """THE acceptance criterion: assert on the rho ACTUALLY passed to the dose call."""
        t = _recording(tmp_path)
        assert _dose(t) == pytest.approx(CAMPAIGN_RHO)                 # attempt 1: campaign dose
        assert N.layer_dose_verdict(t, 1) == N.DOSE_ACTION_SHRINK      # rejected
        assert _dose(t) == pytest.approx(CAMPAIGN_RHO / 3.0)           # attempt 2: rho/3
        assert N.layer_dose_verdict(t, 1) == N.DOSE_ACTION_SHRINK      # rejected again
        assert _dose(t) == pytest.approx(CAMPAIGN_RHO / 10.0)          # attempt 3: rho/10
        assert t._GC.rhos == [pytest.approx(6.0e-2), pytest.approx(2.0e-2), pytest.approx(6.0e-3)]

    def test_the_layer_is_only_given_up_AFTER_the_smallest_rung_fails(self, tmp_path):
        """Abandonment is the LAST resort: three verdicts, only the third releases the layer."""
        t = _recording(tmp_path)
        assert [N.layer_dose_verdict(t, 1) for _ in range(3)] == [
            N.DOSE_ACTION_SHRINK, N.DOSE_ACTION_SHRINK, N.DOSE_ACTION_EXHAUSTED]
        assert N.layer_dose_verdict(t, 1) == N.DOSE_ACTION_EXHAUSTED   # and it stays exhausted
        assert _dose(t) == pytest.approx(6.0e-3)                       # still the smallest rung

    def test_the_ladder_is_PER_LAYER_not_per_miner(self, tmp_path):
        """A rejection on layer 1 must not shrink layer 5's dose: they were measured to respond
        differently (-0.095165 vs -0.054498), and one shared rung is the scalar defect again."""
        t = _recording(tmp_path)
        N.layer_dose_verdict(t, 1)
        assert _dose(t, L=1) == pytest.approx(2.0e-2)
        assert _dose(t, L=5) == pytest.approx(CAMPAIGN_RHO)            # untouched

    def test_the_ladder_multiplies_the_PER_LAYER_dose_not_the_campaign_one(self, tmp_path):
        t = _recording(tmp_path, NEURAHASH_SD_LAYER_RHO_MAP="5:3.0e-3")
        N.layer_dose_verdict(t, 5)
        assert _dose(t, L=5) == pytest.approx(1.0e-3)                  # 3e-3/3, not 6e-2/3

    def test_an_ACCEPT_resets_the_rung_to_the_top(self, tmp_path):
        """A layer that recovers must not stay permanently penalised."""
        t = _recording(tmp_path)
        N.layer_dose_verdict(t, 1)
        N.layer_dose_verdict(t, 1)
        assert _dose(t) == pytest.approx(6.0e-3)                       # bottom rung
        assert N.layer_dose_verdict(t, 1, accepted=True) == N.DOSE_ACTION_RESET
        assert t.ladder.rung(1) == 0
        assert _dose(t) == pytest.approx(CAMPAIGN_RHO)                 # campaign dose again
        assert N.layer_dose_verdict(t, 1) == N.DOSE_ACTION_SHRINK      # and the ladder works again

    def test_an_accept_on_one_layer_does_not_reset_another(self, tmp_path):
        t = _recording(tmp_path)
        N.layer_dose_verdict(t, 1)
        N.layer_dose_verdict(t, 2)
        N.layer_dose_verdict(t, 1, accepted=True)
        assert _dose(t, L=1) == pytest.approx(CAMPAIGN_RHO)
        assert _dose(t, L=2) == pytest.approx(2.0e-2)

    def test_the_default_ladder_is_the_one_the_design_names(self):
        assert N.DOSE_LADDER_DEFAULT == (1.0, 1.0 / 3.0, 1.0 / 10.0)
        assert N.DoseLadder().rungs == (1.0, 1.0 / 3.0, 1.0 / 10.0)

    def test_the_rungs_are_configurable(self, tmp_path):
        t = _recording(tmp_path, NEURAHASH_SD_LAYER_RHO_LADDER="1.0,0.5,0.25,0.125")
        assert t.ladder.rungs == (1.0, 0.5, 0.25, 0.125)
        assert [N.layer_dose_verdict(t, 2) for _ in range(4)][-1] == N.DOSE_ACTION_EXHAUSTED
        assert _dose(t, L=2) == pytest.approx(CAMPAIGN_RHO * 0.125)

    @pytest.mark.parametrize("bad", ["2.0,1.0", "1.0,2.0", "0.5,0.1", "1.0,0.1,0.3", "1.0,0.0",
                                     "1.0,-0.1", "1.0,nan", "1.0,banana", "1.0,1.0"])
    def test_a_malformed_ladder_RAISES(self, tmp_path, bad):
        """Same fail-closed rule as the map, plus: a rung ABOVE the campaign dose is refused. This
        is a BACKOFF -- pushing the dose UP on rejection walks toward the measured divergence edge
        (+6.9% on the reference lr gave 10x the drift, +14.4% gave NaN)."""
        with pytest.raises(N.DoseConfigError):
            _build(tmp_path, NEURAHASH_SD_LAYER_RHO_LADDER=bad)


# ============================================================ 3. flag off is inert
class TestFlagOffIsInert:
    def test_nothing_is_constructed_and_the_seam_degrades_to_todays_behaviour(self):
        """With the flag off the plateau must release the coordinate exactly as it does today."""
        assert N._maybe_build_layer_trainer(None, environ={}, log=None) is None
        assert N._maybe_build_layer_trainer(None, environ={G.FLAG: "off"}, log=None) is None
        assert N.layer_claim_enabled({}) is False
        assert N.layer_dose_verdict(None, 1) == N.DOSE_ACTION_EXHAUSTED
        assert N.layer_dose_verdict(None, 1, accepted=True) == N.DOSE_ACTION_RESET

    def test_a_malformed_map_is_not_even_read_with_the_flag_off(self, tmp_path):
        """The gate is the flag, checked BEFORE any dose parsing: an unflagged miner carrying a
        stale env var must not fail to start."""
        env = {"NEURAHASH_SD_LAYER_RHO_MAP": "1:banana",
               "NEURAHASH_SD_LAYER_RHO_LADDER": "9,9,9"}
        assert N._maybe_build_layer_trainer(types.SimpleNamespace(data_dir=str(tmp_path)),
                                            environ=env, log=None) is None

    def test_the_plateau_branch_consults_the_ladder_only_through_the_flagged_trainer(self):
        """Source-level, like the publish-branch gate: `dose_trainer` is built behind
        layer_claim_enabled(), so with the flag off it is None and the advance below is unchanged."""
        import inspect
        src = inspect.getsource(N._run_async)
        assert "dose_trainer = layer_claim_trainer_for(args, log=log) if layer_claim_enabled()" in src
        i_gate = src.find("dose_trainer = layer_claim_trainer_for")
        assert 0 < i_gate < src.find("layer_dose_verdict(dose_trainer")


# ============================================================ 4. the rung survives a restart
class TestTheRungSurvivesARestart:
    def test_a_backed_off_rung_is_restored_from_the_same_claim_state_file(self, tmp_path):
        path = str(tmp_path / "claim_state.json")
        cd = N.CoordCooldown(seconds=600.0, events=3, now=lambda: 5000.0)
        ladder = N.DoseLadder()
        assert ladder.on_reject(7) is True                    # one rejection -> rho/3
        st = N.ClaimState(path, log=lambda *_a: None)
        assert st.save(cd, (7, 0), event=4, now=lambda: 5000.0, ladder=ladder) is True

        st2 = N.ClaimState(path, log=lambda *_a: None)        # a FRESH process
        ladder2 = N.DoseLadder()
        assert ladder2.rung(7) == 0                           # nothing restored yet
        assert st2.restore_ladder(ladder2) == (1, 0)
        assert ladder2.rung(7) == 1                           # NOT reset to the top
        assert ladder2.multiplier(7) == pytest.approx(1.0 / 3.0)

    def test_the_restored_rung_is_the_dose_the_next_process_actually_uses(self, tmp_path):
        """End of the chain: reject -> save -> restart -> the TRAINER doses at rho/3, not rho."""
        path = str(tmp_path / "claim_state.json")
        ladder = N.DoseLadder()
        ladder.on_reject(1)
        N.ClaimState(path, log=lambda *_a: None).save(N.CoordCooldown(now=lambda: 0.0), (1, 0),
                                                      event=1, now=lambda: 1.0, ladder=ladder)
        t = _recording(tmp_path)                              # a brand-new trainer, top rung
        assert _dose(t) == pytest.approx(CAMPAIGN_RHO)
        assert N.ClaimState(path, log=lambda *_a: None).restore_ladder(t.ladder) == (1, 0)
        assert _dose(t) == pytest.approx(2.0e-2)              # rho/3, not the campaign 6e-2

    def test_the_bottom_rung_survives_so_a_restart_cannot_re_inflict_the_full_dose(self, tmp_path):
        path = str(tmp_path / "claim_state.json")
        ladder = N.DoseLadder()
        ladder.on_reject(3)
        ladder.on_reject(3)
        assert ladder.at_bottom(3) is True
        N.ClaimState(path, log=lambda *_a: None).save(N.CoordCooldown(now=lambda: 0.0), (3, 0),
                                                      event=1, now=lambda: 1.0, ladder=ladder)
        ladder2 = N.DoseLadder()
        N.ClaimState(path, log=lambda *_a: None).restore_ladder(ladder2)
        assert ladder2.rung(3) == 2 and ladder2.at_bottom(3) is True

    def test_an_accepted_layer_persists_as_the_TOP_rung(self, tmp_path):
        path = str(tmp_path / "claim_state.json")
        cd, ladder = N.CoordCooldown(now=lambda: 0.0), N.DoseLadder()
        ladder.on_reject(2)
        st = N.ClaimState(path, log=lambda *_a: None)
        st.save(cd, (2, 0), event=1, now=lambda: 1.0, ladder=ladder)
        assert ladder.on_accept(2) is True
        # A rung change IS a meaningful change: the fingerprint must not skip this write.
        assert st.save(cd, (2, 0), event=2, now=lambda: 2.0, ladder=ladder) is True
        ladder2 = N.DoseLadder()
        N.ClaimState(path, log=lambda *_a: None).restore_ladder(ladder2)
        assert ladder2.rung(2) == 0

    def test_a_rung_change_ALONE_forces_a_write(self, tmp_path):
        """save() skips the write when nothing meaningful changed since the last one. A rung IS
        meaningful -- it is state whose entire value is surviving the next restart -- so it must be
        in the fingerprint. Everything else is held constant here (same coord, same event, same
        empty park table) so ONLY the rung can account for the write."""
        path = str(tmp_path / "claim_state.json")
        cd, ladder = N.CoordCooldown(now=lambda: 0.0), N.DoseLadder()
        st = N.ClaimState(path, log=lambda *_a: None)
        assert st.save(cd, (2, 0), event=7, now=lambda: 1.0, ladder=ladder) is True
        assert st.save(cd, (2, 0), event=7, now=lambda: 2.0, ladder=ladder) is False   # no change
        ladder.on_reject(2)
        assert st.save(cd, (2, 0), event=7, now=lambda: 3.0, ladder=ladder) is True    # rung moved
        ladder2 = N.DoseLadder()
        N.ClaimState(path, log=lambda *_a: None).restore_ladder(ladder2)
        assert ladder2.rung(2) == 1                           # and the new rung is what landed

    def test_an_accept_back_to_the_top_rung_also_forces_a_write(self, tmp_path):
        """The reset direction of the same rule: a skipped write here would leave a restarted miner
        dosing a recovered layer at rho/3 forever."""
        path = str(tmp_path / "claim_state.json")
        cd, ladder = N.CoordCooldown(now=lambda: 0.0), N.DoseLadder()
        ladder.on_reject(2)
        st = N.ClaimState(path, log=lambda *_a: None)
        assert st.save(cd, (2, 0), event=7, now=lambda: 1.0, ladder=ladder) is True
        assert ladder.on_accept(2) is True
        assert st.save(cd, (2, 0), event=7, now=lambda: 2.0, ladder=ladder) is True
        ladder2 = N.DoseLadder()
        N.ClaimState(path, log=lambda *_a: None).restore_ladder(ladder2)
        assert ladder2.rung(2) == 0

    def test_one_file_holds_cursor_parks_AND_rungs(self, tmp_path):
        """No second store: the ladder rides in the ClaimState file that already exists."""
        path = str(tmp_path / "claim_state.json")
        cd = N.CoordCooldown(seconds=600.0, events=3, now=lambda: 100.0)
        cd.park((1, 5), event=2, reason="catch-up budget")
        ladder = N.DoseLadder()
        ladder.on_reject(1)
        N.ClaimState(path, log=lambda *_a: None).save(cd, (1, 0), event=2, now=lambda: 100.0,
                                                      ladder=ladder)
        assert len([f for f in os.listdir(str(tmp_path)) if f.endswith(".json")]) == 1
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        assert blob["schema"] == N.CLAIM_STATE_SCHEMA         # NOT bumped: the key is additive
        assert blob["coord"] == [1, 0] and blob["cooldown"]["parked"] and blob["ladder"]["layers"]

    def test_a_pre_ladder_state_file_keeps_its_cursor_and_parks(self, tmp_path):
        """Backward compatibility: an existing miner's file has no `ladder` key. Bumping the schema
        for it would throw away every live miner's cursor and park table on upgrade -- the exact
        loss ClaimState was built to prevent."""
        path = str(tmp_path / "claim_state.json")
        st = N.ClaimState(path, log=lambda *_a: None)
        st.save(N.CoordCooldown(now=lambda: 0.0), (4, 2), event=3, now=lambda: 9.0)   # no ladder=
        with open(path, "r", encoding="utf-8") as fh:
            assert "ladder" not in json.load(fh)
        st2 = N.ClaimState(path, log=lambda *_a: None)
        assert st2.cursor() == (4, 2)                         # cursor preserved
        ladder = N.DoseLadder()
        assert st2.restore_ladder(ladder) == (0, 0)
        assert ladder.rung(4) == 0

    @pytest.mark.parametrize("bad", [None, {}, {"layers": "nope"}, {"layers": [{"L": "x"}]},
                                     {"layers": [{"L": 1}]}, {"layers": [{"L": 1, "rung": 99}]}])
    def test_a_corrupt_ladder_block_degrades_instead_of_killing_the_miner(self, tmp_path, bad):
        """Same never-fatal contract as restore_cooldown: this runs at boot on a file an operator
        may have hand-edited, and a traceback there is a permanent supervisor restart loop."""
        path = str(tmp_path / "claim_state.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"schema": N.CLAIM_STATE_SCHEMA, "saved_wall": 1.0, "coord": [1, 0],
                       "cooldown": {}, "ladder": bad}, fh)
        ladder = N.DoseLadder()
        restored, _dropped = N.ClaimState(path, log=lambda *_a: None).restore_ladder(ladder)
        assert restored == 0 and ladder.rung(1) == 0

    def test_a_rung_saved_under_a_DIFFERENT_ladder_never_comes_back_LARGER(self, tmp_path):
        """An operator edits the ladder between runs. The stored index means nothing against the
        new rungs, so the restore lands on the largest live rung still <= the one we left."""
        path = str(tmp_path / "claim_state.json")
        old = N.DoseLadder()
        old.on_reject(1)
        old.on_reject(1)                                      # bottom rung: x0.1
        N.ClaimState(path, log=lambda *_a: None).save(N.CoordCooldown(now=lambda: 0.0), (1, 0),
                                                      event=1, now=lambda: 1.0, ladder=old)
        new = N.DoseLadder(rungs=(1.0, 0.5, 0.2, 0.05))
        N.ClaimState(path, log=lambda *_a: None).restore_ladder(new)
        assert new.multiplier(1) <= 0.1                       # never larger than the dose we left
        assert new.multiplier(1) == pytest.approx(0.05)

    def test_a_save_WITHOUT_the_ladder_would_drop_it_so_every_call_site_passes_it(self):
        """save() rewrites the whole file. If any call site omitted `ladder=`, the next process
        would get the campaign dose on a layer that had already exhausted the backoff."""
        import inspect
        src = inspect.getsource(N._run_async)
        saves = [ln for ln in src.splitlines() if "claim_state.save(" in ln]
        assert len(saves) >= 4, saves
        assert all("ladder=dose_ladder" in ln for ln in saves), saves


# ============================================================ 5. the real dose, no stub at all
class TestTheLadderAgainstTheRealDose:
    """Guards the seam the recording fake cannot: that the number train_layer computes is one
    glm_grad_cache accepts and the REAL bisection can actually land."""

    def setup_method(self):
        self.torch = pytest.importorskip("torch")

    @staticmethod
    def _unit(torch, n_tok=6, H=4, E=8, top_k=2, seed=0):
        gen = torch.Generator().manual_seed(seed)
        h = torch.randn(n_tok, H, generator=gen)
        g = torch.randn(n_tok, H, generator=gen)
        pairs = [(t, e) for t in range(n_tok) for e in range(top_k)]
        tk, off = [], [0]
        for e in range(E):
            for t, s in pairs:
                if (t + s) % E == e:
                    tk.append(t)
            off.append(len(tk))
        w = torch.rand(len(tk), generator=gen)
        return {"h": h, "g": g, "tk": torch.tensor(tk, dtype=torch.long), "w": w,
                "off": torch.tensor(off, dtype=torch.long)}

    def test_a_backed_off_layer_lands_the_SMALLER_drift_end_to_end(self, tmp_path):
        torch = self.torch
        p = str(tmp_path / "cache")
        units = [(1001 + i, self._unit(torch, seed=i)) for i in range(3)]
        G.write_cache(p, units, campaign_id="cid", layer=1, corpus_sha="sha", fold=4,
                      spec=G.batch_spec(seed=100, step0=1001, n_batches=4, fold=4),
                      experts=8, top_k=2)
        t = N.LayerClaimTrainer(str(tmp_path), 6.0e-3, campaign_id="cid", corpus_sha="sha", fold=4,
                                log=lambda *_a: None)
        gen = torch.Generator().manual_seed(1)
        gu = torch.randn(8, 10, 4, generator=gen)
        dn = torch.randn(8, 4, 5, generator=gen)

        rep = t.train_layer(1, gu.clone(), dn.clone(), 5, _act, cache_path=p, torch_mod=torch)
        assert rep["target_rho"] == pytest.approx(6.0e-3)
        assert rep["achieved_rho"] == pytest.approx(6.0e-3, rel=G.RHO_TOL)

        assert N.layer_dose_verdict(t, 1) == N.DOSE_ACTION_SHRINK        # the judge rejected it
        rep2 = t.train_layer(1, gu.clone(), dn.clone(), 5, _act, cache_path=p, torch_mod=torch)
        assert rep2["target_rho"] == pytest.approx(2.0e-3)               # the LADDERED target
        assert rep2["achieved_rho"] == pytest.approx(2.0e-3, rel=G.RHO_TOL)
        assert rep2["rel_err"] <= G.RHO_TOL
        assert rep2["lr"] < rep["lr"]                                    # smaller dose, smaller lr

    def test_the_published_target_rho_is_the_LADDERED_one_the_miner_really_used(self, tmp_path):
        """`layer_dose` goes on the wire. It must report the dose actually applied, not the
        campaign's -- a delta labelled with a dose it did not use is an unusable measurement."""
        torch = self.torch
        p = str(tmp_path / "cache")
        units = [(1001 + i, self._unit(torch, seed=i)) for i in range(2)]
        G.write_cache(p, units, campaign_id="cid", layer=1, corpus_sha="sha", fold=4,
                      spec=G.batch_spec(seed=100, step0=1001, n_batches=4, fold=4),
                      experts=8, top_k=2)
        t = N.LayerClaimTrainer(str(tmp_path), 6.0e-3, campaign_id="cid", corpus_sha="sha", fold=4,
                                log=lambda *_a: None)
        gen = torch.Generator().manual_seed(1)
        gu, dn = torch.randn(8, 10, 4, generator=gen), torch.randn(8, 4, 5, generator=gen)
        N.layer_dose_verdict(t, 1)
        N.layer_dose_verdict(t, 1)                                       # bottom rung: rho/10
        rep = t.train_layer(1, gu, dn, 5, _act, cache_path=p, torch_mod=torch)
        assert rep["target_rho"] == pytest.approx(6.0e-4)
        assert rep["achieved_rho"] == pytest.approx(6.0e-4, rel=G.RHO_TOL)
