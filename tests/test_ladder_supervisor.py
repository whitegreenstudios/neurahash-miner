"""
tests/test_ladder_supervisor.py — the capability-ladder decision logic (pure functions).

Covers: convergence patience (warming up / still-improving / plateaued / min-blocks gate),
feasible-rung-by-fleet-size, and the unified promote/hold decision the owner asked for
("bigger model when done AND enough GPUs; wait for the 3070 otherwise; keep earning at the top").

Run:  python -m pytest tests/test_ladder_supervisor.py -q
"""

import json

import pytest

from tools.ladder_supervisor import (RUNGS, RUNG_MIN_MINERS, DEFAULT_RUNGS, decide,
                                      highest_feasible_rung, is_converged,
                                      load_rungs, RungConfigError, main)

CFG = dict(patience=4, min_delta=1e-3, min_blocks=50)


def _hist(losses, blocks_start=0, blocks_step=10):
    return [{"held_out_val_loss": l, "blocks_found": blocks_start + i * blocks_step}
            for i, l in enumerate(losses)]


# ---- highest_feasible_rung -------------------------------------------------
def test_feasible_rung_by_fleet_size():
    # table: tiny=1, small=2, medium=3, large=3
    assert highest_feasible_rung(1) == "tiny"
    assert highest_feasible_rung(2) == "small"
    assert highest_feasible_rung(3) == "large"       # medium(3) and large(3) both fit; pick biggest
    assert highest_feasible_rung(9) == "large"       # capped at the top rung


# ---- is_converged ----------------------------------------------------------
def test_warming_up_not_converged():
    conv, why = is_converged(_hist([2.0, 1.9, 1.8], blocks_start=0), blocks_at_rung_start=0, **CFG)
    assert not conv and "warming up" in why


def test_still_improving_not_converged():
    # steadily dropping loss over > patience samples, plenty of blocks
    conv, why = is_converged(_hist([3.0, 2.5, 2.0, 1.5, 1.0, 0.5], blocks_start=0, blocks_step=20),
                             blocks_at_rung_start=0, **CFG)
    assert not conv and "still improving" in why


def test_plateau_is_converged():
    # first sample low, then flat -> window does not beat prior best
    conv, why = is_converged(_hist([1.0, 1.0009, 1.0007, 1.0006, 1.0005, 1.0004],
                                   blocks_start=0, blocks_step=20),
                             blocks_at_rung_start=0, **CFG)
    assert conv and "not improving" in why


def test_worsening_counts_as_done():
    # held-out loss climbing (like a diverged run) -> "not improving" -> promote-eligible
    conv, _ = is_converged(_hist([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], blocks_start=0, blocks_step=20),
                           blocks_at_rung_start=0, **CFG)
    assert conv


def test_min_blocks_gate_blocks_premature_done():
    # flat loss but the rung has barely trained -> not done yet
    conv, why = is_converged(_hist([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], blocks_start=0, blocks_step=1),
                             blocks_at_rung_start=0, **CFG)
    assert not conv and "too early" in why


# ---- decide (the unified promotion decision) -------------------------------
def _converged_hist():
    return _hist([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], blocks_start=0, blocks_step=20)


def test_hold_while_training():
    d = decide(current_rung="small", miners_online=3,
               history=_hist([3.0, 2.0, 1.0, 0.5, 0.2, 0.1], blocks_step=20),
               blocks_at_rung_start=0, cfg=CFG)
    assert d["action"] == "hold" and not d["converged"]


def test_converged_and_enough_miners_promotes():
    # small converged, 3 miners (the 3070 is in) -> promote to medium (needs 3)
    d = decide(current_rung="small", miners_online=3, history=_converged_hist(),
               blocks_at_rung_start=0, cfg=CFG)
    assert d["action"] == "promote" and d["target"] == "medium" and d["converged"]


def test_converged_but_waiting_for_3070_holds():
    # small converged, only 2 miners -> next rung 'medium' needs 3 -> HOLD, wait for the 3070
    d = decide(current_rung="small", miners_online=2, history=_converged_hist(),
               blocks_at_rung_start=0, cfg=CFG)
    assert d["action"] == "hold" and d["converged"]
    assert "needs 3" in d["reason"] and "3070" in d["reason"]


def test_converged_at_top_rung_holds_and_keeps_earning():
    d = decide(current_rung="large", miners_online=5, history=_converged_hist(),
               blocks_at_rung_start=0, cfg=CFG)
    assert d["action"] == "hold" and d["converged"]
    assert "TOP rung" in d["reason"]


def test_rung_table_is_monotonic_and_consistent():
    # min_miners never decreases as rungs grow (a bigger model never needs fewer cards)
    mins = [RUNG_MIN_MINERS[r] for r in RUNGS]
    assert mins == sorted(mins)
    assert set(RUNG_MIN_MINERS) == set(RUNGS)


# ---- rung table as data (#90): --rungs JSON, propose_only, load_rungs -------
def _qwen_bpe_table():
    """The example table: the four char rungs + a propose_only qwen-BPE rung on top of `large`."""
    return DEFAULT_RUNGS + [dict(name="qwen-bpe", min_miners=3, propose_only=True,
                                 arch_rung="large", corpus="qwen")]


def test_default_rungs_matches_backcompat_views():
    # the module-level RUNGS/RUNG_MIN_MINERS are derived from DEFAULT_RUNGS (no drift)
    assert [r["name"] for r in DEFAULT_RUNGS] == RUNGS
    assert {r["name"]: r["min_miners"] for r in DEFAULT_RUNGS} == RUNG_MIN_MINERS


def test_decide_default_table_unchanged_when_rungs_none():
    # passing rungs=None is byte-identical to the old hardcoded behavior
    d = decide(current_rung="small", miners_online=3, history=_converged_hist(),
               blocks_at_rung_start=0, cfg=CFG, rungs=None)
    assert d["action"] == "promote" and d["target"] == "medium"


def test_propose_only_target_yields_propose_not_promote():
    # large converged, 3 miners can host qwen-bpe (min 3) -> but it's propose_only -> action 'propose'
    d = decide(current_rung="large", miners_online=3, history=_converged_hist(),
               blocks_at_rung_start=0, cfg=CFG, rungs=_qwen_bpe_table())
    assert d["action"] == "propose" and d["target"] == "qwen-bpe" and d["converged"]
    assert "PROPOSE-ONLY" in d["reason"]


def test_propose_only_never_promotes_even_in_enforce(tmp_path, capsys):
    # DELIVERABLE 2 core: a propose_only target in ENFORCE mode must NOT execute promote (no relaunch).
    # Drive main() once with an enforce mode + a real relaunch-cmd; the decision is 'propose' so the
    # relaunch is never invoked and no promotion is recorded in the supervisor state.
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"pool": {"held_out_val_loss": 1.0, "miners_online": 3,
                                          "blocks_found": 200}}), encoding="utf-8")
    state = tmp_path / "ladder_state.json"
    # pre-seed a converged history at the `large` rung so the single tick fires the decision immediately
    hist = [{"held_out_val_loss": 1.0, "miners_online": 3, "blocks_found": 100 + i * 20}
            for i in range(6)]
    state.write_text(json.dumps({"current_rung": "large", "blocks_at_rung_start": 0,
                                 "history": hist, "promotions": []}), encoding="utf-8")
    rungs = tmp_path / "rungs.json"
    rungs.write_text(json.dumps(_qwen_bpe_table()), encoding="utf-8")
    canary = tmp_path / "RELAUNCHED"                       # promote() would run this shell cmd; it must NOT
    rc = main(["--once", "--mode", "enforce", "--stats", str(stats), "--state", str(state),
               "--log", str(tmp_path / "ladder.log"), "--rungs", str(rungs),
               "--patience", "4", "--min-blocks", "50",
               "--relaunch-cmd", f"python -c \"open(r'{canary}','w').close()\""])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"action": "propose"' in out and "qwen-bpe" in out
    assert not canary.exists(), "propose_only rung must NOT execute the relaunch even in enforce mode"
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["current_rung"] == "large" and saved["promotions"] == []   # no promotion recorded


def test_plain_rung_still_promotes_in_enforce(tmp_path, capsys):
    # a NON-propose_only target in enforce mode DOES execute promote (the relaunch runs) — guards that the
    # propose_only gate didn't accidentally disable normal promotions.
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"pool": {"held_out_val_loss": 1.0, "miners_online": 3,
                                          "blocks_found": 200}}), encoding="utf-8")
    state = tmp_path / "ladder_state.json"
    hist = [{"held_out_val_loss": 1.0, "miners_online": 3, "blocks_found": 100 + i * 20}
            for i in range(6)]
    state.write_text(json.dumps({"current_rung": "small", "blocks_at_rung_start": 0,
                                 "history": hist, "promotions": []}), encoding="utf-8")
    canary = tmp_path / "RELAUNCHED"
    rc = main(["--once", "--mode", "enforce", "--stats", str(stats), "--state", str(state),
               "--log", str(tmp_path / "ladder.log"),
               "--state-dir", str(tmp_path / "_state"), "--archive-dir", str(tmp_path / "_arch"),
               "--patience", "4", "--min-blocks", "50",
               "--relaunch-cmd", f"python -c \"open(r'{canary}','w').close()\""])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"action": "promote"' in out and "medium" in out
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["current_rung"] == "medium" and saved["promotions"]        # promotion recorded


def test_load_rungs_roundtrip(tmp_path):
    p = tmp_path / "rungs.json"
    table = _qwen_bpe_table()
    p.write_text(json.dumps(table), encoding="utf-8")
    loaded = load_rungs(str(p))
    assert [r["name"] for r in loaded] == [r["name"] for r in table]
    assert loaded[-1]["propose_only"] is True and loaded[-1]["corpus"] == "qwen"


def test_load_rungs_example_file_is_valid():
    # the committed example config parses + ends with the propose_only qwen-bpe rung
    r = load_rungs("tools/ladder_rungs.example.json")
    assert [x["name"] for x in r] == ["tiny", "small", "medium", "large", "qwen-bpe"]
    assert r[-1] == {"name": "qwen-bpe", "min_miners": 3, "propose_only": True,
                     "arch_rung": "large", "corpus": "qwen"}


@pytest.mark.parametrize("bad", [
    "{ not json ]",                                             # malformed JSON
    json.dumps({"name": "x"}),                                  # not a list
    json.dumps([]),                                             # empty list
    json.dumps([{"min_miners": 1}]),                            # missing name
    json.dumps([{"name": "a", "min_miners": 1}, {"name": "a", "min_miners": 2}]),   # dup name
    json.dumps([{"name": "a", "min_miners": 0}]),               # min_miners < 1
    json.dumps([{"name": "a", "min_miners": True}]),            # bool is not a valid int here
    json.dumps([{"name": "a", "min_miners": 3}, {"name": "b", "min_miners": 1}]),   # non-monotonic
    json.dumps([{"name": "a", "min_miners": 1, "corpus": 5}]),  # corpus must be a string
])
def test_load_rungs_rejects_malformed(tmp_path, bad):
    p = tmp_path / "bad.json"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(RungConfigError):
        load_rungs(str(p))


def test_load_rungs_missing_file_refused():
    with pytest.raises(RungConfigError):
        load_rungs(str("this/does/not/exist.json"))


def test_main_refuses_malformed_rungs_exit_2(tmp_path, capsys):
    # malformed --rungs => clear error + non-zero exit (REFUSE, don't guess a ladder)
    p = tmp_path / "bad.json"
    p.write_text("{ broken", encoding="utf-8")
    rc = main(["--once", "--rungs", str(p), "--stats", str(tmp_path / "nostats.json")])
    assert rc == 2
    assert "not valid JSON" in capsys.readouterr().err


# ---- chain-height reset (#94): baseline must not go negative ----------------
def test_is_converged_clamps_negative_blocks_at_source():
    # #94 (source clamp): baseline captured at a high pre-reset height; the chain then RESET so the
    # observed heights are BELOW that baseline. Old code computed trained = blocks_now - baseline < 0
    # (e.g. -99800), so `trained < min_blocks` was ALWAYS true and the rung froze "too early" forever.
    # The clamp guarantees blocks-this-rung is never negative (reported as 0, not a negative count);
    # actually resuming progress after a reset is main()'s re-anchor job (test below).
    reset_hist = _hist([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], blocks_start=0, blocks_step=20)
    conv, why = is_converged(reset_hist, blocks_at_rung_start=100000, **CFG)
    assert not conv and "only 0 blocks" in why      # clamped to 0, never a negative "too early" count

    # and once enough blocks have accrued ABOVE the (already re-anchored) baseline, it converges again
    conv2, why2 = is_converged(reset_hist, blocks_at_rung_start=0, **CFG)
    assert conv2, why2


def test_main_reanchors_baseline_on_height_reset(tmp_path, capsys):
    # #94 END-TO-END: the supervisor's stored baseline is from BEFORE a chain-height reset (100000),
    # but the live stats + history now report LOWER heights. Old code: is_converged sees a negative
    # trained count forever => never promotes (the ladder silently freezes). Fixed: main() detects
    # current_height < baseline, RE-ANCHORS the baseline to the current height (one ASCII log line),
    # and the decision can once again reach 'promote'.
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"pool": {"held_out_val_loss": 1.0, "miners_online": 3,
                                          "blocks_found": 200}}), encoding="utf-8")
    state = tmp_path / "ladder_state.json"
    # converged history at post-reset (low) heights, but baseline stuck at a huge pre-reset value
    hist = [{"held_out_val_loss": 1.0, "miners_online": 3, "blocks_found": 60 + i * 20}
            for i in range(6)]
    state.write_text(json.dumps({"current_rung": "small", "blocks_at_rung_start": 100000,
                                 "history": hist, "promotions": []}), encoding="utf-8")
    log = tmp_path / "ladder.log"
    canary = tmp_path / "RELAUNCHED"
    rc = main(["--once", "--mode", "enforce", "--stats", str(stats), "--state", str(state),
               "--log", str(log),
               "--state-dir", str(tmp_path / "_state"), "--archive-dir", str(tmp_path / "_arch"),
               "--patience", "4", "--min-blocks", "50",
               "--relaunch-cmd", f"python -c \"open(r'{canary}','w').close()\""])
    assert rc == 0
    out = capsys.readouterr().out
    # the ladder must resume promoting (small -> medium) instead of reporting negative blocks forever
    assert '"action": "promote"' in out and "medium" in out, out
    saved = json.loads(state.read_text(encoding="utf-8"))
    # baseline was re-anchored down (not left at the stale pre-reset height), then advanced on promote
    assert saved["blocks_at_rung_start"] <= 200
    # the re-anchor emitted a human ASCII line on stdout AND a governance record in the log
    assert "re-anchoring rung small" in out and out.isascii()
    log_txt = log.read_text(encoding="utf-8")
    assert '"ladder-reanchor"' in log_txt and "100000" in log_txt and log_txt.isascii()


def test_main_normal_monotonic_height_does_not_reanchor(tmp_path, capsys):
    # behavior-preserving guard: with monotonic (non-reset) height, NO re-anchor happens and the
    # baseline is untouched by the reset logic (still-training run just holds).
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"pool": {"held_out_val_loss": 0.5, "miners_online": 3,
                                          "blocks_found": 300}}), encoding="utf-8")
    state = tmp_path / "ladder_state.json"
    hist = [{"held_out_val_loss": 3.0 - i * 0.4, "miners_online": 3, "blocks_found": 220 + i * 20}
            for i in range(6)]                                  # still improving -> hold
    state.write_text(json.dumps({"current_rung": "small", "blocks_at_rung_start": 200,
                                 "history": hist, "promotions": []}), encoding="utf-8")
    log = tmp_path / "ladder.log"
    rc = main(["--once", "--mode", "enforce", "--stats", str(stats), "--state", str(state),
               "--log", str(log), "--patience", "4", "--min-blocks", "50"])
    assert rc == 0
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["blocks_at_rung_start"] == 200                 # untouched: no reset, no re-anchor
    assert "re-anchor" not in log.read_text(encoding="utf-8").lower()
