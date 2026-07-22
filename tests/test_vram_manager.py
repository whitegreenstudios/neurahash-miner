"""Tests for the UNIFIED VramManager (neurahash/vram_manager.py).

NO real CUDA: every case drives detect()/apply_cap()/report_capacity()/run() with SIMULATED
free-VRAM (either passed explicitly or via a monkeypatched free_total_vram_gib). The manager
composes the GUARD (hard cap), the CAPACITY report, and the TUNER off ONE live-free detect(); these
tests prove the three can never contradict each other.

Required behaviours, each with its own asserting test:
  (a) test_a_flag_off_from_env_returns_none          -- DEFAULT OFF; existing behaviour untouched.
  (b) test_b_grab_shrinks_all_three_in_lockstep_free_grows_back_with_hysteresis
                                                     -- user grabs VRAM => cap + report + resize all
                                                        DOWN from the same detect; freeing grows the
                                                        footprint back only after grow-hysteresis.
  (c) test_c_one_source_of_truth_cap_report_target_never_disagree
                                                     -- for ANY free value the cap always covers the
                                                        footprint of the advertised capacity, and the
                                                        advertised capacity == the tuner's target.
"""

import pytest

from neurahash.vram_autotune import AdaptiveVramTuner
from neurahash.vram_manager import VramManager, DEFAULT_HEADROOM, RUNAWAY_SLACK

# Shared cost model (a 5090-shaped card), identical to test_vram_autotune.
BASE = 4.0
PER = 1.125
HEAD = 1.5
SAFETY = 0.90
TOTAL = 32.0        # card total GiB used for the cap fraction math


def make_manager(min_units=0, max_units=16, grow_patience=3):
    """A VramManager on a cpu device (so apply_cap takes no torch side effect) with a tuner using
    the shared cost model. Constructed directly (not from_env) so tests don't depend on env flags."""
    tuner = AdaptiveVramTuner(
        device="cpu", base_gib=BASE, per_unit_gib=PER, user_headroom_gib=HEAD,
        min_units=min_units, max_units=max_units, grow_patience=grow_patience, safety_frac=SAFETY)
    return VramManager(device="cpu", tuner=tuner)


def expected_target(free, min_units=0, max_units=16):
    """Independent re-implementation of the tuner's documented rule (cross-check, no shared code)."""
    budget = free * SAFETY - HEAD - BASE
    fit = int(budget // PER)
    units = max(min_units, min(fit, max_units))
    return 0 if units * PER > budget else units


def default_cap(free, max_units=16):
    """Independent re-implementation of the manager's DEFAULT cap number (no env override)."""
    max_footprint = BASE + max_units * PER
    return min(free * DEFAULT_HEADROOM, max_footprint * RUNAWAY_SLACK)


# --- (a) DEFAULT OFF: the flag-off path is byte-identical (from_env yields nothing) --------------
def test_a_flag_off_from_env_returns_none(monkeypatch):
    monkeypatch.delenv("NEURAHASH_VRAM_MANAGER", raising=False)
    assert VramManager.from_env("cuda:0", BASE, PER, 16) is None


def test_a2_from_env_on_builds_a_manager_reusing_autotune_knobs(monkeypatch):
    monkeypatch.setenv("NEURAHASH_VRAM_MANAGER", "1")
    monkeypatch.setenv("NEURAHASH_VRAM_USER_HEADROOM_GB", "2.0")
    monkeypatch.setenv("NEURAHASH_VRAM_AUTOTUNE_INTERVAL_S", "5")
    monkeypatch.setenv("NEURAHASH_VRAM_AUTOTUNE_MIN_UNITS", "1")
    monkeypatch.setenv("NEURAHASH_VRAM_AUTOTUNE_MAX_UNITS", "8")
    m = VramManager.from_env("cuda:1", BASE, PER, 16)
    assert m is not None
    # The manager reuses the tuner's cost model and the autotune knobs -- one config, not two.
    assert m.device == "cuda:1"
    assert (m.base_gib, m.per_unit_gib, m.interval_s) == (BASE, PER, 5.0)
    assert (m.tuner.user_headroom_gib, m.tuner.min_units, m.tuner.max_units) == (2.0, 1, 8)


# --- (b) grab => all three DOWN in lockstep; free => footprint grows back WITH hysteresis --------
def test_b_grab_shrinks_all_three_in_lockstep_free_grows_back_with_hysteresis(monkeypatch):
    # No cap override -> the default free-based cap formula is exercised.
    monkeypatch.delenv("NEURAHASH_VRAM_CAP_GB", raising=False)
    monkeypatch.delenv("NEURAHASH_VRAM_CAP_FRAC", raising=False)
    m = make_manager(min_units=0, max_units=16, grow_patience=3)

    # Baseline: 20 GiB free. All three agree on the high state, all from free=20.
    assert m.report_capacity(free_gib=20.0) == 11             # sustainable units advertised
    assert m.tuner.tick(free_gib=20.0) == 11                  # tuner adopts the baseline (resize)
    assert m.tuner.current_units == 11
    cap_hi = m.cap_gib(20.0, TOTAL)
    assert cap_hi == default_cap(20.0) == min(18.0, 33.0)     # 18.0 GiB ceiling
    assert cap_hi >= m.footprint_gib(11)                      # ceiling covers the advertised footprint

    # OWNER GRABS VRAM: free collapses 20 -> 10. From this ONE detect (free=10) all three move DOWN
    # together -- this is the over-task fix: report drops so the coordinator reassigns.
    free_low = 10.0
    report_low = m.report_capacity(free_gib=free_low)
    cap_low = m.cap_gib(free_low, TOTAL)
    resize_low = m.tuner.tick(free_gib=free_low)              # shrink is immediate (no patience down)
    assert report_low == 3                                    # advertised capacity DOWN 11 -> 3
    assert cap_low == default_cap(free_low) == 9.0            # hard cap DOWN 18.0 -> 9.0
    assert resize_low == 3 and m.tuner.current_units == 3     # resident footprint DOWN 11 -> 3
    # LOCKSTEP: the advertised capacity and the resize target are the SAME number from the SAME free.
    assert report_low == resize_low == expected_target(free_low)
    assert cap_low < cap_hi and report_low < 11               # all three strictly lower

    # OWNER FREES VRAM: free jumps 10 -> 30. report_capacity reflects the higher fit immediately, but
    # the actual resident-layer REBUILD (on_resize) is paced by grow-hysteresis: it must NOT fire
    # until grow_patience consecutive high ticks -- a brief free-spike never triggers a rebuild.
    assert m.report_capacity(free_gib=30.0) == 16             # instantaneous sustainable fit (capped at max)
    assert m.tuner.tick(free_gib=30.0) is None                # streak 1: footprint NOT rebuilt yet
    assert m.tuner.tick(free_gib=30.0) is None                # streak 2: still holding 3 (hysteresis)
    assert m.tuner.current_units == 3
    assert m.tuner.tick(free_gib=30.0) == 16                  # streak 3: NOW the footprint grows back
    assert m.tuner.current_units == 16


# --- (c) one source of truth: cap, advertised capacity, and tuner target never disagree ----------
def test_c_one_source_of_truth_cap_report_target_never_disagree(monkeypatch):
    monkeypatch.delenv("NEURAHASH_VRAM_CAP_GB", raising=False)
    monkeypatch.delenv("NEURAHASH_VRAM_CAP_FRAC", raising=False)
    mn, mx = 0, 16
    m = make_manager(min_units=mn, max_units=mx)
    # report_capacity and cap_gib are PURE (no tuner state mutation), so one fresh manager can be
    # swept across every plausible free value on a 32 GiB card.
    frees = [x * 0.25 for x in range(0, int(TOTAL / 0.25) + 1)]   # 0.0 .. 32.0 GiB
    for free in frees:
        units = m.report_capacity(free_gib=free)
        # (1) the number the coordinator schedules against IS the number the tuner resizes toward.
        assert units == m.tuner.target_units(free) == expected_target(free, mn, mx)
        cap = m.cap_gib(free, TOTAL)
        assert cap == default_cap(free, mx)
        if units > 0:
            footprint = m.footprint_gib(units)
            # (2) the hard ceiling ALWAYS covers the footprint of the advertised capacity -- we never
            #     promise the coordinator more units than the cap physically permits.
            assert cap >= footprint - 1e-9, (free, units, footprint, cap)
            # (3) and the owner still keeps their headroom free under that footprint.
            assert free - footprint >= HEAD - 1e-9, (free, units, footprint)
        else:
            # paused: advertise 0 units, so the coordinator schedules nothing (no over-task).
            assert units == 0


# --- run(): ONE detect per tick drives cap + on_report every tick, on_resize only on a change -----
def test_run_single_detect_feeds_cap_report_and_resize(monkeypatch):
    import neurahash.vram_manager as vm

    # Scripted (free, total) the loop will "measure": 20 (baseline), 20 (steady), 10 (owner grabs).
    readings = [(20.0, TOTAL), (20.0, TOTAL), (10.0, TOTAL)]

    def fake_free_total(device):
        return readings.pop(0) if readings else (0.0, 0.0)

    monkeypatch.setattr(vm, "free_total_vram_gib", fake_free_total)
    monkeypatch.delenv("NEURAHASH_VRAM_CAP_GB", raising=False)
    monkeypatch.delenv("NEURAHASH_VRAM_CAP_FRAC", raising=False)

    m = make_manager(min_units=0, max_units=16, grow_patience=3)
    reports, resizes = [], []

    class _Stop:
        def __init__(self, n):
            self.n, self.i = n, 0

        def is_set(self):
            fired = self.i >= self.n
            self.i += 1
            return fired

    m.run(on_resize=lambda old, new: resizes.append((old, new)),
          on_report=lambda cap: reports.append(cap),
          stop_event=_Stop(3), sleep=lambda _s: None)

    # on_report fires EVERY tick with the sustainable capacity from that tick's single detect:
    #   tick1 free=20 -> 11 ; tick2 free=20 -> 11 ; tick3 free=10 -> 3.
    assert reports == [11, 11, 3]
    # on_resize fires ONLY on a change: baseline 0->11, then the shrink 11->3 (steady 20 tick fires nothing).
    assert resizes == [(0, 11), (11, 3)]


# --- apply_cap is CPU-safe and returns the cap number without touching torch ---------------------
def test_apply_cap_cpu_safe(monkeypatch):
    monkeypatch.delenv("NEURAHASH_VRAM_CAP_GB", raising=False)
    monkeypatch.delenv("NEURAHASH_VRAM_CAP_FRAC", raising=False)
    m = make_manager()
    # Explicit free/total on a cpu device: no CUDA index, so it returns the number, no side effect.
    assert m.apply_cap(free_gib=20.0, total_gib=TOTAL) == default_cap(20.0)
    # Auto-detect on cpu: free_total_vram_gib('cpu') == (0.0, 0.0) -> total<=0 -> None (no crash).
    assert m.apply_cap() is None


# --- env cap overrides win, same vocabulary as the guard ----------------------------------------
def test_cap_env_overrides(monkeypatch):
    m = make_manager()
    monkeypatch.setenv("NEURAHASH_VRAM_CAP_GB", "7.5")
    assert m.cap_gib(20.0, TOTAL) == 7.5                      # absolute GB wins
    monkeypatch.delenv("NEURAHASH_VRAM_CAP_GB", raising=False)
    monkeypatch.setenv("NEURAHASH_VRAM_CAP_FRAC", "0.5")
    assert m.cap_gib(20.0, TOTAL) == 0.5 * TOTAL              # fraction of the card total


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
