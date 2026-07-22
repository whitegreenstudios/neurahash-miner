"""Tests for the elastic VRAM auto-tuner core (neurahash/vram_autotune.py).

NO real CUDA: every case drives target_units()/tick() with SIMULATED free-VRAM sequences and
asserts EXACT unit counts. Cost model under test: base 4.0 GiB trunk + 1.125 GiB per resident MoE
layer, plan only against safety_frac of free, always leave the owner user_headroom_gib.

The six required behaviours each get their own asserting test:
  (a) test_a_shrinks_immediately_when_free_drops
  (b) test_b_single_spike_does_not_grow_then_grows_after_patience
  (c) test_c_grows_back_to_max_when_free_sustained_high
  (d) test_d_pauses_to_zero_when_nothing_fits
  (e) test_e_never_exceeds_max_or_falls_between_zero_and_min
  (f) test_f_always_leaves_user_headroom_free
"""

import pytest

from neurahash.vram_autotune import (
    AdaptiveVramTuner,
    free_total_vram_gib,
    _cuda_device_index,
)

# Shared cost-model constants (a 5090-shaped card).
BASE = 4.0
PER = 1.125
HEAD = 1.5
SAFETY = 0.90


def make_tuner(min_units=0, max_units=16, grow_patience=3):
    return AdaptiveVramTuner(
        device="cpu",
        base_gib=BASE,
        per_unit_gib=PER,
        user_headroom_gib=HEAD,
        min_units=min_units,
        max_units=max_units,
        grow_patience=grow_patience,
        safety_frac=SAFETY,
    )


def expected_target(free, min_units=0, max_units=16):
    """Independent re-implementation of the documented rule, used to cross-check target_units."""
    budget = free * SAFETY - HEAD - BASE
    fit = int(budget // PER)
    units = max(min_units, min(fit, max_units))
    return 0 if units * PER > budget else units


# --- acceptance #2: the worked example -----------------------------------------------------------
def test_worked_example_free20_gives_exactly_11():
    # budget = 20*0.90 - 1.5 - 4.0 = 12.5 ; 12.5 // 1.125 = 11 ; 11*1.125 = 12.375 <= 12.5 -> 11
    t = make_tuner(min_units=0, max_units=16)
    assert t.target_units(20.0) == 11


# --- (a) shrink immediately ----------------------------------------------------------------------
def test_a_shrinks_immediately_when_free_drops():
    t = make_tuner(min_units=0, max_units=16, grow_patience=3)
    # startup baseline: 20 GiB free -> 11 units (adopted on the very first tick, no hysteresis)
    assert t.tick(free_gib=20.0) == 11
    assert t.current_units == 11
    # owner opens an app: free collapses to 10 GiB. target = floor((9-5.5)/1.125)=floor(3.11)=3.
    # Shrink must happen on the SAME tick -- no patience on the way down.
    assert t.tick(free_gib=10.0) == 3
    assert t.current_units == 3


# --- (b) no thrash on a single spike; grow only after grow_patience ------------------------------
def test_b_single_spike_does_not_grow_then_grows_after_patience():
    t = make_tuner(min_units=0, max_units=16, grow_patience=3)
    assert t.tick(free_gib=10.0) == 3          # baseline
    # A one-tick spike of free VRAM (owner momentarily freed the card) must NOT grow.
    assert t.tick(free_gib=30.0) is None       # target 16 > 3, but streak=1 < patience
    assert t.current_units == 3
    # ...and if it drops back, the pending grow is cancelled (target 3 == current).
    assert t.tick(free_gib=10.0) is None
    assert t.current_units == 3
    # Now a SUSTAINED high level: grow fires only on the grow_patience-th consecutive tick.
    assert t.tick(free_gib=30.0) is None       # streak 1
    assert t.tick(free_gib=30.0) is None       # streak 2
    assert t.tick(free_gib=30.0) == 16         # streak 3 -> grow (capped at max_units)
    assert t.current_units == 16


def test_b2_grows_to_sustained_min_not_the_transient_peak():
    # Over the patience window the target wobbles 16,15,16 -> we grow to 15 (the level that HELD),
    # never to the 16 peak that appeared only on some ticks.
    t = make_tuner(min_units=0, max_units=16, grow_patience=3)
    assert t.tick(free_gib=10.0) == 3          # baseline (current=3)
    assert expected_target(30.0) == 16 and expected_target(25.0) == 15
    assert t.tick(free_gib=30.0) is None       # streak 1, floor 16
    assert t.tick(free_gib=25.0) is None       # streak 2, floor min(16,15)=15
    assert t.tick(free_gib=30.0) == 15         # streak 3 -> grow to the sustained floor, 15
    assert t.current_units == 15


# --- (c) grow back to max when free is sustained-high --------------------------------------------
def test_c_grows_back_to_max_when_free_sustained_high():
    t = make_tuner(min_units=0, max_units=16, grow_patience=3)
    assert t.tick(free_gib=8.0) == 1           # baseline low (budget 1.7 -> 1 unit)
    # Free stays very high (budget supports far more than max) -> we climb to max_units and stop.
    assert t.tick(free_gib=100.0) is None      # streak 1
    assert t.tick(free_gib=100.0) is None      # streak 2
    assert t.tick(free_gib=100.0) == 16        # streak 3 -> max
    assert t.current_units == 16
    # Already at max: staying high produces no further resize.
    assert t.tick(free_gib=100.0) is None
    assert t.tick(free_gib=100.0) is None
    assert t.tick(free_gib=100.0) is None
    assert t.current_units == 16


# --- (d) pause to 0 when nothing fits ------------------------------------------------------------
def test_d_pauses_to_zero_when_nothing_fits():
    # min_units=2, but a nearly-full card cannot fit even 2 layers (nor the base) -> PAUSE (0).
    t = make_tuner(min_units=2, max_units=16, grow_patience=3)
    # free=6: budget = 6*0.9 - 1.5 - 4.0 = -0.1 -> base itself doesn't fit -> 0.
    assert t.target_units(6.0) == 0
    assert t.tick(free_gib=6.0) == 0           # baseline adopts the pause
    assert t.current_units == 0
    # free=8: budget=1.7 -> only 1 layer fits, but min_units=2 -> still cannot honour min -> pause.
    assert t.target_units(8.0) == 0
    # Recover: free=12 supports >=2 layers -> target is a real >=min value again.
    assert t.target_units(12.0) == max(2, int((12 * SAFETY - HEAD - BASE) // PER))


# --- (e) never exceeds max_units, never lands strictly between 0 and min_units -------------------
def test_e_never_exceeds_max_or_falls_between_zero_and_min():
    mn, mx = 2, 16
    t = make_tuner(min_units=mn, max_units=mx)
    # Sweep a wide range of free VRAM including tiny and huge cards.
    frees = [x * 0.5 for x in range(0, 261)]   # 0.0 .. 130.0 GiB
    for free in frees:
        u = t.target_units(free)
        assert u == expected_target(free, mn, mx)          # matches the documented rule
        assert u == 0 or (mn <= u <= mx)                    # 0 (pause) or within [min,max]
        assert not (0 < u < mn)                             # never a value below the floor
        assert u <= mx                                      # never above the ceiling
    # A giant card is pinned at max, not higher.
    assert t.target_units(1000.0) == mx


# --- (f) the owner always keeps user_headroom_gib free ------------------------------------------
def test_f_always_leaves_user_headroom_free():
    t = make_tuner(min_units=0, max_units=16)
    frees = [x * 0.25 for x in range(0, 521)]  # 0.0 .. 130.0 GiB, fine-grained
    for free in frees:
        u = t.target_units(free)
        if u > 0:
            footprint = BASE + u * PER
            # Actual VRAM left for the owner after our footprint must be >= the reserved headroom.
            assert free - footprint >= HEAD - 1e-9, (free, u, footprint)


# --- run() drives the on_resize seam; only fires on a change -------------------------------------
def test_run_calls_on_resize_only_on_change(monkeypatch):
    import neurahash.vram_autotune as va

    # Scripted free-VRAM readings the loop will "measure" (total is ignored by the tuner).
    readings = [(20.0, 32.0), (20.0, 32.0), (10.0, 32.0)]
    calls = []

    def fake_free_total(device):
        return readings.pop(0) if readings else (0.0, 0.0)

    monkeypatch.setattr(va, "free_total_vram_gib", fake_free_total)

    class _Stop:
        def __init__(self, n):
            self.n, self.i = n, 0

        def is_set(self):
            fired = self.i >= self.n
            self.i += 1
            return fired

    t = va.AdaptiveVramTuner(device="cuda:0", base_gib=BASE, per_unit_gib=PER,
                             user_headroom_gib=HEAD, min_units=0, max_units=16, grow_patience=3)
    t.run(on_resize=lambda old, new: calls.append((old, new)),
          stop_event=_Stop(3), sleep=lambda _s: None)

    # tick1: baseline 0 -> 11 (fires). tick2: 20 GiB again -> 11 == current (no fire).
    # tick3: 10 GiB -> shrink to 3 (fires).
    assert calls == [(0, 11), (11, 3)]


# --- from_env is opt-in and honours the knobs ---------------------------------------------------
def test_from_env_off_returns_none(monkeypatch):
    monkeypatch.delenv("NEURAHASH_VRAM_AUTOTUNE", raising=False)
    assert AdaptiveVramTuner.from_env("cuda:0", BASE, PER, 16) is None


def test_from_env_on_reads_knobs(monkeypatch):
    monkeypatch.setenv("NEURAHASH_VRAM_AUTOTUNE", "1")
    monkeypatch.setenv("NEURAHASH_VRAM_USER_HEADROOM_GB", "2.0")
    monkeypatch.setenv("NEURAHASH_VRAM_AUTOTUNE_INTERVAL_S", "5")
    monkeypatch.setenv("NEURAHASH_VRAM_AUTOTUNE_MIN_UNITS", "1")
    monkeypatch.setenv("NEURAHASH_VRAM_AUTOTUNE_MAX_UNITS", "8")
    t = AdaptiveVramTuner.from_env("cuda:1", BASE, PER, 16)
    assert t is not None
    assert (t.user_headroom_gib, t.interval_s, t.min_units, t.max_units) == (2.0, 5.0, 1, 8)
    assert t.device == "cuda:1"


# --- the self-contained device reader parses indices and is CPU-safe ----------------------------
def test_device_index_parsing_and_cpu_safe_reader():
    assert _cuda_device_index("cuda") == 0
    assert _cuda_device_index("cuda:0") == 0
    assert _cuda_device_index("cuda:1") == 1
    assert _cuda_device_index("cpu") is None
    assert _cuda_device_index("") is None
    # No CUDA on the test box -> (0.0, 0.0), never a crash.
    assert free_total_vram_gib("cpu") == (0.0, 0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
