"""Elastic VRAM auto-tuner for the NeuraHash public miner -- the "good neighbor" layer.

WHY THIS EXISTS (the crash lesson):
    The miner trains on the SAME GPU the owner uses for their own work (a 5090). MEASURED
    2026-07-21: uncapped GLM processes launched next to the live pool soak exhausted a 32 GB
    5090, spilled into shared system RAM over the WDDM driver, and CRASHED the whole host.
    `apply_vram_guard` (tools/sharddiloco_glm_contributor.py) and `apply_vram_cap`
    (sharded_pool_node.py) already stop THIS process from spilling by sizing a HARD per-process
    ceiling from CURRENTLY-FREE VRAM at startup -- but that ceiling is STATIC: it is chosen once
    and never revisited. If the owner opens a game or another app an hour later, a static cap set
    when the card was idle is now too big, and we are back to fighting the owner for VRAM.

    This module adds the DYNAMIC half. It re-reads free VRAM every ~20 s and AUTO-TUNES how many
    training "units" the miner holds so it always leaves the owner a headroom:
      - SHRINK immediately   when the owner grabs VRAM (release first, ask questions later);
      - GROW back with hysteresis when VRAM stays free (never re-grab memory the owner briefly
        freed, never thrash on a 1-unit boundary);
      - PAUSE to 0 units      when not even the minimum footprint fits.

COST MODEL (GLM expert-sharded training, memory glm-capacity-per-card-2026-07-21):
    footprint(units) = base_gib + units * per_unit_gib
    A "unit" = one RESIDENT MoE layer (~1.125 GiB each); the base trunk is ~4.0 GiB. Experts
    within a resident layer are FREE (they share the layer's tensors), so the tunable quantity is
    the number of resident LAYERS, not experts.

SCOPE: this file is the tested CORE only. It computes target unit counts and emits resize events.
    It does NOT rebuild the resident-layer set or move any weights -- that is the caller's job,
    wired through the `on_resize(old_units, new_units)` seam (see AdaptiveVramTuner.run). Default
    OFF: nothing here runs unless NEURAHASH_VRAM_AUTOTUNE is set (see from_env).

Kept deliberately import-light (torch is imported lazily inside the reader) so importing this
module never drags in the 280 KB sharded_pool_node monolith.
"""

import os
import time


def _cuda_device_index(device):
    """Parse the CUDA device index from a device spec, or None if it is not a CUDA device.
    'cuda' -> 0, 'cuda:0' -> 0, 'cuda:1' -> 1, 'cpu'/'' -> None. Accepts a torch.device via str().

    WHY (same reason as sharded_pool_node._cuda_device_index): an exact `str(device) == "cuda"`
    check silently disables the guard on an INDEXED device ('cuda:1') -- the exact host-spill
    class this whole subsystem exists to prevent."""
    s = str(device)
    if not s.startswith("cuda"):
        return None
    if ":" in s:
        try:
            return int(s.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def free_total_vram_gib(device):
    """(free_gib, total_gib) for `device`, read live from the driver. Self-contained on purpose:
    a few lines mirroring the guard's index parsing rather than importing the monolith.

    Returns (0.0, 0.0) for cpu / no-CUDA / any read error, so callers never crash on a machine
    without a GPU. The free figure already reflects EVERY other allocation on the card -- the live
    pool soak AND the owner's own apps -- which is exactly what makes this a good-neighbor signal."""
    idx = _cuda_device_index(device)
    if idx is None:
        return (0.0, 0.0)
    try:
        import torch
        if not torch.cuda.is_available():
            return (0.0, 0.0)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        return (free_b / 2 ** 30, total_b / 2 ** 30)
    except Exception:
        return (0.0, 0.0)


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "on", "yes", "y")


class AdaptiveVramTuner:
    """Decides how many training units to hold as free VRAM moves. PURE core + a thin loop.

    The two knobs that give it its personality:
      safety_frac    -- only ever plan against this fraction of free VRAM, so measurement jitter
                        and allocator slop never push the real footprint past what is free.
      grow_patience  -- number of consecutive ticks a HIGHER level must hold before we grow into
                        it (hysteresis). Shrinking has no patience: we shrink on the first tick.

    Unit semantics: a value in [min_units, max_units] means "hold this many resident MoE layers";
    the special value 0 means PAUSE (hold nothing) and is returned even when min_units > 0 if not
    even the minimum footprint fits. So the reachable set is {0} union {min_units .. max_units}.
    """

    def __init__(self, device, base_gib, per_unit_gib, user_headroom_gib,
                 min_units, max_units, interval_s=20.0, grow_patience=3, safety_frac=0.90):
        if per_unit_gib <= 0:
            raise ValueError("per_unit_gib must be > 0 (got %r)" % (per_unit_gib,))
        if max_units < min_units:
            raise ValueError("max_units (%r) < min_units (%r)" % (max_units, min_units))
        self.device = device
        self.base_gib = float(base_gib)
        self.per_unit_gib = float(per_unit_gib)
        self.user_headroom_gib = float(user_headroom_gib)
        self.min_units = int(min_units)
        self.max_units = int(max_units)
        self.interval_s = float(interval_s)
        self.grow_patience = int(grow_patience)
        self.safety_frac = float(safety_frac)

        # Internal state. current_units is public so a caller/test can inspect what we believe we
        # are holding. _primed=False means we have never observed VRAM yet: the FIRST tick adopts
        # whatever the current free VRAM supports with NO hysteresis (there is no prior state to
        # thrash against -- this is the "detect at startup" behaviour).
        self.current_units = int(min_units)
        self._primed = False
        self._grow_streak = 0          # consecutive ticks whose target exceeded current_units
        self._grow_floor = None        # min target seen during the current grow streak (the level
                                       # that has actually HELD, so growing to it can't over-grab)

    # ---- PURE ---------------------------------------------------------------------------------
    def target_units(self, free_gib):
        """How many units SHOULD fit right now, given `free_gib` free. No state, no side effects.

        RULE (exact):
            budget = free_gib * safety_frac - user_headroom_gib - base_gib   # GiB left for units
            fit    = int(budget // per_unit_gib)      # how many whole units actually fit (can be <0)
            units  = clamp(fit, min_units, max_units)  # = max(min_units, min(fit, max_units))
            # If clamping UP to min_units would need more than `budget` (i.e. not even min_units
            # fits, or the base trunk itself doesn't fit), we must NOT hold min_units -- that is
            # what spills to shared RAM. Return 0 (PAUSE) instead.
            return 0 if units * per_unit_gib > budget else units

        Consequences: the returned footprint base + units*per_unit is always <= free*safety_frac -
        headroom, so the actual free-after is always >= user_headroom_gib (in fact >=
        headroom + free*(1-safety_frac)). The result is 0 or in [min_units, max_units] -- never a
        value strictly between 0 and min_units.

        WORKED EXAMPLE: free=20, safety=0.90, headroom=1.5, base=4.0, per_unit=1.125, max>=16:
            budget = 20*0.90 - 1.5 - 4.0 = 12.5
            fit    = int(12.5 // 1.125) = int(11.11..) = 11
            units  = 11 ; 11*1.125 = 12.375 <= 12.5  ->  returns 11
        """
        budget = free_gib * self.safety_frac - self.user_headroom_gib - self.base_gib
        fit = int(budget // self.per_unit_gib)   # floors toward -inf; negative if base doesn't fit
        units = max(self.min_units, min(fit, self.max_units))
        if units * self.per_unit_gib > budget:   # honouring min_units would oversubscribe -> pause
            return 0
        return units

    # ---- STATEFUL -----------------------------------------------------------------------------
    def _reset_grow(self):
        self._grow_streak = 0
        self._grow_floor = None

    def tick(self, free_gib=None):
        """Advance one step. Reads live free VRAM (or uses `free_gib` for tests), compares the
        target to what we hold, and returns the NEW unit count ONLY when a resize is warranted;
        otherwise None. Mutates current_units on a resize.

        - First observation ever: adopt the target directly (startup baseline, no hysteresis).
        - target < current: SHRINK immediately to target (release VRAM the owner wants back).
        - target > current: GROW only after grow_patience consecutive ticks all above current;
          grow to the MIN target seen across that window (the level that actually held), so a
          single-tick free spike never makes us grab memory that was only briefly free.
        - target == current (or grow not yet earned): return None (no change).
        """
        if free_gib is None:
            free_gib, _ = free_total_vram_gib(self.device)
        target = self.target_units(free_gib)

        if not self._primed:
            self._primed = True
            self._reset_grow()
            if target != self.current_units:
                self.current_units = target
                return target
            return None

        if target < self.current_units:
            self._reset_grow()
            self.current_units = target
            return target

        if target > self.current_units:
            self._grow_streak += 1
            self._grow_floor = target if self._grow_floor is None else min(self._grow_floor, target)
            if self._grow_streak >= self.grow_patience:
                new = self._grow_floor
                self._reset_grow()
                self.current_units = new
                return new
            return None

        # target == current_units: no pressure in either direction; any pending grow is broken.
        self._reset_grow()
        return None

    def run(self, on_resize, stop_event, sleep=None):
        """Blocking loop: every interval_s, tick() and -- ONLY on a change -- call the seam
        `on_resize(old_units, new_units)`. `sleep` and `stop_event` are injectable so tests drive
        it without waiting 20 s.

        THE INTEGRATION SEAM: `on_resize(old_units, new_units)` is where a future wiring step
        rebuilds the resident-layer set / adjusts the hosted expert count to match `new_units`
        (evict layers on a shrink, load layers on a grow, tear down to nothing on new_units==0).
        Sizing the actual footprint to match is the CALLER's responsibility; this loop only tells
        it WHEN and to WHAT. `stop_event` needs an `.is_set()`; `sleep(seconds)` defaults to
        time.sleep."""
        if sleep is None:
            sleep = time.sleep
        while not stop_event.is_set():
            old = self.current_units
            new = self.tick()
            if new is not None:
                on_resize(old, new)
            sleep(self.interval_s)

    @classmethod
    def from_env(cls, device, base_gib, per_unit_gib, max_units):
        """Opt-in factory. Returns a configured tuner, or None when NEURAHASH_VRAM_AUTOTUNE is not
        set (DEFAULT OFF -- with the flag unset this whole subsystem never runs and behaviour is
        byte-identical to the static-guard-only path). Env knobs:

            NEURAHASH_VRAM_AUTOTUNE            on/off master switch (default off -> returns None)
            NEURAHASH_VRAM_USER_HEADROOM_GB    GiB reserved for the owner        (default 1.5)
            NEURAHASH_VRAM_AUTOTUNE_INTERVAL_S re-check period, seconds          (default 20)
            NEURAHASH_VRAM_AUTOTUNE_MIN_UNITS  floor (0 => allowed to pause)     (default 0)
            NEURAHASH_VRAM_AUTOTUNE_MAX_UNITS  ceiling                           (default = arg)
        """
        if not _truthy(os.environ.get("NEURAHASH_VRAM_AUTOTUNE", "")):
            return None
        headroom = float(os.environ.get("NEURAHASH_VRAM_USER_HEADROOM_GB", "1.5"))
        interval = float(os.environ.get("NEURAHASH_VRAM_AUTOTUNE_INTERVAL_S", "20"))
        min_units = int(os.environ.get("NEURAHASH_VRAM_AUTOTUNE_MIN_UNITS", "0"))
        max_env = os.environ.get("NEURAHASH_VRAM_AUTOTUNE_MAX_UNITS")
        eff_max = int(max_env) if max_env not in (None, "") else int(max_units)
        return cls(device=device, base_gib=base_gib, per_unit_gib=per_unit_gib,
                   user_headroom_gib=headroom, min_units=min_units, max_units=eff_max,
                   interval_s=interval)
