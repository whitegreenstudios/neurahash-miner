"""UNIFIED VramManager for the NeuraHash public miner -- ONE object, ONE source of truth.

WHY THIS EXISTS (the owner directive 2026-07-21):
    Three independent mechanisms used to each decide "how much VRAM" on their own, and could
    CONTRADICT each other:
      1. the GUARD   -- a STATIC hard per-process ceiling chosen ONCE at startup from free VRAM
                        (apply_vram_guard in tools/sharddiloco_glm_contributor.py; apply_vram_cap
                        in sharded_pool_node.py). It never revisits its number.
      2. the CAPACITY report -- a STATIC per-worker usable-VRAM figure advertised in the join
                        handshake (worker_usable_vram_gb -> NEURAHASH_CAPACITY_AWARE in
                        sharded_pool_node.py) that the coordinator uses to SIZE the work. Also
                        chosen once, so it can promise more than the guard's ceiling allows.
      3. the TUNER   -- a DYNAMIC re-check-every-20s footprint autotuner (AdaptiveVramTuner in
                        neurahash/vram_autotune.py) that shrinks/grows the resident layer set.

    A static cap set when the card was idle, a join-time capacity report from that same idle
    moment, and a tuner that later shrinks the footprint can all disagree: the worker keeps
    advertising the old high capacity (so the coordinator over-TASKS it) while the tuner has
    already shed layers and the static cap is now the wrong size for the live free VRAM.

    VramManager fixes that by composing all three behind ONE object whose SINGLE SOURCE OF TRUTH
    is the LIVE FREE VRAM read by detect(). Every tick calls detect() ONCE and feeds that same
    `free` figure to:
      - apply_cap()        -- the hard per-process ceiling (the GUARD), sized from live free;
      - report_capacity()  -- the sustainable training-unit count to advertise (what CAPACITY_AWARE
                              consumes), so the coordinator reassigns instead of over-tasking;
      - the tuner's tick   -- the resize decision (shrink fast / grow slow) that rebuilds the
                              actual resident footprint.
    Because all three derive from the SAME `free`, they can never disagree about how much VRAM is
    free -- the guard's cap, the advertised capacity, and the trained footprint stay consistent.

COST MODEL (identical to vram_autotune, memory glm-capacity-per-card-2026-07-21):
    footprint(units) = base_gib + units * per_unit_gib
    A "unit" = one RESIDENT MoE layer (~1.125 GiB); the base trunk is ~4.0 GiB. Experts within a
    resident layer are FREE, so the tunable quantity is the number of resident LAYERS.

DEFAULT OFF: from_env() returns None unless NEURAHASH_VRAM_MANAGER is truthy. With the flag unset
    this whole object never runs and the caller's existing static-guard path is byte-identical.

Kept import-light on purpose (torch is imported lazily, only inside apply_cap's side-effect): the
guard's device-index + free-VRAM logic is REUSED from vram_autotune (free_total_vram_gib /
_cuda_device_index) rather than re-parsed here, so the fixed cuda:N / total-not-free bugs cannot be
reintroduced, and importing this module never drags in the 280 KB sharded_pool_node monolith or the
GLM stack.
"""

import os
import time

from neurahash.vram_autotune import (
    AdaptiveVramTuner,
    free_total_vram_gib,
    _cuda_device_index,
    _truthy,
)

# Mirrored (NOT imported) from tools/sharddiloco_glm_contributor.apply_vram_guard so this module
# stays import-light -- importing the contributor drags in torch + the whole GLM training stack.
# These are the guard's two default-cap knobs; the free-based term is what makes concurrent launches
# self-limiting, the footprint-multiple term keeps one runaway process from eating a whole card.
DEFAULT_HEADROOM = 0.90   # default cap = this fraction of CURRENTLY-FREE VRAM
RUNAWAY_SLACK = 1.5       # ... and never more than this multiple of the role's max footprint


class VramManager:
    """One object that unifies the GUARD (hard cap), the CAPACITY report, and the TUNER, all off a
    single live-free-VRAM detect(). See the module docstring for the why.

    Construct via from_env(...) (opt-in, DEFAULT OFF) or directly with a configured AdaptiveVramTuner
    as the tuning engine. The tuner owns the cost-model constants (base_gib, per_unit_gib, min/max
    units, hysteresis); the manager reuses them so there is exactly one cost model, not two.
    """

    def __init__(self, device, tuner, base_gib=None, per_unit_gib=None,
                 default_headroom=DEFAULT_HEADROOM, runaway_slack=RUNAWAY_SLACK):
        self.device = device
        self.tuner = tuner
        # Reuse the tuner's cost model by default so the cap, the report, and the resize footprint
        # are all computed from the SAME base/per_unit -- a second copy is how the three drift apart.
        self.base_gib = float(tuner.base_gib if base_gib is None else base_gib)
        self.per_unit_gib = float(tuner.per_unit_gib if per_unit_gib is None else per_unit_gib)
        self.default_headroom = float(default_headroom)
        self.runaway_slack = float(runaway_slack)
        self.interval_s = float(tuner.interval_s)

    # ---- the SINGLE SOURCE OF TRUTH ----------------------------------------------------------
    def detect(self):
        """(free_gib, total_gib) read live from the driver for this manager's device. The ONE call
        every other method's number derives from. Device-index aware and CPU-safe (returns
        (0.0, 0.0) off-GPU) because it is the tuner's tested reader, not a re-parse."""
        return free_total_vram_gib(self.device)

    # ---- cost model (shared by cap + report + resize) ----------------------------------------
    def footprint_gib(self, units):
        """GiB the resident set occupies at `units` layers: base trunk + units * per-layer."""
        return self.base_gib + max(0, int(units)) * self.per_unit_gib

    @property
    def max_footprint_gib(self):
        """The largest footprint this role can ever hold (base + max_units). Used as the guard's
        runaway ceiling: the cap is never sized above 1.5x this, so one buggy process can't eat a
        whole card its siblings still need."""
        return self.footprint_gib(self.tuner.max_units)

    # ---- the CAPACITY report (what NEURAHASH_CAPACITY_AWARE consumes) -------------------------
    def report_capacity(self, free_gib=None):
        """Sustainable training UNITS (resident layers/experts) that fit RIGHT NOW, for the worker
        to advertise to the coordinator so it sizes the work to what actually fits. This is exactly
        AdaptiveVramTuner.target_units(free) -- the SAME quantity the tuner resizes toward -- so the
        number the coordinator schedules against and the footprint we train can never diverge.

        On a SHRINK (owner grabs VRAM) target == the tuner's held units, so re-reporting this LOWER
        number is what makes the coordinator reassign instead of over-tasking a card that just shed
        layers. On a GROW it reports the higher instantaneous fit (which genuinely fits in current
        free); the actual resident-layer REBUILD is paced by the tuner's grow-hysteresis in run()."""
        if free_gib is None:
            free_gib, _ = self.detect()
        return self.tuner.target_units(free_gib)

    # ---- the GUARD (hard per-process ceiling) ------------------------------------------------
    def cap_gib(self, free_gib, total_gib):
        """PURE cap NUMBER (GiB), mirroring apply_vram_guard's sizing so there is one vocabulary:
        an explicit NEURAHASH_VRAM_CAP_GB wins; else NEURAHASH_VRAM_CAP_FRAC * card total; else
        min(free * DEFAULT_HEADROOM, max_footprint * RUNAWAY_SLACK). No torch, no side effect --
        split out so it is unit-testable without CUDA and so apply_cap() only wraps the side effect.

        CONSISTENCY (the single-source-of-truth invariant): with no env override, the default cap is
        >= footprint(report_capacity(free)) for the SAME free -- because the tuner guarantees
        footprint <= free*safety_frac - headroom < free*DEFAULT_HEADROOM, and footprint <=
        max_footprint <= max_footprint*RUNAWAY_SLACK. So the hard ceiling always covers the capacity
        we advertise; we never promise the coordinator more than the cap permits."""
        cap_gb = os.environ.get("NEURAHASH_VRAM_CAP_GB")
        cap_frac = os.environ.get("NEURAHASH_VRAM_CAP_FRAC")
        if cap_gb:
            return float(cap_gb)
        if cap_frac:
            return float(cap_frac) * total_gib
        return min(free_gib * self.default_headroom, self.max_footprint_gib * self.runaway_slack)

    def apply_cap(self, free_gib=None, total_gib=None):
        """Apply the hard per-process CUDA ceiling from LIVE FREE VRAM, then return the cap (GiB).

        Reuses the tuner's device-index reader (via _cuda_device_index) and acts on the device's OWN
        index, so an indexed device ('cuda:1') is capped on GPU 1 -- NOT silently ignored (the exact
        cuda:N bug that was fixed) and NOT mis-applied to GPU 0. Sized from FREE, never from total,
        so co-located workers self-limit rather than oversubscribing the card and spilling to system
        RAM. Unlike the startup guard this NEVER raises/refuses: in the live loop we cap continuously
        and let the tuner SHED layers on the next tick rather than crash the process. No-op number
        on CPU / no CUDA / unreadable total; any torch failure degrades to returning the number
        uncapped rather than breaking the loop."""
        if free_gib is None or total_gib is None:
            free_gib, total_gib = self.detect()
        if total_gib <= 0:
            return None
        cap = self.cap_gib(free_gib, total_gib)
        idx = _cuda_device_index(self.device)
        if idx is None:                       # CPU / non-CUDA: the number, no side effect
            return cap
        try:
            import torch
            if not torch.cuda.is_available():
                return cap
            frac = max(0.0, min(1.0, cap / total_gib))
            torch.cuda.set_per_process_memory_fraction(frac, idx)
        except Exception:
            return cap                        # never let a cap failure break the round loop
        return cap

    # ---- the unified ~20s loop ---------------------------------------------------------------
    def run(self, on_resize, on_report, stop_event, sleep=None):
        """Blocking loop. Every interval_s: detect() ONCE, then from that single `free`:
            1. re-apply_cap()                 -- resize the hard ceiling to live free VRAM;
            2. on_report(report_capacity())   -- re-advertise the sustainable capacity so the
                                                 coordinator reassigns (prevents over-tasking);
            3. on_resize(old, new) ONLY on a change -- the tuner's shrink-fast/grow-slow decision
                                                 (the SEAM where the caller rebuilds the resident set).
        All three read the SAME `free`, so they cannot disagree. `stop_event` needs `.is_set()`;
        `sleep(seconds)` is injectable so tests drive it without waiting 20 s.

        THE on_resize SEAM: on_resize(old_units, new_units) is where the caller evicts/loads resident
        MoE layers to match new_units (tear down to nothing on new_units==0). This loop only decides
        WHEN and to WHAT; sizing the real footprint is the caller's job (see the contributor wiring).
        """
        if sleep is None:
            sleep = time.sleep
        while not stop_event.is_set():
            free_gib, total_gib = self.detect()          # ONE detection = the single source of truth
            self.apply_cap(free_gib=free_gib, total_gib=total_gib)
            on_report(self.report_capacity(free_gib=free_gib))
            old = self.tuner.current_units
            new = self.tuner.tick(free_gib=free_gib)      # SAME free drives the resize decision
            if new is not None:
                on_resize(old, new)
            sleep(self.interval_s)

    # ---- opt-in factory (DEFAULT OFF) --------------------------------------------------------
    @classmethod
    def from_env(cls, device, base_gib, per_unit_gib, max_units):
        """Opt-in factory. Returns a configured VramManager, or None when NEURAHASH_VRAM_MANAGER is
        not set (DEFAULT OFF -- with the flag unset the caller's existing static-guard path runs
        unchanged and behaviour is byte-identical). Reuses the autotuner's headroom/interval/units
        knobs so the manager's internal tuner is configured exactly like a standalone one:

            NEURAHASH_VRAM_MANAGER             master switch (default off -> returns None)
            NEURAHASH_VRAM_USER_HEADROOM_GB    GiB reserved for the owner        (default 1.5)
            NEURAHASH_VRAM_AUTOTUNE_INTERVAL_S re-check period, seconds          (default 20)
            NEURAHASH_VRAM_AUTOTUNE_MIN_UNITS  floor (0 => allowed to pause)     (default 0)
            NEURAHASH_VRAM_AUTOTUNE_MAX_UNITS  ceiling                           (default = max_units arg)
            NEURAHASH_VRAM_CAP_GB / _CAP_FRAC  hard-cap overrides (same names as the guard)
        """
        if not _truthy(os.environ.get("NEURAHASH_VRAM_MANAGER", "")):
            return None
        headroom = float(os.environ.get("NEURAHASH_VRAM_USER_HEADROOM_GB", "1.5"))
        interval = float(os.environ.get("NEURAHASH_VRAM_AUTOTUNE_INTERVAL_S", "20"))
        min_units = int(os.environ.get("NEURAHASH_VRAM_AUTOTUNE_MIN_UNITS", "0"))
        max_env = os.environ.get("NEURAHASH_VRAM_AUTOTUNE_MAX_UNITS")
        eff_max = int(max_env) if max_env not in (None, "") else int(max_units)
        tuner = AdaptiveVramTuner(
            device=device, base_gib=base_gib, per_unit_gib=per_unit_gib,
            user_headroom_gib=headroom, min_units=min_units, max_units=eff_max,
            interval_s=interval)
        return cls(device=device, tuner=tuner)
