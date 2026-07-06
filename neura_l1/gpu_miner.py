"""
neura_l1.gpu_miner — REAL GPU work for the desktop miner, capped to a user-chosen VRAM
budget.

The L1's *verifiable* proof-of-useful-work is a small, deterministic CPU model (that
smallness is what makes it cheaply checkable by every node). This module adds, on top of
that, genuine GPU training the user can size: pick a VRAM budget and the miner

  1. HARD-CAPS its GPU memory to that budget (torch.cuda.set_per_process_memory_fraction),
     so the miner process can never exceed it; and
  2. builds a real PyTorch MoE transformer sized to actually USE most of the budget and runs
     real forward/backward/optimizer steps each block.

So "how much VRAM for mining" is literally true. HONEST SCOPE: this GPU training is the heavy
compute *effort*; the block that earns the reward is still produced by the verifiable CPU
PoUW (wiring the GPU model into consensus is a separate, larger change — see SECURITY_L1.md).

Everything is defensive: no CUDA / no torch / an OOM under the cap all degrade gracefully to
"CPU-only mining" rather than crashing the miner.
"""

BYTES_PER_PARAM = 16          # fp32 weight + grad + Adam m + v (the training-state rule)
STEPS_PER_BLOCK = 4           # real GPU training steps run per mined block
_VOCAB = 256
_SEQ = 128

DEFAULT_RESERVE_GB = 2.0      # always leave at least this much VRAM free for other apps / the OS
DEFAULT_MAX_UTIL_FRAC = 0.8   # never claim more than this fraction of what is CURRENTLY free


# ===========================================================================
# VRAM/compute sizing policy — pure math, unit-tested without a GPU
# ===========================================================================
def plan_budget(requested_gb, free_gb, reserve_gb=DEFAULT_RESERVE_GB,
                max_util_frac=DEFAULT_MAX_UTIL_FRAC):
    """Largest VRAM budget that (a) honours the user's request, (b) leaves `reserve_gb` free for
    everything else, and (c) never exceeds `max_util_frac` of what is CURRENTLY free. This is
    what lets the miner auto-adjust to a shared card: size to free-VRAM-now, not the card total.
    Returns 0.0 when there isn't enough free VRAM to run within the reserve."""
    headroom = min(free_gb - reserve_gb, free_gb * max_util_frac)
    return max(0.0, min(float(requested_gb), headroom))


def duty_cycle_sleep(step_seconds, target_util_frac):
    """Seconds to sleep after a compute step so the long-run GPU duty cycle ~= target_util_frac
    (i.e. the card is deliberately NOT fully utilised). From step/(step+sleep)=target ->
    sleep = step*(1-t)/t. Full speed (t>=1) or degenerate inputs -> no sleep."""
    t = float(target_util_frac)
    if t >= 1.0 or t <= 0.0 or step_seconds <= 0.0:
        return 0.0
    return step_seconds * (1.0 - t) / t


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def total_vram_gb(device=0):
    try:
        import torch
        return torch.cuda.get_device_properties(device).total_memory / 1e9
    except Exception:
        return None


def device_name(device=0):
    try:
        import torch
        return torch.cuda.get_device_properties(device).name
    except Exception:
        return None


def free_vram_gb(device=0):
    """Currently-FREE VRAM in GB, driver-level across ALL processes (including the user's other
    GPU work), or None without CUDA. The signal the auto-adjust sizing is built on."""
    try:
        import torch
        free, _total = torch.cuda.mem_get_info(device)
        return free / 1e9
    except Exception:
        return None


class GpuTrainer:
    """Owns a capped GPU MoE-training workload sized to `budget_gb`. Build it on the mining
    worker thread; call step() each block and mem() to report usage; close() to release."""

    def __init__(self, budget_gb, total_gb=None, device=0,
                 reserve_gb=DEFAULT_RESERVE_GB, max_util_frac=DEFAULT_MAX_UTIL_FRAC):
        import torch
        from neurahash_torch.model_torch import MoETransformer
        self.torch = torch
        self.device = device
        self.requested_gb = float(budget_gb)
        self.reserve_gb = float(reserve_gb)
        self.max_util_frac = float(max_util_frac)
        total = total_gb or total_vram_gb(device) or budget_gb
        # AUTO-ADJUST: size to what's FREE right now (not the card total), leaving a reserve so we
        # never fully utilise the GPU or starve the user's other work. Falls back to the raw
        # request if free VRAM can't be read.
        free = free_vram_gb(device)
        if free is not None:
            eff = plan_budget(self.requested_gb, free, self.reserve_gb, self.max_util_frac)
            if eff <= 0:
                raise RuntimeError(
                    f"only {free:.1f} GB VRAM free — not enough to mine with a "
                    f"{self.reserve_gb:.1f} GB reserve; free some VRAM or lower the reserve")
            self.budget_gb = eff
        else:
            self.budget_gb = self.requested_gb
        # HARD CAP: the miner process can never use more than the chosen budget. Do NOT floor the
        # fraction UP — on a large card a tiny auto-adjusted budget divided by total can fall below
        # a 2% floor, and rounding it up would let the process exceed the very reserve plan_budget
        # computed. Instead REFUSE when the budget is too small a slice of the card to mine.
        target = self.budget_gb / total
        if target < 0.02:
            raise RuntimeError(
                f"budget {self.budget_gb:.2f} GB is < 2% of the {total:.0f} GB card; refusing "
                f"to mine rather than rounding the memory cap up past the reserve")
        frac = min(0.95, target)
        torch.cuda.set_per_process_memory_fraction(frac, device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        self.model = None
        self.opt = None
        self.cfg = self._build_to_budget(MoETransformer)

    # ---- sizing -------------------------------------------------------
    def _build_to_budget(self, MoETransformer):
        """Size a model to use a healthy fraction of the budget WITHOUT risking an OOM at the
        cap edge: target ~45% of the budget in training state, warm up the optimizer during
        the build (so later steps allocate nothing new), and shrink-and-retry on OOM. Returns
        the chosen config dict."""
        torch = self.torch
        b = self.budget_gb
        d_model = 256 if b < 4 else (512 if b < 12 else 1024)
        batch = 8 if b < 4 else (16 if b < 12 else 24)
        per_layer = BYTES_PER_PARAM * 36 * d_model * d_model     # ~36 d^2 params/layer
        n_layers = int(max(2, min(48, (0.45 * b * 1e9) // per_layer)))

        for _ in range(6):
            try:
                model, opt = self._try_build(MoETransformer, d_model, n_layers, batch)
                self.model, self.opt = model, opt
                return {"d_model": d_model, "n_layers": n_layers, "batch": batch,
                        "seq": _SEQ, "params": sum(p.numel() for p in model.parameters())}
            except torch.cuda.OutOfMemoryError:
                self._free()
                if n_layers > 2:
                    n_layers = max(2, n_layers // 2)
                elif batch > 2:
                    batch = max(2, batch // 2)
                elif d_model > 128:
                    d_model //= 2
                else:
                    break
        raise RuntimeError("could not build any GPU model under the VRAM cap")

    def _try_build(self, MoETransformer, d_model, n_layers, batch):
        torch = self.torch
        self._free()
        model = MoETransformer(vocab_size=_VOCAB, d_model=d_model, n_head=8,
                               n_layers=n_layers, d_ff=4 * d_model, n_experts=4,
                               block_size=_SEQ).cuda(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        # warm up the FULL per-block workload (Adam state + activations) so step() later
        # never allocates anything new and thus never OOMs at the cap edge.
        for _ in range(STEPS_PER_BLOCK):
            self._step(model, opt, batch)
        torch.cuda.synchronize(self.device)
        return model, opt

    # ---- training -----------------------------------------------------
    def _step(self, model, opt, batch):
        torch = self.torch
        idx = torch.randint(0, _VOCAB, (batch, _SEQ), device=f"cuda:{self.device}")
        tgt = torch.randint(0, _VOCAB, (batch, _SEQ), device=f"cuda:{self.device}")
        _, loss = model(idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach().item())

    def step(self, n=STEPS_PER_BLOCK):
        """Run `n` real GPU training steps. Returns the last loss, or None on OOM (caught)."""
        try:
            last = None
            for _ in range(n):
                last = self._step(self.model, self.opt, self.cfg["batch"])
            return last
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            return None

    # ---- runtime auto-adjust -----------------------------------------
    def headroom_ok(self):
        """True iff there is still enough free VRAM to keep mining without crowding other apps.
        The mining loop should call this each block and, when it returns False, SKIP the GPU
        steps for that block (degrade to CPU mining) — a live model can't be shrunk in place, so
        pausing is the safe response to the user's other work growing. Unknown free => True."""
        free = free_vram_gb(self.device)
        return True if free is None else free >= self.reserve_gb

    # ---- reporting / teardown ----------------------------------------
    def mem(self):
        torch = self.torch
        return (torch.cuda.memory_allocated(self.device) / 1e9,
                torch.cuda.max_memory_allocated(self.device) / 1e9)

    def _free(self):
        self.model = None
        self.opt = None
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self):
        self._free()
        try:
            self.torch.cuda.set_per_process_memory_fraction(1.0, self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass
