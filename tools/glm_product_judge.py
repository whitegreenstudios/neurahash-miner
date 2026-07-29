#!/usr/bin/env python
"""THE PRODUCT-SHAPED ACCEPT JUDGE -- held-out CE on a k-resident-layer GLM (default k=24).

WHY THIS EXISTS (measured, not preference)
------------------------------------------
Run 5's accept gate scored a network with ONE resident MoE layer. That network is not the product:
45 of 46 MoE layers return EXACTLY ZERO (tools/piece_loader.py `_DeadExperts.forward`), so the gate
was optimising something the fleet does not ship. It PAID FOR DAMAGE -- the 25 accepted deltas
improved the 1-layer metric by +15.93% while costing the real 47-layer model **-11.7 ARC-Easy
points** (docs/research/ARC_EASY_RUN5_2026-07-27.md) and **+13.4% full-model CE**
(docs/research/FULLMODEL_CE_RUN5_2026-07-27.md).

The residency bisection (docs/research/GATE_SIGNFLIP_2026-07-27.md) measured where a cheap judge
starts telling the truth:

    k = 18  ->  -0.02488 CE  ->  IMPROVES  (wrong sign, agrees with the 1-layer gate)
    k = 19  ->  +0.01263 CE  ->  WORSENS   (correct sign; the flip point, margin only 0.19%)
    k = 24  ->  +0.15840 CE  ->  WORSENS   (correct sign, margin 2.42%, 36 s + 31 s per pair)
    k = 46+ ->  WORSENS, monotone

k* = 19 is the MINIMUM; **k = 24 is the chosen judge** -- a gate sized exactly at the flip point
mis-rules the next delta set that is 0.2% better than this one. Bit-exact across two independent
loaders (piece_loader vs pack+streaming agreed to all 17 digits at k=12 and k=14), and k=19
reproduced bit-exactly in two separate processes, so the reproducibility floor of this measurement
is 0.

WHAT IS PRESERVED FROM THE RESEARCH TOOL (tools/glm_arc_eval.py, untracked)
--------------------------------------------------------------------------
This module is the SHIPPABLE refactor of that tool's measurement machinery. Four behaviours were
paid for in crashes and wrong numbers and are reproduced here exactly:

  1. VRAM CAP BEFORE THE FIRST CUDA CALL (`cap_vram`). An uncapped one-card load took 32,077 of
     32,607 MiB and crashed the box twice on 2026-07-24 (memory `vram-cap-live-verified`).
     NEURAHASH_VRAM_CAP_GB, default 23.0, and never more than total-8 GiB.
  2. REUSE THE MODEL'S OWN EXPERT MODULES (`ExpertStreamer`). transformers wraps the experts class
     with `use_experts_implementation`, whose forward dispatches on `config._experts_implementation`
     ('grouped_mm' here, fp32 accumulation). A module built from a separately-loaded config has that
     field None and silently falls back to a naive bf16 `index_add_` loop -- logits diverge by up to
     0.25. So weight TENSORS are swapped on the model's own modules; a module is never constructed.
     `selftest()` is the gate that catches a regression here.
  3. torch.inference_mode / no_grad on every eval path -- this is a judge, never a trainer.
  4. BIT-EXACT PATCH READ-BACK. Every row written into a resident slab is read back and compared
     against the value that was written; a mismatch fails closed instead of scoring a network whose
     weights are not the ones the caller asked for. Rows are RESTORED after each pair so the same
     built model can judge event after event.

THE CHUNK-32 CONFIGURATION IS PART OF THE MEASUREMENT. The published k-series ran with
NEURAHASH_GLM_EVAL_CHUNK=32 AND an outer chunk of 32, so inner == outer and the forward batch is
exactly 32 sequences (tools/glm_arc_eval.py:1429). `heldout_ce` reads that env var itself; setting
only one of the two changes the arithmetic's grouping. Both are set here, together, in one place.

PUBLIC API
----------
    ProductJudge(k=24, ...).judge_pair(base_state, candidate_delta)
        -> {"ce_before", "ce_after", "delta", "verdict", "accept", ...}
    JudgeGate(...)          off-loop scheduling: submit()/poll(), verdict cache, fail-closed worker
    product_judge_on(env)   the NEURAHASH_SD_PRODUCT_JUDGE flag (default OFF)
    build_gate_from_env()   flag-off -> None (the caller's path stays byte-identical)

STATE COST: the k=24 network needs the layer pack at D:/_glm_layer_experts (52 GB, regenerable in
~77 s with `tools/glm_arc_eval.py --stage pack`) and the frozen held-out pool at
D:/glm_daily/coord/ids_daily_heldout.npy (sha256 prefix 598baadd9afead46). Both are verified, never
assumed; a missing one is a fail-closed judge, never a silently-skipped gate.

CLI (diagnostics only -- the coordinator imports this module, it never shells out to it):
    P=C:/Python313/python.exe
    PYTHONIOENCODING=utf-8 NEURAHASH_VRAM_CAP_GB=23 $P tools/glm_product_judge.py --selftest
    PYTHONIOENCODING=utf-8 NEURAHASH_VRAM_CAP_GB=23 $P tools/glm_product_judge.py --info
"""
import argparse
import hashlib
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ------------------------------------------------------------------------------------- constants
# k=24, not k*=19: 19 is where the sign flips (margin 0.19%), 24 is where the sign is SAFE (2.42%)
# for 36+31 s per pair. See the module docstring / GATE_SIGNFLIP_2026-07-27.md.
DEFAULT_K = 24
# The campaign's calibrated accept margin (docs/GLM_TRAINER_SYNTHESIS_2026-07-28.md "Margin policy").
# A candidate is accepted only when the judge's CE moves DOWN by at least this much.
DEFAULT_MARGIN = 0.005987
# Part of the measurement, not a performance knob (see the docstring): inner == outer == 32.
EVAL_CHUNK = 32
DEFAULT_PACK_DIR = os.environ.get("NEURAHASH_GLM_PACK_DIR", "D:/_glm_layer_experts")
DEFAULT_PIN_LAYERS = 12                 # 12 pinned slabs + the rest from host RAM == the measured point
DEFAULT_RAM_GIB = 18.0                  # host-RAM slab cache budget (1.125 GiB per slab)
DEFAULT_VRAM_CAP_GB = 23.0              # this box: 31.84 GiB card, >= 8 GiB stays with the desktop

SHARD_DIR = os.environ.get("NEURAHASH_GLM_SHARD_DIR", "D:/hf_models/GLM-4.7-Flash-bf16_shards_100mb")
CONFIG_DIR = os.environ.get("NEURAHASH_GLM_CONFIG_DIR", "D:/hf_models/GLM-4.7-Flash-bf16")
HELDOUT_PATH = os.environ.get("NEURAHASH_GLM_HELDOUT", "D:/glm_daily/coord/ids_daily_heldout.npy")
HELDOUT_SHA256_PREFIX = "598baadd9afead46"
# The published full-47 base anchor (docs/research/FULLMODEL_CE_RUN5_2026-07-27.md). Not a gate here
# (the judge runs at k, not 47) -- recorded so a caller can cite it without re-deriving it.
FULL47_BASE_CE = 4.816991

FLAG = "NEURAHASH_SD_PRODUCT_JUDGE"


def _log(msg):
    print(msg, flush=True)


class JudgeUnavailable(RuntimeError):
    """The judge cannot run (missing pack, missing pool, OOM, load error). ALWAYS fail closed: the
    caller must treat this as "not accepted", never as "gate skipped"."""


# ============================================================================== env / flag helpers
def _truthy(v, default="0"):
    return (v if v is not None else default).strip().lower() not in ("", "0", "false", "no", "off")


def product_judge_on(environ=None):
    """Master gate for the k=24 product judge. DEFAULT OFF -- with the flag unset every caller's
    path is byte-identical to today (same truthiness house style as
    sharddiloco_glm_coordinator._sd_async_on / _glm_quorum_on, but defaulting OFF, not ON)."""
    env = os.environ if environ is None else environ
    return _truthy(env.get(FLAG), "0")


def _env_int(name, default, environ=None):
    env = os.environ if environ is None else environ
    try:
        return int(str(env.get(name, "")).strip())
    except (TypeError, ValueError):
        return int(default)


def _env_float(name, default, environ=None):
    env = os.environ if environ is None else environ
    try:
        return float(str(env.get(name, "")).strip())
    except (TypeError, ValueError):
        return float(default)


def cap_vram(log=_log):
    """HARD per-process VRAM ceiling BEFORE the first CUDA allocation.

    Verbatim in effect from tools/glm_arc_eval.py:117-126 -- an uncapped load crashed this box on
    2026-07-24 (memory `vram-cap-live-verified`). Two rules, both load-bearing: the cap is applied
    before anything allocates, and it can never exceed total-8 GiB no matter what the env says, so
    the desktop always keeps 8 GiB."""
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    cap = _env_float("NEURAHASH_VRAM_CAP_GB", 0.0) or DEFAULT_VRAM_CAP_GB
    cap = min(cap, max(1.0, total - 8.0))
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    log("[judge] VRAM cap %.1f of %.1f GiB (>= 8.0 GiB left for the desktop)" % (cap, total))
    return cap, total


# ======================================================================= network-shape helpers
def moe_layers(cfg):
    return list(range(int(cfg.first_k_dense_replace), int(cfg.num_hidden_layers)))


def pack_path(pack_dir, L):
    return os.path.join(pack_dir, "layer_%d.safetensors" % L)


def fmt_ranges(ids):
    """[0,1,2,13,14] -> '0-2,13-14' (the --pieces syntax)."""
    ids = sorted(set(int(i) for i in ids))
    if not ids:
        return ""
    out, s, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append((s, prev))
        s = prev = i
    out.append((s, prev))
    return ",".join("%d-%d" % (a, b) if a != b else str(a) for a, b in out)


_CLEAN_PIECES = {}
_RESIDENT_E = {}


def clean_pieces_by_layer(shard_dir=None):
    """Per MoE layer, the expert pieces lying ENTIRELY inside it -- read off the live manifest, never
    assumed. Straddlers (a piece whose coordinates span two layers) are EXCLUDED, exactly as
    tools/glm_layer_composition_test.py excluded pieces 12 and 25: including one would push the lower
    layer from 60/64 to 64/64 (so layer 1 would stop being byte-identical to the training network) and
    would materialise a full-width slab holding a single real expert that the router then forces every
    token through. With straddlers dropped EVERY resident layer is 60 of 64 -- the density the whole
    k-series was measured at."""
    global _CLEAN_PIECES
    if _CLEAN_PIECES:
        return _CLEAN_PIECES
    import piece_loader as pl
    man = pl.load_manifest(shard_dir or SHARD_DIR, require_pieces=[], require_files=False)
    by = {}
    for p in man["pieces"]:
        name = str(p["piece"])
        if not name.startswith("experts_"):
            continue
        layers = {int(le[0]) for le in p["experts"]}
        if len(layers) == 1:
            by.setdefault(layers.pop(), []).append(int(name.split("_")[1]))
    _CLEAN_PIECES = {L: sorted(v) for L, v in by.items()}
    return _CLEAN_PIECES


def resident_experts_by_layer(shard_dir=None):
    """{layer: [expert ids covered by that layer's CLEAN pieces]} -- the 60 of 64 the piece path makes
    resident. The pack holds all 64, so the pack path masks itself down to this set on every read."""
    global _RESIDENT_E
    if _RESIDENT_E:
        return _RESIDENT_E
    import piece_loader as pl
    man = pl.load_manifest(shard_dir or SHARD_DIR, require_pieces=[], require_files=False)
    clean = clean_pieces_by_layer(shard_dir)
    want = {p for L in clean for p in clean[L]}
    out = {}
    for p in man["pieces"]:
        name = str(p["piece"])
        if not name.startswith("experts_"):
            continue
        if int(name.split("_")[1]) not in want:
            continue
        for le in p["experts"]:
            out.setdefault(int(le[0]), set()).add(int(le[1]))
    _RESIDENT_E = {L: sorted(v) for L, v in out.items()}
    return _RESIDENT_E


def prefix_pieces(k, shard_dir=None):
    """--pieces for 'MoE layers 1..k resident at 60/64 each, everything above k dead'."""
    by = clean_pieces_by_layer(shard_dir)
    ids = []
    for L in range(1, int(k) + 1):
        if L not in by:
            raise JudgeUnavailable("no clean pieces for layer %d" % L)
        ids.extend(by[L])
    return fmt_ranges(ids)


# =================================================================== the streaming expert machinery
class ExpertStreamer:
    """Manage MoE expert weight RESIDENCY around the stock forward. NEVER touches the math.

    CRITICAL (measured 2026-07-27, cost a whole debugging session): this never CONSTRUCTS an experts
    module. transformers wraps the experts class with `use_experts_implementation`
    (integrations/moe.py), whose forward dispatches on `self.config._experts_implementation` --
    'grouped_mm' here, which accumulates in fp32 via reshape+sum. A separately-constructed module,
    built from a config that never passed through a model, has that field set to None and silently
    falls back to the class's naive bf16 `index_add_` loop instead. The two disagree by up to 0.25 in
    logits. So we reuse the model's OWN module objects and swap only their weight tensors, and
    `selftest()` proves streamed == stock BIT-EXACTLY before any verdict is trusted."""

    def __init__(self, model, cfg, pack_dir, pin, device, layers=None):
        import torch
        self.torch = torch
        self.model, self.cfg, self.pack_dir, self.device = model, cfg, pack_dir, device
        self.pin = set(pin)
        self.cache = {}
        self.handles = []
        self.layers = list(layers) if layers is not None else moe_layers(cfg)
        self.bytes_streamed = 0
        for L in self.layers:
            e = model.model.layers[L].mlp.experts
            if not hasattr(e, "config"):
                raise JudgeUnavailable(
                    "layer %d experts module is %s and carries no `config` -- it is not the "
                    "dispatched module and its forward would differ" % (L, type(e).__name__))
            # Drop the Parameters so plain CUDA tensors can be attached/detached freely (assigning a
            # Tensor over a registered Parameter raises TypeError).
            e._parameters.pop("gate_up_proj", None)
            e._parameters.pop("down_proj", None)
            e.gate_up_proj = None
            e.down_proj = None
            dl = model.model.layers[L]
            self.handles.append(dl.register_forward_pre_hook(self._pre(L)))
            self.handles.append(dl.register_forward_hook(self._post(L)))

    def _read(self, L):
        from safetensors import safe_open
        p = pack_path(self.pack_dir, L)
        with safe_open(p, framework="pt", device=str(self.device)) as sf:
            gu = sf.get_tensor("gate_up")
            dn = sf.get_tensor("down")
        self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
        return gu, dn

    def preload_pinned(self):
        for L in sorted(self.pin):
            self.cache[L] = self._read(L)
        return len(self.cache)

    def _pre(self, L):
        def fn(module, args):
            exp = module.mlp.experts
            if L in self.cache:
                gu, dn = self.cache[L]
            else:
                gu, dn = self._read(L)
            exp.gate_up_proj = gu
            exp.down_proj = dn
            return None
        return fn

    def _post(self, L):
        def fn(module, args, output):
            if L not in self.cache:
                exp = module.mlp.experts
                exp.gate_up_proj = None
                exp.down_proj = None
            return None
        return fn

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class PrefixStreamer(ExpertStreamer):
    """ExpertStreamer for a PREFIX residency: layers 1..k live (60 of 64 experts, the other 4 rows
    zeroed and the router already masked by the caller), everything above k already replaced by the
    piece path's `_DeadExperts` placeholder. Two deltas from the base class:

      * the pack holds all 64 rows, so every slab is masked down to the piece path's 60 on read --
        otherwise this would score a DENSER network than the one the k-series measured;
      * non-pinned slabs are cached in HOST RAM after the first read, so a pass costs an H2D copy
        (~10 GiB/s) instead of a 1.125 GiB NVMe read (the full-47 run spent ~1000 s/arm on disk)."""

    def __init__(self, model, cfg, pack_dir, pin, device, layers, resident, ram_budget_gib=1e9):
        self.resident = resident
        self.ram = {}
        # A slab is 1.125 GiB of host RAM. Past the budget the layer is re-read from the pack every
        # pass instead (slower, but a swap storm is worse than slow).
        self.ram_max = max(0, int(float(ram_budget_gib) / 1.125))
        super().__init__(model, cfg, pack_dir, pin, device, layers=layers)

    def _mask(self, L, gu, dn):
        torch = self.torch
        keep = torch.zeros(gu.shape[0], dtype=torch.bool, device=gu.device)
        keep[torch.tensor(self.resident[L], device=gu.device)] = True
        idx = (~keep).nonzero().flatten()
        if idx.numel():
            gu[idx] = 0
            dn[idx] = 0
        return gu, dn

    def _read_to(self, L, dev):
        from safetensors import safe_open
        with safe_open(pack_path(self.pack_dir, L), framework="pt", device=str(dev)) as sf:
            gu = sf.get_tensor("gate_up")
            dn = sf.get_tensor("down")
        return self._mask(L, gu, dn)

    def _read(self, L):
        if L in self.pin:                                   # resident for the whole pass
            gu, dn = self._read_to(L, self.device)
            self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
            return gu, dn
        if L not in self.ram:                               # one disk read per layer per PROCESS
            if len(self.ram) >= self.ram_max:               # RAM budget spent -> straight from disk
                gu, dn = self._read_to(L, self.device)
                self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
                return gu, dn
            self.ram[L] = self._read_to(L, "cpu")
        gu, dn = self.ram[L]
        self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
        return gu.to(self.device), dn.to(self.device)

    def preload_ram(self):
        for L in self.layers:
            if L not in self.pin and L not in self.ram and len(self.ram) < self.ram_max:
                self.ram[L] = self._read_to(L, "cpu")
        return len(self.ram)

    def slab(self, L):
        """The LIVE (gate_up, down) pair a patch must write into, or None when this layer is re-read
        from disk every pass (a write there would be silently discarded on the next forward)."""
        if L in self.cache:
            return self.cache[L]
        if L in self.ram:
            return self.ram[L]
        return None

    def patchable_layers(self):
        return sorted(set(self.cache) | set(self.ram))


def build_trunk_model(device, log=_log, shard_dir=None, config_dir=None):
    """The frozen trunk, resident, with the model's OWN (still meta) experts modules intact.

    This is piece_loader.build_partial_model's steps 1/3/5 minus the expert handling -- it must NOT
    call build_partial_model([]), because that swaps every experts module for a `_DeadExperts`
    placeholder whose forward returns zeros and which carries none of the `use_experts_implementation`
    dispatch. piece_loader's own helpers are reused, never modified."""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteForCausalLM, Glm4MoeLiteRotaryEmbedding)
    import piece_loader as pl

    t0 = time.time()
    man = pl.load_manifest(shard_dir or SHARD_DIR, require_pieces=[])
    cfg = pl._resolve_config(shard_dir or SHARD_DIR, man, config_dir or CONFIG_DIR)
    with init_empty_weights(include_buffers=False):
        model = Glm4MoeLiteForCausalLM(cfg)
    trunk_sd = {}
    tp = os.path.join(man["shard_dir"], "pieces", "trunk.safetensors")
    with safe_open(tp, framework="pt", device=str(device)) as sf:
        for key in sf.keys():
            if pl._is_dead_mtp_key(key, cfg.num_hidden_layers):          # strip_mtp
                continue
            trunk_sd[key] = sf.get_tensor(key)
    model.load_state_dict(trunk_sd, strict=False, assign=True)
    del trunk_sd
    model.model.rotary_emb = Glm4MoeLiteRotaryEmbedding(cfg, device=torch.device(device))
    log("[judge] trunk resident on %s in %.1fs | experts_implementation=%r"
        % (device, time.time() - t0, cfg._experts_implementation))
    return model, cfg


def build_prefix_pack_model(pack_dir, k, pin_layers, device, log=_log, ram_budget_gib=DEFAULT_RAM_GIB,
                            shard_dir=None, config_dir=None):
    """Prefix-k residency built from the LAYER PACK: MoE layers 1..k live at 60/64 experts, everything
    above k dead, expert slabs streamed so k is bounded by host RAM rather than by the VRAM cap. This
    is the exact network the k-series measured (validated bit-exactly against the piece_loader path at
    the overlapping k=12 and k=14 points). Returns (model, cfg, streamer)."""
    import torch
    import piece_loader as pl
    model, cfg = build_trunk_model(device, log=log, shard_dir=shard_dir, config_dir=config_dir)
    moe = moe_layers(cfg)
    live = [L for L in moe if L <= k]
    resident = resident_experts_by_layer(shard_dir)
    for L in moe:
        mlp = model.model.layers[L].mlp
        if L > k:                                  # dead: EXACTLY the piece path's placeholder
            e = mlp.experts
            mlp.experts = pl._make_dead_experts(e.num_experts, e.hidden_dim, e.intermediate_dim,
                                                e.act_fn)
            continue
        res = resident[L]
        if len(res) < int(cfg.n_routed_experts):    # resident-only routing, verbatim piece_loader
            bias = mlp.gate.e_score_correction_bias
            keep = torch.zeros(int(cfg.n_routed_experts), dtype=torch.bool, device=bias.device)
            keep[torch.tensor(res, device=bias.device)] = True
            bias.masked_fill_(~keep, float("-inf"))
    pin = [L for L in live if L <= max(1, int(pin_layers))]
    streamer = PrefixStreamer(model, cfg, pack_dir, pin, device, live, resident,
                              ram_budget_gib=ram_budget_gib)
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_left:
        raise JudgeUnavailable("%d parameter(s) still on meta after the trunk load, e.g. %r"
                               % (len(meta_left), meta_left[:5]))
    t0 = time.time()
    n_pin = streamer.preload_pinned()
    n_ram = streamer.preload_ram()
    model.eval()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    log("[judge] k=%d: %d live MoE layers (%d pinned on GPU, %d cached in RAM = %.1f GiB), %d dead | "
        "preloaded in %.0fs" % (k, len(live), n_pin, n_ram, n_ram * 1.125, len(moe) - len(live),
                                time.time() - t0))
    return model, cfg, streamer


# ================================================================================= the yardstick
def verify_heldout(path=None):
    """GATE: the frozen yardstick is the file we think it is (memory: probe+heldout are FROZEN, so
    the goal metric's ruler cannot move under a campaign)."""
    p = path or HELDOUT_PATH
    if not os.path.exists(p):
        raise JudgeUnavailable("held-out pool missing at %s" % p)
    with open(p, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if not digest.startswith(HELDOUT_SHA256_PREFIX):
        raise JudgeUnavailable("held-out pool at %s is not the frozen yardstick (sha256 %s..., "
                               "expected prefix %s)" % (p, digest[:16], HELDOUT_SHA256_PREFIX))
    return digest


def load_heldout(path=None):
    """The frozen pool as an [N, T] int64 array, after `verify_heldout`."""
    import numpy as np
    digest = verify_heldout(path)
    return np.load(path or HELDOUT_PATH), digest


def chunked_heldout_ce(G, model, ids, chunk=EVAL_CHUNK):
    """THE GATE'S OWN CE. Identical arithmetic to
    `sharddiloco_glm_coordinator._chunked_heldout_ce` over `sharddiloco_glm_expert.heldout_ce`
    (length-weighted mean over equal-length chunks == the mean over the whole pool), reimplemented
    here rather than imported so this module has NO import edge back into the coordinator that wires
    it in. Kept byte-for-byte in step with that function: if one changes, both change."""
    n = len(ids)
    if chunk <= 0 or n <= chunk:
        return float(G.heldout_ce(model, ids))
    tot, seen = 0.0, 0
    for s in range(0, n, chunk):
        part = ids[s:s + chunk]
        tot += float(G.heldout_ce(model, part)) * len(part)
        seen += len(part)
    return tot / seen


# ========================================================================== row plumbing / digests
_ROW_KEYS = ("gate", "up", "down")


def _as_coord(c):
    if isinstance(c, str):
        L, E = c.split(":")
        return int(L), int(E)
    return int(c[0]), int(c[1])


def normalize_state(state, coord=None):
    """Accept either {coord: {gate,up,down}} or a bare {gate,up,down} (+ `coord`), and return the
    mapping form with tuple keys. One entry per COORDINATE -- the same currency
    `GlmExpertLaneHost.read_slot` / `write_slot` speak, so a caller never has to reshape anything."""
    if state is None:
        return {}
    if all(k in state for k in _ROW_KEYS):
        if coord is None:
            raise ValueError("a bare {gate,up,down} state needs an explicit coord=(layer,expert)")
        return {_as_coord(coord): dict(state)}
    return {_as_coord(k): dict(v) for k, v in state.items()}


def add_delta(base, delta):
    """base + delta, per coordinate, per row -- float64 accumulation then back to the base dtype, so
    the sum does not depend on the order numpy happens to broadcast in."""
    import numpy as np
    out = {}
    for c, rows in base.items():
        d = delta.get(c)
        if d is None:
            out[c] = {k: np.asarray(v) for k, v in rows.items()}
            continue
        out[c] = {}
        for k in _ROW_KEYS:
            b = np.asarray(rows[k])
            out[c][k] = (b.astype(np.float64) + np.asarray(d[k]).astype(np.float64)).astype(b.dtype)
    return out


def state_digest(state):
    """Stable sha256 over a coordinate->rows mapping. Used to key the base-arm cache and to detect
    that a candidate is the SAME candidate a verdict was computed for."""
    import numpy as np
    h = hashlib.sha256()
    for c in sorted(state):
        h.update(("%d:%d|" % c).encode("ascii"))
        for k in _ROW_KEYS:
            a = np.ascontiguousarray(np.asarray(state[c][k], dtype=np.float32))
            h.update(k.encode("ascii"))
            h.update(str(a.shape).encode("ascii"))
            h.update(a.tobytes())
    return h.hexdigest()


# ================================================================================ the judge itself
class ProductJudge:
    """Held-out CE on the k-resident-layer network -- the accept gate that tracks the PRODUCT's sign.

    Build is lazy and one-shot: the first `judge_pair` pays the model build (trunk + k slabs), every
    later pair reuses it. The BASE arm is cached by base digest, so an unchanged base costs one
    forward pass per event instead of two (36 s -> 31 s at k=24; the k-series measured 36 s base +
    31 s trained)."""

    def __init__(self, k=DEFAULT_K, margin=DEFAULT_MARGIN, pack_dir=None, pin_layers=None,
                 ram_budget_gib=None, chunk=EVAL_CHUNK, device="cuda", heldout_path=None,
                 shard_dir=None, config_dir=None, log=_log):
        self.k = int(k)
        self.margin = float(margin)
        self.pack_dir = pack_dir or DEFAULT_PACK_DIR
        self.pin_layers = DEFAULT_PIN_LAYERS if pin_layers is None else int(pin_layers)
        self.ram_budget_gib = DEFAULT_RAM_GIB if ram_budget_gib is None else float(ram_budget_gib)
        self.chunk = int(chunk)
        self.device = device
        self.heldout_path = heldout_path or HELDOUT_PATH
        self.shard_dir, self.config_dir = shard_dir, config_dir
        self.log = log
        self._model = self._cfg = self._streamer = self._G = None
        self._ids = None
        self._pristine = {}          # (L,E) -> the pack's own rows, restored on close()
        self._base_ce = {}           # base digest -> CE (the "one cached number per round" arm)
        self.pairs_judged = 0

    # ---------------------------------------------------------------- build / teardown
    def preflight(self):
        """Everything that can be checked WITHOUT touching CUDA. Called before the build so a missing
        pack or a swapped yardstick is a clean JudgeUnavailable, not a half-loaded GPU."""
        verify_heldout(self.heldout_path)
        if not os.path.isdir(self.pack_dir):
            raise JudgeUnavailable("layer pack dir missing at %s (regenerate: "
                                   "tools/glm_arc_eval.py --stage pack)" % self.pack_dir)
        missing = [L for L in range(1, self.k + 1) if not os.path.exists(pack_path(self.pack_dir, L))]
        if missing:
            raise JudgeUnavailable("layer pack incomplete: %d file(s) missing for k=%d, e.g. layer %d"
                                   % (len(missing), self.k, missing[0]))
        return True

    def ensure_built(self):
        if self._model is not None:
            return
        self.preflight()
        # The chunk is part of the MEASUREMENT: heldout_ce reads NEURAHASH_GLM_EVAL_CHUNK for its own
        # inner batching, and the published k-series ran inner == outer == 32. Set both, together.
        os.environ["NEURAHASH_GLM_EVAL_CHUNK"] = str(self.chunk)
        try:
            import sharddiloco_glm_contributor as N
            self._G = N._G()
            if str(self.device).startswith("cuda"):
                cap_vram(log=self.log)                       # BEFORE the first CUDA allocation
            t0 = time.time()
            self._model, self._cfg, self._streamer = build_prefix_pack_model(
                self.pack_dir, self.k, self.pin_layers, self.device, log=self.log,
                ram_budget_gib=self.ram_budget_gib, shard_dir=self.shard_dir,
                config_dir=self.config_dir)
            ids, digest = load_heldout(self.heldout_path)
            self._ids = ids
            self.log("[judge] READY: k=%d resident MoE layers, chunk=%d, pool=%s (%s, sha %s...), "
                     "margin=%.6f, built in %.0fs"
                     % (self.k, self.chunk, os.path.basename(self.heldout_path), tuple(ids.shape),
                        digest[:16], self.margin, time.time() - t0))
        except JudgeUnavailable:
            self.close()
            raise
        except Exception as ex:                              # noqa: BLE001 -- every build failure is fail-closed
            self.close()
            raise JudgeUnavailable("judge build failed: %s: %s" % (type(ex).__name__, ex))

    def close(self):
        try:
            if self._streamer is not None:
                self._restore_pristine()
                self._streamer.close()
        except Exception:                                    # noqa: BLE001 -- teardown never raises
            pass
        self._model = self._cfg = self._streamer = None
        self._ids = None
        self._pristine = {}
        self._base_ce = {}
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                    # noqa: BLE001
            pass

    # ---------------------------------------------------------------- row patching (bit-exact)
    def _slab_for(self, L):
        s = self._streamer.slab(L)
        if s is None:
            raise JudgeUnavailable(
                "layer %d is streamed from disk on every pass (pinned=%s, ram-cached=%s), so a patch "
                "there would be discarded before the forward. Raise --pin-layers or the RAM budget."
                % (L, sorted(self._streamer.pin), sorted(self._streamer.ram)))
        return s

    def _snapshot_pristine(self, coords):
        for (L, E) in coords:
            if (L, E) in self._pristine:
                continue
            gu, dn = self._slab_for(L)
            I = int(self._cfg.moe_intermediate_size)
            self._pristine[(L, E)] = (gu[E, :I].clone(), gu[E, I:].clone(), dn[E].clone())

    def _restore_pristine(self):
        import torch
        I = int(self._cfg.moe_intermediate_size) if self._cfg is not None else 0
        with torch.no_grad():
            for (L, E), (g, u, d) in list(self._pristine.items()):
                s = self._streamer.slab(L)
                if s is None:
                    continue
                gu, dn = s
                gu[E, :I].copy_(g)
                gu[E, I:].copy_(u)
                dn[E].copy_(d)
        self._pristine = {}

    def _patch(self, state):
        """Write coordinate rows into the resident slabs and READ THEM BACK BIT-EXACTLY.

        The read-back compares against the value that was actually written (i.e. after the float32 ->
        slab-dtype cast), which is the only comparison that means anything on a bf16 slab. A mismatch
        is fail-closed: scoring a network whose weights are not the ones the caller asked for is the
        exact failure class this judge exists to end."""
        import numpy as np
        import torch
        I = int(self._cfg.moe_intermediate_size)
        bad = []
        with torch.no_grad():
            for (L, E), rows in state.items():
                gu, dn = self._slab_for(L)
                want = {}
                for key, dst in (("gate", gu[E, :I]), ("up", gu[E, I:]), ("down", dn[E])):
                    src = torch.as_tensor(np.ascontiguousarray(np.asarray(rows[key]))).to(
                        dst.device, dst.dtype)
                    if tuple(src.shape) != tuple(dst.shape):
                        raise JudgeUnavailable(
                            "coordinate (L%d,E%d) row %r has shape %s, the resident slab wants %s"
                            % (L, E, key, tuple(src.shape), tuple(dst.shape)))
                    dst.copy_(src)
                    want[key] = src
                for key, got in (("gate", gu[E, :I]), ("up", gu[E, I:]), ("down", dn[E])):
                    if not torch.equal(got, want[key]):
                        bad.append("L%d,E%d.%s" % (L, E, key))
        if bad:
            raise JudgeUnavailable("patch read-back FAILED (not bit-exact) at %r" % (bad[:6],))
        return len(state)

    # ---------------------------------------------------------------- the measurement
    def _ce(self, tag):
        import torch
        t0 = time.time()
        with torch.inference_mode():                          # a judge, never a trainer
            ce = chunked_heldout_ce(self._G, self._model, self._ids, self.chunk)
        dt = time.time() - t0
        self.log("[judge] %-22s k=%d held-out CE = %.6f  (%.0fs, %d seqs, chunk=%d)"
                 % (tag, self.k, ce, dt, len(self._ids), self.chunk))
        return float(ce), round(dt, 1)

    def judge_pair(self, base_state, candidate_delta, coord=None, absolute=False, margin=None):
        """Score BASE vs BASE+DELTA on the k-resident network and return the verdict.

        base_state       {(L,E): {gate,up,down}} (or a bare row dict + coord=) -- the coordinate's
                         CURRENT canonical weights, i.e. what `GlmExpertLaneHost.read_slot` returns.
        candidate_delta  the same shape; ADDITIVE by default. Pass absolute=True when you already
                         hold the post-merge weights rather than the difference.
        Returns dict(ce_before, ce_after, delta, verdict, accept, margin, k, ...). `accept` is True
        iff delta <= -margin: the CE must go DOWN by at least the campaign's calibrated margin.
        Raises JudgeUnavailable on anything that stops the measurement -- callers FAIL CLOSED."""
        m = self.margin if margin is None else float(margin)
        base = normalize_state(base_state, coord)
        cand_in = normalize_state(candidate_delta, coord)
        if not base:
            raise JudgeUnavailable("judge_pair called with an empty base state")
        cand = cand_in if absolute else add_delta(base, cand_in)
        missing = [c for c in cand if c not in base]
        if missing:
            raise JudgeUnavailable("candidate touches coordinate(s) %r absent from the base state"
                                   % (missing[:4],))
        self.ensure_built()
        self._snapshot_pristine(base)

        t0 = time.time()
        bdig = state_digest(base)
        if bdig in self._base_ce:
            ce_before, t_b = self._base_ce[bdig], 0.0
            self.log("[judge] ARM 1 BASE           k=%d held-out CE = %.6f  (cached)"
                     % (self.k, ce_before))
        else:
            self._patch(base)
            ce_before, t_b = self._ce("ARM 1 BASE")
            self._base_ce = {bdig: ce_before}                 # one cached number, only the latest base
        n_patched = self._patch(cand)
        ce_after, t_t = self._ce("ARM 2 CANDIDATE")
        # Leave the network on the BASE again: the next event's base arm is then a cache hit and the
        # resident weights never silently carry a rejected candidate into the following verdict.
        self._patch(base)

        delta = ce_after - ce_before
        accept = bool(delta <= -m)
        self.pairs_judged += 1
        out = dict(k=self.k, chunk=self.chunk, margin=m, coords=sorted("%d:%d" % c for c in cand),
                   patched=n_patched, ce_before=ce_before, ce_after=ce_after, delta=delta,
                   frac_delta=(ce_before - ce_after) / ce_before if ce_before else 0.0,
                   verdict="IMPROVES" if delta < 0 else "WORSENS", accept=accept,
                   base_digest=bdig, candidate_digest=state_digest(cand),
                   eval_s_base=t_b, eval_s_candidate=t_t, wall_s=round(time.time() - t0, 1))
        self.log("[judge] ===== k=%d %s: before=%.6f after=%.6f delta=%+.6f (%+.2f%%) margin=%.6f "
                 "=> %s ====="
                 % (self.k, ",".join(out["coords"]), ce_before, ce_after, delta,
                    100.0 * out["frac_delta"], m, "ACCEPT" if accept else "REJECT"))
        return out


# ======================================================================= off-loop scheduling
class JudgeGate:
    """Run the (expensive) judge OFF the caller's event loop.

    The k=24 pair costs 36 s + 31 s measured. An event-driven lane that blocks that long stops
    discovering work, stops evicting idle seats and stops servicing every other coordinate. So the
    verdict is computed on ONE background worker thread and the caller polls for it:

        v = gate.poll(coord, key)          # None -> not ready; requeue the work and carry on
        if v is None: gate.submit(coord, key, base, cand); continue
        if not v["accept"]: reject

    `key` is the caller's own fingerprint of (base, candidate) -- when the base moves under a
    coordinate the key changes, the stale verdict is discarded and a fresh job is submitted, so a
    verdict can never be applied to a candidate it was not computed for.

    The two states handed to `submit` are the lane's own bare {gate,up,down} slot dicts (or already
    coordinate-keyed mappings); `submit` normalizes them, and the CANDIDATE is the ABSOLUTE post-merge
    state by default (`absolute=True`) -- that is what a merge leaves behind in `experts[e_idx]`.

    FAIL CLOSED, ALWAYS. A worker that raises, OOMs or exceeds its wall budget still produces a
    verdict -- accept=False with `error` set and a distinct `tag` for the log -- so the caller always
    makes progress and a broken judge can never turn into a silently-open gate. One job at a time and
    one thread total: the judge owns a 25 GiB model, and two concurrent builds would OOM the card.

    The worker function is injectable (`judge_fn`) so the scheduling logic is unit-testable with a
    small fake network -- no GLM, no pack, no GPU."""

    def __init__(self, judge=None, judge_fn=None, timeout_s=None, max_rejudges=3, absolute=True,
                 log=_log):
        self.judge = judge
        # absolute=True because a caller holds the POST-MERGE weights, not the difference: the lane's
        # `experts[e_idx]` after shard_merge_round IS the candidate. Re-deriving a delta as
        # (candidate - base) and re-adding it would not reproduce the candidate bit-exactly in float32,
        # so the judge would score a network nobody proposed.
        self.absolute = bool(absolute)
        self._fn = judge_fn or self._default_fn
        self.timeout_s = _env_float("NEURAHASH_SD_JUDGE_TIMEOUT_S", 900.0) \
            if timeout_s is None else float(timeout_s)
        self.max_rejudges = int(max_rejudges)
        self.log = log
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue = []             # [(coord, key, base, cand)]
        self._inflight = None        # (coord, key) currently being judged
        self._done = {}              # (coord, key) -> verdict dict
        self._attempts = {}          # coord -> how many DISTINCT keys we have judged for it
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nh-product-judge", daemon=True)
        self._thread.start()
        self.submitted = 0
        self.completed = 0
        self.failed = 0

    # -------------------------------------------------------------- the worker
    def _default_fn(self, base, cand):
        if self.judge is None:
            raise JudgeUnavailable("JudgeGate has neither a ProductJudge nor a judge_fn")
        return self.judge.judge_pair(base, cand, absolute=self.absolute)

    def _run(self):
        while not self._stop.is_set():
            with self._cv:
                while not self._queue and not self._stop.is_set():
                    self._cv.wait(0.5)
                if self._stop.is_set():
                    return
                coord, key, base, cand = self._queue.pop(0)
                self._inflight = (coord, key)
            t0 = time.time()
            try:
                v = dict(self._fn(base, cand))
                v.setdefault("accept", False)
                v.setdefault("tag", "JUDGE-OK")
                v["error"] = None
                self.completed += 1
            except BaseException as ex:                       # noqa: BLE001 -- a judge crash is a REJECT
                v = dict(accept=False, error="%s: %s" % (type(ex).__name__, ex),
                         tag="JUDGE-FAILCLOSED", ce_before=None, ce_after=None, delta=None,
                         verdict="UNAVAILABLE")
                self.failed += 1
            v["wall_s"] = round(time.time() - t0, 1)
            with self._cv:
                self._done[(coord, key)] = v
                self._inflight = None
                self._cv.notify_all()

    # -------------------------------------------------------------- caller-side API
    def submit(self, coord, key, base_state, candidate_delta):
        """Queue one verdict. Returns False (and queues nothing) when this coordinate has already
        burned `max_rejudges` distinct candidates -- a base that keeps moving under a coordinate must
        not be able to spin the judge forever; the caller treats False as a fail-closed REJECT.

        The two states are NORMALIZED to coordinate-keyed mappings here, so a caller holding the lane's
        own bare {gate,up,down} slot dict (GlmExpertLaneHost.read_slot's shape) does not have to know
        that `judge_pair` needs a coord alongside a bare dict -- the coordinate is right there in the
        call. Without this every real candidate would fail closed on a ValueError."""
        coord = _as_coord(coord)
        base_state = normalize_state(base_state, coord)
        candidate_delta = normalize_state(candidate_delta, coord)
        with self._cv:
            if (coord, key) in self._done or self._inflight == (coord, key):
                return True
            if any(q[0] == coord and q[1] == key for q in self._queue):
                return True
            seen = self._attempts.setdefault(coord, set())
            if key not in seen and len(seen) >= self.max_rejudges:
                return False
            seen.add(key)
            self._queue.append((coord, key, base_state, candidate_delta))
            self.submitted += 1
            self._cv.notify_all()
        return True

    @staticmethod
    def key(coord, base_state, candidate_state):
        """Fingerprint of one (coordinate, base, candidate) triple -- see `candidate_key`. Exposed as a
        method so a caller holding only the gate needs no direct import of this module."""
        return candidate_key(coord, base_state, candidate_state)

    def poll(self, coord, key):
        """The verdict for exactly this (coord, key), or None while it is queued/running."""
        with self._lock:
            return self._done.get((_as_coord(coord), key))

    def wait(self, coord, key, timeout=None):
        """Blocking variant -- for tests and for a caller that has nothing else to do. On timeout
        returns a fail-closed verdict rather than raising."""
        coord = _as_coord(coord)
        end = time.time() + (self.timeout_s if timeout is None else float(timeout))
        with self._cv:
            while (coord, key) not in self._done:
                left = end - time.time()
                if left <= 0:
                    return dict(accept=False, tag="JUDGE-TIMEOUT", verdict="UNAVAILABLE",
                                error="no verdict within %.0fs" % (self.timeout_s if timeout is None
                                                                   else float(timeout)),
                                ce_before=None, ce_after=None, delta=None)
                self._cv.wait(min(left, 0.25))
            return self._done[(coord, key)]

    def pending(self):
        """COORDINATEs with a verdict still in flight -- the caller parks these instead of merging."""
        with self._lock:
            out = {q[0] for q in self._queue}
            if self._inflight is not None:
                out.add(self._inflight[0])
            return out

    def forget(self, coord, key=None):
        """Drop cached verdicts for a coordinate once they have been acted on (bounded memory)."""
        coord = _as_coord(coord)
        with self._lock:
            for kk in [kk for kk in self._done if kk[0] == coord and (key is None or kk[1] == key)]:
                self._done.pop(kk, None)

    def stats(self):
        with self._lock:
            return dict(submitted=self.submitted, completed=self.completed, failed=self.failed,
                        queued=len(self._queue), inflight=self._inflight is not None,
                        cached=len(self._done))

    def close(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        try:
            self._thread.join(2.0)
        except RuntimeError:
            pass
        if self.judge is not None:
            self.judge.close()


def candidate_key(coord, base_state, candidate_state):
    """The caller's fingerprint of one (coordinate, base, candidate) triple. A verdict is only ever
    applied to the key it was computed for, so a base that moved while the judge ran invalidates the
    verdict instead of silently mis-ruling the next candidate."""
    return hashlib.sha256(
        ("%d:%d|" % _as_coord(coord)).encode("ascii")
        + state_digest(normalize_state(base_state, coord)).encode("ascii") + b"|"
        + state_digest(normalize_state(candidate_state, coord)).encode("ascii")).hexdigest()


def build_gate_from_env(margin=None, environ=None, log=_log, judge_fn=None):
    """Construct the gate iff NEURAHASH_SD_PRODUCT_JUDGE is on. Flag off -> None, and every caller
    keeps its byte-identical legacy path. Construction NEVER raises: a judge that cannot be built is
    still a gate (every verdict fail-closed), because "the judge is broken" must reject candidates,
    not wave them through."""
    env = os.environ if environ is None else environ
    if not product_judge_on(env):
        return None
    k = _env_int("NEURAHASH_SD_JUDGE_K", DEFAULT_K, env)
    m = _env_float("NEURAHASH_SD_JUDGE_MARGIN", DEFAULT_MARGIN, env) if margin is None else float(margin)
    if judge_fn is not None:
        gate = JudgeGate(judge=None, judge_fn=judge_fn, log=log)
        log("[judge] %s on: injected judge_fn, margin=%.6f" % (FLAG, m))
        return gate
    judge = ProductJudge(k=k, margin=m,
                         pack_dir=env.get("NEURAHASH_GLM_PACK_DIR", DEFAULT_PACK_DIR),
                         pin_layers=_env_int("NEURAHASH_SD_JUDGE_PIN_LAYERS", DEFAULT_PIN_LAYERS, env),
                         ram_budget_gib=_env_float("NEURAHASH_SD_JUDGE_RAM_GIB", DEFAULT_RAM_GIB, env),
                         chunk=_env_int("NEURAHASH_GLM_JUDGE_CHUNK", EVAL_CHUNK, env), log=log)
    try:
        judge.preflight()
        log("[judge] %s on: k=%d margin=%.6f chunk=%d pack=%s -- PRODUCT-SHAPED accept gate ACTIVE "
            "(a candidate is accepted only if held-out CE on the %d-layer network falls by >= the "
            "margin)" % (FLAG, k, m, judge.chunk, judge.pack_dir, k))
    except JudgeUnavailable as ex:
        log("[judge] %s on but the judge is NOT usable (%s) -- the gate stays ACTIVE and every "
            "candidate will be REJECTED fail-closed until it is fixed" % (FLAG, ex))
    return JudgeGate(judge=judge, log=log)


# =========================================================================== selftest / CLI (manual)
def selftest(pack_dir=None, log=_log):
    """EQUIVALENCE GATE, preserved from the research tool: same weights, two mechanisms -> the logits
    must be BIT-IDENTICAL. This is what proves the streaming machinery does not silently change the
    model (it is what caught the `_experts_implementation=None` fallback, 0.25 in logits)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors import safe_open

    pack_dir = pack_dir or DEFAULT_PACK_DIR
    cap_vram(log=log)
    dev = "cuda"
    cfg = AutoConfig.from_pretrained(CONFIG_DIR)
    cfg.num_hidden_layers = 2                 # layer 0 dense + layer 1 MoE
    cfg.num_nextn_predict_layers = 0
    if getattr(cfg, "num_local_experts", None) is None:
        cfg.num_local_experts = cfg.n_routed_experts
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG_DIR, config=cfg, dtype=torch.bfloat16, local_files_only=True).to(dev).eval()
    log("[judge/selftest] 2-layer reference model on %s | experts_implementation=%r"
        % (dev, model.config._experts_implementation))

    g = torch.Generator(device="cpu").manual_seed(7)
    ids = torch.randint(0, int(cfg.vocab_size), (3, 48), generator=g).to(dev)
    with torch.inference_mode():
        ref = model(input_ids=ids, use_cache=False).logits.clone()

    exp0 = model.model.layers[1].mlp.experts
    with safe_open(pack_path(pack_dir, 1), framework="pt", device=dev) as sf:
        pgu, pdn = sf.get_tensor("gate_up"), sf.get_tensor("down")
    same_gu = torch.equal(pgu, exp0.gate_up_proj.data)
    same_dn = torch.equal(pdn, exp0.down_proj.data)
    log("[judge/selftest] packed layer 1 == from_pretrained fused param: gate_up=%s down=%s"
        % (same_gu, same_dn))
    if not (same_gu and same_dn):
        raise JudgeUnavailable("packed layer-1 experts differ from the reference model's own weights "
                               "-- the pack is wrong")

    streamer = ExpertStreamer(model, model.config, pack_dir, pin=(), device=dev, layers=[1])
    with torch.inference_mode():
        got = model(input_ids=ids, use_cache=False).logits.clone()
    streamer.close()
    exact = torch.equal(ref, got)
    maxdiff = float((ref.float() - got.float()).abs().max())
    log("[judge/selftest] streamed vs stock logits: bit-identical=%s max|d|=%.3g shape=%s"
        % (exact, maxdiff, tuple(ref.shape)))
    if not exact:
        raise JudgeUnavailable("streaming changes the model output (max|d|=%.3g)" % maxdiff)
    log("[judge/selftest] PASS -- expert streaming is output-equivalent to stock residency.")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="k=24 product-shaped accept judge (diagnostics CLI)")
    ap.add_argument("--selftest", action="store_true",
                    help="streamed-vs-stock bit-exactness gate (needs the pack + a GPU)")
    ap.add_argument("--info", action="store_true", help="print the judge's configuration and preflight")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--pack-dir", dest="pack_dir", default=DEFAULT_PACK_DIR)
    opts = ap.parse_args(argv)
    if opts.selftest:
        selftest(opts.pack_dir)
        return 0
    j = ProductJudge(k=opts.k, margin=opts.margin, pack_dir=opts.pack_dir)
    _log("[judge] k=%d margin=%.6f chunk=%d pack=%s heldout=%s flag %s=%s"
         % (j.k, j.margin, j.chunk, j.pack_dir, j.heldout_path, FLAG,
            "ON" if product_judge_on() else "OFF (default)"))
    try:
        j.preflight()
        _log("[judge] preflight PASS -- pack complete for k=%d, yardstick frozen (%s...)"
             % (j.k, verify_heldout(j.heldout_path)[:16]))
    except JudgeUnavailable as ex:
        _log("[judge] preflight FAIL (the gate would reject every candidate): %s" % ex)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
