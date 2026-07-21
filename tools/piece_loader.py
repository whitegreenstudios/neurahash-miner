"""Piece LOADER -- the READ side of tools/model_shard.py (Rung B, docs/research/rung-b-fleet-training-plan).

model_shard.py is WRITE-ONLY: it splits a HuggingFace MoE checkpoint into a shared TRUNK piece
(embeddings, attention, layernorms, router/gate, shared-experts, lm_head) + per-EXPERT-group pieces of
~shard_gb each, content-addressed, with a merkle-root manifest. This module is the missing READ side: it
lets a fleet card build a WORKING model that holds ONLY the trunk + its assigned expert pieces resident, and
NEVER materializes the full ~59 GB bf16 model. For GLM-4.7-Flash: trunk (5.67 GB) + ONE expert piece
(6.44 GB) ~= 12.1 GB of weights -- fits a 24 GB card with activations, comfortable on a 32 GB 5090.

WHY the full model is impossible: Glm4MoeLiteForCausalLM is ~31B params, ~59 GB in bf16; no single fleet
card holds it. The whole point of expert-sharding is that a miner trains a disjoint slice of experts against
the shared trunk (memory `rung-b-fleet-training`, `glm52-north-star`).

------------------------------------------------------------------------------------------------------------
NON-RESIDENT EXPERTS: ROUTING-MASK, not raise (design choice)
------------------------------------------------------------------------------------------------------------
The task offered two options for experts NOT in the requested pieces: (a) raise if routed to, or (b) mask
them out of the router's top-k so tokens route only among RESIDENT experts. We implement (b) -- masking --
because "training a slice against resident-only routing" IS the Rung-B regime: a node forward-passes through
all layers, but at a layer it owns only a slice of experts, so tokens must route among the resident slice
(non-resident experts simply do not exist for this node's forward). Raising would make any real forward
impossible; masking yields a finite, trainable forward. Concretely:

  * Fully-non-resident MoE layer (0 experts from the requested pieces): `layer.mlp.experts` is replaced by a
    lightweight `_DeadExperts` placeholder (NO expert weight tensors at all) whose forward returns zeros. The
    layer still runs its SHARED expert (resident, in trunk) + attention, so the residual stream flows
    unbroken; the routed contribution is just zero. `_DeadExperts` deliberately exposes NO `gate_up_proj`
    attribute, so tools/expert_shard_train.attach_expert_lora skips it automatically (it wraps only modules
    that have `gate_up_proj`).
  * Partially-resident MoE layer (some experts resident, some not -- happens at a piece boundary): the fused
    `gate_up_proj [E,2*inter,hidden]` / `down_proj [E,hidden,inter]` params ARE materialized full-width so
    that expert-id indexing stays valid, resident rows are filled from the piece, and non-resident rows stay
    ZERO. Resident-only routing is enforced by writing -inf into the router's `e_score_correction_bias` at
    non-resident expert positions: `Glm4MoeLiteMoE.route_tokens_to_experts` adds that bias to the (sigmoid)
    router logits BEFORE the top-k, so a -inf entry can never be selected, while the true bias of resident
    experts (and thus their relative ranking) is untouched. This relies on n_group==1 for GLM-4.7-Flash
    (single expert group -> group selection is trivial); the non-resident rows are additionally zero-filled
    so even the degenerate case (fewer resident experts than top_k) contributes exactly zero.

------------------------------------------------------------------------------------------------------------
INTEGRATION CONTRACT with tools/expert_shard_train.py (do NOT refactor that file -- out of scope)
------------------------------------------------------------------------------------------------------------
Today expert_shard_train.py / fleet/esh_worker.py do `OlmoeForCausalLM.from_pretrained(FULL_MODEL)` -- they
load the whole model and are OLMoE-hardcoded. The GLM fleet run replaces that single line with:

    from tools.piece_loader import build_partial_model, assigned_expert_ids
    model, summary = build_partial_model(shard_dir, piece_ids=my_pieces, device="cuda", dtype=torch.bfloat16)

Then the EXISTING attach step consumes it unchanged. `attach_expert_lora(model, groups, r, alpha)` iterates
`model.model.layers`, and for every layer whose `mlp.experts` has a `gate_up_proj` (i.e. every RESIDENT MoE
layer produced here) wraps it in `LoRAExperts`; dead layers (no `gate_up_proj`) and the dense layer 0 are
skipped. `groups` is the disjoint per-node partition of (layer, expert) ids; use `assigned_expert_ids(
manifest, piece_ids)` to get exactly the (layer, expert) set THIS node owns and can attach LoRA to (it is a
subset of the resident set). Because LoRAExperts wraps the *fused* Glm4MoeLiteNaiveMoe base tensors (proven a
byte-identical drop-in in tests/test_expert_shard_glm.py), and masking lives one level up in the router, the
LoRA path is agnostic to which experts are resident.

Env: Python C:/Python313/python.exe (never .venv). bf16 throughout; fused 3D expert params are raw
nn.Parameter tensors -- NOT bitsandbytes-quantizable, do not attempt nf4 on them. Keep stdout ASCII (cp1252).
"""

import json
import os
import re
import struct
import time

# torch/transformers/accelerate/safetensors are imported lazily inside the functions that need them so that
# `load_manifest` / `assigned_expert_ids` (pure metadata) work with no heavy deps loaded.

_EXPERT_KEY_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

# ---- opt-in frozen-trunk reduction (docs/research/TRUNK_SIZE_REDUCTION.md). Both default OFF: with
#      strip_mtp=False and trunk_quant=None build_partial_model is byte-identical to before. ----
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# trunk quantizable Linear weights = attention proj + dense-0 MLP + shared-expert MLP (NOT the fused
# routed MoE experts [absent from the trunk], NOT embed/lm_head, NOT norms/router). Matches
# tools/trunk_quant.is_trunk_linear_name. n.b. shared_experts ARE trunk-resident and quantizable.
_TRUNK_QUANT_KEY_RE = re.compile(
    r"(self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)"
    r"|mlp\.(shared_experts\.)?(gate_proj|up_proj|down_proj))\.weight$")
# canonical bitsandbytes NF4 codebook (quantiles of a unit normal, normalized to [-1,1]).
_NF4_LEVELS = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)


def _is_dead_mtp_key(key, n_layers):
    """A trunk key the training model never instantiates (layer index >= n_layers -> MTP/nextn head)."""
    m = _LAYER_RE.search(key)
    if m is not None and int(m.group(1)) >= n_layers:
        return True
    kl = key.lower()
    return any(t in kl for t in ("nextn", "mtp", ".eh_proj"))


def _is_trunk_quant_key(key):
    """True for the trunk's quantizable Linear weights (attention + dense-0 MLP + shared experts). The
    fused routed experts are not present in the trunk piece; embed/lm_head/norms/router are excluded."""
    return _TRUNK_QUANT_KEY_RE.search(key) is not None


def _quant_dequant(w, mode, block=64):
    """Quantize `w` to NF4 or int8 (per-block absmax) then dequantize back to w.dtype. This measures /
    applies the exact frozen-trunk storage-quantization loss on CPU without a bitsandbytes kernel. Kept
    algorithm-identical to tools/trunk_quant.{nf4,int8}_quant_dequant."""
    import torch
    dt = w.dtype
    flat = w.reshape(-1).float()
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    rows = flat.reshape(-1, block)
    absmax = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    if mode == "nf4":
        levels = torch.tensor(_NF4_LEVELS, dtype=torch.float32, device=w.device)
        idx = ((rows / absmax).unsqueeze(-1) - levels).abs().argmin(dim=-1)
        deq = levels[idx] * absmax
    elif mode == "int8":
        scale = absmax / 127.0
        deq = torch.clamp(torch.round(rows / scale), -127, 127) * scale
    else:
        raise ValueError("unknown trunk_quant mode %r (want 'nf4' or 'int8')" % mode)
    return deq.reshape(-1)[:n].reshape(w.shape).to(dt)


# --------------------------------------------------------------------------------------------- manifest I/O
def load_manifest(shard_dir, require_pieces=None, require_files=True):
    """Read + validate shard_dir/model_manifest.json (written by tools/model_shard.write_shards).

    Validates schema (version==1, n_pieces == len(pieces), exactly one 'trunk' piece) and, when
    `require_files`, that the required piece files exist under shard_dir/pieces/. Returns the manifest dict
    (with an added 'shard_dir' key). Raises ValueError/FileNotFoundError/KeyError on any inconsistency.

    `require_pieces` (fleet-run fix, memory `hf-piece-streaming-training`): a FLEET node fetches only its
    assigned expert pieces (a few dozen of the 603), NOT the whole set. When `require_pieces=None` (default)
    EVERY piece the manifest names must be on disk -- the strict, owner-side "complete shard set" check. When
    `require_pieces` is an iterable of expert-piece ids, only the trunk + those requested pieces must exist on
    disk; un-requested pieces are still enumerated in the returned manifest (so assigned_expert_ids /
    build_partial_model see the full expert map) but their files need not be present. This is what lets a pod
    build a WORKING partial model from a partial fetch (build_partial_model passes its piece_ids through).

    `require_files=False` reads the manifest as PURE METADATA and checks NO piece files at all -- for a caller
    that needs only the expert map (e.g. computing a node's layer-block assignment) BEFORE any pieces (not even
    the trunk) have been fetched. This is intentionally distinct from `require_pieces=[]`, which still requires
    the trunk to be present (an empty expert set is still a trunk-only BUILD). Conflating the two crashed cold
    pods with FileNotFoundError on trunk.safetensors during the pre-fetch metadata read (fleet run 2026-07-18)."""
    path = os.path.join(shard_dir, "model_manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError("no model_manifest.json in %s" % shard_dir)
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("version") != 1:
        raise ValueError("unsupported manifest version %r (expected 1)" % man.get("version"))
    pieces = man.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        raise ValueError("manifest has no pieces")
    if man.get("n_pieces") != len(pieces):
        raise ValueError("n_pieces %r != len(pieces) %d" % (man.get("n_pieces"), len(pieces)))
    names = [p["piece"] for p in pieces]
    if names.count("trunk") != 1:
        raise ValueError("expected exactly one 'trunk' piece, found %d" % names.count("trunk"))
    pdir = os.path.join(shard_dir, "pieces")
    if require_pieces is None:
        need = list(names)                                  # strict: every named piece file must be present
    else:
        known = set(names)
        need = ["trunk"]
        for pid in require_pieces:
            nm = "experts_%d" % int(pid)
            if nm not in known:
                raise KeyError("requested piece %r not named in manifest (have %d pieces)" % (nm, len(names)))
            need.append(nm)
    if require_files:
        for nm in need:
            fp = os.path.join(pdir, nm + ".safetensors")
            if not os.path.exists(fp):
                raise FileNotFoundError("piece file missing: %s" % fp)
    man["shard_dir"] = os.path.abspath(shard_dir)
    return man


def _piece_record(manifest, piece_id):
    """Return the manifest record for expert piece `piece_id` (int -> 'experts_<id>'), or raise."""
    name = "experts_%d" % int(piece_id)
    for p in manifest["pieces"]:
        if p["piece"] == name:
            return p
    raise KeyError("no piece %r in manifest (have: %s)" % (name, [p["piece"] for p in manifest["pieces"]]))


def assigned_expert_ids(manifest, piece_ids):
    """The set of (layer, expert) ids covered by the given expert `piece_ids` (list/iterable of ints).

    This is exactly what the LoRA-attach step needs to know which experts THIS node owns. Note: some pieces
    include experts of the multi-token-prediction (MTP / nextn) layer, which a standard Glm4MoeLiteForCausalLM
    does NOT instantiate; those are still returned here (they are simply never wrapped, because
    attach_expert_lora only iterates the model's real layers). build_partial_model reports how many were
    skipped for that reason."""
    out = set()
    for pid in piece_ids:
        rec = _piece_record(manifest, pid)
        for le in rec["experts"]:
            out.add((int(le[0]), int(le[1])))
    return out


# ------------------------------------------------------------------------------------------- config helpers
def _resolve_config(shard_dir, manifest, config_dir=None):
    """Load the Glm4MoeLiteConfig for this shard set. Priority: explicit config_dir -> config.json next to the
    shards -> the original model_dir recorded in the manifest."""
    from transformers import AutoConfig
    candidates = []
    if config_dir:
        candidates.append(config_dir)
    candidates.append(shard_dir)
    md = manifest.get("model_dir")
    if md:
        candidates.append(md)
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "config.json")):
            cfg = AutoConfig.from_pretrained(c)
            # NaiveMoe reads config.num_local_experts (an attribute_map alias for n_routed_experts in current
            # transformers, but guard defensively -- matches tests/test_expert_shard_glm.py).
            if getattr(cfg, "num_local_experts", None) is None:
                cfg.num_local_experts = cfg.n_routed_experts
            return cfg
    raise FileNotFoundError("no config.json found in any of: %s" % [c for c in candidates if c])


# ----------------------------------------------------------------------------- non-resident expert placeholder
def _make_dead_experts(num_experts, hidden_dim, intermediate_dim, act_fn):
    """Build a lightweight `_DeadExperts` stand-in for a fully-non-resident MoE layer's fused experts. It holds
    NO weight tensors (cheap), and deliberately exposes NO `gate_up_proj` attribute so that
    tools/expert_shard_train.attach_expert_lora skips it. Its forward matches Glm4MoeLiteNaiveMoe.forward's
    signature and returns zeros -> the layer contributes only its (resident, trunk) shared expert."""
    import torch.nn as nn

    class _DeadExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = num_experts
            self.hidden_dim = hidden_dim
            self.intermediate_dim = intermediate_dim
            self.act_fn = act_fn
            self.resident = False

        def forward(self, hidden_states, top_k_index, top_k_weights):
            import torch
            return torch.zeros_like(hidden_states)

    return _DeadExperts()


# ---------------------------------------------------------------------------------------------- the loader
def build_partial_model(shard_dir, piece_ids, device="cpu", dtype=None, config_dir=None, verbose=False,
                        strip_mtp=False, trunk_quant=None, quant_block=64):
    """Build a Glm4MoeLiteForCausalLM that holds ONLY the trunk + the experts in `piece_ids` resident.

    Args:
        shard_dir : dir containing model_manifest.json + pieces/*.safetensors (from tools/model_shard.py).
        piece_ids : iterable of expert-piece indices (ints, e.g. [0]) to make resident. The trunk is ALWAYS
                    loaded. An empty list => trunk-only (every MoE layer dead).
        device    : 'cpu' or 'cuda' (or 'cuda:0' ...). Weights are placed directly on this device.
        dtype     : torch dtype for the fused expert params (default torch.bfloat16). Trunk tensors are loaded
                    verbatim from disk (bf16, except the fp32 e_score_correction_bias buffer).
        config_dir: optional override dir holding config.json (else shard_dir, else manifest model_dir).
        strip_mtp : OPT-IN (default False). Drop the dead layer-<n_layers> (MTP/nextn) trunk tensors at load
                    instead of loading-then-discarding them. Behaviorally identical (the model never
                    instantiates them) -- pairs with tools/strip_trunk_mtp.py for the on-disk -1.35 GB win.
        trunk_quant: OPT-IN (default None). None => the current bf16 trunk (unchanged). 'nf4' or 'int8'
                    quantize+dequantize the FROZEN trunk's Linear weights (attention + dense-0 MLP + shared
                    experts) for a storage/VRAM cut; the fused routed experts and embed/lm_head/norms stay
                    bf16. See docs/research/TRUNK_SIZE_REDUCTION.md; gate on held-out CE before fleet use.
        quant_block: absmax block size for trunk_quant (default 64).

    Returns (model, summary_dict). The model is a real, forward-capable partial model:
        * trunk fully resident; rotary buffers real (built on `device`);
        * resident MoE layers have full-width fused params with resident rows filled, non-resident rows zero,
          and the router masked to route only among resident experts;
        * non-resident MoE layers carry a _DeadExperts placeholder (zero routed contribution).
    summary_dict keys: pieces, device, dtype, n_moe_layers, n_expert_slots, n_resident_experts,
        n_placeholder_experts, n_skipped_mtp, resident_layers, partial_layers, dead_layers, meta_params_left.
    """
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteForCausalLM,
        Glm4MoeLiteMoE,
        Glm4MoeLiteRotaryEmbedding,
    )

    if dtype is None:
        dtype = torch.bfloat16
    piece_ids = list(piece_ids)
    device = torch.device(device)
    t0 = time.time()

    manifest = load_manifest(shard_dir, require_pieces=piece_ids)   # tolerate a partial (fleet) fetch
    cfg = _resolve_config(shard_dir, manifest, config_dir)
    n_layers = cfg.num_hidden_layers
    n_experts = cfg.n_routed_experts
    pdir = os.path.join(manifest["shard_dir"], "pieces")

    # 1) empty skeleton on meta (params only -> ~0 bytes). include_buffers=False keeps buffers real, so the
    #    rotary inv_freq and e_score_correction_bias are computed normally (no meta-forward breakage).
    with init_empty_weights(include_buffers=False):
        model = Glm4MoeLiteForCausalLM(cfg)
    # NB: do NOT blanket-cast to `dtype` -- the fp32 e_score_correction_bias buffer must stay fp32
    # (_keep_in_fp32_modules_strict). Trunk tensors carry their disk dtype; fused params are created in `dtype`.

    # 2) resident (layer, expert) set; drop experts of layers the model does not instantiate (MTP/nextn).
    want = assigned_expert_ids(manifest, piece_ids)
    resident_by_layer = {}
    n_skipped_mtp = 0
    for (L, E) in want:
        if L >= n_layers:
            n_skipped_mtp += 1
            continue
        resident_by_layer.setdefault(L, set()).add(E)

    # 3) load the TRUNK piece and assign into the skeleton (assign=True: meta -> real, in place).
    trunk_sd = {}
    with safe_open(os.path.join(pdir, "trunk.safetensors"), framework="pt", device=str(device)) as sf:
        for k in sf.keys():
            if strip_mtp and _is_dead_mtp_key(k, n_layers):
                continue                       # opt-in: skip dead MTP tensors the model discards anyway
            trunk_sd[k] = sf.get_tensor(k)
    trunk_quant_summary = None
    if trunk_quant:                            # opt-in frozen-trunk storage quantization (Linear weights only)
        n_q = 0
        for k in list(trunk_sd):
            if _is_trunk_quant_key(k) and not _is_dead_mtp_key(k, n_layers):
                trunk_sd[k] = _quant_dequant(trunk_sd[k], trunk_quant, quant_block)
                n_q += 1
        trunk_quant_summary = {"mode": trunk_quant, "block": quant_block, "n_linears": n_q}
    missing, unexpected = model.load_state_dict(trunk_sd, strict=False, assign=True)
    # `missing` = the fused expert params (handled next). `unexpected` = layer-<n_layers> (MTP) trunk keys.
    del trunk_sd

    # 4) per MoE layer: materialize resident fused params + mask, or drop in a dead placeholder.
    resident_layers, partial_layers, dead_layers = [], [], []
    n_resident_experts = 0
    open_pieces = {pid: safe_open(os.path.join(pdir, "experts_%d.safetensors" % pid),
                                  framework="pt", device=str(device)) for pid in piece_ids}
    try:
        for L in range(n_layers):
            layer = model.model.layers[L]
            mlp = getattr(layer, "mlp", None)
            if not isinstance(mlp, Glm4MoeLiteMoE):
                continue  # dense layer 0 (Glm4MoeLiteMLP) -- fully in trunk, nothing to do
            experts = mlp.experts
            res = sorted(resident_by_layer.get(L, ()))
            if not res:
                # fully non-resident -> lightweight placeholder, no expert weights.
                mlp.experts = _make_dead_experts(experts.num_experts, experts.hidden_dim,
                                                 experts.intermediate_dim, experts.act_fn)
                dead_layers.append(L)
                continue
            H, I = experts.hidden_dim, experts.intermediate_dim
            gate_up = torch.zeros((n_experts, 2 * I, H), dtype=dtype, device=device)
            down = torch.zeros((n_experts, H, I), dtype=dtype, device=device)
            for E in res:
                pref = "model.layers.%d.mlp.experts.%d." % (L, E)
                sf = _find_piece_with(open_pieces, pref + "gate_proj.weight")
                g = sf.get_tensor(pref + "gate_proj.weight")
                u = sf.get_tensor(pref + "up_proj.weight")
                d = sf.get_tensor(pref + "down_proj.weight")
                gate_up[E].copy_(torch.cat([g, u], dim=0).to(dtype))  # fused: [gate; up] along out-dim
                down[E].copy_(d.to(dtype))
                del g, u, d
            experts.gate_up_proj = nn.Parameter(gate_up, requires_grad=False)
            experts.down_proj = nn.Parameter(down, requires_grad=False)
            n_resident_experts += len(res)
            # resident-only routing: -inf the router bias at non-resident positions (see module docstring).
            if len(res) < n_experts:
                bias = mlp.gate.e_score_correction_bias
                keep = torch.zeros(n_experts, dtype=torch.bool, device=bias.device)
                keep[torch.tensor(res, device=bias.device)] = True
                bias.masked_fill_(~keep, float("-inf"))
                partial_layers.append(L)
            else:
                resident_layers.append(L)
    finally:
        for sf in open_pieces.values():
            try:
                sf.__exit__(None, None, None)
            except Exception:
                pass

    # 5) rotary buffers real on `device` (they are non-persistent -> never in trunk).
    model.model.rotary_emb = Glm4MoeLiteRotaryEmbedding(cfg, device=device)

    # 6) sanity: no PARAMETER should remain on meta.
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]

    n_moe_layers = sum(1 for t in cfg.mlp_layer_types if t == "sparse")
    n_slots = n_moe_layers * n_experts  # each sparse layer has n_experts routed slots
    summary = {
        "pieces": piece_ids,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "n_moe_layers": n_layers - 1,
        "n_expert_slots": n_slots,
        "n_resident_experts": n_resident_experts,
        "n_placeholder_experts": n_slots - n_resident_experts,
        "n_skipped_mtp": n_skipped_mtp,
        "resident_layers": sorted(resident_layers + partial_layers),
        "partial_layers": sorted(partial_layers),
        "dead_layers": dead_layers,
        "meta_params_left": len(meta_left),
        "strip_mtp": bool(strip_mtp),
        "trunk_quant": trunk_quant_summary,
        "wall_s": round(time.time() - t0, 2),
    }
    if meta_left:
        summary["meta_param_examples"] = meta_left[:5]
    if verbose:
        _print_summary(summary)
    return model, summary


def _find_piece_with(open_pieces, key):
    """Return the first open safetensors handle that contains `key` (experts of one layer live in one piece,
    but a boundary layer's experts can straddle two adjacent pieces)."""
    for sf in open_pieces.values():
        if key in sf.keys():
            return sf
    raise KeyError("no requested piece contains %s" % key)


def _print_summary(s):
    print("piece_loader summary")
    print("  pieces loaded      : trunk + experts_%s" % s["pieces"])
    print("  device / dtype     : %s / %s" % (s["device"], s["dtype"]))
    print("  MoE layers         : %d (expert slots %d)" % (s["n_moe_layers"], s["n_expert_slots"]))
    print("  resident experts   : %d" % s["n_resident_experts"])
    print("  placeholder experts: %d" % s["n_placeholder_experts"])
    print("  skipped MTP experts: %d" % s["n_skipped_mtp"])
    print("  full/partial/dead  : %d / %d / %d layers"
          % (len(s["resident_layers"]) - len(s["partial_layers"]), len(s["partial_layers"]),
             len(s["dead_layers"])))
    print("  resident layers    : %s" % s["resident_layers"])
    print("  partial layers     : %s" % s["partial_layers"])
    print("  meta params left   : %d" % s["meta_params_left"])
    print("  wall time (build)  : %.2f s" % s["wall_s"])


# ---------------------------------------------------------------------------- real-shard CPU smoke (CLI)
def _smoke(shard_dir, piece_ids):
    """Build a partial model on CPU from a real shard set and print resident-vs-placeholder counts + RAM."""
    import torch
    try:
        import psutil
        proc = psutil.Process()
        rss0 = proc.memory_info().rss
    except Exception:
        proc = None
        rss0 = 0
    model, summary = build_partial_model(shard_dir, piece_ids, device="cpu", dtype=torch.bfloat16)
    _print_summary(summary)
    if proc is not None:
        peak = proc.memory_info().rss
        print("  process RSS        : %.2f GB (delta %.2f GB)"
              % (peak / 1e9, (peak - rss0) / 1e9))
    # count a couple of real resident weights to prove they are materialized (not meta / not zero)
    L = summary["resident_layers"][0]
    gu = model.model.layers[L].mlp.experts.gate_up_proj
    print("  layer %d gate_up_proj: shape %s dtype %s device %s nonzero-rows %d/%d"
          % (L, tuple(gu.shape), str(gu.dtype).replace("torch.", ""), gu.device,
             int((gu.abs().sum(dim=(1, 2)) > 0).sum()), gu.shape[0]))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="READ side of model_shard.py: build a trunk+pieces partial model.")
    ap.add_argument("shard_dir")
    ap.add_argument("--pieces", default="0", help="comma-separated expert-piece ids, e.g. 0 or 0,1")
    a = ap.parse_args()
    raise SystemExit(_smoke(a.shard_dir, [int(x) for x in a.pieces.split(",") if x != ""]))
