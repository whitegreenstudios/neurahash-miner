"""Layer-major LoRA BACKWARD sweep for GLM-4.7-Flash (RSFT), plus its reference arm.

WHY THIS FILE EXISTS
--------------------
`docs/research/RSFT_TRAINING_PATH_2026-08-13.md` ranks "layer-major LoRA RSFT on the 5090 with a
streamed read-only base" #1, and names the one missing piece: a layer-major BACKWARD sweep. The
43.4x eval win (`ARC_LAYER_MAJOR_WIRING_2026-08-10.md`) was forward-only.

The reverse-sweep SKELETON already exists in `tools/glm_capture_layer_major.py:303-346`, but it
harvests boundary gradients for the grad-cache surrogate; it never accumulates PARAMETER
gradients and never computes a trainable loss. This module adds exactly those two things and
nothing else, reusing the shipped primitives by import:

  * `glm_arc_eval.build_streaming_model` / `ExpertStreamer.hold` / `.release`  (residency)
  * `glm_arc_eval._capture_preamble`                                          (decoder kwargs)
  * `glm_capture_layer_major.GlmStackAdapter` / `_boundary` / `_reload`       (loop + tape)
  * `glm_arc_eval._cap_vram`                                                  (VRAM ceiling)

THE ALGORITHM, AND WHY IT IS EXACT
----------------------------------
Per-layer activation checkpointing with the batch loop hoisted inside the layer loop:

  forward : for L: hold(L); for group g: tape[L+1][g] = layer_L(tape[L][g]); release(L)
  epilogue: for g: gout[g] = d(loss)/d(tape[n][g])
  reverse : for L descending: hold(L); for g: recompute y = layer_L(tape[L][g]) WITH grad,
            then y.backward(gout[g]) -- which (a) writes gout[g] <- d/d(input) and
            (b) ACCUMULATES into every LoRA leaf inside layer L; release(L)

Every layer's VJP sees the same input and the same incoming cotangent it would see in one
monolithic graph, so the result is the true gradient -- not an approximation. The only freedom is
accumulation ORDER, which is why the gate states a tolerance instead of demanding bit-equality.

THE RISK THIS FILE IS BUILT AROUND
----------------------------------
A wrong backward does not crash. It yields a plausible loss curve and a quietly worse model
(memory `pouw-verified-not-useful`: ~900 rounds paid while held-out worsened). Therefore nothing
here is trusted until `lmbwd_gate.py` proves it against stock autograd, from a code path this
file does not own.

CONVENTIONS
-----------
ASCII only in every printed/raised string: this box's console is cp1252 and an em-dash inside a
`raise` becomes a UnicodeEncodeError, turning a loud failure into a silent one.
"""

import os
import sys
import time

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

# Repo defaults (glm_arc_eval.py:143-144, glm_layer_cache_pass.py:89).
CONFIG_DIR = os.environ.get("NEURAHASH_SD_CONFIG_DIR", "D:/hf_models/GLM-4.7-Flash-bf16")
PACK_DIR = os.environ.get("NEURAHASH_SD_LAYER_PACK_DIR", "D:/_glm_layer_experts")
IGNORE_INDEX = -100


def _log(msg):
    """Flushing logger. An unflushed log turns a diagnosable crash into a silent one."""
    sys.stdout.write("%s\n" % msg)
    sys.stdout.flush()


# --------------------------------------------------------------------------- DETERMINISM

# OWNER DECISION 2026-08-13: training uses the DETERMINISTIC math attention backward.
#
# WHY THIS LIVES HERE AND NOT ONLY IN THE GATE. lmbwd_gate.py already defaulted to math SDPA, but
# the selection lived in the GATE ONLY -- so anything that imported this module directly (i.e. a
# real training run) silently got torch's default dispatch instead. That is the exact shape of
# failure this project keeps hitting: a safeguard that exists in the checking path and not in the
# production path, so the thing that ships is not the thing that was verified.
#
# MEASURED 2026-08-13 (scratchpad/lmbwd_gate_sdpadefault.json vs lmbwd_gate_real.json):
#   default dispatch -> memory-efficient attention backward, NONDETERMINISTIC.
#                       26 of 34 LoRA gradient tensors differ run-to-run; self-distance 1.4439e-03.
#   math backend     -> run-to-run self-distance exactly 0.0, asserted with `==`.
# That noise floor is ~1450x the loop-inversion bar (1e-06), so under the default kernel NO version
# of the correctness gate can certify the backward -- `backward_proven_correct` is false there.
# Determinism is a product constraint in this repo (CLAUDE.md; memory
# safetensors-mmap-recompute-determinism), so the slower kernel is the correct default.

def use_deterministic_attention(log=_log, allow_nondeterministic=False):
    """Pin SDPA to the math backend. Call BEFORE any forward/backward that must be reproducible.

    Pass allow_nondeterministic=True only for a throughput experiment whose gradients nothing
    trusts; it logs a LOUD warning because a run that quietly used the fast kernel cannot be
    distinguished afterwards from one that did not.
    """
    import torch
    if allow_nondeterministic:
        log("[lmbwd] WARNING: NONDETERMINISTIC attention backward requested. Gradients from this "
            "run are NOT certifiable (measured run-to-run self-distance 1.4439e-03, ~1450x the "
            "1e-06 loop bar). Do not feed its output to any paid or of-record path.")
        return "default"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    log("[lmbwd] SDPA pinned to math backend (deterministic; owner decision 2026-08-13)")
    return "math"


# --------------------------------------------------------------------------- VRAM

def cap_vram(log=_log):
    """Reuse the shipped cap (glm_arc_eval.py:333). MUST run before the first CUDA allocation.

    An uncapped load crashed this box on 2026-07-24. On Windows WDDM the absence of an OOM does
    NOT prove something fits -- the driver pages to system RAM at 5-9x slowdown -- so the cap is
    what turns a spill into a real exception at a known byte budget.
    """
    import glm_arc_eval as AE
    cap, total = AE._cap_vram()
    log("[lmbwd] VRAM cap %.2f GiB of %.2f GiB total" % (cap, total))
    return cap, total


# --------------------------------------------------------------------------- LoRA

def _lora_linear_cls():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRALinear(nn.Module):
        """y = base(x) + scale * B @ A @ x, computed in fp32, added back in the base dtype.

        Wraps an `nn.Linear` that lives in the TRUNK (attention projections). Trunk modules are
        resident for the whole run and are never touched by ExpertStreamer.hold/release, so this
        wrapper cannot perturb residency accounting. (Wrapping `layer.mlp.experts` WOULD: release
        does `self._layer_mods[L].mlp.experts.gate_up_proj = None` -- glm_arc_eval.py:975-977 --
        which would land on the wrapper and leak the 1.125 GiB slab. See LoRAExpertBlock.)
        """

        def __init__(self, base, r, alpha, gen=None, init_b_nonzero=False):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            dev = base.weight.device
            self.r = int(r)
            self.scale = float(alpha) / float(r)
            i_f, o_f = int(base.in_features), int(base.out_features)
            a = torch.empty(self.r, i_f, device=dev, dtype=torch.float32)
            b = torch.empty(o_f, self.r, device=dev, dtype=torch.float32)
            a.normal_(0.0, 1.0, generator=gen).mul_(0.01)
            # Production init is B=0 so the adapter starts as the identity. The GATE must override
            # that: with B=0, dL/dA is IDENTICALLY zero and the A-gradient comparison would be a
            # vacuous 0-vs-0 pass -- exactly the "gate that cannot fail" this repo shipped before.
            if init_b_nonzero:
                b.normal_(0.0, 1.0, generator=gen).mul_(0.01)
            else:
                b.zero_()
            self.A = nn.Parameter(a)
            self.B = nn.Parameter(b)

        def forward(self, x):
            y = self.base(x)
            d = F.linear(F.linear(x.float(), self.A), self.B)
            return y + (self.scale * d).to(y.dtype)

    return LoRALinear


def _lora_expert_cls():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRAExpertBlock(nn.Module):
        """Per-expert rank-r adapter on the routed-expert block, attached as a FORWARD HOOK.

        A hook is used deliberately instead of wrapping `layer.mlp.experts`: the hook leaves module
        identity untouched, so `ExpertStreamer._pre/_post/release` keep writing gate_up_proj and
        down_proj on the real fused module (glm_arc_eval.py:984-1003, 975-977) and residency
        accounting stays exactly as measured. A wrapper would silently leak the slab.

        Shape: hs [N,H], top_k_index [N,K], top_k_weights [N,K]; output [N,H]. The adapter adds
        sum_k w_nk * scale * B_e A_e hs_n for e = top_k_index[n,k] -- i.e. the same routing and
        the same weighting the base experts use, so it is a true adapter on the expert block.
        """

        def __init__(self, n_experts, hidden, r, alpha, device, gen=None, init_b_nonzero=False):
            super().__init__()
            self.r = int(r)
            self.scale = float(alpha) / float(r)
            a = torch.empty(int(n_experts), self.r, int(hidden), device=device, dtype=torch.float32)
            b = torch.empty(int(n_experts), int(hidden), self.r, device=device, dtype=torch.float32)
            a.normal_(0.0, 1.0, generator=gen).mul_(0.01)
            if init_b_nonzero:
                b.normal_(0.0, 1.0, generator=gen).mul_(0.01)
            else:
                b.zero_()
            self.A = nn.Parameter(a)
            self.B = nn.Parameter(b)

        def __call__(self, module, args, output):
            hs = args[0]
            idx = args[1]
            w = args[2]
            x = hs.float()
            acc = torch.zeros_like(x)
            n_exp = self.A.shape[0]
            # Loop over HIT experts only, mirroring Glm4MoeLiteNaiveMoe.forward's own structure.
            with torch.no_grad():
                hit = torch.unique(idx)
            for e in hit.tolist():
                if e >= n_exp:
                    continue
                pos = (idx == e).nonzero(as_tuple=False)
                if pos.numel() == 0:
                    continue
                tok = pos[:, 0]
                wt = w[tok, pos[:, 1]].float().unsqueeze(-1)
                d = F.linear(F.linear(x[tok], self.A[e]), self.B[e])
                acc = acc.index_add(0, tok, wt * d)
            return output + (self.scale * acc).to(output.dtype)

    return LoRAExpertBlock


ATTN_TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")


def attach_lora(model, n_layers, r=16, alpha=32, seed=0, expert_lora=True,
                init_b_nonzero=False, log=_log):
    """Freeze everything, then attach LoRA to layers [0, n_layers).

    Returns (named_params, handles) where named_params is an ordered list of
    (name, torch.nn.Parameter). Freezing first and creating adapters second is the shipped order
    (sharddiloco_glm_expert.py:716-717) and is what makes the adapters the ONLY autograd leaves --
    an unfrozen trunk populates .grad on 613 tensors for +4.095 GiB
    (glm_layer_cache_pass.py:400-407).
    """
    import torch
    LoRALinear = _lora_linear_cls()
    LoRAExpertBlock = _lora_expert_cls()

    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None

    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    named, handles = [], []
    for L in range(int(n_layers)):
        layer = model.model.layers[L]
        attn = layer.self_attn
        for t in ATTN_TARGETS:
            base = getattr(attn, t, None)
            if base is None or not hasattr(base, "in_features"):
                continue
            w = LoRALinear(base, r, alpha, gen=gen, init_b_nonzero=init_b_nonzero)
            setattr(attn, t, w)
            named.append(("L%d.self_attn.%s.A" % (L, t), w.A))
            named.append(("L%d.self_attn.%s.B" % (L, t), w.B))
        if expert_lora and hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "gate"):
            exp = layer.mlp.experts
            n_e = int(getattr(exp, "num_experts", model.config.n_routed_experts))
            h = int(model.config.hidden_size)
            blk = LoRAExpertBlock(n_e, h, r, alpha, dev, gen=gen, init_b_nonzero=init_b_nonzero)
            # Registered on the model so .to()/state_dict see it; the hook is the call path.
            model.add_module("_lmbwd_expert_lora_%d" % L, blk)
            handles.append(exp.register_forward_hook(blk))
            named.append(("L%d.experts.A" % L, blk.A))
            named.append(("L%d.experts.B" % L, blk.B))

    live = [id(p) for p in model.parameters() if p.requires_grad]
    mine = [id(p) for _n, p in named]
    if sorted(live) != sorted(mine):
        raise RuntimeError(
            "trainable-parameter mismatch: %d live vs %d adapter tensors. Refusing -- a stray "
            "trainable base tensor would blow the memory budget and poison the gradient check."
            % (len(live), len(mine)))
    n_par = sum(int(p.numel()) for _n, p in named)
    log("[lmbwd] LoRA: %d tensors, %d params (r=%d alpha=%d expert_lora=%s b_nonzero=%s)"
        % (len(named), n_par, r, alpha, bool(expert_lora), bool(init_b_nonzero)))
    return named, handles


def zero_grads(named):
    for _n, p in named:
        p.grad = None


def snapshot_grads(named):
    """Detached CPU copies (dtype preserved; compare_grads casts to float64), so a later step
    cannot mutate what the gate compares."""
    out = {}
    for n, p in named:
        out[n] = None if p.grad is None else p.grad.detach().to("cpu", copy=True)
    return out


# --------------------------------------------------------------------------- masked CE

def masked_loss_from_hidden(model, hs, labels, denom, chunk=256):
    """Prompt-masked next-token NLL, SUMMED over kept targets and divided by `denom`.

    Modelled on the shipped `glm_layer_cache_pass.loss_from_hidden` (:242-267) -- same shift, same
    `checkpoint(..., use_reentrant=False)` tail so the [n_tok, vocab] logits are never all live --
    with two changes RSFT requires and which do not exist anywhere in the repo (a repo-wide grep
    for `ignore_index` / -100 label masking over the loss paths returns zero real hits):

      1. positions whose label is IGNORE_INDEX contribute nothing (prompt masking);
      2. the denominator is passed in, not `n`, so a GROUP of microbatches summed together forms
         one mean over the whole group. That is what makes gradient accumulation over a
         layer-major group identical to one big batch.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    _B, _T, H = hs.shape
    flat = hs[:, :-1, :].reshape(-1, H)
    tgt = labels[:, 1:].reshape(-1)
    keep = (tgt != IGNORE_INDEX).nonzero(as_tuple=False).squeeze(-1)
    if keep.numel() == 0:
        return hs.sum() * 0.0
    flat = flat.index_select(0, keep)
    tgt = tgt.index_select(0, keep)
    n = int(flat.shape[0])
    head = model.lm_head

    def tail(x, y):
        lg = head(x).float()
        return (torch.logsumexp(lg, dim=-1) - lg.gather(-1, y.unsqueeze(-1)).squeeze(-1)).sum()

    tot = None
    for i in range(0, n, int(chunk)):
        part = checkpoint(tail, flat[i:i + int(chunk)], tgt[i:i + int(chunk)], use_reentrant=False)
        tot = part if tot is None else tot + part
    return tot / float(denom)


def masked_ce_and_seq_logp(model, hs, labels, denom, chunk=256, gold_only=False):
    """CE exactly as `masked_loss_from_hidden` computes it, PLUS per-sequence log-prob totals.

    Used only by the margin objective (PREREG_E5_MARGIN_OBJECTIVE_2026-08-21.md). A margin term
    needs `log P(completion | prompt)` per SEQUENCE so a gold and a distractor can be compared;
    the CE path only ever needs one pooled scalar, so it sums everything and keeps nothing.

    `masked_loss_from_hidden` is deliberately NOT refactored to share this code. Summing the same
    per-token values in a different order changes the last bits of a float accumulation, and the
    prereg requires the margin-disabled path to be BIT-IDENTICAL to the control. Keeping the old
    function untouched makes that identity structural rather than something to be re-argued.

    Returns (ce_loss, seq_nll_sum[B], seq_kept[B]) where seq_nll_sum is the SUMMED NLL over each
    sequence's kept targets, so -seq_nll_sum/seq_kept is its mean per-token log-prob.

    `gold_only` is the difference between an objective and a contradiction. Rows arrive
    interleaved [gold, rejected, gold, rejected, ...] and the rejected rows carry real labels --
    they must, or their log-prob could not be computed for the margin. But if CE is then taken
    over ALL rows, the loss is CE(gold) + CE(distractor) + margin: cross-entropy actively teaches
    the model to emit the WRONG answer while the margin term tries to push the two apart. Caught
    at step 1 of the first launch, where total loss came back 5.66274 with a margin term of
    1.17711, so the CE half was 4.48563 instead of the control's 3.57880. The per-sequence sums
    are still returned for EVERY row, gold and rejected alike; only the CE scalar is restricted.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    B, T, H = hs.shape
    flat = hs[:, :-1, :].reshape(-1, H)
    tgt = labels[:, 1:].reshape(-1)
    keep = (tgt != IGNORE_INDEX).nonzero(as_tuple=False).squeeze(-1)
    if keep.numel() == 0:
        z = hs.sum() * 0.0
        return z, torch.zeros(B, device=hs.device), torch.zeros(B, device=hs.device)
    seq_of = torch.div(keep, T - 1, rounding_mode="floor")      # which sequence each target is in
    flat = flat.index_select(0, keep)
    tgt = tgt.index_select(0, keep)
    n = int(flat.shape[0])
    head = model.lm_head

    def tail_per_token(x, y):
        # Per-token NLL, NOT summed -- [chunk] floats stay live instead of [chunk, vocab] logits,
        # so this costs the same memory as the pooled version.
        lg = head(x).float()
        return torch.logsumexp(lg, dim=-1) - lg.gather(-1, y.unsqueeze(-1)).squeeze(-1)

    # Which kept targets belong to a GOLD row. Even sequence indices are gold by the interleaving
    # contract; `gold_only=False` restores the plain all-rows behaviour for non-paired callers.
    gold_mask = (seq_of % 2 == 0).to(torch.float32) if gold_only else None

    seq_nll = torch.zeros(B, device=hs.device, dtype=torch.float32)
    tot = None
    for i in range(0, n, int(chunk)):
        part = checkpoint(tail_per_token, flat[i:i + int(chunk)], tgt[i:i + int(chunk)],
                          use_reentrant=False)
        seq_nll = seq_nll.index_add(0, seq_of[i:i + int(chunk)], part)
        s = (part * gold_mask[i:i + int(chunk)]).sum() if gold_only else part.sum()
        tot = s if tot is None else tot + s
    seq_kept = torch.zeros(B, device=hs.device, dtype=torch.float32).index_add(
        0, seq_of, torch.ones(n, device=hs.device, dtype=torch.float32))
    return tot / float(denom), seq_nll, seq_kept


def pair_margin_loss(seq_nll, seq_kept, beta, n_pairs_total):
    """Logistic margin over LENGTH-NORMALISED log-probs for adjacent (gold, distractor) pairs.

    Rows arrive interleaved [gold_0, rej_0, gold_1, rej_1, ...], so pairs are (0,1), (2,3), ...
    within a group -- which is why the batch builder must interleave and why micro-batch must be
    even.

    Length normalisation is REQUIRED, not cosmetic: a distractor's completion is a different
    number of tokens than the gold's, so a margin on raw summed log-probs would be partly a
    length comparison, and the arm would report a length effect as a margin effect.

    Returns (loss, mean_delta, frac_ordered) -- the last two are diagnostics the prereg requires
    logged every step, so a saturated objective (no gradient left in the margin term) is visible
    immediately instead of being inferred from a null eleven hours later.
    """
    import torch
    B = seq_nll.shape[0]
    if B < 2:
        z = seq_nll.sum() * 0.0
        return z, z, z
    lp = -(seq_nll / seq_kept.clamp(min=1.0))          # mean per-token log-prob, per sequence
    gold, rej = lp[0::2], lp[1::2]
    delta = gold - rej
    loss = torch.nn.functional.softplus(-float(beta) * delta).sum() / float(n_pairs_total)
    return loss, delta.detach().mean(), (delta.detach() > 0).float().mean()


def count_kept_gold(groups):
    """Kept targets on the GOLD rows only -- the CE denominator for a paired batch.

    Must exclude the interleaved distractor rows, or the margin arm's CE is divided by
    roughly twice the control's denominator and is not the control's CE at all.
    """
    total = 0
    for g in groups:
        lb = g["labels"][0::2]
        total += int((lb[:, 1:] != IGNORE_INDEX).sum().item())
    return total


def count_pairs(groups):
    return sum(int(g["ids"].shape[0]) // 2 for g in groups)


def count_kept(groups):
    total = 0
    for g in groups:
        lb = g["labels"]
        total += int((lb[:, 1:] != IGNORE_INDEX).sum().item())
    return total


def masking_census(groups):
    """Evidence that the prompt mask is REAL, not merely present in the code.

    Both arms derive their masking from the same `labels` tensor, so a mis-split prompt/completion
    would make them agree perfectly and every gradient stage would still pass while the objective
    was silently next-token-over-the-prompt instead of RSFT. Nothing in a gradient comparison can
    see that. This returns the numbers that can.
    """
    kept = count_kept(groups)
    tot = sum(int(g["labels"][:, 1:].numel()) for g in groups)
    per_row = []
    for g in groups:
        lb = g["labels"]
        for r in range(lb.shape[0]):
            row = lb[r, 1:]
            k = int((row != IGNORE_INDEX).sum().item())
            nz = (row != IGNORE_INDEX).nonzero(as_tuple=False)
            first = int(nz[0].item()) if nz.numel() else -1
            per_row.append({"kept": k, "total": int(row.numel()), "first_scored_pos": first})
    return {"kept": kept, "total_positions": tot,
            "kept_fraction": (kept / tot) if tot else 0.0,
            "rows": per_row[:8], "n_rows": len(per_row),
            "min_kept_row": min((r["kept"] for r in per_row), default=0),
            "max_kept_row": max((r["kept"] for r in per_row), default=0)}


# --------------------------------------------------------------------------- the tape
#
# WHY A TAPE OBJECT EXISTS AT ALL. The layer-major forward keeps ONE boundary activation per
# (group, layer): (n_layers + 1) x n_traces x seq_len x hidden x 2 B. At the design chunk of 128
# traces x 512 tokens over 47 layers that is 48 * 65,536 * 2048 * 2 = 12.88 GB = 12.0 GiB, and the
# original code held every byte of it in HOST RAM (`_boundary(h, "cpu")`). This box has 61.6 GB
# total with ~38 GB free, so the in-RAM tape is the single term that makes the real chunk size
# unaffordable. Moving it to E: (667 GB free) removes the ceiling.
#
# CONSTRAINTS THIS OBEYS, EACH FROM A RECORDED FAILURE IN THIS REPO:
#   * PLAIN FILE HANDLES, NOT MEMMAPS. 60 writable np.memmaps whose total exceeded commit headroom
#     produced os error 1455 and silently left 60 correctly-sized EMPTY files
#     (.claude/rules/10-dispatch.md, 2026-08-03). Nothing here maps anything; each boundary is one
#     sequential write to one handle that is closed immediately.
#   * E:, NEVER D:. D: is the scarcest drive (110 GB free) and holds the base weights that this
#     same run streams at 2.56 GB/s; putting the tape there would contend with the read path.
#     `RSFT_TRAINING_PATH_2026-08-13.md:376-379` says this in advance.
#   * The round trip is BIT-EXACT BY CONSTRUCTION, not by tolerance: `_boundary` was already a pure
#     device copy, and this writes the tensor's own bytes and reads the same bytes back. There is
#     no dtype conversion, no serialization format, no compression. The equivalence check asserts
#     `==`, not a tolerance -- a tape that needed a tolerance would be a different experiment.

_NP_VIEW = {"torch.bfloat16": "int16", "torch.float16": "int16",
            "torch.float32": "int32", "torch.float64": "int64"}


class RamTape(object):
    """The ORIGINAL behaviour, kept as the reference arm for the equivalence check.

    The handle IS the tensor, so this is byte-for-byte what `layer_major_backward` did before the
    tape seam existed.
    """

    kind = "ram"

    def __init__(self, device="cpu"):
        self.device = device
        self.bytes_written = 0
        self.peak_live_bytes = 0
        self._live = 0

    def put(self, key, t):
        import glm_capture_layer_major as LM
        h = LM._boundary(t, self.device)
        n = int(h.numel()) * int(h.element_size())
        self.bytes_written += n
        self._live += n
        self.peak_live_bytes = max(self.peak_live_bytes, self._live)
        return h

    def get(self, h, device):
        import glm_capture_layer_major as LM
        return LM._reload(h, device)

    def drop(self, h):
        if h is not None:
            self._live -= int(h.numel()) * int(h.element_size())

    def close(self):
        pass


# How much room an activation tape needs, MEASURED per volume rather than assumed per machine.
# The larger figure applies when the tape would share a volume with the layer pack the run streams
# from, because there the tape and the weights compete for the same free space. Both are checked
# against shutil.disk_usage at construction; neither is a drive letter.
TAPE_MIN_FREE_GIB = 32.0
TAPE_MIN_FREE_WITH_PACK_GIB = 150.0


class DiskTape(object):
    """Boundary activations on disk: one plain file per (group, layer), removed on drop.

    A handle is (path, dtype_str, shape) -- small enough that the in-RAM footprint of the tape is
    the handle list, not the activations.
    """

    kind = "disk"

    def __init__(self, root, log=_log, keep=False):
        import shutil
        self.root = os.path.abspath(root)
        self._log = log
        self.keep = bool(keep)
        if os.path.isdir(self.root) and not keep:
            shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root, exist_ok=True)
        drv = os.path.splitdrive(self.root)[0].upper()
        # MEASURE the volume; never test its letter. This guard used to be `drv.startswith("D")`,
        # which encoded one machine's layout as a universal law: on the dev box D: really is the
        # scarce drive that holds the streamed weights, but on the 4060 D: is a 902 GB SSD and C:
        # was a 97.5%-full spinning disk. Moving that box's tape onto its fast, empty drive -- the
        # correct action -- was refused by a rule that could only see the letter, and it cost a
        # 14-hour overnight window on both a sweep and a training run.
        free_gib = shutil.disk_usage(self.root).free / 2 ** 30
        pack_dir = os.environ.get("NEURAHASH_SD_LAYER_PACK_DIR", "")
        with_pack = bool(pack_dir) and (
            os.path.splitdrive(os.path.abspath(pack_dir))[0].upper() == drv)
        need = TAPE_MIN_FREE_WITH_PACK_GIB if with_pack else TAPE_MIN_FREE_GIB
        if free_gib < need:
            raise RuntimeError(
                "the activation tape needs %.0f GiB free on %s but has %.1f GiB%s. A tape that "
                "fills the volume its own weights stream from stalls the run it belongs to. "
                "Got %s" % (need, drv or "this volume", free_gib,
                            " (this volume also holds the streamed layer pack)" if with_pack
                            else "", self.root))
        self.bytes_written = 0
        self.peak_live_bytes = 0
        self._live = 0
        self._n_files = 0
        log("[tape] disk tape at %s (plain file handles, no memmaps)" % self.root)

    def _path(self, key):
        return os.path.join(self.root, "b_%s.bin" % ("_".join(str(k) for k in key)))

    def put(self, key, t):
        import torch  # noqa: F401  (kept for symmetry / dtype names)
        c = t.detach().to("cpu").contiguous()
        ds = str(c.dtype)
        nps = _NP_VIEW.get(ds)
        if nps is None:
            raise RuntimeError("no byte-identical view for dtype %s; refusing to convert (a "
                               "conversion would make the tape lossy and the comparison a lie)"
                               % ds)
        import torch as _t
        arr = c.view(getattr(_t, nps)).numpy()
        p = self._path(key)
        with open(p, "wb") as fh:
            arr.tofile(fh)
            fh.flush()
            os.fsync(fh.fileno())
        n = int(c.numel()) * int(c.element_size())
        got = os.path.getsize(p)
        if got != n:
            raise RuntimeError("short write: %s is %d B, expected %d B (a truncated tape reads "
                               "back as plausible garbage)" % (p, got, n))
        self.bytes_written += n
        self._n_files += 1
        self._live += n
        self.peak_live_bytes = max(self.peak_live_bytes, self._live)
        return (p, ds, tuple(int(x) for x in c.shape))

    def get(self, h, device):
        import numpy as np
        import torch
        p, ds, shape = h
        nps = _NP_VIEW[ds]
        arr = np.fromfile(p, dtype=nps)
        want = 1
        for s in shape:
            want *= int(s)
        if arr.size != want:
            raise RuntimeError("tape file %s holds %d elements, expected %d" % (p, arr.size, want))
        t = torch.from_numpy(arr).view(getattr(torch, ds.split(".")[-1])).reshape(shape)
        return t.to(device).clone().detach().requires_grad_(True)

    def drop(self, h):
        if h is None:
            return
        p, ds, shape = h
        n = 1
        for s in shape:
            n *= int(s)
        n *= 2 if _NP_VIEW[ds] == "int16" else (4 if _NP_VIEW[ds] == "int32" else 8)
        self._live -= n
        if not self.keep:
            try:
                os.remove(p)
            except OSError:
                pass

    def close(self):
        import shutil
        if not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)


# --------------------------------------------------------------------------- the sweep

class SweepStats(object):
    def __init__(self):
        self.peak_gib = 0.0
        self.bytes_streamed_delta = 0
        self.layer_unit_s = []      # (phase, L, seconds, n_units)
        self.wall_s = 0.0
        self.n_backward = 0
        self.tape_kind = "ram"
        self.tape_bytes = 0
        self.tape_peak_bytes = 0
        # Margin-objective diagnostics. None on the CE path, so a reader can never mistake "this
        # run had no margin term" for "the margin term measured zero".
        self.margin_loss = None     # the margin term's own value, before LAMBDA
        self.margin_delta = None    # mean lp(gold) - lp(distractor), length-normalised
        self.margin_ordered = None  # fraction of pairs the model already orders correctly
        self.n_pairs = 0


def layer_major_backward(model, streamer, groups, n_layers, loss_chunk=256,
                         boundary_device="cpu", log=None, adapter=None,
                         grad_scale_hook=None, tape=None, margin=None):
    """One layer-major forward+backward over `groups`; accumulates .grad on the LoRA leaves.

    `groups` : list of {"ids": LongTensor [B,T], "labels": LongTensor [B,T]} already on device.
               Every group is one microbatch. All of them share ONE optimizer step, so the loss is
               (sum of kept-token NLL over all groups) / (total kept tokens).
    Returns   (loss_value_float, SweepStats).

    `grad_scale_hook(layer_index)` is the sabotage seam the gate's positive control uses; it is
    None in every real run.

    `tape` is where boundary activations live between the forward and the reverse sweep. None keeps
    the original in-HOST-RAM behaviour (RamTape); pass a DiskTape to spill them to E:. The two are
    bit-identical by construction -- see the tape section above.
    """
    import torch
    import glm_capture_layer_major as LM

    if adapter is None:
        adapter = LM.GlmStackAdapter()
    if tape is None:
        tape = RamTape(boundary_device)
    dev = next(model.parameters()).device
    st = SweepStats()
    t0 = time.time()
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(0)
    b0 = int(getattr(streamer, "bytes_streamed", 0)) if streamer is not None else 0
    if streamer is not None:
        LM._require_hold_release(streamer)

    # The CE denominator counts GOLD targets only when rows are paired. Counting the interleaved
    # distractor targets too would divide the control's numerator by roughly twice its denominator
    # -- a different loss wearing the control's name.
    denom = count_kept_gold(groups) if margin else count_kept(groups)
    if denom <= 0:
        raise RuntimeError("every target position is masked: the loss has no support")

    # ------------------------------------------------------- phase 1: forward, layer-major
    with torch.no_grad():
        for gi, g in enumerate(groups):
            h, kw = adapter.preamble(model, g["ids"])
            g["kw"] = kw
            g["bnd"] = [tape.put((gi, 0), h)]
            g["cur"] = h
        for L in range(n_layers):
            tL = time.time()
            held = streamer.hold(L) if streamer is not None else False
            try:
                for gi, g in enumerate(groups):
                    g["cur"] = adapter.run_layer(model, L, g["cur"], g["kw"])
                    g["bnd"].append(tape.put((gi, L + 1), g["cur"]))
            finally:
                if held:
                    streamer.release(L)
            st.layer_unit_s.append(("fwd", L, time.time() - tL, len(groups)))
        for g in groups:
            g.pop("cur")
    if str(dev).startswith("cuda"):
        st.peak_gib = max(st.peak_gib, torch.cuda.max_memory_allocated(0) / (1 << 30))

    # ------------------------------------------- phase 2: epilogue -> dL/d(top boundary)
    #
    # The margin term lives ENTIRELY here. It is a function of per-sequence log-probs, which this
    # phase already has in `hs`, so phase 3 is untouched: the reverse sweep still just replays
    # layers against dL/d(top boundary), whatever produced that gradient. `margin=None` takes the
    # original code path, unchanged, so the control is the control by construction.
    loss_val = 0.0
    margin_val = 0.0
    n_pairs = count_pairs(groups) if margin else 0
    d_sum, ord_sum, n_g = 0.0, 0.0, 0
    if margin and n_pairs <= 0:
        raise RuntimeError("margin objective requested but no group holds a complete pair: "
                           "micro-batch must be even and rows must be interleaved "
                           "[gold, rejected, gold, rejected, ...]")
    for g in groups:
        x = tape.get(g["bnd"][n_layers], dev)
        hs = adapter.epilogue(model, x)
        if margin:
            loss, seq_nll, seq_kept = masked_ce_and_seq_logp(model, hs, g["labels"], denom,
                                                             chunk=loss_chunk, gold_only=True)
            mloss, dmean, dord = pair_margin_loss(seq_nll, seq_kept, margin["beta"], n_pairs)
            margin_val += float(mloss.detach())
            d_sum += float(dmean)
            ord_sum += float(dord)
            n_g += 1
            loss = loss + float(margin["lam"]) * mloss
        else:
            loss = masked_loss_from_hidden(model, hs, g["labels"], denom, chunk=loss_chunk)
        loss.backward()
        st.n_backward += 1
        if x.grad is None:
            raise RuntimeError(
                "the loss produced no gradient at the top boundary: the epilogue is off the path "
                "from the loss, so every gradient below would be silently wrong")
        g["gout"] = x.grad.detach().clone()
        tape.drop(g["bnd"][n_layers])
        g["bnd"][n_layers] = None
        loss_val += float(loss.detach())
        del x, hs, loss

    # ------------------------------------------------------- phase 3: reverse, layer-major
    for L in range(n_layers - 1, -1, -1):
        tL = time.time()
        held = streamer.hold(L) if streamer is not None else False
        try:
            for g in groups:
                x = tape.get(g["bnd"][L], dev)
                out = adapter.run_layer(model, L, x, g["kw"])
                gout = g["gout"]
                if grad_scale_hook is not None:
                    s = grad_scale_hook(L)
                    if s is not None:
                        gout = gout * float(s)
                out.backward(gradient=gout)
                st.n_backward += 1
                if x.grad is None:
                    raise RuntimeError(
                        "layer %d: the recompute produced no gradient at its input boundary; the "
                        "reverse chain is broken and every layer below is against a stale "
                        "gradient" % L)
                g["gout"] = x.grad.detach().clone()
                tape.drop(g["bnd"][L])
                g["bnd"][L] = None
                del x, out
        finally:
            if held:
                streamer.release(L)
        st.layer_unit_s.append(("bwd", L, time.time() - tL, len(groups)))
        if str(dev).startswith("cuda"):
            st.peak_gib = max(st.peak_gib, torch.cuda.max_memory_allocated(0) / (1 << 30))
        if log is not None and (L % 8 == 0 or L == 0):
            log("[lmbwd] reverse layer %d/%d  %.1fs" % (L, n_layers, time.time() - t0))

    want = len(groups) * (n_layers + 1)
    if st.n_backward != want:
        raise RuntimeError("%d backward(s) over %d group(s) x %d layer(s) + epilogue: expected %d"
                           % (st.n_backward, len(groups), n_layers, want))
    for g in groups:
        g.pop("gout", None)
        g.pop("bnd", None)
        g.pop("kw", None)
    st.bytes_streamed_delta = (int(getattr(streamer, "bytes_streamed", 0)) - b0) \
        if streamer is not None else 0
    st.tape_kind = tape.kind
    st.tape_bytes = int(tape.bytes_written)
    st.tape_peak_bytes = int(tape.peak_live_bytes)
    st.wall_s = time.time() - t0
    if margin and n_g:
        st.margin_loss = margin_val
        st.margin_delta = d_sum / n_g
        st.margin_ordered = ord_sum / n_g
        st.n_pairs = n_pairs
    return loss_val, st


# --------------------------------------------------------------------------- reference arm

def reference_backward(model, streamer, groups, n_layers, log=None):
    """THE INDEPENDENT REFERENCE. Stock transformers forward + stock autograd, nothing of mine.

    Every line of the computation belongs to code this file did not write:
      * `model.model(input_ids=..., use_cache=False)` -- Glm4MoeLiteModel.forward runs the whole
        stack in ONE graph: its own embedding, position_ids, create_causal_mask, rotary_emb and
        its own layer loop. No tape, no recompute, no loop inversion.
      * `model.lm_head` + `torch.nn.functional.cross_entropy(..., ignore_index=-100)` -- stock,
        unchunked, full [n_tok, vocab] logits in one shot (my arm is chunked + checkpointed).
      * `loss.backward()` -- stock autograd over the full stack (my arm is 1 backward per layer).

    It requires every layer resident, so the caller must hold them; this arm does no streaming.
    """
    import torch
    import torch.nn.functional as F
    dev = next(model.parameters()).device
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(0)
    st = SweepStats()
    t0 = time.time()
    denom = count_kept(groups)
    loss_val = 0.0
    for g in groups:
        hs = model.model(input_ids=g["ids"], use_cache=False).last_hidden_state
        logits = model.lm_head(hs[:, :-1, :]).float()
        tgt = g["labels"][:, 1:].reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt,
                               ignore_index=IGNORE_INDEX, reduction="sum") / float(denom)
        loss.backward()
        st.n_backward += 1
        loss_val += float(loss.detach())
        del hs, logits, loss
    if str(dev).startswith("cuda"):
        st.peak_gib = torch.cuda.max_memory_allocated(0) / (1 << 30)
    st.wall_s = time.time() - t0
    return loss_val, st


def reference_backward_chunked_tail(model, streamer, groups, n_layers, loss_chunk=256, log=None):
    """DECOMPOSITION ARM. Stock full-stack forward + stock autograd, but MY chunked masked tail.

    `reference_backward` differs from the layer-major arm in TWO independent ways at once: the
    loop inversion (the thing under test) and the CE tail (unchunked stock cross_entropy vs my
    chunked, checkpointed, masked tail). A single comparison cannot say which one produced a
    disagreement.

    This arm holds the tail fixed and varies only the loop, so:
        this vs reference_backward   -> the tail's contribution alone
        this vs layer_major_backward -> the LOOP INVERSION's contribution alone
    The second is the number that actually licenses or refutes the layer-major backward.
    """
    import torch
    dev = next(model.parameters()).device
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(0)
    st = SweepStats()
    t0 = time.time()
    denom = count_kept(groups)
    loss_val = 0.0
    for g in groups:
        hs = model.model(input_ids=g["ids"], use_cache=False).last_hidden_state
        loss = masked_loss_from_hidden(model, hs, g["labels"], denom, chunk=loss_chunk)
        loss.backward()
        st.n_backward += 1
        loss_val += float(loss.detach())
        del hs, loss
    if str(dev).startswith("cuda"):
        st.peak_gib = torch.cuda.max_memory_allocated(0) / (1 << 30)
    st.wall_s = time.time() - t0
    return loss_val, st


# --------------------------------------------------------------------------- comparison

# PRE-REGISTERED TOLERANCE. Set from bf16 first principles BEFORE any run, and NOT to be moved to
# make a run pass (memory: this repo has loosened a bar before and paid for it).
#
#   bf16 has an 8-bit significand, so unit roundoff u = 2^-9 = 1.953e-3.
#   The two arms compute the SAME VJPs from the SAME inputs and differ only in accumulation order
#   (chunked+checkpointed CE vs one unchunked CE; per-layer cotangent round-trip vs one graph), so
#   the expected disagreement is a small multiple of u on the tensors that pass through bf16.
#
# Primary bar is relative L2 -- it is the metric that a SYSTEMATIC error cannot hide from, whereas
# max-abs is dominated by single outliers. 5e-3 = 2.6 x u.
REL_L2_BAR = 5.0e-3
# Secondary bar guards a single catastrophic element that a good L2 could average away.
MAX_REL_BAR = 5.0e-2


def compare_grads(a, b, rel_l2_bar=REL_L2_BAR, max_rel_bar=MAX_REL_BAR):
    """Compare two {name: tensor-or-None} gradient maps. Returns a verdict dict.

    Metrics per tensor, computed in float64 on CPU:
      rel_l2  = ||a-b||_2 / ||b||_2
      max_rel = max|a-b| / max|b|
    `b` is the reference. A tensor that is None on either side, or all-zero on the reference side,
    is recorded as a PROBLEM, not silently skipped -- an all-zero reference gradient makes the
    comparison vacuous (this is exactly what B=0 LoRA init would have produced for every A).
    """
    import torch
    out = {"per_tensor": {}, "problems": [], "n_compared": 0}
    if sorted(a.keys()) != sorted(b.keys()):
        out["problems"].append("tensor name sets differ: %d vs %d" % (len(a), len(b)))
    for n in sorted(set(a.keys()) & set(b.keys())):
        ta, tb = a[n], b[n]
        if ta is None or tb is None:
            out["problems"].append("%s: grad is None (a=%s b=%s)"
                                   % (n, ta is None, tb is None))
            continue
        x = ta.to("cpu").double()
        y = tb.to("cpu").double()
        if x.shape != y.shape:
            out["problems"].append("%s: shape %s vs %s" % (n, tuple(x.shape), tuple(y.shape)))
            continue
        ny = float(torch.linalg.vector_norm(y).item())
        my = float(y.abs().max().item())
        if ny == 0.0 or my == 0.0:
            out["problems"].append("%s: reference gradient is ALL ZERO -- comparison is vacuous"
                                   % n)
            continue
        d = x - y
        rel_l2 = float(torch.linalg.vector_norm(d).item()) / ny
        max_rel = float(d.abs().max().item()) / my
        out["per_tensor"][n] = {"rel_l2": rel_l2, "max_rel": max_rel,
                                "ref_l2": ny, "n": int(y.numel())}
        out["n_compared"] += 1
    if out["n_compared"] == 0:
        out["problems"].append("ZERO tensors were compared: the comparison is vacuous")
    vals_l2 = [v["rel_l2"] for v in out["per_tensor"].values()]
    vals_mr = [v["max_rel"] for v in out["per_tensor"].values()]
    out["max_rel_l2"] = max(vals_l2) if vals_l2 else float("inf")
    out["med_rel_l2"] = _median(vals_l2)
    out["max_max_rel"] = max(vals_mr) if vals_mr else float("inf")
    out["med_max_rel"] = _median(vals_mr)
    out["rel_l2_bar"] = rel_l2_bar
    out["max_rel_bar"] = max_rel_bar
    ok = (not out["problems"]
          and out["max_rel_l2"] <= rel_l2_bar
          and out["max_max_rel"] <= max_rel_bar)
    out["verdict"] = "PASS" if ok else "FAIL"
    # Name the worst offenders so a FAIL is diagnosable without a rerun.
    worst = sorted(out["per_tensor"].items(), key=lambda kv: -kv[1]["rel_l2"])[:5]
    out["worst_tensors"] = [{"name": k, "rel_l2": v["rel_l2"], "max_rel": v["max_rel"]}
                            for k, v in worst]
    return out


def _median(v):
    if not v:
        return float("inf")
    s = sorted(v)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def bitwise_equal(a, b):
    """Strict `==` over every tensor. Used for the determinism check -- no tolerance allowed."""
    import torch
    names = sorted(set(a.keys()) | set(b.keys()))
    diffs = []
    for n in names:
        ta, tb = a.get(n), b.get(n)
        if ta is None or tb is None:
            diffs.append("%s: None" % n)
            continue
        if ta.shape != tb.shape or not bool(torch.equal(ta, tb)):
            nd = int((ta != tb).sum().item()) if ta.shape == tb.shape else -1
            diffs.append("%s: %d element(s) differ" % (n, nd))
    return (len(diffs) == 0), diffs


# --------------------------------------------------------------------------- data

def load_traces(path, n, tokenizer, seq_len, seed=0):
    """RSFT traces -> (ids, labels) with the PROMPT MASKED OUT.

    labels[i] = ids[i] on completion positions and IGNORE_INDEX on prompt/pad positions, so the
    shifted next-token loss only ever scores tokens the model was supposed to generate.
    """
    import json
    import random
    import torch
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if len(rows) >= max(int(n) * 4, int(n)):
                break
    random.Random(int(seed)).shuffle(rows)
    ids_all, lab_all = [], []
    pad = 154820
    for d in rows:
        p = tokenizer(d["prompt"], add_special_tokens=False)["input_ids"]
        c = tokenizer(d["completion"], add_special_tokens=False)["input_ids"]
        seq = (p + c)[:int(seq_len)]
        if len(seq) < 8:
            continue
        lab = ([IGNORE_INDEX] * min(len(p), len(seq))
               + seq[min(len(p), len(seq)):])
        npad = int(seq_len) - len(seq)
        seq = seq + [pad] * npad
        lab = lab + [IGNORE_INDEX] * npad
        if all(x == IGNORE_INDEX for x in lab[1:]):
            continue
        ids_all.append(seq)
        lab_all.append(lab)
        if len(ids_all) >= int(n):
            break
    return torch.tensor(ids_all, dtype=torch.long), torch.tensor(lab_all, dtype=torch.long)


def make_groups(ids, labels, batch, n_groups, device, start=0):
    gs = []
    for i in range(int(n_groups)):
        lo = start + i * int(batch)
        hi = lo + int(batch)
        if hi > ids.shape[0]:
            break
        gs.append({"ids": ids[lo:hi].to(device), "labels": labels[lo:hi].to(device)})
    return gs
