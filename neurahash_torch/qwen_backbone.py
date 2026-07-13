"""
neurahash_torch.qwen_backbone — a Qwen-class transformer backbone the protocol can train AND verify.

WHY THIS EXISTS
The live protocol model (``model_torch.MoETransformer``) is a TOY in shape: learned position
embeddings, LayerNorm (weight+bias), a 2-matrix GELU FFN, a combined-qkv projection and full MHA.
``neura_l1.hf_import`` already PROVES (and unit-tests) that those four choices are exactly the gaps
that block a faithful import of an open Qwen-class base. This module closes the *code* side of that
gap: a real Qwen-shaped block —

    * RoPE          rotary position embedding (no learned position table)
    * RMSNorm       weight-only normalisation (no bias)
    * SwiGLU        3-matrix gate/up/down MLP
    * GQA           grouped-query attention (n_kv_heads <= n_heads), split q/k/v/o projections

— built so the protocol can train it and (critically) so a verifier can RECOMPUTE it bit-for-bit on
the same machine, which is what the replay-verified chain requires.

WHAT THIS IS AND IS NOT
  * IS: a real, trainable, autograd backbone with the Qwen block design; a state_dict whose KEY SET
    and SHAPES match a Hugging Face Qwen checkpoint (``model.embed_tokens``,
    ``model.layers.{i}.self_attn.{q,k,v,o}_proj``, ``...mlp.{gate,up,down}_proj``,
    ``...input_layernorm`` / ``post_attention_layernorm``, ``model.norm``, ``lm_head``), so an HF
    state_dict maps onto it by NAME with no reshaping of the linear weights.
  * IS NOT: the multi-GB Qwen WEIGHTS, nor the COMPUTE to continue-pretrain them. Standing this
    backbone on a real Qwen-7B/14B base needs the actual checkpoint downloaded + hash-committed
    (``neura_l1.base_import``) and many GPU-hours of continued pretraining — neither of which is
    code. This file makes the protocol *able* to hold and train that shape; it does not conjure the
    base.

DETERMINISM / VERIFIABILITY NOTE
RoPE uses a fixed (non-learned) inv_freq buffer derived only from (head_dim, theta), so two nodes
build identical rotary tables. RMSNorm/SwiGLU/GQA are plain ops. The whole block is therefore a
pure function of (weights, input) — a verifier re-running it on the same hardware reproduces the
activations, which is the property ``shard_verify`` / ``trunk_verify`` rely on. (Cross-VENDOR
bit-exactness is a separate, harder problem handled by the integer ``repops`` path, not here.)
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

# (8GB-card VRAM fit) gradient checkpointing for the qwen decoder layers -- mirrors model_torch.py's
# _GRAD_CKPT knob (NEURAHASH_GRAD_CHECKPOINT), which only wraps the toy MoETransformer's blocks and is
# a no-op on this backbone today. DEFAULT OFF: unset/"0"/"false" => today's exact behavior, byte-for-
# byte. When on, each QwenBlock's forward is wrapped in checkpoint(use_reentrant=False): the forward
# runs fully (autograd-tracked) but its internal activations are NOT retained; backward re-runs the
# same deterministic forward to regenerate them -> identical gradients, lower peak VRAM.
_GRAD_CKPT = os.environ.get("NEURAHASH_GRAD_CHECKPOINT", "") not in ("", "0", "false", "False")
# NEURAHASH_MEM_DEBUG=1 (env, default OFF, shared with tools/diloco_contributor.py's memory probes):
# print ONE confirmation line to stderr the first time the checkpoint branch actually executes, so a
# VRAM-fit diagnosis can tell "grad-checkpoint requested" apart from "grad-checkpoint wired but never
# runs" without adding any per-layer/per-step overhead when unset or once confirmed.
_MEM_DEBUG = os.environ.get("NEURAHASH_MEM_DEBUG", "") not in ("", "0", "false", "False")
_grad_ckpt_confirmed = False


# ---------------------------------------------------------------------------
# RoPE — rotary position embedding (fixed, non-learned)
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len, head_dim, theta=10000.0, device="cpu", dtype=torch.float32):
    """Precompute (cos, sin) of shape (seq_len, head_dim) for rotary embedding. Depends ONLY on
    (seq_len, head_dim, theta) so every node builds the identical table — there is nothing learned
    to diverge. head_dim must be even (rotary pairs)."""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                                / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary embedding to q,k of shape (B, n_head, T, head_dim). cos/sin are (T, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0)                    # (1,1,T,hd)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


# ---------------------------------------------------------------------------
# RMSNorm — weight-only normalisation (Qwen/LLaMA style)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # compute in fp32 for stability then cast back (matches HF RMSNorm)
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dtype))


# ---------------------------------------------------------------------------
# GQA attention — grouped-query, split q/k/v/o, RoPE
# ---------------------------------------------------------------------------
class GQAttention(nn.Module):
    """Grouped-query causal self-attention with RoPE. Split projections named to match HF Qwen:
    q_proj/k_proj/v_proj/o_proj. n_kv_heads <= n_heads; when n_kv_heads == n_heads this is MHA."""

    def __init__(self, d_model, n_head, n_kv_head, block_size, theta=10000.0, bias=False,
                 attn_bias=False, rms_eps=1e-6, qk_norm=False, head_dim=None):
        super().__init__()
        if head_dim is None and d_model % n_head != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_head {n_head}")
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head {n_head} not divisible by n_kv_head {n_kv_head}")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        # Qwen3 sets head_dim EXPLICITLY (e.g. 0.6B: d_model=1024, n_head=16, head_dim=128 != 1024/16=64),
        # so q/k/v/o projections are n_head*head_dim wide, NOT d_model. Default to d_model//n_head (1.7B fits).
        self.head_dim = int(head_dim) if head_dim else d_model // n_head
        self.n_rep = n_head // n_kv_head
        # Qwen3 adds RMSNorm on q and k — per head, over head_dim — applied to the reshaped q/k BEFORE
        # RoPE (Qwen2 does not have this). The HF keys are self_attn.{q,k}_norm.weight of size head_dim.
        # Without it, importing a Qwen3 base silently drops those tensors and the forward diverges.
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_eps)
        # Qwen2/2.5 carry biases on the q/k/v projections (attn_bias=True) but NOT on o_proj;
        # Qwen3/LLaMA carry none (attn_bias=False). Without these, importing a real Qwen2 base
        # silently drops q/k/v_proj.bias and the forward diverges from the reference.
        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(n_head * self.head_dim, d_model, bias=bias)
        cos, sin = build_rope_cache(block_size, self.head_dim, theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.qk_norm:                         # Qwen3 RMSNorm on q,k (over head_dim) BEFORE RoPE
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, k, self.rope_cos[:T], self.rope_sin[:T])

        # expand kv heads to match query heads (GQA -> per-head)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# SwiGLU MLP — 3-matrix gate/up/down (the dense donor for the MoE upcycle)
# ---------------------------------------------------------------------------
class SwiGLUMLP(nn.Module):
    """SiLU-gated MLP: down(silu(gate(x)) * up(x)). Named gate_proj/up_proj/down_proj to match HF."""

    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Qwen-shaped block + transformer
# ---------------------------------------------------------------------------
class QwenBlock(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head, d_ff, block_size, theta=10000.0,
                 rms_eps=1e-6, bias=False, attn_bias=False, qk_norm=False, head_dim=None):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, eps=rms_eps)
        self.self_attn = GQAttention(d_model, n_head, n_kv_head, block_size, theta, bias, attn_bias,
                                     rms_eps=rms_eps, qk_norm=qk_norm, head_dim=head_dim)
        self.post_attention_layernorm = RMSNorm(d_model, eps=rms_eps)
        self.mlp = SwiGLUMLP(d_model, d_ff, bias)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class QwenModel(nn.Module):
    """The decoder stack (embeddings + layers + final norm). Named ``model`` inside QwenBackbone so
    state_dict keys are ``model.embed_tokens.*`` / ``model.layers.{i}.*`` / ``model.norm.weight`` —
    exactly the HF Qwen layout, letting a donor map by name with no key surgery."""

    def __init__(self, vocab_size, d_model, n_head, n_kv_head, n_layers, d_ff, block_size,
                 theta, rms_eps, bias, attn_bias=False, qk_norm=False, head_dim=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            QwenBlock(d_model, n_head, n_kv_head, d_ff, block_size, theta, rms_eps, bias, attn_bias,
                      qk_norm=qk_norm, head_dim=head_dim)
            for _ in range(n_layers)])
        self.norm = RMSNorm(d_model, eps=rms_eps)

    def forward(self, idx):
        x = self.embed_tokens(idx)
        for layer in self.layers:
            # only checkpoint while gradients are tracked (the training forward); a no_grad forward
            # (eval / generate) is unaffected either way -- bit-identical to the plain path.
            if _GRAD_CKPT and torch.is_grad_enabled():
                global _grad_ckpt_confirmed
                if _MEM_DEBUG and not _grad_ckpt_confirmed:
                    print("[mem_debug] qwen grad-checkpoint path IS executing "
                          "(torch.utils.checkpoint active)", file=sys.stderr)
                    _grad_ckpt_confirmed = True
                x = _ckpt.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x)


class QwenBackbone(nn.Module):
    """A Qwen-class decoder-only transformer. Architecture matches an HF Qwen checkpoint so its
    state_dict keys line up by name (see module docstring): the decoder stack lives under a ``model``
    submodule (``model.embed_tokens`` / ``model.layers.{i}`` / ``model.norm``) with a top-level
    ``lm_head`` — identical to HF Qwen. Dense MLP (the base form); the protocol upcycles each SwiGLU
    into a MoE for sharded training via ``base_import.upcycle_dense_to_experts`` once a MoE-SwiGLU
    expert layout exists. This class is the trainable, recompute-verifiable BACKBONE — not the
    weights."""

    def __init__(self, vocab_size, d_model=64, n_head=8, n_kv_head=2, n_layers=3, d_ff=256,
                 block_size=128, theta=10000.0, rms_eps=1e-6, bias=False, attn_bias=False,
                 tie_embeddings=False, qk_norm=False, head_dim=None):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_head=n_head,
                           n_kv_head=n_kv_head, n_layers=n_layers, d_ff=d_ff,
                           block_size=block_size, theta=theta, rms_eps=rms_eps, bias=bias,
                           attn_bias=attn_bias, tie_embeddings=tie_embeddings, qk_norm=qk_norm,
                           head_dim=head_dim)
        self.block_size = block_size
        self.model = QwenModel(vocab_size, d_model, n_head, n_kv_head, n_layers, d_ff,
                               block_size, theta, rms_eps, bias, attn_bias, qk_norm=qk_norm,
                               head_dim=head_dim)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, idx, targets=None):
        x = self.model(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8):
        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------------------------------------------------------------------
# HF key layout this backbone exposes — used by the faithful mapper + tests so the
# expected key set can never drift from the live module.
# ---------------------------------------------------------------------------
def hf_key_layout(n_layers, attn_bias=False, qk_norm=False):
    """The Hugging-Face-style key names a Qwen state_dict must provide to map onto this backbone,
    one entry per (logical) tensor. Linear weights map with NO reshape; this is the spec the
    faithful importer fills. With ``attn_bias=True`` the q/k/v projection biases (which a real
    Qwen2/2.5 base carries) are included; with ``qk_norm=True`` the per-head q/k RMSNorm weights
    (which a Qwen3 base carries) are included, so the spec matches that base exactly."""
    keys = ["model.embed_tokens.weight"]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        keys += [
            p + "input_layernorm.weight",
            p + "self_attn.q_proj.weight",
            p + "self_attn.k_proj.weight",
            p + "self_attn.v_proj.weight",
            p + "self_attn.o_proj.weight",
            p + "post_attention_layernorm.weight",
            p + "mlp.gate_proj.weight",
            p + "mlp.up_proj.weight",
            p + "mlp.down_proj.weight",
        ]
        if attn_bias:                       # Qwen2/2.5 carry q/k/v projection biases
            keys += [
                p + "self_attn.q_proj.bias",
                p + "self_attn.k_proj.bias",
                p + "self_attn.v_proj.bias",
            ]
        if qk_norm:                         # Qwen3 carries per-head q/k RMSNorm weights
            keys += [
                p + "self_attn.q_norm.weight",
                p + "self_attn.k_norm.weight",
            ]
    keys += ["model.norm.weight", "lm_head.weight"]
    return keys
