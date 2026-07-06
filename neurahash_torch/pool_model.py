"""
pool_model.py — the model factory the sharded pool uses to construct the trained network.

Lets the pool build EITHER the toy MoETransformer (default) OR a real DENSE base backbone — a QwenBackbone
loaded with imported pretrained weights — when NEURAHASH_BASE is set. Lives in its own module so both
sharded_pool_node.py AND coord_checkpoint.py import it with no import cycle.

Dense-base principle: a QwenBackbone has ZERO ".moe.experts." keys, so shard_verify.trunk_keys returns
EVERY param (the whole model is the trunk) and the existing trunk-delta + cosine verify path works
unchanged (proven: tests/test_qwen_pool_verify.py + the faithful import in test_qwen3_qknorm.py). The MoE
expert/grow/snapshot machinery becomes a natural no-op at n_experts=0; sharded_pool_node guards only the 3
sites that would otherwise error (add_expert, empty-AdamW, the snapshot expert-accept loop).

Weights enter the network exactly ONCE — at the coordinator's canonical model (load_base=True). The worker
and the recompute-verifier build with load_base=False: they receive the canonical trunk (= whole dense
model) from the coordinator, so importing the base into them would just be overwritten.
"""
import os

import torch

from neurahash_torch.model_torch import MoETransformer

# the QwenBackbone constructor kwargs we thread out of an ARCH dict (everything but vocab_size + kind)
_QWEN_KEYS = ("d_model", "n_head", "n_kv_head", "n_layers", "d_ff", "block_size", "theta",
              "rms_eps", "qk_norm", "tie_embeddings", "attn_bias", "bias", "head_dim")


def qwen_arch(base=None, block_size=512):
    """The dense ARCH dict (kind='qwen', n_experts=0) for a base. Dims are read from the base's HF config
    so the backbone shape MATCHES the weights (Qwen3-0.6B is d_model=1024, 1.7B is 2048, ...). For a small
    smoke that skips the real base, set NEURAHASH_QWEN_DMODEL/HEADS/KV/LAYERS/DFF to override the dims."""
    base = base or os.environ.get("NEURAHASH_BASE", "qwen3-1.7b")
    if os.environ.get("NEURAHASH_QWEN_DMODEL"):                     # smoke: explicit tiny dims, no config load
        g = lambda k, d: int(os.environ.get(k, d))
        d_model, n_head, n_kv = g("NEURAHASH_QWEN_DMODEL", 2048), g("NEURAHASH_QWEN_HEADS", 16), g("NEURAHASH_QWEN_KV", 8)
        n_layers, d_ff, theta, rms_eps, tie = g("NEURAHASH_QWEN_LAYERS", 28), g("NEURAHASH_QWEN_DFF", 6144), 1e6, 1e-6, True
        head_dim = g("NEURAHASH_QWEN_HEADDIM", d_model // n_head)
    else:
        from transformers import AutoConfig
        from model_registry import resolve_model
        cfg = AutoConfig.from_pretrained(resolve_model(base))
        d_model, n_head, n_kv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        n_layers, d_ff = cfg.num_hidden_layers, cfg.intermediate_size
        theta = float(getattr(cfg, "rope_theta", None) or 1e6)
        rms_eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
        tie = bool(getattr(cfg, "tie_word_embeddings", True))
        # Qwen3 sets head_dim explicitly and it is NOT always d_model//n_head (0.6B: 128 != 64).
        head_dim = int(getattr(cfg, "head_dim", None) or (d_model // n_head))
    return dict(kind="qwen", d_model=d_model, n_head=n_head, n_kv_head=n_kv, n_layers=n_layers,
                d_ff=d_ff, block_size=int(block_size), theta=theta, rms_eps=rms_eps, qk_norm=True,
                tie_embeddings=tie, attn_bias=False, bias=False, n_experts=0, head_dim=head_dim)


def _default_dtype():
    """NEURAHASH_DTYPE selects the dense-base compute dtype. Default fp32 (correct + CPU-friendly + matches
    the toy path); set bf16 for a big base that won't otherwise fit VRAM (the wire path casts to fp32 for
    the cosine gate + on-chain hash, so bf16 stays direction-stable same-GPU)."""
    d = (os.environ.get("NEURAHASH_DTYPE", "") or "").strip().lower()
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "half": torch.float16}.get(d, torch.float32)


_BASE_SD = None   # the imported HF base state_dict, cached per process (cloned: no safetensors-mmap drift)


def _load_base_into(m):
    """Import the HF base weights (NEURAHASH_BASE) into a fresh backbone, faithfully (strict)."""
    global _BASE_SD
    if _BASE_SD is None:
        from transformers import AutoModelForCausalLM
        from model_registry import resolve_model
        mid = resolve_model(os.environ.get("NEURAHASH_BASE", "qwen3-1.7b"))
        hf = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16)
        _BASE_SD = {k: v.detach().clone() for k, v in hf.state_dict().items()}   # clone => no mmap reduce-order drift
        del hf
    from neura_l1.qwen_genesis import import_qwen_base
    import_qwen_base(_BASE_SD, m, strict=True)


def build_pool_model(arch, vocab, hosted, device, dtype=None, load_base=True):
    """Construct the model for a pool role. Dense Qwen base when arch['kind']=='qwen', else the toy MoE.
    The toy path is byte-for-byte the previous construction (kind is simply absent)."""
    if arch.get("kind") == "qwen":
        from neurahash_torch.qwen_backbone import QwenBackbone
        cfg = {k: arch[k] for k in _QWEN_KEYS if k in arch}
        m = QwenBackbone(vocab_size=int(vocab), **cfg).to(device=device, dtype=(dtype or _default_dtype()))
        if load_base and not os.environ.get("NEURAHASH_SKIP_BASE_IMPORT"):
            _load_base_into(m)
        return m
    a = {k: v for k, v in arch.items() if k != "kind"}
    return MoETransformer(vocab_size=vocab, moe="sparse", hosted_experts=list(hosted), **a).to(device)
