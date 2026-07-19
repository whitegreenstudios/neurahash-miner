#!/usr/bin/env python3
"""tools/make_base_from_hf.py -- materialize a fresh round-0 DiLoCo base checkpoint by fetching the
base weights from HuggingFace's CDN (OUTBOUND, NAT-safe), NOT from any local multi-GB file.

WHY. The multi-miner orchestrator today scp-pushes a ~4.5GB base checkpoint from the home machine
to every rented pod; a residential uplink cannot move GB-scale data reliably (observed: connection
reset, then a 150MB-chunk scp TimeoutExpired at ~250KB/s to an EU pod). The fix is to let each pod
build its OWN base from HuggingFace (fast CDN, all-outbound, NAT-safe). This script is that builder.

FORMAT. The output is written in EXACTLY the coord_checkpoint.py on-disk format (the same format as
_qwen06/coord_checkpoint.pt), so tools/diloco_contributor.py can:
    contribute   <base>            -> train the trunk, emit a compressed delta
    merge-delta  <base> <delta>    -> fold a delta back into the base (held-out gated)

For a DENSE base (arch kind='qwen', n_experts=0) shard_verify.trunk_keys returns EVERY parameter, so
the model IS the trunk; load_checkpoint asserts set(global_trunk) == set(trunk_keys(model)). Hence
global_trunk must be the FULL model state (an empty dict would fail that assert) -- which is why a
dense checkpoint stores the weights twice (~2x file size, ~4.5GB in fp32 for Qwen3-0.6B).

USAGE:
    python tools/make_base_from_hf.py [out_path] [--device cpu|cuda] [--base qwen3-0.6b]
Reads NEURAHASH_BASE (default qwen3-0.6b) and NEURAHASH_BLOCK (default 512, mirrors the coordinator).
Output is CPU tensors regardless of --device; use --device cpu when the GPU is contended.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root on path

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "neurahash_miner", "basegen", "qwen06_r0.pt")


def main():
    ap = argparse.ArgumentParser(description="Build a round-0 DiLoCo base checkpoint from HF weights.")
    ap.add_argument("out_path", nargs="?", default=DEFAULT_OUT,
                    help=f"output checkpoint path (default {DEFAULT_OUT})")
    ap.add_argument("--device", default="cpu", help="build device: cpu (default) or cuda")
    ap.add_argument("--base", default=None,
                    help="base model key (default: NEURAHASH_BASE env, else qwen3-0.6b)")
    args = ap.parse_args()

    # base resolution. _load_base_into (pool_model.py) reads the HF model id from the NEURAHASH_BASE
    # ENV VAR, not from a parameter, so we pin the env to whatever base we build -> the arch we build
    # and the weights we fetch are guaranteed to be the same model.
    base = args.base or os.environ.get("NEURAHASH_BASE") or "qwen3-0.6b"
    os.environ["NEURAHASH_BASE"] = base
    block_size = int(os.environ.get("NEURAHASH_BLOCK", "512"))

    import torch
    from transformers import AutoConfig
    from model_registry import resolve_model
    from neurahash_torch.pool_model import qwen_arch, build_pool_model
    from neurahash_torch.shard_verify import trunk_keys
    from neurahash.base_checkpoint import save_checkpoint

    mid = resolve_model(base)
    print(f"[make-base] base={base} -> {mid} device={args.device} block_size={block_size}")
    print(f"[make-base] out={args.out_path}")

    # 1) arch0 from the HF config (kind='qwen', n_experts=0). Mirror the coordinator exactly:
    #    sharded_pool_node builds ARCH = qwen_arch(BASE_NAME, block_size=NEURAHASH_BLOCK or 512).
    arch0 = qwen_arch(base, block_size=block_size)
    print(f"[make-base] arch0 kind={arch0['kind']} d_model={arch0['d_model']} n_layers={arch0['n_layers']} "
          f"n_head={arch0['n_head']} n_kv_head={arch0['n_kv_head']} head_dim={arch0['head_dim']} "
          f"n_experts={arch0['n_experts']}")

    # 2) vocab from the HF config -- MUST equal the base's embedding rows so the strict weight import
    #    (import_qwen_base, strict=True) shape-matches the embedding/lm_head.
    cfg = AutoConfig.from_pretrained(mid)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    if not vocab:
        from transformers import AutoTokenizer
        vocab = int(AutoTokenizer.from_pretrained(mid).vocab_size)
    print(f"[make-base] vocab={vocab}")

    # 3) build the model WITH the HF weights imported. load_base=True routes through _load_base_into,
    #    which does AutoModelForCausalLM.from_pretrained(NEURAHASH_BASE) -> the OUTBOUND HF fetch (this
    #    is the whole point: no local 4.5GB file, no scp push). fp32 to match the existing _qwen06 base
    #    format and to keep the delta-merge (contrib_trunk - base_trunk) fp32-clean.
    print("[make-base] fetching HF weights + building backbone (downloads on first run, ~1.2GB)...")
    model = build_pool_model(arch0, vocab, [], args.device, dtype=torch.float32, load_base=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[make-base] model built: {n_params / 1e6:.1f}M params")

    # 4) global_trunk = the full trunk state. For a dense base trunk_keys(sd) == every key, and
    #    load_checkpoint raises unless set(global_trunk) == set(trunk_keys(model)). Match the
    #    contributor's own construction (diloco_contributor new_trunk / test _trunk): clone the trunk.
    sd = model.state_dict()
    global_trunk = {k: sd[k].detach().clone() for k in trunk_keys(sd)}
    print(f"[make-base] trunk keys={len(global_trunk)} of {len(sd)} model keys (dense: trunk == whole model)")

    # 5) empty stats. The model-only base carries NO signed economic ledger (base_checkpoint drops the
    #    ledger field entirely): the miner publishes a delta, never an authoritative coordinator checkpoint.
    stats = {}

    # 6) round-0 checkpoint, written atomically by coord_checkpoint.save_checkpoint (tensors -> CPU).
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_checkpoint(args.out_path, model=model, global_trunk=global_trunk, E=0, round=0,
                    stats=stats, arch0=arch0, vocab=vocab)
    size = os.path.getsize(args.out_path)
    print(f"[make-base] wrote {args.out_path} ({size / 1e9:.2f} GB, {size} bytes)")
    print(f"[make-base] DONE round=0 E=0 vocab={vocab} kind={arch0['kind']}")


if __name__ == "__main__":
    sys.exit(main())
