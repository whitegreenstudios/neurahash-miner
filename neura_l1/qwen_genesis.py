"""
neura_l1.qwen_genesis — a FAITHFUL Qwen->protocol import path, on top of the Qwen-shaped backbone.

WHERE THIS SITS
``hf_import`` is an ANALYZER: it inspects an HF Qwen state_dict and reports the four architectural
gaps (RoPE / RMSNorm / SwiGLU / GQA) that block a faithful import onto the TOY MoETransformer. Those
gaps existed because the protocol had no Qwen-shaped backbone. ``neurahash_torch.qwen_backbone`` now
provides exactly that shape. This module is the bridge:

  * ``qwen_config_from_hf``  — derive a QwenBackbone config from an inferred HF config + chosen
                               head_dim (the analyzer reports the GQA ratio; head_dim picks the rest).
  * ``map_qwen_state``       — map an HF Qwen state_dict onto the backbone's state_dict BY NAME, with
                               no reshaping of linear weights (the keys already line up). Reports any
                               key that is missing or left over, so nothing is silently dropped.
  * ``import_qwen_base``     — load HF weights into a fresh QwenBackbone and validate it round-trips.

WHAT REMAINS NEEDS-REAL-WORLD (this module CANNOT manufacture it):
  1. The actual multi-GB Qwen checkpoint (download + license acceptance).
  2. The COMPUTE to continue-pretrain / upcycle-to-MoE on the protocol's data.
  3. Cross-VENDOR bit-exact recompute for trustless verification across heterogeneous miners — the
     float backbone is recompute-stable on the SAME hardware (enough for same-machine replay), but
     the integer ``repops`` path is what makes it vendor-agnostic; wiring SwiGLU/GQA through repops
     is a separate deliverable.

Once a real Qwen checkpoint is downloaded, ``import_qwen_base`` produces a byte-defined state_dict,
``base_import.write_base_checkpoint`` content-addresses it, and ``base_import.genesis_from_base``
makes it the reproducible on-chain genesis. The CODE path is complete and tested here against a
constructed Qwen-shaped donor; only the weights+compute are external.
"""

from neurahash_torch.qwen_backbone import QwenBackbone, hf_key_layout


class QwenImportError(Exception):
    """Raised when an HF Qwen state_dict cannot be faithfully mapped onto the backbone."""


def qwen_config_from_hf(hf_cfg, head_dim, block_size=128, theta=10000.0, vocab_size=None):
    """Build a QwenBackbone(**config) kwargs dict from an inferred HF config (see
    ``hf_import.infer_hf_config``) plus the chosen rotary ``head_dim``.

    The analyzer reports q_rows = n_head*head_dim and the GQA group factor = n_head/n_kv_head;
    given head_dim we recover n_head and n_kv_head exactly. Raises if the inferred rows are not a
    clean multiple of head_dim (a sign head_dim was guessed wrong)."""
    vocab = vocab_size if vocab_size is not None else hf_cfg.get("vocab")
    d_model = hf_cfg.get("d_model")
    n_layers = hf_cfg.get("n_layers")
    inter = hf_cfg.get("intermediate")
    if None in (vocab, d_model, n_layers, inter):
        raise QwenImportError(f"incomplete HF config: {hf_cfg}")
    if d_model % head_dim != 0:
        raise QwenImportError(f"d_model {d_model} not a multiple of head_dim {head_dim}")
    n_head = d_model // head_dim
    gqa = hf_cfg.get("gqa_group_factor") or 1
    if n_head % gqa != 0:
        raise QwenImportError(f"n_head {n_head} not divisible by GQA factor {gqa}")
    n_kv_head = n_head // gqa
    return dict(vocab_size=vocab, d_model=d_model, n_head=n_head, n_kv_head=n_kv_head,
                n_layers=n_layers, d_ff=inter, block_size=block_size, theta=theta,
                attn_bias=hf_cfg.get("attn_bias", False))


def map_qwen_state(hf_sd, backbone):
    """Map an HF Qwen state_dict onto ``backbone``'s state_dict by NAME. Returns
    (mapped, missing, leftover):
      * mapped   — {backbone_key: tensor} for every backbone parameter we found a source for
      * missing  — backbone keys with no matching HF key (would block a faithful load)
      * leftover — HF keys not consumed (e.g. rotary_emb.inv_freq buffers, biases we don't carry)

    Linear weights map with NO reshape because the backbone deliberately uses HF's q/k/v/o + gate/
    up/down layout. Shape mismatches are reported as `missing` (the key matched but the shape did
    not), never silently coerced."""
    want = backbone.state_dict()
    mapped, missing = {}, []
    used = set()
    for k, target in want.items():
        src = hf_sd.get(k)
        if src is None:
            missing.append(k)
            continue
        if tuple(src.shape) != tuple(target.shape):
            missing.append(k)               # name matched, shape didn't -> not a faithful map
            continue
        mapped[k] = src
        used.add(k)
    leftover = [k for k in hf_sd if k not in used]
    return mapped, missing, leftover


def import_qwen_base(hf_sd, backbone, strict=True):
    """Load an HF Qwen state_dict into ``backbone`` faithfully. With strict=True (default) any
    missing backbone key raises QwenImportError — a partial load would silently train a different
    model than the verifier expects. Returns (leftover_hf_keys). Buffers (RoPE cos/sin, masks) are
    rebuilt by the backbone from config, so HF rotary_emb.inv_freq appearing in `leftover` is
    expected and harmless."""
    mapped, missing, leftover = map_qwen_state(hf_sd, backbone)
    if strict and missing:
        raise QwenImportError(
            f"faithful import blocked: {len(missing)} backbone key(s) unmatched; "
            f"first: {missing[:5]}")
    backbone.load_state_dict(mapped, strict=False)
    return leftover


def expected_backbone_keys(n_layers):
    """The HF-named key set this backbone consumes (mirrors qwen_backbone.hf_key_layout) — handy for
    callers building or validating a donor before constructing the model."""
    return hf_key_layout(n_layers)
