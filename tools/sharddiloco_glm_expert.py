"""Phase-4 (Route A): host a REAL torch GLM (glm4_moe_lite) MoE expert inside the E:/aiCrypto
shardDiLoCo lane -- gated by the SECRET rotated probe and carried over the SIGNED, content-addressed
ShardDeltaLane.

WHY (docs/research/SHARDDILOCO_PHASE4_GLM_READINESS.md, Route A). The GLM per-expert train +
canonical-delta + held-out gate already exist in the miner repo (D:/glm_loader/repo:
fleet/glm_worker.py, tools/expert_shard_train.py). What that GLM coordinator LACKS are the two
rigor features the E:/aiCrypto shardDiLoCo lane already has: (1) the SECRET ROTATED held-out probe
(neurahash.diloco_merge.SecretRotatedProbe) so gating uses a probe a miner cannot see/game, and
(2) the SIGNED content-addressed delta lane (tools.diloco_contributor.ShardDeltaLane: sha256 CID +
fp16 wire + HMAC sig) so each expert delta is signed+CID-verified before it can touch a weight.
This module brings the GLM per-expert unit into the lane's contract WITHOUT modifying either the
numpy-MoELM lane or those two rigor primitives -- it reuses them verbatim.

The lane's merge machinery (shard_merge_round / apply_delta_gated / SecretRotatedProbe / FlopMeter /
ShardDeltaLane) is model-AGNOSTIC: `experts[e]` is a dict of arrays, a delta is a dict of arrays,
and `eval_expert(e, cand, pX, pY)` is a caller-supplied callback. So the WHOLE gate/lane pipeline
runs on numpy exactly as today, and ONLY the eval_expert callback bridges numpy<->torch to run a
real GLM forward-CE. A GLM (layer, expert) unit maps onto the lane's per-expert slot as the
canonical fused-weight triple {gate:[I,H], up:[I,H], down:[H,I]} (the same canonical delta
tools/expert_shard_train.materialize_canonical_from_saved uploads).

DEFAULT-OFF: this is a NEW opt-in module. Importing or running it does not change the numpy-MoELM
lane's behavior (the shard_diloco_on() gate and its tests are untouched; a test asserts byte-identity).

CPU-ONLY / no fleet / no spend. The full GLM-4.7-Flash trunk is 5.67 GB (62 GB sharded); loading it
on CPU is out of scope here, so run_smoke() TRUNCATES to the smallest viable REAL glm4_moe_lite
config (few layers, few experts, small hidden/vocab) instantiated from the real transformers config
class -- a real Glm4MoeLiteForCausalLM with the real sigmoid-top-k router and the real FUSED expert
MLP, NOT the numpy MoELM. The truncation is stated in the smoke output.

VENDORED (minimal, attributed): `LoRAExperts` and the canonical-delta materialization are faithful
copies of D:/glm_loader/repo/tools/expert_shard_train.py (LoRAExperts:47, materialize_canonical_from_saved:461,
apply_canonical_to_fused:493). They are copied rather than imported because expert_shard_train.py
imports the MINER repo's `neurahash` package, which would clash with E:/aiCrypto's `neurahash`
(this module needs E:/aiCrypto's neurahash.diloco_merge). Copying keeps this module self-contained.
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- the shardDiLoCo lane's rigor primitives, reused VERBATIM (no edits) ---
from neurahash.diloco_merge import FlopMeter  # noqa: E402

# `SecretRotatedProbe` / `shard_merge_round` are COORDINATOR-side consensus: the public miner's
# `neurahash/diloco_merge.py` is a 117-line contributor subset that deliberately omits them (it does
# ship FlopMeter). A hard module-level import therefore made this whole file unimportable from a
# public checkout -- measured on a fresh clone, and it would have shipped a broken public repo.
# Bound optionally rather than lazily so that WHERE THEY EXIST they are still the SAME objects as
# the numpy lane's: tests/test_sharddiloco_glm_wiring.py asserts
# `G.shard_merge_round is dm.shard_merge_round` precisely to stop a divergent second copy of the
# merge/gate logic from appearing. Public checkouts get None and only the contributor path works.
try:                                                              # noqa: SIM105
    from neurahash.diloco_merge import SecretRotatedProbe, shard_merge_round  # noqa: E402
except ImportError:                                               # public (contributor-only) checkout
    SecretRotatedProbe = None
    shard_merge_round = None
from diloco_contributor import ShardDeltaLane  # noqa: E402


# =============================================================== vendored GLM per-expert train path
# Faithful copies from D:/glm_loader/repo/tools/expert_shard_train.py (see module docstring). Only
# reformatted imports; the math/semantics are unchanged so the canonical delta is identical to what
# the built GLM worker uploads.
def _lora_experts_cls():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRAExperts(nn.Module):
        """Drop-in replacement for one layer's fused experts. Wraps the frozen fused expert tensors and
        injects a per-expert LoRA for the experts in `node_of` (expert_idx -> node). Faithful copy of
        expert_shard_train.LoRAExperts (D:/glm_loader/repo)."""

        def __init__(self, base, node_of, r, alpha):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.num_experts = base.num_experts
            self.act_fn = base.act_fn
            self.r = r
            self.scale = alpha / r
            self.outer = 1.0
            self.node_of = {int(e): int(n) for e, n in node_of.items()}
            self.enabled_nodes = set()
            H = base.hidden_dim
            I = base.intermediate_dim
            dev = base.gate_up_proj.device
            self.A_gu, self.B_gu, self.A_d, self.B_d = (nn.ParameterDict() for _ in range(4))
            for e in sorted(self.node_of):
                k = str(e)
                self.A_gu[k] = nn.Parameter(torch.randn(r, H, device=dev, dtype=torch.float32) * 0.01)
                self.B_gu[k] = nn.Parameter(torch.zeros(2 * I, r, device=dev, dtype=torch.float32))
                self.A_d[k] = nn.Parameter(torch.randn(r, I, device=dev, dtype=torch.float32) * 0.01)
                self.B_d[k] = nn.Parameter(torch.zeros(H, r, device=dev, dtype=torch.float32))

        def params_for(self, node):
            out = []
            for e, n in self.node_of.items():
                if n == node:
                    k = str(e)
                    out += [self.A_gu[k], self.B_gu[k], self.A_d[k], self.B_d[k]]
            return out

        def _lora_on(self, e):
            return e in self.node_of and self.node_of[e] in self.enabled_nodes

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                em = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
            gu_w = self.base.gate_up_proj
            d_w = self.base.down_proj
            for ei in hit:
                e = int(ei[0])
                if e == self.num_experts:
                    continue
                pos, tok = torch.where(em[e])
                cs = hidden_states[tok]
                gu = F.linear(cs, gu_w[e])
                if self._lora_on(e):
                    k = str(e)
                    gu = gu + (self.scale * self.outer *
                               F.linear(F.linear(cs.float(), self.A_gu[k]), self.B_gu[k])).to(gu.dtype)
                gate, up = gu.chunk(2, dim=-1)
                h = self.act_fn(gate) * up
                dh = F.linear(h, d_w[e])
                if self._lora_on(e):
                    k = str(e)
                    dh = dh + (self.scale * self.outer *
                               F.linear(F.linear(h.float(), self.A_d[k]), self.B_d[k])).to(dh.dtype)
                dh = dh * top_k_weights[tok, pos, None]
                final.index_add_(0, tok, dh.to(final.dtype))
            return final

    return LoRAExperts


def _materialize_canonical(le, E):
    """Canonical per-expert WEIGHT delta {gate:[I,H], up:[I,H], down:[H,I]} from a single expert's LoRA
    factors (scale applied, outer NOT) -- expert_shard_train.materialize_canonical_from_saved semantics
    for one (L,E). Returned as numpy float32 (the lane's transport dtype)."""
    import torch
    with torch.no_grad():
        sc = le.scale
        gu = sc * (le.B_gu[str(E)].float() @ le.A_gu[str(E)].float())   # [2I, H]
        dn = sc * (le.B_d[str(E)].float() @ le.A_d[str(E)].float())     # [H, I]
        I = gu.shape[0] // 2
        return {"gate": gu[:I].cpu().numpy().astype(np.float32),
                "up": gu[I:].cpu().numpy().astype(np.float32),
                "down": dn.cpu().numpy().astype(np.float32)}


LORA_KEYS = ("lora_A_gu", "lora_B_gu", "lora_A_d", "lora_B_d", "lora_scale")


def garbage_lora(shape_ref, r=16, scale=3.0, seed=0):
    """An adversarial contribution ON THE REAL WIRE: random LoRA factors, not a random dense delta.

    The dense `garbage_delta` was an 18,874,493 B body, which the shared lane refuses outright --
    so it tested the store's size limit, not the gate. It also modelled the wrong attacker: nobody
    hostile pays 68x the bandwidth to be rejected. Sizing the attack exactly like an honest
    contribution (278,731 B) means the ONLY thing that can stop it is the secret-probe held-out
    gate, which is the property actually worth proving.

    `shape_ref` is a dense {gate:[I,H], up:[I,H], down:[H,I]} used purely for dimensions.
    """
    rng = np.random.default_rng(seed)
    I, Hd = shape_ref["gate"].shape
    return {"lora_A_gu": (rng.standard_normal((r, Hd)) * scale).astype(np.float32),
            "lora_B_gu": (rng.standard_normal((2 * I, r)) * scale).astype(np.float32),
            "lora_A_d": (rng.standard_normal((r, I)) * scale).astype(np.float32),
            "lora_B_d": (rng.standard_normal((Hd, r)) * scale).astype(np.float32),
            "lora_scale": np.asarray([1.0], dtype=np.float32)}


def is_lora_payload(d):
    """True iff `d` carries LoRA FACTORS rather than a dense {gate,up,down} weight delta."""
    return bool(d) and "lora_B_gu" in d


def validate_lora_factors(payload, I, H, max_rank=512):
    """Check an attacker-controlled LoRA-factor payload against the RESIDENT expert dims BEFORE it is
    fed to the float64 outer product in materialize_from_lora (F8). The payload arrives over the
    shared-token lane; its body size is bounded by content_store's MAX_BODY but its SHAPES are not,
    so a hostile record could otherwise drive a large or dimension-mismatched matmul on the
    coordinator (and, once materialised, a delta that does not even align with the slot it targets).

    For a resident expert with dense dims gate:[I,H], up:[I,H], down:[H,I], the only well-formed
    factors are A_gu:[r,H], B_gu:[2I,r], A_d:[r,I], B_d:[H,r], scale:[>=1]. Returns (ok, reason).
    """
    if not is_lora_payload(payload):
        return False, "not a LoRA-factor payload"
    try:
        A_gu = np.asarray(payload["lora_A_gu"])
        B_gu = np.asarray(payload["lora_B_gu"])
        A_d = np.asarray(payload["lora_A_d"])
        B_d = np.asarray(payload["lora_B_d"])
        scale = np.asarray(payload["lora_scale"]).reshape(-1)
    except Exception as ex:                                        # noqa: BLE001
        return False, "malformed factor array (%s)" % ex
    if any(a.ndim != 2 for a in (A_gu, B_gu, A_d, B_d)):
        return False, "a LoRA factor is not 2-D"
    if scale.size < 1:
        return False, "empty lora_scale"
    r = int(A_gu.shape[0])
    if not (1 <= r <= int(max_rank)):
        return False, "rank r=%d out of [1,%d]" % (r, max_rank)
    want = {"lora_A_gu": (r, H), "lora_B_gu": (2 * I, r), "lora_A_d": (r, I), "lora_B_d": (H, r)}
    got = {"lora_A_gu": A_gu.shape, "lora_B_gu": B_gu.shape, "lora_A_d": A_d.shape, "lora_B_d": B_d.shape}
    for k, exp in want.items():
        if tuple(got[k]) != tuple(exp):
            return False, "%s shape %s != expected %s (resident I=%d H=%d)" % (k, got[k], exp, I, H)
    return True, "ok"


def lora_factors_payload(le, E):
    """The contribution as LoRA FACTORS instead of their materialised product.

    WHY: the dense delta is gate[I,H] + up[I,H] + down[H,I] = 9,437,184 numbers = 18,874,568 B on
    the wire. The factors that GENERATED it are A_gu[r,H] + B_gu[2I,r] + A_d[r,I] + B_d[H,r] =
    139,264 numbers = 278,528 B at fp16 -- a 68x cut for information-identical content, because the
    dense delta is exactly scale*(B@A) and nothing else. Measured consequence: the shared VPS lane
    is a ~894 MB box that reset the connection on 18.87 MB bodies, so this is what makes real-GLM
    shardDiLoCo possible over WAN at all, not merely cheaper.

    Scale rides along as a 1-element array so the coordinator reproduces the product exactly rather
    than assuming alpha/r.
    """
    import torch
    with torch.no_grad():
        return {"lora_A_gu": le.A_gu[str(E)].detach().float().cpu().numpy(),
                "lora_B_gu": le.B_gu[str(E)].detach().float().cpu().numpy(),
                "lora_A_d": le.A_d[str(E)].detach().float().cpu().numpy(),
                "lora_B_d": le.B_d[str(E)].detach().float().cpu().numpy(),
                "lora_scale": np.asarray([float(le.scale)], dtype=np.float32)}


def materialize_from_lora(payload):
    """Inverse of lora_factors_payload: rebuild the dense {gate,up,down} the gate/merge path expects.

    Deliberately numpy-only (no torch): the coordinator runs this on every contribution and must not
    depend on the contributor's framework state. Computed in float64 then cast to float32 -- the
    factors arrive fp16-rounded, and doing the outer product in low precision compounds that error
    across the r-dimension sum for no saving (the product is transient, never transmitted).
    """
    sc = float(np.asarray(payload["lora_scale"]).reshape(-1)[0])
    gu = sc * (np.asarray(payload["lora_B_gu"], dtype=np.float64)
               @ np.asarray(payload["lora_A_gu"], dtype=np.float64))         # [2I, H]
    dn = sc * (np.asarray(payload["lora_B_d"], dtype=np.float64)
               @ np.asarray(payload["lora_A_d"], dtype=np.float64))          # [H, I]
    I = gu.shape[0] // 2
    return {"gate": gu[:I].astype(np.float32),
            "up": gu[I:].astype(np.float32),
            "down": dn.astype(np.float32)}


# =========================================================================== tiny REAL GLM builder
def build_tiny_glm(seed=1, vocab=24, hidden=64, inter=128, moe_inter=48, layers=3,
                   n_experts=2, topk=1):
    """Instantiate the SMALLEST viable REAL glm4_moe_lite: a real Glm4MoeLiteForCausalLM (real router,
    real fused-expert MLP) on CPU, from the real transformers config class. Layer 0 is dense (in the
    trunk); layers 1..layers-1 are MoE. This stands in for the 62 GB GLM-4.7-Flash whose trunk is too
    heavy to load on CPU (readiness doc sec c) -- the code paths exercised (routing, fused experts,
    per-expert LoRA, canonical delta, weight-space fold) are the SAME."""
    import torch
    from transformers import Glm4MoeLiteConfig
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM
    torch.manual_seed(seed)
    cfg = Glm4MoeLiteConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter, moe_intermediate_size=moe_inter,
        num_hidden_layers=layers, mlp_layer_types=["dense"] + ["sparse"] * (layers - 1),
        num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=n_experts,
        num_experts_per_tok=topk, n_group=1, topk_group=1, kv_lora_rank=32, q_lora_rank=48,
        qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=32, max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, dtype="float32",
        tie_word_embeddings=False)
    model = Glm4MoeLiteForCausalLM(cfg)
    model.eval()
    return model, cfg


def glm_fwd_flops_per_example(cfg, seq_len):
    """Approximate forward FLOPs for ONE sequence example (feeds neurahash.diloco_merge.FlopMeter, D3).
    Per token: attention proj (~2*H*(qk+v) heads) + shared expert + top-k routed experts (each an
    up/gate/down over moe_inter) + lm_head. x2 for multiply-add. Only the RELATIVE magnitude matters to
    the meter's gain/FLOP; this is a documented estimate, not a hardware count."""
    H = cfg.hidden_size
    I = cfg.moe_intermediate_size
    V = cfg.vocab_size
    per_expert = 3.0 * H * I          # gate + up + down projections
    routed = (cfg.num_experts_per_tok + cfg.n_shared_experts) * per_expert
    attn = 4.0 * H * H                # crude q/k/v/o projection cost
    lm_head = H * V
    per_token = 2.0 * (attn + routed + lm_head)
    return per_token * float(seq_len)


# ============================================================================= data (tokenizer-free)
def markov_dataset(vocab, seq_len, n, seed, transition):
    """n sequences of length seq_len sampled from a FIXED first-order Markov chain `transition`
    (vocab x vocab). A learnable structure so a real per-expert improvement (and its held-out
    generalization) is measurable on CPU in seconds -- no tokenizer needed (token ids are the vocab)."""
    rng = np.random.default_rng(seed)
    seqs = np.zeros((n, seq_len), dtype=np.int64)
    for i in range(n):
        s = int(rng.integers(0, vocab))
        for t in range(seq_len):
            seqs[i, t] = s
            s = int(rng.choice(vocab, p=transition[s]))
    return seqs


def make_transition(vocab, seed, peak=12):
    rng = np.random.default_rng(seed)
    P = rng.random((vocab, vocab)) ** peak      # peaky -> low-entropy, trivially learnable
    return P / P.sum(1, keepdims=True)


# ============================================================================= CE (mirrors soak_glm)
def heldout_ce(model, ids):
    """Mean next-token cross-entropy over `ids` [N,T] by a FORWARD pass only -- the same CE definition
    as fleet/soak_glm._heldout_ce (the built GLM gate), tokenizer-free."""
    import torch
    model.eval()
    # Token ids must land on the MODEL's device. Every caller of this file until 2026-07-21 ran on
    # CPU (run_smoke pins CUDA_VISIBLE_DEVICES=""), so a cpu ids tensor happened to match; on a
    # cuda model it raises "Expected all tensors to be on the same device ... index is on cpu".
    dev = next(model.parameters()).device
    # CHUNKED, and chunked HERE rather than at the call sites. One forward over the whole pool
    # materialises [N, T-1, vocab] fp32 logits AND the same again through log_softmax: at N=512,
    # T=32, vocab=154880 that is ~10 GiB per copy, which OOM'd the real-GLM run through the
    # contributor's save-best path even though the coordinator had its own chunking. This is the
    # same class of bug as the recorded `_eval_trunk` hardcoded batch=64 x vocab blowup (memory
    # vram-cap-live-verified) -- so the guard belongs in the function that allocates, where no
    # future caller can forget it. Mean over all predicted tokens is exact for constant T.
    n = int(os.environ.get("NEURAHASH_GLM_EVAL_CHUNK", "8"))
    n = max(1, n)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ids), n):
            t = torch.as_tensor(ids[i:i + n]).to(dev)
            logits = model(input_ids=t).logits
            sl = logits[:, :-1, :].float()
            st = t[:, 1:]
            lp = torch.log_softmax(sl, dim=-1)
            tok_lp = lp.gather(-1, st.unsqueeze(-1)).squeeze(-1)
            tot += float(tok_lp.sum())
            cnt += int(st.numel())
            del logits, sl, lp, tok_lp, t
    return float(-(tot / max(1, cnt)))


# ================================================================================ the GLM lane host
class GlmExpertLaneHost:
    """Maps GLM (layer, expert) units onto the shardDiLoCo lane's per-expert `experts[]` slots and
    supplies the numpy<->torch `eval_expert` bridge the lane's shard_merge_round calls. All merge/gate
    math stays in numpy inside the lane; this host only reads/writes the model's fused expert slices."""

    def __init__(self, model, cfg, slots):
        self.model = model
        self.cfg = cfg
        self.slots = [(int(L), int(E)) for (L, E) in slots]     # slot index -> (layer, expert)
        self.I = cfg.moe_intermediate_size
        self._base_slots = None                                 # frozen pre-round base (all slots)

    def _fused(self, L):
        exp = self.model.model.layers[L].mlp.experts
        return exp.base if hasattr(exp, "base") else exp

    def read_slot(self, idx):
        """Current canonical {gate,up,down} of slot idx, as numpy float32 (a copy)."""
        import torch
        L, E = self.slots[idx]
        exp = self._fused(L)
        I = self.I
        with torch.no_grad():
            # .float() BEFORE .numpy(): numpy has no bfloat16, so a real bf16 GLM trunk raises
            # "TypeError: Got unsupported ScalarType BFloat16" if the cast happens numpy-side.
            # (Measured 2026-07-21 on the first real-dtype load; the tiny test model is float32,
            # which is why every prior caller survived.) The returned dtype is unchanged.
            return {"gate": exp.gate_up_proj[E, :I].detach().float().cpu().numpy(),
                    "up": exp.gate_up_proj[E, I:].detach().float().cpu().numpy(),
                    "down": exp.down_proj[E].detach().float().cpu().numpy()}

    def write_slot(self, idx, d):
        """Overwrite slot idx's fused weights with dict d (numpy or torch {gate,up,down})."""
        import torch
        L, E = self.slots[idx]
        exp = self._fused(L)
        I = self.I
        dev, dt = exp.gate_up_proj.device, exp.gate_up_proj.dtype
        with torch.no_grad():
            exp.gate_up_proj[E, :I].copy_(torch.as_tensor(np.asarray(d["gate"])).to(dev, dt))
            exp.gate_up_proj[E, I:].copy_(torch.as_tensor(np.asarray(d["up"])).to(dev, dt))
            exp.down_proj[E].copy_(torch.as_tensor(np.asarray(d["down"])).to(dev, dt))

    def canonical_experts(self):
        """The lane's `experts` argument: list of {gate,up,down} numpy dicts, one per slot. These are
        MUTATED IN PLACE by shard_merge_round on accept (base += outer*delta), exactly mirroring
        expert_shard_train.apply_canonical_to_fused on the real fused weights."""
        return [self.read_slot(i) for i in range(len(self.slots))]

    def begin_round(self, experts):
        """Snapshot the pre-round base of every slot so eval_expert can gate each slot INDEPENDENTLY
        against a fixed base (the model holds pre-round base for all non-active slots throughout)."""
        self._base_slots = [{k: v.copy() for k, v in e.items()} for e in experts]

    def sync_from_canonical(self, experts):
        """After a round, write the (possibly accepted) canonical experts back into the model."""
        for i, e in enumerate(experts):
            self.write_slot(i, e)

    def make_eval_expert(self, meter, seq_len):
        """Return eval_expert(e, cand, pX, pY) -> held-out CE float. Writes `cand` into slot e, runs a
        real GLM forward-CE on the SECRET probe batch pX, restores slot e to the frozen pre-round base,
        and books the forward FLOPs on `meter` (D3). Non-active slots stay at pre-round base -> each
        delta is gated independently, matching the lane's per-expert design."""
        def eval_expert(e, cand, pX, pY):
            self.write_slot(e, cand)
            ce = heldout_ce(self.model, pX)
            self.write_slot(e, self._base_slots[e])       # restore -> model back to pre-round base
            meter.add_verify(len(pX))
            return ce
        return eval_expert


# ================================================================= contributor: train one GLM expert
def train_glm_expert_contribution(model, cfg, L, E, train_ids, val_ids, *, H=120, r=16, alpha=None,
                                   lr=3e-3, batch=48, meter=None, seed=0, sel_outer=0.7):
    """CONTRIBUTOR local step for GLM expert (L,E): attach a per-expert LoRA (the built GLM path), train
    only that LoRA for up to H inner steps on the node's train shard, SAVE-BEST on the node's PUBLIC val
    (memory moe-capability-gain: LR small, fp32, save-best), then materialize the canonical {gate,up,down}
    weight delta = scale*(B@A). The base fused weights are FROZEN and untouched, so the delta is purely
    the contribution. Returns {delta (numpy float32), train_flops, n_examples, best_val_ce}.

    F5 -- PREDICTABLE ACCEPTANCE. The coordinator GATES + MERGES the uploaded adapter at LoRA strength
    `outer` (base += outer*delta; default 0.7), but this SAVE-BEST selection used to evaluate at the
    LoRAExperts default strength 1.0, so the miner's own best_val_ce could not predict the gate. The
    selection/evaluation now runs at `sel_outer` (thread the coordinator's --outer here; default 0.7)
    while TRAINING stays at full strength (le.outer=1.0) -- only the metric the miner selects on aligns
    with what the coordinator will actually apply."""
    import torch
    if alpha is None:
        alpha = 2 * r
    # FREEZE THE TRUNK before attaching LoRA. The docstring above and the lane design both say the
    # GLM trunk is frozen (tools/sharddiloco_glm_expert.py trunk_delta={} at the merge site), and the
    # optimizer below only ever sees LoRA params -- but nothing had actually cleared requires_grad on
    # the trunk, so autograd still allocated and populated a full set of trunk gradients every
    # backward and then discarded them. MEASURED 2026-07-21: ~4 GiB of pure waste on a 4.02 GiB bf16
    # trunk, which is the difference between fitting an 8 GB consumer card and not. LoRA parameters
    # are created AFTER this loop, so they keep requires_grad=True and remain trainable.
    for _p in model.parameters():
        _p.requires_grad_(False)
    LoRAExperts = _lora_experts_cls()
    layer = model.model.layers[L]
    base = layer.mlp.experts
    le = LoRAExperts(base, {E: 0}, r=r, alpha=alpha)
    layer.mlp.experts = le
    le.enabled_nodes = {0}
    opt = torch.optim.AdamW(le.params_for(0), lr=lr)

    def _snap():
        return {n: getattr(le, n)[str(E)].detach().clone() for n in ("A_gu", "B_gu", "A_d", "B_d")}

    def _restore(s):
        with torch.no_grad():
            for n, v in s.items():
                getattr(le, n)[str(E)].copy_(v)

    def _sel_val():
        # F5: SELECT save-best at the SAME LoRA strength the coordinator gates + merges at
        # (`sel_outer`, default 0.7), NOT the training strength (le.outer=1.0). The miner uploads a
        # full-strength adapter that the coordinator applies as base += sel_outer*delta, so a snapshot
        # best at 1.0 need not be best at 0.7; selecting at 1.0 made best_val_ce unable to predict the
        # gate. Only the SELECTION metric aligns -- training below stays at full strength.
        prev = le.outer
        le.outer = float(sel_outer)
        try:
            return heldout_ce(model, val_ids)
        finally:
            le.outer = prev

    fwd = glm_fwd_flops_per_example(cfg, train_ids.shape[1])
    best_val, best_snap = _sel_val(), _snap()
    n_ex = 0
    for step in range(1, H + 1):
        idx = np.random.default_rng(1000 + seed * 7919 + step).integers(0, len(train_ids), size=batch)
        ids = torch.as_tensor(train_ids[idx]).to(next(model.parameters()).device)
        model.train()
        out = model(input_ids=ids, labels=ids)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        n_ex += len(ids)
        if meter is not None:
            meter.add_train(len(ids))
        if step % 8 == 0:
            v = _sel_val()
            if v < best_val:
                best_val, best_snap = v, _snap()
    _restore(best_snap)
    delta = _materialize_canonical(le, E)
    layer.mlp.experts = base                     # detach LoRA -> plain fused model for the coordinator
    train_flops = 3.0 * fwd * n_ex               # forward + backward (D3 convention)
    # `lora` is the SAME contribution as `delta`, 68x smaller: delta == materialize_from_lora(lora).
    # Returned alongside rather than instead so callers choose the wire without changing this kernel.
    return {"delta": delta, "lora": lora_factors_payload(le, E),
            "train_flops": train_flops, "n_examples": n_ex, "best_val_ce": best_val}


def garbage_delta(shape_ref, scale=3.0, seed=123):
    """A harmful/garbage GLM expert delta (same shapes as a real one) -- large random weights that MUST
    worsen held-out CE and be REJECTED by the secret-probe gate."""
    rng = np.random.default_rng(seed)
    return {k: (rng.standard_normal(v.shape).astype(np.float32) * scale) for k, v in shape_ref.items()}


# =========================================================================================== smoke
def run_smoke(seed=1, verbose=True):
    """End-to-end CPU smoke exercising the REAL wired path (readiness doc Route A acceptance):
    2 GLM experts on 2 MoE layers, each LoRA-trained a few CPU steps -> canonical delta -> published on
    the SIGNED content-addressed ShardDeltaLane (sig + CID verified) -> gated by shard_merge_round on the
    SECRET rotated probe (held-out CE) -> merged -> FLOP-metered. Adds a garbage delta that must be
    REJECTED. Returns a result dict (also printed). No GPU / no fleet / no spend.

    COORDINATOR-ONLY: needs the private merge/gate machinery (see the module-level import note).
    This function does not run from a public checkout, by design; the contributor path does."""
    if shard_merge_round is None:
        raise SystemExit("run_smoke() is coordinator-only: neurahash.diloco_merge in this checkout "
                         "does not ship shard_merge_round/SecretRotatedProbe (contributor subset).")
    import torch
    t0 = time.time()
    VOCAB, SEQ = 24, 16
    P = make_transition(VOCAB, seed=7, peak=12)
    train = markov_dataset(VOCAB, SEQ, 3000, seed=100, transition=P)   # contributors' train shard
    val = markov_dataset(VOCAB, SEQ, 160, seed=555, transition=P)      # contributors' PUBLIC val (save-best)
    secret = markov_dataset(VOCAB, SEQ, 128, seed=999, transition=P)   # coordinator's SECRET probe pool

    model, cfg = build_tiny_glm(seed=seed, vocab=VOCAB, n_experts=2, topk=1, layers=3)

    # --- warm-start ALL params (save-best on the secret held-out) to stand in for a PRETRAINED GLM
    # base (real experts carry signal). The 62 GB GLM-4.7-Flash is already pretrained; this is the CPU
    # substitute. After this the base is FROZEN; only per-expert LoRA is trained by contributors.
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02)
    best, best_sd = 1e9, None
    for step in range(1, 601):
        idx = np.random.default_rng(step).integers(0, len(train), size=48)
        ids = torch.as_tensor(train[idx]).to(next(model.parameters()).device)
        model.train()
        out = model(input_ids=ids, labels=ids)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if step % 20 == 0:
            h = heldout_ce(model, secret)
            if h < best:
                best, best_sd = h, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    model.eval()
    uniform_ce = float(np.log(VOCAB))
    entropy_floor = float(-(P * np.log(np.clip(P, 1e-12, 1))).sum(1).mean())

    # --- host 2 GLM experts as 2 disjoint lane slots (different layers) ---
    slots = [(1, 0), (2, 1)]                       # (layer, expert) per slot; disjoint units
    host = GlmExpertLaneHost(model, cfg, slots)
    experts = host.canonical_experts()            # the lane's canonical per-slot params (numpy)
    base_ce = heldout_ce(model, secret)

    # --- SECRET rotated probe: per-slot private held-out pools (miner never sees these) ---
    pools = {i: (secret.copy(), secret.copy()) for i in range(len(slots))}
    probe = SecretRotatedProbe(pools, seed=0, size=96)
    meter = FlopMeter(glm_fwd_flops_per_example(cfg, SEQ))
    lane = ShardDeltaLane()

    # --- contributors: each OWNS one slot, trains its GLM expert, publishes a SIGNED, CID'd delta ---
    RND = 0
    contributions = []
    train_report = []
    for i, (L, E) in enumerate(slots):
        key = ("miner%d-secret-key" % i).encode()
        c = train_glm_expert_contribution(model, cfg, L, E, train, val, H=120, r=16, lr=3e-3,
                                          meter=meter, seed=i, sel_outer=0.7)   # match the merge outer below
        host.write_slot(i, experts[i])            # ensure model back at canonical after training
        cid = lane.put(c["delta"])                # content-address (sha256 over fp16 wire)
        sig = lane.sign(key, cid, RND, "miner%d" % i)
        # coordinator side: verify signature, then re-fetch by CID (tamper -> raises)
        verify_ok = ShardDeltaLane.verify(key, sig, cid, RND, "miner%d" % i)
        wire = lane.get(cid)                      # CID re-verified inside get()
        contributions.append(dict(miner="miner%d" % i, expert=i, base_round=RND, cid=cid, sig=sig,
                                  verify_ok=verify_ok, trunk_delta={},   # GLM trunk FROZEN -> no trunk grad
                                  expert_delta=wire, train_flops=c["train_flops"]))
        train_report.append((i, L, E, c["best_val_ce"], cid[:12], verify_ok))

    # --- a GARBAGE delta for slot 0 (valid signature, harmful content) -> must be REJECTED by the gate ---
    gkey = b"attacker-key"
    gd = garbage_delta(experts[0], scale=3.0)
    gcid = lane.put(gd)
    gsig = lane.sign(gkey, gcid, RND, "attacker")
    contributions.append(dict(miner="attacker", expert=0, base_round=RND, cid=gcid, sig=gsig,
                              verify_ok=ShardDeltaLane.verify(gkey, gsig, gcid, RND, "attacker"),
                              trunk_delta={}, expert_delta=lane.get(gcid),
                              train_flops=1.0))

    # --- COORDINATOR merge round: signed + secret-probe-gated + FLOP-metered (trunk frozen -> {} ) ---
    host.begin_round(experts)
    eval_expert = host.make_eval_expert(meter, SEQ)
    res = shard_merge_round(trunk={}, experts=experts, contributions=contributions,
                            eval_expert=eval_expert, probe=probe, meter=meter, rnd=RND,
                            outer=0.7, margin=0.0)
    host.sync_from_canonical(experts)
    merged_ce = heldout_ce(model, secret)

    # --- verify signed/CID lane rejects tampering (sanity, not part of the merge) ---
    tamper_caught = False
    try:
        bad = dict(wire)  # noqa: F841
        lane._store[contributions[0]["cid"]]["gate"][0, 0] += 9.0
        lane.get(contributions[0]["cid"])
    except ValueError:
        tamper_caught = True

    out = dict(
        wall_s=round(time.time() - t0, 1),
        uniform_ce=round(uniform_ce, 4), entropy_floor=round(entropy_floor, 4),
        base_ce=round(base_ce, 5), merged_ce=round(merged_ce, 5),
        accepts=res["accepts"], rejects=res["rejects"], minted=round(res["minted"], 6),
        trunk_merged=res["trunk_merged"],
        train_flops=round(meter.train, 1), verify_flops=round(meter.verify, 1),
        per_expert=[dict(miner=p["miner"], slot=p["expert"], accepted=p["accepted"],
                         base_val=round(p["base_val"], 5) if p["base_val"] is not None else None,
                         merged_val=round(p["merged_val"], 5) if p["merged_val"] is not None else None,
                         gain=round(p["gain"], 6), gain_per_flop=("%.3e" % p["gain_per_flop"]),
                         delta_norm=round(p["delta_norm"], 4) if p["delta_norm"] is not None else None)
                    for p in res["per_expert"]],
        tamper_caught=tamper_caught,
        n_model_params=int(sum(p.numel() for p in model.parameters())),
    )
    if verbose:
        print("=== shardDiLoCo Phase-4 GLM wiring (Route A) CPU smoke ===")
        print("REAL torch model : Glm4MoeLiteForCausalLM  params=%d  layers=%d (L0 dense, L1/L2 MoE)  "
              "experts/layer=%d topk=%d  vocab=%d" % (out["n_model_params"], cfg.num_hidden_layers,
              cfg.n_routed_experts, cfg.num_experts_per_tok, VOCAB))
        print("task             : fixed 1st-order Markov  uniform_ce=%.3f  entropy_floor=%.3f  "
              "base_ce=%.5f (learned, << uniform)" % (uniform_ce, entropy_floor, base_ce))
        for (i, L, E, bv, cidp, vok) in train_report:
            print("contributor %d    : GLM expert (L%d,E%d) LoRA-trained -> canonical delta  "
                  "best_val_ce=%.5f  cid=%s.. sig_ok=%s" % (i, L, E, bv, cidp, vok))
        print("GARBAGE delta    : slot0 random*3.0  (valid sig, harmful content)")
        print("--- shard_merge_round (secret rotated probe gate, trunk FROZEN) ---")
        for p in out["per_expert"]:
            print("  %-9s slot%d  base_ce=%s merged_ce=%s  gain=%+.6f  |delta|=%s  -> %s" % (
                p["miner"], p["slot"], p["base_val"], p["merged_val"], p["gain"], p["delta_norm"],
                "ACCEPTED" if p["accepted"] else "REJECTED"))
        print("RESULT           : accepts=%d rejects=%d minted=%.6f  base_ce=%.5f -> merged_ce=%.5f "
              "(delta %+.6f)" % (out["accepts"], out["rejects"], out["minted"], base_ce, merged_ce,
              merged_ce - base_ce))
        print("FLOP meter (D3)  : train=%.3e verify=%.3e  gain/flop shown per-expert" % (
            meter.train, meter.verify))
        print("signed lane      : tampered payload caught by CID re-verify = %s" % out["tamper_caught"])
        print("wall=%.1fs  (CPU-only, CUDA_VISIBLE_DEVICES=%r)  NO fleet / NO spend" % (
            out["wall_s"], os.environ.get("CUDA_VISIBLE_DEVICES")))
    return out


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    run_smoke()
