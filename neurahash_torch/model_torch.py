"""
A REAL (small) Mixture-of-Experts transformer in PyTorch.

This is the production-shaped version of the toy NumPy model:
  - causal self-attention (a real GPT block)
  - the FFN in every block is a top-2 Mixture-of-Experts
  - add_expert() grows the model (more capacity) -> the "bigger and bigger" story
  - autograd does the backprop (no manual gradients), runs on the GPU

Because it is real PyTorch on a real GPU, we can MEASURE its VRAM footprint with
torch.cuda instead of scaling a guess (see vram_torch.py).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

# (#45 8GB-card VRAM fit, 2026-07-10) gradient checkpointing for the trunk transformer blocks.
# DEFAULT OFF: unset / "0" / "false" => today's exact behavior, byte-for-byte (verified
# BIT-IDENTICAL trunk_delta_hash OFF vs ON on the real medium recompute). When on, each Block's
# forward is wrapped in checkpoint(use_reentrant=False): the forward runs fully (autograd-tracked)
# but its internal activations are NOT retained; backward re-runs the same deterministic forward to
# regenerate them -> identical gradients, ~21-28% less peak VRAM (fits 8GB cards). use_reentrant=
# False preserves autograd graph connectivity (MoEFFN.last_aux, read after the block loop).
_GRAD_CKPT = os.environ.get("NEURAHASH_GRAD_CHECKPOINT", "") not in ("", "0", "false", "False")


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        # DEAD CODE — DELIBERATELY NOT WIRED INTO forward() (see below). `self.drop` is constructed but
        # never called: dropout is STOCHASTIC, and the pool's whole trust model is bit-exact / direction-
        # stable recompute-verify — the coordinator re-runs the worker's exact round and compares the trunk
        # delta. A worker and the coordinator would draw DIFFERENT dropout masks, rotating the gradient and
        # collapsing the verify cosine -> honest work would be rejected. DO NOT enable it here.
        # TODO(determinism): before dropout can be turned on, it must be a SEEDED/deterministic dropout
        # whose mask is derived from the SAME per-round seed the worker and the coordinator's recompute both
        # use (cf. get_batch's seeded generator), so both sides draw identical masks and recompute-verify
        # still reproduces. Until then it stays off and `dropout` defaults to 0.0 (a no-op even if called).
        self.drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MoEFFN(nn.Module):
    """Top-2 Mixture-of-Experts FFN with STACKED expert weights, so all experts are
    computed in one batched einsum instead of a Python loop over experts. Audit
    measured the old per-expert loop at 27-57x slower; this version is numerically
    identical (verified to ~3e-7) and roughly flat as experts grow.

    Weights are stacked: W1 (E, d_model, d_ff), W2 (E, d_ff, d_model)."""
    def __init__(self, d_model, d_ff, n_experts, top_k=2):
        super().__init__()
        self.d_model, self.d_ff, self.top_k = d_model, d_ff, top_k
        self.W1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.b1 = nn.Parameter(torch.zeros(n_experts, d_ff))
        self.W2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        self.router = nn.Linear(d_model, n_experts)
        self.last_aux = 0.0          # load-balancing aux loss from the last forward
        for e in range(n_experts):
            nn.init.kaiming_uniform_(self.W1[e], a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.W2[e], a=5 ** 0.5)

    @property
    def n_experts(self):
        return self.W1.shape[0]

    def expert_param_count(self):
        """Params in ONE expert (what an expert-host node would hold)."""
        return self.W1[0].numel() + self.b1[0].numel() + self.W2[0].numel() + self.b2[0].numel()

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)                                  # (N, C)
        N = xf.shape[0]
        logits = self.router(xf)                               # (N, E)
        k = min(self.top_k, self.n_experts)
        topv, topi = logits.topk(k, dim=-1)                    # (N, k)
        gates = F.softmax(topv, dim=-1)                        # (N, k)

        # Switch-style load-balancing aux loss: encourages tokens to spread across
        # experts so newly grown experts actually receive traffic (else the router
        # can collapse onto a few and growth adds dead capacity).
        full_gates = F.softmax(logits, dim=-1)                 # (N, E)
        importance = full_gates.mean(0)                        # (E,)
        disp = torch.zeros(N, self.n_experts, device=xf.device)
        disp.scatter_(1, topi, 1.0)
        load = disp.mean(0)                                    # (E,)
        self.last_aux = self.n_experts * (importance * load).sum()

        # all experts, all tokens, in two batched einsums (dense; N >> E so this wins)
        h = torch.einsum('nc,ecf->enf', xf, self.W1) + self.b1.unsqueeze(1)   # (E, N, f)
        h = F.gelu(h)
        oe = torch.einsum('enf,efc->enc', h, self.W2) + self.b2.unsqueeze(1)  # (E, N, C)

        # gather only the top-k experts chosen per token and blend by gate weight
        ar = torch.arange(N, device=xf.device)
        out = torch.zeros_like(xf)
        for slot in range(k):                                  # k=2 iterations, not E
            out = out + gates[:, slot:slot + 1] * oe[topi[:, slot], ar]
        return out.reshape(B, T, C)

    @torch.no_grad()
    def add_expert(self):
        device = self.W1.device
        nW1 = torch.empty(1, self.d_model, self.d_ff, device=device)
        nW2 = torch.empty(1, self.d_ff, self.d_model, device=device)
        nn.init.kaiming_uniform_(nW1[0], a=5 ** 0.5)
        nn.init.kaiming_uniform_(nW2[0], a=5 ** 0.5)
        self.W1 = nn.Parameter(torch.cat([self.W1.data, nW1], 0))
        self.W2 = nn.Parameter(torch.cat([self.W2.data, nW2], 0))
        self.b1 = nn.Parameter(torch.cat([self.b1.data, torch.zeros(1, self.d_ff, device=device)], 0))
        self.b2 = nn.Parameter(torch.cat([self.b2.data, torch.zeros(1, self.d_model, device=device)], 0))
        E = self.n_experts
        new_router = nn.Linear(self.d_model, E).to(device)
        new_router.weight[:E - 1] = self.router.weight
        new_router.bias[:E - 1] = self.router.bias
        new_router.weight[E - 1].normal_(0, 0.02)   # new expert starts ~unused
        new_router.bias[E - 1].zero_()
        self.router = new_router


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, n_experts, block_size, top_k=2,
                 attention="dense", moe="dense", hosted_experts=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        # attention="dense" (default) keeps the existing quadratic causal attention; "nsa"
        # swaps in the deterministic NSA-style sparse attention whose block selection is
        # hardware-stable + commit-verifiable (neurahash_torch.nsa_attention). Default keeps
        # the protocol model and every existing test byte-identical.
        if attention == "nsa":
            from neurahash_torch.nsa_attention import NSAAttention
            self.attn = NSAAttention(d_model, n_head, block_size)
        elif attention == "dense":
            self.attn = CausalSelfAttention(d_model, n_head, block_size)
        else:
            raise ValueError(f"unknown attention {attention!r} (expected 'dense' or 'nsa')")
        self.ln2 = nn.LayerNorm(d_model)
        # moe="dense" (default) = the stacked-einsum MoEFFN (computes ALL experts, byte-identical
        # to before). "sparse" = SparseMoEFFN which holds + computes only the experts this node
        # HOSTS (Milestone 3: lets a node train a strict subset of a big MoE — deletes the wall).
        if moe == "dense":
            self.moe = MoEFFN(d_model, d_ff, n_experts, top_k)
        elif moe == "sparse":
            from neurahash_torch.sparse_moe import SparseMoEFFN
            self.moe = SparseMoEFFN(d_model, d_ff, n_experts, top_k, hosted=hosted_experts)
        else:
            raise ValueError(f"unknown moe {moe!r} (expected 'dense' or 'sparse')")

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class MoETransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_head=4, n_layers=2, d_ff=128,
                 n_experts=2, block_size=32, top_k=2, aux_coef=0.01, attention="dense",
                 moe="dense", hosted_experts=None):
        super().__init__()
        self.block_size = block_size
        self.aux_coef = aux_coef
        self.attention = attention
        self.moe_kind = moe
        # hosted_experts (sparse only): the subset of experts this NODE holds in every layer;
        # None = host all. A model-parallel fleet gives different nodes different subsets.
        self.hosted_experts = hosted_experts
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, d_ff, n_experts, block_size, top_k, attention=attention,
                  moe=moe, hosted_experts=hosted_experts)
            for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def n_experts(self):
        return self.blocks[0].moe.n_experts

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for b in self.blocks:
            # only checkpoint while gradients are tracked (the recompute's H-loop training forward);
            # generate()'s no_grad forward is unaffected either way. Bit-identical to the plain path.
            if _GRAD_CKPT and torch.is_grad_enabled():
                x = _ckpt.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            aux = sum(b.moe.last_aux for b in self.blocks)     # load-balancing term
            loss = loss + self.aux_coef * aux
        return logits, loss

    def add_expert(self):
        """Grow capacity: add one expert to the MoE in every block."""
        for b in self.blocks:
            b.moe.add_expert()

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def expert_param_count(self):
        """Params in ONE expert across all blocks (what an expert-host node holds)."""
        return self.blocks[0].moe.expert_param_count() * len(self.blocks)

    def shared_param_count(self):
        return self.num_params() - self.expert_param_count() * self.n_experts

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8):
        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
