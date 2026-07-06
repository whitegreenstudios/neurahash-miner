"""
sparse_moe.py — Milestone 3: delete the einsum wall.

The existing MoEFFN (model_torch.py) stores experts as ONE stacked parameter
`W1 (E, d_model, d_ff)` and computes EVERY expert each forward via
`einsum('nc,ecf->enf', ...)`. Correct and fast, but it means a node literally
cannot instantiate the module without holding all E experts — which is the wall
that blocks model-parallel sharding of a 256-expert / 671B-class MoE.

This module makes each expert an INDEPENDENTLY-INSTANTIABLE `ExpertFFN`, and the
MoE layer only holds + computes the experts a node actually HOSTS (`hosted=`).
A node hosting experts {3, 17} of 256 allocates two experts' worth of params and
runs only those — true top-k sparse dispatch (compute only routed, hosted experts).
Tokens routed to a non-hosted expert get no local contribution (in a real fleet
they are sent to that expert's host; here they are simply skipped and counted).

When a node hosts ALL experts, the output is numerically equal to the dense
MoEFFN (same experts, same gates) — verified in tests — so this is a faithful
refactor, not a different model. The default model keeps the dense path; the
sparse path is opt-in (`MoETransformer(moe="sparse", hosted_experts=...)`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """One MoE expert as a standalone module. Param layout matches a single slice of the
    stacked dense MoEFFN (W1 (d_model,d_ff), b1, W2 (d_ff,d_model), b2) so a dense expert
    can be copied in 1:1 for exact-equivalence checks."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W2, a=5 ** 0.5)

    def forward(self, x):                       # x: (n_tokens, d_model)
        h = F.gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


class SparseMoEFFN(nn.Module):
    """Top-k MoE FFN that holds and computes only the experts this node HOSTS.

    router          — the full (E-way) gate; it is part of the replicated TRUNK, so every
                      node has it and sees the full routing distribution (needed for the
                      load-balancing aux loss and to know which tokens are 'remote').
    experts         — an nn.ModuleDict keyed by expert index, containing ONLY hosted experts.
    hosted          — sorted list of expert indices this node holds (None = all of them).
    last_aux        — Switch-style load-balancing aux loss (computed over the full router,
                      identical to dense, so it does not depend on which experts are hosted).
    last_dropped    — fraction of (token, slot) routings whose expert was NOT hosted locally
                      (0.0 when hosting all experts); exposed so a node knows its remote load.
    """

    def __init__(self, d_model, d_ff, n_experts, top_k=2, hosted=None):
        super().__init__()
        self.d_model, self.d_ff, self.top_k = d_model, d_ff, top_k
        self._n_experts = n_experts
        self.hosted = list(range(n_experts)) if hosted is None else sorted(set(hosted))
        if not all(0 <= e < n_experts for e in self.hosted):
            raise ValueError(f"hosted experts {self.hosted} out of range [0,{n_experts})")
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleDict({str(e): ExpertFFN(d_model, d_ff) for e in self.hosted})
        self.last_aux = 0.0
        self.last_dropped = 0.0

    @property
    def n_experts(self):
        return self._n_experts

    @property
    def hosts_all(self):
        return len(self.hosted) == self._n_experts

    def expert_param_count(self):
        """Params in ONE expert (what an expert-host node holds per expert)."""
        e = next(iter(self.experts.values()))
        return sum(p.numel() for p in e.parameters())

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)                                   # (N, C)
        N = xf.shape[0]
        logits = self.router(xf)                               # (N, E) — full router
        k = min(self.top_k, self._n_experts)
        topv, topi = logits.topk(k, dim=-1)                    # (N, k)
        gates = F.softmax(topv, dim=-1)                        # (N, k)

        # load-balancing aux loss over the FULL router (identical to dense MoEFFN)
        full_gates = F.softmax(logits, dim=-1)
        importance = full_gates.mean(0)
        disp = torch.zeros(N, self._n_experts, device=xf.device)
        disp.scatter_(1, topi, 1.0)
        load = disp.mean(0)
        self.last_aux = self._n_experts * (importance * load).sum()

        hosted = set(self.hosted)
        out = torch.zeros_like(xf)
        dropped = 0
        for slot in range(k):
            e_idx = topi[:, slot]                              # (N,) expert chosen per token
            g = gates[:, slot]                                 # (N,)
            for e in self.hosted:                              # only experts THIS node holds
                m = (e_idx == e).nonzero(as_tuple=True)[0]
                if m.numel():
                    y = self.experts[str(e)](xf.index_select(0, m))
                    out = out.index_add(0, m, g.index_select(0, m).unsqueeze(1) * y)
            if len(hosted) < self._n_experts:                  # count remote (non-hosted) routings
                dropped += int((~torch.isin(e_idx, torch.tensor(self.hosted, device=xf.device))).sum())
        self.last_dropped = dropped / max(1, N * k)
        return out.reshape(B, T, C)

    @torch.no_grad()
    def add_expert(self):
        """Grow the model by one expert. The new expert is hosted locally (a node that grows
        capacity takes custody of the new expert); the router gains one output."""
        e_new = self._n_experts
        self._n_experts += 1
        self.experts[str(e_new)] = ExpertFFN(self.d_model, self.d_ff).to(self.router.weight.device)
        self.hosted.append(e_new)
        old = self.router
        new = nn.Linear(self.d_model, self._n_experts).to(old.weight.device)
        new.weight[: e_new] = old.weight
        new.bias[: e_new] = old.bias
        new.weight[e_new].normal_(0, 0.02)                     # new expert starts ~unused
        new.bias[e_new].zero_()
        self.router = new

    # ---- interop with the dense MoEFFN (for exact-equivalence checks / import) ----
    @torch.no_grad()
    def load_from_dense(self, dense):
        """Copy weights from a stacked dense MoEFFN so the two produce identical output."""
        self.router.weight.copy_(dense.router.weight)
        self.router.bias.copy_(dense.router.bias)
        for e in self.hosted:
            ex = self.experts[str(e)]
            ex.W1.copy_(dense.W1[e]); ex.b1.copy_(dense.b1[e])
            ex.W2.copy_(dense.W2[e]); ex.b2.copy_(dense.b2[e])
