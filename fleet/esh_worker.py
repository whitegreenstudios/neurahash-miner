"""Rung B fleet WORKER -- transformers-version-AGNOSTIC. A miner runs this to train ONLY its disjoint expert
group and upload CANONICAL per-projection LoRA factors (which the coordinator gathers by weight-add regardless
of the miner's runtime -- see tools/expert_shard_train.py gather-canonical).

Compatibility is handled two ways (belt and suspenders):
  1. CODE: auto-detects the expert layout -- FUSED (transformers >=5.x: layer.mlp.experts.gate_up_proj) OR
     UNFUSED (transformers 4.x: layer.mlp.experts is a ModuleList of gate_proj/up_proj/down_proj Linears) --
     and attaches LoRA correctly to either. Both export the SAME canonical format.
  2. ENV: fleet_join.bat builds a pinned venv (transformers 4.49 = unfused -> nf4-quantizable -> fits 8GB
     cards), so a new miner just double-clicks and is guaranteed compatible.

Small cards: --load-4bit uses bitsandbytes nf4 (works on the UNFUSED path; FUSED expert Parameters are not
bnb-4bit-able, so on transformers 5.x use a card that fits bf16, or the pinned tf4.49 env). Self-contained;
no neurahash import (runs in the miner's own env). The arithmetic task is byte-identical to the coordinator's
so the held-out set matches for its eval.

Optional --relay-name: after saving the trained delta locally, ALSO PUT it to a content store (default: the
project's PUBLIC fleet relay, a separate instance from the main pool's private corpus store) under that
friendly name, so a coordinator anywhere on the real internet can pull it without a direct connection to this
machine -- this is how Colab (which cannot accept inbound connections) and any other WAN worker publishes its
contribution. Needs content_store_client.py alongside this file. NO TOKEN REQUIRED to start: the default
relay uses a PUBLIC demo token (committed below, same spirit as the main pool's public demo PSK) -- it does
not secure anything and isn't meant to; what keeps a bad/garbage upload from ever reaching the trained model
is the coordinator's OWN held-out accept/reject gate (gather-canonical + the soak's propose_and_gate), which
rejects anything that doesn't measurably help before it's ever merged in. Pass --token to use a different,
private relay instead.
"""
import argparse, hashlib, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import OlmoeForCausalLM, AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924"


def assign_uniform(n_layers, n_experts, n_nodes):
    groups = [[] for _ in range(n_nodes)]
    for L in range(n_layers):
        for E in range(n_experts):
            groups[E * n_nodes // n_experts].append((L, E))
    return groups


def make_arith(n_train, n_eval, maxop, tag="olmoe-esh-v1"):
    ops = ("+", "-", "*"); cands = []
    for a in range(maxop + 1):
        for b in range(maxop + 1):
            for op in ops:
                if op == "-" and a < b:
                    continue
                ans = a + b if op == "+" else (a - b if op == "-" else a * b)
                cands.append(("%d %s %d =" % (a, op, b), ans, op))
    cands.sort(key=lambda pa: hashlib.sha256((tag + ":" + pa[0]).encode()).hexdigest())
    ch = cands[:n_train + n_eval]
    return ch[:n_train], ch[n_train:n_train + n_eval]


# --------------------------------------------------------------- UNFUSED path (transformers 4.x ModuleList)
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dev):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev, dtype=torch.float32) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (self.scale * ((x.float() @ self.A.t()) @ self.B.t())).to(self.base(x).dtype)


# ----------------------------------------------------------------- FUSED path (transformers 5.x 3D Parameter)
class FusedLoRAExperts(nn.Module):
    """Drop-in for OlmoeExperts / Glm4MoeLiteNaiveMoe (identical forward): frozen fused gate_up_proj/down_proj +
    per-owned-expert LoRA, injected into a faithful copy of the expert loop."""
    def __init__(self, base, owned, r, alpha, dev):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.num_experts = base.num_experts
        self.act_fn = base.act_fn
        self.scale = alpha / r
        self.owned = set(int(e) for e in owned)
        H, I = base.hidden_dim, base.intermediate_dim
        self.A_gu, self.B_gu, self.A_d, self.B_d = (nn.ParameterDict() for _ in range(4))
        for e in sorted(self.owned):
            k = str(e)
            self.A_gu[k] = nn.Parameter(torch.randn(r, H, device=dev, dtype=torch.float32) * 0.01)
            self.B_gu[k] = nn.Parameter(torch.zeros(2 * I, r, device=dev, dtype=torch.float32))
            self.A_d[k] = nn.Parameter(torch.randn(r, I, device=dev, dtype=torch.float32) * 0.01)
            self.B_d[k] = nn.Parameter(torch.zeros(H, r, device=dev, dtype=torch.float32))

    def params(self):
        out = []
        for k in self.A_gu:
            out += [self.A_gu[k], self.B_gu[k], self.A_d[k], self.B_d[k]]
        return out

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            em = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
        gu_w, d_w = self.base.gate_up_proj, self.base.down_proj
        for ei in hit:
            e = int(ei[0])
            if e == self.num_experts:
                continue
            pos, tok = torch.where(em[e])
            cs = hidden_states[tok]
            gu = F.linear(cs, gu_w[e])
            if e in self.owned:
                k = str(e)
                gu = gu + (self.scale * F.linear(F.linear(cs.float(), self.A_gu[k]), self.B_gu[k])).to(gu.dtype)
            gate, up = gu.chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            dh = F.linear(h, d_w[e])
            if e in self.owned:
                k = str(e)
                dh = dh + (self.scale * F.linear(F.linear(h.float(), self.A_d[k]), self.B_d[k])).to(dh.dtype)
            dh = dh * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, dh.to(final.dtype))
        return final


def attach(model, own, r, alpha, dev):
    """Auto-detect fused vs unfused and attach LoRA to this worker's experts. Returns (params, unfused_wraps,
    fused_mods, layout)."""
    own = set(own)
    params, unfused, fused = [], {}, {}
    layout = None
    for L, layer in enumerate(model.model.layers):
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is None:
            continue
        if hasattr(experts, "gate_up_proj"):                       # FUSED (tf 5.x)
            layout = "fused"
            owned_here = [E for (LL, E) in own if LL == L]
            if not owned_here:
                continue
            le = FusedLoRAExperts(experts, owned_here, r, alpha, dev)
            layer.mlp.experts = le
            fused[L] = le
            params += le.params()
        else:                                                      # UNFUSED ModuleList (tf 4.x)
            layout = layout or "unfused"
            for E, expert in enumerate(experts):
                if (L, E) not in own:
                    continue
                w = {}
                for name in ("gate_proj", "up_proj", "down_proj"):
                    lo = LoRALinear(getattr(expert, name), r, alpha, dev)
                    setattr(expert, name, lo)
                    w[name] = lo
                    params += [lo.A, lo.B]
                unfused[(L, E)] = w
    return params, unfused, fused, layout


def canonical_export(unfused, fused):
    """Both layouts -> the SAME canonical per-projection LoRA factors {(L,E): {gate:(A,B), up:(A,B), down:(A,B)}}.
    For fused, gate/up share A_gu and split B_gu into the first/second I rows."""
    exp = {}
    for (L, E), w in unfused.items():
        exp["%d,%d" % (L, E)] = {n2: (w[n1].A.detach().half().cpu(), w[n1].B.detach().half().cpu())
                                 for n1, n2 in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down"))}
    for L, le in fused.items():
        I = le.base.intermediate_dim
        for k in le.A_gu:
            E = int(k)
            agu = le.A_gu[k].detach().half().cpu(); bgu = le.B_gu[k].detach().half().cpu()
            exp["%d,%d" % (L, E)] = {
                "gate": (agu, bgu[:I]), "up": (agu, bgu[I:]),
                "down": (le.A_d[k].detach().half().cpu(), le.B_d[k].detach().half().cpu())}
    return exp


def build_batch(tok, mb, dev):
    seqs, labs = [], []
    for prompt, gold, op in mb:
        pids = tok(prompt).input_ids
        fids = tok(prompt + " " + str(gold)).input_ids
        if len(fids) <= len(pids):
            continue
        seqs.append(fids); labs.append([-100] * len(pids) + fids[len(pids):])
    if not seqs:
        return None, None
    T = max(len(s) for s in seqs); pad = tok.pad_token_id
    return (torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device=dev),
            torch.tensor([l + [-100] * (T - len(l)) for l in labs], device=dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--r", type=int, default=8); ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ntrain", type=int, default=300); ap.add_argument("--neval", type=int, default=80)
    ap.add_argument("--maxop", type=int, default=12); ap.add_argument("--mbsz", type=int, default=8)
    ap.add_argument("--load-4bit", action="store_true", dest="load_4bit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relay-name", default=None, dest="relay_name",
                    help="if set, PUT the trained delta to --relay-url under this friendly name (WAN publish)")
    ap.add_argument("--relay-url", default="http://47.84.93.96:8711", dest="relay_url",
                    help="PUBLIC Rung B fleet relay by default (separate from the main pool's private corpus store)")
    ap.add_argument("--token", default="2802648a1e87b4b3c6ca6da2688b4308",
                    help="content-store auth token. The default is a PUBLIC demo token for the public fleet "
                         "relay -- it secures nothing (see the module docstring); pass your own to use a "
                         "private relay instead. NEURAHASH_CONTENT_TOKEN env var overrides this if set.")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    import transformers
    print("[worker] transformers %s | device %s | 4bit=%s | loading %s ..." %
          (transformers.__version__, dev, a.load_4bit, MODEL), flush=True)
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = OlmoeForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0})
    else:
        model = OlmoeForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    nL, nE = model.config.num_hidden_layers, model.config.num_experts
    own = assign_uniform(nL, nE, a.nodes)[a.node]
    params, unfused, fused, layout = attach(model, own, a.r, a.alpha, dev)
    print("[worker] node %d owns %d experts | layout=%s | %d LoRA params | VRAM %.1fGB (%.0fs)" %
          (a.node, len(own), layout, len(params), torch.cuda.memory_allocated() / 1e9 if dev == "cuda" else 0, time.time() - t0), flush=True)

    train, _ = make_arith(a.ntrain, a.neval, a.maxop)
    opt = torch.optim.AdamW(params, lr=a.lr)
    steps = max(1, len(train) // a.mbsz); total = a.epochs * steps; warm = max(1, total // 10); g = 0
    for ep in range(a.epochs):
        model.train(); random.Random(1000 + ep).shuffle(train); tot = 0.0; nb = 0
        for i in range(0, len(train) - a.mbsz + 1, a.mbsz):
            lr = a.lr * (g + 1) / warm if g < warm else a.lr * 0.5 * (1 + math.cos(math.pi * (g - warm) / max(1, total - warm)))
            for gp in opt.param_groups:
                gp["lr"] = lr
            ids, lab = build_batch(tok, train[i:i + a.mbsz], dev)
            if ids is None:
                continue
            out = model(input_ids=ids, labels=lab)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += float(out.loss.detach()); nb += 1; g += 1
        print("[worker] epoch %d loss=%.3f (%.0fs)" % (ep + 1, tot / max(1, nb), time.time() - t0), flush=True)

    exp = canonical_export(unfused, fused)
    torch.save({"global_node": a.node, "format": "canonical_lora", "experts": exp,
                "meta": {"model": MODEL, "nodes": a.nodes, "r": a.r, "alpha": a.alpha,
                         "layout": layout, "transformers": transformers.__version__, "n_experts": len(exp)}}, a.out)
    print("[worker] DONE node %d: %d experts (%s) -> %s (%.1f MB, %.0fs)" %
          (a.node, len(exp), layout, a.out, os.path.getsize(a.out) / 1e6, time.time() - t0), flush=True)

    if a.relay_name:
        import content_store_client as cs
        token = os.environ.get("NEURAHASH_CONTENT_TOKEN") or a.token
        data = open(a.out, "rb").read()
        sha = cs.put(a.relay_url, token, data, name=a.relay_name)
        print("[worker] relayed -> %s as '%s' (sha256 %s)" % (a.relay_url, a.relay_name, sha[:16]), flush=True)


if __name__ == "__main__":
    main()
