"""
The actual model: a character-level Mixture-of-Experts language model.

Pure NumPy, manual forward + backprop, so there is NO magic: you can read every
gradient. It really trains and the loss really goes down.

Why MoE? Because it is the architecture that lets the network model "grow bigger
and bigger" over time: each round we can append a new expert sub-network. Total
capacity climbs, but inference stays cheap because the soft router only leans on
a few experts per token.

Params are stored as a flat dict of named arrays. That makes three things trivial:
  - DiLoCo delta computation (subtract two param dicts)
  - deterministic hashing for verification (hash the bytes)
  - growth (add new keys for a new expert)
"""

import copy
import hashlib
import numpy as np

# Fixed vocabulary. Fixed up front so model dimensions never change even as brand
# new text shards arrive. Index 0 is <unk> for any char we did not anticipate.
VOCAB = ["<unk>"] + list("abcdefghijklmnopqrstuvwxyz0123456789 ,.\n'-")
STOI = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)


# Parameter groups, so a node can train/host the WHOLE model or just one expert.
SHARED_KEYS = ["Emb", "Wr", "br", "Wo", "bo"]


def expert_keys(e):
    return [f"W1_{e}", f"b1_{e}", f"W2_{e}", f"b2_{e}"]


def default_expert_seed(e):
    """The PROTOCOL-FIXED rng seed for the e-th expert's default initialization. This is the one
    source of truth shared by MoELM.add_expert (the model's own growth) and
    expert_sharding.grow_experts (the on-chain growth migration F(P_old)); both MUST derive a new
    expert's params from the IDENTICAL seed or a growth-boundary block computed by one path is
    rejected by the other (chain stall). Keep the two callers bound to THIS function so they can
    never silently drift."""
    return 1000 + int(e)


def encode(text):
    return [STOI.get(c, 0) for c in text]


def make_examples(text, C):
    """Slide a context window of length C over text -> (X, y) next-char prediction."""
    ids = encode(text)
    if len(ids) <= C:
        return np.zeros((0, C), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    X, y = [], []
    for i in range(len(ids) - C):
        X.append(ids[i:i + C])
        y.append(ids[i + C])
    return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class MoELM:
    def __init__(self, C=3, Demb=8, H=24, Do=16, n_experts=2, seed=0):
        self.C, self.Demb, self.H, self.Do = C, Demb, H, Do
        self.Din = C * Demb
        self.E = n_experts
        rng = np.random.default_rng(seed)
        self.p = {}
        s = 0.3
        self.p["Emb"] = rng.normal(0, s, (V, Demb))
        self.p["Wr"] = rng.normal(0, s, (self.Din, n_experts))
        self.p["br"] = np.zeros(n_experts)
        for e in range(n_experts):
            self._init_expert(e, rng, s)
        self.p["Wo"] = rng.normal(0, s, (Do, V))
        self.p["bo"] = np.zeros(V)

    def _init_expert(self, e, rng, s=0.3):
        self.p[f"W1_{e}"] = rng.normal(0, s, (self.Din, self.H))
        self.p[f"b1_{e}"] = np.zeros(self.H)
        self.p[f"W2_{e}"] = rng.normal(0, s, (self.H, self.Do))
        self.p[f"b2_{e}"] = np.zeros(self.Do)

    # ---- growth: this is how the model "gets bigger" as contributors join ----
    def add_expert(self, seed=None):
        rng = np.random.default_rng(seed if seed is not None else default_expert_seed(self.E))
        e = self.E
        self._init_expert(e, rng, s=0.2)
        # grow the router by one column (new expert starts with ~zero gate weight)
        new_col = rng.normal(0, 0.05, (self.Din, 1))
        self.p["Wr"] = np.concatenate([self.p["Wr"], new_col], axis=1)
        self.p["br"] = np.concatenate([self.p["br"], np.zeros(1)])
        self.E += 1
        return e

    def num_params(self):
        return int(sum(a.size for a in self.p.values()))

    def all_keys(self):
        return list(self.p.keys())

    def shared_param_count(self):
        return int(sum(self.p[k].size for k in SHARED_KEYS if k in self.p))

    def expert_param_count(self, e=0):
        return int(sum(self.p[k].size for k in expert_keys(e) if k in self.p))

    # ---------------------------- forward ----------------------------
    def forward(self, X, y=None, params=None):
        p = self.p if params is None else params
        n = X.shape[0]
        emb = p["Emb"][X]                      # (n, C, Demb)
        x0 = emb.reshape(n, self.Din)          # (n, Din)

        gate_logits = x0 @ p["Wr"] + p["br"]   # (n, E)
        g = _softmax(gate_logits, axis=1)      # (n, E)

        o = np.zeros((n, self.Do))
        cache_exp = []
        for e in range(self.E):
            pre = x0 @ p[f"W1_{e}"] + p[f"b1_{e}"]   # (n, H)
            h = np.maximum(pre, 0.0)
            oe = h @ p[f"W2_{e}"] + p[f"b2_{e}"]     # (n, Do)
            o += g[:, e:e + 1] * oe
            cache_exp.append((pre, h, oe))

        logits = o @ p["Wo"] + p["bo"]          # (n, V)
        probs = _softmax(logits, axis=1)
        cache = dict(X=X, x0=x0, g=g, o=o, probs=probs, cache_exp=cache_exp)
        if y is None:
            return probs, cache
        loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
        return loss, cache

    def loss(self, X, y, params=None):
        if X.shape[0] == 0:
            return 0.0
        l, _ = self.forward(X, y, params=params)
        return float(l)

    # ===================== sparse single-expert path =====================
    # Top-1 (hard) routing so a block can train/verify ONE expert in isolation: the loss
    # for an example routed to expert e depends ONLY on {Emb, Wr (router), Wo, bo (shared)}
    # + that one expert's FFN. A verifier can therefore recompute the step holding only the
    # shared params + the assigned expert — NOT the other E-1 experts — so per-node cost stays
    # bounded as the fleet (and the expert count) grows. This is the MoE-sharding scaling unlock.
    def route_top1(self, X, params=None):
        """Hard top-1 routing: the argmax expert per example. Deterministic; depends only on
        the (frozen, shared) router + embedding, so proposer and verifier route identically."""
        p = self.p if params is None else params
        n = X.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.int64)
        x0 = p["Emb"][X].reshape(n, self.Din)
        gate_logits = x0 @ p["Wr"] + p["br"]
        return np.argmax(gate_logits, axis=1).astype(np.int64)

    def forward_expert(self, X, y, e, params=None):
        """Sparse forward for expert `e`: route X, keep the examples whose top-1 expert is e,
        and push ONLY those through expert e (output = gate_e * expert_e(x) -> shared Wo/bo).
        Returns (loss, cache). References only shared keys + expert e — never another expert.
        loss is the mean CE over the routed-to-e examples (0.0 if none route here)."""
        p = self.p if params is None else params
        n = X.shape[0]
        x0 = p["Emb"][X].reshape(n, self.Din) if n else np.zeros((0, self.Din))
        gate_logits = x0 @ p["Wr"] + p["br"] if n else np.zeros((0, self.E))
        g = _softmax(gate_logits, axis=1) if n else gate_logits
        route = np.argmax(gate_logits, axis=1) if n else np.zeros((0,), dtype=np.int64)
        mask = (route == e)
        Xe, ye = X[mask], (y[mask] if y is not None else None)
        x0e = x0[mask]
        ge = g[mask, e:e + 1]                       # (ne,1) frozen gate weight for e
        pre = x0e @ p[f"W1_{e}"] + p[f"b1_{e}"]     # (ne,H)
        h = np.maximum(pre, 0.0)
        oe = h @ p[f"W2_{e}"] + p[f"b2_{e}"]        # (ne,Do)
        o = ge * oe                                 # (ne,Do) gated single-expert output
        logits = o @ p["Wo"] + p["bo"]             # (ne,V)
        probs = _softmax(logits, axis=1)
        cache = dict(X=Xe, x0e=x0e, ge=ge, pre=pre, h=h, oe=oe, o=o, probs=probs,
                     mask=mask, e=e, n_routed=int(mask.sum()))
        if ye is None:
            return probs, cache
        ne = Xe.shape[0]
        loss = 0.0 if ne == 0 else -np.mean(np.log(probs[np.arange(ne), ye] + 1e-12))
        return float(loss), cache

    def expert_loss(self, X, y, e, params=None):
        """Held-out single-expert loss (mean CE over the val examples routed to e). The
        expert-sharded work score is the improvement in THIS quantity — computable by a
        verifier holding only {shared, expert e}, matching the sharded verification cost."""
        if X.shape[0] == 0:
            return 0.0
        l, _ = self.forward_expert(X, y, e, params=params)
        return float(l)

    def backward_expert(self, cache, y, e, params=None):
        """Gradients for ONLY expert e's keys (W1_e,b1_e,W2_e,b2_e) from a forward_expert
        cache; shared params (Emb/Wr/Wo/bo) are FROZEN during sharded training so no grads are
        produced for them. The gate weight ge is treated as a constant (router frozen)."""
        p = self.p if params is None else params
        ye = y[cache["mask"]] if y is not None else None
        x0e, ge, pre, h, oe, probs = (cache["x0e"], cache["ge"], cache["pre"],
                                      cache["h"], cache["oe"], cache["probs"])
        ne = x0e.shape[0]
        grads = {k: np.zeros_like(p[k]) for k in expert_keys(e)}
        if ne == 0:
            return grads
        dlogits = probs.copy()
        dlogits[np.arange(ne), ye] -= 1.0
        dlogits /= ne
        do = dlogits @ p["Wo"].T                    # (ne,Do); Wo frozen, used only to backprop
        doe = do * ge                               # scale by the frozen gate weight
        grads[f"W2_{e}"] = h.T @ doe
        grads[f"b2_{e}"] = doe.sum(0)
        dh = doe @ p[f"W2_{e}"].T
        dpre = dh * (pre > 0)
        grads[f"W1_{e}"] = x0e.T @ dpre
        grads[f"b1_{e}"] = dpre.sum(0)
        return grads

    # ================= offline (D4) hard-routed path: trains the TRUNK too =================
    # shardDiLoCo (docs/research/SHARDDILOCO_DESIGN.md sec 12) uses OFFLINE routing: each example is
    # routed to its EXTERNALLY-ASSIGNED expert (a domain/cluster pre-shard, DiPaCo / Branch-Train-MiX),
    # NOT the learned router. Unlike forward_expert (which FREEZES the shared trunk), this path TRAINS
    # {Emb, Wo, bo}, so a contributor can locally improve {trunk + its own expert} for H DiLoCo inner
    # steps. Nothing on the SYNCHRONOUS pool path calls these; they are the per-expert DiLoCo local-
    # training kernel. Gate weight is hard (=1): o = expert_{e_assign[i]}(x0_i). Gradient-checked
    # against finite differences in tests/test_sharddiloco_phase2.py.
    def forward_offline(self, X, y, e_assign, params=None):
        """OFFLINE-ROUTED forward. e_assign:(n,) the assigned expert index per example. Returns
        (loss, cache); pass y=None to get (probs, cache) only. Router (Wr, br) is UNUSED here."""
        p = self.p if params is None else params
        n = X.shape[0]
        x0 = p["Emb"][X].reshape(n, self.Din)
        o = np.zeros((n, self.Do))
        per_e = {}
        for e in np.unique(e_assign):
            e = int(e)
            m = (e_assign == e)
            xe = x0[m]
            pre = xe @ p[f"W1_{e}"] + p[f"b1_{e}"]
            h = np.maximum(pre, 0.0)
            o[m] = h @ p[f"W2_{e}"] + p[f"b2_{e}"]
            per_e[e] = (m, xe, pre, h)
        logits = o @ p["Wo"] + p["bo"]
        probs = _softmax(logits, axis=1)
        cache = dict(X=X, x0=x0, o=o, probs=probs, per_e=per_e)
        if y is None:
            return probs, cache
        loss = float(-np.mean(np.log(probs[np.arange(n), y] + 1e-12)))
        return loss, cache

    def backward_offline(self, cache, y, train_keys, params=None):
        """Gradients for the requested train_keys under offline routing: any of the trunk subset
        {Emb, Wo, bo} and/or expert keys {W1_e,b1_e,W2_e,b2_e}. Pairs with forward_offline. The router
        (Wr, br) is never touched (unused under offline routing)."""
        p = self.p if params is None else params
        X, x0, o, probs, per_e = cache["X"], cache["x0"], cache["o"], cache["probs"], cache["per_e"]
        n = X.shape[0]
        tk = set(train_keys)
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        grads = {}
        if "Wo" in tk:
            grads["Wo"] = o.T @ dlogits
        if "bo" in tk:
            grads["bo"] = dlogits.sum(0)
        do = dlogits @ p["Wo"].T                       # (n, Do)
        dx0 = np.zeros_like(x0)
        for e, (m, xe, pre, h) in per_e.items():
            doe = do[m]                                # (ne, Do)
            if f"W2_{e}" in tk:
                grads[f"W2_{e}"] = h.T @ doe
            if f"b2_{e}" in tk:
                grads[f"b2_{e}"] = doe.sum(0)
            dh = doe @ p[f"W2_{e}"].T
            dpre = dh * (pre > 0)
            if f"W1_{e}" in tk:
                grads[f"W1_{e}"] = xe.T @ dpre
            if f"b1_{e}" in tk:
                grads[f"b1_{e}"] = dpre.sum(0)
            dx0[m] = dpre @ p[f"W1_{e}"].T
        if "Emb" in tk:
            demb = dx0.reshape(n, self.C, self.Demb)
            gEmb = np.zeros_like(p["Emb"])
            np.add.at(gEmb, X, demb)
            grads["Emb"] = gEmb
        return grads

    # ---------------------------- backward ----------------------------
    def backward(self, cache, y, params=None):
        p = self.p if params is None else params
        X, x0, g, o, probs = cache["X"], cache["x0"], cache["g"], cache["o"], cache["probs"]
        n = X.shape[0]
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        grads["Wo"] = o.T @ dlogits
        grads["bo"] = dlogits.sum(0)
        do = dlogits @ p["Wo"].T                # (n, Do)

        dx0 = np.zeros_like(x0)
        dg = np.zeros_like(g)
        for e in range(self.E):
            pre, h, oe = cache["cache_exp"][e]
            doe = do * g[:, e:e + 1]            # (n, Do)
            dg[:, e] = (do * oe).sum(1)         # gate gradient
            grads[f"W2_{e}"] = h.T @ doe
            grads[f"b2_{e}"] = doe.sum(0)
            dh = doe @ p[f"W2_{e}"].T           # (n, H)
            dpre = dh * (pre > 0)
            grads[f"W1_{e}"] = x0.T @ dpre
            grads[f"b1_{e}"] = dpre.sum(0)
            dx0 += dpre @ p[f"W1_{e}"].T

        # softmax-gate backprop
        dgate = g * (dg - (dg * g).sum(1, keepdims=True))
        grads["Wr"] = x0.T @ dgate
        grads["br"] = dgate.sum(0)
        dx0 += dgate @ p["Wr"].T

        # embedding backprop (scatter-add)
        demb = dx0.reshape(n, self.C, self.Demb)
        gEmb = np.zeros_like(p["Emb"])
        np.add.at(gEmb, X, demb)
        grads["Emb"] = gEmb
        return grads

    # ---------------------------- utils ----------------------------
    def clone_params(self):
        return copy.deepcopy(self.p)

    def set_params(self, params):
        self.p = copy.deepcopy(params)

    def predict_next(self, context_str, params=None):
        ids = encode(context_str)[-self.C:]
        if len(ids) < self.C:
            ids = [0] * (self.C - len(ids)) + ids
        X = np.array([ids], dtype=np.int64)
        probs, _ = self.forward(X, params=params)
        return probs[0]

    def generate(self, prompt, n=40, params=None, temp=0.8, seed=0):
        rng = np.random.default_rng(seed)
        out = prompt
        for _ in range(n):
            probs = self.predict_next(out, params=params)
            probs = probs ** (1.0 / temp)
            probs /= probs.sum()
            idx = rng.choice(len(probs), p=probs)
            out += VOCAB[idx] if VOCAB[idx] != "<unk>" else ""
        return out


def sgd_step(params, grads, lr):
    for k in params:
        params[k] -= lr * grads[k]


def param_distance(pa, pb):
    """L2 distance between two param dicts (only over shared keys)."""
    tot = 0.0
    for k in pa:
        if k in pb and pa[k].shape == pb[k].shape:
            tot += float(((pa[k] - pb[k]) ** 2).sum())
    return tot ** 0.5


def hash_params(params):
    """Deterministic SHA-256 over the param bytes. Identical computation -> identical hash.
    This is the backbone of the recompute verification challenge."""
    h = hashlib.sha256()
    for k in sorted(params.keys()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(params[k], dtype=np.float64).tobytes())
    return h.hexdigest()
