#!/usr/bin/env python3
"""shardDiLoCo Phase-2b WAN run-harness -- SHARED library (numpy + stdlib ONLY, NO torch).

This module is the common ground the runnable coordinator-role process
(tools/sharddiloco_coordinator.py) and the runnable contributor CLI
(tools/sharddiloco_contributor.py) both stand on. It carries:

  * HarnessConfig            -- the small MoE + DiLoCo cadence config (env-overridable), so every
                               node builds a BIT-IDENTICAL model + domain shard.
  * build_model / domains    -- the toy real MoELM (neurahash.model) + the deterministic per-domain
                               Markov shards (D4 OFFLINE routing: expert e trains only on domain e).
                               Faithful port of tests/test_sharddiloco_phase2.py's data so the
                               runnable path reproduces the PROVEN result on the SAME distribution.
  * the D1 WIRE + LANE       -- fp16 content-addressed (sha256 CID) delta/state serialization + a
                               ContentLane HTTP client for tools/content_store.py (the live VPS
                               anchor lane, http://47.84.93.96:8710). ALL-OUTBOUND: a contributor
                               only ever GETs the pointer/state and PUTs its delta+record; it never
                               listens. The exact transport a real 4060/RunPod miner uses over WAN.
  * sign / verify            -- the GAP1 signed-identity stand-in (HMAC-SHA256 over the REAL
                               diloco_merge.contrib_canonical_message), identical to
                               tools/diloco_contributor.ShardDeltaLane.sign, so the wire format
                               cannot drift from the phase-2 lane.
  * sync_sparse_baseline     -- the EQUAL-COMPUTE synchronous-MoE control arm, so the coordinator can
                               print the held-out NON-REGRESSION ratio (the goal metric) in the
                               runnable process, not only in the unit test.

WHY torch-free: the coordinator role only needs numpy (neurahash.model / diloco_merge /
training_layer are all numpy-only), so it can run on a lightweight always-on box. The contributor
DOES import tools/diloco_contributor (torch) for the real train_expert_contribution kernel.

Default-off: nothing here reads or trains unless a caller runs it; the coordinator refuses to run
unless NEURAHASH_SHARDDILOCO is set (see tools/sharddiloco_coordinator.py). This module imports no
live-pool code and touches no _poollive/ / _state_* state.
"""
import hashlib
import hmac
import http.client
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np

# transient network faults a WAN (or a busy localhost content_store under burst load) throws and a
# real all-outbound client MUST ride out: connection resets/drops, timeouts, incomplete reads.
_TRANSIENT = (urllib.error.URLError, http.client.HTTPException, OSError)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neurahash.model import MoELM, expert_keys, make_examples, VOCAB, V  # noqa: E402
from neurahash.diloco_merge import contrib_canonical_message              # noqa: E402

# TRUNK = the replicated small shared params (matches tools/diloco_contributor.SHARD_TRUNK_KEYS and
# neurahash.model.forward_offline). Router (Wr, br) is UNUSED under offline routing.
TRUNK = ["Emb", "Wo", "bo"]
POINTER_NAME = "sharddiloco/pointer"           # named object holding the canonical-state pointer JSON


# ============================================================ config
class HarnessConfig:
    """Small MoE + DiLoCo-cadence config. Defaults match tests/test_sharddiloco_phase2._cfg() so the
    runnable harness reproduces the PROVEN non-regression on the same synthetic distribution. Every
    field is env-overridable (NEURAHASH_SD_*) so 5090/4060/RunPod agree by sharing one env."""

    FIELDS = dict(E=3, C=4, Demb=16, H=32, Do=24, rounds=16, H_inner=25, B=64, lr=2e-3,
                  margin=1e-4, probe_size=256, n_chars=40000, seed=0, outer=0.7)

    def __init__(self, **over):
        for k, d in self.FIELDS.items():
            v = os.environ.get("NEURAHASH_SD_" + k.upper())
            if k in over and over[k] is not None:
                v = over[k]
            if v is None:
                v = d
            self.__dict__[k] = type(d)(v)

    def to_json(self):
        return {k: self.__dict__[k] for k in self.FIELDS}

    @classmethod
    def from_json(cls, d):
        return cls(**{k: d[k] for k in cls.FIELDS if k in d})

    def __repr__(self):
        return "HarnessConfig(%s)" % ", ".join("%s=%s" % (k, self.__dict__[k]) for k in self.FIELDS)


def build_model(cfg):
    """The toy REAL MoELM (pure-numpy, manual backprop). Same construction on every node so dims +
    the frozen router match. Offline routing means the OTHER experts + router are never read, so only
    the dims have to agree; the canonical trunk + expert e are overlaid from the fetched state."""
    return MoELM(C=cfg.C, Demb=cfg.Demb, H=cfg.H, Do=cfg.Do, n_experts=cfg.E, seed=0)


def fwd_flops(model):
    """Forward FLOPs for ONE example on MoELM's sparse offline path -- feeds diloco_merge.FlopMeter
    (D3). BYTE-IDENTICAL formula to tools/diloco_contributor.moelm_sparse_fwd_flops (kept here so the
    coordinator stays torch-free); parity guarded by tests, do not let the two drift."""
    return 2.0 * (model.Din * model.H + model.H * model.Do + model.Do * V)


# ============================================================ deterministic per-domain data (D4)
def _domain_text(e, n_chars, seed=0):
    """Domain e = a first-order Markov chain over a DISTINCT favored-char subset (verbatim port of
    tests/test_sharddiloco_phase2._domain_text) -> next-char is learnable AND domain-specific, so
    experts genuinely specialize and offline routing is meaningful (D4)."""
    rng = np.random.default_rng(1000 + e * 7 + seed)
    alpha = [c for c in VOCAB if c != "<unk>"]
    k = len(alpha)
    fav = [(e * 5 + i) % k for i in range(8)]
    seq, cur = [], fav[0]
    for _ in range(n_chars):
        cur = fav[(cur + 1) % len(fav)] if (rng.random() < 0.75 and cur in fav) else int(rng.integers(0, k))
        seq.append(alpha[cur])
    return "".join(seq)


def domain_splits(cfg, e):
    """Per-domain DISJOINT splits for expert e (verbatim port of _build_domains): train /
    secret-probe-pool (coordinator-only) / public-probe / heldout (reported goal metric). A miner
    only ever holds `train` for its own e; the coordinator holds probe + heldout for all e."""
    X, y = make_examples(_domain_text(e, cfg.n_chars, cfg.seed), cfg.C)
    idx = np.random.default_rng(cfg.seed).permutation(len(X))
    X, y = X[idx], y[idx]
    return dict(train=(X[:8000], y[:8000]),
                probe=(X[8000:10000], y[8000:10000]),        # SECRET (coordinator-only) pool
                public=(X[10000:10008], y[10000:10008]),     # tiny PUBLIC probe (miner sees it)
                heldout=(X[10008:12008], y[10008:12008]))     # reported held-out goal metric


# ============================================================ D1 wire: fp16 content-addressed blobs
def pack_arrays(named, wire_dtype):
    """Deterministic wire for a {key: ndarray} dict: a sorted JSON header (dtype + per-key shape) then
    the raw little-endian array bytes. fp16 for pseudo-gradients (D1 review: BORROW OpenDiLoCo's FP16
    transfer), float64 for canonical state. No timestamps/zip -> identical bytes for identical content
    -> a real content address (unlike np.savez)."""
    keys = sorted(named)
    dt = np.dtype(wire_dtype)
    header = {"dtype": dt.str, "keys": [{"k": k, "shape": [int(s) for s in named[k].shape]} for k in keys]}
    hj = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    parts = [struct.pack("<I", len(hj)), hj]
    for k in keys:
        parts.append(np.ascontiguousarray(named[k], dtype=dt).tobytes())
    return b"".join(parts)


def unpack_arrays(body):
    """Inverse of pack_arrays -> {key: float64 ndarray}. fp16 payloads are widened to float64 (the
    fp16 round-trip, exactly like tools/diloco_contributor._shard_fp16_roundtrip)."""
    (hlen,) = struct.unpack("<I", body[:4])
    header = json.loads(body[4:4 + hlen].decode())
    dt = np.dtype(header["dtype"])
    off = 4 + hlen
    out = {}
    for spec in header["keys"]:
        shape = tuple(spec["shape"])
        n = int(np.prod(shape)) if shape else 1
        nbytes = n * dt.itemsize
        arr = np.frombuffer(body[off:off + nbytes], dtype=dt).reshape(shape).astype(np.float64)
        out[spec["k"]] = np.array(arr)      # own the memory (frombuffer is read-only)
        off += nbytes
    return out


def cid_of(body):
    return hashlib.sha256(body).hexdigest()


def pack_state(rnd, cfg, trunk, experts, done=False):
    """Serialize the whole canonical state (round + cfg + trunk + per-expert params) to one blob. Full
    float64 precision -- the canonical weights are the product, not a lossy delta."""
    flat = {("trunk." + k): trunk[k] for k in trunk}
    for e, ed in enumerate(experts):
        for k in ed:
            flat["e%d.%s" % (e, k)] = ed[k]
    arrbody = pack_arrays(flat, np.float64)
    meta = json.dumps(dict(round=int(rnd), done=bool(done), n_experts=len(experts), cfg=cfg.to_json()),
                      sort_keys=True, separators=(",", ":")).encode()
    return struct.pack("<I", len(meta)) + meta + arrbody


def unpack_state(body):
    (mlen,) = struct.unpack("<I", body[:4])
    meta = json.loads(body[4:4 + mlen].decode())
    flat = unpack_arrays(body[4 + mlen:])
    trunk = {k: flat["trunk." + k] for k in TRUNK}
    experts = []
    for e in range(meta["n_experts"]):
        pref = "e%d." % e
        experts.append({k[len(pref):]: v for k, v in flat.items() if k.startswith(pref)})
    return meta, trunk, experts


# ============================================================ D2 signed identity (HMAC stand-in)
def sign(secret_key, cid, base_round, name):
    """HMAC-SHA256 over the REAL canonical GAP1 message (diloco_merge.contrib_canonical_message) --
    identical to tools/diloco_contributor.ShardDeltaLane.sign, so the format cannot drift. Live path
    swaps this for the secp256k1 record signature."""
    msg = contrib_canonical_message(cid, base_round, name, None, None)
    return hmac.new(secret_key, msg, hashlib.sha256).hexdigest()


def verify(secret_key, sig, cid, base_round, name):
    return hmac.compare_digest(sig, sign(secret_key, cid, base_round, name))


# ============================================================ ContentLane -- all-outbound HTTP client
class ContentLane:
    """Client for tools/content_store.py (the live VPS anchor lane). Content is fetched/stored BY
    sha256, so a node gets EXACTLY the committed bytes or a 404 (D1). Everything here is OUTBOUND from
    the caller: PUT the delta+record, GET the pointer/state. The only listener is the store (VPS)."""

    def __init__(self, base_url, token="", timeout=30, retries=6, backoff=0.25):
        self.base = base_url.rstrip("/")
        self.token = token or ""
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _request(self, req):
        """One urlopen with retry-on-transient-fault -- the resilience a WAN client needs. A 4xx (e.g.
        a real sha256 mismatch) is raised immediately; only 5xx and connection-level faults retry."""
        last = None
        for i in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if not (500 <= e.code < 600) or i >= self.retries - 1:
                    raise
                last = e
            except _TRANSIENT as e:
                if i >= self.retries - 1:
                    raise
                last = e
            time.sleep(self.backoff * (i + 1))
        if last:
            raise last

    def _get(self, path):
        return self._request(self.base + path)

    def health(self):
        return json.loads(self._get("/health").decode())

    def manifest(self):
        return json.loads(self._get("/manifest").decode())

    def get_blob(self, cid):
        body = self._get("/o/" + cid)
        if cid_of(body) != cid:
            raise ValueError("CID mismatch from store: content tampered in transit")
        return body

    def put_blob(self, body, name=None):
        cid = cid_of(body)
        req = urllib.request.Request(self.base + "/o/" + cid, data=body, method="PUT")
        if self.token:
            req.add_header("X-Auth", self.token)
        if name:
            req.add_header("X-Name", name)
        json.loads(self._request(req).decode())   # 201 body; raises on non-2xx
        return cid

    # ---- typed helpers ----
    def put_delta(self, delta, name=None):
        return self.put_blob(pack_arrays(delta, np.float16), name=name)

    def get_delta(self, cid):
        return unpack_arrays(self.get_blob(cid))

    def put_state(self, rnd, cfg, trunk, experts, done=False):
        return self.put_blob(pack_state(rnd, cfg, trunk, experts, done=done))

    def get_state(self, cid):
        return unpack_state(self.get_blob(cid))

    def put_json_named(self, name, obj):
        return self.put_blob(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode(), name=name)

    def get_json(self, cid):
        return json.loads(self.get_blob(cid).decode())

    # ---- pointer (canonical-state advertisement) ----
    def publish_pointer(self, rnd, state_cid, done=False):
        return self.put_json_named(POINTER_NAME, dict(round=int(rnd), state_cid=state_cid, done=bool(done)))

    def read_pointer(self):
        man = self.manifest()
        if POINTER_NAME not in man:
            return None
        return self.get_json(man[POINTER_NAME]["sha256"])


def contrib_name(rnd, miner):
    return "c/r%d/%s" % (int(rnd), miner)


def contrib_prefix(rnd):
    return "c/r%d/" % int(rnd)


# ============================================================ equal-compute synchronous baseline
class _HarnessAdamW:
    """Minimal numpy AdamW -- same math as tools/diloco_contributor._ShardAdamW, replicated here so
    the coordinator's baseline stays torch-free. Used ONLY for sync_sparse_baseline."""

    def __init__(self, keys, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = {k: None for k in keys}
        self.v = {k: None for k in keys}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            if self.m[k] is None:
                self.m[k] = np.zeros_like(g)
                self.v[k] = np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            if self.wd:
                params[k] -= self.lr * self.wd * params[k]
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def eval_heldout(model, template, trunk, experts, domains):
    """Composed-model held-out CE, OFFLINE-routed (each domain's heldout -> its expert). The GOAL
    METRIC. Same eval for both arms == fair. Verbatim port of the test's _eval_heldout."""
    tot, n = 0.0, 0
    for e, dom in enumerate(domains):
        Xh, yh = dom["heldout"]
        p = dict(template)
        p.update(trunk)
        for ed in experts:
            p.update(ed)
        ea = np.full(len(Xh), e, dtype=np.int64)
        loss, _ = model.forward_offline(Xh, yh, ea, params=p)
        tot += loss * len(Xh)
        n += len(Xh)
    return tot / n


def sync_sparse_baseline(cfg, domains, model):
    """ARM A -- SYNC-SPARSE control: joint synchronous AdamW over {trunk + ALL experts}, SAME hard
    offline routing, EQUAL per-example step budget (E*rounds*H_inner). Isolates the one variable:
    synchronous joint vs decoupled per-expert DiLoCo. Verbatim port of the test's run_sync_sparse.
    Returns final held-out CE."""
    E = cfg.E
    p = model.clone_params()
    keys = list(TRUNK)
    for e in range(E):
        keys += expert_keys(e)
    opt = _HarnessAdamW(keys, lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    steps = E * cfg.rounds * cfg.H_inner
    per = max(1, cfg.B // E)
    for _ in range(steps):
        Xs, ys, ea = [], [], []
        for e in range(E):
            Xt, yt = domains[e]["train"]
            idx = rng.integers(0, len(Xt), size=per)
            Xs.append(Xt[idx]); ys.append(yt[idx]); ea.append(np.full(per, e, dtype=np.int64))
        X = np.concatenate(Xs); y = np.concatenate(ys); e_assign = np.concatenate(ea)
        loss, cache = model.forward_offline(X, y, e_assign, params=p)
        grads = model.backward_offline(cache, y, keys, params=p)
        opt.step(p, grads)
    strunk = {k: p[k] for k in TRUNK}
    sexperts = [{k: p[k] for k in expert_keys(e)} for e in range(E)]
    return eval_heldout(model, p, strunk, sexperts, domains)
