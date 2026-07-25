#!/usr/bin/env python3
"""FLEET-HOSTED PIPELINE for GLM rollouts -- the model no single card holds, run live by miners.

WHY: G1 rollouts need the FULL 47-layer GLM forward (a truncated stack measured reward 0.0), but
the model is 59 GiB bf16 -- no consumer card. The CE lane already proved miners can each hold ~5
GiB of the model for TRAINING; this module makes the same economics work for GENERATION: the fleet
holds ONE live model TOGETHER, split by layer ranges (a stage = embed+layers, ~1.1 GiB/layer), and
only the per-token hidden-state vector (hidden_size*2 bytes = 4 KB) crosses machines.

TOPOLOGY (all-outbound, NAT-safe -- same rule as everything in this lane):
    worker(driver: tokenizer + final norm + lm_head + sampling)          <- the PAID entity samples
      -> stage0 (embed + layers[0..a))   -> stage1 (layers[a..b)) -> ... -> worker
    Stages NEVER talk to each other directly: every hop is a content-store PUT + advert, and the
    next stage polls for its name. Per token that is (n_stages+1) store round-trips -- slow for one
    stream, but MANY samples ride the pipeline concurrently, so throughput scales with fleet size.

PROTOCOL (advert names under sharddiloco/glm/pipe/):
    <prefix>/<run>/roster                      driver-published stage plan (json)
    <prefix>/<run>/<stream>/p/in<k>            prefill activation entering stage k (k=0: token ids)
    <prefix>/<run>/<stream>/t<t>/in<k>         decode-step activation entering stage k (k=0: token id)
    ...the message LEAVING the last stage (in<n_stages>) is what the worker consumes.
    <prefix>/<run>/<stream>/done               driver signal: stream finished -> stages evict cache

FORWARD FIDELITY: stage_step() mirrors Glm4MoeLiteModel.forward line-for-line (transformers 5.8.1):
create_causal_mask(...), rotary_emb(hidden, position_ids), DynamicCache(config=cfg), layers called
with position_embeddings+position_ids+past_key_values. Each stage keeps its OWN DynamicCache per
stream; only boundary activations cross the wire (prefill: seq*4KB once; decode: 4 KB/token).

TESTABILITY: the codec, bus, and driver loop are torch-free-importable and unit-tested with fake
lanes/stages; torch/transformers load lazily inside the model-touching functions only.
Env: Windows, C:/Python313/python.exe (NEVER .venv). ASCII stdout (cp1252 console).
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PIPE_PREFIX = "sharddiloco/glm/pipe"


# ================================================================ activation codec (torch-lazy)
def act_encode(t):
    """tensor -> bytes: one json header line + raw little-endian payload. bf16 rides as uint16
    (numpy has no bfloat16); shape/dtype restored exactly on decode."""
    import torch
    t = t.detach().contiguous().cpu()
    dt = str(t.dtype).replace("torch.", "")
    raw = (t.view(torch.uint16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
    head = json.dumps({"shape": list(t.shape), "dtype": dt}).encode("utf-8")
    return head + b"\n" + raw


def act_decode(b, device="cpu"):
    """bytes (act_encode) -> tensor on `device`."""
    import numpy as np
    import torch
    nl = b.index(b"\n")
    head = json.loads(b[:nl].decode("utf-8"))
    shape, dt = head["shape"], head["dtype"]
    if dt == "bfloat16":
        arr = np.frombuffer(b[nl + 1:], dtype=np.uint16).copy()
        t = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
    else:
        t = torch.from_numpy(np.frombuffer(b[nl + 1:], dtype=np.dtype(dt)).copy()).reshape(shape)
    return t.to(device)


# ================================================================ store bus (works on any lane
# object with put_blob(body, name=)->cid, get_blob(cid)->bytes, manifest()->{"names":{name:cid}})
class PipeBus:
    """Content-store message bus: send = PUT+advert, recv = poll the advert map then GET by cid.
    Fail-closed: recv verifies nothing beyond the store's own by-sha addressing (the store IS
    content-addressed, so a cid fetch returns the committed bytes or 404)."""

    # Per-hop latency is poll-bound, not compute-bound: every token pays (n_stages+1) advert polls.
    # NEURAHASH_PIPE_POLL_S tunes the floor and MAX_BACKOFF the ceiling -- lower = faster tokens at
    # the cost of more manifest GETs against the store.
    MAX_BACKOFF_S = float(os.environ.get("NEURAHASH_PIPE_POLL_MAX_S", "0.75"))

    def __init__(self, lane, prefix=PIPE_PREFIX, poll_s=None, log=None):
        self.lane, self.prefix = lane, prefix
        self.poll_s = float(os.environ.get("NEURAHASH_PIPE_POLL_S", "0.15")) \
            if poll_s is None else poll_s
        self.log = log or (lambda m: None)

    @staticmethod
    def _cid(v):
        """Manifest values are {'sha256': cid, 'size': n} on the live store; tolerate plain cids."""
        return v.get("sha256") if isinstance(v, dict) else v

    def name(self, run, stream, seg, hop):
        return "%s/%s/%s/%s/in%d" % (self.prefix, run, stream, seg, hop)

    def send(self, name, data):
        body = data if isinstance(data, (bytes, bytearray)) else json.dumps(data).encode("utf-8")
        return self.lane.put_blob(bytes(body), name=name)

    def _names(self):
        """The store manifest IS the flat advert map: name -> {'sha256':cid,'size':n}."""
        return self.lane.manifest() or {}

    def recv(self, name, timeout=300.0, first=None):
        """Poll for `name`; return its bytes. `first` optionally receives the cid on hit."""
        t0 = time.time()
        delay = self.poll_s
        while True:
            cid = self._cid(self._names().get(name))
            if cid:
                if first is not None:
                    first.append(cid)
                return self.lane.get_blob(cid)
            if time.time() - t0 > timeout:
                raise TimeoutError("pipe recv timeout on %s after %.0fs" % (name, timeout))
            time.sleep(delay)
            delay = min(delay * 1.5, self.MAX_BACKOFF_S)


# ================================================================ stage model loading (torch)
def _manifest_pieces_for_layers(shard_dir, lo, hi):
    """Piece ids whose experts live in layers [lo, hi) -- pure metadata via piece_loader.

    SCHEMA (measured 2026-07-25 -- getting this wrong produced a silently EXPERT-LESS model that
    generated pure garbage): each record is {"piece": "trunk"|"experts_<id>", "experts": "trunk"
    | [[layer, expert], ...], ...}. The key is "piece", NOT "name", and the (layer, expert) pairs
    are inline -- an unmatched filter here returns [] and the caller loads trunk weights ONLY."""
    import piece_loader as pl
    man = pl.load_manifest(shard_dir, require_files=False)
    need = []
    for p in man["pieces"]:
        nm = str(p.get("piece") or "")
        pairs = p.get("experts")
        if not nm.startswith("experts_") or not isinstance(pairs, list):
            continue
        if any(lo <= int(L) < hi for (L, _e) in pairs):
            need.append(int(nm.split("_", 1)[1]))
    return sorted(set(need))


def load_stage(shard_dir, config_dir, lo, hi, device="cpu", role="mid", dtype=None, log=None):
    """Build the stage module set: embed (role=first), layers[lo..hi) with ALL their experts, and
    for role='head' ONLY final norm + lm_head (the worker-side sampler; lo/hi ignored). Everything
    else stays meta/Identity and is never touched. Returns (model, cfg).

    VRAM cap note: callers set the per-process cap BEFORE calling this (project rule)."""
    import torch
    from transformers import AutoConfig
    import piece_loader as pl                                  # the lane's PROVEN loader
    log = log or (lambda m: None)
    cfg = AutoConfig.from_pretrained(config_dir, local_files_only=True)

    want_embed = (role == "first")
    want_head = (role == "head")
    keep = set(range(lo, hi)) if not want_head else set()

    # WHY piece_loader and not a hand-rolled safetensors walk (measured 2026-07-25, twice):
    # GLM's experts are FUSED (Glm4MoeLiteNaiveMoe) -- there is no
    # `model.layers.N.mlp.experts.<e>.*` submodule to assign into, so
    # set_module_tensor_to_device raises AttributeError('...has no attribute 0'). Worse, the
    # attempt BEFORE that silently loaded trunk-only and served garbage. build_partial_model is
    # the same code path the CE lane trains and the coordinator gates on -- the model-in-kind.
    pieces = _manifest_pieces_for_layers(shard_dir, lo, hi) if keep else []
    model, summ = pl.build_partial_model(shard_dir, pieces, device=device, config_dir=config_dir,
                                         strip_mtp=True)
    model.eval()
    loaded = len(pieces)
    resident = set(int(x) for x in (summ.get("resident_layers") or []))
    # Only MoE layers need resident experts: the first `first_k_dense_replace` layers use a DENSE
    # MLP that lives in the trunk, so they never appear in resident_layers (measured: layer 0).
    n_dense = int(getattr(cfg, "first_k_dense_replace", 0) or 0)
    missing = sorted(L for L in keep if L >= n_dense and L not in resident)
    if missing:
        raise RuntimeError(
            "stage[%s] layers[%d:%d): piece_loader did NOT make layer(s) %s resident (got %s). "
            "Refusing to serve: a non-resident MoE layer contributes nothing and the stage would "
            "emit garbage silently." % (role, lo, hi, missing[:5], sorted(resident)[:8]))
    # CACHE REMAP (measured 2026-07-24: decode diverged 1.7 abs while prefill was bit-exact): a
    # mid-stage writes its K/V at ABSOLUTE layer indices, but DynamicCache.get_seq_length() reads
    # layer 0 -- which a mid-stage never fills -- so create_causal_mask sized the past as 0 on
    # every decode step. Remap each kept layer's cache slot to a DENSE per-stage index starting at
    # 0; position_ids stay absolute (rotary is unaffected).
    for i in range(cfg.num_hidden_layers):
        if i in keep:
            model.model.layers[i].self_attn.layer_idx = i - lo
    # Un-materialized layers become Identity so nothing meta can ever be touched by accident.
    for i in range(cfg.num_hidden_layers):
        if i not in keep:
            model.model.layers[i] = torch.nn.Identity()
    if not want_embed:
        model.model.embed_tokens = torch.nn.Identity()
    if not want_head:
        model.model.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()
    # FAIL LOUD ON UNFILLED WEIGHTS. init_empty_weights leaves every un-materialized parameter on
    # the META device; a meta MoE expert does NOT raise -- it silently contributes nothing, and the
    # stage emits fluent-looking garbage. MEASURED 2026-07-25: a manifest-schema mismatch loaded
    # trunk-only (321 tensors, ZERO experts) and a full 47-layer run generated '!!!!!!' for 128
    # tokens with no error anywhere. Never ship a stage that cannot prove it is fully materialized.
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    kept_meta = [n for n in meta
                 if (n.startswith("model.layers.") and n.split(".")[2].isdigit()
                     and int(n.split(".")[2]) in keep)
                 or (want_embed and n.startswith("model.embed_tokens."))
                 or (want_head and (n.startswith("model.norm.") or n.startswith("lm_head.")))]
    if kept_meta:
        raise RuntimeError(
            "stage[%s] layers[%d:%d): %d parameter(s) THIS stage owns are still on meta (never "
            "materialized) -- e.g. %s. Refusing to serve: an unfilled expert produces garbage "
            "silently. Check the shard dir has the expert pieces for this range."
            % (role, lo, hi, len(kept_meta), kept_meta[:3]))
    log("stage[%s] layers[%d:%d) resident via %d expert piece(s) on %s "
        "(experts=%s placeholders=%s; all owned params materialized)"
        % (role, lo, hi, loaded, device, summ.get("n_resident_experts"),
           summ.get("n_placeholder_experts")))
    return model, cfg


def stage_step(model, cfg, hidden, position_ids, cache, lo, hi, ids=None):
    """Run layers [lo, hi) exactly the way Glm4MoeLiteModel.forward does (transformers 5.8.1):
    same mask util, same rotary, same layer kwargs. `ids` instead of `hidden` on the first stage."""
    import torch
    from transformers.masking_utils import create_causal_mask
    with torch.inference_mode():
        if ids is not None:
            hidden = model.model.embed_tokens(ids)
        mask = create_causal_mask(config=cfg, inputs_embeds=hidden, attention_mask=None,
                                  past_key_values=cache, position_ids=position_ids)
        pos_emb = model.model.rotary_emb(hidden, position_ids=position_ids)
        for i in range(lo, hi):
            hidden = model.model.layers[i](
                hidden, attention_mask=mask, position_embeddings=pos_emb,
                position_ids=position_ids, past_key_values=cache, use_cache=True)
    return hidden


def new_cache(cfg):
    from transformers import DynamicCache
    return DynamicCache(config=cfg)


# ================================================================ stage server loop
def run_stage(lane, run, stage_idx, model, cfg, lo, hi, device, *, bus=None, log=None,
              idle_exit_s=1800.0, max_streams=64):
    """Poll for activations entering stage `stage_idx`, forward through layers [lo,hi), publish for
    stage_idx+1. First stage (idx 0) receives token IDS (json), not activations. One DynamicCache
    per stream, evicted on the stream's done-advert. Runs until idle_exit_s with no traffic."""
    import torch
    bus = bus or PipeBus(lane, log=log)
    log = log or (lambda m: None)
    caches, positions, done_seen = {}, {}, set()
    seen = set()
    last_traffic = time.time()
    log("pipe stage %d UP layers[%d:%d) device=%s run=%s" % (stage_idx, lo, hi, device, run))
    while time.time() - last_traffic < idle_exit_s:
        names = bus._names()
        moved = False
        for name, rec in list(names.items()):
            cid = bus._cid(rec)
            base = "%s/%s/" % (bus.prefix, run)
            if not name.startswith(base) or name in seen:
                continue
            rest = name[len(base):]                       # <stream>/<seg>/in<k>  or  <stream>/done
            parts = rest.split("/")
            if len(parts) == 2 and parts[1] == "done":
                stream = parts[0]
                if stream not in done_seen:
                    done_seen.add(stream)
                    caches.pop(stream, None)
                    positions.pop(stream, None)
                seen.add(name)
                continue
            if len(parts) != 3 or parts[2] != "in%d" % stage_idx:
                continue
            stream, seg = parts[0], parts[1]
            if stream in done_seen:
                seen.add(name)
                continue
            seen.add(name)
            body = lane.get_blob(cid)
            if stream not in caches:
                caches[stream], positions[stream] = new_cache(cfg), 0
            p0 = positions[stream]
            if stage_idx == 0:
                ids = torch.tensor([json.loads(body.decode("utf-8"))["ids"]],
                                   dtype=torch.long, device=device)
                npos = ids.shape[1]
                pos = torch.arange(p0, p0 + npos, device=device).unsqueeze(0)
                out = stage_step(model, cfg, None, pos, caches[stream], lo, hi, ids=ids)
            else:
                hidden = act_decode(body, device)
                npos = hidden.shape[1]
                pos = torch.arange(p0, p0 + npos, device=device).unsqueeze(0)
                out = stage_step(model, cfg, hidden, pos, caches[stream], lo, hi)
            positions[stream] = p0 + npos
            bus.send("%s%s/%s/in%d" % (base, stream, seg, stage_idx + 1), act_encode(out))
            moved = True
            last_traffic = time.time()
        if not moved:
            time.sleep(bus.poll_s)
    log("pipe stage %d idle for %.0fs -- exiting" % (stage_idx, idle_exit_s))


# ================================================================ driver (worker side)
class PipeDriver:
    """The rollout worker's view of the fleet-hosted model: send the prompt (token ids) into
    stage 0, receive the final-stage hidden state, norm+head+sample LOCALLY (the paid entity keeps
    the sampling seed + signs what it sampled), loop the chosen token back to stage 0."""

    def __init__(self, lane, run, n_stages, head_model, cfg, device="cpu", stream_prefix="s",
                 bus=None, timeout=600.0, log=None):
        self.bus = bus or PipeBus(lane, log=log)
        self.run, self.n, self.head, self.cfg = run, int(n_stages), head_model, cfg
        self.device, self.timeout = device, timeout
        self.log = log or (lambda m: None)
        self._sn = 0
        self.stream_prefix = stream_prefix

    def _base(self, stream):
        return "%s/%s/%s" % (self.bus.prefix, self.run, stream)

    def new_stream(self):
        self._sn += 1
        return "%s%d-%d" % (self.stream_prefix, int(time.time()), self._sn)

    def logits_for(self, hidden_last):
        import torch
        with torch.inference_mode():
            h = self.head.model.norm(hidden_last.to(self.device))
            return self.head.lm_head(h)

    def prefill(self, stream, ids):
        self.bus.send(self._base(stream) + "/p/in0", {"ids": [int(x) for x in ids]})
        out = self.bus.recv(self._base(stream) + "/p/in%d" % self.n, timeout=self.timeout)
        return act_decode(out, self.device)[:, -1:, :]        # last position only

    def step(self, stream, t, token_id):
        self.bus.send(self._base(stream) + "/t%d/in0" % t, {"ids": [int(token_id)]})
        out = self.bus.recv(self._base(stream) + "/t%d/in%d" % (t, self.n), timeout=self.timeout)
        return act_decode(out, self.device)[:, -1:, :]

    def finish(self, stream):
        self.bus.send(self._base(stream) + "/done", {"done": True})


def make_pipeline_backend(lane, run, n_stages, shard_dir, config_dir, device="cpu",
                          tokenizer=None, timeout=600.0, log=None):
    """Return an object satisfying the rollout worker's backend protocol (generate(prompt, ...))
    that runs the FULL fleet-hosted policy. Loads only norm+lm_head (+tokenizer) locally."""
    import torch
    from transformers import AutoTokenizer
    from glm_rollout_worker import _sample_token                     # reuse the worker's sampler
    head, cfg = load_stage(shard_dir, config_dir, 0, 0, device=device, role="head", log=log)
    tok = tokenizer or AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
    drv = PipeDriver(lane, run, n_stages, head, cfg, device=device, timeout=timeout, log=log)

    class _PipeBackend:
        name = "pipeline"

        def generate(self, prompt, *, max_new_tokens, temperature, top_p, seed):
            stream = drv.new_stream()
            ids = tok(prompt, return_tensors=None)["input_ids"]
            if len(ids) > 1024:
                ids = ids[-1024:]
            gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
            hidden = drv.prefill(stream, ids)
            out_ids, logps = [], []
            eos = tok.eos_token_id
            try:
                for t in range(int(max_new_tokens)):
                    logits = drv.logits_for(hidden)[0, -1].float()
                    tid = _sample_token(logits, temperature, top_p, gen)
                    out_ids.append(int(tid))
                    logps.append(float(torch.log_softmax(logits, dim=-1)[int(tid)].item()))
                    if eos is not None and int(tid) == int(eos):
                        break
                    hidden = drv.step(stream, t, int(tid))
            finally:
                drv.finish(stream)
            # KEY NAMES ARE THE CONTRACT (glm_rollout_worker.generate_rollouts reads these with
            # .get() defaults -- a mismatch does NOT raise, it silently packs EMPTY completions and
            # every reward reads 0.0. Measured 2026-07-25: a full 2.5h full-47 window produced 8
            # real generations that were discarded exactly this way. Locked by a contract test.)
            return {"completion_text": tok.decode(out_ids, skip_special_tokens=True),
                    "completion_ids": out_ids, "sample_logprobs": logps,
                    "n_tokens": len(out_ids)}

    return _PipeBackend()
