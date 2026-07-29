#!/usr/bin/env python
"""LAYER-CLAIM TRUE-GRADIENT CACHE -- the shared on-disk format and the dose kernel.

WHY THIS EXISTS (measured, not preference)
------------------------------------------
The per-expert rank-16 micro-dose is CLOSED as payable currency: its best honest yield was
-0.000985 CE, one sixth of the accept margin, and its value lives in a well centred on the
optimizer's exact endpoint (docs/GLM_TRAINER_SYNTHESIS_2026-07-28.md, "Refuted / closed").
The unit that replaced it is the LAYER. All 64 experts of layer 1 (I=1536, 604.0 M params,
FULL-RANK, K=8 chunked), trained on cached TRUE layer-output gradients over 600 fresh batches,
moved full-47 held-out CE

    4.816991 -> 4.725587   (-0.091404 = 15.3x the accept margin 0.005987)

against the per-expert unit's -0.000985 at matched drift -- a 92.83x yield ratio
(docs/research/B3_LAYER_RUN_2026-07-28.md). This module is the shippable refactor of the
measurement machinery in scratchpad/b3_layer_run.py: the producer (coordinator) writes caches
with it, the consumer (miner) verifies and trains from them with it, and neither one re-derives
the arithmetic.

WHAT IS PRESERVED FROM THE RESEARCH RUN, AND WHY EACH ITEM WAS PAID FOR
----------------------------------------------------------------------
  1. CACHED TENSORS ARE DETACHED AT CAPTURE (`assert_detached`, enforced by the writer).
     `cache_slice` once stored routing weights still carrying a `grad_fn`; the first inner-step
     backward died with "Trying to backward through the graph a second time", and had it not
     died it would have pinned 150 backward graphs in host RAM (B3_LAYER_RUN_2026-07-28.md,
     production constraint 3). The guard is permanent and the writer refuses the unit, so a bad
     cache can never reach a miner.
  2. THE DOSE IS A DRIFT TARGET, NEVER A LEARNING RATE (`bisect_lr_for_drift`,
     `dose_spec_from_config`). MEASURED: lr 1.8006e-04 -> rho 6.05e-03, but +6.9% lr = 10x the
     drift and +14.4% = NaN divergence. The linear surrogate is unbounded below, so a client
     that accepts a raw lr from config silently diverges or overshoots. The campaign specifies
     rho*, the CLIENT bisects locally to land it within RHO_TOL, and reports the ACHIEVED rho
     next to its delta. `dose_spec_from_config` RAISES on any config carrying an lr.
  3. K-CHUNKED fp32 MATERIALISATION (`inner_step`). The surrogate is a sum of PER-EXPERT terms,
     so each expert's arithmetic is identical for every K and K only controls how many experts
     are live at once. VERIFIED: one SGD step from zero delta gives byte-identical weights for
     K = 1, 2, 4, 8, 16, max abs diff 0.000e+00. K=8 is the frozen production chunk.
  4. FAIL CLOSED (`verify_cache`). A cache is a directory whose manifest is written LAST, after
     every chunk and its digest; a partial write therefore has no manifest at all and a tampered
     one fails its content hash. The verifier raises; it never returns a partial cache.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
  * It does not JUDGE. tools/glm_product_judge.py owns the k=24 accept decision. A dose that
    diverged is rejected economically there; the guards here exist so the miner does not waste
    the work in the first place.
  * It does not choose F. F is an experiment parameter, not a throughput knob: F=4 vs F=8 on the
    same steps is NOT bit-identical (max abs diff 1.297e+01, cosine as low as -0.0243), so a
    cache built at one fold cannot be replayed against another. F travels in the manifest and the
    verifier enforces it (docs/research/THROUGHPUT_LEVERS_2026-07-28.md, GATE F-NUM).
  * It does not compress the delta. A full-rank layer delta is 1.21 GB bf16. `delta_wire_plan`
    is the seam a compressor plugs into; no compression ships until the raw path is proven.

FLAG: NEURAHASH_SD_LAYER_CLAIMS, DEFAULT OFF. With the flag unset the coordinator constructs
nothing, the contributor imports nothing new, and both paths are byte-identical to today.

Run the tests: C:/Python313/python.exe -m pytest tests/test_glm_layer_claims.py -q
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

# ============================================================================== flag / env helpers
FLAG = "NEURAHASH_SD_LAYER_CLAIMS"


def _truthy(v, default="0"):
    return (v if v is not None else default).strip().lower() not in ("", "0", "false", "no", "off")


def layer_claims_on(environ=None):
    """Master gate for layer-claim true-gradient training. DEFAULT OFF -- with the flag unset
    every caller's path is byte-identical to today (same truthiness house style as
    glm_product_judge.product_judge_on and sharddiloco_glm_coordinator._sd_async_on)."""
    env = os.environ if environ is None else environ
    return _truthy(env.get(FLAG), "0")


def _env_int(name, default, environ=None):
    env = os.environ if environ is None else environ
    try:
        return int(str(env.get(name, "")).strip())
    except (TypeError, ValueError):
        return int(default)


def _env_float(name, default, environ=None):
    env = os.environ if environ is None else environ
    try:
        return float(str(env.get(name, "")).strip())
    except (TypeError, ValueError):
        return float(default)


# ============================================================================= measured constants
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
CHUNK_FMT = "unit_%05d.pt"

K_CHUNK = 8                 # frozen production chunk (K-EXACT verified over K=1,2,4,8,16)
RHO_TOL = 0.005             # 0.5% -- the tolerance the reference run's bisection converged to
MAX_BISECT = 40
MAX_BRACKET = 12            # x4 bracket steps before declaring the target unreachable
MAX_BACKOFF = 12            # halvings before "no finite step exists" (0.5**12 = 1/4096 of the probe
                            # lr, itself 1/1000 of the target -- past that the cache is poison, and
                            # each attempt costs a FULL pass, ~16.5 s on the reference cache)
BACKOFF_FACTOR = 0.5        # a diverged lr is an UPPER BOUND, not a death sentence -- come down
PROBE_LR_FRAC = 1e-3        # slope-probe lr as a fraction of the target drift (MEASURED, not guessed)
UNITS_PER_CHUNK = 8         # ~25 MB/unit at F=4 -> ~200 MB per chunk file

# Host RAM, not VRAM, caps the tap set: 3.79 GB per tap per 600 batches against ~31.7 GB
# available (THROUGHPUT_LEVERS_2026-07-28.md, "What limits the tap set").
GB_PER_TAP_PER_600_BATCHES = 3.79
TAP_CEILING = 8             # 8 x 3.79 = 30.3 GB vs ~31.7 GB available -- the hard ceiling
DEFAULT_TAP_CAP = 6         # ships below the ceiling; NEURAHASH_SD_LAYER_TAP_CAP overrides
DEFAULT_HOST_RAM_GB = 31.7

# Reference-run provenance, so a miner can say what it is reproducing.
REFERENCE = {
    "layer": 1, "experts": 64, "intermediate": 1536, "params": 604_000_000,
    "n_batches": 600, "fold": 4, "k_chunk": 8,
    "base_ce": 4.816991, "dose_b_ce": 4.725587, "dose_b_delta": -0.091404,
    "dose_a_rho": 6.065e-03, "dose_b_rho": 6.065e-02,
    "accept_margin": 0.005987, "per_expert_best": -0.000985,
    "artifact": "docs/research/B3_LAYER_RUN_2026-07-28.md",
}

# A full-rank layer delta, bf16: 604.0 M params x 2 B.
DELTA_BYTES_BF16 = 1_208_000_000
LAN_MB_PER_S_MEASURED = 53.78


# ====================================================================================== exceptions
class GradCacheError(RuntimeError):
    """Base for every fail-closed condition in this module."""


class NotDetachedError(GradCacheError):
    """A tensor offered to the writer still carries requires_grad/grad_fn. Caching it would pin a
    backward graph in host RAM and re-enter the cache-phase autograd graph on the first inner
    step (measured 2026-07-28)."""


class CacheVerifyError(GradCacheError):
    """The cache on disk is not the cache this consumer asked for. Never downgraded to a warning:
    training on someone else's gradients produces a plausible-looking wrong delta."""


class CacheIncompleteError(CacheVerifyError):
    """No manifest, or fewer chunks than the manifest declares -- a partial write."""


class RawLearningRateRejected(GradCacheError):
    """A dose spec carried a learning rate. The dose is a DRIFT TARGET: +6.9% lr = 10x drift,
    +14.4% = NaN divergence, so an lr chosen anywhere but local bisection is a bug."""


class DriftBisectionError(GradCacheError):
    """Bisection could not land the target drift within tolerance -- the client must NOT train."""


class NonFiniteDoseError(GradCacheError):
    """A step produced NaN/Inf. The dose is abandoned; nothing is published."""


# ======================================================================== the detach guard (item 1)
def _is_tensor(x):
    return hasattr(x, "detach") and hasattr(x, "requires_grad")


def assert_detached(obj, name="tensor"):
    """Raise NotDetachedError if `obj` (or any tensor inside a list/tuple/dict) is still attached
    to an autograd graph. Cheap, and it is the guard that a whole debugging cycle bought."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_detached(v, "%s[%r]" % (name, k))
        return obj
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_detached(v, "%s[%d]" % (name, i))
        return obj
    if not _is_tensor(obj):
        return obj
    if bool(obj.requires_grad) or getattr(obj, "grad_fn", None) is not None:
        raise NotDetachedError(
            "%s carries requires_grad=%s grad_fn=%s -- cached tensors MUST be .detach()ed at "
            "capture. An undetached routing weight threw 'backward through the graph a second "
            "time' on 2026-07-28 and would pin one backward graph per cached unit in host RAM."
            % (name, bool(obj.requires_grad), type(getattr(obj, "grad_fn", None)).__name__))
    return obj


# =============================================================================== manifest + digest
_MANIFEST_CORE = ("format_version", "campaign_id", "layer", "fold", "corpus_sha", "batch_spec",
                  "n_units", "unit_steps", "chunks", "dtypes", "experts", "top_k", "created_utc",
                  "producer")


def _canon(x):
    """JSON-canonical form: tuples -> lists, so a manifest survives a save/load round trip and
    two manifests compare equal iff they mean the same thing."""
    if isinstance(x, tuple):
        return [_canon(v) for v in x]
    if isinstance(x, list):
        return [_canon(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, float):
        return float(x)
    if isinstance(x, bool) or isinstance(x, int) or x is None or isinstance(x, str):
        return x
    return str(x)


def manifest_digest(manifest):
    """sha256 over the canonical core of the manifest (which includes every chunk's own digest),
    so any edit to a field OR to a chunk's bytes changes the content hash."""
    core = {k: _canon(manifest.get(k)) for k in _MANIFEST_CORE}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sha_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def batch_spec(seed, step0, n_batches, fold, extra=None):
    """The reproducibility record a consumer checks its cache against: which draws produced these
    gradients. Kept small and exact -- a consumer that got the wrong draws must fail, not guess."""
    spec = {"seed": int(seed), "step0": int(step0), "n_batches": int(n_batches),
            "fold": int(fold)}
    if extra:
        spec.update({str(k): _canon(v) for k, v in extra.items()})
    return spec


# ========================================================================================== writer
class GradCacheWriter:
    """Chunked writer. Order is load-bearing: every chunk file is written to `<name>.tmp` and
    os.replace()d into place, its digest recorded, and the MANIFEST IS WRITTEN LAST. A crash at
    any point therefore leaves a directory with no manifest, which `verify_cache` rejects as
    incomplete -- a partial cache can never be mistaken for a whole one."""

    def __init__(self, path, campaign_id, layer, corpus_sha, fold, spec,
                 experts=None, top_k=None, units_per_chunk=UNITS_PER_CHUNK, producer=None,
                 torch_mod=None):
        if not campaign_id:
            raise ValueError("campaign_id is required -- a flat namespace lets a dead campaign's "
                             "records merge into a live one (memory cross-campaign-record-replay)")
        self.path = str(path)
        self.campaign_id = str(campaign_id)
        self.layer = int(layer)
        self.corpus_sha = str(corpus_sha)
        self.fold = int(fold)
        self.spec = _canon(dict(spec))
        self.experts = None if experts is None else int(experts)
        self.top_k = None if top_k is None else int(top_k)
        self.units_per_chunk = max(1, int(units_per_chunk))
        self.producer = producer or "glm_grad_cache/%d" % FORMAT_VERSION
        self._torch = torch_mod
        self._buf = []            # [(step, unit)] pending for the current chunk
        self._chunks = []         # [{"name","sha256","bytes","units","steps"}]
        self._steps = []
        self._dtypes = {}
        self._closed = False
        os.makedirs(self.path, exist_ok=True)
        mf = os.path.join(self.path, MANIFEST_NAME)
        if os.path.exists(mf):
            os.remove(mf)         # a rewrite must not leave the OLD manifest describing NEW chunks

    # -- lifecycle ---------------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_a):
        if exc_type is None:
            self.close()
        return False

    def _torch_mod(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    # -- units -------------------------------------------------------------------------------
    def add_unit(self, step, unit):
        """Buffer one cached unit. `unit` maps names to detached tensors (h, g, tk, w) plus the
        plain-python `off` offsets. EVERY tensor is checked for detachment here (item 1)."""
        if self._closed:
            raise GradCacheError("writer already closed")
        if not isinstance(unit, dict):
            raise TypeError("unit must be a dict of name -> tensor, got %r" % type(unit).__name__)
        assert_detached(unit, "unit[step=%s]" % step)
        for k, v in unit.items():
            if _is_tensor(v) and k not in self._dtypes:
                self._dtypes[k] = str(getattr(v, "dtype", "?"))
        self._buf.append((int(step), unit))
        self._steps.append(int(step))
        if len(self._buf) >= self.units_per_chunk:
            self._flush()
        return self

    def _flush(self):
        if not self._buf:
            return
        torch = self._torch_mod()
        name = CHUNK_FMT % len(self._chunks)
        dst = os.path.join(self.path, name)
        tmp = dst + ".tmp"
        payload = {"units": [{"step": s, "data": u} for s, u in self._buf]}
        torch.save(payload, tmp)
        os.replace(tmp, dst)
        self._chunks.append({"name": name, "sha256": _sha_file(dst),
                             "bytes": os.path.getsize(dst),
                             "units": len(self._buf),
                             "steps": [s for s, _ in self._buf]})
        self._buf = []

    def close(self):
        """Flush the tail, then write the manifest LAST."""
        if self._closed:
            return self.manifest
        self._flush()
        m = {
            "format_version": FORMAT_VERSION,
            "campaign_id": self.campaign_id,
            "layer": self.layer,
            "fold": self.fold,
            "corpus_sha": self.corpus_sha,
            "batch_spec": self.spec,
            "n_units": len(self._steps),
            "unit_steps": list(self._steps),
            "chunks": _canon(self._chunks),
            "dtypes": _canon(self._dtypes),
            "experts": self.experts,
            "top_k": self.top_k,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "producer": self.producer,
        }
        m["content_hash"] = manifest_digest(m)
        mf = os.path.join(self.path, MANIFEST_NAME)
        tmp = mf + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1, sort_keys=True)
        os.replace(tmp, mf)
        self._closed = True
        self.manifest = m
        return m


def write_cache(path, units, campaign_id, layer, corpus_sha, fold, spec, **kw):
    """Convenience wrapper: `units` is an iterable of (step, unit_dict)."""
    with GradCacheWriter(path, campaign_id, layer, corpus_sha, fold, spec, **kw) as w:
        for step, unit in units:
            w.add_unit(step, unit)
    return w.manifest


# ================================================================================ verifier (reader)
def read_manifest(path):
    mf = os.path.join(str(path), MANIFEST_NAME)
    if not os.path.exists(mf):
        raise CacheIncompleteError(
            "no %s in %s -- the cache is absent or the write was interrupted (the manifest is "
            "written LAST, precisely so this case is unambiguous)" % (MANIFEST_NAME, path))
    try:
        with open(mf, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as ex:                                    # noqa: BLE001 -- fail closed
        raise CacheVerifyError("%s is unreadable: %s: %s" % (mf, type(ex).__name__, ex))


def verify_cache(path, campaign_id=None, layer=None, corpus_sha=None, fold=None, spec=None,
                 check_bytes=True):
    """FAIL CLOSED. Raises CacheVerifyError on ANY mismatch; returns the manifest on success.

    Checked, in order: manifest present (else incomplete) -> content hash over the canonical core
    -> every chunk present, right size and right sha256 -> unit count consistent -> then the
    CONSUMER's expectations (campaign, layer, corpus sha, fold, batch spec). The last group is
    what stops a miner training on another coordinate's or another campaign's gradients."""
    path = str(path)
    m = read_manifest(path)
    if int(m.get("format_version", -1)) != FORMAT_VERSION:
        raise CacheVerifyError("format_version %r != %d" % (m.get("format_version"),
                                                            FORMAT_VERSION))
    want = m.get("content_hash")
    got = manifest_digest(m)
    if not want or want != got:
        raise CacheVerifyError("content hash mismatch: manifest says %r, recomputed %r -- the "
                               "manifest or a chunk digest was edited" % (want, got))
    chunks = m.get("chunks") or []
    total = 0
    for ch in chunks:
        cp = os.path.join(path, str(ch.get("name")))
        if not os.path.exists(cp):
            raise CacheIncompleteError("chunk %s is missing" % ch.get("name"))
        if check_bytes:
            size = os.path.getsize(cp)
            if int(ch.get("bytes", -1)) != size:
                raise CacheVerifyError("chunk %s is %d bytes, manifest says %s"
                                       % (ch.get("name"), size, ch.get("bytes")))
            dig = _sha_file(cp)
            if dig != ch.get("sha256"):
                raise CacheVerifyError("chunk %s sha256 %s != manifest %s"
                                       % (ch.get("name"), dig[:16], str(ch.get("sha256"))[:16]))
        total += int(ch.get("units", 0))
    if total != int(m.get("n_units", -1)):
        raise CacheVerifyError("chunks hold %d units, manifest declares %s"
                               % (total, m.get("n_units")))
    if campaign_id is not None and str(m.get("campaign_id")) != str(campaign_id):
        raise CacheVerifyError("campaign %r != expected %r (cross-campaign replay guard)"
                               % (m.get("campaign_id"), campaign_id))
    if layer is not None and int(m.get("layer", -1)) != int(layer):
        raise CacheVerifyError("cache is for layer %r, this miner claimed layer %r"
                               % (m.get("layer"), layer))
    if corpus_sha is not None and str(m.get("corpus_sha")) != str(corpus_sha):
        raise CacheVerifyError("corpus_sha %r != expected %r" % (m.get("corpus_sha"), corpus_sha))
    if fold is not None and int(m.get("fold", -1)) != int(fold):
        raise CacheVerifyError(
            "fold F=%r != expected %r -- F is an experiment parameter, not a throughput knob: "
            "F=4 and F=8 caches of the SAME steps differ by up to 1.297e+01 with cosine as low "
            "as -0.0243 (GATE F-NUM)" % (m.get("fold"), fold))
    if spec is not None and _canon(dict(spec)) != _canon(m.get("batch_spec") or {}):
        raise CacheVerifyError("batch_spec mismatch: cache %r != expected %r"
                               % (m.get("batch_spec"), _canon(dict(spec))))
    return m


def iter_cache(path, verify=True, torch_mod=None, **expect):
    """Yield (step, unit) in written order. Verifies first unless the caller already did."""
    path = str(path)
    m = verify_cache(path, **expect) if verify else read_manifest(path)
    if torch_mod is None:
        import torch as torch_mod                                    # noqa: PLW0127
    for ch in (m.get("chunks") or []):
        payload = torch_mod.load(os.path.join(path, str(ch["name"])), weights_only=False)
        for rec in payload["units"]:
            yield int(rec["step"]), rec["data"]


def read_cache(path, verify=True, torch_mod=None, **expect):
    """Whole cache as {step: unit}. Fails closed before a single tensor is handed back."""
    return {s: u for s, u in iter_cache(path, verify=verify, torch_mod=torch_mod, **expect)}


# ================================================== MULTI-TAP: N layer caches from ONE backward
# A single backward computes dCE/d(layer_k output) for EVERY k as an intrinsic part of the chain rule;
# the single-tap harness recorded layer 1's and discarded the other 45. Tapping N claimed layers in the
# SAME backward yields N caches at zero extra compute. MEASURED bit-identical to single-tap at 0 ULP at
# both folds with `backward_calls == 1` asserted on a counter, not a stopwatch
# (docs/research/THROUGHPUT_LEVERS_2026-07-28.md, GATE MT-EXACT).
#
# THE RISK THIS EXISTS TO HANDLE: under checkpointing the forward hook fires TWICE and only ONE firing
# carries the gradient. The single-tap harness sidestepped it by exempting the tap layer from
# checkpointing; multi-tap cannot (N exempted layers each retain a ~1.16 GiB streamed slab). So which
# firing carries the gradient is discovered EMPIRICALLY per generation, and two gradient-carrying
# firings RAISE -- a doubled cache is a wrong cache that still looks plausible.
BACKWARD_CALLS = [0]


def reset_backward_counter():
    BACKWARD_CALLS[0] = 0


def multitap_backward(loss, **kw):
    """The ONLY backward entry point. Counting here is what makes "N taps, one backward" an assertion
    on a counter rather than a stopwatch reading."""
    BACKWARD_CALLS[0] += 1
    loss.backward(**kw)


def default_block_getter(model, i):
    """The MoE block of layer i on the model's OWN module (never a separately constructed one)."""
    return model.model.layers[i].mlp.experts


class DoubleCountError(GradCacheError):
    """More than one firing of a tap carried a gradient -> the recompute contributed. Never summed."""


class NoGradientError(GradCacheError):
    """No firing of a tap carried a gradient -- the tap is not on the path from the loss."""


class _LayerTap:
    __slots__ = ("layer", "firings", "inputs", "n_fire", "gen")

    def __init__(self, layer):
        self.layer = layer
        self.clear()

    def clear(self):
        self.firings = []          # zero-leaf tensors, in firing order
        self.inputs = []           # (h, idx, w) tuples, parallel to firings
        self.n_fire = 0
        self.gen = -1


class MultiTapCache:
    """Gradient tap on the MoE-block output of every layer in `layer_indices`.

    A forward hook adds a ZERO fp32 leaf to the block output, so leaf.grad == dCE/d(block output).
    Adding exact zero is value-preserving, so a tap does not perturb the forward and taps do not
    perturb each other.

    Per unit:
        mt.new_batch(); loss = loss_fn(...); multitap_backward(loss)
        caches = mt.harvest()          # {layer: {"grad","h","idx","w","tk","off",...}}
        mt.release()                   # MANDATORY: the leaves own .grad and the captures are live
                                       # graph tensors; holding them leaked ~1.2 GiB per backward.

    Ported from scratchpad/multitap_cache.py (CPU-proven: 25/25 gradient comparisons bit-identical
    across N = 1/2/4 taps, adjacent and distant layers, fp32 and bf16)."""

    def __init__(self, model, layer_indices, block_getter=default_block_getter,
                 capture_inputs=True, lean_firing_index=None, max_firings=8, torch_mod=None):
        self.model = model
        self.layers = list(layer_indices)
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("duplicate layer index in %r" % (self.layers,))
        self.block_getter = block_getter
        self.capture_inputs = bool(capture_inputs)
        # None = record EVERY firing and PROVE which one carries the gradient (default, safe).
        # int  = record only that firing index (only after the proof, to save one fp32 buffer).
        self.lean_firing_index = lean_firing_index
        self.max_firings = int(max_firings)
        self.gen = 0
        self.taps = {i: _LayerTap(i) for i in self.layers}
        self._handles = []
        self._torch = torch_mod
        self.last_report = {}

    def _t(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    # -- lifecycle ---------------------------------------------------------------------------
    def attach(self):
        if self._handles:
            raise GradCacheError("already attached")
        for i in self.layers:
            mod = self.block_getter(self.model, i)
            tap = self.taps[i]
            if self.capture_inputs:
                self._handles.append(mod.register_forward_pre_hook(self._make_pre(tap)))
            self._handles.append(mod.register_forward_hook(self._make_post(tap)))
        return self

    def detach_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    def __enter__(self):
        return self.attach()

    def __exit__(self, *_exc):
        self.detach_hooks()
        self.release()
        return False

    def new_batch(self):
        self.gen += 1
        for tap in self.taps.values():
            tap.clear()
            tap.gen = self.gen
        return self

    def release(self):
        """Drop every live reference: the leaves (which own .grad) and the captured graph tensors."""
        for tap in self.taps.values():
            tap.clear()
        return self

    # -- hooks -------------------------------------------------------------------------------
    def _record(self, tap):
        return True if self.lean_firing_index is None else tap.n_fire == self.lean_firing_index

    def _make_pre(self, tap):
        def pre(_mod, args):
            if len(args) < 3:
                raise GradCacheError(
                    "layer %d: experts forward got %d positional args, expected >=3 "
                    "(hidden_states, top_k_index, top_k_weights)" % (tap.layer, len(args)))
            if self._record(tap):
                tap.inputs.append((args[0], args[1], args[2]))
            return None
        return pre

    def _make_post(self, tap):
        torch = self._t()

        def post(_mod, _args, output):
            if output.dim() != 2:
                raise GradCacheError("layer %d: experts output is %dD %s, expected 2D [n_tok,H]"
                                     % (tap.layer, output.dim(), tuple(output.shape)))
            keep = self._record(tap)
            tap.n_fire += 1
            if tap.n_fire > self.max_firings:
                raise GradCacheError("layer %d: tap fired %d times in one generation (max %d)"
                                     % (tap.layer, tap.n_fire, self.max_firings))
            if not keep:
                return output
            t = torch.zeros(output.shape, dtype=torch.float32, device=output.device,
                            requires_grad=True)
            tap.firings.append(t)
            return output + t.to(output.dtype)
        return post

    # -- harvest -----------------------------------------------------------------------------
    def harvest(self, experts, top_k, to_cpu=True, grad_dtype=None, h_dtype=None):
        """One cache per tapped layer, already in the shape inner_step consumes (h, grad, tk, w,
        off). Raises rather than guessing when the gradient-carrying firing is ambiguous.

        Every tensor is DETACHED here, at capture -- not later, at write time. That ordering is the
        whole point of item 1 in the module docstring."""
        torch = self._t()
        grad_dtype = torch.float32 if grad_dtype is None else grad_dtype
        h_dtype = torch.bfloat16 if h_dtype is None else h_dtype
        out, total = {}, 0
        for i in self.layers:
            tap = self.taps[i]
            if not tap.firings:
                raise NoGradientError("layer %d: tap never fired (generation %d)" % (i, self.gen))
            got = [k for k, t in enumerate(tap.firings) if t.grad is not None]
            if not got:
                raise NoGradientError(
                    "layer %d: %d firing(s), none carried a gradient -- the tap is not on the path "
                    "from the loss, or backward() was never called" % (i, tap.n_fire))
            if len(got) > 1:
                raise DoubleCountError(
                    "layer %d: %d of %d firings carried a gradient (indices %r) -- the recompute "
                    "contributed a SECOND gradient. Refusing to sum them."
                    % (i, len(got), tap.n_fire, got))
            k = got[0]
            g = tap.firings[k].grad.detach()
            g = g.to("cpu", grad_dtype).clone() if to_cpu else g.clone()
            rec = {"grad": g, "firings": tap.n_fire, "grad_firing": k,
                   "recorded_firings": len(tap.firings), "backward_calls": BACKWARD_CALLS[0]}
            nb = g.numel() * g.element_size()
            if self.capture_inputs and tap.inputs:
                h, idx, w = tap.inputs[k if k < len(tap.inputs) else 0]
                h = h.detach()
                h = h.to("cpu", h_dtype).clone() if to_cpu else h.clone()
                tk, wf, off = cache_slice(idx, w, experts, top_k, to_cpu=to_cpu,
                                          torch_mod=torch)
                rec.update(h=h, tk=tk, w=wf, off=off)
                nb += sum(t.numel() * t.element_size() for t in (h, tk, wf))
            rec["bytes"] = nb
            total += nb
            out[i] = assert_detached(rec, "harvest[layer=%d]" % i)
        self.last_report = {"layers": list(self.layers), "generation": self.gen,
                            "bytes_total": total,
                            "bytes_per_layer": {i: out[i]["bytes"] for i in self.layers},
                            "firings_per_layer": {i: out[i]["firings"] for i in self.layers},
                            "grad_firing_per_layer": {i: out[i]["grad_firing"] for i in self.layers},
                            "backward_calls": BACKWARD_CALLS[0]}
        return out


def cache_slice(idx_s, w_s, experts, top_k, to_cpu=True, torch_mod=None):
    """Routed (token, expert) pairs sorted by expert (stable, so within an expert they stay in
    row-major token order). Returns (tk int32, w fp32, off list[E+1]) -- exactly what inner_step
    indexes with.

    BOTH INPUTS ARE DETACHED FIRST. `w` is a live tensor inside the cache-phase autograd graph:
    keeping its grad_fn made the cached routing weights re-enter that graph, so the first inner-step
    backward died with "Trying to backward through the graph a second time" (measured 2026-07-28),
    and every cached step would have pinned a whole backward graph in host RAM. Ported from
    scratchpad/b3_layer_run.py:338."""
    if torch_mod is None:
        import torch as torch_mod                                    # noqa: PLW0127
    idx_s, w_s = idx_s.detach(), w_s.detach()
    flat = idx_s.reshape(-1).to(torch_mod.int64)
    order = torch_mod.argsort(flat, stable=True)
    offs = torch_mod.cat([torch_mod.zeros(1, dtype=torch_mod.long, device=flat.device),
                          torch_mod.bincount(flat, minlength=int(experts)).cumsum(0)])
    tk = torch_mod.div(order, int(top_k), rounding_mode="floor").to(torch_mod.int32)
    w = w_s.reshape(-1)[order].float()
    if to_cpu:
        tk, w = tk.cpu().clone(), w.cpu().clone()
    return tk, w, [int(x) for x in offs.cpu()]


# ============================================================== tap cap (host RAM, not VRAM, caps)
def tap_cap(n_claimed, cap=None, host_ram_gb=None, n_batches=600, log=None):
    """How many layers may be tapped in ONE backward. Multi-tap is bit-identical to single-tap at
    0 ULP with backward_calls == 1, so taps cost no compute -- they cost HOST RAM: 3.79 GB per tap
    per 600 batches against ~31.7 GB available, hence a hard ceiling of 8
    (THROUGHPUT_LEVERS_2026-07-28.md). Returns (n_taps, reason) and LOGS whenever it clamps, so a
    short tap set is never silently mistaken for a short claim set."""
    n_claimed = max(0, int(n_claimed))
    cap = DEFAULT_TAP_CAP if cap is None else int(cap)
    cap = max(1, min(cap, TAP_CEILING))
    ram = DEFAULT_HOST_RAM_GB if host_ram_gb is None else float(host_ram_gb)
    per_tap = GB_PER_TAP_PER_600_BATCHES * (float(n_batches) / 600.0)
    ram_taps = TAP_CEILING if per_tap <= 0 else max(1, int(ram // per_tap))
    limit = min(cap, ram_taps, TAP_CEILING)
    n = min(n_claimed, limit)
    if n_claimed <= limit:
        reason = "no clamp: %d claimed <= limit %d" % (n_claimed, limit)
    else:
        which = ("configured cap %d" % cap) if cap <= min(ram_taps, TAP_CEILING) else (
            "host RAM %.1f GB / %.2f GB per tap = %d taps" % (ram, per_tap, ram_taps)
            if ram_taps <= TAP_CEILING else "hard ceiling %d" % TAP_CEILING)
        reason = ("clamped %d claimed layers to %d taps (%s; %.2f GB per tap per %d batches)"
                  % (n_claimed, n, which, per_tap, n_batches))
        if log:
            log("[layer-claim] tap set %s" % reason)
    return n, reason


def plan_taps(claimed, cap=None, host_ram_gb=None, n_batches=600, log=None):
    """The tapped subset of `claimed`, in the claim order the caller gave (so the oldest/most
    senior claim is never starved by a clamp)."""
    claimed = list(claimed)
    n, reason = tap_cap(len(claimed), cap=cap, host_ram_gb=host_ram_gb, n_batches=n_batches,
                        log=log)
    return claimed[:n], reason


# ================================================================== the dose kernel (items 2 and 3)
def chunk_bounds(E, K):
    """[(c0, c1)] -- verbatim in effect from scratchpad/b3_layer_run.py:_chunks."""
    K = max(1, int(K))
    return [(c, min(E, c + K)) for c in range(0, E, K)]


def inner_step(unit, gu_all, dn_all, Dgu, Ddn, act_fn, K, lr, dev="cpu", torch_mod=None):
    """ONE plain-SGD step on the WHOLE layer from one cached batch. No optimizer state.

    The surrogate is accumulated as a sum of PER-EXPERT terms
        L = sum_e sum_{pairs (t,s) routed to e}  w[t,s] * (f_e(h[t]) . g[t])
    which is the same scalar as (scatter(out) * g).sum() but avoids index_add_ atomics, so the
    step is deterministic AND each expert's arithmetic is identical for every K. Each expert's
    fp32 weights are their OWN autograd leaves; K controls only how many are materialised at once
    and share one backward(). That independence is what GATE K-EXACT measured: K = 1, 2, 4, 8, 16
    give byte-identical weights, max abs diff 0.000e+00.

    Ported from scratchpad/b3_layer_run.py:110 with the arithmetic unchanged."""
    if torch_mod is None:
        import torch as torch_mod                                    # noqa: PLW0127
    F = torch_mod.nn.functional
    E = gu_all.shape[0]
    h = unit["h"].to(dev)
    g = unit["g"].to(dev)
    tk = unit["tk"].to(dev).long()
    w = unit["w"].to(dev)
    off = unit["off"]
    for c0, c1 in chunk_bounds(E, K):
        if off[c1] == off[c0]:
            continue
        Wg = [(gu_all[e].float() + Dgu[e]).detach().requires_grad_(True) for e in range(c0, c1)]
        Wd = [(dn_all[e].float() + Ddn[e]).detach().requires_grad_(True) for e in range(c0, c1)]
        surr = None
        for j in range(c1 - c0):
            lo, hi = off[c0 + j], off[c0 + j + 1]
            if hi == lo:
                continue
            te = tk[lo:hi]
            gup = F.linear(h[te].float(), Wg[j])
            gate, up = gup.chunk(2, dim=-1)
            dh = F.linear(act_fn(gate) * up, Wd[j])
            term = (dh * w[lo:hi][:, None] * g[te]).sum()
            surr = term if surr is None else surr + term
        if surr is not None:
            surr.backward()
            with torch_mod.no_grad():
                for j in range(c1 - c0):
                    if Wg[j].grad is not None:
                        Dgu[c0 + j].add_(Wg[j].grad, alpha=-lr)
                    if Wd[j].grad is not None:
                        Ddn[c0 + j].add_(Wd[j].grad, alpha=-lr)
        del Wg, Wd, surr
    del h, g, tk, w


def sq_blocks(gu, dn, I, K, ref_gu=None, ref_dn=None, torch_mod=None):
    """(gate, up, down) sums of squares in float64, chunked so no full fp32 copy is ever live.
    Ported from scratchpad/b3_layer_run.py:157."""
    if torch_mod is None:
        import torch as torch_mod                                    # noqa: PLW0127
    sg = su = sd = 0.0
    for c0, c1 in chunk_bounds(gu.shape[0], K):
        x = gu[c0:c1].float()
        if ref_gu is not None:
            x = x - ref_gu[c0:c1].to(x.device).float()
        sg += float(torch_mod.sum(x[:, :I] * x[:, :I], dtype=torch_mod.float64))
        su += float(torch_mod.sum(x[:, I:] * x[:, I:], dtype=torch_mod.float64))
        del x
        y = dn[c0:c1].float()
        if ref_dn is not None:
            y = y - ref_dn[c0:c1].to(y.device).float()
        sd += float(torch_mod.sum(y * y, dtype=torch_mod.float64))
        del y
    return sg, su, sd


def base_norms(gu_all, dn_all, I, K=K_CHUNK, torch_mod=None):
    sg, su, sd = sq_blocks(gu_all, dn_all, I, K, torch_mod=torch_mod)
    return {"gate": math.sqrt(sg), "up": math.sqrt(su), "down": math.sqrt(sd)}


def rho_of(Dgu, Ddn, I, K, base_n, torch_mod=None):
    """Relative drift = max over (gate, up, down) of ||delta|| / ||base||. Ported from
    scratchpad/b3_layer_run.py:177."""
    sg, su, sd = sq_blocks(Dgu, Ddn, I, K, torch_mod=torch_mod)
    r = {"gate": math.sqrt(sg) / base_n["gate"], "up": math.sqrt(su) / base_n["up"],
         "down": math.sqrt(sd) / base_n["down"]}
    return max(r.values()), r


def _finite(x):
    return isinstance(x, float) and x == x and x not in (float("inf"), float("-inf"))


# ------------------------------------------------------------------ the dose spec (drift, not lr)
DOSE_LR_KEYS = ("lr", "learning_rate", "inner_lr", "step_size", "eta")


def dose_spec_from_config(cfg, tol=None):
    """Build a dose spec from campaign config. RAISES RawLearningRateRejected if the config names
    a learning rate under ANY of DOSE_LR_KEYS.

    This is not defensive style, it is the measured protocol: lr 1.8006e-04 landed rho 6.05e-03,
    but +6.9% on that lr gave 10x the drift and +14.4% diverged to NaN. The window is razor thin
    and NOT transferable between boxes, caches or layers, so the only safe currency is the target
    drift; the client finds its own lr by bisection (B3_LAYER_RUN_2026-07-28.md, constraint 1)."""
    cfg = dict(cfg or {})
    named = [k for k in DOSE_LR_KEYS if k in cfg and cfg[k] is not None]
    if named:
        raise RawLearningRateRejected(
            "dose config names %s -- a raw learning rate is NEVER accepted. Specify "
            "target_rho instead: +6.9%% on the reference lr gave 10x the drift and +14.4%% gave "
            "NaN, so an lr that is right for one cache silently diverges on another." % (named,))
    if "target_rho" not in cfg or cfg["target_rho"] is None:
        raise ValueError("dose config must carry target_rho (the measured drift target)")
    rho = float(cfg["target_rho"])
    if not (_finite(rho) and rho > 0.0):
        raise ValueError("target_rho must be finite and > 0, got %r" % (cfg["target_rho"],))
    t = RHO_TOL if tol is None else float(tol)
    if "tol" in cfg and cfg["tol"] is not None:
        t = float(cfg["tol"])
    return {"target_rho": rho, "tol": t, "k_chunk": int(cfg.get("k_chunk", K_CHUNK))}


def _pass_or_diverged(pass_fn, lr):
    """One pass -> rho, or +inf when the dose DIVERGED.

    Divergence arrives two ways and both mean the same thing -- this lr is past the stability edge:
    pass_fn returns NaN/Inf, or it RAISES NonFiniteDoseError. The second is what happens on the REAL
    path, because train_layer_dose's full_pass calls assert_finite_delta BEFORE it ever computes
    rho, so a version that only inspected the return value would still have died on the miner."""
    try:
        rho = float(pass_fn(lr))
    except NonFiniteDoseError:
        return float("inf")
    return rho if _finite(rho) else float("inf")


def measure_drift_slope(pass_fn, target_rho, probe_frac=PROBE_LR_FRAC, max_backoff=MAX_BACKOFF,
                        on_pass=None, log=None):
    """MEASURE the CHORD slope rho/lr with ONE deliberately small probe pass. Never guess it.

    Returns {"lr", "rho", "slope", "backoffs"}; the bisection seed is then `target / slope`.

    `slope` is the chord from the ORIGIN (rho_probe / lr_probe), not drho/dlr at the probe point.
    That distinction is load-bearing and in the safe direction: rho(lr) is convex here, so the chord
    UNDER-estimates the derivative and `target/chord` therefore OVERSHOOTS the answer -- which is
    exactly why `lo = 0.0` is a valid initial bracket, and why the back-off below is not optional.
    Measured on the reference cache: chord 33.6176 at lr=6.065e-05 vs 314.1 near the target, so the
    seed came out 9.37x high, diverged 3 times, and bisection walked it down to 1.925077e-04.

    Why this is the default and not an option: the old code fell back to `seed_lr = target` whenever
    a caller omitted `rho_unit_lr1` -- and the only production caller never passed it. A DRIFT used
    as a LEARNING RATE is 6.065e-02 where the reference run's converged lr is 1.924965e-04, ~315x
    too large, so the very first probe diverged and the miner died (run 6, 2026-07-28). A measured
    slope needs no config, holds for any layer/cache/box, and cannot be forgotten by a caller --
    which is precisely how that bug happened."""
    lr0 = float(target_rho) * float(probe_frac)
    lr = lr0
    for k in range(int(max_backoff) + 1):
        rho = _pass_or_diverged(pass_fn, lr)
        if on_pass:
            on_pass(lr, rho, -100 - k)
        if rho != float("inf"):
            slope = (rho / lr) if rho > 0.0 else 0.0
            if log:
                log("[layer-claim] slope probe: lr=%.6e -> rho=%.6e => drho/dlr=%.6e, seed lr=%s"
                    % (lr, rho, slope, ("%.6e" % (float(target_rho) / slope)) if slope > 0.0
                       else "UNUSABLE (probe drift is 0)"))
            return {"lr": float(lr), "rho": float(rho), "slope": float(slope), "backoffs": int(k)}
        lr *= BACKOFF_FACTOR
    raise NonFiniteDoseError(
        "the slope probe diverged at every lr from %.6e down to %.6e (%d back-offs) -- this cache "
        "cannot take even a probe step, so there is no dose to publish"
        % (lr0, lr / BACKOFF_FACTOR, int(max_backoff)))


def bisect_lr_for_drift(pass_fn, target_rho, tol=RHO_TOL, max_bisect=MAX_BISECT,
                        seed_lr=None, rho_unit_lr1=None, n_batches=1,
                        max_bracket=MAX_BRACKET, max_backoff=MAX_BACKOFF,
                        probe_frac=PROBE_LR_FRAC, log=None):
    """Find the lr whose FULL pass lands relative drift `target_rho` within `tol` (relative).

    `pass_fn(lr) -> rho` must run the whole cache from a zeroed delta and return the achieved
    drift. Strategy is the reference run's: seed, bracket UP by x4 until rho >= target, then bisect.

    The seed is MEASURED (one small probe pass -> chord rho/lr -> lr0 = target/chord) unless the caller
    supplies `seed_lr`, or `rho_unit_lr1` when it genuinely knows the slope. There is deliberately
    no "fall back to the target" branch: a drift is not a learning rate.

    A diverged pass is NOT fatal. It is an UPPER BOUND on the lr, so the search backs off and keeps
    going -- exactly what the reference harness does, and it is load-bearing there: dose B's seed
    5.870929e-04 came back NaN, the next pass at 2.935465e-04 came back NaN too, and it still landed
    lr=1.924965e-04 / rho=6.046e-02 eleven passes later (scratchpad/b3_layer_run.json, dose B
    trace). NonFiniteDoseError is raised only when NO lr tried produced a finite pass at all.

    Returns a dict with `converged`. A False verdict means DO NOT TRAIN -- the caller logs and
    skips rather than shipping an unmatched dose."""
    target = float(target_rho)
    if not (_finite(target) and target > 0.0):
        raise ValueError("target_rho must be finite and > 0, got %r" % (target_rho,))
    trace = []
    n_finite = [0]

    def _record(lr, rho, it):
        if rho != float("inf"):
            n_finite[0] += 1
        trace.append({"it": int(it), "lr": float(lr), "rho": float(rho),
                      "finite": bool(rho != float("inf"))})
        if log:
            log("[layer-claim] bisect it=%d lr=%.6e -> rho=%s (target %.6e)"
                % (it, lr, ("%.6e" % rho) if rho != float("inf") else "DIVERGED -- backing off",
                   target))

    def _pass(lr, it):
        rho = _pass_or_diverged(pass_fn, lr)
        _record(lr, rho, it)
        return rho

    probe = None
    if seed_lr is None and rho_unit_lr1:          # optional fast path: the caller measured it already
        seed_lr = target / max(1e-30, float(rho_unit_lr1) * max(1, int(n_batches)))
    if seed_lr is None:
        probe = measure_drift_slope(pass_fn, target, probe_frac=probe_frac,
                                    max_backoff=max_backoff, on_pass=_record, log=log)
        seed_lr = (target / probe["slope"]) if probe["slope"] > 0.0 else probe["lr"]
    hi = float(seed_lr)
    if not (_finite(hi) and hi > 0.0):
        raise ValueError("seed_lr must be finite and > 0, got %r" % (seed_lr,))
    lo = 0.0
    rho = _pass(hi, -1)
    nb = 0
    # `abs(rho - target) > tol * target` guards a seed that ALREADY landed the target: without it a
    # measured/known-slope seed still brackets x4 up and bisects all the way back down, ~11 extra
    # full passes (~3 min of GPU each on the reference cache) to rediscover the lr it started with.
    while rho < target and abs(rho - target) > tol * target and nb < max_bracket and hi < 1e12:
        lo, hi = hi, hi * 4.0
        rho = _pass(hi, -2 - nb)
        nb += 1
    mid = hi
    for it in range(int(max_bisect)):
        if abs(rho - target) <= tol * target:
            break
        mid = 0.5 * (lo + hi)
        rho = _pass(mid, it)
        if rho < target:            # +inf is NOT < target, so a diverged midpoint becomes the new
            lo = mid                # upper bound and the next step halves toward safety. That IS
        else:                       # the back-off; no separate retry loop can get it wrong.
            hi = mid
    if n_finite[0] == 0:
        raise NonFiniteDoseError(
            "all %d passes diverged (lr %.6e down to %.6e) -- no finite step exists on this cache, "
            "so nothing is trained and nothing is published"
            % (len(trace), trace[0]["lr"], trace[-1]["lr"]))
    rel = abs(rho - target) / target
    out = {"lr": float(mid), "achieved_rho": float(rho), "target_rho": target,
           "rel_err": float(rel), "tol": float(tol), "converged": bool(rel <= tol),
           "passes": len(trace), "brackets": nb, "diverged": len(trace) - n_finite[0],
           "probe": probe, "trace": trace}
    if log:
        log("[layer-claim] bisection %s: lr=%.6e rho=%.6e vs target %.6e (rel err %.3f%%) in %d "
            "passes (%d diverged and were backed off)"
            % ("CONVERGED" if out["converged"] else "FAILED", out["lr"], out["achieved_rho"],
               target, 100.0 * rel, out["passes"], out["diverged"]))
    return out


def solve_drift_or_refuse(pass_fn, target_rho, tol=RHO_TOL, log=None, **kw):
    """bisect_lr_for_drift, but a non-converged solution RAISES instead of being returned. Use
    this at the call site that would otherwise apply the delta."""
    sol = bisect_lr_for_drift(pass_fn, target_rho, tol=tol, log=log, **kw)
    if not sol["converged"]:
        raise DriftBisectionError(
            "could not land target drift %.6e within %.2f%% (best %.6e, rel err %.3f%%, %d "
            "passes) -- REFUSING to train: an unmatched dose is wasted work the judge would "
            "reject anyway" % (target_rho, 100.0 * tol, sol["achieved_rho"],
                               100.0 * sol["rel_err"], sol["passes"]))
    return sol


def assert_finite_delta(Dgu, Ddn, torch_mod=None, where="delta"):
    """Guard before anything is applied or published. A NaN in a delta is not recoverable and
    must never reach the wire."""
    if torch_mod is None:
        import torch as torch_mod                                    # noqa: PLW0127
    for name, t in (("gate_up", Dgu), ("down", Ddn)):
        if not bool(torch_mod.isfinite(t).all()):
            raise NonFiniteDoseError(
                "%s: %s contains non-finite values -- the dose diverged; nothing is published "
                "(the k=24 judge would reject it, but the miner's work is wasted, which is the "
                "failure this guard prevents)" % (where, name))
    return True


def apply_delta(gu_all, dn_all, Dgu, Ddn, K=K_CHUNK, torch_mod=None):
    """Fold the fp32 delta into the bf16 slabs, chunked. Returns the max read-back error, which is
    0.0 when the cast is bf16-exact (the reference run asserted exactly that on both doses)."""
    if torch_mod is None:
        import torch as torch_mod                                    # noqa: PLW0127
    assert_finite_delta(Dgu, Ddn, torch_mod=torch_mod, where="apply_delta")
    exact = 0.0
    with torch_mod.no_grad():
        for c0, c1 in chunk_bounds(gu_all.shape[0], K):
            ngu = (gu_all[c0:c1].float() + Dgu[c0:c1]).to(gu_all.dtype)
            ndn = (dn_all[c0:c1].float() + Ddn[c0:c1]).to(dn_all.dtype)
            gu_all[c0:c1].copy_(ngu)
            dn_all[c0:c1].copy_(ndn)
            exact = max(exact, float((gu_all[c0:c1] - ngu).abs().max()),
                        float((dn_all[c0:c1] - ndn).abs().max()))
            del ngu, ndn
    return exact


def train_layer_dose(units, gu_all, dn_all, I, act_fn, target_rho, tol=RHO_TOL, K=K_CHUNK,
                     dev="cpu", seed_lr=None, rho_unit_lr1=None, log=None, torch_mod=None,
                     apply=True):
    """The whole client-side dose: bisect to the DRIFT TARGET, then apply at the found lr.

    `units` is an ordered sequence of cached units (from `iter_cache`). Returns a report carrying
    the ACHIEVED rho next to the lr -- the miner publishes both, because the campaign specified a
    drift and only the miner knows what its own bisection landed.

    Refuses (raises) rather than training when bisection cannot converge, and aborts before any
    apply when a step goes non-finite."""
    if torch_mod is None:
        import torch as torch_mod                                    # noqa: PLW0127
    units = list(units)
    if not units:
        raise GradCacheError("no cached units -- nothing to train on")
    bn = base_norms(gu_all, dn_all, I, K, torch_mod=torch_mod)
    Dgu = torch_mod.zeros_like(gu_all, dtype=torch_mod.float32)
    Ddn = torch_mod.zeros_like(dn_all, dtype=torch_mod.float32)
    t0 = time.time()

    def full_pass(lr):
        Dgu.zero_()
        Ddn.zero_()
        for u in units:
            inner_step(u, gu_all, dn_all, Dgu, Ddn, act_fn, K, lr, dev, torch_mod=torch_mod)
        assert_finite_delta(Dgu, Ddn, torch_mod=torch_mod, where="full_pass(lr=%.6e)" % lr)
        r, _ = rho_of(Dgu, Ddn, I, K, bn, torch_mod=torch_mod)
        return r

    sol = solve_drift_or_refuse(full_pass, target_rho, tol=tol, seed_lr=seed_lr,
                                rho_unit_lr1=rho_unit_lr1, n_batches=len(units), log=log)
    rho_final = full_pass(sol["lr"])          # re-run at the chosen lr: Dgu/Ddn now hold the dose
    _, parts = rho_of(Dgu, Ddn, I, K, bn, torch_mod=torch_mod)
    rep = {"lr": sol["lr"], "achieved_rho": float(rho_final), "target_rho": float(target_rho),
           "rel_err": abs(rho_final - float(target_rho)) / float(target_rho),
           "rho_blocks": parts, "n_units": len(units), "k_chunk": int(K),
           "bisect_passes": sol["passes"], "wall_s": round(time.time() - t0, 3),
           "base_norms": bn, "applied": False, "readback_err": None}
    if apply:
        rep["readback_err"] = apply_delta(gu_all, dn_all, Dgu, Ddn, K=K, torch_mod=torch_mod)
        rep["readback_bf16_exact"] = bool(rep["readback_err"] == 0.0)
        rep["applied"] = True
    rep["delta"] = (Dgu, Ddn)
    if log:
        log("[layer-claim] dose: lr=%.6e achieved rho=%.6e (target %.6e, rel err %.3f%%) over %d "
            "units in %.1fs" % (rep["lr"], rep["achieved_rho"], rep["target_rho"],
                                100.0 * rep["rel_err"], rep["n_units"], rep["wall_s"]))
    return rep


# ============================================================ delta publication: the raw wire seam
def delta_wire_plan(n_params=None, uplink_mbps=None, lan_mb_s=LAN_MB_PER_S_MEASURED):
    """What a full-rank layer delta costs on the wire, and where a compressor would plug in.

    MEASURED: 1.21 GB bf16 moves in ~23 s on the LAN at 53.78 MB/s, and is impossible on a
    ~1.2 Mbps home uplink (~2.2 h). The raw path ships FIRST because correctness comes first;
    `compressible=True` marks the seam a per-expert SVD/low-rank factorization of the WEIGHT
    delta would occupy behind its own flag with its own fidelity check. The project's 67.7x
    LoRA-factor precedent applies to WEIGHT deltas only -- it does NOT apply to gradient caches,
    which are activations and gradients, not low-rank weight updates."""
    nb = DELTA_BYTES_BF16 if n_params is None else int(n_params) * 2
    out = {"bytes": nb, "gb": nb / 1e9, "lan_s": nb / (lan_mb_s * 1e6),
           "compressible": True, "compression": None}
    if uplink_mbps:
        out["uplink_mbps"] = float(uplink_mbps)
        out["uplink_s"] = nb * 8.0 / (float(uplink_mbps) * 1e6)
        out["uplink_feasible"] = bool(out["uplink_s"] <= 3600.0)
    return out


__all__ = [
    "FLAG", "layer_claims_on", "FORMAT_VERSION", "K_CHUNK", "RHO_TOL", "MAX_BISECT",
    "DEFAULT_TAP_CAP", "TAP_CEILING", "GB_PER_TAP_PER_600_BATCHES", "REFERENCE",
    "GradCacheError", "NotDetachedError", "CacheVerifyError", "CacheIncompleteError",
    "RawLearningRateRejected", "DriftBisectionError", "NonFiniteDoseError",
    "assert_detached", "batch_spec", "manifest_digest", "GradCacheWriter", "write_cache",
    "read_manifest", "verify_cache", "iter_cache", "read_cache",
    "MultiTapCache", "multitap_backward", "reset_backward_counter", "cache_slice",
    "DoubleCountError", "NoGradientError", "default_block_getter",
    "tap_cap", "plan_taps", "chunk_bounds", "inner_step", "sq_blocks", "base_norms", "rho_of",
    "dose_spec_from_config", "bisect_lr_for_drift", "solve_drift_or_refuse",
    "assert_finite_delta", "apply_delta", "train_layer_dose", "delta_wire_plan",
]
