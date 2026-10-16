#!/usr/bin/env python3
"""LAYER-MAJOR capture: stream the 47-layer stack ONCE per round instead of once per unit (#136).

THE COST THIS REMOVES. `glm_layer_cache_pass._drive` is BATCH-MAJOR: one backward per fold group,
and each backward walks the whole stack, so every one of the round's 15 groups re-streams all 44
streamed MoE layers -- twice, because the gradient-checkpoint recompute reads them again. Measured
capture traffic for one joint experiment: 7.97 TB. Inverting the loop

    for L in 0..46:  hold(L);  push all 15 groups through layer L, one group at a time;  release(L)

reads each layer once per PHASE instead of once per GROUP.

THE SAVING IS A LAW, AND ITS SIZE IS NOT YET MEASURED ON THE REAL MODEL. Per-unit traffic is
G x (L_fwd + L_recompute) layer reads; layer-major traffic is (L_fwd + L_reverse) and does NOT
depend on G at all. That independence -- not any particular multiple -- is what the gate asserts.
Measured on the CPU fixture (4 layers, taps 1+2): per-unit 6912 B at 1 group and 27648 B at 4
(exactly x4), layer-major 9216 B at both (exactly x1), i.e. 3.00x fewer bytes at G=4 and worse
than per-unit at G=1. Extrapolating the same law to the production shape (44 streamed layers,
taps 1+5, G=15) PREDICTS ~15x, which is where the 7.97 TB -> 0.53 TB target comes from -- a
PREDICTION to be measured by `tools/glm_capture_layer_major_gate.py --gpu-real`, not a result.
The primitive that makes residency follow the inverted loop is
`glm_arc_eval.ExpertStreamer.hold` / `.release` (glm_arc_eval.py:949/969), the same pair that
bought the EVAL loop its measured 43.4x; inverting the loop WITHOUT hold/release changes nothing,
because `_post` nulls a non-pinned layer the instant its forward returns.

WHY IT IS BIT-EXACT BY CONSTRUCTION, NOT BY LUCK. Every GEMM sees exactly the tokens it sees today,
in the same grouping, in the same order: a group's 2048-token batch is still pushed through layer L
as its own call, so no shape the kernel dispatches on ever changes. That distinction is the whole
point of this design. The cheaper-looking alternative -- widen the backward so all 15 groups ride
in ONE call -- is arithmetically the existing fold knob F and is already REFUTED: measured
2026-07-28 (GATE F-NUM) 43.9% of routed pairs land on a DIFFERENT expert, min gradient cosine
-0.0243, max|dg| 12.97. The mechanism is the SKINNY ROUTER GEMM (the wide FFN is bit-identical at
0.000e+00), so byte-identical grouping is not a nicety here, it is the entire reason this variant
can be exact while widening cannot.

PROVENANCE, corrected 2026-08-11. An earlier version of this comment said "the router moves by up
to 2.0 and re-routes 245/2048 tokens" as though those were the REAL-MODEL numbers. They are not:
they come from the synthetic bare-bf16-GEMM arm of the discriminator
(scratchpad/capture_group_discriminator.json, arm_b_mechanism_bare_gemm), which reproduces the
MECHANISM on made-up tensors. The real-model artefact says something ~340x smaller:
scratchpad/b3_levers_fnum.json records h_max_abs = 0.005859375 and tk_frac_differing = 0.439453125
over routed_pairs = 2048 from tokens_probe = 512. Both numbers are real; attributing the synthetic
one to the real run overstated the perturbation by more than two orders of magnitude.
Also note tk_frac_differing is PERMUTATION-SENSITIVE -- it is not a per-token expert diff, so it is
not comparable to a routing-churn metric computed per token. Do not quote it as one.

THE STRUCTURE. Manual per-layer checkpointing, at exactly the granularity torch.utils.checkpoint
already uses in the batch-major path:

  phase 1  layer-major FORWARD under no_grad, saving the BOUNDARY activation entering each layer
           (h is [512, 2048] bf16 = 8.39 MB per group per layer; 15 groups x 48 boundaries = 5.9 GB
           host-resident and pageable). One `hold` per layer, 15 group calls, one `release`.
  phase 2  per group: the epilogue (final norm) + the SHARED lm_head/NLL tail
           (`glm_layer_cache_pass.loss_from_hidden`), backward to dL/d(top boundary).
  phase 3  layer-major REVERSE, L = 46..0: one `hold`, then per group recompute layer L WITH grad
           from its saved boundary and push that group's incoming gradient through it. The
           recompute is the same operation the checkpointed path performs today; the tap
           (`glm_grad_cache.MultiTapCache`) fires exactly ONCE there, so DoubleCountError is
           structurally impossible rather than merely proven.

NOTHING HERE REIMPLEMENTS THE MODEL. The preamble comes from `glm_arc_eval._capture_preamble`
(which runs the model's OWN preamble against a recorder), the per-layer forward is
`model.model.layers[L](...)`, the epilogue is `model.model.norm`, and the loss tail is the
production `loss_from_hidden`. If a future GLM grows per-layer masks or a second threaded state,
`GlmStackAdapter.run_layer` is the one place that has to know.

DEFAULT OFF. `glm_layer_cache_pass.build_unit_source` keeps calling `_drive` unless
NEURAHASH_SD_LAYER_MAJOR_CAPTURE is set; the batch-major path stays reachable and tested.

THE GATE IS THE DELIVERABLE. tools/glm_capture_layer_major_gate.py compares this driver against the
SHIPPED per-unit driver elementwise at max|diff| == 0.000e+00 exactly (never a tolerance), proves
the comparison FAILS on a deliberately perturbed arm, and checks that each arm actually READ what
its path implies -- without that last teeth a silent fall-back to per-unit would pass bit-exactness
trivially and report a win.
"""

import os
import time

#: Provenance of the most recent layer-major pass. Never load-bearing; read by the gate and by
#: post-mortems, exactly like glm_layer_cache_pass.LAST_PASS.
LAST_PASS = {}

#: Env knob. Default OFF -- the shipped batch-major driver stays the default path.
ENV_FLAG = "NEURAHASH_SD_LAYER_MAJOR_CAPTURE"

#: Where boundary activations live between the phases. "cpu" keeps ~5.9 GB off the card; the copy
#: is a pure move and cannot change a bit.
DEFAULT_BOUNDARY_DEVICE = "cpu"


def enabled(environ=None):
    """Is the layer-major capture turned on? DEFAULT ON since 2026-08-11, opt out with the flag.

    Flipped on MEASUREMENT, not on the design argument. The gate ran at PRODUCTION WIDTH (15 groups,
    tools/glm_capture_layer_major_gate.py --gpu-real --n-batches 60 --fold 4) against the shipped
    per-unit driver as an independent oracle, and came back:

        worst max|diff|  0.000e+00 over 480 tensors, 0 problems   <- the bar, not a tolerance
        bytes streamed   1487.2 GiB -> 101.2 GiB   (14.69x)
        wall              456.62 s  ->   67.03 s   ( 6.81x)

    It issues 720 backwards where the batch-major path issues 15 -- 48x more calls -- and is still
    6.81x faster, which is the measure of how completely weight IO dominated. Record:
    scratchpad/capture_lm_gate_prod15.json.

    WHY THE EARLIER 2-GROUP PASS WAS NOT ENOUGH TO FLIP THIS. Bit-exactness held at 2 groups and the
    grouping makes it bit-exact BY CONSTRUCTION, so it was tempting to extrapolate. But "correct by
    construction" is exactly the argument that certified a fleet pipeline here which then generated
    pure garbage on the real run -- a construction argument is a hypothesis until something measures
    it at the width that ships. So the 15-group arm was run before this line changed.

    OPTING OUT: set NEURAHASH_SD_LAYER_MAJOR_CAPTURE to 0/false/no/off. The batch-major driver stays
    reachable and tested; this is a default flip, not a deletion, because the two paths are gated to
    be bit-exact and either must be able to serve as the other's oracle.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV_FLAG, None)
    if raw is None:
        return True                                   # unset -> ON
    v = str(raw or "").strip().lower()
    if v == "":
        return True                                   # set-but-empty is not an opt-out
    return v not in ("0", "false", "no", "off")


class StackAdapter:
    """How to walk a decoder stack one layer at a time. Four methods, no state.

    The batch-major driver never needs this because it hands the whole forward to
    `model.model(...)`. The layer-major driver owns the loop, so it needs the model's own preamble,
    its own per-layer call, and its own epilogue -- reused, never re-derived.
    """

    def n_layers(self, model):
        raise NotImplementedError

    def preamble(self, model, batch):
        """-> (hidden entering layer 0, kwargs every layer is called with)."""
        raise NotImplementedError

    def run_layer(self, model, L, h, kw):
        """-> the hidden state layer L hands to layer L+1."""
        raise NotImplementedError

    def epilogue(self, model, h):
        """-> the final hidden state `loss_from_hidden` consumes."""
        raise NotImplementedError


class GlmStackAdapter(StackAdapter):
    """The production GLM stack, entered through the model's own objects.

    `preamble` delegates to `glm_arc_eval._capture_preamble`, which temporarily swaps the decoder
    list for a recorder so `Glm4MoeLiteModel.forward` runs VERBATIM and hands back the very causal
    mask / position embeddings the stock loop would have used. `att=None` matches what
    `loss_chunked` passes (`model.model(input_ids=t, use_cache=False)`) -- an explicit ones-mask
    would flip sdpa off its is_causal fast path and change the bit pattern.
    """

    def n_layers(self, model):
        return int(model.config.num_hidden_layers)

    def preamble(self, model, batch):
        import glm_arc_eval as AE
        return AE._capture_preamble(model, batch, None)

    def run_layer(self, model, L, h, kw):
        o = model.model.layers[int(L)](h, **kw)
        # transformers >=5 returns a bare tensor; older returns a tuple.
        return o[0] if isinstance(o, tuple) else o

    def epilogue(self, model, h):
        return model.model.norm(h)


def _require_hold_release(streamer):
    """A streamer without hold/release would silently re-read every layer per group.

    That failure is invisible: the numbers stay bit-exact (nothing about the math changes) and the
    run merely costs what it always cost, so the gate would report a win for a loop that saved
    nothing. Refuse loudly instead. ASCII only -- an em-dash in a raise is a UnicodeEncodeError on
    this box's cp1252 console.
    """
    missing = [n for n in ("hold", "release") if not callable(getattr(streamer, n, None))]
    if missing:
        raise TypeError(
            "layer-major capture needs a streamer with hold/release; %s lacks %s. Without them the "
            "inverted loop re-reads each layer once per group and saves nothing, while still "
            "producing bit-exact numbers -- i.e. it would look like a win."
            % (type(streamer).__name__, ", ".join(missing)))


def _boundary(t, device):
    """Detach a boundary activation for storage. A device move is a pure copy: no bit changes."""
    return t.detach().to(device)


def _reload(t, device):
    """Bring a stored boundary back as a fresh autograd LEAF.

    `.clone()` is mandatory, not defensive: with boundary_device == the compute device, `.to(dev)`
    is the identity and `requires_grad_(True)` would flip the STORED tensor, so a later phase would
    find its own saved activation mutated.
    """
    return t.to(device).clone().detach().requires_grad_(True)


def drive_layer_major(prov, taps, plan, seed, batch, experts, top_k, seq_len, log,
                      adapter=None, boundary_device=DEFAULT_BOUNDARY_DEVICE):
    """Yield (step, {layer: harvest-shaped record}) -- same contract as `glm_layer_cache_pass._drive`.

    Deliberately signature-compatible with `_drive` so the gate can drive both from one call site
    and so the seam in `build_unit_source` is a one-line branch rather than a parallel universe.

    The model is built HERE, as the generator's first step, and torn down in the `finally` -- same
    reason as `_drive`: a generator that is never started never runs its finally, so an eager build
    would leak trunk + pinned slabs for the life of the coordinator.
    """
    import numpy as np
    import torch
    from torch.utils.checkpoint import set_checkpoint_early_stop
    import glm_grad_cache as GC
    import glm_layer_cache_pass as P

    if adapter is None:
        adapter = GlmStackAdapter()
    tapset = [int(x) for x in taps]
    ctx = None
    t_pass, n_units, n_back = time.time(), 0, 0
    peak = 0.0
    LAST_PASS.clear()
    LAST_PASS.update(taps=list(tapset), fold_groups=len(plan), driver="layer_major")
    try:
        ctx = prov.open(tapset)
        ctx["batch"] = batch
        for name, want in (("experts", experts), ("top_k", top_k), ("seq_len", seq_len)):
            if int(ctx[name]) != int(want):
                raise GC.GradCacheError(
                    "the built model reports %s=%d but the caches were opened for %s=%d -- the "
                    "config and the model disagree" % (name, int(ctx[name]), name, int(want)))
        model = ctx["model"]
        dev = ctx["device"]
        streamer = ctx["streamer"]              # KeyError, never a silent default
        if streamer is not None:
            _require_hold_release(streamer)
        if not hasattr(prov, "loss_from_hidden"):
            raise GC.GradCacheError(
                "provider %s has no loss_from_hidden(ctx, hs, batch): the layer-major driver owns "
                "the decoder stack and cannot call the provider's whole-model loss. Refusing "
                "rather than guessing a chunk size." % type(prov).__name__)
        n_layers = int(adapter.n_layers(model))
        n_back_before = GC.BACKWARD_CALLS[0]
        P.cache_mode(model, True, log=log)      # SAME module mode as the oracle (model.train())

        # ---------------------------------------------------------------- phase 1: the forward
        groups = []
        for spans, scale in plan:
            rows = np.concatenate([P.batch_rows(seed, s, ctx["n_train"], ctx["batch"])
                                   for s, _lo, _hi in spans])
            groups.append({"spans": spans, "scale": float(scale),
                           "tb": prov.to_device(rows, ctx)})
        with torch.no_grad():
            for g in groups:
                h, kw = adapter.preamble(model, g["tb"])
                g["kw"] = kw
                g["bnd"] = [_boundary(h, boundary_device)]
                g["cur"] = h
            for L in range(n_layers):
                held = streamer.hold(L) if streamer is not None else False
                try:
                    for g in groups:
                        g["cur"] = adapter.run_layer(model, L, g["cur"], g["kw"])
                        g["bnd"].append(_boundary(g["cur"], boundary_device))
                finally:
                    if held:
                        streamer.release(L)
                if log is not None and (L % 8 == 0 or L == n_layers - 1):
                    log("[layer-major] forward  layer %d/%d  %.0fs" % (L + 1, n_layers,
                                                                       time.time() - t_pass))
            for g in groups:
                g.pop("cur")
        if dev.startswith("cuda"):
            peak = max(peak, torch.cuda.max_memory_allocated(0) / (1 << 30))

        # ------------------------------------------------- phase 2: epilogue + loss -> dL/d(top)
        for g in groups:
            x = _reload(g["bnd"][n_layers], dev)
            with set_checkpoint_early_stop(False):
                loss = prov.loss_from_hidden(ctx, adapter.epilogue(model, x), g["tb"])
                GC.multitap_backward(loss)
            n_back += 1
            if x.grad is None:
                raise GC.GradCacheError(
                    "the loss produced no gradient at the top boundary -- the epilogue is not on "
                    "the path from the loss, so every cached gradient below would be wrong")
            g["gout"] = x.grad.detach().clone()
            g["bnd"][n_layers] = None
            del x, loss

        # ----------------------------------------------------------------- phase 3: the reverse
        recs = [{} for _ in groups]
        for L in range(n_layers - 1, -1, -1):
            held = streamer.hold(L) if streamer is not None else False
            mt = None
            if L in tapset:
                mt = GC.MultiTapCache(model, [L], torch_mod=torch)
                mt.attach()
            try:
                for gi, g in enumerate(groups):
                    x = _reload(g["bnd"][L], dev)
                    if mt is not None:
                        mt.new_batch()
                    out = adapter.run_layer(model, L, x, g["kw"])
                    with set_checkpoint_early_stop(False):
                        GC.multitap_backward(out, gradient=g["gout"])
                    n_back += 1
                    if mt is not None:
                        if gi == 0:
                            cap_in = mt.taps[L].inputs
                            if cap_in:
                                P.assert_routing_matches(cap_in[0][1], experts, top_k, L)
                        recs[gi][L] = mt.harvest(experts, top_k)[L]
                        mt.release()
                    if x.grad is None:
                        raise GC.GradCacheError(
                            "layer %d group %d: the recompute produced no gradient at its input "
                            "boundary -- the reverse chain is broken and every layer below would "
                            "be cached against a stale gradient" % (L, gi))
                    g["gout"] = x.grad.detach().clone()
                    g["bnd"][L] = None
                    del x, out
            finally:
                if mt is not None:
                    mt.detach_hooks()
                    mt.release()
                if held:
                    streamer.release(L)
            if dev.startswith("cuda"):
                peak = max(peak, torch.cuda.max_memory_allocated(0) / (1 << 30))
                torch.cuda.empty_cache()
            if log is not None and (L % 8 == 0 or L == 0):
                log("[layer-major] reverse  layer %d/%d  %.0fs%s"
                    % (L, n_layers, time.time() - t_pass,
                       "" if streamer is None
                       else "  %.2f GiB streamed" % (streamer.bytes_streamed / (1 << 30))))

        moved = GC.BACKWARD_CALLS[0] - n_back_before
        want_back = len(groups) * (n_layers + 1)
        if moved != want_back or n_back != want_back:
            raise GC.GradCacheError(
                "%d backward(s) counted globally and %d locally over %d group(s) x %d layer(s) + "
                "%d epilogue(s): expected %d. A backward entered outside multitap_backward and the "
                "caches would double-count" % (moved, n_back, len(groups), n_layers, len(groups),
                                               want_back))

        # -------------------------------------------------- the same per-step split as _drive
        for gi, g in enumerate(groups):
            spans, scale = g["spans"], g["scale"]
            n_tok_expected = int(spans[-1][2])
            for L, rec in recs[gi].items():
                for key in ("grad", "h"):
                    if int(rec[key].shape[0]) != n_tok_expected:
                        raise GC.GradCacheError(
                            "layer %d: the reverse covered %d %s token(s) but the fold plan "
                            "expects %d (batch %d x seq %d x %d step(s)) -- the split would "
                            "silently misalign" % (L, int(rec[key].shape[0]), key, n_tok_expected,
                                                   ctx["batch"], ctx["seq_len"], len(spans)))
            per_layer = {L: P.split_folded_record(rec, spans, scale, experts)
                         for L, rec in recs[gi].items()}
            recs[gi] = None
            for step, _lo, _hi in spans:
                unit = {int(L): per_layer[L][int(step)] for L in per_layer}
                for L, u in unit.items():
                    GC.assert_detached(u, "unit[layer=%d,step=%d]" % (L, step))
                    if int(u["tk"].numel()) and int(u["tk"].max()) >= (_hi - _lo):
                        raise GC.GradCacheError(
                            "layer %d step %d: routed token index %d is outside the step's %d "
                            "tokens -- the fold split is wrong"
                            % (L, step, int(u["tk"].max()), _hi - _lo))
                n_units += 1
                yield int(step), unit
            del per_layer

        LAST_PASS.update(backwards=moved, n_units=n_units, n_layers=n_layers,
                         peak_gib=round(peak, 2), wall_s=round(time.time() - t_pass, 1),
                         bytes_streamed=(None if streamer is None else streamer.bytes_streamed),
                         bytes_from_ram=(None if streamer is None else streamer.bytes_from_ram))
        if log:
            log("[layer-major] pass DONE: %d unit(s) for %d layer(s) from %d backward(s) over %d "
                "group(s) in %.0fs, peak %.2f GiB"
                % (n_units, len(tapset), moved, len(plan), time.time() - t_pass, peak))
    finally:
        if ctx is not None:
            prov.close(ctx)
