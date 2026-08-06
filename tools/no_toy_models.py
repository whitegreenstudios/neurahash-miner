"""NO TOY MODELS -- the owner's standing directive, made MECHANICAL.

The directive (memory `no-toy-models-directive`, 2026-07-09, restated since): REAL models only for
anything that gates, pays, publishes, verifies, or produces a reported capability/benchmark number.
It was recorded for weeks and violated anyway -- the RLVR learner built its policy with
`build_tiny_glm`, so its whole publish gate scored a toy, and that was caught by EYE, late.

A directive nobody can check is a directive that gets violated again. This module is the check.

WHAT IT IS NOT: a ban on toy models. Toy models have legitimate uses here and breaking them would be
worse than the bug -- the GPU smoke vendors `build_tiny_glm` so it never depends on this repo, the
contributor has a documented `--base tiny` wire shakedown, `sharddiloco_glm_expert --selftest` and
`verify_trunk_reduction` are self-verification scripts, and the unit suite is toy by construction.

WHAT IT IS: two mechanisms with different jobs.

  1. `assert_real_model(...)` -- called by any path that makes a decision (gate / pay / publish /
     report a capability number). It refuses anything it cannot prove is real. Fail CLOSED.
  2. `announce_toy_model(...)` -- called at every legitimate toy CONSTRUCTION site, so a toy run is
     self-identifying in its own log and can never be mistaken for a real one by someone reading
     that log a week later.

HOW "REAL" IS DECIDED -- positive structural evidence, never a name match. Someone will rename
`build_tiny_glm`; nobody can rename a 151552-entry embedding into existence. The verdict comes from
the config's own dimensions, measured against the SMALLEST real base this project actually uses:

    model (real)              vocab_size   hidden_size   source
    GLM-5 glm_moe_dsa            151552          6144    neurahash_torch/glm_backbone.py:54
    Qwen3-0.6B / GLM-4.7 vocab   151936          1024    tools/diloco_contributor.py:85
    Granite-1B (capability runs)  49152          1024    _granite_grpo_best/config.json
    OLMoE-1B-7B                   50280          2048    memory no-toy-models-directive

    model (toy)               vocab_size   hidden_size   source
    build_tiny_glm                   24            64    tools/sharddiloco_glm_expert.py:310
    GLMConfig.small                 256            64    neurahash_torch/glm_backbone.py:112
    glm_moe_dsa_config.tiny.json    100             -    tests/fixtures/
    pool ARCH_RUNGS "large"           -           768    sharded_pool_node.py:604

The two populations are separated by more than an order of magnitude on vocab_size, so the
thresholds below sit in open space rather than on a boundary. Both must hold: a toy that borrowed a
real tokenizer still fails on hidden_size, and a wide toy still fails on vocab_size.

DELIBERATELY NOT USED as gate conditions: parameter count and layer count. Hosting here is
COMPOSITIONAL (memory `glm-capacity-per-card`) -- a worker holds 1-2 of the real model's 47+ layers,
so a genuinely real partial model is SMALL. Both are reported as facts, neither can veto.

TORCH-FREE AT IMPORT. tests/test_glm_grpo_learner.py::test_zz_torch_free asserts that importing the
learner does not pull torch/transformers, and the learner imports this module. Nothing here imports
torch, transformers, or numpy -- config facts are read by duck-typing.
"""
import os
import sys

# --------------------------------------------------------------------------------------- constants
#: Smallest real tokenizer this project uses is Granite's 49152; the widest toy is 512. 32000 is
#: Llama-2's vocab, i.e. the floor of "a real tokenizer" in general, and leaves a 62x gap to the
#: widest toy. Raising this would start rejecting real bases; lowering it toward 512 would start
#: admitting toys.
REAL_MIN_VOCAB_SIZE = 32000

#: Smallest real hidden size this project uses is 1024 (Granite-1B, Qwen3-0.6B); the widest toy is
#: 768 (pool ARCH_RUNGS "large"). 1024 is the tightest value that admits every real base.
REAL_MIN_HIDDEN_SIZE = 1024

#: The one escape hatch. Its name states exactly what setting it does. There is no silent default:
#: unset means REFUSE, and when it IS set every guarded call still prints a banner naming the path
#: it just let through, so a run that used it is self-identifying in its own log.
ALLOW_TOY_ENV = "NEURAHASH_ALLOW_TOY_MODEL_IN_GATED_PATHS"

#: Attribute stamped onto a model/config by `announce_toy_model`. This is a POSITIVE toy marker and
#: an ADDITIONAL veto, never the primary evidence -- structural realness is decided without it, so
#: stripping the mark does not launder a toy.
TOY_MARK_ATTR = "_neurahash_toy_model"

REAL = "REAL"
TOY = "TOY"
INDETERMINATE = "INDETERMINATE"

#: Attribute name aliases, because the backbones in this repo disagree: HF configs use
#: `vocab_size`/`hidden_size`, neurahash_torch's MoETransformer uses `d_model`, the numpy MoELM uses
#: `Demb`/`C`. An alias list is not a name MATCH on the constructor -- it is how we read the number.
_VOCAB_KEYS = ("vocab_size", "padded_vocab_size", "n_vocab", "vocab", "V", "C")
_HIDDEN_KEYS = ("hidden_size", "d_model", "n_embd", "hidden_dim", "model_dim", "Demb")
_LAYER_KEYS = ("num_hidden_layers", "n_layers", "n_layer", "num_layers")


class ToyModelInGatedPathError(RuntimeError):
    """Raised when a model that is not provably real reaches a gating/paying/publishing path."""


def _log(msg, log=None):
    """ASCII-only, FLUSHED. The Windows console here is cp1252 (non-ASCII dies with
    UnicodeEncodeError, measured 22x) and an unflushed banner is a banner nobody reads."""
    if log is not None:
        log(msg)
        return
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _first_attr(obj, keys):
    """First present, non-None, positive-int-ish value among `keys`, read from an object OR a dict."""
    if obj is None:
        return None, None
    for k in keys:
        if isinstance(obj, dict):
            v = obj.get(k)
        else:
            v = getattr(obj, k, None)
        if v is None or isinstance(v, bool):
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv > 0:
            return iv, k
    return None, None


def _candidates(obj):
    """Every object worth reading dimensions off, in priority order.

    Accepts a config, a model (`.config`), a `(model, cfg)` tuple as returned by every builder in
    this repo, or a plain dict. Returns (config_like_candidates, model_or_None).
    """
    model = None
    cands = []
    if isinstance(obj, (tuple, list)):
        # Builders here return (model, cfg); accept either order without guessing wrong.
        for item in obj:
            sub_c, sub_m = _candidates(item)
            cands.extend(sub_c)
            model = model or sub_m
        return cands, model
    if obj is None:
        return cands, None
    cfg = getattr(obj, "config", None)
    if cfg is not None and cfg is not obj:
        model = obj
        cands.append(cfg)
        inner = getattr(cfg, "text_config", None)          # HF multimodal wrappers
        if inner is not None:
            cands.append(inner)
    cands.append(obj)
    if model is None and hasattr(obj, "parameters"):
        model = obj
    return cands, model


def _marked_toy(obj):
    cands, model = _candidates(obj)
    for c in list(cands) + ([model] if model is not None else []):
        if c is None:
            continue
        if isinstance(c, dict):
            if bool(c.get(TOY_MARK_ATTR, False)):
                return True
        elif bool(getattr(c, TOY_MARK_ATTR, False)):
            return True
    return False


def model_facts(obj):
    """Structural facts read off `obj` -- never raises, so a caller can log facts about junk.

    Returns a dict with: vocab_size, hidden_size, n_layers (each None when unreadable), the alias
    key each was read from, n_params (informational only; None when uncountable, e.g. meta tensors),
    marked_toy, and the class name.
    """
    cands, model = _candidates(obj)
    vocab = hidden = layers = None
    vkey = hkey = None
    for c in cands:
        if vocab is None:
            vocab, vkey = _first_attr(c, _VOCAB_KEYS)
        if hidden is None:
            hidden, hkey = _first_attr(c, _HIDDEN_KEYS)
        if layers is None:
            layers, _ = _first_attr(c, _LAYER_KEYS)
    n_params = None
    if model is not None and hasattr(model, "parameters"):
        try:                                              # meta tensors / lazy modules may refuse
            n_params = int(sum(int(p.numel()) for p in model.parameters()))
        except Exception:                                 # noqa: BLE001 -- informational only
            n_params = None
    return {"vocab_size": vocab, "vocab_key": vkey,
            "hidden_size": hidden, "hidden_key": hkey,
            "n_layers": layers, "n_params": n_params,
            "marked_toy": _marked_toy(obj),
            "class": type(obj).__name__ if obj is not None else "None"}


def classify_model(obj):
    """(verdict, reason, facts). verdict is REAL / TOY / INDETERMINATE.

    REAL requires POSITIVE evidence on BOTH dimensions. Anything unreadable is INDETERMINATE, never
    REAL -- that is the fail-closed half, and it is what makes a `None`, a mock, or an unrecognised
    backbone refuse instead of sailing through.
    """
    facts = model_facts(obj)
    if facts["marked_toy"]:
        return TOY, "explicitly marked as a toy model (%s=True)" % TOY_MARK_ATTR, facts
    v, h = facts["vocab_size"], facts["hidden_size"]
    missing = []
    if v is None:
        missing.append("vocab_size (tried %s)" % ", ".join(_VOCAB_KEYS))
    if h is None:
        missing.append("hidden_size (tried %s)" % ", ".join(_HIDDEN_KEYS))
    if missing:
        return (INDETERMINATE,
                "cannot read %s off a %s -- realness is UNPROVEN, so it is refused"
                % (" and ".join(missing), facts["class"]), facts)
    small = []
    if v < REAL_MIN_VOCAB_SIZE:
        small.append("vocab_size=%d < %d" % (v, REAL_MIN_VOCAB_SIZE))
    if h < REAL_MIN_HIDDEN_SIZE:
        small.append("hidden_size=%d < %d" % (h, REAL_MIN_HIDDEN_SIZE))
    if small:
        return TOY, "; ".join(small), facts
    return REAL, "vocab_size=%d and hidden_size=%d are both real-scale" % (v, h), facts


def is_real_model(obj):
    """True only for a provably real model. Convenience for branches; the gate is assert_real_model."""
    return classify_model(obj)[0] == REAL


DEFAULT_REMEDY = (
    "load the REAL model -- tools/piece_loader.build_partial_model(shard_dir, piece_ids, ...) for a "
    "resident GLM piece, or tools/glm_pipe.load_stage(...) for a pipeline stage. Hosting is "
    "COMPOSITIONAL (4.02 GiB trunk + ~1.125 GiB per resident layer), so a small card holds a REAL "
    "partial model -- it never needs a toy stand-in.")


def assert_real_model(obj, where, decision, remedy=None, log=None):
    """Refuse to let anything but a provably real model reach a decision. FAIL CLOSED.

    Args:
        obj:      the model, the config, or the `(model, cfg)` pair about to be used.
        where:    `path:symbol` of the calling site, e.g. "tools/glm_arc_eval.py:main".
                  Printed verbatim so the failure names the offending file without a traceback read.
        decision: what this path DECIDES, in the owner's terms -- "publishes a delta and advances
                  the pointer", "reports ARC-Easy accuracy". This is the sentence that tells a
                  reader why a toy here is not allowed.
        remedy:   what to do instead. Defaults to DEFAULT_REMEDY.
        log:      optional callable for the override banner; defaults to flushed stderr.

    Raises:
        ToyModelInGatedPathError unless the verdict is REAL, or the ALLOW_TOY_ENV override is set
        (in which case it prints a banner naming `where` and returns).
    """
    verdict, reason, facts = classify_model(obj)
    if verdict == REAL:
        return facts
    override = os.environ.get(ALLOW_TOY_ENV, "").strip()
    msg = (
        "NO TOY MODELS -- refusing a %s model on a path that makes a real decision.\n"
        "  where    : %s\n"
        "  decision : %s\n"
        "  verdict  : %s (%s)\n"
        "  facts    : vocab_size=%s hidden_size=%s n_layers=%s n_params=%s class=%s\n"
        "  remedy   : %s\n"
        "  override : ONLY for a deliberate wire shakedown, set %s=1 -- it is announced in the log "
        "and any number the run produces is a TOY number, never a result.\n"
        "  directive: owner, 2026-07-09 (restated) -- REAL models only for anything that gates, "
        "pays, publishes, verifies, or reports a capability number."
        % (verdict, where, decision, verdict, reason,
           facts["vocab_size"], facts["hidden_size"], facts["n_layers"], facts["n_params"],
           facts["class"], remedy or DEFAULT_REMEDY, ALLOW_TOY_ENV))
    if override and override != "0":
        _log("[no-toy-models] OVERRIDE ACTIVE (%s=%s): letting a %s model through at %s. "
             "Every number this run produces is a TOY number.\n%s"
             % (ALLOW_TOY_ENV, override, verdict, where, msg), log=log)
        return facts
    raise ToyModelInGatedPathError(msg)


#: Construction sites that have already announced in THIS process. One banner per site per run is
#: what makes a run self-identifying; 500 identical banners are noise a reader learns to filter out.
_ANNOUNCED = set()


def reset_announcements():
    """Clear the once-per-site announcement memo. For tests only."""
    _ANNOUNCED.clear()


def announce_toy_model(obj=None, where="<unknown>", why="unstated", log=None, once=True):
    """Stamp and ANNOUNCE a deliberately-toy model at its construction site.

    Every legitimate toy path calls this, so a toy run is self-identifying in its own output and a
    log read a week later cannot be mistaken for a real run. The stamp (`TOY_MARK_ATTR`) is an
    additional veto for `classify_model`, not its basis.

    `once` (default True) prints one banner per `where` per process -- the STAMP is always applied,
    so dedupe never weakens the guard, only the log volume.

    Returns `obj` so it can wrap a constructor call site inline.
    """
    facts = model_facts(obj)
    for target in ([obj] if obj is not None else []) + list(_candidates(obj)[0]):
        if target is None or isinstance(target, (dict, tuple, list)):
            continue
        try:
            setattr(target, TOY_MARK_ATTR, True)
        except Exception:                                 # noqa: BLE001 -- frozen/slotted configs
            pass
    if once:
        if where in _ANNOUNCED:
            return obj
        _ANNOUNCED.add(where)
    _log("[no-toy-models] TOY MODEL CONSTRUCTED at %s -- vocab_size=%s hidden_size=%s n_layers=%s. "
         "Legitimate use: %s. NOTHING this model produces is a real result: it must never reach a "
         "gate, a payout, a publish, or a reported capability number (owner directive, 2026-07-09)."
         % (where, facts["vocab_size"], facts["hidden_size"], facts["n_layers"], why), log=log)
    return obj
