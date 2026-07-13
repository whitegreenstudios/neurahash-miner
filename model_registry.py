"""
model_registry.py — the single source of truth for which open model the DENSE serve path runs.

This is a pure name->HF-id alias layer. It imports NOTHING heavy (no torch, no transformers, no
network) so it is import-cheap and CPU-safe in a worktree with no GPU and no downloads. Its only job
is to let an operator say `--model=deepseek-r1-distill-1.5b` instead of pasting the full HF repo id,
while STILL accepting any raw `org/repo` id as a pass-through so the registry is a convenience alias,
never a hard gate / whitelist.

Honest scope: this resolves a NAME to an HF id. It does not download, validate, or vouch for the
weights, and it does not whitelist what may be downloaded — a typo'd HF id fails later at download
time, not here. Serving a real DeepSeek/Qwen checkpoint and confirming its quality requires the human
to run the resolved id on a real GPU; nothing here makes any capability claim.

DeepSeek-R1-Distill-Qwen-1.5B/7B are Qwen2ForCausalLM checkpoints, so they load through
serve_real_model.load_causal_lm() (AutoModelForCausalLM) and shard through shard_serve.PipelineStage
with no model-specific code. This registry just gives them friendly names.
"""

# Friendly name -> Hugging Face org/repo id. Lower-case, hyphenated keys.
REGISTRY = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "deepseek-r1-distill-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1-distill-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

# One source of truth for the defaults serve_real_model exposes as DEFAULT_MODEL / FAST_MODEL.
DEFAULT_NAME = "qwen3-1.7b"
FAST_NAME = "qwen3-0.6b"


def list_models():
    """Sorted registry keys — for --help text and KeyError messages. Cheap, no side effects."""
    return sorted(REGISTRY)


def resolve_model(name_or_id):
    """Resolve a friendly registry name OR a raw HF id to a concrete HF org/repo id.

    Resolution order (so the registry is an ALIAS layer, never a hard gate):
      1. exact registry key  -> its HF id;
      2. anything containing '/' (looks like an HF `org/repo`) -> returned UNCHANGED, so the
         integrator can pass any HF id the registry has not heard of;
      3. otherwise (a bare, unknown name) -> KeyError listing the known names.

    The KeyError is deliberately ONLY for bare unknown names: a string with a '/' is assumed to be a
    real (possibly mistyped) HF id and is passed through, failing — if at all — at download time.
    """
    if not isinstance(name_or_id, str):
        raise TypeError(f"model name must be a str, got {type(name_or_id).__name__}")
    if name_or_id in REGISTRY:
        return REGISTRY[name_or_id]
    if "/" in name_or_id:
        return name_or_id  # raw HF id pass-through (not whitelisted on purpose)
    raise KeyError(
        f"unknown model name {name_or_id!r}; known names: {list_models()} "
        f"(or pass a raw HF id like 'org/repo')"
    )
