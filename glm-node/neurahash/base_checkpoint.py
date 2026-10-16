"""
neurahash/base_checkpoint.py -- model-only checkpoint persistence for the ALL-OUTBOUND public miner.

This is the public-safe variant of the coordinator's checkpoint format: it persists ONLY the model
(trunk + experts), the DiLoCo aggregation target (global_trunk), the round counter, the arch/vocab
guards, and the dashboard stats -- and deliberately carries NO signed economic ledger. The miner never
needs the ledger: it fetches a base (tools/make_base_from_hf.py), trains the trunk, and publishes a
small compressed DELTA -- it never writes an authoritative coordinator checkpoint. Dropping the ledger
means this module imports NO economics/consensus core (no neurahash.pool_ledger), and the on-disk blob
never embeds a coordinator signing key.

WHAT IS PERSISTED (one self-consistent file):
  * model_state  -- the full model state_dict (trunk + every expert);
  * global_trunk -- the DiLoCo aggregation target (kept separately so load is exact);
  * E            -- the expert count (grows with the fleet), so the model is rebuilt at the right width;
  * round        -- the round counter;
  * stats        -- accounting/dashboard state (opaque dict);
  * arch0/vocab  -- the architecture + vocab the weights belong to, CHECKED on load so a checkpoint can
                    never be loaded into an incompatible model (silent shape-mismatch corruption).

The model save/load contract matches the coordinator's exactly (same field names, same atomic-write and
rebuild-then-load_state_dict behavior), so a base written by make_base_from_hf loads cleanly in
tools/diloco_contributor. `save_checkpoint` accepts an optional `ledger` argument and `load_checkpoint`
returns `ledger=None` purely so the shared contributor call sites round-trip unchanged; the ledger is
never serialized, deserialized, or imported.
"""

import os

import torch

from neurahash_torch.shard_verify import trunk_keys

CKPT_VERSION = 2
DEFAULT_STATE_DIR = "_state"
CKPT_NAME = "coord_checkpoint.pt"


def checkpoint_path(state_dir=DEFAULT_STATE_DIR):
    return os.path.join(state_dir, CKPT_NAME)


def _atomic_save(obj, path):
    """Write `obj` to `path` so a crash mid-write never corrupts the existing checkpoint: serialize to
    a temp file in the SAME directory (so os.replace is a same-filesystem rename), flush+fsync to get
    the bytes on disk, then os.replace (atomic on POSIX + Windows)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _cpu_detach(t):
    """Return a CPU, detached tensor SHARING storage when the source is already a detached CPU tensor,
    so building the checkpoint blob does NOT clone a second full copy of the model in RAM.
      * already CPU + no autograd  -> t itself (zero extra bytes; torch.save reads the live storage);
      * CPU but requires_grad/grad -> .detach() (a view, still shares storage, breaks the autograd ref);
      * on a GPU host              -> .detach().cpu() (a genuine device->host copy is unavoidable)."""
    if t.device.type != "cpu":
        return t.detach().cpu()
    return t if not (t.requires_grad or t.grad_fn is not None) else t.detach()


def save_checkpoint(path, *, model, global_trunk, E, round, stats, arch0, vocab, ledger=None):
    """Persist one self-consistent model-only checkpoint atomically. Tensors are moved to CPU so the
    file loads on a host with a different (or no) GPU.

    `ledger` is accepted but IGNORED (never serialized, never imported): the model-only format carries
    no signed economic ledger. It exists only so the shared tools/diloco_contributor call site
    (which passes the round-tripped `loaded['ledger']`, always None here) works unchanged."""
    blob = {
        "version": CKPT_VERSION,
        "arch0": dict(arch0),
        "vocab": int(vocab),
        "E": int(E),
        "round": int(round),
        "model_state": {k: _cpu_detach(v) for k, v in model.state_dict().items()},
        "global_trunk": {k: _cpu_detach(v) for k, v in global_trunk.items()},
        "stats": stats,
    }
    _atomic_save(blob, path)
    return path


def _build_model(arch0, vocab, E, device):
    """Rebuild the model at the (grown) expert count E. Routes through the shared pool_model factory so a
    DENSE base (arch0['kind']=='qwen') rebuilds a QwenBackbone instead of the toy MoE; load_base=False
    because the checkpoint's TRAINED weights are loaded on top right after (the base would just be
    overwritten). For the toy MoE this is byte-for-byte the previous construction."""
    from neurahash_torch.pool_model import build_pool_model
    arch = {**arch0, "n_experts": int(E)}
    return build_pool_model(arch, int(vocab), list(range(int(E))), device, load_base=False)


def load_checkpoint(path, device="cpu", *, expect_vocab=None, expect_arch0=None):
    """Load a model-only checkpoint and rebuild the live state. Returns a dict with keys
    {model, global_trunk, E, round, ledger, stats, arch0, vocab}, or None if no checkpoint exists.
    `ledger` is always None here (the model-only format carries no ledger); it is present only so the
    shared contributor code round-trips the field unchanged.

    Guards (fail LOUD, never silently corrupt):
      * version mismatch -> None (treat as no resumable checkpoint, start fresh);
      * vocab / core-arch mismatch vs the running config -> ValueError (refuse to load weights into an
        incompatible model).
    """
    if not os.path.exists(path):
        return None
    # weights_only=False: this is a local model file (the miner's own base), holding plain dicts/lists
    # alongside tensors, which the weights_only allowlist would reject.
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("version") != CKPT_VERSION:
        return None
    vocab = int(blob["vocab"])
    arch0 = dict(blob["arch0"])
    if expect_vocab is not None and int(expect_vocab) != vocab:
        raise ValueError(f"checkpoint vocab {vocab} != running vocab {expect_vocab}; refusing to "
                         f"load weights into an incompatible model")
    if expect_arch0 is not None:
        # n_experts is allowed to differ (E grows with the fleet); everything else must match.
        for k, v in expect_arch0.items():
            if k == "n_experts":
                continue
            if arch0.get(k) != v:
                raise ValueError(f"checkpoint arch0[{k}]={arch0.get(k)} != running {v}; refusing to "
                                 f"load weights into an incompatible model")
    E = int(blob["E"])
    model = _build_model(arch0, vocab, E, device)
    model.load_state_dict({k: v.to(device) for k, v in blob["model_state"].items()})
    global_trunk = {k: blob["global_trunk"][k].to(device) for k in blob["global_trunk"]}
    # sanity: the persisted trunk must cover exactly the model's trunk keys
    tk = set(trunk_keys(model.state_dict()))
    if set(global_trunk) != tk:
        raise ValueError(f"checkpoint trunk keys do not match the rebuilt model "
                         f"({len(global_trunk)} vs {len(tk)} keys)")
    return {
        "model": model,
        "global_trunk": global_trunk,
        "E": E,
        "round": int(blob["round"]),
        "ledger": None,        # model-only checkpoint carries no signed ledger (passthrough field only)
        "stats": blob.get("stats", {}),
        "arch0": arch0,
        "vocab": vocab,
    }
