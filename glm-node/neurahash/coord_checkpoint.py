"""Contributor-side checkpoint surface — a thin re-export of `neurahash.base_checkpoint`.

WHY THIS IS A RE-EXPORT AND NOT AN IMPLEMENTATION. `tools/diloco_contributor.py` does

    from neurahash.coord_checkpoint import load_checkpoint, save_checkpoint, checkpoint_path

in BOTH repos, so the public copy must expose the same three names with the same signatures the
private full node does. It previously carried a hand-written stub whose `load_checkpoint(path,
map_location="cpu")` was a plain `torch.load` — the wrong shape entirely. Nothing exercised it, so
the mismatch shipped, and every public miner died before its first training step:

    File "tools/diloco_contributor.py", line 117, in train_contribution
      loaded = load_checkpoint(ckpt_path, device=device)
    TypeError: load_checkpoint() got an unexpected keyword argument 'device'

(MEASURED 2026-07-21 on a fresh public clone, on the real cold-start path, after the 5.39 GB base
built fine — a total blocker for a stranger on any platform, unrelated to tokens or URLs.)

The REAL implementation already ships publicly in `neurahash/base_checkpoint.py` with the same
signatures as the private module: it rebuilds the live model state, honours `device`, and supports
the `expect_vocab` / `expect_arch0` guards. Re-exporting keeps ONE implementation instead of two
that can silently drift apart again.

The F10 coordinator key-custody logic stays out of the public tree deliberately: `base_checkpoint`
is model-state only and touches no secret material, which is exactly the subset a contributor needs.
"""
from neurahash.base_checkpoint import (       # noqa: F401  (re-exported public surface)
    DEFAULT_STATE_DIR,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)

__all__ = ["DEFAULT_STATE_DIR", "checkpoint_path", "load_checkpoint", "save_checkpoint"]
