"""Minimal contributor-side checkpoint stub.

Exposes only plain torch load/save plus a path helper -- just enough to satisfy the
import surface a miner may reference. The F10 coordinator key-custody logic lives ONLY
in the private full-node package and is intentionally NOT reproduced here; this public
copy touches no secret material.
"""
import os

import torch

DEFAULT_STATE_DIR = "."


def checkpoint_path(state_dir=DEFAULT_STATE_DIR):
    """Path to the coordinator checkpoint .pt file inside `state_dir`."""
    return os.path.join(state_dir, "coord_checkpoint.pt")


def load_checkpoint(path, map_location="cpu"):
    """Plain torch.load of a checkpoint object. No secret material is touched."""
    return torch.load(path, map_location=map_location)


def save_checkpoint(obj, path):
    """Plain torch.save of a checkpoint object to `path`."""
    torch.save(obj, path)
