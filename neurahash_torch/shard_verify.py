"""
shard_verify.py — Milestone 4: verifiable single-tile transitions on the REAL torch model.

M3 let a node HOLD a subset of experts; M4 lets a verifier CHECK one expert's update without
holding the rest of the 671B model. It ports neura_l1.expert_sharding's param-group Merkle
checkpoint from the NumPy toy (flat expert list, fp64) to the torch MoETransformer, generalized
to a (layer, expert) TILE GRID:

    leaf 0                     = "trunk"  (everything except per-expert FFNs: embeddings,
                                            attention, layernorms, routers, head)
    leaf 1 + layer*E + expert  = "tile_{layer}_{expert}"  (that one expert FFN: W1,b1,W2,b2)

A block that trains ONE (layer, expert) tile changes ONLY that leaf. A verifier holding just
{that tile's params} computes that one leaf and — with the block's published Merkle proof
(O(log tiles) sibling hashes, NOT the other 256×60 tiles) — checks the OLD tile leaf rolls up
to the trusted parent checkpoint AND the NEW tile leaf rolls up to the header's new checkpoint.
So a single expert's full-model state transition is verified from one tile + a tiny proof.

Leaves use a CANONICAL fp32 byte recipe (the training dtype, pinned by train_torch), so the same
weights hash identically on any vendor's GPU — the cross-hardware reproducibility the design needs
for the leaf layer (the kernel-level determinism for the *training step itself* is M5/M10).
"""

import hashlib

import numpy as np
import torch

from neurahash.merkle_verify import merkle_root, merkle_proof
from neura_l1.expert_sharding import _pad_pow2, leaf_rolls_up_to   # generic over leaf bytes


def canonical_bytes(t):
    """Vendor-reproducible bytes for a tensor: contiguous little-endian fp32. fp32 is the pinned
    training dtype, so the SAME weights produce the SAME bytes on any GPU/CPU."""
    return np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy()).tobytes()


def _group_leaf(state, keys, name):
    """32-byte leaf committing a param group: H(name ‖ for k in sorted(keys): k ‖ fp32 bytes).
    Same recipe as expert_sharding._group_leaf, on torch tensors."""
    h = hashlib.sha256(name.encode())
    for k in sorted(keys):
        h.update(k.encode())
        h.update(canonical_bytes(state[k]))
    return h.digest()


def trunk_keys(state):
    """Every param EXCEPT per-expert FFNs — the shared trunk (embeddings, attention, layernorms,
    routers, head). This is leaf 0 and is what every node replicates."""
    return [k for k in state if ".moe.experts." not in k]


def tile_keys(layer, expert):
    """The four param keys of one expert FFN in one layer (the sparse-MoE layout from M3)."""
    return [f"blocks.{layer}.moe.experts.{expert}.{p}" for p in ("W1", "b1", "W2", "b2")]


def trunk_leaf(state):
    return _group_leaf(state, trunk_keys(state), "trunk")


def tile_leaf(state, layer, expert):
    """The group leaf for one (layer, expert) tile — computable from that tile's 4 params alone."""
    return _group_leaf(state, tile_keys(layer, expert), f"tile_{layer}_{expert}")


def _all_leaves(state, n_layers, n_experts):
    leaves = [trunk_leaf(state)]
    for layer in range(int(n_layers)):
        for expert in range(int(n_experts)):
            leaves.append(tile_leaf(state, layer, expert))
    return leaves


def tile_index(layer, expert, n_experts):
    """Leaf index of a tile in the grid (leaf 0 is the trunk)."""
    return 1 + int(layer) * int(n_experts) + int(expert)


def checkpoint_root(state, n_layers, n_experts):
    """Canonical sharded checkpoint: hex Merkle root over the (pow2-padded) [trunk, tiles...]."""
    return merkle_root(_pad_pow2(_all_leaves(state, n_layers, n_experts))).hex()


def checkpoint_proof(state, n_layers, n_experts, layer, expert):
    """Merkle proof (sibling hashes) for one tile's leaf — published in the block so a verifier
    holding only that tile rolls its own leaf up to the root. Padded to a power of two so the
    proof is valid for BOTH the pre- and post-update tile leaf (siblings unchanged)."""
    leaves = _pad_pow2(_all_leaves(state, n_layers, n_experts))
    idx = tile_index(layer, expert, n_experts)
    return [[sib.hex(), bool(is_left)] for sib, is_left in merkle_proof(leaves, idx)]


def verify_tile_transition(parent_root, new_root, layer, expert,
                           old_tile_state, new_tile_state, proof):
    """Sharded check: the published `proof` must validate the OLD tile leaf against the trusted
    `parent_root` (authenticating the siblings) AND the NEW tile leaf against `new_root` with the
    SAME siblings. `*_tile_state` need only hold that tile's 4 keys. If ANY other tile or the
    trunk also changed, the siblings differ between the two trees and this fails — so it proves
    EXACTLY this tile changed. Returns (ok, reason)."""
    old_leaf = tile_leaf(old_tile_state, layer, expert)
    if not leaf_rolls_up_to(old_leaf, proof, parent_root):
        return False, "old tile leaf does not roll up to parent checkpoint (bad proof)"
    new_leaf = tile_leaf(new_tile_state, layer, expert)
    if not leaf_rolls_up_to(new_leaf, proof, new_root):
        return False, "new tile leaf does not roll up to new checkpoint (other tiles/trunk changed?)"
    return True, "tile transition verified"
