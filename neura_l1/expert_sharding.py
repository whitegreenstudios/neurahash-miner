"""
neura_l1.expert_sharding — protocol-assigned MoE expert sharding: the build where total model
capacity scales with the FLEET while each miner's train/verify cost stays bounded.

The model (neurahash.model.MoELM) is a Mixture-of-Experts: SHARED params (embedding, router,
output projection) plus E independent expert FFNs (`expert_keys(e)`), and it grows by appending
experts (`add_expert`). A dense/soft MoE couples every expert in the loss, so verifying any
update needs ALL experts — per-node cost grows O(E). This module makes training/verification
SPARSE and per-expert:

  * ASSIGNMENT — each block trains ONE expert chosen by the per-block beacon
    (sha256(prev_hash ‖ proposer)), exactly like the beacon-assigned data shard. A proposer
    cannot pick which expert it trains (anti-grinding) and every verifier re-derives it.
  * SPARSE STEP — `MoELM.forward_expert`/`backward_expert` (top-1 routing) compute the step for
    the assigned expert touching ONLY {shared params, that one expert}. The Merkle PoUW recompute
    (`merkle_verify._one_step(..., expert=e)`) inherits this, so a verifier holding ONLY
    {shared, expert e} — NOT the other E-1 experts — re-checks the block. As E (and the fleet)
    grows, the dominant O(E·expert) storage/compute drops to O(1·expert) per verifier; only the
    small shared router grows O(E).
  * SCORE — the work score is the held-out single-expert loss improvement (`MoELM.expert_loss`),
    likewise computable from {shared, expert e} alone.

This composes with the rest of the stack: the beacon assignment is the same primitive as
`data_assignment`; the assigned data can ride the `data_availability` layer; and growing the
expert count is a natural `Consensus.migrations` step for the live protocol-upgrade machinery
(open thread #8) — `add_expert` IS the deterministic checkpoint migration F(P_old).
"""

import hashlib

import numpy as np

from neurahash.model import SHARED_KEYS, expert_keys, default_expert_seed
from neurahash.merkle_verify import merkle_root, merkle_proof, merkle_verify


def assign_expert(beacon, n_experts):
    """The protocol-assigned expert index for a block, derived from its beacon
    (= pouw_gate.beacon_for(prev_hash, proposer)). Deterministic + uniform over [0, n_experts);
    unpredictable until the parent exists (anti-grinding), recomputable by every verifier.
    `beacon` may be bytes or a hex string."""
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    if isinstance(beacon, str):
        beacon = bytes.fromhex(beacon)
    h = hashlib.sha256(b"expert-assign|" + bytes(beacon)).digest()
    return int.from_bytes(h[:8], "big") % int(n_experts)


def expert_view_keys(e):
    """The param keys a sharded node needs to train/verify expert `e`: the SHARED params
    (embedding, router, output projection) plus that one expert's FFN keys. NOT the other
    experts — that omission is the whole point (bounded per-node cost as E grows)."""
    return list(SHARED_KEYS) + list(expert_keys(e))


def expert_view(params, e):
    """Slice a full param dict down to the sharded view for expert `e`: {shared, expert e}.
    A verifier given only this view can run forward_expert / the sparse Merkle recompute /
    expert_loss for blocks assigned to e, with the other experts entirely absent. Raises
    KeyError if a required key is missing (fail-loud: an incomplete view must not silently
    verify)."""
    return {k: params[k] for k in expert_view_keys(e)}


def view_param_count(params, e):
    """Number of scalar params in the sharded view for expert e (what a sharded verifier
    actually holds) — for asserting it does NOT grow with the total expert count."""
    return int(sum(params[k].size for k in expert_view_keys(e) if k in params))


# ===========================================================================
# Param-group Merkle checkpoint — so a SHARDED verifier can check the model-state
# transition of one expert without holding the other experts.
#
# The canonical model checkpoint is a Merkle root over the param GROUPS, in a fixed order:
#     leaf 0      = group "shared"   (SHARED_KEYS: Emb, Wr, br, Wo, bo)
#     leaf 1+e    = group "expert_e" (W1_e, b1_e, W2_e, b2_e)
# A block that trains expert e changes ONLY leaf 1+e. A verifier holding {shared, expert e}
# computes that one leaf itself, and — given the block's published Merkle proof (the sibling
# group-hashes, O(log E) bytes, NOT the other experts' params) — checks:
#   (a) its OLD expert-e leaf rolls up to the (trusted) parent checkpoint, validating the
#       published siblings against a root it already trusts (so they can't be forged), and
#   (b) its NEW expert-e leaf rolls up, with the SAME siblings, to the header's new checkpoint.
# Thus the full-model transition is verified from {shared, expert e} + a tiny proof.
# (Dense/default mode keeps using neurahash.model.hash_params over the whole model.)
# ===========================================================================
def _group_leaf(params, keys, name):
    """A 32-byte leaf committing a param group: H(name ‖ for k in sorted(keys): k ‖ float64
    bytes). Same byte recipe as model.hash_params, so it is deterministic across nodes."""
    h = hashlib.sha256(name.encode())
    for k in sorted(keys):
        h.update(k.encode())
        h.update(np.ascontiguousarray(params[k], dtype=np.float64).tobytes())
    return h.digest()


# A fixed sentinel leaf used to pad the group-leaf list to a power of two (see _pad_pow2).
_GROUP_PAD = hashlib.sha256(b"neurahash-group-pad").digest()


def _real_group_leaves(params, n_experts):
    """The REAL param-group leaves [shared, expert_0 .. expert_{n-1}] (n_experts+1 leaves),
    UNPADDED. This is the canonical leaf set that gets published/passed around; padding is an
    internal detail of root/proof computation (_pad_pow2)."""
    leaves = [_group_leaf(params, SHARED_KEYS, "shared")]
    for e in range(int(n_experts)):
        leaves.append(_group_leaf(params, expert_keys(e), f"expert_{e}"))
    return leaves


def _pad_pow2(leaves):
    """Pad a leaf list to the next power of two with a FIXED sentinel, so the Merkle tree is a
    PERFECT binary tree: every leaf has a distinct, stable sibling path and NO node is ever
    duplicated. This is what makes a single-leaf-change proof (verify_expert_transition reuses
    one proof for the OLD and NEW expert leaf) sound — under merkle_root's odd-level
    'duplicate last' rule a changed leaf sitting in a self-duplicated position would invalidate
    the reused proof (its sibling is a copy of itself, which also changes). The sentinels never
    change, so the published proof stays valid across the expert's transition."""
    n = len(leaves)
    size = 1
    while size < n:
        size <<= 1
    return list(leaves) + [_GROUP_PAD] * (size - n)


def checkpoint_root(params, n_experts):
    """The canonical sharded model checkpoint: hex Merkle root over the (power-of-two padded)
    param-group leaves."""
    return merkle_root(_pad_pow2(_real_group_leaves(params, n_experts))).hex()


def expert_leaf(params, e):
    """The group leaf for expert e — what a sharded node computes from {expert e} alone."""
    return _group_leaf(params, expert_keys(e), f"expert_{e}")


def checkpoint_proof(params, n_experts, e):
    """The Merkle proof (sibling group-hashes) for expert e's leaf, JSON-safe
    [[sibling_hex, sib_is_left], ...]. Published in the block so a sharded verifier rolls its
    own expert-e leaf up to the root without the other experts' params. Expert e sits at leaf
    index e+1 (leaf 0 is the shared group); the tree is padded to a power of two so the proof is
    valid for both the pre- and post-update expert leaf."""
    leaves = _pad_pow2(_real_group_leaves(params, n_experts))
    return [[sib.hex(), bool(is_left)] for sib, is_left in merkle_proof(leaves, int(e) + 1)]


# A group-proof has O(log2 leaves) siblings; this cap supports >2^60 experts, so it never
# constrains a real proof -- it is purely a DoS bound on the untrusted gossiped proof list (a
# wrong-LENGTH proof is already a correctness reject because it cannot reconstruct the root; this
# stops a multi-million-entry proof from being hex-decoded/hashed before that rejection).
_MAX_GROUP_PROOF = 64


def leaf_rolls_up_to(leaf, proof, root_hex):
    """True iff `leaf` + the published Merkle `proof` reconstruct `root_hex`. The proof arrives
    on untrusted gossip, so its SHAPE is bounded before any decode/hash work: at most
    _MAX_GROUP_PROOF siblings, each a 32-byte (64 hex char) string."""
    try:
        if not isinstance(proof, (list, tuple)) or len(proof) > _MAX_GROUP_PROOF:
            return False
        if not isinstance(root_hex, str) or len(root_hex) != 64:
            return False
        p = []
        for sib, is_left in proof:
            if not isinstance(sib, str) or len(sib) != 64:
                return False
            p.append((bytes.fromhex(sib), bool(is_left)))
        return merkle_verify(leaf, p, bytes.fromhex(root_hex))
    except Exception:
        return False


def verify_expert_transition(parent_root, new_root, e, old_expert_params, new_expert_params,
                             proof):
    """Sharded checkpoint check: the published `proof` must validate the OLD expert-e leaf
    against the trusted `parent_root` (so the siblings are authentic) AND the NEW expert-e leaf
    against `new_root` with the same siblings. `*_expert_params` need only hold expert e's keys
    (+ ignore the rest). Returns (ok, reason)."""
    old_leaf = expert_leaf(old_expert_params, e)
    if not leaf_rolls_up_to(old_leaf, proof, parent_root):
        return False, "old expert leaf does not roll up to parent checkpoint (bad proof)"
    new_leaf = expert_leaf(new_expert_params, e)
    if not leaf_rolls_up_to(new_leaf, proof, new_root):
        return False, "new expert leaf does not roll up to new checkpoint"
    return True, "expert transition verified"


# ===========================================================================
# Expert GROWTH as a live checkpoint migration — the model gets BIGGER on a running chain.
#
# A protocol upgrade can grow the expert count: F(P_old) appends new experts, replicating
# MoELM.add_expert byte-for-byte (so every node computes the identical grown params). This is
# the deterministic Consensus.migrations step that pairs the expert-sharding build with the
# live protocol-upgrade machinery — total model capacity climbs with the fleet, while each
# miner's train/verify cost stays bounded to {shared, ONE expert}.
#
# Growth re-roots the param-group Merkle checkpoint (the shared/router leaf changes and new
# expert leaves are appended), so a sharded verifier — which never holds every expert — cannot
# recompute the new root alone at the boundary. The boundary block therefore publishes the
# (tiny, O(E)·32-byte) list of OLD group-leaf hashes; any verifier authenticates them against
# the TRUSTED old root and re-derives the migrated root from {its own shared params (router
# grown deterministically), the deterministically-created new expert leaves, the authenticated
# old expert leaves carried over unchanged}. See verify_migration_root.
# ===========================================================================
def _infer_expert_dims(params):
    """(H, Do) inferred from any expert FFN present (uniform across experts): W1 is (Din, H),
    W2 is (H, Do). Raises if no expert is present (a view with zero experts cannot be grown)."""
    for k in params:
        if k.startswith("W1_"):
            e = k[len("W1_"):]
            w1, w2 = params.get(f"W1_{e}"), params.get(f"W2_{e}")
            if w1 is not None and w2 is not None:
                return int(w1.shape[1]), int(w2.shape[1])
    raise ValueError("cannot infer expert dims: no expert FFN present in params")


def grow_experts(params, old_n, new_n, dims=None):
    """Deterministically append experts old_n..new_n-1, replicating MoELM.add_expert EXACTLY
    (fresh rng seeded by model.default_expert_seed(e) per expert; W1,W2 ~ N(0, 0.2); biases zero;
    one N(0, 0.05) router column appended to Wr per new expert, br extended by zero). Pure —
    returns new params, the input untouched. Existing groups (shared + old experts) carry through
    byte-identically; only Wr/br grow and the new expert FFNs are added. This is the F(P_old) that
    grows the model.

    `dims` is the protocol-fixed (H, Do) of every expert FFN. When given, the new experts are
    materialized WITHOUT inspecting any held expert (so a verifier holding only {shared} — or
    {shared, a not-yet-existing new expert} — can still recompute the migration at the boundary);
    when None it is inferred from a held expert FFN (back-compat for direct callers). Din always
    comes from the shared router Wr, which every holder has."""
    out = {k: np.array(v, copy=True) for k, v in params.items()}
    if int(new_n) < int(old_n):
        raise ValueError(f"grow_experts cannot shrink the expert count ({old_n} -> {new_n})")
    if int(new_n) == int(old_n):
        return out
    din = out["Wr"].shape[0]
    h, do = (int(dims[0]), int(dims[1])) if dims is not None else _infer_expert_dims(out)
    for e in range(int(old_n), int(new_n)):
        rng = np.random.default_rng(default_expert_seed(e))   # the ONE seed rule add_expert uses
        out[f"W1_{e}"] = rng.normal(0, 0.2, (din, h))
        out[f"b1_{e}"] = np.zeros(h)
        out[f"W2_{e}"] = rng.normal(0, 0.2, (h, do))
        out[f"b2_{e}"] = np.zeros(do)
        new_col = rng.normal(0, 0.05, (din, 1))
        out["Wr"] = np.concatenate([out["Wr"], new_col], axis=1)
        out["br"] = np.concatenate([out["br"], np.zeros(1)])
    return out


def group_leaves_hex(params, n_experts):
    """The ordered REAL group-leaf hashes [shared, expert_0 .. expert_{n-1}] as hex strings
    (UNPADDED) — the canonical leaf set whose padded Merkle root IS checkpoint_root(params,
    n_experts). An upgrade-boundary block publishes this (as pouw['migration_leaves']) so any
    verifier can authenticate the pre-migration leaves against the trusted parent root and
    derive the migrated root without holding every expert's params."""
    return [leaf.hex() for leaf in _real_group_leaves(params, int(n_experts))]


def verify_migration_root(parent_root, new_root, old_leaves_hex, migrated_params,
                          old_n, new_n):
    """Boundary checkpoint check for an expert-growth upgrade. The published `old_leaves_hex`
    (pre-migration REAL group leaves, unpadded) must Merkle-root (padded) to the TRUSTED
    `parent_root`; the migrated root is then derived deterministically — the shared leaf and any
    NEW expert leaves are recomputed from `migrated_params` (held post-migration: shared + the
    verifier's expert(s) + the deterministically-created new experts), while the OLD experts'
    leaves carry over UNCHANGED from the authenticated list — and must equal the header's
    `new_root`. A proposer cannot forge it: faked old leaves would not root to parent_root, and
    the migrated leaves are a pure function of the (authenticated) old leaves + the
    protocol-fixed growth. (ok, reason)."""
    # `old_leaves_hex` is untrusted gossip; bound its SHAPE before decoding so an oversized list
    # is a cheap O(1) reject rather than a multi-million-entry bytes.fromhex pass. The exact count
    # is protocol-fixed (one shared leaf + old_n expert leaves), so enforce it up front.
    if not isinstance(old_leaves_hex, (list, tuple)) or len(old_leaves_hex) != int(old_n) + 1:
        return False, (f"migration_leaves count {len(old_leaves_hex) if hasattr(old_leaves_hex, '__len__') else '?'} "
                       f"!= old group count {int(old_n) + 1}")
    try:
        old_leaves = []
        for x in old_leaves_hex:
            if not isinstance(x, str) or len(x) != 64:
                return False, "malformed migration_leaves (each leaf must be 32-byte hex)"
            old_leaves.append(bytes.fromhex(x))
    except (ValueError, TypeError):
        return False, "malformed migration_leaves"
    if merkle_root(_pad_pow2(old_leaves)).hex() != parent_root:
        return False, "migration_leaves do not match parent checkpoint (not authentic)"
    try:
        new_shared = _group_leaf(migrated_params, SHARED_KEYS, "shared")
        new_expert_leaves = [expert_leaf(migrated_params, e)
                             for e in range(int(old_n), int(new_n))]
    except KeyError as ex:
        return False, f"migrated params missing group for {ex}"
    new_leaves = [new_shared] + old_leaves[1:] + new_expert_leaves
    if merkle_root(_pad_pow2(new_leaves)).hex() != new_root:
        return False, "derived migrated root != header parent_checkpoint"
    return True, "migration root verified"
