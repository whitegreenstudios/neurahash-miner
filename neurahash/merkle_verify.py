"""
Scalable verification #1: Merkle random-step challenge.

Full recompute (verification.py) re-runs all H inner steps — as expensive as the
work itself. This verifies a worker with only ~ONE step of recompute:

  - The worker commits a Merkle root over the hashes of every intermediate state
    s_0 (=global), s_1, ..., s_H (=local result), alongside the delta.
  - The verifier checks the ENDPOINT (committed s_H == global + delta) and then
    challenges a RANDOM step j: the worker reveals s_{j-1} (+ Merkle proofs), the
    verifier recomputes ONE step from it and checks the result hashes to the
    committed s_j.

A worker who skipped a fraction f of the steps has f·H inconsistent transitions, so
each challenge catches it with probability f; c challenges → 1-(1-f)^c. The work is
deterministic (full-batch GD), so an honest worker always passes. Cost: O(c) steps
to verify instead of O(H).
"""

import copy
import hashlib
import numpy as np


def _state_leaf(params, keys):
    h = hashlib.sha256()
    for k in sorted(keys):
        h.update(k.encode())
        h.update(np.ascontiguousarray(params[k], dtype=np.float64).tobytes())
    return h.digest()


def _data_leaf(shard):
    """Commitment to the exact training data, so a worker can't be verified against
    an easier shard than it was assigned."""
    X, y = shard
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.digest()


def _fiat_shamir_indices(root, data_hash, beacon, H, n):
    """Derive challenge step indices from the commitment (root + data) AND an external
    `beacon` revealed only AFTER the worker commits. Mixing the beacon defeats
    commitment-grinding: the worker cannot steer the indices because it doesn't know
    the beacon at commit time. The beacon must come from an unpredictable post-commit
    source (block hash / VRF / drand on-chain; here the verifier supplies fresh
    randomness)."""
    out = []
    for t in range(n):
        d = hashlib.sha256(root + data_hash + beacon + t.to_bytes(4, "big")).digest()
        out.append(int.from_bytes(d, "big") % H + 1)
    return out


def _hpair(a, b):
    return hashlib.sha256(a + b).digest()


def merkle_root(leaves):
    if not leaves:
        return b"\x00" * 32
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_hpair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves, idx):
    proof, level, i = [], list(leaves), idx
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sib = i ^ 1
        proof.append((level[sib], sib < i))      # (sibling, sibling_is_left)
        i //= 2
        level = [_hpair(level[k], level[k + 1]) for k in range(0, len(level), 2)]
    return proof


def merkle_verify(leaf, proof, root):
    h = leaf
    for sib, sib_is_left in proof:
        h = _hpair(sib, h) if sib_is_left else _hpair(h, sib)
    return h == root


def _one_step(model, params, shard, keys, lr, expert=None):
    """One deterministic full-batch GD step restricted to trainable keys.

    When `expert` is set, the step uses the SPARSE single-expert path (model.forward_expert /
    backward_expert): only the examples routed to that expert contribute, and the forward
    touches only {shared params, that one expert}. `keys` is then that expert's keys, so a
    verifier holding ONLY {shared, expert e} (other experts absent from `params`) recomputes
    the identical step — the MoE-sharding scaling property."""
    X, y = shard
    p = copy.deepcopy(params)
    if X.shape[0] > 0:
        if expert is None:
            _, cache = model.forward(X, y, params=p)
            grads = model.backward(cache, y, params=p)
        else:
            _, cache = model.forward_expert(X, y, expert, params=p)
            grads = model.backward_expert(cache, y, expert, params=p)
        for k in keys:
            p[k] -= lr * grads[k]
    return p


class MerkleWorker:
    """Trains and commits the full state trajectory so it can answer challenges.
    `skip_after`: if set (dishonest), it does that many real steps then fabricates
    the rest (the lazy-worker attack)."""
    def __init__(self, address, shard, honest=True, skip_after=None):
        self.address = address
        self.shard = shard
        self.honest = honest
        self.skip_after = skip_after
        self.states = None
        self.leaves = None

    def train(self, model, global_params, H=30, lr=0.5, trainable_keys=None, expert=None):
        keys = trainable_keys or list(global_params.keys())
        p = copy.deepcopy(global_params)
        states = [copy.deepcopy(p)]
        leaves = [_state_leaf(p, keys)]
        for step in range(H):
            if self.honest or self.skip_after is None or step < self.skip_after:
                p = _one_step(model, p, self.shard, keys, lr, expert=expert)
            else:
                # lazy: skip the real update (transition becomes identity -> detectable)
                p = copy.deepcopy(p)
            states.append(p)
            leaves.append(_state_leaf(p, keys))
        self.states, self.leaves = states, leaves
        delta = {k: states[-1][k] - global_params[k] for k in keys}
        return {"address": self.address, "delta": delta, "root": merkle_root(leaves),
                "H": H, "lr": lr, "n_examples": int(self.shard[0].shape[0]),
                "data_hash": _data_leaf(self.shard).hex(), "trainable_keys": keys,
                "expert": expert}

    def state(self, idx):
        return self.states[idx]

    def proof(self, idx):
        return merkle_proof(self.leaves, idx)


def merkle_check(model, global_params, submission, prover, shard, n_challenges=2,
                 lr=0.5, H=None, beacon=b"", rng=None):
    """Verify a submission with ENDPOINT + n Fiat-Shamir single-step challenges.
    Returns (valid, reason). Cost ~ n_challenges steps, not H.

    H is the PROTOCOL step count supplied by the verifier; if given, a submission whose
    declared H differs is rejected (a worker cannot hide under-work by claiming a
    smaller H). `rng` is accepted for backward-compat but ignored — challenge indices
    are derived from the commitment (Fiat-Shamir)."""
    keys = submission["trainable_keys"]
    expert = submission.get("expert")
    if H is not None and submission.get("H") != H:
        return False, "declared H != protocol H (under-work)"
    H = submission["H"] if H is None else H
    root = submission["root"]

    # The revealed trajectory and the endpoint check below only validate the `keys` columns,
    # so a delta carrying EXTRA keys would be applied to live params (consensus._advance_params
    # adds delta[k] for every k present) yet never validated against the committed work. In
    # expert-sharded mode that is a consensus break: an UNVALIDATED extra key lets a proposer
    # perturb shared params or another expert (which the single-expert checkpoint proof cannot
    # see), corrupting canonical state. The honest worker always builds delta over exactly
    # `keys` (MerkleWorker.train), so require the delta key-set to match exactly.
    if set(submission.get("delta", {}).keys()) != set(keys):
        return False, "delta keys != trainable_keys (unvalidated/extra delta entries)"

    def _recompute_step(s_prev):
        # In expert-sharded mode the revealed per-step states carry ONLY the trainable expert
        # keys; the FROZEN shared params (Emb/Wr/Wo/bo) come from global_params, which the
        # verifier holds for {shared + this one expert} but NOT the other experts. Merge so the
        # single-expert forward has what it needs, then recompute exactly one sparse step.
        if expert is None:
            return _one_step(model, s_prev, shard, keys, lr)
        merged = {**global_params, **s_prev}
        return _one_step(model, merged, shard, keys, lr, expert=expert)

    # bind the work to the assigned data: the shard must match the committed data_hash
    if submission.get("data_hash") != _data_leaf(shard).hex():
        return False, "shard does not match committed data_hash"
    if submission.get("n_examples") != int(shard[0].shape[0]):
        return False, "n_examples does not match shard"

    # endpoint: the committed final state s_H must (a) be in the commitment and
    # (b) match the submitted delta. We compare delta to (s_H - global) directly
    # rather than reconstructing global+delta, because floating-point addition is
    # non-associative (global + (s_H - global) != s_H exactly).
    s_H = prover.state(H)
    if not merkle_verify(_state_leaf(s_H, keys), prover.proof(H), root):
        return False, "committed final state not in Merkle root"
    for k in keys:
        if not np.array_equal(s_H[k] - global_params[k], submission["delta"][k]):
            return False, "delta does not match committed final state"

    # Fiat-Shamir single-step challenges (indices fixed by commitment + post-commit beacon)
    for j in _fiat_shamir_indices(root, bytes.fromhex(submission["data_hash"]),
                                  beacon, H, n_challenges):
        s_prev = prover.state(j - 1)
        if not merkle_verify(_state_leaf(s_prev, keys), prover.proof(j - 1), root):
            return False, f"revealed state s_{j-1} not in commitment"
        s_j = _recompute_step(s_prev)
        if not merkle_verify(_state_leaf(s_j, keys), prover.proof(j), root):
            return False, f"step {j} transition inconsistent (work skipped/faked)"
    return True, f"endpoint + {n_challenges} step-challenges passed"


def detection_probability(skip_fraction, n_challenges):
    """P(catch | audited) for a worker who skipped `skip_fraction` of steps."""
    return 1.0 - (1.0 - skip_fraction) ** n_challenges
