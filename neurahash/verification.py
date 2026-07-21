"""
Layer 3: verification. SCORECARD PROBLEM #3.
"How do we cheaply prove a worker actually did the training, not faked it?"

HARDENED after audit. The old version only replayed step 1, which let a lazy
worker do 1 real step (and skip the rest) or attach a fake delta to an empty
shard and still pass. We now bind the accepted delta to the FULL deterministic
trajectory:

  1. STAKE + SLASH      : every worker bonds collateral; cheating burns it.
  2. FULL RECOMPUTE     : the inner loop is deterministic given (global_params,
                          shard, trainable_keys, H, lr). The verifier replays ALL
                          H steps and requires the submitted delta to match the
                          recomputed delta exactly. This catches under-work (lazy
                          1-step), fabricated deltas, AND empty-shard fakes (whose
                          honest delta is exactly zero).
  3. NORM SANITY        : a cheap pre-filter rejects absurd/non-finite deltas.

COST NOTE: full recompute costs as much as the work itself. In production you do
this on a RANDOMLY SAMPLED subset of submissions (audit_prob below) or via a
Merkle-committed random-step challenge / zkML; the stake makes the expected value
of cheating negative even when only a fraction is audited. Here we audit everyone
so the demo is deterministic and airtight.
"""

import copy
import numpy as np
from .model import hash_params


def _recompute_trajectory(model, global_params, shard, trainable_keys, H, lr):
    """Replay the worker's full inner loop and return the expected delta."""
    X, y = shard
    keys = trainable_keys if trainable_keys is not None else list(global_params.keys())
    p = copy.deepcopy(global_params)
    for _ in range(H):
        if X.shape[0] > 0:
            _, cache = model.forward(X, y, params=p)
            grads = model.backward(cache, y, params=p)
            for k in keys:
                p[k] -= lr * grads[k]
    return {k: p[k] - global_params[k] for k in keys}


def recompute_step1(model, global_params, shard, trainable_keys=None, lr=0.5):
    """Kept for backward-compat / cheap pre-check: hash after exactly one step,
    over only the trainable keys (the role's responsibility)."""
    X, y = shard
    keys = trainable_keys if trainable_keys is not None else list(global_params.keys())
    p = copy.deepcopy(global_params)
    if X.shape[0] > 0:
        _, cache = model.forward(X, y, params=p)
        grads = model.backward(cache, y, params=p)
        for k in keys:
            p[k] -= lr * grads[k]
    return hash_params({k: p[k] for k in keys})


def check_submission(model, global_params, submission, shard,
                     H=30, lr=0.5, max_delta_norm=50.0, rtol=1e-6, atol=1e-8):
    """PURE verdict (no slashing): does the submission's delta match the full,
    deterministic H-step recompute? Returns (valid, reason). Because it is
    deterministic, any honest validator computes the same verdict -- which is what
    makes an M-of-N verifier quorum possible (see quorum.py)."""
    keys = submission.get("trainable_keys") or list(global_params.keys())

    dnorm = sum(float((v ** 2).sum()) for v in submission["delta"].values()) ** 0.5
    if not np.isfinite(dnorm) or dnorm > max_delta_norm:
        return False, f"delta norm {dnorm:.1f} out of bounds"

    expected = _recompute_trajectory(model, global_params, shard, keys, H, lr)
    if set(submission["delta"].keys()) != set(expected.keys()):
        return False, "delta touches wrong keys for its role"
    for k in expected:
        if not np.allclose(submission["delta"][k], expected[k], rtol=rtol, atol=atol):
            return False, f"delta != {H}-step recompute (under-work/fake)"
    return True, f"full {H}-step recompute matched, delta norm {dnorm:.2f}"


def verify_submission(model, global_params, submission, shard, ledger,
                      H=30, lr=0.5, max_delta_norm=50.0, rtol=1e-6, atol=1e-8):
    """Single-verifier path: check, and slash the bond on any detected fraud."""
    valid, reason = check_submission(model, global_params, submission, shard,
                                     H=H, lr=lr, max_delta_norm=max_delta_norm,
                                     rtol=rtol, atol=atol)
    if not valid:
        ledger.slash(submission["address"], reason=reason)
        return False, f"rejected: {reason} -> slashed"
    return True, f"verified: {reason}"
