"""
Layer 2: decentralized training with DiLoCo (Distributed Low-Communication).

This is what makes pooling internet GPUs physically possible. Instead of syncing
every step (needs datacenter interconnect), each worker trains LOCALLY for H steps
and only then shares a single delta. Communication drops ~H-fold.

A worker now has a ROLE decided by the capability router:
  - "train" : updates ALL params (needs enough VRAM for a full replica)
  - "host"  : updates only shared params + ONE expert (fits in less VRAM)
The set of weights it may touch is `trainable_keys`. Partial deltas from many
expert-hosts are merged per-key by the outer optimizer, so the network keeps
training even when no single GPU can hold the whole model.

The inner loop is DETERMINISTIC given (global_params, shard, trainable_keys), which
is what lets the Verification Layer cheaply recompute and catch cheaters.
"""

import copy
import numpy as np
from .model import sgd_step, hash_params


def _restricted_sgd(params, grads, lr, trainable_keys):
    for k in trainable_keys:
        params[k] -= lr * grads[k]


class Worker:
    def __init__(self, address, shard, honest=True, role="train",
                 expert_id=None, trainable_keys=None):
        self.address = address
        self.shard = shard            # (X, y)
        self.honest = honest
        self.role = role
        self.expert_id = expert_id
        self.trainable_keys = trainable_keys   # None -> all keys

    def train(self, model, global_params, H=30, lr=0.5):
        X, y = self.shard
        keys = self.trainable_keys if self.trainable_keys is not None else list(global_params.keys())

        if not self.honest:
            rng = np.random.default_rng(777)
            fake_delta = {k: rng.normal(0, 0.05, global_params[k].shape) for k in keys}
            return {"address": self.address, "delta": fake_delta,
                    "step1_hash": "deadbeef" * 8, "claimed_loss": 0.01,
                    "n_examples": int(X.shape[0]), "honest": False,
                    "role": self.role, "trainable_keys": keys}

        p = copy.deepcopy(global_params)
        step1_hash = None
        for step in range(H):
            if X.shape[0] > 0:
                _, cache = model.forward(X, y, params=p)
                grads = model.backward(cache, y, params=p)
                _restricted_sgd(p, grads, lr, keys)
            if step == 0:
                step1_hash = hash_params(p)   # commit to full state after 1 step

        delta = {k: p[k] - global_params[k] for k in keys}
        claimed = model.loss(X, y, params=p) if X.shape[0] > 0 else 0.0
        return {"address": self.address, "delta": delta, "step1_hash": step1_hash,
                "claimed_loss": float(claimed), "n_examples": int(X.shape[0]),
                "honest": True, "role": self.role, "trainable_keys": keys}


def outer_aggregate(global_params, deltas, momentum_buf, outer_lr=0.7, beta=0.9):
    """DiLoCo OUTER optimizer. Handles SPARSE deltas: each key is averaged only over
    the workers that actually trained it (full trainers + relevant expert-hosts)."""
    if not deltas:
        return global_params, momentum_buf
    for k in global_params:
        present = [d[k] for d in deltas if k in d]
        if not present:
            continue
        avg = np.mean(present, axis=0)
        # standard DiLoCo outer SGD-with-momentum: m = beta*m + avg; p += lr*m
        # (the old `(avg + beta*m)` double-counted the freshest delta)
        momentum_buf[k] = beta * momentum_buf.get(k, np.zeros_like(global_params[k])) + avg
        global_params[k] = global_params[k] + outer_lr * momentum_buf[k]
    return global_params, momentum_buf
