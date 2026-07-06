"""
gpu_pool_node — PUBLIC MINER TRIM.

The full node's gpu_pool_node module carries coordinator/orchestration helpers that the public
miner does not ship. This trimmed copy exposes ONLY `_seed`: the pure, deterministic per-round
PRNG seed the worker commits to and the verifier replays. It has no secret and no private import
(stdlib `hashlib` only), so it is safe for the public miner repo.

Kept public-safe: a worker imports `_seed` from here (via neurahash.worker_core) to seed its
per-round batch draw. The coordinator uses the byte-identical function to replay the same draw —
the property recompute-verify relies on.
"""
import hashlib


def _seed(round_id, address):
    """Per-round batch seed a worker commits to; the verifier replays with the SAME seed (taken
    from the submission) so an honest trajectory reproduces exactly."""
    h = int(hashlib.sha256(address.encode()).hexdigest(), 16) % 100000
    return int(round_id) * 100003 + h
