"""neurahash/identity.py -- the key-bound worker-identity sign payload (LEAF; no ledger/consensus deps).

Extracted VERBATIM from pool_ledger.py so client modules (pqc_admission, the miner) can build the
join-signature payload without importing the economics-bearing pool_ledger. Imports only the pure
canon leaf.
"""
from neurahash.canon import _canon


def identity_payload(address, round_hint=0):
    """The bytes a worker SIGNS to prove it controls `address` when joining (bound to the address so
    the signature can't be reused for a different identity)."""
    return _canon({"join": "neurahash-pool", "address": address, "n": int(round_hint)})
