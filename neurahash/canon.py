"""neurahash/canon.py -- deterministic canonical bytes for hashing/signing a pool event body.

Extracted VERBATIM from pool_ledger.py (was the module-local _canon there) so leaf helpers
(identity_payload) and client modules can canonicalize a join/event body without importing the
economics-bearing pool_ledger. Pure: json only, no private-core deps.
"""
import json


def _canon(obj):
    """Deterministic bytes for hashing/signing an event body (sorted keys, no signature/hash)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
