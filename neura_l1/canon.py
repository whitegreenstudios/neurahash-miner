"""
neura_l1.canon — leaf module holding `_canon`, extracted verbatim from neura_l1.block_state so
client-facing modules (e.g. neura_l1.signing) can depend on it without pulling in block_state's
consensus/ledger transitive imports. No behavior change: identical recipe, identical bytes out.
"""

import math


def _canon(x):
    """Full-precision, JSON-safe canonical string for a float (no rounding collisions,
    no non-standard 'Infinity'/'NaN' tokens). Identical recipe to neurahash.chain._canon
    so header hashes are reproducible across nodes and across the two packages."""
    xf = float(x)
    if not math.isfinite(xf):                     # fail closed: never hash a poisoned value
        raise ValueError(f"non-finite value in canonical serialization: {x!r}")
    return format(xf, ".17g")
