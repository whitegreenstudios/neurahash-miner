"""
neurahash/pqc_admission.py — #35 Phase 2: HYBRID (secp256k1 + ML-DSA-44) admission signatures with
ANTI-DOWNGRADE pinning for the sharded pool. All the PQC-admission logic lives here so the
`sharded_pool_node.py` diff stays a thin call site (mirrors how `storage_wire.py` keeps the STORE-role
crypto out of the transport).

WHAT THIS ADDS on top of Phase 1 (`neura_l1.pqc`, the opt-in ML-DSA verifier). Phase 1 made
`ml-dsa-44` a supported, verifiable identity ALONGSIDE secp256k1 but migrated no flow. Phase 2 wires it
into the #44 per-node-signed admission handshake as a HYBRID (CNSA 2.0 dual-sign) identity — a
PQC-capable worker signs the admission challenge with BOTH its classical secp256k1 key AND its ML-DSA
key, and the coordinator requires BOTH to be valid. This is defense-in-depth against a store-now /
break-secp256k1-later attacker: forging admission for a hybrid identity would need to break BOTH the
elliptic-curve AND the lattice signature.

ANTI-DOWNGRADE (the crux, deliverable 3). Once an identity's ledger/TOFU record carries an ML-DSA
pubkey, that identity is PINNED: any later hello for it WITHOUT a valid ML-DSA signature over the same
challenge — verified against the PINNED pubkey, not a fresh attacker-supplied one — is REJECTED. So an
attacker who captured a victim's classical-only hello (or who has broken secp256k1 in the future)
cannot replay/forge a classical-only admission for a hybrid-pinned identity and silently strip the
quantum protection. The pin rides inside the SAME signed+hash-chained register entry as the TOFU key
(`pool_ledger`), so it is tamper-evident AND survives a coordinator restart for free.

DEFAULT OFF (deliverables 4 + 5). `NEURAHASH_PQC` unset -> byte-identical behavior to today: a worker
advertises no `pqc` capability, the coordinator pins nothing, and the admission gate is exactly the #44
secp256k1 handshake. Even with the coordinator's PQC on, a peer that does NOT advertise the capability
keeps joining exactly as before — hybrid is enforced PER-IDENTITY once advertised/pinned, never
fleet-wide. When `NEURAHASH_PQC=hybrid` is set but the FIPS-204 backend (`dilithium-py`) is missing,
we fail LOUDLY at startup (`require_backend_or_die`) rather than silently degrading to classical-only.

WIRE FORMAT (additive; old peers unaffected — an unknown hello field / `auth` field is ignored):
  * worker hello gains (only when NEURAHASH_PQC=hybrid): {pqc: 1, pqc_pk: <ml-dsa pubkey hex>}. This is
    the capability ADVERTISEMENT (mirrors storage_wire's `storage: 1`); it lands in coord.worker_meta.
  * the #44 `auth` reply gains (only for a hybrid worker): {pqc_pk: <hex>, pqc_sig: <ml-dsa sig hex>}
    alongside the existing classical `sig`. The pubkey travels in the auth reply too (not only the
    hello) so the admission gate — which reads the reply, not the hello — has the verification material
    in hand without widening the transport's auth_fn signature.

The pubkey rides on BOTH the hello (advertisement) and the auth reply (verification material); the
coordinator verifies against the reply's pubkey for a first-seen identity, and against the PINNED
pubkey for an already-pinned one (the pin always wins — that is the anti-downgrade guarantee).
"""
from __future__ import annotations

import os

from neura_l1 import pqc
from neurahash.identity import identity_payload   # leaf (was neurahash.pool_ledger) -- no ledger dep

__all__ = ["PQC_CAP", "PQC_CAP_VERSION", "PQC_MODE_HYBRID", "pqc_mode", "pqc_hybrid_enabled",
           "require_backend_or_die", "load_or_create_pqc_key", "worker_hello_fields",
           "worker_auth_fields", "MissingPQCBackendError"]

# The hello capability flag a hybrid-PQC-aware worker advertises (additive; absent == old/unaware or
# PQC-off client). Mirrors storage_wire.STORAGE_CAP so an old peer is entirely unaffected.
PQC_CAP = "pqc"
PQC_CAP_VERSION = 1

# The only NEURAHASH_PQC value that ACTIVATES hybrid admission. Anything else (unset / "off" / "0") is
# off — byte-identical to today. Named so a future "enforce"/"require" mode can be added without churn.
PQC_MODE_HYBRID = "hybrid"


class MissingPQCBackendError(RuntimeError):
    """Raised at startup when NEURAHASH_PQC=hybrid is requested but the FIPS-204 backend (dilithium-py)
    is not installed. A subclass of RuntimeError so it is a hard, loud failure — we NEVER silently
    degrade a hybrid deployment to classical-only (deliverable 5)."""


# ---------------------------------------------------------------------------
# mode / gating
# ---------------------------------------------------------------------------
def pqc_mode(env=None):
    """The configured PQC admission mode, normalized. `NEURAHASH_PQC` unset/empty/off/0/false/no ->
    "" (off, the default); "hybrid" -> "hybrid". Any other value normalizes to "" (off) — an
    unrecognized mode must never accidentally enable enforcement."""
    raw = (env if env is not None else os.environ.get("NEURAHASH_PQC", "")).strip().lower()
    if raw in ("", "off", "0", "false", "no", "none"):
        return ""
    if raw == PQC_MODE_HYBRID:
        return PQC_MODE_HYBRID
    return ""                                             # unrecognized -> off (fail-safe, never enforce)


def pqc_hybrid_enabled(env=None):
    """True iff hybrid PQC admission is turned on (NEURAHASH_PQC=hybrid)."""
    return pqc_mode(env) == PQC_MODE_HYBRID


def require_backend_or_die(env=None):
    """If hybrid PQC is requested, HARD-FAIL at startup unless the ML-DSA backend can be imported.
    Called from both the worker and the coordinator entrypoints (a thin call site). No-op when PQC is
    off. Raises MissingPQCBackendError (loud, clear) — never returns a silently-degraded classical-only
    posture (deliverable 5)."""
    if not pqc_hybrid_enabled(env):
        return False
    if not pqc.available():
        raise MissingPQCBackendError(
            "NEURAHASH_PQC=hybrid requires the FIPS-204 backend (dilithium-py), which is not "
            "installed. Install it (`pip install dilithium-py`) or unset NEURAHASH_PQC. Refusing to "
            "start with hybrid PQC requested but unavailable — a silent downgrade to classical-only "
            "would strip the post-quantum admission protection.")
    return True


# ---------------------------------------------------------------------------
# worker side — a stable ML-DSA identity, hello advertisement, and auth signing
# ---------------------------------------------------------------------------
def load_or_create_pqc_key(address, key_dir=".neurahash_keys"):
    """PERSISTENT ML-DSA identity for `address`, alongside the secp256k1 key (mirrors
    sharded_pool_node.load_or_create_key). Load the address's ML-DSA secret key from disk, or generate
    + save it on first use. A STABLE key across sessions is what makes a worker the SAME hybrid identity
    on reconnect — a fresh ML-DSA key every restart would trip the anti-downgrade pin against itself.
    Returns (pk_hex, sk_bytes). The file holds a secret key — it is gitignored (`.neurahash_keys/`);
    keep it secret. Raises if the backend is absent (only reached when hybrid PQC is on, which
    require_backend_or_die has already gated at startup)."""
    os.makedirs(key_dir, exist_ok=True)
    sk_path = os.path.join(key_dir, f"{str(address).replace(os.sep, '_')}.mldsa.sk")
    pk_path = os.path.join(key_dir, f"{str(address).replace(os.sep, '_')}.mldsa.pk")
    if os.path.exists(sk_path) and os.path.exists(pk_path):
        with open(sk_path) as f:
            sk = bytes.fromhex(f.read().strip())
        with open(pk_path) as f:
            pk_hex = f.read().strip()
        return pk_hex, sk
    pk, sk = pqc.keygen()
    pk_hex = bytes(pk).hex()
    with open(sk_path, "w") as f:
        f.write(bytes(sk).hex())
    with open(pk_path, "w") as f:
        f.write(pk_hex)
    return pk_hex, sk


def worker_hello_fields(pk_hex):
    """The additive hello fields a hybrid worker advertises: the pqc capability flag + its ML-DSA pubkey
    (mirrors storage_wire's {storage: 1}). An old/PQC-off worker calls this only when hybrid is on, so
    the fields are absent otherwise and an old coordinator simply ignores unknown keys."""
    return {PQC_CAP: PQC_CAP_VERSION, "pqc_pk": str(pk_hex)}


def worker_auth_fields(sk, address, nonce, pk_hex):
    """The additive fields a hybrid worker adds to its #44 `auth` reply: its ML-DSA pubkey + an ML-DSA
    signature over the SAME challenge payload the classical secp256k1 sig covers (identity_payload(
    address, nonce)). Bound to the address + fresh nonce exactly like the classical proof, so the two
    signatures attest to the same admission fact and neither can be replayed for a different identity or
    round."""
    payload = identity_payload(address, int(nonce))
    sig = pqc.sign(bytes(sk), payload)
    return {"pqc_pk": str(pk_hex), "pqc_sig": bytes(sig).hex()}
