"""
neura_l1.sig_scheme — crypto-agility: a versioned signature-scheme registry so a
post-quantum scheme (ML-DSA / FIPS-204) can drop in WITHOUT a hard fork (#35, Phase 0).

WHY THIS EXISTS. secp256k1 / ECDSA is an EXISTENTIAL break under Shor's algorithm
(full private-key recovery / forgery, not a weakening). It underpins every signature in
NeuraHash: wallet keys, native txs, commit-reveal block production, governance votes, and
the pool ledger. No 2026 machine is close, but `harvest-now-decrypt-later` means every
public key already on the immutable ledger is retroactively forgeable the instant a
cryptographically-relevant quantum computer exists. The single highest-leverage thing we
can do NOW — cheaper than any migration — is make the signature SCHEME a versioned field
everywhere, so a PQC scheme is a registry entry + a verifier, not a protocol rewrite.

THE STRUCTURAL ADVANTAGE. Because absence of a scheme tag NORMALIZES to the legacy default
(secp256k1), every pre-agility record stays valid — we never need a confiscatory freeze
fork (the thing that makes a Bitcoin/Ethereum PQC migration so painful). New records stamp
their scheme; verification dispatches on it; an unknown scheme is REJECTED (fail-closed),
never silently accepted.

HONEST SCOPE (do not overclaim — this module does NOT, by itself, make NeuraHash quantum-resistant).
  * Phase 0 (shipped): the versioned-scheme SEAM (this registry). Absence of a tag normalizes to
    secp256k1, so every pre-agility record stays valid without a freeze fork.
  * Phase 1 (shipped, opt-in): `ml-dsa-44` (FIPS-204) is now a SUPPORTED, verifiable scheme via
    `neura_l1.pqc` (pure-Python dilithium-py). Because ML-DSA has no public-key recovery, it verifies
    against a SUPPLIED pubkey through `signing.verify_bytes_scheme` (not `recover_bytes_scheme`). It is
    OPT-IN: no existing flow is migrated yet — the capability exists so a caller CAN use a PQC identity.
  * Still RESERVED: the `secp256k1+ml-dsa-44` hybrid (CNSA 2.0 dual-sign, Phase 2). A record claiming a
    KNOWN-but-unsupported scheme is still rejected (fail-closed). The EVM-side PQC verify precompile and
    the full default-flip (Phase 3) are out of scope here.
  * This module remains the policy/registry with NO crypto dependency; the actual verifiers live in
    `neura_l1.signing` (secp256k1 ecrecover) and `neura_l1.pqc` (ML-DSA).
  * Phase 0 does NOT, for already-deployed signed-byte formats (votes, commitments), fold
    the scheme tag INTO the signed bytes (that would invalidate existing signatures). With
    a single supported scheme that is safe: flipping the tag to an unknown scheme only gets
    the record rejected, and flipping absent->secp256k1 is a no-op. Binding the scheme tag
    into the signed bytes per-scheme (anti-downgrade) is Phase 1, alongside the ML-DSA
    verifier. The pool ledger, whose body we fully control, DOES carry the scheme inside the
    signed+hashed body today (tamper-evident for free).

Sources: Google QuantumAI cryptocurrency paper, GRI Quantum Threat Timeline 2025,
NIST IR 8547, FIPS 204/205.
"""
from __future__ import annotations

# --- scheme names ----------------------------------------------------------
SCHEME_SECP256K1 = "secp256k1"                       # the legacy default (verifiable today)
SCHEME_ML_DSA_44 = "ml-dsa-44"                       # FIPS 204 (RESERVED — Phase 1, not yet verifiable)
SCHEME_HYBRID_ECDSA_ML_DSA_44 = "secp256k1+ml-dsa-44"  # CNSA 2.0 transition (RESERVED — Phase 2)

# The scheme assumed when a record carries NO scheme tag. Normalizing absence to the legacy
# scheme is what lets every pre-agility record stay valid without a freeze fork.
DEFAULT_SCHEME = SCHEME_SECP256K1

# Schemes this node can actually VERIFY right now. Anything outside this set is rejected
# (fail-closed), even if it is a KNOWN reserved name. Phase 1 (#35) added ml-dsa-44 (FIPS-204) — the
# first PQC verifier — via `neura_l1.pqc` (verify needs the supplied pubkey; see RECOVERABLE_SCHEMES).
SUPPORTED_SCHEMES = frozenset({SCHEME_SECP256K1, SCHEME_ML_DSA_44})

# Schemes that support public-key RECOVERY from the signature alone (secp256k1) vs. those that need the
# public key SUPPLIED to verify (ml-dsa, like every lattice signature). `recover_bytes_scheme` only
# serves recoverable schemes; non-recoverable ones go through `verify_bytes_scheme(..., pubkey=...)`.
RECOVERABLE_SCHEMES = frozenset({SCHEME_SECP256K1})


def is_recoverable(scheme) -> bool:
    """True iff the signer address can be RECOVERED from a signature alone (secp256k1). False for ml-dsa
    and other lattice schemes, whose verification requires the public key to be supplied."""
    return normalize_scheme(scheme) in RECOVERABLE_SCHEMES

# Every scheme name the protocol RESERVES (so tooling/UagI can display them and so a typo'd
# scheme is distinguishable from a reserved-but-unimplemented one). A reserved name that is
# not yet in SUPPORTED_SCHEMES still rejects.
KNOWN_SCHEMES = frozenset({SCHEME_SECP256K1, SCHEME_ML_DSA_44, SCHEME_HYBRID_ECDSA_ML_DSA_44})


class UnsupportedSchemeError(ValueError):
    """Raised when a signed record declares a signature scheme this node cannot verify.
    A ValueError subclass so existing `except ValueError` paths around signature handling
    treat an unknown scheme as a verification failure (reject), never as a silent pass."""


def normalize_scheme(scheme):
    """Map a record's (possibly absent) scheme tag to a concrete scheme name. None / "" /
    missing -> the legacy DEFAULT_SCHEME (secp256k1), so pre-agility records stay valid.
    Otherwise the tag is returned verbatim (still subject to require_supported)."""
    if scheme is None or scheme == "":
        return DEFAULT_SCHEME
    return str(scheme)


def is_known(scheme) -> bool:
    """True iff the (normalized) scheme is a RESERVED protocol scheme — supported or not."""
    return normalize_scheme(scheme) in KNOWN_SCHEMES


def is_supported(scheme) -> bool:
    """True iff the (normalized) scheme can be VERIFIED on this node today."""
    return normalize_scheme(scheme) in SUPPORTED_SCHEMES


def require_supported(scheme) -> str:
    """Return the normalized scheme name if this node can verify it, else raise
    UnsupportedSchemeError (fail-closed). This is the one gate every verifier calls before
    recovering a signer, so an unknown / not-yet-implemented scheme can never be silently
    accepted as valid."""
    s = normalize_scheme(scheme)
    if s not in SUPPORTED_SCHEMES:
        hint = " (reserved for a future phase, not yet verifiable)" if s in KNOWN_SCHEMES else ""
        raise UnsupportedSchemeError(
            f"signature scheme {s!r} is not verifiable on this node{hint}; "
            f"supported = {sorted(SUPPORTED_SCHEMES)}")
    return s
