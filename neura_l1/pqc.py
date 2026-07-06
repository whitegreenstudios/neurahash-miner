"""
neura_l1.pqc — ML-DSA (FIPS-204) post-quantum signature primitives (#35, Phase 1).

The crypto-agility seam (`sig_scheme.py`, Phase 0) made the signature SCHEME a versioned field; this
module lands the first actual PQC VERIFIER behind it: **ML-DSA-44** (FIPS-204, NIST's lattice signature
standard), so `ml-dsa-44` graduates from a RESERVED name to a SUPPORTED, **opt-in** identity alongside
secp256k1.

KEY DIFFERENCE FROM ECDSA. secp256k1 supports public-key RECOVERY — the signer address is recovered
from the signature alone (`signing.recover_bytes_scheme`). ML-DSA does NOT: verification needs the
public key supplied. So a PQC identity is **pubkey-carrying** — its address is DERIVED from the public
key (`keccak256(pk)[-20:]`, the same 20-byte format as an Ethereum/secp256k1 address, so addresses are
uniform across schemes) and a record using ML-DSA must carry its pubkey. The unified
`signing.verify_bytes_scheme` dispatches: recover-and-compare for secp256k1, verify-against-supplied-
pubkey for ml-dsa.

SIZES (the ~37x cost to design around, per #35): ML-DSA-44 pk = 1312 B, sig = 2420 B, sk = 2560 B vs
secp256k1's 33 B pk / 65 B sig. Fine for an opt-in identity; the EVM side needs a PQC verify precompile
(an external/L2 gate, not closed here).

DEPENDENCY. Pure-Python `dilithium-py` (FIPS-204), pinned. It runs in the VERIFY path (public keys, not
the wallet's decrypted secret), so it is lower-risk than the wallet-critical eth-account; still pinned
per the supply-chain stance (#36). Imported LAZILY so importing this module never hard-fails when the
dep is absent — the verifier then fails CLOSED (`verify` -> False), never silently passes.
"""
import hashlib

from eth_utils import keccak

SCHEME = "ml-dsa-44"
PUBKEY_BYTES = 1312
SIG_BYTES = 2420
SECKEY_BYTES = 2560


def _backend():
    """Lazy ML-DSA-44 backend. Raises a clear error if dilithium-py isn't installed, so a caller that
    REQUESTS ml-dsa (keygen/sign) gets a hard failure rather than a silent skip."""
    try:
        from dilithium_py.ml_dsa import ML_DSA_44
        return ML_DSA_44
    except Exception as e:                      # pragma: no cover - exercised only when the dep is absent
        raise RuntimeError("ml-dsa-44 requested but the FIPS-204 backend (dilithium-py) is not "
                           "installed; run `pip install dilithium-py`") from e


def available():
    """True iff the ML-DSA backend can be imported (callers can gate opt-in PQC on availability)."""
    try:
        _backend()
        return True
    except Exception:
        return False


def keygen(seed=None):
    """Generate an ML-DSA-44 keypair (pk, sk) as bytes. With `seed` the keypair is DETERMINISTIC — for
    reproducible identities and for deriving a keypair from a wallet seed (a seed shorter than 48 bytes
    is stretched with SHA-512). NOTE: seeding sets a global DRBG in dilithium-py, so seeded keygen is
    not thread-safe — derive keys at setup, not concurrently."""
    backend = _backend()
    if seed is not None:
        s = bytes(seed) if len(seed) >= 48 else hashlib.sha512(bytes(seed)).digest()[:48]
        backend.set_drbg_seed(s[:48])
    return backend.keygen()


def sign(sk, data, deterministic=True):
    """ML-DSA-44 sign `data` with secret key `sk`. Deterministic by default (reproducible signatures)."""
    return _backend().sign(bytes(sk), bytes(data), deterministic=deterministic)


def verify(pk, data, sig):
    """ML-DSA-44 verify: True iff `sig` is a valid signature of `data` under public key `pk`. Any
    malformed input or a missing backend -> False (fail-closed), never an exception a caller might
    misread as success."""
    try:
        return bool(_backend().verify(bytes(pk), bytes(data), bytes(sig)))
    except Exception:
        return False


def address(pk):
    """Derive the on-chain identity address from an ML-DSA public key: `0x` + keccak256(pk)[-20:] — the
    same 20-byte format as a secp256k1/Ethereum address, so identities are uniform across schemes."""
    return "0x" + keccak(bytes(pk))[-20:].hex()
