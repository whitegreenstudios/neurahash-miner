"""
neura_l1.signing — real secp256k1 commitments for SIGNED COMMIT-REVEAL block
production and deterministic EQUIVOCATION attribution.

This module turns the PoUW L1 from "the work product (a full revealed proof) IS the
block" into a two-phase, signed protocol:

  COMMIT  — a proposer broadcasts a `Commitment`: a real secp256k1 signature over the
            consensus-critical *header* of the block it intends to publish (height,
            prev_hash, proposer, beacon, parent/new checkpoint, claimed work_score, the
            Merkle ROOT of its training trajectory, and tx_root) — but NOT the trajectory
            itself (no delta, no Merkle-opened challenge states). The best-scoring valid
            signed commitment wins the height.
  REVEAL  — only then does the winner publish the full block (the delta + the Merkle-
            opened states the Fiat-Shamir challenges query). The reveal is bound to the
            commitment: its root/checkpoints/score must match what was signed.

Why signatures change the security model:

  * Anti-theft (defense-in-depth beyond the per-proposer beacon).  A thief cannot forge a
    `Commitment` for a victim address (no private key), and cannot win the height under
    its OWN address without the trajectory (it never saw the reveal, and the beacon binds
    challenge indices to the proposer).  Selection happens on the *signed commitment*
    before any trajectory is revealed, so a lazy worker cannot grind its address to dodge
    skipped steps either — it would have to commit (sign, stake-backed) before knowing the
    challenges and before learning whether it won.

  * Equivocation becomes deterministic + attributable.  Two valid commitments by the SAME
    recovered signer at the SAME height with DIFFERENT commit ids is a nothing-at-stake
    fault.  `EquivocationProof` packages the two signed commitments; `verify_equivocation`
    re-runs ecrecover on both and returns the provably-faulty signer — no trust, any node
    reaches the identical verdict, and the slash is a deterministic state transition.

Keys: real secp256k1 via `eth_account.Account`.  In this single-machine sim each Node
holds its OWN private key (that secrecy is exactly what makes the signature
non-repudiable); the node_id->address binding is registered into genesis State, standing
in for the on-chain fact that, in a real PoUW L1, the proposer field simply *is* the
address.
"""

import json
import hashlib

from eth_account import Account
from eth_account.messages import encode_defunct

from .canon import _canon
from . import sig_scheme
from . import pqc
from .sig_scheme import SCHEME_SECP256K1, SCHEME_ML_DSA_44, UnsupportedSchemeError

# The signature scheme this build PRODUCES. Every signed envelope stamps it so a
# post-quantum scheme can drop in without a hard fork (#35 Phase 0 — see neura_l1.sig_scheme).
SIG_SCHEME = sig_scheme.DEFAULT_SCHEME


# ---------------------------------------------------------------------------
# Key helpers (real secp256k1). gen_account() makes a fresh random keypair; a Node
# keeps its Account private and only ever publishes its address + signatures.
# ---------------------------------------------------------------------------
def gen_account():
    """A fresh random secp256k1 keypair (os.urandom-seeded). Returns an eth_account
    Account; `.address` is the public identity, `.key` the 32-byte private key."""
    return Account.create()


def account_from_key(privkey):
    """Reconstruct an Account from a 32-byte private key (hex str or bytes)."""
    return Account.from_key(privkey)


# ---------------------------------------------------------------------------
# Domain separation + chain binding. Every signed message is prefixed with a fixed tag, the
# CHAIN_ID, and a per-type label before it is signed/recovered, so a signature can never be
# reinterpreted as a different message type (cross-type confusion) or replayed on a different
# chain that happens to share genesis bytes (e.g. a height-1 commitment / equivocation proof).
# NOTE: this only affects the SIGNED bytes, not commit_id()/canonical_bytes, so equivocation
# detection and de-dup are unchanged.
# ---------------------------------------------------------------------------
CHAIN_ID = "neurahash-l1-testnet-0"
_DOMAIN = b"NRH/v1"


def _domain_bytes(kind, data):
    """Prefix a signed message with DOMAIN | CHAIN_ID | kind so it is bound to its type and chain."""
    if isinstance(data, str):
        data = data.encode()
    return _DOMAIN + b"|" + CHAIN_ID.encode() + b"|" + kind.encode() + b"|" + bytes(data)


# ---------------------------------------------------------------------------
# Native transaction signing (real secp256k1) — authenticates transfer/stake/unstake so
# the sender field cannot be forged by anyone who can write the mempool.
# ---------------------------------------------------------------------------
def sign_tx(tx, account):
    """Sign a native Tx with `account` and set tx.sig to the 0x-hex signature over the tx's
    signing_bytes (everything except sig). Returns the tx. The engine's State.apply_tx
    (when require_tx_signatures is on) recovers this and checks it against the address
    registered for tx.sender."""
    msg = encode_defunct(primitive=_domain_bytes("tx", tx.signing_bytes()))
    sig = Account.sign_message(msg, account.key).signature
    tx.sig = sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    return tx


def sign_bytes(account, data):
    """Sign arbitrary bytes with `account`; return the 0x-hex signature. Used to make the
    persisted chain.json tamper-evident (the miner signs it; load/view verify) and to sign
    finality votes. Domain+chain prefixed."""
    msg = encode_defunct(primitive=_domain_bytes("bytes", data))
    sig = Account.sign_message(msg, account.key).signature
    return sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)


def recover_bytes(data, sig):
    """Recover the address that signed `data` with signature `sig` (hex)."""
    if isinstance(sig, str) and not sig.startswith("0x"):
        sig = "0x" + sig
    return Account.recover_message(encode_defunct(primitive=_domain_bytes("bytes", data)),
                                   signature=sig)


def recover_bytes_scheme(data, sig, scheme=None):
    """Scheme-dispatched recovery (#35 crypto-agility). The single point where a signed
    record's declared signature SCHEME selects the verifier, so a post-quantum scheme is a
    branch here, not a protocol rewrite. `scheme` None/"" -> the legacy default (secp256k1),
    so pre-agility records stay valid; an UNKNOWN/not-yet-implemented scheme raises
    UnsupportedSchemeError (fail-closed — never silently accepted). Returns the recovered
    secp256k1 address for the secp256k1 scheme."""
    s = sig_scheme.require_supported(scheme)          # fail-closed; absence -> secp256k1
    if s == SCHEME_SECP256K1:
        return recover_bytes(data, sig)
    # A supported but NON-recoverable scheme (ml-dsa, like every lattice signature) can't yield a signer
    # from the signature alone — the caller must verify against a SUPPLIED pubkey via verify_bytes_scheme.
    raise UnsupportedSchemeError(
        f"scheme {s!r} has no public-key recovery; use verify_bytes_scheme(data, sig, scheme, pubkey)")


def verify_bytes_scheme(data, sig, scheme=None, pubkey=None, address=None):
    """Scheme-aware signature VERIFICATION handling both recoverable (secp256k1) and non-recoverable
    (ml-dsa, #35 Phase 1) schemes — the unified entry a PQC-or-classical caller uses:
      * secp256k1 : recover the signer from the signature; if `address` is given, require recovered ==
        address. (`pubkey` is ignored — secp256k1 recovers it.)
      * ml-dsa-44 : `pubkey` is REQUIRED (no recovery); verify the signature against it, and if `address`
        is given also require `pqc.address(pubkey) == address` (binds the address to the key).
    Returns True/False. Fails CLOSED: an unknown scheme raises UnsupportedSchemeError; any malformed
    input or (for ml-dsa) a missing pubkey returns False."""
    s = sig_scheme.require_supported(scheme)          # fail-closed; absence -> secp256k1
    if s == SCHEME_SECP256K1:
        try:
            recovered = recover_bytes(data, sig)
        except Exception:
            return False
        return address is None or recovered.lower() == str(address).lower()
    if s == SCHEME_ML_DSA_44:
        if pubkey is None:
            return False                              # ml-dsa can't recover a key -> one must be supplied
        if not pqc.verify(pubkey, data, sig):
            return False
        return address is None or pqc.address(pubkey).lower() == str(address).lower()
    return False                                      # defensive (require_supported already gated)


def recover_tx_signer(tx):
    """Recover the address that signed `tx` (over its signing_bytes). Raises if tx.sig is
    absent/placeholder/malformed, or if the tx declares a signature scheme this node cannot
    verify (fail-closed). The scheme is read from an OPTIONAL `tx.sig_scheme` attribute
    (absent -> legacy secp256k1), so today's secp256k1 txs are unchanged and a future
    PQC-signed tx routes through the same gate (#35 crypto-agility)."""
    sig = tx.sig
    if not sig or sig == "dev":
        raise ValueError("tx is not signed")
    scheme = sig_scheme.require_supported(getattr(tx, "sig_scheme", None))
    if isinstance(sig, str) and not sig.startswith("0x"):
        sig = "0x" + sig
    if scheme == SCHEME_SECP256K1:
        return Account.recover_message(
            encode_defunct(primitive=_domain_bytes("tx", tx.signing_bytes())), signature=sig)
    # tx verification is recovery-based; a non-recoverable scheme (ml-dsa) can't be checked this way —
    # a PQC-signed tx must carry its pubkey and route through verify_bytes_scheme (a deeper migration,
    # #35 Phase 2+). Reject here (fail-closed) until that lands.
    raise UnsupportedSchemeError(
        f"scheme {scheme!r} is not recovery-verifiable for native txs (no pubkey-carrying tx path yet)")


# ---------------------------------------------------------------------------
# Commitment — the signed, trajectory-free promise of a block.
# ---------------------------------------------------------------------------
class Commitment:
    """The consensus-critical header content a proposer SIGNS at commit time, with NONE of
    the revealed training trajectory.  Its canonical bytes are what the secp256k1 signature
    covers, and `commit_id()` is the unique tag used for equivocation detection.

    Fields (all bound by the signature):
      height, prev_hash, proposer, beacon, parent_checkpoint, new_checkpoint,
      work_score (the CLAIMED score), merkle_root (root of the trajectory commitment),
      tx_root.

    `merkle_root` is the binding to the (still-hidden) trajectory: at reveal time the full
    block's pouw root must equal this, so the reveal cannot deviate from what was signed.
    """

    FIELDS = ("height", "prev_hash", "proposer", "beacon", "parent_checkpoint",
              "new_checkpoint", "work_score", "merkle_root", "tx_root")

    def __init__(self, height, prev_hash, proposer, beacon, parent_checkpoint,
                 new_checkpoint, work_score, merkle_root, tx_root):
        self.height = int(height)
        self.prev_hash = prev_hash
        self.proposer = proposer
        self.beacon = beacon
        self.parent_checkpoint = parent_checkpoint
        self.new_checkpoint = new_checkpoint
        self.work_score = float(work_score)
        self.merkle_root = merkle_root
        self.tx_root = tx_root

    # ---- canonical, full-precision serialization (matches header recipe) ----
    def canonical(self):
        return {
            "height": self.height,
            "prev_hash": self.prev_hash,
            "proposer": self.proposer,
            "beacon": self.beacon,
            "parent_checkpoint": self.parent_checkpoint,
            "new_checkpoint": self.new_checkpoint,
            "work_score": _canon(self.work_score),
            "merkle_root": self.merkle_root,
            "tx_root": self.tx_root,
        }

    def canonical_bytes(self):
        """Deterministic bytes the signature is computed over (and recovered against).
        Identical across nodes -> any verifier recovers the same signer."""
        return json.dumps(self.canonical(), sort_keys=True,
                          allow_nan=False).encode()

    def commit_id(self):
        """sha256 of the canonical bytes: the unique identity of this commitment. Two
        DIFFERENT commitments by one signer at one height have different commit ids ->
        equivocation."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    # ---- construction from a fully-formed block --------------------------
    @classmethod
    def from_block(cls, block):
        """The commitment a full block implies (the reveal must match this). Reads the
        Merkle root out of the block's pouw payload (hex string)."""
        h = block.header
        root = ""
        try:
            root = block.pouw["submission"]["root"]
        except Exception:
            root = ""
        return cls(
            height=h.height, prev_hash=h.prev_hash, proposer=h.proposer,
            beacon=h.beacon, parent_checkpoint=h.parent_checkpoint,
            new_checkpoint=h.new_checkpoint, work_score=h.work_score,
            merkle_root=root, tx_root=h.tx_root,
        )

    # ---- (de)serialization ----------------------------------------------
    def to_dict(self):
        return dict(self.canonical())

    @classmethod
    def from_dict(cls, d):
        return cls(
            height=d["height"], prev_hash=d["prev_hash"], proposer=d["proposer"],
            beacon=d["beacon"], parent_checkpoint=d["parent_checkpoint"],
            new_checkpoint=d["new_checkpoint"], work_score=float(d["work_score"]),
            merkle_root=d["merkle_root"], tx_root=d["tx_root"],
        )

    def matches_header(self, header):
        """True iff this commitment's signed fields equal a block header's (so a reveal
        cannot publish content other than what was committed)."""
        return (self.height == header.height
                and self.prev_hash == header.prev_hash
                and self.proposer == header.proposer
                and self.beacon == header.beacon
                and self.parent_checkpoint == header.parent_checkpoint
                and self.new_checkpoint == header.new_checkpoint
                and _canon(self.work_score) == _canon(header.work_score)
                and self.tx_root == header.tx_root)

    def __repr__(self):
        return (f"Commitment(h{self.height} {self.proposer} "
                f"score={self.work_score:.4f} id={self.commit_id()[:10]})")


# ---------------------------------------------------------------------------
# Sign / verify (real secp256k1 ecrecover).
# ---------------------------------------------------------------------------
def sign_commitment(commitment, account):
    """Sign a Commitment with a private `account`. Returns a JSON-safe signed envelope:
        {'commitment': {...}, 'signer': <address>, 'signature': <0x hex>, 'scheme': <name>}
    The signer address is included for convenience but is NOT trusted: verify_commitment
    RE-RECOVERS it from the signature so a forged 'signer' field cannot help an attacker.
    The 'scheme' tag (secp256k1 today) makes the envelope crypto-agile (#35): recovery
    dispatches on it, an unknown scheme is rejected, and a missing tag is read as the legacy
    default — so a PQC scheme drops in without invalidating existing commitments."""
    msg = encode_defunct(primitive=_domain_bytes("commit", commitment.canonical_bytes()))
    sig = Account.sign_message(msg, account.key).signature
    return {
        "commitment": commitment.to_dict(),
        "signer": account.address,
        "signature": sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig),
        "scheme": SIG_SCHEME,
    }


def _sig_to_hex(sig):
    return sig if isinstance(sig, str) else sig.hex()


def recover_signer(signed):
    """Recover the address that produced `signed['signature']` over the canonical bytes of
    `signed['commitment']`. Returns the recovered address (checksummed) or raises. The
    self-declared 'signer' field is ignored. Recovery dispatches on the envelope's 'scheme'
    tag (absent -> legacy secp256k1); an unknown scheme raises UnsupportedSchemeError
    (fail-closed) so it can never be silently accepted (#35 crypto-agility)."""
    commitment = Commitment.from_dict(signed["commitment"])
    msg = encode_defunct(primitive=_domain_bytes("commit", commitment.canonical_bytes()))
    scheme = sig_scheme.require_supported(signed.get("scheme"))
    sig = signed["signature"]
    if isinstance(sig, str) and not sig.startswith("0x"):
        sig = "0x" + sig
    if scheme == SCHEME_SECP256K1:
        return Account.recover_message(msg, signature=sig)
    # commitment verification is recovery-based; a non-recoverable scheme (ml-dsa) needs the pubkey and
    # verify_bytes_scheme. Reject here (fail-closed) until the commit-reveal path carries pubkeys.
    raise UnsupportedSchemeError(
        f"scheme {scheme!r} is not recovery-verifiable for commitments (use verify_bytes_scheme + pubkey)")


def verify_commitment(signed, expected_address=None):
    """Verify a signed commitment envelope. Returns (ok, recovered_address_or_reason).

      * recovers the signer from the signature over the commitment's canonical bytes;
      * if `expected_address` is given, requires the recovered signer to equal it
        (this is the node_id->address binding check the consensus layer enforces against
        the registered validator key).

    A tampered commitment (any signed field changed after signing) recovers a DIFFERENT
    address, so it fails the expected_address check -> rejected."""
    try:
        recovered = recover_signer(signed)
    except Exception as e:
        return False, f"signature recovery failed: {e}"
    if expected_address is not None and recovered.lower() != expected_address.lower():
        return False, (f"signer {recovered} != registered key {expected_address} "
                       f"for proposer")
    return True, recovered


# ---------------------------------------------------------------------------
# Equivocation — two signed commitments by one signer at one height.
# ---------------------------------------------------------------------------
class EquivocationProof:
    """A deterministic, self-contained fraud proof of a nothing-at-stake fault: two signed
    commitment envelopes A and B that (i) both carry a valid signature recovering to the
    SAME address, (ii) are for the SAME height, and (iii) have DIFFERENT commit ids (i.e.
    the signer committed to two distinct blocks at one height). Any node can verify it with
    no trust and reach the identical verdict, so the resulting slash is in-protocol."""

    def __init__(self, signed_a, signed_b):
        self.a = signed_a
        self.b = signed_b

    def to_dict(self):
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d):
        return cls(d["a"], d["b"])

    def __repr__(self):
        ok, who = verify_equivocation(self)
        return f"EquivocationProof(valid={ok} signer={who if ok else '-'})"


def verify_equivocation(proof, expected_address=None):
    """Deterministically check an EquivocationProof. Returns (is_equivocation, signer/reason).

    Requirements for a TRUE equivocation:
      1. both envelopes verify (real signatures);
      2. both recover to the SAME signer address (one identity is at fault);
      3. same height;
      4. SAME prev_hash (same parent/branch) — see below;
      5. different commit_id (genuinely two distinct commitments, not a duplicate of one).

    Requirement (4) is essential: the nothing-at-stake fault is committing to two DIFFERENT
    blocks on the SAME parent. Two same-height commitments on DIFFERENT parents are what an
    HONEST proposer legitimately produces when it follows fork choice across a reorg (commit
    once per parent), so slashing those would punish honest fork participation. Binding to
    prev_hash makes only same-branch double-commits slashable.

    `expected_address`, if given, additionally pins the faulting signer to a registered
    validator key. Returns the slashable signer address on success."""
    okA, recA = verify_commitment(proof.a, expected_address)
    if not okA:
        return False, f"envelope A invalid: {recA}"
    okB, recB = verify_commitment(proof.b, expected_address)
    if not okB:
        return False, f"envelope B invalid: {recB}"
    if recA.lower() != recB.lower():
        return False, f"different signers ({recA} vs {recB}) - not equivocation"
    ca = Commitment.from_dict(proof.a["commitment"])
    cb = Commitment.from_dict(proof.b["commitment"])
    if ca.height != cb.height:
        return False, f"different heights ({ca.height} vs {cb.height}) - not equivocation"
    if ca.prev_hash != cb.prev_hash:
        return False, (f"different parents ({ca.prev_hash} vs {cb.prev_hash}) - legitimate "
                       f"same-height commit across a reorg, not equivocation")
    if ca.commit_id() == cb.commit_id():
        return False, "identical commitment (duplicate, not equivocation)"
    return True, recA
