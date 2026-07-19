"""
neurahash/wallet.py — the MINER WALLET: the secp256k1 keypair + canonical address a miner is
PAID to, persisted as a keystore file so the reward survives a restart.

WHY THIS EXISTS. The operator's ask: "the blockchain memory should save every wallet/address
needed for pool OR solo mining, so the reward distributes to the specific wallet." The "blockchain
memory" is the SIGNED, hash-chained pool ledger + the neura_l1 settlement chain
(neurahash.pool_ledger / neurahash.chain_settlement) — rewards already accrue to an ADDRESS there
and reconstruct by replaying the signed log. What was missing was a clean, first-class WALLET a
miner points at: create/import a keypair, derive its canonical address, save it (encrypted at rest
when a password is given, else refused unless plaintext is explicitly allowed), and read back the
address's reconstructed on-chain balance from a persisted ledger.

KEY-CUSTODY HARDENING (issue #36). On top of the basic eth-keystore-v3 (scrypt + AES-128-CTR)
keystore, this module now offers three opt-in, DEFAULT-OFF custody upgrades — none change the
behaviour of an existing keystore or a no-new-flags miner run:
  * Argon2id KDF keystore (`kdf="argon2id"`, CLI `--kdf argon2id`). A custom but eth-keystore-shaped
    JSON (`format: "neurahash-argon2id-v1"`): Argon2id stretches the password (memory-hard, far
    harder to brute-force on GPU/ASIC than scrypt), then the SAME AES-128-CTR + keccak-MAC envelope
    eth-keystore uses. Needs the optional `argon2-cffi` package; absent it, a clear error tells you
    to install it. Loading is automatic by format tag.
  * Hardware-wallet / external-signer path. A WATCH-ONLY keystore (`format:
    "neurahash-watchonly-v1"`) persists ONLY the address — the private key never lands on disk.
    Signing is delegated at runtime to an `ExternalSigner` (a hardware device / HSM / signing
    subprocess) via `Wallet.from_external_signer(...)` / `wallet.attach_signer(...)`. A
    `LocalKeyExternalSigner` reference implementation (in-memory key, never persisted by the wallet)
    stands in for a real device in dev/tests; a Ledger/Trezor/HSM drops in by implementing the same
    two methods. (Wiring the miner's admission/identity path to sign via the device is a follow-up.)
  * Plaintext is now OPT-IN. `save()` REFUSES to write an unencrypted private key unless
    `allow_plaintext=True` (CLI `--allow-plaintext`), so you can no longer leave a real-value key in
    cleartext by simply omitting a password.

WHAT IT IS / IS NOT (honest scope):
  * IS: a thin, well-tested wrapper over neura_l1.signing (real secp256k1 via eth-account). The
    address is the standard EIP-55 checksummed `0x…40hex` derived from the public key, so it is the
    SAME identity the pool ledger binds (TOFU) and the settlement chain credits. The default
    keystore is the standard eth-keystore-v3 JSON (scrypt + AES-128-CTR) — interoperable with
    neura_l1.local_node / neura_wallet, MetaMask, etc.
  * IS NOT: a finished hardware integration. The external-signer interface + watch-only keystore are
    here, but signing the miner's pool-admission challenge through a real device is not yet wired
    end-to-end (the Wallet-side primitives + reference signer ARE). Do not store a real-value key in
    a PLAINTEXT keystore.

CLI:
    python -m neurahash.wallet new [--out PATH] [--password ...] [--kdf scrypt|argon2id]
                                   [--allow-plaintext]                create + print the address
    python -m neurahash.wallet import --phrase "twelve words …" [...] restore from a BIP-39 phrase
    python -m neurahash.wallet watch ADDRESS [--out PATH]            address-only (no key on disk)
    python -m neurahash.wallet signer --keystore K [--password ...]  external-signer backend (stdin/stdout)
    python -m neurahash.wallet show PATH [--password ...]             print the address for a keystore
    python -m neurahash.wallet balance PATH [--ledger FILE]           reconstructed NRH balance
"""

from __future__ import annotations

import hmac
import json
import os
import sys

from eth_account import Account
from eth_utils import is_address, keccak, to_checksum_address

from neura_l1 import signing as _signing
from neura_l1.signing import account_from_key, gen_account


# --------------------------------------------------------------------------- keystore format tags
FMT_ETH_V3 = "eth-keystore-v3"            # standard scrypt/pbkdf2 + AES-128-CTR (eth_account)
FMT_PLAINTEXT = "plaintext"               # UNENCRYPTED private key — opt-in only
FMT_ARGON2ID = "neurahash-argon2id-v1"    # Argon2id KDF + AES-128-CTR + keccak-MAC (this module)
FMT_WATCHONLY = "neurahash-watchonly-v1"  # address only; key lives on an external signer/device


PLAINTEXT_WARNING = (
    "WARNING: this keystore is UNENCRYPTED (plaintext private key on disk). Anyone who reads the "
    "file can spend/forfeit this wallet. Use --password to encrypt at rest, and never store a "
    "real-value key in a plaintext keystore. (Argon2id/hardware custody is issue #36.)"
)

PLAINTEXT_REFUSED = (
    "refusing to write a PLAINTEXT (unencrypted) keystore: pass a password to encrypt the key at "
    "rest (recommended), or explicitly opt in to plaintext with allow_plaintext=True / "
    "--allow-plaintext. Never store a real-value key in plaintext."
)


# Argon2id KDF defaults. memory_cost is in KiB (65536 = 64 MiB). These exceed OWASP-2025 minimums
# (t=2, m=19 MiB, p=1) and target ~0.1–0.3 s on a desktop CPU — strong against GPU/ASIC brute force
# while staying interactive. All are recorded in the keystore so a future tune still decrypts old files.
ARGON2ID_TIME_COST = 3
ARGON2ID_MEMORY_COST = 65536      # KiB (64 MiB)
ARGON2ID_PARALLELISM = 4
ARGON2ID_DKLEN = 32               # 16 B AES-128 key + 16 B MAC key (eth-keystore layout)


# --------------------------------------------------------------------------- address helpers
def is_valid_address(address):
    """True iff `address` is a well-formed Ethereum-style address (`0x` + 40 hex, any/mixed case).
    This is the canonical NeuraHash miner-address shape (the secp256k1 pubkey hash) — the form the
    pool ledger TOFU-binds and the settlement chain credits. A legacy/free-string id like
    `0xSHARD` or `0xmyhost-007` is NOT a valid address and returns False."""
    try:
        return bool(is_address(str(address)))
    except Exception:
        return False


def normalize_address(address):
    """Return the EIP-55 checksummed form of a valid address; raise ValueError on a malformed one.
    Use at admission to reject a malformed payout address with a clear error before it can silently
    send rewards into a black hole."""
    if not is_valid_address(address):
        raise ValueError(
            f"malformed miner/payout address {address!r}: expected an 0x-prefixed 40-hex address "
            f"(the secp256k1 pubkey hash). Create one with `python -m neurahash.wallet new`.")
    return to_checksum_address(str(address))


def address_of_key(privkey):
    """The canonical checksummed address a 32-byte secp256k1 private key derives (hex str or bytes).
    Deterministic: the SAME key always yields the SAME address."""
    return account_from_key(privkey).address


def _verify_declared_address(declared, derived, path):
    """Defense-in-depth on load: the keystore's self-reported `address` is UNAUTHENTICATED metadata
    (the MAC covers only the ciphertext, not this field). If present, require it to match the address
    the loaded key actually DERIVES — so a tampered/corrupt address field, or a swapped IV that
    silently decrypts to a different key, is rejected instead of routing rewards to the wrong address.
    Mirrors neura_l1.local_node.load_account. Absent/empty -> nothing to check."""
    if not declared:
        return
    d = declared if str(declared).startswith("0x") else "0x" + str(declared)
    if d.lower() != str(derived).lower():
        raise ValueError(
            f"{path}: keystore address does not match its key (tampered or corrupt keystore)")


# --------------------------------------------------------------------------- Argon2id keystore crypto
def _require_argon2():
    """Return argon2-cffi's low-level (hash_secret_raw, Type) or raise a clear, actionable error.
    Argon2id is an OPTIONAL custody upgrade — the package is only needed to create/open an
    Argon2id keystore, so the default scrypt/plaintext paths never import it."""
    try:
        from argon2.low_level import Type, hash_secret_raw
    except Exception as e:                                   # pragma: no cover - import-environment dependent
        raise ValueError(
            "Argon2id keystores require the optional 'argon2-cffi' package, which is not installed. "
            "Install it with:  pip install argon2-cffi   (or use the default scrypt keystore by "
            "omitting --kdf argon2id)."
        ) from e
    return hash_secret_raw, Type


def _aes128ctr(key, iv, data):
    """AES-128 in CTR mode (encrypt == decrypt). `iv` is the 16-byte initial counter block, exactly
    as eth-keystore-v3 uses it — so the Argon2id envelope is byte-for-byte the same cipher as the
    standard keystore, only the KDF differs."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
    return cipher.update(data) + cipher.finalize()


def _argon2id_kdf(password, salt, params):
    hash_secret_raw, Type = _require_argon2()
    secret = password.encode("utf-8") if isinstance(password, str) else bytes(password)
    return hash_secret_raw(
        secret=secret,
        salt=bytes(salt),
        time_cost=int(params["time_cost"]),
        memory_cost=int(params["memory_cost"]),
        parallelism=int(params["parallelism"]),
        hash_len=int(params["dklen"]),
        type=Type.ID,
    )


def _argon2id_encrypt(privkey, password, *, time_cost=ARGON2ID_TIME_COST,
                      memory_cost=ARGON2ID_MEMORY_COST, parallelism=ARGON2ID_PARALLELISM):
    """Encrypt a 32-byte secp256k1 private key under `password` with an Argon2id KDF, into an
    eth-keystore-SHAPED `crypto` dict (AES-128-CTR + keccak256 MAC over dk[16:32] || ciphertext)."""
    if not password:
        raise ValueError("an Argon2id keystore requires a password")
    key = privkey if isinstance(privkey, (bytes, bytearray)) else \
        bytes.fromhex(str(privkey).removeprefix("0x"))
    salt = os.urandom(16)
    iv = os.urandom(16)
    params = {"time_cost": time_cost, "memory_cost": memory_cost,
              "parallelism": parallelism, "dklen": ARGON2ID_DKLEN}
    dk = _argon2id_kdf(password, salt, params)
    enc_key, mac_key = dk[:16], dk[16:32]
    ciphertext = _aes128ctr(enc_key, iv, bytes(key))
    mac = keccak(mac_key + ciphertext)
    return {
        "cipher": "aes-128-ctr",
        "cipherparams": {"iv": iv.hex()},
        "ciphertext": ciphertext.hex(),
        "kdf": "argon2id",
        "kdfparams": {**params, "salt": salt.hex()},
        "mac": mac.hex(),
    }


def _argon2id_decrypt(crypto, password):
    """Re-derive the Argon2id key, verify the MAC (constant-time), and AES-128-CTR-decrypt the key.
    A wrong password (or any tamper of the ciphertext/params) fails the MAC -> ValueError, never a
    silently-wrong key."""
    if password is None:
        raise ValueError("this keystore is encrypted; a password is required to load it")
    try:
        kdfparams = crypto["kdfparams"]
        params = {"time_cost": kdfparams["time_cost"], "memory_cost": kdfparams["memory_cost"],
                  "parallelism": kdfparams["parallelism"],
                  "dklen": kdfparams.get("dklen", ARGON2ID_DKLEN)}
        salt = bytes.fromhex(kdfparams["salt"])
        ciphertext = bytes.fromhex(crypto["ciphertext"])
        iv = bytes.fromhex(crypto["cipherparams"]["iv"])
        want_mac = bytes.fromhex(crypto["mac"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed Argon2id keystore: {e}")
    dk = _argon2id_kdf(password, salt, params)
    enc_key, mac_key = dk[:16], dk[16:32]
    if not hmac.compare_digest(keccak(mac_key + ciphertext), want_mac):
        raise ValueError("could not decrypt keystore (wrong password?)")
    return _aes128ctr(enc_key, iv, ciphertext)


# --------------------------------------------------------------------------- external / hardware signer
class ExternalSigner:
    """Interface for a signer that holds the private key OFF this process (a hardware wallet, HSM, or
    a signing subprocess) and signs on request — so the key never lands on this machine's disk.

    A signer only needs two things, because every NeuraHash identity proof (pool admission,
    chain.json, finality votes) is a `sign_bytes` over canonical bytes that `neura_l1.signing`
    recovers from:
        .address              -> the EIP-55 checksummed payout address the device controls
        .sign_bytes(data)     -> a 0x-hex secp256k1 signature over `data` that
                                 neura_l1.signing.recover_bytes(data, sig) recovers to .address

    A real device (Ledger/Trezor/HSM/subprocess) implements these two methods; see
    LocalKeyExternalSigner for a reference/dev implementation."""

    @property
    def address(self):                                       # pragma: no cover - interface
        raise NotImplementedError

    def sign_bytes(self, data):                              # pragma: no cover - interface
        raise NotImplementedError


class LocalKeyExternalSigner(ExternalSigner):
    """A reference ExternalSigner backed by an IN-MEMORY key — it STANDS IN for a hardware device in
    dev/tests. The point is the boundary, not the storage: the Wallet never embeds or persists this
    key; it asks the signer to sign. Swap this for a Ledger/Trezor/HSM/subprocess signer that
    implements the same `.address` + `.sign_bytes(...)` and nothing else changes."""

    def __init__(self, account):
        self._acct = account

    @classmethod
    def create(cls):
        """A fresh random device-held key (stands in for 'generate on the device')."""
        return cls(gen_account())

    @classmethod
    def from_key(cls, privkey):
        """Adopt an existing key into the (simulated) device."""
        return cls(account_from_key(privkey))

    @property
    def address(self):
        return self._acct.address

    def sign_bytes(self, data):
        """Sign with the device-held key using the SAME domain-separated scheme the consensus layer
        recovers (neura_l1.signing.sign_bytes / recover_bytes)."""
        return _signing.sign_bytes(self._acct, data)


class SubprocessExternalSigner(ExternalSigner):
    """An ExternalSigner that delegates to a separate PROCESS — a hardware-wallet bridge, an HSM
    client, or a signing daemon on a more-trusted machine. The private key lives in THAT process /
    device and never enters this one; this side only ships bytes out and reads a signature back.

    Per request the command is invoked fresh with one JSON line on stdin and must answer with one
    JSON line on stdout:
        {"op": "address"}                    -> {"address": "0x…"}
        {"op": "sign", "data_hex": "<hex>"}  -> {"sig": "0x…"}
    The signature must be over `data` with the SAME domain-separated secp256k1 scheme
    neura_l1.signing.recover_bytes expects (the reference backend is `python -m neurahash.wallet
    signer`, which signs via neura_l1.signing). `cmd` is an argv list, or a string (shlex-split).
    Address is taken from `address` if given, else queried from the backend."""

    def __init__(self, cmd, address=None, timeout=30.0):
        import shlex
        self._cmd = list(cmd) if not isinstance(cmd, str) else shlex.split(cmd, posix=(os.name != "nt"))
        if not self._cmd:
            raise ValueError("SubprocessExternalSigner: empty signer command")
        self._timeout = float(timeout)
        if address:
            self._address = normalize_address(address)
        else:
            self._address = normalize_address(self._ask({"op": "address"}).get("address"))

    def _ask(self, request):
        import subprocess
        try:
            proc = subprocess.run(self._cmd, input=json.dumps(request) + "\n",
                                  capture_output=True, text=True, encoding="utf-8",
                                  timeout=self._timeout)
        except (OSError, subprocess.SubprocessError) as e:
            raise ValueError(f"external signer command failed to run ({self._cmd[0]!r}): {e}")
        if proc.returncode != 0:
            raise ValueError(
                f"external signer exited {proc.returncode}: {(proc.stderr or '').strip()[:200]}")
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if not lines:
            raise ValueError("external signer returned no output")
        try:
            resp = json.loads(lines[-1])
        except json.JSONDecodeError as e:
            raise ValueError(f"external signer returned non-JSON: {lines[-1][:120]!r} ({e})")
        if isinstance(resp, dict) and resp.get("error"):
            raise ValueError(f"external signer error: {resp['error']}")
        return resp

    @property
    def address(self):
        return self._address

    def sign_bytes(self, data):
        raw = data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
        resp = self._ask({"op": "sign", "data_hex": bytes(raw).hex(), "address": self._address})
        sig = resp.get("sig")
        if not sig:
            raise ValueError(f"external signer returned no signature (response: {resp})")
        return sig


# --------------------------------------------------------------------------- the Wallet
class Wallet:
    """A miner's payout identity: a secp256k1 keypair whose `.address` is what the pool/solo
    settlement credits. Three custody modes, all sharing one `.address` + one keystore format family:
      * KEYED        — a local private key (create/import); the default.
      * WATCH-ONLY   — address only, NO key in this process (key on an external device).
      * EXTERNAL     — watch-only + an attached ExternalSigner that can sign on the device's behalf.
    Wraps an eth-account Account when keyed; save/load round-trips through a keystore file."""

    def __init__(self, account=None, *, address=None, signer=None, signer_meta=None):
        if account is not None:
            self._acct = account
            self._address = account.address
        else:
            # keyless (watch-only / external signer): the address is the only on-disk material.
            if address is None and signer is not None:
                address = signer.address
            self._acct = None
            self._address = normalize_address(address)
        self._signer = signer
        self._signer_meta = dict(signer_meta or {})

    # ---- construction -----------------------------------------------------
    @classmethod
    def create(cls):
        """A fresh random secp256k1 wallet (os.urandom-seeded)."""
        return cls(gen_account())

    @classmethod
    def from_key(cls, privkey):
        """Import a wallet from a raw 32-byte secp256k1 private key (hex str or bytes)."""
        return cls(account_from_key(privkey))

    @classmethod
    def from_mnemonic(cls, mnemonic):
        """Import a wallet from a BIP-39 recovery phrase (BIP-44 m/44'/60'/0'/0/0 — the same
        derivation neura_l1.local_node / MetaMask use, so the SAME phrase restores the SAME address).
        Raises ValueError on an invalid phrase."""
        Account.enable_unaudited_hdwallet_features()
        try:
            return cls(Account.from_mnemonic((mnemonic or "").strip()))
        except Exception:
            raise ValueError("invalid recovery phrase")

    @classmethod
    def watch_only(cls, address, signer_meta=None):
        """A WATCH-ONLY wallet: the canonical address with NO private key on this machine. Use for a
        hardware-wallet payout address (sign on the device) — save() persists only the address, so a
        plaintext key can never leak from it. Optionally records `signer_meta` (a hint about which
        device/signer holds the key) for the operator."""
        return cls(address=address, signer_meta=signer_meta)

    @classmethod
    def from_external_signer(cls, signer, signer_meta=None):
        """A wallet whose address comes from an ExternalSigner (hardware device/HSM/subprocess). The
        key stays on the device; this wallet can sign (via the signer) but never holds or persists a
        private key. `signer_meta` (else a generic descriptor) is what save() records."""
        return cls(signer=signer,
                   signer_meta=signer_meta or {"type": "external", "scheme": "secp256k1"})

    # ---- identity ---------------------------------------------------------
    @property
    def address(self):
        """The canonical EIP-55 checksummed payout address (works in every custody mode)."""
        return self._address

    @property
    def account(self):
        """The underlying eth-account Account (holds the private key) for KEYED wallets, else None
        (watch-only/external wallets hold no key here). Callers needing a key must check `has_key`."""
        return self._acct

    @property
    def has_key(self):
        """True iff a local private key is held in this process (a KEYED wallet)."""
        return self._acct is not None

    @property
    def is_watch_only(self):
        """True iff there is no local key AND no attached signer — address-only."""
        return self._acct is None and self._signer is None

    @property
    def signer(self):
        """The attached ExternalSigner, if any (None for keyed/pure-watch-only wallets)."""
        return self._signer

    def attach_signer(self, signer):
        """Bind an ExternalSigner (e.g. a now-connected hardware device) to a watch-only wallet so it
        can sign. The signer's address MUST match this wallet's address (you can't sign for someone
        else). Returns self."""
        if normalize_address(signer.address) != self._address:
            raise ValueError(
                f"signer address {signer.address} does not match wallet address {self._address}")
        self._signer = signer
        return self

    def can_sign(self):
        """True iff this wallet can produce a signature (a local key OR an attached signer)."""
        return self._acct is not None or self._signer is not None

    def sign_bytes(self, data):
        """Sign `data` with the wallet's identity using the consensus-recoverable scheme
        (neura_l1.signing). Uses the local key if KEYED, else delegates to the attached
        ExternalSigner. Raises if the wallet is watch-only with no signer attached."""
        if self._acct is not None:
            return _signing.sign_bytes(self._acct, data)
        if self._signer is not None:
            return self._signer.sign_bytes(data)
        raise ValueError(
            "watch-only wallet has no signing key; attach an external signer (attach_signer) to sign")

    def is_valid(self):
        """Self-check. KEYED: the stored key really derives the address it claims (catches a corrupted
        load). KEYLESS: the address is well-formed (there is no key to cross-check) — and if a signer
        is attached, it must control this address."""
        if not is_valid_address(self._address):
            return False
        if self._acct is not None:
            return account_from_key(self._acct.key).address == self._address
        if self._signer is not None:
            try:
                return normalize_address(self._signer.address) == self._address
            except Exception:
                return False
        return True

    # ---- persistence (the "blockchain memory" the reward routes to) -------
    def save(self, path, password=None, *, kdf="scrypt", allow_plaintext=False,
             argon2_time_cost=ARGON2ID_TIME_COST, argon2_memory_cost=ARGON2ID_MEMORY_COST,
             argon2_parallelism=ARGON2ID_PARALLELISM):
        """Persist the wallet so the reward address survives a restart. Behaviour by mode:
          * watch-only / external  -> a WATCH-ONLY keystore (address + signer hint ONLY; no key on
            disk). `password`/`kdf`/`allow_plaintext` are irrelevant (nothing secret to write).
          * password given, kdf="scrypt" (default) or "pbkdf2" -> standard eth-keystore-v3 JSON,
            encrypted at rest (interoperable with neura_l1 / MetaMask).
          * password given, kdf="argon2id" -> an Argon2id keystore (memory-hard KDF; needs argon2-cffi).
          * no password           -> REFUSED unless allow_plaintext=True, in which case a PLAINTEXT
            keystore is written with a LOUD warning (opt-in only; never for a real-value key).
        Writes owner-only when the OS supports it. Returns `path`."""
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if self._acct is None:
            # WATCH-ONLY / EXTERNAL: by design there is no private key to persist.
            doc = {"format": FMT_WATCHONLY, "address": self._address}
            if self._signer_meta:
                doc["signer"] = self._signer_meta
        elif password:
            if kdf == "argon2id":
                crypto = _argon2id_encrypt(self._acct.key, password, time_cost=argon2_time_cost,
                                           memory_cost=argon2_memory_cost,
                                           parallelism=argon2_parallelism)
                doc = {"format": FMT_ARGON2ID, "address": self.address, "crypto": crypto}
            elif kdf in ("scrypt", "pbkdf2"):
                # kdf="scrypt" reproduces the basic keystore byte-for-byte (eth_account defaults to scrypt).
                enc = Account.encrypt(self._acct.key, password) if kdf == "scrypt" \
                    else Account.encrypt(self._acct.key, password, kdf="pbkdf2")
                doc = {"format": FMT_ETH_V3, "address": self.address, "crypto": enc}
            else:
                raise ValueError(f"unknown kdf {kdf!r}: use 'scrypt', 'pbkdf2', or 'argon2id'")
        else:
            if not allow_plaintext:
                raise ValueError(PLAINTEXT_REFUSED)
            print(PLAINTEXT_WARNING, file=sys.stderr, flush=True)
            doc = {"format": FMT_PLAINTEXT, "address": self.address,
                   "privkey": self._acct.key.hex()}

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
        _lock_down(path)
        return path

    @classmethod
    def load(cls, path, password=None):
        """Load a wallet saved by save() (encrypted, Argon2id, plaintext, or watch-only). For an
        encrypted keystore the password is REQUIRED; a wrong/missing one raises ValueError. Also
        accepts a bare standard eth-keystore JSON (no wrapper) so a keystore from neura_l1/MetaMask
        loads too. A watch-only keystore returns a keyless Wallet (no signer attached — re-attach the
        device at runtime with attach_signer). For key-bearing formats the keystore's `address` field
        (unauthenticated metadata) is cross-checked against the address the key derives, so a
        tampered/corrupt keystore is rejected rather than silently used (see _verify_declared_address)."""
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        fmt = doc.get("format")
        if fmt == FMT_WATCHONLY:
            return cls.watch_only(doc["address"], signer_meta=doc.get("signer"))
        if fmt == FMT_PLAINTEXT:
            w = cls.from_key(doc["privkey"])
        elif fmt == FMT_ARGON2ID:
            w = cls.from_key(_argon2id_decrypt(doc["crypto"], password))
        else:
            crypto = doc["crypto"] if fmt == FMT_ETH_V3 else doc   # bare eth-keystore JSON (no wrapper)
            if password is None:
                raise ValueError("this keystore is encrypted; a --password is required to load it")
            try:
                key = Account.decrypt(crypto, password)
            except Exception as e:
                raise ValueError(f"could not decrypt keystore (wrong password?): {e}")
            w = cls.from_key(key)
        _verify_declared_address(doc.get("address"), w.address, path)
        return w

    @staticmethod
    def address_in_file(path):
        """Read JUST the payout address from a keystore WITHOUT decrypting — so `balance` and the
        miner can resolve who-gets-paid without the password (and for a watch-only keystore there is
        nothing to decrypt). Falls back to the keystore's own `address` field (present in all our
        formats and standard eth-keystore JSON)."""
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        addr = doc.get("address") or doc.get("crypto", {}).get("address")
        if not addr:
            raise ValueError(f"{path}: no address field in keystore")
        return to_checksum_address(addr if str(addr).startswith("0x") else "0x" + str(addr))

    def __repr__(self):
        mode = "keyed" if self._acct is not None else \
            ("external" if self._signer is not None else "watch-only")
        return f"Wallet({self.address}, {mode})"


# --------------------------------------------------------------------------- balance from the ledger
def balance_from_ledger(address, ledger_file):
    """Reconstruct `address`'s NRH balance from a PERSISTED signed ledger (the "blockchain memory").

    Accepts either persisted shape the coordinator writes:
      * a neurahash.chain_settlement.ChainSettlement to_state() (the settlement chain — the REWARD
        authority; balances replay from the signed neura_l1 blocks), OR
      * a neurahash.pool_ledger.SignedPoolLedger to_state() (the signed pool log — starter bond +
        rewards + slashes), OR
      * a coordinator checkpoint `stats` dict carrying a `chain` (and/or `ledger`) sub-state.
    Returns the float balance (0.0 if the address never appeared). Raises if the file's signed chain
    fails its integrity check — a tampered ledger does not silently report a balance."""
    with open(ledger_file, encoding="utf-8") as fh:
        doc = json.load(fh)
    chain_state, pool_state = _extract_states(doc)
    addr = str(address)
    # (public-miner v1) These are LAZY, on-demand imports of the PRIVATE settlement/ledger core, reached
    # ONLY by the `--balance` CLI subcommand (never by the mining path). In a public build those modules
    # are ABSENT, so guard with a clear message instead of a raw ModuleNotFoundError. `import neurahash.wallet`
    # and mining are unaffected (they never enter this branch).
    try:
        if chain_state is not None:
            from neurahash.chain_settlement import ChainSettlement
            return ChainSettlement.from_state(chain_state).balance(addr)
        if pool_state is not None:
            from neurahash.pool_ledger import SignedPoolLedger
            return SignedPoolLedger.from_state(pool_state).balance(addr)
    except ImportError as _e:
        raise RuntimeError(
            "reading a signed ledger balance needs the FULL node package (neurahash.chain_settlement / "
            "neurahash.pool_ledger), which the public miner build does not ship. Mining does not require "
            f"this; run it against a full-node install. (underlying import error: {_e})") from _e
    raise ValueError(
        f"{ledger_file}: not a recognized ledger (need a ChainSettlement/SignedPoolLedger to_state() "
        f"or a coordinator checkpoint with a 'chain'/'ledger' sub-state)")


def _extract_states(doc):
    """Find the (chain_settlement_state, pool_ledger_state) inside a persisted doc, trying the wrapper
    shapes the coordinator writes. Either may be None. The settlement chain (reward authority) is
    preferred for balance; the pool ledger is the fallback."""
    # a bare ChainSettlement.to_state(): {"coord_key","blocks",...}
    if "blocks" in doc and "coord_key" in doc and "log" not in doc:
        return doc, None
    # a bare SignedPoolLedger.to_state(): {"coord_key","log","head",...}
    if "log" in doc and "coord_key" in doc:
        return None, doc
    # a coordinator checkpoint: stats may be nested under "stats" or at top level
    stats = doc.get("stats", doc)
    chain = stats.get("chain") if isinstance(stats, dict) else None
    pool = stats.get("ledger") if isinstance(stats, dict) else None
    return chain, pool


# --------------------------------------------------------------------------- file perms
def _lock_down(path):
    """Best-effort owner-only permissions on the keystore. chmod 600 on POSIX; on Windows strip
    inheritance + grant the current user only via icacls (so the encrypted key isn't broadly
    readable). Mirrors neura_l1.local_node._lock_down. A no-op-returning-False on platforms where it
    can't be applied — save() is best-effort here because the key material is encrypted; the
    plaintext path additionally prints a loud warning. Returns True on success, False otherwise."""
    try:
        if os.name == "nt":
            import getpass
            import subprocess
            try:
                user = getpass.getuser()
            except Exception:
                return False
            try:
                r = subprocess.run(["icacls", path, "/inheritance:r"],
                                   capture_output=True, timeout=5)
            except (FileNotFoundError, OSError):
                return False                       # icacls unavailable -> cannot secure
            if r.returncode != 0:
                return False
            if user:
                subprocess.run(["icacls", path, "/grant:r", f"{user}:(F)"],
                               capture_output=True, timeout=5)
            return True
        os.chmod(path, 0o600)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- CLI
def _save_from_cli(wallet, args, header):
    """Shared save path for `new`/`import`: maps --password/--kdf/--allow-plaintext to Wallet.save and
    turns the library's refusals into a clean CLI error+exit instead of a traceback. `header` (the
    'created/imported wallet 0x…' line) is printed ONLY once a save is going to proceed, so a refused
    run never claims it created a wallet it didn't persist. Returns 0/2."""
    kdf = getattr(args, "kdf", "scrypt") or "scrypt"
    if kdf == "argon2id" and not args.password:
        print("--kdf argon2id requires --password (Argon2id encrypts the key under a password).",
              file=sys.stderr)
        return 2
    if not args.password and not getattr(args, "allow_plaintext", False):
        print("refusing to write a PLAINTEXT keystore: pass --password to encrypt (recommended), "
              "or --allow-plaintext to store the key unencrypted.", file=sys.stderr)
        return 2
    print(header)
    try:
        wallet.save(args.out, password=args.password, kdf=kdf,
                    allow_plaintext=getattr(args, "allow_plaintext", False))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    enc = "NO (plaintext)" if not args.password else f"yes ({kdf})"
    print(f"  keystore : {os.path.abspath(args.out)}")
    print(f"  encrypted: {enc}")
    return 0


def _cmd_new(args):
    w = Wallet.create()
    rc = _save_from_cli(w, args, f"created wallet {w.address}")
    if rc == 0:
        print("point a miner at it with:  python run_miner_client.py --wallet "
              f"{args.out} --host <COORD> --port <PORT>")
    return rc


def _cmd_import(args):
    if args.phrase:
        w = Wallet.from_mnemonic(args.phrase)
    elif args.key:
        w = Wallet.from_key(args.key)
    else:
        print("import: provide --phrase \"<12 words>\" or --key <hex privkey>", file=sys.stderr)
        return 2
    return _save_from_cli(w, args, f"imported wallet {w.address}")


def _cmd_watch(args):
    """Write a WATCH-ONLY keystore: the payout address only, no private key on disk (the key lives on
    a hardware device / external signer that signs at runtime)."""
    try:
        w = Wallet.watch_only(args.address)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    w.save(args.out)
    print(f"wrote watch-only wallet {w.address}")
    print(f"  keystore : {os.path.abspath(args.out)}  (NO private key on disk)")
    print("  sign with a hardware device/external signer at runtime (attach_signer).")
    return 0


def _cmd_signer(args):
    """Reference EXTERNAL-SIGNER backend (the `SubprocessExternalSigner` other side). Reads ONE JSON
    request line from stdin and writes ONE JSON response line to stdout, then exits. The private key
    is loaded into THIS process from its own keystore and signs here — so a miner that shells out to
    `python -m neurahash.wallet signer --keystore device.json` never sees the key. Stands in for a
    real hardware-wallet bridge; a Ledger/HSM backend would answer the same two ops.

    Password (for an encrypted device keystore) comes from --password or $NEURAHASH_SIGNER_PASSWORD
    (preferred — keeps it out of the process argv)."""
    req_line = sys.stdin.readline()
    try:
        req = json.loads(req_line or "{}")
    except json.JSONDecodeError:
        print(json.dumps({"error": "request was not valid JSON"}))
        return 2
    op = req.get("op")
    try:
        if op == "address":
            print(json.dumps({"address": Wallet.address_in_file(args.keystore)}))
            return 0
        if op == "sign":
            pw = args.password or os.environ.get("NEURAHASH_SIGNER_PASSWORD")
            w = Wallet.load(args.keystore, password=pw)
            if not w.has_key:
                print(json.dumps({"error": "signer keystore holds no private key"}))
                return 2
            raw = bytes.fromhex(str(req["data_hex"]).removeprefix("0x"))
            print(json.dumps({"sig": w.sign_bytes(raw)}))
            return 0
    except Exception as e:                                   # any load/sign failure -> a JSON error line
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 2
    print(json.dumps({"error": f"unknown op {op!r}"}))
    return 2


def _cmd_show(args):
    # address-only by default (no decrypt needed); --password verifies the key really opens.
    if args.password:
        w = Wallet.load(args.path, password=args.password)
        print(w.address)
        print(f"  key check: {'OK' if w.is_valid() else 'FAILED'}")
    else:
        print(Wallet.address_in_file(args.path))
    return 0


def _cmd_balance(args):
    addr = Wallet.address_in_file(args.path)
    if args.ledger:
        bal = balance_from_ledger(addr, args.ledger)
        print(f"{addr}  balance: {bal:.6f} NRH   (ledger: {args.ledger})")
    else:
        print(f"{addr}  balance: <no --ledger given> "
              "(pass --ledger <chain/checkpoint json> to reconstruct from the signed ledger)")
    return 0


# --------------------------------------------------------------------------- rotate (Phase 0 hygiene)
def _cmd_rotate(args):
    """
    Rotate the miner's identity keypair.
    Archives the old keystore (renames with a timestamp suffix) and writes a fresh
    keypair to the original path, preserving the encryption/KDF settings.
    Prints both the OLD and NEW addresses so the operator can update any external
    references (pool payout configs, coordinator whitelists, etc.).
    """
    from datetime import datetime
    import shutil

    # Load the existing wallet (need the password if it is encrypted).
    pw = args.password
    w = Wallet.load(args.path, password=pw)
    if not w.has_key:
        raise ValueError(f"{args.path}: cannot rotate a watch-only keystore (no private key on disk)")

    old_addr = w.address

    # Preserve the encryption settings: infer the format from whether a password was supplied. An
    # encrypted rotation re-encrypts the fresh key with the SAME password + KDF; a plaintext rotation is
    # refused unless --allow-plaintext is given (re-encrypting is the recommended path). VALIDATE this
    # BEFORE touching the old keystore so a refusal leaves the original file exactly in place (never
    # archive-then-abort, which would strand the only keystore under the .rotated_ path).
    was_encrypted = pw is not None
    kdf = args.kdf
    if not was_encrypted and not args.allow_plaintext:
        raise ValueError(
            "original keystore was plaintext; re-encrypting with --password is recommended. "
            "Use --allow-plaintext to confirm, or provide --password to encrypt the new key.")

    # Archive the old keystore (timestamp suffix) so the previous key is never destroyed in place.
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archived = f"{args.path}.rotated_{ts}"
    shutil.move(args.path, archived)
    print(f"[rotate] archived old keystore -> {archived}")
    print(f"[rotate] old address: {old_addr}")

    # Generate a fresh keypair and write it to the ORIGINAL path, reusing the same encryption params.
    new_account = gen_account()
    new_addr = new_account.address
    new_w = Wallet(new_account)
    if was_encrypted:
        new_w.save(args.path, password=pw, kdf=kdf)
    else:
        new_w.save(args.path, allow_plaintext=True)

    print(f"[rotate] new address: {new_addr}")
    print(f"[rotate] new keystore written -> {os.path.abspath(args.path)}")
    if was_encrypted:
        print(f"[rotate] encrypted with --password + --kdf={kdf}")
    else:
        print("[rotate] WARNING: written as PLAINTEXT (use --password to encrypt at rest)")

    # Optional: if a ledger file is given, show the balance for both addresses. Reuses the guarded
    # public helper (balance_from_ledger), so on a public build without the settlement core this
    # degrades to a clear "skipped" message instead of raising AFTER the key was already rotated.
    if args.ledger:
        try:
            old_bal = balance_from_ledger(old_addr, args.ledger)
            new_bal = balance_from_ledger(new_addr, args.ledger)
            print(f"[rotate] balances (ledger {args.ledger}): old={old_bal:.6f} NRH, new={new_bal:.6f} NRH")
        except Exception as e:
            print(f"[rotate] ledger balance lookup skipped: {e}")

    return 0


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m neurahash.wallet",
        description="NeuraHash miner wallet — create/import the secp256k1 keypair a miner is PAID to.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="create a fresh wallet and print its address")
    p_new.add_argument("--out", default="miner_wallet.json", help="keystore path (default miner_wallet.json)")
    p_new.add_argument("--password", default=None, help="encrypt the keystore at rest (recommended)")
    p_new.add_argument("--kdf", default="scrypt", choices=["scrypt", "pbkdf2", "argon2id"],
                       help="key-derivation for the encrypted keystore (default scrypt; argon2id is "
                            "memory-hard, needs argon2-cffi)")
    p_new.add_argument("--allow-plaintext", action="store_true",
                       help="opt in to writing an UNENCRYPTED keystore when no --password is given")
    p_new.set_defaults(func=_cmd_new)

    p_imp = sub.add_parser("import", help="import a wallet from a recovery phrase or raw private key")
    p_imp.add_argument("--phrase", default=None, help="BIP-39 12/24-word recovery phrase")
    p_imp.add_argument("--key", default=None, help="raw 32-byte secp256k1 private key (hex)")
    p_imp.add_argument("--out", default="miner_wallet.json", help="keystore path to write")
    p_imp.add_argument("--password", default=None, help="encrypt the keystore at rest (recommended)")
    p_imp.add_argument("--kdf", default="scrypt", choices=["scrypt", "pbkdf2", "argon2id"],
                       help="key-derivation for the encrypted keystore (default scrypt; argon2id "
                            "needs argon2-cffi)")
    p_imp.add_argument("--allow-plaintext", action="store_true",
                       help="opt in to writing an UNENCRYPTED keystore when no --password is given")
    p_imp.set_defaults(func=_cmd_import)

    p_watch = sub.add_parser("watch", help="write a watch-only keystore (address only, NO key on disk)")
    p_watch.add_argument("address", help="the payout address the external device/signer controls")
    p_watch.add_argument("--out", default="miner_wallet.json", help="keystore path to write")
    p_watch.set_defaults(func=_cmd_watch)

    p_sign = sub.add_parser("signer", help="reference external-signer backend (one stdin/stdout JSON "
                                           "request); a miner targets it via --signer-cmd")
    p_sign.add_argument("--keystore", required=True, help="the device's OWN keystore (holds the key)")
    p_sign.add_argument("--password", default=None,
                        help="password for the device keystore (or set $NEURAHASH_SIGNER_PASSWORD)")
    p_sign.set_defaults(func=_cmd_signer)

    p_show = sub.add_parser("show", help="print the payout address for a keystore")
    p_show.add_argument("path", help="keystore path")
    p_show.add_argument("--password", default=None, help="also decrypt + verify the key opens")
    p_show.set_defaults(func=_cmd_show)

    p_bal = sub.add_parser("balance", help="reconstruct the address's NRH balance from a signed ledger")
    p_bal.add_argument("path", help="keystore path (its address is read WITHOUT the password)")
    p_bal.add_argument("--ledger", default=None,
                       help="persisted signed ledger: a ChainSettlement/SignedPoolLedger to_state() "
                            "JSON or a coordinator checkpoint with a 'chain'/'ledger' sub-state")
    p_bal.set_defaults(func=_cmd_balance)

    p_rot = sub.add_parser("rotate", help="rotate the miner's keypair: archive the old keystore, generate a fresh one, and write it to the same path (preserving encryption/KDF)")
    p_rot.add_argument("path", help="keystore path to rotate")
    p_rot.add_argument("--password", default=None, help="password for an encrypted keystore (if omitted, assumes plaintext)")
    p_rot.add_argument("--kdf", default="scrypt", choices=["scrypt", "pbkdf2", "argon2id"],
                       help="KDF to re-encrypt with (default scrypt; argon2id needs argon2-cffi)")
    p_rot.add_argument("--allow-plaintext", action="store_true",
                       help="if the original was plaintext -> plaintext rotation (without this, plaintext -> plaintext is refused; re-encrypt with --password instead)")
    p_rot.add_argument("--ledger", default=None,
                       help="optional ledger file to show old/new balances (ChainSettlement or coordinator checkpoint JSON)")
    p_rot.set_defaults(func=_cmd_rotate)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
