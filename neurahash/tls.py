"""
neurahash/tls.py — opt-in TLS + certificate PINNING for the pool transport (#40 / A4).

WHY. `net_transport.py` already authenticates + integrity-checks every frame with an HMAC PSK, but
HMAC gives no CONFIDENTIALITY: once the pool spans hosts, the expert weights, trunk deltas, and
gradients cross the wire in CLEARTEXT, readable by any on-path sniffer. For a permissionless pool
spanning untrusted networks that is a leak of the (eventually valuable) model. This module wraps the
coordinator + worker sockets in TLS so the channel is encrypted, and PINS the coordinator's
certificate by SHA-256 fingerprint so a man-in-the-middle cannot substitute its own cert.

PINNING, NOT A CA. A testnet has no PKI. Instead of trusting a certificate authority, the
coordinator publishes its cert fingerprint out-of-band (printed at startup / handed to workers in the
join config) and each worker checks the presented cert against that EXACT fingerprint. This is
strictly stronger than CA validation for our case (no CA to be tricked or compromised) and needs zero
PKI. The HMAC frame auth stays underneath as defense-in-depth during the transition. This is the same
construction urllib3 uses for `assert_fingerprint` (CERT_NONE handshake + manual fingerprint check).

SCOPE. TLS here is OPT-IN and default-OFF: loopback/dev and the in-process simulator are unchanged.
Self-signed certs minted here are for TESTNET. Production cert issuance / rotation / private-key
custody is an external ops gate (issue #40), NOT closed by this code — this is the in-code TLS+pinning
seam only.

USAGE.
  # one-time on the coordinator host: mint a persistent cert and read off the pin to give workers
  python -m neurahash.tls --gen --cert coord_cert.pem --key coord_key.pem
  #   -> prints  sha256:ab12…  (the pin)
  # coordinator:  NEURAHASH_TLS_CERT=coord_cert.pem NEURAHASH_TLS_KEY=coord_key.pem python testnet_node.py ...
  # worker:       NEURAHASH_TLS_PIN=ab12…            python run_networked_node.py --worker ...
"""

import os
import ssl
import hmac
import socket
import hashlib
import tempfile

DEFAULT_SNI = "neurahash-coordinator"     # SNI / cert CommonName (cosmetic — pinning, not name, is the check)
HANDSHAKE_TIMEOUT = 15.0                   # seconds budget for the TLS handshake (a stalled peer is dropped)
_FP_PREFIX = "sha256:"


class CertPinError(ssl.SSLError):
    """The coordinator's presented certificate did not match the pinned fingerprint (possible MITM).

    Subclasses ssl.SSLError so existing transport `except (ConnectionError, OSError, ValueError)`
    paths do NOT silently swallow a pin failure as an ordinary disconnect — a pin mismatch must be
    loud and must stop the worker from training against an unverified coordinator."""


# ----------------------------- fingerprints -----------------------------
def normalize_fingerprint(fp):
    """Canonicalize a user-supplied SHA-256 cert fingerprint to 64 lowercase hex chars. Accepts the
    `sha256:` prefix, OpenSSL-style colons (`AB:CD:…`), and surrounding whitespace. Raises ValueError
    on anything that is not a 256-bit hex digest, so a typo'd/truncated pin fails closed rather than
    silently never-matching."""
    if isinstance(fp, (bytes, bytearray)):
        fp = bytes(fp).decode("ascii", "strict")
    fp = fp.strip().lower().replace(" ", "")
    if fp.startswith(_FP_PREFIX):
        fp = fp[len(_FP_PREFIX):]
    fp = fp.replace(":", "")
    if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
        raise ValueError(f"not a SHA-256 fingerprint (need 64 hex chars): {fp!r}")
    return fp


def fingerprint_der(der):
    """SHA-256 fingerprint (64 lowercase hex) of a DER-encoded certificate."""
    return hashlib.sha256(der).hexdigest()


def fingerprint_pem(cert_pem):
    """SHA-256 fingerprint of a PEM certificate. Uses stdlib only (no `cryptography`) so a worker can
    fingerprint a cert it was handed without the heavier dep."""
    if isinstance(cert_pem, (bytes, bytearray)):
        cert_pem = bytes(cert_pem).decode("ascii")
    return fingerprint_der(ssl.PEM_cert_to_DER_cert(cert_pem))


# ----------------------------- self-signed testnet cert -----------------------------
def generate_self_signed(common_name=DEFAULT_SNI, days=825):
    """Mint a fresh self-signed P-256 cert + key for a TESTNET coordinator. Returns (cert_pem, key_pem)
    as bytes. 825 days = the CA/Browser-Forum max leaf lifetime. Requires `cryptography` (already in the
    tree via eth-account). The key is unencrypted PKCS#8 — fine for an ephemeral testnet key; production
    key custody is an external ops gate."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    import datetime

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))      # small skew allowance
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(common_name)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(serialization.Encoding.PEM,
                                serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    return cert_pem, key_pem


def write_cert_pair(cert_path, key_path, common_name=DEFAULT_SNI, days=825):
    """Mint a self-signed testnet cert and write it to `cert_path`/`key_path` (key as 0600 where the OS
    honors it). Returns the pin (`sha256:…`) to hand to workers. Use this once per coordinator host so
    the fingerprint stays STABLE across restarts (an ephemeral cert changes the pin every restart, which
    breaks every worker's pin)."""
    cert_pem, key_pem = generate_self_signed(common_name=common_name, days=days)
    with open(cert_path, "wb") as f:
        f.write(cert_pem)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key_pem)
    finally:
        os.close(fd)
    return _FP_PREFIX + fingerprint_pem(cert_pem)


# ----------------------------- SSL contexts -----------------------------
def _load_chain_from_pem(ctx, cert_pem, key_pem):
    """stdlib ssl has no load-cert-chain-from-memory, so spill the PEMs to short-lived temp files
    (key 0600), load, and unlink them immediately — the key is on disk only for the duration of the
    load call."""
    paths = []
    try:
        for data, suffix in ((cert_pem, "_cert.pem"), (key_pem, "_key.pem")):
            fd, p = tempfile.mkstemp(prefix="nh_tls_", suffix=suffix)
            paths.append(p)
            try:
                if suffix == "_key.pem":
                    try:
                        os.chmod(p, 0o600)
                    except OSError:
                        pass
                os.write(fd, data if isinstance(data, (bytes, bytearray)) else data.encode())
            finally:
                os.close(fd)
        ctx.load_cert_chain(paths[0], paths[1])
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def server_context(certfile=None, keyfile=None, cert_pem=None, key_pem=None):
    """Coordinator-side TLS context. Provide either file paths (`certfile`+`keyfile`, recommended — a
    persistent cert keeps the pin stable) OR in-memory PEMs (`cert_pem`+`key_pem`, for an ephemeral
    testnet cert)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    if certfile and keyfile:
        ctx.load_cert_chain(certfile, keyfile)
    elif cert_pem and key_pem:
        _load_chain_from_pem(ctx, cert_pem, key_pem)
    else:
        raise ValueError("server_context needs certfile+keyfile or cert_pem+key_pem")
    return ctx


def pinning_client_context():
    """Worker-side TLS context that does NOT CA-validate — we pin the exact cert instead (see module
    docstring). check_hostname must be cleared BEFORE verify_mode=CERT_NONE (Python enforces the order)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


# ----------------------------- socket wrapping + pin check -----------------------------
def verify_pin(ssl_sock, pinned_fp):
    """Raise CertPinError unless the peer's leaf cert matches `pinned_fp`. Constant-time compare so the
    match can't be probed byte-by-byte by timing."""
    der = ssl_sock.getpeercert(binary_form=True)     # DER even under CERT_NONE (cert was still received)
    if not der:
        raise CertPinError("coordinator presented no certificate (cannot verify pin)")
    actual = fingerprint_der(der)
    expected = normalize_fingerprint(pinned_fp)
    if not hmac.compare_digest(actual, expected):
        raise CertPinError(f"coordinator certificate pin MISMATCH (possible MITM): "
                           f"presented sha256:{actual}, pinned sha256:{expected}")


def wrap_server_socket(raw_sock, ctx, handshake_timeout=HANDSHAKE_TIMEOUT):
    """TLS-wrap a freshly accepted connection server-side. Bounds the handshake with a timeout so a peer
    that TCP-connects but never sends a ClientHello can't pin the accept loop forever. Restores the
    socket's prior timeout for the caller (the framing layer manages its own timeouts thereafter)."""
    orig = raw_sock.gettimeout()
    raw_sock.settimeout(handshake_timeout)
    try:
        ssl_sock = ctx.wrap_socket(raw_sock, server_side=True)
    except BaseException:
        try:
            raw_sock.close()
        except OSError:
            pass
        raise
    ssl_sock.settimeout(orig)
    return ssl_sock


def wrap_client_socket(raw_sock, pinned_fp, server_hostname=DEFAULT_SNI, handshake_timeout=HANDSHAKE_TIMEOUT):
    """TLS-wrap a worker's connection client-side and ENFORCE the pin. Raises CertPinError (closing the
    socket) if the coordinator's cert doesn't match — the worker must not proceed to train against an
    unverified coordinator. The pin is validated for shape BEFORE any network I/O so a malformed pin
    fails fast."""
    expected = normalize_fingerprint(pinned_fp)
    ctx = pinning_client_context()
    orig = raw_sock.gettimeout()
    raw_sock.settimeout(handshake_timeout)
    try:
        ssl_sock = ctx.wrap_socket(raw_sock, server_hostname=server_hostname)
    except BaseException:
        try:
            raw_sock.close()
        except OSError:
            pass
        raise
    try:
        verify_pin(ssl_sock, expected)
    except BaseException:
        try:
            ssl_sock.close()
        except OSError:
            pass
        raise
    ssl_sock.settimeout(orig)
    return ssl_sock


def maybe_wrap_client(sock, pin, server_hostname=DEFAULT_SNI, handshake_timeout=None):
    """One-liner for worker connect paths: TLS-wrap + pin-verify when `pin` is set, else return the
    plaintext socket unchanged (loopback/dev). `sock = tls.maybe_wrap_client(sock, tls.resolve_client_pin())`.
    `handshake_timeout=None` keeps the module default (HANDSHAKE_TIMEOUT); a joiner that must wait out
    the coordinator's once-per-round accept window passes its join patience here (the coordinator only
    services new admissions between rounds, so a patient handshake succeeds where a short one gives up
    mid-backlog — see NEURAHASH_CLIENT_JOIN_TIMEOUT in sharded_pool_node.run_worker)."""
    if not pin:
        return sock
    return wrap_client_socket(sock, pin, server_hostname=server_hostname,
                              handshake_timeout=(HANDSHAKE_TIMEOUT if handshake_timeout is None
                                                 else float(handshake_timeout)))


# ----------------------------- env resolution (deployment seam) -----------------------------
def resolve_server_tls():
    """Coordinator-side TLS from the environment. Returns (ssl_context_or_None, pin_or_None).

      NEURAHASH_TLS_CERT + NEURAHASH_TLS_KEY  -> load that PERSISTENT cert (recommended: stable pin).
      NEURAHASH_TLS in {1,true,on,yes}        -> mint an EPHEMERAL self-signed cert (pin changes every
                                                 restart — fine for one short session, NOT for persistent
                                                 worker pins).
      (neither)                               -> (None, None): plaintext HMAC-only transport (dev)."""
    cert, key = os.environ.get("NEURAHASH_TLS_CERT"), os.environ.get("NEURAHASH_TLS_KEY")
    if cert and key:
        ctx = server_context(certfile=cert, keyfile=key)
        with open(cert, "rb") as f:
            return ctx, _FP_PREFIX + fingerprint_pem(f.read())
    if os.environ.get("NEURAHASH_TLS", "").strip().lower() in ("1", "true", "on", "yes"):
        cert_pem, key_pem = generate_self_signed()
        return server_context(cert_pem=cert_pem, key_pem=key_pem), _FP_PREFIX + fingerprint_pem(cert_pem)
    return None, None


def resolve_client_pin():
    """Worker-side pinned coordinator fingerprint from NEURAHASH_TLS_PIN, or None (plaintext dev). A
    malformed pin raises here so a worker can't accidentally run with an unenforceable pin."""
    pin = os.environ.get("NEURAHASH_TLS_PIN")
    return normalize_fingerprint(pin) if pin else None


# ----------------------------- CLI: mint a cert / read a pin -----------------------------
def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="NeuraHash TLS cert helper (#40 / A4).")
    ap.add_argument("--gen", action="store_true", help="mint a self-signed testnet cert + key")
    ap.add_argument("--pin", action="store_true", help="print the pin of an existing --cert")
    ap.add_argument("--cert", default="coord_cert.pem")
    ap.add_argument("--key", default="coord_key.pem")
    ap.add_argument("--cn", default=DEFAULT_SNI, help="certificate CommonName / SNI")
    ap.add_argument("--days", type=int, default=825)
    a = ap.parse_args(argv)
    if a.gen:
        fp = write_cert_pair(a.cert, a.key, common_name=a.cn, days=a.days)
        print(f"wrote {a.cert} + {a.key}")
        print(f"pin (give workers NEURAHASH_TLS_PIN):\n  {fp}")
    elif a.pin:
        with open(a.cert, "rb") as f:
            print(_FP_PREFIX + fingerprint_pem(f.read()))
    else:
        ap.error("pass --gen (mint a cert) or --pin (read an existing cert's pin)")


if __name__ == "__main__":
    _main()
