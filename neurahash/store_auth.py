"""
store_auth — the `nh-store-put/1` signed-PUT scheme for the NeuraHash content store.

WHY THIS EXISTS
---------------
Opening mining to the public requires that a stranger can publish a delta with NO secret handed to
them by a human. Today the only write gate on the store is a single shared `CONTENT_TOKEN`, and that
exact token already ships publicly as the `--token` default in `fleet/esh_worker.py`, so the shared
secret is not a secret. A miner, however, already holds a secp256k1 wallet key (it already signs its
contributions), so the write gate can simply BE that key: no new key, no new UX, nothing to
distribute. This module is the sign/verify core of that scheme; `tools/content_store.py` is the only
server that consumes it, and `neura_l1/signing.py` (present in both the private repo and the PUBLIC
miner repo) does all the actual cryptography — nothing is hand-rolled here.

WIRE FORMAT (design: docs/STORE_AUTH_DESIGN.md §1)
-------------------------------------------------
Request headers on `PUT /o/<sha256>`:

    X-Sig-Alg : nh-store-put/1        version tag; ANYTHING else is rejected (fail-closed)
    X-Sig-Addr: 0x<40 hex>            claimed signer address
    X-Sig-Ts  : <unix seconds, integer>
    X-Sig     : <hex signature>
    X-Name    : <optional friendly name, unchanged from the legacy header>

The signed message is canonical, newline-delimited and ASCII -- deliberately NOT JSON, so there is
no canonicalisation ambiguity between a signer and a verifier:

    nh-store-put/1\\n<sha256-hex-of-body>\\n<ts>\\n<name-or-empty>\\n

Because a newline is the field separator, a `name` containing a newline (or any other control
character) would make the message ambiguous -- such names are refused at BOTH sign and verify time.

SECURITY NOTES
--------------
* The claimed `X-Sig-Addr` is NEVER trusted on its own: verification recovers the signer from the
  signature and authorises the request only if the recovered address equals the claimed one. The
  address this module RETURNS is the recovered one.
* The body hash is part of the signed message, so a signature is bound to exactly one object: it
  cannot be lifted onto a different body (the server verifies against the hash in the request PATH,
  and independently refuses to store a body whose sha256 is not that path).
* A replayed PUT rewrites identical bytes to the same content-addressed path -- a no-op. The only
  replayable side effect is a name update. Within the timestamp window a captured OLDER PUT can be
  replayed after a newer one and roll the name back (SEC-56); the server closes that with its opt-in
  CONTENT_SIG_MONOTONIC guard (a persisted per-(owner, name) high-water mark of accepted `ts`, see
  `tools/content_store.py` `_update_name`), on top of its name-ownership rules.
* Timestamp skew is checked in BOTH directions (a far-future `ts` is as invalid as a stale one).

STORE BINDING: nh-store-put/2 (opt-in; SEC-56)
----------------------------------------------
A /1 signature names no store, so a PUT captured at one store verifies at any other store that knows
the same signer. /2 adds the store's identity to the signed message:

    nh-store-put/2\\n<store-id>\\n<sha256-hex-of-body>\\n<ts>\\n<name-or-empty>\\n

and travels with `X-Sig-Alg: nh-store-put/2` plus an informational `X-Sig-Store: <store-id>`. The
verifier builds the message from ITS OWN configured id, never from the header, so relabelling a
captured request changes nothing. `sign_put(..., store_id=None)` and `verify_put(...)` without a
`store_id` are byte-identical to /1-only behaviour; /2 exists only where both sides opt in.
"""

import re
import time

from neura_l1 import signing

# Version tag of the scheme. A verifier rejects any other value rather than guessing (fail-closed),
# so a future `nh-store-put/2` can change the signed-message recipe without an ambiguous overlap.
ALG = "nh-store-put/1"
# The store-bound recipe (see "STORE BINDING" above). Accepted only by a verifier given a store_id.
ALG_V2 = "nh-store-put/2"

# A store id is one short token: it is a field of the newline-delimited message, so it may not carry
# a newline, a space or anything else that could blur the framing.
MAX_STORE_ID = 128
_STORE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,%d}" % (MAX_STORE_ID - 1))   # used with fullmatch

# Default accepted |now - ts| window, seconds. The server overrides this from CONTENT_SIG_SKEW.
DEFAULT_SKEW = 300

# Bound on the friendly name a signed writer may claim (the operator path is unchanged/unbounded).
MAX_NAME = 256

# Sanity bound on |X-Sig-Ts| (~34,800 years of seconds). Honest values are ~1.7e9; this only keeps a
# giant integer from reaching float arithmetic, where it raised an uncaught OverflowError.
_TS_ABS_MAX = 1 << 40

_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class SigError(ValueError):
    """Raised by the signing helpers for input that could never produce a verifiable signature."""


def name_is_valid(name):
    """True iff `name` is safe to carry in the newline-delimited signed message.

    Rejects control characters (a newline would make the message ambiguous -- see module docstring),
    non-ASCII (HTTP header values are latin-1 on the wire; keeping names ASCII removes an
    encode/decode mismatch between signer and verifier), and anything over MAX_NAME.
    """
    if name is None:
        return True                                     # absent name == empty name
    if not isinstance(name, str) or len(name) > MAX_NAME:
        return False
    return all(32 <= ord(c) < 127 for c in name)


def signing_bytes(sha256_hex, ts, name=""):
    """The exact bytes both sides sign/recover over:

        nh-store-put/1\\n<sha256-hex>\\n<ts>\\n<name-or-empty>\\n

    `sha256_hex` is the hash of the object body (== the hash in the request path). `ts` is integer
    unix seconds. `name` is the X-Name value, or "" when the PUT carries no name.
    """
    sha = str(sha256_hex).lower()
    if not _SHA_RE.match(sha):
        raise SigError(f"sha256 must be 64 lowercase hex chars, got {sha256_hex!r}")
    name = "" if name is None else str(name)
    if not name_is_valid(name):
        raise SigError("name must be printable ASCII, <= %d chars, no newlines" % MAX_NAME)
    return f"{ALG}\n{sha}\n{int(ts)}\n{name}\n".encode("ascii")


def store_id_is_valid(store_id):
    """True iff `store_id` can be a field of the nh-store-put/2 message (see _STORE_ID_RE)."""
    # fullmatch, not match + "$": "$" also matches before a trailing newline, so "anchor\n" would pass
    return isinstance(store_id, str) and bool(_STORE_ID_RE.fullmatch(store_id))


def signing_bytes_v2(store_id, sha256_hex, ts, name=""):
    """The exact bytes of the STORE-BOUND recipe:

        nh-store-put/2\\n<store-id>\\n<sha256-hex>\\n<ts>\\n<name-or-empty>\\n
    """
    if not store_id_is_valid(store_id):
        raise SigError("store id must be 1-%d chars of [A-Za-z0-9._:-], starting alphanumeric"
                       % MAX_STORE_ID)
    body = signing_bytes(sha256_hex, ts, name)          # validates sha, ts, name exactly as /1 does
    return f"{ALG_V2}\n{store_id}\n".encode("ascii") + body[len(ALG) + 1:]


def sign_put(account, sha256_hex, name=None, ts=None, store_id=None):
    """CLIENT side: return the header dict authorising `PUT /o/<sha256_hex>` as `account`.

    `account` is a `neura_l1.signing` account (`gen_account()` / `account_from_key(privkey)`), i.e.
    the miner's existing wallet key. Returns headers ready to hand to urllib/requests; `X-Name` is
    included only when a name was given, and the name is part of the signed message either way.

    `store_id=None` (the default) produces exactly the nh-store-put/1 headers this function always
    produced. A `store_id` produces the store-bound nh-store-put/2 form instead, which only a store
    configured with that same id accepts.

    Against a store running the CONTENT_SIG_MONOTONIC replay guard, each update of one name must carry
    a `ts` above the last accepted one: a second update within the same second is refused (409), and
    so is any update after a future-dated one until the clock passes it. A client that updates a name
    in quick succession should pass `ts=max(int(time.time()), last_ts + 1)`. Re-sending the newest PUT
    unchanged is accepted as a no-op.
    """
    ts = int(time.time()) if ts is None else int(ts)
    if store_id is None:
        data = signing_bytes(sha256_hex, ts, name or "")
        alg = ALG
    else:
        data = signing_bytes_v2(store_id, sha256_hex, ts, name or "")
        alg = ALG_V2
    sig = signing.sign_bytes(account, data)
    headers = {
        "X-Sig-Alg": alg,
        "X-Sig-Addr": account.address,
        "X-Sig-Ts": str(ts),
        "X-Sig": sig,
    }
    if store_id is not None:
        headers["X-Sig-Store"] = store_id
    if name:
        headers["X-Name"] = name
    return headers


def verify_put(sha256_hex, headers, now=None, skew=DEFAULT_SKEW, store_id=None, accept_v1=True):
    """SERVER side: authorise a signed PUT. Returns `(True, recovered_address)` or `(False, reason)`.

    `headers` is any mapping with `.get` (an `http.client.HTTPMessage` works -- its lookups are
    case-insensitive). `sha256_hex` is the hash from the request PATH: binding to the path (not to
    the received body) means the signature is checked BEFORE the body is read, and the server's
    independent `sha256(body) == path` rule closes the gap.

    Fails closed on every anomaly -- unknown algorithm tag, malformed address/timestamp/signature,
    out-of-window timestamp, unrecoverable signature, or a recovered signer that is not the claimed
    `X-Sig-Addr`. The returned address is the RECOVERED one, never the claimed header value.

    `store_id` (default None = today's verifier, /1 only, byte-identical) additionally accepts
    nh-store-put/2 signatures bound to THAT id: the message is rebuilt from the argument, never from
    the request's `X-Sig-Store` header. `accept_v1=False` then refuses the unbound /1 form.
    """
    alg = headers.get("X-Sig-Alg")
    if alg == ALG and accept_v1:
        v2 = False
    elif alg == ALG_V2 and store_id is not None:
        v2 = True
    elif store_id is None and accept_v1:
        return False, f"unsupported X-Sig-Alg {alg!r} (want {ALG!r})"     # the legacy reason, verbatim
    elif alg == ALG:
        return False, f"{ALG} refused: this store requires {ALG_V2}"
    else:
        want = f"{ALG_V2!r} or {ALG!r}" if accept_v1 else repr(ALG_V2)
        return False, f"unsupported X-Sig-Alg {alg!r} (want {want})"
    if v2:
        presented = headers.get("X-Sig-Store")
        if presented is not None and presented != store_id:
            return False, f"X-Sig-Store {presented!r} is not this store"

    claimed = headers.get("X-Sig-Addr") or ""
    if not _ADDR_RE.match(claimed):
        return False, "malformed X-Sig-Addr"

    raw_ts = headers.get("X-Sig-Ts")
    try:
        ts = int(str(raw_ts).strip())
        if abs(ts) > _TS_ABS_MAX:       # a 400-digit int passes int() and then overflows `now - ts`
            raise ValueError("X-Sig-Ts out of range")
    except Exception:
        return False, f"malformed X-Sig-Ts {raw_ts!r}"
    now = time.time() if now is None else float(now)
    if abs(now - ts) > float(skew):
        return False, f"X-Sig-Ts outside +/-{int(skew)}s window (ts={ts}, now={int(now)})"

    sig = headers.get("X-Sig")
    if not sig:
        return False, "missing X-Sig"

    name = headers.get("X-Name") or ""
    if not name_is_valid(name):
        return False, "invalid X-Name for a signed PUT"

    try:
        data = signing_bytes_v2(store_id, sha256_hex, ts, name) if v2 else signing_bytes(sha256_hex, ts, name)
    except SigError as e:
        return False, str(e)

    # Recover FIRST, then require recovered == claimed. verify_bytes_scheme performs the same
    # binding through the scheme-dispatched path (secp256k1 today, PQC-ready), so the identity we
    # authorise can never be the unverified header value.
    try:
        recovered = signing.recover_bytes(data, sig)
    except Exception as e:
        return False, f"signature recovery failed: {e.__class__.__name__}"
    if not signing.verify_bytes_scheme(data, sig, None, None, claimed):
        return False, f"signer {recovered} != X-Sig-Addr {claimed}"
    return True, recovered
