"""miner_id.py -- X-Miner-Id: one stable, NON-SECRET id per miner on every request it makes to the relay.

WHY (owner, 2026-10-02: "in real world, miner may have 10 or even more miner in the same home address"). The relay
-- the content store at 47.84.93.96:8710 -- identifies a sender only by a salted hash of its IP, so every miner behind
one home NAT looks like the same client. Every relay request now also carries

    X-Miner-Id: <exactly 16 lowercase hex>

so the relay's request log can tell those miners apart. It is a CLAIM: the relay logs it and never trusts it for a
limit or any other decision, and a relay that predates it ignores an unknown header, so sending it is harmless.

THE DERIVATION. A keyless miner (the public default) sends

    sha256(<its wallet address as "0x" + 40 lowercase hex, ascii>).hexdigest()[:16]

-- the value the relay itself computes from a signer it VERIFIED on a signed (X-Sig) write, so one miner's PUT and
GET log lines carry the same id. An operator-keyed miner (--key, no wallet) has no address; it sends the same hash
over its public miner name (utf-8). Only PUBLIC values are hashed: no private key, token, PSK or seed ever enters the
hash input, a log line, or the header. Neither derivation ever raises.

ONE ID PER PROCESS, resolved at first use and cached. An identity the entry point PINS right after it resolves its
signing identity (pin_address / pin_name) is authoritative. Before that, the keyless wallet the GLM contributor uses
is read -- NEURAHASH_SD_WALLET, else ~/.neurahash/glm_miner_key -- READ-ONLY (this module never creates a wallet), and
only its public address is kept. No wallet yet (a box that has never mined) means no header, never an error.

RELAY ONLY, AND NEVER ACROSS A REDIRECT. attach() is the only way the id goes onto a request, and it uses
Request.add_unredirected_header: urllib's redirect handler copies only req.headers onto the redirected request
(urllib/request.py redirect_request, 3.13 :649-651, 3.10 :679-681), so when the relay answers 302 -- its cold tier
points at a public HuggingFace dataset -- the id stays with the relay. Never add it through Request(headers=...) or
add_header. A call site whose URL may itself be a third party (a HuggingFace seed, the GitHub release mirror) passes
relay_only=True: the id is then added only for a relay origin -- the public relay, or a lane a ContentLane was built
on (register_relay).

Stdlib only at import time; neura_l1.signing (eth-account, already a core requirement) is imported lazily and only
when a wallet file has to be read.
"""
import hashlib
import os
import re
import threading
import urllib.parse

HEADER = "X-Miner-Id"
PUBLIC_RELAY = "http://47.84.93.96:8710"     # = tools/sharddiloco_glm_contributor.PUBLIC_LANE_URL
WALLET_ENV = "NEURAHASH_SD_WALLET"           # = the GLM contributor's keyless-wallet override (_default_wallet_path)

_ADDR_RE = re.compile(r"0x[0-9a-f]{40}")
_LOCK = threading.Lock()
_STATE = {"id": None, "resolved": False}
_RELAYS = set()


def id_from_address(address):
    """16 lowercase hex = sha256(address lowercased, ascii)[:16]; None unless it is 0x + 40 hex. Never raises."""
    try:
        a = str(address or "").strip().lower()
        if not _ADDR_RE.fullmatch(a):
            return None
        return hashlib.sha256(a.encode("ascii")).hexdigest()[:16]
    except Exception:                                   # noqa: BLE001 -- a header must never cost the miner
        return None


def id_from_name(name):
    """16 lowercase hex = sha256(public miner name, utf-8)[:16]; None for an empty name. Never raises: a name with a
    lone surrogate (an undecodable argv byte) is encoded with surrogatepass, so it still gets its own id."""
    try:
        n = str(name or "").strip()
        if not n:
            return None
        return hashlib.sha256(n.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    except Exception:                                   # noqa: BLE001
        return None


def _wallet_path():
    p = (os.environ.get(WALLET_ENV, "") or "").strip()
    return p or os.path.join(os.path.expanduser("~"), ".neurahash", "glm_miner_key")


def _wallet_address():
    """The PUBLIC address of the keyless wallet, or None. Read-only: the key never leaves this frame and is never
    logged; any failure means "no header", never a crash."""
    path = _wallet_path()
    if not os.path.isfile(path):
        return None
    try:
        from neura_l1.signing import account_from_key
        with open(path, "r", encoding="utf-8") as f:
            return account_from_key(f.read().strip()).address
    except Exception:                                   # noqa: BLE001 -- unreadable wallet -> no header
        return None


def _pin(mid):
    if mid:
        with _LOCK:
            _STATE["id"], _STATE["resolved"] = mid, True
    return mid


def pin_address(address):
    """Make this process's id the one derived from `address` (its signing wallet). Returns the id, or None (and
    changes nothing) when `address` is not 0x + 40 hex."""
    return _pin(id_from_address(address))


def pin_name(name):
    """Make this process's id the one derived from a public miner name (an operator-keyed miner without a wallet).
    Never raises; an empty or unusable name changes nothing."""
    return _pin(id_from_name(name))


def miner_id():
    """This process's id (16 lowercase hex), or None when it has no identity yet. Resolved once, then cached."""
    with _LOCK:
        if not _STATE["resolved"]:
            _STATE["resolved"] = True
            _STATE["id"] = id_from_address(_wallet_address())
        return _STATE["id"]


def _origin(url):
    """(scheme, host, port), or None. A URL with user info (http://x@host/) is never a relay: urlsplit would report
    `host` while urllib connects to the whole "x@host", so the two could disagree about where the request goes."""
    try:
        p = urllib.parse.urlsplit(str(url or "").strip())
        scheme, host = (p.scheme or "").lower(), (p.hostname or "").lower()
        if scheme not in ("http", "https") or not host or "@" in (p.netloc or ""):
            return None
        return scheme, host, p.port or (443 if scheme == "https" else 80)
    except ValueError:
        return None


def register_relay(base_url):
    """Remember `base_url`'s origin as a relay, so relay_only call sites send the id to it."""
    o = _origin(base_url)
    if o is not None:
        with _LOCK:
            _RELAYS.add(o)


def is_relay_url(url):
    o = _origin(url)
    if o is None:
        return False
    with _LOCK:
        return o == _origin(PUBLIC_RELAY) or o in _RELAYS


def attach(req, relay_only=False):
    """Put this process's X-Miner-Id on urllib Request `req` (if the process has an id) and return `req`.

    Always as an UNREDIRECTED header, so a 302 never carries it to the redirect target (see the module docstring).
    relay_only=True: only when req.full_url is on a relay origin, for a URL that may itself be a third party."""
    if relay_only and not is_relay_url(getattr(req, "full_url", None)):
        return req
    mid = miner_id()
    if mid:
        req.add_unredirected_header(HEADER, mid)
    return req


def _reset():
    """Tests only: forget the cached id and the registered relays."""
    with _LOCK:
        _STATE["id"], _STATE["resolved"] = None, False
        _RELAYS.clear()
