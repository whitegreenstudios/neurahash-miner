"""
neurahash/endpoints.py — multi-endpoint (multi-"door") failover for a miner client.

WHY. Today a miner connects to ONE coordinator address (`--host/--port`). Every real deployment
already runs the pool behind more than one door: the public anchor VPS reverse-tunnels to the home
coordinator, and a WARM STANDBY box (tools/standby_coordinator.py) can take over if the primary dies.
If any single door dies the miner just stops earning until an operator edits its launch command. This
module lets one miner carry a LIST of doors and roll to the next when the one it is on stops answering
— availability decentralization at the client, with zero new trust (all doors are the same operator's;
the TLS pin still proves you reached a coordinator you vouch for).

WHAT THIS IS / IS NOT.
  * IS: an ordered endpoint list + a failover POLICY (try in order, remember the last door that gave a
    real mining session and try it FIRST on reconnect, but let a stale preference DECAY so a takeover
    that ended — primary came back — doesn't pin you to the retired standby forever).
  * IS NOT: consensus. Two live coordinators fork the settlement chain; the miner rolling between them
    is LIVENESS only. Which door is authoritative at an instant is the coordinators' problem (the
    standby refuses to run while the primary answers — see tools/standby_coordinator.py). The residual
    (network partition, both alive, can't see each other) is an accepted testnet risk that the
    #45-gated B12 elected-proposer machinery closes properly (docs/DECENTRALIZE_COORDINATOR.md).

BYTE-IDENTICAL SINGLE-ENDPOINT BEHAVIOUR. A bare `--host H --port P` with no commas parses to exactly
one Endpoint(H, P, global-pin) and the policy never rotates — so a single-door miner behaves precisely
as it does today. The list/rotation code only engages when you actually list more than one door.

ENDPOINT SYNTAX.
    host                      -> (host, --port, global pin)
    host:port                 -> (host, port, global pin)
    host:port#sha256:<64hex>  -> (host, port, THIS endpoint's own pin)      [future standby's own cert]
    host#sha256:<64hex>       -> (host, --port, this endpoint's own pin)
A comma-separated list of the above is the full door list. The `#sha256:` per-endpoint pin exists
because a real standby on a DIFFERENT box may present its OWN cert (not the primary's, which today is
shared via the relay); an entry with no `#pin` inherits the GLOBAL pin (NEURAHASH_TLS_PIN /
tls.resolve_client_pin()). Today, with one shared cert reached via the relay or directly, every entry
uses the global pin and this is invisible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from neurahash import tls

# env equivalent of the --host list; the FLAG wins when both are set (an explicit --host is a
# deliberate override of a box-wide env default).
COORDS_ENV = "NEURAHASH_COORDS"

# a preferred (last-good) endpoint that keeps FAILING is probably stale — a takeover ended and the
# primary returned, so the door that last gave us a session is gone. After this many CONSECUTIVE
# failures of the preferred door, drop the preference and fall back to plain list order.
DEFAULT_PREF_DECAY = 3


@dataclass(frozen=True)
class Endpoint:
    """One coordinator door: host, port, and the TLS pin to enforce against ITS cert (None -> use the
    global NEURAHASH_TLS_PIN). Frozen + hashable so it can key the last-good preference cleanly."""
    host: str
    port: int
    pin: Optional[str] = None          # normalized 64-hex (no 'sha256:' prefix), or None -> global pin

    def key(self) -> str:
        """Stable identity for de-dup + last-good bookkeeping (pin excluded: the same host:port is the
        same door regardless of which pin field carried it)."""
        return f"{self.host}:{self.port}"

    def __str__(self) -> str:
        return self.key() + (f"#sha256:{self.pin[:12]}…" if self.pin else "")


def _split_pin(token: str):
    """Split a 'host[:port]#sha256:<hex>' token into (addr_part, pin_or_None). The pin is validated +
    normalized via tls.normalize_fingerprint so a malformed per-endpoint pin fails LOUD at parse time
    (never silently un-enforced). '#' with no pin, or a non-'sha256:' scheme, is a ValueError."""
    if "#" not in token:
        return token, None
    addr, _, pin_part = token.partition("#")
    pin_part = pin_part.strip()
    if not pin_part:
        raise ValueError(f"endpoint {token!r}: '#' present but no pin follows")
    # normalize_fingerprint already accepts the 'sha256:' prefix and bare hex; require the scheme so a
    # typo like host#deadbeef is rejected rather than read as a bare-hex pin.
    if not pin_part.lower().startswith("sha256:"):
        raise ValueError(f"endpoint {token!r}: per-endpoint pin must be 'sha256:<64hex>', got {pin_part!r}")
    return addr, tls.normalize_fingerprint(pin_part)


def parse_one(token: str, default_port: int) -> Endpoint:
    """Parse a single endpoint token into an Endpoint. Accepts 'host', 'host:port', and either with a
    trailing '#sha256:<hex>' per-endpoint pin. A bare host uses `default_port`. Raises ValueError on a
    malformed token (empty host, non-integer/out-of-range port, malformed pin)."""
    token = token.strip()
    if not token:
        raise ValueError("empty endpoint token")
    addr, pin = _split_pin(token)
    addr = addr.strip()
    if not addr:
        raise ValueError(f"endpoint {token!r}: empty host")
    if ":" in addr:
        host, _, port_s = addr.rpartition(":")
        host = host.strip()
        if not host:
            raise ValueError(f"endpoint {token!r}: empty host before ':'")
        try:
            port = int(port_s)
        except ValueError:
            raise ValueError(f"endpoint {token!r}: port {port_s!r} is not an integer")
    else:
        host, port = addr, int(default_port)
    if not (0 < port < 65536):
        raise ValueError(f"endpoint {token!r}: port {port} out of range 1..65535")
    return Endpoint(host=host, port=port, pin=pin)


def parse_endpoints(spec, default_port: int):
    """Parse a comma-separated door spec ('h1:p1,h2:p2#sha256:…,h3') into a de-duplicated, order-
    preserving list[Endpoint]. Empty items (a trailing comma, doubled commas) are skipped. Duplicates
    by host:port are dropped keeping the FIRST occurrence (so an explicit pin earlier wins over a bare
    repeat later). `spec` None/'' -> []. Raises ValueError (propagated to argparse) on any malformed
    token so a fat-fingered door list fails loud instead of silently mining one door."""
    if not spec:
        return []
    out, seen = [], set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        ep = parse_one(tok, default_port)
        if ep.key() in seen:
            continue
        seen.add(ep.key())
        out.append(ep)
    return out


def resolve_endpoints(host_arg, default_port: int, env=None):
    """Resolve the miner's door list from the --host argument and the NEURAHASH_COORDS env, with the
    FLAG winning. Returns list[Endpoint].

      * host_arg given (even a single bare host)  -> parsed from host_arg; env ignored.
      * host_arg None/''                          -> parsed from env[NEURAHASH_COORDS] (the box-wide
                                                     default door list); [] if that too is unset.

    A single bare 'host' with no comma yields exactly one Endpoint(host, default_port, global-pin) — the
    byte-identical single-door path. `env` defaults to os.environ (injectable for tests)."""
    if env is None:
        env = os.environ
    if host_arg:
        return parse_endpoints(host_arg, default_port)
    return parse_endpoints(env.get(COORDS_ENV, ""), default_port)


def effective_pin(ep: Endpoint, global_pin):
    """The pin to enforce for `ep`: its OWN per-endpoint pin if it carries one, else the global pin
    (NEURAHASH_TLS_PIN). Both are already-normalized 64-hex or None. None -> plaintext (dev/loopback),
    exactly as a single-door miner with no pin behaves today."""
    return ep.pin if ep.pin else global_pin


class FailoverRotator:
    """The failover POLICY over a fixed door list. Not a network object — it just answers "which door do
    I try next?" and records the outcome, so it is trivially unit-testable and the reconnect loop stays a
    thin wrapper.

    Policy:
      * order()      -> the list in try-order for the NEXT connect attempt: the remembered last-good door
                        FIRST (a takeover means the door that last gave a session is the one still up),
                        then the rest in the operator's declared list order. Once the preferred door has
                        FAILED `pref_decay` times in a row, the preference is dropped (a takeover ended;
                        the primary is back at the top of the list) and order() is plain list order until
                        some door produces a session again.
      * note_session(ep) -> that door gave a REAL mining session: make it the preferred door and reset
                        its failure streak. (A "session" is run_worker returning/raising AFTER it was
                        admitted — see the reconnect loop; a bare connect-refused is NOT a session.)
      * note_failure(ep) -> that door failed to give a session (connect refused, or a session that never
                        really started). Advances the preferred door's decay counter.

    Single-door lists make every method a no-op-shaped identity: order() is always [that door].
    """

    def __init__(self, endpoints, pref_decay: int = DEFAULT_PREF_DECAY):
        if not endpoints:
            raise ValueError("FailoverRotator needs at least one endpoint")
        # keep declared order, de-dup defensively (resolve_endpoints already did, but be safe)
        self._eps = []
        seen = set()
        for ep in endpoints:
            if ep.key() in seen:
                continue
            seen.add(ep.key())
            self._eps.append(ep)
        self._by_key = {ep.key(): ep for ep in self._eps}
        self._pref_decay = max(1, int(pref_decay))
        self._preferred_key = None      # key() of the last door that produced a session, or None
        self._pref_fail_streak = 0      # consecutive failures of the preferred door

    @property
    def endpoints(self):
        return list(self._eps)

    @property
    def preferred(self):
        """The current last-good preferred Endpoint, or None if there is none (or it has decayed)."""
        return self._by_key.get(self._preferred_key) if self._preferred_key else None

    def order(self):
        """The endpoints in the order to try for the next reconnect (see class docstring)."""
        pref = self.preferred
        if pref is None or self._pref_fail_streak >= self._pref_decay:
            # no live preference (or it decayed): declared list order.
            return list(self._eps)
        rest = [ep for ep in self._eps if ep.key() != pref.key()]
        return [pref, *rest]

    def note_session(self, ep: Endpoint):
        """Record that `ep` produced a real mining session -> it becomes preferred; failure streak reset."""
        self._preferred_key = ep.key()
        self._pref_fail_streak = 0

    def note_failure(self, ep: Endpoint):
        """Record that `ep` failed to yield a session. If it was the preferred door, advance its decay
        counter (so a preferred door that keeps failing eventually stops jumping the queue). A failure of
        a NON-preferred door doesn't touch the preference — we only demote the door we were favoring."""
        if self._preferred_key is not None and ep.key() == self._preferred_key:
            self._pref_fail_streak += 1
