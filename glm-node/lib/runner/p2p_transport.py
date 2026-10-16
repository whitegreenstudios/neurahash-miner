"""P2P transport with automatic path switching -- STEP 1 (runner-independent) + p2p2 (2026-10-13): the real pion
sidecar client (PionSidecar, sidecar.kind "pion"), pre-byte pair screening, the stall hedge, the integrity latch.

Design: docs/research/P2P_TRANSPORT_AUTOSWITCH_DESIGN_2026-10-10.md (sec. 1.2 seam, 2.3 policy,
2.4 hedge, 2.5 logging, 3 integrity, 8 owner decisions).

Phase 1 contract (what the runner hook in step 3 relies on):
  * ``Transport.try_get(sha, deadline=None)`` returns bytes whose sha256 == sha, or None.
    None ALWAYS means "use the relay" (the existing HTTP GET runs unchanged). It never raises.
    Peer bytes that fail the sha are never returned and never surface to the caller, so they
    can never count toward the runner's G3 miss counter (design sec. 3.2).
  * ``Transport.offer(sha, data, name=None)`` announces a part we already PUT to the relay.
  * ``Transport.record_relay(sha, nbytes, seconds)`` lets the hook feed relay GET timings
    (the relay estimate R_rly is passive, sec. 2.2).
  * ``Transport.stats()`` / ``Transport.close()``.

No network code lives here. The real pion sidecar is reached through ``SidecarClient``; tests
use ``FakeSidecar``. Uploads, names, manifests and commits stay on the relay (sec. 1.2).
Units: every rate in this module is BYTES PER SECOND (r_rly / r_p2p in transport.jsonl too).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

# ---------------------------------------------------------------- policy constants (sec. 2.2-2.4)
# Q6 (owner): constants still to be confirmed before step 3 -- change them HERE only.
EWMA_ALPHA = 0.3
MIN_SAMPLES = 2                     # a path with < 2 samples is UNKNOWN
HYSTERESIS = 1.25                   # relay preferred only if R_rly > 1.25 x R_p2p (both >= 2 samples)
CAP_BPS = 3_000_000 / 8.0           # relay CAPPED if the last 2 samples are both < 3 Mbit/s
RELAY_SAMPLE_MIN_BYTES = 1_000_000  # only relay GETs >= 1 MB feed R_rly
W_HAVE_S = 30.0                     # capped relay + owner connected: wait this long for its HAVE
T_FIRST_S = 10.0                    # hedge: no first chunk by then -> relay
HEDGE_PROJECTION = 1.5              # hedge: projected finish > 1.5 x size / R_rly -> relay
REPROBE_EVERY = 3                   # every 3rd round, prefetch still P2P-fetches (case 4)
PEER_LATCH_BAD_SHA = 1              # latch a peer after 1 sha-bad part
PEER_LATCH_CRASHES = 2              # ... or after 2 sidecar crashes while fetching from it
GLOBAL_WINDOW = 6                   # global latch: >= 50% fallbacks over the last 6 P2P fetches
GLOBAL_FALLBACK_FRAC = 0.5
# runner_n3_xport_2026-10-14 (#4a, docs/research/TRANSPORT_TEST_PLAN_FABLE_2026-10-14.md sec. 4): the global latch is no
# longer a one-way door. (1) A LOST HEDGE RACE -- the relay copy, raced in parallel, simply arrived first -- is NOT a
# fallback: P2P did not fail, the relay was faster; it is counted (lost_races) and leaves the window untouched.
# (2) BOUNDED RE-PROBE: while the global latch holds, the policy is run ONCE (one P2P attempt at most) after every
# GLOBAL_REPROBE_EVERY latched fetch calls OR GLOBAL_REPROBE_AFTER_S seconds on this transport's clock, whichever comes
# first. The probe's P2P success clears the global latch and the window; a fallback, a crash or a lost race re-arms the
# latch's clock and count. Per-peer latches (sha-bad bodies, integrity, crashes, non-public pairs) are UNCHANGED and
# never re-probed: a probe can never pick a latched peer. GLOBAL_REPROBE_EVERY = 0 restores the one-way latch.
GLOBAL_REPROBE_EVERY = 6            # latched fetch calls between two probes (~3 rounds at N = 3: 2 peer parts a round)
GLOBAL_REPROBE_AFTER_S = 600.0      # ... or this long since the latch tripped / the last probe
DEFAULT_FETCH_BUDGET_S = 300.0      # bound when the caller passes no deadline
NO_PROGRESS_GIVE_UP_S = 60.0        # 2026-10-13: a raced P2P fetch with no byte for this long is cancelled
POLL_S = 0.05

MODES = ("off", "auto", "force-direct", "force-turn", "force-relay")
PART_NAME_RE = re.compile(r"^eph/op/[^/]+/[^/]+/x\d{4}$")   # repl_obj_name pattern (R:7716-7717)


# ---------------------------------------------------------------- clock
class Clock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, s: float) -> None:
        time.sleep(s)


class FakeClock(Clock):
    """Deterministic clock: sleep() advances time. Thread-safe."""

    def __init__(self, t0: float = 1000.0):
        self._t = t0
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def sleep(self, s: float) -> None:
        with self._lock:
            self._t += s

    def advance(self, s: float) -> None:
        self.sleep(s)


# ---------------------------------------------------------------- rate estimates
@dataclass(frozen=True)
class EstView:
    ewma: Optional[float]           # bytes/s, None when no sample
    n: int
    capped: bool = False

    @property
    def known(self) -> bool:
        return self.n >= MIN_SAMPLES and self.ewma is not None


UNKNOWN = EstView(None, 0, False)


class RateEst:
    def __init__(self):
        self.ewma: Optional[float] = None
        self.n = 0
        self.last = deque(maxlen=2)

    def add(self, rate_bps: float) -> None:
        self.ewma = rate_bps if self.ewma is None else EWMA_ALPHA * rate_bps + (1 - EWMA_ALPHA) * self.ewma
        self.n += 1
        self.last.append(rate_bps)

    def view(self, cap_check: bool = False) -> EstView:
        capped = cap_check and len(self.last) == 2 and max(self.last) < CAP_BPS
        return EstView(self.ewma, self.n, capped)


# ---------------------------------------------------------------- pure policy (sec. 2.3)
@dataclass(frozen=True)
class PolicyInput:
    sha: str
    have_peers: FrozenSet[str]              # peers whose HAVE for sha we hold (any state)
    connected_peers: FrozenSet[str]
    owner: Optional[str]                    # the part's owner node id, if the caller knows it
    relay: EstView
    p2p: Mapping[str, EstView]
    latched_peers: FrozenSet[str]
    global_latch: bool
    now: float
    have_wait_started: Optional[float] = None
    round_idx: Optional[int] = None
    prefetch: bool = False
    mode: str = "auto"


@dataclass(frozen=True)
class Decision:
    action: str                 # "relay" | "wait_have" | "p2p"
    peer: Optional[str]
    reason: str


def p2p_preferred(relay: EstView, p2p: EstView) -> bool:
    """Case 3 vs 4: P2P is preferred unless BOTH paths are known (>= 2 samples) and
    R_rly > 1.25 x R_p2p. UNKNOWN on either side keeps P2P (bootstrap, sec. 2.2)."""
    if not (relay.known and p2p.known):
        return True
    return not (relay.ewma > HYSTERESIS * p2p.ewma)


class PathPolicy:
    """PURE: decide() depends only on its PolicyInput. No clock reads, no I/O, no state."""

    @staticmethod
    def decide(inp: PolicyInput) -> Decision:
        if inp.mode == "off" or inp.mode == "force-relay":
            return Decision("relay", None, "mode_" + inp.mode)
        if inp.global_latch:
            return Decision("relay", None, "global_latch")
        cands = sorted(p for p in inp.have_peers
                       if p in inp.connected_peers and p not in inp.latched_peers)
        # case 2: no usable HAVE
        if not cands:
            owner_ok = (inp.owner is not None and inp.owner in inp.connected_peers
                        and inp.owner not in inp.latched_peers)
            if inp.relay.capped and owner_ok:
                if inp.have_wait_started is None or inp.now - inp.have_wait_started < W_HAVE_S:
                    return Decision("wait_have", inp.owner, "relay_capped_wait_have")
                return Decision("relay", None, "no_have_after_wait")
            return Decision("relay", None, "no_have")
        # forced P2P modes skip the hysteresis (pair type is enforced during the fetch)
        if inp.mode in ("force-direct", "force-turn"):
            return Decision("p2p", _best_peer(cands, inp), "mode_" + inp.mode)
        # case 3: any candidate where P2P is preferred
        pref = [p for p in cands if p2p_preferred(inp.relay, inp.p2p.get(p, UNKNOWN))]
        if pref:
            peer = _best_peer(pref, inp)
            known = inp.p2p.get(peer, UNKNOWN).known
            return Decision("p2p", peer, "p2p_preferred" if known and inp.relay.known else "p2p_unknown")
        # case 4: relay preferred; re-probe every 3rd round during prefetch only
        if inp.prefetch and inp.round_idx is not None and inp.round_idx % REPROBE_EVERY == 0:
            return Decision("p2p", _best_peer(cands, inp), "reprobe")
        return Decision("relay", None, "relay_preferred")


def _best_peer(cands, inp: PolicyInput) -> str:
    """UNKNOWN peers first (bootstrap), then the fastest known; owner breaks ties; then name."""
    def key(p):
        v = inp.p2p.get(p, UNKNOWN)
        return (0 if not v.known else 1, -(v.ewma or 0.0), 0 if p == inp.owner else 1, p)
    return sorted(cands, key=key)[0]


def hedge_reason(elapsed: float, first_chunk: bool, bytes_done: int, size: Optional[int],
                 relay: EstView, stalled_s: Optional[float] = None) -> Optional[str]:
    """Pure sec. 2.4 hedge test. Returns a reason string when the caller must fall back to
    the relay NOW, else None. ``stalled_s`` (2026-10-13, p2p2): seconds since the received byte count last
    grew; a fetch that started and then stopped (the holder's sidecar died mid-transfer, design S2) hedges after
    T_FIRST_S without one, instead of waiting for the sidecar's own 120 s NoProgressS (sec. 2.4: "the hedge must not
    wait for it"). None = not tracked (the step-1 behaviour)."""
    if not first_chunk:
        return "no_first_chunk" if elapsed >= T_FIRST_S else None
    if stalled_s is not None and stalled_s >= T_FIRST_S:
        return "stalled"
    if size and bytes_done > 0 and relay.ewma:
        projected_total = elapsed * float(size) / float(bytes_done)
        if projected_total > HEDGE_PROJECTION * float(size) / relay.ewma:
            return "slow_projection"
    return None


# ---------------------------------------------------------------- sidecar interface
@dataclass
class FetchProgress:
    bytes_done: int = 0
    size: Optional[int] = None
    done: bool = False
    data: Optional[bytes] = None
    error: Optional[str] = None             # "sidecar_crash" | other
    t_first_byte: Optional[float] = None    # seconds since fetch start
    pair_local_type: Optional[str] = None   # host|srflx|prflx|relay
    pair_remote_type: Optional[str] = None
    addr_class: Optional[str] = None        # "public" | "private" | "loopback" | ...
    pair_remote_addr: Optional[str] = None  # p2p2 10-13: the selected pair's remote address (a TURN relay's IP for relay)


class SidecarError(Exception):
    """The sidecar process died or the control channel broke."""


class FetchHandle:
    def poll(self) -> FetchProgress:            # may raise SidecarError
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError


class SidecarClient:
    """What the real pion sidecar (step 2) must provide. All methods may raise SidecarError."""

    def connect(self, peer: str) -> bool:
        raise NotImplementedError

    def connected_peers(self) -> FrozenSet[str]:
        raise NotImplementedError

    def peers_with(self, sha: str) -> FrozenSet[str]:
        """Peers that sent HAVE for sha."""
        raise NotImplementedError

    def announce(self, sha: str, data: bytes) -> None:
        """HAVE: serve these bytes to peers that WANT sha."""
        raise NotImplementedError

    def fetch(self, peer: str, sha: str) -> FetchHandle:
        """WANT: start fetching sha from peer; progress + cancel via the handle."""
        raise NotImplementedError

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- peer auth (owner Q4: per-miner keys)
class AuthError(Exception):
    pass


class PeerAuth:
    """Signs / verifies signalling records with a per-miner key. verify() raises AuthError on
    any failure and returns the verified payload dict on success. It never 'accepts by default'."""

    def sign(self, payload: dict) -> dict:
        raise NotImplementedError

    def verify(self, envelope: dict, trusted: Mapping[str, bytes], now: Optional[float] = None,
               max_age_s: Optional[float] = None) -> dict:
        raise NotImplementedError


_SIG_DOMAIN = b"neurahash-p2p-signal-v1\n"


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (Ed25519PrivateKey,
                                                                   Ed25519PublicKey)
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
    HAVE_ED25519 = True
except Exception:  # pragma: no cover - depends on the interpreter
    HAVE_ED25519 = False


class StubPeerAuth(PeerAuth):
    """Used ONLY when `cryptography` is missing. Refuses everything -- never silently accepts."""

    def sign(self, payload: dict) -> dict:
        raise AuthError("peer auth unavailable: `cryptography` not importable")

    def verify(self, envelope, trusted, now=None, max_age_s=None) -> dict:
        raise AuthError("peer auth unavailable: `cryptography` not importable")


if HAVE_ED25519:
    class Ed25519PeerAuth(PeerAuth):
        """Envelope = {miner_id, payload, sig(b64)}; signature over DOMAIN + canonical
        JSON of {miner_id, payload}. Keys come from files/env at runtime -- never from source."""

        def __init__(self, miner_id: str, private_key: "Ed25519PrivateKey"):
            self.miner_id = str(miner_id)
            self._sk = private_key

        @classmethod
        def generate(cls, miner_id: str) -> "Ed25519PeerAuth":
            return cls(miner_id, Ed25519PrivateKey.generate())

        @classmethod
        def from_raw_seed(cls, miner_id: str, seed32: bytes) -> "Ed25519PeerAuth":
            return cls(miner_id, Ed25519PrivateKey.from_private_bytes(seed32))

        @classmethod
        def from_key_file(cls, miner_id: str, path: str) -> "Ed25519PeerAuth":
            with open(path, "rb") as f:
                raw = f.read().strip()
            try:
                return cls(miner_id, serialization.load_pem_private_key(raw, password=None))
            except Exception:
                return cls.from_raw_seed(miner_id, bytes.fromhex(raw.decode("ascii")))

        def public_key_bytes(self) -> bytes:
            return self._sk.public_key().public_bytes(serialization.Encoding.Raw,
                                                      serialization.PublicFormat.Raw)

        def sign(self, payload: dict) -> dict:
            body = {"miner_id": self.miner_id, "payload": payload}
            sig = self._sk.sign(_SIG_DOMAIN + _canon(body))
            return {"miner_id": self.miner_id, "payload": payload,
                    "sig": base64.b64encode(sig).decode("ascii")}

        def verify(self, envelope, trusted, now=None, max_age_s=None) -> dict:
            try:
                mid = envelope["miner_id"]
                payload = envelope["payload"]
                sig = base64.b64decode(envelope["sig"], validate=True)
            except Exception as e:
                raise AuthError("malformed envelope: %s" % type(e).__name__)
            pk = trusted.get(mid)
            if pk is None:
                raise AuthError("unknown miner_id")
            try:
                Ed25519PublicKey.from_public_bytes(pk).verify(
                    sig, _SIG_DOMAIN + _canon({"miner_id": mid, "payload": payload}))
            except InvalidSignature:
                raise AuthError("bad signature")
            except Exception as e:
                raise AuthError("verify failed: %s" % type(e).__name__)
            if max_age_s is not None:
                ts = payload.get("ts") if isinstance(payload, dict) else None
                if not isinstance(ts, (int, float)) or now is None or abs(now - ts) > max_age_s:
                    raise AuthError("stale or missing ts")
            return payload

    def make_peer_auth(miner_id: str, key_path: Optional[str] = None) -> PeerAuth:
        return (Ed25519PeerAuth.from_key_file(miner_id, key_path) if key_path
                else Ed25519PeerAuth.generate(miner_id))
else:  # pragma: no cover
    def make_peer_auth(miner_id: str, key_path: Optional[str] = None) -> PeerAuth:
        return StubPeerAuth()


# ---------------------------------------------------------------- transport
class _Inflight:
    def __init__(self):
        self.ev = threading.Event()
        self.result: Optional[bytes] = None


class Transport:
    def __init__(self, sidecar: SidecarClient, *, mode: str = "auto",
                 log_path: Optional[str] = None, clock: Optional[Clock] = None,
                 allow_private: bool = False, sink: Optional[Callable[[dict], None]] = None):
        if mode not in MODES:
            raise ValueError("bad mode %r" % mode)
        self.sidecar = sidecar
        self.mode = mode
        self.clock = clock or Clock()
        self.allow_private = allow_private      # owner Q1: WAN only -> False in production
        self._lock = threading.RLock()
        self._relay = RateEst()
        self._p2p: Dict[str, RateEst] = {}
        self._bad_sha: Dict[str, int] = {}
        self._crashes: Dict[str, int] = {}
        self._latched: Dict[str, str] = {}      # peer -> reason
        self._global_latch: Optional[str] = None
        self._window = deque(maxlen=GLOBAL_WINDOW)   # True = fallback
        self._latch_at: Optional[float] = None       # #4a: when the global latch tripped / was last re-armed
        self._latch_calls = 0                        # #4a: fetch calls decided under the latch since then
        self._probe: Optional[str] = None            # #4a: the sha of the re-probe in flight, if any
        self._lost_races = 0
        self._reprobes = 0
        self._unlatches = 0
        self._inflight: Dict[str, _Inflight] = {}
        self._handles = set()
        self._closed = False
        self._counts: Dict[str, int] = {}
        self._bytes: Dict[str, int] = {}
        self._sink = sink
        self._logf = open(log_path, "a", encoding="ascii") if log_path else None

    # ---- logging (sec. 2.5)
    def _log(self, row: dict) -> None:
        row.setdefault("t", time.time())
        line = json.dumps(row, sort_keys=True, ensure_ascii=True)
        with self._lock:
            if self._logf is not None and not self._logf.closed:
                self._logf.write(line + "\n")
                self._logf.flush()
            if self._sink:
                self._sink(row)

    def _count(self, source: str, nbytes: int) -> None:
        with self._lock:
            self._counts[source] = self._counts.get(source, 0) + 1
            self._bytes[source] = self._bytes.get(source, 0) + int(nbytes)

    # ---- state snapshots for the pure policy
    def _policy_input(self, sha, owner, have_wait_started, round_idx, prefetch, probe=False) -> PolicyInput:
        try:
            haves = frozenset(self.sidecar.peers_with(sha))
            conn = frozenset(self.sidecar.connected_peers())
        except Exception:
            haves, conn = frozenset(), frozenset()
        with self._lock:
            return PolicyInput(
                sha=sha, have_peers=haves, connected_peers=conn, owner=owner,
                relay=self._relay.view(cap_check=True),
                p2p={p: e.view() for p, e in self._p2p.items()},
                latched_peers=frozenset(self._latched),
                global_latch=self._global_latch is not None and not probe,
                now=self.clock.now(), have_wait_started=have_wait_started,
                round_idx=round_idx, prefetch=prefetch, mode=self.mode)

    def _latch_peer(self, peer: str, reason: str) -> None:
        with self._lock:
            if peer not in self._latched:
                self._latched[peer] = reason
                newly = True
            else:
                newly = False
        if newly:
            self._log({"kind": "latch", "scope": "peer", "peer": peer, "decision_reason": reason})

    def _note_attempt(self, fallback: bool, sha: Optional[str] = None) -> None:
        if sha is not None and self._probe == sha:     # #4a: a probe's attempt decides the latch, not the window
            self._end_probe(sha, bool(fallback))
            return
        with self._lock:
            self._window.append(bool(fallback))
            trip = (self._global_latch is None and len(self._window) == GLOBAL_WINDOW
                    and sum(self._window) >= GLOBAL_FALLBACK_FRAC * GLOBAL_WINDOW)
            if trip:
                self._global_latch = "fallbacks_%d_of_%d" % (sum(self._window), GLOBAL_WINDOW)
                self._latch_at, self._latch_calls = self.clock.now(), 0
        if trip:
            self._log({"kind": "latch", "scope": "global", "decision_reason": self._global_latch})

    # ---- #4a (runner_n3_xport_2026-10-14): lost races and the bounded re-probe of the global latch
    def _note_lost_race(self, sha: str) -> None:
        """The raced relay copy arrived first: NOT a P2P failure (the window is untouched). A probe that loses its race
        re-arms the latch (no unlatch: the relay was faster)."""
        with self._lock:
            self._lost_races += 1
            probing = self._probe == sha
            if probing:
                self._probe = None
                self._latch_at, self._latch_calls = self.clock.now(), 0
        if probing:
            self._log({"kind": "reprobe", "sha": sha, "result": "lost_race", "decision_reason": self._global_latch})

    def _take_probe(self, sha: str) -> bool:
        """Under the global latch: count this fetch call and, when a probe is due and none is in flight, make THIS call
        the probe (the policy then runs without the global latch, once). -> True for the probe."""
        with self._lock:
            if self._global_latch is None or GLOBAL_REPROBE_EVERY <= 0:
                return False
            self._latch_calls += 1
            if self._probe is not None:
                return False
            now = self.clock.now()
            due = (self._latch_calls > GLOBAL_REPROBE_EVERY
                   or (self._latch_at is not None and now - self._latch_at >= GLOBAL_REPROBE_AFTER_S))
            if not due:
                return False
            self._probe, self._reprobes = sha, self._reprobes + 1
            calls, age = self._latch_calls, (now - self._latch_at) if self._latch_at is not None else None
        self._log({"kind": "reprobe", "sha": sha, "result": "start", "latched_calls": calls, "latched_s": age,
                   "decision_reason": self._global_latch})
        return True

    def _end_probe(self, sha: str, fallback: Optional[bool]) -> None:
        """The probe's outcome: False = P2P delivered -> the global latch and the window are cleared; True = a
        fallback -> re-armed; None = the policy chose the relay without a P2P attempt (no usable HAVE) -> ALSO
        re-armed: a probe is spent either way, so probes stay bounded (at most one per GLOBAL_REPROBE_EVERY latched
        calls or GLOBAL_REPROBE_AFTER_S) even when no peer ever answers (review 10-14 finding 2)."""
        with self._lock:
            if self._probe != sha:
                return
            self._probe = None
            was = self._global_latch
            if fallback is None or fallback:
                self._latch_at, self._latch_calls = self.clock.now(), 0
            else:
                self._global_latch, self._latch_at, self._latch_calls = None, None, 0
                self._window.clear()
                self._unlatches += 1
        self._log({"kind": "reprobe", "sha": sha,
                   "result": "no_attempt" if fallback is None else "fallback" if fallback else "p2p_ok",
                   "decision_reason": was})
        if not fallback:
            self._log({"kind": "unlatch", "scope": "global", "decision_reason": "reprobe_p2p_ok:" + str(was)})

    # ---- public API
    def record_relay(self, sha: str, nbytes: int, seconds: float) -> None:
        """Feed one relay GET timing (the hook times the HTTP GET after try_get returned None)."""
        if nbytes >= RELAY_SAMPLE_MIN_BYTES and seconds > 0:
            with self._lock:
                self._relay.add(nbytes / seconds)
                v = self._relay.view(cap_check=True)
            self._log({"kind": "relay_sample", "sha": sha, "size": int(nbytes), "seconds": seconds,
                       "r_rly": v.ewma, "relay_capped": v.capped})
        self._count("relay", nbytes)

    def offer(self, sha: str, data: bytes, name: Optional[str] = None) -> bool:
        """Announce a part we already PUT to the relay. No-op for non-part names. Never raises."""
        t0 = self.clock.now()
        if self.mode in ("off", "force-relay") or self._closed:
            return False
        if name is not None and not PART_NAME_RE.match(name):
            return False
        ok = hashlib.sha256(data).hexdigest() == sha
        err = None
        if ok:
            try:
                self.sidecar.announce(sha, data)
            except Exception as e:
                ok, err = False, type(e).__name__
        self._log({"kind": "offer", "sha": sha, "size": len(data), "name": name,
                   "sha_ok": hashlib.sha256(data).hexdigest() == sha,
                   "seconds": self.clock.now() - t0, "announced": ok, "error": err})
        return ok

    def try_get(self, sha: str, deadline: Optional[float] = None, *, owner: Optional[str] = None,
                round_idx: Optional[int] = None, prefetch: bool = False) -> Optional[bytes]:
        """Verified bytes for sha, or None (= use the relay). Never raises.
        ``deadline`` is an ABSOLUTE time on this transport's clock (Transport.clock.now());
        None means now + DEFAULT_FETCH_BUDGET_S. A concurrent call for the same sha shares the
        in-flight fetch instead of downloading twice."""
        try:
            return self._try_get(sha, deadline, owner, round_idx, prefetch)
        except Exception as e:     # never raise into the runner
            self._log({"kind": "try_get", "sha": sha, "source": "relay",
                       "decision_reason": "internal_error:" + type(e).__name__, "sha_ok": None,
                       "seconds": 0.0})
            return None

    # ---- step 3: the REAL race (main-session decision 2026-10-10, README_STEP1 "Decision")
    def fetch(self, sha: str, relay_fetch: Callable[[], bytes], deadline: Optional[float] = None, *,
              owner: Optional[str] = None, round_idx: Optional[int] = None, prefetch: bool = False) -> bytes:
        """The runner hook's entry point. Returns bytes for sha from the FIRST source that yields them:
        P2P bytes only if sha256 == sha; relay bytes exactly as the relay GET returned them (the caller's
        G3 check judges those, as today). On a hedge (sec. 2.4) the relay GET is started IN PARALLEL with the
        still-running P2P fetch; the first sha-verified copy wins and the loser is cancelled (P2P: handle
        cancel; relay: its thread is abandoned, its bytes discarded). A relay exception propagates unchanged
        (the caller's retry ladder handles it, as without a transport). P2P never raises into the caller."""
        race = _RelayRace(relay_fetch)
        res = None
        try:
            res = self._try_get(sha, deadline, owner, round_idx, prefetch, race=race)
        except Exception as e:     # never let a transport bug reach the runner: fall back to the relay
            self._log({"kind": "try_get", "sha": sha, "source": "relay",
                       "decision_reason": "internal_error:" + type(e).__name__, "sha_ok": None, "seconds": 0.0})
            res = None
        if res is not None:
            return res
        if race.started:
            data = race.result()                      # the relay's own exception propagates
            self._relay_sample(sha, len(data), race.secs)
            self._won(sha, "hedge-relay", data, race.secs)
            return data
        t0 = time.monotonic()
        data = relay_fetch()
        dt = time.monotonic() - t0
        self.record_relay(sha, len(data), dt)
        self._won(sha, "relay", data, dt)
        return data

    def _won(self, sha, source, data, seconds) -> None:
        """One row per fetch(): the source whose bytes the caller received (the sec. 2.5 join key)."""
        self._log({"kind": "won", "sha": sha, "source": source, "size": len(data) if data is not None else None,
                   "sha_ok": (hashlib.sha256(data).hexdigest() == sha) if data is not None else None,
                   "seconds": seconds})

    def _relay_sample(self, sha, nbytes, seconds) -> None:
        if seconds is not None:
            self.record_relay(sha, nbytes, max(float(seconds), 1e-9))

    def _try_get(self, sha, deadline, owner, round_idx, prefetch, race=None):
        t0 = self.clock.now()
        if deadline is None:
            deadline = t0 + DEFAULT_FETCH_BUDGET_S
        if self._closed or self.mode == "off":
            return None
        with self._lock:
            inf = self._inflight.get(sha)
            leader = inf is None
            if leader:
                inf = self._inflight[sha] = _Inflight()
        if not leader:
            # follower: wait for the leader's result (bounded by our deadline)
            # real-time wait on the leader's event; the deadline is checked on our clock
            while not inf.ev.wait(POLL_S):
                if self.clock.now() >= deadline:
                    break
            res = inf.result if inf.ev.is_set() else None
            self._log({"kind": "try_get", "sha": sha, "source": "p2p-shared" if res else "relay",
                       "decision_reason": "inflight_shared", "size": len(res) if res else None,
                       "sha_ok": True if res else None, "seconds": self.clock.now() - t0})
            if res is not None and race is not None:
                self._won(sha, "p2p-shared", res, self.clock.now() - t0)
            return res
        res = None
        try:
            res = self._lead(sha, deadline, owner, round_idx, prefetch, t0, race)
            return res
        finally:
            with self._lock:
                inf.result = res
                inf.ev.set()
                self._inflight.pop(sha, None)

    def _screen_peers(self) -> None:
        """2026-10-13 (p2p2): pair classification BEFORE the first byte (sec. 2.1, prior design sec. 3.3; owner Q1
        WAN only). A sidecar that reports each connected peer's selected pair (``peer_pair``) gets every peer whose
        pair is non-public or a hairpin latched here, so no fetch is ever started over it. A FakeSidecar without the
        method keeps the step-1 behaviour (classification at the first poll)."""
        fn = getattr(self.sidecar, "peer_pair", None)
        if fn is None:
            return
        try:
            conn = list(self.sidecar.connected_peers())
        except Exception:
            return
        for p in conn:
            with self._lock:
                if p in self._latched:
                    continue
            try:
                pr = fn(p)
            except Exception:
                continue
            if pr is None:
                continue
            why = self._bad_pair(pr, pre=True)
            if why in ("non_public_pair",):
                self._latch_peer(p, why + ":" + str(pr.addr_class))

    def _lead(self, sha, deadline, owner, round_idx, prefetch, t0, race=None):
        wait_started = None
        if self.mode not in ("off", "force-relay"):
            self._screen_peers()
        probe = self.mode not in ("off", "force-relay") and self._take_probe(sha)    # #4a: bounded re-probe
        try:
            return self._lead_body(sha, deadline, owner, round_idx, prefetch, t0, race, wait_started, probe)
        finally:
            if probe:
                self._end_probe(sha, None)     # still open = no P2P attempt was made: spent, re-armed (no-op otherwise)

    def _lead_body(self, sha, deadline, owner, round_idx, prefetch, t0, race, wait_started, probe):
        while True:
            inp = self._policy_input(sha, owner, wait_started, round_idx, prefetch, probe=probe)
            d = PathPolicy.decide(inp)
            if d.action == "wait_have":
                if wait_started is None:
                    wait_started = inp.now
                if self.clock.now() >= deadline:
                    d = Decision("relay", None, "deadline_waiting_have")
                else:
                    self.clock.sleep(POLL_S * 4)
                    continue
            if d.action == "relay":
                why = d.reason
                if getattr(self.sidecar, "no_candidates", False) and self.mode not in ("off", "force-relay"):
                    why += "|no_sidecar"          # auto without a P2P path (sidecar.kind none): the relay by construction
                self._log(self._row(sha, "relay", d.peer, why, t0, inp))
                return None
            return self._p2p_fetch(sha, d.peer, d.reason, deadline, t0, inp, race)

    def _row(self, sha, source, peer, reason, t0, inp: PolicyInput, **kw) -> dict:
        pv = inp.p2p.get(peer, UNKNOWN) if peer else UNKNOWN
        row = {"kind": "try_get", "sha": sha, "source": source, "peer": peer,
               "decision_reason": reason, "seconds": self.clock.now() - t0,
               "r_rly": inp.relay.ewma, "r_p2p": pv.ewma, "relay_capped": inp.relay.capped,
               "size": None, "sha_ok": None, "t_first_byte": None, "pair_local_type": None,
               "pair_remote_type": None, "addr_class": None}
        row.update(kw)
        return row

    def _p2p_fetch(self, sha, peer, reason, deadline, t0, inp, race=None):
        """One P2P fetch from `peer`. Without `race` (try_get): any hedge cancels P2P and returns None.
        With `race` (fetch): a hedge STARTS the relay GET in parallel and P2P keeps going; the first
        sha-verified copy wins. Returns verified bytes (P2P, or the race's verified relay copy) or None
        (the caller then uses the relay: the race's result if it was started, else a fresh relay GET)."""
        tf0 = self.clock.now()
        prog = FetchProgress()
        first = False
        t_first = None
        try:
            h = self.sidecar.fetch(peer, sha)
        except Exception as e:
            return self._crash(sha, peer, reason, t0, inp, prog, "start:" + type(e).__name__)
        with self._lock:
            self._handles.add(h)
        relay_failed = False
        last_bytes, last_change = -1, tf0          # stall tracking (hedge_reason stalled_s, 2026-10-13)
        try:
            while True:
                # the race: a verified relay copy that arrived first wins; P2P is cancelled
                if race is not None and race.started and not relay_failed and race.done():
                    if race.ok(sha):
                        _cancel(h)
                        # Fable review 10-14 F1: the losing leg's MEASURED progress is a P2P rate sample (0 B -> 0 B/s),
                        # so after MIN_SAMPLES lost races the peer is KNOWN and p2p_preferred's hysteresis can turn it
                        # relay_preferred; case 4's prefetch re-probe (every REPROBE_EVERY rounds) still tries it again
                        lost_secs = max(self.clock.now() - tf0, 1e-9)
                        with self._lock:
                            self._p2p.setdefault(peer, RateEst()).add(max(0, int(prog.bytes_done or 0)) / lost_secs)
                        self._note_lost_race(sha)     # #4a: the relay was faster -- NOT a P2P fallback
                        self._relay_sample(sha, len(race.data), race.secs)
                        self._log(self._row(sha, "hedge-relay", peer, reason + "|race_relay_won", t0, inp,
                                            size=len(race.data), sha_ok=True, t_first_byte=t_first,
                                            pair_local_type=prog.pair_local_type,
                                            pair_remote_type=prog.pair_remote_type, addr_class=prog.addr_class))
                        self._won(sha, "hedge-relay", race.data, race.secs)
                        return race.data
                    relay_failed = True     # relay raised or sent bad bytes: P2P is now the only hope
                try:
                    prog = h.poll()
                except Exception as e:
                    return self._crash(sha, peer, reason, t0, inp, prog, type(e).__name__)
                pair_kw = dict(pair_local_type=prog.pair_local_type,
                               pair_remote_type=prog.pair_remote_type, addr_class=prog.addr_class,
                               pair_remote_addr=prog.pair_remote_addr)
                # pair classification precedes the first byte (sec. 2.1; owner Q1: WAN only)
                bad_pair = self._bad_pair(prog)
                if bad_pair:
                    _cancel(h)
                    if bad_pair == "non_public_pair":
                        self._latch_peer(peer, bad_pair)
                    self._note_attempt(True, sha)
                    self._log(self._row(sha, "hedge-relay", peer, reason + "|" + bad_pair, t0, inp,
                                        **pair_kw))
                    return None
                if prog.error:
                    _cancel(h)
                    if str(prog.error).startswith("integrity"):
                        # 2026-10-13 (p2p2): the sidecar refused the holder's chunks against the holder's own chunk-hash
                        # list (3 chunk failures, xfer.go runFetch) -- a corrupt holder, same class as a sha-bad body
                        # (sec. 2.3 item 5 / design S3): latch it at once; nothing reached the caller.
                        with self._lock:
                            self._bad_sha[peer] = self._bad_sha.get(peer, 0) + 1
                            n_bad = self._bad_sha[peer]
                        if n_bad >= PEER_LATCH_BAD_SHA:
                            self._latch_peer(peer, "sha_bad")
                        self._note_attempt(True, sha)
                        self._log(self._row(sha, "hedge-relay", peer, reason + "|integrity", t0, inp,
                                            size=prog.size, sha_ok=False, t_first_byte=t_first, **pair_kw))
                        return None
                    return self._crash(sha, peer, reason, t0, inp, prog, prog.error)
                elapsed = self.clock.now() - tf0
                if prog.bytes_done != last_bytes:
                    last_bytes, last_change = prog.bytes_done, self.clock.now()
                if not first and (prog.bytes_done > 0 or prog.done):
                    first = True
                    t_first = prog.t_first_byte if prog.t_first_byte is not None else elapsed
                if prog.done:
                    data = prog.data or b""
                    ok = hashlib.sha256(data).hexdigest() == sha
                    secs = max(self.clock.now() - tf0, 1e-9)
                    if not ok:
                        with self._lock:
                            self._bad_sha[peer] = self._bad_sha.get(peer, 0) + 1
                            n_bad = self._bad_sha[peer]
                        if n_bad >= PEER_LATCH_BAD_SHA:
                            self._latch_peer(peer, "sha_bad")
                        self._note_attempt(True, sha)
                        self._log(self._row(sha, "hedge-relay", peer, reason + "|sha_bad", t0, inp,
                                            size=len(data), sha_ok=False, t_first_byte=t_first,
                                            **pair_kw))
                        return None            # never returned, never a G3 miss
                    with self._lock:
                        self._p2p.setdefault(peer, RateEst()).add(len(data) / secs)
                    raced = race is not None and race.started
                    self._note_attempt(False, sha)
                    self._count("hedge-p2p" if raced else "p2p", len(data))
                    self._log(self._row(sha, "hedge-p2p" if raced else "p2p", peer, reason, t0, inp,
                                        size=len(data), sha_ok=True, t_first_byte=t_first, **pair_kw))
                    if race is not None:
                        self._won(sha, "hedge-p2p" if raced else "p2p", data, secs)
                    return data
                if race is not None and race.started:
                    hr = "deadline" if self.clock.now() >= deadline else None
                    if hr is None and first and self.clock.now() - last_change >= NO_PROGRESS_GIVE_UP_S:
                        hr = "stalled_after_race"      # the relay copy is running; stop holding a dead P2P fetch
                else:
                    hr = hedge_reason(elapsed, first, prog.bytes_done, prog.size, inp.relay,
                                      stalled_s=self.clock.now() - last_change)
                    if hr is None and self.clock.now() >= deadline:
                        hr = "deadline"
                    if hr and hr != "deadline" and race is not None:
                        race.start()             # the hedge: relay GET in PARALLEL, P2P keeps going
                        self._log(self._row(sha, "hedge", peer, reason + "|race_start:" + hr, t0, inp,
                                            kind="hedge", size=prog.size, t_first_byte=t_first, **pair_kw))
                        hr = None
                if hr:
                    _cancel(h)
                    self._note_attempt(True, sha)
                    self._log(self._row(sha, "hedge-relay", peer, reason + "|" + hr, t0, inp,
                                        size=prog.size, t_first_byte=t_first, **pair_kw))
                    return None
                self.clock.sleep(POLL_S)
        finally:
            with self._lock:
                self._handles.discard(h)

    def _bad_pair(self, prog: FetchProgress, pre: bool = False) -> Optional[str]:
        if prog.addr_class is not None and prog.addr_class != "public" and not self.allow_private:
            return "non_public_pair"
        rt = prog.pair_remote_type
        if rt is not None:
            if self.mode == "force-direct" and rt == "relay":
                return "pair_not_direct"
            if self.mode == "force-turn" and rt != "relay":
                return "pair_not_turn"
        return None

    def _crash(self, sha, peer, reason, t0, inp, prog, err):
        with self._lock:
            self._crashes[peer] = self._crashes.get(peer, 0) + 1
            n = self._crashes[peer]
        if n >= PEER_LATCH_CRASHES:
            self._latch_peer(peer, "sidecar_crashes_%d" % n)
        self._note_attempt(True, sha)
        self._log(self._row(sha, "hedge-relay", peer, reason + "|crash:" + str(err), t0, inp,
                            size=prog.size, pair_local_type=prog.pair_local_type,
                            pair_remote_type=prog.pair_remote_type, addr_class=prog.addr_class))
        return None

    def stats(self) -> dict:
        with self._lock:
            rv = self._relay.view(cap_check=True)
            return {
                "mode": self.mode,
                "fetches": dict(self._counts), "bytes": dict(self._bytes),
                "r_rly": rv.ewma, "relay_samples": rv.n, "relay_capped": rv.capped,
                "r_p2p": {p: e.ewma for p, e in self._p2p.items()},
                "latched_peers": dict(self._latched), "global_latch": self._global_latch,
                "lost_races": self._lost_races, "reprobes": self._reprobes, "unlatches": self._unlatches,
                "sha_bad": dict(self._bad_sha), "crashes": dict(self._crashes),
                "sidecar": (self.sidecar.stats() if callable(getattr(self.sidecar, "stats", None)) else
                            type(self.sidecar).__name__),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            hs = list(self._handles)
        for h in hs:
            _cancel(h)
        try:
            self.sidecar.close()
        except Exception:
            pass
        with self._lock:
            if self._logf is not None:
                self._logf.close()


def _cancel(h) -> None:
    try:
        h.cancel()
    except Exception:
        pass


def build(mode: str, sidecar: Optional[SidecarClient], **kw) -> Optional[Transport]:
    """Factory the runner copy (step 3) calls. Returns None under --p2p off or without a sidecar,
    so NetAnchor.transport stays None and the HTTP path is byte-for-byte unchanged."""
    if mode == "off" or sidecar is None:
        return None
    return Transport(sidecar, mode=mode, **kw)


# ---------------------------------------------------------------- step 3: the race, config, runner factory
class _RelayRace:
    """The relay leg of fetch()'s race: relay_fetch() on a daemon thread, started at most once. A loser
    relay thread cannot be interrupted mid-read (urllib); it is abandoned and its bytes discarded."""

    def __init__(self, fn: Callable[[], bytes]):
        self.fn = fn
        self.ev = threading.Event()
        self.data: Optional[bytes] = None
        self.exc: Optional[BaseException] = None
        self.started = False
        self.secs: Optional[float] = None

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        t0 = time.monotonic()

        def run():
            try:
                self.data = self.fn()
            except BaseException as e:      # noqa: BLE001 -- handed to the caller by result()
                self.exc = e
            finally:
                self.secs = time.monotonic() - t0
                self.ev.set()
        threading.Thread(target=run, name="p2p-relay-race", daemon=True).start()

    def done(self) -> bool:
        return self.ev.is_set()

    def ok(self, sha: str) -> bool:
        return (self.ev.is_set() and self.exc is None and self.data is not None
                and hashlib.sha256(self.data).hexdigest() == sha)

    def result(self) -> bytes:
        self.ev.wait()
        if self.exc is not None:
            raise self.exc
        return self.data


class NullSidecar(SidecarClient):
    """No peers, no HAVEs: every decision is `relay`. Lets --p2p force-relay run the hook (timing, jsonl,
    TRAIN.json stats) with zero P2P bytes and no sidecar process. p2p2: also --p2p auto with sidecar.kind none +
    auto_without_sidecar (no TURN, one NAT: the owner's 2026-10-13 smoke) -- every row's reason then carries |no_sidecar."""
    no_candidates = True

    def connect(self, peer):
        return False

    def connected_peers(self):
        return frozenset()

    def peers_with(self, sha):
        return frozenset()

    def announce(self, sha, data):
        return None

    def fetch(self, peer, sha):
        raise SidecarError("NullSidecar has no peers")


SECRET_KEY_RE = re.compile(r"(secret|passw|credential|token|private|psk|signal_key|sign_key|seed|api_key)", re.I)
SIDECAR_KINDS = ("none", "pion")   # 2026-10-13 (p2p2): "pion" = PionSidecar below (step-2 sidecar: fetch_progress,
#                                    fetch_cancel, allow_private_relay switch, per-miner signed signalling)
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
NON_SECRET_KEYS = ("allow_private_relay", "peer_pubkeys", "signal_key_from")   # match SECRET_KEY_RE, hold no secret


def load_config(path: Optional[str]) -> dict:
    """--p2p-config: a JSON object with NO secrets. A key that names a secret must end in `_env` (it holds the NAME of an
    environment variable, e.g. "turn_password_env": "NH_TURN_PASSWORD") or in `_file` (it holds a PATH to a file that
    holds the secret, or a {node: path} map; 2026-10-13); a PEM block, a 64-hex value under a `_file` key, or a
    secret-named key holding a value is refused. -> the dict (empty when path is falsy). Raises ValueError."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("--p2p-config must hold a JSON object")
    bad = []

    def walk(o, where, under_file=False):
        if isinstance(o, dict):
            for k, v in o.items():
                kk = "%s.%s" % (where, k) if where else str(k)
                ks = str(k)
                is_file = ks.endswith("_file")
                if (SECRET_KEY_RE.search(ks) and not (ks.endswith("_env") or is_file) and not under_file
                        and ks not in NON_SECRET_KEYS):
                    bad.append(kk)
                walk(v, kk, under_file or is_file)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, "%s[%d]" % (where, i), under_file)
        elif isinstance(o, str) and ("-----BEGIN" in o or (under_file and _HEX64.match(o.strip()))):
            bad.append(where)
    walk(cfg, "")
    if bad:
        raise ValueError("--p2p-config %s holds secret-looking fields %s: put secrets in environment variables or files "
                         "and name them with a `<field>_env` / `<field>_file` key" % (path, ", ".join(bad[:5])))
    return cfg


def _pion_problems(sc: dict) -> list:
    """Static checks of a `sidecar` block of kind pion (no file or env read here; build_from_config reads them)."""
    probs = []
    if not sc.get("exe") or not _HEX64.match(str(sc.get("exe_sha256", ""))):
        probs.append("sidecar.kind pion needs sidecar.exe (path, relative to the runner dir allowed) and "
                     "sidecar.exe_sha256 (64 hex)")
    if not (sc.get("signal_key_env") or sc.get("signal_key_file") or sc.get("signal_key_from") == "relay_token"):
        probs.append("sidecar.kind pion needs signal_key_env, signal_key_file or signal_key_from \"relay_token\" (the "
                     "shared AES-GCM sealing key)")
    if not (sc.get("sign_key_env") or sc.get("sign_key_file")) or not isinstance(sc.get("peer_pubkeys"), dict):
        probs.append("sidecar.kind pion needs per-miner signed signalling (owner Q4): sign_key_env or sign_key_file "
                     "(this node's Ed25519 seed) AND peer_pubkeys {node: 64-hex public key}")
    if sc.get("signal_key_from") not in (None, "relay_token"):
        probs.append("sidecar.signal_key_from: only \"relay_token\"")
    pol = str(sc.get("ice_policy", "nohost"))
    if pol not in ("nohost", "relay"):
        probs.append("sidecar.ice_policy %r: nohost (direct + TURN) or relay (TURN only); 'all' (host candidates, "
                     "LAN) is refused (WAN-only)" % pol)
    if pol == "relay" and not sc.get("turn"):
        probs.append("sidecar.ice_policy relay needs sidecar.turn (at least one TURN URL)")
    if sc.get("turn") and not (sc.get("turn_user") and (sc.get("turn_password_env") or sc.get("turn_password_file"))):
        probs.append("sidecar.turn needs turn_user and turn_password_env / turn_password_file")
    if sc.get("allow_private_relay") and pol != "relay":
        probs.append("sidecar.allow_private_relay needs ice_policy relay")
    if sc.get("loopback"):
        probs.append("sidecar.loopback is a code-test mode (host candidates on 127.0.0.1): refused here")
    return probs


def config_problems(mode: str, cfg: dict) -> list:
    """Launch refusals for (mode, config). force-relay needs no sidecar; auto / force-turn / force-direct need a
    sidecar kind that has a client in this build."""
    probs = []
    if mode not in MODES:
        probs.append("--p2p %r is not one of %s" % (mode, "|".join(MODES)))
        return probs
    if mode in ("off", "force-relay"):
        return probs
    sc = (cfg or {}).get("sidecar") or {}
    kind = str(sc.get("kind", "none"))
    if kind == "none":
        if not (mode == "auto" and sc.get("auto_without_sidecar") is True):
            probs.append("--p2p %s needs a sidecar: --p2p-config with sidecar.kind (only force-relay runs without one; "
                         "auto runs without one only with sidecar.auto_without_sidecar true)" % mode)
    elif kind not in SIDECAR_KINDS:
        probs.append("--p2p-config sidecar.kind %r has no client in this build; built kinds: %s"
                     % (kind, ", ".join(SIDECAR_KINDS)))
    elif kind == "pion":
        probs += _pion_problems(sc)
        if mode == "force-direct" and str(sc.get("ice_policy", "nohost")) == "relay":
            probs.append("--p2p force-direct with sidecar.ice_policy relay can never fetch (TURN-only candidates)")
        if mode == "force-turn" and not sc.get("turn"):
            probs.append("--p2p force-turn needs sidecar.turn")
    return probs


def _read_secret(sc: dict, field: str, node: Optional[str], env: Mapping[str, str], base_dir: str) -> Optional[str]:
    """`<field>_env` -> os.environ[name]; `<field>_file` -> the stripped contents of the file (a {node: path} map picks
    this node's). Relative paths resolve against base_dir. None when neither key is set. Raises ValueError when a named
    source is missing or empty. The secret is returned to the caller only; it is never logged."""
    en, fn = sc.get(field + "_env"), sc.get(field + "_file")
    if en:
        v = env.get(str(en), "")
        if not v.strip():
            raise ValueError("environment variable %s (sidecar.%s_env) is not set" % (en, field))
        return v.strip()
    if fn:
        if isinstance(fn, dict):
            if node not in fn:
                raise ValueError("sidecar.%s_file has no entry for node %s" % (field, node))
            fn = fn[node]
        path = str(fn)
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        try:
            with open(path, "r", encoding="ascii") as f:
                v = f.read().strip()
        except OSError as e:
            raise ValueError("sidecar.%s_file %s: %s" % (field, path, type(e).__name__))
        if not v:
            raise ValueError("sidecar.%s_file %s is empty" % (field, path))
        return v
    return None


def derive_signal_key(relay_token: str, sns: str) -> str:
    """signal_key_from "relay_token": the sealing key of one run's signalling records, derived from the relay token every
    node of the fleet already holds (no new secret to move between boxes) and the run's signalling namespace."""
    import hmac
    return hmac.new(str(relay_token).encode("utf-8"), b"neurahash-p2p-signal-v1/" + str(sns).encode("utf-8"),
                    hashlib.sha256).hexdigest()


def sidecar_name(node: str) -> str:
    """The runner's node id (A, B, C ...) -> the sidecar's node name ([a-z0-9_-]{1,32})."""
    s = re.sub(r"[^a-z0-9_-]", "_", str(node).lower())[:32]
    if not s:
        raise ValueError("empty node name")
    return s


def signal_namespace(ns: str) -> Tuple[str, str]:
    """The runner's anchor namespace -> (name_prefix, ns) for the sidecar's signalling records. The sidecar build only
    writes names under p2ptest/ (main.go validate); the run's namespace keeps every run's records apart."""
    tag = re.sub(r"[^a-z0-9_-]", "_", str(ns).lower()).strip("_")[:80] or "run"
    prefix = "p2ptest/%s/" % tag
    return prefix, prefix + "s"


def build_from_config(mode: str, cfg: dict, log_path: Optional[str] = None,
                      sidecar: Optional[SidecarClient] = None, *, node: Optional[str] = None,
                      nodes: Optional[Iterable[str]] = None, relay_url: Optional[str] = None,
                      relay_token: Optional[str] = None, base_dir: Optional[str] = None,
                      work_dir: Optional[str] = None, env: Optional[Mapping[str, str]] = None,
                      clock: Optional[Clock] = None) -> Optional[Transport]:
    """The runner's factory: None under off (NetAnchor.transport stays None, the HTTP path is unchanged);
    force-relay -> a Transport over NullSidecar unless a sidecar is given; other modes need `sidecar` (a CPU
    test passes a fake one) or a sidecar.kind pion config (2026-10-13: PionSidecar, started later by
    ``Transport.sidecar.start(ns)``; the runner calls that once its anchor namespace is known)."""
    if mode == "off":
        return None
    probs = config_problems(mode, cfg) if sidecar is None else []
    if probs:
        raise ValueError("; ".join(probs))
    allow_private = False
    if sidecar is None:
        sc = (cfg or {}).get("sidecar") or {}
        if mode != "force-relay" and str(sc.get("kind")) == "pion":
            if not node or nodes is None or not relay_url:
                raise ValueError("sidecar.kind pion needs node, nodes and relay_url from the runner")
            sidecar = PionSidecar.from_config(sc, node=node, nodes=list(nodes), relay_url=relay_url,
                                              relay_token=relay_token or "", base_dir=base_dir or os.getcwd(),
                                              work_dir=work_dir or os.getcwd(), env=env if env is not None
                                              else os.environ)
            allow_private = bool(sc.get("allow_private_relay"))      # DIAGNOSTIC only (a LAN TURN in a CPU test)
        else:
            sidecar = NullSidecar()
    return Transport(sidecar, mode=mode, log_path=log_path, allow_private=allow_private, clock=clock)


# ---------------------------------------------------------------- the real pion sidecar (2026-10-13, p2p2)
def _addr_summary(stripes, own_ips) -> Optional[FetchProgress]:
    """The selected pair of a peer's stripes -> FetchProgress(pair_local_type, pair_remote_type, addr_class), worst
    stripe wins: any non-public remote address -> that class; a non-relayed remote whose address is one of OUR public
    addresses -> "hairpin" (one NAT, not a WAN path: owner Q1). None when no stripe has a selected pair yet."""
    best = None
    for st in stripes or []:
        if not isinstance(st, dict):
            continue
        loc, rem = st.get("local"), st.get("remote")
        if not isinstance(loc, dict) or not isinstance(rem, dict):
            continue
        cls = str(rem.get("class") or "unknown")
        rt, lt = str(rem.get("type") or ""), str(loc.get("type") or "")
        if cls == "public" and rt != "relay" and rem.get("addr") in own_ips:
            cls = "hairpin"
        cur = FetchProgress(pair_local_type=lt, pair_remote_type=rt, addr_class=cls,
                            pair_remote_addr=str(rem.get("addr")) if rem.get("addr") else None)
        if best is None or (best.addr_class == "public" and cls != "public"):
            best = cur
    return best


class _PionHandle(FetchHandle):
    def __init__(self, sc: "PionSidecar", fid: str, peer: str, sha: str, size: Optional[int]):
        self.sc, self.id, self.peer, self.sha, self.size = sc, fid, peer, sha, size
        self.t0 = time.monotonic()
        self.bytes = 0
        self.ttfb: Optional[float] = None
        self.reply: Optional[dict] = None
        self.data: Optional[bytes] = None
        self.data_sha_ok: Optional[bool] = None
        self.cancelled = False
        self.lock = threading.Lock()

    def on_event(self, ev: dict) -> None:
        with self.lock:
            name = ev.get("ev")
            if name == "fetch_progress":
                self.bytes = max(self.bytes, int(ev.get("bytes") or 0))
                if ev.get("ttfb_s") is not None and self.ttfb is None:
                    self.ttfb = float(ev["ttfb_s"])
                if ev.get("size"):
                    self.size = int(ev["size"])
            elif name in ("fetched", "fetch_failed", "error", "sidecar_exit"):
                self.reply = dict(ev)
                if ev.get("ttfb_s") is not None and self.ttfb is None:
                    self.ttfb = float(ev["ttfb_s"])

    def on_data(self, data: bytes) -> None:
        with self.lock:
            self.data = data
            self.data_sha_ok = hashlib.sha256(data).hexdigest() == self.sha

    def poll(self) -> FetchProgress:
        pr = self.sc.peer_pair(self.peer) or FetchProgress()
        with self.lock:
            rep, data = self.reply, self.data
            out = FetchProgress(bytes_done=self.bytes, size=self.size, t_first_byte=self.ttfb,
                                pair_local_type=pr.pair_local_type, pair_remote_type=pr.pair_remote_type,
                                addr_class=pr.addr_class, pair_remote_addr=pr.pair_remote_addr)
        if rep is None:
            if self.sc.dead:
                raise SidecarError("sidecar process exited")
            return out
        ev = rep.get("ev")
        if ev == "sidecar_exit":
            raise SidecarError("sidecar process exited")
        if ev == "fetched" and rep.get("verified"):
            if rep.get("stripes"):
                self.sc._note_stripes(self.peer, rep.get("stripes"))
                pr = self.sc.peer_pair(self.peer) or pr
                out.pair_local_type, out.pair_remote_type, out.addr_class = (pr.pair_local_type,
                                                                             pr.pair_remote_type, pr.addr_class)
                out.pair_remote_addr = pr.pair_remote_addr
            if data is None:
                if not rep.get("delivered", True):
                    out.error = "deliver_failed:%s" % str(rep.get("deliver_error"))[:60]
                    return out
                if time.monotonic() - float(rep.get("_rx_t", time.monotonic())) > 30.0:
                    out.error = "deliver_timeout"
                return out                    # the frame is still on the data socket
            out.bytes_done, out.size, out.done, out.data = len(data), len(data), True, data
            return out
        err = str(rep.get("error") or ev)
        out.error = "integrity:" + err if err.startswith("integrity") else ("cancelled" if rep.get("cancelled")
                                                                           else "fetch_failed:" + err[:120])
        return out

    def cancel(self) -> None:
        with self.lock:
            if self.cancelled or (self.reply is not None):
                return
            self.cancelled = True
        self.sc._send_async("fetch_cancel", sha=self.sha, peer=self.sc._sc_name(self.peer))


class PionSidecar(SidecarClient):
    """Client of the pion WebRTC sidecar (E:/aiCrypto_p2p_wt p2p/webrtc/sidecar, step-2 build: fetch_progress,
    fetch_cancel, allow_private_relay, sign_key/peer_pubkeys). One process per node, spoken to over stdin/stdout JSON
    lines; delivered parts arrive on its authenticated 127.0.0.1 data socket and are re-hashed here AND again by the
    Transport. Signalling records travel through the relay (sealed with the shared signal key, signed per miner).
    Peers are the runner's node ids (A, B, C); the sidecar's own names are their lowercase forms. Connections are made
    by a background thread: every node calls `connect` for every peer in the same time slot (epoch = unix time //
    slot_s), so both ends agree on the epoch without any extra record; a failed slot is retried in the next one."""

    def __init__(self, exe: str, node: str, peers: Iterable[str], relay_url: str, relay_token: str, *,
                 signal_key: str, sign_seed_hex: Optional[str], peer_pubkeys: Optional[Mapping[str, str]],
                 work_dir: str, ice_policy: str = "nohost", turn: Iterable[str] = (), turn_user: str = "",
                 turn_pass: str = "", stun: Iterable[str] = (), allow_private_relay: bool = False,
                 slot_s: float = 120.0, connect_timeout_s: float = 60.0, keep_parts: int = 4,
                 extra: Optional[Mapping] = None):
        self.exe = exe
        self.node = str(node)
        self.peers = [str(p) for p in peers if str(p) != str(node)]
        self._names = dict((sidecar_name(p), p) for p in self.peers + [self.node])
        self.relay_url, self.relay_token = relay_url, relay_token
        self.signal_key, self.sign_seed_hex, self.peer_pubkeys = signal_key, sign_seed_hex, dict(peer_pubkeys or {})
        self.work_dir = work_dir
        self.ice_policy, self.turn, self.turn_user, self.turn_pass = ice_policy, list(turn), turn_user, turn_pass
        self.stun, self.allow_private_relay = list(stun), bool(allow_private_relay)
        self.slot_s, self.connect_timeout_s, self.keep_parts = float(slot_s), float(connect_timeout_s), int(keep_parts)
        self.extra = dict(extra or {})
        self.p = None
        self.dead = False
        self.closed = False
        self.ready: Optional[dict] = None
        self.ns: Optional[str] = None
        self._lock = threading.RLock()
        self._wlock = threading.Lock()
        self._seq = 0
        self._waiters: Dict[str, Callable[[dict], None]] = {}
        self._connected: Dict[str, dict] = {}         # peer -> {"epoch":..., "t":...}
        self._stripes: Dict[str, list] = {}
        self._own_ips = set()
        self._haves: Dict[str, Dict[str, int]] = {}   # sha -> {peer: size}
        self._handles: Dict[str, _PionHandle] = {}
        self._registered = deque()
        self._connecting = set()
        self._listeners = []                          # tests: f(ev) for every event (e.g. the S2 kill hook)
        self.counts = {"connect_ok": 0, "connect_failed": 0, "peer_closed": 0, "registered": 0, "fetches": 0}
        self._evf = None
        self._data_sock = None
        self._local_token = None

    # ---- construction from the --p2p-config `sidecar` block
    @classmethod
    def from_config(cls, sc: dict, *, node, nodes, relay_url, relay_token, base_dir, work_dir, env):
        exe = str(sc["exe"])
        if not os.path.isabs(exe):
            exe = os.path.join(base_dir, exe)
        if not os.path.isfile(exe):
            raise ValueError("sidecar.exe %s does not exist" % exe)
        h = hashlib.sha256()
        with open(exe, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        if h.hexdigest() != str(sc["exe_sha256"]).lower():
            raise ValueError("sidecar.exe %s sha256 %s != sidecar.exe_sha256 %s" % (exe, h.hexdigest()[:16],
                                                                                   str(sc["exe_sha256"])[:16]))
        sk = _read_secret(sc, "signal_key", node, env, base_dir)
        if sk is None and sc.get("signal_key_from") == "relay_token":
            if not relay_token:
                raise ValueError("signal_key_from relay_token needs the relay token")
            sk = "relay_token"                   # derived per signalling namespace in start() (derive_signal_key)
        elif not sk or not _HEX64.match(sk):
            raise ValueError("the signal key must be 64 hex chars")
        seed = _read_secret(sc, "sign_key", node, env, base_dir)
        if not seed or not _HEX64.match(seed):
            raise ValueError("the sign key (this node's Ed25519 seed) must be 64 hex chars")
        pk = dict((str(k), str(v)) for k, v in (sc.get("peer_pubkeys") or {}).items())
        missing = [n for n in nodes if n not in pk]
        if missing:
            raise ValueError("sidecar.peer_pubkeys lacks node(s) %s" % ",".join(missing))
        tp = _read_secret(sc, "turn_password", node, env, base_dir) if sc.get("turn") else ""
        extra = dict((k, sc[k]) for k in ("stripes", "chunk_bytes", "window_bytes", "max_part_bytes", "poll_s",
                                          "no_progress_timeout_s", "progress_interval_s", "ice_prime_ttl",
                                          "asn_lookup", "log_level") if k in sc)
        return cls(exe, node, nodes, relay_url, relay_token, signal_key=sk, sign_seed_hex=seed, peer_pubkeys=pk,
                   work_dir=work_dir, ice_policy=str(sc.get("ice_policy", "nohost")), turn=sc.get("turn") or [],
                   turn_user=str(sc.get("turn_user", "")), turn_pass=tp or "", stun=sc.get("stun") or [],
                   allow_private_relay=bool(sc.get("allow_private_relay")),
                   slot_s=float(sc.get("connect_slot_s", 120.0)),
                   connect_timeout_s=float(sc.get("connect_timeout_s", 60.0)),
                   keep_parts=int(sc.get("keep_parts", 4)), extra=extra)

    def _sc_name(self, peer: str) -> str:
        return sidecar_name(peer)

    def _peer_of(self, scname) -> Optional[str]:
        return self._names.get(str(scname))

    # ---- process + control channel
    def start(self, ns: str, events_path: Optional[str] = None) -> dict:
        """Spawn the sidecar, configure it, open the data socket, publish this node's identity under the run's
        signalling namespace, and start the background connector. -> the sidecar's `ready` event."""
        import secrets as _secrets
        import socket as _socket
        import subprocess as _sp
        with self._lock:
            if self.p is not None:
                return self.ready or {}
        os.makedirs(self.work_dir, exist_ok=True)
        self._evf = open(events_path or os.path.join(self.work_dir, "sidecar_events.jsonl"), "a", encoding="ascii")
        errf = open(os.path.join(self.work_dir, "sidecar.stderr.log"), "ab")
        env = dict((k, v) for k, v in os.environ.items()
                   if not SECRET_KEY_RE.search(k) and not k.startswith("NH_P2P") and not k.startswith("NH_TURN"))
        kw = {}
        if os.name == "nt":
            # a long-lived child of a possibly console-less parent: its OWN hidden console + process group
            # (10-dispatch Learned 2026-09-02 / 09-12); stdin/stdout stay pipes
            si = _sp.STARTUPINFO()
            si.dwFlags |= _sp.STARTF_USESHOWWINDOW
            si.wShowWindow = 0
            kw = {"creationflags": 0x00000010 | 0x00000200, "startupinfo": si}
        self.p = _sp.Popen([self.exe], stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=errf, env=env, **kw)
        threading.Thread(target=self._reader, name="p2p-sidecar-reader", daemon=True).start()
        self._local_token = _secrets.token_hex(24)
        prefix, sns = signal_namespace(ns)
        self.ns = sns
        if self.signal_key == "relay_token":
            self.signal_key = derive_signal_key(self.relay_token, sns)
        cfg = {"cmd": "config", "id": "config", "node": sidecar_name(self.node), "relay_url": self.relay_url,
               "relay_token": self.relay_token, "name_prefix": prefix,
               "miner_id": hashlib.sha256(("p2p/" + sns + "/" + self.node).encode()).hexdigest()[:16],
               "local_token": self._local_token, "signal_key": self.signal_key, "stun": self.stun,
               "turn": self.turn, "turn_user": self.turn_user, "turn_pass": self.turn_pass,
               "ice_policy": self.ice_policy, "loopback": False, "asn_lookup": False,
               "allow_private_relay": self.allow_private_relay, "sign_key": self.sign_seed_hex,
               "peer_pubkeys": dict((sidecar_name(k), v) for k, v in self.peer_pubkeys.items())}
        for k, v in self.extra.items():
            cfg.setdefault(k, v)
        if not self.turn:
            cfg.pop("turn_user"), cfg.pop("turn_pass")
        self.ready = self._call_raw(cfg, ("ready",), 60.0)
        host, port = str(self.ready["data_addr"]).rsplit(":", 1)
        s = _socket.create_connection((host, int(port)), timeout=10)
        s.sendall(("AUTH %s\n" % self._local_token).encode("ascii"))
        if self._readline(s) != b"OK":
            raise SidecarError("data socket auth failed")
        s.settimeout(None)
        self._data_sock = s
        threading.Thread(target=self._data_reader, name="p2p-sidecar-data", daemon=True).start()
        self._call("publish_identity", ("identity_published",), 120.0, ns=sns)
        threading.Thread(target=self._connector, name="p2p-sidecar-connect", daemon=True).start()
        return self.ready

    @staticmethod
    def _readline(s) -> bytes:
        b = bytearray()
        while True:
            c = s.recv(1)
            if not c:
                raise EOFError("data socket closed")
            if c == b"\n":
                return bytes(b)
            b += c
            if len(b) > 4096:
                raise ValueError("header too long")

    def _log_ev(self, ev: dict) -> None:
        try:
            with self._lock:
                if self._evf is not None and not self._evf.closed:
                    self._evf.write(json.dumps(dict(ev, t_rx=time.time()), sort_keys=True, ensure_ascii=True) + "\n")
                    self._evf.flush()
        except Exception:
            pass

    def _reader(self) -> None:
        try:
            for raw in self.p.stdout:
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(ev, dict):
                    continue
                ev["_rx_t"] = time.monotonic()
                self._log_ev(ev)
                self._on_event(ev)
        except Exception:
            pass
        finally:
            with self._lock:
                self.dead = True
                ws = list(self._waiters.values())
                hs = list(self._handles.values())
            for w in ws:
                try:
                    w({"ev": "sidecar_exit"})
                except Exception:
                    pass
            for h in hs:
                h.on_event({"ev": "sidecar_exit"})

    def _on_event(self, ev: dict) -> None:
        name = ev.get("ev")
        for f in list(self._listeners):
            try:
                f(ev)
            except Exception:
                pass
        if name == "peer_have":
            peer = self._peer_of(ev.get("peer"))
            sha = str(ev.get("sha", "")).lower()
            if peer and _HEX64.match(sha):
                with self._lock:
                    self._haves.setdefault(sha, {})[peer] = int(ev.get("size") or 0)
                    if len(self._haves) > 512:                        # bounded (the sidecar keeps 256)
                        self._haves.pop(next(iter(self._haves)))
        elif name == "peer_closed":
            peer = self._peer_of(ev.get("peer"))
            if peer:
                with self._lock:
                    c = self._connected.get(peer)
                    if c is not None and int(ev.get("epoch", -1)) == int(c.get("epoch", -2)):
                        self._connected.pop(peer, None)
                        for d in self._haves.values():
                            d.pop(peer, None)
                    self.counts["peer_closed"] += 1
        elif name == "pair_change":
            peer = self._peer_of(ev.get("peer"))
            if peer:
                with self._lock:
                    sts = list(self._stripes.get(peer) or [])
                    idx = int(ev.get("stripe", 0))
                    while len(sts) <= idx:
                        sts.append({})
                    sts[idx] = {"stripe": idx, "local": ev.get("local"), "remote": ev.get("remote")}
                self._note_stripes(peer, sts)
        fid = ev.get("id")
        if fid is not None:
            with self._lock:
                w = self._waiters.get(fid)
                h = self._handles.get(fid)
            if h is not None:
                h.on_event(ev)
                # a failed fetch delivers nothing: drop the handle; a fetched one is dropped once its frame arrived
                # (the sidecar writes the frame BEFORE its `fetched` reply, xfer.go cmdFetch)
                if name in ("fetch_failed", "error") or (name == "fetched" and h.data is not None):
                    with self._lock:
                        self._handles.pop(fid, None)
            if w is not None:
                w(ev)

    def _data_reader(self) -> None:
        s = self._data_sock
        try:
            while True:
                hdr = json.loads(self._readline(s))
                left, buf = int(hdr["size"]), bytearray()
                while left:
                    chunk = s.recv(min(1 << 20, left))
                    if not chunk:
                        raise EOFError("short frame")
                    buf += chunk
                    left -= len(chunk)
                with self._lock:
                    h = self._handles.get(str(hdr.get("id")))
                if h is not None:
                    h.on_data(bytes(buf))
                    if h.reply is not None:
                        with self._lock:
                            self._handles.pop(h.id, None)
        except Exception:
            pass

    def _note_stripes(self, peer: str, stripes) -> None:
        with self._lock:
            self._stripes[peer] = list(stripes or [])
            for st in stripes or []:
                loc = st.get("local") if isinstance(st, dict) else None
                if isinstance(loc, dict):
                    if loc.get("type") in ("srflx", "prflx") and loc.get("addr"):
                        self._own_ips.add(loc["addr"])
                    if loc.get("type") == "relay" and loc.get("rel_addr"):
                        self._own_ips.add(str(loc["rel_addr"]).split(":")[0])

    def _new_id(self, prefix: str) -> str:
        with self._lock:
            self._seq += 1
            return "%s-%d" % (prefix, self._seq)

    def _send(self, obj: dict) -> None:
        line = (json.dumps(obj) + "\n").encode("utf-8")
        with self._wlock:
            if self.p is None or self.dead:
                raise SidecarError("sidecar not running")
            try:
                self.p.stdin.write(line)
                self.p.stdin.flush()
            except (OSError, ValueError) as e:
                raise SidecarError("sidecar stdin: %s" % type(e).__name__)

    def _send_async(self, cmd: str, **kw) -> str:
        cid = self._new_id(cmd)
        try:
            self._send(dict(kw, cmd=cmd, id=cid))
        except SidecarError:
            pass
        return cid

    def _call_raw(self, obj: dict, final, timeout: float) -> dict:
        import queue as _q
        q = _q.Queue()
        with self._lock:
            if self.dead:
                raise SidecarError("sidecar exited before %s" % obj.get("cmd"))
            self._waiters[obj["id"]] = q.put
        try:
            self._send(obj)
            deadline = time.monotonic() + timeout
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise SidecarError("sidecar %s timed out after %.0f s" % (obj.get("cmd"), timeout))
                try:
                    ev = q.get(timeout=left)
                except _q.Empty:
                    continue
                if ev.get("ev") == "sidecar_exit":
                    raise SidecarError("sidecar exited during %s" % obj.get("cmd"))
                if ev.get("ev") == "error":
                    raise SidecarError("sidecar %s: %s" % (obj.get("cmd"), str(ev.get("error"))[:200]))
                if ev.get("ev") in final:
                    return ev
        finally:
            with self._lock:
                self._waiters.pop(obj["id"], None)

    def _call(self, cmd: str, final, timeout: float, **kw) -> dict:
        return self._call_raw(dict(kw, cmd=cmd, id=self._new_id(cmd)), final, timeout)

    # ---- connections
    def _connector(self) -> None:
        while not self.closed and not self.dead:
            now = time.time()
            epoch = int(now // self.slot_s)
            with self._lock:
                todo = [p for p in self.peers if p not in self._connected and p not in self._connecting]
                for p in todo:
                    self._connecting.add(p)
            for p in todo:
                threading.Thread(target=self._connect_one, args=(p, epoch), name="p2p-connect-" + p,
                                 daemon=True).start()
            nxt = (epoch + 1) * self.slot_s
            while not self.closed and not self.dead and time.time() < nxt:
                time.sleep(0.5)

    def _connect_one(self, peer: str, epoch: int) -> None:
        try:
            wait = max(5.0, (epoch + 1) * self.slot_s - time.time() - self.connect_timeout_s - 5.0)
            ev = self._call("connect", ("connected", "connect_failed"), wait + self.connect_timeout_s + 90.0,
                            peer=sidecar_name(peer), ns=self.ns, epoch=epoch, wait_s=wait,
                            connect_timeout_s=self.connect_timeout_s)
            if ev.get("ev") == "connected":
                with self._lock:
                    self._connected[peer] = {"epoch": int(ev.get("epoch", epoch)), "t": time.time()}
                    self.counts["connect_ok"] += 1
                self._note_stripes(peer, ev.get("stripes"))
            else:
                with self._lock:
                    self.counts["connect_failed"] += 1
        except Exception:
            with self._lock:
                self.counts["connect_failed"] += 1
        finally:
            with self._lock:
                self._connecting.discard(peer)

    def wait_connected(self, peers: Optional[Iterable[str]] = None, timeout: float = 120.0) -> bool:
        want = set(peers if peers is not None else self.peers)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if want <= set(self.connected_peers()):
                return True
            time.sleep(0.2)
        return False

    # ---- SidecarClient
    def connect(self, peer: str) -> bool:
        return peer in self.connected_peers()

    def connected_peers(self) -> FrozenSet[str]:
        with self._lock:
            return frozenset(self._connected) if not self.dead else frozenset()

    def peers_with(self, sha: str) -> FrozenSet[str]:
        with self._lock:
            return frozenset((self._haves.get(str(sha).lower()) or {}).keys())

    def peer_pair(self, peer: str) -> Optional[FetchProgress]:
        with self._lock:
            sts, own = list(self._stripes.get(peer) or []), set(self._own_ips)
        return _addr_summary(sts, own)

    def announce(self, sha: str, data: bytes) -> None:
        """Write the part beside the run (work_dir/parts/<sha>.part) and register it: the sidecar hashes it into
        chunk hashes and sends HAVE to every connected peer (and to peers that connect later). Keeps the newest
        keep_parts parts; older ones are unregistered and their files deleted (only files this client wrote)."""
        if self.p is None or self.dead or self.closed:
            raise SidecarError("sidecar not running")
        pdir = os.path.join(self.work_dir, "parts")
        os.makedirs(pdir, exist_ok=True)
        path = os.path.join(pdir, "%s.part" % sha)
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        self._call("register", ("registered",), 120.0, sha=sha, size=len(data), path=path)
        with self._lock:
            self.counts["registered"] += 1
            if sha not in self._registered:
                self._registered.append(sha)
            old = []
            while len(self._registered) > max(1, self.keep_parts):
                old.append(self._registered.popleft())
        for o in old:
            try:
                self._call("unregister", ("unregistered",), 30.0, sha=o)
            except Exception:
                pass
            try:
                os.remove(os.path.join(pdir, "%s.part" % o))
            except OSError:
                pass

    def fetch(self, peer: str, sha: str) -> FetchHandle:
        if self.p is None or self.dead or self.closed:
            raise SidecarError("sidecar not running")
        with self._lock:
            size = (self._haves.get(sha) or {}).get(peer)
        fid = self._new_id("fetch")
        h = _PionHandle(self, fid, peer, sha, size or None)
        with self._lock:
            self._handles[fid] = h
            self.counts["fetches"] += 1
        kw = {"sha": sha, "peer": sidecar_name(peer), "deliver": "socket"}
        if size:
            kw["expect_size"] = int(size)
        self._send(dict(kw, cmd="fetch", id=fid))
        return h

    def kill_process(self) -> None:
        """TEST HOOK (design S2): hard-kill this sidecar process (our own child, by its handle)."""
        if self.p is not None:
            try:
                self.p.kill()
            except Exception:
                pass

    def stats(self) -> dict:
        with self._lock:
            return {"node": self.node, "connected": sorted(self._connected), "counts": dict(self.counts),
                    "dead": self.dead, "pairs": dict((p, (lambda s: s and {"local": s.pair_local_type,
                                                                           "remote": s.pair_remote_type,
                                                                           "class": s.addr_class})(
                                                          _addr_summary(self._stripes.get(p), self._own_ips)))
                                                     for p in self._stripes)}

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            regs = list(self._registered)
            self._registered.clear()
        p = self.p
        if p is not None:
            try:
                self._send({"cmd": "shutdown"})
            except Exception:
                pass
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.wait(timeout=15)
            except Exception:
                try:
                    p.kill()
                    p.wait(timeout=10)
                except Exception:
                    pass
        try:
            if self._data_sock is not None:
                self._data_sock.close()
        except Exception:
            pass
        pdir = os.path.join(self.work_dir, "parts")
        for o in regs:
            try:
                os.remove(os.path.join(pdir, "%s.part" % o))
            except OSError:
                pass
        with self._lock:
            try:
                if self._evf is not None:
                    self._evf.close()
            except Exception:
                pass


# ---------------------------------------------------------------- fake sidecar (tests)
@dataclass
class Behavior:
    data: bytes
    first_chunk_s: float = 0.1          # time to first chunk
    rate_bps: float = 50e6              # bytes/s after the first chunk
    crash_at_s: Optional[float] = None  # poll() raises SidecarError from then on
    corrupt: bool = False               # delivers wrong bytes (bad body)
    raise_on_start: bool = False
    addr_class: str = "public"
    local_type: str = "host"
    remote_type: str = "srflx"


CHUNK = 256 * 1024


class _FakeHandle(FetchHandle):
    def __init__(self, sc: "FakeSidecar", b: Behavior):
        self.sc, self.b, self.t0 = sc, b, sc.clock.now()
        self.cancelled = False

    def poll(self) -> FetchProgress:
        b, t = self.b, self.sc.clock.now() - self.t0
        if b.crash_at_s is not None and t >= b.crash_at_s:
            raise SidecarError("sidecar process exited")
        size = len(b.data)
        pr = FetchProgress(size=size, pair_local_type=b.local_type, pair_remote_type=b.remote_type,
                           addr_class=b.addr_class)
        if t < b.first_chunk_s:
            return pr
        done_b = min(size, CHUNK + int((t - b.first_chunk_s) * b.rate_bps))
        pr.bytes_done, pr.t_first_byte = done_b, b.first_chunk_s
        if done_b >= size:
            pr.done = True
            pr.data = (bytes([b.data[0] ^ 1]) + b.data[1:]) if b.corrupt else b.data
        return pr

    def cancel(self) -> None:
        self.cancelled = True
        self.sc.cancels += 1


class FakeSidecar(SidecarClient):
    def __init__(self, clock: Clock):
        self.clock = clock
        self.peers = set()
        self.haves: Dict[str, Dict[str, float]] = {}       # sha -> {peer: time HAVE appears}
        self.behaviors: Dict[Tuple[str, str], Behavior] = {}
        self.announced: Dict[str, bytes] = {}
        self.fetch_calls = 0
        self.cancels = 0
        self.closed = False
        self._lock = threading.Lock()

    def add_peer(self, peer: str) -> None:
        self.peers.add(peer)

    def give(self, peer: str, data: bytes, at: Optional[float] = None, **beh) -> str:
        sha = hashlib.sha256(data).hexdigest()
        self.haves.setdefault(sha, {})[peer] = self.clock.now() if at is None else at
        self.behaviors[(peer, sha)] = Behavior(data=data, **beh)
        return sha

    def connect(self, peer):
        self.peers.add(peer)
        return True

    def connected_peers(self):
        return frozenset(self.peers)

    def peers_with(self, sha):
        now = self.clock.now()
        return frozenset(p for p, t in self.haves.get(sha, {}).items() if t <= now)

    def announce(self, sha, data):
        self.announced[sha] = data

    def fetch(self, peer, sha):
        with self._lock:
            self.fetch_calls += 1
        b = self.behaviors[(peer, sha)]
        if b.raise_on_start:
            raise SidecarError("sidecar not running")
        return _FakeHandle(self, b)

    def close(self):
        self.closed = True
