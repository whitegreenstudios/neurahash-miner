"""
TCP transport for distributed training: workers on DIFFERENT machines.

`distributed.py` proved the model with multiprocessing (same host). This moves the
same DiLoCo round over real TCP sockets, so a worker can be any machine that can
reach the coordinator. The coordinator broadcasts (global_params, shard, H, lr) each
round and collects deltas with a socket TIMEOUT, tolerating workers that stall or
disconnect (dropped nodes), and discarding stale results via a round-id check.

SECURITY: frames use the safe codec (`safe_codec.py`), NOT pickle, so a malicious
peer cannot execute code by sending a crafted frame. Pass a pre-shared key (`psk`)
to authenticate + integrity-check every frame via HMAC. A frame-size cap blunts
allocation DoS. For CONFIDENTIALITY (HMAC authenticates but does not encrypt — once
the pool spans hosts the weights/deltas cross the wire in cleartext), pass a TLS
context to the coordinator and a cert `pin` to the worker (opt-in, default off; see
`tls.py` and issue #40). See SECURITY.md.
"""

import os
import ssl
import time
import socket
import struct
import selectors

from . import safe_codec
from . import tls as _tls

# ----------------------------- transport pre-shared key (A3 hardening) -----------------------------
DEFAULT_PSK = b"neurahash-demo-psk"      # the PUBLIC, repo-committed demo key — loopback/dev ONLY


def resolve_psk():
    """The transport HMAC pre-shared key. A real deployment sets NEURAHASH_PSK to a SECRET; the
    built-in demo key is committed in the public repo, so it authenticates NOTHING against anyone who
    can read the source — it exists only so local/loopback dev needs no config. Returns
    (psk_bytes, is_default)."""
    env = os.environ.get("NEURAHASH_PSK")
    if env:
        return env.encode("utf-8"), False
    return DEFAULT_PSK, True


def warn_if_insecure_psk(is_default, host):
    """Loudly warn (and return True) when a coordinator binds a NON-loopback `host` with the DEFAULT
    (public) PSK — anyone who read the repo could then connect. A real off-loopback deployment must
    set NEURAHASH_PSK. No-op for loopback binds (dev) or a custom PSK."""
    nonloop = str(host) not in ("127.0.0.1", "localhost", "::1", "")
    if is_default and nonloop:
        print(f"[net] SECURITY WARNING: bound {host} with the DEFAULT demo PSK (public in the repo). "
              f"Set NEURAHASH_PSK to a secret before exposing the coordinator off-loopback.", flush=True)
        return True
    return False

# frame-size cap to limit allocation DoS. A DENSE base (Rung-2) pushes the whole model as the per-round
# "trunk" (~2.4GB for 0.6B fp32), so single-machine dense runs raise it via NEURAHASH_MAX_MSG_MB. Keep the
# 512MB default for networked MoE pools (small shared trunk) — a big cap off-loopback is a real DoS surface.
MAX_MSG_BYTES = int(os.environ.get("NEURAHASH_MAX_MSG_MB", "512")) * 1024 * 1024

# ----------------------------- bounded coordinator-side IO (#92) -----------------------------
# LIVE INCIDENT (#92): the single-threaded coordinator sent tasks/payloads to workers with NO send
# timeout. One peer that stopped draining its socket (an abandoned session whose process stayed alive —
# observed as an ESTABLISHED zombie — or a dead NAT/tunnel flow) filled the TCP window and parked the
# round loop inside `send` FOREVER (0.2 CPU-min over 9 min, log frozen, every new handshake timing out).
# The result-COLLECTION path was already deadline-bounded (#11); these knobs bound the SEND side and the
# admission budget so a single pathological peer can no longer wedge the whole pool. All default-ON: they
# only bound genuinely stalled/slow peers — a healthy worker drains its socket in milliseconds and is
# never affected — so this is a pure robustness fix, not a reward/consensus change (no audit-first gate).
# Every timeout is env-tunable. WORKER-side sends are intentionally left unbounded (a worker that stalls
# its own send only hurts itself, and it drives no shared loop).
SEND_TIMEOUT = float(os.environ.get("NEURAHASH_SEND_TIMEOUT", "60.0"))          # per coordinator->worker send

# ----------------------------- progress-based chunked send (P2, default-OFF) -----------------------------
# SECONDARY RESILIENCE (fleet-stability doc §5(a)): SEND_TIMEOUT above is a TOTAL wall-clock deadline
# (sendall's timeout has been total since py3.5). An HONEST slow-but-DRAINING link (a 25 Mbps Colab relay =
# the 0x3070 straggler) that is still making progress on a big int8 trunk+expert payload gets killed at the
# total deadline EXACTLY like a dead zombie — the measured `drop 0x3070: task send failed (TimeoutError)`.
# When NEURAHASH_SEND_PROGRESS_TIMEOUT > 0, send_msg_bounded switches to a loop over ~1 MiB memoryview
# slices and bounds the send by PER-SLICE PROGRESS instead of total time: a slice is aborted only if it
# makes ZERO bytes of progress within the window, so a moving link is never killed while a genuinely stalled
# one still dies within one window (the #92 anti-zombie property is preserved). NEURAHASH_SEND_MAX_S is a
# generous overall ceiling so even a progressing send cannot run unbounded. DEFAULT 0 = OFF = today's exact
# single-`sendall`-under-total-deadline path, byte-for-byte. Framing (8-byte length prefix + body) is
# UNCHANGED, so the receiver (`_recv_exact`, cumulative-deadline exact reads) is completely unaffected —
# RECEIVER-INVISIBLE.
SEND_PROGRESS_TIMEOUT = float(os.environ.get("NEURAHASH_SEND_PROGRESS_TIMEOUT", "0"))  # 0 = OFF (single sendall)
SEND_MAX_S = float(os.environ.get("NEURAHASH_SEND_MAX_S", "600"))                       # overall ceiling when ON
SEND_SLICE_BYTES = 1024 * 1024                                                          # ~1 MiB per progress slice

TLS_ACCEPT_TIMEOUT = float(os.environ.get("NEURAHASH_TLS_ACCEPT_TIMEOUT", "8.0"))  # server-side TLS handshake
# Whole-admission wall-clock budget per joiner (hello + auth + payload). Bounds `poll_new_workers` even
# against a peer that drains SLOWLY (not fully stalled) — such a peer can't hold the round thread past
# this. A big-rung, slow-WAN joiner pulling a large model payload may legitimately need this raised. The
# real follow-up is an ASYNC admission thread so admission never touches the round loop at all — out of
# scope here; tracked in #92.
ADMIT_TIMEOUT = float(os.environ.get("NEURAHASH_ADMIT_TIMEOUT", "90.0"))


# ----------------------------- framing -----------------------------
def send_msg(sock, obj, key=None):
    data = safe_codec.encode_msg(obj, key=key)
    sock.sendall(struct.pack("!Q", len(data)) + data)


def _send_all_progress(sock, buf, progress_timeout, max_s):
    """(P2 §5(a)) Send the whole framed buffer `buf` (an 8-byte length prefix + body, already built by
    the caller — framing is UNCHANGED) over ~1 MiB `memoryview` slices, bounding by PER-SLICE PROGRESS
    rather than a total wall-clock deadline. For each slice, one `send()` is attempted under a socket
    timeout of `progress_timeout`; if it returns 0 bytes or times out (zero progress within the window)
    the send is aborted with socket.timeout — the #92 anti-zombie kill, unchanged. As long as ANY bytes
    move the timer is effectively reset (the next slice/send gets a fresh window), so an honest slow-but-
    draining link is never killed for being slow. `max_s` is a generous overall ceiling (raises
    socket.timeout when exceeded) so even a progressing send cannot run unbounded. Caller sets/restores
    the socket's timeout around this (send_msg_bounded), same contract as the single-sendall path."""
    mv = memoryview(buf)
    total = len(mv)
    sent = 0
    sock.settimeout(progress_timeout)
    overall_deadline = time.time() + max_s
    while sent < total:
        if time.time() >= overall_deadline:
            raise socket.timeout(f"send exceeded NEURAHASH_SEND_MAX_S ({max_s}s) at {sent}/{total} bytes")
        end = min(sent + SEND_SLICE_BYTES, total)
        # sock.send() writes SOME of the slice (kernel send-buffer worth) and returns the count; a stalled
        # peer whose window is full blocks until `progress_timeout` then raises socket.timeout (an OSError
        # subclass) — the genuinely-dead-pipe kill. A returned 0 is the same signal (no progress).
        n = sock.send(mv[sent:end])
        if n == 0:
            raise socket.timeout("send made zero progress")
        sent += n


def send_msg_bounded(sock, obj, key=None, timeout=None):
    """Coordinator-side send with a WALL-CLOCK timeout so a peer that stops draining its socket cannot
    park the single-threaded round loop inside `send` forever (#92). Sets the socket's send timeout,
    sends, and RESTORES the socket's prior timeout (admission + the selector recv path both rely on the
    timeout state they set themselves, so we must not leak ours). Raises socket.timeout when the send
    can't complete within `timeout` (default NEURAHASH_SEND_TIMEOUT); callers drop the connection on
    socket.timeout/OSError exactly like the existing dead-peer path — the worker simply reconnects.

    (P2 §5(a)) When NEURAHASH_SEND_PROGRESS_TIMEOUT > 0 the bound becomes PER-SLICE PROGRESS instead of
    the total `timeout`: a big honest slow-drain payload is bounded by whether each ~1 MiB slice keeps
    moving, not by a total deadline that would kill a still-draining 25 Mbps relay. DEFAULT 0 = OFF =
    the EXACT single-`sendall`-under-total-`timeout` path below, byte-for-byte. Framing is unchanged
    either way, so the receiver is unaffected."""
    if timeout is None:
        timeout = SEND_TIMEOUT
    prev = sock.gettimeout()
    try:
        if SEND_PROGRESS_TIMEOUT > 0:
            # PROGRESS-BOUNDED path: same frame bytes, sent in slices, bounded per-slice.
            data = safe_codec.encode_msg(obj, key=key)
            _send_all_progress(sock, struct.pack("!Q", len(data)) + data,
                               progress_timeout=SEND_PROGRESS_TIMEOUT, max_s=SEND_MAX_S)
        else:
            # OFF: today's exact single sendall under a total wall-clock deadline.
            sock.settimeout(timeout)
            send_msg(sock, obj, key=key)
    finally:
        try:
            sock.settimeout(prev)
        except OSError:
            pass


def _recv_exact(sock, n, deadline=None):
    """Read exactly n bytes. When `deadline` (an absolute time.time()) is given, enforce a CUMULATIVE
    wall-clock budget across the whole read (security #11): a per-recv socket timeout is reset by every
    chunk, so a peer trickling 1 byte at a time can otherwise stall forever; deadline makes the total
    read bounded regardless of trickle, raising TimeoutError when the budget is spent."""
    buf = bytearray()
    while len(buf) < n:
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("recv deadline exceeded")
            sock.settimeout(remaining)
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock, key=None, deadline=None):
    (length,) = struct.unpack("!Q", _recv_exact(sock, 8, deadline))
    if length > MAX_MSG_BYTES:
        raise ConnectionError(f"frame too large: {length} > {MAX_MSG_BYTES}")
    return safe_codec.decode_msg(_recv_exact(sock, length, deadline), key=key)


# ----------------------------- worker side -----------------------------
def run_worker_client(host, port, address, honest=True, delay=0.0, psk=None, retries=50,
                      tls_pin=None, tls_server_hostname=_tls.DEFAULT_SNI):
    """Runs on a worker machine. Connects, then services training tasks until done. When `tls_pin` is
    set (the coordinator's published cert fingerprint), the connection is TLS-wrapped and the pin is
    enforced before the hello — a coordinator whose cert doesn't match the pin is rejected (CertPinError
    propagates, #40), so the worker never trains against an unverified / MITM coordinator."""
    from .training_layer import Worker

    sock = None
    for _ in range(retries):
        try:
            sock = socket.create_connection((host, port), timeout=10)
            break
        except OSError:
            time.sleep(0.1)
    if sock is None:
        raise ConnectionError(f"could not connect to {host}:{port}")

    sock = _tls.maybe_wrap_client(sock, tls_pin, server_hostname=tls_server_hostname)   # (#40) opt-in TLS+pin
    send_msg(sock, {"type": "hello", "address": address}, key=psk)
    model = recv_msg(sock, key=psk)["model"]
    try:
        while True:
            msg = recv_msg(sock, key=psk)
            if msg["type"] == "done":
                break
            if msg["type"] == "task":
                if delay:
                    time.sleep(delay)
                sub = Worker(address, msg["shard"], honest=honest).train(
                    model, msg["global_params"], H=msg["H"], lr=msg["lr"])
                send_msg(sock, {"type": "result", "sub": sub, "round": msg["round"]}, key=psk)
    except (ConnectionError, OSError, ValueError):
        pass
    finally:
        sock.close()


def enable_tcp_keepalive(conn):
    """Keep a TCP connection warm so an idle NAT/router/tunnel hop doesn't silently reap it.

    Used on BOTH ends: the coordinator arms it on every accepted conn, and the WORKER arms it on its
    outbound socket (a cross-subnet worker whose flow goes through a home router with an aggressive
    conntrack idle-timeout was getting reaped between rounds, then eagerly reconnecting -> dup-reconnect
    churn on the coordinator). SO_KEEPALIVE alone is NOT enough: the OS default idle before the first
    probe is ~2 HOURS. We tighten idle/interval so a probe rides the wire every ~25 s (inside any
    typical middlebox window) even when neither side sends application data. Each knob is best-effort:
    an unsupported platform/option is skipped, never fatal. Env-tunable (NEURAHASH_KEEPALIVE_IDLE /
    _INTVL) for an even more aggressive middlebox."""
    try:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        return                                              # can't even arm keepalive -> nothing more to tune
    idle = max(1, int(float(os.environ.get("NEURAHASH_KEEPALIVE_IDLE", "25"))))    # seconds before 1st probe
    intvl = max(1, int(float(os.environ.get("NEURAHASH_KEEPALIVE_INTVL", "5"))))   # seconds between probes
    # Windows: SIO_KEEPALIVE_VALS takes MILLISECONDS (onoff, idle_ms, interval_ms); no per-socket probe-count.
    if hasattr(socket, "SIO_KEEPALIVE_VALS"):
        try:
            conn.ioctl(socket.SIO_KEEPALIVE_VALS, (1, idle * 1000, intvl * 1000))
        except OSError:
            pass
        return
    # POSIX: TCP_KEEPIDLE / TCP_KEEPINTVL are in SECONDS; TCP_KEEPCNT bounds probes before giving up.
    for opt_name, val in (("TCP_KEEPIDLE", idle), ("TCP_KEEPINTVL", intvl), ("TCP_KEEPCNT", 4)):
        opt = getattr(socket, opt_name, None)
        if opt is None:                                     # e.g. macOS has TCP_KEEPALIVE, not TCP_KEEPIDLE
            continue
        try:
            conn.setsockopt(socket.IPPROTO_TCP, opt, val)
        except OSError:
            pass


# ----------------------------- coordinator side -----------------------------
class TCPCoordinator:
    _LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}

    def __init__(self, host="127.0.0.1", port=0, psk=None, tls_context=None):
        # auth is mandatory for any non-loopback bind: without a pre-shared key,
        # unauthenticated peers could be admitted and spoof worker identities.
        if psk is None and host not in self._LOOPBACK:
            raise ValueError("psk (HMAC key) is required when binding a non-loopback host")
        self.psk = psk
        # opt-in TLS (#40): when set, every accepted connection is TLS-wrapped server-side before its
        # hello is read, so the channel is encrypted (HMAC underneath stays as defense-in-depth). The
        # wrapped SSLSocket is a drop-in for the raw socket — send_msg/recv_msg are unchanged.
        self.tls_context = tls_context
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen()
        self.host, self.port = self.srv.getsockname()
        self.conns = {}                            # address -> socket
        self.worker_meta = {}                      # address -> hello metadata (e.g. reported usable VRAM, #23)

    @staticmethod
    def _enable_keepalive(conn):
        """Arm tightened TCP keepalive on an accepted conn (delegates to the shared module-level helper,
        which the worker's outbound socket also uses — see enable_tcp_keepalive)."""
        enable_tcp_keepalive(conn)

    def _tls_accept(self, conn):
        """Server-side TLS handshake on a freshly accepted conn (#40). Returns the wrapped socket, the
        raw conn unchanged when TLS is off, or None if the handshake failed (e.g. a plaintext peer when
        TLS is required, or a peer that never sent a ClientHello) — never raises into the accept loop.

        The handshake is BOUNDED by NEURAHASH_TLS_ACCEPT_TIMEOUT (#92): a peer that TCP-connects but
        sends no ClientHello (or a slow one) is dropped within that budget instead of parking the single-
        threaded accept/round loop on the blocking `wrap_socket`. `wrap_server_socket` sets the timeout
        before the wrap and RESTORES the socket's prior timeout on success, so a wrapped conn carries the
        SAME timeout it had on accept (None) — `_admit` then sets `hello_timeout` as before. socket.timeout
        is an OSError subclass, so a handshake stall lands in the same drop path as any other TLS error."""
        if self.tls_context is None:
            return conn
        try:
            return _tls.wrap_server_socket(conn, self.tls_context,
                                           handshake_timeout=TLS_ACCEPT_TIMEOUT)
        except (ssl.SSLError, OSError) as _e:
            # (session-lifecycle) log it — a stale backlog conn (its client already gave up and
            # reconnected) dies here on every accept sweep; silently eating them hid how much of the
            # admission budget went to ghosts vs. live joiners.
            print(f"[net] TLS accept failed ({type(_e).__name__}) — stale/plaintext peer dropped",
                  flush=True)
            try:
                conn.close()
            except OSError:
                pass
            return None

    def _admit(self, conn, model=None, hello_timeout=10, hello_payload=None, hello_payload_fn=None,
               auth_fn=None, pushing=None):
        """Handshake one freshly-accepted connection: read its hello, optionally prove its identity
        (`auth_fn`), register it, and send the post-hello payload (the numpy model by default; an
        arbitrary `hello_payload`; or, for per-worker assignment like sharded expert hosting,
        `hello_payload_fn(addr, join_index)` evaluated with the 0-based join order). Returns the worker
        address on success, or None if the peer was rejected (bad hello, duplicate, failed auth, or it
        died during the send) — never raises.

        (#92) The WHOLE admission (hello + auth + payload send) is bounded by NEURAHASH_ADMIT_TIMEOUT so
        one joiner cannot hold `poll_new_workers` (and thus the single-threaded round loop) past that
        budget — not even a peer that drains the model payload SLOWLY rather than stalling outright. The
        model-payload send is additionally send-bounded (NEURAHASH_SEND_TIMEOUT) so a joiner that stops
        reading mid-payload is dropped instead of parking the loop in `send` forever (the incident's
        admission analogue). The socket that enters `self.conns` carries the SAME timeout it did before
        this change (the send bound is restored after the send), so the downstream selector recv path is
        unaffected."""
        self._enable_keepalive(conn)
        admit_deadline = time.time() + ADMIT_TIMEOUT
        # cap the hello recv by the admission budget too (never exceed hello_timeout — same as before when
        # the budget is generous, which it is by default at 90s vs a 10s hello).
        conn.settimeout(max(0.05, min(hello_timeout, admit_deadline - time.time())))
        try:
            hello = recv_msg(conn, key=self.psk)
            addr = hello["address"]
        except (ConnectionError, OSError, KeyError, ValueError):
            print("[net] admission drop: malformed/absent hello", flush=True)
            conn.close()
            return None
        # (session-lifecycle fix 5) DUPLICATE ADDRESS = RECONNECT, not intruder. The old behavior always
        # closed the NEW conn and kept the stale OLD one — but a mid-round reap can leave a DEAD zombie in
        # self.conns while the SAME wallet reconnects on a fresh socket; rejecting the reconnect stranded
        # the honest miner behind its own ghost. When admission auth is ON (auth_fn) we let the newcomer
        # PROVE control of `addr`'s key via the #44 challenge (below): only the real key-holder passes, so
        # this is NOT a hijack vector — an impostor cannot sign the victim's challenge and is rejected,
        # leaving the incumbent untouched. On success we close the OLD socket and register the NEW conn.
        # When auth is OFF we keep today's conservative reject (no way to tell reconnect from impostor).
        replacing_dup = addr in self.conns
        if replacing_dup and auth_fn is None:
            print(f"[net] admission drop {addr}: duplicate address, auth off (kept existing conn)",
                  flush=True)
            conn.close()
            return None
        # (#44) PER-NODE-SIGNED ADMISSION: when an auth_fn is supplied, prove control of `addr`'s key
        # over a fresh challenge BEFORE the connection is registered — so a mere PSK-holder cannot occupy
        # a slot as any address, and a duplicate-address hello cannot hijack a registered victim's slot
        # (the attacker can't sign the challenge with the victim's key). auth_fn does the challenge-
        # response over `conn` (inside TLS when #40 is on) and returns a bound identity, or None to
        # reject. It runs before registration, so a rejected peer never gets a `conns[addr]` entry.
        if auth_fn is not None:
            try:
                if auth_fn(addr, conn) is None:
                    print(f"[net] admission reject {addr}: auth challenge failed", flush=True)
                    conn.close()
                    return None
            except Exception as _e:
                print(f"[net] admission reject {addr}: auth error {type(_e).__name__}", flush=True)
                conn.close()
                return None
        if replacing_dup:
            # auth passed for a duplicate address -> this is the SAME identity reconnecting. Evict the stale
            # socket (best-effort close) and fall through to register the fresh conn in its place.
            old = self.conns.pop(addr, None)
            self.worker_meta.pop(addr, None)
            if old is not None and old is not conn:
                if pushing and addr in pushing:
                    # (ASYNC RE-PARTITION) `old` is owned by an off-round-loop trunk+experts push daemon; a
                    # close here would race that daemon's in-flight send on the same socket (undefined
                    # concurrent close). Leave it — the coordinator's push reaper closes `old` once the daemon
                    # has exited, and it will NOT close this fresh conn (it checks socket identity).
                    pass
                else:
                    try:
                        old.close()
                    except OSError:
                        pass
            print(f"[net] admission: {addr} reconnected (auth ok) -> replaced stale conn", flush=True)
        join_index = len(self.conns)                       # 0-based order this worker joined
        self.conns[addr] = conn
        # keep the worker's hello metadata (reported usable VRAM etc.) for the readiness gate (#23);
        # purely advisory — never trusted for payment, only for the default-closed capacity estimate.
        self.worker_meta[addr] = {k: v for k, v in hello.items() if k != "type"}
        if hello_payload_fn is not None:
            payload = hello_payload_fn(addr, join_index)
            if payload is None:                            # (A3) admission gate REJECTED this peer
                self.conns.pop(addr, None)                 #      (e.g. unstaked Sybil) — undo the
                self.worker_meta.pop(addr, None)           #      provisional registration + drop it.
                print(f"[net] admission reject {addr}: payload gate declined (e.g. unstaked/capacity)",
                      flush=True)
                conn.close()
                return None
        elif hello_payload is not None:
            payload = hello_payload
        else:
            payload = {"type": "model", "model": model}
        # (#92) SEND the (possibly large, big-rung) model payload with a wall-clock bound = the smaller of
        # NEURAHASH_SEND_TIMEOUT and the remaining admission budget, so a joiner that stops draining mid-
        # payload is dropped (worker will reconnect) instead of wedging the round loop in `send`. socket.
        # timeout is an OSError subclass; send_msg_bounded restores the socket's prior timeout so the
        # admitted conn's timeout state is unchanged for the downstream selector recv path.
        try:
            send_msg_bounded(conn, payload, key=self.psk,
                             timeout=max(0.05, min(SEND_TIMEOUT, admit_deadline - time.time())))
        except OSError as _e:                              # includes socket.timeout (stalled/slow drain)
            self.conns.pop(addr, None)                     # undo the provisional registration + its
            self.worker_meta.pop(addr, None)               # advisory meta (keep both in sync, as the
            print(f"[net] admission drop {addr}: payload send failed ({type(_e).__name__}) — "
                  f"stalled/slow drain", flush=True)
            conn.close()                                   # hello_payload_fn-reject path above does)
            return None
        return addr

    def poll_new_workers(self, model=None, budget=0.05, hello_timeout=10, hello_payload=None,
                         hello_payload_fn=None, auth_fn=None, pushing=None):
        """Non-blocking dynamic join: admit any workers that are *currently* waiting to
        connect (up to `budget` seconds of accepting), then return. Call it at the top of
        every round so new miners (a 2nd Colab, the 5090/4060 later) join live without a
        restart. `hello_payload_fn(addr, join_index)` enables per-worker assignment (sharding);
        `auth_fn(addr, conn)` enables a per-node-signed admission handshake (#44). `pushing` is an
        optional set/dict of addresses whose OLD socket is currently owned by an off-round-loop push
        thread — on a reconnect for such an address `_admit` leaves the stale socket for the push
        reaper to close instead of closing it under the daemon. Returns the list of newly-admitted
        addresses."""
        new = []
        deadline = time.time() + budget
        while True:
            self.srv.settimeout(max(0.0, deadline - time.time()))
            try:
                conn, _ = self.srv.accept()
            except (socket.timeout, BlockingIOError, OSError):
                break
            conn = self._tls_accept(conn)            # (#40) TLS handshake (no-op when TLS off)
            if conn is None:                          # handshake failed -> drop, keep accepting
                continue
            addr = self._admit(conn, model, hello_timeout, hello_payload, hello_payload_fn,
                               auth_fn=auth_fn, pushing=pushing)
            if addr is not None:
                new.append(addr)
            if time.time() >= deadline:
                break
        # drop metadata for workers no longer connected so it stays bounded + in sync with conns
        self.worker_meta = {a: m for a, m in self.worker_meta.items() if a in self.conns}
        return new

    def worker_vram_gb(self, addr):
        """Usable VRAM (GB) the worker reported in its hello (#23 readiness telemetry), or 0.0 if it
        reported none (e.g. a CPU worker, or an older client). Advisory only — never paid on."""
        try:
            return float((self.worker_meta.get(addr) or {}).get("vram_gb", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def heartbeat(self, skip=None):
        """Send a no-op ping to every connection to keep it warm while no task is in flight.
        Workers ignore unknown message types, so this is safe with existing worker clients.
        Drops (and removes) any connection that errors on send. (#92) Each ping is send-bounded
        (NEURAHASH_SEND_TIMEOUT) so a peer that stopped draining its socket can't park the heartbeat
        loop — a heartbeat runs on the round thread too (e.g. while waiting for the first miner / during
        a guardian halt), so an unbounded ping to a stalled zombie would wedge the pool just like a task.

        (ASYNC RE-PARTITION) `skip` is an optional set/dict of addresses whose socket is currently owned by
        an OFF-round-loop push thread; pinging one here would be a SECOND writer racing that thread on one
        socket. Any addr in `skip` is left untouched. `skip=None`/empty (the default, and the only state
        when the async-repush lane is off) pings every conn exactly as before (byte-identical)."""
        for addr, sock in list(self.conns.items()):
            if skip and addr in skip:                      # socket owned by an off-loop push thread -> don't race it
                continue
            try:
                send_msg_bounded(sock, {"type": "ping"}, key=self.psk)
            except OSError as _he:                         # includes socket.timeout (stalled drain)
                # (session-lifecycle) log it — a heartbeat-reaped member vanished silently, leaving
                # no trace of WHY the roster shrank between rounds.
                print(f"[net] drop {addr}: heartbeat send failed ({type(_he).__name__})", flush=True)
                self.conns.pop(addr, None)
                try:
                    sock.close()
                except OSError:
                    pass

    def accept_workers(self, n, model=None, timeout=20, hello_timeout=10, hello_payload=None,
                       auth_fn=None):
        """Accept exactly n distinct workers; a peer that never sends a valid hello (or fails the
        per-node-signed admission handshake, `auth_fn`) is dropped (bounded by hello_timeout) rather
        than hanging us."""
        deadline = time.time() + timeout
        while len(self.conns) < n and time.time() < deadline:
            self.srv.settimeout(max(0.05, deadline - time.time()))
            try:
                conn, _ = self.srv.accept()
            except socket.timeout:
                break
            conn = self._tls_accept(conn)            # (#40) TLS handshake (no-op when TLS off)
            if conn is None:
                continue
            self._admit(conn, model, hello_timeout, hello_payload, auth_fn=auth_fn)
        if len(self.conns) < n:
            raise TimeoutError(f"only {len(self.conns)}/{n} workers connected")

    def run_round(self, round_id, global_params, assignments, H=30, lr=0.5, timeout=5.0):
        """Broadcast round `round_id`, collect deltas until timeout. Returns
        (results, dropped). Stale results (different round id) are discarded and the
        worker reported as dropped, NOT slashed."""
        # send the round's task to every live worker; a send that fails means the worker
        # vanished (tunnel drop / Colab closed) — remove it instead of crashing the round.
        live = {}
        for addr, sock in list(self.conns.items()):
            try:
                send_msg(sock, {"type": "task", "round": round_id, "global_params": global_params,
                                "shard": assignments[addr], "H": H, "lr": lr}, key=self.psk)
                live[addr] = sock
            except (OSError, KeyError):
                self.conns.pop(addr, None)
                try:
                    sock.close()
                except OSError:
                    pass

        sel = selectors.DefaultSelector()
        for addr, sock in live.items():
            sel.register(sock, selectors.EVENT_READ, addr)

        results, pending = {}, set(live)
        deadline = time.time() + timeout
        try:
            while pending and time.time() < deadline:
                for key, _ in sel.select(timeout=max(0.01, deadline - time.time())):
                    addr, sock = key.data, key.fileobj
                    sock.settimeout(max(0.05, deadline - time.time()))
                    try:
                        msg = recv_msg(sock, key=self.psk)
                        if msg.get("round") == round_id:
                            results[addr] = msg["sub"]
                    except (ConnectionError, OSError, ValueError):
                        # this worker died mid-round; drop its connection so it can't
                        # poison the next round (the stale-conn crash we hit before)
                        self.conns.pop(addr, None)
                    pending.discard(addr)
                    sel.unregister(sock)
        finally:
            sel.close()
        dropped = [a for a in live if a not in results]
        return results, dropped

    def shutdown(self):
        for sock in self.conns.values():
            try:
                send_msg(sock, {"type": "done"}, key=self.psk)
                sock.close()
            except OSError:
                pass
        self.srv.close()
