"""
neurahash/guardian_halt.py — emergency HALT / RESUME for the live pool, authorized by a k-of-m
GUARDIAN multisig (#41).

A value-bearing chain needs a credibly-neutral "stop" that is NOT a single operator pulling a plug. A
HALT is a signed order: at least k of m pre-registered guardian keys sign a canonical payload; once that
threshold is met the controller flips to halted and the live loop pauses **block production + admission
(stake acceptance)**. A RESUME (same k-of-m) lifts it early; a halt also **auto-expires** after a bounded
duration so a lost-key guardian set can't freeze the chain forever. Every order carries a monotonically
increasing **epoch** so a captured order can't be replayed (the epoch floor is rebuilt from the persisted
ledger on startup, so replay-safety survives a coordinator restart). Each halt/resume is appended to the
pool ledger as an immutable, tamper-evident **intent record** (`SignedPoolLedger.governance`).

SCOPE. The MECHANISM is code (here): threshold verification, the pause state machine, auto-expiry,
replay protection, audit records, and the live-loop wiring. WHO the guardians are and the key ceremony
is **external governance** — the guardian set is injected from config/env, never hardcoded. Default OFF
(no guardian set configured -> no halt capability, behavior unchanged).

ORDER INGESTION. A guardian assembles a signed order with `build_order(...)` and drops it at the path in
`NEURAHASH_HALT_ORDER` (default `halt_order.json`); the coordinator polls it each round. (Production would
gossip the order over the finality vote channel; the file is the simple, real MVP ingestion path — the
verification + state machine are identical either way.)
"""

import os
import json

from neura_l1.signing import sign_bytes, recover_bytes_scheme

DEFAULT_MAX_HALT_S = 3600.0          # auto-expire a halt after this (configurable). Guardians re-issue a
#                                      fresh-epoch halt to EXTEND; the cap means a lost guardian set can't
#                                      brick the chain permanently — it self-heals after the window.
HALT, RESUME = "halt", "resume"


# ----------------------------- signed order payload -----------------------------
def halt_payload(action, epoch, reason="", duration_s=0):
    """The exact bytes each guardian SIGNS to authorize a halt/resume. Binds the action, a monotonic
    epoch (replay protection), the human reason, and the halt duration — so a captured signature can't
    be lifted to a different action, epoch, or duration."""
    return json.dumps({"guard": "neurahash-halt", "action": str(action), "epoch": int(epoch),
                       "reason": str(reason), "duration_s": int(duration_s)},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


# ----------------------------- the k-of-m guardian set -----------------------------
class GuardianSet:
    """An immutable k-of-m guardian set: m member addresses + a threshold k."""

    def __init__(self, members, threshold):
        self.members = {str(a).strip().lower() for a in members if str(a).strip()}
        self.threshold = int(threshold)
        if not self.members:
            raise ValueError("guardian set is empty")
        if not (1 <= self.threshold <= len(self.members)):
            raise ValueError(f"threshold {threshold} out of range for {len(self.members)} guardians")

    def distinct_signers(self, payload, signatures):
        """The set of DISTINCT guardian addresses that validly signed `payload`. A non-guardian or
        unrecoverable signature is ignored; the same guardian signing twice counts ONCE (so you can't
        reach k by duplicating one key)."""
        signers = set()
        for sig in signatures or []:
            try:
                a = recover_bytes_scheme(payload, sig).lower()
            except Exception:
                continue
            if a in self.members:
                signers.add(a)
        return signers

    def meets_threshold(self, payload, signatures):
        return len(self.distinct_signers(payload, signatures)) >= self.threshold


# ----------------------------- the halt/resume state machine -----------------------------
class GuardianHalt:
    """Verified pause controller. `is_halted(now)` is the single bit the live loop checks; everything
    else is the guardian-authorized transition logic + audit trail."""

    def __init__(self, guardian_set, ledger=None, max_halt_s=DEFAULT_MAX_HALT_S):
        self.gs = guardian_set
        self.ledger = ledger
        self.max_halt_s = float(max_halt_s)
        self._halted_until = 0.0          # absolute time; halted while now < this
        self._last_epoch = 0              # monotonic floor: a new order MUST exceed this
        self._init_epoch_from_ledger()

    def _init_epoch_from_ledger(self):
        """Rebuild the epoch floor from the persisted ledger so a captured order can't be replayed across
        a coordinator restart (in-memory state resets to 0, but the signed log remembers)."""
        if self.ledger is None:
            return
        try:
            recs = self.ledger.governance_records()
        except Exception:
            return
        for r in recs:
            rec = r.get("record", r) if isinstance(r, dict) else {}
            if isinstance(rec, dict) and rec.get("guard_intent") in (HALT, RESUME):
                self._last_epoch = max(self._last_epoch, int(rec.get("epoch", 0)))

    # ---- query ----
    def is_halted(self, now):
        return now < self._halted_until

    def remaining(self, now):
        return max(0.0, self._halted_until - now)

    @property
    def last_epoch(self):
        return self._last_epoch

    # ---- transitions ----
    def submit(self, action, epoch, signatures, now, reason="", duration_s=0):
        """Apply a guardian-signed order. Returns (ok, message). Rejects a stale/replayed epoch, an
        action with fewer than k distinct guardian signatures, or an unknown action — WITHOUT changing
        state. On success advances the epoch floor and appends an immutable ledger intent record."""
        epoch = int(epoch)
        if epoch <= self._last_epoch:
            return False, f"stale epoch {epoch} <= floor {self._last_epoch} (replay)"
        if action not in (HALT, RESUME):
            return False, f"unknown action {action!r}"
        payload = halt_payload(action, epoch, reason, int(duration_s))
        if not self.gs.meets_threshold(payload, signatures):
            n = len(self.gs.distinct_signers(payload, signatures))
            return False, f"insufficient guardian signatures: {n}/{self.gs.threshold}"
        self._last_epoch = epoch
        if action == HALT:
            eff = min(float(duration_s), self.max_halt_s) if duration_s > 0 else self.max_halt_s
            self._halted_until = now + eff
            self._record(HALT, epoch, reason, now + eff)
            return True, f"HALTED for {eff:.0f}s (epoch {epoch})"
        self._halted_until = 0.0                                  # RESUME
        self._record(RESUME, epoch, reason, 0.0)
        return True, f"RESUMED (epoch {epoch})"

    def _record(self, action, epoch, reason, expiry):
        if self.ledger is None:
            return
        try:
            self.ledger.governance({"guard_intent": action, "epoch": int(epoch), "reason": str(reason),
                                    "expiry": float(expiry), "k": self.gs.threshold,
                                    "m": len(self.gs.members)})
        except Exception:
            pass                                                  # audit is best-effort; never crash the loop

    # ---- file ingestion (MVP transport) ----
    def poll_file(self, path, now):
        """Read a signed order from `path` and apply it if its epoch is newer. Returns (applied, message)
        or (False, None) when there is no new/valid order. Missing/garbage files are ignored quietly."""
        order = load_order(path)
        if not order:
            return False, None
        if int(order.get("epoch", 0)) <= self._last_epoch:
            return False, None                                    # already applied (or stale) -> ignore
        return self.submit(order.get("action"), order.get("epoch"), order.get("signatures"),
                           now, reason=order.get("reason", ""), duration_s=order.get("duration_s", 0))


# ----------------------------- order assembly + file helpers -----------------------------
def build_order(action, epoch, guardian_accounts, reason="", duration_s=0):
    """Assemble a signed k-of-m order: each account in `guardian_accounts` signs the canonical payload.
    Returns a JSON-able dict {action, epoch, reason, duration_s, signatures}. Guardians sign offline and
    combine signatures out-of-band; this is the helper they (and the tests) use."""
    payload = halt_payload(action, int(epoch), reason, int(duration_s))
    sigs = [sign_bytes(acct, payload) for acct in guardian_accounts]
    return {"action": str(action), "epoch": int(epoch), "reason": str(reason),
            "duration_s": int(duration_s), "signatures": sigs}


def save_order(order, path):
    with open(path, "w") as f:
        json.dump(order, f, indent=2)


def load_order(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ----------------------------- env config -----------------------------
def guardian_set_from_env():
    """Build the guardian set from NEURAHASH_GUARDIANS (comma-separated addresses) +
    NEURAHASH_GUARDIAN_THRESHOLD (default = majority). Returns None when unset (no halt capability —
    the default, so dev/testnet behavior is unchanged)."""
    raw = os.environ.get("NEURAHASH_GUARDIANS", "").strip()
    if not raw:
        return None
    members = [a.strip() for a in raw.split(",") if a.strip()]
    if not members:
        return None
    default_k = (len(members) // 2) + 1
    k = int(os.environ.get("NEURAHASH_GUARDIAN_THRESHOLD", str(default_k)))
    return GuardianSet(members, k)


def halt_order_path():
    return os.environ.get("NEURAHASH_HALT_ORDER", "halt_order.json")


def guardian_halt_from_env(ledger=None):
    """Build a GuardianHalt from env, or None if no guardian set is configured."""
    gs = guardian_set_from_env()
    if gs is None:
        return None
    return GuardianHalt(gs, ledger=ledger,
                        max_halt_s=float(os.environ.get("NEURAHASH_MAX_HALT_S", str(DEFAULT_MAX_HALT_S))))
