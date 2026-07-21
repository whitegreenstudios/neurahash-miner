"""
neura_l1.block_state — on-chain data structures + deterministic native-NRH state machine.

This is the FOUNDATION of the PoUW Layer-1: it owns

  * the block header / hash format (JSON-safe, full-precision floats),
  * the Block (header + txs + the PoUW proof payload),
  * the account State (balances / staked / treasury / minted + the canonical
    model-checkpoint hash),
  * the halving + 2.1B-cap reward schedule (borrowed from neurahash.chain.PoUWChain),
  * deterministic tx application, reward minting (split by contributor weight),
  * slashing, and advancement of the canonical model checkpoint.

There is NO consensus policy and NO proof verification here — only data structures and
a PURE state transition, so any node recomputes byte-identical results and can compare
state_hash() to detect divergence. Proof building/checking lives in neura_l1.pouw_gate;
validity/leader/fork-choice live in neura_l1.consensus.

Floats are hashed at full precision via _canon (same recipe as neurahash.chain._canon) so
two nodes never collide or disagree on a header hash. apply_block records only the
checkpoint HASH of the advanced model; the actual params P_h + delta are advanced by the
Node holding the live MoELM (see neura_l1.p2p_node).
"""

import json
import hashlib
import copy
import math


def _finite(x):
    """float(x) that REJECTS NaN/Infinity (raises ValueError). Used at every untrusted
    numeric boundary so a non-finite amount can't slip past `< 0`/balance comparisons (which
    are vacuously false for NaN) and poison the ledger."""
    xf = float(x)
    if not math.isfinite(xf):
        raise ValueError(f"non-finite number: {x!r}")
    return xf

# ---------------------------------------------------------------------------
# Economic / protocol constants (emission schedule borrowed from neurahash.chain)
# ---------------------------------------------------------------------------
MAX_SUPPLY = 2_100_000_000.0      # hard cap on NRH ever minted
INITIAL_REWARD = 5000.0           # block reward per block at height < HALVING_EVERY
HALVING_EVERY = 210000            # halve the reward every N blocks (Bitcoin-shape:
                                  # 5000 * 210000 * 2 == 2.1B == MAX_SUPPLY, mirroring
                                  # BTC's 50 * 210000 * 2 == 21M. At the 60s target block
                                  # time one halving era is ~145 days.)
MIN_STAKE = 10.0                  # minimum bonded stake to be an eligible proposer
EQUIVOCATION_BOUNTY_FRAC = 0.1    # fraction of a slashed equivocator's bond paid to the
                                  # reporter who included the fraud proof (rest -> treasury)
UNBONDING_PERIOD = 5              # blocks an `unstake` stays bonded-but-withdrawing before
                                  # its funds become liquid. During this window the funds
                                  # remain SLASHABLE, so a proposer cannot unstake out from
                                  # under a fraud proof. (Prototype value; a real chain sets
                                  # this >> the worst-case fraud-proof mining latency.)

# ---------------------------------------------------------------------------
# Difficulty retarget (Bitcoin-style, adapted to PoUW). "Difficulty" here is the REQUIRED
# useful-work FLOOR per block — a lower bound on a block's verified work_score (held-out
# loss improvement). Faster-than-target blocks RAISE the floor; slower blocks LOWER it
# (negative feedback toward TARGET_BLOCK_TIME, same sign as Bitcoin's hash-target retarget).
#
# The floor is carried as an INTEGER fixed-point count of WORK_FLOOR_SCALE^-1 units and ALL
# retarget arithmetic is integer, so two nodes never disagree by a floating-point ULP (the
# compounding-ULP consensus-split the design review flagged as fatal). Because a held-out
# loss improvement is a SHRINKING resource as the MoE converges, a naive retarget would
# ratchet the floor past achievable work and stall the chain forever; the OBSERVED-WORK
# CEILING (floor capped at OBSERVED_WORK_CEIL of a trailing EWMA of accepted work) and the
# per-height STALL backstop make the floor track the converging model DOWNWARD instead.
#
# These quantities are inert until Branch 3 (the fold) / Branch 4 (the consensus check) and
# only take effect at/after RETARGET_ACTIVATION_HEIGHT (Branch 5), gated atomically.
# ---------------------------------------------------------------------------
TARGET_BLOCK_TIME = 60            # seconds per block the retarget steers toward
RETARGET_WINDOW = 2016            # blocks per retarget epoch (Bitcoin-identical count)
TARGET_TIMESPAN = RETARGET_WINDOW * TARGET_BLOCK_TIME   # = 120960 s (~33.6h) per epoch
RETARGET_CLAMP_NUM = 4            # per-epoch adjustment clamped to [1/4x, 4x] (Bitcoin)

WORK_FLOOR_SCALE = 1_000_000_000          # fixed-point: 1 unit == 1e-9 of a work_score
GENESIS_WORK_FLOOR_UNITS = 1_000_000      # seed floor == 0.001 work_score (full upward
                                          # retarget authority near genesis; no quant dead-zone)
WORK_FLOOR_MIN_UNITS = 1                  # absolute lower clamp (never 0 -> never disables)
WORK_FLOOR_MAX_UNITS = 1_000_000_000_000  # absolute upper clamp / overflow guard
OBSERVED_WORK_CEIL_NUM = 3                # floor <= 3/4 of the trailing observed-achievable
OBSERVED_WORK_CEIL_DEN = 4                # work EWMA -> floor can't exceed demonstrated work
WORK_EWMA_NUM = 1                         # EWMA weight 1/16: ewma += (sample-ewma)*1//16
WORK_EWMA_DEN = 16
STALL_RELAX_BLOCKS = 10                   # if on-chain elapsed since last block exceeds
                                          # this * TARGET_BLOCK_TIME, relax the floor to MIN
                                          # for that height (per-height self-healing)
MTP_WINDOW = 11                           # median-time-past window (Bitcoin-identical) for
                                          # the retarget anchor / endpoint timestamps
MAX_BLOCK_FUTURE_DRIFT = 7200             # networking-layer guideline (seconds): honest miners
                                          # stamp real wall-clock, so a peer MAY soft-reject a
                                          # block whose timestamp is >2h ahead of its own clock.
                                          # NOT a consensus rule — clockless validation keeps the
                                          # loose monotonic bound so a block that recovers the
                                          # chain after a multi-hour stall is not rejected;
                                          # divisor grinding is bounded by the MTP-median anchor
                                          # + the [1/4x, 4x] clamp + the observed-work ceiling.
RETARGET_ACTIVATION_HEIGHT = 0            # height the retarget rules turn on. 0 on a fresh
                                          # genesis (the shipped value); a pre-existing chain
                                          # sets a FUTURE height via a protocol_version upgrade.

GENESIS_PREV_HASH = "0" * 64      # JSON-safe genesis parent sentinel
GENESIS_BEACON = "0" * 64         # genesis has no prev block -> fixed beacon sentinel
_GENESIS_SCORE = 0.0              # genesis carries no useful work

# ---------------------------------------------------------------------------
# Live protocol upgrades (the model-scale ladder needs consensus-breaking changes —
# bigger/architecture-changed models — to land on a RUNNING chain without trust or restart;
# see RESEARCH.md "June 2026 survey"). Mechanism: staked validators broadcast
# `upgrade_signal` txs for protocol_version+1; once the signaling stake stays at a >= 2/3
# supermajority for UPGRADE_WINDOW consecutive blocks the upgrade LOCKS IN, and it ACTIVATES
# UPGRADE_ACTIVATION_DELAY blocks later (the grace period for laggards to update software).
# From the activation height every block must carry the new protocol_version, and the first
# new-version block applies the deterministic checkpoint migration P_new = F(P_old)
# (Consensus.migrate_params) so the model lineage continues unbroken.
# ---------------------------------------------------------------------------
GENESIS_VERSION = 1               # protocol_version of genesis / un-upgraded chains
UPGRADE_THRESHOLD_NUM = 2         # signaling-stake supermajority required to lock in an
UPGRADE_THRESHOLD_DEN = 3         #   upgrade: signaled * DEN >= total_bonded * NUM
                                  #   (cross-multiplied -> no float-division threshold)
UPGRADE_WINDOW = 3                # consecutive blocks the supermajority must HOLD to lock in
                                  # (a momentary spike can't trigger a fork; demo value —
                                  # production would use epochs)
UPGRADE_ACTIVATION_DELAY = 3      # blocks between lock-in and activation (update-grace window;
                                  # production sets this >> client-rollout latency and ties the
                                  # boundary to a FINALIZED checkpoint — that needs chain
                                  # context in State, same frontier as open thread #1)


def _canon(x):
    """Full-precision, JSON-safe canonical string for a float (no rounding collisions,
    no non-standard 'Infinity'/'NaN' tokens). Identical recipe to neurahash.chain._canon
    so header hashes are reproducible across nodes and across the two packages."""
    xf = float(x)
    if not math.isfinite(xf):                     # fail closed: never hash a poisoned value
        raise ValueError(f"non-finite value in canonical serialization: {x!r}")
    return format(xf, ".17g")


# ===========================================================================
# Transactions
# ===========================================================================
class Tx:
    """A native-NRH transaction. kind in {'transfer','stake','unstake','slash'}.

    'sig' is a dev placeholder (no real crypto in this single-machine sim); it is
    included in the canonical bytes so two structurally-identical txs from different
    senders still produce distinct txids via 'sender'.

    'data' carries kind-specific structured payload (JSON-safe). For a 'slash' tx it holds
    {'equivocation': <EquivocationProof dict>, 'proposer': <node_id>} — the in-protocol
    fraud proof that, when applied, burns an equivocator's bond (see apply_tx). It is folded
    into the canonical bytes so the fraud proof is bound to the txid."""

    KINDS = ("transfer", "stake", "unstake", "slash", "upgrade_signal")

    def __init__(self, sender, kind, to=None, amount=0.0, nonce=0, sig="dev", data=None):
        if kind not in self.KINDS:
            raise ValueError(f"unknown tx kind: {kind!r}")
        self.sender = sender
        self.kind = kind
        self.to = to
        self.amount = float(amount)
        self.nonce = int(nonce)
        self.sig = sig
        self.data = data

    def to_dict(self):
        d = {
            "sender": self.sender,
            "kind": self.kind,
            "to": self.to,
            "amount": self.amount,
            "nonce": self.nonce,
            "sig": self.sig,
            "data": self.data,
        }
        # crypto-agility (#35): carry a NON-default signature-scheme tag through serialization
        # so a PQC-signed tx keeps its scheme across the mempool / block round-trip (else it
        # silently falls back to secp256k1 verification). Omitted for today's secp256k1 txs, so
        # existing mempool.json / block bytes are unchanged. NOT part of signing_bytes or txid
        # (those use _canonical_obj); folding the scheme into the SIGNED bytes is the Phase-1
        # anti-downgrade step.
        sc = getattr(self, "sig_scheme", None)
        if sc is not None:
            d["sig_scheme"] = sc
        return d

    @classmethod
    def from_dict(cls, d):
        tx = cls(sender=d["sender"], kind=d["kind"], to=d.get("to"),
                 amount=_finite(d.get("amount", 0.0)), nonce=int(d.get("nonce", 0)),
                 sig=d.get("sig", "dev"), data=d.get("data"))
        sc = d.get("sig_scheme")
        if sc is not None:
            tx.sig_scheme = sc                 # restore the optional scheme tag (#35)
        return tx

    def _canonical_obj(self, include_sig):
        obj = {
            "sender": self.sender,
            "kind": self.kind,
            "to": self.to,
            "amount": _canon(self.amount),
            "nonce": self.nonce,
            "data": json.dumps(self.data, sort_keys=True, allow_nan=False)
                    if self.data is not None else None,
        }
        if include_sig:
            obj["sig"] = self.sig
        return obj

    def canonical_bytes(self):
        """Deterministic, full-precision serialization for hashing (includes sig so two
        structurally-identical txs from different senders still produce distinct txids)."""
        return json.dumps(self._canonical_obj(include_sig=True),
                          sort_keys=True, allow_nan=False).encode()

    def signing_bytes(self):
        """The bytes a real secp256k1 signature is computed over: everything EXCEPT `sig`
        (the signature can't sign itself). neura_l1.signing.sign_tx/recover_tx_signer use
        this so the engine can authenticate `sender` against a registered key."""
        return json.dumps(self._canonical_obj(include_sig=False),
                          sort_keys=True, allow_nan=False).encode()

    def txid(self):
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def __repr__(self):
        return (f"Tx({self.sender}->{self.to} {self.kind} "
                f"{self.amount} n={self.nonce})")


def tx_root(txs):
    """Deterministic commitment over a list of Tx (order-preserving). Empty -> zero root.
    A simple sequential hash chain is sufficient and deterministic for the sim."""
    if not txs:
        return "0" * 64
    h = hashlib.sha256()
    for t in txs:
        h.update(bytes.fromhex(t.txid()))
    return h.hexdigest()


# ===========================================================================
# Block header
# ===========================================================================
class BlockHeader:
    """All consensus-critical fields. hash() is sha256 over the canonical JSON so any
    node recomputes the identical block hash. Floats use _canon (full precision)."""

    def __init__(self, height, prev_hash, beacon, parent_checkpoint, new_checkpoint,
                 work_score, proposer, reward, tx_root, timestamp,
                 protocol_version=GENESIS_VERSION, quorum_hash=""):
        self.height = int(height)
        self.prev_hash = prev_hash
        self.beacon = beacon                       # = sha256(prev_hash) hex (set by consensus)
        self.parent_checkpoint = parent_checkpoint  # hash_params(P_h)
        self.new_checkpoint = new_checkpoint        # hash_params(P_h + delta)
        self.work_score = float(work_score)
        self.proposer = proposer
        self.reward = float(reward)
        self.tx_root = tx_root
        self.timestamp = int(timestamp)
        # (B8-1) OPTIONAL sha256 of the M-of-N settlement quorum bundle attached to this block. Empty
        # for every non-quorum block => omitted from canonical() => byte-identical hash to today (no
        # consensus change for the live chain). When a quorum IS attached (the default-off B8 path),
        # this binds it INTO the coordinator-signed block hash, so the quorum can no longer be
        # stripped/swapped after signing (the residual the appended-after-sig quorum field left).
        self.quorum_hash = str(quorum_hash or "")
        # the consensus-rules version this block was produced under. NOT proposer-chosen:
        # validate_block requires it to equal parent_state.child_version(height), so there is
        # exactly one valid value per height (which is also why the signed Commitment doesn't
        # need to cover it — tampering it changes the block hash and fails the version check).
        self.protocol_version = int(protocol_version)

    def canonical(self):
        """JSON-safe dict with full-precision floats -> reproducible across nodes."""
        d = {
            "height": self.height,
            "prev_hash": self.prev_hash,
            "beacon": self.beacon,
            "parent_checkpoint": self.parent_checkpoint,
            "new_checkpoint": self.new_checkpoint,
            "work_score": _canon(self.work_score),
            "proposer": self.proposer,
            "reward": _canon(self.reward),
            "tx_root": self.tx_root,
            "timestamp": self.timestamp,
            "protocol_version": self.protocol_version,
        }
        if self.quorum_hash:                       # only when present => byte-identical hash when absent
            d["quorum_hash"] = self.quorum_hash
        return d

    def hash(self):
        return hashlib.sha256(
            json.dumps(self.canonical(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def to_dict(self):
        return self.canonical()

    @classmethod
    def from_dict(cls, d):
        # _finite rejects NaN/Infinity (incl. overflow literals like 1e400 that parse to inf)
        # so a tampered header can't inject a non-finite work_score/reward that later poisons
        # the reward split with NaN.
        return cls(
            height=d["height"], prev_hash=d["prev_hash"], beacon=d["beacon"],
            parent_checkpoint=d["parent_checkpoint"], new_checkpoint=d["new_checkpoint"],
            work_score=_finite(d["work_score"]), proposer=d["proposer"],
            reward=_finite(d["reward"]), tx_root=d["tx_root"], timestamp=d["timestamp"],
            protocol_version=int(d.get("protocol_version", GENESIS_VERSION)),
            quorum_hash=d.get("quorum_hash", ""),
        )


# ===========================================================================
# Block
# ===========================================================================
class Block:
    """A full block: header + the included txs + the PoUW proof payload.

    `pouw` is the opaque-to-this-module payload produced by
    neura_l1.pouw_gate.build_pouw_payload, shaped roughly:
        {'submission': {...}, 'challenge_answers': {...}, 'shard_meta': {...}}
    block_state only serializes it (it must be JSON/round-trip friendly); proof checking
    is done by pouw_gate during consensus.validate_block.

    `commitment` is the proposer's signed commit-reveal envelope (produced by
    neura_l1.signing.sign_commitment): a real secp256k1 signature over the block's
    consensus-critical header fields + the Merkle ROOT of the trajectory, shaped
    {'commitment': {...}, 'signer': <address>, 'signature': <0x hex>}. block_state only
    carries it through gossip; signature/equivocation checking lives in neura_l1.signing
    and is enforced by consensus.validate_block. Genesis (height 0) carries none.
    """

    def __init__(self, header, txs=None, pouw=None, commitment=None):
        self.header = header
        self.txs = list(txs) if txs else []
        self.pouw = pouw if pouw is not None else {}
        self.commitment = commitment

    # ---- convenient read-only accessors --------------------------------
    @property
    def hash(self):
        return self.header.hash()

    @property
    def height(self):
        return self.header.height

    @property
    def prev_hash(self):
        return self.header.prev_hash

    @property
    def block_work(self):
        """Verified useful-work weight this block contributes to chain_work."""
        return self.header.work_score

    # ---- serialization for gossip --------------------------------------
    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "txs": [t.to_dict() for t in self.txs],
            "pouw": self.pouw,
            "commitment": self.commitment,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            header=BlockHeader.from_dict(d["header"]),
            txs=[Tx.from_dict(t) for t in d.get("txs", [])],
            pouw=d.get("pouw", {}),
            commitment=d.get("commitment"),
        )

    # ---- genesis -------------------------------------------------------
    @classmethod
    def genesis(cls, genesis_checkpoint):
        """The height-0 block. Carries no useful work, no reward, no txs.
        parent_checkpoint == new_checkpoint == hash_params(P_0)."""
        hdr = BlockHeader(
            height=0,
            prev_hash=GENESIS_PREV_HASH,
            beacon=GENESIS_BEACON,
            parent_checkpoint=genesis_checkpoint,
            new_checkpoint=genesis_checkpoint,
            work_score=_GENESIS_SCORE,
            proposer="genesis",
            reward=0.0,
            tx_root=tx_root([]),
            timestamp=0,
        )
        return cls(header=hdr, txs=[], pouw={})

    def __repr__(self):
        return (f"Block(h={self.height} work={self.header.work_score:.4f} "
                f"proposer={self.header.proposer} hash={self.hash[:12]})")


# ===========================================================================
# State — native NRH accounts + deterministic transition
# ===========================================================================
class State:
    """In-memory, hashable, per-block-applied native-NRH state.

    Mirrors neurahash.ledger semantics (balances / staked / treasury / minted) but is
    re-implemented as a PURE transition so block validity is independently checkable and
    speculative-apply (for reorgs) is cheap via copy().

    `model_checkpoint` is the hash of the canonical model params at this state's tip
    (hash_params(P_h)). State NEVER holds the actual numpy params — only the hash — so it
    stays small and trivially comparable. The Node (p2p_node) advances the live params.
    """

    def __init__(self, model_checkpoint="genesis", require_tx_signatures=False,
                 retarget_enabled=False):
        # When True, every transfer/stake/unstake must carry a real secp256k1 signature
        # (Tx.sig) recovering to the address registered for tx.sender — closing the forged-tx
        # hole for chains that opt in (the local wallet/miner apps do). Off by default so the
        # in-process engine demos/tests keep using lightweight unsigned txs. NOT part of
        # state_hash (it is chain policy, not ledger contents), but is carried in copies and
        # persisted so a reloaded chain keeps enforcing.
        self.require_tx_signatures = bool(require_tx_signatures)
        # Master switch for the PoUW difficulty retarget (the work-floor that holds block time
        # to TARGET_BLOCK_TIME). Same policy pattern as require_tx_signatures: carried in
        # copies + persisted so a reloaded chain keeps the same rules, but NOT part of the
        # legacy state_hash. Off by default so the in-process engine demos/tests keep their
        # exact current behavior (no fold, no floor, byte-identical digests); the shipped
        # wallet/miner chain creates its genesis with this ON. When ON, the retarget state
        # (work_floor_q etc.) IS folded into state_hash so divergence is detected.
        self.retarget_enabled = bool(retarget_enabled)
        self.bal = {}          # address -> liquid balance
        self.staked = {}       # address -> bonded (locked) balance
        self.treasury = 0.0    # slashed funds accumulate here
        self.minted = 0.0      # total NRH emitted so far (cap is enforced against this)
        self.nonces = {}       # address -> next expected tx nonce (replay protection)
        self.keys = {}         # proposer node_id -> registered secp256k1 address (validator PKI)
        self.slashed_faults = set()   # fault ids ('<signer>:<height>') already slashed (replay)
        self.unbonding = {}    # address -> [[amount, release_height], ...] withdrawing stake
                               # (still SLASHABLE until released; matures to bal at release)
        self.model_checkpoint = model_checkpoint
        # --- live protocol-upgrade machinery (versioned consensus rules) ---
        self.protocol_version = GENESIS_VERSION
        self.upgrade_signals = set()   # stakers signaling readiness for protocol_version+1
                                       # (weight = their LIVE stake at each evaluation, so an
                                       # unbonding signaler automatically loses signal weight)
        self.upgrade_streak = 0        # consecutive blocks the >=2/3 supermajority has held
        self.pending_activation = None  # height the locked-in upgrade activates at (or None)
        # --- difficulty retarget state (folded forward by apply_block; see the constants
        # block above). All integer fixed-point so the cross-node digest never ULP-splits.
        # Seeded to the genesis defaults here; the genesis timestamp seeds the anchors at
        # chain creation (local_node). Inert until RETARGET_ACTIVATION_HEIGHT.
        self.work_floor_q = GENESIS_WORK_FLOOR_UNITS   # required work floor F this epoch (int)
        self.retarget_anchor_ts = 0    # first (median) timestamp of the current epoch
        self.retarget_anchor_h = 0     # height the current epoch anchored at
        self.last_block_ts = 0         # last accepted block's timestamp (per-height stall test)
        self.work_ewma_q = GENESIS_WORK_FLOOR_UNITS    # trailing observed-achievable work (int)
        self.ts_ring = []              # last MTP_WINDOW committed timestamps (median-time-past)

    # The native-transfer signature policy for a VALUE-BEARING chain (#37). The in-process
    # engine DEFAULT is OFF (lightweight unsigned txs for sims/tests); a chain that holds value
    # MUST require signatures so an unsigned or attacker-signed transfer can never be applied.
    # This named constant is the single source of truth the wallet/miner apps build genesis from
    # (they previously flipped a bare `= True` inline); demos/tests opt out explicitly.
    VALUE_CHAIN_REQUIRE_TX_SIGNATURES = True

    @classmethod
    def for_value_chain(cls, model_checkpoint="genesis", retarget_enabled=True):
        """THE genesis State constructor for a value-bearing chain (#37): native-tx signatures
        are REQUIRED, so an unsigned/forged transfer/stake/unstake is rejected by apply_tx, and
        the difficulty retarget is enabled. The wallet/miner apps use this policy; in-process
        demos/tests that want unsigned txs construct State(...) directly with the off default."""
        return cls(model_checkpoint,
                   require_tx_signatures=cls.VALUE_CHAIN_REQUIRE_TX_SIGNATURES,
                   retarget_enabled=retarget_enabled)

    # ---- speculative copy ----------------------------------------------
    def copy(self):
        s = State(self.model_checkpoint, self.require_tx_signatures, self.retarget_enabled)
        s.bal = dict(self.bal)
        s.staked = dict(self.staked)
        s.treasury = self.treasury
        s.minted = self.minted
        s.nonces = dict(self.nonces)
        s.keys = dict(self.keys)
        s.slashed_faults = set(self.slashed_faults)
        s.unbonding = {a: [list(e) for e in lst] for a, lst in self.unbonding.items()}
        s.protocol_version = self.protocol_version
        s.upgrade_signals = set(self.upgrade_signals)
        s.upgrade_streak = self.upgrade_streak
        s.pending_activation = self.pending_activation
        s.work_floor_q = self.work_floor_q
        s.retarget_anchor_ts = self.retarget_anchor_ts
        s.retarget_anchor_h = self.retarget_anchor_h
        s.last_block_ts = self.last_block_ts
        s.work_ewma_q = self.work_ewma_q
        s.ts_ring = list(self.ts_ring)
        return s

    # ---- reads ----------------------------------------------------------
    def balance(self, addr):
        return self.bal.get(addr, 0.0)

    def stake_of(self, addr):
        """ACTIVE bonded stake (what counts for proposer eligibility). Funds queued for
        unbonding do NOT count here — you can't propose on the strength of stake you are
        withdrawing — but they are still slashable (see at_risk_of)."""
        return self.staked.get(addr, 0.0)

    def unbonding_of(self, addr):
        """Total amount currently in this address's unbonding queue (withdrawing, not yet
        liquid, still slashable)."""
        return sum(amt for amt, _ in self.unbonding.get(addr, []))

    def at_risk_of(self, addr):
        """The total SLASHABLE balance: active bond + everything still in the unbonding
        window. This is what a fault burns, so unstaking can't dodge a slash."""
        return self.stake_of(addr) + self.unbonding_of(addr)

    def key_of(self, proposer):
        """The registered secp256k1 address bound to a proposer node_id (or None). This is
        the validator-PKI binding consensus checks a block signature against. In a real
        PoUW L1 the proposer field simply IS the address; here we register the binding at
        genesis so a thief can't sign under a victim's identity."""
        return self.keys.get(proposer)

    def register_key(self, proposer, address):
        """Bind a proposer node_id to a secp256k1 address (sim genesis validator
        registration; folded into state_hash so all nodes agree on who may sign as whom)."""
        self.keys[proposer] = address

    def child_version(self, height):
        """The protocol_version a block at `height` extending this state MUST carry — the
        single valid value per height: the locked-in next version once `height` reaches the
        activation boundary, else the current version. Consensus rejects any other value, so
        a proposer cannot jump the upgrade early or straggle on old rules after activation."""
        if self.pending_activation is not None and height >= self.pending_activation:
            return self.protocol_version + 1
        return self.protocol_version

    def upgrade_signaled_stake(self):
        """The LIVE bonded stake currently signaling for protocol_version+1 (an unstaked /
        unbonding signaler weighs 0 — signal weight cannot outlive the bond behind it)."""
        return sum(self.staked.get(a, 0.0) for a in self.upgrade_signals)

    # ---- emission schedule (borrowed from neurahash.chain.PoUWChain) ------
    def nominal_reward(self, height):
        """Scheduled reward at `height`, clamped so cumulative emission never exceeds
        MAX_SUPPLY. Halving every HALVING_EVERY blocks."""
        # Genesis carries no reward (height 0). Mirrors consensus.block_reward's guard so the
        # proposer-side writer and the consensus-side verifier agree on every input.
        if height <= 0:
            return 0.0
        r = INITIAL_REWARD * (0.5 ** (height // HALVING_EVERY))
        return max(0.0, min(r, MAX_SUPPLY - self.minted))

    # ---- mutating primitives (used by apply_tx / apply_block) ------------
    def _credit(self, addr, amt):
        self.bal[addr] = self.bal.get(addr, 0.0) + amt

    def _debit(self, addr, amt):
        self.bal[addr] = self.bal.get(addr, 0.0) - amt

    def apply_tx(self, tx, height=None):
        """Apply ONE tx in place. Returns (ok, reason). On failure nothing is mutated.
        Deterministic: identical state + tx -> identical result on every node.

        `height` is the height of the block this tx is being applied in; it is needed to set
        an `unstake`'s release height (height + UNBONDING_PERIOD). It is supplied by
        apply_block / consensus.validate_block / the mempool drainer; direct callers that
        never unstake may omit it (defaults to 0)."""
        # reject non-finite amounts up front: NaN/Inf would slip past `< 0` and balance
        # comparisons (which are vacuously false for NaN) and poison the ledger.
        if not math.isfinite(tx.amount):
            return False, "non-finite amount"
        if tx.amount < 0:
            return False, "negative amount"
        # slash txs carry an in-protocol fraud proof (equivocation or FFG vote fault); they
        # are nonceless and self-justifying (replay-protected by the fault id, not the sender
        # nonce). Their authority is the proof, not a sender signature, so they skip the sig
        # check below.
        if tx.kind == "slash":
            return self._apply_slash(tx)
        # replay protection: each account's txs must use strictly increasing nonces
        if tx.nonce != self.nonces.get(tx.sender, 0):
            return False, f"bad nonce {tx.nonce} (expected {self.nonces.get(tx.sender, 0)})"
        # AUTHENTICATION (opt-in): on chains that require it, a transfer/stake/unstake must
        # carry a real secp256k1 signature recovering to the address REGISTERED for the
        # sender. There is NO trust-on-first-use: an unregistered sender name is rejected
        # outright; only pre-registered identities may send (the apps register every
        # owner-only keystore account at genesis). This stops anyone who can write the
        # mempool from forging a tx from a registered account without its private key.
        if self.require_tx_signatures:
            from .signing import recover_tx_signer
            from .sig_scheme import UnsupportedSchemeError
            reg = self.keys.get(tx.sender)
            if reg is None:
                # NO trust-on-first-use: an unregistered name could be claimed by anyone who
                # can write the mempool (hijacking a funded-but-never-sent account). A sender
                # must be a pre-registered identity (the apps register every owner-only
                # keystore account's address at genesis). Recipients still need no key.
                return False, "sender is not a registered account"
            try:
                signer = recover_tx_signer(tx)
            except UnsupportedSchemeError as e:
                # a tx declaring a signature scheme this node can't verify is rejected
                # fail-closed, but report it as a SCHEME issue (not a forgery) so operators
                # can tell a version/upgrade mismatch from an actual bad signature (#35).
                return False, f"unsupported tx signature scheme: {e}"
            except Exception:
                return False, "invalid or missing tx signature"
            if reg.lower() != signer.lower():
                return False, "tx signature does not match sender's registered key"
        if tx.kind == "transfer":
            if tx.to is None:
                return False, "transfer missing recipient"
            if self.balance(tx.sender) < tx.amount:
                return False, "insufficient balance for transfer"
            self._debit(tx.sender, tx.amount)
            self._credit(tx.to, tx.amount)
            self.nonces[tx.sender] = tx.nonce + 1
            return True, "transfer applied"
        if tx.kind == "stake":
            if self.balance(tx.sender) < tx.amount:
                return False, "insufficient balance to stake"
            self._debit(tx.sender, tx.amount)
            self.staked[tx.sender] = self.staked.get(tx.sender, 0.0) + tx.amount
            self.nonces[tx.sender] = tx.nonce + 1
            return True, "stake applied"
        if tx.kind == "upgrade_signal":
            # a staked validator declares readiness for the NEXT protocol version. The signal
            # itself carries no weight — weight is the sender's LIVE stake, re-tallied every
            # block in apply_block, so unbonding silently withdraws the signal. Sequential
            # upgrades only (current+1): there is exactly one candidate at a time.
            try:
                version = int((tx.data or {})["version"])
            except Exception:
                return False, "upgrade_signal missing/malformed version"
            if version != self.protocol_version + 1:
                return False, (f"can only signal for version {self.protocol_version + 1} "
                               f"(got {version})")
            if self.stake_of(tx.sender) <= 0:
                return False, "upgrade_signal requires active stake"
            self.upgrade_signals.add(tx.sender)
            self.nonces[tx.sender] = tx.nonce + 1
            return True, f"upgrade signal recorded for version {version}"
        if tx.kind == "unstake":
            cur = self.stake_of(tx.sender)
            if cur < tx.amount:
                return False, "insufficient stake to unstake"
            # move the bond into the UNBONDING window: it is no longer active stake (can't
            # propose with it) but stays slashable until it matures at release_height. The
            # funds become liquid only when a block at >= release_height is applied
            # (State._mature_unbonding in apply_block). This is what stops a faulty proposer
            # from unstaking out from under a fraud proof.
            self.staked[tx.sender] = cur - tx.amount
            release = int(height if height is not None else 0) + UNBONDING_PERIOD
            self.unbonding.setdefault(tx.sender, []).append([float(tx.amount), release])
            self.nonces[tx.sender] = tx.nonce + 1
            return True, f"unstake queued (unbonding until height {release})"
        return False, f"unknown kind {tx.kind!r}"

    def _apply_slash(self, tx):
        """Deterministic in-protocol slash from a self-contained fraud proof. Dispatches on
        the proof kind carried in tx.data:

          * 'equivocation' — two signed block commitments, one signer, one height/parent
            (_apply_equivocation_slash);
          * 'vote_fault'   — two signed FFG votes by one validator violating a Casper
            commandment: double vote or surround vote (_apply_vote_fault_slash).

        Both proofs are SELF-CONTAINED signed evidence (re-verified here via deterministic
        ecrecover), so the slash is replayable from the tx alone — no chain context needed.
        (Contrast commit-WITHOUT-reveal, a negative that cannot be proven by a standalone
        tx; that economic slash needs chain context and remains an open thread.)"""
        data = tx.data or {}
        if "vote_fault" in data:
            return self._apply_vote_fault_slash(tx)
        return self._apply_equivocation_slash(tx)

    def _reporter_bounty_ok(self, tx):
        """A slash-tx reporter earns the bounty ONLY if the tx carries a real signature
        recovering to tx.sender's registered key — REGARDLESS of require_tx_signatures.
        Otherwise, on the default unauthenticated engine a mempool-writer could name an
        arbitrary sender on an UNSIGNED slash tx and pocket the bounty. The fault is still
        punished either way (any unpaid bounty burns to treasury)."""
        reg = self.keys.get(tx.sender)
        if reg is None:
            return False
        try:
            from .signing import recover_tx_signer
            return recover_tx_signer(tx).lower() == reg.lower()
        except Exception:
            return False

    def _apply_vote_fault_slash(self, tx):
        """Deterministic slash from an FFG accountable-safety fault proof: two votes signed
        by the SAME registered validator that violate a Casper commandment (double vote or
        surround vote — finality.verify_vote_fault). Mirrors the equivocation slash: verify
        the self-contained evidence, seize the validator's whole at-risk balance, pay an
        authenticated reporter the bounty, burn the rest to treasury, replay-protect by a
        deterministic fault id."""
        from . import finality as fin           # local: finality imports block_state
        data = tx.data or {}
        vf = data.get("vote_fault") or {}
        validator = vf.get("validator")
        if validator is None:
            return False, "vote-fault slash tx missing validator"
        try:
            vote_a = fin.Vote.from_dict(vf["vote_a"])
            vote_b = fin.Vote.from_dict(vf["vote_b"])
        except Exception as e:
            return False, f"malformed vote-fault proof: {e}"
        registered = self.keys.get(validator)
        if registered is None:
            return False, f"validator {validator!r} has no registered key"
        is_fault, kind = fin.verify_vote_fault(validator, vote_a, vote_b, registered)
        if not is_fault:
            return False, f"not a slashable vote fault: {kind}"
        # deterministic, order-independent fault id (the same pair in either order maps to
        # one id; a different conflicting pair is a distinct fault but seizes a now-empty bond)
        fault = "ffg:" + ":".join(sorted((vote_a.vote_id(), vote_b.vote_id())))
        if fault in self.slashed_faults:
            return True, "vote fault already slashed (idempotent)"
        bond = self._seize(validator)
        bounty = bond * EQUIVOCATION_BOUNTY_FRAC if self._reporter_bounty_ok(tx) else 0.0
        if bounty > 0:
            self._credit(tx.sender, bounty)
        self.treasury += (bond - bounty)
        self.slashed_faults.add(fault)
        return True, (f"slashed {bond:.4f} from {validator} for FFG {kind} "
                      f"(bounty {bounty:.4f} -> {tx.sender})")

    def _apply_equivocation_slash(self, tx):
        """Deterministic in-protocol slash from an equivocation fraud proof. Returns
        (ok, reason). PURE w.r.t. balances/stake/treasury (caller validates first via a
        speculative copy). The proof is re-verified here (ecrecover is deterministic) so
        EVERY node that applies this tx burns the identical amount and converges.

        Steps:
          (1) the tx must carry data = {'equivocation': <proof dict>, 'proposer': <node_id>};
          (2) verify_equivocation re-runs ecrecover on both signed commitments -> the proof
              must be a genuine same-signer, same-height, distinct-commit pair;
          (3) the named proposer's REGISTERED key (this state's validator PKI) must equal the
              recovered signer (so we slash the correct staking identity);
          (4) replay: a fault id '<signer>:<height>' is slashed at most once (idempotent
              success if already slashed, so re-inclusion never double-burns);
          (5) burn the proposer's WHOLE bond; pay the reporter (tx.sender) a fixed bounty
              fraction, the remainder to the treasury.
        """
        from .signing import EquivocationProof, verify_equivocation, Commitment
        data = tx.data or {}
        proof_d = data.get("equivocation")
        proposer = data.get("proposer")
        if not proof_d or proposer is None:
            return False, "slash tx missing equivocation proof / proposer"
        try:
            proof = EquivocationProof.from_dict(proof_d)
        except Exception as e:
            return False, f"malformed equivocation proof: {e}"
        ok, signer = verify_equivocation(proof)
        if not ok:
            return False, f"invalid equivocation proof: {signer}"
        registered = self.keys.get(proposer)
        if registered is None or registered.lower() != signer.lower():
            return False, "equivocation signer != registered key for named proposer"
        height = Commitment.from_dict(proof.a["commitment"]).height
        fault = f"{signer.lower()}:{height}"
        if fault in self.slashed_faults:
            return True, "equivocation already slashed (idempotent)"
        # seize the WHOLE at-risk balance (active bond + any in-flight unbonding funds), so
        # an equivocator who raced an `unstake` ahead of this proof is still fully slashed.
        bond = self._seize(proposer)
        bounty = bond * EQUIVOCATION_BOUNTY_FRAC if self._reporter_bounty_ok(tx) else 0.0
        if bounty > 0:
            self._credit(tx.sender, bounty)
        self.treasury += (bond - bounty)
        self.slashed_faults.add(fault)
        return True, (f"slashed {bond:.4f} from {proposer} for equivocation at height "
                      f"{height} (bounty {bounty:.4f} -> {tx.sender})")

    def credit_reward(self, contributor, amount):
        """Mint a block reward to a verified contributor and advance minted."""
        if amount <= 0:
            return 0.0
        self._credit(contributor, amount)
        self.minted += amount
        return amount

    def _seize(self, addr, amount=None):
        """Remove up to `amount` (or ALL if None) of an address's SLASHABLE balance and
        return the total seized — WITHOUT deciding where it goes (caller routes it to the
        treasury and/or a reporter bounty). Active stake is taken first, then unbonding
        entries in deterministic (release_height, amount) order, so slashing reaches funds a
        faulty proposer tried to withdraw during the unbonding window."""
        remaining = None if amount is None else max(0.0, float(amount))
        seized = 0.0
        # (1) active bond first
        cur = self.staked.get(addr, 0.0)
        take = cur if remaining is None else min(cur, remaining)
        if take > 0:
            self.staked[addr] = cur - take
            seized += take
            if remaining is not None:
                remaining -= take
        # (2) then unbonding entries (oldest release first)
        entries = self.unbonding.get(addr)
        if entries and (remaining is None or remaining > 0):
            kept = []
            for amt, rel in sorted(entries, key=lambda e: (e[1], e[0])):
                if remaining is not None and remaining <= 0:
                    kept.append([amt, rel])
                    continue
                t = amt if remaining is None else min(amt, remaining)
                seized += t
                if remaining is not None:
                    remaining -= t
                if amt - t > 0:
                    kept.append([amt - t, rel])
            if kept:
                self.unbonding[addr] = kept
            else:
                self.unbonding.pop(addr, None)
        return seized

    def _mature_unbonding(self, height):
        """Release any unbonding entries whose release_height has been reached at `height`:
        the amount becomes liquid balance. Deterministic block-level transition (run by
        apply_block) so every node frees the identical funds at the identical height."""
        for addr in list(self.unbonding.keys()):
            kept = []
            for amt, rel in self.unbonding[addr]:
                if rel <= height:
                    self._credit(addr, amt)
                else:
                    kept.append([amt, rel])
            if kept:
                self.unbonding[addr] = kept
            else:
                self.unbonding.pop(addr, None)

    def slash(self, addr, amount=None, reason="invalid-block"):
        """Burn (move to treasury) a faulty proposer's SLASHABLE balance (active bond +
        unbonding funds). `amount=None` slashes everything at risk. Returns the amount
        actually slashed."""
        seized = self._seize(addr, amount)
        self.treasury += seized
        return seized

    # ---- difficulty retarget fold ---------------------------------------
    def _fold_retarget(self, ns, height, header_ts, work_score):
        """Fold the difficulty-retarget state forward by one block. PURE INTEGER arithmetic
        (the only float that ever enters is `work_score`, quantized once to integer units via
        round(); it is byte-stable across nodes by the existing score-determinism contract, so
        every replaying node folds the IDENTICAL state). `self` is the PARENT state, `ns` is
        the child (== self.copy(), so it already carries the parent values verbatim).

        Runs only when the chain enabled the retarget AND the height has reached activation;
        otherwise it is a no-op carry (ns keeps the parent's inert seed values).

        Mechanism (Bitcoin-style, adapted to PoUW):
          * a median-time-past ring blunts single-block timestamp grinding of the divisor;
          * an integer EWMA tracks recently-demonstrated achievable work (work_ewma_q);
          * at each RETARGET_WINDOW boundary the floor is scaled by TARGET_TIMESPAN / actual
            elapsed (clamped to [1/4x, 4x]), then HARD-CAPPED at OBSERVED_WORK_CEIL of the
            EWMA so the floor tracks the converging model DOWNWARD and can never ratchet past
            achievable work (the convergence-stall fix).
        """
        if not self.retarget_enabled or height < RETARGET_ACTIVATION_HEIGHT:
            return
        t = int(header_ts)
        # (0) median-time-past over the last MTP_WINDOW committed timestamps
        ring = (list(self.ts_ring) + [t])[-MTP_WINDOW:]
        mtp = sorted(ring)[len(ring) // 2]
        ns.ts_ring = ring
        # (1) trailing observed-achievable-work EWMA (integer fixed-point). Quantize the
        # verified work_score once; thereafter the compounding is integer-only.
        sample_q = max(WORK_FLOOR_MIN_UNITS, round(max(0.0, work_score) * WORK_FLOOR_SCALE))
        ns.work_ewma_q = (self.work_ewma_q
                          + (sample_q - self.work_ewma_q) * WORK_EWMA_NUM // WORK_EWMA_DEN)
        ns.last_block_ts = t
        # (2) epoch boundary -> recompute the floor; else carry the parent's floor/anchors.
        if height % RETARGET_WINDOW == 0:
            raw = mtp - int(self.retarget_anchor_ts)
            lo = TARGET_TIMESPAN // RETARGET_CLAMP_NUM
            hi = TARGET_TIMESPAN * RETARGET_CLAMP_NUM
            clamped = min(hi, max(lo, raw if raw > 0 else lo))  # degenerate/<=0 epoch ramps UP
            new_q = self.work_floor_q * TARGET_TIMESPAN // clamped   # fast epoch => ratio>1 => F up
            new_q = min(WORK_FLOOR_MAX_UNITS, max(WORK_FLOOR_MIN_UNITS, new_q))
            ceil_q = max(WORK_FLOOR_MIN_UNITS,
                         ns.work_ewma_q * OBSERVED_WORK_CEIL_NUM // OBSERVED_WORK_CEIL_DEN)
            ns.work_floor_q = min(new_q, ceil_q)
            ns.retarget_anchor_ts = mtp
            ns.retarget_anchor_h = height
        # (non-boundary blocks already carry parent work_floor_q / anchors via copy())

    # ---- the block-level transition -------------------------------------
    def apply_block(self, block, contributors):
        """Return a NEW State that is `self` after applying `block`. PURE: does not
        mutate `self`. Steps, in deterministic order:

          (a) apply each included tx (must apply cleanly — caller/consensus is expected
              to have validated; an unappliable tx raises so the bug is loud);
          (b) mint the block reward (already halving/cap-clamped in the header) split by
              contributor weight, clamped a second time against the live cap so cumulative
              minted can never exceed MAX_SUPPLY even under a malformed header;
          (c) advance model_checkpoint to header.new_checkpoint (the verified P_h+delta).

        `contributors`: dict[address -> weight]; weights must be >= 0. The reward is
        split pro-rata by weight (matching neurahash.chain.mint_block). If total weight is
        0 (zero-useful-work block) NO reward is minted (Sybil/zero-work blocks gain
        nothing), and minted does not advance.
        """
        ns = self.copy()
        h = block.header.height

        # bind the committed tx_root to the txs actually applied (else a tamperer could swap
        # the block body while keeping the header). tx_root is part of the header hash, which
        # the chain signature / block-hash checks pin, so this makes the binding load-bearing.
        if block.header.tx_root != tx_root(block.txs):
            raise ValueError("apply_block: header.tx_root != tx_root(block.txs)")

        # (a0) mature any unbonding stake whose release height has been reached -> liquid.
        ns._mature_unbonding(h)

        # (a) transactions (height-aware: an unstake's release = h + UNBONDING_PERIOD)
        for tx in block.txs:
            ok, reason = ns.apply_tx(tx, height=h)
            if not ok:
                raise ValueError(f"apply_block: tx failed ({reason}) {tx!r}")

        # (a2) live-upgrade bookkeeping (deterministic; replayed identically by every node):
        #   * ACTIVATION: a block whose header carries the next version (consensus already
        #     enforced it equals child_version(h)) flips the state's protocol_version and
        #     clears the campaign bookkeeping.
        #   * LOCK-IN: otherwise, re-tally the signaling stake against LIVE bonds; the
        #     >=2/3 supermajority (cross-multiplied — no float-division threshold) must hold
        #     for UPGRADE_WINDOW consecutive blocks, then activation is scheduled
        #     UPGRADE_ACTIVATION_DELAY blocks out (the client-update grace window).
        hv = int(getattr(block.header, "protocol_version", GENESIS_VERSION))
        if hv > ns.protocol_version:
            ns.protocol_version = hv
            ns.upgrade_signals = set()
            ns.upgrade_streak = 0
            ns.pending_activation = None
        elif ns.pending_activation is None and ns.upgrade_signals:
            total = sum(ns.staked.values())
            signaled = ns.upgrade_signaled_stake()
            if total > 0 and signaled * UPGRADE_THRESHOLD_DEN >= total * UPGRADE_THRESHOLD_NUM:
                ns.upgrade_streak += 1
            else:
                ns.upgrade_streak = 0
            if ns.upgrade_streak >= UPGRADE_WINDOW:
                ns.pending_activation = h + UPGRADE_ACTIVATION_DELAY

        # (b) reward split by contributor weight, clamped to the live cap
        agg = {}
        for addr, w in contributors.items():
            w = float(w)
            if w < 0:
                raise ValueError("negative contributor weight")
            if w > 0:
                agg[addr] = agg.get(addr, 0.0) + w
        total_w = sum(agg.values())

        # header reward is authoritative but re-clamp against remaining cap for safety. NOTE:
        # apply_block is a SHARED mint primitive — the L1 consensus chain AND the pool's separate
        # ChainSettlement ledger both use it. The L1 EMISSION SHAPE (the halving curve) is enforced
        # for the consensus chain in consensus.validate_block (reward must equal block_reward(height,
        # parent.minted), unconditional) and re-checked on the L1 replay path (p2p_node._rebuild_state);
        # it is deliberately NOT enforced here, because the pool ledger follows its OWN operator-set
        # schedule (testnet_node.reward_at) on a DISTINCT supply, and clamping to the L1 curve here
        # would wrongly couple the two ledgers. Only the hard MAX_SUPPLY cap is common to both (F11).
        reward = max(0.0, min(float(block.header.reward), MAX_SUPPLY - ns.minted))
        if total_w > 0 and reward > 0:
            for addr, w in sorted(agg.items()):
                ns.credit_reward(addr, reward * (w / total_w))

        # (c) advance canonical model checkpoint to the verified delta result
        ns.model_checkpoint = block.header.new_checkpoint

        # (d) fold the difficulty-retarget state forward (no-op unless the chain enabled it).
        # The EWMA sample is the block's verified useful-work = the same max(0, work_score)
        # that weighted the reward split above.
        self._fold_retarget(ns, h, block.header.timestamp, max(0.0, block.header.work_score))
        return ns

    # ---- deterministic digest for cross-node comparison -----------------
    def state_hash(self):
        """A canonical digest over balances + staked + treasury + minted + checkpoint.
        Two nodes that processed the same chain produce the identical state_hash, so the
        demo can assert convergence without trusting any single node."""
        obj = {
            "bal": {a: _canon(v) for a, v in sorted(self.bal.items())},
            "staked": {a: _canon(v) for a, v in sorted(self.staked.items())},
            "treasury": _canon(self.treasury),
            "minted": _canon(self.minted),
            "nonces": {a: n for a, n in sorted(self.nonces.items())},
            "keys": {a: k for a, k in sorted(self.keys.items())},
            "slashed_faults": sorted(self.slashed_faults),
            "unbonding": {a: [[_canon(amt), int(rel)]
                              for amt, rel in sorted(self.unbonding[a], key=lambda e: (e[1], e[0]))]
                          for a in sorted(self.unbonding)},
            "model_checkpoint": self.model_checkpoint,
            "protocol_version": self.protocol_version,
            "upgrade_signals": sorted(self.upgrade_signals),
            "upgrade_streak": self.upgrade_streak,
            "pending_activation": self.pending_activation,
        }
        # Difficulty-retarget state enters the digest ONLY on a retarget-enabled chain, so a
        # legacy (retarget-off) chain's state_hash is byte-identical to before this feature —
        # no rolling-upgrade digest split. On an enabled chain the retarget state IS covered,
        # so any divergence in the floor/anchors/EWMA is detected at the convergence check.
        if self.retarget_enabled:
            obj["retarget"] = {
                "work_floor_q": int(self.work_floor_q),
                "retarget_anchor_ts": int(self.retarget_anchor_ts),
                "retarget_anchor_h": int(self.retarget_anchor_h),
                "last_block_ts": int(self.last_block_ts),
                "work_ewma_q": int(self.work_ewma_q),
                "ts_ring": [int(t) for t in self.ts_ring],
            }
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()

    # ---- JSON persistence (for the local wallet/miner apps) -------------
    def to_dict(self):
        """Full, JSON-safe serialization of the account state (used by the local-node
        persistence layer in neura_l1.local_node). Round-trips via from_dict."""
        return {
            "require_tx_signatures": self.require_tx_signatures,
            "retarget_enabled": self.retarget_enabled,
            "bal": {a: float(v) for a, v in self.bal.items()},
            "staked": {a: float(v) for a, v in self.staked.items()},
            "treasury": float(self.treasury),
            "minted": float(self.minted),
            "nonces": {a: int(n) for a, n in self.nonces.items()},
            "keys": dict(self.keys),
            "slashed_faults": sorted(self.slashed_faults),
            "unbonding": {a: [[float(amt), int(rel)] for amt, rel in lst]
                          for a, lst in self.unbonding.items()},
            "model_checkpoint": self.model_checkpoint,
            "protocol_version": self.protocol_version,
            "upgrade_signals": sorted(self.upgrade_signals),
            "upgrade_streak": self.upgrade_streak,
            "pending_activation": self.pending_activation,
            "work_floor_q": int(self.work_floor_q),
            "retarget_anchor_ts": int(self.retarget_anchor_ts),
            "retarget_anchor_h": int(self.retarget_anchor_h),
            "last_block_ts": int(self.last_block_ts),
            "work_ewma_q": int(self.work_ewma_q),
            "ts_ring": [int(t) for t in self.ts_ring],
        }

    @classmethod
    def from_dict(cls, d):
        # _finite rejects NaN/Infinity from a tampered/corrupt chain.json so it cannot
        # poison balances or the state hash.
        s = cls(d.get("model_checkpoint", "genesis"),
                bool(d.get("require_tx_signatures", False)),
                bool(d.get("retarget_enabled", False)))
        s.bal = {a: _finite(v) for a, v in d.get("bal", {}).items()}
        s.staked = {a: _finite(v) for a, v in d.get("staked", {}).items()}
        s.treasury = _finite(d.get("treasury", 0.0))
        s.minted = _finite(d.get("minted", 0.0))
        s.nonces = {a: int(n) for a, n in d.get("nonces", {}).items()}
        s.keys = dict(d.get("keys", {}))
        s.slashed_faults = set(d.get("slashed_faults", []))
        s.unbonding = {a: [[_finite(amt), int(rel)] for amt, rel in lst]
                       for a, lst in d.get("unbonding", {}).items()}
        s.protocol_version = int(d.get("protocol_version", GENESIS_VERSION))
        s.upgrade_signals = set(d.get("upgrade_signals", []))
        s.upgrade_streak = int(d.get("upgrade_streak", 0))
        pa = d.get("pending_activation")
        s.pending_activation = int(pa) if pa is not None else None
        # Retarget state. Old (pre-retarget) blobs lack these keys -> default to the genesis
        # seeds, which is exactly what a fresh chain starts from, so a v3 blob upgrades cleanly.
        s.work_floor_q = int(d.get("work_floor_q", GENESIS_WORK_FLOOR_UNITS))
        s.retarget_anchor_ts = int(d.get("retarget_anchor_ts", 0))
        s.retarget_anchor_h = int(d.get("retarget_anchor_h", 0))
        s.last_block_ts = int(d.get("last_block_ts", 0))
        s.work_ewma_q = int(d.get("work_ewma_q", GENESIS_WORK_FLOOR_UNITS))
        s.ts_ring = [int(t) for t in d.get("ts_ring", [])]
        return s

    def snapshot(self):
        """Human-readable (address, balance, staked) rows for the demo."""
        rows = []
        for a in sorted(set(self.bal) | set(self.staked)):
            rows.append((a, round(self.bal.get(a, 0.0), 6),
                         round(self.staked.get(a, 0.0), 6)))
        return rows

    def __repr__(self):
        return (f"State(minted={self.minted:.2f} treasury={self.treasury:.2f} "
                f"ckpt={self.model_checkpoint[:12]} hash={self.state_hash()[:12]})")
