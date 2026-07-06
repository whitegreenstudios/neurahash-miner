"""
worker_core.py — the SHARED miner/verifier core, extracted VERBATIM from sharded_pool_node.py.

This module holds the pieces BOTH the (private) coordinator monolith and the (future public)
neurahash-miner must run byte-for-byte identically: the worker mining loop (`run_worker`), the
frozen-expert trunk-recompute engine (`_recompute_trunk_delta`) and its verify predicate
(`_cossim`/`_norm_ratio`), the worker-as-verifier offer (`_serve_net_verify`), the work-signature
binding (`trunk_delta_hash`/`work_commit_payload`), the device/VRAM/identity glue, and the shared
constants. Having ONE definition that both sides import is what preserves recompute-verify
bit-exactness (cosine>=VERIFY_COS gate; the canonical failure is silently false-rejecting honest
miners — see docs/PUBLIC_MINER_EXTRACTION.md §4 and memory `pouw-verified-not-useful`).

IMPORTANT — DETERMINISM: the pin block below (TF32-off, thread caps, CUBLAS_WORKSPACE_CONFIG) is
applied at the TOP of this module, BEFORE `import torch`, exactly as in sharded_pool_node.py, so
every importer gets the identical numeric environment. The coordinator must NOT re-enable TF32 after
importing worker_core. Everything here is a VERBATIM move (no reformat / rename / logic change).

This module imports ONLY public-safe packaged modules (net_transport, tls, wire_compress,
storage_wire, pqc_admission, neurahash.identity, neurahash_torch.{model_torch,pool_model,corpus_torch,
shard_verify,trunk_verify_net}, neura_l1.{signing,gpu_miner}, gpu_pool_node._seed, testnet_node.PSK)
— never the private coordinator/consensus/economics core.
"""
import gc
import hashlib
import json
import os
import socket
import time

# Cap the BLAS/OMP thread pools BEFORE numpy/torch import (the env must precede the import to take effect).
# torch defaults to physical-core width — on a 2-core coordinator box the default 16-wide pools on a bigger
# machine waste memory + CPU. A launcher export of these vars overrides (setdefault).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # determinism hygiene (no-op unless deterministic algos on)

import numpy as np
import torch

# torch intra-op thread width: default 2 (a 2-core box). NEURAHASH_THREADS overrides (LOWMEM sets it to 1).
# Fewer threads = fewer per-thread scratch buffers = lower steady RSS, at some recompute throughput cost.
_NTHREADS = max(1, int(os.environ.get("NEURAHASH_THREADS", "2")))
torch.set_num_threads(_NTHREADS)         # match a 2-core box; the default over-threads + bloats RSS
try:
    torch.set_num_interop_threads(1)     # must precede any inter-op parallel work (module import time is OK)
except RuntimeError:
    pass                                 # already configured elsewhere — leave it

# RECOMPUTE-VERIFY DETERMINISM (matters once the coordinator recomputes on a GPU — Rung 1+): turn OFF TF32 so
# the coordinator's fp32 recompute of a worker's trunk delta stays direction-stable and clears the cosine gate
# instead of drifting on the 5090's tensor cores. Same-GPU-class + TF32-off keeps cosine ~1.0. We deliberately
# do NOT call torch.use_deterministic_algorithms(True): the MoE/embedding backward uses atomic scatter_add,
# which has no deterministic CUDA kernel and would THROW — and bit-exactness isn't needed for cosine>=0.92,
# only direction stability. Harmless on CPU. (See neurahash_torch/train_torch.py for the full recipe.)
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass

# ---- public-safe imports the worker/recompute path needs (doc §1b) ----
from neurahash.net_transport import (send_msg, recv_msg, SEND_TIMEOUT, enable_tcp_keepalive)
from neurahash import tls
from neurahash_torch.pool_model import build_pool_model, qwen_arch   # model factory: toy MoE OR dense base
from neurahash_torch.model_torch import MoETransformer
from neurahash_torch.corpus_torch import build_data, get_batch, corpus_sha, resolve_corpus_mode
from neurahash_torch.shard_verify import trunk_keys, tile_keys
# NOTE (public-miner v1): `neurahash_torch.trunk_verify_net` (the worker-as-verifier duty) is imported
# LAZILY inside `_serve_net_verify` below, NOT here. It transitively pulls the PRIVATE `trunk_committee`
# core, and the verifier duty is DEFERRED from the v1 public miner (doc §1c). A lazy import is
# behavior-preserving (it runs the same, just at first use), and it lets `import neurahash.worker_core`
# succeed when that module is ABSENT (the standalone-import leak proof).
from neura_l1.signing import gen_account, account_from_key, sign_bytes
from neura_l1.gpu_miner import plan_budget, total_vram_gb, free_vram_gb, DEFAULT_RESERVE_GB
from neurahash.identity import identity_payload   # leaf split-out (Step 1)
from gpu_pool_node import _seed
from testnet_node import PSK, PSK_IS_DEFAULT
# NOTE (public-miner v1): `neurahash.storage_wire` (the STORAGE role) is OPTIONAL. It transitively pulls
# the PRIVATE `storage_audit` core and the storage role is DEFERRED from the v1 public miner (doc §1c).
# Import it best-effort so the module loads with it ABSENT; every use below is guarded on `_HAS_STORAGE`.
# When present (private/full build), behavior is byte-identical to before (the miner advertises the
# storage cap and answers challenges from disk); when absent (public v1), the miner trains and mines
# without the storage role.
try:
    from neurahash import storage_wire
    _HAS_STORAGE = True
except Exception:
    storage_wire = None
    _HAS_STORAGE = False
from neurahash import wire_compress
from neurahash import pqc_admission


# ---- shared constants (moved from sharded_pool_node.py) ----
BASE_NAME = (os.environ.get("NEURAHASH_BASE", "") or "").strip()
DENSE_BASE = bool(BASE_NAME)                                         # NEURAHASH_BASE set => train a real dense base
BATCH = int(os.environ.get("NEURAHASH_BATCH", "1" if DENSE_BASE else "128"))  # dense base: tiny batch; from-
# scratch model: 128 default, but NOW env-tunable so a big card can raise the per-step batch to FILL its VRAM
# (the coordinator sends this batch to the worker AND recomputes with it, so the verify stays matched). Pair a
# raised NEURAHASH_BATCH with NEURAHASH_VRAM_CAP_GB so an over-large batch OOMs at the cap instead of spilling.
# WEIGHT DECAY: the AdamW default is 0.01. Make it an explicit, SHARED knob so the worker's train and the
# coordinator's recompute use the IDENTICAL value — they MUST match or the recompute-verify cosine/norm
# gate breaks (a different decay = a different trunk gradient = honest-reject). Default 0.01 preserves
# today's behavior exactly (a restart with no new env is byte-for-byte unchanged). A larger value is one
# lever against the held-out overfitting drift the soak surfaced, but it is DETERMINISM-SENSITIVE: change
# it on the coordinator and EVERY worker, never one side only.
WEIGHT_DECAY = float(os.environ.get("NEURAHASH_WEIGHT_DECAY", "0.01"))

# ---------------------------- COMPRESSED WIRE (thin-pipe miners) ----------------------------
WIRE_COMPRESS = wire_compress.normalize_mode(os.environ.get("NEURAHASH_WIRE_COMPRESS", "off"))

# Two ways to set the ceiling: NEURAHASH_VRAM_CAP_GB is an ABSOLUTE GB; NEURAHASH_VRAM_CAP_FRAC is a FRACTION
# of the card's AUTO-DETECTED total VRAM (e.g. 0.8 => cap at 80% of whatever GPU this runs on — handy for a
# mixed fleet so you never hardcode each card's size). If both are set the absolute GB wins.
VRAM_CAP_GB = float(os.environ.get("NEURAHASH_VRAM_CAP_GB", "0") or "0")
VRAM_CAP_FRAC = float(os.environ.get("NEURAHASH_VRAM_CAP_FRAC", "0") or "0")

# Min DIRECTION agreement (cosine) of submitted vs recompute. 0.92 separates honest work (~0.99-1.00,
# even under between-snapshot expert drift) from a no-work random delta (uncorrelated). REPLAY no
# longer rests on this margin: as of #38 a stale/replayed delta is HARD-rejected by the per-round
# freshness gate (round beacon + work signature) BEFORE recompute, so the 0.86-vs-0.92 cosine gap is
# now defense-in-depth, not the sole replay defense (that gap was thin on a near-converged plateau).
# ASSUMES same-hardware-class recompute (cross-vendor fp drift would lower the honest cosine -> a
# tolerance/holdout fallback is the separate cross-vendor issue, not yet built).
VERIFY_COS = 0.92
VERIFY_NORM_LO, VERIFY_NORM_HI = 0.25, 4.0   # allowed ||submitted|| / ||recompute|| band (AMOUNT of work)


def apply_vram_cap(device, cap_gb=None, cap_frac=None):
    """HARD-cap this process's CUDA caching allocator so the card can be filled but provably never spills
    into system RAM — past the cap the process OOMs instead of falling back to PCIe-speed shared memory. The
    ceiling is resolved, in priority order, from: an explicit ABSOLUTE GB (NEURAHASH_VRAM_CAP_GB), else a
    FRACTION of the AUTO-DETECTED total VRAM (NEURAHASH_VRAM_CAP_FRAC, e.g. 0.8 => 80% of whatever GPU this
    is). No-op on CPU / no CUDA / no knob set / a cap >= the card total. Defensive: any failure degrades to
    "no cap" rather than crashing the node. Returns the applied fraction, else None."""
    cap_gb = VRAM_CAP_GB if cap_gb is None else float(cap_gb)
    cap_frac = VRAM_CAP_FRAC if cap_frac is None else float(cap_frac)
    if str(device) != "cuda":
        return None
    try:
        total = total_vram_gb(0) or 0.0
        if total <= 0:
            return None
        if cap_gb > 0:                      # explicit absolute GB wins
            cap = cap_gb
        elif cap_frac > 0:                  # else a fraction of the auto-detected card total (e.g. 0.8 = 80%)
            cap = min(cap_frac, 1.0) * total
        else:
            return None                     # no knob set => no cap (byte-for-byte today's behavior)
        if cap >= total:
            print(f"[vram-cap] requested {cap:.1f} GB >= card total {total:.1f} GB — no effective cap",
                  flush=True)
            return None
        frac = max(0.0, min(1.0, cap / total))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"[vram-cap] CUDA allocator hard-capped to {cap:.1f} GB "
              f"({frac*100:.0f}% of the {total:.1f} GB card, auto-detected) — OOMs at the cap, never "
              f"spills to sysmem", flush=True)
        return frac
    except Exception as e:
        print(f"[vram-cap] could not apply VRAM cap (continuing uncapped): "
              f"{type(e).__name__}: {e}", flush=True)
        return None


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def worker_usable_vram_gb(device):
    """Usable VRAM (GB) this worker can dedicate to hosting an expert shard, reported in its hello for
    the post-training readiness gate (#23). Reuses the SAME tested plan_budget policy the GPU miner
    uses (free VRAM now minus a reserve, capped at a utilisation fraction). 0.0 on CPU / no CUDA.
    Advisory only (never paid on) and an ESTIMATE — see posttrain_gate / GLM52_MILESTONE.md."""
    if str(device) != "cuda":
        return 0.0
    total = total_vram_gb(0)
    if not total:
        return 0.0
    free = free_vram_gb(0)
    free = total if free is None else free
    return round(plan_budget(total, free, reserve_gb=DEFAULT_RESERVE_GB), 2)


def load_or_create_key(address, key_dir=".neurahash_keys"):
    """PERSISTENT worker identity (security #16 / #13): load this address's secp256k1 key from disk, or
    create + save it on first use. A stable key across sessions is what makes a worker the SAME identity
    on reconnect (so it isn't TOFU-rejected and so a slashed identity can't escape by restarting with a
    fresh key). The file holds a private key — it is gitignored; keep it secret."""
    os.makedirs(key_dir, exist_ok=True)
    path = os.path.join(key_dir, f"{str(address).replace(os.sep, '_')}.key")
    if os.path.exists(path):
        with open(path) as f:
            return account_from_key(f.read().strip())
    acct = gen_account()
    with open(path, "w") as f:
        f.write(acct.key.hex())
    return acct


def expert_state_keys(e, n_layers):
    return [k for layer in range(n_layers) for k in tile_keys(layer, e)]


# ----------------------------- worker -----------------------------
def run_worker(host, port, address, honest=True, psk=PSK, device=None, retries=50, signer=None):
    """`signer` (optional): an external signer (neurahash.wallet ExternalSigner — hardware wallet /
    HSM / signing subprocess) that signs the admission challenge + work commitments WITHOUT a private
    key on this machine. When None (default) the behaviour is unchanged: a local per-node key is
    loaded/created and used. An external-signer worker mines normally but does NOT act as a networked
    committee verifier (that path needs a local Account; a miss is tolerated by the quorum)."""
    # (#35 Phase 2) HYBRID PQC: if NEURAHASH_PQC=hybrid, FAIL LOUDLY before connecting unless the
    # FIPS-204 backend is installed — never silently join as a classical-only worker when hybrid was
    # requested (the coordinator would then have no ML-DSA proof to pin). No-op when PQC is off.
    pqc_admission.require_backend_or_die()
    device = device or pick_device()
    apply_vram_cap(device)          # HARD VRAM ceiling (NEURAHASH_VRAM_CAP_GB) — fill the card, never spill
    # JOIN PATIENCE (#92 follow-up): the coordinator services new admissions in a small window ONCE PER
    # ROUND, so with long rounds (medium+ rungs) a joiner whose connect/handshake gives up in ~10s loses
    # the timing lottery almost every attempt (observed live: Colab/remote joiners flapping for many
    # minutes). Setting NEURAHASH_CLIENT_JOIN_TIMEOUT to a value LONGER than a round (e.g. 90) lets the
    # attempt sit in the accept backlog until the next join window instead of abandoning — turning a
    # ~20%-per-attempt lottery into a first-attempt join. 0/unset keeps today's defaults (10s connect,
    # tls.HANDSHAKE_TIMEOUT handshake).
    try:
        join_patience = float(os.environ.get("NEURAHASH_CLIENT_JOIN_TIMEOUT", "0") or 0)
    except ValueError:
        join_patience = 0.0
    sock = None
    for _ in range(retries):
        try:
            sock = socket.create_connection((host, int(port)), timeout=(join_patience or 10)); break
        except OSError:
            time.sleep(0.1)
    if sock is None:
        raise ConnectionError(f"could not connect to {host}:{port}")
    # (session-lifecycle) Arm tightened TCP keepalive on the WORKER's outbound socket BEFORE the TLS wrap
    # (the option lives on the underlying socket). A cross-subnet worker (e.g. the 4060 routed through a
    # home router to the coordinator) was having its idle flow reaped between rounds by the router's
    # conntrack timeout, then eagerly reconnecting -> repeated dup-reconnect evictions on the coordinator
    # (observed 12x live). ~25s keepalive probes keep the flow warm even while the worker waits for its
    # next task. Best-effort; unsupported options are skipped.
    enable_tcp_keepalive(sock)
    sock = tls.maybe_wrap_client(sock, tls.resolve_client_pin(),   # (#40) opt-in TLS + pin (encrypts weights/deltas)
                                 handshake_timeout=(join_patience or None))
    sock.settimeout(None)

    # (#85) STORAGE CAPABILITY: advertise that this build understands the storage challenge/response
    # message types (additive hello field; an OLD client omits it, so the coordinator never sends it
    # storage frames it would ignore — fail-CLOSED like the #44 admission bump). A store of the miner's
    # assigned coded chunks lives under NEURAHASH_STORAGE_DIR/<addr>/ and answers challenges from DISK.
    # (public-miner v1: storage_wire is optional — absent => no store, no STORAGE_CAP hello field; the
    # coordinator then never sends storage frames, exactly like an old client, and the worker mines fine.)
    _storage_store = storage_wire.StorageStore(address) if _HAS_STORAGE else None
    # (#35 Phase 2) HYBRID PQC: when NEURAHASH_PQC=hybrid, advertise a `pqc` capability + this identity's
    # ML-DSA-44 (FIPS-204) pubkey alongside the classical secp256k1 identity, and co-sign the #44
    # admission challenge with BOTH keys. Additive hello fields (mirrors STORAGE_CAP); PQC-off / old
    # workers omit them entirely, so the coordinator's admission gate is byte-identical to #44 for them.
    _hello = {"type": "hello", "address": address,
              "vram_gb": worker_usable_vram_gb(device),               # #23 readiness telemetry
              # (#92 follow-up) COMPRESSED-WIRE capability: the modes this build can DEQUANTIZE. Additive —
              # an old coordinator ignores the field and keeps sending raw fp32; an old worker omits it and
              # the coordinator never sends it a compressed frame. So this only ever ENABLES compression
              # when BOTH sides understand it.
              wire_compress.WIRE_CAP: list(wire_compress.WIRE_MODES)}
    if _HAS_STORAGE:                                                  # #85 capability (only when storage_wire present)
        _hello[storage_wire.STORAGE_CAP] = storage_wire.STORAGE_CAP_VERSION
    _pqc_pk_hex, _pqc_sk = (None, None)
    if pqc_admission.pqc_hybrid_enabled():
        _pqc_pk_hex, _pqc_sk = pqc_admission.load_or_create_pqc_key(address)
        _hello.update(pqc_admission.worker_hello_fields(_pqc_pk_hex))     # {pqc: 1, pqc_pk: <hex>}
    send_msg(sock, _hello, key=psk)
    # PERSISTENT key-bound identity: same key across reconnects (#13/#16). Load it NOW so we can answer
    # the coordinator's admission challenge (#44) BEFORE we're granted a slot. With an EXTERNAL signer
    # (hardware/watch-only wallet) there is NO local key — `acct` stays None and `_sign` delegates to
    # the device; the committee-verify path (which needs a real Account) is then skipped.
    acct = load_or_create_key(address) if signer is None else None
    _sign = (lambda data: sign_bytes(acct, data)) if signer is None else signer.sign_bytes
    msg = recv_msg(sock, key=psk)
    if isinstance(msg, dict) and msg.get("type") == "challenge":          # (#44) prove key control first
        _auth = {"type": "auth", "sig": _sign(identity_payload(address, int(msg["nonce"])))}
        if _pqc_sk is not None:                             # (#35) add the ML-DSA co-signature (hybrid)
            _auth.update(pqc_admission.worker_auth_fields(_pqc_sk, address, int(msg["nonce"]), _pqc_pk_hex))
        send_msg(sock, _auth, key=psk)
        msg = recv_msg(sock, key=psk)                       # ...then receive the model payload
    hello = msg
    arch, vocab = dict(hello["arch"]), int(hello["vocab"])
    n_layers, block = arch["n_layers"], arch["block_size"]
    batch = int(hello.get("batch", BATCH))
    hosted = list(hello.get("hosted", []))
    n_experts = int(hello.get("n_experts", arch["n_experts"]))
    session_nonce = int(hello.get("nonce", 0))             # == the challenge nonce (coordinator echoes it)

    def build(n_exp, host_list):
        a = dict(arch); a["n_experts"] = int(n_exp)
        # load_base=False: the worker receives the canonical trunk (= whole dense model) from the
        # coordinator each round, so importing the base here would just be overwritten (and waste VRAM).
        return build_pool_model(a, vocab, host_list, device, load_base=False)

    # CORPUS MODE — HELLO-DICTATED (#90): the coordinator advertises the corpus MODE it trains on (like it
    # already dictates the arch) so a real-corpus / qwen-BPE rung needs NO env change on this miner. The
    # hello value WINS over this box's NEURAHASH_CORPUS env; we print a one-line note when they differ so
    # the operator sees the coordinator overrode their env. An OLD coordinator omits the field -> the
    # `corpus` key is absent -> we fall back to this box's env behavior EXACTLY as before (back-compat).
    env_mode = resolve_corpus_mode()                            # what THIS box's env alone would train on
    hello_mode = hello.get("corpus")                            # None if the coordinator predates #90
    if hello_mode is not None:
        corpus_mode = resolve_corpus_mode(hello_mode)           # coordinator dictates -> worker adopts
        if corpus_mode != env_mode:
            print(f"[worker {address}] corpus mode: adopting coordinator's '{corpus_mode}' "
                  f"(overrides this box's NEURAHASH_CORPUS='{env_mode}')", flush=True)
    else:
        corpus_mode = None                                      # old coordinator -> env-only (unchanged)

    # CORPUS CONTENT-HASH CHECK (must precede training): the coordinator advertises the sha256 of its
    # corpus_data/*.txt in the hello. If this worker's local corpus differs, every recompute-verify
    # would SILENTLY reject our honest work (corpus_torch admits both sides must read identical data),
    # so refuse to start with a CLEAR error instead of mining rejected rounds forever. The coordinator
    # may be old (no corpus_sha in hello) — then skip the check for backward compatibility. The sha is
    # computed under the ADOPTED mode so it compares the same data source the coordinator hashed.
    coord_sha = hello.get("corpus_sha")
    if coord_sha:
        local_sha = corpus_sha(corpus_mode)
        if local_sha != coord_sha:
            sock.close()
            raise ValueError(
                f"corpus mismatch: coordinator {coord_sha[:16]}… vs local {local_sha[:16]}… — your "
                f"corpus_data differs from the coordinator's (different *.txt files, or "
                f"NEURAHASH_CORPUS/NEURAHASH_CORPUS_DIR points elsewhere). Sync your corpus_data/ "
                f"to match the coordinator before joining, or the pool will reject all your work.")

    model = build(n_experts, hosted)
    tok, train_data, val_data = build_data(device, seed=0, mode=corpus_mode)
    # Sign the join bound to the coordinator's fresh per-session NONCE so the signature can't be replayed
    # against an offline worker to wrongfully slash it (#16); reused as `sig` in every work submission.
    ident_sig = _sign(identity_payload(address, session_nonce))
    print(f"[worker {address}] connected | device {device} | hosts {hosted} of {n_experts} experts "
          f"| honest={honest}", flush=True)

    def load_np(sd, d):
        with torch.no_grad():
            for k, v in d.items():
                sd[k].copy_(torch.from_numpy(np.ascontiguousarray(v)).to(device))

    custody = {}                                       # (M6) chunk_key -> uint8 chunk this node is custodian of
    # (session-lifecycle fix 7) OBSERVABILITY. Every observed clean "[worker …] done." was a MASKED
    # connection death: the coordinator only ever sends {"type":"done"} at shutdown, so on a live pool a
    # "done." meant the recv had errored (NAT reap / coordinator drop) and been swallowed. Track WHY the
    # loop ended so the final line names it — protocol-done vs link-lost + the exception — and the silence
    # that cost a full day of blind debugging never recurs.
    exit_reason = "link-lost (loop exited without a coordinator 'done')"
    try:
        while True:
            msg = recv_msg(sock, key=psk)
            t = msg.get("type")
            if t == "done":
                exit_reason = "protocol-done (coordinator sent 'done')"
                break
            if t == "ping":
                continue
            if t == "verify_request":                          # (B7-2b) act as a networked verifier for
                if acct is not None:                           # ANOTHER worker's delta, then keep waiting
                    try:                                       # (external-signer workers skip this: it needs
                        _serve_net_verify(sock, msg, acct, arch, vocab, train_data, batch, device, psk)
                    except Exception:
                        pass                                   # best-effort: a miss is tolerated by the quorum
                continue
            # (#85) STORAGE ROLE: persist the assigned coded chunks to DISK and answer a beacon-seeded
            # challenge by reading them BACK from disk (never from the network — that is the soundness
            # point). Best-effort + guarded: a storage error must never break this worker's TRAINING loop.
            # (public-miner v1: guarded on `_HAS_STORAGE` — with storage_wire absent the coordinator never
            # advertises us as a store, so these frames don't arrive; the guard makes that explicit.)
            if t == "storage_commit":                          # record the commitment meta (chunks stream next)
                if _HAS_STORAGE:
                    try:
                        _storage_store.persist_commit(msg["commitment"])
                    except Exception:
                        pass
                continue
            if t == "storage_chunk":                            # persist ONE assigned coded chunk + its proof
                if _HAS_STORAGE:
                    try:
                        _storage_store.persist_chunk(
                            msg["commitment_id"], msg["index"],
                            storage_wire.chunk_from_wire(msg["chunk"]), msg["proof"])
                    except Exception:
                        pass
                continue
            if t == "storage_challenge" and _HAS_STORAGE:       # answer FROM DISK -> storage_proof
                try:
                    responses = _storage_store.answer(msg["commitment_id"], msg["indices"])
                    send_msg(sock, {"type": "storage_proof",
                                    "commitment_id": msg["commitment_id"],
                                    "beacon": msg.get("beacon", ""), "responses": responses}, key=psk)
                except (ConnectionError, OSError):
                    raise                                       # a real disconnect -> let the outer loop handle it
                except Exception:
                    pass                                        # malformed challenge -> ignore (coordinator's
                    #                                             verify then fails us honestly, never crashes)
                continue
            if t == "custody_push":                            # (M6) store RS backup chunks for OTHER nodes'
                # a push carries this node's COMPLETE current custody assignment, so REPLACE (not merge):
                # this evicts chunks it is no longer a custodian of after a re-partition, keeping the store
                # bounded on small nodes instead of accumulating every chunk it ever held. (no reply)
                custody = {str(ck): np.asarray(arr, dtype=np.uint8)
                           for ck, arr in (msg.get("chunks") or {}).items()}
                continue
            if t == "custody_challenge":                       # (M6) prove retrievability of a sampled subset
                have = {str(ck): custody[str(ck)] for ck in (msg.get("keys") or []) if str(ck) in custody}
                try:
                    send_msg(sock, {"type": "custody_proof", "beacon": msg.get("beacon", ""),
                                    "chunks": have}, key=psk)
                except OSError:
                    break
                continue
            if t == "custody_fetch":                           # (M6) serve chunks so the coordinator can
                have = {str(ck): custody[str(ck)] for ck in (msg.get("keys") or []) if str(ck) in custody}
                try:                                           # reconstruct an orphaned expert on churn
                    send_msg(sock, {"type": "custody_chunks", "req": msg.get("req", ""),
                                    "chunks": have}, key=psk)
                except OSError:
                    break
                continue
            if t != "task":
                continue
            # (#92 follow-up) COMPRESSED WIRE. The coordinator quantized this worker's payloads iff it set
            # `wire_compress` on the task (only for a worker that advertised the capability). `_wire` is the
            # active UPLINK mode this worker downcasts its delta/experts to; `_deq` dequantizes an incoming
            # payload — it is SELF-DESCRIBING (wire_compress tags the dict), so an untagged/raw fp32 payload
            # passes through unchanged. Critically, the worker trains from the DEQUANTIZED trunk, which is
            # BIT-IDENTICAL to the reference the coordinator kept for this worker's recompute (wire_compress
            # P2), so the honest-cosine gate is unaffected.
            _wire = wire_compress.normalize_mode(msg.get("wire_compress"))
            _deq = wire_compress.dequantize_state
            if "reassign" in msg:                          # membership/growth event -> rebuild model
                ra = msg["reassign"]
                hosted, n_experts = list(ra["hosted"]), int(ra["n_experts"])
                model = build(n_experts, hosted)
                sd = model.state_dict()
                load_np(sd, _deq(ra["trunk"])); load_np(sd, _deq(ra["experts"]))
                print(f"[worker {address}] reassigned -> hosts {hosted} of {n_experts} experts",
                      flush=True)
            else:                                          # steady round: the synced trunk (+ maybe experts)
                sd = model.state_dict()
                load_np(sd, _deq(msg["trunk"]))
                if "experts" in msg:                       # (M1) coordinator re-pushed its CANONICAL experts
                    load_np(sd, _deq(msg["experts"]))      # post-snapshot -> re-sync our frozen reference so
                    #                                        Phase 1 stays exact even after a rollback/reject
            recv_trunk = {k: sd[k].detach().clone() for k in trunk_keys(sd)}
            s, e = msg["shard_range"]; data = train_data[int(s):int(e)]
            hh, lr = int(msg["H"]), float(msg["lr"]); seed = _seed(msg["round"], address)
            # PER-WORKER BATCH: use the coordinator-assigned per-round batch when present (capacity-aware),
            # else the hello-level `batch` — so when the coordinator doesn't send one, rbatch == batch and
            # the training trajectory + n_examples are byte-identical to today. The coordinator recomputes
            # with this SAME per-worker batch (threaded into the task), so the seeded batch draw matches.
            rbatch = int(msg.get("batch", batch))

            is_trunk = lambda nm: ".moe.experts." not in nm
            if honest:
                # PHASE 1 — the VERIFIED trunk delta: train ONLY the trunk; the experts stay FROZEN at
                # the shared last-snapshot state the coordinator holds. Freezing experts is what removes
                # the between-snapshot DRIFT that rotated the trunk gradient and collapsed the verify
                # cosine — the deadlock (finding #1). The coordinator's recompute trains the trunk the
                # exact same way (experts frozen at its gathered copy), so honest deltas reproduce.
                for nm, p in model.named_parameters():
                    p.requires_grad_(is_trunk(nm))
                opt = torch.optim.AdamW([p for nm, p in model.named_parameters() if is_trunk(nm)],
                                        lr=lr, weight_decay=WEIGHT_DECAY)
                gen = torch.Generator(device=device); gen.manual_seed(seed)
                last = 0.0
                for _ in range(hh):
                    x, y = get_batch(data, block, rbatch, device, generator=gen)
                    _, loss = model(x, y)
                    opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
                nt = model.state_dict()
                trunk_delta = {k: (nt[k] - recv_trunk[k]).detach().cpu().float().numpy() for k in trunk_keys(sd)}
                train_loss = last
                # PHASE 2 — EXPERT progress (snapshot rounds only): NOW train the experts (trunk frozen)
                # so the snapshot the coordinator gathers actually improves them. Experts advance ONLY
                # here, gathered in lockstep with the coordinator, so Phase 1's frozen reference stays
                # identical on both sides. (FOLLOW-UP: on the rare round the coordinator REJECTS + rolls
                # back a snapshot's experts, worker and coordinator briefly diverge until the next
                # reassign resyncs; pushing the coordinator's canonical experts back post-snapshot would
                # keep even that case exact.)
                if msg.get("snapshot") and any(not is_trunk(nm) for nm, _ in model.named_parameters()):
                    for nm, p in model.named_parameters():
                        p.requires_grad_(not is_trunk(nm))
                    opt_e = torch.optim.AdamW(
                        [p for nm, p in model.named_parameters() if not is_trunk(nm)],
                        lr=lr, weight_decay=WEIGHT_DECAY)
                    gen_e = torch.Generator(device=device); gen_e.manual_seed((seed ^ 0x5bd1e995) & 0x7fffffff)
                    for _ in range(hh):
                        x, y = get_batch(data, block, rbatch, device, generator=gen_e)
                        _, loss = model(x, y)
                        opt_e.zero_grad(); loss.backward(); opt_e.step()
            else:
                g = torch.Generator(device=device); g.manual_seed(seed + 7)
                trunk_delta = {k: (0.3 * torch.randn(recv_trunk[k].shape, generator=g,
                                   device=device)).detach().cpu().float().numpy() for k in trunk_keys(sd)}
                train_loss = 0.01

            # (#92 follow-up) DOWNCAST the result delta to the active wire mode. The work-sig + delta hash
            # are taken over the DEQUANTIZED reference (== what the coordinator reconstructs off the wire,
            # bit-identical by wire_compress P2 and always float32, so the dtype byte in trunk_delta_hash is
            # stable). The WIRE form (fp16 ndarrays / int8+scale) is what actually rides `trunk_delta` — that
            # is the 2x/4x saving on the uplink. `_wire == off` => wire_delta IS the fp32 dict and delta_ref
            # is the same arrays, byte-identical to today.
            if _wire != "off":
                wire_delta = wire_compress.quantize_state(trunk_delta, _wire)
                delta_ref = wire_compress.dequantize_state(wire_delta)   # coordinator will get this exactly
            else:
                wire_delta, delta_ref = trunk_delta, trunk_delta
            # per-round freshness binding (#38): echo the coordinator's round beacon and SIGN
            # (address, round, beacon, delta_hash) with the per-node key, so this delta can't be
            # replayed into another round, swapped after signing, nor forged for this address by a
            # mere PSK-holder.
            beacon = msg.get("beacon", "")
            work_sig = (_sign(work_commit_payload(address, msg["round"], beacon,
                                                  trunk_delta_hash(delta_ref)))
                        if beacon else "")
            sub = {"address": address, "hosted": hosted, "trunk_delta": wire_delta,
                   "train_loss": float(train_loss), "n_examples": hh * rbatch, "sig": ident_sig,
                   "beacon": beacon, "work_sig": work_sig}
            if msg.get("snapshot"):
                cur = model.state_dict()
                ekeys = [k for ee in hosted for k in expert_state_keys(ee, n_layers)]
                if ekeys:                                  # a dense base has no experts -> no experts payload
                    exp_np = {k: cur[k].detach().cpu().float().numpy() for k in ekeys}
                    # downcast the uploaded experts too (the coordinator dequantizes before validating +
                    # applying them); untagged/off passes fp32 through unchanged.
                    sub["experts"] = (wire_compress.quantize_state(exp_np, _wire)
                                      if _wire != "off" else exp_np)
            send_msg(sock, {"type": "result", "round": msg["round"], "sub": sub}, key=psk)
            print(f"[mine {address}] round {msg['round']} | hosting "
                  f"{hosted if hosted else 'trunk-only'} | train_loss {float(train_loss):.4f} | sent OK",
                  flush=True)
    except (ConnectionError, OSError, ValueError, KeyError) as _e:
        # (session-lifecycle fix 7) ALWAYS surface the swallowed exception repr — NOT just ValueError/
        # KeyError. The most damaging case was exactly the ConnectionError/OSError link death that used to be
        # swallowed silently: a NAT/conntrack reap of the idle flow (the pending-joiner bug) or a coordinator
        # drop errored the blocked recv, and the loop fell through to a clean-looking "done." with no reason.
        # A ValueError/KeyError is still additionally a real protocol bug (this is how the dense-base wire-cap
        # drops were diagnosed). Record the reason for the final line so no session ever dies unexplained.
        exit_reason = f"link-lost: {type(_e).__name__}: {_e}"
        print(f"[worker {address}] session ended: {type(_e).__name__}: {_e}", flush=True)
    finally:
        sock.close()
    print(f"[worker {address}] done — {exit_reason}.", flush=True)


# ----------------------------- trustless verification (recompute) -----------------------------
def _recompute_trunk_delta(arch0, vocab, hosted, full_state, trunk_np, shard_data, seed, H, lr,
                           batch, block, NL, E, device, wire_mode="off"):
    """Re-run a worker's DETERMINISTIC round on the coordinator: build the worker's model (the trunk
    it was sent + the worker's experts the coordinator holds) and train H FROZEN-EXPERT trunk steps on
    the SAME data shard with the SAME seed/H/lr the coordinator assigned, returning the trunk delta.

    FROZEN-EXPERT trunk step (finding #1 fix): only the TRUNK is trained here; the experts are held
    fixed at the coordinator's gathered copy. The worker computes its VERIFIED trunk delta the SAME way
    (experts frozen at the last-snapshot state it shares with the coordinator), so this reproduces the
    honest trunk gradient WITHOUT the between-snapshot expert DRIFT that used to rotate the gradient and
    collapse the cosine — which deadlocked honest miners (rejected -> experts never re-gathered -> drift
    grows -> permanent rejection). Experts advance only in the snapshot Phase-2 step, gathered in lockstep
    by both sides, so the frozen reference stays identical. A cheat that skipped the work still cannot
    reproduce the trunk direction, so the cosine gate is unchanged in strength.

    (#92 follow-up) COMPRESSED WIRE. For a capable worker the CALLER passes the DEQUANTIZED trunk as
    `trunk_np` (the exact start-state it sent that worker), so the trunk half is already matched. The
    FROZEN experts, however, come from `full_state` (canonical fp32) — but that worker received them
    QUANTIZED on the last reassign/snapshot push and froze `dequant(quant(experts))`. So when `wire_mode`
    is active we dequantize the frozen experts the SAME way here: `dequant(quant(full_state[k]))` is
    bit-identical to what the worker holds (same source tensor, same deterministic codec), keeping the
    frozen-expert reference exact on both sides. `wire_mode == off` => experts loaded raw, byte-identical."""
    a = dict(arch0); a["n_experts"] = int(E)
    _wm = wire_compress.normalize_mode(wire_mode)

    def _expert_ref(t):
        """The frozen-expert tensor the WORKER holds: raw fp32 (off) or dequant(quant(fp32)) (compressed)."""
        if _wm == "off":
            return t.to(device)
        q = wire_compress.dequantize_array(
            wire_compress.quantize_array(t.detach().cpu().float().numpy(), _wm), _wm)
        return torch.from_numpy(np.ascontiguousarray(q)).to(device)
    # load_base=False: the recompute model is loaded with the worker's trunk_np (= whole dense model) just
    # below, so importing the base would be overwritten. For dense, trunk_keys() is every param.
    m = build_pool_model(a, vocab, hosted, device, load_base=False)
    sd = m.state_dict()
    with torch.no_grad():
        for k in trunk_keys(sd):
            sd[k].copy_(torch.from_numpy(np.ascontiguousarray(trunk_np[k])).to(device))
        for ee in hosted:
            for k in expert_state_keys(ee, NL):
                if k in full_state and k in sd:
                    sd[k].copy_(_expert_ref(full_state[k]))
    recv = {k: sd[k].detach().clone() for k in trunk_keys(sd)}
    for n_, p in m.named_parameters():
        p.requires_grad_(".moe.experts." not in n_)     # freeze experts; train the trunk only
    # weight_decay sourced from the SAME shared WEIGHT_DECAY env read as the worker — they MUST match
    # or the recompute trunk delta diverges from the honest worker's and the cosine/norm gate rejects it.
    opt = torch.optim.AdamW([p for n_, p in m.named_parameters() if p.requires_grad], lr=lr,
                            weight_decay=WEIGHT_DECAY)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    for _ in range(H):
        x, y = get_batch(shard_data, block, batch, device, generator=gen)
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    nt = m.state_dict()
    return {k: (nt[k] - recv[k]).detach().cpu().float().numpy() for k in trunk_keys(sd)}


def _cossim(d1, d2):
    """Cosine similarity of two flattened trunk-delta dicts (scale-invariant: a cheat can't pass by
    matching magnitude — it must match the gradient DIRECTION, i.e. actually do the work)."""
    a = np.concatenate([np.asarray(d1[k]).ravel() for k in d1])
    b = np.concatenate([np.asarray(d2[k]).ravel() for k in d1])
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb + 1e-12))


def _norm_ratio(submitted, recompute):
    """||submitted|| / ||recompute||. Cosine fixes DIRECTION but is scale-invariant, so a LAZY worker
    (e.g. 1 step instead of H) keeps the direction yet submits a tiny delta and would pass cosine.
    Requiring this ratio in a band makes the gate also enforce the AMOUNT of committed work."""
    a = np.concatenate([np.asarray(submitted[k]).ravel() for k in submitted])
    b = np.concatenate([np.asarray(recompute[k]).ravel() for k in submitted])
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a) / (nb + 1e-12))


def _serve_net_verify(sock, msg, acct, arch0, vocab, train_data, batch, device, psk):
    """(B7-2b) WORKER-AS-VERIFIER. Handle a coordinator `verify_request` for ANOTHER worker's trunk
    delta: build that worker's model from the sent inputs (its experts + the round trunk), recompute
    the FROZEN-EXPERT trunk step on the named shard (the corpus is held locally) with the public
    `_seed(round, target)`, score the SAME cosine/norm predicate, and reply with a `verify_resp`
    carrying an attestation SIGNED BY THIS NODE'S OWN key (`net_serve_verify_request`). The coordinator
    cannot forge this verdict — only this node's key produces it. Best-effort: on any error we simply
    do not reply, and the committee tolerates a missing verifier."""
    # LAZY import (doc §1c): trunk_verify_net pulls the PRIVATE trunk_committee core and the verifier duty
    # is deferred from the v1 public miner, so this module must import WITHOUT it. Importing here (only when
    # a coordinator actually asks this node to verify) is behavior-preserving; in a public v1 build the
    # module is absent, the import raises, run_worker's try/except around this call swallows it, and the
    # miner simply does not offer verification (the committee tolerates a missing verifier).
    from neurahash_torch.trunk_verify_net import (
        verify_request as net_verify_request, serve_verify_request as net_serve_verify_request)
    r, target = int(msg["round"]), str(msg["target"])
    hosted, E = list(msg["hosted"]), int(msg["n_experts"])
    NL, block = int(arch0["n_layers"]), int(arch0["block_size"])
    experts = {k: torch.from_numpy(np.ascontiguousarray(np.asarray(v))).to(device)
               for k, v in msg["experts"].items()}
    trunk_np = {k: np.asarray(v) for k, v in msg["trunk"].items()}
    s, e = msg["shard_range"]
    # use the TARGET's per-worker batch when the coordinator sent it (capacity-aware per-worker batch);
    # else fall back to this verifier's own `batch` (backward-compatible with a pre-capacity coordinator).
    vbatch = int(msg.get("batch", batch))
    # (#92) `trunk` is the target's start-state (dequantized for a capable target); `wire_mode` makes this
    # verifier freeze experts as dequant(quant(experts)) too, matching the target's view. Absent -> 'off'.
    recomp = _recompute_trunk_delta(arch0, vocab, hosted, experts, trunk_np,
                                    train_data[int(s):int(e)], _seed(r, target), int(msg["H"]),
                                    float(msg["lr"]), vbatch, block, NL, E, device,
                                    wire_mode=msg.get("wire_mode", "off"))
    env = net_verify_request(r, target, msg["delta_hash"], msg["round_beacon"])
    submitted = {k: np.asarray(v) for k, v in msg["submitted_delta"].items()}
    att = net_serve_verify_request(acct, env, submitted_delta=submitted, recompute=recomp,
                                   cos_fn=_cossim, norm_fn=_norm_ratio, verify_cos=VERIFY_COS,
                                   norm_lo=VERIFY_NORM_LO, norm_hi=VERIFY_NORM_HI)
    send_msg(sock, {"type": "verify_resp", "round": r, "target": target, "att": att}, key=psk)


def trunk_delta_hash(td):
    """Canonical content hash of a submitted trunk delta. The worker SIGNS this and the coordinator
    RECOMPUTES it from the RECEIVED delta, so the work signature commits to the EXACT delta bytes — a
    delta swapped in transit (e.g. by a shared-PSK holder) no longer matches the signature (#38)."""
    h = hashlib.sha256()
    for k in sorted(td):
        a = np.ascontiguousarray(td[k])
        h.update(k.encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(repr(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def work_commit_payload(address, round_id, beacon, delta_hash):
    """The bytes a worker SIGNS to attest 'THIS delta is MY work for round_id against this trunk' —
    bound to the address (can't be lifted to another identity), the beacon (can't be replayed to
    another round), AND the delta hash (the signature commits to the exact delta, so a valid
    signature can't be paired with a swapped or stale delta)."""
    return json.dumps({"w": "neurahash-trunk", "addr": str(address), "round": int(round_id),
                       "beacon": str(beacon), "delta": str(delta_hash)},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")
