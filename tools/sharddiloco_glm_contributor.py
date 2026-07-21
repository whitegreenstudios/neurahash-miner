#!/usr/bin/env python3
"""shardDiLoCo GLM lane -- runnable CONTRIBUTOR CLI (all-outbound), the REAL-MODEL twin of
tools/sharddiloco_contributor.py.

WHY. The 2026-07-21 two-box WAN run (tag de33f6, held-out CE 4.5400 -> 2.7888, ratio 1.0081
NON-REGRESSION PASS) proved the lane, but it carried the TOY numpy MoELM
(tools/sharddiloco_harness.build_model, Demb=16). The owner's no-toy-models directive says that is
not the bar. This file keeps the lane EXACTLY as proven -- same ContentLane, same fp16
content-addressed blobs, same HMAC-signed records, same async pointer handshake -- and swaps ONLY
the model+state layer for a REAL glm4_moe_lite (docs/research/SHARDDILOCO_GLM_WAN_PLAN.md sec 2a).
Nothing in tools/sharddiloco_contributor.py, tools/sharddiloco_coordinator.py,
tools/sharddiloco_harness.py or tools/sharddiloco_glm_expert.py is modified: the two module-level
knobs this lane needs (the pointer name and the contribution-name prefix) are overridden AT RUNTIME
from here, so the live lane's de33f6 artifacts (sharddiloco/pointer, c/rN/*) are never clobbered.

DIFFERENCES vs the toy contributor, and why (plan sec 2a/3):
  * NO lane state pull. pack_state serializes float64; one GLM slot is 75.5 MB, over content_store's
    32 MiB MAX_BODY and pointless -- every node already holds the base on disk. Instead every node
    builds the SAME base locally and the coordinator advertises a `model_root` fingerprint in the
    pointer; a mismatch is a hard DRIFT error rather than a silent divergence.
  * NO trunk pseudo-gradient. The GLM trunk is FROZEN (only per-expert LoRA trains), precedent
    tools/sharddiloco_glm_expert.py:423. The coordinator already tolerates trunk_cid=None.
  * Replication of the merge. The coordinator publishes a per-round ACCEPTED record listing the CID
    of every delta it merged; each contributor re-fetches those CIDs and applies
    `base += outer*delta` locally in the same order, so all replicas stay bit-identical to the
    coordinator's model (verified each round via the model_root fingerprint).

This module also holds the SHARED node helpers (lane names, deterministic tiny-GLM base build,
data, fingerprints) that tools/sharddiloco_glm_coordinator.py imports -- one definition, so the two
roles cannot drift apart.

Usage (tiny shakedown, plan step S3):
  C:/Python313/python.exe tools/sharddiloco_glm_contributor.py --miner miner0 --slot 0 \
      --key <hex16> --url http://127.0.0.1:8797 --token <tok> --mode tiny --slots 1:0,1:1
Usage (real GLM, plan step S4):
  ... --mode glm --shard-dir D:/hf_models/GLM-4.7-Flash-bf16_shards_100mb \
      --config-dir D:/hf_models/GLM-4.7-Flash-bf16 --piece 0 --slots 1:0,1:1 \
      --data-dir D:/glm_wan --domains code,gutenberg --device cuda --batch 4
"""
import argparse
import hashlib
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                               # noqa: E402
import sharddiloco_harness as H                                  # noqa: E402  (numpy-only, no torch)


# ==================================================================== lane naming (RUNTIME override)
# sharddiloco_harness.publish_pointer/read_pointer read the module global POINTER_NAME at CALL time,
# so assigning it here retargets the pointer without editing that file (plan sec 2b, risk 7).
GLM_POINTER_NAME = "sharddiloco/glm/pointer"
CONTRIB_PREFIX_FMT = "cg/r%d/"
ACCEPTED_NAME_FMT = "sharddiloco/glm/accepted/r%d"


def use_glm_lane_names():
    """Point sharddiloco_harness at the GLM lane's object names. Runtime assignment ONLY."""
    H.POINTER_NAME = GLM_POINTER_NAME


def contrib_name(rnd, miner):
    return CONTRIB_PREFIX_FMT % int(rnd) + str(miner)


def contrib_prefix(rnd):
    return CONTRIB_PREFIX_FMT % int(rnd)


def accepted_name(rnd):
    return ACCEPTED_NAME_FMT % int(rnd)


def _flush(*a):
    print(*a, flush=True)


def _G():
    """Lazy import of tools/sharddiloco_glm_expert (it pulls torch in through diloco_contributor).
    Kept lazy so the coordinator's default-off refusal and --help stay torch-free and instant."""
    import sharddiloco_glm_expert as G
    return G


# ============================================================================== slots / CLI plumbing
def parse_slots(s):
    """'1:0,1:1' -> [(1, 0), (1, 1)]  -- lane slot index i maps to GLM (layer, expert) pair i."""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        L, _, E = part.partition(":")
        out.append((int(L), int(E)))
    if not out:
        raise SystemExit("[glm-node] --slots must be a non-empty list like 1:0,1:1")
    return out


def add_common_args(ap):
    """Args shared by BOTH roles so coordinator and contributor cannot be configured apart."""
    ap.add_argument("--url", default=os.environ.get("NEURAHASH_CONTENT_URL", "http://127.0.0.1:8797"))
    ap.add_argument("--token", default=os.environ.get("NEURAHASH_CONTENT_TOKEN", ""))
    ap.add_argument("--mode", default=os.environ.get("NEURAHASH_GLM_MODE", "tiny"),
                    choices=("tiny", "glm"),
                    help="tiny = deterministic build_tiny_glm base (wire shakedown, plan S3); "
                         "glm = real GLM-4.7-Flash piece via piece_loader (plan S4+)")
    ap.add_argument("--slots", default=os.environ.get("NEURAHASH_GLM_SLOTS", "1:0,1:1"),
                    help="lane slots as layer:expert pairs, e.g. 1:0,1:1")
    ap.add_argument("--shard-dir", default=os.environ.get(
        "NEURAHASH_GLM_SHARD_DIR", "D:/hf_models/GLM-4.7-Flash-bf16_shards_100mb"))
    ap.add_argument("--config-dir", default=os.environ.get(
        "NEURAHASH_GLM_CONFIG_DIR", "D:/hf_models/GLM-4.7-Flash-bf16"))
    ap.add_argument("--piece", type=int, default=int(os.environ.get("NEURAHASH_GLM_PIECE", "0")),
                    help="expert piece id to keep resident; ALL its experts stay resident on every "
                         "node so contributor and coordinator route identically (plan risk 5)")
    ap.add_argument("--device", default=os.environ.get("NEURAHASH_GLM_DEVICE", "cpu"))
    ap.add_argument("--data-dir", default=os.environ.get("NEURAHASH_GLM_DATA_DIR", "D:/glm_wan"))
    ap.add_argument("--domains", default=os.environ.get("NEURAHASH_GLM_DOMAINS", "code,gutenberg"),
                    help="one corpus domain per slot (mode=glm); ids_<domain>_<split>.npy")
    ap.add_argument("--warm-steps", type=int, default=int(os.environ.get("NEURAHASH_GLM_WARM", "400")),
                    help="mode=tiny only: deterministic warm-start steps standing in for a PRETRAINED "
                         "GLM base. MUST be identical on every node (it defines the shared base)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NEURAHASH_GLM_THREADS", "4")),
                    help="torch CPU thread count -- PINNED so every node's warm-start reduction order "
                         "(and therefore the shared base) is bit-identical")
    return ap


# ================================================================= deterministic tiny-GLM base + data
# mode=tiny stands in for the 5 GB GLM load: a REAL Glm4MoeLiteForCausalLM (real sigmoid-top-k
# router, real fused expert MLP) built from a fixed seed and warm-started by a fully deterministic
# routine, so EVERY node reaches bit-identical weights without shipping them over the lane. The
# model_root fingerprint in the pointer proves that each round.
TINY = dict(vocab=24, seq=16, hidden=64, inter=128, moe_inter=48, layers=3, n_experts=4, topk=2,
            seed=1, warm_n=3000, train_n=2000, val_n=160, probe_n=256, heldout_n=256)


def _tiny_transition():
    G = _G()
    return G.make_transition(TINY["vocab"], seed=7, peak=12)


def tiny_ids(split, slot=0):
    """Deterministic, DISJOINT sample sets from one fixed Markov source -- the tiny-mode stand-in for
    D:/glm_wan/ids_<domain>_<split>.npy. Split semantics are identical to the real data:
    train = miner trains on it, val = miner's own save-best, probe = coordinator's SECRET gate pool,
    heldout = the reported goal metric (touched by nothing else)."""
    G = _G()
    P = _tiny_transition()
    V, T = TINY["vocab"], TINY["seq"]
    spec = {"warm": (TINY["warm_n"], 100), "warmval": (160, 555),
            "train": (TINY["train_n"], 2000 + 10 * int(slot)),
            "val": (TINY["val_n"], 2001 + 10 * int(slot)),
            "probe": (TINY["probe_n"], 2002 + 10 * int(slot)),
            "heldout": (TINY["heldout_n"], 90001)}
    if split not in spec:
        raise SystemExit("[glm-node] unknown split %r" % (split,))
    n, seed = spec[split]
    return G.markov_dataset(V, T, n, seed=seed, transition=P)


def _warm_start_tiny(model, steps, log=None):
    """Deterministic warm-start: a fixed data order + fixed seeds + a pinned thread count, so this
    function is a pure function of `steps`. It stands in for the fact that GLM-4.7-Flash is already
    PRETRAINED (its experts carry signal); after it the base is FROZEN and only per-expert LoRA
    trains, exactly as in the real run."""
    import torch
    G = _G()
    train = tiny_ids("warm")
    val = tiny_ids("warmval")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02)
    best, best_sd = 1e9, None
    for step in range(1, int(steps) + 1):
        idx = np.random.default_rng(step).integers(0, len(train), size=48)
        ids = torch.as_tensor(train[idx])
        model.train()
        out = model(input_ids=ids, labels=ids)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if step % 20 == 0:
            h = G.heldout_ce(model, val)
            if h < best:
                best, best_sd = h, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    if log:
        log("[glm-node] tiny warm-start done: %d steps, best warm-val CE=%.5f" % (steps, best))
    return model


# GLM-mode resident footprint, GiB -- MEASURED 2026-07-21 on the 5090 with one piece resident
# (scratchpad/glm_measure_footprint.py), NOT estimated. The plan's 6.10 estimate was wrong: it
# budgeted 0.35 GiB of activations and the real backward pass across 47 layers costs ~3.8 GiB even
# at batch=2. An earlier run OOM'd at a 9.15 GiB cap because of exactly this.
#   after LOAD                      peak 5.203
#   after EVAL chunk=8              peak 5.542   <- the coordinator only ever evaluates
#   TRAIN batch=2, trunk NOT frozen peak 9.339   <- the old number; ~4 GiB was discarded trunk grads
#   REAL CONTRIBUTOR PATH, frozen   peak 5.557   <- after freezing the trunk in
#                                                   train_glm_expert_contribution
# +15% margin. That 40% cut is what lets an 8 GB consumer card (4060: ~6.9 GiB usable) train GLM at
# all. Re-measure with scratchpad/glm_measure_footprint.py before raising either number.
GLM_CONTRIB_NEED_GIB = 6.40           # trains (forward + backward, trunk frozen)
GLM_COORD_NEED_GIB = 6.40             # evaluates only -- never calls .backward()
GLM_NEED_GIB = GLM_CONTRIB_NEED_GIB   # default = the larger, so a new caller cannot under-book
TINY_NEED_GIB = 0.5
DEFAULT_HEADROOM = 0.90          # of CURRENTLY FREE VRAM, not of the card
RUNAWAY_SLACK = 1.5              # ... and never more than 1.5x this role's measured need


def apply_vram_guard(device, need_gib, log=None):
    """Hard per-process VRAM ceiling + a refuse-to-start preflight. MUST be called before any model
    is materialised on CUDA.

    WHY THIS EXISTS HERE AND NOT VIA sharded_pool_node.apply_vram_cap: that function is wired only
    into run_worker()/__main__ of sharded_pool_node.py and is a documented NO-OP for standalone
    tools (memory vram-cap-live-verified, caveat 0 -- "the DiLoCo P0 smoke OOM'd twice while
    'capped'"). Any tool that puts weights on a GPU has to bring its own ceiling.

    WHY THE DEFAULT IS A FRACTION OF *FREE*, NOT OF TOTAL: the cap is PER-PROCESS (same memory,
    caveat 1) -- N processes each capped at 80% of a 32 GB card still oversubscribe it, and once
    physical is exhausted the WDDM driver silently spills to shared system RAM and thrashes the
    machine to death instead of raising OOM. Sizing from CURRENTLY FREE memory makes concurrent
    launches self-limiting: each new process sees what its predecessors already took. It also
    leaves the live pool coordinator's allocation alone, because that shows up as not-free.

    MEASURED 2026-07-21: three uncapped GLM processes launched next to the live pool coordinator
    exhausted a 32 GB 5090, spilled to shared RAM, and CRASHED the host. That is the failure this
    prevents.

    Overrides: NEURAHASH_VRAM_CAP_GB (absolute GiB) or NEURAHASH_VRAM_CAP_FRAC (fraction of the
    card) -- same variable names as the pool's knob so there is one vocabulary, not two.
    """
    import torch
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    idx = torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(idx)
    free_gib, total_gib = free_b / 2 ** 30, total_b / 2 ** 30

    cap_gb, cap_frac = os.environ.get("NEURAHASH_VRAM_CAP_GB"), os.environ.get("NEURAHASH_VRAM_CAP_FRAC")
    if cap_gb:
        cap_gib, how = float(cap_gb), "NEURAHASH_VRAM_CAP_GB"
    elif cap_frac:
        cap_gib, how = float(cap_frac) * total_gib, "NEURAHASH_VRAM_CAP_FRAC"
    else:
        # Two ceilings, take the lower. The free-fraction keeps concurrent launches from
        # oversubscribing the card; the need-multiple keeps ONE buggy process (a runaway eval, an
        # unchunked log_softmax) from eating a whole card that its siblings still need.
        cap_gib = min(free_gib * DEFAULT_HEADROOM, need_gib * RUNAWAY_SLACK)
        how = "min(%.0f%% of free, %.1fx need)" % (DEFAULT_HEADROOM * 100, RUNAWAY_SLACK)

    if cap_gib < need_gib:
        raise SystemExit(
            "[vram-guard] REFUSING TO START: this role needs ~%.2f GiB but the cap is %.2f GiB "
            "(%s; card %.2f GiB total, %.2f GiB free). Free VRAM first (stop other GPU processes) "
            "or lower the footprint -- do NOT raise the cap past free memory, that is what spills "
            "to shared system RAM and hangs the box." % (need_gib, cap_gib, how, total_gib, free_gib))

    frac = min(1.0, cap_gib / total_gib)
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    msg = ("[vram-guard] capped to %.2f GiB (%.1f%% of the %.2f GiB card; %.2f GiB was free; %s). "
           "OOMs at the cap, never spills to sysmem. Need ~%.2f GiB."
           % (cap_gib, frac * 100, total_gib, free_gib, how, need_gib))
    if log:
        log(msg)
    else:
        print(msg, flush=True)
    return cap_gib


def build_node_model(args, log=None, need_gib=None):
    """Build the node's base model. BOTH roles call this, with the same args, so both hold the same
    weights and (critically, plan risk 5) the same resident expert set -- if a contributor held only
    its own expert, piece_loader.py:381-385 would mask the other 63 to -inf and it would optimize a
    CE the coordinator does not gate on. Returns (model, cfg, seq_len)."""
    import torch
    G = _G()
    torch.set_num_threads(max(1, int(args.threads)))
    if need_gib is None:
        need_gib = GLM_NEED_GIB if args.mode == "glm" else TINY_NEED_GIB
    apply_vram_guard(args.device, need_gib, log=log)
    if args.mode == "tiny":
        model, cfg = G.build_tiny_glm(seed=TINY["seed"], vocab=TINY["vocab"], hidden=TINY["hidden"],
                                      inter=TINY["inter"], moe_inter=TINY["moe_inter"],
                                      layers=TINY["layers"], n_experts=TINY["n_experts"],
                                      topk=TINY["topk"])
        _warm_start_tiny(model, args.warm_steps, log=log)
        return model, cfg, TINY["seq"]
    # ---- real GLM: one piece resident, trunk frozen (plan sec 1: 1 MoE layer x 1.125 GiB slab) ----
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    loader_dir = os.path.join("D:/glm_loader/repo", "tools")
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)
    import piece_loader                                          # noqa: E402 (no neurahash import)
    model, summ = piece_loader.build_partial_model(
        args.shard_dir, [int(args.piece)], device=args.device, dtype=torch.bfloat16,
        config_dir=args.config_dir, strip_mtp=True)
    if int(summ.get("meta_params_left", 0)) != 0:
        raise SystemExit("[glm-node] FATAL: %d meta params left after load (incomplete piece)"
                         % summ["meta_params_left"])
    model.eval()
    if log:
        log("[glm-node] GLM piece %d resident: %s" % (args.piece, summ))
    seq = int(np.load(_ids_path(args, 0, "heldout"), mmap_mode="r").shape[1])
    return model, model.config, seq


def _ids_path(args, slot, split):
    doms = [d.strip() for d in str(args.domains).split(",") if d.strip()]
    dom = doms[int(slot) % len(doms)]
    return os.path.join(args.data_dir, "ids_%s_%s.npy" % (dom, split))


def node_ids(args, slot, split):
    """Split loader that both roles share: real .npy in mode=glm, deterministic Markov in mode=tiny."""
    if args.mode == "tiny":
        return tiny_ids(split, slot=slot)
    return np.load(_ids_path(args, slot, split))


# ================================================================================ base fingerprints
def model_root(host):
    """sha256 over the canonical float32 weights of every lane slot -- the pointer's opaque
    `state_cid` (plan sec 2b). Changes iff a merge moved a slot, so a contributor whose local replay
    of the accepted deltas diverged from the coordinator detects it IMMEDIATELY instead of training
    against a phantom base."""
    h = hashlib.sha256()
    for i in range(len(host.slots)):
        d = host.read_slot(i)
        L, E = host.slots[i]
        h.update(("L%dE%d|" % (L, E)).encode())
        for k in sorted(d):
            h.update(k.encode())
            h.update(np.ascontiguousarray(d[k], dtype=np.float32).tobytes())
    return h.hexdigest()


def base_digest(model, max_numel=50_000_000):
    """sha256 over ALL parameters -- proves two nodes built the SAME base, not just the same slots.
    Skipped (returns 'skipped:<numel>') for a real GLM where hashing 4 GiB every start is wasteful;
    there the shared base is guaranteed by both nodes reading the same on-disk shard files."""
    import torch
    tot = int(sum(p.numel() for p in model.parameters()))
    if tot > max_numel:
        return "skipped:%d" % tot
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in sorted(model.state_dict().items()):
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def apply_accepted(host, lane, record, log=None):
    """Replay the coordinator's merge locally: for each accepted delta, in the coordinator's order,
    base += outer*delta. The delta is re-FETCHED BY CID from the lane, so the contributor applies the
    exact fp16-roundtripped bytes the coordinator gated on (bit-identical to
    diloco_merge.apply_delta_gated:484)."""
    n = 0
    for item in record.get("accepted", []):
        slot = int(item["slot"])
        outer = float(item.get("outer", 0.7))
        d = lane.get_delta(item["cid"])
        cur = host.read_slot(slot)
        host.write_slot(slot, {k: cur[k] + outer * d[k] for k in cur if k in d})
        n += 1
    if log:
        log("[glm-node] applied %d accepted delta(s) for round %s" % (n, record.get("round")))
    return n


def fetch_accepted(lane, rnd, timeout=60.0, poll=0.25):
    """Read the coordinator's per-round ACCEPTED record (named object), waiting for it to appear."""
    name = accepted_name(rnd)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            man = lane.manifest()
            if name in man:
                return lane.get_json(man[name]["sha256"])
        except Exception:                                        # noqa: BLE001
            pass
        time.sleep(poll)
    return None


# ============================================================================== contributor CLI
def _resolve_key(args):
    if args.key:
        return bytes.fromhex(args.key)
    if args.key_file and os.path.exists(args.key_file):
        return bytes.fromhex(open(args.key_file, "r", encoding="utf-8").read().strip())
    env = os.environ.get("NEURAHASH_SD_KEY")
    if env:
        return bytes.fromhex(env)
    raise SystemExit("[glm-contrib] no signing key: pass --key <hex16> / --key-file / NEURAHASH_SD_KEY")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--miner", default=os.environ.get("NEURAHASH_SD_MINER", "miner0"))
    ap.add_argument("--slot", type=int, default=int(os.environ.get("NEURAHASH_SD_EXPERT", "0")),
                    help="lane slot index this miner OWNS (maps to --slots[slot])")
    ap.add_argument("--key", default=None)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--max-rounds", type=int, default=int(os.environ.get("NEURAHASH_SD_MAX_ROUNDS", "1000")))
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--wait-up", type=float, default=300.0,
                    help="seconds to wait for the coordinator pointer (a real GLM load is minutes)")
    ap.add_argument("--round-wait", type=float, default=300.0)
    ap.add_argument("--inner", type=int, default=int(os.environ.get("NEURAHASH_GLM_INNER", "60")),
                    help="H local LoRA steps per outer round (the anti-flap core: zero cross-miner comm)")
    ap.add_argument("--lora-r", type=int, default=int(os.environ.get("NEURAHASH_GLM_R", "16")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("NEURAHASH_GLM_LR", "3e-3")))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("NEURAHASH_GLM_BATCH", "16")),
                    help="B=4 on the 4060 (plan sec 1: vocab 154880 log_softmax is ~1.2 GiB at B=48)")
    ap.add_argument("--garbage", action="store_true",
                    help="ADVERSARIAL control: publish a correctly-SIGNED but harmful random delta "
                         "(sharddiloco_glm_expert.garbage_delta) that the secret-probe gate must REJECT")
    add_common_args(ap)
    args = ap.parse_args(argv)

    use_glm_lane_names()
    key = _resolve_key(args)
    slots = parse_slots(args.slots)
    i = int(args.slot)
    if not (0 <= i < len(slots)):
        raise SystemExit("[glm-contrib] --slot %d out of range for --slots %s" % (i, args.slots))
    L, E = slots[i]
    miner = args.miner
    lane = H.ContentLane(args.url, args.token)
    _flush("[glm-contrib %s] UP owns slot %d = GLM (L%d,E%d) | mode=%s lane=%s (all-outbound)"
           % (miner, i, L, E, args.mode, args.url))

    G = _G()
    model, cfg, seq = build_node_model(args, log=_flush)
    host = G.GlmExpertLaneHost(model, cfg, slots)
    _flush("[glm-contrib %s] base ready: model_root=%s.. base_digest=%s.. seq=%d"
           % (miner, model_root(host)[:12], base_digest(model)[:12], seq))

    train_ids = node_ids(args, i, "train")
    val_ids = node_ids(args, i, "val")

    # wait for the coordinator's first pointer
    ptr, t0 = None, time.time()
    while time.time() - t0 < args.wait_up:
        try:
            ptr = lane.read_pointer()
        except Exception:                                        # noqa: BLE001
            ptr = None
        if ptr is not None:
            break
        time.sleep(args.poll)
    if ptr is None:
        _flush("[glm-contrib %s] FATAL: no coordinator pointer at %s after %.0fs"
               % (miner, args.url, args.wait_up))
        return 4

    done_last = -1
    applied = -1            # last round whose ACCEPTED record has been replayed locally
    rounds_done = 0
    while rounds_done < args.max_rounds:
        try:
            ptr = lane.read_pointer()
        except Exception:                                        # noqa: BLE001
            time.sleep(args.poll)
            continue
        if ptr is None:
            time.sleep(args.poll)
            continue
        if ptr.get("done"):
            _flush("[glm-contrib %s] coordinator signalled DONE; exiting after %d contributions"
                   % (miner, rounds_done))
            return 0
        rnd = int(ptr["round"])
        if rnd <= done_last:
            time.sleep(args.poll)
            continue

        # ---- replay every merge that happened since our last round, so our base == coordinator's ----
        for r in range(applied + 1, rnd):
            rec = fetch_accepted(lane, r, timeout=args.round_wait, poll=args.poll)
            if rec is None:
                _flush("[glm-contrib %s] FATAL: accepted record for round %d never appeared" % (miner, r))
                return 6
            apply_accepted(host, lane, rec)
            applied = r
        root = model_root(host)
        if not args.garbage and ptr.get("state_cid") and ptr["state_cid"] != root:
            _flush("[glm-contrib %s] FATAL DRIFT at round %d: local model_root=%s.. but coordinator "
                   "advertises %s.. (replicas diverged -- refusing to train on a phantom base)"
                   % (miner, rnd, root[:12], str(ptr["state_cid"])[:12]))
            return 7

        # ---- train H local LoRA steps on my slot, with ZERO cross-miner comm ----
        t_tr = time.time()
        if args.garbage:
            delta = G.garbage_delta(host.read_slot(i), scale=3.0, seed=1234 + rnd)
            train_flops, best_val = 1.0, float("nan")
        else:
            c = G.train_glm_expert_contribution(
                model, cfg, L, E, train_ids, val_ids, H=args.inner, r=args.lora_r, lr=args.lr,
                batch=args.batch, seed=rnd * 100 + i)
            delta, train_flops, best_val = c["delta"], c["train_flops"], c["best_val_ce"]

        # ---- publish the fp16 content-addressed delta + a signed record (D1/D2), trunk FROZEN ----
        ecid = lane.put_delta(delta)
        sig = H.sign(key, ecid, rnd, miner)
        delta_bytes = int(len(H.pack_arrays(delta, np.float16)))
        record = dict(miner=miner, expert=int(i), layer=int(L), glm_expert=int(E), base_round=int(rnd),
                      expert_cid=ecid, trunk_cid=None, sig=sig, train_flops=float(train_flops),
                      trunk_bytes=0, delta_bytes=delta_bytes, base_root=root)
        rname = contrib_name(rnd, miner)
        rec_cid = lane.put_json_named(rname, record)
        done_last = rnd
        rounds_done += 1
        _flush("[glm-contrib %s] round %d: %s slot %d (L%d,E%d) in %.1fs, best_val_ce=%.5f, "
               "published expert_cid=%s.. delta=%dB base_root=%s.. flops=%.3e"
               % (miner, rnd, "GARBAGE (adversarial control)" if args.garbage else
                  "trained %d LoRA steps on" % args.inner, i, L, E, time.time() - t_tr, best_val,
                  ecid[:12], delta_bytes, root[:12], train_flops))

        # Keep the record ADVERTISED until the coordinator advances past this round -- content_store's
        # names.json is a read-modify-write shared file, so concurrent named PUTs can drop a name.
        # Re-asserting it whenever it falls out of the manifest self-heals that race (verbatim policy
        # from tools/sharddiloco_contributor.py:148-167).
        t_pub = time.time()
        while time.time() - t_pub < args.round_wait:
            try:
                ptr2 = lane.read_pointer()
            except Exception:                                    # noqa: BLE001
                ptr2 = None
            if ptr2 is not None and (ptr2.get("done") or int(ptr2.get("round", rnd)) > rnd):
                break
            try:
                man = lane.manifest()
                if man.get(rname, {}).get("sha256") != rec_cid:
                    lane.put_json_named(rname, record)
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(args.poll)

    _flush("[glm-contrib %s] hit max-rounds=%d; exiting" % (miner, args.max_rounds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
