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
import re
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                               # noqa: E402
import sharddiloco_harness as H                                  # noqa: E402  (numpy-only, no torch)
from neurahash import diloco_merge as dm                         # noqa: E402  (numpy-only, no torch:
# the W1 async primitives sd_pointer_decode / token telemetry. diloco_merge pulls ONLY numpy +
# delta_codec, so this keeps the --help / coordinator-default-off paths torch-free and instant.)


# ==================================================================== lane naming (RUNTIME override)
# sharddiloco_harness.publish_pointer/read_pointer read the module global POINTER_NAME at CALL time,
# so assigning it here retargets the pointer without editing that file (plan sec 2b, risk 7).
GLM_POINTER_NAME = "sharddiloco/glm/pointer"
CONTRIB_PREFIX_FMT = "cg/r%d/"
ACCEPTED_NAME_FMT = "sharddiloco/glm/accepted/r%d"

# ---- corpus-over-WAN auto sync (W6) --------------------------------------------------------------
# The coordinator advertises DATA_RECORD_NAME (same trust surface + name shape as GLM_POINTER_NAME);
# a contributor fetches ONLY the miner-facing splits it matches -- ids_<domain>_train/val.npy or the
# data manifest. The SECRET probe/heldout splits (sharddiloco_glm_coordinator.py:573) must never
# match, so even a forged record cannot make miner code pull them (F1 defense-in-depth).
DATA_RECORD_NAME = "sharddiloco/glm/data"
DATA_MANIFEST_NAME = "data_manifest.json"
RC_DATA_UNVERIFIED = 9              # exit code: a record file was neither locally-valid nor fetched+verified
_ALLOWED_DATA_RE = re.compile(r"ids_[A-Za-z0-9-]+_(?:train|val)\.npy\Z")


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
    # MINER-FACING data dir: train + val ONLY. The coordinator's SECRET probe/heldout live in a
    # separate coordinator-only dir (tools/glm_wan_prep_data.py writes <out>/miner vs <out>/coord),
    # so this default resolves to a dir a miner can hold and even ship without leaking the gate (F1).
    ap.add_argument("--data-dir", default=os.environ.get("NEURAHASH_GLM_DATA_DIR", "D:/glm_wan/miner"))
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


def _maybe_start_vram_manager(args, need_gib, log=None):
    """OPT-IN unified-VRAM path (env NEURAHASH_VRAM_MANAGER). Returns the started VramManager, or
    None when the flag is OFF -- in which case build_node_model runs the EXISTING apply_vram_guard
    exactly as today (byte-identical; a live soak is unaffected). The flag is checked BEFORE any
    import so the OFF path executes only one env lookup and changes nothing.

    When ON, ONE VramManager (neurahash/vram_manager.py) becomes the single source of truth: its
    apply_cap() replaces the static guard's ceiling (still sized from live FREE VRAM, still honouring
    the same NEURAHASH_VRAM_CAP_GB/_FRAC knobs), and a daemon thread runs the ~20s adaptive loop that
    every tick re-caps, re-advertises the sustainable capacity (on_report), and resizes the resident
    footprint (on_resize) -- all from ONE detect() so the cap, the reported capacity, and the trained
    footprint can never disagree.

    on_resize DISPOSITION -- SEAM, not a full rebuild: evicting/loading resident MoE layers on the
    live GlmExpertLaneHost to match new_units is a larger refactor than this opt-in wiring should
    carry, so on_resize logs the new target and frees the CUDA cache (empty_cache) -- a shrink then
    actually returns VRAM to the owner -- while the real hosted-layer rebuild is left to a follow-up
    (PLAN_CHANGE: resident-layer rebuild seam -- GlmExpertLaneHost needs an add/drop-layer API and
    the coordinator must accept a mid-session capacity change before the footprint can truly shrink)."""
    if (os.environ.get("NEURAHASH_VRAM_MANAGER", "") or "").strip().lower() not in (
            "1", "true", "yes", "on", "y"):
        return None                       # DEFAULT OFF -> caller runs apply_vram_guard unchanged
    from neurahash.vram_manager import VramManager
    _log = log or (lambda m: print(m, flush=True))
    base_gib = float(os.environ.get("NEURAHASH_VRAM_TUNE_BASE_GIB", "4.0"))     # GLM trunk footprint
    per_unit = float(os.environ.get("NEURAHASH_VRAM_TUNE_PER_UNIT_GIB", "1.125"))  # per resident layer
    max_units = int(os.environ.get("NEURAHASH_VRAM_AUTOTUNE_MAX_UNITS", "16"))
    mgr = VramManager.from_env(args.device, base_gib=base_gib, per_unit_gib=per_unit,
                               max_units=max_units)
    if mgr is None:                       # flag truthy but from_env declined -> stay on the guard
        return None
    mgr.apply_cap()                       # the GUARD, unified: hard ceiling from live free VRAM
    _log("[vram-manager] ON (unified guard+capacity+tuner): single source of truth = live free VRAM; "
         "need ~%.2f GiB, base %.2f GiB + %.3f GiB/unit, max %d units" % (need_gib, base_gib, per_unit, max_units))

    def _on_report(cap_units):
        _log("[vram-manager] re-advertising sustainable capacity = %d resident units (live-free)" % cap_units)

    def _on_resize(old_units, new_units):
        _log("[vram-manager] resize %d -> %d resident units (SEAM: real layer rebuild is a PLAN_CHANGE)"
             % (old_units, new_units))
        try:
            import torch
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()  # a shrink actually returns the freed VRAM to the owner
        except Exception:
            pass

    import threading
    stop = threading.Event()
    th = threading.Thread(target=mgr.run,
                          kwargs=dict(on_resize=_on_resize, on_report=_on_report, stop_event=stop),
                          name="vram-manager", daemon=True)
    th.start()
    mgr._loop_stop, mgr._loop_thread = stop, th     # keep the loop refs alive on the returned manager
    return mgr


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
    # OPT-IN unified VRAM manager (NEURAHASH_VRAM_MANAGER). OFF (default) -> None -> the EXISTING
    # apply_vram_guard runs exactly as today (byte-identical). ON -> the manager owns the cap +
    # the advertised capacity + the ~20s adaptive loop, all off one live-free source of truth.
    if _maybe_start_vram_manager(args, need_gib, log=log) is None:
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
    # piece_loader now ships BESIDE this file, so a normal import works anywhere. It used to be
    # imported from a hardcoded "D:/glm_loader/repo/tools" -- a path that exists only on one
    # developer machine, which meant NO stranger could run GLM shardDiLoCo at all. That went
    # unnoticed because every prior GLM run happened on the box where the path existed; a second
    # physical node surfaced it instantly as ModuleNotFoundError. NEURAHASH_GLM_LOADER_DIR remains
    # as an escape hatch for an out-of-tree loader.
    extra = os.environ.get("NEURAHASH_GLM_LOADER_DIR")
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        import piece_loader                                      # noqa: E402 (no neurahash import)
    except ImportError as ex:                                    # pragma: no cover - env problem
        raise SystemExit(
            "cannot import piece_loader (%s). It should ship next to this file in tools/; if you "
            "keep it elsewhere, point NEURAHASH_GLM_LOADER_DIR at that directory." % ex)
    model, summ = piece_loader.build_partial_model(
        args.shard_dir, [int(args.piece)], device=args.device, dtype=torch.bfloat16,
        config_dir=args.config_dir, strip_mtp=True)
    if int(summ.get("meta_params_left", 0)) != 0:
        raise SystemExit("[glm-node] FATAL: %d meta params left after load (incomplete piece)"
                         % summ["meta_params_left"])
    model.eval()
    if log:
        log("[glm-node] GLM piece %d resident: %s" % (args.piece, summ))
    seq = _infer_seq(args)                      # from a split THIS role holds (never the secret one)
    return model, model.config, seq


def _ids_path(args, slot, split, base=None):
    """Path to a split's id file. `base` overrides args.data_dir -- the coordinator passes its
    coordinator-only dir (args.coord_data_dir) for the secret probe/heldout splits (F1), so those
    files are read from a dir that is never present on a miner box."""
    doms = [d.strip() for d in str(args.domains).split(",") if d.strip()]
    dom = doms[int(slot) % len(doms)]
    root = base if base is not None else args.data_dir
    return os.path.join(root, "ids_%s_%s.npy" % (dom, split))


def node_ids(args, slot, split):
    """MINER-FACING split loader both roles share for train/val: real .npy in mode=glm (from the
    miner-facing --data-dir), deterministic Markov in mode=tiny. The coordinator's SECRET splits
    (probe/heldout) go through coord_secret_ids instead so they are never sought in the miner dir."""
    if args.mode == "tiny":
        return tiny_ids(split, slot=slot)
    return np.load(_ids_path(args, slot, split))


def coord_secret_ids(args, slot, split):
    """COORDINATOR-ONLY split loader for probe/heldout (F1). In mode=glm these live in the
    coordinator-only dir (args.coord_data_dir, default <out>/coord), which is NEVER shipped to a
    miner; in mode=tiny they are the deterministic Markov draw (no files, no secrecy risk on a
    single box). Only the coordinator process defines coord_data_dir and calls this."""
    if args.mode == "tiny":
        return tiny_ids(split, slot=slot)
    base = getattr(args, "coord_data_dir", None)
    return np.load(_ids_path(args, slot, split, base=base))


def _infer_seq(args):
    """Sequence length from whatever split this role actually holds. The miner-facing 'val' is
    present on every box; fall back to the coordinator-only 'heldout' for a coordinator box that
    kept only <out>/coord. Avoids reading a SECRET split on a miner box (F1: the old code read
    'heldout' from --data-dir, which no longer contains it)."""
    cands = [_ids_path(args, 0, "val")]
    coord = getattr(args, "coord_data_dir", None)
    if coord:
        cands.append(_ids_path(args, 0, "heldout", base=coord))
    for p in cands:
        if os.path.exists(p):
            return int(np.load(p, mmap_mode="r").shape[1])
    raise SystemExit("[glm-node] cannot infer seq length: none of these split files exist: %s" % cands)


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


def apply_accepted(host, lane, record, log=None, ce_fn=None, tol=None, rejected=None, own_slot=None):
    """Replay the coordinator's merge locally: for each accepted delta, in the coordinator's order,
    base += outer*delta. The delta is re-FETCHED BY CID from the lane, so the contributor applies the
    exact fp16-roundtripped bytes the coordinator gated on (bit-identical to
    diloco_merge.apply_delta_gated:484).

    LOCAL RE-GATE (F2 defense-in-depth). The round pointer and this ACCEPTED record ride an UNSIGNED
    lane whose PUT token is a shared public demo token, so a malicious miner can forge an accepted
    record + a matching state_cid and push an ungated delta into every replica. There is no pinned
    coordinator key to verify against yet (that needs an owner key decision -- see the module notes),
    so instead of trusting the coordinator blindly, each fetched delta is RE-GATED on a LOCAL held-out
    split before it is folded in:
      * ce_fn : optional callable ce_fn(host) -> float = the model's held-out CE right now. The live
                contributor passes one closing over its OWN val split (the coordinator's SECRET
                probe/heldout never ship to a miner box, F1, so 'local held-out' here is the miner's
                val). A delta is folded only if it does NOT raise that CE by more than `tol`; a delta
                that regresses is UNFOLDED (slot restored exactly) and appended to `rejected`.
      * tol   : absolute CE regression allowed per delta. None -> a deliberately LOOSE floor
                max(0.05*|base_ce|, 0.05) -- this catches poisoning (a garbage delta moves held-out
                CE by >>5%), not the marginal interference the coordinator's own merge-gate handles.
      * own_slot : the lane slot index this contributor is ASSIGNED to TRAIN (its single domain).
                The local re-gate is a valid signal ONLY for this slot; an accepted delta for any
                OTHER slot is CROSS-DOMAIN and is folded UNCONDITIONALLY on the coordinator's signed
                accept. This node holds the FULL resident expert set but trains only its own domain,
                so it provably cannot judge another domain on its own val -- folding slot 1's
                gutenberg delta worsens a slot-0 code node's code val, a FALSE positive that used to
                self-abort the whole replica (rc 8). None -> legacy re-gate-EVERY-slot behaviour.
    When ce_fn is None the replay is UNCONDITIONAL and bit-identical to the coordinator's merge (the
    model_root replication invariant the pointer asserts each round, and what the unit test checks).
    Returns the count of deltas actually FOLDED (a rejected delta is not counted)."""
    n = 0
    own = None if own_slot is None else int(own_slot)
    for item in record.get("accepted", []):
        slot = int(item["slot"])
        outer = float(item.get("outer", 0.7))
        d = lane.get_delta(item["cid"])
        # Same wire-agnostic materialisation the coordinator does. This replay MUST reproduce the
        # coordinator's merge bit-for-bit or replicas silently diverge (that is what model_root
        # catches), so both sides have to reconstruct the dense delta from the identical bytes the
        # same way -- fetched by CID, never recomputed from local factors.
        if _G().is_lora_payload(d):
            d = _G().materialize_from_lora(d)
        cur = host.read_slot(slot)
        # Shape guard (defense-in-depth): a delta whose keys/shapes do not match the slot cannot be
        # folded -- skip it rather than broadcast-corrupt the weights. For a legit accepted delta all
        # three keys match (the coordinator shape-gated it, F8), so this never fires on the happy path.
        if not all(k in d and np.shape(d[k]) == np.shape(cur[k]) for k in cur):
            if rejected is not None:
                rejected.append(dict(item, reason="shape-mismatch"))
            if log:
                log("[glm-node] REJECTED accepted delta for slot %d: shape mismatch vs resident" % slot)
            continue
        # F2 local re-gate applies ONLY to the delta for THIS node's OWN trained slot, where its
        # single-domain val is a valid signal. A cross-domain accepted delta (any OTHER slot) is
        # folded UNCONDITIONALLY on the coordinator's signed accept: this node trains only its own
        # domain and provably cannot judge another domain on its own val -- re-gating slot 1's
        # gutenberg delta on a slot-0 code node's code val is a FALSE positive that self-aborted the
        # replica (rc 8). base_ce is measured fresh right before the own-slot fold (i.e. AFTER any
        # cross-domain folds this round), so the check reflects only the own delta's effect.
        regate = ce_fn is not None and (own is None or slot == own)
        base_ce = ce_fn(host) if regate else None
        host.write_slot(slot, {k: cur[k] + outer * d[k] for k in cur})
        if regate:
            new_ce = ce_fn(host)
            allow = tol if tol is not None else max(0.05 * abs(base_ce), 0.05)
            if new_ce > base_ce + allow:
                host.write_slot(slot, cur)                   # UNFOLD: restore the slot exactly
                if rejected is not None:
                    rejected.append(dict(item, base_ce=float(base_ce), new_ce=float(new_ce)))
                if log:
                    log("[glm-node] REJECTED accepted delta for slot %d: local held-out CE "
                        "%.5f -> %.5f (> +%.5f) -- forged/poisoned accepted record?"
                        % (slot, base_ce, new_ce, allow))
                continue
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


# ============================================================ corpus-over-WAN auto sync (W6, DOWNLOAD)
def _default_urlopen(url, timeout):
    """The real network opener behind data_http_get; injected out in tests so the streaming/ceiling logic
    runs with a fake chunked response and ZERO network."""
    return urllib.request.urlopen(url, timeout=timeout)


def _response_content_length(r):
    """Best-effort integer Content-Length from a urllib response OR a test stand-in (getheader/headers);
    None when the header is absent or non-integer. Pure read, no consumption of the body."""
    val = None
    geth = getattr(r, "getheader", None)
    if callable(geth):
        val = geth("Content-Length")
    if val is None:
        hdrs = getattr(r, "headers", None)
        if hdrs is not None:
            try:
                val = hdrs.get("Content-Length")
            except Exception:                                    # noqa: BLE001
                val = None
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def data_http_get(url, timeout=60, expected_size=None, dest_path=None, opener=_default_urlopen):
    """F2: fetch a content object over plain HTTP(S), STREAMING it in 1 MiB chunks straight to `dest_path`
    while hashing incrementally -- so a miner never buffers an arbitrarily large body in RAM. The data
    record rides the UNSIGNED shared-token lane, so a forged record could name a giant body; a hard
    ceiling of expected_size + 65536 slack bounds the transfer: a Content-Length header over the ceiling
    is rejected BEFORE any body byte is read, and an actual over-read aborts MID-STREAM (leaving a partial
    temp file the caller cleans up). `expected_size` comes from the (untrusted) record's files[name]['size'];
    None disables the ceiling. Returns (sha256_hex, n_bytes_written). Injectable at TWO levels for zero-
    network tests: glm_data_autosync injects this whole function, and this function injects `opener`
    (so the ceiling logic itself is exercised through a fake chunked response). Any HTTP/URL error
    propagates so the caller fails this seed and tries the next."""
    if dest_path is None:
        raise ValueError("data_http_get streams to a temp file; dest_path is required")
    ceiling = None if expected_size is None else int(expected_size) + 65536
    h = hashlib.sha256()
    n = 0
    with opener(url, timeout) as r:
        if ceiling is not None:
            clen = _response_content_length(r)
            if clen is not None and clen > ceiling:
                raise ValueError("Content-Length %d over ceiling %d for %s (rejected before body read)"
                                 % (clen, ceiling, url))
        with open(dest_path, "wb") as f:
            while True:
                blk = r.read(1 << 20)
                if not blk:
                    break
                n += len(blk)
                if ceiling is not None and n > ceiling:
                    raise ValueError("body over ceiling %d bytes for %s (read %d so far, aborting mid-stream)"
                                     % (ceiling, url, n))
                h.update(blk)
                f.write(blk)
    return h.hexdigest(), n


def _sha256_stream(path, chunk=1 << 20):
    """sha256 of a local file, streamed in 1 MiB chunks so a ~26 MB ids file is never held in RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _rm_quiet(path):
    """Best-effort remove of a partial/failed temp file, swallowing any OS error -- used to clean up after
    a seed that over-ran the F2 ceiling, was unreachable, or served wrong bytes, before trying the next."""
    try:
        os.remove(path)
    except OSError:
        pass


def _is_allowed_data_name(name):
    """F1 hard-guard: is `name` a file a MINER is allowed to fetch? A data record rides the same
    UNSIGNED shared-token lane as the pointer, so a forged one could name any path. Permit ONLY a pure
    basename that is a miner-facing split (ids_<domain>_train.npy / _val.npy) or the data manifest;
    reject path traversal and -- the whole point of the guard -- the SECRET probe/heldout splits
    (ids_*_probe.npy / _heldout.npy), which live only on the coordinator box
    (sharddiloco_glm_coordinator.py:573) and must never be fetchable by miner code."""
    if not name or name != os.path.basename(name):
        return False
    if "/" in name or "\\" in name or ".." in name:
        return False
    return name == DATA_MANIFEST_NAME or bool(_ALLOWED_DATA_RE.match(name))


def _read_data_record(lane):
    """Read the coordinator's named data record (DATA_RECORD_NAME) via the manifest -- the same
    name -> sha256 -> get_json resolution read_pointer/fetch_accepted use, because ContentLane.get_json
    takes a CID, not a name. Returns the record dict, or None if the record is absent or unreadable, in
    which case the caller NO-OPs to today's local --data-dir behavior."""
    try:
        man = lane.manifest()
        entry = man.get(DATA_RECORD_NAME)
        if not entry:
            return None
        return lane.get_json(entry["sha256"])
    except Exception:                                            # noqa: BLE001
        return None


def glm_data_autosync(lane, data_dir, log=print, http_get=data_http_get):
    """W6 corpus-over-WAN DOWNLOAD half: before training, make a bare stranger clone fetch and VERIFY
    its ids files, and FAIL CLOSED on anything it cannot verify. Today the loaders bare-np.load
    whatever sits in --data-dir with ZERO verification (node_ids); this closes exactly that gap.

    Reads the coordinator's advertised record (W5 data_seeds.json shape: manifest_sha256 / seeds /
    files{name:{sha256,size}}). For each named file: keep a locally-present sha-matching copy (zero
    network), else GET <seed>/o/<sha> from the seeds IN ORDER, verify sha256(body) == sha, and install
    it atomically (tmp + os.replace) -- first verified seed wins. A record file that ends up neither
    locally-valid nor fetched-and-verified exits rc RC_DATA_UNVERIFIED (never train on unverified
    data), naming the file and every seed tried. A record that names a disallowed file is refused
    ENTIRELY (F1). Never deletes or touches a file the record does not name. Opt out with
    NEURAHASH_GLM_DATA_AUTOSYNC=0. http_get is injected like W5's uploaders so tests run this whole
    path with zero network."""
    optout = os.environ.get("NEURAHASH_GLM_DATA_AUTOSYNC", "").strip().lower()
    if optout in ("0", "false", "no", "off"):
        log("[glm-contrib] data autosync OFF (NEURAHASH_GLM_DATA_AUTOSYNC=%s) -- using --data-dir as-is"
            % optout)
        return
    record = _read_data_record(lane)
    if not record:
        log("[glm-contrib] no data record %r advertised on the lane -- using local --data-dir files as-is"
            % DATA_RECORD_NAME)
        return
    files = record.get("files") or {}
    seeds = [str(s) for s in (record.get("seeds") or [])]

    # F1 HARD-GUARD: validate EVERY key BEFORE any I/O -- one bad name poisons the whole record.
    # F2 companion: every entry must also carry a 64-hex sha256 AND a positive int size. Without a
    # size the download ceiling would be disabled, so a forged record could declare-nothing-send-huge;
    # the W5 publisher always emits both, so a well-formed record never trips this. Fail-closed.
    for name in files:
        if not _is_allowed_data_name(name):
            log("[glm-contrib] SECURITY: data record names disallowed file %r -- only "
                "ids_<domain>_(train|val).npy or %s may be fetched (the SECRET probe/heldout splits "
                "must never be); REFUSING the entire record and fetching nothing."
                % (name, DATA_MANIFEST_NAME))
            return
        info = files[name]
        sha_ok = isinstance(info, dict) and isinstance(info.get("sha256"), str) \
            and len(info["sha256"]) == 64 and all(c in "0123456789abcdef" for c in info["sha256"])
        size_ok = isinstance(info, dict) and isinstance(info.get("size"), int) \
            and not isinstance(info.get("size"), bool) and info["size"] > 0
        if not (sha_ok and size_ok):
            log("[glm-contrib] SECURITY: data record entry %r lacks a valid sha256/size (size is the "
                "download ceiling -- an entry without one is unbounded); REFUSING the entire record "
                "and fetching nothing." % name)
            return

    for name in sorted(files):
        info = files[name]
        sha = str(info.get("sha256", "")) if isinstance(info, dict) else ""
        size = info.get("size") if isinstance(info, dict) else None   # F2: untrusted declared size -> ceiling
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and _sha256_stream(path) == sha:
            log("[glm-contrib] data ok (local sha match, no fetch): %s o/%s.." % (name, sha[:12]))
            continue
        tried = []
        installed = False
        for seed in seeds:
            url = seed.rstrip("/") + "/o/" + sha
            tried.append(url)
            os.makedirs(data_dir, exist_ok=True)
            tmp = "%s.tmp.%d" % (path, os.getpid())
            try:
                # F2: stream to tmp under a size+slack ceiling -- a forged over-large body aborts here in
                # bounded RAM, and this seed is treated as failed (partial tmp cleaned up, try the next).
                got, nbytes = http_get(url, timeout=60, expected_size=size, dest_path=tmp)
            except Exception as e:                               # noqa: BLE001
                _rm_quiet(tmp)
                log("[glm-contrib] data seed unusable %s (%s)" % (url, e))
                continue
            if got != sha:
                _rm_quiet(tmp)
                log("[glm-contrib] data seed served WRONG BYTES %s (got %s.. want %s..)"
                    % (url, got[:12], sha[:12]))
                continue
            os.replace(tmp, path)
            log("[glm-contrib] data FETCHED+VERIFIED %s (%d B) from %s" % (name, nbytes, url))
            installed = True
            break
        if not installed:
            log("[glm-contrib] FATAL: cannot verify data file %s (want sha256 %s). Tried %d seed(s): "
                "%s. Refusing to train on unverified data (rc%d)."
                % (name, sha, len(tried), tried, RC_DATA_UNVERIFIED))
            raise SystemExit(RC_DATA_UNVERIFIED)
    log("[glm-contrib] data autosync OK: %d file(s) verified against record %r"
        % (len(files), DATA_RECORD_NAME))


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


# ==================================================================== alpha 2.0 non-blocking cadence (#146)
# The truly-decoupled lane (docs/ALPHA2_PLAN.md sec 1-3). These four helpers are the PURE decisions the
# async cadence makes -- factored out so they are unit-testable with NO socket / GPU / model. The sync
# loop in main() is UNCHANGED (byte-identical alpha-1.0 join path); the async cadence is a separate
# function (_run_async) reached only via the single pointer-driven mode-selection branch.

_ASYNC_OPT_OUT = ("0", "false", "no", "off", "n")   # NEURAHASH_SD_ASYNC values that force the sync path


def _select_async_mode(ptr, env=None):
    """Pointer-driven mode selection (alpha 2.0 #146). Decode the pointer with the W1 codec and return
    True iff the NON-BLOCKING async cadence should run:
      * v1 pointer  -> ALWAYS False (a fresh public clone must still join today's v1 sync lanes
        BYTE-IDENTICALLY -- pointer-version, never env, decides this).
      * v2 pointer  -> True UNLESS NEURAHASH_SD_ASYNC is the explicit opt-out (0/false/no/off/n), in
        which case False = sync fallback. The fallback is safe and does NOT crash: a v2 pointer carries
        the v1 aliases (round==event, state_cid==model_root, diloco_merge.py:1213) as a strict superset,
        so the sync loop reads those two fields and simply ignores the per-slot breakdown.
    `env` defaults to os.environ; passed explicitly by tests. Pure: decode only, no I/O."""
    dec = dm.sd_pointer_decode(ptr)
    if int(dec.get("v", 1)) != 2:
        return False
    e = os.environ if env is None else env
    optout = (e.get("NEURAHASH_SD_ASYNC", "") or "").strip().lower()
    return optout not in _ASYNC_OPT_OUT


def scan_accepted_events(manifest_names, last_applied, max_scan=1_000_000):
    """Non-blocking catch-up scan (alpha 2.0 #146). Given the names currently present in the lane
    manifest (any container supporting `in`; a manifest dict's keys work directly) and the last event
    already folded locally, return the ORDERED list of accepted event numbers to apply next:
    e = last_applied+1, +2, ... while accepted_name(e) is present, STOPPING at the first gap.

    Global events are monotonic and contiguous (SlotClock bumps by 1 per advance and the coordinator
    publishes accepted(e) for each -- diloco_merge.py:1284, ALPHA2_PLAN sec 2), so a gap means 'not yet
    visible'. We never skip a gap: applying e+1 before e would fold deltas out of the coordinator's
    order and diverge the replica. Returns [] when nothing new is visible -- the caller then trains
    against the current base rather than waiting. `max_scan` is a defensive finite cap so a malformed
    manifest cannot loop forever; unreached on the happy path. Pure: no I/O."""
    out = []
    e = int(last_applied) + 1
    n = 0
    while n < int(max_scan) and accepted_name(e) in manifest_names:
        out.append(e)
        e += 1
        n += 1
    return out


def scan_accepted_events_bounded(manifest_names, last_applied, frontier, max_scan=1_000_000):
    """scan_accepted_events BOUNDED by the pointer's authoritative event frontier (restart hygiene).
    The lane store never deletes, so after a coordinator restart the manifest still lists accepted
    records from the PREVIOUS run at events the new run has not reached yet. The pointer is the
    single source of truth for how far THIS run has advanced -- folding any accepted(e) with
    e > pointer.event would replay a dead run's merges onto a fresh base (measured live 2026-07-23:
    a restarted lane poisoned a contributor through exactly this). frontier=None -> unbounded
    (identical to scan_accepted_events). Pure: no I/O."""
    evs = scan_accepted_events(manifest_names, last_applied, max_scan=max_scan)
    if frontier is None:
        return evs
    f = int(frontier)
    return [e for e in evs if e <= f]


def async_should_abort_no_progress(local_root, pointer_root, applied_any, seconds_since_progress,
                                   round_wait):
    """Async lane no-progress abort decision (alpha 2.0 #146, reuses rc6 semantics). Root mismatch is
    NORMAL mid-flight -- another slot advanced between our reads -- so a mismatch ALONE never aborts
    (this is the deliberate departure from the v1 sync rc7 drift-abort). We abort ONLY when the
    coordinator advertises a root we cannot reach AND we have folded no accepted record for
    `round_wait` seconds, i.e. the missing records are unreconstructable rather than merely late.
    Rules, in order:
      * applied_any True     -> False  (any progress this tick resets the timer -> never abort now).
      * falsy pointer_root   -> False  (coordinator advertises no root -> nothing to be stuck against).
      * local == pointer root -> False (fully caught up -> not stuck).
      * else                 -> True iff seconds_since_progress >= round_wait.
    Pure: no clock read (the caller passes the elapsed time), no I/O."""
    if applied_any:
        return False
    if not pointer_root:
        return False
    if local_root == pointer_root:
        return False
    return float(seconds_since_progress) >= float(round_wait)


def build_async_contrib_record(miner, i, L, E, base_event, base_root, expert_cid, sig, train_flops,
                               delta_bytes, steps, tokens):
    """Assemble the async-lane contribution record: today's signed record EXTENDED with the alpha-2
    telemetry the coordinator (W2) reads -- base_event (the event this delta was trained against; the
    r-number in the contrib name MEANS this), base_root (our local model_root), steps (inner steps
    executed) and tokens (rows*seq consumed). It is a strict SUPERSET of the sync record: base_round is
    kept == base_event so a v1-shaped reader still finds a base height, mirroring the v2 pointer's
    superset discipline. Pure dict assembly, no I/O."""
    return dict(
        miner=miner, expert=int(i), layer=int(L), glm_expert=int(E),
        base_round=int(base_event),          # v1-compat alias: the r-number == base_event
        base_event=int(base_event),
        expert_cid=expert_cid, trunk_cid=None, sig=sig,
        train_flops=float(train_flops), trunk_bytes=0, delta_bytes=int(delta_bytes),
        base_root=base_root, steps=int(steps), tokens=int(tokens),
    )


def async_publish_name(base_event, miner, k):
    """F-Q1: the UNIQUE per-publish contribution name = contrib_name(base_event, miner) + a per-miner
    monotonic counter suffix '.<k>'. When a miner completes >=2 H-blocks against ONE base_event (the
    coordinator merge lagging), each publish lands on a DISTINCT manifest name instead of the 2nd atomically
    repointing (and silently losing) the 1st -- the exact lost-work bug F-Q1 closes. The signature covers
    ecid/base_event/miner, NOT the name (H.sign at the publish site), so the suffix is signature-safe, and
    the coordinator reads base_event/miner from the RECORD, not the name. Pure."""
    return "%s.%d" % (contrib_name(base_event, miner), int(k))


def _run_async(args, lane, host, model, cfg, G, key, i, L, E, miner, train_ids, val_ids, seq, log):
    """NON-BLOCKING async cadence (alpha 2.0 #146). Selected by main() only when the coordinator
    publishes a v2 pointer. The contributor NEVER waits on a barrier: each iteration it
      (1) scans the manifest ONCE and folds any accepted records past last_applied (non-blocking
          catch-up, reusing apply_accepted incl. the F2/F5 own-slot re-gate, 4984891) -- if none are
          visible it does NOT wait;
      (2) trains H local LoRA steps on its OWN slot against the CURRENT base;
      (3) publishes its delta + a signed record extended with base_event/base_root/steps/tokens, then
          loops immediately.
    Root mismatch is EXPECTED mid-flight and never aborts on its own; only prolonged no-progress while
    our root cannot reach the coordinator's aborts (rc6). Self-abort codes preserved: rc6 (no progress,
    redefined here), rc8 (poisoned accepted record -- unchanged from sync). rc7 (drift) does NOT exist
    in this path. Returns the process exit code. The only sleeps are args.poll pacing on a transient
    manifest/pointer read failure -- there is no barrier sleep."""
    # LOCAL re-gate closure, IDENTICAL to the sync path: the own-slot delta is re-gated on our own val
    # split (F2 defense-in-depth); cross-domain deltas fold unconditionally on the coordinator's signed
    # accept (own_slot=i scopes the check so a cross-domain accept is not false-positive rejected).
    regate_ce = (lambda h: G.heldout_ce(h.model, val_ids)) if len(val_ids) else None
    last_applied = 0            # events <= this are folded into our base (event 0 == the fresh base)
    publish_k = 0               # F-Q1: per-miner monotonic publish counter -> a UNIQUE record name every
                                # publish, so a 2nd H-block against the same base_event never repoints (and
                                # silently drops) the previous record. Never resets within a run.
    last_progress_t = time.time()
    rounds_done = 0             # OUR OWN published contributions -- this is what --max-rounds counts here
    log("[glm-contrib %s] ASYNC cadence (v2 lane, #146): non-blocking; train continuously, never wait "
        "on a barrier. --max-rounds=%d counts our own contributions." % (miner, args.max_rounds))
    while rounds_done < args.max_rounds:
        # -- pointer read: done flag + the coordinator's advertised root. Transient failure -> pace. --
        try:
            ptr = lane.read_pointer()
        except Exception:                                        # noqa: BLE001
            time.sleep(args.poll)
            continue
        if ptr is None:
            time.sleep(args.poll)
            continue
        try:
            dec = dm.sd_pointer_decode(ptr)                      # decode EVERY pointer read (v1|v2)
        except ValueError:
            time.sleep(args.poll)                               # malformed pointer mid-flight -> pace
            continue
        if dec["done"]:
            log("[glm-contrib %s] coordinator signalled DONE; exiting after %d contributions"
                % (miner, rounds_done))
            return 0
        pointer_root = dec["model_root"]

        # -- (1) NON-BLOCKING catch-up: fold every accepted record past last_applied, IN ORDER. -------
        try:
            man = lane.manifest()
        except Exception:                                        # noqa: BLE001
            time.sleep(args.poll)
            continue
        applied_any = False
        for e in scan_accepted_events_bounded(man, last_applied, dec.get("event")):
            entry = man.get(accepted_name(e))
            if not entry:
                break                                            # gap: stop -- never fold out of order
            try:
                rec = lane.get_json(entry["sha256"])
            except Exception:                                    # noqa: BLE001
                break                                            # transient fetch fail -> retry next tick
            rejected = []
            apply_accepted(host, lane, rec, log=log, ce_fn=regate_ce, rejected=rejected, own_slot=i)
            if rejected:
                log("[glm-contrib %s] SECURITY: locally REJECTED %d accepted delta(s) at event %d "
                    "(regressed local held-out CE or mismatched shape). The pointer + accepted record "
                    "ride an UNSIGNED shared-token lane, so this looks like a forged/poisoned record -- "
                    "refusing to fold it and aborting rather than training on a poisoned base."
                    % (miner, len(rejected), e))
                return 8
            last_applied = e
            applied_any = True
        if applied_any:
            last_progress_t = time.time()

        # -- (2) root mismatch is NORMAL mid-flight; abort ONLY on prolonged no-progress (rc6). -------
        root = model_root(host)
        if async_should_abort_no_progress(root, pointer_root, applied_any,
                                          time.time() - last_progress_t, args.round_wait):
            log("[glm-contrib %s] FATAL: no accepted-record progress for %.0fs while local model_root="
                "%s.. cannot reach coordinator root=%s.. (missing records are unreconstructable, not "
                "merely late)." % (miner, args.round_wait, root[:12], str(pointer_root)[:12]))
            return 6

        # -- (3) train H local LoRA steps on my slot against the CURRENT base, ZERO cross-miner comm. --
        t_tr = time.time()
        if args.garbage:
            delta = G.garbage_delta(host.read_slot(i), scale=3.0, seed=1234 + rounds_done)
            train_flops, best_val, steps, tokens = 1.0, float("nan"), 0, 0
            payload = delta                    # adversarial control stays dense: it has no factors
        else:
            c = G.train_glm_expert_contribution(
                model, cfg, L, E, train_ids, val_ids, H=args.inner, r=args.lora_r, lr=args.lr,
                batch=args.batch, seed=rounds_done * 100 + i, sel_outer=args.outer)   # F5 select@gate
            delta, train_flops, best_val = c["delta"], c["train_flops"], c["best_val_ce"]
            steps = int(args.inner)                              # H inner steps executed
            tokens = int(c.get("n_examples", 0)) * int(seq)      # rows*seq actually consumed
            payload = c["lora"] if (args.wire == "lora" and c.get("lora")) else delta

        # -- (4) publish today's payload + signed record EXTENDED with base_event/base_root/steps/tokens.
        base_event = int(last_applied)         # the event this delta was trained against
        ecid = lane.put_delta(payload)
        sig = H.sign(key, ecid, base_event, miner)               # r-number == base_event (W2 reads it so)
        delta_bytes = int(len(H.pack_arrays(payload, np.float16)))
        record = build_async_contrib_record(miner, i, L, E, base_event, root, ecid, sig, train_flops,
                                             delta_bytes, steps, tokens)
        pub_name = async_publish_name(base_event, miner, publish_k)   # F-Q1: unique name per publish
        lane.put_json_named(pub_name, record)
        publish_k += 1
        rounds_done += 1
        log("[glm-contrib %s] async round %d: %s slot %d (L%d,E%d) in %.1fs, best_val_ce=%.5f, "
            "base_event=%d published as %s expert_cid=%s.. delta=%dB base_root=%s.. steps=%d tokens=%d"
            % (miner, rounds_done, "GARBAGE (adversarial control)" if args.garbage else
               "trained %d LoRA steps on" % args.inner, i, L, E, time.time() - t_tr, best_val,
               base_event, pub_name, ecid[:12], delta_bytes, root[:12], steps, tokens))
        # (5) loop IMMEDIATELY: scan -> train -> publish -> scan. No barrier, no re-advertise wait.

    log("[glm-contrib %s] hit max-rounds=%d; exiting" % (miner, args.max_rounds))
    return 0


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
    ap.add_argument("--outer", type=float, default=float(os.environ.get("NEURAHASH_SD_OUTER", "0.7")),
                    help="LoRA strength the coordinator gates + MERGES at (base += outer*delta). F5: the "
                         "contributor SELECTS its save-best adapter at THIS same strength so best_val_ce "
                         "predicts the gate. MUST match the coordinator's --outer (shared "
                         "NEURAHASH_SD_OUTER default 0.7).")
    ap.add_argument("--lr", type=float, default=float(os.environ.get("NEURAHASH_GLM_LR", "3e-3")))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("NEURAHASH_GLM_BATCH", "16")),
                    help="B=4 on the 4060 (plan sec 1: vocab 154880 log_softmax is ~1.2 GiB at B=48)")
    ap.add_argument("--wire", default=os.environ.get("NEURAHASH_GLM_WIRE", "lora"),
                    choices=("lora", "dense"),
                    help="lora (default) ships the LoRA factors -- 67.7x smaller than the dense "
                         "delta they materialise to, and the only wire the shared lane accepts; "
                         "dense ships the materialised weight delta (18.87 MB/round, LAN only)")
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

    if args.mode != "tiny":
        # W6 corpus-over-WAN: fetch+verify this miner's ids files BEFORE anything reads them.
        # build_node_model() below infers seq length from ids_<dom>_val.npy (_infer_seq), so this MUST
        # run first; a single call here covers BOTH the sync and async cadences that branch below.
        glm_data_autosync(lane, args.data_dir, log=_flush)

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

    # ---- MODE SELECTION (alpha 2.0, #146): pointer-driven, decided ONCE on the first pointer. -------
    # A v2 pointer (coordinator opted into NEURAHASH_SD_ASYNC) runs the NON-BLOCKING async cadence; a v1
    # pointer -- or an explicit NEURAHASH_SD_ASYNC=0 opt-out on a v2 lane -- falls through to the EXISTING
    # sync loop below, BYTE-IDENTICAL, so a fresh public clone still joins today's v1 lanes and an operator
    # can force the old behavior. The opt-out cannot crash: a v2 pointer carries the v1 aliases
    # (round==event, state_cid==model_root) as a strict superset, so the sync loop reads those two fields
    # and ignores the per-slot breakdown. See docs/ALPHA2_PLAN.md sec 2 + _select_async_mode.
    _mode_async = _select_async_mode(ptr, os.environ)
    _pdec = dm.sd_pointer_decode(ptr)
    _flush("[glm-contrib %s] MODE=%s (pointer v%s event=%s name=%s) -- #146 async iff v2"
           % (miner, "ASYNC" if _mode_async else "SYNC", _pdec.get("v"), _pdec.get("event"),
              H.POINTER_NAME))
    if _mode_async:
        return _run_async(args, lane, host, model, cfg, G, key, i, L, E, miner,
                          train_ids, val_ids, seq, _flush)

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
        # RE-GATE the delta for OUR OWN trained slot on our LOCAL val split (F2 defense-in-depth): the
        # pointer + accepted record are UNSIGNED on a shared-token lane, so a forged record could push
        # an ungated delta into every replica. We refuse to fold one that regresses our held-out CE.
        # Cross-domain deltas (OTHER slots) are folded on the coordinator's signed accept -- our
        # single-domain val cannot judge another node's domain, so re-gating them there false-positive
        # rejected legitimate accepts and self-aborted this replica (own_slot=i scopes the check).
        regate_ce = (lambda h: G.heldout_ce(h.model, val_ids)) if len(val_ids) else None
        for r in range(applied + 1, rnd):
            rec = fetch_accepted(lane, r, timeout=args.round_wait, poll=args.poll)
            if rec is None:
                _flush("[glm-contrib %s] FATAL: accepted record for round %d never appeared" % (miner, r))
                return 6
            rejected = []
            apply_accepted(host, lane, rec, log=_flush, ce_fn=regate_ce, rejected=rejected, own_slot=i)
            if rejected:
                _flush("[glm-contrib %s] SECURITY: locally REJECTED %d accepted delta(s) at round %d "
                       "(regressed local held-out CE or mismatched shape). The pointer + accepted "
                       "record ride an UNSIGNED shared-token lane, so this looks like a forged/"
                       "poisoned record -- refusing to fold it and aborting rather than training on a "
                       "poisoned base." % (miner, len(rejected), r))
                return 8
            applied = r
        root = model_root(host)
        if not args.garbage and ptr.get("state_cid") and ptr["state_cid"] != root:
            _flush("[glm-contrib %s] FATAL DRIFT at round %d: local model_root=%s.. but coordinator "
                   "advertises %s.. (replicas diverged -- refusing to train on a phantom base)"
                   % (miner, rnd, root[:12], str(ptr["state_cid"])[:12]))
            return 7

        # ---- train H local LoRA steps on my slot, with ZERO cross-miner comm ----
        t_tr = time.time()
        payload = None                    # what actually goes on the wire (dense delta or factors)
        if args.garbage:
            delta = G.garbage_delta(host.read_slot(i), scale=3.0, seed=1234 + rnd)
            train_flops, best_val = 1.0, float("nan")
            payload = delta               # adversarial control stays dense: it has no factors
        else:
            c = G.train_glm_expert_contribution(
                model, cfg, L, E, train_ids, val_ids, H=args.inner, r=args.lora_r, lr=args.lr,
                batch=args.batch, seed=rnd * 100 + i, sel_outer=args.outer)   # F5: select at the gate outer
            delta, train_flops, best_val = c["delta"], c["train_flops"], c["best_val_ce"]
            # WIRE: ship the LoRA FACTORS, not their materialised product. The dense delta IS
            # scale*(B@A), so the factors carry identical information in 67.7x fewer bytes
            # (18,874,493 -> 278,731 measured at real GLM dims). This is not an optimisation we can
            # skip: the shared VPS lane is a ~894 MB box that RESET THE CONNECTION on 18.87 MB
            # bodies, so dense-over-WAN does not work at all. fp16 transport of the factors
            # reproduces the product as faithfully as fp16 transport of the product itself
            # (relative error ratio 0.65x-1.49x across B magnitudes 1e-4..1e-2 -- both are simply
            # fp16 precision), so the gate cannot decide differently because of the wire.
            payload = c["lora"] if (args.wire == "lora" and c.get("lora")) else delta

        # ---- publish the fp16 content-addressed delta + a signed record (D1/D2), trunk FROZEN ----
        ecid = lane.put_delta(payload)
        sig = H.sign(key, ecid, rnd, miner)
        delta_bytes = int(len(H.pack_arrays(payload, np.float16)))
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
