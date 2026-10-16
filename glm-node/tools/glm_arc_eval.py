#!/usr/bin/env python
"""ARC-EASY on the FULL 47-layer GLM-4.7-Flash: BASE vs BASE + a campaign's accepted deltas.

WHICH campaign comes from --campaign-spec; with no flag it is run 5 (25 accepted deltas), whose
constants are the module defaults and whose results are published. Never write a campaign's name,
count or root into a log line or a report paragraph by hand -- read it from SPEC (live stages) or
from the result file's own `campaign_spec` (stage report). See spec_name() for why.

WHY THIS EXISTS
---------------
Every training claim this project has made is held-out CROSS-ENTROPY. CE is a fine gate but no
human can read it. This is the first REAL-BENCHMARK capability measurement of the GLM shardDiLoCo
training: multiple-choice accuracy on ARC-Easy, scored by answer log-likelihood.

It is deliberately run on the FULL model. The training-time network keeps only ONE MoE layer
resident -- tools/piece_loader.py:274-276 `_DeadExperts.forward` returns zeros, so 45 of the 46 MoE
layers contribute EXACTLY NOTHING. A benchmark on that crippled network is not a measurement of
GLM's capability and must never be presented as one.

WHY LAYER STREAMING (measured 2026-07-27, not a preference)
-----------------------------------------------------------
The full bf16 model is ~56.8 GiB. This box has 61.56 GiB RAM and one 32 GiB card:
  * pure CPU residency is refused -- 61.56 - 56.8 leaves ~5 GiB for Windows, well under the 12 GiB
    safety floor this task was given, and it would swap;
  * accelerate `device_map="auto"` + `offload_folder` (the tools/glm_rollout_worker.py:541 pattern)
    SEGFAULTED at 64% of the load, the instant it began writing disk-offload files (one 402 MB
    file written, then SIGSEGV; 35 GiB RAM still free, pagefile peak 16 GB -- a native crash in
    the offload writer, not memory exhaustion).
So the experts are STREAMED: the frozen trunk (4.02 GiB) stays resident on the GPU, the first
`--pin-layers` MoE layers are pinned, and every other MoE layer's 64 experts are read from a
pre-packed per-layer safetensors file straight into GPU memory in a forward pre-hook and freed in
the forward hook. The stock `Glm4MoeLiteNaiveMoe.forward` runs unmodified -- only weight RESIDENCY
is managed here, never the math.

STAGES
------
  pack     ONE-TIME. Fuse the checkpoint's per-expert tensors into per-layer
           {gate_up [64,3072,2048], down [64,2048,1536]} bf16 files (~1.125 GiB/layer, 46 layers).
           The fusion convention cat([gate,up],0) was PROVEN bit-exact against
           from_pretrained's own fused parameter before this was written.
  selftest EQUIVALENCE GATE. Takes a real 2-layer model, records its logits, then replaces layer
           1's experts with the streaming module + hooks and records them again. Same weights,
           different mechanism -> the logits must be BIT-IDENTICAL. This is what proves the
           streaming machinery does not silently change the model.
  fold     CPU, ~5.2 GiB. The composition test's EXACT gated path: build the piece_loader model at
           run 5's residency (pieces 0-11 = layer 1), register run 5's 8-slot registry, fold the 25
           accepted records through the contributor's own `_fold_accepted_checked`, and require the
           post-fold model_root to equal run 5's final root. Snapshots the affected expert rows
           PRISTINE and FOLDED. A fold that does not reproduce af7f0ce06b5f.. is refused: a
           benchmark of a wrong fold is worse than no benchmark.
  bench    Build the streaming full model, prove 46/46 MoE layers real and 0 dead, run the
           CROSS-PATH GATE (packed layer-1 rows must be BIT-EXACT with the piece_loader rows from
           `fold` -- ground truth from a different loader), score arm 1, patch layer 1, score arm 2.
           Both arms score the IDENTICAL seeded subset, in the identical order, with identical code.

SCORING
-------
Log-likelihood multiple choice, no generation. For each choice: token log-probs of the continuation
given "Question: ...\\nAnswer:". Reported both ways --
  acc_norm (PRIMARY) : mean per continuation token  (length-normalized)
  acc                : raw sum                       (favours short choices)

Usage (Git Bash):
  P=C:/Python313/python.exe
  PYTHONIOENCODING=utf-8 $P tools/glm_arc_eval.py --stage pack
  PYTHONIOENCODING=utf-8 NEURAHASH_VRAM_CAP_GB=23 $P tools/glm_arc_eval.py --stage selftest
  CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 $P tools/glm_arc_eval.py --stage fold --deltas D.pt
  PYTHONIOENCODING=utf-8 NEURAHASH_VRAM_CAP_GB=23 $P tools/glm_arc_eval.py --stage bench \
      --deltas D.pt --n 0 --out docs/research/ARC_EASY_RUN5_2026-07-27.json

READ-ONLY with respect to the fleet: GETs from the lane, never PUTs, never signals a process,
never writes outside --deltas / --out / --pack-dir.
"""
import argparse
import json
import math
import os
import random
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import no_toy_models as _NTM  # noqa: E402  (tools/ is on sys.path from the block above)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ------------------------------------------------------------------ run 5, read off the live artifacts
# (identical constants to tools/glm_layer_composition_test.py -- do not drift them)
CAMPAIGN = "9ac163509f772569"
LANE_URL = "http://47.84.93.96:8710"
LANE_TOKEN = "2802648a1e87b4b3c6ca6da2688b4308"
ACCEPTED_EVENTS = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 14, 17, 25, 28, 32, 34, 35, 37, 41, 54, 57, 63,
                   76, 82, 95]
RUN5_SLOTS = [(1, 0), (1, 12), (1, 45), (1, 18), (1, 2), (1, 50), (1, 23), (1, 35)]
FINAL_ROOT_FULL = "af7f0ce06b5f29be9c9042debd6e7eb6c8b84a7c40d26fa87b5067003b579f0e"
PIECES_L1 = "0-11"                      # run 5's residency: layer 1, experts 0..59

# ------------------------------------------------------------------ CAMPAIGN SPEC (parameterisation)
# WHY this exists. Everything above describes ONE campaign (run 5), and this tool has PUBLISHED
# results computed from exactly those constants -- so they stay the DEFAULT, byte-for-byte. A later
# campaign is the same measurement over a different (campaign, accepted events, ordered slot
# registry, final root) tuple, supplied as a JSON file via --campaign-spec. Every stage reads SPEC
# rather than the constants, so no site can be left behind on a stale run-5 value.
#
# What each field is and where it comes from (all four are DERIVED from the lane, never invented):
#   campaign        the coordinator's campaign_id (D:/glm_daily/coord/glm_campaign.json)
#   accepted_events the events whose accepted record carries a NON-EMPTY `accepted` list. The
#                   accepted lane publishes a record for EVERY event, most of them pure rejects, so
#                   "every name under the prefix" is the wrong list -- folding those is a no-op.
#   slots           the coordinator's ORDERED slot registry (glm_slot_registry.json). Order is
#                   load-bearing: model_root hashes host.slots IN LIST ORDER.
#   final_root      the `model_root` the coordinator advertised at the LAST accepted event. This is
#                   the fold gate's target and it is never derived from our own computation.
#   accepted_scope  "campaign" for lanes scoped by the 2026-07-27 accepted-record fix
#                   (sharddiloco/glm/accepted/<campaign>/rN), "legacy" for the flat pre-fix family
#                   (sharddiloco/glm/accepted/rN) that run 5 published into. Never auto-detected:
#                   the legacy family still holds 1,345 foreign records and guessing wrong would
#                   silently fold another campaign's deltas (memory `cross-campaign-record-replay`).
#   frozen_cut_event the highest event number the snapshot was taken at -- PROVENANCE for a live
#                   campaign, so a later reader can tell what was and was not included.
RUN5_SPEC = dict(
    label="run5",
    campaign=CAMPAIGN,
    accepted_events=list(ACCEPTED_EVENTS),
    slots=[list(c) for c in RUN5_SLOTS],
    final_root=FINAL_ROOT_FULL,
    pieces=PIECES_L1,
    accepted_scope="legacy",
    lane_url=LANE_URL,
    lane_token=LANE_TOKEN,
    frozen_cut_event=None,
)
SPEC = dict(RUN5_SPEC)                  # replaced by apply_campaign_spec() when --campaign-spec is given

# Model locations are PER BOX. They were hardcoded to this workstation's layout, which meant the
# module could not run on any second machine -- i.e. on any miner. Env knobs follow the existing
# NEURAHASH_SD_* convention (see lmbwd_core.PACK_DIR); defaults are the original literals, so the
# 5090 resolves exactly as before and no measurement taken there is affected.
SHARD_DIR = os.environ.get("NEURAHASH_SD_SHARD_DIR", "D:/hf_models/GLM-4.7-Flash-bf16_shards_100mb")
CONFIG_DIR = os.environ.get("NEURAHASH_SD_CONFIG_DIR", "D:/hf_models/GLM-4.7-Flash-bf16")
ARC_PATH = os.path.join(_REPO, "arc_easy_test.jsonl")

# ---------------------------------------------------------------- LAYER-MAJOR INVERSION: DEFAULT
# ONE switch for both inversions (`--ce-layer-major`, `--arc-layer-major`). It is the CLI default
# only: the underlying functions keep OFF-shaped defaults (`run_arm(layer_major=False)`,
# `getattr(opts, "...", False)`), so a caller that never heard of the flag still gets the shipped
# path. Flipping this constant changes what an operator gets when they pass NO flag at all.
#
# ON since 2026-08-10 (owner decision), licensed by a GREEN flag-level gate, not by faith:
#   tools/glm_arc_flag_gate.py exit 0 -> scratchpad/ARC_FLAG_GATE_2026-08-10.json. It drove this
#   file's own seams (`run_arm`, `ce_scored`), not the primitives underneath, and it can fail:
#   its comparators are handed a planted 1-ULP move and a flipped prediction first and must catch
#   both, and each arm's bytes must match its own read hypothesis AND be rejected by the other, so
#   a silent fallback to batch-major (which would produce identical floats) reads as FAIL.
#   Measured there: 640 rows, 0/1280 float mismatches, max |diff| 0.000e+00, acc and acc_norm
#   identical to 17 digits, 0 per-item prediction differences; CE with every flag OFF still
#   4.810013471111175 exactly. Supporting evidence:
#   CE  256 seqs / chunk 32 / pin[1,2]: 4.810013471111175 both ways, 1584.0 -> 49.5 GiB,
#       822.3 -> 30.0 s  (scratchpad/CE_FLAG_VERIFY_2026-08-10.json, all_exact true)
#   ARC 9501 rows: 0 mismatches, max |diff| 0.000e+00, 7375.5 -> 49.5 GiB, 2430.4 -> 51.1 s
#       (tools/glm_arc_layer_major_gate.py), and the flag-level A/B in tools/glm_arc_flag_gate.py
#       which also pins the published ACCURACY and asserts the byte counts in BOTH directions.
# The escape hatch is `--no-ce-layer-major` / `--no-arc-layer-major`, or NEURAHASH_GLM_NO_LAYER_MAJOR=1
# to force both off without editing an invocation (for a bisect, or a box where the inversion is
# suspected). If a paid number ever moves, that env var is the fleet-wide undo.
LAYER_MAJOR_DEFAULT = True
_NO_LM_ENV = "NEURAHASH_GLM_NO_LAYER_MAJOR"


def layer_major_default():
    """The CLI default for both inversions, after the env kill switch has had its say."""
    v = os.environ.get(_NO_LM_ENV, "").strip().lower()
    if v not in ("", "0", "false", "no", "off"):
        return False
    return LAYER_MAJOR_DEFAULT

BASE_ARGV = [
    "--mode", "glm", "--slots", "1:0", "--max-active-slots", "16", "--device", "cpu",
    "--shard-dir", SHARD_DIR, "--config-dir", CONFIG_DIR,
    "--data-dir", "D:/glm_daily/miner", "--coord-data-dir", "D:/glm_daily/coord",
    "--domains", "daily,daily", "--url", LANE_URL, "--token", LANE_TOKEN,
]


def _log(msg):
    print(msg, flush=True)


def _ascii(s):
    """cp1252 console safety: the ARC corpus is mostly ASCII but not guaranteed."""
    return str(s).encode("ascii", "replace").decode("ascii")


# ---------------------------------------------------------------- campaign spec: read + validate
def spec_campaign():
    return SPEC["campaign"]


def spec_events():
    return list(SPEC["accepted_events"])


def spec_slots():
    return [(int(L), int(E)) for (L, E) in SPEC["slots"]]


def spec_root():
    return SPEC["final_root"]


def spec_pieces():
    return SPEC["pieces"]


def spec_label():
    return SPEC["label"]


def spec_name():
    """The string every human-readable label and every line of report prose must be built from.
    Carries BOTH the operator's label and the campaign id, because a label is free text and the id
    is the only thing that ties a paragraph to a lane. Never write a campaign name by hand: on
    2026-08-06 stage_bench logged "run 5's 25 accepted deltas" while it was benchmarking campaign
    31627ec8f184cfd6's 84 deltas, and a log line like that is how a wrong attribution outlives the
    run that produced it."""
    return "%s (campaign %s)" % (SPEC["label"], SPEC["campaign"])


def spec_lane():
    return SPEC["lane_url"], SPEC["lane_token"]


def spec_accepted_prefix():
    """The manifest prefix EVERY accepted record of this campaign shares. Delegates to the
    contributor's own accepted_prefix() so this tool cannot drift from the lane's naming, and passes
    campaign=None for a legacy (pre-2026-07-27) lane -- which is byte-for-byte what run 5 read."""
    import sharddiloco_glm_contributor as N
    return N.accepted_prefix(spec_campaign() if SPEC["accepted_scope"] == "campaign" else None)


def validate_spec(s):
    """Reject a malformed spec LOUDLY at load time. Every check here guards a way to produce a
    confidently wrong number: a short root would make the gate compare prefixes, a duplicated
    coordinate would put two slot indices on one expert, and an unsorted/duplicated event list would
    fold a record twice. Returns the normalised spec."""
    out = dict(RUN5_SPEC)
    unknown = sorted(set(s) - set(RUN5_SPEC))
    if unknown:
        raise SystemExit("[arc/spec] FATAL: unknown key(s) %r; known: %r"
                         % (unknown, sorted(RUN5_SPEC)))
    out.update(s)
    cid = str(out["campaign"] or "")
    if not cid or any(c not in "0123456789abcdef" for c in cid):
        raise SystemExit("[arc/spec] FATAL: campaign %r is not lowercase hex" % (out["campaign"],))
    # LENGTH BOUND -- not cosmetic. The lane's own grammar is
    # sharddiloco_glm_contributor._CAMPAIGN_RE = [0-9a-f]{8,64}, and normalize_campaign_id() returns
    # None for anything outside it. Hex-but-too-short (a truncated "31627ec", one char shy of the
    # real id) used to pass this check, and then accepted_prefix(None) SILENTLY fell back to the
    # LEGACY flat family -- 1,345 foreign records from other campaigns, i.e. folding someone else's
    # deltas and publishing them as this campaign's (memory `cross-campaign-record-replay`).
    if not (8 <= len(cid) <= 64):
        raise SystemExit("[arc/spec] FATAL: campaign id must be 8-64 lowercase hex chars (the "
                         "lane's own _CAMPAIGN_RE); got %d char(s): %r" % (len(cid), cid))
    root = str(out["final_root"] or "")
    if len(root) != 64 or any(c not in "0123456789abcdef" for c in root):
        raise SystemExit("[arc/spec] FATAL: final_root must be 64 lowercase hex chars, got %r" % root)
    evs = [int(e) for e in out["accepted_events"]]
    if not evs:
        raise SystemExit("[arc/spec] FATAL: accepted_events is empty")
    if evs != sorted(evs) or len(set(evs)) != len(evs):
        raise SystemExit("[arc/spec] FATAL: accepted_events must be sorted and unique")
    slots = [(int(L), int(E)) for (L, E) in out["slots"]]
    if not slots:
        raise SystemExit("[arc/spec] FATAL: slots is empty")
    if len(set(slots)) != len(slots):
        raise SystemExit("[arc/spec] FATAL: duplicate coordinate in slots -- the registry is "
                         "append-only and an index is never reused")
    if out["accepted_scope"] not in ("legacy", "campaign"):
        raise SystemExit("[arc/spec] FATAL: accepted_scope must be 'legacy' or 'campaign', got %r"
                         % (out["accepted_scope"],))
    if out["accepted_scope"] == "campaign":
        # The id is about to be pasted into object names by accepted_prefix(cid). Ask the lane's OWN
        # normaliser whether it would accept it unchanged, rather than trusting the two checks above
        # to stay in sync with _CAMPAIGN_RE. A rejected id there means accepted_prefix(None) -> the
        # legacy family, i.e. a scoped spec quietly reading the unscoped lane.
        import sharddiloco_glm_contributor as N
        if N.normalize_campaign_id(cid) != cid:
            raise SystemExit("[arc/spec] FATAL: accepted_scope='campaign' but the lane's own "
                             "normalize_campaign_id(%r) returns %r -- the campaign-scoped prefix "
                             "would fall back to the LEGACY flat family"
                             % (cid, N.normalize_campaign_id(cid)))
    out["accepted_events"], out["slots"] = evs, [list(c) for c in slots]
    out["campaign"], out["final_root"] = cid, root
    return out


def is_run5_default():
    """True iff the RESOLVED spec is run 5's published constants -- the WHOLE tuple, not just the
    campaign id. Gating anything on the id alone is unsafe: a LATER CUT of the same campaign (same
    id, more accepted_events, a new final_root -- precisely what `frozen_cut_event` exists to
    record) is a DIFFERENT measurement, and letting it pass an id-only check is how the published
    run-5 JSONs would get overwritten with numbers that are not the published run's."""
    return SPEC == validate_spec(dict(RUN5_SPEC))


def apply_campaign_spec(path):
    """Install a campaign spec read from `path`. No path -> SPEC stays the run-5 default untouched."""
    global SPEC
    if not path:
        return SPEC
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    SPEC = validate_spec(raw)
    return SPEC


def log_spec(tag):
    _log("[%s] campaign      = %s (label=%s, accepted_scope=%s, frozen_cut_event=%s)"
         % (tag, SPEC["campaign"], SPEC["label"], SPEC["accepted_scope"], SPEC["frozen_cut_event"]))
    _log("[%s] accepted events= %d (first=%d last=%d)"
         % (tag, len(SPEC["accepted_events"]), SPEC["accepted_events"][0],
            SPEC["accepted_events"][-1]))
    _log("[%s] slots          = %d ordered coordinates, head=%r tail=%r"
         % (tag, len(SPEC["slots"]), SPEC["slots"][:4], SPEC["slots"][-2:]))
    _log("[%s] final_root     = %s" % (tag, SPEC["final_root"]))
    _log("[%s] accepted prefix= %s" % (tag, spec_accepted_prefix()))


def _cap_vram():
    """HARD per-process VRAM cap BEFORE the first CUDA allocation (memory vram-cap-live-verified;
    an uncapped load crashed this box on 2026-07-24). Always leaves >= 8 GiB for the desktop."""
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    cap = _resolve_vram_cap(total)
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    _log("[arc] VRAM cap %.1f of %.1f GiB (%.2f GiB left for the desktop)"
         % (cap, total, total - cap))
    return cap, total



def _desktop_headroom(total_gib):
    """How much VRAM to leave the owner's desktop, as a function of the card.

    8 GiB is right for a 32 GiB card and catastrophic for an 8 GiB one: there `total - 8.0` is
    0.0, and the old `max(1.0, ...)` floor then clamped EVERY cap -- the NEURAHASH_VRAM_CAP_GB env
    var and an explicit flag alike -- down to 1.0 GiB, silently. MEASURED on the 4060 2026-08-22:
    env said 7.4, flag said 7.0, resolved cap was 1.00 of 8.00 GiB, against a recipe that needs
    6.41 GiB. That is the 8 GB tier the fleet depends on, capped below its own working set.

    Cards >= 16 GiB keep the 8.0 reserve exactly as before, so the 5090's resolved cap is
    bit-identical to the pre-fix value -- this change cannot alter any measurement taken there.
    Small cards get a proportionate reserve instead of an impossible one."""
    return 8.0 if total_gib >= 16.0 else 1.25


def _resolve_vram_cap(total_gib):
    """THE cap. No caller may clamp this further -- that is the whole point.

    An explicit NEURAHASH_VRAM_CAP_GB is OPERATOR INTENT; the headroom reserve is a heuristic.
    When they conflict there are exactly two legal outcomes: the explicit value survives verbatim,
    or we abort loudly printing all three numbers. There is no third path where arithmetic
    silently wins -- that path is what turned an explicit 7.4 into 1.00 GiB on an 8 GiB card,
    and a percent-band on "acceptable" reductions would merely re-open the same window smaller.

    Measured consequence of getting this wrong, 2026-08-22: capping the 4060 at 6.75 when the
    operator had set 7.4 left 541 MiB free and the MoE backward OOM'd asking for 768 MiB."""
    head = _desktop_headroom(total_gib)
    explicit = float(os.environ.get("NEURAHASH_VRAM_CAP_GB", "0") or 0)
    if explicit:
        if explicit > total_gib - 0.25:
            raise SystemExit(
                "ABORT: NEURAHASH_VRAM_CAP_GB=%.2f exceeds the physical safe maximum %.2f GiB on a "
                "%.2f GiB card. Lower it deliberately; this code will not silently do it for you."
                % (explicit, total_gib - 0.25, total_gib))
        return explicit
    derived = max(4.0, total_gib - head)
    if derived < 0.50 * total_gib:
        raise SystemExit(
            "ABORT: derived VRAM cap %.2f GiB is under half of the %.2f GiB card (reserve %.2f). "
            "The shipped recipes need ~80%% of an 8 GiB card; this is a config bug, not safety."
            % (derived, total_gib, head))
    return derived


def _moe_layers(cfg):
    return list(range(int(cfg.first_k_dense_replace), int(cfg.num_hidden_layers)))


def _pack_path(pack_dir, L):
    return os.path.join(pack_dir, "layer_%d.safetensors" % L)


# =========================================================== mmap-free safetensors reading (#80)
# WHY THIS EXISTS -- the cache producer's 0xC0000005 (exit 3221225477).
#
# `safe_open(...).get_tensor(...)` does NOT copy the bytes: it memory-maps the pack and carves
# every tensor out of that mapping. MEASURED 2026-08-05, both facts, on this box:
#   * overwriting the file's bytes underneath a returned tensor CHANGES that tensor's value
#     (1.0 -> 7.0), so the tensor genuinely aliases the mapping;
#   * the tensor's UntypedStorage reports `filename is None`, i.e. the mapping is safetensors'
#     OWN (memmap2), not one torch created and can sanity-check -- which is why the recorded
#     faulting frame is torch/storage.py:470 `UntypedStorage.__getitem__`, torch slicing a
#     buffer whose page-in it does not own.
# A third measurement: the mapping is copy-on-write, so an in-place write into a returned tensor
# (PrefixStreamer._mask) silently commits private pages instead of reaching the file.
#
# Why that is fatal HERE specifically. The producer calls the reader inside a gradient-checkpoint
# recompute hook (tools/glm_layer_cache_pass.py:243 use_reentrant=False, :495
# set_checkpoint_early_stop(False)), i.e. on the AUTOGRAD ENGINE'S WORKER THREAD, ~46 layers x 2
# (forward + recompute) per unit over hundreds of units. On a mapping a failed page-in is NOT an
# IOError -- the CPU takes a fault and Windows raises 0xC0000005, tearing the process down with no
# traceback. Runs died at units ~184, ~324 and ~565 (scratchpad/dd_cache_all.py:113-117), and
# scratchpad/pack_integrity_scan.json read all 46 packs end-to-end with 0 bad bytes, so the DATA
# is sound and the fault is in the mapping machinery.
#
# All three surviving hypotheses (handle thread-safety / page invalidation during recompute /
# allocator interaction) share ONE necessary precondition: that the bytes are served through a
# mapping at all. This reader removes the precondition rather than betting on which hypothesis is
# right -- the header is parsed and the byte range is plain-read into a tensor that owns its heap
# storage, so nothing can be faulted in later, on any thread, and a genuinely bad read surfaces as
# an ordinary catchable OSError. The block at ExpertStreamer._read already recorded this as the
# only remaining form of the idea that could work (the persistent-handle variant was measured and
# reverted: it took RSS from ~11-12 GiB to 46.51 GiB).
_ST_DTYPE = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "U16": "uint16",
    "F16": "float16", "BF16": "bfloat16", "I32": "int32", "U32": "uint32",
    "F32": "float32", "I64": "int64", "U64": "uint64", "F64": "float64",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}
_ST_HEADER_MAX = 100 << 20

# ---------------------------------------------------------------- reader concurrency (phase B)
# MEASURED (scratchpad/drive_probe.log, 2026-08-10): a SINGLE reader thread tops out at ~3.3 GB/s
# on every drive in this box -- 2.90 / 3.31 / 3.27 GB/s on Gen4 x2 / Gen4 x4 / Gen5 x4 -- while the
# same drives deliver 3.29 / 6.48 / 14.00 with 8 threads. The wall is per-thread latency, not the
# link, so a single-threaded reader leaves up to 4x on the table on a fast drive and NOTHING on a
# slow one (which is why the thread count is a function of the drive, not a constant).
# `MINER_STORAGE_AUTOTUNE_PLAN_2026-08-10.md` section 9.1 confirmed the shape transfers to what
# ExpertStreamer actually does: N threads on disjoint ranges of ONE file matched N threads on N
# separate files in 9 of 9 cells (worst ratio 1.023 against a 0.80 bar), so splitting a pack's byte
# ranges is a legitimate way to spend the concurrency.
#
# THE BYTES CANNOT MOVE. Every worker reads a DISJOINT range into a disjoint slice of the same
# destination buffer, so the assembled tensor is the same object it would have been sequentially --
# tests/test_glm_st_read_threads.py asserts that by sha256, single vs N-threaded, on the fixture
# packs AND on a real 1.125 GiB production pack.
_ST_READ_THREADS_MAX = 8            # R5 never asks for more than 4; this is the hard ceiling
_ST_SPLIT_MIN_BYTES = 8 << 20       # a range smaller than this is never split (seek beats setup)
_ST_THREADS_CACHE = None            # (n, source) -- resolved once per process
ENV_ST_READ_THREADS = "NEURAHASH_ST_READ_THREADS"


def _st_resolve_threads(env, log=None):
    """Precedence (plan section 6): explicit env > cached auto-tune decision > legacy 1.

    Returns (n, source). NEVER raises and never probes hardware: `cached_decision()` reads the JSON
    the fetch/contributor path already seeded, so merely importing this module cannot cost a 7 s
    PowerShell spawn, and a box that never ran a fetch simply gets 1 = today's behaviour.
    """
    raw = str(env.get(ENV_ST_READ_THREADS, "") or "").strip()
    if raw:
        try:
            return max(1, min(_ST_READ_THREADS_MAX, int(raw))), "env:%s" % ENV_ST_READ_THREADS
        except ValueError:
            if log is not None:
                log("[arc/read] ignoring %s=%r (not an integer) -- single-threaded reads"
                    % (ENV_ST_READ_THREADS, raw))
            return 1, "env-invalid"
    try:
        from neurahash import storage_autotune as _SA
        d = _SA.cached_decision(env=env)
        if d is None:
            return 1, "autotune:none"
        # The plan ships the DECISION and its ACTUATOR on separate switches (section 6):
        # `read_threads` is the recommendation the decision line prints, and it may only become
        # the applied value under NEURAHASH_STORAGE_AUTOTUNE_APPLY=full. The VRAM-guard incident
        # is why.
        #
        # The gate is re-evaluated against the CURRENT env rather than read off the cached
        # decision, because `applied_read_threads` was frozen at cache-WRITE time: this box's
        # cache holds action=KEEP, read_threads=4, applied=1 (it was seeded with the apply switch
        # unset), so trusting the stored field would make the documented switch a no-op until
        # something happened to re-seed the cache. The recommendation is a property of the drive;
        # whether we are allowed to act on it is a property of this run.
        apply_full = str(env.get(_SA.ENV_APPLY, "") or "").strip().lower() == "full"
        n = int((getattr(d, "read_threads", 1) if apply_full
                 else getattr(d, "applied_read_threads", 1)) or 1)
        return max(1, min(_ST_READ_THREADS_MAX, n)), ("autotune:apply=full" if apply_full
                                                      else "autotune:gated")
    except Exception:                                                     # noqa: BLE001
        return 1, "autotune-unavailable"


def _st_read_threads(env=None, refresh=False, log=None):
    """Concurrency for `_st_read`. 1 means TODAY'S CODE PATH, byte for byte -- see `_st_read`."""
    global _ST_THREADS_CACHE
    memo = env is None and not refresh
    if memo and _ST_THREADS_CACHE is not None:
        return _ST_THREADS_CACHE[0]
    n, src = _st_resolve_threads(os.environ if env is None else env, log=log)
    if memo:
        _ST_THREADS_CACHE = (n, src)
        if n > 1:
            _log("[arc/read] parallel range reads: %d threads, each with its OWN file handle (%s)"
                 % (n, src))
    return n


def _st_plan_pieces(offset, mv, threads, min_split=_ST_SPLIT_MIN_BYTES):
    """Cut ONE tensor's byte range into disjoint (file_offset, destination_slice) pieces.

    Disjointness is the whole safety argument, so it is arithmetic here and asserted in the tests:
    the cuts partition [0, len(mv)) exactly once, in order, with no overlap and no gap.
    """
    n = len(mv)
    if n == 0:
        return []
    parts = max(1, min(int(threads), n // int(min_split)))
    if parts <= 1:
        return [(offset, mv)]
    step = -(-n // parts)                                     # ceil, so the last piece is short
    return [(offset + s, mv[s:min(s + step, n)]) for s in range(0, n, step)]


def _st_read_pieces(path, pieces, threads, what):
    """Fill every piece with its file bytes, using `threads` workers.

    NON-NEGOTIABLE: each worker opens its OWN handle. A shared handle has ONE file position, so
    every seek+read would have to be serialised behind a lock and the queue depth would collapse
    straight back to 1 -- which is the entire bug this function exists to fix. The per-worker
    `with open(...)` below is load-bearing, not style.
    """
    import threading
    errs, lock, nxt = [], threading.Lock(), [0]

    def _worker():
        try:
            with open(path, "rb") as fh:                       # OWN handle -- see the docstring
                while True:
                    with lock:
                        i = nxt[0]
                        nxt[0] += 1
                    if i >= len(pieces):
                        return
                    off, mv = pieces[i]
                    fh.seek(off)
                    got, want = 0, len(mv)
                    while got < want:                          # readinto may short-read
                        k = fh.readinto(mv[got:])
                        if not k:
                            raise EOFError("%s: short read for %s at offset %d -- %d of %d bytes"
                                           % (os.path.basename(path), what, off, got, want))
                        got += k
        except BaseException as e:                             # noqa: BLE001 -- re-raised below
            with lock:
                errs.append(e)

    ts = [threading.Thread(target=_worker, name="st-read-%d" % i, daemon=True)
          for i in range(max(1, int(threads)))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]


def _st_read(path, names=None, device="cpu", keep=None, threads=None):
    """Materialise tensors from a safetensors file with a PLAIN READ -- never a memory map.

    Byte-for-byte identical to `safe_open(...).get_tensor(...)` (asserted by sha256 in
    tests/test_glm_arc_no_mmap.py), but the returned tensors own ordinary heap storage, so:
      * no page is faulted in after this function returns -- there is no later moment, and no
        other thread, at which a read of the tensor can raise 0xC0000005;
      * an in-place write (PrefixStreamer._mask) hits owned memory, not a copy-on-write mapping;
      * a truncated/unreadable file raises EOFError/OSError HERE, where a caller can see it.

    `names=None` reads every tensor; `keep` is an optional predicate over names.

    `threads` (None = ask `_st_read_threads()`, i.e. the auto-tune decision) issues that many
    CONCURRENT range reads, each worker on its own handle, into disjoint slices of the same
    destination buffers. `threads=1` is the single-handle forward sweep this function has always
    done, unchanged -- the branch below is the whole difference, and the sequential arm is the
    same statements in the same order as before phase B.
    """
    import torch
    nthreads = _st_read_threads() if threads is None else max(1, int(threads))
    out = {}
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise EOFError("%s: file is shorter than a safetensors header" % os.path.basename(path))
        n = struct.unpack("<Q", raw_len)[0]
        if not 0 < n <= _ST_HEADER_MAX:
            raise ValueError("%s: safetensors header length %d out of range"
                             % (os.path.basename(path), n))
        hdr = json.loads(fh.read(n).decode("utf-8"))
        base = 8 + n
        if names is None:
            names = [k for k in hdr if k != "__metadata__"]
            if keep is not None:
                names = [k for k in names if keep(k)]
        # Read in FILE order: one forward sweep, never seeking backwards over a 1.125 GiB pack.
        order = sorted(names, key=lambda k: hdr[k]["data_offsets"][0])
        if nthreads <= 1:
            for name in order:
                meta = hdr[name]
                tname = _ST_DTYPE.get(meta["dtype"])
                if tname is None or not hasattr(torch, tname):
                    raise ValueError("%s: unsupported safetensors dtype %r for tensor %r"
                                     % (os.path.basename(path), meta["dtype"], name))
                start, end = meta["data_offsets"]
                nbytes = end - start
                buf = torch.empty(nbytes, dtype=torch.uint8)      # owns its heap storage
                mv = memoryview(buf.numpy())
                fh.seek(base + start)
                got = 0
                while got < nbytes:                               # readinto may short-read
                    k = fh.readinto(mv[got:])
                    if not k:
                        raise EOFError("%s: short read for %r -- %d of %d bytes"
                                       % (os.path.basename(path), name, got, nbytes))
                    got += k
                del mv                                            # drop the numpy alias before .to()
                out[name] = buf.view(getattr(torch, tname)).reshape(meta["shape"]).to(device)
            return out
        # ---- parallel arm: same tensors, same bytes, N handles. Allocate every destination
        # first (so the header handle can close before any worker opens its own), then fill.
        plan = []
        for name in order:
            meta = hdr[name]
            tname = _ST_DTYPE.get(meta["dtype"])
            if tname is None or not hasattr(torch, tname):
                raise ValueError("%s: unsupported safetensors dtype %r for tensor %r"
                                 % (os.path.basename(path), meta["dtype"], name))
            start, end = meta["data_offsets"]
            buf = torch.empty(end - start, dtype=torch.uint8)     # owns its heap storage
            mv = memoryview(buf.numpy())
            plan.append((name, meta, tname, buf, mv))
    pieces = []
    for name, _meta, _tname, _buf, mv in plan:
        pieces.extend(_st_plan_pieces(base + hdr[name]["data_offsets"][0], mv, nthreads))
    _st_read_pieces(path, pieces, nthreads, "%s of %s" % (order, os.path.basename(path)))
    for _off, mv in pieces:
        mv.release()                                              # sub-views before their parent
    for name, meta, tname, buf, mv in plan:
        mv.release()                                              # drop the numpy alias before .to()
        out[name] = buf.view(getattr(torch, tname)).reshape(meta["shape"]).to(device)
    return out


# ============================================================ read-ahead of the next slab (phase C)
# WHY. Every MoE layer's slab is read twice per optimizer step (forward 1->46, reverse 46->1: 92
# reads, 103.5 GiB), and each read is strictly serial with the GPU work around it: disk -> host
# buffer -> H2D -> compute, and the next layer's disk read starts only once this layer is done. On
# the RTX 4060 the disk leg is ~31 s of an ~81 s step (4 read threads), so reading layer L+1 on a
# background thread WHILE layer L is copied and computed can hide most of it.
#
# THE BYTES CANNOT MOVE, BY CONSTRUCTION. The read-ahead reads the SAME byte ranges of the SAME file
# (plain reads, never a mapping -- #80) into a reusable host buffer; the consumer then issues the
# SAME `.to(device)` copies, one per tensor, in the SAME file order, on the same stream, that
# `_st_read` issues -- so the GPU sees the identical copy-then-compute sequence and receives
# identical bytes. A layer the predictor did not guess is read by `_st_read` itself, unchanged.
# test_glm_st_prefetch.py asserts it by sha256 per delivered tensor, read-ahead on vs off, over
# ascending, descending, zig-zag, cyclic and random request orders and deliberately wrong guesses.
#
# BUFFER SAFETY. At most ONE read-ahead is outstanding and it owns one of two host buffers; the
# consumer owns the other while it copies out of it. A buffer goes back on the free list only after
# its copies have completed (`_h2d_fence`), so no read can land in memory a copy may still be
# reading. The worker thread never touches CUDA.
#
# OFF BY DEFAULT. NEURAHASH_ST_PREFETCH unset or "0" constructs nothing -- no object, no thread, no
# buffer -- and `ExpertStreamer._read` stays the exact function object it always was.
ENV_ST_PREFETCH = "NEURAHASH_ST_PREFETCH"
_ST_RA_ALIGN = 4096                 # each tensor starts page-aligned inside a read-ahead buffer
_ST_RA_JOIN_S = 60.0                # longest a shutdown waits for an in-flight read to land


def _st_prefetch_requested(env=None, log=None):
    """True iff NEURAHASH_ST_PREFETCH is exactly "1". Unset or "0" -> False, silently."""
    raw = str((os.environ if env is None else env).get(ENV_ST_PREFETCH, "") or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    if log is not None:
        log("[arc/prefetch] ignoring %s=%r (want 0 or 1) -- read-ahead stays OFF"
            % (ENV_ST_PREFETCH, _ascii(raw)))
    return False


def _st_file_stamp(path):
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


def _st_read_into(path, names, mv, threads):
    """`_st_read`'s bytes, plain-read into the writable memoryview `mv` (a reusable host buffer).

    Returns (layout, stamp):
      layout -- [(name, dtype_name, shape, buffer_offset, nbytes)] in FILE order, which is the
                order `_st_read` copies tensors to the device; None if they do not fit in `mv`;
      stamp  -- `_st_file_stamp(path)` taken BEFORE the file is opened, so a file replaced at any
                point after it can only ever cause a refusal at use time, never a stale hit.
    Tensors land at _ST_RA_ALIGN-aligned offsets; the placement is invisible to the bytes. Reads use
    the same helpers as `_st_read`: one forward sweep on one handle, or `_st_plan_pieces` +
    `_st_read_pieces` (a handle per worker) for threads > 1.
    """
    import torch
    stamp = _st_file_stamp(path)
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise EOFError("%s: file is shorter than a safetensors header" % os.path.basename(path))
        n = struct.unpack("<Q", raw_len)[0]
        if not 0 < n <= _ST_HEADER_MAX:
            raise ValueError("%s: safetensors header length %d out of range"
                             % (os.path.basename(path), n))
        hdr = json.loads(fh.read(n).decode("utf-8"))
        base = 8 + n
        order = sorted(names, key=lambda k: hdr[k]["data_offsets"][0])
        plan, at = [], 0
        for name in order:
            meta = hdr[name]
            tname = _ST_DTYPE.get(meta["dtype"])
            if tname is None or not hasattr(torch, tname):
                raise ValueError("%s: unsupported safetensors dtype %r for tensor %r"
                                 % (os.path.basename(path), meta["dtype"], name))
            start, end = meta["data_offsets"]
            at = -(-at // _ST_RA_ALIGN) * _ST_RA_ALIGN
            plan.append((name, tname, list(meta["shape"]), at, end - start, base + start))
            at += end - start
        if at > len(mv):
            return None, stamp
        if threads <= 1:
            for name, _tn, _sh, off, nbytes, fpos in plan:
                dst = mv[off:off + nbytes]
                fh.seek(fpos)
                got = 0
                while got < nbytes:                               # readinto may short-read
                    k = fh.readinto(dst[got:])
                    if not k:
                        raise EOFError("%s: short read for %r -- %d of %d bytes"
                                       % (os.path.basename(path), name, got, nbytes))
                    got += k
                dst.release()
    if threads > 1:
        pieces = []
        for _nm, _tn, _sh, off, nbytes, fpos in plan:
            pieces.extend(_st_plan_pieces(fpos, mv[off:off + nbytes], threads))
        try:
            _st_read_pieces(path, pieces, threads, "%s of %s" % (order, os.path.basename(path)))
        finally:
            for _off, piece in pieces:
                piece.release()
    return [(nm, tn, sh, off, nb) for nm, tn, sh, off, nb, _fp in plan], stamp


def _h2d_fence(device):
    """Return only once every copy already issued on `device`'s current stream has completed.

    A host->device `.to()` is synchronous today (non_blocking=False ends in cudaStreamSynchronize),
    so this returns at once; it is here so the buffer-reuse argument never rests on that detail.
    It adds no copy and no kernel, so the stream's copy-then-compute order is untouched."""
    import torch
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.current_stream(dev).synchronize()


class _SweepPredictor:
    """Guess the next `_read` from the ones seen so far. Pure bookkeeping, no I/O.

    The layer-major driver requests the streamed layers S ascending (forward) then descending
    (reverse), step after step: 1..46, 46..1, 1..46, ... So the successor of L is its neighbour in
    the current direction and, at either end, the SAME layer again -- the reverse sweep starts with
    the layer the forward sweep ended on -- which is why a repeat flips the direction. The last
    observed successor of each (previous, current) pair overrides that rule once seen, which also
    learns a cyclic 1..46, 1..46 (batch-major) pattern after one pass. A wrong guess costs one
    wasted read and nothing else.
    """

    def __init__(self, layers):
        self.order = sorted(set(int(x) for x in layers))
        self._pos = {L: i for i, L in enumerate(self.order)}
        self._p1 = None
        self._p2 = None
        self._dir = 1
        self._succ = {}

    def observe(self, L):
        L = int(L)
        if L not in self._pos:
            return None
        p1, p2 = self._p1, self._p2
        if p1 is not None:
            if p2 is not None:
                self._succ[(p2, p1)] = L
            if L > p1:
                self._dir = 1
            elif L < p1:
                self._dir = -1
            else:
                self._dir = -self._dir                     # a repeat is the sweep turnaround
        self._p2, self._p1 = p1, L
        if p1 is not None:
            learned = self._succ.get((p1, L))
            if learned is not None:
                return learned
        i = self._pos[L] + self._dir
        if 0 <= i < len(self.order):
            return self.order[i]
        return L                                           # turnaround: the next sweep repeats L


class _SlabReadAhead:
    """One background reader and two reusable host buffers: the read-ahead behind ExpertStreamer.

    Protocol, all state under one condition variable:
      schedule(L, path, threads) -- make (L, path) the ONE outstanding read (any other is dropped);
      take(L, path)              -- that read, finished (waits while it is in flight); None = miss;
      give_back(job)             -- the consumer's copies out of the job's buffer are complete.
    A dropped read that is already in flight keeps its buffer until it lands, then frees it; a read
    error is re-raised by take() in the caller that asked for that layer.
    """

    NBUF = 2

    class _Buf(object):
        __slots__ = ("t", "mv", "pinned")

    class _Job(object):
        __slots__ = ("L", "path", "threads", "state", "buf", "layout", "stamp", "err", "dropped")

    def __init__(self, names, device, layers, log=_log):
        import threading
        self.names = tuple(names)
        self.device = device
        self.log = log
        self.predictor = _SweepPredictor(layers)
        self._cv = threading.Condition()
        self._job = None
        self._bufs = []
        self._free = []
        self._pinned = None
        self._thread = None
        self._stop = False
        self._disabled = None
        self._atexit = None
        self.hits = self.misses = self.dropped = self.stale = self.nofit = self.starved = 0
        self._n_reading = 0                        # reads in flight, INCLUDING dropped ones
        self.bytes_read = 0
        self.wait_s = 0.0

    @classmethod
    def maybe(cls, streamer, log=_log):
        """The read-ahead iff NEURAHASH_ST_PREFETCH=1 (and the RAM tier is off), else None."""
        if not _st_prefetch_requested(log=log):
            return None
        if type(streamer)._read is not ExpertStreamer._read:
            # A subclass that overrides `_read` (PrefixStreamer masks non-resident expert rows and
            # keeps its own RAM cache) would have that override silently bypassed by `_read_ra`.
            log("[arc/prefetch] NOT started: %s overrides _read, which the read-ahead would bypass "
                "-- reads stay synchronous" % type(streamer).__name__)
            return None
        if streamer.ram_tier is not None:
            log("[arc/prefetch] NOT started: the host-RAM tier (%s) is on and the two are not "
                "combined -- reads stay synchronous" % _RAM_TIER_ENV)
            return None
        layers = [L for L in streamer.layers if L not in streamer.pin]
        ra = cls(("gate_up", "down"), streamer.device, layers, log=log)
        ra._register_atexit()
        log("[arc/prefetch] ON (%s=1): the predicted next layer is read on 1 background thread "
            "into 1 of %d reusable host buffers (allocated on first use); %d streamed layer(s)"
            % (ENV_ST_PREFETCH, cls.NBUF, len(layers)))
        return ra

    # ------------------------------------------------------------------ consumer side
    def observe(self, L):
        with self._cv:
            return self.predictor.observe(L)

    def schedule(self, L, path, threads):
        """Make (L, path) the one outstanding read-ahead. True if a read was queued."""
        with self._cv:
            if self._disabled is not None:
                return False
            j = self._job
            if j is not None and j.L == L and j.path == path:
                return False
            if self._stop and self._thread is not None and self._thread.is_alive():
                return False                       # a stop timed out; that worker is winding down
            if j is not None:
                self._drop_locked(j)
            if not self._bufs and not self._alloc_locked(path):
                return False
            j = self._Job()
            j.L, j.path, j.threads = L, path, max(1, int(threads))
            j.state, j.buf, j.layout, j.stamp, j.err, j.dropped = "queued", None, None, None, None, False
            self._job = j
            self._ensure_worker_locked()
            self._cv.notify_all()
            return True

    def take(self, L, path):
        """The finished read-ahead of exactly (L, path), or None (a miss: the caller reads it
        synchronously). Re-raises the read's error. Never waits on a read that cannot complete."""
        with self._cv:
            j = self._job
            if j is None or j.L != L or j.path != path:
                if j is not None:
                    self._drop_locked(j)
                self.misses += 1
                return None
            t0 = time.perf_counter()
            while j.state != "done":
                if self._job is not j:             # dropped by another caller while we waited
                    self.misses += 1
                    return None
                w = self._thread
                if w is None or not w.is_alive():
                    self._job = None
                    raise RuntimeError("[arc/prefetch] the read-ahead worker is gone with layer %d "
                                       "still pending -- refusing to wait forever" % L)
                if j.state == "queued" and not self._free and self._n_reading == 0:
                    # Every buffer is lent out and none is being read into, so nothing can ever free
                    # one (a consumer was interrupted between take and give_back): serve it sync.
                    self._drop_locked(j)
                    self.starved += 1
                    self.misses += 1
                    return None
                self._cv.wait(0.5)
            self.wait_s += time.perf_counter() - t0
            if self._job is not j:                 # finished, then dropped by another caller
                self.misses += 1
                return None
            self._job = None
            ok = False
            try:                                   # from here on the buffer is freed on ANY exit
                if j.err is not None:
                    raise j.err
                if j.layout is None:               # did not fit the buffers
                    self.nofit += 1
                else:
                    try:
                        now = _st_file_stamp(path)
                    except OSError:
                        now = None
                    if now == j.stamp:
                        ok = True
                    else:                          # the file changed since it was read ahead
                        self.stale += 1
            finally:
                if not ok:
                    self._free_locked(j)
            if not ok:
                self.misses += 1
                return None
            self.hits += 1
            return j

    def drop_pending(self):
        """Drop the outstanding read-ahead, if any (its buffer comes back when its read lands)."""
        with self._cv:
            if self._job is not None:
                self._drop_locked(self._job)

    def views(self, j):
        """[(name, typed view into the job's buffer)] in FILE order -- `_st_read`'s copy order."""
        import torch
        t = j.buf.t
        return [(name, t[off:off + nbytes].view(getattr(torch, tname)).reshape(shape))
                for name, tname, shape, off, nbytes in j.layout]

    def give_back(self, j):
        with self._cv:
            self._free_locked(j)

    # ------------------------------------------------------------------ lifecycle
    def quiesce(self, timeout=_ST_RA_JOIN_S):
        """Drop any outstanding read-ahead and stop the worker (the next schedule() restarts it).
        On return no read is in flight, unless the join timed out -- which is logged."""
        with self._cv:
            j = self._job
            if j is not None:
                self._drop_locked(j)
            self._stop = True
            self._cv.notify_all()
            w = self._thread
        if w is not None:
            w.join(timeout)
        with self._cv:
            if w is not None and w.is_alive():
                self.log("[arc/prefetch] WARNING: the read-ahead worker did not stop within %.0fs "
                         "(a read is still in flight); leaving it to finish as a daemon" % timeout)
                return False
            self._thread = None
            self._stop = False
            return True

    def close(self):
        stopped = self.quiesce()
        with self._cv:
            if stopped:
                for b in self._bufs:
                    b.mv.release()
                self._bufs, self._free = [], []
            if self._disabled is None:
                self._disabled = "closed"
        self._unregister_atexit()
        self.log("[arc/prefetch] summary: %s" % self.summary())

    def summary(self):
        kind = "none" if self._pinned is None else ("pinned" if self._pinned else "pageable")
        return ("hits=%d misses=%d dropped=%d stale=%d nofit=%d starved=%d | waited %.1fs on reads "
                "in flight | read ahead %.2f GiB | buffers %s"
                % (self.hits, self.misses, self.dropped, self.stale, self.nofit, self.starved, self.wait_s,
                   self.bytes_read / (1 << 30), kind))

    def _register_atexit(self):
        import atexit
        import weakref
        ref = weakref.ref(self)

        def _at_exit():
            ra = ref()
            if ra is not None:
                try:
                    ra.close()
                except Exception:                                             # noqa: BLE001
                    pass
        self._atexit = _at_exit
        atexit.register(_at_exit)

    def _unregister_atexit(self):
        import atexit
        if self._atexit is not None:
            atexit.unregister(self._atexit)
            self._atexit = None

    # ------------------------------------------------------------------ internals (lock held)
    def _free_locked(self, j):
        if j.buf is not None:
            self._free.append(j.buf)
            j.buf = None
            self._cv.notify_all()

    def _drop_locked(self, j):
        if self._job is j:
            self._job = None
        self.dropped += 1
        if j.state == "reading":
            j.dropped = True                       # the worker frees the buffer when the read lands
        elif j.state == "done":
            self._free_locked(j)

    def _alloc_locked(self, path):
        """Two host buffers sized for `path`: PINNED when the target is CUDA (so the H2D is a true
        DMA), else -- or if pinning fails -- pageable. Logs which. False = read-ahead disabled."""
        import torch
        try:
            need = os.path.getsize(path) + _ST_RA_ALIGN * (len(self.names) + 1)
        except OSError:
            return False                           # let the synchronous read raise for this path
        pinned = torch.device(self.device).type == "cuda"
        why = "" if pinned else "the target device is not CUDA"
        bufs = []
        if pinned:
            try:
                for _ in range(self.NBUF):
                    bufs.append(torch.empty(need, dtype=torch.uint8, pin_memory=True))
            except Exception as e:                                            # noqa: BLE001
                bufs, pinned = [], False
                why = "pinning failed: %s" % _ascii(e)[:200]
        if not bufs:
            try:
                for _ in range(self.NBUF):
                    bufs.append(torch.empty(need, dtype=torch.uint8))
            except Exception as e:                                            # noqa: BLE001
                self._disabled = "host buffer allocation failed: %s" % _ascii(e)[:200]
                self.log("[arc/prefetch] DISABLED: %s -- reads stay synchronous" % self._disabled)
                return False
        for t in bufs:
            b = self._Buf()
            b.t, b.mv, b.pinned = t, memoryview(t.numpy()), pinned
            self._bufs.append(b)
            self._free.append(b)
        self._pinned = pinned
        if pinned:
            self.log("[arc/prefetch] %d x %.3f GiB PINNED (page-locked) host buffers (torch's pinned "
                     "allocator rounds each up to %.2f GiB)"
                     % (self.NBUF, need / (1 << 30), (1 << (need - 1).bit_length()) / (1 << 30)))
        else:
            self.log("[arc/prefetch] %d x %.3f GiB PAGEABLE host buffers (%s)"
                     % (self.NBUF, need / (1 << 30), why))
        return True

    def _ensure_worker_locked(self):
        import threading
        w = self._thread
        if w is not None and w.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="st-readahead", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ worker
    def _run(self):
        while True:
            with self._cv:
                while not self._stop and not (self._job is not None
                                              and self._job.state == "queued" and self._free):
                    self._cv.wait()
                if self._stop:
                    return
                j = self._job
                j.buf = self._free.pop()
                j.state = "reading"
                self._n_reading += 1
            layout = stamp = err = None
            try:
                layout, stamp = _st_read_into(j.path, self.names, j.buf.mv, j.threads)
            except BaseException as e:                     # noqa: BLE001 -- re-raised by take()
                err = e
            with self._cv:
                self._n_reading -= 1
                j.layout, j.stamp, j.err, j.state = layout, stamp, err, "done"
                if layout is not None:
                    self.bytes_read += sum(item[4] for item in layout)
                if j.dropped:
                    self._free_locked(j)
                self._cv.notify_all()


# ================================================= lever 1: (b) unbuffered, (a) side-stream H2D, (c) residency
# docs/research/WEB_SURVEY_BIG_GPU_UTILISATION_2026-09-23.md section 0, lever 1. Three options on top of the
# read-ahead above, each OFF unless its env var is exactly "1". With all three unset or "0",
# `ExpertStreamer.__init__` runs the read-ahead's two lines verbatim and NOTHING below is constructed.
#
# (b) NEURAHASH_ST_UNBUFFERED=1 -- every streamed slab's DATA bytes come from FILE_FLAG_NO_BUFFERING reads,
#     one PRIVATE handle per reader thread, so a read no longer depends on the Windows page cache (other
#     programs squeezing RAM evict it; that caused 2-5x stall steps on the 5090, 2026-09-23). Each tensor's
#     range is widened to whole 4096-byte sectors, read into a page-aligned host buffer and then SLICED
#     EXACTLY, so the typed view holds the file's bytes and nothing else. Inside the read-ahead it replaces
#     the worker's reads and the misses' reads; without it, one reusable page-aligned host buffer (pinned
#     for CUDA) serves every read. EXPECT IT TO BE SLOWER wherever the page cache was carrying real hits
#     (4060: ~40 % of slab bytes came from cache at 5.49 GB/s vs 2.55-2.86 GB/s from the drive).
# (a) NEURAHASH_ST_H2D_STREAM=1 (needs NEURAHASH_ST_PREFETCH=1) -- once the read-ahead worker has layer
#     L+1 in its pinned buffer it copies it host->device on a SIDE CUDA stream into freshly allocated
#     device tensors (the second VRAM "slot"), records an event, waits for it on the WORKER thread and
#     frees the host buffer. The consumer then only orders its own stream after that event
#     (`wait_event`) and marks the tensors as used there (`record_stream`), so the compute stream's
#     KERNELS are exactly today's, in today's order; only the two memcpys leave it. Lifetime stays
#     reference-counted by the caching allocator -- nothing is ever overwritten in place, so a tensor
#     some autograd graph still holds can never change under it. Enabled only when the process's
#     VRAM budget covers the declared training peak + pinned layers + margin + TWO extra slabs (the
#     staged one, and one whose free is still waiting on the compute stream's event); otherwise it
#     logs DISABLED with every number and does nothing (the 4060: cap 6.5 GiB, peak 5.77).
# (c) NEURAHASH_ST_RESIDENCY=1 -- logs, at build time, how many MoE layers could stay resident under the
#     process's VRAM cap (torch.cuda.mem_get_info + the DECLARED training peak + margin, reserving (a)'s
#     two slabs when (a) is on) and stores it on the streamer as `residency`. It changes NOTHING: the
#     number is an input for the runner's --pin-layers, which still passes whatever it passes.
#     `st_resident_layers(...)` computes the same number for a caller that wants it before the build.
ENV_ST_UNBUFFERED = "NEURAHASH_ST_UNBUFFERED"
ENV_ST_H2D_STREAM = "NEURAHASH_ST_H2D_STREAM"
ENV_ST_RESIDENCY = "NEURAHASH_ST_RESIDENCY"
ENV_ST_TRAIN_PEAK_GIB = "NEURAHASH_ST_TRAIN_PEAK_GIB"      # measured training peak at 0 pinned layers
ENV_ST_VRAM_MARGIN_GIB = "NEURAHASH_ST_VRAM_MARGIN_GIB"
_ST_TRAIN_PEAK_GIB = 5.8            # 5090 twin workload ~5.8 GiB; 4060 5.77 GiB at cap 6.5 (09-23)
_ST_VRAM_MARGIN_GIB = 1.0           # fragmentation + anything the declared peak did not see
_ST_H2D_SLABS = 2                   # VRAM (a) needs above the peak: the staged slab + one awaiting reuse
_ST_UB_ALIGN = 4096                 # file offset, byte count AND buffer address of every unbuffered read
_ST_UB_CHUNK = 16 << 20             # bytes per ReadFile request (a multiple of _ST_UB_ALIGN)
_ST_SLAB_NAMES = ("gate_up", "down")
_ST_UB_OPS = None                   # the process's unbuffered backend, built on first use


def _st_flag(name, env=None, log=None, tag="arc"):
    """True iff env[name] is exactly "1" (whitespace ignored). Unset/"0" -> False silently; else logged."""
    raw = str((os.environ if env is None else env).get(name, "") or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    if log is not None:
        log("[%s] ignoring %s=%r (want 0 or 1) -- it stays OFF" % (tag, name, _ascii(raw)))
    return False


def _st_l1_requested(env=None):
    """Is ANY lever-1 variable set to something other than unset/"0"? False -> nothing below runs."""
    e = os.environ if env is None else env
    return any(str(e.get(k, "") or "").strip() not in ("", "0")
               for k in (ENV_ST_UNBUFFERED, ENV_ST_H2D_STREAM, ENV_ST_RESIDENCY))


def _st_header(path):
    """`_st_read`'s header parse -- the same reads, checks and messages. Returns (hdr, base)."""
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise EOFError("%s: file is shorter than a safetensors header" % os.path.basename(path))
        n = struct.unpack("<Q", raw_len)[0]
        if not 0 < n <= _ST_HEADER_MAX:
            raise ValueError("%s: safetensors header length %d out of range"
                             % (os.path.basename(path), n))
        hdr = json.loads(fh.read(n).decode("utf-8"))
    return hdr, 8 + n


class _UBUnsupported(Exception):
    """This file, volume or OS cannot take 4096-byte unbuffered reads: the caller reads it buffered."""


class _WinUnbufferedOps(object):
    """CreateFileW(FILE_FLAG_NO_BUFFERING) / positional ReadFile / CloseHandle, through ctypes.

    NO_BUFFERING's rules (Win32 "File Buffering"): the file offset, the byte count and the buffer
    address of every read are multiples of the volume's sector size. Every read here uses 4096 for all
    three, and `open` REFUSES (-> _UBUnsupported -> buffered fallback, counted and logged) a volume
    whose logical sector or buffer-alignment requirement does not divide 4096, or that will not say.
    A synchronous read that reaches end-of-file returns the bytes up to EOF -- a count that need not
    be a sector multiple -- and that is the only short read a disk file produces."""

    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x1
    FILE_SHARE_WRITE = 0x2
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x20000000
    ERROR_HANDLE_EOF = 38
    FILE_STORAGE_INFO_CLASS = 16
    FILE_ALIGNMENT_INFO_CLASS = 17

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct, self.wt = ctypes, wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._create = k32.CreateFileW
        self._create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                 wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
        self._create.restype = ctypes.c_void_p
        self._read = k32.ReadFile
        self._read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        self._read.restype = wintypes.BOOL
        self._close = k32.CloseHandle
        self._close.argtypes = [ctypes.c_void_p]
        self._close.restype = wintypes.BOOL
        self._info = k32.GetFileInformationByHandleEx
        self._info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self._info.restype = wintypes.BOOL

        class _Overlapped(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                        ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", ctypes.c_void_p)]

        class _StorageInfo(ctypes.Structure):
            _fields_ = [("LogicalBytesPerSector", wintypes.ULONG),
                        ("PhysicalBytesPerSectorForAtomicity", wintypes.ULONG),
                        ("PhysicalBytesPerSectorForPerformance", wintypes.ULONG),
                        ("FileSystemEffectivePhysicalBytesPerSectorForAtomicity", wintypes.ULONG),
                        ("Flags", wintypes.ULONG), ("ByteOffsetForSectorAlignment", wintypes.ULONG),
                        ("ByteOffsetForPartitionAlignment", wintypes.ULONG)]

        class _AlignInfo(ctypes.Structure):
            _fields_ = [("AlignmentRequirement", wintypes.ULONG)]

        self._Overlapped, self._StorageInfo, self._AlignInfo = _Overlapped, _StorageInfo, _AlignInfo
        self._invalid = ctypes.c_void_p(-1).value
        self.flags = self.FILE_FLAG_NO_BUFFERING           # what `open` passes (asserted by the tests)

    def geometry(self, h):
        """(logical sector, physical sector for performance, buffer alignment) or None if unknown."""
        ct = self.ct
        si = self._StorageInfo()
        if not self._info(h, self.FILE_STORAGE_INFO_CLASS, ct.byref(si), ct.sizeof(si)):
            return None
        ai = self._AlignInfo()
        amask = (int(ai.AlignmentRequirement)
                 if self._info(h, self.FILE_ALIGNMENT_INFO_CLASS, ct.byref(ai), ct.sizeof(ai)) else 0)
        return int(si.LogicalBytesPerSector), int(si.PhysicalBytesPerSectorForPerformance), amask + 1

    def open(self, path):
        ct = self.ct
        h = self._create(os.path.abspath(path), self.GENERIC_READ,
                         self.FILE_SHARE_READ | self.FILE_SHARE_WRITE, None, self.OPEN_EXISTING,
                         self.flags, None)
        if h is None or h == self._invalid:
            raise ct.WinError(ct.get_last_error())
        try:
            g = self.geometry(h)
            if g is None:
                raise _UBUnsupported("%s: the volume does not report its sector size"
                                     % os.path.basename(path))
            logical, _phys, balign = g
            if logical <= 0 or _ST_UB_ALIGN % logical or balign > _ST_UB_ALIGN or _ST_UB_ALIGN % balign:
                raise _UBUnsupported("%s: logical sector %d / buffer alignment %d do not divide %d"
                                     % (os.path.basename(path), logical, balign, _ST_UB_ALIGN))
        except BaseException:
            self._close(h)
            raise
        return h

    def read_at(self, h, addr, n, off):
        """One positional read of n bytes at file offset `off` into address `addr`. Returns the count."""
        ct = self.ct
        ov = self._Overlapped()
        ov.Offset = off & 0xFFFFFFFF
        ov.OffsetHigh = (off >> 32) & 0xFFFFFFFF
        got = self.wt.DWORD(0)
        if not self._read(h, ct.c_void_p(addr), n, ct.byref(got), ct.byref(ov)):
            err = ct.get_last_error()
            if err == self.ERROR_HANDLE_EOF:
                return 0
            raise ct.WinError(err)
        return int(got.value)

    def close(self, h):
        self._close(h)


def _st_ub_ops():
    """The process's unbuffered backend, or _UBUnsupported (not Windows)."""
    global _ST_UB_OPS
    if _ST_UB_OPS is None:
        if os.name != "nt":
            raise _UBUnsupported("not Windows (FILE_FLAG_NO_BUFFERING is a Win32 flag)")
        _ST_UB_OPS = _WinUnbufferedOps()
    return _ST_UB_OPS


def _st_ub_plan(hdr, base, names, buf_ptr, align=_ST_UB_ALIGN, what="?"):
    """Place each tensor's sector-aligned SUPERSET range in a host buffer. Pure arithmetic.

    Returns (plan, used): plan = [(name, dtype_name, shape, nbytes, fpos, a0, span, region)] in FILE
    order -- the file range [a0, a0 + span) is read to buffer offset `region`, and the tensor's own
    bytes then sit at region + (fpos - a0). a0 and span are multiples of `align`, and so is the
    ADDRESS buf_ptr + region (the first region starts at the first aligned address in the buffer).
    Each tensor gets its own region, so a boundary sector shared by two tensors is simply read twice.
    `used` = bytes of the buffer the plan needs (from its start)."""
    import torch
    order = sorted(names, key=lambda k: hdr[k]["data_offsets"][0])
    at = (-int(buf_ptr)) % align
    plan = []
    for name in order:
        meta = hdr[name]
        tname = _ST_DTYPE.get(meta["dtype"])
        if tname is None or not hasattr(torch, tname):
            raise ValueError("%s: unsupported safetensors dtype %r for tensor %r"
                             % (what, meta["dtype"], name))
        start, end = meta["data_offsets"]
        nbytes = end - start
        fpos = base + start
        a0 = fpos - fpos % align
        span = -(-(fpos + nbytes - a0) // align) * align if nbytes else 0     # an empty tensor reads nothing
        plan.append((name, tname, list(meta["shape"]), nbytes, fpos, a0, span, at))
        at += span
    return plan, at


def _st_ub_chunks(plan, chunk=None):
    """Every region cut into reads of <= chunk bytes: [(file_off, buf_off, length, need)], all three
    of file_off/buf_off/length multiples of the alignment. `need` = how many leading bytes of that read
    MUST arrive (they belong to the file before the tensor's last byte); the rest may lie past EOF."""
    chunk = _ST_UB_CHUNK if chunk is None else int(chunk)
    out = []
    for _name, _tn, _sh, nbytes, fpos, a0, span, region in plan:
        end = fpos + nbytes
        for s in range(0, span, chunk):
            ln = min(chunk, span - s)
            out.append((a0 + s, region + s, ln, max(0, min(ln, end - (a0 + s)))))
    return out


def _st_ub_fill(path, chunks, threads, buf_ptr, ops):
    """Issue every chunk's unbuffered read; returns the byte count each one got.

    One PRIVATE handle per worker (see `_st_read_pieces` for why a shared handle is the bug), workers
    pull chunks from one counter, the first error stops the others from taking new chunks and is
    re-raised here after every worker has closed its handle and exited -- never a hang, never a leak.
    threads <= 1 does the reads on the calling thread, on one handle."""
    import threading
    got = [0] * len(chunks)
    errs, lock, nxt = [], threading.Lock(), [0]

    def _worker():
        h = None
        try:
            h = ops.open(path)                                   # OWN handle
            while True:
                with lock:
                    if errs:
                        return
                    i = nxt[0]
                    nxt[0] += 1
                if i >= len(chunks):
                    return
                off, boff, ln, _need = chunks[i]
                n = 0
                while n < ln:
                    k = ops.read_at(h, buf_ptr + boff + n, ln - n, off + n)
                    if k <= 0:
                        break
                    n += k
                    if k % _ST_UB_ALIGN:                         # only EOF ends off a sector boundary
                        break
                got[i] = n
        except BaseException as e:                               # noqa: BLE001 -- re-raised below
            with lock:
                errs.append(e)
        finally:
            if h is not None:
                try:
                    ops.close(h)
                except Exception:                                # noqa: BLE001
                    pass

    n_workers = max(1, min(int(threads), len(chunks)))
    if not chunks:
        pass
    elif n_workers <= 1:
        _worker()
    else:
        ts = [threading.Thread(target=_worker, name="st-ubread-%d" % i, daemon=True)
              for i in range(n_workers)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    if errs:
        raise errs[0]
    return got


def _st_ub_verify(path, plan, chunks, got):
    """Every byte up to each tensor's end arrived. Otherwise raise the EOFError `_st_read`'s sequential
    arm raises for that file: the FIRST tensor (file order) reaching past EOF, and how much of it exists."""
    eof = None
    for (off, _b, ln, _need), n in zip(chunks, got):
        if n < ln:
            eof = off + n if eof is None else min(eof, off + n)
    if eof is None:
        return
    for name, _tn, _sh, nbytes, fpos, _a0, _span, _region in plan:
        if nbytes and fpos + nbytes > eof:                   # `_st_read` never fails an empty tensor
            raise EOFError("%s: short read for %r -- %d of %d bytes"
                           % (os.path.basename(path), name, max(0, min(nbytes, eof - fpos)), nbytes))


def _st_ub_layout(buf, plan):
    """[(name, dtype_name, shape, offset, nbytes)] -- where each tensor's EXACT bytes now sit in `buf`.

    A tensor whose first byte lands off its element-size boundary (only possible when the file itself
    stores it at such an offset, e.g. bf16 after an odd-length U8 tensor) is moved down by < itemsize
    bytes with memmove, inside its OWN region -- over padding that belongs to no tensor -- so the typed
    view is legal. The bytes are moved, never changed."""
    import ctypes
    import torch
    ptr = buf.data_ptr()
    out = []
    for name, tname, shape, nbytes, fpos, a0, _span, region in plan:
        off = region + (fpos - a0) if nbytes else region     # an empty view sits at an aligned offset
        r = (ptr + off) % getattr(torch, tname).itemsize
        if r and nbytes:
            ctypes.memmove(ptr + off - r, ptr + off, nbytes)
            off -= r
        out.append((name, tname, shape, off, nbytes))
    return out


def _st_ub_need(hdr, base, names):
    """Buffer bytes that always suffice for `_st_ub_plan` of these tensors, whatever the address."""
    return (base + max([hdr[k]["data_offsets"][1] for k in names] or [0])
            + _ST_UB_ALIGN * (2 * len(names) + 2))


def _st_read_into_ub(path, names, buf, threads, ops=None, hdr_base=None, stamp=True):
    """(b): `_st_read_into`'s contract, with every data byte from an unbuffered read into the uint8 host
    tensor `buf` (pinned or not). Returns (layout, stamp) -- layout as `_st_read_into`'s, in FILE order,
    or None when the tensors do not fit `buf`; stamp taken BEFORE the file is opened (None if not
    asked for). Raises _UBUnsupported when the volume cannot take 4096-byte unbuffered reads."""
    ops = _st_ub_ops() if ops is None else ops
    st = _st_file_stamp(path) if stamp else None
    hdr, base = _st_header(path) if hdr_base is None else hdr_base
    plan, used = _st_ub_plan(hdr, base, names, buf.data_ptr(), what=os.path.basename(path))
    if used > buf.numel():
        return None, st
    chunks = _st_ub_chunks(plan)
    got = _st_ub_fill(path, chunks, threads, buf.data_ptr(), ops)
    _st_ub_verify(path, plan, chunks, got)
    return _st_ub_layout(buf, plan), st


def _st_views(buf, layout):
    """[(name, typed view)] into `buf` for a layout, in its (FILE) order -- `_st_read`'s copy order."""
    import torch
    return [(name, buf[off:off + nbytes].view(getattr(torch, tname)).reshape(shape))
            for name, tname, shape, off, nbytes in layout]


def _st_read_ub(path, names, device, threads, buf=None, log=None):
    """(b)'s synchronous read: `_st_read(path, names, device)`'s dict -- the same tensors, the same
    `.to(device)` per tensor in FILE order -- from unbuffered reads into `buf` (a reusable uint8 host
    tensor, used only if big enough) or into a temporary pageable one. The copies are fenced before
    return, so the caller may reuse `buf` at once. Returns (out, unbuffered_bool): False = the volume
    refused 4096-byte unbuffered reads and `_st_read` served it, buffered."""
    import torch
    hdr, base = _st_header(path)
    if buf is None or buf.numel() < _st_ub_need(hdr, base, names):
        buf = torch.empty(_st_ub_need(hdr, base, names), dtype=torch.uint8)
    try:
        layout, _st = _st_read_into_ub(path, names, buf, threads, hdr_base=(hdr, base), stamp=False)
    except _UBUnsupported as e:
        if log is not None:
            log("[arc/unbuffered] %s -- this read is served buffered" % _ascii(e)[:200])
        return _st_read(path, names, device, threads=threads), False
    out = {}
    for name, t in _st_views(buf, layout):
        out[name] = _slab_to_device(t, device)
    _h2d_fence(device)
    return out, True


class _UnbufferedSlabReader(object):
    """(b) with the read-ahead OFF: one reusable page-aligned host buffer -- PINNED for a CUDA target, else
    (or when pinning fails) pageable, and the log says which -- behind a lock, so a second caller waits
    instead of reading into memory a copy is still reading from."""

    def __init__(self, names, device, log=_log):
        import threading
        self.names = tuple(names)
        self.device = device
        self.log = log
        self._lock = threading.Lock()
        self._t = None
        self._pinned = None
        self.reads = self.fallbacks = 0
        self.bytes = 0
        self.seconds = 0.0
        self.closed = False

    def read(self, path, threads):
        with self._lock:
            if self.closed:                          # after close(): today's buffered read, no buffer
                return _st_read(path, self.names, self.device, threads=threads)
            t0 = time.perf_counter()
            hdr, base = _st_header(path)
            need = _st_ub_need(hdr, base, self.names)
            if self._t is None or self._t.numel() < need:
                self._alloc(need)
            try:
                layout, _st = _st_read_into_ub(path, self.names, self._t, threads,
                                               hdr_base=(hdr, base), stamp=False)
            except _UBUnsupported as e:
                self.fallbacks += 1
                if self.fallbacks == 1:
                    self.log("[arc/unbuffered] %s -- served buffered (logged once; counted in the "
                             "summary)" % _ascii(e)[:200])
                return _st_read(path, self.names, self.device, threads=threads)
            out = {}
            for name, t in _st_views(self._t, layout):
                out[name] = _slab_to_device(t, self.device)
            _h2d_fence(self.device)                  # the buffer is reused only after its copies
            self.reads += 1
            self.bytes += sum(item[4] for item in layout)
            self.seconds += time.perf_counter() - t0
            return out

    def _alloc(self, need):
        import torch
        self._t = None
        pinned = torch.device(self.device).type == "cuda"
        why = "" if pinned else "the target device is not CUDA"
        t = None
        if pinned:
            try:
                t = torch.empty(need, dtype=torch.uint8, pin_memory=True)
            except Exception as e:                                            # noqa: BLE001
                pinned, why = False, "pinning failed: %s" % _ascii(e)[:200]
        if t is None:
            t = torch.empty(need, dtype=torch.uint8)
        self._t, self._pinned = t, pinned
        self.log("[arc/unbuffered] 1 x %.3f GiB %s host buffer%s"
                 % (need / (1 << 30), "PINNED (page-locked)" if pinned else "PAGEABLE",
                    "" if pinned else " (%s)" % why))

    def summary(self):
        kind = "none" if self._pinned is None else ("pinned" if self._pinned else "pageable")
        return ("unbuffered reads=%d (%.2f GiB, %.1fs incl. copies) buffered fallbacks=%d | buffer %s"
                % (self.reads, self.bytes / (1 << 30), self.seconds, self.fallbacks, kind))

    def close(self):
        with self._lock:
            if self.closed:
                return
            self._t = None
            self.closed = True
        self.log("[arc/unbuffered] summary: %s" % self.summary())


class _CudaSideStreamStager(object):
    """(a): one slab host->device on a dedicated CUDA stream. `stage` runs on the read-ahead WORKER thread,
    `adopt` on the consumer's."""

    def __init__(self, device):
        import torch
        self.dev = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.dev)

    def stage(self, views):
        """Allocate on the side stream, copy (non_blocking from pinned memory), record an event, and block
        THIS thread until the copies have landed -- only then may the host buffer be read into again."""
        import torch
        out = []
        with torch.cuda.device(self.dev), torch.cuda.stream(self.stream):
            for name, t in views:
                d = torch.empty(tuple(t.shape), dtype=t.dtype, device=self.dev)
                d.copy_(t, non_blocking=True)
                out.append((name, d))
            ev = torch.cuda.Event(blocking=True)
            ev.record(self.stream)
        ev.synchronize()
        return out, ev

    def drain(self):
        """After a failed stage: wait until nothing on the card can still be reading the host buffer."""
        import torch
        torch.cuda.synchronize(self.dev)

    def adopt(self, dev, ev):
        """Order the CONSUMER's current stream after the copies (a stream wait: no kernel, no copy) and tell
        the caching allocator that stream uses the tensors, so their memory is reused only after it."""
        import torch
        cur = torch.cuda.current_stream(self.dev)
        cur.wait_event(ev)
        for _name, d in dev:
            d.record_stream(cur)
        return dev


class _SlabReadAheadL1(_SlabReadAhead):
    """The read-ahead above, unchanged in protocol, plus (b) unbuffered worker reads and (a) device staging.

    A staged job carries `dev` (device tensors) + `ev` (the copy event) and NO host buffer: the worker
    frees it the moment the copies land. Every path that frees a job (drop, stale, error, give_back)
    also drops its device tensors, which were used only on the side stream -- stream order makes that
    safe without an event."""

    class _Job(object):
        __slots__ = ("L", "path", "threads", "state", "buf", "layout", "stamp", "err", "dropped",
                     "dev", "ev", "stager")

        def __init__(self):
            self.dev = self.ev = self.stager = None

    def __init__(self, names, device, layers, log=_log):
        _SlabReadAhead.__init__(self, names, device, layers, log=log)
        self.unbuffered = False
        self._stager = None
        self.ub_reads = self.ub_fallbacks = self.ub_sync = 0
        self.staged = self.adopted = self.stage_fail = 0

    @classmethod
    def maybe(cls, streamer, log=_log, unbuffered=False):
        ra = super(_SlabReadAheadL1, cls).maybe(streamer, log=log)
        if ra is not None and unbuffered:
            ra.unbuffered = True
            log("[arc/unbuffered] ON inside the read-ahead (%s=1): the worker's reads AND the misses' reads "
                "are FILE_FLAG_NO_BUFFERING, %d-byte aligned, one private handle per reader thread"
                % (ENV_ST_UNBUFFERED, _ST_UB_ALIGN))
        return ra

    def attach_stager(self, stager):
        with self._cv:
            self._stager = stager

    # ------------------------------------------------------------------ consumer side
    def adopt(self, j):
        """A staged hit's device tensors, ordered after their copy on the caller's stream. The job gives up
        its references; the caller's slab dict now owns them."""
        dev, ev, stager = j.dev, j.ev, j.stager
        j.dev = j.ev = j.stager = None
        stager.adopt(dev, ev)
        with self._cv:
            self.adopted += 1
        return dev

    def sync_read(self, path):
        """A miss: `_st_read` verbatim, or with (b) the unbuffered read into a temporary host buffer."""
        if not self.unbuffered:
            return _st_read(path, self.names, self.device)
        out, ub = _st_read_ub(path, self.names, self.device, _st_read_threads(), log=self.log)
        with self._cv:
            self.ub_sync += 1
            if not ub:
                self.ub_fallbacks += 1
        return out

    def summary(self):
        s = _SlabReadAhead.summary(self)
        if self.unbuffered:
            s += (" | unbuffered: worker reads %d, miss reads %d, buffered fallbacks %d"
                  % (self.ub_reads, self.ub_sync, self.ub_fallbacks))
        if self._stager is not None or self.staged or self.stage_fail:
            s += (" | side-stream H2D: staged %d, adopted %d, staging failures %d"
                  % (self.staged, self.adopted, self.stage_fail))
        return s

    # ------------------------------------------------------------------ internals (lock held)
    def _free_locked(self, j):
        j.dev = j.ev = j.stager = None
        _SlabReadAhead._free_locked(self, j)

    def _free_host_locked(self, j):
        _SlabReadAhead._free_locked(self, j)

    def _alloc_locked(self, path):
        """v2's two host buffers, sized for the unbuffered layout when (b) is on (each tensor widened to whole
        sectors plus one page of address slack). Same pinned-else-pageable policy, same log lines."""
        if not self.unbuffered:
            return _SlabReadAhead._alloc_locked(self, path)
        import torch
        try:
            hdr, base = _st_header(path)
            need = _st_ub_need(hdr, base, self.names)
        except Exception:                                                     # noqa: BLE001
            return False                           # let the synchronous read raise for this path
        pinned = torch.device(self.device).type == "cuda"
        why = "" if pinned else "the target device is not CUDA"
        bufs = []
        if pinned:
            try:
                for _ in range(self.NBUF):
                    bufs.append(torch.empty(need, dtype=torch.uint8, pin_memory=True))
            except Exception as e:                                            # noqa: BLE001
                bufs, pinned = [], False
                why = "pinning failed: %s" % _ascii(e)[:200]
        if not bufs:
            try:
                for _ in range(self.NBUF):
                    bufs.append(torch.empty(need, dtype=torch.uint8))
            except Exception as e:                                            # noqa: BLE001
                self._disabled = "host buffer allocation failed: %s" % _ascii(e)[:200]
                self.log("[arc/prefetch] DISABLED: %s -- reads stay synchronous" % self._disabled)
                return False
        for t in bufs:
            b = self._Buf()
            b.t, b.mv, b.pinned = t, memoryview(t.numpy()), pinned
            self._bufs.append(b)
            self._free.append(b)
        self._pinned = pinned
        if pinned:
            self.log("[arc/prefetch] %d x %.3f GiB PINNED (page-locked) host buffers, unbuffered layout "
                     "(torch's pinned allocator rounds each up to %.2f GiB)"
                     % (self.NBUF, need / (1 << 30), (1 << (need - 1).bit_length()) / (1 << 30)))
        else:
            self.log("[arc/prefetch] %d x %.3f GiB PAGEABLE host buffers, unbuffered layout (%s)"
                     % (self.NBUF, need / (1 << 30), why))
        return True

    # ------------------------------------------------------------------ worker
    def _read_into(self, j):
        """(layout, stamp, unbuffered?) for job j -- `_st_read_into` verbatim unless (b) is on."""
        if self.unbuffered:
            try:
                layout, stamp = _st_read_into_ub(j.path, self.names, j.buf.t, j.threads)
                return layout, stamp, True
            except _UBUnsupported as e:
                with self._cv:
                    self.ub_fallbacks += 1
                    first = self.ub_fallbacks == 1
                if first:
                    self.log("[arc/unbuffered] %s -- read-ahead served buffered (logged once; counted in "
                             "the summary)" % _ascii(e)[:200])
        layout, stamp = _st_read_into(j.path, self.names, j.buf.mv, j.threads)
        return layout, stamp, False

    def _stage(self, stager, j, layout):
        """(a): the job's bytes onto the device via the side stream. (None, None) = not staged: the
        consumer copies from the host buffer exactly as without (a). A failure disables (a) for good."""
        try:
            return stager.stage(_st_views(j.buf.t, layout))
        except Exception as e:                                                # noqa: BLE001
            with self._cv:
                self.stage_fail += 1
                if self._stager is stager:
                    self._stager = None
            try:
                stager.drain()                     # nothing may still be reading the host buffer
            except Exception:                                                 # noqa: BLE001
                pass
            self.log("[arc/h2d] DISABLED after a staging failure (%s) -- this and every later layer is "
                     "copied on the compute stream, as without (a)" % _ascii(e)[:200])
            return None, None

    def _run(self):
        while True:
            with self._cv:
                while not self._stop and not (self._job is not None
                                              and self._job.state == "queued" and self._free):
                    self._cv.wait()
                if self._stop:
                    return
                j = self._job
                j.buf = self._free.pop()
                j.state = "reading"
                self._n_reading += 1
                stager = self._stager
            layout = stamp = err = dev = ev = None
            ub = False
            try:
                layout, stamp, ub = self._read_into(j)
                if layout is not None and stager is not None and not j.dropped:
                    dev, ev = self._stage(stager, j, layout)
            except BaseException as e:                     # noqa: BLE001 -- re-raised by take()
                err = e
            with self._cv:
                self._n_reading -= 1
                j.layout, j.stamp, j.err, j.state = layout, stamp, err, "done"
                if layout is not None:
                    self.bytes_read += sum(item[4] for item in layout)
                    if ub:
                        self.ub_reads += 1
                if dev is not None:
                    self.staged += 1
                    if not j.dropped:
                        j.dev, j.ev = dev, ev
                        j.stager = stager
                    self._free_host_locked(j)      # the bytes are on the device now
                elif j.dropped:
                    self._free_locked(j)
                self._cv.notify_all()


def _st_ub_refusal(s):
    """Why (b) must not start on streamer s, or None."""
    if type(s)._read is not ExpertStreamer._read:
        return "%s overrides _read, which the unbuffered reader would bypass" % type(s).__name__
    if s.ram_tier is not None:
        return "the host-RAM tier (%s) is on and the two are not combined" % _RAM_TIER_ENV
    try:
        _st_ub_ops()
    except _UBUnsupported as e:
        return _ascii(e)
    except Exception as e:                                                    # noqa: BLE001
        return "the unbuffered backend did not load: %s" % _ascii(e)[:200]
    return None


def _st_env_gib(name, default, env=None, log=None):
    raw = str((os.environ if env is None else env).get(name, "") or "").strip()
    if not raw:
        return float(default), "default"
    try:
        v = float(raw)
        if v >= 0:
            return v, "env:%s" % name
    except ValueError:
        pass
    if log is not None:
        log("[arc/residency] ignoring %s=%r (want GiB >= 0) -- using %.2f" % (name, _ascii(raw), default))
    return float(default), "default"


def _st_smi_free(dev):
    """Device-RESIDENT free bytes per nvidia-smi, or None. `mem_get_info`'s free is NOT that on WDDM:
    docs/research/OLMOE_INSTRUCT_CONTROL_2026-08-30.md:355-361 measured 30.26 GiB (mem_get_info) vs
    8.80 GiB (nvidia-smi) in one process, seconds apart. A query only; CREATE_NO_WINDOW for a console-less
    parent; any failure -> None (the other two bounds still apply, and the log says n/a)."""
    import subprocess
    import torch
    try:
        uuid = str(getattr(torch.cuda.get_device_properties(dev), "uuid", "") or "").strip().lower()
        o = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, encoding="utf-8",
                           errors="replace", timeout=30, creationflags=0x08000000 if os.name == "nt" else 0)
        rows = [[x.strip() for x in ln.split(",")] for ln in (o.stdout or "").splitlines() if ln.strip()]
        rows = [r for r in rows if len(r) == 3]
        pick = [r for r in rows if uuid and r[0].lower().endswith(uuid)] or (rows if len(rows) == 1 else [])
        if not pick:
            return None
        return int((float(pick[0][2]) - float(pick[0][1])) * (1 << 20))
    except Exception:                                                         # noqa: BLE001
        return None


def _st_cuda_mem(device):
    """(free, total, reserved, cap_bytes, cap_source, smi_free) for this process on `device`. Queries only:
    no allocation, no kernel. The cap is the fraction `set_per_process_memory_fraction` applied (the runner
    sets it BEFORE the first CUDA call), else NEURAHASH_VRAM_CAP_GB, else the whole card. smi_free = the
    device-resident free bytes (nvidia-smi) or None."""
    import torch
    dev = torch.device(device)
    if dev.index is None:          # get_per_process_memory_fraction REJECTS a bare "cuda" (4060 pre-flight, 09-23)
        dev = torch.device(dev.type, torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info(dev)
    reserved = int(torch.cuda.memory_reserved(dev))
    cap, src = None, None
    fn = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if fn is not None:
        try:
            frac = float(fn(dev))
            if 0.0 < frac < 1.0:
                cap, src = int(frac * total), "per-process fraction %.4f" % frac
        except Exception:                                                     # noqa: BLE001
            cap = None
    if cap is None:
        raw = str(os.environ.get("NEURAHASH_VRAM_CAP_GB", "") or "").strip()
        try:
            if raw and float(raw) > 0:
                cap, src = int(float(raw) * (1 << 30)), "NEURAHASH_VRAM_CAP_GB"
        except ValueError:
            cap = None
    if cap is None:
        cap, src = int(total), "whole card"
    return int(free), int(total), reserved, int(cap), src, _st_smi_free(dev)


def _st_vram_plan(cap, free, reserved, peak, margin, slab, n_layers, pinned=0, h2d=False, smi_free=None):
    """The arithmetic behind (a) and (c), in bytes. Pure, so the policy is testable without a GPU.

      budget   = min(cap, free + reserved, smi_free + reserved)   what this process may hold on the card
                                             (its own cached blocks count as available, other processes'
                                             use does not; smi_free is skipped when unknown)
      room     = budget - peak - margin      above the DECLARED training peak, measured at 0 pinned
      h2d_ok   = room - pinned*slab >= _ST_H2D_SLABS*slab
      resident = floor((room - (_ST_H2D_SLABS*slab if h2d else 0)) / slab), clamped to [0, n_layers]"""
    budget = min(int(cap), int(free) + int(reserved))
    if smi_free is not None:
        budget = min(budget, int(smi_free) + int(reserved))
    room = budget - int(peak) - int(margin)
    reserve = _ST_H2D_SLABS * int(slab) if h2d else 0
    resident = max(0, min(int(n_layers), (room - reserve) // int(slab))) if slab > 0 else 0
    return {"budget": budget, "room": room, "h2d_need": _ST_H2D_SLABS * int(slab),
            "h2d_ok": bool(slab > 0 and room - int(pinned) * int(slab) >= _ST_H2D_SLABS * int(slab)),
            "resident": int(resident), "h2d_reserve": reserve}


def _st_slab_bytes(pack_dir, layers):
    """Bytes of one streamed slab (gate_up + down), from the first existing pack's HEADER only."""
    for L in layers:
        p = _pack_path(pack_dir, L)
        if os.path.isfile(p):
            hdr, _base = _st_header(p)
            return sum(hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0] for k in _ST_SLAB_NAMES)
    return 0


def st_resident_layers(pack_dir, layers, device="cuda", pinned=0, h2d=None, env=None, log=None):
    """(c) as a callable: the number of MoE layers that can stay resident on `device` (a dict with every
    input is returned as well). Needs CUDA initialised under the runner's cap. Changes nothing."""
    peak, psrc = _st_env_gib(ENV_ST_TRAIN_PEAK_GIB, _ST_TRAIN_PEAK_GIB, env=env, log=log)
    margin, msrc = _st_env_gib(ENV_ST_VRAM_MARGIN_GIB, _ST_VRAM_MARGIN_GIB, env=env, log=log)
    if h2d is None:
        h2d = _st_flag(ENV_ST_H2D_STREAM, env=env) and _st_flag(ENV_ST_PREFETCH, env=env)
    slab = _st_slab_bytes(pack_dir, layers)
    free, total, reserved, cap, csrc, smi = _st_cuda_mem(device)
    v = _st_vram_plan(cap, free, reserved, int(peak * (1 << 30)), int(margin * (1 << 30)), slab,
                      len(list(layers)), pinned=pinned, h2d=h2d, smi_free=smi)
    v.update({"free": free, "total": total, "reserved": reserved, "cap": cap, "cap_source": csrc,
              "smi_free": smi,
              "peak": int(peak * (1 << 30)), "peak_source": psrc, "margin": int(margin * (1 << 30)),
              "margin_source": msrc, "slab": slab, "n_layers": len(list(layers)), "pinned": int(pinned),
              "h2d": bool(h2d)})
    return v["resident"], v


def _st_gib(x):
    return x / float(1 << 30)


def _st_budget_str(v):
    return ("budget %.2f GiB = min(cap %.2f [%s], mem_get_info free %.2f + reserved %.2f, nvidia-smi free %s "
            "+ reserved)" % (_st_gib(v["budget"]), _st_gib(v["cap"]), v["cap_source"], _st_gib(v["free"]),
                             _st_gib(v["reserved"]),
                             "n/a" if v.get("smi_free") is None else "%.2f" % _st_gib(v["smi_free"])))


def _st_h2d_attach(s, ra, log):
    """(a)'s gate: the L1 read-ahead is running, the target is CUDA, and the VRAM budget has room."""
    if not isinstance(ra, _SlabReadAheadL1):
        log("[arc/h2d] NOT started: it stages what the read-ahead read, and the read-ahead is off (%s=1 "
            "needed) -- H2D stays on the compute stream" % ENV_ST_PREFETCH)
        return None
    import torch
    if torch.device(s.device).type != "cuda":
        log("[arc/h2d] NOT started: the target device is not CUDA")
        return None
    try:
        _n, v = st_resident_layers(s.pack_dir, [L for L in s.layers if L not in s.pin], s.device,
                                   pinned=len(s.pin), h2d=False, log=log)
    except Exception as e:                                                    # noqa: BLE001
        log("[arc/h2d] NOT started: the VRAM budget could not be read (%s)" % _ascii(e)[:200])
        return None
    head = v["room"] - v["pinned"] * v["slab"]
    nums = ("%s; declared training peak %.2f GiB [%s] + %d pinned x %.3f GiB; margin %.2f GiB [%s]"
            % (_st_budget_str(v), _st_gib(v["peak"]), v["peak_source"], v["pinned"], _st_gib(v["slab"]),
               _st_gib(v["margin"]), v["margin_source"]))
    if not v["h2d_ok"]:
        log("[arc/h2d] DISABLED: VRAM headroom %.2f GiB < %.2f GiB needed for %d extra slab(s) (%s) -- "
            "H2D stays on the compute stream, exactly as without (a)"
            % (_st_gib(head), _st_gib(v["h2d_need"]), _ST_H2D_SLABS, nums))
        return None
    try:
        stager = _CudaSideStreamStager(s.device)
    except Exception as e:                                                    # noqa: BLE001
        log("[arc/h2d] NOT started: side stream creation failed (%s)" % _ascii(e)[:200])
        return None
    ra.attach_stager(stager)
    log("[arc/h2d] ON (%s=1): the read-ahead worker copies each read-ahead slab host->device on a side "
        "stream; the compute stream only waits on its event | headroom %.2f GiB >= %.2f GiB (%s)"
        % (ENV_ST_H2D_STREAM, _st_gib(head), _st_gib(v["h2d_need"]), nums))
    return stager


def _st_residency_attach(s, h2d_on, log):
    """(c): log the resident-layer number and keep it on the streamer. Reads a header and queries CUDA."""
    import torch
    streamed_or_all = list(s.layers)
    if torch.device(s.device).type != "cuda":
        log("[arc/residency] n/a: the target device is not CUDA")
        return None
    try:
        n, v = st_resident_layers(s.pack_dir, streamed_or_all, s.device, pinned=len(s.pin), h2d=h2d_on,
                                  log=log)
    except Exception as e:                                                    # noqa: BLE001
        log("[arc/residency] n/a: the VRAM budget could not be read (%s)" % _ascii(e)[:200])
        return None
    log("[arc/residency] %d of %d MoE layers could stay resident (room %.2f GiB - (a) reserve %.2f GiB, "
        "/ %.3f GiB per slab; %s; declared training peak %.2f GiB [%s]; margin %.2f GiB [%s]) | this run "
        "pins %d -- the runner decides (--pin-layers); nothing was changed"
        % (n, v["n_layers"], _st_gib(v["room"]), _st_gib(v["h2d_reserve"]), _st_gib(v["slab"]),
           _st_budget_str(v), _st_gib(v["peak"]), v["peak_source"], _st_gib(v["margin"]),
           v["margin_source"], len(s.pin)))
    v["number"] = n
    return v


def _lever1_attach(s, log=_log):
    """Wire (a)/(b)/(c) into a freshly built ExpertStreamer. Called from __init__ ONLY when a lever-1
    variable is set; it then also owns the read-ahead's construction (`_SlabReadAhead.maybe` when neither
    (a) nor (b) is on, so the read-ahead is exactly v2's)."""
    ub = _st_flag(ENV_ST_UNBUFFERED, log=log, tag="arc/unbuffered")
    h2d = _st_flag(ENV_ST_H2D_STREAM, log=log, tag="arc/h2d")
    res = _st_flag(ENV_ST_RESIDENCY, log=log, tag="arc/residency")
    if ub:
        why = _st_ub_refusal(s)
        if why is not None:
            log("[arc/unbuffered] NOT started: %s -- reads stay buffered" % why)
            ub = False
    ra = _SlabReadAheadL1.maybe(s, log=log, unbuffered=ub) if (ub or h2d) else _SlabReadAhead.maybe(s, log=log)
    s._prefetch = ra
    if ra is not None:
        s._read = s._read_l1 if isinstance(ra, _SlabReadAheadL1) else s._read_ra
    elif ub:
        s._ub = _UnbufferedSlabReader(_ST_SLAB_NAMES, s.device, log=log)
        s._read = s._read_ub
        log("[arc/unbuffered] ON (%s=1, read-ahead off): every streamed slab is read FILE_FLAG_NO_BUFFERING, "
            "%d-byte aligned, one private handle per reader thread, into 1 reusable host buffer "
            "(allocated on first use)" % (ENV_ST_UNBUFFERED, _ST_UB_ALIGN))
    stager = _st_h2d_attach(s, ra, log) if h2d else None
    if res:
        s.residency = _st_residency_attach(s, stager is not None, log)


# ================================================= pack codec: NEURAHASH_ST_PACK_CODEC=nhz1 (compressed expert packs)
# docs/research/PACK_CODEC_DESIGN_2026-09-24.md. A pure I/O lever on top of the read-ahead and lever 1: every
# streamed AND pinned slab is DECODED from a losslessly compressed copy of its pack, `<codec dir>/layer_N.nhz`
# (nhz_codec.py: the bf16 high byte -- sign + 7 exponent bits, 2.7 bits of entropy -- raw-deflated with
# Z_HUFFMAN_ONLY, the low byte stored raw, in independent 2 MiB chunks), instead of being read from
# `layer_N.safetensors`. Measured on layer_10 (MEASURE.json): 1.4888x fewer bytes, so a miss reads 0.67x the
# bytes AND the same page cache holds 1.49x the layers (the OS caches the COMPRESSED file).
#
# THE BYTES CANNOT MOVE. The decoded bytes are the pack's bytes exactly: every chunk carries a crc32 of its
# DECODED bytes (seeded by its position, so a swapped chunk fails too), every tensor the sha256 of its source
# bytes (checked on the first decode of each file in this process by default), and any mismatch, truncation,
# unknown version or missing file is a NAMED nhz_codec.NhzError -- never a silent fallback to the raw pack.
# The consumer side is unchanged: the same `.to(device)` per tensor in the pack's data order, the same
# read-ahead protocol and buffer rules, the same (a) staging.
#
# NEURAHASH_ST_PACK_CODEC      unset/"0" -> NOTHING below runs and `_read` is the exact function it always was.
#                              "nhz1"    -> the codec. Anything else is logged "ignoring" and it stays OFF.
# NEURAHASH_ST_PACK_CODEC_DIR  where the .nhz files are; default <pack_dir>/nhz1 -- a SEPARATE directory with
#                              another extension, so no raw-pack lookup or glob can ever pick one up. (On the
#                              5090, D:/_glm_layer_experts is a junction to E:, so the default lands on E:.)
# NEURAHASH_ST_DECODE_THREADS  decode threads per slab, default 4, 1..16. Each has its own file handle and
#                              reads + inflates + interleaves + crc-checks its own chunks, so disk and CPU
#                              overlap. NEURAHASH_ST_READ_THREADS does not apply to .nhz reads.
# NEURAHASH_ST_PACK_VERIFY     "first" (default): sha256 of each tensor on its first decode in this process;
#                              "always": on every decode (slow: ~0.3 s/slab); "crc": the chunk crc32s only.
# With NEURAHASH_ST_PREFETCH=1 the read-ahead WORKER decodes the predicted next layer into its (pinned) host
# buffer while the current layer computes; otherwise every decode is synchronous. NEURAHASH_ST_H2D_STREAM=1
# stages a decoded slab exactly as lever 1 (a) stages a read one. NEURAHASH_ST_UNBUFFERED is NOT combined: the
# codec reads its files BUFFERED on purpose -- the page cache holding compressed bytes is half of the win.
# Refused, logged, reads stay raw (today's wiring runs): a subclass that overrides `_read` (PrefixStreamer),
# and the host-RAM tier. Checked at BUILD, before any step: every managed layer's .nhz exists, has a valid and
# complete header with both slab tensors, was converted from `layer_L.safetensors` (every GLM pack has the same
# size and identical header bytes, so the NAME is what tells layers apart), carries a binding record written by
# a verified conversion or audit that says neither it nor the raw pack changed since, and -- when the raw pack is
# present -- matches it in size, header and a crc spot-check of one chunk per tensor. Anything else raises the
# named error; `convert_packs.py --reverify` re-binds a pack that was touched but not changed. Every decode then
# checks that its file is still the one the build bound (a .nhz replaced mid-run is refused, not served).
ENV_ST_PACK_CODEC = "NEURAHASH_ST_PACK_CODEC"
ENV_ST_PACK_CODEC_DIR = "NEURAHASH_ST_PACK_CODEC_DIR"
ENV_ST_DECODE_THREADS = "NEURAHASH_ST_DECODE_THREADS"
ENV_ST_PACK_VERIFY = "NEURAHASH_ST_PACK_VERIFY"
_ST_CODECS = ("nhz1",)
_ST_DECODE_THREADS = 4
_ST_DECODE_THREADS_MAX = 16
_ST_PACK_VERIFY = {"first": "first", "always": "sha", "crc": "crc"}
_ST_NHZ_SUBDIR = "nhz1"


def _st_codec_requested(env=None):
    """Is NEURAHASH_ST_PACK_CODEC set to anything other than unset/"0"? False -> nothing below runs."""
    return str((os.environ if env is None else env).get(ENV_ST_PACK_CODEC, "") or "").strip() not in ("", "0")


def _st_codec_name(env=None, log=None):
    """The requested codec ("nhz1"), or None: unset/"0" silently, anything else logged and ignored."""
    raw = str((os.environ if env is None else env).get(ENV_ST_PACK_CODEC, "") or "").strip()
    if raw in ("", "0"):
        return None
    if raw in _ST_CODECS:
        return raw
    if log is not None:
        log("[arc/codec] ignoring %s=%r (want 0 or %s) -- it stays OFF, reads stay raw safetensors"
            % (ENV_ST_PACK_CODEC, _ascii(raw), " / ".join(_ST_CODECS)))
    return None


def _st_decode_threads(env=None, log=None):
    """NEURAHASH_ST_DECODE_THREADS as an int in 1.._ST_DECODE_THREADS_MAX; unset or invalid -> 4."""
    raw = str((os.environ if env is None else env).get(ENV_ST_DECODE_THREADS, "") or "").strip()
    if not raw:
        return _ST_DECODE_THREADS
    try:
        return max(1, min(_ST_DECODE_THREADS_MAX, int(raw)))
    except ValueError:
        if log is not None:
            log("[arc/codec] ignoring %s=%r (not an integer) -- %d decode threads"
                % (ENV_ST_DECODE_THREADS, _ascii(raw), _ST_DECODE_THREADS))
        return _ST_DECODE_THREADS


def _st_pack_verify(env=None, log=None):
    """The codec's verify mode for nhz_codec.decode_into: "first" (default), "sha" (always) or "crc"."""
    raw = str((os.environ if env is None else env).get(ENV_ST_PACK_VERIFY, "") or "").strip().lower()
    if not raw:
        return "first"
    if raw in _ST_PACK_VERIFY:
        return _ST_PACK_VERIFY[raw]
    if log is not None:
        log("[arc/codec] ignoring %s=%r (want first / always / crc) -- verify stays 'first'"
            % (ENV_ST_PACK_VERIFY, _ascii(raw)))
    return "first"


def _st_nhz_dir(pack_dir, env=None):
    raw = str((os.environ if env is None else env).get(ENV_ST_PACK_CODEC_DIR, "") or "").strip()
    return raw if raw else os.path.join(pack_dir, _ST_NHZ_SUBDIR)


def _st_nhz_path(nhz_dir, L):
    return os.path.join(nhz_dir, "layer_%d.nhz" % L)


def _st_nhz_module():
    """(nhz_codec, its file, sha256[:16]). Imported ONLY when the codec is on (tools/ and the repo root are on
    sys.path from this module's own prologue); an ImportError here is loud on purpose -- the flag is on."""
    import hashlib
    import nhz_codec as mod
    f = os.path.abspath(mod.__file__)
    with open(f, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()[:16]
    return mod, f, sha


def _st_codec_refusal(s):
    """Why the pack codec must not start on streamer s, or None."""
    if type(s)._read is not ExpertStreamer._read:
        return "%s overrides _read, which the pack codec would bypass" % type(s).__name__
    if s.ram_tier is not None:
        return "the host-RAM tier (%s) is on and the two are not combined" % _RAM_TIER_ENV
    return None


class _NhzSlabCodec(object):
    """The codec's half of ExpertStreamer: synchronous decodes (misses, pinned layers, read-ahead off), the
    read-ahead worker's decode-into, the build-time file check and the counters. Shared by the worker and the
    consumer, so the counters sit behind a lock; decoding itself holds no lock (nhz_codec is thread-safe)."""

    def __init__(self, mod, nhz_dir, names, device, threads, verify, log=_log):
        import threading
        self.mod = mod
        self.nhz_dir = nhz_dir
        self.names = tuple(names)
        self.device = device
        self.threads = max(1, int(threads))
        self.verify = verify
        self.log = log
        self._lock = threading.Lock()
        self._need = {}                            # path -> host bytes for read_into (from its header)
        self._bound = {}                           # path -> the file's (size, mtime_ns) when the build checked it
        self._raw = {}                             # path -> (raw pack path, its stamp at the build) when present
        self.sync_reads = self.ra_reads = 0
        self.bytes_in = self.bytes_out = 0
        self.decode_s = 0.0
        self.sha_checked = 0
        self.tier_bypass = 0
        self.checked = None
        self.closed = False

    def path(self, L):
        return _st_nhz_path(self.nhz_dir, L)

    def _need_of(self, h):
        return sum(h.entry(n).nbytes for n in self.names) + _ST_RA_ALIGN * (len(self.names) + 1)

    def check_layers(self, layers, pack_dir):
        """Every managed layer: .nhz present, header valid and complete, both slab tensors present with a
        torch-known dtype, and BOUND to its layer: converted from `layer_L.safetensors` (every GLM pack has the
        same size and identical header bytes, so the name is what tells layers apart) and, when the raw pack is
        present, matching it in size, header and a crc spot-check of one chunk per tensor (4 MiB read per GLM
        layer). A layer with no raw pack to compare against is logged as UNBOUND. Raises nhz_codec's NAMED
        errors (NhzMissingError / NhzFormatError / NhzVersionError / NhzTruncatedError / NhzStaleError)."""
        layers = list(layers)
        missing, unbound, raw_checked, raw_bytes, stored, decoded = [], [], 0, 0, 0, 0
        for L in layers:
            p = self.path(L)
            if not os.path.isfile(p):
                missing.append(L)
                continue
            h = self.mod.read_header(p)            # NhzFormatError / NhzVersionError / NhzTruncatedError
            for name in self.names:
                e = h.entry(name)                  # NhzFormatError when absent
                if _ST_DTYPE.get(e.dtype) is None:
                    raise self.mod.NhzFormatError("%s: tensor %r has dtype %r, which this reader cannot view"
                                                  % (p, name, e.dtype))
            raw = _pack_path(pack_dir, L)
            if os.path.isfile(raw):
                raw_bytes += self.mod.check_source(h, raw)          # name, size, header, binding, spot-check
                raw_checked += 1
                # the reference is the stamp the BINDING vouches for, not a fresh stat (a change between the
                # two would otherwise become the reference and never be seen)
                self._raw[p] = (raw, tuple(self.mod.read_binding(h)["raw_stamp"]))
            else:
                rec = self.mod.check_unbound(h, os.path.basename(raw))   # right layer, bound, unchanged since
                unbound.append("%d:%s" % (L, str(rec.get("file_sha256") or "?")[:16]))
                self._raw[p] = (raw, None)                          # a raw pack appearing later is refused
            self._bound[p] = h.stamp                               # a later decode must see THIS file
            self._need[p] = self._need_of(h)
            stored += h.stored_bytes(self.names)
            decoded += h.decoded_bytes(self.names)
        if missing:
            raise self.mod.NhzMissingError(
                2, "%s=nhz1 but %d of %d layer file(s) are missing in %s (layers %s) -- convert them "
                   "(scratchpad/packcodec_0924/convert_packs.py) or unset %s"
                   % (ENV_ST_PACK_CODEC, len(missing), len(layers), self.nhz_dir,
                      ",".join(str(x) for x in missing[:12]) + ("..." if len(missing) > 12 else ""),
                      ENV_ST_PACK_CODEC))
        if unbound:
            self.log("[arc/codec] WARNING: %d of %d layer(s) have no raw pack in %s to compare their .nhz with: "
                     "bound by name + binding record only; recorded source sha256 per layer (compare with an "
                     "independent manifest): %s" % (len(unbound), len(layers), pack_dir,
                                                    " ".join(unbound[:12]) + (" ..." if len(unbound) > 12 else "")))
        self.checked = {"layers": len(layers), "raw_checked": raw_checked, "raw_spot_bytes": raw_bytes,
                        "unbound": len(unbound), "stored": stored, "decoded": decoded}
        return self.checked

    def note_tier_bypass(self, L):
        """A host-RAM tier attached AFTER construction serves layer L through today's raw `_read` (the tier and
        the codec are never combined): counted, and logged the first time, so a run cannot believe it decoded."""
        with self._lock:
            self.tier_bypass += 1
            first = self.tier_bypass == 1
        if first:
            self.log("[arc/codec] a host-RAM tier was attached after construction: its reads (layer %d first) take "
                     "the RAW path, not the codec -- counted in the summary" % L)

    def need(self, path):
        n = self._need.get(path)
        if n is None:
            n = self._need_of(self.mod.read_header(path))
            self._need[path] = n
        return n

    def _note(self, st, ra):
        with self._lock:
            if ra:
                self.ra_reads += 1
            else:
                self.sync_reads += 1
            self.bytes_in += st["bytes_in"]
            self.bytes_out += st["bytes_out"]
            self.decode_s += st["seconds"]
            self.sha_checked += st["sha_checked"]

    def _same_file(self, path, h):
        """Every decode serves the file the BUILD bound, and only while the raw pack it was bound to is unchanged:
        a .nhz replaced mid-run (even a valid one) is refused, and so is a decode after the RAW pack changed (or
        vanished) -- the raw path would now serve other bytes, so the codec may not keep serving the old ones.
        One os.stat of the raw pack per decode (microseconds)."""
        want = self._bound.get(path)
        if want is None or h.stamp != want:
            raise self.mod.NhzStaleError("%s changed since this streamer was built (stamp %s, built %s) -- "
                                         "refusing to serve it; rebuild the streamer"
                                         % (os.path.basename(path), list(h.stamp), want and list(want)))
        self.raw_same(path)

    def raw_same(self, path):
        """The raw pack this .nhz was bound to is still the one the build bound (its stamp), or -- for a layer
        built with no raw pack -- still absent. Called before a decode, after a synchronous decode, and by the
        consumer right before it serves a read-ahead hit, so no window between a check and the delivery is
        left where a changed raw pack goes unseen."""
        bound_raw = self._raw.get(path)
        if bound_raw is None:
            return
        raw, st = bound_raw
        try:
            now = _st_file_stamp(raw)
        except OSError:
            now = None
        if st is None:
            if now is not None:
                raise self.mod.NhzStaleError("%s appeared after this streamer was built without it: the raw path "
                                             "would serve ITS bytes now -- refusing; --reverify, then rebuild" % raw)
        elif now != st:
            raise self.mod.NhzStaleError("%s changed since this streamer was built (stamp %s, built %s): the "
                                         "raw path would serve other bytes now -- refusing to serve the old "
                                         ".nhz; reconvert or --reverify, then rebuild"
                                         % (raw, now and list(now), list(st)))

    def _tname(self, path, name, dt):
        import torch
        tname = _ST_DTYPE.get(dt)
        if tname is None or not hasattr(torch, tname):
            raise ValueError("%s: unsupported safetensors dtype %r for tensor %r"
                             % (os.path.basename(path), dt, name))
        return tname

    def read(self, path):
        """`_st_read(path, names, device)`'s contract from the .nhz: each tensor decoded into its OWN heap
        buffer (no mapping, no aliasing), then `.to(device)` per tensor in the pack's data (FILE) order."""
        import torch
        h = self.mod.read_header(path)
        self._same_file(path, h)
        plan = []
        for name in h.order(self.names):
            e = h.entry(name)
            tname = self._tname(path, name, e.dtype)
            plan.append((name, e, tname, torch.empty(e.nbytes, dtype=torch.uint8)))   # owns its heap storage
        views = [(name, memoryview(buf.numpy())) for name, _e, _t, buf in plan]
        try:
            st = self.mod.decode_into(path, views, threads=self.threads, verify=self.verify, hdr=h)
        finally:
            for _n, mv in views:
                mv.release()                       # drop the numpy alias before .to()
        self.raw_same(path)                        # the raw pack did not move during the decode either
        self._note(st, False)
        out = {}
        for name, e, tname, buf in plan:
            out[name] = buf.view(getattr(torch, tname)).reshape(e.shape).to(self.device)
        return out

    def read_into(self, path, mv, threads):
        """`_st_read_into`'s contract from the .nhz: (layout, stamp) -- layout [(name, dtype_name, shape,
        buffer_offset, nbytes)] in FILE order at _ST_RA_ALIGN offsets of `mv`, or None when it does not fit;
        stamp taken BEFORE the file is opened, so a file replaced after it can only ever be refused."""
        try:
            stamp = _st_file_stamp(path)
        except FileNotFoundError as e:             # removed since the build: the NAMED error, via take()
            raise self.mod.NhzMissingError(e.errno, "%s: no such nhz file (removed since the build)" % path) from None
        h = self.mod.read_header(path)
        self._same_file(path, h)
        lay, total = h.layout(self.names, _ST_RA_ALIGN)
        out = [(name, self._tname(path, name, dt), shape, off, nbytes) for name, dt, shape, off, nbytes in lay]
        if total > len(mv):
            return None, stamp
        targets = [(name, mv[off:off + nbytes]) for name, _t, _s, off, nbytes in out]
        try:
            st = self.mod.decode_into(path, targets, threads=threads, verify=self.verify, hdr=h)
        finally:
            for _n, sub in targets:
                sub.release()
        self._note(st, True)
        return out, stamp

    def summary(self):
        with self._lock:
            n = self.sync_reads + self.ra_reads
            return ("decoded %d slab(s) (%d synchronous, %d in the read-ahead worker) | read %.2f GiB of .nhz -> "
                    "%.2f GiB of bf16 (%.4fx) | decode %.1fs (%.2f GB/s of bf16 out) | %d tensor sha256 check(s) | "
                    "%d decode thread(s), verify %s%s"
                    % (n, self.sync_reads, self.ra_reads, self.bytes_in / (1 << 30), self.bytes_out / (1 << 30),
                       (self.bytes_out / float(self.bytes_in)) if self.bytes_in else 0.0, self.decode_s,
                       (self.bytes_out / 1e9 / self.decode_s) if self.decode_s > 0 else 0.0, self.sha_checked,
                       self.threads, self.verify,
                       " | RAW reads through a late-attached RAM tier: %d" % self.tier_bypass if self.tier_bypass else ""))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.log("[arc/codec] summary: %s" % self.summary())


class _SlabReadAheadNHZ(_SlabReadAheadL1):
    """The read-ahead with its worker DECODING the predicted next layer's .nhz into the (pinned) host buffer.
    Protocol, predictor, buffer safety, stale-file refusal and (a) staging are lever 1's, unchanged; only
    what fills the buffer (`_read_into`), how big it must be (`_alloc_locked`) and a miss (`sync_read`) differ."""

    def __init__(self, names, device, layers, log=_log):
        _SlabReadAheadL1.__init__(self, names, device, layers, log=log)
        self.codec = None

    @classmethod
    def maybe(cls, streamer, codec, log=_log):
        ra = super(_SlabReadAheadL1, cls).maybe(streamer, log=log)
        if ra is not None:
            ra.codec = codec
            log("[arc/codec] the read-ahead worker DECODES the predicted next layer's .nhz (%d decode thread(s)) "
                "into its host buffer while the current layer computes; a miss decodes synchronously"
                % codec.threads)
        return ra

    def sync_read(self, path):
        """A miss: the codec's synchronous decode (`_st_read`'s contract)."""
        return self.codec.read(path)

    def summary(self):
        s = _SlabReadAheadL1.summary(self)
        if self.codec is not None:
            s += " | .nhz decodes in the worker: %d" % self.codec.ra_reads
        return s

    def _alloc_locked(self, path):
        """v2's two host buffers, sized for the DECODED slab (the .nhz file is smaller than what it decodes to).
        Same pinned-else-pageable policy and log lines as v2's."""
        import torch
        try:
            need = self.codec.need(path)
        except Exception:                                                     # noqa: BLE001
            return False                           # let the synchronous decode raise for this path
        pinned = torch.device(self.device).type == "cuda"
        why = "" if pinned else "the target device is not CUDA"
        bufs = []
        if pinned:
            try:
                for _ in range(self.NBUF):
                    bufs.append(torch.empty(need, dtype=torch.uint8, pin_memory=True))
            except Exception as e:                                            # noqa: BLE001
                bufs, pinned = [], False
                why = "pinning failed: %s" % _ascii(e)[:200]
        if not bufs:
            try:
                for _ in range(self.NBUF):
                    bufs.append(torch.empty(need, dtype=torch.uint8))
            except Exception as e:                                            # noqa: BLE001
                self._disabled = "host buffer allocation failed: %s" % _ascii(e)[:200]
                self.log("[arc/prefetch] DISABLED: %s -- reads stay synchronous" % self._disabled)
                return False
        for t in bufs:
            b = self._Buf()
            b.t, b.mv, b.pinned = t, memoryview(t.numpy()), pinned
            self._bufs.append(b)
            self._free.append(b)
        self._pinned = pinned
        if pinned:
            self.log("[arc/prefetch] %d x %.3f GiB PINNED (page-locked) host buffers, decoded .nhz layout "
                     "(torch's pinned allocator rounds each up to %.2f GiB)"
                     % (self.NBUF, need / (1 << 30), (1 << (need - 1).bit_length()) / (1 << 30)))
        else:
            self.log("[arc/prefetch] %d x %.3f GiB PAGEABLE host buffers, decoded .nhz layout (%s)"
                     % (self.NBUF, need / (1 << 30), why))
        return True

    def _read_into(self, j):
        """(layout, stamp, False): job j's slab decoded into its buffer with j.threads decode threads."""
        layout, stamp = self.codec.read_into(j.path, j.buf.mv, j.threads)
        return layout, stamp, False


def _codec_attach(s, log=_log):
    """Wire the pack codec into a freshly built ExpertStreamer. Called from __init__ ONLY when
    NEURAHASH_ST_PACK_CODEC is set to something other than unset/"0". True = the codec owns the reads; False =
    it declined (an unknown value or a refusal, both logged) and __init__ runs today's wiring, unchanged. A
    codec that is ON but finds a file missing, corrupt, of another version or stale RAISES here, at build."""
    name = _st_codec_name(log=log)
    if name is None:
        return False
    why = _st_codec_refusal(s)
    if why is not None:
        log("[arc/codec] NOT started: %s -- reads stay raw safetensors" % why)
        return False
    mod, mfile, msha = _st_nhz_module()
    threads = _st_decode_threads(log=log)
    verify = _st_pack_verify(log=log)
    cx = _NhzSlabCodec(mod, _st_nhz_dir(s.pack_dir), _ST_SLAB_NAMES, s.device, threads, verify, log=log)
    chk = cx.check_layers(s.layers, s.pack_dir)
    ub = _st_flag(ENV_ST_UNBUFFERED, log=log, tag="arc/unbuffered")
    h2d = _st_flag(ENV_ST_H2D_STREAM, log=log, tag="arc/h2d")
    res = _st_flag(ENV_ST_RESIDENCY, log=log, tag="arc/residency")
    if ub:
        log("[arc/unbuffered] NOT started: the pack codec (%s=%s) reads its compressed files BUFFERED on purpose "
            "-- the page cache holding compressed bytes is half of what it buys" % (ENV_ST_PACK_CODEC, name))
    s._codec = cx
    ra = _SlabReadAheadNHZ.maybe(s, cx, log=log)
    s._prefetch = ra
    s._read = s._read_nhz
    s._ra_schedule = s._nhz_schedule               # prefetch_hint() schedules the .nhz too
    log("[arc/codec] ON (%s=%s): %d MoE layer(s) DECODED from %s -- %.2f GiB stored for %.2f GiB of slabs "
        "(%.4fx) | %d decode thread(s) | verify %s | %d raw pack(s) bound to their .nhz (name, size, header, "
        "%.1f MiB content spot-check), %d unbound | nhz_codec %s sha256 %s | read-ahead %s"
        % (ENV_ST_PACK_CODEC, name, chk["layers"], cx.nhz_dir, chk["stored"] / (1 << 30),
           chk["decoded"] / (1 << 30), (chk["decoded"] / float(chk["stored"])) if chk["stored"] else 0.0,
           threads, verify, chk["raw_checked"], chk["raw_spot_bytes"] / float(1 << 20), chk["unbound"], mfile, msha,
           "ON" if ra is not None else "off"))
    stager = _st_h2d_attach(s, ra, log) if h2d else None
    if res:
        s.residency = _st_residency_attach(s, stager is not None, log)
    return True


# ============================================================================== STAGE -- PACK
def stage_pack(opts):
    """Fuse the checkpoint's per-expert tensors into one file per MoE layer."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(CONFIG_DIR)
    idx = json.load(open(os.path.join(CONFIG_DIR, "model.safetensors.index.json")))["weight_map"]
    n_exp, I, H = int(cfg.n_routed_experts), int(cfg.moe_intermediate_size), int(cfg.hidden_size)
    want_bytes = n_exp * 2 * I * H * 2 + n_exp * H * I * 2
    os.makedirs(opts.pack_dir, exist_ok=True)
    layers = _moe_layers(cfg)
    _log("[arc/pack] %d MoE layers -> %s | %.3f GiB each, %.1f GiB total"
         % (len(layers), opts.pack_dir, want_bytes / (1 << 30), len(layers) * want_bytes / (1 << 30)))
    t_all = time.time()
    made = 0
    for L in layers:
        out = _pack_path(opts.pack_dir, L)
        if os.path.exists(out) and abs(os.path.getsize(out) - want_bytes) < (1 << 20):
            continue
        t0 = time.time()
        gate_up = torch.empty((n_exp, 2 * I, H), dtype=torch.bfloat16)
        down = torch.empty((n_exp, H, I), dtype=torch.bfloat16)
        need = {}
        for E in range(n_exp):
            for part in ("gate_proj", "up_proj", "down_proj"):
                k = "model.layers.%d.mlp.experts.%d.%s.weight" % (L, E, part)
                need.setdefault(idx[k], []).append((k, E, part))
        for shard, entries in need.items():
            with safe_open(os.path.join(CONFIG_DIR, shard), framework="pt", device="cpu") as sf:
                for k, E, part in entries:
                    t = sf.get_tensor(k)
                    if part == "gate_proj":
                        gate_up[E, :I].copy_(t)
                    elif part == "up_proj":
                        gate_up[E, I:].copy_(t)
                    else:
                        down[E].copy_(t)
        save_file({"gate_up": gate_up, "down": down}, out)
        del gate_up, down
        made += 1
        _log("[arc/pack] layer %2d -> %s (%.1fs, %d shard(s))"
             % (L, os.path.basename(out), time.time() - t0, len(need)))
    _log("[arc/pack] done: %d built, %d already present, %.0fs total"
         % (made, len(layers) - made, time.time() - t_all))
    return 0


# ================================================================= T1 -- the host-RAM slab tier
# MEASURED 2026-07-28 (docs/research/UNSLOTH_AND_STREAMING_TIER_2026-07-28.md:100-135): the
# production `safe_open -> CUDA` path moves 1.88 GB/s cold and only 2.43 GB/s with a WARM page
# cache, while a plain host-RAM -> CUDA copy moves 17.09 GB/s pageable / 54.26 GB/s pinned. The 22x
# gap survives with the disk entirely out of the picture, so the cost is not the drive: it is
# re-opening the file, re-parsing the safetensors header and copying out of an mmap, 34x per
# backward. Holding the MATERIALISED tensor in host RAM skips all of it.
#
# The bytes are bit-identical by construction -- same tensors, different vehicle. That is asserted,
# not assumed: tests/test_glm_arc_ram_tier.py sha256s what `_read` hands the consumer on both paths.
#
# Default OFF (NEURAHASH_SD_RAM_TIER unset -> `ExpertStreamer.ram_tier is None`, no tier object, no
# allocation, byte-for-byte today's code path). This file ships nothing to miners, but the
# no-regression directive is absolute.
_RAM_TIER_ENV = "NEURAHASH_SD_RAM_TIER"                 # "1"/"true"/"yes"/"on" -> tier on
_RAM_TIER_GB_ENV = "NEURAHASH_SD_RAM_TIER_GB"           # explicit budget, still clamped by margin
_RAM_TIER_MARGIN_GB_ENV = "NEURAHASH_SD_RAM_TIER_MARGIN_GB"
_RAM_TIER_MARGIN_GB = 8.0                               # never leave the box with less than this


def _host_avail_bytes():
    """Physical RAM currently available. Fail-soft: 0 means 'unknown'.

    0 collapses the budget to 0, so every slab falls back to the disk path -- i.e. an unknown host
    degrades to exactly today's behaviour rather than guessing and swapping the box.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:                                                  # Windows, no third-party dependency
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = _MemStatus()
        ms.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return int(ms.ullAvailPhys)
    except Exception:
        pass
    return 0


def _ram_tier_budget_bytes(avail_bytes, margin_bytes, explicit_gb=None):
    """How many bytes the tier may hold. Pure function so the policy is testable without RAM pressure.

    An explicit budget is still clamped to (available - margin) when availability is known: a
    fat-fingered env var must not be able to page this box into the ground (memory
    `vram-cap-live-verified` -- the same box has already been crashed twice by memory pressure).
    """
    headroom = max(0, int(avail_bytes) - int(margin_bytes))
    if explicit_gb is None:
        return headroom
    want = max(0, int(float(explicit_gb) * (1 << 30)))
    return min(want, headroom) if int(avail_bytes) > 0 else want


def _slab_to_device(t, device):
    """Hand the consumer its OWN tensor, never the cached one.

    In production the cache is CPU and `device` is cuda, so `.to()` copies anyway. When they match
    (CPU-only tests) a bare `.to()` returns the SAME object, and the model would then hold a
    reference INTO the tier -- one in-place write and every later touch of that layer is silently
    corrupted. `copy=True` closes that hole on both paths for the price of nothing in production.
    """
    import torch
    return t.to(torch.device(device), copy=True)


class HostRamSlabTier:
    """Materialised streamed slabs held in pageable host RAM, admitted until the budget is spent.

    NO EVICTION, deliberately. The access pattern is a cyclic scan over all 34 streamed layers, and
    LRU on a cyclic scan larger than the cache has a 100% miss rate -- it would evict precisely the
    entry needed next, every time, and pay a disk read for it. A fixed resident subset gives a
    deterministic K/34 hit rate instead of 0. Overflow layers fall back to the disk path unchanged.

    PAGEABLE, not pinned: T1 in the source doc explicitly says pageable first (17.09 GB/s, already
    7-9x today's 1.88-2.43). Pinned memory is non-swappable and is a separate, later lever (T2).
    """

    def __init__(self, budget_bytes=None, margin_bytes=None, log=_log):
        self.log = log
        self.slabs = {}                       # path -> {name: cpu tensor}
        self.declined = set()                 # paths that did not fit; never retried (stable subset)
        self.used = 0
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self._full_logged = False
        if budget_bytes is not None:
            # Programmatic caller (tests, or a script that has already done its own sizing).
            self.budget = max(0, int(budget_bytes))
            self.avail = None
        else:
            if margin_bytes is None:
                margin_bytes = float(os.environ.get(_RAM_TIER_MARGIN_GB_ENV, "")
                                     or _RAM_TIER_MARGIN_GB) * (1 << 30)
            self.avail = _host_avail_bytes()
            gb = os.environ.get(_RAM_TIER_GB_ENV, "").strip()
            self.budget = _ram_tier_budget_bytes(self.avail, margin_bytes,
                                                 explicit_gb=(gb or None))
            self.log("[arc/ramtier] ON: budget %.2f GiB (host avail %.2f GiB, margin %.2f GiB)"
                     % (self.budget / (1 << 30), self.avail / (1 << 30),
                        margin_bytes / (1 << 30)))

    @classmethod
    def maybe(cls, log=_log):
        """The tier iff the env flag is set, else None. Flag off => nothing is constructed."""
        if os.environ.get(_RAM_TIER_ENV, "").strip().lower() not in ("1", "true", "yes", "on"):
            return None
        return cls(log=log)

    def admit(self, path, names):
        """Return {name: cpu tensor} for `path`, or None if the caller must read it from disk.

        Sizing uses the file size BEFORE reading anything, so a slab that cannot fit costs one
        `stat`, not a wasted 1.125 GiB read.
        """
        hit = self.slabs.get(path)
        if hit is not None:
            self.hits += 1
            return hit
        if path in self.declined:
            self.misses += 1
            return None
        try:
            need = os.path.getsize(path)
        except OSError:
            self.declined.add(path)                 # missing/unreadable -> let the disk path raise
            self.misses += 1
            return None
        if self.used + need > self.budget:
            self.declined.add(path)
            self.misses += 1
            if not self._full_logged:
                self._full_logged = True
                self.log("[arc/ramtier] budget spent: %s" % self.summary())
            return None
        try:
            slab = self._load(path, names)
        except Exception as e:                      # fail-soft: never take the run down over a cache
            self.declined.add(path)
            self.misses += 1
            self.log("[arc/ramtier] load failed for %s (%s) -- streaming it from disk instead"
                     % (os.path.basename(path), _ascii(e)))
            return None
        self.slabs[path] = slab
        self.used += sum(t.numel() * t.element_size() for t in slab.values())
        self.loads += 1
        self.hits += 1
        return slab

    @staticmethod
    def _load(path, names):
        """Read once, MATERIALISED.

        Materialisation is load-bearing, not defensive. The old `safe_open().get_tensor().clone()`
        needed the explicit `.clone()` because `get_tensor` returned a lazily-mapped VIEW, so
        without it this "RAM tier" would have held nothing but page-table entries -- no speed-up,
        and the same mmap-laziness family that produced the ~1 ULP honest hash-rejects in memory
        `safetensors-mmap-recompute-determinism`. `_st_read` materialises by construction (it
        plain-reads the byte range and never maps the file), so the property now holds without a
        clone -- and without ever paying for the mapping. What the tier serves is still unaffected
        by a later overwrite of the file, which tests/test_glm_arc_ram_tier.py asserts.
        """
        return _st_read(path, names, "cpu")

    def summary(self):
        held, streamed = len(self.slabs), len(self.declined)
        return ("holds %d slab(s) in host RAM (%.2f of %.2f GiB budget), %d slab(s) stream from disk"
                % (held, self.used / (1 << 30), self.budget / (1 << 30), streamed))

    def log_summary(self):
        self.log("[arc/ramtier] %s | hits=%d misses=%d loads=%d"
                 % (self.summary(), self.hits, self.misses, self.loads))


# =================================================================== the streaming expert machinery
class ExpertStreamer:
    """Manage MoE expert weight RESIDENCY around the stock forward. Never touches the math.

    CRITICAL (measured 2026-07-27): this NEVER constructs an experts module. transformers wraps the
    experts class with `use_experts_implementation` (integrations/moe.py:548-565), whose forward
    dispatches on `self.config._experts_implementation` -- 'grouped_mm' here, which accumulates in
    fp32 via reshape+sum. A separately-constructed module built from a config that never passed
    through a model has that field set to None and silently falls back to the class's naive bf16
    `index_add_` loop instead. The two disagree by up to 0.25 in logits; the selftest caught it.
    So we reuse the model's OWN module objects and swap only their weight tensors.
    """

    def __init__(self, model, cfg, pack_dir, pin, device, layers=None, ram_tier=None):
        import torch
        self.torch = torch
        self.model, self.cfg, self.pack_dir, self.device = model, cfg, pack_dir, device
        self.pin = set(pin)
        self.cache = {}
        # Layers made resident by `hold()` for one layer-major sweep. Tracked SEPARATELY from
        # `pin` so `release()` can never evict a pinned layer: `cache` is the union of the two and
        # is the only thing `_pre`/`_post` consult, which is what keeps those two hooks unchanged.
        self._held = set()
        self.handles = []
        self.layers = list(layers) if layers is not None else _moe_layers(cfg)
        self._managed = set(self.layers)       # only these have a pack file to stream
        self.bytes_streamed = 0
        self.bytes_from_ram = 0
        # T1 host-RAM tier. `ram_tier=None` (the default) means "let the env decide", and with
        # NEURAHASH_SD_RAM_TIER unset `maybe()` returns None -> no object, no allocation, and
        # `_read` takes the exact disk path it took before this lever existed.
        self.ram_tier = ram_tier if ram_tier is not None else HostRamSlabTier.maybe()
        # Resolve layer modules through a SNAPSHOT, never through `model.model.layers`, because
        # `_capture_preamble` temporarily rebinds that very attribute to a 1-element ModuleList.
        # Today all captures finish before the sweep starts, so indexing the live attribute would
        # happen to work -- but the driver's own docstring invites lazy per-group capture as the
        # fix for the retained mask, and under that change `release(L)` would raise IndexError
        # mid-sweep AFTER the weights were already popped from `cache`, leaving the layer both
        # uncached and still attached. The snapshot makes the hazard structurally impossible.
        self._layer_mods = list(model.model.layers)
        for L in self.layers:
            e = model.model.layers[L].mlp.experts
            if not hasattr(e, "config"):
                raise SystemExit("[arc] FATAL: layer %d experts module is %s and carries no `config` "
                                 "-- it is not the dispatched module and its forward would differ."
                                 % (L, type(e).__name__))
            # Drop the Parameters so plain CUDA tensors can be attached/detached freely
            # (assigning a Tensor over a registered Parameter raises TypeError).
            e._parameters.pop("gate_up_proj", None)
            e._parameters.pop("down_proj", None)
            e.gate_up_proj = None
            e.down_proj = None
            dl = model.model.layers[L]
            self.handles.append(dl.register_forward_pre_hook(self._pre(L)))
            self.handles.append(dl.register_forward_hook(self._post(L)))
        # Read-ahead (NEURAHASH_ST_PREFETCH=1, `_SlabReadAhead`). Unset/"0" -> None: no object, no
        # thread, no buffer, and `_read` stays the exact function below. When on, `_read_ra` is
        # bound over `_read` on THIS instance, so every caller (hold, _pre, preload_pinned) gets it.
        # Lever 1 (NEURAHASH_ST_UNBUFFERED / _H2D_STREAM / _RESIDENCY, `_lever1_attach`): all three unset
        # or "0" -> the else-branch, i.e. the read-ahead's two lines exactly as before, and nothing else.
        # Pack codec (NEURAHASH_ST_PACK_CODEC=nhz1, `_codec_attach`): unset/"0" -> the first condition is False
        # before anything is looked at, so the two branches below run exactly as before. On, it owns the reads
        # -- unless it declines (logged), and then the two branches below run exactly as before as well.
        if _st_codec_requested() and _codec_attach(self):
            pass
        elif _st_l1_requested():
            _lever1_attach(self)
        else:
            self._prefetch = _SlabReadAhead.maybe(self)
            if self._prefetch is not None:
                self._read = self._read_ra

    def _read(self, L):
        # THIS IS THE #80 CRASH SITE. It used to be `with safe_open(...) as sf: sf.get_tensor(...)`,
        # which memory-maps the pack; PYTHONFAULTHANDLER caught the fourth access violation exactly
        # on that call, inside a gradient-checkpoint recompute on the autograd worker thread. It now
        # goes through `_st_read`, which plain-reads the byte range -- see the long rationale at
        # `_st_read` for the three measurements that condemned the mapping.
        #
        # HISTORY, so the mmap is not reintroduced as an "optimisation":
        # DO NOT turn the per-read open into a persistent handle. MEASURED AND REVERTED 2026-08-01.
        # The theory was sound-sounding: the old code rebuilt a fresh mapping of a 1.2 GB pack on
        # EVERY read, ~45 layers per unit over hundreds of units, so thousands of map/unmap cycles.
        # Caching one open handle per layer took the process from ~11-12 GiB RSS to 46.51 GiB and
        # left the box with 1.0 GB of 61.6 GB free, with the pagefile at 14.3 GB. The claim that
        # the mapped pages were "clean, file-backed and evictable" was WRONG in practice: touched
        # pages stay in the working set for as long as the mapping is open, and it was exactly the
        # per-read close that released them.
        # A bounded LRU of handles does not rescue it either -- each unit sweeps all 46 layers in
        # order, which is the pathological case for LRU, so any cache smaller than 46 restores the
        # full churn while a cache of 46 restores the 46 GiB.
        # Eliminating mmap altogether -- what `_st_read` now does -- was the only remaining form of
        # the idea that could work, and it bounds RSS by construction: the bytes are read, handed
        # over, and freed, with no mapping left behind to accumulate a working set.
        p = _pack_path(self.pack_dir, L)
        if self.ram_tier is not None:
            slab = self.ram_tier.admit(p, ("gate_up", "down"))
            if slab is not None:
                gu = _slab_to_device(slab["gate_up"], self.device)
                dn = _slab_to_device(slab["down"], self.device)
                # `bytes_streamed` keeps meaning "bytes DELIVERED to the consumer" so every existing
                # reader of it (scratchpad/b3_*.py) reports the same number it always did; the
                # RAM-served share is broken out separately rather than folded in.
                self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
                self.bytes_from_ram += gu.numel() * 2 + dn.numel() * 2
                return gu, dn
        slab = _st_read(p, ("gate_up", "down"), self.device)
        gu, dn = slab["gate_up"], slab["down"]
        self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
        return gu, dn

    def _read_ra(self, L):
        """`_read` with NEURAHASH_ST_PREFETCH=1 -- same bytes, same copies, same order.

        HIT: the read-ahead of L is finished (or is waited for); each tensor is copied out of the
        host buffer with the same `.to(device)` `_st_read` issues, in the same file order. MISS
        (nothing, or another layer, was read ahead): `_st_read`, i.e. today's read, verbatim.
        The predicted successor's read is started BEFORE this layer's copies on a hit (the other
        buffer is free) and AFTER the synchronous read on a miss (so the read on the critical path
        keeps the disk to itself). `bytes_streamed` keeps meaning "bytes delivered".
        """
        if L in self.pin or self.ram_tier is not None:
            # Pinned layers (preload_pinned, once per run) and a RAM tier attached AFTER
            # construction (scripts assign `streamer.ram_tier = HostRamSlabTier(...)`) take
            # today's `_read`, verbatim -- the tier and the read-ahead are never combined.
            if self.ram_tier is not None:
                self._prefetch.drop_pending()
            return ExpertStreamer._read(self, L)
        ra = self._prefetch
        p = _pack_path(self.pack_dir, L)
        nxt = ra.observe(L)
        job = None
        try:
            job = ra.take(L, p)
            if job is None:
                slab = _st_read(p, ("gate_up", "down"), self.device)
                self._ra_schedule(nxt)
            else:
                self._ra_schedule(nxt)
                slab = {}
                for name, t in ra.views(job):
                    slab[name] = _slab_to_device(t, self.device)
                _h2d_fence(self.device)
        finally:
            if job is not None:
                ra.give_back(job)                  # on every exit, exceptions included
        gu, dn = slab["gate_up"], slab["down"]
        self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
        return gu, dn

    def _read_l1(self, L):
        """`_read_ra` with lever 1: (b) its reads are unbuffered, (a) a hit may already be on the device.

        Same order of events as `_read_ra`. HIT: schedule the successor, then deliver -- from the device if
        the worker staged it (the caller's stream waits on the copy's event; no copy, no kernel on it), else
        from the host buffer with `_st_read`'s `.to(device)` per tensor in file order. MISS: the synchronous
        read (`_st_read` verbatim, or with (b) its unbuffered twin), then schedule. Same bytes delivered,
        same `bytes_streamed`."""
        if L in self.pin or self.ram_tier is not None:
            if self.ram_tier is not None:
                self._prefetch.drop_pending()
            return ExpertStreamer._read(self, L)
        ra = self._prefetch
        p = _pack_path(self.pack_dir, L)
        nxt = ra.observe(L)
        job = None
        try:
            job = ra.take(L, p)
            if job is None:
                slab = ra.sync_read(p)
                self._ra_schedule(nxt)
            else:
                self._ra_schedule(nxt)
                if job.dev is not None:
                    slab = dict(ra.adopt(job))
                else:
                    slab = {}
                    for name, t in ra.views(job):
                        slab[name] = _slab_to_device(t, self.device)
                    _h2d_fence(self.device)
        finally:
            if job is not None:
                ra.give_back(job)                  # on every exit, exceptions included
        gu, dn = slab["gate_up"], slab["down"]
        self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
        return gu, dn

    def _read_ub(self, L):
        """`_read` with NEURAHASH_ST_UNBUFFERED=1 and the read-ahead off: the same tensors through the same
        `.to(device)` per tensor in file order, the bytes read unbuffered into one reusable host buffer.
        Pinned layers and a RAM tier take today's `_read`, verbatim."""
        if L in self.pin or self.ram_tier is not None:
            return ExpertStreamer._read(self, L)
        slab = self._ub.read(_pack_path(self.pack_dir, L), _st_read_threads())
        gu, dn = slab["gate_up"], slab["down"]
        self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
        return gu, dn

    def _read_nhz(self, L):
        """`_read` with NEURAHASH_ST_PACK_CODEC=nhz1: the slab is DECODED from `<codec dir>/layer_L.nhz` -- the
        same tensors, byte for byte (every chunk crc32-checked, sha256 on first decode), delivered through the
        same `.to(device)` per tensor in the pack's file order. Pinned layers, and every layer with the
        read-ahead off, decode synchronously. With the read-ahead on the order of events is `_read_l1`'s: a hit
        schedules the successor first, then delivers (from the device if (a) staged it); a miss decodes
        synchronously, then schedules. A RAM tier attached after construction takes today's `_read`, verbatim --
        logged the first time and counted, so the run's log never claims a decode that did not happen.
        `bytes_streamed` keeps meaning "bytes delivered"; the codec's own counters say what was read."""
        cx = self._codec
        if self.ram_tier is not None:
            if self._prefetch is not None:
                self._prefetch.drop_pending()
            cx.note_tier_bypass(L)
            return ExpertStreamer._read(self, L)
        p = cx.path(L)
        ra = self._prefetch
        if ra is None or L in self.pin:
            slab = cx.read(p)
        else:
            nxt = ra.observe(L)
            job = None
            try:
                job = ra.take(L, p)
                if job is None:
                    slab = ra.sync_read(p)
                    self._nhz_schedule(nxt)
                else:
                    cx.raw_same(p)                 # the raw pack may have moved since the worker checked it
                    self._nhz_schedule(nxt)
                    if job.dev is not None:
                        slab = dict(ra.adopt(job))
                    else:
                        slab = {}
                        for name, t in ra.views(job):
                            slab[name] = _slab_to_device(t, self.device)
                        _h2d_fence(self.device)
            finally:
                if job is not None:
                    ra.give_back(job)              # on every exit, exceptions included
        gu, dn = slab["gate_up"], slab["down"]
        self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
        return gu, dn

    def _nhz_schedule(self, L):
        """`_ra_schedule` for the codec: the predicted layer's .nhz, decoded with the codec's thread count."""
        if L is None or L in self.cache or L not in self._managed or self._prefetch is None:
            return False
        return self._prefetch.schedule(L, self._codec.path(L), self._codec.threads)

    def _ra_schedule(self, L):
        if L is None or L in self.cache or L not in self._managed:
            return False
        return self._prefetch.schedule(L, _pack_path(self.pack_dir, L), _st_read_threads())

    def prefetch_hint(self, L):
        """Caller hint: layer L is the NEXT `_read`. Starts its read now (dropping any other
        outstanding read-ahead). Returns False, doing nothing, when the read-ahead is off."""
        if self._prefetch is None:
            return False
        return self._ra_schedule(L)

    def preload_pinned(self):
        for L in sorted(self.pin):
            self.cache[L] = self._read(L)
        return len(self.cache)

    def hold(self, L):
        """Make layer L resident for a WHOLE layer-major sweep instead of one forward.

        This is the other half of the loop inversion. Inverting the loop alone changes nothing,
        because `_post` nulls a non-pinned layer the instant its forward returns -- so a
        layer-major sweep would still re-read the same 1.125 GiB once per batch. `hold` puts the
        layer into `cache`, which is exactly the predicate `_pre` ("reuse") and `_post` ("do not
        null") already test, so the residency change costs ZERO edits to either hook and the
        batch-major path stays byte-identical.

        Returns True if this call performed the read (i.e. the layer was not already resident).
        """
        if L not in self._managed:
            return False                       # dense layer (0..first_k_dense_replace-1): no pack
        if L in self.cache:
            return False                       # pinned, or already held by this sweep
        self.cache[L] = self._read(L)
        self._held.add(L)
        return True

    def release(self, L):
        """Drop a layer made resident by `hold`. A PINNED layer is never released."""
        if L not in self._held:
            return False
        self.cache.pop(L, None)
        self._held.discard(L)
        exp = self._layer_mods[L].mlp.experts
        exp.gate_up_proj = None
        exp.down_proj = None
        return True

    def release_all(self):
        for L in list(self._held):
            self.release(L)
        if self._prefetch is not None:
            self._prefetch.quiesce()               # nothing left in flight; restarts on next read

    def _pre(self, L):
        def fn(module, args):
            exp = module.mlp.experts
            if L in self.cache:
                gu, dn = self.cache[L]
            else:
                gu, dn = self._read(L)
            exp.gate_up_proj = gu
            exp.down_proj = dn
            return None
        return fn

    def _post(self, L):
        def fn(module, args, output):
            if L not in self.cache:
                exp = module.mlp.experts
                exp.gate_up_proj = None
                exp.down_proj = None
            return None
        return fn

    def close(self):
        # Insurance only: the driver holds and releases one layer per iteration inside a
        # try/finally, so `_held` is empty between layers even if a sweep raises. This makes a
        # leaked hold impossible to survive teardown rather than merely unlikely.
        self.release_all()
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.ram_tier is not None:
            self.ram_tier.log_summary()
        if self._prefetch is not None:
            self._prefetch.close()
        ub = getattr(self, "_ub", None)            # lever 1 (b) with the read-ahead off
        if ub is not None:
            ub.close()
        cx = getattr(self, "_codec", None)         # the pack codec (NEURAHASH_ST_PACK_CODEC=nhz1)
        if cx is not None:
            cx.close()


def build_trunk_model(device, log=_log):
    """The frozen trunk, resident, with the model's OWN (still meta) experts modules intact.

    This is piece_loader.build_partial_model's steps 1/3/5 minus the expert handling -- it must NOT
    call build_partial_model([]), because that swaps every experts module for a `_DeadExperts`
    placeholder (piece_loader.py:384-386) whose forward returns zeros and which carries none of the
    `use_experts_implementation` dispatch. piece_loader's own helpers are reused, not modified.
    """
    import torch
    from accelerate import init_empty_weights
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteForCausalLM, Glm4MoeLiteRotaryEmbedding)
    import piece_loader as pl

    t0 = time.time()
    man = pl.load_manifest(SHARD_DIR, require_pieces=[])
    cfg = pl._resolve_config(SHARD_DIR, man, CONFIG_DIR)
    with init_empty_weights(include_buffers=False):
        model = Glm4MoeLiteForCausalLM(cfg)
    tp = os.path.join(man["shard_dir"], "pieces", "trunk.safetensors")
    # `assign=True` makes these tensors BE the model's parameters for the life of the model, so
    # they must not be mmap views: through the old safe_open path every later touch of a trunk
    # parameter could fault a page in from a mapping the `with` block had already left (the same
    # 0xC0000005 exposure as `_st_read`'s header comment, just with a 4.02 GiB lifetime).
    trunk_sd = _st_read(tp, None, device,
                        keep=lambda k: not pl._is_dead_mtp_key(k, cfg.num_hidden_layers))  # strip_mtp
    model.load_state_dict(trunk_sd, strict=False, assign=True)
    del trunk_sd
    model.model.rotary_emb = Glm4MoeLiteRotaryEmbedding(cfg, device=torch.device(device))
    log("[arc] trunk resident on %s in %.1fs | experts_implementation=%r"
        % (device, time.time() - t0, cfg._experts_implementation))
    return model, cfg


def build_streaming_model(pack_dir, pin_layers, device, log=_log):
    """Trunk resident + every MoE layer's experts streamed. Returns (model, cfg, streamer)."""
    import torch
    model, cfg = build_trunk_model(device, log=log)
    streamer = ExpertStreamer(model, cfg, pack_dir, pin_layers, device)
    # Nothing except the (popped) expert weights may still be on meta -- a leftover meta parameter
    # is a silently-broken layer, not a warning.
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_left:
        raise SystemExit("[arc] BLOCKED: %d parameter(s) still on meta after the trunk load, e.g. %r"
                         % (len(meta_left), meta_left[:5]))
    n = streamer.preload_pinned()
    model.eval()
    torch.cuda.synchronize()
    # RECORD, so every CE this model produces can name its own residency (issue #125).
    set_ce_pin(list(pin_layers) if not isinstance(pin_layers, int) else list(range(1, pin_layers + 1)))
    log("[arc] streaming ready: %d MoE layers, %d pinned resident, %d streamed per pass | "
        "GPU allocated %.2f GiB"
        % (len(streamer.layers), n, len(streamer.layers) - n,
           torch.cuda.memory_allocated(0) / (1 << 30)))
    return model, cfg, streamer


# ============================================================================== STAGE -- SELFTEST
def stage_selftest(opts):
    """EQUIVALENCE GATE: same weights, two mechanisms -> logits must be bit-identical."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteNaiveMoe
    from safetensors import safe_open

    _cap_vram()
    dev = "cuda"
    cfg = AutoConfig.from_pretrained(CONFIG_DIR)
    I = int(cfg.moe_intermediate_size)
    cfg.num_hidden_layers = 2                 # layer 0 dense + layer 1 MoE
    cfg.num_nextn_predict_layers = 0
    if getattr(cfg, "num_local_experts", None) is None:
        cfg.num_local_experts = cfg.n_routed_experts
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG_DIR, config=cfg, dtype=torch.bfloat16, local_files_only=True).to(dev).eval()
    _log("[arc/selftest] 2-layer reference model on %s" % dev)

    g = torch.Generator(device="cpu").manual_seed(7)
    ids = torch.randint(0, int(cfg.vocab_size), (3, 48), generator=g).to(dev)
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits.clone()

    # ---- the packed file must equal the reference model's own fused parameter, bit-exact ----
    exp0 = model.model.layers[1].mlp.experts
    with safe_open(_pack_path(opts.pack_dir, 1), framework="pt", device=dev) as sf:
        pgu, pdn = sf.get_tensor("gate_up"), sf.get_tensor("down")
    same_gu = torch.equal(pgu, exp0.gate_up_proj.data)
    same_dn = torch.equal(pdn, exp0.down_proj.data)
    _log("[arc/selftest] packed layer 1 == from_pretrained fused param: gate_up=%s down=%s"
         % (same_gu, same_dn))
    if not (same_gu and same_dn):
        raise SystemExit("[arc/selftest] BLOCKED: packed layer-1 experts differ from the reference "
                         "model's own weights -- the pack stage is wrong.")

    # ---- swap in the REAL streaming mechanism (the same ExpertStreamer bench uses) and re-run ----
    _log("[arc/selftest] model experts_implementation=%r" % (model.config._experts_implementation,))
    streamer = ExpertStreamer(model, model.config, opts.pack_dir, pin=(), device=dev, layers=[1])
    with torch.no_grad():
        got = model(input_ids=ids, use_cache=False).logits.clone()
    streamer.close()

    exact = torch.equal(ref, got)
    maxdiff = float((ref.float() - got.float()).abs().max())
    _log("[arc/selftest] streamed vs stock logits: bit-identical=%s max|d|=%.3g shape=%s"
         % (exact, maxdiff, tuple(ref.shape)))
    _log("[arc/selftest] freed after forward: gate_up_proj is %r"
         % (model.model.layers[1].mlp.experts.gate_up_proj,))
    if not exact:
        raise SystemExit("[arc/selftest] BLOCKED: streaming changes the model output (max|d|=%.3g)"
                         % maxdiff)
    _log("[arc/selftest] PASS -- expert streaming is output-equivalent to stock residency.")
    return 0


# ============================================================================== STAGE -- FOLD (CPU)
def build_args(pieces):
    """The coordinator's own parser subset, so every default is the one run 5 ran with."""
    import sharddiloco_glm_contributor as N
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-active-slots", dest="max_active_slots", type=int,
                    default=(int(os.environ["NEURAHASH_SD_MAX_ACTIVE"])
                             if os.environ.get("NEURAHASH_SD_MAX_ACTIVE") else 16))
    ap.add_argument("--eval-chunk", type=int,
                    default=int(os.environ.get("NEURAHASH_GLM_EVAL_CHUNK", "64")))
    ap.add_argument("--coord-data-dir", default="D:/glm_daily/coord")
    ap.add_argument("--probe-seed", dest="probe_seed", type=int, default=None)
    N.add_common_args(ap)
    # The lane comes LAST so a spec may retarget it; argparse keeps the last occurrence, and the
    # default spec repeats BASE_ARGV's own values, so the parsed namespace is unchanged for run 5.
    url, token = spec_lane()
    return ap.parse_args(BASE_ARGV + ["--pieces", pieces, "--url", url, "--token", token])


def _param_digests(model):
    """sha256 per PARAMETER TENSOR. Run before and after the fold, the diff answers a question
    `model_root` cannot: model_root hashes only the 8 REGISTERED slots, so it would not notice a
    fold that also moved a router, a norm, or an unregistered expert row. The `ce` stage rebuilds
    arm 2 by PATCHING the snapshotted rows instead of re-folding, and that shortcut is only sound
    if the fold's blast radius is exactly those rows -- so measure it, don't assume it."""
    import hashlib
    import torch
    out = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            b = p.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
            out[n] = hashlib.sha256(b).hexdigest()
    return out


def snapshot_rows(model, coords):
    """Clone the fused expert rows for `coords` straight off the model (bf16, CPU)."""
    import torch
    out = {}
    with torch.no_grad():
        for (L, E) in coords:
            exp = model.model.layers[L].mlp.experts
            exp = exp.base if hasattr(exp, "base") else exp
            out[(L, E)] = {"gate_up": exp.gate_up_proj[E].detach().clone().cpu(),
                           "down": exp.down_proj[E].detach().clone().cpu()}
    return out


def stage_fold(opts):
    import torch
    import sharddiloco_harness as H
    import sharddiloco_glm_contributor as N

    _log("[arc/fold] torch %s CUDA_VISIBLE_DEVICES=%r"
         % (torch.__version__, os.environ.get("CUDA_VISIBLE_DEVICES")))
    t0 = time.time()
    log_spec("arc/fold")
    SLOTS, EVENTS, ROOT = spec_slots(), spec_events(), spec_root()
    # PREFIX FOLD (--max-records N): fold only the FIRST N accepted records in event order. The
    # composition question -- where between 1 and 84 accepted deltas the model stops improving and
    # starts breaking -- is the same fold measured at many N, and event order is the order the
    # COORDINATOR itself applied them, so a prefix is a real intermediate state of this campaign
    # rather than a re-ordering of it. Every record is still FETCHED (that is seconds) so the lane
    # cross-check below stays unconditional; only the fold loop is truncated.
    n_max = int(getattr(opts, "max_records", 0) or 0)
    prefix = bool(n_max and n_max < len(EVENTS))
    args = build_args(spec_pieces())
    G = N._G()
    N.use_glm_lane_names()
    slots = N.parse_slots(args.slots)
    lane = H.ContentLane(*spec_lane())

    model, cfg, seq = N.build_node_model(args, log=_log)
    claimable = N.node_claimable_coords(args)
    # max_active bounds how many coordinates may be ADMITTED live; it is not part of the model's
    # identity. model_root hashes every entry of host.slots regardless of `active`, and read_slot
    # reads straight off the model, so eviction -- which is how the live coordinator stays under its
    # own --max-active-slots while its registry grows -- provably cannot move the root. A replay
    # therefore has to hold the WHOLE registry at once, and capping it at the coordinator's live
    # value would only make register() raise part-way through a list the coordinator itself built.
    max_active = max(int(args.max_active_slots or 0), len(SLOTS)) or None
    host = G.GlmExpertLaneHost(model, cfg, slots, claimable=claimable, max_active=max_active)
    N.bind_campaign_id(host, spec_campaign())
    root0 = N.model_root(host)
    I = int(cfg.moe_intermediate_size)
    _log("[arc/fold] loaded pieces=%s claimable=%d max_active=%s root=%s.. in %.1fs"
         % (spec_pieces(), 0 if claimable is None else len(claimable), max_active, root0[:12],
            time.time() - t0))

    pristine = snapshot_rows(model, SLOTS)
    sd_before = _param_digests(model)

    for want_idx, (L, E) in enumerate(SLOTS):
        if host.index_of(L, E) is None:
            host.register(L, E)
        got = host.index_of(L, E)
        if int(got) != want_idx:
            raise SystemExit("[arc/fold] FATAL: (L%d,E%d) landed at slot %d, the spec has %d"
                             % (L, E, got, want_idx))
    if list(host.slots) != SLOTS:
        raise SystemExit("[arc/fold] FATAL: slot list %r != the spec's %r" % (host.slots, SLOTS))
    _log("[arc/fold] registry = %d slots %s" % (len(host.slots), list(host.slots)))

    t_man = time.time()
    man = lane.manifest()
    pref = spec_accepted_prefix()
    recs, missing = [], []
    for ev in EVENTS:
        nm = "%s%d" % (pref, ev)
        if nm not in man:
            missing.append(nm)
            continue
        recs.append((ev, lane.get_json(man[nm]["sha256"])))
    if missing:
        raise SystemExit("[arc/fold] FATAL: %d accepted record(s) absent: %s"
                         % (len(missing), missing[:5]))
    _log("[arc/fold] fetched %d records in %.0fs" % (len(recs), time.time() - t_man))

    expect_root = str(recs[-1][1]["model_root"])
    if expect_root != ROOT:
        raise SystemExit("[arc/fold] FATAL: lane's final root %s != expected %s"
                         % (expect_root, ROOT))

    use = recs[:n_max] if prefix else recs
    if prefix:
        _log("[arc/fold] PREFIX FOLD: --max-records %d -> folding the first %d of %d accepted "
             "records (events %d..%d); the remaining %d are fetched and cross-checked but NOT folded"
             % (n_max, len(use), len(recs), use[0][0], use[-1][0], len(recs) - len(use)))

    folded, skipped = 0, []
    for ev, rec in use:
        before = N.model_root(host)
        ok, why, _rej = N._fold_accepted_checked(host, lane, rec, None, -1, log=None)
        if not ok:
            skipped.append((ev, why))
            _log("[arc/fold] event %d NOT folded (%s)" % (ev, why))
            continue
        if N.model_root(host) == before:
            skipped.append((ev, "ok-but-no-op"))
            _log("[arc/fold] event %d folded ok but model_root DID NOT MOVE" % ev)
            continue
        folded += 1
    root1 = N.model_root(host)

    # A PREFIX fold has NO externally recorded target root: the coordinator's `model_root` at
    # accepted event N was hashed over the slot registry AS IT STOOD THEN (append-only, still
    # growing), while a replay must hold the WHOLE registry at once (see max_active above), so the
    # two roots are not comparable and comparing them anyway would be a gate that lies. What
    # replaces it is the OTHER half of what a full fold proves -- every requested record folded AND
    # moved the root -- checked HERE, before the root gate, and aborting the same way.
    if prefix:
        ok = (folded == len(use) and root1 != root0)
        _log("[arc/fold] PREFIX GATE: folded %d of the first %d accepted records | root %s.. -> "
             "%s.. (moved=%s) -> %s"
             % (folded, len(use), root0[:12], root1[:12], root1 != root0, "PASS" if ok else "FAIL"))
        if not ok:
            raise SystemExit("[arc/fold] BLOCKED: prefix fold of the first %d accepted record(s) "
                             "folded only %d of them (skipped=%r); refusing to emit deltas."
                             % (len(use), folded, skipped))
        ROOT = root1        # the prefix's ACHIEVED root -- recorded as provenance, never as a check

    gate = (root1 == ROOT)
    _log("[arc/fold] folded %d/%d records" % (folded, len(use)))
    _log("[arc/fold] GATE post-fold model_root")
    _log("[arc/fold]   got  = %s" % root1)
    _log("[arc/fold]   want = %s%s"
         % (ROOT, "  (PREFIX: the achieved root, gated above -- NOT a recorded target)"
            if prefix else ""))
    _log("[arc/fold]   -> %s" % ("PASS" if gate else "FAIL"))
    if not gate:
        raise SystemExit("[arc/fold] BLOCKED: fold does not reproduce campaign %s's final root; "
                         "refusing to emit deltas. folded=%d/%d skipped=%r"
                         % (spec_campaign(), folded, len(use), skipped))

    foldedrows = snapshot_rows(model, SLOTS)
    sd_after = _param_digests(model)
    changed_tensors = sorted(n for n in sd_after if sd_after[n] != sd_before.get(n))
    _log("[arc/fold] FOLD BLAST RADIUS: %d of %d parameter tensors changed -> %r"
         % (len(changed_tensors), len(sd_after), changed_tensors[:6]))

    changed = {}
    for c in SLOTS:
        dg = foldedrows[c]["gate_up"].float() - pristine[c]["gate_up"].float()
        dd = foldedrows[c]["down"].float() - pristine[c]["down"].float()
        n = float(dg.norm() ** 2 + dd.norm() ** 2) ** 0.5
        base = float(pristine[c]["gate_up"].float().norm() ** 2
                     + pristine[c]["down"].float().norm() ** 2) ** 0.5
        changed[c] = dict(delta_l2=n, base_l2=base, rel=(n / base if base else float("nan")),
                          moved=bool(n > 0))
        _log("[arc/fold] (L%d,E%d) delta_l2=%.6g base_l2=%.6g rel=%.3e moved=%s"
             % (c[0], c[1], n, base, changed[c]["rel"], changed[c]["moved"]))
    _log("[arc/fold] coordinates whose weights actually MOVED: %d of %d"
         % (sum(1 for c in changed if changed[c]["moved"]), len(SLOTS)))

    payload = dict(
        campaign=spec_campaign(), pieces=spec_pieces(),
        accepted_events=[ev for ev, _r in use], spec=dict(SPEC),
        prefix_records=(len(use) if prefix else None),
        slots=[list(c) for c in SLOTS], root_pristine=root0, root_folded=root1,
        root_expected=(None if prefix else ROOT), gate_pass=gate, folded=folded, requested=len(use),
        skipped=skipped, moe_intermediate=I, changed_tensors=changed_tensors,
        n_param_tensors=len(sd_after),
        coord_stats={"%d:%d" % c: changed[c] for c in changed},
        pristine={"%d:%d" % c: pristine[c] for c in pristine},
        foldedrows={"%d:%d" % c: foldedrows[c] for c in foldedrows},
        wall_s=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(opts.deltas)), exist_ok=True)
    torch.save(payload, opts.deltas)
    _log("[arc/fold] wrote %s (%.1f MB) in %.0fs"
         % (opts.deltas, os.path.getsize(opts.deltas) / 1e6, time.time() - t0))
    assert not torch.cuda.is_initialized(), "CUDA got initialised in the fold stage"
    return 0


# =============================================================================== SCORING
def load_arc(path, n, seed):
    import random
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    total = len(items)
    idx = sorted(random.Random(seed).sample(range(total), n)) if (n and n < total) else list(range(total))
    return [items[i] for i in idx], total, idx


def build_requests(items, tok):
    """One scoring request per (item, choice)."""
    reqs = []
    for qi, it in enumerate(items):
        ctx = "Question: %s\nAnswer:" % it["question"].strip()
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        for ci, ch in enumerate(it["choices"]["text"]):
            cont = " " + str(ch).strip()
            full_ids = tok.encode(ctx + cont, add_special_tokens=False)
            n_cont = len(full_ids) - len(ctx_ids)
            if n_cont < 1:
                full_ids = ctx_ids + tok.encode(cont, add_special_tokens=False)
                n_cont = len(full_ids) - len(ctx_ids)
            reqs.append(dict(qi=qi, ci=ci, ids=full_ids, n_ctx=len(full_ids) - n_cont, n_cont=n_cont))
    return reqs


def _pad_batch(rows, pad_id, device):
    """Right-pad one batch of rows to [B, T] + its attention mask.

    Extracted VERBATIM out of score_batch (no behaviour change) so the batch-major and layer-major
    paths cannot drift in how they build inputs -- there is exactly one copy of this.
    """
    import torch
    T = max(len(r["ids"]) for r in rows)
    B = len(rows)
    inp = torch.full((B, T), pad_id, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    for i, r in enumerate(rows):
        L = len(r["ids"])
        inp[i, :L] = torch.tensor(r["ids"], dtype=torch.long)
        att[i, :L] = 1
    return inp.to(device), att.to(device)


def _head_and_gather(model, h, inp, rows, sub):
    """lm_head + (gather - logsumexp) in `sub`-row chunks over an ALREADY-NORMED hidden state.

    Extracted VERBATIM out of score_batch. Both scoring paths call THIS function, so the only
    thing the loop inversion can possibly change is the ORDER in which layers execute -- the
    scoring tail is physically the same code, not a re-derivation of it.
    """
    import torch
    B = len(rows)
    out = []
    for i in range(0, B, sub):
        lg = model.lm_head(h[i:i + sub, :-1, :]).float()
        tgt = inp[i:i + sub, 1:]
        lse = torch.logsumexp(lg, dim=-1)
        g = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tl = (g - lse).cpu()
        for j in range(tl.shape[0]):
            r = rows[i + j]
            a, b = r["n_ctx"] - 1, r["n_ctx"] - 1 + r["n_cont"]
            s = float(tl[j, a:b].sum())
            out.append((s, s / max(1, r["n_cont"])))
        del lg, lse, g, tl
    return out


def score_batch(model, rows, pad_id, device, sub=4):
    """Sum + mean log-prob of each row's continuation tokens. Right-padded, causal, no generation.

    The [B, T, 154880] logits are never materialized in fp32: hidden states come from model.model,
    then lm_head + (gather - logsumexp) run in `sub`-row chunks. Mathematically identical to
    log_softmax-then-gather, but bounded memory, which is what allows a large B (the streaming cost
    is per PASS, so a big batch is the main speed lever).
    """
    import torch
    inp, att = _pad_batch(rows, pad_id, device)
    with torch.no_grad():
        h = model.model(input_ids=inp, attention_mask=att, use_cache=False).last_hidden_state
        out = _head_and_gather(model, h, inp, rows, sub)
        del h
    return out


_CAPTURE_CLS = None


def _capture_cls():
    """Lazily built so this module keeps its no-torch-at-import-time property."""
    global _CAPTURE_CLS
    if _CAPTURE_CLS is None:
        import torch

        class _PreambleRecorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.got = None

            def forward(self, hidden_states, **kw):
                self.got = (hidden_states, kw)
                return hidden_states

        _CAPTURE_CLS = _PreambleRecorder
    return _CAPTURE_CLS


def _capture_preamble(model, inp, att):
    """Run the model's OWN preamble and capture the exact kwargs it hands a decoder layer.

    NOTHING about the preamble is re-derived here -- that is the whole point. The decoder stack is
    TEMPORARILY swapped for a one-element list holding a recorder, so `Glm4MoeLiteModel.forward`
    executes verbatim (embed_tokens -> position_ids -> create_causal_mask -> rotary_emb;
    modeling_glm4_moe_lite.py:611-631) and its layer-loop body (:633-642) runs exactly once,
    against the recorder. The captured causal mask / position_embeddings / position_ids are
    therefore the very objects the stock loop would have used, not a copy of that code that can
    drift when transformers changes.

    Reusing ONE capture for every layer is licensed by :633-642, where the loop passes the same
    kwargs to every layer -- this model has a single `create_causal_mask` call and no
    sliding-window / per-layer-type mask dispatch.

    `att` may be None: `heldout_ce` calls the model with no attention_mask at all, and passing an
    explicit ones-mask instead would flip sdpa off its `is_causal` fast path -- a different kernel,
    hence a different bit pattern. None is forwarded verbatim so the captured mask is whatever the
    shipped call would have produced.
    """
    import torch
    cap = _capture_cls()()
    real = model.model.layers
    try:
        model.model.layers = torch.nn.ModuleList([cap])
        model.model(input_ids=inp, attention_mask=att, use_cache=False)
    finally:
        model.model.layers = real
    if cap.got is None:
        raise RuntimeError(
            "[arc/lm] preamble capture never reached the decoder loop: this model's forward no "
            "longer iterates `self.layers`. Refusing -- the layer-major driver would otherwise "
            "score every batch against a stale/absent attention mask and look perfectly healthy.")
    return cap.got


def score_batches_layer_major(model, batches, pad_id, device, sub=4, group=0, streamer=None,
                              log=None):
    """LAYER-MAJOR scoring: load each expert layer ONCE and push EVERY batch through it.

    WHY THIS IS THE FIX. Batch-major scoring re-streams the whole non-pinned expert set once per
    batch, because `ExpertStreamer._post` nulls a non-pinned layer the instant its forward returns
    -- a 100% cache-miss rate by construction. Inverting the loop is only half the fix: without
    `streamer.hold`/`release` the inverted loop would still re-read each layer once per batch.

    WHY IT IS BIT-IDENTICAL, BY CONSTRUCTION (verified against the installed transformers 5.8.1
    source, not assumed):
      * `use_cache=False` keeps `past_key_values` None (modeling_glm4_moe_lite.py:614), so there is
        no KV cache; everything runs under `no_grad`, so there is no autograd graph.
      * every decoder layer gets the SAME kwargs -- one causal mask, one position_embeddings
        (:633-642) -- and a layer returns a plain tensor whose only threaded state is
        `hidden_states` (:514-543).
      * batches are disjoint and share no state.
    So a layer's output is a pure function of (its input hidden state, its weights, that batch's
    preamble). Visiting the (layer, batch) pairs layer-major covers the identical set of pure
    evaluations as batch-major, hence identical numbers. This is an ordering change, not a
    numerical one -- and it is still gated on the GPU A/B in tools/glm_arc_layer_major_gate.py.

    `group` bounds how many batches are in flight at once (0 = all). Streaming drops by exactly the
    group size, so this is the speedup knob AND the VRAM knob. Retained per in-flight batch:
      * hidden state   B * T * hidden * 2 bytes  (bf16)
      * the captured causal mask, which is B * T * T * dtype and therefore grows QUADRATICALLY in
        T -- for long batches this, not the hidden state, is the term that bites.
    Measured for ARC-Easy at batch=64 the hidden states total ~1.4 GiB across all 149 batches, so
    group=0 (all) is affordable there; re-check the mask term before assuming that for a longer
    benchmark.

    Returns a list parallel to `batches`; element i is what `score_batch` would have returned.
    """
    import torch
    n_layers = int(model.config.num_hidden_layers)
    n = len(batches)
    # max(1, ...) matters: with an empty `batches` and group=0 this would otherwise be 0 and
    # `range(0, 0, 0)` raises "range() arg 3 must not be zero" instead of returning [].
    G = max(1, n if not group or int(group) <= 0 else int(group))
    results = [None] * n
    for g0 in range(0, n, G):
        idx = list(range(g0, min(g0 + G, n)))
        t_grp = time.time()
        with torch.no_grad():
            st = []
            for bi in idx:
                rows = batches[bi]
                inp, att = _pad_batch(rows, pad_id, device)
                h, kw = _capture_preamble(model, inp, att)
                st.append([rows, inp, h, kw])
            for li in range(n_layers):
                layer = model.model.layers[li]
                held = streamer.hold(li) if streamer is not None else False
                try:
                    for s in st:
                        o = layer(s[2], **s[3])
                        # transformers >=5 returns a bare tensor (:543); older returns a tuple.
                        s[2] = o[0] if isinstance(o, tuple) else o
                finally:
                    if held:
                        streamer.release(li)
                # A sweep is one long silent stretch (47 layers x every batch). Emit progress so a
                # stall is diagnosable instead of a job that simply sits there -- _log flushes.
                if log is not None and (li % 8 == 0 or li == n_layers - 1):
                    log("[arc/lm]   group %d/%d  layer %d/%d  %.0fs%s"
                        % (g0 // G + 1, (n + G - 1) // G, li + 1, n_layers, time.time() - t_grp,
                           "" if streamer is None
                           else "  %.1f GiB streamed" % (streamer.bytes_streamed / (1 << 30))))
            for k, bi in enumerate(idx):
                rows, inp, h, _kw = st[k]
                h = model.model.norm(h)
                results[bi] = _head_and_gather(model, h, inp, rows, sub)
                st[k] = None
                del h
            del st
    return results


def run_arm(model, tok, reqs, items, device, batch, tag, sub=4, log_every=5,
            layer_major=False, lm_group=0, streamer=None):
    """Score every ARC request and reduce to per-item predictions + accuracy.

    THE ONE SEAM the `--arc-layer-major` flag turns, mirroring `ce_scored` for the CE path.

    DEFAULT OFF, AND THE OFF BRANCH IS THE SHIPPED LOOP, UNCHANGED. ARC accuracy is a published
    capability number, so the flag may never be able to move it: with `layer_major` false the loop
    below is the identical `score_batch` loop this function always ran -- same batching, same
    argument order, same fill, same progress log. Nothing was reordered or "tidied" to make room
    for the branch (`git diff -w` shows the OFF loop moved only by its new indent).

    ON, the batching is IDENTICAL and only the visiting order of the (layer, batch) pairs changes:
    `score_batches_layer_major` returns a list parallel to `batches` whose element i is exactly what
    `score_batch` would have returned, so the fill is the same two lines. Gated bit-exact on the
    real 47-layer model -- 9501 rows, 0 mismatches, max |diff| 0.000e+00, 7375.5 -> 49.5 GiB and
    2430.4 -> 51.1 s (tools/glm_arc_layer_major_gate.py, and the flag-level A/B in
    tools/glm_arc_flag_gate.py which also pins the ACCURACY and asserts the byte counts in BOTH
    directions so a silent fallback to batch-major cannot read as a pass).

    `streamer` is what makes ON cheaper, not the inversion alone: without `streamer.hold/release`
    the inverted loop still re-reads every layer once per batch. Passing `layer_major=True` with
    `streamer=None` is therefore a correct-but-pointless configuration, and it says so in the log.
    """
    import torch
    # NO TOY MODELS: every ARC-Easy accuracy this file reports comes out of run_arm, so this is the
    # single choke point for the capability number. There is no legitimate toy use of this script --
    # a toy's ARC accuracy is the random-guess floor, which reads as a real null result.
    _NTM.assert_real_model(
        model, where="tools/glm_arc_eval.py:run_arm",
        decision="reports ARC-Easy multiple-choice accuracy as this model's CAPABILITY (arm=%s)" % tag)
    order = sorted(range(len(reqs)), key=lambda i: len(reqs[i]["ids"]))
    sums, means = [0.0] * len(reqs), [0.0] * len(reqs)
    t0, done, npass = time.time(), 0, 0
    if layer_major:
        # Same `order`, same slicing arithmetic, same batch contents as the OFF branch -- the two
        # lines below are `for s in range(0, len(order), batch): sel = order[s:s + batch]` hoisted
        # into a list, so a divergence in BATCHING (as opposed to ordering) is impossible.
        sels = [order[s:s + batch] for s in range(0, len(order), batch)]
        batches = [[reqs[i] for i in sel] for sel in sels]
        if streamer is None:
            _log("[arc/%s]   NOTE: layer-major with NO streamer -- the ordering changes but the "
                 "reads do not; expect batch-major's byte count." % tag)
        _log("[arc/%s] LAYER-MAJOR scoring: %d batches / %d seqs, group=%s"
             % (tag, len(batches), len(order), lm_group or "all"))
        res = score_batches_layer_major(model, batches, tok.pad_token_id or 0, device, sub,
                                        group=lm_group, streamer=streamer, log=_log)
        for sel, rb in zip(sels, res):
            for i, (a, b) in zip(sel, rb):
                sums[i], means[i] = a, b
        done, npass = len(order), len(batches)
        if done:
            el = time.time() - t0
            _log("[arc/%s]   %d/%d seqs  %.0fs  %.3fs/seq  [LAYER-MAJOR group=%s]"
                 % (tag, done, len(order), el, el / done, lm_group or "all"))
    else:
        for s in range(0, len(order), batch):
            sel = order[s:s + batch]
            res = score_batch(model, [reqs[i] for i in sel], tok.pad_token_id or 0, device, sub)
            for i, (a, b) in zip(sel, res):
                sums[i], means[i] = a, b
            done += len(sel)
            npass += 1
            if npass % log_every == 0 or done == len(order):
                el = time.time() - t0
                _log("[arc/%s]   %d/%d seqs  %.0fs  %.3fs/seq  eta %.0fs"
                     % (tag, done, len(order), el, el / done, el / done * (len(order) - done)))
    wall = time.time() - t0

    by_item = {}
    for j, r in enumerate(reqs):
        by_item.setdefault(r["qi"], []).append(j)
    per_item = []
    for qi, it in enumerate(items):
        rows = sorted(by_item[qi], key=lambda j: reqs[j]["ci"])
        gold = it["choices"]["label"].index(it["answerKey"])
        p_sum = max(range(len(rows)), key=lambda k: sums[rows[k]])
        p_mean = max(range(len(rows)), key=lambda k: means[rows[k]])
        per_item.append(dict(qi=qi, id=it["id"], n_choices=len(rows), gold=gold,
                             pred_sum=p_sum, pred_mean=p_mean,
                             ok_sum=(p_sum == gold), ok_mean=(p_mean == gold),
                             scores_sum=[sums[j] for j in rows],
                             scores_mean=[means[j] for j in rows]))
    acc = sum(1 for p in per_item if p["ok_sum"]) / len(per_item)
    acc_norm = sum(1 for p in per_item if p["ok_mean"]) / len(per_item)
    _log("[arc/%s] DONE %d items / %d seqs in %.0fs -> acc=%.4f acc_norm=%.4f"
         % (tag, len(per_item), len(reqs), wall, acc, acc_norm))
    return dict(per_item=per_item, acc=acc, acc_norm=acc_norm, wall_s=round(wall, 1),
                seqs=len(reqs), passes=npass, s_per_item=round(wall / len(per_item), 3),
                raw_means=means, raw_sums=sums)


# ============================================================================== STAGE -- BENCH
def stage_bench(opts):
    import torch
    from transformers import AutoTokenizer

    cap, total_vram = _cap_vram()
    d = torch.load(opts.deltas, map_location="cpu", weights_only=False)
    log_spec("arc/bench")
    # Read ONCE, here, so the calibration and both arms provably use the same setting -- a
    # calibration that disagrees with the arms it sizes is the failure mode this hoist prevents.
    arc_lm = bool(getattr(opts, "arc_layer_major", False))
    arc_lm_group = int(getattr(opts, "arc_lm_group", 0) or 0)
    _log("[arc/bench] scoring path: %s"
         % ("LAYER-MAJOR (--arc-layer-major, group=%s) -- gated bit-exact vs batch-major"
            % (arc_lm_group or "all") if arc_lm else "batch-major score_batch (shipped default)"))
    # The target root comes from the SPEC, never from the payload -- a file that self-certifies its
    # own gate is not a gate. Pass the same --campaign-spec you folded with.
    if not d.get("gate_pass") or d["root_folded"] != spec_root():
        raise SystemExit("[arc/bench] BLOCKED: deltas file did not pass the fold gate "
                         "(root_folded=%s, spec final_root=%s)"
                         % (d.get("root_folded"), spec_root()))
    if str(d.get("campaign")) != spec_campaign():
        raise SystemExit("[arc/bench] BLOCKED: deltas file is campaign %s, spec is %s"
                         % (d.get("campaign"), spec_campaign()))
    _log("[arc/bench] deltas OK: root_folded=%s (gate PASS, %d/%d records folded)"
         % (d["root_folded"], d["folded"], d["requested"]))

    pin = list(range(1, 1 + max(1, int(opts.pin_layers))))
    model, cfg, streamer = build_streaming_model(opts.pack_dir, pin, "cuda")
    tok = AutoTokenizer.from_pretrained(CONFIG_DIR, local_files_only=True)

    # ---- ACCEPTANCE 1: full residency, 0 dead layers ----
    layers = model.model.layers
    moe = _moe_layers(cfg)
    dead = [L for L in moe
            if not hasattr(model.model.layers[L].mlp.experts, "gate_up_proj")]
    missing = [L for L in moe if not os.path.exists(_pack_path(opts.pack_dir, L))]
    _log("[arc/bench] RESIDENT: %d transformer layers | %d MoE layers | dead_layers=%s | "
         "missing pack files=%s"
         % (len(layers), len(moe), dead if dead else "[]  (0 dead)", missing if missing else "[]"))
    if len(layers) != cfg.num_hidden_layers or len(moe) != 46 or dead or missing:
        raise SystemExit("[arc/bench] BLOCKED: expected 47 layers / 46 MoE / 0 dead / 0 missing, "
                         "got %d/%d/%d/%d" % (len(layers), len(moe), len(dead), len(missing)))

    # ---- CROSS-PATH GATE: packed layer-1 rows == piece_loader's pristine rows, bit-exact ----
    gu1, dn1 = streamer.cache[1]
    mism = [k for k, snap in d["pristine"].items()
            if int(k.split(":")[0]) == 1
            and not (torch.equal(gu1[int(k.split(":")[1])].cpu(), snap["gate_up"])
                     and torch.equal(dn1[int(k.split(":")[1])].cpu(), snap["down"]))]
    _log("[arc/bench] CROSS-PATH GATE (packed rows == piece_loader rows, bit-exact): %d/%d match "
         "-> %s" % (len(d["pristine"]) - len(mism), len(d["pristine"]),
                    "PASS" if not mism else "FAIL"))
    if mism:
        raise SystemExit("[arc/bench] BLOCKED: pristine expert rows differ between loaders at %r"
                         % (mism,))

    # ---- calibration: time a warm-up on the real model, then size N to the wall budget ----
    all_items, total, _ = load_arc(ARC_PATH, 0, opts.seed)
    warm_reqs = build_requests(all_items[:opts.warmup_items], tok)
    _log("[arc/bench] calibrating on %d items (%d seqs), batch=%d ..."
         % (opts.warmup_items, len(warm_reqs), opts.batch))
    t_w = time.time()
    # The calibration must run through the SAME scoring path as the arms it is sizing. Timing a
    # batch-major warm-up and then running layer-major arms would over-estimate s/pass by the whole
    # speedup and silently shrink N (--n unset auto-sizes from this number) -- a flag that quietly
    # evaluates fewer items is worse than a flag that does nothing.
    if arc_lm:
        score_batches_layer_major(
            model, [warm_reqs[s:s + opts.batch] for s in range(0, len(warm_reqs), opts.batch)],
            tok.pad_token_id or 0, "cuda", opts.sub, group=arc_lm_group, streamer=streamer)
    else:
        for s in range(0, len(warm_reqs), opts.batch):
            score_batch(model, warm_reqs[s:s + opts.batch], tok.pad_token_id or 0, "cuda", opts.sub)
    torch.cuda.synchronize()
    warm_s = time.time() - t_w
    n_warm_pass = (len(warm_reqs) + opts.batch - 1) // opts.batch
    mean_choices = sum(len(i["choices"]["text"]) for i in all_items) / len(all_items)
    s_per_pass = warm_s / max(1, n_warm_pass)
    s_per_item_arm = s_per_pass * mean_choices / opts.batch
    _log("[arc/bench] calibration: %.1fs / %d passes = %.2f s/pass -> %.3f s/item/arm; "
         "full %d items both arms ~ %.2f h  | GPU peak %.2f GiB"
         % (warm_s, n_warm_pass, s_per_pass, s_per_item_arm, total,
            2 * total * s_per_item_arm / 3600.0, torch.cuda.max_memory_allocated(0) / (1 << 30)))

    if opts.n and opts.n > 0:
        n_pick = min(opts.n, total)
        _log("[arc/bench] N fixed by --n = %d" % n_pick)
    else:
        n_fit = int(opts.budget_min * 60.0 / max(1e-6, 2 * s_per_item_arm))
        n_pick = max(opts.min_n, min(total, n_fit))
        _log("[arc/bench] N auto-sized to %.0f min budget: fits %d -> using %d (min_n=%d max=%d)"
             % (opts.budget_min, n_fit, n_pick, opts.min_n, total))

    items, total, idx = load_arc(ARC_PATH, n_pick, opts.seed)
    reqs = build_requests(items, tok)
    floor = sum(1.0 / len(it["choices"]["text"]) for it in items) / len(items)
    _log("[arc/bench] subset N=%d of %d (seed=%d) | %d seqs | floor=%.4f | seq len min/mean/max "
         "%d/%.1f/%d" % (len(items), total, opts.seed, len(reqs), floor,
                         min(len(r["ids"]) for r in reqs),
                         sum(len(r["ids"]) for r in reqs) / len(reqs),
                         max(len(r["ids"]) for r in reqs)))

    _log("[arc/bench] ===== ARM 1: BASE (pristine full model) =====")
    b0 = streamer.bytes_streamed
    base = run_arm(model, tok, reqs, items, "cuda", opts.batch, "base", opts.sub,
                   layer_major=arc_lm, lm_group=arc_lm_group, streamer=streamer)
    _log("[arc/bench] arm 1 streamed %.1f GiB of expert weights"
         % ((streamer.bytes_streamed - b0) / (1 << 30)))

    # ---- patch layer 1 (pinned, so its tensors are real and stay resident) ----
    n_patched = 0
    with torch.no_grad():
        for key, snap in d["foldedrows"].items():
            L, E = (int(x) for x in key.split(":"))
            if L != 1:
                raise SystemExit("[arc/bench] BLOCKED: delta targets layer %d; only layer 1 is "
                                 "pinned in this run" % L)
            gu1[E].copy_(snap["gate_up"].to(gu1.device))
            dn1[E].copy_(snap["down"].to(dn1.device))
            n_patched += 1
    bad = [k for k, snap in d["foldedrows"].items()
           if not (torch.equal(gu1[int(k.split(":")[1])].cpu(), snap["gate_up"])
                   and torch.equal(dn1[int(k.split(":")[1])].cpu(), snap["down"]))]
    moved = [k for k, v in d["coord_stats"].items() if v["moved"]]
    _log("[arc/bench] PATCHED %d coordinates (read-back mismatches: %d); coordinates whose weights "
         "actually differ from base: %d -> %s" % (n_patched, len(bad), len(moved), moved))
    if bad:
        raise SystemExit("[arc/bench] BLOCKED: patch read-back failed at %r" % (bad,))

    _log("[arc/bench] ===== ARM 2: BASE + %d accepted deltas from %s ====="
         % (len(spec_events()), spec_name()))
    trained = run_arm(model, tok, reqs, items, "cuda", opts.batch, "trained", opts.sub,
                      layer_major=arc_lm, lm_group=arc_lm_group, streamer=streamer)

    agree_mean = sum(1 for a, b in zip(base["per_item"], trained["per_item"])
                     if a["pred_mean"] == b["pred_mean"]) / len(items)
    agree_sum = sum(1 for a, b in zip(base["per_item"], trained["per_item"])
                    if a["pred_sum"] == b["pred_sum"]) / len(items)
    _log("[arc/bench] per-item agreement: acc_norm-pred %.4f | acc-pred %.4f" % (agree_mean, agree_sum))

    # NULL-EFFECT DETECTOR: distinguishes "the deltas do not help" (a valid finding) from "the
    # patch never took effect" (a broken measurement).
    diffs = [abs(a - b) for a, b in zip(base["raw_means"], trained["raw_means"])]
    n_changed = sum(1 for x in diffs if x > 0)
    _log("[arc/bench] OUTPUT-CHANGE check: %d/%d sequence scores differ | mean|d|=%.6g max|d|=%.6g"
         % (n_changed, len(diffs), sum(diffs) / len(diffs), max(diffs)))
    if n_changed == 0:
        _log("[arc/bench] *** WARNING: the patch changed NO output. The arm-2 number is NOT a "
             "measurement of the deltas. ***")

    samples = []
    for k in range(min(3, len(items))):
        it = items[k]
        pb, pt = base["per_item"][k], trained["per_item"][k]
        s = dict(id=it["id"], question=it["question"], gold=it["choices"]["text"][pb["gold"]],
                 base_choice=it["choices"]["text"][pb["pred_mean"]],
                 trained_choice=it["choices"]["text"][pt["pred_mean"]],
                 base_correct=pb["ok_mean"], trained_correct=pt["ok_mean"])
        samples.append(s)
        _log("[arc/bench] SAMPLE %d  %s" % (k + 1, _ascii(it["id"])))
        _log("[arc/bench]   Q       : %s" % _ascii(it["question"])[:220])
        _log("[arc/bench]   GOLD    : %s" % _ascii(s["gold"]))
        _log("[arc/bench]   BASE    : %s   [%s]"
             % (_ascii(s["base_choice"]), "correct" if pb["ok_mean"] else "wrong"))
        _log("[arc/bench]   TRAINED : %s   [%s]"
             % (_ascii(s["trained_choice"]), "correct" if pt["ok_mean"] else "wrong"))

    per_item_s = (base["wall_s"] + trained["wall_s"]) / (2 * len(items))
    out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        benchmark="arc_easy", arc_path=ARC_PATH, n=len(items), n_total=total, seed=opts.seed,
        subset_indices=idx, batch=opts.batch, random_floor=floor,
        scoring="answer log-likelihood; acc_norm = mean per continuation token (PRIMARY), "
                "acc = raw sum",
        # WHICH DRIVER produced these floats travels with them, exactly as `ce_function` does for
        # the CE envelope. The two paths are gated bit-identical, but "we believe they are equal"
        # is not a reason to publish a number that cannot say which one it came from.
        arc_scoring_function=("tools/glm_arc_eval.score_batches_layer_major (layer-major inversion "
                              "of the score_batch loop)" if arc_lm else
                              "tools/glm_arc_eval.score_batch (batch-major, shipped)"),
        arc_layer_major=arc_lm, arc_lm_group=arc_lm_group,
        campaign=spec_campaign(), accepted_events=spec_events(), slots=d["slots"],
        campaign_spec=dict(SPEC),
        fold_root=d["root_folded"], fold_root_expected=spec_root(), fold_gate="PASS",
        folded_records="%d/%d" % (d["folded"], d["requested"]),
        coord_stats=d["coord_stats"], coords_moved=moved,
        model=dict(dir=CONFIG_DIR, layers=len(layers), moe_layers=len(moe), dead_layers=0,
                   dtype="bfloat16", residency="trunk resident + %d pinned MoE layers + %d streamed"
                                               % (len(pin), len(moe) - len(pin)),
                   pinned_layers=pin, vram_cap_gib=cap, vram_total_gib=round(total_vram, 2),
                   gpu_peak_gib=round(torch.cuda.max_memory_allocated(0) / (1 << 30), 2)),
        base=dict(acc=base["acc"], acc_norm=base["acc_norm"], wall_s=base["wall_s"],
                  seqs=base["seqs"], passes=base["passes"], s_per_item=base["s_per_item"]),
        trained=dict(acc=trained["acc"], acc_norm=trained["acc_norm"], wall_s=trained["wall_s"],
                     seqs=trained["seqs"], passes=trained["passes"],
                     s_per_item=trained["s_per_item"]),
        delta_acc=trained["acc"] - base["acc"], delta_acc_norm=trained["acc_norm"] - base["acc_norm"],
        agreement_acc_norm=agree_mean, agreement_acc=agree_sum,
        output_change=dict(seqs_changed=n_changed, seqs_total=len(diffs),
                           mean_abs=sum(diffs) / len(diffs), max_abs=max(diffs)),
        samples=samples, s_per_item_per_arm=round(per_item_s, 3),
        full_2376_estimate_h=round(2 * 2376 * per_item_s / 3600.0, 2),
        gib_streamed_total=round(streamer.bytes_streamed / (1 << 30), 1),
        per_item_base=base["per_item"], per_item_trained=trained["per_item"])
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    with open(opts.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    _log("[arc/bench] wrote %s" % opts.out)

    _log("\n[arc/bench] ============ RESULTS (N=%d, seed=%d) ============" % (len(items), opts.seed))
    _log("[arc/bench]   random floor      : %.4f" % floor)
    _log("[arc/bench]   BASE              : acc_norm=%.4f  acc=%.4f" % (base["acc_norm"], base["acc"]))
    _log("[arc/bench]   BASE+DELTAS       : acc_norm=%.4f  acc=%.4f"
         % (trained["acc_norm"], trained["acc"]))
    _log("[arc/bench]   delta             : acc_norm=%+.4f acc=%+.4f"
         % (out["delta_acc_norm"], out["delta_acc"]))
    _log("[arc/bench]   agreement         : %.4f (acc_norm predictions)" % agree_mean)
    _log("[arc/bench]   %.3f s/item/arm -> full 2376 both arms ~ %.2f h | %.0f GiB streamed"
         % (per_item_s, out["full_2376_estimate_h"], out["gib_streamed_total"]))
    streamer.close()
    return 0


# ================================================================== STAGE -- CE (the GATE METRIC)
# The decisive diagnostic for the ARC result: run 5's 25 deltas IMPROVED the gate metric (held-out
# CE 8.82870 -> 7.42533, measured with ONE MoE layer resident) yet made the FULL 47-layer model
# WORSE on ARC-Easy (acc_norm 0.8107 -> 0.6940). Two hypotheses predict OPPOSITE signs of one
# number -- the same held-out CE, on the same frozen pool, computed on the FULL model:
#   H-A GATE-NETWORK MISMATCH   full-model CE gets WORSE -> the gate optimises a different network
#                               than the one we ship (45 of 46 MoE layers were zeroed in training).
#   H-B CE-CAPABILITY DIVERGENCE full-model CE still IMPROVES -> the gate measures what it claims
#                               and CE on this pool is simply not a capability proxy.
HELDOUT_PATH = "D:/glm_daily/coord/ids_daily_heldout.npy"
HELDOUT_SHA256_PREFIX = "598baadd9afead46"
# docs/research/glm_layer_composition.json, configs A/B/C/D -- same frozen pool, same CE function,
# CPU, 1024 seqs. These are the numbers this stage must reproduce (1 layer) or cite (2 layers).
# The published BASE held-out CE of the full 47-layer model, and the ONLY configuration it is a
# number for: 256 sequences (a prefix of the frozen 1024 pool), outer chunk 32, INNER sub-batch 8
# (the shipped default), MoE layers 1-2 pinned. Reproduced bit-identically across two independent
# runs with the host-RAM tier ON and OFF (README.md:872,
# scratchpad/ramtier_ce_selfcontrol.FINAL.json) and again by tools/glm_ce_layer_major_gate.py:64.
# NOT 4.816991313811272 -- that is the 1024-sequence figure and must never be diffed against this.
CE_REFERENCE_256_32_8_PIN2 = 4.8100134711111746
CE_1L_BASE, CE_1L_TRAINED = 8.828700827014062, 7.425325232167399
CE_2L_BASE, CE_2L_TRAINED = 8.647360774778551, 7.337261849834074
ANCHOR_TOL = 0.01


def verify_heldout(path=HELDOUT_PATH):
    """GATE: the frozen yardstick is the file we think it is (memory: probe+heldout are FROZEN)."""
    import hashlib
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    ok = digest.startswith(HELDOUT_SHA256_PREFIX)
    _log("[arc/ce] held-out pool : %s" % path)
    _log("[arc/ce]   sha256      = %s" % digest)
    _log("[arc/ce]   expect prefix %s -> %s" % (HELDOUT_SHA256_PREFIX, "MATCH" if ok else "MISMATCH"))
    if not ok:
        raise SystemExit("[arc/ce] BLOCKED: held-out pool is not the frozen yardstick -- refusing.")
    return digest


def heldout_pool():
    """The COORDINATOR's own held-out selection (`CO._heldout_ids`), asserted equal to the frozen
    file. Using the coordinator's selector rather than np.load directly is what makes the number
    comparable to run 5's; asserting equality with the file is what makes it auditable."""
    import numpy as np
    import sharddiloco_glm_contributor as N
    import sharddiloco_glm_coordinator as CO
    args = build_args(spec_pieces())
    slots = N.parse_slots(args.slots)
    ids = CO._heldout_ids(args, len(slots))
    ref = np.load(HELDOUT_PATH)
    same = tuple(ids.shape) == tuple(ref.shape) and bool((ids == ref).all())
    _log("[arc/ce] CO._heldout_ids(args, %d) -> shape=%s dtype=%s | identical to the frozen file: %s"
         % (len(slots), tuple(ids.shape), ids.dtype, same))
    if not same:
        raise SystemExit("[arc/ce] BLOCKED: the coordinator's held-out selection is not the frozen file.")
    return ids


# ============================================ CE PROVENANCE: the four knobs that MOVE a held-out CE
# WHY (issue #125). Four independent settings change the number this tool prints, and until now NONE
# of them travelled with it:
#   n_seqs     how many held-out sequences were scored. `heldout_pool()` returns 1024; the published
#              reference 4.8100134711111746 is a 256-sequence figure. Different pools, different CE,
#              no way to tell from the output.
#   chunk      the OUTER chunk (`CO._chunked_heldout_ce`). Regroups the summation only.
#   sub_batch  the INNER sub-batch (`sharddiloco_glm_expert.heldout_ce:398`, default 8) -- i.e. the
#              FORWARD BATCH SIZE. Do not file this under "summation order": measured 2026-08-10 on
#              the real 47-layer model, 8 -> 32 moves the CE from 4.810013471111175 to
#              4.813307792909684, a gap of 3.3 MILLINATS, ~10 orders of magnitude beyond fp
#              re-association. The expert GEMM is batch-composition-sensitive here (memory
#              glm52-milestone), so a different sub-batch is a different computation. `stage_ce`
#              used to force this equal to --ce-chunk unconditionally, which is exactly how the
#              reference became unreachable from the CLI.
#   pin        which MoE layers were resident vs streamed. Does not change the math, but it does
#              change which code path produced it, which is what an auditor is trying to establish.
# So: print the tuple next to EVERY CE this tool emits. A mismatch then reads as a configuration
# difference instead of as a regression -- which is the whole complaint (there is already an
# unexplained 4.8170-vs-4.826324 gap on the public page whose cause is still unknown).
#
# Three of the four are derived AT THE POINT OF MEASUREMENT (len(ids), the chunk argument, and the
# env var as the scorer will actually read it), so they cannot be stale. Only `pin` has to be
# recorded by whoever built the model.
_CE_PIN_DESC = "unset"


def set_ce_pin(desc):
    """Record the residency of the model about to be scored. Called by the model builders, so a CE
    log line cannot describe a residency the model does not have."""
    global _CE_PIN_DESC
    _CE_PIN_DESC = str(desc)


def ce_sub_batch():
    """The INNER sub-batch `sharddiloco_glm_expert.heldout_ce` will actually use -- read from the
    environment exactly as it reads it (:398), never from an opts field that might disagree."""
    return max(1, int(os.environ.get("NEURAHASH_GLM_EVAL_CHUNK", "8")))


def ce_config_tuple(ids, chunk):
    """(n_seqs, chunk, sub_batch, pin) for the CE about to be / just computed."""
    n = -1
    try:
        n = len(ids)
    except TypeError:
        pass
    return (n, int(chunk), ce_sub_batch(), _CE_PIN_DESC)


def ce_config_str(ids, chunk):
    n, c, s, p = ce_config_tuple(ids, chunk)
    return "(n_seqs=%s, chunk=%d, sub_batch=%d, pin=%s)" % ("?" if n < 0 else n, c, s, p)


def ce_ids(opts):
    """The scored held-out set: the frozen pool, then the `--ce-seqs` PREFIX of it.

    `heldout_pool()` returns all 1024 sequences, but the published reference CE is a 256-sequence
    figure taken as `ids[:256]` (tools/glm_ce_layer_major_gate.py:82). Until --ce-seqs existed there
    was no way to say that from this CLI, so the reference configuration was inexpressible and the
    published constant unreachable. Default 0 = the whole pool = the historical behaviour.

    A PREFIX, not a random sample, and deliberately so: the reference took a prefix, and a differently
    drawn 256 is a different measurement that would land within noise of the right answer and be
    impossible to tell apart from it.
    """
    digest = verify_heldout()
    ids = heldout_pool()
    want = int(getattr(opts, "ce_seqs", 0) or 0)
    if want:
        # `< 0` is not pedantry: -256 is truthy, so `ids[:-256]` would quietly score 768 sequences
        # from the FRONT and log it as a request that was honoured.
        if want < 0 or want > len(ids):
            raise SystemExit("[arc/ce] BLOCKED: --ce-seqs %d is not in 1..%d (the frozen pool); "
                             "a negative value would silently truncate from the END"
                             % (want, len(ids)))
        ids = ids[:want]
        _log("[arc/ce] --ce-seqs %d: scoring the first %d of the frozen pool's sequences"
             % (want, len(ids)))
    return ids, digest


def gate_ce(model, ids, chunk, tag):
    """THE GATE'S OWN CE -- `sharddiloco_glm_coordinator._chunked_heldout_ce` over
    `sharddiloco_glm_expert.heldout_ce`. That is the identical call the live coordinator makes
    (sharddiloco_glm_coordinator.py:1920/2233/2427) and the identical one
    tools/glm_layer_composition_test.py:257 used to produce the published 1- and 2-layer numbers.
    Nothing is re-derived here: this file supplies the model and the ids, and nothing else."""
    import sharddiloco_glm_contributor as N
    import sharddiloco_glm_coordinator as CO
    # NO TOY MODELS: this is the live coordinator's OWN accept-gate CE, reproduced here. A toy CE is
    # a toy verdict, and the published 1-/2-layer composition numbers were produced by this call.
    _NTM.assert_real_model(
        model, where="tools/glm_arc_eval.py:gate_ce",
        decision="reproduces the coordinator's held-out-CE ACCEPT GATE and publishes the number "
                 "(tag=%s)" % tag)
    G = N._G()
    t0 = time.time()
    ce = float(CO._chunked_heldout_ce(G, model, ids, chunk))
    dt = time.time() - t0
    _log("[arc/ce] %-28s held-out CE = %.6f   (%.0fs) %s"
         % (tag, ce, dt, ce_config_str(ids, chunk)))
    _log("[arc/ce] %-28s full float repr %r" % ("", ce))
    return ce, round(dt, 1)


def ce_scored(opts, model, ids, chunk, tag, streamer=None):
    """THE ONE SEAM the `--ce-layer-major` flag turns: shipped `gate_ce` (default) or the inversion.

    DEFAULT OFF, AND THE OFF BRANCH IS THE SHIPPED CALL, UNCHANGED. Miners are paid against this
    number, so the flag may never be able to move it: with the flag absent this function is a
    one-line forwarder to exactly the `gate_ce(model, ids, chunk, tag)` every CE helper called
    before it existed -- same arguments, same order, same everything. Nothing about the shipped
    path was reordered or "tidied" to make room for the branch.

    ON, the arithmetic is `heldout_ce_layer_major`'s, which reproduces the shipped reduction op for
    op (see its docstring) and was gated bit-exact against it on the real 47-layer model:
    scratchpad/CE_LAYER_MAJOR_GATE_2026-08-10.json, exit_code 0, all 8 per-chunk floats identical
    digit for digit and 32.00x fewer bytes read (1584.0 -> 49.5 GiB, 885.4 s -> 30.3 s).

    Returns `(ce, seconds)` -- the same 2-tuple `gate_ce` returns, so no call site changes shape.
    """
    if not bool(getattr(opts, "ce_layer_major", False)):
        return gate_ce(model, ids, chunk, tag)
    # NO TOY MODELS: this branch PUBLISHES the gate metric, exactly as `gate_ce` does, so it carries
    # the same assertion. (`heldout_ce_layer_major` itself deliberately does not -- it is a
    # primitive the CPU equivalence tests drive with a fake model.)
    _NTM.assert_real_model(
        model, where="tools/glm_arc_eval.py:ce_scored(--ce-layer-major)",
        decision="publishes the held-out-CE ACCEPT-GATE number through the layer-major inversion "
                 "(tag=%s)" % tag)
    group = int(getattr(opts, "ce_lm_group", 0) or 0)
    b0 = streamer.bytes_streamed if streamer is not None else 0
    t0 = time.time()
    ce, per_chunk, _per_unit = heldout_ce_layer_major(model, ids, chunk, streamer=streamer,
                                                      group=group, log=_log)
    dt = time.time() - t0
    gib = ((streamer.bytes_streamed - b0) / (1 << 30)) if streamer is not None else 0.0
    _log("[arc/ce] %-28s held-out CE = %.6f   (%.0fs) %s [LAYER-MAJOR group=%s]"
         % (tag, ce, dt, ce_config_str(ids, chunk), group or "all"))
    _log("[arc/ce]   layer-major: %d chunk value(s), %.1f GiB streamed (%.2f GiB/s) | full float "
         "repr %r" % (len(per_chunk), gib, (gib / dt) if dt > 0 else 0.0, ce))
    return ce, round(dt, 1)


def heldout_ce_layer_major(model, ids, chunk, streamer=None, group=0, log=None):
    """`CO._chunked_heldout_ce(G, model, ids, chunk)` with the LAYER LOOP OUTSIDE the data loop.

    THE PROBLEM. The shipped CE is two nested data loops -- chunks of `chunk` in
    `sharddiloco_glm_coordinator._chunked_heldout_ce:285-288`, sub-batches of
    NEURAHASH_GLM_EVAL_CHUNK in `sharddiloco_glm_expert.heldout_ce:402-403` -- with the 47-layer
    forward innermost. `ExpertStreamer._post` nulls a non-pinned layer the instant its forward
    returns, so every one of those forwards re-streams the whole non-pinned expert set: measured
    8 chunks x 4 sub-batches x 44 streamed layers x 1.125 GiB = 1584 GiB to do 49.5 GiB of work.

    THE INVERSION. The 32 (chunk, sub-batch) pairs become 32 flat "units", each the exact tensor
    `heldout_ce:403` builds. One layer-major sweep advances EVERY unit's hidden state through layer
    L while L is held resident, so each layer is read once per GROUP rather than once per unit.

    WHAT IS DELIBERATELY *NOT* INVERTED: the reduction. Miners are paid against this number, so it
    must come back bit-identical, and float addition is not associative. The tail below is copied
    from `heldout_ce:404-412` and the two-level mean from `_chunked_heldout_ce:281-289`, op for op:

      * lm_head runs over the FULL T and the slice happens AFTER (`:404-405`). Do NOT reuse this
        file's `_head_and_gather`: it slices BEFORE lm_head, which changes the GEMM M-dim 32->31,
        and it uses the gather-minus-logsumexp identity instead of `log_softmax`. Both are
        mathematically equal to the shipped tail and neither is bitwise equal to it.
      * the fp32 cast is exactly `logits[:, :-1, :].float()` -- no earlier, no later.
      * one `.sum()` per unit, materialised through `float()` per unit; never a fused sum across
        units and never a GPU accumulator, because that would reorder the additions.
      * within a chunk: python-float `tot += s; cnt += c` in UNIT order, then `-(tot/max(1,cnt))`.
      * across chunks: `tot += ce*len(part); seen += len(part)` in CHUNK order, then `tot/seen`,
        including `_chunked_heldout_ce`'s `chunk <= 0 or n <= chunk` early-return branch, which
        returns the single chunk's value directly rather than through a 1-term weighted mean.

    `group` = units in flight per sweep (0 = all). It is an IO knob only: results are keyed by unit
    index and reduced after every group has finished, so group size cannot reach the arithmetic.
    Bytes = ceil(n_units/group) x (streamed layers x 1.125 GiB).

    `attention_mask` is None, not a ones-mask: `heldout_ce:404` passes no mask at all, and an
    explicit ones-mask flips sdpa off its `is_causal` fast path -- a different kernel, so a
    different bit pattern.

    Returns `(ce, per_chunk, per_unit)` -- the scalar the shipped path returns, plus the two
    intermediate levels so a gate can localise a mismatch instead of only detecting one.

    NOTE: no `_NTM.assert_real_model` here on purpose -- this is a primitive, and the CPU
    equivalence tests drive it with a fake model. The REAL-model assertion belongs to the caller
    that publishes the number (`gate_ce` carries it for the shipped path).
    """
    import torch
    # Same env read as sharddiloco_glm_expert.heldout_ce:398-399 -- if that default ever moves, the
    # unit boundaries must move with it or the two paths stop summing the same groups of tokens.
    sub = max(1, int(os.environ.get("NEURAHASH_GLM_EVAL_CHUNK", "8")))
    dev = next(model.parameters()).device
    model.eval()
    n = len(ids)
    # _chunked_heldout_ce:282 -- one bound means the early-return branch, not a 1-term mean.
    single = (chunk <= 0 or n <= chunk)
    bounds = [(0, n)] if single else [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    units = []                                  # (chunk_index, [B,T] int64 unit tensor)
    for ci, (s, e) in enumerate(bounds):
        part = ids[s:e]
        for i in range(0, len(part), sub):
            units.append((ci, torch.as_tensor(part[i:i + sub]).to(dev)))

    n_layers = int(model.config.num_hidden_layers)
    nu = len(units)
    # max(1, ...) so an empty pool cannot produce `range(0, 0, 0)` -> "arg 3 must not be zero".
    G = max(1, nu if not group or int(group) <= 0 else int(group))
    per_unit = [None] * nu
    with torch.no_grad():
        for g0 in range(0, nu, G):
            idx = list(range(g0, min(g0 + G, nu)))
            t_grp = time.time()
            # `stash`, not `st`: the CE tail below reuses the SHIPPED code's own variable names so
            # it is character-for-character `sharddiloco_glm_expert.heldout_ce:404-410`, and `st`
            # there is the target-token slice. test_the_ce_tail_is_the_shipped_tail_character_for_
            # character pins that, because the CPU A/B provably cannot see a wrong tail.
            stash = []
            for k in idx:
                t = units[k][1]
                h, kw = _capture_preamble(model, t, None)
                stash.append([k, t, h, kw])
            for li in range(n_layers):
                layer = model.model.layers[li]
                held = streamer.hold(li) if streamer is not None else False
                try:
                    for s in stash:
                        o = layer(s[2], **s[3])
                        # transformers >=5 returns a bare tensor; older returns a tuple.
                        s[2] = o[0] if isinstance(o, tuple) else o
                finally:
                    if held:
                        streamer.release(li)
                if log is not None and (li % 8 == 0 or li == n_layers - 1):
                    log("[arc/ce-lm]   group %d/%d  layer %d/%d  %.0fs%s"
                        % (g0 // G + 1, (nu + G - 1) // G, li + 1, n_layers, time.time() - t_grp,
                           "" if streamer is None
                           else "  %.1f GiB streamed" % (streamer.bytes_streamed / (1 << 30))))
            # ---- CE tail, per unit, IN UNIT ORDER (sharddiloco_glm_expert.heldout_ce:404-410) ----
            for s in stash:
                k, t, h = s[0], s[1], s[2]
                h = model.model.norm(h)
                logits = model.lm_head(h)                       # FULL T, then slice -- see :404
                sl = logits[:, :-1, :].float()
                st = t[:, 1:]
                lp = torch.log_softmax(sl, dim=-1)              # full-vocab, NOT gather-logsumexp
                tok_lp = lp.gather(-1, st.unsqueeze(-1)).squeeze(-1)
                per_unit[k] = (float(tok_lp.sum()), int(st.numel()))
                del logits, sl, lp, tok_lp, h, st
                s[2] = None
            del stash

    # ---- reduction level 1: within a chunk (sharddiloco_glm_expert.heldout_ce:400,409-412) ----
    per_chunk = []
    for ci in range(len(bounds)):
        tot, cnt = 0.0, 0
        for k, (cidx, _t) in enumerate(units):
            if cidx != ci:
                continue
            s_k, c_k = per_unit[k]
            tot += s_k
            cnt += c_k
        if cnt == 0:
            # SAME fail-open the shipped primitive carried until 2026-08-10 (#96): `max(1, cnt)`
            # turned "this chunk scored ZERO predicted tokens" into -0.0, a PERFECT CE, for a
            # measurement that never happened. This port is the DEFAULT scoring path for the
            # published gate metric, so leaving it would let the default bypass the fix at the
            # source. Raise the SAME exception type the shipped `heldout_ce` raises, so one
            # `except` catches both drivers.
            import sharddiloco_glm_expert as _X
            raise _X.EmptyHeldoutPool(
                "layer-major CE chunk %d of %d scored ZERO predicted tokens over %d sequence(s) "
                "(ids shape %s): a next-token CE needs at least one sequence of length >= 2. "
                "Refusing to return a number -- the value this used to produce was -0.0, a "
                "PERFECT score." % (ci, len(bounds), len(ids), tuple(getattr(ids, "shape", ()))))
        per_chunk.append(float(-(tot / cnt)))

    # ---- reduction level 2: across chunks (_chunked_heldout_ce:282-289) ----
    if single:
        ce = per_chunk[0]
    else:
        tot, seen = 0.0, 0
        for ci, (s, e) in enumerate(bounds):
            tot += per_chunk[ci] * (e - s)
            seen += (e - s)
        ce = tot / seen
    return ce, per_chunk, per_unit


def _rows_match(gu, dn, snap_map, keyfilter=None):
    """Bit-exact identity of fused expert rows against a `fold`-stage snapshot."""
    import torch
    bad = []
    for k, snap in snap_map.items():
        L, E = (int(x) for x in k.split(":"))
        if keyfilter is not None and not keyfilter(L):
            continue
        if not (torch.equal(gu[E].cpu(), snap["gate_up"]) and torch.equal(dn[E].cpu(), snap["down"])):
            bad.append(k)
    return bad


def _ce_anchor(opts, ids, device, d):
    """SANITY ANCHOR + the 1-layer point of the series.

    Rebuilds run 5's OWN network -- piece_loader at pieces 0-11, i.e. layer 1 resident and 45 of 46
    MoE layers returning EXACTLY ZERO -- and scores both arms with the same `gate_ce`. Arm 2 is the
    `fold` stage's snapshotted rows, so the anchor and the full-model arm 2 carry byte-identical
    weights. If this does not reproduce 8.82870 / 7.42533 the CE definition differs from the gate's
    and every full-model number is uninterpretable.
    """
    import gc
    import torch
    import sharddiloco_glm_contributor as N
    args = build_args(spec_pieces())
    args.device = device
    t0 = time.time()
    model, cfg, seq = N.build_node_model(args, log=_log)
    moe = _moe_layers(cfg)
    dead = [L for L in moe if not hasattr(model.model.layers[L].mlp.experts, "gate_up_proj")]
    _log("[arc/ce/anchor-%s] RESIDENT: %d transformer layers | %d MoE layers | dead_layers=%d %s..."
         % (device, len(model.model.layers), len(moe), len(dead), dead[:5]))
    if len(dead) != 45:
        raise SystemExit("[arc/ce/anchor] BLOCKED: expected 45 dead MoE layers (the training-time "
                         "network: exactly ONE resident MoE layer), got %d" % len(dead))
    exp1 = model.model.layers[1].mlp.experts
    exp1 = exp1.base if hasattr(exp1, "base") else exp1
    bad = _rows_match(exp1.gate_up_proj, exp1.down_proj, d["pristine"], lambda L: L == 1)
    _log("[arc/ce/anchor-%s] base rows == fold-stage pristine snapshot, bit-exact: %d/%d -> %s"
         % (device, len(d["pristine"]) - len(bad), len(d["pristine"]), "PASS" if not bad else "FAIL"))
    if bad:
        raise SystemExit("[arc/ce/anchor] BLOCKED: pristine rows differ at %r" % (bad,))
    _log("[arc/ce/anchor-%s] loaded in %.0fs, params=%d, seq=%d"
         % (device, time.time() - t0, sum(p.numel() for p in model.parameters()), seq))
    # The anchor has no streamer: residency is whatever piece_loader made resident, so say THAT
    # rather than leaving the streaming model's pin set behind to mislabel this CE.
    set_ce_pin("piece_loader pieces=%s (1 resident MoE layer, %d dead)" % (spec_pieces(), len(dead)))

    ce_base, t_b = ce_scored(opts, model, ids, opts.ce_chunk, "anchor-%s ARM 1 BASE" % device)
    n_patched = 0
    with torch.no_grad():
        for k, snap in d["foldedrows"].items():
            L, E = (int(x) for x in k.split(":"))
            if L != 1:
                raise SystemExit("[arc/ce/anchor] BLOCKED: delta targets layer %d, not resident" % L)
            exp1.gate_up_proj[E].copy_(snap["gate_up"].to(exp1.gate_up_proj.device))
            exp1.down_proj[E].copy_(snap["down"].to(exp1.down_proj.device))
            n_patched += 1
    bad = _rows_match(exp1.gate_up_proj, exp1.down_proj, d["foldedrows"], lambda L: L == 1)
    _log("[arc/ce/anchor-%s] PATCHED %d coordinates | read-back mismatches: %d"
         % (device, n_patched, len(bad)))
    if bad:
        raise SystemExit("[arc/ce/anchor] BLOCKED: patch read-back failed at %r" % (bad,))
    ce_trained, t_t = ce_scored(opts, model, ids, opts.ce_chunk, "anchor-%s ARM 2 TRAINED" % device)

    ok_b = abs(ce_base - CE_1L_BASE) <= ANCHOR_TOL
    ok_t = abs(ce_trained - CE_1L_TRAINED) <= ANCHOR_TOL
    _log("[arc/ce/anchor-%s] GATE 3 reproduction: base %.6f vs %.6f |d|=%.2e -> %s | "
         "trained %.6f vs %.6f |d|=%.2e -> %s"
         % (device, ce_base, CE_1L_BASE, abs(ce_base - CE_1L_BASE), "PASS" if ok_b else "FAIL",
            ce_trained, CE_1L_TRAINED, abs(ce_trained - CE_1L_TRAINED), "PASS" if ok_t else "FAIL"))
    del model, exp1
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return dict(device=device, resident_layers=1, dead_moe_layers=len(dead), pieces=spec_pieces(),
                ce_base=ce_base, ce_trained=ce_trained, patched=n_patched,
                ref_base=CE_1L_BASE, ref_trained=CE_1L_TRAINED, tol=ANCHOR_TOL,
                repro_base=ok_b, repro_trained=ok_t, gate3_pass=bool(ok_b and ok_t),
                eval_s_base=t_b, eval_s_trained=t_t)


def _fmt_ranges(ids):
    """[0,1,2,13,14] -> '0-2,13-14' (the --pieces syntax)."""
    ids = sorted(set(int(i) for i in ids))
    out, s, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append((s, prev))
        s = prev = i
    out.append((s, prev))
    return ",".join("%d-%d" % (a, b) if a != b else str(a) for a, b in out)


_CLEAN_PIECES = {}


def clean_pieces_by_layer():
    """Per MoE layer, the expert pieces that lie ENTIRELY inside it -- read off the live manifest,
    never assumed. Straddlers (a piece whose 5 coordinates span two layers) are excluded, exactly as
    tools/glm_layer_composition_test.py excluded pieces 12 and 25: including one would (a) push the
    lower layer from 60/64 to 64/64 -- so layer 1 would stop being byte-identical to run 5's own
    network -- and (b) materialise a full-width MoE slab holding a single real expert whose router
    then forces every token through it. With straddlers dropped EVERY resident layer is 60 of 64."""
    global _CLEAN_PIECES
    if _CLEAN_PIECES:
        return _CLEAN_PIECES
    import piece_loader as pl
    man = pl.load_manifest(SHARD_DIR, require_pieces=[], require_files=False)
    by = {}
    for p in man["pieces"]:
        name = str(p["piece"])
        if not name.startswith("experts_"):
            continue
        layers = {int(le[0]) for le in p["experts"]}
        if len(layers) == 1:
            by.setdefault(layers.pop(), []).append(int(name.split("_")[1]))
    _CLEAN_PIECES = {L: sorted(v) for L, v in by.items()}
    return _CLEAN_PIECES


def prefix_pieces(k):
    """--pieces for 'MoE layers 1..k resident, 60 of 64 experts each, everything above k dead'."""
    by = clean_pieces_by_layer()
    ids = []
    for L in range(1, int(k) + 1):
        if L not in by:
            raise SystemExit("[arc/ce/prefix] BLOCKED: no clean pieces for layer %d" % L)
        ids.extend(by[L])
    return _fmt_ranges(ids)


_RESIDENT_E = {}


def resident_experts_by_layer():
    """{layer: [expert ids covered by that layer's CLEAN pieces]} -- the 60 of 64 the piece path makes
    resident. The pack holds all 64, so the pack path has to reproduce this mask itself."""
    global _RESIDENT_E
    if _RESIDENT_E:
        return _RESIDENT_E
    import piece_loader as pl
    man = pl.load_manifest(SHARD_DIR, require_pieces=[], require_files=False)
    clean = clean_pieces_by_layer()
    want = {p: set() for L in clean for p in clean[L]}
    out = {}
    for p in man["pieces"]:
        name = str(p["piece"])
        if not name.startswith("experts_"):
            continue
        pid = int(name.split("_")[1])
        if pid not in want:
            continue
        for le in p["experts"]:
            out.setdefault(int(le[0]), set()).add(int(le[1]))
    _RESIDENT_E = {L: sorted(v) for L, v in out.items()}
    return _RESIDENT_E


class PrefixStreamer(ExpertStreamer):
    """ExpertStreamer for a PREFIX residency: layers 1..k live (60 of 64 experts, the other 4 rows
    zeroed and the router already masked by the caller), everything above k already replaced by a
    _DeadExperts placeholder. Two deltas from the base class:

      * the pack holds all 64 rows, so every slab is masked down to the piece path's 60 on read --
        otherwise this path would score a DENSER network than the k<=15 points and the series would
        compare two different network families;
      * non-pinned slabs are cached in CPU RAM after the first read, so a pass costs an H2D copy
        (~10 GiB/s) instead of a 1.125 GiB NVMe read -- the full-47 run spent 1000 s/arm on disk.
    """

    def __init__(self, model, cfg, pack_dir, pin, device, layers, resident, ram_budget_gib=1e9):
        self.resident = resident
        self.ram = {}
        # A slab is 1.125 GiB of host RAM. Past the budget the layer is re-read from the pack every
        # pass instead (slower, but a swap storm is worse than slow).
        self.ram_max = max(0, int(float(ram_budget_gib) / 1.125))
        super().__init__(model, cfg, pack_dir, pin, device, layers=layers)
        # This subclass OVERRIDES `_read` (below) with its own masked host-RAM cache, so the base
        # class's T1 tier can never fire here. Drop it rather than let NEURAHASH_SD_RAM_TIER log
        # "ON" and then hold nothing -- a flag that silently does nothing is how a run gets
        # attributed to a lever it never used.
        if self.ram_tier is not None:
            _log("[arc/ramtier] PrefixStreamer overrides _read and keeps its OWN masked RAM cache "
                 "(ram_max=%d slabs); %s does not apply here." % (self.ram_max, _RAM_TIER_ENV))
            self.ram_tier = None

    def _mask(self, L, gu, dn):
        torch = self.torch
        keep = torch.zeros(gu.shape[0], dtype=torch.bool, device=gu.device)
        keep[torch.tensor(self.resident[L], device=gu.device)] = True
        idx = (~keep).nonzero().flatten()
        if idx.numel():
            gu[idx] = 0
            dn[idx] = 0
        return gu, dn

    def _read_to(self, L, dev):
        # `_mask` writes IN PLACE into these tensors. Through the old mmap path that was a
        # copy-on-write fault per touched page against a read-only file mapping (measured
        # 2026-08-05); `_st_read` hands back owned heap memory, so the write is an ordinary store.
        slab = _st_read(_pack_path(self.pack_dir, L), ("gate_up", "down"), dev)
        return self._mask(L, slab["gate_up"], slab["down"])

    def _read(self, L):
        if L in self.pin:                                   # resident for the whole pass
            gu, dn = self._read_to(L, self.device)
            self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
            return gu, dn
        if L not in self.ram:                               # one disk read per layer per PROCESS
            if len(self.ram) >= self.ram_max:               # RAM budget spent -> straight from disk
                gu, dn = self._read_to(L, self.device)
                self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
                return gu, dn
            self.ram[L] = self._read_to(L, "cpu")
        gu, dn = self.ram[L]
        self.bytes_streamed += gu.numel() * 2 + dn.numel() * 2
        return gu.to(self.device), dn.to(self.device)

    def preload_ram(self):
        for L in self.layers:
            if L not in self.pin and L not in self.ram and len(self.ram) < self.ram_max:
                self.ram[L] = self._read_to(L, "cpu")
        return len(self.ram)


def build_prefix_pack_model(pack_dir, k, pin_layers, device, log=_log, ram_budget_gib=18.0):
    """Prefix-k residency built from the LAYER PACK instead of the shard pieces -- the same network
    the piece path builds (asserted against it at overlapping k), but with the expert slabs streamed
    so k is bounded by RAM, not by the VRAM cap. Returns (model, cfg, streamer)."""
    import torch
    import piece_loader as pl
    model, cfg = build_trunk_model(device, log=log)
    moe = _moe_layers(cfg)
    live = [L for L in moe if L <= k]
    resident = resident_experts_by_layer()
    for L in moe:
        mlp = model.model.layers[L].mlp
        if L > k:                                  # dead: EXACTLY the piece path's placeholder
            e = mlp.experts
            mlp.experts = pl._make_dead_experts(e.num_experts, e.hidden_dim, e.intermediate_dim,
                                                e.act_fn)
            continue
        res = resident[L]
        if len(res) < int(cfg.n_routed_experts):    # resident-only routing, verbatim piece_loader:404-408
            bias = mlp.gate.e_score_correction_bias
            keep = torch.zeros(int(cfg.n_routed_experts), dtype=torch.bool, device=bias.device)
            keep[torch.tensor(res, device=bias.device)] = True
            bias.masked_fill_(~keep, float("-inf"))
    pin = [L for L in live if L <= max(1, int(pin_layers))]
    streamer = PrefixStreamer(model, cfg, pack_dir, pin, device, live, resident,
                              ram_budget_gib=ram_budget_gib)
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_left:
        raise SystemExit("[arc/ce/prefix-pack] BLOCKED: %d meta parameter(s) left, e.g. %r"
                         % (len(meta_left), meta_left[:5]))
    t0 = time.time()
    n_pin = streamer.preload_pinned()
    n_ram = streamer.preload_ram()
    model.eval()
    torch.cuda.synchronize()
    set_ce_pin("prefix-pack k=%d (%d pinned on GPU, %d cached in RAM)" % (k, n_pin, n_ram))
    log("[arc/ce/prefix-pack] k=%d: %d live MoE layers (%d pinned on GPU, %d cached in RAM = %.1f "
        "GiB), %d dead | preloaded in %.0fs | GPU allocated %.2f GiB"
        % (k, len(live), n_pin, n_ram, n_ram * 1.125, len(moe) - len(live), time.time() - t0,
           torch.cuda.memory_allocated(0) / (1 << 30)))
    return model, cfg, streamer


def _prefix_device(opts, k):
    """cuda while trunk (4.02 GiB) + k full-width slabs (1.125 GiB each) + activations fit under the
    cap with >= 8 GiB left for the desktop; cpu above that. Never guesses past the cap."""
    if opts.ce_device in ("cpu", "cuda"):
        return opts.ce_device
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        total = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    except Exception:
        return "cpu"
    cap = _resolve_vram_cap(total)
    need = 4.02 + 1.125 * int(k) + 1.8          # weights + measured activation headroom at chunk 32
    return "cuda" if need <= cap else "cpu"


def _ce_prefix(opts, ids, device, d, k):
    """ONE POINT of the residency series: MoE layers 1..k resident (60/64 each), 46-k dead.

    Identical in every respect to `_ce_anchor` except how many layers are resident -- same
    `build_node_model` piece path, same frozen pool, same `gate_ce`, same arm-2 rows patched from the
    fold snapshot. k=1 IS run 5's own network, so k=1/k=2 double as the published-anchor check.
    """
    import gc
    import torch
    import sharddiloco_glm_contributor as N
    pieces = prefix_pieces(k)
    args = build_args(pieces)
    args.device = device
    if device.startswith("cuda"):
        _cap_vram()
    t0 = time.time()
    model, cfg, seq = N.build_node_model(args, log=_log)
    load_s = time.time() - t0
    claimable = N.node_claimable_coords(args) or []
    moe = _moe_layers(cfg)
    dead = [L for L in moe if not hasattr(model.model.layers[L].mlp.experts, "gate_up_proj")]
    live = [L for L in moe if L not in dead]
    per_layer = {}
    for (L, E) in claimable:
        per_layer[L] = per_layer.get(L, 0) + 1
    dens = sorted(set(per_layer.values()))
    set_ce_pin("piece_loader prefix k=%d (%d live MoE layers, %d dead)" % (k, len(live), len(dead)))
    _log("[arc/ce/prefix:%d] RESIDENT: %d transformer layers | %d MoE layers | live=%s | dead=%d | "
         "claimable=%d | per-layer density=%s of %d"
         % (k, len(model.model.layers), len(moe), _fmt_ranges(live) if live else "[]",
            len(dead), len(claimable), dens, int(cfg.n_routed_experts)))
    if live != list(range(1, k + 1)) or len(dead) != len(moe) - k:
        raise SystemExit("[arc/ce/prefix:%d] BLOCKED: residency is not the 1..%d prefix (live=%s)"
                         % (k, k, live[:8]))
    if dens != [60] or len(claimable) != 60 * k:
        raise SystemExit("[arc/ce/prefix:%d] BLOCKED: expected 60/64 in every resident layer, got %s"
                         % (k, dens))

    exp1 = model.model.layers[1].mlp.experts
    exp1 = exp1.base if hasattr(exp1, "base") else exp1
    bad = _rows_match(exp1.gate_up_proj, exp1.down_proj, d["pristine"], lambda L: L == 1)
    _log("[arc/ce/prefix:%d] layer-1 base rows == fold-stage pristine snapshot, bit-exact: %d/%d -> %s"
         % (k, len(d["pristine"]) - len(bad), len(d["pristine"]), "PASS" if not bad else "FAIL"))
    if bad:
        raise SystemExit("[arc/ce/prefix:%d] BLOCKED: layer 1 is not byte-identical across k: %r" % (k, bad))
    _log("[arc/ce/prefix:%d] device=%s pieces=%s loaded in %.0fs, params=%d, seq=%d"
         % (k, device, pieces, load_s, sum(p.numel() for p in model.parameters()), seq))

    ce_base, t_b = ce_scored(opts, model, ids, opts.ce_chunk, "prefix:%d(%s) ARM 1 BASE" % (k, device))
    n_patched = 0
    with torch.no_grad():
        for key, snap in d["foldedrows"].items():
            L, E = (int(x) for x in key.split(":"))
            if L != 1:
                raise SystemExit("[arc/ce/prefix] BLOCKED: delta targets layer %d, not layer 1" % L)
            exp1.gate_up_proj[E].copy_(snap["gate_up"].to(exp1.gate_up_proj.device))
            exp1.down_proj[E].copy_(snap["down"].to(exp1.down_proj.device))
            n_patched += 1
    bad = _rows_match(exp1.gate_up_proj, exp1.down_proj, d["foldedrows"], lambda L: L == 1)
    if bad:
        raise SystemExit("[arc/ce/prefix:%d] BLOCKED: patch read-back failed at %r" % (k, bad))
    ce_trained, t_t = ce_scored(opts, model, ids, opts.ce_chunk,
                                "prefix:%d(%s) ARM 2 TRAINED" % (k, device))

    rec = dict(k=k, device=device, pieces=pieces, resident_layers=k, dead_moe_layers=len(dead),
               claimable_coords=len(claimable), experts_per_resident_layer=60,
               params=int(sum(p.numel() for p in model.parameters())),
               ce_base=ce_base, ce_trained=ce_trained,
               delta=ce_trained - ce_base,
               frac_delta=(ce_base - ce_trained) / ce_base,
               patched=n_patched, load_s=round(load_s, 1),
               eval_s_base=t_b, eval_s_trained=t_t)
    ref = {1: (CE_1L_BASE, CE_1L_TRAINED), 2: (CE_2L_BASE, CE_2L_TRAINED)}.get(k)
    if ref:
        ok_b, ok_t = abs(ce_base - ref[0]) <= ANCHOR_TOL, abs(ce_trained - ref[1]) <= ANCHOR_TOL
        _log("[arc/ce/prefix:%d] PUBLISHED-ANCHOR reproduction: base %.6f vs %.6f |d|=%.2e -> %s | "
             "trained %.6f vs %.6f |d|=%.2e -> %s"
             % (k, ce_base, ref[0], abs(ce_base - ref[0]), "PASS" if ok_b else "FAIL",
                ce_trained, ref[1], abs(ce_trained - ref[1]), "PASS" if ok_t else "FAIL"))
        rec.update(ref_base=ref[0], ref_trained=ref[1], tol=ANCHOR_TOL,
                   repro_base=ok_b, repro_trained=ok_t, anchor_pass=bool(ok_b and ok_t))
        if not (ok_b and ok_t):
            raise SystemExit("[arc/ce/prefix:%d] BLOCKED: the published anchor did not reproduce" % k)
    if device.startswith("cuda"):
        rec["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated(0) / (1 << 30), 2)
    _log("[arc/ce/prefix:%d] ===== k=%d %s: base=%.6f trained=%.6f delta=%+.6f (%+.2f%% improvement)"
         " => %s | %.0fs + %.0fs ====="
         % (k, k, device, ce_base, ce_trained, ce_trained - ce_base, 100.0 * rec["frac_delta"],
            "IMPROVES" if ce_trained < ce_base else "WORSENS", t_b, t_t))
    del model, exp1
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
    return rec


def _ce_prefix_pack(opts, ids, d, k):
    """The same series point as `_ce_prefix`, built from the layer pack + streaming so k is not
    capped by VRAM. Validated by running both paths at the same k (they must agree to ~1e-3)."""
    import gc
    import torch
    cap, total_vram = _cap_vram()
    t0 = time.time()
    model, cfg, streamer = build_prefix_pack_model(opts.pack_dir, k, opts.pin_layers, "cuda",
                                                   ram_budget_gib=opts.prefix_ram_gib)
    load_s = time.time() - t0
    moe = _moe_layers(cfg)
    dead = [L for L in moe if L > k]
    still_live = [L for L in dead if hasattr(model.model.layers[L].mlp.experts, "gate_up_proj")]
    if still_live:
        raise SystemExit("[arc/ce/prefix-pack:%d] BLOCKED: layers %r above k still carry expert "
                         "weights -- they are not dead" % (k, still_live[:5]))
    missing = [L for L in moe if L <= k and not os.path.exists(_pack_path(opts.pack_dir, L))]
    dens = sorted({len(resident_experts_by_layer()[L]) for L in moe if L <= k})
    _log("[arc/ce/prefix-pack:%d] RESIDENT: %d transformer layers | %d MoE layers | live=%s | "
         "dead=%d | per-layer density=%s of %d | missing pack files=%s"
         % (k, len(model.model.layers), len(moe), _fmt_ranges([L for L in moe if L <= k]),
            len(dead), dens, int(cfg.n_routed_experts), missing if missing else "[]"))
    if missing or dens != [60] or len(dead) != len(moe) - k:
        raise SystemExit("[arc/ce/prefix-pack:%d] BLOCKED: residency/density/pack-file check failed" % k)

    gu1, dn1 = streamer.cache[1]
    bad = _rows_match(gu1, dn1, d["pristine"], lambda L: L == 1)
    _log("[arc/ce/prefix-pack:%d] CROSS-PATH GATE (packed rows == piece_loader rows, bit-exact): "
         "%d/%d -> %s" % (k, len(d["pristine"]) - len(bad), len(d["pristine"]),
                          "PASS" if not bad else "FAIL"))
    if bad:
        raise SystemExit("[arc/ce/prefix-pack:%d] BLOCKED: pristine rows differ between loaders at %r"
                         % (k, bad))

    b0 = streamer.bytes_streamed
    ce_base, t_b = ce_scored(opts, model, ids, opts.ce_chunk, "prefix-pack:%d ARM 1 BASE" % k,
                             streamer=streamer)
    gib_arm = (streamer.bytes_streamed - b0) / (1 << 30)
    n_patched = 0
    with torch.no_grad():
        for key, snap in d["foldedrows"].items():
            L, E = (int(x) for x in key.split(":"))
            if L != 1:
                raise SystemExit("[arc/ce/prefix-pack] BLOCKED: delta targets layer %d, not 1" % L)
            gu1[E].copy_(snap["gate_up"].to(gu1.device))
            dn1[E].copy_(snap["down"].to(dn1.device))
            n_patched += 1
    bad = _rows_match(gu1, dn1, d["foldedrows"], lambda L: L == 1)
    if bad:
        raise SystemExit("[arc/ce/prefix-pack:%d] BLOCKED: patch read-back failed at %r" % (k, bad))
    ce_trained, t_t = ce_scored(opts, model, ids, opts.ce_chunk, "prefix-pack:%d ARM 2 TRAINED" % k,
                                streamer=streamer)
    peak = torch.cuda.max_memory_allocated(0) / (1 << 30)
    rec = dict(k=k, device="cuda", loader="pack", pieces="pack:layers 1-%d" % k,
               resident_layers=k, dead_moe_layers=len(dead),
               claimable_coords=60 * k, experts_per_resident_layer=60,
               pinned_layers=sorted(streamer.pin), ram_cached_layers=len(streamer.ram),
               ce_base=ce_base, ce_trained=ce_trained, delta=ce_trained - ce_base,
               frac_delta=(ce_base - ce_trained) / ce_base, patched=n_patched,
               load_s=round(load_s, 1), eval_s_base=t_b, eval_s_trained=t_t,
               gib_per_arm=round(gib_arm, 1), gpu_peak_gib=round(peak, 2),
               vram_cap_gib=cap, vram_total_gib=round(total_vram, 2))
    _log("[arc/ce/prefix-pack:%d] ===== k=%d cuda/pack: base=%.6f trained=%.6f delta=%+.6f "
         "(%+.2f%% improvement) => %s | %.0fs + %.0fs | GPU peak %.2f GiB ====="
         % (k, k, ce_base, ce_trained, ce_trained - ce_base, 100.0 * rec["frac_delta"],
            "IMPROVES" if ce_trained < ce_base else "WORSENS", t_b, t_t, peak))
    streamer.close()
    del model, streamer, gu1, dn1
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    return rec


# =============================================================== DOSE (1-D line search on alpha)
# WHY THIS EXISTS. Campaign 31627ec8f184cfd6's 84 accepted deltas made the full 47-layer model
# WORSE (held-out CE 4.816991 -> 5.251066, ARC-Easy 0.8237 -> 0.3375). That measurement cannot tell
# apart two very different worlds: the deltas point in the WRONG DIRECTION (nothing rescues them),
# or they point in a useful direction and are merely OVER-DOSED (a scalar line search rescues them).
# The `fold` stage already snapshots the affected expert rows both PRISTINE and FOLDED, so the whole
# dose curve is available by linear interpolation for FREE -- forward passes only, no retraining.
DOSE_ZERO_TOL = 1e-5    # alpha=0 must reproduce the base arm to 5 decimals or the sweep is refused

DOSE_PREREGISTRATION = {
    "question": "Are a campaign's accepted deltas DIRECTIONALLY WRONG, or merely OVER-DOSED?",
    "construction": "w(alpha) = pristine + alpha * (folded - pristine), applied to the patched "
                    "expert rows at PATCH time (never at fold time -- the fold gate still has to "
                    "reproduce the recorded final root at full dose). alpha=0 and alpha=1 are "
                    "SELECTED bit-exactly from the fold snapshots, not computed, so they are "
                    "controls rather than approximations.",
    "clauses": [
        "MONOTONE: CE monotonically increasing in alpha -> the deltas are DIRECTIONALLY WRONG; "
        "weight-space delta mining is dead and no dose rescues it.",
        "MINIMUM: a clear minimum at some alpha* in (0,1) with CE(alpha*) < CE(0) -> it is a DOSE "
        "problem; the merge is recoverable by line-searching alpha against a secret held-out.",
        "NO-DOSE-HELPS: CE(alpha) >= CE(0) for every alpha > 0, flat or noisy -> no dose helps; "
        "the same practical conclusion as MONOTONE, but it must be STATED AS SUCH rather than "
        "reported as a monotonicity that was not observed.",
    ],
    "rule": "Report which clause fired. Do not invent a fourth reading to make the result look "
            "better.",
    "written_before_results": True,
}


def parse_dose(s):
    """--dose -> ordered, de-duplicated list of alphas. A single value is the historical scalar."""
    out = []
    for tok in str(s).replace(" ", "").split(","):
        if not tok:
            continue
        a = float(tok)
        if not (0.0 <= a <= 1.0):
            raise SystemExit("[arc/ce/dose] FATAL: alpha %r is outside [0,1]" % a)
        if a not in out:
            out.append(a)
    if not out:
        raise SystemExit("[arc/ce/dose] FATAL: --dose resolved to no alphas")
    return out


def dose_rows(d, alpha):
    """w(alpha) = pristine + alpha * (folded - pristine), per fused expert row.

    THE ENDPOINTS ARE SELECTED, NOT COMPUTED. alpha==0 returns the fold stage's PRISTINE snapshot
    and alpha==1 its FOLDED snapshot -- the identical objects the un-dosed path patches. That is
    what makes them controls: the caller's read-back gate is bit-exact `torch.equal`, so a rounding
    difference at an endpoint would be a FAILURE, not a tolerance.

    (`import torch` sits below the endpoint branches on purpose: selecting an endpoint must be
    provable without a torch install, which is what lets tests/test_glm_arc_campaign_spec.py check
    it as pure python.)
    """
    if alpha == 0.0:
        return d["pristine"]
    if alpha == 1.0:
        return d["foldedrows"]
    import torch
    out = {}
    with torch.no_grad():
        for k, f in d["foldedrows"].items():
            p = d["pristine"][k]
            row = {}
            for part in ("gate_up", "down"):
                a, b = p[part], f[part]
                # accumulate in fp32, store back in the model's bf16: the difference of two bf16
                # values is exact in fp32, so alpha=0/1 would land exactly even on this path.
                row[part] = (a.float() + alpha * (b.float() - a.float())).to(a.dtype)
            out[k] = row
    return out


def _ce_full(opts, ids, d, sink=None):
    """THE MEASUREMENT: the gate metric on the FULL 47-layer model.

    ARM 1 is BASE (pristine rows) and is scored ONCE. Then one arm per --dose alpha, each patched
    from the fold snapshots by `dose_rows` -- so with the default --dose 1.0 this is exactly the
    two-arm BASE-vs-TRAINED measurement it has always been, and a longer --dose list is the same
    measurement repeated along the ray at no extra base cost.

    Same streaming machinery, same VRAM cap and same pin set as the ARC benchmark -- so this CE and
    that accuracy describe the SAME network.
    """
    import torch
    cap, total_vram = _cap_vram()
    pin = list(range(1, 1 + max(1, int(opts.pin_layers))))
    model, cfg, streamer = build_streaming_model(opts.pack_dir, pin, "cuda")

    # ---- GATE 2: full residency, zero dead layers ----
    layers = model.model.layers
    moe = _moe_layers(cfg)
    dead = [L for L in moe if not hasattr(model.model.layers[L].mlp.experts, "gate_up_proj")]
    missing = [L for L in moe if not os.path.exists(_pack_path(opts.pack_dir, L))]
    _log("[arc/ce/full] RESIDENT: %d transformer layers | %d MoE layers | dead_layers=%s | "
         "missing pack files=%s"
         % (len(layers), len(moe), dead if dead else "[]  (0 dead)", missing if missing else "[]"))
    if len(layers) != cfg.num_hidden_layers or len(moe) != 46 or dead or missing:
        raise SystemExit("[arc/ce/full] BLOCKED: expected 47 layers / 46 MoE / 0 dead / 0 missing, "
                         "got %d/%d/%d/%d" % (len(layers), len(moe), len(dead), len(missing)))

    # ---- CROSS-PATH GATE: packed layer-1 rows == piece_loader rows, bit-exact ----
    gu1, dn1 = streamer.cache[1]
    bad = _rows_match(gu1, dn1, d["pristine"], lambda L: L == 1)
    _log("[arc/ce/full] CROSS-PATH GATE (packed rows == piece_loader rows, bit-exact): %d/%d -> %s"
         % (len(d["pristine"]) - len(bad), len(d["pristine"]), "PASS" if not bad else "FAIL"))
    if bad:
        raise SystemExit("[arc/ce/full] BLOCKED: pristine expert rows differ between loaders at %r" % bad)

    b0 = streamer.bytes_streamed
    ce_base, t_b = ce_scored(opts, model, ids, opts.ce_chunk, "full-47 ARM 1 BASE", streamer=streamer)
    gib_arm = (streamer.bytes_streamed - b0) / (1 << 30)
    _log("[arc/ce/full] arm 1 streamed %.1f GiB of expert weights | GPU peak %.2f GiB"
         % (gib_arm, torch.cuda.max_memory_allocated(0) / (1 << 30)))

    moved = [k for k, v in d["coord_stats"].items() if v["moved"]]
    alphas = parse_dose(getattr(opts, "dose", "1.0"))
    _log("[arc/ce/full] DOSE LINE SEARCH: w(alpha) = pristine + alpha*(folded - pristine) | "
         "alphas = %s" % (", ".join("%g" % a for a in alphas),))

    points = []
    ce_trained, t_t = None, None

    def _envelope():
        return dict(device="cuda", resident_layers=len(moe), dead_moe_layers=0,
                    transformer_layers=len(layers), pinned_layers=pin,
                    ce_base=ce_base, ce_trained=ce_trained, patched=n_patched, coords_moved=moved,
                    eval_s_base=t_b, eval_s_trained=t_t,
                    dose=alphas, dose_points=list(points),
                    dose_preregistration=DOSE_PREREGISTRATION,
                    dose_zero_tol=DOSE_ZERO_TOL,
                    vram_cap_gib=cap, vram_total_gib=round(total_vram, 2),
                    gpu_peak_gib=round(torch.cuda.max_memory_allocated(0) / (1 << 30), 2),
                    gib_streamed_total=round(streamer.bytes_streamed / (1 << 30), 1))

    for alpha in alphas:
        rows = dose_rows(d, alpha)
        endpoint = {0.0: "pristine", 1.0: "folded"}.get(alpha)
        n_patched = 0
        with torch.no_grad():
            for k, snap in rows.items():
                L, E = (int(x) for x in k.split(":"))
                if L != 1:
                    raise SystemExit("[arc/ce/full] BLOCKED: delta targets layer %d; only layer 1 pinned" % L)
                gu1[E].copy_(snap["gate_up"].to(gu1.device))
                dn1[E].copy_(snap["down"].to(dn1.device))
                n_patched += 1
        # UNCHANGED GATE: the rows on the GPU must be bit-identical to what we meant to write. At
        # alpha=1 this is byte-for-byte the old `d["foldedrows"]` check.
        bad = _rows_match(gu1, dn1, rows, lambda L: L == 1)
        _log("[arc/ce/full] alpha=%.4f (%s) PATCHED %d coordinates | read-back mismatches: %d | "
             "coordinates whose FULL-dose weights differ from base: %d"
             % (alpha, endpoint or "interpolated", n_patched, len(bad), len(moved)))
        if bad:
            raise SystemExit("[arc/ce/full] BLOCKED: patch read-back failed at %r" % (bad,))
        rows = None

        ce_a, t_a = ce_scored(opts, model, ids, opts.ce_chunk,
                              "full-47 DOSE alpha=%.4f" % alpha, streamer=streamer)
        # CONTROL, checked before hours are spent on the rest of the ray: alpha=0 patches the
        # PRISTINE rows back in, so the network is the base network and the CE must be the base CE.
        # A mismatch means the interpolation/patch path is wrong and every later point is garbage.
        if alpha == 0.0:
            dev = abs(ce_a - ce_base)
            _log("[arc/ce/full] CONTROL alpha=0: dosed CE %.12f vs BASE arm %.12f | |d|=%.3e "
                 "(tol %.0e) -> %s" % (ce_a, ce_base, dev, DOSE_ZERO_TOL,
                                       "PASS" if dev <= DOSE_ZERO_TOL else "FAIL"))
            if dev > DOSE_ZERO_TOL:
                raise SystemExit("[arc/ce/full] BLOCKED: alpha=0 did not reproduce the BASE arm "
                                 "(|d|=%.3e > %.0e). The interpolation is wrong -- refusing to "
                                 "spend hours on the rest of the sweep." % (dev, DOSE_ZERO_TOL))
        elif ce_a == ce_base:
            _log("[arc/ce/full] *** WARNING: identical CE to 17 digits at alpha=%g -- the patch "
                 "may be a no-op. ***" % alpha)
        ce_trained, t_t = ce_a, t_a
        points.append(dict(alpha=alpha, endpoint=endpoint, ce=ce_a,
                           delta_vs_base=ce_a - ce_base, patched=n_patched, eval_s=t_a))
        if sink is not None:
            sink(_envelope())

    _log("[arc/ce/full] ===== DOSE CURVE (full-47 held-out CE, %d seqs) =====" % len(ids))
    _log("[arc/ce/full]   %8s %-12s %12s %12s %9s" % ("alpha", "rows", "CE", "d vs base", "eval s"))
    _log("[arc/ce/full]   %8.4f %-12s %12.6f %12s %9.0f"
         % (0.0, "BASE arm", ce_base, "--", t_b))
    for p in points:
        _log("[arc/ce/full]   %8.4f %-12s %12.6f %+12.6f %9.0f"
             % (p["alpha"], p["endpoint"] or "interp", p["ce"], p["delta_vs_base"], p["eval_s"]))
    streamer.close()
    return _envelope()


def _ce_base(opts):
    """--ce-mode base: THE AUDIT PATH. Score the UNPATCHED full-47 model and nothing else.

    Why it has to exist for the reference to be reproducible at all: every other --ce-mode requires
    a `--deltas` snapshot that passes the fold gate, i.e. a campaign's accepted records. The
    published reference 4.8100134711111746 is the BASE arm -- no deltas were ever folded into it
    (tools/glm_ce_layer_major_gate.py builds the streaming model and calls gate_ce directly). So
    "reproduce the paid number from the CLI" was impossible not because of a wrong default but
    because the CLI had no mode that scores a base model. This is that mode.

    Same builder, same streamer, same `ce_scored` seam and the same CROSS-PATH-free path the other
    modes use -- the ONLY thing removed is the patching, because there is nothing to patch.
    """
    import torch
    cap, total_vram = _cap_vram()
    pin = list(range(1, 1 + max(1, int(opts.pin_layers))))
    model, cfg, streamer = build_streaming_model(opts.pack_dir, pin, "cuda")
    layers = model.model.layers
    moe = _moe_layers(cfg)
    dead = [L for L in moe if not hasattr(model.model.layers[L].mlp.experts, "gate_up_proj")]
    missing = [L for L in moe if not os.path.exists(_pack_path(opts.pack_dir, L))]
    _log("[arc/ce/base] RESIDENT: %d transformer layers | %d MoE layers | dead_layers=%s | "
         "missing pack files=%s"
         % (len(layers), len(moe), dead if dead else "[]  (0 dead)", missing if missing else "[]"))
    if len(layers) != cfg.num_hidden_layers or len(moe) != 46 or dead or missing:
        raise SystemExit("[arc/ce/base] BLOCKED: expected 47 layers / 46 MoE / 0 dead / 0 missing, "
                         "got %d/%d/%d/%d" % (len(layers), len(moe), len(dead), len(missing)))
    ids, digest = ce_ids(opts)
    b0 = streamer.bytes_streamed
    ce, secs = ce_scored(opts, model, ids, opts.ce_chunk, "base-47 (no deltas)", streamer=streamer)
    gib = (streamer.bytes_streamed - b0) / (1 << 30)
    n_seqs, chunk, sub_batch, pin_desc = ce_config_tuple(ids, opts.ce_chunk)
    _log("[arc/ce/base] ===== BASE held-out CE = %r =====" % ce)
    _log("[arc/ce/base] config (n_seqs, chunk, sub_batch, pin) = (%d, %d, %d, %s)"
         % (n_seqs, chunk, sub_batch, pin_desc))
    _log("[arc/ce/base] published reference %r at (256, 32, 8, [1, 2]) -> %s"
         % (CE_REFERENCE_256_32_8_PIN2,
            "EXACT MATCH" if ce == CE_REFERENCE_256_32_8_PIN2 else
            "differs by %+.3e (this is a DIFFERENT configuration unless the tuple above reads "
            "(256, 32, 8, [1, 2]))" % (ce - CE_REFERENCE_256_32_8_PIN2)))
    streamer.close()
    return dict(device="cuda", resident_layers=len(moe), dead_moe_layers=0,
                transformer_layers=len(layers), pinned_layers=pin,
                ce_base=ce, ce_trained=None, patched=0, coords_moved=[],
                eval_s_base=secs, eval_s_trained=None,
                reference=CE_REFERENCE_256_32_8_PIN2,
                reference_config=[256, 32, 8, [1, 2]],
                reference_exact=bool(ce == CE_REFERENCE_256_32_8_PIN2),
                vram_cap_gib=cap, vram_total_gib=round(total_vram, 2),
                gpu_peak_gib=round(torch.cuda.max_memory_allocated(0) / (1 << 30), 2),
                gib_streamed_total=round(gib, 1)), ids, digest


def stage_ce(opts):
    import torch
    # THE INNER SUB-BATCH IS A MEASUREMENT PARAMETER, NOT AN IMPLEMENTATION DETAIL (issue #125).
    # Historically this line was `os.environ[...] = str(int(opts.ce_chunk))` unconditionally: inner
    # == outer, so `--ce-chunk 32` also forced the sub-batch to 32, while the published reference
    # 4.8100134711111746 was produced with the SHIPPED DEFAULT of 8. That combination was not
    # expressible from this CLI at all, so an auditor reproducing the paid number got a different
    # value with no way to tell configuration from regression.
    #
    # HOW DIFFERENT, MEASURED 2026-08-10 on the real 47-layer model, one variable changed:
    #     (256, 32,  8, pin[1,2]) -> 4.810013471111175   == the published reference, EXACT
    #     (256, 32, 32, pin[1,2]) -> 4.813307792909684   +3.294e-03
    # 3.3 MILLINATS. The old comment here claimed the sub-batch only moves "fp summation order";
    # that is FALSE and the measurement refutes it by ~10 orders of magnitude (fp64 accumulation
    # over 7936 token log-probs is ~1e-13). The sub-batch is the FORWARD BATCH SIZE, and this
    # project already recorded that the expert GEMM is batch-composition-sensitive (memory
    # glm52-milestone) -- so a different sub-batch is a genuinely different computation through the
    # MoE path, not a re-association of the same one. A gap that size is the same order as the
    # unexplained 4.8170-vs-4.826324 discrepancy on the public page.
    #
    # --ce-sub-batch now names it explicitly. Default 0 == mirror --ce-chunk == byte-for-byte the
    # old behaviour, so no existing invocation moves. It is a FLAG rather than "respect the ambient
    # NEURAHASH_GLM_EVAL_CHUNK" on purpose: a machine-wide stray env var is exactly how
    # NEURAHASH_GLM_CAP_GB=7.4 hid a real bug for weeks (memory stranger-4060-join-proof), and a
    # value that appears in the command line is auditable while ambient state is not. The env var
    # is still what the scorer reads, so it is SET here, from the flag, and printed in the tuple.
    sub = int(getattr(opts, "ce_sub_batch", 0) or 0) or int(opts.ce_chunk)
    os.environ["NEURAHASH_GLM_EVAL_CHUNK"] = str(max(1, sub))
    # Fail on a misplaced --dose in SECONDS, not after a 2.2 GB torch.load and a model build. Only
    # --ce-mode full implements the ray; silently ignoring the flag elsewhere would hand back a
    # full-dose number wearing a dose label.
    alphas = parse_dose(getattr(opts, "dose", "1.0"))
    if alphas != [1.0] and opts.ce_mode != "full":
        raise SystemExit("[arc/ce] --dose %r is implemented for --ce-mode full only (got %r)"
                         % (opts.dose, opts.ce_mode))
    _log("[arc/ce] torch %s | CUDA_VISIBLE_DEVICES=%r | NEURAHASH_GLM_EVAL_CHUNK=%s | mode=%s "
         "| dose=%s"
         % (torch.__version__, os.environ.get("CUDA_VISIBLE_DEVICES"),
            os.environ["NEURAHASH_GLM_EVAL_CHUNK"], opts.ce_mode,
            ",".join("%g" % a for a in alphas)))

    # --ce-mode base scores the UNPATCHED model, so there is no deltas file, no fold gate and no
    # arm 2. It returns before any of that rather than threading `d = None` through gates whose
    # whole job is to validate a payload that does not exist here.
    if opts.ce_mode == "base":
        env, ids, digest = _ce_base(opts)
        out = _ce_out_dict(opts, ids, digest, None, dict(base=env))
        os.makedirs(os.path.dirname(os.path.abspath(opts.ce_out)), exist_ok=True)
        with open(opts.ce_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        _log("[arc/ce] wrote %s" % opts.ce_out)
        return 0

    d = torch.load(opts.deltas, map_location="cpu", weights_only=False)
    log_spec("arc/ce")
    # Same rule as bench: the target root and the expected fold count are the SPEC's, not the
    # payload's own. `!= len(spec_events())` is run 5's `!= 25` generalised -- every accepted record
    # in the spec has to have folded AND moved the root, or the arm 2 weights are not the campaign's.
    n_expect = len(spec_events())
    # A PREFIX deltas file (fold --max-records N) cannot reproduce the campaign's FINAL root by
    # construction, so the root check is replaced -- not dropped -- by an exact match on what the
    # file itself DECLARES it is: N, and the first N accepted events of this same spec. That way a
    # full-fold .pt can never be scored as a prefix point, and a prefix .pt for a different N can
    # never be scored as this one.
    n_max = int(getattr(opts, "max_records", 0) or 0)
    prefix = bool(n_max and n_max < n_expect)
    if prefix:
        n_expect = n_max
    root_ok = d["root_folded"] == spec_root()
    if prefix:
        root_ok = (int(d.get("prefix_records") or 0) == n_expect
                   and [int(e) for e in (d.get("accepted_events") or [])] == spec_events()[:n_expect])
    gate1 = bool(d.get("gate_pass")) and root_ok \
        and str(d.get("campaign")) == spec_campaign()
    _log("[arc/ce] GATE 1 fold: %d/%d records folded | post-fold model_root" % (d["folded"], d["requested"]))
    _log("[arc/ce]   got  = %s" % d["root_folded"])
    _log("[arc/ce]   want = %s"
         % ("PREFIX %d/%d: no recorded target root; requires prefix_records=%d over events %r"
            % (n_expect, len(spec_events()), n_expect, spec_events()[:n_expect])
            if prefix else spec_root()))
    _log("[arc/ce]   -> %s | fold blast radius: %s"
         % ("PASS" if gate1 else "FAIL", d.get("changed_tensors", "not recorded")))
    if not gate1 or d["folded"] != n_expect:
        raise SystemExit("[arc/ce] BLOCKED: deltas file did not pass the fold gate "
                         "(gate1=%s folded=%s expected=%d)" % (gate1, d.get("folded"), n_expect))

    ids, digest = ce_ids(opts)

    res = {}
    ks = []

    def _dump(res_now):
        out = _ce_out_dict(opts, ids, digest, d, res_now)
        os.makedirs(os.path.dirname(os.path.abspath(opts.ce_out)), exist_ok=True)
        with open(opts.ce_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        return out

    if opts.ce_mode in ("anchor-cpu", "anchor-cuda"):
        res["anchor"] = _ce_anchor(opts, ids, "cpu" if opts.ce_mode.endswith("cpu") else "cuda", d)
    elif opts.ce_mode == "full":
        def _sink(partial):
            # written after EVERY dose point: a 6-point ray that dies at alpha=0.75 must not lose
            # the four points already paid for (same rule as the prefix sweep below).
            res["full"] = partial
            _dump(res)
            _log("[arc/ce] wrote %s (%d/%d dose points)"
                 % (opts.ce_out, len(partial["dose_points"]), len(partial["dose"])))
        res["full"] = _ce_full(opts, ids, d, sink=_sink)
    elif opts.ce_mode == "prefix":
        ks = [int(x) for x in str(opts.prefix_k).replace(" ", "").split(",") if x]
        if not ks:
            raise SystemExit("[arc/ce] --ce-mode prefix needs --prefix-k (e.g. --prefix-k 3,4,6)")
        res["prefix"] = []
    else:
        raise SystemExit("[arc/ce] unknown --ce-mode %r" % opts.ce_mode)

    for k in ks:
        # written after EVERY point: a sweep that dies at k=12 must not lose k=3..8.
        if opts.prefix_loader == "pack":
            res["prefix"].append(_ce_prefix_pack(opts, ids, d, k))
        else:
            res["prefix"].append(_ce_prefix(opts, ids, _prefix_device(opts, k), d, k))
        _dump(res)
        _log("[arc/ce] wrote %s (%d/%d points)" % (opts.ce_out, len(res["prefix"]), len(ks)))
    if ks:
        _log("[arc/ce] ===== SERIES (k = resident MoE layers of 46) =====")
        _log("[arc/ce]   %2s %-5s %10s %10s %10s %9s  %s" %
             ("k", "dev", "base", "trained", "delta", "impr%", "sign | eval s (base+trained)"))
        for r in res["prefix"]:
            _log("[arc/ce]   %2d %-5s %10.5f %10.5f %+10.5f %+8.2f%%  %s | %.0f+%.0f"
                 % (r["k"], r["device"], r["ce_base"], r["ce_trained"], r["delta"],
                    100.0 * r["frac_delta"], "IMPROVES" if r["delta"] < 0 else "WORSENS",
                    r["eval_s_base"], r["eval_s_trained"]))
        return 0

    out = _ce_out_dict(opts, ids, digest, d, res)
    os.makedirs(os.path.dirname(os.path.abspath(opts.ce_out)), exist_ok=True)
    with open(opts.ce_out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    _log("[arc/ce] wrote %s" % opts.ce_out)
    r = res.get("anchor") or res["full"]
    _log("[arc/ce] ===== %s: base=%.6f trained=%.6f delta=%+.6f (%.2f%%) ====="
         % (opts.ce_mode, r["ce_base"], r["ce_trained"], r["ce_trained"] - r["ce_base"],
            100.0 * (r["ce_base"] - r["ce_trained"]) / r["ce_base"]))
    return 0


def _ce_out_dict(opts, ids, digest, d, res):
    """The result envelope -- every gate that licensed the numbers travels WITH them.

    `d` is None for --ce-mode base: there is no deltas payload, so every field lifted from one is
    reported as absent rather than as a default that would read like a passed gate.
    """
    import torch
    n_seqs, chunk, sub_batch, pin_desc = ce_config_tuple(ids, opts.ce_chunk)
    return dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               mode=opts.ce_mode, ce_chunk=int(opts.ce_chunk),
               # THE CONFIGURATION THIS CE IS A NUMBER FOR (issue #125). Four knobs move the value;
               # without them on the artifact a mismatch is unattributable.
               ce_config=dict(n_seqs=n_seqs, chunk=chunk, sub_batch=sub_batch, pin=pin_desc),
               ce_sub_batch=sub_batch, ce_seqs=int(getattr(opts, "ce_seqs", 0) or 0),
               heldout_path=HELDOUT_PATH, heldout_sha256=digest,
               heldout_shape=list(ids.shape), heldout_dtype=str(ids.dtype),
               # WHICH DRIVER produced these floats travels with them. The two paths are gated
               # bit-identical, but "we believe they are equal" is not a reason to publish a
               # number that cannot say which one it came from.
               ce_function=("tools/glm_arc_eval.heldout_ce_layer_major (layer-major inversion of "
                            "sharddiloco_glm_coordinator._chunked_heldout_ce over "
                            "sharddiloco_glm_expert.heldout_ce)"
                            if bool(getattr(opts, "ce_layer_major", False)) else
                            "sharddiloco_glm_coordinator._chunked_heldout_ce over "
                            "sharddiloco_glm_expert.heldout_ce"),
               ce_layer_major=bool(getattr(opts, "ce_layer_major", False)),
               ce_lm_group=int(getattr(opts, "ce_lm_group", 0) or 0),
               ref_2layer=dict(base=CE_2L_BASE, trained=CE_2L_TRAINED,
                               source="docs/research/glm_layer_composition.json configs C/D"),
               torch=torch.__version__, **_ce_fold_fields(d), **res)


def _ce_fold_fields(d):
    """The half of the CE envelope lifted from the DELTAS payload -- split out so --ce-mode base,
    which has no payload at all, cannot reach the delta-derived expressions rather than being
    threaded through them with `if d is not None` at every field."""
    if d is None:
        # NOT "PASS": a base-mode artifact claiming a passed fold gate would be the exact false
        # provenance this envelope exists to prevent. Nothing was folded, so every field says so --
        # INCLUDING the campaign fields. A base CE is a property of the BASE MODEL and has nothing
        # to do with whichever campaign --campaign-spec happened to resolve; stamping run 5's id on
        # it would let it be read as run 5's number.
        return dict(accepted_events=[], prefix_records=None, slots=[], fold_root=None,
                    fold_gate="N/A (--ce-mode base: no deltas folded)", folded_records="0/0",
                    fold_changed_tensors=None, coord_stats={},
                    campaign=None, campaign_spec=None, fold_root_expected=None)
    return dict(
        # the events the DELTAS FILE actually folded, not the spec's full list: a prefix
        # point that reported all 84 would mis-attribute its own number (spec_name()'s lesson)
        accepted_events=[int(e) for e in (d.get("accepted_events") or spec_events())],
        prefix_records=d.get("prefix_records"), slots=d["slots"],
        fold_root=d["root_folded"],
        fold_gate="PASS", folded_records="%d/%d" % (d["folded"], d["requested"]),
        fold_changed_tensors=d.get("changed_tensors"),
        coord_stats=d["coord_stats"],
        campaign=spec_campaign(), campaign_spec=dict(SPEC), fold_root_expected=spec_root())


# ============================================================================== STAGE -- REPORT
def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


# The smallest positive binary64. Named because it is returned as a CLAMP, never as a measurement.
_MIN_POSITIVE_DOUBLE = 5e-324


def _mcnemar_p(b, c):
    """Two-sided EXACT binomial p on the discordant pairs (the correct paired test: both arms
    scored the SAME items, so only items where they DISAGREE carry information).

    STILL EXACT PAST n=1023, which the old form was not. It divided by `2.0 ** n` -- a C double,
    so `2.0 ** 1024` raises OverflowError -- and the test therefore died exactly when the effect
    was LARGE, because a large effect is what makes most items discordant. Measured on
    docs/research/ARC_EASY_31627ec8_2026-08-06.json: b=1207, c=75, n=1282 discordant. `--stage
    report` could not render that artifact at all; it crashed at this line.

    The repair is a DENOMINATOR, not an approximation. `1 << n` is an exact Python int, and
    int/int true division is correctly rounded for operands of any size -- CPython divides the
    integers directly and never materialises an intermediate float, so nothing can overflow while
    the quotient itself is in range (and here it always is: the partial sum is bounded by 2**n,
    so the quotient lies in (0, 1]). Verified bit-identical to the old code over 9165 (b, c)
    pairs covering every n in 0..1023 -- zero differences -- so published exact p-values
    (p=1.25e-40, p=9.17e-31) remain comparable digit for digit. No lgamma, no normal or
    chi-square approximation, no new dependency.

    The partial sum uses the exact integer recurrence comb(n, i+1) == comb(n, i) * (n-i) // (i+1)
    (always divisible, so `//` loses nothing) instead of recomputing comb() per term. Same
    integer, less work at n in the thousands.

    ONE approximation survives, and it is announced rather than silent: below about 5e-324 no
    binary64 can hold the answer. Returning 0.0 there would print "p=0", which is not a p-value,
    and would hide how far below the floor the truth sits. That case logs the real base-10
    exponent (derived from bit lengths, no float involved) and returns the smallest positive
    double as an explicit clamp.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    term = 1                                    # comb(n, 0)
    total = 1                                   # sum over i in 0..k, built exactly in integers
    for i in range(k):
        term = term * (n - i) // (i + 1)        # == comb(n, i + 1)
        total += term
    tail = total / (1 << n)                     # exact operands, correctly rounded quotient
    if tail == 0.0:
        exp10 = (total.bit_length() - 1 - n) * math.log10(2.0)
        _log("[arc/report] McNemar p UNDERFLOWS binary64 at n=%d discordant (b=%d, c=%d): the "
             "true two-sided p is about 1e%.0f. Reporting %r -- that is the CLAMP at the "
             "smallest positive double, not the measured value."
             % (n, b, c, exp10, _MIN_POSITIVE_DOUBLE))
        return _MIN_POSITIVE_DOUBLE
    return min(1.0, 2.0 * tail)


def _result_spec(r):
    """The campaign THIS RESULT FILE was produced by -- read from the file, never from the module
    default. `--stage report` can run long after (or on a different box from) the bench that made
    the JSON, so taking the campaign from SPEC would let a default (or a --campaign-spec typo)
    describe someone else's numbers. `campaign_spec` is written by every bench since the tool was
    parameterised; older files (the published run-5 JSON, generated 2026-07-27) carry only
    campaign/accepted_events/slots, so those are lifted and the LABEL is derived from the id --
    it is never assumed to be run 5's."""
    s = dict(r.get("campaign_spec") or {})
    if not s:
        s = dict(campaign=r.get("campaign") or "unknown",
                 accepted_events=list(r.get("accepted_events") or []),
                 slots=[list(c) for c in (r.get("slots") or [])],
                 final_root=r.get("fold_root_expected") or r.get("fold_root") or "",
                 frozen_cut_event=None)
        s["label"] = (RUN5_SPEC["label"] if s["campaign"] == RUN5_SPEC["campaign"]
                      else str(s["campaign"]))
    return s


def _result_name(s):
    """Prose name for a result's campaign: label AND id, so no paragraph can be read as being about
    a different run than the one that produced its numbers."""
    return "%s (campaign %s)" % (s.get("label", "unknown"), s.get("campaign", "unknown"))


# =============================================== RESOLUTION: margin + Brier from the SAME scores
# WHY. Accuracy is a STEP FUNCTION of the per-option log-likelihoods `run_arm` already computed and
# already wrote to the result JSON (:1647-1651, keys `scores_sum` / `scores_mean` / `gold`). It
# throws away everything except the sign of one comparison, and this project is currently blocked on
# resolution, not on ideas: task #115's +1.68pp at n=119 HALVED to +0.85pp at n=235 with McNemar
# b=13 c=15, p=0.85 against an MDE of 7.44pp -- we could not resolve our own effect.
#
# Everything below is PURE POST-PROCESSING of numbers already on disk. It runs in `stage_report`,
# never in `stage_bench`, for one reason: a metric that is computed from the stored scores CANNOT
# move the accuracy or the CE, because it does not touch the code that produces them. It also means
# every result file this tool has ever written -- including the published ones -- gains the new
# metrics retroactively, with no GPU-hours.
#
# THREE quantities, and the difference between them is load-bearing (measured, see
# docs/research/ARC_RESOLUTION_MARGIN_BRIER_2026-08-10.md):
#   margin        chosen option's score MINUS the runner-up's. Always >= 0. This is CONFIDENCE, and
#                 it is deliberately NOT the sensitivity lever: it is blind to WHICH option won, so
#                 an arm that becomes confidently wrong scores the same as one that becomes
#                 confidently right. Measured, it is WORSE than accuracy (0.22x-1.36x the power).
#   margin_gold   the GOLD option's score minus the best non-gold option. Signed: `margin_gold >= 0`
#                 iff the item is correct, with equality EXACTLY on a tie (and ties are real here --
#                 9 of run 5's 1997 base items, see item_resolution). It is the continuous
#                 relaxation whose threshold at zero is the accuracy indicator, and it is the one
#                 that buys resolution. Do NOT "tighten" this to `sign(margin_gold) == correct`:
#                 that strict form is FALSE on the published data and would block every render.
#   brier         multi-class Brier over the options renormalised to probabilities. Proper scoring
#                 rule, bounded [0, 2], lower is better; sensitive to the whole distribution rather
#                 than to the top-2 gap alone.
def _softmax(xs):
    """Options -> probabilities, renormalised over the option set.

    Shift-invariant, so an additive constant common to all options cannot move it. That does NOT
    make the result a calibrated posterior: the inputs are per-option log-likelihoods of DIFFERENT
    continuations, which differ by an option-dependent term rather than a constant. The Brier score
    over them is well-defined and comparable BETWEEN ARMS on the same items, which is all it is used
    for here -- it is not a probability the model would report.
    """
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def item_resolution(p, key):
    """margin / margin_gold / brier for ONE item, from the per-option scores under `key`.

    Returns None when the item predates the scores being stored, or is malformed. That is not
    defensive padding: `stage_report` must keep working on any result file this tool has ever
    written, and a missing metric has to read as ABSENT, never as zero.
    """
    sc = p.get(key)
    gold = p.get("gold")
    if not isinstance(sc, (list, tuple)) or len(sc) < 2 or gold is None:
        return None
    gold = int(gold)
    if not (0 <= gold < len(sc)):
        return None
    sc = [float(x) for x in sc]
    # `sorted` is stable and `run_arm` uses `max`, which also returns the FIRST maximal element
    # (`p_sum`/`p_mean`, run_arm) -- so the tie-break here is the one that produced the published
    # accuracy. That agreement is not assumed: paired_resolution re-checks it item by item against
    # the `ok_mean`/`ok_sum` the bench stored.
    order = sorted(range(len(sc)), key=lambda i: -sc[i])
    top, runner = order[0], order[1]
    best_other = max(sc[i] for i in range(len(sc)) if i != gold)
    pr = _softmax(sc)
    mg = sc[gold] - best_other
    return dict(chosen=top, correct=1.0 if top == gold else 0.0,
                margin=sc[top] - sc[runner],
                margin_gold=mg,
                # EXACT ties are real here and not rare: a single-token answer scored in bf16
                # collides outright (measured 9 of 1997 items in the published run-5 base arm, e.g.
                # Mercury_SC_405062 = [-15.6165847778, -13.6165847778, -9.6165847778,
                # -9.6165847778]). On those items ACCURACY is decided by choice ORDER, not by the
                # model, while margin_gold correctly reports 0.0 = no evidence either way. Counting
                # them is the point: they are a resolution defect accuracy cannot express.
                tie=1.0 if mg == 0.0 else 0.0,
                brier=sum((pr[i] - (1.0 if i == gold else 0.0)) ** 2 for i in range(len(sc))),
                p_gold=pr[gold])


RESOLUTION_METRICS = ("correct", "margin", "margin_gold", "brier")
# 80% power, two-sided alpha=0.05: z_{1-a/2} + z_{power}. Required N for a paired test scales as
# (this / d_z)^2, which is the whole basis of the "items saved" column.
_Z80 = 1.959963984540054 + 0.8416212335729143


def _mean_sd(xs):
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), 0.0
    mu = sum(xs) / n
    return mu, math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def _dz(xs):
    """Paired standardized effect size mean/sd -- the scale-free currency in which a continuous
    metric and a 0/1 indicator can be compared at all."""
    mu, sd = _mean_sd(xs)
    return mu, sd, (mu / sd if sd > 0 else 0.0)


def _n_for_power(dz_):
    """Items needed for 80% power at the OBSERVED effect. inf when the effect is indistinguishable
    from zero -- reporting a finite number there would be the underpowered-result trap itself."""
    return (_Z80 / abs(dz_)) ** 2 if dz_ else float("inf")


def paired_resolution(pb, pt, key, boot=1000, seed=20260810):
    """Per-item paired differences of every resolution metric, plus the MEASURED sensitivity gain.

    THE RATIO IS MEASURED HERE, NEVER ASSERTED. For a paired test the required N is proportional to
    1/d_z^2, so `items_saved = (d_z_metric / d_z_correct)^2` is exactly "how many fewer items this
    metric needs to detect the SAME perturbation at the same power". Both arms scored the identical
    items in the identical order, so the pairing is exact and no unit conversion is involved.

    The CI comes from a PAIRED bootstrap over items: one resample of item indices is shared by every
    metric, so the strong dependence between d_z_metric and d_z_correct (same items, same model) is
    preserved. Bootstrapping each metric separately would understate the CI badly.

    `resolvable` is the honest guard. When the binary arm's own effect is not significant, the ratio
    is a quotient of two noise terms and its CI spans orders of magnitude -- so the ratio is
    reported UNRESOLVED rather than as a number someone could quote.
    """
    B = [item_resolution(p, key) for p in pb]
    T = [item_resolution(p, key) for p in pt]
    if any(x is None for x in B) or any(x is None for x in T) or not B:
        return None
    if len(B) != len(T):
        # zip() would silently truncate and the per-item loop below would then IndexError AFTER the
        # accuracy markdown was built but BEFORE it was written -- losing the whole report over an
        # optional section. Decline the section instead.
        _log("[arc/report] resolution SKIPPED for %s: arms have %d and %d items"
             % (key, len(B), len(T)))
        return None
    n = len(B)
    diffs = {m: [t[m] - b[m] for b, t in zip(B, T)] for m in RESOLUTION_METRICS}
    # SELF-CHECK, and it has to be against an INDEPENDENT producer to mean anything. `correct` and
    # `margin_gold` are both derived from the same `sc` array three lines apart in item_resolution,
    # so "correct => margin_gold >= 0" is a theorem of that code, not a check on it -- no input can
    # trip it. The real risk is that the tie-break here drifts from the one `run_arm` used, which
    # would silently make margin_gold the relaxation of a DIFFERENT accuracy than the published one.
    # So compare the recomputed verdict against `ok_mean`/`ok_sum` as the BENCH stored them: a
    # different code path, at a different time, on a different machine.
    stored_key = {"scores_mean": "ok_mean", "scores_sum": "ok_sum"}.get(key)
    for arm, recomputed in ((pb, B), (pt, T)):
        for p, x in zip(arm, recomputed):
            if (x["correct"] > 0.5 and x["margin_gold"] < 0) or \
                    (x["correct"] <= 0.5 and x["margin_gold"] > 0):
                raise SystemExit("[arc/report] BLOCKED: margin_gold (%r) contradicts the "
                                 "correctness indicator (%r) -- the metrics are inconsistent."
                                 % (x["margin_gold"], x["correct"]))
            if stored_key in p and bool(p[stored_key]) != (x["correct"] > 0.5):
                raise SystemExit(
                    "[arc/report] BLOCKED: item %r -- this file's stored %s is %r but rescoring "
                    "its per-option %s says %r. The tie-break or the argmax has drifted from the "
                    "one that produced the published accuracy, so margin_gold is the relaxation "
                    "of a different metric. Refusing to report a sensitivity ratio over it."
                    % (p.get("id"), stored_key, bool(p[stored_key]), key, x["correct"] > 0.5))
    ties = dict(base=sum(x["tie"] for x in B), trained=sum(x["tie"] for x in T))
    out = {}
    for m in RESOLUTION_METRICS:
        mu, sd, z = _dz(diffs[m])
        out[m] = dict(base_mean=_mean_sd([x[m] for x in B])[0],
                      trained_mean=_mean_sd([x[m] for x in T])[0],
                      diff_mean=mu, diff_sd=sd, dz=z,
                      mde_at_n=(_Z80 * sd / math.sqrt(n)) if n else float("inf"),
                      n_for_80pct_power=_n_for_power(z))
    z0 = abs(out["correct"]["dz"])
    # is the ACCURACY effect itself resolvable at this n? |d_z|*sqrt(n) is the paired t statistic.
    tstat = z0 * math.sqrt(n)
    resolvable = tstat >= 1.959963984540054
    for m in RESOLUTION_METRICS:
        # None -- not a number -- when the accuracy arm's own effect is not significant. Suppressing
        # it only in the markdown would leave a quotable figure in the sidecar JSON, and a grep over
        # ARC_*.resolution.json is exactly how someone would come to quote "1.0x" from two noise
        # terms. `dz` and `n_for_80pct_power` stay: they are honest inputs to the NEXT run's power
        # calculation, and neither can be mistaken for a measured gain.
        out[m]["items_saved_vs_accuracy"] = (((abs(out[m]["dz"]) / z0) ** 2)
                                             if (z0 > 0 and resolvable) else None)
    ci = {}
    if boot and resolvable:
        rng = random.Random(seed)
        cols = {m: diffs[m] for m in RESOLUTION_METRICS}
        draws = {m: [] for m in RESOLUTION_METRICS}
        for _ in range(int(boot)):
            idx = [rng.randrange(n) for _ in range(n)]
            zs = {m: abs(_dz([cols[m][i] for i in idx])[2]) for m in RESOLUTION_METRICS}
            if zs["correct"] <= 0:
                continue
            for m in RESOLUTION_METRICS:
                draws[m].append((zs[m] / zs["correct"]) ** 2)
        for m in RESOLUTION_METRICS:
            s = sorted(draws[m])
            ci[m] = [s[int(0.025 * len(s))], s[int(0.975 * len(s))]] if s else None
    return dict(key=key, n=n, boot=int(boot) if resolvable else 0,
                accuracy_effect_resolvable=bool(resolvable), accuracy_t=tstat,
                exact_ties=ties, metrics=out, items_saved_ci95=ci,
                per_item=[dict(id=pb[i].get("id"), qi=pb[i].get("qi"),
                               base={m: B[i][m] for m in RESOLUTION_METRICS + ("tie",)},
                               trained={m: T[i][m] for m in RESOLUTION_METRICS + ("tie",)},
                               diff={m: diffs[m][i] for m in RESOLUTION_METRICS})
                          for i in range(n)])


def _resolution_lines(res, label):
    """The markdown block for one scoring key. Kept separate from stage_report so the numbers and
    the prose that frames them cannot drift apart."""
    L = []
    A = L.append
    M = res["metrics"]
    A("")
    A("### %s (n = %d)" % (label, res["n"]))
    A("")
    A("| metric | BASE mean | trained mean | paired diff | sd | d_z | MDE @ n=%d | items for 80%% power |"
      % res["n"])
    A("|---|---|---|---|---|---|---|---|")
    for m in RESOLUTION_METRICS:
        x = M[m]
        A("| %s | %.4f | %.4f | %+.4f | %.4f | %+.4f | %.4f | %s |"
          % (m, x["base_mean"], x["trained_mean"], x["diff_mean"], x["diff_sd"], x["dz"],
             x["mde_at_n"],
             "n/a" if x["n_for_80pct_power"] == float("inf") else "%.0f" % x["n_for_80pct_power"]))
    A("")
    t = res["exact_ties"]
    if t["base"] or t["trained"]:
        A("- **Exact ties: %d of %d items (BASE), %d (trained).** The gold option's score equals "
          "the best other option's BIT-FOR-BIT, so accuracy on those items is decided by the "
          "choice-order tie-break, not by the model. `margin_gold` reports 0.0 there -- no evidence "
          "either way -- which is the honest value accuracy cannot express. Single-token answers "
          "scored in bf16 collide outright; this is a resolution defect, not a rounding artifact."
          % (int(t["base"]), res["n"], int(t["trained"])))
        A("")
    if not res["accuracy_effect_resolvable"]:
        A("**Sensitivity gain: UNRESOLVED at this n.** The accuracy arm's own paired effect is not "
          "significant (|t| = %.2f < 1.96), so `items_saved` would be a quotient of two noise terms "
          "-- its bootstrap CI spans orders of magnitude and it must not be quoted. The d_z column "
          "above is still the honest input to a power calculation for the NEXT run."
          % res["accuracy_t"])
        return L
    A("Sensitivity gain, MEASURED on these items (`items_saved = (d_z_metric / d_z_correct)^2`, "
      "i.e. how many fewer items the metric needs to detect THIS perturbation at 80%% power; "
      "95%% CI from a paired bootstrap over items, %d resamples):" % res["boot"])
    A("")
    A("| metric | items saved vs accuracy | 95% CI |")
    A("|---|---|---|")
    for m in RESOLUTION_METRICS:
        if m == "correct":
            continue
        r = M[m]["items_saved_vs_accuracy"]
        c = res["items_saved_ci95"].get(m)
        A("| %s | %s | %s |" % (m, "n/a" if r is None else "%.2fx" % r,
                                "n/a" if not c else "%.2fx - %.2fx" % (c[0], c[1])))
    return L


def stage_report(opts):
    with open(opts.out, encoding="utf-8") as fh:
        r = json.load(fh)
    spec = _result_spec(r)
    camp = _result_name(spec)
    n_deltas = len(spec["accepted_events"])
    n_slots = len(spec["slots"])
    moved = list(r["coords_moved"])
    layers_moved = sorted({k.split(":")[0] for k in moved})
    n = r["n"]
    pb, pt = r["per_item_base"], r["per_item_trained"]
    md_path = os.path.splitext(opts.out)[0] + ".md"
    # The per-item margin/Brier go in their own file rather than back into --out: the result JSON is
    # the BENCH's artifact and several of them are published, so a report run must never rewrite it.
    reso_path = os.path.splitext(opts.out)[0] + ".resolution.json"
    rows = []
    for metric, key in (("acc_norm (length-normalized, PRIMARY)", "ok_mean"), ("acc (raw sum)", "ok_sum")):
        kb = sum(1 for p in pb if p[key])
        kt = sum(1 for p in pt if p[key])
        b = sum(1 for x, y in zip(pb, pt) if x[key] and not y[key])
        c = sum(1 for x, y in zip(pb, pt) if y[key] and not x[key])
        rows.append(dict(metric=metric, base=kb / n, trained=kt / n, delta=(kt - kb) / n,
                         base_ci=_wilson(kb, n), trained_ci=_wilson(kt, n),
                         b=b, c=c, p=_mcnemar_p(b, c)))
    prim = rows[0]
    sig = prim["p"] < 0.05
    floor = r["random_floor"]
    above = prim["base"] > floor + (prim["base_ci"][1] - prim["base"])
    verdict = ("The %d accepted deltas from %s %s ARC-Easy accuracy: %+.2f pp "
               "(%.2f%% -> %.2f%%), McNemar p=%.3g over %d discordant items -- %s."
               % (n_deltas, camp, "CHANGE" if sig else "do NOT measurably change",
                  100 * prim["delta"],
                  100 * prim["base"], 100 * prim["trained"], prim["p"], prim["b"] + prim["c"],
                  "statistically significant at p<0.05" if sig
                  else "indistinguishable from noise"))

    L = []
    A = L.append
    A("# ARC-Easy on the full 47-layer GLM-4.7-Flash: BASE vs BASE + %d accepted deltas from %s"
      % (n_deltas, camp))
    A("")
    A("Generated %s by `tools/glm_arc_eval.py` for %s%s."
      % (r["generated_utc"], camp,
         "" if spec.get("frozen_cut_event") is None
         else ", frozen at event %s" % spec["frozen_cut_event"]))
    if spec.get("campaign") == RUN5_SPEC["campaign"]:
        A("")
        A("This was the project's first real-benchmark capability measurement of the GLM "
          "shardDiLoCo training; every previous training claim was held-out cross-entropy only.")
    A("")
    A("## Verdict")
    A("")
    A(verdict)
    A("")
    A("Base ARC-Easy accuracy is **%.2f%%** against a **%.2f%%** random-guess floor, so the full "
      "model is genuinely solving the task rather than guessing." % (100 * prim["base"], 100 * floor))
    A("")
    A("## Results")
    A("")
    A("N = %d of %d ARC-Easy test items (seeded subset, seed=%d); identical subset, order and "
      "scoring code for both arms. Random-guess floor %.4f. Scoring: answer log-likelihood, no "
      "generation." % (n, r["n_total"], r["seed"], floor))
    A("")
    A("| metric | BASE | BASE + deltas | delta | McNemar p | discordant (b/c) |")
    A("|---|---|---|---|---|---|")
    for x in rows:
        A("| %s | %.4f (95%% CI %.4f-%.4f) | %.4f (95%% CI %.4f-%.4f) | %+.4f | %.3g | %d/%d |"
          % (x["metric"], x["base"], x["base_ci"][0], x["base_ci"][1], x["trained"],
             x["trained_ci"][0], x["trained_ci"][1], x["delta"], x["p"], x["b"], x["c"]))
    A("")
    A("- Per-item agreement between arms: **%.4f** (acc_norm predictions), %.4f (acc predictions)."
      % (r["agreement_acc_norm"], r["agreement_acc"]))
    oc = r["output_change"]
    A("- Output-change check: %d of %d sequence scores changed after patching (mean |d| %.4g, "
      "max %.4g) -- the deltas demonstrably reach the model's output, so a null accuracy result "
      "is a real null, not a no-op patch."
      % (oc["seqs_changed"], oc["seqs_total"], oc["mean_abs"], oc["max_abs"]))
    A("")

    # ---- RESOLUTION: margin + Brier, computed from the per-option scores ALREADY in this file ----
    # Placed AFTER the accuracy block is closed, so it cannot split that bullet list, and computed
    # from `pb`/`pt` only -- the accuracy table above is already built and is never revisited. If
    # the result file predates the per-option scores being stored, the whole section is omitted
    # rather than filled with zeros.
    reso = {}
    for key, label in (("scores_mean", "acc_norm scores (length-normalized, PRIMARY)"),
                       ("scores_sum", "acc scores (raw sum)")):
        got = paired_resolution(pb, pt, key, boot=int(getattr(opts, "resolution_boot", 1000) or 0))
        if got is not None:
            reso[key] = (label, got)
    if reso:
        A("## Resolution: margin and Brier (continuous metrics over the SAME scores)")
        A("")
        A("Accuracy is a step function of the per-option log-likelihoods; the three metrics below "
          "are continuous functions of the IDENTICAL numbers, recomputed here from the stored "
          "scores -- no model was run and nothing above this heading was touched.")
        A("")
        A("- `margin` -- the chosen option's score minus the runner-up's. Always >= 0. This is "
          "CONFIDENCE, and it is deliberately not the sensitivity lever: it is blind to WHICH "
          "option won, so an arm that becomes confidently wrong scores like one that becomes "
          "confidently right. Measured below, it is usually WORSE than accuracy.")
        A("- `margin_gold` -- the GOLD option's score minus the best non-gold option. Signed, and "
          "it is the continuous relaxation of the accuracy indicator: correct items have "
          "margin_gold >= 0 and wrong items have margin_gold <= 0, with equality exactly on a tie. "
          "This is the metric that buys resolution.")
        A("- `brier` -- multi-class Brier score over the options renormalized to probabilities. A "
          "proper scoring rule, bounded [0, 2], LOWER is better, so its sign convention is opposite "
          "to the other two.")
        A("")
        A("`d_z` is the paired standardized effect (mean/sd of the per-item difference); required N "
          "scales as 1/d_z^2, which is what makes the last two columns comparable across metrics "
          "whose units are not.")
        for key in ("scores_mean", "scores_sum"):
            if key in reso:
                L.extend(_resolution_lines(reso[key][1], reso[key][0]))
        A("")
        A("Per-item margin and Brier for every item and both arms: `%s`."
          % os.path.basename(reso_path))
        A("")
    A("## What was actually compared")
    A("")
    A("- **Model**: the FULL 47-layer GLM-4.7-Flash (46 MoE layers, 64 experts each), bf16, "
      "**0 dead layers**. This matters: the training-time network keeps only ONE MoE layer "
      "resident, so 45 of 46 MoE layers contribute exactly zero there. Benchmarking that crippled "
      "network would not measure GLM's capability.")
    A("- **Arm 2** = the same model with the folded expert rows of %s patched in. Its ordered slot "
      "registry holds **%d coordinate(s)**; %d of them actually moved, across %d of 46 MoE "
      "layer(s): %s."
      % (camp, n_slots, len(moved), len(layers_moved), ", ".join(moved) or "none"))
    A("")
    A("| coordinate | delta L2 | base L2 | relative |")
    A("|---|---|---|---|")
    for k, v in sorted(r["coord_stats"].items(), key=lambda kv: -kv[1]["delta_l2"]):
        A("| L%s E%s | %.4g | %.4g | %.3g |" % (tuple(k.split(":")) + (v["delta_l2"], v["base_l2"],
                                                                       v["rel"])))
    A("")
    A("## Controls that had to pass first")
    A("")
    A("1. **Fold gate** -- the %d accepted records were folded through the contributor's own "
      "`_fold_accepted_checked`, and the post-fold `model_root` had to equal the final root this "
      "campaign's coordinator advertised at its last accepted event, bit-for-bit: `%s` (folded %s "
      "records)." % (n_deltas, r["fold_root"], r["folded_records"]))
    A("2. **Streaming equivalence gate** (`--stage selftest`) -- a real 2-layer model was scored "
      "with stock expert residency, then re-scored with the streaming mechanism. Logits came back "
      "**bit-identical (max|d| = 0)**. This caught a real bug first: a separately constructed "
      "experts module silently fell back to the naive bf16 `index_add_` loop instead of the "
      "dispatched `grouped_mm` fp32 kernel, shifting logits by up to 0.25.")
    A("3. **Cross-path gate** -- the packed layer-1 expert rows had to be BIT-EXACT against the "
      "rows read from the independently built `piece_loader` model (8/8 matched). Ground truth "
      "from a different loader, not from the code under test.")
    A("4. **Patch read-back** -- the applied weights were re-read and compared to the intended "
      "ones (0 mismatches).")
    A("")
    A("## Decoded samples (ground truth read as text, not as a metric)")
    A("")
    for i, s in enumerate(r["samples"]):
        A("**%d. %s**  " % (i + 1, s["id"]))
        A("Q: %s  " % s["question"])
        A("Gold: *%s*  " % s["gold"])
        A("BASE chose: *%s* (%s)  " % (s["base_choice"], "correct" if s["base_correct"] else "wrong"))
        A("BASE+deltas chose: *%s* (%s)"
          % (s["trained_choice"], "correct" if s["trained_correct"] else "wrong"))
        A("")
    m = r["model"]
    A("## Cost and how it ran")
    A("")
    A("The full bf16 model is ~56.8 GiB and this box has 61.56 GiB RAM + one 32 GiB card, so it "
      "fits in neither. Pure-CPU residency was refused (it would leave ~5 GiB for Windows and "
      "swap), and accelerate `device_map=\"auto\"` + `offload_folder` **segfaulted** at 64%% of the "
      "load the instant it began writing disk-offload files. The model therefore runs with the "
      "trunk resident, %d MoE layers pinned, and %d streamed from pre-packed per-layer safetensors "
      "into GPU memory per forward pass."
      % (len(m["pinned_layers"]), m["moe_layers"] - len(m["pinned_layers"])))
    A("")
    A("- VRAM cap %.1f of %.1f GiB (>= 8 GiB always left for the desktop); measured peak %.2f GiB."
      % (m["vram_cap_gib"], m["vram_total_gib"], m["gpu_peak_gib"]))
    A("- %.3f s per item per arm; %.0f GiB of expert weights streamed in total."
      % (r["s_per_item_per_arm"], r["gib_streamed_total"]))
    A("- A full %d-item run of both arms would cost about **%.2f h** at this rate."
      % (r["n_total"], r["full_2376_estimate_h"]))
    A("")
    A("## Honest limits")
    A("")
    A("- This campaign changed **%d of the model's 2944 routed-expert coordinates** (%d expert(s) "
      "across %d of 46 MoE layers, ~%.2g%% of routed capacity), yet %d of %d sequence scores moved. "
      "A perturbation in an early layer propagates through every downstream layer, so a tiny "
      "changed-parameter count does NOT bound the behavioural effect -- as the measured %+.2f pp "
      "shows."
      % (len(moved), len(moved), len(layers_moved), 100.0 * len(moved) / 2944.0,
         oc["seqs_changed"], oc["seqs_total"], 100 * prim["delta"]))
    _worst = max(r["coord_stats"].items(), key=lambda kv: kv[1]["rel"])
    A("- The deltas are not gentle nudges. For L%s E%s the update's L2 norm is **%.2fx the base "
      "expert's own norm**, i.e. that expert was effectively overwritten rather than refined."
      % (tuple(_worst[0].split(":")) + (_worst[1]["rel"],)))
    A("- ARC-Easy accuracy and the campaign's held-out CE are different yardsticks measured on "
      "different data, so this does not contradict whatever CE gain %s recorded -- it measures "
      "something the CE gate never looked at. Both can be true at once, which is precisely this "
      "project's `pouw-verified-not-useful` lesson." % camp)
    A("- The deltas were trained against a network where 45 of 46 MoE layers were zeroed, then "
      "evaluated inside the full network. That distribution shift is exactly what the layer-"
      "composition question is about and is not controlled for here.")
    A("- N = %d of %d items; the 95%% CIs above are the sampling uncertainty." % (n, r["n_total"]))
    A("")
    A("Raw per-item results, subset indices and all gate values: `%s`." % os.path.basename(opts.out))
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    _log("[arc/report] wrote %s" % md_path)
    if reso:
        with open(reso_path, "w", encoding="utf-8") as fh:
            json.dump(dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           source=os.path.basename(opts.out), campaign=spec.get("campaign"),
                           n=n, keys={k: v[1] for k, v in reso.items()}), fh, indent=2)
        _log("[arc/report] wrote %s (per-item margin + Brier, both arms)" % reso_path)
        for key, (label, res) in reso.items():
            M = res["metrics"]
            _log("[arc/report] %-12s margin %+.4f | margin_gold %+.4f | brier %+.4f | "
                 "items saved vs accuracy: %s"
                 % (key, M["margin"]["diff_mean"], M["margin_gold"]["diff_mean"],
                    M["brier"]["diff_mean"],
                    "UNRESOLVED (accuracy |t|=%.2f)" % res["accuracy_t"]
                    if not res["accuracy_effect_resolvable"] else
                    "margin %.2fx, margin_gold %.2fx, brier %.2fx"
                    % (M["margin"]["items_saved_vs_accuracy"],
                       M["margin_gold"]["items_saved_vs_accuracy"],
                       M["brier"]["items_saved_vs_accuracy"])))
    _log("[arc/report] %s" % verdict)
    return 0


def stage_spec(opts):
    """Print the RESOLVED campaign spec and exit. Torch-free and network-free, so it is the cheap
    way to prove which campaign a command line is about to fold BEFORE spending an hour on it --
    and, with no --campaign-spec, to show the defaults are still run 5's published constants."""
    log_spec("arc/spec")
    same = is_run5_default()
    _log("[arc/spec] identical to the published run-5 constants: %s" % ("YES" if same else "NO"))
    _log(json.dumps(SPEC, indent=1, sort_keys=True))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="ARC-Easy on the full 47-layer GLM: base vs +run5 deltas")
    ap.add_argument("--stage", choices=("pack", "selftest", "fold", "bench", "report", "ce", "spec"),
                    required=True)
    ap.add_argument("--campaign-spec", dest="campaign_spec", default="",
                    help="JSON file describing WHICH campaign to fold and score: keys campaign, "
                         "accepted_events, slots (ordered registry), final_root, accepted_scope "
                         "(legacy|campaign), pieces, label, frozen_cut_event, lane_url, lane_token. "
                         "OMIT IT and every stage runs run 5 exactly as it always has. Pass the SAME "
                         "file to fold, ce and bench -- ce/bench check the deltas file against it. "
                         "`--stage spec` prints the resolved spec and exits.")
    ap.add_argument("--ce-mode", choices=("anchor-cpu", "anchor-cuda", "full", "prefix", "base"),
                    default="full",
                    help="--stage ce: which residency to score the gate metric on. 'base' is the "
                         "AUDIT path: the unpatched full-47 model, no --deltas required, one arm. "
                         "It exists because the published reference CE is a BASE number and every "
                         "other mode demands a campaign's fold snapshot to reach it. NOTE that "
                         "base mode's own DEFAULTS are (1024, 32, 32, pin 1..12) and reproduce "
                         "NOTHING published -- the reference needs all four knobs named: "
                         "`--stage ce --ce-mode base --ce-seqs 256 --ce-chunk 32 --ce-sub-batch 8 "
                         "--pin-layers 2`, which prints %r." % CE_REFERENCE_256_32_8_PIN2)
    ap.add_argument("--prefix-k", default="",
                    help="--ce-mode prefix: comma list of k = number of resident MoE layers "
                         "(prefix 1..k, 60/64 experts each). Each k is a full 2-arm point.")
    ap.add_argument("--prefix-loader", choices=("piece", "pack"), default="piece",
                    help="--ce-mode prefix: 'piece' = piece_loader residency (k limited by the VRAM "
                         "cap / RAM); 'pack' = layer pack + streaming (any k, needs --stage pack)")
    ap.add_argument("--prefix-ram-gib", type=float, default=18.0,
                    help="--prefix-loader pack: host RAM budget for cached expert slabs "
                         "(1.125 GiB each); layers past it are re-read from the pack every pass")
    ap.add_argument("--ce-device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="--ce-mode prefix: device per point (auto = cuda while trunk+k*1.125 GiB "
                         "fits the VRAM cap, else cpu)")
    ap.add_argument("--ce-chunk", type=int, default=32,
                    help="--stage ce: OUTER chunk -- sequences per `_chunked_heldout_ce` slice")
    ap.add_argument("--ce-sub-batch", dest="ce_sub_batch", type=int, default=0,
                    help="--stage ce: INNER sub-batch (NEURAHASH_GLM_EVAL_CHUNK, read at "
                         "sharddiloco_glm_expert.heldout_ce:398). 0 = mirror --ce-chunk, which is "
                         "byte-for-byte the historical behaviour. It is a separate knob because it "
                         "is the FORWARD BATCH SIZE and the expert GEMM is batch-composition-"
                         "sensitive here -- NOT because of floating-point re-association. Measured "
                         "2026-08-10 at (256 seqs, chunk 32, pin[1,2]): sub-batch 8 -> %r, "
                         "sub-batch 32 -> 4.813307792909684, a gap of 3.3 millinats. The shipped "
                         "default of 8 is what produced the published reference; forcing "
                         "inner == outer made it unreachable." % CE_REFERENCE_256_32_8_PIN2)
    ap.add_argument("--ce-seqs", dest="ce_seqs", type=int, default=0,
                    help="--stage ce: score only the first N sequences of the frozen held-out pool "
                         "(0 = all %d). The published reference is a 256-sequence PREFIX; without "
                         "this flag that configuration could not be expressed." % 1024)
    ap.add_argument("--ce-layer-major", dest="ce_layer_major",
                    action=argparse.BooleanOptionalAction, default=layer_major_default(),
                    help="--stage ce: score with the LAYER-MAJOR inversion "
                         "(heldout_ce_layer_major) instead of the shipped nested data loops. "
                         "DEFAULT %s. --no-ce-layer-major forces the shipped gate_ce path back "
                         "(so does %s=1, which kills both inversions at once) -- use it for a "
                         "bisect or whenever a paid number looks wrong. It is bit-identical "
                         "arithmetic -- gated "
                         % ("ON" if layer_major_default() else "OFF", _NO_LM_ENV) +
                         "on the real 47-layer model at 256 seqs / chunk 32 / pin 2, all 8 "
                         "per-chunk floats equal digit for digit, 1584.0 -> 49.5 GiB and "
                         "885.4 -> 30.3 s (scratchpad/CE_LAYER_MAJOR_GATE_2026-08-10.json). It "
                         "only pays where the experts STREAM (--ce-mode full / prefix-loader "
                         "pack); with no streamer it is the same reads and the same number.")
    ap.add_argument("--ce-lm-group", dest="ce_lm_group", type=int, default=0,
                    help="--ce-layer-major: forward units held in flight per layer sweep "
                         "(0 = all, the fewest reads). Purely an IO/activation-memory knob -- "
                         "results are keyed by unit index and reduced after every group finishes, "
                         "so the group size cannot reach the arithmetic. Bytes read = "
                         "ceil(n_units/group) x streamed_layers x 1.125 GiB.")
    ap.add_argument("--dose", default="1.0",
                    help="--stage ce --ce-mode full: comma list of alphas in [0,1] for the dose "
                         "line search w(alpha) = pristine + alpha*(folded - pristine), applied at "
                         "PATCH time (the fold gate is untouched and still reproduces the recorded "
                         "root at full dose). Each alpha is one extra scored arm against the SAME "
                         "base arm. 1.0 (the default) is the ordinary full-dose measurement; 0.0 "
                         "is a control that must reproduce the base arm to %.0e." % DOSE_ZERO_TOL)
    ap.add_argument("--max-records", dest="max_records", type=int, default=0,
                    help="fold/ce: score the PREFIX of the campaign -- only the first N accepted "
                         "records in event order (0 = all of them, the historical behaviour). This "
                         "is how CE is measured as a function of HOW MANY accepted deltas are "
                         "folded. A prefix has no recorded target root (the coordinator's root at "
                         "event N was hashed over a smaller registry), so fold gates on 'all N "
                         "folded and the root moved' and ce gates on the deltas file declaring the "
                         "same N. Pass the SAME --max-records to fold and to ce.")
    ap.add_argument("--ce-out", default=os.path.join(_REPO, "docs", "research",
                                                     "FULLMODEL_CE_RUN5_2026-07-27.json"))
    ap.add_argument("--deltas", default="", help="run-5 delta snapshot (.pt); fold writes, bench reads")
    ap.add_argument("--out", default=os.path.join(_REPO, "docs", "research",
                                                  "ARC_EASY_RUN5_2026-07-27.json"))
    ap.add_argument("--resolution-boot", dest="resolution_boot", type=int, default=1000,
                    help="--stage report: paired-bootstrap resamples behind the 95%% CI on the "
                         "MEASURED items-saved ratio of margin/Brier vs accuracy (0 = point "
                         "estimate only). The bootstrap resamples ITEMS once per draw and scores "
                         "every metric on that same resample, because the ratio's two d_z values "
                         "are computed on the same items and are strongly dependent.")
    ap.add_argument("--pack-dir", default="D:/_glm_layer_experts")
    ap.add_argument("--pin-layers", type=int, default=12,
                    help="MoE layers 1..K kept resident on the GPU (layer 1 is always pinned)")
    ap.add_argument("--n", type=int, default=0, help="subset size; 0 = auto-size to --budget-min")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sub", type=int, default=4, help="lm_head chunk rows (VRAM knob)")
    ap.add_argument("--warmup-items", type=int, default=10)
    ap.add_argument("--arc-layer-major", dest="arc_layer_major",
                    action=argparse.BooleanOptionalAction, default=layer_major_default(),
                    help="--stage bench: score both ARM 1 / ARM 2 (and the calibration warm-up) "
                         "with the LAYER-MAJOR inversion (score_batches_layer_major) instead of "
                         "the shipped per-batch score_batch loop. DEFAULT %s. "
                         "--no-arc-layer-major forces the shipped score_batch loop back (so does "
                         "%s=1, which kills both inversions at once) -- use it for a bisect or "
                         "whenever a published accuracy looks wrong. "
                         % ("ON" if layer_major_default() else "OFF", _NO_LM_ENV) +
                         "It is the same "
                         "batches in the same order with only the (layer, batch) visiting order "
                         "inverted -- gated bit-exact on the real 47-layer model over 9501 rows, "
                         "0 mismatches, max |diff| 0.000e+00, 7375.5 -> 49.5 GiB and 2430.4 -> "
                         "51.1 s. It only pays where the experts STREAM; with everything pinned it "
                         "is the same reads and the same number.")
    ap.add_argument("--arc-lm-group", dest="arc_lm_group", type=int, default=0,
                    help="--arc-layer-major: BATCHES held in flight per layer sweep (0 = all, the "
                         "fewest reads). Purely an IO/activation-memory knob -- results are keyed "
                         "by batch index and every batch's tail runs after its sweep, so the group "
                         "size cannot reach the arithmetic. Bytes read = ceil(n_batches/group) x "
                         "streamed_layers x 1.125 GiB. Retained per in-flight batch: the hidden "
                         "state (B*T*hidden*2 B) AND the causal mask, which grows QUADRATICALLY "
                         "in T -- lower this first if the sweep runs out of VRAM.")
    ap.add_argument("--budget-min", type=float, default=180.0,
                    help="wall-clock budget in MINUTES for both scoring arms combined")
    ap.add_argument("--min-n", type=int, default=150)
    opts = ap.parse_args(argv)
    apply_campaign_spec(opts.campaign_spec)
    # --out / --ce-out still DEFAULT to run 5's published files, which is correct for run 5 and a
    # trap for anything else: before --campaign-spec existed those paths could only ever receive
    # run-5 numbers, and now a forgotten flag would overwrite a published result with a different
    # campaign's. Refuse instead of clobbering. Nothing changes when the spec is the default.
    #
    # The gate is the WHOLE spec, not the campaign id (is_run5_default). An id-only test passes a
    # spec that keeps run 5's campaign but changes accepted_events / final_root / slots -- i.e. a
    # LATER CUT of the same campaign, the exact case `frozen_cut_event` exists to record -- and that
    # is a different measurement, so it must not land in the published run-5 files either.
    #
    # `report` is in the map because it WRITES too: splitext(--out)[0] + ".md" beside the JSON.
    # Leaving it out meant a foreign-campaign report could overwrite the published run-5 markdown
    # with another campaign's prose while the JSON next to it stayed run 5's.
    if not is_run5_default():
        writes = {"bench": [("--out", opts.out)], "ce": [("--ce-out", opts.ce_out)],
                  "report": [("--out", opts.out)]}
        for flag, path in writes.get(opts.stage, []):
            if "RUN5" in os.path.basename(path).upper():
                ap.error("%s still points at run 5's published file (%s) but --campaign-spec "
                         "selected %s with %d accepted event(s). Pass an explicit %s for it."
                         % (flag, os.path.basename(path), spec_name(),
                            len(SPEC["accepted_events"]), flag))
    # --ce-mode base scores the UNPATCHED model, so a deltas snapshot is not merely optional there,
    # it is meaningless -- demanding one is what made the published BASE reference unreproducible.
    if opts.stage in ("fold", "bench", "ce") and not opts.deltas \
            and not (opts.stage == "ce" and opts.ce_mode == "base"):
        ap.error("--deltas is required for --stage %s" % opts.stage)
    if opts.stage == "ce" and opts.ce_mode == "base" and opts.deltas:
        ap.error("--ce-mode base scores the UNPATCHED model and folds nothing; passing --deltas "
                 "would imply those deltas were applied. Drop --deltas, or pick --ce-mode full.")
    return {"pack": stage_pack, "selftest": stage_selftest, "fold": stage_fold,
            "bench": stage_bench, "report": stage_report, "ce": stage_ce,
            "spec": stage_spec}[opts.stage](opts)


if __name__ == "__main__":
    raise SystemExit(main())
