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

Usage (JOIN the public lane -- what a STRANGER runs; no key, no signup, and no --url/--token:
those two default to the public content lane, see add_common_args):
  1. fetch the GLM base ONCE (tokenless, per-piece sha256-verified; --config-cid /
     NEURAHASH_GLM_CONFIG_CID is required or the loader fails after the download):
       C:/Python313/python.exe tools/fetch_glm_base.py --dest D:/hf_models/glm_base --pieces 0 \
           --config-cid <published config sha256>
  2. mine. --mode glm and --device cuda are REQUIRED for real training: the tiny/cpu defaults are
     the hermetic-test path, NOT mining. An EMPTY --data-dir self-fills from the advertised corpus:
       C:/Python313/python.exe tools/sharddiloco_glm_contributor.py --mode glm --device cuda \
           --shard-dir D:/hf_models/glm_base --config-dir D:/hf_models/glm_base/config \
           --data-dir D:/glm_wan/miner --domains daily
Usage (tiny shakedown against a LOCAL store, plan step S3 -- loopback joins NOTHING public):
  C:/Python313/python.exe tools/sharddiloco_glm_contributor.py --miner miner0 --slot 0 \
      --key <hex16> --url http://127.0.0.1:8797 --token <tok> --mode tiny --slots 1:0,1:1
Usage (real GLM, plan step S4). Pass NO piece flag and residency now fills the whole resident layer
by itself: 60 claimable layer-1 coordinates at a BYTE-IDENTICAL parameter count to the old --piece 0
five (MEASURED 2026-07-26; a resident layer's fused params are allocated full-width either way --
docs/SHARD_CLAIM_DESIGN.md C12). --piece 0 still pins residency to 5 coordinates, which is what
stalled run 4 when it was the default:
  ... --mode glm --shard-dir D:/hf_models/GLM-4.7-Flash-bf16_shards_100mb \
      --config-dir D:/hf_models/GLM-4.7-Flash-bf16 --slots 1:0,1:1 \
      --data-dir D:/glm_wan --domains code,gutenberg --device cuda --batch 4
"""
import argparse
import copy
import hashlib
import json
import os
import re
import secrets
import sys
import threading
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
CONTRIB_PREFIX_FMT = "cg/r%d/"                     # LEGACY (pre-campaign) shape; see CAMPAIGN SCOPING
CONTRIB_CAMPAIGN_PREFIX_FMT = "cg/%s/r%d/"         # campaign-scoped shape: cg/<campaign_id>/r<base_event>/
ACCEPTED_NAME_FMT = "sharddiloco/glm/accepted/r%d"

# ---- corpus-over-WAN auto sync (W6) --------------------------------------------------------------
# The coordinator advertises DATA_RECORD_NAME (same trust surface + name shape as GLM_POINTER_NAME);
# a contributor fetches ONLY the miner-facing splits it matches -- ids_<domain>_train/val.npy or the
# data manifest. The SECRET probe/heldout splits (sharddiloco_glm_coordinator.py:573) must never
# match, so even a forged record cannot make miner code pull them (F1 defense-in-depth).
DATA_RECORD_NAME = "sharddiloco/glm/data"
DATA_MANIFEST_NAME = "data_manifest.json"
RC_DATA_UNVERIFIED = 9              # exit code: a record file was neither locally-valid nor fetched+verified
RC_DOMAINS_MISMATCH = 10            # exit code: our --domains list disagrees with the coordinator's (C6)
RC_NO_CAMPAIGN = 11                 # exit code: the pointer advertises no campaign_id (see campaign_refusal)
_ALLOWED_DATA_RE = re.compile(r"ids_[A-Za-z0-9-]+_(?:train|val)\.npy\Z")


def use_glm_lane_names():
    """Point sharddiloco_harness at the GLM lane's object names. Runtime assignment ONLY."""
    H.POINTER_NAME = GLM_POINTER_NAME


# ===================================================================== CAMPAIGN SCOPING (2026-07-25)
# WHY (measured, scratchpad/FINDING_cross_campaign_replay.md). The lane store NEVER deletes and record
# names carried no run identity ("cg/r<base_event>/<miner>"), so ONE flat namespace held 11,229 objects
# from every campaign that ever ran. Two consequences, both measured on the live lane:
#   1. a fresh coordinator's discovery scanned all 11,229 names and revealed one more dead r-number per
#      event forever (run 4: 198 records "processed", 189 of them UNVERIFIED old ones);
#   2. worse -- every campaign starts from the SAME pristine base, so at genesis a fresh campaign's
#      base_root/base_slot_root EQUAL a dead campaign's early roots. The lineage guard therefore could
#      not tell them apart and a dead run's delta was lineage-VALID: identity glm-ea20C873, which
#      belongs to no live miner, was MINTED into runs 2, 3 and 4.
# THE FIX, in two halves that must ship together:
#   * the NAMESPACE half (campaign_prefix / contrib_name / contrib_prefix): records live under
#     cg/<campaign_id>/r<N>/..., so discovery filters to this campaign and the 11,229-name scan
#     collapses. This alone is only an OPTIMISATION -- a replayer can rename into our prefix.
#   * the LINEAGE half (_campaign_seed_into, fed into model_root/slot_root): the campaign id SEEDS the
#     root digest, so two campaigns over an identical pristine base share NO lineage root at ANY
#     height. This is what actually stops the merge. The per-coordinate lineage design is untouched --
#     only the digest's seed changes, and an unset campaign reproduces the old digest byte-for-byte.
CAMPAIGN_POINTER_KEY = "campaign_id"               # the ADDITIVE v2-pointer field the coordinator stamps
_CAMPAIGN_RE = re.compile(r"\A[0-9a-f]{8,64}\Z")   # lowercase hex only: never "/" (name-safe) and never
                                                   # "r<digits>" (so cg/<id>/ can't be read as cg/r<N>/)


def new_campaign_id():
    """Mint a fresh campaign nonce: 16 lowercase hex chars (64 bits from `secrets`). Minted ONCE per
    campaign by the coordinator and then PERSISTED -- a restart that re-minted would orphan the
    campaign's own records, which is a worse bug than the cross-campaign replay this closes."""
    return secrets.token_hex(8)


def normalize_campaign_id(cid):
    """The canonical form of `cid`, or None when it is absent/unusable. Fail-closed on purpose: the
    pointer is UNSIGNED on a shared-token lane, so an id we cannot validate is treated as no id at all
    rather than pasted into object names or hash seeds."""
    if cid is None:
        return None
    s = str(cid).strip().lower()
    return s if _CAMPAIGN_RE.match(s) else None


def campaign_scope_on(environ=None):
    """NEURAHASH_SD_CAMPAIGN_SCOPE -- the documented opt-out for a LEGACY lane (same truthiness house
    style as dm.shard_diloco_on). DEFAULT ON. Set 0/false/off/no to join a lane that predates campaign
    scoping. What it does per role, precisely:
      * COORDINATOR: mints nothing, seeds nothing, advertises nothing -> v3.4.1 exactly (flat `cg/rN/`
        names, unseeded roots).
      * MINER: suppresses the fail-closed REFUSAL only (campaign_refusal). If the coordinator advertises
        a campaign anyway, the miner still adopts it -- publishing unscoped into a scoped lane would
        make every delta wrong-lineage-slot-root, and that is not a behaviour worth reproducing."""
    env = os.environ if environ is None else environ
    return (env.get("NEURAHASH_SD_CAMPAIGN_SCOPE", "1") or "1").strip().lower() \
        not in ("0", "false", "off", "no")


def bind_campaign_id(host, cid):
    """Attach this process's campaign scope to the lane `host` and return the normalized id (None =
    unscoped/legacy). The scope rides the HOST, not a module global, because every root function already
    takes the host -- so both roles hash identically, no call site has to remember an extra argument,
    and two hosts in one process (a test's coordinator + miner) cannot leak into each other."""
    val = normalize_campaign_id(cid)
    host.campaign_id = val
    return val


def host_campaign_id(host):
    """The campaign scope bound to `host`, or None. Tolerates any host object (test fakes included)."""
    return normalize_campaign_id(getattr(host, "campaign_id", None))


def campaign_prefix(campaign=None):
    """"cg/<campaign_id>/" -- the namespace every contribution record of one campaign lives under
    (legacy "cg/" when unscoped). Pure."""
    cid = normalize_campaign_id(campaign)
    return "cg/" if cid is None else "cg/%s/" % cid


def contrib_name(rnd, miner, campaign=None):
    return contrib_prefix(rnd, campaign) + str(miner)


def contrib_prefix(rnd, campaign=None):
    cid = normalize_campaign_id(campaign)
    if cid is None:
        return CONTRIB_PREFIX_FMT % int(rnd)                      # legacy lane, byte-identical
    return CONTRIB_CAMPAIGN_PREFIX_FMT % (cid, int(rnd))


def accepted_name(rnd):
    return ACCEPTED_NAME_FMT % int(rnd)


def pointer_campaign_id(ptr):
    """The campaign id a RAW v2 pointer advertises, or None. Read off the raw dict (like
    `domains_digest`) because dm.sd_pointer_decode normalizes to a fixed key set and drops additive
    fields. Pure."""
    if not isinstance(ptr, dict):
        return None
    return normalize_campaign_id(ptr.get(CAMPAIGN_POINTER_KEY))


def campaign_refusal(ptr, environ=None):
    """FAIL-CLOSED miner gate: may we publish against this pointer? None = yes, else the ONE log line
    that names the reason and the way out.

    A pointer with no campaign id is a coordinator that predates campaign scoping. Publishing anyway
    would put our records back into the shared `cg/rN/` namespace -- the exact surface that let a dead
    campaign's delta be minted into three live runs -- so the default is to REFUSE and say why, instead
    of silently joining a lane where our own work and a dead run's are indistinguishable. Pure."""
    if not campaign_scope_on(environ):
        return None                                               # operator opted into the legacy lane
    if pointer_campaign_id(ptr) is not None:
        return None
    raw = (ptr or {}).get(CAMPAIGN_POINTER_KEY) if isinstance(ptr, dict) else None
    return ("no CAMPAIGN ID on the coordinator's pointer (campaign_id=%r). This coordinator predates "
            "campaign scoping, so every contribution would be published into the SHARED cg/rN/ "
            "namespace where a dead campaign's records are indistinguishable from ours -- MEASURED: a "
            "foreign identity's deltas were minted into three separate runs from that namespace. "
            "REFUSING to publish. Fix: upgrade the coordinator (it then stamps campaign_id on the "
            "pointer), or, to join a legacy lane deliberately, re-run with "
            "NEURAHASH_SD_CAMPAIGN_SCOPE=0." % (raw,))


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


def parse_coord(s):
    """'1:3' -> (1, 3). The shard-claim address: a GLM (layer, expert) COORDINATE."""
    txt = str(s).strip()
    L, sep, E = txt.partition(":")
    if not sep:
        raise SystemExit("[glm-node] --expert must look like L:E (e.g. 1:3), got %r" % txt)
    try:
        return int(L), int(E)
    except ValueError:
        raise SystemExit("[glm-node] --expert must look like L:E with integers, got %r" % txt)


def parse_pieces(spec):
    """'0-12' / '0,1,2' / '3' -> a SORTED, DEDUPLICATED list of expert-piece ids. Pure.

    MAPPING, stated because getting it wrong is silent: a piece id indexes the manifest's
    `experts_<id>` NAME, not its position in the pieces list (piece_loader._piece_record looks up
    "experts_%d"). The list also carries the trunk at position 0, so name-vs-position is off by one
    -- and an off-by-one here does not raise, it just leaves a layer partially resident (measured on
    tools/glm_router_domain_probe.py: 59/64 instead of 64/64). node_piece_ids' residency assertion is
    the second line of defence; this docstring is the first.

    Ranges are INCLUSIVE on both ends ('0-12' is 13 pieces). Every malformed shape raises SystemExit
    rather than degrading to piece 0: silently selecting the default would reproduce exactly the
    five-coordinate ceiling this flag exists to remove."""
    txt = str(spec).strip()
    out = set()
    for part in txt.split(","):
        part = part.strip()
        if not part:
            raise SystemExit("[glm-node] --pieces has an empty entry in %r; use e.g. 0-12 or 0,1,2"
                             % txt)
        if "-" in part:
            a, sep, b = part.partition("-")
            try:
                lo, hi = int(a.strip()), int(b.strip())
            except ValueError:
                raise SystemExit("[glm-node] --pieces range %r is not two integers (want e.g. 0-12)"
                                 % part)
            if lo > hi:
                raise SystemExit("[glm-node] --pieces range %r is descending; ranges are inclusive "
                                 "and ascending (want e.g. %d-%d)" % (part, hi, lo))
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(part))
            except ValueError:
                raise SystemExit("[glm-node] --pieces entry %r is not an integer (want e.g. 0-12 "
                                 "or 0,1,2)" % part)
    if not out:
        raise SystemExit("[glm-node] --pieces must be a non-empty list like 0-12 or 0,1,2")
    if min(out) < 0:
        raise SystemExit("[glm-node] --pieces contains a negative piece id (%d) in %r"
                         % (min(out), txt))
    return sorted(out)


# ------------------------------------------------------------ DEFAULT residency: FILL the layer
# The piece a node falls back to when the operator names none. It used to be the ONLY piece it would
# load; it is now the ANCHOR whose layer(s) the default residency fills.
DEFAULT_ANCHOR_PIECE = 0

# (abspath(shard_dir), config_dir, anchor, manifest mtime_ns, manifest size) -> resolved default.
# claim_all_coords runs the resolution once per plateau check, and the manifest read behind it is not
# free (603 pieces); the stat in the key keeps the memo honest if the file is rewritten under us.
_DEFAULT_PIECES_MEMO = {}


def piece_layer_map(manifest, config=None):
    """{expert-piece id -> frozenset of the REAL layer ids it covers}. Pure dict/int work, no torch.

    "Real" means the layers an instantiated Glm4MoeLiteForCausalLM actually builds, i.e. exactly
    piece_loader.claimable_expert_ids' two filters: layer 0 is a DENSE MLP with no routed experts, and
    the shard manifest also carries the MTP/nextn layer (L == num_hidden_layers) that the model never
    instantiates. A piece with NO real layer therefore maps to an EMPTY set -- on the live manifest
    pieces 589-601 are 100% MTP -- which is what lets the default-residency rule below drop them
    instead of paying disk for coordinates that can never train. `config=None` disables both filters
    (every layer in the manifest counts), which is only useful for inspection."""
    dense = n_layers = None
    if config is not None:
        dense = int(getattr(config, "first_k_dense_replace", 0) or 0)
        n_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    out = {}
    for rec in manifest.get("pieces", ()) or ():
        name = str(rec.get("piece", ""))
        if not name.startswith("experts_"):
            continue                                  # the trunk piece; always resident, never claimed
        try:
            pid = int(name.split("_", 1)[1])
        except ValueError:
            continue
        layers = set()
        for le in rec.get("experts", ()) or ():
            L = int(le[0])
            if config is None or dense <= L < n_layers:
                layers.add(L)
        out[pid] = frozenset(layers)
    return out


def default_piece_ids(manifest, config, anchor=DEFAULT_ANCHOR_PIECE):
    """The pieces a node keeps resident when the operator names NEITHER --pieces nor --piece. Pure.

    Returns (ids, excluded): `ids` is a sorted list that always contains `anchor`; `excluded` is a
    sorted list of (piece id, [the layers it would ADD]) for the pieces this rule deliberately leaves
    out, so the startup log can show the choice instead of hiding it.

    THE RULE -- fill the anchor's layer(s), never add one. piece_loader allocates a resident layer's
    fused expert params FULL WIDTH (all 64 rows) the moment ONE of its pieces loads
    (piece_loader.py:389-408), so every other piece of that same layer only fills rows that already
    exist. MEASURED 2026-07-26 on D:/hf_models/GLM-4.7-Flash-bf16_shards_100mb: `--piece 0` = 5
    coordinates at 2,764,301,056 params / 1.857 GiB resident, and pieces 0-11 = 60 coordinates at a
    BYTE-IDENTICAL 2,764,301,056 params / 1.859 GiB. Twelve times the trainable universe for +0.002
    GiB. Crossing a LAYER boundary is a different transaction: piece experts_12 holds (1,60)..(1,63)
    PLUS (2,0), and that one coordinate materialises a SECOND full-width MoE layer (+603,979,776
    params = 64 x 3 x 1536 x 2048, +1.126 GiB resident). So a straddler is EXCLUDED here and left to
    the operator: filling a layer is free, buying one is a spending decision.

    WHY IT IS THE DEFAULT and not just a documented flag (measured, run 4,
    scratchpad/FINDING_five_coordinate_ceiling.md): a campaign whose nodes all took the single-piece
    default had a trainable universe of FIVE coordinates, plateaued at held-out CE 6.51103, then ran
    ~620 events over ~7.5 h with ZERO accepted merges while the miners made 129 PLATEAU -> RELEASE ->
    CLAIM cycles across the same five. `--pieces` fixes that only for someone who already knows to
    type it; a stranger joining tomorrow walks into the identical wall.

    A piece that straddles only into the MTP layer is NOT a straddler here (its real layer set is the
    one real layer), because that layer is never instantiated and so costs nothing.

    ANCHOR THAT IS ITSELF A STRADDLER: its layer set is simply BOTH layers, and the rule is unchanged.
    That is not a special case but the same economics -- the anchor has already materialised both
    layers full-width just by loading, so filling both is still free, and refusing to would leave rows
    of tensors we already paid for permanently untrainable. An anchor with NO real layers at all
    (live pieces 589-601) yields just [anchor], so resolve_claim's existing "holds NO real experts"
    error still fires instead of this function inventing a substitute nobody asked for."""
    layers = piece_layer_map(manifest, config)
    base = layers.get(int(anchor))
    if not base:
        return [int(anchor)], []
    keep, excluded = [], []
    for pid in sorted(layers):
        cov = layers[pid]
        if not cov:
            continue                                  # MTP-only piece: real disk cost, zero claimable
        if cov <= base:
            keep.append(pid)
        elif cov & base:
            excluded.append((pid, sorted(cov - base)))
    return keep, excluded


def _piece_path(shard_dir, pid):
    """Where the loader will look for expert piece `pid`. Deliberately a literal mirror of the path
    piece_loader.load_manifest builds before raising 'piece file missing' (piece_loader.py:178-181):
    if those two ever disagree, the best-effort filter below silently stops protecting anything."""
    return os.path.join(str(shard_dir), "pieces", "experts_%d.safetensors" % int(pid))


def pieces_on_disk(shard_dir, piece_ids):
    """Split `piece_ids` into (present, absent) for THIS shard dir. Never raises; stat work only.

    WHY THIS EXISTS (measured live 2026-07-26): the layer-filling default asks for pieces 0-11, but a
    real miner on the fleet had fetched only experts_0 -- the README quickstart literally says
    `--pieces 0` -- and build_node_model died at startup with
    `FileNotFoundError: piece file missing: C:/Users/User/glm_base\\pieces\\experts_1.safetensors`
    (piece_loader.py:181, reached via build_partial_model -> load_manifest(require_pieces=...)). A
    node that CAN train 5 coordinates must not refuse to train at all because it cannot train 60, and
    every new joiner walked into that same wall.

    THE THIRD STATE MATTERS. A shard dir with NO pieces/ subdirectory at all is the PRE-FETCH METADATA
    case that load_manifest(require_files=False) exists for (piece_loader.py:146-150): a cold node
    reads the expert map to decide what to fetch BEFORE fetching anything, and the claim probe reads
    it without ever loading a weight. Nothing is being loaded there, so there is nothing to intersect
    with and the requested set passes through untouched -- filtering it to [] would break cold-start
    planning to fix a warm-start crash. Once pieces/ exists the dir is a real (possibly partial)
    fetch, and what is in it is the truth about what can be loaded."""
    ids = sorted(int(p) for p in piece_ids)
    pdir = os.path.join(str(shard_dir or ""), "pieces")
    if not shard_dir or not os.path.isdir(pdir):
        return ids, []
    here = [p for p in ids if os.path.exists(_piece_path(shard_dir, p))]
    absent = [p for p in ids if p not in set(here)]
    return here, absent


# (abspath(shard_dir), requested ids) -> the (present, absent) split measured the FIRST time this
# process asked. See _frozen_disk_filter.
_DISK_FILTER_MEMO = {}


def _frozen_disk_filter(shard_dir, piece_ids):
    """pieces_on_disk, but the answer is frozen for the life of the PROCESS.

    RESIDENCY IS A STARTUP DECISION. build_node_model loads the weights once; node_claimable_coords
    is then re-read on every plateau check. Without this freeze, an operator who pasted the fetch
    command this module prints WITHOUT restarting would widen the claimable set at the next check to
    coordinates the running model does not hold -- and a non-resident row of a resident layer is
    writable but router-masked to -inf (piece_loader.py:366-385), i.e. it would train forever and be
    gate-rejected forever with nothing in any log. The advice printed alongside says 'then restart';
    this makes forgetting to harmless instead of silently poisonous."""
    key = (os.path.abspath(str(shard_dir)) if shard_dir else None,
           tuple(int(p) for p in piece_ids))
    if key not in _DISK_FILTER_MEMO:
        _DISK_FILTER_MEMO[key] = pieces_on_disk(shard_dir, piece_ids)
    return _DISK_FILTER_MEMO[key]


def fetch_pieces_cmd(shard_dir, piece_ids):
    """The exact command that fetches the named pieces into this shard dir. An operator told WHAT is
    missing must not then have to work out HOW to get it -- that gap is why the single-piece quickstart
    silently became a five-coordinate ceiling in the first place.

    `--skip-trunk` is appended only when trunk.safetensors is already there: it saves a multi-GB
    re-download in the normal case (a node that is already training obviously has the trunk), but
    pasting it at a node that lacks the trunk would hand back a dir that still cannot load."""
    cmd = ("python tools/fetch_glm_base.py --dest %s --pieces %s"
           % (shard_dir, ",".join(str(int(p)) for p in sorted(piece_ids))))
    if os.path.exists(os.path.join(str(shard_dir or ""), "pieces", "trunk.safetensors")):
        cmd += " --skip-trunk"
    return cmd


def _default_pieces_for(args, anchor):
    """(ids, excluded, layers, why) for the layer-filling default. NEVER raises.

    Both roles print this in their startup log line, and a broken/absent manifest already has exactly
    one loud owner (node_claimable_coords -> _raise_claim_probe, which fires on the very next call for
    a real --mode glm run). Two different crashes for one cause is worse than one, so every failure
    here degrades to the historical single-anchor residency and says so in `why`."""
    shard_dir = getattr(args, "shard_dir", None)
    if getattr(args, "mode", None) != "glm" or not shard_dir:
        return ([int(anchor)], [], None,
                "anchor piece %d only (no GLM shard manifest in this mode)" % anchor)
    try:
        st = os.stat(os.path.join(shard_dir, "model_manifest.json"))
        key = (os.path.abspath(shard_dir), getattr(args, "config_dir", None), int(anchor),
               st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _DEFAULT_PIECES_MEMO:
        return _DEFAULT_PIECES_MEMO[key]
    try:
        import piece_loader
        man = piece_loader.load_manifest(shard_dir, require_files=False)
        cfg = _resolve_claim_config(args)
        ids, excluded = default_piece_ids(man, cfg, anchor)
        layers = sorted(piece_layer_map(man, cfg).get(int(anchor), ()))
        out = (ids, excluded, layers,
               "filled layer(s) %s from anchor piece %d"
               % (",".join(str(L) for L in layers) or "(none real)", anchor))
    except Exception as ex:                                          # noqa: BLE001
        out = ([int(anchor)], [], None,
               "anchor piece %d only (%s: %s)" % (anchor, type(ex).__name__, ex))
    if key is not None:
        _DEFAULT_PIECES_MEMO[key] = out
    return out


def resolve_piece_selection(args):
    """How THIS node's resident piece set was chosen -- the ONE place either role resolves it.

    Returns a dict: `ids` (sorted piece ids), `source` ('--pieces' | '--piece' | 'default' | 'none'),
    `anchor`, `layers` (the real layers the anchor covers, or None when unknown), `excluded`
    ([(piece id, [layers it would add])]), `absent` (default-set pieces dropped because their files
    are not on disk) and `note` (one line of plain English for the startup log).

    LOCKSTEP. The coordinator imports this module as N and calls the same function, which is the whole
    reason it lives here: a miner that widened its residency alone would just be told `not hostable
    here` for every coordinate the coordinator does not hold, and the two roles cannot drift if there
    is only one implementation.

    PRECEDENCE, deliberately in this order: an explicit --pieces wins, then an explicit --piece
    (NEURAHASH_GLM_PIECES / NEURAHASH_GLM_PIECE are the same request typed elsewhere and count as
    explicit), and ONLY when the operator named neither does the layer-filling default apply. So
    `--piece 0` still means exactly the five coordinates it meant yesterday: the new default can never
    silently change a launch command that already exists.

    AND THE DEFAULT IS BEST-EFFORT OVER WHAT IS ACTUALLY FETCHED (pieces_on_disk). The layer-filling
    rule asks for pieces 0-11 on the live manifest, but the published quickstart tells a new joiner to
    fetch `--pieces 0`, so on 2026-07-26 a live 4060 that had only experts_0 died at startup with
    `FileNotFoundError: piece file missing: .../pieces/experts_1.safetensors`. A node that can train 5
    coordinates must not refuse to train because it cannot train 60: the default intersects with disk,
    logs what it skipped and the fetch command for it, and only an EMPTY intersection is fatal.
    EXPLICIT selections keep failing loudly instead -- the operator named those exact pieces, and
    quietly handing back a smaller set than someone asked for is worse than the crash."""
    spec = getattr(args, "pieces", None)
    if spec is not None and str(spec).strip() != "":
        return {"ids": parse_pieces(spec), "source": "--pieces", "anchor": None, "layers": None,
                "excluded": [], "absent": [], "note": "explicit --pieces %s" % str(spec).strip()}
    p = getattr(args, "piece", None)
    if p is not None:
        return {"ids": [int(p)], "source": "--piece", "anchor": int(p), "layers": None,
                "excluded": [], "absent": [],
                "note": "explicit --piece %d -- residency PINNED to that one piece, the "
                        "layer-filling default is off" % int(p)}
    if not hasattr(args, "piece") and not hasattr(args, "pieces"):
        # A namespace predating both flags (the async lane's dirty-namespace test builds one). It
        # cannot express a selection at all, and [] keeps claims UNCHECKED exactly as they were.
        return {"ids": [], "source": "none", "anchor": None, "layers": None, "excluded": [],
                "absent": [], "note": "no piece selection in this namespace"}
    anchor = DEFAULT_ANCHOR_PIECE
    ids, excluded, layers, why = _default_pieces_for(args, anchor)
    shard_dir = getattr(args, "shard_dir", None)
    ids, absent = _frozen_disk_filter(shard_dir, ids)
    if not ids:
        # Genuinely unusable, not a degrade: pieces/ exists and holds none of them, so there is no
        # expert to train at all. Naming the dir AND the command is the difference between a stranger
        # fixing this in one paste and a stranger giving up.
        raise SystemExit(
            "[glm-node] FATAL: no expert piece is on disk. The DEFAULT resident set for anchor piece "
            "%d is %s, and %s/pieces holds none of them, so this node has nothing to train. Fetch at "
            "least the anchor:\n    %s"
            % (anchor, fmt_coord_pieces(absent) or "(empty)", shard_dir,
               fetch_pieces_cmd(shard_dir, absent or [anchor])))
    note = "DEFAULT (no --pieces/--piece given), " + why
    if absent:
        # BEST EFFORT, and SAID OUT LOUD. Training fewer coordinates than the layer offers is a
        # legitimate state (it is what one fetched piece buys), but a silent one would recreate the
        # five-coordinate ceiling with nothing in any log to explain the plateau -- run 4 exactly.
        note += ("; using %d of %d piece(s) -- SKIPPED %s because %s does NOT have those files (a "
                 "partial fetch; this node trains proportionally fewer coordinates than its layer "
                 "offers). Get the rest with (then RESTART -- residency is fixed at load): %s"
                 % (len(ids), len(ids) + len(absent), fmt_coord_pieces(absent),
                    os.path.join(str(shard_dir), "pieces"), fetch_pieces_cmd(shard_dir, absent)))
    if excluded:
        note += ("; excluded straddler(s) %s because each would ADD layer(s) %s -- a new layer is a "
                 "full-width MoE slab (measured +1.126 GiB on GLM-4.7-Flash), not a free fill; pass "
                 "--pieces to buy them"
                 % (", ".join(str(pid) for pid, _ in excluded),
                    ", ".join(str(L) for _, ls in excluded for L in ls)))
    return {"ids": ids, "source": "default", "anchor": anchor, "layers": layers,
            "excluded": excluded, "absent": absent, "note": note}


def node_piece_ids(args):
    """The expert pieces THIS node keeps resident -- the single place both roles resolve them.

    WHY THIS EXISTS (measured, run 4, scratchpad/FINDING_five_coordinate_ceiling.md): `--piece` is one
    int, one piece covers 5 (layer, expert) coordinates, so a whole campaign's trainable universe was
    5 coordinates. It plateaued at held-out CE 6.51103 and then ran ~620 events with ZERO accepted
    merges while both miners cycled PLATEAU -> RELEASE -> CLAIM across the same five forever. Shard
    Claim was working; it had nowhere to go. Layer 1 is covered by pieces 0..12 (all 64 experts), and
    piece_loader already allocates a resident layer's fused params FULL WIDTH, so widening residency
    fills rows of tensors that are allocated either way.

    Selection and precedence live in resolve_piece_selection; this is the ids-only view of it."""
    return resolve_piece_selection(args)["ids"]


def fmt_pieces(args):
    """How this node's piece selection is NAMED in an error message -- the flag the operator actually
    passed, so the fix they are told to make is the fix that applies to their command line. When they
    passed NOTHING, say that: telling a stranger to "fix --piece 0" when their command line contains
    no --piece at all sends them hunting for something that is not there."""
    sel = resolve_piece_selection(args)
    if sel["source"] == "--pieces":
        return "--pieces %s" % str(getattr(args, "pieces", None)).strip()
    if sel["source"] == "--piece":
        return "--piece %s" % (getattr(args, "piece", None),)
    if sel["source"] == "none":
        return "(no piece selection)"
    return ("the DEFAULT resident set (%d piece(s): %s; no --pieces/--piece given)"
            % (len(sel["ids"]), fmt_coord_pieces(sel["ids"]) or "(none)"))


def fmt_piece_selection(args):
    """The startup-log description of the resident piece set: WHAT was chosen and WHY, one line, on
    BOTH roles.

    An operator who disagrees with the default has to be able to see it and override it without
    reading the source. Run 4 is the cost of the opposite: nothing in any log said "your trainable
    universe is five coordinates" until a 7.5 h post-mortem went looking."""
    sel = resolve_piece_selection(args)
    return "%s (%d piece(s), %s)" % (fmt_coord_pieces(sel["ids"]) or "(none)",
                                     len(sel["ids"]), sel["note"])


def fmt_coord_pieces(piece_ids, limit=8):
    """'0, 1, 2 ... (+10 more)' -- compact piece list for a startup log line (13 ids would otherwise
    push the useful part of the summary off the line)."""
    ids = [int(p) for p in piece_ids]
    head = ", ".join(str(p) for p in ids[:limit])
    extra = len(ids) - limit
    return head + ((" (+%d more)" % extra) if extra > 0 else "")


def check_residency(n_resident, claimable, piece_ids):
    """FAIL LOUDLY when the loaded expert count does not equal the requested one. Pure (ints only).

    A piece id resolves by NAME (`experts_<id>`), and the manifest's list also carries the trunk, so
    an off-by-one in a --pieces range does not raise anywhere: build_partial_model happily loads the
    wrong 13 pieces, the intended layer comes back e.g. 59/64 resident, and the 5 missing rows are
    writable-but-inert (zero weights, router pinned to -inf). A miner claiming one of them would be
    gate-rejected forever with nothing in any log to say why -- the failure mode this whole
    claimability guard exists to prevent. `claimable is None` means residency is unchecked (tiny mode
    / no manifest), which stays a no-op exactly as before.

    "REQUESTED" IS THE POST-DISK-FILTER SET, not the ideal one. Both sides of this comparison come
    from resolve_piece_selection (piece_ids here, and `claimable` via node_claimable_coords ->
    node_piece_ids), so a best-effort default that dropped un-fetched pieces is compared against what
    it actually asked the loader for. Comparing against the ideal set instead would make this fire on
    every partially-fetched node -- turning a working 5-coordinate miner into a crash, which is the
    bug this whole path exists to remove."""
    if claimable is None or n_resident is None:
        return None
    got, want = int(n_resident), len(claimable)
    if got != want:
        raise SystemExit(
            "[glm-node] FATAL residency mismatch: asked for %d piece(s) %s = %d claimable "
            "coordinate(s), but the loaded model has %d resident expert(s). A piece id names "
            "`experts_<id>` in the manifest, NOT a position in its piece list -- an off-by-one here "
            "is silent (the missing rows are writable but never routed). Refusing to train."
            % (len(list(piece_ids)), fmt_coord_pieces(piece_ids), want, got))
    return got


def coord_data_slot(L, E):
    """The `slot` argument to pass to _ids_path / node_ids / coord_secret_ids when work is addressed by
    COORDINATE rather than by position.

    Why this has to exist. `_ids_path` picks a data domain as `doms[slot % len(doms)]`, and the miner's
    TRAIN shard and the coordinator's SECRET PROBE pool must land on the same domain or every delta is
    gated against text from a domain it never trained on -- a systematic reject with no error anywhere.
    Under positional addressing both sides happened to pass the same list index. Once the miner names a
    coordinate and the coordinator assigns its own registry index, those two integers are unrelated, so
    the domain has to come from the coordinate itself.

    Returning E reproduces today's LIVE mapping exactly -- (1,0) -> code, (1,1) -> gutenberg, the same
    domains slots 0 and 1 resolve to now -- so the frozen probe/heldout pools (c0d4cbd) keep their
    meaning and the before/after held-out CE stays comparable."""
    return int(E)


def domains_list(args):
    """This process's EFFECTIVE data-domain list, parsed in EXACTLY ONE place so no caller can drift from
    _ids_path -- which is the thing that actually resolves a shard, as `doms[slot % len(doms)]`."""
    return [d.strip() for d in str(args.domains).split(",") if d.strip()]


def domains_canonical(doms):
    """Reduce a domain list to the MINIMAL list that induces the SAME shard mapping.

    What has to agree between the roles is the FUNCTION `E -> doms[E % len(doms)]`, not the spelling. The
    live pair really is spelled two ways -- coordinator `daily,daily`, contributor `daily` -- and both send
    every coordinate to the same domain, so digesting the literal list would refuse to start a configuration
    that is provably safe. A purely periodic sequence has a unique minimal period and that period DIVIDES
    its length, so the smallest-period prefix is a canonical form: two lists induce the same mapping iff
    their canonical forms are equal. ['daily','daily'] -> ['daily']; ['code','gutenberg','web'] is already
    minimal; ['gutenberg','code'] stays distinct from ['code','gutenberg']. Pure."""
    doms = [str(d) for d in doms]
    n = len(doms)
    for p in range(1, n + 1):
        if n % p == 0 and all(doms[i] == doms[i % p] for i in range(n)):
            return doms[:p]
    return doms


def domains_digest(doms):
    """sha256 over the CANONICAL form of a domain list (domains_canonical). Order is part of the identity,
    not cosmetic: a shard resolves as `doms[coord_data_slot(L,E) % len(doms)]`, so 'code,gutenberg' and
    'gutenberg,code' send E=0 to DIFFERENT domains, and appending one entry renumbers every modulus. Pure."""
    return hashlib.sha256("\n".join(domains_canonical(doms)).encode("utf-8")).hexdigest()


def domains_pointer_fields(args):
    """The two ADDITIVE keys the coordinator stamps onto its v2 pointer so the roles can cross-check the
    one flag nothing else validates. `domains` is only there to make a mismatch message readable -- the
    domain NAMES are already public (the data record advertises ids_<domain>_train.npy); the secret is the
    probe/heldout CONTENT, which is never named here."""
    doms = domains_list(args)
    return {"domains": list(doms), "domains_digest": domains_digest(doms)}


def domains_mismatch(ptr, args):
    """FIX B (C6, 2026-07-25): does the coordinator resolve data shards against the same domain list we do?
    Returns None when they agree, or when the pointer carries NO digest (a pre-Shard-Claim coordinator --
    additive by design, never a hard fail); otherwise a message naming BOTH lists.

    Why this has to exist. `--domains` is a PER-PROCESS flag and nothing cross-verified it. The miner's
    TRAIN shard is `doms[coord_data_slot(L,E) % len(doms)]` on ITS list and the coordinator's SECRET PROBE
    pool is the same expression on ITS list, so coordinator `code,gutenberg` against a miner on
    `code,gutenberg,web` gives E=2 the probe pool "code" versus the train shard "web": every delta is then
    gated on text it never trained on and rejected SYSTEMATICALLY WITH NO ERROR ANYWHERE. Comparison is on
    the CANONICAL mapping (domains_canonical), so the live pair -- `daily,daily` against `daily`, two
    spellings of one mapping -- keeps running; only a genuinely different mapping is refused. Pure."""
    if not isinstance(ptr, dict):
        return None
    theirs_digest = ptr.get("domains_digest")
    if not theirs_digest:
        return None                                          # pre-Shard-Claim peer: nothing to compare
    ours = domains_list(args)
    if str(theirs_digest) == domains_digest(ours):
        return None
    theirs = ptr.get("domains")
    theirs_txt = ",".join(str(d) for d in theirs) if theirs else "(not published)"
    return ("--domains MISMATCH. Coordinator resolves data shards against [%s] (digest %s..), this miner "
            "against [%s] (digest %s..). A shard is doms[E %% len(doms)], so a different list -- or the "
            "same names in a different ORDER -- makes the coordinator gate every delta on a domain this "
            "miner never trained on: a systematic reject with NO error anywhere. Refusing to start; "
            "restart with --domains %s (or NEURAHASH_GLM_DOMAINS=%s)."
            % (theirs_txt, str(theirs_digest)[:12], ",".join(ours), domains_digest(ours)[:12],
               theirs_txt, theirs_txt))


def fmt_coords(coords, limit=8):
    """'1:0, 1:1, ... (+N more)' -- compact coordinate list for a startup log line."""
    if coords is None:
        return "unchecked"
    head = ", ".join("%d:%d" % (int(L), int(E)) for (L, E) in list(coords)[:limit])
    extra = len(coords) - limit
    return head + ((" (+%d more)" % extra) if extra > 0 else "")


def node_claimable_coords(args):
    """The coordinates THIS node can genuinely host, or None in tiny mode / when the manifest is
    unavailable (then claims are unchecked, as before).

    This is the guard against the worst failure mode in the lane. piece_loader allocates a resident
    layer's fused expert params FULL WIDTH -- all 64 rows -- and fills only the rows its pieces cover,
    masking the rest of the router to -inf (piece_loader.py:366-385, measured 2026-07-25). So a
    coordinate this node does NOT hold is still writable and reads back as zeros: a miner claiming one
    would train happily, publish, and be gate-rejected forever, with nothing in any log to say why.
    Refusing the claim at startup is the only cheap defence."""
    # Only a REAL GLM run has a shard manifest to check against. tiny mode -- and any caller passing a
    # partial namespace (the async lane's dirty-namespace test does exactly that) -- means "we cannot
    # tell", which must be UNCHECKED, not fatal: refusing to start because we could not find a manifest
    # we never needed would be a self-inflicted outage.
    if getattr(args, "mode", None) != "glm":
        return None
    if not getattr(args, "shard_dir", None):
        return None
    pieces = node_piece_ids(args)
    if not pieces:
        return None
    try:
        import piece_loader
        man = piece_loader.load_manifest(args.shard_dir, require_files=False)
        cfg = _resolve_claim_config(args)
        # UNION over every selected piece -- claimable_expert_ids already takes a list and folds the
        # per-piece expert sets together, then filters the dense layer 0 and the MTP/nextn layer out.
        return piece_loader.claimable_expert_ids(man, pieces, cfg)
    except ImportError:
        return None
    except Exception as ex:                                          # noqa: BLE001
        return _raise_claim_probe(ex)


def _raise_claim_probe(ex):
    raise SystemExit("[glm-node] cannot determine this node's claimable coordinates from "
                     "--shard-dir/--pieces (%s: %s). Fix the shard dir, or pass --expert only after "
                     "confirming the selected piece(s) cover it." % (type(ex).__name__, ex))


def _resolve_claim_config(args):
    """Just enough config for the claimability filter: num_hidden_layers + first_k_dense_replace. Read
    from config.json directly so this stays a cheap JSON read -- no model, no torch, no GPU."""
    import types
    for d in (getattr(args, "config_dir", None), getattr(args, "shard_dir", None)):
        if not d:
            continue
        p = os.path.join(d, "config.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                c = json.load(f)
            return types.SimpleNamespace(
                num_hidden_layers=int(c.get("num_hidden_layers", 0)),
                first_k_dense_replace=int(c.get("first_k_dense_replace", 0)))
    raise SystemExit("[glm-node] no config.json under --config-dir or --shard-dir; cannot tell which "
                     "layers are real MoE layers, so a claim cannot be validated")


def pick_start_coord(claimable, identity):
    """Deterministic starting coordinate derived from the miner's IDENTITY.

    Shard claim has no registry and no lock, so two miners must not both start at coordinate 0 by
    default. Hashing the wallet address spreads independent miners across the claimable space with
    zero coordination -- a stranger just runs the miner and lands somewhere. Duplicate work is
    WASTEFUL, NOT INCORRECT (both deltas are gated; the better one wins), which is why this is a hash
    and not a lease: a lease needs a registry, and a registry needs to be trusted. Mirrors the
    existing hash-of-address precedent in derive_glm_miner_name."""
    if not claimable:
        raise SystemExit("[glm-node] no claimable coordinates to start from")
    h = int(hashlib.sha256(str(identity).encode()).hexdigest(), 16)
    return tuple(claimable[h % len(claimable)])


def claim_walk_order(claimable, identity=None, ranked=None):
    """The ORDER in which `identity` sweeps the claimable set: a per-identity PERMUTATION of the same
    coordinates, produced by sorting them on sha256(identity|L:E). `identity=None` -> the input order
    (the legacy shared sweep). Registry-free, lock-free and pure -- the same hash-of-address trick
    pick_start_coord already uses, applied to the walk instead of just its first step.

    `ranked` (--claim-by affinity) OVERRIDES the hash permutation with a measured order: the ESFT
    affinity ranking, highest first, filtered to what is claimable here. Any claimable coordinate the
    probe did not score is appended in input order so the sweep still eventually covers everything --
    a coordinate silently dropped from the walk would be starved forever."""
    coords = [tuple(c) for c in claimable or []]
    if ranked:
        seen = set()
        order = [tuple(c) for c in ranked if tuple(c) in set(coords) and not (
            tuple(c) in seen or seen.add(tuple(c)))]
        return order + [c for c in coords if c not in seen]
    if identity is None:
        return coords
    return sorted(coords, key=lambda c: hashlib.sha256(
        ("%s|%d:%d" % (identity, int(c[0]), int(c[1]))).encode()).hexdigest())


def next_claim_coord(claimable, current, identity=None, ranked=None):
    """The coordinate to claim after `current`, cycling through THIS identity's walk order.

    This is the owner's "finish one then start the 2nd one" -- sweeping the space is just
    claim -> work -> plateau -> release -> claim next. Returns None when the set has only this one
    coordinate (nothing to advance to, so the caller should stay put rather than churn).

    WHY THE ORDER IS PER-IDENTITY (measured live 2026-07-25). pick_start_coord spreads miners by hashing
    the wallet, but a shared +1 advance walked everyone through the SAME sequence, so a one-off collision
    became a PERMANENT one: the 5090 (glm-ea20C873) swept 1:1 -> 1:2 -> 1:3 -> 1:4 into the 4060
    (glm-361447E3) sitting on 1:2, and they stayed together for events 12-15 -- every one a reject, with
    held-out CE frozen at 7.76966 -- while the coordinates neither miner reached starved. Duplicate work
    is wasteful-but-not-incorrect by design; duplicate work that cannot END is a different bug. Walking a
    per-identity permutation makes the collision transient (the two identities' successors differ) without
    a registry or a lease. Still deterministic per identity, so a restarted miner behaves predictably; and
    still a single cycle over the whole permutation, so every claimable coordinate is eventually visited
    and the coordinate we are on now is never returned.

    `ranked` (--claim-by affinity) replaces that permutation with the ESFT affinity order, so "advance
    on plateau" becomes "drop to the NEXT-HIGHEST-affinity coordinate" rather than to the next hash
    bucket -- see claim_walk_order and the ESFT block below.

    NEVER-BLOCK V0: the async loop now advances through `advance_claim`, which walks the SAME order
    but can keep going when a candidate is on cooldown -- this function is its first candidate, and
    the specification of "next" that it must agree with (see the equivalence test)."""
    coords = [tuple(c) for c in claimable or []]
    if len(coords) <= 1:
        return None
    order = claim_walk_order(coords, identity, ranked=ranked)
    cur = tuple(current)
    if cur not in order:
        return order[0]
    return order[(order.index(cur) + 1) % len(order)]


def record_touched_coord(rec, coord):
    """Did this accepted record MERGE `coord`? Reads the per-coordinate `slot_roots` map the
    coordinator stamps for exactly the one slot each event moved -- so this is a precise "the
    coordinator processed MY expert at this event" signal, which the top-level `slot` int (the
    coordinator's own registry index) is not."""
    return ("%d_%d" % (int(coord[0]), int(coord[1]))) in (rec.get("slot_roots") or {})


def event_judged_us(rec, published_base_event):
    """F5a: could this accepted record possibly be a VERDICT on a delta of ours -- i.e. was one of our
    deltas in flight when the coordinator committed it? True iff we have published at least once and the
    record's event is >= the base_event of our last publish.

    Why the plateau counter needs it. `reject_streak` used to increment for ANY event whose `slot_roots`
    named our coordinate and whose accepted rows did not name us -- which includes events that never
    judged us at all: a co-claimant winning the same coordinate, our own record being dropped for
    lineage/staleness/validate reasons, and above all records that PREDATE our first publish. A fresh
    miner joining a running campaign folds the entire history in ONE catch-up pass, so the streak blew
    straight past --advance-after (default 3) and it abandoned its coordinate before publishing once.
    `published_base_event` None (nothing published yet) -> False: nothing can have judged us. Pure."""
    if published_base_event is None:
        return False
    try:
        return int(rec.get("event")) >= int(published_base_event)
    except (TypeError, ValueError):
        return False                                     # undateable record: never counts as a verdict


def resolve_claim(args, slots, log=print, identity=None):
    """Resolve which GLM coordinate this miner works on, from --expert L:E (shard claim) or the
    DEPRECATED --slot index. Returns (L, E, i, source).

    `i` is only this miner's LOCAL index into its own `slots` list -- the position it reads and writes
    weights at. A shard-claim coordinator resolves work by the (layer, expert) coordinate on the wire
    and assigns its own registry index, so the two integers no longer have to agree. `slots` is
    EXTENDED in place when a claimed coordinate is not already in it, because the lane host is built
    over this list and can only touch a coordinate it contains."""
    claimable = node_claimable_coords(args)
    if getattr(args, "expert", None):
        L, E = parse_coord(args.expert)
        src = "--expert"
        if args.slot is not None:
            log("[glm-contrib] NOTE: --slot %d ignored; --expert %s:%s wins (--slot is deprecated)"
                % (int(args.slot), L, E))
    elif args.slot is None and not os.environ.get("NEURAHASH_SD_EXPERT") and claimable:
        # Nothing was asked for -> SPREAD. A stranger who just runs the miner must not collide with
        # every other default-configured miner on coordinate 0.
        L, E = pick_start_coord(claimable, identity if identity is not None else "anonymous")
        src = "wallet-hash (auto-spread)"
    else:
        idx = int(os.environ.get("NEURAHASH_SD_EXPERT", "0") if args.slot is None else args.slot)
        if not (0 <= idx < len(slots)):
            raise SystemExit("[glm-contrib] --slot %d out of range for --slots %s" % (idx, args.slots))
        L, E = slots[idx]
        src = "--slot (deprecated)"
    # Order matters: an EMPTY claimable set means the piece covers only the MTP/nextn layer, and it
    # needs its own message -- reporting "piece 0 does not hold (L1,E0)" for a piece that holds nothing
    # at all sends the reader looking for the wrong problem.
    if claimable is not None and not claimable:
        raise SystemExit("[glm-contrib] %s holds NO real experts (it covers only the MTP/nextn "
                         "layer, which the model never instantiates). Pick another piece."
                         % fmt_pieces(args))
    if claimable is not None and (L, E) not in claimable:
        shown = ", ".join("%d:%d" % c for c in claimable[:16]) or "(none)"
        raise SystemExit(
            "[glm-contrib] REFUSING to claim (L%d,E%d): %s does not hold it, so this node "
            "would train an INERT expert -- writable, never routed, rejected forever, silently. "
            "Claimable here: %s%s. Load the piece(s) covering (L%d,E%d) instead (--pieces accepts a "
            "range, e.g. 0-12 for all 64 experts of layer 1)."
            % (L, E, fmt_pieces(args), shown,
               (" ... (%d total)" % len(claimable)) if len(claimable) > 16 else "", L, E))
    if (L, E) in slots:
        i = slots.index((L, E))
    else:
        i = len(slots)
        slots.append((L, E))            # the host is built over `slots`; it must contain our claim
    return L, E, i, src


def claim_all_coords(args, slots):
    """The coordinate set a miner may sweep: its claimable set when known, else just `slots`.

    A miner that advances on plateau must stay inside the coordinates it genuinely holds, or it walks
    straight into the inert-slot trap (writable, never routed, rejected forever, silent)."""
    c = node_claimable_coords(args)
    return [tuple(x) for x in (c if c else slots)]


# ======================================================== ESFT expert-affinity selection (--claim-by)
# WHY THIS EXISTS -- MEASURED ELSEWHERE, do not re-derive (docs/research/MOE_POSTTRAIN_2026-07-25.md,
# docs/SHARD_CLAIM_DESIGN.md "Selecting the coordinate by AFFINITY").
#
# pick_start_coord decides which expert to train by HASHING THE WALLET ADDRESS, and next_claim_coord
# advances along a per-identity permutation. Both are routing-BLIND, and routing-blind selection is the
# one variant published work has measured to LOSE:
#   * MoE-Sieve (arXiv:2603.24044) measured random expert selection 2.5 percentage points WORSE than
#     router-guided selection at a matched budget.
#   * Mixtral (arXiv:2401.04088 sec 5) measured expert assignment across 8 Pile domains at layers
#     0/15/31: "Surprisingly, we do not observe obvious patterns in the assignment of experts based on
#     the topic" -- routers organise by SYNTAX/POSITION, not subject. Nobody gets to DECIDE what an
#     expert specialises in; the frozen router already decided, so the job is to MEASURE its choice.
#   * Branch-Train-Merge (arXiv:2208.03306), at this lane's exact 64-expert count: domain
#     specialization is REQUIRED -- "LM ensembles with random data splits do not perform well".
# ESFT (arXiv:2407.01906, EMNLP 2024) is the published method on a near-identical architecture
# (DeepSeek-V2-Lite: 64 routed + 2 shared experts, top-6): it PROBES first with a small subset (32
# samples x 4096 tokens ~= 131K tokens), scores every expert, and trains only the top-scored ones --
# "the routing distribution for a specific task tends to be highly concentrated, while the distribution
# of activated experts varies significantly across different tasks."
ESFT_P_GATE = 0.1          # ESFT verbatim: "The threshold p is set to 0.1 for ESFT-Gate and 0.2 for
ESFT_P_TOKEN = 0.2         # ESFT-Token, respectively"
ESFT_PROBE_SAMPLES = 32    # ESFT's own N_s. Forward passes only -- no optimizer, no backward.


def esft_select(scores, p):
    """ESFT's selection rule: the SMALLEST top-scored set E_s^l with SUM_{i in E_s^l} R_i^l >= p.

    `scores` is a {coord: score} mapping or an iterable of (coord, score). Returns the coordinates in
    descending-score order, stopping at the first one that brings the running sum to >= p. Ties break
    on the coordinate itself so the result is deterministic across processes. p <= 0 -> the empty set
    (the smallest set clearing 0 really is nothing); a total below p -> every coordinate, because there
    is no smaller set that can clear it. Pure -- no torch, no model."""
    items = list(scores.items() if hasattr(scores, "items") else scores)
    items.sort(key=lambda kv: (-float(kv[1]), tuple(kv[0])))
    out, acc, p = [], 0.0, float(p)
    if p <= 0.0:
        return out
    for c, s in items:
        out.append(tuple(c))
        acc += float(s)
        if acc >= p:
            break
    return out


def esft_select_layers(scores, p):
    """esft_select applied PER LAYER, then unioned -- ESFT's actual rule, whose scores are g_i^l / r_i^l
    with the layer superscript, and whose thresholds are calibrated against a per-layer total of 1.

    Pooling layers instead would silently inflate the threshold's reach: with 2 candidate layers the
    pooled scores sum to 2, so p=0.1 clears on half the mass it is meant to describe. (On a real
    single-piece node every claimable coordinate sits in ONE layer, so the two agree there -- but a node
    holding more pieces spans layers and would diverge.) Pure."""
    items = list(scores.items() if hasattr(scores, "items") else scores)
    by_layer = {}
    for c, s in items:
        by_layer.setdefault(int(tuple(c)[0]), []).append((tuple(c), float(s)))
    out = []
    for L in sorted(by_layer):
        out.extend(esft_select(by_layer[L], p))
    return sorted(out)


def _routed_experts_module(host, L):
    """The module `mlp` actually CALLS with (hidden_states, top_k_index, top_k_weights) for layer L --
    the only place the frozen router's decisions are observable without re-deriving them.

    It is `layers[L].mlp.experts`, which is the fused Glm4MoeLiteNaiveMoe normally and the LoRAExperts
    WRAPPER while a contribution is training. Deliberately not host._fused(L): that unwraps to
    `.base`, whose forward LoRAExperts never calls (it reaches into base.gate_up_proj directly), so a
    hook there would silently never fire. host._fused(L) is still called first, purely to reuse its two
    named failures -- IndexError for the MTP/nextn layer the model never instantiates, AttributeError
    for a fully non-resident layer's _DeadExperts placeholder."""
    host._fused(int(L))                                  # validation only (raises the named errors)
    return host.model.model.layers[int(L)].mlp.experts


def probe_expert_affinity(host, ids, coords=None, samples=ESFT_PROBE_SAMPLES, batch=8, log=None):
    """ESFT's expert-affinity probe: FORWARD PASSES ONLY over a bounded token sample, scoring every
    candidate coordinate with both ESFT metrics and returning a ranking.

    The metrics, transcribed from arXiv:2407.01906 (K = experts per token, N_s = samples, L_j = length
    of sample j, g_{i,k}^l = the gate score expert i received for token k of layer l, 0 if unselected):

        ESFT-Gate    g_i^l = (1/N_s) * SUM_j (1/L_j) * SUM_k  g_{i,k}^l
        ESFT-Token   r_i^l = (1/N_s) * SUM_j (1/L_j) * SUM_k [ 1(g_{i,k}^l > 0) / K ]

    Every sample here has the SAME length T (`ids` is [N,T]), so SUM_j (1/L_j) SUM_k collapses exactly
    into one mean over the N_s*T tokens -- which is what makes batching sequences metric-preserving
    rather than an approximation.

    NO TRAINING, EVER. torch.no_grad(), no optimizer, no backward; the only state touched is one
    forward PRE-hook per candidate layer, removed in a finally. tests/test_glm_shard_claim.py asserts
    every parameter and buffer is bit-identical before/after, because a "probe" that trained would
    silently corrupt the frozen base that every per-coordinate lineage root is computed over.

    It reuses the model the miner ALREADY built (`host.model`) and never builds or loads a second one:
    on the real lane the base is 4.02 GiB of trunk plus 1.125 GiB per resident layer (memory
    glm-capacity-per-card), so a second copy does not fit. It also skips the lm_head entirely -- only
    routing is needed, and the [B,T,154880] logits are the single biggest allocation in a GLM forward.

    NOTE ON THE -inf ROUTER MASK. piece_loader allocates a resident layer's fused params FULL WIDTH
    (all 64 expert rows) and masks the non-resident rows of the router to -inf
    (piece_loader.py:403-408), so on a real node only the ~5 resident experts of a layer can be
    selected at all and the surviving gate scores are renormalised over just those. That changes the
    ABSOLUTE scores -- they are affinities within the resident set, not within all 64 -- but NOT the
    RANKING among the coordinates this node can actually claim, which is the only thing this function
    is used to decide.

    Returns a dict: `ranking` [(coord, gate, token), ...] highest-first, `gate`/`token` mappings,
    `select_gate`/`select_token` (esft_select_layers at ESFT's published thresholds -- PER LAYER, since
    both metrics carry the layer superscript and sum to 1 within a layer), `n_samples`, `n_tokens`,
    `topk`, `scaling`.
    """
    import torch
    model = host.model
    cands = sorted({(int(L), int(E)) for (L, E) in
                    (coords if coords is not None else host.slots)})
    if not cands:
        raise SystemExit("[glm-node] affinity probe: no candidate coordinates to score")
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] < 2:
        raise SystemExit("[glm-node] affinity probe needs a [N,T] token sample with N>=1, T>=2; got %r"
                         % (tuple(ids.shape),))
    n = max(1, min(int(samples), int(ids.shape[0])))
    sample = ids[:n]                    # the FIRST n sequences -- deterministic, never a random draw
    by_layer = {}
    for (L, E) in cands:
        by_layer.setdefault(L, []).append(E)

    # `routed_scaling_factor` is applied to the gate weights AFTER top-k normalisation, so per token
    # they sum to that factor, not to 1 (GLM-4.7-Flash: 1.8; DeepSeek-V2-Lite, the model ESFT measured:
    # 1.0). Dividing it out is what keeps ESFT's published p a FRACTION of the routed gate mass -- left
    # in, p=0.1 would silently mean 0.0556 of it on this architecture.
    scaling = float(getattr(host.cfg, "routed_scaling_factor", 1.0) or 1.0)
    topk = int(getattr(host.cfg, "num_experts_per_tok", 1) or 1)
    gate_acc = {c: 0.0 for c in cands}      # SUM over tokens of g_{i,k}   (float64 accumulation)
    tok_acc = {c: 0 for c in cands}         # COUNT of (token, slot) picks landing on i
    tok_seen = {L: 0 for L in by_layer}     # tokens routed through each candidate layer

    def _pre_hook(L):
        def pre(_mod, inputs):
            # inputs == (hidden_states, top_k_index, top_k_weights): exactly what the frozen router
            # handed the experts on THIS forward, so these are the model's real routing decisions and
            # not a re-derivation that could drift from them.
            idx, w = inputs[1], inputs[2]
            flat_i = idx.reshape(-1)
            flat_w = w.reshape(-1).to(torch.float64) / scaling
            for E in by_layer[L]:
                m = flat_i == int(E)
                hits = int(m.sum())
                if hits:
                    gate_acc[(L, int(E))] += float(flat_w[m].sum())
                    tok_acc[(L, int(E))] += hits
            tok_seen[L] += int(idx.shape[0])
            return None
        return pre

    handles = []
    was_training = bool(model.training)
    try:
        for L in by_layer:
            handles.append(_routed_experts_module(host, L).register_forward_pre_hook(_pre_hook(L)))
        model.eval()
        trunk = getattr(model, "model", None)        # skip lm_head: routing is all this needs
        fwd = trunk if trunk is not None else model
        b = max(1, int(batch))
        with torch.no_grad():
            for s in range(0, n, b):
                fwd(input_ids=torch.as_tensor(sample[s:s + b]).to(
                    next(model.parameters()).device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    gate = {c: (gate_acc[c] / tok_seen[c[0]]) if tok_seen[c[0]] else 0.0 for c in cands}
    tokr = {c: (tok_acc[c] / (tok_seen[c[0]] * topk)) if tok_seen[c[0]] else 0.0 for c in cands}
    # Rank on the GATE metric, tie-break on the token ratio then the coordinate. The gate metric leads
    # because the token ratio SATURATES on a real node: 5 resident experts with top_k=4 means 4 of 5
    # are selected for nearly every token, so r_i is ~uniform there while g_i still discriminates.
    ranking = sorted(cands, key=lambda c: (-gate[c], -tokr[c], c))
    out = {"ranking": [(c, gate[c], tokr[c]) for c in ranking],
           "gate": gate, "token": tokr,
           "select_gate": esft_select_layers(gate, ESFT_P_GATE),
           "select_token": esft_select_layers(tokr, ESFT_P_TOKEN),
           "n_samples": n, "n_tokens": int(n * ids.shape[1]), "topk": topk, "scaling": scaling}
    if log:
        log("[glm-node] ESFT affinity probe: %d sample(s) x %d tokens = %d tokens, forward-only, %d "
            "candidate coordinate(s)" % (n, int(ids.shape[1]), out["n_tokens"], len(cands)))
        for c, g, r in out["ranking"]:
            log("[glm-node]   %d:%d  gate=%.6f  token_ratio=%.6f%s%s"
                % (c[0], c[1], g, r,
                   "  [ESFT-Gate p=%.1f]" % ESFT_P_GATE if c in out["select_gate"] else "",
                   "  [ESFT-Token p=%.1f]" % ESFT_P_TOKEN if c in out["select_token"] else ""))
    return out


def affinity_claim(args, host, ids, L, E, i, miner="", log=print):
    """--claim-by affinity: replace the wallet-hash claim with the HIGHEST-AFFINITY claimable
    coordinate, and return the full ranking so the plateau advance follows it too.

    Returns (L, E, i, ranked) where `ranked` is the affinity-descending coordinate list (highest
    first). On any probe failure NOTHING changes -- the miner keeps the hash-chosen coordinate and
    `ranked` is None, so a bad probe degrades to today's behaviour instead of stopping a public miner.
    The caller must re-read this coordinate's data shard afterwards (coord_data_slot), exactly as the
    plateau advance does.

    The probe sample is THIS miner's own train split for the coordinate it resolved by hash. That is a
    deliberate, documented limitation: a shard is doms[coord_data_slot(L,E) % len(doms)], so the
    ranking answers "which of my claimable coordinates does the frozen router prefer for THIS text",
    not "which coordinate is best for its own domain". It is still strictly more informative than a
    wallet hash, which correlates with nothing at all."""
    cands = claim_all_coords(args, list(host.slots))
    try:
        rep = probe_expert_affinity(host, ids, coords=cands, log=log)
    except Exception as ex:                                          # noqa: BLE001
        log("[glm-contrib %s] affinity probe FAILED (%s: %s) -- keeping the hash-chosen claim "
            "(L%d,E%d)" % (miner, type(ex).__name__, ex, L, E))
        return L, E, i, None
    ranked = [c for (c, _g, _r) in rep["ranking"]]
    top = ranked[0]
    if tuple(top) == (int(L), int(E)):
        log("[glm-contrib %s] --claim-by affinity: (L%d,E%d) is ALREADY the highest-affinity "
            "coordinate here (gate=%.6f)" % (miner, L, E, rep["gate"][tuple(top)]))
        return L, E, i, ranked
    idx = host.index_of(*top)
    if idx is None:
        try:
            idx = host.register(*top)
        except (ValueError, RuntimeError) as ex:
            log("[glm-contrib %s] --claim-by affinity: cannot claim top-affinity (L%d,E%d): %s -- "
                "keeping (L%d,E%d)" % (miner, top[0], top[1], ex, L, E))
            return L, E, i, ranked
    log("[glm-contrib %s] --claim-by affinity: RECLAIM (L%d,E%d) -> (L%d,E%d) [local slot %d], "
        "gate %.6f > %.6f. Routing-blind selection is the measured-loser variant (MoE-Sieve: random "
        "2.5pp worse than router-guided); this follows the frozen router instead."
        % (miner, L, E, top[0], top[1], idx, rep["gate"][tuple(top)],
           rep["gate"].get((int(L), int(E)), float("nan"))))
    return int(top[0]), int(top[1]), int(idx), ranked


# ================================================================== JOIN defaults (public testing)
# Owner directive (memory public-testing-unlimited-slots-directive): public testing, UNLIMITED slots,
# anyone may join. A join default that only works for the author is therefore a bug, not a policy:
# a LOOPBACK --url reaches nothing from another machine and an EMPTY --token cannot write, so the
# published Mine snippet was uncopyable and a stranger could not join at all.
#
# WHY PUBLISHING THIS TOKEN IS SAFE -- do NOT "fix" it back into a secret or a <placeholder>:
#   * it is the PUBLIC DEMO token of the anchor content store, deliberately spam-open, and it
#     already ships as the --token default of the esh_worker client (precedent: public commit
#     1fdcd5a, "default the PUBLIC demo content token (no token setup, #71)");
#   * ROTATING IT BREAKS EVERY RUNNING MINER (memory content-token-is-public-demo-token-2026-07-21);
#   * the store's real write defense is per-writer signed PUT + rate/size/prefix bounds
#     (docs/STORE_AUTH_DESIGN.md, tests/test_store_auth.py), not the secrecy of this string.
# The env vars still win over both defaults, so a private or local lane stays one export away.
PUBLIC_LANE_URL = "http://47.84.93.96:8710"
PUBLIC_LANE_TOKEN = "2802648a1e87b4b3c6ca6da2688b4308"


def add_common_args(ap):
    """Args shared by BOTH roles so coordinator and contributor cannot be configured apart."""
    # public lane, not loopback: a stranger must reach a real store with zero flags (see above).
    ap.add_argument("--url", default=os.environ.get("NEURAHASH_CONTENT_URL", PUBLIC_LANE_URL),
                    help="content-lane base URL (default: %(default)s); NEURAHASH_CONTENT_URL wins. "
                         "The default IS the public anchor store, so joining needs no flag.")
    # public DEMO token, not a secret: rotating it breaks every miner (see above).
    ap.add_argument("--token", default=os.environ.get("NEURAHASH_CONTENT_TOKEN", PUBLIC_LANE_TOKEN),
                    help="content-lane write token; NEURAHASH_CONTENT_TOKEN wins. Defaults to the "
                         "PUBLIC DEMO token " + PUBLIC_LANE_TOKEN + " -- shared by design, not a "
                         "credential. (Printed as a literal, not as the resolved default, so an "
                         "operator's private lane token never lands in a --help paste.)")
    ap.add_argument("--mode", default=os.environ.get("NEURAHASH_GLM_MODE", "tiny"),
                    choices=("tiny", "glm"),
                    help="tiny = deterministic build_tiny_glm base (wire shakedown, plan S3); "
                         "glm = real GLM-4.7-Flash piece via piece_loader (plan S4+). REAL MINING "
                         "NEEDS --mode glm: the 'tiny' default is the hermetic-test lane.")
    ap.add_argument("--slots", default=os.environ.get("NEURAHASH_GLM_SLOTS", "1:0,1:1"),
                    help="lane slots as layer:expert pairs, e.g. 1:0,1:1")
    # These two defaults are one node's local layout, NOT a lane fact: mode=glm needs YOUR own
    # `tools/fetch_glm_base.py --dest <dir>` output (--shard-dir <dir>, --config-dir <dir>/config).
    ap.add_argument("--shard-dir", default=os.environ.get(
        "NEURAHASH_GLM_SHARD_DIR", "D:/hf_models/GLM-4.7-Flash-bf16_shards_100mb"),
        help="dir holding pieces/ + manifests, i.e. `fetch_glm_base.py --dest` (mode=glm)")
    ap.add_argument("--config-dir", default=os.environ.get(
        "NEURAHASH_GLM_CONFIG_DIR", "D:/hf_models/GLM-4.7-Flash-bf16"),
        help="dir holding the model config.json, i.e. <fetch --dest>/config (mode=glm)")
    # Residency now defaults to FULL-LAYER (see default_piece_ids): giving NEITHER flag fills every
    # piece whose experts lie entirely inside the layer(s) anchor piece 0 already makes resident, and
    # excludes straddlers. Both flags still override it, so no existing launch script changes.
    # That default is BEST-EFFORT over what is on disk (pieces_on_disk): a node that fetched only the
    # quickstart's `--pieces 0` runs on piece 0 and says so, instead of dying on experts_1.safetensors
    # the way the live 4060 did on 2026-07-26. An EXPLICIT selection is never quietly shrunk.
    ap.add_argument("--pieces", default=os.environ.get("NEURAHASH_GLM_PIECES") or None,
                    help="expert piece ids to keep resident: a single id, a comma list, or an "
                         "INCLUSIVE range ('0-12'). DEFAULT WHEN NEITHER --pieces NOR --piece IS "
                         "GIVEN: every piece whose experts lie ENTIRELY within the layer(s) anchor "
                         "piece 0 already makes resident -- on the live GLM manifest that is pieces "
                         "0-11 = 60 claimable coordinates at a BYTE-IDENTICAL parameter count to one "
                         "piece's 5 (measured 2026-07-26: 2,764,301,056 params and ~1.86 GiB either "
                         "way), because piece_loader allocates a resident layer's fused expert params "
                         "FULL WIDTH regardless. Pieces that would pull in a NEW layer (straddlers: "
                         "experts_12 holds (1,60)..(1,63) PLUS (2,0), a second full-width MoE slab at "
                         "+1.126 GiB) are EXCLUDED from that default and named in the startup log -- "
                         "filling a layer is free, buying one is a spending decision, so name them "
                         "here to buy them. That default is BEST-EFFORT over what you actually "
                         "fetched: pieces whose files are not under <shard-dir>/pieces/ are skipped "
                         "(named in the startup log together with the fetch_glm_base.py command that "
                         "gets them), so fetching only piece 0 gives you a working 5-coordinate node "
                         "instead of a FileNotFoundError. NAMING pieces here is NOT best-effort -- a "
                         "piece you asked for and did not fetch is still a hard failure, because "
                         "silently training a smaller set than you asked for is worse. ALL selected "
                         "experts stay resident on every node so contributor and coordinator route "
                         "identically (plan risk 5); only the CLAIMED coordinate trains, so optimizer "
                         "state does not scale. Overrides --piece when both are given. "
                         "Env: NEURAHASH_GLM_PIECES.")
    ap.add_argument("--piece", type=int, default=os.environ.get("NEURAHASH_GLM_PIECE") or None,
                    help="DEPRECATED single expert piece id. Passing it PINS residency to that ONE "
                         "piece (5 coordinates on the live manifest) and turns the layer-filling "
                         "default OFF -- kept exactly so an existing launch script keeps its "
                         "behaviour byte-for-byte. Unset, piece 0 is the ANCHOR the default fills "
                         "around. Env: NEURAHASH_GLM_PIECE.")
    ap.add_argument("--device", default=os.environ.get("NEURAHASH_GLM_DEVICE", "cpu"),
                    help="torch device. REAL MINING NEEDS --device cuda; the 'cpu' default keeps "
                         "the test suite (and a --help on any box) off the GPU.")
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


_VRAM_MGR = None   # per-process VramManager singleton (set below; one manager per process by design)


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
    global _VRAM_MGR
    _VRAM_MGR = mgr                                 # round loops consult this via _vram_units()
    return mgr


def _vram_units(vm=None):
    """Current sustainable resident-unit capacity, or None when no manager runs (flag off)."""
    m = vm if vm is not None else _VRAM_MGR
    if m is None:
        return None
    try:
        return int(m.tuner.current_units)
    except Exception:                               # noqa: BLE001 -- capacity probe must never crash a round
        return None


def _vram_pause_if_starved(log, miner="", vm=None, poll_s=15.0, sleep_fn=None, max_waits=None):
    """The elastic-VRAM PAUSE the design promised ("it pauses instead of spilling"): when the
    manager advertises 0 sustainable units, BLOCK here (log once, re-check every poll_s) instead
    of entering train/eval that will OOM. Crash this prevents (2026-07-24, keyless live test):
    "[vram-manager] resize 12 -> 0" immediately followed by a fatal CUDA OOM inside heldout_ce
    (openadm_contrib5090.log:60-92). Returns the number of waits; no-manager / units>0 -> 0
    immediately. max_waits + sleep_fn are test seams."""
    sleep_fn = sleep_fn or time.sleep
    waits = 0
    u = _vram_units(vm)
    if u is None or u > 0:
        return 0
    log("[glm-contrib %s] VRAM starved: manager advertises 0 sustainable units -- PAUSED "
        "(re-checking every %.0fs; training resumes when capacity returns)" % (miner, poll_s))
    while True:
        sleep_fn(poll_s)
        waits += 1
        u = _vram_units(vm)
        if u is None or u > 0:
            log("[glm-contrib %s] VRAM recovered (%s unit(s)) -- resuming after %d wait(s)"
                % (miner, "?" if u is None else str(u), waits))
            return waits
        if max_waits is not None and waits >= max_waits:
            return waits


def _is_cuda_oom(exc):
    """True for CUDA out-of-memory errors. torch.cuda.OutOfMemoryError subclasses RuntimeError,
    so a message check covers both it and the classic RuntimeError('CUDA out of memory') --
    and keeps this helper importable torch-free for tests."""
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


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
    piece_ids = node_piece_ids(args)
    model, summ = piece_loader.build_partial_model(
        args.shard_dir, piece_ids, device=args.device, dtype=torch.bfloat16,
        config_dir=args.config_dir, strip_mtp=True)
    if int(summ.get("meta_params_left", 0)) != 0:
        raise SystemExit("[glm-node] FATAL: %d meta params left after load (incomplete piece)"
                         % summ["meta_params_left"])
    # RESIDENCY ASSERTION: what got loaded must equal what was asked for. A piece id indexes the
    # manifest's `experts_<id>` NAME, not its list position, and a mismatch there does NOT raise --
    # it leaves a layer partially resident and every non-resident row inert (router -inf), i.e. the
    # exact silent failure this lane already paid for once.
    check_residency(summ.get("n_resident_experts"), node_claimable_coords(args), piece_ids)
    model.eval()
    if log:
        log("[glm-node] GLM pieces_here=%s resident: %s" % (fmt_piece_selection(args), summ))
    seq = _infer_seq(args)                      # from a split THIS role holds (never the secret one)
    return model, model.config, seq


def _ids_path(args, slot, split, base=None):
    """Path to a split's id file. `base` overrides args.data_dir -- the coordinator passes its
    coordinator-only dir (args.coord_data_dir) for the secret probe/heldout splits (F1), so those
    files are read from a dir that is never present on a miner box."""
    doms = domains_list(args)                 # ONE parse for the whole module (see domains_digest)
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
    _campaign_seed_into(h, host_campaign_id(host))
    for i in range(len(host.slots)):
        _slot_digest_into(h, host, i)
    return h.hexdigest()


def _campaign_seed_into(h, campaign_id):
    """Seed a root digest with the CAMPAIGN, so two campaigns over an identical pristine base do not
    share a lineage root at any height (see CAMPAIGN SCOPING at the top of this file: at genesis they
    provably did, and a dead run's delta was therefore lineage-valid and got minted).

    Deliberately the FIRST thing fed into the digest -- a prefix, not a suffix -- so it scopes the whole
    hash rather than being appended to a value someone else could have already produced. An unset
    campaign (None / "") feeds NOTHING, which is what keeps every pre-campaign root, pointer and
    accepted record byte-identical. Pure."""
    if campaign_id:
        h.update(("campaign:%s|" % campaign_id).encode())


def _slot_digest_into(h, host, idx):
    """Feed ONE slot's coordinate tag + canonical fp32 weights into `h`. Factored out of model_root so
    model_root and slot_root cannot drift apart -- they must agree on what a slot's identity IS."""
    d = host.read_slot(idx)
    L, E = host.slots[idx]
    h.update(("L%dE%d|" % (L, E)).encode())
    for k in sorted(d):
        h.update(k.encode())
        h.update(np.ascontiguousarray(d[k], dtype=np.float32).tobytes())


def slot_root(host, idx):
    """SHARD CLAIM: the lineage fingerprint of ONE coordinate, not of the whole slot list.

    Why this exists. `model_root` hashes EVERY slot in `host.slots` in list order, and the
    coordinator rejects a contribution whose `base_root` does not equal its own root at that event
    (_lineage_ok). That coupling makes dynamic slot registration impossible: the moment the
    coordinator admits a new coordinate its global root changes, and every miner already training --
    on untouched, perfectly valid weights -- is dropped as `wrong-lineage-root`.

    Per-coordinate roots break the coupling, and they are strictly MORE precise for this lane, whose
    whole premise is that each expert's LoRA trains against a FROZEN trunk with no cross-expert
    dependency: what actually matters is "did this miner train against MY current weights for THIS
    expert", and another expert moving is irrelevant to that question.

    Two side benefits (2026-07-25):
      * O(1) PER CALL: one coordinate instead of n_slots. What that has NOT yet bought is an O(1)
        round path -- the O(n_slots) `model_root` calls are still there, and this docstring used to
        overclaim. Measured: model_root costs 0.0211 s per slot and is called ~3x/event
        coordinator-side, 1x/round miner-side, and once PER FOLDED RECORD on replay -- at 64 slots a
        1345-record replay is ~30 min of pure hashing; at 2944 slots ~23 h. STILL O(n_slots) today:
        the round-path model_root in _run_async (its FATAL-DRIFT check) and the async no-progress
        guard, the coordinator's own model_root/root_hist bookkeeping, replica_root_ok's global
        fallback for a record with no slot_roots, and the `range(len(host.slots))` snapshot loops in
        _fold_accepted_checked and resume_to_root. Retiring those is separate work; what slot_root
        removes today is the per-coordinate LINEAGE coupling, not the hashing bill.
      * A node only hashes a coordinate it actually holds, so a wide claimable set no longer requires
        whole-model residency, and two nodes whose shard manifests were cut at different --shard-gb
        (piece id -> coordinate differs, and nothing cross-verifies it) no longer disagree on the
        root of a coordinate they both hold."""
    h = hashlib.sha256()
    _campaign_seed_into(h, host_campaign_id(host))    # campaign scope: see _campaign_seed_into
    _slot_digest_into(h, host, idx)
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


def _resolve_accepted_slot(host, item, log=None):
    """Map one accepted-record row onto THIS node's slot index, or None if we cannot host it.

    Coordinate first (`layer`/`glm_expert`), raw `slot` index second. The fallback is only correct for
    pre-Shard-Claim records, where every node derived its slot list from the same `--slots` string and
    the indices therefore matched; a coordinate-bearing record must NEVER be applied by raw index."""
    if item.get("layer") is not None and item.get("glm_expert") is not None:
        try:
            coord = (int(item["layer"]), int(item["glm_expert"]))
        except (TypeError, ValueError):
            if log:
                log("[glm-node] SKIP accepted delta: non-integer coordinate %r" % (item.get("layer"),))
            return None
        idx = host.index_of(*coord)
        if idx is None:
            # Normal on a shard-claim network: another miner's coordinate that this node does not hold.
            # Nothing to fold -- our copy of that expert is not resident, so there is nothing to diverge.
            if log:
                log("[glm-node] skip accepted delta for (L%d,E%d): not resident here" % coord)
            return None
        return idx
    try:
        idx = int(item["slot"])
    except (KeyError, TypeError, ValueError):
        return None
    return idx if 0 <= idx < len(host.slots) else None


def accepted_names_me(record, miner):
    """Did the coordinator ACCEPT this miner's delta in `record`? The verdict the miner could not see.

    Before shard claim a miner had no way to distinguish "my delta lost the gate" from "another miner
    won my slot" from "the accepted record never arrived": apply_accepted matched only on the slot and
    ignored `miner` entirely. The accepted rows have always carried `miner` (the coordinator stamps it),
    so the signal existed -- nothing read it. The plateau rule (K consecutive rejects -> release the
    coordinate and claim the next) is built on this."""
    return any(str(it.get("miner")) == str(miner) for it in (record.get("accepted") or []))


def apply_accepted(host, lane, record, log=None, ce_fn=None, tol=None, rejected=None, own_slot=None,
                   skipped=None, folded_slots=None):
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
    Returns the count of deltas actually FOLDED (a rejected delta is not counted).

    F2 (SHARD CLAIM, 2026-07-25) -- `skipped`: rows this node CANNOT PLACE (a coordinate it does not hold,
    or an unusable coordinate/slot field) are appended here, NEVER to `rejected`. On a shard-claim network
    "not resident here" is the NORMAL case -- miners deliberately hold different coordinates -- and nothing
    can diverge in an expert we do not have. Routing it through `rejected` made _fold_accepted_checked
    classify it as `poison`, catch_up_accepted return abort 8, and _run_async exit rc8 "forged/poisoned
    record" on the FIRST accepted record for another miner's coordinate; the same call from the
    coordinator's _resume_from_lane stopped the resume replay at the first dynamically-registered
    coordinate and rolled the campaign back to the frozen base. `rejected` is now strictly the local
    re-gate + shape-guard channel (the two signals that really do mean "do not trust this record").
    `skipped` None -> the rows are simply skipped, as before.

    F3 -- `folded_slots`: if given (a set), every slot index actually FOLDED by this call is added to it, so
    replica_root_ok can require that each one was ADVERTISED in `slot_roots` and verified. Without that,
    a forged record could move a resident coordinate while advertising only a non-resident (or resident but
    untouched) one and be accepted with no verification at all."""
    n = 0
    own = None if own_slot is None else int(own_slot)
    for item in record.get("accepted", []):
        # SHARD CLAIM: resolve by COORDINATE when the record carries one. The `slot` field is the
        # COORDINATOR's registry index; once miners address work by (layer, expert) that index is not
        # ours, so folding by it would apply the delta to a different expert -- or IndexError. Fall back
        # to the raw index for pre-Shard-Claim records, where the two indices did coincide.
        slot = _resolve_accepted_slot(host, item, log=log)
        if slot is None:
            # F2: NOT a rejection -- there is nothing here to reject. We cannot place this row (another
            # miner's coordinate, or an unusable coordinate/slot field), so no weight of ours moves and
            # nothing of ours can diverge. Report it on the `skipped` channel; `rejected` stays reserved
            # for the re-gate and shape failures that actually mean "distrust this record".
            if skipped is not None:
                skipped.append(dict(item, reason="unknown-coordinate"))
            continue
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
        if folded_slots is not None:
            folded_slots.add(int(slot))            # F3: what replica_root_ok must find advertised
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


# ==================================================================== alpha 3.0 periodic corpus resync
# glm_data_autosync (W6) runs ONCE at startup. Alpha 3.0 Objective 2 lets a RUNNING miner pick up a
# freshly-published corpus WITHOUT a restart: behind NEURAHASH_GLM_DATA_RESYNC (default OFF -> the
# whole block is unreachable and the lane is byte-identical to alpha 2.0), the async loop re-reads the
# advertised data record at a round boundary and, ONLY when the manifest sha changed, re-runs the SAME
# fail-closed fetch+verify (glm_data_autosync -- not duplicated). "Fail-closed" here means KEEP the old
# verified corpus on any unverifiable update: a running miner never trains on garbage and never dies
# because tomorrow's seed was briefly unreachable. The change-detection is a PURE function so the
# common no-change case is one dict compare (zero fetch) and is unit-tested with no I/O.
_RESYNC_OPT_IN = ("1", "true", "yes", "on")


def _data_resync_enabled(env=None):
    """THE guard for the alpha-3.0 periodic re-sync (acceptance #4). Returns True iff
    NEURAHASH_GLM_DATA_RESYNC is an explicit opt-in (1/true/yes/on); unset or anything else -> False,
    and _run_async then never reaches the re-check (flag-off == alpha-2.0 byte-identical)."""
    # DEFAULT ON since 2026-07-25 (owner directive). A miner that never picks up the daily corpus
    # trains forever on stale data -- the opposite of what an open campaign needs. The re-sync is
    # fail-closed (an unverifiable manifest keeps the old corpus) and was proven live on both
    # miners mid-run with no restart. Opt out with NEURAHASH_GLM_DATA_RESYNC=0.
    e = os.environ if env is None else env
    return (e.get("NEURAHASH_GLM_DATA_RESYNC", "1") or "1").strip().lower() \
        not in ("0", "false", "off", "no")


def plan_data_resync(prev_record, new_record):
    """PURE decision (no I/O) the periodic re-sync makes at every round boundary, so the common
    "nothing changed" case costs one dict compare and NEVER fetches. Given the record the miner last
    verified against (prev_record) and the one now advertised (new_record; None if absent/unreadable),
    return (changed, old_sha, new_sha, changed_files):
      * changed       -- True iff new_record is a dict carrying a manifest_sha256 that DIFFERS from
                         prev_record's. A missing/None/sha-less new_record -> False (never treat a
                         vanished or malformed record as a reason to disturb the running corpus).
      * changed_files -- sorted names whose files[name]['sha256'] differs between the two records
                         (added, removed, or re-hashed) -- the N in "re-fetched N file(s)". Only
                         computed once the sha gate has already fired.
    manifest_sha256 is the sha of the exact manifest bytes (glm_publish_data.py:178), so it flips iff
    any published file changed -- the cheapest possible change signal."""
    new_sha = new_record.get("manifest_sha256") if isinstance(new_record, dict) else None
    old_sha = prev_record.get("manifest_sha256") if isinstance(prev_record, dict) else None
    if not new_sha or new_sha == old_sha:
        return False, old_sha, new_sha, []
    old_files = (prev_record.get("files") or {}) if isinstance(prev_record, dict) else {}
    new_files = new_record.get("files") or {}

    def _sha_of(files, name):
        v = files.get(name)
        return v.get("sha256") if isinstance(v, dict) else None

    changed_files = sorted(n for n in (set(old_files) | set(new_files))
                           if _sha_of(old_files, n) != _sha_of(new_files, n))
    return True, old_sha, new_sha, changed_files


def glm_data_periodic_resync(lane, data_dir, prev_record, log=print, http_get=data_http_get, env=None):
    """Alpha-3.0 periodic corpus re-sync STEP (Objective 2), safe to call at every async round
    boundary. Returns (record_now, refreshed):
      * flag OFF -> (prev_record, False), ZERO I/O. (_run_async's guard already skips it; the internal
        check makes the function independently safe + testable.)
      * flag ON, record unchanged -> (prev_record, False): one lane read + a dict compare, NO fetch.
      * flag ON, manifest sha changed -> re-run glm_data_autosync (the SAME fail-closed fetch+verify,
        reused not duplicated). SUCCESS -> log "corpus resync: manifest <old>..-><new>.. re-fetched N
        file(s)" and return (new_record, True) so the caller reloads its ids + advances its baseline.
        FAIL-CLOSED: if the new corpus cannot be verified (autosync raises SystemExit), the old
        verified files on disk are left untouched -- log the refusal and return (prev_record, False)
        so the miner keeps training on the KNOWN-GOOD corpus and retries next boundary.
    (glm_data_autosync's own F1 guard still protects the disk if the new record names a disallowed
    file -- it fetches nothing in that case, so no unverified/secret split can ever land.)"""
    if not _data_resync_enabled(env):
        return prev_record, False
    new_record = _read_data_record(lane)
    changed, old_sha, new_sha, changed_files = plan_data_resync(prev_record, new_record)
    if not changed:
        return prev_record, False
    try:
        glm_data_autosync(lane, data_dir, log=log, http_get=http_get)
    except SystemExit as e:
        log("[glm-contrib] corpus resync REFUSED: new manifest %s..->%s.. failed fail-closed verify "
            "(rc%s) -- keeping the current verified corpus, will retry next round"
            % (str(old_sha)[:8], str(new_sha)[:8], getattr(e, "code", "?")))
        return prev_record, False
    log("[glm-contrib] corpus resync: manifest %s..->%s.. re-fetched %d file(s)"
        % (str(old_sha)[:8], str(new_sha)[:8], len(changed_files)))
    return new_record, True


# ============================================================================== contributor CLI
def _resolve_key(args):
    """The keyed (operator-issued) HMAC signing key, or None. --key / --key-file / NEURAHASH_SD_KEY -> a
    16-byte HMAC key (the roster identity, byte-identical to before). NONE set -> None, and the caller falls
    back to the LOCAL wallet identity (keyless open admission). No longer raises: keyless is now the default."""
    if args.key:
        return bytes.fromhex(args.key)
    if args.key_file and os.path.exists(args.key_file):
        return bytes.fromhex(open(args.key_file, "r", encoding="utf-8").read().strip())
    env = os.environ.get("NEURAHASH_SD_KEY")
    if env:
        return bytes.fromhex(env)
    return None


def derive_glm_miner_name(address):
    """Keyless miner id derived from a wallet address: 'glm-' + the first 8 hex chars (0x stripped). 32 bits
    of address -> collision-free enough for a first-seen name pin, and needs NO operator-issued name. The
    coordinator derives the SAME name from the RECOVERED signer address, so a keyless miner can only ever use
    the name its own key produces -- that binding is open admission's no-spoofing property."""
    a = str(address)
    if a[:2] in ("0x", "0X"):
        a = a[2:]
    return "glm-" + a[:8]


def _default_wallet_path(args):
    """Where the keyless wallet identity lives: --wallet-file / NEURAHASH_SD_WALLET / ~/.neurahash/glm_miner_key
    (durable, so the identity -- and its payout address -- is STABLE across restarts)."""
    p = (getattr(args, "wallet_file", None) or os.environ.get("NEURAHASH_SD_WALLET", "") or "").strip()
    return p or os.path.join(os.path.expanduser("~"), ".neurahash", "glm_miner_key")


def _load_or_create_wallet(args, log=None):
    """Load (or CREATE on first run) this miner's LOCAL secp256k1 wallet identity -- the keyless-admission
    identity AND payout address. Same on-disk format as a pool worker key (a bare private-key hex), so the key
    created here doubles as a wallet key. Mirrors tools/diloco_contributor._miner_account, but ALWAYS returns
    an account (a stranger must be able to join with NO config), creating + persisting one when absent."""
    from neura_l1.signing import account_from_key, gen_account
    path = _default_wallet_path(args)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            acct = account_from_key(f.read().strip())
        if log:
            log("[glm-contrib] wallet identity loaded from %s -> %s" % (path, acct.address))
        return acct
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    acct = gen_account()
    with open(path, "w", encoding="utf-8") as f:
        f.write(acct.key.hex())
    try:
        os.chmod(path, 0o600)                                # best-effort secrecy (no-op semantics on Windows)
    except Exception:                                        # noqa: BLE001 -- never fail over a chmod
        pass
    if log:
        log("[glm-contrib] wallet identity CREATED at %s -> %s (keyless open admission)" % (path, acct.address))
    return acct


def _resolve_identity(args, log=None):
    """Resolve THIS miner's signing identity -> (key, wallet). Keyed: (16-byte HMAC key, None) when an
    operator key was supplied (byte-identical path). Keyless: (None, secp256k1 account) otherwise -- the
    LOCAL wallet signs a RECOVERABLE ECDSA record the coordinator open-admits by its derived name."""
    key = _resolve_key(args)
    if key is not None:
        return key, None
    return None, _load_or_create_wallet(args, log=log)


def _sign_contrib(key, wallet, cid, base_round, name):
    """Sign one contribution over the SAME canonical GLM message either role verifies against
    (dm.contrib_canonical_message, exactly as H.sign builds it): keyed -> HMAC-SHA256 (byte-identical to
    before); keyless -> secp256k1 sign_bytes over that identical message -- a RECOVERABLE signature the
    coordinator ecrecovers to the wallet address. Exactly one of (key, wallet) is set."""
    if key is not None:
        return H.sign(key, cid, base_round, name)
    from neura_l1.signing import sign_bytes
    msg = dm.contrib_canonical_message(cid, base_round, name, None, None)
    return sign_bytes(wallet, msg)


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


def _clamp_base_event(last_applied, frontier):
    """P3 (dead-run lineage fix): a contribution's base_event MUST NOT exceed the frontier the latched
    pointer advertises. Publishing base_event > pointer.event is exactly the 'future-base-event' the
    coordinator drops (measured live 2026-07-24: a dead run's leftovers pushed a contributor's frontier
    hundreds of events past a fresh coordinator -> 561+ drops, mints starved). Returns min(last_applied,
    frontier); frontier None -> unbounded (== last_applied). Pure."""
    b = int(last_applied)
    if frontier is None:
        return b
    f = int(frontier)
    return f if b > f else b


class _CowSlots:
    """COPY-ON-WRITE rollback shim around a lane host: snapshot a slot the FIRST time it is about to
    be written, never before.

    WHY (measured 2026-07-26). _fold_accepted_checked used to deep-copy EVERY resident slot up front
    so it could restore them if the fold turned out to be off-lineage. One GLM coordinate's canonical
    {gate,up,down} triple is 18,874,493 B, so at the 60-coordinate residency that became the default
    that is ~1.05 GiB copied PER FOLDED RECORD -- 12x the ~94 MB it cost when residency was 5 slots,
    and paid on every record of every catch-up replay. An accepted record moves exactly ONE
    coordinate (the coordinator stamps one `slot_roots` entry per event), and most records on a
    shard-claim network move a coordinate this node does not even hold, so the copy was 59/60 waste.

    EXACTNESS IS THE POINT, not the saving: this is the fail-closed guarantee that keeps un-gated,
    off-lineage weights out of the base. It holds by construction rather than by argument -- every
    mutation the fold can make goes through write_slot, we capture the pre-image before the first one
    lands, and a slot that was never written is already byte-identical to its pre-call state, so
    restoring it would be a no-op. Everything else is delegated, so the fold sees the real host.

    Deliberately duck-typed (`__getattr__`), like every other host consumer in this module: the
    coordinator's _resume_from_lane folds through the same function against its own host."""

    def __init__(self, host):
        self._host = host
        self._saved = {}                         # slot idx -> pre-write DEEP copy

    def __getattr__(self, name):
        return getattr(self._host, name)

    def read_slot(self, j):
        return self._host.read_slot(j)

    def write_slot(self, j, d):
        j = int(j)
        if j not in self._saved:
            # DEEP copy: read_slot returns numpy VIEWS over the model's storage on a float32 CPU
            # model (sharddiloco_glm_expert.GlmExpertLaneHost.read_slot), so the write below would
            # otherwise mutate the snapshot too and make the rollback a silent no-op.
            self._saved[j] = {k: np.array(v, copy=True)
                              for k, v in self._host.read_slot(j).items()}
        return self._host.write_slot(j, d)

    @property
    def touched(self):
        """Slot indices this fold snapshotted, i.e. the ones a rollback has to restore."""
        return sorted(self._saved)

    def rollback(self):
        """Restore every written slot to its pre-write bytes. Slots never written are untouched."""
        for j, d in self._saved.items():
            self._host.write_slot(j, d)
        return len(self._saved)


def _fold_accepted_checked(host, lane, rec, regate_ce, own_slot, log=None):
    """Fold ONE coordinator accepted record, then VERIFY replica bit-exactness: our local model_root MUST
    equal the record's advertised model_root. That invariant is the whole reason model_root exists (see its
    docstring) -- a divergent local replay is caught IMMEDIATELY. Two fail-CLOSED outcomes stop a
    never-deleting store's dead-run leftovers (measured live 2026-07-24) from poisoning our base:
      * own-slot re-gate tripped     -> (False, 'poison', rejected): caller aborts rc8 (severity unchanged).
      * post-fold root != advertised -> (False, 'lineage', []): the record is NOT on our latched lineage;
        EVERY slot is restored exactly (rollback), so the base is byte-identical to before this call.
        F3: "advertised" now also means COVERED -- every coordinate this fold actually moved has to have
        its own `slot_roots` entry, and that entry has to validate. A record that moves a coordinate it
        never advertised is off-lineage by construction and rolls back here.
    A row this node cannot place (another miner's coordinate) is NOT poison -- see apply_accepted's
    `skipped` channel (F2). It is counted nowhere and changes nothing, which is the whole point.
    Clean fold -> (True, 'ok', []). Mirrors the coordinator's _lineage_ok wrong-lineage-root drop,
    node-side. A record without an advertised model_root (never emitted by the async coordinator) is folded
    as before (no false reject).

    The rollback snapshot is COPY-ON-WRITE (_CowSlots, 2026-07-26): a slot is deep-copied the first
    time the fold is about to write it, instead of copying all of them up front. Byte-for-byte the
    same restore -- a slot that was never written is already its own pre-call state -- at O(slots
    this record moves) instead of O(residency), which at 60 resident coordinates was ~1.05 GiB copied
    per folded record."""
    cow = _CowSlots(host)
    rejected, skipped, folded = [], [], set()
    try:
        apply_accepted(cow, lane, rec, log=log, ce_fn=regate_ce, rejected=rejected,
                       own_slot=own_slot, skipped=skipped, folded_slots=folded)
    except BaseException:
        # FAIL-CLOSED on a MID-FOLD failure. apply_accepted folds the accepted rows one at a time, so
        # anything raising part-way (a lane fetch erroring, or -- since never-block V0 -- a fetch
        # exceeding its deadline) used to leave the rows applied so far in the base with no rollback
        # at all. Restore first, then re-raise: the caller decides what the failure means, but it
        # never inherits a half-folded, unverified base.
        cow.rollback()
        raise
    if rejected:
        return False, "poison", rejected                          # caller handles rc8 (do not advance)
    if not replica_root_ok(host, rec, folded=folded):
        cow.rollback()                                            # UNFOLD everything: off-lineage, fail-closed
        return False, "lineage", []
    return True, "ok", []


def replica_root_ok(host, rec, folded=None):
    """Did our local replay reproduce what the coordinator advertised? True also when there is nothing
    to check (a record with neither root, which the async coordinator never emits).

    SHARD CLAIM: prefer the PER-COORDINATE roots. The global `model_root` is a hash over the whole slot
    list, so a replica can only ever match it if it holds the coordinator's EXACT slot set in the same
    order. On a shard-claim network miners deliberately hold different coordinates, so the global check
    would fail for every one of them and, being fail-closed, would freeze every replica's frontier
    permanently. Checking only the coordinates we actually hold keeps the invariant that matters -- our
    copy of expert X is bit-identical to the coordinator's -- without asserting anything about experts we
    were never given.

    F3 (SECURITY, 2026-07-25) -- `folded`: the slot indices this fold ACTUALLY MOVED (apply_accepted's
    folded_slots). Every one of them MUST have an advertised `slot_roots` entry, and that entry must
    validate; a folded coordinate with no entry FAILS CLOSED. Checking only the intersection of
    "advertised" and "resident" was bypassable with PUBLIC data on an UNSIGNED lane: a forged accepted
    record whose delta targets a resident but NOT-own coordinate, advertising either a non-resident
    coordinate (checked == 0 -> True) or a resident-but-untouched one (its already-published root still
    matches), folded with no verification whatsoever -- measured 0.0 -> 9000.0 weights on a 9e3-magnitude
    delta. The own-slot re-gate does not cover it (the target is not own_slot) and the lane's PUT token is
    a shared PUBLIC demo token, so the attack is unauthenticated. `folded` None -> the old behaviour
    (advertised-and-resident only), for callers that do not track it."""
    sr = rec.get("slot_roots")
    if isinstance(sr, dict) and sr:
        advertised = {}                                    # resident slot idx -> advertised root
        for key, want in sr.items():
            L, _, E = str(key).partition("_")
            try:
                idx = host.index_of(int(L), int(E))
            except ValueError:
                continue                                   # unparseable key: not ours to judge
            if idx is None:
                continue                                   # coordinate not resident here -> nothing to verify
            advertised[int(idx)] = want
        for idx, want in advertised.items():
            if slot_root(host, idx) != str(want):
                return False
        for idx in (folded or ()):
            if int(idx) not in advertised:
                return False                               # F3: moved but never advertised -> fail closed
        # Either every advertised coordinate we hold verified, or none of the advertised coordinates are
        # ours -- and in the latter case (F3) nothing of ours was folded either, so there is nothing to
        # verify. Falling through to the global root here would reintroduce the freeze this function
        # exists to prevent.
        return True
    want = rec.get("model_root")
    return want is None or model_root(host) == str(want)


def global_root_comparable(host, dec):
    """SHARD CLAIM: is the coordinator's GLOBAL model_root even COMPARABLE to ours?

    Only when it hashes the SAME slot set. `model_root` digests every coordinate in `host.slots`, so
    two nodes that deliberately hold different coordinates -- the entire premise of shard claim --
    can never produce the same digest no matter how bit-identical their shared experts are. Comparing
    it anyway reads as "base MISMATCH", drags a healthy miner into a resume replay it can never
    finish, and (being fail-closed downstream) ends in a rollback + a permanent lineage stall.

    The pointer hands us the coordinator's slot set for free: the v2 `rounds` map is keyed "L_E" for
    every slot it has active (sharddiloco_glm_coordinator._slot_key / _build_pointer). Equal key sets
    -> the global root IS a meaningful comparison; anything else -> only the per-coordinate roots are
    (replica_root_ok). A pre-v2 pointer carries no map, so return True and behave EXACTLY as before.

    Accepts either the decoded pointer (dm.sd_pointer_decode -> `slot_rounds`) or a raw v2 pointer
    dict (`rounds`); both key names are read so a caller cannot silently disable the check by passing
    the other shape. Pure."""
    if not isinstance(dec, dict):
        return True
    rounds = dec.get("slot_rounds")
    if rounds is None:
        rounds = dec.get("rounds")
    if not rounds:
        return True                                  # pre-v2 / empty map: nothing to compare against
    theirs = {str(k) for k in rounds}
    ours = {"%d_%d" % (int(L), int(E)) for (L, E) in host.slots}
    return theirs == ours


def pointer_slot_count(dec):
    """How many coordinates the pointer says the coordinator has active (0 for a pre-v2 pointer).
    Split out so the startup diagnostic can name the number without re-deriving the key lookup."""
    if not isinstance(dec, dict):
        return 0
    rounds = dec.get("slot_rounds")
    if rounds is None:
        rounds = dec.get("rounds")
    return len(rounds or {})


# ============================================================ NEVER-BLOCK V0 (catch-up is bounded)
# docs/NEVER_BLOCK_HANDOVER.md 0-PRE + 7.1. MEASURED 2026-07-25: the 5090 plateaued on (L1,E50),
# claimed (L1,E0) and BLOCKED 23 MINUTES -- process alive, "Responding: True", GPU 9%, nothing in any
# log. scratchpad/wan_miner5090.log:113 (`PLATEAU ... CLAIM (L1,E0)`) is the last line that process
# ever wrote, and there is NO `post-advance catch-up` line after it, while the three earlier advances
# in the same run each printed one after trying only 7-17 records. So it was NOT accumulated
# per-record replay cost: it entered resume_to_root and never came out of ONE call.
#
# Three bounds, in the order they bind:
#   1. PER-CALL DEADLINE (_DeadlineLane) -- the actual root cause. ContentLane's `timeout` is a
#      SOCKET timeout (urllib), so a connection that keeps trickling bytes never trips it and
#      `r.read()` can block for as long as the peer keeps dribbling; and with retries=6 even a fully
#      dead socket costs ~184 s per call. A wall budget on the loop cannot bound a call that never
#      returns unless the call itself is interruptible, which urllib's is not -- hence the worker
#      thread with a join deadline.
#   2. WALL BUDGET + no-fold stall abort -- the backstop for "many calls, each individually fine".
#   3. COOLDOWN + advance (CoordCooldown, advance_claim) -- on abort the miner does not sit there.
CATCHUP_BUDGET_S = 180.0            # B_wall. See _catchup_budget_s for how this number was chosen.
CATCHUP_CALL_TIMEOUT_S = 90.0       # per lane call inside catch-up; also capped by remaining budget
CATCHUP_STALL_S = 30.0              # abort after this long with ZERO records folded
COORD_COOLDOWN_S = 900.0            # 15 min ...
COORD_COOLDOWN_EVENTS = 10          # ... or 10 pointer events, whichever is LATER
_CATCHUP_ABORTED = ("budget", "stall", "call-timeout")   # reasons that mean BLOCKED, not "unreachable"


def _env_num(name, default, cast, environ=None):
    """One NEURAHASH_* numeric knob, fail-soft: an unset/blank/garbage value keeps `default` rather
    than killing a miner over a typo in an env var."""
    raw = ((environ if environ is not None else os.environ).get(name) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _catchup_budget_s(environ=None):
    """B_wall: the whole catch-up must finish inside this, or it fail-closes and the coordinate is
    parked. NEURAHASH_SD_CATCHUP_BUDGET_S overrides.

    CALIBRATION (2026-07-26, and honest about what could not be measured). U5 asks for
    `max(120 s, 2 x p99)` over the distribution of SUCCESSFUL catch-up reach times. That p99 is
    UNMEASURABLE from the artifacts we have: none of the 8 run logs in scratchpad/wan_*.log carries a
    timestamp on any line, and no line reports an elapsed time for a resume -- n = 0 duration samples
    (the corpus holds exactly ONE successful `resume:` line, wan_miner5090_run4.log:92, "after 30
    record(s)", with no clock beside it). So the formula's FLOOR applies, 120 s, and it is raised to
    180 s on the one measured anchor that does exist: memory glm-lane-manifest-throughput-bound
    measured lane.manifest() at 23.79 s over an 11,051-object store, and the coordinator logs of
    these runs report 19,438-23,503 objects at start, which puts ONE manifest call at ~42-51 s. 180 s
    leaves >=3.5x headroom over that while still being 7.7x under the 23-minute incident. Every
    catch-up now LOGS its own elapsed time, so the p99 this could not be calibrated from will exist
    after the next run -- recalibrate then."""
    return max(0.0, _env_num("NEURAHASH_SD_CATCHUP_BUDGET_S", CATCHUP_BUDGET_S, float, environ))


def _catchup_call_timeout_s(environ=None):
    """Per-lane-call deadline inside catch-up (NEURAHASH_SD_CATCHUP_CALL_TIMEOUT_S). Default 90 s =
    ~1.8x the ~42-51 s a single manifest() costs at today's store size, so it bounds a hung call
    without aborting a healthy slow one."""
    return max(0.0, _env_num("NEURAHASH_SD_CATCHUP_CALL_TIMEOUT_S", CATCHUP_CALL_TIMEOUT_S, float,
                             environ))


def _catchup_stall_s(environ=None):
    """Abort a catch-up that has folded NOTHING for this long (NEURAHASH_SD_CATCHUP_STALL_S). Timed
    from the start of the RECORD LOOP, not from the call, so the one legitimately slow manifest()
    cannot trip it."""
    return max(0.0, _env_num("NEURAHASH_SD_CATCHUP_STALL_S", CATCHUP_STALL_S, float, environ))


class CatchupTimeout(Exception):
    """A lane call inside catch-up did not return within its deadline. Distinct from the transient
    fetch failures the replay already tolerates, because it means BLOCKED, not "try again"."""


class _DeadlineLane:
    """Wrap a lane so EVERY call through it is bounded by a wall-clock deadline.

    Why a thread and not a timeout argument: the calls on this path are urllib GETs inside
    sharddiloco_harness.ContentLane, whose `timeout` is per-SOCKET-OPERATION. A peer that keeps
    sending a byte now and then never trips it, so `r.read()` on the ~20 MB manifest of a
    never-deleting store can block indefinitely -- and the fold path reaches the lane through
    apply_accepted's `lane.get_delta(cid)` too, not only get_json/manifest, so bounding one named
    method is not enough. Wrapping the OBJECT covers every fetch the fold path can reach.

    The abandoned worker is a daemon, so it can never hold up process exit; and when the wrapped lane
    exposes urllib-style knobs we hand the worker a SHALLOW CLONE with a tightened socket timeout and
    at most 2 retries, so an abandoned call dies on its own instead of lingering on a trickling
    socket. The clone is why the live lane's own timeout/retries are never mutated -- other threads
    (the main loop's manifest scan) share that object."""

    def __init__(self, lane, call_timeout_s, deadline=None, now=None):
        self._now = now or time.monotonic
        self._call_timeout = float(call_timeout_s)
        self._deadline = deadline
        self._lane = self._tighten(lane, self._call_timeout)
        self.abandoned = 0                        # calls we walked away from (each leaked one thread)

    @staticmethod
    def _tighten(lane, call_timeout_s):
        """A clone whose socket timeout cannot outlive our deadline by more than one retry. Returns
        the lane UNCHANGED when it exposes no such knobs (every in-process test fake)."""
        if not hasattr(lane, "timeout"):
            return lane
        try:
            clone = copy.copy(lane)
            clone.timeout = min(float(getattr(lane, "timeout", call_timeout_s) or call_timeout_s),
                                max(1.0, call_timeout_s))
            if hasattr(clone, "retries"):
                clone.retries = max(1, min(int(getattr(lane, "retries", 1) or 1), 2))
            return clone
        except Exception:                                        # noqa: BLE001 -- never fail to bound
            return lane

    def remaining(self):
        """Seconds left on the whole-catch-up deadline (None = no deadline set)."""
        return None if self._deadline is None else (self._deadline - self._now())

    def __getattr__(self, name):
        # Never proxy our OWN internals. Without this, any access to `_lane` before __init__ finished
        # (or a dunder probe from copy/pickle) recurses into __getattr__ forever.
        if name.startswith("_"):
            raise AttributeError(name)
        target = getattr(self._lane, name)
        if not callable(target):
            return target

        def _bounded(*a, **kw):
            budget = self._call_timeout
            left = self.remaining()
            if left is not None:
                budget = min(budget, left)
            if budget <= 0:
                raise CatchupTimeout("no time left in the catch-up budget before lane.%s" % name)
            box = {}

            def _run():
                try:
                    box["v"] = target(*a, **kw)
                except BaseException as ex:                      # noqa: BLE001 -- re-raised below
                    box["e"] = ex
            th = threading.Thread(target=_run, name="nh-catchup-%s" % name, daemon=True)
            th.start()
            th.join(budget)
            if th.is_alive():
                self.abandoned += 1
                raise CatchupTimeout("lane.%s did not return within %.1fs" % (name, budget))
            if "e" in box:
                raise box["e"]
            return box.get("v")
        return _bounded


class CoordCooldown:
    """Coordinates parked after a catch-up abort or a refused registration, so the claim walk skips
    them instead of retrying the same wall every advance.

    "15 minutes OR 10 events, whichever is LATER" (design 1.4): a quiet lane must not expire a
    cooldown just because no events happened, and a fast lane must not expire it just because 15
    minutes of wall clock passed while the coordinator raced ahead -- so BOTH have to elapse. Pure
    except for the injected clock, which is what makes it testable without sleeping."""

    def __init__(self, seconds=None, events=None, now=None):
        self._now = now or time.monotonic
        self.seconds = COORD_COOLDOWN_S if seconds is None else float(seconds)
        self.events = COORD_COOLDOWN_EVENTS if events is None else int(events)
        self._parked = {}                        # coord -> dict(reason, until_t, until_event)

    @staticmethod
    def _key(coord):
        return (int(coord[0]), int(coord[1]))

    def park(self, coord, event, reason):
        self._parked[self._key(coord)] = dict(
            reason=str(reason), until_t=self._now() + self.seconds,
            until_event=int(event or 0) + self.events)
        return self._parked[self._key(coord)]

    def blocked(self, coord, event):
        p = self._parked.get(self._key(coord))
        if p is None:
            return False
        if self._now() >= p["until_t"] and int(event or 0) >= p["until_event"]:
            del self._parked[self._key(coord)]           # both halves elapsed -> claimable again
            return False
        return True

    def reason(self, coord):
        p = self._parked.get(self._key(coord))
        return None if p is None else p["reason"]

    def left(self, coord, event):
        """(seconds_left, events_left) for a parked coordinate; (0.0, 0) if it is not parked."""
        p = self._parked.get(self._key(coord))
        if p is None:
            return 0.0, 0
        return max(0.0, p["until_t"] - self._now()), max(0, p["until_event"] - int(event or 0))

    def describe(self, coords, event):
        """One ASCII line per still-parked coordinate, for the repair-mode heartbeat."""
        out = []
        for c in coords:
            if self.blocked(c, event):
                s, e = self.left(c, event)
                out.append("(L%d,E%d) %s [%.0fs / %d event(s) left]"
                           % (int(c[0]), int(c[1]), self.reason(c), s, e))
        return out


def resume_to_root(host, lane, target_root, log, max_records=200000, own_coord=None,
                   budget_s=None, call_timeout_s=None, stall_s=None, now=None, outcome=None):
    """CONTRIBUTOR-SIDE RESUME SYMMETRY: replay accepted records until our base reproduces the
    coordinator's advertised genesis root, so a RESUMED coordinator can accept our work.

    WHY (measured live 2026-07-25): the coordinator gained --resume, which replays a previous run's
    accepted records so a restart continues the campaign (held-out CE 10.40 from frozen base vs
    8.64 resumed). But contributors rebuild from the FROZEN base and cannot reach a root produced
    by records the new run has not published -- so EVERY contribution was dropped
    wrong-lineage-root (1527 of them), one miner died 'unreconstructable', and the network accepted
    ZERO work. Resuming one side only is strictly worse than not resuming. This is the other side.

    ROOT-TARGETED, not event-bounded: the resumed coordinator republishes genesis at event 0, so
    there is no event window to catch up on -- we fold the SAME historical records it folded, in
    order, through the SAME verified fold (_fold_accepted_checked, which rolls back anything that
    does not reproduce its own advertised root), and STOP the moment our root equals the target.

    FAIL-CLOSED: if the target is unreachable, EVERY slot is restored to the pre-replay snapshot,
    so we are byte-identical to the frozen base and the caller may proceed (its contributions will
    be lineage-dropped, which is the honest signal) rather than training on a half-folded base.
    Returns (n_applied, reached).

    SHARD CLAIM (`own_coord=(L, E)`): target OUR OWN COORDINATE instead of the global root. The
    global root is a hash over the coordinator's whole slot list, so on a shard-claim network -- where
    the coordinator registers coordinates we do not hold -- it is a root no replay of ours can ever
    reproduce, and this loop would fold all 1345 records and then roll everything back. With
    own_coord the stop condition is the one the lineage guard actually checks (coordinator
    _lineage_ok, base_slot_root): the most recent record that ADVERTISED a `slot_roots` entry for our
    coordinate has been folded, and our local slot_root equals that advertised value. `own_coord=None`
    keeps the global behaviour byte-identical.

    NEVER-BLOCK V0 (2026-07-26, docs/NEVER_BLOCK_HANDOVER.md 0-PRE). Three bounds, all fail-CLOSED
    through the SAME rollback this function already had -- no new rollback semantics:
      * `call_timeout_s`: every lane call goes through _DeadlineLane, so no single fetch can hang
        forever. This is the measured root cause of the 23-minute block.
      * `budget_s` (B_wall): the whole replay aborts once elapsed exceeds it.
      * `stall_s`: abort after this long with ZERO records folded (timed from the record loop, so a
        legitimately slow manifest cannot trip it).
    `outcome`, if a dict is passed, is filled in place with reason/elapsed_s/records/aborted -- the
    caller needs to tell "BLOCKED, park this coordinate" (reason in _CATCHUP_ABORTED) apart from the
    honest "unreachable" it has always tolerated. Return arity is unchanged: (n_applied, reached)."""
    t0 = (now or time.monotonic)()
    _now = now or time.monotonic
    budget = _catchup_budget_s() if budget_s is None else float(budget_s)
    stall = _catchup_stall_s() if stall_s is None else float(stall_s)
    call_to = _catchup_call_timeout_s() if call_timeout_s is None else float(call_timeout_s)

    def _done(reason, applied, reached, records=None):
        # `records` is what the replay actually TRIED (it can be non-zero while `applied` is 0: a
        # rollback returns 0 by long-standing convention). The caller's diagnostics want the former.
        if outcome is not None:
            outcome.update(reason=reason, elapsed_s=_now() - t0,
                           records=int(applied if records is None else records),
                           aborted=reason in _CATCHUP_ABORTED)
        return applied, reached
    if not target_root or model_root(host) == str(target_root):
        return _done("already-at-root", 0, True)
    target = str(target_root)
    coord_key, own_idx = None, None
    if own_coord is not None:
        coord_key = "%d_%d" % (int(own_coord[0]), int(own_coord[1]))
        own_idx = host.index_of(int(own_coord[0]), int(own_coord[1]))
        if own_idx is None:                                      # not resident -> no per-coordinate target
            log("[glm-contrib] resume: coordinate (L%d,E%d) is not registered locally -- staying on "
                "the frozen base" % (int(own_coord[0]), int(own_coord[1])))
            return _done("not-registered", 0, False)
    lane = _DeadlineLane(lane, call_to, deadline=t0 + budget, now=_now)
    try:
        man = lane.manifest()
    except CatchupTimeout as e:
        log("[glm-contrib] resume: BLOCKED reading the manifest after %.1fs (%s) -- staying on the "
            "frozen base" % (_now() - t0, e))
        return _done("call-timeout", 0, False)
    except Exception as e:                                       # noqa: BLE001
        log("[glm-contrib] resume: manifest unavailable (%r) -- staying on the frozen base" % (e,))
        return _done("manifest-unavailable", 0, False)
    prefix = ACCEPTED_NAME_FMT % 0
    prefix = prefix[:prefix.rfind("0")]
    events = sorted(int(n[len(prefix):]) for n in man
                    if str(n).startswith(prefix) and str(n)[len(prefix):].isdigit())
    if not events:
        log("[glm-contrib] resume: no accepted records to replay -- staying on the frozen base")
        return _done("no-records", 0, False)
    snap = [{k: np.array(v, copy=True) for k, v in host.read_slot(j).items()}
            for j in range(len(host.slots))]                     # DEEP copy: read_slot returns VIEWS
    applied = 0
    aborted = None                            # never-block: 'budget' | 'stall' | 'call-timeout'
    t_loop = last_fold_t = _now()             # the stall clock starts AFTER the manifest, not before
    coord_want, coord_hit = None, False       # newest advertised root for own_coord + did we reproduce it
    for e in events[:int(max_records)]:
        if _now() - t0 > budget:
            aborted = "budget"
            break
        if stall > 0 and (_now() - last_fold_t) > stall:
            aborted = "stall"
            break
        entry = man.get(accepted_name(e))
        if not entry:
            continue
        try:
            rec = lane.get_json(entry["sha256"])
        except CatchupTimeout:
            aborted = "call-timeout"
            break
        except Exception:                                        # noqa: BLE001
            break
        try:
            ok, _reason, _rej = _fold_accepted_checked(host, lane, rec, None, -1, log=None)
        except CatchupTimeout:                                   # a delta fetch INSIDE the fold hung
            aborted = "call-timeout"
            break
        if not ok:
            continue                                             # off-lineage record: already rolled back
        applied += 1
        last_fold_t = _now()
        if coord_key is None:
            if model_root(host) == target:
                log("[glm-contrib] resume: reached the coordinator's root %s.. after %d record(s) "
                    "in %.1fs" % (target[:12], applied, _now() - t0))
                return _done("ok", applied, True)
            continue
        # Per-coordinate target: only a FOLDED record that advertised our coordinate can move it, and
        # the LAST such record is the coordinator's current state for it -- so keep replaying (a later
        # record may advance our coordinate again) and judge the newest advertisement at the end.
        adv = (rec.get("slot_roots") or {}).get(coord_key)
        if adv is None:
            continue
        coord_want = str(adv)
        coord_hit = (slot_root(host, own_idx) == coord_want)
    if coord_key is not None and coord_hit and aborted is None:
        log("[glm-contrib] resume: reached the coordinator's root for coordinate %s (%s..) after %d "
            "record(s) in %.1fs; the global root is not comparable on a shard-claim network"
            % (coord_key, coord_want[:12], applied, _now() - t0))
        return _done("ok", applied, True)
    for j, d in enumerate(snap):                                 # UNREACHABLE -> full rollback
        host.write_slot(j, d)
    if aborted is not None:
        # NEVER-BLOCK: the SAME fail-closed rollback as an unreachable root, but the caller is told
        # this was a BOUND firing rather than an honest "the coordinator's base is not reproducible",
        # so it parks the coordinate and advances instead of training into a wall.
        log("[glm-contrib] resume: ABORTED catch-up for %s after %.1fs (%s, %d record(s) folded, "
            "budget=%.0fs stall=%.0fs call-timeout=%.0fs); rolled back to the frozen base"
            % (coord_key or ("root %s.." % target[:12]), _now() - t0, aborted, applied, budget,
               stall, call_to))
        return _done(aborted, 0, False, records=applied)
    if coord_key is not None:
        log("[glm-contrib] resume: could NOT reach the coordinator's root for coordinate %s (%s, %d "
            "record(s) tried, %.1fs); rolled back to the frozen base -- our contributions will be "
            "lineage-dropped until the coordinator's base for this coordinate is reachable"
            % (coord_key, "advertised %s.." % coord_want[:12] if coord_want else
               "no record advertised it", applied, _now() - t0))
        return _done("unreachable", 0, False, records=applied)
    log("[glm-contrib] resume: could NOT reach the advertised root %s.. (%d record(s) tried, %.1fs); "
        "rolled back to the frozen base -- our contributions will be lineage-dropped until the "
        "coordinator's base is reachable" % (target[:12], applied, _now() - t0))
    return _done("unreachable", 0, False, records=applied)


def advance_claim(host, lane, claim_coords, current, identity, ranked, pointer_root, event,
                  cooldown, log, miner, plateau_rejects=0, budget_s=None, call_timeout_s=None,
                  stall_s=None, now=None):
    """NEVER-BLOCK 1.4: walk this identity's claim order from `current` to the first coordinate the
    miner can actually START on, parking every one that BLOCKS it on the way.

    Blocking means exactly two things, and neither of them is "the coordinator's base is
    unreachable": (a) registration refused it (not hostable here, or no seat under
    --max-active-slots), or (b) its catch-up hit one of the V0 bounds -- per-call deadline, wall
    budget, or no-fold stall. An HONEST unreachable root is NOT blocking and never was: the fold
    rolled back to the frozen base and the miner trains anyway, its contributions lineage-dropped,
    which is the designed signal. Preserving that distinction is what keeps this change small.

    Termination: the candidate list is one pass over a finite per-identity permutation of the
    claimable set (claim_walk_order), current excluded, so the walk cannot cycle. Returns
    ((L, E), local_idx, records_folded, reached) or None when the whole pass is blocked -- the
    caller's 1.5 repair mode."""
    order = claim_walk_order(claim_coords, identity, ranked=ranked)
    cur = tuple(current)
    if len(order) <= 1:
        return None                              # nothing to advance TO (next_claim_coord agrees)
    start = (order.index(cur) + 1) if cur in order else 0
    cands = [c for c in (order[(start + k) % len(order)] for k in range(len(order))) if c != cur]
    first = True
    for cand in cands:
        if cooldown.blocked(cand, event):
            continue
        try:
            ni = host.register(*cand)
        except (ValueError, RuntimeError) as ex:
            cooldown.park(cand, event, "register refused (%s)" % (str(ex)[:60],))
            log("[glm-contrib %s] cannot advance to (L%d,E%d): %s -- COOLDOWN and walking on"
                % (miner, cand[0], cand[1], ex))
            continue
        if first:
            log("[glm-contrib %s] PLATEAU on (L%d,E%d) after %d consecutive rejects -> RELEASE, "
                "CLAIM (L%d,E%d) [local slot %d]. Sweeping: %d coordinate(s) claimable here."
                % (miner, cur[0], cur[1], int(plateau_rejects), cand[0], cand[1], ni,
                   len(claim_coords)))
            first = False
        else:
            log("[glm-contrib %s] walking past a blocked coordinate -> CLAIM (L%d,E%d) [local slot "
                "%d]" % (miner, cand[0], cand[1], ni))
        outcome = {}
        n_res, reached = resume_to_root(host, lane, pointer_root, log, own_coord=cand,
                                        outcome=outcome, budget_s=budget_s,
                                        call_timeout_s=call_timeout_s, stall_s=stall_s, now=now)
        if outcome.get("aborted"):
            cooldown.park(cand, event, "catch-up %s" % outcome.get("reason"))
            log("[glm-contrib %s] COOLDOWN (L%d,E%d): catch-up %s after %.1fs -- parked for %.0fs / "
                "%d event(s), advancing to the next claimable coordinate instead of blocking on it"
                % (miner, cand[0], cand[1], outcome.get("reason"), outcome.get("elapsed_s", 0.0),
                   cooldown.seconds, cooldown.events))
            continue
        log("[glm-contrib %s] post-advance catch-up for (L%d,E%d): folded %d record(s), coordinator "
            "root %s (%.1fs)"
            % (miner, cand[0], cand[1], n_res, "REACHED" if reached else "NOT reached (frozen base)",
               outcome.get("elapsed_s", 0.0)))
        return (cand[0], cand[1]), ni, n_res, reached
    return None


def catch_up_accepted(host, lane, man, last_applied, frontier, regate_ce, own_slot, miner, log,
                      folded=None):
    """The non-blocking accepted-record catch-up of _run_async, FACTORED OUT so the dead-run lineage guard
    is unit-testable against a dirty namespace. Fold every visible accepted record in (last_applied,
    frontier] IN ORDER via _fold_accepted_checked, which fail-CLOSES on an off-lineage record: the frontier
    NEVER advances past a record we could not validate, so a previous run's leftovers (still listed by the
    never-deleting store) can never become our training base. Returns (last_applied, applied_any, abort):
    abort is None normally, or 8 when an own-slot delta tripped the local re-gate (rc8, unchanged). Emits at
    most ONE aggregate LINEAGE-SKIP line (never one per record). Non-blocking: no visible record ->
    (unchanged, False, None) and the caller trains against the current base.

    NEVER-BLOCK V0: "non-blocking" was only true of the SCAN, not of the FETCHES inside it -- a
    get_json (or the get_delta apply_accepted issues per accepted row) that never returned blocked
    this loop exactly as it blocked resume_to_root. Every fetch here now carries the same per-call
    deadline; a call that trips it takes the pre-existing transient-failure exit (stop the scan, train
    against the current base, retry next tick), so the failure SEMANTICS are unchanged."""
    applied_any = False
    skipped_at = None
    lane = _DeadlineLane(lane, _catchup_call_timeout_s())
    for e in scan_accepted_events_bounded(man, last_applied, frontier):
        entry = man.get(accepted_name(e))
        if not entry:
            break                                            # gap: stop -- never fold out of order
        try:
            rec = lane.get_json(entry["sha256"])
        except CatchupTimeout as ex:
            log("[glm-contrib %s] catch-up: BLOCKED fetching accepted event %d (%s) -- training "
                "against the current base and retrying next tick" % (miner, e, ex))
            break
        except Exception:                                    # noqa: BLE001
            break                                            # transient fetch fail -> retry next tick
        try:
            ok, reason, rejected = _fold_accepted_checked(host, lane, rec, regate_ce, own_slot,
                                                          log=log)
        except CatchupTimeout as ex:                         # a delta fetch INSIDE the fold hung
            log("[glm-contrib %s] catch-up: BLOCKED fetching a delta of accepted event %d (%s) -- "
                "training against the current base and retrying next tick" % (miner, e, ex))
            break
        if reason == "poison":
            log("[glm-contrib %s] SECURITY: locally REJECTED %d accepted delta(s) at event %d "
                "(regressed local held-out CE or mismatched shape). The pointer + accepted record "
                "ride an UNSIGNED shared-token lane, so this looks like a forged/poisoned record -- "
                "refusing to fold it and aborting rather than training on a poisoned base."
                % (miner, len(rejected), e))
            return last_applied, applied_any, 8
        if reason == "lineage":
            skipped_at = e
            break                                            # frontier held; never fold past an off-lineage record
        last_applied = e
        applied_any = True
        if folded is not None:
            # SHARD CLAIM: hand the caller the records we actually folded, so it can read the
            # coordinator's VERDICT on its own contributions (accepted_names_me + record_touched_coord)
            # and decide whether its expert has plateaued. Appending rather than changing the return
            # arity keeps every existing caller and test working.
            folded.append(rec)
    if skipped_at is not None:
        log("[glm-contrib %s] LINEAGE-SKIP: accepted record at event %d does not extend our latched "
            "lineage (local model_root != its advertised root) -- a previous run's leftover in the "
            "never-deleting store; refusing to fold it, frontier held at event %d (fail-closed)."
            % (miner, skipped_at, last_applied))
    return last_applied, applied_any, None


def async_should_abort_no_progress(local_root, pointer_root, applied_any, seconds_since_progress,
                                   round_wait, comparable=True):
    """Async lane no-progress abort decision (alpha 2.0 #146, reuses rc6 semantics). Root mismatch is
    NORMAL mid-flight -- another slot advanced between our reads -- so a mismatch ALONE never aborts
    (this is the deliberate departure from the v1 sync rc7 drift-abort). We abort ONLY when the
    coordinator advertises a root we cannot reach AND we have folded no accepted record for
    `round_wait` seconds, i.e. the missing records are unreconstructable rather than merely late.
    Rules, in order:
      * comparable False     -> False  (F6: the two roots hash DIFFERENT slot sets -- see below).
      * applied_any True     -> False  (any progress this tick resets the timer -> never abort now).
      * falsy pointer_root   -> False  (coordinator advertises no root -> nothing to be stuck against).
      * local == pointer root -> False (fully caught up -> not stuck).
      * else                 -> True iff seconds_since_progress >= round_wait.
    Pure: no clock read (the caller passes the elapsed time), no I/O.

    F6 (SHARD CLAIM, 2026-07-25) -- `comparable` (caller passes global_root_comparable(host, dec)): this
    guard compares GLOBAL model_roots, and `model_root` digests the whole slot list. A shard-claim miner
    holds a different slot set from the coordinator BY CONSTRUCTION, so `local == pointer` is
    unsatisfiable for it and its ONLY protection against a false rc6 was folding some record within
    --round-wait (default 300 s) -- one quiet 5-minute window killed every miner on the network. When the
    roots are not comparable the mismatch carries no information at all, so the guard is skipped;
    per-coordinate roots (replica_root_ok / the lineage guard) are authoritative there. Default True keeps
    the pre-shard-claim behaviour byte-identical whenever the slot sets DO match."""
    if not comparable:
        return False
    if applied_any:
        return False
    if not pointer_root:
        return False
    if local_root == pointer_root:
        return False
    return float(seconds_since_progress) >= float(round_wait)


def build_async_contrib_record(miner, i, L, E, base_event, base_root, expert_cid, sig, train_flops,
                               delta_bytes, steps, tokens, address=None, base_slot_root=None):
    """Assemble the async-lane contribution record: today's signed record EXTENDED with the alpha-2
    telemetry the coordinator (W2) reads -- base_event (the event this delta was trained against; the
    r-number in the contrib name MEANS this), base_root (our local model_root), steps (inner steps
    executed) and tokens (rows*seq consumed). It is a strict SUPERSET of the sync record: base_round is
    kept == base_event so a v1-shaped reader still finds a base height, mirroring the v2 pointer's
    superset discipline. Pure dict assembly, no I/O.

    `address` (open admission, additive): a keyless miner stamps its CLAIMED wallet address here for
    transparency; the coordinator TRUSTS only the address recovered from `sig`, never this field. None
    (keyed / flag-off) -> the key is OMITTED, byte-identical to before.

    `base_slot_root` (SHARD CLAIM, additive): the root of THIS COORDINATE only (N.slot_root). A
    Shard-Claim coordinator judges lineage on it and ignores the global `base_root`, which is what lets
    it register and evict coordinates without dropping everyone mid-flight. `base_root` is still sent
    so a pre-Shard-Claim coordinator keeps working unchanged; None omits the new key entirely."""
    rec = dict(
        miner=miner, expert=int(i), layer=int(L), glm_expert=int(E),
        base_round=int(base_event),          # v1-compat alias: the r-number == base_event
        base_event=int(base_event),
        expert_cid=expert_cid, trunk_cid=None, sig=sig,
        train_flops=float(train_flops), trunk_bytes=0, delta_bytes=int(delta_bytes),
        base_root=base_root, steps=int(steps), tokens=int(tokens),
    )
    if address is not None:
        rec["address"] = str(address)
    if base_slot_root is not None:
        rec["base_slot_root"] = str(base_slot_root)
    return rec


def async_publish_name(base_event, miner, k, campaign=None):
    """F-Q1: the UNIQUE per-publish contribution name = contrib_name(base_event, miner) + a per-miner
    monotonic counter suffix '.<k>'. When a miner completes >=2 H-blocks against ONE base_event (the
    coordinator merge lagging), each publish lands on a DISTINCT manifest name instead of the 2nd atomically
    repointing (and silently losing) the 1st -- the exact lost-work bug F-Q1 closes. The signature covers
    ecid/base_event/miner, NOT the name (H.sign at the publish site), so the suffix is signature-safe, and
    the coordinator reads base_event/miner from the RECORD, not the name. Pure.

    `campaign` scopes the name to one campaign (cg/<campaign_id>/r<N>/<miner>.<k>); None keeps the legacy
    flat shape. See CAMPAIGN SCOPING at the top of this file."""
    return "%s.%d" % (contrib_name(base_event, miner, campaign), int(k))


# ==================================================================== v3.2.1 signed auto-update wire
# Why this exists: tools/self_update.py (the signed, pinned-key, fail-closed updater) was fully built
# but NOTHING in the GLM-only client ever called it -- the automatic startup/periodic checks lived in
# the deprecated legacy pool client, so a running GLM miner sat on an old release forever (found live
# 2026-07-24: the 4060 stayed on v3.1.0 after v3.2.0 was signed). This wire calls it at process
# startup and at the SAME safe between-rounds boundary as the corpus resync. Properties, all
# inherited from check_and_update(): FAIL-CLOSED (any error -> keep mining on current code),
# signature verified against the PINNED release key (no host can forge one, only withhold), forward-
# only (no rollback), 6h rate limit persisted in a dotfile stamped BEFORE the attempt (a broken
# release cannot re-exec loop), opt-out NEURAHASH_AUTOUPDATE=off. On a verified forward release it
# RE-EXECS this process with its original argv -- the lane treats that as normal miner churn.
def _maybe_self_update(log=_flush, _check=None):
    """One rate-limited signed-update check; never raises Exception into the mining loop (SystemExit
    from the updater's own re-exec/exit path is deliberately NOT swallowed -- swallowing it would
    cancel the update). Returns the UpdateResult, or None when the check itself failed."""
    try:
        if _check is None:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from self_update import check_and_update as _check
        r = _check()
        act = getattr(r, "action", None)
        if act not in ("rate-limited", "disabled", "no-op-not-forward", None):
            log("[auto-update] %s" % (r,))
        return r
    except Exception as e:                                       # noqa: BLE001 -- never kill mining
        try:
            log("[auto-update] WARN: check failed (%r); mining continues on current code" % (e,))
        except Exception:                                        # noqa: BLE001
            pass
        return None


def _run_async(args, lane, host, model, cfg, G, key, i, L, E, miner, train_ids, val_ids, seq, log,
               *, wallet=None, claim_ranked=None):
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
    manifest/pointer read failure -- there is no barrier sleep.

    `claim_ranked` (--claim-by affinity) is the ESFT affinity order from probe_expert_affinity; when
    given, the plateau advance walks it instead of the wallet-hash permutation, so releasing a
    plateaued expert lands on the NEXT-HIGHEST-affinity coordinate this node holds."""
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
    # SHARD CLAIM: claim-and-advance state. `reject_streak` counts CONSECUTIVE events where the
    # coordinator merged OUR coordinate and did not accept our delta (see accepted_names_me --
    # before this, a miner literally could not tell a rejection from a lost race or a missing record).
    reject_streak = 0
    # F5a: the base_event of our LAST publish. Until we have published something, no accepted record can
    # be a verdict on us (event_judged_us), so folding a running campaign's history never moves the streak.
    last_pub_base_event = None
    advance_after = int(getattr(args, "advance_after", 0) or 0)
    # NEVER-BLOCK V0: coordinates whose catch-up hit a bound (or whose registration was refused) are
    # parked here so the claim walk skips them, and `repair_since` is the 1.5(b) all-blocked state --
    # loud, idle, and explicitly NOT training on a base we know the coordinator will reject.
    cooldown = CoordCooldown(seconds=_env_num("NEURAHASH_SD_COORD_COOLDOWN_S", COORD_COOLDOWN_S,
                                              float),
                             events=_env_num("NEURAHASH_SD_COORD_COOLDOWN_EVENTS",
                                             COORD_COOLDOWN_EVENTS, int),
                             now=time.time)
    repair_since, last_repair_log = None, 0.0
    claim_coords = claim_all_coords(args, list(host.slots))
    # The sweep ORDER is per-identity (see next_claim_coord): a shared +1 advance turned a one-off
    # collision between two miners into a permanent one. Same durable identity pick_start_coord used.
    claim_identity = wallet.address if wallet is not None else (miner or "anonymous")
    log("[glm-contrib %s] shard claim: %d coordinate(s) claimable here, advance_after=%s, order=%s, "
        "walk=%s"
        % (miner, len(claim_coords),
           ("%d consecutive rejects" % advance_after) if advance_after else "OFF (never advance)",
           "ESFT affinity (measured)" if claim_ranked else "wallet-hash permutation",
           " -> ".join("%d:%d" % c for c in claim_walk_order(
               claim_coords, claim_identity, ranked=claim_ranked)[:6])
           + ("..." if len(claim_coords) > 6 else "")))
    log("[glm-contrib %s] ASYNC cadence (v2 lane, #146): non-blocking; train continuously, never wait "
        "on a barrier. --max-rounds=%d counts our own contributions." % (miner, args.max_rounds))
    # -- alpha 3.0 Objective 2: periodic corpus re-sync baseline. OFF unless NEURAHASH_GLM_DATA_RESYNC;
    # when off, _resync_on is False and NOTHING below (no lane read, no fetch, no reload) ever runs, so
    # this lane stays byte-identical to alpha 2.0. The seed record is the one the startup autosync just
    # verified against, so the first round's compare is a no-op (nothing changed yet).
    # -- RESUME SYMMETRY (2026-07-25): if the coordinator booted with --resume, its advertised
    # genesis root is NOT the frozen base and nothing we train can be accepted until we reach it.
    # Root-targeted replay, fail-closed (rolls back to the frozen base when unreachable). A normal
    # from-base coordinator advertises our own root, so this is a no-op single comparison there.
    # -- SHARD CLAIM (2026-07-25): the comparison below is only meaningful when the coordinator hashes
    # the SAME slot set we do. Once it can REGISTER a coordinate we do not hold, its global root is
    # unreachable by construction and this check would report "base MISMATCH" on every healthy miner,
    # then burn a full replay that must end in a rollback. global_root_comparable gates it.
    try:
        _ptr0 = lane.read_pointer()
        _dec0 = dm.sd_pointer_decode(_ptr0) if _ptr0 else None
        _root0 = _dec0.get("model_root") if _dec0 else None
    except Exception:                                            # noqa: BLE001 -- pointer races are normal
        _dec0, _root0 = None, None
    if _root0 and not global_root_comparable(host, _dec0):
        log("[glm-contrib %s] shard claim: skipping the global-root comparison (coordinator has %d "
            "active coordinate(s), we hold %d) -- per-coordinate roots are authoritative here"
            % (miner, pointer_slot_count(_dec0), len(host.slots)))
    elif _root0 and str(_root0) != model_root(host):
        log("[glm-contrib %s] base MISMATCH vs coordinator genesis (ours=%s.. theirs=%s..) -- "
            "attempting resume replay" % (miner, model_root(host)[:12], str(_root0)[:12]))
        resume_to_root(host, lane, _root0, log, own_coord=(L, E))

    _resync_on = _data_resync_enabled(os.environ)
    _prev_data_record = _read_data_record(lane) if _resync_on else None
    if _resync_on:
        log("[glm-contrib %s] periodic corpus resync ENABLED (NEURAHASH_GLM_DATA_RESYNC): re-checking "
            "the advertised data record at each round boundary; fail-closed keeps the old corpus" % miner)
    while rounds_done < args.max_rounds:
        # -- (0a) v3.2.1 signed auto-update: same SAFE between-rounds boundary as the resync below;
        # internally 6h rate-limited via dotfile, so this is one cheap file-stat on all but ~4
        # checks/day. A verified forward release re-execs us here -- never mid-train.
        _maybe_self_update(log)
        # -- (0) alpha 3.0 periodic corpus re-sync: a SAFE between-rounds boundary (never mid-train).
        # Flag-gated + zero I/O when the record is unchanged. On a VERIFIED new corpus, reload our ids so
        # the next train step uses the fresh data with NO restart; on an unverifiable one, keep the old.
        if _resync_on:
            _prev_data_record, _refreshed = glm_data_periodic_resync(
                lane, args.data_dir, _prev_data_record, log=log)
            if _refreshed:
                train_ids = node_ids(args, coord_data_slot(L, E), "train")
                val_ids = node_ids(args, coord_data_slot(L, E), "val")
                regate_ce = (lambda h: G.heldout_ce(h.model, val_ids)) if len(val_ids) else None
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
        # CAMPAIGN CHANGED: the lane is no longer the campaign we latched at boot -- the coordinator
        # restarted into a new one. Every record we publish from here would land under a prefix it
        # never scans, hashed against roots it never had, so we would train forever and be ignored with
        # NOTHING logged anywhere: an undiscovered record is never even dropped. Exit loudly; a restart
        # latches the current campaign at boot. (Both None on a legacy lane -> never fires.)
        _ptr_camp = pointer_campaign_id(ptr)
        if _ptr_camp != host_campaign_id(host):
            log("[glm-contrib %s] FATAL: campaign CHANGED on the lane (ours=%s, pointer now advertises "
                "%s) -- the coordinator restarted into a different campaign. Our contributions would be "
                "published where it never looks and hashed against roots it never had: silent "
                "starvation, so exiting instead. Restart to join %s."
                % (miner, host_campaign_id(host) or "none", _ptr_camp or "none",
                   _ptr_camp or "the current campaign"))
            return RC_NO_CAMPAIGN
        pointer_root = dec["model_root"]

        # -- (1) NON-BLOCKING catch-up: fold every accepted record past last_applied, IN ORDER, but ONLY
        #    those that extend OUR latched lineage (P2 dead-run guard: a never-deleting store still lists a
        #    previous run's accepted records at events THIS run has not reached; folding them poisons our
        #    base -> the future-base-event drop storm measured live 2026-07-24). catch_up_accepted verifies
        #    each fold reproduces the coordinator's advertised root and fail-CLOSES otherwise, holding the
        #    frontier. Non-blocking: nothing visible -> train against the current base rather than wait. ---
        try:
            man = lane.manifest()
        except Exception:                                        # noqa: BLE001
            time.sleep(args.poll)
            continue
        _folded = []
        last_applied, applied_any, _abort = catch_up_accepted(
            host, lane, man, last_applied, dec.get("event"), regate_ce, i, miner, log, folded=_folded)
        if _abort is not None:
            return _abort                                        # rc8: poisoned own-slot delta (unchanged)
        if applied_any:
            last_progress_t = time.time()

        # -- (1b) SHARD CLAIM: read the coordinator's verdict on OUR expert and advance when plateaued.
        # A record whose slot_roots names our coordinate means the coordinator MERGED our expert at that
        # event; if our miner id is not in its accepted rows, our delta lost the gate. K consecutive
        # losses = this expert has stopped yielding for us, so release it and claim the next one. This is
        # the owner's "finish one, store it, start the next": storing is a no-op because an accepted
        # delta is already merged into the model, so the model IS the store.
        for _rec in _folded:
            if not record_touched_coord(_rec, (L, E)):
                continue                                         # some other expert's event
            if not event_judged_us(_rec, last_pub_base_event):
                continue                                         # F5a: predates our work -> not a verdict
            if accepted_names_me(_rec, miner):
                reject_streak = 0
            else:
                reject_streak += 1
        if advance_after and reject_streak >= advance_after:
            # F5b: the freshly claimed coordinate's local weights are the FROZEN BASE. If anyone
            # already trained it, our base_slot_root can never match the coordinator's and EVERY
            # later contribution is dropped `wrong-lineage-slot-root` forever, silently --
            # catch_up_accepted only scans (last_applied, frontier], so the historical records that
            # moved this coordinate are never replayed. advance_claim brings it up to the
            # coordinator's state, targeted at THIS coordinate (the global root is not reachable on a
            # shard-claim network) and fail-closed. NEVER-BLOCK V0: that catch-up is now BOUNDED, and
            # a coordinate whose catch-up hits a bound is parked and walked past instead of blocking
            # the miner (measured 23-minute hang, docs/NEVER_BLOCK_HANDOVER.md 0-PRE).
            landed = advance_claim(host, lane, claim_coords, (L, E), claim_identity, claim_ranked,
                                   pointer_root, dec.get("event"), cooldown, log, miner,
                                   plateau_rejects=reject_streak)
            reject_streak = 0
            if landed is None:
                if len(claim_coords) <= 1:
                    log("[glm-contrib %s] plateaued on (L%d,E%d) after %d consecutive gate rejects, "
                        "but this node holds no other coordinate to claim -- staying put."
                        % (miner, L, E, advance_after))
                else:
                    # 1.5(b) LOUD REPAIR, chosen over the owner-literal "train anyway": an
                    # idle-but-loud miner costs 0 and is visible; a miner training a base it knows is
                    # off-lineage burns watts to manufacture rejects and hides the outage. That is
                    # this project's ~900-rounds-paid-for-nothing failure with better uptime.
                    repair_since = repair_since or time.time()
                    last_repair_log = 0.0
            else:
                (L, E), i, _n_res, _reached = landed
                repair_since = None
                # Re-read this coordinate's data shard and rebind the re-gate closure over the new
                # val split -- the same mid-loop reload the corpus-resync path already performs.
                train_ids = node_ids(args, coord_data_slot(L, E), "train")
                val_ids = node_ids(args, coord_data_slot(L, E), "val")
                regate_ce = (lambda h: G.heldout_ce(h.model, val_ids)) if len(val_ids) else None
                last_pub_base_event = None       # F5a: nothing of OURS is in flight on this coordinate
        elif repair_since is not None:
            # Cooldowns expire on wall clock AND events, so retry the walk every iteration: when
            # every coordinate is parked this costs zero lane calls (advance_claim skips them all
            # before it can register or fetch anything).
            landed = advance_claim(host, lane, claim_coords, (L, E), claim_identity, claim_ranked,
                                   pointer_root, dec.get("event"), cooldown, log, miner)
            if landed is not None:
                (L, E), i, _n_res, _reached = landed
                log("[glm-contrib %s] REPAIR CLEARED after %.0fs: resuming on (L%d,E%d)"
                    % (miner, time.time() - repair_since, L, E))
                repair_since = None
                train_ids = node_ids(args, coord_data_slot(L, E), "train")
                val_ids = node_ids(args, coord_data_slot(L, E), "val")
                regate_ce = (lambda h: G.heldout_ce(h.model, val_ids)) if len(val_ids) else None
                last_pub_base_event = None
        if repair_since is not None:
            # NO TRAINING while every claimable coordinate is blocked -- see 1.5(b) above. Say why,
            # per coordinate, every 30 s: a silent idle miner and a healthy one look identical.
            if time.time() - last_repair_log >= 30.0:
                last_repair_log = time.time()
                log("[glm-contrib %s] REPAIR MODE (%.0fs): every claimable coordinate is blocked, so "
                    "NOT training -- %s" % (miner, time.time() - repair_since,
                                            "; ".join(cooldown.describe(claim_coords,
                                                                        dec.get("event"))) or
                                            "no coordinate is claimable at all"))
            time.sleep(max(float(args.poll or 0.0), 1.0))
            continue

        # -- (2) root mismatch is NORMAL mid-flight; abort ONLY on prolonged no-progress (rc6). -------
        # F6: and ONLY when the two roots are even comparable. A shard-claim miner holds a different slot
        # set from the coordinator by construction, so local == pointer is unsatisfiable and this guard
        # would rc6 every healthy miner after one quiet --round-wait window.
        root = model_root(host)
        if async_should_abort_no_progress(root, pointer_root, applied_any,
                                          time.time() - last_progress_t, args.round_wait,
                                          comparable=global_root_comparable(host, dec)):
            log("[glm-contrib %s] FATAL: no accepted-record progress for %.0fs while local model_root="
                "%s.. cannot reach coordinator root=%s.. (missing records are unreconstructable, not "
                "merely late)." % (miner, args.round_wait, root[:12], str(pointer_root)[:12]))
            return 6

        # -- (3) train H local LoRA steps on my slot against the CURRENT base, ZERO cross-miner comm. --
        # VRAM-starve PAUSE + OOM round-skip (2026-07-24): never enter train/eval while the manager
        # advertises 0 sustainable units, and a CUDA OOM mid-round skips the round (cache freed,
        # pause, retry next round) instead of killing the miner.
        _vram_pause_if_starved(log, miner=miner)
        t_tr = time.time()
        try:
            if args.garbage:
                delta = G.garbage_delta(host.read_slot(i), scale=3.0, seed=1234 + rounds_done)
                train_flops, best_val, steps, tokens = 1.0, float("nan"), 0, 0
                payload = delta                    # adversarial control stays dense: it has no factors
            else:
                c = G.train_glm_expert_contribution(
                    model, cfg, L, E, train_ids, val_ids, H=args.inner, r=args.lora_r, lr=args.lr,
                    batch=args.batch, seed=rounds_done * 100 + i, sel_outer=args.outer)   # F5 select@gate
                delta, train_flops, best_val = c["delta"], c["train_flops"], c["best_val_ce"]
                # Steps ACTUALLY taken, not the H we asked for: a step whose batch routed no token to
                # our expert has no gradient and is skipped (sharddiloco_glm_expert.py). Publishing H
                # regardless would inflate token_quality_weight(steps, tokens) -- pay for work not done.
                steps = int(c.get("steps_trained", args.inner))
                tokens = int(c.get("n_examples", 0)) * int(seq)      # rows*seq actually consumed
                if int(c.get("steps_skipped", 0)):
                    log("[glm-contrib %s] (L%d,E%d): %d of %d inner step(s) routed NO token to this "
                        "expert and were skipped (no gradient exists for them); trained %d."
                        % (miner, L, E, int(c["steps_skipped"]), int(args.inner), steps))
                payload = c["lora"] if (args.wire == "lora" and c.get("lora")) else delta
        except Exception as _oom:                                    # noqa: BLE001
            if not _is_cuda_oom(_oom):
                raise
            log("[glm-contrib %s] round SKIPPED: CUDA OOM under memory pressure -- freeing cache + "
                "pausing, will retry next round (%s)" % (miner, str(_oom)[:120]))
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                                        # noqa: BLE001
                pass
            _vram_pause_if_starved(log, miner=miner)
            time.sleep(5.0)
            continue

        # -- (4) publish today's payload + signed record EXTENDED with base_event/base_root/steps/tokens.
        # P3: never publish a base_event beyond the frontier the latched pointer advertises (fail-closed
        # clamp; the coordinator drops base_event > cur_event as future-base-event). With the P2 guard in
        # place last_applied <= frontier already, so this is a defense-in-depth no-op on the happy path.
        base_event = _clamp_base_event(last_applied, dec.get("event"))
        if base_event != int(last_applied):
            log("[glm-contrib %s] BASE-CLAMP: last_applied=%d exceeds pointer frontier event=%s -- clamped "
                "base_event to the frontier (dead-run leftover folded past the live coordinator?)."
                % (miner, int(last_applied), dec.get("event")))
        ecid = lane.put_delta(payload)
        sig = _sign_contrib(key, wallet, ecid, base_event, miner)  # HMAC (keyed) or secp256k1 (keyless); r-number == base_event (W2 reads it so)
        delta_bytes = int(len(H.pack_arrays(payload, np.float16)))
        record = build_async_contrib_record(miner, i, L, E, base_event, root, ecid, sig, train_flops,
                                             delta_bytes, steps, tokens,
                                             address=(wallet.address if wallet is not None else None),
                                             base_slot_root=slot_root(host, i))
        # F-Q1: unique name per publish; CAMPAIGN-SCOPED so a future run cannot discover our records
        # (and we cannot discover a dead run's) -- the id came from the coordinator's own pointer.
        pub_name = async_publish_name(base_event, miner, publish_k, host_campaign_id(host))
        lane.put_json_named(pub_name, record)
        publish_k += 1
        rounds_done += 1
        last_pub_base_event = int(base_event)      # F5a: from here on, events >= this can judge us
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
    ap.add_argument("--miner", default=os.environ.get("NEURAHASH_SD_MINER"),
                    help="miner id. Keyed: defaults to 'miner0'. KEYLESS (no --key): IGNORED -- the id is "
                         "derived from the wallet address ('glm-'+addr[2:10]) so the coordinator can bind it")
    ap.add_argument("--expert", default=os.environ.get("NEURAHASH_SD_COORD"),
                    help="the GLM expert COORDINATE this miner claims, as L:E (e.g. 1:3). This is the "
                         "shard-claim address: any coordinate this node holds is claimable, whether or "
                         "not the coordinator has ever seen it, so a stranger no longer has to guess a "
                         "free index. Overrides --slot when both are given.")
    ap.add_argument("--slot", type=int, default=None,
                    help="DEPRECATED positional index into --slots, kept so <=v3.3.2 miners and scripts "
                         "keep working. Prefer --expert L:E: an index only means something relative to a "
                         "slot list the coordinator fixed at startup, which is exactly what shard claim "
                         "removes. Defaults to NEURAHASH_SD_EXPERT, else 0.")
    ap.add_argument("--key", default=None)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--wallet-file", dest="wallet_file", default=None,
                    help="path to the LOCAL secp256k1 wallet identity for KEYLESS open admission (created on "
                         "first run if absent); default NEURAHASH_SD_WALLET or ~/.neurahash/glm_miner_key")
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
    ap.add_argument("--advance-after", dest="advance_after", type=int,
                    default=int(os.environ.get("NEURAHASH_SD_ADVANCE_AFTER", "3")),
                    help="SHARD CLAIM: after this many CONSECUTIVE gate rejects on the claimed expert, "
                         "declare it plateaued, release it and claim the next coordinate this node "
                         "holds. That is the sweep -- claim, work, plateau, release, claim next. "
                         "0 disables advancing (stay on one coordinate forever, pre-shard-claim "
                         "behaviour).")
    ap.add_argument("--claim-by", dest="claim_by",
                    default=os.environ.get("NEURAHASH_SD_CLAIM_BY", "hash"),
                    choices=("hash", "affinity"),
                    help="HOW to choose which expert coordinate to train. hash (default, unchanged "
                         "behaviour) derives it from the wallet address -- registry-free but "
                         "routing-BLIND, and routing-blind selection is the variant MoE-Sieve measured "
                         "2.5pp WORSE than router-guided at matched budget. affinity runs ESFT's "
                         "forward-pass-only probe (arXiv:2407.01906) over this node's own train sample "
                         "at startup, claims the HIGHEST-affinity claimable coordinate, and advances "
                         "on plateau to the next-highest instead of the next hash bucket. The full "
                         "ranking is logged once so the choice is auditable.")
    ap.add_argument("--garbage", action="store_true",
                    help="ADVERSARIAL control: publish a correctly-SIGNED but harmful random delta "
                         "(sharddiloco_glm_expert.garbage_delta) that the secret-probe gate must REJECT")
    add_common_args(ap)
    args = ap.parse_args(argv)

    # v3.2.1: signed auto-update at STARTUP -- before the heavy model load, so a fresh release
    # re-execs cheaply. Fail-closed + 6h rate-limited + opt-out (NEURAHASH_AUTOUPDATE=off).
    _maybe_self_update(_flush)

    use_glm_lane_names()
    key, wallet = _resolve_identity(args, log=_flush)
    slots = parse_slots(args.slots)
    L, E, i, _claim_src = resolve_claim(
        args, slots, log=_flush,
        identity=(wallet.address if wallet is not None else (args.miner or "miner0")))
    # SHARD CLAIM: hold EVERY claimable coordinate in the lane host, not just the one we claimed.
    #
    # MEASURED 2026-07-25 on the live WAN lane: holding only the claimed coordinate makes every
    # accepted delta for the OTHERS get skipped as "not resident here", so they sit at the frozen base
    # while the coordinator's move on. The moment we plateau and advance, the new coordinate is stale,
    # which forced a blocking resume_to_root replay -- and on a real lane that replay (24 s manifest
    # read plus a per-record fold) took longer than the coordinator's 600 s idle window, so a HEALTHY
    # advancing miner made the coordinator conclude the lane was idle and shut down at event 29.
    #
    # Registering the whole claimable set up front means the normal catch-up path folds accepted deltas
    # for all of them as events stream by, so an advance needs no replay at all -- resume_to_root then
    # early-returns because the root already matches. Cost is trivial: 5 coordinates is ~189 MB of fp32
    # working state and ~0.1 s per model_root, versus minutes of stall per advance.
    for _c in claim_all_coords(args, list(slots)):
        if tuple(_c) not in slots:
            slots.append(tuple(_c))
    # KEYLESS: the miner id IS the wallet-address derivation (the coordinator binds the name to the recovered
    # key, so any other name is rejected). KEYED: --miner / NEURAHASH_SD_MINER, else the legacy 'miner0'.
    if wallet is not None:
        derived = derive_glm_miner_name(wallet.address)
        if args.miner and args.miner != derived:
            _flush("[glm-contrib] NOTE: --miner %s ignored for the keyless identity; the coordinator binds "
                   "the name to the key -> using derived %s" % (args.miner, derived))
        miner = derived
    else:
        miner = args.miner or "miner0"
    lane = H.ContentLane(args.url, args.token)
    _flush("[glm-contrib %s] UP claims GLM (L%d,E%d) via %s | local_slot=%d domain_slot=%d | "
           "identity=%s mode=%s lane=%s (all-outbound)"
           % (miner, L, E, _claim_src, i, coord_data_slot(L, E),
              ("keyed" if key is not None else "keyless " + wallet.address), args.mode, args.url))

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

    train_ids = node_ids(args, coord_data_slot(L, E), "train")
    val_ids = node_ids(args, coord_data_slot(L, E), "val")

    # ---- --claim-by affinity: replace the routing-BLIND wallet-hash claim with a MEASURED one. -------
    # Runs here and nowhere earlier because it needs the built model, and it reuses THAT model rather
    # than loading a second one (4.02 GiB trunk + 1.125 GiB/resident layer would not fit twice). Forward
    # passes only. Default is still `hash`, so a miner that does not ask for this is byte-identical to
    # v3.3.2. See the ESFT block above for the three measured results that make hash the losing variant.
    _claim_ranked = None
    if str(getattr(args, "claim_by", "hash")) == "affinity":
        L, E, i, _claim_ranked = affinity_claim(args, host, train_ids, L, E, i, miner=miner,
                                                log=_flush)
        # Same mid-flight reload the plateau advance performs: the data shard is a function of the
        # COORDINATE (doms[coord_data_slot(L,E) % len(doms)]), so re-claiming changes which files we
        # train and self-gate on. Skipping this trains one domain and gates on another -- a systematic
        # reject with no error anywhere (C6).
        train_ids = node_ids(args, coord_data_slot(L, E), "train")
        val_ids = node_ids(args, coord_data_slot(L, E), "val")

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

    # ---- FIX B (C6): cross-check the ONE flag nothing else validates -- --domains. -------------------
    # Both roles compute their data shard as doms[coord_data_slot(L,E) % len(doms)] on their OWN list, so a
    # divergent list (or merely a divergent ORDER) gates every delta on text this miner never trained on and
    # rejects it systematically WITH NO ERROR ANYWHERE. Refuse to start instead, naming both lists. Additive:
    # a coordinator that publishes no digest is a pre-Shard-Claim peer -- log once and continue as before.
    _dom_mismatch = domains_mismatch(ptr, args)
    if _dom_mismatch:
        _flush("[glm-contrib %s] FATAL: %s" % (miner, _dom_mismatch))
        return RC_DOMAINS_MISMATCH
    if not ptr.get("domains_digest"):
        _flush("[glm-contrib %s] NOTE: coordinator publishes no domain digest (pre-Shard-Claim peer) -- "
               "cannot cross-check our --domains %s; an undetected mismatch would reject every delta "
               "silently." % (miner, ",".join(domains_list(args))))

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

    # ---- CAMPAIGN SCOPE: latch the coordinator's campaign id, or refuse to publish (fail-closed). ----
    # Everything downstream reads it off the host: the record NAMES we publish under and the lineage
    # ROOTS we hash. Bound here, once, from the first pointer -- before any root that a coordinator will
    # judge is computed (the boot "base ready" line above is only a log). See CAMPAIGN SCOPING above.
    #
    # AFTER the mode decision, and ASYNC-ONLY, deliberately: a v1 SYNC pointer has no field to carry a
    # campaign id, and the coordinator makes scoping inert in sync mode for exactly that reason
    # (sharddiloco_glm_coordinator.main). Refusing there would mean a default-configured miner could no
    # longer join ANY legacy v1 lane -- breaking the "the sync path stays byte-identical" contract this
    # file has kept since alpha-2.
    if _mode_async:
        _camp_refusal = campaign_refusal(ptr)
        if _camp_refusal:
            _flush("[glm-contrib %s] FATAL: %s" % (miner, _camp_refusal))
            return RC_NO_CAMPAIGN
        _campaign = bind_campaign_id(host, pointer_campaign_id(ptr))
        if _campaign:
            _flush("[glm-contrib %s] campaign=%s (from the coordinator's pointer): publishing under %s "
                   "and hashing lineage roots seeded with it -- a dead campaign's records cannot be "
                   "confused with ours." % (miner, _campaign, campaign_prefix(_campaign) + "r<N>/"))
        else:
            _flush("[glm-contrib %s] campaign scoping OFF (NEURAHASH_SD_CAMPAIGN_SCOPE=0) and this "
                   "coordinator advertises none: publishing into the SHARED %sr<N>/ namespace with "
                   "unseeded lineage roots, where a dead campaign's records are indistinguishable from "
                   "ours. Legacy lanes only." % (miner, campaign_prefix(None)))
    if _mode_async:
        return _run_async(args, lane, host, model, cfg, G, key, i, L, E, miner,
                          train_ids, val_ids, seq, _flush, wallet=wallet,
                          claim_ranked=_claim_ranked)

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
        # VRAM-starve PAUSE + OOM round-skip (2026-07-24): same protection as the async lane.
        _vram_pause_if_starved(log, miner=miner)
        t_tr = time.time()
        payload = None                    # what actually goes on the wire (dense delta or factors)
        try:
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
        except Exception as _oom:                                    # noqa: BLE001
            if not _is_cuda_oom(_oom):
                raise
            log("[glm-contrib %s] round %d SKIPPED: CUDA OOM under memory pressure -- freeing cache + "
                "pausing, will retry next round (%s)" % (miner, rnd, str(_oom)[:120]))
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                                        # noqa: BLE001
                pass
            _vram_pause_if_starved(log, miner=miner)
            time.sleep(5.0)
            continue

        # ---- publish the fp16 content-addressed delta + a signed record (D1/D2), trunk FROZEN ----
        ecid = lane.put_delta(payload)
        sig = _sign_contrib(key, wallet, ecid, rnd, miner)       # HMAC (keyed) or secp256k1 (keyless)
        delta_bytes = int(len(H.pack_arrays(payload, np.float16)))
        record = dict(miner=miner, expert=int(i), layer=int(L), glm_expert=int(E), base_round=int(rnd),
                      expert_cid=ecid, trunk_cid=None, sig=sig, train_flops=float(train_flops),
                      trunk_bytes=0, delta_bytes=delta_bytes, base_root=root,
                      base_slot_root=slot_root(host, i))     # SHARD CLAIM: per-coordinate lineage
        if wallet is not None:
            record["address"] = wallet.address                  # keyless: claimed address (coordinator trusts recovered)
        rname = contrib_name(rnd, miner, host_campaign_id(host))   # campaign-scoped (None -> legacy)
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
