"""neurahash.diloco_merge -- CONTRIBUTOR-SAFE subset (miner client).
Only the client-side helpers a miner needs: the shardDiLoCo flag check, the canonical GAP1
message the miner signs, the streaming-subset wire codec for its OWN delta, and the client
store-URL resolver. The coordinator merge/reward/economics + secret-probe (shard_merge_round,
FlopMeter, SecretRotatedProbe, ...) live ONLY in the private full-node package. Functions are
copied verbatim from the private source so signatures/wire formats match the coordinator."""

import io

import json

import os

import sys

import urllib.request

import numpy as np

CONTRIB_SIG_VERSION = "neurahash-diloco-contrib-v1"

def contrib_canonical_message(delta_cid, base_round, name, val_before, val_after):
    """The EXACT bytes a contribution record is signed over and verified against (GAP1). BOTH the signer
    (tools/diloco_contributor.publish_delta) and the verifier (poll_contrib_records) build the message
    through THIS one function, so they cannot drift out of sync — a drift would silently break every
    signature. `name` is the contributor IDENTITY string (the record's own 'contributor' field, the
    value passed as `diloco --name`), NOT the registry-slot key. Fields are newline-joined after the
    version tag; the numeric/None fields go through str() so JSON round-tripping is lossless
    (json.dumps uses float.__repr__, which round-trips, and None <-> null <-> 'None')."""
    return (
        CONTRIB_SIG_VERSION + "\n"
        + str(delta_cid) + "\n"
        + str(base_round) + "\n"
        + str(name) + "\n"
        + str(val_before) + "\n"
        + str(val_after)
    ).encode("utf-8")

def registry_posture():
    """D2 master gate NEURAHASH_DILOCO_REGISTRY = off|audit|enforce (default/unknown -> 'off'). off ->
    single store, byte-identical to today; audit -> replicate + poll replicas but RETURN the primary-only
    view (observe-only, zero control-flow change); enforce -> the merged multi-store view is the record
    set. Mirrors shipped B12-1 NEURAHASH_ELECTED_PROPOSER (docs/DECENTRALIZE_D_SERIES.md 'Conventions')."""
    v = (os.environ.get("NEURAHASH_DILOCO_REGISTRY", "off") or "off").strip().lower()
    return v if v in ("off", "audit", "enforce") else "off"

def registry_store_urls(primary):
    """The ORDERED, deduped content-store base-URLs the registry spans -- the ONE place the D2 flag is
    parsed, shared by BOTH publish (multi-PUT) and poll (multi-source merge). Always starts with `primary`;
    when the gate is ON (registry_posture() != 'off') it APPENDS the NEURAHASH_DILOCO_MERGE_URLS comma-list
    of replica stores (blanks and a replica equal to an existing entry dropped). Gate off, or no replicas
    configured -> [primary] alone, so publish and poll stay byte-identical to the single-store path. A
    falsy `primary` yields [] (preserves today's 'no registry_url -> no PUT')."""
    primary = (primary or "").rstrip("/")
    urls = [primary] if primary else []
    if registry_posture() == "off":
        return urls
    for u in (os.environ.get("NEURAHASH_DILOCO_MERGE_URLS", "") or "").split(","):
        u = u.strip().rstrip("/")
        if u and u not in urls:
            urls.append(u)
    return urls

def shard_diloco_on():
    """NEURAHASH_SHARDDILOCO master gate (off|audit|enforce house style, read as on/off here; mirrors
    corpus_enforce_on() / registry_posture()). Unset / '0' / 'off' -> False: the #133 merge lane is
    byte-identical to today (whole-model trunk, experts FROZEN). Truthy -> True: the coordinator ALSO
    gates + merges per-expert pseudo-gradients via shard_merge_round. Committed code never sets it."""
    return (os.environ.get("NEURAHASH_SHARDDILOCO", "0") or "0").strip().lower() \
        not in ("", "0", "false", "off", "no")

def _stream_frac_env():
    v = os.environ.get("NEURAHASH_SHARDDILOCO_STREAM_FRAC", "")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 1.0
    return f if 0.0 < f < 1.0 else 1.0

def stream_frac():
    """NEURAHASH_SHARDDILOCO_STREAM_FRAC (#126): the rolling FRACTION of trunk params synced per outer
    step. Unset / '' / <=0 / >=1 -> 1.0 == full-trunk sync every step (byte-identical to today). A value
    in (0,1) turns on streaming with num_fragments = round(1/frac) staggered fragments."""
    return _stream_frac_env()

def stream_num_fragments(frac):
    """num_fragments for a streaming fraction: round(1/frac), floored at 1. frac>=1 or <=0 -> 1 (full)."""
    f = float(frac)
    if not (0.0 < f < 1.0):
        return 1
    return max(1, int(round(1.0 / f)))

def _fragment_sel(n, num_fragments, fragment):
    """Boolean 1-D selector over n flat elements: element i is in `fragment` iff i % num_fragments ==
    fragment. The ONE partition rule shared by mask / extract / reconstruct so they cannot drift."""
    return (np.arange(int(n)) % int(num_fragments)) == (int(fragment) % int(num_fragments))

def extract_trunk_fragment(trunk_delta, num_fragments, fragment):
    """CONTRIBUTOR side: compact the fragment of a full trunk delta to {key: 1-D ndarray of the selected
    elements in flat order}. Bytes ~ (1/num_fragments) of the full delta -- the streaming bandwidth win.
    reconstruct_trunk_fragment inverts it on the coordinator using the SAME partition rule."""
    out = {}
    for k, arr in trunk_delta.items():
        flat = np.asarray(arr).reshape(-1)
        out[k] = flat[_fragment_sel(flat.size, num_fragments, fragment)]
    return out

def stream_publish_trunk(trunk_delta, base_round, stream_frac=None):
    """CONTRIBUTOR helper: reduce a FULL trunk pseudo-grad to the rolling fragment for outer round
    `base_round` when streaming is on. Returns the compact fragment {key: 1-D} (bytes ~ frac*full) when
    frac in (0,1); the UNCHANGED full trunk_delta when off (frac>=1) so the wire is byte-identical to
    today. Fragment = base_round % num_fragments (the staggered rolling schedule)."""
    frac = _stream_frac_env() if stream_frac is None else float(stream_frac)
    if not (0.0 < frac < 1.0):
        return trunk_delta
    nfrag = stream_num_fragments(frac)
    return extract_trunk_fragment(trunk_delta, nfrag, int(base_round) % nfrag)
