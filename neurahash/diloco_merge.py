"""Coordinator side of DiLoCo-over-IPFS: pull contributed trunk deltas and merge the ones that IMPROVE
the held-out loss into the live global trunk. The companion to tools/diloco_contributor.py.

FLOW. A remote contributor (a Colab T4, another box) fetches the pool's latest checkpoint by CID,
trains the trunk locally, and publishes back TWO things:
  * the trunk DELTA (trained_trunk - base_trunk) as a safe .npz to IPFS/Pinata -> a `delta_cid`;
  * a tiny record {contributor, delta_cid, base_round, ...} to the VPS content store under the stable
    name "contrib-<contributor>".
The coordinator POLLS the content-store manifest for those "contrib-*" records, FETCHES each new delta
(CID-verified), and — GATED on a real held-out improvement — folds it into `global_trunk` with the pool's
DiLoCo outer step (global_trunk += OUTER * delta). A bad or adversarial delta cannot help the held-out
set, so it is rejected; the model can only move DOWNHILL in held-out loss via this path.

SAFETY. The slow, untrusted work (network fetch, .npz parse) is done OFF the round loop. The decision +
mutation (`apply_delta_gated`) is a pure function of numpy arrays and an eval callback, meant to run
INSIDE the single-threaded round loop so it never races the trainer. Deltas load via numpy (np.load,
allow_pickle=False) — never torch.load — so a hostile artifact cannot execute code. Default OFF; AUDIT
logs the verdict without applying; ENFORCE applies. Mirrors the repo's audit-first discipline.
"""
import io
import json
import os
import sys
import urllib.request

import numpy as np

from neurahash import delta_codec  # numpy-only decoder for COMPRESSED q8k deltas (no torch pulled in)

_UA = "Mozilla/5.0 (NeuraHash-pool diloco_merge) Gecko/20100101"
DEFAULT_GATEWAYS = (
    "https://gateway.pinata.cloud/ipfs/{cid}",
    "https://{cid}.ipfs.w3s.link",
    "https://{cid}.ipfs.dweb.link",
    "https://ipfs.io/ipfs/{cid}",
)

# GAP1 (docs/GO_PUBLIC_DESIGN.md): wallet-signed contribution records. A public contribution today is
# unsigned free text and proves NOTHING about who produced it. We bind each delta to a wallet address
# the coordinator can verify by re-recovering the signer from a secp256k1 signature (never trusting a
# self-declared address). Bumping any signed field must bump this version tag — an old signature can
# then never be reinterpreted against a new field layout.
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


def _log(msg):
    """One ASCII line to stderr. The coordinator console is cp1252, so keep this ASCII-only: CIDs and
    0x-addresses already are, and we deliberately do NOT echo the free-text contributor name here."""
    print("[diloco_merge] " + msg, file=sys.stderr, flush=True)


def _recover_contrib_signer(rec):
    """Recover the address that signed `rec` from rec['sig'] over the canonical message REBUILT from the
    record's own fields (contrib_canonical_message). Returns the recovered 0x address, or None when the
    record is unsigned or the signature is malformed/unrecoverable. The record's self-declared 'address'
    is NEVER consulted here — only the value recovered from the signature is returned, so a forged
    'address' cannot help an attacker. Reuses the repo's domain-separated secp256k1 recover_bytes."""
    sig = rec.get("sig")
    if not sig:
        return None
    try:
        from neura_l1.signing import recover_bytes          # lazy: eth_account only when a sig is present
        msg = contrib_canonical_message(rec.get("delta_cid"), rec.get("base_round"),
                                        rec.get("contributor"), rec.get("val_before"),
                                        rec.get("val_after"))
        return recover_bytes(msg, sig)
    except Exception:                                       # noqa: BLE001 — malformed sig -> treat as unsigned
        return None


def _dedup_prefer_signed(records):
    """Dedup contribution records by delta_cid, PREFERRING a signed (sig_ok) record over an unsigned one
    for the same CID. Returns the surviving records in first-appearance order of their winners. Kept as a
    SEPARATE function (not inline in the poll loop) so the dedup policy is independently testable/reviewable.

    WHY (GAP1 residual — CID-slot squatting denial). The manifest's iteration order is NOT time order, so
    at the default REQUIRE_SIG=0 an UNSIGNED record bearing a victim's CID can be seen BEFORE the honest
    signed one. A plain first-seen-wins dedup would then keep the unsigned squatter and DROP the genuine
    signed record — the delta still merges (held-out-gated) but the real author is never paid: a denial /
    starve-the-payment attack that needs no key. RULE:
      * a sig_ok record DISPLACES a non-sig_ok record already holding a CID slot, regardless of order;
      * among multiple sig_ok records for one CID, keep the FIRST seen and log 'cid-contested';
      * among only-unsigned records for a CID, keep the first seen and log 'cid-theft or duplicate';
      * an unsigned record NEVER displaces a signed one already in the slot.

    RESIDUAL, NOT fixed here (documented deliberately). Signed-THEFT — an attacker who RE-SIGNS a victim's
    CID with the attacker's OWN key produces a sig_ok record too, so signature preference alone cannot tell
    the genuine author from the re-signer. The real fix is a commit-reveal binding: the author publishes
    H(cid||nonce) BEFORE revealing the CID, so the first committer is provably the author. This interim
    sig-preference closes only the unsigned-squatter denial above.
    TODO(GAP1): implement commit-reveal (publish H(cid||nonce) first, reveal later) to close signed-theft."""
    chosen = {}           # cid -> current winning record
    order = []            # cids in first-appearance order of their winner's first slot
    for rec in records:
        cid = rec.get("delta_cid")
        if cid not in chosen:
            chosen[cid] = rec
            order.append(cid)
            continue
        incoming_signed = bool(rec.get("sig_ok"))
        current_signed = bool(chosen[cid].get("sig_ok"))
        if incoming_signed and not current_signed:
            # a signed record displaces the unsigned squatter that happened to be seen first
            _log("signed record displaces unsigned squatter for cid " + str(cid) + " (cid-contested)")
            chosen[cid] = rec
        elif incoming_signed and current_signed:
            _log("drop duplicate signed contribution for cid " + str(cid) + " (cid-contested)")
        else:
            _log("drop duplicate contribution for cid " + str(cid) + " (cid-theft or duplicate)")
    return [chosen[c] for c in order]


# --------------------------------------------------------------- D2: replicated registry (N stores + gossip)
# The contribution registry must not depend on a SINGLE VPS being the manifest authority (owner directive:
# no trusted coordinator -- memory full-decentralization-goal; docs/DECENTRALIZE_D_SERIES.md D2). Two
# additive, default-off pieces implement that, REUSING the shipped sig-verify + dedup below:
#   * MULTI-STORE: a miner PUTs its signed record to N content stores (tools/diloco_contributor.publish_delta
#     multi-PUT) and poll_contrib_records MERGES records fetched from N stores, deduped by delta_cid with
#     signed-preference so a signed record still wins ACROSS stores (_dedup_prefer_signed, unchanged).
#   * GOSSIP: a thin (delta_cid, signer) announce so a node learns of a CID its local stores have not
#     indexed yet -- kept a PURE function over an injected broadcast callable (announce_contrib), no live
#     sockets this phase; the receiver still fetches + re-verifies the real record.
# Master gate NEURAHASH_DILOCO_REGISTRY=off|audit|enforce (B12/D-series house style): off/unset == today's
# single store, byte-identical; audit == replicate + poll replicas but RETURN primary-only (observe-only
# shadow); enforce == the merged multi-store view gates. Replica list: NEURAHASH_DILOCO_MERGE_URLS.


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


def merge_contrib_records(*record_lists):
    """Accumulate contribution records from N sources (several stores' _poll_one_store outputs, and -- once
    wired -- gossip-delivered records) into ONE deduped view: by delta_cid with signed-preference, reusing
    the shipped _dedup_prefer_signed so a sig_ok record wins a contested CID regardless of which source it
    came from. Pure. Named (not inlined) so the cross-source merge policy is independently testable and
    reused by the gossip receive path once it is wired (see announce_contrib's WIRING TODO)."""
    flat = [rec for lst in record_lists for rec in (lst or [])]
    return _dedup_prefer_signed(flat)


# ---- gossip announce (D2 gossip half): payload-kind + build/parse + a transport-agnostic best-effort adapter
CONTRIB_ANNOUNCE_KIND = "diloco-contrib-announce"


def contrib_announce_payload(delta_cid, signer):
    """Build the gossip payload announcing a contribution EXISTS: its delta_cid + the RECOVERED signer
    address (never a self-declared one). A thin discovery HINT, not the record -- a receiver still fetches
    the full signed record from a content store and re-verifies it (poll_contrib_records) before it can win
    a CID slot; the announce only makes peers AWARE of a CID their local stores may not have indexed yet.
    Mirrors neura_l1/gossip_net.py's application-dict shape ({'kind': ...}) so TCPGossipNode.broadcast
    floods it unchanged."""
    return {"kind": CONTRIB_ANNOUNCE_KIND, "delta_cid": str(delta_cid),
            "signer": (None if signer is None else str(signer))}


def parse_contrib_announce(payload):
    """Receive side: validate a gossip payload is a well-formed contribution announce and return
    {'delta_cid', 'signer'}, else None (wrong kind / malformed). Pure; safe inside a gossip on_message
    callback. The announce is NEVER trusted -- it is only a hint to go fetch + verify the real record from
    a store, so a forged announce costs at most a wasted fetch, never a spend."""
    if not isinstance(payload, dict) or payload.get("kind") != CONTRIB_ANNOUNCE_KIND:
        return None
    cid = payload.get("delta_cid")
    if not cid:
        return None
    return {"delta_cid": str(cid), "signer": payload.get("signer")}


def announce_contrib(broadcast_fn, delta_cid, signer):
    """Best-effort gossip announce of a contribution's (delta_cid, signer). `broadcast_fn` is any callable
    taking one JSON-able payload dict; returns the announced payload, or None on failure (announce is
    best-effort -- it must never block a publish). Transport-agnostic on purpose (a plain callable, no
    socket) so it is a pure, testable function needing no live gossip node in the default build.

    WIRING TODO (D2 gossip half -- NOT wired to the live loop this phase). In production the coordinator/
    miner holds one neura_l1.gossip_net.TCPGossipNode and passes `node.broadcast` here at the contribution
    PUT site; the receive side runs parse_contrib_announce inside that node's on_message callback and feeds
    merge_contrib_records. Left unwired deliberately per docs/DECENTRALIZE_D_SERIES.md D2(f): the gossip
    half is one small adapter, and cross-store SIGNED-CID-theft under partition (an attacker re-signing a
    victim's CID with its OWN key, then announcing it) is a DESIGN-ONLY residual SHARED WITH GAP1 -- see
    _dedup_prefer_signed's TODO(GAP1); its real fix is commit-reveal on the CID, not this adapter."""
    try:
        payload = contrib_announce_payload(delta_cid, signer)
        broadcast_fn(payload)
        return payload
    except Exception as e:                                   # noqa: BLE001 -- best-effort; never block publish
        _log("gossip announce failed for cid " + str(delta_cid) + ": " + repr(e))
        return None


def _poll_one_store(base_url, timeout=15):
    """Read ONE content store's manifest and return its contribution records -- one per contributor slot
    named 'contrib-<name>' -- as a list of dicts {name, contributor, delta_cid, base_round, sig_ok,
    address, ...}. The single-store worker; poll_contrib_records polls N of these and merges (D2).

    SIGNATURE VERIFICATION (GAP1). Every record is checked against its wallet signature: the signer is
    RE-RECOVERED from rec['sig'] over the canonical message rebuilt from the record's own fields, then
    compared to the record's self-declared 'address'. Two keys are attached to EVERY returned record and
    ALWAYS overwrite whatever the record claimed:
      * 'sig_ok'  : True only when a signature is present, recovers, AND matches the declared address.
                    Unsigned, or tampered (ANY signed field altered), -> False.
      * 'address' : the RECOVERED signer address, or None when unsigned/unrecoverable. Downstream
                    settlement (GAP2) must pay THIS value, never the self-declared field.
    NEURAHASH_DILOCO_REQUIRE_SIG=1 (default 0, off|audit|enforce house style) DROPS every record with
    sig_ok False (one log line per drop) — the enforce posture for a public pool.

    ANTI-THEFT DEDUP (_dedup_prefer_signed). Records are deduped by delta_cid with a SIGNED-PREFERENCE:
    for a contested CID a sig_ok record wins over an unsigned squatter regardless of manifest order (the
    order is not time order), among multiple signed records the first-seen wins ('cid-contested'), and an
    unsigned duplicate is dropped ('cid-theft or duplicate'). This blunts an unsigned front-runner that
    would otherwise starve the genuine signed author of payment. Signed-theft (an attacker re-signing the
    victim's CID with its OWN key) is a documented RESIDUAL whose real fix is a commit-reveal on the CID —
    see _dedup_prefer_signed's TODO(GAP1).

    Best-effort throughout: a malformed record is skipped, a store error returns []."""
    try:
        man = _get_json(base_url.rstrip("/") + "/manifest", timeout)
    except Exception:                                       # noqa: BLE001 — store unreachable -> nothing to do
        return []
    require_sig = os.environ.get("NEURAHASH_DILOCO_REQUIRE_SIG", "0") not in ("", "0", "false", "False")
    verified = []
    for name, meta in (man or {}).items():
        if not name.startswith("contrib-"):
            continue
        try:
            rec = _get_json(base_url.rstrip("/") + "/o/" + meta["sha256"], timeout)
        except Exception:                                   # noqa: BLE001 — skip a bad record, keep the rest
            continue
        if not (isinstance(rec, dict) and rec.get("delta_cid")):
            continue
        rec["name"] = name
        # GAP1 verify: recover the signer and trust ONLY that; the self-declared 'address' is discarded.
        declared = rec.get("address")
        recovered = _recover_contrib_signer(rec)
        rec["sig_ok"] = bool(recovered is not None and declared is not None
                             and str(recovered).lower() == str(declared).lower())
        rec["address"] = recovered
        if require_sig and not rec["sig_ok"]:
            _log("drop unsigned/bad-sig contribution for cid " + str(rec.get("delta_cid"))
                 + " (NEURAHASH_DILOCO_REQUIRE_SIG=1)")
            continue
        verified.append(rec)
    # anti-theft dedup by delta_cid with SIGNED-PREFERENCE, run AFTER the require_sig gate and over the
    # full verified set (so a signed record can displace an unsigned squatter seen earlier in the manifest,
    # not just be dropped behind it). See _dedup_prefer_signed for the rule + the signed-theft residual.
    return _dedup_prefer_signed(verified)


# --------------------------------------------------------------- declared-corpus gate (merge/reward safety)
# The SYNCHRONOUS join path hard-rejects a corpus_sha mismatch (sharded_pool_node.py ~L2379) before a worker
# ever mines. The ASYNC DiLoCo contribution path had NO such check: a contributor that trained on a DIFFERENT
# corpus could have its delta held-out-gated and (GAP2) REWARDED -- a pouw-verified-not-useful-class hole,
# since the held-out eval is a noisy 8-iter gate, NOT a corpus-identity proof. This adds a declared-corpus
# gate as DEFENSE-IN-DEPTH (an ADDITIONAL gate, never a replacement for the held-out eval): a contribution
# record DECLARES the corpus_sha it trained under (tools/diloco_contributor.publish_delta, additive field),
# and the coordinator DROPS a record whose declared sha MISMATCHES its own BEFORE fetching/merging/rewarding.
#
# DEFAULT-SAFE / backward-compatible: a record with NO corpus_sha field (every pre-this-change contributor)
# is treated exactly as today (no check) so the default path is byte-identical. NEURAHASH_DILOCO_CORPUS_ENFORCE
# (off|1, default off) additionally REQUIRES the field -- for a future all-synced fleet it drops records that
# omit it. corpus_sha is an UNSIGNED additive field (deliberately NOT part of contrib_canonical_message):
# signing it would force a CONTRIB_SIG_VERSION bump that invalidates every existing signature for no
# threat-model gain -- a malicious contributor would simply sign its lie, and third-party field tampering can
# only cause a REJECT (a denial already inside the documented CID-theft residual class), never a false-accept
# (the held-out eval still gates everything that passes here).


def corpus_enforce_on():
    """NEURAHASH_DILOCO_CORPUS_ENFORCE gate (off|audit|enforce house style, read as a simple on/off here):
    default OFF (unset / '' / '0' / 'false' -> False). When ON, a contribution record that OMITS corpus_sha
    is rejected (the declaration becomes mandatory) -- the enforce posture for an all-synced fleet. OFF
    (default) -> a missing field is not checked, so the default ingestion path is byte-identical to before
    this change."""
    return os.environ.get("NEURAHASH_DILOCO_CORPUS_ENFORCE", "0") not in ("", "0", "false", "False")


def corpus_sha_ok(rec, coord_corpus_sha, *, enforce=None):
    """Decide whether a polled DiLoCo contribution `rec` may proceed to fetch/merge/reward, based ONLY on
    its DECLARED corpus_sha vs the coordinator's own `coord_corpus_sha`. Pure; no I/O. Returns (ok, reason):

      * rec HAS corpus_sha and it MATCHES coord_corpus_sha -> (True,  "corpus-match")
      * rec HAS corpus_sha and it MISMATCHES               -> (False, "corpus-mismatch")   REJECT
      * rec LACKS corpus_sha (a pre-this-change contributor):
          - enforce OFF (default) -> (True,  "no-declared-corpus")   backward-compatible, no check
          - enforce ON            -> (False, "corpus-missing")       the declaration is required

    `enforce` defaults to the NEURAHASH_DILOCO_CORPUS_ENFORCE env (corpus_enforce_on()); pass an explicit
    bool to override (tests). When `coord_corpus_sha` is falsy (the coordinator has no corpus identity to
    compare against) the gate is a NO-OP -> (True, "no-coord-corpus"), so a mis-provisioned coordinator can
    never silently reject every contribution. This gate is ADDITIVE to the held-out eval, never a
    replacement -- everything that passes here still faces apply_delta_gated's held-out gate."""
    if enforce is None:
        enforce = corpus_enforce_on()
    if not coord_corpus_sha:
        return True, "no-coord-corpus"
    declared = rec.get("corpus_sha")
    if declared is None:
        return (False, "corpus-missing") if enforce else (True, "no-declared-corpus")
    if str(declared).strip().lower() == str(coord_corpus_sha).strip().lower():
        return True, "corpus-match"
    return False, "corpus-mismatch"


def poll_contrib_records(base_url, timeout=15):
    """Return the current contribution records MERGED across the replicated registry (D2). With the gate
    off/unset (registry_posture() == 'off') this is byte-identical to polling the single store `base_url`
    -- exactly today's behavior and the only path the live coordinator takes until an operator opts in
    (the sole caller, sharded_pool_node.py, is unchanged). With the gate ON it ALSO polls the
    NEURAHASH_DILOCO_MERGE_URLS replica stores and merges by delta_cid with signed-preference
    (merge_contrib_records): 'audit' logs what the replicas would add but RETURNS the primary-only view
    (observe-only shadow), 'enforce' RETURNS the merged multi-store view. Every store is polled
    best-effort (_poll_one_store swallows an unreachable store), so replication only ADDS availability,
    never a new failure mode."""
    urls = registry_store_urls(base_url)
    primary = urls[0] if urls else base_url
    primary_recs = _poll_one_store(primary, timeout)
    posture = registry_posture()
    if len(urls) <= 1 or posture == "off":
        return primary_recs                                 # single store: byte-identical to today
    replica_recs = [_poll_one_store(u, timeout) for u in urls[1:]]
    merged = merge_contrib_records(primary_recs, *replica_recs)
    if posture == "audit":
        _log("audit: %d stores would merge to %d records (primary alone %d); returning primary-only "
             "(observe-only)" % (len(urls), len(merged), len(primary_recs)))
        return primary_recs
    _log("enforce: merged %d stores -> %d records (primary alone %d)"
         % (len(urls), len(merged), len(primary_recs)))
    return merged


def _decode_delta_payload(data):
    """Turn fetched delta bytes into a dict {REAL trunk_key: float32 ndarray}, AUTO-DETECTING the wire
    format so apply_delta_gated always receives trunk-keyed tensors it can match. Two formats exist:

      * COMPRESSED q8k (neurahash/delta_codec) — what public miners publish via
        run_miner/--publish-compressed-delta. It is an npz whose keys are the codec's internals
        (__manifest__/_scales/g0/v0/...), which share ZERO names with the trunk; a bare np.load would
        hand apply_delta_gated those junk keys -> no matching trunk keys -> a non-informative REJECT
        (the production gap, E2E proof 2026-07-11). We DETECT it by the delta_codec manifest signature
        (the __manifest__ array present AND its JSON magic == delta_codec.MAGIC) and route the RAW bytes
        through delta_codec.decompress_delta, which reconstructs {name: float32 array} of the ORIGINAL
        trunk shapes (real trunk keys).
      * LEGACY uncompressed npz — a plain {trunk_key: ndarray} archive (the pre-compression path).
        Kept byte-for-byte unchanged for back-compat.

    Both paths load with allow_pickle=False, so a hostile artifact CANNOT execute code (delta_codec's
    decode holds the same discipline). Raises if the payload isn't a well-formed npz."""
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        is_compressed = False
        if delta_codec._MANIFEST_KEY in z.files:             # probe the codec manifest signature
            try:
                manifest = json.loads(z[delta_codec._MANIFEST_KEY].tobytes().decode("utf-8"))
                is_compressed = manifest.get("magic") == delta_codec.MAGIC
            except Exception:                               # noqa: BLE001 — __manifest__ that isn't ours -> legacy
                is_compressed = False
        if not is_compressed:
            return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}   # LEGACY path, unchanged
    # COMPRESSED q8k: reuse the codec's own decoder (numpy-only, allow_pickle=False) on the raw bytes.
    return delta_codec.decompress_delta(data)


def _cid_matches(cid, data):
    """Verify fetched delta bytes reproduce `cid`. Trust is in the CID: the miner's signature covers the
    canonical message over THIS delta_cid (contrib_canonical_message), so the bytes<->CID binding IS the
    integrity chain — a malicious/compromised gateway that substitutes bytes must be rejected here (the
    held-out gate does NOT close this gap: a substituted delta can pass held-out while reward attributes
    to the ORIGINAL signer). Delegates to the ONE canonical checker in tools/ipfs_checkpoint._cid_matches
    (a version-aware offline `ipfs add -n` re-add). If tools isn't importable in this launch context we
    mirror that checker's OWN fallback semantics: allow unless the operator set NEURAHASH_IPFS_STRICT, so
    a coordinator with no local ipfs still runs but an operator can demand strict verification."""
    try:
        from tools.ipfs_checkpoint import _cid_matches as _canonical
    except ImportError:
        return os.environ.get("NEURAHASH_IPFS_STRICT", "") not in ("1", "true", "yes", "on")
    return _canonical(cid, data)


def fetch_delta(cid, gateways=DEFAULT_GATEWAYS, timeout=180, max_bytes=1_000_000_000, verify_cid=True):
    """Fetch a contribution delta by CID from the first working gateway and return it as a dict
    {trunk_key: float32 ndarray}. Auto-detects the wire format: a COMPRESSED q8k delta (delta_codec,
    what public miners publish) is decompressed to real trunk keys; a LEGACY uncompressed npz is loaded
    as-is (see _decode_delta_payload). Loaded with allow_pickle=False so a hostile artifact CANNOT
    execute code. When `verify_cid` (default) each gateway's bytes must reproduce `cid` before decoding
    (trust is in the CID, not the gateway — a substituted body is rejected and the next gateway tried),
    so a malicious/compromised IPFS gateway cannot swap in a different delta under the signed CID. Raises
    if every gateway fails, verification fails everywhere, or the payload isn't a well-formed delta."""
    last = None
    for tmpl in gateways:
        url = tmpl.format(cid=cid)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read(max_bytes + 1)
            if len(data) > max_bytes:
                last = f"{url}: payload exceeds {max_bytes} bytes"
                continue
            if verify_cid and not _cid_matches(cid, data):
                last = f"{url}: CID verification FAILED (body != {cid})"
                continue
            delta = _decode_delta_payload(data)
            if not delta:
                last = f"{url}: empty delta"
                continue
            return delta
        except Exception as e:                              # noqa: BLE001 — try the next gateway
            last = f"{url}: {e!r}"
            continue
    raise RuntimeError(f"all gateways failed for delta {cid}; last: {last}")


def apply_delta_gated(global_trunk_np, delta_np, eval_fn, *, outer=0.7, margin=0.0, apply=True):
    """Decide (and optionally apply) a contributed delta, GATED on held-out improvement. PURE except for
    `eval_fn` — safe to call inside the single-threaded round loop.

      global_trunk_np : {key: float32 ndarray}  the coordinator's current trunk (mutated IN PLACE iff
                        accepted and apply=True).
      delta_np        : {key: float32 ndarray}  the contributed trunk delta.
      eval_fn(trunk)  : -> float held-out loss for a candidate trunk dict (the caller wires this to the
                        model + val set; forward-only).
      outer, margin   : merged = trunk + outer*delta; accept iff eval(merged) <= eval(base) - margin.

    Returns a verdict dict {accepted, base_val, merged_val, delta_norm, keys_matched, keys_missing}.
    On accept+apply, global_trunk_np[k] += outer*delta[k] for the matched keys. On reject, untouched."""
    keys = [k for k in global_trunk_np if k in delta_np and
            np.shape(global_trunk_np[k]) == np.shape(delta_np[k])]
    missing = [k for k in global_trunk_np if k not in keys]
    verdict = {"accepted": False, "keys_matched": len(keys), "keys_missing": len(missing),
               "base_val": None, "merged_val": None, "delta_norm": None}
    if not keys:
        verdict["reason"] = "no matching trunk keys"
        return verdict
    base_val = float(eval_fn(global_trunk_np))
    merged = dict(global_trunk_np)
    for k in keys:
        merged[k] = global_trunk_np[k] + outer * delta_np[k]
    merged_val = float(eval_fn(merged))
    dn = float(np.sqrt(sum(float((delta_np[k] ** 2).sum()) for k in keys)))
    # STRICT `<`, not `<=`. At margin=0 the old `<=` accepted a ZERO delta: merged_val == base_val
    # satisfies `base_val <= base_val - 0`, so a miner could submit nothing, be reported MINT with
    # gain 0.0, and count as a contributor -- a free-riding path, and one a public testnet finds
    # fast. (Observed live 2026-07-21 at round 7 of a shakedown: `MINT` with the model_root and
    # held-out CE both unchanged.) Strict inequality is a no-op at any positive margin, because
    # exact float equality there is measure-zero -- so this tightens the degenerate case only.
    verdict.update(base_val=base_val, merged_val=merged_val, delta_norm=dn,
                   accepted=bool(merged_val < base_val - margin))
    if verdict["accepted"] and apply:
        for k in keys:
            global_trunk_np[k] = global_trunk_np[k] + outer * delta_np[k]
    return verdict


# ------------------------------------------------------ BATCHED gain-weighted merge (Decoupled DiLoCo)
# WHY. The serial merge path above applies EXACTLY ONE contributed delta per eligible round against an
# already-mutated trunk, so a second same-window contributor's delta lands many rounds later as a STALE
# update. DeepMind's Decoupled DiLoCo (arxiv 2604.21428, Table 5) measured that "merge whoever is ready"
# alone hurts quality, while GATHERING concurrent candidates within a bounded GRACE window and merging
# them TOGETHER (weighted by their measured contribution) recovers it. These two pure functions implement
# exactly that policy so it is unit-testable without the 280 KB coordinator monolith. Both are DEFAULT-OFF
# at the call site: with the batch flag off the coordinator keeps popping one candidate per round and the
# single-delta apply_delta_gated above is the ONLY merge path (byte-for-byte today's behavior).


def gather_batch(pending, now, grace_s, grace_max_s, batch_max):
    """Decide whether to MERGE a batch of pending DiLoCo candidates now, or DEFER to let the grace window
    gather more concurrent arrivals. PURE; no I/O, no clock read (the caller passes `now`). Returns
    (batch_list, defer_bool):

      pending      : list of (rec, delta, arrival_ts) in FIFO (arrival) order; arrival_ts is a wall clock
                     seconds value (time.time()) captured when the coordinator's poller queued the item.
      now          : current wall-clock seconds (time.time()) — passed in so this stays a pure function.
      grace_s      : the grace window. While the NEWEST arrival is younger than this, keep waiting for
                     more of the same window (return ([], True)). grace_s <= 0 disables deferral entirely
                     (the flag-off / no-grace path returns the FIFO batch immediately, never defers).
      grace_max_s  : ANTI-SLOWLORIS cap. A steady drip of fresh arrivals must NOT extend deferral forever:
                     once the OLDEST pending candidate has waited grace_max_s, we merge regardless of how
                     fresh the newest arrival is. Bounds worst-case merge latency at grace_max_s after the
                     first pending candidate arrived.
      batch_max    : cap on how many candidates enter one merge (FIFO, oldest first).

    DEFER (return ([], True)) iff ALL hold: grace_s > 0 AND (now - newest_arrival) < grace_s AND
    (now - oldest_arrival) < grace_max_s. Otherwise return (pending[:batch_max], False). Empty pending is
    never a defer (returns ([], False))."""
    if not pending:
        return [], False
    arrivals = [ts for (_r, _d, ts) in pending]
    newest, oldest = max(arrivals), min(arrivals)
    if grace_s > 0 and (now - newest) < grace_s and (now - oldest) < grace_max_s:
        return [], True                                       # window still open, slowloris cap not hit -> wait
    n = int(batch_max) if int(batch_max) > 0 else len(pending)
    return list(pending[:n]), False


def apply_deltas_weighted_gated(global_trunk_np, candidates, eval_fn, *, outer=0.7, margin=0.0, apply=True):
    """Merge a BATCH of contributed deltas in ONE gain-weighted step, GATED on held-out improvement, with a
    serial fallback so the batched path is strictly no worse than today's one-at-a-time merge. PURE except
    for `eval_fn` — safe inside the single-threaded round loop, like apply_delta_gated.

      global_trunk_np : {key: float32 ndarray}  the coordinator's current trunk (mutated IN PLACE iff the
                        batch is accepted and apply=True; otherwise untouched — every probe/merge below runs
                        on COPIES).
      candidates      : list of (rec, delta_np). `rec` is the raw contribution record (its 'delta_cid' is
                        the deterministic serial tie-break); delta_np is {key: float32 ndarray}.
      eval_fn(trunk)  : -> float held-out loss for a candidate trunk dict.
      outer, margin   : combined = trunk + outer * sum_i(w_i * delta_i); accept iff eval(combined) <=
                        eval(base) - margin.

    POLICY.
      1. PROBE each candidate individually (reusing apply_delta_gated with apply=False — the merge/eval math
         is NOT duplicated): gain_i = base_val - eval(trunk + outer*delta_i).
      2. EXCLUDE gain_i <= 0 candidates from the weighting (a delta that does not help held-out on its own).
         They still appear in per_candidate as REJECTED (weight 0, accepted False) so rewards/logs see them.
      3. If >= 1 positive candidate: w_i = gain_i / sum(positive gains); combined = trunk + outer *
         sum_i(w_i * delta_i); accept iff eval(combined) <= base_val - margin. On accept (+apply) mutate the
         trunk in place. Per-candidate attributed gain = (base_val - merged_val) * w_i, which SUMS to the
         total gain (ledger/economics conservation — the caller pays the attributed gain).
      4. If the combined merge FAILS the gate (positive deltas can conflict), SERIAL FALLBACK: apply each
         positive candidate individually via apply_delta_gated in DESCENDING probe-gain order (deterministic
         tie-break: ascending delta_cid) against the progressively-updated trunk — exactly today's serial
         semantics, just greedy-ordered. applied_order records the order actually applied.
      5. A single-candidate batch delegates straight to apply_delta_gated, so it is BYTE-IDENTICAL to the
         serial path (values AND trunk mutation) — the batching adds nothing when only one candidate is ready.

    NUMERICS / DETERMINISM. The weighted sum accumulates in float64 in a FIXED order (candidate list order,
    then each delta's key order) and is cast to float32 ONCE per key. This makes the accumulation STEP
    deterministic given identical weights+deltas (no order-dependent float32 rounding). Note the w_i derive
    from the held-out eval, which is itself host-dependent, so the merged trunk is NOT cross-arch
    bit-reproducible on its own and remains subject to the SAME recompute-verify / cross-arch determinism
    gates as the rest of the merge path (memory safetensors-mmap-recompute-determinism / gpu-testbed-cross-arch).
    Single-candidate delegation inherits apply_delta_gated's exact float32 arithmetic, so it cannot drift by
    even 1 ULP from the serial path.

    Returns a dict: {mode: 'combined'|'serial_fallback'|'all_rejected', accepted: bool, base_val, merged_val,
    per_candidate: [{rec, gain, weight, accepted, keys_matched, delta_norm}], applied_order: [delta_cid,...]}.
    per_candidate is in INPUT order; applied_order is non-empty only for serial_fallback."""
    # ---- single-candidate fast path: EXACT equivalence with the serial merge (values + mutation)
    if len(candidates) == 1:
        rec, delta = candidates[0]
        v = apply_delta_gated(global_trunk_np, delta, eval_fn, outer=outer, margin=margin, apply=apply)
        acc = bool(v["accepted"])
        base_val, merged_val = v["base_val"], v["merged_val"]
        gain = float(base_val - merged_val) if (acc and base_val is not None) else 0.0
        mode = "combined" if acc else "all_rejected"
        pc = {"rec": rec, "gain": gain, "weight": (1.0 if acc else 0.0), "accepted": acc,
              "keys_matched": v["keys_matched"], "delta_norm": v["delta_norm"]}
        return {"mode": mode, "accepted": acc, "base_val": base_val, "merged_val": merged_val,
                "per_candidate": [pc], "applied_order": []}

    base_val = float(eval_fn(global_trunk_np))
    # ---- probe every candidate individually (reuse the single-delta machinery; never mutates the trunk)
    probes = []                                               # (idx, rec, delta, gain, matched_keys, verdict)
    for idx, (rec, delta) in enumerate(candidates):
        pv = apply_delta_gated(global_trunk_np, delta, eval_fn, outer=outer, margin=margin, apply=False)
        mk = [k for k in global_trunk_np if k in delta and
              np.shape(global_trunk_np[k]) == np.shape(delta[k])]
        gain = float(base_val - pv["merged_val"]) if pv["merged_val"] is not None else 0.0
        probes.append({"idx": idx, "rec": rec, "delta": delta, "gain": gain, "keys": mk, "verdict": pv})
    positives = [p for p in probes if p["gain"] > 0.0]

    def _blank_pc(p, gain=0.0, weight=0.0, accepted=False):
        return {"rec": p["rec"], "gain": float(gain), "weight": float(weight), "accepted": bool(accepted),
                "keys_matched": p["verdict"]["keys_matched"], "delta_norm": p["verdict"]["delta_norm"]}

    # ---- no candidate helps held-out on its own: reject the whole batch, trunk untouched
    if not positives:
        return {"mode": "all_rejected", "accepted": False, "base_val": base_val, "merged_val": base_val,
                "per_candidate": [_blank_pc(p) for p in probes], "applied_order": []}

    gsum = sum(p["gain"] for p in positives)
    for p in positives:
        p["weight"] = p["gain"] / gsum                        # w_i = gain_i / sum(positive gains); sum(w)=1

    # ---- combined weighted merge on a COPY (float64 accumulate in fixed order, one float32 cast per key)
    acc64 = {}                                                # key -> float64 weighted-delta accumulator
    for p in positives:                                       # candidate list order == positives' input order
        for k in p["keys"]:
            if k not in acc64:
                acc64[k] = np.zeros(np.shape(global_trunk_np[k]), dtype=np.float64)
            acc64[k] += p["weight"] * p["delta"][k].astype(np.float64)
    combined = dict(global_trunk_np)
    for k, a in acc64.items():
        combined[k] = (global_trunk_np[k].astype(np.float64) + float(outer) * a).astype(np.float32)
    combined_val = float(eval_fn(combined))

    if combined_val <= base_val - margin:                    # combined merge ACCEPTED
        pos_idx = {p["idx"] for p in positives}
        per = []
        for p in probes:
            if p["idx"] in pos_idx:
                per.append(_blank_pc(p, gain=(base_val - combined_val) * p["weight"],
                                     weight=p["weight"], accepted=True))
            else:
                per.append(_blank_pc(p))                      # excluded (gain<=0): weight 0, rejected
        if apply:
            for k in acc64:
                global_trunk_np[k] = combined[k]
        return {"mode": "combined", "accepted": True, "base_val": base_val, "merged_val": combined_val,
                "per_candidate": per, "applied_order": []}

    # ---- SERIAL FALLBACK: greedy descending-gain apply_delta_gated against the progressively-updated trunk
    order = sorted(positives, key=lambda p: (-p["gain"], str(p["rec"].get("delta_cid", ""))))
    work = global_trunk_np if apply else {k: v.copy() for k, v in global_trunk_np.items()}
    running = base_val
    applied_order, serial_result = [], {}
    for p in order:
        sv = apply_delta_gated(work, p["delta"], eval_fn, outer=outer, margin=margin, apply=True)
        applied_order.append(str(p["rec"].get("delta_cid", "")))
        if sv["accepted"]:
            serial_result[p["idx"]] = (running - sv["merged_val"], True)   # attributed = drop this step made
            running = sv["merged_val"]
        else:
            serial_result[p["idx"]] = (0.0, False)
    per = []
    for p in probes:
        if p["idx"] in serial_result:
            g, ok = serial_result[p["idx"]]
            per.append(_blank_pc(p, gain=g, weight=p.get("weight", 0.0), accepted=ok))
        else:
            per.append(_blank_pc(p))                          # excluded (gain<=0)
    return {"mode": "serial_fallback", "accepted": any(ok for _g, ok in serial_result.values()),
            "base_val": base_val, "merged_val": running, "per_candidate": per, "applied_order": applied_order}


# --------------------------------------------------------------------------- goodput telemetry (read-only)
# Decoupled-DiLoCo (DeepMind) treats GOODPUT -- the fraction of fleet work that lands as useful ACCEPTED
# work -- as a headline metric. The coordinator already MAKES DiLoCo candidate verdicts (accept / reject /
# committee-veto / corpus-drop) but never ACCUMULATES them, so a soak leaves no numbers behind. These three
# PURE helpers let the round loop tally verdicts into one counters dict and summarize it. They NEVER feed any
# merge/reward/gate decision -- reporting only (evidence for issue #126).
def goodput_counters():
    """Fresh zeroed goodput/batch counters for ONE coordinator run (cumulative across all rounds; never reset).

    Goodput proxy: accepted candidates / processed candidates (see goodput_summary). Integer fields:
      processed             : DiLoCo candidates that reached a merge OR committee-veto verdict.
      accepted              : of those, merged (held-out gate accepted).
      rejected_gate         : of those, rejected by the held-out gate.
      vetoed                : candidates the committee VETOED before the merge (also counted in `processed`).
      corpus_dropped        : candidates dropped at the corpus-sha gate (never fetched/merged; NOT `processed`).
      batches               : batch-merge invocations (apply_deltas_weighted_gated calls).
      batch_combined        : batch invocations whose result mode was 'combined'.
      batch_serial_fallback : batch invocations whose result mode was 'serial_fallback'.
      batch_all_rejected    : batch invocations whose result mode was 'all_rejected'.
      batch_deferrals       : grace-window defer decisions (a defer is NOT processing).
      batch_candidates      : total candidates folded across all batch invocations."""
    return {"processed": 0, "accepted": 0, "rejected_gate": 0, "vetoed": 0, "corpus_dropped": 0,
            "batches": 0, "batch_combined": 0, "batch_serial_fallback": 0, "batch_all_rejected": 0,
            "batch_deferrals": 0, "batch_candidates": 0}


def goodput_summary(c):
    """Read-only snapshot: the derived goodput ratio plus a shallow COPY of every counter. Pure; never mutates.

    Goodput proxy = accepted / processed, where "processed" = candidates that reached a merge/veto verdict
    (corpus-dropped candidates are counted SEPARATELY and are NOT processed; grace-window deferrals are NOT
    processing). goodput is rounded to 4dp, or None when processed == 0 (no ratio is defined yet)."""
    processed = c.get("processed", 0)
    gp = round(c["accepted"] / processed, 4) if processed else None
    return {"goodput": gp, **dict(c)}


def count_batch_result(c, res):
    """Fold ONE apply_deltas_weighted_gated result dict into counters `c` IN PLACE (read-only telemetry).

    `res` shape (from apply_deltas_weighted_gated): {mode: 'combined'|'serial_fallback'|'all_rejected',
    per_candidate: [{accepted, ...}], ...}. Per batch: +batches, +batch_candidates (= len(per_candidate)),
    and +batch_combined/serial_fallback/all_rejected keyed by `mode`. Per candidate reaching the merge:
    +processed and +accepted or +rejected_gate. (Committee-vetoed candidates are excluded from per_candidate
    upstream and are tallied at the veto site instead.) Returns `c`."""
    per = res.get("per_candidate") or []
    c["batches"] += 1
    c["batch_candidates"] += len(per)
    mode = res.get("mode")
    if mode == "combined":
        c["batch_combined"] += 1
    elif mode == "serial_fallback":
        c["batch_serial_fallback"] += 1
    elif mode == "all_rejected":
        c["batch_all_rejected"] += 1
    for pc in per:
        c["processed"] += 1
        if pc.get("accepted"):
            c["accepted"] += 1
        else:
            c["rejected_gate"] += 1
    return c


def _get_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ============================================================ shardDiLoCo Phase-2 (per-expert DiLoCo)
# The genuinely-NEW coordinator glue over today's WHOLE-MODEL #133 lane: gate + merge each EXPERT's
# pseudo-gradient INDEPENDENTLY (docs/research/SHARDDILOCO_DESIGN.md sec 12; binding review D1-D4 in
# sec 10). DEFAULT-OFF: shard_diloco_on() is False unless NEURAHASH_SHARDDILOCO is set, so the #133 lane
# stays byte-identical (the single-delta apply_delta_gated above remains the ONLY merge path). These are
# pure numpy + an eval callback (like apply_delta_gated), so they are unit-testable WITHOUT the 280 KB
# coordinator monolith, exactly like gather_batch / apply_deltas_weighted_gated above.


def shard_diloco_on():
    """NEURAHASH_SHARDDILOCO master gate (off|audit|enforce house style, read as on/off here; mirrors
    corpus_enforce_on() / registry_posture()). Unset / '0' / 'off' -> False: the #133 merge lane is
    byte-identical to today (whole-model trunk, experts FROZEN). Truthy -> True: the coordinator ALSO
    gates + merges per-expert pseudo-gradients via shard_merge_round. Committed code never sets it."""
    return (os.environ.get("NEURAHASH_SHARDDILOCO", "0") or "0").strip().lower() \
        not in ("", "0", "false", "off", "no")


class FlopMeter:
    """D3 FLOP-economics meter. DiPaCo wins wall-clock but is LESS FLOP-efficient per perplexity, and a
    PAID fleet is billed for every redundant FLOP (SHARDDILOCO_DESIGN.md sec 10 D3): the coordinator must
    pay / gate per (held-out gain / FLOP spent), NOT per raw contribution. Records TRAIN FLOPs (the
    contributor inner loop) and VERIFY FLOPs (the coordinator held-out probe) separately. Convention: an
    (n x a)@(a x b) matmul = 2*n*a*b FLOPs; backward ~= 2x forward, so one trained example ~= 3x its
    forward cost. `fwd_flops_per_example` is the model's forward cost for ONE example (model-agnostic --
    the caller computes it, e.g. moelm_sparse_fwd_flops in tools/diloco_contributor.py)."""

    def __init__(self, fwd_flops_per_example):
        self.fwd = float(fwd_flops_per_example)
        self.train = 0.0
        self.verify = 0.0

    def add_train(self, n_examples):
        self.train += 3.0 * self.fwd * float(n_examples)   # forward + backward

    def add_verify(self, n_examples):
        self.verify += self.fwd * float(n_examples)        # forward-only eval

    @property
    def total(self):
        return self.train + self.verify

    def gain_per_flop(self, gain, flops=None):
        """Held-out gain per FLOP. `flops` defaults to self.total (whole run); pass a per-contribution
        FLOP count for the per-contribution economics gate. Returns 0.0 when no FLOPs are recorded.

        REPORTING ONLY -- DO NOT PROMOTE THIS INTO RANKING OR PAYMENT (#144). The per-contribution
        `flops` is dominated by the miner's self-reported `train_flops`, which the contribution
        signature does NOT cover (contrib_canonical_message signs only version/cid/round/name/
        val_before/val_after). Signing it would not help either: a miner simply signs its own lie.
        Remote FLOP counts are not verifiable -- that is verifiable-computation territory.

        Today `minted` is the sum of coordinator-MEASURED held-out gain, which a miner cannot forge,
        and this value feeds no accept/reject or payout path. Keep it that way. If gain/FLOP ever
        needs to influence what anyone is paid, the denominator must be DERIVED by the coordinator
        from quantities it controls, never taken from the record.
        """
        f = self.total if flops is None else float(flops)
        return (float(gain) / f) if f else 0.0


class SecretRotatedProbe:
    """D2 trust primitive: a SECRET, ROTATED held-out probe. Bit-exact recompute-verify CANNOT check an
    H-step LOCAL training trajectory (not cheaply re-derivable), so the shardDiLoCo trust layer =
    held-out accept gate + stake + signed identity + THIS probe (SHARDDILOCO_DESIGN.md sec 10 D2). The
    coordinator keeps a PRIVATE pool of held-out (X, y) per expert that miners never see, and each
    outer-sync round draws a FRESH random subset (rotated by round index) -- so a miner cannot overfit /
    game a fixed PUBLIC probe. apply_delta_gated is evaluated on this secret sample, not a public one."""

    @staticmethod
    def seed_from_env(env_var="NEURAHASH_PROBE_SEED", *, required=False):
        """Resolve the gate seed from a SECRET, never a literal.

        WHY THIS EXISTS (found 2026-07-21, before publishing the coordinator). `batch()` below is
        fully deterministic in (seed, round, pool, size), and `sharddiloco_harness.domain_splits`
        -- which derives the pool -- is ALREADY PUBLIC in the miner repo. So the only thing making
        this probe secret is the seed. Call sites used a hardcoded `seed=777`; publishing the
        coordinator would therefore have let any miner reconstruct the exact probe batch for every
        round, overfit to it, and walk past a gate that exists to catch precisely that. The gate
        would still have LOOKED like it worked -- it rejects garbage on its own merits -- which is
        the dangerous kind of broken.

        Kerckhoffs: publish the algorithm, keep the key. Returns the int seed.

        NOTE for the quorum path: independent verifiers must reproduce the same batch, so this seed
        is shared with the STAKED VERIFIER SET, not with miners. The stronger construction is
        commit-reveal (coordinator publishes H(seed) before the round, reveals after), which removes
        the need to share a standing secret at all -- tracked as the follow-up; this closes the
        published-literal hole today.
        """
        import hashlib as _hashlib
        import os
        import secrets as _secrets
        v = (os.environ.get(env_var) or "").strip()
        if v:
            try:
                return int(v, 0)
            except ValueError:
                return int.from_bytes(_hashlib.sha256(v.encode()).digest()[:8], "big")
        if required:
            raise SystemExit(
                "%s is not set. The held-out gate's secrecy rests entirely on this seed: it is "
                "deterministic in (seed, round, pool, size) and the pool derivation is public, so a "
                "known seed makes the probe reconstructible and the gate gameable. Set it to a "
                "high-entropy secret." % env_var)
        return _secrets.randbits(63)                      # unpredictable; never a literal

    def __init__(self, pools, seed=0, size=256):
        # pools: {expert_index: (X ndarray, y ndarray)}  the coordinator's PRIVATE held-out per expert
        self._pools = pools
        self._seed = int(seed)
        self._size = int(size)

    def batch(self, expert, rnd):
        X, y = self._pools[int(expert)]
        rng = np.random.default_rng(self._seed + int(rnd))   # ROTATES every outer round
        n = min(self._size, len(X))
        idx = rng.integers(0, len(X), size=n)
        return X[idx], y[idx]


def shard_merge_round(trunk, experts, contributions, eval_expert, probe, meter, rnd, *,
                      outer=0.7, margin=0.0, outer_beta=0.9, momentum=None,
                      stream_frac=None, max_stale=None):
    """One shardDiLoCo (per-expert DiLoCo) OUTER-SYNC merge round for the coordinator role -- the
    genuinely-new glue over today's WHOLE-MODEL #133 lane (SHARDDILOCO_DESIGN.md sec 12). PURE except
    for `eval_expert`; safe inside the single-threaded round loop, like apply_delta_gated. Only called
    when shard_diloco_on() (flag off -> the #133 single-delta apply_delta_gated is the ONLY merge path).

    STREAMING-SUBSET TRUNK SYNC (#126, SHARDDILOCO_DESIGN.md sec 4c/5/13). `stream_frac` (None ->
    NEURAHASH_SHARDDILOCO_STREAM_FRAC env; >=1 or <=0 -> 1.0) selects how much of the trunk syncs this
    outer step: 1.0 = the FULL-trunk outer_aggregate, byte-identical to today; a value in (0,1) syncs
    only the rolling fragment `rnd % round(1/frac)` (staggered partition, see stream_publish_trunk /
    outer_aggregate_masked) so per-step trunk bandwidth drops ~ proportional to frac while every trunk
    param is covered over 1/frac steps. Incoming trunk_delta is the compact fragment when frac<1.

    MAX-STALENESS (#126). `max_stale` (None -> NEURAHASH_SHARDDILOCO_MAX_STALENESS env; None/<=0 ->
    unbounded == today) ages out a contribution whose base_round is more than `max_stale` outer rounds
    behind `rnd` BEFORE it touches any canonical weight (staleness_ok); such drops count in `staled`.

      trunk         : {k: ndarray}  canonical trunk, MUTATED IN PLACE by the DiLoCo OUTER step
                      (outer_aggregate over the contributed trunk deltas -- UNGATED, DiLoCo consensus).
      experts       : list[{k: ndarray}]  canonical per-expert params; experts[e] MUTATED IN PLACE iff
                      contribution e's expert delta passes its OWN independent held-out gate.
      contributions : list of dicts, each {miner, expert, trunk_delta, expert_delta, train_flops,
                      verify_ok}. Already signature-verified by the caller (verify_ok=False -> skipped,
                      never touches canonical weights).
      eval_expert(e, cand_expert_e, pX, pY) -> float : held-out loss of a CANDIDATE expert-e param dict
                      on the SECRET probe batch (pX, pY). The caller wires it to the model AND must add
                      its forward FLOPs to `meter` (meter.add_verify(len(pX))) so gain/FLOP is measured.
      probe         : SecretRotatedProbe (D2). meter: FlopMeter (D3). rnd: outer-round index (rotates
                      the probe). momentum: DiLoCo outer momentum buffer (caller owns it across rounds).

    Returns {accepts, rejects, staled, minted, trunk_merged, per_expert:[{miner, expert, accepted,
    base_val, merged_val, gain, verify_flops, train_flops, gain_per_flop, delta_norm}]}. `minted` = sum
    of accepted held-out gains (D3 pay-per-MEASURED-gain); each row's gain_per_flop = gain / (train +
    verify FLOPs). `staled` = contributions aged out by the max-staleness policy (counted in rejects)."""
    from neurahash.training_layer import outer_aggregate   # lazy: keep this module's import graph thin
    if momentum is None:
        momentum = {}
    frac = _stream_frac_env() if stream_frac is None else float(stream_frac)
    if not (0.0 < frac < 1.0):
        frac = 1.0
    ms = _max_staleness_env() if max_stale is None else max_stale
    # ---- signature gate + (#126) max-staleness age-out BEFORE any weights move ----
    valid_sig = [c for c in contributions if c.get("verify_ok", True)]
    valid, staled = [], 0
    for c in valid_sig:
        ok, _r = staleness_ok(c.get("base_round"), rnd, ms)
        if ok:
            valid.append(c)
        else:
            staled += 1
    rejects = (len(contributions) - len(valid_sig)) + staled
    # ---- TRUNK: DiLoCo outer step across contributing miners. frac>=1 -> FULL sync (byte-identical to
    #      today); frac in (0,1) -> streaming SUBSET: reconstruct each miner's compact fragment and apply
    #      a MASKED outer step so only round rnd's fragment moves (#126, SHARDDILOCO_DESIGN.md sec 4c). ----
    trunk_deltas = [c["trunk_delta"] for c in valid if c.get("trunk_delta")]
    trunk_merged = False
    if trunk_deltas:
        if frac >= 1.0:
            outer_aggregate(trunk, trunk_deltas, momentum, outer_lr=outer, beta=outer_beta)
        else:
            nfrag = stream_num_fragments(frac)
            fragment = int(rnd) % nfrag
            shapes = {k: np.shape(trunk[k]) for k in trunk}
            masks = trunk_fragment_mask(shapes, nfrag, fragment)
            recon = [reconstruct_trunk_fragment(td, shapes, nfrag, fragment) for td in trunk_deltas]
            outer_aggregate_masked(trunk, recon, momentum, masks, outer_lr=outer, beta=outer_beta)
        trunk_merged = True
    # ---- EXPERTS: each delta gated INDEPENDENTLY on the SECRET ROTATED probe (D2) ----
    accepts, minted = 0, 0.0
    per_expert = []
    for c in valid:
        e = int(c["expert"])
        pX, pY = probe.batch(e, rnd)
        v0 = meter.verify

        def _eval(cand, _e=e, _pX=pX, _pY=pY):
            return float(eval_expert(_e, cand, _pX, _pY))

        verdict = apply_delta_gated(experts[e], c.get("expert_delta") or {}, _eval,
                                    outer=outer, margin=margin, apply=True)
        vflops = meter.verify - v0
        tflops = float(c.get("train_flops", 0.0))
        gain = 0.0
        if verdict["accepted"]:
            accepts += 1
            gain = max(0.0, (verdict["base_val"] or 0.0) - (verdict["merged_val"] or 0.0))
            minted += gain
        else:
            rejects += 1
        gpf = meter.gain_per_flop(gain, tflops + vflops)
        per_expert.append(dict(miner=c.get("miner"), expert=e, accepted=bool(verdict["accepted"]),
                               base_val=verdict["base_val"], merged_val=verdict["merged_val"],
                               gain=gain, verify_flops=vflops, train_flops=tflops,
                               gain_per_flop=gpf, delta_norm=verdict.get("delta_norm")))
    return dict(accepts=accepts, rejects=rejects, staled=staled, minted=minted,
                trunk_merged=trunk_merged, per_expert=per_expert)


# ============================================================ shardDiLoCo streaming-subset sync (#126)
# Trunk STREAMING-SUBSET (fragment-staggered) sync -- Streaming DiLoCo (DeepMind 2025) applied to the
# shardDiLoCo trunk (SHARDDILOCO_DESIGN.md sec 4c/5/13; == issue #126 "fragment-staggered sync +
# max-staleness", built ONCE for both efforts). Today shard_merge_round syncs the WHOLE trunk pseudo-
# gradient every outer step; that trunk delta is the per-step WAN cost (experts are already sharded).
# Streaming instead syncs only a ROLLING FRACTION of trunk params each outer step, on a deterministic
# STAGGERED schedule, so over 1/frac steps every trunk param is synced exactly once and the per-step
# trunk bandwidth drops ~ proportional to the fraction -- at a bounded convergence cost (the measured
# tradeoff, docs/research/SHARDDILOCO_PHASE2_RESULTS.md). DEFAULT-OFF: NEURAHASH_SHARDDILOCO_STREAM_FRAC
# unset / '' / <=0 / >=1 -> frac 1.0 -> the FULL-trunk outer_aggregate path, byte-identical to today.
#
# Partition (deterministic, staggered): a trunk param element with FLAT index i belongs to fragment
# i % num_fragments; the fragment synced at outer round R is R % num_fragments. Because every i maps to
# exactly one fragment and the schedule cycles through all fragments, num_fragments consecutive rounds
# cover every trunk element exactly once (a test asserts the union == all params). Only the fragment's
# VALUES travel (the indices are derivable from the round), so the wire is ~ frac * the full delta.


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


def trunk_fragment_mask(shapes, num_fragments, fragment):
    """{key: bool ndarray} selecting the elements of each trunk key that belong to `fragment`. The union
    over fragment in range(num_fragments) is all-True for every key (full coverage) -- a test asserts it."""
    out = {}
    for k, shp in shapes.items():
        n = int(np.prod(shp)) if shp else 1
        out[k] = _fragment_sel(n, num_fragments, fragment).reshape(shp)
    return out


def extract_trunk_fragment(trunk_delta, num_fragments, fragment):
    """CONTRIBUTOR side: compact the fragment of a full trunk delta to {key: 1-D ndarray of the selected
    elements in flat order}. Bytes ~ (1/num_fragments) of the full delta -- the streaming bandwidth win.
    reconstruct_trunk_fragment inverts it on the coordinator using the SAME partition rule."""
    out = {}
    for k, arr in trunk_delta.items():
        flat = np.asarray(arr).reshape(-1)
        out[k] = flat[_fragment_sel(flat.size, num_fragments, fragment)]
    return out


def reconstruct_trunk_fragment(fragment_delta, shapes, num_fragments, fragment):
    """COORDINATOR side: scatter a compact fragment back to full-shaped trunk arrays (ZEROS for the
    un-synced elements) so it can feed the DiLoCo outer step. Only this fragment's params are non-zero,
    so only they move this outer step (outer_aggregate_masked keeps the rest byte-untouched)."""
    out = {}
    for k, shp in shapes.items():
        n = int(np.prod(shp)) if shp else 1
        sel = _fragment_sel(n, num_fragments, fragment)
        full = np.zeros(n, dtype=np.float64)
        vals = fragment_delta.get(k)
        if vals is not None:
            full[sel] = np.asarray(vals, dtype=np.float64).reshape(-1)
        out[k] = full.reshape(shp)
    return out


def outer_aggregate_masked(global_params, deltas, momentum_buf, masks, outer_lr=0.7, beta=0.9):
    """DiLoCo outer step (IDENTICAL math to training_layer.outer_aggregate: m = beta*m + avg(deltas);
    p += outer_lr*m) restricted to the boolean `masks` -- the streaming-subset merge (#126). For an
    un-masked (un-synced) trunk element BOTH its params and its momentum are byte-untouched this round,
    so each element evolves by heavy-ball only on ITS fragment's rounds. With an all-True mask this is
    BIT-identical to outer_aggregate (a test asserts it)."""
    if not deltas:
        return global_params, momentum_buf
    for k in global_params:
        present = [d[k] for d in deltas if k in d]
        if not present:
            continue
        avg = np.mean(present, axis=0)
        prev = momentum_buf.get(k)
        if prev is None:
            prev = np.zeros_like(global_params[k])
        newm = beta * prev + avg
        m = masks.get(k) if masks else None
        if m is None:
            momentum_buf[k] = newm
            global_params[k] = global_params[k] + outer_lr * newm
        else:
            momentum_buf[k] = np.where(m, newm, prev)
            global_params[k] = np.where(m, global_params[k] + outer_lr * newm, global_params[k])
    return global_params, momentum_buf


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


def _max_staleness_env():
    v = os.environ.get("NEURAHASH_SHARDDILOCO_MAX_STALENESS", "")
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def max_staleness():
    """NEURAHASH_SHARDDILOCO_MAX_STALENESS (#126 max-staleness policy). Unset / <=0 / non-int -> None ==
    unbounded (today's behavior: a contribution's base_round is NEVER age-checked). A positive int N ->
    a delta trained against a base_round more than N outer rounds behind the coordinator is aged out."""
    return _max_staleness_env()


def staleness_ok(base_round, coord_round, max_stale=None):
    """(#126) Is a contribution trained against `base_round` fresh enough to merge at `coord_round`?
    Pure. Returns (ok, reason). max_stale None/<=0 -> (True, 'no-staleness-bound') == today, byte-
    identical. base_round None -> (True, 'no-base-round') (cannot age-check; never reject on missing
    data). Else fresh iff coord_round - base_round <= max_stale."""
    if max_stale is None or int(max_stale) <= 0:
        return True, "no-staleness-bound"
    if base_round is None:
        return True, "no-base-round"
    age = int(coord_round) - int(base_round)
    if age <= int(max_stale):
        return True, "fresh(age=%d<=%d)" % (age, int(max_stale))
    return False, "stale(age=%d>%d)" % (age, int(max_stale))
