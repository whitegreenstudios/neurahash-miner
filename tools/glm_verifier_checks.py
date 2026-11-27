#!/usr/bin/env python
"""Default-off extra checks for the GLM verifier shadow (tools/glm_verifier_shadow.py), 2026-11-25.

ITEM 3 -- re-derive what payouts rest on:
  CE    the published held-out CE (`heldout_ce`; a mint pays max(0, prev_ce - ce), see
        docs/research/PAYOUT_QUORUM_STEP3_2026-11-24.md). The coordinator recomputes it ONLY on an event
        that folded an accept (sharddiloco_glm_coordinator.py `ce = _chunked_heldout_ce(...) if
        res["accepts"] else cur_ce`); every other event carries the previous number unchanged. So:
          accept event -> recompute with the coordinator's own _chunked_heldout_ce on the replayed state
                          and compare within a pinned tolerance (CPU vs the coordinator's GPU is not
                          bit-exact; docs/research/VERIFIER_CE_OMISSION_2026-11-25.md);
          other event  -> the published number must equal the previous published one (free).
  ROOT  the published `model_root` after the event, recomputed with the contributor's own
        _campaign_seed_into / _slot_digest_into over the coordinator's ORDERED slot list. That order is
        not published as a list; it is reconstructed from every record's `slot` index (+ the coordinate
        named by its `slot_roots` key, + accepted rows). A coordinate the coordinator registered without
        a judged record leaves a hole: the check then says root_unverifiable, never disagree. When the
        live pointer is for the event being checked, its `rounds` key set fixes the slot count and can
        fill one hole.

ITEM 4 -- what the coordinator did NOT do:
  OMITTED      a miner whose valid (secp256k1 signature recovers its own name), lineage-ok
               (_lineage_ok against the verifier's slot-root history), non-stale contribution is NEWER
               than the newest one the coordinator ever judged from it, and was first SEEN by this
               verifier >= window events ago, while the coordinator judged nothing of that miner since.
               One row per miner per episode (re-armed when the miner is judged again).
  CHAIN_BREAK  event counter going back (pointer), a processed accepted record rewritten or removed, a
               record whose `event` field disagrees with its name, a model_root returning to an older
               non-adjacent root, a slot index re-assigned to another coordinate, an event gap, a pointer
               campaign change with no `sharddiloco/glm/seed/<campaign>` marker, and -- feature-detected,
               only when present -- a previous-record hash link that does not match the previous record.

Pure helpers + small state machines; numpy/stdlib at import. Tests: tests/test_glm_verifier_checks.py.
"""
from __future__ import annotations

import hashlib
import json
import re
import types

CE_TOL_DEFAULT = 2e-5          # pinned 2026-11-25 (A/A + CPU-vs-published, see the doc); never loosen silently
# review 2 fixes 2026-11-25: the PAYOUT AMOUNT (prev_ce - ce) is compared directly. The CPU-vs-GPU offset is
# systematic and cancels in the difference (measured r497: amount d 1.6e-07), so the amount bound is
# AMOUNT_TOL_ABS + AMOUNT_TOL_REL * |verifier amount|, far tighter than 2 x CE_TOL (which let one in-tol nudge
# per accept publish a +40% basis).
# review 3 fixes 2026-11-25 (C-N6): RELATIVE FIRST. The bound is max(AMOUNT_TOL_ABS, AMOUNT_TOL_REL x |amount|):
# the absolute term is only a FLOOR for tiny amounts, set at ~3x the measured honest amount gap (1.6e-7), no
# longer added on top. The old 2e-6 + 2% let a 1e-5 gain publish +22%; now it is +5% at 1e-5, +2% from 2.5e-5 up,
# and 5e-7 absolute on a zero gain (was ~2e-6).
AMOUNT_TOL_ABS = 5e-7
AMOUNT_TOL_REL = 0.02
UNDECIDED_BACKOFF_MAX = 64     # events between re-fetches of an unfetchable record, doubling from 1
POINTER_NAME = "sharddiloco/glm/pointer"
SEED_FMT = "sharddiloco/glm/seed/%s"
# Previous-record link fields the coordinator may add (hash-chained accepted records, parallel work). Checked
# only when present; the first one found wins.
PREV_LINK_KEYS = ("prev_record_sha256", "prev_record_hash", "prev_sha256", "prev_hash")
SIG_KEYS = ("coord_sig", "coordinator_sig", "record_sig", "sig")
_CG_RE = re.compile(r"^cg/([0-9a-f]+)/r(\d+)/(.+?)\.(\d+)$")
RECORD_MAX_BYTES = 1 << 20     # the coordinator reads every record under H.RECORD_MAX_BYTES (same default)
JSON_MAX_DEPTH = 64            # records are a few levels deep; 100k-deep nesting crashed json.loads


# ============================================================================ safe parsing (review fix 2026-11-25)
class UnreadableRecord(ValueError):
    """A relay body this verifier refuses to parse: over the byte cap, nested too deep, not UTF-8 / JSON."""


class ObjectTooLarge(ValueError):
    """A relay object larger than the caller's cap (declared Content-Length, or more bytes than allowed)."""


def _json_depth_ok(raw, max_depth):
    """Bracket depth of a JSON text, ignoring brackets inside strings, without recursion."""
    depth, in_str, esc = 0, False, False
    for ch in raw:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
            if depth > max_depth:
                return False
        elif ch in "]}":
            depth -= 1
    return True


def safe_loads(raw, max_bytes=RECORD_MAX_BYTES, max_depth=JSON_MAX_DEPTH):
    """bytes|str -> parsed JSON, or UnreadableRecord. Never raises anything else (RecursionError,
    UnicodeDecodeError, MemoryError and json errors all become UnreadableRecord)."""
    try:
        if raw is None:
            raise UnreadableRecord("no body")
        if len(raw) > int(max_bytes):
            raise UnreadableRecord("body %d bytes > cap %d" % (len(raw), int(max_bytes)))
        txt = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        if not _json_depth_ok(txt, int(max_depth)):
            raise UnreadableRecord("JSON nested deeper than %d" % int(max_depth))
        return json.loads(txt)
    except UnreadableRecord:
        raise
    except (ValueError, RecursionError, MemoryError, TypeError) as ex:      # UnicodeDecodeError is a ValueError
        raise UnreadableRecord("%s: %s" % (type(ex).__name__, str(ex)[:120]))


def amount_tol(amount_verifier, abs_tol=None, rel_tol=None):
    """Allowed |published amount - verifier amount|: max(abs floor, rel x |verifier amount|) (review 3 fixes
    2026-11-25, C-N6; was abs + rel)."""
    a = AMOUNT_TOL_ABS if abs_tol is None else float(abs_tol)
    r = AMOUNT_TOL_REL if rel_tol is None else float(rel_tol)
    return max(a, r * abs(float(amount_verifier)))


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def record_shape_error(rec):
    """None when an ACCEPTED record has the shape the shadow walk reads, else a reason. A record that fails
    this is reported and the walk pauses on it (never half-processed, never a crash)."""
    if not isinstance(rec, dict):
        return "record is %s, not a JSON object" % type(rec).__name__
    j = rec.get("judged")
    if j is not None and not isinstance(j, dict):
        return "judged is not an object"
    if isinstance(j, dict):
        g = j.get("gate")
        if g is not None and not isinstance(g, dict):
            return "judged.gate is not an object"
        rows = (g or {}).get("rows")
        if rows is not None and (not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows)):
            return "judged.gate.rows is not a list of objects"
        for r in rows or []:
            for k in ("base_val", "merged_val"):
                if r.get(k) is not None and not _is_num(r.get(k)):
                    return "judged.gate.rows[].%s is not a number" % k
        if (g or {}).get("margin") is not None and not _is_num(g.get("margin")):
            return "judged.gate.margin is not a number"
        if j.get("name") and rows:
            try:
                int(rec["event"])
            except (KeyError, TypeError, ValueError):
                return "judged record without an integer event"
    acc = rec.get("accepted")
    if acc is not None and (not isinstance(acc, list) or not all(isinstance(a, dict) for a in acc)):
        return "accepted is not a list of objects"
    for i, a in enumerate(acc or []):              # review 2: every row the replay applies must be applicable
        for k in ("layer", "glm_expert"):
            if not isinstance(a.get(k), int) or isinstance(a.get(k), bool):
                return "accepted[%d].%s is not an integer" % (i, k)
        if not isinstance(a.get("cid"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", a.get("cid")):
            return "accepted[%d].cid is not a 64-hex content id" % i
        if a.get("outer") is not None and not _is_num(a.get("outer")):
            return "accepted[%d].outer is not a number" % i
    sr = rec.get("slot_roots")
    if sr is not None and not isinstance(sr, dict):
        return "slot_roots is not an object"
    if rec.get("heldout_ce") is not None and not _is_num(rec.get("heldout_ce")):
        return "heldout_ce is not a number"
    if rec.get("model_root") is not None and not isinstance(rec.get("model_root"), str):
        return "model_root is not a string"
    return None


# ============================================================================ CE
def make_ce_fn(C, G, eval_chunk, layer_major=False):
    """fn(model, ids) -> float: the coordinator's _chunked_heldout_ce(G, model, ids, eval_chunk), imported,
    not copied. layer_major swaps G.heldout_ce for glm_verifier_eval.heldout_ce_rows_layer_major(..)[0]
    (pinned == heldout_ce on the same device, tests/test_glm_verifier_eval.py), so lazy fp32 does not
    re-upcast every layer once per sub-batch."""
    if layer_major:
        import glm_verifier_eval as GE
        shim = types.SimpleNamespace(heldout_ce=lambda m, ids: GE.heldout_ce_rows_layer_major(m, ids)[0])
    else:
        shim = types.SimpleNamespace(heldout_ce=G.heldout_ce)
    chunk = int(eval_chunk)
    return lambda model, ids: float(C._chunked_heldout_ce(shim, model, ids, chunk))


# ============================================================================ ROOT
def ordered_model_root(N, host, coords):
    """N.model_root(host) computed over `coords` in THAT order (the coordinator's slot order), with the
    contributor's own digest pieces. Registers a coordinate missing from the verifier's host (pristine)."""
    h = hashlib.sha256()
    N._campaign_seed_into(h, N.host_campaign_id(host))
    for (L, E) in coords:
        idx = host.index_of(L, E)
        if idx is None:
            idx = host.register(L, E)
        N._slot_digest_into(h, host, idx)
    return h.hexdigest()


def record_slot_pairs(rec):
    """[(slot_index, (L, E))] that one accepted record reveals about the coordinator's slot order."""
    out = []
    for a in rec.get("accepted") or []:
        try:
            out.append((int(a["slot"]), (int(a["layer"]), int(a["glm_expert"]))))
        except (KeyError, TypeError, ValueError):
            continue
    sr = rec.get("slot_roots") or {}
    if rec.get("slot") is not None and len(sr) == 1:
        try:
            L, E = (int(x) for x in next(iter(sr)).split("_"))
            out.append((int(rec["slot"]), (L, E)))
        except (TypeError, ValueError):
            pass
    return out


class SlotOrder:
    """The coordinator's ordered slot list as far as public records reveal it."""

    def __init__(self, startup_coords):
        self.by_idx, self.by_coord = {}, {}
        for i, c in enumerate(startup_coords):
            self.by_idx[i] = (int(c[0]), int(c[1]))
            self.by_coord[self.by_idx[i]] = i

    def learn(self, idx, coord):
        """-> (grew, conflict_reason|None)."""
        idx, coord = int(idx), (int(coord[0]), int(coord[1]))
        have = self.by_idx.get(idx)
        if have is not None:
            if have != coord:
                return False, "slot %d was L%dE%d, record says L%dE%d" % (idx, have[0], have[1], coord[0], coord[1])
            return False, None
        other = self.by_coord.get(coord)
        if other is not None and other != idx:
            return False, "L%dE%d was slot %d, record says slot %d" % (coord[0], coord[1], other, idx)
        self.by_idx[idx], self.by_coord[coord] = coord, idx
        return True, None

    def contiguous(self):
        n = 0
        while n in self.by_idx:
            n += 1
        return n

    def prefix(self, n_target=None):
        """(coords, gap_reason|None) for the first n slots (n = n_target or the known contiguous prefix)."""
        n = self.contiguous()
        if n_target is not None:
            if n < int(n_target):
                return None, "slot %d unknown (pointer has %d slots)" % (n, int(n_target))
            n = int(n_target)
        elif any(k >= n for k in self.by_idx):
            return None, "slot %d unknown (records name slot %d)" % (n, max(self.by_idx))
        return [self.by_idx[i] for i in range(n)], None


# ============================================================================ signature / lineage
def make_sig_ok(C, N):
    """fn(body, miner, base_event) -> bool: the coordinator's keyless open-admission check (the recovered
    secp256k1 signer's derived name must equal `miner`), imported. Keyed-roster (HMAC) miners cannot be
    verified without their key: they return False and are never counted as OMITTED."""
    def sig_ok(body, miner, base_event):
        a = C._open_admission_recover(body.get("sig"), body.get("expert_cid"), base_event, miner)
        return a is not None and N.derive_glm_miner_name(a) == miner
    return sig_ok


class _Committed:
    """root_hist stand-in for C._lineage_ok: every event 0..cur is committed (the verifier walked them)."""

    def __init__(self, cur):
        self.cur = int(cur)

    def get(self, b, default=None):
        return "committed" if 0 <= int(b) <= self.cur else default


def make_lineage_ok(C):
    """fn(hist, coord, base_event, cur, base_slot_root) -> 'ok' | 'bad' | 'undecided', with the
    coordinator's own C._expected_slot_root + C._lineage_ok (imported). `hist` is the VERIFIER's slot-root
    history {(L, E): [(event, root)]}. A record without base_slot_root (pre-Shard-Claim client), an unknown
    coordinate or a not-yet-committed base event is undecided, never counted."""
    def lineage(hist, coord, base_event, cur, base_slot_root):
        if base_slot_root is None:
            return "undecided"
        want = C._expected_slot_root(hist, coord, base_event)
        ok, why = C._lineage_ok(base_event, None, cur, _Committed(cur), base_slot_root=base_slot_root,
                                want_slot_root=want)
        if ok:
            return "ok"
        return "bad" if why in C._TERMINAL_LINEAGE_REASONS else "undecided"
    return lineage


def make_delta_ok(C, G, relay, host, H=None):
    """fn(body) -> None | (verdict, why) for OmissionTracker (review fix 2026-11-25, finding 19). Applies the
    coordinator's admission checks on the delta blob that can be re-run from public data, with its own pieces
    (imported): byte budget C._delta_read_budget(delta_bytes), unpack (H.unpack_arrays), and for a LoRA payload
    G.validate_lora_factors against the resident expert dims. A blob the coordinator would drop terminally
    -> ("bad", ...) (not an omission); an unfetchable blob -> ("undecided", ...) (maybe not written yet).
    NOT covered (documented): the dense-LoRA guard and the materialized-merge path; such records can still be
    reported omitted. None = all re-runnable checks pass."""
    if H is None:
        import sharddiloco_harness as H

    def dims(L, E):
        idx = host.index_of(L, E)
        W = host.read_slot(0 if idx is None else idx)
        I, Hd = W["gate"].shape
        return int(I), int(Hd)

    def delta_ok(body):
        cid = str(body.get("expert_cid") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", cid):
            return ("bad", "coordinator would drop: no valid expert_cid")
        budget = int(C._delta_read_budget(body.get("delta_bytes")))
        capped = getattr(relay, "obj_capped", None)
        try:
            # review 2 fixes 2026-11-25: never pull a whole blob before the budget check -- a capped reader checks
            # the declared size first and stops streaming at budget + 1 bytes
            raw = capped(cid, budget) if capped is not None else relay.obj(cid)
        except ObjectTooLarge as ex:
            return ("bad", "coordinator would drop: delta > read budget %d (%s)" % (budget, ex))
        except Exception as ex:                          # noqa: BLE001 -- not fetchable (yet)
            return ("undecided", "delta %s.. not fetchable: %s" % (cid[:12], type(ex).__name__))
        if len(raw) > budget:
            return ("bad", "coordinator would drop: delta %d bytes > read budget %d" % (len(raw), budget))
        try:
            d = H.unpack_arrays(raw)
            if G.is_lora_payload(d):
                ok, why = G.validate_lora_factors(d, *dims(int(body["layer"]), int(body["glm_expert"])))
                if not ok:
                    return ("bad", "coordinator would drop: LoRA factor shape gate rejected (%s)" % why)
        except Exception as ex:                          # noqa: BLE001 -- the coordinator drops on any fetch error
            return ("bad", "coordinator would drop: delta unreadable (%s)" % type(ex).__name__)
        return None
    return delta_ok


def parse_cg_name(name, campaign):
    m = _CG_RE.match(str(name))
    if not m or m.group(1) != str(campaign):
        return None
    return {"base_event": int(m.group(2)), "miner": m.group(3), "k": int(m.group(4))}


def recency(be, k):
    return (int(be), int(k))


# ============================================================================ OMITTED
class OmissionTracker:
    """See the module docstring. `fetch_json(name)`, `sig_ok`, `lineage(body, coord, base_event, cur)` ->
    'ok' | 'bad' | 'undecided', `max_stale` (None = unbounded, like dm.max_staleness())."""

    def __init__(self, campaign, window, fetch_json, sig_ok, lineage, max_stale=None, claimable=None,
                 delta_ok=None):
        self.campaign, self.window = str(campaign), int(window)
        self.fetch_json, self.sig_ok, self.lineage = fetch_json, sig_ok, lineage
        self.max_stale = None if max_stale is None or int(max_stale) <= 0 else int(max_stale)
        self.claimable = claimable
        # delta_ok(body) -> None | (verdict, why): the coordinator's own admission checks on the delta blob
        # (fetchable, under budget, right factor shape). A record it would drop is not an omission (review fix
        # 2026-11-25, finding 19); an unfetchable blob is undecided, never omitted.
        self.delta_ok = delta_ok
        self.first_seen = {}          # name -> head event when this verifier first saw it
        self.by_miner = {}            # miner -> [(recency, name)]
        self.judged = set()
        self.newest_judged = {}       # miner -> recency of the newest judged record
        self.open = {}                # miner -> name flagged in the current episode
        self._valid = {}              # name -> (verdict, why)
        self._retry = {}              # name -> (next event to retry an undecided fetch, backoff) -- review 2
        self.counts = {"undecided": 0, "unreadable": 0, "would_drop": 0}

    def observe(self, names, head):
        pre = "cg/%s/" % self.campaign
        for n in names:
            if n.startswith(pre) and n not in self.first_seen:
                p = parse_cg_name(n, self.campaign)
                if p is None:
                    continue
                self.first_seen[n] = int(head)
                self.by_miner.setdefault(p["miner"], []).append((recency(p["base_event"], p["k"]), n))

    def note_judged(self, rec):
        if not isinstance(rec, dict):
            return
        names = []
        j = rec.get("judged") or {}
        if isinstance(j, dict) and j.get("name"):
            names.append(str(j["name"]))
        acc = rec.get("accepted") or []
        names += [str(a["name"]) for a in (acc if isinstance(acc, list) else []) if isinstance(a, dict)
                  and a.get("name")]
        for n in names:
            self.judged.add(n)
            p = parse_cg_name(n, self.campaign)
            if p is None:
                continue
            r = recency(p["base_event"], p["k"])
            if r > self.newest_judged.get(p["miner"], (-1, -1)):
                self.newest_judged[p["miner"]] = r
                self.open.pop(p["miner"], None)            # served again: a new episode may start

    def _validity(self, name, miner, cur):
        hit = self._valid.get(name)
        if hit is not None and hit[0] != "undecided":
            return hit
        nxt = self._retry.get(name)
        if hit is not None and nxt is not None and int(cur) < nxt[0]:
            return hit                                   # review 2: negative result cached, exponential backoff
        try:
            v = self._validity_inner(name, miner, cur)
        except UnreadableRecord as ex:                   # junk body: a coordinator drops it terminally too
            self.counts["unreadable"] += 1
            v = ("bad", "unreadable body: %s" % ex)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError, MemoryError) as ex:
            v = ("bad", "malformed body: %s" % type(ex).__name__)
        except Exception as ex:                          # noqa: BLE001 -- relay / network trouble: retry later
            v = ("undecided", "fetch failed: %s" % type(ex).__name__)
        self._valid[name] = v
        if v[0] == "undecided":
            b = min(UNDECIDED_BACKOFF_MAX, 2 * nxt[1]) if nxt is not None else 1
            self._retry[name] = (int(cur) + b, b)
        else:
            self._retry.pop(name, None)
        return v

    def _validity_inner(self, name, miner, cur):
        body = self.fetch_json(name)
        if not isinstance(body, dict):
            v = ("bad", "body not a JSON object")
        elif str(body.get("miner", miner)) != miner:
            v = ("bad", "body miner != name miner")
        else:
            try:
                be = int(body.get("base_event", body.get("base_round")))
                coord = (int(body["layer"]), int(body["glm_expert"]))
            except (KeyError, TypeError, ValueError):
                v = ("bad", "no base_event / coordinate")
            else:
                if self.claimable is not None and not self.claimable(*coord):
                    v = ("bad", "coordinate not hostable")
                elif self.max_stale is not None and cur - be > self.max_stale:
                    v = ("bad", "stale (max_stale %d)" % self.max_stale)
                elif not self.sig_ok(body, miner, be):
                    v = ("bad", "signature does not recover the miner's name")
                else:
                    lv = self.lineage(body, coord, be, cur)
                    v = (lv, "lineage %s" % lv)
                    if lv == "ok" and self.delta_ok is not None:
                        d = self.delta_ok(body)
                        if d is not None:
                            v = d
                            if d[0] == "bad":
                                self.counts["would_drop"] += 1
        return v

    def evaluate(self, cur):
        """Rows to emit at processed event `cur`."""
        out = []
        for miner, cands in self.by_miner.items():
            if miner in self.open:
                continue
            nj = self.newest_judged.get(miner, (-1, -1))
            pend = [(r, n) for r, n in cands if n not in self.judged and r > nj
                    and self.first_seen[n] + self.window <= int(cur)]
            if not pend:
                continue
            for r, n in sorted(pend, reverse=True):          # newest first: the one the queue would hold
                v, why = self._validity(n, miner, cur)
                if v == "undecided":
                    self.counts["undecided"] += 1
                if v == "ok":
                    self.open[miner] = n
                    out.append({"event": int(cur), "kind": "omission", "class": "omitted", "miner": miner,
                                "record": n, "first_seen_event": self.first_seen[n], "window": self.window,
                                "newest_judged": list(nj), "pending_valid_or_unchecked": len(pend),
                                "reason": ("valid contribution first seen at event %d, nothing of %s judged "
                                           "through event %d" % (self.first_seen[n], miner, int(cur)))})
                    break
        return out


# ============================================================================ CHAIN_BREAK
def _canon_sha(rec):
    body = {k: v for k, v in rec.items() if k not in SIG_KEYS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _audit_hash(rec):
    """accepted_audit.record_hash(rec), or None when it cannot be computed (untrusted record)."""
    try:
        from neurahash import accepted_audit as _AA
        if not _AA.depth_ok(rec):
            return None
        return _AA.record_hash(rec)
    except Exception:                                    # noqa: BLE001 -- untrusted bytes
        return None


class ChainTracker:
    """See the module docstring. Every method returns a list of rows (possibly empty)."""

    def __init__(self, campaign):
        self.campaign = str(campaign)
        self.rec_sha = {}            # event -> manifest sha256 of the record we processed
        self.reported = set()
        self.max_ptr_event = None
        self.roots = {}              # model_root -> first event it was published at
        self.prev_root = None
        self.prev = None             # (event, manifest sha, record) of the last processed record

    def _row(self, cur, key, reason):
        if key in self.reported:
            return []
        self.reported.add(key)
        return [{"event": int(cur), "kind": "chain", "class": "chain_break", "reason": reason}]

    def observe_manifest(self, names, cur, accepted_name):
        out = []
        for e, sha in self.rec_sha.items():
            v = names.get(accepted_name(e))
            got = None if v is None else (v["sha256"] if isinstance(v, dict) else v)
            if got is None:
                out += self._row(cur, ("removed", e), "accepted record r%d was removed from the relay" % e)
            elif got != sha:
                out += self._row(cur, ("rewritten", e, got), "accepted record r%d was rewritten (%s.. -> %s..)"
                                 % (e, sha[:12], str(got)[:12]))
        return out

    def observe_pointer(self, ptr, cur, names, last_event):
        out = []
        if not isinstance(ptr, dict):
            return out
        camp = str(ptr.get("campaign_id") or "")
        if camp and camp != self.campaign:
            if SEED_FMT % camp in names:
                key = ("rotated", camp)
                if key not in self.reported:
                    self.reported.add(key)
                    out.append({"event": int(cur), "kind": "chain", "class": "campaign_rotated",
                                "reason": "pointer moved to campaign %s (seed marker published)" % camp})
            else:
                out += self._row(cur, ("campaign", camp), "pointer campaign changed %s -> %s with no published "
                                 "seed marker" % (self.campaign, camp))
            return out
        try:
            pe = int(ptr.get("event"))
        except (TypeError, ValueError):
            return out
        if self.max_ptr_event is not None and pe < self.max_ptr_event:
            out += self._row(cur, ("ptr_back", pe), "pointer event went back %d -> %d" % (self.max_ptr_event, pe))
        # the pointer is written right after the record; one event of lag is the commit window
        if last_event is not None and pe < int(last_event) - 1:
            out += self._row(cur, ("ptr_behind", pe), "pointer event %d is behind processed record r%d"
                             % (pe, int(last_event)))
        self.max_ptr_event = pe if self.max_ptr_event is None else max(self.max_ptr_event, pe)
        return out

    def observe_record(self, e, rec, sha):
        out = []
        try:
            re_ = int(rec.get("event"))
        except (TypeError, ValueError):
            re_ = None
        if re_ != int(e):
            out += self._row(e, ("evfield", e), "record r%d carries event %r" % (e, rec.get("event")))
        for k in PREV_LINK_KEYS:
            if k in rec:
                if self.prev is not None and self.prev[0] == int(e) - 1:
                    want = str(rec[k] or "").lower()
                    if want not in (self.prev[1], _canon_sha(self.prev[2])):
                        out += self._row(e, ("link", e), "record r%d %s=%s.. does not hash-link r%d"
                                         % (e, k, want[:12], e - 1))
                break
        # review 3 fixes 2026-11-25 (#23): the coordinator's hash chain lives in the AUDIT block
        # (neurahash/accepted_audit.py): rec["audit"]["prev_hash"] = accepted_audit.record_hash(r<e-1>). A None
        # prev_hash is a reset (accepted_audit.verify_chain judges its marker); anything else must link r<e-1>.
        aud = rec.get("audit")
        if isinstance(aud, dict) and aud.get("prev_hash") is not None and self.prev is not None \
                and self.prev[0] == int(e) - 1:
            want = str(aud.get("prev_hash") or "").lower()
            if want != _audit_hash(self.prev[2]):
                out += self._row(e, ("audit_link", e), "record r%d audit.prev_hash=%s.. does not hash-link r%d"
                                 % (e, want[:12], e - 1))
        root = rec.get("model_root")
        if root:
            first = self.roots.get(root)
            if first is not None and root != self.prev_root and first < int(e) - 1:
                out += self._row(e, ("revisit", e), "model_root at r%d returns to the root of r%d (not adjacent)"
                                 % (e, first))
            self.roots.setdefault(root, int(e))
            self.prev_root = root
        self.rec_sha[int(e)] = sha
        self.prev = (int(e), sha, rec)
        return out

    def gap(self, expected, got):
        # review 2 fixes 2026-11-25: keyed `at_event` (r<expected> is NOT processed yet), so --resume never
        # skips it
        rows = self._row(expected, ("gap", expected), "event gap: expected r%d, next published is r%d"
                         % (expected, got))
        for r in rows:
            r["at_event"] = r.pop("event")
        return rows

    def slot_conflict(self, e, why):
        return self._row(e, ("slot", why), "slot order changed: %s" % why)
