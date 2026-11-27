#!/usr/bin/env python
"""STEP 2 of "no coordinator": a SHADOW verifier for the LIVE GLM lane (2026-11-23).

It re-scores every gate decision the live coordinator publishes, on this machine's CPU in fp32, and appends one
JSONL row per judged event: what the coordinator decided, what an independent re-score decides, and why they
differ. It is DRY-RUN BY CONSTRUCTION: this file has no PUT, no publish flag, never reads a relay token, and its
relay client refuses any method but GET (tests/test_glm_verifier_shadow.py::test_runner_cannot_put).

WHAT IS PUBLIC, AND WHAT IS RECONSTRUCTED. The live coordinator runs with Stage 1 OFF (no --mint-evidence), so it
publishes no committee proposals (`glm-committee/...`) and no per-mint evidence. What it DOES publish, per event,
is the accepted record `sharddiloco/glm/accepted/<campaign>/r<event>`:
  judged.{name, accepted, gate.{margin, rows[{base_val, merged_val, delta_norm, expert}]}}, accepted[{layer,
  glm_expert, cid, outer, ...}], model_root, slot_roots{"L_E": root}, heldout_ce.
and the miner's signed contribution record (judged.name) with {layer, glm_expert, expert_cid, address, ...}.
From those this module RECONSTRUCTS a proposal in the Stage-2 shape (neurahash/glm_committee.py,
tools/glm_verifier_eval.py) and labels it `evidence.reconstructed = true`. A reconstructed proposal is weaker than
a coordinator-published one: nothing binds the coordinator to it, and the coordinator's probe rows are not in it.

THE PROBE. The coordinator's gate rows are `SecretRotatedProbe(pool, seed=NEURAHASH_PROBE_SEED).batch(e, event)`
over its coordinator-only pool. The pool file is the same 512-row `ids_daily_probe.npy` (sha256 pinned by
coord_freeze.json); the SEED is a secret this module never looks for. Two modes:
  independent (default): the verifier draws its OWN 256 rows (with replacement, rotated per event) from the same
  pool with its own random seed (kept in a local state file, never published). Agreement then measures what an
  independent committee member would conclude -- sampling noise included. The coordinator's published base_val
  is NOT reproducible in this mode.
  exact: the operator exports the shared seed in an env var and passes --probe-seed-env NAME. The rows equal the
  coordinator's, and base_val must reproduce (checked, |d| <= 1e-6). Sharing the seed with verifiers is an OWNER
  decision (security custody); this module only reads the variable name it is given.
  full_pool (--probe-full-pool, 2026-11-23 owner "score the whole pool"): every pool row once, fixed order, no
  seed -- matches a coordinator running NEURAHASH_SD_PROBE_FULL_POOL=1, so base_val must reproduce (checked as in
  exact mode) and each row carries probe_rows_sha256 to compare with the coordinator's boot line.

THE DECISION, both sides:
  coordinator  accept | reject-margin (published gain < margin) | veto-after-margin (gain >= margin but not
               accepted: Z-VETO, DRIFT CAP, held-out tol or canary -- the record does not say which)
  verifier     accept | reject-margin | z-veto | drift-cap  (paired-z on the verifier's rows; the drift cap is
               EXACT: same weights, same pristine packs, no probe involved). Held-out tol and the canary latch
               are not re-checked (the held-out pool check is offered once per run: --heldout-check).
The chain state always FOLLOWS THE COORDINATOR: a coordinator accept is applied (W += outer * delta, rounded to
the slot dtype, exactly GlmGateEvaluator.advance) and checked against the published slot root.

Import-light at module level (numpy + stdlib). The real-model build lives in build_real(); tests drive
ShadowCore with a tiny GLM and a fake relay. Doc: docs/research/VERIFIER_SHADOW_STEP2_2026-11-23.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RELAY = "http://47.84.93.96:8710"
TOL_EXACT = 1e-6            # exact mode: |base_val verifier - coordinator| must be within this


def mask_addr(a):
    a = str(a or "")
    return "0x<w>" if a.lower().startswith("0x") else a


# ============================================================================ read-only relay
class ReadOnlyRelay:
    """GET-only client for the content store. There is deliberately no put/publish method and no token
    argument; `_get` refuses any Request whose method is not GET. WAN-only (glm_committee.assert_wan_url)."""

    def __init__(self, url=RELAY, timeout=60, opener=None):
        from neurahash.glm_committee import assert_wan_url
        self.url = assert_wan_url(str(url).rstrip("/"))
        self.timeout = float(timeout)
        self._open = opener or urllib.request.urlopen
        self._manifest = {}

    def _get(self, path):
        req = urllib.request.Request(self.url + path, method="GET")
        if req.get_method() != "GET":                       # belt and braces: nothing else ever leaves here
            raise RuntimeError("ReadOnlyRelay refuses %s" % req.get_method())
        with self._open(req, timeout=self.timeout) as r:
            return r.read()

    def refresh(self):
        self._manifest = json.loads(self._get("/manifest"))
        return self._manifest

    def names(self):
        return self._manifest

    def obj(self, sha):
        sha = str(sha).strip().lower()
        body = self._get("/o/" + sha)
        if hashlib.sha256(body).hexdigest() != sha:
            raise ValueError("relay object %s.. does not hash to its name" % sha[:12])
        return body

    def obj_capped(self, sha, max_bytes):
        """obj() that never holds more than max_bytes + 1 bytes (review 2 fixes 2026-11-25): a declared
        Content-Length over the cap is refused before reading, and the body is streamed and abandoned past the
        cap. Raises glm_verifier_checks.ObjectTooLarge."""
        import glm_verifier_checks as K
        sha, cap = str(sha).strip().lower(), int(max_bytes)
        req = urllib.request.Request(self.url + "/o/" + sha, method="GET")
        if req.get_method() != "GET":
            raise RuntimeError("ReadOnlyRelay refuses %s" % req.get_method())
        with self._open(req, timeout=self.timeout) as r:
            cl = None
            try:
                cl = int((getattr(r, "headers", None) or {}).get("Content-Length"))
            except (TypeError, ValueError):
                cl = None
            if cl is not None and cl > cap:
                raise K.ObjectTooLarge("declared %d bytes > cap %d" % (cl, cap))
            chunks, n = [], 0
            while True:
                b = r.read(min(1 << 16, cap + 1 - n))
                if not b:
                    break
                chunks.append(b)
                n += len(b)
                if n > cap:
                    raise K.ObjectTooLarge("more than %d bytes" % cap)
        body = b"".join(chunks)
        if hashlib.sha256(body).hexdigest() != sha:
            raise ValueError("relay object %s.. does not hash to its name" % sha[:12])
        return body

    def by_name(self, name):
        v = self._manifest.get(name)
        if v is None:
            return None
        return self.obj(v["sha256"] if isinstance(v, dict) else v)

    def json_by_name(self, name):
        b = self.by_name(name)
        return None if b is None else json.loads(b)

    def json_by_name_safe(self, name, max_bytes=None):
        """json_by_name for UNTRUSTED names (review fix 2026-11-25): the manifest-declared size is checked
        before the GET, the body after it, and parsing is depth-bounded. Raises
        glm_verifier_checks.UnreadableRecord for junk; None when the name is absent."""
        import glm_verifier_checks as K
        cap = int(K.RECORD_MAX_BYTES if max_bytes is None else max_bytes)
        v = self._manifest.get(name)
        if v is None:
            return None
        size = v.get("size") if isinstance(v, dict) else None
        if isinstance(size, int) and not isinstance(size, bool) and size > cap:
            raise K.UnreadableRecord("declared size %d > cap %d" % (size, cap))
        return K.safe_loads(self.obj(v["sha256"] if isinstance(v, dict) else v), max_bytes=cap)


# ============================================================================ lane reconstruction
def accepted_name(campaign, event):
    return "sharddiloco/glm/accepted/%s/r%d" % (campaign, int(event))


def published_events(names, campaign):
    """Sorted event numbers with an accepted record in the manifest."""
    pre = "sharddiloco/glm/accepted/%s/r" % campaign
    out = []
    for k in names:
        if k.startswith(pre):
            t = k[len(pre):]
            if t.isdigit():
                out.append(int(t))
    return sorted(out)


def parse_event(rec):
    """Normalise one accepted record into what the shadow needs. Returns None for an event with no judged
    delta (nothing to re-score)."""
    j = rec.get("judged") or {}
    rows = ((j.get("gate") or {}).get("rows") or [])
    if not j.get("name") or not rows:
        return None
    row = rows[0]
    bv, mv = row.get("base_val"), row.get("merged_val")
    margin = (j.get("gate") or {}).get("margin")
    acc = list(rec.get("accepted") or [])
    return {"event": int(rec["event"]), "name": str(j["name"]), "miner": j.get("miner"),
            "accepted": bool(j.get("accepted")), "base_val": None if bv is None else float(bv),
            "merged_val": None if mv is None else float(mv), "margin": None if margin is None else float(margin),
            "delta_norm": row.get("delta_norm"), "slot_idx": rec.get("slot"), "accepted_rows": acc,
            "model_root": rec.get("model_root"), "slot_roots": dict(rec.get("slot_roots") or {}),
            "heldout_ce": rec.get("heldout_ce"),
            "veto_reason": j.get("veto_reason") or j.get("veto") or (j.get("gate") or {}).get("veto_reason")}


def coordinator_decision(ev):
    """(decision, gain). gain = published base_val - merged_val on the coordinator's own rows."""
    gain = None if ev["base_val"] is None else ev["base_val"] - ev["merged_val"]
    if ev["accepted"]:
        return "accept", gain
    if gain is None or ev["margin"] is None or gain < ev["margin"]:
        return "reject-margin", gain
    return "veto-after-margin", gain


def reconstruct_proposal(ev, contrib, campaign, outer, margin, probe_meta):
    """A Stage-2 proposal rebuilt from public records, labelled reconstructed. `probe_meta` = {mode, size,
    batch_sha256} of the rows THIS verifier will score on."""
    L, E = int(contrib["layer"]), int(contrib["glm_expert"])
    return {"v": 1, "ns": campaign, "height": int(ev["event"]), "delta_cid": str(contrib["expert_cid"]),
            "recipient": contrib.get("address"), "miner": contrib.get("miner"), "amount": None,
            "claimed": {"base_val": ev["base_val"], "merged_val": ev["merged_val"], "accepted": ev["accepted"]},
            "evidence": {"reconstructed": True, "campaign": campaign, "coord": [L, E], "outer": float(outer),
                         "margin": float(margin), "probe": dict(probe_meta), "record_name": ev["name"],
                         "event": int(ev["event"])}}


def verifier_decision(gate, drift_why):
    if not gate.get("margin_accepted"):
        return "reject-margin"
    if gate.get("z_vetoed"):
        return "z-veto"
    if drift_why is not None:
        return "drift-cap"
    return "accept"


def compare(coord_dec, ver_dec, veto_reason=None):
    """(agree, reason, cls). Agreement is on the binary accept/not-accept outcome; the reason names the gap.

    cls (2026-11-24, owner "count canary halts correctly") sorts the outcome for the tally:
      agree           same outcome.
      disagree        a real gap in what the verifier re-checks (margin, paired-z, drift cap).
      canary_halt     coordinator vetoed after the margin, the record names the canary latch as the reason,
                      and the verifier's own gate passed: the gates AGREE, merging was paused.
      coord_only_veto coordinator vetoed after the margin with NO reason in the record, while the verifier's
                      margin, paired-z AND drift checks all passed. What is left is the canary latch or the
                      held-out tolerance, neither re-checkable from public records -- so it is not counted as a
                      disagreement, and not as an agreement either."""
    agree = (coord_dec == "accept") == (ver_dec == "accept")
    if agree:
        return True, None, "agree"
    if coord_dec == "accept":
        return False, "coordinator accepted; verifier %s" % ver_dec, "disagree"
    if coord_dec == "veto-after-margin" and ver_dec == "accept":
        r = str(veto_reason or "").lower()
        if "canary" in r:
            return False, "coordinator canary halt (merging paused); verifier gate passed", "canary_halt"
        if not r:
            return False, ("coordinator veto-after-margin with no reason published (canary latch or held-out "
                           "tol, not re-checkable); verifier margin, paired-z and drift passed"), "coord_only_veto"
    return False, "coordinator %s%s; verifier accepted" % (
        coord_dec, (" (%s)" % veto_reason) if veto_reason else ""), "disagree"


# ============================================================================ probe sources
class ProbeSource:
    """Rows the verifier scores on: SecretRotatedProbe over the pool, rotated by the EVENT number exactly as
    the coordinator's shard_merge_round(rnd = clock.event). mode 'independent' (own seed), 'exact' (the shared
    seed), or 'full_pool' (2026-11-23: every pool row once, fixed order, no seed -- the SAME
    SecretRotatedProbe(full_pool=True) the coordinator builds under NEURAHASH_SD_PROBE_FULL_POOL=1, so the rows
    and therefore base_val reproduce exactly; `size` is then the pool length)."""

    def __init__(self, pool, seed, size, mode, dm=None):
        if dm is None:
            from neurahash import diloco_merge as dm
        self.mode = str(mode)
        full = self.mode == "full_pool"
        self._p = dm.SecretRotatedProbe({0: (pool, pool)}, seed=0 if full else int(seed), size=int(size),
                                        full_pool=full)
        self.size = int(self._p.gate_row_count(0))
        self._dm = dm

    def rows(self, event):
        return self._p.batch(0, int(event))

    def digest(self, X, y):
        """dm.full_pool_digest of a batch: the same digest the coordinator logs at boot in full-pool mode."""
        return self._dm.full_pool_digest(X, y)


# ============================================================================ memory guard
class MemGuard:
    """Abort the process (os._exit(3)) if available physical RAM drops under `floor_gib`. A shadow must never
    squeeze the miners or the owner's desktop. `peak_rss_gib` is tracked for the cost report."""

    def __init__(self, floor_gib=8.0, period_s=0.5, log=print, exit_fn=None):
        import psutil
        self._ps = psutil
        self.floor = float(floor_gib)
        self.period = float(period_s)
        self.log = log
        self.exit_fn = exit_fn or (lambda: os._exit(3))
        self.peak_rss = 0
        self.min_avail = None
        self._stop = threading.Event()
        self._t = None

    def sample(self):
        rss = self._ps.Process().memory_info().rss
        avail = self._ps.virtual_memory().available
        self.peak_rss = max(self.peak_rss, rss)
        self.min_avail = avail if self.min_avail is None else min(self.min_avail, avail)
        if avail < self.floor * (1 << 30):
            self.log("[shadow] MEMGUARD: available %.2f GiB < floor %.2f GiB (rss %.2f GiB) -- aborting"
                     % (avail / 2 ** 30, self.floor, rss / 2 ** 30))
            self.exit_fn()
            return False
        return True

    def start(self):
        self.sample()

        def run():
            while not self._stop.wait(self.period):
                self.sample()
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()

    def peak_rss_gib(self):
        return self.peak_rss / 2 ** 30

    def min_avail_gib(self):
        return None if self.min_avail is None else self.min_avail / 2 ** 30


# ============================================================================ the shadow core
def _ce_skip(stats, reason):
    stats["ce_skipped"] = stats.get("ce_skipped", 0) + 1
    return {"class": "ce_skipped", "reason": reason}


class ShadowCore:
    """Follow one campaign's published events in order. Events < rescore_from are only REPLAYED (coordinator
    accepts applied); events >= rescore_from are RE-SCORED, then replayed.

    evaluator : GlmGateEvaluator whose eval_expert was built with rows=True (paired-z).
    relay     : ReadOnlyRelay (or a fake with refresh/names/obj/json_by_name).
    probe     : ProbeSource.
    drift     : object with .check(coord, cand, pre) -> None | reason (the coordinator's DriftCap), or None.
    slot_root : fn(host, idx) -> hex (contributor.slot_root).
    """

    def __init__(self, evaluator, relay, probe, campaign, out_path, *, outer=0.7, margin=1e-5, paired_z=3.0,
                 drift=None, slot_root=None, rescore_from=1, log=print, clock=time.perf_counter, memguard=None,
                 check_ce=False, ce_fn=None, heldout=None, ce_tol=None, check_root=False, N=None,
                 omission_window=0, omission_max_stale=None, sig_ok=None, lineage_ok=None, chain_check=False,
                 omission_delta_ok=None, attester=None):
        self.ev, self.relay, self.probe = evaluator, relay, probe
        # step 4 stage S1 (2026-11-26, default None = unchanged): neurahash.glm_shadow_attester.ShadowAttester. It
        # sees each row AFTER it is written; it never changes a row, a decision or the walk.
        self.attester = attester
        self.campaign, self.out_path = str(campaign), out_path
        self.outer, self.margin, self.paired_z = float(outer), float(margin), float(paired_z)
        self.drift, self.slot_root = drift, slot_root
        self.rescore_from = int(rescore_from)
        self.log, self.clock, self.memguard = log, clock, memguard
        self.next_event = 1
        self.stats = {"replayed_accepts": 0, "rescored": 0, "agree": 0, "disagree": 0, "canary_halt": 0,
                      "coord_only_veto": 0, "errors": 0}
        # ---- 2026-11-25 default-off checks (tools/glm_verifier_checks.py). All off -> nothing below runs,
        # no extra GET, no extra registration, no extra row field: output byte-identical (tested).
        self.check_ce, self.check_root = bool(check_ce), bool(check_root)
        self.omission_window, self.chain_check = int(omission_window or 0), bool(chain_check)
        self._extra = self.check_ce or self.check_root or self.omission_window > 0 or self.chain_check
        if not self._extra:
            return
        import glm_verifier_checks as K
        self._K = K
        self.N = N
        if self.check_ce:
            if ce_fn is None or heldout is None:
                raise ValueError("check_ce needs ce_fn and heldout")
            self.ce_fn, self.heldout = ce_fn, heldout
            self.ce_tol = float(K.CE_TOL_DEFAULT if ce_tol is None else ce_tol)
            self.stats.update(ce_agree=0, ce_disagree=0, ce_skipped=0)
        if self.check_root:
            if N is None:
                raise ValueError("check_root needs N (sharddiloco_glm_contributor)")
            self.stats.update(root_agree=0, root_disagree=0, root_unverifiable=0)
        if self.omission_window > 0:
            if sig_ok is None or lineage_ok is None:
                raise ValueError("omission_window needs sig_ok and lineage_ok")
            host = self.ev.host
            fetch = getattr(self.relay, "json_by_name_safe", None) or self.relay.json_by_name
            self._omit = K.OmissionTracker(self.campaign, self.omission_window, fetch, sig_ok,
                                           self._lineage_verdict, max_stale=omission_max_stale,
                                           claimable=getattr(host, "is_claimable", None),
                                           delta_ok=omission_delta_ok)
            self._lineage_ok = lineage_ok          # fn(base_event, cur, base_slot_root, want) -> (ok, why)
            self._slot_hist = {}                   # (L, E) -> [(event, verifier slot root)]
            self.stats.update(omitted=0)
        if self.chain_check:
            self._chain = K.ChainTracker(self.campaign)
            self.stats.update(chain_break=0)
        if self.check_root or self.omission_window > 0:
            self._order = K.SlotOrder(list(self.ev.host.slots))
        self._prev_pub_ce = self._prev_pub_root = self._prev_ver_ce = None
        self._root_state, self._root_taint = None, False
        self._ptr, self._head, self._last_event = None, 0, None
        self._held = set()                         # (event, sha|reason) already reported as a walk pause
        self.stats.update(unreadable_records=0, walk_paused=0)
        self._paused = None                        # review 2: {"at_event", "reason", "since"} while paused
        self._broken = None                        # review 2: replay state unusable -> permanent pause

    # ------------------------------------------------------------------ helpers
    def _fetch(self, cid):
        return self.relay.obj(cid)

    def _coord_idx(self, L, E):
        h = self.ev.host
        idx = h.index_of(L, E)
        if idx is None:
            idx = h.register(L, E)
        return int(idx)

    def _apply(self, ev, deltas=None):
        """Apply the coordinator's accept(s) of this event and check the published slot root."""
        fetch = self._fetch if deltas is None else (lambda cid: deltas[str(cid)] if str(cid) in deltas
                                                    else self._fetch(cid))
        for a in ev["accepted_rows"]:
            L, E = int(a["layer"]), int(a["glm_expert"])
            idx = self._coord_idx(L, E)
            self.ev.advance({"coord": [L, E], "outer": float(a.get("outer", self.outer)),
                             "delta_cid": a["cid"]}, fetch)
            want = ev["slot_roots"].get("%d_%d" % (L, E))
            if want and self.slot_root is not None:
                got = self.slot_root(self.ev.host, idx)
                if got != want:
                    raise RuntimeError("REPLAY DIVERGED at event %d L%dE%d: slot root %s.. vs published %s.."
                                       % (ev["event"], L, E, got[:12], str(want)[:12]))
            self.stats["replayed_accepts"] += 1

    def _drift_check(self, proposal, ev_gate):
        if self.drift is None:
            return None, "off"
        L, E = proposal["evidence"]["coord"]
        idx = self._coord_idx(L, E)
        W = {k: np.array(v, dtype=np.float32, copy=True) for k, v in self.ev.host.read_slot(idx).items()}
        delta, keys = self.ev._decode(self._fetch(proposal["delta_cid"]), W)
        cand = dict(W)
        for k in keys:
            cand[k] = W[k] + self.outer * np.asarray(delta[k], dtype=np.float32)
        return self.drift.check((L, E), cand, W), "on"

    def rescore(self, ev):
        contrib = self.relay.json_by_name(ev["name"])
        if contrib is None:
            raise ValueError("contribution record %s not on the relay" % ev["name"])
        X, y = self.probe.rows(ev["event"])
        from glm_verifier_eval import batch_digest
        pmeta = {"mode": self.probe.mode, "size": int(self.probe.size), "batch_sha256": batch_digest((X, y)),
                 "rotation": "none" if self.probe.mode == "full_pool" else "event"}
        prop = reconstruct_proposal(ev, contrib, self.campaign, self.outer, self.margin, pmeta)
        self.ev.probe_rows = lambda _p, _X=X, _y=y: (_X, _y)
        L, E = prop["evidence"]["coord"]
        self._coord_idx(L, E)
        t0 = self.clock()
        gate = self.ev(prop, self._fetch, paired_z=self.paired_z)
        t_gate = self.clock() - t0
        drift_why, drift_state = (None, "skipped") if not gate.get("margin_accepted") or gate.get("z_vetoed") \
            else self._drift_check(prop, gate)
        ver = verifier_decision(gate, drift_why)
        cdec, cgain = coordinator_decision(ev)
        agree, why, cls = compare(cdec, ver, ev.get("veto_reason"))
        vgain = gate["base_val"] - gate["merged_val"]
        pz = gate.get("paired_z") or {}
        row = {"event": ev["event"], "coord": [L, E], "slot_idx_coord": ev["slot_idx"],
               "miner": ev["miner"], "address": mask_addr(contrib.get("address")), "record": ev["name"],
               "delta_cid": prop["delta_cid"], "reconstructed": True, "probe_mode": self.probe.mode,
               "coordinator": {"decision": cdec, "gain": cgain, "base_val": ev["base_val"],
                               "merged_val": ev["merged_val"], "margin": ev["margin"]},
               "verifier": {"decision": ver, "gain": vgain, "base_val": gate["base_val"],
                            "merged_val": gate["merged_val"], "z": pz.get("z"), "z_min": self.paired_z,
                            "unique_rows": pz.get("n_unique", pz.get("n")), "margin_accepted": gate["margin_accepted"],
                            "z_vetoed": gate["z_vetoed"], "drift": drift_state, "drift_why": drift_why},
               "agree": bool(agree), "class": cls, "disagreement": why, "gate_s": round(t_gate, 3)}
        if self.probe.mode == "full_pool" and hasattr(self.probe, "digest"):
            row["probe_rows_sha256"] = self.probe.digest(X, y)     # compare with the coordinator's boot line
        if self.probe.mode in ("exact", "full_pool") and ev["base_val"] is not None:
            d = abs(gate["base_val"] - ev["base_val"])
            row["exact_base_val_match"] = bool(d <= TOL_EXACT)
            row["exact_base_val_diff"] = d
        if self.memguard is not None:
            row["peak_rss_gib"] = round(self.memguard.peak_rss_gib(), 3)
        return row

    def _emit(self, row):
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if self.attester is not None:
            self.attester.observe(dict(row))         # a copy: the attester can never edit the core's row

    # ------------------------------------------------------------------ one pass
    def step(self, upto=None):
        """Process every published event from next_event up to `upto` (inclusive) or the newest. Returns the
        number of events processed."""
        if self._extra:
            return self._step_extra(upto)
        self.relay.refresh()
        evs = [e for e in published_events(self.relay.names(), self.campaign) if e >= self.next_event]
        if upto is not None:
            evs = [e for e in evs if e <= int(upto)]
        n = 0
        for e in evs:
            if e != self.next_event:
                raise RuntimeError("event gap: expected r%d, next published is r%d" % (self.next_event, e))
            rec = self.relay.json_by_name(accepted_name(self.campaign, e))
            ev = parse_event(rec)
            if ev is not None and e >= self.rescore_from:
                try:
                    row = self.rescore(ev)
                except Exception as ex:                     # noqa: BLE001 -- a shadow logs, never stops the walk
                    self.stats["errors"] += 1
                    row = {"event": e, "error": "%s: %s" % (type(ex).__name__, ex), "record": ev["name"],
                           "coordinator": {"decision": coordinator_decision(ev)[0]}}
                else:
                    self.stats["rescored"] += 1
                    self.stats[row.get("class") or ("agree" if row["agree"] else "disagree")] += 1
                self._emit(row)
                self.log("[shadow] event %d %s" % (e, json.dumps(
                    {k: row.get(k) for k in ("agree", "class", "disagreement", "gate_s", "error")})))
            if ev is not None and ev["accepted_rows"]:
                self._apply(ev)
            self.next_event = e + 1
            n += 1
        return n

    # ------------------------------------------------------------------ 2026-11-25 default-off checks
    def _emit_extra(self, rows):
        for r in rows:
            k = r.get("class")
            if k in self.stats:
                self.stats[k] += 1
            self._emit(r)
            self.log("[shadow] %s %s" % (k, json.dumps({x: r.get(x) for x in ("event", "miner", "record", "reason")})))

    def _lineage_verdict(self, body, coord, base_event, cur):
        return self._lineage_ok(self._slot_hist, coord, base_event, cur, body.get("base_slot_root"))

    def _note_slot_root(self, coord, event):
        h = self.ev.host
        idx = h.index_of(*coord)
        if idx is None:
            try:
                idx = h.register(*coord)
            except (ValueError, RuntimeError):
                return                                          # not hostable / no seat: lineage stays undecided
        self._slot_hist.setdefault(tuple(coord), []).append((int(event), self.slot_root(h, idx)))

    def _learn_slots(self, e, rec):
        """-> (grew, conflict_reason|None). Also seeds the slot-root history of a newly seen coordinate with
        its weights BEFORE this event's accept (they are pristine: nothing merged it before we saw it)."""
        grew, conflict = False, None
        for idx, coord in self._K.record_slot_pairs(rec):
            g, why = self._order.learn(idx, coord)
            grew = grew or g
            conflict = conflict or why
            if self.omission_window > 0 and tuple(coord) not in self._slot_hist:
                self._note_slot_root(coord, e - 1)
        return grew, conflict

    def _fill_from_pointer(self, keys):
        coords = set()
        for k in keys:
            try:
                coords.add(tuple(int(x) for x in str(k).split("_")))
            except ValueError:
                return
        unknown = coords - set(self._order.by_coord)
        missing = [i for i in range(len(keys)) if i not in self._order.by_idx]
        if len(unknown) == 1 and len(missing) == 1:
            self._order.learn(missing[0], next(iter(unknown)))

    def _ce_check(self, rec, applied):
        """Review fix 2026-11-25 (finding 17). The coordinator carries cur_ce BIT-EXACTLY on an event with no
        accept (sharddiloco_glm_coordinator.py `ce = ... if res["accepts"] else cur_ce`, merge-rollback
        `ce = prev`), so a carried value must be EXACTLY equal. It legitimately re-measures without an accept
        only on a product-judge reject (rolls back + re-scores) and at boot (init_ce); a moved carry is
        therefore RE-MEASURED on the unchanged replayed state and compared within tol to the verifier's own
        value, never to the previous published one (that let +tol per event accumulate: 4 nudges + one
        accept published a payout basis +76% above the true gain). On accept events the payout amount
        max(0, prev - ce) is classified too: |published - verifier| > 2 tol is ce_disagree. Skips counted."""
        pub, prev = rec.get("heldout_ce"), self._prev_pub_ce
        if pub is None:
            return _ce_skip(self.stats, "no published heldout_ce")
        pub = float(pub)
        if applied:
            t0 = self.clock()
            v = float(self.ce_fn(self.ev.host.model, self.heldout))
            d = v - pub
            ok = abs(d) <= self.ce_tol
            out = {"mode": "recomputed", "published": pub, "verifier": v, "diff": d, "tol": self.ce_tol,
                   "ce_s": round(self.clock() - t0, 3)}
            why = "published held-out CE %.9f, verifier %.9f (|d| %.2e > tol %.1e)" % (pub, v, abs(d), self.ce_tol)
            if prev is not None:
                out["amount_published"] = max(0.0, float(prev) - pub)
                if self._prev_ver_ce is not None:
                    out["amount_verifier"] = max(0.0, self._prev_ver_ce - v)
                    da = out["amount_published"] - out["amount_verifier"]
                    out["amount_diff"] = da
                    # review 2 fixes 2026-11-25: the AMOUNT has its own (relative + absolute) bound; 2 x tol let
                    # one in-tol nudge per accept inflate the paid basis by 40% with every CE agreeing
                    import glm_verifier_checks as _K
                    atol = _K.amount_tol(out["amount_verifier"])
                    out["amount_tol"] = atol
                    if ok and abs(da) > atol:
                        ok = False
                        why = ("payout amount published %.9g vs verifier %.9g (|d| %.2e > amount tol %.2e)"
                               % (out["amount_published"], out["amount_verifier"], abs(da), atol))
                else:
                    out["amount_unverified"] = "no verifier CE for the previous event yet"
            self._prev_ver_ce = v
        else:
            if prev is None:
                return _ce_skip(self.stats, "no previous published CE to carry")
            d = pub - float(prev)
            if d == 0.0:
                ok = True
                out = {"mode": "carried", "published": pub, "previous": float(prev), "diff": d,
                       "tol": self.ce_tol, "exact": True}
                why = ""
            else:
                t0 = self.clock()
                v = float(self.ce_fn(self.ev.host.model, self.heldout))
                dv = v - pub
                ok = abs(dv) <= self.ce_tol
                out = {"mode": "remeasured", "published": pub, "previous": float(prev), "diff": d,
                       "verifier": v, "verifier_diff": dv, "tol": self.ce_tol, "exact": False,
                       "ce_s": round(self.clock() - t0, 3)}
                self._prev_ver_ce = v
                why = ("no accepted merge, yet the published CE moved %.9f -> %.9f; re-measured %.9f (|d| %.2e "
                       "> tol %.1e)" % (float(prev), pub, v, abs(dv), self.ce_tol))
        out["class"] = "ce_agree" if ok else "ce_disagree"
        if not ok:
            out["reason"] = why
        self.stats[out["class"]] += 1
        return out

    def _root_check(self, e, rec, changed, slot_err):
        K, h = self._K, self.ev.host
        pub = rec.get("model_root")
        out, n_target = {}, None
        ptr = self._ptr
        if (isinstance(ptr, dict) and str(ptr.get("campaign_id") or "") == self.campaign
                and str(ptr.get("event")) == str(e) and ptr.get("model_root") == pub):
            keys = list((ptr.get("rounds") or {}).keys())
            n_target = len(keys)
            self._fill_from_pointer(keys)
            out["pointer_slots"] = n_target
        if slot_err is None:                                    # published per-coordinate roots, every event
            for k, want in (rec.get("slot_roots") or {}).items():
                try:
                    idx = h.index_of(*(int(x) for x in k.split("_")))
                except ValueError:
                    continue
                if idx is not None and want and self.slot_root(h, idx) != want:
                    slot_err = "slot root %s: verifier %s.. vs published %s.." % (
                        k, self.slot_root(h, idx)[:12], str(want)[:12])
        if slot_err is not None:
            cls, out["reason"] = "root_disagree", slot_err
        elif not pub:
            out.update(mode="skipped", reason="no published model_root")
            return dict(out, **{"class": "root_skipped"})
        elif not changed and n_target is None and pub == self._prev_pub_root \
                and self._root_state in ("agree", "unverifiable"):
            cls, out["mode"] = "root_" + self._root_state, "carried"
        elif not changed and n_target is None and self._root_state == "agree":
            cls, out["mode"] = "root_unverifiable", "moved"
            out["reason"] = "model_root moved with no accepted merge and no newly named slot (a coordinate " \
                            "registered without a judged record, or a forged root)"
            self._root_taint = True
        else:
            coords, gap = self._order.prefix(n_target)
            if gap is not None:
                cls, out["reason"], out["mode"] = "root_unverifiable", gap, "gap"
                self._root_taint = True
            else:
                t0 = self.clock()
                got = K.ordered_model_root(self.N, h, coords)
                out.update(mode="computed", slots=len(coords), verifier=got, root_s=round(self.clock() - t0, 3))
                if got == pub:
                    cls, self._root_taint = "root_agree", False
                elif self._root_taint and n_target is None:
                    cls, out["reason"] = "root_unverifiable", "mismatch after an unexplained root move"
                else:
                    cls, out["reason"] = "root_disagree", "model_root %s.. vs published %s.." % (got[:12], str(pub)[:12])
        out["published"] = pub
        out["class"] = cls
        self._root_state = cls[len("root_"):] if cls != "root_disagree" else "disagree"
        self.stats[cls] += 1
        return out

    def _pause(self, event, key, reason):
        """Report a walk pause once per (event, key); the walk retries on the next step."""
        if self._paused is None or self._paused.get("at_event") != int(event) \
                or self._paused.get("reason") != reason:
            self._paused = {"at_event": int(event), "reason": reason, "since": time.time()}
        if (int(event), key) in self._held:
            return
        self._held.add((int(event), key))
        self.stats["walk_paused"] += 1
        self._emit({"at_event": int(event), "kind": "walk", "class": "walk_paused", "reason": reason})
        self.log("[shadow] walk_paused %s" % reason)

    def status(self):
        """Operator status (review 2 fixes 2026-11-25): state running | paused | broken, plus why/where."""
        st = {"state": "running", "next_event": int(self.next_event), "stats": dict(self.stats)}
        if getattr(self, "_paused", None) is not None:
            st.update(state="broken" if getattr(self, "_broken", None) else "paused", **self._paused)
        return st

    def _prepare(self, ev):
        """Validate + PREFETCH every delta the replay will apply, before any state moves (review 2 fixes
        2026-11-25). Raises on anything unusable; the caller pauses the walk at this event (retried next pass,
        nothing applied twice)."""
        cache = {}
        for a in ev["accepted_rows"]:
            int(a["layer"]), int(a["glm_expert"]), float(a.get("outer", self.outer))
            cid = str(a["cid"])
            if cid not in cache:
                cache[cid] = self.relay.obj(cid)
        return cache

    def _step_extra(self, upto=None):
        """step() with the default-off checks; the flag-off path above is untouched."""
        if self._broken is not None:
            self._pause(self._paused["at_event"] if self._paused else self.next_event, "broken", self._broken)
            return 0
        n = self._step_extra_inner(upto)
        if self._paused is not None and self._broken is None and self.next_event > self._paused["at_event"]:
            self._paused = None                    # the walk moved past the event it paused on
        return n

    def _step_extra_inner(self, upto=None):
        K = self._K
        self.relay.refresh()
        names = self.relay.names()
        pub = published_events(names, self.campaign)
        self._head = pub[-1] if pub else self._head
        if self.omission_window > 0:
            self._omit.observe(names, self._head)
        self._ptr = None
        if self.chain_check or self.check_root:
            try:
                self._ptr = self.relay.json_by_name(K.POINTER_NAME)
            except Exception:                                   # noqa: BLE001 -- a missing pointer is not an error
                self._ptr = None
        if self.chain_check:
            cur = self._last_event or 0
            self._emit_extra(self._chain.observe_manifest(names, cur, lambda ev: accepted_name(self.campaign, ev))
                             + self._chain.observe_pointer(self._ptr, cur, names, self._last_event))
        evs = [e for e in pub if e >= self.next_event]
        if upto is not None:
            evs = [e for e in evs if e <= int(upto)]
        n = 0
        for e in evs:
            if e != self.next_event:
                # review fix 2026-11-25: relay content (a junk future r<N> name) must not kill the shadow; the
                # walk PAUSES here (reported once) and resumes when r<next_event> appears.
                if self.chain_check:
                    self._emit_extra(self._chain.gap(self.next_event, e))
                self._pause(self.next_event, "gap:%d" % e, "event gap: expected r%d, next published is r%d"
                            % (self.next_event, e))
                break
            nm = accepted_name(self.campaign, e)
            v = names.get(nm)
            sha = str(v["sha256"] if isinstance(v, dict) else v)
            try:
                rec = self.relay.json_by_name_safe(nm)
                bad = K.record_shape_error(rec)
            except K.UnreadableRecord as ex:
                rec, bad = None, "unreadable: %s" % ex
            if bad is not None:
                self.stats["unreadable_records"] += 1
                self._pause(e, sha, "accepted record r%d (%s..) unusable: %s" % (e, sha[:12], bad))
                break
            try:                                                # review 2: validate + prefetch BEFORE any state
                _ev0 = parse_event(rec)
                deltas = self._prepare(_ev0) if _ev0 is not None else {}
            except Exception as ex:                             # noqa: BLE001 -- review 2: never a crash
                self.stats["errors"] += 1
                self._emit({"at_event": e, "kind": "walk", "class": "event_error",
                            "error": "%s: %s" % (type(ex).__name__, str(ex)[:200]),
                            "reason": "accepted rows of r%d cannot be replayed (nothing applied; retried)" % e})
                self._pause(e, "prep:" + sha, "accepted record r%d (%s..): its accepted rows cannot be replayed "
                            "(%s: %s)" % (e, sha[:12], type(ex).__name__, str(ex)[:120]))
                break
            try:
                self._process(e, rec, sha, deltas)
            except Exception as ex:                             # noqa: BLE001 -- review 2: never a crash
                self.stats["errors"] += 1
                self._broken = ("replay state unusable after r%d (%s: %s); restart the shadow from scratch"
                                % (e, type(ex).__name__, str(ex)[:160]))
                self._emit({"at_event": e, "kind": "walk", "class": "event_error",
                            "error": "%s: %s" % (type(ex).__name__, str(ex)[:200]), "reason": self._broken})
                self._pause(e, "broken", self._broken)
                break
            n += 1
        return n

    def _process(self, e, rec, sha, deltas):
        if True:                                                # (body kept at its old indent: minimal diff)
            after = []
            if self.chain_check:
                after += self._chain.observe_record(e, rec, sha)
            grew, conflict = (False, None)
            if self.check_root or self.omission_window > 0:
                grew, conflict = self._learn_slots(e, rec)
                if conflict and self.chain_check:
                    after += self._chain.slot_conflict(e, conflict)
            ev = parse_event(rec)
            checked = e >= self.rescore_from
            row = None
            if ev is not None and checked:
                try:
                    row = self.rescore(ev)
                except Exception as ex:                         # noqa: BLE001 -- a shadow logs, never stops the walk
                    self.stats["errors"] += 1
                    row = {"event": e, "error": "%s: %s" % (type(ex).__name__, ex), "record": ev["name"],
                           "coordinator": {"decision": coordinator_decision(ev)[0]}}
                else:
                    self.stats["rescored"] += 1
                    self.stats[row.get("class") or ("agree" if row["agree"] else "disagree")] += 1
            applied = bool(ev is not None and ev["accepted_rows"])
            slot_err = None
            if applied:
                try:
                    self._apply(ev, deltas)
                except RuntimeError as ex:
                    if not (self.check_root and checked and "REPLAY DIVERGED" in str(ex)):
                        raise
                    slot_err = str(ex)                          # counted as root_disagree, the walk goes on
                if self.omission_window > 0:
                    for a in ev["accepted_rows"]:
                        self._note_slot_root((int(a["layer"]), int(a["glm_expert"])), e)
            if checked and (self.check_ce or self.check_root):
                if row is None:
                    row = {"event": e, "kind": "checks"}
                if self.check_ce:
                    row["ce_check"] = self._ce_check(rec, applied)
                if self.check_root:
                    row["root_check"] = self._root_check(e, rec, applied or grew or bool(conflict), slot_err)
            self._prev_pub_ce, self._prev_pub_root = rec.get("heldout_ce"), rec.get("model_root")
            if self.omission_window > 0:
                self._omit.note_judged(rec)
                after += self._omit.evaluate(e)
            if row is not None:
                self._emit(row)
                self.log("[shadow] event %d %s" % (e, json.dumps(
                    {k: row.get(k) for k in ("agree", "class", "disagreement", "gate_s", "error")}
                    | {k: (row.get(k) or {}).get("class") for k in ("ce_check", "root_check") if k in row})))
            self._emit_extra(after)
            self.next_event, self._last_event = e + 1, e


# ============================================================================ real build (5090 PC, CPU fp32)
def build_real(a, log=print):
    """The coordinator's own chain on CPU: build_node_model (pieces 0-23), GlmExpertLaneHost with the live
    claimable set, bind_campaign_id, fp32 upcast (GlmGateEvaluator fp32=True, same recipe as
    _gate_fp32_install), eval_expert rows=True (paired-z). Returns (evaluator, host, args)."""
    os.environ.setdefault("NEURAHASH_GLM_PIECES", a.pieces)
    import sharddiloco_glm_contributor as N
    from neurahash import diloco_merge as dm
    import glm_verifier_eval as GE
    N.use_glm_lane_names()
    G = N._G()
    args = argparse.Namespace(mode="glm", device="cpu", shard_dir=a.shard_dir, config_dir=a.config_dir,
                              threads=a.threads, warm_steps=0, pieces=a.pieces, piece=None,
                              data_dir=a.coord_data_dir, coord_data_dir=a.coord_data_dir,
                              domains=a.domains, slots="1:0", max_active_slots=a.max_active_slots,
                              claim_by="hash")
    t0 = time.time()
    model, cfg, seq = N.build_node_model(args, log=log, need_gib=N.GLM_COORD_NEED_GIB)
    host = G.GlmExpertLaneHost(model, cfg, N.parse_slots("1:0"), claimable=N.node_claimable_coords(args),
                               max_active=a.max_active_slots)
    N.bind_campaign_id(host, a.campaign)
    ev = GE.GlmGateEvaluator(host, None, None, campaign=a.campaign, G=G, N=N, dm=dm, fp32=True,
                             lazy_fp32=bool(getattr(a, "lazy_fp32", False)))
    meter = dm.FlopMeter(G.glm_fwd_flops_per_example(cfg, seq))
    if bool(getattr(a, "layer_major", False)):
        ev.eval_expert = GE.make_eval_expert_layer_major(host, meter)
    else:
        ev.eval_expert = host.make_eval_expert(meter, seq, rows=True)
    log("[shadow] model ready in %.1fs: base model_root=%s.. seq=%d fp32=True" % (time.time() - t0,
                                                                                 N.model_root(host)[:12], seq))
    return ev, host, args, N, G


def _load_state(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def resume_point(out_path, default):
    """--resume: the first event after the newest one already written to `out_path` (or `default`)."""
    last = None
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = int(json.loads(line)["event"])
                except (ValueError, KeyError, TypeError):
                    continue
                last = e if last is None else max(last, e)
    except OSError:
        return int(default)
    return int(default) if last is None else max(int(default), last + 1)


DISAGREE_STATS = ("disagree", "ce_disagree", "root_disagree", "chain_break", "omitted", "errors")
EXIT_DISAGREE, EXIT_PAUSED = 1, 3


def write_status(path, core, extra=None):
    """Atomically write the operator status file (review 2 fixes 2026-11-25)."""
    st = core.status() if hasattr(core, "status") else {"state": "running", "stats": dict(core.stats)}
    st["updated"] = time.time()
    st.update(extra or {})
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, sort_keys=True, default=str)
    os.replace(tmp, path)
    return st


def run_loop(core, *, upto=None, follow=False, poll_s=120.0, log=print, status_path=None, sleep=time.sleep,
             relog_s=1800.0, clock=time.time):
    """Drive core.step() and return the exit code (review 2 fixes 2026-11-25). An exception out of step() is
    logged as state "crashed" (status file, exit EXIT_PAUSED), never a bare traceback. One-shot: EXIT_PAUSED
    when the walk is paused/broken, EXIT_DISAGREE when any disagreement / chain break / omission / error was
    counted, else 0. --follow keeps polling and logs an active pause at most once per relog_s."""
    last_log = None
    crashed = None
    while True:
        try:
            core.step(upto=upto)
        except Exception as ex:                                 # noqa: BLE001
            crashed = "%s: %s" % (type(ex).__name__, str(ex)[:300])
        st = core.status() if hasattr(core, "status") else {"state": "running"}
        if crashed is not None:
            st = {"state": "crashed", "reason": crashed, "at_event": getattr(core, "next_event", None)}
        if status_path:
            write_status(status_path, core, st if crashed else None)
        if st.get("state") != "running":
            if last_log is None or clock() - last_log >= relog_s:
                log("[shadow] WALK %s at event %s since %s: %s" % (str(st.get("state")).upper(), st.get("at_event"),
                                                                  st.get("since"), st.get("reason")))
                last_log = clock()
        else:
            last_log = None
        if crashed is not None:
            return EXIT_PAUSED
        if not follow or (upto is not None and core.next_event > upto):
            break
        sleep(poll_s)
    if st.get("state") != "running":
        return EXIT_PAUSED
    if any(int(core.stats.get(k, 0) or 0) > 0 for k in DISAGREE_STATS):
        return EXIT_DISAGREE
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Shadow verifier for the live GLM lane (dry-run, GET only).")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--relay", default=RELAY)
    ap.add_argument("--out", required=True, help="JSONL, appended")
    ap.add_argument("--state-file", default=None, help="local only: the independent probe seed (default <out>.state.json)")
    ap.add_argument("--rescore-from", type=int, default=1, help="first event to RE-SCORE (earlier ones are replayed)")
    ap.add_argument("--upto", type=int, default=None, help="stop after this event (offline proof); default: newest")
    ap.add_argument("--follow", action="store_true", help="keep polling for new events (always-on shadow)")
    ap.add_argument("--resume", action="store_true",
                    help="re-score only events after the newest row already in --out (restart without duplicates)")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--probe-size", type=int, default=256)
    ap.add_argument("--probe-pool", default="D:/glm_daily/coord/ids_daily_probe.npy")
    ap.add_argument("--probe-pool-sha256", default="28bba0c05cdc8a556f656678b560b511dc7acd3c06fa74832c8193dbe67b3704",
                    help="coord_freeze.json's pin of the live pool")
    ap.add_argument("--probe-seed-env", default=None,
                    help="EXACT mode: name of an env var holding the coordinator's probe seed (owner decision); "
                         "unset = independent mode")
    ap.add_argument("--probe-full-pool", action="store_true",
                    help="score EVERY pool row once, fixed order, no seed (matches a coordinator run with "
                         "NEURAHASH_SD_PROBE_FULL_POOL=1; then base_val must reproduce). Default off. "
                         "Exclusive with --probe-seed-env; --probe-size is ignored.")
    ap.add_argument("--outer", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=1e-5)
    ap.add_argument("--paired-z", type=float, default=3.0)
    ap.add_argument("--drift-cap", type=float, default=0.02)
    ap.add_argument("--drift-base-dir", default="E:/_glm_layer_experts", help="pristine per-layer packs (read only)")
    ap.add_argument("--shard-dir", default="D:/hf_models/GLM-4.7-Flash-bf16_shards_100mb")
    ap.add_argument("--config-dir", default="D:/hf_models/GLM-4.7-Flash-bf16")
    ap.add_argument("--coord-data-dir", default="D:/glm_daily/coord")
    ap.add_argument("--domains", default="daily,daily")
    ap.add_argument("--pieces", default="0-23")
    ap.add_argument("--max-active-slots", type=int, default=120)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--mem-floor-gib", type=float, default=8.0)
    ap.add_argument("--lazy-fp32", action="store_true",
                    help="keep decoder layers bf16 at rest, upcast per module in the forward (same numbers, ~half RAM)")
    ap.add_argument("--layer-major", action="store_true",
                    help="score layer-major (each layer upcast once per pass, not once per sub-batch); same numbers")
    ap.add_argument("--heldout-check", action="store_true",
                    help="once, before the first re-score: held-out CE of the replayed state vs the published one")
    # ---- 2026-11-25, all default OFF (docs/research/VERIFIER_CE_OMISSION_2026-11-25.md) ----
    ap.add_argument("--check-ce", action="store_true",
                    help="re-compute the published held-out CE on every accept event (CPU, the coordinator's "
                         "_chunked_heldout_ce over <coord-data-dir>/ids_daily_heldout.npy); carries checked exactly")
    ap.add_argument("--ce-tol", type=float, default=None, help="|verifier CE - published| bound (default: pinned "
                                                               "glm_verifier_checks.CE_TOL_DEFAULT)")
    ap.add_argument("--ce-eval-chunk", type=int, default=int(os.environ.get("NEURAHASH_GLM_EVAL_CHUNK", "64")),
                    help="the coordinator's --eval-chunk (same default)")
    ap.add_argument("--check-root", action="store_true",
                    help="re-compute the published model_root (coordinator slot order rebuilt from records) and "
                         "every published slot root")
    ap.add_argument("--omission-window", type=int, default=0,
                    help="OMITTED detection: events a valid contribution may wait unjudged (0 = off)")
    ap.add_argument("--omission-max-stale", type=int, default=None,
                    help="skip records older than this many events (default: dm.max_staleness(), unbounded)")
    ap.add_argument("--chain-check", action="store_true", help="CHAIN_BREAK detection (counter back, rewrite, "
                                                              "root revisit, campaign change w/o marker, link)")
    a = ap.parse_args(argv)

    if a.resume:
        a.rescore_from = resume_point(a.out, a.rescore_from)
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "-1":
        print("[shadow] REFUSING: set CUDA_VISIBLE_DEVICES=-1 (the GPU belongs to the miners)", flush=True)
        return 2

    def log(m):
        print(time.strftime("%Y-%m-%dT%H:%M:%S"), m, flush=True)

    guard = MemGuard(a.mem_floor_gib, log=log).start()
    with open(a.probe_pool, "rb") as f:
        pool_sha = hashlib.sha256(f.read()).hexdigest()
    if a.probe_pool_sha256 and pool_sha != a.probe_pool_sha256:
        log("[shadow] FATAL: probe pool sha256 %s.. != pinned %s.." % (pool_sha[:12], a.probe_pool_sha256[:12]))
        return 2
    pool = np.load(a.probe_pool)
    if a.probe_full_pool and a.probe_seed_env:
        log("[shadow] FATAL: --probe-full-pool and --probe-seed-env are exclusive (full pool needs no seed)")
        return 2
    if a.probe_full_pool:
        seed, mode = 0, "full_pool"
    elif a.probe_seed_env:
        raw = os.environ.get(a.probe_seed_env, "")
        if not raw.strip():
            log("[shadow] FATAL: --probe-seed-env %s is empty" % a.probe_seed_env)
            return 2
        from neurahash import diloco_merge as dm
        seed, mode = dm.SecretRotatedProbe.seed_from_env(a.probe_seed_env, required=True), "exact"
    else:
        st_path = a.state_file or (a.out + ".state.json")
        st = _load_state(st_path)
        if "independent_seed" not in st:
            st["independent_seed"] = int.from_bytes(os.urandom(7), "big")
            with open(st_path, "w", encoding="utf-8") as f:
                json.dump(st, f)
        seed, mode = int(st["independent_seed"]), "independent"
    probe = ProbeSource(pool, seed, a.probe_size, mode)
    log("[shadow] probe mode=%s size=%d pool=%s.. (seed never printed)" % (mode, probe.size, pool_sha[:12]))
    if mode == "full_pool":
        _X, _y = probe.rows(0)
        log("[shadow] probe FULL POOL: %d rows sha256 %s.. (compare with the coordinator's boot line)"
            % (len(_X), probe.digest(_X, _y)[:16]))

    ev, host, args, N, G = build_real(a, log=log)
    log("[shadow] after build: rss %.2f GiB, min available %.2f GiB" % (guard.peak_rss_gib(), guard.min_avail_gib()))
    drift = None
    if a.drift_cap and a.drift_cap > 0:
        import glm_expert_drift as GD
        import sharddiloco_glm_coordinator as C
        why = C._drift_cap_selfcheck(GD, a.drift_cap)
        if why is not None:
            log("[shadow] FATAL: drift self-check: %s" % why)
            return 2
        drift = C.DriftCap(a.drift_cap, a.drift_base_dir, GD)
    import sharddiloco_glm_coordinator as C
    why = C._paired_z_selfcheck(a.paired_z)
    if why is not None:
        log("[shadow] FATAL: paired-z self-check: %s" % why)
        return 2
    relay = ReadOnlyRelay(a.relay)
    extra = {}
    if a.check_ce or a.check_root or a.omission_window > 0 or a.chain_check:
        import glm_verifier_checks as K
        extra = dict(check_root=a.check_root, N=N, omission_window=a.omission_window, chain_check=a.chain_check)
        if a.check_ce:
            extra.update(check_ce=True, ce_tol=a.ce_tol, heldout=np.load(os.path.join(a.coord_data_dir,
                                                                                      "ids_daily_heldout.npy")),
                         ce_fn=K.make_ce_fn(C, G, a.ce_eval_chunk, layer_major=bool(a.layer_major)))
        if a.omission_window > 0:
            from neurahash import diloco_merge as _dm
            extra.update(sig_ok=K.make_sig_ok(C, N), lineage_ok=K.make_lineage_ok(C),
                         omission_max_stale=(a.omission_max_stale if a.omission_max_stale is not None
                                             else _dm.max_staleness()),
                         omission_delta_ok=K.make_delta_ok(C, G, relay, host))
        log("[shadow] checks: ce=%s root=%s omission_window=%d chain=%s" % (a.check_ce, a.check_root,
                                                                            a.omission_window, a.chain_check))
    core = ShadowCore(ev, relay, probe, a.campaign, a.out, outer=a.outer, margin=a.margin, paired_z=a.paired_z,
                      drift=drift, slot_root=N.slot_root, rescore_from=a.rescore_from, log=log, memguard=guard,
                      **extra)
    t0 = time.time()
    if a.rescore_from > 1:
        core.step(upto=a.rescore_from - 1)
        log("[shadow] replayed events 1..%d (%d accepts) in %.1fs; rss peak %.2f GiB"
            % (a.rescore_from - 1, core.stats["replayed_accepts"], time.time() - t0, guard.peak_rss_gib()))
        if a.heldout_check:
            heldout = np.load(os.path.join(a.coord_data_dir, "ids_daily_heldout.npy"))
            prev = relay.json_by_name(accepted_name(a.campaign, a.rescore_from - 1)) or {}
            t1 = time.time()
            ce = C._chunked_heldout_ce(G, host.model, heldout, 64)
            log("[shadow] HELDOUT CHECK at event %d: replayed %.9f vs published %.9f (d %+.2e) in %.1fs"
                % (a.rescore_from - 1, ce, float(prev.get("heldout_ce") or float("nan")),
                   ce - float(prev.get("heldout_ce") or float("nan")), time.time() - t1))
    rc = run_loop(core, upto=a.upto, follow=a.follow, poll_s=a.poll_s, log=log, status_path=a.out + ".status.json")
    log("[shadow] stats %s rss_peak=%.2f GiB min_avail=%.2f GiB elapsed=%.0fs exit=%d"
        % (json.dumps(core.stats), guard.peak_rss_gib(), guard.min_avail_gib(), time.time() - t0, rc))
    guard.stop()
    return rc


if __name__ == "__main__":
    sys.exit(main())
