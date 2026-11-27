"""STAGE 2 (docs/research/PLAN_BITCOIN_STYLE_PAYOUTS_2026-11-19.md): an INDEPENDENT M-of-N verifier
committee for the GLM lane.

Today the GLM lane's "M-of-N quorum" is one operator signing N times: every validator key loads from ONE
local file (tools/sharddiloco_glm_coordinator.py _load_validator_keyset), so the coordinator alone decides
every payout. This module replaces that gatherer -- behind a default-OFF flag -- with a committee whose
members hold their OWN keys on OTHER machines and whose vote is a real veto:

  * the coordinator turns each paid row into a PROPOSAL (recipient, amount, height, delta cid, the gate's
    claimed numbers, and the evidence a verifier needs to recompute the gate: base root, probe spec,
    margin);
  * each verifier (tools/glm_verifier_node.py) recomputes the gate on its own hardware with a tolerance
    band and returns a SIGNED vote: approve (carrying a settlement attestation under its own key) or veto;
  * the coordinator settles only when at least M distinct roster verifiers approved AND fewer than
    `veto_threshold` vetoed; a veto, a timeout, a forged vote or too few approvals WITHHOLDS (fail-closed);
  * any address whose key the coordinator itself holds is EXCLUDED from the roster before counting, so the
    coordinator can never satisfy its own quorum.

The approve path emits the SAME settlement_block_proof the lane already uses (neurahash.diloco_settlement),
built from the verifiers' attestations over the verifiers' roster, so ChainSettlement and every replayer
verify it unchanged. Pure module: no sockets at import, no torch, no GPU; the transport is injected.

Interface for Stage 1 (per-mint evidence; merge note): a paid row may carry
    pe["evidence"] = {"base_root": str, "probe": {...}, "margin": float, "campaign": str, ...}
and the committee forwards it verbatim in the proposal. A row without base_root/probe/margin is still
proposed, but an honest verifier VETOES it (it cannot recompute), so the committee is fail-closed until
Stage 1's evidence record lands. See STAGE2_VERIFIER_COMMITTEE_2026-11-19.md.
"""
from __future__ import annotations

import concurrent.futures as _cf
import hashlib
import ipaddress
import json
import math
import os
import threading
import time
from urllib.parse import quote, urlsplit

PROPOSAL_VERSION = 1
VOTE_DOMAIN = "glm-committee-vote"            # signing domain label (distinct from settle-vote)
APPROVE, VETO = "approve", "veto"
COMMITTEE_ENV = "NEURAHASH_GLM_COMMITTEE"
ROSTER_ENV = "NEURAHASH_GLM_COMMITTEE_ROSTER"
TIMEOUT_ENV = "NEURAHASH_GLM_COMMITTEE_TIMEOUT_S"
DEFAULT_RELAY = "http://47.84.93.96:8710"     # WAN data plane (public GET; PUT gated by X-Auth)
DEFAULT_TIMEOUT_S = 120.0
NAME_PREFIX = "glm-committee"
REQUIRED_EVIDENCE = ("base_root", "probe", "margin")
_FORBIDDEN_ROSTER_KEYS = ("key", "priv", "private", "secret", "seed", "mnemonic")


def committee_on(environ=None):
    """NEURAHASH_GLM_COMMITTEE master gate. DEFAULT OFF: unset/'0'/'false'/'off'/'no' -> False, and the
    coordinator never imports this module (byte-identical lane)."""
    env = os.environ if environ is None else environ
    return (env.get(COMMITTEE_ENV, "") or "").strip().lower() in ("1", "true", "on", "yes")


# --------------------------------------------------------------------------- canonical encoding
def canonical(obj):
    """Sorted-keys, compact, ASCII JSON bytes: the ONE encoding both sides hash and sign."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def proposal_digest(proposal):
    return hashlib.sha256(canonical(proposal)).hexdigest()


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def make_proposal(*, recipient, amount, height, delta_cid, row=None, evidence=None, namespace=""):
    """The exact accept a verifier is asked to recompute. `row` is the coordinator's per_expert row (its
    claimed gate numbers ride along for audit; a verifier never trusts them); `evidence` is Stage 1's
    per-mint record (base_root, probe spec, margin, campaign)."""
    row = row or {}
    return {
        "v": PROPOSAL_VERSION,
        "ns": str(namespace or ""),
        "recipient": str(recipient),
        "amount": round(float(amount), 12),
        "height": int(height),
        "delta_cid": str(delta_cid or ""),
        "expert": (int(row["expert"]) if row.get("expert") is not None else None),
        "miner": (str(row["miner"]) if row.get("miner") is not None else None),
        "claimed": {"base_val": _num(row.get("base_val")), "merged_val": _num(row.get("merged_val")),
                    "gain": _num(row.get("gain"))},
        "evidence": dict(evidence or {}),
    }


def evidence_complete(proposal):
    ev = proposal.get("evidence") or {}
    return all(ev.get(k) not in (None, "", {}) for k in REQUIRED_EVIDENCE) and _num(ev.get("margin")) is not None


# --------------------------------------------------------------------------- the published reward rule
# GAP-1 (review 2026-11-19): a verifier must not take `amount` from the proposal. The GLM lane pays with ONE
# deterministic, published rule (neurahash.diloco_merge.attribute_minted, "F4 pay the goal metric"): the event's
# held-out gain split over its ACCEPTED rows in proportion to each row's positive probe gain. The coordinator puts
# the rule's INPUTS in the evidence (evidence["reward"]); the verifier re-runs the rule itself, checks the proposal's
# amount equals the rule's output for this row, and checks the row's gain against the gain it MEASURED.
REWARD_RULE = "attribute_minted/v1"
AMOUNT_RTOL = 1e-9          # float re-evaluation of one deterministic formula; proposals round to 12 decimals


def reward_inputs(per_expert, heldout_gain):
    """The rule's inputs for one event (what the coordinator publishes in evidence["reward"])."""
    rows = [{"miner": (str(pe.get("miner")) if pe.get("miner") is not None else None),
             "expert": (int(pe["expert"]) if pe.get("expert") is not None else None),
             "accepted": bool(pe.get("accepted")), "gain": _num(pe.get("gain"))}
            for pe in (per_expert or ())]
    return {"rule": REWARD_RULE, "heldout_gain": _num(heldout_gain), "rows": rows}


def recompute_amount(reward, miner, expert):
    """(amount, row) for (miner, expert) under the published rule, or (None, reason) when the inputs are missing,
    malformed, name another rule, or do not hold exactly one such row. Pure; re-runs attribute_minted itself."""
    from neurahash import diloco_merge as dm
    if not isinstance(reward, dict) or reward.get("rule") != REWARD_RULE:
        return None, "reward rule missing or not %s" % REWARD_RULE
    hg = _num(reward.get("heldout_gain"))
    rows = reward.get("rows")
    if hg is None or hg < 0.0 or not isinstance(rows, list) or not rows:
        return None, "reward inputs incomplete"
    work = []
    for r in rows:
        if not isinstance(r, dict) or (r.get("accepted") and _num(r.get("gain")) is None):
            return None, "reward row malformed"
        work.append({"miner": r.get("miner"), "expert": r.get("expert"), "accepted": bool(r.get("accepted")),
                     "gain": _num(r.get("gain")) or 0.0})
    dm.attribute_minted(work, hg)
    mine = [w for w in work if str(w.get("miner")) == str(miner) and w.get("expert") == expert]
    if len(mine) != 1:
        return None, "reward inputs hold %d rows for this miner/expert (need exactly 1)" % len(mine)
    return float(mine[0]["paid"]), mine[0]


def amount_matches(amount, expected):
    a, e = _num(amount), _num(expected)
    if a is None or e is None:
        return False
    return abs(a - round(e, 12)) <= max(1e-12, AMOUNT_RTOL * abs(e))


# --------------------------------------------------------------------------- signed votes
def _vote_payload(proposal_sha, verdict, reason, recomputed):
    return {"domain": VOTE_DOMAIN, "proposal_sha": str(proposal_sha), "verdict": str(verdict),
            "reason": str(reason or "")[:200], "recomputed": recomputed or {}}


def sign_vote(account, proposal, verdict, reason="", recomputed=None):
    """One verifier's signed vote over THIS proposal. An approve also carries a settlement attestation
    (diloco_settlement.sign_settlement) over the proposal's exact reward decision, which is what lets the
    coordinator build a chain-verifiable proof out of verifier keys only."""
    from neura_l1.signing import sign_bytes
    from neurahash import diloco_settlement as ds
    if verdict not in (APPROVE, VETO):
        raise ValueError("verdict must be approve or veto")
    rec = {k: _num(v) for k, v in (recomputed or {}).items()}
    payload = _vote_payload(proposal_digest(proposal), verdict, reason, rec)
    vote = dict(payload, address=account.address,
                sig=sign_bytes(account, canonical(payload), domain=VOTE_DOMAIN))
    if verdict == APPROVE:
        vote["settlement"] = ds.sign_settlement(account, recipient=proposal["recipient"],
                                                amount=proposal["amount"], height=proposal["height"],
                                                delta_cid=proposal["delta_cid"])
    return vote


def verify_vote(vote, proposal):
    """(address_lower, None) when `vote` is a valid signature by its claimed address over THIS proposal,
    else (None, reason). The claimed address is NEVER trusted: the signer is recovered and must match."""
    from neura_l1.signing import recover_bytes
    if not isinstance(vote, dict):
        return None, "malformed vote"
    try:
        payload = _vote_payload(vote.get("proposal_sha"), vote.get("verdict"), vote.get("reason"),
                                vote.get("recomputed"))
        if payload["verdict"] not in (APPROVE, VETO):
            return None, "bad verdict"
        if payload["proposal_sha"] != proposal_digest(proposal):
            return None, "vote is for a different proposal"
        if payload["reason"] != str(vote.get("reason") or ""):
            return None, "reason altered"
        signer = str(recover_bytes(canonical(payload), vote.get("sig"), domain=VOTE_DOMAIN)).lower()
    except Exception as e:                                         # noqa: BLE001
        return None, "unrecoverable signature (%s)" % type(e).__name__
    if signer != str(vote.get("address", "")).lower():
        return None, "signature does not match the claimed address"
    return signer, None


# --------------------------------------------------------------------------- roster (PUBLIC keys only)
def load_roster(path):
    """Roster file = PUBLIC identities only:
        {"verifiers": [{"address": "0x..", "name": "4060-pc"}, ...], "m": 2, "veto_threshold": 1}
    Refuses (ValueError) a file that carries anything that looks like key material, so a private keyset
    can never be pointed at as a roster by mistake. Returns {"members": [addr_lower], "names": {...},
    "m": int|None, "veto_threshold": int}."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("verifiers"), list):
        raise ValueError("roster must be an object with a 'verifiers' list")
    members, names = [], {}
    for ent in raw["verifiers"]:
        if not isinstance(ent, dict):
            raise ValueError("roster entries must be objects")
        for k in ent:
            if any(f in str(k).lower() for f in _FORBIDDEN_ROSTER_KEYS):
                raise ValueError("roster entry carries a field named %r: a roster holds PUBLIC addresses only"
                                 % k)
        a = str(ent.get("address", "")).strip().lower()
        if not (a.startswith("0x") and len(a) == 42):
            raise ValueError("bad verifier address %r" % ent.get("address"))
        if a not in members:
            members.append(a)
            names[a] = str(ent.get("name", ""))
    m = raw.get("m")
    return {"members": members, "names": names, "m": (int(m) if m is not None else None),
            "veto_threshold": max(1, int(raw.get("veto_threshold", 1)))}


def effective_quorum(members, excluded, m=None):
    """(eligible_members, m_eff). Members whose key the coordinator holds are removed FIRST. m_eff is the
    configured m raised to a strict majority of the eligible set (the settlement proof requires one, and a
    minority committee would let a coordinator-friendly subset pay), and never below 1."""
    ex = {str(a).lower() for a in (excluded or ())}
    elig = [a for a in members if str(a).lower() not in ex]
    maj = len(elig) // 2 + 1
    m_eff = max(maj, int(m) if m else maj, 1)
    return elig, m_eff


# --------------------------------------------------------------------------- decision
def decide(proposal, votes, eligible, m, veto_threshold=1):
    """Aggregate votes for ONE proposal. Counts at most one vote per distinct eligible address (first valid
    one wins); everything else is ignored with a reason. Approved iff approvals >= m AND vetoes <
    veto_threshold AND every counted approval carries a settlement attestation that recovers to its own
    signer over this exact decision."""
    from neurahash import diloco_settlement as ds
    from neura_l1.signing import recover_bytes_scheme
    elig = {str(a).lower() for a in eligible}
    payload = ds.settlement_payload(proposal["recipient"], proposal["amount"], proposal["height"],
                                    proposal["delta_cid"])
    approvals, vetoes, ignored, atts, seen = [], [], [], [], set()
    for vote in votes or ():
        addr, why = verify_vote(vote, proposal)
        if addr is None:
            ignored.append((str((vote or {}).get("address", "?")) if isinstance(vote, dict) else "?", why))
            continue
        if addr not in elig:
            ignored.append((addr, "not an eligible roster verifier"))
            continue
        if addr in seen:
            ignored.append((addr, "duplicate vote"))
            continue
        if vote["verdict"] == APPROVE:
            att = vote.get("settlement")
            try:
                s = recover_bytes_scheme(payload, (att or {}).get("sig"), domain=ds.SETTLE_VOTE_DOMAIN).lower()
            except Exception:                                      # noqa: BLE001
                s = None
            if s != addr:
                ignored.append((addr, "approve without a matching settlement attestation"))
                continue
            seen.add(addr)
            approvals.append(addr)
            atts.append(att)
        else:
            seen.add(addr)
            vetoes.append(addr)
    if len(vetoes) >= int(veto_threshold):
        reason = "VETO by %d verifier(s)" % len(vetoes)
    elif len(approvals) < int(m):
        reason = "quorum not met: %d of %d approvals" % (len(approvals), int(m))
    else:
        reason = "approved"
    return {"approved": reason == "approved", "approvals": approvals, "vetoes": vetoes, "ignored": ignored,
            "attestations": atts, "reason": reason, "m": int(m), "veto_threshold": int(veto_threshold)}


def gather_votes(transport, proposal, members, timeout):
    """Ask every member CONCURRENTLY under ONE shared deadline (the recorded D1 liveness flaw was a
    sequential call). transport(addr, proposal, timeout_s) -> vote|None. A member that errors or misses the
    deadline simply has no vote. Never waits past the deadline (executor shut down without joining)."""
    if not members:
        return []
    deadline = time.monotonic() + float(timeout)
    pool = _cf.ThreadPoolExecutor(max_workers=max(1, len(members)), thread_name_prefix="glm-committee")
    futs = [pool.submit(transport, a, proposal, float(timeout)) for a in members]
    out = []
    try:
        for f in futs:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                v = f.result(timeout=left)
            except Exception:                                      # noqa: BLE001 (timeout or transport error)
                continue
            if v is not None:
                out.append(v)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return out


class CommitteeGatherer:
    """Drop-in replacement for the coordinator's settlement gatherer: callable as
    gather(recipient=, amount=, height=, delta_cid=) -> settlement_block_proof | None.

    `committee_context(per_expert, cid_by_miner, height, pay_addr_by_miner)` is the coordinator hook: it
    builds every paid row's proposal and gathers ALL their votes at once (one deadline per event), caching
    the decisions; __call__ then answers from the cache. Called without context (one-shot helpers), it
    builds an evidence-less proposal, which an honest verifier vetoes: fail-closed."""

    def __init__(self, members, *, m=None, veto_threshold=1, transport, excluded=(), timeout=DEFAULT_TIMEOUT_S,
                 namespace="", evidence_fn=None, log=None, journal_path=None, stake=1000.0, stake_quorum=None):
        self.excluded = sorted({str(a).lower() for a in (excluded or ())})
        self.dropped = [a for a in members if str(a).lower() in self.excluded]
        self.eligible, self.m = effective_quorum(members, self.excluded, m)
        self.veto_threshold = max(1, int(veto_threshold))
        self.transport, self.timeout, self.namespace = transport, float(timeout), str(namespace or "")
        self.evidence_fn = evidence_fn or (lambda row, cid: (row or {}).get("evidence") or {})
        self.log, self.journal_path = log, journal_path
        self.settle_roster = {a: float(stake) for a in self.eligible}
        # RELEASE 2026-11-19 (stage 3 wiring, NEURAHASH_GLM_STAKE_GATE, default None = head count only): an
        # extra requirement, never a relaxation. The settlement proof and the authoritative replayer still count
        # heads (STAGE3 doc "Replay gap"), so a mint needs the head-count quorum AND a strict stake majority.
        self.stake_quorum = stake_quorum
        self._cache = {}
        self._lock = threading.Lock()
        self.decisions = []                       # every decision this process made (false-reject audit)

    # -- proposal building --------------------------------------------------------------------
    def _proposal(self, recipient, amount, height, cid, row=None):
        return make_proposal(recipient=recipient, amount=amount, height=height, delta_cid=cid, row=row,
                             evidence=self.evidence_fn(row, cid) if row is not None else {},
                             namespace=self.namespace)

    @staticmethod
    def _key(recipient, amount, height, cid):
        return (str(recipient), round(float(amount), 12), int(height), str(cid or ""))

    def committee_context(self, per_expert, cid_by_miner, height, pay_addr_by_miner=None):
        addr_of = pay_addr_by_miner or {}
        props = []
        for pe in per_expert or ():
            if not pe.get("accepted") or float(pe.get("paid", 0.0)) <= 0.0:
                continue
            miner = pe.get("miner")
            cid = cid_by_miner.get(miner) if cid_by_miner else None
            props.append(self._proposal(addr_of.get(miner, miner), float(pe.get("paid", 0.0)), height, cid, pe))
        if not props:
            return
        publish = getattr(self.transport, "publish", None)
        for p in props:
            if publish is not None:
                try:
                    publish(p)
                except Exception:                                  # noqa: BLE001 -> verifiers cannot see it
                    pass
        with _cf.ThreadPoolExecutor(max_workers=len(props)) as ex:
            results = list(ex.map(self._decide_one, props))
        with self._lock:
            for p, d in zip(props, results):
                self._cache[self._key(p["recipient"], p["amount"], p["height"], p["delta_cid"])] = (p, d)

    def _decide_one(self, proposal):
        votes = gather_votes(self.transport, proposal, self.eligible, self.timeout)
        d = decide(proposal, votes, self.eligible, self.m, self.veto_threshold)
        if self.stake_quorum is not None:
            try:
                sd = self.stake_quorum.decide_over(d, self.eligible)
            except Exception as e:                                 # noqa: BLE001 -> unreadable stake withholds
                sd = {"approved": False, "reason": "stake gate read failed (%s)" % type(e).__name__}
            d = dict(d)
            d["stake_gate"] = {k: sd.get(k) for k in ("approved", "reason", "approve_stake_wei", "veto_stake_wei",
                                                      "total_stake_wei")}
            if d["approved"] and not sd.get("approved"):
                d["approved"], d["reason"] = False, "stake gate: %s" % sd.get("reason")
        rec = {"t": time.time(), "proposal_sha": proposal_digest(proposal), "height": proposal["height"],
               "delta_cid": proposal["delta_cid"], "recipient": proposal["recipient"],
               "approved": d["approved"], "reason": d["reason"], "approvals": d["approvals"],
               "vetoes": d["vetoes"], "n_eligible": len(self.eligible), "m": self.m,
               "votes": [{"address": v.get("address"), "verdict": v.get("verdict"), "reason": v.get("reason"),
                          "recomputed": v.get("recomputed")} for v in votes if isinstance(v, dict)]}
        with self._lock:
            self.decisions.append(rec)
            if self.journal_path:
                try:
                    with open(self.journal_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=True) + "\n")
                except OSError:
                    pass
        if self.log:
            self.log("[glm-committee] height=%d cid=%s.. %s (approve=%d veto=%d of %d eligible, m=%d)"
                     % (proposal["height"], proposal["delta_cid"][:12], d["reason"], len(d["approvals"]),
                        len(d["vetoes"]), len(self.eligible), self.m))
        return d

    def __call__(self, recipient, amount, height, delta_cid):
        if not self.eligible:
            return None
        key = self._key(recipient, amount, height, delta_cid)
        with self._lock:
            hit = self._cache.pop(key, None)
        if hit is None:
            p = self._proposal(recipient, amount, height, delta_cid)
            publish = getattr(self.transport, "publish", None)
            if publish is not None:
                try:
                    publish(p)
                except Exception:                                  # noqa: BLE001
                    pass
            hit = (p, self._decide_one(p))
        p, d = hit
        if not d["approved"]:
            return None
        from neurahash import diloco_settlement as ds
        try:
            proof = ds.settlement_block_proof(recipient=p["recipient"], amount=p["amount"], height=p["height"],
                                              delta_cid=p["delta_cid"], roster=self.settle_roster,
                                              attestations=d["attestations"], m=self.m)
        except Exception:                                          # noqa: BLE001 -> fail-closed
            return None
        ok, _why, _n, _req = ds.verify_settlement_block_proof(proof)
        return proof if ok else None


# --------------------------------------------------------------------------- WAN relay transport
def assert_wan_url(url):
    """The DATA plane is WAN-only (owner directive 2026-07-31): refuse a private/loopback/link-local host."""
    host = urlsplit(url).hostname or ""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if host in ("localhost",) or host.endswith(".local"):
            raise ValueError("relay must be a WAN address, got %r" % host)
        return url
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified:
        raise ValueError("relay must be a WAN address, got %r" % host)
    return url


def proposal_name(ns, height, k):
    return "%s/%s/p/%d/%d" % (NAME_PREFIX, ns, int(height), int(k))


def vote_name(ns, proposal_sha, addr):
    return "%s/%s/v/%s/%s" % (NAME_PREFIX, ns, proposal_sha, str(addr).lower())


def vote_copy_name(ns, proposal_sha, addr, body):
    """GAP-4: a CONTENT-BOUND copy of a vote under the voter's own name: <vote_name>/<sha256(body)[:16]>. The body
    carries the verifier's signature, so nobody can name this copy before the verifier has signed: a squatter
    holding the shared relay token can pre-claim `vote_name` (and block that PUT) but not this one."""
    return "%s/%s" % (vote_name(ns, proposal_sha, addr), hashlib.sha256(bytes(body)).hexdigest()[:16])


MAX_PROPOSAL_K = 4096       # bound on the per-height slot probe (a squatter cannot make publish loop forever)


class RelayStore:
    """Minimal client for the content-store relay: GET /n/<name> -> {"hash"/"sha256"...}, GET /o/<sha>,
    PUT /o/<sha> with X-Name + X-Auth. The token is read from an env var and never logged."""

    def __init__(self, base=DEFAULT_RELAY, token_env="NEURAHASH_CONTENT_TOKEN", timeout=30.0, opener=None,
                 signer=None):
        self.base = assert_wan_url(base.rstrip("/"))
        self.token_env, self.timeout = token_env, float(timeout)
        # GAP-4: signer = this node's OWN account -> PUTs are signed-writer PUTs (neurahash.store_auth; relay
        # CONTENT_SIGNED_PUT), so each name it creates is OWNED by its address. None -> the shared token (as before).
        self.signer = signer
        import urllib.request
        self._open = opener or urllib.request.urlopen
        self._Request = urllib.request.Request

    def get_name(self, name):
        import urllib.error
        try:
            with self._open(self.base + "/n/" + quote(name, safe="/"), timeout=self.timeout) as r:
                ent = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        h = None
        if isinstance(ent, dict):
            inner = ent.get("entry") if isinstance(ent.get("entry"), dict) else ent
            h = inner.get("sha256") or inner.get("hash") or inner.get("h")
        if not h:
            return None
        return self.get_object(str(h))

    def get_object(self, sha):
        with self._open(self.base + "/o/" + str(sha).lower(), timeout=self.timeout) as r:
            body = r.read()
        if hashlib.sha256(body).hexdigest() != str(sha).lower():
            raise ValueError("relay object does not hash to its name")
        return body

    def list_names(self, prefix):
        """Names in the relay's public /manifest starting with `prefix` (GAP-4 fallback discovery)."""
        with self._open(self.base + "/manifest", timeout=self.timeout) as r:
            man = json.loads(r.read().decode("utf-8"))
        names = man.get("names", man) if isinstance(man, dict) else {}
        return sorted(str(n) for n in (names or {}) if str(n).startswith(prefix))

    def put(self, name, body):
        sha = hashlib.sha256(body).hexdigest()
        if self.signer is not None:
            from neurahash import store_auth
            headers = dict(store_auth.sign_put(self.signer, sha, name=name))
        else:
            headers = {"X-Name": name, "X-Auth": os.environ.get(self.token_env, "")}
        headers["Content-Type"] = "application/octet-stream"
        req = self._Request(self.base + "/o/" + sha, data=body, method="PUT", headers=headers)
        with self._open(req, timeout=self.timeout) as r:
            r.read()
        return sha


class RelayCommitteeTransport:
    """Coordinator side over the WAN relay: publish() writes the proposal under p/<height>/<k>; the call
    polls for each member's vote under v/<proposal_sha>/<addr> (and its content-bound copies) until the timeout.

    GAP-2 (review 2026-11-19): `k` used to restart at 0 in every process, so after a coordinator restart the next
    proposal at a height reused p/<h>/0 -- a name the relay refuses to rewrite (403), or one a verifier already
    voted on. publish() now takes the FIRST k at that height whose relay name is ABSENT, read from the relay
    itself, so names are unique and monotone across restarts with no local state to lose. A byte-identical
    re-publish reuses its existing name; a refused PUT (name claimed meanwhile) moves on to the next k."""

    def __init__(self, store, namespace, poll_s=5.0, list_every_s=30.0):
        self.store, self.ns, self.poll_s = store, str(namespace), float(poll_s)
        self.list_every_s = float(list_every_s)
        self._lock = threading.Lock()
        self.names = {}                         # proposal_sha -> the relay name it was published under

    def publish(self, proposal):
        body = canonical(proposal)
        h = int(proposal["height"])
        with self._lock:
            k = 0                     # scan from 0 every time: a byte-identical copy anywhere is reused
            while True:
                if k >= MAX_PROPOSAL_K:
                    raise RuntimeError("no free proposal slot at height %d" % h)
                name = proposal_name(self.ns, h, k)
                try:
                    cur = self.store.get_name(name)
                except Exception:                                  # noqa: BLE001 -> unreadable == occupied
                    cur = b"?"
                if cur is None:
                    try:
                        self.store.put(name, body)
                    except Exception:                              # noqa: BLE001 (403: claimed meanwhile)
                        k += 1
                        continue
                    break
                if bytes(cur) == body:                             # this exact proposal is already there
                    break
                k += 1
            self.names[proposal_digest(proposal)] = name

    @staticmethod
    def _valid(body, addr, proposal):
        try:
            v = json.loads(bytes(body).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, TypeError):
            return None
        signer, _why = verify_vote(v, proposal)
        return v if signer is not None and signer == str(addr).lower() else None

    def __call__(self, addr, proposal, timeout):
        """GAP-4: a vote counts only when its signature recovers to `addr` over THIS proposal. Whatever sits at the
        canonical name is checked; if it is absent or not a valid vote (a squatter's bytes), the content-bound copies
        under that name are listed from the relay manifest and the first valid one is taken."""
        name = vote_name(self.ns, proposal_digest(proposal), addr)
        end = time.monotonic() + float(timeout)
        next_list = time.monotonic()
        lister = getattr(self.store, "list_names", None)
        while time.monotonic() < end:
            try:
                body = self.store.get_name(name)
            except Exception:                                      # noqa: BLE001 -> retry until deadline
                body = None
            if body:
                v = self._valid(body, addr, proposal)
                if v is not None:
                    return v
            if lister is not None and time.monotonic() >= next_list:
                next_list = time.monotonic() + self.list_every_s
                try:
                    copies = lister(name + "/")
                except Exception:                                  # noqa: BLE001
                    copies = []
                for cn in copies:
                    try:
                        cb = self.store.get_name(cn)
                    except Exception:                              # noqa: BLE001
                        cb = None
                    v = self._valid(cb, addr, proposal) if cb else None
                    if v is not None:
                        return v
            time.sleep(min(self.poll_s, max(0.0, end - time.monotonic())))
        return None
