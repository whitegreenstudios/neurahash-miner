"""tests/test_glm_s1_attester.py -- the S1 shadow-attestation runner (tools/glm_s1_attester.py).

CPU only, fakes only: no model, no network, no relay writes.
  C:/Python313/python.exe -m pytest tests/test_glm_s1_attester.py -q

PUBLIC MINER COPY (3.9.8): the public tree ships LIGHT mode only. The full-mode re-score test of the private repo
(test_full_mode_reproduces_classify_gate_verdicts) is not carried: it drives glm_verifier_shadow.ShadowCore over the
private coordinator stack. The few helpers it borrowed from the private tests (FakeStore, _row, CAMP) are inlined
below, verbatim. Added here: --mode full (and light --omission-window > 0) refuse with a clear message when the
coordinator module is absent.
"""
import hashlib
import io
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eth_account import Account  # noqa: E402

import glm_s1_attester as R  # noqa: E402
import glm_verifier_shadow as S  # noqa: E402
from neurahash import accepted_audit as AA  # noqa: E402
from neurahash import glm_shadow_attester as A  # noqa: E402
from neurahash import signed_put as SP  # noqa: E402

_HAS_COORD = R.coordinator_available()


# ---- inlined verbatim from the private tests/test_glm_verifier_shadow.py + test_glm_shadow_attester.py ----
class FakeStore:
    def __init__(self):
        self.objs, self.names, self.methods = {}, {}, []

    def put_obj(self, body):
        sha = hashlib.sha256(body).hexdigest()
        self.objs[sha] = body
        return sha

    def put_name(self, name, body):
        self.names[name] = {"sha256": self.put_obj(body), "size": len(body)}

    def opener(self, req, timeout=None):
        self.methods.append(req.get_method())
        if req.get_method() != "GET":
            raise AssertionError("non-GET request reached the store: %s" % req.get_method())
        path = req.full_url.split("8710", 1)[1]
        if path == "/manifest":
            body = json.dumps(self.names).encode()
        elif path.startswith("/o/"):
            body = self.objs[path[3:]]
        else:
            raise AssertionError("unexpected path %s" % path)

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R(body)


MARGIN = 1e-5


def _row(cdec, cgain, vdec, vgain, *, agree=None, cls=None, z=6.0, probe="full_pool", event=7):
    a, _, c = S.compare(cdec, vdec) if agree is None else (agree, None, cls or ("agree" if agree else "disagree"))
    return {"event": event, "record": "cg/x/r0/glm-m.%d" % event, "delta_cid": "cid-%d" % event,
            "probe_mode": probe, "agree": a, "class": cls or c,
            "coordinator": {"decision": cdec, "gain": cgain, "margin": MARGIN},
            "verifier": {"decision": vdec, "gain": vgain, "z": z, "z_min": 3.0}}

CAMP = "c0ffee0000000001"            # 16 hex
TOL = A.Tolerance()
POLICY_RULE = {"prefix": "sharddiloco/glm/attest/", "mode": "self", "addr_segment": 5, "max_bytes": 65536,
               "max_names": 2048, "pattern": "sharddiloco/glm/attest/[0-9a-f]{16}/[0-9]{1,12}/0x[0-9a-f]{40}"}


def _acct(i=0):
    return Account.from_key(hashlib.sha256(b"s1-runner|%d" % i).digest())


class FakePub:
    def __init__(self, status=201, raise_=False):
        self.status, self.raise_, self.puts, self.dels = status, raise_, [], []

    def put(self, name, data):
        if self.raise_:
            raise ConnectionError("relay unreachable")
        self.puts.append((name, data))
        return self.status, b""

    def delete(self, name):
        self.dels.append(name)
        return 200, b""


def _attester(tmp_path, pub, mode="full", retain=500, i=0):
    acct = _acct(i)
    out = R.Output(str(tmp_path / ("att%d" % i)), CAMP, acct.address.lower(), log=lambda m: None)
    return R.Attester(acct, CAMP, mode, out, pub, TOL, "testcommit", retain, log=lambda m: None)


# ------------------------------------------------------------------------------------------ signing + names
def test_signing_round_trip_and_tamper():
    acct = _acct()
    row = _row("accept", 3e-4, "accept", 3e-4 + 4e-7, event=11)
    rec = {"event": 11, "model_root": "ab" * 32, "accepted": [],
           "judged": {"name": "cg/x/r10/glm-m.1", "accepted": True,
                      "gate": {"margin": 1e-5, "rows": [{"base_val": 5.0, "merged_val": 5.0 - 3e-4}]}}}
    att = R.build_attestation(acct, CAMP, 11, rec, [row], [("audit", A.AGREE, "x")], "full", TOL, "c0mm1t")
    assert att["verdict"] == A.AGREE and att["coord"]["decision"] == "accept"
    assert R.signer(att) == acct.address.lower()
    again = json.loads(R.line_bytes(att).decode("ascii"))                  # survives the JSONL / relay encoding
    assert R.signer(again) == acct.address.lower()
    bad = dict(att, verdict=A.DISAGREE)
    assert R.signer(bad) is None                                          # tamper -> no signer
    forged = dict(att, attester_addr=_acct(1).address.lower())
    assert R.signer(forged) is None
    assert "measured" in att and isinstance(att["measured"]["coord_gain_pico"], int)
    assert len(R.line_bytes(att)) < 4096                                  # small


def test_name_binds_the_signer_address_under_the_relay_policy():
    acct, other = _acct(), _acct(1)
    addr = acct.address.lower()
    name = R.attest_name(CAMP, 591, acct.address)                        # checksummed input is lower-cased
    assert name == "sharddiloco/glm/attest/%s/591/%s" % (CAMP, addr)
    assert R.name_address(name) == addr
    pol = SP.parse_policy(json.dumps({"operators": ["0x" + "11" * 20], "rules": [POLICY_RULE]}))
    ok = SP.decide(pol, name, addr, True, nbytes=1500)
    assert ok[0] is True                                                  # own address: allowed
    no = SP.decide(pol, name, other.address.lower(), True, nbytes=1500)
    assert no[0] is False and no[1] == 403                                # someone else's name: refused
    assert SP.decide(pol, name, addr, False)[1] == 403                    # unsigned: refused
    assert SP.decide(pol, name + "~x", addr, True)[1] == 403              # no suffixes under this rule
    assert SP.embedded_address(name, POLICY_RULE["addr_segment"]) == addr
    att = R.build_attestation(acct, CAMP, 591, None, [], [("audit", A.AGREE, "x")], "light", TOL, "c")
    data = R.line_bytes(att)
    assert R.verify_published(name, data) == (True, addr)
    assert R.verify_published(R.attest_name(CAMP, 591, other.address), data)[0] is False
    for bad in ("c0ffee00", "c0ffee0000000001x", "g0ffee0000000001"):
        try:
            R.attest_name(bad, 1, addr)
        except ValueError:
            continue
        raise AssertionError("bad campaign accepted: %s" % bad)


def test_key_file_formats(tmp_path):
    acct = _acct(3)
    hexkey = acct.key.hex()
    (tmp_path / "k.json").write_text(json.dumps({"key": hexkey}), encoding="utf-8")
    (tmp_path / "k.hex").write_text(hexkey + "\n", encoding="utf-8")
    assert R.load_account(str(tmp_path / "k.json")).address == acct.address
    assert R.load_account(str(tmp_path / "k.hex")).address == acct.address


# ------------------------------------------------------------------------------------------ publishing
def test_publish_failure_is_non_fatal_and_retried_with_identical_bytes(tmp_path):
    down = FakePub(raise_=True)
    at = _attester(tmp_path, down)
    rec = {"event": 5, "model_root": "cd" * 32, "accepted": []}
    a1 = at.attest(5, rec, [], [("audit", A.AGREE, "ok")])               # must not raise
    assert at.out.state["pending"] == [[5, R.attest_name(CAMP, 5, at.addr)]]
    assert at.out.counts["publish_fail"] == 1 and at.out.state["last_attested"] == 5
    line = open(at.out.jsonl, "rb").read().rstrip(b"\n")
    assert line == R.line_bytes(a1)
    refused = FakePub(status=503)
    at.pub = refused
    at.flush_publish()
    assert at.out.state["pending"] and len(refused.puts) == 1            # 503: still queued
    up = FakePub()
    at.pub = up
    at.flush_publish()
    assert at.out.state["pending"] == [] and up.puts == [(R.attest_name(CAMP, 5, at.addr), line)]
    # resume: a new Output on the same dir keeps the state
    out2 = R.Output(os.path.dirname(at.out.jsonl), CAMP, at.addr, log=lambda m: None)
    assert out2.state["last_attested"] == 5 and out2.state["published"] == [5]


def test_retention_deletes_only_own_old_names(tmp_path):
    pub = FakePub()
    at = _attester(tmp_path, pub, retain=2)
    for e in range(1, 6):
        at.attest(e, {"event": e, "model_root": "ee" * 32, "accepted": []}, [], [("audit", A.AGREE, "ok")])
    assert pub.dels == [R.attest_name(CAMP, e, at.addr) for e in (1, 2, 3)]
    assert sorted(at.out.state["published"]) == [4, 5]
    assert all(R.name_address(n) == at.addr for n in pub.dels)


# ------------------------------------------------------------------------------------------ full mode
def test_heldout_tol_veto_abstains_instead_of_false_disagree():
    row = _row("veto-after-margin", 3e-4, "accept", 3e-4, agree=False, cls="disagree")
    row["disagreement"] = ("coordinator veto-after-margin (held-out tol: the merged result moved held-out CE past "
                           "the tolerance); verifier accepted")
    assert A.classify_gate(row, TOL)[0] == A.DISAGREE                     # the raw rule would call it a flip
    assert R._gate_check(row, TOL) == (A.ABSTAIN, "heldout_tol_not_rechecked")
    other = dict(row, disagreement="coordinator accepted; verifier z-veto")
    other["coordinator"] = dict(row["coordinator"], decision="accept")
    other["verifier"] = dict(row["verifier"], decision="z-veto", z=1.0)
    assert R._gate_check(other, TOL)[0] == A.DISAGREE                     # control: a real flip still disagrees


# ------------------------------------------------------------------------------------------ light mode
def _sealed_store(coord, n=4, tamper_at=None):
    store = FakeStore()
    prev, root = None, "aa" * 32
    for e in range(1, n + 1):
        acc = (e == 2)
        if acc:
            root = "bb" * 32
        rec = {"event": e, "model_root": root, "slot_roots": {}, "heldout_ce": 8.5,
               "accepted": ([{"layer": 1, "glm_expert": 7, "cid": "cc" * 32, "outer": 0.7}] if acc else []),
               "judged": {"name": "cg/%s/r%d/glm-m.%d" % (CAMP, e - 1, e), "accepted": acc, "miner": "glm-m",
                          "gate": {"margin": 1e-5, "rows": [{"base_val": 5.0, "merged_val": 5.0 - 3e-4}]}}}
        ph = AA.record_hash(prev) if prev is not None else None
        if tamper_at == e:
            ph = "00" * 32
        rec = AA.seal(rec, coord, AA.build_audit(campaign=CAMP, event=e, coord_address=coord.address,
                                                 prev_hash=ph, prev_ce=8.5,
                                                 reset=("campaign start" if ph is None else None)))
        store.put_name(S.accepted_name(CAMP, e), json.dumps(rec).encode())
        prev = rec
    return store


def _light_run(tmp_path, store, coord):
    relay = R.TeeRelay("http://47.84.93.96:8710", opener=store.opener)
    pub = FakePub()
    at = _attester(tmp_path, pub, mode="light")
    lc = R.LightCore(relay, CAMP, coord.address, 0, log=lambda m: None)
    lc.start(1)

    def fn(e, rec, extra, late):
        at.attest(e, rec, [], extra)
    lc.step(fn)
    return [json.loads(ln) for ln in open(at.out.jsonl, encoding="utf-8")], at, pub


def test_light_mode_checks_audit_chain_and_root_link(tmp_path):
    coord = _acct(9)
    atts, at, pub = _light_run(tmp_path, _sealed_store(coord), coord)
    assert [a["event"] for a in atts] == [1, 2, 3, 4]
    for a in atts:
        ck = {c["k"]: (c["v"], c["r"]) for c in a["checks"]}
        assert ck["gate"] == (A.ABSTAIN, "light_no_rescore")
        assert ck["audit"][0] == A.AGREE, ck
        assert a["verdict"] == A.ABSTAIN and a["mode"] == "light" and "measured" not in a
    assert {c["k"]: c["v"] for c in atts[2]["checks"]}["root_link"] == A.AGREE
    assert len(pub.puts) == 4 and all(R.verify_published(n, d)[0] for n, d in pub.puts)


def test_light_mode_flags_a_broken_audit_link_and_a_foreign_signer(tmp_path):
    coord = _acct(9)
    atts, _, _ = _light_run(tmp_path, _sealed_store(coord, tamper_at=3), coord)
    a3 = [a for a in atts if a["event"] == 3][0]
    assert a3["verdict"] == A.DISAGREE and "audit_prev_hash_mismatch" in a3["reason"]
    atts2, _, _ = _light_run(tmp_path / "b", _sealed_store(_acct(8)), coord)   # signed by a different key
    assert all(a["verdict"] == A.DISAGREE and "audit_bad_signature" in a["reason"] for a in atts2)


def test_light_mode_never_imports_torch(tmp_path):
    """A child process blocks any torch import, then runs a full light pass (omission tracker on, which imports the
    coordinator module) through main()'s light path with a fake relay. It must finish and never touch torch."""
    script = r'''
import sys, json, os, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch.") or name == "transformers":
            raise ImportError("light mode imported " + name)
sys.meta_path.insert(0, Block())
ROOT = sys.argv[1]
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests")):
    sys.path.insert(0, p)
import test_glm_s1_attester as T
import glm_s1_attester as R
coord = T._acct(9)
store = T._sealed_store(coord)
relay = R.TeeRelay("http://47.84.93.96:8710", opener=store.opener)
lc = R.LightCore(relay, T.CAMP, coord.address, int(sys.argv[2]), log=lambda m: None)
lc.start(1)
seen = []
lc.step(lambda e, rec, extra, late: seen.append(e))
assert seen == [1, 2, 3, 4], seen
assert "torch" not in sys.modules and "transformers" not in sys.modules
if sys.argv[2] == "0":
    assert "sharddiloco_glm_coordinator" not in sys.modules
print("LIGHT_OK", len(sys.modules))
'''
    r = subprocess.run([sys.executable, "-c", script, ROOT, "96" if _HAS_COORD else "0"], capture_output=True, encoding="utf-8",
                       errors="replace", timeout=300)
    assert r.returncode == 0 and "LIGHT_OK" in r.stdout, (r.stdout[-2000:], r.stderr[-3000:])


def test_full_mode_refuses_without_the_private_coordinator(tmp_path, monkeypatch, capsys):
    """The public miner ships light mode only: --mode full must stop before loading a key or touching a path, with
    one clear message and a nonzero exit, instead of dying on an ImportError deep in build_full_core."""
    monkeypatch.setattr(R, "coordinator_available", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    args = ["--campaign", CAMP, "--key-file", str(tmp_path / "missing.json"), "--attest-dir", str(tmp_path / "a"),
            "--no-publish"]
    assert R.main(["--mode", "full", "--out", str(tmp_path / "o.jsonl"), "--probe-full-pool"] + args) == 2
    out = capsys.readouterr().out
    assert "full mode re-scores contributions and needs the private coordinator code" in out
    assert "ships light mode only" in out
    assert R.main(["--mode", "light", "--omission-window", "96"] + args) == 2
    assert "--omission-window > 0" in capsys.readouterr().out
    assert not (tmp_path / "a").exists()                       # refused before any directory or key work


def test_public_tree_has_no_coordinator_and_full_mode_cli_refuses(tmp_path):
    """Run as a subprocess exactly as an outsider would; only meaningful in the public tree (no coordinator)."""
    if _HAS_COORD:
        import pytest
        pytest.skip("private repo: the coordinator is present, full mode is allowed here")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "glm_s1_attester.py"), "--mode", "full",
                        "--out", str(tmp_path / "o.jsonl"), "--probe-full-pool", "--campaign", CAMP,
                        "--key-file", str(tmp_path / "k.json"), "--attest-dir", str(tmp_path / "a"), "--no-publish"],
                       capture_output=True, encoding="utf-8", errors="replace", timeout=120, env=env)
    assert r.returncode == 2 and "ships light mode only" in r.stdout, (r.stdout, r.stderr)
