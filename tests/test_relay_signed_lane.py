"""Relay signed writes + signed campaign pointer -- the client primitives shipped in 3.9.7.

Design: RELAY_SIGNED_WRITES_DESIGN_2026-11-26 (private repo docs/research). Both pieces are DEFAULT-OFF:
  * ContentLane takes the signed path (nh-store-put/2 bound to the relay's store id, per-name strictly
    increasing X-Sig-Ts, token-only shadow fallback) ONLY when its owner sets `lane.sign_shadow = True`.
    A lane without it sends exactly the 3.9.6 request.
  * neurahash.signed_put: the relay's per-prefix policy verdict + the coordinator-signed pointer
    (sign_pointer / verify_pointer / PointerGuard). Pure; nothing imports it unless opted in.
No network: the lane's _request is replaced by a recorder.
"""
import json
import os
import sys
import urllib.error

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_harness as H                    # noqa: E402

pytest.importorskip("eth_account")
from neura_l1.signing import gen_account           # noqa: E402
from neurahash import signed_put as SP             # noqa: E402
from neurahash import store_auth                   # noqa: E402

STORE_ID = "neurahash-relay-sg1"
NOW = 1795000000.0


def _recording_lane(monkeypatch, account, refuse_signed_at=None, **attrs):
    lane = H.ContentLane("http://127.0.0.1:9", "tok", account=account)
    for k, v in attrs.items():
        setattr(lane, k, v)
    sent = []

    def fake_request(req):
        hd = {k.lower(): v for k, v in req.header_items()}
        sent.append((req.full_url, req.get_method(), req.data, hd))
        if refuse_signed_at is not None and "x-sig" in hd and len(sent) == refuse_signed_at:
            raise urllib.error.HTTPError(req.full_url, 403, "refused", {}, None)
        return b'{"ok":true}'
    monkeypatch.setattr(lane, "_request", fake_request)
    monkeypatch.setattr(H.time, "time", lambda: NOW)
    return lane, sent


def test_signed_lane_sends_increasing_ts_per_name_and_falls_back_token_only(monkeypatch):
    logs = []
    lane, sent = _recording_lane(monkeypatch, gen_account(), refuse_signed_at=3,
                                 store_id=STORE_ID, sign_shadow=True, sign_log=logs.append)
    lane.put_json_named("n1", {"a": 1})
    lane.put_json_named("n1", {"a": 2})
    lane.put_json_named("n1", {"a": 3})               # signed attempt refused -> token-only re-send
    hds = [s[3] for s in sent]
    assert [int(h["x-sig-ts"]) for h in hds[:3]] == [int(NOW), int(NOW) + 1, int(NOW) + 2]
    assert hds[0]["x-sig-alg"] == store_auth.ALG_V2 and hds[0]["x-sig-store"] == STORE_ID
    assert "x-sig" not in hds[3] and hds[3]["x-auth"] == "tok" and hds[3]["x-name"] == "n1"
    assert any("re-sending it token-only" in m for m in logs)


def test_signed_lane_headers_pass_the_relay_verifier_and_policy(monkeypatch):
    acct = gen_account()
    lane, sent = _recording_lane(monkeypatch, acct, store_id=STORE_ID, sign_shadow=True)
    addr = acct.address.lower()
    name = "sharddiloco/glm/cosign/%s/%d/%s" % ("ab" * 8, 7, addr)
    cid = lane.put_json_named(name, {"x": 1})
    url, method, body, hd = sent[0]
    assert method == "PUT" and url.endswith("/o/" + cid) and H.cid_of(body) == cid
    hdr = {k.title(): v for k, v in hd.items()}
    ok, who = store_auth.verify_put(cid, hdr, now=NOW, store_id=STORE_ID, accept_v1=False)
    assert ok and who.lower() == addr, who
    policy = SP.parse_policy(json.dumps({"operators": ["0x" + "11" * 20], "rules": [
        {"prefix": "sharddiloco/glm/cosign/", "mode": "self", "addr_segment": 5, "max_names": 10},
        {"prefix": "sharddiloco/glm/pointer", "mode": "operator"}]}))
    assert SP.decide(policy, name, who, True)[0] is True
    assert SP.decide(policy, name, None, False)[:2] == (False, 403)            # token-only: refused
    other = "sharddiloco/glm/cosign/%s/%d/%s" % ("ab" * 8, 7, "0x" + "22" * 20)
    assert SP.decide(policy, other, who, True)[:2] == (False, 403)             # foreign embedded address
    assert SP.decide(policy, "sharddiloco/glm/pointer", who, True)[:2] == (False, 403)


def test_flags_off_lane_requests_are_the_396_requests(monkeypatch):
    """No sign_shadow: account=None -> no X-Sig-*; account set -> today's v1 headers, ts = now, no store."""
    lane0, sent0 = _recording_lane(monkeypatch, None)
    lane0.put_json_named("n", {"a": 1})
    assert not any(k.startswith("x-sig") for k in sent0[0][3])
    acct = gen_account()
    lane1, sent1 = _recording_lane(monkeypatch, acct)
    lane1.put_json_named("n", {"a": 1})
    lane1.put_json_named("n", {"a": 1})
    hd = sent1[0][3]
    assert hd["x-sig-alg"] == store_auth.ALG and "x-sig-store" not in hd
    assert sent1[0][3]["x-sig-ts"] == sent1[1][3]["x-sig-ts"] == str(int(NOW))
    assert "_name_sig_ts" not in lane1.__dict__


def _pointer(event=5, published_at=1795000000000, campaign="c" * 16):
    return {"v": 2, "event": event, "rounds": {"1_50": event}, "model_root": "ab" * 32,
            "done": False, "published_at": published_at, "campaign_id": campaign}


def test_signed_pointer_verifies_and_guard_refuses_forgery_and_rollback():
    coord, other = gen_account(), gen_account()
    p5 = SP.sign_pointer(coord, _pointer(5), "", STORE_ID)
    ok, info = SP.verify_pointer(p5, [coord.address], STORE_ID)
    assert ok and info["campaign"] == "c" * 16 and info["event"] == 5
    assert SP.verify_pointer(p5, [coord.address], "other-relay")[0] is False
    assert SP.verify_pointer(SP.sign_pointer(other, _pointer(6), "", STORE_ID), [coord.address], STORE_ID)[0] is False
    tampered = dict(p5, event=9)
    assert SP.verify_pointer(tampered, [coord.address], STORE_ID)[0] is False
    g = SP.PointerGuard([coord.address], STORE_ID)
    assert g.offer(p5)[0] is True
    assert g.offer(SP.sign_pointer(coord, _pointer(4, 1794999999000), "", STORE_ID))[0] is False
    assert g.current["event"] == 5
    # an old (pre-3.9.7) reader decodes a signed pointer unchanged: psig is an ignored extra key
    from neurahash import diloco_merge as dm
    assert dm.sd_pointer_decode(p5) == dm.sd_pointer_decode(_pointer(5))


# --------------------------------------------------------------------------- the contributor wiring (3.9.7)
import copy                                        # noqa: E402
import importlib.util                              # noqa: E402
import types                                       # noqa: E402

import sharddiloco_glm_contributor as N            # noqa: E402

ON = {"NEURAHASH_RELAY_SIGNED_WRITES": "1", "NEURAHASH_RELAY_SIGNED_POINTER": "1",
      "NEURAHASH_RELAY_STORE_ID": STORE_ID}
OFF = {"NEURAHASH_RELAY_SIGNED_WRITES": "0", "NEURAHASH_RELAY_SIGNED_POINTER": "0"}


def _args(tmp_path, wallet_file=None):
    return types.SimpleNamespace(wallet_file=str(wallet_file or (tmp_path / "no_such_key")))


def _contrib_lane(monkeypatch, tmp_path, wallet, env, wallet_file=None, refuse=None):
    """The contributor's lane exactly as main() builds it, then the 3.9.7 hooks with `env`."""
    lane, sent = _recording_lane(monkeypatch, wallet)
    if refuse:
        orig = lane._request

        def fake(req):
            hd = {k.lower(): v for k, v in req.header_items()}
            if hd.get("x-sig-alg") == store_auth.ALG_V2:
                sent.append((req.full_url, req.get_method(), req.data, hd))
                raise urllib.error.HTTPError(req.full_url, refuse, "refused", {}, None)
            return orig(req)
        monkeypatch.setattr(lane, "_request", fake)
    logs = []
    acct = N.configure_relay_signing(lane, wallet, _args(tmp_path, wallet_file), log=logs.append, env=env)
    N.install_relay_pointer_check(lane, log=logs.append, env=env)
    return lane, sent, logs, acct


def test_settings_default_on_and_zero_is_off():
    st = N.relay_sign_settings({})
    assert st["writes"] and st["pointer"] and not st["strict"]
    assert st["store_id"] == "neurahash-relay-sg1" and st["coord"] == [N.RELAY_COORD_ADDR_DEFAULT]
    off = N.relay_sign_settings(dict(OFF, NEURAHASH_RELAY_POINTER_STRICT="1"))
    assert not off["writes"] and not off["pointer"] and off["strict"]
    c = N.relay_sign_settings({"NEURAHASH_RELAY_COORD_ADDR": "0x" + "ab" * 20 + ", junk"})
    assert c["coord"] == ["0x" + "ab" * 20] and c["bad_coord"] == ["junk"]


def test_flags_off_contributor_lane_sends_the_396_bytes(monkeypatch, tmp_path):
    w = gen_account()
    ref_lane, ref = _recording_lane(monkeypatch, w)                 # 3.9.6: no hooks at all
    ref_lane.put_json_named("cg/x/r1/m.1", {"a": 1})
    ref_lane.put_blob(b"d")
    lane, sent, _, acct = _contrib_lane(monkeypatch, tmp_path, w, OFF)
    lane.put_json_named("cg/x/r1/m.1", {"a": 1})
    lane.put_blob(b"d")
    assert acct is None and sent == ref
    assert not hasattr(lane, "pointer_check") and not getattr(lane, "sign_shadow", False)


def test_keyless_signs_store_bound_v2_with_its_wallet_and_keeps_x_auth(monkeypatch, tmp_path):
    w = gen_account()
    lane, sent, logs, acct = _contrib_lane(monkeypatch, tmp_path, w, ON)
    cid = lane.put_json_named("miners/glm-x/card.json", {"c": 1})
    hd = sent[0][3]
    assert acct is w and hd["x-auth"] == "tok" and hd["x-sig-alg"] == store_auth.ALG_V2
    ok, who = store_auth.verify_put(cid, {k.title(): v for k, v in hd.items()}, now=NOW, store_id=STORE_ID,
                                    accept_v1=False)
    assert ok and who.lower() == w.address.lower()
    assert any("writes signed nh-store-put/2" in m for m in logs)


def test_keyed_signs_only_with_an_existing_wallet_file(monkeypatch, tmp_path):
    lane, sent, logs, acct = _contrib_lane(monkeypatch, tmp_path, None, ON)
    lane.put_blob(b"d")
    assert acct is None and not any(k.startswith("x-sig") for k in sent[0][3])
    assert not (tmp_path / "no_such_key").exists()                   # never created for a keyed miner
    w = gen_account()
    kf = tmp_path / "key"
    kf.write_text(w.key.hex())
    lane2, sent2, _, acct2 = _contrib_lane(monkeypatch, tmp_path, None, ON, wallet_file=kf, refuse=403)
    lane2.put_blob(b"d")
    assert acct2.address == w.address
    assert sent2[0][3]["x-sig-alg"] == store_auth.ALG_V2 and sent2[0][3]["x-auth"] == "tok"
    assert not any(k.startswith("x-sig") for k in sent2[1][3])      # fallback = keyed 3.9.6: token only


@pytest.mark.parametrize("code", [401, 403, 409, 413, 429])
def test_keyless_refusal_falls_back_to_its_396_v1_request(monkeypatch, tmp_path, code):
    w = gen_account()
    lane, sent, _, _ = _contrib_lane(monkeypatch, tmp_path, w, ON, refuse=code)
    lane.put_json_named("cg/x/r1/m.1", {"a": 1})
    assert sent[0][3]["x-sig-alg"] == store_auth.ALG_V2
    fb = sent[1][3]
    assert fb["x-sig-alg"] == store_auth.ALG and "x-sig-store" not in fb and fb["x-auth"] == "tok"


def test_three_refusals_switch_to_the_legacy_request_for_a_while(monkeypatch, tmp_path):
    w = gen_account()
    lane, sent, logs, _ = _contrib_lane(monkeypatch, tmp_path, w, ON, refuse=403)
    for k in range(3):
        lane.put_blob(b"d" + bytes([48 + k]))
    assert len(sent) == 6
    lane.put_blob(b"d3")                                             # backoff: no signed attempt
    assert len(sent) == 7 and sent[6][3]["x-sig-alg"] == store_auth.ALG
    assert any("in a row were refused" in m for m in logs)
    monkeypatch.setattr(H.time, "time", lambda: NOW + N.RELAY_SIGN_BACKOFF_S + 1)
    lane.put_blob(b"d4")
    assert sent[7][3]["x-sig-alg"] == store_auth.ALG_V2              # retried after the window


def _serve_pointer(monkeypatch, lane, ptr_box):
    monkeypatch.setattr(lane, "manifest", lambda: {"sharddiloco/glm/pointer": {"sha256": "x"}})
    monkeypatch.setattr(lane, "get_json", lambda cid: ptr_box[0])
    monkeypatch.setattr(H, "POINTER_NAME", "sharddiloco/glm/pointer")


def _bad_pointers(coord):
    return (_pointer(6),                                                          # unsigned
            SP.sign_pointer(gen_account(), _pointer(7), "", STORE_ID),            # forged (wrong key)
            SP.sign_pointer(coord, _pointer(4, 1794999999000), "", STORE_ID))     # rolled back


def test_pointer_check_report_mode_logs_loudly_and_keeps_using_the_pointer(monkeypatch, tmp_path):
    coord = gen_account()
    env = dict(ON, NEURAHASH_RELAY_COORD_ADDR=coord.address)
    lane, _, logs, _ = _contrib_lane(monkeypatch, tmp_path, gen_account(), env)
    box = [SP.sign_pointer(coord, _pointer(5), "", STORE_ID)]
    _serve_pointer(monkeypatch, lane, box)
    assert lane.read_pointer() == box[0]
    assert any("signed pointer VERIFIED" in m for m in logs)
    for bad in _bad_pointers(coord):
        box[0] = bad
        assert lane.read_pointer() == bad                             # report mode: still used
    loud = [m for m in logs if m.startswith("[relay-pointer] !!!")]
    assert len(loud) == 3 and all("using it anyway" in m for m in loud)
    assert lane.pointer_check.counts == {"ok": 1, "unsigned": 1, "bad": 1, "rollback": 1}
    clone = copy.copy(lane)                                           # the catch-up's _DeadlineLane clone
    assert clone.pointer_check is lane.pointer_check


def test_pointer_check_strict_refuses_bad_pointers(monkeypatch, tmp_path):
    coord = gen_account()
    env = dict(ON, NEURAHASH_RELAY_COORD_ADDR=coord.address, NEURAHASH_RELAY_POINTER_STRICT="1")
    lane, _, logs, _ = _contrib_lane(monkeypatch, tmp_path, gen_account(), env)
    good = SP.sign_pointer(coord, _pointer(5), "", STORE_ID)
    box = [good]
    _serve_pointer(monkeypatch, lane, box)
    assert lane.read_pointer() == good
    for bad in _bad_pointers(coord):
        box[0] = bad
        assert lane.read_pointer() is None
    loud = [m for m in logs if m.startswith("[relay-pointer] !!!")]
    assert len(loud) == 3 and all("REFUSED" in m for m in loud)


def test_launcher_sets_the_defaults_and_an_explicit_value_wins(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_glm_miner_t",
                                                  os.path.join(_REPO, "tools", "run_glm_miner.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k in ON:
        monkeypatch.delenv(k, raising=False)
    env = mod._child_env(None)
    assert {k: env[k] for k in ON} == {"NEURAHASH_RELAY_SIGNED_WRITES": "1", "NEURAHASH_RELAY_SIGNED_POINTER": "1",
                                       "NEURAHASH_RELAY_STORE_ID": "neurahash-relay-sg1"}
    monkeypatch.setenv("NEURAHASH_RELAY_SIGNED_WRITES", "0")
    assert mod._child_env(None)["NEURAHASH_RELAY_SIGNED_WRITES"] == "0"
