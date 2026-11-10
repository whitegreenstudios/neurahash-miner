"""ContentLane signed PUT (client half of docs/STORE_AUTH_DESIGN.md), as shipped in 3.9.0.

The GLM contributor now builds its lane as `H.ContentLane(url, token, account=wallet)`, so every
keyless miner signs its PUTs with the wallet key it already holds for identity. Two properties:
  * account=None is the v1 token-only request, unchanged (no X-Sig-* header);
  * with an account, X-Auth is still sent and the X-Sig-* headers verify to the wallet's address.
No network: urllib.request.urlopen is replaced by a recorder.
"""
import email.message
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_harness as H                    # noqa: E402

pytest.importorskip("eth_account")
from neura_l1 import signing                       # noqa: E402
from neurahash import store_auth                   # noqa: E402


class _Resp:
    def __init__(self, body):
        self._b = body

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def sent(monkeypatch):
    got = []

    def fake_urlopen(req, timeout=None):
        got.append(req)
        return _Resp(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(H.urllib.request, "urlopen", fake_urlopen)
    return got


def _headers(req):
    return {k.lower(): v for k, v in req.header_items()}


def test_without_an_account_the_put_is_token_only(sent):
    lane = H.ContentLane("http://lane.invalid:8710", "tok")
    cid = lane.put_blob(b"payload", name="sharddiloco/x")
    h = _headers(sent[-1])
    assert h["x-auth"] == "tok" and h["x-name"] == "sharddiloco/x"
    assert not [k for k in h if k.startswith("x-sig")]
    assert sent[-1].full_url.endswith("/o/" + cid)


def test_with_an_account_the_put_is_signed_and_still_carries_the_token(sent):
    acct = signing.gen_account()
    lane = H.ContentLane("http://lane.invalid:8710", "tok", account=acct)
    cid = lane.put_blob(b"payload", name="sharddiloco/x")
    req = sent[-1]
    h = _headers(req)
    assert h["x-auth"] == "tok"
    assert [k for k, _ in req.header_items()].count("X-name") == 1     # X-Name added exactly once
    wire = email.message.Message()            # what the store sees: case-insensitive, like HTTPMessage
    for k, v in req.header_items():
        wire[k] = v
    ok, who = store_auth.verify_put(cid, wire)
    assert ok, who
    assert who.lower() == acct.address.lower()


def test_the_contributor_hands_its_wallet_to_the_lane():
    src = open(os.path.join(_REPO, "tools", "sharddiloco_glm_contributor.py"), encoding="utf-8").read()
    assert "H.ContentLane(args.url, args.token, account=wallet)" in src
