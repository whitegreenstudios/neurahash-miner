"""X-Miner-Id: every request the miner makes to the relay carries ONE stable, non-secret id (release 3.8.5).

WHY. The relay -- the content store at 47.84.93.96:8710 -- identifies a sender only by a salted hash of its IP, so N
miners behind one home NAT look like one client (owner, 2026-10-02: "in real world, miner may have 10 or even more
miner in the same home address"). neurahash/miner_id.py derives one id per process from the miner's PUBLIC identity
and every relay request site sends it. This file proves it ON THE WIRE: real HTTP servers on 127.0.0.1, started and
stopped here, record the headers each request arrives with.

  * every relay request site sends X-Miner-Id matching ^[0-9a-f]{16}$, exactly once per request;
  * the value is stable across calls, sites and processes, and equals sha256(<"0x" + 40 lowercase hex>)[:16] --
    what the relay computes for a signer it verified, so one miner's PUT and GET log lines match;
  * two identities give two different values, and an entry point's pin wins over the wallet file;
  * a URL that may be a third party (HuggingFace, GitHub, an IPFS gateway) never receives it, and NEITHER DOES THE
    TARGET OF A REDIRECT: the live relay answers 302 to a public HuggingFace dataset for its cold tier, and urllib
    copies every header set via Request(headers=...) / add_header onto the redirected request (pre-release review,
    2026-10-02: the first patch leaked the id that way at all 6 sites tested);
  * no identity yet (no wallet file) -> no header; a release missing the helper still updates and mines.

No network beyond 127.0.0.1, no GPU, no real wallet: every identity is a throwaway key under tmp_path.

Run: C:/Python313/python.exe -m pytest tests/test_miner_id_header.py -q
"""
import contextlib
import hashlib
import http.server
import importlib.util
import inspect
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neurahash                                            # noqa: E402
from neura_l1.signing import gen_account                   # noqa: E402
from neurahash import miner_id as M                         # noqa: E402

ID_RE = re.compile(r"^[0-9a-f]{16}$")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _expected(address):
    return _sha(address.lower().encode("ascii"))[:16]


# ------------------------------------------------------------------ a recording content store on 127.0.0.1
class _Store(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.objects, self.names, self.seen = {}, {}, []
        self.redirect_get_to = self.redirect_put_to = None       # base URL: answer 302 to <base><same path>
        self.lock = threading.Lock()

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.server_address[1]

    def add(self, body, name=None):
        sha = _sha(body)
        with self.lock:
            self.objects[sha] = body
            if name:
                self.names[name] = {"sha256": sha, "size": len(body)}
        return sha

    def count(self):
        with self.lock:
            return len(self.seen)

    def since(self, n):
        with self.lock:
            return list(self.seen[n:])


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _note(self):
        vals = self.headers.get_all("X-Miner-Id") or []
        with self.server.lock:
            self.server.seen.append({"method": self.command, "path": self.path, "ids": list(vals)})

    def _send(self, code, body, ctype="application/json", location=None):
        self.send_response(code)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._note()
        s = self.server
        if s.redirect_get_to:                                     # the relay's cold tier: same address, other host
            return self._send(302, b"", location=s.redirect_get_to + self.path)
        with s.lock:
            names, obj = dict(s.names), s.objects.get(self.path[3:]) if self.path.startswith("/o/") else None
        if self.path == "/health":
            return self._send(200, json.dumps({"ok": True, "objects": len(s.objects)}).encode())
        if self.path == "/manifest":
            return self._send(200, json.dumps(names).encode())
        if self.path == "/release.json":
            return self._send(200, b"{}")
        if obj is not None:
            return self._send(200, obj, "application/octet-stream")
        return self._send(404, b'{"ok": false}')

    def do_PUT(self):
        self._note()
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.server.redirect_put_to:
            return self._send(302, b"", location=self.server.redirect_put_to + self.path)
        sha = self.path[3:] if self.path.startswith("/o/") else ""
        if _sha(body) != sha:
            return self._send(400, b'{"ok": false}')
        self.server.add(body, self.headers.get("X-Name"))
        return self._send(201, json.dumps({"ok": True, "sha256": sha}).encode())


@contextlib.contextmanager
def _serving():
    s = _Store()
    t = threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        yield s
    finally:
        s.shutdown()
        s.server_close()
        t.join(timeout=10)


@pytest.fixture
def store():
    """The relay: a lane (ContentLane registers it) or an explicitly registered content store."""
    with _serving() as s:
        yield s


@pytest.fixture
def third_party():
    """A SECOND server on another port that is never registered: stands in for HuggingFace / a gateway."""
    with _serving() as s:
        yield s


def _key_file(tmp_path, name):
    acct = gen_account()
    p = tmp_path / name
    p.write_text(acct.key.hex(), encoding="utf-8")
    return str(p), acct.address


@pytest.fixture
def wallet(tmp_path, monkeypatch):
    """A throwaway keyless wallet exactly where the GLM contributor looks for one (NEURAHASH_SD_WALLET)."""
    path, address = _key_file(tmp_path, "glm_miner_key")
    monkeypatch.setenv("NEURAHASH_SD_WALLET", path)
    M._reset()
    yield path, address, _expected(address)
    M._reset()


def _fake_hf(monkeypatch, FG):
    """fetch_glm_base.main without HuggingFace: every HF download writes "{}", the IPv4 patch is skipped."""
    def _fetch(api, rel_, dest_root, expect_sha=None, label="", local=None):
        out = os.path.join(dest_root, (local or rel_).replace("/", os.sep))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("{}")
        return out
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: types.SimpleNamespace())
    monkeypatch.setattr(FG, "_force_ipv4", lambda: None)
    monkeypatch.setattr(FG, "fetch", _fetch)


def _npz_bundle():
    buf = io.BytesIO()
    np.savez(buf, gsm_train=np.arange(4, dtype=np.int32))
    return buf.getvalue()


# ------------------------------------------------------------------ the derivation and the mechanism
def test_id_is_sha256_of_the_lowercase_address():
    acct = gen_account()
    want = _sha(acct.address.lower().encode("ascii"))[:16]
    assert M.id_from_address(acct.address) == want            # EIP-55 checksummed in
    assert M.id_from_address(acct.address.lower()) == want    # lowercase in
    assert ID_RE.match(want)
    for bad in ("", None, "0x123", acct.address[2:], "0x" + "g" * 40, "0x" + "a" * 64):
        assert M.id_from_address(bad) is None
    assert ID_RE.match(M.id_from_name("glm-1234abcd"))
    assert M.id_from_name("") is None


def test_name_derivation_never_raises():
    class _Unprintable:
        def __str__(self):
            raise RuntimeError("no str")
    M._reset()
    try:
        surrogate = M.pin_name("miner\udcff")                      # an undecodable argv byte (surrogateescape)
        assert ID_RE.match(surrogate) and M.miner_id() == surrogate
        assert surrogate != M.id_from_name("miner\udcfe")          # surrogatepass keeps such names distinct
        assert M.id_from_name(_Unprintable()) is None and M.id_from_address(_Unprintable()) is None
    finally:
        M._reset()


def test_attach_is_unredirected_so_urllib_never_forwards_it(wallet):
    _path, _address, want = wallet
    req = M.attach(urllib.request.Request("http://47.84.93.96:8710/o/" + "0" * 64))
    assert req.unredirected_hdrs.get("X-miner-id") == want and "X-miner-id" not in req.headers
    moved = urllib.request.HTTPRedirectHandler().redirect_request(
        req, None, 302, "Found", {}, "https://huggingface.co/datasets/x/resolve/main/" + "0" * 64)
    assert moved is not None and not moved.has_header("X-miner-id")      # what urllib's 302 handler would send
    gated = M.attach(urllib.request.Request("https://huggingface.co/x"), relay_only=True)
    assert not gated.has_header("X-miner-id")


def test_wallet_file_resolves_once_and_is_read_only(wallet):
    path, _address, want = wallet
    before = open(path, "rb").read()
    assert M.miner_id() == want
    assert M.miner_id() == want                                 # cached: one id per process
    assert open(path, "rb").read() == before


def test_no_identity_means_no_header_and_no_wallet_created(store, tmp_path, monkeypatch):
    import sharddiloco_harness as H
    absent = tmp_path / "nowhere" / "glm_miner_key"
    monkeypatch.setenv("NEURAHASH_SD_WALLET", str(absent))
    M._reset()
    try:
        assert M.miner_id() is None
        assert not M.attach(urllib.request.Request(store.url)).has_header("X-miner-id")
        H.ContentLane(store.url).health()
        assert store.since(0)[-1]["ids"] == []
        assert not absent.exists() and not absent.parent.exists()
    finally:
        M._reset()


def test_two_identities_give_two_ids(tmp_path, monkeypatch):
    ids = []
    for name in ("key_a", "key_b"):
        path, address = _key_file(tmp_path, name)
        monkeypatch.setenv("NEURAHASH_SD_WALLET", path)
        M._reset()
        ids.append(M.miner_id())
        assert ids[-1] == _expected(address)
    M._reset()
    assert ids[0] != ids[1] and all(ID_RE.match(i) for i in ids)
    assert M.id_from_name("glm-aaaa0000") != M.id_from_name("glm-aaaa0001")


def test_a_pin_wins_over_the_wallet_file(wallet):
    _path, _address, want = wallet
    assert M.miner_id() == want
    other = gen_account()
    assert M.pin_address(other.address) == _expected(other.address)
    assert M.miner_id() == _expected(other.address) != want
    assert M.pin_address("not-an-address") is None              # a bad pin changes nothing
    assert M.miner_id() == _expected(other.address)


def test_the_id_never_depends_on_a_secret(wallet):
    path, address, want = wallet
    key_hex = open(path, encoding="utf-8").read().strip().lower().replace("0x", "")
    a = M.attach(urllib.request.Request("http://47.84.93.96:8710/health", headers={"X-Auth": "token-1"}))
    b = M.attach(urllib.request.Request("http://47.84.93.96:8710/health", headers={"X-Auth": "token-2"}))
    assert a.get_header("X-miner-id") == b.get_header("X-miner-id") == want
    assert want not in key_hex and want == _expected(address)


def test_relay_only_gate():
    M._reset()
    try:
        assert M.is_relay_url("http://47.84.93.96:8710/o/" + "0" * 64)
        assert M.is_relay_url("http://47.84.93.96:8710/release.json")
        for third in ("https://huggingface.co/datasets/x/resolve/main/release.json",
                      "https://raw.githubusercontent.com/x/y/main/release.json",
                      "https://ipfs.io/ipfs/bafy", "https://47.84.93.96:8710/o/x", "http://47.84.93.96/o/x",
                      "http://127.0.0.1:1/o/x", "not a url", "", None):
            assert not M.is_relay_url(third), third
        M.register_relay("http://127.0.0.1:1/")
        assert M.is_relay_url("http://127.0.0.1:1/o/x") and not M.is_relay_url("http://127.0.0.1:2/o/x")
    finally:
        M._reset()


def test_same_wallet_same_id_across_processes(wallet):
    path, _address, want = wallet
    code = "import sys; sys.path.insert(0, %r); from neurahash import miner_id; print(miner_id.miner_id())" % _REPO
    env = dict(os.environ, NEURAHASH_SD_WALLET=path)
    outs = [subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
            for _ in range(2)]
    assert [o.stdout.strip() for o in outs] == [want, want], [o.stderr[-300:] for o in outs]


# ------------------------------------------------------------------ every relay request site, on the wire
def test_every_relay_request_site_sends_the_id(wallet, store, third_party, tmp_path, monkeypatch):
    _path, _address, want = wallet
    import sharddiloco_harness as H
    import sharddiloco_glm_contributor as N
    import self_update as SU
    import fetch_glm_base as FG
    import fetch_ids_bundle as FI
    import bundle_pointer as BP
    import ipfs_checkpoint as IC
    import glm_rollout_worker as RW
    import diloco_contributor as DC
    from neurahash import corpus_sync as CS

    report = []

    def site(name, fn, relay_hits=1, third_hits=0):
        n0, t0 = store.count(), third_party.count()
        fn()
        got, other = store.since(n0), third_party.since(t0)
        report.append((name, len(got), sorted({i for g in got for i in g["ids"]}), [o["ids"] for o in other]))
        assert len(got) >= relay_hits, (name, got)
        assert all(g["ids"] == [want] for g in got), (name, got)          # present, exactly once, stable
        assert len(other) >= third_hits and all(o["ids"] == [] for o in other), (name, other)

    lane = H.ContentLane(store.url, "tok")                    # registers store.url as a relay origin
    blob = b"miner-id blob"
    cid = _sha(blob)
    site("harness ContentLane.health (GET /health)", lambda: lane.health())
    site("harness ContentLane.manifest (GET /manifest)", lambda: lane.manifest())
    site("harness ContentLane.put_blob (PUT /o/)", lambda: lane.put_blob(blob, name="t/blob"))
    site("harness ContentLane.get_blob (GET /o/)", lambda: lane.get_blob(cid))

    data = b"\x01" * 1000
    dsha = store.add(data)
    third_party.add(data)
    site("contributor data_http_get -> _default_urlopen (relay seed)",
         lambda: N.data_http_get(store.url + "/o/" + dsha, timeout=5, expected_size=len(data),
                                 dest_path=str(tmp_path / "d1")))
    site("contributor data_http_get -> _default_urlopen (third-party seed: no id)",
         lambda: N.data_http_get(third_party.url + "/o/" + dsha, timeout=5, expected_size=len(data),
                                 dest_path=str(tmp_path / "d2")), relay_hits=0, third_hits=1)

    rel = store.url + "/release.json"
    monkeypatch.setattr(SU, "_ALLOWED_HTTP_URLS", frozenset({rel}))
    site("self_update _default_fetch (relay mirror)", lambda: SU._default_fetch(rel, timeout=5))

    csha = store.add(b'{"model_type": "glm4_moe_lite"}')
    _fake_hf(monkeypatch, FG)
    monkeypatch.setattr(sys, "argv", ["fetch_glm_base.py", "--dest", str(tmp_path / "fg"), "--skip-trunk",
                                      "--pieces", "0", "--lane", store.url, "--config-cid", csha])
    site("fetch_glm_base main: config GET /o/", lambda: FG.main())

    store.add(_npz_bundle(), name="glm/ids/test")
    monkeypatch.setattr(sys, "argv", ["fetch_ids_bundle.py", "--url", store.url, "--name", "glm/ids/test",
                                      "--dest", str(tmp_path / "fi")])
    site("fetch_ids_bundle main: GET /manifest + GET /o/", lambda: FI.main(), relay_hits=2)

    bundle = b"PK-bundle"
    bsha = store.add(bundle)
    rec = {"sha256": bsha, "size": len(bundle),
           "seeds": [third_party.url + "/o/{sha}", store.url + "/o/{sha}"]}       # third party first: 404
    site("bundle_pointer resolve_bundle -> _http_get (third party first, then relay)",
         lambda: BP.resolve_bundle(rec, str(tmp_path / "bundle.zip")), third_hits=1)

    trk = json.dumps({"round": 1, "checkpoint_cid": "bafy-x"}).encode()
    tsha = store.add(trk, name="tracker")
    third_party.add(trk)
    site("ipfs_checkpoint read_tracker (relay url)", lambda: IC.read_tracker(store.url + "/o/" + tsha))
    site("ipfs_checkpoint read_tracker (third-party url: no id)",
         lambda: IC.read_tracker(third_party.url + "/o/" + tsha), relay_hits=0, third_hits=1)
    site("ipfs_checkpoint read_tracker_from_store -> _store_get_named", lambda: IC.read_tracker_from_store(store.url),
         relay_hits=2)
    site("ipfs_checkpoint announce_pin (PUT /o/)", lambda: IC.announce_pin(store.url, "bafy-c", "peer1", token="t"))
    site("ipfs_checkpoint known_pinners (GET /manifest + GET /o/)",
         lambda: IC.known_pinners(store.url, "bafy-c"), relay_hits=2)

    for k in ("NEURAHASH_MINER_KEY", "NEURAHASH_DILOCO_REGISTRY", "NEURAHASH_DILOCO_MERGE_URLS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(DC.ic, "_pinata_jwt", lambda *a, **k: "")
    monkeypatch.setattr(DC.ic, "publish", lambda path: "bafy-delta")
    site("diloco_contributor publish_delta (registry PUT /o/)",
         lambda: DC.publish_delta(str(tmp_path / "delta.npz"), "minerX", 3, registry_url=store.url,
                                  registry_token="t"))

    ksha = store.add(b'{"task_id": "t1"}\n')
    site("glm_rollout_worker _fetch_content_addressed (third party first, then relay)",
         lambda: RW._fetch_content_addressed([third_party.url, store.url], ksha, timeout=5), third_hits=1)

    site("corpus_sync fetch_manifest (GET /manifest)", lambda: CS.fetch_manifest(store.url))
    site("corpus_sync _http_get (GET /o/)", lambda: CS._http_get(store.url + "/o/" + dsha, 5))

    print("\nX-Miner-Id expected=%s  relay requests=%d  third-party requests=%d"
          % (want, sum(r[1] for r in report), sum(len(r[3]) for r in report)))
    for name, n, ids, other in report:
        print("  %-78s relay=%d id=%s third=%s" % (name, n, ids or "-", other or "-"))
    assert sum(r[1] for r in report) >= 20


def test_a_redirect_target_never_receives_the_id(wallet, store, third_party, tmp_path, monkeypatch):
    """REVIEW BLOCKER 2026-10-02 (privacy). The live relay answers GET /o/<sha> with a 302 to a public HuggingFace
    dataset for an object in its cold tier, and urllib copies every header set through Request(headers=...) or
    add_header onto the redirected request (urllib/request.py redirect_request). Here the relay (`store`) answers
    EVERY GET with a 302 to `third_party`, which serves the same bytes: at every GET site the relay must see the id
    and the redirect target must not. PUTs are never followed by urllib (it raises on a 302 for PUT); that kind is
    checked too. Against the first patch (headers via Request(headers=...)), every GET site here fails."""
    _path, _address, want = wallet
    import sharddiloco_harness as H
    import sharddiloco_glm_contributor as N
    import self_update as SU
    import fetch_glm_base as FG
    import fetch_ids_bundle as FI
    import bundle_pointer as BP
    import ipfs_checkpoint as IC
    import glm_rollout_worker as RW
    from neurahash import corpus_sync as CS

    third_party.objects, third_party.names = store.objects, store.names       # the cold tier holds the same bytes
    store.redirect_get_to = third_party.url
    lane = H.ContentLane(store.url, "tok")                    # registers store.url as a relay origin
    rows = []

    def site(name, fn):
        n0, t0 = store.count(), third_party.count()
        fn()
        rows.append((name, [g["ids"] for g in store.since(n0)], [m["ids"] for m in third_party.since(t0)]))

    blob = b"cold blob"
    bsha = store.add(blob)
    site("harness ContentLane.health", lambda: lane.health())
    site("harness ContentLane.manifest", lambda: lane.manifest())
    site("harness ContentLane.get_blob", lambda: lane.get_blob(bsha))
    site("contributor data_http_get", lambda: N.data_http_get(store.url + "/o/" + bsha, timeout=5,
                                                              expected_size=len(blob), dest_path=str(tmp_path / "c")))
    rel = store.url + "/release.json"
    monkeypatch.setattr(SU, "_ALLOWED_HTTP_URLS", frozenset({rel}))
    site("self_update _default_fetch", lambda: SU._default_fetch(rel, timeout=5))
    csha = store.add(b'{"model_type": "glm4_moe_lite"}')
    _fake_hf(monkeypatch, FG)
    monkeypatch.setattr(sys, "argv", ["fetch_glm_base.py", "--dest", str(tmp_path / "fg"), "--skip-trunk",
                                      "--pieces", "0", "--lane", store.url, "--config-cid", csha])
    site("fetch_glm_base config GET", lambda: FG.main())
    store.add(_npz_bundle(), name="glm/ids/cold")
    monkeypatch.setattr(sys, "argv", ["fetch_ids_bundle.py", "--url", store.url, "--name", "glm/ids/cold",
                                      "--dest", str(tmp_path / "fi")])
    site("fetch_ids_bundle _get (x2)", lambda: FI.main())
    site("bundle_pointer _http_get", lambda: BP.resolve_bundle(
        {"sha256": bsha, "size": len(blob), "seeds": [store.url + "/o/{sha}"]}, str(tmp_path / "b.zip")))
    tsha = store.add(json.dumps({"round": 2, "checkpoint_cid": "bafy-y"}).encode(), name="tracker")
    store.add(json.dumps({"cid": "bafy-p", "peer_id": "p1", "ts": int(time.time())}).encode(), name="pinner-p1")
    site("ipfs_checkpoint read_tracker", lambda: IC.read_tracker(store.url + "/o/" + tsha))
    site("ipfs_checkpoint _store_get_named (x2)", lambda: IC.read_tracker_from_store(store.url))
    site("ipfs_checkpoint known_pinners (x2)", lambda: IC.known_pinners(store.url, "bafy-p"))
    site("glm_rollout_worker _fetch_content_addressed",
         lambda: RW._fetch_content_addressed([store.url], bsha, timeout=5))
    site("corpus_sync fetch_manifest", lambda: CS.fetch_manifest(store.url))
    site("corpus_sync _http_get", lambda: CS._http_get(store.url + "/o/" + bsha, 5))

    store.redirect_get_to, store.redirect_put_to = None, third_party.url        # the PUT kind
    n0, t0 = store.count(), third_party.count()
    with pytest.raises(urllib.error.HTTPError):
        lane.put_blob(b"put kind", name="t/put")
    rows.append(("harness ContentLane.put_blob (PUT: urllib does not follow)",
                 [g["ids"] for g in store.since(n0)], [m["ids"] for m in third_party.since(t0)]))

    print("\nX-Miner-Id expected=%s; relay answers every GET with 302 -> %s" % (want, third_party.url))
    for name, at_relay, at_target in rows:
        print("  %-58s relay ids=%s  redirect target ids=%s" % (name, at_relay, at_target))
    leaks = [(n, t) for n, _r, t in rows if any(t)]
    assert not leaks, "the redirect target received X-Miner-Id: %s" % leaks
    assert all(r and all(ids == [want] for ids in r) for _n, r, _t in rows), rows
    assert all(t for n, _r, t in rows if "PUT" not in n), "a GET site did not follow the redirect: %s" % rows


@pytest.mark.parametrize("rel", ["tools/self_update.py", "tools/sharddiloco_harness.py",
                                 "tools/sharddiloco_glm_contributor.py", "tools/ipfs_checkpoint.py"])
def test_a_release_missing_the_helper_still_updates_and_mines(rel, monkeypatch):
    """3.7.1 rule. run_glm_miner.py runs self_update only at startup, so the updater AND the contributor's import chain
    (contributor -> sharddiloco_harness; sharddiloco_glm_expert -> diloco_contributor -> ipfs_checkpoint) must load
    without neurahash/miner_id.py: no header, never a crash. (The import gate still fails such a release.)"""
    monkeypatch.delattr(neurahash, "miner_id", raising=False)
    monkeypatch.setitem(sys.modules, "neurahash.miner_id", None)            # `from neurahash import miner_id` fails
    with pytest.raises(ImportError):
        from neurahash import miner_id  # noqa: F401
    loaded_before = set(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location("_nomid_" + os.path.basename(rel)[:-3],
                                                      os.path.join(_REPO, rel))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:                                     # never leave a header-less copy of a module for later tests
        for name in set(sys.modules) - loaded_before:
            if getattr(sys.modules.get(name), "_miner_id", "absent") is None:
                sys.modules.pop(name, None)
    assert mod._miner_id is None
    if rel.endswith("self_update.py"):
        seen = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read1(self, n):
                return b""

        def _urlopen(req, timeout=None):
            seen["header"] = req.get_header("X-miner-id")
            return _Resp()
        monkeypatch.setattr(mod.urllib.request, "urlopen", _urlopen)
        mod._default_fetch(mod.VPS_MANIFEST_URL, timeout=5)
        assert seen == {"header": None}


def test_self_update_sends_the_id_to_the_vps_mirror_only(wallet, monkeypatch):
    _path, _address, want = wallet
    import self_update as SU
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read1(self, n):
            return b""

    def _fake_urlopen(req, timeout=None):
        seen[req.full_url] = (req.get_header("X-miner-id"), "X-miner-id" in req.headers)
        return _Resp()

    monkeypatch.setattr(SU.urllib.request, "urlopen", _fake_urlopen)
    for _name, url in SU.MIRRORS:
        SU._default_fetch(url, timeout=5)
    assert seen[SU.VPS_MANIFEST_URL] == (want, False)                       # present, and unredirected
    assert seen[SU.MANIFEST_URL] == (None, False) and seen[SU.HF_MANIFEST_URL] == (None, False)


def test_legacy_contributor_pins_its_signing_key(wallet, store, tmp_path, monkeypatch):
    _path, _address, glm_id = wallet
    import diloco_contributor as DC
    key_path, address = _key_file(tmp_path, "legacy_key")
    monkeypatch.setenv("NEURAHASH_MINER_KEY", key_path)
    for k in ("NEURAHASH_DILOCO_REGISTRY", "NEURAHASH_DILOCO_MERGE_URLS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(DC.ic, "_pinata_jwt", lambda *a, **k: "")
    monkeypatch.setattr(DC.ic, "publish", lambda path: "bafy-delta")
    DC.publish_delta(str(tmp_path / "delta.npz"), "minerY", 1, registry_url=store.url, registry_token="t")
    assert store.since(0)[-1]["ids"] == [_expected(address)] != [glm_id]


def test_rollout_worker_pins_its_wallet_before_any_lane_request(tmp_path, monkeypatch):
    import glm_rollout_worker as RW
    monkeypatch.setenv("NEURAHASH_SD_WALLET", str(tmp_path / "unused"))
    M._reset()
    try:
        wallet_file = str(tmp_path / "rw_key")
        seen = {}

        def _pointer(lane):
            seen["id_at_first_request"] = M.miner_id()
            return {}

        args = types.SimpleNamespace(wallet_file=wallet_file, max_tasks=None, once=False)
        assert RW.run_worker(args, lane=object(), backend=object(), tasks=[], fetch_pointer_fn=_pointer,
                             log=lambda m: None) == 0
        from neura_l1.signing import account_from_key
        address = account_from_key(open(wallet_file, encoding="utf-8").read().strip()).address
        assert seen["id_at_first_request"] == _expected(address)
    finally:
        M._reset()


def test_contributor_pins_its_identity_before_its_lane_exists():
    """main() loads a model, so it is not run here; its wiring is checked in order instead: the pin follows
    _resolve_identity and precedes the ContentLane every later relay request rides."""
    import sharddiloco_glm_contributor as N
    src = inspect.getsource(N.main)
    i_res = src.index("_resolve_identity(args")
    i_pin = src.index("_miner_id.pin_address(wallet.address)")
    i_name = src.index("_miner_id.pin_name(")
    i_lane = src.index("H.ContentLane(args.url")
    assert i_res < i_pin < i_lane and i_res < i_name < i_lane
