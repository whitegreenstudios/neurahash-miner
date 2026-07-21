"""Gate for the SIGNED NETWORK MANIFEST (docs/MINER_MANIFEST_DESIGN.md).

WHY THIS SUITE EXISTS. A joiner (issue #71) ran a client whose CODE required three environment
variables while the DOCS named two, got an opaque HTTP 401, and then had no channel to receive the
fix once it was written. Code, config and the network's expectations travelled separately and
silently disagreed. The manifest collapses them into ONE signed artifact -- which is only an
improvement if it cannot be abused. So this suite pins the properties that make it safe:

  * MIRRORS are an availability mechanism, never a trust one: the highest VERIFIED version across
    all mirrors wins, a mirror serving a FORGED manifest is ignored while a good mirror still wins,
    and every mirror being down is a NON-EVENT (keep the code and config we already have);
  * `config` is applied as DEFAULTS ONLY -- explicit environment always wins -- through a STRICT
    ALLOWLIST, so the network can never inject a path, a credential, or an unknown knob;
  * `min_client_version` REFUSES TO PUBLISH but KEEPS TRAINING, naming the reason (a stranger must
    never get a crash, and must never submit what the network will reject);
  * the v2 optional fields are SIGNED when present, so they can be neither added to nor stripped
    from a manifest by anyone but the release-key holder.

NO network, git, pip, or process replacement: every side effect is an injected fake, and the
verifying entrypoints are bound to a TEST key so nothing here depends on the real pinned key.
"""
import functools
import json
import types

import pytest

from neura_l1.signing import account_from_key, sign_bytes
from tools import run_miner
from tools import self_update as su

TEST_PRIV = "0x" + "11" * 32
TEST_PUBKEY = account_from_key(TEST_PRIV).address
WRONG_PRIV = "0x" + "22" * 32
COMMIT = "b" * 40

verify_manifest = functools.partial(su.verify_manifest, pubkey=TEST_PUBKEY)
fetch_best_manifest = functools.partial(su.fetch_best_manifest, pubkey=TEST_PUBKEY)
sync_from_manifest = functools.partial(su.sync_from_manifest, pubkey=TEST_PUBKEY)

_ENV_KEYS = ("NEURAHASH_DILOCO_MERGE_URL", "NEURAHASH_CONTENT_URL", "NEURAHASH_CORPUS_SHA",
             "NEURAHASH_SIGNED_PUT", "NEURAHASH_CONTRIB_SIG_VERSION", "NEURAHASH_CONTENT_TOKEN",
             "NEURAHASH_MINER_KEY", "NEURAHASH_AUTOUPDATE", "PINATA_JWT", "PINATA_JWT_FILE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from an EMPTY relevant environment and a disarmed publish block, so no
    test can pass because of another test's leftovers (or the developer's own shell)."""
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(run_miner, "PUBLISH_BLOCK_REASON", None, raising=False)
    yield
    run_miner.PUBLISH_BLOCK_REASON = None


def sign(version="0.2.0", commit=COMMIT, ts=1721001600, priv=TEST_PRIV,
         min_client_version=None, config=None):
    """Produce a signed manifest exactly the way tools/sign_release.py does, optionally carrying
    the v2 fields. The signature covers whatever optional fields are present."""
    acct = account_from_key(priv)
    body = {"version": version, "git_commit": commit, "published_ts": ts}
    if min_client_version is not None:
        body["min_client_version"] = min_client_version
    if config is not None:
        body["config"] = config
    m = dict(body)
    m["signature"] = sign_bytes(acct, su.canonical_manifest_bytes(body))
    m["signer"] = acct.address
    return m


def mirrors_serving(*payloads):
    """Build (mirrors, fetch_fn) where mirror i serves payloads[i]. A payload may be a manifest
    dict, a raw string, or an Exception instance to raise (an unreachable host)."""
    mirrors = tuple((f"m{i}", f"https://mirror{i}.example/release.json")
                    for i in range(len(payloads)))
    by_url = {u: p for (_n, u), p in zip(mirrors, payloads)}

    def _fetch(url, timeout=None):
        p = by_url[url]
        if isinstance(p, Exception):
            raise p
        return p if isinstance(p, str) else json.dumps(p)

    return mirrors, _fetch


def make_repo(tmp_path, version="0.1.0", reqs="numpy\n"):
    (tmp_path / "VERSION").write_text(version + "\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(reqs, encoding="utf-8")
    return str(tmp_path)


class FakeGit:
    def __init__(self):
        self.calls = []
        self.head = None

    def __call__(self, repo, *args, **kw):
        self.calls.append(args)
        if args[0] == "checkout" and args[-1] != "-":
            self.head = args[-1]
            return 0, ""
        if args[0] == "rev-parse":
            return 0, (self.head or "") + "\n"
        return 0, ""

    def checkouts(self):
        return [a[-1] for a in self.calls if a and a[0] == "checkout" and a[-1] != "-"]


class FakeReexec:
    def __init__(self):
        self.called = False

    def __call__(self, argv):
        self.called = True


class FakePip:
    def __init__(self):
        self.calls = 0

    def __call__(self, repo, **kw):
        self.calls += 1
        return 0, ""


# =============================================================== CRITERION 1: mirrors, best wins
def test_highest_valid_version_wins_across_three_mirrors():
    mirrors, fetch = mirrors_serving(sign("0.2.0"), sign("0.4.0"), sign("0.3.0"))
    res = fetch_best_manifest(mirrors, fetch_fn=fetch)
    assert res.ok is True
    assert res.manifest["version"] == "0.4.0"           # highest, not first and not last
    assert res.source == "https://mirror1.example/release.json"
    assert [s for _n, _u, s in res.tried] == ["valid v0.2.0", "valid v0.4.0", "valid v0.3.0"]


def test_forged_mirror_is_ignored_and_a_valid_mirror_still_wins():
    # mirror0 is signed by the WRONG key, mirror1 has a tampered version (signature no longer
    # covers it), mirror2 is honest -- and mirror0 claims the HIGHEST version, so a naive
    # "take the biggest number" implementation would hand the fleet to the attacker.
    forged_high = sign("9.9.9", priv=WRONG_PRIV)
    tampered = sign("0.2.0")
    tampered["version"] = "8.8.8"
    mirrors, fetch = mirrors_serving(forged_high, tampered, sign("0.3.0"))
    res = fetch_best_manifest(mirrors, fetch_fn=fetch)
    assert res.ok is True
    assert res.manifest["version"] == "0.3.0"           # the only VERIFIED one wins
    assert res.source == "https://mirror2.example/release.json"
    assert res.tried[0][2].startswith("REJECTED") and res.tried[1][2].startswith("REJECTED")


def test_all_mirrors_unreachable_keeps_current_code_and_never_raises(tmp_path):
    repo = make_repo(tmp_path, version="0.1.0")
    before = (tmp_path / "VERSION").read_bytes()
    mirrors, fetch = mirrors_serving(OSError("dns"), TimeoutError("t/o"), OSError("reset"))
    git, rex = FakeGit(), FakeReexec()
    sync = sync_from_manifest(repo, argv=["tools/run_miner.py"], mirrors=mirrors, fetch_fn=fetch,
                              git_fn=git, pip_fn=FakePip(), reexec_fn=rex, environ={},
                              state_path=str(tmp_path / "s.json"))
    assert sync.manifest is None and sync.publish_block is None and sync.config_applied == []
    assert git.checkouts() == [] and rex.called is False
    assert (tmp_path / "VERSION").read_bytes() == before
    assert sync.fetch.tried and all("unreachable" in s for _n, _u, s in sync.fetch.tried)


def test_garbage_and_forgery_on_every_mirror_is_still_a_no_op(tmp_path):
    repo = make_repo(tmp_path, version="0.1.0")
    mirrors, fetch = mirrors_serving("not json at all", sign("9.9.9", priv=WRONG_PRIV),
                                     {"version": "9.9.9"})
    git, rex = FakeGit(), FakeReexec()
    sync = sync_from_manifest(repo, argv=["m"], mirrors=mirrors, fetch_fn=fetch, git_fn=git,
                              pip_fn=FakePip(), reexec_fn=rex, environ={},
                              state_path=str(tmp_path / "s.json"))
    assert sync.manifest is None
    assert git.checkouts() == [] and rex.called is False


def test_one_good_mirror_among_dead_ones_still_updates(tmp_path):
    repo = make_repo(tmp_path, version="0.1.0")
    mirrors, fetch = mirrors_serving(OSError("dns"), sign("0.2.0"), OSError("reset"))
    git, rex = FakeGit(), FakeReexec()
    sync = sync_from_manifest(repo, argv=["tools/run_miner.py"], mirrors=mirrors, fetch_fn=fetch,
                              git_fn=git, pip_fn=FakePip(), reexec_fn=rex, environ={},
                              state_path=str(tmp_path / "s.json"))
    assert sync.manifest_version == "0.2.0"
    assert git.checkouts() == [COMMIT] and rex.called is True


# ==================================================== CRITERION 2: v1 compatibility of the schema
def test_optional_fields_do_not_change_v1_canonical_bytes():
    v1 = {"version": "0.1.0", "git_commit": "a" * 40, "published_ts": 1784444362}
    assert su.canonical_manifest_bytes(v1) == su.canonical_manifest_bytes(dict(v1, signer="0xdead"))
    # present-but-EMPTY declares nothing, so it must be byte-identical to omitting the key --
    # otherwise the same declared content has two different signable representations.
    for empty in ({"config": None}, {"config": {}}, {"min_client_version": None},
                  {"min_client_version": ""}):
        assert su.canonical_manifest_bytes(dict(v1, **empty)) == su.canonical_manifest_bytes(v1)
    # a manifest that CARRIES the optional fields must canonicalise DIFFERENTLY (they are signed)
    v2 = dict(v1, min_client_version="0.1.1", config={"merge_url": "http://x.example"})
    assert su.canonical_manifest_bytes(v2) != su.canonical_manifest_bytes(v1)


def test_config_cannot_be_added_to_or_stripped_from_a_signed_manifest():
    m = sign("0.2.0", config={"merge_url": "http://good.example"})
    assert verify_manifest(m)[0] is True
    injected = dict(m)
    injected["config"] = {"merge_url": "http://evil.example"}
    assert verify_manifest(injected)[0] is False        # swapped -> signature no longer recovers
    stripped = dict(m)
    del stripped["config"]
    assert verify_manifest(stripped)[0] is False        # removed -> signature no longer recovers
    plain = sign("0.2.0")
    added = dict(plain)
    added["config"] = {"merge_url": "http://evil.example"}
    assert verify_manifest(added)[0] is False           # added to a v1 manifest -> rejected


def test_min_client_version_is_signed_and_malformed_values_are_rejected():
    m = sign("0.2.0", min_client_version="0.1.1")
    assert verify_manifest(m)[0] is True
    bumped = dict(m)
    bumped["min_client_version"] = "9.9.9"              # would lock every miner out if unsigned
    assert verify_manifest(bumped)[0] is False
    bad = sign("0.2.0", min_client_version="not-a-version")
    ok, reason = verify_manifest(bad)
    assert ok is False and "min_client_version malformed" in reason


def test_non_object_config_is_rejected_outright():
    m = sign("0.2.0", config={"merge_url": "http://x.example"})
    m["config"] = ["merge_url", "http://x.example"]
    ok, reason = verify_manifest(m)
    assert ok is False and "config is not a JSON object" in reason


# ================================================ CRITERION 3: config is DEFAULTS through an ALLOWLIST
def test_config_is_applied_as_defaults():
    env = {}
    applied, ignored = su.apply_manifest_config({
        "merge_url": "http://47.84.93.96:8710",
        "content_url": "http://47.84.93.96:8710",
        "corpus_sha": "FA73AE43DEADBEEF",
        "protocol": {"signed_put": True, "contrib_sig_version": "neurahash-diloco-contrib-v1"},
    }, environ=env)
    assert env["NEURAHASH_DILOCO_MERGE_URL"] == "http://47.84.93.96:8710"
    assert env["NEURAHASH_CONTENT_URL"] == "http://47.84.93.96:8710"
    assert env["NEURAHASH_CORPUS_SHA"] == "fa73ae43deadbeef"
    assert env["NEURAHASH_SIGNED_PUT"] == "1"
    assert env["NEURAHASH_CONTRIB_SIG_VERSION"] == "neurahash-diloco-contrib-v1"
    assert len(applied) == 5 and ignored == []


def test_explicit_env_beats_config():
    env = {"NEURAHASH_DILOCO_MERGE_URL": "http://my-own-box:9999",
           "NEURAHASH_CONTRIB_SIG_VERSION": "pinned-by-operator"}
    applied, ignored = su.apply_manifest_config({
        "merge_url": "http://network-says.example",
        "content_url": "http://network-says.example",
        "protocol": {"contrib_sig_version": "network-says"},
    }, environ=env)
    assert env["NEURAHASH_DILOCO_MERGE_URL"] == "http://my-own-box:9999"     # operator wins
    assert env["NEURAHASH_CONTRIB_SIG_VERSION"] == "pinned-by-operator"      # operator wins
    assert env["NEURAHASH_CONTENT_URL"] == "http://network-says.example"     # unset -> defaulted
    assert applied == ["NEURAHASH_CONTENT_URL=http://network-says.example"]
    assert any("explicit NEURAHASH_DILOCO_MERGE_URL" in i for i in ignored)


def test_unknown_config_key_is_ignored_not_applied():
    env = {}
    applied, ignored = su.apply_manifest_config({
        "merge_url": "http://ok.example",
        "some_future_knob": "whatever",
        "protocol": {"signed_put": True, "future_protocol_flag": "x"},
    }, environ=env)
    assert env == {"NEURAHASH_DILOCO_MERGE_URL": "http://ok.example",
                   "NEURAHASH_SIGNED_PUT": "1"}
    assert "some_future_knob (not on the allowlist)" in ignored
    assert "protocol.future_protocol_flag (not on the allowlist)" in ignored
    assert len(applied) == 2


@pytest.mark.parametrize("cfg", [
    {"miner_key": "C:/Users/victim/.ssh/id_rsa"},          # a filesystem path
    {"content_token": "super-secret"},                      # a credential
    {"PINATA_JWT": "eyJ..."},                               # a credential by its real env name
    {"python": "python -c 'import os;os.system(1)'"},        # anything executable
    {"merge_url": "file:///C:/Windows/System32"},            # non-http scheme
    {"merge_url": "http://x.example\nX-Evil: 1"},            # header injection via a newline
    {"merge_url": "http://user:pass@evil.example"},          # a CREDENTIAL smuggled in a url
    {"merge_url": "http://x.example\x00.evil"},              # null byte
    {"corpus_sha": "../../etc/passwd"},                      # traversal in a digest slot
    {"protocol": {"signed_put": "yes-please"}},              # wrong type for a boolean
    # the ALLOWLISTED free-text slot is the one that actually has to hold the line: it reaches
    # every child process through os.environ, so it must never read as a path or as a flag.
    {"protocol": {"contrib_sig_version": "../../../etc/passwd"}},
    {"protocol": {"contrib_sig_version": "/etc/passwd"}},
    {"protocol": {"contrib_sig_version": ".."}},
    {"protocol": {"contrib_sig_version": "x/../../y"}},
    {"protocol": {"contrib_sig_version": "--publish-delta"}},
    {"protocol": {"contrib_sig_version": "a b"}},
    {"protocol": {"contrib_sig_version": "a\nb"}},
])
def test_config_can_never_set_a_path_credential_or_executable(cfg):
    env = {}
    applied, ignored = su.apply_manifest_config(cfg, environ=env)
    assert env == {} and applied == [] and ignored


def test_config_is_only_applied_from_a_VERIFIED_manifest(tmp_path):
    """A forged manifest carrying a juicy `config` must not move a single environment variable."""
    repo = make_repo(tmp_path, version="0.1.0")
    forged = sign("9.9.9", priv=WRONG_PRIV, config={"merge_url": "http://attacker.example"})
    mirrors, fetch = mirrors_serving(forged)
    env = {}
    sync = sync_from_manifest(repo, argv=["m"], mirrors=mirrors, fetch_fn=fetch, git_fn=FakeGit(),
                              pip_fn=FakePip(), reexec_fn=FakeReexec(), environ=env,
                              state_path=str(tmp_path / "s.json"))
    assert sync.manifest is None and env == {} and sync.config_applied == []


def test_sync_applies_config_end_to_end(tmp_path):
    repo = make_repo(tmp_path, version="0.1.0")
    mirrors, fetch = mirrors_serving(sign("0.1.0", config={"merge_url": "http://from-manifest:8710"}))
    env = {}
    sync = sync_from_manifest(repo, argv=["m"], mirrors=mirrors, fetch_fn=fetch, git_fn=FakeGit(),
                              pip_fn=FakePip(), reexec_fn=FakeReexec(), environ=env,
                              state_path=str(tmp_path / "s.json"))
    assert env["NEURAHASH_DILOCO_MERGE_URL"] == "http://from-manifest:8710"
    assert sync.config_applied == ["NEURAHASH_DILOCO_MERGE_URL=http://from-manifest:8710"]


# ============================== CRITERION 4: min_client_version refuses to PUBLISH, keeps TRAINING
def test_min_client_version_above_local_refuses_publish_with_a_named_reason(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, version="0.1.0")
    mirrors, fetch = mirrors_serving(sign("0.1.0", min_client_version="0.2.0"))
    sync = sync_from_manifest(repo, argv=["m"], mirrors=mirrors, fetch_fn=fetch, git_fn=FakeGit(),
                              pip_fn=FakePip(), reexec_fn=FakeReexec(), environ={},
                              state_path=str(tmp_path / "s.json"))
    assert sync.publish_block is not None
    assert "min_client_version 0.2.0" in sync.publish_block
    assert "0.1.0" in sync.publish_block

    # even with EVERY publish prerequisite satisfied, the block wins and is reported BY NAME
    monkeypatch.setenv("NEURAHASH_DILOCO_MERGE_URL", "http://x:8710")
    monkeypatch.setenv("NEURAHASH_CONTENT_TOKEN", "tok")
    monkeypatch.setenv("PINATA_JWT", "jwt")
    monkeypatch.setattr(run_miner, "PUBLISH_BLOCK_REASON", sync.publish_block)
    is_live, reason = run_miner.publish_mode()
    assert is_live is False
    assert "min_client_version" in reason and "training continues" in reason


def test_publish_block_does_not_stop_training(tmp_path, monkeypatch):
    """The blocked client must still TRAIN -- it just runs the contribute child WITHOUT the
    --publish-delta flags. A stranger who is out of date gets slower progress, never a crash."""
    seen = {}

    def _fake_run(cmd, **kw):
        seen["cmd"] = list(cmd)
        return types.SimpleNamespace(returncode=0, stdout="held-out 1.0 -> 0.9 (accept)\n", stderr="")

    monkeypatch.setattr(run_miner, "resolve_base", lambda a, w, e: (str(tmp_path / "base.pt"), "fake"))
    monkeypatch.setattr(run_miner.subprocess, "run", _fake_run)
    monkeypatch.setattr(run_miner, "PUBLISH_BLOCK_REASON",
                        "client v0.1.0 is below ... min_client_version 0.2.0 ... training continues")
    is_live, _reason = run_miner.publish_mode()
    args = types.SimpleNamespace(base="qwen3-0.6b", steps=1, lr=1e-5, device="cpu", base_source=None)
    run_miner.run_iteration(args, str(tmp_path), "miner-x", 7, is_live)
    assert "contribute" in seen["cmd"]                       # training DID run
    assert "--publish-delta" not in seen["cmd"]              # publishing did NOT


def test_equal_or_newer_client_is_not_blocked():
    assert su.publish_block_reason({"min_client_version": "0.1.1"}, "0.1.1") is None
    assert su.publish_block_reason({"min_client_version": "0.1.1"}, "0.2.0") is None
    assert su.publish_block_reason({}, "0.0.1") is None                     # v1 manifest: no gate
    assert su.publish_block_reason({"min_client_version": "0.2.0"}, "0.1.9") is not None


def test_unknown_local_version_with_a_required_minimum_blocks_publishing():
    reason = su.publish_block_reason({"min_client_version": "0.2.0"}, None)
    assert reason is not None and "min_client_version 0.2.0" in reason


# =========================================================== startup ALWAYS checks, periodic does not
def test_startup_check_ignores_the_6h_rate_limit(tmp_path):
    repo = make_repo(tmp_path, version="0.1.0")
    state = str(tmp_path / "state.json")
    calls = {"n": 0}
    mirrors = (("m0", "https://m0.example/release.json"),)

    def _count(url, timeout=None):
        calls["n"] += 1
        return json.dumps(sign("0.1.0"))

    common = dict(mirrors=mirrors, fetch_fn=_count, git_fn=FakeGit(), pip_fn=FakePip(),
                  reexec_fn=FakeReexec(), environ={}, state_path=state)
    t0 = 1_700_000_000.0
    sync_from_manifest(repo, argv=["m"], startup=True, now=t0, **common)
    sync_from_manifest(repo, argv=["m"], startup=True, now=t0 + 60, **common)
    assert calls["n"] == 2, "a restart must ALWAYS re-check -- that is the joiner-gets-the-fix case"

    # the PERIODIC (non-startup) check still honours the limit and does not even reach git.
    res = su.check_and_update(repo, argv=["m"], pubkey=TEST_PUBKEY, honor_rate_limit=True,
                              rate_limit_s=6 * 3600, now=t0 + 120, state_path=state,
                              mirrors=mirrors, fetch_fn=_count, git_fn=FakeGit(), pip_fn=FakePip(),
                              reexec_fn=FakeReexec())
    assert res.action == "rate-limited" and calls["n"] == 2


def test_opt_out_disables_the_whole_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURAHASH_AUTOUPDATE", "0")
    repo = make_repo(tmp_path, version="0.1.0")

    def _must_not_fetch(url, timeout=None):
        raise AssertionError("fetched while opted out")

    env = {}
    sync = sync_from_manifest(repo, argv=["m"], fetch_fn=_must_not_fetch, environ=env,
                              state_path=str(tmp_path / "s.json"))
    assert sync.manifest is None and sync.publish_block is None and env == {}


def test_a_genuine_but_superseded_manifest_cannot_roll_config_back(tmp_path):
    """THE REPLAY ATTACK. The forward-only gate protects the CODE; without a floor it does NOT
    protect `config`. An attacker who answers ONE mirror while withholding the others replays a
    GENUINE OLD signed manifest whose `config` points the miner's publish endpoint (and its
    NEURAHASH_CONTENT_TOKEN) at a host of their choosing. It must be rejected."""
    repo = make_repo(tmp_path, version="0.1.1")
    state = str(tmp_path / "state.json")
    new = sign("0.1.1", ts=2_000_000_000, config={"merge_url": "http://honest.example:8710"})
    old = sign("0.1.0", ts=1_000_000_000, config={"merge_url": "http://attacker.example"})

    env = {}
    m_new, f_new = mirrors_serving(new)
    s1 = sync_from_manifest(repo, argv=["m"], mirrors=m_new, fetch_fn=f_new, git_fn=FakeGit(),
                            pip_fn=FakePip(), reexec_fn=FakeReexec(), environ=env, state_path=state)
    assert env["NEURAHASH_DILOCO_MERGE_URL"] == "http://honest.example:8710"
    assert s1.manifest_version == "0.1.1"

    # now the attacker withholds the honest mirrors and replays the OLD, genuinely-signed manifest
    env2 = {}
    m_old, f_old = mirrors_serving(old)
    s2 = sync_from_manifest(repo, argv=["m"], mirrors=m_old, fetch_fn=f_old, git_fn=FakeGit(),
                            pip_fn=FakePip(), reexec_fn=FakeReexec(), environ=env2, state_path=state)
    assert s2.manifest is None, "a superseded manifest must not be accepted"
    assert env2 == {}, "the attacker's endpoint must never reach the environment"
    assert any("replay" in s for _n, _u, s in s2.fetch.tried)


def test_a_withheld_manifest_cannot_lift_an_already_declared_publish_gate(tmp_path):
    """Withholding every mirror is fail-OPEN for training but must not silently RE-ENABLE
    publishing that a signed manifest already forbade."""
    repo = make_repo(tmp_path, version="0.1.0")
    state = str(tmp_path / "state.json")
    m, f = mirrors_serving(sign("0.1.0", ts=2_000_000_000, min_client_version="0.2.0"))
    s1 = sync_from_manifest(repo, argv=["m"], mirrors=m, fetch_fn=f, git_fn=FakeGit(),
                            pip_fn=FakePip(), reexec_fn=FakeReexec(), environ={}, state_path=state)
    assert s1.publish_block is not None

    dead, fdead = mirrors_serving(OSError("dns"), OSError("dns"))
    s2 = sync_from_manifest(repo, argv=["m"], mirrors=dead, fetch_fn=fdead, git_fn=FakeGit(),
                            pip_fn=FakePip(), reexec_fn=FakeReexec(), environ={}, state_path=state)
    assert s2.manifest is None
    assert s2.publish_block is not None and "min_client_version 0.2.0" in s2.publish_block


def test_a_signed_manifest_with_a_non_numeric_version_is_rejected_outright():
    """Otherwise ONE such mirror poisons the loop and which mirror is first decides whether the
    fleet's update channel survives."""
    m = sign("0.2.0")
    m["version"] = "latest"
    ok, reason = su.verify_manifest(m, pubkey=TEST_PUBKEY)
    assert ok is False and "version malformed" in reason
    # and it cannot suppress a good mirror regardless of ORDER
    for order in ((m, sign("0.3.0")), (sign("0.3.0"), m)):
        mirrors, fetch = mirrors_serving(*order)
        res = fetch_best_manifest(mirrors, fetch_fn=fetch)
        assert res.manifest["version"] == "0.3.0"


def test_fetch_is_bounded_in_bytes_and_wall_clock():
    """A dribbling or endless mirror must not stall startup or eat memory. Verified against a real
    local socket -- no external network."""
    import http.server, socket, threading, time as _t

    class Dribble(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            for _ in range(100000):                 # one byte every 0.2s, forever
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except Exception:
                    return
                _t.sleep(0.2)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Dribble)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/release.json"
    try:
        object.__setattr__(su, "_ALLOWED_HTTP_URLS", frozenset({url}))   # allow the test url only
        t0 = _t.monotonic()
        with pytest.raises(Exception):
            su._default_fetch(url, timeout=2)
        elapsed = _t.monotonic() - t0
        assert elapsed < 8, f"fetch ran {elapsed:.1f}s against a 2s budget -- not actually bounded"
    finally:
        su._ALLOWED_HTTP_URLS = frozenset({su.VPS_MANIFEST_URL})
        srv.shutdown()
    assert su.MAX_MANIFEST_BYTES <= 1 << 20


def test_a_supplied_manifest_is_re_verified_before_any_checkout(tmp_path):
    """check_and_update is public: even a manifest handed to it directly must recover the pinned
    key before a single git command runs. Nothing reaches `git checkout` unverified, ever."""
    repo = make_repo(tmp_path, version="0.1.0")
    git, rex = FakeGit(), FakeReexec()
    res = su.check_and_update(repo, argv=["m"], pubkey=TEST_PUBKEY, honor_rate_limit=False,
                              state_path=str(tmp_path / "s.json"),
                              manifest=sign("0.9.0", priv=WRONG_PRIV),
                              git_fn=git, pip_fn=FakePip(), reexec_fn=rex)
    assert res.action == "verify-failed"
    assert git.calls == [] and rex.called is False


def test_http_is_refused_for_any_url_not_on_the_compiled_mirror_list():
    with pytest.raises(ValueError):
        su._default_fetch("http://attacker.example/release.json")
    with pytest.raises(ValueError):
        su._default_fetch("ftp://attacker.example/release.json")
    assert su.VPS_MANIFEST_URL in su._ALLOWED_HTTP_URLS       # the ONE compiled-in exception


# ============================================================================ --doctor preflight
def _sync_stub(manifest=None, block=None, source="https://m0.example/release.json"):
    fetch = su.ManifestFetch(manifest, source, [("m0", source, "valid" if manifest else "unreachable")])
    return su.SyncResult(fetch=fetch, manifest=manifest, publish_block=block, local_version="0.1.1")


def test_doctor_passes_when_everything_is_configured(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NEURAHASH_DILOCO_MERGE_URL", "http://store.example:8710")
    monkeypatch.setenv("NEURAHASH_CONTENT_TOKEN", "tok")
    monkeypatch.setenv("PINATA_JWT", "jwt")
    monkeypatch.setenv("NEURAHASH_MINER_KEY", str(tmp_path / "miner_key.hex"))
    (tmp_path / "miner_key.hex").write_text("0x" + "33" * 32, encoding="utf-8")
    code, checks = run_miner.doctor(str(tmp_path), device="cpu",
                                    sync=_sync_stub(sign("0.1.1")),
                                    head_fn=lambda u: (True, "HTTP 200"))
    out = capsys.readouterr().out
    assert code == 0, [c for c in checks if not c["ok"]]
    assert out.count("PASS") == len(checks) and "FAIL" not in out


def test_doctor_exits_nonzero_and_names_a_remedy_for_each_failure(tmp_path, monkeypatch, capsys):
    # pin the two host-dependent probes so this asserts the DOCTOR, not the dev machine's kubo.
    monkeypatch.setattr(run_miner, "_kubo_available", lambda: False)
    monkeypatch.setattr(run_miner, "_pinata_configured", lambda: False)
    code, checks = run_miner.doctor(str(tmp_path), device="cpu",
                                    sync=_sync_stub(None, block="client v0.1.0 is below ... 0.2.0"),
                                    head_fn=lambda u: (False, "TimeoutError"))
    out = capsys.readouterr().out
    assert code == 1
    failed = [c for c in checks if not c["ok"]]
    assert {"client version", "signed manifest", "registry reachable",
            "publish auth", "pinning backend"} <= {c["name"] for c in failed}
    assert all(c["remedy"] for c in failed)                  # every failure names a remedy
    assert "run again to auto-update" in out and "install kubo, or set PINATA_JWT" in out


def test_doctor_never_hangs_on_a_dead_registry(tmp_path):
    """The network probe is bounded and its failure is a FAIL line, never an exception."""
    ok, detail = run_miner._doctor_registry_reachable("http://127.0.0.1:1/nope", timeout=1)
    assert ok is False and isinstance(detail, str)
