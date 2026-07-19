"""Fail-closed gate for the SIGNED miner auto-updater (tools/self_update.py).

Auto-running pushed code on strangers' machines is only safe if the client NEVER runs code that is
not signed by the pinned release key and never downgrades. This suite pins exactly that policy with
NO real network, git, pip, or process replacement -- every side effect is an injected fake:

  * signature verify ACCEPTS a correctly-signed manifest and REJECTS a tampered field, a wrong-key
    signature, a missing signature, and a downgrade (version <= current);
  * the updater is a NO-OP unless the manifest verifies AND is a strict forward move, and it attempts
    a `git checkout` ONLY in that case;
  * on an unreachable manifest OR a bad signature it logs a warning and returns with the working tree
    (VERSION file) BYTE-UNCHANGED and no checkout attempted;
  * NEURAHASH_AUTOUPDATE=0 (or enabled=False) makes it do nothing (no fetch, no git).

The pinned release key in tools/self_update.py is a TEST key derived from the obviously-fake private
key 0x1111...11; this suite is the only place that private key is used.
"""
import json

import functools

from neura_l1.signing import account_from_key, sign_bytes
from tools import self_update as su
from tools.self_update import (
    canonical_manifest_bytes, is_forward, parse_version,
)

# These tests verify against a TEST key, INDEPENDENT of the production PINNED_RELEASE_PUBKEY (which is
# the operator's real release address). We wrap the two verifying entrypoints so every call site below
# trusts TEST_PUBKEY without passing it explicitly. This keeps the suite green after the real key is
# pinned, while still exercising the real verify/checkout/fail-closed logic against a known keypair.
TEST_PRIV = "0x" + "11" * 32
TEST_PUBKEY = account_from_key(TEST_PRIV).address
verify_manifest = functools.partial(su.verify_manifest, pubkey=TEST_PUBKEY)
check_and_update = functools.partial(su.check_and_update, pubkey=TEST_PUBKEY)
WRONG_PRIV = "0x" + "22" * 32
FORWARD_COMMIT = "b" * 40           # a bare 40-hex commit id the fake git will "check out"


def _sign(version, commit=FORWARD_COMMIT, ts=1721001600, priv=TEST_PRIV):
    """Produce a signed manifest exactly the way tools/sign_release.py does."""
    acct = account_from_key(priv)
    body = {"version": version, "git_commit": commit, "published_ts": ts}
    m = dict(body)
    m["signature"] = sign_bytes(acct, canonical_manifest_bytes(body))
    m["signer"] = acct.address
    return m


def _make_repo(tmp_path, version="0.1.0", reqs="numpy\ntorch\n"):
    (tmp_path / "VERSION").write_text(version + "\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(reqs, encoding="utf-8")
    return str(tmp_path)


class FakeGit:
    """Records git calls; simulates a successful fetch/checkout without touching a real repo.
    `head_override` forces `rev-parse HEAD` to lie (to test the post-checkout HEAD-mismatch guard).
    `on_checkout(repo, commit)` lets a test mutate the tree (e.g. change requirements.txt)."""

    def __init__(self, head_override=None, fail_on=None, on_checkout=None):
        self.calls = []
        self.head = None
        self.head_override = head_override
        self.fail_on = set(fail_on or ())
        self.on_checkout = on_checkout

    def __call__(self, repo, *args, **kw):
        self.calls.append(args)
        sub = args[0]
        if sub in self.fail_on:
            return 1, f"fake failure: {sub}"
        if sub == "checkout":
            target = args[-1]
            if target != "-":
                self.head = target
                if self.on_checkout:
                    self.on_checkout(repo, target)
            return 0, ""
        if sub == "rev-parse":
            h = self.head_override if self.head_override is not None else (self.head or "")
            return 0, h + "\n"
        return 0, ""          # fetch, etc.

    def checkouts(self):
        return [a[-1] for a in self.calls if a and a[0] == "checkout" and a[-1] != "-"]


class FakeReexec:
    def __init__(self):
        self.called = False
        self.argv = None

    def __call__(self, argv):
        self.called = True
        self.argv = list(argv)


class FakePip:
    def __init__(self):
        self.calls = 0

    def __call__(self, repo, **kw):
        self.calls += 1
        return 0, ""


def _boom(url):
    raise OSError("simulated: manifest URL unreachable")


# --------------------------------------------------------------------------- CRITERION 1: verify
def test_verify_accepts_correctly_signed():
    ok, who = verify_manifest(_sign("0.2.0"))
    assert ok is True
    assert who.lower() == TEST_PUBKEY.lower()


def test_verify_rejects_tampered_version():
    m = _sign("0.2.0")
    m["version"] = "9.9.9"                 # signature was over 0.2.0 -> recovers to a different addr
    ok, reason = verify_manifest(m)
    assert ok is False and "pinned release key" in reason


def test_verify_rejects_tampered_commit():
    m = _sign("0.2.0")
    m["git_commit"] = "c" * 40
    ok, reason = verify_manifest(m)
    assert ok is False and "pinned release key" in reason


def test_verify_rejects_wrong_key():
    ok, reason = verify_manifest(_sign("0.2.0", priv=WRONG_PRIV))
    assert ok is False and "pinned release key" in reason


def test_verify_rejects_missing_signature():
    m = _sign("0.2.0")
    del m["signature"]
    assert verify_manifest(m)[0] is False
    m2 = _sign("0.2.0")
    m2["signature"] = ""
    assert verify_manifest(m2)[0] is False


def test_verify_rejects_downgrade_and_equal():
    # policy-level: version <= current is not a forward move (no-downgrade / no-replay).
    assert is_forward("0.2.0", "0.1.0") is True
    assert is_forward("0.1.0", "0.1.0") is False          # equal
    assert is_forward("0.1.0", "0.2.0") is False          # downgrade
    assert parse_version("0.2.0") > parse_version("0.1.9")


# ------------------------------------------------------- CRITERION 2: no-op unless verified forward
def test_no_op_when_local_newer_than_manifest(tmp_path):
    repo = _make_repo(tmp_path, version="0.2.0")
    git, rex = FakeGit(), FakeReexec()
    res = check_and_update(repo, argv=["run_miner_client.py"], enabled=True, honor_rate_limit=False,
                           state_path=str(tmp_path / "state.json"),
                           fetch_fn=lambda u: json.dumps(_sign("0.1.0")),
                           git_fn=git, pip_fn=FakePip(), reexec_fn=rex)
    assert res.applied is False and res.action == "no-op-not-forward"
    assert git.checkouts() == []                          # never touched the tree
    assert rex.called is False


def test_checkout_only_for_verified_forward(tmp_path):
    repo = _make_repo(tmp_path, version="0.1.0")
    git, rex = FakeGit(), FakeReexec()
    res = check_and_update(repo, argv=["run_miner_client.py", "--host", "x"], enabled=True,
                           honor_rate_limit=False, state_path=str(tmp_path / "state.json"),
                           fetch_fn=lambda u: json.dumps(_sign("0.2.0", commit=FORWARD_COMMIT)),
                           git_fn=git, pip_fn=FakePip(), reexec_fn=rex)
    assert res.applied is True and res.action == "applied"
    assert git.checkouts() == [FORWARD_COMMIT]            # checked out EXACTLY the signed commit
    assert rex.called is True and rex.argv[0] == "run_miner_client.py"


def test_unverified_forward_manifest_never_checks_out(tmp_path):
    repo = _make_repo(tmp_path, version="0.1.0")
    git, rex = FakeGit(), FakeReexec()
    res = check_and_update(repo, argv=["run_miner_client.py"], enabled=True, honor_rate_limit=False,
                           state_path=str(tmp_path / "state.json"),
                           fetch_fn=lambda u: json.dumps(_sign("0.2.0", priv=WRONG_PRIV)),
                           git_fn=git, pip_fn=FakePip(), reexec_fn=rex)
    assert res.applied is False and res.action == "verify-failed"
    assert git.checkouts() == [] and rex.called is False


# ---------------------------------------------------- CRITERION 3: fail-closed, tree left unchanged
def test_fail_closed_on_unreachable_manifest(tmp_path):
    repo = _make_repo(tmp_path, version="0.1.0")
    before = (tmp_path / "VERSION").read_bytes()
    git, rex = FakeGit(), FakeReexec()
    res = check_and_update(repo, argv=["run_miner_client.py"], enabled=True, honor_rate_limit=False,
                           state_path=str(tmp_path / "state.json"),
                           fetch_fn=_boom, git_fn=git, pip_fn=FakePip(), reexec_fn=rex)
    assert res.applied is False and res.action == "fetch-failed"
    assert (tmp_path / "VERSION").read_bytes() == before  # working tree byte-unchanged
    assert git.checkouts() == [] and rex.called is False


def test_fail_closed_on_bad_signature(tmp_path):
    repo = _make_repo(tmp_path, version="0.1.0")
    before = (tmp_path / "VERSION").read_bytes()
    m = _sign("0.2.0")
    m["signature"] = "00" * 65                            # structurally-plausible but wrong signature
    git, rex = FakeGit(), FakeReexec()
    res = check_and_update(repo, argv=["run_miner_client.py"], enabled=True, honor_rate_limit=False,
                           state_path=str(tmp_path / "state.json"),
                           fetch_fn=lambda u: json.dumps(m),
                           git_fn=git, pip_fn=FakePip(), reexec_fn=rex)
    assert res.applied is False and res.action == "verify-failed"
    assert (tmp_path / "VERSION").read_bytes() == before
    assert git.checkouts() == [] and rex.called is False


def test_fail_closed_on_post_checkout_head_mismatch(tmp_path):
    # git checkout "succeeds" but HEAD does not equal the signed commit -> refuse to re-exec.
    repo = _make_repo(tmp_path, version="0.1.0")
    git = FakeGit(head_override="d" * 40)                 # rev-parse HEAD lies
    rex = FakeReexec()
    res = check_and_update(repo, argv=["run_miner_client.py"], enabled=True, honor_rate_limit=False,
                           state_path=str(tmp_path / "state.json"),
                           fetch_fn=lambda u: json.dumps(_sign("0.2.0")),
                           git_fn=git, pip_fn=FakePip(), reexec_fn=rex)
    assert res.applied is False and res.action == "head-mismatch"
    assert rex.called is False


# --------------------------------------------------------------------- CRITERION 4: opt-out is total
def test_optout_env_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURAHASH_AUTOUPDATE", "0")
    repo = _make_repo(tmp_path, version="0.1.0")

    def _must_not_fetch(url):
        raise AssertionError("fetch attempted while opted out")

    git, rex = FakeGit(), FakeReexec()
    res = check_and_update(repo, argv=["run_miner_client.py"], enabled=None, honor_rate_limit=False,
                           state_path=str(tmp_path / "state.json"),
                           fetch_fn=_must_not_fetch, git_fn=git, pip_fn=FakePip(), reexec_fn=rex)
    assert res.applied is False and res.action == "disabled"
    assert git.calls == [] and rex.called is False


def test_optout_flag_enabled_false(tmp_path):
    repo = _make_repo(tmp_path, version="0.1.0")

    def _must_not_fetch(url):
        raise AssertionError("fetch attempted while enabled=False")

    res = check_and_update(repo, argv=["run_miner_client.py"], enabled=False, honor_rate_limit=False,
                           state_path=str(tmp_path / "state.json"),
                           fetch_fn=_must_not_fetch, git_fn=FakeGit(), pip_fn=FakePip(),
                           reexec_fn=FakeReexec())
    assert res.applied is False and res.action == "disabled"


# ----------------------------------------------------------- extra: pip runs ONLY on a reqs change
def test_pip_runs_only_when_requirements_change(tmp_path):
    # checkout that DOES change requirements.txt -> pip runs.
    repo = _make_repo(tmp_path, version="0.1.0", reqs="numpy\n")

    def _change_reqs(r, commit):
        (tmp_path / "requirements.txt").write_text("numpy\nnewdep\n", encoding="utf-8")

    pip = FakePip()
    check_and_update(repo, argv=["m"], enabled=True, honor_rate_limit=False,
                     state_path=str(tmp_path / "s1.json"),
                     fetch_fn=lambda u: json.dumps(_sign("0.2.0")),
                     git_fn=FakeGit(on_checkout=_change_reqs), pip_fn=pip, reexec_fn=FakeReexec())
    assert pip.calls == 1

    # checkout that leaves requirements.txt identical -> pip NOT run.
    (tmp_path / "b").mkdir(exist_ok=True)
    repo2 = _make_repo(tmp_path / "b", version="0.1.0", reqs="numpy\n")
    pip2 = FakePip()
    check_and_update(repo2, argv=["m"], enabled=True, honor_rate_limit=False,
                     state_path=str(tmp_path / "s2.json"),
                     fetch_fn=lambda u: json.dumps(_sign("0.3.0")),
                     git_fn=FakeGit(), pip_fn=pip2, reexec_fn=FakeReexec())
    assert pip2.calls == 0


# ------------------------------------------------------------------------- extra: rate limit throttles
def test_rate_limit_skips_second_check(tmp_path):
    repo = _make_repo(tmp_path, version="0.1.0")
    state = str(tmp_path / "state.json")
    calls = {"n": 0}

    def _count(url):
        calls["n"] += 1
        return json.dumps(_sign("0.1.0"))

    t0 = 1_700_000_000.0        # a realistic unix timestamp (now - 0 >> rate limit, so first proceeds)
    check_and_update(repo, argv=["m"], enabled=True, honor_rate_limit=True, rate_limit_s=6 * 3600,
                     state_path=state, now=t0, fetch_fn=_count, git_fn=FakeGit(),
                     pip_fn=FakePip(), reexec_fn=FakeReexec())
    res = check_and_update(repo, argv=["m"], enabled=True, honor_rate_limit=True, rate_limit_s=6 * 3600,
                           state_path=state, now=t0 + 60, fetch_fn=_count, git_fn=FakeGit(),
                           pip_fn=FakePip(), reexec_fn=FakeReexec())
    assert calls["n"] == 1 and res.action == "rate-limited"
