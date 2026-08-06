"""The release signer must refuse to sign a manifest whose --version disagrees with the VERSION
file AT THE SIGNED COMMIT.

WHY THIS EXISTS (near miss, 2026-07-25). A signed release manifest declared v3.4.0 while pointing
at a commit whose VERSION file said 3.3.2, and `tools/sign_release.py` printed

    pinned match : YES -- clients pinning the current key will accept this manifest

because the SIGNATURE was valid. It was valid over the WRONG TREE. A miner taking that manifest
checks out the commit, reads 3.3.2, is still offered 3.4.0, and re-execs -- fleet-wide, with a good
signature on it. The signer resolved --commit to a full 40-hex hash and fail-fasted on a malformed
--version, but never once looked at what the commit actually CONTAINED.

Every test here builds a REAL throwaway git repo in tmp_path and points the signer at it via its
module-level REPO. Nothing touches the real release key, the real release.json, or the real repo.

The load-bearing test is `test_mismatched_version_refuses`: it is the POSITIVE CONTROL that proves
the gate fires at all. A file that only exercised the happy path would read as coverage while
checking nothing -- the failure mode this repo was burned by in
tests/test_published_tree_imports_resolve.py (see its
`test_lazy_imports_are_actually_being_checked`). If the gate is deleted or neutered, that test MUST
go red.

Run: C:/Python313/python.exe -m pytest tests/test_sign_release_version_gate.py -q
"""
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools import sign_release                                   # noqa: E402
from tools.self_update import canonical_manifest_bytes           # noqa: E402
from neura_l1.signing import account_from_key, recover_bytes     # noqa: E402

# A throwaway key that exists only in this test file. NOT the release key: the real one lives
# offline and never enters a test process.
THROWAWAY_KEY = "0x" + "11" * 32

# Committer identity + gpgsign off so the fixture repo builds on any machine regardless of the
# operator's global git config.
_GIT_ID = ["-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
           "-c", "commit.gpgsign=false"]


def _git(repo, *args, autocrlf=None):
    cmd = ["git", "-C", str(repo)] + list(_GIT_ID)
    if autocrlf is not None:
        cmd += ["-c", "core.autocrlf=%s" % autocrlf]
    cmd += [str(a) for a in args]
    out = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    assert out.returncode == 0, "git %s failed: %s" % (args, out.stderr.strip())
    return out.stdout


def _fixture_repo(tmp_path, name, version_bytes):
    """Build a one-commit git repo. version_bytes=None -> the commit has NO VERSION file."""
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README.md").write_text("fixture repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    if version_bytes is not None:
        (repo / "VERSION").write_bytes(version_bytes)
        # autocrlf=false so the bytes we wrote are the bytes that land in the blob -- otherwise a
        # machine with core.autocrlf=true (this one) would silently normalize the CRLF fixture and
        # test_crlf_committed_version_still_matches would prove nothing.
        _git(repo, "add", "VERSION", autocrlf="false")
    _git(repo, "commit", "-q", "-m", "fixture commit")
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert len(head) == 40, head
    return repo, head


def _blob_bytes(repo, rev_path):
    return subprocess.check_output(["git", "-C", str(repo), "show", rev_path])


def _keyfile(tmp_path):
    p = tmp_path / "throwaway_key.hex"
    p.write_text(THROWAWAY_KEY + "\n", encoding="utf-8")
    return p


def _sign(tmp_path, repo, monkeypatch, version, commit, out_name="release.json", extra=()):
    """Run the signer against the FIXTURE repo (never the real one) and return (rc, out_path)."""
    monkeypatch.setattr(sign_release, "REPO", str(repo))
    out = tmp_path / out_name
    argv = ["--version", version, "--commit", commit,
            "--key", str(_keyfile(tmp_path)), "--out", str(out),
            "--published-ts", "1750000000"] + list(extra)
    return sign_release.main(argv), out


# ----------------------------------------------------------------------------- happy path

def test_matching_version_signs(tmp_path, monkeypatch):
    repo, head = _fixture_repo(tmp_path, "match", b"3.4.0\n")
    rc, out = _sign(tmp_path, repo, monkeypatch, "3.4.0", head)
    assert rc == 0
    m = json.loads(out.read_text(encoding="utf-8"))
    assert m["version"] == "3.4.0"
    assert m["git_commit"] == head
    # and it is a REAL signature over the canonical bytes, not just a file that got written
    body = {k: m[k] for k in ("version", "git_commit", "published_ts")}
    acct = account_from_key(THROWAWAY_KEY)
    assert recover_bytes(canonical_manifest_bytes(body), m["signature"]).lower() == \
        acct.address.lower()


# --------------------------------------------------- POSITIVE CONTROL: the gate must actually fire

def test_mismatched_version_refuses(tmp_path, monkeypatch):
    """The exact 2026-07-25 near miss: manifest says 3.4.0, the commit's VERSION says 3.3.2.

    POSITIVE CONTROL for this whole file. Neuter _assert_version_matches_commit (early `return`)
    and this test goes red -- verified by hand before landing.
    """
    repo, head = _fixture_repo(tmp_path, "mismatch", b"3.3.2\n")
    with pytest.raises(SystemExit) as ei:
        _sign(tmp_path, repo, monkeypatch, "3.4.0", head)
    msg = str(ei.value)
    assert "3.4.0" in msg, msg          # the version being signed
    assert "3.3.2" in msg, msg          # the version actually at the commit
    assert head in msg, msg             # which commit
    assert "refus" in msg.lower(), msg
    assert not (tmp_path / "release.json").exists(), "refused, but a manifest was still written"


def test_mismatch_refuses_before_the_private_key_is_read(tmp_path, monkeypatch):
    """A bad release must die without the operator's key entering the process."""
    repo, head = _fixture_repo(tmp_path, "nokey", b"3.3.2\n")
    monkeypatch.setattr(sign_release, "REPO", str(repo))
    monkeypatch.setattr(sign_release, "_load_privkey",
                        lambda *a, **k: pytest.fail("the private key was read on a REFUSED release"))
    with pytest.raises(SystemExit):
        sign_release.main(["--version", "3.4.0", "--commit", head,
                           "--key", str(_keyfile(tmp_path)),
                           "--out", str(tmp_path / "release.json")])


# ------------------------------------------------------------------- VERSION absent at that commit

def test_version_absent_at_commit_refuses(tmp_path, monkeypatch):
    """Absent is exactly the 3.4.0 shape -- it must FAIL, not silently pass."""
    repo, head = _fixture_repo(tmp_path, "absent", None)
    assert sign_release._version_at_commit(head, str(repo)) is None
    with pytest.raises(SystemExit) as ei:
        _sign(tmp_path, repo, monkeypatch, "3.4.0", head)
    msg = str(ei.value)
    assert "VERSION" in msg, msg
    assert "predates" in msg.lower(), msg
    assert head in msg, msg
    assert not (tmp_path / "release.json").exists()


# ------------------------------------------------------------------------------- CRLF drift

def test_crlf_committed_version_still_matches(tmp_path, monkeypatch):
    """A VERSION blob committed with CRLF must still match -- proves the comparison strips."""
    repo, head = _fixture_repo(tmp_path, "crlf", b"3.4.0\r\n")
    # fixture-first: if the blob is not actually CRLF, this test would pass for the wrong reason
    assert b"\r\n" in _blob_bytes(repo, head + ":VERSION"), "fixture did not commit CRLF"
    assert sign_release._version_at_commit(head, str(repo)) == "3.4.0"
    rc, out = _sign(tmp_path, repo, monkeypatch, "3.4.0", head)
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["version"] == "3.4.0"


def test_dirty_working_tree_cannot_satisfy_the_gate(tmp_path, monkeypatch):
    """The gate reads the COMMITTED blob. Bumping VERSION without committing must not pass."""
    repo, head = _fixture_repo(tmp_path, "dirty", b"3.3.2\n")
    (repo / "VERSION").write_bytes(b"3.4.0\n")           # bumped but NOT committed
    with pytest.raises(SystemExit) as ei:
        _sign(tmp_path, repo, monkeypatch, "3.4.0", head)
    assert "3.3.2" in str(ei.value)


# ------------------------------------------------------------------ no escape hatch, by construction

@pytest.mark.parametrize("flag", ["--force", "--skip-version-check", "--no-version-check"])
def test_no_bypass_flag_exists(tmp_path, monkeypatch, flag):
    repo, head = _fixture_repo(tmp_path, "flag" + flag.strip("-"), b"3.3.2\n")
    with pytest.raises(SystemExit) as ei:
        _sign(tmp_path, repo, monkeypatch, "3.4.0", head, extra=[flag])
    assert ei.value.code == 2, "argparse accepted %s -- a bypass flag was added" % flag


def test_no_env_override(tmp_path, monkeypatch):
    repo, head = _fixture_repo(tmp_path, "env", b"3.3.2\n")
    for var in ("NEURAHASH_SKIP_VERSION_CHECK", "SKIP_VERSION_CHECK", "NEURAHASH_FORCE_SIGN",
                "NEURAHASH_ALLOW_VERSION_MISMATCH"):
        monkeypatch.setenv(var, "1")
    with pytest.raises(SystemExit) as ei:
        _sign(tmp_path, repo, monkeypatch, "3.4.0", head)
    assert "3.3.2" in str(ei.value), "an environment variable disabled the release gate"


# --------------------------------------------------------------------------------- .gitattributes

def test_gitattributes_pins_version_to_lf():
    """CRLF drift between machines must not be able to reach the comparison in the first place."""
    text = open(os.path.join(_REPO, ".gitattributes"), "r", encoding="utf-8").read()
    assert any(ln.split("#")[0].strip() == "VERSION text eol=lf" for ln in text.splitlines()), \
        ".gitattributes has no `VERSION text eol=lf` rule"
