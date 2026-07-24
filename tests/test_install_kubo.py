"""Fail-closed gate for the SAFE kubo auto-installer (tools/install_kubo.py).

Auto-downloading a BINARY and then EXECUTING it, on machines that also hold miners' wallet keys, is
only acceptable if the client refuses every archive it cannot prove is the pinned one. This suite
pins exactly that policy with NO real network and NO real downloads -- the fetcher and the version
probe are injected fakes, exactly as tests/test_self_update.py injects fetch_fn/git_fn/pip_fn:

  * a correct archive installs, lands ONLY under <work_dir>/kubo/, and returns the binary path;
  * a ONE-BYTE-ALTERED archive (same size) is REJECTED and nothing whatsoever is installed;
  * an unsupported platform, and a network error, each return a clean failure without raising;
  * a zip-slip / tar-slip member fails the WHOLE install and writes nothing outside the work dir;
  * NEURAHASH_AUTO_INSTALL_KUBO=0 skips the install entirely -- the fetcher is never even called;
  * the compiled-in digest table has the right SHAPE (128 hex chars, official https URL), which is
    the cheap guard against a typo'd or truncated constant.

The pinned digests themselves are the PUBLISHED ones from https://dist.ipfs.tech/kubo/v0.42.0/
<archive>.sha512; this suite deliberately does NOT re-download them (that would make the test a
network test and would prove nothing the constants do not already assert).
"""
import io
import hashlib
import os
import tarfile
import zipfile

import pytest

from tools import install_kubo as ik

TEST_KEY = ("testos", "amd64")
PAYLOAD = b"MZ-fake-ipfs-binary-" + bytes(4096)      # stands in for the ~40-90MB real binary


def _zip_bytes(member="kubo/ipfs.exe", payload=PAYLOAD, extra=()):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, payload)
        for name, data in extra:
            zf.writestr(name, data)
    return buf.getvalue()


def _tar_bytes(member="kubo/ipfs", payload=PAYLOAD, extra=(), symlink=None):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo(member)
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))
        for name, data in extra:
            t2 = tarfile.TarInfo(name)
            t2.size = len(data)
            tf.addfile(t2, io.BytesIO(data))
        if symlink:
            t3 = tarfile.TarInfo(symlink)
            t3.type = tarfile.SYMTYPE
            t3.linkname = "/etc/passwd"
            tf.addfile(t3)
    return buf.getvalue()


def _release(data, archive="kubo_test.zip", member="kubo/ipfs.exe", binary="ipfs.exe"):
    """A Release pinned to EXACTLY these bytes -- the same shape as the real compiled-in entries."""
    return ik.Release(archive, hashlib.sha512(data).hexdigest(), len(data), member, binary)


def _fake_which(cmd, *_a, **_k):
    """shutil.which with an EMPTY PATH: a bare name never resolves, but an explicit path to an
    existing file does -- exactly how the real which() behaves for NEURAHASH_IPFS_BIN=<abs path>."""
    cmd = str(cmd)
    has_dir = ("/" in cmd) or ("\\" in cmd)
    return cmd if has_dir and os.path.isfile(cmd) else None


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A hermetic install environment: no PATH kubo, no env opt-out, a log sink, and a work dir."""
    monkeypatch.setattr(ik.shutil, "which", _fake_which)
    monkeypatch.delenv(ik.AUTO_INSTALL_ENV, raising=False)
    monkeypatch.setenv(ik.IPFS_BIN_ENV, "")          # recorded, so teardown restores the real value
    lines = []
    return {"work_dir": str(tmp_path / "wd"), "lines": lines, "log": lines.append}


def _ok_probe(_binary):
    return True, "ipfs version 0.42.0"


def _install(wired, data, release, **kw):
    calls = []

    def fetch(url, timeout=None):
        calls.append(url)
        return data

    kw.setdefault("fetch_fn", fetch)
    kw.setdefault("probe_fn", _ok_probe)
    path, reason = ik.ensure_kubo(wired["work_dir"], key=TEST_KEY,
                                  log_fn=wired["log"], min_binary_bytes=16, **kw)
    return path, reason, calls


# ------------------------------------------------------- CRITERION 1: a correct archive installs
@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_correct_archive_installs_and_returns_binary_path(wired, monkeypatch, kind):
    if kind == "zip":
        data = _zip_bytes()
        rel = _release(data)
    else:
        data = _tar_bytes()
        rel = _release(data, archive="kubo_test.tar.gz", member="kubo/ipfs", binary="ipfs")
    monkeypatch.setitem(ik.RELEASES, TEST_KEY, rel)

    path, reason, calls = _install(wired, data, rel)

    assert path == os.path.join(ik.install_dir(wired["work_dir"]), rel.binary)
    assert os.path.isfile(path)
    with open(path, "rb") as f:
        assert f.read() == PAYLOAD                       # the real member, byte-for-byte
    assert "auto-installed" in reason
    assert len(calls) == 1 and calls[0] == rel.url
    # local install only: <work_dir>/kubo/<binary> and nothing else survives (temp dir cleaned up)
    assert sorted(os.listdir(ik.install_dir(wired["work_dir"]))) == [rel.binary]
    assert os.listdir(wired["work_dir"]) == ["kubo"]
    # the publish child (tools/ipfs_checkpoint.py) reads this at import time -- no PATH change
    assert os.environ[ik.IPFS_BIN_ENV] == path
    # a second call must NOT download again
    _p2, reason2, calls2 = _install(wired, data, rel)
    assert calls2 == [] and "already available" in reason2


# ------------------------------------------- CRITERION 2: one altered byte => install NOTHING
def test_one_byte_altered_archive_is_rejected_and_installs_nothing(wired, monkeypatch):
    good = _zip_bytes()
    rel = _release(good)                                 # digest pinned to the GOOD bytes
    monkeypatch.setitem(ik.RELEASES, TEST_KEY, rel)
    i = len(good) // 2
    tampered = good[:i] + bytes([good[i] ^ 0x01]) + good[i + 1:]
    assert len(tampered) == len(good) and tampered != good   # same size: only the digest can catch it

    path, reason, calls = _install(wired, tampered, rel)

    assert path is None
    assert "sha512 MISMATCH" in reason
    assert calls == [rel.url]
    assert not os.path.exists(ik.install_dir(wired["work_dir"]))
    assert not os.path.exists(wired["work_dir"]) or os.listdir(wired["work_dir"]) == []
    assert any("REFUSED" in ln for ln in wired["lines"])


def test_size_mismatch_is_rejected_before_hashing(wired, monkeypatch):
    good = _zip_bytes()
    rel = _release(good)
    monkeypatch.setitem(ik.RELEASES, TEST_KEY, rel)
    path, reason, _ = _install(wired, good + b"x", rel)
    assert path is None and "size mismatch" in reason
    assert not os.path.exists(ik.install_dir(wired["work_dir"]))


# ----------------------------------------------- CRITERION 3: unsupported platform, cleanly
def test_unsupported_platform_returns_clean_failure(wired, monkeypatch):
    monkeypatch.setattr(ik, "platform_key", lambda *_a, **_k: None)

    def fetch(_url, timeout=None):
        raise AssertionError("must not download on an unsupported platform")

    path, reason = ik.ensure_kubo(wired["work_dir"], fetch_fn=fetch, probe_fn=_ok_probe,
                                  log_fn=wired["log"])
    assert path is None
    assert "unsupported platform" in reason
    assert not os.path.exists(ik.install_dir(wired["work_dir"]))


def test_platform_key_maps_only_pinned_targets():
    assert ik.platform_key("Windows", "AMD64") == ("windows", "amd64")
    assert ik.platform_key("Linux", "x86_64") == ("linux", "amd64")
    assert ik.platform_key("Darwin", "arm64") == ("darwin", "arm64")
    assert ik.platform_key("Linux", "aarch64") == ("linux", "arm64")
    assert ik.platform_key("Linux", "i386") is None          # unpinned arch: never guess a build
    assert ik.platform_key("Linux", "riscv64") is None
    assert ik.platform_key("Freebsd", "x86_64") is None       # unpinned OS


# --------------------------------------------------- CRITERION 4: a network error never raises
def test_network_error_returns_clean_failure_without_raising(wired, monkeypatch):
    monkeypatch.setitem(ik.RELEASES, TEST_KEY, _release(_zip_bytes()))

    def boom(_url, timeout=None):
        raise OSError("simulated: kubo dist unreachable")

    path, reason = ik.ensure_kubo(wired["work_dir"], key=TEST_KEY, fetch_fn=boom,
                                  probe_fn=_ok_probe, log_fn=wired["log"])
    assert path is None
    assert "OSError" in reason and "unreachable" in reason
    assert not os.path.exists(ik.install_dir(wired["work_dir"]))
    assert any("staying in LOCAL mode" in ln for ln in wired["lines"])


def test_binary_that_will_not_run_is_removed(wired, monkeypatch):
    data = _zip_bytes()
    rel = _release(data)
    monkeypatch.setitem(ik.RELEASES, TEST_KEY, rel)
    path, reason, _ = _install(wired, data, rel, probe_fn=lambda _b: (False, "rc=1"))
    assert path is None and "failed its version probe" in reason
    assert not os.path.isfile(os.path.join(ik.install_dir(wired["work_dir"]), rel.binary))


# ------------------------------------------------------------- CRITERION 5: zip-slip refused
@pytest.mark.parametrize("evil", ["../../evil.txt", "/abs/evil.txt", "kubo/../../evil.txt"])
def test_zip_slip_member_refuses_the_whole_install(wired, monkeypatch, tmp_path, evil):
    data = _zip_bytes(extra=[(evil, b"pwned")])
    rel = _release(data)
    monkeypatch.setitem(ik.RELEASES, TEST_KEY, rel)

    path, reason, _ = _install(wired, data, rel)

    assert path is None
    assert "escapes the destination" in reason
    # neither the escaping file nor the legitimate member was written
    assert not (tmp_path / "evil.txt").exists()
    assert not os.path.isfile(os.path.join(ik.install_dir(wired["work_dir"]), rel.binary))


def test_tar_slip_and_symlink_members_are_refused(wired, monkeypatch):
    slip = _tar_bytes(extra=[("../../evil.txt", b"pwned")])
    rel = _release(slip, archive="k.tar.gz", member="kubo/ipfs", binary="ipfs")
    monkeypatch.setitem(ik.RELEASES, TEST_KEY, rel)
    path, reason, _ = _install(wired, slip, rel)
    assert path is None and "escapes the destination" in reason

    link = _tar_bytes(symlink="kubo/passwd")
    rel2 = _release(link, archive="k.tar.gz", member="kubo/ipfs", binary="ipfs")
    monkeypatch.setitem(ik.RELEASES, TEST_KEY, rel2)
    path2, reason2, _ = _install(wired, link, rel2)
    assert path2 is None and "non-regular member" in reason2


def test_escapes_helper(tmp_path):
    dest = str(tmp_path / "kubo")
    assert ik._escapes(dest, "../x") is True
    assert ik._escapes(dest, "..\\x") is True
    assert ik._escapes(dest, "/etc/passwd") is True
    assert ik._escapes(dest, "C:/windows/x") is True
    assert ik._escapes(dest, "kubo/ipfs") is False
    assert ik._escapes(dest, "a/b/../c") is False


# ------------------------------------------------------------------- CRITERION 6: the opt-out
def test_opt_out_env_skips_the_install_entirely(wired, monkeypatch):
    monkeypatch.setenv(ik.AUTO_INSTALL_ENV, "0")
    monkeypatch.setitem(ik.RELEASES, TEST_KEY, _release(_zip_bytes()))

    def fetch(_url, timeout=None):
        raise AssertionError("opt-out must not touch the network")

    path, reason = ik.ensure_kubo(wired["work_dir"], key=TEST_KEY, fetch_fn=fetch,
                                  probe_fn=_ok_probe, log_fn=wired["log"])
    assert path is None
    assert reason == "auto-install disabled (%s=0)" % ik.AUTO_INSTALL_ENV
    assert not os.path.exists(wired["work_dir"])
    assert wired["lines"] == []


@pytest.mark.parametrize("val,on", [("0", False), ("false", False), ("no", False), ("off", False),
                                    ("", False), ("1", True), ("yes", True)])
def test_env_enabled_matches_the_autoupdate_knob(val, on):
    assert ik.env_enabled({ik.AUTO_INSTALL_ENV: val}) is on
    assert ik.env_enabled({}) is True                   # default ON


# -------------------------------------------------------- the compiled-in table cannot be junk
def test_pinned_release_table_shape():
    assert ik.KUBO_VERSION == "v0.42.0"
    assert ("windows", "amd64") in ik.RELEASES and ("linux", "amd64") in ik.RELEASES
    assert ("darwin", "arm64") in ik.RELEASES
    for key, rel in ik.RELEASES.items():
        assert rel.url.startswith("https://dist.ipfs.tech/kubo/v0.42.0/"), key
        assert rel.url.endswith(rel.archive) and ik.KUBO_VERSION in rel.archive, key
        assert len(rel.sha512) == 128, key                       # a truncated constant is a bug
        assert all(c in "0123456789abcdef" for c in rel.sha512), key
        assert 10_000_000 < rel.size < ik.MAX_ARCHIVE_BYTES, key
        assert rel.member in ("kubo/ipfs", "kubo/ipfs.exe"), key
        assert rel.member.endswith(rel.binary), key
    # every pinned digest is distinct: no copy-paste of one platform's hash onto another
    assert len({r.sha512 for r in ik.RELEASES.values()}) == len(ik.RELEASES)


def test_default_fetch_refuses_non_https():
    with pytest.raises(ValueError):
        ik._default_fetch("http://dist.ipfs.tech/kubo/v0.42.0/kubo.zip")


# The run_miner --doctor wiring tests were removed with the deprecated Qwen turnkey lane
# (2026-07-24); install_kubo itself remains covered by every test above.
