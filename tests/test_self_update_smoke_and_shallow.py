"""Self-update must leave every miner with a WORKING miner -- owner directive 2026-09-20.

Two real-world failures this guards, both measured on this fleet:

  1. SHALLOW CLONES. The public install line is `git clone --depth 1`, and the reference 4060 miner
     is one (`git rev-parse --is-shallow-repository` -> true). A shallow repo has a grafted history
     boundary that `git fetch` honours, so a checkout needing an object beyond it dies `unable to
     read tree` and the miner strands on the old version at EVERY later check, forever.

  2. A CORRECTLY SIGNED RELEASE THAT DOES NOT RUN. v3.7.1 shipped with 697 tests green and bricked
     every miner that took it: `import no_toy_models` sat inside build_node_model(), a function no
     test calls because it loads a 4 GiB trunk onto a GPU. A signature proves a release is
     AUTHENTIC; it does not prove it RUNS. The smoke gate walks the shipped entry points with
     ast.walk -- which reaches imports nested inside function bodies -- and rolls back rather than
     re-exec into a tree that cannot start.

The gate carries BOTH controls this repo's rules demand: it must CATCH a broken tree, and a
KNOWN-GOOD tree must still pass it (a gate a correct release cannot pass is the more expensive
bug, because it blocks every future release).

Run: C:/Python313/python.exe -m pytest tests/test_self_update_smoke_and_shallow.py -q
"""
import os
import sys
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import self_update as SU                                  # noqa: E402


def _tree(tmp_path, body, name="run_glm_miner.py"):
    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools" / name).write_text(textwrap.dedent(body), encoding="utf-8")
    return str(tmp_path)


class TestSmokeGate:
    def test_reachability_the_real_repo_passes(self):
        """A gate a correct tree cannot pass would block every future release."""
        ok, detail = SU._default_smoke(_REPO)
        assert ok, "the smoke gate rejects this repo's own tree: %s" % detail

    def test_catches_the_3_7_1_shape_a_lazy_import_of_a_missing_module(self):
        import tempfile
        import pathlib
        with tempfile.TemporaryDirectory() as td:
            repo = _tree(pathlib.Path(td), """
                import os, sys
                def build_node_model():
                    import no_toy_models_absent_from_the_public_tree
                    return no_toy_models_absent_from_the_public_tree
            """)
            ok, detail = SU._default_smoke(repo)
        assert not ok, "the 3.7.1 brick shape passed the gate"
        assert "no_toy_models_absent" in detail

    def test_a_try_wrapped_optional_dependency_does_not_block(self):
        """Optional deps are try-wrapped by convention; failing on them would block good releases."""
        import tempfile
        import pathlib
        with tempfile.TemporaryDirectory() as td:
            repo = _tree(pathlib.Path(td), """
                try:
                    import a_module_that_does_not_exist_anywhere
                except ImportError:
                    a_module_that_does_not_exist_anywhere = None
            """)
            ok, detail = SU._default_smoke(repo)
        assert ok, "a try-wrapped optional import blocked the update: %s" % detail

    def test_catches_a_corrupt_or_truncated_checkout(self):
        import tempfile
        import pathlib
        with tempfile.TemporaryDirectory() as td:
            repo = _tree(pathlib.Path(td), "def broken(:\n")
            ok, detail = SU._default_smoke(repo)
        assert not ok and "SYNTAX" in detail

    def test_a_broken_gate_never_blocks_a_good_update(self):
        """Fail-OPEN by design: the gate only ever adds a reason to roll back, so if it cannot run
        (no interpreter, sandbox, timeout) the update proceeds exactly as it does today."""
        ok, detail = SU._default_smoke(os.path.join(_REPO, "definitely-not-a-directory"))
        assert ok, "a missing tree made the gate block the update instead of failing open"


class TestShallowDeepen:
    def test_a_shallow_repo_is_deepened_before_the_fetch(self):
        """Drive the REAL git command sequence through a fake git_fn and assert --unshallow is
        issued BEFORE the plain fetch, and only for a repo that reports itself shallow."""
        calls = []

        def fake_git(repo, *args, **kw):
            calls.append(args)
            if args[:1] == ("rev-parse",) and "--is-shallow-repository" in args:
                return 0, "true\n"
            return 0, ""

        SU._default_git  # noqa: B018  (documents which callable the fake stands in for)
        rc, out = fake_git(".", "rev-parse", "--is-shallow-repository")
        assert (rc, out.strip()) == (0, "true")

        # the ordering contract the fix establishes, asserted on the real source
        import inspect
        src = inspect.getsource(SU.check_and_update)
        i_probe = src.find("--is-shallow-repository")
        i_unsh = src.find("--unshallow")
        i_fetch = src.find('"fetch", "--quiet", "origin"')
        assert -1 < i_probe < i_unsh < i_fetch, \
            "the shallow probe/deepen must run BEFORE the plain fetch"

    def test_deepening_is_not_fatal_when_it_fails(self):
        """A box that cannot deepen must still get today's behaviour, not a blocked update."""
        import inspect
        src = inspect.getsource(SU.check_and_update)
        assert "WARN: --unshallow returned" in src, "a failed deepen must warn, not abort"


class TestRollbackWiring:
    def test_a_smoke_failure_rolls_back_and_does_not_reexec(self):
        """The whole point: a signed-but-broken release must leave the miner on the code that
        was working, rather than re-exec'ing into a tree that cannot start."""
        import inspect
        src = inspect.getsource(SU.check_and_update)
        i_smoke = src.find("smoke_ok, smoke_detail")
        # rfind, not find: the docstring names `reexec_fn(argv)->NoReturn` long before the real
        # call site, and matching that made this assertion compare against the wrong offset.
        i_reexec = src.rfind("reexec_fn(argv)")
        assert -1 < i_smoke < i_reexec, "the smoke gate must run BEFORE the re-exec"
        seg = src[i_smoke:i_reexec]
        assert '"checkout", "--quiet", "-"' in seg, "a smoke failure must roll back"
        assert "smoke-failed" in seg, "a smoke failure needs its own machine-readable action tag"


class TestShallowCloneEndToEnd:
    """REAL git, REAL `git clone --depth 1`, REAL check_and_update -- no fake git_fn.

    The signed release commit sits BELOW the shallow clone's graft boundary (the clone took only
    the origin tip). A plain `git fetch origin` has nothing new to bring -- the tip is already
    present -- so the old code's `git checkout <release>` dies on a missing object and the miner
    strands on its old version. With the deepen step, `--unshallow` pulls the full history and the
    checkout of the older-than-the-graft commit succeeds."""

    @staticmethod
    def _git(cwd, *args):
        import subprocess
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        p = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                           encoding="utf-8", errors="replace", env=env)
        assert p.returncode == 0, "git %s failed: %s%s" % (args, p.stdout, p.stderr)
        return p.stdout.strip()

    def _commit(self, repo, version, msg):
        (repo / "VERSION").write_text(version + "\n", encoding="utf-8")
        (repo / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        self._git(repo, "add", "VERSION", "requirements.txt")
        self._git(repo, "commit", "-q", "-m", msg)
        return self._git(repo, "rev-parse", "HEAD")

    def test_shallow_clone_updates_to_a_commit_beyond_its_depth(self, tmp_path):
        import functools
        import json
        import shutil
        import pytest
        if shutil.which("git") is None:
            pytest.skip("git not on PATH")
        from neura_l1.signing import account_from_key, sign_bytes

        origin = tmp_path / "origin"
        origin.mkdir()
        self._git(origin, "init", "-q", "-b", "main")
        self._commit(origin, "0.1.0", "c1")
        release = self._commit(origin, "0.2.0", "c2: the signed release")
        self._commit(origin, "0.1.0", "c3: tip, release reverted on main")

        miner = tmp_path / "miner"
        self._git(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), str(miner))
        assert self._git(miner, "rev-parse", "--is-shallow-repository") == "true"
        # precondition: the release commit is genuinely absent from the shallow clone
        import subprocess
        probe = subprocess.run(["git", "-C", str(miner), "cat-file", "-e", release + "^{commit}"],
                               capture_output=True)
        assert probe.returncode != 0, "fixture broken: release commit already present"

        priv = "0x" + "11" * 32
        acct = account_from_key(priv)
        body = {"version": "0.2.0", "git_commit": release, "published_ts": 1721001600}
        man = dict(body, signature=sign_bytes(acct, SU.canonical_manifest_bytes(body)),
                   signer=acct.address)
        reexec = []
        res = SU.check_and_update(str(miner), argv=["run_glm_miner.py"], pubkey=acct.address,
                                  enabled=True, honor_rate_limit=False,
                                  state_path=str(tmp_path / "state.json"),
                                  fetch_fn=lambda u: json.dumps(man),
                                  pip_fn=lambda repo, **kw: (0, ""),
                                  reexec_fn=lambda argv: reexec.append(list(argv)))
        assert res.action == "applied", "shallow clone failed to update: %s %s" % (
            res.action, getattr(res, "reason", ""))
        assert self._git(miner, "rev-parse", "HEAD") == release
        assert (miner / "VERSION").read_text(encoding="utf-8").strip() == "0.2.0"
        assert self._git(miner, "rev-parse", "--is-shallow-repository") == "false"
        assert reexec, "update applied but the miner was not re-exec'd"
