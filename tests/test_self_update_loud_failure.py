"""Self-update failure must be LOUD, COMPLETE, and REPORTABLE ON DEMAND.

WHY THIS EXISTS. On 2026-08-12 we found our own 4060 had mined a lane with no coordinator and no
judge for 4.27 days. The obvious way to ship the fix is the signed auto-update path every miner
already runs -- except that self-update failure was effectively SILENT, so we could not have told
whether anyone received it. Three separate silences stacked up:

  1. `tools/self_update.py` reported failures with a single `log("WARN: ...")` line that scrolls
     past between two training-loss lines and is gone;
  2. the miner's own wire, `sharddiloco_glm_contributor._maybe_self_update`, then SUPPRESSED even
     that for `action in ("rate-limited", "disabled", "no-op-not-forward", None)` -- and `None` is
     exactly what its `except` branch returns when the check itself raised. A self-update that
     failed every 6 hours for months printed nothing at all;
  3. what did get printed was `out.strip()[-200:]`. A chained failure puts the PRIMARY cause FIRST
     and the unwinding finalizer's secondary exception LAST, so the tail is the half that explains
     nothing. That exact truncation once discarded a plain out-of-disk RuntimeError and cost six
     days of blaming torch.

UNTRACKED-FILE AWARENESS (issue #156) is not hypothetical either. `git checkout <commit>` ABORTS
when a file the target commit introduces already exists in the working tree as an UNTRACKED file --
and commit 4b03c06 newly TRACKED five files that had existed only as untracked copies on working
machines, `tools/self_update.py` itself among them. Any clone carrying those untracked fails its
next checkout forever, at every check, and used to say so in 200 characters of git's tail.

WHAT IS DELIBERATELY NOT TESTED HERE: nothing weakens. There is no test asserting that a bad
signature is now tolerated, because it is not -- `test_a_rejected_signature_is_loud_but_still_
refuses_the_code` pins that the loud path still REFUSES.

Torch-free, network-free, git-free: every git/pip/re-exec call is an injected fake, everything
lands in pytest tmp dirs.

Run: C:/Python313/python.exe -m pytest tests/test_self_update_loud_failure.py -q
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.self_update import (                                          # noqa: E402
    MAX_ERROR_CHARS,
    UpdateResult,
    canonical_manifest_bytes,
    check_and_update,
    format_update_status,
    untracked_blockers,
    update_status,
)

_TEST_KEY = "0x" + "11" * 32                     # the throwaway test key; NEVER a release key

# The REAL message git prints. Reproduced verbatim (tab-indented paths, trailing advice) because
# the parser's whole job is to survive git's actual formatting.
GIT_UNTRACKED_ABORT = (
    "error: The following untracked working tree files would be overwritten by checkout:\n"
    "\ttools/self_update.py\n"
    "\tneurahash/canon.py\n"
    "\ttests/conftest.py\n"
    "Please move or remove them before you switch branches.\n"
    "Aborting\n"
)


def _repo(tmp_path, version="3.7.2"):
    root = str(tmp_path / "miner")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "VERSION"), "w", encoding="utf-8") as f:
        f.write(version + "\n")
    return root


def _signed(version="3.7.3", commit="a" * 40, ts=1000):
    from neura_l1.signing import account_from_key, sign_bytes
    acct = account_from_key(_TEST_KEY)
    body = {"version": version, "git_commit": commit, "published_ts": ts}
    m = dict(body)
    m["signature"] = sign_bytes(acct, canonical_manifest_bytes(body))
    return m, acct.address


def _run(root, state, git_out="", git_rc=0, manifest=None, pubkey=None, checkout_rc=None):
    """One real check_and_update with fakes for git/pip/re-exec."""
    m, addr = (manifest, pubkey) if manifest is not None else _signed()
    commit = m["git_commit"]

    def git_fn(repo, *args, **kw):
        if args[:1] == ("fetch",):
            return 0, ""
        if args[:1] == ("checkout",):
            return (git_rc if checkout_rc is None else checkout_rc), git_out
        if args[:1] == ("rev-parse",):
            return 0, commit + "\n"
        return 0, ""

    return check_and_update(root, argv=[], manifest=m, pubkey=(pubkey or addr),
                            state_path=state, honor_rate_limit=False, environ={},
                            git_fn=git_fn, pip_fn=lambda r: (0, ""),
                            reexec_fn=lambda argv: None)


# ============================================================ 1. UNTRACKED-FILE AWARENESS (#156)
def test_untracked_blockers_parses_the_real_git_abort_message():
    """POSITIVE CONTROL for the parser: git's actual wording and tab-indented list."""
    assert untracked_blockers(GIT_UNTRACKED_ABORT) == [
        "tools/self_update.py", "neurahash/canon.py", "tests/conftest.py"]


@pytest.mark.parametrize("out", [
    "",
    "fatal: couldn't find remote ref origin\n",
    "error: pathspec 'deadbeef' did not match any file(s) known to git\n",
    # The LOCAL-MODIFICATION abort is a DIFFERENT condition with a different remedy -- it must not
    # be mislabelled as an untracked-file collision.
    "error: Your local changes to the following files would be overwritten by checkout:\n"
    "\ttools/self_update.py\nPlease commit your changes or stash them.\nAborting\n",
])
def test_untracked_blockers_is_empty_for_every_other_git_failure(out):
    """NEGATIVE CONTROL: a non-empty return is the classification, so it must not over-fire."""
    assert untracked_blockers(out) == []


def test_a_checkout_blocked_by_untracked_files_is_classified_named_and_remediated(tmp_path, capsys):
    """The whole point of #156. The outcome gets its OWN action tag, the blocking paths are named
    individually, and the operator is told what to do -- without the updater deleting anything."""
    root = _repo(tmp_path)
    res = _run(root, str(tmp_path / "s.json"), git_out=GIT_UNTRACKED_ABORT, git_rc=1)

    assert res.applied is False
    assert res.action == "git-checkout-untracked", res.action
    assert res.blockers == ["tools/self_update.py", "neurahash/canon.py", "tests/conftest.py"]

    printed = capsys.readouterr().out
    for rel in res.blockers:
        assert rel in printed, "the blocking file was not NAMED to the operator: %s" % rel
    assert "REMEDY" in printed
    assert "move" in printed.lower()
    # It must never offer to solve this by deleting: a miner's directory holds the wallet keystore.
    assert "git clean" not in printed
    assert "checkout -f" not in printed


def test_an_ordinary_checkout_failure_keeps_the_generic_tag(tmp_path):
    """NEGATIVE CONTROL at the orchestrator level: only the untracked condition gets the tag."""
    res = _run(_repo(tmp_path), str(tmp_path / "s.json"),
               git_out="fatal: reference is not a tree: aaaa\n", git_rc=1)
    assert res.action == "git-checkout-failed"
    assert res.blockers == []


# ============================================================ 2. THE FULL ERROR SURVIVES
def test_the_primary_cause_survives_instead_of_a_200_char_tail(tmp_path):
    """FAILS ON THE OLD CODE (which stored `out.strip()[-200:]`). The cause is at the TOP and the
    unwinding noise is 3 kB below it -- the exact shape that cost six days."""
    primary = "error: The following untracked working tree files would be overwritten by checkout:"
    noise = "\n".join("hint: irrelevant advice line %d" % i for i in range(200))
    out = GIT_UNTRACKED_ABORT + noise
    assert len(out) > 3000, "the fixture must exceed the old 200-char window by far"

    res = _run(_repo(tmp_path), str(tmp_path / "s.json"), git_out=out, git_rc=1)

    assert primary in res.reason, "the PRIMARY cause was truncated away"
    assert len(res.reason) >= 2000, "less than 2000 chars of the failure were preserved"
    # ...and the preserved text is the HEAD, not the tail.
    assert res.reason.index(primary) < 400


def test_a_gigantic_error_is_capped_but_capped_from_the_end(tmp_path):
    """A hostile/berserk subprocess must not blow up the state file, and the cap must still keep
    the beginning."""
    out = "error: FIRST LINE IS THE CAUSE\n" + ("x" * 200000)
    res = _run(_repo(tmp_path), str(tmp_path / "s.json"), git_out=out, git_rc=1)
    assert "FIRST LINE IS THE CAUSE" in res.reason
    assert len(res.reason) < MAX_ERROR_CHARS + 500


# ============================================================ 3. LOUD, AND ONLY WHEN IT SHOULD BE
def test_every_failure_prints_an_unmissable_banner(tmp_path, capsys):
    """FAILS ON THE OLD CODE, which emitted one `WARN:` line per failure."""
    res = _run(_repo(tmp_path), str(tmp_path / "s.json"), git_out="fatal: nope\n", git_rc=1)
    out = capsys.readouterr().out

    assert "!" * 78 in out, "no banner rule -- this scrolls past unnoticed"
    assert "NEURAHASH SELF-UPDATE FAILURE" in out
    assert res.action in out, "the banner does not name WHAT failed"
    assert "--status" in out, "the banner does not say how to get the details back"
    assert res.failed is True


def test_a_rejected_signature_is_loud_but_still_refuses_the_code(tmp_path, capsys):
    """Loud is the goal; PERMISSIVE is not. A manifest signed by the wrong key still fails closed,
    and now says so unmistakably."""
    m, _addr = _signed()
    wrong = "0x" + "22" * 20
    res = _run(_repo(tmp_path), str(tmp_path / "s.json"), manifest=m, pubkey=wrong)

    assert res.applied is False
    assert res.action == "verify-failed"
    assert "!" * 78 in capsys.readouterr().out


@pytest.mark.parametrize("action,kw", [
    ("disabled", {"enabled": False}),
    ("rate-limited", {"honor_rate_limit": True}),
])
def test_benign_outcomes_never_shout(tmp_path, capsys, action, kw):
    """NEGATIVE CONTROL. 'Nothing to do' is not a failure. A banner on every rate-limited check
    would train every operator to ignore banners, which is how you re-create the silence."""
    root = _repo(tmp_path)
    state = str(tmp_path / "s.json")
    if action == "rate-limited":
        with open(state, "w", encoding="utf-8") as f:
            json.dump({"last_check": 9e9}, f)                 # a check "just happened"
    m, addr = _signed()
    res = check_and_update(root, argv=[], manifest=m, pubkey=addr, state_path=state,
                           now=9e9, environ={}, git_fn=lambda r, *a, **k: (0, ""),
                           pip_fn=lambda r: (0, ""), reexec_fn=lambda a: None, **kw)
    assert res.action == action, res
    assert res.failed is False
    assert "!" * 78 not in capsys.readouterr().out


def test_an_already_current_miner_never_shouts(tmp_path, capsys):
    """NEGATIVE CONTROL for the healthy steady state: the manifest is not forward, so there is
    nothing wrong and nothing to say."""
    root = _repo(tmp_path, version="9.9.9")
    res = _run(root, str(tmp_path / "s.json"))
    assert res.action == "no-op-not-forward"
    assert res.failed is False
    assert "!" * 78 not in capsys.readouterr().out


def test_a_successful_update_never_shouts(tmp_path, capsys):
    """NEGATIVE CONTROL: a healthy update is not a failure."""
    res = _run(_repo(tmp_path), str(tmp_path / "s.json"))
    assert res.applied is True and res.action == "applied"
    assert "!" * 78 not in capsys.readouterr().out


# ============================================================ 4. REPORT STATE ON DEMAND
def test_the_last_failure_is_reportable_on_demand_with_the_full_reason(tmp_path):
    """FAILS ON THE OLD CODE, which persisted only `last_check` -- there was no way to ask a
    running miner what happened the last time it tried to update."""
    root = _repo(tmp_path)
    state = str(tmp_path / "s.json")
    _run(root, state, git_out=GIT_UNTRACKED_ABORT, git_rc=1)

    st = update_status(root, state_path=state)
    assert st["version"] == "3.7.2"
    last = st["last_update"]
    assert last and last["action"] == "git-checkout-untracked"
    assert last["applied"] is False
    assert last["target_version"] == "3.7.3"
    assert "untracked working tree files" in last["reason"]
    assert last["blockers"] == ["tools/self_update.py", "neurahash/canon.py", "tests/conftest.py"]

    rendered = "\n".join(format_update_status(st))
    assert "3.7.2" in rendered and "git-checkout-untracked" in rendered
    for rel in last["blockers"]:
        assert rel in rendered


def test_a_successful_update_is_recorded_before_the_reexec(tmp_path):
    """The real re-exec never returns, so anything recorded after it is never recorded at all."""
    root = _repo(tmp_path)
    state = str(tmp_path / "s.json")
    seen = {}

    m, addr = _signed()

    def reexec_fn(argv):
        seen["state_at_reexec"] = json.load(open(state, "r", encoding="utf-8"))

    check_and_update(root, argv=[], manifest=m, pubkey=addr, state_path=state,
                     honor_rate_limit=False, environ={},
                     git_fn=lambda r, *a, **k: (0, m["git_commit"] + "\n"),
                     pip_fn=lambda r: (0, ""), reexec_fn=reexec_fn)

    last = seen["state_at_reexec"]["last_update"]
    assert last["applied"] is True and last["action"] == "applied"
    assert last["target_version"] == "3.7.3"


def test_status_with_nothing_recorded_yet_is_not_an_error(tmp_path):
    st = update_status(_repo(tmp_path), state_path=str(tmp_path / "absent.json"))
    assert st["last_update"] is None
    assert "none recorded yet" in "\n".join(format_update_status(st))


# ============================================================ 5. cp1252 SAFETY
def test_the_banner_and_the_status_report_are_pure_ascii(tmp_path, capsys):
    """The Windows console is cp1252. One non-ASCII byte in a failure banner turns the loud
    failure this module exists to produce into a UnicodeEncodeError -- which would be darkly
    funny. git emits UTF-8 branch names and paths, so this is a live input, not a hypothetical."""
    root = _repo(tmp_path)
    state = str(tmp_path / "s.json")
    nasty = ("error: The following untracked working tree files would be overwritten by checkout:\n"
             "\tdocs/r\u00e9sum\u00e9\u2014notes.md\n"
             "Please move or remove them before you switch branches.\n")
    res = _run(root, state, git_out=nasty, git_rc=1)
    assert res.action == "git-checkout-untracked"

    out = capsys.readouterr().out
    out.encode("cp1252")                                   # raises if the banner is not printable
    out.encode("ascii")                                    # stricter: the banner is pure ASCII

    for line in format_update_status(update_status(root, state_path=state)):
        line.encode("ascii")


def test_a_result_with_no_detail_still_produces_a_usable_banner():
    """Defensive: `loud_lines` must never raise on a sparsely-populated result."""
    r = UpdateResult(False, "unexpected-error")
    body = "\n".join(r.loud_lines())
    assert "unexpected-error" in body
    body.encode("ascii")
