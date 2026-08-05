"""The 0xC0000005 fold-crash: the F2 re-gate closure must never outlive its released mapping.

WHY THIS EXISTS (incident, reference 4060, 2026-08-03..05). The miner died with exit 3221225477
(= 0xC0000005, Windows ACCESS VIOLATION -- the process never unwinds, so no Python traceback) about
once an hour, always right after a completed round, never mid-train. 9 of 13 usable fatal stack
beats were IDENTICAL:

    sharddiloco_glm_expert.py:393 heldout_ce   <- <lambda>   <- apply_accepted   <- _fold_accepted_checked

Mechanism, all on MainThread -- no thread race involved:
  * make_regate_ce returns `lambda h: G.heldout_ce(h.model, val_ids)` -- a closure over the
    val_ids ARRAY OBJECT (sharddiloco_glm_contributor.py, make_regate_ce).
  * node_ids hands out np.memmap views, and the round loop's corpus-resync boundary calls
    _release_ids(train_ids, val_ids) EVERY tick (resync defaults ON since 2026-07-25).
    _release_ids force-close()s the mapping: a closed memmap is NOT protected by refcounts --
    the array object survives, its pages do not.
  * The loop then re-opened its LOCALS but rebuilt regate_ce only `if _refreshed:` -- i.e. almost
    never, while the release ran every tick. The next own-slot accepted fold called the stale
    closure, and heldout_ce's `torch.as_tensor(ids[i:i+n]).to(dev)` (expert.py:393) read the
    unmapped pages: instant access violation, no exception, dead miner.

The fix rebinds regate_ce under the SAME condition as the ids re-open (both the resync and the
rotation dance). These tests pin: (1) the hazard itself -- release really unmaps under a live
closure, which is WHY the rebind is mandatory; (2) the regression -- one real _run_async tick on
the unchanged-corpus path must rebuild the re-gate with a LIVE mapping (fails on the pre-fix
code, which leaves the closure holding the closed one); (3) determinism -- the rebound closure's
CE is EXACTLY equal (float ==, not approx) to the CE over the original mapping and over an owned
in-RAM copy, so the fix cannot move the pay-gate number by even 1 ULP.

NOTE for future editors: no test here ever READS data through a released memmap -- doing so is
exactly the access violation and would kill the pytest process, not fail the test.

Run: C:/Python313/python.exe -m pytest tests/test_glm_regate_stale_mmap.py -q
"""
import gc
import os
import sys
import types

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_glm_contributor as N                               # noqa: E402
import sharddiloco_glm_expert as G                                    # noqa: E402


class _Args:
    """Exactly the attributes the code under test reads before the tick is cut short."""

    def __init__(self, data_dir, mode="glm", domains="daily"):
        self.mode = mode
        self.data_dir = data_dir
        self.domains = domains
        self.max_rounds = 2
        self.poll = 0.01
        self.corpus_rotate_rounds = 0


def _write_ids(d, split, rows=12, seq=16, vocab=24, seed=0):
    """Real .npy split with token ids < vocab (build_tiny_glm's default vocab), int64."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, vocab, size=(rows, seq), dtype=np.int64)
    np.save(os.path.join(d, "ids_daily_%s.npy" % split), a)
    return a


class _OneTick(BaseException):
    """Raised by the fake lane's in-loop read_pointer to stop _run_async after EXACTLY one
    resync boundary. A BaseException subclass on purpose: the round loop's `except Exception`
    paths must not swallow it."""


class _FakeLane:
    """Serves one data record (manifest -> sha -> get_json, the _read_data_record resolution).
    The record never changes, so the periodic resync always takes the cheap unchanged path
    (_refreshed=False) -- the exact path that left the stale closure armed. read_pointer is the
    tick-cutter: the init call (no man=) reports no pointer; the in-loop call (man=...) raises."""

    def __init__(self):
        self.record = {"manifest_sha256": "a" * 64, "seeds": ["http://seedA"], "files": {}}

    def manifest(self):
        return {N.DATA_RECORD_NAME: {"sha256": "REC-CID"}}

    def get_json(self, cid):
        if cid == "REC-CID":
            return self.record
        raise KeyError(cid)

    def read_pointer(self, man=None, **kw):
        if man is None:
            return None                       # _run_async's init probe: no pointer yet
        raise _OneTick()                      # first in-loop read: one full resync boundary ran


def _quiet_init(monkeypatch, calls):
    """Stub the claim/update machinery that runs before the seam under test -- none of it is part
    of the release/re-open boundary -- and wrap make_regate_ce with a recorder that still builds
    the REAL closure."""
    monkeypatch.delenv("NEURAHASH_GLM_DATA_RESYNC", raising=False)    # default = ON
    monkeypatch.delenv("NEURAHASH_GLM_DATA_AUTOSYNC", raising=False)
    monkeypatch.setattr(N, "_maybe_self_update", lambda log: None)
    monkeypatch.setattr(N, "claim_all_coords", lambda args, slots: [(0, 0)])
    monkeypatch.setattr(N, "claim_walk_order",
                        lambda coords, identity=None, ranked=None: list(coords))
    monkeypatch.setattr(N, "layer_claim_enabled", lambda environ=None: False)
    _state = types.SimpleNamespace(path=None,
                                   restore_cooldown=lambda cooldown, event=None: (0, 0),
                                   restore_ladder=lambda ladder: (0, 0))
    monkeypatch.setattr(N, "ClaimState",
                        types.SimpleNamespace(for_args=lambda args, ident, log=None: _state))
    real = N.make_regate_ce

    def recorder(G_, model_, val_ids, miner="", log=None):
        calls.append({"val": val_ids, "log": log})
        return real(G_, model_, val_ids, miner=miner, log=log)

    monkeypatch.setattr(N, "make_regate_ce", recorder)


def test_release_really_unmaps_under_a_live_closure(tmp_path):
    """The HAZARD the rebind exists for (passes before and after the fix -- this is the mechanism,
    not the regression): make_regate_ce captures the val_ids array OBJECT in its closure, and
    _release_ids closes the mapping under it anyway. Refcounts do not save you: after this, any
    data read through the captured array is the 0xC0000005."""
    d = str(tmp_path)
    _write_ids(d, "val")
    val = N.node_ids(_Args(d), 0, "val")
    assert isinstance(val, np.memmap)

    fn = N.make_regate_ce(types.SimpleNamespace(heldout_ce=lambda m, v: 0.0),
                          types.SimpleNamespace(), val, miner="t", log=None)
    assert fn is not None and not isinstance(fn, N.RegateUnavailable)
    assert any(c.cell_contents is val for c in fn.__closure__), \
        "make_regate_ce no longer closes over the array object -- this suite's premise changed"

    assert N._release_ids(val) is True
    assert val._mmap.closed is True, \
        "close() was refused -- if numpy starts refusing, the whole hazard class is gone"
    # DELIBERATELY no data access through `val` or `fn` beyond this point.


def test_one_tick_rebinds_the_regate_after_the_release(tmp_path, monkeypatch):
    """THE REGRESSION (fails on the pre-fix code). One real _run_async tick on the
    unchanged-corpus resync path must rebuild regate_ce because the boundary released the mapping
    its old closure captured -- rebuild-on-_refreshed-only leaves a dangling closure armed for the
    next own-slot fold. Asserts on the recorder, never on the dead pages."""
    d = str(tmp_path)
    _write_ids(d, "train")
    _write_ids(d, "val")
    args = _Args(d)
    calls = []
    _quiet_init(monkeypatch, calls)

    train = N.node_ids(args, 0, "train")
    val = N.node_ids(args, 0, "val")
    logs = []
    with pytest.raises(_OneTick):
        N._run_async(args, _FakeLane(), types.SimpleNamespace(slots=[(0, 0)]),
                     types.SimpleNamespace(),            # trunk-bearing stand-in (no _nh_trunk_free)
                     None, G, b"k", 0, 0, 0, "t0", train, val, 16,
                     (lambda *a: logs.append(" ".join(str(x) for x in a))))

    # EVERY assert below compares PLAIN SCALARS extracted first, never the recorder entries
    # themselves: on the pre-fix code the first assert FAILS, and if `calls` (which holds the
    # RELEASED memmap) appears in a failing assert expression, pytest's saferepr walks it,
    # numpy's repr reduces over the dead mapping, and the test run dies with the very access
    # violation under test instead of reporting a failure (observed live, 2026-08-05).
    # Reading `._mmap.closed` is metadata only -- it never touches the mapped pages.
    try:
        n_binds = len(calls)
        old_closed = bool(calls[0]["val"]._mmap.closed) if n_binds >= 1 else None
        vals_distinct = (n_binds >= 2) and (calls[1]["val"] is not calls[0]["val"])
        new_live = (n_binds >= 2) and (calls[1]["val"]._mmap.closed is False)
        rebind_silent = (n_binds >= 2) and (calls[1]["log"] is None)
        assert n_binds >= 2, (
            "regate_ce was NOT rebound after _release_ids closed the mapping its closure "
            "captured (binds=%d, old mapping closed=%s): the next own-slot fold would read "
            "unmapped pages through heldout_ce (sharddiloco_glm_expert.py:393) and die with "
            "0xC0000005 and no traceback (reference-4060 crash, 2026-08-03..05)"
            % (n_binds, old_closed))
        assert vals_distinct, "rebind must capture the RE-OPENED array, not the released one"
        assert old_closed is True, \
            "the boundary is expected to have closed the old mapping (that is the hazard)"
        assert new_live, "the rebound closure must hold a LIVE mapping"
        assert rebind_silent, \
            "the every-tick unchanged rebind must stay silent (trunk-free spam guard)"
        assert n_binds == 2, "one tick = exactly one rebind (init + boundary), got %d" % n_binds
    finally:
        for c in calls:                        # let Windows unlink tmp_path afterwards
            N._release_ids(c["val"])
        N._release_ids(train, val)
        gc.collect()


def test_regate_ce_is_bit_identical_across_reopen_and_owned_copy(tmp_path, monkeypatch):
    """DETERMINISM (the pay-gate constraint): the CE the REBOUND closure computes -- a re-opened
    mapping of the same verified file -- equals the CE over the ORIGINAL mapping and over an owned
    in-RAM copy EXACTLY (float ==, no tolerance). Same bytes, same real tiny-GLM forward, same
    chunking; a byte-copy cannot move the number by 1 ULP. Chunk forced to 4 so the 10-row split
    exercises multiple chunks including a ragged last one."""
    torch = pytest.importorskip("torch")                              # noqa: F841
    monkeypatch.setenv("NEURAHASH_GLM_EVAL_CHUNK", "4")
    d = str(tmp_path)
    _write_ids(d, "val", rows=10, seq=16, vocab=24, seed=7)
    args = _Args(d)
    model, _cfg = G.build_tiny_glm(seed=3)                            # real Glm4MoeLite, CPU, fp32
    host = types.SimpleNamespace(model=model)

    val1 = N.node_ids(args, 0, "val")
    fn_before = N.make_regate_ce(G, model, val1, miner="ce", log=None)
    ce_original = fn_before(host)             # over the original zero-copy mapping, while it lives

    assert N._release_ids(val1) is True       # the resync boundary
    val2 = N.node_ids(args, 0, "val")         # the fix's re-open of the SAME file
    try:
        fn_after = N.make_regate_ce(G, model, val2, miner="ce", log=None)
        ce_reopened = fn_after(host)
        ce_owned_copy = G.heldout_ce(model, np.array(val2))           # writable in-RAM copy

        assert np.isfinite(ce_original)
        assert ce_original == ce_reopened, (
            "re-opening the same corpus file changed heldout CE: %.17g != %.17g"
            % (ce_original, ce_reopened))
        assert ce_original == ce_owned_copy, (
            "an owned byte-copy changed heldout CE: %.17g != %.17g"
            % (ce_original, ce_owned_copy))
    finally:
        N._release_ids(val2)
        gc.collect()
