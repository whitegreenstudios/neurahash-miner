"""Corpus-on-demand: parts, assignment, the fetch filter, and eviction.

WHY THESE EXIST. A joining miner downloads the whole 14.97 GiB corpus and reads ~41 MB of it -- an
external volunteer with 20.73 GiB free could not complete the install at all (issue #71). The fix is
to publish the corpus as shuffled parts and let a miner hold one at a time
(docs/CORPUS_ON_DEMAND_DESIGN.md). Three of these tests guard properties that would be silently
catastrophic if they regressed:

  - the security allowlist must still refuse the coordinator's SECRET probe/heldout splits, including
    part-suffixed forgeries of them (F1);
  - the fetch filter must narrow what is DOWNLOADED without ever widening what is ACCEPTED;
  - _ids_path must never hand a part path to the coordinator's secret-split dir.

And one guards the point of the exercise: eviction must actually free disk, or "fetch one part"
just means "accumulate every part".
"""
import io
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_glm_contributor as N                               # noqa: E402


def _silent(*_a, **_k):
    pass


class _Args(object):
    """The three attributes _ids_path / resolve_corpus_part actually read."""

    def __init__(self, data_dir, domains="daily", corpus_part="auto"):
        self.data_dir = data_dir
        self.domains = domains
        self.corpus_part = corpus_part
        self.mode = "glm"


def _touch_parts(root, dom, idxs, split="train", rows=4):
    for i in idxs:
        a = np.arange(rows * 32, dtype=np.int64).reshape(rows, 32) + i
        np.save(os.path.join(root, N.part_filename(dom, split, i)), a)


# ---------------------------------------------------------------------------------------------
# F1: the allowlist must gain part names WITHOUT gaining the secret splits
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "ids_daily_train.npy", "ids_daily_val.npy",
    "ids_daily_train.p000.npy", "ids_daily_train.p059.npy", "ids_daily_train.p1234.npy",
    "ids_daily_val.p007.npy", "data_manifest.json",
])
def test_allowlist_accepts_splits_and_parts(name):
    assert N._is_allowed_data_name(name) is True, name


@pytest.mark.parametrize("name", [
    # the whole reason the guard exists: the coordinator-only splits, plain and part-suffixed
    "ids_daily_probe.npy", "ids_daily_heldout.npy",
    "ids_daily_probe.p000.npy", "ids_daily_heldout.p012.npy",
    # path traversal and non-basenames
    "../ids_daily_train.npy", "sub/ids_daily_train.npy", "sub\\ids_daily_train.npy",
    # near-miss part shapes that must not open a hole
    "ids_daily_train.p00.npy", "ids_daily_train.p00000.npy", "ids_daily_train.pabc.npy",
    "ids_daily_train.p000.npy.exe", "ids_daily_train.p000.pt",
    "", "config.json", "trunk.safetensors",
])
def test_allowlist_still_refuses_secrets_and_junk(name):
    assert N._is_allowed_data_name(name) is False, name


# ---------------------------------------------------------------------------------------------
# discovery + path resolution
# ---------------------------------------------------------------------------------------------

def test_parts_present_finds_only_well_formed_parts(tmp_path):
    root = str(tmp_path)
    _touch_parts(root, "daily", [0, 3, 11])
    for junk in ("ids_daily_train.npy", "ids_daily_train.pxx.npy", "ids_other_train.p001.npy",
                 "notes.txt"):
        open(os.path.join(root, junk), "wb").close()
    assert N.corpus_parts_present(root, "daily", "train") == [0, 3, 11]
    assert N.corpus_parts_present(root, "daily", "val") == []
    assert N.corpus_parts_present(os.path.join(root, "nope"), "daily") == []


def test_ids_path_prefers_a_present_part_and_falls_back(tmp_path):
    root = str(tmp_path)
    args = _Args(root)
    monolith = os.path.join(root, "ids_daily_train.npy")

    # no part chosen -> monolith
    assert N._ids_path(args, 0, "train") == monolith

    # part chosen but ABSENT -> still the monolith, so a half-migrated box degrades instead of dying
    args.corpus_part_idx = 5
    assert N._ids_path(args, 0, "train") == monolith

    _touch_parts(root, "daily", [5])
    assert N._ids_path(args, 0, "train") == os.path.join(root, "ids_daily_train.p005.npy")
    # val is never sharded by this design (131,200 B, walked in full every 8 steps)
    assert N._ids_path(args, 0, "val") == os.path.join(root, "ids_daily_val.npy")


def test_ids_path_never_returns_a_part_for_the_coordinator_secret_dir(tmp_path):
    """`base` is only ever args.coord_data_dir, which holds probe/heldout and is never sharded.
    If a part path could be returned for it, a coordinator would read miner data as its secret
    yardstick -- the leak the F1 guard exists to prevent, arriving by a different door."""
    root, coord = str(tmp_path / "miner"), str(tmp_path / "coord")
    os.makedirs(root); os.makedirs(coord)
    args = _Args(root)
    args.corpus_part_idx = 2
    _touch_parts(root, "daily", [2])
    _touch_parts(coord, "daily", [2])                 # even if one somehow exists there
    assert N._ids_path(args, 0, "train", base=coord) == os.path.join(coord, "ids_daily_train.npy")


# ---------------------------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------------------------

def test_resolve_corpus_part_auto_is_deterministic_and_spreads(tmp_path):
    root = str(tmp_path)
    _touch_parts(root, "daily", range(16))
    seen = {}
    for i in range(200):
        args = _Args(root)
        ident = "0x%040x" % i
        idx = N.resolve_corpus_part(args, ident, log=_silent)
        assert 0 <= idx < 16
        assert idx == N.resolve_corpus_part(_Args(root), ident, log=_silent)   # deterministic
        seen[idx] = seen.get(idx, 0) + 1
    # 200 identities over 16 parts: every part used, and none swallowing the field
    assert len(seen) == 16, seen
    assert max(seen.values()) < 60, seen


def test_resolve_corpus_part_off_explicit_and_absent(tmp_path):
    root = str(tmp_path)
    _touch_parts(root, "daily", [0, 1, 2, 3])
    assert N.resolve_corpus_part(_Args(root, corpus_part="off"), "id", log=_silent) is None
    assert N.resolve_corpus_part(_Args(root, corpus_part="2"), "id", log=_silent) == 2
    assert N.resolve_corpus_part(_Args(root, corpus_part="9"), "id", log=_silent) == 1   # wraps
    # no parts on disk at all -> monolith mode, whatever the flag says
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    assert N.resolve_corpus_part(_Args(empty), "id", log=_silent) is None


def test_next_corpus_part_wraps():
    assert N.next_corpus_part(0, 4) == 1
    assert N.next_corpus_part(3, 4) == 0
    assert N.next_corpus_part(2, 4, step=3) == 1
    assert N.next_corpus_part(0, 0) is None


# ---------------------------------------------------------------------------------------------
# the fetch filter: narrows the DOWNLOAD, never the validation
# ---------------------------------------------------------------------------------------------

def test_wanted_data_names_keeps_one_part_plus_everything_else():
    files = {"data_manifest.json": {}, "ids_daily_val.npy": {}}
    files.update({N.part_filename("daily", "train", i): {} for i in range(60)})
    want = N.wanted_data_names(files, "daily", 7)
    assert want == {"data_manifest.json", "ids_daily_val.npy", "ids_daily_train.p007.npy"}
    # 62 declared files -> 3 fetched: this is the whole point (60 parts would be 15 GiB again)
    assert len(files) == 62 and len(want) == 3


def test_wanted_data_names_falls_back_to_everything_when_unsure():
    files = {"ids_daily_train.npy": {}, "ids_daily_val.npy": {}}
    assert N.wanted_data_names(files, "daily", None) == set(files)     # monolith mode
    assert N.wanted_data_names(files, "daily", 3) == set(files)        # no parts declared
    parts = {N.part_filename("daily", "train", i): {} for i in (0, 1)}
    # assigned part is NOT in the record -> fetch everything rather than train on nothing
    assert N.wanted_data_names(parts, "daily", 9) == set(parts)


# ---------------------------------------------------------------------------------------------
# eviction: the half that actually bounds disk
# ---------------------------------------------------------------------------------------------

def test_evict_deletes_every_part_but_the_kept_ones(tmp_path):
    root = str(tmp_path)
    _touch_parts(root, "daily", range(6), rows=64)
    open(os.path.join(root, "ids_daily_val.npy"), "wb").close()
    before = sum(os.path.getsize(os.path.join(root, f)) for f in os.listdir(root))

    deleted, freed = N.evict_corpus_parts(root, "daily", keep=[2], log=_silent)

    assert deleted == [0, 1, 3, 4, 5]
    assert N.corpus_parts_present(root, "daily") == [2]
    assert freed > 0
    after = sum(os.path.getsize(os.path.join(root, f)) for f in os.listdir(root))
    assert before - after == freed
    assert os.path.isfile(os.path.join(root, "ids_daily_val.npy"))     # non-parts untouched


def test_evict_is_a_noop_when_nothing_to_drop(tmp_path):
    root = str(tmp_path)
    _touch_parts(root, "daily", [4])
    assert N.evict_corpus_parts(root, "daily", keep=[4], log=_silent) == ([], 0)
    assert N.corpus_parts_present(root, "daily") == [4]


# ---------------------------------------------------------------------------------------------
# the splitter: a shuffle is only safe if it is provably lossless
# ---------------------------------------------------------------------------------------------

def test_split_is_lossless_and_shuffles(tmp_path):
    """A sha256 cannot verify a shuffle -- it changes by design. So verify the two things a shuffle
    must preserve: the multiset of rows, and therefore the token histogram."""
    import glm_corpus_split as S

    src_dir, out_dir = tmp_path / "src", tmp_path / "out"
    os.makedirs(src_dir); os.makedirs(out_dir)
    src = str(src_dir / "ids_daily_train.npy")
    n, seq = 1000, 32
    rows = (np.arange(n * seq, dtype=np.int64) % 977).reshape(n, seq)
    rows[:, 0] = np.arange(n)                       # row 0 column tags each row uniquely
    np.save(src, rows)

    man = S.split(src, str(out_dir), part_rows=128, seed=5, verify=True, log=_silent)

    assert man["parts"] == 8 and man["source_rows"] == n
    assert man["verified"]["rows_match"] and man["verified"]["histogram_match"]

    got = []
    for name in sorted(man["files"]):
        a = np.load(os.path.join(str(out_dir), name), mmap_mode="r")
        got.append(np.asarray(a))
    allrows = np.concatenate(got, axis=0)
    assert allrows.shape == (n, seq)
    # every source row present exactly once -> a true permutation, nothing dropped or duplicated
    assert sorted(allrows[:, 0].tolist()) == list(range(n))
    # ...and it is actually SHUFFLED: contiguous slicing would leave the tags sorted
    assert allrows[:, 0].tolist() != list(range(n))
    # each part carries rows from across the whole source, not one contiguous stretch
    first = np.asarray(np.load(os.path.join(str(out_dir), N.part_filename("daily", "train", 0)),
                               mmap_mode="r"))
    assert first[:, 0].max() - first[:, 0].min() > n // 2


# ---------------------------------------------------------------------------------------------
# rotation: the replacement must be verified BEFORE the incumbent is deleted
# ---------------------------------------------------------------------------------------------

def test_parts_declared_reads_the_record_not_the_disk():
    files = {"data_manifest.json": {}, "ids_daily_val.npy": {}, "ids_daily_train.npy": {}}
    files.update({N.part_filename("daily", "train", i): {} for i in (0, 5, 12)})
    assert N.corpus_parts_declared(files, "daily") == [0, 5, 12]
    assert N.corpus_parts_declared({}, "daily") == []
    assert N.corpus_parts_declared(files, "other") == []


def _rot_args(tmp_path, cur, total):
    args = _Args(str(tmp_path))
    args.corpus_part_idx = cur
    args.corpus_parts_total = total
    return args


def test_rotation_fetches_then_evicts(tmp_path):
    root = str(tmp_path)
    _touch_parts(root, "daily", [0])
    args = _rot_args(tmp_path, 0, 4)

    def fake_sync(lane, data_dir, log=None, want=None, **kw):
        assert want == {"ids_daily_train.p001.npy", "data_manifest.json", "ids_daily_val.npy"}
        _touch_parts(data_dir, "daily", [1])                 # the fetch lands

    got = N.rotate_corpus_part(args, lane=None, dom="daily", log=_silent, autosync=fake_sync)
    assert got == 1 and args.corpus_part_idx == 1
    assert N.corpus_parts_present(root, "daily") == [1]      # old part gone, disk stays bounded


def test_rotation_that_cannot_verify_keeps_the_part_it_has(tmp_path):
    """glm_data_autosync fail-closes with SystemExit on unverifiable data. During ROTATION that must
    not be fatal and must not evict: a transient network failure otherwise leaves a miner with no
    corpus, which is strictly worse than training one more round on the part it already trusts."""
    root = str(tmp_path)
    _touch_parts(root, "daily", [0])
    args = _rot_args(tmp_path, 0, 4)

    def boom(lane, data_dir, log=None, want=None, **kw):
        raise SystemExit(41)

    got = N.rotate_corpus_part(args, lane=None, dom="daily", log=_silent, autosync=boom)
    assert got == 0 and args.corpus_part_idx == 0
    assert N.corpus_parts_present(root, "daily") == [0]       # NOT evicted


def test_rotation_survives_an_arbitrary_fetch_exception(tmp_path):
    root = str(tmp_path)
    _touch_parts(root, "daily", [2])
    args = _rot_args(tmp_path, 2, 4)

    def boom(lane, data_dir, log=None, want=None, **kw):
        raise RuntimeError("connection reset")

    assert N.rotate_corpus_part(args, lane=None, dom="daily", log=_silent, autosync=boom) == 2
    assert N.corpus_parts_present(root, "daily") == [2]


def test_rotation_refuses_when_the_new_part_is_not_on_disk(tmp_path):
    """A sync that returns cleanly but leaves nothing behind must not advance the index -- otherwise
    _ids_path silently falls back to a monolith that is not there and training dies later, far from
    the cause."""
    root = str(tmp_path)
    _touch_parts(root, "daily", [3])
    args = _rot_args(tmp_path, 3, 8)
    assert N.rotate_corpus_part(args, lane=None, dom="daily", log=_silent,
                                autosync=lambda *a, **k: None) == 3
    assert args.corpus_part_idx == 3
    assert N.corpus_parts_present(root, "daily") == [3]


def test_rotation_is_a_noop_without_parts_or_with_only_one(tmp_path):
    assert N.rotate_corpus_part(_rot_args(tmp_path, None, 0), None, "daily", log=_silent) is None
    assert N.rotate_corpus_part(_rot_args(tmp_path, 0, 1), None, "daily", log=_silent) == 0


def test_rotation_wraps_around_the_whole_corpus(tmp_path):
    """Over a long run a miner must eventually see every part -- that is what makes 'one part at a
    time' cost nothing in data coverage, only in what is resident at once."""
    root = str(tmp_path)
    _touch_parts(root, "daily", [0])
    args = _rot_args(tmp_path, 0, 5)

    def fake_sync(lane, data_dir, log=None, want=None, **kw):
        idx = int(sorted(n for n in want if ".p" in n)[0].split(".p")[1][:3])
        _touch_parts(data_dir, "daily", [idx])

    seen = [0]
    for _ in range(7):
        seen.append(N.rotate_corpus_part(args, None, "daily", log=_silent, autosync=fake_sync))
    assert seen == [0, 1, 2, 3, 4, 0, 1, 2]
    assert set(seen) == {0, 1, 2, 3, 4}                       # full coverage over time
    assert N.corpus_parts_present(root, "daily") == [2]       # exactly ONE resident throughout


# ---------------------------------------------------------------------------------------------
# starting batch: a small card should not have to OOM twice to find its limit
# ---------------------------------------------------------------------------------------------

class _B(object):
    def __init__(self, batch):
        self.batch = batch


@pytest.mark.parametrize("total_gib,want", [
    (31.8, 16),   # 5090 -- unchanged from the old hardcoded default
    (24.0, 16),
    (23.9, 8),
    (16.0, 8),
    (15.9, 4),
    (8.0, 4),     # MEASURED: 16 and 8 both OOM here, 4 trains at 6.36 GiB peak
    (6.0, 4),
])
def test_auto_batch_picks_from_card_size(total_gib, want, monkeypatch):
    monkeypatch.setattr(N, "_CARD_TOTAL_GIB", total_gib)
    a = _B("auto")
    assert N.resolve_start_batch(a, log=_silent) == want
    assert a.batch == want


def test_explicit_batch_is_never_second_guessed(monkeypatch):
    """An operator who typed a number means it -- including a big one on a small card, where the
    OOM backoff is the thing that rescues them."""
    monkeypatch.setattr(N, "_CARD_TOTAL_GIB", 8.0)
    for given, want in (("16", 16), ("48", 48), ("1", 1), (12, 12)):
        a = _B(given)
        assert N.resolve_start_batch(a, log=_silent) == want


def test_auto_batch_is_idempotent(monkeypatch):
    """main() resolves once, but the loops re-read args.batch every round and _oom_backoff mutates
    it. Re-resolving must never undo a backoff."""
    monkeypatch.setattr(N, "_CARD_TOTAL_GIB", 8.0)
    a = _B("auto")
    assert N.resolve_start_batch(a, log=_silent) == 4
    a.batch = 2                                       # as if _oom_backoff had halved it
    assert N.resolve_start_batch(a, log=_silent) == 2
    assert a.batch == 2


def test_auto_batch_without_a_card_keeps_the_old_default(monkeypatch):
    """CPU or an un-probed card must behave exactly as before this change (16). Note the dev box has
    a 5090, so this MUST force is_available False -- otherwise it would pass by reading the real
    card and prove nothing."""
    import torch
    monkeypatch.setattr(N, "_CARD_TOTAL_GIB", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    a = _B("auto")
    assert N.resolve_start_batch(a, log=_silent) == 16


# ---------------------------------------------------------------------------------------------
# port-damage guards. These exist because a transplant into the PUBLIC miner on 2026-08-03 sliced
# one block too far and DUPLICATED the model build -- shipped and pushed before it was noticed.
# Two models resident at once would OOM exactly the 8 GB cards this work is meant to serve, and no
# test looked, because every test called build_node_model directly instead of main().
# ---------------------------------------------------------------------------------------------

def _contributor_source():
    return io.open(N.__file__, encoding="utf-8").read()


def test_the_model_is_built_exactly_once():
    src = _contributor_source()
    n = src.count("model, cfg, seq = build_node_model(args, log=_flush)")
    assert n == 1, ("build_node_model appears %d times in main(); a duplicated build holds two "
                    "models resident and OOMs a small card" % n)


def test_no_block_is_duplicated_back_to_back():
    """Catch the whole class, not just the one instance: a bad transplant repeats a run of lines."""
    lines = _contributor_source().splitlines()
    dupes = []
    for i in range(len(lines) - 12):
        win = lines[i:i + 6]
        if sum(1 for x in win if x.strip()) < 4:
            continue
        if win == lines[i + 6:i + 12]:
            dupes.append((i + 1, win[0].strip()[:60]))
    assert not dupes, "duplicated 6-line block(s): %r" % dupes[:3]


def test_every_ported_symbol_appears_exactly_once():
    src = _contributor_source()
    for marker in ("_ALLOWED_DATA_RE = re.compile", "def evict_corpus_parts",
                   "def wanted_data_names", "def resolve_corpus_part", "def rotate_corpus_part",
                   "def corpus_parts_declared", "def resolve_start_batch", "def _ids_path",
                   "def node_ids", "fetch_names = sorted(files)",
                   "resolve_start_batch(args, log=_flush)",
                   "glm_data_autosync(lane, args.data_dir"):
        assert src.count(marker) == 1, "%r appears %d times (want 1)" % (marker, src.count(marker))


def test_part_names_round_trip_through_the_allowlist():
    """The splitter's names and the miner's guard must agree -- they are edited in different files
    and a mismatch would only surface as a stranger's fetch being refused."""
    for i in (0, 7, 59, 999):
        nm = N.part_filename("daily", "train", i)
        assert N._is_allowed_data_name(nm), nm
        assert N.corpus_parts_present.__name__                       # module wired
        import glm_corpus_split as S
        assert S.part_name("ids_daily_train.npy", i) == nm
