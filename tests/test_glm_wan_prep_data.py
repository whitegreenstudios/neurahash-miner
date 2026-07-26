"""THE TRAIN SPLIT MUST NEVER CONTAIN THE YARDSTICK'S TEXT -- structurally, not by remembering.

Held-out cross-entropy is this project's goal metric, and it only means "generalization" if the
model never trained on the held-out text. tests/test_corpus_split_stability.py covers the invariant
under corpus REFRESH. This file covers the hole that was left open under a --train-rows CHANGE, and
the content check that catches whatever the structure cannot.

THE BUG THESE TESTS PIN DOWN (measured 2026-07-26 on the real 2026-07-24 corpus, pre-fix code).
`split_of_document` assigned by content hash, but the bucket EDGES were a function of the split
proportions, and `--train-rows` changes those proportions. Raising it squeezed probe and heldout
into a narrower band; every document that fell out of the band was re-assigned to TRAIN, while
`freeze_coord` kept the OLD probe/heldout .npy files on disk. Going from the live --train-rows 64 to
4,484,375 moved 99 of 99 probe documents and 254 of 255 heldout documents into train/val -- the
model trains on the exam, every sha256 stays green, and the goal metric silently means nothing.
The pre-existing `assert not (seen & set(rng))` could not see it: it compared ROW RANGES within one
carve, and the two sides of this leak are in different carves.

Layer 1 (bucket_plan) makes the re-assignment inexpressible; layer 2 (assert_no_frozen_leak) scans
for 13-token overlaps and fails the prep on a contiguous run of LEAK_ABORT_RUN (16) tokens. Both are
tested here, and layer 1 is additionally proved BIT-IDENTICAL to the pre-fix classifier at the
reference table, so the fix cannot have moved a document that was not already moving.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

P = pytest.importorskip("glm_wan_prep_data")

B = P.SPLIT_BUCKETS
SEQ = 16                 # >= LEAK_NGRAM, so a row is long enough for the 13-token check to bite
# A production-SHAPED reference table at test scale: same four splits in the same order, same
# frozen suffix. main() passes P.SPLITS here; the mechanism under test is identical.
REF = (("train", 8), ("val", 2), ("probe", 2), ("heldout", 4))


def _with_train(rows, base=REF):
    return tuple((n, (rows if n == "train" else c)) for n, c in base)


def _splits(rows):
    """the production runtime table, exactly as main() builds it"""
    return _with_train(rows, P.SPLITS)


def _prefix_classifier(h, splits):
    """The classifier EXACTLY as it stood at HEAD before this fix (float edge accumulator), by
    bucket index. Kept here so 'bit-identical' is checked against the real prior arithmetic rather
    than against a paraphrase of it."""
    total = float(sum(n for _, n in splits))
    edge = 0.0
    for name, n in splits:
        edge += n / total
        if h < edge * B:
            return name
    return splits[-1][0]


class FakeTok:
    """One token per character: rows are predictable and no 6 GB tokenizer directory is needed."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(ord(c) % 900) + 1 for c in text]}


_ALPHA = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _docs(prefix, n, length=48):
    out = []
    for i in range(n):
        rng = np.random.default_rng(abs(hash((prefix, i))) % (2 ** 32))
        body = "".join(_ALPHA[k] for k in rng.integers(0, len(_ALPHA), size=length))
        out.append("%s %s-%04d" % (body, prefix, i))
    return out


# --------------------------------------------------------------------------------------------
# LAYER 1 -- structural
# --------------------------------------------------------------------------------------------

def test_frozen_bucket_ranges_are_identical_at_every_train_rows():
    """THE FIX, stated as one assertion. probe and heldout must occupy the same buckets whatever
    --train-rows is, because that is what makes re-assignment inexpressible rather than merely
    unlikely."""
    ranges = {}
    for T in (1, 8, 64, 4096, 100_000, 4_486_096, 6_200_000, 50_000_000):
        for name, lo, hi in P.bucket_plan(_with_train(T), B, REF):
            if name in P.COORD_ONLY:
                ranges.setdefault(name, set()).add((lo, hi))
    for name, seen in ranges.items():
        assert len(seen) == 1, "%s moved across --train-rows values: %s" % (name, sorted(seen))
    assert set(ranges) == set(P.COORD_ONLY)


def test_bucket_plan_tiles_the_whole_hash_space():
    """No gap (a document with no split) and no overlap (a document claimed by two)."""
    for T in (1, 64, 4096, 4_486_096):
        plan = P.bucket_plan(_with_train(T), B, REF)
        assert plan[0][1] == 0
        assert plan[-1][2] == B
        for (na, _, ha), (nb, lb, _) in zip(plan, plan[1:]):
            assert ha == lb, "gap/overlap between %s and %s at T=%d" % (na, nb, T)
        assert all(hi > lo for _, lo, hi in plan), plan


def test_reference_table_layout_is_bit_identical_to_the_pre_fix_classifier():
    """EXHAUSTIVE over all 2**20 buckets, for the production table and for the test table: when the
    runtime table IS the declared table (no --train-rows override), the new plan classifies every
    single bucket exactly as the pre-fix code did. Without this, 'the fix preserves today's
    assignments' would be an assertion rather than a measurement."""
    for splits in (P.SPLITS, REF):
        plan = P.bucket_plan(splits, B, splits)
        edges = np.array([hi for _, _, hi in plan])
        names = [n for n, _, _ in plan]
        got = np.take(names, np.searchsorted(edges, np.arange(B), side="right"))
        want = np.array([_prefix_classifier(h, splits) for h in range(B)])
        bad = int(np.count_nonzero(got != want))
        assert bad == 0, "%d/%d buckets differ for %s" % (bad, B, [n for n, _ in splits])


def test_raising_train_rows_no_longer_moves_documents_out_of_frozen_splits():
    """RED-then-GREEN on the mechanism itself, at document level.

    RED: under the pre-fix classifier, documents that belong to probe/heldout at a small
    --train-rows are re-assigned to train/val at a large one. GREEN: under bucket_plan, zero move.
    """
    docs = _docs("mig", 4000)
    small, large = _with_train(8), _with_train(4_486_096)
    buckets = [P.document_bucket(d, B) for d in docs]

    old_small = [_prefix_classifier(h, small) for h in buckets]
    old_large = [_prefix_classifier(h, large) for h in buckets]
    old_moved = sum(1 for a, b in zip(old_small, old_large)
                    if a in P.COORD_ONLY and b not in P.COORD_ONLY)
    assert old_moved > 0, ("the pre-fix classifier did not migrate any frozen document -- the bug "
                           "this test exists for cannot be reproduced, so the fix is unjustified")

    new_small = [P.split_of_document(d, small, B, REF) for d in docs]
    new_large = [P.split_of_document(d, large, B, REF) for d in docs]
    new_moved = [i for i, (a, b) in enumerate(zip(new_small, new_large))
                 if a in P.COORD_ONLY and b not in P.COORD_ONLY]
    assert not new_moved, ("%d document(s) still left a frozen split when --train-rows grew, e.g. "
                           "%r" % (len(new_moved), docs[new_moved[0]][:60] if new_moved else ""))


def test_fixed_size_splits_keep_a_usable_share_at_huge_train_rows():
    """MEASURED REGRESSION GUARD. With the frozen splits pinned, val's proportional slice of what is
    left collapses as train grows: at the live --train-rows it came to 89 buckets, which supplied
    415 rows against the 512 val needs, and the prep died. The floor keeps it usable. train must
    still keep the lion's share -- the live corpus has to yield 143.5M train tokens."""
    floor = int(P.MIN_FIXED_SPLIT_BUCKET_FRAC * B)
    for table in (REF, P.SPLITS):
        for T in (100_000, 4_486_096, 50_000_000):
            plan = dict((n, (lo, hi)) for n, lo, hi in
                        P.bucket_plan(_with_train(T, table), B, table))
            vlo, vhi = plan["val"]
            assert vhi - vlo >= floor, "val starved to %d buckets at T=%d" % (vhi - vlo, T)
    # and the production table must still leave train enough of the corpus: the live run needs
    # 143.5M of the ~205M tokens, i.e. > 70% of the hash space.
    for T in (100_000, 4_486_096, 50_000_000):
        plan = dict((n, (lo, hi)) for n, lo, hi in P.bucket_plan(_splits(T), B, P.SPLITS))
        tlo, thi = plan["train"]
        assert (thi - tlo) / B > 0.70, "train squeezed to %.3f of the space at T=%d" % (
            (thi - tlo) / B, T)


def test_bucket_plan_rejects_a_table_it_cannot_reason_about():
    with pytest.raises(SystemExit) as e:                       # frozen splits not a suffix
        P.bucket_plan((("probe", 1), ("train", 8), ("heldout", 2)), B)
    assert "contiguous suffix" in str(e.value)
    with pytest.raises(SystemExit) as e:                       # reference names a different table
        P.bucket_plan(REF, B, (("train", 8), ("val", 2), ("heldout", 4), ("probe", 2)))
    assert "same splits in the same order" in str(e.value)


# --------------------------------------------------------------------------------------------
# LAYER 1 -- end to end through build_domain
# --------------------------------------------------------------------------------------------

def _prep(tmp_path, docs, tag, train_rows, coord=None, **kw):
    miner = tmp_path / ("miner_%s" % tag)
    miner.mkdir()
    coord = coord or (tmp_path / "coord")
    if not coord.exists():
        coord.mkdir()
    P.build_domain(FakeTok(), "\n".join(docs), SEQ, str(miner), str(coord), "daily",
                   vocab_size=100000, splits=_with_train(train_rows), ref_splits=REF, **kw)
    return miner, coord


def _sha(path):
    import hashlib
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def test_frozen_files_are_byte_identical_across_train_rows_settings(tmp_path):
    """Acceptance shape of the real proof: carve once, then re-prep at wildly different
    --train-rows, and the coordinator's two files must not change by one byte."""
    docs = _docs("e2e", 400)
    _, coord = _prep(tmp_path, docs, "a", 8)
    want = {n: _sha(coord / ("ids_daily_%s.npy" % n)) for n in ("probe", "heldout")}
    for i, T in enumerate((16, 64, 256, 800)):
        _prep(tmp_path, docs, "t%d" % i, T, coord=coord)
        got = {n: _sha(coord / ("ids_daily_%s.npy" % n)) for n in ("probe", "heldout")}
        assert got == want, "the yardstick moved at --train-rows=%d" % T


# --------------------------------------------------------------------------------------------
# LAYER 2 -- the content leak assertion
# --------------------------------------------------------------------------------------------

def test_leak_scan_reports_zero_on_genuinely_disjoint_streams():
    frozen = [np.arange(100, 900, dtype=np.int64).reshape(25, 32)]
    stream = np.arange(5000, 9000, dtype=np.int64)
    checked, over, longest, ex = P.ngram_leak_scan(stream, P.frozen_ngram_index(frozen))
    assert over == 0 and longest == 0 and ex == []
    assert checked == stream.size - P.LEAK_NGRAM + 1


def test_leak_scan_finds_a_planted_frozen_sequence():
    rng = np.random.default_rng(7)
    frozen = [rng.integers(0, 5000, size=(8, 32), dtype=np.int64)]
    stream = rng.integers(20000, 25000, size=4000, dtype=np.int64)
    planted = frozen[0].reshape(-1)[40:40 + P.LEAK_NGRAM]
    stream[1000:1000 + P.LEAK_NGRAM] = planted
    _, over, longest, ex = P.ngram_leak_scan(stream, P.frozen_ngram_index(frozen))
    assert over == 1, over
    assert longest == P.LEAK_NGRAM
    assert ex[0][0] == 1000 and tuple(planted.tolist()) == ex[0][1]


def test_a_twelve_token_overlap_is_not_reported_as_a_leak():
    """The threshold is exact, and the exact-tuple re-check means a near miss cannot be counted."""
    rng = np.random.default_rng(11)
    frozen = [rng.integers(0, 5000, size=(8, 32), dtype=np.int64)]
    stream = rng.integers(20000, 25000, size=2000, dtype=np.int64)
    stream[500:500 + P.LEAK_NGRAM - 1] = frozen[0].reshape(-1)[10:10 + P.LEAK_NGRAM - 1]
    _, over, _, _ = P.ngram_leak_scan(stream, P.frozen_ngram_index(frozen))
    assert over == 0


def test_probe_and_heldout_are_shingled_separately():
    """An n-gram straddling the probe/heldout seam is an artefact of concatenation, not a real
    sequence, and must not become a shingle that train could 'match'."""
    a = np.arange(1, 33, dtype=np.int64).reshape(1, 32)
    b = np.arange(101, 133, dtype=np.int64).reshape(1, 32)
    idx = P.frozen_ngram_index([a, b])
    seam = np.concatenate([a.reshape(-1)[-6:], b.reshape(-1)[:7]])
    _, over, _, _ = P.ngram_leak_scan(seam, idx)
    assert over == 0


def test_stride_sampling_still_catches_a_run_of_the_documented_length():
    """The docstring promises: a stride catches any leak of >= n + stride - 1 contiguous tokens.
    Hold it to that."""
    rng = np.random.default_rng(3)
    frozen = [rng.integers(0, 5000, size=(8, 32), dtype=np.int64)]
    stride = 4
    run = P.LEAK_NGRAM + stride - 1
    for start in range(stride):                    # whatever the phase, it must be caught
        stream = rng.integers(20000, 25000, size=3000, dtype=np.int64)
        stream[700 + start:700 + start + run] = frozen[0].reshape(-1)[:run]
        _, over, _, _ = P.ngram_leak_scan(stream, P.frozen_ngram_index(frozen), stride=stride)
        assert over >= 1, "stride %d missed a %d-token run at phase %d" % (stride, run, start)


def test_assert_no_frozen_leak_announces_what_it_checked_even_when_clean():
    """A silent guard is indistinguishable from a missing one."""
    lines = []
    by_split = {"train": np.arange(9000, 13000, dtype=np.int64).reshape(125, 32),
                "val": np.arange(20000, 20320, dtype=np.int64).reshape(10, 32),
                "probe": np.arange(100, 420, dtype=np.int64).reshape(10, 32),
                "heldout": np.arange(500, 820, dtype=np.int64).reshape(10, 32)}
    over = P.assert_no_frozen_leak(by_split, REF, "daily", log=lines.append)
    assert over == 0
    blob = "\n".join(lines)
    assert "FULL PASS" in blob and "shingles" in blob and "0 overlaps" in blob
    assert "train" in blob and "val" in blob

    lines2 = []
    P.assert_no_frozen_leak(by_split, REF, "daily", stride=8, log=lines2.append)
    assert "SAMPLED 1-in-8" in "\n".join(lines2), "sampling must never be silent"


def test_assert_no_frozen_leak_aborts_and_names_the_count():
    frozen = np.arange(1, 641, dtype=np.int64).reshape(20, 32)
    train = np.arange(50000, 53200, dtype=np.int64).reshape(100, 32)
    train[3, :20] = frozen.reshape(-1)[100:120]          # 20 tokens of the yardstick, verbatim
    with pytest.raises(SystemExit) as e:
        P.assert_no_frozen_leak({"train": train, "probe": frozen}, REF, "daily", log=lambda *_: None)
    msg = str(e.value)
    assert "CONTENT LEAK" in msg and "daily" in msg
    assert "8" in msg, msg                     # 20 tokens -> 20-13+1 = 8 overlapping windows


def test_build_domain_aborts_and_deletes_the_contaminated_miner_files(tmp_path):
    """The end-to-end failure: a document whose text repeats a frozen document's text lands in
    train. The prep must die, and must not leave a shippable ids_daily_train.npy behind for
    glm_publish_data.py to pick up."""
    docs = _docs("leak", 400)
    miner1, coord = _prep(tmp_path, docs, "base", 8)
    probe_docs = [d for d in docs if P.split_of_document(d, _with_train(8), B, REF) == "probe"]
    assert probe_docs, "fixture produced no probe documents"

    # a NEW document that carries a probe document's text and itself hashes into train
    victim = probe_docs[0]
    planted = None
    for i in range(20000):
        cand = "%s zz%05d" % (victim, i)
        if P.split_of_document(cand, _with_train(8), B, REF) == "train":
            planted = cand
            break
    assert planted, "could not construct a train-bucket document carrying probe text"

    miner2 = tmp_path / "miner_leak"
    miner2.mkdir()
    with pytest.raises(SystemExit) as e:
        P.build_domain(FakeTok(), "\n".join([planted] + docs), SEQ, str(miner2), str(coord),
                       "daily", vocab_size=100000, splits=_with_train(8), ref_splits=REF)
    assert "CONTENT LEAK" in str(e.value)
    assert not os.path.exists(str(miner2 / "ids_daily_train.npy")), \
        "a contaminated train file was left on disk where the publisher would ship it"
    # the coordinator's frozen files are untouched by the failure
    assert os.path.exists(str(coord / "ids_daily_probe.npy"))
