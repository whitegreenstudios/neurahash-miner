"""THE GOAL METRIC'S INVARIANT: a document must never migrate between splits.

Held-out cross-entropy is this project's goal metric. It only means "generalization" if the model
never trained on the held-out text -- and that has to hold ACROSS corpus refreshes, not merely
within one prep.

FOUND 2026-07-25 (not yet shipped to any live run): glm_wan_prep_data carved splits by POSITION out
of one flat token stream, while tools/daily_corpus_extract.py --roll rebuilds a CUMULATIVE window
each day. Every refresh therefore re-cut a growing pool at fixed offsets, so a row held out
yesterday could become training data today -- silently, with every mechanical check green. That is
the failure this project was founded on (memory pouw-verified-not-useful: ~900 rounds paid while
held-out validation worsened). A fresh corpus was deliberately NOT published on 2026-07-25 because
of this.

test_documents_never_migrate_between_splits is the one that matters: it FAILS on the old
position-based carve and passes on content-addressed assignment.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

P = pytest.importorskip("glm_wan_prep_data")

SMALL = (("train", 4), ("val", 1), ("probe", 1), ("heldout", 2))
SEQ = 4


class FakeTok:
    """One token per character, so `seq`-token rows are predictable and no model is needed."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(ord(c) % 900) + 1 for c in text]}


_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _docs(prefix, n, length=48):
    """Documents must be DISTINCT text, like a real corpus. Padding them with a repeated character
    makes different documents tokenize to identical rows, and the disjointness checks below then
    fail on row collisions that have nothing to do with split assignment."""
    out = []
    for i in range(n):
        rng = np.random.default_rng(abs(hash((prefix, i))) % (2 ** 32))
        body = "".join(_ALPHABET[k] for k in rng.integers(0, len(_ALPHABET), size=length))
        # unique body FIRST: a shared "prefix-0001 " head would tokenize to the same opening row
        # in every document, and the disjointness checks would trip on that, not on split leakage.
        out.append("%s %s-%04d" % (body, prefix, i))
    return out


def _prep(tmp_path, docs, tag):
    miner = tmp_path / ("miner_%s" % tag)
    coord = tmp_path / ("coord_%s" % tag)
    miner.mkdir()
    coord.mkdir()
    P.build_domain(FakeTok(), "\n".join(docs), SEQ, str(miner), str(coord),
                   "daily", vocab_size=100000, splits=SMALL)
    return miner, coord


def _rows(d, name, domain="daily"):
    a = np.load(os.path.join(str(d), "ids_%s_%s.npy" % (domain, name)))
    return {tuple(int(x) for x in r) for r in a}


def test_no_document_supplies_two_splits(tmp_path):
    """THE REAL DEFECT, and the one that fails on the old code: the flat carve sliced ONE token
    stream at fixed row offsets with no regard for document boundaries, so a document straddling a
    boundary supplied two splits at once. Measured on the pre-fix build with these fixtures, a
    single document supplied BOTH `train` and `heldout`.

    At production split sizes the blast radius is small -- only the three documents sitting on the
    train/val, val/probe and probe/heldout boundaries -- but one of those three is shared between
    the coordinator's secret probe pool and the held-out goal metric, and nothing detected it.
    Splitting per document removes the failure mode by construction."""
    docs = _docs("s", 60)
    miner, coord = _prep(tmp_path, docs, "straddle")
    per_split = {n: _rows(d, n) for n, d in
                 (("train", miner), ("val", miner), ("probe", coord), ("heldout", coord))}
    tok = FakeTok()
    for i, doc in enumerate(docs):
        ids = tok(doc)["input_ids"]
        rows = {tuple(ids[k:k + SEQ]) for k in range(0, len(ids) - SEQ + 1)}
        hit = sorted(n for n, r in per_split.items() if rows & r)
        assert len(hit) <= 1, (
            "document %d's text appears in %s -- one document must never supply two splits, or the "
            "held-out metric is scored on text the model trained on" % (i, hit))


def test_split_membership_survives_a_rebuilt_roll(tmp_path):
    """A document keeps its split when the corpus is rebuilt, whatever the order.

    HONEST SCOPE: with today's roll this is a GUARD, not a bug fix. daily_corpus_extract.roll_domain
    walks `_dates_back` NEWEST FIRST and dedups by first occurrence, so each refresh PREPENDS -- old
    content only ever moves to later offsets or falls off the end, never earlier into the train
    region. The old position-carve therefore did not actually migrate held-out rows into training,
    and a test claiming it did passes on the broken code. Verified twice before this was written.

    But that safety was an accident of roll ordering that nothing recorded or tested. Sort the roll,
    walk it oldest-first, or filter a line out of the middle, and a position-carve starts moving
    documents between splits immediately -- silently, with every mechanical check green (memory
    pouw-verified-not-useful). Content-addressed assignment makes the invariant hold by
    construction, so this test stays green under ANY future roll ordering."""
    day1 = _docs("a", 60)
    _, coord1 = _prep(tmp_path, day1, "d1")
    heldout_day1 = _rows(coord1, "heldout")

    for tag, day2 in (("prepend", _docs("b", 60) + day1),        # what roll_domain does today
                      ("append", day1 + _docs("c", 60)),
                      ("sorted", sorted(day1 + _docs("d", 60)))):  # any future reordering
        miner2, _ = _prep(tmp_path, day2, "d2_" + tag)
        leaked = heldout_day1 & _rows(miner2, "train")
        assert not leaked, (
            "%s: %d row(s) held out on day 1 became TRAINING data after the roll was rebuilt"
            % (tag, len(leaked)))


def test_split_assignment_is_pure_content(tmp_path):
    """Same document -> same split, regardless of what else is in the corpus or in what order."""
    doc = "the quick brown fox jumps over the lazy dog"
    a = P.split_of_document(doc, SMALL)
    b = P.split_of_document(doc, SMALL)
    assert a == b
    # position in the corpus is irrelevant: the function never sees it
    assert P.split_of_document(doc, SMALL) == a


def test_every_split_is_populated_and_disjoint(tmp_path):
    miner, coord = _prep(tmp_path, _docs("c", 120), "mix")
    seen, per = set(), {}
    for name, d in (("train", miner), ("val", miner), ("probe", coord), ("heldout", coord)):
        r = _rows(d, name)
        per[name] = len(r)
        assert r, "%s split is empty" % name
        assert not (seen & r), "%s shares rows with an earlier split" % name
        seen |= r
    assert per["train"] == 4 and per["heldout"] == 2, per


def test_secret_splits_never_land_in_the_miner_dir(tmp_path):
    """F1 structural: probe + heldout are the coordinator's secret pool and goal metric."""
    miner, coord = _prep(tmp_path, _docs("d", 120), "f1")
    files = os.listdir(str(miner))
    assert not [f for f in files if "probe" in f or "heldout" in f], files
    assert sorted(os.listdir(str(coord))) == ["ids_daily_heldout.npy", "ids_daily_probe.npy"]


def test_too_little_text_fails_loudly_rather_than_shrinking_a_split(tmp_path):
    with pytest.raises(SystemExit) as e:
        _prep(tmp_path, _docs("e", 3), "thin")
    assert "need" in str(e.value)
