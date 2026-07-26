#!/usr/bin/env python3
"""Tokenize the real corpus into per-domain id files for the GLM shardDiLoCo WAN run.

WHY THIS EXISTS. `tools/sharddiloco_glm_expert.py` today trains and evaluates on
`markov_dataset(vocab=24)` (:193-204), and `tools/sharddiloco_glm_gpu_smoke.py` on random ids
0..95 out of GLM's 154,880-token vocabulary. A cross-entropy on either is arithmetically real but
is NOT a language number, so it cannot serve as the goal metric for a real-model run. This script
produces genuine GLM-tokenized text so the held-out CE the coordinator reports means something.

SPLIT DISCIPLINE (mirrors `sharddiloco_harness.domain_splits`): four DISJOINT row sets per domain.
  train   - the miner trains on this          -> MINER-FACING dir (<out>/miner)
  val     - the miner's own save-best signal   -> MINER-FACING dir (<out>/miner), public to the miner
  probe   - the COORDINATOR's secret gate pool -> COORDINATOR-ONLY dir (<out>/coord)
  heldout - the reported goal metric           -> COORDINATOR-ONLY dir (<out>/coord)
Each DOCUMENT (corpus line) is assigned to a split by a hash of its content and each split is
tokenized separately, so a document keeps its split however the corpus is later rebuilt and no row
can straddle two splits. probe + heldout are additionally FROZEN once written (--refresh-coord-
splits to re-cut): the goal metric's yardstick must not move under a daily corpus refresh, or
"held-out CE improved" compares two different exams.

NO HELD-OUT TEXT MAY REACH THE TRAIN SPLIT, AND THAT IS ENFORCED HERE, IN TWO LAYERS (2026-07-26).
Until this date the enforcement lived in one-off scripts outside the repo that someone had to
remember to run, and the tool itself had a hole: split assignment was a function of the split
PROPORTIONS, so raising --train-rows narrowed probe/heldout's band and every document that fell out
of it was re-assigned to TRAIN, while the frozen .npy files on disk kept the old text. The model
then trained on the exam with every sha256 check green and the goal metric quietly meaningless.
MEASURED on the real 2026-07-24 corpus with the pre-fix code: going from the live --train-rows 64
to 4,484,375 moved 99 of 99 probe documents and 254 of 255 heldout documents into train/val.
  LAYER 1, STRUCTURAL -- bucket_plan(). probe and heldout are allocated their slice of the hash
    space from a denominator that does not contain --train-rows, so their bucket ranges are the same
    at every train size and the re-assignment above is not expressible. --train-rows now only
    redistributes the region below them, between train and val.
  LAYER 2, DEFENSIVE -- assert_no_frozen_leak(). After carving, every miner-facing token stream is
    scanned for 13-token sequences that also occur in the frozen probe/heldout arrays, and the prep
    ABORTS on a contiguous overlap of LEAK_ABORT_RUN tokens (16, a measured threshold -- read its
    comment before changing it). This catches what layer 1 structurally cannot: the same text
    arriving in train from a DIFFERENT source document, and any future edit that re-opens the hole.
    Measured on the live artifacts: 49,035 shingles vs 143,555,060 positions, full pass, 16.4 s.
Both layers print what they checked and what they found on every run -- a silent guard is
indistinguishable from a missing one.

PROBE-POOL SECRECY IS A HARD OPERATIONAL REQUIREMENT. The per-round rotation (dm.SecretRotatedProbe
draws a fresh subset each round) does NOT protect against a miner who obtains the whole probe POOL:
that miner can simply train on the entire pool, a superset of every per-round draw, and the gate is
defeated -- re-creating this project's "verified != useful" disaster. The probe and heldout id files
must therefore NEVER be shipped to a miner box. They are written to a SEPARATE coordinator-only
subdir (<out>/coord) that lives ONLY on the coordinator box; only <out>/miner (train+val) is ever
copied to a miner. Do not ship the <out> root either -- ship <out>/miner. tools/
sharddiloco_glm_contributor.py's default --data-dir is <out>/miner precisely so the natural
"ship my data dir" action carries no secret split; the coordinator reads probe/heldout from
--coord-data-dir (<out>/coord).

ONE DOMAIN PER EXPERT SLOT. GLM routes with its OWN learned router -- there is no offline
domain-routing here as there is in the toy harness -- so expert specialisation has to come from
the DATA. Slot 0 gets one corpus, slot 1 another.

Env: C:/Python313/python.exe (never .venv). Keep stdout ASCII (cp1252 console).
Offline: the tokenizer is loaded from a local directory; HF_HUB_OFFLINE is set before the import
so no code path can reach the hub (an unbounded hub check is a measured indefinite hang here).
"""
import argparse
import functools
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_corpus_v2")
DEFAULT_TOK = r"D:\hf_models\GLM-4.7-Flash-bf16"
DEFAULT_OUT = r"D:\glm_wan"

# rows per split, in carve order (train count overridable via --train-rows for long soaks)
SPLITS = (("train", 4096), ("val", 512), ("probe", 512), ("heldout", 1024))


def _splits_with_train(train_rows):
    return tuple((name, (train_rows if name == "train" else n)) for name, n in SPLITS)
# splits that must NEVER reach a miner box -> written to the coordinator-only subdir (F1)
COORD_ONLY = frozenset({"probe", "heldout"})


# Hash granularity for content-addressed split assignment. Only affects how EVENLY documents
# distribute -- but the granularity is a hard floor on how small a split's share can be.
#
# Raised from 8192 to 2**20 on 2026-07-25. At 8192 buckets a ~205M-token corpus needs
# --train-rows ~6.2M, and train's share then rounds up to the point where `val` is allocated ZERO
# buckets, so prep dies with "split val: only 0 rows ... need 512". That is structural, not bad luck:
# once train consumes nearly the whole corpus, val's share falls below one bucket. A finer grid lets
# small splits keep a nonzero share at large train sizes.
#
# CORRECTION 2026-07-26. This comment used to end with "this does NOT move the goal metric: probe
# and heldout are read back from disk when they already exist". That is true of the FILES and it is
# BESIDE THE POINT, and as written it talks the reader out of looking at the real hazard, which is on
# the TRAIN side: the frozen .npy files stayed put while the DOCUMENTS behind them were re-assigned
# into train, so the miner trained on the goal metric's own text with every sha256 green. See
# bucket_plan() for the structural fix and ngram_leak_scan() for the check that catches what the
# structure misses. MEASURED on the real 2026-07-24 corpus with the pre-fix code: raising
# --train-rows from the live 64 to 4,484,375 moved 99 of 99 probe documents and 254 of 255 heldout
# documents into train/val.
SPLIT_BUCKETS = 1 << 20         # 1,048,576

# THE SPLITS TABLE ABOVE IS THE *REFERENCE*, AND --train-rows DOES NOT CHANGE IT.
#
# That sentence is the whole structural fix. probe and heldout are sized as n / sum(SPLITS) of the
# hash space -- a denominator holding the DECLARED train count (4096), never the --train-rows the
# caller passed -- so their bucket ranges come out identical at every train size and a document that
# hashes into probe cannot be re-assigned to train by making train bigger. --train-rows redistributes
# only the region BELOW those ranges, between train and val.
#
# Mechanically: every function here takes the runtime table as `splits` and the reference table as
# `ref_splits`; main() passes `_splits_with_train(args.train_rows)` and `SPLITS` respectively. When
# `ref_splits` is omitted it defaults to `splits`, so a caller that declares its own table (the tests
# do) gets exactly the pre-fix behaviour for that table -- the fix engages precisely where the bug
# was, at the point --train-rows diverges from the declared table.
#
# Changing SPLITS' train entry re-cuts probe/heldout on the next --refresh-coord-splits and therefore
# STARTS A NEW CAMPAIGN's baseline, exactly as --refresh-coord-splits does. Treat it that way.

# Documents per tokenizer call. Bounds peak RAM during prep -- see the chunking comment in carve().
# 2000 docs of ~1.3 KB is a few MB of text per call, which the fast tokenizer handles comfortably.
TOK_DOC_BATCH = 2000


ELASTIC_SPLIT = "train"          # the only split --train-rows resizes; everything else is fixed-size

# Floor on the bucket share of a fixed-size, miner-facing split (i.e. `val`). MEASURED NECESSITY,
# not taste: with the frozen splits pinned (below), train+val share 75% of the hash space, and at the
# live --train-rows=4,486,096 val's proportional slice of that came to 89 buckets = 76 documents =
# 415 rows against the 512 it needs -- the prep would have died with "split val: only 415 rows ...
# need 512". 1/128 of the space is ~1.6M tokens of the live 205M-token corpus, about 100x val's
# requirement, and costs train 0.8% of the corpus (it keeps ~152M tokens against the 143.5M it
# needs). val is a REGENERATED miner-facing artifact -- the miner's own save-best signal -- so
# resizing its share moves no yardstick; probe and heldout are the ones that must not move.
MIN_FIXED_SPLIT_BUCKET_FRAC = 1.0 / 128


@functools.lru_cache(maxsize=None)
def bucket_plan(splits=SPLITS, buckets=SPLIT_BUCKETS, ref_splits=None):
    """Map each split to a half-open bucket range [lo, hi) of the 0..buckets-1 hash space.

    LAYER 1 OF THE ANTI-LEAK DESIGN (the structural one). The frozen splits (COORD_ONLY: probe and
    heldout -- the coordinator's secret gate pool and the goal metric) are allocated their ranges
    FIRST, out of ``ref_splits`` -- the DECLARED table, whose train count --train-rows never touches.
    Whatever --train-rows is passed, probe and heldout occupy the same buckets, so growing train
    cannot move a document out of a frozen split. Only the region BELOW them is partitioned using the
    runtime counts, which is the one thing --train-rows is allowed to change.

    ``ref_splits`` defaults to ``splits``, so a caller that declares its own table gets exactly the
    pre-fix layout for that table; main() passes the runtime table as ``splits`` and the module-level
    SPLITS as ``ref_splits``, which is precisely where the two diverge and where the bug lived.

    Boundaries are exact integer ceilings of n*buckets/total. The pre-fix code compared
    ``h < edge*buckets`` with a float accumulator; ceil() reproduces that comparison for every one of
    the 2**20 buckets (proved exhaustively in the tests, not assumed).

    Requires the frozen splits to be a contiguous suffix of ``splits`` -- they are (the carve order
    is train, val, probe, heldout) -- so the non-frozen splits get one contiguous region and the
    whole layout is auditable by reading four numbers.
    """
    ref = tuple(ref_splits) if ref_splits else tuple(splits)
    names = [name for name, _ in splits]
    if [name for name, _ in ref] != names:
        raise SystemExit("bucket_plan: ref_splits %s must name the same splits in the same order "
                         "as %s" % ([n for n, _ in ref], names))
    frozen_at = [i for i, name in enumerate(names) if name in COORD_ONLY]
    if frozen_at and frozen_at != list(range(frozen_at[0], len(names))):
        raise SystemExit("bucket_plan: frozen splits %s must be a contiguous suffix of %s"
                         % (sorted(COORD_ONLY), names))
    cut = frozen_at[0] if frozen_at else len(names)

    ref_total = sum(n for _, n in ref)
    frozen_lo = -(-sum(n for _, n in ref[:cut]) * buckets // ref_total)   # ceil()

    # --- region [0, frozen_lo): the miner-facing splits, sized by the RUNTIME counts --------------
    nf = list(splits[:cut])
    nf_total = sum(n for _, n in nf) or 1
    edges = [-(-sum(n for _, n in nf[:i]) * frozen_lo // nf_total) for i in range(len(nf) + 1)]
    widths = [edges[i + 1] - edges[i] for i in range(len(nf))]
    floor = int(MIN_FIXED_SPLIT_BUCKET_FRAC * buckets)
    elastic = next((i for i, (name, _) in enumerate(nf) if name == ELASTIC_SPLIT), None)
    if elastic is not None:
        for i, (name, _) in enumerate(nf):
            if i != elastic and widths[i] < floor:
                widths[elastic] -= floor - widths[i]
                widths[i] = floor
        if widths[elastic] < 1:
            raise SystemExit(
                "bucket_plan: the fixed-size splits' floor (%d buckets each) leaves nothing for %s. "
                "The hash grid is too coarse for this table -- raise SPLIT_BUCKETS."
                % (floor, ELASTIC_SPLIT))

    plan, cum = [], 0
    for (name, _), w in zip(nf, widths):
        plan.append((name, cum, cum + w))
        cum += w
    cum = sum(n for _, n in ref[:cut])
    for (name, _), (_, rn) in zip(splits[cut:], ref[cut:]):   # probe, heldout: fixed, at the top
        lo = -(-cum * buckets // ref_total)
        cum += rn
        hi = -(-cum * buckets // ref_total)
        plan.append((name, lo, hi))
    if plan:
        plan[-1] = (plan[-1][0], plan[-1][1], buckets)        # absorb any rounding tail
    return tuple(plan)                        # immutable: bucket_plan is lru_cached


def document_bucket(doc, buckets=SPLIT_BUCKETS):
    """The document's bucket -- 32 bits of sha256 over its UTF-8 bytes, folded into `buckets`."""
    return int(hashlib.sha256(doc.encode("utf-8", "replace")).hexdigest()[:8], 16) % buckets


def split_of_document(doc, splits=SPLITS, buckets=SPLIT_BUCKETS, ref_splits=None):
    """Which split a document belongs to -- a pure function of its CONTENT, forever.

    THIS IS THE INVARIANT THE GOAL METRIC RESTS ON. Held-out cross-entropy only means
    "generalization" if the model never trained on the held-out text. That has to hold across
    corpus refreshes AND across --train-rows changes, not just within one prep.

    The previous scheme carved splits by POSITION out of one flat token stream (`arr[off:off+n]`
    per split), which had a MEASURED defect: the stream was sliced with no regard for document
    boundaries, so a document sitting on a boundary supplied two splits at once. With small
    fixtures a single document supplied train, val, probe AND heldout; at production sizes it is
    the three documents on the internal boundaries, one of them shared between the coordinator's
    secret probe pool and the goal metric.

    (Recorded because it was checked and found FALSE: position-carving did NOT migrate held-out
    rows into training across daily refreshes. `daily_corpus_extract.roll_domain` walks
    `_dates_back` newest-first and dedups by first occurrence, so a refresh PREPENDS -- old content
    only moves to later offsets or falls off the end, never earlier into the train region. Two
    tests written to demonstrate that migration both passed on the unfixed code.)

    Content-addressing made split membership permanent under corpus GROWTH, but until 2026-07-26 it
    was still a function of the split PROPORTIONS, and --train-rows changes those. Raising
    --train-rows squeezed probe and heldout into a narrower band, and every document that fell out
    of the band landed in train while the frozen .npy files on disk kept the old text -- a silent
    train-on-the-exam leak that no sha256 could see. bucket_plan() removes that degree of freedom:
    the frozen splits' buckets are cut from a denominator that does not mention --train-rows.
    """
    h = document_bucket(doc, buckets)
    for name, lo, hi in bucket_plan(splits, buckets, ref_splits):
        if lo <= h < hi:
            return name
    return splits[-1][0]


LEAK_NGRAM = 13                  # contiguous TOKENS scanned for on both sides

# Contiguous overlapping TOKENS that constitute a leak rather than a coincidence. MEASURED, not
# picked: scanning the live 143,555,072-token train split against the live frozen probe+heldout at
# several n gave 13 -> 9 hits, 16 -> 0, 20 -> 0, 24 -> 0, 32 -> 0, 48 -> 0, and decoding all nine
# 13-token hits showed the SAME stock academic phrase every time --
# ". The aim of this paper is twofold. First," -- in unrelated documents (the first sits inside a
# paper on ultra-wideband systems). A 13-TOKEN window is only about 9 words, well inside the noise
# floor of academic English; the external decontamination pass that reported "0 leaks" used 13-WORD
# grams, which is roughly 18-20 tokens. Aborting the prep on stock phrasing is how a guard gets
# switched off, so the scan still runs and REPORTS at 13 tokens -- nothing is hidden -- while the
# abort fires on a contiguous run of at least this many, where the live corpus measures zero.
LEAK_ABORT_RUN = 16

_LEAK_BASE = 1_000_003           # odd multiplier for the mod-2**64 rolling hash
_LEAK_CHUNK = 1 << 23            # 8.4M token positions per vectorised pass (~200 MB of int64)


def _ngram_hashes(flat, n=LEAK_NGRAM):
    """uint64 polynomial hash of every contiguous n-gram of `flat`, vectorised.

    Deliberately mod 2**64 by natural numpy wraparound rather than an explicit modulus: it drops one
    elementwise op per position out of the inner loop, and the hash is never the final word anyway --
    every hit is re-checked against the exact token tuple, so a collision costs one wasted comparison
    and can never produce a false leak report.
    """
    if flat.size < n:
        return np.zeros(0, dtype=np.uint64)
    src = flat.astype(np.uint64, copy=False)
    m = src.size - n + 1
    acc = np.zeros(m, dtype=np.uint64)
    base = np.uint64(_LEAK_BASE)
    with np.errstate(over="ignore"):
        for k in range(n):
            acc = acc * base + src[k:k + m]
    return acc


def frozen_ngram_index(frozen, n=LEAK_NGRAM):
    """Build the yardstick side of the leak check from the FROZEN token arrays themselves.

    Returns (sorted unique hashes, {hash: set of exact n-gram tuples}, n_shingles). Each frozen
    array is shingled SEPARATELY -- probe and heldout are not contiguous text, so an n-gram spanning
    the seam between them would be an artefact, not a real sequence.

    Cheap by construction: probe (512x32) + heldout (1024x32) is 49,152 tokens, so the index is tens
    of thousands of entries however large the corpus gets. The frozen splits are the SMALL side --
    that is what makes a full pass over the train stream affordable.
    """
    exact, hs = {}, []
    for arr in frozen:
        flat = np.asarray(arr).reshape(-1)
        h = _ngram_hashes(flat, n)
        hs.append(h)
        lst = flat.tolist()
        for i, hv in enumerate(h.tolist()):
            exact.setdefault(hv, set()).add(tuple(lst[i:i + n]))
    allh = np.unique(np.concatenate(hs)) if hs else np.zeros(0, dtype=np.uint64)
    return allh, exact, int(allh.size)


def ngram_leak_scan(stream, index, n=LEAK_NGRAM, chunk=_LEAK_CHUNK, stride=1):
    """LAYER 2 OF THE ANTI-LEAK DESIGN (the defensive one): count positions of `stream` that start an
    n-gram also present in the frozen probe/heldout token arrays.

    Layer 1 (bucket_plan) stops the failure mode we KNOW about -- a frozen document re-assigned into
    train by a --train-rows change. It cannot stop the same text arriving in the train split from a
    DIFFERENT source document (the daily feedstock pulls overlapping arXiv/Wikipedia/HN content), and
    it cannot stop a future edit from re-introducing the old behaviour. This scan is the check that
    fails the prep instead of letting either through.

    Method: one uint64 rolling hash per position, then np.searchsorted against the frozen hash array
    (log2(~49k) = 16 comparisons per position, fully vectorised), then EXACT token-tuple comparison
    on the handful of candidates that survive -- so the reported count is a count of real overlaps,
    with hash collisions eliminated rather than tolerated.

    `stride` > 1 checks only every stride-th START POSITION. It still catches any leak of at least
    n + stride - 1 contiguous tokens (some window of every longer run is examined), and misses a leak
    of exactly n tokens with probability 1 - 1/stride. Nothing here chooses a stride on its own: the
    caller passes it, main() only passes one when explicitly asked for, and the value is printed.

    Returns (positions_checked, overlaps, longest_run, examples). `longest_run` is the longest span
    of CONTIGUOUS overlapping tokens, reconstructed from adjacent hit positions: k hits in a row,
    each `stride` apart, means (k-1)*stride + n shared tokens. That is the number that separates a
    real leak from a stock phrase -- see LEAK_ABORT_RUN.
    """
    allh, exact, _ = index
    flat = np.asarray(stream).reshape(-1)
    if allh.size == 0 or flat.size < n:
        return 0, 0, 0, []
    checked, overlaps, examples = 0, 0, []
    step = max(1, int(stride))
    longest, run, prev = 0, 0, None
    for c0 in range(0, flat.size - n + 1, chunk):
        seg = flat[c0:c0 + chunk + n - 1]
        h = _ngram_hashes(seg, n)
        if step > 1:
            offs = np.arange(0, h.size, step, dtype=np.int64)
            h = h[offs]
        else:
            offs = None
        checked += int(h.size)
        pos = np.searchsorted(allh, h)
        pos[pos >= allh.size] = 0
        hit = np.nonzero(allh[pos] == h)[0]
        for j in hit.tolist():                    # candidates only -- exact check kills collisions
            i = int(offs[j]) if offs is not None else j
            gram = tuple(seg[i:i + n].tolist())
            if gram in exact.get(int(h[j]), ()):
                overlaps += 1
                at = c0 + i
                run = run + 1 if prev is not None and at - prev == step else 1
                prev = at
                longest = max(longest, (run - 1) * step + n)
                if len(examples) < 3:
                    examples.append((at, gram))
    return checked, overlaps, longest, examples


def assert_no_frozen_leak(by_split, splits, domain, stride=1, log=print):
    """Run layer 2 over every miner-facing split and ABORT the prep on a real overlap.

    "Real" means a contiguous run of LEAK_ABORT_RUN tokens, not a single 13-token window -- read that
    constant's comment for the measurement that set the threshold. Sub-threshold coincidences are
    still counted and printed; nothing is swallowed.

    A silent guard is indistinguishable from a missing one, so this always prints what it checked
    (shingle count, positions scanned, longest run, wall time) whether or not it finds anything.
    Returns the overlap COUNT (0 when clean), having raised SystemExit if any run hit the threshold.
    """
    frozen = [by_split[name] for name, _ in splits if name in COORD_ONLY and name in by_split]
    shipped = [(name, by_split[name]) for name, _ in splits
               if name not in COORD_ONLY and name in by_split]
    if not frozen or not shipped:
        log("   leak check: SKIPPED (need both frozen and miner-facing splits present)")
        return 0
    t0 = time.time()
    index = frozen_ngram_index(frozen)
    total_checked, total_over, worst, first = 0, 0, 0, []
    for name, arr in shipped:
        checked, over, longest, ex = ngram_leak_scan(arr, index, stride=stride)
        total_checked += checked
        total_over += over
        worst = max(worst, longest)
        if over and not first:
            first = [(name,) + e for e in ex]
        log("   leak check: %-5s %11d positions, %d overlapping %d-gram(s), longest run %d tokens"
            % (name, checked, over, LEAK_NGRAM, longest))
    dt = time.time() - t0
    mode = "FULL PASS" if stride == 1 else ("SAMPLED 1-in-%d start positions" % stride)
    log("   leak check: %d frozen %d-gram shingles from %s | %s | %d positions in %.1fs"
        % (index[2], LEAK_NGRAM,
           "+".join(name for name, _ in splits if name in COORD_ONLY and name in by_split),
           mode, total_checked, dt))
    if worst >= LEAK_ABORT_RUN:
        raise SystemExit(
            "domain %s: CONTENT LEAK -- %d overlapping %d-gram position(s) in the miner-facing\n"
            "splits, longest contiguous run %d tokens (abort threshold %d). The goal metric is\n"
            "meaningless while this holds: the miner would be graded on text it trained on.\n"
            "First hit: %s\n"
            "Fix the CORPUS (drop the offending documents at source), never the check."
            % (domain, total_over, LEAK_NGRAM, worst, LEAK_ABORT_RUN, first[:1]))
    if total_over:
        # Below the abort threshold: real, reported, and explained -- never swallowed.
        log("   leak check: %d %d-token window(s) coincide, longest run %d tokens < abort threshold "
            "%d -- stock phrasing, not a leak (see LEAK_ABORT_RUN for the measurement). First: %s"
            % (total_over, LEAK_NGRAM, worst, LEAK_ABORT_RUN, first[:1]))
    else:
        log("   leak check: OK (0 overlaps -- no miner-facing token %d-gram appears in probe/heldout)"
            % LEAK_NGRAM)
    return total_over


def build_domain(tok, text, seq, miner_dir, coord_dir, domain, vocab_size, splits=SPLITS,
                 freeze_coord=True, leak_stride=1, ref_splits=None):
    """Tokenize + write one domain's splits. Documents are lines (the corpus roll files are
    line-per-document), assigned to splits by content hash and tokenized SEPARATELY, so no row can
    straddle two splits -- the old flat-stream chunking let a held-out row carry the tail of a
    training document even within a single prep."""
    docs = [ln for ln in text.splitlines() if ln.strip()]
    if not docs:
        raise SystemExit("domain %s: corpus has no non-empty lines" % domain)
    by_split = {}
    for d in docs:
        by_split.setdefault(split_of_document(d, splits, SPLIT_BUCKETS, ref_splits), []).append(d)

    written, parts, used, carved = [], [], 0, {}
    for name, n in splits:
        # probe + heldout are the GOAL METRIC's yardstick. Freeze them: once written they are never
        # regenerated, so held-out CE stays comparable across every corpus refresh for the life of a
        # campaign. Without this, a daily refresh silently REPLACES the held-out set -- documents
        # keep their split, but `mine` is in roll order (newest first) and we take the first n rows,
        # so today's documents push yesterday's out. Day-to-day CE would then compare two different
        # exams and "the model got smarter" would be unfalsifiable. Pass --refresh-coord-splits to
        # deliberately re-cut them, which STARTS A NEW CAMPAIGN's baseline.
        dst = coord_dir if name in COORD_ONLY else miner_dir
        path = os.path.join(dst, "ids_%s_%s.npy" % (domain, name))
        if name in COORD_ONLY and freeze_coord and os.path.exists(path):
            part = np.load(path)
            written.append((name, part.shape, path + "  [FROZEN: kept, not re-cut]"))
            parts.append(part)
            carved[name] = part
            used += int(part.shape[0])
            continue
        mine = by_split.get(name, [])
        # CHUNKED tokenization, and STOP once we have enough.
        #
        # The single-shot form -- tok("\n".join(mine)) -- died with
        # "memory allocation of 9126805520 bytes failed" on a 1 GB corpus, on a box with 45 GB free.
        # Three costs stack in that one line: a ~700 MB joined string, a Python list of ~143M int
        # objects (~28 bytes each), and then a numpy copy of the same data. Peak far exceeds the 9.1 GB
        # the allocator happened to ask for last.
        #
        # Only `n * seq` tokens are ever USED (see the slice below), and that was already true of the
        # old code, so tokenizing the whole corpus was pure waste. Chunking + early exit bounds memory
        # to one batch and makes prep time proportional to what is needed rather than to corpus size.
        #
        # Boundary note: batches are joined separately, so the few tokens at each batch seam differ
        # from a single-shot tokenization. Harmless -- train/val are regenerated artifacts, not frozen
        # ones, and probe/heldout never reach this path when freeze_coord is on (see above).
        need = n * seq
        chunks, have_tok = [], 0
        for b0 in range(0, len(mine), TOK_DOC_BATCH):
            batch = mine[b0:b0 + TOK_DOC_BATCH]
            bids = tok("\n".join(batch), add_special_tokens=False)["input_ids"]
            chunks.append(np.asarray(bids, dtype=np.int64))
            have_tok += len(bids)
            if have_tok >= need:
                break
        ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
        have = len(ids) // seq
        if have < n:
            raise SystemExit(
                "domain %s split %s: only %d rows of %d tokens available, need %d. Splits are "
                "content-addressed (a document keeps its split forever), so the fix is MORE TEXT --"
                " never re-carve to make the counts fit, that is what lets a document change split."
                % (domain, name, have, seq, n))
        part = np.asarray(ids[: n * seq], dtype=np.int64).reshape(n, seq)
        if int(part.max()) >= vocab_size:
            raise SystemExit("domain %s: token id %d >= vocab_size %d"
                             % (domain, part.max(), vocab_size))
        # (dst/path resolved above; probe + heldout land in the coordinator-only dir, never
        # shipped to a miner -- F1.)
        np.save(path, part)
        written.append((name, part.shape, path))
        parts.append(part)
        carved[name] = part
        used += n

    # LAYER 2. Splits are carved; now prove the miner-facing text does not repeat the yardstick.
    # On failure, delete the miner-facing files this call just wrote: an aborted prep must not leave
    # a contaminated ids_*.npy on disk where the next glm_publish_data.py run would ship it.
    try:
        assert_no_frozen_leak(carved, splits, domain, stride=leak_stride)
    except SystemExit:
        for name, _shape, p in written:
            if name not in COORD_ONLY and os.path.exists(p):
                os.remove(p)
                print("   leak check: removed contaminated %s" % p)
        raise
    return np.concatenate(parts, axis=0), written, used


def _sha256_file(path, chunk=1 << 20):
    """Streamed sha256 (1 MiB chunks) so a multi-hundred-MB ids file is fingerprinted without ever
    being read into RAM whole -- the same shape publish_corpus_to_hf._sha256_file uses."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_data_manifest(miner_dir, domains, seq, train_rows):
    """Fingerprint the MINER-FACING id files into <miner_dir>/data_manifest.json for corpus-over-WAN
    auto sync (W5). The manifest is the contract both halves of the transport read: the publisher
    (tools/glm_publish_data.py) content-addresses each file by its sha256, and the contributor
    auto-sync sha-verifies every download -- ending the 12h-soak hand-copy of ids_*.npy.

    F1 IS STRUCTURAL, NOT A CHECK. This function is handed ``miner_dir`` and NOTHING about
    ``<out>/coord``, so it is physically incapable of enumerating probe/heldout -- the coordinator's
    secret gate pool can never leak into a manifest even if a future edit is careless. Only
    ``ids_*.npy`` are listed (the shipped data; the manifest itself and any stray file are skipped),
    sorted so the manifest is deterministic and diff-stable across re-runs.
    """
    files = {}
    for name in sorted(os.listdir(miner_dir)):
        if not (name.startswith("ids_") and name.endswith(".npy")):
            continue
        p = os.path.join(miner_dir, name)
        if not os.path.isfile(p):
            continue
        files[name] = {"sha256": _sha256_file(p), "size": os.path.getsize(p)}
    manifest = {"v": 1, "domains": list(domains), "seq": int(seq),
                "train_rows": int(train_rows), "files": files}
    path = os.path.join(miner_dir, "data_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=DEFAULT_CORPUS)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOK)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--domains", default="code,gutenberg", help="comma-separated corpus basenames")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--max-chars", type=int, default=6_000_000,
                    help="chars read per domain (bounds tokenizer time). NOTE: this SILENTLY TRUNCATES "
                         "-- a 1 GB corpus with the 6M default reads only its first 6 MB and the run "
                         "looks successful. Raise it past the corpus size for any large prep.")
    ap.add_argument("--refresh-coord-splits", dest="refresh_coord", action="store_true",
                    help="RE-CUT the frozen probe/heldout pools. This moves the goal metric's "
                         "yardstick, so held-out CE before and after are NOT comparable -- use it "
                         "only when deliberately starting a new campaign baseline.")
    ap.add_argument("--train-rows", type=int, default=4096,
                    help="rows in the TRAIN split per domain (raise for long soaks; needs enough "
                         "--max-chars to supply train+val+probe+heldout rows). Since 2026-07-26 this "
                         "CANNOT move a document out of probe/heldout -- see bucket_plan().")
    ap.add_argument("--leak-check-stride", type=int, default=1,
                    help="check every Nth start position in the %d-token leak scan instead of every "
                         "one. 1 (default) is a full pass and is what you want; a stride still "
                         "catches any leak of >= N+%d contiguous tokens but can miss a bare "
                         "%d-token one. The mode and the value are printed either way -- this tool "
                         "never samples silently." % (LEAK_NGRAM, LEAK_NGRAM - 1, LEAK_NGRAM))
    args = ap.parse_args()
    if args.leak_check_stride < 1:
        raise SystemExit("--leak-check-stride must be >= 1")
    splits = _splits_with_train(args.train_rows)

    # MINER-FACING vs COORDINATOR-ONLY split of the output tree (F1). Only <out>/miner is ever
    # shipped to a miner; <out>/coord (probe + heldout) stays on the coordinator box.
    miner_dir = os.path.join(args.out_dir, "miner")
    coord_dir = os.path.join(args.out_dir, "coord")
    os.makedirs(miner_dir, exist_ok=True)
    os.makedirs(coord_dir, exist_ok=True)
    from transformers import AutoConfig, AutoTokenizer          # noqa: E402  (after offline env)

    cfg = AutoConfig.from_pretrained(args.tokenizer_dir, local_files_only=True,
                                     trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    print("tokenizer vocab_size=%d  config vocab_size=%d  seq=%d"
          % (len(tok), cfg.vocab_size, args.seq))

    for domain in [d.strip() for d in args.domains.split(",") if d.strip()]:
        src = os.path.join(args.corpus_dir, domain + ".txt")
        if not os.path.isfile(src):
            raise SystemExit("missing corpus file: " + src)
        with open(src, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(args.max_chars)
        print("\n[%s] %s -> %d chars" % (domain, src, len(text)))

        # LAYER 1, stated before the carve so the log shows what protected this run. The frozen
        # splits' buckets must be IDENTICAL at every --train-rows; print them next to the reference
        # plan so a reader can see that with two numbers instead of taking it on faith.
        plan = bucket_plan(splits, SPLIT_BUCKETS, SPLITS)
        refmap = {name: (lo, hi) for name, lo, hi in bucket_plan(SPLITS, SPLIT_BUCKETS, SPLITS)}
        prev_hi = 0
        for name, lo, hi in plan:
            # partition check: the ranges must tile [0, buckets) with no gap and no overlap, so no
            # document can be claimed by two splits or by none.
            assert lo == prev_hi, "bucket gap/overlap in %s before %s" % (domain, name)
            prev_hi = hi
            tag = "FROZEN, train-rows-independent" if name in COORD_ONLY else "resized by --train-rows"
            same = "" if name not in COORD_ONLY else (
                "  == ref %s" % (refmap[name],) if refmap[name] == (lo, hi) else "  !! MOVED vs ref")
            print("   buckets %-8s [%7d, %7d)  %6.2f%%  [%s]%s"
                  % (name, lo, hi, 100.0 * (hi - lo) / SPLIT_BUCKETS, tag, same))
        assert prev_hi == SPLIT_BUCKETS, "bucket plan does not cover the hash space in %s" % domain
        for name, lo, hi in plan:
            assert name not in COORD_ONLY or refmap[name] == (lo, hi), (
                "FROZEN SPLIT MOVED: %s got %s but the reference plan says %s -- --train-rows must "
                "not be able to do this" % (name, (lo, hi), refmap[name]))
        print("   buckets: partition OK (tile [0,%d), frozen ranges == reference)" % SPLIT_BUCKETS)

        arr, written, used = build_domain(tok, text, args.seq, miner_dir, coord_dir, domain,
                                          cfg.vocab_size, splits=splits,
                                          freeze_coord=not args.refresh_coord,
                                          leak_stride=args.leak_check_stride,
                                          ref_splits=SPLITS)
        print("   -> %d rows of %d tokens (used %d rows)" % (arr.shape[0], args.seq, used))
        for name, shape, path in written:
            tag = "COORD-ONLY" if name in COORD_ONLY else "miner-facing"
            print("   %-8s %-12s [%-12s] %s" % (name, str(shape), tag, path))
    # DATA MANIFEST (W5 corpus-over-WAN): now that every domain's id files are on disk, fingerprint
    # the MINER-FACING dir so a publisher can content-address them and a contributor can sha-verify
    # each download. F1 STRUCTURAL: write_data_manifest is handed miner_dir ALONE (never coord_dir),
    # so it physically cannot enumerate probe/heldout -- the secret split can never appear in a
    # manifest that ships to a miner.
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    manifest, manifest_path = write_data_manifest(miner_dir, domains, args.seq, args.train_rows)
    print("\nDATA MANIFEST: %s  (%d miner files fingerprinted, v%d)"
          % (manifest_path, len(manifest["files"]), manifest["v"]))
    print("SHIP ONLY: %s  (train + val + data_manifest.json -- miner-facing)" % miner_dir)
    print("NEVER SHIP: %s  (probe + heldout -- coordinator's SECRET gate pool + goal metric; a miner"
          "\n            that obtains these can train on the whole pool and defeat the gate)" % coord_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
