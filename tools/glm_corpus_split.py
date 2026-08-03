#!/usr/bin/env python3
"""Split the monolithic training corpus into SHUFFLED parts a miner can fetch one at a time.

WHY. A joining miner downloads all 14.97 GiB of ids_daily_train.npy and then reads 0.25% of it
(10,000 steps x batch 16 = 160,000 of 62,805,344 rows = ~41 MB). It also re-hashes the whole 16 GB
at every startup before it will train. An external volunteer with 20.73 GiB free could not complete
the install at all (issue #71). Parts fix all three: fetch one 256 MiB part, hash one 256 MiB part,
delete it when you rotate to the next.

WHY SHUFFLE RATHER THAN JUST SLICE. The corpus is written source-ordered -- glm_wan_prep_data.py:513
concatenates sources with no shuffle anywhere in the prep path -- and that ordering is MEASURABLE:
sampling 60,000 rows from each of five 512 MiB regions gives TV 0.0727 between regions against a
0.0570 within-region noise floor (ratio 1.274), with a clean distance gradient away from part 0
(scratchpad/corpus_part_probe.py). So a naive contiguous slice is a different data distribution, not
just a different file. Permuting rows at publish time removes that entirely: every part becomes a
uniform random sample of the corpus, so which part a miner gets is a scheduling detail.

Permuting is SAFE because row order carries no training signal. Each row is an independent 32-token
sequence, the trainer feeds every row as its own sequence (sharddiloco_glm_expert.py:750,
`model(input_ids=ids, labels=ids)` -- no cross-row context) and already draws rows uniformly at
random (:747). Nothing downstream depends on which row sits next to which.

METHOD. One sequential pass over the source. A permutation decides each row's destination part; rows
are appended to their part in source order, which is fine because assignment is what must be random
-- order WITHIN a part is never observable to a trainer that draws uniformly inside it. Peak RAM is
one int64 permutation (~502 MB for 62.8M rows), transient, plus one chunk buffer.

VERIFICATION is permutation-invariant by construction: a sha256 cannot prove a shuffle lossless, so
this compares the total row count and the full token histogram of the parts against the source. That
catches a dropped, duplicated or corrupted row, which is exactly what a reshuffle can get wrong.

  python tools/glm_corpus_split.py --src D:/glm_daily/miner/ids_daily_train.npy --out-dir C:/glm_parts
  python tools/glm_corpus_split.py --src ... --out-dir ... --part-rows 2097152 --skip-verify
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

DEFAULT_PART_ROWS = 1 << 20          # 1,048,576 rows x 256 B = 256 MiB of payload
CHUNK_ROWS = 1 << 18                 # sequential read granularity (64 MiB)
VOCAB = 154880


def _log(msg):
    """Unbuffered progress. The first real run died silently and its buffered stdout went with it,
    so a multi-minute job could not say how far it had got or why it stopped."""
    print(msg, flush=True)


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def part_name(src_basename, idx):
    """ids_daily_train.npy + 7 -> ids_daily_train.p007.npy

    The `.pNNN` suffix goes AFTER the split word so the miner's security allowlist keeps working by
    construction: _ALLOWED_DATA_RE matches ids_<domain>_(train|val) BEFORE the optional part suffix,
    so no part name can ever spell the coordinator's secret probe/heldout splits.
    """
    stem = src_basename[:-4] if src_basename.endswith(".npy") else src_basename
    return "%s.p%03d.npy" % (stem, idx)


def token_histogram(arr, chunk=CHUNK_ROWS, vocab=VOCAB, label="", log=_log):
    """Full token histogram, streamed. Permutation-invariant: identical for any row ordering."""
    h = np.zeros(vocab, dtype=np.int64)
    n = arr.shape[0]
    t0 = time.time()
    for c0 in range(0, n, chunk):
        h += np.bincount(np.asarray(arr[c0:c0 + chunk]).reshape(-1), minlength=vocab)
        if label and c0 and (c0 // chunk) % 50 == 0:
            log("    %s %5.1f%% (%.0fs)" % (label, 100.0 * c0 / n, time.time() - t0))
    return h


def split(src, out_dir, part_rows=DEFAULT_PART_ROWS, seed=20260803, verify=True, log=_log):
    ids = np.load(src, mmap_mode="r")
    n, seq = int(ids.shape[0]), int(ids.shape[1])
    nparts = (n + part_rows - 1) // part_rows
    base = os.path.basename(src)
    os.makedirs(out_dir, exist_ok=True)
    log("src   : %s  rows=%d seq=%d dtype=%s (%d B)" % (src, n, seq, ids.dtype, os.path.getsize(src)))
    log("out   : %s  part_rows=%d -> %d parts (last holds %d)"
        % (out_dir, part_rows, nparts, n - (nparts - 1) * part_rows))

    # Destination part of every row. Built from a permutation so part sizes are exact and the
    # assignment is uniform; the int64 permutation is released before the copy loop starts.
    t0 = time.time()
    perm = np.random.default_rng(seed).permutation(n)
    part_of_row = np.empty(n, dtype=np.int16 if nparts <= 32767 else np.int32)
    part_of_row[perm] = (np.arange(n, dtype=np.int64) // part_rows).astype(part_of_row.dtype)
    del perm
    log("perm  : assigned %d rows to %d parts in %.0fs" % (n, nparts, time.time() - t0))

    sizes = np.bincount(part_of_row, minlength=nparts)

    # PLAIN FILE HANDLES, NOT MEMMAPS. The first version of this opened all `nparts` outputs with
    # np.lib.format.open_memmap(mode="w+"), which asks Windows to back nparts * part_bytes of
    # WRITABLE mapped pages at once -- 16.1 GiB for the real corpus. MEASURED on this box during the
    # first real run: commit free was 17.8 GiB, the process died with no traceback, and every part
    # was left at its preallocated size with nothing written. Same commit-charge family as the
    # os error 1455 in memory judge-ram-cache-mmap-1455. Streaming writes make peak commit the
    # permutation (~600 MB) instead of the whole output, and each part still receives large
    # sequential appends (~1.1 MB per chunk at the default CHUNK_ROWS).
    import numpy.lib.format as npf
    paths, cursors = [], [0] * nparts
    fhs = []
    try:
        for p in range(nparts):
            path = os.path.join(out_dir, part_name(base, p))
            paths.append(path)
            fh = open(path, "wb")
            npf.write_array_header_1_0(fh, {"descr": npf.dtype_to_descr(ids.dtype),
                                            "fortran_order": False,
                                            "shape": (int(sizes[p]), seq)})
            fhs.append(fh)

        t0 = time.time()
        for c0 in range(0, n, CHUNK_ROWS):
            c1 = min(c0 + CHUNK_ROWS, n)
            block = np.asarray(ids[c0:c1])
            owners = part_of_row[c0:c1]
            for p in np.unique(owners):
                sel = block[owners == p]
                fhs[p].write(np.ascontiguousarray(sel).tobytes())
                cursors[p] += int(sel.shape[0])
            if (c0 // CHUNK_ROWS) % 25 == 0:
                log("  copy %5.1f%% (%.0fs)" % (100.0 * c0 / n, time.time() - t0))
    finally:
        for fh in fhs:
            try:
                fh.close()
            except OSError:
                pass
    bad = [(p, cursors[p], int(sizes[p])) for p in range(nparts) if cursors[p] != int(sizes[p])]
    if bad:
        raise SystemExit("SPLIT BUG: part fill mismatch %r" % bad[:5])
    log("copy  : %d parts written in %.0fs" % (nparts, time.time() - t0))

    manifest = {"v": 1, "source": base, "source_rows": n, "seq": seq,
                "part_rows": part_rows, "parts": nparts, "seed": seed,
                "dtype": str(ids.dtype), "files": {}}
    for p, path in enumerate(paths):
        manifest["files"][os.path.basename(path)] = {
            "sha256": _sha256_file(path), "size": os.path.getsize(path),
            "rows": int(sizes[p]), "index": p}

    if verify:
        log("verify: streaming token histograms (permutation-invariant; a sha cannot do this)")
        hs = token_histogram(ids, label="source", log=log)
        hp = np.zeros(VOCAB, dtype=np.int64)
        rows = 0
        for p, path in enumerate(paths):
            a = np.load(path, mmap_mode="r")
            hp += token_histogram(a, log=log)
            rows += int(a.shape[0])
            del a
        ok_rows = rows == n
        ok_hist = bool((hs == hp).all())
        log("verify: rows %d vs %d -> %s" % (rows, n, "OK" if ok_rows else "MISMATCH"))
        if not ok_hist:
            d = int(np.abs(hs.astype(np.int64) - hp.astype(np.int64)).sum())
            log("verify: token histogram MISMATCH, total abs diff %d" % d)
        else:
            log("verify: token histogram identical across %d distinct ids (%d tokens) -> OK"
                % (int((hs > 0).sum()), int(hs.sum())))
        manifest["verified"] = {"rows_match": ok_rows, "histogram_match": ok_hist,
                                "total_tokens": int(hs.sum())}
        if not (ok_rows and ok_hist):
            raise SystemExit("VERIFY FAILED: the split is NOT lossless -- parts left in place for "
                             "inspection, do NOT publish them")
    else:
        manifest["verified"] = None
        log("verify: SKIPPED (--skip-verify). Do not publish an unverified split.")

    mpath = os.path.join(out_dir, "parts_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    log("wrote : %s" % mpath)
    total = sum(v["size"] for v in manifest["files"].values())
    log("done  : %d parts, %.2f GiB total, largest %.0f MiB"
        % (nparts, total / 2 ** 30, max(v["size"] for v in manifest["files"].values()) / 2 ** 20))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="monolithic ids_<domain>_train.npy")
    ap.add_argument("--out-dir", required=True, help="directory to write parts + parts_manifest.json")
    ap.add_argument("--part-rows", type=int, default=DEFAULT_PART_ROWS,
                    help="rows per part (default %(default)d = 256 MiB at seq 32 int64)")
    ap.add_argument("--seed", type=int, default=20260803, help="permutation seed (recorded)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the histogram proof (two extra streaming passes)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args()

    ids = np.load(a.src, mmap_mode="r")
    n = int(ids.shape[0])
    nparts = (n + a.part_rows - 1) // a.part_rows
    need = os.path.getsize(a.src)
    print("rows=%d  part_rows=%d  parts=%d  output needs ~%.2f GiB free in %s"
          % (n, a.part_rows, nparts, need / 2 ** 30, a.out_dir))
    if a.dry_run:
        return 0
    split(a.src, a.out_dir, a.part_rows, a.seed, verify=not a.skip_verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
