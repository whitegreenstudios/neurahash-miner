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
import hashlib
import json
import os
import sys

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


SPLIT_BUCKETS = 8192            # hash granularity; only affects how evenly documents distribute


def split_of_document(doc, splits=SPLITS, buckets=SPLIT_BUCKETS):
    """Which split a document belongs to -- a pure function of its CONTENT, forever.

    THIS IS THE INVARIANT THE GOAL METRIC RESTS ON. Held-out cross-entropy only means
    "generalization" if the model never trained on the held-out text. That has to hold across
    corpus refreshes, not just within one prep.

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

    Assigning by content hash makes split membership permanent -- growing or reordering the corpus
    adds documents to each split but never MOVES one between splits -- so the invariant holds by
    construction rather than by an accident of roll ordering that nothing tested. Ratios follow the
    configured row counts, so the mix is unchanged.
    """
    total = float(sum(n for _, n in splits))
    h = int(hashlib.sha256(doc.encode("utf-8", "replace")).hexdigest()[:8], 16) % buckets
    edge = 0.0
    for name, n in splits:
        edge += n / total
        if h < edge * buckets:
            return name
    return splits[-1][0]


def build_domain(tok, text, seq, miner_dir, coord_dir, domain, vocab_size, splits=SPLITS,
                 freeze_coord=True):
    """Tokenize + write one domain's splits. Documents are lines (the corpus roll files are
    line-per-document), assigned to splits by content hash and tokenized SEPARATELY, so no row can
    straddle two splits -- the old flat-stream chunking let a held-out row carry the tail of a
    training document even within a single prep."""
    docs = [ln for ln in text.splitlines() if ln.strip()]
    if not docs:
        raise SystemExit("domain %s: corpus has no non-empty lines" % domain)
    by_split = {}
    for d in docs:
        by_split.setdefault(split_of_document(d, splits), []).append(d)

    written, parts, used = [], [], 0
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
            used += int(part.shape[0])
            continue
        mine = by_split.get(name, [])
        ids = tok("\n".join(mine), add_special_tokens=False)["input_ids"] if mine else []
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
        used += n
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
                    help="chars read per domain (bounds tokenizer time; 6M >> the ~800KB needed)")
    ap.add_argument("--refresh-coord-splits", dest="refresh_coord", action="store_true",
                    help="RE-CUT the frozen probe/heldout pools. This moves the goal metric's "
                         "yardstick, so held-out CE before and after are NOT comparable -- use it "
                         "only when deliberately starting a new campaign baseline.")
    ap.add_argument("--train-rows", type=int, default=4096,
                    help="rows in the TRAIN split per domain (raise for long soaks; needs enough "
                         "--max-chars to supply train+val+probe+heldout rows)")
    args = ap.parse_args()
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
        arr, written, used = build_domain(tok, text, args.seq, miner_dir, coord_dir, domain,
                                          cfg.vocab_size, splits=splits,
                                          freeze_coord=not args.refresh_coord)
        print("\n[%s] %s -> %d chars -> %d rows of %d tokens (used %d rows)"
              % (domain, src, len(text), arr.shape[0], args.seq, used))
        for name, shape, path in written:
            tag = "COORD-ONLY" if name in COORD_ONLY else "miner-facing"
            print("   %-8s %-12s [%-12s] %s" % (name, str(shape), tag, path))

        # disjointness is by construction (contiguous carve); assert it anyway so a future edit
        # that reorders the carve cannot silently leak the heldout set into training.
        seen = set()
        off = 0
        for name, n in splits:
            rng = range(off, off + n)
            assert not (seen & set(rng)), "split overlap in %s at %s" % (domain, name)
            seen |= set(rng)
            off += n
        print("   splits disjoint: OK (%d rows total, no row in two splits)" % len(seen))
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
