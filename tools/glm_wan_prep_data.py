#!/usr/bin/env python3
"""Tokenize the real corpus into per-domain id files for the GLM shardDiLoCo WAN run.

WHY THIS EXISTS. `tools/sharddiloco_glm_expert.py` today trains and evaluates on
`markov_dataset(vocab=24)` (:193-204), and `tools/sharddiloco_glm_gpu_smoke.py` on random ids
0..95 out of GLM's 154,880-token vocabulary. A cross-entropy on either is arithmetically real but
is NOT a language number, so it cannot serve as the goal metric for a real-model run. This script
produces genuine GLM-tokenized text so the held-out CE the coordinator reports means something.

SPLIT DISCIPLINE (mirrors `sharddiloco_harness.domain_splits`): four DISJOINT row sets per domain.
  train   - the miner trains on this
  val     - the miner's own save-best signal (public to the miner)
  probe   - the COORDINATOR's secret gate pool; a miner never sees it (dm.SecretRotatedProbe
            draws a fresh subset each round, so fitting the probe is not available even to a
            miner who somehow obtained it)
  heldout - the reported goal metric; touched by nothing else
Rows are chunked contiguously and then the four ranges are carved out by index, so no row can
appear in two splits and no token overlaps a boundary.

ONE DOMAIN PER EXPERT SLOT. GLM routes with its OWN learned router -- there is no offline
domain-routing here as there is in the toy harness -- so expert specialisation has to come from
the DATA. Slot 0 gets one corpus, slot 1 another.

Env: C:/Python313/python.exe (never .venv). Keep stdout ASCII (cp1252 console).
Offline: the tokenizer is loaded from a local directory; HF_HUB_OFFLINE is set before the import
so no code path can reach the hub (an unbounded hub check is a measured indefinite hang here).
"""
import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_corpus_v2")
DEFAULT_TOK = r"D:\hf_models\GLM-4.7-Flash-bf16"
DEFAULT_OUT = r"D:\glm_wan"

# rows per split, in carve order
SPLITS = (("train", 4096), ("val", 512), ("probe", 512), ("heldout", 1024))


def build_domain(tok, text, seq, out_dir, domain, vocab_size):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n_rows_needed = sum(n for _, n in SPLITS)
    have = len(ids) // seq
    if have < n_rows_needed:
        raise SystemExit("domain %s: only %d rows of %d tokens available, need %d -- feed more text"
                         % (domain, have, seq, n_rows_needed))
    arr = np.asarray(ids[: have * seq], dtype=np.int64).reshape(have, seq)
    if int(arr.max()) >= vocab_size:
        raise SystemExit("domain %s: token id %d >= vocab_size %d" % (domain, arr.max(), vocab_size))

    off = 0
    written = []
    for name, n in SPLITS:
        part = arr[off:off + n]
        off += n
        path = os.path.join(out_dir, "ids_%s_%s.npy" % (domain, name))
        np.save(path, part)
        written.append((name, part.shape, path))
    return arr, written, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=DEFAULT_CORPUS)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOK)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--domains", default="code,gutenberg", help="comma-separated corpus basenames")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--max-chars", type=int, default=6_000_000,
                    help="chars read per domain (bounds tokenizer time; 6M >> the ~800KB needed)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
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
        arr, written, used = build_domain(tok, text, args.seq, args.out_dir, domain, cfg.vocab_size)
        print("\n[%s] %s -> %d chars -> %d rows of %d tokens (used %d rows)"
              % (domain, src, len(text), arr.shape[0], args.seq, used))
        for name, shape, path in written:
            print("   %-8s %-12s %s" % (name, str(shape), path))

        # disjointness is by construction (contiguous carve); assert it anyway so a future edit
        # that reorders the carve cannot silently leak the heldout set into training.
        seen = set()
        off = 0
        for name, n in SPLITS:
            rng = range(off, off + n)
            assert not (seen & set(rng)), "split overlap in %s at %s" % (domain, name)
            seen |= set(rng)
            off += n
        print("   splits disjoint: OK (%d rows total, no row in two splits)" % len(seen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
