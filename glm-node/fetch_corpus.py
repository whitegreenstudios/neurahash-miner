#!/usr/bin/env python3
"""glm-node: put the training corpus in place, and PROVE it is the right file.

The node trains on exactly one file, pinned by its content:

    e5_scimc_train.jsonl   sha256 f9b00e91e50d9eea668e6e520d1b0ba5e1228fa5911de7cb4b4b35520fbf85da
                           19,792 rows (one JSON object per line), 10,508,492 bytes

We do NOT redistribute that file: it is made from public datasets whose licences belong to their owners
(see LICENCES below). This script REBUILDS it byte for byte from those datasets, which you download from
huggingface.co at pinned commits -- or it takes a copy your operator gives you. Every mode ends with the
same check on the file as written to disk: sha256, row count and byte count. A file that fails it is
deleted and the exit code is non-zero; a file that passes is renamed into place atomically, so a crash or
a Ctrl-C never leaves a half-written corpus where the node looks for it.

    python glm-node/fetch_corpus.py --rebuild                       # download the pinned sources (~14 MB), rebuild
                                                                    #   (also what it does when no mode is given)
    python glm-node/fetch_corpus.py --rebuild --offline-cache DIR   # rebuild from an existing HF hub cache, no network
    python glm-node/fetch_corpus.py --from-file PATH                # take a copy your operator gave you
    python glm-node/fetch_corpus.py --from-url URL                  # download a copy your operator gave you
    python glm-node/fetch_corpus.py --print-manifest                # the pinned facts, as JSON

Paths:
    --home DIR           the node's data dir: default env NEURAHASH_GLM_HOME, else ~/glmnode
    --out PATH           where the corpus goes: default <home>/corpus/e5_scimc_train.jsonl
    --config-dir DIR     (--rebuild) the GLM-4.7-Flash tokenizer: default <home>/base/config, which
                         `python glm-node/build_packs.py --base-only` fills with tokenizer.json,
                         tokenizer_config.json and config.json from zai-org/GLM-4.7-Flash at revision
                         7dd20894a642a0aa287e9827cb1a1f7f91386b67. Every row stores its token ids, so the
                         exact tokenizer is part of the file's identity.
    --cache-dir DIR      (--rebuild) where huggingface_hub keeps the downloads: default its own cache
                         (HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub)
    --offline-cache DIR  (--rebuild) read the pinned files from an existing hub cache instead: the dir that
                         holds datasets--ORG--NAME/ folders, or its parent that holds hub/. Nothing is
                         written there and nothing is fetched.

Needs Python 3.10+. --from-file and --from-url use the standard library only. --rebuild also needs
pyarrow (reads the parquet files; requirements-node.txt pins pyarrow==22.0.0, the
version the verified rebuild used) plus huggingface_hub and transformers (both in requirements-node.txt).
Network: --rebuild contacts huggingface.co only; --from-url contacts only the URL you give it;
--offline-cache and --from-file use no network at all.

What --rebuild does: the original build of 2026-08-15, replayed step by step (each function below names
the part of the original it replays):
  1. POOL. The TRAIN splits of ARC-Easy, ARC-Challenge, OpenBookQA (main) and SciQ, mapped to one
     multiple-choice schema. SciQ ships the gold answer + 3 distractors; their order comes from
     random.Random(20260815), and the 36 SciQ rows whose 4 options are not distinct are dropped:
     2,251 + 1,119 + 4,957 + 11,643 = 19,970 items.
  2. PURGE. An item is removed if its normalised question equals, or shares any normalised 8-gram with, a
     text of ARC-Easy test, ARC-Challenge test, HellaSwag validation, WinoGrande-XL validation, BoolQ
     validation or PIQA validation: 178 removed, 19,792 kept. A verbatim ARC-Easy test item is planted
     first and must be caught (a scanner that cannot fire proves nothing). All six families are required:
     a missing one would change what is removed, so an unavailable source FAILS the run, never skips it.
  3. FORMAT. Each item becomes the eval's own gold-choice request, "Question: <q>\\nAnswer:" + " <gold>",
     tokenized by the GLM-4.7-Flash tokenizer and written as json.dumps(row) + "\\n", UTF-8, LF.
Every source file is pinned by repo, commit AND sha256, and every intermediate count is compared with the
original run, so a failure names the step that diverged instead of only saying "sha256 mismatch".

LICENCES, as recorded in each dataset's card at the pinned revision. You download the data from its
owners and their terms apply to you; read them before use.
  text that ends up IN the corpus:
    allenai/ai2_arc      cc-by-sa-4.0
    allenai/openbookqa   "unknown" (that is what its card says)
    allenai/sciq         cc-by-nc-3.0, non-commercial (11,622 of the 19,792 rows come from SciQ)
  used ONLY to decide which items to remove (none of their text is in the corpus):
    Rowan/hellaswag      MIT (stated in the card's text)
    allenai/winogrande   no licence stated in the card
    google/boolq         cc-by-sa-3.0
    baber/piqa           no dataset card at the pinned revision (the upstream ybisk/piqa card says "unknown")

Exit code: 0 a verified corpus is in place; 1 refused or failed (nothing new is left behind); 2 bad
arguments. Console output is ASCII only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request

# huggingface_hub's Xet transfer backend has wedged silently on residential links (measured on our own
# nodes); plain HTTPS does not. The setting is read when huggingface_hub is IMPORTED (transformers imports
# it too), so it is set here, before anything in this process can import either.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")       # keeps the console short and ASCII
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# =========================================================================================== PINNED FACTS
CORPUS = {"name": "e5_scimc_train.jsonl",
          "sha256": "f9b00e91e50d9eea668e6e520d1b0ba5e1228fa5911de7cb4b4b35520fbf85da",
          "rows": 19792,
          "bytes": 10508492}

# The ten dataset files the build reads, each pinned by repo, commit and content (sha256 of the file).
# use "corpus": its text ends up in the output.  use "purge": its text only decides what is removed.
# PIQA: the original first tried ybisk/piqa, which is a loading script datasets 4.x no longer runs, and
# fell back to baber/piqa; this goes straight to the baber/piqa revision the original actually read.
_ARC = "210d026faf9955653af8916fad021475a3f00453"
SOURCES = (
    {"key": "arc_easy_train", "use": "corpus", "repo": "allenai/ai2_arc", "revision": _ARC,
     "path": "ARC-Easy/train-00000-of-00001.parquet", "rows": 2251, "bytes": 330598,
     "sha256": "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5", "licence": "cc-by-sa-4.0"},
    {"key": "arc_challenge_train", "use": "corpus", "repo": "allenai/ai2_arc", "revision": _ARC,
     "path": "ARC-Challenge/train-00000-of-00001.parquet", "rows": 1119, "bytes": 189909,
     "sha256": "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb", "licence": "cc-by-sa-4.0"},
    {"key": "openbookqa_train", "use": "corpus", "repo": "allenai/openbookqa",
     "revision": "388097ea7776314e93a529163e0fea805b8a6454",
     "path": "main/train-00000-of-00001.parquet", "rows": 4957, "bytes": 495845,
     "sha256": "98148f8a54e62eb862346a75192d5fb824d6cbb68f2f59aecd793d39ecb5cd8b", "licence": "unknown"},
    {"key": "sciq_train", "use": "corpus", "repo": "allenai/sciq",
     "revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
     "path": "data/train-00000-of-00001.parquet", "rows": 11679, "bytes": 3993099,
     "sha256": "19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167", "licence": "cc-by-nc-3.0"},
    {"key": "arc_easy_test", "use": "purge", "repo": "allenai/ai2_arc", "revision": _ARC,
     "path": "ARC-Easy/test-00000-of-00001.parquet", "rows": 2376, "bytes": 346257,
     "sha256": "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177", "licence": "cc-by-sa-4.0"},
    {"key": "arc_challenge_test", "use": "purge", "repo": "allenai/ai2_arc", "revision": _ARC,
     "path": "ARC-Challenge/test-00000-of-00001.parquet", "rows": 1172, "bytes": 203808,
     "sha256": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9", "licence": "cc-by-sa-4.0"},
    {"key": "hellaswag_val", "use": "purge", "repo": "Rowan/hellaswag",
     "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
     "path": "data/validation-00000-of-00001.parquet", "rows": 10042, "bytes": 6315951,
     "sha256": "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd", "licence": "MIT (card text)"},
    {"key": "winogrande_dev", "use": "purge", "repo": "allenai/winogrande",
     "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
     "path": "winogrande_xl/validation-00000-of-00001.parquet", "rows": 1267, "bytes": 85928,
     "sha256": "f9d914d1818c0aba98cabdf8af2bc4bd943c462b9c5e7062647e05c2ce3315d1",
     "licence": "not stated in the card"},
    {"key": "boolq_val", "use": "purge", "repo": "google/boolq",
     "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
     "path": "data/validation-00000-of-00001.parquet", "rows": 3270, "bytes": 1257630,
     "sha256": "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8", "licence": "cc-by-sa-3.0"},
    {"key": "piqa_val", "use": "purge", "repo": "baber/piqa",
     "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
     "path": "piqa_validation.parquet", "rows": 1838, "bytes": 299694,
     "sha256": "2b17ee1215b1ceb2ddabf4c90a05c5a8b833bef9d207960f57d111c43a2080f3",
     "licence": "no dataset card at this revision"},
)

TOKENIZER = {"repo": "zai-org/GLM-4.7-Flash",
             "revision": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
             "files": {"config.json": "dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb",
                       "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
                       "tokenizer_config.json":
                           "31a173e2797ddc8b72ac996803513e627fc28d7aad02cfcce321a431d865c86d"}}

SCIQ_SHUFFLE_SEED = 20260815
NGRAM_N = 8

# What the original run measured at every step (its build log + report). "text_sha256" is the sha256 of
# every row WITHOUT its tokenizer-dependent keys (ids, n_ctx, n_cont), serialized like the corpus; it was
# computed from the original file, and it tells "the data differs" apart from "the tokenizer differs".
EXPECT = {
    "pool_kept": {"arc_challenge_train": 1119, "arc_easy_train": 2251, "openbookqa_train": 4957,
                  "sciq_train": 11643},
    "pool_dropped": {"sciq_train": 36},
    "index_exact_questions": 19896,
    "index_distinct_ngrams": 396989,
    "purged_by_source": {"arc_challenge_train": 51, "arc_easy_train": 91, "openbookqa_train": 15,
                         "sciq_train": 21},
    "kept_by_source": {"arc_challenge_train": 1068, "arc_easy_train": 2160, "openbookqa_train": 4942,
                       "sciq_train": 11622},
    "text_sha256": "a79389822ee58bad4c314588b0462ece8814b742706ef759111e8bceba4b8f59",
}

# The original build ran datasets.load_dataset on these files; this script reads the same parquet files
# with pyarrow (a single-file split yields its rows in file order either way) and reproduces the sha256.
ORIGINAL_LIBRARIES = {"datasets": "4.4.1", "huggingface_hub": "1.10.2", "pyarrow": "22.0.0",
                      "tokenizers": "0.22.1", "transformers": "5.8.1"}
ROW_KEYS = ["id", "source", "answerKey", "gold_idx", "question", "gold_text", "prompt", "completion",
            "ids", "n_ctx", "n_cont"]


# =========================================================================================== small helpers
class Refused(Exception):
    """A check failed. The message says which one, and what to do about it."""


def say(msg=""):
    """ASCII only: Windows consoles are often cp1252, and a non-ASCII path must not crash the run."""
    sys.stdout.write(str(msg).encode("ascii", "backslashreplace").decode("ascii") + "\n")
    sys.stdout.flush()


def _abs(p):
    return os.path.abspath(os.path.expanduser(p))


def _jd(d):
    return json.dumps(d, sort_keys=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def measure(path):
    """(sha256, bytes, rows, first two bytes) of a file AS IT IS ON DISK."""
    h, n, rows, last, head = hashlib.sha256(), 0, 0, b"", b""
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            if not head:
                head = b[:2]
            h.update(b)
            n += len(b)
            rows += b.count(b"\n")
            last = b[-1:]
    if n and last != b"\n":
        rows += 1                                   # a last line without its newline is still a row
    return h.hexdigest(), n, rows, head


def problems_of(sha, n, rows, head=b""):
    """Every way a measured file differs from the pinned corpus, plus a hint for two common mistakes."""
    out = []
    if sha != CORPUS["sha256"]:
        out.append("sha256 %s, pinned %s" % (sha, CORPUS["sha256"]))
    if rows != CORPUS["rows"]:
        out.append("%d rows, pinned %d" % (rows, CORPUS["rows"]))
    if n != CORPUS["bytes"]:
        out.append("%d bytes, pinned %d" % (n, CORPUS["bytes"]))
    if out and head == b"\x1f\x8b":
        out.append("hint: it is gzip-compressed, decompress it first")
    elif out and rows == CORPUS["rows"] and n == CORPUS["bytes"] + CORPUS["rows"]:
        out.append("hint: exactly one extra byte per row -- its line endings were converted to CRLF "
                   "(git core.autocrlf or a Windows editor); get an unmodified copy")
    return out


def _repo_dir(s):
    return "datasets--" + s["repo"].replace("/", "--")


def expect(name, got, want, why):
    """Compare one intermediate result with the original run; stop at the first step that diverges."""
    if got != want:
        raise Refused("%s: got %s, the original build had %s -- %s"
                      % (name, _jd(got) if isinstance(got, dict) else got,
                         _jd(want) if isinstance(want, dict) else want, why))
    say("OK   %-24s %s" % (name, _jd(got) if isinstance(got, dict) else got))


# =========================================================================================== --rebuild: sources
def fetch_sources(offline_cache, cache_dir):
    """Local paths of the ten pinned files, each verified against its pinned sha256 and size."""
    paths = {}
    if offline_cache:
        root = _abs(offline_cache)

        def has_repos(d):
            return any(os.path.isdir(os.path.join(d, _repo_dir(s))) for s in SOURCES)

        if not has_repos(root) and has_repos(os.path.join(root, "hub")):
            root = os.path.join(root, "hub")
        say("INFO reading the pinned dataset files from the hub cache %s (no network, nothing written)" % root)
        for s in SOURCES:
            p = os.path.join(root, _repo_dir(s), "snapshots", s["revision"], *s["path"].split("/"))
            if not os.path.isfile(p):
                raise Refused("%s (%s @ %s, %s) is not in the offline cache %s. All ten files are required "
                              "(a missing purge family would change the corpus); run --rebuild without "
                              "--offline-cache to download them" % (s["key"], s["repo"], s["revision"],
                                                                    s["path"], root))
            paths[s["key"]] = p
    else:
        from huggingface_hub import hf_hub_download
        say("INFO fetching the pinned dataset files from huggingface.co (%.1f MB in total; a file already "
            "in the cache is not downloaded again)" % (sum(s["bytes"] for s in SOURCES) / 1e6))
        for s in SOURCES:
            try:
                paths[s["key"]] = hf_hub_download(repo_id=s["repo"], filename=s["path"], repo_type="dataset",
                                                  revision=s["revision"], cache_dir=cache_dir or None)
            except Exception as e:
                raise Refused("%s (%s @ %s, %s) could not be fetched: %s: %s. All ten files are required "
                              "(a missing purge family would change the corpus); check the network or proxy "
                              "and run again" % (s["key"], s["repo"], s["revision"], s["path"],
                                                 type(e).__name__, str(e)[:300]))
    import pyarrow.parquet as pq
    for s in SOURCES:
        p = paths[s["key"]]
        got, n = sha256_file(p), os.path.getsize(p)
        if got != s["sha256"] or n != s["bytes"]:
            raise Refused("%s: %s is sha256 %s / %d bytes, pinned %s / %d -- a corrupt or substituted "
                          "file; delete it and run again" % (s["key"], p, got, n, s["sha256"], s["bytes"]))
        rows = pq.ParquetFile(p).metadata.num_rows
        if rows != s["rows"]:
            raise Refused("%s: %d rows, pinned %d" % (s["key"], rows, s["rows"]))
        say("OK   %-19s %s @ %s  %s  sha256 ok, %d rows" % (s["key"], s["repo"], s["revision"][:8], s["path"],
                                                           rows))
    return paths


def read_rows(path, columns):
    """All rows of one parquet file, in file order: the order datasets.load_dataset yields for a split
    stored as this single file, which is the order the original build iterated."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).read(columns=columns).to_pylist()


# =========================================================================================== --rebuild: 1 POOL
def _key_to_index(answer_key, labels):
    # Original builder _key_to_index + alt_map, verbatim (its third branch repeats the second; kept as is).
    ak = str(answer_key).strip()
    if ak in labels:
        return labels.index(ak)
    alt = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}.get(ak)
    if alt is not None and alt in labels:
        return labels.index(alt)
    inv = {v: k for k, v in {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}.items()}
    if ak in inv and inv[ak] in labels:
        return labels.index(inv[ak])
    return None


def _shuffle(rng, x):
    """random.Random.shuffle exactly as CPython 3.10-3.13 implement it (Fisher-Yates from the end, each j
    drawn by rejection sampling on getrandbits(n.bit_length())). Python documents only random() as
    reproducible across versions for a given seed; shuffle() is an algorithm on top of getrandbits() that a
    future version may change, so it is spelled out here to pin SciQ's answer order."""
    for i in reversed(range(1, len(x))):
        n = i + 1
        k = n.bit_length()
        j = rng.getrandbits(k)
        while j >= n:
            j = rng.getrandbits(k)
        x[i], x[j] = x[j], x[i]


def build_pool(paths):
    """The original builder's load_pool, verbatim except that rows come from the pinned parquet files."""
    items, counts, dropped = [], {}, {}

    def add(src, rid, question, texts, labels, key_idx):
        if key_idx is None or not question or not str(question).strip():
            dropped[src] = dropped.get(src, 0) + 1
            return
        texts = [str(t).strip() for t in texts]
        if any(not t for t in texts):
            dropped[src] = dropped.get(src, 0) + 1
            return
        items.append({"id": "%s:%s" % (src, rid), "source": src,
                      "question": str(question).strip(),
                      "choices": {"text": texts, "label": labels},
                      "answerKey": labels[key_idx], "gold_idx": key_idx})
        counts[src] = counts.get(src, 0) + 1

    for src in ("arc_easy_train", "arc_challenge_train"):
        for r in read_rows(paths[src], ["id", "question", "choices", "answerKey"]):
            labels = list(r["choices"]["label"])
            add(src, r["id"], r["question"], r["choices"]["text"], labels,
                _key_to_index(r["answerKey"], labels))

    for r in read_rows(paths["openbookqa_train"], ["id", "question_stem", "choices", "answerKey"]):
        labels = list(r["choices"]["label"])
        add("openbookqa_train", r["id"], r["question_stem"], r["choices"]["text"], labels,
            _key_to_index(r["answerKey"], labels))

    # SciQ ships gold + 3 distractors, not an ordered choice list: the order is drawn from ONE seeded rng
    # walked in row order, and a dropped row draws nothing (it is skipped before the shuffle).
    rng = random.Random(SCIQ_SHUFFLE_SEED)
    sciq = read_rows(paths["sciq_train"], ["question", "distractor1", "distractor2", "distractor3",
                                           "correct_answer"])
    for i, r in enumerate(sciq):
        gold = str(r["correct_answer"]).strip()
        opts = [gold] + [str(r["distractor%d" % k]).strip() for k in (1, 2, 3)]
        if len(set(o.lower() for o in opts)) != 4:
            dropped["sciq_train"] = dropped.get("sciq_train", 0) + 1
            continue
        order = list(range(4))
        _shuffle(rng, order)
        texts = [opts[j] for j in order]
        add("sciq_train", "sciq-%05d" % i, r["question"], texts, ["A", "B", "C", "D"], texts.index(gold))
    return items, counts, dropped


# =========================================================================================== --rebuild: 2 PURGE
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def norm(s):
    s = _PUNCT.sub(" ", str(s).lower())
    return _WS.sub(" ", s).strip()


def ngrams(text, n=NGRAM_N):
    w = norm(text).split()
    if len(w) < n:
        return set()
    return set(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))


def purge(items, paths):
    """The original builder's contamination purge (its C5 control), verbatim: plant a known duplicate,
    index the six held-out families, drop every item that hits the index."""
    arc_test = read_rows(paths["arc_easy_test"], ["id", "question", "choices", "answerKey"])
    p0 = arc_test[0]
    planted = {"id": "PLANTED_POSITIVE_CONTROL:%s" % p0["id"], "source": "planted_control",
               "question": p0["question"],
               "choices": {"text": list(p0["choices"]["text"]), "label": list(p0["choices"]["label"])},
               "answerKey": p0["answerKey"],
               "gold_idx": list(p0["choices"]["label"]).index(p0["answerKey"])}
    items = items + [planted]

    fams = {"arc_easy_test": [r["question"] for r in arc_test],
            "arc_challenge_test": [r["question"] for r in read_rows(paths["arc_challenge_test"], ["question"])],
            "hellaswag_val": [r["ctx"] for r in read_rows(paths["hellaswag_val"], ["ctx"])],
            "winogrande_dev": [r["sentence"] for r in read_rows(paths["winogrande_dev"], ["sentence"])],
            "boolq_val": [r["question"] for r in read_rows(paths["boolq_val"], ["question"])],
            "piqa_val": [r["goal"] for r in read_rows(paths["piqa_val"], ["goal"])]}
    exact, gram_owner = {}, {}
    for fam, texts in sorted(fams.items()):
        for t in texts:
            nq = norm(t)
            if nq:
                exact.setdefault(nq, fam)
            for g in ngrams(t):
                gram_owner.setdefault(g, fam)
    expect("index_exact_questions", len(exact), EXPECT["index_exact_questions"], "a held-out text differs")
    expect("index_distinct_ngrams", len(gram_owner), EXPECT["index_distinct_ngrams"], "a held-out text differs")

    kept, purged = [], []
    for it in items:
        nq = norm(it["question"])
        hit_fam = None
        if nq and nq in exact:
            hit_fam = exact[nq]
        else:
            for g in ngrams(it["question"]):
                if g in gram_owner:
                    hit_fam = gram_owner[g]
                    break
        if hit_fam is None:
            kept.append(it)
        else:
            purged.append((it["id"], it["source"], hit_fam))

    planted_hits = [p for p in purged if p[0] == planted["id"]]
    if len(planted_hits) != 1 or any(it["id"] == planted["id"] for it in kept):
        raise Refused("the purge did not catch the planted verbatim ARC-Easy test item, so its result "
                      "would be meaningless -- this is a bug, not a data problem")
    say("OK   %-24s caught (%s)" % ("purge_positive_control", planted["id"]))
    real = [p for p in purged if p[0] != planted["id"]]
    by_src, by_fam = {}, {}
    for _, src, fam in real:
        by_src[src] = by_src.get(src, 0) + 1
        by_fam[fam] = by_fam.get(fam, 0) + 1
    expect("purged_by_source", by_src, EXPECT["purged_by_source"], "the held-out sets or the pool differ")
    # NOT compared with the original: an item whose 8-grams hit several families is credited to the one
    # whose gram comes first in a Python set, i.e. in per-process string-hash order (measured: the split
    # moves with PYTHONHASHSEED while the purged SET, and so the corpus, does not).
    say("INFO %-24s %s (attribution only; varies run to run)" % ("purged_by_family", _jd(by_fam)))
    return kept


# =========================================================================================== --rebuild: 3 FORMAT
def build_requests(items, tok):
    """One scoring request per (item, choice).

    VERBATIM copy of build_requests in glm-node/tools/glm_arc_eval.py (lines 3459-3473 of the copy whose
    sha256 starts 6ce813f8), the eval's own scoring formatter: the training row IS the eval's gold-choice
    request. Copied rather than imported because importing glm_arc_eval sets HF_HUB_OFFLINE=1 and
    TRANSFORMERS_OFFLINE=1 in os.environ, which would silently break a later download in this process."""
    reqs = []
    for qi, it in enumerate(items):
        ctx = "Question: %s\nAnswer:" % it["question"].strip()
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        for ci, ch in enumerate(it["choices"]["text"]):
            cont = " " + str(ch).strip()
            full_ids = tok.encode(ctx + cont, add_special_tokens=False)
            n_cont = len(full_ids) - len(ctx_ids)
            if n_cont < 1:
                full_ids = ctx_ids + tok.encode(cont, add_special_tokens=False)
                n_cont = len(full_ids) - len(ctx_ids)
            reqs.append(dict(qi=qi, ci=ci, ids=full_ids, n_ctx=len(full_ids) - n_cont, n_cont=n_cont))
    return reqs


class _SpyTok(object):
    """The original builder's _SpyTok, verbatim: forwards every call to the real tokenizer and records the
    STRINGS build_requests constructs, so the prompt/completion strings are read out of the formatter."""

    def __init__(self, tok):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "calls", [])

    def encode(self, s, **kw):
        self.calls.append(s)
        return self._tok.encode(s, **kw)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_tok"), k)

    def __call__(self, *a, **kw):
        return self._tok(*a, **kw)


def format_item(item, tok):
    """The original builder's format_item, verbatim: one row, or (None, reason) when the formatter's seam
    is not exactly what it is asserted to be -- counted, never silently adapted."""
    spy = _SpyTok(tok)
    reqs = build_requests([item], spy)
    gi = item["gold_idx"]
    if len(reqs) != len(item["choices"]["text"]):
        return None, "req_count"
    req = reqs[gi]
    if req["ci"] != gi or req["qi"] != 0:
        return None, "req_index"
    if len(spy.calls) != 1 + len(item["choices"]["text"]):
        return None, "call_count"
    prompt = spy.calls[0]
    full = spy.calls[1 + gi]
    if not full.startswith(prompt):
        return None, "prefix"
    completion = full[len(prompt):]
    ids = list(req["ids"])
    n_ctx, n_cont = int(req["n_ctx"]), int(req["n_cont"])
    if n_cont < 1 or n_ctx < 1 or n_ctx + n_cont != len(ids):
        return None, "span"
    if tok.encode(full, add_special_tokens=False) != ids:
        return None, "reencode"
    return {"id": item["id"], "source": item["source"], "answerKey": item["answerKey"],
            "gold_idx": gi, "question": item["question"], "gold_text": item["choices"]["text"][gi],
            "prompt": prompt, "completion": completion,
            "ids": ids, "n_ctx": n_ctx, "n_cont": n_cont}, None


def text_digest(rows):
    """sha256 over every row minus its tokenizer-dependent keys, serialized like the corpus."""
    h = hashlib.sha256()
    for r in rows:
        h.update((json.dumps({k: v for k, v in r.items() if k not in ("ids", "n_ctx", "n_cont")})
                  + "\n").encode("utf-8"))
    return h.hexdigest()


def check_tokenizer_files(config_dir):
    bad = []
    for name, want in sorted(TOKENIZER["files"].items()):
        p = os.path.join(config_dir, name)
        if not os.path.isfile(p):
            bad.append("%s is missing" % name)
            continue
        got = sha256_file(p)
        if got != want:
            bad.append("%s is sha256 %s, pinned %s" % (name, got, want))
    if bad:
        raise Refused("the tokenizer in --config-dir %s is not the pinned one: %s. It must hold "
                      "tokenizer.json, tokenizer_config.json and config.json from %s @ %s; "
                      "`python glm-node/build_packs.py --base-only` puts them in <home>/base/config"
                      % (config_dir, "; ".join(bad), TOKENIZER["repo"], TOKENIZER["revision"]))
    say("OK   %-24s %s (%s @ %s)" % ("tokenizer_files", config_dir, TOKENIZER["repo"], TOKENIZER["revision"][:8]))


def show_versions():
    from importlib import metadata
    say("INFO python %s on %s" % (platform.python_version(), platform.system()))
    for name in ("pyarrow", "huggingface_hub", "transformers", "tokenizers"):
        try:
            got = metadata.version(name)
        except Exception:
            got = "not installed"
        if got == ORIGINAL_LIBRARIES[name]:
            say("OK   %-24s %s" % (name, got))
        else:
            say("WARN %-24s %s (the verified rebuild used %s; fine if the final sha256 matches)"
                % (name, got, ORIGINAL_LIBRARIES[name]))


def rebuild(a, home):
    """Every check of the rebuild, then the corpus bytes -- nothing is written to disk in here."""
    offline = bool(a.offline_cache)
    if offline:
        # Nothing below needs the network; with these set the Hugging Face stack cannot reach it either.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    need = ["pyarrow", "transformers"] + ([] if offline else ["huggingface_hub"])
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if missing:
        raise Refused("--rebuild needs the Python packages %s: pip install pyarrow==22.0.0 and "
                      "pip install -r glm-node/requirements-node.txt" % ", ".join(missing))
    config_dir = _abs(a.config_dir or os.path.join(home, "base", "config"))
    check_tokenizer_files(config_dir)
    show_versions()

    t0 = time.time()
    paths = fetch_sources(a.offline_cache, a.cache_dir)
    say("INFO 1/3 pool")
    items, counts, dropped = build_pool(paths)
    expect("pool_kept", counts, EXPECT["pool_kept"], "a train split differs")
    expect("pool_dropped", dropped, EXPECT["pool_dropped"], "a train split differs")
    say("INFO 2/3 contamination purge")
    kept = purge(items, paths)
    by_src = {}
    for it in kept:
        by_src[it["source"]] = by_src.get(it["source"], 0) + 1
    expect("kept_by_source", by_src, EXPECT["kept_by_source"], "the purge differs")

    say("INFO 3/3 format + tokenize")
    try:
        from transformers import AutoTokenizer
        # The original call, with the node's config dir in place of the build box's model dir.
        tok = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
    except Exception as e:
        raise Refused("could not load the tokenizer from %s: %s: %s" % (config_dir, type(e).__name__,
                                                                         str(e)[:300]))
    rows, fmt_fail = [], {}
    for it in kept:
        r, why = format_item(it, tok)
        if r is None:
            fmt_fail[why] = fmt_fail.get(why, 0) + 1
            continue
        rows.append(r)
    if fmt_fail:
        raise Refused("%d items did not pass through the eval formatter's seam (%s); the original had none "
                      "-- the tokenizer differs" % (sum(fmt_fail.values()), _jd(fmt_fail)))
    expect("rows_formatted", len(rows), CORPUS["rows"], "the purge differs")
    expect("text_sha256", text_digest(rows), EXPECT["text_sha256"],
           "the rows' TEXT differs from the original although every count matched")

    blob = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")   # the original writer's bytes
    sha = hashlib.sha256(blob).hexdigest()
    if sha != CORPUS["sha256"] or len(blob) != CORPUS["bytes"]:
        raise Refused("the text of every row matches the original but the token ids do not (sha256 %s, "
                      "%d bytes): the tokenizer differs. Check --config-dir and the transformers/tokenizers "
                      "versions above (the verified rebuild used transformers %s, tokenizers %s)"
                      % (sha, len(blob), ORIGINAL_LIBRARIES["transformers"], ORIGINAL_LIBRARIES["tokenizers"]))
    say("INFO rebuilt in memory in %.0f s: %d rows, sha256 matches -- writing it" % (time.time() - t0, len(rows)))
    return blob


# =========================================================================================== the other two modes
def check_source_file(path):
    """--from-file: refuse a wrong file BEFORE anything is copied."""
    src = _abs(path)
    if not os.path.isfile(src):
        raise Refused("--from-file %s: no such file" % src)
    say("INFO checking %s" % src)
    bad = problems_of(*measure(src))
    if bad:
        raise Refused("%s is not the corpus, refused (nothing copied): %s" % (src, "; ".join(bad)))
    return src


def check_url(url):
    """--from-url: only http(s); returns the URL as it may be printed (no credentials, no query)."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("https", "http"):
        raise Refused("--from-url needs an http(s) URL, got scheme %r" % parts.scheme)
    shown = "%s://%s%s" % (parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path)
    if parts.query:
        shown += "?<query not shown>"
    return shown


def download(url, tmp):
    """--from-url: stream into the temp file; stop as soon as more bytes arrive than the corpus has."""
    say("INFO downloading %s" % check_url(url))
    if urllib.parse.urlsplit(url).scheme.lower() == "http":
        say("WARN plain http: the sha256 check still guarantees the content, but the transfer is not private")
    req = urllib.request.Request(url, headers={"User-Agent": "glm-node-fetch-corpus/1"})
    t0, n = time.time(), 0
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fo:
            while True:
                b = r.read(1 << 16)
                if not b:
                    break
                n += len(b)
                if n > CORPUS["bytes"]:
                    raise Refused("the URL sent more than %d bytes, so it is not this corpus; stopped"
                                  % CORPUS["bytes"])
                fo.write(b)
            fo.flush()
            os.fsync(fo.fileno())
    except Refused:
        raise
    except Exception as e:
        raise Refused("download failed: %s: %s" % (type(e).__name__, str(e)[:300]))
    say("INFO downloaded %d bytes in %.1f s" % (n, time.time() - t0))


# =========================================================================================== install + main
def new_temp(out):
    d = os.path.dirname(out)
    try:
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(out) + ".", suffix=".partial", dir=d)
        os.close(fd)
    except OSError as e:
        raise Refused("cannot write in %s: %s" % (d, e))
    return tmp


def remove_partial(tmp):
    if tmp is not None and os.path.exists(tmp):
        try:
            os.remove(tmp)
            say("INFO removed the partial file %s" % tmp)
        except OSError as e:
            say("WARN could not remove the partial file %s: %s -- delete it by hand" % (tmp, e))


def install(tmp, out):
    """Verify the temp file on disk, rename it into place, verify it again at its final path."""
    sha, n, rows, head = measure(tmp)
    bad = problems_of(sha, n, rows, head)
    if bad:
        raise Refused("the file does not verify, refused (nothing installed): " + "; ".join(bad))
    try:
        os.replace(tmp, out)
    except OSError as e:
        raise Refused("cannot move the verified file to %s: %s (is a node process holding it open?)" % (out, e))
    sha, n, rows, head = measure(out)              # the artefact at its final path, not the announcement
    bad = problems_of(sha, n, rows, head)
    if bad:
        os.remove(out)
        raise Refused("%s failed its re-check after the rename and was deleted: %s -- check the disk"
                      % (out, "; ".join(bad)))
    return sha, n, rows


def say_verified(out, sha, n, rows, t0):
    say("OK   sha256 %s == pinned" % sha)
    say("OK   rows   %d == pinned" % rows)
    say("OK   bytes  %d == pinned" % n)
    say("CORPUS READY  %s  (%.0f s)" % (out, time.time() - t0))


def manifest():
    return {
        "what": "pinned facts of the glm-node training corpus. The authoritative copy is the constants "
                "block of glm-node/fetch_corpus.py; this file is its `--print-manifest` output. The corpus "
                "itself is NOT distributed: fetch_corpus.py --rebuild makes it from the public sources "
                "below (each user downloads them from their owners), or --from-file / --from-url takes an "
                "operator's copy; every mode ends with the sha256 + row-count check.",
        "corpus": CORPUS,
        "tokenizer": TOKENIZER,
        "build": {"sciq_shuffle_seed": SCIQ_SHUFFLE_SEED, "purge_ngram_n": NGRAM_N,
                  "prompt": "Question: <question>\nAnswer:", "completion": " <gold choice>",
                  "row_keys": ROW_KEYS, "writer": "json.dumps(row) + \"\\n\", UTF-8, LF"},
        "sources": [dict(s) for s in SOURCES],
        "expected_counts": EXPECT,
        "original_build": {"date": "2026-08-15", "libraries": ORIGINAL_LIBRARIES,
                           "loader": "datasets.load_dataset (fetch_corpus.py reads the same parquet files "
                                     "with pyarrow and reproduces the sha256)"},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fetch_corpus.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--rebuild", action="store_true", help="rebuild the corpus from the pinned public datasets")
    mode.add_argument("--from-file", metavar="PATH", default="", help="take the corpus from a file")
    mode.add_argument("--from-url", metavar="URL", default="", help="download the corpus from an http(s) URL")
    mode.add_argument("--print-manifest", action="store_true", help="print the pinned facts as JSON and exit")
    ap.add_argument("--home", default=os.environ.get("NEURAHASH_GLM_HOME", ""),
                    help="the node's data dir (default env NEURAHASH_GLM_HOME, else ~/glmnode)")
    ap.add_argument("--out", default="", help="the corpus path (default <home>/corpus/%s)" % CORPUS["name"])
    ap.add_argument("--config-dir", default="", help="--rebuild: the tokenizer files (default <home>/base/config)")
    ap.add_argument("--cache-dir", default="", metavar="DIR",
                    help="--rebuild: huggingface_hub download cache (default: its own, HF_HUB_CACHE / HF_HOME)")
    ap.add_argument("--offline-cache", default="", metavar="DIR",
                    help="--rebuild: read the pinned files from this existing hub cache; no network")
    ap.add_argument("--force", action="store_true", help="redo the work even if --out already verifies")
    a = ap.parse_args(argv)

    if a.print_manifest:
        text = json.dumps(manifest(), indent=1) + "\n"
        sys.stdout.flush()
        buf = getattr(sys.stdout, "buffer", None)
        if buf is not None:
            buf.write(text.encode("ascii"))                 # bytes: no CRLF translation on Windows
            buf.flush()
        else:
            sys.stdout.write(text)
        return 0
    defaulted = not (a.rebuild or a.from_file or a.from_url)
    if defaulted:
        a.rebuild = True             # the kit's own hints say plain `python glm-node/fetch_corpus.py`
    if not a.rebuild and (a.config_dir or a.cache_dir or a.offline_cache):
        ap.error("--config-dir, --cache-dir and --offline-cache only apply to --rebuild")
    if a.offline_cache and a.cache_dir:
        ap.error("--offline-cache and --cache-dir are alternatives; give one")
    home = _abs(a.home or os.path.join("~", "glmnode"))
    out = _abs(a.out or os.path.join(home, "corpus", CORPUS["name"]))
    if os.path.isdir(out):
        ap.error("--out %s is a directory; give the file path" % out)

    t0 = time.time()
    how = "rebuild" if a.rebuild else ("from-file" if a.from_file else "from-url")
    say("glm-node fetch_corpus (%s) -> %s" % (how, out))
    say("pinned: %s sha256 %s, %d rows, %d bytes" % (CORPUS["name"], CORPUS["sha256"], CORPUS["rows"],
                                                     CORPUS["bytes"]))
    if defaulted:
        say("INFO no mode given, so --rebuild (see --help for --from-file PATH / --from-url URL)")
    if os.path.isfile(out) and not a.force:
        sha, n, rows, head = measure(out)
        bad = problems_of(sha, n, rows, head)
        if not bad:
            say("OK   already in place and verified; nothing to do (--force redoes it)")
            say_verified(out, sha, n, rows, t0)
            return 0
        say("WARN %s exists but is NOT the corpus (%s); it is replaced only by a file that verifies"
            % (out, "; ".join(bad)))

    tmp = None
    try:
        if a.rebuild:
            blob = rebuild(a, home)
            tmp = new_temp(out)
            with open(tmp, "wb") as fo:
                fo.write(blob)
                fo.flush()
                os.fsync(fo.fileno())
        elif a.from_file:
            src = check_source_file(a.from_file)
            tmp = new_temp(out)
            with open(src, "rb") as fi, open(tmp, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
                fo.flush()
                os.fsync(fo.fileno())
        else:
            check_url(a.from_url)                    # refuse a bad URL before anything is created
            tmp = new_temp(out)
            download(a.from_url, tmp)
        sha, n, rows = install(tmp, out)
    except (Refused, KeyboardInterrupt) as e:
        say("FAIL %s" % (e if isinstance(e, Refused) else "interrupted"))
        remove_partial(tmp)
        say("CORPUS NOT INSTALLED (exit 1)")
        return 1
    finally:
        remove_partial(tmp)          # also on an unexpected error: never leave a partial file behind
    say_verified(out, sha, n, rows, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
