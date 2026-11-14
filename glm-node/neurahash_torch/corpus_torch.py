"""Tiny text corpus + tokenizer for the PyTorch demo. Same synthetic language as
the NumPy demo so the loss visibly drops in seconds.

REAL CORPUS (Rung 1): set NEURAHASH_CORPUS=real (or a dir path) to train on the real
text files in corpus_data/*.txt (code + prose) instead of the 216-sentence toy grammar.
The toy corpus floors a model in ~30 rounds because it has almost no entropy; a real
corpus has no practical convergence horizon, so verification keeps working round after
round (held-out loss keeps dropping, deltas stay healthy). build_data still returns the
SAME (tok, train_data, val_data) contract and get_batch is UNCHANGED, so the coordinator's
recompute-verifier replays the worker's exact batches bit-for-bit. NOTE: both worker and
coordinator must read IDENTICAL corpus files (same vocab/encoding); guaranteed on one
machine — cross-machine needs a content-addressed corpus (a later rung)."""

import glob
import hashlib
import json
import os

import torch
import numpy as np

SUBJECTS = ["the cat", "the dog", "a bird", "the fox", "my friend", "the child"]
VERBS = ["sat on", "ran to", "looked at", "jumped over", "found", "loved"]
OBJECTS = ["the mat", "the sun", "the hill", "a box", "the lake", "the moon"]


def make_text(n_sentences=4000, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_sentences):
        parts.append(f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}. ")
    return "".join(parts)


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def _corpus_dir(mode):
    """Resolve the real-corpus directory. NEURAHASH_CORPUS may be a directory path; otherwise
    NEURAHASH_CORPUS_DIR, else corpus_data/ at the repo root (parent of this package)."""
    if mode and (os.path.sep in mode or "/" in mode) and os.path.isdir(mode):
        return mode
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("NEURAHASH_CORPUS_DIR", os.path.join(repo_root, "corpus_data"))


# ---------------------------------------------------------------------------
# Grounding corpora (issue #51): scale the real corpus FAR past the 3.9MB toy by ingesting large,
# permissively-licensed public sources (S2ORC / Pile of Law / Principia). OPT-IN ONLY — activated by
# NEURAHASH_CORPUS_SOURCES (or a "grounding:<srcs>" mode); with neither set, the toy/real path below
# is byte-identical to before. The merged corpus is content-addressed + deterministically ordered in
# neurahash.grounding_corpora, so the coordinator and every worker derive the SAME corpus + sha.
def _grounding_sources(mode=None):
    """Resolve the requested grounding sources, or None if the grounding path is not requested.

    Triggers: NEURAHASH_CORPUS_SOURCES=<list> (preferred), or a mode/NEURAHASH_CORPUS of the form
    "grounding" / "grounding:s2orc,law" (bare "grounding" -> all sources)."""
    if mode is None:
        mode = (os.environ.get("NEURAHASH_CORPUS", "") or "").strip()
    spec = (os.environ.get("NEURAHASH_CORPUS_SOURCES", "") or "").strip()
    m = (mode or "").strip()
    if not spec and m.lower().startswith("grounding"):
        # "grounding" -> all; "grounding:s2orc,law" -> those
        rest = m.split(":", 1)[1] if ":" in m else "all"
        spec = rest.strip() or "all"
    if not spec:
        return None
    from neurahash.grounding_corpora import parse_sources
    return parse_sources(spec)


def build_grounding_data(device, sources, char_level=True, base=None):
    """Char-level corpus built from the grounding sources (issue #51). Same (tok, train, val) contract
    + per-record 90/10 split as build_real_data, so get_batch stays seed-reproducible (the
    recompute-verify replays the exact batch). The corpus text + its order are content-addressed in
    grounding_corpora (deterministic across machines)."""
    from neurahash.grounding_corpora import build_grounding_records
    records, report = build_grounding_records(sources, use_sample=True)
    parts = [txt for _, _, txt in records]
    if not parts:
        raise FileNotFoundError(
            f"NEURAHASH_CORPUS_SOURCES={sources!r} but no records admitted (configure a "
            f"*_DIR env to a JSONL dump, or rely on the bundled task_data/grounding_*_sample.jsonl)")
    if char_level:
        tok = CharTokenizer("\n".join(parts))
    else:
        tok = _qwen_tokenizer(base or "qwen3-1.7b")
    train_ids, val_ids = [], []
    for t in parts:                                   # per-record 90/10 (every source represented in val)
        ids = tok.encode(t)
        n = int(0.9 * len(ids))
        train_ids.extend(ids[:n])
        val_ids.extend(ids[n:])
    train = torch.tensor(train_ids, dtype=torch.long, device=device)
    val = torch.tensor(val_ids, dtype=torch.long, device=device)
    return tok, train, val


def real_corpus_parts(corpus_dir):
    """Deterministically read corpus_data/*.txt (sorted) -> (file_list, [text, ...])."""
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    parts = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            parts.append(fh.read())
    return files, parts


def corpus_sha(mode=None):
    """Canonical content-address of the corpus the pool is training on, so a worker and the
    coordinator can confirm they read IDENTICAL data BEFORE training begins (corpus_torch admits both
    sides must read the same vocab/encoding or recompute-verify silently rejects honest work).

    The hash is over the SORTED list of (basename, raw bytes) for corpus_data/*.txt — the same sorted
    order build_real_data/build_qwen_bpe_data tokenize in — so it captures exactly what enters the
    tokenizer. The basename (not the full path) is committed so two machines with the corpus at
    different absolute paths still agree. Mirrors the content-addressing pattern in
    neura_l1.base_import (hash the values, never the surrounding container).

    Returns a 64-hex sha256. For the TOY corpus (no real-corpus mode set / empty dir) returns the
    sha of the empty manifest, an explicit, stable sentinel — the toy grammar is generated from a
    fixed seed on both sides so it needs no content check; mismatch only matters for the real corpus.
    `mode` follows the same resolution as build_data (a dir path, "real"/"qwen", or None to read
    NEURAHASH_CORPUS from the env)."""
    if mode is None:
        mode = (os.environ.get("NEURAHASH_CORPUS", "") or "").strip()
    # Grounding corpora (issue #51) have their OWN content-addressed manifest -> defer to it so the
    # coordinator/worker handshake covers the merged S2ORC/Law/Principia corpus.
    sources = _grounding_sources(mode)
    if sources:
        from neurahash.grounding_corpora import grounding_corpus_sha
        return grounding_corpus_sha(sources)
    # Only the real/dir corpus is content-addressed; the toy + qwen-BPE corpora both read the same
    # corpus_data/*.txt, so hash that directory whenever it is the data source.
    is_toy = (not mode) or mode.lower() in ("toy", "synthetic", "0", "off", "false", "no")
    h = hashlib.sha256()
    if not is_toy:
        cdir = _corpus_dir(mode)
        files, _ = real_corpus_parts(cdir)
        h.update(b"neurahash-corpus-v1")
        for f in files:
            name = os.path.basename(f).encode("utf-8")
            with open(f, "rb") as fh:
                data = fh.read()
            h.update(len(name).to_bytes(8, "big")); h.update(name)
            h.update(len(data).to_bytes(8, "big")); h.update(data)
    else:
        h.update(b"neurahash-corpus-toy")
    return h.hexdigest()


def build_real_data(device, mode="real"):
    """Char-level real corpus from corpus_data/*.txt. Builds ONE tokenizer over all files
    (deterministic sorted(set)), then splits EACH file 90/10 so the held-out set is
    representative of every source (not just the tail file)."""
    cdir = _corpus_dir(mode)
    files, parts = real_corpus_parts(cdir)
    if not parts or not any(parts):
        raise FileNotFoundError(
            f"NEURAHASH_CORPUS={mode!r} but no non-empty *.txt files in {cdir} "
            f"(drop text files there, or unset NEURAHASH_CORPUS for the toy corpus)")
    tok = CharTokenizer("\n".join(parts))
    train_ids, val_ids = [], []
    for t in parts:
        ids = tok.encode(t)
        n = int(0.9 * len(ids))
        train_ids.extend(ids[:n])
        val_ids.extend(ids[n:])
    train = torch.tensor(train_ids, dtype=torch.long, device=device)
    val = torch.tensor(val_ids, dtype=torch.long, device=device)
    return tok, train, val


class _HFTokAdapter:
    """Wrap an HF tokenizer to the (vocab_size, encode, decode) contract build_data returns. vocab_size is
    pinned to the MODEL's embedding size (>= the tokenizer's token count) so the pool builds an embedding
    that matches the imported base's lm_head/embed_tokens; every token id the tokenizer emits is < that."""

    def __init__(self, hf_tok, vocab_size):
        self._t = hf_tok
        self._vocab = int(vocab_size)

    @property
    def vocab_size(self):
        return self._vocab

    def encode(self, s):
        return self._t.encode(s, add_special_tokens=False)

    def decode(self, ids):
        return self._t.decode([int(i) for i in ids])


def _qwen_tokenizer(base="qwen3-1.7b"):
    """Build the base model's BPE tokenizer wrapped to the build_data tokenizer contract (vocab pinned
    to the model embedding size). Factored out so the grounding path can reuse the same tokenizer."""
    from transformers import AutoTokenizer, AutoConfig                      # lazy: heavy, base-path only
    try:
        from model_registry import resolve_model
        model_id = resolve_model(base)
    except Exception:
        model_id = base
    hf_tok = AutoTokenizer.from_pretrained(model_id)
    vocab = int(getattr(AutoConfig.from_pretrained(model_id), "vocab_size", None) or hf_tok.vocab_size)
    return _HFTokAdapter(hf_tok, vocab)


def build_qwen_bpe_data(device, base="qwen3-1.7b", corpus_dir=None):
    """Real corpus tokenized with the BASE model's own BPE tokenizer (NOT char-level), so the token ids
    line up with the imported base's embedding. Same (tok, train, val) contract + per-file 90/10 split;
    get_batch stays seeded so the coordinator's recompute replays the worker's exact batch."""
    tok = _qwen_tokenizer(base)
    cdir = _corpus_dir(corpus_dir or "real")
    files, parts = real_corpus_parts(cdir)
    if not parts or not any(parts):
        raise FileNotFoundError(f"NEURAHASH_CORPUS=qwen but no non-empty *.txt in {cdir}")
    train_ids, val_ids = [], []
    for t in parts:
        ids = tok.encode(t)
        n = int(0.9 * len(ids))
        train_ids.extend(ids[:n])
        val_ids.extend(ids[n:])
    train = torch.tensor(train_ids, dtype=torch.long, device=device)
    val = torch.tensor(val_ids, dtype=torch.long, device=device)
    return tok, train, val


# ---------------------------------------------------------------------------
# GSM8K-rationale corpus (P2 SFT-on-rationales): NEURAHASH_CORPUS=gsm trains the same BPE-tokenized
# pipeline as build_qwen_bpe_data, but the token stream comes from math-reasoning rationales instead of
# generic prose/code, so P2 can move GSM accuracy on a pre-registered held-out subset instead of loss on
# corpus_data/*.txt. DATA HYGIENE (hard requirement): this module reads ONLY the TRAIN split file; the
# other GSM8K split (the one with public leaderboard scores attached) is never opened anywhere below --
# grep this file for that filename and it will not appear. The merge-gate's held-out slice is instead the
# TAIL of the train split (env-tunable item count), so a contribution can never be graded on data that
# also trained it.
def _gsm_train_path():
    """Path to the GSM8K TRAIN-split JSONL (repo root, sibling of this package's parent dir).
    NEURAHASH_GSM_TRAIN_PATH overrides it (e.g. for tests with a tiny fixture file)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("NEURAHASH_GSM_TRAIN_PATH", os.path.join(repo_root, "gsm8k_train.jsonl"))


def gsm_train_items(path=None):
    """Read the GSM8K train-split JSONL in FILE ORDER (never shuffled) -> list of {"question","answer"}
    dicts. File order is what makes the head/tail split in build_gsm_data reproducible across machines
    without needing to ship a saved index."""
    p = path or _gsm_train_path()
    items = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_gsm_data(device, base="qwen3-1.7b"):
    """GSM8K-rationale corpus tokenized with the BASE model's own BPE tokenizer (same tokenizer
    mechanism as build_qwen_bpe_data), so token ids line up with the imported base's embedding. Each
    item is formatted "<question>\\nAnswer:\\n<full rationale, including the trailing '#### N' line>"
    and items are joined by the tokenizer's EOS token (mirrors tools/draft_contributor.py's stream()
    helper), so training never blends the end of one problem into the start of the next.

    Held-out split (hygiene, hard requirement): NEURAHASH_GSM_VAL_ITEMS (default 400) items are taken
    from the TAIL of the train split for `val`; `train` is built ONLY from the remaining head. The two
    item sets are asserted disjoint before tokenizing (guards a future off-by-one/refactor bug, not just
    trusting the slice)."""
    tok = _qwen_tokenizer(base)
    items = gsm_train_items()
    if not items:
        raise FileNotFoundError(f"NEURAHASH_CORPUS=gsm but no items in {_gsm_train_path()}")
    n_val = int(os.environ.get("NEURAHASH_GSM_VAL_ITEMS", "400"))
    n_val = max(0, min(n_val, len(items) - 1))          # always leave >=1 item for training
    split = len(items) - n_val
    train_items, val_items = items[:split], items[split:]
    train_q = {it["question"] for it in train_items}
    val_q = {it["question"] for it in val_items}
    assert not (train_q & val_q), "build_gsm_data: train/val item overlap (data-hygiene violation)"

    eos_id = getattr(tok._t, "eos_token_id", None)

    def _encode(item_list):
        ids = []
        for ex in item_list:
            text = ex["question"] + "\nAnswer:\n" + ex["answer"]
            ids.extend(tok.encode(text))
            if eos_id is not None:
                ids.append(int(eos_id))
        return ids

    train_ids = _encode(train_items)
    val_ids = _encode(val_items)
    train = torch.tensor(train_ids, dtype=torch.long, device=device)
    val = torch.tensor(val_ids, dtype=torch.long, device=device)
    return tok, train, val


def resolve_corpus_mode(mode=None):
    """Canonical corpus-mode STRING the pool is training on, so the coordinator can ADVERTISE it in the
    hello and a worker can compare it against its own env (issue #90 hello-dictated corpus mode). This is
    a pure string resolver — it does not read any *.txt — so it is cheap enough to call on every join.

    `mode=None` reads NEURAHASH_CORPUS from the env (the historical source of truth); a non-None `mode`
    (a dir path, "real"/"qwen"/"toy"/…) is used verbatim. An empty/unset env normalises to "toy" — the
    stable name for the seed-fixed synthetic grammar — so the advertised value is never the empty string.
    Grounding modes ("grounding[:srcs]") and dir paths pass through unchanged; the actual data selection
    still happens in build_data, this only NAMES the mode for the handshake."""
    if mode is None:
        mode = (os.environ.get("NEURAHASH_CORPUS", "") or "").strip()
    else:
        mode = (mode or "").strip()
    if not mode or mode.lower() in ("toy", "synthetic", "0", "off", "false", "no"):
        return "toy"
    return mode


def build_data(device, seed=0, mode=None):
    # REAL corpus when NEURAHASH_CORPUS is set to anything truthy (or a dir path); else the toy grammar.
    # `mode` (issue #90): when None, read NEURAHASH_CORPUS from the env EXACTLY as before (so every
    # existing caller is byte-identical); when the coordinator dictates a mode via the hello, the worker
    # passes it here so build_data follows the coordinator instead of this box's env.
    if mode is None:
        mode = (os.environ.get("NEURAHASH_CORPUS", "") or "").strip()
    else:
        mode = (mode or "").strip()
    sources = _grounding_sources(mode)                                      # issue #51 grounding corpora
    if sources:
        char_level = mode.lower() not in ("qwen", "qwen-bpe", "bpe")
        return build_grounding_data(device, sources, char_level=char_level,
                                    base=os.environ.get("NEURAHASH_BASE", "qwen3-1.7b"))
    if mode.lower() in ("qwen", "qwen-bpe", "bpe"):                          # base BPE tokenizer (real base)
        return build_qwen_bpe_data(device, os.environ.get("NEURAHASH_BASE", "qwen3-1.7b"))
    if mode.lower() in ("gsm", "gsm8k", "gsm-rationales"):                   # P2 SFT-on-rationales (math)
        return build_gsm_data(device, os.environ.get("NEURAHASH_BASE", "qwen3-1.7b"))
    if mode and mode.lower() not in ("toy", "synthetic", "0", "off", "false", "no"):
        return build_real_data(device, mode)
    text = make_text(seed=seed)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long, device=device)
    n = int(0.9 * len(data))
    return tok, data[:n], data[n:]


def get_batch(data, block_size, batch_size, device, generator=None):
    # a seeded generator makes batch selection reproducible, which is what lets a
    # verifier replay a worker's exact training trajectory (see verify_torch.py).
    ix = torch.randint(len(data) - block_size - 1, (batch_size,),
                       device=device, generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y
