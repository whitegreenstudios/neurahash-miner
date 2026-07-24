"""G1 verifiable-reward verifier -- the pure, deterministic reward function for the G1 RLVR
campaign (docs/G1_PREREGISTRATION_2026-07-24.md).

A *rollout* is one model-generated attempt at a task. For a math task the reward is 1.0 iff the
attempt's FINAL answer numerically equals the task's gold answer, else 0.0. Both the verifier
(miner) role and the learner call this module, so it must be simple, fast, dependency-free (stdlib
only: re, json, math) and impossible to game with formatting tricks -- the FINAL answer wins, never
a gold value that merely appears mid-reasoning.

Task records come from tools/glm_task_prep.py:
    {"task_id": str, "domain": "math", "prompt": str, "gold": str, "gold_raw": str}
where `gold` is already normalized by glm_task_prep.normalize_gold (text after the final '####',
stripped, commas removed -- e.g. "16800" from a gsm8k "#### 16,800"). We stay tolerant anyway so a
raw/unnormalized gold ("$16,800", "16800.0") still compares correctly.

Public API:
    extract_final_answer(text)        -> str | None
    normalize_number(s)               -> str | None
    math_reward(completion, gold)     -> float   (1.0 / 0.0)
    score_rollout(task, completion)   -> dict
    batch_score(tasks_by_id, rollouts)-> dict    (per-rollout scores + aggregate)
"""

import json
import math
import re

# A number token: optional sign, digits with optional thousands-commas, optional decimal; OR a
# bare leading-dot decimal (".5"). Commas / currency / percent are cleaned off in normalize_number,
# so this only has to LOCATE a number, not canonicalize it.
_NUM_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")

# "\boxed{...}" LaTeX final-answer marker (common in math chain-of-thought).
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")

# "The answer is <ans>" / "answer: <ans>" / "answer = <ans>" (case-insensitive). We take the LAST
# occurrence and read the first number after it, so a later declaration overrides an earlier one.
_ANSWER_RE = re.compile(r"answer\s*(?:is|=|:)\s*", re.IGNORECASE)


# ---- normalization + numeric equality -----------------------------------------------------------
def normalize_number(s):
    """Clean formatting off a single value so numeric-equal strings become comparable.

    Strips surrounding whitespace, thousands-separator commas, a leading '$', a trailing '%', a
    single trailing sentence-period ('42.' -> '42', but '16800.0' is kept), and a leading '+'.
    Keeps a leading '-'. Returns the cleaned string, or None if nothing is left. Non-numeric input
    is returned cleaned (not rejected) so gold-compare can fall back to a string match.
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    t = t.replace(",", "").replace("$", "").strip()
    if t.endswith("%"):
        t = t[:-1].strip()
    if t.endswith("."):          # trailing sentence period, not a decimal point
        t = t[:-1]
    if t.startswith("+"):
        t = t[1:]
    t = t.strip()
    return t or None


def _to_float(x):
    """float(x) or None if x is None / not parseable as a number."""
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _values_equal(a, b):
    """True iff a and b represent the same answer. Float-close compare when BOTH parse as numbers
    (so '16800' == '16800.0' == '16,800'); otherwise an exact string compare (non-numeric fallback).
    Either being None is never equal."""
    if a is None or b is None:
        return False
    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-9)
    return str(a) == str(b)


# ---- final-answer extraction --------------------------------------------------------------------
def _first_number(s):
    """Normalized string of the FIRST number in s, or None."""
    m = _NUM_RE.search(s)
    return normalize_number(m.group(0)) if m else None


def _last_number(s):
    """Normalized string of the LAST number in s, or None."""
    last = None
    for m in _NUM_RE.finditer(s):
        last = m
    return normalize_number(last.group(0)) if last else None


def _hash_marker(text):
    """gsm8k '#### <ans>' -- number right after the LAST '####' marker, or None."""
    if "####" not in text:
        return None
    return _first_number(text.rsplit("####", 1)[-1])


def _boxed_marker(text):
    """LaTeX '\\boxed{<ans>}' -- number inside the LAST boxed group, or None."""
    last = None
    for m in _BOXED_RE.finditer(text):
        last = m
    return _first_number(last.group(1)) if last else None


def _answer_is_marker(text):
    """'The answer is <ans>' / 'answer: <ans>' -- number after the LAST such marker, or None."""
    last = None
    for m in _ANSWER_RE.finditer(text):
        last = m
    return _first_number(text[last.end():]) if last else None


def extract_final_answer(text):
    """Pull the model's FINAL numeric answer from a (chain-of-thought) completion.

    Priority, highest first -- the first source that yields a number wins:
      1. an explicit '#### <ans>' marker (gsm8k style),
      2. a '\\boxed{<ans>}' LaTeX marker,
      3. 'The answer is <ans>' / 'answer: <ans>' (case-insensitive),
      4. else the LAST number anywhere in the text.
    Returns the normalized answer string, or None if the completion is empty / has no number.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return None
    for finder in (_hash_marker, _boxed_marker, _answer_is_marker, _last_number):
        val = finder(text)
        if val is not None:
            return val
    return None


# ---- reward functions ---------------------------------------------------------------------------
def math_reward(completion, gold):
    """1.0 iff the completion's extracted final answer numerically equals `gold`, else 0.0.

    Empty / None completion -> 0.0 (no number to extract). A gold value that appears mid-reasoning
    does NOT count -- only the extracted FINAL answer is compared.
    """
    extracted = extract_final_answer(completion)
    return 1.0 if _values_equal(extracted, normalize_number(gold)) else 0.0


def score_rollout(task, completion):
    """Score one rollout against its task record; returns the settlement/attribution row:
        {"task_id", "reward": float, "extracted": str|None, "gold": str, "domain"}.

    Domain-dispatched so a future 'code' domain can be added. An unknown domain raises
    NotImplementedError (never a silent pass) -- a task with no verifier must not score as 0 reward
    and look like a merely-wrong answer.
    """
    if not isinstance(task, dict):
        raise TypeError("score_rollout: task must be a dict, got %r" % type(task).__name__)
    domain = task.get("domain", "math")
    gold = task.get("gold", "")
    task_id = task.get("task_id")
    if domain == "math":
        extracted = extract_final_answer(completion)
        reward = 1.0 if _values_equal(extracted, normalize_number(gold)) else 0.0
    else:
        raise NotImplementedError(
            "score_rollout: no reward implemented for domain %r (only 'math' is supported)"
            % (domain,))
    return {"task_id": task_id, "reward": reward, "extracted": extracted,
            "gold": gold, "domain": domain}


def batch_score(tasks_by_id, rollouts):
    """Score a batch of rollouts against a {task_id: task_record} map.

    `rollouts` is a list of {"task_id", "completion"}. A rollout whose task_id is unknown is scored
    0.0 and its id collected under "unknown_task_ids" (never crashes the batch). Returns:
        {"scores": [per-rollout dict], "n", "n_correct", "mean_reward", "unknown_task_ids": [...]}.
    """
    scores, unknown, seen_unknown = [], [], set()
    total, n_correct = 0.0, 0
    for r in rollouts:
        tid = r.get("task_id")
        completion = r.get("completion", "")
        task = tasks_by_id.get(tid)
        if task is None:
            row = {"task_id": tid, "reward": 0.0,
                   "extracted": extract_final_answer(completion),
                   "gold": None, "domain": None}
            if tid not in seen_unknown:
                seen_unknown.add(tid)
                unknown.append(tid)
        else:
            row = score_rollout(task, completion)
        scores.append(row)
        total += row["reward"]
        if row["reward"] == 1.0:
            n_correct += 1
    n = len(scores)
    return {"scores": scores, "n": n, "n_correct": n_correct,
            "mean_reward": (total / n) if n else 0.0,
            "unknown_task_ids": unknown}


# ---- convenience: load task records from a jsonl produced by glm_task_prep ----------------------
def load_tasks_by_id(path):
    """Read a tasks jsonl (one task record per line) into a {task_id: record} dict. Blank / invalid
    lines are skipped. Pure I/O helper for callers and tests; not used by the scoring functions."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict) and obj.get("task_id") is not None:
                out[obj["task_id"]] = obj
    return out
