"""G1 RLVR rollout worker -- tools/glm_rollout_worker.py (see docs/G1_PREREGISTRATION_2026-07-24.md
sec 5b). Exercises the PURE core (generate_rollouts / score_and_pack / publish_rollout_set) with a
FAKE deterministic model backend and a FAKE content lane: ZERO torch, zero network, pure stdlib.
The final test asserts torch was never imported, guaranteeing the miner "train" role's data path is
CPU-testable (the real GPU sampling path is a separate --once smoke the operator runs).

Run: C:/Python313/python.exe -m pytest tests/test_glm_rollout_worker.py -q
"""
import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glm_rollout_worker as W      # noqa: E402
import glm_reward                   # noqa: E402  (the reward oracle the worker must agree with)


# ---- a gold-answer math task + canned completions (one correct '42', two wrong) -----------------
GOLD_TASK = {"task_id": "abc123", "domain": "math", "prompt": "What is 6 * 7?",
             "gold": "42", "gold_raw": "#### 42"}

# (completion_text, completion_ids, per-token sample_logprobs)
CORRECT = ("The answer is 42", [1, 2, 3], [-0.1, -0.2, -0.3])
WRONG_41 = ("I think it is 41", [4, 5], [-0.5, -0.6])
WRONG_100 = ("no idea, maybe 100", [6], [-0.9])


class FakeBackend(object):
    """Deterministic canned backend implementing the ModelBackend seam. Sample i (call order within
    one generate_rollouts call) -> completions[i % len]. Records every generate() call so a test can
    assert seed determinism and that the sampling args were threaded through. NO torch."""

    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = []

    def generate(self, prompt, *, max_new_tokens, temperature, top_p, seed):
        text, ids, logps = self.completions[len(self.calls) % len(self.completions)]
        self.calls.append({"prompt": prompt, "seed": seed, "temperature": temperature,
                           "top_p": top_p, "max_new_tokens": max_new_tokens})
        return {"completion_ids": list(ids), "completion_text": text, "sample_logprobs": list(logps)}


class FakeLane(object):
    """Captures put_blob / put_json_named calls; content-addresses exactly like ContentLane so cids
    match. `fail_put_blob` simulates a store outage to exercise publish_rollout_set's fail-soft path."""

    def __init__(self, fail_put_blob=False):
        self.blobs = {}
        self.named = {}
        self.put_blob_calls = []
        self.put_json_named_calls = []
        self.fail_put_blob = fail_put_blob

    def put_blob(self, body, name=None):
        if self.fail_put_blob:
            raise RuntimeError("store down")
        cid = hashlib.sha256(body).hexdigest()
        self.blobs[cid] = body
        self.put_blob_calls.append((cid, name, body))
        return cid

    def put_json_named(self, name, obj):
        body = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cid = hashlib.sha256(body).hexdigest()
        self.named[name] = obj
        self.put_json_named_calls.append((name, obj))
        return cid


# ---- (1) generate_rollouts: N samples, rewards agree with glm_reward ----------------------------
def test_generate_rollouts_n_samples_and_reward_agreement():
    comps = [CORRECT, WRONG_41, WRONG_100]
    be = FakeBackend(comps)
    rolls = W.generate_rollouts(be, GOLD_TASK, n_samples=3, max_new_tokens=8,
                                temperature=0.7, top_p=0.9, seed=123)
    assert len(rolls) == 3
    for r, (text, ids, logps) in zip(rolls, comps):
        assert r["task_id"] == GOLD_TASK["task_id"]
        assert r["completion_text"] == text
        assert r["completion_ids"] == ids
        assert r["sample_logprobs"] == [float(x) for x in logps]
        # the worker's reward must EQUAL the reward oracle on the same completion/gold
        assert r["reward"] == glm_reward.math_reward(text, GOLD_TASK["gold"])
        assert r["extracted"] == glm_reward.extract_final_answer(text)
        assert abs(r["sum_logprob"] - sum(logps)) < 1e-12
        assert r["n_tokens"] == len(ids)


# ---- (2) correct completion == gold -> 1.0 ; wrong -> 0.0 ---------------------------------------
def test_correct_is_one_wrong_is_zero():
    be = FakeBackend([CORRECT, WRONG_41])
    rolls = W.generate_rollouts(be, GOLD_TASK, n_samples=2, seed=1)
    assert rolls[0]["reward"] == 1.0     # "The answer is 42" == gold 42
    assert rolls[1]["reward"] == 0.0     # "...41" != 42


# ---- (3) reward_mean / reward_std computed correctly (population std) ----------------------------
def test_reward_mean_and_std():
    # extracted answers: 42 (hit), 7 (miss), 9 (miss) -> rewards [1, 0, 0]
    comps = [("#### 42", [1], [-0.1]), ("#### 7", [2], [-0.2]), ("#### 9", [3], [-0.3])]
    be = FakeBackend(comps)
    rolls = W.generate_rollouts(be, GOLD_TASK, n_samples=3, seed=5)
    rec = W.score_and_pack(rolls, GOLD_TASK, adapter_version=5, miner_addr="glm-deadbeef")
    assert rec["n"] == 3
    assert abs(rec["reward_mean"] - (1.0 / 3.0)) < 1e-12
    m = 1.0 / 3.0
    exp_std = math.sqrt(((1 - m) ** 2 + (0 - m) ** 2 + (0 - m) ** 2) / 3.0)
    assert abs(rec["reward_std"] - exp_std) < 1e-12


# ---- (4) packed record carries every GRPO field the learner needs -------------------------------
def test_packed_record_has_grpo_fields():
    comps = [CORRECT, WRONG_41]
    be = FakeBackend(comps)
    rolls = W.generate_rollouts(be, GOLD_TASK, n_samples=2, seed=7)
    rec = W.score_and_pack(rolls, GOLD_TASK, adapter_version=3, miner_addr="glm-abcd1234")
    assert rec["task_id"] == GOLD_TASK["task_id"]
    assert rec["domain"] == "math"
    assert rec["adapter_version"] == 3
    assert rec["miner"] == "glm-abcd1234"
    assert rec["n"] == 2
    assert len(rec["samples"]) == 2
    for s, (text, ids, logps) in zip(rec["samples"], comps):
        assert s["completion_ids"] == ids                    # sampled tokens (learner recomputes new logprob)
        assert isinstance(s["reward"], float)                # per-sample reward (group-relative advantage)
        assert abs(s["sum_logprob"] - sum(logps)) < 1e-12    # old-policy sequence logprob (importance ratio)
        assert s["n_tokens"] == len(ids)                     # token count
        assert "extracted" in s
    json.dumps(rec)                                          # must be JSON-serializable (publishable)


# ---- (5) publish_rollout_set: signs, put_blob + put_json_named under the right name --------------
def test_publish_rollout_set_signs_and_names():
    lane = FakeLane()
    rec = {"task_id": "t1", "domain": "math", "adapter_version": 9, "miner": "glm-cafebabe",
           "samples": [{"completion_ids": [1], "reward": 1.0, "sum_logprob": -0.5,
                        "n_tokens": 1, "extracted": "42"}],
           "reward_mean": 1.0, "reward_std": 0.0, "n": 1}
    seen = {}

    def sign_fn(cid, adapter_version, miner):
        seen["args"] = (cid, adapter_version, miner)
        return "sig:" + cid[:8]

    cid = W.publish_rollout_set(lane, rec, sign_fn)
    exp_cid = hashlib.sha256(W._canonical_bytes(rec)).hexdigest()
    assert cid == exp_cid                                    # returns the rollout-set content address
    assert len(lane.put_blob_calls) == 1                    # blob'd the signed set
    assert len(lane.put_json_named_calls) == 1              # advertised it once
    name, advert = lane.put_json_named_calls[0]
    assert name == "sharddiloco/glm/rollouts/9/glm-cafebabe/t1"   # enumerable per adapter_version/miner
    assert advert["cid"] == cid
    assert advert["sig"] == "sig:" + cid[:8]
    assert advert["signer"] == "glm-cafebabe"
    assert advert["adapter_version"] == 9 and advert["task_id"] == "t1"
    # signed LIKE A CONTRIBUTION: sign_fn was handed the cid + adapter_version + miner, not raw bytes
    assert seen["args"] == (cid, 9, "glm-cafebabe")


def test_publish_rollout_set_is_fail_soft():
    lane = FakeLane(fail_put_blob=True)
    rec = {"task_id": "t", "domain": "math", "adapter_version": 1, "miner": "glm-x",
           "samples": [], "reward_mean": 0.0, "reward_std": 0.0, "n": 0}
    out = W.publish_rollout_set(lane, rec, lambda *a: "sig")
    assert out is None                                       # store outage -> None, no exception


# ---- (6) determinism given a seed ---------------------------------------------------------------
def test_determinism_given_seed():
    comps = [CORRECT, WRONG_41, WRONG_100]
    be1, be2 = FakeBackend(list(comps)), FakeBackend(list(comps))
    r1 = W.generate_rollouts(be1, GOLD_TASK, 3, seed=42)
    r2 = W.generate_rollouts(be2, GOLD_TASK, 3, seed=42)
    assert r1 == r2                                          # same seed -> identical rollout set
    seeds1 = [c["seed"] for c in be1.calls]
    seeds2 = [c["seed"] for c in be2.calls]
    assert seeds1 == seeds2                                  # per-sample seeds are deterministic
    assert len(set(seeds1)) == 3                             # and distinct per sample index
    be3 = FakeBackend(list(comps))
    W.generate_rollouts(be3, GOLD_TASK, 3, seed=99)
    assert [c["seed"] for c in be3.calls] != seeds1          # a different top-level seed -> different seeds
    # default sampling knobs threaded through untouched
    assert be1.calls[0]["temperature"] == 0.8 and be1.calls[0]["top_p"] == 0.95


# ---- (7) empty / degenerate tasks handled (no crash) --------------------------------------------
def test_degenerate_inputs_no_crash():
    be = FakeBackend([("#### 42", [1], [-0.1])])
    assert W.generate_rollouts(be, GOLD_TASK, 0, seed=1) == []          # n_samples=0 -> []
    empty_task = {"task_id": "e1", "domain": "math", "prompt": "", "gold": ""}
    rolls = W.generate_rollouts(be, empty_task, 1, seed=1)             # empty gold -> reward 0, no crash
    assert len(rolls) == 1 and rolls[0]["reward"] == 0.0
    rec = W.score_and_pack(rolls, empty_task, adapter_version=None, miner_addr="glm-0")
    assert rec["n"] == 1 and rec["reward_mean"] == 0.0
    rec0 = W.score_and_pack([], GOLD_TASK, 1, "glm-0")                 # empty rollout list -> zeros
    assert rec0["n"] == 0 and rec0["reward_mean"] == 0.0 and rec0["reward_std"] == 0.0
    no_domain = {"task_id": "n1", "prompt": "2+2?", "gold": "4"}       # missing domain -> defaults, no crash
    assert W.generate_rollouts(be, no_domain, 1, seed=2)[0]["reward"] in (0.0, 1.0)


# ---- structural dry run mirroring acceptance criterion 2 ----------------------------------------
def test_dry_run_reward_mean_is_one_third():
    task = {"task_id": "dry", "domain": "math", "prompt": "6*7?", "gold": "42", "gold_raw": "#### 42"}
    comps = [("Let me compute.\n#### 42", [10, 11, 12], [-0.2, -0.3, -0.4]),   # correct
             ("#### 41", [20], [-0.7]),                                          # wrong
             ("#### 7", [30, 31], [-0.5, -0.6])]                                 # wrong
    be = FakeBackend(comps)
    rolls = W.generate_rollouts(be, task, 3, max_new_tokens=32, temperature=0.8, top_p=0.95, seed=2026)
    rec = W.score_and_pack(rolls, task, adapter_version=1, miner_addr="glm-drytest")
    assert abs(rec["reward_mean"] - (1.0 / 3.0)) < 1e-6
    for s in rec["samples"]:
        assert "sum_logprob" in s and "n_tokens" in s


# ---- torch was NEVER imported by any of the above (CPU-testable data path) -----------------------
def test_zz_no_torch_imported():
    """Fresh-interpreter check: importing the worker must not pull any heavy lib. Subprocess-based
    because in-process sys.modules is order-dependent across the suite (other test files import
    torch legitimately -- measured 2026-07-24, same flake class as the 61eb234 fix)."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import glm_rollout_worker; "
            "bad = [m for m in ('torch', 'transformers', 'peft') if m in sys.modules]; "
            "print('heavy:', bad); sys.exit(1 if bad else 0)" % os.path.join(_REPO, "tools"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, "worker import pulled heavy libs: %s %s" % (r.stdout, r.stderr)
