#!/usr/bin/env python3
"""G1 RLVR ROLLOUT WORKER -- the miner "train" role for the G1 GRPO campaign
(docs/G1_PREREGISTRATION_2026-07-24.md sec 5b).

In RLVR the ROLLOUT-GENERATION step IS the compute-dominant training work, so a rollout miner is
TRAINING the model, not testing it. This worker:

  (a) loads the frozen GLM base + the CURRENT policy LoRA adapter (real GPU path -- see
      TorchGLMBackend / load_policy_backend, both lazy-torch),
  (b) for each math task samples N candidate completions,
  (c) scores each with the verifiable reward function (tools/glm_reward.py:score_rollout),
  (d) records the per-sample data the learner needs for a policy-gradient (GRPO) step -- the
      sampled tokens, the per-sample reward, and the summed sequence logprob under the CURRENT
      policy (the learner's importance ratio numerator/denominator),
  (e) signs + publishes the rollout set over the SAME content lane the deltas ride
      (tools/sharddiloco_harness.py ContentLane), signed like a contribution
      (neurahash/diloco_merge.contrib_canonical_message + neura_l1.signing).

Tasks come from tools/glm_task_prep.py records ({task_id, domain, prompt, gold, gold_raw})
distributed by tools/glm_publish_tasks.py under the lane record `sharddiloco/glm/tasks`; they are
fetched the SAME fail-closed content-addressed way the corpus is (sha256-verified, fail on
mismatch). The current adapter version is read from the lane pointer `sharddiloco/glm/pointer`.

TESTABILITY (design invariant): ALL torch/model use sits behind an injectable MODEL BACKEND seam
(the `backend.generate(...)` protocol). The pure core -- generate_rollouts / score_and_pack /
publish_rollout_set -- is stdlib-only and is unit-tested with a FAKE deterministic backend and a
FAKE lane; importing this module NEVER imports torch (glm_reward is stdlib; harness / diloco_merge
/ transformers / peft are all imported lazily inside the real-path functions). The real GPU path is
exercised by a separate `--once` smoke the operator runs later.

Env: Windows, C:/Python313/python.exe (NEVER .venv). Keep stdout ASCII (cp1252 console).
"""
import argparse
import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from glm_reward import score_rollout, extract_final_answer   # noqa: E402  (stdlib-only: re/json/math)

try:                                     # Protocol is documentation-only; never a runtime dependency
    from typing import Protocol
except ImportError:                      # pragma: no cover - py<3.8 not supported here anyway
    Protocol = object

# ---- lane object names (mirror the constants the GLM contributor/coordinator/mirror use) ---------
ROLLOUT_LANE_PREFIX = "sharddiloco/glm/rollouts"   # advert names: <prefix>/<adapter_version>/<miner>/<task_id>
TASKS_LANE_NAME = "sharddiloco/glm/tasks"          # task_seeds.json record (glm_publish_tasks.LANE_NAME)
GLM_POINTER_NAME = "sharddiloco/glm/pointer"       # current canonical adapter/state pointer
DEFAULT_WALLET_ENV = "NEURAHASH_SD_WALLET"


# ================================================================ small pure helpers
def _noop(_msg):
    return None


def _stderr_log(msg):
    """One ASCII line to stderr (cp1252 console safe)."""
    try:
        print("[glm-rollout] " + str(msg), file=sys.stderr, flush=True)
    except UnicodeEncodeError:            # a stray non-Latin-1 glyph must never crash the worker
        enc = getattr(sys.stderr, "encoding", None) or "ascii"
        print(("[glm-rollout] " + str(msg)).encode(enc, "backslashreplace").decode(enc, "replace"),
              file=sys.stderr, flush=True)


def _derive_seed(base_seed, task_id, i):
    """Deterministic per-sample seed from (base_seed, task_id, sample_index). Stable across runs and
    machines (sha256, not Python hash()), so a rollout set is reproducible given the top-level seed."""
    h = hashlib.sha256(("%s:%s:%d" % (base_seed, task_id, int(i))).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _mean_std(xs):
    """(mean, POPULATION std) of a list of floats; (0.0, 0.0) for an empty list. Population std
    (ddof=0) is the GRPO group-normalization convention -- advantage_i = (r_i - mean) / (std + eps)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = math.fsum(xs) / n
    var = math.fsum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(var)


def _canonical_bytes(obj):
    """Deterministic JSON bytes -- byte-identical to ContentLane.put_json_named's serialization
    (sort_keys + compact separators), so the sha256 of these bytes IS the content address the store
    assigns. utf-8; ensure_ascii=False keeps a prompt's unicode intact without changing the hash
    contract (both sides agree on utf-8)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def derive_glm_miner_name(address):
    """Keyless miner id derived from a wallet address: 'glm-' + first 8 hex chars (0x stripped) --
    identical to sharddiloco_glm_contributor.derive_glm_miner_name, so the coordinator binds the
    same name it recovers from the signature."""
    a = str(address)
    if a.lower().startswith("0x"):
        a = a[2:]
    return "glm-" + a[:8]


# ================================================================ (1) generate rollouts
class ModelBackend(Protocol):            # pragma: no cover - structural documentation of the seam
    """The injectable model seam. A REAL backend wraps the frozen GLM base + the current LoRA policy
    (TorchGLMBackend); a FAKE backend returns canned data so the pure core is torch-free-testable.

    generate(prompt, *, max_new_tokens, temperature, top_p, seed) -> dict with keys:
        completion_ids   : list[int]         -- the SAMPLED tokens (empty allowed)
        completion_text  : str               -- decoded completion (scored by the reward verifier)
        sample_logprobs  : list[float]|None  -- per-token logprob of each SAMPLED token under the
                                                CURRENT policy; None => provide summed form instead:
        sum_logprob      : float             -- (optional) summed sequence logprob, if not per-token
        n_tokens         : int               -- (optional) token count, if sample_logprobs is None
    Must be deterministic given `seed`.
    """
    def generate(self, prompt, *, max_new_tokens, temperature, top_p, seed):
        ...


def generate_rollouts(backend, task, n_samples, *, max_new_tokens=256, temperature=0.8,
                      top_p=0.95, seed=0):
    """Sample `n_samples` candidate completions for one `task` and score each with the verifiable
    reward (glm_reward.score_rollout). Returns a list of per-rollout dicts:

        {"task_id", "completion_ids": [int], "completion_text": str, "reward": float,
         "extracted": str|None, "sample_logprobs": [float]|None, "sum_logprob": float,
         "n_tokens": int}

    `backend` is the ModelBackend seam (it unifies the model + tokenizer the spec names separately).
    Deterministic given `seed`: sample i uses _derive_seed(seed, task_id, i). Degenerate input is
    tolerated -- n_samples<=0 -> []; a task with an empty/missing prompt or gold -> reward 0.0, no
    crash (glm_reward defaults domain='math' and scores a missing gold as a miss)."""
    task_id = task.get("task_id") if isinstance(task, dict) else None
    prompt = task.get("prompt", "") if isinstance(task, dict) else ""
    out = []
    for i in range(int(n_samples)):
        s = _derive_seed(seed, task_id, i)
        gen = backend.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                               top_p=top_p, seed=s)
        comp_ids = [int(t) for t in (gen.get("completion_ids") or [])]
        comp_text = gen.get("completion_text", "") or ""
        logps = gen.get("sample_logprobs")
        if logps is None:                                    # backend returned the summed form
            sample_logprobs = None
            sum_lp = float(gen.get("sum_logprob", 0.0))
            n_tok = int(gen.get("n_tokens", len(comp_ids)))
        else:
            sample_logprobs = [float(x) for x in logps]
            sum_lp = math.fsum(sample_logprobs)
            n_tok = len(comp_ids) or len(sample_logprobs)
        try:
            sr = score_rollout(task, comp_text)              # {task_id,reward,extracted,gold,domain}
            reward = float(sr.get("reward", 0.0))
            extracted = sr.get("extracted")
        except (NotImplementedError, TypeError, KeyError):   # unknown domain / bad task -> miss, no crash
            extracted = extract_final_answer(comp_text)
            reward = 0.0
        out.append({"task_id": task_id, "completion_ids": comp_ids, "completion_text": comp_text,
                    "reward": reward, "extracted": extracted, "sample_logprobs": sample_logprobs,
                    "sum_logprob": sum_lp, "n_tokens": n_tok})
    return out


# ================================================================ (2) pack a signable rollout set
def score_and_pack(rollouts, task, adapter_version, miner_addr):
    """Pack scored rollouts into ONE rollout-set record, ready to sign + publish, carrying enough for
    the learner's GRPO group-relative advantage (per-sample reward + summed logprob + token count):

        {"task_id", "domain", "adapter_version", "miner",
         "samples": [{"completion_ids", "reward", "sum_logprob", "n_tokens", "extracted"}],
         "reward_mean", "reward_std", "n"}

    reward_std is the POPULATION std over the group (advantage_i = (r_i - reward_mean) / reward_std).
    Robust to either rollout logprob form: a per-token `sample_logprobs` is summed; otherwise the
    rollout's own `sum_logprob`/`n_tokens` are used."""
    domain = (task.get("domain") if isinstance(task, dict) else None) or "math"
    task_id = task.get("task_id") if isinstance(task, dict) else None
    samples, rewards = [], []
    for r in rollouts:
        comp_ids = [int(t) for t in (r.get("completion_ids") or [])]
        logps = r.get("sample_logprobs")
        if logps is not None:
            sum_lp = math.fsum(float(x) for x in logps)
            n_tok = len(comp_ids) or len(logps)
        else:
            sum_lp = float(r.get("sum_logprob", 0.0))
            n_tok = int(r.get("n_tokens", len(comp_ids)))
        reward = float(r.get("reward", 0.0))
        rewards.append(reward)
        samples.append({"completion_ids": comp_ids, "reward": reward, "sum_logprob": sum_lp,
                        "n_tokens": n_tok, "extracted": r.get("extracted")})
    mean, std = _mean_std(rewards)
    return {"task_id": task_id, "domain": domain, "adapter_version": adapter_version,
            "miner": miner_addr, "samples": samples, "reward_mean": mean, "reward_std": std,
            "n": len(samples)}


# ================================================================ (3) sign + publish
def rollout_set_name(adapter_version, miner, task_id, prefix=ROLLOUT_LANE_PREFIX):
    """Enumerable, collision-free advert name: <prefix>/<adapter_version>/<miner>/<task_id>. Sharing
    the <prefix>/<adapter_version>/ (and .../<miner>/) prefix lets the learner enumerate every
    rollout set for the current adapter version straight from the store manifest; the trailing
    task_id keeps one miner's many per-task sets from overwriting each other."""
    return "%s/%s/%s/%s" % (prefix, adapter_version, miner, task_id)


def publish_rollout_set(lane, record, sign_fn, *, prefix=ROLLOUT_LANE_PREFIX, log=None):
    """Sign + publish one rollout-set `record` over the content lane, SIGNED LIKE A CONTRIBUTION.

    Flow (mirrors the GLM contributor's publish site): put_blob the canonical record bytes to get its
    content address `cid`; sign_fn(cid, adapter_version, miner) -> sig (the real signer builds
    diloco_merge.contrib_canonical_message(cid, adapter_version, miner, None, None) and secp256k1-
    signs it with the keyless wallet, exactly like _sign_contrib); then put_json_named an advert that
    references cid + sig under the enumerable name, so the learner can pull the set by content address
    and verify the signer. Returns the rollout-set `cid`, or None on any lane/sign failure (FAIL-SOFT
    -- a miner that cannot publish one set must keep going, never crash)."""
    log = log or _noop
    try:
        canonical = _canonical_bytes(record)
        cid = lane.put_blob(canonical)                       # content-address the rollout set
        av = record.get("adapter_version")
        miner = record.get("miner")
        task_id = record.get("task_id")
        name = rollout_set_name(av, miner, task_id, prefix=prefix)
        sig = sign_fn(cid, av, miner)                        # sign like a contribution (over the cid)
        advert = {"cid": cid, "sig": sig, "signer": miner, "adapter_version": av, "task_id": task_id,
                  "domain": record.get("domain"), "reward_mean": record.get("reward_mean"),
                  "reward_std": record.get("reward_std"), "n": record.get("n"), "name": name}
        lane.put_json_named(name, advert)                    # enumerable advert -> cid + sig
        return cid
    except Exception as e:                                    # noqa: BLE001 -- fail-soft by contract
        log("publish_rollout_set FAILED (%s): %r" % (record.get("task_id"), e))
        return None


# ================================================================ keyless wallet identity + signer
def default_wallet_path(wallet_file=None):
    """Where the keyless wallet identity lives: explicit path / env NEURAHASH_SD_WALLET /
    ~/.neurahash/glm_miner_key -- identical resolution to sharddiloco_glm_contributor."""
    p = (wallet_file or os.environ.get(DEFAULT_WALLET_ENV, "") or "").strip()
    return p or os.path.join(os.path.expanduser("~"), ".neurahash", "glm_miner_key")


def make_wallet_signer(wallet_file=None, log=None):
    """Load (or CREATE on first run) this miner's LOCAL secp256k1 wallet identity and return
    (account, sign_fn, miner_name). KEYLESS open admission -- mirrors _load_or_create_wallet +
    _sign_contrib (keyless branch) without importing the private-only contributor. sign_fn signs the
    canonical contribution message built from the rollout-set cid, so the coordinator ecrecovers the
    same address it binds the miner name to. Torch-free (neura_l1.signing + diloco_merge are
    numpy/stdlib); imported lazily so a test never pays for it."""
    from neura_l1.signing import account_from_key, gen_account, sign_bytes   # lazy, torch-free
    from neurahash.diloco_merge import contrib_canonical_message             # lazy, numpy-only
    log = log or _noop
    path = default_wallet_path(wallet_file)
    if os.path.isfile(path):
        acct = account_from_key(open(path).read().strip())
        log("wallet identity loaded from %s -> %s" % (path, acct.address))
    else:
        acct = gen_account()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(acct.key.hex())
        try:
            os.chmod(path, 0o600)                            # best-effort (Windows ignores modes)
        except OSError:
            pass
        log("wallet identity CREATED at %s -> %s (keyless open admission)" % (path, acct.address))

    def sign_fn(cid, adapter_version, miner):
        msg = contrib_canonical_message(cid, adapter_version, miner, None, None)
        return sign_bytes(acct, msg)

    return acct, sign_fn, derive_glm_miner_name(acct.address)


# ================================================================ fail-closed task/adapter fetch
def make_lane(url, token=""):
    """A ContentLane (tools/sharddiloco_harness) -- the exact all-outbound transport a real WAN miner
    uses. Lazy import: harness pulls numpy (not torch), and tests inject a fake lane instead."""
    import sharddiloco_harness as H                          # lazy (numpy-only)
    return H.ContentLane(url, token)


def fetch_adapter_pointer(lane, pointer_name=GLM_POINTER_NAME):
    """Read the current canonical adapter/state pointer from the lane -> the pointer dict
    ({event|round, model_root|state_cid, ...}) or None if not advertised yet. The adapter VERSION the
    rollouts are tagged with is pointer['event'] (falls back to 'round')."""
    man = lane.manifest()
    ent = man.get(pointer_name)
    if not ent:
        return None
    return lane.get_json(ent["sha256"])


def pointer_adapter_version(ptr):
    """Extract the adapter version (int-ish) from a pointer dict, tolerant of the event/round naming."""
    if not isinstance(ptr, dict):
        return None
    return ptr.get("event", ptr.get("round"))


def _fetch_content_addressed(seeds, sha, timeout=30, lane=None):
    """Fetch one object BY sha256 and VERIFY it -- FAIL-CLOSED: try the co-located store first
    (lane.get_blob verifies the CID for us), then each seed base as `<base>/o/<sha>` with an explicit
    sha256 check; raise if NOTHING serves the exact committed bytes. Identical trust model to the
    corpus fetch: a node gets exactly the pinned bytes or an error, never silently-wrong data."""
    if lane is not None:
        try:
            return lane.get_blob(sha)                        # store verifies sha internally
        except Exception:                                    # noqa: BLE001 -- fall through to seeds
            pass
    import urllib.request                                    # lazy (stdlib)
    tried = 0
    for base in (seeds or []):
        tried += 1
        url = base.rstrip("/") + "/o/" + sha
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                body = r.read()
        except Exception:                                    # noqa: BLE001 -- try the next seed
            continue
        if hashlib.sha256(body).hexdigest() == sha:
            return body
    raise RuntimeError("fail-closed: no source served sha256 %s (tried %d seed(s))" % (sha, tried))


def fetch_task_shard(lane, tasks_lane_name=TASKS_LANE_NAME, seeds=None, timeout=30, max_tasks=None):
    """Fetch + parse the MINER-FACING train task shard the SAME fail-closed way the corpus is
    fetched: read the `sharddiloco/glm/tasks` record ({manifest_sha256, seeds, files}); for each
    tasks_*_train.jsonl file, fetch its bytes by sha256 (sha-verified), and parse one task record per
    line. Only tasks_*_train.jsonl are consumed -- the frozen coordinator-only eval file is never
    named in this record and would be refused anyway. Returns a list of task dicts."""
    man = lane.manifest()
    ent = man.get(tasks_lane_name)
    if not ent:
        raise RuntimeError("no task-lane record '%s' advertised yet" % tasks_lane_name)
    rec = lane.get_json(ent["sha256"])
    files = rec.get("files", {})
    seed_bases = seeds if seeds is not None else rec.get("seeds", [])
    tasks = []
    for name in sorted(files):
        if not (name.startswith("tasks_") and name.endswith("_train.jsonl")):
            continue                                         # miner-facing TRAIN only; never eval
        sha = files[name]["sha256"]
        body = _fetch_content_addressed(seed_bases, sha, timeout=timeout, lane=lane)
        for line in body.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict) and obj.get("task_id") is not None:
                tasks.append(obj)
                if max_tasks and len(tasks) >= int(max_tasks):
                    return tasks
    return tasks


# ================================================================ real GPU backend (lazy torch)
class TorchGLMBackend(object):
    """REAL model backend: manual autoregressive sampling over any HF-style causal LM whose forward
    returns `.logits` (the GLM lite/full model, tokenizer-free ids OR a real tokenizer). Records the
    per-token logprob of each SAMPLED token under the UNTEMPERED policy softmax (log pi(token|ctx)) --
    the quantity the learner recomputes with the new params for the GRPO importance ratio; temperature
    + top_p shape only the SAMPLING (behavior) distribution, not the recorded logprob. Not covered by
    the unit tests (torch); the operator's --once smoke exercises it and may fix it mid-run (the
    pre-registration permits signed CLIENT-code fixes)."""

    def __init__(self, model, tokenizer=None, device=None, eos_id=None, max_prompt_tokens=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eos_id = eos_id
        self.max_prompt_tokens = max_prompt_tokens

    def _encode(self, prompt):
        if isinstance(prompt, (list, tuple)):                # already token ids
            return [int(t) for t in prompt]
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(prompt)
            return list(ids)[-self.max_prompt_tokens:]
        raise ValueError("TorchGLMBackend: text prompt needs a tokenizer (or pass token ids)")

    def _decode(self, ids):
        if self.tokenizer is not None:
            try:
                return self.tokenizer.decode(ids)
            except Exception:                                # noqa: BLE001
                return " ".join(str(t) for t in ids)
        return " ".join(str(t) for t in ids)

    def generate(self, prompt, *, max_new_tokens, temperature, top_p, seed):
        import torch                                         # lazy: real path only
        import torch.nn.functional as F
        eos = self.eos_id
        if eos is None and self.tokenizer is not None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        dev = self.device or next(self.model.parameters()).device
        gen = torch.Generator(device=str(self.device) if str(self.device).startswith("cuda") else "cpu")
        gen.manual_seed(int(seed) & 0x7FFFFFFF)
        prompt_ids = self._encode(prompt)
        ids_t = torch.as_tensor([prompt_ids], dtype=torch.long, device=dev)

        comp_ids, logps = [], []
        past = None
        with torch.no_grad():
            step_in = ids_t
            for _ in range(int(max_new_tokens)):
                try:
                    out = self.model(input_ids=step_in, past_key_values=past, use_cache=True)
                    past = getattr(out, "past_key_values", None)
                except TypeError:                            # model without KV-cache kwargs
                    out = self.model(input_ids=ids_t)
                    past = None
                logits_row = out.logits[0, -1, :].float()    # [V]
                logp_full = F.log_softmax(logits_row, dim=-1)
                tok = _sample_token(logits_row, temperature, top_p, gen)
                comp_ids.append(int(tok))
                logps.append(float(logp_full[tok].item()))
                nxt = torch.as_tensor([[tok]], dtype=torch.long, device=dev)
                ids_t = torch.cat([ids_t, nxt], dim=1)
                step_in = nxt if past is not None else ids_t
                if eos is not None and int(tok) == int(eos):
                    break
        return {"completion_ids": comp_ids, "completion_text": self._decode(comp_ids),
                "sample_logprobs": logps}


def _sample_token(logits_row, temperature, top_p, generator):
    """Sample one token id from `logits_row` [V] with temperature + top_p (nucleus). temperature<=0 =>
    greedy argmax. Returns an int token id."""
    import torch                                             # lazy
    import torch.nn.functional as F
    if not temperature or temperature <= 0:
        return int(torch.argmax(logits_row).item())
    probs = F.softmax(logits_row / float(temperature), dim=-1)
    if top_p and 0.0 < float(top_p) < 1.0:
        sp, si = torch.sort(probs, descending=True)
        csum = torch.cumsum(sp, dim=-1)
        keep = (csum - sp) < float(top_p)                    # keep tokens up to & incl. the crossing one
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    total = float(probs.sum().item())
    if total <= 0.0:                                         # degenerate (e.g. -inf logits) -> greedy
        return int(torch.argmax(logits_row).item())
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator).item())


def load_policy_backend(args, ptr, log=None):
    """REAL GPU path: build the frozen GLM base + apply the CURRENT policy LoRA adapter, return a
    TorchGLMBackend. Lazy (transformers/peft/torch). This is the operator-facing seam the --once smoke
    drives and may fix mid-run; kept small and defensive on purpose. `ptr` is the lane pointer (its
    model_root/adapter reference names the current adapter to overlay)."""
    log = log or _stderr_log
    import torch                                             # lazy
    from transformers import AutoTokenizer                   # lazy
    base_dir = getattr(args, "shard_dir", None) or getattr(args, "base_dir", None)
    if not base_dir:
        raise RuntimeError("load_policy_backend: no --shard-dir / base model dir given")
    cfg_dir = getattr(args, "config_dir", None) or base_dir
    device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    # HARD VRAM CAP -- set for EVERY load path BEFORE the first CUDA allocation (project rule
    # vram-cap-live-verified; 2026-07-24 incident: an uncapped load took 32,077/32,607 MiB,
    # starved the display, crashed the box). Honors NEURAHASH_VRAM_CAP_GB; default leaves ~8 GiB
    # for the desktop / a co-resident CE miner.
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1 << 30) \
        if torch.cuda.is_available() else 0.0
    cap_gib = float(os.environ.get("NEURAHASH_VRAM_CAP_GB", "0") or 0) or max(4.0, total_gib - 8.0)
    if total_gib > 0:
        cap_gib = min(cap_gib, max(1.0, total_gib - 2.0))
        torch.cuda.set_per_process_memory_fraction(cap_gib / total_gib, 0)   # hard backstop
    # FULL-MODEL mode (the real G1 rollout policy): the whole 47-layer GLM. MEASURED 2026-07-24:
    # bf16 is 59 GiB on disk, so NO consumer card holds it; bnb "4-bit" does NOT rescue it (bnb
    # quantizes only nn.Linear -- the FUSED expert modules stay bf16 -> 32 GiB, crashed the box)
    # and bnb also REFUSES CPU offload of quantized modules (ValueError, measured). So this branch
    # loads PLAIN bf16 with accelerate offload: cap_gib on GPU, NEURAHASH_CPU_OFFLOAD_GB in RAM,
    # the remainder spilled to disk. Box-safe but SLOW -- a bootstrap for big-RAM operators only.
    # The miner-grade rollout engine is fleet-hosted PIPELINE generation across cards (the proven
    # r2-prod pattern): ~57 GiB of layers spread over many 8 GiB cards -- i.e. MORE MINERS.
    if getattr(args, "full_model", False):
        from transformers import AutoModelForCausalLM                    # lazy
        cpu_cap_gib = float(os.environ.get("NEURAHASH_CPU_OFFLOAD_GB", "26"))
        off_dir = os.environ.get("NEURAHASH_OFFLOAD_DIR") or os.path.join(
            os.path.dirname(os.path.abspath(str(base_dir))), "_glm_offload")
        os.makedirs(off_dir, exist_ok=True)
        max_mem = {"cpu": "%dGiB" % int(cpu_cap_gib)}
        if torch.cuda.is_available():
            max_mem[0] = "%dGiB" % int(cap_gib)
        log("loading FULL GLM bf16 (offloaded) from %s | VRAM cap %.1f of %.1f GiB, CPU cap %.0f "
            "GiB, disk spill -> %s (slow bootstrap path)"
            % (cfg_dir, cap_gib, total_gib, cpu_cap_gib, off_dir))
        model = AutoModelForCausalLM.from_pretrained(
            cfg_dir, device_map="auto", max_memory=max_mem, offload_folder=off_dir,
            local_files_only=True, dtype=torch.bfloat16).eval()
        tok = AutoTokenizer.from_pretrained(cfg_dir, local_files_only=True)
        used = torch.cuda.memory_allocated(0) / (1 << 30) if torch.cuda.is_available() else 0.0
        log("full model resident under cap (GPU %.1f GiB used); adapter overlay lands with the "
            "learner service" % used)
        return TorchGLMBackend(model, tokenizer=tok, device=device)
    # PARTIAL mode: the lane's GLM never loads via AutoModelForCausalLM (the shard dir is a piece
    # manifest, and the full 31B bf16 would not fit a consumer card). Load EXACTLY the way every
    # other lane component does: piece_loader partial model -- frozen trunk (strip-MTP) + resident
    # expert piece(s) -- the model-in-kind whose CE the lane gates and pays on.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import piece_loader as pl                                # lazy (lane-standard loader)
    pieces = [int(x) for x in str(getattr(args, "pieces", "") or "0").split(",") if str(x).strip() != ""]
    log("loading frozen GLM base via piece_loader: shard=%s pieces=%s device=%s (strip_mtp)"
        % (base_dir, pieces, device))
    model, summ = pl.build_partial_model(base_dir, pieces, device=device, config_dir=cfg_dir,
                                         strip_mtp=True)
    model.eval()
    log("resident: layers=%s experts=%s" % (summ.get("resident_layers"), summ.get("n_resident_experts")))
    # CAPACITY GATE (measured 2026-07-24): a truncated stack (trunk + a few resident layers) rolls
    # out a broken policy -- reward 0.0 on GSM8K, ZERO learning signal -- so partial rollouts are
    # waste, not training. Rollout duty therefore needs the FULL stack resident (every layer, no
    # placeholder experts). Smaller cards keep earning on the CE lane (the contributor); the
    # fleet-hosted pipeline path lifts this gate as aggregate fleet VRAM grows.
    n_layers_total = None
    try:
        with open(os.path.join(cfg_dir, "config.json"), "r", encoding="utf-8") as f:
            n_layers_total = int(json.load(f).get("num_hidden_layers") or 0) or None
    except (OSError, ValueError, TypeError):
        pass
    resident = set(summ.get("resident_layers") or [])
    full_stack = (n_layers_total is not None and len(resident) >= n_layers_total
                  and not summ.get("partial_layers")
                  and int(summ.get("n_placeholder_experts") or 0) == 0)
    if not full_stack and not getattr(args, "allow_partial", False):
        raise SystemExit(
            "CAPACITY: resident stack is %d/%s layers -- a truncated policy scores reward 0.0 "
            "(measured 2026-07-24), so rollout duty needs the full model (59 GiB bf16). Keep this "
            "card on CE-lane training (sharddiloco_glm_contributor.py); use --full-model on a "
            "big-RAM box, or --allow-partial for smoke tests only."
            % (len(resident), n_layers_total if n_layers_total is not None else "?"))
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(cfg_dir, local_files_only=True)
    except Exception as e:                                   # noqa: BLE001 -- tokenizer-free models are valid
        log("no tokenizer at config dir (%r) -- assuming token-id prompts" % (e,))
    # Current policy adapter: applied via the lane's own LoRA delta path once the learner publishes
    # one (adapter_version > genesis). At genesis there is no adapter -- roll out the BASE policy,
    # which is exactly arm A of the pre-registration.
    adapter_ref = ptr.get("adapter_dir") if isinstance(ptr, dict) else None
    if adapter_ref:
        log("NOTE: adapter overlay wiring lands with the learner service; rolling out BASE policy for now")
    else:
        log("no adapter published yet (genesis) -- rolling out the BASE policy")
    return TorchGLMBackend(model, tokenizer=tok, device=device)


# ================================================================ (4) run loop scaffold
def run_worker(args, *, lane=None, backend=None, sign_fn=None, miner=None, tasks=None,
               fetch_tasks_fn=None, fetch_pointer_fn=None, publish_fn=publish_rollout_set, log=None):
    """Fetch the current task shard + adapter-version pointer, then for each task
    generate -> score_and_pack -> sign -> publish. Every external dependency (lane, backend, signer,
    task source, publisher) is an injectable seam with a real default, so the loop is testable with
    fakes AND runs for real over WAN. Returns the number of tasks processed.

    Real defaults: lane = ContentLane(--url,--token); identity = keyless wallet
    (~/.neurahash/glm_miner_key); tasks = fail-closed fetch of `sharddiloco/glm/tasks`; backend =
    load_policy_backend (frozen base + current adapter)."""
    log = log or _stderr_log
    if lane is None:
        lane = make_lane(args.url, getattr(args, "token", "") or "")
    if sign_fn is None:
        _acct, sign_fn, miner = make_wallet_signer(getattr(args, "wallet_file", None), log=log)
    elif miner is None:
        miner = getattr(args, "miner", None) or "glm-anon"

    ptr = (fetch_pointer_fn or fetch_adapter_pointer)(lane)
    adapter_version = pointer_adapter_version(ptr)
    log("current adapter_version = %s" % adapter_version)

    if tasks is None:
        if fetch_tasks_fn is not None:
            tasks = fetch_tasks_fn(lane)
        else:
            tasks = fetch_task_shard(lane, max_tasks=getattr(args, "max_tasks", None))
    log("fetched %d task(s)" % len(tasks))

    if backend is None:
        backend = load_policy_backend(args, ptr, log=log)

    n = 0
    max_tasks = getattr(args, "max_tasks", None)
    for task in tasks:
        base_seed = _derive_seed(getattr(args, "seed", 0), task.get("task_id"), 0)
        rollouts = generate_rollouts(backend, task, getattr(args, "n_samples", 8),
                                     max_new_tokens=getattr(args, "max_new_tokens", 256),
                                     temperature=getattr(args, "temperature", 0.8),
                                     top_p=getattr(args, "top_p", 0.95), seed=base_seed)
        record = score_and_pack(rollouts, task, adapter_version, miner)
        cid = publish_fn(lane, record, sign_fn)
        log("task %s: n=%d reward_mean=%.4f reward_std=%.4f cid=%s"
            % (task.get("task_id"), record["n"], record["reward_mean"], record["reward_std"], cid))
        n += 1
        if max_tasks and n >= int(max_tasks):
            break
        if getattr(args, "once", False):
            break
    log("done: processed %d task(s) for adapter_version %s" % (n, adapter_version))
    return n


# ================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="glm_rollout_worker.py",
                                 description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("NEURAHASH_CONTENT_URL", ""),
                    help="content-store base URL (the lane the deltas + tasks ride)")
    ap.add_argument("--token", default=os.environ.get("NEURAHASH_CONTENT_TOKEN", ""),
                    help="content-store X-Auth token (or set NEURAHASH_CONTENT_TOKEN)")
    ap.add_argument("--shard-dir", dest="shard_dir", default=None,
                    help="frozen GLM base model dir (the piece/checkpoint root the backend loads)")
    ap.add_argument("--config-dir", dest="config_dir", default=None,
                    help="optional dir holding run config (adapter surface, chat template, etc.)")
    ap.add_argument("--tasks-dir", dest="tasks_dir", default=None,
                    help="optional local task dir; when set, tasks are read from here instead of the lane")
    ap.add_argument("--n-samples", dest="n_samples", type=int, default=8,
                    help="rollouts sampled per task (G1 GRPO cap is 8; default %(default)s)")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=256,
                    help="max sampled tokens per rollout (default %(default)s)")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="sampling temperature for rollout exploration (default %(default)s)")
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95,
                    help="nucleus top-p for rollout sampling (default %(default)s)")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    ap.add_argument("--wallet-file", dest="wallet_file", default=None,
                    help="LOCAL secp256k1 keyless wallet path (created on first run; default "
                         "NEURAHASH_SD_WALLET or ~/.neurahash/glm_miner_key)")
    ap.add_argument("--seed", type=int, default=0, help="top-level rollout seed (default %(default)s)")
    ap.add_argument("--max-tasks", dest="max_tasks", type=int, default=None,
                    help="stop after this many tasks (default: all in the shard)")
    ap.add_argument("--once", action="store_true", help="process a single task then stop (smoke)")
    ap.add_argument("--pieces", default="0",
                    help="comma-separated resident piece ids for the partial-model path "
                         "(default %(default)s)")
    ap.add_argument("--allow-partial", dest="allow_partial", action="store_true",
                    help="permit rollouts from a TRUNCATED resident stack (smoke/testing only: a "
                         "partial policy scores ~0 reward, so it produces no training signal)")
    ap.add_argument("--full-model", dest="full_model", action="store_true",
                    help="load the FULL GLM bf16 with a hard VRAM cap + CPU/disk offload (59 GiB "
                         "on disk -- slow bootstrap for big-RAM boxes; the fleet answer is "
                         "pipeline rollouts across many cards)")
    return ap


def _load_tasks_dir(tasks_dir, max_tasks=None):
    """Read tasks_*_train.jsonl task records from a LOCAL dir (offline / --tasks-dir path). Same
    train-only discipline as the lane fetch."""
    tasks = []
    for name in sorted(os.listdir(tasks_dir)):
        if not (name.startswith("tasks_") and name.endswith("_train.jsonl")):
            continue
        with open(os.path.join(tasks_dir, name), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict) and obj.get("task_id") is not None:
                    tasks.append(obj)
                    if max_tasks and len(tasks) >= int(max_tasks):
                        return tasks
    return tasks


def main(argv=None):
    args = build_parser().parse_args(argv)
    # v3.2.1: signed auto-update at startup (same fail-closed, pinned-key, 6h-rate-limited check the
    # contributor runs; opt-out NEURAHASH_AUTOUPDATE=off). A verified forward release re-execs us
    # BEFORE the heavy model load; any failure just continues on current code.
    try:
        from self_update import check_and_update                 # lazy; sys.path has _HERE
        check_and_update()
    except Exception as e:                                       # noqa: BLE001 -- never block rollouts
        _stderr_log("auto-update check failed (%r); continuing on current code" % (e,))
    if not args.url and not args.tasks_dir:
        raise SystemExit("ERROR: give --url (lane) or --tasks-dir (local tasks); neither was set.")
    fetch_tasks_fn = None
    if args.tasks_dir:
        fetch_tasks_fn = lambda _lane: _load_tasks_dir(args.tasks_dir, max_tasks=args.max_tasks)  # noqa: E731
    run_worker(args, fetch_tasks_fn=fetch_tasks_fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
