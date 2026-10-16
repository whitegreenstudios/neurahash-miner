"""WAN-GRADIENT: the first arm in this repo where a gradient-derived weight slice crosses the
public internet and ends up INSIDE the scored weights.

Design (frozen, read it first): docs/research/WAN_GRADIENT_SURFACE_PARALLEL_DESIGN_2026-09-06.md
  section 2  -- 8 GB residency on the 4060 and the fused-expert trap
  section 5  -- the pre-registered arms, instruments, VOID set, KILL
  section 6  -- cost
  section 7  -- the three first-hour gates (a) footprint (b) base-consistency (c) H=1 tracking
This file is the RUNNER the design never had. Nothing here is launched by writing it; the owner
reviews `--plan` output before any GPU spend. Written 2026-09-08 (Fable subagent).

WHAT CHANGED FROM THE DESIGN, AND WHY (owner directive 2026-09-07 "do not repeat testing the
same thing", applied with the prior-work search of 2026-09-08):
  * C-SOLO is CUT. It is byte-identical in every field to the banked D1-TB pair
    (E:/rsft_runs/stylectl/d1tb, d1tb_s2: same corpus sha 72072237f341c247, same seeds
    20260829/20260830, seq 1024, 625 steps, 4,194,304 trainable, OneCycle 1e-4) which reads
    battery +0.9951 / +0.7646 pp at MDE 0.7261 / 0.7391, with per-step losses (625 each) and
    per-item battery/GSM8K-1319/ARC files on disk. A fresh C-SOLO cannot beat that resolution.
    The banked pair IS the control twin; this chain re-derives its deltas from the per-item files
    at preflight (design P1: +0.6..+1.5 pp, else VOID) and runs a fresh A/A on the base so the
    banked per-item files are proven current before any arm is paired against them.
  * The design's section 0 sentence "the only OLMoE delta that ever crossed with a capability
    instrument attached returned -1.44" is wrong: that arm was GLM-4.7-Flash
    (docs/RESULTS_LEDGER.md:93; FLEET_AS_LEARNER_DESIGN_2026-08-26.md:2-8 correction). No OLMoE
    weight delta has ever crossed any network with a capability readout. Every treatment arm here
    is therefore un-banked (prior-work Q1/Q2/Q6: no hits).
  * The 4060 has NO OLMoE base on disk today (probe 2026-09-08: HF cache holds Qwen3-1.7B only).
    The stage role fetches the TRUE base by outbound HF download on the box itself (doctrine:
    memory multi-miner-runpod-lessons) and refuses to proceed unless all three shards match the
    5090's full-file sha256 (the Instruct-under-a-base-name trap that cost six days).
  * The CLIP is the twin's (review 2026-09-08 HIGH-2). d1_sft.py clips the 16-layer GLOBAL norm;
    a per-slice clip is only identical while that norm stays <= 1.0, and the "0.09-0.36 measured"
    claim this file used to carry could not have come from the twin (its train logs carry no gnorm
    field; the only 10k-row d1_sft runs that log it reach 0.440 and 2.717). Each box now computes
    the 16-layer pre-clip norm from its own view (peer slice's gradient measured, never stepped),
    applies that factor to its owned slice, and records per step: owned norm, global norm, peer
    norm, factor applied -- see clip_step() and the manifest's "clip" record.

BASE POLICY (owner directive 2026-09-08 ~22:30, verbatim "go straight to both nf4 after the
15mins"; his challenge: "why u make both card train in different quantize? where do u learn that
and what prove u have that it will work?" -- no proof exists: the prior-work search of 2026-09-08
found no nf4-vs-bf16 gap ever measured on OLMoE and no arm ever trained on an nf4 base). The
design's section-2 coupling -- 5090 on the bf16 base, 4060 on the nf4 base -- is now ONE policy of
two: `--base-policy mixed` (the design as registered; the DEFAULT, so every existing launch line
is byte-unchanged) and `--base-policy nf4` (both boxes on the SAME bitsandbytes NF4 base). Base
dtype is decoupled from box identity: every role takes `--base-dtype {auto,bf16,nf4}`, init_meta
carries base_policy + base_dtype and the 4060 VOIDs at step 0 on any mismatch
(init_meta_problem), every manifest records the dtype actually loaded, and the expert-layout
assertion runs on WHICHEVER box loads nf4. Both-nf4 has a hardware precondition this file cannot
wish away: the 5090's C:/Python313 carries transformers 5.8.1, whose OLMoE experts are ONE fused
3-D nn.Parameter per layer (gate_up_proj / down_proj, modeling_olmoe.py:310) that bitsandbytes'
Linear4bit replacement never touches -- probed on the meta device 2026-09-08: 0 of 3072 expert
projections quantizable, 32 fused params. Under that interpreter a "4-bit" load leaves ~6.9B of
6.9B expert weights in bf16 and quantizes only the attention, so the nf4 policy REQUIRES a
transformers-4.x interpreter on the 5090 as well (`--py5090`, probed in preflight exactly as
`--py4060` is; recipe in TF4_ENV_RECIPE). A 5090 that cannot produce the unfused nf4 base VOIDs
loudly; it never trains bf16 under an nf4 label. And both-nf4 moves the quantization confound from
4 of 16 layers INSIDE the arm to all 16 layers AGAINST the bf16 control twin, so under the nf4
policy the control is the nf4 twin (SPEC["twin_nf4"], produced by scratchpad/nf4_twin_chain.py
via d1_sft.py --nf4-base: the banked recipe with exactly one variable changed) and preflight
refuses to run arms until that twin exists.

ROLES (one file, shipped to the 4060 by sha through the anchor -- the trainer/gate roles import
only stdlib + torch/transformers/peft/safetensors(+bitsandbytes on the 4060), never the repo):
  chain            5090 orchestrator (default). Gates -> arms -> scoring -> verdicts.
  trainer          the layer-split trainer, runs on BOTH boxes (--box 5090|4060).
  gate-footprint   design 7(a), runs on the 4060.
  gate-basece      design 7(b), runs on both boxes.
  stage-4060       runs on the 4060: base fetch + fingerprint, corpus fetch by sha.
DATA PLANE = the anchor http://47.84.93.96:8710 only (WAN; owner directive "there is no LAN in the
real world"). CONTROL PLANE = the 4060's control agent (spawn / logs / ps), read-only here.
"""
import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------- frozen spec
SPEC = {
    "design": "docs/research/WAN_GRADIENT_SURFACE_PARALLEL_DESIGN_2026-09-06.md",
    "model_dir_5090": "E:/olmoe_base_true_bf16",
    "model_dir_4060": r"D:\wangrad\olmoe_base",
    "hf_repo": "allenai/OLMoE-1B-7B-0924",
    "corpus_5090": "E:/rsft_data/gsm8k_rsft_traces_clean.jsonl",
    "corpus_4060": r"D:\wangrad\gsm8k_rsft_traces_clean.jsonl",
    # design section 5 recipe (= d1_sft.py, the trainer every banked arm used)
    "n_traces": 10000, "seq_len": 1024, "micro_batch": 2, "accum": 8, "lr": 1e-4, "rank": 16,
    "warmup": 20, "clip": 1.0, "betas": (0.9, 0.95),
    "seeds": {"s1": 20260829, "s2": 20260830},
    "own_layers": {"5090": list(range(0, 12)), "4060": list(range(12, 16))},   # section 5
    "n_layers": 16, "trainable_total": 4194304,
    "anchor": "http://47.84.93.96:8710",
    "token_env": "NEURAHASH_CONTENT_TOKEN",
    # the banked control twin (design C-SOLO, cut as a duplicate -- see module docstring)
    "twin": {20260829: "E:/rsft_runs/stylectl/d1tb", 20260830: "E:/rsft_runs/stylectl/d1tb_s2"},
    "twin_ext": {20260829: "E:/rsft_runs/gsm8k_ext/S1EXT_D1TB-s1.json",
                 20260830: "E:/rsft_runs/gsm8k_ext/S1EXT_D1TB-s2.json"},
    # the nf4 control twin (base policy nf4, 2026-09-08): the banked recipe on the NF4 base, one
    # variable changed. Written by scratchpad/nf4_twin_chain.py; its GSM8K-807ext lands beside
    # its other artefacts as S1EXT_<tag>.json (twin_ext_path resolves both conventions).
    "twin_nf4": {20260829: "E:/rsft_runs/nf4_twin_s1", 20260830: "E:/rsft_runs/nf4_twin_s2"},
    "base_gsm512": "E:/rsft_runs/_scorer_repro/S1_base_today.json",
    "base_gsmext": "E:/rsft_runs/gsm8k_ext/S1EXT_A0_truebase.json",
    "run_root": "E:/rsft_runs/wangrad",
    "remote_root": r"D:\wangrad",
}
# Full-file sha256 of the TRUE base, measured 2026-09-08 on E:/olmoe_base_true_bf16 (13.6 s).
# The 4060's copy must match ALL of these or nothing trains (memory olmoe-base-was-actually-instruct).
TRUE_BASE_SHA = {
    "model-00001-of-00003.safetensors":
        ("5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e", 4997744872),
    "model-00002-of-00003.safetensors":
        ("15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7", 4997235176),
    "model-00003-of-00003.safetensors":
        ("a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04", 3843741912),
    "config.json": ("3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335", 759),
    "tokenizer.json": ("a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40", 2115417),
    "tokenizer_config.json": ("78a839c7851f14f9fb30e664c2b46166dc0628f2900679e5ec160656f702edff", 5372),
    "special_tokens_map.json": ("b77491e270c6fcc5b2ecf22370f7318a6a18d3cabea09ba7bab92e9bf12656c2", 65),
}
ARMS = [   # order is load-bearing: H1 first so the section-5 KILL can stop the rest
    ("h1_s1", "h1", "s1"),
    # 2026-09-10: registered by AMENDMENT 1 A1.4 -- clauses 2 and 5 close only on TWO seeds.
    # At index 1, not appended: ARMS[:1] (the --stop-after-h1 set, :1760) is unchanged, while
    # ARMS[1:] (the remaining-arms loop, :2492) now runs the second seed BEFORE the three arms
    # the 2026-09-10 ordering review dropped (h25_s1/h25_s2 premise measured away at 1.43 s/sync;
    # indep_s1 sits beside the closed layer-delta-interference negative).
    ("h1_s2", "h1", "s2"),
    ("h25_s1", "h25", "s1"),
    ("h25_s2", "h25", "s2"),
    ("indep_s1", "indep", "s1"),
]
H_OF = {"h1": 1, "h25": 25, "indep": 0}        # 0 = H-infinity (design W-INDEP-SUM)

# The banked BF16 twins the nf4 control must reproduce (nf4_twin chain P0, 2026-09-09).
# Used by the resolution-aware P1 gate in preflight -- see the comment there.
BF16_TWIN_BATTERY = {20260829: 0.9951, 20260830: 0.7646}   # pp vs the fingerprinted base
BF16_TWIN_MDE = {20260829: 0.7261, 20260830: 0.7391}       # pp, at each twin's discordance

assert SPEC["n_traces"] % 16 == 0, "N_TRACES must be a multiple of 16 (house rule)"
EXPECT_STEPS = SPEC["n_traces"] // (SPEC["micro_batch"] * SPEC["accum"])       # 625
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# ----------------------------------------------------------------------------- base policy (2026-09-08)
BASE_DTYPES = ("bf16", "nf4")
BASE_POLICIES = {                          # chain-level policy -> the base dtype each box loads
    "mixed": {"5090": "bf16", "4060": "nf4"},    # the design as registered (section 2 / 7b)
    "nf4":   {"5090": "nf4",  "4060": "nf4"},    # owner 2026-09-08: "go straight to both nf4"
}
# What the nf4 policy needs on the 5090 and does not have (probe 2026-09-08): a transformers-4.x
# interpreter that can still see a CUDA torch + bitsandbytes + peft. C:/Python313's packages live
# in the USER site (AppData/Roaming/Python/Python313/site-packages), which a venv does NOT
# inherit even with --system-site-packages, so the venv must carry its own torch. Pins are the
# versions the 5090 runs today (torch 2.9.1+cu128, peft 0.18.0, bnb 0.48.2, accelerate 1.8.1,
# safetensors 0.7.0, tokenizers 0.22.1); transformers 4.57.1 is what the 4060's D:\tf4env carries
# and it needs huggingface_hub < 1.0. Pin numpy explicitly: the 4060's env imported peft with
# `numpy.core.umath failed to import` (a numpy ABI mismatch), which preflight now refuses.
TF4_ENV_RECIPE = [
    r'C:\Python313\python.exe -m venv E:\tf4env',
    r'E:\tf4env\Scripts\python.exe -m pip install --upgrade pip',
    r'E:\tf4env\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128',
    r'E:\tf4env\Scripts\python.exe -m pip install "transformers==4.57.1" "huggingface_hub<1.0" '
    r'"tokenizers==0.22.1" "peft==0.18.0" "bitsandbytes==0.48.2" "accelerate==1.8.1" '
    r'"safetensors==0.7.0" "numpy<2.3"',
    r'E:\tf4env\Scripts\python.exe E:\aiCrypto\scratchpad\wan_gradient_chain.py --role env-probe',
]


def resolve_base_dtype(box, base_dtype, policy="mixed"):
    """`auto` (the default) = what --base-policy prescribes for this box; under the default
    policy that is the pre-2026-09-08 coupling (bf16 on the 5090, nf4 on the 4060), so every
    existing launch line is byte-unchanged. An explicit value is honoured on EITHER box."""
    if policy not in BASE_POLICIES:
        raise SystemExit("unknown --base-policy %r (choices: %s)" % (policy, ", ".join(BASE_POLICIES)))
    if base_dtype in (None, "", "auto"):
        return BASE_POLICIES[policy][box]
    if base_dtype not in BASE_DTYPES:
        raise SystemExit("unknown --base-dtype %r (choices: auto, %s)" % (base_dtype, ", ".join(BASE_DTYPES)))
    return base_dtype


def policy_problem(box, base_dtype, policy):
    """None, or why this box's dtype is not what the policy prescribes (a LOCAL check, before
    any anchor traffic; the cross-box check is init_meta_problem)."""
    want = BASE_POLICIES.get(policy, {}).get(box)
    if want is None:
        return "unknown base policy %r" % (policy,)
    if base_dtype != want:
        return ("--base-dtype %s contradicts --base-policy %s, which prescribes %s for the %s"
                % (base_dtype, policy, want, box))
    return None


def init_meta_problem(meta, mine):
    """The 4060's step-0 check of the 5090's init_meta (design section 5 VOID set + the base-policy
    VOID of 2026-09-08). Pure, so it is unit-checked on CPU. Returns None or the reason. The boxes
    must (a) train the same batches (data sha, token parity, order), (b) apply the same clip
    regime, (c) run the same BASE POLICY, each on the dtype it prescribes -- a 5090 that fell back
    to bf16 under the nf4 policy is exactly the run the owner must never get."""
    for k in ("data_sha256", "token_parity_sha", "selected_order_sha256", "clip_mode", "base_policy"):
        if meta.get(k) != mine.get(k):
            return ("%s differs from the 5090's (%s vs %s) -- the boxes would not train the same run"
                    % (k, str(meta.get(k))[:16], str(mine.get(k))[:16]))
    pol = BASE_POLICIES.get(mine.get("base_policy") or "")
    if pol is None:
        return "unknown base_policy %r" % (mine.get("base_policy"),)
    if meta.get("base_dtype") != pol["5090"]:
        return ("the 5090 loaded its base as %s but policy %s prescribes %s for it"
                % (meta.get("base_dtype"), mine.get("base_policy"), pol["5090"]))
    if mine.get("base_dtype") != pol["4060"]:
        return ("this box loaded its base as %s but policy %s prescribes %s for the 4060"
                % (mine.get("base_dtype"), mine.get("base_policy"), pol["4060"]))
    return None


def twin_ext_path(seed, twin_dir):
    """The GSM8K-807ext artefact of a control twin: the banked pair keeps it in gsm8k_ext/
    (SPEC["twin_ext"]); a twin produced by nf4_twin_chain.py keeps it beside its other artefacts
    as S1EXT_<tag>.json (the wan score_arm convention)."""
    if twin_dir == SPEC["twin"].get(seed):
        return SPEC["twin_ext"][seed]
    return os.path.join(twin_dir, "S1EXT_%s.json" % os.path.basename(twin_dir))


ENV_PROBE_SRC = r'''
import json, sys
d = dict(exe=sys.executable, python=sys.version.split()[0], ok=False)
try:
    import torch; d["torch"] = torch.__version__; d["cuda"] = bool(torch.cuda.is_available())
except Exception as e: d["torch_error"] = repr(e)[:160]
try:
    import transformers; d["transformers"] = transformers.__version__
except Exception as e: d["transformers_error"] = repr(e)[:160]
try:
    import peft; d["peft"] = peft.__version__
except Exception as e: d["peft_error"] = repr(e)[:160]
try:
    import bitsandbytes as b; d["bitsandbytes"] = b.__version__
except Exception as e: d["bnb_error"] = repr(e)[:160]
try:
    # the make-or-break, on the META device (no weights, no CUDA): can bitsandbytes reach the
    # experts under THIS transformers? tf5.x: fused 3-D nn.Parameter -> 0 of 3072; tf4.x: 3072.
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(sys.argv[1])
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    fused = [n for n, p in m.named_parameters() if ".experts" in n and (p.dim() == 3 or "gate_up_proj" in n)]
    lin = [n for n, mod in m.named_modules() if ".mlp.experts." in n and isinstance(mod, torch.nn.Linear)]
    want = cfg.num_hidden_layers * cfg.num_experts * 3
    d.update(expected_expert_linears=want, expert_linear_modules=len(lin), fused_expert_params=len(fused),
             fused_examples=fused[:2], bnb_can_quantize_experts=(len(lin) == want and not fused))
except Exception as e: d["layout_error"] = repr(e)[:200]
d["ok"] = bool(d.get("torch")) and bool(d.get("transformers"))
print("ENVJSON " + json.dumps(d), flush=True)
'''


def local_env_probe(py, work_dir, log, model_dir=None, timeout=900):
    """Versions + expert layout under interpreter `py`, file-based (the same probe the chain runs
    on the 4060, run locally). Read-only apart from the probe file. Returns a dict; on failure
    {'ok': False, 'error': ...}."""
    import subprocess
    os.makedirs(work_dir, exist_ok=True)
    probe = os.path.join(work_dir, "_env_probe_local.py")
    with io.open(probe, "w", encoding="utf-8") as fh:
        fh.write(ENV_PROBE_SRC)
    if not py or not os.path.exists(py):
        return {"ok": False, "error": "interpreter missing: %r" % (py,)}
    try:
        r = subprocess.run([py, probe, model_dir or SPEC["model_dir_5090"]], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as e:                                               # noqa: BLE001
        return {"ok": False, "error": repr(e)[:200]}
    for line in (r.stdout or "").splitlines():
        if line.startswith("ENVJSON "):
            return json.loads(line[8:])
    return {"ok": False, "error": "rc=%s stdout=%s stderr=%s" % (r.returncode, (r.stdout or "")[-160:],
                                                                (r.stderr or "")[-300:])}


def nf4_env_problems(env, py, box="5090"):
    """Why `env` (a local_env_probe / remote_env_probe dict) cannot run the nf4 base: [] if it can.
    Every item is a full sentence naming the owner action, because this is the loud stop the
    owner asked for instead of a silently-bf16 run."""
    probs = []
    if not env.get("ok"):
        probs.append("%s interpreter %s unusable: %s" % (box, py, str(env.get("error", "?"))[:200]))
        return probs
    tfv = str(env.get("transformers", ""))
    if not tfv.startswith("4."):
        probs.append("%s interpreter %s has transformers %s, need 4.x (fused-expert trap: under 5.x "
                     "OLMoE's experts are one nn.Parameter per layer that bitsandbytes cannot "
                     "quantize)" % (box, py, tfv))
    if env.get("bnb_can_quantize_experts") is False:
        probs.append("%s interpreter %s: expert layout is FUSED (%s of %s projections quantizable, "
                     "%s fused params) -- bitsandbytes cannot produce the nf4 base here"
                     % (box, py, env.get("expert_linear_modules"), env.get("expected_expert_linears"),
                        env.get("fused_expert_params")))
    elif "bnb_can_quantize_experts" not in env and env.get("layout_error"):
        probs.append("%s interpreter %s: layout probe failed: %s" % (box, py, env.get("layout_error")))
    if not env.get("cuda"):
        probs.append("%s interpreter %s: torch.cuda.is_available() is False" % (box, py))
    if not env.get("bitsandbytes"):
        probs.append("%s interpreter %s: bitsandbytes missing (%s)" % (box, py, env.get("bnb_error", "")))
    if not env.get("peft"):
        probs.append("%s interpreter %s: peft does not import (%s) -- the trainer dies at attach_lora"
                     % (box, py, env.get("peft_error", "missing")))
    return probs


def now():
    return time.strftime("%m-%d %H:%M:%S")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def gib(n):
    return n / float(1 << 30)


# ----------------------------------------------------------------------------- the anchor (data plane)
class Anchor:
    """sha256-keyed content store, tools/content_store.py. PUT /o/<sha> is refused unless
    sha256(body) == the path (content_store.py:38); X-Auth gates authorization; X-Name records a
    friendly name that GET /manifest resolves. The request log collapses hashes to 12 chars, so
    /manifest (~2.2 MB at 56,931 objects on 2026-09-08) is the ONLY name resolver -- every poll
    is counted and reported in the run manifest because the July lane was manifest-bound
    (memory glm-lane-manifest-throughput-bound)."""

    def __init__(self, url, token, log):
        self.url = url.rstrip("/")
        self.token = token
        self.log = log
        self.bytes_up = 0
        self.bytes_down = 0
        self.manifest_polls = 0
        self.manifest_bytes = 0

    def _open(self, req, timeout):
        return urllib.request.urlopen(req, timeout=timeout)

    def put(self, data, name, retries=6):
        sha = sha256_bytes(data)
        for i in range(retries):
            try:
                req = urllib.request.Request(self.url + "/o/" + sha, data=data, method="PUT")
                req.add_header("Content-Type", "application/octet-stream")
                if self.token:
                    req.add_header("X-Auth", self.token)
                if name:
                    req.add_header("X-Name", name)
                with self._open(req, 600) as r:
                    r.read()
                    if r.status not in (200, 201):
                        raise RuntimeError("PUT status %s" % r.status)
                self.bytes_up += len(data)
                return sha
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise RuntimeError("anchor refused PUT (%d): bad/absent %s"
                                       % (e.code, SPEC["token_env"]))
                self.log("PUT %s try %d failed: HTTP %s" % (name, i + 1, e.code))
            except Exception as e:                                       # noqa: BLE001
                self.log("PUT %s try %d failed: %s" % (name, i + 1, repr(e)[:120]))
            time.sleep(min(60, 5 * (i + 1)))
        raise RuntimeError("PUT %s failed after %d tries" % (name, retries))

    def get(self, sha, retries=6):
        for i in range(retries):
            try:
                with self._open(urllib.request.Request(self.url + "/o/" + sha), 900) as r:
                    b = r.read()
                if sha256_bytes(b) != sha:
                    raise RuntimeError("sha mismatch on GET %s" % sha[:12])
                self.bytes_down += len(b)
                return b
            except Exception as e:                                       # noqa: BLE001
                self.log("GET %s try %d failed: %s" % (sha[:12], i + 1, repr(e)[:120]))
                time.sleep(min(60, 5 * (i + 1)))
        raise RuntimeError("GET %s failed after %d tries" % (sha[:12], retries))

    def manifest(self):
        with self._open(urllib.request.Request(self.url + "/manifest"), 300) as r:
            b = r.read()
        self.manifest_polls += 1
        self.manifest_bytes += len(b)
        return json.loads(b.decode("utf-8"))

    def resolve(self, name):
        # 2026-09-11 21:40: ask the store for ONE name (/n/<name>, ~150 B, added tonight) instead of
        # the whole 2.5 MB manifest. Measured under the anchor's egress throttle (~1.5 Mbit/s from
        # ~20:50): a manifest GET took 33 s, two boxes polled it every 3 s, and the pair fell from
        # 50 to >100 s/step at step 100-118. A 404 means "not published yet"; any other failure
        # (older store, network) falls back to the manifest path, which is the pre-tonight code.
        try:
            from urllib.parse import quote
            with self._open(urllib.request.Request(self.url + "/n/" + quote(name, safe="/")), 30) as r:
                ent = json.loads(r.read().decode("utf-8"))
            self.name_polls = getattr(self, "name_polls", 0) + 1
            return (ent or {}).get("sha256")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self.name_polls = getattr(self, "name_polls", 0) + 1
                return None
        except Exception as e:                                           # noqa: BLE001
            self.log("name poll failed (%s) -- falling back to the manifest" % repr(e)[:80])
        try:
            ent = self.manifest().get(name)
        except Exception as e:                                           # noqa: BLE001
            self.log("manifest poll failed: %s" % repr(e)[:120])
            return None
        return (ent or {}).get("sha256")

    def wait(self, name, timeout_s, poll_s, alive=None):
        """Block until `name` resolves, then GET+verify it. `alive` is an optional callable that
        returns False when the producer is known dead (early abort, never a kill)."""
        t0 = time.time()
        while True:
            sha = self.resolve(name)
            if sha:
                return sha, self.get(sha)
            if alive is not None and not alive():
                raise RuntimeError("peer died before publishing %s" % name)
            if time.time() - t0 > timeout_s:
                raise RuntimeError("timeout %.0fs waiting for %s" % (timeout_s, name))
            time.sleep(poll_s)

    def stats(self):
        return {"bytes_up": self.bytes_up, "bytes_down": self.bytes_down,
                "manifest_polls": self.manifest_polls, "manifest_bytes": self.manifest_bytes,
                "name_polls": getattr(self, "name_polls", 0)}

    def health(self):
        with self._open(urllib.request.Request(self.url + "/health"), 30) as r:
            return json.loads(r.read().decode("utf-8"))


def read_token(log):
    tok = os.environ.get(SPEC["token_env"], "")
    if not tok and os.name == "nt":
        try:                     # the same USER-env fallback c4060.py needed for its own secret
            import subprocess
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "[Environment]::GetEnvironmentVariable('%s','User')"
                                  % SPEC["token_env"]],
                                 capture_output=True, encoding="utf-8", errors="replace",
                                 timeout=30, creationflags=0x08000000)
            tok = (out.stdout or "").strip()
        except Exception:                                                # noqa: BLE001
            tok = ""
    log("content token: %s (never printed)" % ("PRESENT" if tok else "ABSENT"))
    return tok


# ----------------------------------------------------------------------------- corpus (verbatim d1_sft.py)
def select_rows(path, n_traces, seed):
    """d1_sft.py:select_rows, copied verbatim so both boxes draw the SAME shuffled batches as the
    banked twin. Iterates the file object (splits on newline only -- never .splitlines(), which
    tears records on U+2028; rules 10-dispatch Learned 2026-09-06)."""
    rows = []
    for line in io.open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("completion") and d.get("prompt"):
            rows.append(d)
    rows_in_file = len(rows)
    random.Random(seed).shuffle(rows)
    rows = rows[:n_traces]
    ids = [str(r.get("id")) for r in rows]
    info = {"rows_in_file": rows_in_file, "rows_selected": len(rows), "n_traces": n_traces,
            "selection": "shuffle_then_slice(seed)",
            "seed_dependent_subset": rows_in_file > n_traces,
            "selected_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest(),
            "selected_order_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()}
    return rows, info


def probe_rows(path, n=128):
    """Design 7(b) probe: the first n usable rows in FILE order -- seed-independent, identical on
    both boxes once the corpus sha matches."""
    out = []
    for line in io.open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("completion") and d.get("prompt"):
            out.append(d)
            if len(out) >= n:
                break
    return out


# ----------------------------------------------------------------------------- torch-side helpers
def make_logger(path):
    fh = io.open(path, "a", encoding="utf-8") if path else None

    def log(m):
        line = "[%s] %s" % (now(), m)
        try:
            print(line, flush=True)
        except Exception:                                                # noqa: BLE001
            pass
        if fh:
            fh.write(line + "\n")
            fh.flush()
    return log


def set_vram_cap(box, cap_gib, log):
    """Design 7(a)/section 2: the cap goes BEFORE the first CUDA allocation on BOTH boxes. On the
    5090 it is d1_sft.py's own formula (leave >= 8 GiB); on the 4060 an explicit GiB cap so an
    overflow is a real OOM, never a silent WDDM spill into system RAM (20-judgment Learned 2026-08-06)."""
    import torch
    free, total = torch.cuda.mem_get_info()
    if box == "5090":
        frac = min(0.72, max(0.10, (free - (8 << 30)) / float(total)))
    else:
        frac = min(0.95, max(0.10, cap_gib * (1 << 30) / float(total)))
    torch.cuda.set_per_process_memory_fraction(frac)
    log("vram cap %.3f of %.2f GiB (%.2f GiB free) on the %s" % (frac, gib(total), gib(free), box))
    return frac, total


def load_base(box, model_dir, log, base_dtype="auto"):
    """bf16: exactly d1_sft.py. nf4: the bitsandbytes NF4 frozen base (design section 2), whose
    experts MUST be an nn.ModuleList of nn.Linear (transformers 4.x) so bitsandbytes covers them
    -- asserted by assert_expert_layout on whichever box loads it, never assumed. The dtype is
    decoupled from `box` (2026-09-08): `auto` keeps the design's coupling, an explicit value is
    honoured on either box."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base_dtype = resolve_base_dtype(box, base_dtype)
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    log("loading base %s as %s on the %s" % (model_dir, base_dtype, box))
    if base_dtype == "bf16":
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
        model = model.to("cuda")
        model.gradient_checkpointing_enable()
    else:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        kw = dict(quantization_config=bnb, device_map={"": 0})
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **kw)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return tok, model


def assert_expert_layout(model, log):
    """Design section 2 / 7(a): every expert projection must be a bnb Linear4bit whose weight is
    Params4bit (uint8). A fused nn.Parameter (transformers 5.x `gate_up_proj`) or any bf16 expert
    weight FAILS the gate -- that is the trap that blew past 8 GB on GLM."""
    import torch
    fused = [n for n, p in model.named_parameters()
             if ".experts" in n and (p.dim() == 3 or "gate_up_proj" in n)]
    lin4, bad, bf16_expert_bytes = 0, [], 0
    for n, m in model.named_modules():
        if ".mlp.experts." in n and isinstance(m, torch.nn.Linear):
            w = getattr(m, "weight", None)
            if (type(m).__name__ == "Linear4bit" and w is not None
                    and type(w).__name__ == "Params4bit" and w.dtype == torch.uint8):
                lin4 += 1
            else:
                bad.append(n)
                if w is not None:
                    bf16_expert_bytes += w.numel() * w.element_size()
    layout = {"fused_expert_params": fused[:4], "n_fused": len(fused),
              "expert_linear4bit": lin4, "expert_not_4bit": bad[:4], "n_not_4bit": len(bad),
              "expert_non4bit_bytes": bf16_expert_bytes,
              "expected_linear4bit": SPEC["n_layers"] * 64 * 3}
    ok = (not fused) and (not bad) and lin4 == layout["expected_linear4bit"]
    log("expert layout: %s" % json.dumps(layout))
    return ok, layout


def attach_lora(model, seed, log):
    """d1_sft.py:239-250 verbatim: seed every RNG BEFORE get_peft_model (peft draws LoRA-A from
    torch's global RNG), attention-only r16/alpha32/dropout0 on q,k,v,o over ALL 16 layers."""
    import torch
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=SPEC["rank"], lora_alpha=2 * SPEC["rank"], lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed % (2 ** 32))
    except Exception:                                                    # noqa: BLE001
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = get_peft_model(model, cfg)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("lora attached: %d trainable (spec %d)" % (n_tr, SPEC["trainable_total"]))
    if n_tr != SPEC["trainable_total"]:
        raise SystemExit("SURFACE MISMATCH: %d trainable != %d" % (n_tr, SPEC["trainable_total"]))
    return model


def canon(name):
    """peft parameter name -> canonical wire key (drops the `base_model.model.` wrapper so the
    same key matches on tf4.x/bnb and tf5.x/bf16 module trees)."""
    return name.replace("base_model.model.", "", 1)


def lora_params(model):
    return {canon(n): p for n, p in model.named_parameters() if "lora_" in n}


def layer_of(key):
    m = LAYER_RE.search(key)
    return int(m.group(1)) if m else -1


def freeze_to_owned(model, own_layers, log, peer_grad=False):
    """Ownership. The OWNED layers' LoRA is trainable; the PEER layers' LoRA is never stepped by
    this box. `peer_grad=True` (clip-mode global, review 2026-09-08 HIGH-2) keeps the peer slice's
    requires_grad ON so its gradient is COMPUTED each step -- measured for the 16-layer clip norm,
    then discarded -- without ever entering the optimizer (lora_sets builds the optimizer's list by
    layer ownership, never by requires_grad)."""
    own = set(own_layers)
    n_own = n_peer = 0
    for k, p in lora_params(model).items():
        if layer_of(k) in own:
            p.requires_grad_(True)
            n_own += p.numel()
        else:
            p.requires_grad_(bool(peer_grad))
            n_peer += p.numel()
    log("ownership: layers %s -> %d trainable, %d peer params (%s)"
        % (sorted(own), n_own, n_peer,
           "grad measured for the global clip, never stepped" if peer_grad else "frozen"))
    return n_own, n_peer


def lora_sets(model, own_layers):
    """(own_params, peer_params, clip_params). own/peer are split by LAYER OWNERSHIP -- never by
    requires_grad, which under clip-mode global is True on both. clip_params is d1_sft.py:367's
    exact list, `[p for p in model.parameters() if p.requires_grad]` in model order: all 16
    layers' LoRA under clip-mode global, the owned slice alone under clip-mode local."""
    own = set(own_layers)
    lp = lora_params(model)
    own_params = [p for k, p in lp.items() if layer_of(k) in own]
    peer_params = [p for k, p in lp.items() if layer_of(k) not in own]
    clip_params = [p for p in model.parameters() if p.requires_grad]
    return own_params, peer_params, clip_params


def grad_norm(params):
    """2-norm over the present grads of `params`, read-only (torch's own stack-of-norms formula)."""
    import torch
    gs = [p.grad.detach() for p in params if p.grad is not None]
    if not gs:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gs])))


def clip_step(own_params, peer_params, clip_params, max_norm, mode):
    """The clip of d1_sft.py:367-368, reproduced on a split box (review 2026-09-08 HIGH-2).

    d1_sft clips the GLOBAL norm over all 16 layers' LoRA and scales every gradient by
    min(1, max_norm / (global + 1e-6)). A per-slice clip (mode "local", the pre-fix code) is only
    identical on steps where the global norm stays <= max_norm; local <= global always, so on any
    step the twin scaled, the local clip would leave this box's update UNscaled -- a difference in
    the update rule, not in the distribution of the gradient, and the whole experiment is a
    non-inferiority test against that twin.

    mode "global": the peer slice's gradient was computed this step (requires_grad kept on, see
    freeze_to_owned); torch's clip_grad_norm_ runs over clip_params -- the SAME call, the SAME
    parameter list in the SAME model order as d1_sft -- so the owned slice receives exactly the
    factor a single-box run would apply from this box's current weights (own slice + peer slice
    as last synced, on this box's base dtype -- policy mixed: bf16 on the 5090, nf4 on the 4060;
    policy nf4: nf4 on both). The peer gradient is
    then discarded: never stepped, never published, and cleared so it cannot accumulate into the
    next step's norm (autograd accumulates .grad; the optimizer only zeroes its own params).
    mode "local": the pre-fix behaviour, kept only as an explicit escape hatch.

    Returns (global_pre, local_pre, peer_pre, coef): the pre-clip norms and the factor the owned
    slice's gradient actually received (1.0 when the clip did not bind)."""
    import torch
    local = grad_norm(own_params)
    if mode == "global":
        peer = grad_norm(peer_params)
        gn = float(torch.nn.utils.clip_grad_norm_(clip_params, max_norm))
        for p in peer_params:
            p.grad = None
    else:
        peer = 0.0
        gn = float(torch.nn.utils.clip_grad_norm_(own_params, max_norm))
    coef = min(1.0, max_norm / (gn + 1e-6))          # torch's own clamp(max_norm/(total+1e-6), 1)
    return gn, local, peer, coef


TENSORS_PER_LAYER = 8          # q,k,v,o x (lora_A, lora_B); 16 layers = 128 = the whole surface


def expected_slice_tensors(layers):
    return len(set(layers)) * TENSORS_PER_LAYER


def slice_bytes(model, layers, dtype_name):
    """Serialize the LoRA params of `layers` as a safetensors blob (bf16 on the wire for the
    periodic syncs, fp32 for init/final -- design section 1's byte table is the bf16 one)."""
    import torch
    from safetensors.torch import save as st_save
    dt = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    lay = set(layers)
    d = {k: p.detach().to(dt).cpu().contiguous() for k, p in lora_params(model).items()
         if layer_of(k) in lay}
    want = expected_slice_tensors(lay)
    if len(d) != want:                       # never publish a PARTIAL slice (LOW-2 family)
        raise SystemExit("SLICE BUILD ABORT: %d tensors for layers %s, expected %d"
                         % (len(d), sorted(lay), want))
    return st_save(d), len(d)


def load_slice(model, blob, layers, log):
    """Copy a received slice into this model's LoRA params for `layers`, matched by canonical key.
    ABORTS unless EVERY expected tensor of those layers was matched and copied (review 2026-09-08
    LOW-2): the old check aborted only at ZERO matches, so a blob that matched 64 of 128 init
    tensors, or 20 of 32 peer tensors, would have been applied PARTIALLY and logged as loaded --
    the same silent-partial family as a dead surface (HIGH-1). Expected counts: 128 for the
    16-layer init, 32 for the 4060's slice, 96 for the 5090's; the blob must carry exactly those."""
    import torch
    from safetensors.torch import load as st_load
    d = st_load(blob)
    lay = set(layers)
    own = lora_params(model)
    want = expected_slice_tensors(lay)
    have = sum(1 for k in own if layer_of(k) in lay)
    matched, problems = 0, []
    with torch.no_grad():
        for k, v in d.items():
            if layer_of(k) not in lay:
                problems.append("%s: outside layers" % k)
                continue
            p = own.get(k)
            if p is None:
                problems.append("%s: no such param here" % k)
                continue
            if tuple(p.shape) != tuple(v.shape):
                problems.append("%s: shape %s != %s" % (k, tuple(v.shape), tuple(p.shape)))
                continue
            p.copy_(v.to(p.device, p.dtype))
            matched += 1
    if not (matched == want == len(d) == have):
        raise SystemExit("SLICE LOAD ABORT: matched %d of %d blob tensors; expected exactly %d for "
                         "layers %s (this model holds %d); first problems: %s"
                         % (matched, len(d), want, sorted(lay), have,
                            "; ".join(problems[:4]) or "none"))
    return matched, len(d)


def make_batchify(tok, seq_len, stats):
    """d1_sft.py:batchify verbatim (prompt masked to -100, right-truncation with EOS restore,
    drop <8 tokens or fully-masked rows). SEQ_STATS feeds the zero-truncation VOID."""
    import torch

    def batchify(chunk):
        ids_list, lab_list = [], []
        for r in chunk:
            p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            full_ids = p_ids + c_ids
            full_labs = [-100] * len(p_ids) + c_ids
            ids = full_ids[:seq_len]
            labs = full_labs[:seq_len]
            if len(full_ids) > seq_len:
                stats["truncated"] += 1
                if any(x != -100 for x in labs) and ids[-1] != tok.eos_token_id:
                    ids[-1] = tok.eos_token_id
                    labs[-1] = tok.eos_token_id
                    stats["eos_restored"] += 1
            if len(ids) < 8 or all(x == -100 for x in labs):
                stats["dropped"] += 1
                continue
            ids_list.append(ids)
            lab_list.append(labs)
        if not ids_list:
            return None, None
        L = max(len(x) for x in ids_list)
        inp = torch.full((len(ids_list), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(ids_list), L), -100, dtype=torch.long)
        att = torch.zeros((len(ids_list), L), dtype=torch.long)
        for i, (a, b) in enumerate(zip(ids_list, lab_list)):
            inp[i, :len(a)] = torch.tensor(a)
            lab[i, :len(b)] = torch.tensor(b)
            att[i, :len(a)] = 1
        return (inp.to("cuda"), att.to("cuda")), lab.to("cuda")
    return batchify


def token_parity_sha(tok, rows, n=64):
    """sha256 over the token ids of the first n selected rows: both boxes must agree exactly, or
    they are not training the same batches (tokenizer drift across transformers 4.x/5.x)."""
    h = hashlib.sha256()
    for r in rows[:n]:
        ids = tok(r["prompt"], add_special_tokens=False)["input_ids"] + \
            tok(r["completion"], add_special_tokens=False)["input_ids"]
        h.update((",".join(map(str, ids)) + "\n").encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- role: trainer
def _bnb_version():
    """Recorded in the manifest (fable review 16:07): torch/bnb/peft resolve from the ROAMING user
    site on this box, so a stray pip install elsewhere changes the trainer's kernels silently."""
    try:
        import bitsandbytes
        return getattr(bitsandbytes, "__version__", "?")
    except Exception:                                                    # noqa: BLE001
        return None


def _ckpt_state(model):
    """LoRA tensors only (4,194,304 params) -- self-contained, no peft-version coupling."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def save_step_ckpt(path, model, opt, sched, step, i, seen, acc_lists, log):
    """Atomic checkpoint of everything the loop needs to resume BIT-IDENTICALLY.

    Written AFTER the sync block, so the peer slice held in `model` is the one both boxes agreed
    on at this step. At H=1 the boxes are in lockstep and the interval is fixed, so both sides
    checkpoint at the SAME step number -- which is what makes a joint resume coherent. Atomic
    (tmp + replace): a crash mid-write must not leave a torn checkpoint that reads as valid.
    """
    import random
    import numpy as _np
    import torch
    tmp = path + ".tmp"
    blob = {"format": 1, "step": step, "row_i": i, "seen": seen,
            "lora": _ckpt_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "acc": acc_lists,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "rng_np": _np.random.get_state(), "rng_py": random.getstate()}
    torch.save(blob, tmp)
    os.replace(tmp, path)
    log("  ckpt @%d -> %s (%.1f MB)" % (step, os.path.basename(path),
                                        os.path.getsize(path) / 1e6))


def load_step_ckpt(path, model, opt, sched, log):
    """Restore. Returns (step, row_i, seen, acc_lists). Raises rather than half-restoring."""
    import random
    import numpy as _np
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != 1:
        raise RuntimeError("checkpoint format %r not understood" % blob.get("format"))
    missing = model.load_state_dict(blob["lora"], strict=False)
    if getattr(missing, "unexpected_keys", []):
        raise RuntimeError("checkpoint has %d unexpected LoRA keys" % len(missing.unexpected_keys))
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    torch.set_rng_state(blob["rng_torch"].to(torch.uint8).cpu()
                        if hasattr(blob["rng_torch"], "to") else blob["rng_torch"])
    if blob.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["rng_cuda"])
    _np.random.set_state(blob["rng_np"])
    random.setstate(blob["rng_py"])
    log("  RESUMED from %s at step %d (row %d, seen %d)"
        % (os.path.basename(path), blob["step"], blob["row_i"], blob["seen"]))
    return blob["step"], blob["row_i"], blob["seen"], blob["acc"]


def role_trainer(a):
    """The layer-split trainer. Both boxes run this on the SAME shuffled batches (select_rows with
    the same seed on a sha-identical corpus). Each box updates ONLY its owned layers' LoRA and
    carries the peer's slice as a frozen adjunct that the periodic sync refreshes. Combine = exact
    concatenation over disjoint parameter support (design section 3). H=1 is the Jacobi
    block-coordinate limit; H=0 here means H-infinity (design W-INDEP-SUM: no syncs, one terminal
    concat)."""
    # --deterministic (2026-09-11, AMENDMENT 3 finding): the resume drill showed this trainer's
    # bookkeeping and FORWARD reproduce bit-for-bit after a resume, but the 5090's BACKWARD does
    # not (grad-norm 0.626332700 vs 0.626332462 at the first resumed step, compounding to 0.5%
    # LoRA drift in 25 steps), so AMENDMENT 1 A1.2's escape ("resume shown bit-exact") was
    # unreachable. Seeds were always set; deterministic KERNELS never were. The workspace knob
    # must be in the environment before CUDA initialises, hence before the torch import below.
    # Default OFF: a registered run is unchanged unless its chain passes the flag explicitly.
    if getattr(a, "deterministic", False):
        # hard set, not setdefault (fable review 16:07): an inherited different value would
        # silently change the kernels the certificate was issued for
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    box, peer = a.box, ("4060" if a.box == "5090" else "5090")
    os.makedirs(a.out_dir, exist_ok=True)
    log = make_logger(os.path.join(a.out_dir, "trainer_%s.log" % box))
    log("=" * 78)
    log("WAN-GRADIENT TRAINER | box %s | arm %s | H %s | seed %d | run %s | steps_limit %s"
        % (box, a.arm, a.H if a.H else "inf", a.seed, a.run_id, a.steps_limit or "full"))
    if getattr(a, "deterministic", False):
        # warn_only=False on purpose: an op with no deterministic implementation must ABORT the
        # run at its first step, not silently fall back to the nondeterministic kernel and let a
        # later drill "prove" something about a different code path.
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # attention backward is the usual atomics offender; the math kernel is the deterministic one
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        # cuDNN attention is a FOURTH backend (torch 2.9.1, cuDNN 9.10) and stays selectable
        # unless disabled explicitly -- missed in the first cut, caught by the 16:07 review;
        # repo precedent scratchpad/lmbwd_gate.py:103-104 disables it for the same reason.
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        log("DETERMINISTIC kernels: use_deterministic_algorithms=True cudnn.deterministic=True "
            "benchmark=False sdp=math-only (flash/mem_efficient/cudnn off) CUBLAS_WORKSPACE_CONFIG=%s"
            % os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    # base policy (2026-09-08): resolve this box's dtype, refuse a contradiction BEFORE any
    # anchor traffic or CUDA allocation. The cross-box check is init_meta_problem below.
    base_dtype = resolve_base_dtype(box, a.base_dtype, a.base_policy)
    log("base policy %s -> this box loads %s (5090 %s / 4060 %s)"
        % (a.base_policy, base_dtype, BASE_POLICIES[a.base_policy]["5090"],
           BASE_POLICIES[a.base_policy]["4060"]))
    why = policy_problem(box, base_dtype, a.base_policy)
    if why:
        log("VOID: " + why)
        return 3
    token = read_token(log)
    if not token:
        log("ABORT: no content token -- the owned slice could not be published")
        return 2
    anchor = Anchor(a.anchor, token, log)
    ns = "wangrad/%s" % a.run_id
    pns = ("wangrad/%s" % a.peer_ns) if getattr(a, "peer_ns", "") else ns   # ghost-peer reads (diagnostic); writes stay on ns

    # -- corpus identity (design section 5 VOID: corpus sha mismatch)
    data_sha = sha256_file(a.data)
    if a.expect_data_sha and not data_sha.startswith(a.expect_data_sha):
        log("VOID: corpus sha %s != registered %s" % (data_sha[:16], a.expect_data_sha[:16]))
        return 3
    rows, sel = select_rows(a.data, SPEC["n_traces"], a.seed)
    log("corpus %s sha %s | rows_in_file %d | selected %d | order_sha %s"
        % (a.data, data_sha[:16], sel["rows_in_file"], sel["rows_selected"],
           sel["selected_order_sha256"][:16]))

    frac, total_vram = set_vram_cap(box, a.cap_gib, log)          # BEFORE the first CUDA call
    tok, model = load_base(box, a.model_dir, log, base_dtype)
    alloc_base = torch.cuda.memory_allocated()
    log("base loaded: %.3f GiB allocated" % gib(alloc_base))
    layout = None
    if base_dtype == "nf4":                  # on WHICHEVER box loads nf4 (2026-09-08)
        ok, layout = assert_expert_layout(model, log)
        if not ok:
            log("VOID (design 7a): expert layout on the %s is not fully nf4 -- see layout above. "
                "Under transformers 5.x OLMoE's experts are a fused nn.Parameter bitsandbytes "
                "skips, so this interpreter cannot produce the nf4 base; it needs transformers "
                "4.x (TF4_ENV_RECIPE). Not continuing on a half-quantized base." % box)
            return 3
    model = attach_lora(model, a.seed, log)
    own_layers = SPEC["own_layers"][box]
    peer_layers = SPEC["own_layers"][peer]
    # clip-mode global (review 2026-09-08 HIGH-2): the peer slice keeps requires_grad so its
    # gradient can be MEASURED for the 16-layer clip norm; it is never stepped (see clip_step).
    n_own, n_peer = freeze_to_owned(model, own_layers, log, peer_grad=(a.clip_mode == "global"))
    tp_sha = token_parity_sha(tok, rows)

    # -- step-0 exchange: the 5090 publishes the FULL fp32 init (its torch stream made the banked
    #    twin's init, so both boxes start from exactly D1-TB's LoRA-A); the 4060 loads all 16
    #    layers from it and checks token parity + data sha.
    init_meta = {"run": a.run_id, "box": box, "data_sha256": data_sha, "token_parity_sha": tp_sha,
                 "selected_order_sha256": sel["selected_order_sha256"], "seed": a.seed,
                 "trainable_total": SPEC["trainable_total"], "H": a.H, "arm": a.arm,
                 "clip_mode": a.clip_mode, "base_policy": a.base_policy, "base_dtype": base_dtype}
    if box == "5090":
        blob, n_t = slice_bytes(model, range(SPEC["n_layers"]), "fp32")
        sha = anchor.put(blob, ns + "/init")
        anchor.put(json.dumps(init_meta).encode(), ns + "/init_meta")
        log("published init: %d tensors, %d B, sha %s" % (n_t, len(blob), sha[:12]))
    else:
        sha, meta_b = anchor.wait(pns + "/init_meta", a.peer_timeout_s, a.poll_s)
        meta = json.loads(meta_b.decode("utf-8"))
        # same batches, same clip regime, same BASE POLICY with each box on its prescribed dtype
        # (init_meta_problem; the base-policy VOID of 2026-09-08)
        why = init_meta_problem(meta, init_meta)
        if why:
            log("VOID: " + why)
            return 3
        sha, blob = anchor.wait(pns + "/init", a.peer_timeout_s, a.poll_s)
        m, n_t = load_slice(model, blob, range(SPEC["n_layers"]), log)
        log("loaded the 5090's init: %d of %d tensors (all 16 layers)" % (m, n_t))
        anchor.put(json.dumps(init_meta).encode(), ns + "/init_meta_4060")

    stats = {"truncated": 0, "dropped": 0, "eos_restored": 0}
    batchify = make_batchify(tok, SPEC["seq_len"], stats)
    # Optimizer over the OWNED slice by layer ownership -- never by requires_grad, which under
    # clip-mode global is also True on the peer slice (measurement only). clip_params is d1_sft's
    # exact `[p for p in model.parameters() if p.requires_grad]`; the two sets must be disjoint
    # and the clip set must be exactly own (+ peer under global), or the split is not what it says.
    own_params, peer_params, clip_params = lora_sets(model, own_layers)
    want_clip = len(own_params) + (len(peer_params) if a.clip_mode == "global" else 0)
    if (sum(p.numel() for p in own_params) != n_own or len(clip_params) != want_clip
            or set(map(id, own_params)) & set(map(id, peer_params))):
        raise SystemExit("OWNERSHIP MISMATCH: optimizer %d params (owned %d), clip set %d (want %d), "
                         "peer %d" % (sum(p.numel() for p in own_params), n_own, len(clip_params),
                                      want_clip, len(peer_params)))
    opt = torch.optim.AdamW(own_params, lr=SPEC["lr"], betas=SPEC["betas"], weight_decay=0.0)
    total_steps = EXPECT_STEPS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=SPEC["lr"], total_steps=total_steps,
        pct_start=min(0.3, SPEC["warmup"] / max(total_steps, 1)), anneal_strategy="cos")
    stop_at = a.steps_limit or total_steps
    log("%d optimizer steps planned (stopping at %d) | %d rows/step" % (total_steps, stop_at, 16))

    model.train()
    t0, step, seen, losses, gnorms, step_s = time.time(), 0, 0, [], [], []
    sync_log, clip_bound = [], 0
    # per-step clip record (HIGH-2): 16-layer pre-clip norm, peer-slice norm, factor applied
    g_global, g_peer, coefs, bound_steps = [], [], [], []
    n_global_bound = n_local_bound = n_global_not_local = 0
    i = 0
    peer_alive = None
    ckpt_path = os.path.join(a.out_dir, "step_ckpt_%s.pt" % box)
    if a.resume_from:
        step, i, seen, _acc = load_step_ckpt(a.resume_from, model, opt, sched, log)
        losses, gnorms, g_global, g_peer, coefs, step_s, sync_log = (
            _acc["losses"], _acc["gnorms"], _acc["g_global"], _acc["g_peer"],
            _acc["coefs"], _acc["step_s"], _acc["sync_log"])
        n_global_bound = _acc["n_global_bound"]
        n_local_bound = _acc["n_local_bound"]
        n_global_not_local = _acc["n_global_not_local"]
    while i < len(rows) and step < stop_at and (time.time() - t0) / 3600.0 < a.max_hours:
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(SPEC["accum"]):
            chunk = rows[i:i + SPEC["micro_batch"]]
            i += SPEC["micro_batch"]
            if not chunk:
                break
            xb, lab = batchify(chunk)
            if xb is None:
                continue
            out = model(input_ids=xb[0], attention_mask=xb[1], labels=lab)
            ce = out.loss
            (ce / SPEC["accum"]).backward()
            acc += float(ce.detach()) / SPEC["accum"]
            seen += len(chunk)
        # The twin's clip, reproduced (review 2026-09-08 HIGH-2; rationale in clip_step). Under
        # clip-mode global `gn` is the 16-layer pre-clip norm d1_sft.py:367 computes from this
        # box's current weights and `coef` the factor the owned slice's gradient actually received;
        # `loc` is the owned slice's own pre-clip norm -- what the pre-fix code clipped on. Every
        # step is recorded so a reader can separate a genuine INFERIOR from a clip-regime artefact.
        gn, loc, pn, coef = clip_step(own_params, peer_params, clip_params, SPEC["clip"], a.clip_mode)
        if gn > SPEC["clip"]:                            # the APPLIED clip bound on this step
            clip_bound += 1
            bound_steps.append(step + 1)
        if loc > SPEC["clip"]:
            n_local_bound += 1
        if a.clip_mode == "global" and gn > SPEC["clip"]:
            n_global_bound += 1                          # the twin's regime scaled this step ...
            if loc <= SPEC["clip"]:
                n_global_not_local += 1                  # ... and a per-slice clip would not have
        opt.step()
        if step < total_steps:
            sched.step()
        step += 1
        losses.append(acc)
        gnorms.append(loc)          # OWNED-slice pre-clip norm: `gnorm_max > 0` is the liveness gate
        g_global.append(gn)
        g_peer.append(pn)
        coefs.append(coef)
        step_s.append(time.time() - ts)
        if step % 25 == 0 or step == 1:
            log("  step %4d | loss %.4f | %d traces | %.2f h | lr %.2e | gnorm %.3f | global %.3f | "
                "coef %.4f | %.2f s/step"
                % (step, acc, seen, (time.time() - t0) / 3600.0, sched.get_last_lr()[0], loc, gn,
                   coef, step_s[-1]))
        # -- periodic sync (design section 5: every H steps each box PUTs its owned slice and GETs
        #    the other's, holding the peer slice as a frozen adjunct)
        if a.H and step % a.H == 0 and step < stop_at:
            tsy = time.time()
            blob, n_t = slice_bytes(model, own_layers, a.wire_dtype)
            sha = anchor.put(blob, "%s/%s/s%04d" % (ns, box, step))
            psha, pblob = anchor.wait("%s/%s/s%04d" % (pns, peer, step), a.peer_timeout_s,
                                      a.poll_s, alive=peer_alive)
            m, _ = load_slice(model, pblob, peer_layers, log)
            sync_log.append({"step": step, "own_sha": sha, "own_bytes": len(blob),
                             "peer_sha": psha, "peer_bytes": len(pblob), "peer_tensors": m,
                             "secs": time.time() - tsy})
            if step % (25 * a.H) == 0 or a.H >= 25:
                log("  sync @%d: up %d B, down %d B, %.1f s" % (step, len(blob), len(pblob),
                                                                  time.time() - tsy))
        # Checkpoint AFTER the sync: `model` now holds the peer slice both boxes agreed on at this
        # step, so a resume starts from a state the two sides share. Fixed interval => both boxes
        # land on the same step numbers => a joint resume is coherent without any extra handshake.
        if a.ckpt_every and step % a.ckpt_every == 0 and step < stop_at:
            try:
                # 2026-09-10: step-NAMED files, newest TWO kept. The boxes save the same step only
                # if neither dies inside its own sync->save window; one earlier generation on each
                # side keeps a common step across a one-interval mismatch, so a joint resume never
                # falls back to step 0 for that reason. Listing the names is enough to find it.
                ckpt_path = os.path.join(a.out_dir, "step_ckpt_%s_s%04d.pt" % (box, step))
                save_step_ckpt(ckpt_path, model, opt, sched, step, i, seen,
                               {"losses": losses, "gnorms": gnorms, "g_global": g_global,
                                "g_peer": g_peer, "coefs": coefs, "step_s": step_s,
                                "sync_log": sync_log, "n_global_bound": n_global_bound,
                                "n_local_bound": n_local_bound,
                                "n_global_not_local": n_global_not_local}, log)
                _mine = sorted(f for f in os.listdir(a.out_dir)
                               if re.match(r"^step_ckpt_%s_s\d{4}\.pt$" % box, f))
                for _old in _mine[:-2]:
                    os.remove(os.path.join(a.out_dir, _old))
            except Exception as _e:                                      # noqa: BLE001
                # Insurance must never be able to kill the run it is insuring.
                log("  ckpt @%d FAILED (run continues): %s" % (step, repr(_e)[:160]))

    budget_stopped = bool(i < len(rows) and step < stop_at)
    el = (time.time() - t0) / 3600.0
    # -- terminal: publish the owned slice in fp32 (exact); the 5090 assembles the full adapter.
    blob, n_t = slice_bytes(model, own_layers, "fp32")
    fsha = anchor.put(blob, ns + "/%s/final" % box)
    log("published FINAL owned slice: %d tensors, %d B, sha %s" % (n_t, len(blob), fsha[:12]))
    adapter_out = None
    if box == "5090":
        # The 4060 is ~5x slower; at H-infinity (INDEP) the 5090 finishes ~1 h earlier, so the
        # terminal wait gets the whole training budget, not the per-sync timeout.
        psha, pblob = anchor.wait(ns + "/4060/final", max(a.peer_timeout_s, a.max_hours * 3600),
                                  a.poll_s)
        m, n_t = load_slice(model, pblob, peer_layers, log)
        log("loaded the 4060's FINAL slice: %d of %d tensors, sha %s" % (m, n_t, psha[:12]))
        adapter_out = os.path.join(a.out_dir, "adapter")
        model.save_pretrained(adapter_out)
        log("saved the concatenated 16-layer adapter -> %s" % adapter_out)
    vram = {"peak_alloc_gib": gib(torch.cuda.max_memory_allocated()),
            "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
            "device_total_gib": gib(torch.cuda.get_device_properties(0).total_memory),
            "base_alloc_gib": gib(alloc_base), "cap_fraction": frac}
    log("peak VRAM: %.3f GiB alloc / %.3f GiB reserved of %.3f" % (
        vram["peak_alloc_gib"], vram["peak_reserved_gib"], vram["device_total_gib"]))
    clip_record = {
        "mode": a.clip_mode, "max_norm": SPEC["clip"],
        "applied_norm": ("16-layer global norm, d1_sft.py:367 reproduced from this box's weights "
                         "(peer slice as last synced; base %s)" % base_dtype
                         if a.clip_mode == "global" else "owned slice only (pre-fix behaviour)"),
        "gnorms_key_is": "owned-slice pre-clip norm per step (liveness gate: gnorm_max > 0)",
        "applied_bound_steps": clip_bound, "applied_bound_step_ids": bound_steps,
        "global_bound_steps": n_global_bound if a.clip_mode == "global" else None,
        "local_bound_steps": n_local_bound,
        "global_bound_local_not_steps": n_global_not_local if a.clip_mode == "global" else None,
        "max_global": max(g_global) if g_global else None,
        "max_local": max(gnorms) if gnorms else None,
        "max_peer": max(g_peer) if g_peer else None,
        "coef_min": min(coefs) if coefs else None,
        "how_to_read": ("With mode=global on BOTH boxes the owned slice received the twin's factor "
                        "(clip_coef[step]) on every step, so a loss against the twin cannot be a "
                        "clip-regime artefact. global_bound_local_not_steps counts the steps on "
                        "which a per-slice clip would have left this box's update UNscaled while "
                        "the twin scaled it; under mode=local that many steps escaped the twin's "
                        "regime, each by 1/coef. global_bound_steps == 0 means the clip never "
                        "bound and the question is moot for this box.")}
    manifest = {"role": "trainer", "box": box, "arm": a.arm, "H": a.H, "seed": a.seed,
                "deterministic": bool(getattr(a, "deterministic", False)),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "bitsandbytes": _bnb_version(),
                "run_id": a.run_id, "own_layers": own_layers, "peer_layers": peer_layers,
                "trainable_total": SPEC["trainable_total"], "trainable_owned": n_own,
                "frozen_peer": n_peer, "steps": step, "planned_steps": total_steps,
                "steps_limit": a.steps_limit, "traces_seen": seen, "hours": el, "losses": losses,
                "gnorms": gnorms, "gnorm_max": max(gnorms) if gnorms else None,
                "gnorm_global": g_global, "gnorm_global_max": max(g_global) if g_global else None,
                "gnorm_peer": g_peer, "gnorm_peer_max": max(g_peer) if g_peer else None,
                "clip_coef": coefs, "clip_coef_min": min(coefs) if coefs else None,
                "clip_bound_steps": clip_bound, "clip": clip_record, "step_seconds": step_s,
                "step_seconds_median": sorted(step_s)[len(step_s) // 2] if step_s else None,
                "seq_stats": stats, "seq_len": SPEC["seq_len"], "model": a.model_dir,
                "data": a.data, "data_sha256": data_sha, "token_parity_sha": tp_sha,
                "selected_ids_sha256": sel["selected_ids_sha256"],
                "selected_order_sha256": sel["selected_order_sha256"],
                "rows_in_file": sel["rows_in_file"], "rows_selected": sel["rows_selected"],
                "micro_batch": SPEC["micro_batch"], "accum": SPEC["accum"], "lr": SPEC["lr"],
                "rank": SPEC["rank"], "warmup": SPEC["warmup"], "surface": "attention-only",
                "wire_dtype": a.wire_dtype, "syncs": sync_log, "n_syncs": len(sync_log),
                "final_slice_sha": fsha, "adapter": adapter_out, "vram": vram,
                "base_dtype": base_dtype, "base_policy": a.base_policy,
                "expert_layout": layout, "anchor": anchor.stats(), "budget_stopped": budget_stopped,
                "argv": list(sys.argv), "final_checkpoint_not_best": True,
                "torch": torch.__version__}
    try:
        import transformers, peft
        manifest["transformers"] = transformers.__version__
        manifest["peft"] = peft.__version__
    except Exception:                                                    # noqa: BLE001
        pass
    path = os.path.join(a.out_dir, "TRAIN_%s.json" % box)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    anchor.put(json.dumps(manifest).encode("utf-8"), ns + "/%s/train_json" % box)
    log("wrote %s (+ published)" % path)
    if stats["truncated"] or stats["dropped"]:
        log("==== TERMINAL: VOID -- seq_stats %s (zero-truncation rule) ====" % json.dumps(stats))
        return 3
    if budget_stopped:
        log("==== TERMINAL: BUDGET-STOPPED at step %d of %d -- NOT COMPLETE ====" % (step, stop_at))
        return 4
    log("==== TERMINAL: WAN-GRADIENT TRAIN COMPLETE (%s, %d steps) ====" % (box, step))
    return 0


# ----------------------------------------------------------------------------- role: gate-footprint (4060)
def role_gate_footprint(a):
    """Design 7(a): load OLMoE nf4 under an explicit VRAM cap; assert experts are bnb Params4bit
    (not bf16) and memory_allocated <= 4.5 GiB after load; then time real training steps at the
    arm's exact shape (micro 2 x seq 1024, accum 8, grad-checkpoint) and refuse a ms/step cliff
    (WDDM sysmem paging shows as a 2-5x timing cliff with no OOM). Box-generic since 2026-09-08:
    on the 4060 it is the design's gate (published as wangrad/<run>/gate/footprint); on the 5090
    (`--box 5090 --base-dtype nf4`, under a transformers-4.x interpreter) it is the proof that
    the 5090 can produce the SAME unfused nf4 base the nf4 policy needs (published as
    gate/footprint_5090; nf4_twin_chain.py runs it as its first gate)."""
    import torch
    os.makedirs(a.out_dir, exist_ok=True)
    log = make_logger(os.path.join(a.out_dir, "gate_footprint.log"))
    log("=" * 78)
    base_dtype = resolve_base_dtype(a.box, a.base_dtype, a.base_policy)
    log("GATE 7(a) FOOTPRINT on the %s | base %s | cap %.2f GiB | model %s"
        % (a.box, base_dtype, a.cap_gib, a.model_dir))
    token = read_token(log)
    anchor = Anchor(a.anchor, token, log)
    res = {"gate": "7a_footprint", "box": a.box, "base_dtype": base_dtype, "base_policy": a.base_policy,
           "run_id": a.run_id, "cap_gib": a.cap_gib, "pass": False, "reasons": []}
    try:
        if base_dtype != "nf4":
            raise RuntimeError("the footprint gate is an nf4 gate; --base-dtype resolved to %s" % base_dtype)
        frac, total = set_vram_cap(a.box, a.cap_gib, log)
        res["cap_fraction"], res["device_total_gib"] = frac, gib(total)
        t_load = time.time()
        tok, model = load_base(a.box, a.model_dir, log, base_dtype)
        res["load_secs"] = time.time() - t_load
        alloc = torch.cuda.memory_allocated()
        res["alloc_after_load_gib"] = gib(alloc)
        log("allocated after load: %.3f GiB (bound 4.5)" % gib(alloc))
        ok, layout = assert_expert_layout(model, log)
        res["expert_layout"] = layout
        if not ok:
            res["reasons"].append("experts not fully nf4 (fused=%d, not4bit=%d)"
                                  % (layout["n_fused"], layout["n_not_4bit"]))
        if gib(alloc) > 4.5:
            res["reasons"].append("alloc_after_load %.3f GiB > 4.5" % gib(alloc))
        try:
            import transformers, peft, bitsandbytes
            res["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                               "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__}
            if not transformers.__version__.startswith("4."):
                res["reasons"].append("transformers %s is not 4.x (fused-expert trap)"
                                      % transformers.__version__)
        except Exception as e:                                           # noqa: BLE001
            res["reasons"].append("version probe failed: %s" % repr(e)[:80])
        model = attach_lora(model, SPEC["seeds"]["s1"], log)
        freeze_to_owned(model, SPEC["own_layers"][a.box], log, peer_grad=(a.clip_mode == "global"))
        rows, _ = select_rows(a.data, SPEC["n_traces"], SPEC["seeds"]["s1"])
        stats = {"truncated": 0, "dropped": 0, "eos_restored": 0}
        batchify = make_batchify(tok, SPEC["seq_len"], stats)
        # the arm's exact clip path (clip_step), so the s/step and footprint measured here include
        # the peer-slice gradient that clip-mode global computes on every step
        own, peer_, clipset = lora_sets(model, SPEC["own_layers"][a.box])
        opt = torch.optim.AdamW(own, lr=SPEC["lr"], betas=SPEC["betas"], weight_decay=0.0)
        g_loc, g_glob = [], []
        model.train()
        times, i = [], 0
        for s in range(a.n_steps):
            ts = time.time()
            opt.zero_grad(set_to_none=True)
            for _ in range(SPEC["accum"]):
                chunk = rows[i:i + SPEC["micro_batch"]]
                i += SPEC["micro_batch"]
                xb, lab = batchify(chunk)
                if xb is None:
                    continue
                out = model(input_ids=xb[0], attention_mask=xb[1], labels=lab)
                (out.loss / SPEC["accum"]).backward()
            gn, loc, pn, coef = clip_step(own, peer_, clipset, SPEC["clip"], a.clip_mode)
            g_loc.append(loc)
            g_glob.append(gn)
            opt.step()
            torch.cuda.synchronize()
            times.append(time.time() - ts)
            log("  step %d: %.2f s | alloc %.3f GiB | reserved %.3f GiB | gnorm own %.3f global %.3f"
                % (s + 1, times[-1], gib(torch.cuda.memory_allocated()),
                   gib(torch.cuda.memory_reserved()), loc, gn))
        body = times[2:] if len(times) > 3 else times
        med = sorted(body)[len(body) // 2]
        cliff = max(body) / med if med else float("inf")
        res.update({"step_seconds": times, "step_seconds_median": med, "cliff_ratio": cliff,
                    "peak_alloc_gib": gib(torch.cuda.max_memory_allocated()),
                    "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
                    "seq_stats": stats, "clip_mode": a.clip_mode,
                    "gnorm_local": g_loc, "gnorm_global": g_glob,
                    "est_hours_per_arm_625_steps": EXPECT_STEPS * med / 3600.0})
        if g_loc and not (max(g_loc) > 0.0):     # HIGH-1's liveness, caught in the first hour
            res["reasons"].append("owned slice received NO gradient in %d steps (gnorm_local max 0)"
                                  % len(g_loc))
        if cliff > a.cliff_max:
            res["reasons"].append("ms/step cliff %.2fx > %.2f (WDDM spill signature)" % (cliff, a.cliff_max))
        if stats["truncated"] or stats["dropped"]:
            res["reasons"].append("seq_stats %s" % json.dumps(stats))
    except Exception as e:                                               # noqa: BLE001
        res["reasons"].append("EXCEPTION %s" % repr(e)[:300])
        log("EXCEPTION: %r" % e)
    res["pass"] = not res["reasons"]
    log("GATE 7(a) %s | %s" % ("PASS" if res["pass"] else "FAIL", "; ".join(res["reasons"]) or "ok"))
    log("GATE-JSON: " + json.dumps(res))
    with io.open(os.path.join(a.out_dir, "FOOTPRINT.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if token:
        anchor.put(json.dumps(res).encode("utf-8"), "wangrad/%s/gate/footprint%s"
                   % (a.run_id, "" if a.box == "4060" else "_" + a.box))
    return 0 if res["pass"] else 3


# ----------------------------------------------------------------------------- role: gate-basece (both)
def role_gate_basece(a):
    """Design 7(b): per-token CE of the base on a fixed 128-example probe (first 128 usable rows
    in FILE order, completion tokens only, prompt masked exactly as training). Each box runs it
    on the base dtype its policy prescribes (mixed: nf4 on the 4060, bf16 on the 5090; nf4: nf4
    on both); the chain applies the VOID threshold. Publishes wangrad/<run>/gate/basece_<box>."""
    import torch
    os.makedirs(a.out_dir, exist_ok=True)
    log = make_logger(os.path.join(a.out_dir, "gate_basece_%s.log" % a.box))
    log("=" * 78)
    base_dtype = resolve_base_dtype(a.box, a.base_dtype, a.base_policy)
    log("GATE 7(b) BASE-CE on the %s | base %s (policy %s) | model %s | probe n=%d"
        % (a.box, base_dtype, a.base_policy, a.model_dir, a.probe_n))
    token = read_token(log)
    anchor = Anchor(a.anchor, token, log)
    data_sha = sha256_file(a.data)
    rows = probe_rows(a.data, a.probe_n)
    set_vram_cap(a.box, a.cap_gib, log)
    tok, model = load_base(a.box, a.model_dir, log, base_dtype)
    if base_dtype == "nf4":                  # a CE on a half-quantized base is a wrong number
        ok, layout = assert_expert_layout(model, log)
        if not ok:
            log("VOID (design 7a): expert layout on the %s is not fully nf4 -- this interpreter "
                "cannot produce the nf4 base (needs transformers 4.x); no CE recorded" % a.box)
            return 3
    model.eval()
    stats = {"truncated": 0, "dropped": 0, "eos_restored": 0}
    batchify = make_batchify(tok, SPEC["seq_len"], stats)
    tot_nll, tot_tok, per_row = 0.0, 0, []
    with torch.no_grad():
        for r in rows:
            xb, lab = batchify([r])
            if xb is None:
                continue
            out = model(input_ids=xb[0], attention_mask=xb[1])
            lg = out.logits[:, :-1, :].float()
            tgt = lab[:, 1:]
            keep = tgt != -100
            lp = torch.log_softmax(lg, dim=-1).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            nll = float(-(lp * keep).sum())
            n = int(keep.sum())
            tot_nll += nll
            tot_tok += n
            per_row.append([nll, n])
    ce = tot_nll / max(tot_tok, 1)
    res = {"gate": "7b_basece", "box": a.box, "run_id": a.run_id, "model": a.model_dir,
           "dtype": base_dtype, "base_policy": a.base_policy, "probe_n": len(per_row),
           "probe_source": "first %d usable rows in file order" % a.probe_n,
           "data_sha256": data_sha, "tokens": tot_tok, "ce_per_token": ce, "per_row": per_row,
           "seq_stats": stats, "torch": torch.__version__}
    log("CE per completion token = %.6f over %d tokens (%d rows)" % (ce, tot_tok, len(per_row)))
    log("GATE-JSON: " + json.dumps({k: v for k, v in res.items() if k != "per_row"}))
    with io.open(os.path.join(a.out_dir, "BASECE_%s.json" % a.box), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if token:
        anchor.put(json.dumps(res).encode("utf-8"), "wangrad/%s/gate/basece_%s" % (a.run_id, a.box))
    return 0


# ----------------------------------------------------------------------------- role: stage-4060
def role_stage_4060(a):
    """Runs ON the 4060 (stdlib + huggingface_hub only). Makes the box able to train:
      1. TRUE base present at SPEC[model_dir_4060] with every shard's full sha256 == the 5090's
         (else fetched by OUTBOUND HF download -- the box fetches its own base, never pushed from
         home -- and re-hashed; a mismatch after download is an ABORT, not a retry).
      2. corpus fetched from the anchor by sha (WAN), verified.
    Publishes wangrad/<run>/stage/4060 with the verdict."""
    os.makedirs(a.out_dir, exist_ok=True)
    log = make_logger(os.path.join(a.out_dir, "stage_4060.log"))
    log("=" * 78)
    log("STAGE 4060 | base %s | corpus %s" % (a.model_dir, a.data))
    token = read_token(log)
    anchor = Anchor(a.anchor, token, log)
    res = {"stage": "4060", "run_id": a.run_id, "base_ok": False, "corpus_ok": False, "reasons": []}
    # Fail fast on the interpreter BEFORE a 13.8 GB download: the trainer runs under THIS python.
    try:
        import torch, transformers, peft, bitsandbytes
        res["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                           "cuda": bool(torch.cuda.is_available()),
                           "transformers": transformers.__version__, "peft": peft.__version__,
                           "bitsandbytes": bitsandbytes.__version__}
        log("versions: %s" % json.dumps(res["versions"]))
        if not transformers.__version__.startswith("4."):
            res["reasons"].append("transformers %s is not 4.x: OLMoE experts are fused, bnb cannot "
                                  "quantize them, the base cannot fit 8 GB" % transformers.__version__)
        if not torch.cuda.is_available():
            res["reasons"].append("torch.cuda.is_available() is False")
    except Exception as e:                                               # noqa: BLE001
        res["reasons"].append("env import failed: %s" % repr(e)[:160])
    if res["reasons"]:
        res["ok"] = False
        log("STAGE FAIL (env): %s" % "; ".join(res["reasons"]))
        log("GATE-JSON: " + json.dumps(res))
        with io.open(os.path.join(a.out_dir, "STAGE.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        if token:
            anchor.put(json.dumps(res).encode("utf-8"), "wangrad/%s/stage/4060" % a.run_id)
        return 3

    def base_matches():
        bad = []
        for name, (sha, size) in TRUE_BASE_SHA.items():
            p = os.path.join(a.model_dir, name)
            if not os.path.exists(p) or os.path.getsize(p) != size:
                bad.append("%s missing/size" % name)
                continue
            if name.endswith(".safetensors") or name == "config.json":
                got = sha256_file(p)
                if got != sha:
                    bad.append("%s sha %s != %s" % (name, got[:12], sha[:12]))
        return bad

    bad = base_matches() if os.path.isdir(a.model_dir) else ["dir absent"]
    if bad and not a.no_download:
        log("base not verified (%s) -> outbound HF download of %s" % ("; ".join(bad)[:200], SPEC["hf_repo"]))
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ.pop(k, None)
        try:
            from huggingface_hub import snapshot_download
            t0 = time.time()
            snapshot_download(repo_id=SPEC["hf_repo"], local_dir=a.model_dir,
                              allow_patterns=["*.safetensors", "*.json", "*.txt"],
                              max_workers=4)
            log("download done in %.1f min" % ((time.time() - t0) / 60.0))
        except Exception as e:                                           # noqa: BLE001
            res["reasons"].append("download failed: %s" % repr(e)[:200])
            log("download failed: %r" % e)
        bad = base_matches()
    res["base_ok"] = not bad
    if bad:
        res["reasons"].append("base identity: " + "; ".join(bad)[:300])
        log("BASE IDENTITY FAILED: %s" % "; ".join(bad)[:300])
    else:
        log("base identity OK: all shards match the 5090's true base by full sha256")
    # corpus by sha through the anchor (data plane)
    try:
        need = a.corpus_sha
        if os.path.exists(a.data) and sha256_file(a.data) == need:
            log("corpus already present and verified")
        else:
            b = anchor.get(need)
            os.makedirs(os.path.dirname(a.data), exist_ok=True)
            with open(a.data, "wb") as fh:
                fh.write(b)
            log("corpus fetched: %d B, sha %s" % (len(b), need[:16]))
        res["corpus_ok"] = sha256_file(a.data) == need
    except Exception as e:                                               # noqa: BLE001
        res["reasons"].append("corpus: %s" % repr(e)[:200])
    res["ok"] = res["base_ok"] and res["corpus_ok"]
    log("STAGE %s | %s" % ("OK" if res["ok"] else "FAIL", "; ".join(res["reasons"]) or "ok"))
    log("GATE-JSON: " + json.dumps(res))
    with io.open(os.path.join(a.out_dir, "STAGE.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if token:
        anchor.put(json.dumps(res).encode("utf-8"), "wangrad/%s/stage/4060" % a.run_id)
    return 0 if res["ok"] else 3


# ----------------------------------------------------------------------------- role: chain (5090)
def role_chain(a):
    """5090 orchestrator. Everything the design registers as a stop condition is executable code
    here, quoting the clause. Phases (each writes an artefact; a re-run skips onto artefacts):
      P0 preflight (no GPU)  P1 A/A on the base  P2 stage the 4060  P3 gate 7(a)  P4 gate 7(b)
      P5 gate 7(c)  P6 arm h1_s1 + KILL  P7 arms h25_s1, h25_s2, indep_s1  P8 summary."""
    import subprocess
    HERE = "E:/aiCrypto/scratchpad"
    sys.path.insert(0, HERE + "/stylectl")
    sys.path.insert(0, HERE)
    import style_control_chain as scc      # noqa: E402  (BASE, BASE_BATTERY, BASE_ARC, Chain)
    import paired_read                     # noqa: E402
    import post_t30_chain as pt            # noqa: E402  (PostChain.step validates artefacts)
    import c4060                           # noqa: E402  (control plane: spawn/logs/ps, read-only)

    class WanChain(pt.PostChain):
        def __init__(self, a):
            pt.PostChain.__init__(self, a)
            self.anchor = None
            self.remote = {}                       # phase -> remote pid
            self.summary = {"design": SPEC["design"], "started": now(), "phases": {}, "arms": {}}

        # ------------------------------------------------------------ small helpers
        def save(self):
            with io.open(self.run + "/WANGRAD_SUMMARY.json", "w", encoding="utf-8") as fh:
                json.dump(self.summary, fh, indent=1, default=str)

        # ---- base policy (2026-09-08): which twin is the control, which interpreter the 5090 uses
        def twins(self):
            """seed -> control-twin dir for the policy: the banked bf16 D1-TB pair under `mixed`,
            the nf4 twin (nf4_twin_chain.py) under `nf4` -- a both-nf4 arm read against the bf16
            twin would carry base quantization as a confound on all 16 layers."""
            return SPEC["twin_nf4"] if self.a.base_policy == "nf4" else SPEC["twin"]

        def twin_dir(self, seed):
            return self.twins()[seed]

        def py_for(self, box="5090"):
            """The interpreter the 5090's GPU roles run under: scc.PY (C:/Python313, transformers
            5.8.1) for a bf16 base; --py5090 (a transformers-4.x env, probed in preflight) when
            the policy puts the 5090 on nf4. The 4060 always runs --py4060."""
            if box == "5090" and BASE_POLICIES[self.a.base_policy]["5090"] == "nf4":
                return self.a.py5090 or scc.PY        # empty is refused in preflight; the role VOIDs too
            return scc.PY

        def step_with(self, py, name, argv, artefact, budget_h):
            """Chain.step under interpreter `py`. Chain.step builds `[PY] + argv` from the
            module-level scc.PY at call time, so it is swapped for the duration of the call."""
            old = scc.PY
            scc.PY = py
            try:
                return self.step(name, argv, artefact, budget_h)
            finally:
                scc.PY = old

        def phase(self, name, status, **kw):
            d = {"status": status, "at": now()}
            d.update(kw)
            self.summary["phases"][name] = d
            self.log("PHASE %s -> %s %s" % (name, status, json.dumps(kw, default=str)[:400]))
            self.save()

        def terminal(self, status, why):
            self.summary["terminal"] = {"status": status, "why": why, "at": now()}
            self.save()
            self.log("==== TERMINAL: WAN-GRADIENT CHAIN %s -- %s ====" % (status, why))

        def remote_sh(self, cmd, timeout=120):
            r = c4060.sh(cmd, timeout=timeout)
            return r.get("rc"), (r.get("stdout") or ""), (r.get("stderr") or "")

        def remote_gpu(self):
            """(memory.used MiB, util %) on the 4060, or (-1, -1)."""
            try:
                rc, out, _ = self.remote_sh(
                    "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits")
                u, g = [int(x.strip()) for x in out.strip().splitlines()[0].split(",")]
                return u, g
            except Exception as e:                                       # noqa: BLE001
                self.log("4060 nvidia-smi unreadable: %s" % repr(e)[:100])
                return -1, -1

        def wait_4060_idle(self):
            """The 4060 runs a sampler today (7.8 GB used at 97%). NEVER killed here: wait up to
            --idle-timeout-s for < 1500 MiB used, else refuse (the owner stops the sampler)."""
            t0 = time.time()
            while time.time() - t0 < self.a.idle_timeout_s:
                used, util = self.remote_gpu()
                self.log("  4060: %d MiB used, %d%% util" % (used, util))
                if 0 <= used < 1500:
                    return True
                if self.a.plan:
                    return True
                time.sleep(60)
            return False

        def remote_env(self, offline=True):
            # 2026-09-09: D:	f4env is a --system-site-packages venv, so it inherits the base
            # Python 3.10's TensorFlow. transformers/peft import TF by default; that TF's
            # ml_dtypes._custom_floats is a numpy-1.x binary under numpy 2.2.6, so `import peft`
            # dies with "numpy.core.umath failed to import" -- which reads as a numpy problem and
            # is not one. One-variable control on the 4060, same interpreter and probe:
            #   without USE_TF -> ImportError: numpy.core.umath failed to import
            #   with USE_TF=0  -> PEFT OK 0.18.0
            # Without this the trainer dies at attach_lora AFTER a 13.8 GB stage.
            env = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                   "USE_TF": "0", "USE_TORCH": "1", "TRANSFORMERS_NO_TF": "1",
                   SPEC["token_env"]: self.token}
            if offline:                                       # the HF-hub pre-load deadlock trap
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
            return env

        def remote_env_probe(self):
            """Versions under --py4060, via a file-based probe (quotes do not survive the RPC).
            Read-only. Returns a dict; {'ok': False, 'error': ...} on any failure."""
            probe = SPEC["remote_root"] + r"\_env_probe.py"
            lines = [
                "import json, sys, shutil",
                "d = dict(exe=sys.executable, python=sys.version.split()[0])",
                "try:",
                "    import torch; d['torch'] = torch.__version__; d['cuda'] = bool(torch.cuda.is_available())",
                "except Exception as e: d['torch_error'] = repr(e)[:120]",
                "try:",
                "    import transformers; d['transformers'] = transformers.__version__",
                "except Exception as e: d['transformers_error'] = repr(e)[:120]",
                "try:",
                "    import peft; d['peft'] = peft.__version__",
                "except Exception as e: d['peft_error'] = repr(e)[:120]",
                "try:",
                "    import bitsandbytes as b; d['bitsandbytes'] = b.__version__",
                "except Exception as e: d['bnb_error'] = repr(e)[:120]",
                "try:",
                "    d['free_gb_D'] = shutil.disk_usage('D:/').free / 1e9",
                "except Exception as e: d['disk_error'] = repr(e)[:120]",
                "d['ok'] = bool(d.get('torch')) and bool(d.get('transformers'))",
                "print('ENVJSON ' + json.dumps(d), flush=True)",
            ]
            for l in lines:
                assert not any(c in l for c in "&|<>^%"), l
            cmd = "mkdir %s 2>nul & " % SPEC["remote_root"]
            # joined with a bare `&`: cmd attaches the space before ` &` to the echoed text
            # (measured 2026-09-08: every line came back with a trailing space, harmless to
            # python but not byte-identical).
            cmd += "&".join(["echo %s>%s" % (lines[0], probe)] +
                            ["echo %s>>%s" % (l, probe) for l in lines[1:]])
            rc, out, err = self.remote_sh(cmd, timeout=60)
            if rc not in (0, None):
                return {"ok": False, "error": "probe write rc=%s %s" % (rc, err[:120])}
            # Same TF trap as remote_env(): this probe runs the interpreter DIRECTLY, so it
            # must carry the vars itself or preflight reports peft broken on a sound env.
            rc, out, err = self.remote_sh(
                "set USE_TF=0& set USE_TORCH=1& set TRANSFORMERS_NO_TF=1& "
                + self.a.py4060 + " " + probe, timeout=300)
            for line in (out or "").splitlines():
                if line.startswith("ENVJSON "):
                    return json.loads(line[8:])
            return {"ok": False, "error": "rc=%s stdout=%s stderr=%s" % (rc, out[-160:], err[-160:])}

        def bootstrap_lines(self, argv):
            """A tiny stdlib bootstrap written on the 4060 via cmd `echo` lines (no quotes cross
            the RPC): GET this file by sha from the anchor, verify, exec the role with argv."""
            D = SPEC["remote_root"]
            lines = [
                "import urllib.request, hashlib, os, subprocess, sys",
                "A = '%s'" % SPEC["anchor"],
                "D = r'%s'" % D,
                "os.makedirs(D, exist_ok=True)",
                "b = urllib.request.urlopen(A + '/o/%s', timeout=600).read()" % self.script_sha,
                "assert hashlib.sha256(b).hexdigest() == '%s', 'script sha mismatch'" % self.script_sha,
                "open(D + r'\\wan_gradient_chain.py', 'wb').write(b)",
                "print('BOOT ok', len(b), sys.executable, flush=True)",
                "rc = subprocess.call([sys.executable, D + r'\\wan_gradient_chain.py'] + %s)" % json.dumps(argv),
                "print('BOOT child rc', rc, flush=True)",
                "sys.exit(rc)",
            ]
            for l in lines:                    # cmd metacharacters would break `echo`
                assert not any(c in l for c in "&|<>^%"), l
            return lines

        def spawn_remote(self, tag, argv, offline=True):
            """Write the bootstrap on the 4060 and /spawn it (control plane). Returns the pid."""
            boot = SPEC["remote_root"] + r"\_boot_%s.py" % tag
            lines = self.bootstrap_lines(argv)
            cmd = "mkdir %s 2>nul & " % SPEC["remote_root"]
            cmd += "&".join(["echo %s>%s" % (lines[0], boot)] +
                            ["echo %s>>%s" % (l, boot) for l in lines[1:]])
            self.log("REMOTE %s | %s" % (tag, " ".join(argv)))
            if self.a.plan:
                return "plan"
            rc, out, err = self.remote_sh(cmd, timeout=60)
            if rc not in (0, None):
                raise RuntimeError("bootstrap write failed rc=%s: %s" % (rc, err[:200]))
            # EXPLICIT interpreter: a bare `python` is rewritten by the control agent to ITS OWN
            # interpreter (gpu_testbed/control_agent.py:77), and both interpreters probed on the
            # 4060 on 2026-09-08 carry transformers 5.8.1 (fused experts, cannot nf4). The
            # trainer needs the tf4.x venv named by --py4060, verified in preflight.
            # CRITICAL-1 (review 2026-09-08): the agent shlex.splits a STRING command
            # (control_agent.py:149) and POSIX shlex eats backslashes, so an unquoted
            # D:\... interpreter arrives as D:wangradvenvScriptspython.exe and every
            # remote role dies with FileNotFoundError. Quoting survives the split.
            _cmd = '"%s" "%s"' % (self.a.py4060, boot)
            pid, reply = c4060.spawn(_cmd, cwd=SPEC["remote_root"],
                                     env=self.remote_env(offline))
            if not pid:
                raise RuntimeError("spawn failed: %s" % json.dumps(reply)[:300])
            self.remote[tag] = pid
            io.open(self.run + "/remote_pids.txt", "a").write("%s %s %s\n" % (now(), tag, pid))
            self.log("  spawned on the 4060: pid %s" % pid)
            return pid

        def remote_alive(self, pid):
            try:
                alive, rc = c4060.alive(pid, timeout=120)
                return alive
            except Exception:                                            # noqa: BLE001
                return True                                              # unknown != dead

        def pull_remote_log(self, tag, pid):
            try:
                t, _ = c4060.logs(pid, 12000, timeout=120)
                with io.open(self.run + "/logs/remote_%s.log" % tag, "w", encoding="utf-8") as fh:
                    fh.write(t)
                return t
            except Exception as e:                                       # noqa: BLE001
                return "(logs unavailable: %s)" % repr(e)[:80]

        def wait_remote_artefact(self, tag, pid, name, timeout_s, poll_s=30):
            """Score the ARTEFACT (an object on the anchor), never the announcement. The pid is
            only an early-abort signal: process gone AND no artefact -> FAIL."""
            if self.a.plan:
                return None
            t0 = time.time()
            last_pull = 0
            while True:
                sha = self.anchor.resolve(name)
                if sha:
                    return json.loads(self.anchor.get(sha).decode("utf-8"))
                if time.time() - last_pull > 300:
                    tail = self.pull_remote_log(tag, pid)
                    self.log("  [%s] remote tail: %s" % (tag, tail[-300:].replace("\n", " | ")))
                    last_pull = time.time()
                if not self.remote_alive(pid):
                    time.sleep(20)
                    sha = self.anchor.resolve(name)
                    if sha:
                        return json.loads(self.anchor.get(sha).decode("utf-8"))
                    tail = self.pull_remote_log(tag, pid)
                    raise RuntimeError("remote %s (pid %s) ended without publishing %s | tail: %s"
                                       % (tag, pid, name, tail[-400:]))
                if time.time() - t0 > timeout_s:
                    raise RuntimeError("timeout %.0f s waiting for %s" % (timeout_s, name))
                if self.stop_requested():
                    raise RuntimeError("STOP requested while waiting for %s" % name)
                time.sleep(poll_s)

        def local_trainer_argv(self, arm_tag, kind, seed, run_id, out_dir, steps_limit=0,
                               resume_from=""):
            # 2026-09-10: THIS file, never the literal "wan_gradient_chain.py". Chain.step runs
            # [PY] + argv from cwd E:/aiCrypto/scratchpad (stylectl/style_control_chain.py:212),
            # so the literal launched the ckpt-LESS original on the 5090 while the 4060 ran the
            # bytes of this runner -- only one side of the pair ever checkpointed.
            return [os.path.basename(os.path.abspath(__file__)), "--role", "trainer", "--box", "5090",
                    "--arm", kind, "--H", str(H_OF[kind]), "--seed", str(seed),
                    "--run-id", run_id, "--out-dir", out_dir,
                    "--model-dir", SPEC["model_dir_5090"], "--data", SPEC["corpus_5090"],
                    "--expect-data-sha", self.corpus_sha, "--steps-limit", str(steps_limit),
                    "--peer-timeout-s", str(self.a.peer_timeout_s), "--poll-s",
                    str(self.a.poll_s_h1 if H_OF[kind] == 1 else self.a.poll_s),
                    "--clip-mode", self.a.clip_mode,
                    "--max-hours", str(self.a.train_max_h),
                    "--base-policy", self.a.base_policy,
                    "--base-dtype", BASE_POLICIES[self.a.base_policy]["5090"],
                    "--ckpt-every", str(self.a.ckpt_every)] + self._det_argv("5090") + self._resume_argv(resume_from)

        def _det_argv(self, box):
            """--deterministic-boxes 5090[,4060]: which trainer(s) get deterministic kernels."""
            boxes = [b.strip() for b in (self.a.deterministic_boxes or "").split(",") if b.strip()]
            return ["--deterministic"] if box in boxes else []

        def _resume_argv(self, resume_from):
            return ["--resume-from", resume_from] if resume_from else []

        def remote_trainer_argv(self, kind, seed, run_id, steps_limit=0, out_dir="", resume_from=""):
            # out_dir is pinned by resume_plan to <remote_root>\<original run_id> so the 4060's
            # checkpoints stay in ONE place while the anchor namespace (run_id) changes per attempt.
            return ["--role", "trainer", "--box", "4060", "--arm", kind, "--H", str(H_OF[kind]),
                    "--seed", str(seed), "--run-id", run_id,
                    "--out-dir", out_dir or (SPEC["remote_root"] + "\\" + run_id),
                    "--model-dir", SPEC["model_dir_4060"], "--data", SPEC["corpus_4060"],
                    "--expect-data-sha", self.corpus_sha, "--steps-limit", str(steps_limit),
                    "--cap-gib", str(self.a.cap_4060_gib),
                    "--peer-timeout-s", str(self.a.peer_timeout_s), "--poll-s",
                    str(self.a.poll_s_h1 if H_OF[kind] == 1 else self.a.poll_s),
                    "--clip-mode", self.a.clip_mode,
                    "--max-hours", str(self.a.train_max_h_4060),
                    "--base-policy", self.a.base_policy,
                    "--base-dtype", BASE_POLICIES[self.a.base_policy]["4060"],
                    "--ckpt-every", str(self.a.ckpt_every)] + self._det_argv("4060") + self._resume_argv(resume_from)

        # ------------------------------------------------------------ P0 preflight
        def preflight(self):
            self.log("P0 preflight (no GPU)")
            probs = []
            # corpus identity (design section 5 VOID: corpus sha mismatch)
            if not os.path.exists(SPEC["corpus_5090"]):
                probs.append("corpus missing")
            else:
                self.corpus_sha = sha256_file(SPEC["corpus_5090"])
                tw = json.load(io.open(SPEC["twin"][20260829] + "/TRAIN.json", encoding="utf-8"))
                if tw.get("data_sha256") != self.corpus_sha:
                    probs.append("corpus sha %s != banked twin's %s"
                                 % (self.corpus_sha[:16], str(tw.get("data_sha256"))[:16]))
            # the banked control twin, field by field (C-SOLO was cut as its duplicate)
            #
            # 2026-09-09 SCOPE (not a weakening): this gate exists so a run that cannot produce a
            # verdict never starts. Only `basece` (reads the twin's CE drop to set its 7(b)
            # threshold) and `arms` (the twin IS the C-SOLO control) consume a twin. `stage` and
            # `footprint` consume NOTHING from it -- footprint's whole job is to measure the
            # 4060's s/step so the per-arm cost is known BEFORE the twins finish, which is
            # precisely the 2026-08-14 lesson (cost the measurement before buying the run).
            # Blocking those on a twin made a preparation run unable to pass a gate written for
            # verdict runs -- the reachability failure of 20-judgment's 2026-08-29 entry.
            # The gate keeps every tooth for any run that requests a twin-consuming phase.
            _only = set(self.a.only.split(",")) if self.a.only else None
            _twin_consumers = {"basece", "arms"}
            needs_twin = _only is None or bool(_only & _twin_consumers)
            # Which SEEDS are actually consumed. basece reads seed 20260829's twin (7b threshold);
            # each arm reads its own stag's twin as C-SOLO. Under --stop-after-h1 only h1_s1 runs,
            # so seed 20260830's twin is never read and demanding it would idle the priority job
            # for hours behind a control nobody bought.
            _wanted = set(x for x in (self.a.arms or "").split(",") if x)   # --arms filter
            _armset = ARMS[:1] if self.a.stop_after_h1 else [a for a in ARMS if not _wanted or a[0] in _wanted]
            _seeds_needed = {SPEC["seeds"][st] for _, _, st in _armset} if (
                _only is None or "arms" in _only) else set()
            if _only is None or "basece" in _only:
                _seeds_needed.add(20260829)
            self.twin_batt = {}
            twins = self.twins()
            self.log("  base policy %s -> control twin %s%s"
                     % (self.a.base_policy, json.dumps(twins),
                        "" if needs_twin else
                        "  [twin checks SKIPPED: --only %s consumes no twin; any of %s would "
                        "re-arm them]" % (",".join(sorted(_only)), ",".join(sorted(_twin_consumers)))))
            for seed, d in ([(k, v) for k, v in twins.items() if k in _seeds_needed]
                            if needs_twin else []):
                try:
                    t = json.load(io.open(d + "/TRAIN.json", encoding="utf-8"))
                except Exception as e:                                   # noqa: BLE001
                    probs.append("twin %s TRAIN.json unreadable: %s%s"
                                 % (seed, repr(e)[:60],
                                    " -- policy nf4 needs the nf4 CONTROL twin: run "
                                    "scratchpad/nf4_twin_chain.py first (both seeds)"
                                    if self.a.base_policy == "nf4" else ""))
                    continue
                if self.a.base_policy == "nf4" and (
                        t.get("base_dtype") != "nf4" or not (t.get("expert_layout") or {}).get("ok")):
                    probs.append("twin %s at %s is not an nf4-base twin (base_dtype %s, expert layout "
                                 "ok %s)" % (seed, d, t.get("base_dtype"),
                                             (t.get("expert_layout") or {}).get("ok")))
                want = {"seed": seed, "seq_len": SPEC["seq_len"], "steps": EXPECT_STEPS,
                        "traces_seen": SPEC["n_traces"], "trainable": SPEC["trainable_total"],
                        "micro_batch": SPEC["micro_batch"], "accum": SPEC["accum"], "lr": SPEC["lr"],
                        "rank": SPEC["rank"], "warmup": SPEC["warmup"], "model": SPEC["model_dir_5090"]}
                for k, v in want.items():
                    if t.get(k) != v:
                        probs.append("twin %s: %s=%s != %s" % (seed, k, t.get(k), v))
                if len(t.get("losses") or []) != EXPECT_STEPS:
                    probs.append("twin %s: %d losses (need %d)" % (seed, len(t.get("losses") or []), EXPECT_STEPS))
                ss = t.get("seq_stats") or {}
                if ss.get("truncated") or ss.get("dropped"):
                    probs.append("twin %s: seq_stats %s" % (seed, json.dumps(ss)))
                tag = os.path.basename(d)
                bp = d + "/BATTERY_%s.json" % tag
                ap = d + "/ARC_%s.json" % tag
                gp = d + "/S1_%s.json" % tag
                ep = twin_ext_path(seed, d)
                for p in (bp, ap, gp, ep):
                    if not os.path.exists(p):
                        probs.append("twin %s artefact missing: %s" % (seed, p))
                # design P1 / VOID -- RESOLUTION-AWARE, owner decision 2026-09-09.
                #
                # The original gate was a POINT WINDOW: 0.6 <= delta <= 1.5. That window is 0.9 pp
                # wide on a quantity this instrument measures with MDE ~0.73 pp, i.e. it is barely
                # wider than its own measurement error, so it cannot reliably separate pass from
                # fail. It fired on the nf4 twin at +0.5761 pp (p 0.0302) -- a REAL, significant
                # gain that missed the floor by 0.0239 pp, about 3% of the instrument's resolution,
                # and would have VOIDed the entire nf4 program on a difference nothing can measure.
                #
                # What the gate is actually for: the nf4 twin must reproduce the BANKED BF16 twin,
                # because the whole point of the control is that quantising the base did not change
                # what training does. So test THAT, against the resolution we actually have:
                #   (a) the twin shows a real gain at all            -> delta > 0 and p < 0.05
                #   (b) it is indistinguishable from the bf16 twin   -> |delta - banked| <= MDE
                #   (c) it has not run away upward                   -> delta <= 1.5 (kept as-is)
                # Banked bf16 references: seed 20260829 +0.9951 (MDE 0.7261),
                #                         seed 20260830 +0.7646 (MDE 0.7391).
                # This REPLACES an under-resolved threshold with the comparison it was proxying;
                # it does not lower a bar. Recorded in the prereg alongside this change.
                if os.path.exists(bp):
                    c = paired_read.compare(scc.BASE_BATTERY, bp, 0.75)
                    self.twin_batt[seed] = {k: c.get(k) for k in ("n", "disc", "delta", "p", "mde", "ci_lo", "ci_hi")}
                    banked = BF16_TWIN_BATTERY.get(seed)
                    gap = abs(c["delta"] - banked) if banked is not None else None
                    tol = max(c["mde"], BF16_TWIN_MDE.get(seed, 0.0))
                    real = c["delta"] > 0 and c["p"] < 0.05
                    matches = banked is None or gap <= tol
                    runaway = c["delta"] > 1.5
                    if not real:
                        probs.append("twin %s battery %+.4f pp (p %.3g): not a real gain -- the "
                                     "control did not train" % (seed, c["delta"], c["p"]))
                    elif runaway:
                        probs.append("twin %s battery %+.4f pp exceeds +1.5 pp -- runaway control"
                                     % (seed, c["delta"]))
                    elif not matches:
                        probs.append("twin %s battery %+.4f pp differs from the banked bf16 twin "
                                     "%+.4f pp by %.4f pp, MORE than the %.4f pp this instrument "
                                     "can resolve -- quantising the base changed what training does"
                                     % (seed, c["delta"], banked, gap, tol))
                    else:
                        self.log("  twin %s battery %+.4f pp p=%.3g MDE %.4f D=%d | banked bf16 "
                                 "%+.4f, gap %.4f <= tol %.4f -- P1 OK (resolution-aware)"
                                 % (seed, c["delta"], c["p"], c["mde"], c["disc"],
                                    banked if banked is not None else float("nan"),
                                    gap if gap is not None else float("nan"), tol))
            # every banked file the verdict pairs against must exist NOW, not after 1.4 h of training
            for p in (scc.BASE_BATTERY, scc.BASE_ARC, SPEC["base_gsm512"], SPEC["base_gsmext"]):
                if not os.path.exists(p):
                    probs.append("verdict reference missing: %s" % p)
            # anchor: health + a write round-trip (the design's execution precondition)
            try:
                h = self.anchor.health()
                self.log("  anchor: %s objects, %.2f GB" % (h.get("objects"), h.get("bytes", 0) / 1e9))
                if not self.a.plan:
                    probe = ("wangrad probe %s %s" % (self.run_id, time.time())).encode()
                    sha = self.anchor.put(probe, "wangrad/%s/probe" % self.run_id)
                    back = self.anchor.get(sha)
                    if back != probe:
                        probs.append("anchor round-trip mismatch")
                    else:
                        self.log("  anchor PUT+GET round-trip OK (%d B)" % len(probe))
            except Exception as e:                                       # noqa: BLE001
                probs.append("anchor: %s" % repr(e)[:120])
            if not self.token:
                probs.append("%s absent" % SPEC["token_env"])
            # the 4060 control plane + the interpreter the trainer will run under. Design section
            # 2's "the 4060 pins transformers 4.x" was TRUE in July and is FALSE today (probe
            # 2026-09-08: glmenv and Python313 both report transformers 5.8.1, torch 2.9.1+cu128,
            # bnb 0.48.2). Under 5.x OLMoE's experts are a fused nn.Parameter that bitsandbytes
            # cannot quantize, so the base cannot fit 8 GB -- gate 7(a) would fail by design.
            # The env probe below fails FAST, before a 13.8 GB download is started.
            try:
                rc, out, _ = self.remote_sh("ver", timeout=30)
                self.log("  4060 control agent reachable (rc %s)" % rc)
                env = self.remote_env_probe()
                self.summary["remote_env"] = env
                tfv = str(env.get("transformers", ""))
                env_probs = []
                if not env.get("ok"):
                    env_probs.append("4060 interpreter %s unusable: %s" % (self.a.py4060, env.get("error", "?")[:160]))
                elif not tfv.startswith("4."):
                    env_probs.append("4060 interpreter %s has transformers %s, need 4.x (fused-expert trap; "
                                     "owner action: a tf4.x venv, e.g. `python -m venv --system-site-packages "
                                     "D:\\wangrad\\venv` then `pip install transformers==4.57.1`)"
                                     % (self.a.py4060, tfv))
                elif not env.get("cuda"):
                    env_probs.append("4060 interpreter %s: torch.cuda.is_available() is False" % self.a.py4060)
                elif not env.get("bitsandbytes"):
                    env_probs.append("4060 interpreter %s: bitsandbytes missing" % self.a.py4060)
                elif not env.get("peft"):
                    # 2026-09-08: D:\tf4env reported peft_error "numpy.core.umath failed to
                    # import" (numpy ABI mismatch) and the old check let it through -- the
                    # trainer would have died at attach_lora after a 13.8 GB stage.
                    env_probs.append("4060 interpreter %s: peft does not import (%s) -- the trainer dies "
                                     "at attach_lora; owner action: reinstall numpy/peft consistently "
                                     "in that env" % (self.a.py4060, str(env.get("peft_error", "missing"))[:120]))
                else:
                    self.log("  4060 env OK: %s" % json.dumps(env)[:300])
                free_d = env.get("free_gb_D")
                if free_d is not None and free_d < 30:
                    env_probs.append("4060 D: has %.1f GB free (< 30 needed for the base)" % free_d)
                if env_probs and self.a.plan:
                    # PLAN shows the whole step list AND the blocker; REAL stops here.
                    self.summary["plan_would_void_on"] = env_probs
                    for p in env_probs:
                        self.log("  PLAN: a real run would VOID here -> %s" % p)
                else:
                    probs += env_probs
            except Exception as e:                                       # noqa: BLE001
                probs.append("4060 control agent: %s" % repr(e)[:120])
            # base policy nf4 (2026-09-08): the 5090 must ALSO produce the unfused nf4 base, which
            # its default interpreter cannot (transformers 5.8.1, fused experts: 0 of 3072
            # projections quantizable). Probe --py5090 the way --py4060 is probed. A failure here
            # is the loud stop the owner asked for, never a silently-bf16 run under an nf4 label.
            self.summary["base_policy"] = self.a.base_policy
            self.summary["base_dtypes"] = BASE_POLICIES[self.a.base_policy]
            if self.a.base_policy == "nf4":
                py = self.a.py5090
                env = (local_env_probe(py, self.run, self.log) if py
                       else {"ok": False, "error": "--py5090 not given (a transformers-4.x env is required)"})
                self.summary["local_env_5090"] = env
                lp = nf4_env_problems(env, py or "(none)", "5090")
                if lp:
                    lp.append("owner action for the 5090 (no GPU, ~10 min, ~3 GB download): "
                              + " ; ".join(TF4_ENV_RECIPE))
                    if self.a.plan:
                        self.summary.setdefault("plan_would_void_on", [])
                        for p in lp:
                            self.summary["plan_would_void_on"].append(p)
                            self.log("  PLAN: a real run would VOID here -> %s" % p)
                    else:
                        probs += lp
                else:
                    self.log("  5090 nf4 env OK under %s: %s" % (py, json.dumps(env)[:300]))
            # publish this script by sha for the 4060's bootstrap
            me = os.path.abspath(__file__)
            self.script_bytes = open(me, "rb").read()
            self.script_sha = sha256_bytes(self.script_bytes)
            if not self.a.plan and not probs:
                self.anchor.put(self.script_bytes, "wangrad/%s/script" % self.run_id)
                self.log("  published this runner by sha %s" % self.script_sha[:12])
            self.summary["twin_battery"] = self.twin_batt
            self.summary["corpus_sha256"] = getattr(self, "corpus_sha", None)
            self.summary["script_sha256"] = self.script_sha
            return probs

        # ------------------------------------------------------------ P1 A/A
        def aa_phase(self):
            """Design section 5: 'A/A (0 discordant) run before any arm is read'. Fresh, into THIS
            run dir (PostChain.aa_guard would skip onto the 09-03 artefact and reject it as stale).
            The positive control C2 (Instruct +2.65 pp) is banked at
            E:/rsft_runs/r1_ladder/BATTERY_r1_c2_instruct0924.json (2026-09-03); a 0-discordant A/A
            proves the scorer unchanged since, so C2 is not re-run (--with-c2 forces it)."""
            out = self.run + "/aa/BATTERY_aa_truebase.json"
            os.makedirs(self.run + "/aa", exist_ok=True)
            ok = self.step("P1:AA", ["battery_eval.py", "--model-dir", scc.BASE,
                                      "--label", "wangrad_aa", "--out", out], out, 0.6)
            if self.a.plan:
                return True
            if not ok:
                self.phase("P1_AA", "STOP", why="A/A artefact missing or rejected")
                return False
            c = self.paired(scc.BASE_BATTERY, out, 0.75, min_n=9000)
            self.phase("P1_AA", "GREEN" if c["disc"] == 0 else "STOP", n=c["n"], disc=c["disc"],
                       delta=c["delta"])
            if c["disc"] != 0:
                self.log("VOID (design section 5): A/A discordance %d != 0 -- the instrument moved; "
                         "nothing trains" % c["disc"])
                return False
            if self.a.with_c2:
                c2 = self.run + "/aa/BATTERY_c2_instruct.json"
                self.step("P1:C2", ["battery_eval.py", "--model-dir", "E:/olmoe_bf16",
                                     "--expect-model", "instruct_0924", "--label", "wangrad_c2",
                                     "--out", c2], c2, 0.6)
            return True

        # ------------------------------------------------------------ P2 stage + P3/P4/P5 gates
        def stage_phase(self):
            if not self.wait_4060_idle():
                self.phase("P2_STAGE", "BLOCKED", why="4060 busy (sampler) -- not killed; owner stops it")
                return False
            argv = ["--role", "stage-4060", "--run-id", self.run_id,
                    "--out-dir", SPEC["remote_root"] + r"\stage",
                    "--model-dir", SPEC["model_dir_4060"], "--data", SPEC["corpus_4060"],
                    "--corpus-sha", self.corpus_sha]
            if not self.a.plan and not self.anchor.resolve("wangrad/%s/corpus" % self.run_id):
                self.anchor.put(open(SPEC["corpus_5090"], "rb").read(), "wangrad/%s/corpus" % self.run_id)
                self.log("  corpus published by sha %s (%d B)" % (self.corpus_sha[:12],
                                                                   os.path.getsize(SPEC["corpus_5090"])))
            pid = self.spawn_remote("stage", argv, offline=False)
            try:
                res = self.wait_remote_artefact("stage", pid, "wangrad/%s/stage/4060" % self.run_id,
                                                self.a.stage_timeout_h * 3600)
            except Exception as e:                                       # noqa: BLE001
                self.phase("P2_STAGE", "FAIL", why=repr(e)[:400])
                return False
            if self.a.plan:
                self.phase("P2_STAGE", "PLAN")
                return True
            self.phase("P2_STAGE", "OK" if res.get("ok") else "FAIL", result=res)
            return bool(res.get("ok"))

        def footprint_phase(self):
            """Design 7(a). VOID set: 'the 4060 base-load footprint check fails (experts stayed
            bf16, or WDDM spill)'. Also the cost re-derivation the 2026-08-14 lesson demands: the
            measured s/step x 625 must fit --max-4060-h-per-arm before any arm is bought."""
            # 2026-09-09: --box was MISSING here and defaults to "5090" (:2297), so the role ran
            # ON THE 4060 believing it was a 5090 and took that branch of set_vram_cap:
            # (free - 8 GiB)/total is NEGATIVE on an 8 GiB card, so max(0.10, ...) floored the cap
            # at 0.10 -> 818.75 MiB, --cap-gib 6.5 was never read (the 5090 branch ignores it), and
            # the gate OOM'd allocating 2.87 GiB with 6.93 GiB free. basece_phase and the trainer
            # both pass --box; only this one did not.
            argv = ["--role", "gate-footprint", "--box", "4060", "--run-id", self.run_id,
                    "--out-dir", SPEC["remote_root"] + r"\gate_footprint",
                    "--model-dir", SPEC["model_dir_4060"], "--data", SPEC["corpus_4060"],
                    "--cap-gib", str(self.a.cap_4060_gib), "--n-steps", "8",
                    "--clip-mode", self.a.clip_mode,
                    "--cliff-max", str(self.a.cliff_max),
                    "--base-policy", self.a.base_policy, "--base-dtype", "nf4"]
            pid = self.spawn_remote("footprint", argv)
            try:
                res = self.wait_remote_artefact("footprint", pid,
                                                "wangrad/%s/gate/footprint" % self.run_id, 3600)
            except Exception as e:                                       # noqa: BLE001
                self.phase("P3_FOOTPRINT", "VOID", why=repr(e)[:400])
                return False
            if self.a.plan:
                self.phase("P3_FOOTPRINT", "PLAN")
                return True
            self.phase("P3_FOOTPRINT", "PASS" if res.get("pass") else "VOID",
                       alloc_gib=res.get("alloc_after_load_gib"), s_step=res.get("step_seconds_median"),
                       cliff=res.get("cliff_ratio"), reasons=res.get("reasons"),
                       est_h_per_arm=res.get("est_hours_per_arm_625_steps"), versions=res.get("versions"))
            if not res.get("pass"):
                return False
            est = res.get("est_hours_per_arm_625_steps") or 0.0
            self.summary["cost_rederived"] = {
                "s_per_step_4060": res.get("step_seconds_median"),
                "h_per_arm_4060_compute": est,
                "manifest_bytes_h1_est": 2.3e6 * EXPECT_STEPS * 3,
                "wire_bytes_h1_est": 8.39e6 * EXPECT_STEPS,
                "bound_h_per_arm": self.a.max_4060_h_per_arm}
            if est > self.a.max_4060_h_per_arm:
                self.phase("P3_COST", "STOP", est_h=est, bound=self.a.max_4060_h_per_arm,
                           why="measured 4060 s/step makes one arm cost more than the bound; owner decides")
                return False
            return True

        def basece_phase(self):
            """Design 7(b): |CE_4060 - CE_5090| of the frozen base on the 128-row probe. Under
            policy `mixed` that is nf4(4060) vs bf16(5090): quantization AND the cross-arch axis
            together, which is what the design registered. Under policy `nf4` both boxes load the
            SAME NF4 base and the gap is the cross-arch axis alone (bnb dequant + matmul on Ada vs
            Blackwell): the gate keeps its "are the boxes training the same model?" job, while
            the quantization COST -- what 7(b) could only proxy on 128 rows of CE -- is measured
            by the nf4 twin on the real instruments (nf4_twin_chain.py, paired against the bf16
            twin). The design's bar '25% of D1-TB's measured battery effect' is in pp; CE is in
            nats. Operationalised on the same quantity the training moves: 25% of the control
            twin's on-task CE drop (losses[0]-losses[-1] of seed 20260829, the policy's own twin)
            -- printed with its derivation, and overridable by --basece-gap-max-nats."""
            tw = json.load(io.open(self.twin_dir(20260829) + "/TRAIN.json", encoding="utf-8"))
            drop = float(tw["losses"][0] - tw["losses"][-1])
            gap_max = self.a.basece_gap_max_nats if self.a.basece_gap_max_nats > 0 else 0.25 * drop
            dt = BASE_POLICIES[self.a.base_policy]
            self.log("  7(b) threshold: 0.25 x twin CE drop %.4f = %.4f nats%s | policy %s: 4060 on %s, "
                     "5090 on %s" % (drop, 0.25 * drop, "" if self.a.basece_gap_max_nats <= 0
                                     else " (overridden to %.4f)" % gap_max, self.a.base_policy,
                                     dt["4060"], dt["5090"]))
            argv = ["--role", "gate-basece", "--box", "4060", "--run-id", self.run_id,
                    "--out-dir", SPEC["remote_root"] + r"\gate_basece",
                    "--model-dir", SPEC["model_dir_4060"], "--data", SPEC["corpus_4060"],
                    "--cap-gib", str(self.a.cap_4060_gib), "--probe-n", "128",
                    "--base-policy", self.a.base_policy, "--base-dtype", dt["4060"]]
            pid = self.spawn_remote("basece", argv)
            loc = self.run + "/gate_basece"
            ok = self.step_with(self.py_for("5090"), "P4:basece_5090",
                                ["wan_gradient_chain.py", "--role", "gate-basece", "--box", "5090",
                                 "--run-id", self.run_id, "--out-dir", loc,
                                 "--model-dir", SPEC["model_dir_5090"], "--data", SPEC["corpus_5090"],
                                 "--probe-n", "128",
                                 "--base-policy", self.a.base_policy, "--base-dtype", dt["5090"]],
                                loc + "/BASECE_5090.json", 0.5)
            try:
                r4 = self.wait_remote_artefact("basece", pid, "wangrad/%s/gate/basece_4060" % self.run_id, 3600)
            except Exception as e:                                       # noqa: BLE001
                self.phase("P4_BASECE", "VOID", why=repr(e)[:400])
                return False
            if self.a.plan:
                self.phase("P4_BASECE", "PLAN", threshold_nats=gap_max)
                return True
            if not ok:
                self.phase("P4_BASECE", "VOID", why="5090 probe artefact missing")
                return False
            r5 = json.load(io.open(loc + "/BASECE_5090.json", encoding="utf-8"))
            if r4.get("dtype") != dt["4060"] or r5.get("dtype") != dt["5090"]:
                self.phase("P4_BASECE", "VOID", why="probe ran on the wrong base dtype: 4060 %s (want %s), "
                           "5090 %s (want %s)" % (r4.get("dtype"), dt["4060"], r5.get("dtype"), dt["5090"]))
                return False
            gap = abs(float(r4["ce_per_token"]) - float(r5["ce_per_token"]))
            self.phase("P4_BASECE", "PASS" if gap <= gap_max else "VOID",
                       ce_4060=r4["ce_per_token"], ce_5090=r5["ce_per_token"],
                       dtype_4060=r4.get("dtype"), dtype_5090=r5.get("dtype"), gap=gap,
                       threshold=gap_max, tokens_4060=r4["tokens"], tokens_5090=r5["tokens"],
                       axis=("quantization + cross-arch (design as registered)"
                             if self.a.base_policy == "mixed" else
                             "cross-arch only: shared nf4 base, quantization cost is the nf4 twin's read"))
            if r4.get("tokens") != r5.get("tokens"):
                self.log("  WARNING: probe token counts differ (%s vs %s) -- tokenizer drift"
                         % (r4.get("tokens"), r5.get("tokens")))
            return gap <= gap_max

        _CKPT_RE = re.compile(r"^step_ckpt_(5090|4060)_s(\d{4})\.pt$")

        def local_ckpt_steps(self, out):
            """{step: path} of the 5090's step checkpoints in `out` (newest two are kept)."""
            d = {}
            for f in (os.listdir(out) if os.path.isdir(out) else []):
                m = self._CKPT_RE.match(f)
                if m and m.group(1) == "5090":
                    d[int(m.group(2))] = os.path.join(out, f)
            return d

        def remote_ckpt_steps(self, rdir):
            """{step: path} of the 4060's step checkpoints in its arm dir, via a READ-ONLY
            `dir /b` over the control plane (also under --plan, so a plan shows the real resume
            decision). {} when the dir or the agent is absent."""
            try:
                rc, o, e = self.remote_sh("dir /b %s" % rdir, timeout=60)
            except Exception as e:                                       # noqa: BLE001
                self.log("  remote ckpt listing failed: %s" % repr(e)[:120])
                return {}
            d = {}
            for f in (o or "").split("\n"):
                m = self._CKPT_RE.match(f.strip())
                if m and m.group(1) == "4060":
                    d[int(m.group(2))] = rdir + "\\" + f.strip()
            return d

        def resume_plan(self, tag, run_id, out):
            """2026-09-10 (two boxes lost in nine hours, 7 h of work gone): if BOTH boxes hold a
            step checkpoint at the same step, resume both from it. Every launch of this pair
            counts as an attempt and gets a FRESH anchor namespace <run_id>_rN (the anchor never
            deletes, so the dead attempt's step-numbered slices must never resolve for the
            resumed one). Returns (ns_id, resume_5090, resume_4060, remote_out_dir); the resume
            paths are "" when the pair starts at step 0."""
            rdir = SPEC["remote_root"] + "\\" + run_id
            marker = out + "/ATTEMPTS.txt"
            n = (sum(1 for _ in io.open(marker, encoding="utf-8")) if os.path.exists(marker) else 0) + 1
            ns_id = run_id if n == 1 else "%s_r%d" % (run_id, n)
            l5, l4 = self.local_ckpt_steps(out), self.remote_ckpt_steps(rdir)
            common = sorted(set(l5) & set(l4))
            r5 = r4 = ""
            cert_at = "-"
            step_used = "-"
            if common and self.a.cert_resume:
                # AMENDMENT 4/5 (2026-09-11): a resume is only bankable if it is SHOWN bit-exact
                # (A1.2). The trainer keeps its newest two checkpoints, so both boxes normally hold
                # s(k) and s(k-interval). Step back ONE interval: hide s(k) on both boxes as
                # *.cert_ref (invisible to _CKPT_RE), resume from s(k-interval), and let
                # resume_cert_watcher.py compare the re-written s(k) with the hidden original,
                # tensor for tensor, on both boxes. Costs one interval (~16 min). If no such pair
                # exists, or the hide fails on either box, RESTART THE SEED FROM STEP 0 in the
                # fresh namespace: a restart carries nothing over and needs no certificate, while
                # an uncertified resume would be VOID anyway (the 09-11 drill measured why).
                k = common[-1]
                back = k - self.a.ckpt_every
                if back in common and self.hide_for_cert(tag, l5[k], l4[k]):
                    cert_at = k
                    k = back
                    r5, r4 = l5[k], l4[k]
                    step_used = k
                    self.phase("P6_%s_RESUME" % tag, "RESUME_CERT", attempt=n, step=k, cert_at=cert_at,
                               ns=ns_id, local=r5, remote=r4, local_steps=sorted(l5), remote_steps=sorted(l4))
                else:
                    self.log("  %s: no certifiable resume point (common %s, hide ok=%s) -- attempt %d "
                             "RESTARTS the seed from step 0 in namespace %s" % (tag, common, back in common, n, ns_id))
                    self.phase("P6_%s_RESUME" % tag, "RESTART_FROM_0", attempt=n, ns=ns_id, common=common)
                    # D4 (fable night review 17:22): retention keeps the two HIGHEST step numbers,
                    # so the dead attempt's s(k)/s(k-25) would outlive -- and prune -- the
                    # restarted run's own early checkpoints, and a later resume would pick the
                    # stale pair. Set every existing step checkpoint aside on both boxes.
                    self.stale_all_ckpts(tag, l5, l4, n)
            elif common:
                k = common[-1]
                r5, r4 = l5[k], l4[k]
                step_used = k
                self.phase("P6_%s_RESUME" % tag, "RESUME", attempt=n, step=k, ns=ns_id,
                           local=r5, remote=r4, local_steps=sorted(l5), remote_steps=sorted(l4))
            elif l5 or l4:
                self.log("  %s: no COMMON checkpoint (5090 %s | 4060 %s) -- attempt %d starts at step 0"
                         % (tag, sorted(l5), sorted(l4), n))
            if not self.a.plan:
                with io.open(marker, "a", encoding="utf-8") as fh:
                    fh.write("%s attempt %d ns %s resume_step %s 5090 %s 4060 %s cert_at %s\n"
                             % (now(), n, ns_id, step_used, r5 or "-", r4 or "-", cert_at))
                # D7 (fable night review 17:22): a reboot between the 5090's TRAIN manifest and the
                # 4060's leaves TRAIN_5090.json alone; Chain.step would then SKIP the 5090 trainer
                # on its artefact and the 4060 would wait for an init that never comes. Any attempt
                # after the first sets a lone 5090 manifest aside.
                t5, t4 = out + "/TRAIN_5090.json", out + "/TRAIN_4060.json"
                if n > 1 and os.path.exists(t5) and not os.path.exists(t4):
                    os.replace(t5, t5 + ".pre_r%d" % n)
                    self.log("  %s: set aside a lone TRAIN_5090.json from the dead attempt" % tag)
            return ns_id, r5, r4, rdir

        def stale_all_ckpts(self, tag, l5, l4, n):
            """Rename every step checkpoint on both boxes to *.stale_r<n> (restart-from-0 only);
            the suffix is invisible to _CKPT_RE and to the trainer's own retention regex."""
            if self.a.plan:
                return
            for p in l5.values():
                try:
                    os.replace(p, p + ".stale_r%d" % n)
                except Exception as ex:                                      # noqa: BLE001
                    self.log("  %s: could not set aside %s: %s" % (tag, p, repr(ex)[:100]))
            for p in l4.values():
                try:
                    base = p.rsplit("\\", 1)[-1]
                    self.remote_sh("cmd /c ren %s %s.stale_r%d" % (p, base, n), timeout=60)
                except Exception as ex:                                      # noqa: BLE001
                    self.log("  %s: could not set aside remote %s: %s" % (tag, p, repr(ex)[:100]))
            self.log("  %s: set aside %d local + %d remote stale checkpoints" % (tag, len(l5), len(l4)))

        def hide_for_cert(self, tag, local_path, remote_path):
            """Rename s(k) -> s(k).cert_ref on BOTH boxes (never overwriting an existing .cert_ref:
            the first hidden original stays the reference for every later attempt). True only if
            both renames verifiably happened; on any failure the caller restarts from step 0."""
            if self.a.plan:
                return True
            try:
                dst = local_path + ".cert_ref"
                if os.path.exists(dst):
                    os.replace(local_path, local_path + ".cert_ref_r%d" % int(time.time()))
                else:
                    os.replace(local_path, dst)
                if os.path.exists(local_path) or not os.path.exists(dst):
                    self.log("  %s: local hide did not take effect" % tag)
                    return False
                base = remote_path.rsplit("\\", 1)[-1]
                rc, o, e = self.remote_sh("cmd /c if exist %s.cert_ref (ren %s %s.cert_ref_r%d) else (ren %s %s.cert_ref)"
                                          % (remote_path, remote_path, base, int(time.time()), remote_path, base),
                                          timeout=60)
                rc2, o2, _ = self.remote_sh("cmd /c dir /b %s.cert_ref* & if exist %s echo STILL_THERE"
                                            % (remote_path, remote_path), timeout=60)
                if "STILL_THERE" in (o2 or "") or ".cert_ref" not in (o2 or ""):
                    self.log("  %s: remote hide did not take effect (%s | %s)" % (tag, (o or e)[:120], (o2 or "")[:120]))
                    return False
                self.log("  %s: hid s(k) on both boxes as .cert_ref for the resume certificate" % tag)
                return True
            except Exception as ex:                                          # noqa: BLE001
                self.log("  %s: hide failed: %s" % (tag, repr(ex)[:160]))
                return False

        def run_pair(self, tag, kind, seed, steps_limit=0, budget_h=3.0):
            """One distributed run: spawn the 4060 trainer, run the 5090 trainer through
            Chain.step (card-idle, budget, artefact), then collect both TRAIN.json files."""
            run_id = "%s_%s" % (self.run_id, tag)
            out = self.run + "/" + tag
            os.makedirs(out, exist_ok=True)
            if os.path.exists(out + "/TRAIN_5090.json") and os.path.exists(out + "/TRAIN_4060.json"):
                self.log("  %s: both TRAIN manifests present, skipping" % tag)
                return (json.load(io.open(out + "/TRAIN_5090.json", encoding="utf-8")),
                        json.load(io.open(out + "/TRAIN_4060.json", encoding="utf-8")))
            if not self.wait_4060_idle():
                raise RuntimeError("4060 busy")
            # The 5090 must be idle BEFORE the 4060 is launched: the remote trainer waits for the
            # 5090's init object with --peer-timeout-s, and a 5090 parked behind another job (the
            # OOD scorer holds it for hours today) would let that wait expire.
            if not self.a.plan and not self.card_idle(3):
                raise RuntimeError("5090 busy -- not launching the 4060 into a timeout")
            ns_id, r5, r4, rdir = self.resume_plan(tag, run_id, out)
            pid = self.spawn_remote(tag, self.remote_trainer_argv(kind, seed, ns_id, steps_limit,
                                                                  out_dir=rdir, resume_from=r4))
            ok = self.step_with(self.py_for("5090"), "%s:train5090" % tag,
                                self.local_trainer_argv(tag, kind, seed, ns_id, out, steps_limit,
                                                        resume_from=r5),
                                out + "/TRAIN_5090.json", budget_h)
            if self.a.plan:
                return None, None
            t4 = self.wait_remote_artefact(tag, pid, "wangrad/%s/4060/train_json" % ns_id, 3600)
            with io.open(out + "/TRAIN_4060.json", "w", encoding="utf-8") as fh:
                json.dump(t4, fh, indent=1)
            if not ok:
                raise RuntimeError("5090 trainer produced no manifest for %s" % tag)
            t5 = json.load(io.open(out + "/TRAIN_5090.json", encoding="utf-8"))
            for box, t in (("5090", t5), ("4060", t4)):
                ss = t.get("seq_stats") or {}
                if ss.get("truncated") or ss.get("dropped"):
                    raise RuntimeError("VOID zero-truncation rule: %s seq_stats %s" % (box, json.dumps(ss)))
            # HIGH-1 (review 2026-09-08): a dead 4060 surface is SILENT. If its owned LoRA receives
            # no gradient, clip_grad_norm_ returns 0, AdamW no-ops, every published slice is the
            # zero-init B and the 5090 concatenates zeros -- the arm then scores as a 12-layer
            # single-box LoRA while the manifest claims a gradient-derived slice crossed the WAN.
            # A zero here is not a weak result, it is a VOID: the treatment never existed.
            for box, t in (("5090", t5), ("4060", t4)):
                gmax = t.get("gnorm_max")
                if gmax is None:
                    raise RuntimeError("VOID: %s recorded no gnorm_max -- cannot show its surface "
                                       "carried gradient" % box)
                if not (float(gmax) > 0.0):
                    raise RuntimeError("VOID gradient-liveness: %s gnorm_max=%s -- its owned slice "
                                       "never moved, so no gradient crossed and the arm is not the "
                                       "treatment it claims to be" % (box, gmax))
                if t.get("budget_stopped"):
                    raise RuntimeError("%s trainer was BUDGET-STOPPED (not complete)" % box)
                if t.get("trainable_total") != SPEC["trainable_total"]:
                    raise RuntimeError("%s trainable_total %s" % (box, t.get("trainable_total")))
            return t5, t4

        def tracking_phase(self):
            """Design 7(c): ~30 steps of W-DILOCO-H1 vs C-SOLO on identical batches; the loss
            curves must track within a small band. C-SOLO's curve is the banked twin's losses[:30]
            (seed 20260829). Two checks: step-1 losses must agree to --track-step1-tol (the
            adapter is zero on both sides at step 1, so this is a pure numerics/tokenization
            control), and max |diff| over the 30 steps <= --track-band-max nats."""
            try:
                t5, t4 = self.run_pair("track", "h1", SPEC["seeds"]["s1"], steps_limit=self.a.track_steps,
                                       budget_h=1.0)
            except Exception as e:                                       # noqa: BLE001
                self.phase("P5_TRACK", "STOP", why=repr(e)[:400])
                return False
            if self.a.plan:
                self.phase("P5_TRACK", "PLAN")
                return True
            tw = json.load(io.open(self.twin_dir(SPEC["seeds"]["s1"]) + "/TRAIN.json", encoding="utf-8"))
            ref = tw["losses"][:self.a.track_steps]
            got = t5["losses"][:self.a.track_steps]
            n = min(len(ref), len(got))
            diffs = [abs(ref[i] - got[i]) for i in range(n)]
            step1 = diffs[0] if diffs else float("inf")
            band = max(diffs) if diffs else float("inf")
            ok = n >= self.a.track_steps and step1 <= self.a.track_step1_tol and band <= self.a.track_band_max
            self.phase("P5_TRACK", "PASS" if ok else "STOP", n=n, step1_diff=step1,
                       max_abs_diff=band, mean_abs_diff=(sum(diffs) / n if n else None),
                       band_max=self.a.track_band_max, step1_tol=self.a.track_step1_tol,
                       s_step_5090=t5.get("step_seconds_median"), s_step_4060=t4.get("step_seconds_median"),
                       syncs=t5.get("n_syncs"), anchor=t5.get("anchor"),
                       clip={b: {k: (t.get("clip") or {}).get(k) for k in
                                 ("mode", "global_bound_steps", "global_bound_local_not_steps",
                                  "max_global", "max_local", "coef_min")}
                             for b, t in (("5090", t5), ("4060", t4))})
            if not ok:
                self.log("STOP (design 7c): H=1 does not track the single-box curve -- the "
                         "block-coordinate mechanism is broken at the root; no full arm is bought")
            return ok

        # ------------------------------------------------------------ arms + scoring + verdicts
        def score_arm(self, tag, seed, adapter):
            out = self.run + "/" + tag
            P = dict(battery=out + "/BATTERY_%s.json" % tag, arc=out + "/ARC_%s.json" % tag,
                     gsm=out + "/S1_%s.json" % tag, gsmext=out + "/S1EXT_%s.json" % tag)
            gsm = ["--max-new", "320", "--data", "E:/aiCrypto/gsm8k_test.jsonl",
                   "--prompt-format", "fewshot", "--seed", "20260824"]
            for label, script, art, budget, extra in (
                    ("battery-9547", "battery_eval.py", P["battery"], 0.6, []),
                    ("GSM8K-512", "s1_frozen_baseline.py", P["gsm"], 0.6, ["--n", "512", "--offset", "0"] + gsm),
                    ("GSM8K-807ext", "s1_frozen_baseline.py", P["gsmext"], 0.8, ["--n", "807", "--offset", "512"] + gsm),
                    ("ARC-Easy", "arc_falsifier.py", P["arc"], 0.3, [])):
                # MEDIUM-2: self.step's return is the artefact's ACCEPTANCE, not its existence.
                # Chain.step skips on an existing file and PostChain.step then rejects it; dropping
                # this value is how a stale score from a previous attempt gets paired with a fresh
                # adapter.
                ok = self.step(tag + ":" + label,
                               [script, "--model-dir", scc.BASE, "--adapter", adapter]
                               + extra + ["--label", tag + "_" + label, "--out", art], art, budget)
                P["ok_" + {"battery-9547": "battery", "GSM8K-512": "gsm",
                           "GSM8K-807ext": "gsmext", "ARC-Easy": "arc"}[label]] = bool(ok)
            return P

        def pooled(self, guard, ext):
            a, _ = paired_read.per_item(json.load(io.open(guard, encoding="utf-8")))
            b, _ = paired_read.per_item(json.load(io.open(ext, encoding="utf-8")))
            m = dict(a)
            m.update(b)
            return m

        def verdict(self, tag, kind, seed, P, t5, t4):
            """Design section 5 primary = NON-INFERIORITY vs the control twin, paired on identical
            items: W minus C >= -MDE on the battery AND on GSM8K-1319 -> PARITY (registered
            success). W - C < -MDE with p < 0.05 -> INFERIOR. Otherwise UNRESOLVED. ARC-Easy
            regression at p < 0.05 vetoes; ex_boolq reported (memory the-objective clause 4)."""
            twd = self.twin_dir(seed)               # the policy's control (bf16 twin / nf4 twin)
            ttag = os.path.basename(twd)
            v = {"arm": tag, "kind": kind, "H": H_OF[kind], "seed": seed,
                 "registered_by": SPEC["design"], "twin": twd,
                 "base_policy": self.a.base_policy, "base_dtypes": BASE_POLICIES[self.a.base_policy],
                 "trained_base_dtype": {"5090": (t5 or {}).get("base_dtype"), "4060": (t4 or {}).get("base_dtype")},
                 "wire": {"5090": (t5 or {}).get("anchor"), "4060": (t4 or {}).get("anchor")},
                 "syncs": (t5 or {}).get("n_syncs"), "clip_bound_steps": {
                     "5090": (t5 or {}).get("clip_bound_steps"), "4060": (t4 or {}).get("clip_bound_steps")}}
            reads = {}
            if P.get("ok_battery") and os.path.exists(P["battery"]):
                tb, _ = paired_read.per_item(json.load(io.open(twd + "/BATTERY_%s.json" % ttag, encoding="utf-8")))
                wb, _ = paired_read.per_item(json.load(io.open(P["battery"], encoding="utf-8")))
                m = paired_read.mcnemar(tb, wb)
                m.pop("ids", None)
                reads["battery_W_minus_C"] = m
                c = paired_read.compare(scc.BASE_BATTERY, P["battery"], 0.75)
                c.pop("gained_ids", None)
                v["battery_vs_base"] = c
                v["ex_boolq_vs_base"] = c.get("ex_boolq")
            if (P.get("ok_gsm") and P.get("ok_gsmext")
                    and os.path.exists(P["gsm"]) and os.path.exists(P["gsmext"])):
                mine = self.pooled(P["gsm"], P["gsmext"])
                twin = self.pooled(twd + "/S1_%s.json" % ttag, twin_ext_path(seed, twd))
                ids = set(twin) & set(mine)
                m = paired_read.mcnemar({k: twin[k] for k in ids}, {k: mine[k] for k in ids})
                m.pop("ids", None)
                reads["gsm1319_W_minus_C"] = m
                base = self.pooled(SPEC["base_gsm512"], SPEC["base_gsmext"])
                ids = set(base) & set(mine)
                b = paired_read.mcnemar({k: base[k] for k in ids}, {k: mine[k] for k in ids})
                b.pop("ids", None)
                v["gsm1319_vs_base"] = b
            vetoes = []
            if P.get("ok_arc") and os.path.exists(P["arc"]):
                c = paired_read.compare(scc.BASE_ARC, P["arc"], 1.5)
                c.pop("gained_ids", None)
                v["arc_vs_base"] = c
                if c["delta"] < 0 and c["p"] < 0.05:
                    vetoes.append("arc_easy %+.4f pp p=%.3g" % (c["delta"], c["p"]))
            v["reads"] = reads
            verdicts = {}
            for k, m in reads.items():
                if m["delta"] >= -m["mde"]:
                    verdicts[k] = "PARITY"
                elif m["p"] < 0.05:
                    verdicts[k] = "INFERIOR"
                else:
                    verdicts[k] = "UNRESOLVED"
            v["per_instrument"] = verdicts
            # MEDIUM-2: a missing ARC leaves the VETO silently absent, and a missing battery makes
            # the section-5 KILL unable to fire -- so "some reads succeeded" is not enough. All four
            # artefacts must have been accepted, or the arm is INCOMPLETE and licenses nothing.
            missing = [k for k in ("battery", "gsm", "gsmext", "arc") if not P.get("ok_" + k)]
            v["artefacts_rejected"] = missing
            if missing:
                v["verdict"] = "INCOMPLETE"
            elif len(reads) < 2:
                v["verdict"] = "INCOMPLETE"
            elif all(x == "PARITY" for x in verdicts.values()):
                v["verdict"] = "PARITY"
            elif any(x == "INFERIOR" for x in verdicts.values()):
                v["verdict"] = "INFERIOR"
            else:
                v["verdict"] = "UNRESOLVED"
            if vetoes:
                v["named_vetoes"] = vetoes
                v["verdict_before_veto"] = v["verdict"]
                v["verdict"] = "VETOED"
            for k, m in reads.items():
                self.log("  %s %s: W-C %+.4f pp CI[%+.3f,%+.3f] p=%.3g D=%d MDE %.4f -> %s"
                         % (tag, k, m["delta"], m["ci_lo"], m["ci_hi"], m["p"], m["disc"], m["mde"], verdicts[k]))
            self.log("VERDICT %s: %s%s" % (tag, v["verdict"], (" (vetoes: %s)" % vetoes) if vetoes else ""))
            with io.open(self.run + "/%s/VERDICT_%s.json" % (tag, tag), "w", encoding="utf-8") as fh:
                json.dump(v, fh, indent=1, default=str)
            self.summary["arms"][tag] = {"verdict": v["verdict"], "per_instrument": verdicts,
                                         "reads": reads, "vetoes": vetoes}
            self.save()
            return v

        def arm_phase(self, tag, kind, stag):
            seed = SPEC["seeds"][stag]
            out = self.run + "/" + tag
            if os.path.exists(out + "/VERDICT_%s.json" % tag):
                v = json.load(io.open(out + "/VERDICT_%s.json" % tag, encoding="utf-8"))
                self.log("SKIP %s (verdict exists: %s)" % (tag, v.get("verdict")))
                self.summary["arms"][tag] = {"verdict": v.get("verdict"), "reads": v.get("reads")}
                return v
            self.log("ARM %s | %s | seed %d | H %s" % (tag, kind, seed, H_OF[kind] or "inf"))
            try:
                t5, t4 = self.run_pair(tag, kind, seed, budget_h=self.a.train_max_h_4060)
            except Exception as e:                                       # noqa: BLE001
                self.phase("P6_" + tag, "STOP", why=repr(e)[:400])
                return None
            adapter = out + "/adapter"
            if not self.a.plan and not os.path.exists(adapter + "/adapter_config.json"):
                self.phase("P6_" + tag, "STOP", why="no concatenated adapter")
                return None
            P = self.score_arm(tag, seed, adapter)
            if self.a.plan:
                return {"verdict": "PLAN"}
            return self.verdict(tag, kind, seed, P, t5, t4)

        # ------------------------------------------------------------ DP-2 (2026-09-12)
        # Everything under `--dp2`: the prereg's arms, the fidelity pair, the VOID/KILL gates as
        # code, the twin4060 verdict, and `--score-only`. Nothing here runs unless --dp2 (or
        # --score-only) is given; go() branches at its first line.
        def dp2_stag(self, tag):
            return tag.rsplit("_", 1)[-1]

        def local_dp2_argv(self, tag, seed, run_id, out_dir, steps_limit=0, resume_from="", fidelity_json=None):
            fj = DP2["fidelity_json"] if fidelity_json is None else fidelity_json
            return [os.path.basename(os.path.abspath(__file__)), "--role", "dp2-trainer", "--box", "5090",
                    "--arm", "dp2", "--seed", str(seed), "--run-id", run_id, "--out-dir", out_dir,
                    "--model-dir", SPEC["model_dir_5090"], "--data", SPEC["corpus_5090"],
                    "--expect-data-sha", self.corpus_sha, "--steps-limit", str(steps_limit),
                    "--peer-timeout-s", str(self.a.peer_timeout_s), "--poll-s", str(self.a.poll_s_h1),
                    "--max-hours", str(self.a.train_max_h), "--base-policy", "nf4", "--base-dtype", "nf4",
                    "--ckpt-every", str(self.a.ckpt_every), "--epochs", str(DP2["epochs"]),
                    "--wire-dtype", self.a.wire_dtype, "--fidelity-json", fj] \
                + (["--anchor-gc"] if not self.a.no_anchor_gc else []) \
                + self._det_argv("5090") + self._resume_argv(resume_from)

        def remote_dp2_argv(self, seed, run_id, steps_limit=0, out_dir="", resume_from="", fidelity_json=None):
            fj = DP2["fidelity_json"] if fidelity_json is None else fidelity_json
            return ["--role", "dp2-trainer", "--box", "4060", "--arm", "dp2", "--seed", str(seed),
                    "--run-id", run_id, "--out-dir", out_dir or (SPEC["remote_root"] + "\\" + run_id),
                    "--model-dir", SPEC["model_dir_4060"], "--data", SPEC["corpus_4060"],
                    "--expect-data-sha", self.corpus_sha, "--steps-limit", str(steps_limit),
                    "--cap-gib", str(self.a.cap_4060_gib),
                    "--peer-timeout-s", str(self.a.peer_timeout_s), "--poll-s", str(self.a.poll_s_h1),
                    "--max-hours", str(self.a.train_max_h_4060), "--base-policy", "nf4", "--base-dtype", "nf4",
                    "--ckpt-every", str(self.a.ckpt_every), "--epochs", str(DP2["epochs"]),
                    "--wire-dtype", self.a.wire_dtype, "--fidelity-json", fj] \
                + self._det_argv("4060") + self._resume_argv(resume_from)

        def dp2_preflight_extra(self, want_arms):
            """DP-2 additions to preflight(): the prereg's execution preconditions as refusals."""
            probs = []
            if self.a.base_policy != "nf4":
                probs.append("DP-2 is registered on the nf4 base on both boxes: pass --base-policy nf4")
            det = set(b.strip() for b in (self.a.deterministic_boxes or "").split(",") if b.strip())
            if det != {"5090", "4060"}:
                probs.append("DP-2 runs deterministic kernels on BOTH boxes (prereg section 1): "
                             "--deterministic-boxes 5090,4060 (got %r)" % (self.a.deterministic_boxes or ""))
            if want_arms and not self.a.cert_resume:
                probs.append("VOID 5 (uncertified resume) is only enforceable under --cert-resume; pass it")
            if want_arms and (self.a.train_max_h < 9.0 or self.a.train_max_h_4060 < 9.0):
                probs.append("a DP-2 arm is estimated at 7.85 h (prereg section 1): pass --train-max-h 11 "
                             "--train-max-h-4060 11 (got %.1f / %.1f)" % (self.a.train_max_h, self.a.train_max_h_4060))
            free = anchor_free_gb()
            self.summary["anchor_free_gb"] = free
            if free is None:
                probs.append("anchor free space unreadable over ssh (df) -- preflight needs >= %.0f GB"
                             % DP2["anchor_min_free_gb"])
            elif free < DP2["anchor_min_free_gb"]:
                probs.append("anchor has %.2f GB free < %.0f GB required (prereg section 3)"
                             % (free, DP2["anchor_min_free_gb"]))
            else:
                self.log("  anchor free %.2f GB (>= %.0f GB OK)" % (free, DP2["anchor_min_free_gb"]))
            if want_arms:
                # F2: passed AND bound to THIS chain file (FIDELITY.chain_sha256 == self.script_sha,
                # computed by preflight() from the bytes on disk) AND 0 gate repairs (F4)
                passed, sha, info = dp2_fidelity_bound(DP2["fidelity_json"], self.script_sha)
                self.summary["fidelity"] = {"path": DP2["fidelity_json"], "passed": passed, "sha256": sha, "info": info}
                if passed is not True:
                    probs.append("VOID 1: fidelity gate artefact %s passed=%r %s -- run dp2_fidelity_gate.py first"
                                 % (DP2["fidelity_json"], passed, json.dumps(info)[:260]))
                else:
                    self.log("  fidelity gate PASSED and bound to chain sha %s (%s, artefact sha %s)"
                             % (self.script_sha[:12], DP2["fidelity_json"], (sha or "")[:12]))
                for tag, _k, stag in DP2_ARMS:
                    d = DP2["twin4060"][stag]
                    if not os.path.exists(d + "/S1_%s.json" % os.path.basename(d)):
                        self.log("  NOTE: primary control %s not scored yet (needed at verdict time, not for "
                                 "training): run --score-only %s" % (d, os.path.basename(d)))
            return probs

        def run_pair_dp2(self, tag, seed, steps_limit=0, budget_h=11.0, fidelity_json=None, fresh=False):
            """One DP-2 pair: spawn the 4060 dp2-trainer, run the 5090's through Chain.step, collect
            both manifests, refuse anything the prereg VOIDs at the manifest level."""
            # the fidelity pair is its own run (tag == run_id): namespace wangrad/dp2fid_<ts>, as registered
            run_id = self.run_id if tag == self.run_id else "%s_%s" % (self.run_id, tag)
            out = self.run + "/" + tag
            os.makedirs(out, exist_ok=True)
            if os.path.exists(out + "/TRAIN_5090.json") and os.path.exists(out + "/TRAIN_4060.json"):
                self.log("  %s: both TRAIN manifests present, skipping" % tag)
                return (json.load(io.open(out + "/TRAIN_5090.json", encoding="utf-8")),
                        json.load(io.open(out + "/TRAIN_4060.json", encoding="utf-8")))
            if not self.wait_4060_idle():
                raise RuntimeError("4060 busy")
            if not self.a.plan and not self.card_idle(3):
                raise RuntimeError("5090 busy -- not launching the 4060 into a timeout")
            ns_id, r5, r4, rdir = self.resume_plan(tag, run_id, out)
            if fresh and (r5 or r4):
                raise RuntimeError("%s must start from step 0 (a resumed lineage is not a fidelity proof); "
                                   "use a fresh run dir" % tag)
            pid = self.spawn_remote(tag, self.remote_dp2_argv(seed, ns_id, steps_limit, out_dir=rdir,
                                                              resume_from=r4, fidelity_json=fidelity_json))
            ok = self.step_with(self.py_for("5090"), "%s:train5090" % tag,
                                self.local_dp2_argv(tag, seed, ns_id, out, steps_limit, resume_from=r5,
                                                    fidelity_json=fidelity_json),
                                out + "/TRAIN_5090.json", budget_h)
            if self.a.plan:
                return None, None
            t4 = self.wait_remote_artefact(tag, pid, "wangrad/%s/4060/train_json" % ns_id, 3600)
            with io.open(out + "/TRAIN_4060.json", "w", encoding="utf-8") as fh:
                json.dump(t4, fh, indent=1)
            if not ok:
                raise RuntimeError("5090 dp2-trainer produced no manifest for %s" % tag)
            t5 = json.load(io.open(out + "/TRAIN_5090.json", encoding="utf-8"))
            for box, t in (("5090", t5), ("4060", t4)):
                if t.get("role") != "dp2-trainer":
                    raise RuntimeError("%s manifest is not a dp2-trainer manifest (role %r)" % (box, t.get("role")))
                if t.get("void"):
                    raise RuntimeError("VOID declared by the %s trainer: %s" % (box, str(t.get("void"))[:300]))
                ss = t.get("seq_stats") or {}
                if ss.get("truncated") or ss.get("dropped"):
                    raise RuntimeError("VOID zero-truncation rule: %s seq_stats %s" % (box, json.dumps(ss)))
                if t.get("budget_stopped"):
                    raise RuntimeError("%s trainer was BUDGET-STOPPED (not complete)" % box)
                if t.get("trainable_total") != SPEC["trainable_total"]:
                    raise RuntimeError("%s trainable_total %s" % (box, t.get("trainable_total")))
            return t5, t4

        def dp2_post_run_void(self, tag, t5, t4):
            """Prereg section 2 VOID clauses evaluated over the finished pair's manifests (the trainers
            evaluate the same predicates every 25 steps; this is the artefact-level re-check the chain
            scores before it spends 1.83 h scoring). Writes VOID_<tag>.json when any fires.
            The manifest predicates live in dp2_manifest_void_problems (pure, dp2_selftest (iv))."""
            probs = dp2_manifest_void_problems(t5, t4)
            # VOID 5: every resume in the lineage must carry a bit-exact certificate (AMENDMENT 5)
            att = self.run + "/%s/ATTEMPTS.txt" % tag
            if os.path.exists(att):
                import resume_cert_watcher as rcw            # noqa: E402  (pure parts only)
                text = io.open(att, encoding="utf-8").read()
                certs = {}
                cp = self.run + "/%s/RESUME_CERTS.json" % tag
                if os.path.exists(cp):
                    try:
                        certs = json.load(io.open(cp, encoding="utf-8")).get("certs") or {}
                    except Exception as e:                                  # noqa: BLE001
                        probs.append("VOID 5: RESUME_CERTS.json unreadable: %s" % repr(e)[:80])
                ok, why = rcw.lineage_ok(rcw.parse_attempts(text), certs)
                if not ok:
                    probs.append("VOID 5: resume lineage not certified bit-exact -- %s" % why)
            probs = sorted(set(probs))
            if probs:
                v = {"arm": tag, "problems": probs, "at": now(),
                     "registered_by": "docs/research/PREREG_DP2_BEATS_ONE_4060_2026-09-12.md section 2"}
                with io.open(self.run + "/%s/VOID_%s.json" % (tag, tag), "w", encoding="utf-8") as fh:
                    json.dump(v, fh, indent=1)
                self.phase("P6_" + tag, "VOID", problems=probs)
            return probs

        def dp2_verdict(self, tag, stag, P, t5, t4, primary_dir=None):
            """Prereg section 2, one seed (dp2_build_verdict does the arithmetic; this writes the
            artefacts). The primary control must be scored already, or NO verdict is written:
            VERDICT_PENDING_<tag>.json records the state and the chain STOPs."""
            pdir = primary_dir or DP2["twin4060"][stag]
            v = dp2_build_verdict(P, stag, SPEC["seeds"][stag], pdir, t5, t4, paired_read,
                                  scc.BASE_BATTERY, scc.BASE_ARC, self.log, tag=tag)
            v["chain_sha256"] = self.script_sha                   # F2: the runner that scored/verdicted
            if v.get("verdict") == "PENDING_CONTROL":
                self.log("VERDICT %s: PENDING -- %s" % (tag, v["pending_why"]))
                with io.open(self.run + "/%s/VERDICT_PENDING_%s.json" % (tag, tag), "w", encoding="utf-8") as fh:
                    json.dump(v, fh, indent=1, default=str)
                return None
            vetoes = v.get("named_vetoes") or []
            self.log("VERDICT %s: %s%s%s" % (tag, v["verdict"], (" (vetoes: %s)" % vetoes) if vetoes else "",
                                              (" | KILL for s2: %s" % v["kill_s2"]) if v.get("kill_s2") else ""))
            with io.open(self.run + "/%s/VERDICT_%s.json" % (tag, tag), "w", encoding="utf-8") as fh:
                json.dump(v, fh, indent=1, default=str)
            self.summary["arms"][tag] = {"verdict": v["verdict"], "reads": v.get("reads"), "vetoes": vetoes,
                                         "kill_s2": v.get("kill_s2")}
            self.save()
            return v

        def dp2_arm_phase(self, tag, stag):
            seed = SPEC["seeds"][stag]
            out = self.run + "/" + tag
            if os.path.exists(out + "/VERDICT_%s.json" % tag):
                v = json.load(io.open(out + "/VERDICT_%s.json" % tag, encoding="utf-8"))
                why = dp2_verdict_problem(v, stag)          # F1/F5: never skip onto an unbound verdict
                if why:
                    self.phase("P6_" + tag, "STOP", why="VERDICT_%s.json is not skipped onto: %s -- re-score it "
                               "with --score-only %s (which now enforces the VOID clauses)" % (tag, why, tag))
                    return None
                self.log("SKIP %s (verdict exists: %s)" % (tag, v.get("verdict")))
                self.summary["arms"][tag] = {"verdict": v.get("verdict"), "reads": v.get("reads"), "kill_s2": v.get("kill_s2")}
                return v
            if os.path.exists(out + "/VOID_%s.json" % tag):
                self.phase("P6_" + tag, "VOID", why="VOID_%s.json exists -- not scored" % tag)
                return None
            self.log("ARM %s | dp2 | seed %d | 625 steps x 32 rows (2 epochs) | det | cert-resume" % (tag, seed))
            try:
                t5, t4 = self.run_pair_dp2(tag, seed, budget_h=self.a.train_max_h_4060)
            except Exception as e:                                       # noqa: BLE001
                self.phase("P6_" + tag, "STOP", why=repr(e)[:400])
                return None
            if self.a.plan:
                self.log("STEP %s:score (plan) | battery-9547, GSM8K-512, GSM8K-807ext, ARC-Easy on %s/adapter" % (tag, out))
                return {"verdict": "PLAN"}
            if self.dp2_post_run_void(tag, t5, t4):
                return None
            adapter = out + "/adapter"
            if not os.path.exists(adapter + "/adapter_config.json"):
                self.phase("P6_" + tag, "STOP", why="no adapter")
                return None
            P = self.score_arm(tag, seed, adapter)
            return self.dp2_verdict(tag, stag, P, t5, t4)

        def fidpair_phase(self):
            """The DP-2 half of the fidelity gate (clause 1): 30 steps of DP-2 on both boxes, seed s1,
            det, the artefact requirement itself EXEMPT (--fidelity-json ""). dp2_fidelity_gate.py
            runs the d1_sft.py reference and the comparison."""
            tag = self.a.fid_tag or self.run_id
            try:
                t5, t4 = self.run_pair_dp2(tag, SPEC["seeds"]["s1"], steps_limit=self.a.fid_steps, budget_h=1.5,
                                           fidelity_json="none", fresh=True)
            except Exception as e:                                       # noqa: BLE001
                self.phase("P_FIDPAIR", "STOP", why=repr(e)[:400])
                return False
            if self.a.plan:
                self.phase("P_FIDPAIR", "PLAN")
                return True
            self.phase("P_FIDPAIR", "DONE", steps=t5.get("steps"), final_identical=t5.get("final_identical_vs_peer"),
                       crit_sync_mean=t5.get("critical_sync_mean"), repair_events=t4.get("repair_events"),
                       adapter=t5.get("adapter"))
            return True

        def score_only(self):
            """--score-only <arm> --adapter <dir> [--control-dir <dir>]: the SAME scorer chain the h1
            arms used, into E:/rsft_runs/wangrad/<arm>/, then the DP-2 verdict against the primary
            control (default twin4060_<seed> for dp2_* arms; none for a control being scored)."""
            tag = self.a.score_only
            stag = self.dp2_stag(tag)
            if stag not in SPEC["seeds"]:
                self.terminal("STOP", "--score-only %s: cannot read the seed tag (want *_s1 / *_s2)" % tag)
                return 2
            seed = SPEC["seeds"][stag]
            adapter = self.a.adapter or (self.run + "/%s/adapter" % tag)
            if not os.path.exists(adapter + "/adapter_config.json"):
                self.terminal("STOP", "--score-only %s: no adapter at %s" % (tag, adapter))
                return 2
            missing = [p for p in (scc.BASE_BATTERY, scc.BASE_ARC, SPEC["base_gsm512"], SPEC["base_gsmext"])
                       if not os.path.exists(p)]
            if missing:
                self.terminal("STOP", "--score-only %s: verdict reference(s) missing: %s" % (tag, ", ".join(missing)))
                return 2
            os.makedirs(self.run + "/" + tag, exist_ok=True)
            self.log("SCORE-ONLY %s | seed %d | adapter %s | control %s" % (
                tag, seed, adapter, self.a.control_dir or ("twin4060_%s (default)" % stag if tag.startswith("dp2_") else "none (vs base only)")))
            t5 = t4 = None
            for box in ("5090", "4060"):
                p = self.run + "/%s/TRAIN_%s.json" % (tag, box)
                if os.path.exists(p):
                    if box == "5090":
                        t5 = json.load(io.open(p, encoding="utf-8"))
                    else:
                        t4 = json.load(io.open(p, encoding="utf-8"))
            if tag.startswith("dp2_"):
                # F1/F5 (fable review 2026-09-12): the VOID clauses and the control identity are
                # settled BEFORE 1.83 h of scoring; a VOID pair never gets a verdict on this path either
                why = dp2_score_only_problem(tag, stag, self.run, self.a.control_dir, t5, t4)
                if why:
                    self.terminal("STOP", "--score-only %s refused: %s" % (tag, why))
                    return 2
                if self.dp2_post_run_void(tag, t5, t4):
                    self.terminal("VOID", "--score-only %s: the pair VOIDs on its manifests -- see VOID_%s.json" % (tag, tag))
                    return 3
            P = self.score_arm(tag, seed, adapter)
            if self.a.plan:
                self.terminal("PLAN", "score-only plan printed")
                return 0
            if tag.startswith("dp2_") or self.a.control_dir:
                v = self.dp2_verdict(tag, stag, P, t5, t4, primary_dir=self.a.control_dir or None)
                self.terminal("COMPLETE" if v else "STOP", "score-only %s -> %s" % (tag, (v or {}).get("verdict", "PENDING_CONTROL")))
                return 0 if v else 2
            # a control being scored: vs-base reads only, written as VERDICT_<tag>.json for the record
            v = {"arm": tag, "kind": "control", "seed": seed, "artefacts_rejected":
                 [k for k in ("battery", "gsm", "gsmext", "arc") if not P.get("ok_" + k)]}
            try:
                sj = json.load(io.open(P["gsm"], encoding="utf-8"))
                v["versions"] = sj.get("versions")
            except Exception:                                                # noqa: BLE001
                v["versions"] = None
            if P.get("ok_battery"):
                c = paired_read.compare(scc.BASE_BATTERY, P["battery"], 0.75)
                c.pop("gained_ids", None)
                v["battery_vs_base"] = c
            if P.get("ok_gsm") and P.get("ok_gsmext"):
                mine = self.pooled(P["gsm"], P["gsmext"])
                base = self.pooled(SPEC["base_gsm512"], SPEC["base_gsmext"])
                ids = set(base) & set(mine)
                b = paired_read.mcnemar({k: base[k] for k in ids}, {k: mine[k] for k in ids})
                b.pop("ids", None)
                v["gsm1319_vs_base"] = b
            if P.get("ok_arc"):
                c = paired_read.compare(scc.BASE_ARC, P["arc"], 1.5)
                c.pop("gained_ids", None)
                v["arc_vs_base"] = c
            v["verdict"] = "SCORED" if not v["artefacts_rejected"] else "INCOMPLETE"
            with io.open(self.run + "/%s/VERDICT_%s.json" % (tag, tag), "w", encoding="utf-8") as fh:
                json.dump(v, fh, indent=1, default=str)
            self.terminal("COMPLETE", "score-only %s -> %s" % (tag, v["verdict"]))
            return 0 if v["verdict"] == "SCORED" else 3

        def go_dp2(self):
            self.token = read_token(self.log)
            self.anchor = Anchor(SPEC["anchor"], self.token, self.log)
            self.run_id = self.a.run_id or time.strftime("dp2%Y%m%d_%H%M")
            self.summary["run_id"] = self.run_id
            self.summary["mode"] = "dp2"
            self.corpus_sha = ""
            self.script_sha = sha256_file(os.path.abspath(__file__))   # F2: real sha from the first line (preflight re-reads it)
            io.open(self.run + "/chain.pid", "w").write("%d\n" % os.getpid())
            self.log("=" * 78)
            if self.a.score_only:
                self.corpus_sha = sha256_file(SPEC["corpus_5090"]) if os.path.exists(SPEC["corpus_5090"]) else ""
                return self.score_only()
            only = set(self.a.only.split(",")) if self.a.only else None

            def want(p):
                return only is None or p in only
            self.log("DP-2 CHAIN %s | run %s | deadline %s | only=%s | anchor_gc=%s | arms %s"
                     % ("PLAN" if self.a.plan else "REAL", self.run_id, self.a.deadline, self.a.only or "all",
                        "OFF (--no-anchor-gc)" if self.a.no_anchor_gc else "ON (owner-approved 2026-09-12)",
                        ", ".join(t for t, _, _ in DP2_ARMS)))
            self.log("registered by docs/research/PREREG_DP2_BEATS_ONE_4060_2026-09-12.md; 5090 interpreter %s, "
                     "4060 interpreter %s" % (self.py_for("5090"), self.a.py4060))
            probs = self.preflight() + self.dp2_preflight_extra(want_arms=want("arms"))
            self.phase("P0_PREFLIGHT", "VOID" if probs else "OK", problems=probs)
            if probs:
                self.terminal("VOID", "preflight: " + "; ".join(probs)[:600])
                return 3
            if only is not None and "fidpair" in only:          # opt-in: dp2_fidelity_gate.py asks for it
                if not self.fidpair_phase():
                    self.terminal("STOP", "fidelity pair did not complete")
                    return 3
                if not want("arms") or (only is not None and "arms" not in only):
                    self.terminal("COMPLETE" if not self.a.plan else "PLAN", "fidelity pair done")
                    return 0
            if want("arms"):
                _wanted = set(x for x in (self.a.arms or "").split(",") if x)
                _bad = _wanted - {t for t, _, _ in DP2_ARMS}
                if _bad:
                    self.terminal("STOP", "--arms names unknown DP-2 arm(s): %s" % ",".join(sorted(_bad)))
                    return 2
                for tag, _k, stag in DP2_ARMS:
                    if _wanted and tag not in _wanted:
                        self.log("SKIP %s (not in --arms %s)" % (tag, self.a.arms))
                        continue
                    if self.stop_requested():
                        self.terminal("STOP", "STOP file")
                        return 2
                    if tag == "dp2_s2":
                        # KILL after seed 1 (prereg section 2), read from the ARTEFACT, never assumed
                        # (dp2_kill_from_dir: pure over the run dir, dp2_selftest (iv); F3: an
                        # INCOMPLETE / unevaluable seed 1 REFUSES, it never licenses by default)
                        why, v1 = dp2_kill_from_dir(self.run)
                        if why:
                            self.phase("P6_KILL", "KILL", why=why)
                            self.terminal("KILL", "%s -- dp2_s2 not launched" % why)
                            return 5
                        self.log("KILL check on dp2_s1: licensed (%s)" % (v1 or {}).get("verdict"))
                    if self.dp2_arm_phase(tag, stag) is None:
                        self.terminal("STOP", "%s did not complete (VOID, STOP or PENDING_CONTROL -- see the phase record)" % tag)
                        return 2
                # two-seed summary (prereg: PASS requires BOTH seeds)
                vs = {t: (self.summary["arms"].get(t) or {}).get("verdict") for t, _, _ in DP2_ARMS}
                if all(v in ("PASS",) for v in vs.values()):
                    overall = "PASS"
                elif any(v in ("BAD",) for v in vs.values()):
                    overall = "BAD"
                elif any(v in ("VETOED", "VOID") for v in vs.values()):
                    overall = "VETOED/VOID"
                elif any(v is None or v in ("PLAN", "INCOMPLETE") for v in vs.values()):
                    overall = "INCOMPLETE"
                else:
                    overall = "UNRESOLVED"
                self.summary["dp2_overall"] = {"per_seed": vs, "overall": overall}
                with io.open(self.run + "/DP2_SUMMARY.json", "w", encoding="utf-8") as fh:
                    json.dump(self.summary["dp2_overall"], fh, indent=1)
                self.log("DP-2 two-seed summary: %s (%s)" % (overall, json.dumps(vs)))
            self.summary["finished"] = now()
            self.save()
            self.terminal("COMPLETE" if not self.a.plan else "PLAN", "DP-2 chain done")
            return 0

        # ------------------------------------------------------------ go
        def go(self):
            if self.a.dp2 or self.a.score_only:      # 2026-09-12: the DP-2 chain lives in go_dp2
                return self.go_dp2()
            self.token = read_token(self.log)
            self.anchor = Anchor(SPEC["anchor"], self.token, self.log)
            self.run_id = self.a.run_id or time.strftime("wg%Y%m%d_%H%M")
            self.summary["run_id"] = self.run_id
            self.corpus_sha = ""
            self.script_sha = ""
            io.open(self.run + "/chain.pid", "w").write("%d\n" % os.getpid())
            self.log("=" * 78)
            self.log("WAN-GRADIENT CHAIN %s | run %s | deadline %s | only=%s"
                     % ("PLAN" if self.a.plan else "REAL", self.run_id, self.a.deadline, self.a.only or "all"))
            self.log("arms: %s (C-SOLO cut: banked D1-TB twin)" % ", ".join(t for t, _, _ in ARMS))
            self.log("base policy %s: 5090 on %s (interpreter %s), 4060 on %s (interpreter %s); control "
                     "twin %s" % (self.a.base_policy, BASE_POLICIES[self.a.base_policy]["5090"],
                                  self.py_for("5090"), BASE_POLICIES[self.a.base_policy]["4060"],
                                  self.a.py4060, json.dumps(self.twins())))
            only = set(self.a.only.split(",")) if self.a.only else None

            def want(p):
                return only is None or p in only

            probs = self.preflight()
            self.phase("P0_PREFLIGHT", "VOID" if probs else "OK", problems=probs)
            if probs:
                self.terminal("VOID", "preflight: " + "; ".join(probs)[:600])
                return 3
            if want("aa") and not self.aa_phase():
                self.terminal("VOID", "A/A")
                return 3
            if want("stage") and not self.stage_phase():
                self.terminal("BLOCKED", "4060 staging")
                return 2
            if want("footprint") and not self.footprint_phase():
                self.terminal("VOID", "design 7(a) footprint gate / cost bound")
                return 3
            if want("basece") and not self.basece_phase():
                self.terminal("VOID", "design 7(b) base-consistency gate")
                return 3
            if want("track") and not self.tracking_phase():
                self.terminal("STOP", "design 7(c) tracking gate")
                return 3
            # P6: the H=1 fidelity anchor FIRST, then the section-5 KILL
            if want("arms"):
                _wanted = set(x for x in (self.a.arms or "").split(",") if x)
                _bad = _wanted - {t for t, _, _ in ARMS}
                if _bad:
                    self.terminal("STOP", "--arms names unknown arm(s): %s" % ",".join(sorted(_bad)))
                    return 2
                if _wanted and "h1_s1" not in _wanted and not os.path.exists(
                        self.run + "/h1_s1/VERDICT_h1_s1.json"):
                    self.terminal("STOP", "--arms %s excludes h1_s1 but no h1_s1 verdict exists -- the "
                                          "section-5 KILL read must precede every other arm; nothing "
                                          "bought" % self.a.arms)
                    return 2
                v = self.arm_phase("h1_s1", "h1", "s1")
                if v is None:
                    self.terminal("STOP", "h1_s1 did not complete")
                    return 2
                if not self.a.plan and v.get("verdict") == "INCOMPLETE":
                    # MEDIUM-2: the section-5 KILL reads battery_W_minus_C. If h1_s1's artefacts
                    # were rejected that read is ABSENT, the KILL cannot fire, and the chain would
                    # spend H25 x2 + INDEP on a verdict that was never computed.
                    self.terminal("STOP", "h1_s1 INCOMPLETE (artefacts rejected: %s) -- the "
                                          "section-5 KILL cannot be evaluated, so the remaining "
                                          "arms are not licensed"
                                          % ",".join(v.get("artefacts_rejected") or ["unknown"]))
                    return 3
                if not self.a.plan:
                    m = (v.get("reads") or {}).get("battery_W_minus_C")
                    if m and (-m["delta"]) > 2.0 * m["mde"]:
                        self.phase("P6_KILL", "KILL", C_minus_W=-m["delta"], two_mde=2.0 * m["mde"])
                        self.terminal("KILL", "design section 5: W-DILOCO-H1 worse than C-SOLO by "
                                              "%.4f pp > 2xMDE %.4f -- the mechanism does not hold "
                                              "even at H=1; H25/INDEP hours not spent"
                                              % (-m["delta"], 2.0 * m["mde"]))
                        return 5
                if self.a.stop_after_h1:
                    self.phase("P7_ARMS", "STOP_AFTER_H1", spent_arm="h1_s1",
                               not_bought=[t for t, _, _ in ARMS[1:]],
                               why="owner bought h1_s1 only (~7 h) at the measured 7.014 h/arm; "
                                   "the remaining three arms are licensed by h1_s1's KILL read, "
                                   "not by this run")
                    self.summary["finished"] = now()
                    self.save()
                    self.terminal("COMPLETE", "h1_s1 complete; ARMS[1:] deliberately not bought")
                    return 0
                for tag, kind, stag in ARMS[1:]:
                    if _wanted and tag not in _wanted:
                        self.log("SKIP %s (not in --arms %s)" % (tag, self.a.arms))
                        continue
                    if self.stop_requested():
                        self.terminal("STOP", "STOP file")
                        return 2
                    if self.arm_phase(tag, kind, stag) is None:
                        self.terminal("STOP", "%s did not complete" % tag)
                        return 2
            self.summary["finished"] = now()
            self.save()
            self.terminal("COMPLETE" if not self.a.plan else "PLAN", "all registered arms have verdicts")
            return 0

    # 2026-09-12 07:35: `a.arms = []` used to sit here and silently discarded the CLI value, so
    # `--arms h1_s2` never filtered anything -- after h1_s2's verdict the chain launched h25_s1
    # on both boxes (stopped at +2 min). Plan mode had shown it on 09-11 15:41 and was misread.
    os.makedirs(a.run_dir, exist_ok=True)
    return WanChain(a).go()


# ----------------------------------------------------------------------------- role: env-probe (any box)
def role_env_probe(a):
    """Print what THIS interpreter can do for the nf4 base (versions + meta-device expert layout)
    and the nf4 verdict on it. No GPU, no weights. The last line of TF4_ENV_RECIPE; also what
    nf4_twin_chain.py runs in preflight under --py-train."""
    os.makedirs(a.out_dir, exist_ok=True)
    log = make_logger(None)
    env = local_env_probe(sys.executable, a.out_dir, log, a.model_dir or SPEC["model_dir_5090"])
    probs = nf4_env_problems(env, sys.executable, a.box)
    log("ENV " + json.dumps(env))
    log("NF4-VERDICT %s%s" % ("OK: this interpreter can produce the unfused nf4 base" if not probs
                              else "NO", "" if not probs else " -- " + " | ".join(probs)))
    return 0 if not probs else 3


# ----------------------------------------------------------------------------- DP-2 (2026-09-12)
# Two-box DATA-PARALLEL gradient averaging at H=1, registered in
# docs/research/PREREG_DP2_BEATS_ONE_4060_2026-09-12.md (design: FABLE_FLEET_VS_ONE_BOX_2026-09-12.md
# section 4.1/4.4). Everything below is NEW code behind `--dp2` / `--role dp2-trainer`; the H=1
# ownership path above is byte-untouched (its `--plan` output is the regression check).
#
# The step, in one paragraph. Both boxes hold the full 16-layer LoRA and the full AdamW state. Step t
# takes rows [32t, 32t+32) of ONE shared sequence (epoch 0 = the banked select_rows order, epoch k>=1
# the same rows re-shuffled by random.Random(seed + 7919*k)); the 5090 computes the SUM of its 10
# micro-batch gradients on rows [32t, 32t+20), the 4060 the sum of its 6 on [32t+20, 32t+32) --
# `ce.backward()` UNscaled, so `.grad` is the plain sum. Each box flattens its sum in the fixed
# parameter order (d1_sft.py's `[p for p in model.parameters() if p.requires_grad]`), rounds it to
# bf16 for the wire, publishes it, fetches the peer's, and forms the SAME fp32 mean on both sides:
# (bf16(S_5090).float() + bf16(S_4060).float()) / 16 -- 16 = the total micro-batch count, so this is
# d1_sft.py `--accum 16` (which accumulates `(ce_j/16).backward()`) on the same 32 rows: scaling by a
# power of two commutes exactly with IEEE rounding, the only differences are the bf16 wire rounding
# and the association order of the sums. Then the twin's clip (global 2-norm, max 1.0, torch's own
# coef formula) with the norm computed DETERMINISTICALLY (float64 pairwise sum on the CPU, rounded to
# fp32, so two architectures agree on the scalar), and AdamW with foreach=False, fused=False (plain
# elementwise IEEE ops). Every 25 steps the 5090 publishes its fp32 adapter; the 4060 compares it
# bit-for-bit with its own, logs RESYNC, and ADOPTS the 5090's copy if they differ (a repair event).
#
# WIRE OBJECTS. Every object is content-addressed on the anchor; a step's gradient is TWO objects:
# the payload (header + bytes, PUT with NO name, so the store's names.json is never rewritten for it)
# and a ~150 B pointer {"kind":"ptr","sha":<payload sha>} PUT under the per-step NAME
# wangrad/<ns>/grad_<box>_s<step> and VERIFIED by a name lookup after the PUT: the store's names.json
# write is an unlocked read-modify-write (content_store.py:322, read 2026-09-12 over ssh), so two
# concurrent named PUTs -- which DP-2 produces, both boxes finishing within seconds of each other --
# can lose one registration; the sender detects that and re-registers a 150 B object instead of
# re-uploading 8.4 MB. The receiver never confuses step t-1 with step t: the name carries the step,
# the header carries step/box/kind/numel/dtype and is asserted on receipt (dp2_unpack).
DP2 = {
    "rows_per_step": 32, "epochs": 2,
    "micro": {"5090": 10, "4060": 6},          # x micro_batch 2 = 20 / 12 rows per step
    "offset": {"5090": 0, "4060": 20},         # rows [32t+0, 32t+20) / [32t+20, 32t+32)
    "total_micro": 16,                          # = d1_sft.py --accum 16 on the same 32 rows
    "share": {"5090": 20.0 / 32.0, "4060": 12.0 / 32.0},      # 62.5 / 37.5
    "resync_every": 25, "eval_every": 25, "gc_keep_steps": 4,
    # prereg section 2 VOID thresholds (clauses 2, 3, 6)
    "sync_void_mean_s": 10.0, "share_tol_rel": 0.10, "repair_max": 2,
    # AMENDMENT 2 (2026-09-12, fable decision 2 Q2): clause 2's every-25-step EARLY STOP is the
    # PROJECTED run mean, not the running mean -- an OUTAGE step (critical-path sync > 120 s) is
    # excluded from the baseline B taken over the last 50 steps (dp2_sync_stats)
    "sync_outage_s": 120.0, "sync_window_steps": 50,
    # clause 1: the fidelity gate artefact and its bands
    "fidelity_json": "E:/rsft_runs/wangrad/dp2_fidelity/FIDELITY.json",
    "fidelity_gate_step": 30, "fid_max_abs_diff": 1e-3,
    "fid_loss_step1_tol": 0.002, "fid_loss_band": 0.05, "fid_gnorm_rel_tol": 0.01,
    # controls (prereg section 1): primary = the 4060 twin at equal wall-clock; secondaries
    "twin4060": {"s1": "E:/rsft_runs/wangrad/twin4060_s1", "s2": "E:/rsft_runs/wangrad/twin4060_s2"},
    "twin2ep": {"s1": "E:/rsft_runs/wangrad/twin2ep_s1", "s2": "E:/rsft_runs/wangrad/twin2ep_s2"},
    "primary_bar_pp": 3.00, "alpha": 0.05,
    # anchor hygiene (prereg section 3)
    "anchor_min_free_gb": 12.0,
    "ssh": (lambda t: None if not t else (json.loads(t) if t[:1] == "[" else t.split()))(   # PUBLIC copy:
        os.environ.get("NEURAHASH_ANCHOR_SSH", "").strip()),   # ssh argv from the env ONLY; unset = None = OFF
    "vps_store_root": os.environ.get("NEURAHASH_ANCHOR_STORE_ROOT", ""),   # PUBLIC copy: env only, no default
}
DP2_ARMS = [("dp2_s1", "dp2", "s1"), ("dp2_s2", "dp2", "s2")]     # KILL after s1 gates s2
DP2_STEPS = SPEC["n_traces"] * DP2["epochs"] // DP2["rows_per_step"]           # 625
assert DP2_STEPS == EXPECT_STEPS, "DP-2 must take exactly the twin's 625 optimizer steps"
assert sum(DP2["micro"].values()) == DP2["total_micro"]
assert DP2["total_micro"] * SPEC["micro_batch"] == DP2["rows_per_step"]
assert DP2["offset"]["4060"] == DP2["micro"]["5090"] * SPEC["micro_batch"]
DP2_MAGIC = b"DP2G"
DP2_WIRE_FMT = 1


def dp2_content_key(r):
    """d1_sft.content_key, verbatim: sha256(prompt + NUL + completion)[:16]. The corpus carries NO
    `id` field, so the manifest's selected_order_sha256 formula (str(row id) per row) hashes 10,000 x
    "None" and cannot tell two orders apart (D1SFT_FLAGS_READY.txt 2026-09-12); this key can."""
    return hashlib.sha256((r["prompt"] + "\x00" + r["completion"]).encode("utf-8")).hexdigest()[:16]


def epoch_sequence(rows0, seed, epochs):
    """Prereg section 1 row rule: epoch 0 = rows0 (select_rows' shuffle-then-slice order, untouched);
    epoch k >= 1 = a FRESH COPY of the same rows shuffled by random.Random(seed + 7919 * k) -- the
    global `random` state is never consumed. Returns (seq, epoch_order_sha256, epoch_content_sha256),
    one digest per epoch: the first is the manifest's id formula (degenerate on this corpus, kept for
    the record), the second is d1_sft.epoch_sequence's content digest -- the one both boxes and the
    fidelity reference must agree on. Same arithmetic as d1_sft.epoch_sequence (cross-checked by
    dp2_selftest.py (i)); duplicated here because the trainer role never imports the repo."""
    seq, order, content = [], [], []
    for k in range(int(epochs)):
        ep = list(rows0)
        if k >= 1:
            random.Random(seed + 7919 * k).shuffle(ep)
        order.append(hashlib.sha256("\n".join(str(r.get("id")) for r in ep).encode("utf-8")).hexdigest())
        content.append(hashlib.sha256("\n".join(dp2_content_key(r) for r in ep).encode("utf-8")).hexdigest())
        seq.extend(ep)
    return seq, order, content


def dp2_step_rows(t, box):
    """[lo, hi) row indices of the shared sequence that `box` consumes at 0-based step t."""
    lo = t * DP2["rows_per_step"] + DP2["offset"][box]
    return lo, lo + DP2["micro"][box] * SPEC["micro_batch"]


def dp2_ref_step_rows(t):
    """The single-box reference (d1_sft.py --accum 16): rows [32t, 32t+32) as 16 micro-batches."""
    lo = t * DP2["rows_per_step"]
    return lo, lo + DP2["rows_per_step"]


# ---- wire format
def dp2_pack(header, payload):
    """magic | uint32 header length | header JSON | payload. The header gets the payload's sha256 and
    byte count added; dp2_unpack refuses anything that disagrees with itself or with `expect`."""
    h = dict(header)
    h["fmt"] = DP2_WIRE_FMT
    h["payload_sha256"] = sha256_bytes(payload)
    h["payload_bytes"] = len(payload)
    hb = json.dumps(h, sort_keys=True).encode("utf-8")
    return DP2_MAGIC + len(hb).to_bytes(4, "little") + hb + bytes(payload)


def dp2_unpack(blob, expect=None):
    """-> (header, payload). ValueError on a bad magic/length/sha, or on ANY field of `expect`
    (step, box, kind, numel, dtype, ...) that the header does not carry with the same value --
    this is the check that a step-(t-1) object can never be taken for step t."""
    if blob[:4] != DP2_MAGIC:
        raise ValueError("bad magic %r" % blob[:4])
    n = int.from_bytes(blob[4:8], "little")
    if n <= 0 or 8 + n > len(blob):
        raise ValueError("bad header length %d for %d bytes" % (n, len(blob)))
    h = json.loads(blob[8:8 + n].decode("utf-8"))
    payload = blob[8 + n:]
    if h.get("fmt") != DP2_WIRE_FMT:
        raise ValueError("wire fmt %r != %d" % (h.get("fmt"), DP2_WIRE_FMT))
    if len(payload) != h.get("payload_bytes"):
        raise ValueError("payload %d B != header %r" % (len(payload), h.get("payload_bytes")))
    if sha256_bytes(payload) != h.get("payload_sha256"):
        raise ValueError("payload sha256 mismatch")
    for k, v in (expect or {}).items():
        if h.get(k) != v:
            raise ValueError("header %s=%r, expected %r" % (k, h.get(k), v))
    return h, payload


def dp2_tensor_bytes(t, dtype_name):
    """fp32 tensor -> little-endian bytes as bf16 (2 B/elem, round-to-nearest-even) or fp32."""
    import torch
    t = t.detach().reshape(-1)
    if dtype_name == "bf16":
        return t.to(torch.bfloat16).contiguous().cpu().view(torch.int16).numpy().tobytes()
    return t.to(torch.float32).contiguous().cpu().numpy().tobytes()


def dp2_bytes_tensor(b, dtype_name, numel):
    """bytes -> CPU tensor (bf16 or fp32) of exactly `numel` elements, or ValueError."""
    import numpy as _np
    import torch
    width = 2 if dtype_name == "bf16" else 4
    if len(b) != numel * width:
        raise ValueError("payload %d B != %d x %d" % (len(b), numel, width))
    if dtype_name == "bf16":
        return torch.from_numpy(_np.frombuffer(b, dtype=_np.int16).copy()).view(torch.bfloat16)
    return torch.from_numpy(_np.frombuffer(b, dtype=_np.float32).copy())


# ---- the math (pure over tensors; the selftest drives these on CPU)
def dp2_param_order(model):
    """d1_sft.py:426's exact list -- every requires_grad param in model order -- plus a sha over
    (canonical name, shape, dtype) so the two boxes can assert they flatten in the SAME order."""
    params = [p for p in model.parameters() if p.requires_grad]
    names = [canon(n) for n, p in model.named_parameters() if p.requires_grad]
    h = hashlib.sha256()
    for n, p in zip(names, params):
        h.update(("%s|%s|%s\n" % (n, tuple(p.shape), str(p.dtype))).encode("utf-8"))
    return params, names, h.hexdigest()


def dp2_flatten(params, attr="grad"):
    """Concatenate `.grad` (or the params themselves) in the fixed order into ONE fp32 tensor.
    A missing gradient is zeros (and counted by the caller through dp2_none_grads)."""
    import torch
    parts = []
    for p in params:
        t = p.grad if attr == "grad" else p.data
        parts.append((torch.zeros_like(p) if t is None else t).detach().reshape(-1).to(torch.float32))
    return torch.cat(parts)


def dp2_none_grads(params):
    return sum(1 for p in params if p.grad is None)


def dp2_scatter_(params, flat, attr="grad"):
    """Write `flat` back over the params' grads (views of one buffer) or their data (copy_)."""
    import torch
    off = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            chunk = flat[off:off + n].view_as(p)
            if attr == "grad":
                p.grad = chunk.to(p.dtype) if chunk.dtype != p.dtype else chunk
            else:
                p.copy_(chunk.to(p.dtype))
            off += n
    if off != flat.numel():
        raise RuntimeError("scatter: %d of %d elements used" % (off, flat.numel()))


def dp2_average(g5090, g4060, total_micro=None):
    """The identical averaged gradient both boxes step with: fp32 (g5090 + g4060) / total_micro in
    that fixed order. Inputs are the two WIRE tensors (each box uses the bf16 copy of its OWN sum
    too, never its local fp32, so the operands are bit-identical on both sides)."""
    import torch
    tm = float(total_micro or DP2["total_micro"])
    return (g5090.to(torch.float32) + g4060.to(torch.float32)) / tm


def deterministic_norm(flat):
    """2-norm of an fp32 vector computed the SAME way on any x86 box: numpy float64 pairwise sum
    of exact squares (an fp32 square is exact in float64), sqrt in float64, rounded to fp32.
    A CUDA reduction's order differs by architecture and can move the last ULP; torch's CPU
    reduction depends on the thread count. Returns a python float that IS an fp32 value."""
    import numpy as _np
    import torch
    x = flat.detach().reshape(-1).to("cpu", dtype=torch.float32).numpy().astype(_np.float64)
    ss = _np.add.reduce(_np.square(x))
    return float(_np.float32(_np.sqrt(ss)))


def dp2_clip_(params, max_norm, total_norm):
    """torch.nn.utils.clip_grad_norm_'s update with the norm supplied: coef = clamp(max_norm /
    (total_norm + 1e-6), max=1.0) in fp32 tensor arithmetic (torch's own formula), applied as one
    elementwise multiply to every gradient (torch multiplies even when coef == 1.0; x*1.0 is exact).
    Returns the coef as a python float."""
    import torch
    tn = torch.tensor(float(total_norm), dtype=torch.float32)
    coef = torch.clamp(torch.tensor(float(max_norm), dtype=torch.float32) / (tn + 1e-6), max=1.0)
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        torch._foreach_mul_(grads, coef.to(grads[0].device))
    return float(coef)


# ---- VOID / KILL predicates (pure; the trainer evaluates them every 25 steps, the chain after the
# run, dp2_selftest.py on synthetic records with BOTH controls for each gate)
def dp2_sync_stats(cs, planned_steps, outage_s=None, window=None):
    """AMENDMENT 2 (prereg section 2 clause 2 (ii), fable decision 2 Q2, 2026-09-12): outage
    bookkeeping and the PROJECTED run mean, computed identically on both boxes from the same
    critical_sync_seconds list. An OUTAGE step is a step whose critical-path sync > outage_s
    (120 s). B = mean over the NON-outage steps among the last `window` (50) steps -- all
    non-outage steps if fewer than 50 exist, the overall non-outage mean if the window holds none,
    None if no non-outage step exists at all (the caller fires). P = (sum over all steps so far
    + B x (planned - steps so far)) / planned. Pure; returns a dict (B/P None when undefined)."""
    outage_s = float(DP2["sync_outage_s"] if outage_s is None else outage_s)
    window = int(DP2["sync_window_steps"] if window is None else window)
    T = int(planned_steps or DP2_STEPS)
    cs = [float(x) for x in (cs or [])]
    n = len(cs)
    out_idx = [i for i, x in enumerate(cs) if x > outage_s]
    episodes = sum(1 for i in out_idx if i == 0 or cs[i - 1] <= outage_s)
    last = cs[-window:] if window > 0 else cs
    non_win = [x for x in last if x <= outage_s]
    non_all = [x for x in cs if x <= outage_s]
    B = (sum(non_win) / len(non_win)) if non_win else ((sum(non_all) / len(non_all)) if non_all else None)
    total = sum(cs)
    P = ((total + B * max(T - n, 0)) / T) if (B is not None and T > 0) else None
    return {"n": n, "planned_steps": T, "sum_s": total, "mean_s": (total / n) if n else None,
            "outage_threshold_s": outage_s, "outage_steps": len(out_idx), "outage_episodes": episodes,
            "outage_seconds": sum(cs[i] for i in out_idx), "window_steps": window,
            "window_non_outage": len(non_win), "B": B, "P": P}


def dp2_void_problems(rec):
    """rec keys: crit_sync (list of per-step critical-path sync seconds, both boxes' view),
    share_own / share_expect (realised vs registered row share of THIS box), gnorm_own / gnorm_peer
    (per-step partial-sum norms, one list per box), repair_events (int), fidelity_required (bool),
    fidelity_passed (bool|None), window (int, default 25), step (int), planned_steps (int, default
    625), final (bool: True = the FINISHED run's manifest re-check, clause 2 (i); False/absent = the
    trainer's every-25-step EARLY STOP, clause 2 (ii), AMENDMENT 2). Returns ['VOID n: ...'].
    Clause numbers are the prereg's section 2."""
    probs = []
    w = int(rec.get("window") or DP2["eval_every"])
    cs = [float(x) for x in (rec.get("crit_sync") or [])]
    if cs and rec.get("final"):
        # clause 2 (i), FINAL and UNCHANGED: the run mean over ALL steps, outage steps included
        mean = sum(cs) / len(cs)
        if mean > DP2["sync_void_mean_s"]:
            probs.append("VOID 2: critical-path sync mean %.2f s/step over %d steps > %.1f s"
                         % (mean, len(cs), DP2["sync_void_mean_s"]))
    elif cs:
        # clause 2 (ii), AMENDMENT 2 (2026-09-12): the every-25-step EARLY STOP is the PROJECTED
        # run mean P with outage steps (> 120 s) excluded from the baseline B -- the plain running
        # mean voided attempt 1 at step 50 on ONE external 13-min anchor stall (steps 31-33) that the
        # registered run-mean rule tolerates (projected 8.50 s). Fires iff P > 10 or B > 10.
        st = dp2_sync_stats(cs, rec.get("planned_steps"))
        tail = ("B %s s over %d non-outage of the last %d steps; %d outage steps (> %.0f s) in %d episodes, "
                "%.0f s; sum %.0f s over %d of %d steps"
                % (("%.2f" % st["B"]) if st["B"] is not None else "n/a", st["window_non_outage"],
                   min(len(cs), st["window_steps"]), st["outage_steps"], st["outage_threshold_s"],
                   st["outage_episodes"], st["outage_seconds"], st["sum_s"], st["n"], st["planned_steps"]))
        if st["B"] is None:
            probs.append("VOID 2: every step so far is an outage step -- no non-outage baseline exists (%s)" % tail)
        elif st["B"] > DP2["sync_void_mean_s"]:
            probs.append("VOID 2: non-outage sync mean B %.2f s/step > %.1f s (P %.2f; %s)"
                         % (st["B"], DP2["sync_void_mean_s"], st["P"], tail))
        elif st["P"] > DP2["sync_void_mean_s"]:
            probs.append("VOID 2: projected run mean P %.2f s/step > %.1f s (%s)" % (st["P"], DP2["sync_void_mean_s"], tail))
    so, se = rec.get("share_own"), rec.get("share_expect")
    if so is not None and se:
        rel = abs(float(so) - float(se)) / float(se)
        if rel > DP2["share_tol_rel"]:
            probs.append("VOID 3: realised row share %.4f vs registered %.4f (%.1f%% relative > %.0f%%)"
                         % (float(so), float(se), 100 * rel, 100 * DP2["share_tol_rel"]))
    for box in ("own", "peer"):
        g = rec.get("gnorm_" + box)
        if g:
            win = [float(x) for x in g[-w:]]
            if max(win) == 0.0:
                probs.append("VOID 4: gradient-liveness -- gnorm_max == 0 on the %s box over the last %d steps"
                             % (box, len(win)))
    re_ = rec.get("repair_events")
    if re_ is not None and int(re_) > DP2["repair_max"]:
        probs.append("VOID 6: %d repair events at the 25-step re-sync > %d -- the two boxes are not "
                     "holding one model" % (int(re_), DP2["repair_max"]))
    if rec.get("fidelity_required"):
        step = int(rec.get("step") or 0)
        if step >= DP2["fidelity_gate_step"] + 1 and rec.get("fidelity_passed") is not True:
            probs.append("VOID 1: fidelity gate artefact %s at step %d (passed=%r)"
                         % ("absent/failed", step, rec.get("fidelity_passed")))
    return probs


def dp2_manifest_void_problems(t5, t4):
    """Prereg section 2 VOID clauses over a FINISHED pair's two manifests -- the artefact-level
    re-check the chain (dp2_arm_phase) and --score-only run before 1.83 h of scoring. Clause 2 is
    read here over the WHOLE run (critical_sync_seconds = every step): the run mean the prereg
    names (clause 2 (i), unchanged by AMENDMENT 2); the trainers' every-25-step evaluation is the
    PROJECTED-mean EARLY STOP of AMENDMENT 2 (dp2_sync_stats; was the running mean, fable review
    F7, both documented in DP2_RUNNER_NOTES). VOID 6b (fable F8; prereg clause
    6b, amended 2026-09-12): the terminal fp32 adapters not bit-identical -- kept as a VOID in its
    own right because it protects the ONE-MODEL claim, distinct from clause 6's > 2 repairs."""
    probs = []
    fid5 = t5.get("fidelity") or {}
    if t5.get("final_identical_vs_peer") is not True:
        probs.append("VOID 6b: the two boxes did not end on ONE model -- final_identical_vs_peer=%r "
                     "max_abs_diff=%r" % (t5.get("final_identical_vs_peer"), t5.get("final_max_abs_diff")))
    for box, t in (("5090", t5), ("4060", t4)):
        if t.get("void"):
            probs.append("trainer %s declared VOID: %s" % (box, str(t.get("void"))[:200]))
        rec = {"crit_sync": t.get("critical_sync_seconds") or [],
               "share_own": t.get("realised_share_own"), "share_expect": t.get("row_share_expect_own"),
               "gnorm_own": t.get("gnorms") or [], "gnorm_peer": t.get("gnorm_peer") or [],
               "repair_events": t.get("repair_events"),
               "fidelity_required": bool(fid5.get("required")),
               "fidelity_passed": fid5.get("passed_at_launch"),
               "window": DP2["eval_every"], "step": t.get("steps") or 0,
               "planned_steps": t.get("planned_steps"), "final": True}
        for p in dp2_void_problems(rec):
            probs.append("%s: %s" % (box, p))
        if not (float(t.get("gnorm_max") or 0.0) > 0.0):
            probs.append("%s: VOID 4 gnorm_max=%r over the run" % (box, t.get("gnorm_max")))
    return probs


def dp2_fidelity_state(path):
    """(passed: bool|None, sha256|None, info) -- None when the artefact is absent/unreadable."""
    try:
        b = open(path, "rb").read()
        j = json.loads(b.decode("utf-8"))
        return (j.get("passed") is True), sha256_bytes(b), {"steps": j.get("steps"),
                                                            "max_abs_diff": j.get("max_abs_diff"),
                                                            "at": j.get("at"),
                                                            "chain_sha256": j.get("chain_sha256"),
                                                            "pair_repair_events": j.get("pair_repair_events")}
    except Exception as e:                                               # noqa: BLE001
        return None, None, {"error": repr(e)[:120]}


def dp2_fidelity_bound(path, script_sha):
    """F2 (fable review 2026-09-12): the certificate must be BOUND to the code it certifies.
    (passed_and_bound, sha, info): passed is True AND FIDELITY.chain_sha256 == sha256 of the chain
    file that is running (script_sha) AND the gate's pair needed 0 repairs (F4). A stale or unbound
    certificate reads False with info['why'] naming both shas."""
    passed, sha, info = dp2_fidelity_state(path)
    if passed is not True:
        return passed, sha, info
    cert = info.get("chain_sha256")
    if cert != script_sha:
        info["why"] = ("fidelity certificate is for chain sha %s, running sha %s -- re-run the gate"
                       % (str(cert)[:12] if cert else "NONE (unbound artefact)", str(script_sha)[:12]))
        return False, sha, info
    if info.get("pair_repair_events") != 0:
        info["why"] = "fidelity certificate records pair_repair_events=%r (must be 0) -- re-run the gate" % info.get("pair_repair_events")
        return False, sha, info
    return True, sha, info


def dp2_kill_reason(v1):
    """Prereg section 2 KILL after seed 1: `dp2_s1` vs `twin4060_s1` GSM8K <= -MDE at p < 0.05, or an
    ARC veto -> do not launch s2. Returns None (licensed) or a string; an UNREADABLE read is a refusal
    too (the KILL must be evaluable before seed 2 is bought). F3 (fable review 2026-09-12): seed 1
    licenses seed 2 ONLY with a decided primary read (verdict PASS/BAD/UNRESOLVED) AND an evaluable
    ARC read; INCOMPLETE, VOID, PENDING or a missing read never license by default."""
    if not isinstance(v1, dict) or not v1:
        return "KILL-UNDECIDABLE: no dp2_s1 verdict"
    if v1.get("verdict") == "VETOED" or v1.get("named_vetoes"):
        return "KILL: ARC veto on dp2_s1 (%s)" % "; ".join(v1.get("named_vetoes") or ["VETOED"])
    if v1.get("verdict") not in ("PASS", "BAD", "UNRESOLVED"):
        return "KILL-UNDECIDABLE: dp2_s1 verdict is %r, not a decided primary read (PASS/BAD/UNRESOLVED)" % v1.get("verdict")
    m = (v1.get("reads") or {}).get("gsm1319_W_minus_C")
    if not isinstance(m, dict) or any(m.get(k) is None for k in ("delta", "mde", "p")):
        return "KILL-UNDECIDABLE: dp2_s1 has no gsm1319_W_minus_C read vs twin4060_s1"
    arc = v1.get("arc_vs_base")
    if not isinstance(arc, dict) or any(arc.get(k) is None for k in ("delta", "p")):
        return "KILL-UNDECIDABLE: dp2_s1 has no evaluable ARC-Easy read (arc_vs_base) -- the veto half of the KILL cannot be read"
    if float(m["delta"]) <= -float(m["mde"]) and float(m["p"]) < DP2["alpha"]:
        return ("KILL: dp2_s1 vs twin4060_s1 GSM8K %+.4f pp <= -MDE %.4f at p=%.3g -- the fleet COSTS "
                "accuracy at equal wall-clock" % (float(m["delta"]), float(m["mde"]), float(m["p"])))
    return None


def dp2_kill_from_dir(run_dir):
    """(why, v1): the KILL check over the ARTEFACTS in run_dir/dp2_s1 -- VERDICT_dp2_s1.json overlaid
    with its _ADJUDICATED verdict when that is VOID/VETOED. A missing file is a refusal
    (KILL-UNDECIDABLE), an unparsable one raises: nothing licenses seed 2 by default."""
    vp = run_dir + "/dp2_s1/VERDICT_dp2_s1.json"
    v1 = json.load(io.open(vp, encoding="utf-8")) if os.path.exists(vp) else None
    ap = run_dir + "/dp2_s1/VERDICT_dp2_s1_ADJUDICATED.json"
    if v1 is not None and os.path.exists(ap):
        adj = json.load(io.open(ap, encoding="utf-8"))
        if adj.get("adjudicated_verdict") in ("VOID", "VETOED"):
            v1 = dict(v1, verdict=adj["adjudicated_verdict"],
                      named_vetoes=(v1.get("named_vetoes") or []) + ["adjudicated %s" % adj["adjudicated_verdict"]])
    return dp2_kill_reason(v1), v1


def dp2_score_only_problem(tag, stag, run_dir, control_dir, t5, t4):
    """F1/F5 (fable review 2026-09-12): why `--score-only <dp2 arm>` must NOT proceed, or None.
    The control may only be the registered primary (DP2['twin4060'][stag]); a VOID marker or a
    missing TRAIN manifest refuses (the VOID clauses are evaluated over both manifests)."""
    want = DP2["twin4060"][stag]
    if control_dir and os.path.normcase(os.path.abspath(control_dir)) != os.path.normcase(os.path.abspath(want)):
        return ("--control-dir %s is not the registered primary control %s for %s (refused, not warned)"
                % (control_dir, want, tag))
    if os.path.exists(run_dir + "/%s/VOID_%s.json" % (tag, tag)):
        return "VOID_%s.json exists -- a VOID pair is never scored" % tag
    if not isinstance(t5, dict) or not isinstance(t4, dict):
        return ("both TRAIN_5090.json and TRAIN_4060.json are required under %s/%s (5090 %s, 4060 %s) -- "
                "the VOID clauses are evaluated over them" % (run_dir, tag, "present" if isinstance(t5, dict) else "MISSING",
                                                              "present" if isinstance(t4, dict) else "MISSING"))
    return None


def dp2_verdict_problem(v, stag):
    """F1/F5: a VERDICT_dp2_*.json the chain may skip onto must itself prove the pair ended as ONE
    model, needed <= repair_max repairs and was scored against twin4060_<seed>; else why (the guard's
    dp2 profile applies the same three refusals independently)."""
    if not isinstance(v, dict):
        return "not a verdict object"
    if v.get("final_identical_vs_peer") is not True:
        return "final_identical_vs_peer=%r (VOID 6b: the pair is not proven to be one model)" % v.get("final_identical_vs_peer")
    re_ = v.get("repair_events")
    if isinstance(re_, bool) or not isinstance(re_, int) or re_ > DP2["repair_max"]:
        return "repair_events=%r (must be an integer <= %d, VOID 6)" % (re_, DP2["repair_max"])
    pc = os.path.basename(str(v.get("primary_control") or "").replace("\\", "/").rstrip("/"))
    if pc != "twin4060_%s" % stag:
        return "primary_control=%r is not twin4060_%s (scored against the wrong control)" % (v.get("primary_control"), stag)
    return None


def dp2_single_seed_verdict(m, vetoes):
    """Prereg section 2 primary on ONE seed from the paired read m = {delta, p, mde}: PASS if
    delta >= max(MDE, +3.00) and p < 0.05 and no veto; BAD if miss with MDE < 3.00; UNRESOLVED if
    miss with MDE >= 3.00 or |delta| < MDE. (Two-seed PASS is the chain summary's job.)"""
    if vetoes:
        return "VETOED"
    if not isinstance(m, dict) or any(m.get(k) is None for k in ("delta", "p", "mde")):
        return "INCOMPLETE"
    d, p, mde = float(m["delta"]), float(m["p"]), float(m["mde"])
    bar = max(mde, DP2["primary_bar_pp"])
    if d >= bar and p < DP2["alpha"]:
        return "PASS"
    if abs(d) < mde or mde >= DP2["primary_bar_pp"]:
        return "UNRESOLVED"
    return "BAD"


# ---- anchor: two-object publish/fetch, verified names, garbage collection over ssh
def put_named_verified(anchor, data, name, log, tries=4):
    """Named PUT + a name lookup: the store's names.json write is an unlocked read-modify-write
    (content_store.py:322), so a registration can be lost to a concurrent PUT; re-register (the
    object itself is content-addressed and already stored) instead of re-uploading."""
    sha = None
    for i in range(tries):
        sha = anchor.put(data, name)
        got = anchor.resolve(name)
        if got == sha:
            return sha
        log("  name %s not registered (resolved %s, mine %s) -- re-registering, try %d"
            % (name, str(got)[:12], sha[:12], i + 1))
        time.sleep(1.0 + i)
    raise RuntimeError("could not register %s after %d tries" % (name, tries))


def dp2_publish(anchor, name, blob, log):
    """payload PUT with NO name (no names.json write), then a verified 150 B pointer under `name`.
    Returns (payload sha, pointer sha)."""
    big = anchor.put(blob, None)
    ptr = json.dumps({"kind": "ptr", "name": name, "sha": big, "bytes": len(blob)}).encode("utf-8")
    return big, put_named_verified(anchor, ptr, name, log)


def dp2_fetch(anchor, name, timeout_s, poll_s, alive=None):
    """Resolve the pointer under `name`, then GET the payload it names. Returns (payload sha, blob)."""
    psha, pb = anchor.wait(name, timeout_s, poll_s, alive=alive)
    j = json.loads(pb.decode("utf-8"))
    if not isinstance(j, dict) or j.get("kind") != "ptr" or not j.get("sha"):
        raise RuntimeError("object under %s is not a DP-2 pointer" % name)
    return j["sha"], anchor.get(j["sha"])


# The garbage collector, run ON THE VPS in one short python invocation (read -> drop -> atomic
# os.replace of names.json; then unlink payloads no remaining name or pointer references). Deletes
# ONLY names under THIS run's namespace whose step < `below`; every other namespace is invisible to
# it. Ordering guarantee (see role_dp2_trainer): it runs on the 5090 after the 5090 has fetched the
# 4060's step-s gradient and before the 5090's own next named PUT; the 4060's next named PUT (its
# step-(s+1) pointer) waits for the `gc_s<s>` marker the 5090 publishes after the rewrite -- so no
# named PUT of this run overlaps the rewrite.
DP2_GC_SCRIPT = r'''
import json, os, re, sys, shutil
ROOT = %(root)r; NAMES = ROOT + "/names.json"; OBJ = ROOT + "/content"
ns, below = sys.argv[1], int(sys.argv[2])
pref = ns.rstrip("/") + "/"
pat = re.compile(r"^(grad_(?:5090|4060)|adapter|gc|final_(?:5090|4060))_s(\d{4})$")
def ptr_sha(sha):
    try:
        p = os.path.join(OBJ, sha)
        if os.path.getsize(p) < 4096:
            j = json.load(open(p))
            if isinstance(j, dict) and j.get("kind") == "ptr" and j.get("sha"):
                return j["sha"]
    except Exception:
        pass
    return None
d = json.load(open(NAMES))
drop, cand = [], set()
for name, ent in d.items():
    if not name.startswith(pref):
        continue
    m = pat.match(name[len(pref):])
    if not m or int(m.group(2)) >= below:
        continue
    drop.append(name)
    s = (ent or {}).get("sha256")
    if s:
        cand.add(s)
        b = ptr_sha(s)
        if b:
            cand.add(b)
for name in drop:
    d.pop(name, None)
if drop:
    tmp = NAMES + ".gc.tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, NAMES)
still = set((e or {}).get("sha256") for e in d.values() if isinstance(e, dict))
for name, ent in d.items():          # payloads behind any REMAINING DP-2 pointer, all namespaces
    if pat.match(name.rsplit("/", 1)[-1]):
        b = ptr_sha((ent or {}).get("sha256") or "")
        if b:
            still.add(b)
deleted, freed = 0, 0
for s in sorted(cand - still):
    p = os.path.join(OBJ, s)
    try:
        sz = os.path.getsize(p); os.remove(p); freed += sz; deleted += 1
    except FileNotFoundError:
        pass
st = shutil.disk_usage("/")
print("GCJSON " + json.dumps({"ns": ns, "below": below, "names_dropped": len(drop), "deleted": deleted,
                              "freed_mb": freed / 1e6, "free_gb": st.free / 1e9}))
'''


def anchor_free_gb(timeout=60):
    """`df --output=avail -B1 /` on the anchor over ssh (read-only). None when unreadable."""
    import subprocess
    try:
        r = subprocess.run(DP2["ssh"] + ["df --output=avail -B1 / | tail -1"], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=timeout, creationflags=0x08000000)
        return int((r.stdout or "").strip().splitlines()[-1].strip()) / 1e9
    except Exception:                                                    # noqa: BLE001
        return None


def run_anchor_gc(ns, below, log, timeout=120):
    """One GC pass over ssh. Never raises: a failed pass is reported ({'error': ...}) and the run
    continues -- the marker the caller publishes means 'the rewrite window is closed', not 'freed'."""
    import subprocess
    t0 = time.time()
    script = DP2_GC_SCRIPT % {"root": DP2["vps_store_root"]}
    try:
        r = subprocess.run(DP2["ssh"] + ["python3", "-", ns, str(int(below))], input=script,
                           capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
                           creationflags=0x08000000)
        out = None
        for line in (r.stdout or "").splitlines():
            if line.startswith("GCJSON "):
                out = json.loads(line[7:])
        if out is None:
            out = {"error": "rc=%s stdout=%s stderr=%s" % (r.returncode, (r.stdout or "")[-160:],
                                                            (r.stderr or "")[-160:])}
    except Exception as e:                                               # noqa: BLE001
        out = {"error": repr(e)[:200]}
    out["secs"] = time.time() - t0
    out["ns"] = ns
    out["below"] = int(below)
    log("ANCHOR_GC ns=%s deleted=%s freed_mb=%.1f free_gb=%s%s"
        % (ns, out.get("deleted", 0), float(out.get("freed_mb") or 0.0),
           ("%.2f" % out["free_gb"]) if out.get("free_gb") is not None else "?",
           (" ERROR=%s" % out["error"]) if out.get("error") else ""))
    return out


def dp2_deterministic_kernels(log):
    """The `--deterministic` block of role_trainer, verbatim in effect (2026-09-11 AMENDMENT 3):
    CUBLAS_WORKSPACE_CONFIG must be in the environment before CUDA initialises -- the caller sets
    it before `import torch`; this part configures torch after the import."""
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    log("DETERMINISTIC kernels: use_deterministic_algorithms=True cudnn.deterministic=True "
        "benchmark=False sdp=math-only (flash/mem_efficient/cudnn off) CUBLAS_WORKSPACE_CONFIG=%s"
        % os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def dp2_meta_problem(meta, mine):
    """The 4060's launch check against the 5090's init_meta: every field that defines 'the same
    experiment' must agree, or the 4060 VOIDs at step 0 (before any GPU hour is spent)."""
    keys = ("run", "data_sha256", "token_parity_sha", "selected_order_sha256", "epoch_order_sha256",
            "epoch_content_sha256", "seed", "arm", "trainable_total", "base_policy", "base_dtype", "rows_per_step",
            "offset", "micro", "total_micro", "epochs", "planned_steps", "steps_limit",
            "param_order_sha256", "wire_dtype", "resync_every", "ckpt_every", "deterministic",
            "fidelity_required", "clip")
    for k in keys:
        if meta.get(k) != mine.get(k):
            return "init_meta %s: 5090 %r != 4060 %r" % (k, meta.get(k), mine.get(k))
    if mine.get("fidelity_required") and meta.get("fidelity_passed") is not True:
        return "fidelity gate not passed on the 5090 (passed=%r) -- clause 1" % meta.get("fidelity_passed")
    return None


class Dp2PeerVoid(RuntimeError):
    pass


def role_dp2_trainer(a):
    """The DP-2 role, both boxes (--box 5090|4060). Reads the module comment above first."""
    if getattr(a, "deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # before the torch import
    import torch
    box, peer = a.box, ("4060" if a.box == "5090" else "5090")
    os.makedirs(a.out_dir, exist_ok=True)
    log = make_logger(os.path.join(a.out_dir, "trainer_%s.log" % box))
    log("=" * 78)
    log("DP-2 TRAINER | box %s | arm %s | seed %d | run %s | steps_limit %s | epochs %d | anchor_gc %s | "
        "wire %s | fidelity %s"
        % (box, a.arm, a.seed, a.run_id, a.steps_limit or "full", a.epochs,
           bool(a.anchor_gc) if box == "5090" else "n/a(4060)", a.wire_dtype,
           a.fidelity_json if (a.fidelity_json and a.fidelity_json.lower() != "none") else "EXEMPT (the gate run itself)"))
    if getattr(a, "deterministic", False):
        dp2_deterministic_kernels(log)
    else:
        log("WARNING: non-deterministic kernels -- a registered DP-2 arm runs det on both boxes")
    # base policy: DP-2 is registered on the nf4 base on BOTH boxes (prereg section 1)
    base_dtype = resolve_base_dtype(box, a.base_dtype, a.base_policy)
    if a.base_policy != "nf4" or base_dtype != "nf4":
        log("VOID: DP-2 is registered under --base-policy nf4 on both boxes; got policy %s / dtype %s"
            % (a.base_policy, base_dtype))
        return 3
    if int(a.epochs) != DP2["epochs"]:
        log("VOID: DP-2 is registered at %d epochs; got --epochs %s" % (DP2["epochs"], a.epochs))
        return 3
    token = read_token(log)
    if not token:
        log("ABORT: no content token -- gradients could not be published")
        return 2
    anchor = Anchor(a.anchor, token, log)
    ns = "wangrad/%s" % a.run_id
    # clause 1 at LAUNCH: the 5090 reads the artefact (it lives on the 5090's disk); the 4060 checks
    # the 5090's init_meta now and the 5090's step-31 header later -- it never reads the file.
    fid_required = bool(a.fidelity_json) and str(a.fidelity_json).lower() != "none"
    script_sha = sha256_file(os.path.abspath(__file__))       # F2: the file that IS training, by content
    fid_passed, fid_sha, fid_info = (dp2_fidelity_bound(a.fidelity_json, script_sha)
                                     if (fid_required and box == "5090") else (None, None, {}))
    if box == "5090" and fid_required and fid_passed is not True:
        log("VOID (clause 1): fidelity gate artefact %s passed=%r %s -- not launching"
            % (a.fidelity_json, fid_passed, json.dumps(fid_info)))
        return 3

    # -- corpus + the shared sequence
    data_sha = sha256_file(a.data)
    if a.expect_data_sha and not data_sha.startswith(a.expect_data_sha):
        log("VOID: corpus sha %s != registered %s" % (data_sha[:16], a.expect_data_sha[:16]))
        return 3
    rows0, sel = select_rows(a.data, SPEC["n_traces"], a.seed)
    seq, epoch_shas, epoch_content = epoch_sequence(rows0, a.seed, a.epochs)
    log("corpus %s sha %s | rows_in_file %d | selected %d | order_sha %s | epochs %d -> %d rows | epoch content shas %s"
        % (a.data, data_sha[:16], sel["rows_in_file"], sel["rows_selected"], sel["selected_order_sha256"][:16],
           a.epochs, len(seq), ",".join(s[:12] for s in epoch_content)))
    if len(set(epoch_content)) != len(epoch_content):
        log("VOID: two epochs share one content order (%s) -- the re-shuffle did not happen" % epoch_content)
        return 3
    total_steps = len(seq) // DP2["rows_per_step"]
    if total_steps != DP2_STEPS:
        log("VOID: %d rows / %d per step = %d steps != registered %d" % (len(seq), DP2["rows_per_step"], total_steps, DP2_STEPS))
        return 3
    stop_at = a.steps_limit or total_steps

    frac, total_vram = set_vram_cap(box, a.cap_gib, log)          # BEFORE the first CUDA call
    tok, model = load_base(box, a.model_dir, log, base_dtype)
    alloc_base = torch.cuda.memory_allocated()
    log("base loaded: %.3f GiB allocated" % gib(alloc_base))
    ok, layout = assert_expert_layout(model, log)
    if not ok:
        log("VOID (design 7a): expert layout on the %s is not fully nf4" % box)
        return 3
    model = attach_lora(model, a.seed, log)
    for k, p in lora_params(model).items():                        # all 16 layers trainable
        p.requires_grad_(True)
    params, pnames, porder_sha = dp2_param_order(model)
    n_tr = sum(p.numel() for p in params)
    if n_tr != SPEC["trainable_total"] or len(params) != SPEC["n_layers"] * TENSORS_PER_LAYER:
        raise SystemExit("SURFACE MISMATCH: %d params in %d tensors" % (n_tr, len(params)))
    log("surface: %d tensors, %d params, order sha %s" % (len(params), n_tr, porder_sha[:16]))
    tp_sha = token_parity_sha(tok, rows0)
    my_micro, my_off = DP2["micro"][box], DP2["offset"][box]
    share_expect = DP2["share"][box]

    init_meta = {"run": a.run_id, "box": box, "data_sha256": data_sha, "token_parity_sha": tp_sha,
                 "selected_order_sha256": sel["selected_order_sha256"], "epoch_order_sha256": epoch_shas,
                 "epoch_content_sha256": epoch_content,
                 "seed": a.seed, "arm": a.arm, "trainable_total": SPEC["trainable_total"],
                 "base_policy": a.base_policy, "base_dtype": base_dtype,
                 "rows_per_step": DP2["rows_per_step"], "offset": DP2["offset"], "micro": DP2["micro"],
                 "total_micro": DP2["total_micro"], "epochs": int(a.epochs), "planned_steps": total_steps,
                 "steps_limit": int(a.steps_limit or 0), "param_order_sha256": porder_sha,
                 "wire_dtype": a.wire_dtype, "resync_every": DP2["resync_every"], "ckpt_every": int(a.ckpt_every),
                 "deterministic": bool(getattr(a, "deterministic", False)),
                 "fidelity_required": fid_required, "fidelity_passed": fid_passed, "fidelity_sha256": fid_sha,
                 "script_sha256": script_sha,
                 "anchor_gc": bool(a.anchor_gc) if box == "5090" else None,
                 "clip": {"max_norm": SPEC["clip"], "norm": "float64-pairwise-cpu->fp32", "coef": "torch clamp(max/(n+1e-6),1)"},
                 "adamw": {"lr": SPEC["lr"], "betas": list(SPEC["betas"]), "weight_decay": 0.0, "eps": 1e-8,
                           "foreach": False, "fused": False}}
    if box == "5090":
        blob, n_t = slice_bytes(model, range(SPEC["n_layers"]), "fp32")
        sha = anchor.put(blob, ns + "/init")
        anchor.put(json.dumps(init_meta).encode("utf-8"), ns + "/init_meta")
        log("published init: %d tensors, %d B, sha %s" % (n_t, len(blob), sha[:12]))
        peer_gc = bool(a.anchor_gc)
    else:
        sha, meta_b = anchor.wait(ns + "/init_meta", a.peer_timeout_s, a.poll_s)
        meta = json.loads(meta_b.decode("utf-8"))
        why = dp2_meta_problem(meta, init_meta)
        if why:
            log("VOID: " + why)
            return 3
        sha, blob = anchor.wait(ns + "/init", a.peer_timeout_s, a.poll_s)
        m, n_t = load_slice(model, blob, range(SPEC["n_layers"]), log)
        log("loaded the 5090's init: %d of %d tensors (all 16 layers)" % (m, n_t))
        anchor.put(json.dumps(init_meta).encode("utf-8"), ns + "/init_meta_4060")
        peer_gc = bool(meta.get("anchor_gc"))

    stats = {"truncated": 0, "dropped": 0, "eos_restored": 0}
    batchify = make_batchify(tok, SPEC["seq_len"], stats)
    opt = torch.optim.AdamW(params, lr=SPEC["lr"], betas=SPEC["betas"], weight_decay=0.0,
                            foreach=False, fused=False)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=SPEC["lr"], total_steps=total_steps,
        pct_start=min(0.3, SPEC["warmup"] / max(total_steps, 1)), anneal_strategy="cos")
    log("%d optimizer steps planned (stopping at %d) | %d rows/step, this box %d (micro %d x %d) at offset %d"
        % (total_steps, stop_at, DP2["rows_per_step"], my_micro * SPEC["micro_batch"], my_micro,
           SPEC["micro_batch"], my_off))

    model.train()
    t0, step, seen = time.time(), 0, 0
    R = {"losses": [], "losses_own": [], "gnorms": [], "gnorm_global": [], "coefs": [], "clip_bound_steps": [],
         "compute_s": [], "put_s": [], "wait_s": [], "sync_s": [], "peer_sync_s": {}, "crit_sync": [],
         "gnorm_peer": [], "peer_seen": [], "resync": [], "repair_events": 0, "anchor_gc": [],
         "fidelity_checks": [], "none_grads": 0, "bytes_step": []}
    peer_rec = {"repair_events": 0}
    last_adapter_step = None
    peer_gc_after = False

    def publish_adapter(at_step):
        """5090: the fp32 LoRA after step `at_step` as ONE payload + pointer adapter_s<step>; the
        4060 compares/adopts it at step at_step+1 (RESYNC). Returns at_step."""
        tpa = time.time()
        flat = dp2_flatten(params, "data")
        ab = dp2_pack({"kind": "adapter", "step": at_step, "box": box, "numel": n_tr, "dtype": "fp32",
                       "param_order_sha256": porder_sha, "ns": ns, "run_id": a.run_id},
                      dp2_tensor_bytes(flat, "fp32"))
        dp2_publish(anchor, "%s/adapter_s%04d" % (ns, at_step), ab, log)
        log("  published fp32 adapter @%d (%d B, %.1f s) for the 4060's re-sync" % (at_step, len(ab), time.time() - tpa))
        return at_step

    if a.resume_from:
        step, i_row, seen, _acc = load_step_ckpt(a.resume_from, model, opt, sched, log)
        for k in R:
            if k in _acc:
                R[k] = _acc[k]
        peer_rec["repair_events"] = int(_acc.get("peer_repair_events", 0))
        if box == "5090" and step > 0 and step % DP2["resync_every"] == 0 and step < stop_at:
            # A resumed run must replay the original's re-sync at step+1 (the original's 5090 had
            # published adapter_s<step> after this checkpoint was saved), or the 4060's re-written
            # checkpoint cannot match its hidden original (AMENDMENT 5 certificate).
            last_adapter_step = publish_adapter(step)

    def peer_void_check():
        """alive() for the waits: False once the peer has published VOID_<peer> (checked every 30 s)."""
        now_t = time.time()
        if now_t - peer_void_check.t < 30:
            return True
        peer_void_check.t = now_t
        return anchor.resolve("%s/VOID_%s" % (ns, peer)) is None
    peer_void_check.t = 0.0

    def declare_void(problems):
        v = {"arm": a.arm, "box": box, "run_id": a.run_id, "step": step, "problems": problems,
             "at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "registered_by": "docs/research/PREREG_DP2_BEATS_ONE_4060_2026-09-12.md section 2"}
        with io.open(os.path.join(a.out_dir, "VOID_%s.json" % a.arm), "w", encoding="utf-8") as fh:
            json.dump(v, fh, indent=1)
        try:                                     # F6: verified registration, like the gradient pointers
            put_named_verified(anchor, json.dumps(v).encode("utf-8"), "%s/VOID_%s" % (ns, box), log)
        except Exception as e:                                           # noqa: BLE001
            log("  could not publish VOID_%s: %s" % (box, repr(e)[:120]))
        for p_ in problems:
            log("VOID: " + p_)

    void = None
    budget_stopped = False
    try:
        while step < stop_at and (time.time() - t0) / 3600.0 < a.max_hours:
            ts = time.time()
            t = step                                             # 0-based index of this step
            lo, hi = dp2_step_rows(t, box)
            epoch = lo // SPEC["n_traces"]
            opt.zero_grad(set_to_none=True)
            loss_sum, n_mb = 0.0, 0
            for j in range(my_micro):
                chunk = seq[lo + j * SPEC["micro_batch"]: lo + (j + 1) * SPEC["micro_batch"]]
                if not chunk:
                    break
                xb, lab = batchify(chunk)
                if xb is None:
                    continue
                out = model(input_ids=xb[0], attention_mask=xb[1], labels=lab)
                ce = out.loss
                ce.backward()                                    # UNscaled: .grad = the SUM
                loss_sum += float(ce.detach())
                n_mb += 1
                seen += len(chunk)
            R["none_grads"] += dp2_none_grads(params)
            own = dp2_flatten(params, "grad")
            own_gnorm = deterministic_norm(own)
            own_wire_b = dp2_tensor_bytes(own, a.wire_dtype)
            own_wire = dp2_bytes_tensor(own_wire_b, a.wire_dtype, n_tr)
            compute_s = time.time() - ts
            del own
            # clause 1 re-assert at step 31 (this header is step 31's); the 4060 checks the 5090's field
            fid_now = None
            if fid_required and box == "5090" and step + 1 == DP2["fidelity_gate_step"] + 1:
                fid_now, _s, _i = dp2_fidelity_bound(a.fidelity_json, script_sha)     # F2: passed AND bound
                R["fidelity_checks"].append({"step": step + 1, "passed": fid_now, "sha256": _s, "info": _i})
                if fid_now is not True:
                    declare_void(["VOID 1: fidelity artefact %s not passed/bound at step %d: %s"
                                  % (a.fidelity_json, step + 1, json.dumps(_i)[:300])])
                    void = "fidelity"
                    break
            gc_after = bool(box == "5090" and a.anchor_gc and (step + 1) % DP2["eval_every"] == 0
                            and (step + 1) < stop_at)
            prev = ({"step": step, "compute_s": R["compute_s"][-1], "put_s": R["put_s"][-1],
                     "wait_s": R["wait_s"][-1], "sync_s": R["sync_s"][-1]}
                    if R["compute_s"] and len(R["compute_s"]) == step else None)
            header = {"kind": "grad", "step": step + 1, "box": box, "epoch": epoch, "row_lo": lo, "row_hi": hi,
                      "n_micro": n_mb, "numel": n_tr, "dtype": a.wire_dtype, "loss_sum": loss_sum,
                      "gnorm_own": own_gnorm, "seen": seen, "prev": prev,
                      "repair_events": R["repair_events"], "last_adapter_step": last_adapter_step,
                      "fidelity_passed": fid_now, "gc_after_step": gc_after, "ns": ns, "run_id": a.run_id,
                      "param_order_sha256": porder_sha}
            blob = dp2_pack(header, own_wire_b)
            # -- exchange. The 4060 gates its named PUT on the 5090's gc marker when the 5090 said it
            #    would rewrite names.json after the previous step (no named PUT overlaps the rewrite).
            if box == "4060" and peer_gc_after:
                anchor.wait("%s/gc_s%04d" % (ns, step), a.peer_timeout_s, a.poll_s, alive=peer_void_check)
                peer_gc_after = False
            tp = time.time()
            my_name = "%s/grad_%s_s%04d" % (ns, box, step + 1)
            big_sha, _ = dp2_publish(anchor, my_name, blob, log)
            put_s = time.time() - tp
            tw = time.time()
            peer_name = "%s/grad_%s_s%04d" % (ns, peer, step + 1)
            try:
                psha, pblob = dp2_fetch(anchor, peer_name, a.peer_timeout_s, a.poll_s, alive=peer_void_check)
            except RuntimeError as e:
                if "died" in str(e) or anchor.resolve("%s/VOID_%s" % (ns, peer)) is not None:
                    raise Dp2PeerVoid("peer %s published VOID while step %d was pending" % (peer, step + 1))
                raise
            ph, ppay = dp2_unpack(pblob, expect={"kind": "grad", "step": step + 1, "box": peer, "numel": n_tr,
                                                 "dtype": a.wire_dtype, "param_order_sha256": porder_sha})
            peer_wire = dp2_bytes_tensor(ppay, a.wire_dtype, n_tr)
            wait_s = time.time() - tw
            R["compute_s"].append(compute_s)
            R["put_s"].append(put_s)
            R["wait_s"].append(wait_s)
            R["sync_s"].append(put_s + wait_s)
            R["gnorm_peer"].append(float(ph.get("gnorm_own") or 0.0))
            R["peer_seen"].append(int(ph.get("seen") or 0))
            R["bytes_step"].append(len(blob) + len(pblob))
            pv = ph.get("prev")
            if isinstance(pv, dict) and pv.get("step") == step and step >= 1:
                R["peer_sync_s"][str(step)] = pv["sync_s"]
                R["crit_sync"].append(min(R["sync_s"][step - 1], float(pv["sync_s"])))
            if box == "5090":
                peer_rec["repair_events"] = int(ph.get("repair_events") or 0)
            else:
                peer_gc_after = bool(ph.get("gc_after_step"))
                if fid_required and (step + 1) == DP2["fidelity_gate_step"] + 1 and ph.get("fidelity_passed") is not True:
                    declare_void(["VOID 1: the 5090 reports fidelity passed=%r at step %d" % (ph.get("fidelity_passed"), step + 1)])
                    void = "fidelity"
                    break
                # -- 25-step re-sync: the 5090 published its fp32 adapter after step s; compare + adopt
                las = ph.get("last_adapter_step")
                if las is not None and las == step and not any(r["step"] == las for r in R["resync"]):
                    asha, ablob = dp2_fetch(anchor, "%s/adapter_s%04d" % (ns, las), a.peer_timeout_s, a.poll_s,
                                            alive=peer_void_check)
                    ah, apay = dp2_unpack(ablob, expect={"kind": "adapter", "step": las, "box": "5090",
                                                         "numel": n_tr, "dtype": "fp32", "param_order_sha256": porder_sha})
                    theirs = dp2_bytes_tensor(apay, "fp32", n_tr)
                    mine = dp2_flatten(params, "data").cpu()
                    identical = bool(torch.equal(mine, theirs))
                    mad = float((mine - theirs).abs().max()) if not identical else 0.0
                    adopted = False
                    if not identical:
                        dp2_scatter_(params, theirs.to(params[0].device), "data")
                        R["repair_events"] += 1
                        adopted = True
                    R["resync"].append({"step": las, "identical": identical, "max_abs_diff": mad,
                                        "adopted": adopted, "sha": asha})
                    log("RESYNC step=%d identical=%s max_abs_diff=%.3e%s" % (
                        las, identical, mad, " ADOPTED the 5090's copy (repair_events=%d)" % R["repair_events"]
                        if adopted else ""))
            # -- the identical averaged gradient, the twin's clip, the plain AdamW step
            g5, g4 = (own_wire, peer_wire) if box == "5090" else (peer_wire, own_wire)
            avg_cpu = dp2_average(g5, g4, DP2["total_micro"])        # identical bits on both boxes
            gn = deterministic_norm(avg_cpu)                          # identical scalar on both boxes
            avg = avg_cpu.to(params[0].device)
            dp2_scatter_(params, avg, "grad")
            coef = dp2_clip_(params, SPEC["clip"], gn)
            if gn > SPEC["clip"]:
                R["clip_bound_steps"].append(step + 1)
            opt.step()
            if step < total_steps:
                sched.step()
            step += 1
            step_loss = (loss_sum + float(ph.get("loss_sum") or 0.0)) / DP2["total_micro"]
            R["losses"].append(step_loss)
            R["losses_own"].append(loss_sum / max(n_mb, 1))
            R["gnorms"].append(own_gnorm)
            R["gnorm_global"].append(gn)
            R["coefs"].append(coef)
            del avg, avg_cpu, own_wire, peer_wire
            if step % 25 == 0 or step == 1:
                cs = R["crit_sync"]
                log("  step %4d | loss %.4f (own %.4f) | %d own rows, peer %d | %.2f h | lr %.2e | gnorm own %.3f "
                    "peer %.3f | global %.3f | coef %.4f | compute %.1f s | put %.1f | wait %.1f | crit mean %s"
                    % (step, step_loss, R["losses_own"][-1], seen, R["peer_seen"][-1], (time.time() - t0) / 3600.0,
                       sched.get_last_lr()[0], own_gnorm, R["gnorm_peer"][-1], gn, coef, compute_s, put_s, wait_s,
                       ("%.2f" % (sum(cs) / len(cs))) if cs else "-"))
            # -- every 25 steps, in this order: VOID evaluation (both boxes, same data -> same
            #    outcome; a VOID publish is a named PUT and must precede any rewrite), then the
            #    names.json GC (5090), then the checkpoint, then the adapter publish (5090).
            if step % DP2["eval_every"] == 0 and step < stop_at:
                rec = {"crit_sync": R["crit_sync"],
                       "share_own": (seen / float(seen + R["peer_seen"][-1])) if (seen + R["peer_seen"][-1]) else None,
                       "share_expect": share_expect, "gnorm_own": R["gnorms"], "gnorm_peer": R["gnorm_peer"],
                       "repair_events": R["repair_events"] if box == "4060" else peer_rec["repair_events"],
                       "fidelity_required": fid_required,
                       "fidelity_passed": (fid_passed if not R["fidelity_checks"] else R["fidelity_checks"][-1]["passed"])
                       if box == "5090" else True,
                       "window": DP2["eval_every"], "step": step, "planned_steps": total_steps}
                # AMENDMENT 2 boundary line: the projected run mean P and the outage bookkeeping the
                # early stop (dp2_void_problems, clause 2 (ii)) is about to decide on -- same data,
                # same numbers on both boxes
                so = dp2_sync_stats(R["crit_sync"], total_steps)
                log("  sync @%d (AMENDMENT 2): B %s s | P %s s | outage steps %d in %d episodes, %.0f s (> %.0f s) "
                    "| run mean %s s over %d of %d | fires iff P > %.1f or B > %.1f"
                    % (step, ("%.2f" % so["B"]) if so["B"] is not None else "n/a",
                       ("%.2f" % so["P"]) if so["P"] is not None else "n/a", so["outage_steps"], so["outage_episodes"],
                       so["outage_seconds"], so["outage_threshold_s"],
                       ("%.2f" % so["mean_s"]) if so["mean_s"] is not None else "n/a", so["n"], so["planned_steps"],
                       DP2["sync_void_mean_s"], DP2["sync_void_mean_s"]))
                probs = dp2_void_problems(rec)
                if probs:
                    declare_void(probs)
                    void = "; ".join(probs)
                    break
                if gc_after:
                    g = run_anchor_gc(ns, step - DP2["gc_keep_steps"], log)
                    g["step"] = step
                    R["anchor_gc"].append(g)
                    # the marker closes the rewrite window; the 4060's next named PUT waits for it
                    put_named_verified(anchor, json.dumps(g).encode("utf-8"), "%s/gc_s%04d" % (ns, step), log)
                if a.ckpt_every and step % a.ckpt_every == 0:
                    try:
                        ckpt_path = os.path.join(a.out_dir, "step_ckpt_%s_s%04d.pt" % (box, step))
                        acc = dict(R)
                        acc["peer_repair_events"] = peer_rec["repair_events"]
                        save_step_ckpt(ckpt_path, model, opt, sched, step, step * DP2["rows_per_step"], seen, acc, log)
                        _mine = sorted(f for f in os.listdir(a.out_dir)
                                       if re.match(r"^step_ckpt_%s_s\d{4}\.pt$" % box, f))
                        for _old in _mine[:-2]:
                            os.remove(os.path.join(a.out_dir, _old))
                    except Exception as _e:                              # noqa: BLE001
                        log("  ckpt @%d FAILED (run continues): %s" % (step, repr(_e)[:160]))
                if box == "5090" and step % DP2["resync_every"] == 0:
                    last_adapter_step = publish_adapter(step)
        budget_stopped = bool(void is None and step < stop_at)
    except Dp2PeerVoid as e:
        log("VOID (peer): %s" % e)
        void = "peer: %s" % e
    el = (time.time() - t0) / 3600.0

    # -- terminal: both boxes publish their fp32 LoRA; the 5090 proves the two copies identical
    final_identical, final_mad, adapter_out = None, None, None
    if void is None and not budget_stopped:
        flat = dp2_flatten(params, "data")
        fb = dp2_pack({"kind": "final", "step": step, "box": box, "numel": n_tr, "dtype": "fp32",
                       "param_order_sha256": porder_sha, "ns": ns, "run_id": a.run_id}, dp2_tensor_bytes(flat, "fp32"))
        fsha, _ = dp2_publish(anchor, "%s/final_%s_s%04d" % (ns, box, step), fb, log)
        log("published FINAL fp32 LoRA: %d B, sha %s" % (len(fb), fsha[:12]))
        adapter_out = os.path.join(a.out_dir, "adapter")
        model.save_pretrained(adapter_out)
        log("saved the 16-layer adapter -> %s" % adapter_out)
        if box == "5090":
            psha, pb = dp2_fetch(anchor, "%s/final_4060_s%04d" % (ns, step), max(a.peer_timeout_s, 1800.0), a.poll_s)
            fh_, fpay = dp2_unpack(pb, expect={"kind": "final", "step": step, "box": "4060", "numel": n_tr,
                                               "dtype": "fp32", "param_order_sha256": porder_sha})
            theirs = dp2_bytes_tensor(fpay, "fp32", n_tr)
            mine = flat.cpu()
            final_identical = bool(torch.equal(mine, theirs))
            final_mad = float((mine - theirs).abs().max()) if not final_identical else 0.0
            log("FINAL bit-exact vs the 4060: identical=%s max_abs_diff=%.3e" % (final_identical, final_mad))
        del flat
    vram = {"peak_alloc_gib": gib(torch.cuda.max_memory_allocated()),
            "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
            "device_total_gib": gib(torch.cuda.get_device_properties(0).total_memory),
            "base_alloc_gib": gib(alloc_base), "cap_fraction": frac}
    cs = R["crit_sync"]
    so = dp2_sync_stats(cs, total_steps)          # AMENDMENT 2: outage bookkeeping, both boxes identical
    peer_seen_last = R["peer_seen"][-1] if R["peer_seen"] else 0
    manifest = {"role": "dp2-trainer", "box": box, "arm": a.arm, "seed": a.seed, "run_id": a.run_id,
                "registered_by": "docs/research/PREREG_DP2_BEATS_ONE_4060_2026-09-12.md",
                "deterministic": bool(getattr(a, "deterministic", False)),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "bitsandbytes": _bnb_version(), "torch": torch.__version__,
                "epochs": int(a.epochs), "epoch_order_sha256": epoch_shas, "epoch_content_sha256": epoch_content,
                "epoch_order_key_degenerate": all(r.get("id") is None for r in rows0),
                "rows_per_step": DP2["rows_per_step"],
                "row_share": DP2["share"], "row_share_expect_own": share_expect, "offset": my_off,
                "micro_own": my_micro, "micro_batch": SPEC["micro_batch"], "total_micro": DP2["total_micro"],
                "steps": step, "planned_steps": total_steps, "steps_limit": a.steps_limit,
                "traces_seen": seen, "peer_traces_seen": peer_seen_last,
                "rows_consumed_total": step * DP2["rows_per_step"],
                "realised_share_own": (seen / float(seen + peer_seen_last)) if (seen + peer_seen_last) else None,
                "hours": el, "losses": R["losses"], "losses_own": R["losses_own"],
                "gnorms": R["gnorms"], "gnorm_max": max(R["gnorms"]) if R["gnorms"] else None,
                "gnorm_peer": R["gnorm_peer"], "gnorm_peer_max": max(R["gnorm_peer"]) if R["gnorm_peer"] else None,
                "gnorm_global": R["gnorm_global"], "gnorm_global_max": max(R["gnorm_global"]) if R["gnorm_global"] else None,
                "clip_coef": R["coefs"], "clip_coef_min": min(R["coefs"]) if R["coefs"] else None,
                "clip_bound_steps": len(R["clip_bound_steps"]), "clip_bound_step_ids": R["clip_bound_steps"],
                "clip": {"max_norm": SPEC["clip"], "norm": "float64 pairwise on CPU -> fp32 (deterministic_norm)",
                         "coef": "torch clamp(max_norm/(norm+1e-6), max=1) fp32", "gnorms_key_is": "own partial-sum norm per step"},
                "adamw": init_meta["adamw"], "wire_dtype": a.wire_dtype, "wire_numel": n_tr,
                "wire_bytes_per_step": (R["bytes_step"][-1] if R["bytes_step"] else None),
                "step_seconds": R["compute_s"], "put_seconds": R["put_s"], "wait_seconds": R["wait_s"],
                "sync_seconds": R["sync_s"], "peer_sync_seconds": R["peer_sync_s"],
                "critical_sync_seconds": cs, "critical_sync_mean": (sum(cs) / len(cs)) if cs else None,
                "outage_steps": so["outage_steps"], "outage_episodes": so["outage_episodes"],
                "outage_seconds": so["outage_seconds"], "sync_outage": so,      # AMENDMENT 2 (B, P, threshold)
                "step_seconds_median": sorted(R["compute_s"])[len(R["compute_s"]) // 2] if R["compute_s"] else None,
                "sync_log": [{"step": k + 1, "put_s": R["put_s"][k], "wait_s": R["wait_s"][k], "sync_s": R["sync_s"][k],
                              "bytes": R["bytes_step"][k]} for k in range(len(R["sync_s"]))],
                "resync": R["resync"], "repair_events": R["repair_events"] if box == "4060" else peer_rec["repair_events"],
                "anchor_gc": R["anchor_gc"], "anchor_gc_enabled": bool(a.anchor_gc) if box == "5090" else peer_gc,
                "fidelity": {"path": a.fidelity_json if fid_required else None, "required": fid_required, "passed_at_launch": fid_passed,
                             "sha256": fid_sha, "checks": R["fidelity_checks"]},
                "script_sha256": script_sha,                # F2: the gate binds FIDELITY.json to this
                "void": void, "none_grads": R["none_grads"],
                "final_identical_vs_peer": final_identical, "final_max_abs_diff": final_mad,
                "seq_stats": stats, "seq_len": SPEC["seq_len"], "model": a.model_dir, "data": a.data,
                "data_sha256": data_sha, "token_parity_sha": tp_sha,
                "selected_ids_sha256": sel["selected_ids_sha256"], "selected_order_sha256": sel["selected_order_sha256"],
                "rows_in_file": sel["rows_in_file"], "rows_selected": sel["rows_selected"],
                "lr": SPEC["lr"], "rank": SPEC["rank"], "warmup": SPEC["warmup"], "surface": "attention-only",
                "trainable_total": SPEC["trainable_total"], "param_order_sha256": porder_sha,
                "adapter": adapter_out, "vram": vram, "base_dtype": base_dtype, "base_policy": a.base_policy,
                "expert_layout": layout, "anchor": anchor.stats(), "budget_stopped": budget_stopped,
                "argv": list(sys.argv), "final_checkpoint_not_best": True}
    try:
        import transformers, peft
        manifest["transformers"] = transformers.__version__
        manifest["peft"] = peft.__version__
    except Exception:                                                    # noqa: BLE001
        pass
    path = os.path.join(a.out_dir, "TRAIN_%s.json" % box)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    anchor.put(json.dumps(manifest).encode("utf-8"), ns + "/%s/train_json" % box)
    log("wrote %s (+ published)" % path)
    if void is not None:
        log("==== TERMINAL: VOID -- %s ====" % void[:300])
        return 3
    if stats["truncated"] or stats["dropped"]:
        log("==== TERMINAL: VOID -- seq_stats %s (zero-truncation rule) ====" % json.dumps(stats))
        return 3
    if budget_stopped:
        log("==== TERMINAL: BUDGET-STOPPED at step %d of %d -- NOT COMPLETE ====" % (step, stop_at))
        return 4
    log("==== TERMINAL: DP-2 TRAIN COMPLETE (%s, %d steps) ====" % (box, step))
    return 0



# ---- the DP-2 verdict as a pure function of artefact paths (chain method dp2_verdict wraps it;
#      dp2_selftest.py drives it on the banked h1_s2 artefacts vs the nf4 twin and must reproduce
#      VERDICT_h1_s2.json's own paired numbers)
def dp2_control_files(d):
    """(S1, S1EXT, BATTERY) of a control scored by score_arm's convention in dir `d`."""
    ctag = os.path.basename(d)
    seed = SPEC["seeds"].get(ctag.rsplit("_", 1)[-1])
    ext = twin_ext_path(seed, d) if seed is not None else os.path.join(d, "S1EXT_%s.json" % ctag)
    return d + "/S1_%s.json" % ctag, ext, d + "/BATTERY_%s.json" % ctag


def dp2_pooled(paired_read, guard, ext):
    a, _ = paired_read.per_item(json.load(io.open(guard, encoding="utf-8")))
    b, _ = paired_read.per_item(json.load(io.open(ext, encoding="utf-8")))
    m = dict(a)
    m.update(b)
    return m


def dp2_paired(paired_read, P, cdir):
    """{gsm1319, battery} paired reads of the arm's P against the control in `cdir`; None per read
    when a control artefact is missing."""
    s1, ext, bat = dp2_control_files(cdir)
    out = {"control_dir": cdir}
    if all(os.path.exists(x) for x in (s1, ext, P["gsm"], P["gsmext"])):
        mine, twin = dp2_pooled(paired_read, P["gsm"], P["gsmext"]), dp2_pooled(paired_read, s1, ext)
        ids = set(twin) & set(mine)
        m = paired_read.mcnemar({k: twin[k] for k in ids}, {k: mine[k] for k in ids})
        m.pop("ids", None)
        out["gsm1319"] = m
    else:
        out["gsm1319"] = None
    if os.path.exists(bat) and os.path.exists(P["battery"]):
        tb, _ = paired_read.per_item(json.load(io.open(bat, encoding="utf-8")))
        wb, _ = paired_read.per_item(json.load(io.open(P["battery"], encoding="utf-8")))
        m = paired_read.mcnemar(tb, wb)
        m.pop("ids", None)
        out["battery"] = m
    else:
        out["battery"] = None
    return out


def dp2_wallclock_label(h_dp2, h_twin, tol=1.10):
    """AMENDMENT 2 (fable decision 2 Q3, 2026-09-12): realised wall-clock as a SECONDARY, never a
    gate. R_wc = dp2 training hours (TRAIN_5090.json hours) / the twin4060's TRAIN.json hours (read
    from the control dir, never hard-coded). R_wc <= 1.10 keeps the prereg section-5 label "at equal
    wall-clock"; above it a PASS is labelled "2 epochs at R_wc x the 4060's wall-clock" and the
    equal-wall-clock claim is NOT made. The accuracy verdict is untouched either way. Pure."""
    try:
        r = float(h_dp2) / float(h_twin)
    except (TypeError, ValueError, ZeroDivisionError):
        return {"wallclock_ratio_vs_twin4060": None, "hours_dp2": h_dp2, "hours_twin4060": h_twin,
                "wallclock_label": "unknown (hours missing)", "equal_wallclock_claim": None, "wallclock_tol": tol}
    eq = bool(r <= tol)
    return {"wallclock_ratio_vs_twin4060": r, "hours_dp2": float(h_dp2), "hours_twin4060": float(h_twin),
            "wallclock_label": "equal wall-clock" if eq else "2 epochs at %.3fx the 4060's wall-clock" % r,
            "equal_wallclock_claim": eq, "wallclock_tol": tol}


def dp2_wallclock_secondary(h_dp2, pdir):
    """dp2_wallclock_label against the control dir's TRAIN.json (twin4060_s1: 8.6286 h on disk)."""
    h_twin = None
    try:
        h_twin = json.load(io.open(os.path.join(pdir, "TRAIN.json"), encoding="utf-8")).get("hours")
    except Exception:                                                    # noqa: BLE001
        pass
    return dp2_wallclock_label(h_dp2, h_twin)


def dp2_build_verdict(P, stag, seed, pdir, t5, t4, paired_read, base_battery, base_arc, log, tag="dp2"):
    """Prereg section 2, one seed: PRIMARY dp2 vs twin4060 (same seed) from `pdir`, SECONDARY vs
    nf4_twin and vs twin2ep (reported, never substituted), ARC-Easy veto, vs-base reads for context.
    Returns the verdict dict; verdict == "PENDING_CONTROL" when the primary control is not scored."""
    v = {"arm": tag, "kind": "dp2", "seed": seed,
         "registered_by": "docs/research/PREREG_DP2_BEATS_ONE_4060_2026-09-12.md",
         "primary_control": pdir, "base_policy": "nf4",
         "wire": {"5090": (t5 or {}).get("anchor"), "4060": (t4 or {}).get("anchor")},
         "critical_sync_mean": (t5 or {}).get("critical_sync_mean"),
         # AMENDMENT 2: outage bookkeeping (both boxes identical), the 5090 step-time regime, R_wc
         "outage_steps": (t5 or {}).get("outage_steps"), "outage_episodes": (t5 or {}).get("outage_episodes"),
         "outage_seconds": (t5 or {}).get("outage_seconds"),
         "step_seconds_median_5090": (t5 or {}).get("step_seconds_median"),
         "repair_events": (t4 or {}).get("repair_events"), "resync": (t4 or {}).get("resync"),
         "final_identical_vs_peer": (t5 or {}).get("final_identical_vs_peer"),
         "realised_share": {"5090": (t5 or {}).get("realised_share_own"), "4060": (t4 or {}).get("realised_share_own")},
         "hours": {"5090": (t5 or {}).get("hours"), "4060": (t4 or {}).get("hours")},
         "anchor_gc": ((t5 or {}).get("anchor_gc") or [])[-1:]}
    v.update(dp2_wallclock_secondary((t5 or {}).get("hours"), pdir))    # AMENDMENT 2 Q3: R_wc + label
    try:
        sj = json.load(io.open(P["gsm"], encoding="utf-8"))
        v["versions"] = sj.get("versions")
    except Exception:                                                    # noqa: BLE001
        v["versions"] = None
    prim = dp2_paired(paired_read, P, pdir)
    reads = {}
    if prim.get("gsm1319"):
        reads["gsm1319_W_minus_C"] = prim["gsm1319"]
    if prim.get("battery"):
        reads["battery_W_minus_C"] = prim["battery"]
    v["reads"] = reads
    sec = {}
    for name, d in (("nf4_twin", SPEC["twin_nf4"].get(seed)), ("twin2ep", DP2["twin2ep"].get(stag))):
        if d and os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(pdir):
            sec[name] = dp2_paired(paired_read, P, d)
    # the second-epoch effect on ONE box (twin2ep vs nf4_twin): the context the primary needs
    t2d, n4d = DP2["twin2ep"].get(stag), SPEC["twin_nf4"].get(seed)
    if t2d and n4d and os.path.isdir(t2d) and os.path.isdir(n4d):
        s1a, exta, _ = dp2_control_files(t2d)
        s1b, extb, _ = dp2_control_files(n4d)
        if all(os.path.exists(x) for x in (s1a, exta, s1b, extb)):
            A, B = dp2_pooled(paired_read, s1a, exta), dp2_pooled(paired_read, s1b, extb)
            ids = set(A) & set(B)
            m = paired_read.mcnemar({k: B[k] for k in ids}, {k: A[k] for k in ids})
            m.pop("ids", None)
            sec["twin2ep_minus_nf4_twin"] = {"gsm1319": m}
    v["secondary"] = sec
    if P.get("ok_battery") and os.path.exists(P["battery"]):
        c = paired_read.compare(base_battery, P["battery"], 0.75)
        c.pop("gained_ids", None)
        v["battery_vs_base"] = c
        v["ex_boolq_vs_base"] = c.get("ex_boolq")
        v["constituents"] = c.get("per_task")
    if P.get("ok_gsm") and P.get("ok_gsmext") and os.path.exists(P["gsm"]) and os.path.exists(P["gsmext"]):
        mine = dp2_pooled(paired_read, P["gsm"], P["gsmext"])
        base = dp2_pooled(paired_read, SPEC["base_gsm512"], SPEC["base_gsmext"])
        ids = set(base) & set(mine)
        b = paired_read.mcnemar({k: base[k] for k in ids}, {k: mine[k] for k in ids})
        b.pop("ids", None)
        v["gsm1319_vs_base"] = b
    vetoes = []
    if P.get("ok_arc") and os.path.exists(P["arc"]):
        c = paired_read.compare(base_arc, P["arc"], 1.5)
        c.pop("gained_ids", None)
        v["arc_vs_base"] = c
        if c["delta"] < 0 and c["p"] < DP2["alpha"]:
            vetoes.append("arc_easy %+.4f pp p=%.3g" % (c["delta"], c["p"]))
    missing = [k for k in ("battery", "gsm", "gsmext", "arc") if not P.get("ok_" + k)]
    v["artefacts_rejected"] = missing
    if vetoes:
        v["named_vetoes"] = vetoes
    if missing:
        v["verdict"] = "INCOMPLETE"
    elif not reads.get("gsm1319_W_minus_C"):
        v["verdict"] = "PENDING_CONTROL"
        v["pending_why"] = ("primary control %s has no S1/S1EXT (score it with --score-only %s), then re-run "
                            "this chain: training and scoring artefacts are skipped onto" % (pdir, os.path.basename(pdir)))
        return v
    else:
        v["verdict"] = dp2_single_seed_verdict(reads["gsm1319_W_minus_C"], vetoes)
        if vetoes:
            v["verdict_before_veto"] = dp2_single_seed_verdict(reads["gsm1319_W_minus_C"], [])
    m = reads.get("gsm1319_W_minus_C")
    if m:
        t2 = (sec.get("twin2ep") or {}).get("gsm1319")
        v["prediction_check"] = {"registered": "dp2 vs twin4060 +1..+3 pp; vs twin2ep parity",
                                 "vs_twin4060_in_band": bool(1.0 <= m["delta"] <= 3.0),
                                 "vs_twin2ep_parity": (abs(t2["delta"]) < t2["mde"]) if t2 else None}
        log("  %s gsm1319 vs %s: W-C %+.4f pp CI[%+.3f,%+.3f] p=%.3g D=%d MDE %.4f -> %s"
            % (tag, os.path.basename(pdir), m["delta"], m["ci_lo"], m["ci_hi"], m["p"], m["disc"], m["mde"], v["verdict"]))
    for name, r in sec.items():
        g = r.get("gsm1319")
        if g:
            log("  %s SECONDARY gsm1319 vs %s: %+.4f pp p=%.3g MDE %.4f" % (tag, name, g["delta"], g["p"], g["mde"]))
    log("  %s wall-clock SECONDARY (AMENDMENT 2): R_wc %s -> %s | equal_wallclock_claim %r | 5090 step_seconds median %s "
        "| outage steps %r in %r episodes, %r s"
        % (tag, ("%.4f" % v["wallclock_ratio_vs_twin4060"]) if v.get("wallclock_ratio_vs_twin4060") is not None else "n/a",
           v.get("wallclock_label"), v.get("equal_wallclock_claim"), v.get("step_seconds_median_5090"),
           v.get("outage_steps"), v.get("outage_episodes"), v.get("outage_seconds")))
    v["kill_s2"] = dp2_kill_reason(v)
    return v


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--role", default="chain",
                    choices=("chain", "trainer", "gate-footprint", "gate-basece", "stage-4060", "env-probe",
                             "dp2-trainer"))
    # shared
    ap.add_argument("--box", default="5090", choices=("5090", "4060"))
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--anchor", default=SPEC["anchor"])
    ap.add_argument("--cap-gib", type=float, default=6.5, help="4060 VRAM cap (design section 2)")
    # base policy (owner directive 2026-09-08): dtype is decoupled from box identity
    ap.add_argument("--base-dtype", default="auto", choices=("auto", "bf16", "nf4"),
                    help="the frozen base THIS box loads; auto = what --base-policy prescribes for the "
                         "box (default policy mixed: bf16 on the 5090, nf4 on the 4060 -- byte-unchanged)")
    ap.add_argument("--base-policy", default="mixed", choices=tuple(BASE_POLICIES),
                    help="mixed = the design as registered (5090 bf16, 4060 nf4); nf4 = both boxes on the "
                         "same NF4 base (owner 2026-09-08). Carried in init_meta; the 4060 VOIDs on a "
                         "mismatch; nf4 needs --py5090 (transformers 4.x) and the nf4 control twin")
    # trainer
    ap.add_argument("--arm", default="h25", choices=("h1", "h25", "indep", "dp2"))
    ap.add_argument("--H", type=int, default=25, help="sync period in steps; 0 = H-infinity")
    ap.add_argument("--seed", type=int, default=SPEC["seeds"]["s1"])
    ap.add_argument("--steps-limit", type=int, default=0, help="0 = the full 625")
    ap.add_argument("--expect-data-sha", default="")
    ap.add_argument("--wire-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--clip-mode", default="global", choices=("global", "local"),
                    help="global = d1_sft's 16-layer clip reproduced on each box from its own view "
                         "(review 2026-09-08 HIGH-2; the default, and the recipe); local = per-slice "
                         "clip, the pre-fix behaviour, kept only as an explicit escape hatch")
    ap.add_argument("--peer-timeout-s", type=float, default=2700.0)
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--max-hours", type=float, default=4.0)
    # 2026-09-09: the box hung at 16:02 and an arm died at step 50 of 625, losing the whole
    # run because the trainer only saved at the end. At H=1 both boxes are in lockstep, so a
    # checkpoint at a fixed interval is consistent across them and a joint resume is coherent.
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="checkpoint every N steps (0 = off); caps a crash at N steps")
    ap.add_argument("--cert-resume", action="store_true",
                    help="chain: certifiable resumes (AMENDMENT 4/5) -- step back one interval, hide "
                         "s(k) on both boxes as .cert_ref for resume_cert_watcher.py; restart from 0 "
                         "when no certifiable point exists. Off = the 2026-09-10 resume behaviour.")
    ap.add_argument("--deterministic-boxes", default="",
                    help="chain: comma list of boxes whose trainer gets --deterministic (e.g. 5090). "
                         "Empty = registered behaviour, nothing changes.")
    ap.add_argument("--peer-ns", default="",
                    help="trainer role, DIAGNOSTIC ONLY: read the peer's init and step slices from "
                         "this namespace instead of --run-id (a recorded 'ghost peer'); all writes "
                         "still go to --run-id. Empty = normal behaviour.")
    ap.add_argument("--deterministic", action="store_true",
                    help="trainer role only: deterministic kernels (see role_trainer). OFF by "
                         "default; a registered run must pass it explicitly via the chain.")
    ap.add_argument("--resume-from", default="",
                    help="path to a step checkpoint written by --ckpt-every")
    # gates
    ap.add_argument("--n-steps", type=int, default=8)
    ap.add_argument("--cliff-max", type=float, default=1.5)
    ap.add_argument("--probe-n", type=int, default=128)
    ap.add_argument("--corpus-sha", default="")
    ap.add_argument("--no-download", action="store_true")
    # chain
    ap.add_argument("--run-dir", default=SPEC["run_root"])
    ap.add_argument("--deadline", default="2026-09-12 23:00")
    ap.add_argument("--plan", action="store_true", help="print every step; launch nothing")
    ap.add_argument("--only", default="", help="comma list of phases: aa,stage,footprint,basece,track,arms")
    ap.add_argument("--with-c2", action="store_true", help="re-run the banked Instruct positive control")
    ap.add_argument("--need-ram-gb", type=float, default=12.0)
    ap.add_argument("--idle-timeout-s", type=int, default=6 * 3600)
    ap.add_argument("--util-max-pct", type=int, default=25)
    ap.add_argument("--since", default="")
    ap.add_argument("--skip-aa", action="store_true")
    ap.add_argument("--cap-4060-gib", type=float, default=6.5)
    ap.add_argument("--py4060", default=r"D:\wangrad\venv\Scripts\python.exe",
                    help="the 4060 interpreter for every remote role; MUST carry transformers 4.x "
                         "(probe 2026-09-08: glmenv and Python313 there are 5.8.1, unusable)")
    ap.add_argument("--py5090", default="",
                    help="the 5090 interpreter for its GPU roles under --base-policy nf4; MUST carry "
                         "transformers 4.x with a CUDA torch + bitsandbytes + peft (C:/Python313 is "
                         "5.8.1: fused experts, cannot nf4 -- probed 2026-09-08). Recipe: TF4_ENV_RECIPE. "
                         "Ignored under policy mixed (the 5090 trains bf16 under C:/Python313 as before)")
    ap.add_argument("--max-4060-h-per-arm", type=float, default=3.0)
    # Owner decision 2026-09-09, after P3_FOOTPRINT measured the 4060 at 40.401 s/step
    # = 7.014 h/arm (design assumed 1.4): buy h1_s1 ONLY (~7 h) and let its section-5 KILL
    # read license or kill the remaining ~21 h. Without this flag that decision is prose
    # and the runner spends the other three arms anyway -- the 2026-09-06 lesson that a
    # stop condition is not registered until the code that spends the budget can execute it.
    ap.add_argument("--stop-after-h1", action="store_true",
                    help="run the h1_s1 fidelity anchor, then stop before ARMS[1:]")
    # 2026-09-09 overnight ruling: run a NAMED subset of ARMS. When h1_s1 is excluded its verdict
    # must already exist, so the section-5 KILL read always precedes the arms it licenses. This
    # flag IS the executable stop condition for an overnight launch: the P7 loop skips every arm
    # not named and the chain exits after the last named arm's verdict.
    ap.add_argument("--arms", default="",
                    help="comma list of arm tags to run (subset of h1_s1,h1_s2,h25_s1,h25_s2,indep_s1); "
                         "empty = all. h1_s1 is always evaluated first (SKIP when its verdict exists)")
    ap.add_argument("--stage-timeout-h", type=float, default=6.0)
    ap.add_argument("--basece-gap-max-nats", type=float, default=0.0, help="0 = 25%% of the twin's CE drop")
    ap.add_argument("--track-steps", type=int, default=30)
    ap.add_argument("--track-band-max", type=float, default=0.05)
    ap.add_argument("--track-step1-tol", type=float, default=0.002)
    ap.add_argument("--poll-s-h1", type=float, default=3.0)
    ap.add_argument("--train-max-h", type=float, default=4.0)
    ap.add_argument("--train-max-h-4060", type=float, default=4.0)
    # DP-2 (2026-09-12, PREREG_DP2_BEATS_ONE_4060). Chain: --dp2 selects go_dp2 (arms dp2_s1/dp2_s2,
    # --only fidpair for the fidelity pair, --score-only for the scorer chain on any adapter).
    # Trainer: --role dp2-trainer (or --role trainer --dp2). None of these change the H=1 path.
    ap.add_argument("--dp2", action="store_true",
                    help="chain: the DP-2 (data-parallel, gradient-averaging) chain; trainer: alias for "
                         "--role dp2-trainer")
    ap.add_argument("--no-anchor-gc", action="store_true",
                    help="chain (dp2): do NOT pass --anchor-gc to the 5090 trainer. Default ON for dp2 "
                         "arms (owner decision 2026-09-12 08:5x)")
    ap.add_argument("--anchor-gc", action="store_true",
                    help="dp2-trainer on the 5090: every 25 steps delete THIS run's anchor objects with "
                         "step < current-4 over ssh (read->drop->atomic replace of names.json), then "
                         "publish the gc_s<step> marker the 4060 waits for before its next named PUT")
    ap.add_argument("--epochs", type=int, default=2,
                    help="dp2-trainer: passes over the 10,000 selected rows (registered: 2)")
    ap.add_argument("--fidelity-json", default=DP2["fidelity_json"],
                    help="dp2-trainer: the clause-1 artefact the 5090 asserts at launch and at step 31; "
                         "'none' = exempt (ONLY the fidelity gate's own pair)")
    ap.add_argument("--fid-steps", type=int, default=DP2["fidelity_gate_step"],
                    help="chain (dp2, --only fidpair): steps of the fidelity pair")
    ap.add_argument("--fid-tag", default="", help="chain (dp2, --only fidpair): pair tag/dir (default = run id)")
    ap.add_argument("--score-only", default="",
                    help="chain: run the h1 scorer chain on --adapter for this arm tag (<name>_s1|s2) into "
                         "<run-dir>/<tag>/ and write its verdict; dp2_* tags compare against twin4060_<seed> "
                         "unless --control-dir overrides")
    ap.add_argument("--adapter", default="", help="chain (--score-only): adapter dir (default <run-dir>/<tag>/adapter)")
    ap.add_argument("--control-dir", default="",
                    help="chain (--score-only): primary control dir holding S1_<name>.json/S1EXT/BATTERY")
    a = ap.parse_args(argv)
    if a.role == "chain":
        return role_chain(a)
    if not a.out_dir:
        a.out_dir = os.path.join(SPEC["remote_root"] if a.box == "4060" else SPEC["run_root"],
                                 a.run_id or "adhoc")
    if not a.model_dir:
        a.model_dir = SPEC["model_dir_%s" % a.box]
    if not a.data:
        a.data = SPEC["corpus_%s" % a.box]
    if a.role == "env-probe":
        return role_env_probe(a)
    if a.role == "dp2-trainer" or (a.role == "trainer" and a.dp2):
        return role_dp2_trainer(a)
    if a.role == "trainer":
        return role_trainer(a)
    if a.role == "gate-footprint":
        return role_gate_footprint(a)
    if a.role == "gate-basece":
        return role_gate_basece(a)
    if a.role == "stage-4060":
        return role_stage_4060(a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
