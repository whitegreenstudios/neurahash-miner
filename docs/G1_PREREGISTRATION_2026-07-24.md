# G1 pre-registration -- the smarter-model gate (frozen 2026-07-24, BEFORE any training)

G1 is the gate from docs/GLM_SMARTER_PLAN_2026-07-24.md: a 2-GPU, pre-registered base-vs-post
ablation on GLM-4.7-Flash. **No fleet scaling happens unless G1 passes. If G1 fails twice, we
stop and rethink the recipe -- 3000 miners never run anything this gate has not passed.**
Results are published (repo + issue #91) win or lose. This file freezes the protocol; a
follow-up "pin commit" freezes the exact eval item lists (sha256) before the first training
step. Changing anything below after the pin commit voids the attempt.

## 1. Model under test

- Base: GLM-4.7-Flash, the exact local snapshot the lane trains (D:/hf_models/GLM-4.7-Flash-bf16;
  31B-class MoE, 47 layers, 64 experts/layer). The base arm and every post arm are evaluated with
  the SAME weights-precision path (paired design): quantized inference is allowed only if base
  and post use the identical quantization.
- Arms: (A) BASE untouched; (B) +SFT on open verified reasoning traces; (C) B + scoped GRPO.
  The GATE compares C vs A. B vs A is reported as the distillation ablation.

## 2. Held-out benchmarks (chosen for contamination resistance; primary = statistical gate)

| id | set | N (target) | metric | reward check |
|---|---|---|---|---|
| P1 (primary) | LiveCodeBench, newest rolling window post-dating the base's cutoff | all items in window (expect 100-300) | pass@1 | unit-test execution |
| P2 (primary) | Fresh verified competition-math bank slice, 2026-dated items only | 200 | exact-match | math-verify (sympy/string) |
| P3 (primary) | MMLU-Pro, stratified random slice, seed=20260724 | 500 | exact letter | choice extraction |
| D1 (descriptive) | AIME 2026 (both sessions) | 30 | exact-match | math-verify |
| D2 (descriptive) | GPQA-Diamond | 198 | exact letter | choice extraction |

D1/D2 are REPORTED but carry no gate weight (N too small / expected movement too small -- see the
plan's T4). The pin commit records sha256 of every item list + the exact prompts.

## 3. Inference protocol (identical for every arm)

- temperature 0.6, top_p 0.95, n=1 response per item, per-item seed = sha256(item_id) % 2^31.
- max_new_tokens: 8192 (P1), 6144 (P2/D1), 2048 (P3/D2). Same chat template for all arms.
- Answer extraction rules are code, committed with the harness, identical across arms.

## 4. Training budget caps (anti-goalpost-moving; exceeding any cap voids the attempt)

- SFT (arm B): open verified traces only (OpenThoughts-3 / R1-distill families, Apache/MIT),
  <= 50,000 traces, <= 2 epochs, context <= 8192. LoRA rank <= 64, alpha = 2r, on attention +
  MLP projections of ALL layers (not expert-slot-only). No benchmark test items in training
  data -- enforced by n-gram decontamination against every pinned eval set (report the hit count).
- GRPO (arm C): <= 2,000 verifiable problems (math + code, none from any pinned eval set),
  <= 8 rollouts/problem/step, <= 300 optimizer steps, same LoRA surface. Rewards: math-verify
  exact-match; code unit-test pass. No reward model, no human preference data.
- Hardware: local 5090 (+ optional RunPod, spend-capped by the owner); compute spent is reported.

## 5. The gate (decided by numbers, not judgment)

- Per primary benchmark: McNemar exact test on paired per-item outcomes (A vs C), via the
  epoch_verdict machinery, alpha = 0.05.
- **G1 PASSES iff: at least one of P1/P2/P3 improves with p < 0.05 AND no primary benchmark
  regresses with p < 0.05.** (The anti-forgetting clause mirrors epoch_verdict's design.)
- Everything else (effect sizes, D1/D2 movement, arm B) is reported context, not gate input.
- An attempt = one full A/B/C cycle under these caps. Two failed attempts => STOP per the plan's
  kill criteria; the failure analysis is published like a result.

## 5b. Real open training campaign (owner directive, 2026-07-24 — added before the pin commit)

G1 is **a real training run, not a time-box and not a test harness.** Its one goal: find out
whether this training recipe makes GLM-4.7-Flash measurably smarter on the frozen held-out
benchmarks. Anyone may join and **train** through `neurahash-miner`; the more miners, the sooner
and the more trustworthy the answer.

- **Miners do the actual training** (this is real mining, not error-hunting). By hardware, from
  the measured fleet arithmetic: 24 GB+ GPUs generate RL ROLLOUTS (quantized GLM-4.7-Flash + the
  current LoRA policy adapter, pulled over the same verified rails that ship the corpus) — in
  RLVR the rollouts ARE the compute-dominant training work, so a rollout miner is training the
  model, not testing it; smaller GPUs / CPUs run the verifiable REWARD checks (math exact-match,
  sandboxed unit tests) that turn rollouts into a learning signal. The serial gradient step (SFT
  warmup then GRPO) runs on the operator's learner tier. Every miner is paid per verified
  contribution through the existing held-out-gated, quorum-settled mint.
- **More miners = faster to the verdict.** Rollout generation is embarrassingly parallel, so fleet
  size scales training throughput near-linearly until the learner's serial step saturates (far
  above any near-term fleet size). Honest bound: more compute reaches the answer sooner and
  better-tested — a YES (ship the smarter model) or a definitive NO (this recipe doesn't move the
  benchmark at this scale). It cannot convert a real NO into a YES; that honesty is the point.
- **Outcome-based stopping (NOT a fixed calendar window).** The run continues until ONE fires,
  all declared here before training:
  * SUCCESS — a primary benchmark shows a McNemar-significant gain (§5) that is STABLE: it holds
    across >=2 consecutive declared checkpoints, not a one-checkpoint spike. -> the model got
    smarter; freeze that adapter, publish, promote to alpha-4.
  * FUTILITY — no primary benchmark shows a significant positive trend across >=4 consecutive
    checkpoints spanning a documented rollout-token volume (target >= a DeepScaleR-scale budget,
    ~3.5e10 rollout tokens). -> recorded as a real NEGATIVE result at this scale/recipe.
  * BACKSTOP — a generous max horizon (declared at pin commit) purely so it cannot run forever;
    reaching it is reported as FUTILITY, not success.
  Checkpoints are at declared rollout-token milestones; the FULL benchmark-vs-compute curve is
  published (no cherry-picking the best point). This makes "run until smarter" honest: the eval
  is frozen, the stopping rules are pre-declared, and optional-stopping bias is bounded by the
  consecutive-checkpoint stability requirement.
- **Eval + task set stay frozen; CLIENT code may be fixed mid-run.** Rollout-worker / verifier /
  transport fixes ship as signed self-updates, each logged in the results file (errors surfaced by
  real strangers' hardware are expected and get fixed live — a byproduct of running for real, not
  the goal). The eval protocol, the held-out sets, and the pinned task set may NOT change; any
  change there voids the attempt and restarts the clock.

## 6. Commit discipline

1. This file lands first (protocol freeze).
2. Pin commit: harness + materialized item lists + their sha256 manifest + decontamination
   report. After it, training may start.
3. Results commit: raw per-item outcomes for every arm (jsonl), the McNemar outputs, and the
   verdict -- pass or fail, published unedited.
