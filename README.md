# NeuraHash Miner

The **miner client** for NeuraHash — a proof-of-useful-work network where the "work" is training a
shared Mixture-of-Experts model. Your GPU trains its assigned expert slots (compact LoRA deltas on
a frozen GLM trunk), signs each delta with your own locally-generated key, and publishes it —
**all-outbound** (works behind NAT, nothing to port-forward) and **decoupled** (fast GPUs never
wait for slow ones). What earns credit is not "the work ran": it is **measured improvement on a
secret, rotated held-out set** — a contribution that does not make the shared model better pays
zero, and on the trustless lane the payout itself is co-signed by a staked M-of-N validator quorum
rather than trusted to any single coordinator. (An earlier round-based pool lane, where the
coordinator recompute-verified each training step bit-for-bit, was the network's original design —
deprecated 2026-07-24; see the deprecation notice below.)

This repository is the **client half only**. It does not contain the coordinator, the consensus /
verdict logic, the ledger, or the emission/reward economics — you point it at a coordinator someone
else runs (or that you run from the full node package).

---

## ⚠️ Honest status — read this before you rely on it

- **This is the MINER CLIENT only.** It trains its assigned expert slot and publishes signed
  deltas outbound; it does not settle money on its own, does not run the coordinator role, and
  ships none of the reward/ledger/consensus server core.
- **No economic-security guarantees.** This is a working prototype for a testnet. Do not treat any
  balance it shows as real, redeemable value. The reward accounting lives on the
  coordinator/full-node side, which is not part of this repo.
- **The store write token is a PUBLIC demo credential.** It opens the shared content store but
  secures nothing (and doesn't need to): integrity comes from content-addressing + signatures, and
  the model is protected by the held-out gate — garbage deltas simply pay zero and are not folded.
- **Your wallet key is yours, generated locally.** The miner creates a per-node secp256k1 identity on
  your machine (gitignored, never uploaded). Back it up; losing it loses the address your work
  credits. No private key ships in this repo.
- **No fragile determinism requirement.** The GLM lane gates on measured held-out improvement, not
  bit-exact recompute across different GPU architectures — your card's ~1-ULP numeric quirks cannot
  false-reject honest work.

---

## One lane: GLM shardDiLoCo (deprecation notice, 2026-07-24)

This repo now ships **exactly one way to contribute**: the GLM shardDiLoCo lane — your GPU trains
one expert slot of GLM-4.7-Flash and publishes compact LoRA deltas, all-outbound, corpus
self-syncing. Everything the Alpha 2.0 / 3.0 sections below describe *is* this lane.

The three earlier lanes — the Qwen open-base turnkey miner (`tools/run_miner.py`), the Rung B
OLMoE fleet worker (`fleet/esh_worker.py`), and the original round-based pool client
(`run_miner_client.py`) — are **deprecated as of 2026-07-24** and their code has moved to the
private full-node repo (it remains in this repo's git history). They were how the network proved
its transport, verification, and economics; the GLM lane is where all of that now lives. Their
dated result sections below are kept as the project's historical record.

---

## The road to a smarter model — G1, pre-registered (2026-07-24)

Honesty first: today's lane proves the **network** (trustless training, verified payouts, living
corpus) — it does not make the base model smarter on standard benchmarks, and we won't pretend
otherwise. The path that does is **verifiable-reward post-training (RLVR)**, and it is coming as
a **real open training campaign — G1** — whose one goal is a measurably smarter GLM, where every
joining miner does real training work (generating and verifying reasoning rollouts is the
compute-dominant part of RLVR) and more miners means the verdict arrives sooner.

- **The protocol is frozen and public**:
  [docs/G1_PREREGISTRATION_2026-07-24.md](docs/G1_PREREGISTRATION_2026-07-24.md) — held-out
  LiveCodeBench / competition-math / MMLU-Pro, McNemar significance, outcome-based stopping
  (stable success, or an honest published negative), the eval sets never shipped to miners.
  Published *before* any training so nobody — including us — can move the goalposts.
- **The training engine is built and tested**: verifiable math tasks distributed like the corpus,
  an un-gameable reward checker, the rollout worker (the miner "train" role), and the GRPO
  learner — all landing after final on-GPU verification. The campaign opens on this same keyless
  client; joining it will be the same one command.
- **The trained model belongs to everyone and cannot be lost**: every accepted training result is
  mirrored to HuggingFace (`neurahash-data/glm_ckpt`) with a `best.json` naming the
  best-so-far checkpoint — verified reconstructable from HF alone, with the operator's
  infrastructure switched off.

If G1's recipe fails its own gate, we say so publicly and rethink; miners' time is never spent on
a recipe the gate has not passed.

---

## Install

```bash
pip install -r requirements.txt
```

Install a **torch** build that matches your machine (CPU-only or a CUDA version) from the PyTorch
site — `requirements.txt` leaves torch unpinned on purpose.

Then run the client test suite once — it covers the GLM lane, the delta codec, the signed
self-update chain, and the VRAM manager:

```bash
python -m pytest tests/ -q
```

(The GLM lane does not require bit-exact recompute across GPU architectures — it gates on
held-out improvement, so there is no fragile torch/BLAS determinism requirement to satisfy.)

---

## Mine — join the GLM shardDiLoCo lane

**No key, no signup, no account.** Your machine creates its own wallet identity on first run
(`~/.neurahash/glm_miner_key` — back it up, it owns your payouts), signs every contribution with
it, and the network admits you on your first valid signed contribution. Your miner name *is* your
address (`glm-<addr[2:10]>`), so nobody can impersonate you and no operator can gate you:

```bash
python tools/sharddiloco_glm_contributor.py --mode glm \
  --slot <n> \
  --shard-dir <glm-shards> --config-dir <glm-config> \
  --data-dir <empty-dir> --domains daily \
  --url <content-store-url> --token <store-token> --device cuda
```

Everything heavy is fetched and verified for you: the GLM base shards come from the public bundle
(see [BUNDLE.md](BUNDLE.md)), an **empty `--data-dir` self-fills** with the advertised corpus
(sha256-verified, fail-closed), with `NEURAHASH_GLM_DATA_RESYNC=1` your running miner picks up
each newly published daily corpus with no restart — and if VRAM gets tight on a shared GPU, the
miner **pauses instead of crashing** and resumes when memory returns. All traffic is outbound;
NAT is fine. Payouts settle **to your wallet address** through the staked validator quorum —
proven live the day this shipped: keyless strangers' mints settled as
`settled miner=0xc47c93…` with quorum co-signatures. To pull a newly signed client release, run
`python tools/self_update.py` (signature-verified against the pinned release key).

(`--miner`/`--key` remain supported for operator-rostered miners; they are no longer required.)

### Useful environment variables

| Variable | Purpose |
|---|---|
| `NEURAHASH_GLM_DATA_RESYNC=1` | v3: a running miner picks up a newly published corpus with no restart (fail-closed) |
| `NEURAHASH_VRAM_MANAGER=on` | elastic VRAM: shed/grow training layers around whatever else uses your GPU |
| `NEURAHASH_VRAM_CAP_GB` / `NEURAHASH_VRAM_CAP_FRAC` | hard per-process GPU memory ceiling |

### G1 train-role — RLVR rollouts (v3.2, capacity-gated)

The [G1 campaign](docs/G1_PREREGISTRATION_2026-07-24.md)'s rollout worker ships in the client:

```bash
python tools/glm_rollout_worker.py --url <content-store-url> --token <store-token> \
  --shard-dir <glm-shards> --config-dir <glm-config>
```

It samples candidate solutions to verifiable math tasks, scores them with the in-repo reward
(`tools/glm_reward.py` — auditable), and publishes signed rollout sets the GRPO learner trains
on. **Honest note:** the full rollout policy is 59 GiB bf16, so today this role needs
`--full-model` on a big-RAM box (slow, VRAM-capped, box-safe) — the worker refuses
truncated-stack rollouts because a partial policy measured reward 0.0 (no training signal). The
full-speed engine is fleet-hosted pipeline rollouts across many 8 GiB cards as the fleet grows;
until your card can take rollout duty, the CE lane above is real training and real earning.

---

## What is (and isn't) in this repo

**Included (the client):** the GLM shardDiLoCo contributor stack
(`tools/sharddiloco_glm_contributor.py`, `tools/sharddiloco_glm_expert.py`,
`tools/sharddiloco_harness.py`, `tools/piece_loader.py`, `tools/diloco_contributor.py`), the base
bundle fetchers (`tools/fetch_glm_base.py`, `tools/bundle_pointer.py` + the kubo/IPFS fallback),
the signed self-update chain (`tools/self_update.py`, `tools/sign_release.py`, `release.json`),
the G1 train-role (`tools/glm_rollout_worker.py` + the verifiable reward `tools/glm_reward.py`),
elastic-VRAM management, wallet/identity, and — for auditability of the trust root — the staked
M-of-N settlement/quorum verification code and its tests.

**Not included (the private core):** the coordinator, the committee/verdict logic, trustverify, the
ledger, emission/reward economics, stake gates, and the settlement chain's server side. The
deprecated lanes (Qwen turnkey, Rung B fleet, legacy pool client) also now live there.

---

## Rung B — fleet-wide MoE training (historical; lane deprecated 2026-07-24)

*(This lane's code moved to the private full-node repo on 2026-07-24; the results below stand as
the project's historical record — Rung B was the first proof that per-expert sharding works on
strangers' hardware, the idea the GLM lane now carries.)* Your GPU trained **only its own disjoint
slice of experts** of a real Mixture-of-Experts model (`allenai/OLMoE-1B-7B-0924`, 64 experts × 16
layers) — no machine, including yours, ever held or trained the whole thing.

**Status (2026-07-08):** proven live, over the real internet (not LAN), on 3 independently-owned
machines:

| Node | Machine | Result |
|---|---|---|
| 0 | RTX 5090 (operator's rig; also runs the coordinator) | trunk node; 10h/949-round solo soak completed clean (88.7% held-out, best 89.7%) |
| 1 | RTX 4060 (separate box, WAN-relayed through the content store) | 76.4MB delta relayed and sha256-verified round-trip |
| 2 | Google Colab (free T4, zero setup, fresh `git clone`) | 38.9MB delta relayed, sha256-verified, structurally validated |

Each node was proven independently reachable and contributing over WAN. **Not yet done (as of
2026-07-08):** a single aligned gather combining fresh contributions from all 3 at once.
**UPDATE (2026-07-16): done and exceeded** — a 10-hour Rung-B run with **5 miners** completed 89
rounds in 7.45 h and took the base model from **51% → 100%** on its held-out gate.

---

## shardDiLoCo — training a model no single miner holds (proven over real WAN)

**Status (2026-07-20).** shardDiLoCo is the per-expert, async-DiLoCo training mode: each miner trains
only its own expert-shard, the model is *composed* and never fully resident on any one machine — the
mechanism behind the north-star goal of a consumer-GPU fleet training a model too big for a single
card. It completed a **full multi-round run over the real internet**: a coordinator and two per-expert
contributors, running as independent processes on **separate machines** (an RTX 5090 and an RTX 4060),
coordinating only over a remote content lane. Result: **both experts credited every round, zero
stalls, held-out cross-entropy fell 4.54 → 2.94, and the sharded-vs-synchronous compute ratio stayed
≈ 1.03** (the redundancy tax stays small). *(The coordinator/merge side lives in the full node package,
not this client repo.)*

How to join this lane: see **"Mine — join the GLM shardDiLoCo lane"** above.

## Trustless coordinator — the pool no longer runs on trust

**Status (2026-07-20).** The coordinator used to be a single trusted signer. It isn't any more: an
independent replayer now accepts a block **only if a genuine M-of-N quorum of staked validators signed
its exact mint** — a coordinator that tries to inflate, strip, or forge a payout is rejected. And a
second coordinator can take over on failure via a **signed, majority-agreed view-change**: proven live
across **three physically separate machines** (a 5090, a 4060, and a cloud datacenter node) — when the
elected leader was crashed, the two survivors formed a real 2-of-3 quorum and one took over with **no
chain fork**. This means the operator running the pool cannot silently cheat miners on payouts. Full
production activation (a real on-chain validator set + an external audit) is still gated on the
operator; the mechanism is proven and ships **default-off** until then.

## Being a good GPU neighbor — elastic VRAM (2026-07-22)

**Status (2026-07-22, landing in the miner now).** The miner is being made safe to run on the same GPU
you game or work on. It detects how much VRAM is actually free (accounting for everything else on the
card — the pool, your apps, anything), reserves a headroom for **you**, and re-checks every ~20 seconds:
if you launch something that needs the GPU, the miner **immediately sheds training layers** to give the
memory back, and only grows again once the memory stays free for a while (so it never thrashes or fights
you for the card). If not even one layer fits, it **pauses** instead of spilling into system RAM and
hanging your machine. The static VRAM cap (`NEURAHASH_VRAM_CAP_GB` / `NEURAHASH_VRAM_CAP_FRAC`) was also
hardened to work on multi-GPU boxes (`cuda:1`) and to size from *free* memory rather than total. Opt-in,
and unified with the capacity-aware work assignment so the coordinator only ever hands you work that
fits what you can currently spare.

## Alpha 3.2 (2026-07-24) — the G1 train-role ships in the client

The G1 campaign's **rollout worker and its reward function are now part of the miner**
(`tools/glm_rollout_worker.py`, `tools/glm_reward.py`). In RLVR the rollout-generation step *is*
the compute-dominant training work: the worker fetches math tasks from the lane
(sha256-verified, fail-closed), samples N candidate solutions from the current policy, scores
each with the **verifiable reward** (final-answer match against gold — the exact function that
decides pay is in this repo, auditable line by line), and signs + publishes the rollout set over
the same content lane the training deltas ride. Same keyless wallet identity as the CE lane.

**Capacity honesty, so nobody burns GPU for nothing:**

- The full rollout policy is the whole 47-layer GLM — **59 GiB in bf16**. No consumer card holds
  it, and 4-bit does not rescue it (measured: bnb quantizes only `nn.Linear`; GLM's fused expert
  modules stay bf16). The worker therefore **refuses truncated-stack rollouts by default**: a
  partial policy measured reward 0.0 — zero learning signal — so generating from it would be
  waste dressed up as work (`--allow-partial` exists for smoke tests only).
- `--full-model` loads plain bf16 under a **hard VRAM cap** with CPU/disk offload — box-safe,
  slow: a bootstrap path for big-RAM operators, not the real engine.
- The real engine is the same answer as everything else here: **fleet-hosted pipeline rollouts**,
  ~57 GiB of layers spread across many ordinary 8 GiB cards (the proven cross-card generation
  pattern). Meaning: the G1 rollout engine is literally *made of miners* — every card that joins
  brings it closer to running at full speed.
- Every load path in the worker now sets the per-process VRAM cap **before** the first CUDA
  allocation, so a rollout worker can never starve your desktop or a co-resident CE miner.

CE-lane mining is unchanged — small cards keep training and earning exactly as in v3.1.0.

## Alpha 3.1 (2026-07-24) — keyless mining, and a crash that can't happen again

**Status (2026-07-24, SHIPPED as the owner-signed `v3.1.0`** — the update chain was re-proven on
release day: a stale v3.0.0 clone verified the signature against the pinned key and applied
`v3.1.0` with no re-exec loop, and a brand-new user (fresh clone of the signed release, no key,
empty data dir) booted straight into mining: wallet auto-created → corpus self-fetched and
verified → training, first try.**)

- **Keyless open admission — nobody issues you anything.** Run the contributor with no `--key`
  and no `--miner`: your machine makes its own secp256k1 wallet, your name derives from your
  address (spoof-proof by construction), the coordinator admits you on your first valid signed
  contribution, and your mints settle **to your wallet address** through the staked M-of-N
  quorum. Proven live before shipping: two stranger machines joined with nothing but a fresh
  clone, trained real GLM (held-out CE 7.71 → 7.45), and their payouts settled to their
  self-made addresses with quorum co-signatures.
- **VRAM resilience.** The elastic-VRAM "pause instead of crash" promise is now real: at 0
  sustainable capacity the miner pauses and re-checks (never enters a doomed train/eval), and a
  CUDA OOM mid-round now costs one skipped round, not the miner. This was the exact crash a
  stranger on a busy shared GPU hit during the keyless live test — found and fixed the same day.
- **One lane** (see the deprecation notice above), and **the G1 pre-registration is published**
  ([docs/G1_PREREGISTRATION_2026-07-24.md](docs/G1_PREREGISTRATION_2026-07-24.md)): the frozen
  protocol for the real open training campaign — the run whose goal is a **measurably smarter
  model**, where every joining miner does real training work and makes the verdict arrive
  sooner. Published before any training so the goalposts cannot move.

## Alpha 3.0 (2026-07-24) — daily corpus, auto-updated to every running miner

**Status (2026-07-24, shipping as `v3.0.0`).** Alpha 3.0 makes the training data a living thing:

- **Daily corpus, zero effort.** The coordinator now publishes a fresh, license-clean daily corpus
  (arXiv abstracts / Wikipedia summaries / Hacker News) with a signed sha256 manifest. A miner with an
  **empty** data dir fills it by itself; nothing to download or configure.
- **Auto-update while running.** With `NEURAHASH_GLM_DATA_RESYNC=1`, a *running* miner re-checks the
  advertised corpus at every round boundary and, when a new version is published, re-fetches + verifies
  and trains on it with **no restart**. Fail-closed: an unverifiable corpus is refused and the
  known-good one kept. Proven live on two stranger machines (RTX 5090 + RTX 4060 over the real WAN) —
  both picked up a mid-run v2 re-publish (`corpus resync: manifest a5c6f0be..->9648c756..`).
- **Restart-proof lineage.** A coordinator restarting on a content store that still holds an old run's
  records can no longer strand miners: it publishes a genesis pointer at boot, and the miner-side
  catch-up now verifies every folded record against the advertised lineage (fail-closed rollback +
  frontier clamp), covered by new regression tests.
- **Research honesty note (why alpha-3 ships few features):** we spent the cycle answering the question
  the training plateau demanded. Verdict: the plateau is a base-model *capability* ceiling, not a
  data/storage one — so the roadmap now points at verifiable-reward post-training (alpha-4). Details
  land with the alpha-4 release.

## Alpha 2.0 (2026-07-24) — truly decoupled, self-syncing corpus, trustless-settled

**Status (2026-07-24, shipped as the signed `v2.0.0` auto-update — you are reading this because your
client can pull it).** Three things landed on the shardDiLoCo lane, all proven live on an RTX 5090 +
RTX 4060 training over the real internet as fresh stranger clones, then a 12-hour soak that settled
141 real mints through the quorum with zero withheld and zero errors:

- **Truly decoupled (#146).** The lane no longer makes a fast GPU wait for a slow one: each expert
  slot advances on **its own event clock** (DeepMind Decoupled-DiLoCo style, quorum K=1). Measured on
  the pair, the 5090 went from ~33 rounds/hr (old lock-step) to **~60**, while the 4060 ran free at its
  own ~36 — the fast card is never barriered on the slow one again. Behind `NEURAHASH_SD_ASYNC`;
  default-off and byte-identical on today's synchronous lanes.
- **Corpus auto-sync.** You no longer stage the corpus by hand: the coordinator advertises a **sha256
  manifest**, and the miner auto-downloads any missing/mismatched file (HuggingFace CDN first) and
  **verifies it fail-closed** before training. Proven: both boxes started with empty data dirs and
  self-filled over WAN. The coordinator's secret probe/held-out splits are structurally excluded.
- **Trustless settlement on the training lane.** Every training payout now settles through the same
  **staked M-of-N quorum trust root** — a mint is credited only if a majority of staked validators
  co-sign it, else it is withheld. Proven live: real GPU-trained mints settled with a quorum hash, and
  a **forged (inflated) mint was refused by the validator majority and left no ledger entry.**
  Default-off (`NEURAHASH_GLM_QUORUM`); the coordinator/settlement side lives in the full node package.

Together with the signed self-update the miner already had: **auto-update + auto-corpus + decoupled
GPU/WAN training + trustless settlement**, all in one lane.

*Alpha 1.0 (`v1.0.0`, 2026-07-21) is the baseline this builds on:* proven signed self-update against
the pinned release key, the then-current **zero-config public miner** (safe defaults — a bare
`run_miner.py --once` earned with no env vars; that lane was deprecated 2026-07-24), and the
shardDiLoCo + trustless-coordinator + elastic-VRAM work above.
