# NeuraHash Miner

The **miner client** for NeuraHash — a proof-of-useful-work pool where the "work" is training a
shared Mixture-of-Experts model. Your GPU (or CPU) connects to a pool coordinator, trains its
assigned expert-shard for each round, signs the resulting trunk delta with your own key, and submits
it. Honest work that the coordinator can reproduce (recompute-verify) is what earns credit.

This repository is the **client half only**. It does not contain the coordinator, the consensus /
verdict logic, the ledger, or the emission/reward economics — you point it at a coordinator someone
else runs (or that you run from the full node package).

---

## ⚠️ Honest status — read this before you rely on it

- **This is the MINER CLIENT only.** It connects to a coordinator and trains; it does not settle
  money on its own, does not run the pool, and ships none of the reward/ledger/consensus core.
- **No economic-security guarantees.** This is a working prototype for a testnet / demo pool. Do not
  treat any balance it shows as real, redeemable value. The reward accounting lives on the
  coordinator/full-node side, which is not part of this repo.
- **The built-in demo PSK is PUBLIC.** The transport pre-shared key defaults to
  `neurahash-demo-psk`, which is committed here in the clear. It authenticates **nothing** against
  anyone who can read this source — it exists only so a loopback/dev smoke test needs no config.
  **For any real deployment, set a secret PSK out of band on BOTH the miner and the coordinator:**
  `--psk <secret>` or `NEURAHASH_PSK=<secret>`. If you leave the demo PSK on, the client warns you.
- **Your wallet key is yours, generated locally.** The miner creates a per-node secp256k1 identity on
  your machine (gitignored, never uploaded). Back it up; losing it loses the address your work
  credits. No private key ships in this repo.
- **Determinism matters.** Your training step must reproduce byte-for-byte what the coordinator
  recomputes, or your honest work is rejected. `tests/test_worker_determinism.py` pins the expected
  recompute hash — run it after install (see below).

---

## All-outbound miner (latest) — `tools/run_miner.py`

**Status: 2026-07-13.** A newer, turnkey miner that needs **no inbound port and no live coordinator
link** — it works behind any NAT. Each iteration fetches the base from HuggingFace (outbound), trains
the trunk locally, and publishes a small (<10 MB) signed, compressed delta outbound. A stranger can
`git clone` and run it with nothing else installed from the private core:

```bash
pip install -r requirements.txt
python tools/run_miner.py --once
```

- **Safe defaults (2026-07-22).** `--lr` now defaults to **`1e-5`** — the value that actually
  improves the base (verified same-seed: held-out CE `3.1936 → 2.8949`). The old `3e-4` default
  *destroyed* the base (`3.1936 → 6.4395`, 5/5 seeds), so a higher `--lr` remains a footgun — don't
  raise it. `--work-dir` now defaults to a portable `~/.neurahash/run_miner` (it previously
  hard-coded a maintainer-only absolute path). Proven this day in a fresh-clone WAN test on an
  RTX 5090 + RTX 4060: base fetched outbound from HuggingFace, real corpus synced **by hash over
  WAN** (`--sync-corpus`), signed delta published over WAN.
- `--once` runs a single fetch → train → publish cycle then exits; drop it to loop forever.
- **Publishing to EARN is zero-config (2026-07-22).** `run_miner.py` auto-defaults everything needed
  to go live — no env vars, no token to request: the merge registry URL, a per-machine signing key
  (created on first run), **and** the public demo write token (`NEURAHASH_CONTENT_TOKEN`, the same
  value `esh_worker` ships; it opens the shared store but secures nothing, since integrity is enforced
  by content-addressing). A pinning backend is auto-installed (a pinned, digest-verified kubo) when you
  have neither `PINATA_JWT` nor a local `ipfs`. So a bare `python tools/run_miner.py --once` fetches,
  trains, **and** publishes. Any explicit value you set (e.g. a private relay's token) always wins; if
  a backend genuinely can't be reached the miner stays in **LOCAL mode** (trains + keeps the delta on
  disk) and names what's missing instead of crashing.
- Other flags: `--device cuda|cpu` (default `cuda`), `--base qwen3-0.6b`, `--steps N`,
  `--wallet <name-or-keypath>`. First cold-start downloads the base weights from HuggingFace (~1.2 GB)
  via `tools/make_base_from_hf.py`. Set `NEURAHASH_MINER_KEY=<path>` to sign your contributions (GAP1).

Determinism gate for the delta codec (run after install):

```bash
python -m pytest tests/test_delta_codec_golden.py -q
```

The old networked "Rung B fleet worker" (`run_miner_client.py`, documented below) is unchanged and
still supported — this all-outbound path is an addition, not a replacement.

---

## Install

```bash
pip install -r requirements.txt
```

Install a **torch** build that matches your machine (CPU-only or a CUDA version) from the PyTorch
site — `requirements.txt` leaves torch unpinned on purpose.

Verify your build reproduces the pool's recompute path (this is the gate that matters):

```bash
python -m pytest tests/test_worker_determinism.py -q
```

Both tests must pass. A mismatch means your torch/BLAS build diverges from the pool's reference and
your honest work would be false-rejected — fix the environment before mining.

---

## Mine

Point it at a coordinator and give yourself a name (your payout address is derived from your local
key; `--name` labels the connection):

```bash
python run_miner_client.py --host <coordinator-host> --port <port> --name my-rig --sync-corpus
```

- `--host` — the coordinator host (or a comma-separated failover list of doors).
- `--connect <url>` — instead of `--host`, auto-discover the live coordinator from a published
  `connect.json`.
- `--sync-corpus` — fetch the coordinator's exact corpus by content-hash (needs
  `NEURAHASH_CONTENT_STORE` set to a reachable store). Your local corpus MUST be byte-identical to
  the coordinator's or your work is rejected; this flag guarantees that.
- `--psk <secret>` / `NEURAHASH_PSK` — the shared transport key. **Set this for any real pool.**
- `--device auto|cpu|cuda` — where to train (default auto-detects a GPU).
- `--doctor` — run a preflight checklist (torch/CUDA, coordinator reachability, TLS pin, corpus hash)
  and exit, without mining.

### Useful environment variables

| Variable | Purpose |
|---|---|
| `NEURAHASH_PSK` | secret transport PSK (overrides the public demo key) |
| `NEURAHASH_CORPUS=real` | train on the real text corpus in `corpus_data/` (vs the toy grammar) |
| `NEURAHASH_CONTENT_STORE` | URL of the content store `--sync-corpus` fetches the corpus from |
| `NEURAHASH_TLS_PIN` | the coordinator's 64-hex sha256 cert fingerprint (enforced before the handshake) |
| `NEURAHASH_VRAM_CAP_GB` / `NEURAHASH_VRAM_CAP_FRAC` | hard per-process GPU memory ceiling |
| `NEURAHASH_PQC=hybrid` | opt-in post-quantum (ML-DSA-44) admission alongside secp256k1 |

---

## What is (and isn't) in this repo

**Included (the client):** the launcher (`run_miner_client.py`), the worker/training loop
(`neurahash/worker_core.py`), the wire protocol, TLS/pin handling, the model architecture + factory,
the corpus loader, VRAM budgeting, wallet/identity, and the determinism gate.

**Not included (the private core):** the coordinator, the committee/verdict logic, trustverify, the
ledger, emission/reward economics, stake gates, and the settlement chain. `--solo` (spinning up a
local coordinator) therefore needs the full node package and is not usable from this repo alone.

---

## Run on a free cloud GPU

See **[COLAB.md](COLAB.md)** for a token-free Google Colab cell (works with the free T4) and
**[JOIN.md](JOIN.md)** for a step-by-step operator walkthrough.

---

## Rung B — fleet-wide MoE training (new, proven on 3 real machines)

A second, newer contribution mode lives in **[`fleet/`](fleet/)**. Instead of mining rounds of the
pool's toy/char model, your GPU trains **only its own disjoint slice of experts** of a real
Mixture-of-Experts model (`allenai/OLMoE-1B-7B-0924`, 64 experts × 16 layers) — no machine, including
yours, ever holds or trains the whole thing:

```bash
python fleet/esh_worker.py --node <0..N-1> --nodes <N> --load-4bit --relay-name my-shard
```

No account, no signup, **no token** — the default relay credential is a public demo token committed in
the source (same safety model as the main pool's demo PSK: it secures nothing, and doesn't need to —
the coordinator's own held-out accuracy gate is what protects the model, garbage deltas just get
rejected). See **[fleet/COLAB_RUNGB.md](fleet/COLAB_RUNGB.md)** for a one-cell Colab/Kaggle recipe.

**Status (2026-07-08):** proven live, over the real internet (not LAN), on 3 independently-owned
machines:

| Node | Machine | Result |
|---|---|---|
| 0 | RTX 5090 (operator's rig; also runs the coordinator) | trunk node; 10h/949-round solo soak completed clean (88.7% held-out, best 89.7%) |
| 1 | RTX 4060 (separate box, WAN-relayed through the content store) | 76.4MB delta relayed and sha256-verified round-trip |
| 2 | Google Colab (free T4, zero setup, fresh `git clone`) | 38.9MB delta relayed, sha256-verified, structurally validated |

Each node was proven independently reachable and contributing over WAN. **Not yet done:** a single
aligned gather combining fresh contributions from all 3 at once into one reported held-out number —
that's next now that the solo soak has freed the 5090.

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
the pinned release key, the **zero-config public miner** (safe defaults — a bare `run_miner.py --once`
earns with no env vars), and the shardDiLoCo + trustless-coordinator + elastic-VRAM work above.
