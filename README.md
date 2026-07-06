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
