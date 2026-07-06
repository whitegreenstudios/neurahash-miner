# Joining a NeuraHash pool — operator walkthrough

This is the step-by-step version of the README. It gets one machine (yours, or a rented/cloud GPU)
mining into a coordinator someone is running.

## 0. Prerequisites

- Python 3.10+.
- A **torch** build for your machine (CPU-only works; a CUDA build uses your GPU).
- The **coordinator's connection info**: a host + port (e.g. `6.tcp.ngrok.io 12345`) or a
  `connect.json` URL, plus — for any non-demo pool — the **secret PSK** and optionally the
  **TLS pin**, handed to you out of band by whoever runs the coordinator.

## 1. Install

```bash
git clone https://github.com/OWNER/neurahash-miner
cd neurahash-miner
pip install -r requirements.txt
```

(Replace `OWNER` with the actual GitHub owner once the repo is published.)

## 2. Prove your build matches the pool

```bash
python -m pytest tests/test_worker_determinism.py -q
```

Both tests must pass. This confirms your recompute path produces the exact delta bytes the
coordinator expects — if it fails, your honest work would be rejected, so fix the environment
(usually a torch/BLAS version mismatch) before going further.

## 3. Preflight

```bash
python run_miner_client.py --host <coordinator-host> --port <port> --doctor
```

The doctor checks torch/CUDA, that the coordinator port is reachable, the TLS pin (if set), and your
local corpus hash. Fix anything red before mining.

## 4. Corpus must match the coordinator

Your training data must be **byte-identical** to the coordinator's, or every round you submit is
rejected. Two ways to guarantee it:

- **Recommended:** set a content store and let the client fetch by hash:
  ```bash
  export NEURAHASH_CONTENT_STORE=<store-url>
  # then run with --sync-corpus (step 5)
  ```
- **Manual:** ask the coordinator operator for their `corpus_sha` and hand-match the files in
  `corpus_data/`. The client prints your local `corpus_sha` so you can compare.

## 5. Mine

```bash
python run_miner_client.py \
  --host <coordinator-host> --port <port> \
  --name my-rig \
  --psk "$NEURAHASH_PSK" \
  --sync-corpus
```

- `--name` labels your connection; your **payout address** is derived from a secp256k1 key the client
  generates and stores locally the first time (back it up — it is the identity your work credits).
- `--psk` (or `NEURAHASH_PSK`) is the shared transport key. **Set it for any real pool.** With no PSK
  the client falls back to the PUBLIC demo key and warns you — fine for a loopback smoke test, not for
  a real deployment.
- The client **auto-reconnects** with capped backoff when the coordinator/tunnel drops, keeping the
  same address so you slot back into the same expert shard.

## 6. Security notes

- Set a **real secret PSK** on both sides. The committed `neurahash-demo-psk` authenticates nothing.
- If the coordinator publishes a **TLS pin**, set `NEURAHASH_TLS_PIN=<64-hex sha256>` so the client
  refuses any coordinator whose cert doesn't match (MITM protection).
- Cap GPU memory with `NEURAHASH_VRAM_CAP_GB=<n>` if you share the card with other work.
- Your wallet key lives under a local, gitignored directory. Never commit it, never paste it.

## Limits

- `--solo` (a local single-node coordinator) needs the full node package — it is **not** available
  from this client-only repo. Join a coordinator someone runs instead.
- This is a testnet/demo miner: no economic-security guarantees, and balances are not redeemable
  value. See the README's honest-status section.
