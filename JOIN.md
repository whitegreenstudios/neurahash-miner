# Joining a NeuraHash pool — operator walkthrough

This is the step-by-step version of the README. It gets one machine (yours, or a rented/cloud GPU)
mining into the pool.

## Two ways to mine

**1. Turnkey, all-outbound (recommended, current).** Clone and train locally — no inbound coordinator
connection, no PSK, no port config. It fetches the base model from HuggingFace and trains outbound; to
*publish* your work (earn credit) it pushes a small delta to a pinning backend + the merge registry.

```bash
git clone https://github.com/whitegreenstudios/neurahash-miner
cd neurahash-miner
pip install -r requirements.txt
python tools/run_miner.py --once --lr 1e-5    # --lr 1e-5 is REQUIRED (the 3e-4 default destroys the base)
```

By default it runs in **LOCAL mode** (deltas trained + kept on disk). To publish you need **all three**:
`NEURAHASH_DILOCO_MERGE_URL=<merge-registry-url>`, `NEURAHASH_CONTENT_TOKEN=<registry write token>` (sent
as the `X-Auth` header — without it the registry PUT returns HTTP 401; ask a maintainer for one), and a
pinning backend (see the README). `run_miner.py` names whichever one is missing. The training
bundle is content-addressed + **hash-verified** from interchangeable seeds (HuggingFace / VPS / IPFS) —
see [BUNDLE.md](BUNDLE.md).

> **Status (2026-07):** the shared coordinator + merge loop is intermittently **down**; a delta published
> while it is down sits unmerged until it is back up. Local training works regardless.

**2. Connect to a live coordinator (synchronous).** The round-by-round path documented below. It needs a
coordinator's host/port (+ PSK / TLS pin) and a byte-matched corpus — use it when someone is running a
live coordinator you are joining.

---

## 0. Prerequisites

- Python 3.10+.
- A **torch** build for your machine (CPU-only works; a CUDA build uses your GPU).
- The **coordinator's connection info**: a host + port (e.g. `6.tcp.ngrok.io 12345`) or a
  `connect.json` URL, plus — for any non-demo pool — the **secret PSK** and optionally the
  **TLS pin**, handed to you out of band by whoever runs the coordinator.

## 1. Install

```bash
git clone https://github.com/whitegreenstudios/neurahash-miner
cd neurahash-miner
pip install -r requirements.txt
```

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
