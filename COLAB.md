# Mine on a free cloud GPU (Google Colab / Kaggle)

Spin up a cloud GPU worker that joins your pool — no local card needed. Free Colab gives a **T4**,
which is plenty to mine. This repo is **public**, so — unlike a private setup — the Colab cell needs
**no GitHub token**: it just `git clone`s this repository directly.

> **Two honest limits.** (1) Free cloud GPUs **time out after a few hours** — great for a test
> session, not a 24/7 fleet. (2) The worker connects **outward** to your coordinator, so your pool
> must be reachable from the internet (a public host, or a tunnel like ngrok). Colab can only be a
> *worker*, never the coordinator (it blocks inbound connections).

## Turnkey (all-outbound) — simplest

If you just want to train and (optionally) contribute without wiring up a coordinator connection, use the
all-outbound turnkey miner — one cell, no host/port/PSK:

```python
!git clone https://github.com/whitegreenstudios/neurahash-miner
%cd neurahash-miner
!pip install -q -r requirements.txt
!python tools/run_miner.py --once --lr 1e-5    # --lr 1e-5 is REQUIRED (the 3e-4 default destroys the base)
```

This trains in **LOCAL mode** (delta kept on disk). To publish, set `NEURAHASH_DILOCO_MERGE_URL` + a
pinning backend (see the README / [BUNDLE.md](BUNDLE.md)). Note (2026-07): the shared coordinator + merge
loop is intermittently down, so a delta published while it is down sits unmerged until it is back up.

---

## The one cell (token-free) — connect to a live coordinator

Open a new Colab notebook, set **Runtime → Change runtime type → T4 GPU**, and paste this into a
single cell. Replace the placeholders:

- `<coordinator-host>` / `<port>` — your coordinator's public address.
- `NEURAHASH_PSK` — your secret pool key (leave the demo default only for a throwaway test).

```python
# 1) clone the PUBLIC miner repo (no token needed)
!git clone https://github.com/whitegreenstudios/neurahash-miner
%cd neurahash-miner

# 2) install deps (Colab already ships a CUDA torch)
!pip install -q -r requirements.txt

# 3) configure the join
import os
os.environ["NEURAHASH_CORPUS"] = "real"                      # train on the real corpus
os.environ["NEURAHASH_CONTENT_STORE"] = "<content-store-url>"  # where --sync-corpus fetches the corpus
os.environ["NEURAHASH_TLS_PIN"] = "<64-hex-sha256-cert-fingerprint>"  # coordinator's cert pin (MITM guard)
os.environ["NEURAHASH_PSK"] = "<your-secret-pool-psk>"       # REAL pools: set this. Demo default authenticates nothing.

# 4) mine (auto-reconnects; --sync-corpus makes your corpus byte-match the coordinator)
!python run_miner_client.py \
    --host <coordinator-host> --port <port> \
    --name colab-t4 \
    --sync-corpus
```

## Tips

- **Check it's alive:** the client prints your `corpus_sha` and a per-round log. If rounds are being
  rejected, your corpus almost certainly doesn't match the coordinator — confirm
  `NEURAHASH_CONTENT_STORE` is reachable and `--sync-corpus` ran.
- **Preflight first:** add `--doctor` to the last command to run the checklist (torch/CUDA,
  reachability, TLS pin, corpus hash) and exit, before committing a real session.
- **Kaggle** works the same way (30h/week GPU quota): same clone + install + run cell in a Kaggle
  notebook with a GPU accelerator enabled.
- **Keep the tab awake:** free Colab disconnects idle sessions; a long mining run needs the tab open.

## Security reminder

The committed `neurahash-demo-psk` is **public** and secures nothing. For any pool you actually care
about, set a secret `NEURAHASH_PSK` (and ideally `NEURAHASH_TLS_PIN`) on **both** the Colab worker and
the coordinator. See the README's honest-status section.
