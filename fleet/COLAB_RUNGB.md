# Rung B fleet training on a free cloud GPU (Google Colab / Kaggle)

This is a **different, newer mechanism** from the main pool (`run_miner_client.py`, see `../COLAB.md`).
Instead of mining rounds of the toy/char-level model, your GPU trains **only its own disjoint slice of
experts** of a real Mixture-of-Experts model — no machine, including yours, ever holds or trains the whole
thing. Your slice's trained update is uploaded (a few tens of MB, not the model), and a coordinator merges
everyone's slices into the next model. See the design notes in the private repo's
`docs/research/rung-b-per-expert-training-2026-07-07.md` if you want the full story; this file is just the
"how do I join" version.

> **Two honest limits (same as the main pool).** (1) Free cloud GPUs **time out after a few hours** — a
> real test, not a 24/7 fleet slot. (2) Colab **cannot accept inbound connections**, so it never runs the
> coordinator — it always plays the worker role, publishing its result to a small public relay instead of
> being reached directly.

## Why this works from a sandboxed notebook: the content-store relay

`fleet/esh_worker.py` trains your slice and saves it to a local file, then (if you pass `--relay-name`)
**uploads** it to a small public content-addressed store the coordinator polls — a plain HTTP PUT of a
sha256-named blob (`content_store.py`; see the file for the ~90-line protocol). Colab only ever makes
**outbound** HTTPS calls, exactly like any other library install — no tunnel, no port-forward, no inbound
firewall rule needed on your end.

## The one cell (token-free)

Open a new Colab notebook, set **Runtime → Change runtime type → T4 GPU**, and paste this into a single
cell. It clones this **public** repo (no GitHub token needed) and runs one training round.

```python
# 1) clone the PUBLIC miner repo (no token needed)
!git clone https://github.com/whitegreenstudios/neurahash-miner
%cd neurahash-miner

# 2) install deps (Colab already ships a CUDA torch)
!pip install -q transformers bitsandbytes

# 3) the content-store upload needs a token -- set it as a Colab SECRET (key icon in the left sidebar),
#    NEVER paste the raw value into a cell. Ask whoever runs the coordinator for the token out of band.
from google.colab import userdata
import os
os.environ["NEURAHASH_CONTENT_TOKEN"] = userdata.get("NEURAHASH_CONTENT_TOKEN")

# 4) train your slice and publish it. --node/--nodes must match what the coordinator is using for THIS
#    round (ask the coordinator operator); --load-4bit keeps this well under Colab's T4 memory.
!python fleet/esh_worker.py \
    --node 2 --nodes 3 \
    --load-4bit \
    --out my_shard.pt \
    --relay-name rungb-colab-latest
```

That's it — the last line trains, saves `my_shard.pt` locally, and PUTs it to the public relay under the
friendly name `rungb-colab-latest`, where the coordinator (running wherever the pool operator has it, not
necessarily reachable from Colab at all) picks it up on its next poll.

## Tips

- **Which `--node`?** The coordinator operator assigns you a slot (`--node K` of `--nodes N`) so your
  expert range doesn't overlap anyone else's — ask before running, or the gather will reject you.
- **VRAM:** `--load-4bit` (nf4 via bitsandbytes) uses ~4 GB, comfortably inside a T4's 16 GB even alongside
  Colab's own overhead. Drop it (full bf16, ~14 GB) only on a bigger GPU (A100).
- **Keep the tab awake:** like the main pool, a free Colab session disconnects when idle — keep the tab
  open for the duration of a training run (a few minutes to tens of minutes depending on config).
- **Kaggle** works the same way (30 h/week GPU quota): same clone + install + run cell in a Kaggle
  notebook with a GPU accelerator enabled.

## Security reminder

Never paste the content-store token (or any secret) directly into a notebook cell — use **Colab
Secrets** (the key icon in the left sidebar) so it never ends up in the notebook file, which could be
shared or made public. The token only grants *write* access to the relay; it is not your wallet key and
not admin access to anything else.
