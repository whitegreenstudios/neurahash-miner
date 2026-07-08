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

## The one cell (genuinely token-free — no account, no secret, no signup)

Open a new Colab notebook, set **Runtime → Change runtime type → T4 GPU**, and paste this into a single
cell. It clones this **public** repo (no GitHub token needed) and runs one training round.

```python
# 1) clone the PUBLIC miner repo (no token needed)
!git clone https://github.com/whitegreenstudios/neurahash-miner
%cd neurahash-miner

# 2) install deps (Colab already ships a CUDA torch)
!pip install -q transformers bitsandbytes

# 3) train your slice and publish it -- no token needed, esh_worker.py ships a public demo relay by
#    default. --node/--nodes must match what the coordinator is using for THIS round (ask the operator
#    if you're joining a specific run); --load-4bit keeps this well under Colab's T4 memory.
!python fleet/esh_worker.py \
    --node 2 --nodes 3 \
    --load-4bit \
    --out my_shard.pt \
    --relay-name rungb-colab-latest
```

That's it — no GitHub token, no Colab Secret, no signup. The last line trains, saves `my_shard.pt`
locally, and PUTs it to the public relay under the friendly name `rungb-colab-latest`, where the
coordinator (running wherever the pool operator has it, not necessarily reachable from Colab at all)
picks it up on its next poll.

**Why "no token" is safe:** the relay's write auth is a public demo token (committed in
`fleet/esh_worker.py`, same idea as the main pool's public demo PSK) — it exists to keep casual/bot
traffic off a small shared box, not to gate who can contribute. What actually protects the model is the
coordinator's own held-out gate: every contribution is evaluated before it's ever merged in, and anything
that doesn't measurably help gets rejected and rolled back (see `propose_and_gate` in
`fleet/soak_coord_5090.py` in the private repo). A garbage or malicious upload wastes relay bandwidth,
nothing more.

## Tips

- **Which `--node`?** The coordinator operator assigns you a slot (`--node K` of `--nodes N`) so your
  expert range doesn't overlap anyone else's — ask before running, or the gather will reject you.
- **VRAM:** `--load-4bit` (nf4 via bitsandbytes) uses ~4 GB, comfortably inside a T4's 16 GB even alongside
  Colab's own overhead. Drop it (full bf16, ~14 GB) only on a bigger GPU (A100).
- **Keep the tab awake:** like the main pool, a free Colab session disconnects when idle — keep the tab
  open for the duration of a training run (a few minutes to tens of minutes depending on config).
- **Kaggle** works the same way (30 h/week GPU quota): same clone + install + run cell in a Kaggle
  notebook with a GPU accelerator enabled.

## Security note

The default relay token is intentionally public (see above) — there is nothing to protect here. If you
run your own PRIVATE coordinator + relay instead of the public default, pass `--token <yours>` (or set
`NEURAHASH_CONTENT_TOKEN`) and, as with any real secret, use a **Colab Secret** (key icon in the left
sidebar) rather than pasting it into a cell.
