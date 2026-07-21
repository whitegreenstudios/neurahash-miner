# The capability ladder

The pool trains a **strictly bigger model as it converges and the fleet grows**. Each step up is a
*rung*. This page is the one-page model: what a rung is, the two conditions that trigger a promotion,
how a promotion executes, `propose` vs `enforce`, and how the future imported-base rungs will consult
the per-base readiness gate.

Code: [`tools/ladder_supervisor.py`](../tools/ladder_supervisor.py) (the autopilot),
[`sharded_pool_node.py`](../sharded_pool_node.py) `ARCH_RUNGS` / `_arch` / `run_worker` (the arch +
corpus the pool trains), and the example table
[`tools/ladder_rungs.example.json`](../tools/ladder_rungs.example.json).

## The rung model

A rung is one row of the ladder table. The built-in table is four char-level MoE widths:

| rung   | min_miners | what grows                              |
|--------|-----------:|-----------------------------------------|
| tiny   | 1          | d_model 64, 2 layers (fast soak)        |
| small  | 2          | d_model 384, 6 layers (historical default) |
| medium | 3          | d_model 512, 8 layers                   |
| large  | 3          | d_model 768, 12 layers                  |

The widths live in `sharded_pool_node.ARCH_RUNGS`; the supervisor mirrors the **names + min_miners**
so the tool needs no torch import. `min_miners` is the fleet size needed to *host* the rung; it is
**monotonic** (a bigger model never needs fewer cards), and the loader enforces that.

**Invariant — every rung is a separate lineage.** A bigger arch is checkpoint-incompatible
(`coord_checkpoint.load_checkpoint` refuses a mismatched arch — fail-loud), so a rung is never one
model *grown in place*: each promotion starts a **fresh state dir**, and the previous rung's state is
**archived, never destroyed**.

The table is **data** (issue #90). `--rungs <path>` loads a JSON list the governance vote (#community
loop) can set; the default is the built-in table (unchanged). Schema — an ordered list of objects:

```json
{"name": "large", "min_miners": 3, "propose_only": false, "arch_rung": "large", "corpus": "real"}
```

- `name` — the rung id (and the `NEURAHASH_ARCH_RUNG` value unless `arch_rung` overrides it).
- `min_miners` — int ≥ 1, monotonic non-decreasing across the list.
- `propose_only` — optional bool (see below).
- `arch_rung` — optional; the `NEURAHASH_ARCH_RUNG` to relaunch with (defaults to `name`).
- `corpus` — optional; the `NEURAHASH_CORPUS` mode this rung trains on.

A malformed `--rungs` file is a **refusal** (the supervisor exits) — never a silent fallback, because
a guessed table could promote to a nonexistent arch or skip the capacity gate.

## The two trigger conditions

A promotion needs **BOTH**, which is deliberately non-thrashing:

1. **DONE** — the current model has **converged**: standard early-stopping patience over the held-out
   loss (`patience` consecutive polls that don't beat the prior best by `min_delta`), *and* the rung
   has trained at least `min_blocks` blocks (a just-started rung is never "done"). "Not improving"
   covers flat **and** worsening loss — both mean more of the same model won't help.
2. **CAPACITY** — the fleet is **big enough to host the next rung** (`min_miners`). Capacity alone
   never promotes a still-improving model; convergence alone waits at the top of the feasible fleet.

This is exactly the pair the owner asked for: *"train a bigger model when the existing one is done"*
(DONE) and *"when the 3070 joins, train a bigger model"* (CAPACITY — the next rung's `min_miners` is
only met once the 3rd GPU is in).

`decide()` is a **pure function** (fully unit-tested in `tests/test_ladder_supervisor.py`); the poll
loop and the OS actions are thin wrappers around it.

## How a promotion executes

The mechanism is cheap because the coordinator **dictates** to every worker in the `hello` message and
workers **auto-adopt** — so a promotion changes nothing on any miner box:

1. **Stop** the coordinator (`--stop-cmd`).
2. **Archive** the current state dir into `--archive-dir` (moved, never deleted → the rung's lineage is
   preserved).
3. **Relaunch** the coordinator (`--relaunch-cmd`) with `NEURAHASH_ARCH_RUNG=<arch_rung>` and a
   **fresh** state dir. If the rung sets `corpus`, `NEURAHASH_CORPUS` is set on the relaunch too.
4. Miners **reconnect** and pick up the new arch — and now the new corpus mode — from the `hello`:
   - **arch**: `run_worker` reads `hello["arch"]` and rebuilds its model (existing behavior).
   - **corpus** (issue #90): `run_worker` reads `hello["corpus"]` and adopts it for `build_data` +
     the `corpus_sha` handshake. **Precedence: the hello value wins over the worker's
     `NEURAHASH_CORPUS` env** (a one-line note prints when they differ). A hello *without* the field
     (an old coordinator) leaves the worker on its env behavior, exactly as before — the field is
     additive, so old workers and old coordinators both still interoperate.

   Dictating the corpus mode is what lets a real-corpus / qwen-BPE rung ship with **no env change on
   any miner**. The `corpus_sha` handshake still fails loud if two boxes' `corpus_data/*.txt` bytes
   differ under that mode.

## `propose` vs `enforce`

- **`propose`** (default) — decide + log a governance-shaped `ladder-decision` record and do **nothing
  else**. Watch the log first.
- **`enforce`** — additionally execute the archive + relaunch above.

Same audit-first rollout as `NEURAHASH_USEFULNESS` / `NEURAHASH_TRUSTVERIFY`: run `propose`, read the
log, then flip to `enforce`.

**`propose_only` rungs.** A rung marked `propose_only: true` is a governance-gated jump. `decide()`
returns `action="propose"` for it (never `"promote"`) **even in enforce mode**, so the supervisor logs
the recommendation but never auto-executes it — `main()` treats `propose` exactly like `hold` for side
effects. This keeps the autopilot from jumping a corpus or base swap a human/governance step must clear
first. The decision is made in the *pure* `decide()` (so the propose-vs-promote boundary is
unit-tested and identical whether the loop runs propose or enforce).

The example table adds one such rung on top of `large`:

```json
{"name": "qwen-bpe", "min_miners": 3, "propose_only": true, "arch_rung": "large", "corpus": "qwen"}
```

Same pool machinery + the `large` arch, but a **real BPE tokenizer** (`NEURAHASH_CORPUS=qwen`) instead
of char-level. It stays `propose_only` until (a) the corpus-mode hello (above) is **deployed
fleet-wide** — old workers ignore the field and would keep their env corpus — and (b) the imported-base
gate integration below lands.

## Future imported-base rungs (issue #34)

The next real capability jumps are **DATA and BASE**, not just width. Issue #34's stepping-stone
rollout is a ladder of progressively larger *downloaded* open bases as the fleet grows —
**qwen-1.7b → ~30B → glm-5.2** — each a separate pretrained model imported + post-trained, never one
model grown bigger.

For these rungs the capacity condition is **per-base readiness**, not just fleet size. The pool already
has the sizing primitive: `neurahash.posttrain_gate.gate_config_for(base)` sizes the readiness gate for
a given base (`base_from_name()` resolves the name; a small base is ready with a handful of miners +
one small GPU, glm-5.2 needs hundreds + ~744 GB sharded). The division of labour:

> **The supervisor PROPOSES; the gate DECIDES.**

Concretely, when the imported-base integration lands, an imported-base rung is `propose_only` in the
table, and the promotion is gated on `gate_config_for(base_from_name(<rung base>))` going green rather
than on `min_miners` alone. This is a **reference, not yet wired** — `gate_config_for` exists today;
the supervisor does not call it yet. Until then these rungs surface as `propose` recommendations the
owner acts on.

Ties: #34 (per-rung promotion), #49 (a converged rung moves to storage/serve roles while the next rung
trains), and the vote page (holders pick the base).
