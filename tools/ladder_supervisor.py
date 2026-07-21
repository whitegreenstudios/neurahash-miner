"""
tools/ladder_supervisor.py — the capability LADDER autopilot: promote the pool to a strictly bigger
model when the current one has CONVERGED and the fleet is BIG ENOUGH to host the next rung.

The two triggers the owner asked for, unified:
  * "train a bigger model when the existing one is done"  → the DONE trigger (held-out loss plateau).
  * "when the 3070 joins, train a bigger model"           → the CAPACITY gate (the next rung's
    min_miners is only met once the 3rd GPU is in) — so a promotion needs BOTH: converged AND enough
    cards. That is deliberately non-thrashing: capacity alone never promotes a still-improving model.

Mechanism (why this is cheap): the coordinator DICTATES the architecture to every worker in the
`hello` message and workers AUTO-ADOPT it (sharded_pool_node._arch / run_worker). So a promotion is
just "restart the coordinator with NEURAHASH_ARCH_RUNG=<next> and a FRESH state dir"; the miners
reconnect and train the bigger model with NO change on any miner box. A bigger arch is
checkpoint-incompatible (coord_checkpoint.load_checkpoint refuses a mismatched arch — fail-loud), so
each rung is a fresh lineage; the previous rung's state is ARCHIVED, never destroyed.

SAFETY: default mode is `propose` — it decides + logs a governance-shaped `ladder-decision` record
and does NOTHING else. `enforce` additionally executes the archive+relaunch. Same audit-first rollout
as NEURAHASH_USEFULNESS / NEURAHASH_TRUSTVERIFY: watch the propose log first, then flip to enforce.

RUNG TABLE AS DATA (issue #90): the ladder is a JSON list the governance vote (#community loop) can set
via `--rungs <path>` (default = the built-in DEFAULT_RUNGS, byte-identical to the old hardcoded dict).
A rung may be marked `propose_only` — decide() then returns action="propose" (never "promote") for it
EVEN in enforce mode, so a governance-gated jump (a corpus or imported-base swap) is logged as a
recommendation the owner acts on, never auto-executed. A malformed --rungs file is a REFUSAL (exit),
never a silent fallback.

QWEN-BPE + IMPORTED-BASE RUNGS (issue #90 / #34) — the example table tools/ladder_rungs.example.json:
  Above the four char-MoE width rungs it adds
      {name:"qwen-bpe", min_miners:3, propose_only:true, arch_rung:"large", corpus:"qwen"}
  — same pool machinery + the `large` arch, but a REAL BPE tokenizer (NEURAHASH_CORPUS=qwen) instead of
  char-level. It is PROPOSE-ONLY on purpose, for two reasons that must land first:
    1. The corpus mode must reach every miner. Deliverable 1 makes the coordinator DICTATE the corpus
       mode in the hello (like the arch), so a qwen relaunch needs no env change on any miner — but that
       hello field has to be DEPLOYED FLEET-WIDE (old workers ignore it and would keep their env corpus)
       before an auto-promote is safe. Until then the promotion is a coordinated window the owner runs.
    2. The imported-base rungs (#34's qwen-1.7b → ~30B → glm-5.2 stepping-stones) gate on per-base
       readiness, not just fleet size: neurahash.posttrain_gate.gate_config_for(base) already sizes the
       readiness gate per rung. The supervisor PROPOSES; that gate (wired later — see docs/LADDER.md)
       DECIDES. Marking these rungs propose_only keeps the autopilot from jumping a base swap the gate
       has not cleared. (JSON has no comments; this docstring section IS the example table's commentary.)

decide() is a PURE function (fully unit-tested); the loop and the OS actions are thin wrappers.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# The ladder as DATA (issue #90): an ordered list of rung records the governance vote (#community loop)
# can set via --rungs <json>. Each record:
#   name        : the rung id (also the NEURAHASH_ARCH_RUNG value unless arch_rung overrides it)
#   min_miners  : fleet size needed to HOST this rung (a bigger model never needs fewer cards -> monotonic)
#   propose_only: (optional, default False) a rung the supervisor may only PROPOSE, never auto-promote —
#                 the promotion needs a human/governance step (e.g. a corpus/base swap not yet fleet-wide).
#   arch_rung   : (optional) the NEURAHASH_ARCH_RUNG the coordinator relaunches with (defaults to `name`).
#   corpus      : (optional) the NEURAHASH_CORPUS mode this rung trains on (documentation / future auto-set).
# The built-in table below is the four char-MoE width rungs; it MIRRORS sharded_pool_node.ARCH_RUNGS and is
# byte-identical to the old hardcoded dict (no --rungs given == unchanged behavior). Kept name-only so the
# tool needs no torch import. tools/ladder_rungs.example.json adds the qwen-BPE rung on top of these.
DEFAULT_RUNGS = [
    dict(name="tiny",   min_miners=1),
    dict(name="small",  min_miners=2),
    dict(name="medium", min_miners=3),
    dict(name="large",  min_miners=3),
]


def rung_names(rungs):
    return [r["name"] for r in rungs]


def rung_min_miners(rungs):
    return {r["name"]: int(r["min_miners"]) for r in rungs}


# Back-compat module-level views of the DEFAULT table (existing tests + the old hardcoded call sites read
# these). The data-driven decide() below takes an explicit `rungs` list; these remain the default.
RUNGS = rung_names(DEFAULT_RUNGS)
RUNG_MIN_MINERS = rung_min_miners(DEFAULT_RUNGS)

DEFAULTS = dict(
    patience=6,           # consecutive polls with no best-loss improvement => "converged"
    min_delta=1e-3,       # a poll only counts as improvement if best held-out loss drops by > this
    min_blocks=50,        # the rung must have actually trained this many blocks before it can be "done"
    poll_seconds=120,     # how often to sample pool_stats
    history_keep=64,      # rolling samples retained
)


def _rung_index(name, rungs=None):
    names = RUNGS if rungs is None else rung_names(rungs)
    return names.index(name) if name in names else -1


def highest_feasible_rung(miners_online, rungs=None):
    """The biggest rung the current fleet can host (its min_miners <= miners_online). `rungs` defaults to
    the built-in table so existing callers are unchanged; a custom table (from --rungs) is threaded in."""
    table = DEFAULT_RUNGS if rungs is None else rungs
    names, mins = rung_names(table), rung_min_miners(table)
    feasible = [r for r in names if mins[r] <= int(miners_online)]
    return feasible[-1] if feasible else names[0]


def is_converged(history, *, patience, min_delta, min_blocks, blocks_at_rung_start):
    """DONE = standard early-stopping patience: the trailing `patience` samples did NOT beat the best
    held-out loss seen before them by more than min_delta, AND the rung has trained >= min_blocks
    blocks (so a just-started rung is never 'done'). "Not improving" covers flat AND worsening loss —
    both mean more of the same model won't help, so it's time to grow. Pure over the sample history
    (each sample: {held_out_val_loss, blocks_found})."""
    samples = [s for s in history if s.get("held_out_val_loss") is not None]
    if len(samples) <= patience:
        return False, "warming up (not enough samples yet)"
    blocks_now = int(samples[-1].get("blocks_found", 0))
    # #94: a chain-height reset (fresh --state-dir / chain restart) can leave blocks_at_rung_start
    # ABOVE the current height, making the raw difference negative — which used to pin the rung
    # "too early to judge" forever and silently freeze the ladder. Clamp at the source so
    # blocks-this-rung is never negative; count from the post-reset height (main() re-anchors the
    # persisted baseline to match, so this is belt-and-suspenders for the pure path).
    trained = max(0, blocks_now - int(blocks_at_rung_start))
    if trained < int(min_blocks):
        return False, f"only {trained} blocks this rung (< {min_blocks} — too early to judge)"
    prior = samples[:-patience]
    window = samples[-patience:]
    best_prior = min(float(s["held_out_val_loss"]) for s in prior)
    best_window = min(float(s["held_out_val_loss"]) for s in window)
    improved = best_prior - best_window                       # new best the window achieved
    if improved > float(min_delta):
        return False, f"still improving (window beat prior best by {improved:.4f})"
    return True, f"not improving ({patience} polls beat prior best by only {improved:.4f})"


def decide(*, current_rung, miners_online, history, blocks_at_rung_start, cfg, rungs=None):
    """The core promotion decision — PURE (mode-agnostic; main() decides side effects). Returns a dict:
      {action: 'promote'|'propose'|'hold', target: <rung>, reason: str, converged: bool, feasible: <rung>}.

    Advance to the next rung iff: the model has CONVERGED, a next rung EXISTS, and the fleet is big enough
    to host that next rung (min_miners). Otherwise HOLD, with the reason why. The advance action is:
      * 'promote'  — the supervisor may execute the archive+relaunch itself (a plain width rung).
      * 'propose'  — the target rung is propose_only (a governance-gated jump, e.g. the qwen-BPE / imported
                     -base rungs): decide RECOMMENDS it but NEVER auto-executes, EVEN in enforce mode. This
                     is decided here (not in main) so the propose-vs-promote boundary is unit-tested and
                     the same whether the loop runs propose or enforce; main() treats 'propose' like 'hold'
                     for side effects and just logs the governance-shaped decision.
    `rungs` defaults to the built-in DEFAULT_RUNGS (unchanged); a custom table (from --rungs) is threaded in."""
    table = DEFAULT_RUNGS if rungs is None else rungs
    names, mins = rung_names(table), rung_min_miners(table)
    ci = _rung_index(current_rung, table)
    conv, why = is_converged(history, patience=cfg["patience"], min_delta=cfg["min_delta"],
                             min_blocks=cfg["min_blocks"], blocks_at_rung_start=blocks_at_rung_start)
    feasible = highest_feasible_rung(miners_online, table)
    at_top = ci >= len(names) - 1
    if not conv:
        return dict(action="hold", target=current_rung, converged=False, feasible=feasible,
                    reason=f"training {current_rung}: {why}")
    if at_top:
        return dict(action="hold", target=current_rung, converged=True, feasible=feasible,
                    reason=f"{current_rung} converged and is the TOP rung — keep earning "
                           f"(storage/serve roles); no bigger rung to grow into")
    nxt_rec = table[ci + 1]
    nxt = nxt_rec["name"]
    if mins[nxt] > int(miners_online):
        return dict(action="hold", target=current_rung, converged=True, feasible=feasible,
                    reason=f"{current_rung} converged but the next rung '{nxt}' needs "
                           f"{mins[nxt]} miners (have {miners_online}) — waiting for more "
                           f"GPUs (e.g. the 3070) before growing")
    # TODO(#94-secondary): gate promotion on per-worker VRAM feasibility (every rostered worker's
    # advertised usable vram_gb must fit the target rung's per-worker footprint at its dictated batch),
    # not just miner COUNT — the manual `large` promotion OOM-crash-looped the fleet. Needs the hello's
    # per-worker vram_gb telemetry (#23) threaded into decide(); out of scope for this minimal fix.
    if bool(nxt_rec.get("propose_only", False)):
        return dict(action="propose", target=nxt, converged=True, feasible=feasible,
                    reason=f"{current_rung} converged AND {miners_online} miners can host '{nxt}' "
                           f"(needs {mins[nxt]}), but '{nxt}' is PROPOSE-ONLY (governance-gated: "
                           f"corpus/base swap) → propose, do NOT auto-promote")
    return dict(action="promote", target=nxt, converged=True, feasible=feasible,
                reason=f"{current_rung} converged AND {miners_online} miners can host '{nxt}' "
                       f"(needs {mins[nxt]}) → promote")


# ---------------------------------------------------------------------------
# I/O wrappers (thin; the decision above is where the logic lives)
# ---------------------------------------------------------------------------
def read_stats(stats_path):
    try:
        d = json.load(open(stats_path, encoding="utf-8")).get("pool", {})
        return {"held_out_val_loss": d.get("held_out_val_loss", d.get("val_loss")),
                "miners_online": int(d.get("miners_online", 0)),
                "blocks_found": int(d.get("blocks_found", 0)), "t": None}
    except Exception as e:
        return {"held_out_val_loss": None, "miners_online": 0, "blocks_found": 0, "err": str(e)}


class RungConfigError(ValueError):
    """A --rungs JSON file that is malformed / fails schema validation. Raised (not swallowed) so the
    supervisor REFUSES to run on a broken table instead of silently guessing a ladder — a bad rung table
    could otherwise promote to a nonexistent arch or skip the capacity gate."""


def load_rungs(path):
    """Load + VALIDATE a rung table from a JSON file (issue #90 — the ladder as governance-settable data).

    Schema: a non-empty JSON LIST of objects, each {name:str, min_miners:int>=1, propose_only?:bool,
    arch_rung?:str, corpus?:str}. Names must be unique; min_miners must be MONOTONIC non-decreasing (a
    bigger model never needs fewer cards — the same invariant the built-in table upholds). On ANY problem
    raise RungConfigError with a clear message: the caller (main) exits rather than run on a guessed table.
    Returns the parsed list of dicts (ready to pass to decide(rungs=...))."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RungConfigError(f"--rungs file not found: {path}")
    except json.JSONDecodeError as e:
        raise RungConfigError(f"--rungs {path} is not valid JSON: {e}")
    if not isinstance(data, list) or not data:
        raise RungConfigError(f"--rungs {path} must be a non-empty JSON list of rung objects")
    seen, last_min = set(), None
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise RungConfigError(f"--rungs {path}: rung #{i} is not an object: {rec!r}")
        name = rec.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RungConfigError(f"--rungs {path}: rung #{i} has no valid 'name'")
        if name in seen:
            raise RungConfigError(f"--rungs {path}: duplicate rung name '{name}'")
        seen.add(name)
        mm = rec.get("min_miners")
        if not isinstance(mm, int) or isinstance(mm, bool) or mm < 1:
            raise RungConfigError(f"--rungs {path}: rung '{name}' min_miners must be an int >= 1 (got {mm!r})")
        if last_min is not None and mm < last_min:
            raise RungConfigError(f"--rungs {path}: rung '{name}' min_miners {mm} < previous {last_min} "
                                  f"— the table must be monotonic (a bigger model never needs fewer cards)")
        last_min = mm
        if "propose_only" in rec and not isinstance(rec["propose_only"], bool):
            raise RungConfigError(f"--rungs {path}: rung '{name}' propose_only must be true/false")
        for k in ("arch_rung", "corpus"):
            if k in rec and not isinstance(rec[k], str):
                raise RungConfigError(f"--rungs {path}: rung '{name}' {k} must be a string")
    return data


def _load_state(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {"current_rung": os.environ.get("NEURAHASH_ARCH_RUNG", "small"),
                "blocks_at_rung_start": 0, "history": [], "promotions": []}


def _save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, path)


def _log(log_path, rec):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def promote(target_rung, *, state_dir, archive_dir, relaunch_cmd, stop_cmd, blocks_now):
    """ENFORCE action: stop the pool, ARCHIVE the current state dir (never destroyed), relaunch the
    coordinator at `target_rung` with a fresh state dir. relaunch_cmd/stop_cmd are shell strings so
    this is host-agnostic (the live deploy passes the PowerShell launchers).

    `target_rung` may be the rung NAME (str) or the rung RECORD (dict from the table). A record lets the
    relaunch set NEURAHASH_ARCH_RUNG from `arch_rung` (defaults to the name) AND NEURAHASH_CORPUS from
    `corpus` — so a data/base rung swaps the corpus mode on relaunch, and the coordinator then DICTATES
    that mode to every worker via the hello (issue #90 deliverable 1), needing no env change on any miner.
    NOTE: main() never enforces a propose_only rung, so in practice only plain width rungs reach here."""
    rec = target_rung if isinstance(target_rung, dict) else {"name": target_rung}
    name = rec["name"]
    arch_rung = rec.get("arch_rung", name)
    if stop_cmd:
        subprocess.run(stop_cmd, shell=True)
    if state_dir and os.path.isdir(state_dir):
        os.makedirs(archive_dir, exist_ok=True)
        dest = os.path.join(archive_dir, f"{os.path.basename(state_dir)}_pre_{name}_{blocks_now}")
        shutil.move(state_dir, dest)
    env = dict(os.environ, NEURAHASH_ARCH_RUNG=arch_rung)
    if rec.get("corpus"):
        env["NEURAHASH_CORPUS"] = rec["corpus"]
    subprocess.Popen(relaunch_cmd, shell=True, env=env)
    return {"archived_to": dest if state_dir and os.path.isdir(archive_dir) else None,
            "arch_rung": arch_rung, "corpus": rec.get("corpus")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="capability-ladder autopilot (propose|enforce)")
    ap.add_argument("--stats", default="_poollive/pool_stats.json")
    ap.add_argument("--state", default="_poollive/ladder_state.json", help="supervisor's own memory")
    ap.add_argument("--log", default="_poollive/ladder.log")
    ap.add_argument("--mode", choices=["propose", "enforce"], default="propose")
    ap.add_argument("--current-rung", default=None, help="override the tracked current rung")
    ap.add_argument("--once", action="store_true", help="evaluate once and exit (for tests/cron)")
    ap.add_argument("--poll-seconds", type=int, default=DEFAULTS["poll_seconds"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--min-delta", type=float, default=DEFAULTS["min_delta"])
    ap.add_argument("--min-blocks", type=int, default=DEFAULTS["min_blocks"])
    ap.add_argument("--rungs", default=None,
                    help="path to a JSON rung table (default = the built-in tiny/small/medium/large "
                         "table). See tools/ladder_rungs.example.json for the schema.")
    # enforce-only knobs (host launch commands):
    ap.add_argument("--state-dir", default="_poollive/_state")
    ap.add_argument("--archive-dir", default="_poollive/_archive")
    ap.add_argument("--relaunch-cmd", default=None)
    ap.add_argument("--stop-cmd", default=None)
    a = ap.parse_args(argv)
    cfg = dict(patience=a.patience, min_delta=a.min_delta, min_blocks=a.min_blocks,
               poll_seconds=a.poll_seconds, history_keep=DEFAULTS["history_keep"])

    # RUNG TABLE (issue #90): --rungs loads a governance-settable JSON table; default = built-in.
    # A malformed table is a REFUSAL (exit 2), never a silent fallback — running the ladder on a guessed
    # table could promote to a nonexistent arch or skip the capacity gate.
    if a.rungs:
        try:
            rungs = load_rungs(a.rungs)
        except RungConfigError as e:
            print(f"[ladder] {e}", file=sys.stderr)
            return 2
    else:
        rungs = DEFAULT_RUNGS
    rung_by_name = {r["name"]: r for r in rungs}

    st = _load_state(a.state)
    if a.current_rung:
        st["current_rung"] = a.current_rung

    def tick():
        s = read_stats(a.stats)
        st["history"].append(s)
        st["history"] = st["history"][-cfg["history_keep"]:]
        # #94: the baseline is an ABSOLUTE chain height captured when this rung started. A chain-height
        # reset (fresh --state-dir, chain restart, coordinator resume that restarts height numbering)
        # drops the live height BELOW that baseline, so blocks-this-rung = height - baseline goes
        # negative and the rung is judged "too early" FOREVER -> the ladder silently stops promoting.
        # Re-anchor: when the observed height is below the stored baseline, reset the baseline to the
        # current height (blocks-this-rung restarts from 0 here) and emit one ASCII governance line.
        baseline = int(st.get("blocks_at_rung_start", 0))
        cur_h = int(s["blocks_found"])
        if cur_h < baseline:
            # Anchor to the LOWEST height seen since the reset (the post-reset floor), not just the
            # current sample, so blocks legitimately trained after the reset are still counted toward
            # min_blocks. In the common case (no earlier post-reset samples) that floor == cur_h.
            heights = [int(h.get("blocks_found", cur_h)) for h in st["history"]]
            new_baseline = min([cur_h] + [h for h in heights if h <= baseline])
            print(f"[ladder] height reset detected: re-anchoring rung {st['current_rung']} "
                  f"baseline {baseline} -> {new_baseline}")
            _log(a.log, {"kind": "ladder-reanchor", "current_rung": st["current_rung"],
                         "old_baseline": baseline, "new_baseline": new_baseline})
            st["blocks_at_rung_start"] = new_baseline
        d = decide(current_rung=st["current_rung"], miners_online=s["miners_online"],
                   history=st["history"], blocks_at_rung_start=st.get("blocks_at_rung_start", 0),
                   cfg=cfg, rungs=rungs)
        rec = {"kind": "ladder-decision", "mode": a.mode, "current_rung": st["current_rung"],
               "miners_online": s["miners_online"], "blocks_found": s["blocks_found"],
               "held_out_val_loss": s["held_out_val_loss"], **d}
        _log(a.log, rec)
        print(json.dumps(rec))
        # 'propose' (a propose_only target) is a GOVERNANCE-SHAPED decision: log it, execute NOTHING,
        # even in enforce mode — main treats it exactly like 'hold' for side effects (the guard below only
        # fires on 'promote'). This is where the qwen-BPE / imported-base rungs surface as recommendations.
        if d["action"] == "promote" and a.mode == "enforce":
            if not a.relaunch_cmd:
                print("[ladder] PROMOTE decided but --relaunch-cmd not set; refusing to act",
                      file=sys.stderr)
            else:
                info = promote(rung_by_name.get(d["target"], d["target"]),
                               state_dir=a.state_dir, archive_dir=a.archive_dir,
                               relaunch_cmd=a.relaunch_cmd, stop_cmd=a.stop_cmd,
                               blocks_now=s["blocks_found"])
                st["current_rung"] = d["target"]
                st["blocks_at_rung_start"] = s["blocks_found"]
                st["promotions"].append({"to": d["target"], "at_block": s["blocks_found"], **info})
                _log(a.log, {"kind": "ladder-promoted", "to": d["target"], **info})
        _save_state(a.state, st)
        return d

    if a.once:
        return 0 if tick()["action"] in ("hold", "promote", "propose") else 1
    while True:
        try:
            tick()
        except Exception as e:
            _log(a.log, {"kind": "ladder-error", "error": str(e)})
        time.sleep(cfg["poll_seconds"])


if __name__ == "__main__":
    raise SystemExit(main())
