#!/usr/bin/env python3
"""shardDiLoCo Phase-2b -- runnable CONTRIBUTOR CLI (all-outbound, like tools/diloco_contributor.py).

The miner half of the WAN run-harness. A contributor OWNS one expert + a domain shard, trains
{trunk replica + its expert} for H local inner steps with ZERO cross-miner comm (the anti-flap
core), and PUBLISHES a per-expert + trunk pseudo-gradient on the content-addressed + signed lane,
then loops on the coordinator's pointer. It is ALL-OUTBOUND: it only GETs the pointer/state and PUTs
its delta + a signed record -- exactly the transport a real 4060 (over WAN via the VPS anchor
47.84.93.96) or a RunPod GPU uses. It reuses the REAL phase-2 kernel
tools/diloco_contributor.train_expert_contribution (offline-routed, MoELM.forward/backward_offline).

Because there is NO per-round synchronous barrier -- a contributor pulls the pointer, trains at its
OWN pace, publishes, and pulls again -- a slow or dropping WAN miner never stalls a round (the whole
point of Phase-2: the 2026-07-19 flap is gone).

Usage:
  C:/Python313/python.exe tools/sharddiloco_contributor.py --miner miner0 --expert 0 \
      --key <hex16> --url http://127.0.0.1:8797 --token <tok>
Env fallbacks: NEURAHASH_CONTENT_URL, NEURAHASH_CONTENT_TOKEN, NEURAHASH_SD_MINER,
NEURAHASH_SD_EXPERT, NEURAHASH_SD_KEY (or --key-file), plus HarnessConfig NEURAHASH_SD_* overrides
(the pulled canonical state's cfg is authoritative for dims/cadence).
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                               # noqa: E402
import sharddiloco_harness as H                                   # noqa: E402
import diloco_contributor as dc                                  # noqa: E402  (torch: the REAL kernel)
import neurahash.diloco_merge as dm                              # noqa: E402  (streaming-subset #126)


def _flush(*a):
    print(*a, flush=True)


def _resolve_key(args):
    if args.key:
        return bytes.fromhex(args.key)
    if args.key_file and os.path.exists(args.key_file):
        return bytes.fromhex(open(args.key_file, "r", encoding="utf-8").read().strip())
    env = os.environ.get("NEURAHASH_SD_KEY")
    if env:
        return bytes.fromhex(env)
    raise SystemExit("[contrib] no signing key: pass --key <hex16> / --key-file / NEURAHASH_SD_KEY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miner", default=os.environ.get("NEURAHASH_SD_MINER", "miner0"))
    ap.add_argument("--expert", type=int, default=int(os.environ.get("NEURAHASH_SD_EXPERT", "0")))
    ap.add_argument("--url", default=os.environ.get("NEURAHASH_CONTENT_URL", "http://127.0.0.1:8797"))
    ap.add_argument("--token", default=os.environ.get("NEURAHASH_CONTENT_TOKEN", ""))
    ap.add_argument("--key", default=None)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--max-rounds", type=int, default=int(os.environ.get("NEURAHASH_SD_MAX_ROUNDS", "1000")),
                    help="safety cap so a dead coordinator does not loop the contributor forever")
    ap.add_argument("--poll", type=float, default=0.25, help="pointer poll interval (s)")
    ap.add_argument("--wait-up", type=float, default=30.0, help="seconds to wait for the coordinator pointer")
    ap.add_argument("--round-wait", type=float, default=120.0,
                    help="max s to keep a published record advertised before giving up on a round")
    args = ap.parse_args()

    key = _resolve_key(args)
    e = args.expert
    miner = args.miner
    lane = H.ContentLane(args.url, args.token)
    _flush("[contrib %s] UP owns expert %d | lane=%s (all-outbound)" % (miner, e, args.url))

    # wait for the coordinator's first pointer
    ptr = None
    t0 = time.time()
    while time.time() - t0 < args.wait_up:
        try:
            ptr = lane.read_pointer()
        except Exception:                                        # noqa: BLE001
            ptr = None
        if ptr is not None:
            break
        time.sleep(args.poll)
    if ptr is None:
        _flush("[contrib %s] FATAL: no coordinator pointer at %s after %.0fs" % (miner, args.url, args.wait_up))
        return 4

    done_last = -1
    rounds_done = 0
    while rounds_done < args.max_rounds:
        try:
            ptr = lane.read_pointer()
        except Exception:                                        # noqa: BLE001
            time.sleep(args.poll)
            continue
        if ptr is None:
            time.sleep(args.poll)
            continue
        if ptr.get("done"):
            _flush("[contrib %s] coordinator signalled DONE; exiting after %d contributions" % (miner, rounds_done))
            return 0
        rnd = int(ptr["round"])
        if rnd <= done_last:
            time.sleep(args.poll)                                # wait for the round to advance
            continue

        # ---- pull canonical trunk + my expert e for this round ----
        meta, trunk_all, experts_all = lane.get_state(ptr["state_cid"])
        cfg = H.HarnessConfig.from_json(meta["cfg"])
        if e >= len(experts_all):
            _flush("[contrib %s] FATAL: expert %d out of range (state has %d)" % (miner, e, len(experts_all)))
            return 5
        trunk = trunk_all
        expert_e = experts_all[e]
        model = H.build_model(cfg)                               # dims only; offline routing ignores others

        # ---- train H local steps on my domain-e shard (the REAL phase-2 kernel) ----
        Xtr, ytr = H.domain_splits(cfg, e)["train"]
        contribution = dc.train_expert_contribution(
            model, trunk, expert_e, e, (Xtr, ytr),
            H=cfg.H_inner, lr=cfg.lr, batch=cfg.B, seed=rnd * 100 + e + cfg.seed)

        # ---- publish per-expert + trunk pseudo-grad blobs, then a signed record (D1/D2) ----
        # STREAMING-SUBSET (#126): when NEURAHASH_SHARDDILOCO_STREAM_FRAC in (0,1), publish only the
        # rolling trunk fragment for this outer round (bytes ~ frac*full); flag off -> full trunk delta,
        # byte-identical to before. The coordinator reconstructs it in shard_merge_round.
        ecid = lane.put_delta(contribution["expert_delta"])
        trunk_wire = dm.stream_publish_trunk(contribution["trunk_delta"], rnd)
        tcid = lane.put_delta(trunk_wire)
        trunk_bytes = len(H.pack_arrays(trunk_wire, np.float16))
        sig = H.sign(key, ecid, rnd, miner)
        record = dict(miner=miner, expert=int(e), base_round=int(rnd), expert_cid=ecid,
                      trunk_cid=tcid, sig=sig, train_flops=float(contribution["train_flops"]),
                      trunk_bytes=int(trunk_bytes))
        rname = H.contrib_name(rnd, miner)
        rec_cid = lane.put_json_named(rname, record)
        done_last = rnd
        rounds_done += 1
        _flush("[contrib %s] round %d: trained %d steps on domain %d, published expert_cid=%s... "
               "trunk_cid=%s... trunk_wire=%dB flops=%.3e"
               % (miner, rnd, cfg.H_inner, e, ecid[:12], tcid[:12], trunk_bytes,
                  contribution["train_flops"]))

        # Keep the record ADVERTISED until the coordinator advances the pointer past this round. The
        # content_store names.json is a read-modify-write shared file: concurrent named PUTs can drop a
        # name (lost update). Re-asserting the record whenever it falls out of the manifest self-heals
        # that race, so a clobber never stalls a round (mirrors a real miner keeping its delta available
        # until consumed). Cheap: only re-PUTs when the name is actually missing.
        t_pub = time.time()
        while time.time() - t_pub < args.round_wait:
            try:
                ptr2 = lane.read_pointer()
            except Exception:                                    # noqa: BLE001
                ptr2 = None
            if ptr2 is not None and (ptr2.get("done") or int(ptr2.get("round", rnd)) > rnd):
                break                                            # coordinator consumed this round
            try:
                man = lane.manifest()
                if man.get(rname, {}).get("sha256") != rec_cid:
                    lane.put_json_named(rname, record)
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(args.poll)

    _flush("[contrib %s] hit max-rounds=%d; exiting" % (miner, args.max_rounds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
