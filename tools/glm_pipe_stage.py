#!/usr/bin/env python3
"""PIPELINE STAGE miner -- hold a layer range of GLM and forward activations for the fleet.

This is the per-miner half of the fleet-hosted rollout engine (tools/glm_pipe.py): load embed (+
layers) or just a layer slice (~1.1 GiB/layer -- same VRAM class as CE-lane training), then serve:
poll the content store for activations addressed to this stage, run the layers, publish for the
next stage. All-outbound, NAT-safe, keyless-compatible. Idle-exits after --idle-exit seconds.

Every launch is VRAM-capped BEFORE the first CUDA touch (project rule vram-cap-live-verified).
Env: Windows, C:/Python313/python.exe (NEVER .venv). ASCII stdout (cp1252 console).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _log(msg):
    sys.stderr.write("[glm-pipe-stage] %s\n" % msg)
    sys.stderr.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("NEURAHASH_CONTENT_URL", ""))
    ap.add_argument("--token", default=os.environ.get("NEURAHASH_CONTENT_TOKEN", ""))
    ap.add_argument("--run", required=True, help="pipeline run id (shared by all stages + driver)")
    ap.add_argument("--stage", type=int, required=True, help="stage index (0 = embed + first span)")
    ap.add_argument("--layers", required=True, help="layer span this stage holds, as lo:hi")
    ap.add_argument("--shard-dir", dest="shard_dir", required=True)
    ap.add_argument("--config-dir", dest="config_dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--idle-exit", dest="idle_exit", type=float, default=1800.0,
                    help="exit after this many seconds with no traffic (default %(default)s)")
    args = ap.parse_args(argv)
    if not args.url:
        raise SystemExit("ERROR: --url (content store) required")

    import torch                                             # lazy
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # HARD VRAM CAP before any CUDA allocation (vram-cap-live-verified).
    if str(device).startswith("cuda") and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
        cap = float(os.environ.get("NEURAHASH_VRAM_CAP_GB", "0") or 0) or max(2.0, total - 8.0)
        cap = min(cap, max(1.0, total - 2.0))
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        _log("VRAM cap %.1f of %.1f GiB" % (cap, total))

    lo, hi = (int(x) for x in args.layers.split(":"))
    role = "first" if args.stage == 0 else "mid"
    import glm_pipe as GP                                    # lazy (same dir)
    import sharddiloco_harness as H
    lane = H.ContentLane(args.url, args.token or "")
    model, cfg = GP.load_stage(args.shard_dir, args.config_dir or args.shard_dir, lo, hi,
                               device=device, role=role, log=_log)
    GP.run_stage(lane, args.run, args.stage, model, cfg, lo, hi, device,
                 log=_log, idle_exit_s=args.idle_exit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
