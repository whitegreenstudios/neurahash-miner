#!/usr/bin/env python3
"""Fetch the GLM base (trunk + the expert pieces you were assigned) from HuggingFace.

WHY OUTBOUND-FROM-HF AND NOT PUSHED TO YOU. A residential uplink cannot move GB-scale to a remote
node -- measured the hard way (memory multi-miner-runpod-lessons: a 4.5 GB push from home failed
twice, once by reset and once at ~250 KB/s). So every node pulls its own base from the CDN, and the
only thing that ever travels back is the <300 KB per-round delta.

WHAT YOU NEED, and why it is less than the whole 62 GB model: shardDiLoCo makes you responsible for
a few EXPERTS, not the whole network. You hold the shared trunk plus the piece(s) holding your
experts. The trunk is ~5.67 GB; each expert piece is ~94 MB.

INTEGRITY: every file is verified against `pieces_sha256.json` after download. A short or corrupted
file is deleted, not left on disk to fail confusingly later. Re-running skips files that already
verify, so an interrupted fetch resumes.

NETWORK FOOTGUN (measured on a residential link, 2026-07-17): huggingface_hub routes big files
through the Xet engine, which can wedge silently -- process alive, zero sockets, zero bytes, no
timeout, no error -- and IPv6 to huggingface.co can be a blackhole while IPv4 is instant. This
script disables Xet and forces IPv4 before importing huggingface_hub. If a download ever hangs at
0 bytes, that is the fingerprint.

  python tools/fetch_glm_base.py --dest D:/glm_base --pieces 0
  python tools/fetch_glm_base.py --dest D:/glm_base --pieces 0,1,2 --skip-trunk
"""
import argparse
import hashlib
import os
import shutil
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # must precede the huggingface_hub import
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

REPO = "whitegreenstudios888/neurahash-data"
PREFIX = "glm47_pieces_100mb"


def _force_ipv4():
    import socket
    orig = socket.getaddrinfo

    def v4(host, port, family=0, *a, **kw):
        return orig(host, port, socket.AF_INET, *a, **kw)
    socket.getaddrinfo = v4


def sha256_of(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(api, rel, dest_root, expect_sha=None, label="", local=None):
    """Download `rel` from the dataset repo, verifying sha256 when known.

    `local` is the path RELATIVE TO dest_root to write to. It is separate from `rel` because the
    repo nests everything under glm47_pieces_100mb/ while piece_loader expects a shard dir with
    pieces/ + the two manifests directly at its root -- copying the repo's layout verbatim produces
    a tree the loader cannot read.
    """
    out = os.path.join(dest_root, (local or rel).replace("/", os.sep))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and expect_sha:
        if sha256_of(out) == expect_sha:
            print("  [have] %s" % rel)
            return out
        print("  [bad ] %s -- sha mismatch, refetching" % rel)
        os.remove(out)
    elif os.path.exists(out) and not expect_sha:
        print("  [have] %s (no sha to check)" % rel)
        return out

    t0 = time.time()
    print("  [get ] %s %s" % (rel, label), flush=True)
    tmp = api.hf_hub_download(repo_id=REPO, repo_type="dataset", filename=rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copyfile(tmp, out)
    dt = max(time.time() - t0, 1e-6)
    mb = os.path.getsize(out) / 1e6
    if expect_sha:
        got = sha256_of(out)
        if got != expect_sha:
            os.remove(out)
            raise SystemExit("INTEGRITY FAIL on %s: sha256 %s != expected %s (file deleted)"
                             % (rel, got[:16], expect_sha[:16]))
        print("  [ok  ] %s  %.1f MB in %.0fs (%.2f MB/s)  sha OK" % (rel, mb, dt, mb / dt))
    else:
        print("  [ok  ] %s  %.1f MB in %.0fs (%.2f MB/s)" % (rel, mb, dt, mb / dt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="shard dir to populate (holds pieces/ + manifests)")
    ap.add_argument("--pieces", default="0", help="comma-separated expert piece indices to fetch")
    ap.add_argument("--skip-trunk", action="store_true", help="trunk already present and verified")
    args = ap.parse_args()

    _force_ipv4()
    from huggingface_hub import HfApi                      # noqa: E402 (after env + IPv4)
    api = HfApi()

    os.makedirs(args.dest, exist_ok=True)
    print("dest=%s  repo=%s" % (args.dest, REPO))

    import json
    fetch(api, "%s/model_manifest.json" % PREFIX, args.dest, local="model_manifest.json")
    fetch(api, "%s/pieces_sha256.json" % PREFIX, args.dest, local="pieces_sha256.json")
    shas = json.load(open(os.path.join(args.dest, "pieces_sha256.json"), encoding="utf-8"))
    if isinstance(shas, dict) and "files" in shas:
        shas = shas["files"]

    def sha_for(name):
        # pieces_sha256.json keys are EXTENSION-LESS ("trunk", "experts_0"), 603 of them, mapping
        # straight to a hex digest. Verified against the real file rather than assumed.
        stem = name[:-len(".safetensors")] if name.endswith(".safetensors") else name
        v = shas.get(stem, shas.get(name))
        return v.get("sha256") if isinstance(v, dict) else v

    want = ["trunk.safetensors"] if not args.skip_trunk else []
    want += ["experts_%d.safetensors" % int(p) for p in args.pieces.split(",") if p.strip() != ""]

    total = 0
    for name in want:
        p = fetch(api, "%s/pieces/%s" % (PREFIX, name), args.dest,
                  expect_sha=sha_for(name), local="pieces/%s" % name)
        total += os.path.getsize(p)
    print("\nDONE: %d file(s), %.2f GB under %s" % (len(want), total / 2 ** 30, args.dest))
    print("Point the contributor at it with --shard-dir %s" % args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
