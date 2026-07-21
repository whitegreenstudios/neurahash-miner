#!/usr/bin/env python3
"""Fetch a tokenized id-split bundle from the content lane by CID and unpack it to .npy files.

WHY VIA THE LANE. Regenerating splits on each node needs the full corpus (125 MB for gutenberg
alone) plus the tokenizer, to produce ~1.5 MB of output -- and any drift in tokenizer version or
chunking would give two nodes DIFFERENT held-out sets, which silently invalidates every comparison
between them. Distributing the exact bytes by content address removes that whole class of bug: the
CID is the guarantee that every node trained and evaluated on identical data.

The bundle deliberately excludes the coordinator-secret `probe` split. A contributor must never
hold the pool the gate samples from -- that is the difference between a held-out gate and a
public benchmark it can overfit.

  python tools/fetch_ids_bundle.py --cid <sha256> --dest D:/glm_wan
  python tools/fetch_ids_bundle.py --name glm/ids/v1 --dest D:/glm_wan --url http://host:8710
"""
import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request

import numpy as np


def _get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("NEURAHASH_CONTENT_URL",
                                                    "http://47.84.93.96:8710"))
    ap.add_argument("--cid", default=None, help="sha256 of the bundle (preferred: self-verifying)")
    ap.add_argument("--name", default=None, help="lane name to resolve if --cid is not given")
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    cid = args.cid
    if not cid:
        if not args.name:
            raise SystemExit("give --cid (preferred) or --name")
        man = json.loads(_get(args.url.rstrip("/") + "/manifest").decode())
        if args.name not in man:
            raise SystemExit("name %r not on the lane" % args.name)
        cid = man[args.name]["sha256"]
        print("resolved %s -> %s" % (args.name, cid[:16]))

    body = _get("%s/o/%s" % (args.url.rstrip("/"), cid))
    got = hashlib.sha256(body).hexdigest()
    if got != cid:
        # A name lookup is mutable and unsigned; the CONTENT ADDRESS is what actually binds bytes
        # to identity, so verify it even when the lane handed us the object.
        raise SystemExit("CID MISMATCH: got %s, expected %s -- refusing to write" % (got[:16], cid[:16]))
    print("fetched %d bytes, sha256 verified" % len(body))

    os.makedirs(args.dest, exist_ok=True)
    z = np.load(io.BytesIO(body))
    for k in sorted(z.files):
        dom, split = k.rsplit("_", 1)
        out = os.path.join(args.dest, "ids_%s_%s.npy" % (dom, split))
        np.save(out, z[k])
        print("  wrote %-34s %s %s" % (os.path.basename(out), z[k].shape, z[k].dtype))
    print("\nDONE -> %s" % args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
