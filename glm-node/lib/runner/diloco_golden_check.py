"""runner_diloco_2026-10-08 -- review F4: the CROSS-MACHINE check of the --exchange diloco math, numpy only (NO torch is
imported, so this process can never touch a GPU). Run it on BOTH boxes against the SAME runner file and compare the
output: every digest line must be identical (the RESULT line's "all" hash folds them into one value).

It loads ONLY the definitions the diloco math needs from the runner's own source (dl_golden, dl_outer, dl_inner_F,
_dl_norm64, dl_digest, repl_golden_part, repl_bf16_decode, ... and whatever they reference, found by walking the AST),
byte for byte, so it tests the runner's code under each box's Python / numpy -- not a copy. It prints:
  1. G2c-dl: dl_golden() -- the digest every node puts in init_meta["dl_g2c"] at step 0 (24 outer + 8 inner steps);
  2. GLM size (P = 29,535,232 trainable fp32 values, the E5 dense surface): the sha256 of the INPUTS, then the digest
     after 3 chained outer steps (rows 150/125/125, then A+C 150/125, then B+C 125/125 -- the 3-node arm's
     row-weighted averaging, bf16 deltas) and after 4 inner AdamW steps (dl_inner_F, binding and non-binding clips).

This PC (Python 3.13):
    C:/Python313/python.exe -B scratchpad/runner_diloco_2026-10-08/diloco_golden_check.py
The 4060 box, node B (Python 3.10 glmenv; interpreter per scratchpad/n3repl_arm_s3_1004/n3repl_arm_launch.py:164),
from the directory holding the SAME runner file copied beside this script:
    C:\\Users\\User\\glmenv\\Scripts\\python.exe -B diloco_golden_check.py --runner glm47_h1x_chain.py
On the 4060 do NOT set CUDA_VISIBLE_DEVICES=-1 (memory 4060-cpu-only-crash-nvidia-dll-unload-2026-09-29: -1 plus a
CUDA probe crashes that box; this script never imports torch, so nothing needs hiding), and run it as a plain command
(never encoded PowerShell: AVG kills it). Its C: is a spinning disk: the first numpy import can take a minute.
Takes ~1-3 min and ~1.5 GB RAM. Exit 0 always unless the extraction fails (2); the COMPARISON is the operator's.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import struct
import sys
import threading
import time
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
P_GLM = 29535232
NEED = ("dl_golden", "dl_outer", "dl_inner_F", "dl_digest", "repl_golden_part", "repl_bf16_decode", "repl_state_zero",
        "repl_digest", "_repl_mix64", "REPL_EPS", "REPL_CHUNK", "REPL_LANES")


def extract(path):
    """-> (namespace, {name: sha256 of its source segment}) for NEED and its transitive top-level dependencies."""
    src = open(path, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    tree = ast.parse(src)
    defs = OrderedDict()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defs[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            defs[node.targets[0].id] = node
    want, todo = set(), list(NEED)
    while todo:
        n = todo.pop()
        if n in want or n not in defs:
            continue
        want.add(n)
        for sub in ast.walk(defs[n]):
            if isinstance(sub, ast.Name) and sub.id in defs and sub.id not in want:
                todo.append(sub.id)
    missing = [n for n in NEED if n not in want]
    if missing:
        raise SystemExit("extraction: %s not found in %s" % (missing, path))
    ns = {"__name__": "diloco_golden_extract", "hashlib": hashlib, "math": math, "os": os, "sys": sys, "json": json,
          "re": re, "struct": struct, "time": time, "threading": threading, "OrderedDict": OrderedDict}
    shas = OrderedDict()
    for name, node in defs.items():                       # source order: dependencies first, as in the runner
        if name in want:
            seg = ast.get_source_segment(src, node)
            shas[name] = hashlib.sha256(seg.encode("utf-8")).hexdigest()
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns, shas


def main():
    ap = argparse.ArgumentParser(description="diloco cross-machine golden check (numpy only)")
    ap.add_argument("--runner", default=os.path.join(HERE, "glm47_h1x_chain.py"))
    ap.add_argument("--skip-glm", action="store_true", help="only the G2c-dl golden (seconds)")
    a = ap.parse_args()
    t0 = time.time()
    import numpy as np
    rsha = hashlib.sha256(open(a.runner, "rb").read()).hexdigest()
    try:
        G, shas = extract(a.runner)
    except SystemExit as e:
        print("EXTRACTION FAILED: %s" % e)
        return 2
    out = OrderedDict()
    print("host %s | python %s | numpy %s | runner %s | %d definitions extracted (sha of their sources %s)"
          % (platform.node(), platform.python_version(), np.__version__, rsha,
             len(shas), hashlib.sha256(json.dumps(shas).encode("ascii")).hexdigest()[:16]), flush=True)
    out["g2c_dl"] = G["dl_golden"]()
    print("G2c-dl dl_golden()                 %s  (%.1f s)" % (out["g2c_dl"], time.time() - t0), flush=True)
    if not a.skip_glm:
        P = P_GLM
        th = G["_repl_mix64"](P, 0x2545F4914F6CDD1D + P)
        bits = (((th >> np.uint64(5)) & np.uint64(1)).astype(np.uint32) << np.uint32(31)) | \
            (np.uint32(121) << np.uint32(23)) | (th & np.uint64(0x7FFFFF)).astype(np.uint32)
        theta = bits.view(np.float32).copy()
        del th, bits
        parts = [G["repl_golden_part"](1000 + k, k, P, binding=bool(k % 2))[1] for k in range(3)]
        out["in_theta0"] = hashlib.sha256(memoryview(theta).cast("B")).hexdigest()
        out["in_parts"] = [hashlib.sha256(memoryview(u).cast("B")).hexdigest() for u in parts]
        print("GLM P %d inputs: theta0 %s | parts %s" % (P, out["in_theta0"][:16], [s[:16] for s in out["in_parts"]]),
              flush=True)
        for i, (rows, idx) in enumerate((((150, 125, 125), (0, 1, 2)), ((150, 125), (0, 2)), ((125, 125), (1, 2)))):
            theta, _acc, info = G["dl_outer"](theta, [(r, parts[j]) for r, j in zip(rows, idx)], 1.0)
            d = G["dl_digest"](theta)
            out["outer_%d" % (i + 1)] = d
            print("GLM outer step %d rows %-15s weights %s rel %.6e  digest %s  (%.1f s)"
                  % (i + 1, "/".join(str(r) for r in rows), ["%.8f" % w for w in info["weights"]], info["rel"], d,
                     time.time() - t0), flush=True)
        del parts
        st = G["repl_state_zero"](theta)
        hp = {"b1": 0.9, "b2": 0.999, "eps": G["REPL_EPS"], "wd": 0.01, "clip": 1.0}
        for i in range(4):
            g32 = G["repl_bf16_decode"](G["repl_golden_part"](2000 + i, 0, P, binding=bool(i % 2))[1]).copy()
            res = G["dl_inner_F"](st, g32, 1e-5, hp)
            d = G["repl_digest"](st)
            out["inner_%d" % (i + 1)] = d
            print("GLM inner step %d norm %.9e coef %.9e  digest %s  (%.1f s)" % (i + 1, res["norm"], res["coef"], d,
                                                                                   time.time() - t0), flush=True)
    out["all"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode("ascii")).hexdigest()
    print("RESULT " + json.dumps(OrderedDict([("runner_sha256", rsha), ("python", platform.python_version()),
                                              ("numpy", np.__version__), ("all", out["all"]),
                                              ("g2c_dl", out["g2c_dl"])])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
