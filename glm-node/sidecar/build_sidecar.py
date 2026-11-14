#!/usr/bin/env python3
"""Build the P2P sidecar (Go, pion WebRTC) from sidecar/src -- reproducibly.

    python glm-node/sidecar/build_sidecar.py                 # this OS/arch -> glm-node/lib/runner/bin/
    python glm-node/sidecar/build_sidecar.py --goos linux --goarch amd64
    python glm-node/sidecar/build_sidecar.py --check         # build to a temp name, compare with the shipped exe

Needs Go; the exact toolchain go1.27.1 is fetched by Go itself (GOTOOLCHAIN=go1.27.1) when another version is
installed. Flags: CGO_ENABLED=0, -trimpath, -buildvcs=false, -ldflags "-s -w". With those, the output depends only on
the source, the go.sum-pinned modules and the toolchain -- not on the directory or a git checkout around it.

Provenance: on Windows amd64 this rebuilds the shipped bin/p2psidecar.exe byte-for-byte
(sha256 be38da9aac4f64a1857c38b37e1f163a70ed9282c4ae9753ca08363e2730c39b). The fleet's own nodes run
fcca157943bf795b98f0f51458ba6743f8020d38ba7805db3f4ea5b41c8f4333 = the SAME source built inside the operator's git
checkout, which stamps that checkout's commit into the binary (the only difference; `go version -m` shows it).
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
BIN = os.path.join(os.path.dirname(HERE), "lib", "runner", "bin")
WINDOWS_AMD64_SHA256 = "be38da9aac4f64a1857c38b37e1f163a70ed9282c4ae9753ca08363e2730c39b"
TOOLCHAIN = "go1.27.1"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goos", default="")
    ap.add_argument("--goarch", default="")
    ap.add_argument("--go", default=shutil.which("go") or "", help="the go binary (default: go on PATH)")
    ap.add_argument("--check", action="store_true", help="build to a temp file and compare with the shipped one")
    a = ap.parse_args()
    if not a.go:
        sys.exit("go is not installed: https://go.dev/dl/ (any recent version; it fetches %s itself)" % TOOLCHAIN)
    env = dict(os.environ, CGO_ENABLED="0", GOTOOLCHAIN=TOOLCHAIN, GOFLAGS="-mod=readonly")
    if a.goos:
        env["GOOS"] = a.goos
    if a.goarch:
        env["GOARCH"] = a.goarch
    goos = subprocess.run([a.go, "env", "GOOS"], env=env, capture_output=True, text=True).stdout.strip()
    goarch = subprocess.run([a.go, "env", "GOARCH"], env=env, capture_output=True, text=True).stdout.strip()
    name = "p2psidecar.exe" if goos == "windows" else "p2psidecar"
    os.makedirs(BIN, exist_ok=True)
    out = os.path.join(BIN, name + (".check" if a.check else ""))
    cmd = [a.go, "build", "-buildvcs=false", "-trimpath", "-ldflags", "-s -w", "-o", out, "."]
    print("building %s/%s with %s: %s" % (goos, goarch, TOOLCHAIN, " ".join(cmd)), flush=True)
    r = subprocess.run(cmd, cwd=SRC, env=env)
    if r.returncode != 0:
        sys.exit("go build failed (rc %d)" % r.returncode)
    got = sha256_file(out)
    print("built %s sha256 %s" % (out, got))
    if goos == "windows" and goarch == "amd64":
        print("reproducible build: %s (shipped windows/amd64 is %s...)" % (
            "MATCH" if got == WINDOWS_AMD64_SHA256 else "DIFFERENT", WINDOWS_AMD64_SHA256[:16]))
        if a.check:
            os.remove(out)
        return 0 if got == WINDOWS_AMD64_SHA256 else 1
    print("non-Windows build: glm_node.py checks a sidecar against the run card only when the card names a sha256 "
          "for %s/%s in p2p.exe_sha256; send this sha256 to the operator so the fleet can pin it" % (goos, goarch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
