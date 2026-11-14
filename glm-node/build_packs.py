#!/usr/bin/env python3
"""Build the GLM-4.7-Flash files a glm-node needs, from PUBLIC Hugging Face repos, and prove they are
BYTE-IDENTICAL to the fleet's by sha256 (manifests/packs_manifest.json).

What it makes (under --home, default env NEURAHASH_GLM_HOME or ~/glmnode):
  packs/layer_1.safetensors .. layer_46.safetensors   46 x 1,207,959,728 B = 55.6 GB (the routed experts of each
                                                      MoE layer, streamed by the node every training step)
  base/pieces/trunk.safetensors                       5,668,965,432 B (embeddings, attention, shared experts)
  base/model_manifest.json                            292,376 B (piece map the loader reads)
  base/config/{config.json, tokenizer.json, tokenizer_config.json}

Sources (pinned revisions -- nothing floats):
  model  : zai-org/GLM-4.7-Flash @ 7dd20894a642a0aa287e9827cb1a1f7f91386b67 (MIT). The packs are a pure byte
           re-arrangement of its BF16 expert tensors: no arithmetic, no torch, so the result is deterministic.
  trunk  : whitegreenstudios888/neurahash-data @ ad2b69ed... file glm47_pieces_100mb/pieces/trunk_full_679keys.safetensors.
           (NOT that repo's pieces/trunk.safetensors, which is a stripped 4.32 GB trunk the fleet would refuse.)

Disk: the build streams ONE source shard at a time (download -> build its pack -> verify -> delete the shard), so
the peak is ~62 GB of results + ~2.6 GB in flight; ~65 GB is downloaded in total (60 GB of model shards + the
5.7 GB trunk). Use an NVMe SSD.

    python glm-node/build_packs.py                          # everything; safe to re-run, resumes and skips verified work
    python glm-node/build_packs.py --verify-only            # re-hash what is on disk against the manifest
    python glm-node/build_packs.py --base-only              # only trunk + manifest + config
    python glm-node/build_packs.py --source DIR             # build from an existing local snapshot of the model
    python glm-node/build_packs.py --prove-layer 1 --source DIR   # hash-only rebuild of one layer, writes nothing

Every pack is hashed while it is written and renamed into place only when its sha256 equals the fleet's; a
mismatch deletes the temp file and stops with a non-zero exit. Python 3.10+, standard library + huggingface_hub.
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import struct
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")       # measured: the Xet path can wedge silently on home links
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifests", "packs_manifest.json")
EXPERT_KEY = "model.layers.%d.mlp.experts.%d.%s.weight"
CHUNK = 8 << 20


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def die(msg, rc=2):
    log("ERROR: " + msg)
    sys.exit(rc)


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path, progress=None):
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
            n += len(b)
    return h.hexdigest()


def force_ipv4():
    """IPv6 to huggingface.co can be a blackhole on home links while IPv4 is instant (measured 2026-07-17)."""
    orig = socket.getaddrinfo

    def v4(host, port, family=0, *a, **kw):
        return orig(host, port, socket.AF_INET, *a, **kw)
    socket.getaddrinfo = v4


def free_bytes(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free


# ------------------------------------------------------------------------------------------------ downloads
def hf_fetch(repo_id, filename, revision, dest_dir, repo_type=None):
    """Download one file at a pinned revision into dest_dir (resumable); returns the local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die("huggingface_hub is not installed: python -m pip install huggingface_hub==1.10.2")
    os.makedirs(dest_dir, exist_ok=True)
    t0 = time.time()
    kw = {"repo_id": repo_id, "filename": filename, "revision": revision, "local_dir": dest_dir}
    if repo_type:
        kw["repo_type"] = repo_type
    last = None
    for attempt in range(1, 6):
        try:
            p = hf_hub_download(**kw)
            sz = os.path.getsize(p)
            log("  downloaded %s (%.2f GB in %.0f s)" % (filename, sz / 1e9, time.time() - t0))
            return p
        except Exception as e:                                            # noqa: BLE001
            last = e
            wait = min(120, 10 * attempt)
            log("  download of %s failed (try %d/5: %s: %s); retrying in %d s"
                % (filename, attempt, type(e).__name__, str(e)[:160], wait))
            time.sleep(wait)
    die("could not download %s from %s after 5 tries: %s" % (filename, repo_id, last))


def verified_copy(src, dst, want_sha, want_size, what, move=False):
    """Copy -- or, for our own download on the same drive, move -- src -> dst atomically, only if sha256 and size
    match. A wrong download is deleted so the next run fetches it again."""
    size = os.path.getsize(src)
    if size != want_size:
        if move:
            os.remove(src)
        die("%s: %s has %d bytes, the fleet's has %d" % (what, src, size, want_size))
    got = sha256_file(src)
    if got != want_sha:
        if move:
            os.remove(src)
        die("%s: sha256 %s != the fleet's %s (corrupt or wrong file: %s)" % (what, got[:16], want_sha[:16], src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if move and os.path.splitdrive(os.path.abspath(src))[0].lower() == os.path.splitdrive(os.path.abspath(dst))[0].lower():
        os.replace(src, dst)
        log("  OK %s -> %s (sha256 %s)" % (what, dst, want_sha[:16]))
        return
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    if sha256_file(tmp) != want_sha:
        os.remove(tmp)
        die("%s: the copy at %s does not hash like its source -- disk problem?" % (what, tmp))
    os.replace(tmp, dst)
    log("  OK %s -> %s (sha256 %s)" % (what, dst, want_sha[:16]))


def have_ok(path, want_sha, want_size):
    return os.path.isfile(path) and os.path.getsize(path) == want_size and sha256_file(path) == want_sha


# ------------------------------------------------------------------------------------------------ base
def build_base(man, base, source, dl_dir, trunk_from="", skip_trunk=False):
    m = man["model"]
    b = man["base"]
    # 1. config files from the model repo
    for name, ent in man["config_files"].items():
        dst = os.path.join(base, "config", name)
        if have_ok(dst, ent["sha256"], ent["size"]):
            log("  OK config/%s already verified" % name)
            continue
        src = os.path.join(source, name) if source else hf_fetch(m["repo_id"], name, m["revision"], dl_dir)
        verified_copy(src, dst, ent["sha256"], ent["size"], "config/" + name)
    # 2. the full trunk (5.67 GB) from the dataset repo, saved under the name the loader reads
    t = b["trunk"]
    dst = os.path.join(base, *t["local"].split("/"))
    if skip_trunk:
        log("  trunk SKIPPED (--skip-trunk)")
    elif have_ok(dst, t["sha256"], t["size"]):
        log("  OK %s already verified" % t["local"])
    elif trunk_from:
        verified_copy(trunk_from, dst, t["sha256"], t["size"], "trunk (from %s)" % trunk_from)
    else:
        if free_bytes(base) < t["size"] + (2 << 30):
            die("not enough free disk for the trunk under %s (need %.1f GB)" % (base, (t["size"] + (2 << 30)) / 1e9))
        tdir = os.path.join(dl_dir, "_dataset")
        src = hf_fetch(b["dataset_repo_id"], t["remote"], b["dataset_revision"], tdir, repo_type="dataset")
        verified_copy(src, dst, t["sha256"], t["size"], "trunk", move=True)
        if os.path.exists(src):
            os.remove(src)
    # 3. model_manifest.json: the published one describes the stripped trunk; the fleet's differs in exactly two
    #    fields, so derive it and prove it by sha256.
    mm = b["model_manifest"]
    dst = os.path.join(base, mm["local"])
    if have_ok(dst, mm["sha256"], mm["size"]):
        log("  OK model_manifest.json already verified")
    else:
        tdir = os.path.join(dl_dir, "_dataset")
        src = hf_fetch(b["dataset_repo_id"], mm["remote"], b["dataset_revision"], tdir, repo_type="dataset")
        with open(src, "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
        doc["model_root"] = mm["derive"]["model_root"]
        names = [p.get("piece") for p in doc.get("pieces", [])]
        if names.count("trunk") != 1:
            die("the published model_manifest.json has no single trunk piece")
        doc["pieces"][names.index("trunk")] = mm["derive"]["trunk_piece"]
        data = json.dumps(doc, indent=2).replace("\n", "\r\n").encode("utf-8")
        got = hashlib.sha256(data).hexdigest()
        if got != mm["sha256"] or len(data) != mm["size"]:
            die("derived model_manifest.json sha256 %s != the fleet's %s" % (got[:16], mm["sha256"][:16]))
        tmp = dst + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
        log("  OK model_manifest.json derived (sha256 %s)" % got[:16])


# ------------------------------------------------------------------------------------------------ packs
def read_st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 100 << 20:
            raise ValueError("%s: implausible safetensors header length %d" % (path, n))
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr, 8 + n


def pack_header(fmt):
    d = fmt["dims"]
    E, I, H = d["n_routed_experts"], d["moe_intermediate_size"], d["hidden_size"]
    down = E * H * I * 2
    gate_up = E * 2 * I * H * 2
    hdr = {"down": {"dtype": "BF16", "shape": [E, H, I], "data_offsets": [0, down]},
           "gate_up": {"dtype": "BF16", "shape": [E, 2 * I, H], "data_offsets": [down, down + gate_up]}}
    s = json.dumps(hdr, separators=(",", ":")).encode("ascii")
    s += b" " * ((8 - len(s) % 8) % 8)                 # safetensors pads its header to 8 bytes with spaces
    out = struct.pack("<Q", len(s)) + s
    if len(out) != fmt["data_start"] or hashlib.sha256(out).hexdigest() != fmt["header_sha256_first_176_bytes"]:
        die("internal: the pack header does not reproduce the fleet's (%s)" % hashlib.sha256(out).hexdigest()[:16])
    return out, down + gate_up


def pack_chunks(shard, L, fmt):
    """Yield the bytes of layer L's pack: header, then down[0..63], then (gate[E] + up[E]) for E in 0..63 --
    exactly what tools/glm_arc_eval.py stage_pack writes through torch/safetensors, as raw byte copies."""
    d = fmt["dims"]
    E, I, H = d["n_routed_experts"], d["moe_intermediate_size"], d["hidden_size"]
    hdr, base = read_st_header(shard)
    head, data_len = pack_header(fmt)

    def tensor(e, part, shape):
        k = EXPERT_KEY % (L, e, part)
        t = hdr.get(k)
        if t is None:
            raise KeyError("%s: tensor %s missing" % (os.path.basename(shard), k))
        if t["dtype"] != "BF16" or list(t["shape"]) != shape:
            raise ValueError("%s: %s is %s %s, want BF16 %s" % (os.path.basename(shard), k, t["dtype"], t["shape"], shape))
        a, b = t["data_offsets"]
        if b - a != shape[0] * shape[1] * 2:
            raise ValueError("%s: %s has %d bytes" % (os.path.basename(shard), k, b - a))
        return base + a, b - a

    plan_down = [tensor(e, "down_proj", [H, I]) for e in range(E)]
    plan_gu = [(tensor(e, "gate_proj", [I, H]), tensor(e, "up_proj", [I, H])) for e in range(E)]
    yield head
    with open(shard, "rb") as f:
        def read(off, n):
            f.seek(off)
            b = f.read(n)
            if len(b) != n:
                raise IOError("%s: short read at %d (%d of %d)" % (shard, off, len(b), n))
            return b
        for off, n in plan_down:
            yield read(off, n)
        for (go, gn), (uo, un) in plan_gu:
            yield read(go, gn)
            yield read(uo, un)


def build_pack(shard, L, fmt, want, out_path=None):
    """-> (sha256, bytes). With out_path: written to out_path.tmp, fsync'd, renamed only on a sha match."""
    h, n = hashlib.sha256(), 0
    tmp = (out_path + ".tmp") if out_path else None
    f = open(tmp, "wb") if tmp else None
    try:
        for b in pack_chunks(shard, L, fmt):
            h.update(b)
            n += len(b)
            if f:
                f.write(b)
        if f:
            f.flush()
            os.fsync(f.fileno())
            f.close()
            f = None
        got = h.hexdigest()
        if out_path:
            if got != want["sha256"] or n != want["size"]:
                os.remove(tmp)
                die("layer %d: built %d bytes sha256 %s, the fleet's is %d bytes %s -- NOT installed"
                    % (L, n, got[:16], want["size"], want["sha256"][:16]), rc=3)
            os.replace(tmp, out_path)
        return got, n
    finally:
        if f:
            f.close()
            if tmp and os.path.exists(tmp):
                os.remove(tmp)


def write_stamp(packs, man):
    files = {}
    for p in man["packs"]:
        fp = os.path.join(packs, p["file"])
        files[p["file"]] = {"sha256": p["sha256"], "size": p["size"], "mtime": os.path.getmtime(fp)}
    with open(os.path.join(packs, "PACKS_VERIFIED.json"), "w", encoding="utf-8") as f:
        json.dump({"ok": True, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "manifest": "manifests/packs_manifest.json",
                   "files": files}, f, indent=1)


def verify_packs(packs, man, layers):
    bad = []
    for p in man["packs"]:
        if p["layer"] not in layers:
            continue
        fp = os.path.join(packs, p["file"])
        if not os.path.isfile(fp):
            bad.append("%s missing" % p["file"])
            continue
        if os.path.getsize(fp) != p["size"]:
            bad.append("%s size %d" % (p["file"], os.path.getsize(fp)))
            continue
        got = sha256_file(fp)
        log("  %s %s %s" % ("OK  " if got == p["sha256"] else "BAD ", p["file"], got[:16]))
        if got != p["sha256"]:
            bad.append("%s sha256 %s != %s" % (p["file"], got[:16], p["sha256"][:16]))
    return bad


def parse_layers(spec):
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    bad = [x for x in out if not 1 <= x <= 46]
    if bad:
        die("--layers: %s outside 1..46" % bad)
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--home", default=os.environ.get("NEURAHASH_GLM_HOME", ""))
    ap.add_argument("--packs", default="", help="default <home>/packs")
    ap.add_argument("--base", default="", help="default <home>/base")
    ap.add_argument("--download-dir", default="", help="where source shards land while in use (default <home>/hf)")
    ap.add_argument("--source", default="", help="an existing local snapshot of zai-org/GLM-4.7-Flash @ the pinned "
                                                 "revision (no download; its files are never modified or deleted)")
    ap.add_argument("--layers", default="1-46")
    ap.add_argument("--keep-source", action="store_true", help="keep downloaded shards (~60 GB) after building")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--packs-only", action="store_true")
    ap.add_argument("--prove-layer", default="", help="hash-only rebuild of these layers from --source; writes nothing")
    ap.add_argument("--no-ipv4", action="store_true", help="do not force IPv4 for downloads")
    ap.add_argument("--trunk-from", default="", help="a local copy of the fleet's trunk (verified, then copied)")
    ap.add_argument("--skip-trunk", action="store_true", help="do not fetch the trunk now")
    a = ap.parse_args(argv)
    man = load_manifest()
    home = os.path.abspath(a.home or os.path.join(os.path.expanduser("~"), "glmnode"))
    packs = os.path.abspath(a.packs or os.path.join(home, "packs"))
    base = os.path.abspath(a.base or os.path.join(home, "base"))
    dl_dir = os.path.abspath(a.download_dir or os.path.join(home, "hf", "GLM-4.7-Flash"))
    source = os.path.abspath(a.source) if a.source else ""
    layers = parse_layers(a.layers)
    fmt = man["pack_format"]
    by_layer = dict((p["layer"], p) for p in man["packs"])
    shard_meta = dict((f["file"], f) for f in man["model"]["files"])
    if not a.no_ipv4:
        force_ipv4()
    log("glm-node build_packs | model %s @ %s | packs -> %s | base -> %s%s" % (
        man["model"]["repo_id"], man["model"]["revision"][:12], packs, base,
        (" | source " + source) if source else ""))

    if a.prove_layer:
        if not source:
            die("--prove-layer needs --source (a local snapshot)")
        ok = True
        for L in parse_layers(a.prove_layer):
            want = by_layer[L]
            shard = os.path.join(source, want["source_shard"])
            t0 = time.time()
            ssha = sha256_file(shard)
            sm = shard_meta[want["source_shard"]]
            log("source %s sha256 %s (HF etag %s) %s" % (want["source_shard"], ssha[:16], sm["sha256"][:16],
                                                        "MATCH" if ssha == sm["sha256"] else "MISMATCH"))
            got, n = build_pack(shard, L, fmt, want)
            same = got == want["sha256"] and n == want["size"]
            ok &= same and ssha == sm["sha256"]
            log("PROVE layer %d: rebuilt %d bytes sha256 %s | fleet %s | %s (%.0f s)" % (
                L, n, got, want["sha256"], "BYTE-IDENTICAL" if same else "DIFFERENT", time.time() - t0))
        return 0 if ok else 3

    if a.verify_only:
        bad = verify_packs(packs, man, layers)
        for name, ent in man["config_files"].items():
            p = os.path.join(base, "config", name)
            if not have_ok(p, ent["sha256"], ent["size"]):
                bad.append("base/config/%s" % name)
        for key in ("trunk", "model_manifest"):
            ent = man["base"][key]
            p = os.path.join(base, *ent["local"].split("/"))
            if not have_ok(p, ent["sha256"], ent["size"]):
                bad.append("base/%s" % ent["local"])
        if bad:
            log("VERIFY FAILED (%d): %s" % (len(bad), "; ".join(bad[:8])))
            return 3
        if layers == list(range(1, 47)):
            write_stamp(packs, man)
        log("VERIFY OK: %d packs + base files are byte-identical to the fleet's" % len(layers))
        return 0

    if not a.packs_only:
        log("== base (trunk + model_manifest.json + config)")
        build_base(man, base, source, dl_dir, trunk_from=a.trunk_from, skip_trunk=a.skip_trunk)
    if a.base_only:
        log("DONE (base only)")
        return 0

    log("== packs (%d layers)" % len(layers))
    os.makedirs(packs, exist_ok=True)
    verified = set()
    for L in layers:
        want = by_layer[L]
        out = os.path.join(packs, want["file"])
        if os.path.isfile(out) and os.path.getsize(out) == want["size"]:
            got = sha256_file(out)
            if got == want["sha256"]:
                log("layer %2d: already present and verified" % L)
                verified.add(L)
                continue
            log("layer %2d: present but sha256 %s is wrong -- rebuilding" % (L, got[:16]))
            os.remove(out)
        need = want["size"] + (shard_meta[want["source_shard"]]["size"] if not source else 0) + (2 << 30)
        if free_bytes(packs) < need:
            die("not enough free disk under %s for layer %d (need %.1f GB free)" % (packs, L, need / 1e9))
        shard = os.path.join(source, want["source_shard"]) if source else os.path.join(dl_dir, want["source_shard"])
        sm = shard_meta[want["source_shard"]]
        if not source:
            if not (os.path.isfile(shard) and os.path.getsize(shard) == sm["size"]):
                hf_fetch(man["model"]["repo_id"], want["source_shard"], man["model"]["revision"], dl_dir)
            ssha = sha256_file(shard)
            if ssha != sm["sha256"]:
                os.remove(shard)
                die("source %s sha256 %s != Hugging Face's %s: deleted, re-run to download it again"
                    % (want["source_shard"], ssha[:16], sm["sha256"][:16]))
        t0 = time.time()
        got, n = build_pack(shard, L, fmt, want, out_path=out)
        log("layer %2d: built %s (%.2f GB) sha256 %s == fleet, %.0f s" % (L, want["file"], n / 1e9, got[:16],
                                                                      time.time() - t0))
        verified.add(L)
        if not source and not a.keep_source:
            os.remove(shard)
    if verified == set(range(1, 47)):
        write_stamp(packs, man)                      # every pack hashed in THIS run, so the stamp is earned
    log("DONE: %d packs byte-identical to the fleet's under %s%s" % (
        len(verified), packs, "" if len(verified) == 46 else " (partial: --layers)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
