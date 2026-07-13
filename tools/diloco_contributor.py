"""DiLoCo-over-IPFS: let a remote contributor (a Colab T4, another location) improve the model
WITHOUT a live low-latency link to the coordinator — the path that survives both a throttled relay and
a restrictive NAT (see docs and tools/ipfs_checkpoint.py).

THE IDEA. The live pool already does DiLoCo: workers train the trunk a few (H) steps, send a trunk
delta, and the coordinator averages accepted deltas into `global_trunk` with an outer step
(global_trunk += OUTER * mean(deltas)). This module does the SAME thing ASYNCHRONOUSLY over IPFS: a
contributor fetches the latest checkpoint by CID, trains the trunk LOCALLY for many inner steps, and
publishes an improved checkpoint back by CID. A merge step folds that contribution in with the same
outer-step math the pool uses — gated on a held-out improvement so a bad (or adversarial) contribution
is rejected, never averaged in blind.

Only tiny CIDs cross the metered VPS; the bulk model bytes ride IPFS. Nobody needs an inbound port.

Correctness contract (matches the pool's Phase-1 verified delta): the contributor trains ONLY the
trunk with experts FROZEN, so the delta lives in the same subspace the coordinator's outer step
expects; the merge recomputes the held-out loss and applies the delta only if it does not regress.
"""
import argparse
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root on path

import torch

from neurahash.base_checkpoint import load_checkpoint, save_checkpoint, checkpoint_path
from neurahash_torch.shard_verify import trunk_keys
from neurahash_torch.corpus_torch import build_data, get_batch, resolve_corpus_mode

# reuse the shipped IPFS helpers (fetch by CID, publish, tracker) without importing sharded_pool_node
_IC_SPEC = importlib.util.spec_from_file_location(
    "ipfs_checkpoint", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ipfs_checkpoint.py"))
ic = importlib.util.module_from_spec(_IC_SPEC)
_IC_SPEC.loader.exec_module(ic)

OUTER = float(os.environ.get("NEURAHASH_DILOCO_OUTER", "0.7"))     # matches the pool's DiLoCo outer step
WEIGHT_DECAY = float(os.environ.get("NEURAHASH_WD", "0.01"))


_MEM_DEBUG = os.environ.get("NEURAHASH_MEM_DEBUG", "") not in ("", "0", "false", "False")


def _mem_probe(label):
    """NEURAHASH_MEM_DEBUG=1 (env, default OFF): print one CUDA memory line to stderr. Unset is a true
    no-op -- not even a torch.cuda call happens -- so this costs nothing on the hot path. Built to
    diagnose the 8GB-card VRAM fit (train_contribution): run with this on to see exactly which stage
    (load / base_trunk capture / eval / bf16 cast / optimizer / backward / step) sets the peak."""
    if not _MEM_DEBUG or not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[mem_debug] {label}: allocated={alloc:.3f}GB reserved={reserved:.3f}GB peak={peak:.3f}GB",
          file=sys.stderr)


def _resolve_train_dtype():
    """NEURAHASH_TRAIN_DTYPE (env): unset/"fp32"/"float32" -> None, meaning train_contribution takes
    the EXACT path it always has (whatever dtype load_checkpoint built the model in -- today always
    fp32; see pool_model._default_dtype). "bf16"/"bfloat16" -> torch.bfloat16: train_contribution casts
    the WHOLE model to bfloat16 before the training loop, so params + grads + AdamW moments all shrink
    to ~half -- the fix that lets a real dense base's contribute step fit an 8GB card (fp32 params
    ~2.4GB + AdamW moments ~4.8GB + grads ~2.4GB ~= 9.6GB for Qwen3-0.6B; bf16 halves the params/grads/
    moments). The checkpoint is upcast back to fp32 before it is saved (see train_contribution) so a
    bf16-trained contribution still merges cleanly against the fp32 base. Any other value fails loud
    (typo protection -- silently training in the wrong precision would be worse than crashing)."""
    v = (os.environ.get("NEURAHASH_TRAIN_DTYPE", "") or "").strip().lower()
    if v in ("", "fp32", "float32"):
        return None
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"NEURAHASH_TRAIN_DTYPE={v!r} not supported (use 'fp32' or 'bf16')")


def _eval_trunk(model, val_data, block, batch=64, iters=8, seed=999):
    """Held-out loss of the model's CURRENT weights (forward-only).

    NEURAHASH_EVAL_MICROBATCH (env, int): unset/0 = byte-identical to before (one forward per eval
    batch, the `else` branch below). When set to k>0, each `batch`-sized eval batch is split into
    chunks of at most k rows and the forward runs once per chunk instead of once for the whole
    batch -- this bounds the peak fp32 logits tensor to k*block*vocab elements instead of
    batch*block*vocab (the OOM: 64*512*151936 fp32 logits ~= 20 GiB for the Qwen0.6B-vocab base).
    Chunk losses are combined into the SAME scalar the unchunked path produces: F.cross_entropy's
    reduction is a MEAN over every (row, token) element and block_size is constant across chunks,
    so weighting each chunk's mean loss by its row count and dividing by the full batch size
    recovers the exact full-batch mean (no cross-batch term exists anywhere else in the forward:
    attention is per-sequence, and this trunk-only checkpoint has no MoE aux loss)."""
    microbatch = int(os.environ.get("NEURAHASH_EVAL_MICROBATCH", "0") or 0)
    model.eval()
    g = torch.Generator(device=val_data.device); g.manual_seed(seed)
    tot = 0.0
    with torch.inference_mode():
        for _ in range(iters):
            x, y = get_batch(val_data, block, batch, val_data.device, generator=g)
            if microbatch > 0 and microbatch < x.size(0):
                acc, n_seen = 0.0, 0
                for s in range(0, x.size(0), microbatch):
                    xc, yc = x[s:s + microbatch], y[s:s + microbatch]
                    _, loss_c = model(xc, yc)
                    acc += loss_c.item() * xc.size(0)
                    n_seen += xc.size(0)
                tot += acc / n_seen
            else:
                _, loss = model(x, y)
                tot += loss.item()
    model.train()
    return tot / iters


def train_contribution(ckpt_path, out_path, *, steps=200, lr=3e-4, batch=32, device="cpu", seed=0):
    """Load `ckpt_path`, train the TRUNK for `steps` inner steps (experts frozen), and write an improved
    checkpoint to `out_path` (same format, model_state + global_trunk updated). Returns a dict with the
    held-out loss before/after and the round. This is the contributor's local-SGD phase."""
    loaded = load_checkpoint(ckpt_path, device=device)
    if loaded is None:
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    model, arch0, vocab, E, rnd = (loaded["model"], loaded["arch0"], loaded["vocab"],
                                   loaded["E"], loaded["round"])
    # (8GB-card VRAM fit, root cause) load_checkpoint() fully materializes global_trunk on `device` --
    # for a dense Qwen base (n_experts=0) trunk_keys returns EVERY key, so this is a WHOLE-MODEL-sized
    # fp32 GPU copy. train_contribution never reads it (it builds its OWN base_trunk below), so it is
    # pure dead weight -- drop it UNCONDITIONALLY and IMMEDIATELY, before base_trunk/val_before run.
    # Measured root cause of the original 9.72GB peak: this used to be dropped only under
    # NEURAHASH_TRAIN_DTYPE=bf16, and only AFTER base_trunk was already cloned and val_before had
    # already run -- by then the peak (model + global_trunk + base_trunk, three live fp32 copies of the
    # same weights) had already happened, so the LATER bf16 cast could never bring the peak back down.
    loaded.pop("global_trunk", None)
    _mem_probe("after load_checkpoint (global_trunk dropped)")
    block = int(arch0.get("block_size", 128))
    corpus_mode = resolve_corpus_mode()
    # `seed` (CLI --seed, default 0): threaded into build_data so a non-default seed also diversifies
    # the toy-corpus text (build_data's seed only matters for that fallback branch -- build_qwen_bpe_data
    # / build_real_data / build_grounding_data ignore it). Default 0 reproduces the prior hardcoded
    # `seed=0` exactly, so this is byte-identical to before when --seed is omitted.
    _tok, train_data, val_data = build_data(device, seed=seed, mode=corpus_mode)
    # NEURAHASH_TRAIN_BATCH (env, int): unset/0 = byte-identical to before (`batch` as passed in,
    # e.g. the CLI's --batch / this function's default). Set to override the training batch size
    # (same OOM fix as NEURAHASH_EVAL_MICROBATCH, applied to the training forward's logits tensor).
    batch = int(os.environ.get("NEURAHASH_TRAIN_BATCH", "0") or 0) or batch

    sd = model.state_dict()
    # (8GB-card VRAM fit) base_trunk is captured straight to CPU (`.to("cpu", copy=True)`, a plain
    # device->host copy) instead of `.detach().clone()` (which would allocate a second WHOLE-MODEL fp32
    # copy ON the GPU, alongside the live model, right before the eval below). base_trunk is only ever
    # read again at the very end for the CPU-only npz delta, so it never needs to be GPU-resident.
    base_trunk = {k: sd[k].detach().to("cpu", copy=True) for k in trunk_keys(sd)}
    _mem_probe("after base_trunk capture (CPU-direct)")
    val_before = _eval_trunk(model, val_data, block)
    _mem_probe("after val_before eval")

    # NEURAHASH_TRAIN_DTYPE=bf16 (8GB-card fit): cast the WHOLE model to bfloat16 here, AFTER capturing
    # base_trunk/val_before (which stay fp32, matching the untouched base file) but BEFORE the training
    # loop, so params/grads/AdamW moments all train in bf16. train_dtype is None (unset/"fp32") ==
    # this whole block is skipped -> byte-identical code path to before.
    #   * `sd` (captured above): in the untouched fp32 path it's a VIEW sharing storage with the live,
    #     in-place-updated params (zero extra cost, which is why this was never a problem before). But
    #     model.to(dtype=...) below MUST allocate new per-param storage (fp32 and bf16 differ in
    #     element width, so the cast can't be truly in-place) -- an un-dropped `sd` would keep the OLD
    #     fp32 storage of every param alive (orphaned, unused) for the rest of this function.
    train_dtype = _resolve_train_dtype()
    if train_dtype is not None:
        del sd
        model = model.to(dtype=train_dtype)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _mem_probe("after bf16 cast + empty_cache")
        if _MEM_DEBUG:
            print(f"[mem_debug] sample param dtype after cast: {next(model.parameters()).dtype}",
                  file=sys.stderr)

    is_trunk = lambda nm: ".moe.experts." not in nm
    for nm, p in model.named_parameters():
        p.requires_grad_(is_trunk(nm))
    opt = torch.optim.AdamW([p for nm, p in model.named_parameters() if is_trunk(nm)],
                            lr=lr, weight_decay=WEIGHT_DECAY)
    _mem_probe("after optimizer creation")
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    last = 0.0
    for i in range(steps):
        x, y = get_batch(train_data, block, batch, device, generator=gen)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        if i == 0:
            _mem_probe("after first backward")
        opt.step()
        if i == 0:
            _mem_probe("after first opt.step")
            if _MEM_DEBUG:
                p0 = next(p for nm, p in model.named_parameters() if is_trunk(nm))
                st = opt.state.get(p0, {})
                if "exp_avg" in st:
                    print(f"[mem_debug] exp_avg dtype after step1: {st['exp_avg'].dtype}", file=sys.stderr)
        last = loss.item()

    val_after = _eval_trunk(model, val_data, block)
    if train_dtype is not None:
        # MERGE COMPATIBILITY: upcast the trained weights back to fp32 before save_checkpoint, so the
        # delta merge_contribution computes (contrib_trunk - base_trunk) stays fp32-clean against the
        # untouched fp32 base. Move off the GPU FIRST (still bf16), then upcast on CPU -- a transient
        # CPU-only upcast, so this never adds VRAM pressure on top of the training peak we just paid.
        model = model.to(device="cpu")
        model = model.to(dtype=torch.float32)
    # new global_trunk = the trained trunk (the merge computes the delta vs the base it started from)
    sd = model.state_dict()
    new_trunk = {k: sd[k].detach().clone() for k in trunk_keys(sd)}
    save_checkpoint(out_path, model=model, global_trunk=new_trunk, E=E, round=rnd,
                    ledger=loaded["ledger"], stats=loaded["stats"], arch0=arch0, vocab=vocab)
    # ALSO save the trunk DELTA (trained - base) as a safe .npz — this is what the coordinator's merge
    # fetches + folds in (numpy, allow_pickle=False, no code-exec risk; smaller than the full checkpoint).
    # NEURAHASH_SKIP_DELTA (env): unset/0 = byte-identical to before (npz always written). Set to skip
    # the npz for LOCAL contribute+merge loops: merge_contribution loads the full checkpoints and
    # recomputes the delta itself, so the npz's only consumer is publish_delta (the IPFS return path) —
    # skipping saves ~1/3 of the disk per contribution (observed ENOSPC mid-npz write on a tight disk).
    #
    # NEURAHASH_DELTA_COMPRESS (env): unset/0 = byte-identical to before (no compressed file, the two
    # extra return keys are None). Set to ALSO write a compact top-k int8 compressed delta
    # `<out>.delta.q8k.bin` via neurahash.delta_codec (target <10 MB) — the artifact small enough for a
    # NAT'd contributor to PUSH outbound instead of SERVE inbound (RunPod test 2026-07-10). Merged back
    # by the `merge-delta` path. NEURAHASH_DELTA_TOPK overrides the kept-fraction (default delta_codec
    # DEFAULT_TOPK). base_trunk is always CPU; new_trunk may be GPU-resident in the plain fp32 path
    # (train_dtype is None), so move each new_trunk tensor to CPU BEFORE subtracting — a bare
    # `new_trunk[k] - base_trunk[k]` would raise a cross-device RuntimeError when device="cuda".
    skip_full = os.environ.get("NEURAHASH_SKIP_DELTA", "") not in ("", "0", "false", "False")
    compress_on = os.environ.get("NEURAHASH_DELTA_COMPRESS", "") not in ("", "0", "false", "False")
    delta_path = out_path + ".delta.npz"
    compressed_delta_path = None
    compressed_delta_bytes = None
    if (not skip_full) or compress_on:
        # compute the CPU fp32 trunk delta once, reuse for the full npz and/or the compressed artifact
        delta_dict = {k: (new_trunk[k].detach().cpu().float() - base_trunk[k].float())
                      for k in new_trunk}
        if not skip_full:
            import numpy as _np
            _np.savez(delta_path, **{k: v.numpy() for k, v in delta_dict.items()})
        else:
            delta_path = None
        if compress_on:
            from neurahash import delta_codec
            topk = float(os.environ.get("NEURAHASH_DELTA_TOPK", "") or delta_codec.DEFAULT_TOPK)
            compressed_delta_path = out_path + ".delta.q8k.bin"
            compressed_delta_bytes = delta_codec.save_compressed_delta(
                compressed_delta_path, delta_dict, topk)
    else:
        delta_path = None
    return {"round": rnd, "steps": steps, "train_loss": last,
            "val_before": val_before, "val_after": val_after,
            "improved": val_after < val_before, "delta_path": delta_path,
            "compressed_delta_path": compressed_delta_path,
            "compressed_delta_bytes": compressed_delta_bytes}


def _miner_account():
    """Resolve the miner's wallet signing identity for GAP1 wallet-signed contributions, or None.

    NEURAHASH_MINER_KEY (env) = path to a secp256k1 private-key file in the SAME raw-hex format the
    pool's load_or_create_key (sharded_pool_node.py) writes: the file's entire contents are the private
    key hex. If the path EXISTS it is loaded; if it does NOT it is CREATED (fresh keypair, parent dirs
    made, best-effort 0600) so a first run self-provisions a stable identity that survives restarts.
    Unset -> None, and the contribution record stays UNSIGNED (pre-GAP1 behavior, byte-identical record).
    A key created here can be reused as a pool worker key (and vice-versa) — same on-disk format."""
    key_path = os.environ.get("NEURAHASH_MINER_KEY", "").strip()
    if not key_path:
        return None
    from neura_l1.signing import account_from_key, gen_account
    if os.path.exists(key_path):
        with open(key_path) as f:
            return account_from_key(f.read().strip())
    parent = os.path.dirname(os.path.abspath(key_path))
    os.makedirs(parent, exist_ok=True)
    acct = gen_account()
    with open(key_path, "w") as f:
        f.write(acct.key.hex())
    try:
        os.chmod(key_path, 0o600)                              # best-effort secrecy (no-op semantics on Windows)
    except Exception:                                         # noqa: BLE001 — never fail publish over a chmod
        pass
    return acct


def publish_delta(delta_path, contributor, base_round, *, val_before=None, val_after=None,
                  registry_url=None, registry_token=None):
    """Pin the trunk-delta .npz to IPFS/Pinata and REGISTER a tiny record under the stable content-store
    name 'contrib-<contributor>' so the coordinator's merge poller finds it. Returns the delta CID.
    `registry_url`/`registry_token` default to NEURAHASH_DILOCO_MERGE_URL / NEURAHASH_CONTENT_TOKEN."""
    import json as _json
    import urllib.request as _url
    cid = ic.publish(delta_path) if not ic._pinata_jwt() else ic.pin_file_to_pinata(
        delta_path, name=f"neurahash-contrib-{contributor}")
    registry_url = (registry_url or os.environ.get("NEURAHASH_DILOCO_MERGE_URL", "")).rstrip("/")
    registry_token = registry_token or os.environ.get("NEURAHASH_CONTENT_TOKEN", "")
    if registry_url:
        base_round_i = int(base_round)
        rec = {"contributor": contributor, "delta_cid": cid, "base_round": base_round_i,
               "val_before": val_before, "val_after": val_after}
        # GAP1 (docs/GO_PUBLIC_DESIGN.md): when a wallet key is configured, SIGN the record so the
        # coordinator can VERIFY who produced this delta (and, in GAP2, pay that verified address). The
        # canonical bytes are built by the ONE shared builder poll_contrib_records also uses, so signer
        # and verifier can never drift. No key -> the record stays unsigned (pre-GAP1, back-compat).
        _acct = _miner_account()
        if _acct is not None:
            from neura_l1.signing import sign_bytes
            from neurahash.contrib_message import contrib_canonical_message
            _msg = contrib_canonical_message(cid, base_round_i, contributor, val_before, val_after)
            rec["address"] = _acct.address
            rec["sig"] = sign_bytes(_acct, _msg)
        body = _json.dumps(rec).encode()
        import hashlib as _hl
        h = _hl.sha256(body).hexdigest()
        req = _url.Request(f"{registry_url}/o/{h}", data=body, method="PUT",
                           headers={"X-Auth": registry_token, "X-Name": f"contrib-{contributor}"})
        # WAN-robust PUT: explicit timeout + bounded retries w/ backoff (was a bare urlopen(timeout=30),
        # zero retries -- see ic._put_retry's docstring for the 2026-07-10 incident this closes: a
        # contributor's ~38 MB delta publish died on one silent timeout and stayed stuck for 19+ hours).
        ic._put_retry(lambda: _url.urlopen(req, timeout=ic.PUT_TIMEOUT).read(),
                      label=f"{registry_url}/o/{h} (contrib-{contributor})")
    return cid


def _apply_trunk_delta_gated(base, delta, out_path, *, outer=OUTER, device="cpu", accept_margin=0.0):
    """Shared accept-gate: fold a trunk `delta` (dict of torch tensors, one per trunk key) into the
    loaded `base` checkpoint with the pool's DiLoCo outer step, GATED on a held-out improvement.
    Evaluates base vs base + outer*delta on the held-out set and writes the merged checkpoint to
    `out_path` ONLY if merged held-out loss is <= base - accept_margin; on reject the base stands and
    `out_path` is NOT written. This is the single gate for BOTH merge paths (full-checkpoint
    merge_contribution and compressed merge_delta) — do not duplicate the eval/accept logic elsewhere."""
    model, arch0, vocab, E = base["model"], base["arch0"], base["vocab"], base["E"]
    block = int(arch0.get("block_size", 128))
    _tok, _train, val_data = build_data(device, seed=0, mode=resolve_corpus_mode())

    bt = base["global_trunk"]
    if set(bt) != set(delta):
        raise ValueError("trunk key mismatch between base and delta")

    sd = model.state_dict()
    val_base = _eval_trunk(model, val_data, block)
    merged_trunk = {k: bt[k] + outer * delta[k] for k in bt}
    merged_sd = dict(sd); merged_sd.update({k: merged_trunk[k] for k in merged_trunk})
    model.load_state_dict(merged_sd)
    val_merged = _eval_trunk(model, val_data, block)

    accept = val_merged <= (val_base - accept_margin)
    verdict = {"val_base": val_base, "val_merged": val_merged,
               "delta_norm": float(sum((delta[k].float() ** 2).sum() for k in delta) ** 0.5),
               "outer": outer, "accepted": bool(accept)}
    if accept:
        save_checkpoint(out_path, model=model, global_trunk=merged_trunk, E=E, round=base["round"],
                        ledger=base["ledger"], stats=base["stats"], arch0=arch0, vocab=vocab)
        verdict["out"] = out_path
    else:
        model.load_state_dict(sd)                                  # restore base (reject the contribution)
    return verdict


def merge_contribution(base_ckpt, contrib_ckpt, out_path, *, outer=OUTER, device="cpu",
                       accept_margin=0.0):
    """Fold a contributed CHECKPOINT into `base_ckpt` with the pool's DiLoCo outer step, GATED on a
    held-out improvement. Computes delta = contrib_trunk - base_trunk, then defers the eval/accept/save
    to _apply_trunk_delta_gated. Returns a verdict dict; on reject `out_path` is NOT written (the base
    stands). This is the trust gate: a bad/adversarial contribution is dropped."""
    base = load_checkpoint(base_ckpt, device=device)
    contrib = load_checkpoint(contrib_ckpt, device=device)
    if base is None or contrib is None:
        raise FileNotFoundError("base or contrib checkpoint missing")
    if base["arch0"] != contrib["arch0"] or base["vocab"] != contrib["vocab"]:
        raise ValueError("arch/vocab mismatch between base and contribution")

    bt, ct = base["global_trunk"], contrib["global_trunk"]
    if set(bt) != set(ct):
        raise ValueError("trunk key mismatch between base and contribution")
    delta = {k: (ct[k] - bt[k]) for k in bt}
    return _apply_trunk_delta_gated(base, delta, out_path, outer=outer, device=device,
                                    accept_margin=accept_margin)


def merge_delta(base_ckpt, delta_source, out_path, *, outer=OUTER, device="cpu", accept_margin=0.0):
    """Fold a COMPRESSED trunk delta (delta_codec .q8k.bin, local path or IPFS CID) into `base_ckpt`,
    reusing the SAME held-out accept gate as merge_contribution (_apply_trunk_delta_gated). This is the
    coordinator side of the push-through-NAT path: the contributor pushed a <10 MB compressed delta;
    here we load the base, decompress the delta, and gate it on real held-out improvement. Returns a
    verdict dict; on reject `out_path` is NOT written (the base stands)."""
    from neurahash import delta_codec
    base = load_checkpoint(base_ckpt, device=device)
    if base is None:
        raise FileNotFoundError("base checkpoint missing")

    # delta_source is a local file OR a bare CID — fetch by CID (IPFS) when it isn't on disk.
    delta_path = delta_source
    if not os.path.exists(delta_source):
        dest = os.path.join(os.path.dirname(os.path.abspath(out_path)) or ".", "fetched_delta.q8k.bin")
        gw = ic.fetch(delta_source, dest, verify_cid=False)
        print(f"[diloco] fetched compressed delta {delta_source[:16]}... via {gw}")
        delta_path = dest

    delta_np = delta_codec.load_compressed_delta(delta_path)       # {name: float32 ndarray}, safe (no pickle)
    bt = base["global_trunk"]
    missing = [k for k in bt if k not in delta_np]
    if missing:
        raise ValueError(f"compressed delta is missing {len(missing)} trunk key(s), e.g. {missing[0]!r}")
    # build a torch delta over EXACTLY the base's trunk keys (ignore any extras), matching dtype/device
    delta = {k: torch.as_tensor(delta_np[k], dtype=bt[k].dtype, device=bt[k].device).reshape(bt[k].shape)
             for k in bt}
    return _apply_trunk_delta_gated(base, delta, out_path, outer=outer, device=device,
                                    accept_margin=accept_margin)


# --------------------------------------------------------------------------- CLI
def _resolve_ckpt(arg, work_dir, device):
    """`arg` is a local checkpoint path, or a tracker URL/path (fetch its checkpoint_cid over IPFS), or a
    bare CID. Returns a local checkpoint path."""
    if os.path.exists(arg):
        return arg
    if arg.startswith(("http://", "https://")) or arg.endswith(".json"):
        doc = ic.read_tracker(arg)
        cid = doc["checkpoint_cid"]
    else:
        cid = arg                                                  # treat as a bare CID
    dest = os.path.join(work_dir, "fetched_ckpt.pt")
    gw = ic.fetch(cid, dest, verify_cid=False)
    print(f"[diloco] fetched checkpoint {cid[:16]}... via {gw}")
    return dest


def main():
    ap = argparse.ArgumentParser(description="DiLoCo-over-IPFS contributor / merger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contribute", help="fetch latest checkpoint, train locally, write + optionally publish")
    c.add_argument("source", help="checkpoint path | tracker url/path | CID")
    c.add_argument("--out", default="contrib_ckpt.pt")
    c.add_argument("--steps", type=int, default=200)
    c.add_argument("--lr", type=float, default=3e-4)
    c.add_argument("--batch", type=int, default=32)
    c.add_argument("--seed", type=int, default=0,
                   help="RNG seed for training-batch sampling (default 0 = legacy behavior; give "
                        "each contributor a distinct seed so parallel contributions are not the "
                        "same computation)")
    c.add_argument("--device", default="cpu")
    c.add_argument("--publish", action="store_true", help="publish the FULL contribution checkpoint to IPFS")
    c.add_argument("--publish-delta", action="store_true",
                   help="pin the trunk DELTA + register it for the coordinator's merge poller (the return path)")
    c.add_argument("--compress-delta", action="store_true",
                   help="ALSO write a compact top-k int8 compressed delta (<10MB) via delta_codec — the "
                        "artifact small enough for a NAT'd contributor to PUSH outbound (sets "
                        "NEURAHASH_DELTA_COMPRESS=1 for this run)")
    c.add_argument("--publish-compressed-delta", action="store_true",
                   help="with --publish-delta, pin the COMPRESSED delta (<10MB) instead of the full npz")
    c.add_argument("--name", default="0xcontrib", help="contributor id (registry slot contrib-<name>)")
    m = sub.add_parser("merge", help="fold a contribution into a base checkpoint (held-out gated)")
    m.add_argument("base"); m.add_argument("contrib"); m.add_argument("--out", default="merged_ckpt.pt")
    m.add_argument("--outer", type=float, default=OUTER); m.add_argument("--device", default="cpu")
    md = sub.add_parser("merge-delta",
                        help="fold a COMPRESSED trunk delta (path or CID) into a base checkpoint (held-out gated)")
    md.add_argument("base"); md.add_argument("delta", help="compressed delta path (.q8k.bin) or IPFS CID")
    md.add_argument("--out", default="merged_ckpt.pt")
    md.add_argument("--outer", type=float, default=OUTER); md.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.cmd == "contribute":
        if a.compress_delta:
            os.environ["NEURAHASH_DELTA_COMPRESS"] = "1"           # env-driven; the flag is a convenience
        src = _resolve_ckpt(a.source, os.path.dirname(os.path.abspath(a.out)) or ".", a.device)
        t0 = time.time()
        r = train_contribution(src, a.out, steps=a.steps, lr=a.lr, batch=a.batch, device=a.device,
                               seed=a.seed)
        print(f"[diloco] round {r['round']}: {r['steps']} steps, held-out {r['val_before']:.4f} -> "
              f"{r['val_after']:.4f} ({'IMPROVED' if r['improved'] else 'no-improve'}), {time.time()-t0:.0f}s")
        if r.get("compressed_delta_bytes") is not None:
            print(f"[diloco] compressed delta: {r['compressed_delta_path']} "
                  f"({r['compressed_delta_bytes'] / 1e6:.2f} MB)")
        if a.publish:
            cid = ic.publish(a.out)
            print(f"[diloco] published contribution CID: {cid}")
        if a.publish_delta:
            to_pub = (r.get("compressed_delta_path")
                      if (a.publish_compressed_delta and r.get("compressed_delta_path"))
                      else r["delta_path"])
            cid = publish_delta(to_pub, a.name, r["round"],
                                val_before=r["val_before"], val_after=r["val_after"])
            print(f"[diloco] published + registered trunk delta CID: {cid} (slot contrib-{a.name})")
    elif a.cmd == "merge-delta":
        v = merge_delta(a.base, a.delta, a.out, outer=a.outer, device=a.device)
        print(f"[diloco] merge-delta held-out {v['val_base']:.4f} -> {v['val_merged']:.4f} "
              f"(delta_norm {v['delta_norm']:.3f}, outer {v['outer']}): "
              f"{'ACCEPTED -> ' + v['out'] if v['accepted'] else 'REJECTED (base stands)'}")
    else:
        v = merge_contribution(a.base, a.contrib, a.out, outer=a.outer, device=a.device)
        print(f"[diloco] merge held-out {v['val_base']:.4f} -> {v['val_merged']:.4f} "
              f"(delta_norm {v['delta_norm']:.3f}, outer {v['outer']}): "
              f"{'ACCEPTED -> ' + v['out'] if v['accepted'] else 'REJECTED (base stands)'}")


if __name__ == "__main__":
    main()
