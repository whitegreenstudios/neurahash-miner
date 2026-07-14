# Content-addressed miner bundle

The NeuraHash miner bundle is distributed **decentralized-by-construction**: everyone can read it,
nobody can edit it. Integrity is verified by the **consumer** against a hash, so a tampered copy from
*any* source is rejected — which makes the sources interchangeable and droppable.

## The property

- **"Nobody can edit it"** — the bundle is content-addressed by `sha256` (and an IPFS CID). A tampered
  bundle has a different hash and is rejected at fetch time, no matter which mirror served it.
- **"Everyone can read it"** — it is fetched from any of several public seeds; no token needed.
- **Which bundle is canonical** — the coordinator publishes a *signed* pointer record naming the
  canonical `sha256`/`cid`. A consumer verifies that signature against the coordinator's **pinned
  address**, so a forged pointer (one not signed by the coordinator) is rejected even though its bytes
  would self-verify against the attacker's own hash. Signature pins **who** wrote the pointer; the hash
  pins **what** the bytes are.

## Seeds (interchangeable, droppable — availability, not trust)

| Seed | Where |
|---|---|
| **IPFS** | by CID — the decentralized backbone (grows as miners self-pin) |
| **VPS content-store** | `http://47.84.93.96:8710/o/<sha256>` |
| **HuggingFace** | `https://huggingface.co/datasets/whitegreenstudios888/neurahash-miner/resolve/main/bundle_<sha256>.zip` |

Removing any one seed only reduces availability, never correctness. HuggingFace / the VPS are convenience
seeds, **not** the decentralization — that comes from content-addressing + the signed (eventually
on-chain) pointer + miners self-pinning to IPFS.

## Fetch + verify (reference client)

`tools/bundle_pointer.py` is the client-side reference implementation (Python stdlib + `neura_l1.signing`):

```python
from tools import bundle_pointer as bp

# The coordinator publishes a SIGNED governance log (e.g. bundle_pointer_log.json from the content
# store). EXPECTED_COORDINATOR_ADDRESS is pinned OUT-OF-BAND (from an announcement / this repo), never
# read from the log itself.

# 1) verify WHO wrote the pointer: verify the signed log against the pinned coordinator address and take
#    the newest canonical pointer from it (a forged / tampered log raises).
record = bp.verified_bundle_record(signed_log, EXPECTED_COORDINATOR_ADDRESS)

# 2) verify WHAT the bytes are: fetch from the seeds in order, hash-checked; a wrong-bytes seed is
#    rejected and the next is tried. Returns the seed that served it.
bp.resolve_bundle(record, "bundle.zip")
```

If you already trust the pointer's source, `resolve_bundle` alone gives you hash-verified bytes;
`verified_bundle_record` adds the signature pin for an untrusted pointer channel.

> **Boundary (honest):** signature + hash guarantee **authenticity + integrity**, not **freshness** — a
> genuine but *older* pointer (a withheld tail) still verifies. A consumer that needs anti-rollback must
> ratchet on a monotonic timestamp/height it tracks itself.

## The current public bundle

The clean public miner bundle is mirrored on HuggingFace at
[`whitegreenstudios888/neurahash-miner`](https://huggingface.co/datasets/whitegreenstudios888/neurahash-miner):
`bundle_70837a999eec951c1b6442003c573fde84edbf56c274f3641c5f1d5d55922b54.zip` — a content-addressed mirror
of this repo. It is a drop-in seed; the same bytes are fetchable by hash from any seed above.
