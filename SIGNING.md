# Signing releases — the NeuraHash miner auto-update trust root

The public miner auto-updates itself, but it will **only ever run code signed by the project's
release key**. This document is for the **operator** who cuts releases. It covers: generating the
release keypair, keeping the private key offline, pinning the public key into the client, and
publishing a signed release that the whole fleet picks up on its next check.

If you are a miner, you do not need any of this — just run the client; it verifies updates for you.

---

## The trust model in one paragraph

Every miner has a **pinned release public key** baked into `tools/self_update.py`
(`PINNED_RELEASE_PUBKEY`). To push an update you publish a small **signed manifest**
(`release.json`) at a hard-coded URL (GitHub raw of `main`). On startup (rate-limited to once / 6h)
each miner fetches that manifest, **verifies its secp256k1 signature against the pinned key**, and
only if it verifies **and** its `version` is strictly greater than the miner's local `VERSION` does
it `git checkout` the manifest's `git_commit` and re-exec. Anything else — bad signature, wrong
key, unreachable/tampered manifest, a commit that doesn't match after checkout, or a downgrade — is
logged as a warning and **ignored**; the miner keeps running the code it already has. The updater
never runs a shell command or a URL taken from the manifest; the only actions are `git fetch`,
`git checkout <hex-commit>`, and `pip install -r requirements.txt`.

The crypto is the repo's own signing library (`neura_l1.signing`, real secp256k1 via
`eth-account`) — no hand-rolled cryptography anywhere in the update path.

> **The pinned key shipped in this branch is a TEST key** (address
> `0x19E7E376E7C213B7E7e7e46cc70A5dD086DAff2A`, derived from the obviously-fake private key
> `0x1111…11`). It guards nothing. Before this protects a real fleet you MUST generate a real
> release keypair and swap the pinned constant (steps below).

---

## 1. Generate the release keypair (do this OFFLINE)

Do this on an air-gapped or otherwise trusted machine. The private key must never touch a
build/CI/hosting box.

```bash
# a fresh random secp256k1 keypair, using the repo's own signing lib
python -c "from neura_l1.signing import gen_account; a=gen_account(); \
print('ADDRESS:', a.address); print('PRIVKEY:', a.key.hex())"
```

- **ADDRESS** (starts `0x…`, 40 hex) → this is public; it becomes `PINNED_RELEASE_PUBKEY`.
- **PRIVKEY** (`0x…`, 64 hex) → this is the secret. Store it OFFLINE.

Keep the private key on an offline medium (hardware token, encrypted USB in a safe). Write it to a
one-line file, e.g. `release_key.hex`, only on the offline machine when signing. **Never commit it,
never paste it into a terminal on an online box, never put it in an environment variable on a
shared machine.** If it leaks, an attacker can push code to every miner — rotate immediately
(section 5).

## 2. Pin the public key into the client (one-time, and on every rotation)

Edit `tools/self_update.py` and replace the `PINNED_RELEASE_PUBKEY` constant with your real
release **address**:

```python
PINNED_RELEASE_PUBKEY = "0x<your real release address>"   # replace the TEST key
```

Commit that change through the normal review flow. Until this lands, the client trusts only the
test key and will reject your real releases (correctly).

## 3. Cut a signed release

1. Land the code you want miners to run on `main` and note its commit (`git rev-parse HEAD`).
2. Bump the repo-root **`VERSION`** file to the new version **in that same commit** (strictly
   greater than the shipped version — miners ignore anything not a forward move). Numeric dotted
   only, e.g. `0.2.0`.
3. On the **offline** machine, with the private key file present, sign a manifest:

```bash
python tools/sign_release.py \
    --version 0.2.0 \
    --commit  <the commit hash from step 1> \
    --key     release_key.hex \
    --out      release.json
```

   (`--commit` defaults to the current `HEAD`; `--published-ts` defaults to now. You can also read
   the key from an env var with `--key-env NEURAHASH_RELEASE_KEY` instead of a file.)

   The tool prints the signer address and whether it matches `PINNED_RELEASE_PUBKEY`. It must say
   **`pinned match : YES`** — if not, you signed with the wrong key or haven't pinned the real one.

4. **Publish `release.json` at the pinned URL** — commit it to the repo root on `main` so it is
   served at:

   `https://raw.githubusercontent.com/whitegreenstudios/neurahash-miner/main/release.json`

   (This URL is the hard-coded `MANIFEST_URL` in `tools/self_update.py`. If you host the repo
   elsewhere, change that one constant to match — never make the fetch URL manifest-controlled.)

Within one rate-limit interval (≤ 6h) every running miner fetches it, verifies it, sees the forward
version, checks out your commit, and re-execs onto the new code. Miners started fresh pick it up on
launch.

## 4. `release.json` format

```json
{
  "version": "0.2.0",
  "git_commit": "608b0dd46c55239d24915eaf0f649ca1a004fda9",
  "published_ts": 1721001600,
  "signature": "…secp256k1 signature (hex)…",
  "signer": "0x…"
}
```

- `version`, `git_commit`, `published_ts` are the signed payload (plus a fixed domain tag
  `neurahash-miner-release`). `signer` is informational only — verification **re-recovers** the
  signer from the signature and compares it to the pinned key, so a forged `signer` field changes
  nothing.
- `git_commit` must be a bare hex commit id (7–64 hex). No branches, tags, or refs.
- Tampering with any signed field makes the signature recover to a different address → rejected.

## 5. Rotating / revoking the key

If the private key is compromised, or on scheduled rotation:

1. Generate a new keypair (section 1).
2. Ship a **normal signed release with the OLD key** whose code carries the **new**
   `PINNED_RELEASE_PUBKEY` (section 2). Miners still trusting the old key will accept this update
   and, after re-exec, will trust only the new key.
3. From then on sign with the new key. Miners that missed the rotation window must be updated
   out-of-band (re-clone) — there is deliberately no way for a miner to trust a key it never pinned.

Because trust is anchored in the pinned constant, an attacker who compromises the repo/mirror
**cannot** push code: they can serve any `release.json` they like, but without the release private
key it will not verify, and the miner stays on its current code.

## 6. Operator checklist (per release)

- [ ] Code merged to `main`; `VERSION` bumped in the same commit (strictly greater).
- [ ] `tools/sign_release.py` run offline; output shows `pinned match : YES`.
- [ ] `release.json` committed to repo root on `main` (served at the pinned raw URL).
- [ ] Spot-check: `python -c "import json,tools.self_update as s; \
      print(s.verify_manifest(json.load(open('release.json'))))"` prints `(True, '0x…pinned…')`.
- [ ] Private key returned to offline storage; no copy left on an online machine.
