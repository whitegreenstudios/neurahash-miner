#!/usr/bin/env python3
"""tools/sign_release.py -- OPERATOR tool: cut a SIGNED release manifest (release.json).

The operator runs this WITH THEIR OFFLINE RELEASE PRIVATE KEY to produce a `release.json` that
every miner will verify against the pinned public key in tools/self_update.py. Publishing that
release.json at the pinned MANIFEST_URL (GitHub raw of main) is what pushes an update to the fleet.

    python tools/sign_release.py --version 0.2.0 --key release_key.hex --out release.json

--commit defaults to the repo's current HEAD (`git rev-parse HEAD`), i.e. the exact commit you
want miners to run. --published-ts defaults to now. The private key is read from a FILE (a 0x-hex
32-byte secp256k1 key, one line) or from --key-env <ENVVAR>; it is never taken on the command line
(shell history) and never printed.

Crypto is the repo's own lib (neura_l1.signing, real secp256k1) -- this tool hand-rolls nothing.
It signs the SAME canonical bytes tools/self_update.verify_manifest recovers against, so a manifest
produced here verifies there and nowhere else. See SIGNING.md for keygen + the full release recipe.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from neura_l1.signing import account_from_key, sign_bytes, recover_bytes   # real secp256k1
from tools.self_update import canonical_manifest_bytes, parse_version, PINNED_RELEASE_PUBKEY


def _load_privkey(key_path, key_env):
    if key_env:
        val = os.environ.get(key_env, "").strip()
        if not val:
            raise SystemExit(f"ERROR: env var {key_env} is empty/unset")
        return val
    if not key_path:
        raise SystemExit("ERROR: provide --key <file> or --key-env <ENVVAR> (the release private key)")
    with open(key_path, "r", encoding="utf-8") as f:
        val = f.read().strip()
    if not val:
        raise SystemExit(f"ERROR: key file {key_path} is empty")
    return val


def _load_config(spec):
    """Return the v2 `config` object from a file path or a literal JSON string, or None if unset.

    Kept strict on purpose: this object is SIGNED and then applied by every miner as its network
    defaults, so a typo here propagates to the whole fleet with a valid signature on it. A non-object
    (list, string, number) is rejected rather than silently signed.
    """
    if spec is None:
        return None
    text = spec
    if os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as f:
            text = f.read()
    try:
        cfg = json.loads(text)
    except Exception as e:
        raise SystemExit(f"ERROR: --config is neither a readable JSON file nor valid JSON ({e})")
    if not isinstance(cfg, dict):
        raise SystemExit("ERROR: --config must be a JSON OBJECT (got %s)" % type(cfg).__name__)
    return cfg


def _full_commit(ref):
    """Resolve `ref` to a FULL 40-hex commit id via git, or exit.

    MEASURED 2026-07-21 against a real published release: signing a SHORT hash produces a manifest
    that is dead on arrival. The signature verifies and `git checkout <short>` succeeds, but the
    client then compares the resulting 40-char HEAD against the manifest's 7-char string, sees a
    mismatch, refuses to re-exec and rolls back -- every miner, silently, forever. The updater's
    commit regex accepts 7-64 hex, so nothing upstream catches it. Resolving here means an operator
    can still type a short hash and get a correct manifest.
    """
    try:
        out = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--verify", f"{ref}^{{commit}}"],
                                      encoding="utf-8", errors="replace").strip()
    except Exception as e:
        raise SystemExit(f"ERROR: could not resolve --commit {ref!r} to a full hash via git ({e}). "
                         f"Is it pushed and fetched in this clone?")
    if len(out) != 40 or any(c not in "0123456789abcdef" for c in out.lower()):
        raise SystemExit(f"ERROR: git resolved {ref!r} to {out!r}, which is not a 40-hex commit id")
    return out


def _default_commit():
    try:
        out = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                      encoding="utf-8", errors="replace").strip()
        return out
    except Exception:
        return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sign a NeuraHash miner release manifest (release.json).")
    ap.add_argument("--version", required=True,
                    help="the release version to publish (numeric dotted, e.g. 0.2.0). MUST be > the "
                         "VERSION currently shipped, or miners will (correctly) ignore it.")
    ap.add_argument("--commit", default=None,
                    help="git commit miners should run (default: current HEAD via git rev-parse)")
    ap.add_argument("--published-ts", type=int, default=None,
                    help="unix seconds (default: now)")
    ap.add_argument("--key", default=None, metavar="FILE",
                    help="path to the release PRIVATE key (0x-hex 32-byte secp256k1, one line)")
    ap.add_argument("--key-env", default=None, metavar="ENVVAR",
                    help="read the release private key from this environment variable instead of a file")
    ap.add_argument("--out", default=os.path.join(REPO, "release.json"),
                    help="output path for the signed manifest (default: repo-root release.json)")
    # ---- v2 optional fields. Omit BOTH and the output is byte-identical to a v1 manifest, so an
    # ordinary release keeps working exactly as before and every already-signed manifest stays valid.
    ap.add_argument("--min-client-version", default=None, metavar="VER",
                    help="v2: clients older than this REFUSE TO PUBLISH (they still train) and say so "
                         "by name. Use it when a release changes something the network requires, so an "
                         "out-of-date miner fails loudly instead of submitting work that gets rejected.")
    ap.add_argument("--config", default=None, metavar="FILE_OR_JSON",
                    help="v2: path to a JSON file, or a literal JSON object, carrying the signed "
                         "network config (merge_url / content_url / corpus_sha / protocol). Clients "
                         "apply it as DEFAULTS -- an explicitly set environment variable still wins.")
    args = ap.parse_args(argv)

    parse_version(args.version)                       # fail fast on a bad version string
    if args.min_client_version is not None:
        parse_version(args.min_client_version)        # same fail-fast: a malformed gate would brick publishing
    commit = _full_commit(args.commit) if args.commit else _default_commit()
    if not commit:
        raise SystemExit("ERROR: could not determine --commit (pass it explicitly)")
    ts = args.published_ts if args.published_ts is not None else int(time.time())

    acct = account_from_key(_load_privkey(args.key, args.key_env))
    manifest_body = {"version": str(args.version), "git_commit": str(commit), "published_ts": int(ts)}
    if args.min_client_version is not None:
        manifest_body["min_client_version"] = str(args.min_client_version)
    cfg = _load_config(args.config)
    if cfg is not None:
        manifest_body["config"] = cfg
    data = canonical_manifest_bytes(manifest_body)
    signature = sign_bytes(acct, data)

    # self-check: the manifest we are about to write must recover to THIS signer.
    if recover_bytes(data, signature).lower() != acct.address.lower():
        raise SystemExit("ERROR: internal self-check failed (signature did not recover to signer)")

    manifest = dict(manifest_body)
    manifest["signature"] = signature
    manifest["signer"] = acct.address                  # informational only; verify re-recovers it
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote signed release manifest: {args.out}", flush=True)
    print(f"  version      : {args.version}", flush=True)
    print(f"  git_commit   : {commit}", flush=True)
    print(f"  published_ts : {ts}", flush=True)
    if "min_client_version" in manifest_body:
        print(f"  min_client   : {manifest_body['min_client_version']} "
              f"(older clients will refuse to publish)", flush=True)
    if "config" in manifest_body:
        print(f"  config keys  : {', '.join(sorted(manifest_body['config']))}", flush=True)
    print(f"  manifest ver : {'v2' if len(manifest_body) > 3 else 'v1 (byte-identical to before)'}",
          flush=True)
    print(f"  signer       : {acct.address}", flush=True)
    if acct.address.lower() == PINNED_RELEASE_PUBKEY.lower():
        print("  pinned match : YES -- clients pinning the current key will accept this manifest",
              flush=True)
    else:
        print("  pinned match : NO  -- clients pin " + PINNED_RELEASE_PUBKEY, flush=True)
        print("                 (update PINNED_RELEASE_PUBKEY in tools/self_update.py to this signer,",
              flush=True)
        print("                  or sign with the key matching the pinned address). See SIGNING.md.",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
