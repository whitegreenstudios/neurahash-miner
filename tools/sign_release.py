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
import hashlib
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


def files_at_commit(commit, repo=None):
    """{path: sha256hex} for EVERY file in the tree at `commit`, read out of the object store.

    Built with one `git ls-tree -r` plus one streaming `git cat-file --batch`, so it is the tree the
    miner will actually have after `git checkout <commit>` -- not the operator's dirty working
    directory (the same reasoning as _version_at_commit above)."""
    repo = REPO if repo is None else repo
    out = subprocess.check_output(["git", "-C", repo, "ls-tree", "-r", "--full-tree",
                                   "--format=%(objectname) %(path)", commit],
                                  encoding="utf-8", errors="replace")
    entries = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        oid, _sp, path = line.partition(" ")
        if oid and path:
            entries.append((oid, path))
    if not entries:
        raise SystemExit(f"ERROR: `git ls-tree` found no files at commit {commit}")
    p = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        files = {}
        # ONE REQUEST, ONE RESPONSE, in lockstep. Queueing every oid up front and only then reading
        # DEADLOCKS on any real repo: git fills the stdout pipe buffer and blocks, while this
        # process is still blocked writing stdin. (Measured: hung past a 120 s timeout on this
        # repo's HEAD.) The lockstep costs nothing that matters -- this is a release tool.
        for oid, path in entries:
            p.stdin.write((oid + "\n").encode("ascii"))
            p.stdin.flush()
            header = p.stdout.readline().decode("utf-8", "replace").split()
            if len(header) < 3:
                raise SystemExit(f"ERROR: unexpected `git cat-file --batch` header for {path!r}: "
                                 f"{header!r}")
            want = int(header[2])
            blob = b""
            while len(blob) < want:                             # read() may return short
                chunk = p.stdout.read(want - len(blob))
                if not chunk:
                    raise SystemExit(f"ERROR: truncated blob for {path!r}")
                blob += chunk
            p.stdout.read(1)                                    # the trailing newline
            files[path] = hashlib.sha256(blob).hexdigest()
        p.stdin.close()
    finally:
        for h in (p.stdin, p.stdout):
            try:
                h.close()
            except Exception:
                pass
        p.wait()
    return files


def _load_files(spec, commit):
    """Return the v3 `files` map {relative/path: sha256hex}, or None if unset.

    `--files auto` builds it from the SIGNED COMMIT. Anything else is read as a JSON file or a
    literal JSON object and then CROSS-CHECKED against that commit -- it must match exactly.

    WHY THERE IS NO ESCAPE HATCH (adversarial review 2026-08-08, finding F2). This map is the
    DELETION ALLOWLIST the miner will apply on the NEXT release: a path this release declares and
    the next one drops becomes a deletion candidate. An operator who reads "the files this release
    ships" as "the files this release CHANGED" would hand every miner a map that makes the rest of
    the install obsolete -- measured on a 5-file fixture, a 1-path map deleted tools/self_update.py
    itself. A partial map is not a smaller mistake than a wrong one, so it is refused here, exactly
    like the VERSION/commit gate above. (The client carries its own mass-delete ceiling too; this is
    the belt to that pair of braces.) Path SAFETY is re-checked at the miner's deletion site -- a
    signed manifest is never a licence to name `../../`."""
    if spec is None:
        return None
    actual = files_at_commit(commit)
    if str(spec).strip().lower() == "auto":
        return actual
    text = spec
    if os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as f:
            text = f.read()
    try:
        files = json.loads(text)
    except Exception as e:
        raise SystemExit(f"ERROR: --files is neither `auto`, a readable JSON file, nor valid JSON ({e})")
    if not isinstance(files, dict):
        raise SystemExit("ERROR: --files must be a JSON OBJECT {path: sha256} (got %s)"
                         % type(files).__name__)
    for k, v in files.items():
        if not isinstance(k, str) or not k.strip():
            raise SystemExit(f"ERROR: --files has a non-string/empty path key: {k!r}")
        if not (isinstance(v, str) and len(v.strip()) == 64
                and all(c in "0123456789abcdefABCDEF" for c in v.strip())):
            raise SystemExit(f"ERROR: --files[{k!r}] is not a 64-hex sha256: {v!r}")
    files = {k: v.strip().lower() for k, v in files.items()}
    missing = sorted(set(actual) - set(files))
    extra = sorted(set(files) - set(actual))
    wrong = sorted(k for k in set(files) & set(actual) if files[k] != actual[k])
    if missing or extra or wrong:
        raise SystemExit(
            f"ERROR: refusing to sign -- --files does not match the tree at {commit}.\n"
            f"       missing from --files : {len(missing)} (first: {missing[:3]})\n"
            f"       not in the commit    : {len(extra)} (first: {extra[:3]})\n"
            f"       wrong sha256         : {len(wrong)} (first: {wrong[:3]})\n"
            f"       This map is the miners' DELETION ALLOWLIST for the next release: every path\n"
            f"       it omits becomes a deletion candidate on every miner. Use `--files auto`.")
    return files


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


def _version_at_commit(commit, repo=None):
    """Return the STRIPPED text of the VERSION file AS COMMITTED at `commit`, or None if absent.

    `git show <rev>:VERSION` reads the blob out of the object store, so this is the version the
    miner will actually find after `git checkout <commit>` -- not whatever the operator's dirty
    working tree happens to say. Missing path -> None (the caller refuses); any OTHER git failure
    (git absent, unreadable object store) is a hard error rather than a silent "absent".
    """
    repo = REPO if repo is None else repo
    try:
        out = subprocess.check_output(["git", "-C", repo, "show", f"{commit}:VERSION"],
                                      encoding="utf-8", errors="replace",
                                      stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return None                                    # no VERSION at that commit
    except Exception as e:
        raise SystemExit(f"ERROR: could not read VERSION at commit {commit} via git ({e})")
    return out.strip()


def _assert_version_matches_commit(version, commit, repo=None):
    """UNCONDITIONAL gate: the VERSION file AT THE SIGNED COMMIT must equal --version, or refuse.

    NEAR MISS 2026-07-25 (memory `shard-claim-v340-released`): a signed manifest declared v3.4.0
    while pointing at a commit whose VERSION file said 3.3.2, and this tool printed
    `pinned match : YES` -- because the SIGNATURE was valid. It was valid over the WRONG TREE.
    Every miner that takes such a manifest checks out the commit, reads 3.3.2, sees the manifest
    still offering 3.4.0, and re-execs forever: a fleet-wide update loop with a good signature on it.

    Compares STRIPPED text (the blob is compared as committed, so a CRLF-committed VERSION still
    matches -- .gitattributes pins `VERSION text eol=lf` so it should not happen in the first place).

    There is deliberately NO --force, no --skip-version-check and no env override: a release signer
    with an escape hatch is not a gate. Bump VERSION, commit it, and sign THAT commit.
    """
    want = str(version).strip()
    at_commit = _version_at_commit(commit, repo)
    if at_commit is None:
        raise SystemExit(
            f"ERROR: refusing to sign version {want} -- commit {commit} has NO VERSION file.\n"
            f"       That commit predates VERSION tracking (VERSION became tracked in b6db6ce), or "
            f"VERSION was deleted there.\n"
            f"       A miner checking out {commit} would find no version to compare against, which "
            f"is exactly the 3.4.0 near-miss shape. Sign a commit that CONTAINS VERSION = {want}.")
    if at_commit != want:
        raise SystemExit(
            f"ERROR: refusing to sign -- VERSION MISMATCH between the manifest and the signed tree.\n"
            f"       --version says   : {want}\n"
            f"       VERSION at commit: {at_commit}\n"
            f"       commit           : {commit}\n"
            f"       The signature would be valid over the WRONG TREE: miners would check out this "
            f"commit, read {at_commit}, still be offered {want}, and re-exec in a loop.\n"
            f"       Fix: bump VERSION to {want}, commit it, and sign THAT commit.")


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
    ap.add_argument("--files", default=None, metavar="FILE_OR_JSON",
                    help="v3: `auto` (recommended -- built from the signed commit), or a JSON file / literal "
                         "JSON object mapping every SHIPPED relative path to its sha256, which is then "
                         "CROSS-CHECKED against that commit and refused unless it matches exactly. Clients "
                         "may reclaim a dropped path later (opt-in, dry-run by default). Omit it and the "
                         "manifest is byte-identical to before.")
    args = ap.parse_args(argv)

    parse_version(args.version)                       # fail fast on a bad version string
    if args.min_client_version is not None:
        parse_version(args.min_client_version)        # same fail-fast: a malformed gate would brick publishing
    commit = _full_commit(args.commit) if args.commit else _default_commit()
    if not commit:
        raise SystemExit("ERROR: could not determine --commit (pass it explicitly)")
    # Refuse BEFORE the private key is ever read: a mismatched release is caught without the
    # operator's offline key touching this process at all.
    _assert_version_matches_commit(args.version, commit)
    ts = args.published_ts if args.published_ts is not None else int(time.time())

    acct = account_from_key(_load_privkey(args.key, args.key_env))
    manifest_body = {"version": str(args.version), "git_commit": str(commit), "published_ts": int(ts)}
    if args.min_client_version is not None:
        manifest_body["min_client_version"] = str(args.min_client_version)
    cfg = _load_config(args.config)
    if cfg is not None:
        manifest_body["config"] = cfg
    files = _load_files(args.files, commit)
    if files is not None:
        manifest_body["files"] = files
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
    if "files" in manifest_body:
        print(f"  shipped files: {len(manifest_body['files'])} paths (reclaim allowlist for the "
              f"NEXT release)", flush=True)
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
