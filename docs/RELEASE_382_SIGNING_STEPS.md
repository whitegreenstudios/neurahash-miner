# Release 3.8.2 -- signing steps for the operator

**Audience: the operator who holds the release private key. Miners need none of this.**

This is the per-release checklist for 3.8.2. The general trust model lives in `SIGNING.md`; this
file is the exact command list, in order, for this one release.

## Read this before you paste anything

**Every command below is PowerShell 5.1.** Not Git Bash, not cmd. Open PowerShell, run them there,
and do not translate them. This is not a style preference -- during the 3.4.0 release a Git Bash
`for f in ...; do cp ...; done` loop was pasted into PowerShell, failed as a **parse error that
changed nothing**, and the release clone never received the feature code. Nothing reported the
failure, so a manifest was signed over a tree that did not contain the release. Consequently this
file contains no `&&`, no `||`, no `for ... in ... do` loops, and no ternary operators.

**Every command prints something you can check.** If a command prints nothing at all, treat that as
a failure and stop -- silence is the failure mode that has actually cost this project a release.

Two more standing rules for this file:

- **Never** write the VERSION file with `echo x > VERSION` in PowerShell. PowerShell writes UTF-16
  there, and `read_local_version()` cannot read UTF-16 -- that mistake has shipped once already.
  VERSION is already correct in the candidate commit; step 2 only *verifies* it.
- The private key file is read **only** by `tools/sign_release.py` in step 5. Do not `Get-Content`
  it, do not echo it, do not put it in an environment variable on a shared machine.

## What is in 3.8.2

Two fixes, both about the miner no longer being quiet about things the miner's owner needs to know.

| Area | Change |
|---|---|
| `tools/sharddiloco_glm_contributor.py` | Stale-lane detector. If the coordinator's `event` counter and the accepted-record count both stop moving for 3 h (`NEURAHASH_SD_STALE_LANE_S`, `--stale-lane-s`, `0` disables), the miner prints a banner and **pauses training**, keeps polling, and resumes by itself with `LANE RECOVERED`. It never exits. |
| `tools/self_update.py` | Self-update failures print a full, untruncated, pure-ASCII banner instead of one ~200-char line; adds `--status`; detects untracked files that make `git checkout` abort on every future update. |
| `tests/` | `test_self_update_loud_failure.py` (20 tests), `test_stale_lane_detector.py` (21 tests). |

---

## 0. Set up the shell

Run each line in PowerShell, in order, from any directory.

```powershell
Set-Location D:\aiCrypto_work\neurahash-miner; Get-Location
```

Expected: `D:\aiCrypto_work\neurahash-miner`.

```powershell
$Py = "C:\Python313\python.exe"; Write-Host "python  : $Py"; Test-Path $Py
```

Expected: the path, then `True`. Never use a `.venv` python here.

```powershell
$KeyFile = "<full path to your release key file>"; Write-Host "key file: $KeyFile"; Test-Path $KeyFile
```

Expected: the path, then `True`. `Test-Path` does not read the file's contents. If this prints
`False`, fix the path now -- do not continue and do not open the file to "check" it.

---

## 1. Confirm the candidate commit

```powershell
git status --short
```

Expected: nothing, or only untracked operator files of your own such as `?? cfg.json`. Any modified
tracked file listed here is **not** in the commit you are about to sign -- commit it or revert it
before continuing.

```powershell
git log --oneline -1
```

Expected: the 3.8.2 candidate commit, subject beginning `release 3.8.2:`.

```powershell
$Sha = (git rev-parse HEAD).Trim(); Write-Host "candidate commit: $Sha"
```

Expected: a 40-hex sha. Everything below signs **this** sha.

---

## 2. GATE B -- the VERSION file at the signed commit

Release 3.4.0 was signed over a commit whose `VERSION` file read `3.3.2`, and the verifier still
printed `pinned match : YES`, because the signature itself was perfectly valid. A valid signature
says nothing about *what* you signed. Check the blob in the object store, not the file on disk:

```powershell
git show "$($Sha):VERSION"
```

Expected, exactly: `3.8.2`

The `"$($Sha):VERSION"` quoting is required. Bare `$Sha:VERSION` is parsed by PowerShell as a
scope-qualified variable and will not do what you want.

```powershell
Write-Host "VERSION bytes as hex:"; Get-Content .\VERSION -Encoding Byte | ForEach-Object { "{0:X2}" -f $_ }
```

Expected, one byte per line: `33 2E 38 2E 32 0A` -- that is `3.8.2` plus one LF, six bytes total,
pure ASCII. If the first two bytes are `FF FE`, the file is UTF-16 and `read_local_version()`
cannot read it: stop and rewrite it. (`-Encoding Byte` is PowerShell 5.1 syntax; on PowerShell 6+
it is `-AsByteStream`, which is one more reason to run this file in 5.1.)

`tools/sign_release.py` enforces this same gate itself and refuses to sign a mismatch, with no
override flag. Step 2 is here so you find out *before* you touch the key.

---

## 3. GATE A -- prove a FRESH CLONE starts

Release 3.7.1 shipped with 697 passing tests and bricked every miner that took it: an
`import no_toy_models` sat **inside** `build_node_model()`, a function no test calls because it
loads a 4.02 GiB trunk onto a GPU. A module that existed only in the private repo was therefore
invisible to every mechanical check. A passing test suite is not the acceptance artifact. **A fresh
clone that runs is.**

Clone onto `E:` -- `D:` is the scarcest drive on this machine.

```powershell
New-Item -ItemType Directory -Force E:\nh_release_gate | Out-Null; Remove-Item -Recurse -Force E:\nh_release_gate\382 -ErrorAction SilentlyContinue; Write-Host "gate dir ready and cleared"
```

```powershell
git clone --quiet . E:\nh_release_gate\382; Write-Host "clone exit code: $LASTEXITCODE"
```

Expected: `clone exit code: 0`.

```powershell
git -C E:\nh_release_gate\382 log --oneline -1
```

Expected: the same 3.8.2 candidate subject as step 1.

```powershell
git -C E:\nh_release_gate\382 show HEAD:VERSION
```

Expected: `3.8.2`. This is the fresh clone's own copy -- not the one you checked in step 2.

```powershell
Set-Location E:\nh_release_gate\382; & $Py -m pytest tests\test_published_tree_imports_resolve.py -q
```

Expected: `7 passed`. This is the static gate: it resolves every import in each shipped entry point
**including imports nested inside functions**, which is the 3.7.1 class of break. It carries its own
positive control (`test_lazy_imports_are_actually_being_checked`) so it cannot silently stop walking
function bodies and keep reading as coverage.

Known limit, worth your attention: that gate treats any `from neurahash import X` as resolved
because `neurahash` is a local package, so a **missing submodule** of a shipped package would slip
through it. That is exactly why the private-only `apply_storage_autotune` was left out of the public
tree by hand rather than trusted to the gate.

```powershell
& $Py tools\self_update.py --status
```

Expected: several lines of update state, exit without a traceback.

```powershell
& $Py tools\sharddiloco_glm_contributor.py --help
```

Expected: the usage block, and `--stale-lane-s STALE_LANE_S` present in it. This actually
**executes** the miner entry point far enough to run every module-level import and build the full
argument parser.

```powershell
& $Py -m pytest tests\test_self_update_loud_failure.py tests\test_stale_lane_detector.py tests\test_glm_autoupdate_wire.py -q
```

Expected: `46 passed`.

```powershell
Set-Location D:\aiCrypto_work\neurahash-miner; Get-Location
```

---

## 4. Push the code commit

The manifest points at a commit; miners `git checkout` it, so it has to exist on origin **before**
the manifest that names it is published.

```powershell
git push origin main
```

Expected: a `->` line ending in `main`, or `Everything up-to-date`.

```powershell
git ls-remote origin main
```

Expected: the sha printed in step 1, followed by `refs/heads/main`. Compare them character by
character.

---

## 5. Sign the manifest

This is the only step that touches the key.

```powershell
& $Py tools\sign_release.py --version 3.8.2 --commit $Sha --key $KeyFile --out release.json
```

Expected in the output:

- `pinned match : YES` -- the signer address equals the pinned trust root
  `0x5168F6cc4cc05bfd6d4714906d68e083c02dDC66`. If it says `NO`, **stop**: you signed with the wrong
  key. Do not edit the pinned constant to make it match; that would orphan every miner running today.
- No `VERSION MISMATCH` error. The tool re-reads `VERSION` at the signed commit and refuses
  unconditionally on a mismatch.

```powershell
Get-Content .\release.json
```

Expected: `version` is `3.8.2`, `git_commit` equals `$Sha`, `signer` is
`0x5168F6cc4cc05bfd6d4714906d68e083c02dDC66`.

---

## 6. Publish the manifest

```powershell
git add release.json
```

Stage that one path. Never `git add -A` or `git add .` here.

```powershell
git status --short
```

Expected: `M  release.json` staged, and nothing else staged.

```powershell
git commit -m "release 3.8.2: signed manifest ($Sha, VERSION 3.8.2)"
```

Expected: one file changed.

```powershell
git push origin main
```

Expected: a `->` line ending in `main`.

---

## 7. Verify what the fleet will actually fetch

Not what you meant to publish -- what the pinned URL now serves, checked against the pinned key by
the miner's own verifier.

```powershell
Set-Location E:\nh_release_gate\382; & $Py -c 'import json,urllib.request,sys; sys.path.insert(0,"tools"); from self_update import verify_manifest, PINNED_RELEASE_PUBKEY; m=json.load(urllib.request.urlopen("https://raw.githubusercontent.com/whitegreenstudios/neurahash-miner/main/release.json",timeout=20)); print("pinned key :", PINNED_RELEASE_PUBKEY); print("version    :", m["version"]); print("commit     :", m["git_commit"]); print("VERIFIES   :", verify_manifest(m))'
```

Expected: version `3.8.2`, commit equal to `$Sha`, and `VERIFIES : (True, '0x5168F6cc4cc05bfd6d4714906d68e083c02dDC66')`.

If GitHub raw still serves the old manifest, it is cached; wait a minute and run it again. Do not
work around a `False` here by changing any verification code.

---

## 8. Prove a miner at the old version actually takes it

```powershell
Remove-Item -Recurse -Force E:\nh_release_gate\takeup -ErrorAction SilentlyContinue; Write-Host "old takeup dir cleared"
```

```powershell
git clone --quiet https://github.com/whitegreenstudios/neurahash-miner.git E:\nh_release_gate\takeup; Write-Host "clone exit code: $LASTEXITCODE"
```

```powershell
git -C E:\nh_release_gate\takeup checkout --quiet acd5d0b4049c5859b65d99c23fa9aa1ae84229ca; git -C E:\nh_release_gate\takeup show HEAD:VERSION
```

Expected: `3.8.1` -- a miner sitting on the previous release.

```powershell
Set-Location E:\nh_release_gate\takeup; & $Py tools\self_update.py
```

Expected: the updater reports it applied the update. Then confirm the tree really moved:

```powershell
Get-Content .\VERSION
```

Expected: `3.8.2`.

```powershell
Set-Location D:\aiCrypto_work\neurahash-miner; Get-Location
```

---

## 9. Clean up the scratch clones

```powershell
Remove-Item -Recurse -Force E:\nh_release_gate\382 -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force E:\nh_release_gate\takeup -ErrorAction SilentlyContinue; Write-Host "scratch clones removed"
```

---

## If something is wrong after publishing

Do **not** delete or rewrite the signed commit; miners may already have checked it out. Publish a
**new, higher** version instead -- the update path only ever moves forward, so a fix ships as 3.8.3
by repeating this file with the new number. A downgrade manifest is ignored by every miner by
design, so it is not a rollback mechanism.

The one thing that is always safe: a miner that cannot update keeps mining the code it already has.
Failure here is fail-closed, not fail-open.
