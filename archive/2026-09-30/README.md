# archive/2026-09-30 -- files moved out of the live tree, not deleted

Date: 2026-09-30. Moved at public HEAD `3e99212e431ac12dcf26bcfd6226da6e45c0e601` (VERSION 3.8.4).
Every file below keeps its original relative path under this directory and its full git history
(`git log --follow`). Nothing here is imported, executed, collected by the test suite, or read by
the updater.

## What moved and why

| original path | archived at | bytes | last commit touching it | why it is unused and unneeded |
|---|---|---:|---|---|
| `docs/LADDER.md` | `archive/2026-09-30/docs/LADDER.md` | 8039 | 98fb572 (2026-07-21) | 0 inbound references in the 117 tracked files, in the README, and in the private repo's public-facing docs; not reached by any miner or operator entry point nor by `tests/`; not part of `release.json`. It documents the Rung-B capability ladder whose code (`tools/ladder_supervisor.py`) was removed from this repo on 2026-07-24 (commit `4db2f3e`, single-lane cleanup), so its own code links are dead here. The private repo keeps the canonical, byte-identical copy at `docs/LADDER.md`. |
| `docs/RELEASE_382_SIGNING_STEPS.md` | `archive/2026-09-30/docs/RELEASE_382_SIGNING_STEPS.md` | 11825 | da49647 (2026-08-14) | 0 inbound references anywhere (public tree, private tree, GitHub issues/PRs); a per-release operator checklist for 3.8.2 only, superseded by releases 3.8.3 and 3.8.4. The general, still-current signing procedure is `SIGNING.md` at the repo root, and the operator runbook lives in the private repo. Miners never read it. |

No `.py` file moved. No runtime, test, packaging or updater path changed:

- `tools/self_update.py` fetches `main/release.json` by URL and applies signed commits with
  `git checkout`; it never scans directories, so this folder is inert to updates.
- `release.json` carries no per-file `files` map, so no signed file list references these paths.
- `python -m pytest tests/ -q` (the documented install check) collects nothing here.

## Rules for this directory

- The directory name contains hyphens, so it is not a valid Python identifier: nothing under it can
  be imported as a package even if `archive/` ever lands on `sys.path`.
- Never add an `__init__.py` or a `conftest.py` anywhere under `archive/`. `tools/` is a namespace
  package (no `__init__.py`), so an `archive/tools/` with an `__init__.py` would merge into the
  `tools.*` namespace, and a `conftest.py` would put this folder on `sys.path` for pytest.
- Never archive a `test_*.py` here without renaming it: a bare `pytest` from the repo root has no
  `pytest.ini` and would collect it.

## How to restore a file

Either move it back in a new commit:

    git mv archive/2026-09-30/docs/LADDER.md docs/LADDER.md

or take the pre-archive bytes straight from history:

    git checkout 3e99212e431ac12dcf26bcfd6226da6e45c0e601 -- docs/LADDER.md
