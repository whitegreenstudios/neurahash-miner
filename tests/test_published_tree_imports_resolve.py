"""Every import a shipped miner can execute must resolve against the tree we actually publish.

WHY THIS EXISTS (2026-08-06, a live break, same day). Release 3.7.1 was cut with the full public
suite GREEN -- 697 passed, 0 failed -- and it bricked the miner on startup:

    File "tools/sharddiloco_glm_contributor.py", line 2854, in build_node_model
        import no_toy_models as _NTM
    ModuleNotFoundError: No module named 'no_toy_models'

`no_toy_models.py` had been written in the PRIVATE repo and never copied to the public one. The
suite could not see it, for a specific and general reason: **the import is LAZY -- it sits inside
build_node_model()**. Nothing at import time touches it, and no test calls that function (it loads
a 4 GiB trunk onto a GPU). So the module was missing in the one place it mattered and every
mechanical check was green. The 4060 self-updated, crashed, and sat dead.

The general shape: a test suite proves the code it EXECUTES. A lazy import inside a rarely-called
function is executed by neither the import system nor the tests, so it is invisible to both. The
only cheap check that sees it is a STATIC one -- read the source, resolve every name.

HARDENED 2026-09-30. An audit of release 3.8.4 (3e99212e) found the first version blind three ways:

  1. It accepted any import whose TOP-LEVEL package exists, so `import neura_l1.canon_DOES_NOT_EXIST`
     passed, and `from pkg import gone` was never checked past `pkg`. A missing package submodule
     is exactly as fatal as a missing top-level module.
  2. It read the entry points and nothing they import. The 3.7.1 shape moved one file deeper was
     invisible -- and was already there: neurahash_torch/pool_model.py imported the private-only
     `model_registry` unguarded, several imports below every entry point.
  3. It resolved against the FILESYSTEM, so an untracked local copy of a module passed a check that
     a fresh clone fails.

Now every dotted path is resolved one component at a time, case-exact, the way Python's path finder
does it (`from pkg import name` must name a submodule of pkg or a name pkg/__init__.py binds); the
walk is the TRANSITIVE closure of every miner and operator entry point, function bodies included,
following relative imports and scripts loaded or launched by file name; and the tree is what git
tracks. A module this tree does not ship is tolerated only at the exact sites listed in
ABSENT_BY_DESIGN -- never merely because a try/except surrounds it: a guard turns a crash into a
silently missing feature, which is not a pass. Positive controls plant each failure shape, in
synthetic trees and in a copy of the real one, and require the gate to report it; without them this
file could decay into a vacuous pass that reads as coverage. Files are parsed at the grammar of the
oldest Python a miner runs (3.10, the reference 4060), not the one running this suite.

NOT COVERED -- a static import walk cannot see these, and each can still take a miner down: the ORDER
in which an entry point puts the repo root on sys.path (the walk assumes tools/ and the root are both
there); a class a pickle or checkpoint names; circular-import errors; third-party internals, and
whether requirements.txt actually installs what THIRD_PARTY lists.

Static on purpose: nothing is imported or executed and nothing touches a GPU, so it is fast and safe
to run everywhere, and it fails for exactly the reason a miner would.

Run: C:/Python313/python.exe -m pytest tests/test_published_tree_imports_resolve.py -q
"""
import ast
import collections
import os
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# Distributed on PyPI, not by us -- requirements.txt's problem, not the tree's. Checked at the
# top-level name only: a third-party package's internals are not this tree's to resolve.
THIRD_PARTY = {
    "numpy", "torch", "transformers", "safetensors", "requests", "eth_account", "eth_keys",
    "eth_utils", "eth_hash", "coincurve", "tqdm", "huggingface_hub", "datasets", "psutil",
    "yaml", "regex", "sentencepiece", "tokenizers", "accelerate", "scipy", "pytest",
    "setuptools", "pkg_resources", "pytest_timeout", "bitsandbytes", "peft", "hf_transfer",
    "cryptography", "argon2", "dilithium_py", "web3",
}

# Every file a miner or operator runs by path, and where that is documented. The gate walks the
# transitive closure of each. Scripts an entry point launches or loads by file name (run_glm_miner.py
# starts its children that way) are followed automatically, and
# test_every_documented_command_is_an_entry_point fails if the docs tell anyone to run a script that
# is not listed here.
SHIPPED_ENTRY_POINTS = {
    "tools/fetch_glm_base.py": "miner: README install step 1",
    "tools/sharddiloco_glm_contributor.py": "miner: README install step 2",
    "tools/run_glm_miner.py": "miner: README, the recommended supervisor",
    "tools/self_update.py": "miner + operator: README; re-execs the caller after an update",
    "tools/sharddiloco_glm_expert.py": "miner: the contributor's expert lane",
    "tools/piece_loader.py": "miner: the base-piece loader",
    "tools/glm_rollout_worker.py": "miner, G1 train-role: README",
    "tools/glm_pipe_stage.py": "miner, G1 pipeline stage: README",
    "tools/sign_release.py": "operator: README + SIGNING.md",
}

# Modules this tree does not ship ON PURPOSE, and the ONLY sites allowed to import them:
# {missing module: {(file, enclosing def): why}}. Anything else that does not resolve fails -- a new
# module, a new file, or the same module from another function. An UNGUARDED entry is a known crash
# on a path the GLM lane never takes; its reason names the switch that reaches it.
# test_absent_by_design_has_no_stale_entries fails when a listed site stops importing its module,
# so this cannot rot into a blanket.
ABSENT_BY_DESIGN = {
    "model_registry": {
        ("neurahash_torch/pool_model.py", "_resolve_base"):
            "guarded: the private-repo alias table for the retired Qwen dense-base path; without it "
            "a raw HF id passes through and a bare alias fails with the fix",
        ("neurahash_torch/corpus_torch.py", "_qwen_tokenizer"):
            "guarded (except Exception): falls back to the raw base id",
    },
    "neurahash.grounding_corpora": {
        ("neurahash_torch/corpus_torch.py", "_grounding_sources"):
            "UNGUARDED: the private-repo grounding corpus lane; reached only when "
            "NEURAHASH_CORPUS_SOURCES is set or NEURAHASH_CORPUS=grounding*",
        ("neurahash_torch/corpus_torch.py", "build_grounding_data"):
            "UNGUARDED: same lane, same switches",
        ("neurahash_torch/corpus_torch.py", "corpus_sha"):
            "UNGUARDED: same lane, same switches",
    },
    "neurahash_torch.nsa_attention": {
        ("neurahash_torch/model_torch.py", "Block.__init__"):
            "UNGUARDED: private-repo NSA attention; reached only by an arch with attention='nsa', "
            "which nothing shipped sets",
    },
    "neura_l1.finality": {
        ("neura_l1/block_state.py", "State._apply_vote_fault_slash"):
            "UNGUARDED: private-repo FFG finality; reached only when an L1 State applies a vote-fault "
            "slash tx -- a miner reaches block_state only for signing.py's _canon, and State is built "
            "by the settlement audit code (neurahash/chain_settlement.py), not by any entry point",
    },
}

# A miner runs every entry point as `python tools/<name>.py` from the repo root: tools/ is
# sys.path[0], and the entry points that need the packages put the repo root on sys.path.
_SEARCH_DIRS = ("tools", "")

if sys.version_info < (3, 10):
    pytest.skip("needs Python 3.10+ (sys.stdlib_module_names)", allow_module_level=True)

# The oldest Python a miner runs -- the reference 4060 is on 3.10. Syntax or a stdlib module newer
# than that parses and imports fine on the 3.13 box that runs this suite, and dies on that miner.
_MINER_PYTHON = (3, 10)
_STDLIB_NEWER_THAN_MINERS = {"tomllib"}                                # 3.11
_STDLIB = (set(sys.stdlib_module_names) | set(sys.builtin_module_names)) - _STDLIB_NEWER_THAN_MINERS
_TRY = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
_CATCHES_IMPORT_ERROR = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
_PY_FILE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

_Import = collections.namedtuple("_Import", "line kind module names level where guarded")
_Problem = collections.namedtuple("_Problem", "file line where missing why guarded")


def _join(*parts):
    return "/".join(p for p in parts if p)


def _git(root, *args, **kwargs):
    """git -C root ..., minus inherited GIT_* variables: under a git hook GIT_DIR / GIT_INDEX_FILE
    point at the hook's repository, and `git -C <tmp> add` would stage into THAT index."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    return subprocess.run(["git", "-C", root] + list(args), env=env, capture_output=True, **kwargs)


def _git_tracked(root):
    """The files ('a/b.py') a fresh clone of `root` would hold, per `git ls-files` -- or None when
    `root` is not the top of its own checkout (the synthetic trees the controls build), and the disk
    is the tree. Tracked files deleted from the working tree are left out: they cannot be read."""
    try:
        top = _git(root, "rev-parse", "--show-toplevel", encoding="utf-8", errors="replace",
                   timeout=60)
        if top.returncode != 0 or not os.path.samefile(top.stdout.strip(), root):
            return None
        ls = _git(root, "ls-files", "-z", timeout=180)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if ls.returncode != 0:
        return None
    names = (p.decode("utf-8", "replace") for p in ls.stdout.split(b"\0") if p)
    return {p for p in names if os.path.isfile(os.path.join(root, *p.split("/")))}


def _catches_import_error(handler):
    if handler.type is None:
        return True
    for t in handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]:
        name = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else None
        if name in _CATCHES_IMPORT_ERROR:
            return True
    return False


def _literal_import(call):
    """`importlib.import_module("x")` / `__import__("x")` with a literal name -> "x"."""
    f = call.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
    if name in ("import_module", "__import__") and call.args:
        arg = call.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _DOTTED.match(arg.value):
            return arg.value
    return None


def _collect_imports(tree):
    """Every import in a module, INCLUDING ones nested inside functions -- the 3.7.1 break was an
    import inside build_node_model() -- plus `importlib.import_module("x")` / `__import__("x")` with a
    literal name, and any string literal that is exactly `<name>.py` (a script launched or loaded by
    file name). Each is tagged with its enclosing def and with whether a try/except that catches
    ImportError surrounds it where it RUNS: a function body runs when it is called, outside any try
    around its def."""
    out = []
    stack = [(node, "<module>", False) for node in tree.body]
    while stack:
        node, where, guarded = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            name = getattr(node, "name", "<lambda>")
            inner = name if where == "<module>" else "%s.%s" % (where, name)
            head = list(getattr(node, "decorator_list", []))
            if isinstance(node, ast.ClassDef):
                head += node.bases + [kw.value for kw in node.keywords]
                body_guarded = guarded                    # a class body runs where it stands
            else:
                head += [node.args] + ([node.returns] if getattr(node, "returns", None) else [])
                body_guarded = False
            body = node.body if isinstance(node.body, list) else [node.body]
            stack += [(n, where, guarded) for n in head]
            stack += [(n, inner, body_guarded) for n in body]
        elif isinstance(node, _TRY):
            covered = guarded or any(_catches_import_error(h) for h in node.handlers)
            stack += [(n, where, guarded) for n in node.handlers + node.orelse + node.finalbody]
            stack += [(n, where, covered) for n in node.body]
        elif isinstance(node, ast.Import):
            out += [_Import(node.lineno, "import", a.name, (), 0, where, guarded) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append(_Import(node.lineno, "from", node.module or "",
                               tuple(a.name for a in node.names), node.level or 0, where, guarded))
        else:
            literal = _literal_import(node) if isinstance(node, ast.Call) else None
            if literal:
                out.append(_Import(node.lineno, "import", literal, (), 0, where, guarded))
            elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and _PY_FILE.match(node.value)):
                out.append(_Import(node.lineno, "file", node.value, (), 0, where, guarded))
            stack += [(n, where, guarded) for n in ast.iter_child_nodes(node)]
    out.sort(key=lambda r: r.line)
    return out


def _is_main_guard(test):
    """`if __name__ == "__main__":` -- a body that never runs when the module is imported."""
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__" and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == "__main__")


def _bound_names(tree):
    """(names a module binds at top level when IMPORTED, open_ended). Static and generous -- every
    assignment target outside a def or class body, def and class names, import aliases, `global`
    declarations -- except an annotation with no value and anything under `if __name__ ==
    "__main__":`, which bind nothing on import; open-ended when a star import or a PEP 562 module
    __getattr__ can bind names nobody can see."""
    names, open_ended = set(), False
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.AnnAssign) and node.value is None:
            continue                                        # `X: int` alone binds nothing
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            open_ended |= node.name == "__getattr__" and not isinstance(node, ast.ClassDef)
            names.update(n for sub in ast.walk(node) if isinstance(sub, ast.Global) for n in sub.names)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    open_ended = True
                else:
                    names.add(alias.asname or alias.name.split(".")[0])
        else:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            stack.extend(ast.iter_child_nodes(node))
    return names, open_ended


def _absolute(package, level, module):
    """PEP 328: `from ..x import y` in package a.b is a.x. None if it climbs above the top."""
    parts = package.split(".") if package else []
    if level > len(parts):
        return None
    return ".".join(parts[:len(parts) - level + 1] + ([module] if module else []))


class _Tree:
    """A source tree, and Python's path finder run over it without executing anything."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.tracked = _git_tracked(self.root)
        self._listing = {}
        self._parsed = {}
        for rel in self.tracked or ():
            parts = rel.split("/")
            for i, part in enumerate(parts):
                self._listing.setdefault("/".join(parts[:i]), {})[part] = (
                    "file" if i == len(parts) - 1 else "dir")

    def listdir(self, rel):
        """{name: 'file' | 'dir'} directly inside directory `rel` ('' is the root), case-exact the
        way CPython's importer is on every OS (PEP 235) -- which os.path.isfile() is not on Windows."""
        if rel not in self._listing:
            if self.tracked is not None:
                return {}
            full = os.path.join(self.root, *rel.split("/")) if rel else self.root
            entries = {}
            try:
                names = os.listdir(full)
            except OSError:
                names = []
            for name in names:
                path = os.path.join(full, name)
                if os.path.isfile(path):
                    entries[name] = "file"
                elif os.path.isdir(path) and name not in (".git", "__pycache__"):
                    entries[name] = "dir"
            self._listing[rel] = entries
        return self._listing[rel]

    def kind(self, rel):
        head, _, name = rel.rpartition("/")
        return self.listdir(head).get(name)

    def _parse(self, rel):
        if rel not in self._parsed:
            with open(os.path.join(self.root, *rel.split("/")), "rb") as fh:
                source = fh.read()
            try:
                tree = ast.parse(source, filename=rel, feature_version=_MINER_PYTHON)
            except (SyntaxError, ValueError) as exc:
                self._parsed[rel] = exc
            else:
                self._parsed[rel] = (_collect_imports(tree), _bound_names(tree))
        if isinstance(self._parsed[rel], Exception):
            raise self._parsed[rel]
        return self._parsed[rel]

    def imports(self, rel):
        return self._parse(rel)[0]

    def binds(self, rel, name):
        try:
            names, open_ended = self._parse(rel)[1]
        except (SyntaxError, ValueError):
            return True                     # reported where the walk parses the file itself
        return open_ended or name in names

    def module_of(self, rel):
        """(dotted name, package) a file is imported as: tools/x.py is the top-level module `x`
        (tools/ is on sys.path); neurahash/a.py is `neurahash.a`, in package `neurahash`."""
        for d in _SEARCH_DIRS:
            prefix = d + "/" if d else ""
            if rel.startswith(prefix):
                parts = rel[len(prefix):-len(".py")].split("/")
                if parts[-1] == "__init__":
                    return ".".join(parts[:-1]), ".".join(parts[:-1])
                return ".".join(parts), ".".join(parts[:-1])
        return rel, ""

    def find(self, dirs, name):
        """One step of Python's path finder over `dirs`: ('package', its __init__.py, [dir]),
        ('module', file, []), ('namespace', None, [dirs]) or None -- with the same precedence (a
        package or module in an earlier dir wins; namespace portions are the fallback)."""
        portions = []
        for d in dirs:
            entries = self.listdir(d)
            sub = _join(d, name)
            if entries.get(name) == "dir" and self.listdir(sub).get("__init__.py") == "file":
                return ("package", _join(sub, "__init__.py"), [sub])
            if entries.get(name + ".py") == "file":
                return ("module", _join(d, name + ".py"), [])
            if entries.get(name) == "dir":
                portions.append(sub)
        return ("namespace", None, portions) if portions else None

    def resolve(self, dotted):
        """Resolve an absolute module name -> (spec, files, missing, why). spec is the final find()
        result, or None when the name is not in this tree; files are the sources the import executes
        (each package __init__.py on the way down, then the module); missing is the first dotted
        prefix that does not exist, or None, and why says how it is missing."""
        parts = dotted.split(".")
        spec = self.find(_SEARCH_DIRS, parts[0])
        if spec is None:
            if parts[0] in _STDLIB or parts[0] in THIRD_PARTY:
                return None, [], None, None
            return None, [], parts[0], "not in this tree, the standard library or THIRD_PARTY"
        files = [spec[1]] if spec[1] else []
        for i in range(1, len(parts)):
            parent, here = ".".join(parts[:i]), ".".join(parts[:i + 1])
            if spec[0] == "module":
                return None, files, here, "%s is a module, not a package" % parent
            spec = self.find(spec[2], parts[i])
            if spec is None:
                return None, files, here, "package %s has no submodule %s" % (parent, parts[i])
            if spec[1]:
                files.append(spec[1])
        return spec, files, None, None


def _resolve(tree, rel, rec):
    """One import record of file `rel` -> (local files it executes, [(missing name, why)])."""
    if rec.kind == "file":
        here = rel.rpartition("/")[0]
        for d in (here, "tools", ""):
            if tree.listdir(d).get(rec.module) == "file":
                return [_join(d, rec.module)], []
        return [], [(rec.module, "loaded or launched by file name, and no such file is in this tree")]
    base = rec.module
    if rec.level:
        base = _absolute(tree.module_of(rel)[1], rec.level, rec.module)
        if base is None:
            return [], [("." * rec.level + (rec.module or rec.names[0]),
                         "a relative import above the top-level package")]
    spec, files, missing, why = tree.resolve(base)
    if missing:
        return files, [(missing, why)]
    if rec.kind == "import" or spec is None:        # stdlib / third party: its names are not ours
        return files, []
    out = []
    for name in rec.names:
        if name == "*":
            continue
        if spec[0] != "module":
            sub = tree.find(spec[2], name)
            if sub is not None:
                files += [sub[1]] if sub[1] else []
                continue
        if spec[0] == "namespace" or not tree.binds(spec[1], name):
            if spec[0] == "module":
                why = "%s binds no name %s" % (base, name)
            else:
                why = "package %s has no submodule %s and binds no such name" % (base, name)
            out.append(("%s.%s" % (base, name), why))
    return files, out


def _walk(tree, entries, absent_by_design=None):
    """The transitive import closure of `entries` -> (reached, problems). reached maps every file
    the walk reached to (importer, line), or None for an entry point; problems lists every import
    that does not resolve and is not an absent_by_design site (default ABSENT_BY_DESIGN)."""
    absent = ABSENT_BY_DESIGN if absent_by_design is None else absent_by_design
    reached, queue, problems = {}, [], []
    for entry in entries:
        if entry not in reached:
            reached[entry] = None
            queue.append(entry)
    for rel in queue:                       # breadth-first: the list grows while it is read
        try:
            records = tree.imports(rel)
        except (SyntaxError, ValueError) as exc:
            problems.append(_Problem(rel, getattr(exc, "lineno", None) or 0, "<module>", rel,
                                     "does not parse: %s" % (exc,), False))
            continue
        for rec in records:
            files, missing = _resolve(tree, rel, rec)
            for f in files:
                if f not in reached:
                    reached[f] = (rel, rec.line)
                    queue.append(f)
            for name, why in missing:
                if (rel, rec.where) not in absent.get(name, {}):
                    problems.append(_Problem(rel, rec.line, rec.where, name, why, rec.guarded))
    return reached, problems


def _chain(reached, rel):
    """How a miner gets to `rel`: 'tools/entry.py:12 -> pkg/mod.py:3 -> <rel>'."""
    hops, cur = [], rel
    while reached.get(cur):
        importer, line = reached[cur]
        hops.append("%s:%d" % (importer, line))
        cur = importer
    return " -> ".join(hops[::-1] + [rel])


def _explain(reached, problems):
    return "\n".join(
        "  %s:%d in %s, %s: %s -- %s\n      reached: %s" % (
            p.file, p.line, p.where, "guarded" if p.guarded else "UNGUARDED", p.missing, p.why,
            _chain(reached, p.file))
        for p in problems)


_PUBLISHED = {}


def _published():
    """This checkout as a _Tree, built once (the parametrized tests share its parse cache)."""
    if "tree" not in _PUBLISHED:
        _PUBLISHED["tree"] = _Tree(_REPO)
    return _PUBLISHED["tree"]


def _is_published(tree):
    """The public miner tree carries the signed manifest self_update fetches. This file ships in
    the private dev repo too, which has neither that manifest nor every public-only entry point."""
    return tree.kind("release.json") == "file"


def _entries(tree):
    return [e for e in SHIPPED_ENTRY_POINTS if tree.kind(e) == "file"]


@pytest.mark.parametrize("entry", sorted(SHIPPED_ENTRY_POINTS))
def test_every_import_reachable_from_an_entry_point_resolves(entry):
    """A miner runs these from a fresh clone. An import that does not resolve anywhere below one --
    at module level or inside a function, in the entry point or five imports down -- is a dead miner
    on the path that reaches it."""
    tree = _published()
    if tree.kind(entry) != "file":
        if _is_published(tree):
            pytest.fail("%s (%s) is not in this tree: a miner who runs it gets 'No such file'. If it "
                        "was retired, drop it from SHIPPED_ENTRY_POINTS in the same commit."
                        % (entry, SHIPPED_ENTRY_POINTS[entry]))
        pytest.skip("%s is not in this unpublished tree" % entry)
    reached, problems = _walk(tree, [entry])
    assert not problems, (
        "%d import(s) reachable from %s (%s) do not resolve against this tree. A miner that reaches "
        "an unguarded one crashes; a guarded one silently loses the feature:\n%s\n"
        "Fix: ship the module here (if it lives in the private repo, copy it before releasing), or "
        "guard the import and list its exact site in ABSENT_BY_DESIGN with why no miner takes that "
        "path." % (len(problems), entry, SHIPPED_ENTRY_POINTS[entry], _explain(reached, problems)))


# `python`, `python3.11`, `python.exe` (with any path in front), `py -3`, then any flags, then a
# tools/ script or `-m module`.
_DOC_COMMAND = re.compile(
    r"\b(?:python[0-9.]*|py(?:\s+-[0-9][0-9.]*)?)(?:\.exe)?(?:\s+-[A-Za-z]+)*?\s+"
    r"(?:(tools[/\\][A-Za-z0-9_]+\.py)|-m\s+([A-Za-z_][A-Za-z0-9_.]*))")


def _documented_commands(tree):
    """{script or module the docs tell people to run: 'DOC.md:line'} -- a module that does not
    exist is reported by name; one that is not ours (`python -m pytest`) is left out."""
    found = {}
    for doc in ("README.md", "SIGNING.md", "BUNDLE.md"):
        if tree.kind(doc) != "file":
            continue
        with open(os.path.join(tree.root, doc), encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                for script, module in _DOC_COMMAND.findall(line):
                    if module:
                        _spec, files, missing, _why = tree.resolve(module.rstrip("."))
                        if not files:
                            continue
                        script = missing or files[-1]
                    found.setdefault(script.replace("\\", "/"), "%s:%d" % (doc, lineno))
    return found


def test_every_documented_command_is_an_entry_point():
    """A script the docs tell a miner or operator to run IS an entry point -- and must exist. This
    keeps SHIPPED_ENTRY_POINTS from quietly falling behind the README."""
    tree = _published()
    if not _is_published(tree):
        pytest.skip("an unpublished tree's docs are not what miners follow")
    documented = _documented_commands(tree)
    # POSITIVE CONTROL: the install command itself must be found, or a regex that stopped matching
    # anything would pass this test forever.
    assert "tools/sharddiloco_glm_contributor.py" in documented, (
        "the README's install command was not recognised -- _DOC_COMMAND no longer reads the docs")
    unlisted = {s: where for s, where in documented.items() if s not in SHIPPED_ENTRY_POINTS}
    assert not unlisted, (
        "the docs tell people to run these, but SHIPPED_ENTRY_POINTS does not list them, so nothing "
        "checks that they exist or what they import: %s"
        % ", ".join("%s (%s)" % kv for kv in sorted(unlisted.items())))


def test_the_tree_is_read_from_git_when_it_is_a_checkout():
    """Blind spot 3 from the other side: if `git ls-files` ever failed here -- dubious ownership, a
    git that prints /c/ paths, no git on PATH -- the gate would fall back to the disk without a word,
    and an untracked module would pass again."""
    tree = _published()
    if not os.path.exists(os.path.join(tree.root, ".git")):
        pytest.skip("not a git checkout (an archive download): the disk is the tree")
    assert tree.tracked is not None, "this is a git checkout but `git ls-files` could not be read"


def _targets(tree, rel, rec):
    """The absolute names one import record names: the module, and module.name for each name."""
    if rec.kind == "file":
        return {rec.module}
    base = rec.module
    if rec.level:
        base = _absolute(tree.module_of(rel)[1], rec.level, rec.module) or ""
    return {base} | {"%s.%s" % (base, n) for n in rec.names if n != "*"}


def test_absent_by_design_has_no_stale_entries():
    """Each ABSENT_BY_DESIGN site must still import its module -- a stale entry is a pre-approved
    hole waiting for the next import to reuse it -- and in the published tree the module must still
    be absent: once it ships, the entry would only hide a later removal."""
    tree = _published()
    stale = []
    for module, sites in sorted(ABSENT_BY_DESIGN.items()):
        if _is_published(tree) and tree.resolve(module)[2] is None:
            stale.append("%s resolves in this tree now -- delete its entry" % module)
        for rel, where in sorted(sites):
            records = tree.imports(rel) if tree.kind(rel) == "file" else []
            if not any(r.where == where and any(t == module or t.startswith(module + ".")
                                                for t in _targets(tree, rel, r))
                       for r in records):
                stale.append("%s, in %s, no longer imports %s" % (rel, where, module))
    assert not stale, "stale ABSENT_BY_DESIGN entries:\n  " + "\n  ".join(stale)


def test_the_walk_reaches_deep_into_what_a_miner_executes():
    """POSITIVE CONTROL on the real tree. The closure of the entry points must leave tools/ and reach
    modules at least three imports down. If the walk ever stopped being transitive, every
    entry-point test would pass while reading one file each -- the first version's blind spot."""
    tree = _published()
    entries = _entries(tree)
    if not entries:
        pytest.skip("no entry point in this tree")
    reached, _ = _walk(tree, entries)
    depth = {f: _chain(reached, f).count(" -> ") for f in reached}
    deepest = max(sorted(depth), key=depth.get)
    assert depth[deepest] >= 3, "the deepest file reached is %d import(s) down: %s" % (
        depth[deepest], _chain(reached, deepest))
    assert any(not f.startswith("tools/") for f in reached), "the walk never left tools/"


def test_the_module_that_actually_broke_3_7_1_is_present():
    """Regression, named. build_node_model() does `import no_toy_models` lazily; the walk must
    follow that nested import, and tools/no_toy_models.py must be in the tree."""
    tree = _published()
    contributor = "tools/sharddiloco_glm_contributor.py"
    if tree.kind(contributor) != "file":
        pytest.skip("contributor not in this tree")
    with open(os.path.join(tree.root, *contributor.split("/")), "rb") as fh:
        if b"import no_toy_models" not in fh.read():       # the source, not the collector under test
            pytest.skip("the contributor no longer imports no_toy_models")
    assert any(r.module == "no_toy_models" for r in tree.imports(contributor)), (
        "the source imports no_toy_models but _collect_imports does not see it -- the walk has "
        "stopped reading function bodies")
    reached, _ = _walk(tree, [contributor])
    assert "tools/no_toy_models.py" in reached, (
        "sharddiloco_glm_contributor.py imports no_toy_models but tools/no_toy_models.py is not in "
        "this tree. This exact gap shipped as 3.7.1 and took the reference 4060 down.")


def _plant(root, files):
    """Write a synthetic tree ({relpath: source}) under `root` and return it as a _Tree."""
    for rel, source in files.items():
        path = os.path.join(str(root), *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(textwrap.dedent(source).lstrip("\n"))
    return _Tree(str(root))


def _found(tree, entries, absent=None):
    return sorted((p.file, p.missing) for p in _walk(tree, entries, absent or {})[1])


def test_a_planted_missing_submodule_fails_the_gate_in_every_spelling(tmp_path):
    """POSITIVE CONTROL for blind spot 1: `import neura_l1.canon_DOES_NOT_EXIST` passed the first
    version because `neura_l1` exists. Every spelling of a missing submodule or name must fail --
    dotted, `from pkg import`, `from pkg.gone import`, relative, through a module that is not a
    package, and a name bound only by a bare annotation or only under `if __name__ == "__main__":`
    -- while the same spellings of things that exist must not."""
    tree = _plant(tmp_path, {
        "tools/entry.py": """
            import pkg.present
            import pkg.gone_as_import
            from pkg import present, bound_in_init
            from pkg import gone_as_from
            from pkg.gone_as_from_dotted import anything
            from pkg.present import VALUE
            from pkg.present import NOT_A_NAME_THERE
            from pkg.present import ANNOTATED_ONLY, MAIN_ONLY

            def lazy():
                import pkg.present.not_a_package
        """,
        "pkg/__init__.py": "bound_in_init = 1\n",
        "pkg/present.py": """
            VALUE = 1
            ANNOTATED_ONLY: int
            from .sibling import thing
            from .gone_as_relative import other

            if __name__ == "__main__":
                MAIN_ONLY = 1
        """,
        "pkg/sibling.py": "thing = 2\n",
    })
    assert _found(tree, ["tools/entry.py"]) == [
        ("pkg/present.py", "pkg.gone_as_relative"),
        ("tools/entry.py", "pkg.gone_as_from"),
        ("tools/entry.py", "pkg.gone_as_from_dotted"),
        ("tools/entry.py", "pkg.gone_as_import"),
        ("tools/entry.py", "pkg.present.ANNOTATED_ONLY"),
        ("tools/entry.py", "pkg.present.MAIN_ONLY"),
        ("tools/entry.py", "pkg.present.NOT_A_NAME_THERE"),
        ("tools/entry.py", "pkg.present.not_a_package"),
    ]


def test_a_planted_nested_import_fails_the_gate_even_deep_in_the_closure(tmp_path):
    """POSITIVE CONTROL for the 3.7.1 shape and blind spot 2: a missing module imported inside a
    function must fail the gate in the entry point itself AND three imports below it (through a plain
    import, a `from pkg import submodule`, and a relative import, into a method body)."""
    tree = _plant(tmp_path, {
        "tools/entry.py": """
            import helper

            def build_node_model():
                import no_such_module_in_the_entry_point
        """,
        "tools/helper.py": "from pkg import mod\n",
        "pkg/__init__.py": "",
        "pkg/mod.py": "from .deep import build\n",
        "pkg/deep.py": """
            class Loader:
                def build(self):
                    import no_such_module_three_imports_down

            build = Loader().build
        """,
    })
    reached, problems = _walk(tree, ["tools/entry.py"], {})
    assert sorted((p.file, p.where, p.missing) for p in problems) == [
        ("pkg/deep.py", "Loader.build", "no_such_module_three_imports_down"),
        ("tools/entry.py", "build_node_model", "no_such_module_in_the_entry_point"),
    ]
    assert _chain(reached, "pkg/deep.py") == (
        "tools/entry.py:1 -> tools/helper.py:1 -> pkg/mod.py:1 -> pkg/deep.py")


def test_a_guard_is_not_a_pass(tmp_path):
    """A try/except ImportError around a missing module turns a crash into a silently missing
    feature -- the way `from self_update import check_and_update` once failed closed and SILENT in
    the private contributor. So only an exact ABSENT_BY_DESIGN site passes: the same module from
    another function fails guarded or not, a handler that does not catch ImportError is no guard,
    and a try around a DEF does not guard its body, which runs later."""
    tree = _plant(tmp_path, {
        "tools/entry.py": """
            try:
                import optional_absent
            except ImportError:
                optional_absent = None

            try:
                import not_listed_anywhere
            except (OSError, ImportError):
                pass

            try:
                import wrong_handler
            except ValueError:
                pass

            try:
                def later():
                    import optional_absent
            except ImportError:
                pass

            def elsewhere():
                try:
                    import optional_absent
                except ModuleNotFoundError:
                    pass

            def unguarded():
                import optional_absent
        """,
    })
    absent = {"optional_absent": {("tools/entry.py", "<module>"): "the one approved site"}}
    _, problems = _walk(tree, ["tools/entry.py"], absent)
    assert sorted((p.where, p.missing, p.guarded) for p in problems) == [
        ("<module>", "not_listed_anywhere", True),
        ("<module>", "wrong_handler", False),
        ("elsewhere", "optional_absent", True),
        ("later", "optional_absent", False),
        ("unguarded", "optional_absent", False),
    ]


def test_a_script_launched_by_file_name_is_walked_and_must_exist(tmp_path):
    """run_glm_miner.py starts its children by file name. Those files are code a miner runs, so the
    walk follows them, and a name that is not in the tree fails like a missing import."""
    tree = _plant(tmp_path, {
        "tools/supervisor.py": """
            import os
            HERE = os.path.dirname(os.path.abspath(__file__))
            CHILDREN = [os.path.join(HERE, "child.py"), os.path.join(HERE, "renamed_away.py")]
        """,
        "tools/child.py": """
            def main():
                import gone_from_the_child
        """,
    })
    assert _found(tree, ["tools/supervisor.py"]) == [
        ("tools/child.py", "gone_from_the_child"),
        ("tools/supervisor.py", "renamed_away.py"),
    ]


def test_a_relative_import_outside_a_package_fails(tmp_path):
    """tools/ is a sys.path directory, not a package: `from . import x` there dies at import time."""
    tree = _plant(tmp_path, {"tools/entry.py": "from . import sibling\n", "tools/sibling.py": ""})
    assert _found(tree, ["tools/entry.py"]) == [("tools/entry.py", ".sibling")]


def test_case_must_match_exactly(tmp_path):
    """`import Helper` does not find helper.py -- not on Linux, and not on Windows either unless
    PYTHONCASEOK is set -- although os.path.isfile("Helper.py") says True on Windows."""
    tree = _plant(tmp_path, {"tools/entry.py": "import Helper\n", "tools/helper.py": ""})
    assert _found(tree, ["tools/entry.py"]) == [("tools/entry.py", "Helper")]


def test_newer_python_than_the_miners_run_fails(tmp_path):
    """The suite runs on 3.13; the reference 4060 runs 3.10. `except*` and `import tomllib` work here
    and die there, so both must fail the gate."""
    tree = _plant(tmp_path, {
        "tools/entry.py": "import helper\nimport tomllib\n",
        "tools/helper.py": """
            try:
                pass
            except* ValueError:
                pass
        """,
    })
    assert _found(tree, ["tools/entry.py"]) == [
        ("tools/entry.py", "tomllib"),
        ("tools/helper.py", "tools/helper.py"),                # does not parse on 3.10
    ]


def test_an_untracked_module_does_not_count(tmp_path):
    """Blind spot 3: the tree is what git tracks. An untracked local copy of a module -- a private
    file dropped into a public checkout to try something -- is not in a fresh clone."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    _plant(tmp_path, {"tools/entry.py": "import tracked_mod\nimport untracked_mod\n",
                      "tools/tracked_mod.py": "", "tools/untracked_mod.py": ""})
    for args in (["init", "-q"], ["add", "tools/entry.py", "tools/tracked_mod.py"]):
        _git(str(tmp_path), *args, encoding="utf-8", errors="replace", check=True)
    tree = _Tree(str(tmp_path))
    assert tree.tracked is not None, "git mode did not engage"
    assert _found(tree, ["tools/entry.py"]) == [("tools/entry.py", "untracked_mod")]


def test_a_clean_tree_passes(tmp_path):
    """NEGATIVE CONTROL: the positive controls mean nothing if the gate fails everything. Stdlib,
    THIRD_PARTY, dotted and from-imported submodules, names a package binds, relative imports, the
    `tools.` namespace, a literal import_module and a script named by file all resolve."""
    tree = _plant(tmp_path, {
        "tools/entry.py": """
            from __future__ import annotations
            import os, json
            import os.path
            import importlib
            from concurrent.futures import ThreadPoolExecutor
            import numpy
            import helper
            import tools.helper
            from pkg import sub, NAME
            from pkg.sub import thing as _thing
            CHILD = "child.py"

            def lazy():
                import pkg.sub
                from pkg import sub as again
                return importlib.import_module("pkg.sub")
        """,
        "tools/helper.py": "X = 1\n",
        "tools/child.py": "import pkg\n",
        "pkg/__init__.py": "from .sub import thing\nNAME = thing\n",
        "pkg/sub.py": "def thing():\n    return 1\n",
    })
    reached, problems = _walk(tree, ["tools/entry.py"], {})
    assert problems == []
    assert {"tools/helper.py", "tools/child.py", "pkg/__init__.py", "pkg/sub.py"} <= set(reached)


def test_planting_both_shapes_into_a_copy_of_the_real_tree_fails_it(tmp_path):
    """POSITIVE CONTROL on the REAL import graph (the synthetic controls prove the mechanics; this
    proves this tree's layout hides nothing from them). Copy every file the real walk reaches, plant
    a missing package submodule at module level of the deepest module and a missing module nested in
    a function of the next deepest, and the gate must report exactly what it reports for the real
    tree plus both plants."""
    real = _published()
    entries = _entries(real)
    if not entries:
        pytest.skip("no entry point in this tree")
    reached, real_problems = _walk(real, entries)
    packages = sorted(f.split("/")[0] for f in reached
                      if f.count("/") == 1 and f.endswith("/__init__.py"))
    if not packages:
        pytest.skip("the walk reaches no package to plant a submodule into")
    deepest = sorted((f for f in reached if reached[f]),
                     key=lambda f: (-_chain(reached, f).count(" -> "), f))
    deep_a, deep_b = deepest[0], deepest[min(1, len(deepest) - 1)]
    for rel in reached:
        dst = os.path.join(str(tmp_path), *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(real.root, *rel.split("/")), dst)
    missing_submodule = packages[0] + ".planted_submodule_that_does_not_exist"
    missing_module = "planted_module_that_does_not_exist"
    with open(os.path.join(str(tmp_path), *deep_a.split("/")), "a", encoding="utf-8") as fh:
        fh.write("\n\nimport %s\n" % missing_submodule)
    with open(os.path.join(str(tmp_path), *deep_b.split("/")), "a", encoding="utf-8") as fh:
        fh.write("\n\ndef _planted_function_nobody_calls():\n    import %s\n" % missing_module)
    _, problems = _walk(_Tree(str(tmp_path)), entries)
    assert sorted((p.file, p.missing) for p in problems) == sorted(
        [(p.file, p.missing) for p in real_problems]
        + [(deep_a, missing_submodule), (deep_b, missing_module)])
