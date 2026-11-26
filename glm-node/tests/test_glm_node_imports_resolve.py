"""glm-node: every import the node can execute must resolve against the glm-node/ tree we publish.

ADAPTED from the public miner's gate tests/test_published_tree_imports_resolve.py (hardened 2026-09-30) -- same
resolver, same transitive walk, same exact-site allowlist, same positive controls -- pointed at glm-node/.

WHY (release 3.7.1, 2026-08-06): a release with a GREEN suite bricked every miner that took it, because a module
imported LAZILY inside a function existed only in the private repo. Nothing executes such an import until a real
node reaches it on a GPU, so only a STATIC walk of every import -- module level AND inside functions -- sees it.
glm-node has the same shape: the runner imports sharddiloco_glm_expert inside _streamed_lora_experts_cls and the
pack reader inside _train, and those pull in a dozen more files.

The walk is the TRANSITIVE closure of every glm-node entry point, function bodies included, resolved one dotted
component at a time against the directories the runner puts on sys.path (glm-node/, lib/runner/, tools/, lib/).
A module glm-node does not ship is tolerated only at the exact (file, def) sites listed in ABSENT_BY_DESIGN, and
-- beyond the public gate -- test_absent_sites_are_off_the_node_path proves with an over-approximate call graph
that every one of those sites is in a function no node entry point can reach (or, for the two listed in
REACHED_GUARDED, that the import is guarded and its fallback is the right behaviour).

NOT COVERED (as in the public gate): sys.path ORDER beyond the no-duplicates rule, pickled class names, circular
imports, third-party internals, and whether requirements-node.txt installs what THIRD_PARTY lists.

Static: nothing is imported or executed, nothing touches a GPU. Files are parsed at Python 3.10's grammar (the
fleet's 8 GB node runs 3.10.8).

Run: python -m pytest glm-node/tests/test_glm_node_imports_resolve.py -q
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
    "numpy", "torch", "transformers", "safetensors", "peft", "accelerate", "tokenizers", "huggingface_hub",
    "psutil", "cryptography", "pytest",
    # optional: never on the node path (each site sits in a function test_absent_sites_are_off_the_node_path
    # proves unreached), or the corpus rebuild, which needs them only when --rebuild is used:
    "bitsandbytes", "datasets", "pyarrow",
}

# Every file a miner or operator runs by path, and where that is documented. The gate walks the
# transitive closure of each. Scripts an entry point launches or loads by file name (run_glm_miner.py
# starts its children that way) are followed automatically, and
# test_every_documented_command_is_an_entry_point fails if the docs tell anyone to run a script that
# is not listed here.
SHIPPED_ENTRY_POINTS = {
    "lib/runner/glm47_h1x_chain.py": "the training node (glm_node.py runs it; README: --selftest)",
    "lib/runner/diloco_golden_check.py": "README: the golden digest check",
    "glm_node.py": "README: the launcher",
    "node_selfcheck.py": "README: the self-check",
    "build_packs.py": "README: the pack / base builder",
    "fetch_corpus.py": "README: the corpus fetch / rebuild",
    "sidecar/build_sidecar.py": "README: building the P2P sidecar",
}

# Modules this tree does not ship ON PURPOSE, and the ONLY sites allowed to import them:
# {missing module: {(file, enclosing def): why}}. Anything else that does not resolve fails -- a new
# module, a new file, or the same module from another function. An UNGUARDED entry is a known crash
# on a path the GLM lane never takes; its reason names the switch that reaches it.
# test_absent_by_design_has_no_stale_entries fails when a listed site stops importing its module,
# so this cannot rot into a blanket.
ABSENT_BY_DESIGN = {
    # ---- REACHED on the node path, GUARDED, with a correct fallback. test_absent_sites_are_off_the_node_path allows
    #      a reached site only when it is guarded AND named in REACHED_GUARDED below.
    "neurahash.storage_autotune": {
        ("tools/glm_arc_eval.py", "_st_resolve_threads"):
            "guarded (except Exception -> 1 read thread, 'autotune-unavailable'); not executed via glm_node.py, which "
            "sets NEURAHASH_ST_READ_THREADS=4 (the fleet's 8 GB node's value), so the function returns first",
    },
    "glm47_lora_to_gguf": {
        ("lib/runner/glm47_h1x_chain.py", "role_selftest"):
            "guarded (except ImportError): the GGUF-scorer shim that --selftest exercises when present; GLM-H1E5 "
            "nodes never convert to GGUF, and the selftest then notes 'shim import unavailable'",
    },
    # ---- NOT reached from any node entry point (proved by test_absent_sites_are_off_the_node_path)
    "sharddiloco_glm_contributor": {
        ("tools/glm_arc_eval.py", w): "UNGUARDED: the old shardDiLoCo lane's eval / fold stages of the reader's own "
        "CLI; a node imports glm_arc_eval as a library and never calls them"
        for w in ("spec_accepted_prefix", "validate_spec", "build_args", "stage_fold", "heldout_pool", "gate_ce",
                  "_ce_anchor", "_ce_prefix")},
    "sharddiloco_glm_coordinator": {
        ("tools/glm_arc_eval.py", w): "UNGUARDED: coordinator-side held-out scoring in the reader's CLI"
        for w in ("heldout_pool", "gate_ce")},
    "sharddiloco_harness": {
        ("tools/glm_arc_eval.py", "stage_fold"): "UNGUARDED: the reader CLI's fold stage",
    },
    "glm_grad_cache": {
        ("tools/glm_capture_layer_major.py", "drive_layer_major"):
            "UNGUARDED: the layer-cache producer's driver; lmbwd_core uses this module's adapter classes only",
    },
    "glm_layer_cache_pass": {
        ("tools/glm_capture_layer_major.py", "drive_layer_major"): "UNGUARDED: same driver, same reason",
    },
    "style_control_chain": {
        ("lib/wan_gradient_chain_ckpt.py", "role_chain"):
            "UNGUARDED: the OLMoE WAN-gradient experiment chain, another program mode of this module; the GLM runner "
            "uses it as a library (relay client, token, helpers) and never calls role_chain",
    },
    "paired_read": {("lib/wan_gradient_chain_ckpt.py", "role_chain"): "UNGUARDED: same chain, same reason"},
    "post_t30_chain": {("lib/wan_gradient_chain_ckpt.py", "role_chain"): "UNGUARDED: same chain, same reason"},
    "c4060": {
        ("lib/wan_gradient_chain_ckpt.py", "role_chain"):
            "UNGUARDED: the operator's own control client for one of its boxes, used only by that chain",
    },
    "resume_cert_watcher": {
        ("lib/wan_gradient_chain_ckpt.py", "role_chain.WanChain.dp2_post_run_void"): "UNGUARDED: same chain",
    },
    # scripts the OLMoE chain of wan_gradient_chain_ckpt.py LAUNCHES BY FILE NAME (never part of a GLM node run)
    "battery_eval.py": {
        ("lib/wan_gradient_chain_ckpt.py", "role_chain.WanChain.aa_phase"): "UNGUARDED: the OLMoE chain's scorer",
        ("lib/wan_gradient_chain_ckpt.py", "role_chain.WanChain.score_arm"): "UNGUARDED: the OLMoE chain's scorer",
    },
    "wan_gradient_chain.py": {
        ("lib/wan_gradient_chain_ckpt.py", "role_chain.WanChain.basece_phase"): "UNGUARDED: the OLMoE chain's base-CE step",
    },
    "s1_frozen_baseline.py": {
        ("lib/wan_gradient_chain_ckpt.py", "role_chain.WanChain.score_arm"): "UNGUARDED: the OLMoE chain's baseline",
    },
    "arc_falsifier.py": {
        ("lib/wan_gradient_chain_ckpt.py", "role_chain.WanChain.score_arm"): "UNGUARDED: the OLMoE chain's falsifier",
    },
    "_env_probe_local.py": {
        ("lib/wan_gradient_chain_ckpt.py", "local_env_probe"):
            "UNGUARDED: the OLMoE chain's environment probe (a script it writes and runs); not a GLM node path",
    },
    "neura_l1": {
        ("tools/diloco_contributor.py", "_miner_account"):
            "UNGUARDED: the public miner's wallet signing. glm-node ships diloco_contributor.py only because "
            "sharddiloco_glm_expert binds ShardDeltaLane from it at import time",
        ("tools/diloco_contributor.py", "publish_delta"): "UNGUARDED: the old lane's publish path",
        ("neurahash/diloco_merge.py", "_recover_contrib_signer"): "guarded: the old lane's merge",
        ("neurahash/miner_id.py", "_wallet_address"): "guarded: a wallet-derived miner id",
        ("neurahash_torch/shard_verify.py", "checkpoint_root"): "UNGUARDED: the old lane's checkpoint proofs",
        ("neurahash_torch/shard_verify.py", "checkpoint_proof"): "UNGUARDED: the old lane's checkpoint proofs",
        ("neurahash_torch/shard_verify.py", "verify_tile_transition"): "UNGUARDED: the old lane's tile proofs",
    },
    "neurahash.model": {
        ("tools/diloco_contributor.py", "moelm_sparse_fwd_flops"): "UNGUARDED: the old lane's FLOP estimate",
        ("tools/diloco_contributor.py", "train_expert_contribution"): "UNGUARDED: the old lane's expert trainer",
    },
    "neurahash.merkle_verify": {
        ("neurahash_torch/shard_verify.py", w): "UNGUARDED: the old lane's checkpoint proofs"
        for w in ("checkpoint_root", "checkpoint_proof")},
    "neurahash.training_layer": {
        ("neurahash/diloco_merge.py", "shard_merge_round"):
            "UNGUARDED: the coordinator's merge round; sharddiloco_glm_expert only BINDS the name at import",
    },
    "neurahash.grounding_corpora": {
        ("neurahash_torch/corpus_torch.py", w): "UNGUARDED: the private grounding-corpus lane (the public gate lists "
        "the same sites); reached only through NEURAHASH_CORPUS*, which glm_node.py clears"
        for w in ("_grounding_sources", "build_grounding_data", "corpus_sha")},
    "model_registry": {
        ("neurahash_torch/corpus_torch.py", "_qwen_tokenizer"): "guarded (except Exception): the retired Qwen path",
    },
    "neurahash_torch.pool_model": {
        ("neurahash/base_checkpoint.py", "_build_model"): "UNGUARDED: the old lane's checkpoint loader",
    },
}

# String literals that LOOK like a script file name (the gate treats any "<name>.py" literal as a launch) but are
# DATA, by exact (file, enclosing def, literal). Any other literal of that shape -- even in the same function -- is
# still a launch that must resolve (test_a_data_literal_exception_is_exact), and
# test_not_a_file_reference_entries_are_exact fails when a listed site drops its literal or the name starts to
# resolve to a shipped file (it could then be a real launch: delete the entry and let the walk follow it).
NOT_A_FILE_REFERENCE = {
    ("lib/runner/glm47_h1x_chain.py", "_s0_redact", "x.py"):
        "the Stage-0 runner's (s0-redact) selftest: a SYNTHETIC argv word of the TRAIN.json fixture that "
        "s0_strip_paths_obj path-strips; nothing imports, loads or launches a file of that name",
}

# ABSENT sites the node path DOES reach -- allowed only because they are guarded and the fallback is correct.
REACHED_GUARDED = {
    ("tools/glm_arc_eval.py", "_st_resolve_threads", "neurahash.storage_autotune"),
    ("lib/runner/glm47_h1x_chain.py", "role_selftest", "glm47_lora_to_gguf"),
}

# The runner puts lib/runner/ (its own dir), tools/ and lib/ on sys.path (glm47_h1x_chain.py:205-220: _HERE, _TOOLS,
# _SCR), and the tools modules add glm-node/ itself (glm_arc_eval.py / sharddiloco_glm_expert.py prologues).
# test_no_module_name_is_shipped_twice makes the ORDER irrelevant: no name exists in two of these dirs.
_SEARCH_DIRS = ("lib/runner", "tools", "lib", "")

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
    """The files ('a/b.py', relative to root) a fresh clone would hold under `root`, per `git ls-files` -- or None
    when root is not inside a git work tree (the synthetic trees the controls build) or git tracks nothing under it
    yet (an uncommitted glm-node/ during development): then the disk is the tree. Tracked files deleted from the
    working tree are left out: they cannot be read."""
    try:
        inside = _git(root, "rev-parse", "--is-inside-work-tree", encoding="utf-8", errors="replace", timeout=60)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
        ls = _git(root, "ls-files", "-z", timeout=180)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if ls.returncode != 0:
        return None
    names = [p.decode("utf-8", "replace") for p in ls.stdout.split(b"\0") if p]
    if not names:
        return None
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
        for d in (here, "lib/runner", "tools", "lib", ""):
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
                if rec.kind == "file" and (rel, rec.where, name) in NOT_A_FILE_REFERENCE:
                    continue                                    # a data literal of the file-name shape, by exact site
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
    """A published glm-node/ carries MANIFEST.sha256 (the synthetic control trees do not)."""
    return tree.kind("MANIFEST.sha256") == "file"


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
    r"(?:glm-node[/\\]((?:[A-Za-z0-9_]+[/\\])*[A-Za-z0-9_]+\.py)|-m\s+([A-Za-z_][A-Za-z0-9_.]*))")


def _documented_commands(tree):
    """{script or module the docs tell people to run: 'DOC.md:line'} -- a module that does not
    exist is reported by name; one that is not ours (`python -m pytest`) is left out."""
    found = {}
    for doc in ("README.md", "TROUBLESHOOTING.md"):
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
    assert "glm_node.py" in documented and "node_selfcheck.py" in documented, (
        "the README's commands were not recognised -- _DOC_COMMAND no longer reads the docs")
    unlisted = {s: where for s, where in documented.items()
                if s not in SHIPPED_ENTRY_POINTS and not s.startswith("tests/")}
    assert not unlisted, (
        "the docs tell people to run these, but SHIPPED_ENTRY_POINTS does not list them, so nothing "
        "checks that they exist or what they import: %s"
        % ", ".join("%s (%s)" % kv for kv in sorted(unlisted.items())))


def test_the_tree_is_read_from_git_when_it_is_a_checkout():
    """Blind spot 3 from the other side: if `git ls-files` ever failed here -- dubious ownership, a
    git that prints /c/ paths, no git on PATH -- the gate would fall back to the disk without a word,
    and an untracked module would pass again."""
    tree = _published()
    inside = _git(tree.root, "rev-parse", "--is-inside-work-tree", encoding="utf-8", errors="replace", timeout=60)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.skip("not a git checkout (an archive download): the disk is the tree")
    ls = _git(tree.root, "ls-files", timeout=180)
    if ls.returncode == 0 and not ls.stdout.strip():
        pytest.skip("glm-node/ is not committed yet: the disk is the tree until it is")
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


def test_the_lazy_imports_that_carry_the_node_are_followed():
    """Regression, named (the 3.7.1 shape, here). The runner imports sharddiloco_glm_expert INSIDE
    _streamed_lora_experts_cls and the pack reader INSIDE _train; the reader imports no_toy_models at its top. The
    walk must follow those nested imports, and every one of those files must be in this tree."""
    tree = _published()
    runner = "lib/runner/glm47_h1x_chain.py"
    if tree.kind(runner) != "file":
        pytest.skip("runner not in this tree")
    with open(os.path.join(tree.root, *runner.split("/")), "rb") as fh:
        raw = fh.read()                                     # the source, not the collector under test
    for needle in (b"import sharddiloco_glm_expert as SE", b"import glm_arc_eval as AE", b"import lmbwd_core"):
        assert needle in raw, "the runner no longer contains %r -- update this regression" % needle
    nested = {(r.module, r.where) for r in tree.imports(runner) if r.kind == "import"}
    assert ("sharddiloco_glm_expert", "_streamed_lora_experts_cls") in nested, (
        "the source imports sharddiloco_glm_expert inside _streamed_lora_experts_cls but _collect_imports does not "
        "see it -- the walk has stopped reading function bodies")
    reached, _ = _walk(tree, [runner])
    for f in ("tools/sharddiloco_glm_expert.py", "tools/glm_arc_eval.py", "tools/no_toy_models.py",
              "lib/lmbwd_core.py", "lib/wan_gradient_chain_ckpt.py", "tools/diloco_contributor.py",
              "tools/piece_loader.py", "tools/glm_capture_layer_major.py"):
        assert f in reached, "the runner reaches %s (nested imports) but it is not in this tree" % f


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


def test_not_a_file_reference_entries_are_exact():
    """Each NOT_A_FILE_REFERENCE entry must still sit at its site (else it is a stale pre-approved hole) and its name
    must NOT resolve to a shipped file (else the literal may be a real launch the walk has to follow)."""
    tree = _published()
    stale = []
    for (rel, where, lit) in sorted(NOT_A_FILE_REFERENCE):
        recs = [r for r in (tree.imports(rel) if tree.kind(rel) == "file" else [])
                if r.kind == "file" and r.where == where and r.module == lit]
        if not recs:
            stale.append("%s, in %s, no longer carries %r -- delete its entry" % (rel, where, lit))
        elif _resolve(tree, rel, recs[0])[0]:
            stale.append("%r in %s now names a shipped file -- it may be a real launch: delete its entry" % (lit, rel))
    assert not stale, "stale NOT_A_FILE_REFERENCE entries:\n  " + "\n  ".join(stale)


def test_a_data_literal_exception_is_exact(tmp_path):
    """POSITIVE CONTROL for NOT_A_FILE_REFERENCE: the listed literal at its listed site is skipped, but another
    file-name literal in the SAME function, and the same literal in another function, still fail the gate."""
    tree = _plant(tmp_path, {
        "lib/runner/glm47_h1x_chain.py": """
            def _s0_redact():
                man = {"argv": ["x.py", "--out-dir", "D:/runs/c"]}
                launch = ["y.py"]
                return man, launch

            def elsewhere():
                return ["x.py"]
        """,
    })
    assert _found(tree, ["lib/runner/glm47_h1x_chain.py"]) == [
        ("lib/runner/glm47_h1x_chain.py", "x.py"),          # the one in elsewhere()
        ("lib/runner/glm47_h1x_chain.py", "y.py"),
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


# ============================================================================ glm-node additions (not in the public gate)
# Function-level reachability. The public gate trusts a human for "no miner takes that path"; glm-node proves it. The
# call graph is deliberately OVER-approximate, so "not reached" is a strong statement: reaching a def reaches every
# def or class it names (bare names, `from m import f` aliases, `module_alias.attr`), every nested def, every method
# of a class it names, and -- for `obj.attr` with an unknown obj -- EVERY method of that name in any shipped module;
# an import statement executed in a reached scope runs the imported module's top level. Only `if __name__ ==
# "__main__":` blocks of imported modules are skipped (they do not run on import).

_NODE_ROOTS = (("lib/runner/glm47_h1x_chain.py", "<module>"), ("lib/runner/glm47_h1x_chain.py", "main"),
               ("lib/runner/diloco_golden_check.py", "<module>"), ("lib/runner/diloco_golden_check.py", "main"))
_COMMON_ATTRS = set("""append extend get set items keys values pop update add join split strip lower upper replace
format encode decode read write close open seek tell flush copy sort index count startswith endswith exists isfile
isdir makedirs remove rename dirname basename abspath item tolist numpy to cpu cuda float half bfloat16 view reshape
size shape dim sum mean max min clone detach zero_ fill_ copy_ add_ mul_ div_ sub_ __init__ __enter__ __exit__
hexdigest digest start run wait is_alive setdefault discard clear insert search match group groups sub findall
compile dumps loads dump load info warning error debug exception""".split())


class _Graph(object):
    """defs and references of every shipped .py file, keyed by (relpath, qualname); '<module>' = module top level."""

    def __init__(self, tree, files):
        self.tree, self.mods, self.defs, self.methods = tree, {}, {}, collections.defaultdict(list)
        self.by_module = {}
        for rel in files:
            with open(os.path.join(tree.root, *rel.split("/")), "rb") as fh:
                t = ast.parse(fh.read(), filename=rel, feature_version=_MINER_PYTHON)
            name = tree.module_of(rel)[0]
            self.by_module[name] = rel
            info = {"tree": t, "top": {}, "classes": {}, "alias": {}, "from": {}, "imports": []}
            self.mods[rel] = info
            self._index(rel, info, t, [], None)

    def _index(self, rel, info, node, ctx, cls):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = ".".join(ctx + [ch.name])
                self.defs[(rel, q)] = ch
                if not ctx:
                    info["top"][ch.name] = q
                if isinstance(ch, ast.ClassDef):
                    info["classes"][q] = []
                    self._index(rel, info, ch, ctx + [ch.name], q)
                else:
                    if cls is not None and ctx and ctx[-1] == cls.split(".")[-1]:
                        info["classes"][cls].append(q)
                        self.methods[ch.name].append((rel, q))
                    self._index(rel, info, ch, ctx + [ch.name], None)
                continue
            if isinstance(ch, ast.Import):
                for a in ch.names:
                    info["alias"][a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                    info["imports"].append((".".join(ctx) or "<module>", a.name, ()))
            elif isinstance(ch, ast.ImportFrom):
                base = ch.module or ""
                if ch.level:
                    base = _absolute(self.tree.module_of(rel)[1], ch.level, ch.module) or ""
                for a in ch.names:
                    info["from"][a.asname or a.name] = (base, a.name)
                info["imports"].append((".".join(ctx) or "<module>", base, tuple(a.name for a in ch.names)))
            self._index(rel, info, ch, ctx, cls)

    @staticmethod
    def _refs(node, top_level):
        names, attrs = set(), set()

        def walk(n):
            for ch in ast.iter_child_nodes(n):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for d in ch.decorator_list:
                        walk(d)
                    continue
                if top_level and isinstance(ch, ast.If) and _is_main_guard(ch.test):
                    continue
                if isinstance(ch, ast.Name) and isinstance(ch.ctx, ast.Load):
                    names.add(ch.id)
                elif isinstance(ch, ast.Attribute):
                    attrs.add((ch.value.id if isinstance(ch.value, ast.Name) else None, ch.attr))
                walk(ch)
        walk(node)
        if top_level:                                   # class bodies run at import: their non-def statements
            for st in node.body:
                if isinstance(st, ast.ClassDef):
                    for b in st.body:
                        if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            n2, a2 = _Graph._refs_stmt(b)
                            names |= n2
                            attrs |= a2
        return names, attrs

    @staticmethod
    def _refs_stmt(st):
        wrapper = ast.Module(body=[st], type_ignores=[])
        return _Graph._refs(wrapper, False)

    def reach(self, roots):
        seen, queue = {}, []

        def add(key, why):
            if key in self.defs or key[1] == "<module>":
                if key[0] in self.mods and key not in seen:
                    seen[key] = why
                    queue.append(key)
        for r in roots:
            add(r, "root")
        while queue:
            rel, q = queue.pop()
            info = self.mods[rel]
            why = "%s:%s" % (rel, q)
            for scope, base, names in info["imports"]:
                if scope != q:
                    continue
                for cand in [base] + ["%s.%s" % (base, n) for n in names]:
                    tgt = self.by_module.get(cand)
                    if tgt:
                        add((tgt, "<module>"), why)
            if q == "<module>":
                names, attrs = self._refs(info["tree"], True)
            else:
                node = self.defs[(rel, q)]
                names, attrs = self._refs(node, False)
                for (r2, q2) in list(self.defs):
                    if r2 == rel and q2.startswith(q + ".") and q2.count(".") == q.count(".") + 1:
                        add((r2, q2), why)
                for m in info["classes"].get(q, []):
                    add((rel, m), why)
            for nm in names:
                if q != "<module>" and (rel, q + "." + nm) in self.defs:
                    add((rel, q + "." + nm), why)
                if nm in info["top"]:
                    add((rel, info["top"][nm]), why)
                if nm in info["from"]:
                    base, attr = info["from"][nm]
                    tgt = self.by_module.get(base)
                    if tgt and attr in self.mods[tgt]["top"]:
                        add((tgt, self.mods[tgt]["top"][attr]), why)
                    sub = self.by_module.get("%s.%s" % (base, attr))
                    if sub:
                        add((sub, "<module>"), why)
                if nm in info["alias"] and self.by_module.get(info["alias"][nm]):
                    add((self.by_module[info["alias"][nm]], "<module>"), why)
            for base, at in attrs:
                tgt = None
                if base is not None and base in info["alias"]:
                    tgt = self.by_module.get(info["alias"][base])
                elif base is not None and base in info["from"]:
                    b0, a0 = info["from"][base]
                    tgt = self.by_module.get("%s.%s" % (b0, a0)) or None
                if tgt:
                    if at in self.mods[tgt]["top"]:
                        add((tgt, self.mods[tgt]["top"][at]), why)
                    continue
                if at in _COMMON_ATTRS:
                    continue
                for key in self.methods.get(at, []):
                    add(key, why + " (.%s)" % at)
        return seen


def _absent_reach_problems(tree, absent, reached_guarded, roots):
    """-> [(file, where, module, why)] for every ABSENT site the node path reaches that is not an allowed
    guarded one, plus every REACHED_GUARDED entry whose import is not actually guarded."""
    files = [f for f in (tree.tracked or _disk_py(tree.root)) if f.endswith(".py")]
    g = _Graph(tree, files)
    seen = g.reach(roots)
    out = []
    for module, sites in absent.items():
        for (rel, where) in sites:
            key = (rel, where)
            if key not in seen:
                continue
            guarded = any(r.where == where and r.guarded and (r.module == module or r.module.startswith(module + ".")
                                                            or module.startswith(r.module + "."))
                          for r in tree.imports(rel))
            if (rel, where, module) in reached_guarded and guarded:
                continue
            out.append((rel, where, module, "reached via %s%s" % (seen[key], "" if guarded else " and UNGUARDED")))
    return out


def _disk_py(root):
    out = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".git", "__pycache__")]
        for f in fs:
            out.append(os.path.relpath(os.path.join(d, f), root).replace("\\", "/"))
    return out


def test_absent_sites_are_off_the_node_path():
    """Every ABSENT_BY_DESIGN site is either unreachable from the node's entry points under the over-approximate
    call graph above, or listed in REACHED_GUARDED with a real try/except around it."""
    tree = _published()
    if tree.kind("lib/runner/glm47_h1x_chain.py") != "file":
        pytest.skip("runner not in this tree")
    problems = _absent_reach_problems(tree, ABSENT_BY_DESIGN, REACHED_GUARDED, _NODE_ROOTS)
    assert not problems, "ABSENT_BY_DESIGN sites the node can execute:\n  " + "\n  ".join(
        "%s:%s imports %s -- %s" % p for p in problems)


def test_the_reachability_check_catches_a_reached_absent_site(tmp_path):
    """POSITIVE CONTROL for the check above: a missing module imported in a function the entry point calls through a
    module alias, a nested def, a method call and a class reference must be reported; the same import in a function
    nobody calls must not be; a REACHED_GUARDED entry without a real guard must be reported."""
    tree = _plant(tmp_path, {
        "lib/runner/glm47_h1x_chain.py": """
            import helper

            def main():
                helper.go()
                h = helper.Holder()
                h.method_never_named_elsewhere()
                if __name__ == "__main__":
                    pass
        """,
        "lib/runner/diloco_golden_check.py": "def main():\n    pass\n",
        "tools/helper.py": """
            def go():
                def inner():
                    import gone_via_nested
                inner()

            class Holder:
                def method_never_named_elsewhere(self):
                    import gone_via_method

            def nobody_calls_me():
                import gone_but_unreached

            def claimed_guarded():
                import claimed_but_not_guarded

            if __name__ == "__main__":
                claimed_guarded()
        """,
    })
    absent = {"gone_via_nested": {("tools/helper.py", "go.inner"): "x"},
              "gone_via_method": {("tools/helper.py", "Holder.method_never_named_elsewhere"): "x"},
              "gone_but_unreached": {("tools/helper.py", "nobody_calls_me"): "x"},
              "claimed_but_not_guarded": {("tools/helper.py", "claimed_guarded"): "x"}}
    got = sorted((p[0], p[1], p[2]) for p in _absent_reach_problems(
        tree, absent, {("tools/helper.py", "claimed_guarded", "claimed_but_not_guarded")}, _NODE_ROOTS))
    assert got == [("tools/helper.py", "Holder.method_never_named_elsewhere", "gone_via_method"),
                   ("tools/helper.py", "go.inner", "gone_via_nested")], got
    # and the unguarded REACHED_GUARDED claim is caught once something reaches it
    tree2 = _plant(tmp_path / "t2", {
        "lib/runner/glm47_h1x_chain.py": "import helper\n\ndef main():\n    helper.claimed_guarded()\n",
        "lib/runner/diloco_golden_check.py": "",
        "tools/helper.py": "def claimed_guarded():\n    import claimed_but_not_guarded\n",
    })
    got2 = _absent_reach_problems(tree2, {"claimed_but_not_guarded": {("tools/helper.py", "claimed_guarded"): "x"}},
                                  {("tools/helper.py", "claimed_guarded", "claimed_but_not_guarded")}, _NODE_ROOTS)
    assert [(p[0], p[2]) for p in got2] == [("tools/helper.py", "claimed_but_not_guarded")], got2


def test_no_module_name_is_shipped_twice():
    """The runner and the tools modules put four dirs on sys.path in an order that differs between entry points. A
    module name that exists in two of them would resolve differently depending on who imported first -- so none
    may (this is what makes the gate's fixed _SEARCH_DIRS order safe)."""
    tree = _published()
    seen = collections.defaultdict(list)
    for d in _SEARCH_DIRS:
        for name, kind in tree.listdir(d).items():
            if kind == "file" and name.endswith(".py"):
                seen[name[:-3]].append(d or ".")
            elif kind == "dir" and tree.listdir(_join(d, name)).get("__init__.py") == "file":
                seen[name].append(d or ".")
    dup = dict((k, v) for k, v in seen.items() if len(v) > 1)
    assert not dup, "module names shipped in more than one sys.path dir: %s" % dup
