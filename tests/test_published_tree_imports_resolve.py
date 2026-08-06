"""Every import in the shipped miner must resolve against the tree we actually publish.

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

This test is that check. It is deliberately static (no imports are executed, nothing is loaded onto
a GPU), so it is fast and safe to run everywhere, and it fails for exactly the reason a miner would.

Run: C:/Python313/python.exe -m pytest tests/test_published_tree_imports_resolve.py -q
"""
import ast
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")

# Distributed on PyPI, not by us -- requirements.txt's problem, not the tree's.
THIRD_PARTY = {
    "numpy", "torch", "transformers", "safetensors", "requests", "eth_account", "eth_keys",
    "eth_utils", "eth_hash", "coincurve", "tqdm", "huggingface_hub", "datasets", "psutil",
    "yaml", "regex", "sentencepiece", "tokenizers", "accelerate", "scipy", "pytest",
    "setuptools", "pkg_resources", "pytest_timeout", "bitsandbytes", "peft", "hf_transfer",
}

# Pre-existing and correct when the tool is run from the repo root, which is how they are invoked.
KNOWN_ROOT_RELATIVE = {"tools"}

# The files a miner actually executes. Keep this list SMALL and load-bearing: it is the set whose
# breakage takes a miner down, and every name here must stay runnable from a fresh clone.
SHIPPED_ENTRY_POINTS = [
    "sharddiloco_glm_contributor.py",
    "sharddiloco_glm_expert.py",
    "self_update.py",
    "fetch_glm_base.py",
    "piece_loader.py",
]


def _local_module_names():
    """Everything importable from this repo: top-level tool modules and package submodules."""
    names = set()
    if os.path.isdir(_TOOLS):
        names |= {f[:-3] for f in os.listdir(_TOOLS) if f.endswith(".py")}
    names |= {f[:-3] for f in os.listdir(_REPO) if f.endswith(".py")}
    for pkg in ("neurahash", "neurahash_torch", "neura_l1"):
        p = os.path.join(_REPO, pkg)
        if os.path.isdir(p):
            names.add(pkg)
            names |= {"%s.%s" % (pkg, f[:-3]) for f in os.listdir(p) if f.endswith(".py")}
    return names


def _imports_of(path):
    """(lineno, module) for every import, INCLUDING ones nested inside functions -- which is the
    whole point: the 3.7.1 break was an import inside build_node_model()."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.lineno, node.module))
    return out


def _unresolvable(path, local):
    bad = []
    for lineno, mod in _imports_of(path):
        top = mod.split(".")[0]
        if top in sys.stdlib_module_names or top in THIRD_PARTY:
            continue
        if mod in local or top in local or top in KNOWN_ROOT_RELATIVE:
            continue
        bad.append((lineno, mod))
    return bad


@pytest.mark.parametrize("name", SHIPPED_ENTRY_POINTS)
def test_every_import_in_a_shipped_entry_point_resolves(name):
    """A miner runs these from a fresh clone. An unresolvable import here is a dead miner."""
    path = os.path.join(_TOOLS, name)
    if not os.path.exists(path):
        pytest.skip("%s is not present in this tree" % (name,))
    bad = _unresolvable(path, _local_module_names())
    assert not bad, (
        "tools/%s imports %d module(s) this tree does not contain: %s\n"
        "A miner cloning this repo would crash. If the module lives in the private repo, copy it "
        "here before releasing." % (name, len(bad), ", ".join("%s (line %d)" % (m, l) for l, m in bad))
    )


def test_the_module_that_actually_broke_3_7_1_is_present():
    """Regression, named. build_node_model() does `import no_toy_models` lazily at ~line 2854."""
    contributor = os.path.join(_TOOLS, "sharddiloco_glm_contributor.py")
    if not os.path.exists(contributor):
        pytest.skip("contributor not in this tree")
    with open(contributor, encoding="utf-8") as fh:
        src = fh.read()
    if "import no_toy_models" not in src:
        pytest.skip("the contributor no longer imports no_toy_models")
    assert os.path.exists(os.path.join(_TOOLS, "no_toy_models.py")), (
        "sharddiloco_glm_contributor.py imports no_toy_models but tools/no_toy_models.py is absent. "
        "This exact gap shipped as 3.7.1 and took the reference 4060 down.")


def test_lazy_imports_are_actually_being_checked():
    """POSITIVE CONTROL. If _imports_of ever stopped walking into function bodies this file would
    pass vacuously and be worse than useless -- it would read as coverage while checking nothing."""
    import tempfile
    src = "import os\n\n\ndef f():\n    import a_module_nested_inside_a_function\n    return 1\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(src)
        tmp = fh.name
    try:
        found = {m for _l, m in _imports_of(tmp)}
        assert "a_module_nested_inside_a_function" in found, "nested imports are NOT being walked"
        assert "os" in found
    finally:
        os.unlink(tmp)
