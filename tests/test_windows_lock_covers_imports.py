"""The Windows lock must provide every third-party module a miner can execute.

WHY. Three quickstarts died at their LAST step on a module nobody had declared: transformers
(2026-08-03, after a 6.33 GB download), accelerate (2026-08-06, after a 5.67 GB download) and pytest
(issue #71). Each was invisible on every machine that had ever mined, because those machines already
had it. The Windows installer installs ONLY `requirements-win-cu128.lock` (`uv pip sync
--require-hashes`), so anything the miner imports and the lock does not carry fails on a fresh PC and
nowhere else.

HOW. The walk is the import gate's own (tests/test_published_tree_imports_resolve.py): the
transitive closure of every MINER entry point, function bodies included, over the git-tracked tree.
Every third-party top-level module it reaches must map to a distribution pinned in the lock, unless it
is listed in OPTIONAL with the opt-in that reaches it -- and an OPTIONAL module may only be imported
under a try/except ImportError, so a missing optional package costs a feature, never a startup crash.
A positive control proves the check fires on a lock that drops a needed package, and that the walk
reaches imports inside function bodies (accelerate is imported only inside build_partial_model()).
"""
import os
import re
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_published_tree_imports_resolve as GATE        # noqa: E402

LOCK = os.path.join(GATE._REPO, "requirements-win-cu128.lock")

# import name -> PyPI distribution name, for every name the import gate classifies as third party.
IMPORT_TO_DIST = {
    "numpy": "numpy", "torch": "torch", "transformers": "transformers", "safetensors": "safetensors",
    "requests": "requests", "eth_account": "eth-account", "eth_keys": "eth-keys", "eth_utils": "eth-utils",
    "eth_hash": "eth-hash", "coincurve": "coincurve", "tqdm": "tqdm", "huggingface_hub": "huggingface-hub",
    "datasets": "datasets", "psutil": "psutil", "yaml": "pyyaml", "regex": "regex",
    "sentencepiece": "sentencepiece", "tokenizers": "tokenizers", "accelerate": "accelerate",
    "scipy": "scipy", "pytest": "pytest", "setuptools": "setuptools", "pkg_resources": "setuptools",
    "pytest_timeout": "pytest-timeout", "bitsandbytes": "bitsandbytes", "peft": "peft",
    "hf_transfer": "hf-transfer", "cryptography": "cryptography", "argon2": "argon2-cffi",
    "dilithium_py": "dilithium-py", "web3": "web3",
}

# Reachable from a miner entry point, but only behind an explicit opt-in; deliberately NOT in the lock.
OPTIONAL = {
    "dilithium_py": "opt-in post-quantum admission (NEURAHASH_PQC=hybrid), neura_l1/pqc.py _backend",
}


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(text):
    """{normalised distribution name: version} for every `name==version` line of a uv/pip lock."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)", line)
        if m:
            out[_norm(m.group(1))] = m.group(2)
    return out


def miner_entries(tree):
    return [e for e in GATE._entries(tree) if not GATE.SHIPPED_ENTRY_POINTS[e].startswith("operator")]


def third_party_sites(tree, entries):
    """{top-level third-party module: [(file, line, enclosing def, guarded)]} over the miner closure."""
    reached, _ = GATE._walk(tree, entries)
    sites = {}
    for rel in reached:
        for rec in tree.imports(rel):
            if rec.kind == "file" or rec.level:
                continue
            top = rec.module.split(".")[0]
            if top in GATE.THIRD_PARTY:
                sites.setdefault(top, []).append((rel, rec.line, rec.where, rec.guarded))
    return sites


def lock_problems(sites, pins, optional=None):
    optional = OPTIONAL if optional is None else optional
    probs = []
    for top, where in sorted(sites.items()):
        first = "%s:%d in %s" % where[0][:3]
        if top in optional:
            for rel, line, fn, guarded in where:
                if not guarded:
                    probs.append("%s is OPTIONAL but imported UNGUARDED at %s:%d in %s" % (top, rel, line, fn))
            continue
        dist = IMPORT_TO_DIST.get(top)
        if dist is None:
            probs.append("%s (first at %s) has no IMPORT_TO_DIST entry" % (top, first))
        elif _norm(dist) not in pins:
            probs.append("%s -> %s is imported (first at %s) but not pinned in the lock" % (top, dist, first))
    return probs


@pytest.fixture(scope="module")
def tree():
    return GATE._published()


@pytest.fixture(scope="module")
def lock_text():
    if not os.path.exists(LOCK):
        pytest.fail("requirements-win-cu128.lock is missing from the tree")
    with open(LOCK, encoding="utf-8") as fh:
        return fh.read()


def test_the_lock_provides_every_module_a_miner_imports(tree, lock_text):
    probs = lock_problems(third_party_sites(tree, miner_entries(tree)), lock_pins(lock_text))
    assert not probs, "the Windows lock does not cover the miner:\n  " + "\n  ".join(probs)


def test_positive_control_a_lock_missing_a_needed_package_fails(tree, lock_text):
    """Drop accelerate from a COPY of the lock: the check must name it. accelerate is imported only
    inside piece_loader.build_partial_model(), so this also proves the walk reads function bodies."""
    sites = third_party_sites(tree, miner_entries(tree))
    assert "accelerate" in sites, "the walk no longer reaches piece_loader's accelerate import"
    assert all(fn != "<module>" for _, _, fn, _ in sites["accelerate"]), \
        "accelerate is expected only inside a function body; update this control"
    sabotaged = "\n".join(l for l in lock_text.splitlines() if not l.lower().startswith("accelerate=="))
    probs = lock_problems(sites, lock_pins(sabotaged))
    assert any(p.startswith("accelerate ->") for p in probs), probs


def test_positive_control_an_unguarded_optional_import_fails():
    sites = {"dilithium_py": [("neura_l1/pqc.py", 1, "<module>", False)]}
    assert lock_problems(sites, {}) == [
        "dilithium_py is OPTIONAL but imported UNGUARDED at neura_l1/pqc.py:1 in <module>"]


def test_optional_entries_are_still_reached(tree):
    """A stale OPTIONAL entry is a pre-approved hole; drop it when the miner stops importing it."""
    sites = third_party_sites(tree, miner_entries(tree))
    stale = sorted(set(OPTIONAL) - set(sites))
    assert not stale, "OPTIONAL entries no longer imported by the miner: %s" % stale


def test_every_lock_pin_is_hash_pinned_and_torch_is_cu128(lock_text):
    blocks = re.split(r"\n(?=[A-Za-z0-9])", lock_text)
    pinned = [b for b in blocks if re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*==", b)]
    assert len(pinned) >= 10, "the lock looks truncated (%d pins)" % len(pinned)
    unhashed = [b.split()[0] for b in pinned if "--hash=sha256:" not in b]
    assert not unhashed, "pins without --hash (uv pip sync --require-hashes would refuse): %s" % unhashed
    assert lock_pins(lock_text).get("torch", "").endswith("+cu128"), "torch must be the cu128 build"
