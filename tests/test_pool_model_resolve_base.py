"""pool_model's Qwen dense-base path must not die on an unguarded lazy `import model_registry`.

model_registry is a PRIVATE-repo alias table; the public miner does not ship it, and
neurahash_torch/pool_model.py imported it unguarded inside qwen_arch() and _load_base_into() -- the
3.7.1 shape (a lazy import no test executes), found by tests/test_published_tree_imports_resolve.py
on 2026-09-30. _resolve_base() now uses the registry where the tree has it and otherwise applies the
registry's own pass-through rule. These tests pin both halves in either repo, by putting a fake
registry -- or None, which makes the import fail exactly as it does in the public tree -- into
sys.modules.

Run: C:/Python313/python.exe -m pytest tests/test_pool_model_resolve_base.py -q
"""
import sys
import types

import pytest

from neurahash_torch import pool_model


def test_without_model_registry_a_raw_hf_id_passes_through(monkeypatch):
    monkeypatch.setitem(sys.modules, "model_registry", None)
    assert pool_model._resolve_base("Qwen/Qwen3-1.7B") == "Qwen/Qwen3-1.7B"


def test_without_model_registry_a_bare_alias_fails_with_the_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "model_registry", None)
    with pytest.raises(ModuleNotFoundError, match="full Hugging Face id") as exc:
        pool_model._resolve_base("qwen3-1.7b")
    assert exc.value.name == "model_registry"


def test_with_model_registry_it_is_the_only_resolver(monkeypatch):
    registry = types.ModuleType("model_registry")
    registry.resolve_model = lambda name: "resolved:" + name
    monkeypatch.setitem(sys.modules, "model_registry", registry)
    assert pool_model._resolve_base("qwen3-1.7b") == "resolved:qwen3-1.7b"
    assert pool_model._resolve_base("Qwen/Qwen3-1.7B") == "resolved:Qwen/Qwen3-1.7B"
