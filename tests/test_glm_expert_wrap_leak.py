"""REGRESSION: a failed training round must not leave the model wrapped in LoRAExperts.

MEASURED on the live 5090, 2026-07-25 (scratchpad/nu5090.log): at async round 27 a CUDA OOM hit
mid-training. The contributor's designed self-heal caught it exactly as intended --

    [glm-contrib glm-396Ec4c3] round SKIPPED: CUDA OOM under memory pressure -- freeing cache +
    pausing, will retry next round (CUDA out of memory. Tried to allocate 2.00 MiB. ...)
    [glm-contrib glm-396Ec4c3] VRAM recovered (1 unit(s)) -- resuming after 4 wait(s)

-- and then the miner died on the very next round:

    File "tools/sharddiloco_glm_expert.py", line 96, in __init__
        H = base.hidden_dim
    AttributeError: 'LoRAExperts' object has no attribute 'hidden_dim'

Cause: train_glm_expert_contribution() did `layer.mlp.experts = le` on entry and only restored it
on the SUCCESS path, so the escaping OOM left the wrapper installed; the next call read that
wrapper as `base` and tried to wrap a wrapper. Net effect: every *recoverable* training error --
OOM, NaN, a bad batch -- silently converted the recovery path into a permanent kill. The miner was
down ~18 hours and the coordinator ran with one empty slot.

These tests drive the control flow only (stub LoRA class + stub eval), because the bug was control
flow: it reproduces with any base object and needs no GLM weights.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

X = pytest.importorskip("sharddiloco_glm_expert")


class _FusedBase:
    """Stands in for the frozen fused experts (Glm4MoeLiteNaiveMoe). The real one has hidden_dim;
    the LoRA wrapper does NOT -- that asymmetry is what turned the leak into a crash."""

    hidden_dim = 8

    def parameters(self):
        return iter(())


class _StubLoRA:
    """Minimal stand-in for LoRAExperts: wraps a base and exposes .base, like the real class."""

    def __init__(self, base, node_of, r=16, alpha=None):
        import torch

        # Reproduce the real failure mode: wrapping a wrapper explodes here, exactly as the real
        # LoRAExperts.__init__ does on `H = base.hidden_dim`.
        self.hidden_dim_probe = base.hidden_dim
        self.base = base
        self.node_of = dict(node_of)
        self.enabled_nodes = set()
        self.outer = 1.0
        for name in ("A_gu", "B_gu", "A_d", "B_d"):
            setattr(self, name, {str(e): torch.zeros(2) for e in self.node_of})

    def params_for(self, node):
        return []


class _Layer:
    def __init__(self, experts):
        self.mlp = type("MLP", (), {"experts": experts})()


class _Model:
    def __init__(self, layer):
        self.model = type("Inner", (), {"layers": [layer]})()

    def parameters(self):
        import torch

        return iter([torch.zeros(1)])            # the trainer reads .device off the first param

    def train(self):
        return self

    def __call__(self, **kw):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB.")


def _patch(monkeypatch, boom_in_eval=False):
    monkeypatch.setattr(X, "_lora_experts_cls", lambda: _StubLoRA)
    monkeypatch.setattr(X, "glm_fwd_flops_per_example", lambda cfg, seq: 1.0)
    monkeypatch.setattr(X, "_materialize_canonical", lambda le, E: {"gate": np.zeros(1, np.float32)})
    monkeypatch.setattr(X, "lora_factors_payload", lambda le, E: {})

    def _ce(model, ids):
        if boom_in_eval:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB.")
        return 1.0

    monkeypatch.setattr(X, "heldout_ce", _ce)

    class _Opt:
        def __init__(self, *a, **k):
            pass

        def zero_grad(self):
            pass

        def step(self):
            pass

    # The trainer does a function-local `import torch`, so patch the real module (same object it
    # resolves). AdamW would otherwise reject the stub's empty parameter list.
    import torch

    monkeypatch.setattr(torch.optim, "AdamW", _Opt)


def _run(monkeypatch, **kw):
    _patch(monkeypatch, **kw)
    fused = _FusedBase()
    layer = _Layer(fused)
    model = _Model(layer)
    ids = np.zeros((4, 4), dtype=np.int64)
    with pytest.raises(RuntimeError, match="out of memory"):
        X.train_glm_expert_contribution(model, object(), 0, 0, ids, ids, H=4, batch=2)
    return layer, fused


def test_training_failure_restores_the_fused_experts(monkeypatch):
    """The OOM escapes (the caller's self-heal handles it) but the layer must be left CLEAN."""
    layer, fused = _run(monkeypatch)
    assert layer.mlp.experts is fused, (
        "after a failed round the layer must hold the FUSED experts again, not the LoRA wrapper -- "
        "otherwise the next round wraps a wrapper and the miner dies permanently"
    )


def test_failure_during_eval_also_restores(monkeypatch):
    """The first thing the trainer does after wrapping is _sel_val(); a failure THERE leaked too."""
    layer, fused = _run(monkeypatch, boom_in_eval=True)
    assert layer.mlp.experts is fused


def test_a_second_round_after_a_failed_round_still_works(monkeypatch):
    """The actual user-visible symptom: round N+1 after a failed round N. Pre-fix this raised
    AttributeError: 'LoRAExperts' object has no attribute 'hidden_dim' and killed the process."""
    layer, fused = _run(monkeypatch)

    # Round N+1: same model object, this time training succeeds.
    monkeypatch.setattr(_Model, "__call__",
                        lambda self, **kw: type("Out", (), {"loss": _NoopLoss()})())
    ids = np.zeros((4, 4), dtype=np.int64)
    out = X.train_glm_expert_contribution(layer_model(layer), object(), 0, 0, ids, ids, H=4, batch=2)
    assert "delta" in out
    assert layer.mlp.experts is fused, "the successful round must also leave the layer clean"


def test_defensive_unwrap_recovers_an_already_poisoned_model(monkeypatch):
    """Belt-and-braces: a model object that arrives already wrapped (from an older build, or any
    leak path we have not found) must be unwrapped rather than killing the miner."""
    _patch(monkeypatch)
    fused = _FusedBase()
    poisoned = _StubLoRA(fused, {0: 0})          # simulate the leaked wrapper
    layer = _Layer(poisoned)
    monkeypatch.setattr(_Model, "__call__",
                        lambda self, **kw: type("Out", (), {"loss": _NoopLoss()})())
    ids = np.zeros((4, 4), dtype=np.int64)
    out = X.train_glm_expert_contribution(layer_model(layer), object(), 0, 0, ids, ids, H=4, batch=2)
    assert "delta" in out
    assert layer.mlp.experts is fused, "unwrap must reach the real fused experts, not a nested wrapper"


class _NoopLoss:
    def backward(self):
        pass


def layer_model(layer):
    return _Model(layer)
