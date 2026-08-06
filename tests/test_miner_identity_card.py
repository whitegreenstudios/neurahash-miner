"""A miner must be able to say what box it is -- and must never fail to mine because it couldn't.

WHY THIS EXISTS (2026-08-06). Until v3.7.1 a miner told the pool nothing about its hardware. The
operator could see that `glm-c47c9334` published a record but not whether that was a 4060 or an
H100, how much VRAM it had, or what client version it ran. The live dashboard only knew about cards
it scraped itself -- its own nvidia-smi, plus one LAN control agent -- which by construction
excludes every actual stranger. So "did anyone join, and on what hardware?" was unanswerable for
exactly the miners we most wanted to see.

The identity card fixes that with one small self-reported JSON object per miner. Two properties are
pinned here, and the SECOND is the one that matters:

  1. The card carries the facts (GPU name, VRAM, version) under stable key names the dashboard reads.
  2. Publishing it CANNOT break mining. It is a dashboard nicety sitting in the startup path of a
     process whose job is to earn. A store hiccup, a missing CUDA runtime, a broken VERSION file --
     none of them may raise into the mining loop. A miner that cannot publish a card must simply
     show as "GPU not reported" and keep earning.

Also pinned: no bare NaN ever reaches the JSON. Python's json module happily EMITS `NaN`, which is
invalid JSON that real parsers (and every browser) reject -- so one non-finite float would take the
whole public dashboard down rather than blanking one field.

Run: C:/Python313/python.exe -m pytest tests/test_miner_identity_card.py -q
"""
import json
import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOLS = os.path.join(_REPO, "tools")
for _p in (_REPO, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sharddiloco_glm_contributor as N                               # noqa: E402


class _FakeLane:
    """Records what was PUT. `boom` makes every PUT raise, standing in for a dead content store."""

    def __init__(self, boom=False):
        self.puts = []
        self.boom = boom

    def put_json_named(self, name, obj):
        if self.boom:
            raise RuntimeError("content store unreachable")
        self.puts.append((name, obj))
        return "cid-%d" % (len(self.puts),)


def _silent(*_a, **_k):
    pass


# ---- naming -----------------------------------------------------------------------------------

def test_card_name_is_namespaced_per_miner():
    """The dashboard globs this prefix, so the shape is a contract, not an implementation detail."""
    assert N.miner_card_name("glm-c47c9334") == "miners/glm-c47c9334/card.json"
    assert N.miner_card_name("a") != N.miner_card_name("b")


# ---- the card contents ------------------------------------------------------------------------

def test_card_carries_the_keys_the_dashboard_renders():
    card = N.build_miner_card("glm-test", address="0xabc", coord="L1,E44", mode="async")
    for key in ("schema", "miner_id", "address", "gpu_name", "vram_total_gb", "vram_free_gb",
                "vram_cap_gb", "version", "platform", "python", "coord", "mode", "published_ts"):
        assert key in card, "dashboard reads %r; dropping it silently blanks a panel" % (key,)
    assert card["miner_id"] == "glm-test"
    assert card["address"] == "0xabc"
    assert card["coord"] == "L1,E44"
    assert card["mode"] == "async"


def test_a_box_with_no_cuda_reports_gpu_none_instead_of_failing(monkeypatch):
    """The Colab CPU miner is a real, supported case -- it minted over WAN. It must still get a card."""
    monkeypatch.setitem(sys.modules, "torch", None)             # `import torch` -> raises/None attrs
    card = N.build_miner_card("cpu-miner")
    assert card["gpu_name"] is None
    assert card["gpu_count"] == 0
    assert card["miner_id"] == "cpu-miner"


def test_extra_fields_merge_without_clobbering_the_schema():
    card = N.build_miner_card("m", extra={"pod": "runpod-xyz"})
    assert card["pod"] == "runpod-xyz"
    assert card["schema"] == 1


# ---- the JSON must be REAL json ---------------------------------------------------------------

def test_non_finite_floats_become_null_not_bare_NaN():
    """json.dumps emits a bare `NaN` token, which is invalid JSON. One of those would break the
    whole public page rather than blanking one field."""
    card = N.build_miner_card("m", extra={"bad": float("nan"), "worse": float("inf")})
    assert card["bad"] is None and card["worse"] is None
    blob = json.dumps(card)
    assert "NaN" not in blob and "Infinity" not in blob
    # A strict parser must accept it. parse_constant fires only on NaN/Infinity/-Infinity.
    def _boom(tok):                                             # pragma: no cover - must not fire
        raise AssertionError("non-finite constant %r reached the JSON" % (tok,))
    json.loads(blob, parse_constant=_boom)


def test_every_value_is_json_serialisable():
    card = N.build_miner_card("m", address="0xabc")
    json.dumps(card)                                            # raises TypeError if anything isn't


# ---- VERSION reading --------------------------------------------------------------------------

@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_version_survives_the_encodings_a_release_has_actually_been_cut_in(tmp_path, monkeypatch,
                                                                           encoding):
    """A past release wrote VERSION with PowerShell `echo x > VERSION`, i.e. UTF-16, and the naive
    reader saw garbage -- the signed-manifest-vs-VERSION mismatch incident. Read it BOM-tolerantly."""
    fake_tools = tmp_path / "tools"
    fake_tools.mkdir()
    (tmp_path / "VERSION").write_text("3.7.1\n", encoding=encoding)
    monkeypatch.setattr(N, "__file__", str(fake_tools / "sharddiloco_glm_contributor.py"))
    assert N._local_client_version() == "3.7.1"


def test_a_missing_VERSION_is_none_not_an_exception(tmp_path, monkeypatch):
    fake_tools = tmp_path / "tools"
    fake_tools.mkdir()
    monkeypatch.setattr(N, "__file__", str(fake_tools / "sharddiloco_glm_contributor.py"))
    assert N._local_client_version() is None


# ---- THE PROPERTY THAT MATTERS: publishing cannot stop mining ---------------------------------

def test_publish_puts_the_card_under_the_right_name():
    lane = _FakeLane()
    assert N.publish_miner_card(lane, "glm-x", address="0xa", coord="L1,E2", mode="async",
                                log=_silent) is True
    assert len(lane.puts) == 1
    name, obj = lane.puts[0]
    assert name == "miners/glm-x/card.json"
    assert obj["miner_id"] == "glm-x" and obj["coord"] == "L1,E2"


def test_a_dead_content_store_does_not_raise_into_the_mining_loop():
    """THE point of the whole file. If this ever regresses, a store outage stops miners earning."""
    lane = _FakeLane(boom=True)
    assert N.publish_miner_card(lane, "glm-x", log=_silent) is False   # reported, not raised


def test_the_failure_is_announced_and_names_the_visible_consequence():
    """A silent False would leave the operator wondering why a miner shows no GPU. Say it out loud."""
    seen = []
    lane = _FakeLane(boom=True)
    N.publish_miner_card(lane, "glm-x", log=lambda m: seen.append(m))
    assert seen, "a failed card publish must log something"
    joined = " ".join(seen)
    assert "NOT published" in joined
    assert "GPU not reported" in joined, "tell the operator what they will SEE on the dashboard"


def test_a_lane_that_is_not_even_a_lane_is_survived():
    """Defence in depth: None/garbage must be caught by the same guard, not AttributeError upward."""
    assert N.publish_miner_card(None, "glm-x", log=_silent) is False
    assert N.publish_miner_card(object(), "glm-x", log=_silent) is False


def test_a_broken_card_builder_still_cannot_raise(monkeypatch):
    """Even if collection itself explodes, the miner keeps going."""
    monkeypatch.setattr(N, "build_miner_card",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("probe exploded")))
    assert N.publish_miner_card(_FakeLane(), "glm-x", log=_silent) is False


def test_success_path_announces_the_hardware():
    """The startup line is how the operator confirms, from the miner's own stdout, what it reported."""
    seen = []
    N.publish_miner_card(_FakeLane(), "glm-x", log=lambda m: seen.append(m))
    assert any("identity card published" in m for m in seen)
