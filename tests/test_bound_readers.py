"""Reader side of the mint-split binding ("Stage 0", blocker B2) in the PUBLIC client.

The GLM coordinator will sign each settlement mint's per-address payout split into the block header
(BlockHeader.split_hash = sha256 of a canonical split string, neura_l1.block_state.mint_split_hash). A released
client that replays the settlement chain (verify_public_chain, ChainSettlement.from_state, the wallet's
`--balance`, the round-authority Candidate.accept) must understand that field: before this change the public
BlockHeader.from_dict DROPPED it, so the recomputed header hash differed from the one the coordinator signed, the
signature recovered to an unrelated address, and the whole chain was rejected ("block 0: signed by 0x..., not the
coordinator") -- blocker B2.

Fixtures (tests/fixtures/, throwaway keys = sha256('bound-readers-fixture/<label>')):
  settlement_chain_bound_private.json  4 bound blocks made by the PRIVATE coordinator code's
                                       settle(bind_mint_split=True) (branch stage0-mint-split)
  settlement_chain_legacy_main.json    4 legacy blocks made by public main 3.9.3's settle()
Both record the balances / minted / height / per-block header hashes their own code computed.

CPU only: no torch, no CUDA.
"""
import copy
import hashlib
import json
import os

import pytest

from neura_l1.block_state import BlockHeader, mint_split_hash
from neura_l1.signing import account_from_key
from neurahash.chain_settlement import (BIND_MINT_SPLIT_ENV, ChainSettlement, verify_public_chain,
                                        verify_public_chain_report)

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SPLIT_REASON = "contributors do not match header.split_hash"
T0 = 1_760_000_000


@pytest.fixture(autouse=True)
def _no_env_flags(monkeypatch):
    monkeypatch.delenv(BIND_MINT_SPLIT_ENV, raising=False)


def _load(name):
    with open(os.path.join(FIX, name), encoding="ascii") as fh:
        return json.load(fh)


def _acct(label):
    return account_from_key(hashlib.sha256(("bound-readers-fixture/" + label).encode()).hexdigest())


def _rebuild(bind):
    """The same 4 settles the fixture generator ran, with THIS (public) code."""
    coord = _acct("coord")
    a, b, c = _acct("alice").address, _acct("bob").address, _acct("carol").address
    cs = ChainSettlement(coord, genesis_checkpoint="pool-genesis", bind_mint_split=bind)
    for i, sp in enumerate([{a: 1.0}, {a: 0.25, b: 0.75}, {c: 3.0, a: 1.5, b: 0.1}, {b: 2.0 / 3.0, c: 1e-9}]):
        cs.settle(i + 1, 0.0123 * (i + 1), sp, T0 + 60 * i)
    return cs


def _check_against_fixture(fx, report):
    exp = fx["expected"]
    assert report["ok"], report["reason"]
    assert report["balances"] == exp["balances"]          # exact float equality, not approx
    assert report["minted"] == exp["minted"]
    assert report["height"] == exp["height"]
    assert [BlockHeader.from_dict(r["header"]).hash() for r in fx["export"]["blocks"]] == exp["header_hashes"]


# (a) legacy chains verify exactly as before ------------------------------------------------------------------

def test_a_legacy_chain_made_by_public_main_verifies_to_identical_balances_and_hashes():
    fx = _load("settlement_chain_legacy_main.json")
    r = verify_public_chain_report(fx["export"], expected_coord_address=fx["coord_address"])
    _check_against_fixture(fx, r)
    assert r["unbound_mints"] == [0, 1, 2, 3] and r["n_bound_mints"] == 0
    # the 5-tuple API is unchanged for every existing caller
    ok, reason, bal, minted, height = verify_public_chain(fx["export"], expected_coord_address=fx["coord_address"])
    assert (ok, reason, bal, minted, height) == (True, "ok", r["balances"], r["minted"], r["height"])


def test_this_code_with_binding_off_emits_blocks_byte_identical_to_public_main():
    fx = _load("settlement_chain_legacy_main.json")
    cs = _rebuild(bind=False)
    assert json.loads(json.dumps(cs.public_state())) == fx["export"]       # headers, splits AND signatures
    assert cs.unbound_mints() == [0, 1, 2, 3]
    for rec in cs.blocks:
        assert "split_hash" not in rec["header"]


def test_the_header_field_is_absent_and_unhashed_unless_set():
    hdr = BlockHeader.from_dict(_load("settlement_chain_legacy_main.json")["export"]["blocks"][0]["header"])
    assert hdr.split_hash == "" and "split_hash" not in hdr.to_dict()
    h0 = hdr.hash()
    hdr.split_hash = "ab" * 32
    assert hdr.to_dict()["split_hash"] == "ab" * 32 and hdr.hash() != h0
    # a falsy value parses as "not bound" and leaves the hash unchanged
    d = dict(hdr.to_dict(), split_hash="")
    assert BlockHeader.from_dict(d).hash() == h0


# (b) bound chains made by the private coordinator verify here ------------------------------------------------

def test_b_a_bound_chain_made_by_the_private_coordinator_verifies_to_identical_balances():
    fx = _load("settlement_chain_bound_private.json")
    r = verify_public_chain_report(fx["export"], expected_coord_address=fx["coord_address"])
    _check_against_fixture(fx, r)
    assert r["unbound_mints"] == [] and r["n_bound_mints"] == 4
    # strict mode accepts an all-bound chain and rejects the legacy one
    assert verify_public_chain_report(fx["export"], expected_coord_address=fx["coord_address"],
                                      require_split_binding=True)["ok"]
    lg = _load("settlement_chain_legacy_main.json")
    r2 = verify_public_chain_report(lg["export"], expected_coord_address=lg["coord_address"],
                                    require_split_binding=True)
    assert not r2["ok"] and "block 0: mint split is not bound into the signed header" in r2["reason"]


def test_public_emission_with_binding_on_matches_the_private_coordinator_byte_for_byte():
    fx = _load("settlement_chain_bound_private.json")
    cs = _rebuild(bind=True)
    assert json.loads(json.dumps(cs.public_state())) == fx["export"]
    assert cs.unbound_mints() == []
    for rec in cs.blocks:
        assert rec["header"]["split_hash"] == mint_split_hash(rec["contributors"])


def test_from_state_and_verify_replay_a_bound_chain():
    fx = _load("settlement_chain_bound_private.json")
    cs = _rebuild(bind=True)
    restored = ChainSettlement.from_state(json.loads(json.dumps(cs.to_state())))
    assert restored.verify() == (True, "ok")
    for addr, bal in fx["expected"]["balances"].items():
        assert restored.balance(addr) == bal
    # a verify-only (scrubbed) restore of the same chain replays too -- the wallet --balance path
    scrubbed = json.loads(json.dumps(cs.to_state(include_key=False)))
    assert ChainSettlement.from_state(scrubbed).minted() == fx["expected"]["minted"]


def test_a_reordered_bound_split_verifies_to_bit_identical_state():
    fx = _load("settlement_chain_bound_private.json")
    ex = copy.deepcopy(fx["export"])
    rec = ex["blocks"][2]
    rec["contributors"] = dict(reversed(list(rec["contributors"].items())))
    r = verify_public_chain_report(ex, expected_coord_address=fx["coord_address"])
    _check_against_fixture(dict(fx, export=ex), r)


# (c) tampering with a bound block's split fails -----------------------------------------------------------------

@pytest.mark.parametrize("name", ["repoint", "reweight", "add", "drop", "retype"])
def test_c_every_split_tamper_of_a_bound_block_is_rejected(name):
    fx = _load("settlement_chain_bound_private.json")
    ex = copy.deepcopy(fx["export"])
    rec = ex["blocks"][1]
    split = rec["contributors"]
    first = sorted(split)[0]
    attacker = _acct("mallory").address
    if name == "repoint":
        split[attacker] = split.pop(first)
    elif name == "reweight":
        split[first] = split[first] * 2
    elif name == "add":
        split[attacker] = 0.5
    elif name == "drop":
        split.pop(first)
    elif name == "retype":
        split[first] = str(split[first])
    r = verify_public_chain_report(ex, expected_coord_address=fx["coord_address"])
    assert not r["ok"] and r["reason"].startswith("block 1: does not apply") and SPLIT_REASON in r["reason"]
    assert r["balances"] == {} and r["minted"] == 0.0
    # from_state refuses the same tampered chain
    st = {"coord_address": fx["coord_address"], "genesis_checkpoint": ex["genesis_checkpoint"], "blocks": ex["blocks"]}
    with pytest.raises(ValueError, match="failed integrity check"):
        ChainSettlement.from_state(st)


def test_the_commitment_cannot_be_stripped_or_swapped_without_breaking_the_signature():
    fx = _load("settlement_chain_bound_private.json")
    for edit in ("strip", "swap"):
        ex = copy.deepcopy(fx["export"])
        rec = ex["blocks"][0]
        if edit == "strip":
            del rec["header"]["split_hash"]
        else:
            rec["contributors"] = {_acct("mallory").address: 1.0}
            rec["header"]["split_hash"] = mint_split_hash(rec["contributors"])
        r = verify_public_chain_report(ex, expected_coord_address=fx["coord_address"])
        assert not r["ok"] and "not the coordinator" in r["reason"]


def test_an_unbound_legacy_block_is_still_re_pointable_and_is_reported_as_such():
    """The documented limit of a LEGACY block: its split is not signed. It verifies after a re-point, but the
    report names it, so a wallet can tell the user that block's payout is not covered by the signature."""
    fx = _load("settlement_chain_legacy_main.json")
    ex = copy.deepcopy(fx["export"])
    split = ex["blocks"][0]["contributors"]
    split[_acct("mallory").address] = split.pop(next(iter(split)))
    r = verify_public_chain_report(ex, expected_coord_address=fx["coord_address"])
    assert r["ok"] and 0 in r["unbound_mints"]


# (d) why a release is needed: a pre-binding reader rejects a bound chain (blocker B2) ---------------------------

def test_d_a_pre_binding_reader_rejects_every_bound_chain():
    """Model of public main 3.9.3's reader: its BlockHeader.from_dict ignored "split_hash" and its canonical()
    never emitted it, which is exactly the header with the key removed. The recomputed hash then differs from the
    signed one, the signature recovers an unrelated address, and the chain fails at block 0. Measured against the
    real main code too (`git show main:...` in a scratch dir): verify_public_chain on the bound fixture returned
    (False, "block 0: signed by 0xF459...98e4, not the coordinator")."""
    fx = _load("settlement_chain_bound_private.json")
    ex = copy.deepcopy(fx["export"])
    for rec in ex["blocks"]:
        rec["header"].pop("split_hash")                    # what the old from_dict silently did
    ok, reason, bal, minted, height = verify_public_chain(ex, expected_coord_address=fx["coord_address"])
    assert not ok and reason.startswith("block 0: signed by ") and reason.endswith("not the coordinator")
    assert (bal, minted, height) == ({}, 0.0, 0)
