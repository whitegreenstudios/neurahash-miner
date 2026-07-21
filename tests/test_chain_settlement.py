"""
#5 — settle pool rewards on the neura_l1 chain ledger. A reward is a signed, replay-protected chain
block; balances are the neura_l1 State (reconstructed by replaying the signed blocks), not a coordinator
dict; the chain's MAX_SUPPLY cap is enforced by chain code; a tampered reward/amount fails verify().
"""
import pytest

from neura_l1.signing import gen_account
from neurahash.chain_settlement import ChainSettlement, CHAIN_MAX_SUPPLY, verify_public_chain


def _cs():
    return ChainSettlement(gen_account())


def test_settle_credits_pro_rata_on_chain():
    cs = _cs()
    paid = cs.settle(height=1, reward_pot=100.0, ex_by={"a": 3.0, "b": 1.0}, timestamp=10)
    assert paid["a"] == pytest.approx(75.0) and paid["b"] == pytest.approx(25.0)   # 3:1 split
    assert cs.balance("a") == pytest.approx(75.0) and cs.balance("b") == pytest.approx(25.0)
    assert cs.minted() == pytest.approx(100.0)
    assert cs.height() == 1


def test_balances_accumulate_across_heights():
    cs = _cs()
    cs.settle(1, 100.0, {"a": 1.0}, 10)
    cs.settle(2, 40.0, {"a": 1.0, "b": 1.0}, 20)
    assert cs.balance("a") == pytest.approx(120.0)     # 100 + 20
    assert cs.balance("b") == pytest.approx(20.0)
    assert cs.minted() == pytest.approx(140.0)
    ok, why = cs.verify()
    assert ok, why


def test_zero_work_height_mints_nothing():
    cs = _cs()
    paid = cs.settle(1, 50.0, {}, 10)                   # no contributors
    assert paid == {} and cs.minted() == 0.0


def test_hard_cap_is_chain_enforced():
    cs = _cs()
    cs.settle(1, CHAIN_MAX_SUPPLY + 1e9, {"a": 1.0}, 10)   # ask for more than the whole cap
    assert cs.balance("a") == pytest.approx(CHAIN_MAX_SUPPLY)
    assert cs.minted() == pytest.approx(CHAIN_MAX_SUPPLY)
    extra = cs.settle(2, 1000.0, {"b": 1.0}, 20)            # cap already reached -> nothing more mints
    assert extra.get("b", 0.0) == 0.0 and cs.balance("b") == 0.0


def test_each_block_is_signed_by_the_coordinator():
    from neura_l1.signing import recover_bytes
    from neura_l1.block_state import BlockHeader
    cs = _cs()
    cs.settle(1, 10.0, {"a": 1.0}, 10)
    rec = cs.blocks[0]
    hdr = BlockHeader.from_dict(rec["header"])
    assert recover_bytes(hdr.hash().encode(), rec["sig"]).lower() == cs.coord.address.lower()


def test_balances_reconstruct_from_the_signed_chain():
    cs = _cs()
    cs.settle(1, 100.0, {"a": 2.0, "b": 2.0}, 10)
    cs.settle(2, 60.0, {"a": 1.0}, 20)
    # replay from genesis equals the live state -> balances come from the chain, not a stored dict
    ok, why = cs.verify()
    assert ok, why


def test_tampered_reward_breaks_signature():
    cs = _cs()
    cs.settle(1, 100.0, {"a": 1.0}, 10)
    cs.blocks[0]["header"]["reward"] = 999999.0          # forge a bigger reward without re-signing
    ok, why = cs.verify()
    assert ok is False and ("signature" in why or "not the coordinator" in why)


def test_tampered_contributor_split_breaks_balance_reconstruction():
    cs = _cs()
    cs.settle(1, 100.0, {"a": 1.0, "b": 1.0}, 10)
    cs.blocks[0]["contributors"]["a"] = 99.0             # skew the split (not covered by the header sig)
    ok, why = cs.verify()
    assert ok is False and "balance mismatch" in why


def test_broken_chain_link_detected():
    cs = _cs()
    cs.settle(1, 10.0, {"a": 1.0}, 10)
    cs.settle(2, 10.0, {"a": 1.0}, 20)
    cs.blocks[1]["header"]["prev_hash"] = "0" * 64       # snap the prev_hash link
    ok, why = cs.verify()
    assert ok is False and ("broken chain" in why or "signature" in why or "not the coordinator" in why)


def test_persistence_roundtrip_reconstructs_and_verifies():
    cs = _cs()
    cs.settle(1, 100.0, {"a": 3.0, "b": 1.0}, 10)
    cs.settle(2, 50.0, {"a": 1.0}, 20)
    restored = ChainSettlement.from_state(cs.to_state())
    assert restored.balance("a") == pytest.approx(cs.balance("a"))
    assert restored.balance("b") == pytest.approx(cs.balance("b"))
    assert restored.minted() == pytest.approx(cs.minted())
    ok, why = restored.verify()
    assert ok, why


def test_tampered_checkpoint_fails_on_restore():
    cs = _cs()
    cs.settle(1, 100.0, {"a": 1.0}, 10)
    blob = cs.to_state()
    blob["blocks"][0]["header"]["reward"] = 5.0          # tamper the persisted chain
    with pytest.raises(ValueError, match="integrity check"):
        ChainSettlement.from_state(blob)


# ------------------- DiLoCo reward idempotency (restart-replay guard) -------------------
def test_paid_cid_index_persists_across_restart():
    # the durable per-delta_cid guard that stops a coordinator restart from re-minting an already-paid
    # DiLoCo contribution: it must round-trip through to_state/from_state alongside the signed blocks.
    cs = _cs()
    cs.settle(1, 5.0, {"a": 1.0}, 10)
    cs.mark_paid("bafyPAID")
    assert cs.already_paid("bafyPAID") is True
    restored = ChainSettlement.from_state(cs.to_state())
    assert restored.already_paid("bafyPAID") is True     # survives the restart
    assert restored.already_paid("bafyOTHER") is False    # a genuinely new cid is still payable


def test_paid_cid_index_omitted_when_empty_backcompat():
    # DEFAULT-OFF: with nothing paid the index is empty, so to_state emits NO 'paid_cids' key (the blob is
    # byte-identical to a pre-guard checkpoint), and a pre-guard blob (absent field) restores to empty.
    cs = _cs()
    cs.settle(1, 5.0, {"a": 1.0}, 10)
    blob = cs.to_state()
    assert "paid_cids" not in blob                        # empty index -> key omitted
    restored = ChainSettlement.from_state(blob)           # pre-guard blob (no field) restores cleanly
    assert restored.paid_cids == set()


def test_already_paid_and_mark_paid_ignore_falsy_cid():
    cs = _cs()
    assert cs.already_paid(None) is False and cs.already_paid("") is False
    cs.mark_paid(None)
    cs.mark_paid("")
    assert cs.paid_cids == set()                          # falsy cids are not recordable dedup keys


# ------------------- public_state() / verify_public_chain() (wallet chain-sync, #Phase1) -------------------
def test_public_state_omits_coord_key_and_paid_cids():
    # mirrors test_paid_cid_index_omitted_when_empty_backcompat: the public export must never
    # carry the coordinator's private key, and never carry the internal DiLoCo paid-cid index.
    cs = _cs()
    cs.settle(1, 100.0, {"a": 1.0}, 10)
    cs.mark_paid("bafySomeDelta")
    pub = cs.public_state()
    assert "coord_key" not in pub
    assert "paid_cids" not in pub
    assert pub["coord_address"] == cs.coord.address
    assert pub["genesis_checkpoint"] == cs.genesis_checkpoint
    assert pub["blocks"] == [dict(b) for b in cs.blocks]


def test_verify_public_chain_accepts_a_genuine_chain_pinned():
    cs = _cs()
    cs.settle(1, 100.0, {"a": 3.0, "b": 1.0}, 10)
    cs.settle(2, 40.0, {"a": 1.0, "b": 1.0}, 20)
    export = cs.public_state()
    ok, reason, balances, minted, height = verify_public_chain(
        export, expected_coord_address=cs.coord.address)
    assert ok, reason
    assert "unpinned" not in reason.lower()
    assert balances["a"] == pytest.approx(cs.balance("a"))
    assert balances["b"] == pytest.approx(cs.balance("b"))
    assert minted == pytest.approx(cs.minted())
    assert height == cs.height() == 2


def test_verify_public_chain_unpinned_falls_back_to_export_address_and_flags_it():
    cs = _cs()
    cs.settle(1, 50.0, {"a": 1.0}, 10)
    export = cs.public_state()
    ok, reason, balances, minted, height = verify_public_chain(export)   # no expected_coord_address
    assert ok, reason
    assert "unpinned" in reason.lower()
    assert balances["a"] == pytest.approx(cs.balance("a"))


def test_verify_public_chain_rejects_a_tampered_reward():
    cs = _cs()
    cs.settle(1, 100.0, {"a": 1.0}, 10)
    export = cs.public_state()
    export["blocks"][0]["header"]["reward"] = 999999.0    # forge a bigger reward without re-signing
    ok, reason, balances, minted, height = verify_public_chain(
        export, expected_coord_address=cs.coord.address)
    assert ok is False and ("signature" in reason or "not the coordinator" in reason)
    assert balances == {} and minted == 0.0 and height == 0


def test_verify_public_chain_rejects_wrong_expected_coord_address_forged_chain():
    # The forged-chain / anti-spoofing case (scoping doc Risk #2): an attacker signs their OWN
    # fully self-consistent chain and publishes it with THEIR OWN address as `coord_address`.
    # A caller that pins the real coordinator's address must reject it even though the export is
    # internally consistent end-to-end -- `export["coord_address"]` must never be trusted for the
    # trust decision when a pinned address is supplied.
    real_coord = gen_account()
    attacker = gen_account()
    forged = ChainSettlement(attacker)
    forged.settle(1, 100.0, {"attacker_friend": 1.0}, 10)
    export = forged.public_state()
    assert export["coord_address"] == attacker.address     # self-consistent, self-signed by attacker

    ok, reason, balances, minted, height = verify_public_chain(
        export, expected_coord_address=real_coord.address)
    assert ok is False
    assert "not the coordinator" in reason
    assert balances == {} and minted == 0.0 and height == 0

    # the SAME export verifies fine against the attacker's own address (proves it's not just
    # malformed) -- the rejection above is specifically about the pinned identity, not the data.
    ok2, reason2, _, _, _ = verify_public_chain(export, expected_coord_address=attacker.address)
    assert ok2, reason2


def test_verify_public_chain_prefix_export_verifies_ok_with_lower_balance_known_limitation():
    # DOCUMENTED, NOT FIXED (scoping doc Risk #1): a strict prefix of a genuine chain (the last
    # block withheld) still replays and verifies as internally consistent -- chain replay alone
    # cannot detect withholding/rollback. This test records that as a known limitation so nobody
    # "fixes" it later by weakening this test; withholding protection is the wallet-side height
    # ratchet (wallet_app/backend/chain_sync.py), not this function.
    cs = _cs()
    cs.settle(1, 100.0, {"a": 1.0}, 10)
    cs.settle(2, 100.0, {"a": 1.0}, 20)
    full_export = cs.public_state()
    prefix_export = dict(full_export)
    prefix_export["blocks"] = full_export["blocks"][:1]    # drop the last (real) block

    ok, reason, balances, minted, height = verify_public_chain(
        prefix_export, expected_coord_address=cs.coord.address)
    assert ok is True, reason                               # verifies cleanly -- the limitation
    assert height == 1                                       # lower than the true height (2)
    assert balances["a"] == pytest.approx(100.0)             # lower than the true balance (200.0)
    assert balances["a"] < cs.balance("a")


def test_verify_public_chain_rejects_malformed_export():
    ok, reason, balances, minted, height = verify_public_chain({"not_blocks": []})
    assert ok is False and "blocks" in reason
    ok2, reason2, _, _, _ = verify_public_chain({"blocks": []})   # no coord_address anywhere
    assert ok2 is False and "coord_address" in reason2
