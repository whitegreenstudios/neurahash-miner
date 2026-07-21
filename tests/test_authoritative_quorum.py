"""B8-1 AUTHORITATIVE QUORUM -- the trustless-coordinator proof at the replay/verification layer.

Demonstrates the property the owner asked to prove: a block is accepted by an INDEPENDENT read-only
replayer (no coordinator key) ONLY if an M-of-N staked-validator quorum authorized its exact mint.
A malicious coordinator -- one that owns the transport and its own signing key -- still CANNOT:
  * omit the quorum (enforce rejects a block with no proof),
  * forge/short the quorum (< M distinct pinned-roster signers -> rejected),
  * self-select a shill roster (signers outside the PINNED validator set are not counted),
  * inflate the mint (the validators signed a specific amount; a bumped header.reward no longer
    matches the quorum -> rejected) even though the coordinator re-signs the tampered header.

Default replay (require_quorum=False) is byte-identical to the single-coordinator chain: these same
malicious blocks all pass, proving the enforce path -- not a behavior change to today -- is what
rejects them. Run: C:/Python313/python.exe -m pytest tests/test_authoritative_quorum.py -q
"""
import copy
from neura_l1.signing import gen_account, sign_bytes
from neurahash.chain_settlement import ChainSettlement, verify_public_chain, BlockHeader
from neurahash import diloco_settlement as ds

M = 3            # required quorum
N = 5            # validator set size


def _mk():
    coord = gen_account()
    validators = [gen_account() for _ in range(N)]
    roster = [v.address for v in validators]
    return coord, validators, roster


def _quorum(validators, roster, *, recipient, amount, height, delta_cid, n_signers, m=M):
    atts = [ds.sign_settlement(validators[i], recipient=recipient, amount=amount,
                               height=height, delta_cid=delta_cid) for i in range(n_signers)]
    return ds.settlement_block_proof(recipient=recipient, amount=amount, height=height,
                                     delta_cid=delta_cid, roster=roster, attestations=atts, m=m)


def _settle_one(coord, validators, roster, *, height, reward, n_signers=M, delta_cid="cidA"):
    """One settlement block whose sole contributor is a fresh miner, carrying an n-signer quorum."""
    cs = ChainSettlement(coordinator_account=coord, genesis_checkpoint="g")
    miner = gen_account().address
    q = _quorum(validators, roster, recipient=miner, amount=reward, height=height,
                delta_cid=delta_cid, n_signers=n_signers)
    cs.settle(height=height, reward_pot=reward, ex_by={miner: 1.0}, timestamp=1000 + height, quorum=q)
    return cs, miner


def _export(cs, coord):
    return {"blocks": copy.deepcopy(cs.blocks), "genesis_checkpoint": cs.genesis_checkpoint,
            "coord_address": coord.address}


# --------------------------------------------------------------------- 1. honest chain PASSES enforce
def test_honest_quorum_chain_passes_enforce():
    coord, validators, roster = _mk()
    cs, miner = _settle_one(coord, validators, roster, height=1, reward=10.0, n_signers=M)
    exp = _export(cs, coord)
    ok, reason, bal, minted, h = verify_public_chain(
        exp, expected_coord_address=coord.address, require_quorum=True, quorum_roster=roster, quorum_m=M)
    assert ok, reason
    assert "quorum-enforced" in reason
    assert round(bal.get(miner, 0.0), 6) > 0.0


# --------------------------------------------------------------------- 2. no quorum -> REJECTED
def test_missing_quorum_rejected_under_enforce_but_accepted_off():
    coord, validators, roster = _mk()
    cs, miner = _settle_one(coord, validators, roster, height=1, reward=10.0, n_signers=M)
    exp = _export(cs, coord)
    del exp["blocks"][0]["quorum"]                    # malicious: coordinator strips the quorum
    ok_e, reason_e, *_ = verify_public_chain(
        exp, expected_coord_address=coord.address, require_quorum=True, quorum_roster=roster, quorum_m=M)
    assert not ok_e and "no quorum proof" in reason_e
    # default replay is UNCHANGED: the same block still verifies on the coordinator signature alone.
    ok_off, _r, *_ = verify_public_chain(exp, expected_coord_address=coord.address)
    assert ok_off


# --------------------------------------------------------------------- 3. sub-M quorum -> REJECTED
def test_insufficient_signers_rejected():
    coord, validators, roster = _mk()
    cs, miner = _settle_one(coord, validators, roster, height=1, reward=10.0, n_signers=M - 1)
    exp = _export(cs, coord)
    ok, reason, *_ = verify_public_chain(
        exp, expected_coord_address=coord.address, require_quorum=True, quorum_roster=roster, quorum_m=M)
    assert not ok and "quorum not met" in reason


# --------------------------------------------------------------------- 4. shill roster -> REJECTED
def test_coordinator_self_selected_shill_roster_rejected():
    coord, validators, roster = _mk()
    # coordinator builds a "valid-looking" M-of-N quorum entirely from its OWN throwaway accounts
    shills = [gen_account() for _ in range(M)]
    shill_roster = [s.address for s in shills]
    cs = ChainSettlement(coordinator_account=coord, genesis_checkpoint="g")
    miner = gen_account().address
    shill_q = _quorum(shills, shill_roster, recipient=miner, amount=10.0, height=1,
                      delta_cid="cidA", n_signers=M)
    cs.settle(height=1, reward_pot=10.0, ex_by={miner: 1.0}, timestamp=1001, quorum=shill_q)
    exp = _export(cs, coord)
    # PINNED to the REAL validator roster: the shills are not members -> not counted -> rejected.
    ok, reason, *_ = verify_public_chain(
        exp, expected_coord_address=coord.address, require_quorum=True, quorum_roster=roster, quorum_m=M)
    assert not ok and "quorum not met" in reason


# --------------------------------------------------------------------- 5. inflated mint -> REJECTED
def test_inflated_reward_rejected_even_with_recoordinator_signature():
    coord, validators, roster = _mk()
    cs, miner = _settle_one(coord, validators, roster, height=1, reward=10.0, n_signers=M)
    exp = _export(cs, coord)
    # malicious: bump the header reward 10 -> 500, then RE-SIGN the header with the coordinator's own
    # key (so the coordinator signature still verifies) -- but the quorum authorized amount=10.
    rec = exp["blocks"][0]
    rec["header"]["reward"] = 500.0
    hdr = BlockHeader.from_dict(rec["header"])
    rec["sig"] = sign_bytes(coord, hdr.hash().encode())
    # coordinator signature ALONE still passes (default replay) -- the tamper is invisible there...
    ok_off, _r, _b, minted_off, _h = verify_public_chain(exp, expected_coord_address=coord.address)
    assert ok_off and minted_off > 100.0            # single-coordinator trust would mint the inflated 500
    # ...but the QUORUM (signed over amount=10) no longer binds to header.reward=500 -> rejected.
    ok_e, reason_e, *_ = verify_public_chain(
        exp, expected_coord_address=coord.address, require_quorum=True, quorum_roster=roster, quorum_m=M)
    assert not ok_e and ("amount" in reason_e or "quorum not met" in reason_e)


# --------------------------------------------------------------------- 6. default OFF is byte-identical
def test_default_off_ignores_quorum_field():
    coord, validators, roster = _mk()
    cs, miner = _settle_one(coord, validators, roster, height=1, reward=10.0, n_signers=M)
    blocks_with = copy.deepcopy(cs.blocks)
    # a chain built WITHOUT any quorum field must replay identically to one WITH it, when enforce is off
    cs2 = ChainSettlement(coordinator_account=coord, genesis_checkpoint="g")
    miner2 = list(cs.blocks[0]["contributors"].keys())[0]
    cs2.settle(height=1, reward_pot=10.0, ex_by={miner2: 1.0}, timestamp=1001)  # no quorum=
    ok1, _r1, bal1, m1, h1 = verify_public_chain({"blocks": blocks_with, "genesis_checkpoint": "g",
                                                  "coord_address": coord.address},
                                                 expected_coord_address=coord.address)
    ok2, _r2, bal2, m2, h2 = verify_public_chain({"blocks": cs2.blocks, "genesis_checkpoint": "g",
                                                  "coord_address": coord.address},
                                                 expected_coord_address=coord.address)
    assert ok1 and ok2 and h1 == h2 and abs(m1 - m2) < 1e-9


# ------------------------------------------------- 7. enforce WITHOUT a pinned roster is REFUSED
def test_enforce_without_pinned_roster_refused():
    coord, validators, roster = _mk()
    cs, miner = _settle_one(coord, validators, roster, height=1, reward=10.0, n_signers=M)
    exp = _export(cs, coord)
    ok, reason, *_ = verify_public_chain(
        exp, expected_coord_address=coord.address, require_quorum=True, quorum_roster=None, quorum_m=M)
    assert not ok and "pinned validator roster" in reason


# ------------------------------------------------- 8. multi-contributor quorum FAILS CLOSED (rejects)
def test_multicontributor_recipient_none_fails_closed():
    coord, validators, roster = _mk()
    cs = ChainSettlement(coordinator_account=coord, genesis_checkpoint="g")
    m1, m2 = gen_account().address, gen_account().address
    # a quorum built for a SINGLE recipient (m1) attached to a 2-contributor block: enforce sets
    # recipient=None, so the rebuilt payload names "None" -> the m1-recipient sigs don't recover ->
    # 0 pinned signers -> rejected. The point: the recipient=None branch does NOT weakly accept.
    q = _quorum(validators, roster, recipient=m1, amount=10.0, height=1, delta_cid="cidA", n_signers=M)
    cs.settle(height=1, reward_pot=10.0, ex_by={m1: 1.0, m2: 1.0}, timestamp=1001, quorum=q)
    exp = _export(cs, coord)
    ok, reason, *_ = verify_public_chain(
        exp, expected_coord_address=coord.address, require_quorum=True, quorum_roster=roster, quorum_m=M)
    assert not ok and "quorum not met" in reason


# --------------------------- 9. STAKE-WEIGHTED authoritative path (B8-4 load-bearing, Sybil defense)
def _chain_signed_by(coord, signers, recipient, stake_roster, *, amount=10.0, height=1, cid="cidA"):
    cs = ChainSettlement(coordinator_account=coord, genesis_checkpoint="g")
    atts = [ds.sign_settlement(s, recipient=recipient, amount=amount, height=height, delta_cid=cid) for s in signers]
    q = ds.settlement_block_proof(recipient=recipient, amount=amount, height=height, delta_cid=cid,
                                  roster=list(stake_roster), attestations=atts, m=None)
    cs.settle(height=height, reward_pot=amount, ex_by={recipient: 1.0}, timestamp=1000 + height, quorum=q)
    return _export(cs, coord)


def test_stake_weighted_authoritative_path_defeats_headcount_sybil():
    coord = gen_account()
    honest = [gen_account() for _ in range(3)]     # 3 x 1000 stake = stake-majority, headcount-MINORITY
    shells = [gen_account() for _ in range(5)]     # 5 x 10 stake   = stake-minority, headcount-MAJORITY
    stake_roster = {v.address: 1000.0 for v in honest}
    stake_roster.update({s.address: 10.0 for s in shells})
    addr_set = set(stake_roster)                    # same members, but NO stakes -> count-weighted

    exp_shells = _chain_signed_by(coord, shells, gen_account().address, stake_roster)
    exp_honest = _chain_signed_by(coord, honest, gen_account().address, stake_roster)

    # DICT roster => STAKE-weighted: the 5 shells are a stake-minority -> REJECTED...
    ok_s, r_s, *_ = verify_public_chain(exp_shells, expected_coord_address=coord.address,
                                        require_quorum=True, quorum_roster=stake_roster)
    assert not ok_s, r_s
    # ...while the 3 honest are a stake-MAJORITY (though a headcount minority) -> ACCEPTED.
    ok_h, r_h, *_ = verify_public_chain(exp_honest, expected_coord_address=coord.address,
                                        require_quorum=True, quorum_roster=stake_roster)
    assert ok_h, r_h
    # CONTROL: the SAME 5 shells against a bare address SET are a headcount majority -> count-accepts
    # (the Sybil hole a non-stake roster leaves). Proves the dict path is what enforces stake-weight.
    ok_set, *_ = verify_public_chain(exp_shells, expected_coord_address=coord.address,
                                     require_quorum=True, quorum_roster=addr_set)
    assert ok_set


# --------------------------- 10. sha256(quorum) folded into the SIGNED header (no strip/swap after signing)
def test_header_quorum_hash_binds_the_quorum_and_rejects_a_swap():
    coord, validators, roster = _mk()
    cs, miner = _settle_one(coord, validators, roster, height=1, reward=10.0, n_signers=M)
    exp = _export(cs, coord)
    # settle now folds sha256(quorum) into the SIGNED header:
    assert exp["blocks"][0]["header"].get("quorum_hash"), "settle must bind the quorum into the header"
    # attacker swaps in a DIFFERENT internally-valid quorum (different mint), WITHOUT re-signing the header:
    other = _quorum(validators, roster, recipient=gen_account().address, amount=10.0, height=1,
                    delta_cid="OTHER", n_signers=M)
    exp["blocks"][0]["quorum"] = other
    ok, reason, *_ = verify_public_chain(exp, expected_coord_address=coord.address,
                                         require_quorum=True, quorum_roster=roster, quorum_m=M)
    assert not ok and "header quorum_hash" in reason


def test_no_quorum_block_header_omits_quorum_hash_byte_identical():
    coord = gen_account()
    cs = ChainSettlement(coordinator_account=coord, genesis_checkpoint="g")
    m = gen_account().address
    cs.settle(height=1, reward_pot=10.0, ex_by={m: 1.0}, timestamp=1001)   # NO quorum
    assert "quorum_hash" not in cs.blocks[0]["header"], "a no-quorum header must omit quorum_hash (byte-identical)"
    ok, *_ = verify_public_chain({"blocks": cs.blocks, "genesis_checkpoint": "g",
                                  "coord_address": coord.address}, expected_coord_address=coord.address)
    assert ok
