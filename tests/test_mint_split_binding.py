"""Mint-split binding (2026-09-24): a settlement mint's per-address split is covered by the coordinator signature
when the emission flag is on, and every verifier checks it.

THE HOLE (reproduced 2026-09-24; memory settlement-mint-split-unsigned-2026-09-24): ChainSettlement.settle() signed
a header that covers sum(weights) and the reward, but the `contributors` split lived only in the block RECORD and
every replayer applied it unchecked -- so whoever served a public export could re-point a pool mint and
verify_public_chain (the wallet's chain_sync reader) still returned ok.

THE FIX: BlockHeader.split_hash = mint_split_hash(contributors), folded into the SIGNED header by settle() when
NEURAHASH_BIND_MINT_SPLIT / bind_mint_split is on (DEFAULT OFF), and bound to the applied split inside
State.apply_block, so EVERY replayer checks it: verify / from_state / verify_public_chain, the round authority's
Candidate.accept (and, in the private node, chain_transfers' replay). A block without split_hash is LEGACY-UNBOUND: it replays exactly as
before and is REPORTED; strict mode (require_split_binding) rejects it.

PORTED 2026-11-19 to the public client from the private branch stage0-mint-split (reader side of
Stage 0 / blocker B2). Left out, because the public client does not ship that code: the chain_transfers
cases, the pre-change byte-identity vector over delta_cid settles (tests/test_bound_readers.py pins byte
identity against a fixture made by public main instead), and the quorum-trust-root constructor arguments.

Every key here is a THROWAWAY (random, or sha256 of a public label). Real secp256k1 signing throughout.
Run: C:/Python313/python.exe -m pytest tests/test_mint_split_binding.py -q
"""
import collections
import copy
import hashlib
import json
import math

import pytest

from neura_l1.block_state import (Block, BlockHeader, State, canonical_mint_split, mint_split_hash,
                                  MINT_SPLIT_HASH_TAG)
from neura_l1.signing import account_from_key, gen_account
from neurahash import diloco_round_authority as ra
from neurahash import diloco_settlement as ds
from neurahash.chain_settlement import (BIND_MINT_SPLIT_ENV, ChainSettlement, _replay_blocks,
                                        verify_public_chain, verify_public_chain_report)

T0 = 1_760_000_000
SPLIT_REASON = "contributors do not match header.split_hash"
STRICT_REASON = "mint split is not bound into the signed header"
LEGACY_HEADER_KEYS = {"height", "prev_hash", "beacon", "parent_checkpoint", "new_checkpoint", "work_score",
                      "proposer", "reward", "tx_root", "timestamp", "protocol_version"}


@pytest.fixture(autouse=True)
def _no_env_flags(monkeypatch):
    """Each test states its posture explicitly; a flag in the developer's shell must not leak in."""
    monkeypatch.delenv(BIND_MINT_SPLIT_ENV, raising=False)


def _acct(label, prefix="mintsplit-test/"):
    return account_from_key(hashlib.sha256((prefix + label).encode()).hexdigest())


def _export(cs):
    return json.loads(json.dumps(cs.public_state()))        # a real JSON round trip, as a wallet receives it


def _verify(ex, coord, **kw):
    return verify_public_chain(ex, expected_coord_address=coord.address, **kw)


def _mut(ex, fn):
    e = copy.deepcopy(ex)
    fn(e)
    return e


def _bound_chain(coord, a, b, c, **kw):
    """Every block shape settle() produces, with binding ON: single recipient, multi recipient, an empty split
    (reward > 0 that mints nothing), another multi split."""
    cs = ChainSettlement(coord, genesis_checkpoint="g", bind_mint_split=True, **kw)
    cs.settle(1, 100.0, {a.address: 1.0}, T0)
    cs.settle(1, 60.0, {a.address: 2.0, b.address: 1.0}, T0 + 1)
    cs.settle(2, 30.0, {}, T0 + 2)
    cs.settle(2, 10.0, {b.address: 0.25, c.address: 0.75}, T0 + 3)
    return cs


# ------------------------------------------------------------------------------------ 1. the canonical encoding
def test_canonical_split_is_sorted_exact_case_sensitive_and_matches_a_hand_written_vector():
    up, low = "0xBBBB", "0xaaaa"                             # "B" (0x42) sorts before "a" (0x61)
    s = canonical_mint_split({low: 1.0, up: 0.1})
    assert MINT_SPLIT_HASH_TAG == "nrh-mint-split/v1"
    assert s == '["nrh-mint-split/v1",[["0xBBBB","0x1.999999999999ap-4"],["0xaaaa","0x1.0000000000000p+0"]]]'
    assert mint_split_hash({low: 1.0, up: 0.1}) == hashlib.sha256(s.encode("ascii")).hexdigest()
    assert mint_split_hash({}) == hashlib.sha256(b'["nrh-mint-split/v1",[]]').hexdigest()
    base = mint_split_hash({up: 0.1, low: 1.0})
    assert mint_split_hash({low: 1.0, up: 0.1}) == base                                  # stored order: irrelevant
    assert mint_split_hash({up.lower(): 0.1, low: 1.0}) != base                          # address case: bound
    assert mint_split_hash({up: math.nextafter(0.1, 1.0), low: 1.0}) != base            # one ULP: bound
    assert mint_split_hash({up: 0.1, low: 1.0, "0xcccc": 0.0}) != base                  # a zero entry: bound
    assert mint_split_hash({up: 0.1}) != base                                            # a dropped entry: bound


@pytest.mark.parametrize("bad", [{"0xA": 1}, {"0xA": True}, {"0xA": "1.0"}, {"0xA": float("nan")},
                                 {"0xA": float("inf")}, {1: 1.0}, [("0xA", 1.0)], None,
                                 {"0xA" + chr(0xD83D) + chr(0xDE00): 1.0},
                                 collections.OrderedDict({"0xA": 1.0})],
                         ids=["int", "bool", "str", "nan", "inf", "int-key", "list", "none",
                              "joinable-surrogates", "dict-subclass"])
def test_a_split_with_no_canonical_form_is_refused(bad):
    with pytest.raises(ValueError):
        mint_split_hash(bad)


# ------------------------------------------------------------------------------------ 2. header byte-identity
def test_the_header_field_is_absent_and_unhashed_unless_set():
    h = BlockHeader(height=1, prev_hash="0" * 64, beacon="b", parent_checkpoint="c", new_checkpoint="c",
                    work_score=1.0, proposer="p", reward=1.0, tx_root="0" * 64, timestamp=1)
    assert set(h.canonical()) == LEGACY_HEADER_KEYS and h.split_hash == ""
    for junk in (None, "", 0, False):        # a falsy value parses as UNBOUND and leaves the signed hash unchanged
        parsed = BlockHeader.from_dict(dict(h.to_dict(), split_hash=junk))
        assert parsed.split_hash == "" and parsed.hash() == h.hash() and "split_hash" not in parsed.to_dict()
    bound = BlockHeader.from_dict(dict(h.to_dict(), split_hash="ab" * 32))
    assert bound.to_dict()["split_hash"] == "ab" * 32 and bound.hash() != h.hash()
    assert BlockHeader.from_dict(bound.to_dict()).hash() == bound.hash()                # round-trips




# ------------------------------------------------------------------------------------ 3. binding ON: the tamper matrix
def test_the_original_probe_fails_closed_with_binding_on_and_passes_with_it_off():
    for bind in (False, True):
        coord, a, attacker = gen_account(), gen_account(), gen_account()
        cs = ChainSettlement(coord, genesis_checkpoint="probe", bind_mint_split=bind)
        cs.settle(1, 100.0, {a.address: 1.0}, 1000)
        ex = _export(cs)
        assert _verify(ex, coord)[:3] == (True, "ok", {a.address: 100.0})                # positive case, both
        ex["blocks"][0]["contributors"] = {attacker.address: 1.0}
        ok, reason, bals, *_ = _verify(ex, coord)
        if bind:
            assert not ok and reason.startswith("block 0: does not apply") and SPLIT_REASON in reason
            assert bals == {}
        else:                                                   # the legacy residual, unchanged -- and REPORTED
            assert ok and bals == {attacker.address: 100.0}
            assert verify_public_chain_report(ex, expected_coord_address=coord.address)["unbound_mints"] == [0]


def _tampers(a, b, c, evil):
    return {
        "re-pointed recipient": (0, lambda k: {evil: 1.0}),
        "swapped weights": (1, lambda k: {a: 1.0, b: 2.0}),
        "added recipient": (1, lambda k: dict(k, **{evil: 1.0})),
        "removed recipient": (1, lambda k: {a: 2.0}),
        "added zero-weight entry": (1, lambda k: dict(k, **{evil: 0.0})),
        "recipient added to a bound empty split": (2, lambda k: {evil: 1.0}),
        "re-cased recipient": (0, lambda k: {a.lower(): 1.0}),
        "weight re-typed to int": (0, lambda k: {a: 1}),
        "weight re-typed to str": (0, lambda k: {a: "1.0"}),
        "weight moved one ULP": (3, lambda k: {b: 0.25, c: math.nextafter(0.75, 1.0)}),
        "split emptied": (3, lambda k: {}),
        "split replaced by a list": (3, lambda k: [[b, 0.25], [c, 0.75]]),
    }


@pytest.mark.parametrize("name", list(_tampers("a", "b", "c", "e")))
def test_every_split_tamper_of_a_bound_block_is_rejected_and_the_genuine_chain_verifies(name):
    coord, a, b, c, evil = gen_account(), gen_account(), gen_account(), gen_account(), gen_account()
    cs = _bound_chain(coord, a, b, c)
    genuine = _export(cs)
    assert all(rec["header"].get("split_hash") for rec in genuine["blocks"])            # every block bound
    ok, reason, bals, minted, height = _verify(genuine, coord)                          # positive case
    assert (ok, reason, height) == (True, "ok", 2) and bals == dict(cs.state.bal) and minted == cs.minted()
    rep = verify_public_chain_report(genuine, expected_coord_address=coord.address)
    assert rep["unbound_mints"] == [] and rep["n_bound_mints"] == 4 and cs.unbound_mints() == []
    idx, fn = _tampers(a.address, b.address, c.address, evil.address)[name]
    bad = _mut(genuine, lambda e: e["blocks"][idx].update(contributors=fn(e["blocks"][idx]["contributors"])))
    ok, reason, bals, minted, height = _verify(bad, coord)
    assert not ok and reason.startswith(f"block {idx}: does not apply") and "split_hash" in reason, reason
    assert (bals, minted, height) == ({}, 0.0, 0)
    for kw in ({}, {"require_split_binding": True}):                                    # strict changes nothing
        assert not verify_public_chain_report(bad, expected_coord_address=coord.address, **kw)["ok"]


def test_the_header_commitment_cannot_be_stripped_or_swapped_without_breaking_the_signature():
    coord, a, b, c, evil = gen_account(), gen_account(), gen_account(), gen_account(), gen_account()
    genuine = _export(_bound_chain(coord, a, b, c))
    stripped = _mut(genuine, lambda e: e["blocks"][0]["header"].pop("split_hash"))
    swapped = _mut(genuine, lambda e: (e["blocks"][0].update(contributors={evil.address: 1.0}),
                                       e["blocks"][0]["header"].update(
                                           split_hash=mint_split_hash({evil.address: 1.0}))))
    nulled = _mut(genuine, lambda e: e["blocks"][0]["header"].update(split_hash=None))
    for name, ex in (("stripped", stripped), ("swapped", swapped), ("nulled", nulled)):
        ok, reason, bals, *_ = _verify(ex, coord)
        assert not ok and reason.startswith("block 0: signed by") and bals == {}, (name, reason)


def test_a_reordered_bound_split_verifies_to_bit_identical_state():
    """The commitment is over the SET of (address, weight) pairs (JSON member order is not portable), so a
    reordered bound split verifies -- and it must then replay to bit-identical state. A PROPERTY pin, not a
    discriminator: on CPython >= 3.12 sum() is compensated and a 20k-trial search found no order-sensitive positive
    split, so this holds even without apply_block's canonical-order path; that path guarantees it on every
    interpreter (the same pre-3.12 concern chain_transfers._COMPENSATED_FLOAT_SUM records)."""
    coord = gen_account()
    ws = {gen_account().address: w for w in (0.1, 0.2, 0.3, 1e-9, 7.0, 1.0 / 3.0, 2.5e-7, 0.7)}
    cs = ChainSettlement(coord, genesis_checkpoint="g", bind_mint_split=True)
    cs.settle(1, 123.456, ws, T0)
    genuine = _export(cs)
    rev = _mut(genuine, lambda e: e["blocks"][0].update(contributors=dict(reversed(list(ws.items())))))
    st_g, ok_g, _ = _replay_blocks(genuine["blocks"], coord.address, "g")
    st_r, ok_r, _ = _replay_blocks(rev["blocks"], coord.address, "g")
    assert ok_g and ok_r and st_g.state_hash() == st_r.state_hash() == cs.state.state_hash()


# ------------------------------------------------------------------------------------ 4. mixed chains + strict mode
def test_mixed_chain_reports_exactly_the_legacy_blocks_and_rejects_tampering_on_the_bound_ones():
    coord, a, b, evil = gen_account(), gen_account(), gen_account(), gen_account()
    cs = ChainSettlement(coord, genesis_checkpoint="g")         # flag OFF (legacy)
    cs.settle(1, 10.0, {a.address: 1.0}, T0)
    cs.settle(1, 20.0, {a.address: 1.0, b.address: 3.0}, T0 + 1)
    cs.bind_mint_split = True                                   # the owner's flip (a restart with the env set)
    cs.settle(2, 30.0, {b.address: 1.0}, T0 + 2)
    cs.settle(2, 40.0, {a.address: 0.5, b.address: 0.5}, T0 + 3)
    ex = _export(cs)
    assert _verify(ex, coord)[:2] == (True, "ok") and cs.verify() == (True, "ok")
    rep = verify_public_chain_report(ex, expected_coord_address=coord.address)
    assert rep["ok"] and rep["unbound_mints"] == [0, 1] and rep["n_bound_mints"] == 2 and rep["n_blocks"] == 4
    assert cs.unbound_mints() == [0, 1]
    # a bound block: tamper rejected ...
    ok, reason, *_ = _verify(_mut(ex, lambda e: e["blocks"][3].update(contributors={evil.address: 1.0})), coord)
    assert not ok and reason.startswith("block 3: does not apply") and SPLIT_REASON in reason
    # ... a legacy block: the residual stays (nothing signed it) but it is exactly what the report names
    legacy_tamper = _mut(ex, lambda e: e["blocks"][1].update(contributors={evil.address: 1.0}))
    r = verify_public_chain_report(legacy_tamper, expected_coord_address=coord.address)
    assert r["ok"] and r["balances"][evil.address] == 20.0 and 1 in r["unbound_mints"]
    # strict mode refuses the mixed chain at its FIRST legacy block, with a named reason
    ok, reason, bals, *_ = _verify(ex, coord, require_split_binding=True)
    assert not ok and reason.startswith("block 0: ") and STRICT_REASON in reason and bals == {}
    # a restart keeps the verdict: from_state replays the bound blocks' check and reports the same legacy set
    restored = ChainSettlement.from_state(json.loads(json.dumps(cs.to_state())))
    assert restored.unbound_mints() == [0, 1] and restored.verify() == (True, "ok")


def test_strict_mode_rejects_a_legacy_chain_and_accepts_an_all_bound_one():
    coord, a, b, c = gen_account(), gen_account(), gen_account(), gen_account()
    legacy = ChainSettlement(coord, genesis_checkpoint="g")
    legacy.settle(1, 5.0, {a.address: 1.0}, T0)
    ok, reason, bals, minted, height = _verify(_export(legacy), coord, require_split_binding=True)
    assert not ok and reason == ("block 0: mint split is not bound into the signed header "
                                 "(legacy unbound mint; split binding required)")
    assert (bals, minted, height) == ({}, 0.0, 0)
    assert _verify(_export(legacy), coord)[:2] == (True, "ok")                           # non-strict: as before
    bound = _bound_chain(coord, a, b, c)
    assert _verify(_export(bound), coord, require_split_binding=True)[:2] == (True, "ok")
    empty = ChainSettlement(coord, genesis_checkpoint="g")     # no blocks: vacuously ok
    assert _verify(_export(empty), coord, require_split_binding=True)[:2] == (True, "ok")


# ------------------------------------------------------------------------------------ 5. the quorum path
def _proof(vals, roster, *, recipient, amount, height):
    atts = [ds.sign_settlement(v, recipient=recipient, amount=amount, height=height, delta_cid="") for v in vals]
    return ds.settlement_block_proof(recipient=recipient, amount=amount, height=height, delta_cid="",
                                     roster=roster, attestations=atts)


@pytest.mark.parametrize("bind", [False, True])
def test_quorum_chain_carries_both_hashes_and_the_split_closes_what_a_multi_recipient_quorum_leaves(bind):
    """A pinned-roster quorum binds (recipient, amount, height) -- but for a MULTI-recipient split it signs
    recipient=None, so WHO shares the mint was never bound under require_quorum either. split_hash binds it."""
    coord, a, b, evil = gen_account(), gen_account(), gen_account(), gen_account()
    vals = [gen_account() for _ in range(3)]
    roster = {v.address: 1000.0 for v in vals}
    cs = ChainSettlement(coord, genesis_checkpoint="q", bind_mint_split=bind)
    cs.settle(1, 10.0, {a.address: 1.0}, T0, quorum=_proof(vals, roster, recipient=a.address, amount=10.0, height=1))
    cs.settle(2, 8.0, {a.address: 1.0, b.address: 1.0}, T0 + 1,
              quorum=_proof(vals, roster, recipient=None, amount=8.0, height=2))
    ex = _export(cs)
    kw = {"require_quorum": True, "quorum_roster": roster}
    assert all(bool(r["header"].get("quorum_hash")) for r in ex["blocks"])
    assert all(bool(r["header"].get("split_hash")) == bind for r in ex["blocks"])
    ok, reason, bals, *_ = _verify(ex, coord, **kw)
    assert ok and reason.startswith("ok (quorum-enforced") and bals == {a.address: 14.0, b.address: 4.0}
    assert cs.verify() == (True, "ok")
    assert ChainSettlement.from_state(json.loads(json.dumps(cs.to_state()))).verify() == (True, "ok")
    repointed = _mut(ex, lambda e: e["blocks"][1].update(contributors={a.address: 1.0, evil.address: 1.0}))
    ok, reason, bals, *_ = _verify(repointed, coord, **kw)
    if bind:
        assert not ok and reason.startswith("block 1: does not apply") and SPLIT_REASON in reason
    else:                                        # the pre-fix residual on the quorum path, stated honestly
        assert ok and bals[evil.address] == 4.0
    stripped = _mut(ex, lambda e: e["blocks"][0].pop("quorum"))
    assert not _verify(stripped, coord, **kw)[0]                                       # the quorum still binds


# ------------------------------------------------------------------------------------ 6. restore + the flag itself
def test_from_state_refuses_a_checkpoint_whose_bound_split_was_edited():
    coord, a, b, c, evil = gen_account(), gen_account(), gen_account(), gen_account(), gen_account()
    cs = _bound_chain(coord, a, b, c)
    state = json.loads(json.dumps(cs.to_state()))
    assert ChainSettlement.from_state(copy.deepcopy(state)).verify() == (True, "ok")    # positive case
    state["blocks"][1]["contributors"] = {a.address: 2.0, evil.address: 1.0}
    with pytest.raises(ValueError, match="block 1: does not apply .*split_hash"):
        ChainSettlement.from_state(state)


def test_the_emission_flag_env_ctor_precedence_and_the_restart_flip(monkeypatch):
    coord, a = gen_account(), gen_account()
    assert ChainSettlement(coord).bind_mint_split is False                              # default OFF
    for off in ("0", "off", "false", "", "garbage"):
        monkeypatch.setenv(BIND_MINT_SPLIT_ENV, off)
        assert ChainSettlement(coord).bind_mint_split is False
    monkeypatch.setenv(BIND_MINT_SPLIT_ENV, "1")
    assert ChainSettlement(coord).bind_mint_split is True
    assert ChainSettlement(coord, bind_mint_split=False).bind_mint_split is False       # explicit arg wins
    monkeypatch.delenv(BIND_MINT_SPLIT_ENV)
    legacy = ChainSettlement(coord, genesis_checkpoint="g")
    legacy.settle(1, 5.0, {a.address: 1.0}, T0)
    state = legacy.to_state()
    assert "bind_mint_split" not in json.dumps(state)                                   # never persisted
    monkeypatch.setenv(BIND_MINT_SPLIT_ENV, "1")                                        # the coordinated restart
    resumed = ChainSettlement.from_state(json.loads(json.dumps(state)))
    assert resumed.bind_mint_split is True and resumed.blocks == legacy.blocks          # history untouched
    resumed.settle(2, 5.0, {a.address: 1.0}, T0 + 1)
    assert resumed.unbound_mints() == [0] and resumed.blocks[1]["header"]["split_hash"]
    assert _verify(_export(resumed), coord)[:2] == (True, "ok")


def test_a_bound_settle_with_a_non_string_address_raises_and_appends_nothing():
    coord, a = gen_account(), gen_account()
    cs = ChainSettlement(coord, genesis_checkpoint="g", bind_mint_split=True)
    cs.settle(1, 5.0, {a.address: 1.0}, T0)
    before = (len(cs.blocks), cs.head, dict(cs.state.bal), cs.minted(), set(cs.paid_cids))
    with pytest.raises(ValueError, match="must be a str"):
        cs.settle(2, 5.0, {12345: 1.0}, T0 + 1)
    assert (len(cs.blocks), cs.head, dict(cs.state.bal), cs.minted(), set(cs.paid_cids)) == before
    assert cs.verify() == (True, "ok")


def test_apply_block_binds_the_split_whoever_replays_it():
    a, b = gen_account().address, gen_account().address
    split = {a: 1.0, b: 3.0}
    hdr = BlockHeader(height=1, prev_hash="0" * 64, beacon="b", parent_checkpoint="c", new_checkpoint="c",
                      work_score=4.0, proposer="p", reward=8.0, tx_root="0" * 64, timestamp=1,
                      split_hash=mint_split_hash(split))
    st = State(model_checkpoint="c").apply_block(Block(header=hdr, txs=[], pouw={}), split)
    assert (st.balance(a), st.balance(b)) == (2.0, 6.0)
    with pytest.raises(ValueError, match=SPLIT_REASON):
        State(model_checkpoint="c").apply_block(Block(header=hdr, txs=[], pouw={}), {a: 3.0, b: 1.0})
    legacy = BlockHeader.from_dict({k: v for k, v in hdr.to_dict().items() if k != "split_hash"})
    st2 = State(model_checkpoint="c").apply_block(Block(header=legacy, txs=[], pouw={}), {a: 3.0, b: 1.0})
    assert (st2.balance(a), st2.balance(b)) == (6.0, 2.0)                               # legacy: unchecked, as before


# ------------------------------------------------------------------------------------ 7. the other replayers
@pytest.mark.parametrize("bind", [False, True])
def test_round_authority_candidate_accept_refuses_a_repointed_bound_block(monkeypatch, bind):
    if bind:
        monkeypatch.setenv(BIND_MINT_SPLIT_ENV, "1")           # candidates build their replicas from the env
    auth = ra.build_round_authority(n_candidates=2, n_validators=5, quorum_m=3, grace_ticks=3)
    order = auth.begin_round(1)["order"]
    proposer = auth.by_identity[ra._norm(order[0])]
    peer = auth.by_identity[ra._norm(order[1])]
    recipient, evil = gen_account().address, gen_account().address
    quorum = auth._build_quorum(recipient=recipient, amount=10.0, height=1, delta_cid="ra-round-1")
    env = proposer.propose(1, height=1, reward=10.0, recipient=recipient, quorum=quorum, timestamp=T0)
    assert bool(env["block"]["header"].get("split_hash")) == bind
    tampered = copy.deepcopy(env)
    tampered["block"]["contributors"] = {evil: 1.0}
    ok, reason = peer.accept(tampered, order, unavailable=frozenset())
    if bind:
        assert (ok, reason) == (False, "block does not apply (ValueError)") and peer.chain.blocks == []
        assert peer.accept(env, order, unavailable=frozenset()) == (True, "ok")          # the genuine block
        assert peer.chain.state.balance(recipient) == 10.0 and peer.block_hashes() == proposer.block_hashes()
    else:                                        # the pre-fix hole on this path: the peer credits the attacker
        assert (ok, reason) == (True, "ok") and peer.chain.state.balance(evil) == 10.0


# ------------------------------------------------------------------------------------ 8. refutation-review regressions
# Built with chr() so no escaping layer (editor, JSON transport, shell) can silently turn one form into the other.
ASTRAL = chr(0x1F600)                        # one code point...
TWIN = chr(0xD83D) + chr(0xDE00)             # ...and the adjacent lone-surrogate pair that escapes to the SAME text


def test_a_surrogate_twin_of_an_astral_address_cannot_share_its_commitment():
    """Review 2026-09-24 (A1-A3): under ensure_ascii an astral character and its joinable lone-surrogate twin escape
    identically, so the twin key -- smuggled in as raw surrogate bytes (json.loads(bytes) decodes them), or by a
    registry declaring charset=unicode_escape to the wallet's HTTP client -- verified on every path and took the
    victim's mint. A joinable pair now has no canonical form; a lone surrogate still binds and round-trips."""
    assert len(ASTRAL) == 1 and len(TWIN) == 2 and json.dumps(ASTRAL) == json.dumps(TWIN)   # the collision
    victim = "0xV" + ASTRAL
    with pytest.raises(ValueError, match="joinable surrogate pair"):
        mint_split_hash({"0xV" + TWIN: 1.0})
    coord = gen_account()
    cs = ChainSettlement(coord, genesis_checkpoint="g", bind_mint_split=True)
    cs.settle(1, 100.0, {victim: 1.0}, T0)
    text = json.dumps(cs.public_state(), ensure_ascii=False)
    genuine = json.loads(text.encode("utf-8"))
    assert _verify(genuine, coord)[:3] == (True, "ok", {victim: 100.0})                 # positive case
    via_bytes = json.loads(text.replace(victim, "0xV" + TWIN).encode("utf-8", "surrogatepass"))
    via_charset = json.loads(json.dumps(cs.public_state()).encode("ascii").decode("unicode_escape"))
    for forged in (via_bytes, via_charset):
        assert list(forged["blocks"][0]["contributors"]) == ["0xV" + TWIN]              # the twin got through
        ok, reason, bals, *_ = _verify(forged, coord)
        assert not ok and reason.startswith("block 0: does not apply") and "surrogate" in reason and bals == {}
    before = len(cs.blocks)
    with pytest.raises(ValueError, match="joinable surrogate pair"):     # the producer refuses one too
        cs.settle(2, 1.0, {"0xV" + TWIN: 1.0}, T0 + 1)
    assert len(cs.blocks) == before
    lone = "0xV" + chr(0xD83D) + "X"         # a LONE surrogate is not a collision: it binds and round-trips as-is
    cs.settle(2, 3.0, {lone: 1.0}, T0 + 1)
    back = json.loads(json.dumps(cs.public_state()))
    assert list(back["blocks"][1]["contributors"]) == [lone]
    assert _verify(back, coord)[:3] == (True, "ok", {victim: 100.0, lone: 3.0})


class _TwoFaced(dict):
    """Shows its genuine contents on the first items() call and another split afterwards (review A15)."""

    def __init__(self, genuine, other):
        super().__init__(genuine)
        self._other, self._calls = other, 0

    def items(self):
        self._calls += 1
        return dict.items(self) if self._calls == 1 else self._other.items()


def test_a_dict_subclass_cannot_show_the_hash_one_split_and_the_credit_loop_another():
    a, evil = gen_account().address, gen_account().address
    hdr = BlockHeader(height=1, prev_hash="0" * 64, beacon="b", parent_checkpoint="c", new_checkpoint="c",
                      work_score=1.0, proposer="p", reward=5.0, tx_root="0" * 64, timestamp=1,
                      split_hash=mint_split_hash({a: 1.0}))
    with pytest.raises(ValueError, match="plain dict"):
        State(model_checkpoint="c").apply_block(Block(header=hdr, txs=[], pouw={}), _TwoFaced({a: 1.0}, {evil: 1.0}))
    st = State(model_checkpoint="c").apply_block(Block(header=hdr, txs=[], pouw={}), {a: 1.0})   # positive case
    assert st.balance(a) == 5.0 and st.balance(evil) == 0.0


def test_a_txs_key_cannot_hide_a_legacy_mint_from_unbound_mints():
    """Review 2026-09-24 (A17): the base replay ignores a record's "txs" key, so a checkpoint that tacked one onto a
    legacy mint restored fine -- and unbound_mints() used to skip it. Only a provably empty transfer block is
    skipped now."""
    coord, a = gen_account(), gen_account()
    legacy = ChainSettlement(coord, genesis_checkpoint="g")
    legacy.settle(1, 100.0, {a.address: 1.0}, T0)
    for junk in ([], ["x"]):
        state = json.loads(json.dumps(legacy.to_state()))
        state["blocks"][0]["txs"] = junk
        restored = ChainSettlement.from_state(state)
        assert restored.verify() == (True, "ok") and restored.unbound_mints() == [0], junk
