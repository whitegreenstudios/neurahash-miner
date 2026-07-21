"""
Read-only adapter: on-chain validator ETH bonds (contracts/VerifierQuorum.sol) -> a plain
{addr: stake} Python dict.

WHY: NeuraHash's stake-weighted M-of-N settlement quorum
(neurahash/diloco_settlement.py::collect_settlement_signatures + ::staked_roster,
neurahash/diloco_committee.py::aggregate_diloco_attestations) authorizes on a {addr: stake}
weight map. That map flows in via `staked_roster`/`roster_source_select`
(diloco_settlement.py:116,418), which duck-type their `stake_source` argument: an object
with `.stake_of` is treated as a neura_l1 `State`, ELSE it is treated as a plain
`{address: stake}` mapping (diloco_settlement.py:126-131). This module builds that plain
mapping from each validator's REAL slashable ETH bond, posted on-chain via
VerifierQuorum.postBond() and read back from the `bondOf` mapping (VerifierQuorum.sol:48),
so a stake-weighted quorum can eventually be backed by real staked ETH on an EVM L2 instead
of an off-chain-only stake figure.

SCOPE: standalone, reusable READ path only.
  - It never sends a transaction (no postBond / attestFraud / governance calls) -- connects
    read-only and only ever calls `.call()`, never `.transact()`.
  - It is NOT wired into sharded_pool_node.py, bridge.py, diloco_settlement.py, or
    diloco_committee.py. Wiring this in (e.g. at sharded_pool_node.py:5825's
    `roster_source_select` call site) is a SEPARATE, later, owner-gated step once a real L2
    VerifierQuorum deployment exists. Do not import this module from the live coordinator
    path without that explicit follow-up.

UNIT: `validator_bonds()` / `validator_bonds_from_contract()` return
`{address_lowercased: bond_in_ether}` where the value is a Python `float`. `bondOf` is
denominated in wei on-chain (`uint256`); this module converts wei -> ether via exact
`Decimal` division by 10**18 and only casts to `float` at the very end. `float` (IEEE-754
binary64, ~15-17 significant decimal digits) is exact for any realistic validator bond
(single-digit to low-thousands ETH) but is NOT wei-exact in general -- a caller that needs
wei-exact precision should read `bondOf(addr)` directly instead of going through this
float-converting helper.

FAIL-CLOSED: a connection/RPC/ABI/call failure raises `EvmStakeSourceError`. This
deliberately does NOT catch-and-return `{}` on error: an empty map would fail-OPEN a
stake-weighted quorum (zero required stake to reach any threshold) instead of surfacing the
outage to the caller. An empty dict is returned ONLY when the on-chain validator set is
genuinely empty (`validatorCount() == 0`) -- a verified real answer, not a failure
masquerading as one.
"""

import json
import os
from decimal import Decimal

from web3 import Web3

_ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "contracts")
_CONTRACT_NAME = "VerifierQuorum"
_WEI_PER_ETHER = Decimal(10) ** 18


class EvmStakeSourceError(Exception):
    """On-chain stake read failed (RPC unreachable, bad/missing ABI artifact, invalid
    address, or a reverted/failed call). Never silently mapped to {} -- see this module's
    docstring, "FAIL-CLOSED" section, for why."""


def _load_abi(name: str = _CONTRACT_NAME) -> list:
    """Load a compiled contract's ABI the same way neurahash/bridge.py does
    (bridge.py:17-23: artifacts/contracts/{name}.sol/{name}.json). Re-implemented here
    rather than imported so this module has zero dependency on bridge.py, whose deploy
    path is unrelated to reading and is currently broken for VerifierQuorum (passes 4
    constructor args; the contract needs 5)."""
    path = os.path.join(_ART_DIR, f"{name}.sol", f"{name}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
    except FileNotFoundError as e:
        raise EvmStakeSourceError(
            f"{name} ABI artifact not found at {path} -- run `npx hardhat compile` first"
        ) from e
    except json.JSONDecodeError as e:
        raise EvmStakeSourceError(f"{name} artifact at {path} is not valid JSON: {e!r}") from e
    try:
        return artifact["abi"]
    except KeyError as e:
        raise EvmStakeSourceError(f"{name} artifact at {path} has no 'abi' key") from e


def connect(rpc_url: str) -> Web3:
    """Connect to an EVM JSON-RPC endpoint, read-only (no account/signing setup).
    Raises EvmStakeSourceError rather than ever handing back a Web3 instance that isn't
    actually live -- callers should never need to re-check `.is_connected()` themselves."""
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        connected = w3.is_connected()
    except Exception as e:
        raise EvmStakeSourceError(f"RPC connection to {rpc_url!r} failed: {e!r}") from e
    if not connected:
        raise EvmStakeSourceError(f"no EVM node reachable at {rpc_url!r}")
    return w3


def get_quorum_contract(w3: Web3, quorum_address: str):
    """Build a read-only VerifierQuorum contract handle at `quorum_address` on `w3`."""
    abi = _load_abi(_CONTRACT_NAME)
    try:
        checksummed = Web3.to_checksum_address(quorum_address)
    except ValueError as e:
        raise EvmStakeSourceError(f"not a valid address: {quorum_address!r}: {e!r}") from e
    return w3.eth.contract(address=checksummed, abi=abi)


def validator_bonds_from_contract(quorum) -> dict:
    """Read `{validator_addr_lowercased: bond_in_ether}` from an already-connected
    VerifierQuorum contract handle (e.g. built via `get_quorum_contract`, or any
    web3.py-compatible Contract/test-double exposing the same `.functions.*` surface).

    Pure read path: `validatorCount()` once, then `validatorList(i)` + `bondOf(addr)` for
    each `i` in `[0, count)`. VerifierQuorum.sol exposes no `getAllValidators()`
    convenience function -- confirmed: only the auto-getters for the `validatorCount`
    uint256 (VerifierQuorum.sol:41) and the `validatorList` address[] array
    (VerifierQuorum.sol:56). No transaction is ever sent by this function -- every call
    below is a `.call()`, never a `.transact()`.

    Kept separate from `validator_bonds()` so callers (including tests) can pass in a
    contract handle built against any provider -- including a non-network test double --
    without this function hard-depending on a live RPC connection.
    """
    try:
        count = quorum.functions.validatorCount().call()
    except Exception as e:
        raise EvmStakeSourceError(f"validatorCount() read failed: {e!r}") from e

    result = {}
    for i in range(count):
        try:
            addr = quorum.functions.validatorList(i).call()
        except Exception as e:
            raise EvmStakeSourceError(f"validatorList({i}) read failed: {e!r}") from e
        try:
            bond_wei = quorum.functions.bondOf(addr).call()
        except Exception as e:
            raise EvmStakeSourceError(f"bondOf({addr}) read failed: {e!r}") from e
        result[addr.lower()] = float(Decimal(bond_wei) / _WEI_PER_ETHER)  # wei -> ether
    return result


def validator_bonds(rpc_url: str, quorum_address: str) -> dict:
    """Standalone entry point: connect to `rpc_url`, read the VerifierQuorum deployed at
    `quorum_address`, and return `{validator_addr_lowercased: bond_in_ether}`.

    Empty on-chain validator set -> `{}` (a verified real answer, read via
    `validatorCount() == 0`).
    Any connection/RPC/ABI/call failure -> `EvmStakeSourceError` (never a silent `{}`; see
    this module's docstring, "FAIL-CLOSED" section -- an empty map would fail-OPEN a
    stake-weighted quorum).
    """
    w3 = connect(rpc_url)
    quorum = get_quorum_contract(w3, quorum_address)
    return validator_bonds_from_contract(quorum)
