"""
neura_l1 — a standalone Proof-of-Useful-Work Layer-1 prototype (PUBLIC MINER TRIM).

"The verified training delta IS the block." Useful work (training the shared MoE model) is the
consensus, not hashing.

PUBLIC MINER NOTE: the full package's __init__ eagerly re-exports the consensus foundation
(`block_state`, `consensus`, `p2p_node`, `pouw_gate`), which are PRIVATE core modules NOT shipped
in the public miner. This trimmed __init__ imports NOTHING at package load, so the public miner
can `import neura_l1.signing` / `neura_l1.gpu_miner` / `neura_l1.pqc` without dragging the private
consensus core. The private full-node package keeps its own __init__ with the full re-exports.
"""

__all__ = []
