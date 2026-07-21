"""
Decentralized verification: an M-of-N validator quorum instead of one trusted
coordinator.

The audit's biggest design gap was the single coordinator: it can censor, collude,
or be the attacker. Because verification is now DETERMINISTIC (check_submission
replays the full trajectory), every honest validator independently reaches the
SAME verdict on a submission. So we can require agreement among a quorum:

  - N validators each independently vote valid / invalid on a worker's submission.
  - If >= M say INVALID  -> the worker is slashed (fraud confirmed by quorum).
  - If >= M say VALID    -> the submission is accepted.
  - Validators who voted AGAINST the decided outcome are themselves slashed
    (lying / lazy validators lose their bond), so a dishonest minority is punished
    and honest validators are rewarded.

THRESHOLD: M MUST be a strict majority (M > N/2). Otherwise the valid/invalid bands
overlap and a split could satisfy both — quorum_verify enforces this.

Security vs liveness (do not conflate):
  - SAFETY  (no wrong outcome): holds iff fewer than M validators are dishonest AND
    M > N/2 (honest validators form the deciding majority). Any M colluding
    validators can force a wrong outcome, so pick N,M accordingly.
  - LIVENESS (no censorship):  a decision is reached as long as >= M validators
    respond honestly; absent/timed-out validators lower the effective N.
Recommended: M = floor(N/2) + 1.
"""

from .verification import check_submission


class Validator:
    """A staked verifier. Honest validators report the true deterministic verdict;
    a dishonest one flips it (vouches for cheaters / censors honest work)."""
    def __init__(self, address, honest=True):
        self.address = address
        self.honest = honest

    def vote(self, model, global_params, submission, shard, H, lr):
        valid, _ = check_submission(model, global_params, submission, shard, H=H, lr=lr)
        return valid if self.honest else (not valid)


def quorum_verify(model, global_params, submission, shard, validators, ledger,
                  M, H=30, lr=0.5, validator_slash=10.0, log=None):
    """Run the quorum. Returns (accepted, info). Slashes the worker on an invalid
    quorum, and slashes validators on the losing side of the vote."""
    if not (M * 2 > len(validators)):
        raise ValueError(f"M must be a strict majority (M > N/2); got M={M}, N={len(validators)}")
    if len({v.address for v in validators}) != len(validators):
        raise ValueError("duplicate validator addresses")
    worker = submission["address"]
    votes = {v.address: v.vote(model, global_params, submission, shard, H, lr)
             for v in validators}
    n_valid = sum(1 for ok in votes.values() if ok)
    n_invalid = len(votes) - n_valid

    info = {"votes": votes, "n_valid": n_valid, "n_invalid": n_invalid,
            "accepted": None, "worker_slashed": False, "validators_slashed": []}

    if n_invalid >= M:
        accepted = False
        ledger.slash(worker, reason=f"quorum: {n_invalid}/{len(validators)} say invalid")
        info["worker_slashed"] = True
    elif n_valid >= M:
        accepted = True
    else:
        accepted = None     # no quorum either way -> dispute (worker untouched)

    # punish validators who disagreed with a decisive outcome (lying/lazy)
    if accepted is not None:
        for addr, ok in votes.items():
            if ok != accepted:
                ledger.slash(addr, validator_slash, reason="validator off-consensus")
                info["validators_slashed"].append(addr)

    info["accepted"] = accepted
    if log:
        log(f"  quorum on {worker}: valid={n_valid} invalid={n_invalid} "
            f"(M={M}) -> accepted={accepted} "
            f"| validators slashed: {info['validators_slashed'] or 'none'}")
    return accepted, info
