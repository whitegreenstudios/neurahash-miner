"""
neura_l1.incentives — the economic layer that makes optimistic / refereed verification
actually work, by defeating the **Verifier's Dilemma**.

Our refereed-delegation verifier (`neura_l1/refereed.py`) is only secure if rational
verifiers *actually verify*. Luu et al. (2015) showed they won't: if cheating is rare and
verification is costly, a rational verifier just reports "valid" without checking — and once
no one checks, the security collapses. This is acute for ML PoUW because verification is
expensive.

This module implements the research solutions, mapped to our chain:

  * **Proof of Sampling** (Zhang et al., arXiv:2405.00295): challenge each task with
    probability p; there is a *unique pure-strategy Nash equilibrium* in which everyone is
    honest as long as
            p  >  C / ((1 - r) (R + S))                                  (PoSP, Prop. 1)
    where C = verification cost, R = reward, S = slash, r = the fraction of validators an
    attacker controls. With a large slash this p* is tiny (sub-1%), so verification is cheap
    *and* secure — no zk proofs needed.

  * **Proof-of-Learning with Incentive Security** (Zhang et al., arXiv:2404.09005): the
    training seed = Hash(prev_block, prover_id) and per-stage weight-hash commitments,
    audited at random — which is *exactly* NeuraHash's beacon + per-step trajectory
    commitment (`pouw_torch`/`refereed`). It adds the incentive analysis below.

  * **Truebit** (Teutsch & Reitwiessner): inject a "forced error" with probability f so a
    verifier who never checks is eventually caught and penalised — guaranteeing verifiers
    keep a positive expected value from checking even when honest provers are the norm.

All functions are pure economics (no torch/CUDA); the audit *selector* is a deterministic,
beacon-bound pseudo-random draw so selection is unpredictable yet publicly checkable.
"""

import hashlib

DEFAULT_ADVERSARY_FRACTION = 0.10     # r: max share of validators an attacker is assumed to hold
DEFAULT_BOUNTY_FRAC = 0.5             # fraction of a slash paid to the successful challenger


# ===========================================================================
# Proof-of-Sampling: minimum challenge probability for a Nash-honest equilibrium
# ===========================================================================
def min_sample_probability(verify_cost, reward, slash, adversary_fraction=DEFAULT_ADVERSARY_FRACTION):
    """PoSP p* = C / ((1 - r)(R + S)). Challenging at probability >= this makes honesty the
    unique pure-strategy Nash equilibrium (verifiers can't profitably skip; provers can't
    profitably cheat). All of C, R, S must be in the same units (e.g. multiples of the
    verification cost C, or token value)."""
    denom = (1.0 - adversary_fraction) * (reward + slash)
    if denom <= 0:
        return 1.0
    return min(1.0, verify_cost / denom)


def is_nash_honest(sample_prob, verify_cost, reward, slash,
                   adversary_fraction=DEFAULT_ADVERSARY_FRACTION):
    """True iff sampling at `sample_prob` meets the PoSP threshold (honest = Nash)."""
    return sample_prob >= min_sample_probability(verify_cost, reward, slash, adversary_fraction)


# ===========================================================================
# Prover side: cheating must be negative-EV
# ===========================================================================
def cheater_ev(reward, slash, work_saved, catch_prob):
    """Expected value of cheating: keep the reward when undetected, lose the slash when
    caught, and pocket `work_saved` (the compute the cheater skipped). `catch_prob` is the
    probability the cheat is detected (≈ sample_prob for recompute-based audit)."""
    return (1.0 - catch_prob) * reward - catch_prob * slash + work_saved


def prover_deterred(reward, slash, work_saved, catch_prob):
    """Honest (EV = reward) strictly dominates cheating."""
    return cheater_ev(reward, slash, work_saved, catch_prob) < reward


def min_catch_prob_to_deter(reward, slash, work_saved):
    """Smallest catch probability that makes cheating negative-EV: p > work_saved/(R+S)."""
    rs = reward + slash
    return 1.0 if rs <= 0 else min(1.0, work_saved / rs)


# ===========================================================================
# Verifier side: verifying must be positive-EV (the actual Verifier's-Dilemma fix)
# ===========================================================================
def verifier_ev(verify_cost, bounty, fraud_prob):
    """A challenged verifier pays `verify_cost` and earns `bounty` when it catches fraud.
    `fraud_prob` includes any deliberately injected forced errors. Positive => verifying is
    rational."""
    return fraud_prob * bounty - verify_cost


def forced_error_min_rate(verify_cost, bounty):
    """Truebit: minimum forced-error injection rate f so that verifying is +EV even when real
    cheating is ~0: f > C / bounty. Below this, verifiers rationally stop checking."""
    return 1.0 if bounty <= 0 else min(1.0, verify_cost / bounty)


def challenge_bounty(slash, frac=DEFAULT_BOUNTY_FRAC):
    """Reward paid to a verifier who successfully proves fraud, taken from the slash."""
    return max(0.0, slash * frac)


# ===========================================================================
# Unpredictable-but-verifiable audit selection (PoSP "unknown validator selection")
# ===========================================================================
def must_audit(randomness, verifier_id, sample_prob, salt="audit"):
    """Deterministic coin flip: returns True iff `verifier_id` is selected to audit, at rate
    ~`sample_prob`, seeded by `randomness`. Publicly recomputable (a selected verifier can't deny
    it; an unselected one can't fake it).

    CONTRACT — `randomness` MUST be fixed only AFTER the prover has committed the work being
    audited (e.g. the SUCCESSOR block's hash, a future VRF/beacon output, or a verifier-side
    commit-reveal). If it is something the prover already knows when building its proof (e.g. the
    PARENT hash, a plain counter, or any value the prover can grind), the prover can locally
    compute exactly which verifiers will be selected and cheat only on blocks where no honest
    verifier is chosen — defeating Proof-of-Sampling. The parameter is named `randomness` (not
    `prev_hash`) precisely to flag this: pass post-commitment randomness."""
    if sample_prob >= 1.0:
        return True
    if sample_prob <= 0.0:
        return False
    h = hashlib.sha256(f"{randomness}|{verifier_id}|{salt}".encode()).hexdigest()
    # map the first 52 bits to [0,1)
    frac = int(h[:13], 16) / float(1 << 52)
    return frac < sample_prob


# ===========================================================================
# One-shot recommendation + security check for a parameter set
# ===========================================================================
def recommend(reward, verify_cost, slash, work_saved=None,
              adversary_fraction=DEFAULT_ADVERSARY_FRACTION, bounty_frac=DEFAULT_BOUNTY_FRAC,
              margin=1.5):
    """Return a coherent, incentive-secure parameter set for the verification game. The PoSP
    p* and the forced-error rate are *thresholds*; we set the recommended values `margin`×
    above them so the equilibrium holds strictly (verifying is strictly +EV, etc.)."""
    if work_saved is None:
        work_saved = verify_cost          # a cheater saves ~the cost of doing the work
    p_star = min_sample_probability(verify_cost, reward, slash, adversary_fraction)
    sample_prob = min(1.0, p_star * margin)
    bounty = challenge_bounty(slash, bounty_frac)
    f_min = forced_error_min_rate(verify_cost, bounty)
    forced = min(1.0, f_min * margin)
    return {
        "sample_prob": sample_prob,
        "posp_threshold": p_star,
        "bounty": bounty,
        "forced_error_rate": forced,
        "prover_catch_prob_needed": min_catch_prob_to_deter(reward, slash, work_saved),
        "secure": is_incentive_secure(sample_prob, verify_cost, reward, slash, bounty,
                                      work_saved, forced, adversary_fraction)[0],
    }


def is_incentive_secure(sample_prob, verify_cost, reward, slash, bounty, work_saved,
                        forced_error_rate, adversary_fraction=DEFAULT_ADVERSARY_FRACTION):
    """Check all three legs hold. Returns (ok, reasons[])."""
    reasons = []
    if not is_nash_honest(sample_prob, verify_cost, reward, slash, adversary_fraction):
        reasons.append("sample_prob below PoSP Nash threshold")
    if not prover_deterred(reward, slash, work_saved, sample_prob):
        reasons.append("cheating is not negative-EV at this sample_prob")
    # a forced-error-driven verifier must come out ahead: f * bounty - C > 0
    if verifier_ev(verify_cost, bounty, forced_error_rate) <= 0:
        reasons.append("verifying is not positive-EV (verifier's dilemma unsolved)")
    return (len(reasons) == 0, reasons)
