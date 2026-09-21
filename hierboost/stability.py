"""Stability selection wrapper (2026-09-19), motivated directly by the precision
investigation in genomics_precision_investigation.md: the threshold sweep there
found no single cutoff on theta_hat gives good precision on the real ground-truth
problem (167 candidates, N=1164, K_CAUSAL=4) -- too strict collapses power, too
loose (the PPL-filter's own stopping rule) over-retains to ~5% precision. That is
exactly the instability Meinshausen & Buhlmann (2010) "Stability Selection" was
designed to fix: instead of trusting one fit's threshold, refit on many random
subsamples of the candidate pool and keep only features that get selected under a
LOOSE inner rule *consistently* across subsamples. Real signal survives whichever
subset it lands in (given adequate power per subset); features that only pass the
loose inner rule because of subsample-specific noise wash out under aggregation.
This also borrows the random-subspace idea from Random Forests: subsampling
CANDIDATE FEATURES (not just rows) is what lets correlated near-neighbors compete
against each other in different subsets rather than always co-occurring.

Deliberately decoupled from any specific fitter or response family: the caller
supplies `fit_selector`, a closure that takes a subset of candidate indices and
returns the LOCAL indices (positions within that subset) it selects. This keeps
the primitive reusable across the Gaussian pMOM fitter used first, the local-prior
fitter, or a future Bernoulli one -- none of that is stability_select's concern.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class StabilityResult:
    stability: np.ndarray          # (p,) selection frequency among subsets where candidate
    candidate_count: np.ndarray    # (p,) number of subsets each feature was sampled into
    n_subsets: int
    subsample_frac: float


def stability_select(fit_selector, p, n_subsets=100, subsample_frac=0.5, rng=None, min_subset=2):
    """Random-subspace stability selection. Draws `n_subsets` random subsets of the p
    candidate features (size max(min_subset, round(p*subsample_frac)), without
    replacement within a subset), calls `fit_selector(subset_idx)` on each, and
    accumulates how often each feature is selected relative to how often it was even
    a candidate (features are not candidates in every subset, so the denominator is
    per-feature, not n_subsets).

    `fit_selector(subset_idx) -> array of LOCAL indices into subset_idx` (i.e. values
    in [0, len(subset_idx)), NOT the original feature ids) that the inner selection
    rule keeps for that subset. The inner rule is intentionally the caller's choice
    (a loose theta_hat threshold, top-K, a PPL-filter's best step, ...) -- stability
    selection's own error-control mechanism is the aggregation across subsets, not
    the per-subset rule's stringency, so the inner rule should generally be looser
    than what you'd trust from a single fit.
    """
    rng = rng if rng is not None else np.random.default_rng()
    subset_size = max(min_subset, int(round(p * subsample_frac)))
    subset_size = min(subset_size, p)
    candidate_count = np.zeros(p)
    selected_count = np.zeros(p)
    for _ in range(n_subsets):
        subset = rng.choice(p, size=subset_size, replace=False)
        local_selected = fit_selector(subset)
        candidate_count[subset] += 1
        if len(local_selected):
            selected_count[subset[np.asarray(local_selected, dtype=int)]] += 1
    stability = np.divide(selected_count, candidate_count,
                           out=np.zeros(p), where=candidate_count > 0)
    return StabilityResult(stability=stability, candidate_count=candidate_count,
                            n_subsets=n_subsets, subsample_frac=subsample_frac)


def admit(result: StabilityResult, threshold):
    """Feature indices with stability strictly above `threshold`."""
    return np.where(result.stability > threshold)[0]


def stability_select_complementary_pairs(fit_selector, p, n_pairs=50, rng=None):
    """Complementary-pairs variant of stability_select (Shah & Samworth 2013, "Variable
    selection with error control: another look at stability selection"). Instead of
    stability_select's independent random subsets (size p*subsample_frac, drawn without
    regard to each other), each of `n_pairs` draws PARTITIONS the p candidates into two
    disjoint halves A and A^c (A union A^c = every candidate, A intersect A^c = empty)
    and fits the selector on both. Every feature is therefore a candidate on EVERY
    draw (falling in exactly one of the two halves), unlike stability_select where
    candidacy itself is random per subset -- and the two halves' selection outcomes are
    negatively associated by construction (they share no features), which is the
    structural property the tighter finite-sample error-control bound in Shah &
    Samworth relies on, under a weaker condition (their "r-concordance") than
    Meinshausen & Buhlmann (2010)'s full-exchangeability assumption that
    stability_select inherits. Same fit_selector(subset_idx) -> local-indices contract
    as stability_select; `admit` works on the returned StabilityResult unchanged.

    Fixed subsample_frac=0.5 (a partition, not a tunable fraction -- that's the whole
    mechanism). `n_pairs` draws = 2*n_pairs total selector fits, directly comparable to
    stability_select's `n_subsets`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    half = p // 2
    candidate_count = np.zeros(p)
    selected_count = np.zeros(p)
    for _ in range(n_pairs):
        perm = rng.permutation(p)
        for subset in (perm[:half], perm[half:]):
            local_selected = fit_selector(subset)
            candidate_count[subset] += 1
            if len(local_selected):
                selected_count[subset[np.asarray(local_selected, dtype=int)]] += 1
    stability = np.divide(selected_count, candidate_count,
                           out=np.zeros(p), where=candidate_count > 0)
    return StabilityResult(stability=stability, candidate_count=candidate_count,
                            n_subsets=2 * n_pairs, subsample_frac=half / p)
