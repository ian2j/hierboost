"""Spatial Boost model fitting: thin re-export of hierboost's generic spike-and-slab
engine (Johnston et al. 2016, Sec 4). Kept as `spatial_boost.model` for backward
compatibility; new code should generally import from `hierboost` directly.
"""
from hierboost.spike_slab import (
    EMResult, FilterStep, EMFilterResult, GibbsResult,
    fit_em, em_filter, gibbs_sampler, centroid_estimate, embfdr,
    select_kappa_by_embfdr, theta_conditional, ppl, xi_bounds,
)
from hierboost.rank_utils import TruncatedDesign, woodbury_solve, woodbury_mean_and_sample, exact_mean_and_sample

__all__ = [
    "EMResult", "FilterStep", "EMFilterResult", "GibbsResult",
    "fit_em", "em_filter", "gibbs_sampler", "centroid_estimate", "embfdr",
    "select_kappa_by_embfdr", "theta_conditional", "ppl", "xi_bounds",
    "TruncatedDesign", "woodbury_solve", "woodbury_mean_and_sample", "exact_mean_and_sample",
]
