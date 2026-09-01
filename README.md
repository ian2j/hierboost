# hierboost — hierarchical proximity-boosted spike-and-slab, generalized

A Python implementation of Ian Johnston's PhD work on hierarchical Bayesian
GWAS models — the *Spatial Boost* paper (Johnston, Hancock, Mamitsuka &
Carvalho, *Gene-Proximity Models for Genome-Wide Association Studies*,
arXiv:1311.0431) and Chapter 4 of the dissertation it's drawn from
(*Hierarchical Bayesian Models for Genome-Wide Association Studies*, Boston
University, 2015) — plus a generalization of both past genomics, and three
concrete modernizations of the dissertation's weakest points.

The core idea, stripped of genomics: fit a spike-and-slab GLM where a
feature's prior probability of inclusion is boosted by its affinity to
higher-level groups, weighted by each group's relevance to the outcome
(SNPs/genes in GWAS; words/topics, sensors/zones, or any features with a
notion of proximity to annotated regions, in general). Correlated raw
features can optionally be pre-processed into decorrelated group-level
latents (Chapter 4's contribution) before that same spike-and-slab engine
runs on them.

## Paper in progress

`paper/` contains a LaTeX draft (`main.tex`) launching the block-latent model
on real 1000 Genomes Phase 3 data: ~10 well-characterized loci
(`genomics_1kg_locus.py` drives each one -- fetch, LD-block, CV against
standard baselines + SuSiE, full-data fit, JSON summary), benchmarked against
SuSiE (Wang et al. 2020), the current standard for Bayesian fine-mapping.
Per-locus results live in `results/*.json`; `paper/build_results_table.py`
regenerates `paper/results_table.tex` from them (never hand-edit the table).

## Quickstart: the unified API

`HierBoostClassifier` (Bernoulli/Binomial outcomes), `HierBoostRegressor` (continuous
outcomes), and `HierBoostCountRegressor` (Poisson/Negative-Binomial event counts) wrap
every piece below into one sklearn/statsmodels-shaped object: `fit(X, y)`,
`predict(X_new)` with uncertainty bands, `.summary()`, `.plot_*()`. This is the
recommended entry point for fitting these models to a new dataset; the lower-level
functional API in the next section is for manual control over any one stage.

```python
from hierboost import HierBoostClassifier, HierBoostRegressor

# Plain (non-boosted) spike-and-slab -- always available, no coords/groups needed:
clf = HierBoostClassifier(fit_method="em")           # or "em_filter", "gibbs"
clf.fit(X, y)
print(clf.summary())                                  # statsmodels-style report
pred = clf.predict_proba(X_new, return_std=True)      # PredictionResult(mean, std, lower, upper)
clf.plot_inclusion(); clf.plot_coefficients(); clf.plot_diagnostics()

# + boosting-prior structure (Ch2/5): groups -> affinity -> relevance -> prior boost.
# Bandwidth auto-fits (fit_gaussian_bandwidth) against |corr(X)| if not given.
clf.fit(X, y, coords=positions, group_l=gene_l, group_r=gene_r, group_relevance=gene_relevance)

# + block-latent decorrelation (Ch4): blocks either supplied or data-driven
# (block_method="threshold"|"correlation"|"graphical_lasso"). decorrelate="sar" for
# spatially/identity-adjacent blocks, "ar1" for temporally-adjacent ones (lags, rolling
# windows), "star" for both jointly -- one VAR(1) latent trajectory per block, all
# blocks coupled through a single SAR-weighted transition matrix instead of independent
# AR(1)s (hierboost.spacetime). Continuous/count outcomes fit closed-form/IRLS, no JAX
# (factor.py/state_space.py/spacetime.py); a Binomial outcome needs JAX (hierboost.latent,
# "sar"/"ar1" only) -- run that specific combination from .venv-jax.
reg = HierBoostRegressor(decorrelate="ar1", block_method="threshold")
reg.fit(X, y, coords=lag_index)          # block_id auto-determined if not given
reg.predict(X_new, return_std=True)       # applies the *fitted* latent transform to new data
reg.plot_latent(block=reg.block_ids_[0])  # the smoothed AR(1) trajectory, +-2 SD band

# Practitioner extras, all on every HierBoost* instance:
reg.describe_blocks()                     # per-block raw member names + theta_hat + coef
reg.save("model.pkl"); HierBoostRegressor.load("model.pkl")   # persistence (pickle)
from hierboost.model_selection import cross_val_score, cross_val_predict
cross_val_score(reg, X, y, cv=5, coords=lag_index)             # purpose-built CV (see below)
```

See `test_estimator.py` (main env) and `test_hierboost_jax.py`'s
`test_estimator_classifier_ar1_decorrelate_predicts_held_out` (`.venv-jax`) for more
complete worked examples of every `decorrelate`/`fit_method` combination.

`hierboost_tutorial.ipynb` is a self-contained, main-env-only walkthrough (fit → read
`.summary()` → plot → predict with uncertainty → compare against baselines) on a real,
built-in dataset (sklearn's Breast Cancer Wisconsin) -- the place to start if you're new
to the package; the domain-specific demos below assume you've already seen this.

### Cross-validation

`hierboost.model_selection.cross_val_score`/`cross_val_predict` exist because sklearn's
own versions don't fit this API: `.fit()` takes extra feature-level metadata (`coords`,
`block_id`, `group_*`) that must stay fixed across folds rather than be split, `offset`
(count models) is per-observation and DOES need splitting, and `.predict()` deliberately
returns a natural-scale mean (a probability or an expected count), not a hard label --
so sklearn's default classifier scorer would silently score nonsense if plugged in
directly. `HierBoost*.get_params()`/`.set_params()` follow sklearn's convention (so
`sklearn.base.clone()` also works), but there's no `.score()`/Classifier-Regressor-Mixin
for the reasons above -- use `hierboost.model_selection.cross_val_score` instead of
sklearn's `cross_val_score`/`GridSearchCV` for this package's estimators.

## Layout

**`hierboost/`** — the generic core. Nothing here mentions genes or SNPs.

- `kernels.py` — feature↔group affinity: `gaussian_affinity_1d` (the paper's
  gene-weight kernel, generalized to any 1D coordinate), `gaussian_affinity_points`
  (point/centroid groups in any dimension — embeddings, lat/lon), `graph_affinity`
  (kernel-on-graph-distance), `sar_weight_matrix` (Ch4's SAR spatial weights),
  `fit_gaussian_bandwidth` (Sec 5.1's LD-decay bandwidth fit, generalized),
  `causal_affinity_1d` (**temporal**: a one-sided decay kernel — a feature may only be
  boosted by a group that has already resolved by its own timestamp, so a spatial
  affinity's symmetry doesn't leak future information into a backtest's prior),
  `ar1_weight_matrix` (**temporal**: `sar_weight_matrix`'s directed counterpart, an
  Ornstein-Uhlenbeck/AR(1) correlation over a time index instead of a symmetric spatial
  weight).
- `blocks.py` — `merge_overlapping_extents` (non-overlapping group partition,
  Sec 3.1), `threshold_blocks_1d`/`threshold_blocks_graph` (Ch4 Sec 4.1.1's
  distance-threshold blocking, generalized to graphs).
- `spike_slab.py` — the domain-agnostic inference engine: EM algorithm,
  EM-filtering pipeline, Pólya-Gamma Gibbs sampler, centroid estimator,
  EMBFDR-based κ selection.
- `spike_slab_gaussian.py` — the fully-conjugate Gaussian-response counterpart (no
  Pólya-Gamma/IRLS needed): closed-form EM, EM-filtering, and Gibbs.
- `spike_slab_glm.py` — Poisson/Negative-Binomial spike-and-slab for count outcomes
  (page views, case counts, species observations, ...): canonical/near-canonical
  log-link IRLS reduces to the same weighted-normal-equations M-step the other two
  response modules use. `fit_em_poisson`/`em_filter_poisson`; `fit_em_nb`/`em_filter_nb`
  plus `gibbs_sampler_nb` (Pólya-Gamma augmentation via the NB-as-logistic identity,
  generalizing `spike_slab.py`'s Gibbs sampler — dispersion `r` is a per-sweep
  profile-MLE plug-in, not a full posterior draw, since it has no simple conjugate full
  conditional). Poisson has no known exact data-augmentation scheme, so it stays
  EM/EM-filter-only. `.fit(..., offset=log_population)` supports rate models.
- `rank_utils.py` — rank-truncated SVD/Woodbury linear algebra (Eq. 8) so the
  above stays fast at p in the tens of thousands.
- `latent.py` — **Chapter 4, generalized and modernized**: the SAR block-wise
  latent feature model, fit with JAX autodiff (`grad`/`hessian`) instead of
  the dissertation's hand-derived Newton/Taylor-expansion formulas (Eqs.
  4.9–4.16, 4.19–4.23), and reparameterized-Monte-Carlo moment matching
  instead of 2nd/3rd-order Taylor truncation. Also takes `structure="ar1"` —
  **the temporal analogue**: Chapter 4's `z = Bz + eps` mechanism with a
  *symmetric* B says "each block member blends with every nearby member"
  (right for undirected spatial proximity); swapping in a *directed*,
  one-step-back B (decaying like an Ornstein-Uhlenbeck process over the gap to
  the previous member in time order) turns the exact same reparameterization
  and Newton machinery into a discrete/binomial temporal block-latent model —
  e.g. several lags of one signal sharing one latent per individual instead of
  several LD-correlated SNPs. *Requires JAX.*
- `state_space.py` — **the continuous-observation counterpart of `latent.py`'s
  temporal option**, mirroring how `factor.py` relates to `latent.py`'s SAR
  case: a one-factor dynamic-state model (`x_t = loadings·z_t + eps_t`,
  `z_t = rho·z_{t-1} + eta_t`) fit by EM with a closed-form scalar Kalman
  filter/RTS smoother (Shumway & Stoffer) — fully conjugate, no autodiff
  needed, but (unlike `factor.py`'s one-shot SVD) still an EM loop since the
  temporal dependence `rho` itself has to be estimated.
- `spacetime.py` — **`decorrelate="star"`**: `factor.py`'s spatial idea and
  `state_space.py`'s temporal one combined into one Kronecker-structured space×time
  block-latent model, per the dissertation's own deferred "eventually" aside. Instead of
  K independent AR(1) trajectories (one per block), all K blocks' latent states
  `z_t ∈ R^K` evolve jointly under one shared, *structurally constrained* transition
  `Phi = rho1·I + rho2·W` (W = `sar_weight_matrix` on the blocks' spatial centroids) —
  `rho1` is each block's own persistence, `rho2` is how much a spatially nearby block's
  past state leaks into this block's present. Two scalars, not an unconstrained K×K
  VAR(1) matrix, so the (rho1, rho2) M-step reduces to a closed-form 2×2 linear system.
  Fit via a vector-state Kalman filter/RTS smoother, processing each raw member as a
  scalar measurement (generalizing `state_space.py`'s scalar update to a K-dim rank-1
  update) — O(T·M·K²), not O(T·M³). With one block, W is the 1×1 zero matrix, rho2 drops
  out, and this reduces exactly to `state_space.py`'s scalar AR(1) fit. Fully conjugate,
  no JAX.
- `copula.py` — **the Gaussian-copula generalization of `factor.py`/`state_space.py`/
  `spacetime.py`'s shared assumption that each raw feature is already roughly Gaussian**
  (`factor.py`'s own docstring: "ideally already standardized"). Maps each block member
  through its own fitted marginal (empirical/rank-based by default — the "nonparanormal"
  of Liu, Lafferty & Wasserman 2009 — or a named `scipy.stats` family, with a
  jittered/continuized CDF for discrete count families per Denuit & Lambert 2005) onto a
  shared Gaussian scale *before* the existing closed-form factor/state-space/spacetime
  machinery runs — those modules themselves are untouched, only the marginal-to-latent
  step changes. Targets blocks whose members are correlated but have wildly different
  marginal shapes (e.g. a station's own precipitation total, mm, next to its wet-day
  count). Wired into `estimator.py` as `marginal="copula"` (continuous branch only, i.e.
  every response family except Binomial, which routes through Chapter 4's discrete/JAX
  path and assumes the raw features are Binomial already). Validated first on a
  controlled synthetic case (`copula_synthetic_validation.py`: recovers the true shared
  latent factor markedly better than raw `gaussian_block_factor` as marginal distortion
  gets more severe, +0.24 to +0.45 correlation-with-truth), then on two real datasets —
  `finance_copula_demo.py` (AAPL vs. other ETFs' own {return, dollar volume} activity
  pairs, motivated by the volume-volatility relation/Clark 1973 MDH: a real economic
  story, but neither raw nor copula generalizes to held-out AAPL return/volatility here —
  an honest negative result, though copula consistently overfits less) and
  `uk_weather_copula_demo.py` (reusing the one domain already confirmed to carry real
  spatial signal — see project memory — extended with block-latent decorrelation for the
  first time: each station's {wet-day count, mm total} pair, predicting a held-out
  target station's count. Clean positive result: held-out Poisson deviance 0.43 (copula)
  vs. 0.61 (raw) vs. 0.92 (naive), corr(pred, actual) 0.76 vs. 0.62, and the top
  copula-favored stations are dominated by the physically nearest ones), and
  `genomics_1kg_copula_demo.py` (back on hierboost's own origin domain, 1000 Genomes LD
  blocks — LCT/SLC24A5/DARC, all three already-fetched real loci. A clean, theoretically
  interesting NEGATIVE result: copula, both the empirical/rank variant and a new
  `fit_parametric_marginal(..., "binom")` variant added for this test, slightly hurts CV
  accuracy on all three loci despite confirmed real within-block MAF heterogeneity —
  because genomic LD (r²) is *by definition* the Pearson correlation of raw dosage
  vectors, unlike finance's/weather's raw units, which are arbitrary conventions
  standing in for a latent quantity copula-Gaussianizing gets closer to. Copula helps
  only when the raw scale is an incidental measurement unit, not when it already *is*
  the scientifically-defined quantity the correlation is about — see project memory for
  the fuller writeup).
- `calibration.py` — **simulation-based coverage/calibration checks**, response- and
  decorrelate-agnostic (`check_ci_coverage`, `check_theta_calibration`): repeatedly
  simulate from a known ground truth, fit, and check whether the reported credible
  intervals/`theta_hat` are actually honest — a rigor check on a claim the practitioner
  API already makes, not a new modeling capability. See `calibration_check.py` for the
  empirical study: the `decorrelate=None` baseline is well-calibrated (~95% coverage,
  ECE≈0.02), but `decorrelate="sar"` collapses to ~60% causal-effect coverage — the
  two-stage plug-in pipeline's stage-1 latent-estimation uncertainty never propagates
  into stage-2's reported SE (classic errors-in-variables/attenuation bias), exactly the
  limitation the dissertation itself flags as the reason `joint.py` exists, now with a
  number attached. Practical implication: every `decorrelate`-based demo in this project
  has plausibly-overconfident block-level CIs; point estimates/selection look sound, but
  the reported uncertainty bands shouldn't be trusted until `joint.py` is itself checked
  to restore coverage (untested so far). See project memory for the full write-up.
- **Coverage fix (2026-08-29)**: calibration testing found `decorrelate="sar"`'s reported
  credible intervals badly under-cover a real causal block's true effect (~55-63% against
  a 95% nominal target) — the classical "generated regressors" problem (Pagan 1984):
  a block's raw features estimate the factor loadings AND (via the fitted score) the
  outcome coefficient, and the naive Laplace SE treats the fitted score as if it were the
  true, known latent, ignoring its own estimation uncertainty entirely. Two fixes, in
  order of completeness: (1) `factor.py`'s `gaussian_block_factor(..., return_variance=
  True)` now also returns the closed-form posterior variance of the factor score given
  the (fixed) loadings — `estimator.py` uses it to inflate the Gaussian response's
  residual variance (`_compute_approx_covariance`), a real but only PARTIAL fix (~63%
  coverage) since it doesn't capture the loadings' own estimation error; (2) the complete
  fix, `_HierBoostBase.bootstrap_ci()` — refits the whole pipeline (decorrelation +
  spike-and-slab) on each resample of individuals, capturing every source of two-stage
  uncertainty at once, general across every `decorrelate`/response combination since it
  doesn't need a bespoke analytic correction per branch. Confirmed via calibration
  testing: causal coverage 91.7% (n=120 trials, `decorrelate="sar"`, Gaussian) — close to
  the 95% nominal target and a large improvement over both the naive (55-63%) and
  closed-form-corrected (63%) Laplace SE. Recommended whenever `decorrelate` is set and
  the reported uncertainty needs to be trusted, at the cost of `n_boot` full refits
  instead of one. Tested on the discrete/Binomial branch (`hierboost.latent`) too —
  does NOT cleanly fix it (~50% coverage over 8 replications, interval centers scattered
  wildly, std 0.67 around a true value of 0.43) — bootstrap faithfully reproduces a real,
  separate, already-documented instability (`newton_update_ztilde`'s quasi-separation
  problem) rather than fixing it; bootstrap only correctly quantifies uncertainty for an
  estimator that's itself stable. `ar1`/`star` (whose posterior variances are
  per-timestep/heteroscedastic, not yet wired into the closed-form correction) are
  natural next steps, not attempted this session. See project memory for the full
  multi-day diagnosis (including two analytically-principled fixes to
  `hierboost/joint.py`'s regularized horseshoe prior that, unlike this one, did NOT
  close the gap).
- `joint.py` — **modernization #2**: joint NUTS inference over the block
  latents *and* the outcome model together (via NumPyro), replacing the
  two-stage plug-in pipeline (fit latents by Newton's method, then treat them
  as fixed data) that the dissertation flags as a known limitation
  ("a fully Bayesian approach is impractical..."). Uses a regularized
  horseshoe prior in place of discrete spike-and-slab, since discrete
  indicators don't mix under HMC. *Requires JAX + NumPyro.*
- `structure.py` — **modernization #3a**: data-driven block structure
  (correlation-threshold or graphical-lasso connected components) instead of
  a fixed distance cutoff ζ.
- `relevance.py` — **modernization #3b**: a swappable `RelevanceSource`
  interface (`TextEmbeddingRelevance`) replacing hand-curated relevance
  lookups (e.g. MalaCards) — the slot where a production system plugs in an
  LLM/embedding API; ships with an offline TF-IDF stand-in for demos.
- `estimator.py` — **the unified practitioner API**: `HierBoostClassifier`/
  `HierBoostRegressor`/`HierBoostCountRegressor`, composing boosting-prior affinity,
  optional block-latent decorrelation (data-driven or supplied blocks, `"sar"`/`"ar1"`/
  `"star"`, optionally `marginal="copula"` — see `copula.py`), and the spike-and-slab GLM
  fit (em/em_filter/gibbs) into one
  `fit`/`predict`/`summary`/`plot_*` object, plus `.describe_blocks()` (per-block raw
  member names alongside inclusion probability/coefficient — the block-composition view
  that isn't otherwise visible without going back to `block_membership_lists` directly),
  `.save()`/`.load()` (pickle-based persistence), and `.get_params()`/`.set_params()`
  (sklearn-clone-compatible, walking the full MRO's `__init__` signatures so
  `HierBoostCountRegressor`'s `family`/`r_init` on top of the shared params round-trip
  too). New-data prediction reuses each module's *fitted* parameters via
  `project_block_factor` (SAR/Gaussian), `filter_temporal_block_factor` (AR1/Gaussian, a
  fixed-parameter Kalman pass), `filter_spacetime_block_factor` (STAR, the vector-state
  counterpart), and `BlockLatentFit.infer_ztilde_from_data` (SAR/AR1 discrete, a
  data-only Newton update since held-out individuals have no known y). `decorrelate`
  is response-family-agnostic wherever it only transforms raw X (Gaussian, Poisson, and
  Negative-Binomial all share the exact same `"sar"`/`"ar1"`/`"star"` code path) — only a
  Binomial response routes to Chapter 4's literal discrete/JAX machinery instead,
  because there the *raw features themselves* (not just the response) are assumed
  Binomial (e.g. genotypes); that path supports `"sar"`/`"ar1"` only, not yet `"star"`.
- `viz.py` — standalone plotting functions (`plot_inclusion`, `plot_coefficients`,
  `plot_diagnostics`, `plot_temporal_latent`, `plot_loadings`) behind `estimator.py`'s
  thin `.plot_*()` wrappers; matplotlib is only imported here.
- `model_selection.py` — `cross_val_score`/`cross_val_predict`, a K-fold CV loop
  purpose-built for `HierBoost*`'s `fit(X, y, coords=..., block_id=..., offset=...)`
  signature and natural-scale-mean `predict()` (see the Quickstart section above for
  why sklearn's own `cross_val_score` doesn't fit this API directly).

**`spatial_boost/`** — the original GWAS paper, now a thin wrapper over `hierboost`.

- `weights.py` — gene weights and φ-selection, calling into `hierboost.kernels`/`blocks`.
- `model.py` — re-exports `hierboost.spike_slab` for backward compatibility.
- `simulate.py` — LD-correlated genotype simulator (AR(1)-in-space latent
  Gaussian), gene annotations, gene-proximity-weighted causal markers.

**Demos & tests**

- `demo.py` — GWAS Monte Carlo comparison + detailed walkthrough with plots in `figures/`.
- `demo_nongenomic.py` — **proves the generalization**: the same `hierboost`
  core applied to a 2D sensor network (point groups, not genomic intervals) —
  zero genomics code involved.
- `benchmark_efficiency.py` — dense vs rank-truncated timing at p=20,000 (~360x speedup).
- `finance_demo.py`, `finance_factor_selection.py`, `finance_cross_sectional.py` —
  Chapter 4's *spatial* block-latent idea on real ETF/stock data (see below).
- `temporal_demo.py` — the **temporal** block-latent idea (`state_space.py`) on real
  market data: several rolling-window return features of one stock, collinear for the
  same underlying reason LD-correlated SNPs are, decorrelated into one shared,
  AR(1)-smoothed "trend state" instead of one static factor per block.
- `test_spatial_boost.py`, `test_hierboost.py` — sanity checks, run in the main env
  (includes `causal_affinity_1d`/`ar1_weight_matrix`/`state_space.py`/`spacetime.py`,
  the last with a K=1-degenerates-to-scalar-AR(1) consistency check and a 3-block
  synthetic spatial-coupling recovery test).
- `test_spike_slab_glm.py` — Poisson/NB EM, EM-filtering, offsets, and `gibbs_sampler_nb`
  (coefficient + dispersion recovery). Main env.
- `test_estimator.py` — `HierBoostClassifier`/`Regressor`/`CountRegressor`:
  no-decorrelate, `sar`/`ar1`/`star` decorrelate (Gaussian and count response alike),
  `em`/`em_filter`/`gibbs` fit methods, out-of-sample `predict`, `.summary()`,
  `.describe_blocks()`, `.save()`/`.load()`, `.get_params()`/clone, `cross_val_score`/
  `cross_val_predict`, and plotting smoke tests. Main env.
- `test_hierboost_jax.py` — sanity checks for `latent.py` (SAR *and* `structure="ar1"`),
  `joint.py`, and `HierBoostClassifier`'s binomial `decorrelate` path, run in `.venv-jax`.

## Installation & the functional API (manual control)

`pip install -e .` (see `pyproject.toml`) installs `hierboost`/`spatial_boost` with the
core dependencies (numpy&lt;2, scipy, scikit-learn, matplotlib, polyagamma). The
`jax` extra (`pip install -e ".[jax]"`) is for `latent.py`/`joint.py` specifically and
pulls in numpy≥2 — **install it into a separate virtualenv** (`.venv-jax` below), never
alongside the base install, since it will break scikit-learn/scipy. The `demos` extra
(`pandas`, `yfinance`, `statsmodels`) is only needed to run the demo scripts, not the
package itself.

Everything below is the lower-level functional API `estimator.py` is built from, for
when you want manual control over one stage (a custom affinity kernel, a hand-fit block
structure, direct access to the Gibbs sampler's raw draws, ...) instead of going through
`HierBoostClassifier`/`HierBoostRegressor`.

```bash
# main environment (GWAS core, hierboost core, structure/relevance, estimator/viz)
pip install -e .
python3 test_spatial_boost.py
python3 test_hierboost.py
python3 test_spike_slab_glm.py
python3 test_estimator.py
python3 demo.py
python3 demo_nongenomic.py
python3 benchmark_efficiency.py     # slow (~5 min): dense vs rank-truncated

# isolated environment (Chapter 4 latent model + joint inference + the
# decorrelate="sar"/"ar1" binomial path of HierBoostClassifier)
python3 -m virtualenv .venv-jax
source .venv-jax/bin/activate
pip install -e ".[jax]"
python3 test_hierboost_jax.py
```

```python
# Generic core, on any hierarchical feature-selection problem:
from hierboost import gaussian_affinity_points, combine_affinity_with_relevance, em_filter, gibbs_sampler

affinity = gaussian_affinity_points(feature_coords, group_coords, bandwidth=8.0)
wr = combine_affinity_with_relevance(affinity, group_relevance)
filt = em_filter(X, y, wr, xi0=-3.0, xi1=3.0, kappa=100.0, nu=1.0, lam=1.0, rank=200)

# Chapter 4 + joint inference (in .venv-jax):
from hierboost.latent import fit_latent_block_model
from hierboost.joint import run_joint_inference, posterior_association_summary

result = fit_latent_block_model(X, y, coords, block_id, wr, xi0=-3, xi1=3, kappa=100, nu=1, lam=1)
mcmc, block_ids = run_joint_inference(X, y, block_id, result["fits"])
summary = posterior_association_summary(mcmc, block_ids)  # posterior P(association) + credible intervals

# Temporal: same Chapter 4 machinery, directed instead of symmetric block structure
result_t = fit_latent_block_model(X, y, lag_order, block_id, wr, xi0=-3, xi1=3, kappa=100,
                                   nu=1, lam=1, structure="ar1")

# Temporal, continuous outcome (main env, no JAX): closed-form EM + Kalman/RTS smoother
from hierboost.state_space import fit_temporal_block_factor

result_cont = fit_temporal_block_factor(X_block)   # (T, m) -> smoothed trend + loadings + rho
```

## What the demos actually show

**GWAS Monte Carlo comparison** (`demo.py`, 15 replicates, p=1200, n=60, m_causal=8):

|                  | single-SNP | SB, no boost | SB, gene boost |
|------------------|-----------:|-------------:|---------------:|
| informative scen.|      0.578 |         0.573|         **0.703** |
| non-informative  |      0.563 |         0.561|         0.529 |

Reproduces the paper's own finding (Sec 6.2/8): the gene-proximity prior
helps *only* when the gene relevances/φ are actually informative — with a
misspecified prior it can slightly *hurt*, spending prior mass boosting the
wrong markers. The model's edge is in ranking (AUC), not loud hard calls: the
centroid estimator selects very few markers at conventional γ.

**Non-genomic generalization** (`demo_nongenomic.py`, 2D sensor network): a
marginal correlation test gets AUC 0.51 (chance), joint spike-and-slab without
the zone boost gets 0.63, and *with* the zone-proximity boost gets **0.89** —
using the identical `hierboost` machinery as the GWAS demo, just with
`gaussian_affinity_points` (2D centroids) instead of `gaussian_affinity_1d`
(1D genomic intervals).

**Chapter 4 modernized** (`test_hierboost_jax.py`): on a synthetic
LD-correlated block dataset with one causal block, the autodiff-fit latent
block model assigns the causal block θ̂ ≈ 1.0 (vs ≈0.04 for the others) and
recovers a block-level latent correlating 0.5–0.8 with the true underlying
signal — from a JAX-differentiated objective, not five pages of hand
calculus. The joint NUTS model agrees on *which* block matters (posterior
P(association) ≈ 1.0, well-mixed r-hat on the reparameterized coefficients)
while also reporting a wide credible interval on the effect's *magnitude* —
uncertainty a point-estimate EM pipeline has no way to express at all, which
is exactly the gap Chapter 4 leaves open.
