# tests

Run with `pytest tests/` (main env). One file needs the separate JAX env:
`.venv-jax/bin/pytest tests/test_hierboost_jax.py`.

| File | Covers |
|---|---|
| `test_hierboost.py` | Core spike-and-slab engine (EM, EM-filter, Gibbs). |
| `test_spatial_boost.py` | The GWAS-specific wrapper package. |
| `test_estimator.py` | The `HierBoost*` classifier/regressor API end to end. |
| `test_copula.py` | Gaussian-copula marginal transforms. |
| `test_calibration.py` | CI/coverage calibration checks. |
| `test_spike_slab_glm.py` | Poisson/Negative-Binomial spike-and-slab. |
| `test_sumstats.py` | Summary-statistics (sumstats-only) fitting path. |
| `test_hierboost_jax.py` | JAX-dependent block-latent model — needs `.venv-jax`. |

## Sibling directories

- `../demos/` — runnable examples of the framework, one domain each.
- `../applications/genomics/`, `../applications/predom/`, `../applications/riemann/` — deeper per-domain analyses.
- `../validation/` — calibration/coverage/efficiency studies (not tests).
- `../hierboost/` — the library these tests exercise.
