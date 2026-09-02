# tests

Run with `pytest tests/` (main env). One file needs the separate JAX env:
`.venv-jax/bin/pytest tests/test_hierboost_jax.py`.

| File | Covers |
|---|---|
| `test_hierboost.py` | Core spike-and-slab engine (EM, EM-filter, Gibbs). |
| `test_estimator.py` | The `HierBoost*` classifier/regressor API end to end. |
| `test_copula.py` | Gaussian-copula marginal transforms. |
| `test_calibration.py` | CI/coverage calibration checks. |
| `test_spike_slab_glm.py` | Poisson/Negative-Binomial spike-and-slab. |
| `test_sumstats.py` | Summary-statistics (sumstats-only) fitting path. |
| `test_hierboost_jax.py` | JAX-dependent block-latent model — needs `.venv-jax`. |

These exercise `../hierboost/`, the library itself.
