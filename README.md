# hierboost

Spike-and-slab feature selection where a feature's prior odds of being kept
can be boosted by how close it is to a relevant group (genes, sensors, time
lags, ...), and correlated features can optionally be collapsed into one
shared latent per group before selection runs. Generalized from Ian
Johnston's PhD work on GWAS models (`paper/`, arXiv:1311.0431).

## Install

```bash
pip install -e .
```

Needs `numpy<2` (see **Environments** below for the JAX-only parts).

## Quickstart

```python
from hierboost import HierBoostClassifier, HierBoostRegressor

clf = HierBoostClassifier(fit_method="em")
clf.fit(X, y)
clf.summary()                       # text report
clf.predict_proba(X_new, return_std=True)
clf.plot_inclusion()

# with a proximity-boosted prior + block-latent decorrelation:
reg = HierBoostRegressor(decorrelate="ar1", block_method="threshold")
reg.fit(X, y, coords=lag_index)
reg.predict(X_new, return_std=True)
```

Start with `hierboost_tutorial.ipynb` — a runnable walkthrough on a
built-in dataset. `tests/test_estimator.py` has a worked example of every
option combination.

## Layout

| Path | What's there |
|---|---|
| `hierboost/` | The library. One module per concern (kernels, blocking, spike-and-slab fit, block-latent decorrelation, the `HierBoost*` estimator API). Each file's docstring explains what it does. |
| `spatial_boost/` | Thin GWAS-specific wrapper kept for backward compatibility. |
| `demos/` | Runnable examples of the framework on different domains. |
| `applications/` | Deeper per-domain analyses (`genomics/`, `predom/`, `riemann/`). |
| `validation/` | Calibration/coverage/efficiency studies — not part of the test suite. |
| `tests/` | The pytest suite — see [`tests/README.md`](tests/README.md). |
| `paper/` | LaTeX draft benchmarking the block-latent model against SuSiE on real 1000 Genomes data. |

Everything else at the top level (`earthquake_japan/`, `m6_backtest/`,
`predom_*/`, `state_econ/`, `streamflow_delaware/`, `uk_weather/`) is a
self-contained domain folder: its own fetch/build/demo scripts plus a
`data/` cache.

## Environments

- **Main** (`pip install -e .`): everything except the two items below.
- **`.venv-jax`** (`pip install -e ".[jax]"`, separate virtualenv — needs
  numpy≥2, which conflicts with the main env): `hierboost.latent`,
  `hierboost.joint`, and `HierBoostClassifier(decorrelate=...)` on a
  Binomial response.
- **`.venv-genomics`**: only for fetching 1000 Genomes data (`pysam`).

Run the suite: `pytest tests/` (main env) and, separately,
`.venv-jax/bin/pytest tests/test_hierboost_jax.py`.
