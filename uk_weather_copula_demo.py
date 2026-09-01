"""Second real-data test of the Gaussian-copula marginal transform, deliberately reusing
uk_weather -- the one domain this project already confirmed has genuine, generalizing
signal (see project memory: weekly best-point train r=0.806 -> test r=0.835). That prior
work only used the boosting-prior mechanism (step 1); it never exercised block-latent
DECORRELATION (step 2, factor.py) at all. This demo adds that: each station's own weekly
{wet-day COUNT, total precipitation AMOUNT in mm} pair shares an obvious real latent
("how wet was this week"), but a Poisson-ish bounded count and a continuous, right-skewed
mm total have very different marginal shapes -- structurally the same "real correlation,
wrong shared-Gaussian-marginal assumption" case as finance_copula_demo.py, on a domain
already known (unlike the finance target) to carry real, generalizing spatial signal.

Reuses uk_weather/data/weekly_wetdays.npz (wet-day counts, already built) and
precip_raw.npz (raw daily mm, aggregated here to the same weekly bins) -- no new fetch.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import mean_poisson_deviance

from hierboost.estimator import HierBoostCountRegressor

TRAIN_FRAC = 0.75


def load_weekly():
    wd = np.load("uk_weather/data/weekly_wetdays.npz")
    counts, centroids, target_k, bin_days = wd["counts"], wd["centroids"], int(wd["target_k"]), int(wd["bin_days"])
    pr = np.load("uk_weather/data/precip_raw.npz")
    precip = pr["precip"]  # (n_points, n_days)
    n_bins = counts.shape[0]
    trimmed = precip[:, :n_bins * bin_days]
    mm_totals = trimmed.reshape(precip.shape[0], n_bins, bin_days).sum(axis=2).T  # (n_bins, n_points)
    mm_totals = np.nan_to_num(mm_totals, nan=0.0)
    return counts, mm_totals, centroids, target_k


def build_design(counts, mm_totals, target_k):
    n_bins, n_points = counts.shape
    predictor_idx = [j for j in range(n_points) if j != target_k]
    y = counts[:, target_k].astype(float)

    cols, block_id = [], []
    X_parts = []
    for k, j in enumerate(predictor_idx):
        X_parts += [counts[:, j:j + 1].astype(float), mm_totals[:, j:j + 1]]
        cols += [f"pt{j}_wetcount", f"pt{j}_mm"]
        block_id += [k, k]
    X = np.column_stack(X_parts)
    return X, y, np.array(block_id, dtype=float), cols, len(predictor_idx)


if __name__ == "__main__":
    counts, mm_totals, centroids, target_k = load_weekly()
    X, y, block_id, cols, n_blocks = build_design(counts, mm_totals, target_k)
    n = len(y)
    n_train = int(n * TRAIN_FRAC)
    print(f"{n} weekly bins, {n_blocks} predictor stations (2 features each: wet-day count + mm total), "
          f"target station {target_k}, {n_train} train / {n - n_train} test (chronological)")

    print("\nSkewness of a sample of raw predictor columns (count vs mm total):")
    for c, j in list(zip(cols, range(len(cols))))[:6]:
        print(f"  {c:14s} skew={stats.skew(X[:n_train, j]):+.2f}")

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    coords = block_id

    def fit_and_eval(marginal):
        reg = HierBoostCountRegressor(family="negbinomial", decorrelate="sar", marginal=marginal,
                                       fit_method="em", xi0=-1.0, kappa=100.0)
        reg.fit(X_train, y_train, coords=coords, block_id=block_id)
        mu_test = np.clip(reg.predict(X_test), 1e-6, None)
        dev = mean_poisson_deviance(y_test, mu_test)
        corr = np.corrcoef(mu_test, y_test)[0, 1]
        return reg, dev, corr

    reg_raw, dev_raw, corr_raw = fit_and_eval(None)
    reg_cop, dev_cop, corr_cop = fit_and_eval("copula")

    naive_mu = np.full_like(y_test, y_train.mean())
    dev_naive = mean_poisson_deviance(y_test, np.clip(naive_mu, 1e-6, None))

    print(f"\nHeld-out Poisson deviance (lower is better) and corr(pred, actual):")
    print(f"  naive constant-rate                  : deviance={dev_naive:.4f}")
    print(f"  raw block-latent (marginal=None)     : deviance={dev_raw:.4f}  corr={corr_raw:.4f}")
    print(f"  copula block-latent (marginal=copula): deviance={dev_cop:.4f}  corr={corr_cop:.4f}")

    print(f"\nPosterior inclusion probability (theta_hat) per station's activity latent, "
          f"raw vs copula, top 10 by copula:")
    order = np.argsort(-reg_cop.theta_hat_)[:10]
    predictor_idx = [j for j in range(counts.shape[1]) if j != target_k]
    for i in order:
        dist = np.linalg.norm(centroids[predictor_idx[i]] - centroids[target_k])
        print(f"  pt{predictor_idx[i]:3d} (dist={dist:5.2f})  raw theta={reg_raw.theta_hat_[i]:.4f}   "
              f"copula theta={reg_cop.theta_hat_[i]:.4f}")
