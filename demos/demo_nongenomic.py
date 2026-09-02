"""Proves hierboost isn't genomics-specific: a 2D sensor-network anomaly-detection task
using the same generic core as the GWAS demo, with point-like sensors/zones
(gaussian_affinity_points) instead of 1D genes/SNPs. No genomics code involved."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from hierboost.kernels import gaussian_affinity_points, combine_affinity_with_relevance
from hierboost.spike_slab import em_filter, gibbs_sampler, centroid_estimate

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)


def simulate_sensor_field(n=150, p=200, n_zones=12, m_causal=6, xi1=4.0, seed=0):
    rng = np.random.default_rng(seed)

    sensor_xy = rng.uniform(0, 100, size=(p, 2))
    zone_xy = rng.uniform(0, 100, size=(n_zones, 2))
    zone_relevance = rng.lognormal(mean=0.0, sigma=1.2, size=n_zones)  # a few "hot" zones

    bandwidth = 8.0
    affinity = gaussian_affinity_points(sensor_xy, zone_xy, bandwidth)
    wr = combine_affinity_with_relevance(affinity, zone_relevance, normalize=True)

    # spatially correlated continuous sensor readings via an RBF-kernel Gaussian field
    d2 = ((sensor_xy[:, None, :] - sensor_xy[None, :, :]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * 15.0 ** 2)) + 1e-6 * np.eye(p)
    L = np.linalg.cholesky(K)
    Xraw = (L @ rng.standard_normal((p, n))).T  # n x p, spatially smooth per-sample field

    xi0 = np.log(m_causal / p) - np.log(1 - m_causal / p)
    prior_prob = np.clip(expit(xi0 + xi1 * wr), 1e-8, 1 - 1e-8)
    causal_idx = rng.choice(p, size=m_causal, replace=False, p=prior_prob / prior_prob.sum())
    theta_true = np.zeros(p, dtype=bool)
    theta_true[causal_idx] = True

    beta_features = np.where(theta_true, rng.normal(0, 1.2, p), rng.normal(0, 0.05, p))
    beta_true = np.concatenate([[0.0], beta_features])
    X = np.column_stack([np.ones(n), Xraw])
    y = (rng.random(n) < expit(X @ beta_true)).astype(float)

    return dict(X=X, y=y, sensor_xy=sensor_xy, zone_xy=zone_xy, zone_relevance=zone_relevance,
                wr=wr, theta_true=theta_true, xi0=xi0)


def marginal_corr_auc(X_features, y, theta_true):
    Xc = X_features - X_features.mean(axis=0)
    yc = y - y.mean()
    stat = np.abs(Xc.T @ yc) / (np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12)
    return roc_auc_score(theta_true, stat)


if __name__ == "__main__":
    data = simulate_sensor_field(n=150, p=200, n_zones=12, m_causal=6, xi1=4.0, seed=0)

    auc_marginal = marginal_corr_auc(data["X"][:, 1:], data["y"], data["theta_true"])

    filt_boost = em_filter(data["X"], data["y"], data["wr"], xi0=data["xi0"], xi1=4.0,
                            kappa=100.0, nu=1.0, lam=1.0, filter_frac=0.25, min_features=10,
                            max_outer=30, rank=100)
    filt_noboost = em_filter(data["X"], data["y"], np.zeros_like(data["wr"]), xi0=data["xi0"],
                              xi1=0.0, kappa=100.0, nu=1.0, lam=1.0, filter_frac=0.25,
                              min_features=10, max_outer=30, rank=100)

    auc_first_boost = roc_auc_score(data["theta_true"], filt_boost.history[0].theta_hat)
    auc_first_noboost = roc_auc_score(data["theta_true"], filt_noboost.history[0].theta_hat)
    print(f"Marginal correlation test AUC:      {auc_marginal:.3f}")
    print(f"Spike-and-slab, no zone boost AUC:  {auc_first_noboost:.3f}")
    print(f"Spike-and-slab, WITH zone boost AUC:{auc_first_boost:.3f}")

    best = filt_boost.best
    sub_true = data["theta_true"][best.retained_idx]
    print(f"\nEM filter retained {best.n_features} sensors, "
          f"{sub_true.sum()}/{data['theta_true'].sum()} true causal sensors kept")

    X_sub = data["X"][:, np.concatenate([[0], best.retained_idx + 1])]
    wr_sub = data["wr"][best.retained_idx]
    gr = gibbs_sampler(X_sub, data["y"], wr_sub, xi0=data["xi0"], xi1=4.0, kappa=100.0,
                        nu=1.0, lam=1.0, n_samples=3000, burn_in=1000, seed=0)
    auc_gibbs = roc_auc_score(sub_true, gr.pi_hat)
    print(f"Gibbs posterior AUC on retained subset: {auc_gibbs:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, title, theta_hat_vals, idx in [
        (axes[0], "First EM step (all 200 sensors)", filt_boost.history[0].theta_hat, np.arange(200)),
    ]:
        sc = ax.scatter(data["sensor_xy"][idx, 0], data["sensor_xy"][idx, 1],
                         c=theta_hat_vals, cmap="viridis", s=40, vmin=0)
        ax.scatter(data["sensor_xy"][data["theta_true"], 0], data["sensor_xy"][data["theta_true"], 1],
                   facecolors="none", edgecolors="red", s=140, linewidths=1.5, label="truly causal")
        ax.scatter(data["zone_xy"][:, 0], data["zone_xy"][:, 1], marker="^", c="black",
                   s=data["zone_relevance"] * 40, label="zone (size ~ relevance)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="P(theta=1)")

    ax = axes[1]
    full_pi = np.zeros(200)
    full_pi[best.retained_idx] = gr.pi_hat
    sc = ax.scatter(data["sensor_xy"][:, 0], data["sensor_xy"][:, 1], c=full_pi, cmap="viridis", s=40, vmin=0)
    ax.scatter(data["sensor_xy"][data["theta_true"], 0], data["sensor_xy"][data["theta_true"], 1],
               facecolors="none", edgecolors="red", s=140, linewidths=1.5, label="truly causal")
    ax.scatter(data["zone_xy"][:, 0], data["zone_xy"][:, 1], marker="^", c="black",
               s=data["zone_relevance"] * 40, label="zone (size ~ relevance)")
    ax.set_title(f"After EM filtering + Gibbs ({best.n_features} sensors retained)")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="P(theta=1|y)")

    fig.suptitle("Same hierboost core applied to a 2D sensor network (no genomics code involved)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nongenomic_sensor_demo.png"), dpi=140)
    plt.close(fig)
    print(f"\nFigure written to {OUT}/nongenomic_sensor_demo.png")
