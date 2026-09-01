"""End-to-end demonstration of the spatial_boost package.

Two parts:
  1. A Monte Carlo comparison (many replicates) of single-SNP ranking vs. the
     Spatial Boost model with and without the gene-proximity boost, echoing the
     comparison in Sec 6.2 of the paper.
  2. A single detailed walkthrough (one replicate) showing the full pipeline --
     phi selection, EM filtering trace, kappa selection via EMBFDR, Gibbs
     sampling, and the centroid estimator -- with plots saved to ./figures.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve

from spatial_boost.simulate import simulate_dataset
from spatial_boost.weights import select_phi_by_region
from spatial_boost.model import (fit_em, em_filter, gibbs_sampler, centroid_estimate,
                                  embfdr, select_kappa_by_embfdr)

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def logit(p):
    return np.log(p) - np.log(1.0 - p)


def marginal_corr_auc(X_markers, y, theta_true):
    """Fast proxy for the paper's single-SNP (SS) test: rank markers by |corr(X_j, y)|."""
    Xc = X_markers - X_markers.mean(axis=0)
    yc = y - y.mean()
    num = Xc.T @ yc
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    denom[denom == 0] = np.inf
    stat = np.abs(num / denom)
    return roc_auc_score(theta_true, stat)


# ---------------------------------------------------------------------------
# Part 1: Monte Carlo comparison (Sec 6.2 style)
# ---------------------------------------------------------------------------
def monte_carlo_comparison(n_reps=15, n=60, p=1200, n_genes=100, m_causal=8, rank=60):
    rows = []
    for scenario in ("informative", "non_informative"):
        for rep in range(n_reps):
            data = simulate_dataset(n=n, p=p, n_genes=n_genes, scenario=scenario,
                                     m_causal=m_causal, seed=1000 * (scenario == "informative") + rep)
            xi0 = logit(m_causal / p)
            xi1 = 3.0 if scenario == "informative" else 1.0

            auc_ss = marginal_corr_auc(data.X[:, 1:], data.y, data.theta_true)

            filt_boost = em_filter(data.X, data.y, data.wr, xi0=xi0, xi1=xi1, kappa=100.0,
                                    nu=1.0, lam=1.0, filter_frac=0.25, min_features=10,
                                    max_outer=30, rank=rank)
            filt_null = em_filter(data.X, data.y, np.zeros_like(data.wr), xi0=xi0, xi1=0.0,
                                   kappa=100.0, nu=1.0, lam=1.0, filter_frac=0.25,
                                   min_features=10, max_outer=30, rank=rank)

            for name, filt in (("SB_boost", filt_boost), ("SB_noboost", filt_null)):
                first = filt.history[0]
                best = filt.best
                sub_true = data.theta_true[best.retained_idx]
                auc_first = roc_auc_score(data.theta_true, first.theta_hat)
                auc_best = roc_auc_score(sub_true, best.theta_hat) if sub_true.any() and not sub_true.all() else np.nan
                rows.append(dict(scenario=scenario, rep=rep, method=name,
                                  auc_first=auc_first, auc_best=auc_best,
                                  n_best=best.n_features))
            rows.append(dict(scenario=scenario, rep=rep, method="single_SNP",
                              auc_first=auc_ss, auc_best=np.nan, n_best=p))
    return rows


def summarize_and_plot_mc(rows):
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["scenario"], r["method"])].append(r["auc_first"])

    print("\n=== Monte Carlo comparison: AUC ranking markers at the FIRST EM step "
          "(joint model) vs marginal single-SNP test ===")
    for (scenario, method), vals in sorted(agg.items()):
        vals = np.array(vals)
        print(f"  {scenario:16s} {method:12s}  mean AUC = {vals.mean():.3f}  (sd {vals.std():.3f}, n={len(vals)})")

    agg_best = collections.defaultdict(list)
    for r in rows:
        if r["method"].startswith("SB") and not np.isnan(r["auc_best"]):
            agg_best[(r["scenario"], r["method"])].append(r["auc_best"])
    print("\n=== AUC on the retained subset at the best (min-PPL) EM filtering step ===")
    for (scenario, method), vals in sorted(agg_best.items()):
        vals = np.array(vals)
        print(f"  {scenario:16s} {method:12s}  mean AUC = {vals.mean():.3f}  (sd {vals.std():.3f}, n={len(vals)})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, scenario in zip(axes, ("informative", "non_informative")):
        data_by_method = {}
        for r in rows:
            if r["scenario"] == scenario:
                data_by_method.setdefault(r["method"], []).append(r["auc_first"])
        labels = ["single_SNP", "SB_noboost", "SB_boost"]
        ax.boxplot([data_by_method[l] for l in labels], labels=labels)
        ax.set_title(scenario)
        ax.set_ylabel("AUC" if scenario == "informative" else "")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    fig.suptitle("Marker ranking AUC: single-SNP test vs Spatial Boost (with/without gene boost)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "monte_carlo_auc.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 2: detailed single-replicate walkthrough
# ---------------------------------------------------------------------------
def detailed_walkthrough(n=200, p=400, n_genes=50, m_causal=6, causal_var=0.6,
                          noise_var=0.001, xi1=4.0, rank=None, seed=7):
    print("\n" + "=" * 70)
    print("DETAILED WALKTHROUGH (one simulated dataset)")
    data = simulate_dataset(n=n, p=p, n_genes=n_genes, scenario="informative",
                             m_causal=m_causal, causal_var=causal_var, noise_var=noise_var,
                             seed=seed)

    t0 = time.time()
    phi_hat = select_phi_by_region(data.positions, data.X[:, 1:], min_gap=30_000)
    print(f"phi selection: {time.time()-t0:.2f}s, median phi = {np.median(phi_hat):.0f} "
          f"(true phi used in simulation = {data.phi})")

    xi0 = logit(m_causal / p)
    kappas = [2, 5, 10, 50, 100, 500, 1000, 5000]
    theta_by_kappa = {}
    for k in kappas:
        r = fit_em(data.X, data.y, data.wr, xi0=xi0, xi1=xi1, kappa=k, nu=1.0, lam=1.0, rank=rank)
        theta_by_kappa[k] = r.theta_hat
    gammas = np.linspace(0.05, 20, 150)
    kappa_sel, gamma_sel, table = select_kappa_by_embfdr(theta_by_kappa, gammas,
                                                          target_bfdr=0.1, gamma_fixed=9.0)
    print(f"kappa selected by EMBFDR (largest kappa with BFDR <= 0.1 at threshold "
          f"(1+gamma)^-1={1/(1+gamma_sel):.2f}): {kappa_sel}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in kappas:
        curve = table[k]
        ax.plot(1.0 / (1.0 + gammas), curve, label=f"kappa={k}")
    ax.axhline(0.1, color="black", linestyle="--", linewidth=1, label="target BFDR")
    ax.set_xlabel("threshold  (1+gamma)^-1")
    ax.set_ylabel("EMBFDR")
    ax.set_title("EMBFDR vs threshold, by kappa (Fig 3 analogue)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "embfdr_curves.png"), dpi=140)
    plt.close(fig)

    t0 = time.time()
    filt = em_filter(data.X, data.y, data.wr, xi0=xi0, xi1=xi1, kappa=kappa_sel,
                      nu=1.0, lam=1.0, filter_frac=0.25, min_features=10, max_outer=30,
                      rank=rank, verbose=True)
    print(f"EM filtering: {len(filt.history)} steps in {time.time()-t0:.3f}s")

    steps = [h.step for h in filt.history]
    n_features = [h.n_features for h in filt.history]
    rppl = [h.rppl for h in filt.history]
    aucs = []
    for h in filt.history:
        sub_true = data.theta_true[h.retained_idx]
        aucs.append(roc_auc_score(sub_true, h.theta_hat) if sub_true.any() and not sub_true.all() else np.nan)

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(steps, rppl, "o-", color="tab:blue", label="relative PPL")
    ax1.axvline(filt.best.step, color="gray", linestyle="--", label="chosen step (min PPL)")
    ax1.set_xlabel("EM filter step")
    ax1.set_ylabel("relative PPL", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(steps, aucs, "s-", color="tab:red", label="AUC on retained subset")
    ax2.set_ylabel("AUC", color="tab:red")
    fig.suptitle("EM filtering trace (Fig 5/8 analogue)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "em_filter_trace.png"), dpi=140)
    plt.close(fig)

    best = filt.best
    X_sub = data.X[:, np.concatenate([[0], best.retained_idx + 1])]
    wr_sub = data.wr[best.retained_idx]
    theta_true_sub = data.theta_true[best.retained_idx]
    p_tilde = best.n_features
    xi0_final = logit(max(int(theta_true_sub.sum()), 1) / p_tilde)
    print(f"post-filter: p_tilde={p_tilde}, {theta_true_sub.sum()}/{m_causal} true causal markers retained")

    res_final_em = fit_em(X_sub, data.y, wr_sub, xi0=xi0_final, xi1=xi1, kappa=kappa_sel, nu=1.0, lam=1.0)

    t0 = time.time()
    gr = gibbs_sampler(X_sub, data.y, wr_sub, xi0=xi0_final, xi1=xi1, kappa=kappa_sel,
                        nu=1.0, lam=1.0, n_samples=4000, burn_in=1500,
                        beta_init=res_final_em.beta, sigma2_init=res_final_em.sigma2, seed=0)
    print(f"Gibbs sampling: {time.time()-t0:.2f}s for 4000 post-burn-in draws on p={p_tilde}")

    auc_ss_full = marginal_corr_auc(data.X[:, 1:], data.y, data.theta_true)
    auc_gibbs = roc_auc_score(theta_true_sub, gr.pi_hat) if theta_true_sub.any() and not theta_true_sub.all() else np.nan
    print(f"AUC: single-SNP test (full p) = {auc_ss_full:.3f} | "
          f"Spatial Boost EM best-step (subset) = {roc_auc_score(theta_true_sub, best.theta_hat):.3f} | "
          f"Spatial Boost Gibbs (subset) = {auc_gibbs:.3f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    fpr, tpr, _ = roc_curve(theta_true_sub, gr.pi_hat)
    ax.plot(fpr, tpr, label=f"Spatial Boost (Gibbs), AUC={auc_gibbs:.3f}")
    stat = np.abs(np.corrcoef(np.vstack([data.X[:, 1:][:, best.retained_idx].T, data.y]))[-1, :-1])
    fpr2, tpr2, _ = roc_curve(theta_true_sub, stat)
    ax.plot(fpr2, tpr2, label="single-SNP test (same subset)")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_title("ROC on retained subset")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roc_comparison.png"), dpi=140)
    plt.close(fig)

    print("\ncentroid estimator selections at various gamma (sensitivity-specificity trade-off):")
    for gamma in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        sel = centroid_estimate(gr.pi_hat, gamma=gamma)
        tp = int((sel & theta_true_sub).sum())
        fp = int((sel & ~theta_true_sub).sum())
        print(f"  gamma={gamma:5.1f}  n_selected={sel.sum():4d}  TP={tp}  FP={fp}")

    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(data.positions[best.retained_idx])
    pos_sorted = data.positions[best.retained_idx][order]
    wr_sorted = wr_sub[order]
    pi_sorted = gr.pi_hat[order]
    causal_sorted = theta_true_sub[order]
    ax.bar(pos_sorted, wr_sorted, width=(data.positions.max() - data.positions.min()) / 400,
           color="lightgray", label="gene-proximity weight w_j^T r")
    sc = ax.scatter(pos_sorted, pi_sorted, c=np.where(causal_sorted, "red", "black"), s=15,
                     label="posterior P(theta_j=1|y)")
    ax.set_xlabel("genomic position")
    ax.set_ylabel("weight / posterior probability")
    ax.set_title("Gene-proximity weights vs posterior inclusion probabilities\n(red = truly causal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "weights_vs_posterior.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    rows = monte_carlo_comparison(n_reps=15, n=60, p=1200, n_genes=100, m_causal=8, rank=60)
    summarize_and_plot_mc(rows)
    detailed_walkthrough()
    print(f"\nFigures written to {OUT}/")
