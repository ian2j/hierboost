"""Plotting for hierboost.estimator's fitted models -- standalone functions taking plain
arrays (so they're usable outside the estimator too), with thin `.plot_*()` wrappers on
HierBoostClassifier/Regressor gathering the right fitted arrays and calling these.
Matplotlib is only imported here, kept out of the rest of the package.
"""
import numpy as np


def plot_inclusion(theta_hat, names=None, top_n=20, ax=None):
    """Horizontal bar chart of posterior inclusion probability theta_hat, highest first."""
    import matplotlib.pyplot as plt
    theta_hat = np.asarray(theta_hat)
    names = list(names) if names is not None else [f"x{j}" for j in range(len(theta_hat))]
    order = np.argsort(theta_hat)[::-1][:top_n]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(2.5, 0.3 * len(order))))
    y = np.arange(len(order))
    ax.barh(y, theta_hat[order][::-1], color="teal", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([names[j] for j in order][::-1], fontsize=8)
    ax.set_xlabel("posterior inclusion probability")
    ax.set_xlim(0, 1)
    ax.set_title(f"Top {len(order)} of {len(theta_hat)} by inclusion probability")
    return ax


def plot_coefficients(beta, se=None, names=None, ci=0.95, top_n=20, ax=None):
    """Bar chart of coefficient estimates (excluding intercept), ranked by |coefficient|,
    with confidence-interval error bars when `se` is given."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    beta = np.asarray(beta)
    names = list(names) if names is not None else [f"x{j}" for j in range(len(beta))]
    order = np.argsort(np.abs(beta))[::-1][:top_n]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.4 * len(order)), 5))
    x = np.arange(len(order))
    yerr = None
    if se is not None:
        z = norm.ppf((1 + ci) / 2)
        yerr = z * np.asarray(se)[order]
    ax.bar(x, beta[order], yerr=yerr, capsize=4, color="tab:red", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([names[j] for j in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("coefficient")
    title = f"Top {len(order)} coefficients by |coef|"
    if se is not None:
        title += f" (error bars = {ci:.0%} CI)"
    ax.set_title(title)
    return ax


def plot_diagnostics(y, mu, history=None, axes=None):
    """Two-panel diagnostic plot: fitted-vs-actual, and either the EM-filter PPL trace
    (when the model was fit with fit_method='em_filter') or a residual histogram."""
    import matplotlib.pyplot as plt
    y, mu = np.asarray(y), np.asarray(mu)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1, ax2 = axes
    ax1.scatter(mu, y, s=12, alpha=0.5, color="teal")
    lo, hi = min(mu.min(), y.min()), max(mu.max(), y.max())
    ax1.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("fitted")
    ax1.set_ylabel("actual")
    ax1.set_title("Fitted vs actual")

    if history:
        n_features = [h.n_features for h in history]
        ppl_vals = [h.ppl for h in history]
        ax2.plot(n_features, ppl_vals, marker="o", color="tab:red", markersize=3)
        ax2.set_xlabel("features/blocks retained")
        ax2.set_ylabel("PPL")
        ax2.invert_xaxis()
        ax2.set_title("EM-filter trace")
    else:
        resid = y - mu
        ax2.hist(resid, bins=30, color="tab:red", alpha=0.8)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("residual")
        ax2.set_title("Residuals")
    return axes


def plot_temporal_latent(z, z_var=None, rho=None, ax=None):
    """The smoothed AR(1) block-latent trajectory with a +-2 SD uncertainty band --
    the same plot as temporal_demo.py's Figure 3, generalized to any fitted block."""
    import matplotlib.pyplot as plt
    z = np.asarray(z)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    t = np.arange(len(z))
    if z_var is not None:
        sd = np.sqrt(np.asarray(z_var))
        ax.fill_between(t, z - 2 * sd, z + 2 * sd, color="teal", alpha=0.2, label="+-2 SD")
        ax.legend(loc="upper right", fontsize=8)
    ax.plot(t, z, color="teal", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    title = "Smoothed AR(1) block-latent trajectory"
    if rho is not None:
        title += f" (fitted rho={rho:.3f})"
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("latent state")
    return ax


def plot_loadings(loadings, member_names=None, ax=None):
    """Bar chart of a SAR block's per-member loadings on its shared latent factor."""
    import matplotlib.pyplot as plt
    loadings = np.asarray(loadings)
    member_names = list(member_names) if member_names is not None else \
        [f"m{j}" for j in range(len(loadings))]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(loadings))
    ax.bar(x, loadings, color="teal", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(member_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("loading on shared block latent")
    ax.set_title("SAR block-factor loadings")
    return ax
