"""Pilot: does the finite partial sum P(s) (M=floor(t/pi) terms) carry structure
predictive of whether a zero is a "homotopy exception"? Features are the approach-path
shape of the partial-sum trajectory (9 checkpoint ratios r_j = S_j/S_M) plus 3 final-value
summaries -- not the raw sum itself, which would be tautological. Ground truth: Ian's
150-zero, 17-exception dataset."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from hierboost.estimator import HierBoostClassifier

DATA_PATH = os.path.expanduser(
    "~/Research/Riemann/writeup/homotopy_exception_gap_data.json")
CHECKPOINTS = np.arange(0.1, 1.0, 0.1)  # 9 fractions, 0.1..0.9
SIGMA = 0.5


def partial_sum_trajectory(t, M):
    n = np.arange(1, M + 1)
    signs = np.where(n % 2 == 1, 1.0, -1.0)
    terms = signs * n ** (-(SIGMA + 1j * t))
    return np.cumsum(terms)  # S_j at index j-1


def row_features(t):
    M = int(np.floor(t / np.pi))
    M = max(M, 10)  # guard against degenerate tiny-M rows
    S = partial_sum_trajectory(t, M)
    S_M = S[-1]
    idx = np.round(CHECKPOINTS * M).astype(int).clip(1, M) - 1
    ratios = S[idx] / S_M
    feats = np.concatenate([ratios.real, ratios.imag,
                             [np.log(np.abs(S_M)), np.cos(np.angle(S_M)), np.sin(np.angle(S_M))]])
    return feats, M


def build_dataset():
    data = json.load(open(DATA_PATH))
    ts = np.array([d["t"] for d in data])
    y = np.array([0.0 if d["on_line"] else 1.0 for d in data])  # 1 = exception
    feats, Ms = zip(*(row_features(t) for t in ts))
    X = np.vstack(feats)
    trivial = np.column_stack([np.log(ts), np.log(np.array(Ms))])
    # the ACTUAL predictors files 39/40 already use for exception detection -- the bar
    # the P(s)-shape features need to clear to claim independent value, not just beat chance
    gap = np.column_stack([[d["gap_before"] for d in data], [d["gap_after"] for d in data],
                            [d["ng_before"] for d in data], [d["ng_after"] for d in data]])
    feature_names = ([f"Re(r_{int(f*100)})" for f in CHECKPOINTS]
                      + [f"Im(r_{int(f*100)})" for f in CHECKPOINTS]
                      + ["log|S_M|", "cos(arg S_M)", "sin(arg S_M)"])
    print(f"{len(ts)} zeros, {int(y.sum())} exceptions, t in [{ts.min():.1f}, {ts.max():.1f}], "
          f"M in [{min(Ms)}, {max(Ms)}]")
    return X, trivial, gap, y, feature_names


def _hb_fold(Xtr, ytr, Xte, xi0):
    # HierBoostClassifier does NOT standardize internally (confirmed by reading
    # estimator.py/spike_slab.py -- no StandardScaler anywhere in the fit path), and
    # its slab variance is shared/scale-sensitive across features, so unstandardized
    # features on different raw scales (e.g. gap_before ~1-5 vs a cos/sin feature ~-1..1)
    # would get inconsistent shrinkage. Standardize on TRAIN only, apply to test.
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    hb = HierBoostClassifier(xi0=xi0, fit_method="em_filter", filter_frac=0.3,
                              min_features=3, max_outer=50)
    hb.fit(Xtr_s, ytr)
    p = np.asarray(hb.predict(Xte_s))
    n_ret = len(hb.retained_idx_) if getattr(hb, "retained_idx_", None) is not None else Xtr.shape[1]
    return p, n_ret


def cross_validate(X, trivial, gap, y, n_folds=5, n_repeats=10, seed=0):
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    xi0 = np.log((y.mean()) / (1 - y.mean()))  # prior centered on observed exception rate
    combined = np.column_stack([gap, X])
    rows = []

    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        ytr, yte = y[tr], y[te]

        # majority-class baseline
        p_const = np.full(len(yte), ytr.mean())
        rows.append(dict(fold=fold, method="majority-class baseline", auc=np.nan,
                          brier=float(np.mean((p_const - yte) ** 2))))

        # trivial (log t, log M) logistic -- the height confound check
        _log_fold("trivial (log t, log M) logistic", trivial, tr, te, ytr, yte, rows, fold)

        # gap-statistic logistic -- approximates the EXISTING exception-detection model
        # (files 39/40), the real bar the P(s)-shape features need to clear
        _log_fold("gap-stat logistic (existing model's features)", gap, tr, te, ytr, yte, rows, fold)

        # L1-logistic on the 21 P(s)-shape features
        _log_fold("L1-logistic (21 P(s)-shape features)", X, tr, te, ytr, yte, rows, fold,
                  penalty="l1", solver="liblinear", C=0.5)

        # Random Forest on the same 21 features
        rf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=seed,
                                     class_weight="balanced").fit(X[tr], ytr)
        p = rf.predict_proba(X[te])[:, 1]
        rows.append(dict(fold=fold, method="Random Forest (21 P(s)-shape features)",
                          auc=_safe_auc(yte, p), brier=float(np.mean((p - yte) ** 2))))

        # hierboost: spike-and-slab, no decorrelation, on the 21 P(s)-shape features alone
        p, n_ret = _hb_fold(X[tr], ytr, X[te], xi0)
        rows.append(dict(fold=fold, method="hierboost (21 P(s)-shape features)",
                          auc=_safe_auc(yte, p), brier=float(np.mean((p - yte) ** 2)), n_features=n_ret))

        # hierboost on gap stats + P(s)-shape features combined -- the real question:
        # does P(s) shape add anything ON TOP OF what the existing model already has
        p, n_ret = _hb_fold(combined[tr], ytr, combined[te], xi0)
        rows.append(dict(fold=fold, method="hierboost (gap stats + P(s) shape, combined)",
                          auc=_safe_auc(yte, p), brier=float(np.mean((p - yte) ** 2)), n_features=n_ret))

    return pd.DataFrame(rows)


def _log_fold(name, feats, tr, te, ytr, yte, rows, fold, **lr_kwargs):
    sc = StandardScaler().fit(feats[tr])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                             **lr_kwargs).fit(sc.transform(feats[tr]), ytr)
    p = lr.predict_proba(sc.transform(feats[te]))[:, 1]
    rows.append(dict(fold=fold, method=name, auc=_safe_auc(yte, p),
                      brier=float(np.mean((p - yte) ** 2))))


def _safe_auc(y_true, p):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, p)


def full_data_fit(X, y, feature_names):
    xi0 = np.log(y.mean() / (1 - y.mean()))
    X_s = StandardScaler().fit_transform(X)
    hb = HierBoostClassifier(xi0=xi0, fit_method="em_filter", filter_frac=0.3,
                              min_features=3, max_outer=50)
    hb.fit(X_s, y, feature_names=feature_names)
    print("\nFull-data hierboost fit -- feature inclusion probabilities:")
    print(hb.summary())
    return hb


if __name__ == "__main__":
    X, trivial, gap, y, feature_names = build_dataset()
    cv_df = cross_validate(X, trivial, gap, y)
    summary = cv_df.groupby("method").agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                                           brier_mean=("brier", "mean")).reset_index()
    summary = summary.sort_values("auc_mean", ascending=False)
    print("\n5-fold x 10-repeat stratified CV (predicting homotopy-exception label):")
    print(summary.to_string(index=False))

    full_data_fit(X, y, feature_names)
