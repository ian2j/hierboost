"""Cross-validation for hierboost.estimator's HierBoost* classes.

sklearn's own cross_val_score/cross_validate don't fit this API cleanly: `.fit()` takes
extra feature-level metadata (coords, block_id, group_l/r/coords/relevance) that must be
held fixed across folds rather than split, `offset` (when present) is per-observation and
DOES need splitting, and `.predict()` deliberately returns the natural-scale mean (a
probability or an expected count), not a hard label -- so sklearn's default classifier
scorer (which assumes `.predict()` returns labels) would silently score nonsense if
plugged in directly. This is exactly the "clone, fit per fold, score" loop
`genomics_1kg_demo.py`'s `cross_validate`/`cross_validate_hierboost_only`,
`finance_factor_selection.py`, etc. each hand-roll -- generalized once here instead of
copy-pasted per demo script.
"""
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def _clone(estimator):
    return type(estimator)(**estimator.get_params())


def _default_scorer(estimator, y_true, y_pred):
    """Accuracy for a Bernoulli classifier (n_trials=1), R^2 otherwise (Gaussian,
    Binomial with n_trials>1, Poisson, Negative-Binomial) -- bigger-is-better in both
    cases, matching sklearn's scoring convention."""
    if estimator._response == "binomial" and getattr(estimator, "n_trials_", 1) == 1:
        return float(np.mean((y_pred >= 0.5) == y_true))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _iter_folds(estimator, X, y, cv, shuffle, random_state, stratified):
    if stratified is None:
        stratified = estimator._response == "binomial"
    if stratified:
        splitter = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        return splitter.split(X, y)
    splitter = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    return splitter.split(X)


def _fit_predict_fold(estimator, X, y, offset, train_idx, test_idx, fit_kwargs):
    est = _clone(estimator)
    fold_kwargs = dict(fit_kwargs)
    predict_kwargs = {}
    if offset is not None:
        fold_kwargs["offset"] = offset[train_idx]
        predict_kwargs["offset_new"] = offset[test_idx]
    est.fit(X[train_idx], y[train_idx], **fold_kwargs)
    y_pred = np.asarray(est.predict(X[test_idx], **predict_kwargs))
    return est, y_pred


def cross_val_score(estimator, X, y, cv=5, scoring=None, shuffle=True, random_state=None,
                     stratified=None, **fit_kwargs):
    """K-fold CV score for a HierBoost* estimator, cloned fresh (same constructor
    params, via `.get_params()`) and re-fit on each fold's training split.

    `fit_kwargs` (coords, block_id, group_l/r/coords/relevance, affinity_kind,
    affinity_bandwidth, feature_names, n_trials) are passed to every fold's `.fit()`
    unchanged, EXCEPT `offset`, which -- being per-observation, unlike the others -- is
    split by row like X/y and passed through to `.predict()` as `offset_new`.

    `scoring(estimator, y_true, y_pred) -> float` defaults to accuracy (Bernoulli
    classifier) or R^2 (everything else -- Gaussian, Binomial with n_trials>1, Poisson,
    Negative-Binomial); `stratified` defaults to StratifiedKFold-on-y for a Bernoulli
    classifier, plain KFold otherwise.

    Returns an (cv,) array of per-fold scores.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    scoring = scoring or _default_scorer
    offset = fit_kwargs.pop("offset", None)
    offset = None if offset is None else np.asarray(offset, dtype=float)

    scores = []
    for train_idx, test_idx in _iter_folds(estimator, X, y, cv, shuffle, random_state, stratified):
        est, y_pred = _fit_predict_fold(estimator, X, y, offset, train_idx, test_idx, fit_kwargs)
        scores.append(scoring(est, y[test_idx], y_pred))
    return np.array(scores)


def cross_val_predict(estimator, X, y, cv=5, shuffle=True, random_state=None,
                       stratified=None, **fit_kwargs):
    """Out-of-fold predictions for a HierBoost* estimator, in the original row order --
    an honest fitted-vs-actual view (e.g. for hierboost.viz.plot_diagnostics) that isn't
    contaminated by the in-sample fit every `.summary()`/`.plot_diagnostics()` otherwise
    shows. Same fold/clone/offset handling as cross_val_score; see its docstring."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = fit_kwargs.pop("offset", None)
    offset = None if offset is None else np.asarray(offset, dtype=float)

    y_pred_oof = np.full(y.shape[0], np.nan)
    for train_idx, test_idx in _iter_folds(estimator, X, y, cv, shuffle, random_state, stratified):
        _, y_pred = _fit_predict_fold(estimator, X, y, offset, train_idx, test_idx, fit_kwargs)
        y_pred_oof[test_idx] = y_pred
    return y_pred_oof
