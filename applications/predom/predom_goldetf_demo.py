"""Pre-registered 10-domain test, domain F2 (predicted FAIL): a basket of near-identical
gold-tracking ETFs (GLD, IAU, SGOL, OUNZ, BAR, AAAU -- all physically-backed gold trusts
that track the same underlying spot price to within basis points). Predict one gold ETF's
daily return from the OTHER gold ETFs in the basket.

Mechanism under test (preregistration_10domain_test.md, F2): these ETFs are correlated
already near-deterministically (>0.999 pairwise) -- there is essentially no idiosyncratic,
denoisable noise left for hierboost's block-latent decorrelation to average away. The
prediction is FAIL, but specifically because hierboost adds no VALUE on top of the raw
correlation, not because the correlation is absent or non-stationary (the generalization
gate is expected to PASS trivially here -- see module docstring in the prereg doc). Per
the run instructions, the falsifiable WORK/FAIL/MIXED rule is applied mechanically to
whatever numbers actually come out, not adjusted to match the prior verdict.

Two contrast assets are pulled alongside the gold basket purely to document the
saturation mechanism in the correlation matrix (SLV, a genuinely different metal with a
correlated-but-not-saturated relationship to gold; GDX, gold-mining equities, which carry
company/equity-market risk on top of the gold-price exposure) -- neither is used as a
model predictor for the primary task, since the domain's task is specifically "predict
one gold ETF from the OTHER gold ETFs," not from unrelated assets. A secondary block-
structure sweep across the full 7-predictor set (gold + contrast) is included to show
directly that the saturation is specific to the gold basket, not a generic "everything is
one block" artifact of the correlation-threshold method.

Baseline suite (Lasso, PCA+linear, Random Forest, naive) and hierboost (correlation-
threshold blocking, em_filter spike-and-slab) are all evaluated on the same chronological
train/test split, no leakage: block loadings, PCA components, and feature
standardization are all fit on the TRAIN window only and applied out-of-sample to TEST
(hierboost.factor.project_block_factor is the out-of-sample projection for the block
score, mirroring how a real forecasting deployment would use a fitted block-latent model).
"""
import json
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor, project_block_factor
from hierboost.spike_slab_gaussian import em_filter_gaussian

TARGET = "GLD"
GOLD_PREDICTORS = ["IAU", "SGOL", "OUNZ", "BAR", "AAAU"]  # the basket F2 is actually about
CONTRAST = ["SLV", "GDX"]  # correlation-matrix documentation only, not model inputs
ALL_TICKERS = [TARGET] + GOLD_PREDICTORS + CONTRAST

TRAIN_FRAC = 0.8
RHO_MAIN = 0.75  # same threshold finance_demo.py uses; the rho sweep below confirms the
                 # gold-basket collapse is not an artifact of this particular choice
RHO_SWEEP = [0.3, 0.5, 0.75, 0.9, 0.95, 0.99]
KAPPA_SWEEP = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
XI0, XI1, NU, LAM = -2.0, 2.0, 1.0, 1.0  # fixed, disclosed prior range (not tuned here)
FILTER_FRAC = 0.2
SEED = 0
OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "predom_goldetf.json")


# ---------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------

def load_returns():
    df = yf.download(ALL_TICKERS, period="3y", progress=False, auto_adjust=True)["Close"]
    df = df[ALL_TICKERS]
    rets = np.log(df).diff().dropna()
    return rets


def chrono_split_idx(n, train_frac=TRAIN_FRAC):
    n_train = int(n * train_frac)
    return n_train


def zscore_train_apply(train, *others):
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    out = [(train - mu) / sd]
    for o in others:
        out.append((o - mu) / sd)
    return out


# ---------------------------------------------------------------------------------
# Generalization gate (run and reported FIRST, before anything else)
# ---------------------------------------------------------------------------------

def generalization_gate(y, X, names, n_train):
    y_train, y_test = y[:n_train], y[n_train:]
    X_train, X_test = X[:n_train], X[n_train:]
    train_corr = np.array([np.corrcoef(X_train[:, j], y_train)[0, 1] for j in range(X_train.shape[1])])
    best_j = int(np.nanargmax(np.abs(train_corr)))
    test_corr = float(np.corrcoef(X_test[:, best_j], y_test)[0, 1])
    tr = float(train_corr[best_j])
    passed = bool(np.sign(test_corr) == np.sign(tr) and abs(test_corr) >= 0.5 * abs(tr))
    return dict(
        best_feature=names[best_j],
        train_corr=tr,
        test_corr=test_corr,
        held_out_over_train_ratio=abs(test_corr) / abs(tr) if tr != 0 else float("nan"),
        passed=passed,
        all_train_corr={n: float(c) for n, c in zip(names, train_corr)},
    )


# ---------------------------------------------------------------------------------
# Baseline suite
# ---------------------------------------------------------------------------------

def run_baselines(X_train, X_test, y_train, y_test):
    results = {}

    mu_naive = np.full_like(y_test, y_train.mean())
    results["naive (train-mean)"] = dict(
        r2=float(r2_score(y_test, mu_naive)), n_features=0)

    Xz_train, Xz_test = zscore_train_apply(X_train, X_test)

    lasso = LassoCV(cv=5, random_state=SEED, max_iter=20000).fit(Xz_train, y_train)
    pred_lasso = lasso.predict(Xz_test)
    n_nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-10))
    results[f"Lasso (alpha={lasso.alpha_:.2e})"] = dict(
        r2=float(r2_score(y_test, pred_lasso)), n_features=n_nonzero,
        coef=lasso.coef_.tolist(), alpha=float(lasso.alpha_))

    pca = PCA(n_components=0.95, svd_solver="full", random_state=SEED).fit(Xz_train)
    Zp_train, Zp_test = pca.transform(Xz_train), pca.transform(Xz_test)
    lin = LinearRegression().fit(Zp_train, y_train)
    pred_pca = lin.predict(Zp_test)
    results[f"PCA({pca.n_components_} comps, {pca.explained_variance_ratio_.sum():.4f} var)+linear"] = dict(
        r2=float(r2_score(y_test, pred_pca)), n_features=int(pca.n_components_),
        explained_variance_ratio=pca.explained_variance_ratio_.tolist())

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=SEED)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results[f"Random Forest ({X_train.shape[1]} raw features)"] = dict(
        r2=float(r2_score(y_test, pred_rf)), n_features=X_train.shape[1],
        feature_importances=rf.feature_importances_.tolist())

    return results


# ---------------------------------------------------------------------------------
# hierboost: correlation-threshold blocking
# ---------------------------------------------------------------------------------

def block_structure_sweep(X_train, names, rhos):
    """Confirm (or not) the expected mechanism: does the basket collapse into one
    giant block across a reasonable range of correlation thresholds?"""
    out = {}
    for rho in rhos:
        labels = blocks_from_correlation_threshold(X_train, rho=rho)
        membership = block_membership_lists(labels)
        out[str(rho)] = {
            "n_blocks": len(membership),
            "blocks": {str(b): [names[i] for i in idx] for b, idx in membership.items()},
        }
    return out


def build_block_design(X_train, X_test, names, rho):
    """Correlation-threshold blocking fit on TRAIN ONLY; block factor (PPCA) loadings
    also fit on TRAIN ONLY, projected out-of-sample onto TEST via project_block_factor
    -- no leakage. Singleton blocks pass the (train-standardized) raw feature through."""
    labels = blocks_from_correlation_threshold(X_train, rho=rho)
    membership = block_membership_lists(labels)
    names_arr = np.array(names)

    Xz_train, Xz_test = zscore_train_apply(X_train, X_test)

    block_ids_sorted = sorted(membership.keys())
    Z_train = np.zeros((Xz_train.shape[0], len(block_ids_sorted)))
    Z_test = np.zeros((Xz_test.shape[0], len(block_ids_sorted)))
    block_names, loadings_report = [], {}

    for k, b in enumerate(block_ids_sorted):
        idx = membership[b]
        members = names_arr[idx]
        if len(idx) == 1:
            Z_train[:, k] = Xz_train[:, idx[0]]
            Z_test[:, k] = Xz_test[:, idx[0]]
            bname = members[0]
            loadings_report[bname] = {members[0]: 1.0}
        else:
            scores, loadings = gaussian_block_factor(Xz_train[:, idx])
            train_mean = Xz_train[:, idx].mean(axis=0)
            Z_train[:, k] = scores
            Z_test[:, k] = project_block_factor(Xz_test[:, idx], loadings, train_mean)
            bname = "+".join(members) + "_factor"
            loadings_report[bname] = dict(zip(members, loadings.tolist()))
        block_names.append(bname)

    return Z_train, Z_test, block_names, loadings_report, {
        str(b): [names_arr[i] for i in idx] for b, idx in membership.items()}


def em_filter_sweep(Z_train, Z_test, y_train, y_test, names, kappas):
    """Sweep kappa (sparsity-prior strength) with a flat (unboosted) prior wr=1, exactly
    the "flat prior" ablation state_econ/earthquake_japan use -- there is no group-
    affinity structure to boost by here (no physical distance/time-order analog for a
    basket of interchangeable trackers), so this directly measures whether the spike-
    and-slab layer alone can prune, independent of any boosting mechanism."""
    n_train = Z_train.shape[0]
    p = Z_train.shape[1]
    Xd_train = np.column_stack([np.ones(n_train), Z_train])
    Xd_test = np.column_stack([np.ones(Z_test.shape[0]), Z_test])
    wr = np.ones(p)
    min_features = max(1, min(2, p))

    out = {}
    for kappa in kappas:
        filt = em_filter_gaussian(Xd_train, y_train, wr, XI0, XI1, kappa, NU, LAM,
                                   filter_frac=FILTER_FRAC, min_features=min_features)
        best = filt.best
        retained = best.retained_idx
        mu_test = Xd_test[:, np.concatenate([[0], retained + 1])] @ best.beta
        out[str(kappa)] = dict(
            n_retained=int(len(retained)),
            n_candidates=p,
            frac_retained=float(len(retained) / p),
            retained_names=[names[i] for i in retained],
            theta_hat=best.theta_hat.tolist(),
            r2=float(r2_score(y_test, mu_test)),
        )
    return out


# ---------------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------------

def apply_outcome_rule(gate, baselines, hb_primary_r2, hb_n_features_frac,
                        em_filter_block, em_filter_raw):
    notes = []

    gate_pass = gate["passed"]
    notes.append(f"Gate: best raw feature={gate['best_feature']!r}, "
                  f"train_corr={gate['train_corr']:+.4f}, test_corr={gate['test_corr']:+.4f}, "
                  f"{'PASSES' if gate_pass else 'FAILS'} (same sign & |test|>=50% |train|).")

    best_baseline_name = max(
        (n for n in baselines if n != "naive (train-mean)"),
        key=lambda n: baselines[n]["r2"])
    best_baseline_r2 = baselines[best_baseline_name]["r2"]
    naive_r2 = baselines["naive (train-mean)"]["r2"]

    r2_gap = hb_primary_r2 - best_baseline_r2
    crit_2a = gate_pass and (abs(r2_gap) <= 0.05) and (hb_n_features_frac <= 0.5)
    notes.append(f"Criterion 2a: hierboost R2={hb_primary_r2:.4f} vs best baseline "
                 f"({best_baseline_name})={best_baseline_r2:.4f} (gap={r2_gap:+.4f}, "
                 f"threshold 0.05); hierboost retains {hb_n_features_frac:.1%} of raw "
                 f"feature count (threshold <=50%). "
                 f"{'SATISFIED' if crit_2a else 'NOT satisfied'}.")

    crit_2b_beats_naive = hb_primary_r2 > naive_r2 + 0.05
    crit_2b = False  # no independently-known structural variable applies to a basket of
                      # interchangeable trackers (no physical distance/causal order/pathway
                      # analog) -- mechanism-validation is not available for this domain,
                      # so 2b cannot be satisfied regardless of the naive-beating margin
    notes.append(f"Criterion 2b: hierboost R2={hb_primary_r2:.4f} vs naive={naive_r2:.4f} "
                 f"(beats naive by >0.05: {crit_2b_beats_naive}), BUT no independently-known "
                 f"structural variable exists for a basket of interchangeable gold trackers "
                 f"(no physical-distance/causal-order/pathway analog) -- mechanism-validation "
                 f"half of 2b is not applicable/available, so 2b is NOT satisfied regardless.")

    work_ok = gate_pass and (crit_2a or crit_2b)

    fail_1 = not gate_pass
    fail_2 = hb_primary_r2 < naive_r2
    frac_block = em_filter_block
    frac_raw = em_filter_raw
    sparsity_never_engages = (min(frac_block) > 0.9) and (min(frac_raw) > 0.9)
    fail_3 = sparsity_never_engages and not crit_2b
    notes.append(f"Criterion FAIL-1 (gate fails): {fail_1}.")
    notes.append(f"Criterion FAIL-2 (worse than naive): hierboost R2={hb_primary_r2:.4f} "
                 f"< naive R2={naive_r2:.4f}: {fail_2}.")
    notes.append(f"Criterion FAIL-3 (sparsity never engages, >90% retained across kappa "
                 f"sweep, AND no mechanism-validation): block-design min frac retained="
                 f"{min(frac_block):.2f}, raw-design min frac retained={min(frac_raw):.2f}: "
                 f"{fail_3}.")

    fail_ok = fail_1 or fail_2 or fail_3

    if work_ok and not fail_ok:
        verdict = "WORK"
    elif fail_ok and not work_ok:
        verdict = "FAIL"
    else:
        verdict = "MIXED"
    notes.append(f"WORK criteria met: {work_ok}. FAIL criteria met: {fail_ok}. "
                 f"=> Mechanical verdict per the pre-registered rule: {verdict}")
    return dict(verdict=verdict, work_ok=work_ok, fail_ok=fail_ok,
                best_baseline=best_baseline_name, best_baseline_r2=best_baseline_r2,
                naive_r2=naive_r2, r2_gap=r2_gap, notes=notes)


# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    rets = load_returns()
    n = len(rets)
    n_train = chrono_split_idx(n)
    print(f"{n} trading days ({n_train} train / {n - n_train} test, chronological split)")
    print(f"target={TARGET}, gold basket predictors={GOLD_PREDICTORS}, "
          f"contrast (documentation only)={CONTRAST}")

    full_corr = rets[ALL_TICKERS].corr()
    print("\nFull-sample pairwise correlation matrix (all 8 tickers, documentation only):")
    print(full_corr.round(4))

    y = rets[TARGET].values
    X = rets[GOLD_PREDICTORS].values

    print("\n" + "=" * 78)
    print("STEP 1: GENERALIZATION GATE (run first, reported regardless of outcome)")
    print("=" * 78)
    gate = generalization_gate(y, X, GOLD_PREDICTORS, n_train)
    print(f"best train-correlated predictor: {gate['best_feature']} "
          f"(train r={gate['train_corr']:+.4f}) -> held-out test r={gate['test_corr']:+.4f} "
          f"(ratio={gate['held_out_over_train_ratio']:.3f}) "
          f"-> {'PASSES' if gate['passed'] else 'FAILS'}")

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("\n" + "=" * 78)
    print("STEP 2: baseline suite (held-out, chronological split)")
    print("=" * 78)
    baselines = run_baselines(X_train, X_test, y_train, y_test)
    for name, res in baselines.items():
        print(f"  {name:<45} R2={res['r2']:.4f}  n_features={res['n_features']}")

    print("\n" + "=" * 78)
    print("STEP 3: hierboost -- correlation-threshold block structure sweep (gold basket)")
    print("=" * 78)
    block_sweep = block_structure_sweep(X_train, GOLD_PREDICTORS, RHO_SWEEP)
    for rho_str, info in block_sweep.items():
        print(f"  rho={rho_str}: {info['n_blocks']} block(s) -> {info['blocks']}")

    print(f"\nMain block structure at rho={RHO_MAIN}:")
    Z_train, Z_test, block_names, loadings_report, membership_main = build_block_design(
        X_train, X_test, GOLD_PREDICTORS, RHO_MAIN)
    for bname, load in loadings_report.items():
        print(f"  {bname}: {load}")

    print("\nSecondary block structure sweep, full 7-predictor set (gold + contrast) "
          "-- documents that saturation is specific to the gold basket:")
    Xc_train_full = rets[GOLD_PREDICTORS + CONTRAST].values[:n_train]
    contrast_sweep = block_structure_sweep(Xc_train_full, GOLD_PREDICTORS + CONTRAST, RHO_SWEEP)
    for rho_str, info in contrast_sweep.items():
        print(f"  rho={rho_str}: {info['n_blocks']} block(s) -> {info['blocks']}")

    print("\n" + "=" * 78)
    print("STEP 4: hierboost primary fit -- OLS on block factor(s), held-out R2")
    print("=" * 78)
    lin_hb = LinearRegression().fit(Z_train, y_train)
    pred_hb = lin_hb.predict(Z_test)
    hb_primary_r2 = float(r2_score(y_test, pred_hb))
    hb_n_features_frac = len(block_names) / len(GOLD_PREDICTORS)
    print(f"hierboost block-factor OLS: {len(block_names)} block(s) of {len(GOLD_PREDICTORS)} "
          f"raw features ({hb_n_features_frac:.1%}), held-out R2={hb_primary_r2:.4f}")

    print("\n" + "=" * 78)
    print("STEP 5: em_filter sparsity-engagement sweep over kappa")
    print("=" * 78)
    print("-- on the block-factor design --")
    em_block = em_filter_sweep(Z_train, Z_test, y_train, y_test, block_names, KAPPA_SWEEP)
    for k, r in em_block.items():
        print(f"  kappa={k:>8}: retained {r['n_retained']}/{r['n_candidates']} "
              f"({r['frac_retained']:.1%}) -- {r['retained_names']}, R2={r['r2']:.4f}")

    print("-- on the raw (unblocked) 5-feature design --")
    Xz_train, Xz_test = zscore_train_apply(X_train, X_test)
    em_raw = em_filter_sweep(Xz_train, Xz_test, y_train, y_test, GOLD_PREDICTORS, KAPPA_SWEEP)
    for k, r in em_raw.items():
        print(f"  kappa={k:>8}: retained {r['n_retained']}/{r['n_candidates']} "
              f"({r['frac_retained']:.1%}) -- {r['retained_names']}, R2={r['r2']:.4f}")

    frac_block_list = [em_block[str(k)]["frac_retained"] for k in KAPPA_SWEEP]
    frac_raw_list = [em_raw[str(k)]["frac_retained"] for k in KAPPA_SWEEP]

    print("\n" + "=" * 78)
    print("STEP 6: apply the pre-registered WORK/FAIL/MIXED rule mechanically")
    print("=" * 78)
    verdict = apply_outcome_rule(gate, baselines, hb_primary_r2, hb_n_features_frac,
                                  frac_block_list, frac_raw_list)
    for line in verdict["notes"]:
        print(line)
    print(f"\n>>> F2 (gold-tracking ETF basket) verdict: {verdict['verdict']} <<<")
    print(f"(predicted in the preregistration: FAIL)")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    out = dict(
        domain="F2_gold_etf_basket",
        predicted="FAIL",
        n_obs=n, n_train=n_train, n_test=n - n_train,
        target=TARGET, gold_predictors=GOLD_PREDICTORS, contrast_assets=CONTRAST,
        full_correlation_matrix={r: {c: float(full_corr.loc[r, c]) for c in full_corr.columns}
                                  for r in full_corr.index},
        generalization_gate=gate,
        baselines=baselines,
        block_structure_sweep_gold_only=block_sweep,
        block_structure_sweep_with_contrast=contrast_sweep,
        hierboost_main_block_structure={"rho": RHO_MAIN, "membership": membership_main,
                                         "loadings": loadings_report},
        hierboost_primary_r2=hb_primary_r2,
        hierboost_primary_n_blocks=len(block_names),
        hierboost_primary_feature_retention_frac=hb_n_features_frac,
        em_filter_sweep_block_design=em_block,
        em_filter_sweep_raw_design=em_raw,
        kappa_sweep=KAPPA_SWEEP,
        rho_sweep=RHO_SWEEP,
        verdict=verdict,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults written to {OUT_JSON}")
