"""Third real-data test of hierboost.copula, and the one Ian specifically asked about:
does the Gaussian-copula marginal transform help back on hierboost's own origin domain,
1000 Genomes LD-block fine-mapping (genomics_1kg_demo.py)?

Motivating check, run BEFORE any CV (same discipline as the finance/weather copula
demos -- confirm the mechanism's premise holds before trusting a result): genotype
dosage (0/1/2) is Binomial(2, MAF) in distribution, and MAF varies substantially *within*
a single LD block (SNPs a few hundred bp apart can have very different minor allele
frequencies while still being in strong LD) -- confirmed empirically on the LCT region,
118 multi-member blocks, mean within-block MAF spread 0.221 (range 0-0.44), mean
within-block dosage-skewness spread 2.50. So the same "real correlation (LD/r^2), very
different marginal shapes (different MAFs)" case the finance/weather demos targeted is
genuinely present here too -- genomics_1kg_demo.py's own `fit_block_factors` runs
`gaussian_block_factor` directly on raw 0/1/2 dosage, the same "ideally already
standardized" assumption mismatch.

Note this is the CONTINUOUS branch only (factor.py, dosage treated as continuous, same
convention genomics_1kg_demo.py itself uses) -- NOT genomics_1kg_binomial_latent.py's
literal discrete/Binomial Ch4 JAX path, which is a different mechanism already handling
each SNP's own Binomial marginal correctly via its link function (see project memory:
that branch is already an unlabeled copula in the sense discussed with Ian). The
question here is narrower: does explicitly copula-transforming the raw dosage help the
*existing, already-validated* continuous PPCA block-latent pipeline specifically.

Reuses genomics_1kg_demo.py's own data loading/blocking/CV-fold structure unchanged
(imported, not duplicated) for a fair apples-to-apples comparison; only
`fit_block_factors` gets a copula-transform variant.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor, project_block_factor
from hierboost.copula import fit_block_transforms, apply_block_transforms
from hierboost.spike_slab import em_filter

from genomics_1kg_demo import load_data, build_blocks

LOCI = {
    "LCT (lactase persistence)": "~/genomics_1kg/lct_region.npz",
    "SLC24A5 (skin pigmentation)": "~/genomics_1kg/slc24a5_region.npz",
    "DARC/ACKR1 (Duffy null, malaria resistance)": "~/genomics_1kg/darc_region.npz",
}


def marginal_heterogeneity_report(X, block_id):
    maf = np.minimum(X.mean(axis=0) / 2, 1 - X.mean(axis=0) / 2)
    blocks = block_membership_lists(block_id)
    multi = {b: idx for b, idx in blocks.items() if len(idx) > 1}
    spreads = []
    for idx in multi.values():
        spreads.append(maf[idx].max() - maf[idx].min())
    spreads = np.array(spreads) if spreads else np.array([0.0])
    print(f"  {len(multi)} multi-member LD blocks, within-block MAF spread: "
          f"mean={spreads.mean():.3f}, median={np.median(spreads):.3f}, max={spreads.max():.3f}")


def fit_block_factors(X_train, X_test, block_id, marginal=None):
    """genomics_1kg_demo.py's own function, extended with an optional copula pass --
    fit each block's marginal transforms on TRAIN only, apply to both train and test
    (same fit-on-train/apply-to-held-out discipline `project_block_factor` already uses
    one level up), before gaussian_block_factor/project_block_factor run unchanged.
    """
    blocks = block_membership_lists(block_id)
    block_ids = sorted(blocks.keys())
    Ztr = np.zeros((X_train.shape[0], len(block_ids)))
    Zte = np.zeros((X_test.shape[0], len(block_ids))) if X_test is not None else None
    for k, b in enumerate(block_ids):
        idx = blocks[b]
        Xb_tr = X_train[:, idx]
        Xb_te = X_test[:, idx] if X_test is not None else None
        if marginal == "copula":
            transforms = fit_block_transforms(Xb_tr, kind="empirical")
            Xb_tr = apply_block_transforms(Xb_tr, transforms)
            if Xb_te is not None:
                Xb_te = apply_block_transforms(Xb_te, transforms)
        elif marginal == "copula-binom":
            # Genotype dosage's actual generating family (Binomial(n=2, p=MAF)) instead
            # of the generic rank-based transform -- the rank/empirical transform has no
            # tie-breaking jitter, so on a 3-valued variable it degenerates to a fixed
            # rescaling of {0,1,2} rather than a genuine per-individual Gaussianization;
            # the parametric route's jittered CDF (see copula.py docstring) is the
            # theoretically correct choice for a known discrete family.
            families = ["binom"] * Xb_tr.shape[1]
            transforms = fit_block_transforms(Xb_tr, kind="parametric", families=families)
            Xb_tr = apply_block_transforms(Xb_tr, transforms)
            if Xb_te is not None:
                Xb_te = apply_block_transforms(Xb_te, transforms)
        if len(idx) == 1:
            Ztr[:, k] = Xb_tr[:, 0]
            if Zte is not None:
                Zte[:, k] = Xb_te[:, 0]
            continue
        score, loadings = gaussian_block_factor(Xb_tr)
        Ztr[:, k] = score
        if Zte is not None:
            Zte[:, k] = project_block_factor(Xb_te, loadings, Xb_tr.mean(axis=0))
    return Ztr, Zte, block_ids


def hierboost_fold(X_train, y_train, X_test, y_test, block_id, xi0, marginal=None,
                    kappa=100.0, filter_frac=0.2, min_features=10):
    Ztr, Zte, _ = fit_block_factors(X_train, X_test, block_id, marginal=marginal)
    wr = np.ones(Ztr.shape[1])
    Xd_train = np.column_stack([np.ones(len(y_train)), Ztr])
    filt = em_filter(Xd_train, y_train, wr, xi0=xi0, xi1=0.0, kappa=kappa, nu=1.0, lam=1.0,
                      filter_frac=filter_frac, min_features=min_features, max_outer=30)
    best = filt.best
    Xd_test = np.column_stack([np.ones(len(y_test)), Zte[:, best.retained_idx]])
    pred = (Xd_test @ best.beta) > 0
    return float(np.mean(pred == (y_test > 0.5))), best.n_features


def cross_validate_raw_vs_copula(X, y, block_id, n_folds=10, seed=0):
    xi0 = np.log(0.15 / 0.85)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        acc_raw, n_raw = hierboost_fold(Xtr, ytr, Xte, yte, block_id, xi0, marginal=None)
        acc_cop, n_cop = hierboost_fold(Xtr, ytr, Xte, yte, block_id, xi0, marginal="copula")
        acc_bin, n_bin = hierboost_fold(Xtr, ytr, Xte, yte, block_id, xi0, marginal="copula-binom")
        rows.append(dict(acc_raw=acc_raw, n_raw=n_raw, acc_copula=acc_cop, n_copula=n_cop,
                          acc_binom=acc_bin, n_binom=n_bin))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for label, path in LOCI.items():
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            print(f"\n{label}: no cached data at {path}, skipping")
            continue
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        X, y, positions, pop_a, pop_b = load_data(path)
        block_id = build_blocks(positions)
        marginal_heterogeneity_report(X, block_id)

        df = cross_validate_raw_vs_copula(X, y, block_id, n_folds=10, seed=0)
        n_folds = len(df)
        print(f"\n10-fold CV accuracy ({pop_a} vs {pop_b}):")
        print(f"  raw block-latent (marginal=None)              : {df.acc_raw.mean():.4f} +/- {df.acc_raw.std():.4f}"
              f"  (mean {df.n_raw.mean():.1f} features)")
        print(f"  copula, empirical/rank (marginal=copula)      : {df.acc_copula.mean():.4f} +/- {df.acc_copula.std():.4f}"
              f"  (mean {df.n_copula.mean():.1f} features)")
        print(f"  copula, parametric Binomial (marginal=copula-binom): {df.acc_binom.mean():.4f} +/- {df.acc_binom.std():.4f}"
              f"  (mean {df.n_binom.mean():.1f} features)")
        for name, acc_col in [("empirical", df.acc_copula), ("binom", df.acc_binom)]:
            diff = acc_col - df.acc_raw
            print(f"  [{name} - raw] mean={diff.mean():+.4f}, wins={int((diff > 0).sum())}/{n_folds}, "
                  f"ties={int((diff == 0).sum())}, losses={int((diff < 0).sum())}")
