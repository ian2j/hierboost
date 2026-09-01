"""W4 of preregistration_10domain_test.md: gene co-expression predicting a real
clinical/molecular trait (predicted WORK). Direct structural analog of
genomics_1kg_eqtl.py -- same "raw feature -> hierboost.structure block-latent
factor -> spike-and-slab regression, held-out split, baseline suite" pipeline --
but this time genuinely designed to have a plausible biological signal end to end,
closing the "real phenotype, not proxy" gap the eQTL thread (LCT/APOL1) never fully
closed: LCT/APOL1 predicted GENE EXPRESSION from SNP genotype; this predicts a real
QUANTITATIVE MOLECULAR/CLINICAL VARIABLE (tumor mutational burden) from
gene-expression CO-EXPRESSION MODULES, blocked with the same
hierboost.structure.blocks_from_correlation_threshold machinery used in
finance_cross_sectional.py, instead of LD-distance blocking.

Data source: cBioPortal public REST API (cbioportal.org/api), verified live
2026-08-30, OPEN-ACCESS tier only (no auth used or required, no controlled-access
TCGA data touched):
  - study:      brca_tcga_pan_can_atlas_2018 (TCGA Breast Cancer, PanCancer Atlas)
  - expression: <study>_rna_seq_v2_mrna_median_all_sample_Zscores (RNA Seq V2 RSEM,
                z-scored relative to all samples in the cohort) -- 1082 samples with
                complete data over the gene panel below, verified live (NA
                fraction = 0.0 over the full fetched matrix).
  - target:     TMB_NONSYNONYMOUS, a SAMPLE-level clinical attribute (tumor
                mutational burden, nonsynonymous mutations per Mb) -- a genuinely
                measured, continuous molecular variable, NOT an ancestry-label
                proxy and NOT a gene-expression-derived signature score (the
                cohort's hypoxia/ESTIMATE-style scores were deliberately avoided
                for exactly that reason: they are themselves linear combinations of
                expression, which would make "predicting them from expression
                modules" circular by construction). 1066 of 1082 samples have a
                non-missing TMB value; 1064 samples overlap after joining to the
                expression matrix -- confirmed live before committing to this
                cohort/variable pair.

Why TMB is a fair, non-circular target: it is computed from an independent assay
(the somatic mutation calls), not derived from the RNA-seq data at all. It also has
a well-documented, moderate (not saturated) mechanistic link to gene expression: a
higher neoantigen load from more somatic mutations recruits immune infiltration
(cytolytic/interferon-response genes), and DNA-repair/replication-stress pathway
activity both drives and responds to mutation accumulation -- exactly the kind of
"real biological signal, plausible but not already a known-null single gene" this
domain is supposed to test, in contrast to the eQTL thread's single-SNP-vs-single-
gene design.

Gene panel: 235 genes (see GENE_GROUPS below), chosen for an a priori biological
rationale BEFORE any correlation with TMB was checked -- proliferation/cell-cycle,
DNA-damage-response/repair, and immune/interferon-response genes (all with a
documented mechanistic link to somatic mutation burden), plus PAM50-adjacent
subtype-identity genes and a diffuse housekeeping/pan-cancer-driver background set
(both included as genuine "should NOT particularly predict TMB" controls, not
padding). This grouping is also used, unchanged, for the mechanism-validation check
in outcome rule 2(b): each gene's a priori group is fixed at panel-design time and
never revised after seeing results.
"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor, project_block_factor
from hierboost.spike_slab_gaussian import em_filter_gaussian

API = "https://www.cbioportal.org/api"
STUDY = "brca_tcga_pan_can_atlas_2018"
EXPR_PROFILE = f"{STUDY}_rna_seq_v2_mrna_median_all_sample_Zscores"
SAMPLE_LIST = f"{STUDY}_rna_seq_v2_mrna"
CLINICAL_ATTR = "TMB_NONSYNONYMOUS"

DATA_DIR = "predom_tcga/data"
GENE_MAP_JSON = f"{DATA_DIR}/gene_map.json"
EXPR_PICKLE = f"{DATA_DIR}/expr_matrix.pkl"
CLINICAL_PICKLE = f"{DATA_DIR}/clinical_sample.pkl"
RESULTS_JSON = "results/predom_tcga.json"

TRAIN_FRAC = 0.8
SEED = 0
RHO_MAIN = 0.7  # co-expression blocking threshold for the headline run (see
                # blocking_rho_sensitivity below for why: rho<=0.5 collapses most
                # of the panel into one or two giant transitively-merged blocks --
                # e.g. rho=0.3 gives one 226-gene block -- while rho=0.7 gives 128
                # well-separated blocks whose largest members recover exactly the
                # designed proliferation/immune/stromal modules)

# --- gene panel, grouped by a priori biological rationale (fixed before any TMB
# correlation was computed) ---------------------------------------------------
PROLIFERATION = [  # replication-stress / cell-cycle: more cycling cells -> more
    # opportunity for replication errors to become fixed somatic mutations
    "MKI67", "CCNB1", "CCNB2", "CCNE1", "CCNE2", "CDK1", "CDK2", "CDC20", "CDC25A",
    "CDC25C", "PLK1", "AURKA", "AURKB", "BUB1", "BUB1B", "TOP2A", "RRM2", "TYMS",
    "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "E2F1", "E2F2", "MYBL2",
    "FOXM1", "BIRC5", "TTK", "KIF2C", "CENPA", "CENPF", "NDC80", "NUSAP1", "UBE2C",
]
DNA_REPAIR = [  # DNA damage response / mismatch & homologous repair: loss of
    # function here is the canonical mechanistic driver of elevated TMB
    "TP53", "BRCA1", "BRCA2", "ATM", "ATR", "MLH1", "MSH2", "MSH6", "PMS2", "RAD51",
    "RAD51C", "PARP1", "CHEK1", "CHEK2", "ERCC1", "ERCC2", "XRCC1", "XRCC2", "FANCA",
    "FANCD2", "FANCC", "POLE", "POLD1", "MDC1", "H2AX", "RPA1",
]
IMMUNE = [  # cytolytic/interferon response: higher TMB -> more neoantigens -> more
    # immune infiltration is a well-documented pan-cancer correlate (e.g. Rooney
    # et al. 2015's cytolytic activity score)
    "CD8A", "CD8B", "CD274", "PDCD1", "CTLA4", "GZMA", "GZMB", "PRF1", "IFNG",
    "STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11", "TAP1", "TAP2", "HLA-A", "HLA-B",
    "HLA-C", "B2M", "IDO1", "LAG3", "HAVCR2", "TIGIT", "CD3D", "CD3E", "CD2",
    "PTPRC", "GBP1", "ICAM1",
]
SUBTYPE_IDENTITY = [  # luminal/basal/HER2 identity (PAM50-adjacent): a strong,
    # well-known co-expression module, but mechanistically ORTHOGONAL to mutation
    # burden -- a genuine control module, not a TMB predictor
    "ESR1", "PGR", "ERBB2", "GRB7", "FOXA1", "GATA3", "MLPH", "NAT1", "XBP1",
    "SLC39A6", "KRT5", "KRT14", "KRT17", "MIA", "MLANA", "MMP11", "BAG1", "BCL2",
    "CCND1", "MDM2",
]
BACKGROUND = [  # housekeeping + a diffuse panel of common cancer/signaling genes
    # with no a priori TMB hypothesis -- genuine "noise" background so sparsity has
    # something real to select against, not a hand-picked TMB-correlated set
    "ACTB", "GAPDH", "TUBB", "RPL13A", "RPLP0", "HPRT1", "TBP", "YWHAZ", "PPIA",
    "SDHA", "PGK1", "UBC", "GUSB", "POLR2A", "KRAS", "PIK3CA", "PTEN", "APC",
    "EGFR", "BRAF", "NRAS", "SMAD4", "CDH1", "VEGFA", "MET", "ALK", "IDH1", "IDH2",
    "NOTCH1", "JAK2", "AKT1", "MTOR", "RB1", "CDKN2A", "SRC", "ABL1", "KIT", "FLT3",
    "JUN", "FOS", "NFKB1", "RELA", "SMAD3", "TGFB1", "IL6", "TNF", "VIM", "CDH2",
    "SNAI1", "SNAI2", "ZEB1", "TWIST1", "COL1A1", "COL1A2", "FN1", "ITGB1", "ITGA5",
    "LAMA1", "THBS1", "SPARC", "TIMP1", "MMP2", "MMP9", "PLAU", "SERPINE1",
    "ANGPT2", "FLT1", "KDR", "TEK", "PDGFRB", "PDGFB", "FGF2", "FGFR1", "HGF",
    "IGF1", "IGF1R", "INSR", "LEP", "ADIPOQ", "PPARG", "SREBF1", "FASN", "ACACA",
    "HMGCR", "LDLR", "APOE", "ALB", "TTR", "AFP", "CEACAM5", "MUC1", "EPCAM",
    "KRT8", "KRT18", "KRT19", "DES", "MYH11", "ACTA2", "CNN1", "TAGLN", "PECAM1",
    "VWF", "CDH5", "CD34", "THY1", "NES", "SOX2", "POU5F1", "NANOG", "KLF4", "MYC",
    "CCND2", "CDKN1A", "CDKN1B", "GADD45A", "BAX", "BCL2L1", "CASP3", "CASP9",
    "XIAP", "AKT2", "PIK3R1",
]
GENE_GROUPS = {
    "proliferation": PROLIFERATION, "dna_repair": DNA_REPAIR, "immune": IMMUNE,
    "subtype_identity": SUBTYPE_IDENTITY, "background": BACKGROUND,
}
# a priori hypothesis for the mechanism-validation check: proliferation/DNA-repair/
# immune genes are hypothesized MORE relevant to TMB than subtype-identity/background
TMB_HYPOTHESIZED_RELEVANT = set(PROLIFERATION) | set(DNA_REPAIR) | set(IMMUNE)
ALL_GENES = sorted(set(PROLIFERATION + DNA_REPAIR + IMMUNE + SUBTYPE_IDENTITY + BACKGROUND))


def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json", "Accept": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def fetch_expression():
    """mRNA z-score matrix (samples x genes) for the panel above, cached to disk --
    a single live fetch of all 235 genes takes ~3 minutes over the REST API."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EXPR_PICKLE):
        mat = pd.read_pickle(EXPR_PICKLE)
        print(f"[fetch] {EXPR_PICKLE} already present, skipping download ({mat.shape})")
        return mat
    print(f"[fetch] mapping {len(ALL_GENES)} HUGO symbols -> Entrez IDs via {API}/genes/fetch")
    gmap = _post(f"{API}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", ALL_GENES)
    sym2entrez = {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in gmap}
    missing = set(ALL_GENES) - set(sym2entrez)
    if missing:
        raise RuntimeError(f"cBioPortal did not resolve these symbols: {missing}")
    entrez2sym = {v: k for k, v in sym2entrez.items()}
    with open(GENE_MAP_JSON, "w") as f:
        json.dump(sym2entrez, f, indent=2)

    print(f"[fetch] fetching {EXPR_PROFILE} for {len(sym2entrez)} genes, "
          f"sample list {SAMPLE_LIST} (this is the slow step, ~3 min)")
    t0 = time.time()
    body = {"entrezGeneIds": list(sym2entrez.values()), "sampleListId": SAMPLE_LIST}
    expr = _post(f"{API}/molecular-profiles/{EXPR_PROFILE}/molecular-data/fetch?projection=SUMMARY", body)
    print(f"[fetch] {len(expr)} (sample, gene) records in {time.time()-t0:.0f}s")

    df = pd.DataFrame(expr)
    df["gene"] = df["entrezGeneId"].map(entrez2sym)
    mat = df.pivot_table(index="sampleId", columns="gene", values="value")
    print(f"[fetch] matrix shape {mat.shape}, NA fraction {mat.isna().mean().mean():.4f}")
    mat = mat.dropna(axis=0, how="any")  # keep only samples with the full panel
    mat.to_pickle(EXPR_PICKLE)
    return mat


def fetch_clinical_tmb():
    """SAMPLE-level TMB_NONSYNONYMOUS for the whole study, cached to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CLINICAL_PICKLE):
        s = pd.read_pickle(CLINICAL_PICKLE)
        print(f"[fetch] {CLINICAL_PICKLE} already present, skipping download (n={len(s)})")
        return s
    print(f"[fetch] fetching SAMPLE-level clinical data for {STUDY}")
    clin = _get(f"{API}/studies/{STUDY}/clinical-data?clinicalDataType=SAMPLE&pageSize=50000&pageNumber=0")
    tmb = {r["sampleId"]: float(r["value"]) for r in clin if r["clinicalAttributeId"] == CLINICAL_ATTR}
    s = pd.Series(tmb, name=CLINICAL_ATTR)
    print(f"[fetch] {len(s)} samples with non-missing {CLINICAL_ATTR}")
    s.to_pickle(CLINICAL_PICKLE)
    return s


def build_dataset():
    mat = fetch_expression()
    tmb = fetch_clinical_tmb()
    common = mat.index.intersection(tmb.index)
    print(f"[build] {len(common)} samples with BOTH expression and TMB "
          f"(of {mat.shape[0]} expr, {len(tmb)} clinical)")
    X = mat.loc[common].to_numpy()
    genes = mat.columns.to_numpy()
    y_raw = tmb.loc[common].to_numpy()
    # log10(TMB + 1): standard transform for mutation-burden data, which is heavily
    # right-skewed (median 1.3, max 180.8 mutations/Mb in this cohort) -- a handful
    # of hypermutators would otherwise dominate a raw-scale linear-Gaussian fit.
    # Decided from the raw distribution alone, before any correlation was examined.
    y = np.log10(y_raw + 1.0)
    print(f"[build] y = log10(TMB+1): min={y.min():.3f} median={np.median(y):.3f} "
          f"max={y.max():.3f} std={y.std():.3f}")
    return X, y, genes, common.to_numpy(), y_raw


def split_indices(n, train_frac=TRAIN_FRAC, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    return idx[:n_train], idx[n_train:]


def check_generalization(X, y, genes, label):
    """PRE-REGISTERED GATE, run first, before any other fit: does the single best
    train-correlated raw gene's correlation with TMB survive a held-out random split
    (same-sign, held-out |corr| >= 50% of train |corr|)? Random split is fine here
    per the project discipline -- there is no time axis to a cross-sectional cohort."""
    n = len(y)
    tr, te = split_indices(n)
    train_corr = np.array([np.corrcoef(X[tr, j], y[tr])[0, 1] for j in range(X.shape[1])])
    best_j = int(np.nanargmax(np.abs(train_corr)))
    test_corr = float(np.corrcoef(X[te, best_j], y[te])[0, 1])
    same_sign = bool(np.sign(train_corr[best_j]) == np.sign(test_corr))
    ratio = abs(test_corr) / abs(train_corr[best_j]) if train_corr[best_j] != 0 else float("nan")
    passes = bool(same_sign and ratio >= 0.5)
    print(f"[{label}] generalization gate: best train-correlated gene = {genes[best_j]} "
          f"(train r={train_corr[best_j]:+.4f}) -> held-out test r={test_corr:+.4f} "
          f"(ratio={ratio:.3f}, same_sign={same_sign}) -> {'PASS' if passes else 'FAIL'}")
    top10 = np.argsort(-np.abs(train_corr))[:10]
    print(f"[{label}] top-10 train-correlated genes: "
          f"{[(str(genes[j]), round(float(train_corr[j]), 3)) for j in top10]}")
    return dict(best_gene=str(genes[best_j]), best_gene_group=_gene_group(genes[best_j]),
                train_corr=float(train_corr[best_j]), test_corr=test_corr,
                ratio=float(ratio), same_sign=same_sign, passes=passes,
                top10_train_corr=[dict(gene=str(genes[j]), corr=float(train_corr[j])) for j in top10])


def _gene_group(gene):
    for name, members in GENE_GROUPS.items():
        if gene in members:
            return name
    return "unknown"


def build_blocks(X_train, rho):
    """Co-expression blocking on TRAIN data only (blocks_from_correlation_threshold,
    same machinery as finance_cross_sectional.py) -- computed on the training fold
    alone so no test-set structure leaks into the block definition."""
    block_id = blocks_from_correlation_threshold(X_train, rho=rho)
    sizes = np.bincount(block_id)
    print(f"[blocks] rho={rho}: {len(sizes)} co-expression blocks, "
          f"size range [{sizes.min()}, {sizes.max()}]")
    return block_id


def fit_block_factors(X_train, X_test, block_id):
    """Same per-block PPCA factor fit/project as genomics_1kg_eqtl.py's
    fit_block_factors, just renamed variables (genes instead of SNPs)."""
    blocks = block_membership_lists(block_id)
    block_ids = sorted(blocks.keys())
    Ztr = np.zeros((X_train.shape[0], len(block_ids)))
    Zte = np.zeros((X_test.shape[0], len(block_ids))) if X_test is not None else None
    for k, b in enumerate(block_ids):
        idx = blocks[b]
        Xb_tr = X_train[:, idx]
        if len(idx) == 1:
            Ztr[:, k] = Xb_tr[:, 0]
            if Zte is not None:
                Zte[:, k] = X_test[:, idx][:, 0]
            continue
        score, loadings = gaussian_block_factor(Xb_tr)
        Ztr[:, k] = score
        if Zte is not None:
            Zte[:, k] = project_block_factor(X_test[:, idx], loadings, Xb_tr.mean(axis=0))
    return Ztr, Zte, block_ids, blocks


def fit_hierboost(Ztr, ytr, Zte, kappa=100.0, filter_frac=0.2, min_features=10):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Ztr_s, Zte_s = (Ztr - mu) / sd, (Zte - mu) / sd
    K = Ztr.shape[1]
    wr = np.ones(K)
    Xd_train = np.column_stack([np.ones(len(ytr)), Ztr_s])
    filt = em_filter_gaussian(Xd_train, ytr, wr, xi0=np.log(0.15 / 0.85), xi1=0.0,
                               kappa=kappa, nu=1.0, lam=1.0, nu_y=1.0, lam_y=1.0,
                               filter_frac=filter_frac, min_features=min_features, max_outer=30)
    best = filt.best
    Xd_test = np.column_stack([np.ones(Zte_s.shape[0]), Zte_s[:, best.retained_idx]])
    pred = Xd_test @ best.beta
    return pred, best, filt


def run_main(X, y, genes, rho=RHO_MAIN, seed=SEED):
    n, p = X.shape
    tr, te = split_indices(n, seed=seed)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    block_id = build_blocks(Xtr, rho)
    K = len(np.unique(block_id))
    Ztr, Zte, block_ids, blocks = fit_block_factors(Xtr, Xte, block_id)

    results = {}
    naive_mu = np.full_like(yte, ytr.mean())
    results["naive (train mean)"] = (naive_mu, 0)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    lasso = LassoCV(cv=5, max_iter=20000, random_state=seed).fit(Xtr_s, ytr)
    results[f"Lasso/L1 (raw, {p} genes)"] = (lasso.predict(Xte_s), int((lasso.coef_ != 0).sum()))

    K_pca = min(K, Xtr.shape[0] - 1)
    pca = PCA(n_components=K_pca, random_state=seed).fit(Xtr_s)
    from sklearn.linear_model import LinearRegression
    lr_pca = LinearRegression().fit(pca.transform(Xtr_s), ytr)
    results[f"PCA({K_pca})+linear"] = (lr_pca.predict(pca.transform(Xte_s)), K_pca)

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed).fit(Xtr, ytr)
    results[f"Random Forest (raw, {p} genes)"] = (rf.predict(Xte), p)

    pred_hb, best_hb, filt_hb = fit_hierboost(Ztr, ytr, Zte)
    retained_blocks = [block_ids[j] for j in best_hb.retained_idx]
    n_genes_retained = int(sum(len(blocks[b]) for b in retained_blocks))
    results[f"hierboost (co-expr block-latent + spike-slab, K={K} blocks)"] = (pred_hb, best_hb.n_features)

    print(f"\n{'#'*88}\nn={n} samples ({len(tr)} train / {len(te)} test, random split, seed={seed}), "
          f"p={p} raw genes, K={K} co-expression blocks (rho={rho})")
    print("-" * 88)
    print(f"{'model':<52}{'test R2':>10}{'corr':>10}{'n_feat':>8}")
    print("-" * 88)
    row_results = {}
    for name, (mu, nf) in results.items():
        r2 = r2_score(yte, mu)
        corr = np.corrcoef(mu, yte)[0, 1] if np.std(mu) > 1e-10 else float("nan")
        print(f"{name:<52}{r2:>10.4f}{corr:>10.4f}{nf:>8d}")
        row_results[name] = dict(r2=float(r2), corr=float(corr) if np.isfinite(corr) else None, n_features=int(nf))

    # --- mechanism-validation check (outcome rule 2b): does posterior inclusion
    # confidence, on the FULL (pre-filtering) block set, correlate with each block's
    # a priori "hypothesized TMB-relevant pathway" membership fraction, fixed at
    # panel-design time (proliferation/DNA-repair/immune = 1, else 0)? ---
    first_step = filt_hb.history[0]  # full K-block model, before any EM-filter removal
    assert first_step.n_features == K
    frac_relevant = np.array([
        np.mean([g in TMB_HYPOTHESIZED_RELEVANT for g in genes[blocks[block_ids[j]]]])
        for j in range(K)
    ])
    theta_full = first_step.theta_hat
    mech_rho, mech_p = stats.spearmanr(theta_full, frac_relevant)
    print(f"\nmechanism check: spearman(theta_hat [full {K}-block model], "
          f"frac_genes_in_hypothesized-relevant_pathway) = rho={mech_rho:+.3f} (p={mech_p:.3g})")
    top5_idx = np.argsort(-theta_full)[:5]
    for j in top5_idx:
        b = block_ids[j]
        members = list(genes[blocks[b]])
        print(f"  block {b:4d}  theta={theta_full[j]:.3f}  frac_relevant={frac_relevant[j]:.2f}  "
              f"n_genes={len(members)}  sample_members={members[:8]}")

    p_raw = X.shape[1]
    gene_retention_frac = n_genes_retained / p_raw
    block_retention_frac = best_hb.n_features / K
    print(f"\nhierboost retained {best_hb.n_features}/{K} blocks "
          f"({block_retention_frac:.1%} of blocks), covering {n_genes_retained}/{p_raw} raw genes "
          f"({gene_retention_frac:.1%} of raw genes)")

    return dict(
        n=n, n_train=len(tr), n_test=len(te), p_raw_genes=p_raw, K_blocks=K, rho=rho,
        results=row_results,
        hierboost_n_blocks_retained=int(best_hb.n_features),
        hierboost_n_genes_retained=n_genes_retained,
        hierboost_block_retention_frac=float(block_retention_frac),
        hierboost_gene_retention_frac=float(gene_retention_frac),
        mechanism_check=dict(spearman_rho=float(mech_rho), spearman_p=float(mech_p),
                              top5_blocks_by_theta=[dict(block=int(block_ids[j]), theta=float(theta_full[j]),
                                                          frac_hypothesized_relevant=float(frac_relevant[j]),
                                                          n_genes=int(len(genes[blocks[block_ids[j]]])),
                                                          sample_genes=list(genes[blocks[block_ids[j]]])[:8])
                                                     for j in top5_idx]),
    )


def sparsity_sweep(X, y, rho=RHO_MAIN, seed=SEED):
    """FAIL-rule check #3: does em_filter's sparsity ever meaningfully engage across
    a reasonable kappa sweep (holding the co-expression blocking fixed), or does it
    retain >90% of the candidate blocks fed into it regardless?"""
    n = len(y)
    tr, _ = split_indices(n, seed=seed)
    Xtr, ytr = X[tr], y[tr]
    block_id = build_blocks(Xtr, rho)
    K = len(np.unique(block_id))
    Ztr, _, block_ids, _ = fit_block_factors(Xtr, None, block_id)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Ztr_s = (Ztr - mu) / sd
    Xd_train = np.column_stack([np.ones(len(ytr)), Ztr_s])
    wr = np.ones(K)

    print(f"\n[sweep] em_filter block retention vs kappa (K={K} candidate co-expression blocks):")
    sweep = []
    for kap in [10.0, 100.0, 1000.0, 10000.0]:
        filt = em_filter_gaussian(Xd_train, ytr, wr, xi0=np.log(0.15 / 0.85), xi1=0.0, kappa=kap,
                                   nu=1.0, lam=1.0, filter_frac=0.2, min_features=10, max_outer=30)
        n_ret = filt.best.n_features
        frac = n_ret / K
        print(f"  kappa={kap:<10g} retained {n_ret}/{K} blocks ({frac:.1%})")
        sweep.append(dict(kappa=kap, n_retained=n_ret, k_blocks=K, frac_retained=float(frac)))
    max_frac = max(s["frac_retained"] for s in sweep)
    return sweep, max_frac


def blocking_rho_sensitivity(X, y, seed=SEED):
    """Extra transparency, not required by the outcome rule: how sensitive is the
    block count / largest-block size to the correlation threshold rho? Reported so
    the RHO_MAIN=0.7 choice is auditable rather than a black-boxed pick."""
    n = len(y)
    tr, _ = split_indices(n, seed=seed)
    Xtr = X[tr]
    rows = []
    for rho in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        block_id = blocks_from_correlation_threshold(Xtr, rho=rho)
        sizes = np.bincount(block_id)
        rows.append(dict(rho=rho, n_blocks=int(len(sizes)), max_block_size=int(sizes.max()),
                          n_singleton_blocks=int((sizes == 1).sum())))
        print(f"  rho={rho}: {len(sizes)} blocks, max size {sizes.max()}, "
              f"{int((sizes==1).sum())} singletons")
    return rows


def apply_outcome_rule(gate, main_run, sweep_max_frac):
    """Applies preregistration_10domain_test.md's WORK/FAIL/MIXED rule EXACTLY as
    written to this run's actual numbers -- no discretion. Note on "raw feature
    count" vs "candidate features": rule 2(a)'s "<=50% of the raw feature count"
    is evaluated against RAW GENES (n_genes_in_retained_blocks / 235), matching the
    document's literal wording; FAIL-rule 3's "candidate features" is evaluated
    against BLOCKS (the actual columns em_filter_gaussian selects over), matching
    the sparsity_sweep's own units and genomics_1kg_eqtl.py's precedent. Both
    fractions are reported in the JSON regardless of which rule uses which."""
    reasons = []

    gate_pass = gate["passes"]
    reasons.append(f"gate: {'PASS' if gate_pass else 'FAIL'} (same_sign={gate['same_sign']}, "
                    f"ratio={gate['ratio']:.3f} {'>=' if gate['ratio']>=0.5 else '<'} 0.5)")

    r = main_run["results"]
    best_conv = max(max(v["r2"] for k, v in r.items() if k.startswith("Lasso")),
                     max(v["r2"] for k, v in r.items() if k.startswith("PCA")),
                     max(v["r2"] for k, v in r.items() if k.startswith("Random Forest")))
    hb_r2 = max(v["r2"] for k, v in r.items() if k.startswith("hierboost"))
    naive_r2 = r["naive (train mean)"]["r2"]
    gene_retention_frac = main_run["hierboost_gene_retention_frac"]

    close_to_best = (best_conv - hb_r2) <= 0.05
    sparse_enough = gene_retention_frac <= 0.5
    rule2a = close_to_best and sparse_enough

    beats_naive_by_margin = (hb_r2 - naive_r2) >= 0.05
    mech = main_run["mechanism_check"]
    # theoretically-expected direction: blocks with a HIGHER fraction of
    # hypothesized-TMB-relevant genes (proliferation/DNA-repair/immune) should show
    # HIGHER posterior inclusion confidence -> positive spearman rho
    mechanism_validated = mech["spearman_rho"] > 0.2 and mech["spearman_p"] < 0.05
    rule2b = beats_naive_by_margin and mechanism_validated

    reasons.append(f"2a: close_to_best_conventional={close_to_best} (best_conv_r2={best_conv:.4f}, "
                    f"hb_r2={hb_r2:.4f}, gap={best_conv-hb_r2:.4f}) AND "
                    f"sparse<=50%_of_raw_genes={sparse_enough} "
                    f"(retained {main_run['hierboost_n_genes_retained']}/{main_run['p_raw_genes']}"
                    f"={gene_retention_frac:.1%}) -> {rule2a}")
    reasons.append(f"2b: beats_naive_by_margin={beats_naive_by_margin} (hb_r2={hb_r2:.4f} vs "
                    f"naive_r2={naive_r2:.4f}, margin={hb_r2-naive_r2:.4f}) AND "
                    f"mechanism_validated={mechanism_validated} "
                    f"(spearman rho={mech['spearman_rho']:+.3f}, p={mech['spearman_p']:.3g}) -> {rule2b}")

    worse_than_naive = hb_r2 < naive_r2
    sparsity_never_engages = sweep_max_frac > 0.9 and not mechanism_validated

    reasons.append(f"FAIL-2: hierboost worse than naive: {worse_than_naive} "
                    f"(hb_r2={hb_r2:.4f} vs naive_r2={naive_r2:.4f})")
    reasons.append(f"FAIL-3: sparsity never engages (max block-retention frac across kappa "
                    f"sweep={sweep_max_frac:.1%} > 90%) AND no mechanism validation: {sparsity_never_engages}")

    if not gate_pass:
        verdict = "FAIL"
        reasons.append("-> FAIL rule 1 triggered (gate failed): verdict is FAIL regardless of downstream fit.")
    elif worse_than_naive or sparsity_never_engages:
        verdict = "FAIL"
        reasons.append("-> a FAIL condition (2 or 3) is triggered despite the gate passing: verdict is FAIL.")
    elif rule2a or rule2b:
        verdict = "WORK"
        reasons.append("-> gate passed AND (2a or 2b) satisfied, no FAIL condition triggered: verdict is WORK.")
    else:
        verdict = "MIXED"
        reasons.append("-> gate passed, no FAIL condition triggered, but neither 2a nor 2b cleanly satisfied: MIXED.")

    return verdict, reasons, dict(best_conventional_r2=float(best_conv), hierboost_r2=float(hb_r2),
                                    naive_r2=float(naive_r2), gene_retention_frac=float(gene_retention_frac))


def main():
    print("\n" + "=" * 88)
    print("STEP 0: fetch + build dataset (cBioPortal REST API, open-access)")
    print("=" * 88)
    X, y, genes, sample_ids, y_raw_tmb = build_dataset()

    print("\n" + "=" * 88)
    print("STEP 1: GENERALIZATION GATE (run first, per project discipline)")
    print("=" * 88)
    gate = check_generalization(X, y, genes, label="W4 TCGA-BRCA co-expression -> TMB")

    print("\n" + "=" * 88)
    print("STEP 2: hierboost (co-expression blocking) vs. baseline suite, held-out split")
    print("=" * 88)
    main_run = run_main(X, y, genes, rho=RHO_MAIN)

    print("\n" + "=" * 88)
    print("STEP 3: sparsity-engagement sweep (FAIL-rule check #3)")
    print("=" * 88)
    sweep, sweep_max_frac = sparsity_sweep(X, y, rho=RHO_MAIN)

    print("\n" + "=" * 88)
    print("STEP 3b: blocking-rho sensitivity (transparency, not part of the outcome rule)")
    print("=" * 88)
    rho_sensitivity = blocking_rho_sensitivity(X, y)

    print("\n" + "=" * 88)
    print("STEP 4: apply the preregistered WORK/FAIL/MIXED rule to the numbers above")
    print("=" * 88)
    verdict, reasons, verdict_numbers = apply_outcome_rule(gate, main_run, sweep_max_frac)
    for line in reasons:
        print(" -", line)
    print(f"\n>>> W4 (gene co-expression predicting a real clinical trait) VERDICT: {verdict}  "
          f"(preregistered prediction was WORK)")

    out = dict(
        domain="W4: gene co-expression predicting a real clinical trait",
        data_source=dict(
            api="https://www.cbioportal.org/api", study=STUDY, expression_profile=EXPR_PROFILE,
            sample_list=SAMPLE_LIST, clinical_attribute=CLINICAL_ATTR,
            note=("cBioPortal public REST API, open-access tier only, verified live 2026-08-30. "
                  "TCGA-BRCA PanCancer Atlas chosen for its large N (1082 samples with complete "
                  "RNA-seq over the panel, 1064 overlapping with non-missing TMB) and because "
                  "TMB_NONSYNONYMOUS is a real, independently-assayed (not expression-derived) "
                  "quantitative molecular variable -- unlike the cohort's hypoxia/ESTIMATE-style "
                  "scores, which are themselves expression-derived and would make this circular."),
        ),
        gene_panel=dict(groups={k: v for k, v in GENE_GROUPS.items()}, n_genes=len(ALL_GENES),
                         rationale="see module docstring; groups fixed before any TMB correlation was computed"),
        n_samples=int(X.shape[0]), n_genes=int(X.shape[1]),
        target_transform="log10(TMB_NONSYNONYMOUS + 1)",
        generalization_gate=gate,
        main_run=main_run,
        sparsity_sweep=sweep,
        sparsity_sweep_max_retained_frac=sweep_max_frac,
        blocking_rho_sensitivity=rho_sensitivity,
        verdict=verdict,
        verdict_reasoning=reasons,
        verdict_numbers=verdict_numbers,
        preregistered_prediction="WORK",
    )
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[main] results saved -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
