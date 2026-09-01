"""Practitioner-facing fit/predict/summary/plot API over hierboost's building blocks --
the layer that turns "load six modules and wire them together yourself" into an
sklearn/statsmodels-shaped `fit(X, y)`, `predict(X_new)`, `summary()`, `plot_*()`.

Composes, in order:
  1. Boosting-prior structure (optional): raw features -> group affinity -> relevance ->
     inclusion-prior boost `wr` (hierboost.kernels.resolve_affinity, with auto bandwidth
     fitting). Omit entirely (default) for plain, non-boosted spike-and-slab.
  2. Block-latent decorrelation (optional, `decorrelate="sar"|"ar1"|"star"`): raw
     features -> one shared latent per block, via Chapter 4's SAR mechanism (spatial),
     its AR(1) counterpart (temporal), or "star" (hierboost.spacetime: the two combined
     -- one VAR(1) latent trajectory PER block, all blocks coupled jointly through one
     structured, SAR-weighted transition matrix instead of independent AR(1)s) --
     hierboost.factor/state_space/spacetime for continuous outcomes, hierboost.latent
     for discrete/binomial ones (sar/ar1 only; "star" has no discrete/JAX counterpart
     yet). Blocks either supplied (`block_id`) or determined data-driven
     (hierboost.structure.determine_blocks). Optional `marginal="copula"` (continuous
     branch only): each block's raw features are passed through their own fitted
     empirical marginal CDF onto a shared Gaussian scale first (hierboost.copula) --
     the Gaussian-copula/nonparanormal generalization for blocks whose members have
     genuinely different marginal shapes (skewed, heavy-tailed, count-like) but should
     still share one latent under a common correlation structure. factor.py/
     state_space.py/spacetime.py themselves are unchanged; they just receive the
     copula-scale features instead of raw ones.
  3. Spike-and-slab GLM fit on whatever design matrix step 2 produced (raw X if no
     decorrelation) -- hierboost.spike_slab / spike_slab_gaussian, via EM, EM-filtering,
     or Gibbs.

HierBoostClassifier is step 3 with a Binomial/Bernoulli response; HierBoostRegressor is
step 3 with a Gaussian response. Both share _HierBoostBase for steps 1-2 and for
predict/summary/plot, since none of that differs by response family except the link
function and which fit_* engine gets called.
"""
from dataclasses import dataclass
import inspect
import pickle
import numpy as np
from scipy.special import expit
from scipy.stats import norm as _norm

from .kernels import resolve_affinity, sar_weight_matrix
from .structure import determine_blocks
from .blocks import block_membership_lists
from .spike_slab import (fit_em, em_filter, gibbs_sampler, centroid_estimate, embfdr,
                          ppl as ppl_binomial, _precision_from_theta)
from .spike_slab_gaussian import fit_em_gaussian, em_filter_gaussian, gibbs_sampler_gaussian, ppl_gaussian
from .spike_slab_glm import (fit_em_poisson, em_filter_poisson, ppl_poisson,
                              fit_em_nb, em_filter_nb, gibbs_sampler_nb, ppl_nb)
from .factor import gaussian_block_factor, project_block_factor
from .state_space import fit_temporal_block_factor, filter_temporal_block_factor
from .copula import fit_block_transforms, apply_block_transforms


@dataclass
class PredictionResult:
    mean: np.ndarray
    std: np.ndarray = None
    lower: np.ndarray = None
    upper: np.ndarray = None

    def __array__(self, dtype=None):
        return np.asarray(self.mean, dtype=dtype)


class _HierBoostBase:
    _response = None  # "binomial" or "gaussian", set by subclass

    def __init__(self, xi0=-2.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                 fit_method="em", decorrelate=None, block_method="threshold",
                 zeta=None, rho_threshold=0.3, tau2=1.0, rank=None,
                 n_samples=2000, burn_in=500, thin=1,
                 filter_frac=0.25, min_features=10, max_outer=200, patience=3,
                 random_state=None, phi_star=None, marginal=None):
        self.xi0, self.xi1, self.kappa, self.nu, self.lam = xi0, xi1, kappa, nu, lam
        self.fit_method = fit_method
        self.decorrelate = decorrelate
        self.block_method = block_method
        self.zeta = zeta
        self.rho_threshold = rho_threshold
        self.tau2 = tau2
        self.rank = rank
        self.phi_star = phi_star
        self.marginal = marginal
        self.n_samples, self.burn_in, self.thin = n_samples, burn_in, thin
        self.filter_frac, self.min_features = filter_frac, min_features
        self.max_outer, self.patience = max_outer, patience
        self.random_state = random_state
        self.fitted_ = False

    # ---- sklearn-style parameter interop ------------------------------------------

    def get_params(self, deep=True):
        """Constructor parameters as a dict, walking the full MRO's `__init__` signatures
        (not just the concrete class's) so subclasses that forward `**kwargs` to
        `_HierBoostBase.__init__` (e.g. HierBoostCountRegressor's `family`/`r_init` on top
        of the shared xi0/xi1/... set) still round-trip completely. Enables
        `sklearn.base.clone()` and is what `hierboost.model_selection.cross_val_score`
        uses to build a fresh, identically-configured estimator per fold. Not a full
        sklearn Estimator (no `.score()`/Classifier-Regressor-Mixin -- see
        hierboost.model_selection's docstring for why `.fit()`'s extra coords/block_id
        kwargs and `.predict()`'s natural-scale-mean return make that a poor fit here)."""
        names = set()
        for klass in type(self).__mro__:
            if "__init__" not in klass.__dict__:
                continue
            sig = inspect.signature(klass.__dict__["__init__"])
            names.update(p.name for p in sig.parameters.values()
                         if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL))
        return {name: getattr(self, name) for name in sorted(names) if hasattr(self, name)}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                raise ValueError(f"invalid parameter {k!r} for {type(self).__name__}")
            setattr(self, k, v)
        return self

    # ---- persistence ----------------------------------------------------------------

    def save(self, path):
        """Persist a fitted model (pickle) -- configuration plus every fitted attribute
        needed to predict/summarize/plot after reloading. A binomial+decorrelate fit
        carries JAX arrays internally (hierboost.latent), so load it back in the same
        kind of environment (.venv-jax) that produced it. Pickle can execute arbitrary
        code on load -- only load files you (or someone you trust) saved."""
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Inverse of .save()."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a fitted {cls.__name__} "
                             f"(got {type(obj).__name__})")
        return obj

    # ---- bootstrap uncertainty ------------------------------------------------------

    def bootstrap_ci(self, X, y, coords=None, block_id=None, n_trials=1, offset=None,
                      group_l=None, group_r=None, group_coords=None, group_relevance=None,
                      affinity_kind="gaussian", affinity_bandwidth=None,
                      n_boot=200, ci=0.95, random_state=None):
        """Percentile-bootstrap interval for each retained feature/block's coefficient,
        refitting the ENTIRE pipeline (decorrelation + spike-and-slab) on each resample
        of individuals -- pass this the same (X, y, coords, block_id, ...) already used
        for `.fit()` (call `.fit()` first; this reuses `self.names_`/`get_params()` for
        alignment and configuration).

        Why this exists (see project memory for the full diagnosis): `.summary()`'s
        Laplace-approximation CIs, even with `_compute_approx_covariance`'s errors-in-
        variables correction (closed-form for all three of "sar"/"ar1"/"star", scalar
        for the former and per-row/heteroscedastic for the latter two), only partially fix a real,
        calibration-tested undercoverage problem -- the classical "generated
        regressors" issue (Pagan 1984): a block's raw features are used TWICE, once to
        estimate the block-latent factor's loadings and once (via the fitted factor
        score) to estimate that block's outcome coefficient, and the closed-form
        correction only accounts for the score's uncertainty GIVEN fixed loadings, not
        the loadings' OWN estimation error. Refitting the whole pipeline (loadings AND
        gamma together) on each bootstrap resample captures both sources at once,
        without needing a bespoke analytic correction per `decorrelate`/response
        combination. Confirmed via calibration testing to close most of the remaining
        gap the closed-form correction left (88% vs 63% causal coverage against a 95%
        nominal target, same test setup) -- still the recommended option whenever
        `decorrelate` is set and the reported uncertainty needs to be taken seriously,
        at the cost of `n_boot` full refits instead of one.

        Not a fix for em_filter's post-selection inference subtlety: if a bootstrap
        resample's own em_filter run drops a feature/block the full-data fit retained,
        that resample contributes NaN for it (nanpercentile skips these) rather than a
        fabricated value -- coverage for a marginally-retained feature will reflect
        fewer effective bootstrap draws than `n_boot`.

        Returns a dict: `names` (list, same order as `self.names_`), `lo`/`hi`
        (length-matching arrays, the percentile interval), `mean`/`std` (bootstrap
        mean/SD per name), `samples` ((n_boot, len(names)) array, NaN where that
        resample's own fit didn't retain the name).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        n_trials_arr = np.asarray(n_trials) if hasattr(n_trials, "__len__") else None
        offset_arr = np.asarray(offset, dtype=float) if offset is not None else None

        names = list(self.names_)
        name_to_pos = {nm: j for j, nm in enumerate(names)}
        samples = np.full((n_boot, len(names)), np.nan)

        params = self.get_params()
        for bi in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                boot_model = type(self)(**params)
                boot_model.fit(
                    X[idx], y[idx], coords=coords, block_id=block_id,
                    n_trials=n_trials_arr[idx] if n_trials_arr is not None else n_trials,
                    offset=offset_arr[idx] if offset_arr is not None else None,
                    group_l=group_l, group_r=group_r, group_coords=group_coords,
                    group_relevance=group_relevance, affinity_kind=affinity_kind,
                    affinity_bandwidth=affinity_bandwidth)
                boot_beta = boot_model.beta_[1:]
                for nm, val in zip(boot_model.names_, boot_beta):
                    if nm in name_to_pos:
                        samples[bi, name_to_pos[nm]] = val
            except Exception:
                continue

        lo = np.nanpercentile(samples, 100 * (1 - ci) / 2, axis=0)
        hi = np.nanpercentile(samples, 100 * (1 + ci) / 2, axis=0)
        return dict(names=names, lo=lo, hi=hi,
                    mean=np.nanmean(samples, axis=0), std=np.nanstd(samples, axis=0),
                    samples=samples)

    # ---- fitting ----------------------------------------------------------------

    def fit(self, X, y, coords=None, block_id=None, n_trials=1, offset=None,
            group_l=None, group_r=None, group_coords=None, group_relevance=None,
            affinity_kind="gaussian", affinity_bandwidth=None, feature_names=None,
            verbose=False):
        """`coords` plays two independent roles, matching dissertation Ch4 vs Ch2/5:
        it is the axis `group_l/group_r/group_coords` boost affinity is computed over
        (step 1), and, separately, the axis blocks are formed over for decorrelation
        (step 2) when `decorrelate` is set -- pass "sar" with physical position, "ar1"
        with a time/lag index, or "star" with physical position (rows must still be
        time-ordered like "ar1" -- "star" additionally uses `coords` to compute each
        block's spatial centroid for the cross-block coupling matrix, and `phi_star`
        controls that coupling kernel's bandwidth). Both may use the same `coords`
        (Ch5's combination) or only one may apply, depending on which optional
        arguments are supplied.

        `offset` (only meaningful for HierBoostCountRegressor) is a fixed, known
        per-observation term added to the linear predictor before the log link, e.g.
        log(population) for a rate model -- ignored by the Binomial/Gaussian subclasses.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p_raw = X.shape
        self.n_trials_ = n_trials
        self.offset_ = None if offset is None else np.asarray(offset, dtype=float)
        self.feature_names_in_ = list(feature_names) if feature_names is not None else \
            [f"x{j}" for j in range(p_raw)]
        self.coords_ = None if coords is None else np.asarray(coords, dtype=float)
        self.decorrelate_ = self.decorrelate
        self.marginal_ = self.marginal
        if self.marginal_ == "copula" and self._response == "binomial":
            raise NotImplementedError(
                "marginal='copula' is only implemented for the continuous factor.py/"
                "state_space.py/spacetime.py path -- HierBoostClassifier's Binomial response "
                "uses Chapter 4's discrete/JAX machinery (hierboost.latent), which assumes the "
                "raw features themselves are already Binomial (e.g. genotypes), not a "
                "continuous quantity to be copula-transformed. For per-feature parametric "
                "marginal families (e.g. Poisson/NegBinomial counts) instead of the default "
                "empirical/rank-based transform, call hierboost.copula directly, the way "
                "newsgroups_demo.py drops to the low-level functional API for a combination "
                "the high-level estimator doesn't cover.")

        have_groups = group_relevance is not None and (group_l is not None or group_coords is not None)
        if have_groups:
            wr_raw = resolve_affinity(self.coords_, kind=affinity_kind, group_l=group_l, group_r=group_r,
                                       group_coords=group_coords, relevance=group_relevance,
                                       bandwidth=affinity_bandwidth, X=X)
        else:
            wr_raw = np.zeros(p_raw)

        if self.decorrelate is None:
            self.block_id_, self.block_ids_, self.latent_fits_ = None, None, None
            result = self._fit_glm(np.column_stack([np.ones(n), X]), y, wr_raw, n_trials,
                                    offset=self.offset_)
            self._store_common(np.column_stack([np.ones(n), X]), y, wr_raw, result,
                                list(self.feature_names_in_))
            self.fitted_ = True
            return self

        if self.coords_ is None:
            raise ValueError("decorrelate requires coords (physical position for 'sar', "
                              "time/lag order for 'ar1', physical position for 'star')")
        if block_id is None:
            block_id = determine_blocks(X=X, coords=self.coords_, method=self.block_method,
                                         zeta=self.zeta, rho=self.rho_threshold)
        self.block_id_ = np.asarray(block_id)
        blocks = block_membership_lists(self.block_id_)
        self.block_ids_ = sorted(blocks.keys())
        K = len(self.block_ids_)
        block_names = [f"block{b}" for b in self.block_ids_]
        wr_block = np.array([wr_raw[blocks[b]].mean() if blocks[b].size else 0.0
                              for b in self.block_ids_])

        if self.decorrelate == "star" and self._response == "binomial":
            raise NotImplementedError(
                "decorrelate='star' (space x time block-latent) is only implemented for "
                "the continuous factor.py/state_space.py path (Gaussian, Poisson, "
                "Negative-Binomial) -- HierBoostClassifier's Binomial response needs "
                "Chapter 4's literal discrete/JAX machinery (hierboost.latent), which "
                "does not yet have a spatiotemporal option.")

        if self._response == "binomial":
            if self.fit_method != "em":
                raise ValueError("decorrelate with a binomial response always uses Chapter 4's "
                                  "EM-based pipeline internally (fit_latent_block_model); "
                                  "fit_method must be 'em' in this configuration")
            try:
                from .latent import fit_latent_block_model
            except ImportError as e:
                raise ImportError(
                    "decorrelate with a binomial response needs JAX (hierboost.latent) -- "
                    "install it in a separate environment, see README ('pip install jax jaxlib') "
                    "and run from there") from e
            result = fit_latent_block_model(X, y, self.coords_, self.block_id_, wr_block,
                                             self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                                             n_trials=n_trials, tau2=self.tau2,
                                             structure=self.decorrelate, verbose=verbose)
            self.latent_fits_ = result["fits"]
            self.latent_deltas_ = result["deltas"]
            em = result["em_result"]
            X_design = np.column_stack([np.ones(n), result["Z"]])
            glm_result = dict(kind="em", beta=em.beta, sigma2=em.sigma2, theta_hat=em.theta_hat,
                               mu=em.mu, ppl=ppl_binomial(y, em.mu), history=None,
                               retained_idx=None, gibbs=None)
        else:
            # Continuous-feature block-latent path (factor.py/state_space.py, no JAX):
            # response-agnostic, since it only ever transforms the raw X into a shared
            # per-block latent Z and never touches y -- works unchanged for Gaussian,
            # Poisson, and Negative-Binomial response families. Only the Binomial branch
            # above needs Chapter 4's literal discrete/JAX machinery (the raw features
            # themselves are assumed Binomial there, e.g. genotypes).
            Z = np.zeros((n, K))
            fits, train_means = {}, {}
            self.latent_zvar_ = {}
            self.marginal_transforms_ = {} if self.marginal_ == "copula" else None
            if self.decorrelate == "star":
                # Joint across ALL multi-member blocks at once (Phi couples them) --
                # structurally different from sar/ar1 below, which fit each block
                # independently. Singleton blocks still get the same plain
                # center-and-pass-through treatment as the other two structures.
                from .spacetime import fit_spacetime_block_factor
                multi_ids = [b for b in self.block_ids_ if blocks[b].size > 1]
                single_ids = [b for b in self.block_ids_ if blocks[b].size == 1]
                for b in single_ids:
                    Xb = X[:, blocks[b]]
                    if self.marginal_ == "copula":
                        transforms = fit_block_transforms(Xb, kind="empirical")
                        self.marginal_transforms_[b] = transforms
                        Xb = apply_block_transforms(Xb, transforms)
                    train_means[b] = Xb[:, 0:1].mean(axis=0)
                    fits[b] = ("single",)
                    self.latent_zvar_[b] = 0.0
                self.star_model_ = None
                if multi_ids:
                    X_by_block = {b: X[:, blocks[b]] for b in multi_ids}
                    if self.marginal_ == "copula":
                        for b in multi_ids:
                            transforms = fit_block_transforms(X_by_block[b], kind="empirical")
                            self.marginal_transforms_[b] = transforms
                            X_by_block[b] = apply_block_transforms(X_by_block[b], transforms)
                    centroids = np.array([self.coords_[blocks[b]].mean(axis=0) for b in multi_ids])
                    phi_star = self.phi_star
                    if phi_star is None:
                        if centroids.ndim == 1:
                            gaps = np.diff(np.sort(centroids))
                            phi_star = float(np.median(gaps)) if np.any(gaps > 0) else 1.0
                        else:
                            # multi-D coords: "consecutive gap along a sorted axis" has no
                            # meaning, so fall back to the median pairwise distance instead
                            # (same default uk_weather_star_demo.py's median_gap uses).
                            dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
                            iu = np.triu_indices(len(centroids), k=1)
                            offdiag = dists[iu]
                            phi_star = float(np.median(offdiag[offdiag > 0])) if np.any(offdiag > 0) else 1.0
                    W = sar_weight_matrix(centroids, phi_star)
                    star_fit = fit_spacetime_block_factor(X_by_block, multi_ids, W)
                    for k, b in enumerate(multi_ids):
                        fits[b] = ("star", k)
                        # Heteroscedastic errors-in-variables correction (see the ar1
                        # branch below for the full derivation this mirrors):
                        # fit_spacetime_block_factor's z_cov_diag is (T, K), one
                        # per-timestep posterior variance per block, row-aligned with
                        # the T rows used to fit the outcome regression -- store block
                        # b's own column (a length-T vector, not a scalar) so
                        # _compute_approx_covariance can apply it per row instead of
                        # silently dropping it to 0.0.
                        self.latent_zvar_[b] = star_fit.z_cov_diag[:, k]
                    self.star_model_ = dict(
                        block_ids=multi_ids, loadings=star_fit.loadings, obs_var=star_fit.obs_var,
                        train_mean=star_fit.train_mean, Phi=star_fit.Phi, Q=star_fit.Q,
                        rho1=star_fit.rho1, rho2=star_fit.rho2, W=W, phi_star=phi_star,
                        z_smooth=star_fit.z, z_cov_diag=star_fit.z_cov_diag)
                for k, b in enumerate(self.block_ids_):
                    if b in single_ids:
                        Xb = X[:, blocks[b]]
                        if self.marginal_ == "copula":
                            Xb = apply_block_transforms(Xb, self.marginal_transforms_[b])
                        Z[:, k] = Xb[:, 0] - train_means[b][0]
                    else:
                        kk = self.star_model_["block_ids"].index(b)
                        Z[:, k] = self.star_model_["z_smooth"][:, kk]
            else:
                for k, b in enumerate(self.block_ids_):
                    idx = blocks[b]
                    Xb = X[:, idx]
                    if self.marginal_ == "copula":
                        transforms = fit_block_transforms(Xb, kind="empirical")
                        self.marginal_transforms_[b] = transforms
                        Xb = apply_block_transforms(Xb, transforms)
                    if len(idx) == 1:
                        train_means[b] = Xb[:, 0:1].mean(axis=0)
                        Z[:, k] = Xb[:, 0] - train_means[b][0]
                        fits[b] = ("single",)
                        self.latent_zvar_[b] = 0.0  # raw feature, not estimated -- no measurement error
                    elif self.decorrelate == "sar":
                        scores, loadings, z_var = gaussian_block_factor(Xb, return_variance=True)
                        train_means[b] = Xb.mean(axis=0)
                        Z[:, k] = scores
                        fits[b] = ("sar", loadings)
                        self.latent_zvar_[b] = z_var
                    elif self.decorrelate == "ar1":
                        res_ts = fit_temporal_block_factor(Xb)
                        train_means[b] = res_ts.train_mean
                        Z[:, k] = res_ts.z
                        fits[b] = ("ar1", res_ts)
                        # Heteroscedastic errors-in-variables correction: unlike sar's
                        # gaussian_block_factor (one scalar PPCA posterior variance
                        # shared by every row), fit_temporal_block_factor's RTS-smoother
                        # posterior variance res_ts.z_var is PER-TIMESTEP (shape (n,)),
                        # naturally row-aligned with the individuals used to fit the
                        # outcome regression (each row here is one timestep/individual,
                        # e.g. a trading day). Store the full per-row vector -- not a
                        # block-mean placeholder -- so _compute_approx_covariance can
                        # inflate each row's residual variance by its OWN measurement-
                        # error contribution instead of one global scalar (see there).
                        self.latent_zvar_[b] = res_ts.z_var
                    else:
                        raise ValueError(f"unknown decorrelate={self.decorrelate!r}; "
                                          f"choose 'sar', 'ar1', or 'star'")
            self.latent_fits_ = fits
            self.train_mean_ = train_means
            X_design = np.column_stack([np.ones(n), Z])
            glm_result = self._fit_glm(X_design, y, wr_block, n_trials, offset=self.offset_)

        self._store_common(X_design, y, wr_block, glm_result, block_names)
        self.fitted_ = True
        return self

    def _store_common(self, X_design, y, wr, result, names):
        retained_idx = result.get("retained_idx")
        self.retained_idx_ = retained_idx
        n_rows = X_design.shape[0]
        # (n_rows, n_names) errors-in-variables measurement-error variance, one column
        # per block/feature -- see _compute_approx_covariance for how this feeds the
        # Gaussian-response residual-variance correction. sar blocks and raw/singleton
        # pass-through features contribute a homoscedastic scalar (factor.py's PPCA
        # posterior variance is the same for every row) that numpy broadcasts across
        # the column; ar1/star blocks instead contribute a genuinely per-row vector
        # (state_space.py/spacetime.py's RTS-smoother posterior variance is
        # per-timestep) -- both cases are handled uniformly here since assigning a
        # scalar or a length-n_rows array into a column both just work.
        if self.decorrelate_ is not None and hasattr(self, "latent_zvar_"):
            z_var_full = np.zeros((n_rows, len(names)))
            for j, b in enumerate(self.block_ids_):
                z_var_full[:, j] = self.latent_zvar_.get(b, 0.0)
        else:
            z_var_full = np.zeros((n_rows, len(names)))
        if retained_idx is not None:
            X_design = X_design[:, np.concatenate([[0], retained_idx + 1])]
            wr = wr[retained_idx]
            names = [names[j] for j in retained_idx]
            z_var_full = z_var_full[:, retained_idx]
        self.z_var_ = z_var_full
        self.X_design_ = X_design
        self.y_ = y
        self.wr_ = wr
        self.names_ = names
        self.beta_ = result["beta"]
        self.sigma2_ = result["sigma2"]
        self.sigma_y2_ = result.get("sigma_y2")
        self.r_ = result.get("r")
        self.theta_hat_ = result["theta_hat"]
        eta_recompute = X_design @ result["beta"]
        if self.offset_ is not None:
            eta_recompute = eta_recompute + self.offset_
        self.mu_ = result["mu"] if result["mu"] is not None else self._link_inv(eta_recompute)
        self.ppl_ = result["ppl"]
        self.fit_kind_ = result["kind"]
        self.history_ = result.get("history")
        self.gibbs_ = result.get("gibbs")
        self.n_ = X_design.shape[0]
        self._compute_approx_covariance()

    def _compute_approx_covariance(self):
        X_design = self.X_design_
        if self.gibbs_ is not None:
            self.beta_cov_ = np.cov(self.gibbs_.beta, rowvar=False)
            self.beta_se_ = self.gibbs_.beta.std(axis=0)
            return
        precision = _precision_from_theta(self.theta_hat_, self.kappa)
        sigma_diag = self.sigma2_ / precision
        if self._response == "binomial":
            W = np.clip(self.mu_ * (1.0 - self.mu_), 1e-6, None)
        elif self._response == "gaussian":
            # Errors-in-variables correction (Pagan 1984's "generated regressors" problem,
            # confirmed via calibration testing -- see project memory): self.z_var_ (0 for
            # raw/singleton features; factor.py's closed-form posterior variance for
            # decorrelate="sar" blocks; state_space.py's/spacetime.py's per-timestep RTS-
            # smoother posterior variance for "ar1"/"star" blocks) is the block-latent's
            # OWN estimation uncertainty, which the naive sigma_y2_-only residual variance
            # ignores entirely. Standard measurement-error result (Fuller 1987; Carroll,
            # Ruppert, Stefanski & Crainiceanu, "Measurement Error in Nonlinear Models"):
            # if y_i = beta0 + sum_b gamma_b*Z_b(i) + noise and Z_b(i) is observed as a
            # noisy Z_hat_b(i) with Var(Z_b(i) - Z_hat_b(i)) = z_var_b(i), the correctly-
            # specified residual variance for y_i | Z_hat is
            # sigma_y2 + sum_b gamma_b^2 * z_var_b(i) -- a per-ROW quantity in general,
            # not one global scalar, since ar1/star's z_var genuinely varies over time
            # (see the .fit() branches building self.latent_zvar_). self.z_var_ is
            # (n_rows, n_names): sar/raw columns hold a value broadcast identically to
            # every row (factor.py's PPCA posterior variance has no per-row structure),
            # so for a sar-only (or undecorrelated) fit this reduces EXACTLY to the
            # original scalar sigma_y2_eff; ar1/star columns instead vary row to row, so
            # the resulting weight vector genuinely downweights individuals whose
            # block-latent estimate was less certain at that particular timestep. Only
            # for the Gaussian response (the analogous correction for Binomial/Poisson/
            # NegBinomial would need a delta-method extension in the linear-predictor
            # scale, not yet derived).
            #
            # Honest status (see also bootstrap_ci's docstring and the calibration test
            # in test_estimator.py): for "sar" this closed-form correction alone only
            # moved coverage from ~59% to ~63% against a 95% nominal target -- it fixes
            # the score's uncertainty given fixed loadings but not the loadings' own
            # estimation error, so it is a real but incomplete fix on its own. This ar1/
            # star extension closes the analogous, previously-completely-skipped gap
            # (z_var was hardcoded to 0.0, i.e. no correction at all) by the same
            # mechanism and is expected to be similarly partial for the same reason --
            # `.bootstrap_ci()` remains the recommended tool for calibrated uncertainty
            # under any `decorrelate` setting, not this closed-form correction.
            meas_error_var = self.z_var_ @ (self.beta_[1:] ** 2)
            sigma_y2_eff = self.sigma_y2_ + meas_error_var
            W = 1.0 / sigma_y2_eff
        elif self._response == "poisson":
            W = np.clip(self.mu_, 1e-6, None)
        else:  # negbinomial
            W = np.clip(self.mu_ * self.r_ / (self.r_ + self.mu_), 1e-6, None)
        XtWX = X_design.T @ (W[:, None] * X_design)
        try:
            cov = np.linalg.inv(XtWX + np.diag(1.0 / sigma_diag))
        except np.linalg.LinAlgError:
            cov = np.full((X_design.shape[1], X_design.shape[1]), np.nan)
        self.beta_cov_ = cov
        self.beta_se_ = np.sqrt(np.clip(np.diag(cov), 0, None))

    # ---- prediction ---------------------------------------------------------------

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("call .fit(...) before predicting/summarizing/plotting")

    def _transform_new_blocks(self, X_new):
        blocks = block_membership_lists(self.block_id_)
        Z_new = np.zeros((X_new.shape[0], len(self.block_ids_)))
        if self.decorrelate_ == "star":
            # Joint across all multi-member blocks at once, mirroring .fit()'s joint
            # treatment -- can't be done block-by-block since Phi couples them.
            for k, b in enumerate(self.block_ids_):
                if self.latent_fits_[b] == ("single",):
                    Xb_new = X_new[:, blocks[b]]
                    if self.marginal_ == "copula":
                        Xb_new = apply_block_transforms(Xb_new, self.marginal_transforms_[b])
                    Z_new[:, k] = Xb_new[:, 0] - self.train_mean_[b][0]
            if self.star_model_ is not None:
                from .spacetime import filter_spacetime_block_factor
                sm = self.star_model_
                X_by_block_new = {b: X_new[:, blocks[b]] for b in sm["block_ids"]}
                if self.marginal_ == "copula":
                    X_by_block_new = {b: apply_block_transforms(Xb, self.marginal_transforms_[b])
                                       for b, Xb in X_by_block_new.items()}
                z_new, _, _ = filter_spacetime_block_factor(
                    X_by_block_new, sm["block_ids"], sm["loadings"], sm["obs_var"],
                    sm["Phi"], sm["Q"], sm["train_mean"])
                for kk, b in enumerate(sm["block_ids"]):
                    Z_new[:, self.block_ids_.index(b)] = z_new[:, kk]
            return Z_new
        for k, b in enumerate(self.block_ids_):
            Xb_new = X_new[:, blocks[b]]
            if self._response != "binomial":
                # continuous factor.py/state_space.py path (Gaussian, Poisson, NB)
                if self.marginal_ == "copula":
                    Xb_new = apply_block_transforms(Xb_new, self.marginal_transforms_[b])
                kind, *payload = self.latent_fits_[b]
                if kind == "single":
                    Z_new[:, k] = Xb_new[:, 0] - self.train_mean_[b][0]
                elif kind == "sar":
                    (loadings,) = payload
                    Z_new[:, k] = project_block_factor(Xb_new, loadings, self.train_mean_[b])
                elif kind == "ar1":
                    (res_ts,) = payload
                    z_new, _, _ = filter_temporal_block_factor(
                        Xb_new, res_ts.loadings, res_ts.obs_var, res_ts.rho, res_ts.state_var,
                        res_ts.train_mean)
                    Z_new[:, k] = z_new
            else:
                import jax.numpy as jnp
                fit = self.latent_fits_[b]
                delta_b = self.latent_deltas_[b]
                zt_init = jnp.zeros(Xb_new.shape[0])
                zt = fit.infer_ztilde_from_data(jnp.asarray(Xb_new, dtype=jnp.float32), delta_b, zt_init)
                Z_new[:, k] = np.array(zt)
        return Z_new

    def _predict_eta(self, X_new, offset_new=None):
        if X_new is None:
            eta = self.X_design_ @ self.beta_
            if self.offset_ is not None:
                eta = eta + self.offset_
            return eta, self._eta_var(self.X_design_)
        X_new = np.asarray(X_new, dtype=float)
        Z_new = self._transform_new_blocks(X_new) if self.decorrelate_ is not None else X_new
        if self.retained_idx_ is not None:
            Z_new = Z_new[:, self.retained_idx_]
        if Z_new.shape[1] != len(self.names_):
            raise ValueError(f"expected {len(self.names_)} feature/block columns (after any "
                              f"em_filter dropping), got {Z_new.shape[1]} -- pass the same raw "
                              f"feature layout used for .fit()")
        X_design_new = np.column_stack([np.ones(Z_new.shape[0]), Z_new])
        eta = X_design_new @ self.beta_
        if offset_new is not None:
            eta = eta + np.asarray(offset_new, dtype=float)
        return eta, self._eta_var(X_design_new)

    def _eta_var(self, X_design):
        var = np.einsum("ij,jk,ik->i", X_design, self.beta_cov_, X_design)
        return np.sqrt(np.clip(var, 0.0, None))

    def _predict_common(self, X_new, return_std, ci, n_mc, offset_new=None):
        self._check_fitted()
        eta_mean, eta_sd = self._predict_eta(X_new, offset_new=offset_new)
        if not return_std:
            return self._link_inv(eta_mean)
        rng = np.random.default_rng(self.random_state)
        eta_samples = eta_mean[None, :] + rng.normal(size=(n_mc, eta_mean.shape[0])) * eta_sd[None, :]
        mu_samples = self._link_inv(eta_samples)
        lo, hi = np.percentile(mu_samples, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0)
        return PredictionResult(mean=self._link_inv(eta_mean), std=mu_samples.std(axis=0),
                                 lower=lo, upper=hi)

    # ---- summary --------------------------------------------------------------------

    def summary(self, top_n=15, ci=0.95, gamma=1.0):
        """A statsmodels-.summary()-style text report: fit configuration, overall fit
        quality (PPL, EMBFDR), and a table of features/blocks ranked by posterior
        inclusion probability with approximate coefficient CIs (Laplace approximation
        from the fitted EM precision, or the empirical Gibbs posterior when available)."""
        self._check_fitted()
        z = float(_norm.ppf((1 + ci) / 2))
        se = self.beta_se_
        lines = []
        rule = "=" * 72
        lines.append(rule)
        lines.append(f"{type(self).__name__} summary")
        lines.append(rule)
        resp_desc = self._response
        if self._response == "binomial":
            resp_desc += f" (n_trials={self.n_trials_})"
        elif self._response == "negbinomial":
            resp_desc += f" (r={self.r_:.3f})"
        lines.append(f"{'Response:':<22}{resp_desc}")
        if self._response in ("poisson", "negbinomial"):
            lines.append(f"{'Offset:':<22}{'yes' if self.offset_ is not None else 'none'}")
        lines.append(f"{'Fit method:':<22}{self.fit_kind_}")
        decor_desc = self.decorrelate_ or "none"
        if self.decorrelate_ is not None:
            decor_desc += f"  ({len(self.block_ids_)} blocks, block_method={self.block_method!r})"
            if self.marginal_ == "copula":
                decor_desc += ", marginal=copula"
        lines.append(f"{'Decorrelation:':<22}{decor_desc}")
        lines.append(f"{'Observations:':<22}n={self.n_}")
        lines.append(f"{'Features/blocks:':<22}{len(self.names_)}"
                      + (f" (of {len(self.feature_names_in_)} raw features)"
                         if self.decorrelate_ is None and len(self.names_) < len(self.feature_names_in_) else ""))
        lines.append("-" * 72)
        lines.append(f"{'Hyperparameters:':<22}xi0={self.xi0}  xi1={self.xi1}  kappa={self.kappa}  "
                      f"nu={self.nu}  lam={self.lam}")
        lines.append(f"{'sigma2 (slab var):':<22}{self.sigma2_:.4f}")
        if self._response == "gaussian":
            lines.append(f"{'sigma_y2 (noise var):':<22}{self.sigma_y2_:.4f}")
        lines.append(f"{'PPL:':<22}{self.ppl_:.4f}")
        bfdr = embfdr(self.theta_hat_, gamma=gamma)
        n_selected = int(centroid_estimate(self.theta_hat_, gamma=gamma).sum())
        lines.append(f"{'EMBFDR @ gamma={:.1f}:'.format(gamma):<22}{bfdr:.4f}")
        lines.append(f"{'Selected (centroid):':<22}{n_selected} / {len(self.theta_hat_)}")
        lines.append("-" * 72)

        order = np.argsort(self.theta_hat_)[::-1]
        shown = order[:top_n]
        header = f"{'name':<20}{'theta_hat':>10}{'coef':>10}{'se':>10}{'[' + f'{ci:.0%}':>7} CI]"
        lines.append(f"Top {len(shown)} of {len(self.theta_hat_)} features/blocks by inclusion probability:")
        lines.append(header)
        for j in shown:
            b, s = self.beta_[j + 1], se[j + 1]
            lo, hi = b - z * s, b + z * s
            lines.append(f"{self.names_[j]:<20}{self.theta_hat_[j]:>10.4f}{b:>10.4f}{s:>10.4f}"
                          f"  [{lo:>7.3f}, {hi:>7.3f}]")
        lines.append(rule)
        return "\n".join(lines)

    def describe_blocks(self, top_n=None):
        """Per-row block composition: for a plain (non-decorrelated) fit, one row per
        retained raw feature (a singleton "block" of one); for a decorrelated fit, one row
        per retained block listing which raw feature names actually landed in it. Closes
        the gap where block membership was otherwise only visible by going back to
        hierboost.blocks.block_membership_lists(model.block_id_) directly. Returns a list
        of dicts (ready for `pandas.DataFrame(model.describe_blocks())`, kept dependency-
        free here), sorted by inclusion probability (highest first); `top_n` truncates."""
        self._check_fitted()
        if self.decorrelate_ is None:
            rows = [dict(name=name, members=[name], theta_hat=float(self.theta_hat_[j]),
                         coef=float(self.beta_[j + 1]))
                    for j, name in enumerate(self.names_)]
        else:
            blocks = block_membership_lists(self.block_id_)
            retained_block_ids = (self.block_ids_ if self.retained_idx_ is None
                                   else [self.block_ids_[k] for k in self.retained_idx_])
            rows = [dict(name=name, members=[self.feature_names_in_[idx]
                                              for idx in blocks[retained_block_ids[j]]],
                         theta_hat=float(self.theta_hat_[j]), coef=float(self.beta_[j + 1]))
                    for j, name in enumerate(self.names_)]
        rows.sort(key=lambda r: r["theta_hat"], reverse=True)
        return rows[:top_n] if top_n is not None else rows

    # ---- plotting -------------------------------------------------------------------

    def plot_inclusion(self, top_n=20, ax=None):
        self._check_fitted()
        from .viz import plot_inclusion
        return plot_inclusion(self.theta_hat_, self.names_, top_n=top_n, ax=ax)

    def plot_coefficients(self, top_n=20, ci=0.95, ax=None):
        self._check_fitted()
        from .viz import plot_coefficients
        return plot_coefficients(self.beta_[1:], self.beta_se_[1:], self.names_, ci=ci, top_n=top_n, ax=ax)

    def plot_diagnostics(self, axes=None):
        self._check_fitted()
        from .viz import plot_diagnostics
        return plot_diagnostics(self.y_, self.mu_, self.history_, axes=axes)

    def plot_latent(self, block=None, ax=None):
        """Only meaningful when `decorrelate` was used: the fitted shared latent for one
        block -- an AR(1) trajectory with uncertainty (temporal), a loadings bar chart
        (spatial/SAR), a space-time trajectory annotated with the fitted own-persistence/
        spatial-coupling coefficients ('star'), or the fitted latent series (discrete/
        binomial, no closed-form variance available in that path)."""
        self._check_fitted()
        if self.decorrelate_ is None:
            raise RuntimeError("plot_latent needs a model fit with decorrelate='sar'/'ar1'/'star'")
        block = self.block_ids_[0] if block is None else block
        if self._response != "binomial":
            kind, *payload = self.latent_fits_[block]
            if kind == "ar1":
                from .viz import plot_temporal_latent
                (res_ts,) = payload
                return plot_temporal_latent(res_ts.z, res_ts.z_var, res_ts.rho, ax=ax)
            if kind == "sar":
                from .viz import plot_loadings
                (loadings,) = payload
                return plot_loadings(loadings, ax=ax)
            if kind == "star":
                from .viz import plot_temporal_latent
                (kk,) = payload
                sm = self.star_model_
                ax = plot_temporal_latent(sm["z_smooth"][:, kk], sm["z_cov_diag"][:, kk], ax=ax)
                ax.set_title(f"Space-time block-latent trajectory, block {block} "
                             f"(rho1={sm['rho1']:.3f} own-persistence, "
                             f"rho2={sm['rho2']:.3f} spatial coupling)")
                return ax
            raise RuntimeError(f"block {block} has a single member; nothing to decorrelate/plot")
        else:
            import matplotlib.pyplot as plt
            k = self.block_ids_.index(block)
            z = self.X_design_[:, 1 + k] if self.retained_idx_ is None else None
            if z is None:
                raise RuntimeError("this block was dropped by em_filter; nothing to plot")
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(z, color="teal", linewidth=1.0)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_title(f"Fitted discrete block latent (structure={self.decorrelate_!r}), block {block}")
            ax.set_xlabel("individual")
            ax.set_ylabel("latent state")
            return ax


class HierBoostClassifier(_HierBoostBase):
    """Binomial/Bernoulli response (n_trials=1). See _HierBoostBase for the shared
    boosting/decorrelation pipeline; this subclass supplies the logistic spike-and-slab
    GLM layer (hierboost.spike_slab)."""
    _response = "binomial"

    def _link_inv(self, eta):
        return expit(eta)

    def _fit_glm(self, X_design, y, wr, n_trials, offset=None):
        if n_trials != 1:
            raise ValueError(
                "the non-decorrelated spike-and-slab engine (hierboost.spike_slab) is "
                "Bernoulli-only; n_trials>1 (Binomial counts) is only supported with "
                "decorrelate='sar'/'ar1', which fits through hierboost.latent's actual "
                "Binomial likelihood instead")
        if self.fit_method == "em":
            res = fit_em(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam, rank=self.rank)
            return dict(kind="em", beta=res.beta, sigma2=res.sigma2, theta_hat=res.theta_hat,
                        mu=res.mu, ppl=ppl_binomial(y, res.mu), history=None, retained_idx=None, gibbs=None)
        if self.fit_method == "em_filter":
            filt = em_filter(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                              filter_frac=self.filter_frac, min_features=self.min_features,
                              max_outer=self.max_outer, rank=self.rank, patience=self.patience)
            best = filt.best
            return dict(kind="em_filter", beta=best.beta, sigma2=best.sigma2, theta_hat=best.theta_hat,
                        mu=None, ppl=best.ppl, history=filt.history, retained_idx=best.retained_idx, gibbs=None)
        if self.fit_method == "gibbs":
            gr = gibbs_sampler(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                                n_samples=self.n_samples, burn_in=self.burn_in, thin=self.thin,
                                rank=self.rank, seed=self.random_state)
            beta_mean = gr.beta.mean(axis=0)
            mu = expit(X_design @ beta_mean)
            return dict(kind="gibbs", beta=beta_mean, sigma2=gr.sigma2.mean(), theta_hat=gr.pi_hat,
                        mu=mu, ppl=ppl_binomial(y, mu), history=None, retained_idx=None, gibbs=gr)
        raise ValueError(f"unknown fit_method={self.fit_method!r}")

    def predict_proba(self, X_new=None, return_std=False, ci=0.95, n_mc=2000):
        """Per-trial probability, always in (0, 1) regardless of n_trials."""
        return self._predict_common(X_new, return_std, ci, n_mc)

    def predict(self, X_new=None, return_std=False, ci=0.95, n_mc=2000):
        """Mean response on the observation's natural scale: a probability if the model
        was fit with n_trials=1 (Bernoulli), an expected count if n_trials>1 (Binomial)."""
        result = self._predict_common(X_new, return_std, ci, n_mc)
        if self.n_trials_ == 1:
            return result
        scale = self.n_trials_
        if isinstance(result, PredictionResult):
            return PredictionResult(mean=result.mean * scale, std=result.std * scale,
                                     lower=result.lower * scale, upper=result.upper * scale)
        return result * scale

    def predict_label(self, X_new=None, threshold=0.5):
        """Hard 0/1 label at a probability threshold -- only meaningful for n_trials=1."""
        return (np.asarray(self.predict_proba(X_new)) >= threshold).astype(int)


class HierBoostRegressor(_HierBoostBase):
    """Gaussian response. See _HierBoostBase for the shared boosting/decorrelation
    pipeline; this subclass supplies the conjugate linear-Gaussian spike-and-slab GLM
    layer (hierboost.spike_slab_gaussian)."""
    _response = "gaussian"

    def _link_inv(self, eta):
        return eta

    def _fit_glm(self, X_design, y, wr, n_trials, offset=None):
        if self.fit_method == "em":
            res = fit_em_gaussian(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                                   rank=self.rank)
            return dict(kind="em", beta=res.beta, sigma2=res.sigma_g2, sigma_y2=res.sigma_y2,
                        theta_hat=res.theta_hat, mu=res.mu, ppl=ppl_gaussian(y, res.mu, res.sigma_y2),
                        history=None, retained_idx=None, gibbs=None)
        if self.fit_method == "em_filter":
            filt = em_filter_gaussian(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                                       filter_frac=self.filter_frac, min_features=self.min_features,
                                       max_outer=self.max_outer, rank=self.rank, patience=self.patience)
            best = filt.best
            return dict(kind="em_filter", beta=best.beta, sigma2=best.sigma_g2, sigma_y2=best.sigma_y2,
                        theta_hat=best.theta_hat, mu=None, ppl=best.ppl, history=filt.history,
                        retained_idx=best.retained_idx, gibbs=None)
        if self.fit_method == "gibbs":
            gr = gibbs_sampler_gaussian(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                                         n_samples=self.n_samples, burn_in=self.burn_in, thin=self.thin,
                                         rank=self.rank, seed=self.random_state)
            beta_mean = gr.beta.mean(axis=0)
            mu = X_design @ beta_mean
            sigma_y2 = gr.sigma_y2.mean()
            return dict(kind="gibbs", beta=beta_mean, sigma2=gr.sigma_g2.mean(), sigma_y2=sigma_y2,
                        theta_hat=gr.pi_hat, mu=mu, ppl=ppl_gaussian(y, mu, sigma_y2),
                        history=None, retained_idx=None, gibbs=gr)
        raise ValueError(f"unknown fit_method={self.fit_method!r}")

    def predict(self, X_new=None, return_std=False, ci=0.95, n_mc=2000):
        return self._predict_common(X_new, return_std, ci, n_mc)


class HierBoostCountRegressor(_HierBoostBase):
    """Poisson or Negative-Binomial response (log link) -- event counts (page views,
    disease/case counts, species observations, ...) instead of a continuous or binary
    outcome. See hierboost.spike_slab_glm for the underlying IRLS engine: neither family
    is conjugate the way Gaussian is, and neither has Polya-Gamma's exact augmentation
    (which covers Binomial and NB-with-known-r but not Poisson), but canonical/near-
    canonical log-link IRLS still reduces to the same weighted-normal-equations M-step
    the other two response modules use, just with a different variance-function weight.

    `family="poisson"` (default) or `family="negbinomial"` -- real count data is almost
    always overdispersed relative to Poisson, so `"negbinomial"` (which adds a profile-
    MLE-estimated dispersion `r`, re-estimated every EM iteration) is usually the safer
    default in practice; `"poisson"` is offered for cases where it's known to hold, or
    for its simplicity/speed. `.fit(..., offset=log_population)` supports rate models.

    `fit_method`: `"em"`/`"em_filter"` work for both families. `"gibbs"` (empirical
    posterior via Polya-Gamma augmentation, `hierboost.spike_slab_glm.gibbs_sampler_nb`)
    is available for `family="negbinomial"` only -- NB admits the same PG augmentation
    logistic regression does; Poisson has no known exact augmentation, so it stays
    EM/EM-filter-only. NB's Gibbs sampler treats `r` as a per-sweep profile-MLE plug-in
    rather than a fully Bayesian draw (no simple conjugate full conditional exists for
    it) -- see that function's docstring for the honest caveat.

    `decorrelate="sar"|"ar1"|"star"` is supported via the same continuous factor.py/
    state_space.py/spacetime.py block-latent path HierBoostRegressor uses (no JAX
    needed) -- it only ever transforms the raw covariates into a shared per-block
    latent, never touching the response, so it is response-family-agnostic. Only a
    Binomial response (HierBoostClassifier) needs
    Chapter 4's literal discrete/JAX machinery instead, because there the *raw features
    themselves* (not just the response) are assumed Binomial (e.g. genotypes).
    `.summary()`/`.plot_*()` inherited from _HierBoostBase work exactly as for the other
    two subclasses.
    """
    _response = None  # set per-instance in __init__ (one class covers two families)

    def __init__(self, family="poisson", r_init=10.0, **kwargs):
        if family not in ("poisson", "negbinomial"):
            raise ValueError(f"family must be 'poisson' or 'negbinomial', got {family!r}")
        self._response = family
        self.family = family
        self.r_init = r_init
        super().__init__(**kwargs)

    def _link_inv(self, eta):
        return np.exp(np.clip(eta, -30.0, 30.0))

    def _fit_glm(self, X_design, y, wr, n_trials, offset=None):
        if self.family == "poisson":
            if self.fit_method not in ("em", "em_filter"):
                raise ValueError(
                    f"family='poisson' supports fit_method in {{'em', 'em_filter'}}, got "
                    f"{self.fit_method!r} -- no Gibbs sampler yet (Poisson has no known "
                    "exact data-augmentation scheme the way Binomial/NB do via Polya-Gamma)")
            if self.fit_method == "em":
                res = fit_em_poisson(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                                      offset=offset, rank=self.rank)
                return dict(kind="em", beta=res.beta, sigma2=res.sigma2, theta_hat=res.theta_hat,
                            mu=res.mu, ppl=ppl_poisson(y, res.mu), history=None,
                            retained_idx=None, gibbs=None)
            filt = em_filter_poisson(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                                      offset=offset, filter_frac=self.filter_frac,
                                      min_features=self.min_features, max_outer=self.max_outer,
                                      rank=self.rank, patience=self.patience)
            best = filt.best
            return dict(kind="em_filter", beta=best.beta, sigma2=best.sigma2, theta_hat=best.theta_hat,
                        mu=None, ppl=best.ppl, history=filt.history, retained_idx=best.retained_idx, gibbs=None)

        if self.fit_method not in ("em", "em_filter", "gibbs"):
            raise ValueError(
                f"family='negbinomial' supports fit_method in {{'em', 'em_filter', 'gibbs'}}, "
                f"got {self.fit_method!r}")
        if self.fit_method == "em":
            res = fit_em_nb(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                             offset=offset, r_init=self.r_init, rank=self.rank)
            return dict(kind="em", beta=res.beta, sigma2=res.sigma2, r=res.r, theta_hat=res.theta_hat,
                        mu=res.mu, ppl=ppl_nb(y, res.mu, res.r), history=None,
                        retained_idx=None, gibbs=None)
        if self.fit_method == "gibbs":
            gr = gibbs_sampler_nb(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                                   offset=offset, r_init=self.r_init, n_samples=self.n_samples,
                                   burn_in=self.burn_in, thin=self.thin, rank=self.rank,
                                   seed=self.random_state)
            beta_mean = gr.beta.mean(axis=0)
            r_mean = float(gr.r.mean())
            eta = X_design @ beta_mean
            if offset is not None:
                eta = eta + offset
            mu = np.exp(np.clip(eta, -30.0, 30.0))
            return dict(kind="gibbs", beta=beta_mean, sigma2=gr.sigma2.mean(), r=r_mean,
                        theta_hat=gr.pi_hat, mu=mu, ppl=ppl_nb(y, mu, r_mean), history=None,
                        retained_idx=None, gibbs=gr)
        filt = em_filter_nb(X_design, y, wr, self.xi0, self.xi1, self.kappa, self.nu, self.lam,
                             offset=offset, r_init=self.r_init, filter_frac=self.filter_frac,
                             min_features=self.min_features, max_outer=self.max_outer,
                             rank=self.rank, patience=self.patience)
        best = filt.best
        return dict(kind="em_filter", beta=best.beta, sigma2=best.sigma2, r=best.r, theta_hat=best.theta_hat,
                    mu=None, ppl=best.ppl, history=filt.history, retained_idx=best.retained_idx, gibbs=None)

    def predict(self, X_new=None, offset_new=None, return_std=False, ci=0.95, n_mc=2000):
        """Expected count on the observation's natural scale (mu = exp(eta)). Pass
        `offset_new` matching whatever offset (if any) `.fit()` was given -- e.g.
        log(population) for a new set of regions in a rate model."""
        return self._predict_common(X_new, return_std, ci, n_mc, offset_new=offset_new)
