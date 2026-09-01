import numpy as np
from calibration_check_joint import FITS, COORDS, BLOCK_ID, N_TRIALS, XI0, K_BLOCKS, CAUSAL_BLOCKS, simulate_one_block
from hierboost.estimator import HierBoostClassifier
from hierboost.joint import run_joint_inference
from scipy.special import expit

rng = np.random.default_rng(7)
n = 3000
X_parts = []
eta = np.zeros(n)
for b in range(K_BLOCKS):
    Xb, Zt = simulate_one_block(rng, FITS[b], n)
    X_parts.append(Xb)
    if b == CAUSAL_BLOCKS[0]:
        eta += 0.6 * Zt
X = np.column_stack(X_parts)
y = (rng.random(n) < expit(eta)).astype(float)

clf = HierBoostClassifier(decorrelate="sar", fit_method="em", xi0=XI0, xi1=0.0, kappa=100.0)
clf.fit(X, y, coords=COORDS, block_id=BLOCK_ID, n_trials=N_TRIALS)
print("plugin beta causal:", clf.beta_[1:][CAUSAL_BLOCKS[0]])

mcmc, block_ids = run_joint_inference(X, y, BLOCK_ID, clf.latent_fits_, n_trials=N_TRIALS,
                                       num_warmup=500, num_samples=1000, seed=1, progress_bar=False)
samples = mcmc.get_samples()
gamma_causal = np.array(samples["gamma"])[:, CAUSAL_BLOCKS[0]]
tau = np.array(samples["tau"])
local_scale_causal = np.array(samples["local_scale"])[:, CAUSAL_BLOCKS[0]]

print("gamma_causal quantiles [5,25,50,75,95]:", np.percentile(gamma_causal, [5, 25, 50, 75, 95]))
print("gamma_causal mean:", gamma_causal.mean(), "median:", np.median(gamma_causal))
print("prob |gamma|>0.1:", (np.abs(gamma_causal) > 0.1).mean())
print("prob |gamma|>1.0:", (np.abs(gamma_causal) > 1.0).mean())
print("tau quantiles:", np.percentile(tau, [5, 50, 95]))
print("local_scale_causal quantiles:", np.percentile(local_scale_causal, [5, 50, 95]))

zt_samples = np.array(samples[f"Zt_{CAUSAL_BLOCKS[0]}"])  # (n_samples, n)
zt_mean = zt_samples.mean(axis=0)
print("Zt posterior mean: std", zt_mean.std(), "range", zt_mean.min(), zt_mean.max())

# compare to the plugin's own fitted Z for the SAME block
kind, payload = clf.latent_fits_[CAUSAL_BLOCKS[0]] if False else (None, None)
Z_plugin = None
for k, b in enumerate(clf.block_ids_):
    if b == CAUSAL_BLOCKS[0]:
        Z_plugin = clf.X_design_[:, 1 + k] if clf.retained_idx_ is None else None
print("plugin Z std:", None if Z_plugin is None else Z_plugin.std())
if Z_plugin is not None:
    print("corr(zt_posterior_mean, plugin_Z):", np.corrcoef(zt_mean, Z_plugin)[0, 1])
    print("ratio of std(zt_mean)/std(Z_plugin):", zt_mean.std() / Z_plugin.std())
