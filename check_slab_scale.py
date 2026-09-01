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

for slab_scale in [2.0, 10.0, 50.0]:
    mcmc, block_ids = run_joint_inference(X, y, BLOCK_ID, clf.latent_fits_, n_trials=N_TRIALS,
                                           slab_scale=slab_scale,
                                           num_warmup=500, num_samples=1000, seed=1, progress_bar=False)
    gamma_causal = np.array(mcmc.get_samples()["gamma"])[:, CAUSAL_BLOCKS[0]]
    print(f"slab_scale={slab_scale}: gamma_causal mean={gamma_causal.mean():.4f} median={np.median(gamma_causal):.4f}")
