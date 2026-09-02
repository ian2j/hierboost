"""Tests hierboost.latent's literal Chapter 4 discrete/Binomial block-latent machinery
(JAX Newton + moment matching) on real data for the first time -- genotype dosage
(0/1/2) is genuinely Binomial(n=2), the exact observation model this path was built for.
Same LCT-region data/blocks as genomics_1kg_demo.py, for a direct continuous-vs-discrete
comparison. Single train/test split (JAX recompilation cost makes CV impractical here)."""
import argparse
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

from hierboost.structure import determine_blocks
from hierboost.blocks import block_membership_lists
from hierboost.latent import fit_latent_block_model


def main(data_path, out_path):
    d = np.load(data_path)
    X, y, positions = d["X"].astype(float), d["y"], d["positions"]
    pop_a = str(d["pop_a"]) if "pop_a" in d else "EUR"
    pop_b = str(d["pop_b"]) if "pop_b" in d else "AFR"
    print(f"{X.shape[0]} individuals, {X.shape[1]} SNPs, {int(y.sum())} {pop_a} / {int((1-y).sum())} {pop_b}")

    block_id = determine_blocks(coords=positions, method="threshold")
    sizes = np.bincount(block_id)
    print(f"{len(sizes)} LD blocks, size range [{sizes.min()}, {sizes.max()}], mean {sizes.mean():.1f}")

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    print(f"train n={len(ytr)}, test n={len(yte)}")

    K = len(np.unique(block_id))
    wr = np.ones(K)
    xi0 = np.log(0.15 / 0.85)

    print("\nfitting Chapter 4's literal discrete/Binomial block-latent model "
          "(JAX Newton + moment matching) -- this is slower than the continuous branch...")
    result = fit_latent_block_model(Xtr, ytr, positions, block_id, wr, xi0=xi0, xi1=0.0,
                                     kappa=100.0, nu=1.0, lam=1.0, n_trials=2, tau2=1.0,
                                     n_outer=15, n_inner_newton=8, hyper_n_steps=200,
                                     seed=0, verbose=True)

    em = result["em_result"]
    block_ids = result["block_ids"]
    theta = em.theta_hat
    print(f"\nconverged. theta_hat range [{theta.min():.4f}, {theta.max():.4f}], "
          f"n_iter={em.n_iter}")

    order = np.argsort(-theta)[:15]
    blocks = block_membership_lists(block_id)
    block_pos = {b: positions[blocks[b]].mean() for b in blocks}
    print("\ntop 15 LD blocks by posterior inclusion probability:")
    for i in order:
        b = block_ids[i]
        print(f"  block{b}  pos~{block_pos[b]:.0f}  theta={theta[i]:.4f}  "
              f"coef={em.beta[i+1]:.4f}  n_snps={len(blocks[b])}")

    # out-of-sample prediction on the held-out test set
    print("\nprojecting held-out individuals into the fitted block-latent space...")
    Z_test = np.zeros((Xte.shape[0], K))
    for k, b in enumerate(block_ids):
        idx = blocks[b]
        fit = result["fits"][b]
        delta_b = result["deltas"][b]
        Xb_test = jnp.asarray(Xte[:, idx], dtype=jnp.float32)
        zt_init = jnp.zeros(Xte.shape[0])
        zt = fit.infer_ztilde_from_data(Xb_test, delta_b, zt_init)
        Z_test[:, k] = np.array(zt)

    Xd_test = np.column_stack([np.ones(len(yte)), Z_test])
    from scipy.special import expit
    mu_test = expit(Xd_test @ em.beta)
    pred = (mu_test > 0.5).astype(float)
    acc = float(np.mean(pred == yte))
    print(f"\nheld-out test accuracy: {acc:.4f}  (n_test={len(yte)})")

    np.savez(out_path, theta=theta, beta=em.beta, block_ids=np.array(block_ids),
             positions=np.array([block_pos[b] for b in block_ids]), acc=acc)
    print(f"\nsaved result summary to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/ian/genomics_1kg/lct_region.npz")
    p.add_argument("--out", default="/home/ian/genomics_1kg/binomial_latent_result.npz")
    args = p.parse_args()
    main(args.data, args.out)
