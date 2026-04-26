"""
bootstrap_validation.py
-----------------------
Bootstrap confidence intervals and permutation null test for the
k=2 lymphoid-vs-myeloid clustering validation on BSCCM-tiny.

Requires:
    unified_descriptors_Phi.npy
    label_class_labels.npy 
    label_indices.npy
    train_indices.txt

Usage:
    python bootstrap_validation.py \
        --phi      unified_descriptors_Phi.npy \
        --labels   label_class_labels.npy \
        --indices  label_indices.npy \
        --train    train_indices.txt \
        --n_boot   1000 \
        --n_perm   1000 \
        --seed     42
"""

import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(phi_path, labels_path, indices_path, train_path):
    Phi        = np.load(phi_path)                  # (1000, 2560)
    labels     = np.load(labels_path)               # (28,)
    lab_idx    = np.load(indices_path)              # (28,) dataset indices
    train_idx  = np.array([int(x) for x in
                           open(train_path).read().split()])  # (1000,)

    # Map dataset-level label indices -> row positions in Phi
    idx_map = {v: i for i, v in enumerate(train_idx)}
    positions = np.array([idx_map[int(d)] for d in lab_idx])

    return Phi, labels, positions


def preprocess(Phi, positions, n_components=15, K=512, C=5):
    """Channel-wise L2 norm -> per-atom standard scaling -> PCA."""
    X = Phi[positions].copy()                       # (28, 2560)

    # Channel-wise L2 normalisation
    for c in range(C):
        block = X[:, c*K:(c+1)*K]
        norms = np.linalg.norm(block, axis=1, keepdims=True) + 1e-12
        X[:, c*K:(c+1)*K] = block / norms

    # Per-atom standard scaling
    X = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    return X_pca


def run_kmeans(X, k, n_init=20, seed=0):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    return km.fit_predict(X)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    rng = np.random.default_rng(args.seed)

    # --- Load and preprocess ---
    Phi, labels, positions = load_data(
        args.phi, args.labels, args.indices, args.train)

    X_pca = preprocess(Phi, positions)
    labels = np.array(labels)

    # --- k=2 subset: drop Monocytes (class 2) ---
    mask2   = labels != 2
    X2      = X_pca[mask2]
    labels2 = labels[mask2]
    print(f"\nk=2 subset: N={len(labels2)}, "
          f"classes={dict(zip(*np.unique(labels2, return_counts=True)))}")

    # --- Observed metrics ---
    pred_obs = run_kmeans(X2, k=2, n_init=20, seed=args.seed)
    obs_ari  = adjusted_rand_score(labels2, pred_obs)
    obs_nmi  = normalized_mutual_info_score(labels2, pred_obs)
    print(f"\nObserved  ARI = {obs_ari:.4f}   NMI = {obs_nmi:.4f}")

    # --- Bootstrap CI ---
    boot_ari, boot_nmi = [], []
    for _ in range(args.n_boot):
        idx   = rng.choice(len(X2), size=len(X2), replace=True)
        X_b   = X2[idx]
        y_b   = labels2[idx]
        if len(np.unique(y_b)) < 2:
            continue
        pred_b = run_kmeans(X_b, k=2, seed=int(rng.integers(1_000_000)))
        boot_ari.append(adjusted_rand_score(y_b, pred_b))
        boot_nmi.append(normalized_mutual_info_score(y_b, pred_b))

    boot_ari = np.array(boot_ari)
    boot_nmi = np.array(boot_nmi)
    ci_ari   = np.percentile(boot_ari, [2.5, 97.5])
    ci_nmi   = np.percentile(boot_nmi, [2.5, 97.5])

    print(f"\nBootstrap 95% CI  (n={len(boot_ari)} resamples):")
    print(f"  ARI : [{ci_ari[0]:.3f}, {ci_ari[1]:.3f}]")
    print(f"  NMI : [{ci_nmi[0]:.3f}, {ci_nmi[1]:.3f}]")

    # --- Permutation null distribution ---
    null_ari, null_nmi = [], []
    for _ in range(args.n_perm):
        y_perm = rng.permutation(labels2)
        pred_p = run_kmeans(X2, k=2, seed=int(rng.integers(1_000_000)))
        null_ari.append(adjusted_rand_score(y_perm, pred_p))
        null_nmi.append(normalized_mutual_info_score(y_perm, pred_p))

    null_ari = np.array(null_ari)
    null_nmi = np.array(null_nmi)
    p_ari    = np.mean(null_ari >= obs_ari)
    p_nmi    = np.mean(null_nmi >= obs_nmi)

    print(f"\nPermutation null  (n={args.n_perm} permutations):")
    print(f"  ARI  null mean={null_ari.mean():.3f}  "
          f"95th pct={np.percentile(null_ari, 95):.3f}  p={p_ari:.4f}")
    print(f"  NMI  null mean={null_nmi.mean():.3f}  "
          f"95th pct={np.percentile(null_nmi, 95):.3f}  p={p_nmi:.4f}")

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"Observed k=2: ARI={obs_ari:.3f}, NMI={obs_nmi:.3f}")
    print(f"Permutation p (ARI): {p_ari:.4f}  |  p (NMI): {p_nmi:.4f}")
    print(f"Bootstrap 95% CI: ARI [{ci_ari[0]:.2f}, {ci_ari[1]:.2f}], "
          f"NMI [{ci_nmi[0]:.2f}, {ci_nmi[1]:.2f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap validation for BSCCM clustering.")
    parser.add_argument("--phi",     required=True, help="Path to unified_descriptors_Phi.npy")
    parser.add_argument("--labels",  required=True, help="Path to label_class_labels.npy")
    parser.add_argument("--indices", required=True, help="Path to label_indices.npy")
    parser.add_argument("--train",   required=True, help="Path to train_indices.txt")
    parser.add_argument("--n_boot",  type=int, default=1000, help="Bootstrap resamples (default 1000)")
    parser.add_argument("--n_perm",  type=int, default=1000, help="Permutation resamples (default 1000)")
    parser.add_argument("--seed",    type=int, default=42,   help="Random seed (default 42)")
    args = parser.parse_args()
    main(args)
