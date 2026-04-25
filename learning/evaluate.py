"""
evaluate.py — Reconstruction quality evaluation and biological validation.

psnr(x, y, data_range)
    Peak Signal-to-Noise Ratio between two arrays.

evaluate_and_save_reconstructions(...)
    Per-cell reconstruction error report (eq. 81 success-rate metric).

run_biological_validation(...)
    k-means clustering of unified descriptors phi_j and psi_j against
    ground-truth cell-type labels; reports ARI, NMI, silhouette.
    Requires scikit-learn.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
from typing import Dict, List, Optional

import numpy as np

from .config import LearnConfig


# ---------------------------------------------------------------------------
# PSNR helper
# ---------------------------------------------------------------------------

def psnr(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Parameters
    ----------
    x, y       : arrays of identical shape
    data_range : maximum possible pixel value (default 1.0 for [0,1] images)

    Returns
    -------
    PSNR in dB (capped at 99.0 dB when MSE ≤ 1e-18)
    """
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-18:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


# ---------------------------------------------------------------------------
# Per-channel reconstruction quality
# ---------------------------------------------------------------------------

def evaluate_and_save_reconstructions(
    out_dir: str,
    images: np.ndarray,
    indices: List[int],
    D: np.ndarray,
    A: np.ndarray,
    cfg: LearnConfig,
    channel: str = "unknown",
    hist: Optional[Dict[str, List[float]]] = None,
) -> None:
    """
    Paper success-rate evaluation (eq. 81): e_j = ||x_j - D a_j|| / ||x_j||.

    Writes per-cell CSV, summary JSON, and a human-readable text report.

    Parameters
    ----------
    out_dir  : output directory
    images   : (N, H, W) float array — training images
    indices  : list of dataset indices corresponding to each row of images
    D        : (n, K) orthonormal dictionary
    A        : (N, K) codes
    cfg      : LearnConfig (uses eps_relerr, min_x_norm, ignore_low_norm)
    channel  : channel name for labelling
    hist     : optional training history (currently unused, reserved for future)
    """
    od = pathlib.Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    rel_errs: List[float] = []
    rows = []

    for local_i, ds_idx in enumerate(indices):
        x = images[local_i].reshape(-1).astype(np.float64)
        a = A[local_i].astype(np.float64)
        recon = D @ a
        x_norm = float(np.linalg.norm(x))

        if cfg.ignore_low_norm and x_norm < cfg.min_x_norm:
            rows.append({"dataset_index": float(ds_idx), "rel_err": float("nan"), "ignored": True})
            continue

        r = float(np.linalg.norm(x - recon) / max(x_norm, cfg.min_x_norm))
        rel_errs.append(r)
        rows.append({"dataset_index": float(ds_idx), "rel_err": r, "ignored": False})

    csv_path = od / "paper_reconstruction_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_index", "rel_err", "ignored"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    if rel_errs:
        arr = np.array(rel_errs)
        fidelity_arr = 100.0 * (1.0 - arr)
        summary = {
            "channel": channel,
            "n_samples": len(arr),
            "n_ignored": sum(1 for r in rows if r["ignored"]),
            "rel_err_mean": float(np.mean(arr)),
            "rel_err_median": float(np.median(arr)),
            "rel_err_std": float(np.std(arr)),
            "rel_err_min": float(np.min(arr)),
            "rel_err_max": float(np.max(arr)),
            "fidelity_mean_pct": float(np.mean(fidelity_arr)),
            "fidelity_median_pct": float(np.median(fidelity_arr)),
            "fidelity_min_pct": float(np.min(fidelity_arr)),
            "fidelity_max_pct": float(np.max(fidelity_arr)),
            "K": int(D.shape[1]),
            "n": int(D.shape[0]),
        }
        lines = [
            "=" * 72,
            f"RECONSTRUCTION QUALITY  —  channel: {channel}",
            f"  K={D.shape[1]}  n={D.shape[0]}  N_valid={len(arr)}",
            f"  Mean rel. error : {summary['rel_err_mean']:.6f}",
            f"  Median          : {summary['rel_err_median']:.6f}",
            f"  Std             : {summary['rel_err_std']:.6f}",
            f"  Min / Max       : {summary['rel_err_min']:.6f} / {summary['rel_err_max']:.6f}",
            f"  Fidelity        : mean={summary['fidelity_mean_pct']:.2f}%  "
            f"median={summary['fidelity_median_pct']:.2f}%  "
            f"min={summary['fidelity_min_pct']:.2f}%  "
            f"max={summary['fidelity_max_pct']:.2f}%",
            "=" * 72,
        ]
        report = "\n".join(lines)
        (od / "paper_reconstruction_report.txt").write_text(report + "\n", encoding="utf-8")
        (od / "paper_reconstruction_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(report)
    else:
        print(f"[{channel}] No valid samples for evaluation.")


# ---------------------------------------------------------------------------
# Biological validation
# ---------------------------------------------------------------------------

def run_biological_validation(
    Phi: np.ndarray,
    train_indices: List[int],
    label_indices_path: str,
    label_class_labels_path: str,
    out_dir: str,
) -> None:
    """
    Cluster two unified descriptors for the labeled subset and evaluate
    against ground-truth cell-type labels using ARI, NMI, and silhouette.

    Two descriptors compared:

    phi_j in R^{C*K} — concatenated per-channel codes (channel-wise L2 norm
        + StandardScaler + PCA).

    psi_j in R^K     — cross-channel atom activation norms:
        psi_j[k] = ||(a_j^(1))_k, ..., (a_j^(C))_k||_2
        More compact and more cell-type-specific than raw codes.

    Two clustering experiments per descriptor:
      k=2  Lymphocyte vs Granulocyte (N=26)
      k=3  All three classes         (N=28, Monocyte n=2)

    Requires: scikit-learn, matplotlib, label files from bsccm_real.py.

    Parameters
    ----------
    Phi                      : (N, C*K) unified descriptors from Algorithm 2
    train_indices            : list of BSCCM dataset indices (Phi row -> ds index)
    label_indices_path       : path to label_indices.npy
    label_class_labels_path  : path to label_class_labels.npy
    out_dir                  : output directory for metrics JSON and figure
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            silhouette_score,
        )
        from sklearn.preprocessing import normalize, StandardScaler
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError:
        print("[BioValid] scikit-learn not available. Skipping biological validation.")
        return

    label_indices = np.load(label_indices_path).astype(int)
    class_labels  = np.load(label_class_labels_path).astype(int)

    train_idx_to_row = {int(idx): row for row, idx in enumerate(train_indices)}
    valid_mask = np.array([idx in train_idx_to_row for idx in label_indices])
    if valid_mask.sum() == 0:
        print("[BioValid] No overlap between labeled indices and training indices. "
              "Skipping validation.")
        return

    phi_rows    = np.array([train_idx_to_row[int(idx)]
                            for idx in label_indices[valid_mask]])
    Phi_labeled = Phi[phi_rows]
    gt_labels   = class_labels[valid_mask]
    N_labeled   = int(len(gt_labels))

    label_names = {0: "Lymphocyte", 1: "Granulocyte", 2: "Monocyte"}
    n_lymp = int(np.sum(gt_labels == 0))
    n_gran = int(np.sum(gt_labels == 1))
    n_mono = int(np.sum(gt_labels == 2))

    print(f"\n[BioValid] Labeled subset: {N_labeled} cells "
          f"(Lymphocyte={n_lymp}, Granulocyte={n_gran}, Monocyte={n_mono})")
    print(f"[BioValid] Descriptor dimension: {Phi_labeled.shape[1]}")

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    C_channels = 5
    K_atoms    = Phi_labeled.shape[1] // C_channels

    # Descriptor phi_j: channel-wise L2 norm + StandardScaler + PCA
    channel_blocks = []
    for c in range(C_channels):
        block = Phi_labeled[:, c * K_atoms:(c + 1) * K_atoms]
        channel_blocks.append(normalize(block, norm="l2"))
    Phi_ch_norm = np.concatenate(channel_blocks, axis=1)
    Phi_scaled  = StandardScaler().fit_transform(Phi_ch_norm)
    n_components = min(N_labeled - 1, 15)
    pca = PCA(n_components=n_components, random_state=0)
    Phi_pca = pca.fit_transform(Phi_scaled)
    print(f"[BioValid] phi_j PCA({n_components}), "
          f"explained_var={pca.explained_variance_ratio_.sum():.3f}")

    # Descriptor psi_j: cross-channel atom activation norms
    Phi_matrix   = Phi_labeled.reshape(N_labeled, C_channels, K_atoms)
    Phi_matrix_kc = Phi_matrix.transpose(0, 2, 1)    # (N, K, C)
    Psi_labeled  = np.linalg.norm(Phi_matrix_kc, axis=2)   # (N, K)
    Psi_scaled   = StandardScaler().fit_transform(Psi_labeled)
    n_comp_psi   = min(N_labeled - 1, 15)
    pca_psi      = PCA(n_components=n_comp_psi, random_state=0)
    Psi_pca      = pca_psi.fit_transform(Psi_scaled)
    print(f"[BioValid] psi_j PCA({n_comp_psi}), "
          f"explained_var={pca_psi.explained_variance_ratio_.sum():.3f}")

    def _cluster(desc_pca, gt, mask_2, tag):
        desc_2, gt_2 = desc_pca[mask_2], gt[mask_2]
        N_2 = int(len(gt_2))
        km2 = KMeans(n_clusters=2, n_init=50, random_state=0).fit_predict(desc_2)
        ari2 = adjusted_rand_score(gt_2, km2)
        nmi2 = normalized_mutual_info_score(gt_2, km2, average_method="arithmetic")
        sil2 = silhouette_score(desc_2, km2) if N_2 > 2 else float("nan")
        km3 = KMeans(n_clusters=3, n_init=50, random_state=0).fit_predict(desc_pca)
        ari3 = adjusted_rand_score(gt, km3)
        nmi3 = normalized_mutual_info_score(gt, km3, average_method="arithmetic")
        sil3 = silhouette_score(desc_pca, km3) if len(gt) > 3 else float("nan")
        print(f"\n[BioValid] [{tag}] k=2 (N={N_2}): ARI={ari2:.4f}  NMI={nmi2:.4f}  Sil={sil2:.4f}")
        print(f"[BioValid] [{tag}] k=3 (N={len(gt)}): ARI={ari3:.4f}  NMI={nmi3:.4f}  Sil={sil3:.4f}")
        return {"N_2": N_2, "ari2": ari2, "nmi2": nmi2, "sil2": sil2,
                "ari3": ari3, "nmi3": nmi3, "sil3": sil3}

    mask_2cls = (gt_labels == 0) | (gt_labels == 1)
    res_phi = _cluster(Phi_pca, gt_labels, mask_2cls, "phi_j")
    res_psi = _cluster(Psi_pca, gt_labels, mask_2cls, "psi_j")

    ari2,  nmi2,  sil2  = res_phi["ari2"],  res_phi["nmi2"],  res_phi["sil2"]
    ari3,  nmi3,  sil3  = res_phi["ari3"],  res_phi["nmi3"],  res_phi["sil3"]
    ari2p, nmi2p, sil2p = res_psi["ari2"],  res_psi["nmi2"],  res_psi["sil2"]
    ari3p, nmi3p, sil3p = res_psi["ari3"],  res_psi["nmi3"],  res_psi["sil3"]
    N_2cls = res_phi["N_2"]

    metrics = {
        "N_labeled": N_labeled,
        "class_counts": {label_names[k]: int(np.sum(gt_labels == k)) for k in label_names},
        "phi_j": {
            "k2_Lymphocyte_vs_Granulocyte": {
                "N": N_2cls, "ARI": float(ari2), "NMI": float(nmi2), "silhouette": float(sil2)},
            "k3_all_classes": {
                "N": N_labeled, "ARI": float(ari3), "NMI": float(nmi3), "silhouette": float(sil3)},
        },
        "psi_j": {
            "k2_Lymphocyte_vs_Granulocyte": {
                "N": N_2cls, "ARI": float(ari2p), "NMI": float(nmi2p), "silhouette": float(sil2p)},
            "k3_all_classes": {
                "N": N_labeled, "ARI": float(ari3p), "NMI": float(nmi3p), "silhouette": float(sil3p)},
        },
    }
    with open(out_path / "bio_validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[BioValid] Metrics saved: {out_path / 'bio_validation_metrics.json'}")

    # Figure: 5 panels
    fig, axes = plt.subplots(1, 5, figsize=(26, 5))

    ax = axes[0]
    bars = ax.bar(
        [label_names[k] for k in range(3)],
        [n_lymp, n_gran, n_mono],
        color=["steelblue", "tomato", "mediumseagreen"],
    )
    for bar, val in zip(bars, [n_lymp, n_gran, n_mono]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(val),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Ground-truth\ndistribution (N=28)", fontsize=10)
    ax.set_ylabel("Cell count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    def _bar_panel(ax, vals, title):
        colors_bar = ["steelblue", "darkorange", "mediumseagreen"]
        bars = ax.bar(["ARI", "NMI", "Silhouette"], vals, color=colors_bar)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height() + 0.01, 0.02),
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _bar_panel(axes[1], [ari2,  nmi2,  sil2],
               f"Phi_j  k=2: Lymph vs Gran\n(N={N_2cls})")
    _bar_panel(axes[2], [ari3,  nmi3,  sil3],
               f"Phi_j  k=3: All classes\n(N={N_labeled}, Mono n=2)")
    _bar_panel(axes[3], [ari2p, nmi2p, sil2p],
               f"Psi_j  k=2: Lymph vs Gran\n(N={N_2cls})")
    _bar_panel(axes[4], [ari3p, nmi3p, sil3p],
               f"Psi_j  k=3: All classes\n(N={N_labeled}, Mono n=2)")

    plt.suptitle(
        "Biological validation: Phi_j (concatenated per-channel codes)"
        " vs Psi_j (cross-channel atom norms)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_path = out_path / "bio_validation_metrics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[BioValid] Figure saved: {fig_path}")
