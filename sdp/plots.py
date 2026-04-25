"""
plots.py — All figure-generating functions.

Functions
---------
save_convergence_plots          Full 2-row × 4-column convergence figure.
save_single_unified_cell        3-panel single-cell portrait (Section 9).
save_unified_cell_figure        Multi-cell × multi-channel grid figure.
save_reconstructed_images       Orig / focused / reconstruction 3-row figure.
save_unified_vs_truth           Unified image vs ground-truth comparison.
"""
from __future__ import annotations

import io
import math
import pathlib
import sys
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional BSCCM / focus imports
# ---------------------------------------------------------------------------
_DEV = pathlib.Path(__file__).resolve().parent.parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

try:
    from read_bsccm_data import BSCCM_Img_Reader
    _HAS_BSCCM = True
except Exception:
    BSCCM_Img_Reader = None  # type: ignore
    _HAS_BSCCM = False

try:
    from focus_l1 import get_focus_analysis_for_image
    _HAS_FOCUS = True
except Exception:
    get_focus_analysis_for_image = None  # type: ignore
    _HAS_FOCUS = False

from .config   import LearnConfig
from .evaluate import psnr


# ===========================================================================
# Convergence figure
# ===========================================================================

def save_convergence_plots(
    hist: Dict[str, List[float]],
    output_dir: str = "error_analysis",
    pdhg_residuals: Optional[List[float]] = None,
    per_image_rows: Optional[List[Dict]] = None,
    channels: Optional[List[str]] = None,
    eps_relerr: float = 0.05,
) -> None:
    """
    Save full convergence figure.

    Layout — 2 rows × 4 columns:

      Row 0  (4 panels):
        [0,0] Reconstruction fidelity    0.5*||x - Da||^2
        [0,1] Dictionary convergence     ||D^{t+1} - D^t||_F
        [0,2] Inner PDHG convergence     mean & max final residual per epoch
        [0,3] Learning objective         total E(y;x)

      Row 1  (1 full-width panel spanning all 4 columns):
        All channels overlaid — 100*||x-Da||/||x|| [%] vs outer iteration, log y
    """
    import matplotlib.gridspec as mgridspec
    import matplotlib.ticker

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ch_epoch_data: Dict[str, Dict[int, List[float]]] = {}
    if per_image_rows and channels:
        for ch in channels:
            ch_epoch_data[ch] = {}
        for row in per_image_rows:
            ch  = row.get("channel", "")
            if ch not in ch_epoch_data:
                continue
            ep  = int(row.get("outer_iter", 0))
            rel = row.get("rel_err", float("nan"))
            if rel is not None and not (isinstance(rel, float) and math.isnan(rel)):
                ch_epoch_data[ch].setdefault(ep, []).append(float(rel))

    fig = plt.figure(figsize=(20.0, 11.0))
    gs = mgridspec.GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.0, 0.85], hspace=0.55, wspace=0.38,
        left=0.06, right=0.97, top=0.91, bottom=0.07,
    )

    def _logplot(ax, vals, color, ylabel, title, marker="o"):
        if not vals or all(math.isnan(v) if isinstance(v, float) else False for v in vals):
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            return
        v  = np.maximum(np.asarray(vals, dtype=float), 1e-16)
        ep = np.arange(1, len(v) + 1)
        ax.plot(ep, v, f"{marker}-", lw=2, ms=5, color=color)
        ax.set_yscale("log")
        ax.set_xlabel("Outer Iteration", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", which="both")

    ax1 = fig.add_subplot(gs[0, 0])
    _logplot(ax1, hist.get("fidelity", []), "C0",
             "0.5*||x - Da||^2", "Reconstruction Fidelity\n0.5*||x - Da||^2")

    ax2 = fig.add_subplot(gs[0, 1])
    _logplot(ax2, hist.get("dict_change", []), "C3",
             "||D^{t+1} - D^t||_F", "Dictionary Convergence\n||D^{t+1} - D^t||_F")

    ax3 = fig.add_subplot(gs[0, 2])
    mfr = hist.get("pdhg_mean_final_res", [])
    xfr = hist.get("pdhg_max_final_res", [])
    if mfr and any(not (isinstance(v, float) and math.isnan(v)) for v in mfr):
        ep3 = np.arange(1, len(mfr) + 1)
        ax3.plot(ep3, np.maximum(np.asarray(mfr, float), 1e-16),
                 "o-", lw=2, ms=5, color="C4", label="mean final ||Δy||")
        if xfr and len(xfr) == len(mfr):
            ax3.plot(ep3, np.maximum(np.asarray(xfr, float), 1e-16),
                     "^--", lw=1.2, ms=4, color="C5", alpha=0.8,
                     label="max final ||Δy||")
        ax3.set_yscale("log")
        ax3.legend(fontsize=8, loc="upper right")
    else:
        ax3.text(0.5, 0.5, "no pdhg_mean_final_res data",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=9)
    ax3.set_xlabel("Outer Iteration", fontsize=10)
    ax3.set_ylabel("||y^{n+1}-y^n||_2  (final per epoch)", fontsize=10)
    ax3.set_title("Inner PDHG Convergence per Epoch\n(final residual of inner loop)",
                  fontsize=10, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle="--", which="both")

    ax4 = fig.add_subplot(gs[0, 3])
    obj = hist.get("obj", [])
    if obj:
        ep4 = np.arange(1, len(obj) + 1)
        ax4.plot(ep4, np.maximum(np.asarray(obj, float), 1e-16), "D-", lw=2, ms=5, color="C2")
    ax4.set_yscale("log")
    ax4.set_xlabel("Outer Iteration", fontsize=10)
    ax4.set_ylabel("0.5*||x - Da||^2 + lambda_TV*TV(y*)", fontsize=10)
    ax4.set_title("Learning Objective Decay\n0.5*||x - Da||^2 + lambda_TV*TV(y*)",
                  fontsize=10, fontweight="bold")
    ax4.grid(True, alpha=0.3, linestyle="--", which="both")

    ax_ch = fig.add_subplot(gs[1, :])
    ch_colors = ["C0", "C1", "C3", "C4", "C6", "C7", "C8", "C9"]
    any_plotted = False
    for r, ch in enumerate(channels or []):
        edata = ch_epoch_data.get(ch, {})
        ep_sorted = sorted(edata.keys())
        if not ep_sorted:
            continue
        mean_pct = [float(np.mean(edata[e])) for e in ep_sorted]
        color    = ch_colors[r % len(ch_colors)]
        ch_label = ch.replace("DPC_", "DPC-").replace("Brightfield", "BF")
        ax_ch.plot(np.array(ep_sorted, dtype=float),
                   np.maximum(mean_pct, 1e-6),
                   "o-", lw=2, ms=5, color=color, label=ch_label)
        any_plotted = True

    if any_plotted:
        ax_ch.set_yscale("log")
        ax_ch.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.2g}%")
        )
        ax_ch.legend(fontsize=10, loc="upper right", framealpha=0.9,
                     title="Channel", title_fontsize=9)
    else:
        ax_ch.text(0.5, 0.5, "No per-channel data", ha="center", va="center",
                   transform=ax_ch.transAxes, fontsize=11)

    ax_ch.set_xlabel("Outer Iteration", fontsize=11)
    ax_ch.set_ylabel("||x-Da||/||x||  [%]  (log scale)", fontsize=11)
    ax_ch.set_title("Per-Channel Reconstruction Error Decay",
                    fontsize=11, fontweight="bold")
    ax_ch.grid(True, alpha=0.3, linestyle="--", which="both")

    fig.suptitle("Dictionary Learning - Full Convergence Analysis",
                 fontsize=13, fontweight="bold")

    timestamp = time.strftime("%d%m%Y%H%M%S")
    fp = Path(output_dir) / f"convergence_{timestamp}.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Convergence plot saved: {fp}")


# ===========================================================================
# Single unified cell portrait (Section 9, eq. 88)
# ===========================================================================

def save_single_unified_cell(
    D_per_channel: Dict[str, np.ndarray],
    A_per_channel: Dict[str, np.ndarray],
    images_per_channel: Dict[str, np.ndarray],
    cfg: "LearnConfig",
    out_dir: str,
    cell_idx: int = 0,
    n_top_atoms: int = 8,
) -> None:
    """
    Unified single-cell figure grounded in the manuscript (Section 9, eq. 88).

    Three panels:
      A — Phi_j heatmap (K × C matrix, unified descriptor eq. 88)
      B — Top-n_top_atoms atoms with largest cross-channel activation norm
      C — Unified image u_j = Σ w_c D^(c) a_j^(c)  (inverse-residual weighted)
    """
    channels = list(A_per_channel.keys())
    C = len(channels)
    K = A_per_channel[channels[0]].shape[1]
    N, H, W = images_per_channel[channels[0]].shape
    cell_idx = min(cell_idx, N - 1)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    Phi_j = np.column_stack([A_per_channel[ch][cell_idx] for ch in channels])
    atom_norms = np.linalg.norm(Phi_j, axis=1)
    order = np.argsort(atom_norms)[::-1]
    Phi_sorted = Phi_j[order]
    norms_sorted = atom_norms[order]

    # Panel C: inverse-residual weighted image-space unification
    eps_w = 1e-8
    weights = []
    recon_images = []
    for ch in channels:
        D_ch = D_per_channel[ch]
        x_ch = images_per_channel[ch][cell_idx].ravel()
        a_ch = A_per_channel[ch][cell_idx]
        u_ch = D_ch @ a_ch
        resid = float(np.sum((x_ch - u_ch) ** 2))
        weights.append(1.0 / (resid + eps_w))
        recon_images.append(u_ch)

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()
    u_j_flat = sum(weights[ci] * recon_images[ci] for ci in range(len(channels)))
    u_j = u_j_flat.reshape(H, W)
    umin, umax = u_j.min(), u_j.max()
    unified = (u_j - umin) / (umax - umin + 1e-12)

    print("  [UnifiedCell] channel weights (inverse-residual):")
    for c_idx, ch in enumerate(channels):
        print(f"    {ch:<20s}  w={weights[c_idx]:.4f}")

    ch_short = [ch.replace("DPC_", "").replace("Brightfield", "BF") for ch in channels]
    n_top = min(n_top_atoms, K)
    top_indices = order[:n_top]

    a_w = max(C, 4)
    b_w = n_top
    c_w = 3
    total_w = a_w + b_w + c_w + 2
    fig_w = max(16, total_w * 1.1)
    fig_h = max(6, H / W * fig_w / total_w * 3.5)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[a_w, b_w, c_w],
                  wspace=0.35, left=0.06, right=0.97, top=0.84, bottom=0.12)

    # Panel A: Phi_j heatmap
    ax_heat = fig.add_subplot(gs[0])
    vabs = float(np.percentile(np.abs(Phi_j), 98)) + 1e-12
    im = ax_heat.imshow(Phi_sorted, aspect="auto", cmap="RdBu_r",
                        vmin=-vabs, vmax=vabs, interpolation="nearest")
    ax_heat.set_xticks(range(C))
    ax_heat.set_xticklabels(ch_short, fontsize=8, fontweight="bold")
    ax_heat.set_xlabel("Channel", fontsize=9)
    ax_heat.set_ylabel("Dictionary atom k  (sorted by activation strength)", fontsize=8)
    step = max(1, K // 12)
    yticks = list(range(0, K, step))
    ax_heat.set_yticks(yticks)
    ax_heat.set_yticklabels([str(order[i]) for i in yticks], fontsize=7)
    plt.colorbar(im, ax=ax_heat, shrink=0.5, pad=0.02, label="(a_j^c)_k")
    ax_heat.set_title("A.  Unified descriptor Phi_j (eq. 88)",
                      fontsize=9, fontweight="bold", pad=6)

    # Panel B: top-atom images + bar charts
    D_ref = D_per_channel[channels[0]]
    gs_b = GridSpecFromSubplotSpec(2, n_top, subplot_spec=gs[1], hspace=0.15, wspace=0.05)
    for rank in range(n_top):
        k = top_indices[rank]
        atom_img = D_ref[:, k].reshape(H, W)
        amin, amax = atom_img.min(), atom_img.max()
        atom_disp = (atom_img - amin) / (amax - amin + 1e-12)

        ax_top = fig.add_subplot(gs_b[0, rank])
        ax_top.imshow(atom_disp, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax_top.axis("off")
        ax_top.set_title(f"a{k}", fontsize=6, pad=2)

        ax_bot = fig.add_subplot(gs_b[1, rank])
        bar_vals = Phi_j[order[rank], :]
        colors = ["steelblue" if v >= 0 else "crimson" for v in bar_vals]
        ax_bot.bar(range(C), bar_vals, color=colors, edgecolor="none", width=0.8)
        ax_bot.axhline(0, color="k", lw=0.5)
        ax_bot.set_xticks(range(C))
        ax_bot.set_xticklabels(ch_short, fontsize=5, rotation=45)
        ax_bot.tick_params(axis="y", labelsize=5)
        ax_bot.set_title(f"||.||={norms_sorted[rank]:.1f}", fontsize=5, pad=1)
        if rank == 0:
            ax_bot.set_ylabel("code", fontsize=5)
        atom_range = max(float(np.max(np.abs(bar_vals))), 1e-6)
        ax_bot.set_ylim(-atom_range * 1.3, atom_range * 1.3)

    fig.text(0.62, 0.91,
             f"B.  Top-{n_top} atoms  (structural primitives)",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel C: unified image
    ax_c = fig.add_subplot(gs[2])
    ax_c.imshow(unified, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax_c.axis("off")
    ax_c.set_title("C.  Unified image $u_j = D\\bar{a}_j$\n"
                   "(weighted code-space unification)")
    for sp in ax_c.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("saddlebrown")
        sp.set_linewidth(1.5)

    fig.suptitle(
        f"Unified single-cell  cell #{cell_idx}, K={K}, C={C} -- "
        "Panel A: Phi_j heatmap | Panel B: dominant atoms | Panel C: u_j",
        fontsize=9, y=0.97,
    )

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    fig_path = (pathlib.Path(out_dir) /
                f"unified_single_cell_{cell_idx}_{timestamp}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Unified single-cell figure saved: {fig_path}")


# ===========================================================================
# Multi-cell × multi-channel reconstruction grid (Section 8.6)
# ===========================================================================

def save_unified_cell_figure(
    D_per_channel: Dict[str, np.ndarray],
    A_per_channel: Dict[str, np.ndarray],
    images_per_channel: Dict[str, np.ndarray],
    out_dir: str,
    n_cells: int = 5,
) -> None:
    """
    n_cells rows × C columns grid showing, for each (cell, channel):
      top half    — focused training image x_j^(c)
      bottom half — reconstruction D^(c) a_j^(c)
    with a white separator and a relative-error annotation.
    """
    channels = list(images_per_channel.keys())
    C = len(channels)
    if C == 0:
        return

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    N, H, W = images_per_channel[channels[0]].shape
    n_cells = min(n_cells, N)
    cell_indices = np.linspace(0, N - 1, n_cells, dtype=int).tolist()
    sep = max(1, H // 30)

    fig, axs = plt.subplots(n_cells, C, figsize=(3.2 * C, 3.0 * n_cells), squeeze=False)

    for row, cell_idx in enumerate(cell_indices):
        for col, ch in enumerate(channels):
            x   = images_per_channel[ch][cell_idx]
            a   = A_per_channel[ch][cell_idx]
            D_ch = D_per_channel[ch]
            rec = (D_ch @ a).reshape(H, W)
            rec_disp = np.clip(rec, 0.0, 1.0)
            r_min, r_max = float(rec.min()), float(rec.max())
            if r_max > r_min:
                rec_disp = (rec - r_min) / (r_max - r_min)

            rel_err = float(
                np.linalg.norm(x.ravel() - (D_ch @ a)) /
                max(float(np.linalg.norm(x.ravel())), 1e-9)
            )
            separator = np.ones((sep, W), dtype=np.float64)
            panel = np.vstack([x, separator, rec_disp])

            ax = axs[row, col]
            ax.imshow(panel, cmap="inferno", vmin=0.0, vmax=1.0,
                      interpolation="nearest", aspect="auto")
            ax.axis("off")

            if row == 0:
                short = ch.replace("DPC_", "").replace("Brightfield", "BF")
                ax.set_title(short, fontsize=10, fontweight="bold", pad=4)
            if col == 0:
                ax.set_ylabel(f"cell {cell_idx}", fontsize=9, rotation=0,
                              labelpad=36, va="center")
            ax.text(0.98, 0.02, f"e={rel_err:.3f}",
                    transform=ax.transAxes, fontsize=7, color="white",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45))

    fig.text(0.01, 0.5, "top: input  |  bottom: D·a",
             va="center", rotation=90, fontsize=8, color="0.4")
    fig.suptitle(
        f"Unified cell reconstructions across all {C} channels\n"
        f"(shared dictionary D, {n_cells} cells shown)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    fig_path = pathlib.Path(out_dir) / f"unified_cell_figure_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Unified cell figure saved: {fig_path}")


# ===========================================================================
# Orig / focused / reconstruction figure
# ===========================================================================

def save_reconstructed_images(
    D_per_channel: Dict[str, np.ndarray],
    A_per_channel: Dict[str, np.ndarray],
    images_per_channel: Dict[str, np.ndarray],
    indices: List[int],
    out_dir: str,
    lam_tv: float,
    cfg: "LearnConfig",
) -> None:
    """
    3-row × C-column figure for the first training cell:
      Row 0 — original BSCCM cell image (raw, normalized to [0,1])
      Row 1 — focused crop (gradient-energy crop in images_per_channel)
      Row 2 — dictionary reconstruction D^(c) a_j

    Requires BSCCM_Img_Reader and focus_l1. Silently skipped if unavailable.
    """
    if not _HAS_BSCCM:
        print("save_reconstructed_images: BSCCM reader unavailable, skipping.")
        return

    channels = list(images_per_channel.keys())
    C = len(channels)
    if C == 0:
        return

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    cell_idx = 0
    ds_idx   = int(indices[cell_idx]) if indices else 0
    _, H, W  = images_per_channel[channels[0]].shape

    reader = BSCCM_Img_Reader(output_dir="bsccm_out_image")
    reader.load_dataset(location="BSCCM-tiny", tiny=True)

    fig, axs = plt.subplots(3, C, figsize=(4 * C, 10))
    if C == 1:
        axs = axs.reshape(3, 1)

    psnr_vals: List[float] = []

    for c_idx, ch in enumerate(channels):
        raw = np.asarray(reader.bsccm.read_image(ds_idx, channel=ch), dtype=np.float64)
        if raw.ndim != 2:
            raw = np.squeeze(raw)
        raw = raw - float(np.min(raw))
        mx = float(np.max(raw))
        if mx > 0:
            raw /= mx

        focused = images_per_channel[ch][cell_idx]

        a = A_per_channel[ch][cell_idx]
        recon2d = (D_per_channel[ch] @ a).reshape(H, W)
        recon2d -= float(np.min(recon2d))
        mx2 = float(np.max(recon2d))
        if mx2 > 0:
            recon2d /= mx2

        p = psnr(focused, recon2d, data_range=1.0)
        psnr_vals.append(p)

        for row, (img, title) in enumerate([
            (raw,     f"{ch}\nOriginal (raw BSCCM)"),
            (focused, f"{ch}\nFocused crop"),
            (recon2d, f"{ch}\nReconstruction  Da\nPSNR={p:.2f} dB"),
        ]):
            im = axs[row, c_idx].imshow(img, cmap="inferno", vmin=0.0, vmax=1.0)
            axs[row, c_idx].axis("off")
            axs[row, c_idx].set_title(title, fontsize=9, fontweight="bold", pad=6)
            plt.colorbar(im, ax=axs[row, c_idx], shrink=0.7)

    fig.suptitle(
        f"Cell #{ds_idx:04d} | lambda_TV={lam_tv:.2e} | "
        f"mean PSNR (focused vs recon)={float(np.mean(psnr_vals)):.2f} dB",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    outpath = pathlib.Path(out_dir) / f"reconstructed_image_{timestamp}.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Reconstructed-image figure saved: {outpath}")


# ===========================================================================
# Unified image vs ground-truth comparison
# ===========================================================================

def save_unified_vs_truth(
    D_per_channel: Dict[str, np.ndarray],
    A_per_channel: Dict[str, np.ndarray],
    images_per_channel: Dict[str, np.ndarray],
    train_indices: List[int],
    out_dir: str,
    gt_channel: str = "Brightfield",
) -> None:
    """
    For each labeled cell in the training set, display its ground-truth
    focused crop (gt_channel) alongside its unified image
    u_j = Σ w_c D^(c) a_j^(c) (inverse-residual weighted).

    Organized by cell type (Lymphocyte / Granulocyte / Monocyte).
    Label files loaded from bsccm_real_out/ next to this script.

    Call this function once per channel to produce one figure per channel.
    In run_training.py / run_post_training_figures.py the caller loops over
    ``images_per_channel.keys()`` so a separate PNG is written for every
    available imaging channel, e.g.::

        for gt_ch in images_per_channel.keys():
            save_unified_vs_truth(..., gt_channel=gt_ch)

    Output filename: unified_vs_truth_<gt_channel>_<timestamp>.png

    Requires BSCCM_Img_Reader, focus_l1, and label files from bsccm_real.py.
    """
    if not _HAS_BSCCM:
        print("[UnifiedVsTruth] BSCCM reader unavailable, skipping.")
        return

    bio_label_dir = _DEV / "bsccm_real_out"
    li_path = bio_label_dir / "label_indices.npy"
    lc_path = bio_label_dir / "label_class_labels.npy"
    if not li_path.exists() or not lc_path.exists():
        print("[UnifiedVsTruth] Label files not found. Run bsccm_real.py first.")
        return

    label_indices  = np.load(str(li_path)).astype(int)
    class_labels   = np.load(str(lc_path)).astype(int)
    train_idx_map  = {int(idx): row for row, idx in enumerate(train_indices)}
    label_names    = {0: "Lymphocyte", 1: "Granulocyte", 2: "Monocyte"}

    channels = list(D_per_channel.keys())
    _, H, W  = images_per_channel[channels[0]].shape
    eps_w    = 1e-8

    reader = BSCCM_Img_Reader(output_dir="bsccm_out_image")
    reader.load_dataset(location="BSCCM-tiny", tiny=True)

    cell_h, cell_w, dpi = 2.5, 2.5, 300
    rendered = []

    for k, name in label_names.items():
        cell_ds_idxs = [
            int(idx) for idx, lbl in zip(label_indices, class_labels)
            if lbl == k and int(idx) in train_idx_map
        ]
        n_avail = len(cell_ds_idxs)
        if n_avail == 0:
            print(f"  [UnifiedVsTruth] {name}: no cells in training set.")
            continue

        fig = plt.figure(figsize=(n_avail * cell_w, 2 * cell_h + 0.7))
        fig.suptitle(
            f"{name}  —  {n_avail} cells  |  "
            f"Row 0: {gt_channel} GT    Row 1: u_j (image-space weighted)",
            fontsize=10, fontweight="bold", x=0.01, ha="left", y=0.99,
        )
        gs = GridSpec(2, n_avail, figure=fig, hspace=0.08, wspace=0.12,
                      top=0.88, bottom=0.06, left=0.07, right=0.99)

        for col, ds_idx in enumerate(cell_ds_idxs):
            row = train_idx_map[ds_idx]

            # Row 0 — ground-truth focused crop
            raw_img = np.asarray(
                reader.bsccm.read_image(ds_idx, channel=gt_channel), dtype=np.float64
            )
            if raw_img.ndim != 2:
                raw_img = np.squeeze(raw_img)
            if _HAS_FOCUS:
                focus = get_focus_analysis_for_image(
                    raw_img, ds_idx, gt_channel, method="gradient"
                )
                foc = focus["focused_image"].astype(np.float64)
            else:
                foc = raw_img
            f_min, f_max = foc.min(), foc.max()
            if f_max > f_min:
                foc = (foc - f_min) / (f_max - f_min)

            ax0 = fig.add_subplot(gs[0, col])
            ax0.imshow(foc, cmap="inferno", aspect="equal", vmin=0, vmax=1)
            ax0.set_xticks([]); ax0.set_yticks([])
            ax0.set_title(f"idx {ds_idx}", fontsize=7, pad=2)
            if col == 0:
                ax0.set_ylabel(f"GT ({gt_channel})", fontsize=8,
                               fontweight="bold", rotation=90, labelpad=4, va="center")

            # Row 1 — unified image
            weights = []
            recon_imgs = []
            for ch in channels:
                D_ch = D_per_channel[ch]
                x_ch = images_per_channel[ch][row].ravel()
                a_ch = A_per_channel[ch][row]
                u_ch = D_ch @ a_ch
                resid = float(np.sum((x_ch - u_ch) ** 2))
                weights.append(1.0 / (resid + eps_w))
                recon_imgs.append(u_ch)

            weights = np.array(weights, dtype=np.float64)
            weights /= weights.sum()
            u_j = sum(weights[ci] * recon_imgs[ci]
                      for ci in range(len(channels))).reshape(H, W)
            u_min, u_max = u_j.min(), u_j.max()
            if u_max > u_min:
                u_j = (u_j - u_min) / (u_max - u_min)

            ax1 = fig.add_subplot(gs[1, col])
            ax1.imshow(u_j, cmap="inferno", aspect="equal", vmin=0, vmax=1)
            ax1.set_xticks([]); ax1.set_yticks([])
            if col == 0:
                ax1.set_ylabel("u_j = Σ w_c D^(c)a^(c)", fontsize=8,
                               fontweight="bold", rotation=90, labelpad=4, va="center")

        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        rendered.append(buf)

    if not rendered:
        print("[UnifiedVsTruth] No figures generated.")
        return

    from PIL import Image as PILImage
    pil_imgs = [PILImage.open(b).convert("RGB") for b in rendered]
    max_w = max(im.width for im in pil_imgs)
    padded = []
    for im in pil_imgs:
        if im.width < max_w:
            canvas = PILImage.new("RGB", (max_w, im.height), (255, 255, 255))
            canvas.paste(im, (0, 0))
            padded.append(canvas)
        else:
            padded.append(im)

    total_h  = sum(im.height for im in padded)
    combined = PILImage.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for im in padded:
        combined.paste(im, (0, y))
        y += im.height

    ts  = time.strftime("%d%m%Y-%H%M%S")
    out = pathlib.Path(out_dir) / f"unified_vs_truth_{gt_channel}_{ts}.png"
    combined.save(str(out), dpi=(dpi, dpi))
    print(f"[UnifiedVsTruth] Saved: {out}")
