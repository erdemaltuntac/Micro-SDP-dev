# --------------------------------------------------------------------------
# BSCCM FlowCytometry Dataset — Real Labeled Data
# References:
# [1] https://github.com/Waller-Lab/BSCCM/blob/main/Getting_started.ipynb
# [2] https://datadryad.org/dataset/doi:10.5061/dryad.sxksn038s
# [3] https://arxiv.org/pdf/2402.06191
# --------------------------------------------------------------------------
"""
Download (once) and work with the full BSCCM dataset that includes
cell-type classification labels (3-class: Lymphocyte / Granulocyte / Monocyte).

The tiny dataset does NOT contain classification labels.
The full dataset is ~several GB; the download runs only once.

Usage:
    python bsccm_real.py
"""

from bsccm import BSCCM
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os, sys, time
from focus_l1 import get_focus_analysis_for_image
from matplotlib.patches import Rectangle

timestamp = time.strftime("%d%m%Y-%H%M%S")

# ------------------------------------------------- Paths -------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
DATASET_DIR  = SCRIPT_DIR / "BSCCM-tiny"
OUTPUT_DIR   = SCRIPT_DIR / "bsccm_real_out"   # output directory for figures and label arrays
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3-class label names (as defined by the BSCCM authors)
LABEL_NAMES = {0: "Lymphocyte", 1: "Granulocyte", 2: "Monocyte"}

# LED-array channels we care about
LED_CHANNELS = ["DPC_Left", "DPC_Right", "DPC_Top", "DPC_Bottom", "Brightfield"]

# ------------------------------------------------- Dataset loading -------------------------------------------------

def load_dataset() -> BSCCM:
    """
    Open the BSCCM-tiny dataset (already downloaded locally).
    It contains 1000 cells at 128x128 px, including classification labels.
    """
    metadata_file = DATASET_DIR / "BSCCM_global_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_DIR}.\n"
            "Expected the BSCCM-tiny folder to already be present."
        )
    print(f"Opening dataset at: {DATASET_DIR}")
    return BSCCM(str(DATASET_DIR))


# ------------------------------------------------- Label loading -------------------------------------------------

def load_labels(dataset: BSCCM, ten_class_version: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    indices      : np.ndarray - cell indices that have a label
    class_labels : np.ndarray - integer label (0/1/2) per cell
    """
    indices, class_labels = dataset.get_cell_type_classification_data(
        ten_class_version=ten_class_version
    )

    if ten_class_version:
        label_names = {i: str(i) for i in np.unique(class_labels)}
    else:
        label_names = LABEL_NAMES

    labelled_indices = list(zip(indices, class_labels))

    for k, name in label_names.items():
        count = int(np.sum(class_labels == k))
        print(f"  Class {k}  ({name:<12}): {count:>5} cells")
    print(f"  Total labelled cells : {len(indices)}")
    return labelled_indices, label_names


# ------------------------------------------------- Visualisation -------------------------------------------------

def save_label_distribution(class_labels: np.ndarray):
    """Bar chart of the 3-class label counts."""
    counts = {LABEL_NAMES[k]: int(np.sum(class_labels == k)) for k in LABEL_NAMES}

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "tomato", "mediumseagreen"]
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor="white", linewidth=0.8)

    max_val = max(counts.values())
    ax.set_ylim(0, max_val * 1.2)

    for bar, val in zip(bars, counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            str(val),
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    ax.set_title("BSCCM - Class Label Distribution\n(BSCCM-tiny subset, N=28 labelled cells)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Cell type", fontsize=12)
    ax.set_ylabel("Cell count", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout(pad=1.5)
    out = OUTPUT_DIR / f"label_distribution_{timestamp}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def save_example_grid(dataset: BSCCM, indices: np.ndarray, class_labels: np.ndarray,
                      n_per_class: int = 6, channel: str = "DPC_Left"):
    """
    One figure per class. Each figure contains exactly 2 rows x n_avail columns:
      Row 0: original images with focus bounding box overlay (raw pixel values)
      Row 1: focused crops normalized to [0,1] via per-image min-max,
             matching the training data normalization.

    - Figure width scales to n_avail columns (left-aligned, no padding).
    - Axes show pixel coordinate ticks (3 ticks: 0, mid, max) on both axes.
    - Tick labels are small but readable.
    - Row identity shown as y-axis label on column 0.
    - All per-class figures are saved into a single combined PNG by stacking
      them vertically using PIL after individual renders.
    """
    from matplotlib.gridspec import GridSpec
    from PIL import Image as PILImage
    import io

    counts = {LABEL_NAMES[k]: int(np.sum(class_labels == k)) for k in LABEL_NAMES}
    print(f"\nChannel '{channel}': {counts}")

    cell_h = 2.5    # inches per image row
    cell_w = 2.5    # inches per image column
    dpi    = 300    # render DPI for individual class figures

    rendered_buffers = []   # in-memory PNG buffers per class

    for k, name in LABEL_NAMES.items():
        n_avail      = int(np.sum(class_labels == k))
        cell_indices = indices[class_labels == k]

        print(f"  Class {k}  ({name:<12}): {n_avail} cells")

        fig_w = n_avail * cell_w
        fig_h = 2 * cell_h + 0.6   # +0.6 for class title headroom

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(
            f"{name}  —  {n_avail} cells  |  channel: {channel}",
            fontsize=11, fontweight="bold",
            x=0.01, ha="left", y=0.98
        )

        gs = GridSpec(2, n_avail, figure=fig, hspace=0.12, wspace=0.15,
                      top=0.88, bottom=0.06, left=0.06, right=0.99)

        for col in range(n_avail):
            idx = int(cell_indices[col])
            img = dataset.read_image(idx, channel=channel)
            H_orig, W_orig = img.shape[:2]

            focus_result = get_focus_analysis_for_image(
                img, idx, channel, method="gradient"
            )
            focused_img             = focus_result["focused_image"]
            H_foc, W_foc            = focused_img.shape[:2]
            (minr, maxr, minc, maxc) = focus_result["bounding_box"]

            # ── Row 0: original image ──────────────────────────────────
            ax_orig = fig.add_subplot(gs[0, col])
            ax_orig.imshow(img, cmap="inferno", aspect="equal")
            rect = Rectangle(
                (minc, minr),
                width=maxc - minc + 1,
                height=maxr - minr + 1,
                fill=False, linewidth=1.2,
                linestyle="--", edgecolor="white"
            )
            ax_orig.add_patch(rect)

            # Readable ticks: 0, midpoint, max
            ax_orig.set_xticks([0, W_orig // 2, W_orig - 1])
            ax_orig.set_yticks([0, H_orig // 2, H_orig - 1])
            ax_orig.tick_params(labelsize=10, length=3, pad=2)
            ax_orig.xaxis.set_tick_params(labelbottom=True)

            # Row label on column 0 only — horizontal, outside left
            if col == 0:
                ax_orig.set_ylabel("original", fontsize=12, fontweight="bold",
                                   rotation=90, labelpad=38, va="center")

            # ── Row 1: focused crop — per-image min-max to [0,1] ──────
            focused_disp = focused_img.astype(np.float64)
            f_min = float(np.min(focused_disp))
            f_max = float(np.max(focused_disp))
            if f_max > f_min:
                focused_disp = (focused_disp - f_min) / (f_max - f_min)
            ax_foc = fig.add_subplot(gs[1, col])
            ax_foc.imshow(focused_disp, cmap="inferno", aspect="equal",
                          vmin=0.0, vmax=1.0)

            ax_foc.set_xticks([0, W_foc // 2, W_foc - 1])
            ax_foc.set_yticks([0, H_foc // 2, H_foc - 1])
            ax_foc.tick_params(labelsize=10, length=3, pad=2)

            if col == 0:
                ax_foc.set_ylabel("focused", fontsize=12, fontweight="bold",
                                  rotation=90, labelpad=38, va="center")

        # Render to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        rendered_buffers.append(buf)

    # ── Stack all class figures vertically into one combined PNG ──────
    pil_imgs = [PILImage.open(b).convert("RGB") for b in rendered_buffers]

    # Unify widths by padding narrower images on the right with white
    max_w = max(im.width for im in pil_imgs)
    padded = []
    for im in pil_imgs:
        if im.width < max_w:
            canvas = PILImage.new("RGB", (max_w, im.height), (255, 255, 255))
            canvas.paste(im, (0, 0))
            padded.append(canvas)
        else:
            padded.append(im)

    total_h = sum(im.height for im in padded)
    combined = PILImage.new("RGB", (max_w, total_h), (255, 255, 255))
    y_offset = 0
    for im in padded:
        combined.paste(im, (0, y_offset))
        y_offset += im.height

    out = OUTPUT_DIR / f"labeled_examples_{channel}_{timestamp}.png"
    combined.save(str(out), dpi=(dpi, dpi))
    print(f"Saved: {out}")


def save_labels_npy(indices: np.ndarray, class_labels: np.ndarray):
    """
    Persist label indices and class labels as .npy for downstream use
    (biological validation in dictionary_learning_algorithm_TV.py).
    """
    np.save(OUTPUT_DIR / "label_indices.npy",      indices)
    np.save(OUTPUT_DIR / "label_class_labels.npy", class_labels)
    print(f"Saved label arrays to: {OUTPUT_DIR}")


def save_unified_vs_truth_grid(
    dataset: BSCCM,
    indices: np.ndarray,
    class_labels: np.ndarray,
    artifact_dir: Path,
    channel: str = "Brightfield",
):
    """
    For each labeled cell, place its Brightfield focused crop (ground-truth)
    next to its unified image u_j = D ā_j (dictionary reconstruction),
    organized by cell type.

    Layout per class:
      Row 0: Brightfield focused crop  (ground-truth)
      Row 1: unified image u_j = D ā_j (weighted code-space unification)

    Index alignment:
      train_indices[row] maps training row -> BSCCM dataset index.
      Only labeled cells that appear in the training set are shown.
      The BSCCM dataset index is used to load the ground-truth image and
      to find the corresponding row in D, A_per_channel.

    Parameters
    ----------
    dataset      : open BSCCM dataset object
    indices      : (N_labeled,) BSCCM dataset indices of labeled cells
    class_labels : (N_labeled,) integer class labels (0/1/2)
    artifact_dir : path to training artifact directory containing
                   dictionary_D_shared.npy, codes_A_*.npy, train_indices.txt
    channel      : which channel to use as ground-truth (default: Brightfield)
    """
    from matplotlib.gridspec import GridSpec
    from PIL import Image as PILImage
    import io

    CHANNELS_TRAIN = ["DPC_Left", "DPC_Right", "DPC_Top", "DPC_Bottom", "Brightfield"]

    # ── Load training artifacts ──────────────────────────────────────────
    artifact_dir = Path(artifact_dir)
    D_path = artifact_dir / "dictionary_D_shared.npy"
    if not D_path.exists():
        print(f"[UnifiedVsTruth] dictionary_D_shared.npy not found in {artifact_dir}")
        return

    D = np.load(str(D_path))                               # (n, K)
    n, K = D.shape

    A_per_channel = {}
    for ch in CHANNELS_TRAIN:
        p = artifact_dir / f"codes_A_{ch}.npy"
        if not p.exists():
            print(f"[UnifiedVsTruth] Missing {p} — skipping.")
            return
        A_per_channel[ch] = np.load(str(p))               # (N_train, K)

    idx_path = artifact_dir / "train_indices.txt"
    train_indices = [int(l.strip()) for l in
                     idx_path.read_text().splitlines() if l.strip()]
    train_idx_to_row = {int(idx): row for row, idx in enumerate(train_indices)}

    H = W = int(round(n ** 0.5))                          # assumes square images
    eps_w = 1e-8

    # ── Find labeled cells present in training set ───────────────────────
    valid_mask = np.array([int(idx) in train_idx_to_row for idx in indices])
    if valid_mask.sum() == 0:
        print("[UnifiedVsTruth] No labeled cells found in training set.")
        return

    valid_indices = indices[valid_mask]
    valid_labels  = class_labels[valid_mask]
    print(f"[UnifiedVsTruth] {int(valid_mask.sum())} labeled cells matched to training rows")

    # ── Build figure per class ────────────────────────────────────────────
    cell_h = 2.5
    cell_w = 2.5
    dpi    = 300
    rendered_buffers = []

    for k, name in LABEL_NAMES.items():
        cell_ds_indices = valid_indices[valid_labels == k]
        n_avail = len(cell_ds_indices)
        if n_avail == 0:
            continue

        fig_w = n_avail * cell_w
        fig_h = 2 * cell_h + 0.7

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(
            f"{name}  —  {n_avail} cells  |  GT: {channel}  vs  u_j = Dā_j",
            fontsize=11, fontweight="bold",
            x=0.01, ha="left", y=0.99
        )

        gs = GridSpec(2, n_avail, figure=fig, hspace=0.08, wspace=0.12,
                      top=0.88, bottom=0.06, left=0.07, right=0.99)

        for col, ds_idx in enumerate(cell_ds_indices):
            ds_idx = int(ds_idx)
            row    = train_idx_to_row[ds_idx]

            # ── Row 0: Brightfield focused crop (ground-truth) ───────────
            raw_img = dataset.read_image(ds_idx, channel=channel)
            focus_result = get_focus_analysis_for_image(
                raw_img, ds_idx, channel, method="gradient"
            )
            focused_img = focus_result["focused_image"].astype(np.float64)
            f_min, f_max = focused_img.min(), focused_img.max()
            if f_max > f_min:
                focused_img = (focused_img - f_min) / (f_max - f_min)

            ax_gt = fig.add_subplot(gs[0, col])
            ax_gt.imshow(focused_img, cmap="inferno", aspect="equal",
                         vmin=0.0, vmax=1.0)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            ax_gt.set_title(f"idx {ds_idx}", fontsize=7, pad=2)
            if col == 0:
                ax_gt.set_ylabel(f"GT\n({channel})", fontsize=9,
                                 fontweight="bold", rotation=90,
                                 labelpad=4, va="center")

            # ── Row 1: unified image u_j = D ā_j ─────────────────────────
            # Compute inverse-L2-residual weighted mean code ā_j
            weights = []
            for ch in CHANNELS_TRAIN:
                a_c   = A_per_channel[ch][row]             # (K,)
                # Use training image stored in A: approximate residual
                # as ||Da - Da||=0 is not useful — use code energy instead.
                # Weight = 1 / (||a_c||^2 + eps): atoms with small codes
                # (poor fit) contribute less. This is a proxy when X is
                # not stored; if X is available use L2 residual instead.
                w_c = 1.0 / (float(np.dot(a_c, a_c)) + eps_w)
                weights.append(w_c)

            weights = np.array(weights, dtype=np.float64)
            weights /= weights.sum()

            a_bar = np.zeros(K, dtype=np.float64)
            for ci, ch in enumerate(CHANNELS_TRAIN):
                a_bar += weights[ci] * A_per_channel[ch][row]

            u_j = (D @ a_bar).reshape(H, W)
            u_min, u_max = u_j.min(), u_j.max()
            if u_max > u_min:
                u_j = (u_j - u_min) / (u_max - u_min)

            ax_u = fig.add_subplot(gs[1, col])
            ax_u.imshow(u_j, cmap="inferno", aspect="equal",
                        vmin=0.0, vmax=1.0)
            ax_u.set_xticks([])
            ax_u.set_yticks([])
            if col == 0:
                ax_u.set_ylabel("u_j = Dā_j", fontsize=9,
                                fontweight="bold", rotation=90,
                                labelpad=4, va="center")

        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        rendered_buffers.append(buf)

    if not rendered_buffers:
        print("[UnifiedVsTruth] No figures generated.")
        return

    # ── Stack vertically ─────────────────────────────────────────────────
    pil_imgs = [PILImage.open(b).convert("RGB") for b in rendered_buffers]
    max_w    = max(im.width for im in pil_imgs)
    padded   = []
    for im in pil_imgs:
        if im.width < max_w:
            canvas = PILImage.new("RGB", (max_w, im.height), (255, 255, 255))
            canvas.paste(im, (0, 0))
            padded.append(canvas)
        else:
            padded.append(im)

    total_h  = sum(im.height for im in padded)
    combined = PILImage.new("RGB", (max_w, total_h), (255, 255, 255))
    y_off    = 0
    for im in padded:
        combined.paste(im, (0, y_off))
        y_off += im.height

    ts  = time.strftime("%d%m%Y-%H%M%S")
    out = OUTPUT_DIR / f"unified_vs_truth_{channel}_{ts}.png"
    combined.save(str(out), dpi=(dpi, dpi))
    print(f"[UnifiedVsTruth] Saved: {out}")



    """
    Persist indices and labels as .npy for downstream use.
    """
    np.save(OUTPUT_DIR / "label_indices.npy",      indices)
    np.save(OUTPUT_DIR / "label_class_labels.npy", class_labels)
    print(f"Saved label arrays to: {OUTPUT_DIR}")


# ------------------------------------------------- Main -------------------------------------------------

if __name__ == "__main__":
    # 1. Open the dataset BSCCM-tiny (already downloaded locally)
    dataset = load_dataset()

    # 2. Load 3-class labels
    print("\nLoading cell-type classification labels...")
    labelled_indices, label_names = load_labels(dataset)

    indices      = np.array([x[0] for x in labelled_indices])
    class_labels = np.array([x[1] for x in labelled_indices])

    # 3. Persist labels as .npy
    save_labels_npy(indices, class_labels)

    # 4. Label distribution bar chart
    save_label_distribution(class_labels)

    # 5. Save labeled data grid for each LED channel
    for ch in LED_CHANNELS:
        save_example_grid(dataset, indices, class_labels, n_per_class=6, channel=ch)

    print("\nDone. All outputs written to:", OUTPUT_DIR)