"""
run_post_training_figures.py

Generate all post-training figures and biological validation from saved
artifacts, without retraining. Use this when training completed but
figure generation crashed.

Usage:
    python run_post_training_figures.py <artifact_dir>

Example:
    python run_post_training_figures.py \
        training_artifacts_Joint5ch_20260420T234823_seed0

Generates:
    unified_single_cell_<row>_<timestamp>.png
    unified_cell_figure_<timestamp>.png
    bio_validation/bio_validation_metrics.json
    bio_validation/bio_validation_metrics.png
    unified_vs_truth_Brightfield_<timestamp>.png
    reconstructed_image_<timestamp>.png
"""

import sys
import json
import pathlib
import numpy as np

from dictionary_learning_algorithm_TV import (
    save_single_unified_cell,
    save_unified_cell_figure,
    save_reconstructed_images,
    save_unified_vs_truth,
    run_biological_validation,
    load_all_channels,
    LearnConfig,
    BSCCM_CHANNELS,
)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CHANNELS   = ["DPC_Left", "DPC_Right", "DPC_Top", "DPC_Bottom", "Brightfield"]


def load_artifacts(artifact_dir: pathlib.Path):
    """Load all saved training artifacts."""
    print(f"\n[Load] {artifact_dir}")

    # Per-channel dictionaries
    D_per_channel = {}
    for ch in CHANNELS:
        p = artifact_dir / f"dictionary_D_{ch}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        D_per_channel[ch] = np.load(str(p))
    print(f"  D_per_channel: {list(D_per_channel.keys())}  "
          f"shape={D_per_channel[CHANNELS[0]].shape}")

    # Per-channel codes
    A_per_channel = {}
    for ch in CHANNELS:
        p = artifact_dir / f"codes_A_{ch}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        A_per_channel[ch] = np.load(str(p))
    print(f"  A_per_channel: shape={A_per_channel[CHANNELS[0]].shape}")

    # Unified descriptors
    Phi = np.load(str(artifact_dir / "unified_descriptors_Phi.npy"))
    print(f"  Phi: shape={Phi.shape}")

    # Train indices
    train_indices = [
        int(l.strip())
        for l in (artifact_dir / "train_indices.txt").read_text().splitlines()
        if l.strip()
    ]
    print(f"  train_indices: {len(train_indices)} entries")

    # Config
    cfg_dict = json.loads((artifact_dir / "config.json").read_text())
    cfg = LearnConfig(
        mu_tv=cfg_dict.get("mu_tv", 1.0),
        delta=cfg_dict.get("delta", 1e-4),
        lam_tv_init=cfg_dict.get("lam_tv_init", 0.05),
        lam_tv_decay=cfg_dict.get("lam_tv_decay", 3.0),
        tau_tv=cfg_dict.get("tau_tv", 0.25),
        sigma_tv=cfg_dict.get("sigma_tv", 0.25),
        pdhg_iters=cfg_dict.get("pdhg_iters", 700),
        pdhg_tol=cfg_dict.get("pdhg_tol", 1e-7),
        outer_iters=cfg_dict.get("outer_iters", 30),
        outer_tol_dict=cfg_dict.get("outer_tol_dict", 1e-8),
        outer_tol_obj=cfg_dict.get("outer_tol_obj", 1e-6),
        outer_stop_patience=cfg_dict.get("outer_stop_patience", 5),
        max_samples=cfg_dict.get("max_samples", 0),
        shuffle=cfg_dict.get("shuffle", True),
        dict_kind=cfg_dict.get("dict_kind", "rand"),
        seed=cfg_dict.get("seed", 0),
        eps_relerr=cfg_dict.get("eps_relerr", 0.05),
        target_fraction=cfg_dict.get("target_fraction", 0.95),
        min_x_norm=cfg_dict.get("min_x_norm", 1e-3),
        ignore_low_norm=cfg_dict.get("ignore_low_norm", True),
    )
    print(f"  Config loaded")

    return D_per_channel, A_per_channel, Phi, train_indices, cfg


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_post_training_figures.py <artifact_dir>")
        sys.exit(1)

    artifact_dir = pathlib.Path(sys.argv[1]).resolve()
    if not artifact_dir.exists():
        print(f"ERROR: {artifact_dir} does not exist")
        sys.exit(1)

    out_dir = str(artifact_dir)

    # ------------------------------------------------------------------ #
    # Load artifacts
    # ------------------------------------------------------------------ #
    D_per_channel, A_per_channel, Phi, train_indices, cfg = load_artifacts(artifact_dir)

    # ------------------------------------------------------------------ #
    # Reload training images (needed for figures and unified_vs_truth)
    # ------------------------------------------------------------------ #
    print("\n[Load] Reloading training images from BSCCM-tiny...")
    images_per_channel, _ = load_all_channels(
        location="BSCCM-tiny",
        tiny=True,
        channels=BSCCM_CHANNELS,
        n_images=0,
        use_focused=True,
        focus_method="gradient",
        output_dir="bsccm_out_image",
    )

    # ------------------------------------------------------------------ #
    # Select portrait cell — first labeled cell in training set
    # ------------------------------------------------------------------ #
    bio_label_dir = SCRIPT_DIR / "bsccm_real_out"
    li_path       = bio_label_dir / "label_indices.npy"
    portrait_idx  = 0
    if li_path.exists():
        label_idxs    = np.load(str(li_path)).astype(int)
        train_idx_map = {int(idx): row for row, idx in enumerate(train_indices)}
        for ds_idx in label_idxs:
            if int(ds_idx) in train_idx_map:
                portrait_idx = train_idx_map[int(ds_idx)]
                print(f"[Portrait] Using labeled cell ds_idx={ds_idx} "
                      f"(row={portrait_idx})")
                break

    # ------------------------------------------------------------------ #
    # 1. Unified single-cell portrait
    # ------------------------------------------------------------------ #
    print("\n[1] Saving unified single-cell portrait...")
    save_single_unified_cell(
        D_per_channel=D_per_channel,
        A_per_channel=A_per_channel,
        images_per_channel=images_per_channel,
        cfg=cfg,
        out_dir=out_dir,
        cell_idx=portrait_idx,
    )

    # ------------------------------------------------------------------ #
    # 2. Unified cell figure (multi-cell, multi-channel)
    # ------------------------------------------------------------------ #
    print("\n[2] Saving unified cell figure...")
    save_unified_cell_figure(
        D_per_channel=D_per_channel,
        A_per_channel=A_per_channel,
        images_per_channel=images_per_channel,
        out_dir=out_dir,
        n_cells=5,
    )

    # ------------------------------------------------------------------ #
    # 3. Biological validation
    # ------------------------------------------------------------------ #
    print("\n[3] Running biological validation...")
    label_indices_path      = str(bio_label_dir / "label_indices.npy")
    label_class_labels_path = str(bio_label_dir / "label_class_labels.npy")
    if (pathlib.Path(label_indices_path).exists() and
            pathlib.Path(label_class_labels_path).exists()):
        run_biological_validation(
            Phi=Phi,
            train_indices=train_indices,
            label_indices_path=label_indices_path,
            label_class_labels_path=label_class_labels_path,
            out_dir=str(artifact_dir / "bio_validation"),
        )
    else:
        print("  Label files not found — run bsccm_real.py first.")

    # ------------------------------------------------------------------ #
    # 4. Unified vs ground-truth comparison
    # ------------------------------------------------------------------ #
    print("\n[4] Saving unified vs ground-truth comparison (all channels)...")
    for _gt_ch in images_per_channel.keys():
        save_unified_vs_truth(
            D_per_channel=D_per_channel,
            A_per_channel=A_per_channel,
            images_per_channel=images_per_channel,
            train_indices=train_indices,
            out_dir=out_dir,
            gt_channel=_gt_ch,
        )

    # ------------------------------------------------------------------ #
    # 5. Reconstructed image figure
    # ------------------------------------------------------------------ #
    print("\n[5] Saving reconstructed image figure...")
    lam_tv = float(cfg.mu_tv * cfg.delta)
    save_reconstructed_images(
        D_per_channel=D_per_channel,
        A_per_channel=A_per_channel,
        images_per_channel=images_per_channel,
        indices=train_indices,
        out_dir=out_dir,
        lam_tv=lam_tv,
        cfg=cfg,
    )

    print(f"\nDone. All figures saved to: {artifact_dir}/")


if __name__ == "__main__":
    main()
