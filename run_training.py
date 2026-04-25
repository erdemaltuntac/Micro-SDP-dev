"""
run_training.py — Entry point for joint multi-channel dictionary learning.

Usage
-----
    cd /path/to/Micro-SDP/dev
    source ../myvenv/bin/activate
    python run_training.py

What it does
------------
1. Configure LearnConfig with the paper's hyper-parameters.
2. Load all five BSCCM-tiny channels via the BSCCM data pipeline.
3. Run Algorithm 2 (learn_joint_multichannel).
4. Save all results (dictionaries, codes, unified descriptors, CSV, JSON).
5. Evaluate per-channel reconstruction quality (Table 1 of paper).
6. Generate convergence plots.
7. Generate unified single-cell portrait (Section 9).
8. Generate multi-cell reconstruction grid.
9. Run biological validation (requires bsccm_real.py label files).
10. Generate unified-vs-truth comparison figure.
11. Generate orig/focused/reconstruction comparison figure.

Outputs land in:
    dev/training_results_Joint5ch_<timestamp>_seed<seed>/
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Ensure dev/ directory is on sys.path so that read_bsccm_data / focus_l1
# can be imported by the learning sub-modules.
# ---------------------------------------------------------------------------
DEV_DIR = pathlib.Path(__file__).resolve().parent
if str(DEV_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DIR))

import matplotlib
matplotlib.use("Agg")   # headless rendering

from learning.config       import LearnConfig, BSCCM_CHANNELS
from learning.data_loader  import load_all_channels
from learning.train_joint  import learn_joint_multichannel
from learning.results    import save_joint_results
from learning.evaluate     import evaluate_and_save_reconstructions, run_biological_validation
from learning.pdhg_solver  import prox_tv_nn_pdhg
from learning.plots        import (
    save_convergence_plots,
    save_single_unified_cell,
    save_unified_cell_figure,
    save_reconstructed_images,
    save_unified_vs_truth,
)


def main() -> None:
    """Joint multi-channel training on all five BSCCM LED-array channels."""

    stamp = time.strftime("%d-%m-%Y-%H-%M-%S")

    # ------------------------------------------------------------------ #
    #  Configuration                                                       #
    # ------------------------------------------------------------------ #
    cfg = LearnConfig(
        mu_tv=1.0,
        delta=1e-4,
        lam_tv_init=0.05,
        lam_tv_decay=3.0,
        tau_tv=0.25,
        sigma_tv=0.25,
        pdhg_iters=700,
        pdhg_tol=1e-7,
        outer_iters=30,
        outer_tol_dict=1e-8,
        outer_tol_obj=1e-6,
        outer_stop_patience=5,
        max_samples=0,
        shuffle=True,
        dict_kind="rand",
        seed=0,
        eps_relerr=0.05,
        target_fraction=0.95,
        min_x_norm=1e-3,
        ignore_low_norm=True,
    )

    # ------------------------------------------------------------------ #
    #  Load all five BSCCM channels                                        #
    # ------------------------------------------------------------------ #
    print("Loading BSCCM images for all channels ...")
    t_load_start = time.perf_counter()
    images_per_channel, train_indices = load_all_channels(
        location="BSCCM-tiny",
        tiny=True,
        channels=BSCCM_CHANNELS,
        n_images=0,
        use_focused=True,
        focus_method="gradient",
        output_dir="bsccm_out_image",
    )
    t_load_end = time.perf_counter()
    load_time_s = t_load_end - t_load_start

    ref_ch = list(images_per_channel.keys())[0]
    N_loaded, H, W = images_per_channel[ref_ch].shape
    n = H * W
    k = min(cfg.k_max, n)

    print(f"n={n}  K={k}  channels={list(images_per_channel.keys())}")
    print(
        f"\n{'='*60}\n"
        f"  [TIMING] Data loading + focus preprocessing\n"
        f"    channels : {len(images_per_channel)}\n"
        f"    images   : {N_loaded} per channel  "
        f"({N_loaded * len(images_per_channel)} total)\n"
        f"    spatial  : {H} x {W}  (n={n})\n"
        f"    elapsed  : {load_time_s:.2f} s\n"
        f"{'='*60}"
    )

    # ------------------------------------------------------------------ #
    #  Joint multi-channel training  (Algorithm 2)                        #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  [TIMING] Training started ...")
    print(f"{'='*60}")
    t_train_start = time.perf_counter()
    D_per_channel, A_per_channel, Phi, hist, per_image_rows = learn_joint_multichannel(
        images_per_channel=images_per_channel,
        k=k,
        cfg=cfg,
        dataset_indices=train_indices,
    )
    t_train_end = time.perf_counter()
    train_time_s = t_train_end - t_train_start

    n_outer_done = len(hist.get("fidelity", [cfg.outer_iters]))
    total_samples = N_loaded * len(images_per_channel) * n_outer_done
    throughput = total_samples / max(train_time_s, 1e-9)
    print(
        f"\n{'='*60}\n"
        f"  [TIMING] Joint training completed\n"
        f"    outer iterations : {n_outer_done}\n"
        f"    total elapsed    : {train_time_s:.1f} s  "
        f"({train_time_s/max(n_outer_done,1):.2f} s/epoch)\n"
        f"    throughput       : {throughput:.0f} samples/s\n"
        f"    (load + train)   : {load_time_s + train_time_s:.1f} s total\n"
        f"{'='*60}"
    )

    # ------------------------------------------------------------------ #
    #  Save results                                                      #
    # ------------------------------------------------------------------ #
    out_dir = str(DEV_DIR / f"training_results_Joint5ch_{stamp}_seed{cfg.seed}")

    hist["load_time_s"]  = round(load_time_s, 3)
    hist["train_time_s"] = round(train_time_s, 3)

    save_joint_results(
        out_dir=out_dir,
        D_per_channel=D_per_channel,
        A_per_channel=A_per_channel,
        Phi=Phi,
        images_per_channel=images_per_channel,
        indices=train_indices,
        cfg=cfg,
        per_image_rows=per_image_rows,
        hist=hist,
    )

    # ------------------------------------------------------------------ #
    #  Per-channel evaluation (Table 1 of paper)                          #
    # ------------------------------------------------------------------ #
    for ch, imgs_ch in images_per_channel.items():
        evaluate_and_save_reconstructions(
            out_dir=out_dir,
            images=imgs_ch,
            indices=train_indices,
            D=D_per_channel[ch],
            A=A_per_channel[ch],
            cfg=cfg,
            channel=ch,
            hist=hist,
        )

    # ------------------------------------------------------------------ #
    #  Convergence plots                                                   #
    # ------------------------------------------------------------------ #
    rep_ch  = list(images_per_channel.keys())[0]
    rep_img = images_per_channel[rep_ch][train_indices[0]].reshape(H, W)
    lam_tv_plot = float(cfg.mu_tv * cfg.delta)
    _, pdhg_res = prox_tv_nn_pdhg(
        x_datum=rep_img,
        lam_tv=lam_tv_plot,
        n_iters=cfg.plot_pdhg_iters,
        tau_tv=cfg.tau_tv,
        sigma_tv=cfg.sigma_tv,
        tol=cfg.plot_pdhg_tol,
        return_residuals=True,
    )
    save_convergence_plots(
        hist,
        output_dir=str(pathlib.Path(out_dir) / "error_analysis"),
        pdhg_residuals=pdhg_res,
        per_image_rows=per_image_rows,
        channels=list(images_per_channel.keys()),
        eps_relerr=cfg.eps_relerr,
    )

    # ------------------------------------------------------------------ #
    #  Unified single-cell portrait  (Section 9)                          #
    # ------------------------------------------------------------------ #
    portrait_label_dir   = DEV_DIR / "bsccm_real_out"
    li_path         = portrait_label_dir / "label_indices.npy"
    portrait_idx    = 0
    if li_path.exists():
        label_idxs    = np.load(str(li_path)).astype(int)
        train_idx_map = {int(idx): row for row, idx in enumerate(train_indices)}
        for ds_idx in label_idxs:
            if int(ds_idx) in train_idx_map:
                portrait_idx = train_idx_map[int(ds_idx)]
                print(f"[Portrait] Using labeled cell ds_idx={ds_idx} "
                      f"(row={portrait_idx}) for unified portrait.")
                break

    save_single_unified_cell(
        D_per_channel=D_per_channel,
        A_per_channel=A_per_channel,
        images_per_channel=images_per_channel,
        cfg=cfg,
        out_dir=out_dir,
        cell_idx=portrait_idx,
    )

    # ------------------------------------------------------------------ #
    #  Multi-cell reconstruction grid  (Section 8.6)                      #
    # ------------------------------------------------------------------ #
    save_unified_cell_figure(
        D_per_channel=D_per_channel,
        A_per_channel=A_per_channel,
        images_per_channel=images_per_channel,
        out_dir=out_dir,
        n_cells=cfg.n_cells_grid,
    )

    # ------------------------------------------------------------------ #
    #  Biological validation                                               #
    # ------------------------------------------------------------------ #
    bio_label_dir            = DEV_DIR / "bsccm_real_out"
    label_indices_path       = str(bio_label_dir / "label_indices.npy")
    label_class_labels_path  = str(bio_label_dir / "label_class_labels.npy")

    if (pathlib.Path(label_indices_path).exists() and
            pathlib.Path(label_class_labels_path).exists()):
        run_biological_validation(
            Phi=Phi,
            train_indices=train_indices,
            label_indices_path=label_indices_path,
            label_class_labels_path=label_class_labels_path,
            out_dir=str(pathlib.Path(out_dir) / "bio_validation"),
        )
        for gt_ch in images_per_channel.keys():
            save_unified_vs_truth(
                D_per_channel=D_per_channel,
                A_per_channel=A_per_channel,
                images_per_channel=images_per_channel,
                train_indices=train_indices,
                out_dir=out_dir,
                gt_channel=gt_ch,
            )
    else:
        print(
            f"\n[BioValid] Label files not found in {bio_label_dir}.\n"
            "  Run bsccm_real.py first to generate label_indices.npy "
            "and label_class_labels.npy."
        )

    # ------------------------------------------------------------------ #
    #  Reconstructed-image figure                                          #
    # ------------------------------------------------------------------ #
    save_reconstructed_images(
        D_per_channel=D_per_channel,
        A_per_channel=A_per_channel,
        images_per_channel=images_per_channel,
        indices=train_indices,
        out_dir=out_dir,
        lam_tv=lam_tv_plot,
        cfg=cfg,
    )

    # ------------------------------------------------------------------ #
    #  Summary CSVs                                                        #
    # ------------------------------------------------------------------ #
    summary_rows = [
        {
            "channel": ch, "Phi_dim": int(Phi.shape[1]), "K": k, "n": n,
            "load_time_s": round(load_time_s, 3),
            "train_time_s": round(train_time_s, 3),
            "out_dir": out_dir,
        }
        for ch in images_per_channel.keys()
    ]
    summary_csv = str(pathlib.Path(out_dir) / "joint_run_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    timing_fields = [
        "load_time_s", "train_time_s", "total_time_s",
        "n_outer_iters", "n_images", "n_channels", "K", "n",
        "s_per_epoch", "samples_per_s",
    ]
    timing_row = {
        "load_time_s":   round(load_time_s, 3),
        "train_time_s":  round(train_time_s, 3),
        "total_time_s":  round(load_time_s + train_time_s, 3),
        "n_outer_iters": n_outer_done,
        "n_images":      N_loaded,
        "n_channels":    len(images_per_channel),
        "K":             k,
        "n":             n,
        "s_per_epoch":   round(train_time_s / max(n_outer_done, 1), 3),
        "samples_per_s": round(throughput, 1),
    }
    with open(str(pathlib.Path(out_dir) / "timing.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=timing_fields)
        w.writeheader()
        w.writerow(timing_row)

    print(f"\n{'='*60}")
    print(f"  All results in: {out_dir}/")
    print(f"  Files:")
    for p in sorted(pathlib.Path(out_dir).rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"    {p.relative_to(out_dir)}  ({size_kb:.1f} kB)")
    print(f"{'='*60}")
    print(f"  Unified descriptor Phi: shape {Phi.shape}")
    print(f"  Ready for downstream classification.")


if __name__ == "__main__":
    main()
