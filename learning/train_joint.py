"""
train_joint.py - Algorithm 2: joint multi-channel dictionary learning.

Algorithm 2 wraps Algorithm 1 as its inner step: for each outer iteration the
same per-channel PDHG pass (channel_inner_pass from train_single) is applied
to every channel under a shared lam_tv schedule and a shared convergence
criterion, before each channel's dictionary is updated via Procrustes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config          import LearnConfig
from .dictionary_init import dictionary
from .stiefel         import procrustes_update
from .train_single    import channel_inner_pass


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def print_joint_quality_report(
    X: Dict[str, np.ndarray],
    D_per_channel: Dict[str, np.ndarray],
    A_per_channel: Dict[str, np.ndarray],
    idxs: np.ndarray,
    n_used: int,
    cfg: LearnConfig,
    channels: List[str],
) -> None:
    """Print per-channel reconstruction quality after joint training."""
    print("\n[Joint] Final reconstruction quality per channel:")
    for ch in channels:
        D_ch = D_per_channel[ch]
        rel_errs = []
        for ii in range(n_used):
            j = int(idxs[ii])
            x_norm = float(np.linalg.norm(X[ch][j]))
            if x_norm >= cfg.min_x_norm:
                recon = D_ch @ A_per_channel[ch][j]
                rel_errs.append(float(np.linalg.norm(X[ch][j] - recon) / x_norm))
        if rel_errs:
            arr = np.array(rel_errs)
            fidelity_mean = float(np.mean(100.0 * (1.0 - arr)))
            print(
                f"  {ch:20s}  mean={np.mean(arr):.4f}  "
                f"Fidelity(mean)={fidelity_mean:.2f}%"
            )


# ---------------------------------------------------------------------------
# Algorithm 2
# ---------------------------------------------------------------------------

def learn_joint_multichannel(
    images_per_channel: Dict[str, np.ndarray],
    k: int,
    cfg: LearnConfig,
    dataset_indices: Optional[List[int]] = None,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
    Dict[str, List[float]],
    List[Dict],
]:
    """
    Algorithm 2: joint multi-channel dictionary learning and cell feature
    unification (Section 9 of the paper).
    """
    channels = list(images_per_channel.keys())
    C = len(channels)
    if C == 0:
        raise ValueError("images_per_channel must contain at least one channel.")

    ref_shape = images_per_channel[channels[0]].shape
    for ch in channels:
        if images_per_channel[ch].shape != ref_shape:
            raise ValueError(
                f"Channel {ch} has shape {images_per_channel[ch].shape}, "
                f"expected {ref_shape}."
            )

    N, H, W = ref_shape
    n = H * W

    if dataset_indices is None:
        dataset_indices = list(range(N))

    assert cfg.tau_tv * cfg.sigma_tv * 8.0 < 1.0, (
        f"tau_tv*sigma_tv*8 = {cfg.tau_tv*cfg.sigma_tv*8:.4f} >= 1."
    )

    lam_tv_floor = float(cfg.mu_tv * cfg.delta)
    print(
        f"[Joint-{C}ch] lambda_TV: init={cfg.lam_tv_init:.3e}  "
        f"decay={cfg.lam_tv_decay}  floor={lam_tv_floor:.3e}  "
        f"K={k}  n={n}  N={N}  C={C}  channels={channels}"
    )

    X: Dict[str, np.ndarray] = {
        ch: images_per_channel[ch].reshape(N, n).astype(np.float64)
        for ch in channels
    }

    # Independent per-channel initialisation (distinct seeds)
    D_per_channel: Dict[str, np.ndarray] = {
        ch: dictionary(n=n, k=k, kind=cfg.dict_kind, seed=cfg.seed + ci)
        for ci, ch in enumerate(channels)
    }
    A_per_channel: Dict[str, np.ndarray] = {
        ch: np.zeros((N, k), dtype=np.float64) for ch in channels
    }
    Y_per_channel: Dict[str, np.ndarray] = {
        ch: np.zeros((N, n), dtype=np.float64) for ch in channels
    }

    hist: Dict[str, List[float]] = {
        "obj": [], "tv_energy": [], "fidelity": [],
        "nonneg_violation": [], "grad_norm": [], "dict_change": [],
        "rel_err_mean": [], "eta": [],
        "pdhg_mean_final_res": [], "pdhg_max_final_res": [],
        "pdhgmean_iters": [], "pdhg_converged_frac": [],
    }
    per_image_rows: List[Dict] = []

    idxs = np.arange(N)
    consecutive_stop = 0
    obj_prev = float("inf")
    n_used = N  # defined here so it is always bound after the loop

    for t in range(cfg.outer_iters):
        # ---------------------------------------------------------------
        # TODO: Investigate data-adaptive lam_tv schedule (e.g. based on
        #       primal/dual residuals or relative objective change).
        lam_tv = max(lam_tv_floor, cfg.lam_tv_init / (1.0 + cfg.lam_tv_decay * t))
        # ---------------------------------------------------------------
        if cfg.shuffle:
            np.random.default_rng(cfg.seed + t).shuffle(idxs)

        n_used = N if cfg.max_samples <= 0 else min(cfg.max_samples, N)

        tv_acc = fid_acc = obj_acc = nonneg_acc = rel_err_acc = 0.0
        n_total = 0
        all_pdhg_res: List[float] = []
        all_pdhg_iters: List[int] = []

        # Step 1 - inner PDHG pass for every channel (Algorithm 1 inner step)
        for ch in channels:
            (pfr, piu,
             tv_ch, fid_ch, obj_ch, nonneg_ch, rel_ch,
             rows) = channel_inner_pass(
                X[ch], D_per_channel[ch],
                A_per_channel[ch], Y_per_channel[ch],
                idxs, n_used, lam_tv, t + 1,
                cfg, ch, dataset_indices, H, W,
                images_orig=None, 
            )
            all_pdhg_res.extend(pfr)
            all_pdhg_iters.extend(piu)
            tv_acc      += tv_ch
            fid_acc     += fid_ch
            obj_acc     += obj_ch
            nonneg_acc  += nonneg_ch
            rel_err_acc += rel_ch
            n_total     += n_used
            per_image_rows.extend(rows)

        # Step 2 - per-channel Procrustes dictionary update (Section 7.8)
        dict_change = 0.0
        grad_norm   = 0.0
        for ch in channels:
            D_ch_old = D_per_channel[ch].copy()
            D_per_channel[ch] = procrustes_update(
                X[ch][idxs[:n_used]], A_per_channel[ch][idxs[:n_used]]
            )
            dict_change += float(np.linalg.norm(D_per_channel[ch] - D_ch_old, "fro"))
            M_ch = X[ch][idxs[:n_used]].T @ A_per_channel[ch][idxs[:n_used]]
            grad_norm   += float(np.linalg.norm(M_ch, "fro"))

        # Step 3 - refresh codes with the updated per-channel dictionaries
        for ch in channels:
            A_per_channel[ch][idxs[:n_used]] = (
                Y_per_channel[ch][idxs[:n_used]] @ D_per_channel[ch]
            )

        # --- history ---
        n_pairs = float(n_total)
        pfr_arr = np.array(all_pdhg_res, dtype=float)
        piu_arr = np.array(all_pdhg_iters, dtype=float)
        conv_frac  = float(np.mean(piu_arr < cfg.pdhg_iters)) if len(piu_arr) > 0 else 0.0
        mean_fres  = float(np.mean(pfr_arr)) if len(pfr_arr) > 0 else float("nan")
        max_fres   = float(np.max(pfr_arr))  if len(pfr_arr) > 0 else float("nan")
        mean_iters = float(np.mean(piu_arr)) if len(piu_arr) > 0 else float("nan")

        hist["tv_energy"].append(tv_acc / n_pairs)
        hist["fidelity"].append(fid_acc / n_pairs)
        hist["obj"].append(obj_acc / n_pairs)
        hist["nonneg_violation"].append(nonneg_acc / n_pairs)
        hist["grad_norm"].append(grad_norm)
        hist["dict_change"].append(dict_change)
        hist["rel_err_mean"].append(rel_err_acc / n_pairs)
        hist["eta"].append(float("nan"))
        hist["pdhg_mean_final_res"].append(mean_fres)
        hist["pdhg_max_final_res"].append(max_fres)
        hist["pdhgmean_iters"].append(mean_iters)
        hist["pdhg_converged_frac"].append(conv_frac)

        print(
            f"[Joint] outer {t+1:3d}/{cfg.outer_iters}  "
            f"lambda_TV={lam_tv:.3e}  fidelity={fid_acc/n_pairs:.4e}  "
            f"tv={tv_acc/n_pairs:.4e}  nonneg_viol={nonneg_acc/n_pairs:.2e}  "
            f"||Grad_D||={grad_norm:.4e}  Delta_D={dict_change:.4e}  "
            f"rel_err={rel_err_acc/n_pairs:.4f}"
        )
        print(
            f" --> [inner PDHG]  mean_final_res={mean_fres:.3e}  "
            f"max_final_res={max_fres:.3e}  "
            f"mean_iters={mean_iters:.1f}/{cfg.pdhg_iters}  "
            f"converged={conv_frac*100:.1f}%"
        )

        # Step 4 - convergence check (aggregate over all channels)
        obj_cur = fid_acc / n_pairs
        obj_rel_change = abs(obj_cur - obj_prev) / (abs(obj_prev) + 1e-12)
        obj_prev = obj_cur
        dict_change_norm = (dict_change / float(C)) / max(1.0, float(np.sqrt(k)))
        if obj_rel_change < cfg.outer_tol_obj or dict_change_norm < cfg.outer_tol_dict:
            consecutive_stop += 1
        else:
            consecutive_stop = 0

        if consecutive_stop >= cfg.outer_stop_patience:
            print(
                f"[Joint] Converged at outer iter {t+1}: "
                f"rel_obj_change={obj_rel_change:.2e} < {cfg.outer_tol_obj}"
            )
            break

    # Unified cell descriptor Phi_j = (a_j^(1), ..., a_j^(C)) in R^{C*K}
    Phi = np.concatenate(
        [A_per_channel[ch] for ch in channels], axis=1
    )  # (N, C*K)

    print(
        f"\n[Joint] Unified descriptor Phi shape: {Phi.shape}  "
        f"(N={N} cells, C={C} channels, K={k} atoms)"
    )
    print_joint_quality_report(X, D_per_channel, A_per_channel, idxs, n_used, cfg, channels)

    return D_per_channel, A_per_channel, Phi, hist, per_image_rows
