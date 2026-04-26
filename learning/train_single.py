"""
train_single.py - Algorithm 1: single-channel dictionary learning.

The core inner pass (_channel_inner_pass) is also imported by train_joint.py
so that Algorithm 2 reuses the same per-channel PDHG step without duplication.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config          import LearnConfig
from .dictionary_init import dictionary
from .pdhg_solver     import prox_tv_nn_pdhg
from .stiefel         import procrustes_update
from .tv_operators    import grad_forward

# Optional focus helper (not required for core algorithm)
try:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from focus_l1 import get_focus_analysis_for_image
    _HAS_FOCUS = True
except Exception:
    get_focus_analysis_for_image = None
    _HAS_FOCUS = False


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def print_quality_report(
    X: np.ndarray,
    D: np.ndarray,
    A: np.ndarray,
    idxs: np.ndarray,
    n_used: int,
    cfg: LearnConfig,
    channel: str,
) -> None:
    """Print final per-channel reconstruction quality after training."""
    rel_errs = []
    for ii in range(n_used):
        j = int(idxs[ii])
        x_norm = float(np.linalg.norm(X[j]))
        if x_norm >= cfg.min_x_norm:
            recon = D @ A[j]
            rel_errs.append(float(np.linalg.norm(X[j] - recon) / x_norm))
    if rel_errs:
        arr = np.array(rel_errs)
        fidelity = 100.0 * (1.0 - arr)
        print(
            f"\n[{channel}] Final quality — "
            f"mean={np.mean(arr):.4f}  median={np.median(arr):.4f}  "
            f"min={np.min(arr):.4f}  max={np.max(arr):.4f}  "
            f"Fidelity(mean)={np.mean(fidelity):.2f}%\n"
        )


# ---------------------------------------------------------------------------
# Inner PDHG pass  (shared with Algorithm 2 via train_joint.py)
# ---------------------------------------------------------------------------

def channel_inner_pass(
    X_ch: np.ndarray,
    D_ch: np.ndarray,
    A_ch: np.ndarray,
    Y_ch: np.ndarray,
    idxs: np.ndarray,
    n_used: int,
    lam_tv: float,
    outer_iter: int,
    cfg: LearnConfig,
    channel: str,
    dataset_indices: List[int],
    H: int,
    W: int,
    images_orig: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[int], float, float, float, float, float, List[Dict]]:
    """
    Inner PDHG pass for one channel over n_used images (one outer iteration).

    Updates A_ch and Y_ch in place.  Returns
        (pdhg_final_res, pdhg_iters_used,
         tv_acc, fid_acc, obj_acc, nonneg_acc, rel_err_acc,
         per_image_rows)

    Parameters
    ----------
    X_ch        : (N, n) flattened images for this channel.
    D_ch        : (n, k) current dictionary.
    A_ch        : (N, k) sparse codes — updated in place.
    Y_ch        : (N, n) TV-NN projected images — updated in place.
    idxs        : shuffled sample indices.
    n_used      : number of samples to process this iteration.
    lam_tv      : current TV regularisation weight.
    outer_iter  : 1-based outer iteration number (used for logging rows only).
    cfg         : LearnConfig.
    channel     : channel name string (for logging rows).
    dataset_indices : mapping from local index to dataset index.
    H, W        : spatial dimensions.
    images_orig : original (N, H, W) images; passed only when focus analysis
                  is wanted (Algorithm 1).  Pass None to skip.
    """
    tv_acc = fid_acc = obj_acc = nonneg_acc = rel_err_acc = 0.0
    pdhg_final_res: List[float] = []
    pdhg_iters_used: List[int] = []
    per_image_rows: List[Dict] = []

    for ii in range(n_used):
        j = int(idxs[ii])
        x2d = X_ch[j].reshape(H, W)

        y2d, res_list = prox_tv_nn_pdhg(
            x_datum=x2d, lam_tv=lam_tv,
            n_iters=cfg.pdhg_iters, tau_tv=cfg.tau_tv, sigma_tv=cfg.sigma_tv,
            theta=1.0, tol=cfg.pdhg_tol, return_residuals=True,
        )
        pdhg_final_res.append(res_list[-1] if res_list else 0.0)
        pdhg_iters_used.append(len(res_list))

        y = y2d.ravel()
        a_new = D_ch.T @ y
        A_ch[j] = a_new
        Y_ch[j] = y

        r = X_ch[j] - D_ch @ a_new
        fidelity = 0.5 * float(np.dot(r, r))
        gv = grad_forward(y2d)
        tv_val = float(np.sum(np.sqrt(gv[..., 0] ** 2 + gv[..., 1] ** 2)))
        # fidelity + lam_tv * TV(y) + indicator_{y >= 0}
        obj = fidelity + lam_tv * tv_val
        nonneg_viol = float(np.mean(y < -1e-10))
        x_norm = float(np.linalg.norm(X_ch[j]))
        rel_err = 100.0 * float(np.linalg.norm(r) / max(x_norm, 1e-9))

        tv_acc      += lam_tv * tv_val
        fid_acc     += fidelity
        obj_acc     += obj
        nonneg_acc  += nonneg_viol
        rel_err_acc += rel_err

        row: Dict = {
            "outer_iter":       float(outer_iter),
            "channel":          channel,
            "image_pos":        float(ii),
            "dataset_index":    float(dataset_indices[j]),
            "fidelity":         fidelity,
            "tv_energy":        lam_tv * tv_val,
            "obj":              obj,
            "nonneg_violation": nonneg_viol,
            "rel_err":          rel_err,
            "lam_tv":           lam_tv,
            "tau_tv":           cfg.tau_tv,
            "sigma_tv":         cfg.sigma_tv,
        }
        if images_orig is not None and _HAS_FOCUS:
            fx = get_focus_analysis_for_image(
                images_orig[j], int(dataset_indices[j]),
                channel=channel, method="gradient",
            )
            for key in ("l1_total", "metric_value", "area_reduction_ratio"):
                if key in fx:
                    row[f"focus_{key}"] = float(fx[key])
        per_image_rows.append(row)

    return (
        pdhg_final_res, pdhg_iters_used,
        tv_acc, fid_acc, obj_acc, nonneg_acc, rel_err_acc,
        per_image_rows,
    )


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

def learn_dictionary_from_images(
    images: np.ndarray,
    k: int,
    cfg: LearnConfig,
    dataset_indices: Optional[List[int]] = None,
    channel: str = "unknown",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]], List[Dict]]:
    """
    Algorithm 1: alternating proximal-gradient dictionary learning
    for a single imaging channel.

    Outer loop
    ----------
    For each iteration t:
      1. _channel_inner_pass  — PDHG solve per image, update codes A and
         projected images Y.
      2. Procrustes dictionary update  — exact SVD projection onto Stiefel
         manifold (Section 7.8 of the paper).
      3. Refresh codes with the updated dictionary.
      4. Check convergence; stop early if patience is exceeded.
    """
    if images.ndim != 3:
        raise ValueError("images must have shape (N, H, W)")
    N, H, W = images.shape
    n = H * W

    if dataset_indices is None:
        dataset_indices = list(range(N))

    assert cfg.tau_tv * cfg.sigma_tv * 8.0 < 1.0, (
        f"tau_tv*sigma_tv*8 = {cfg.tau_tv*cfg.sigma_tv*8:.4f} >= 1. "
        "Use tau_tv = sigma_tv = 1/4."
    )

    lam_tv_floor = float(cfg.mu_tv * cfg.delta)
    print(
        f"[{channel}] lambda_TV schedule: init={cfg.lam_tv_init:.3e}  "
        f"decay={cfg.lam_tv_decay}  floor={lam_tv_floor:.3e}"
    )

    X = images.reshape(N, n).astype(np.float64)
    D = dictionary(n=n, k=k, kind=cfg.dict_kind, seed=cfg.seed)
    A = np.zeros((N, k), dtype=np.float64)
    Y = np.zeros((N, n), dtype=np.float64)

    hist: Dict[str, List[float]] = {
        "tv_energy": [], "fidelity": [], "obj": [],
        "nonneg_violation": [], "grad_norm": [], "dict_change": [],
        "rel_err_mean": [], "eta": [], "lam_tv": [],
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

        # Step 1 — inner PDHG pass
        (pdhg_final_res, pdhg_iters_used,
         tv_acc, fid_acc, obj_acc, nonneg_acc, rel_err_acc,
         rows) = channel_inner_pass(
            X, D, A, Y,
            idxs, n_used, lam_tv, t + 1,
            cfg, channel, dataset_indices, H, W,
            images_orig=images,
        )
        per_image_rows.extend(rows)

        # Step 2 — Procrustes dictionary update (Section 7.8)
        D_old = D.copy()
        D = procrustes_update(X[idxs[:n_used]], A[idxs[:n_used]])
        dict_change = float(np.linalg.norm(D - D_old, "fro"))
        grad_norm   = float(np.linalg.norm(X[idxs[:n_used]].T @ A[idxs[:n_used]], "fro"))

        # Step 3 — refresh codes with the updated dictionary
        A[idxs[:n_used]] = Y[idxs[:n_used]] @ D

        # --- history ---
        pfr = np.array(pdhg_final_res, dtype=float)
        piu = np.array(pdhg_iters_used, dtype=float)
        conv_frac  = float(np.mean(piu < cfg.pdhg_iters)) if len(piu) > 0 else 0.0
        mean_fres  = float(np.mean(pfr)) if len(pfr) > 0 else float("nan")
        max_fres   = float(np.max(pfr))  if len(pfr) > 0 else float("nan")
        mean_iters = float(np.mean(piu)) if len(piu) > 0 else float("nan")

        hist["tv_energy"].append(tv_acc / n_used)
        hist["fidelity"].append(fid_acc / n_used)
        hist["obj"].append(obj_acc / n_used)
        hist["nonneg_violation"].append(nonneg_acc / n_used)
        hist["grad_norm"].append(grad_norm)
        hist["dict_change"].append(dict_change)
        hist["rel_err_mean"].append(rel_err_acc / n_used)
        hist["eta"].append(float("nan"))   # not applicable with Procrustes
        hist["lam_tv"].append(lam_tv)
        hist["pdhg_mean_final_res"].append(mean_fres)
        hist["pdhg_max_final_res"].append(max_fres)
        hist["pdhgmean_iters"].append(mean_iters)
        hist["pdhg_converged_frac"].append(conv_frac)

        print(
            f"[{channel}] outer {t+1:3d}/{cfg.outer_iters}  "
            f"lambda_TV={lam_tv:.3e}  fidelity={fid_acc/n_used:.4e}  "
            f"tv={tv_acc/n_used:.4e}  nonneg_viol={nonneg_acc/n_used:.2e}  "
            f"||Grad_D||={grad_norm:.4e}  Delta_D={dict_change:.4e}  "
            f"rel_err={rel_err_acc/n_used:.4f}"
        )
        print(
            f"  --> [inner PDHG]  mean_final_res={mean_fres:.3e}  "
            f"max_final_res={max_fres:.3e}  "
            f"mean_iters={mean_iters:.1f}/{cfg.pdhg_iters}  "
            f"converged={conv_frac*100:.1f}%"
        )

        # Step 4 — convergence check
        obj_cur = fid_acc / n_used
        obj_rel_change = abs(obj_cur - obj_prev) / (abs(obj_prev) + 1e-12)
        obj_prev = obj_cur
        dict_change_norm = dict_change / max(1.0, float(np.linalg.norm(D_old, "fro")))
        if obj_rel_change < cfg.outer_tol_obj or dict_change_norm < cfg.outer_tol_dict:
            consecutive_stop += 1
        else:
            consecutive_stop = 0

        if consecutive_stop >= cfg.outer_stop_patience:
            print(
                f"[{channel}] Converged at outer iter {t+1}: "
                f"rel_obj_change={obj_rel_change:.2e} < {cfg.outer_tol_obj}"
            )
            break

    print_quality_report(X, D, A, idxs, n_used, cfg, channel)
    return D, A, hist, per_image_rows
