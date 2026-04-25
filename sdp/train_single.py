"""
train_single.py — Algorithm 1: single-channel dictionary learning.

Public API
----------
learn_dictionary_from_images(images, k, cfg, dataset_indices, channel)
    -> D, A, hist, per_image_rows

Helper (used internally by train_joint.py as well):
_print_quality_report(X, D, A, idxs, n_used, cfg, channel)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config          import LearnConfig
from .dictionary_init import dictionary
from .pdhg_solver     import prox_tv_nn_pdhg
from .stiefel         import _procrustes_update
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
# Internal helper
# ---------------------------------------------------------------------------

def _print_quality_report(
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

    Inference step (per sample, fixed D)
    -------------------------------------
    y_j* = argmin_y  1/2 ||y - x_j||^2  +  lambda_TV * TV(y)  +  iota_{R^n_+}(y)
    solved by prox_tv_nn_pdhg() with tau = sigma = 1/4.

    Code update
    -----------
    a_j = D^T y_j*   (exact back-projection, D^T D = I_K)

    Dictionary update (Procrustes SVD, Section 7.8)
    -------------------------------------------------
    G = (1/N) sum_j (D a_j - y_j*) a_j^T
    Exact global optimum: D* = U V^T from SVD of X^T A.

    Parameters
    ----------
    images          : (N, H, W) float array of single-cell images in [0, 1]
    k               : number of dictionary atoms
    cfg             : LearnConfig
    dataset_indices : optional list mapping batch row -> dataset index (for logging)
    channel         : channel name string (for logging/CSV)

    Returns
    -------
    D              : (n, k)   learned orthonormal dictionary
    A              : (N, k)   codes a_j = D^T y_j* for each training sample
    hist           : dict of per-outer-iteration scalar metrics
    per_image_rows : list of per-sample dicts for CSV logging
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
        f"[{channel}] λ_TV schedule: init={cfg.lam_tv_init:.3e}  "
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
        "pdhg_mean_iters": [], "pdhg_converged_frac": [],
    }
    per_image_rows: List[Dict] = []

    idxs = np.arange(N)
    consecutive_stop = 0
    obj_prev = float("inf")

    for t in range(cfg.outer_iters):
        lam_tv = max(lam_tv_floor, cfg.lam_tv_init / (1.0 + cfg.lam_tv_decay * t))

        if cfg.shuffle:
            np.random.default_rng(cfg.seed + t).shuffle(idxs)

        n_used = N if cfg.max_samples <= 0 else min(cfg.max_samples, N)

        tv_acc = fid_acc = obj_acc = nonneg_acc = rel_err_acc = 0.0
        _pdhg_final_res: List[float] = []
        _pdhg_iters_used: List[int] = []

        for ii in range(n_used):
            j = int(idxs[ii])
            x2d = X[j].reshape(H, W)

            y2d, _res_list = prox_tv_nn_pdhg(
                x_datum=x2d, lam_tv=lam_tv,
                n_iters=cfg.pdhg_iters, tau_tv=cfg.tau_tv, sigma_tv=cfg.sigma_tv,
                theta=1.0, tol=cfg.pdhg_tol, return_residuals=True,
            )
            _pdhg_final_res.append(_res_list[-1] if _res_list else 0.0)
            _pdhg_iters_used.append(len(_res_list))

            y = y2d.ravel()
            a_new = D.T @ y
            A[j] = a_new
            Y[j] = y

            r = X[j] - D @ a_new
            fidelity = 0.5 * float(np.dot(r, r))
            gv = grad_forward(y2d)
            tv_val = float(np.sum(np.sqrt(gv[..., 0] ** 2 + gv[..., 1] ** 2)))
            obj = fidelity + lam_tv * tv_val
            nonneg_viol = float(np.mean(y < -1e-10))
            x_norm = float(np.linalg.norm(X[j]))
            recon = D @ a_new
            rel_err = 100.0 * float(np.linalg.norm(X[j] - recon) / max(x_norm, 1e-9))

            tv_acc += lam_tv * tv_val
            fid_acc += fidelity
            obj_acc += obj
            nonneg_acc += nonneg_viol
            rel_err_acc += rel_err

            row: Dict = {
                "outer_iter": float(t + 1),
                "channel": channel,
                "image_pos": float(ii),
                "dataset_index": float(dataset_indices[j]),
                "fidelity": fidelity,
                "tv_energy": lam_tv * tv_val,
                "obj": obj,
                "nonneg_violation": nonneg_viol,
                "rel_err": rel_err,
                "lam_tv": lam_tv,
                "tau_tv": cfg.tau_tv,
                "sigma_tv": cfg.sigma_tv,
            }
            if _HAS_FOCUS:
                fx = get_focus_analysis_for_image(
                    images[j], int(dataset_indices[j]),
                    channel=channel, method="gradient"
                )
                for key in ("l1_total", "metric_value", "area_reduction_ratio"):
                    if key in fx:
                        row[f"focus_{key}"] = float(fx[key])
            per_image_rows.append(row)

        # --- Dictionary update: exact Procrustes SVD (Section 7.8) ---
        D_old = D.copy()
        X_used = X[idxs[:n_used]]
        A_used = A[idxs[:n_used]]
        D = _procrustes_update(X_used, A_used)
        dict_change = float(np.linalg.norm(D - D_old, "fro"))
        grad_norm   = float(np.linalg.norm(X_used.T @ A_used, "fro"))

        # Refresh codes with the new dictionary
        A[idxs[:n_used]] = Y[idxs[:n_used]] @ D

        _pfr = np.array(_pdhg_final_res, dtype=float)
        _piu = np.array(_pdhg_iters_used, dtype=float)
        _conv_frac  = float(np.mean(_piu < cfg.pdhg_iters)) if len(_piu) > 0 else 0.0
        _mean_fres  = float(np.mean(_pfr)) if len(_pfr) > 0 else float("nan")
        _max_fres   = float(np.max(_pfr))  if len(_pfr) > 0 else float("nan")
        _mean_iters = float(np.mean(_piu)) if len(_piu) > 0 else float("nan")

        hist["tv_energy"].append(tv_acc / n_used)
        hist["fidelity"].append(fid_acc / n_used)
        hist["obj"].append(obj_acc / n_used)
        hist["nonneg_violation"].append(nonneg_acc / n_used)
        hist["grad_norm"].append(grad_norm)
        hist["dict_change"].append(dict_change)
        hist["rel_err_mean"].append(rel_err_acc / n_used)
        hist["eta"].append(float("nan"))   # not applicable with Procrustes
        hist["lam_tv"].append(lam_tv)
        hist["pdhg_mean_final_res"].append(_mean_fres)
        hist["pdhg_max_final_res"].append(_max_fres)
        hist["pdhg_mean_iters"].append(_mean_iters)
        hist["pdhg_converged_frac"].append(_conv_frac)

        print(
            f"[{channel}] outer {t+1:3d}/{cfg.outer_iters}  "
            f"λ_TV={lam_tv:.3e}  fidelity={fid_acc/n_used:.4e}  "
            f"tv={tv_acc/n_used:.4e}  nonneg_viol={nonneg_acc/n_used:.2e}  "
            f"||∇D||={grad_norm:.4e}  ΔD={dict_change:.4e}  "
            f"rel_err={rel_err_acc/n_used:.4f}"
        )
        print(
            f"  └─ [inner PDHG]  mean_final_res={_mean_fres:.3e}  "
            f"max_final_res={_max_fres:.3e}  "
            f"mean_iters={_mean_iters:.1f}/{cfg.pdhg_iters}  "
            f"converged={_conv_frac*100:.1f}%"
        )

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

    _print_quality_report(X, D, A, idxs, n_used, cfg, channel)
    return D, A, hist, per_image_rows
