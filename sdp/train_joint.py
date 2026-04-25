"""
train_joint.py — Algorithm 2: joint multi-channel dictionary learning.

Public API
----------
learn_joint_multichannel(images_per_channel, k, cfg, dataset_indices)
    -> D_per_channel, A_per_channel, Phi, hist, per_image_rows

All C channels share the structural primitive vocabulary but each have
an independent dictionary D^(c). Per-channel codes are concatenated
into the unified cell descriptor Phi (eq. 87 of the manuscript).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config          import LearnConfig
from .dictionary_init import dictionary
from .pdhg_solver     import prox_tv_nn_pdhg
from .stiefel         import _procrustes_update
from .tv_operators    import grad_forward


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _print_joint_quality_report(
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

    Each channel gets an independent dictionary D^(c) (n x K), learned from
    its own channel data via a per-channel Procrustes SVD. This eliminates
    the C:1 channel bias of a shared-dictionary approach.

    After convergence the unified cell descriptor is formed as:
        phi_j = ( a_j^(1), a_j^(2), ..., a_j^(C) ) in R^{C*K}   (eq. 87)

    Parameters
    ----------
    images_per_channel : dict channel -> (N, H, W) float64 in [0, 1]
    k                  : number of dictionary atoms (shared across channels)
    cfg                : LearnConfig
    dataset_indices    : optional list of BSCCM dataset indices (for logging)

    Returns
    -------
    D_per_channel  : dict channel -> (n, K) orthonormal dictionary
    A_per_channel  : dict channel -> (N, K) codes
    Phi            : (N, C*K) unified cell descriptors phi_j
    hist           : dict of per-outer-iteration scalar metrics
    per_image_rows : list of per-sample dicts for CSV logging
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
        f"[Joint-{C}ch] λ_TV: init={cfg.lam_tv_init:.3e}  "
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
        n_total = 0
        _pdhg_final_res: List[float] = []
        _pdhg_iters_used: List[int] = []

        # Middle loop: iterate over channels (Algorithm 2 lines 4–8)
        for ch in channels:
            X_ch = X[ch]

            for ii in range(n_used):
                j = int(idxs[ii])

                x2d = X_ch[j].reshape(H, W)
                y2d, _res_list = prox_tv_nn_pdhg(
                    x_datum=x2d, lam_tv=lam_tv,
                    n_iters=cfg.pdhg_iters, tau_tv=cfg.tau_tv, sigma_tv=cfg.sigma_tv,
                    theta=1.0, tol=cfg.pdhg_tol, return_residuals=True,
                )
                _pdhg_final_res.append(_res_list[-1] if _res_list else 0.0)
                _pdhg_iters_used.append(len(_res_list))
                y = y2d.ravel()

                D_ch = D_per_channel[ch]
                a_new = D_ch.T @ y
                A_per_channel[ch][j] = a_new
                Y_per_channel[ch][j] = y

                r = X_ch[j] - D_ch @ a_new
                fidelity = 0.5 * float(np.dot(r, r))
                gv = grad_forward(y2d)
                tv_val = float(np.sum(np.sqrt(gv[..., 0] ** 2 + gv[..., 1] ** 2)))
                obj = fidelity + lam_tv * tv_val
                nonneg_viol = float(np.mean(y < -1e-10))
                x_norm = float(np.linalg.norm(X_ch[j]))
                rel_err = 100.0 * float(np.linalg.norm(r) / max(x_norm, 1e-9))

                tv_acc += lam_tv * tv_val
                fid_acc += fidelity
                obj_acc += obj
                nonneg_acc += nonneg_viol
                rel_err_acc += rel_err
                n_total += 1

                per_image_rows.append({
                    "outer_iter": float(t + 1),
                    "channel": ch,
                    "image_pos": float(ii),
                    "dataset_index": float(dataset_indices[j]),
                    "fidelity": fidelity,
                    "tv_energy": lam_tv * tv_val,
                    "obj": obj,
                    "nonneg_violation": nonneg_viol,
                    "rel_err": rel_err,
                    "lam_tv": lam_tv,
                })

        # Per-channel Procrustes dictionary update (Section 7.8)
        dict_change = 0.0
        grad_norm = 0.0
        for ch in channels:
            D_ch_old = D_per_channel[ch].copy()
            D_per_channel[ch] = _procrustes_update(
                X[ch][idxs[:n_used]], A_per_channel[ch][idxs[:n_used]]
            )
            dict_change += float(np.linalg.norm(D_per_channel[ch] - D_ch_old, "fro"))
            M_ch = X[ch][idxs[:n_used]].T @ A_per_channel[ch][idxs[:n_used]]
            grad_norm += float(np.linalg.norm(M_ch, "fro"))

        # Refresh codes with the new per-channel dictionaries
        for ch in channels:
            A_per_channel[ch][idxs[:n_used]] = (
                Y_per_channel[ch][idxs[:n_used]] @ D_per_channel[ch]
            )

        n_pairs = float(n_total)
        _pfr = np.array(_pdhg_final_res, dtype=float)
        _piu = np.array(_pdhg_iters_used, dtype=float)
        _conv_frac  = float(np.mean(_piu < cfg.pdhg_iters)) if len(_piu) > 0 else 0.0
        _mean_fres  = float(np.mean(_pfr)) if len(_pfr) > 0 else float("nan")
        _max_fres   = float(np.max(_pfr))  if len(_pfr) > 0 else float("nan")
        _mean_iters = float(np.mean(_piu)) if len(_piu) > 0 else float("nan")

        hist["tv_energy"].append(tv_acc / n_pairs)
        hist["fidelity"].append(fid_acc / n_pairs)
        hist["obj"].append(obj_acc / n_pairs)
        hist["nonneg_violation"].append(nonneg_acc / n_pairs)
        hist["grad_norm"].append(grad_norm)
        hist["dict_change"].append(dict_change)
        hist["rel_err_mean"].append(rel_err_acc / n_pairs)
        hist["eta"].append(float("nan"))
        hist["pdhg_mean_final_res"].append(_mean_fres)
        hist["pdhg_max_final_res"].append(_max_fres)
        hist["pdhg_mean_iters"].append(_mean_iters)
        hist["pdhg_converged_frac"].append(_conv_frac)

        print(
            f"[Joint] outer {t+1:3d}/{cfg.outer_iters}  "
            f"λ_TV={lam_tv:.3e}  fidelity={fid_acc/n_pairs:.4e}  "
            f"tv={tv_acc/n_pairs:.4e}  nonneg_viol={nonneg_acc/n_pairs:.2e}  "
            f"||∇D||={grad_norm:.4e}  ΔD={dict_change:.4e}  "
            f"rel_err={rel_err_acc/n_pairs:.4f}"
        )
        print(
            f"  └─ [inner PDHG]  mean_final_res={_mean_fres:.3e}  "
            f"max_final_res={_max_fres:.3e}  "
            f"mean_iters={_mean_iters:.1f}/{cfg.pdhg_iters}  "
            f"converged={_conv_frac*100:.1f}%"
        )

        # Early stopping
        obj_cur = fid_acc / n_pairs
        obj_rel_change = abs(obj_cur - obj_prev) / (abs(obj_prev) + 1e-12)
        obj_prev = obj_cur
        C_ch = float(len(channels))
        dict_change_norm = (dict_change / C_ch) / max(1.0, float(np.sqrt(k)))
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

    # Unified cell descriptor phi_j = (a_j^(1), ..., a_j^(C)) in R^{C*K}
    Phi = np.concatenate(
        [A_per_channel[ch] for ch in channels], axis=1
    )  # (N, C*K)

    print(
        f"\n[Joint] Unified descriptor Phi shape: {Phi.shape}  "
        f"(N={N} cells, C={C} channels, K={k} atoms)"
    )
    _print_joint_quality_report(X, D_per_channel, A_per_channel, idxs, n_used, cfg, channels)

    return D_per_channel, A_per_channel, Phi, hist, per_image_rows
