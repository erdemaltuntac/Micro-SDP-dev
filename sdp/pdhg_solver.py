"""
pdhg_solver.py — Primal-Dual Hybrid Gradient (PDHG) solver for the
                 TV + non-negativity proximal problem.

Solves Algorithm 1 (inner loop) from the manuscript:

    y* = argmin_y  1/2 ||y - x||^2  +  lambda_TV * TV(y)  +  iota_{R^n_+}(y)

Step-size condition (Lemma 3, Theorem 4):
    tau * sigma * ||grad||^2 <= tau * sigma * 8 < 1
    Default: tau = sigma = 1/4  =>  (1/16)*8 = 1/2 < 1  ✓

Dual proximal map   : pixelwise Euclidean-ball projection of radius lambda_TV
Primal proximal map : prox_{tau * F}(w) = max( (w + tau*x) / (1+tau),  0 )
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from .tv_operators import div_backward, grad_forward, project_l2_ball


def prox_tv_nn_pdhg(
    x_datum: np.ndarray,
    lam_tv: float,
    n_iters: int = 100,
    tau_tv: float = 0.25,
    sigma_tv: float = 0.25,
    theta: float = 1.0,
    tol: float = 1e-6,
    return_residuals: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[float]]]:
    """
    Solve the per-sample TV + non-negativity proximal problem for fixed D.

    Problem (per sample, Algorithm 1 inner loop)
    --------------------------------------------
    y* = argmin_y  1/2 ||y - x_datum||^2  +  lambda_TV * TV(y)  +  iota_{R^n_+}(y)

    PDHG iteration (Theorem 10):
      Dual   : q^{n+1} = Pi_{B_{lam_tv}}( q^n + sigma_tv * grad(y_bar^n) )
      Primal : y^{n+1} = max( (y^n - tau_tv * div(q^{n+1}) + tau_tv * x) / (1+tau_tv), 0 )
      Over-relax: y_bar^{n+1} = y^{n+1} + theta * (y^{n+1} - y^n)

    Parameters
    ----------
    x_datum  : (H, W) float — observed cell image for this sample
    lam_tv   : float >= 0   — TV regularization weight
    n_iters  : int          — maximum inner PDHG iterations
    tau_tv   : float        — primal step size  (paper: 1/4)
    sigma_tv : float        — dual step size    (paper: 1/4)
    theta    : float [0,1]  — over-relaxation parameter (1 = Chambolle-Pock)
    tol      : float        — stopping tolerance on ||y^{n+1} - y^n||_2
    return_residuals : bool — if True, also return list of per-iteration residuals

    Returns
    -------
    y : (H, W) float — minimizer y*
    (y, residuals) if return_residuals=True
    """
    if lam_tv <= 0.0:
        result = np.maximum(x_datum, 0.0)
        return (result, []) if return_residuals else result

    assert tau_tv * sigma_tv * 8.0 < 1.0, (
        f"Step-size condition violated: tau*sigma*8 = {tau_tv*sigma_tv*8:.4f} >= 1. "
        f"Use tau_tv = sigma_tv = 1/4."
    )

    y     = x_datum.copy()
    y_bar = y.copy()
    q     = np.zeros((*y.shape, 2), dtype=np.float64)

    _pdhg_residuals: List[float] = []
    for _ in range(n_iters):
        # Dual update (Theorem 2 Part 3, Algorithm 1 line 10)
        q = project_l2_ball(q + sigma_tv * grad_forward(y_bar), radius=lam_tv)

        # Primal update (Theorem 10, Algorithm 1 line 11)
        y_prev = y
        w = y - tau_tv * div_backward(q)
        y = np.maximum((w + tau_tv * x_datum) / (1.0 + tau_tv), 0.0)

        # Over-relaxation: y_bar = y + theta * (y - y_prev)
        y_bar = y + theta * (y - y_prev)

        # Inner stopping criterion
        res = float(np.linalg.norm(y - y_prev))
        if return_residuals:
            _pdhg_residuals.append(res)
        if res < tol:
            break

    if return_residuals:
        return y, _pdhg_residuals
    return y
