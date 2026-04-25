"""
tv_operators.py — Discrete TV operators (grad, div, ball projection).

These implement the finite-difference gradient and its adjoint used in the
PDHG inner loop (Algorithm 1, Theorem 4).

    ||∇||^2 ≤ 8  for the 2D forward-difference operator with Neumann BC.
    Step-size condition:  tau * sigma * 8 < 1  =>  tau = sigma = 1/4 is safe.
"""
from __future__ import annotations

import numpy as np


def grad_forward(y2d: np.ndarray) -> np.ndarray:
    """
    2D forward-difference gradient.

    Parameters
    ----------
    y2d : (H, W) float array

    Returns
    -------
    g : (H, W, 2) float array — g[..., 0] = x-direction, g[..., 1] = y-direction
        Neumann (zero-flux) boundary conditions.
    """
    H, W = y2d.shape
    g = np.zeros((H, W, 2), dtype=y2d.dtype)
    g[:, :-1, 0] = y2d[:, 1:] - y2d[:, :-1]   # x-direction
    g[:-1, :, 1] = y2d[1:, :] - y2d[:-1, :]   # y-direction
    return g


def div_backward(p: np.ndarray) -> np.ndarray:
    """
    Adjoint of grad_forward (negative divergence).

    Parameters
    ----------
    p : (H, W, 2) float array

    Returns
    -------
    div : (H, W) float array
    """
    H, W, _ = p.shape
    div = np.zeros((H, W), dtype=p.dtype)
    # x component
    div[:, 0]    += p[:, 0, 0]
    div[:, 1:-1] += p[:, 1:-1, 0] - p[:, :-2, 0]
    div[:, -1]   -= p[:, -2, 0]
    # y component
    div[0, :]    += p[0, :, 1]
    div[1:-1, :] += p[1:-1, :, 1] - p[:-2, :, 1]
    div[-1, :]   -= p[-2, :, 1]
    return div


def project_l2_ball(p: np.ndarray, radius: float) -> np.ndarray:
    """
    Pointwise Euclidean-ball projection of radius ``radius`` for each pixel
    vector p[i, j, :].

    Parameters
    ----------
    p      : (H, W, 2) float array
    radius : float >= 0

    Returns
    -------
    projected : (H, W, 2) float array with ||projected[i,j,:]||_2 <= radius
    """
    if radius <= 0.0:
        return np.zeros_like(p)
    norm = np.linalg.norm(p, axis=2, keepdims=True)
    return p / np.maximum(1.0, norm / radius)
