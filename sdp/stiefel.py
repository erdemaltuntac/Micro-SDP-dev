"""
stiefel.py — Stiefel manifold projection and Procrustes dictionary update.

The Stiefel manifold St(n, K) = {D in R^{n x K} : D^T D = I_K}.

_project_to_stiefel : projects an arbitrary matrix onto St(n, K) via QR.
procrustes_update   : exact global optimum of the dictionary subproblem
                      via economy SVD (Section 7.8 of the manuscript).
"""
from __future__ import annotations

import numpy as np


def project_to_stiefel(A: np.ndarray) -> np.ndarray:
    """
    Project A (n x K) onto {D : D^T D = I_K} via QR decomposition.

    Parameters
    ----------
    A : (n, K) float array

    Returns
    -------
    Q : (n, K) float array with Q^T Q = I_K
    """
    Q, R = np.linalg.qr(A)
    diag = np.sign(np.diag(R))
    diag[diag == 0] = 1.0
    return (Q * diag)[:, : A.shape[1]]


# Keep underscore alias for internal callers that use the original name.
_project_to_stiefel = project_to_stiefel


def procrustes_update(X_batch: np.ndarray, A_batch: np.ndarray) -> np.ndarray:
    """
    Exact Procrustes dictionary update (Section 7.8, global optimum for fixed codes).

    For fixed codes {a_j} the dictionary subproblem is:

        min_{D: D^T D = I_K}  (1/2) sum_j ||x_j - D a_j||^2

    Expanding and dropping constants this is equivalent to:

        max_{D^T D = I_K}  tr(D^T M),   M = X_batch^T @ A_batch

    The unique global maximizer is given by the economy SVD of M:

        M = U S V^T  =>  D* = U V^T

    This replaces gradient descent + Armijo backtracking with a single SVD,
    finding the globally optimal dictionary for the current codes in O(n K^2)
    flops. Compatible with Section 7.8: the paper's gradient step is a
    linearised approximation of this same first-order optimality condition.

    Parameters
    ----------
    X_batch : (n_samples, n) float  — vectorized cell images (or stacked channels)
    A_batch : (n_samples, K) float  — current codes

    Returns
    -------
    D_new : (n, K) float with D_new^T D_new = I_K
    """
    M = X_batch.T @ A_batch                        # (n, K)
    U, _s, Vt = np.linalg.svd(M, full_matrices=False)   # U:(n,K), Vt:(K,K)
    return U @ Vt                                  # (n, K), orthonormal columns


# Underscore alias kept for internal callers.
_procrustes_update = procrustes_update
