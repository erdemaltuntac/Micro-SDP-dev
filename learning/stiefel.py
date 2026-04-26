"""
stiefel.py - Stiefel manifold projection and Procrustes dictionary update.

The Stiefel manifold St(n, K) = {D in R^{n x K} : D^T D = I_K}.
"""
from __future__ import annotations

import numpy as np


def project_to_stiefel(A: np.ndarray) -> np.ndarray:
    """
    Project A (n x K) onto {D : D^T D = I_K} via QR decomposition.
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

    The unique global maximizer is given by the economy SVD of M:

        M = U S V^T  =>  D* = U V^T
    """
    M = X_batch.T @ A_batch                        # (n, K)
    U, _s, Vt = np.linalg.svd(M, full_matrices=False)   # U:(n,K), Vt:(K,K)
    return U @ Vt                                  # (n, K), orthonormal columns
