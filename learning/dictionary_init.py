"""
dictionary_init.py — Dictionary initialisation on the Stiefel manifold.

Provides two strategies:
  "dct"  — deterministic DCT-II orthonormal basis (first K columns)
  "rand" — random orthonormal matrix from a QR factorization (seeded)

Both return D in R^{n x K} with D^T D = I_K.
"""
from __future__ import annotations

import math
import numpy as np

from .stiefel import project_to_stiefel


def dictionary(n: int, k: int, kind: str = "dct", seed: int = 0) -> np.ndarray:
    """
    Return D in R^{n x k} with orthonormal columns (D^T D = I_k).

    Parameters
    ----------
    n    : signal dimension (flattened image size, e.g. H*W)
    k    : number of atoms  (k <= n)
    kind : "dct"  — deterministic DCT-II orthonormal basis
           "rand" — random orthonormal dictionary (QR) with fixed seed
    seed : random seed; used only for kind="rand"

    Returns
    -------
    D : (n, k) float64 array, D^T D = I_k

    Raises
    ------
    ValueError : if k > n or kind is unrecognised
    """
    if k > n:
        raise ValueError(f"k must be <= n, got k={k}, n={n}")
    kind = kind.lower().strip()
    if kind == "dct":
        i = np.arange(n, dtype=np.float64)[:, None]
        j = np.arange(n, dtype=np.float64)[None, :]
        D_full = np.cos(np.pi * (i + 0.5) * j / n)
        D_full[:, 0] *= math.sqrt(1.0 / n)
        if n > 1:
            D_full[:, 1:] *= math.sqrt(2.0 / n)
        return project_to_stiefel(D_full[:, :k].astype(np.float64, copy=False))
    if kind == "rand":
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, k))
        return project_to_stiefel(A)
    raise ValueError(f"Unknown kind={kind!r}. Use 'dct' or 'rand'.")
