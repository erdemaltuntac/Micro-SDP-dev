"""
config.py — Configuration dataclass and channel list.

LearnConfig holds every hyper-parameter that controls training:
TV regularisation schedule, PDHG step sizes, outer-loop stopping
criteria, and evaluation thresholds.

BSCCM_CHANNELS lists the five LED-array channels used in Algorithm 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Five BSCCM LED-array channels used for joint multi-channel unification
# ---------------------------------------------------------------------------
BSCCM_CHANNELS: List[str] = [
    "DPC_Left", "DPC_Right", "DPC_Top", "DPC_Bottom", "Brightfield"
]


@dataclass
class LearnConfig:
    """
    Hyper-parameters for Algorithm 1 (single-channel) and
    Algorithm 2 (joint multi-channel) dictionary learning.

    TV schedule
    -----------
    lambda_TV(t) = max(mu_tv * delta,  lam_tv_init / (1 + lam_tv_decay * t))

    At t=0     : lambda_TV = lam_tv_init  (strong smoothing, poor initial D)
    As t -> inf: lambda_TV -> mu_tv * delta  (Theorem 9 noise-floor guarantee)

    PDHG step sizes
    ---------------
    tau_tv = sigma_tv = 1/4 satisfies the condition
        tau * sigma * ||grad||^2 <= tau * sigma * 8 = 1/2 < 1  (Theorem 4).
    """

    # --- TV regularisation floor (Theorem 9) ----------------------------
    delta: float = 1e-3        # estimated noise level
    mu_tv: float = 1.0         # floor: lam_tv_floor = mu_tv * delta

    # --- Dynamic lambda_TV schedule -------------------------------------
    lam_tv_init: float  = 0.1  # initial TV weight (strong regularisation)
    lam_tv_decay: float = 1.0  # decay rate gamma

    # --- Inner PDHG (TV solver) -----------------------------------------
    tau_tv:    float = 0.25    # primal step size  (tau*sigma*8 = 1/2 < 1)
    sigma_tv:  float = 0.25    # dual step size
    pdhg_iters: int  = 1000    # max inner iterations
    pdhg_tol:  float = 1e-7    # inner stopping tolerance (eqs. 74-75)

    # --- Outer loop -----------------------------------------------------
    outer_iters:        int   = 15
    outer_tol_dict:     float = 1e-6   # ||D_new - D_old||_F / max(1,||D||_F)
    outer_tol_obj:      float = 5e-5   # relative fidelity-change threshold
    outer_stop_patience: int  = 5      # consecutive iters below threshold

    # --- Dataset / batch ------------------------------------------------
    # max_samples=0 means USE ALL N images every outer iteration.
    max_samples: int  = 0
    shuffle:     bool = True
    dict_kind:   str  = "dct"
    seed:        int  = 0

    # --- Evaluation (success-rate metric, eq. 81) -----------------------
    eps_relerr:      float = 0.05
    target_fraction: float = 0.95
    min_x_norm:      float = 1e-3
    ignore_low_norm: bool  = True
    n_runs:          int   = 1
    base_seed:       int   = 0
