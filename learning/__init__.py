"""
learning — Single-cell Dictionary Prior
====================================
Modular implementation of Algorithm 1 (single-channel) and
Algorithm 2 (joint multi-channel) dictionary learning with
TV + non-negativity regularization.

Public API
----------
from learning.config          import LearnConfig, BSCCM_CHANNELS
from learning.dictionary_init import dictionary
from learning.tv_operators    import grad_forward, div_backward, project_l2_ball
from learning.stiefel         import project_to_stiefel, procrustes_update
from learning.pdhg_solver     import prox_tv_nn_pdhg
from learning.train_single    import learn_dictionary_from_images
from learning.train_joint     import learn_joint_multichannel
from learning.data_loader     import load_images_bsccm_pipeline, load_all_channels
from learning.results       import save_training_results, save_joint_results
from learning.evaluate        import psnr, evaluate_and_save_reconstructions, run_biological_validation
from learning.plots           import (save_convergence_plots, save_single_unified_cell,
                                 save_unified_cell_figure, save_reconstructed_images,
                                 save_unified_vs_truth)
"""
