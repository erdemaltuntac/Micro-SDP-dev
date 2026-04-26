<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/aegis_logo_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="logo/aegis_logo.svg"/>
    <img src="assets/aegis_logo.svg" alt="Aegis Digital Technologies" height="700"/>
  </picture>
</p>

# Micro-SDP

**Smart Detection and Prediction of Single Cells within Heterogeneous Microscopic Images**  
*Dictionary Learning Module*

This repository contains the dictionary learning component of the Micro-SDP system. It provides channel-specific and joint multi-channel dictionary learning from microscopy images using Total Variation (TV) regularisation and non-negativity constraints. Companion code for the manuscript:

> *Learned Dictionaries with Total Variation and Non-Negativity for Single-Cell Microscopy: Convergence Theory and Deterministic Multi-Channel Cell Feature Unification*, Aegis Digital Technologies, 2026.  
> **arXiv:** [2604.05211](https://arxiv.org/abs/2604.05211)  [math.NA]

---

## Overview

The `learning` package implements two algorithms:

- **Algorithm 1; Single-channel dictionary learning**: learns a dictionary from images of one imaging channel using a PDHG inner solver with TV + non-negativity proximal operators.
- **Algorithm 2; Joint multi-channel dictionary learning**: learns a shared dictionary across multiple imaging channels (e.g. Brightfield, DPC Left/Right/Top/Bottom), with channel-specific sparse codes.

Validation is performed on the [BSCCM dataset](https://github.com/henrypinkard/BSCCM) - a single-cell microscopy collection with ground-truth cell-type labels.

---

## Repository Structure

The top-level scripts handle training and figure generation; the `learning/` package contains all algorithmic components.

```
Micro-SDP-dev/

  run_training.py               # main entry point - train, evaluate, save
  run_post_training_figures.py  # re-run figures without retraining

  regen_cell_figure.py          # regenerate multi-cell grid from saved results
  regen_unified_cell.py         # regenerate single-cell portrait
  regen_convergence_plot.py     # regenerate convergence figure

  requirements.txt
  LICENSE
  assets/

  learning/                     # core package
      config.py                 # LearnConfig, channel list
      dictionary_init.py        # DCT / random-QR initialisation
      tv_operators.py           # finite-difference gradient and its adjoint
      stiefel.py                # Procrustes / Stiefel projection
      pdhg_solver.py            # inner TV + non-negativity solver (PDHG)
      train_single.py           # Algorithm 1: single-channel
      train_joint.py            # Algorithm 2: joint multi-channel
      data_loader.py            # BSCCM image loading and focus cropping
      results.py                # save dictionaries, codes, history to disk
      evaluate.py               # PSNR, reconstruction stats, bio validation
      plots.py                  # all figure-generating functions
      bootstrap_validation.py   # bootstrap CI and permutation null test
```

---

## Installation

```bash
git clone https://github.com/erdemaltuntac/Micro-SDP-dev.git
cd Micro-SDP-dev
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, numpy, matplotlib, scikit-learn, Pillow.

---

## Quick Start

### 1. Prepare data

Download the BSCCM-tiny dataset and place it at `dev/BSCCM-tiny/` (or update the path in `run_training.py`).
For the structure and use of datasource, we encourage users to refer to https://waller-lab.github.io/BSCCM/ 

### 2. Train

```bash
python run_training.py
```

This will:
- Train a dictionary for each imaging channel (Brightfield, DPC Left/Right/Top/Bottom)
- Train the joint multi-channel model
- Save results to `training_results_<channel>_<timestamp>_seed<n>/`
- Generate convergence plots, reconstructions, and unified-vs-truth figures

### 3. Regenerate figures from saved results

```bash
python run_post_training_figures.py
```

---

## Package API

```python
from learning.config import LearnConfig
from learning.train_single import learn_dictionary_from_images
from learning.train_joint import learn_joint_multichannel

cfg = LearnConfig(n_atoms=64, patch_size=8, n_iter=500)

# Single-channel
D, codes, history = learn_dictionary_from_images(images, config=cfg)

# Joint multi-channel
results = learn_joint_multichannel(images_per_channel, config=cfg)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{microsdp2026,
  title         = {Learned Dictionaries with Total Variation and Non-Negativity for Single-Cell Microscopy: Convergence Theory and Deterministic Multi-Channel Cell Feature Unification},
  author        = {Aegis Digital Technologies},
  year          = {2026},
  url           = {https://arxiv.org/abs/2604.05211}
}
```

---

## License

Copyright 2026 Aegis Digital Technologies  
Licensed under the [Apache License 2.0](LICENSE).
