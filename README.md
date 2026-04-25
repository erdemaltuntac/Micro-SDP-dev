<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/aegis_logo_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="logo/aegis_logo.svg"/>
    <img src="assets/aegis_logo.svg" alt="Aegis Digital Technologies" height="700"/>
  </picture>
</p>

# Micro-SDP

**Structured Dictionary Priors for Quantitative Microscopy**

A Python package for learning channel-specific and joint multi-channel dictionaries from microscopy images using Total Variation (TV) regularisation and non-negativity constraints. Companion code for the manuscript:

> *[Title]*, [Authors], *[Journal]*, 2026.

---

## Overview

`sdp` (Structured Dictionary Priors) implements two algorithms:

- **Algorithm 1 — Single-channel dictionary learning**: learns a dictionary from images of one imaging channel using a PDHG inner solver with TV + non-negativity proximal operators.
- **Algorithm 2 — Joint multi-channel dictionary learning**: learns a shared latent structure across multiple imaging channels (e.g. Brightfield, DPC Left/Right/Top/Bottom) simultaneously.

Validation is performed on the [BSCCM dataset](https://github.com/henrypinkard/BSCCM) — a blood-smear cell microscopy collection with ground-truth cell-type labels.

---

## Repository Structure

```
Micro-SDP-dev/
├── run_training.py          # Entry point: trains all channels + joint model
├── run_post_training_figures.py  # Re-generate figures from saved artifacts
├── requirements.txt
├── LICENSE
├── assets/
│   └── aegis_logo.svg
└── sdp/
    ├── __init__.py
    ├── config.py            # LearnConfig dataclass, channel definitions
    ├── tv_operators.py      # Forward gradient, backward divergence, L2-ball projection
    ├── stiefel.py           # Stiefel manifold projection, Procrustes update
    ├── dictionary_init.py   # DCT and random orthonormal initialisation
    ├── pdhg_solver.py       # PDHG inner loop (TV + non-negativity)
    ├── train_single.py      # Algorithm 1: single-channel training
    ├── train_joint.py       # Algorithm 2: joint multi-channel training
    ├── data_loader.py       # BSCCM data loading pipeline
    ├── artifacts.py         # Save / load training artifacts (.npy)
    ├── evaluate.py          # PSNR, reconstruction evaluation, biological validation
    └── plots.py             # Convergence plots, unified cell figures, truth comparisons
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

### 2. Train

```bash
python run_training.py
```

This will:
- Train a dictionary for each imaging channel (Brightfield, DPC Left/Right/Top/Bottom)
- Train the joint multi-channel model
- Save artifacts to `training_artifacts_<channel>_<timestamp>_seed<n>/`
- Generate convergence plots, reconstructions, and unified-vs-truth figures

### 3. Regenerate figures from saved artifacts

```bash
python run_post_training_figures.py
```

---

## Package API

```python
from sdp.config import LearnConfig
from sdp.train_single import learn_dictionary_from_images
from sdp.train_joint import learn_joint_multichannel

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
@article{microsdp2026,
  title   = {[Title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026}
}
```

---

## License

Copyright 2026 Aegis Digital Technologies  
Licensed under the [Apache License 2.0](LICENSE).
