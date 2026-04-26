"""
results.py - Saving training results to disk.
"""
from __future__ import annotations

import csv
import json
import pathlib
from typing import Dict, List

import numpy as np

from .config import LearnConfig


def save_training_results(
    out_dir: str,
    D: np.ndarray,
    A: np.ndarray,
    images: np.ndarray,
    indices: List[int],
    cfg: LearnConfig,
    per_image_rows: List[Dict],
    hist: Dict[str, List[float]],
    channel: str = "unknown",
) -> None:
    """
    Save Algorithm 1 (single-channel) results.
    """
    od = pathlib.Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    if per_image_rows:
        csv_path = od / "training_scores.csv"
        fieldnames = list(per_image_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_image_rows:
                w.writerow(r)

    (od / "train_indices.txt").write_text(
        "\n".join(str(i) for i in indices), encoding="utf-8"
    )
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["channel"] = channel
    cfg_dict["n_train_images"] = int(images.shape[0])
    (od / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    np.save(od / "dictionary_D.npy", D)
    np.save(od / "codes_A.npy", A)
    np.save(od / "train_images.npy", images)
    (od / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"Artifacts saved to {od}/")


def save_joint_results(
    out_dir: str,
    D_per_channel: Dict[str, np.ndarray],
    A_per_channel: Dict[str, np.ndarray],
    Phi: np.ndarray,
    images_per_channel: Dict[str, np.ndarray],
    indices: List[int],
    cfg: LearnConfig,
    per_image_rows: List[Dict],
    hist: Dict[str, List[float]],
) -> None:
    """
    Save Algorithm 2 (joint multi-channel) results.
    """
    od = pathlib.Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    if per_image_rows:
        csv_path = od / "joint_training_scores.csv"
        fieldnames = list(per_image_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_image_rows:
                w.writerow(r)

    (od / "train_indices.txt").write_text(
        "\n".join(str(i) for i in indices), encoding="utf-8"
    )
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["channels"] = list(A_per_channel.keys())
    cfg_dict["n_train_images"] = int(Phi.shape[0])
    (od / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    for ch, D_ch in D_per_channel.items():
        np.save(od / f"dictionary_D_{ch}.npy", D_ch)
    np.save(od / "unified_descriptors_Phi.npy", Phi)
    for ch, A_ch in A_per_channel.items():
        np.save(od / f"codes_A_{ch}.npy", A_ch)
    ref_ch = list(images_per_channel.keys())[0]
    np.save(od / "train_images_ref_channel.npy", images_per_channel[ref_ch])
    (od / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    # Epoch-by-epoch CSV (human-readable spreadsheet)
    if hist:
        keys = [k for k, v in hist.items() if isinstance(v, list) and len(v) > 0]
        n_epochs = max(len(hist[k]) for k in keys)
        hist_csv_path = od / "history_per_epoch.csv"
        with hist_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["outer_iter"] + keys)
            w.writeheader()
            for t in range(n_epochs):
                row: Dict = {"outer_iter": t + 1}
                for k in keys:
                    row[k] = hist[k][t] if t < len(hist[k]) else ""
                w.writerow(row)

    print(f"Joint results saved to {od}/")
