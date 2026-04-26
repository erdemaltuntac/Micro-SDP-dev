"""
data_loader.py - BSCCM image loading pipeline.
"""
from __future__ import annotations

import sys
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional BSCCM / focus imports (resolved relative to parent dev/ directory)
# ---------------------------------------------------------------------------
_DEV = pathlib.Path(__file__).resolve().parent.parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

try:
    from read_bsccm_data import BSCCM_Img_Reader
    _HAS_BSCCM = True
except Exception:
    BSCCM_Img_Reader = None  # type: ignore
    _HAS_BSCCM = False

try:
    from focus_l1 import get_focus_analysis_for_image
    _HAS_FOCUS = True
except Exception:
    get_focus_analysis_for_image = None  # type: ignore
    _HAS_FOCUS = False

from .config import BSCCM_CHANNELS


def load_images_bsccm_pipeline(
    location: str = ".",
    tiny: bool = True,
    channel: str = "DPC_Left",
    n_images: int = 50,
    use_focused: bool = True,
    focus_method: str = "gradient",
    output_dir: str = "bsccm_out_image",
) -> Tuple[np.ndarray, List[int]]:
    """
    Load and preprocess images from the BSCCM dataset for one channel.
    """
    if not _HAS_BSCCM:
        raise RuntimeError(
            "BSCCM_Img_Reader is not available. "
            "Ensure read_bsccm_data.py is on the Python path."
        )

    reader = BSCCM_Img_Reader(output_dir=output_dir)
    reader.load_dataset(location=location, tiny=tiny)

    if reader.bsccm is None or reader.valid_indices is None:
        raise RuntimeError("BSCCM dataset failed to load.")

    idxs = (
        list(reader.valid_indices) if n_images <= 0
        else list(reader.valid_indices)[:n_images]
    )
    imgs = []

    for idx in idxs:
        img = reader.bsccm.read_image(idx, channel=channel)
        img = np.asarray(img, dtype=np.float64)
        if img.ndim != 2:
            img = np.squeeze(img)
            if img.ndim != 2:
                raise ValueError(
                    f"Expected 2D image for channel={channel}, index={idx}"
                )

        if use_focused:
            if not _HAS_FOCUS:
                raise RuntimeError(
                    "focus_l1 not available but use_focused=True."
                )
            focus = get_focus_analysis_for_image(
                img, int(idx), str(channel), method=focus_method
            )
            img = np.asarray(focus["focused_image"], dtype=np.float64)

        img -= float(np.min(img))
        mx = float(np.max(img))
        if mx > 0:
            img /= mx
        imgs.append(img)

    H = min(im.shape[0] for im in imgs)
    W = min(im.shape[1] for im in imgs)
    imgs = [im[:H, :W] for im in imgs]

    return np.stack(imgs, axis=0), idxs


def load_all_channels(
    location: str = ".",
    tiny: bool = True,
    channels: List[str] = BSCCM_CHANNELS,
    n_images: int = 50,
    use_focused: bool = True,
    focus_method: str = "gradient",
    output_dir: str = "bsccm_out_image",
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """
    Load all specified channels from the BSCCM dataset.
    """
    if not _HAS_BSCCM:
        raise RuntimeError("BSCCM_Img_Reader not available.")

    reader = BSCCM_Img_Reader(output_dir=output_dir)
    reader.load_dataset(location=location, tiny=tiny)

    if reader.bsccm is None or reader.valid_indices is None:
        raise RuntimeError("BSCCM dataset failed to load.")

    # Filter to channels actually present
    probe_idx = int(reader.valid_indices[0])
    available = []
    for ch in channels:
        try:
            reader.bsccm.read_image(probe_idx, channel=ch)
            available.append(ch)
        except Exception:
            print(f"  Warning: channel {ch} not available, skipping.")
    channels = available
    if not channels:
        raise RuntimeError("None of the requested channels are available.")

    raw: Dict[str, Tuple[np.ndarray, List[int]]] = {}
    shared_indices: Optional[List[int]] = None
    for ch in channels:
        imgs, idxs = load_images_bsccm_pipeline(
            location=location, tiny=tiny, channel=ch,
            n_images=n_images, use_focused=use_focused,
            focus_method=focus_method, output_dir=output_dir,
        )
        raw[ch] = (imgs, idxs)
        if shared_indices is None:
            shared_indices = idxs
        else:
            n_common = min(len(shared_indices), len(idxs))
            shared_indices = shared_indices[:n_common]

    H_min = min(raw[ch][0].shape[1] for ch in channels)
    W_min = min(raw[ch][0].shape[2] for ch in channels)
    N = min(raw[ch][0].shape[0] for ch in channels)

    images_per_channel: Dict[str, np.ndarray] = {
        ch: raw[ch][0][:N, :H_min, :W_min] for ch in channels
    }

    print(
        f"Loaded {N} images per channel, {len(channels)} channels, "
        f"spatial size ({H_min}, {W_min})."
    )
    return images_per_channel, (shared_indices or [])[:N]
