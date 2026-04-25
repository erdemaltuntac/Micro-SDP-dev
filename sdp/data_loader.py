"""
data_loader.py — BSCCM image loading pipeline.

Loads single-cell images from the BSCCM dataset (or BSCCM-tiny) for one
or all channels, applies optional gradient-energy focus cropping, and
normalizes each image independently to [0, 1].

Requires read_bsccm_data.py (BSCCM_Img_Reader) to be on the Python path.
Optionally uses focus_l1.py (get_focus_analysis_for_image) when use_focused=True.
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

    Each image is normalised independently to [0, 1] via min-max scaling.
    Optional gradient-energy focus cropping is applied when use_focused=True.

    Parameters
    ----------
    location     : path to the BSCCM dataset root (or "BSCCM-tiny")
    tiny         : if True, use the BSCCM-tiny subset
    channel      : imaging channel name (e.g. "DPC_Left", "Brightfield")
    n_images     : max number of images to load; 0 = all available
    use_focused  : apply gradient-energy focus crop (requires focus_l1.py)
    focus_method : focus metric; "gradient" (default)
    output_dir   : scratch directory for BSCCM_Img_Reader

    Returns
    -------
    images  : (N, H, W) float64 in [0, 1]
    indices : list of BSCCM dataset indices used
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

    All channels are cropped to the same (H, W) determined by the minimum
    focused-image size across all channels and images, so every channel
    array has identical shape (N, H, W) ready for joint training.

    Parameters
    ----------
    location     : path to BSCCM dataset root (or "BSCCM-tiny")
    tiny         : if True, use BSCCM-tiny subset
    channels     : list of channel names (default: all five BSCCM channels)
    n_images     : max images per channel; 0 = all available
    use_focused  : apply focus crop
    focus_method : focus metric
    output_dir   : scratch directory for BSCCM_Img_Reader

    Returns
    -------
    images_per_channel : dict channel -> (N, H, W) float64 in [0, 1]
    indices            : shared list of dataset indices (same for all channels)
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
