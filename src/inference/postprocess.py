from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image


SOCIAL_GAZE_LABELS = (
    "share",
    "mutual",
    "single",
    "miss",
    "void",
)


def tensor_heatmap_to_numpy(heatmap: torch.Tensor) -> np.ndarray:
    """Convert [1,H,W], [H,W], or singleton-batched heatmap to float32 [H,W]."""
    hm = heatmap.detach().float().cpu()
    while hm.ndim > 2 and hm.shape[0] == 1:
        hm = hm.squeeze(0)
    if hm.ndim != 2:
        raise ValueError(f"Expected one 2-D heatmap, got shape {tuple(hm.shape)}.")
    return hm.numpy().astype(np.float32, copy=False)


def resize_heatmap(heatmap: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Resize a floating-point heatmap to original image (width, height)."""
    width, height = image_size
    pil_hm = Image.fromarray(heatmap.astype(np.float32), mode="F")
    resized = pil_hm.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def heatmap_argmax(heatmap: np.ndarray) -> Tuple[int, int]:
    """Return argmax as (x, y), not numpy's native (row, col)."""
    if heatmap.ndim != 2:
        raise ValueError("heatmap_argmax expects [H, W].")
    y, x = np.unravel_index(int(np.nanargmax(heatmap)), heatmap.shape)
    return int(x), int(y)


def point_to_normalized(
    point_px: Tuple[int, int],
    image_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Convert original-image pixel point to [0,1] normalized xy."""
    x, y = point_px
    width, height = image_size
    # Use W-1/H-1 so the final pixel maps exactly to 1.0.
    nx = x / max(width - 1, 1)
    ny = y / max(height - 1, 1)
    return float(nx), float(ny)


def extract_class_id(pred_pattern) -> int:
    """
    Accept BaseGazeModel-formatted class IDs or raw logits.

    - scalar / [1] -> class id
    - [5] / [1,5] -> argmax
    """
    if torch.is_tensor(pred_pattern):
        x = pred_pattern.detach().cpu()
        if x.numel() == 1:
            return int(x.reshape(-1)[0].item())
        if x.ndim == 1:
            return int(torch.argmax(x).item())
        if x.ndim == 2 and x.shape[0] == 1:
            return int(torch.argmax(x[0]).item())
        raise ValueError(f"Cannot extract one class from shape {tuple(x.shape)}.")

    arr = np.asarray(pred_pattern)
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    return int(arr.reshape(-1).argmax())


def class_name(class_id: int, labels: Sequence[str] = SOCIAL_GAZE_LABELS) -> str:
    if not 0 <= class_id < len(labels):
        raise ValueError(
            f"Social gaze class id {class_id} is outside label range 0..{len(labels)-1}."
        )
    return str(labels[class_id])
