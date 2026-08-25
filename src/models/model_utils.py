import torch
import einops
from itertools import repeat
import math
from typing import Iterable, Union
import torch.nn.functional as F


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")

    assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"

    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])

    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = einops.rearrange(pos_emb, "b (h w) d -> b d h w", h=h, w=w, d=embed_dim)

    return pos_emb

def pair(size):
    return size if isinstance(size, (list, tuple)) else (size, size)

def to_2tuple(x):
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))

def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)

def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def compute_spatial_dispersion(pred_heatmap: torch.Tensor,):
    """
    Computes spatial dispersion for a batch of predicted heatmaps.

    Args:
        pred_heatmap (Tensor): (B, H, W) predicted gaze heatmaps.

    Returns:
        v (Tensor): (B,) tensor of spatial dispersion values, where higher values indicate more dispersed predictions.
    """
    if pred_heatmap.ndim != 3:
        raise ValueError(
            "pred_heatmap must have shape [B, H, W], got "
            f"{tuple(pred_heatmap.shape)}."
        )

    B, H, W = pred_heatmap.shape
    device = pred_heatmap.device

    # Keep invalid activations from propagating into the confidence gate.
    pred_heatmap = torch.nan_to_num(pred_heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    pred_heatmap = pred_heatmap.clamp_min(0.0)

    # Normalize the heatmaps without allocating device scalars in the loop.
    normalizer = pred_heatmap.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    pred_heatmap = pred_heatmap / normalizer

    # Create coordinate grid
    y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
    x_coords = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)

    # Find (x_hat, y_hat) = location of max value in heatmap for each batch
    flat_indices = pred_heatmap.view(B, -1).argmax(dim=1)  # (B,)
    y_hat = flat_indices // W
    x_hat = flat_indices % W

    # Expand gaze points for distance computation
    x_hat = x_hat.view(B, 1, 1)
    y_hat = y_hat.view(B, 1, 1)

    # Compute distance map (B, H, W)
    D = torch.sqrt((x_coords - x_hat)**2 + (y_coords - y_hat)**2)

    # Compute weighted average dispersion (B,)
    v = (D * pred_heatmap).sum(dim=[1, 2])

    return v  # shape: (B,)


def compute_confidence(
        pred_heatmap: torch.Tensor,
        mu_dist: Union[float, torch.Tensor] = 3.0,
        sigma_dist: Union[float, torch.Tensor] = 1.0,
        eps: float = 1e-8,
        clamp_output: bool = True,
    ) -> torch.Tensor:
    """
    Compute confidence scores from pred_heatmap based on spatial dispersion.

    Args:
        pred_heatmap (Tensor): (B, H, W) predicted gaze heatmaps.
        mu_dist (Tensor|float): Mean of log-dispersion distribution (scalar or broadcastable to dispersion).
        sigma_dist (Tensor|float): Std of log-dispersion distribution (scalar or broadcastable to dispersion).
        eps (float): Numerical stability constant (used for log and sigma clamp).
        clamp_output (bool): If True, clamp final confidence to [0, 1].
    Returns:
        c (Tensor): (B,) confidence scores between 0 and 1, where higher values indicate higher confidence.
                    The confidence is inversely proportional to the spatial dispersion.
    """
    if not torch.is_tensor(pred_heatmap):
        raise TypeError("Heatmap must be a torch.Tensor")
    
    dispersion = compute_spatial_dispersion(pred_heatmap)
    log_v = torch.log(dispersion.clamp_min(eps))
    sigma = torch.as_tensor(
        sigma_dist, dtype=log_v.dtype, device=log_v.device
    ).clamp_min(eps)
    mu = torch.as_tensor(mu_dist, dtype=log_v.dtype, device=log_v.device)
    z = (log_v - mu) / sigma

    # 1 - Φ(z) = 0.5 * erfc(z / sqrt(2)). Using a Python scalar avoids
    # creating a new CUDA tensor for sqrt(2) for every person in every batch.
    c = 0.5 * torch.erfc(z / math.sqrt(2.0))

    if clamp_output:
        c = c.clamp(0.0, 1.0)
    return c
