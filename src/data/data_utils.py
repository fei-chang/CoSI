from typing import Tuple
import torch
from torchvision import transforms
import numpy as np
import pandas as pd

"""
Some functions in this script are adapted from ViTGaze (https://github.com/hustvl/ViTGaze),
with modifications made for this project. 
We gratefully acknowledge the original authors and their contributions.
"""

import numpy as np

def perturb_heatmap(heatmap,
                    intensity_range=(0.6, 1.5),
                    noise_std=0.1,
                    dropout_prob=0.1,
                    use_torch=True):
    """
    Apply random intensity perturbations to a heatmap while preserving spatial structure.
    
    Args:
        heatmap: Input heatmap (numpy array or torch tensor)
        intensity_range: Tuple of (min_scale, max_scale) for random intensity scaling
        noise_std: Standard deviation of Gaussian noise to add
        dropout_prob: Probability of randomly dropping out values
        
    Returns:
        Perturbed heatmap with same spatial structure
    """
    # Convert to numpy if needed
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().numpy()
    
    # 1. Random intensity scaling
    scale = np.random.uniform(*intensity_range)
    heatmap = heatmap * scale
    
    # 2. Add Gaussian noise
    noise = np.random.normal(0, noise_std, heatmap.shape)
    heatmap = heatmap + noise
    
    # 3. Random value dropout (only where heatmap > 0)
    if dropout_prob > 0:
        mask = np.random.rand(*heatmap.shape) > dropout_prob
        # Only apply dropout to non-zero regions
        mask = np.where(heatmap > 0, mask, 1)
        heatmap = heatmap * mask
    
    # Ensure values are non-negative
    heatmap = np.clip(heatmap, 0, None)
    
    # Normalize to maintain similar intensity range
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Convert back to torch if needed
    if use_torch:
        heatmap = torch.from_numpy(heatmap).float()
    return heatmap

def generate_random_points(center_x: float, center_y: float, k: int=5, radius: float = 0.1) -> list:
    """
    Generate k random points around a center point within 0-1 bounds.
    
    Args:
        center_x (float): X coordinate of center point (0-1)
        center_y (float): Y coordinate of center point (0-1)
        k (int): Number of points to generate
        radius (float): Maximum radius for generated points
        
    Returns:
        list: List of dictionaries containing x, y coordinates
    """
    if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
        return [] # just return the empty list
    
    points = []
    max_attempts = 20
    
    for _ in range(k):
        new_x, new_y = center_x, center_y
        attempts = 0
        
        while attempts < max_attempts:
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, radius)
            
            new_x = center_x + distance * np.cos(angle)
            new_y = center_y + distance * np.sin(angle)
            
            if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                break
                
            attempts += 1
        
        new_x = np.clip(new_x, 0, 1)
        new_y = np.clip(new_y, 0, 1)
        
        points.append({
            'x': round(float(new_x), 4),
            'y': round(float(new_y), 4)
        })
    
    return points

def draw_heatmap_with_errors(img: torch.Tensor,
                            center_pt: tuple,
                            error_sigma: float = 2.0,
                            main_sigma: float = 2.0,
                            num_error_points: int = 5,
                            error_radius: float = 10.0,  # Now in pixels
                            error_intensity: float = 0.3,
                            type: str = "Gaussian") -> torch.Tensor:
    """
    Draw a heatmap with additional error points around the center point.
    
    Args:
        img (torch.Tensor): Input image to draw heatmap on
        center_pt (tuple): (x, y) coordinates of the center point in pixel space
        error_sigma (float): Sigma for error point Gaussians
        main_sigma (float): Sigma for main point Gaussian
        num_error_points (int): Number of error points to generate
        error_radius (float): Maximum radius for error points in pixels
        error_intensity (float): Intensity multiplier for error points (0-1)
        type (str): Type of distribution ("Gaussian" or "Cauchy")
        
    Returns:
        torch.Tensor: Heatmap with main point and error points
    """
    h, w = img.shape[-2:]
    
    # Convert center point to relative coordinates (0-1) for generate_random_points
    center_x, center_y = center_pt[0] / w, center_pt[1] / h

    error_radius_rel = error_radius / min(w, h)  # Convert pixel radius to relative
    
    # Generate error points in relative coordinates
    error_points = generate_random_points(
        center_x, center_y,
        k=num_error_points,
        radius=error_radius_rel
    )
    
    # Draw main heatmap
    img = draw_labelmap(img, center_pt, main_sigma, type=type)
    
    # Draw error points with lower intensity
    for point in error_points:
        # Convert relative coordinates to pixel space
        error_pt = (int(point['x'] * w), int(point['y'] * h))
        error_map = draw_labelmap(
            torch.zeros_like(img),
            error_pt,
            error_sigma,
            type=type
        )
        img = img + error_map * error_intensity
    
    # Normalize the combined heatmap
    if torch.max(img) > 0:
        img = img / torch.max(img)
    
    return img.float()

def compute_normalized_gaze_vector(headbox, gaze_point):
    """
    Compute the normalized vector from headbox center to gaze point.
    
    Args:
        headbox (tuple): Tuple of (xmin, ymin, xmax, ymax) coordinates
        gaze_point (tuple): Tuple of (x, y) coordinates for gaze point
    
    Returns:
        numpy.ndarray: Normalized vector from headbox center to gaze point
    """
    # Compute headbox center
    center_x = (headbox[0] + headbox[2]) / 2
    center_y = (headbox[1] + headbox[3]) / 2
    
    # Compute vector from center to gaze point
    vector = np.array([
        gaze_point[0] - center_x,
        gaze_point[1] - center_y
    ])
    
    # Normalize the vector
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)  # Return zero vector if norm is 0
    normalized_vector = vector / norm
    
    return normalized_vector

def compute_batch_gaze_vectors(normalized_hboxes, normalized_gazes):
    """
    Compute normalized gaze vectors for batches of headboxes and gaze points.
    
    Args:
        normalized_hboxes (np.ndarray): Array of shape (N, 4) containing normalized headbox coordinates
        normalized_gazes (np.ndarray): Array of shape (N, 2) containing normalized gaze points
    
    Returns:
        np.ndarray: Array of shape (N, 2) containing normalized gaze vectors
    """
    N = len(normalized_hboxes)
    gaze_vectors = np.zeros((N, 2))
    
    for i in range(N):
        gaze_vectors[i] = compute_normalized_gaze_vector(
            normalized_hboxes[i],
            normalized_gazes[i]
        )
    
    return gaze_vectors

def to_numpy(tensor: torch.Tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray: np.ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def get_head_box_channel(
    xmin, ymin, xmax, ymax, width, height, resolution, coordconv=False
):
    head_box = (
        np.array([xmin / width, ymin / height, xmax / width, ymax / height])
        * resolution
    )
    int_head_box = head_box.astype(int)
    int_head_box = np.clip(int_head_box, 0, resolution - 1)
    if int_head_box[0] == int_head_box[2]:
        if int_head_box[0] == 0:
            int_head_box[2] = 1
        elif int_head_box[2] == resolution - 1:
            int_head_box[0] = resolution - 2
        elif abs(head_box[2] - int_head_box[2]) > abs(head_box[0] - int_head_box[0]):
            int_head_box[2] += 1
        else:
            int_head_box[0] -= 1
    if int_head_box[1] == int_head_box[3]:
        if int_head_box[1] == 0:
            int_head_box[3] = 1
        elif int_head_box[3] == resolution - 1:
            int_head_box[1] = resolution - 2
        elif abs(head_box[3] - int_head_box[3]) > abs(head_box[1] - int_head_box[1]):
            int_head_box[3] += 1
        else:
            int_head_box[1] -= 1
    head_box = int_head_box
    if coordconv:
        unit = np.array(range(0, resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit + i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution, resolution), dtype=np.float32)
        head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel


def draw_labelmap(img, pt, sigma, type="Gaussian", use_torch=True):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    size = int(6 * sigma + 1)
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [ul[0] + size, ul[1] + size]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif type == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    # img = img / np.max(img)
    if use_torch:
        return to_torch(img)
    return img


def draw_labelmap_no_quant(img, pt, sigma, type="Gaussian"):
    img = to_numpy(img)
    shape = img.shape
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dist_matrix = (yy - float(pt[0])) ** 2 + (xx - float(pt[1])) ** 2
    if type == "Gaussian":
        g = np.exp(-dist_matrix / (2 * sigma**2))
    elif type == "Cauchy":
        g = sigma / ((dist_matrix + sigma**2) ** 1.5)
    g[dist_matrix > 10 * sigma**2] = 0
    img += g
    # img = img / np.max(img)
    return to_torch(img)


def multi_hot_targets(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int, [p[0] * float(w), p[1] * float(h)])
            x = min(x, w - 1)
            y = min(y, h - 1)
            target_map[y, x] = 1
    return target_map


def get_cone(tgt, src, wh, theta=150):
    eye = src * wh
    gaze = tgt * wh

    pixel_mat = np.stack(
        np.meshgrid(np.arange(wh[0]), np.arange(wh[1])),
        -1,
    )

    dot_prod = np.sum((pixel_mat - eye) * (gaze - eye), axis=-1)
    gaze_vector_norm = np.sqrt(np.sum((gaze - eye) ** 2))
    pixel_mat_norm = np.sqrt(np.sum((pixel_mat - eye) ** 2, axis=-1))

    gaze_cones = dot_prod / (gaze_vector_norm * pixel_mat_norm)
    gaze_cones = np.nan_to_num(gaze_cones, nan=1)

    theta = theta * (np.pi / 180)
    beta = np.arccos(gaze_cones)
    # Create mask where true if beta is less than theta/2
    pixel_mat_presence = beta < (theta / 2)

    # Zero out values outside the gaze cone
    gaze_cones[~pixel_mat_presence] = 0
    gaze_cones = np.clip(gaze_cones, 0, None)

    return torch.from_numpy(gaze_cones).unsqueeze(0).float()


def get_transform(
    input_resolution: int, mean: Tuple[int, int, int], std: Tuple[int, int, int]
):
    return transforms.Compose(
        [
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def smooth_by_conv(window_size, df, col):
    padded_track = pd.concat(
        [
            pd.DataFrame([[df.iloc[0][col]]] * (window_size // 2), columns=[0]),
            df[col],
            pd.DataFrame([[df.iloc[-1][col]]] * (window_size // 2), columns=[0]),
        ]
    )
    smoothed_signals = np.convolve(
        padded_track.squeeze(), np.ones(window_size) / window_size, mode="valid"
    )
    return smoothed_signals

def point_in_bbox(x, y, bbox):
    """Check if a point (x, y) is inside a bounding box (xmin, ymin, xmax, ymax) in normalized coordinates [0,1]."""
    xmin, ymin, xmax, ymax = bbox
    return xmin <= x <= xmax and ymin <= y <= ymax

def associate_label(label_idx: int) -> int:
    # Original label list
    label_ls = ['Share', 'Mutual', 'Single', 'Miss', 'Void']

    # Build mapping {label: idx}
    mapping = {label: idx for idx, label in enumerate(label_ls)}

    # Define swap mapping in terms of indices
    swap_mapping = {
        mapping['Share']:  mapping['Share'],
        mapping['Mutual']: mapping['Mutual'],
        mapping['Single']: mapping['Miss'],
        mapping['Miss']:   mapping['Single'],
        mapping['Void']:   mapping['Void'],
    }

    return swap_mapping[label_idx]
