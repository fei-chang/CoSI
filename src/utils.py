from typing import Union, Iterable, Tuple, Optional
import numpy as np
import torch
import os
import random
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, accuracy_score, classification_report
def normalize_heatmaps_batch(heatmaps):
    """
    Normalize a batch of heatmaps to range [0, 1] while preserving distribution shape.
    
    Args:
        heatmaps (torch.Tensor): Input tensor of shape (BN, 224, 224)
        
    Returns:
        torch.Tensor: Normalized heatmaps with values scaled to [0, 1]
    """
    # Find min and max for each heatmap in the batch
    # keepdim=True maintains the dimensions for broadcasting
    batch_min = heatmaps.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    batch_max = heatmaps.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    
    # Handle constant heatmaps (where max == min)
    is_constant = batch_max == batch_min
    
    # Compute normalization
    normalized = torch.zeros_like(heatmaps)
    non_constant_mask = ~is_constant.squeeze()
    
    if non_constant_mask.any():
        normalized[non_constant_mask] = (
            (heatmaps[non_constant_mask] - batch_min[non_constant_mask]) / 
            (batch_max[non_constant_mask] - batch_min[non_constant_mask])
        )
    
    return normalized

def save_heatmaps_np(heatmap, save_dir, filename='heatmap'):
    """
    Save a single heatmap to disk
    
    Args:
        heatmap (numpy.ndarray): Single heatmap array of shape (224, 224)
        save_dir (str): Directory to save the heatmap
        filename (str): Name for the saved file (without .npy extension)
    """
    # Create directory if it doesn't exist
    save_path = os.path.join(save_dir, f'{filename}.npy')

    # Get the directory path by removing the filename
    folder_path = os.path.dirname(save_path)
    # Create the directory and all necessary parent directories
    os.makedirs(folder_path, exist_ok=True)
    np.save(save_path, heatmap)

def load_heatmaps_np(save_dir, filename='heatmap', to_torch=False):
    """
    Load a single heatmap from disk
    
    Args:
        save_dir (str): Directory containing the heatmap
        filename (str): Name of the file (without .npy extension)
        to_torch (bool): If True, converts numpy array to torch tensor
    
    Returns:
        numpy.ndarray or torch.Tensor: Loaded heatmap of shape (224, 224)
    """
    load_path = os.path.join(save_dir, f'{filename}.npy')
    heatmap = np.load(load_path)
    
    if to_torch:
        return torch.from_numpy(heatmap)
    return heatmap

def pretty_print_losses(loss_dict):
    max_key_length = max(len(key) for key in loss_dict.keys())
    msg = "\nLoss Values:\n" + "=" * (max_key_length + 12) + "\n"
    
    for key, value in loss_dict.items():
        val = float(value.detach().cpu())
        msg += f"{key:<{max_key_length}} : {val:>8.4f}\n"
    
    msg += "=" * (max_key_length + 12)
    return msg

def compute_angular_error(pred_vector, target_vector):
    """
    Compute angular error in degrees between predicted and target gaze vectors.
    
    Args:
        pred_vector (np.ndarray): Predicted gaze vector [2]
        target_vector (np.ndarray): Ground truth gaze vector [2]
        
    Returns:
        float: Angular error in degrees (°)
    """
    # Normalize vectors
    pred_norm = pred_vector / np.linalg.norm(pred_vector)
    target_norm = target_vector / np.linalg.norm(target_vector)
    
    # Compute cosine similarity
    cos_sim = np.clip(np.dot(pred_norm, target_norm), -1.0 + 1e-6, 1.0 - 1e-6)
    
    # Convert to degrees
    angle_error = np.arccos(cos_sim) * 180.0 / np.pi
    
    return float(angle_error)

def set_random_seed(seed: Optional[int] = None, 
                   use_cuda: bool = True,
                   deterministic_cudnn: bool = True) -> int:
    """
    Set random seed for reproducibility across multiple libraries.
    
    Args:
        seed (int, optional): Random seed to use. If None, generates a random seed.
        use_cuda (bool): Whether to set CUDA seeds for PyTorch. Defaults to True.
        deterministic_cudnn (bool): Whether to make cuDNN deterministic. Defaults to True.
    
    Returns:
        int: The seed that was set
        
    Note:
        Setting deterministic_cudnn=True may impact performance but ensures reproducibility
    """
    if seed is None:
        # Generate a random seed if none is provided
        seed = random.randint(1, 2**32 - 1)
    
    # Python's built-in random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    
    # Set environment variable for additional libraries that check it
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    return seed

def inverse_transform(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().permute(0, 2, 3, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor * std + mean
    return cv2.cvtColor((tensor.numpy() * 255).astype(np.uint8)[0], cv2.COLOR_RGB2BGR)

def draw(data, heatmap, out_path, on_img=True):
    img = inverse_transform(data["images"])
    head_channel = cv2.applyColorMap(
        (data["head_channels"].squeeze().detach().cpu().numpy() * 255).astype(np.uint8),
        cv2.COLORMAP_BONE,
    )
    hm = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = hm
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    if on_img:
        img = cv2.addWeighted(img, 1, heatmap, 0.5, 1)
    else:
        img = heatmap
    # img = cv2.addWeighted(img, 1, head_channel, 0.1, 1)
    cv2.imwrite(out_path, img)

def draw_origin_img(data, out_path):
    img = inverse_transform(data["images"])
    hm = cv2.applyColorMap(
        (data["heatmaps"].squeeze().detach().cpu().numpy() * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    hm[data["heatmaps"].squeeze().detach().cpu().numpy() == 0] = 0
    hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
    head_channel = cv2.applyColorMap(
        (data["head_channels"].squeeze().detach().cpu().numpy() * 255).astype(np.uint8),
        cv2.COLORMAP_BONE,
    )
    head_channel[data["head_channels"].squeeze().detach().cpu().numpy() < 0.1] = 0
    hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
    ori = cv2.addWeighted(img, 1, hm, 0.5, 1)
    ori = cv2.addWeighted(ori, 1, head_channel, 0.1, 1)
    cv2.imwrite(out_path, ori)


def classify_gaze(main_gaze, other_gaze, main_head, other_head,
                share_thres=0.0250,
                pattern_type='multi_class'):
    """
    Classifies gaze interactions into:
    0 = Share (or = 1 if pattern_type==binary_share)
    1 = Mutual(or collapse to 0 if pattern_type==binary_mutual)
    2 = Single (or = 1 if pattern_type==binary_LAH, or collapse to 0 if pattern_type!=multi_class )
    3 = Miss (or collapse to 0 if pattern_type!=multi_class)
    4 = Void (or collapse to 0 if pattern_type!=multi_class)

    Parameters
    ----------
    main_gaze : np.ndarray  # shape (2,) [x, y]
    other_gaze : np.ndarray # shape (2,) [x, y]
    main_head : np.ndarray  # [x_min, y_min, x_max, y_max]
    other_head : np.ndarray # [x_min, y_min, x_max, y_max]
    multi_class : bool      # whether to return multi-class result

    Returns
    -------
    int : classification code
    """

    if pattern_type not in ["multi_class", "binary_share", 'binary_mutual', 'binary_LAH']:
        raise ValueError("pattern_type must be one of the following \n 'multi_class'/'binary_share'/'binary_mutual'/'binary_LAH'.")
    
    def is_inside(point, box):
        """Check if point (x, y) is inside bounding box [x_min, y_min, x_max, y_max]."""
        return (box[0] <= point[0] <= box[2]) and (box[1] <= point[1] <= box[3])

    def distance(p1, p2):
        """Euclidean distance between two points."""
        return np.linalg.norm(p1 - p2)
    
    # --- Mutual ---
    if is_inside(main_gaze, other_head) and is_inside(other_gaze, main_head):
        if pattern_type != "binary_share":
            return 1
        else:  
            return 0

    # --- Single: main_gaze inside other_head ---
    if is_inside(main_gaze, other_head) and not is_inside(other_gaze, main_head):
        if pattern_type == "multi_class":
            return 2 
        elif pattern_type == "binary_LAH":
            return 1
        else:
            return 0

    # --- Miss : other_gaze inside main_head ---
    if not is_inside(main_gaze, other_head) and is_inside(other_gaze, main_head):
        return 3 if pattern_type == "multi_class" else 0

    # --- Share ---
    if distance(main_gaze, other_gaze) <= share_thres:
        if pattern_type == "binary_share":
            return 1
        return 0

    # ---  Void ---
    return 4 if pattern_type == "multi_class" else 0
