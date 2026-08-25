from pathlib import Path
from typing import Sequence, Tuple, Union, Dict, List

import numpy as np
import torch
from PIL import Image
from .types import Box
from ultralytics import YOLO

ImageInput = Union[str, Path, Image.Image, np.ndarray]

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

def load_head_detector(
    weights="yolo_head_best.pt",
    device="cuda:0",
):
    """Load a yolo head detector and return."""
    model = YOLO(weights, verbose=False)
    model.to(device)
    model.conf = 0.25
    model.iou = 0.45
    model.max_det = 10

    return model

def detect_heads(
    model,
    image,
    *,
    conf=None,
    iou=None,
    max_det=None,
    imgsz=None,
):
    """
    Detect heads and return pixel-space boxes with confidences.

    Optional inference arguments are passed through to ``ultralytics.YOLO``.
    Existing callers can continue to use ``detect_heads(model, image)``.
    """
    img = Image.open(image).convert("RGB")
    width, height = img.size

    predict_kwargs = {"verbose": False}
    if conf is not None:
        predict_kwargs["conf"] = float(conf)
    if iou is not None:
        predict_kwargs["iou"] = float(iou)
    if max_det is not None:
        predict_kwargs["max_det"] = int(max_det)
    if imgsz is not None:
        predict_kwargs["imgsz"] = int(imgsz)

    results = model(image, **predict_kwargs)
    boxes = results[0].boxes

    heads = []

    for det in boxes:
        x1, y1, x2, y2 = det.xyxy[0].detach().cpu().numpy()
        conf = float(det.conf[0])

        heads.append({
            "box": [
                int(round(np.clip(x1, 0, width - 1))),
                int(round(np.clip(y1, 0, height - 1))),
                int(round(np.clip(x2, 0, width - 1))),
                int(round(np.clip(y2, 0, height - 1)))
            ],
            "confidence": float(conf),
        })

    return heads

def order_two_heads(detections: Sequence[Dict]) -> Tuple[Dict, Dict]:
    """
    1. Keep the two highest-confidence heads.
    2. Order those two by horizontal center.
    3. Left = principal, right = associate.
    """
    if len(detections) < 2:
        raise RuntimeError(
            f"Expected at least 2 detected heads, but YOLOv5 returned "
            f"{len(detections)}. Try lowering --head-conf."
        )

    best_two = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    )[:2]

    best_two.sort(
        key=lambda item: (item["box"][0] + item["box"][2]) / 2.0
    )

    return best_two[0], best_two[1]

def select_one_head(detections: Sequence[Dict]) -> Dict:
    """Return the highest-confidence detected head for single-person inference."""
    if len(detections) < 1:
        raise RuntimeError(
            "No head was detected. Try lowering --head-conf or pass --head-box manually."
        )

    return max(detections, key=lambda item: item["confidence"])


def normalized_box(box: Sequence[float], image_size: Tuple[int, int]) -> List[float]:
    width, height = image_size
    x1, y1, x2, y2 = map(float, box)
    return [x1 / width, y1 / height, x2 / width, y2 / height]



def load_rgb_image(image: ImageInput) -> Image.Image:
    """Load a path/PIL/numpy image and return RGB PIL."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("numpy image must have shape [H, W, 3] or [H, W, 4]")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr[..., :3]).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def validate_box(box: Sequence[float], image_size: Tuple[int, int]) -> Box:
    """
    Validate and clip an xyxy box expressed in ORIGINAL-IMAGE PIXELS.

    Returns:
        (x1, y1, x2, y2), clipped to image bounds.
    """
    if len(box) != 4:
        raise ValueError("Each head box must be [x1, y1, x2, y2].")

    width, height = image_size
    x1, y1, x2, y2 = map(float, box)

    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid head box after clipping: {(x1, y1, x2, y2)} for image {image_size}."
        )
    return x1, y1, x2, y2


def image_to_tensor(image: Image.Image, input_size: Tuple[int, int]) -> torch.Tensor:
    """
    Resize to model resolution and apply ImageNet normalization.

    Returns:
        float tensor [3, H, W].
    """
    target_w, target_h = input_size
    resized = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def box_to_head_channel(
    box: Box,
    original_size: Tuple[int, int],
    input_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert an original-image pixel xyxy head box to a binary model-space mask.

    Returns:
        float tensor [1, H, W].
    """
    orig_w, orig_h = original_size
    target_w, target_h = input_size
    x1, y1, x2, y2 = box

    sx = target_w / float(orig_w)
    sy = target_h / float(orig_h)

    ix1 = int(np.floor(x1 * sx))
    iy1 = int(np.floor(y1 * sy))
    ix2 = int(np.ceil(x2 * sx))
    iy2 = int(np.ceil(y2 * sy))

    ix1 = max(0, min(ix1, target_w - 1))
    iy1 = max(0, min(iy1, target_h - 1))
    ix2 = max(ix1 + 1, min(ix2, target_w))
    iy2 = max(iy1 + 1, min(iy2, target_h))

    channel = torch.zeros((1, target_h, target_w), dtype=torch.float32)
    channel[:, iy1:iy2, ix1:ix2] = 1.0
    return channel


def prepare_dyadic_batch(
    image: ImageInput,
    head_boxes: Sequence[Sequence[float]],
    input_size: Tuple[int, int],
):
    """
    Build the minimal batch expected by CoSi/BaseGazeModel at inference time.

    Input:
        image: path/PIL/numpy image
        head_boxes: exactly two original-image pixel boxes in xyxy order
        input_size: (width, height) model input resolution

    Output:
        batch dict, RGB PIL image, validated boxes
    """
    if len(head_boxes) != 2:
        raise ValueError(f"Expected exactly 2 head boxes, got {len(head_boxes)}.")

    pil_image = load_rgb_image(image)
    original_size = pil_image.size
    boxes = [validate_box(box, original_size) for box in head_boxes]

    image_tensor = image_to_tensor(pil_image, input_size).unsqueeze(0)
    head_channels = [
        box_to_head_channel(box, original_size, input_size).unsqueeze(0)
        for box in boxes
    ]

    batch = {
        "images": image_tensor,
        "principal": {"head_channel": head_channels[0]},
        "associate": {"head_channel": head_channels[1]},
    }
    return batch, pil_image, boxes

def prepare_single_person_batch(
    image: ImageInput,
    head_box: Sequence[float],
    input_size: Tuple[int, int],
    *,
    person_key: str = "principal",
):
    """
    Build the minimal inference batch for one person's gaze-follow prediction.

    Unlike ``prepare_dyadic_batch``, this does not create a dummy second person.
    The resulting batch contains only ``images`` and one person entry with a
    binary ``head_channel``.

    Returns:
        batch dict, RGB PIL image, validated head box
    """
    if person_key not in ("principal", "associate"):
        raise ValueError(
            f"person_key must be 'principal' or 'associate', got {person_key!r}."
        )

    pil_image = load_rgb_image(image)
    original_size = pil_image.size
    box = validate_box(head_box, original_size)

    image_tensor = image_to_tensor(pil_image, input_size).unsqueeze(0)
    head_channel = box_to_head_channel(
        box, original_size, input_size
    ).unsqueeze(0)

    batch = {
        "images": image_tensor,
        person_key: {"head_channel": head_channel},
    }
    return batch, pil_image, box

