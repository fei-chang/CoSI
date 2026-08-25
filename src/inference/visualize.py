from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

from .preprocess import ImageInput, load_rgb_image
from .types import DyadicPrediction, SinglePersonGazePrediction


# OpenCV uses BGR colors.
PRINCIPAL_COLOR = (0, 255, 0)   # green
ASSOCIATE_COLOR = (0, 0, 255)   # red

def _heatmap_to_colormap(heatmap: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Convert a float heatmap into a uint8 OpenCV JET color map."""
    width, height = image_size

    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        hm = np.squeeze(hm)
    if hm.ndim != 2:
        raise ValueError(f"Heatmap must be 2D after squeeze, got shape {hm.shape}")

    if hm.shape != (height, width):
        hm = cv2.resize(hm, (width, height), interpolation=cv2.INTER_LINEAR)

    finite = np.isfinite(hm)
    if not finite.any():
        hm_u8 = np.zeros((height, width), dtype=np.uint8)
    else:
        valid_values = hm[finite]
        hm_min = float(valid_values.min())
        hm_max = float(valid_values.max())

        hm = np.nan_to_num(hm, nan=hm_min, posinf=hm_max, neginf=hm_min)

        if hm_max > hm_min:
            hm = (hm - hm_min) / (hm_max - hm_min)
        else:
            hm = np.zeros_like(hm)

        hm_u8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)

    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)


def _overlay_heatmap(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.40,
) -> np.ndarray:
    """Overlay a gaze heatmap on an image using OpenCV only."""
    height, width = image_bgr.shape[:2]
    heatmap_bgr = _heatmap_to_colormap(heatmap, (width, height))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def _draw_person(
    canvas: np.ndarray,
    box_px: Tuple[int, int, int, int],
    gaze_point_px: Sequence[float],
    label: str,
    color: Tuple[int, int, int],
    *,
    person_name: str,
    box_thickness: int = 1,
    line_thickness: int = 1,
    point_radius: int = 5,
) -> np.ndarray:
    """Draw one person's head box, gaze ray, gaze point, and label."""
    x1, y1, x2, y2 = box_px
    gx, gy = int(round(gaze_point_px[0])), int(round(gaze_point_px[1]))

    height, width = canvas.shape[:2]
    gx = int(np.clip(gx, 0, width - 1))
    gy = int(np.clip(gy, 0, height - 1))

    head_center = (
        int(round((x1 + x2) / 2.0)),
        int(round((y1 + y2) / 2.0)),
    )

    # Head bounding box.
    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        color,
        thickness=box_thickness,
        lineType=cv2.LINE_AA,
    )

    # Gaze direction line from head center to predicted gaze point.
    cv2.line(
        canvas,
        head_center,
        (gx, gy),
        color,
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )

    # Filled gaze point.
    cv2.circle(
        canvas,
        (gx, gy),
        point_radius,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Person / social-gaze label in the same person's color.
    text = f"{person_name}: {label}"
    text_y = max(24, y1 - 10)
    cv2.putText(
        canvas,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return canvas


def save_prediction_visualizations(
    image: ImageInput,
    head_boxes: Sequence[Sequence[float]],
    prediction: DyadicPrediction,
    output_dir,
    *,
    heatmap_alpha: float = 0.40,
):
    """
    Save visualization images using PIL/OpenCV only (no matplotlib).

    Colors:
        principal -> green
        associate -> red

    Args:
        image:
            Input image path/PIL/numpy image.
        head_boxes:
            Exactly two xyxy boxes. By default they are interpreted as
            normalized [0,1] coordinates.
        prediction:
            DyadicPrediction returned by DyadicGazePredictor.
        output_dir:
            Directory where PNG files are written.
        boxes_normalized:
            True for normalized [0,1] xyxy boxes; False for pixel xyxy boxes.
        heatmap_alpha:
            Strength of the heatmap overlay.

    Saves:
        heatmaps.png
        gaze_points.png
    """
    if len(head_boxes) != 2:
        raise ValueError(f"Expected exactly two head boxes, got {len(head_boxes)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load_rgb_image gives RGB; OpenCV drawing/writing uses BGR.
    rgb = np.asarray(load_rgb_image(image))
    base_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    height, width = base_bgr.shape[:2]

    principal_box = head_boxes[0]
    associate_box = head_boxes[1]

    people = (
        (
            "principal",
            prediction.principal,
            principal_box,
            PRINCIPAL_COLOR,
        ),
        (
            "associate",
            prediction.associate,
            associate_box,
            ASSOCIATE_COLOR,
        ),
    )

    # One heatmap visualization for each person.
    canvas = base_bgr.copy()
    combined_heatmaps = None
    for name, pred, box_px, color in people:
        if combined_heatmaps is None:
            combined_heatmaps = pred.heatmap.copy() 
        else:
            # Add subsequent heatmaps
            combined_heatmaps += pred.heatmap
    canvas = _overlay_heatmap(
        canvas,
        combined_heatmaps,
        alpha=heatmap_alpha,
        )
    for name, pred, box_px, color in people:
        _draw_person(
            canvas,
            box_px,
            pred.gaze_point_px,
            pred.social_gaze_label,
            color,
            person_name=name,
        )
    cv2.imwrite(str(output_dir / f"heatmaps.png"), canvas)

    # Combined visualization: original image + both people.
    combined = base_bgr.copy()

    for name, pred, box_px, color in people:
        _draw_person(
            combined,
            box_px,
            pred.gaze_point_px,
            pred.social_gaze_label,
            color,
            person_name=name,
        )

    cv2.imwrite(str(output_dir / "gaze_points.png"), combined)

def save_single_gaze_visualizations(
    image: ImageInput,
    head_box: Sequence[float],
    prediction: SinglePersonGazePrediction,
    output_dir,
    *,
    heatmap_alpha: float = 0.40,
):
    """Save ``heatmap.png`` and ``gaze_point.png`` for one-person gaze follow."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(load_rgb_image(image))
    base_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = base_bgr.shape[:2]

    x1, y1, x2, y2 = [int(round(float(v))) for v in head_box]
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    box_px = (x1, y1, x2, y2)

    gx = int(round(prediction.gaze_point_px[0]))
    gy = int(round(prediction.gaze_point_px[1]))
    gx = int(np.clip(gx, 0, width - 1))
    gy = int(np.clip(gy, 0, height - 1))
    center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))

    def draw_gaze(canvas):
        cv2.rectangle(
            canvas, (x1, y1), (x2, y2), PRINCIPAL_COLOR,
            thickness=1, lineType=cv2.LINE_AA,
        )
        cv2.line(
            canvas, center, (gx, gy), PRINCIPAL_COLOR,
            thickness=1, lineType=cv2.LINE_AA,
        )
        cv2.circle(
            canvas, (gx, gy), 6, PRINCIPAL_COLOR,
            thickness=-1, lineType=cv2.LINE_AA,
        )
        return canvas

    heatmap_canvas = _overlay_heatmap(
        base_bgr.copy(), prediction.heatmap, alpha=heatmap_alpha
    )
    draw_gaze(heatmap_canvas)
    cv2.imwrite(str(output_dir / "heatmap.png"), heatmap_canvas)

    gaze_canvas = draw_gaze(base_bgr.copy())
    cv2.imwrite(str(output_dir / "gaze_point.png"), gaze_canvas)

