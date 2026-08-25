"""Shared setup and I/O helpers for the prediction entry points."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import torch
from hydra import compose, initialize_config_dir
from tqdm import tqdm

from models import build_model
from .preprocess import detect_heads, load_head_detector


def add_head_detector_arguments(
    parser: argparse.ArgumentParser,
    *,
    allow_manual_box: bool = False,
) -> None:
    """Add the head-detector options shared by all prediction scripts."""
    if allow_manual_box:
        parser.add_argument(
            "--head-box",
            type=float,
            nargs=4,
            metavar=("X1", "Y1", "X2", "Y2"),
            help="Head xyxy box in original-image pixels; bypasses YOLO.",
        )
    parser.add_argument("--head-detector", default="../weights/yolo_head_best.pt")
    parser.add_argument("--head-img-size", type=int, default=640)
    parser.add_argument("--head-conf", type=float, default=0.25)
    parser.add_argument("--head-iou", type=float, default=0.45)
    parser.add_argument("--head-max-det", type=int, default=20)
    parser.add_argument(
        "--detector-device",
        default=None,
        help="Head-detector device, e.g. cuda:0 or cpu. Defaults to --device.",
    )


def add_gaze_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the CoSi/model options shared by all prediction scripts."""
    parser.add_argument(
        "--pretrained",
        default="../weights/cosi_weights.pth",
        help="Path to the pretrained CoSi checkpoint.",
    )
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--stage", default="eval_dyadic")
    parser.add_argument("--integration", default="confidence_coordinated")
    parser.add_argument(
        "--device", default=None, help="Gaze-model device, e.g. cuda:0 or cpu."
    )


def resolve_devices(args) -> Tuple[str, str]:
    gaze_device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    return gaze_device, args.detector_device or gaze_device


def build_gaze_model(args, device):
    """Compose the Hydra config and construct the requested gaze model."""
    overrides = [
        f"stage={args.stage}",
        "model=cosi",
        f"pretrained_weights={args.pretrained}",
    ]
    overrides.extend(
        [f"model.integration={args.integration}", "data.transform.input_resolution=448"]
    )
    if args.device is not None:
        overrides.append(f"device={args.device}")

    with initialize_config_dir(
        config_dir=os.path.abspath(args.config_dir), version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)
    return build_model(device, cfg, verbose=False), cfg


def build_head_detector(args, device):
    return load_head_detector(weights=args.head_detector, device=device)


def detect_heads_from_args(detector, image, args):
    """Run detection while consistently honoring every detector CLI option."""
    return detect_heads(
        detector,
        image,
        conf=args.head_conf,
        iou=args.head_iou,
        max_det=args.head_max_det,
        imgsz=args.head_img_size,
    )


def write_json(path, value) -> None:
    with open(path, "w") as output_file:
        json.dump(value, output_file, indent=2)


FrameHandler = Callable[[str, object, Tuple[int, int], int, str], Optional[Tuple]]


def process_video(video_path, output_dir, frame_handler: FrameHandler):
    """Run a path-based inference callback for each frame and write two videos.

    The callback returns ``(heatmap_frame, gaze_frame, json_record)``. Returning
    ``None`` preserves timing by writing the unmodified input frame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions: {width}x{height}")
    if fps <= 0:
        print("Warning: input FPS is unavailable; using 30 FPS for outputs.")
        fps = 30.0

    paths = (output_dir / "heatmaps.mp4", output_dir / "gaze_points.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = tuple(cv2.VideoWriter(str(p), fourcc, fps, (width, height)) for p in paths)
    if not all(writer.isOpened() for writer in writers):
        cap.release()
        for writer in writers:
            writer.release()
        raise RuntimeError("Could not initialize one or both output video writers.")

    predictions = []
    frame_count = 0
    skipped_count = 0
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = os.path.join(temp_dir, "current_frame.jpg")
            with tqdm(total=total or None, desc="Processing Video Frames", unit="frame") as bar:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    for generated in Path(temp_dir).iterdir():
                        if str(generated) != frame_path:
                            generated.unlink()
                    if not cv2.imwrite(frame_path, frame):
                        raise RuntimeError(f"Could not write temporary frame {frame_count}.")

                    result = frame_handler(
                        frame_path, frame, (width, height), frame_count, temp_dir
                    )
                    if result is None:
                        heatmap_frame = gaze_frame = frame
                        skipped_count += 1
                    else:
                        heatmap_frame, gaze_frame, record = result
                        predictions.append(record)

                    for writer, rendered in zip(writers, (heatmap_frame, gaze_frame)):
                        if rendered is None:
                            rendered = frame
                        if rendered.shape[:2] != (height, width):
                            rendered = cv2.resize(rendered, (width, height))
                        writer.write(rendered)
                    frame_count += 1
                    bar.update(1)
    finally:
        cap.release()
        for writer in writers:
            writer.release()

    return predictions, frame_count, skipped_count, paths
