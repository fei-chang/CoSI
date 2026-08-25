"""Unified image/video inference and visualization entry point.

Examples
--------
Single-person gaze prediction on an image::

    python predict_and_visualize.py --mode one --source image --input photo.jpg

Dyadic social-gaze prediction on a video::

    python predict_and_visualize.py --mode two --source video --input clip.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

from inference import DyadicGazePredictor
from inference.predictor import SinglePersonGazePredictor
from inference.preprocess import normalized_box, order_two_heads, select_one_head
from inference.utils import (
    add_gaze_model_arguments,
    add_head_detector_arguments,
    build_gaze_model,
    build_head_detector,
    detect_heads_from_args,
    process_video,
    resolve_devices,
    write_json,
)
from inference.visualize import (
    save_prediction_visualizations,
    save_single_gaze_visualizations,
)


OUTPUT_DIRS = {
    ("one", "image"): "../predictions/gaze_images",
    ("one", "video"): "../predictions/gaze_videos",
    ("two", "image"): "../predictions/social_images",
    ("two", "video"): "../predictions/social_videos",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-person or dyadic gaze inference on an image or video."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("one", "two"),
        help="Use 'one' for single-person gaze or 'two' for dyadic social gaze.",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=("image", "video"),
        help="Type of input to process.",
    )
    parser.add_argument("--input", required=True, help="Path to the input image or video.")
    add_head_detector_arguments(parser, allow_manual_box=True)
    add_gaze_model_arguments(parser)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. A mode/source-specific directory is used by default.",
    )
    args = parser.parse_args()
    if args.head_box is not None and (args.mode != "one" or args.source != "image"):
        parser.error("--head-box is only supported with --mode one --source image.")
    if args.output_dir is None:
        args.output_dir = OUTPUT_DIRS[(args.mode, args.source)]
    return args


def build_predictor(args, gaze_device):
    gaze_model, cfg = build_gaze_model(args, gaze_device)
    predictor_args = {
        "input_size": cfg.data.transform.input_resolution,
        "device": gaze_device,
    }
    if args.mode == "one":
        return SinglePersonGazePredictor(
            gaze_model, person_key="principal", **predictor_args
        )
    return DyadicGazePredictor(gaze_model, **predictor_args)


def add_detection_metadata(output, detections, image_size):
    """Add detector information using the schema for the selected mode."""
    if len(detections) == 1:
        detection = detections[0]
        output.update(
            head_box_px=[float(value) for value in detection["box"]],
            head_box_norm=normalized_box(detection["box"], image_size),
            detector_confidence=(
                None
                if detection["confidence"] is None
                else float(detection["confidence"])
            ),
        )
        return

    for name, detection in zip(("principal", "associate"), detections):
        output[name]["bbox_norm"] = normalized_box(detection["box"], image_size)
        output[name]["detector_confidence"] = float(detection["confidence"])


def choose_detections(args, detector, input_path):
    if args.head_box is not None:
        return [{"box": list(map(float, args.head_box)), "confidence": None}]
    detections = detect_heads_from_args(detector, input_path, args)
    if args.mode == "one":
        return [select_one_head(detections)]
    return list(order_two_heads(detections))


def predict(args, predictor, input_path, detections):
    head_boxes = [detection["box"] for detection in detections]
    if args.mode == "one":
        return predictor.predict(input_path, head_boxes[0]), head_boxes
    return predictor.predict(input_path, head_boxes), head_boxes


def save_visualizations(args, input_path, head_boxes, prediction, output_dir):
    if args.mode == "one":
        save_single_gaze_visualizations(
            input_path, head_boxes[0], prediction, output_dir
        )
    else:
        save_prediction_visualizations(input_path, head_boxes, prediction, output_dir)


def process_image(args, detector, predictor):
    print("[2/3] Predicting gaze...")
    detections = choose_detections(args, detector, args.input)
    prediction, head_boxes = predict(args, predictor, args.input, detections)

    print("[3/3] Saving outputs...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.input) as image:
        image_size = image.size

    output = prediction.to_dict(include_heatmap=False) if args.mode == "one" else prediction.to_dict(include_heatmaps=False)
    output["image_path"] = args.input
    add_detection_metadata(output, detections, image_size)
    if args.mode == "one":
        output["head_source"] = "manual" if args.head_box is not None else "detector"

    write_json(output_dir / "prediction.json", output)
    save_visualizations(args, args.input, head_boxes, prediction, output_dir)
    print(json.dumps(output, indent=2))
    print(f"Saved outputs to: {output_dir.resolve()}")


def process_video_source(args, detector, predictor):
    def predict_frame(frame_path, _frame, image_size, frame_id, temp_dir):
        try:
            detections = choose_detections(args, detector, frame_path)
        except RuntimeError:
            return None

        prediction, head_boxes = predict(args, predictor, frame_path, detections)
        record = (
            prediction.to_dict(include_heatmap=False)
            if args.mode == "one"
            else prediction.to_dict(include_heatmaps=False)
        )
        record.update(video_path=args.input, frame_id=frame_id)
        add_detection_metadata(record, detections, image_size)
        save_visualizations(args, frame_path, head_boxes, prediction, temp_dir)

        heatmap_name = "heatmap.png" if args.mode == "one" else "heatmaps.png"
        gaze_name = "gaze_point.png" if args.mode == "one" else "gaze_points.png"
        return (
            cv2.imread(str(Path(temp_dir) / heatmap_name)),
            cv2.imread(str(Path(temp_dir) / gaze_name)),
            record,
        )

    print(f"[2/3] Processing video: {args.input}...")
    predictions, frame_count, skipped, video_paths = process_video(
        args.input, args.output_dir, predict_frame
    )
    print("[3/3] Saving outputs...")
    json_path = video_paths[0].parent / "prediction.json"
    write_json(json_path, predictions)
    print(f"Processed frames: {frame_count}")
    print(f"Frames with predictions: {len(predictions)}")
    print(f"Frames skipped due to missing heads: {skipped}")
    print(f"Saved outputs to: {json_path.parent.resolve()}")


def main():
    args = parse_args()
    gaze_device, detector_device = resolve_devices(args)
    print("[1/3] Loading models...")

    detector = None
    if args.head_box is None:
        detector = build_head_detector(args, detector_device)
    predictor = build_predictor(args, gaze_device)

    if args.source == "image":
        process_image(args, detector, predictor)
    else:
        process_video_source(args, detector, predictor)


if __name__ == "__main__":
    main()
