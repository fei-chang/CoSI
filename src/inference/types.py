from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


@dataclass
class PersonPrediction:
    """Prediction for one member of the dyad."""

    gaze_point_px: Point
    gaze_point_norm: Point
    heatmap: np.ndarray
    social_gaze_id: int
    social_gaze_label: str

    def to_dict(self, include_heatmap: bool = False) -> Dict:
        data = {
            "gaze_point_px": [float(self.gaze_point_px[0]), float(self.gaze_point_px[1])],
            "gaze_point_norm": [float(self.gaze_point_norm[0]), float(self.gaze_point_norm[1])],
            "social_gaze_id": int(self.social_gaze_id),
            "social_gaze_label": self.social_gaze_label,
            "heatmap_shape": list(self.heatmap.shape),
        }
        if include_heatmap:
            data["heatmap"] = self.heatmap.tolist()
        return data


@dataclass
class DyadicPrediction:
    principal: PersonPrediction
    associate: PersonPrediction
    image_size: Tuple[int, int]  # (width, height)

    def to_dict(self, include_heatmaps: bool = False) -> Dict:
        return {
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "principal": self.principal.to_dict(include_heatmap=include_heatmaps),
            "associate": self.associate.to_dict(include_heatmap=include_heatmaps),
        }

@dataclass
class SinglePersonGazePrediction:
    """Gaze-follow prediction for one person, with no social-gaze label."""

    gaze_point_px: Point
    gaze_point_norm: Point
    heatmap: np.ndarray
    image_size: Tuple[int, int]  # (width, height)

    def to_dict(self, include_heatmap: bool = False) -> Dict:
        data = {
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "gaze_point_px": [float(self.gaze_point_px[0]), float(self.gaze_point_px[1])],
            "gaze_point_norm": [float(self.gaze_point_norm[0]), float(self.gaze_point_norm[1])],
            "heatmap_shape": list(self.heatmap.shape),
        }
        if include_heatmap:
            data["heatmap"] = self.heatmap.tolist()
        return data

