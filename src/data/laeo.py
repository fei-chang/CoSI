
from . import augmentation

from typing import Callable, Optional
import os
import numpy as np

from .base_dataset import BinaryGaze_Base

class LAEO(BinaryGaze_Base):
    def __init__(
        self,
        frame_dir: str,
        annotation_csv: str,
        transform: Callable,
        input_size: int,
        output_size: int,
        is_train: bool = True,
        load_gaze_heatmap: bool = False,
        *,
        bbox_jitter: float = 0.5,
        rand_crop: float = 0.5,
        rand_flip: float = 0.5,
        color_jitter: float = 0.5,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0
    ):
        super().__init__(frame_dir, annotation_csv, 
                         transform, input_size, output_size, 
                         is_train)

    def _get_img_and_boxes(self, row):
        """Extracts image path and bounding boxes from the CSV row."""
        row = list(row)
        vid_id, frame_num, *boxes, label = row
        img_path = os.path.join(self.frame_dir, f"{vid_id}/{int(frame_num):06d}.jpg")

        # Extract and reshape bounding boxes
        p1_box = np.array(boxes[0:4]).astype(float)
        p2_box = np.array(boxes[4:8]).astype(float)
        raw_hboxes = [p1_box, p2_box]
        label = [label, label]

        return img_path, raw_hboxes, label