import os
import numpy as np
from .base_dataset import SocialGaze_Base
from .data_utils import associate_label
import pandas as pd

class LAEO(SocialGaze_Base):
    dataset_name = 'uco_laeo'
    def _load_data(self, annotation_csv):
        df = pd.read_csv(annotation_csv) 
        return df
    def _get_img_and_boxes(self, row):
        """Extracts image path and bounding boxes from the CSV row."""
        vid_id, frame_num, *boxes, label = list(row)
        img_path = os.path.join(self.frame_dir, f"{vid_id}/{int(frame_num):06d}.jpg")

        # Extract and reshape bounding boxes
        p1_box = np.array(boxes[0:4]).astype(float)
        p2_box = np.array(boxes[4:8]).astype(float)
        raw_hboxes = [p1_box, p2_box]
        label = [label, label]

        return img_path, raw_hboxes, label
    

class VideoCoAtt(SocialGaze_Base):
    dataset_name = 'video_coatt'
    def _load_data(self, annotation_csv):
        df = pd.read_csv(annotation_csv) 
        return df
    def _get_img_and_boxes(self, row):
        """Extracts image path and bounding boxes from the CSV row."""
        vid_id, frame_num, *boxes, label = list(row)
        img_path = os.path.join(self.frame_dir, f"{vid_id}/{int(frame_num):05d}_{vid_id}.jpg")

        # Extract and reshape bounding boxes
        p1_box = np.array(boxes[0:4]).astype(float)
        p2_box = np.array(boxes[4:8]).astype(float)
        raw_hboxes = [p1_box, p2_box]
        label = [label, label]

        return img_path, raw_hboxes, label
    

class GP_Static(SocialGaze_Base):
    dataset_name='gt_static'

    def _load_data(self, annotation_csv):
        df = pd.read_csv(annotation_csv) 
        df = df[['vid_id', 'frame_num', 'p1_xmin', 'p1_ymin', 'p1_xmax', 'p1_ymax', 'p2_xmin', 'p2_ymin', 'p2_xmax', 'p2_ymax', 'static_label_idx']]
        return df
    
    def _get_img_and_boxes(self, row):
        vid_id, frame_num, *boxes, label = list(row)
        
        # Load the image and add it to the info dictionary
        img_path = os.path.join(self.frame_dir, f"{vid_id}/{int(frame_num):05d}.jpg")
        
        # Extract and reshape bounding boxes
        p1_box = np.array(boxes[0:4]).astype(float)
        p2_box = np.array(boxes[4:8]).astype(float)

        raw_hboxes = [p1_box, p2_box]
        label = [label, associate_label(label)]

        return img_path, raw_hboxes, label