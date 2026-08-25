import os
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import pandas as pd

from . import augmentation
from .masking import MaskGenerator
from . import data_utils


class GazeFollow(Dataset):

    TRAIN_COLUMNS = [
        "path", "idx", "body_bbox_x", "body_bbox_y", "body_bbox_w", "body_bbox_h",
        "eye_x", "eye_y", "gaze_x", "gaze_y", "bbox_xmin", "bbox_ymin",
        "bbox_xmax", "bbox_ymax", "inout", "meta0", "meta1"
    ]
    
    TEST_COLUMNS = [
        "path", "idx", "body_bbox_x", "body_bbox_y", "body_bbox_w", "body_bbox_h",
        "eye_x", "eye_y", "gaze_x", "gaze_y", "bbox_xmin", "bbox_ymin",
        "bbox_xmax", "bbox_ymax", "meta0", "meta1"
    ]
    
    def __init__(
        self,
        frame_dir: str,
        annotation_csv: str,
        transform: Callable,
        input_size: int,
        output_size: int,
        quant_labelmap: bool = True,
        is_train: bool = True,
        *,
        mask_generator: Optional[MaskGenerator] = None,
        head_heatmaps: Optional[str] = None,
        bbox_jitter: float = 0.5,
        rand_crop: float = 0.5,
        rand_flip: float = 0.5,
        color_jitter: float = 0.5,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0,
    ):
        """Initialize GazeFollow dataset.
        
        Args:
            frame_dir: Directory containing image frames (Reminder: the path in gazefollow starts with train/test2)
            head_heatmaps: Directory containing head heatmap images
            annotation_csv: Path to CSV containing gaze annotations
            transform: Image transform pipeline
            input_size: Input image size
            output_size: Output heatmap size
            head_map_size: Size of head feature maps
            quant_labelmap: Whether to use quantized label maps
            is_train: Whether this is training mode
            mask_generator: Optional mask generator for training
            bbox_jitter: Bounding box jitter probability
            rand_crop: Random crop probability
            rand_flip: Random flip probability
            color_jitter: Color jitter probability
            rand_rotate: Random rotation probability
            rand_lsj: Random large scale jitter probability
        """
        self.dataset_name = 'gazefollow'
        if is_train:
            column_names = self.TRAIN_COLUMNS
        else:
            column_names = self.TEST_COLUMNS

        df = pd.read_csv(
            annotation_csv,
            sep=",",
            names=column_names,
            index_col=False,
            encoding="utf-8-sig",
        )
        if is_train:
            df = df[
                df["inout"] != -1
            ]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df.reset_index(inplace=True)
            
            self.y_train = df[
                [
                    "bbox_xmin",
                    "bbox_ymin",
                    "bbox_xmax",
                    "bbox_ymax",
                    "eye_x",
                    "eye_y",
                    "gaze_x",
                    "gaze_y",
                    "inout",
                ]
            ]
            self.X_train = df["path"]
            self.length = len(df)
        else:
            df = df[
                [
                    "path",
                    "eye_x",
                    "eye_y",
                    "gaze_x",
                    "gaze_y",
                    "bbox_xmin",
                    "bbox_ymin",
                    "bbox_xmax",
                    "bbox_ymax",
                ]
            ].groupby(["path", "eye_x"])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)

        self.frame_dir = frame_dir
        self.head_heatmaps = head_heatmaps
        self.transform = transform
        self.is_train = is_train
        self.input_size = input_size
        self.output_size = output_size

        self.draw_labelmap = (
            data_utils.draw_labelmap if quant_labelmap else data_utils.draw_labelmap_no_quant
        )

        if self.is_train:
            ## data augmentation
            self.augment = augmentation.AugmentationList(
                [
                    # augmentation.ColorJitter(color_jitter),
                    augmentation.BoxJitter(bbox_jitter),
                    augmentation.RandomCrop(rand_crop),
                    augmentation.RandomFlip(rand_flip),
                    augmentation.RandomRotate(rand_rotate),
                    augmentation.RandomLSJ(rand_lsj),
                ]
            )

            self.mask_generator = mask_generator

    def __getitem__(self, index):
        if not self.is_train:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for _, row in g.iterrows():
                path = row["path"]
                xmin = row["bbox_xmin"]
                ymin = row["bbox_ymin"]
                xmax = row["bbox_xmax"]
                ymax = row["bbox_ymax"]
                gaze_x = row["gaze_x"]
                gaze_y = row["gaze_y"]
                cont_gaze.append(
                    [gaze_x, gaze_y]
                )  # all ground truth gaze are stacked up
            for _ in range(len(cont_gaze), 20):
                cont_gaze.append(
                    [-1, -1]
                )  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True  # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            (
                xmin,
                ymin,
                xmax,
                ymax,
                eye_x,
                eye_y,
                gaze_x,
                gaze_y,
                inout,
            ) = self.y_train.iloc[index]
            gaze_inside = bool(inout)

        if self.head_heatmaps:
            head_heatmaps = Image.open(os.path.join(self.head_heatmaps, path)) 
        else:
            head_heatmaps = None
        
        img = Image.open(os.path.join(self.frame_dir, path))
        img = img.convert("RGB")
        
        width, height = img.size
        
        xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        # expand face bbox a bit
        k = 0.1
        xmin = max(xmin - k * abs(xmax - xmin), 0)
        ymin = max(ymin - k * abs(ymax - ymin), 0)
        xmax = min(xmax + k * abs(xmax - xmin), width - 1)
        ymax = min(ymax + k * abs(ymax - ymin), height - 1)

        if self.is_train:
            img, bbox, gaze, size, head_heatmaps= self.augment(
                img,
                [(xmin, ymin, xmax, ymax)],
                [(gaze_x, gaze_y)],
                (width, height),
                head_heatmaps,
            )
            xmin, ymin, xmax, ymax = bbox[0]
            gaze_x, gaze_y = gaze[0]
            width, height = size

        head_crop = img.crop((xmin, ymin, xmax, ymax))
        normalized_hboxes = (xmin/width, ymin/height, xmax/width, ymax/height)
        normalized_gaze_vectors = data_utils.compute_batch_gaze_vectors(
            [(xmin/width, ymin/height, xmax/width, ymax/height)], 
            [(gaze_x, gaze_y)],
        )
        head_channel = data_utils.get_head_box_channel(
            xmin,
            ymin,
            xmax,
            ymax,
            width,
            height,
            resolution=self.input_size,
            coordconv=False,
        ).unsqueeze(0)

        if self.is_train and self.mask_generator is not None:
            image_mask = self.mask_generator(
                (xmin / width,
                ymin / height,
                xmax / width,
                ymax / height),
                head_channel,
            )
            
        if self.transform is not None:
            img = self.transform(img)
            head_crop = self.transform(head_crop)


        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(
            self.output_size, self.output_size
        )  # set the size of the output
        if not self.is_train:  # aggregated heatmap
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_heatmap += self.draw_labelmap(
                        torch.zeros(self.output_size, self.output_size),
                        [gaze_x * self.output_size, gaze_y * self.output_size],
                        3,
                        type="Gaussian",
                    )
            gaze_heatmap /= num_valid
        else:
            # if gaze_inside:
            gaze_heatmap = self.draw_labelmap(
                gaze_heatmap,
                [gaze_x * self.output_size, gaze_y * self.output_size],
                3,
                type="Gaussian",
            )
        if self.head_heatmaps:
            head_heatmaps = TF.to_tensor(
                TF.resize(head_heatmaps, (self.input_size, self.input_size))
                )
        imsize = torch.IntTensor([width, height])

        if self.is_train:
            out_dict = {
                "path": path,
                "images": img,
                "principal":{
                    "head_crop": head_crop,
                    "head_channel": head_channel,
                    "head_box": torch.FloatTensor(normalized_hboxes),
                    "gaze_heatmap": gaze_heatmap,
                    "gaze_vector": torch.FloatTensor(normalized_gaze_vectors[0]),
                    "gaze": torch.FloatTensor([gaze_x, gaze_y]),
                    "inout": torch.FloatTensor([gaze_inside]),
                },
                "imsize": imsize,
            }
            if self.mask_generator is not None:
                out_dict["image_masks"] = image_mask
            if self.head_heatmaps is not None:
                out_dict["head_heatmaps"] =  head_heatmaps

            return out_dict
        else:
            out_dict = {
                "path": path,
                "images": img,
                "principal":{
                    "head_crop": head_crop,
                    "head_channel": head_channel,
                    "gaze_heatmap": gaze_heatmap,
                    "head_box": torch.FloatTensor(normalized_hboxes),
                    "gaze_vector": torch.FloatTensor(normalized_gaze_vectors[0]),
                    "gaze": cont_gaze,
                    "inout": torch.FloatTensor([gaze_inside]),
                },
                "imsize": imsize,
            }
            if self.head_heatmaps is not None:
                out_dict["head_heatmaps"] =  head_heatmaps
                
            return out_dict
            

    def __len__(self):
        return self.length
