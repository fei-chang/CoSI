import os
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import pandas as pd
import numpy as np

from .masking import MaskGenerator
from . import data_utils
from . import augmentation


class SocialGaze_Base(Dataset):
    def __init__(
        self,
        frame_dir: str,
        annotation_csv: str,
        transform: Callable,
        input_size: int,
        output_size: int,
        is_train: bool = True,
        mask_generator: Optional[MaskGenerator] = None,
        bbox_jitter: float = 0.5,
        rand_crop: float = 0.5,
        rand_flip: float = 0.5,
        color_jitter: float = 0.5,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0,
        debug: bool = False
    ):
        super().__init__()
        self.frame_dir = frame_dir
        self.annotation_csv = annotation_csv
        self.transform = transform
        self.input_size = input_size
        self.output_size = output_size
        self.is_train = is_train

        # Load annotation data
        self.df = self._load_data(annotation_csv)
        self.length = len(self.df)
        self.debug = debug

        self.augment = None
        if self.is_train:
            ## data augmentation
            self.augment = augmentation.AugmentationList(
                [
                    augmentation.ColorJitter(color_jitter),
                    augmentation.BoxJitter(bbox_jitter),
                    augmentation.RandomCrop(rand_crop),
                    augmentation.RandomFlip(rand_flip),
                    augmentation.RandomRotate(rand_rotate),
                    augmentation.RandomLSJ(rand_lsj),
                ]
            )

            self.mask_generator = mask_generator

    def _get_img_and_boxes(self, row):
        raise NotImplementedError
    def _load_data(self, annotation_csv):
        raise NotImplementedError
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path, raw_hboxes, label = self._get_img_and_boxes(row)

        # Load image
        raw_img = Image.open(img_path).convert("RGB")
        width, height = raw_img.size
    
        # Normalize bounding boxes
        scale = np.array([width, height, width, height])
        normalized_hboxes = [np.array(box) / scale for box in raw_hboxes]
        
        # Dummy placeholders for gaze and pattern (since LAEO does not include these)
        normalized_gazes = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]

        # Augmentation
        if self.is_train:
            principal_idx = np.random.randint(0, 2)

            # Augmentations (image, bounding boxes, and gaze heatmaps)
            raw_img, raw_hboxes, normalized_gazes, (width, height), _ = self.augment(
                raw_img,
                raw_hboxes,
                normalized_gazes,
                (width, height),
                None  # No gaze heatmaps in LAEO
            )

            scale = np.array([width, height, width, height])
            normalized_hboxes = [np.array(box) / scale for box in raw_hboxes]

        else:
            principal_idx = 0

        associate_idx = principal_idx ^ 1  

        head_channels = []
        raw_head_crops = []
        for hbox in raw_hboxes:
            xmin, ymin, xmax, ymax = hbox            
            
            # Head Image
            crop = raw_img.crop((xmin, ymin, xmax, ymax))
            raw_head_crops.append(crop)


            # Head Position Map (Head Channel)
            head_channel = data_utils.get_head_box_channel(
                xmin, ymin, xmax, ymax,
                width, height,
                resolution=self.input_size,
                coordconv=False
            ).unsqueeze(0)
            
            head_channels.append(head_channel)

        # Apply transformations
        head_crops = []
        if self.transform is not None:
            img = self.transform(raw_img)
            for h in raw_head_crops:
                head_crops.append(self.transform(h))

        # Create output dictionary
        out_dict = {
            "path": img_path,
            "images": img,
            "principal": {
                "head_crop": head_crops[principal_idx],
                "head_channel": head_channels[principal_idx],
                "head_box": torch.FloatTensor(normalized_hboxes[principal_idx]),
                "pattern": torch.FloatTensor([float(label[principal_idx])]),
            },
            "associate": {
                "head_crop": head_crops[associate_idx],
                "head_channel": head_channels[associate_idx],
                "head_box": torch.FloatTensor(normalized_hboxes[associate_idx]),
                "pattern": torch.FloatTensor([float(label[associate_idx])]),
            },
            "imsize": torch.IntTensor([width, height])
        }

        return out_dict
    
    def __len__(self) -> int:
        return min(500, self.length) if self.debug else self.length



class GazeFollow_SocialGaze(Dataset):
    def __init__(
        self,
        frame_dir: str,
        annotation_csv: str,
        transform: Callable, 
        input_size: int,
        output_size: int,
        quant_labelmap: bool = True,
        is_train: bool = True,
        mask_generator: Optional[MaskGenerator] = None,
        head_heatmaps: Optional[str] = None,
        bbox_jitter: float = 0.5,
        rand_crop: float = 0.5,
        rand_flip: float = 0.5,
        color_jitter: float = 0.5,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()
        self.frame_dir = frame_dir
        self.transform = transform
        self.input_size = input_size
        self.output_size = output_size
        self.is_train = is_train

        # Load Annotations
        self.data, self.keys = self._load_data(annotation_csv)
        self.length = len(self.keys)

        # Check if head_heatmaps and path is valid
        if head_heatmaps and os.path.exists(head_heatmaps):
            self.head_heatmaps = head_heatmaps
            self.using_head_heatmaps = True
        else:
            self.head_heatmaps = None
            self.using_head_heatmaps=False

        self.draw_labelmap = (
            data_utils.draw_labelmap if quant_labelmap else data_utils.draw_labelmap_no_quant
        )

        if self.is_train:
            ## data augmentation
            self.augment = augmentation.AugmentationList(
                [
                    augmentation.ColorJitter(color_jitter),
                    augmentation.BoxJitter(bbox_jitter),
                    augmentation.RandomCrop(rand_crop),
                    augmentation.RandomFlip(rand_flip),
                    augmentation.RandomRotate(rand_rotate),
                    augmentation.RandomLSJ(rand_lsj),
                ]
            )

            self.mask_generator = mask_generator
        else:
            self.augment = None

        self.debug = debug

    def _load_data(self, annotation_csv):
        """
        Load data
        """
        raise NotImplementedError

    def _get_info(self, index):
        """
        returns the raw_image, 
        the height, 
        width info, 
        the headboxes, 
        gazes, (Optional)
        inout_ls, (Optional)
        pattern_ls
        of the people in the image
        """
        raise NotImplementedError

    def __getitem__(self, index):

        info = self._get_info(index)
        info['head_channels'] = []
        info['gaze_heatmaps'] = []
        
        # Load head heatmaps only if they exist
        if self.using_head_heatmaps:
            heatmap_path = os.path.join(self.head_heatmaps, info["path"])
            info['head_heatmaps'] = Image.open(heatmap_path)
        else:
            info['head_heatmaps'] = None

        has_gaze = ("normalized_gazes" in info) and (info["normalized_gazes"] is not None)
        has_inout = ("inout_ls" in info) and (info["inout_ls"] is not None)


        if self.is_train and self.augment is not None:
            gazes = info["normalized_gazes"] if has_gaze else [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
            raw_img, raw_hboxes, normalized_gazes, size, head_heatmaps = self.augment(
                info['raw_img'],
                info['raw_hboxes'],
                gazes,
                (info['width'], info['height']),
                info['head_heatmaps']
            )

            info['head_heatmaps'] = head_heatmaps
            info['raw_img'] = raw_img
            info['raw_hboxes'] = raw_hboxes
            info['width'], info['height'] = size
            if has_gaze:
                info["normalized_gazes"] = normalized_gazes


        # Normalize head boxes (pixel -> 0..1)
        scale = np.array([info['width'], info['height'], info['width'], info['height']])
        info['normalized_hboxes'] = [raw_hbox / scale for raw_hbox in info['raw_hboxes']]
        
        # Normalize gaze vectors
        if has_gaze:    
            info['normalized_gaze_vectors'] = data_utils.compute_batch_gaze_vectors(
                info['normalized_hboxes'], info['normalized_gazes'],
            )


        n_persons = len(info["raw_hboxes"])
        if n_persons < 2:
            raise RuntimeError(f"Less than 2 persons found for index={index}, path={info.get('path','?')}")
        
        principal_idx = np.random.randint(0, n_persons) if (self.is_train) else 0
        associate_idx = (principal_idx + 1) % n_persons


        raw_head_crops, gaze_heatmaps = [], []
        for person_idx, (xmin, ymin, xmax, ymax) in enumerate(info["raw_hboxes"]):
            # Head Image
            crop = info["raw_img"].crop((xmin, ymin, xmax, ymax))
            raw_head_crops.append(crop)

            # Head Position Map (Head Channel)
            head_channel = data_utils.get_head_box_channel(
                xmin, ymin, xmax, ymax,
                info['width'], info['height'],
                resolution=self.input_size, coordconv=False,
            ).unsqueeze(0)
            info['head_channels'].append(head_channel)

            # Gaze Heatmap
            if has_gaze:
                hm = torch.zeros(self.output_size, self.output_size)
                if has_inout:
                    inout = (info['inout_ls'][person_idx])
                else:
                    inout = 1 
                if has_inout & inout:
                    hm = self.draw_labelmap(
                        hm,
                        [info['normalized_gazes'][person_idx][0] * self.output_size,
                        info['normalized_gazes'][person_idx][1] * self.output_size],
                        3,
                        type="Gaussian",
                    )
                gaze_heatmaps.append(hm)



        # Mask generation
        if self.is_train and self.mask_generator is not None:
            image_mask = self.mask_generator(
                info['normalized_hboxes'],
                torch.stack(info['head_channels']).sum(0)
            )


        head_heatmaps = None
        if self.using_head_heatmaps and info.get('head_heatmaps'):
            head_heatmaps = TF.to_tensor(
                TF.resize(info['head_heatmaps'], (self.input_size, self.input_size))
            )

        info['head_crop']=[]
        if self.transform is not None:
            img = self.transform(info['raw_img'])
            for h in raw_head_crops:
                info['head_crop'].append(self.transform(h))
        else:
            img = info['raw_img']
            for h in raw_head_crops:
                info['head_crop'].append(h)
        
        out_dict = {
            "path": info['path'],
            "images": img,
            "principal": {
                "head_crop": info['head_crop'][principal_idx],
                "head_channel": info['head_channels'][principal_idx],
                "head_box": torch.FloatTensor(info['normalized_hboxes'][principal_idx]),
                "pattern": torch.tensor(info['pattern_ls'][principal_idx], dtype=torch.float),
            },
            "associate": {
                "head_crop": info['head_crop'][associate_idx],
                "head_channel": info['head_channels'][associate_idx],
                "head_box": torch.FloatTensor(info['normalized_hboxes'][associate_idx]),
                "pattern": torch.tensor(info['pattern_ls'][associate_idx], dtype=torch.float),
            },
            "imsize": torch.IntTensor([info['width'], info['height']])
        }


        if has_gaze:
            out_dict["principal"].update({
                "gaze_heatmap": gaze_heatmaps[principal_idx],
                "gaze": torch.FloatTensor(info["normalized_gazes"][principal_idx]),
                "gaze_vector": torch.FloatTensor(info["normalized_gaze_vectors"][principal_idx]),

            })

            out_dict["associate"].update({
                "gaze_heatmap": gaze_heatmaps[associate_idx],
                "gaze": torch.FloatTensor(info["normalized_gazes"][associate_idx]),
                "gaze_vector": torch.FloatTensor(info["normalized_gaze_vectors"][associate_idx]),
            })
        
        if has_inout:
            out_dict["principal"].update({
                "inout": torch.tensor(info['inout_ls'][principal_idx], dtype=torch.float)
            })
            out_dict["associate"].update({
                "inout": torch.tensor(info['inout_ls'][associate_idx], dtype=torch.float)
            })

        if self.is_train and self.mask_generator is not None:
            out_dict["image_masks"] = image_mask

        if self.using_head_heatmaps and head_heatmaps is not None:
            out_dict["head_heatmaps"]= head_heatmaps 
        
        return out_dict

    def __len__(self) -> int:
        return min(500, self.length) if self.debug else self.length
