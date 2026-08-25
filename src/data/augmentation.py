import math
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF

"""
Some functions in this script are adapted from ViTGaze (https://github.com/hustvl/ViTGaze),
with modifications made for this project. 
We gratefully acknowledge the original authors and their contributions.
"""

class Augmentation:
    def __init__(self, p: float) -> None:
        self.p = p

    def transform(
        self,
        image: Image,
        bbox_ls: List[Tuple],
        gaze_ls: List[Tuple],
        size: Tuple[int],
        head_heatmap = None
    ):
        raise NotImplementedError

    def __call__(
        self,
        image: Image,
        bbox_ls: List[Tuple],
        gaze_ls: List[Tuple],
        size: Tuple[int],
        head_heatmap = None,
    ):
        if np.random.random_sample() < self.p:
            return self.transform(image, bbox_ls, gaze_ls, size, head_heatmap)
        return image, bbox_ls, gaze_ls, size, head_heatmap


class AugmentationList:
    def __init__(self, augmentations: List[Augmentation]) -> None:
        self.augmentations = augmentations

    def __call__(
        self,
        image: Image,
        bbox_ls: List[Tuple],
        gaze_ls: List[Tuple],
        size: Tuple[int],
        head_heatmap=None
    ):
        for aug in self.augmentations:
            image, bbox_ls, gaze_ls, size, head_heatmap = aug(image, bbox_ls, gaze_ls, size, head_heatmap)
        return image, bbox_ls, gaze_ls, size, head_heatmap


class BoxJitter(Augmentation):
    # Jitter (expansion-only) bounding box size
    def __init__(self, p: float, expansion: float = 0.2) -> None:
        super().__init__(p)
        self.expansion = expansion

    def transform(
        self,
        image: Image,
        bbox_ls: List[Tuple],
        gaze_ls: List[Tuple],
        size: Tuple[int],
        head_heatmap=None
    ):
        width, height = size
        augmented_bbox_ls = []
        for idx in range(len(bbox_ls)):
            xmin, ymin, xmax, ymax = bbox_ls[idx]

            k = np.random.random_sample() * self.expansion
            xmin = np.clip(xmin - k * abs(xmax - xmin), 0, width - 1)
            ymin = np.clip(ymin - k * abs(ymax - ymin), 0, height - 1)
            xmax = np.clip(xmax + k * abs(xmax - xmin), 0, width - 1)
            ymax = np.clip(ymax + k * abs(ymax - ymin), 0, height - 1)
            augmented_bbox_ls.append((xmin, ymin, xmax, ymax))
        
        return image, augmented_bbox_ls, gaze_ls, size, head_heatmap


class RandomCrop(Augmentation):
    def __init__(self, p: float) -> None:
        super().__init__(p)

    def transform(
        self,
        image: Image,
        bbox_ls: List[Tuple],
        gaze_ls: List[Tuple],
        size: Tuple[int],
        head_heatmap=None,
    ):
        width, height = size
        crop_xmins = []
        crop_ymins = []
        crop_xmaxs = []
        crop_ymaxs = []

        for idx in range(len(bbox_ls)):
            xmin, ymin, xmax, ymax = bbox_ls[idx]
            gaze_x, gaze_y = gaze_ls[idx]

            # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
            crop_xmins.append(np.min([gaze_x * width, xmin, xmax]))
            crop_ymins.append(np.min([gaze_y * height, ymin, ymax]))
            crop_xmaxs.append(np.max([gaze_x * width, xmin, xmax]))
            crop_ymaxs.append(np.max([gaze_y * height, ymin, ymax]))

        crop_xmin = np.min(crop_xmins)
        crop_ymin = np.min(crop_ymins)
        crop_xmax = np.max(crop_xmaxs)
        crop_ymax = np.max(crop_ymaxs)

        # Randomly select a random top left corner
        crop_xmin = np.random.uniform(0, crop_xmin)
        crop_ymin = np.random.uniform(0, crop_ymin)

        # Find the range of valid crop width and height starting from the (crop_xmin, crop_ymin)
        crop_width_min = crop_xmax - crop_xmin
        crop_height_min = crop_ymax - crop_ymin
        crop_width_max = width - crop_xmin
        crop_height_max = height - crop_ymin

        # Randomly select a width and a height
        crop_width = np.random.uniform(crop_width_min, crop_width_max)
        crop_height = np.random.uniform(crop_height_min, crop_height_max)

        # Round to integers
        crop_ymin, crop_xmin, crop_height, crop_width = map(
            int, map(round, (crop_ymin, crop_xmin, crop_height, crop_width))
        )

        # Crop it
        image = TF.crop(image, crop_ymin, crop_xmin, crop_height, crop_width)
        if head_heatmap:
            head_heatmap = TF.crop(head_heatmap, crop_ymin, crop_xmin, crop_height, crop_width)

        augmented_bbox = []
        augmented_gaze = []
        # convert coordinates into the cropped frame
        for idx in range(len(bbox_ls)):
            xmin, ymin, xmax, ymax = bbox_ls[idx]
            gaze_x, gaze_y = gaze_ls[idx]
            xmin, ymin, xmax, ymax = (
                xmin - crop_xmin,
                ymin - crop_ymin,
                xmax - crop_xmin,
                ymax - crop_ymin,
            )
            augmented_bbox.append((xmin, ymin, xmax, ymax))
            gaze_x = (gaze_x * width - crop_xmin) / float(crop_width)
            gaze_y = (gaze_y * height - crop_ymin) / float(crop_height)
            augmented_gaze.append((gaze_x, gaze_y))
        return (
            image,
            augmented_bbox,
            augmented_gaze,
            (crop_width, crop_height),
            head_heatmap
        )


class RandomFlip(Augmentation):
    def __init__(self, p: float) -> None:
        super().__init__(p)

    def transform(
        self,
        image: Image,
        bbox_ls: List[Tuple],
        gaze_ls: List[Tuple],
        size: Tuple[int],
        head_heatmap=None
    ):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if head_heatmap:
            head_heatmap = head_heatmap.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_bbox = []
        augmented_gaze = []
        for idx in range(len(bbox_ls)):
            xmin, ymin, xmax, ymax = bbox_ls[idx]
            xmin, xmax = size[0] - xmax, size[0] - xmin
            augmented_bbox.append((xmin, ymin, xmax, ymax))

            gaze_x, gaze_y = 1 - gaze_ls[idx][0], gaze_ls[idx][1]
            augmented_gaze.append((gaze_x, gaze_y))   
        return image, augmented_bbox, augmented_gaze, size, head_heatmap


class RandomRotate(Augmentation):
    def __init__(
        self, p: float, max_angle: int = 20, resample: int = Image.BILINEAR
    ) -> None:
        super().__init__(p)
        self.max_angle = max_angle
        self.resample = resample

    def _random_rotation_matrix(self):
        angle = (2 * np.random.random_sample() - 1) * self.max_angle
        angle = -math.radians(angle)
        return [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

    @staticmethod
    def _transform(x, y, matrix):
        return (
            matrix[0] * x + matrix[1] * y + matrix[2],
            matrix[3] * x + matrix[4] * y + matrix[5],
        )

    @staticmethod
    def _inv_transform(x, y, matrix):
        x, y = x - matrix[2], y - matrix[5]
        return matrix[0] * x + matrix[3] * y, matrix[1] * x + matrix[4] * y

    def transform(
        self,
        image: Image,
        bbox_ls: List[Tuple],
        gaze_ls: List[Tuple],
        size: Tuple[int],
        head_heatmap=None
    ):
        width, height = size
        rot_mat = self._random_rotation_matrix()
        # Calculate offsets
        rot_center = (width / 2.0, height / 2.0)
        rot_mat[2], rot_mat[5] = self._transform(
            -rot_center[0], -rot_center[1], rot_mat
        )
        rot_mat[2] += rot_center[0]
        rot_mat[5] += rot_center[1]
        xx = []
        yy = []
        for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
            x, y = self._transform(x, y, rot_mat)
            xx.append(x)
            yy.append(y)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))
        rot_mat[2], rot_mat[5] = self._transform(
            -(nw - width) / 2.0, -(nh - height) / 2.0, rot_mat
        )
        
        image = image.transform((nw, nh), Image.AFFINE, rot_mat, self.resample)

        if head_heatmap:
            head_heatmap = head_heatmap.transform((nw, nh), Image.AFFINE, rot_mat, self.resample)
        
        augmented_bbox = []
        augmented_gaze = []
        for idx in range(len(bbox_ls)):
            xmin, ymin, xmax, ymax = bbox_ls[idx]
            gaze_x, gaze_y = gaze_ls[idx]

            xx = []
            yy = []
            for x, y in (
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymin),
                (xmax, ymax),
            ):
                x, y = self._inv_transform(x, y, rot_mat)
                xx.append(x)
                yy.append(y)
            xmax, xmin = min(max(xx), nw), max(min(xx), 0)
            ymax, ymin = min(max(yy), nh), max(min(yy), 0)
            augmented_bbox.append((xmin, ymin, xmax, ymax))
            
            gaze_x, gaze_y = self._inv_transform(gaze_x * width, gaze_y * height, rot_mat)
            gaze_x = max(min(gaze_x / nw, 1), 0)
            gaze_y = max(min(gaze_y / nh, 1), 0)
            augmented_gaze.append((gaze_x, gaze_y))

        return (
            image,
            augmented_bbox,
            augmented_gaze,
            (nw, nh),
            head_heatmap
        )


class ColorJitter(Augmentation):
    def __init__(
        self,
        p: float,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
    ) -> None:
        super().__init__(p)
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def transform(
        self,
        image: Image,
        bbox_ls: List[Tuple],
        gaze_ls: List[Tuple],
        size: Tuple[int],
        head_heatmap=None
    ):
        return self.color_jitter(image), bbox_ls, gaze_ls, size, head_heatmap


class RandomLSJ(Augmentation):
    def __init__(self, p: float, min_scale: float = 0.1) -> None:
        super().__init__(p)
        self.min_scale = min_scale

    def transform(
        self,
        image: Image,
        bbox_ls: List[Tuple],
        gaze_ls: List[Tuple],
        size: Tuple[int],
        head_heatmap = None
    ):

        width, height = size
        scale = self.min_scale + np.random.random_sample() * (1 - self.min_scale)
        # assert scale > 0
        # assert width > 0
        # assert height > 0 
        # random.
        nh, nw = int(height * scale), int(width * scale)
        image = TF.resize(image, (nh, nw))
        image = ImageOps.expand(image, (0, 0, width - nw, height - nh))
        if head_heatmap:
            head_heatmap = TF.resize(head_heatmap, (nh, nw))
            head_heatmap = ImageOps.expand(head_heatmap, (0, 0, width - nw, height - nh))
            
        augmented_bbox = []
        augmented_gaze = []
        for idx in range(len(bbox_ls)):
            xmin, ymin, xmax, ymax = bbox_ls[idx]
            gaze_x, gaze_y = gaze_ls[idx]

            xmin, ymin, xmax, ymax = (
                xmin * scale,
                ymin * scale,
                xmax * scale,
                ymax * scale,
            )
            gaze_x, gaze_y = gaze_x * scale, gaze_y * scale
            augmented_bbox.append((xmin, ymin, xmax, ymax))
            augmented_gaze.append((gaze_x, gaze_y))
        return image, augmented_bbox, augmented_gaze, (nw, nh), head_heatmap
