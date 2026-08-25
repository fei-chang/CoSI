from typing import Sequence, Tuple, Union

import torch
from torch import nn

from .postprocess import (
    SOCIAL_GAZE_LABELS,
    class_name,
    extract_class_id,
    heatmap_argmax,
    point_to_normalized,
    resize_heatmap,
    tensor_heatmap_to_numpy,
)
from .preprocess import ImageInput, prepare_dyadic_batch, prepare_single_person_batch
from .types import DyadicPrediction, PersonPrediction, SinglePersonGazePrediction


class DyadicGazePredictor:
    """
    Public deployment API.

    Contract:
        image + two xyxy head boxes
        -> two gaze points + two heatmaps + two social-gaze labels

    The wrapped model is expected to follow the existing BaseGazeModel inference format:
        {
          "principal": {
              "pred_heatmap": Tensor[B,H,W],
              "pred_patterns": Tensor[B] or logits
          },
          "associate": {...}
        }
    """

    PERSONS = ("principal", "associate")

    def __init__(
        self,
        model: nn.Module,
        *,
        input_size: Union[int, Tuple[int, int]] = 448,
        device: Union[str, torch.device, None] = None,
        social_gaze_labels: Sequence[str] = SOCIAL_GAZE_LABELS,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.labels = tuple(social_gaze_labels)

        self.model = model.to(self.device)
        self.model.eval()

    def _move_batch(self, batch):
        batch["images"] = batch["images"].to(self.device)
        for person in self.PERSONS:
            batch[person]["head_channel"] = batch[person]["head_channel"].to(self.device)
        return batch

    @torch.inference_mode()
    def predict(
        self,
        image: ImageInput,
        head_boxes: Sequence[Sequence[float]],
    ) -> DyadicPrediction:
        batch, pil_image, _ = prepare_dyadic_batch(
            image=image,
            head_boxes=head_boxes,
            input_size=self.input_size,
        )
        batch = self._move_batch(batch)

        output = self.model(batch)
        image_size = pil_image.size  # (width, height)

        person_results = {}
        for person in self.PERSONS:
            if person not in output:
                raise KeyError(f"Model output is missing '{person}'.")
            person_output = output[person]

            heatmap_tensor = person_output.get("pred_heatmap")
            if heatmap_tensor is None:
                raise KeyError(f"Model output for '{person}' has no pred_heatmap.")

            pattern_output = person_output.get("pred_patterns")
            if pattern_output is None:
                # Useful if a cleaned model returns raw logits instead.
                pattern_output = person_output.get("pred_pattern")
            if pattern_output is None:
                raise KeyError(f"Model output for '{person}' has no social-gaze prediction.")

            # Take the one sample from the singleton inference batch.
            if heatmap_tensor.ndim == 3:
                heatmap_tensor = heatmap_tensor[0]
            elif heatmap_tensor.ndim == 4:
                heatmap_tensor = heatmap_tensor[0, 0]

            model_heatmap = tensor_heatmap_to_numpy(heatmap_tensor)
            image_heatmap = resize_heatmap(model_heatmap, image_size)
            gaze_px = heatmap_argmax(image_heatmap)
            gaze_norm = point_to_normalized(gaze_px, image_size)

            class_id = extract_class_id(pattern_output)

            person_results[person] = PersonPrediction(
                gaze_point_px=(float(gaze_px[0]), float(gaze_px[1])),
                gaze_point_norm=gaze_norm,
                heatmap=image_heatmap,
                social_gaze_id=class_id,
                social_gaze_label=class_name(class_id, self.labels),
            )

        return DyadicPrediction(
            principal=person_results["principal"],
            associate=person_results["associate"],
            image_size=image_size,
        )

class SinglePersonGazePredictor:
    """
    Gaze-follow inference API for one person.

    Contract:
        image + one xyxy head box -> gaze point + heatmap

    This intentionally bypasses the dyadic social-gaze branch. The model must
    provide ``forward_gaze(batch, person=...)``; CoSi is extended with that
    method in this update.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        input_size: Union[int, Tuple[int, int]] = 448,
        device: Union[str, torch.device, None] = None,
        person_key: str = "principal",
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        if person_key not in ("principal", "associate"):
            raise ValueError(
                f"person_key must be 'principal' or 'associate', got {person_key!r}."
            )
        self.person_key = person_key

        self.model = model.to(self.device)
        self.model.eval()

    def _move_batch(self, batch):
        batch["images"] = batch["images"].to(self.device)
        batch[self.person_key]["head_channel"] = batch[self.person_key][
            "head_channel"
        ].to(self.device)
        return batch

    @torch.inference_mode()
    def predict(
        self,
        image: ImageInput,
        head_box: Sequence[float],
    ) -> SinglePersonGazePrediction:
        batch, pil_image, _ = prepare_single_person_batch(
            image=image,
            head_box=head_box,
            input_size=self.input_size,
            person_key=self.person_key,
        )
        batch = self._move_batch(batch)

        forward_gaze = getattr(self.model, "forward_gaze", None)
        if forward_gaze is None:
            raise AttributeError(
                "The gaze model has no forward_gaze() method."
            )

        person_output = forward_gaze(batch, person=self.person_key)
        heatmap_tensor = person_output.get("pred_heatmap")
        if heatmap_tensor is None:
            raise KeyError("Gaze-only model output has no 'pred_heatmap'.")

        # Take the singleton inference sample.
        if heatmap_tensor.ndim == 3:
            heatmap_tensor = heatmap_tensor[0]
        elif heatmap_tensor.ndim == 4:
            heatmap_tensor = heatmap_tensor[0, 0]

        image_size = pil_image.size  # (width, height)
        model_heatmap = tensor_heatmap_to_numpy(heatmap_tensor)
        image_heatmap = resize_heatmap(model_heatmap, image_size)
        gaze_px = heatmap_argmax(image_heatmap)
        gaze_norm = point_to_normalized(gaze_px, image_size)

        return SinglePersonGazePrediction(
            gaze_point_px=(float(gaze_px[0]), float(gaze_px[1])),
            gaze_point_norm=gaze_norm,
            heatmap=image_heatmap,
            image_size=image_size,
        )

