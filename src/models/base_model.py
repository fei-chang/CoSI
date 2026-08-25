from typing import Union, Dict, Tuple, Optional
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .criterion import GazeCriterion

class BaseGazeModel(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        device: Union[torch.device, str] = "cuda"
    ) -> None:
        super().__init__()
        self._setup_loss_flags(cfg)
        self.device = torch.device(device)
        self.binary_class = (cfg.stage.pattern_type!='multi_class')
        self.criterion = GazeCriterion(
            heatmap_weight=cfg.stage.weights.heatmap,
            pattern_weight=cfg.stage.weights.pattern,
            inout_weight=cfg.stage.weights.inout,
            binary_class=self.binary_class
        )

    def freeze_gaze_backbone(self):
        for param in self.gaze_backbone.parameters():
            param.requires_grad = False

    def _setup_loss_flags(self, cfg):
        self.use_heatmap = cfg.stage.weights.heatmap > 0
        self.use_inout = cfg.stage.weights.inout > 0
        self.use_pattern = cfg.stage.weights.pattern > 0

    def _preprocess_images(self, batched_inputs):
        return (
            batched_inputs["images"].to(self.device),
            batched_inputs["head_heatmaps"].to(self.device) if 'head_heatmaps' in batched_inputs else None,
            batched_inputs["image_masks"].to(self.device) if 'image_masks' in batched_inputs else None
        )

    def _preprocess_individual(self, batched_inputs, person: str) -> Dict:
        person_inputs = batched_inputs[person]
        return (
            person_inputs["head_channel"].to(self.device),
            person_inputs["head_crop"].to(self.device) if 'head_crop' in person_inputs else None,
            person_inputs["gaze_heatmap"].to(self.device) if 'gaze_heatmap' in person_inputs else None,
            person_inputs["inout"].to(self.device) if 'inout' in person_inputs else None,
            person_inputs["gaze_vector"].to(self.device) if 'gaze_vector' in person_inputs else None,
            person_inputs["pattern"].to(self.device) if 'pattern' in person_inputs else None,
            person_inputs['head_box'].to(self.device) if 'head_box' in person_inputs else None,
        )

    def _format_output(self, predictions: Dict) -> Dict:
        output_dict = {}
        for person in predictions:
            if not predictions[person]:
                continue
            preds = predictions[person]
            if self.use_pattern and 'pred_pattern' in preds:
                pattern_logits = preds['pred_pattern']
                if self.binary_class:
                    pred_patterns = pattern_logits.sigmoid().reshape(-1)
                else:
                    if pattern_logits.ndim == 1:
                        pattern_logits = pattern_logits.unsqueeze(0)
                    pred_patterns = torch.argmax(pattern_logits, dim=-1)
            else:
                pred_patterns = None
            output_dict[person] = {
                "pred_heatmap": preds.get('pred_heatmap', None),
                # The in/out head returns logits for BCEWithLogitsLoss during
                # training. Convert them to probabilities at inference only.
                "pred_inouts": preds.get('pred_inout', None).sigmoid() if self.use_inout else None,
                "pred_patterns": pred_patterns,
                "pred_heatmap_conf": preds.get('pred_heatmap_conf', None),
            }
        return output_dict
