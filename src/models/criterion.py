import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class GazeCriterion(nn.Module):
    def __init__(self,
                heatmap_weight: float = 1000.0,
                inout_weight: float = 1.0,
                pattern_weight: float = 1.0,
                binary_class: bool = False
                ):

        """
        Args:
            heatmap_weight: weight for heatmap loss
            inout_weight: weight for in/out loss
            pattern_weight: weight for pattern loss
        """
        super().__init__()
        self.binary_class = binary_class
        # Store loss weights
        self.weights = {
            "heatmap": heatmap_weight,
            "inout": inout_weight,
            "pattern": pattern_weight
        }


        # Initialize loss functions
        self.loss_functions = {
            'heatmap': nn.MSELoss(),
            'inout': nn.BCEWithLogitsLoss(),
            'pattern': nn.BCEWithLogitsLoss() if binary_class else nn.CrossEntropyLoss()
        }

    def compute_individual_loss(self, pred_key: str, predictions: Dict, person: str) -> Tuple[torch.Tensor, bool]:
        """Compute individual loss component if predictions are available."""
        pred_name = f'pred_{pred_key}'
        gt_name = f'gt_{pred_key}'

        if (pred_name not in predictions[person] or
            gt_name not in predictions[person] or
            predictions[person][pred_name] is None or
            predictions[person][gt_name] is None):
            return None, False

        # Get predictions and ground truth
        pred = predictions[person][pred_name]
        gt = predictions[person][gt_name]

        if pred_key == 'inout':
            pred = pred.squeeze()
            gt = gt.float().squeeze()
        if pred_key == 'pattern':
            if self.binary_class:
                # LAEO/CoAtt labels may collate as [B] or [B, 1]. Normalize
                # both predictions and targets without dropping batch size 1.
                pred = pred.reshape(-1)
                gt = gt.float().reshape(-1)
            else:
                # CrossEntropyLoss expects [B, C] logits and [B] class IDs.
                if pred.ndim == 1:
                    pred = pred.unsqueeze(0)
                gt = gt.long().reshape(-1)
        # Compute loss
        loss = self.loss_functions[pred_key](pred, gt)
        weighted_loss = loss * self.weights[pred_key]

        return weighted_loss, True

    def forward(self, predictions: Dict) -> Dict:
        total_loss = 0
        losses = {}

        # List of all possible loss components
        loss_components = self.weights.keys()

        for person in ['principal', 'associate']:
            if person not in predictions or not predictions[person]:
                continue
            # Compute each loss component
            for component in loss_components:
                loss, computed = self.compute_individual_loss(component, predictions, person)
                if computed:
                    loss_name = f'{person}_{component}'
                    losses[loss_name] = loss
                    total_loss += loss

        losses['total'] = total_loss
        return losses
