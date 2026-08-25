import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


from .backbones import ClassificationHead

class BaseGazePatternModel(nn.Module):
    """Base class for gaze pattern models with optional consistency loss"""
    
    def __init__(
        self,
        device: Union[torch.device, str] = "cuda",
        use_consistency: bool = False
    ) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(device)

    def forward(self, batched_inputs):
        """Forward pass through the model"""
        processed_inputs = self.preprocess_inputs(batched_inputs)

        predictions = {
            person: self.predict_pattern(processed_inputs, person)
            for person in ('principal', 'associate')
        }

        return self._compute_losses(predictions) if self.training else self._format_inference_output(predictions)

    def _compute_losses(self, predictions):
        """Compute classification and consistency losses"""
        loss_dict = {
            person: self.criterion(pred['pred_pattern'], pred['gt_pattern'])
            for person, pred in predictions.items()
        }


        total_loss = 0
        for loss_value in loss_dict.values():
            total_loss += loss_value
        loss_dict['total'] = total_loss

        return loss_dict

    def _format_inference_output(self, predictions):
        """Format predictions for inference mode"""
        return {
            person: {
                "pred_patterns": torch.argmax(pred['pred_pattern'], dim=1)
            }
            for person, pred in predictions.items()
        }

    def preprocess_inputs(self, batched_inputs):
        """To be implemented by child classes"""
        raise NotImplementedError

    def predict_pattern(self, processed_inputs, person):
        """To be implemented by child classes"""
        raise NotImplementedError

class SpatialRelator(nn.Module):
    def __init__(self, input_channel=2, 
                 embed_dim=256):
        super(SpatialRelator, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, embed_dim, kernel_size=3, padding=1)  
        self.bn5 = nn.BatchNorm2d(embed_dim)

        # Add global pooling and flattening
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) 

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) 

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) 

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.global_pool(x)  
        x = self.flatten(x)      

        return x

class WrappedSpatialRelator(BaseGazePatternModel):
    def __init__(
        self,
        device: Union[torch.device, str] = "cuda",
        embed_dim: int = 256,
        num_classes: int = 5,
        use_consistency: bool = False,
        indices_dict=None
    ) -> None:
        super().__init__(device=device, use_consistency=use_consistency)

        self.indices_dict = indices_dict or {
            'principal': [2, 3, 4],
            'associate': [2, 4, 3]
        }

        input_channels = len(self.indices_dict['principal'])
        self.spatial_relator = SpatialRelator(embed_dim=embed_dim, input_channel=input_channels)
        self.pattern_classifier = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes)

    def preprocess_inputs(self, batched_inputs):
        """Preprocess inputs for both principal and associate"""
        input_dict = {}
        
        for person in ('principal', 'associate'):
            person_data = batched_inputs[person]

            input_dict[person] = {
                "head_channel": person_data["head_channel"].to(self.device),
                "gaze_heatmap": person_data["gaze_heatmap"].to(self.device),
                "gt_pattern": person_data.get("pattern", None).to(self.device) if "pattern" in person_data else None
            }
        # Create combined heatmaps
        p, a = input_dict['principal'], input_dict['associate']
        input_dict['heatmap'] = [
            p['gaze_heatmap'] + p['head_channel'],   # p_gaze + p_head
            a['gaze_heatmap'] + a['head_channel'],   # a_gaze + a_head
            p['gaze_heatmap'] + a['gaze_heatmap'],   # p_gaze + a_gaze
            p['head_channel'] + a['gaze_heatmap'],   # p_head + a_gaze
            a['head_channel'] + p['gaze_heatmap']    # a_head + p_gaze
        ]

        return input_dict

    def predict_pattern(self, processed_inputs, person):
        """Predict pattern for a given person"""
        indices = self.indices_dict[person]
        input_tensor = torch.cat([processed_inputs['heatmap'][i] for i in indices], dim=1)

        # Forward pass
        features = self.spatial_relator(input_tensor)
        pred_pattern = self.pattern_classifier(features)

        return {
            'pred_pattern': pred_pattern,
            'gt_pattern': processed_inputs[person]['gt_pattern']
        }
