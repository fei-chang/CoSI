import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
import torchvision.transforms as transforms
from typing import Dict, Union
from omegaconf import DictConfig
from .base_model import BaseGazeModel
from .backbones import  ClassificationHead, ContextPattern
from .spatial_relator import SpatialRelator
from .model_utils import compute_confidence

# Abstract Backbone class
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self):
        pass

# Official DINOv2 backbones from torch hub (https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub)
class DinoV2Backbone(Backbone):
    def __init__(self, model_name, 
                 load_local=False,
                 local_dir = ''):
        super(DinoV2Backbone, self).__init__()
        print("Here loading the dino model..")
        if load_local:
            self.model = torch.hub.load(local_dir, model_name, force_reload=True, source='local')
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        print("Done loading the dino model")

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2) # "b (out_h out_w) c -> b c out_h out_w"
        return x
    
    def get_dimension(self):
        return self.model.embed_dim
    
    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.patch_size, w // self.model.patch_size)
    
    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            transforms.Resize(in_size),
        ])


class CoSIGazeBackbone(nn.Module):
    def __init__(self, backbone, 
                 dim=256, 
                 num_layers=3, 
                 in_size=(448, 448), 
                 out_size=(64, 64),
                 inout = True):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.model.patch_size
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.head_token = nn.Embedding(1, self.dim)
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))
        if self.inout: self.inout_token = nn.Embedding(1, self.dim)
        self.pattern_token = nn.Embedding(1, self.dim)

        self.transformer = nn.Sequential(*[
            Block(
                dim=self.dim, 
                num_heads=8, 
                mlp_ratio=4, 
                drop_path=0.1)
                for i in range(num_layers)
                ])

    def unfreeze_pattern_embedding(self):
        for param in self.pattern_token.parameters():
            param.requires_grad = True
        
    def forward(self, images, head_channels):

        num_ppl_per_img = head_channels.shape[1]
        x = self.backbone.forward(images)
        x = self.linear(x)
        x = x + self.pos_embed

        x = x.repeat_interleave(num_ppl_per_img, dim=0) 
        head_maps = F.max_pool2d(head_channels, kernel_size=self.patch_size, 
                                 stride=self.patch_size)
        bn, pn, w, h = head_maps.shape
        head_maps = head_maps.view(bn*pn,w,h)
        head_map_embeddings = head_maps.unsqueeze(dim=1)*self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = torch.cat([self.pattern_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.transformer(x)

        inout_feature = None
        pattern_feature = None

        offset = 0
        # slice off inout tokens from scene tokens
        if self.inout: 
            inout_feature =  x[:, offset, :] 
            offset+=1


        pattern_feature = x[:, offset, :]
        offset+=1

        heatmap_feature = x[:, offset:, :]

        heatmap_feature = heatmap_feature.reshape(heatmap_feature.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2) 
        
        return heatmap_feature, inout_feature, pattern_feature
    
    def export_component_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}
        
    def load_component_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set([k for k in keys1 if not k.startswith("backbone")])
            keys2 = set([k for k in keys2 if not k.startswith("backbone")])
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]
        
        self.load_state_dict(current_state_dict, strict=False)


# From https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
    

class CoSi(BaseGazeModel):
    def __init__(
        self,
        cfg: DictConfig,
        device: Union[torch.device, str] = "cuda"
    ) -> None:
        super().__init__(cfg, device)
        
        self.use_context = False
        self.use_spatial = False
        self.integration = cfg.model.integration
        if self.use_pattern:
            if self.integration == 'context':
                self.use_context = True
                self.spatial_relator = None
            elif self.integration == 'spatial':
                self.use_spatial = True
            else:
                self.use_context = True
                self.use_spatial = True

            if self.use_spatial:
                self.indices_dict = cfg.model.spatial_relator.indices_dict
                self.spatial_relator = SpatialRelator(input_channel=len(self.indices_dict['principal']), embed_dim=cfg.model.dim)
            
            if (self.use_spatial) & (self.use_context):
                self.projector_context = nn.Sequential(
                    nn.Linear(cfg.model.dim, cfg.model.dim),
                    nn.ReLU(),
                    nn.LayerNorm(cfg.model.dim)
                )
                self.projector_spatial = nn.Sequential(
                    nn.Linear(cfg.model.dim, cfg.model.dim),
                    nn.ReLU(),
                    nn.LayerNorm(cfg.model.dim)
                )
                self.projector_concated = nn.Sequential(
                    nn.Linear(2 * cfg.model.dim, cfg.model.dim),
                    nn.ReLU(),
                )

            if self.integration == 'confidence_coordinated':
                self.gate = nn.Sequential(
                    nn.Linear(2 * cfg.model.dim + 1, cfg.model.dim),
                    nn.ReLU(),
                    nn.Linear(cfg.model.dim, 2)
                )
            if self.integration == 'confidence_gate':
                self.conf_thres = cfg.model.conf_thres
            if cfg.stage.pattern_type == 'multi_class':
                pattern_num = 5
            else:
                pattern_num = 1
                
            self.pattern_classifier = ClassificationHead(embed_dim=cfg.model.dim, num_classes = pattern_num)
        
            print("Initialized with Pattern Mode %s"%self.integration)

        # Initialize CoSI's DINOv2 gaze backbone.
        backbone = DinoV2Backbone(
            model_name=cfg.model.backbone.name,
            load_local=cfg.model.backbone.load_local,
            local_dir=cfg.model.backbone.local_dir)
        self.out_size = cfg.data.transform.output_resolution
        self.gaze_backbone = CoSIGazeBackbone(
            backbone=backbone,
            inout=self.use_inout,
            dim=cfg.model.dim,
            num_layers=cfg.model.num_layers,
            in_size=(cfg.data.transform.input_resolution, cfg.data.transform.input_resolution),
            out_size=self.out_size
        )
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(cfg.model.dim, cfg.model.dim, kernel_size=2, stride=2),
            nn.Conv2d(cfg.model.dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        if self.use_inout: 
            self.inout_head = nn.Sequential(
                nn.Linear(cfg.model.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
    def freeze_gaze_backbone(self):
        for param in self.gaze_backbone.parameters():
            param.requires_grad = False
        for param in self.heatmap_head.parameters():
            param.requires_grad = False
        if self.use_inout:
            for param in self.inout_head.parameters():
                param.requires_grad = False

    def freeze_spatial(self):
        assert self.spatial_relator
        for param in self.spatial_relator.parameters():
            param.requires_grad = False

    def freeze_dino_backbone(self):
        for param in self.gaze_backbone.backbone.parameters():
            param.requires_grad = False

    def forward_gaze(self, x, person: str = "principal"):
        """Run only the single-person gaze-follow branch."""
        if person not in x:
            raise KeyError(f"Single-person batch is missing {person!r}.")

        images, _, _ = self._preprocess_images(x)
        head_channel, _, _, _, _, _, _ = self._preprocess_individual(x, person)

        if head_channel.ndim != 4 or head_channel.shape[1] != 1:
            raise ValueError(
                "forward_gaze expects one head channel per image with shape "
                f"[B, 1, H, W], got {tuple(head_channel.shape)}."
            )

        heatmap_feature, inout_feature, _ = self.gaze_backbone(
            images, head_channel
        )
        heatmap_pred = self.heatmap_head(heatmap_feature).squeeze(dim=1)
        heatmap_pred = torchvision.transforms.functional.resize(
            heatmap_pred, self.out_size
        )

        output = {"pred_heatmap": heatmap_pred}
        if self.use_inout and inout_feature is not None:
            # forward_gaze is an inference API, so expose probabilities here.
            output["pred_inout"] = torch.sigmoid(self.inout_head(inout_feature))
        return output

    def forward(self, x):
        predictions = {
            'principal': {},
            'associate': {}
        }
        
        # Preprocess images
        images, _, _ = self._preprocess_images(x)
        
        # Prepare gaze-backbone inputs.
        head_channels = []
        # Process each person's data
        for person in ('principal', 'associate'):
            if person not in x:
                continue
            head_channel, _, gt_heatmap, gt_inouts, gt_gazes, gt_patterns, _ = self._preprocess_individual(x, person)
            head_channels.append(head_channel)
            predictions[person].update({
                'gt_heatmap': gt_heatmap,
                'gt_gaze_vector': gt_gazes,
                'gt_inout': gt_inouts,
                'gt_pattern': gt_patterns
            })
        head_channels = torch.concat(head_channels, dim = 1)
        bn, num_ppl_per_img, _, _ = head_channels.shape
        # Forward pass through the gaze backbone.
        heatmap_feature, inout_feature, context_pattern_feature = self.gaze_backbone(images, head_channels)

        heatmap_pred = self.heatmap_head(heatmap_feature).squeeze(dim=1)
        heatmap_pred = torchvision.transforms.functional.resize(heatmap_pred, self.out_size)
        heatmap_pred = heatmap_pred.view(bn, num_ppl_per_img, *heatmap_pred.shape[1:])

        if self.use_inout:
            inout_pred = self.inout_head(inout_feature)
            inout_pred = inout_pred.view(bn, num_ppl_per_img, *inout_pred.shape[1:])
        
        if self.use_context:
            context_pattern_feature = context_pattern_feature.view(bn, num_ppl_per_img, *context_pattern_feature.shape[1:])

        # update gaze follow task predictions
        for idx, person in enumerate(['principal', 'associate']):
            if person not in x:
                continue
            predictions[person].update({
                'pred_heatmap': heatmap_pred[:,idx],  
                'pred_inout':  inout_pred[:, idx] if self.use_inout else None,
            })

        if self.use_pattern:
            if self.use_spatial:
                head_channels = torchvision.transforms.functional.resize(head_channels, self.out_size)
                spatial_gaze_maps = [
                    predictions['principal']['pred_heatmap'] + head_channels[:, 0],            # p_gaze + p_head
                    predictions['associate']['pred_heatmap'] + head_channels[:, 1],            # a_gaze + a_head
                    predictions['principal']['pred_heatmap'] + predictions['associate']['pred_heatmap'],    # p_gaze + a_gaze
                    head_channels[:, 0] + predictions['associate']['pred_heatmap'],            # p_head + a_gaze
                    head_channels[:, 1] + predictions['principal']['pred_heatmap']             # a_head + p_gaze
                ]        
        for idx, person in enumerate(['principal', 'associate']):
            if person not in x:
                continue
                
            if self.use_pattern:
                if self.use_context:
                    person_context_pattern_feature = context_pattern_feature[:, idx]
                if self.use_spatial:
                    indices = self.indices_dict[person]
                    person_gaze_maps = torch.cat([spatial_gaze_maps[i].unsqueeze(1) for i in indices], dim=1)
                    spatial_pattern_feature = self.spatial_relator(person_gaze_maps)

                if self.integration == 'context':
                    pattern_feature = person_context_pattern_feature

                elif self.integration == 'spatial':
                    pattern_feature = spatial_pattern_feature

                elif self.integration == 'concated':
                    person_context_pattern_feature = self.projector_context(person_context_pattern_feature)
                    spatial_pattern_feature = self.projector_spatial(spatial_pattern_feature)
                    pattern_feature = self.projector_concated(torch.concat([person_context_pattern_feature, spatial_pattern_feature], dim = 1))
                
                elif 'confidence' in self.integration:
                    # Compute Confidence
                    heatmap_conf = compute_confidence(pred_heatmap=predictions[person]['pred_heatmap']).unsqueeze(1)
                    predictions[person]['pred_heatmap_conf'] = heatmap_conf # (BN, 1)

                    if self.integration == 'confidence_coordinated':
                        person_context_pattern_feature = self.projector_context(person_context_pattern_feature)
                        spatial_pattern_feature = self.projector_spatial(spatial_pattern_feature)
                        gate_input = torch.cat([person_context_pattern_feature, spatial_pattern_feature, heatmap_conf], dim=-1) 
                        gate_logits = self.gate(gate_input) 
                        gate_weights = F.softmax(gate_logits, dim=-1)  
                        wC, wG = gate_weights[:, 0:1], gate_weights[:, 1:2]
                        pattern_feature = wC * person_context_pattern_feature + wG * spatial_pattern_feature

                    elif self.integration == 'confidence_gate':
                        # Use spatial relator if confidence > conf_thres, otherwise use context feature
                        mask = (heatmap_conf > self.conf_thres).squeeze(-1)  # (BN,)
                        # Initialize pattern feature with zeros
                        pattern_feature = torch.zeros_like(person_context_pattern_feature)
                        
                        # Use spatial relator for high confidence samples
                        if mask.any():
                            indices = self.indices_dict[person]
                            person_gaze_maps = torch.cat([spatial_gaze_maps[i].unsqueeze(1) for i in indices], dim=1)
                            spatial_pattern_feature = self.spatial_relator(person_gaze_maps)
                            pattern_feature[mask] = spatial_pattern_feature[mask]
                        
                        # Use context feature for low confidence samples
                        if (~mask).any():
                            pattern_feature[~mask] = person_context_pattern_feature[~mask]

                    elif self.integration == 'confidence_weighted':
                        pattern_feature = heatmap_conf * person_context_pattern_feature + (1 - heatmap_conf) * spatial_pattern_feature
                    
                    else:
                        raise TypeError("Unknown integration mode. Please Check the Configuration File")
                pred_pattern = self.pattern_classifier(pattern_feature)
                predictions[person].update({
                'pred_pattern': pred_pattern,  
            })
              
        output_dict = self._format_output(predictions)
        if self.training:
            return self.criterion(predictions)
        
        return output_dict
