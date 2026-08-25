import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from .components import Block, PatchEmbed, Mlp


class ClassificationHead(nn.Module):
    def __init__(self, 
                 embed_dim = 256, 
                 num_classes=5,
                 dropout = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim*2, num_classes),
            )
    def forward(self, x):
        # Always preserve [batch, classes]. Shape normalization for a specific
        # loss or inference output belongs at that consumer boundary.
        return self.mlp(x)

class GazeTargetHead(nn.Module):
    def __init__(self, embed_dim = 384, use_inout=True):
        super(GazeTargetHead, self).__init__()
        # Gaze heatmap decoder with upsampling
        self.heatmap_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(24, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

            nn.Conv2d(1, 1, kernel_size=1)
        )
        
        # In/out prediction head
        if use_inout:
            self.inout_head =  nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
        )
        else:
            self.inout_head = None

    def forward(self, heatmap_gaze_feature, inout_gaze_feature=None):
        outputs = {}
        # Generate heatmap prediction
        pred_heatmap = self.heatmap_decoder(heatmap_gaze_feature)
        outputs['pred_heatmap'] = pred_heatmap

        if self.inout_head is not None:
            pred_inout = self.inout_head(inout_gaze_feature)
            outputs['pred_inout'] = pred_inout

        return outputs

class HeadBackbone(nn.Module):
    def __init__(
        self,
        kernel_size=14,
        embed_dim=8,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.patch_embed = nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, 
                    kernel_size=(self.kernel_size, self.kernel_size), 
                    stride=(self.kernel_size, self.kernel_size), 
                    padding=(0, 0)
                ),
                nn.ReLU(),
                nn.Conv2d(
                    embed_dim, 1, 
                    kernel_size=(1, 1), 
                    stride=(1, 1), 
                    padding=(0, 0)
                )
            )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, scene_images, head_box_channels):
        head_attn = F.max_pool2d(
            head_box_channels,
            (self.kernel_size, self.kernel_size),
            (self.kernel_size, self.kernel_size),
            (0, 0),
        )
        embedded_scene = self.patch_embed(scene_images)
        embedded_head = embedded_scene * head_attn
        
        embedded_head = embedded_head.masked_fill(head_attn <= 0, -1e9)

        flattened = embedded_head.view(len(embedded_head), -1)
        softmaxed = F.softmax(flattened, dim=1)
        embedded_head = softmaxed.reshape_as(embedded_head)
                
        return embedded_head

class GazeFeatureExtractor(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1,
        use_cls_token=True,
        use_mask_token=True,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        attn_idx_list = [1,4,8]
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values

            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
            attn_idx_list:  (list) indices of blocks for which attention maps should be returned
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False
        )
        
        self.use_cls_token = use_cls_token
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        )

        self.use_mask_token = use_mask_token
        self.mask_token = (
            nn.Parameter(torch.zeros(1, embed_dim)) if self.use_mask_token else None
        )

        self.num_patches = (img_size // patch_size) * (
            img_size // patch_size
        )
        self.num_tokens = (self.num_patches + 1) if self.use_cls_token else self.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.cls_patch_offset = int(self.use_cls_token)

        # Transformer blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [drop_path_rate] * depth if drop_path_uniform else [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=init_values,
                return_attn=(i in attn_idx_list)
            )
            for i in range(depth)
        ])
        
        # self.norm = norm_layer(embed_dim)
        
        self.head_embed = HeadBackbone()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

        # Initialize head embeding layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def prepare_tokens_with_masks(self, x, masks=None):

        x, map_size = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.get_abs_pos(
                self.pos_embed, self.use_cls_token, (x.shape[1], x.shape[2])
            )

        h, w = map_size
        b = x.shape[0]
        x = x.reshape(b, h * w, -1)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        return x, map_size
    
        
    def get_abs_pos(self, abs_pos, has_cls_token, hw):
        """
        Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
            dimension for the original embeddings.
        Args:
            abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
            has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
            hw (Tuple): size of input image tokens.

        Returns:
            Absolute positional embeddings after processing with shape (1, H, W, C)
        """
        h, w = hw
        if has_cls_token:
            abs_pos = abs_pos[:, 1:]
        xy_num = abs_pos.shape[1]
        size = int(math.sqrt(xy_num))
        assert size * size == xy_num

        if size != h or size != w:
            new_abs_pos = F.interpolate(
                abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )

            return new_abs_pos.permute(0, 2, 3, 1)
        else:
            return abs_pos.reshape(1, h, w, -1)
        
    def forward(self, scene_images, head_box_channels, masks=None):
        # Get patch embeddings for scene images
        B, *_= scene_images.shape
        scene_tokens, map_size = self.prepare_tokens_with_masks(scene_images, masks)
        W, H = map_size
        # Process through transformer blocks and collect attention maps
        attention_maps = []
        for blk in self.blocks:
            if blk.return_attn:
                scene_tokens, attn = blk(scene_tokens)
                attn = attn.reshape(B, self.num_heads, self.num_tokens, -1)

                attn = attn[
                :,
                :,
                self.cls_patch_offset : ,
                self.cls_patch_offset : ,
            ]
                attention_maps.append(attn)
            else:
                scene_tokens = blk(scene_tokens)

        # # Concatenate attention maps
        attention_maps = torch.cat(attention_maps, dim=1) # [B, 12*4 = 36, num_patches = 256,  num_patches = 256]
        # Apply head embedding layers
        head_tokens = self.head_embed(scene_images, head_box_channels)
        B, _, H, W = head_tokens.shape
        attention_maps = attention_maps.reshape(
            B, -1, H*W, H, W 
        )
        total_attention_layers = attention_maps.shape[1]
        embedded_heads = head_tokens.reshape(B, 1, -1, 1, 1).repeat(1, total_attention_layers, 1, 1, 1)

        gaze_heatmap_feature = (attention_maps*embedded_heads).sum(dim=2) 
        reshaped_head_tokens = head_tokens.repeat(1, self.embed_dim, 1,1)
        reshaped_scene_tokens = scene_tokens[:, self.cls_patch_offset: ].reshape(B, H, W, -1).permute(0, 3, 1, 2)

        gaze_inout_feature = (reshaped_head_tokens*reshaped_scene_tokens).sum(dim=(2, 3)).reshape(B, -1)

        return gaze_heatmap_feature, gaze_inout_feature

class ContextPattern(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.principal_proj =  nn.Sequential(
                nn.Linear(dim, dim*2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim*2, dim),
            )
        self.associate_proj = nn.Sequential(
                nn.Linear(dim, dim*2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim*2, dim),
            )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_proj = nn.Linear(dim*2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, p1_features, p2_features):
        B, C, W, H = p1_features.shape
        # assert not p1_features equal to p2_features 
        assert not torch.allclose(p1_features, p2_features)

        p1_pooled = self.avg_pool(p1_features).view(B, C)  # B, C
        p1_pooled = self.norm(p1_pooled)
        p1_proj = self.principal_proj(p1_pooled)

        p2_pooled = self.avg_pool(p2_features).view(B, C)
        p2_pooled = self.norm(p2_pooled)
        p2_proj = self.associate_proj(p2_pooled)
        
        x = torch.cat([p1_proj, p2_proj], dim=1)
        out = self.out_proj(x)
        return out
