from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import DictConfig
from .dyadic_gaze import DyadicGaze
from .gazefollow import GazeFollow
from .social_gaze_datasets import LAEO, VideoCoAtt, GP_Static
from .masking import MaskGenerator
from .data_utils import get_transform

def create_mask_generator(cfg, is_train):
    """Create mask generator from config."""
    mask_cfg = cfg.data.mask
    return MaskGenerator(
        input_size=cfg.data.transform.input_resolution // 14,
        mask_scene=mask_cfg.mask_scene,
        mask_head=mask_cfg.mask_head,
        max_scene_patches_ratio=mask_cfg.max_scene_patches_ratio,
        max_head_patches_ratio=mask_cfg.max_head_patches_ratio,
        mask_prob=mask_cfg.mask_prob
    ) if is_train else None


def build_dataset(cfg: DictConfig, dataset_name: str, is_train=True, sampled_annotation_csv=None):
    """Build dataset based on config."""
    dataset_map = {
        'gazefollow': GazeFollow,
        'dyadic': DyadicGaze,
        'gp_static': GP_Static,
        'laeo': LAEO,
        'coatt': VideoCoAtt,
    }

    common_params = {
        'transform': get_transform(
            input_resolution=cfg.data.transform.input_resolution,
            mean=cfg.data.transform.mean,
            std=cfg.data.transform.std,
        ),
        'input_size': cfg.data.transform.input_resolution,
        'output_size': cfg.data.transform.output_resolution,
        'is_train': is_train,
    }

    dataset_cfg = cfg.data.datasets[dataset_name]

    if dataset_name in ['gazefollow', 'dyadic', 'gp_static']:
        common_params.update({
            'frame_dir': dataset_cfg.frames.train if is_train else dataset_cfg.frames.val,
            # 'head_heatmaps': dataset_cfg.head_heatmaps,
            'annotation_csv': dataset_cfg.annotations.train if is_train else dataset_cfg.annotations.val,
            'mask_generator': create_mask_generator(cfg, is_train),
        })

        if sampled_annotation_csv:
            common_params.update({'annotation_csv':sampled_annotation_csv})
            print("[WARNING] overwrite annotation path to %s"%sampled_annotation_csv)
            
        return dataset_map[dataset_name](**common_params)
    
    elif dataset_name in ['laeo', 'coatt']:
        common_params.update({
            'frame_dir': dataset_cfg.frames.train if is_train else dataset_cfg.frames.val,
            'annotation_csv': dataset_cfg.annotations.train if is_train else dataset_cfg.annotations.val,
        })
        return dataset_map[dataset_name](**common_params)
    
    else:   
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Choose one of: {', '.join(dataset_map)}"
        )



def build_dataloader(cfg: DictConfig, dataset_name: str, is_train=True, 
                     sampled_annotation_csv=None,
                     batch_size=32,
                     num_workers=2,
                     pin_memory=False,
                     prefetch_factor=1,
                     persistent_workers=False,
                     distributed=False) -> DataLoader:
    """Generic DataLoader builder with config."""
    dataset = build_dataset(cfg, dataset_name, is_train, sampled_annotation_csv)
    
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        sampler=DistributedSampler(dataset, shuffle=is_train) if distributed else None,
        shuffle=is_train if not distributed else False,
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(**loader_kwargs)
