import os

import torch

from .cosi import CoSi


def build_model(device, cfg, verbose=True):
    """Build the only supported model: CoSI."""
    if cfg.model.model_name.lower() != "cosi":
        raise ValueError("This package supports only model.model_name=cosi.")

    model = CoSi(cfg=cfg, device=device)
    if cfg.pretrained_weights:
        load_pretrained_weights(model, cfg.pretrained_weights, verbose)
    elif verbose:
        print("[Warning] No pretrained weights loaded")
    return model.to(device)


def load_pretrained_weights(model, checkpoint_path, verbose=False):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    current = model.state_dict()
    compatible = {}
    for raw_name, value in checkpoint.items():
        # str.removeprefix() is unavailable in the Python 3.8 environment
        # used by the remote server.
        name = raw_name[len("module."):] if raw_name.startswith("module.") else raw_name
        if name in current and current[name].shape == value.shape:
            compatible[name] = value
        elif verbose:
            print(f"[Warning] Skipped incompatible checkpoint key: {raw_name}")

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if verbose:
        print(f"Loaded {len(compatible)} checkpoint tensors")
        if missing:
            print(f"[Warning] Missing model keys: {len(missing)}")
        if unexpected:
            print(f"[Warning] Unexpected checkpoint keys: {len(unexpected)}")
    return model


__all__ = ["CoSi", "build_model", "load_pretrained_weights"]
