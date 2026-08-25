"""Fine-tune CoSI on one of the supported datasets/stages."""

import argparse
import warnings

from hydra import compose, initialize
from omegaconf import DictConfig

from engine.trainer import Trainer


STAGES = (
    "finetune_coatt",
    "finetune_dyadic_gf",
    "finetune_dyadic_pattern",
    "finetune_gp_static",
    "finetune_laeo",
    "finetune_gazefollow",
)
INTEGRATIONS = (
    "context",
    "spatial",
    "confidence_coordinated",
    "concated",
    "confidence_gate",
    "confidence_weighted",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the CoSI model.")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--pretrained", default=None, help="Optional checkpoint path.")
    parser.add_argument("--integration", required=True, choices=INTEGRATIONS)
    parser.add_argument("--freeze-spatial", action="store_true")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0.")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = [
        f"stage={args.stage}",
        "model=cosi",
        f"model.integration={args.integration}",
        f"model.freeze_spatial={args.freeze_spatial}",
        "data.transform.input_resolution=448",
    ]
    if args.pretrained:
        overrides.append(f"pretrained_weights={args.pretrained}")
    if args.device:
        overrides.append(f"device={args.device}")

    with initialize(config_path="config", version_base=None):
        cfg: DictConfig = compose(config_name="config", overrides=overrides)
    Trainer(cfg).train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
