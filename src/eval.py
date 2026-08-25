"""Evaluate a CoSI checkpoint on a supported benchmark."""

import argparse
import os

import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from data import build_dataloader
from engine.evaluator import Evaluator
from models import build_model


STAGES = ("eval_dyadic", "eval_gp_static", "eval_coatt", "eval_laeo")
INTEGRATIONS = (
    "context",
    "spatial",
    "confidence_coordinated",
    "concated",
    "confidence_gate",
    "confidence_weighted",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the CoSI model.")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--pretrained", required=True, help="CoSI checkpoint path.")
    parser.add_argument("--integration", required=True, choices=INTEGRATIONS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metric-save-file", default="metrics.txt")
    parser.add_argument("--pred-save-file", default=None)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--rule-based", action="store_true")
    parser.add_argument("--print-classwise-metrics", action="store_true")
    parser.add_argument("--skip-auc-computation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = [
        f"stage={args.stage}",
        "model=cosi",
        f"model.integration={args.integration}",
        f"pretrained_weights={args.pretrained}",
        "data.transform.input_resolution=448",
    ]
    if args.device:
        overrides.append(f"device={args.device}")

    with initialize(config_path="config", version_base=None):
        cfg: DictConfig = compose(config_name="config", overrides=overrides)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(device, cfg, verbose=True)
    val_loader = build_dataloader(
        cfg,
        cfg.stage.dataset_name,
        is_train=False,
        batch_size=cfg.stage.batch_size,
        num_workers=cfg.stage.num_workers,
        pin_memory=cfg.stage.pin_memory,
        prefetch_factor=cfg.stage.prefetch_factor,
        persistent_workers=cfg.stage.persistent_workers,
        distributed=False,
    )

    evaluator = Evaluator(model, device, pattern_type=cfg.stage.pattern_type)
    evaluator.load_data(
        cfg.stage.dataset_name,
        val_loader,
        eval_pattern=cfg.stage.weights.pattern > 0,
        eval_inout=cfg.stage.weights.inout > 0,
    )

    output_dir = args.output_dir or os.path.join(cfg.save_dir, args.stage)
    os.makedirs(output_dir, exist_ok=True)
    model_id = f"cosi_{args.integration}" + ("_rule" if args.rule_based else "")
    metric_path = os.path.join(output_dir, f"{model_id}_{args.metric_save_file}")
    pred_path = (
        os.path.join(output_dir, f"{model_id}_{args.pred_save_file}")
        if args.pred_save_file
        else None
    )
    evaluator.evaluate(
        model_id=model_id,
        metric_save_path=metric_path,
        pred_save_path=pred_path,
        print_classwise_metrics=args.print_classwise_metrics,
        get_pattern_from_rules=args.rule_based,
        skip_auc_computation=args.skip_auc_computation,
        get_pred_confidence="confidence" in args.integration,
    )


if __name__ == "__main__":
    main()
