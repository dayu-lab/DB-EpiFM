import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from datasets import chb_dataset, tuab_dataset, tuev_dataset
from finetune_trainer import Trainer
from models import model_for_chb, model_for_tuab, model_for_tuev


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a Boolean value, received {value!r}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_parser():
    parser = argparse.ArgumentParser(description="DB-EpiFM downstream fine-tuning")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--clip_value", type=float, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--classifier", type=str, default="all_patch_reps")
    parser.add_argument("--downstream_dataset", type=str, default="TUEV",
                        choices=["TUAB", "CHB-MIT", "TUEV"])
    parser.add_argument("--datasets_dir", type=str,
                        default="./data/processed/tuev_3class")
    parser.add_argument("--num_of_classes", type=int, default=3)
    parser.add_argument("--model_dir", type=str,
                        default="./outputs/finetuning/tuev")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--multi_lr", type=str2bool, nargs="?", const=True,
                        default=True)
    parser.add_argument("--frozen", type=str2bool, nargs="?", const=True,
                        default=False)
    parser.add_argument("--use_pretrained_weights", type=str2bool, nargs="?",
                        const=True, default=True)
    parser.add_argument("--foundation_dir", type=str,
                        default="./checkpoints/DB-EpiFM_pretrain.pth")
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None,
                        help="Run one CHB-MIT fold; omit to run all five folds")
    parser.add_argument("--split_manifest", type=str,
                        default="./splits/chbmit_patient_5fold.json")
    return parser


def validate_checkpoint(params):
    if not params.use_pretrained_weights:
        return
    checkpoint_path = Path(params.foundation_dir).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}. "
            "Provide the correct path with --foundation_dir."
        )
    params.foundation_dir = str(checkpoint_path)


def run_single(params):
    setup_seed(params.seed)
    if params.downstream_dataset == "TUAB":
        loader = tuab_dataset.LoadDataset(params).get_data_loader()
        model = model_for_tuab.Model(params)
    elif params.downstream_dataset == "CHB-MIT":
        loader = chb_dataset.LoadDataset(params).get_data_loader()
        model = model_for_chb.Model(params)
    else:
        loader = tuev_dataset.LoadDataset(params).get_data_loader()
        model = model_for_tuev.Model(params)
    trainer = Trainer(params, loader, model)
    if params.downstream_dataset == "TUEV":
        return trainer.train_for_multiclass()
    return trainer.train_for_binaryclass()


def run_chbmit_folds(params):
    folds = [params.fold] if params.fold else [1, 2, 3, 4, 5]
    fold_results = []
    for fold in folds:
        fold_params = copy.deepcopy(params)
        fold_params.fold = fold
        fold_params.model_dir = os.path.join(params.model_dir, f"fold_{fold}")
        print(f"Starting CHB-MIT fold {fold} with seed {params.seed}")
        result = run_single(fold_params)
        result["fold"] = fold
        fold_results.append(result)

    summary = {
        metric: float(np.mean([item[metric] for item in fold_results]))
        for metric in ("balanced_accuracy", "sensitivity", "specificity")
    }
    summary["balanced_accuracy"] = (
        summary["sensitivity"] + summary["specificity"]
    ) / 2.0
    output = {"seed": params.seed, "folds": fold_results, "run_level": summary}
    os.makedirs(params.model_dir, exist_ok=True)
    with open(os.path.join(params.model_dir, "run_level_metrics.json"), "w",
              encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
    print("CHB-MIT run-level result:", summary)
    return output


def main():
    params = build_parser().parse_args()
    validate_checkpoint(params)
    torch.cuda.set_device(params.cuda)
    if params.downstream_dataset == "CHB-MIT":
        run_chbmit_folds(params)
    else:
        run_single(params)


if __name__ == "__main__":
    main()
