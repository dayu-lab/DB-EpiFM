"""
Fine-tune / evaluate DB-EpiFM on CHB-MIT window-level binary classification.

This script is designed for the current DB-EpiFM project structure:
    DB-EpiFM/
        datasets/chb_dataset.py
        models/model_for_chb.py
        finetune_chbmit_window_metrics.py   <-- put this script here

Window-level positive class definition:
    y = 1: seizure-related / seizure segment
    y = 0: non-seizure segment

Main outputs:
    - Balanced Accuracy
    - AUC-PR
    - AUROC
    - Sensitivity = TP / (TP + FN)
    - Specificity = TN / (TN + FP)
    - FPR = FP / (FP + TN)
    - confusion matrix [[TN, FP], [FN, TP]]
    - optional CSV containing y_true, y_prob, y_pred for each test window
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import signal
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Make sure local project imports work when this file is placed in DB-EpiFM root.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import model_for_chb  # noqa: E402


class CHBMITWindowDataset(Dataset):
    """
    Dataset for processed CHB-MIT 10-s window .pkl files.

    Expected directory structure:
        datasets_dir/
            train/*.pkl
            val/*.pkl
            test/*.pkl

    Expected pkl format:
        {
            "X": numpy array, shape usually (16, 2560),  # 16 channels, 10 s at 256 Hz
            "y": 0 or 1
        }

    Preprocessing is kept consistent with datasets/chb_dataset.py:
        1) resample time dimension to 2000 points;
        2) reshape to (16, 10, 200);
        3) divide by 100.
    """

    def __init__(self, datasets_dir: str, split: str = "train"):
        self.split = split
        self.split_dir = Path(datasets_dir) / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.files = sorted(self.split_dir.glob("*.pkl"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pkl files found in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with open(path, "rb") as f:
            data_dict = pickle.load(f)

        x = np.asarray(data_dict["X"], dtype=np.float32)  # expected: (16, 2560)
        y = int(data_dict["y"])

        # Keep identical logic to datasets/chb_dataset.py
        x = signal.resample(x, 2000, axis=1).astype(np.float32)
        if x.shape[0] != 16 or x.shape[1] != 2000:
            raise ValueError(
                f"Unexpected X shape after resampling in {path}: {x.shape}. "
                "Expected (16, 2000). Please check channel number or preprocessing."
            )
        x = x.reshape(16, 10, 200) / 100.0

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32), str(path)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Build CHB-MIT model by only calling the existing model_for_chb.Model."""
    model = model_for_chb.Model(args)
    model.to(device)
    return model


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: nn.Module, checkpoint: str, device: torch.device, strict: bool = True) -> nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=strict)
    return model


def compute_pos_weight(train_set: CHBMITWindowDataset) -> torch.Tensor:
    labels = []
    for path in train_set.files:
        with open(path, "rb") as f:
            labels.append(int(pickle.load(f)["y"]))
    labels = np.asarray(labels)
    neg = int(np.sum(labels == 0))
    pos = int(np.sum(labels == 1))
    if pos == 0:
        raise ValueError("No positive samples in training set; cannot compute pos_weight.")
    pos_weight = neg / pos
    print(f"Training label count: neg={neg}, pos={pos}, pos_weight={pos_weight:.6f}")
    return torch.tensor([pos_weight], dtype=torch.float32)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    clip_value: float,
) -> float:
    model.train()
    losses: List[float] = []

    for x, y, _ in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x).view(-1)
        loss = criterion(logits, y.view(-1))
        loss.backward()

        if clip_value and clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    model.eval()
    y_true_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []
    path_all: List[str] = []
    losses: List[float] = []

    for x, y, paths in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        logits = model(x).view(-1)
        prob = torch.sigmoid(logits)

        if criterion is not None:
            loss = criterion(logits, y.view(-1))
            losses.append(float(loss.detach().cpu().item()))

        y_true_all.append(y.detach().cpu().numpy().astype(int).reshape(-1))
        y_prob_all.append(prob.detach().cpu().numpy().reshape(-1))
        path_all.extend(list(paths))

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=float)
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return y_true, y_prob, path_all, mean_loss


def compute_window_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)

    # Force binary layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    try:
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    except ValueError:
        balanced_acc = float("nan")

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        auc_pr = auc(recall, precision)
    except ValueError:
        auc_pr = float("nan")

    precision_value = precision_score(y_true, y_pred, zero_division=0)
    f1_value = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "num_windows": int(len(y_true)),
        "num_negative": int(np.sum(y_true == 0)),
        "num_positive": int(np.sum(y_true == 1)),
        "balanced_accuracy": float(balanced_acc),
        "auc_pr": float(auc_pr),
        "auroc": float(auroc),
        "sensitivity_recall_tpr": float(sensitivity),
        "specificity_tnr": float(specificity),
        "fpr": float(fpr),
        "precision_ppv": float(precision_value),
        "f1": float(f1_value),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix_rows_true_cols_pred": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def print_metrics(name: str, metrics: Dict[str, object], loss: Optional[float] = None) -> None:
    print(f"\n========== {name} ==========")
    if loss is not None and not np.isnan(loss):
        print(f"Loss              : {loss:.6f}")
    print(f"Windows           : {metrics['num_windows']}  "
          f"neg={metrics['num_negative']}  pos={metrics['num_positive']}")
    print(f"Threshold         : {metrics['threshold']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.6f}")
    print(f"AUC-PR            : {metrics['auc_pr']:.6f}")
    print(f"AUROC             : {metrics['auroc']:.6f}")
    print(f"Sensitivity/Recall: {metrics['sensitivity_recall_tpr']:.6f}")
    print(f"Specificity       : {metrics['specificity_tnr']:.6f}")
    print(f"FPR               : {metrics['fpr']:.6f}")
    print(f"Precision         : {metrics['precision_ppv']:.6f}")
    print(f"F1                : {metrics['f1']:.6f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(metrics["confusion_matrix_rows_true_cols_pred"]))


def save_predictions_csv(paths: List[str], y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_csv: str) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("path,y_true,y_prob,y_pred\n")
        for p, yt, yp, ypd in zip(paths, y_true, y_prob, y_pred):
            f.write(f"{p},{int(yt)},{float(yp):.10f},{int(ypd)}\n")
    print(f"Saved window predictions to: {out_path}")


def save_metrics_json(metrics: Dict[str, object], out_json: str) -> None:
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune/evaluate DB-EpiFM on CHB-MIT and report window-level sensitivity, specificity, and FPR."
    )

    # Reproducibility / device
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cuda", type=int, default=0)

    # Data
    parser.add_argument("--datasets_dir", type=str,
                        default='./data/processed/chbmit/balanced_dataset')
    parser.add_argument("--num_workers", type=int, default=4)

    # Model parameters required by models/model_for_chb.py
    parser.add_argument("--foundation_dir", type=str, default="./checkpoints/db_epifm_pretrained.pth",
                        help="Path to pretrained DB-EpiFM backbone weights. Used only when training with pretrained weights.")
    parser.add_argument("--no_pretrained_weights", action="store_true",
                        help="Disable loading pretrained backbone weights at model initialization.")
    parser.add_argument("--classifier", type=str, default="all_patch_reps",
                        choices=["all_patch_reps", "all_patch_reps_twolayer", "all_patch_reps_onelayer", "avgpooling_patch_reps"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--frozen", action="store_true", help="Freeze backbone during training.")

    # Training
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate a saved fine-tuned checkpoint on the test split.")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Fine-tuned whole-model checkpoint. Required for --eval_only; optional for resume/test.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD"])
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--multi_lr", action="store_true",
                        help="Use lr for backbone and 5*lr for classifier, consistent with the original Trainer option.")
    parser.add_argument("--auto_pos_weight", action="store_true",
                        help="Use BCEWithLogitsLoss(pos_weight=neg/pos) computed from train split.")
    parser.add_argument("--best_metric", type=str, default="auroc",
                        choices=["auroc", "auc_pr", "balanced_accuracy", "sensitivity_recall_tpr", "f1"],
                        help="Validation metric used for saving best checkpoint.")

    # Evaluation / output
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for converting sigmoid probability into class label.")
    parser.add_argument("--output_dir", type=str, default="./outputs/chbmit_window_metrics")
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    # Make args compatible with existing model_for_chb.Model(params)
    args.use_pretrained_weights = not args.no_pretrained_weights
    if args.eval_only:
        # During eval, the full fine-tuned checkpoint is loaded afterwards;
        # avoid accidentally loading an unrelated foundation checkpoint first.
        args.use_pretrained_weights = False

    setup_seed(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
    print(f"Using device: {device}")
    print(f"Positive class: y=1 seizure-related window; Negative class: y=0 non-seizure window")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = CHBMITWindowDataset(args.datasets_dir, "train")
    val_set = CHBMITWindowDataset(args.datasets_dir, "val")
    test_set = CHBMITWindowDataset(args.datasets_dir, "test")
    print(f"Dataset size: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args, device)

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when using --eval_only")
        print(f"Loading fine-tuned checkpoint: {args.checkpoint}")
        model = load_checkpoint(model, args.checkpoint, device, strict=True)

        criterion = nn.BCEWithLogitsLoss().to(device)
        y_true, y_prob, paths, test_loss = predict(model, test_loader, criterion, device)
        metrics = compute_window_metrics(y_true, y_prob, args.threshold)
        print_metrics("Test", metrics, loss=test_loss)

        metrics_path = output_dir / "test_window_metrics.json"
        save_metrics_json(metrics, str(metrics_path))
        if args.save_predictions:
            save_predictions_csv(paths, y_true, y_prob, args.threshold, str(output_dir / "test_window_predictions.csv"))
        return

    # Training mode
    if args.auto_pos_weight:
        pos_weight = compute_pos_weight(train_set).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)

    # Optionally resume from a fine-tuned whole-model checkpoint.
    if args.checkpoint:
        print(f"Loading checkpoint before training/resuming: {args.checkpoint}")
        model = load_checkpoint(model, args.checkpoint, device, strict=True)

    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = not args.frozen
            backbone_params.append(param)
        else:
            other_params.append(param)

    if args.optimizer == "AdamW":
        opt_cls = torch.optim.AdamW
        opt_kwargs = {"weight_decay": args.weight_decay}
    else:
        opt_cls = torch.optim.SGD
        opt_kwargs = {"momentum": 0.9, "weight_decay": args.weight_decay}

    if args.multi_lr:
        optimizer = opt_cls([
            {"params": [p for p in backbone_params if p.requires_grad], "lr": args.lr},
            {"params": [p for p in other_params if p.requires_grad], "lr": args.lr * 5},
        ], **opt_kwargs)
    else:
        optimizer = opt_cls([p for p in model.parameters() if p.requires_grad], lr=args.lr, **opt_kwargs)

    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best_score = -float("inf")
    best_epoch = 0
    best_path = output_dir / "best_chbmit_window_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, args.clip_value
        )

        y_val, p_val, _, val_loss = predict(model, val_loader, criterion, device)
        val_metrics = compute_window_metrics(y_val, p_val, args.threshold)
        score = val_metrics[args.best_metric]

        print(f"\nEpoch {epoch}/{args.epochs} | train_loss={train_loss:.6f}")
        print_metrics("Val", val_metrics, loss=val_loss)

        if not np.isnan(score) and score > best_score:
            best_score = float(score)
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"Best {args.best_metric} improved to {best_score:.6f}; saved: {best_path}")

    print(f"\nTraining finished. Best epoch={best_epoch}, best {args.best_metric}={best_score:.6f}")
    print(f"Loading best checkpoint for final test: {best_path}")
    model = load_checkpoint(model, str(best_path), device, strict=True)

    y_test, p_test, test_paths, test_loss = predict(model, test_loader, criterion, device)
    test_metrics = compute_window_metrics(y_test, p_test, args.threshold)
    print_metrics("Test", test_metrics, loss=test_loss)

    metrics_path = output_dir / "test_window_metrics.json"
    save_metrics_json(test_metrics, str(metrics_path))
    if args.save_predictions:
        save_predictions_csv(test_paths, y_test, p_test, args.threshold, str(output_dir / "test_window_predictions.csv"))


if __name__ == "__main__":
    main()
