import os
import glob
import pickle
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ====== 根据你的工程结构导入 SeizurePredictionModel ======
# 确保 finetune_chbmit_pred.py 和这个脚本在同一工程下
from finetune_chbmit_pred import SeizurePredictionModel_eval


# --------------------------
# 1. 构造并加载模型
# --------------------------

def build_model(weight_path: str, device: torch.device):
    """
    weight_path: 训练完成后保存的 best_chbmit_pred.pt
    这个 pt 文件是整个 SeizurePredictionModel 的 state_dict。
    """
    # 1) 构造空壳模型（结构要和训练时一模一样）
    model = SeizurePredictionModel_eval(
        #pretrained_weight_path=None,   # eval 阶段不要再加载 TU 预训练
        d_model=200,
        num_classes=2,
    )

    # 2) 直接加载整个模型的 state_dict
    print(f"加载下游微调权重: {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


# --------------------------
# 2. 收集某个病人的所有窗口文件
# --------------------------

def collect_windows_for_patient(process2_test_dir, patient_id):
    """
    从 process_2_pred/test 下收集指定病人所有窗口文件，
    返回:
        record_files: dict[rec_base] = list[(start_idx, full_path)]
                      其中 rec_base 类似 'chb23_07.edf'
    """
    all_pkls = glob.glob(os.path.join(process2_test_dir, "*.pkl"))
    record_files = defaultdict(list)

    for path in all_pkls:
        fname = os.path.basename(path)  # 例如 chb23_07.edf-10240.pkl
        rec_part, start_part = fname.split("-")
        rec_base = rec_part              # 'chb23_07.edf'
        start_idx = int(os.path.splitext(start_part)[0])

        # 从 rec_base 中解析病人 id: 'chb23_07.edf' -> 'chb23'
        pat = rec_base.split("_")[0]
        if pat != patient_id:
            continue

        record_files[rec_base].append((start_idx, path))

    # 每个记录内部按起始 sample 排序
    for rec_base in record_files:
        record_files[rec_base].sort(key=lambda x: x[0])

    return record_files


# --------------------------
# 3. 在窗级别评估一个病人
# --------------------------

def evaluate_window_level(
    model,
    device,
    record_files,
    patch_size=200,
):
    """
    不做任何后处理，直接在窗级别评估分类效果。

    参数:
        model: 已加载权重的 SeizurePredictionModel
        device: cuda 或 cpu
        record_files: dict[rec_base] = list[(start_idx, pkl_path)]
        patch_size: 跟你训练 chb_pred_dataset 时用的一致（如果不为 200，请改这里）

    返回:
        y_true: 所有窗口的真实标签 (0/1)
        y_prob: 所有窗口的 preictal 概率
    """
    y_true = []
    y_prob = []

    for rec_base, win_list in record_files.items():
        print(f"处理记录: {rec_base}, 窗口数: {len(win_list)}")

        for start_idx, path in win_list:
            data = pickle.load(open(path, "rb"))
            x = data["X"].astype(np.float32)   # (C, T)
            y = int(data["y"])                 # 0/1 标签

            C, T = x.shape

            # ====== 这一段 patch 切法要和训练时保持一致 ======
            # 如果你在 chb_pred_dataset.__getitem__ 里面有更复杂的逻辑，
            # 建议直接复制那几行替换这一块。
            T_trunc = (T // patch_size) * patch_size
            x = x[:, :T_trunc]
            patch_num = T_trunc // patch_size
            x_tensor = torch.from_numpy(x).view(1, C, patch_num, patch_size).to(device)
            # ==================================================

            with torch.no_grad():
                logits = model(x_tensor)              # (1, 2)
                prob = F.softmax(logits, dim=-1)[0, 0].item()  # preictal 概率

            y_true.append(y)
            y_prob.append(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    return y_true, y_prob


def compute_metrics(y_true, y_prob, prob_threshold=0.5):
    """
    根据 y_true, y_prob 计算一堆指标
    """
    y_pred = (y_prob >= prob_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    # 处理 AUC 中可能出现的“只有一个类别”的情况
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")

    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["interictal(0)", "preictal(1)"])

    pos_ratio = y_true.mean()

    return {
        "acc": acc,
        "roc_auc": roc,
        "pr_auc": ap,
        "conf_mat": cm,
        "report": report,
        "pos_ratio": pos_ratio,
    }


# --------------------------
# 4. 命令行入口
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="Window-level evaluation for one CHB-MIT patient")
    parser.add_argument("--process2_root", type=str,
                        default=r"./data/processed/chbmit/process_2_pred_5_20",
                        help="第二阶段预测任务路径 (含 train/val/test)")
    parser.add_argument("--patient_id", type=str, default="chb21",
                        help="病人 ID, 例如 'chb23'")
    parser.add_argument("--checkpoint", type=str,
                        default=r"./checkpoints/chbmit_pred.pt",
                        help="窗级预测模型权重路径")
    parser.add_argument("--patch_size", type=int, default=200,
                        help="与训练 chb_pred_dataset 时使用的 patch_size 一致")
    parser.add_argument("--prob_threshold", type=float, default=0.5,
                        help="将概率二值化为 0/1 的阈值")
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 1) 构造并加载模型
    model = build_model(args.checkpoint, device)

    # 2) 收集该病人的所有窗口文件 (test split)
    process2_test_dir = os.path.join(args.process2_root, "val")
    record_files = collect_windows_for_patient(process2_test_dir, args.patient_id)
    if len(record_files) == 0:
        print(f"在 {process2_test_dir} 中没有找到病人 {args.patient_id} 的窗口文件.")
        return

    # 3) 窗级前向 & 收集概率/标签
    y_true, y_prob = evaluate_window_level(
        model=model,
        device=device,
        record_files=record_files,
        patch_size=args.patch_size,
    )

    # 4) 计算指标
    metrics = compute_metrics(y_true, y_prob, prob_threshold=args.prob_threshold)

    print("=====================================")
    print(f"病人: {args.patient_id}")
    print(f"总窗口数        : {len(y_true)}")
    print(f"正样本比例 (preictal=1): {metrics['pos_ratio']:.4f}")
    print(f"Accuracy        : {metrics['acc']:.4f}")
    print(f"ROC-AUC         : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC (preictal为正类): {metrics['pr_auc']:.4f}")
    print("混淆矩阵 (行: 真值, 列: 预测):")
    print(metrics["conf_mat"])
    print("分类报告:")
    print(metrics["report"])
    print("=====================================")


if __name__ == "__main__":
    main()
