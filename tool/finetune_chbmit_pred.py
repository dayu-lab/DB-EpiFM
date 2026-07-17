import os
import glob
import argparse
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.functional import softmax
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn as nn

def compute_class_weight_from_dataset(dataset):
    """
    dataset: CHBMITPredictionDataset(train)
    根据 train split 中 0/1 的数量自动算出 class weight
    """
    labels = []
    for path in dataset.files:      # dataset.files 在 __init__ 里已经存了所有 pkl 路径
        data = pickle.load(open(path, "rb"))
        labels.append(int(data["y"]))

    counter = Counter(labels)
    neg = counter.get(0, 0)
    pos = counter.get(1, 0)
    total = neg + pos

    # 避免除 0
    if neg == 0 or pos == 0:
        print("警告：某一类样本数量为 0，无法自动计算 class weight，退回到等权重 [1., 1.]")
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    # 一种常见做法：weight ∝ 1 / freq
    w0 = total / (2.0 * neg)
    w1 = total / (2.0 * pos)
    print(f"类别统计: neg={neg}, pos={pos} -> class_weight=[{w0:.3f}, {w1:.3f}]")

    return torch.tensor([w0, w1], dtype=torch.float32)

# =======================
# 1. Dataset: 读取预测任务数据
# =======================

class CHBMITPredictionDataset(Dataset):
    """
    读取 process_2_pred/{train,val,test} 下的 .pkl
    每个样本结构:
        {
            "X": (n_channels, n_samples),
            "y": 0/1,
            "time_to_next_seizure_min": float
        }
    """
    def __init__(self, root_dir, split="train"):
        """
        root_dir: 例如 ./data/processed/chbmit/process_2_pred
        split: 'train' / 'val' / 'test'
        """
        self.split_dir = os.path.join(root_dir, split)
        assert os.path.isdir(self.split_dir), f"{self.split_dir} 不存在"

        self.files = sorted(glob.glob(os.path.join(self.split_dir, "*.pkl")))
        assert len(self.files) > 0, f"{self.split_dir} 下没有 pkl 文件"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = pickle.load(open(path, "rb"))

        x = data["X"].astype(np.float32)  # (C, T)
        y = int(data["y"])

        # 这里不做任何标准化，和你原来的 detection 保持一致；
        # 如果原来在 Dataset 里有 z-score，可以照搬过来。
        x = torch.from_numpy(x)  # (C, T)

        # 你的 DB-EpiFM 输入是 (B, ch, patch_num, patch_size)
        # 如果你在原 chb_dataset 里已经实现了切 patch 的逻辑，
        # 可以直接把那段代码复制到这里。
        # 下面给一个通用示例：假设 patch_size=200，把时间维拆成若干 patch。
        # 如果你已有更准确的实现，请替换这段。

        # ------- 示例 patch 处理开始 -------
        patch_size = 200
        C, T = x.shape
        T_trunc = (T // patch_size) * patch_size
        x = x[:, :T_trunc]
        patch_num = T_trunc // patch_size
        x = x.view(C, patch_num, patch_size)   # (C, patch_num, patch_size)
        # ------- 示例 patch 处理结束 -------

        return x, y


# =======================
# 2. 模型：DB-EpiFM backbone + 线性分类头
# =======================

# 你项目里应该已经有 DB-EpiFM 定义，这里直接从 models.db_epifm 导入；
# 如有不同，请把 import 改成你自己的路径。

import torch
import torch.nn as nn
from models.db_epifm import DBEpiFM

class SeizurePredictionModel_eval(nn.Module):
    """
    DB-EpiFM backbone + 线性分类头
    注意：这里不再在 __init__ 里自动加载任何权重。
    预训练 TU 权重 / 下游微调权重都在外部脚本中 load_state_dict。
    """
    def __init__(self, d_model=200, num_classes=2):
        super().__init__()

        self.backbone = DBEpiFM(
            in_dim=d_model,
            out_dim=d_model,
            d_model=d_model,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x: (B, C, patch_num, patch_size)
        """
        feats = self.backbone(x)          # (B, C, patch_num, d_model)
        feats = feats.mean(dim=(1, 2))    # (B, d_model)
        logits = self.cls_head(feats)     # (B, num_classes)
        return logits

class SeizurePredictionModel(nn.Module):
    """
    使用预训练 DB-EpiFM 做特征提取，
    在 CHBMIT 预测任务上训练一个窗级二分类器 (preictal vs interictal)。
    """
    def __init__(self, pretrained_weight_path, d_model=200, num_classes=2, freeze_backbone=False):
        super().__init__()
        self.backbone = DBEpiFM(
            in_dim=d_model, out_dim=d_model, d_model=d_model,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )

        # 加载 TU 预训练权重
        print(f"加载预训练权重: {pretrained_weight_path}")
        state_dict = torch.load(pretrained_weight_path, map_location="cpu")
        # The new backbone contains additional Frequency-MAE modules; load non-strictly.
        self.backbone.load_state_dict(state_dict, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 窗级分类头：对 (B, ch, patch_num, d_model) 做全局平均池化
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x: (B, C, patch_num, patch_size)
        """
        feats = self.backbone(x)              # (B, C, patch_num, d_model)
        feats = feats.mean(dim=(1, 2))        # (B, d_model)
        logits = self.cls_head(feats)         # (B, 2)
        return logits

# =======================
# 3. 训练 & 验证循环
# =======================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)         # (B, C, patch_num, patch_size)
        y = y.to(device)         # (B,)

        optimizer.zero_grad()
        logits = model(x)        # (B, 2)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, split="Val"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in tqdm(loader, desc=split, leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


# =======================
# 4. 主函数：组装一切
# =======================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DB-EpiFM on CHB-MIT prediction task")
    parser.add_argument("--datasets_dir", type=str,
                        default=r"./data/processed/chbmit/process_2_pred_5_20",
                        help="预处理后的预测数据根目录 (包含 train/val/test)")
    parser.add_argument("--pretrained_weight", type=str,
                        default=r"./checkpoints/db_epifm_pretrained.pth",
                        help="TU 预训练权重路径")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="是否冻结 DB-EpiFM，仅训练分类头")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 1) 构造 Dataset & DataLoader
    train_set = CHBMITPredictionDataset(args.datasets_dir, split="train")
    val_set   = CHBMITPredictionDataset(args.datasets_dir, split="val")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # 2) 模型
    model = SeizurePredictionModel(
        pretrained_weight_path=args.pretrained_weight,
        d_model=200,
        num_classes=2,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # 3) 损失函数：考虑类别不平衡，用 class_weight
    # class_weight = compute_class_weight_from_dataset(train_set).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weight)
    # 直接写死 class_weight，跳过统计
    class_weight = torch.tensor([0.515, 17.629], dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.datasets_dir, "STSF_best_chbmit_pred.pt")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, split="Val")

        print(f"Train  loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val    loss={val_loss:.4f}, acc={val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Val acc 提升到 {best_val_acc:.4f}，已保存权重到 {best_ckpt_path}")

    print("训练结束，最佳验证集准确率:", best_val_acc)


if __name__ == "__main__":
    main()
