import argparse
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from datasets import tuab_dataset
from models import model_for_tuab


def setup_seed(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_state_dict_safely(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_ckpt[k] = v
    missing, unexpected = model.load_state_dict(new_ckpt, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print("Unexpected key examples:", unexpected[:5])
    if len(missing) > 0:
        print("Missing key examples:", missing[:5])


def extract_features(model, loader, device, max_points=None):
    model.eval()
    all_feats, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.detach().cpu().numpy().reshape(-1)

            # model.backbone(x): [B, C, S, D], e.g., [B, 16, 10, 200]
            patch_feats = model.backbone(x)

            # Recommended sample-level representation for t-SNE: [B, D]
            pooled_feats = patch_feats.mean(dim=(1, 2))

            logits = model.classifier(patch_feats)
            scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

            all_feats.append(pooled_feats.detach().cpu().numpy())
            all_labels.append(y)
            all_scores.append(scores)

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(int)
    scores = np.concatenate(all_scores, axis=0)

    if max_points is not None and feats.shape[0] > max_points:
        rng = np.random.default_rng(3407)
        idx = rng.choice(feats.shape[0], size=max_points, replace=False)
        feats, labels, scores = feats[idx], labels[idx], scores[idx]

    return feats, labels, scores


def run_tsne(feats, seed=3407, use_pca=True):
    feats = StandardScaler().fit_transform(feats)

    if use_pca and feats.shape[1] > 50:
        feats = PCA(n_components=50, random_state=seed).fit_transform(feats)

    n = feats.shape[0]
    perplexity = min(30, max(5, (n - 1) // 10))
    print(f"Running t-SNE: n={n}, dim={feats.shape[1]}, perplexity={perplexity}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1000,
    )
    return tsne.fit_transform(feats)


def plot_tsne(features, labels, out_path):
    """
    features: [N, D]
    labels:   [N], 0=Normal, 1=Abnormal
    """

    # 标准化
    features = StandardScaler().fit_transform(features)

    # PCA 预降维
    pca = PCA(n_components=min(50, features.shape[1]), random_state=42)
    features_pca = pca.fit_transform(features)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        early_exaggeration=18,
        init='pca',
        random_state=42
    )
    tsne_result = tsne.fit_transform(features_pca)

    # ===== 自动方向优化：Abnormal -> 左下, Normal -> 右上 =====
    normal_pts = tsne_result[labels == 0]
    abnormal_pts = tsne_result[labels == 1]

    normal_center = normal_pts.mean(axis=0)
    abnormal_center = abnormal_pts.mean(axis=0)

    # 调整 x 方向
    if abnormal_center[0] > normal_center[0]:
        tsne_result[:, 0] = -tsne_result[:, 0]

    # 调整 y 方向
    if abnormal_center[1] > normal_center[1]:
        tsne_result[:, 1] = -tsne_result[:, 1]

    # 重新取点
    normal_pts = tsne_result[labels == 0]
    abnormal_pts = tsne_result[labels == 1]

    # 画图
    plt.figure(figsize=(6, 6))
    plt.scatter(
        normal_pts[:, 0], normal_pts[:, 1],
        s=12, alpha=0.75, label='Normal'
    )
    plt.scatter(
        abnormal_pts[:, 0], abnormal_pts[:, 1],
        s=12, alpha=0.75, label='Abnormal'
    )

    plt.title("DB-EpiFM")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace('.png', '.pdf'))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default="./data/processed/tuab")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/tuab_finetuned.pth", help="Fine-tuned TUAB .pth weight")
    parser.add_argument("--foundation_dir", type=str, default="", help="Only used to satisfy the model init; can equal checkpoint if needed")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--classifier", type=str, default="all_patch_reps",
                        choices=["all_patch_reps", "all_patch_reps_twolayer", "all_patch_reps_onelayer", "avgpooling_patch_reps"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_points", type=int, default=3000)
    parser.add_argument("--out_dir", type=str, default="./tsne_tuab")
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # Make args compatible with your existing Model(params)
    args.use_pretrained_weights = False
    args.downstream_dataset = "TUAB"
    args.num_of_classes = 2
    args.lr = 1e-5
    args.weight_decay = 1e-2
    args.optimizer = "AdamW"
    args.clip_value = 1
    args.label_smoothing = 0.0
    args.multi_lr = True
    args.frozen = False
    args.model_dir = args.out_dir

    load_dataset = tuab_dataset.LoadDataset(args)
    data_loader = load_dataset.get_data_loader()[args.split]

    model = model_for_tuab.Model(args).to(device)
    load_state_dict_safely(model, args.checkpoint, device)

    feats, labels, scores = extract_features(model, data_loader, device, max_points=args.max_points)
    print("Feature shape:", feats.shape)
    print("Label distribution:", {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})

    z = run_tsne(feats, seed=args.seed)

    npz_path = os.path.join(args.out_dir, f"tuab_{args.split}_features_tsne.npz")
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(npz_path, feats=feats, labels=labels, scores=scores, tsne=z)
    print(f"Saved data: {npz_path}")

    out_png = os.path.join(args.out_dir, f"tuab_{args.split}_tsne.png")
    plot_tsne(z, labels, out_png)


if __name__ == "__main__":
    main()
