import argparse
import random

import numpy as np
import torch

from datasets import  tuab_dataset, tusl_dataset,chb_dataset, tuev_dataset
from finetune_trainer import Trainer
from models import model_for_tuab, model_for_tusl,model_for_chb, model_for_tuev

from pathlib import Path

def str2bool(value):
    """Parse common command-line Boolean representations safely."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a Boolean value, but received {value!r}. "
        "Use true/false, yes/no, on/off, or 1/0."
    )


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value,default=1')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')

    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='TUEV',
                        help='[ TUAB, TUSL, TUSZ,CHB-MIT,TUEV]')
    parser.add_argument('--datasets_dir', type=str,
                        default='./data/processed/tuev_3class',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=3, help='number of classes')
    #微调后的模型权重保存路径
    parser.add_argument('--model_dir', type=str, default='./outputs/finetuning/tuev_0.15_0.15', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=str2bool, nargs='?', const=True, default=True,
                        help='use different learning rates for different modules (default: true)')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=str2bool, nargs='?', const=True, default=False, help='freeze the pretrained backbone (default: false)')
    parser.add_argument('--use_pretrained_weights', type=str2bool, nargs='?', const=True,
                        default=True, help='load pretrained backbone weights (default: true)')
    #输入预训练出来的权重
    parser.add_argument('--foundation_dir', type=str,
                        default='./checkpoints/DB-EpiFM_pretrain.pth',
                        help='foundation_dir')

    params = parser.parse_args()
    if params.use_pretrained_weights:
    checkpoint_path = Path(params.foundation_dir).expanduser()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "The pretrained DB-EpiFM checkpoint was not found at "
            f"{checkpoint_path}. Run "
            "`python scripts/download_pretrained.py` "
            "or provide the correct path with `--foundation_dir`."
        )

    params.foundation_dir = str(checkpoint_path)
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))

    # 根据数据集选择对应的数据加载器和模型
    if params.downstream_dataset == 'TUAB':
        load_dataset = tuab_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuab.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'CHB-MIT':# 添加CHB分支
        load_dataset = chb_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_chb.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()  

    elif params.downstream_dataset == 'TUEV':
        load_dataset = tuev_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuev.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    else:
        raise ValueError(f"不支持的dataset: {params.downstream_dataset}")
    



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
