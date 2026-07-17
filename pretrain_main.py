import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pretraining_dataset import PretrainingDataset
from models.db_epifm import DBEpiFM
from pretrain_trainer import Trainer


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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 0)')
    parser.add_argument('--parallel', type=str2bool, nargs='?', const=True, default=False, help='enable data parallelism (default: false)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='lr_scheduler: CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, CyclicLR')

    # 模型参数
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    
  
    parser.add_argument('--need_mask', type=str2bool, nargs='?', const=True, default=True, help='enable masked pretraining (default: true)')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='random temporal-spatial mask ratio')

    # Spatial-Frequency (band-token) MAE & alignment losses
    # (keep old arg name for compatibility)
    parser.add_argument('--freq_mask_ratio', type=float, default=0.15, help='band mask ratio (L_F)')
    parser.add_argument('--sf_fs', type=float, default=200.0, help='sampling rate for SF band features')
    parser.add_argument(
        '--sf_bands',
        type=str,
        default='0.5-4,4-8,8-13,13-30,30-45',
        help='comma-separated band ranges, e.g. "0.5-4,4-8,8-13,13-30,30-45"',
    )
    parser.add_argument('--lambda_f', type=float, default=1.0, help='weight for frequency reconstruction loss')
    parser.add_argument('--lambda_align', type=float, default=0.15, help='weight for alignment loss')

    parser.add_argument('--dataset_dir', type=str, default='./data/processed/pretraining/tuep_tusz_no_tuab_tuev_overlap.lmdb',
                        help='dataset_dir')
    parser.add_argument('--model_dir',   type=str,   default='./outputs/pretraining', help='model_dir')
    
    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    
    #创建数据加载器
    pretrained_dataset = PretrainingDataset(dataset_dir=params.dataset_dir)
    print(len(pretrained_dataset))
    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=0,
        shuffle=True,
    )
    def _parse_bands(spec: str):
        bands = []
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            lo, hi = part.split('-')
            bands.append((float(lo), float(hi)))
        return bands

    sf_bands = _parse_bands(params.sf_bands)

    # Use ST (no-FFT) backbone + SF (band-token) branch + alignment
    model = DBEpiFM(
        in_dim=params.in_dim,
        out_dim=params.out_dim,
        d_model=params.d_model,
        dim_feedforward=params.dim_feedforward,
        seq_len=params.seq_len,
        n_layer=params.n_layer,
        nhead=params.nhead,
        sf_fs=params.sf_fs,
        sf_bands=sf_bands,
    )
    trainer = Trainer(params, data_loader, model)
    trainer.train()
    pretrained_dataset.db.close()

if __name__ == '__main__':
    main()