import os
import random
import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import random

def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask

def generate_channel_set_mask(bz, ch_num, patch_num, mask_ratio=0.15, group_size=3, device='cuda'):
    """
    基于EpilepsyFM的通道集掩码策略
    """
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    
    for batch_idx in range(bz):
        # 将通道分组
        num_groups = ch_num // group_size
        remaining_channels = ch_num % group_size
        
        # 掩码完整组
        if num_groups > 0:
            group_indices = torch.randperm(num_groups, device=device)
            num_masked_groups = max(1, int(num_groups * mask_ratio))
            masked_groups = group_indices[:num_masked_groups]
            
            for group_idx in masked_groups:
                start_ch = group_idx * group_size
                end_ch = start_ch + group_size
                mask[batch_idx, start_ch:end_ch, :] = 1
        
        # 掩码剩余通道
        if remaining_channels > 0:
            remaining_start = num_groups * group_size
            for ch_offset in range(remaining_channels):
                if torch.rand(1, device=device) < mask_ratio:
                    mask[batch_idx, remaining_start + ch_offset, :] = 1
    
    return mask

def symmetric_channel_set_mask(bz, ch_num, patch_num, mask_ratio=0.15, device='cuda'):
    """
    对称掩码策略：返回原掩码和逆掩码
    """
    mask1 = generate_channel_set_mask(bz, ch_num, patch_num, mask_ratio, device=device)
    mask2 = 1 - mask1  # 逆掩码
    return mask1, mask2

def generate_epilepsy_channel_set_mask(bz, ch_num, patch_num, mask_ratio=0.15, 
                                     group_size=3, symmetric=True, device='cuda'):
    """
    癫痫专用的通道集掩码策略 - 主接口函数
    """
    if symmetric:
        return symmetric_channel_set_mask(bz, ch_num, patch_num, mask_ratio, device)
    else:
        return generate_channel_set_mask(bz, ch_num, patch_num, mask_ratio, group_size, device)

def to_tensor(array):
    return torch.from_numpy(array).float()

if __name__ == '__main__':
    # 测试新的掩码策略
    mask = generate_channel_set_mask(2, 16, 30, mask_ratio=0.15, device='cpu')
    print("通道集掩码形状:", mask.shape)
    print("Batch 0的掩码分布:", mask[0].sum().item(), "/", mask[0].numel())