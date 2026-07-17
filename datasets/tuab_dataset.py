import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from scipy import signal

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.mode = mode
        self.data_dir = os.path.join(data_dir, mode)
        
        # 检查目录是否存在
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")
        
        self.files = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]
        
        # 添加详细的数据检查
        print(f"\n=== {mode} 数据集检查 ===")
        print(f"数据目录: {self.data_dir}")
        print(f"文件数量: {len(self.files)}")
        
        # 检查前几个文件的标签分布
        self.labels = []
        for i, file in enumerate(self.files[:10]):  # 只检查前10个文件避免过载
            try:
                data_dict = pickle.load(open(file, 'rb'))
                label = data_dict['y']
                self.labels.append(label)
                if i < 3:  # 打印前3个文件的详细信息
                    print(f"文件 {file}: 数据形状 {data_dict['X'].shape}, 标签 {label}")
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
        
        if self.labels:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print(f"{mode}集前10个样本标签分布: {dict(zip(unique_labels, counts))}")
        
        # # 完整统计所有文件的标签
        # print(f"正在完整统计{mode}集标签分布...")
        # self.all_labels = []
        # for file in self.files:
        #     try:
        #         data_dict = pickle.load(open(file, 'rb'))
        #         self.all_labels.append(data_dict['y'])
        #     except:
        #         continue
        
        # if self.all_labels:
        #     unique_labels, counts = np.unique(self.all_labels, return_counts=True)
        #     print(f"{mode}集完整标签分布: {dict(zip(unique_labels, counts))}")
            
        #     # 检查是否只有单一类别
        #     if len(unique_labels) == 1:
        #         print(f"❌ 严重问题: {mode}集只有单一类别 {unique_labels[0]}!")
        #     else:
        #         print(f"✅ {mode}集包含 {len(unique_labels)} 个类别")


    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['X']
        label = data_dict['y']
        # data = signal.resample(data, 2000, axis=-1)
        data = data.reshape(16, 10, 200)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
        
        # 检查主数据目录
        print(f"\n=== 主数据目录检查 ===")
        print(f"数据根目录: {self.datasets_dir}")
        if not os.path.exists(self.datasets_dir):
            print(f"❌ 数据根目录不存在!")
            return
            
        subdirs = ['train', 'val', 'test']
        for subdir in subdirs:
            subdir_path = os.path.join(self.datasets_dir, subdir)
            if os.path.exists(subdir_path):
                file_count = len(os.listdir(subdir_path))
                print(f"{subdir} 目录: 存在, 文件数: {file_count}")
            else:
                print(f"{subdir} 目录: ❌ 不存在")

    def get_data_loader(self):
        print("\n" + "="*50)
        print("开始创建数据加载器...")
        print("="*50)
        
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        
        print(f"\n=== 数据集大小汇总 ===")
        print(f"训练集: {len(train_set)} 样本")
        print(f"验证集: {len(val_set)} 样本") 
        print(f"测试集: {len(test_set)} 样本")
        print(f"总计: {len(train_set) + len(val_set) + len(test_set)} 样本")
        
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        
        # 最终检查
        print(f"\n=== 数据加载器创建完成 ===")
        return data_loader