import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.util import to_tensor


class TUEVDataset(Dataset):
    def __init__(self, data_dir, mode='train', scale=100.0):
        super(TUEVDataset, self).__init__()
        self.mode = mode
        self.scale = scale

        split_map = {
            'train': 'processed_train',
            'val': 'processed_eval',
            'test': 'processed_test'
        }

        if mode not in split_map:
            raise ValueError(f"Unsupported mode: {mode}")

        self.data_dir = os.path.join(data_dir, split_map[mode])

        if not os.path.exists(self.data_dir):
            raise ValueError(f"TUEV数据目录不存在: {self.data_dir}")

        self.files = [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith('.pkl')
        ]
        self.files.sort()

        print(f"\n=== TUEV {mode} 数据集检查 ===")
        print(f"数据目录: {self.data_dir}")
        print(f"文件数量: {len(self.files)}")

        if len(self.files) == 0:
            raise ValueError(f"在 {self.data_dir} 中没有找到 pkl 文件")

        self.labels = []
        for i, file in enumerate(self.files[:10]):
            try:
                data_dict = pickle.load(open(file, 'rb'))
                label = int(data_dict['label'])
                self.labels.append(label)
                if i < 3:
                    print(
                        f"文件 {os.path.basename(file)}: "
                        f"signal形状 {data_dict['signal'].shape}, "
                        f"label {label}, "
                        f"label_tuev {data_dict.get('label_tuev', 'N/A')}"
                    )
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

        if len(self.labels) > 0:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print(f"TUEV {mode} 集前10个样本标签分布: {dict(zip(unique_labels, counts))}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        try:
            data_dict = pickle.load(open(file, 'rb'))

            # 读取预处理后的10秒窗 EEG
            data = data_dict['signal']   # (16, 2000)
            label = int(data_dict['label'])  # 0~5

            # reshape 成 backbone 需要的输入
            data = data.reshape(16, 10, 200).astype(np.float32)

            # 和 TUAB/TUSZ 保持量级一致，避免数值过大
            if self.scale is not None and self.scale > 0:
                data = data / self.scale

            return data, label

        except Exception as e:
            print(f"警告: 文件 {file} 加载失败: {e}")
            fallback_idx = random.randint(0, len(self.files) - 1)
            return self.__getitem__(fallback_idx)

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch], dtype=np.float32)
        y_label = np.array([x[1] for x in batch], dtype=np.int64)
        return to_tensor(x_data), torch.from_numpy(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

        print(f"\n=== TUEV主数据目录检查 ===")
        print(f"TUEV数据根目录: {self.datasets_dir}")

        if not os.path.exists(self.datasets_dir):
            print("❌ TUEV数据根目录不存在!")
            return

        subdirs = ['processed_train', 'processed_eval', 'processed_test']
        for subdir in subdirs:
            subdir_path = os.path.join(self.datasets_dir, subdir)
            if os.path.exists(subdir_path):
                file_count = len([f for f in os.listdir(subdir_path) if f.endswith('.pkl')])
                print(f"{subdir} 目录: 存在, 文件数: {file_count}")
            else:
                print(f"{subdir} 目录: ❌ 不存在")

    def get_data_loader(self):
        print("\n" + "=" * 50)
        print("开始创建TUEV数据加载器...")
        print("=" * 50)

        train_set = TUEVDataset(self.datasets_dir, mode='train')
        val_set = TUEVDataset(self.datasets_dir, mode='val')
        test_set = TUEVDataset(self.datasets_dir, mode='test')

        print(f"\n=== TUEV数据集大小汇总 ===")
        print(f"TUEV训练集: {len(train_set)} 样本")
        print(f"TUEV验证集: {len(val_set)} 样本")
        print(f"TUEV测试集: {len(test_set)} 样本")
        print(f"TUEV总计: {len(train_set) + len(val_set) + len(test_set)} 样本")

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=self.params.num_workers,
                pin_memory=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=True,
            ),
        }

        print(f"\n=== TUEV数据加载器创建完成 ===")
        return data_loader