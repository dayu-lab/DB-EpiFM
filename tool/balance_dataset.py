import os
import pickle
import numpy as np
import shutil
from collections import defaultdict, Counter
import random
import argparse

def analyze_dataset_balance(root_folder):
    """
    分析数据集的类别平衡情况
    """
    balance_info = {}
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_folder, split)
        if not os.path.exists(split_path):
            continue
            
        files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
        labels = []
        
        for file in files:
            file_path = os.path.join(split_path, file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    labels.append(data['y'])
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
        
        if labels:
            label_counts = Counter(labels)
            total = len(labels)
            balance_info[split] = {
                'total': total,
                'label_0': label_counts.get(0, 0),
                'label_1': label_counts.get(1, 0),
                'imbalance_ratio': label_counts.get(0, 1) / max(1, label_counts.get(1, 1))
            }
    
    return balance_info

def create_stratified_sample(source_folder, target_folder, sample_ratio, target_ratio, seed):
    """
    创建分层抽样数据集，确保每个患者和每个类别的代表性
    
    参数:
    - source_folder: 源数据文件夹
    - target_folder: 目标数据文件夹
    - sample_ratio: 抽样比例 (相对于原始数据集)
    - target_ratio: 目标不平衡比例 (正常:癫痫)
    - seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建目标文件夹
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_folder, split), exist_ok=True)
    
    print("开始创建分层抽样数据集...")
    print(f"抽样比例: {sample_ratio*100}%")
    print(f"目标不平衡比例: {target_ratio}:1 (正常:癫痫)")
    
    for split in ['train', 'val', 'test']:
        source_split_path = os.path.join(source_folder, split)
        target_split_path = os.path.join(target_folder, split)
        
        if not os.path.exists(source_split_path):
            print(f"跳过不存在的文件夹: {source_split_path}")
            continue
            
        files = [f for f in os.listdir(source_split_path) if f.endswith('.pkl')]
        
        # 按患者和标签分类文件
        patient_files = defaultdict(lambda: {'label_0': [], 'label_1': []})
        
        for file in files:
            file_path = os.path.join(source_split_path, file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    # 从文件名提取患者ID
                    patient_id = file.split('_')[0]
                    if data['y'] == 0:
                        patient_files[patient_id]['label_0'].append(file)
                    else:
                        patient_files[patient_id]['label_1'].append(file)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
        
        print(f"/n{split} 文件夹:")
        total_label_0 = 0
        total_label_1 = 0
        
        # 为每个患者进行分层抽样
        for patient_id, files_dict in patient_files.items():
            label_0_count = len(files_dict['label_0'])
            label_1_count = len(files_dict['label_1'])
            
            # 计算该患者需要抽样的正常样本数量
            # 确保至少保留一些正常样本，但不超过目标比例
            target_label_0_count = min(
                max(1, int(label_0_count * sample_ratio)),
                int(label_1_count * target_ratio) if label_1_count > 0 else int(label_0_count * sample_ratio)
            )
            
            # 确保不超过可用样本数
            target_label_0_count = min(target_label_0_count, label_0_count)
            
            # 随机选择要保留的正常样本
            selected_label_0_files = random.sample(files_dict['label_0'], target_label_0_count)
            
            # 保留所有癫痫样本（因为它们很少）
            selected_label_1_files = files_dict['label_1']
            
            # 复制选择的样本到目标文件夹
            for file in selected_label_0_files + selected_label_1_files:
                source_path = os.path.join(source_split_path, file)
                target_path = os.path.join(target_split_path, file)
                shutil.copy2(source_path, target_path)
            
            total_label_0 += len(selected_label_0_files)
            total_label_1 += len(selected_label_1_files)
            
            print(f"  患者 {patient_id}: 正常 {len(selected_label_0_files)}/{label_0_count}, "
                  f"癫痫 {len(selected_label_1_files)}/{label_1_count}")
        
        print(f"  总计 - 正常: {total_label_0}, 癫痫: {total_label_1}")
        if total_label_1 > 0:
            print(f"  不平衡比例: {total_label_0/total_label_1:.2f}:1")
        else:
            print(f"  警告: 没有癫痫样本!")
    
    print(f"/n分层抽样数据集已保存到: {target_folder}")

def generate_dataset_report(dataset_folder):
    """
    生成数据集的详细报告
    """
    print("/n" + "=" * 60)
    print("数据集详细报告")
    print("=" * 60)
    
    total_files = 0
    total_label_0 = 0
    total_label_1 = 0
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_folder, split)
        if not os.path.exists(split_path):
            continue
            
        files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
        labels = []
        
        for file in files:
            file_path = os.path.join(split_path, file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    labels.append(data['y'])
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
        
        if labels:
            label_counts = Counter(labels)
            split_total = len(labels)
            split_label_0 = label_counts.get(0, 0)
            split_label_1 = label_counts.get(1, 0)
            
            total_files += split_total
            total_label_0 += split_label_0
            total_label_1 += split_label_1
            
            print(f"/n{split} 文件夹:")
            print(f"  总文件数: {split_total}")
            print(f"  正常样本: {split_label_0} ({split_label_0/split_total*100:.2f}%)")
            print(f"  癫痫样本: {split_label_1} ({split_label_1/split_total*100:.2f}%)")
            if split_label_1 > 0:
                print(f"  不平衡比例: {split_label_0/split_label_1:.2f}:1")
    
    print(f"/n整体统计:")
    print(f"  总文件数: {total_files}")
    print(f"  正常样本: {total_label_0} ({total_label_0/total_files*100:.2f}%)")
    print(f"  癫痫样本: {total_label_1} ({total_label_1/total_files*100:.2f}%)")
    if total_label_1 > 0:
        imbalance_ratio = total_label_0 / total_label_1
        print(f"  不平衡比例: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print("  警告: 数据集仍然存在严重的类别不平衡!")
        elif imbalance_ratio > 5:
            print("  注意: 数据集存在明显的类别不平衡")
        else:
            print("  数据集类别分布相对平衡")
    else:
        print("  错误: 没有癫痫样本!")

def main():
    parser = argparse.ArgumentParser(description='使用分层抽样方法平衡癫痫检测数据集')
    parser.add_argument('--source', type=str,default='./data/processed/chbmit/process_2',
                        help='源数据文件夹路径')
    parser.add_argument('--target', type=str, default='./data/processed/chbmit/balanced_dataset',
                        help='目标数据文件夹路径')
    parser.add_argument('--target_ratio', type=float, default=2.0,
                        help='目标不平衡比例 (正常:癫痫)，默认5.0 (5:1)')
    parser.add_argument('--sample_ratio', type=float, default=0.05,
                        help='正常样本的抽样比例，默认0.1 (10%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，默认42')
    
    args = parser.parse_args()
    
    # 检查源文件夹是否存在
    if not os.path.exists(args.source):
        print(f"错误: 源文件夹不存在: {args.source}")
        return
    
    # 分析原始数据集
    print("=" * 60)
    print("分析原始数据集")
    print("=" * 60)
    balance_info = analyze_dataset_balance(args.source)
    for split, info in balance_info.items():
        print(f"{split}: 正常 {info['label_0']}, 癫痫 {info['label_1']}, "
              f"比例 {info['imbalance_ratio']:.2f}:1")
    
    # 使用分层抽样方法平衡数据集
    print("/n" + "=" * 60)
    print("开始分层抽样平衡")
    print("=" * 60)
    
    create_stratified_sample(
        args.source, args.target,
        args.sample_ratio,
        args.target_ratio,
        args.seed
    )
    
    # 分析平衡后的数据集
    print("/n" + "=" * 60)
    print("平衡后数据集分析")
    print("=" * 60)
    
    generate_dataset_report(args.target)

if __name__ == "__main__":
    main()