import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd

def analyze_dataset_structure(root_folder):
    """
    分析数据集结构，统计各文件夹中的文件数量和标签分布
    """
    print("=" * 60)
    print("数据集结构分析")
    print("=" * 60)
    
    results = {}
    
    # 检查根目录下的子文件夹
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_folder, split)
        if os.path.exists(split_path):
            print(f"\n分析 {split} 文件夹:")
            print("-" * 40)
            
            # 统计文件数量
            files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
            print(f"总文件数: {len(files)}")
            
            # 统计标签分布
            labels = []
            file_sizes = []
            
            for file in files:
                file_path = os.path.join(split_path, file)
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        labels.append(data['y'])
                        file_sizes.append(data['X'].shape)
                except Exception as e:
                    print(f"读取文件 {file} 时出错: {e}")
            
            if labels:
                label_counts = Counter(labels)
                total_files = len(labels)
                
                print(f"标签分布:")
                for label, count in sorted(label_counts.items()):
                    percentage = (count / total_files) * 100
                    print(f"  标签 {label}: {count} 个文件 ({percentage:.2f}%)")
                
                # 检查数据形状
                unique_shapes = Counter(file_sizes)
                print(f"数据形状分布:")
                for shape, count in unique_shapes.most_common(5):
                    print(f"  {shape}: {count} 个文件")
                
                results[split] = {
                    'total_files': total_files,
                    'label_distribution': dict(label_counts),
                    'data_shapes': dict(unique_shapes)
                }
            else:
                print("未找到有效的pkl文件")
                results[split] = {'total_files': 0, 'label_distribution': {}, 'data_shapes': {}}
        else:
            print(f"\n{split} 文件夹不存在: {split_path}")
            results[split] = {'total_files': 0, 'label_distribution': {}, 'data_shapes': {}}
    
    return results

def analyze_patient_distribution(root_folder):
    """
    分析患者在不同数据集划分中的分布
    """
    print("\n" + "=" * 60)
    print("患者分布分析")
    print("=" * 60)
    
    patient_distribution = defaultdict(lambda: defaultdict(int))
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_folder, split)
        if os.path.exists(split_path):
            files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
            
            for file in files:
                # 从文件名提取患者ID (例如: chb01_01-0.pkl -> chb01)
                patient_id = file.split('_')[0]
                
                file_path = os.path.join(split_path, file)
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        label = data['y']
                        patient_distribution[patient_id][f'{split}_total'] += 1
                        patient_distribution[patient_id][f'{split}_label_{label}'] += 1
                except Exception as e:
                    print(f"读取文件 {file} 时出错: {e}")
    
    # 打印患者分布
    df_patients = []
    for patient, stats in patient_distribution.items():
        row = {'patient': patient}
        for split in ['train', 'val', 'test']:
            total = stats.get(f'{split}_total', 0)
            label_0 = stats.get(f'{split}_label_0', 0)
            label_1 = stats.get(f'{split}_label_1', 0)
            
            row[f'{split}_total'] = total
            row[f'{split}_label_0'] = label_0
            row[f'{split}_label_1'] = label_1
            if total > 0:
                row[f'{split}_label_1_pct'] = (label_1 / total) * 100
            else:
                row[f'{split}_label_1_pct'] = 0
        
        df_patients.append(row)
    
    if df_patients:
        df = pd.DataFrame(df_patients)
        print("\n患者分布详情:")
        print(df.to_string(index=False))
        
        return df
    else:
        print("未找到患者分布数据")
        return None

def plot_label_distribution(results, output_path=None):
    """
    绘制标签分布图
    """
    splits = ['train', 'val', 'test']
    labels_0 = []
    labels_1 = []
    
    for split in splits:
        if split in results and 'label_distribution' in results[split]:
            dist = results[split]['label_distribution']
            labels_0.append(dist.get(0, 0))
            labels_1.append(dist.get(1, 0))
        else:
            labels_0.append(0)
            labels_1.append(0)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 柱状图
    x = np.arange(len(splits))
    width = 0.35
    
    ax1.bar(x - width/2, labels_0, width, label='标签 0 (正常)', alpha=0.8)
    ax1.bar(x + width/2, labels_1, width, label='标签 1 (癫痫)', alpha=0.8)
    
    ax1.set_xlabel('数据集划分')
    ax1.set_ylabel('文件数量')
    ax1.set_title('各数据集划分中的标签分布')
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.legend()
    
    # 在柱子上添加数值
    for i, v in enumerate(labels_0):
        ax1.text(i - width/2, v + max(labels_0 + labels_1)*0.01, str(v), ha='center')
    for i, v in enumerate(labels_1):
        ax1.text(i + width/2, v + max(labels_0 + labels_1)*0.01, str(v), ha='center')
    
    # 饼图
    total_0 = sum(labels_0)
    total_1 = sum(labels_1)
    total = total_0 + total_1
    
    if total > 0:
        ax2.pie([total_0, total_1], 
                labels=[f'正常 ({total_0}, {total_0/total*100:.1f}%)', 
                       f'癫痫 ({total_1}, {total_1/total*100:.1f}%)'],
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('整体标签分布')
    else:
        ax2.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('整体标签分布 (无数据)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {output_path}")
    
    plt.show()

def generate_detailed_report(results, df_patients, root_folder):
    """
    生成详细的分析报告
    """
    print("\n" + "=" * 60)
    print("详细分析报告")
    print("=" * 60)
    
    total_files = 0
    total_label_0 = 0
    total_label_1 = 0
    
    for split in ['train', 'val', 'test']:
        if split in results:
            split_data = results[split]
            total_files += split_data['total_files']
            dist = split_data['label_distribution']
            total_label_0 += dist.get(0, 0)
            total_label_1 += dist.get(1, 0)
    
    print(f"数据集总文件数: {total_files}")
    print(f"正常片段 (标签 0): {total_label_0} ({total_label_0/total_files*100:.2f}%)")
    print(f"癫痫片段 (标签 1): {total_label_1} ({total_label_1/total_files*100:.2f}%)")
    
    if total_files > 0:
        imbalance_ratio = max(total_label_0, total_label_1) / min(total_label_0, total_label_1)
        print(f"类别不平衡比例: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 5:
            print("警告: 数据集存在严重的类别不平衡问题!")
        elif imbalance_ratio > 3:
            print("注意: 数据集存在明显的类别不平衡问题")
        else:
            print("数据集类别分布相对平衡")
    
    # 检查数据形状一致性
    print("\n数据形状检查:")
    all_shapes = set()
    for split in ['train', 'val', 'test']:
        if split in results and 'data_shapes' in results[split]:
            shapes = set(results[split]['data_shapes'].keys())
            all_shapes.update(shapes)
    
    if len(all_shapes) == 1:
        print(f"所有数据形状一致: {list(all_shapes)[0]}")
    else:
        print(f"发现 {len(all_shapes)} 种不同的数据形状:")
        for shape in all_shapes:
            print(f"  {shape}")

def main():
    """
    主函数 - 分析数据集
    """
    # 设置数据集路径
    dataset_path = "./data/processed/chbmit/process_2"
    
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        print("请确保路径正确，并且已经运行了预处理脚本")
        return
    
    print(f"分析数据集: {dataset_path}")
    
    # 1. 分析数据集结构
    results = analyze_dataset_structure(dataset_path)
    
    # 2. 分析患者分布
    df_patients = analyze_patient_distribution(dataset_path)
    
    # 3. 生成详细报告
    generate_detailed_report(results, df_patients, dataset_path)
    
    # 4. 绘制标签分布图
    plot_label_distribution(results, os.path.join(dataset_path, "label_distribution.png"))
    
    # 5. 保存详细统计结果到CSV
    if df_patients is not None:
        csv_path = os.path.join(dataset_path, "dataset_statistics.csv")
        df_patients.to_csv(csv_path, index=False)
        print(f"\n详细统计已保存到: {csv_path}")

if __name__ == "__main__":
    main()