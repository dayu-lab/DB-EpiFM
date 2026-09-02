import os
import pickle
import numpy as np
from collections import defaultdict
import pyedflib
import pyedflib.highlevel as hl
import multiprocessing as mp
from tqdm import tqdm

# ============================
# 第一阶段：原始EDF文件处理
# ============================

def process_metadata(summary, filename):
    """处理元数据，提取癫痫发作信息"""
    f = open(summary, "r")

    metadata = {}
    lines = f.readlines()
    times = []
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 3 and line[2] == filename:
            j = i + 1
            processed = False
            while not processed:
                if lines[j].split()[0] == "Number":
                    seizures = int(lines[j].split()[-1])
                    processed = True
                j = j + 1

            # 如果文件有癫痫发作，获取开始和结束时间
            if seizures > 0:
                j = i + 1
                for s in range(seizures):
                    processed = False
                    while not processed:
                        l = lines[j].split()
                        if l[0] == "Seizure" and "Start" in l:
                            start = int(l[-2]) * 256 - 1  # 开始时间索引
                            end = int(lines[j + 1].split()[-2]) * 256 - 1  # 结束时间索引
                            processed = True
                        j = j + 1
                    times.append((start, end))

            metadata["seizures"] = seizures
            metadata["times"] = times

    return metadata

def drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None):
    """从EDF文件中保留指定通道，移除其他通道"""
    signals, signal_headers, header = hl.read_edf(
        edf_source, ch_nrs=to_keep, digital=False
    )
    clean_file = {}
    for signal, header in zip(signals, signal_headers):
        channel = header.get("label")
        if channel in clean_file.keys():
            channel = channel + "-2"
        clean_file[channel] = signal
    return clean_file

def move_channels(clean_dict, channels, target):
    """保留有效通道并保存为pickle文件"""
    # 只保留有效通道
    keys_to_delete = []
    for key in clean_dict:
        if key != "metadata" and key not in channels.keys():
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del clean_dict[key]

    # 获取numpy数组大小
    size = 0
    for item in clean_dict.keys():
        if item != "metadata":
            size = len(clean_dict.get(item))
            break

    # 为缺失的通道创建零数组
    for k in channels.keys():
        if k not in clean_dict.keys():
            clean_dict[k] = np.zeros(size, dtype=float)

    # 保存为pickle文件
    pickle.dump(clean_dict, open(target + ".pkl", "wb"))

def process_files(pacient, valid_channels, channels, start, end, signals_path, clean_path):
    """处理患者的EDF文件"""
    for num in range(start, end + 1):
        to_keep = []

        num_str = ("0" + str(num))[-2:]
        filename = "{path}/chb{p}/chb{p}_{n}.edf".format(
            path=signals_path, p=pacient, n=num_str
        )

        # 检查参考文件，确定需要保留的通道
        try:
            signals, signal_headers, header = hl.read_edf(filename, digital=False)
            n = 0
            for h in signal_headers:
                if h.get("label") in valid_channels:
                    if n not in to_keep:
                        to_keep.append(n)
                n = n + 1

        except OSError:
            print("****************************************")
            print("警告 - 无需担心")
            print("文件", filename, "不存在。/n处理下一个文件。")
            print("****************************************")
            continue

        if len(to_keep) > 0:
            try:
                print(
                    "从文件",
                    "chb{p}_{n}.edf".format(p=pacient, n=num_str),
                    "中移除",
                    len(signal_headers) - len(to_keep),
                    "个通道"
                )
                clean_dict = drop_channels(
                    filename,
                    edf_target="{path}/chb{p}/chb{p}_{n}.edf".format(
                        path=clean_path, p=pacient, n=num_str
                    ),
                    to_keep=to_keep,
                )
                print("处理文件 ", filename)
            except AssertionError:
                print("****************************************")
                print("警告 - 无需担心")
                print("文件", filename, "不存在。/n处理下一个文件。")
                print("****************************************")
                continue

        # 处理元数据
        metadata = process_metadata(
            "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient),
            "chb{p}_{n}.edf".format(p=pacient, n=num_str),
        )
        metadata["channels"] = valid_channels
        clean_dict["metadata"] = metadata
        target = "{path}/chb{p}/chb{p}_{n}.edf".format(
            path=clean_path, p=pacient, n=num_str
        )
        move_channels(clean_dict, channels, target)

def start_process_stage1(params):
    """第一阶段的多进程处理函数"""
    pacient, num, start, end, sum_ind, signals_path, clean_path = params
    
    # 创建输出目录
    if not os.path.exists("{path}/chb{p}".format(p=pacient, path=clean_path)):
        os.makedirs("{path}/chb{p}".format(p=pacient, path=clean_path))
    
    # 读取summary文件
    f = open(
        "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient), "r"
    )

    channels = defaultdict(list)  # 通道和索引的字典
    valid_channels = []  # 有效通道
    to_keep = []  # 要保留的通道索引

    channel_index = 1  # 每个通道的索引
    summary_index = 0  # 选择summary文件中通道参考的索引

    # 处理summary文件
    for line in f:
        line = line.split()
        if len(line) == 0:
            continue

        if line[0] == "Channels" and line[1] == "changed:":
            summary_index += 1

        if (
            line[0] == "Channel"
            and summary_index == sum_ind
            and (line[2] != "-" and line[2] != ".")
        ):  # '-' 表示空通道
            if (
                line[2] in channels.keys()
            ):  # 如果通道重复，在标签后添加'-2'
                name = line[2] + "-2"
            else:
                name = line[2]

            # 添加通道到字典并更新列表
            channels[name].append(str(channel_index))
            channel_index += 1
            valid_channels.append(name)
            to_keep.append(int(line[1][:-1]) - 1)

    # 处理参考文件
    filename = "{path}/chb{p}/chb{p}_{n}.edf".format(
        path=signals_path, p=pacient, n=num
    )
    target = "{path}/chb{p}/chb{p}_{n}.edf".format(path=clean_path, p=pacient, n=num)

    clean_dict = drop_channels(filename, edf_target=target, to_keep=to_keep)

    # 处理元数据：癫痫发作次数和开始/结束时间
    metadata = process_metadata(
        "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient),
        "chb{p}_{n}.edf".format(p=pacient, n=num),
    )

    metadata["channels"] = valid_channels
    clean_dict["metadata"] = metadata

    pickle.dump(clean_dict, open(target + ".pkl", "wb"))

    # 处理其余文件，使其与参考文件具有相同的通道
    process_files(pacient, valid_channels, channels, start, end, signals_path, clean_path)

# ============================
# 第二阶段：数据分段处理
# ============================

def sub_to_segments_stage2(params):
    """第二阶段的多进程处理函数"""
    folder, out_folder, root, channels = params
    print(f"处理 {folder}...")
    
    # 每个记录文件
    for f in tqdm(os.listdir(os.path.join(root, folder))):
        print(f"处理 {folder}/{f}...")
        record = pickle.load(open(os.path.join(root, folder, f), "rb"))
        
        signal = []
        for channel in channels:
            if channel in record:
                signal.append(record[channel])
            else:
                raise ValueError(f"通道 {channel} 在记录 {f} 中未找到")
        signal = np.array(signal)

        if "times" in record["metadata"]:
            seizure_times = record["metadata"]["times"]
        else:
            seizure_times = []

        # 将信号按10秒分段（采样率256Hz）
        SAMPLING_RATE = 256
        for i in range(0, signal.shape[1], SAMPLING_RATE * 10):
            segment = signal[:, i : i + 10 * SAMPLING_RATE]
            if segment.shape[1] == 10 * SAMPLING_RATE:
                # 判断片段是否包含癫痫发作
                label = 0

                segment_end = i + 10 * SAMPLING_RATE
                for seizure_time in seizure_times:
                    if i < seizure_time[1] and segment_end > seizure_time[0]:
                        label = 1
                        break

                # 保存片段
                pickle.dump(
                    {
                        "X": segment,
                        "y": label,
                        "case_id": folder,
                        "recording_id": f.split(".pkl")[0],
                        "window_start_sample": i,
                    },
                    open(
                        os.path.join(out_folder, f"{f.split('.')[0]}-{i}.pkl"),
                        "wb",
                    ),
                )

# ============================
# 主函数
# ============================

def main():
    # 参数设置
    signals_path = r"./data/raw/CHBMIT/chb-mit-scalp-eeg-database-1.0.0"  # 原始数据主目录路径
    clean_path = r"./data/processed/chbmit/process_1"  # 第一阶段输出路径
    out_path = r"./data/processed/chbmit/process_2"  # 第二阶段输出路径
    
    # 创建输出目录
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 第一阶段处理参数
    stage1_parameters = [
        ("01", "01", 2, 46, 0),
        ("02", "01", 2, 35, 0),
        ("03", "01", 2, 38, 0),
        ("05", "01", 2, 39, 0),
        ("06", "01", 2, 24, 0),
        ("07", "01", 2, 19, 0),
        ("08", "02", 3, 29, 0),
        ("10", "01", 2, 89, 0),
        ("11", "01", 2, 99, 0),
        ("14", "01", 2, 42, 0),
        ("20", "01", 2, 68, 0),
        ("21", "01", 2, 33, 0),
        ("22", "01", 2, 77, 0),
        ("23", "06", 7, 20, 0),
        ("24", "01", 3, 21, 0),
        ("04", "07", 1, 43, 1),
        ("09", "02", 1, 19, 1),
        ("15", "02", 1, 63, 1),
        ("16", "01", 2, 19, 0),
        ("18", "02", 1, 36, 1),
        ("19", "02", 1, 30, 1),
    ]

    # 标准通道列表
    channels_list = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
    ]

    print("开始第一阶段处理：原始EDF文件处理...")
    
    # 第一阶段多进程处理
    stage1_params = []
    for params in stage1_parameters:
        stage1_params.append(params + (signals_path, clean_path))
    
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(start_process_stage1, stage1_params)
    
    print("第一阶段处理完成！")
    print("开始第二阶段处理：生成按病例组织的非重叠10秒窗口...")
    
    # 第二阶段参数准备
    folders = os.listdir(clean_path)
    stage2_params = []
    
    for folder in folders:
        # Patient-level folds are applied later from the released manifest.
        out_folder = os.path.join(out_path, "by_subject", folder)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        stage2_params.append((folder, out_folder, clean_path, channels_list))
    
    # 第二阶段多进程处理
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(sub_to_segments_stage2, stage2_params)
    
    print("第二阶段处理完成！")
    print(f"所有处理完成！最终输出保存在: {out_path}")

if __name__ == "__main__":
    main()
