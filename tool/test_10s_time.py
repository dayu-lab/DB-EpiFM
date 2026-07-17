import torch
import time
import numpy as np
import pickle
import os
from models.model_for_tuab import Model

class Param:
    def __init__(self):
        self.use_pretrained_weights = False
        self.cuda = 0
        self.foundation_dir = ''
        # 使用与权重文件匹配的分类器类型
        self.classifier = 'all_patch_reps'
        self.dropout = 0.5

def load_pkl_data(pkl_path):
    """加载pkl格式的脑电数据，处理字典格式"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 检查数据类型
    print(f"  加载的数据类型: {type(data)}")
    
    # 如果是字典，尝试找到脑电数据
    if isinstance(data, dict):
        print(f"  字典键: {list(data.keys())}")
        
        # 常见的脑电数据键名
        possible_keys = ['eeg', 'data', 'x', 'signal', 'X', 'eeg_data']
        for key in possible_keys:
            if key in data:
                eeg_data = data[key]
                print(f"  找到脑电数据键: '{key}', 形状: {eeg_data.shape}")
                return eeg_data, data
        
        # 如果没有找到标准键，尝试第一个numpy数组
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"  使用键 '{key}' 的数据，形状: {value.shape}")
                return value, data
        
        # 如果都没有，返回整个字典（可能需要进一步处理）
        print("⚠️  未找到明确的脑电数据，返回整个字典")
        return data, data
    else:
        # 如果不是字典，直接返回
        return data, None

def interpret_prediction(prediction, threshold=0.5):
    """
    解释模型预测结果
    假设是二分类问题：正常脑电 vs 异常脑电
    """
    # 将模型输出转换为概率（如果使用sigmoid激活）
    probability = torch.sigmoid(prediction).item()
    
    # 根据阈值进行分类
    if probability >= threshold:
        label = "异常脑电"
        confidence = probability
    else:
        label = "正常脑电"
        confidence = 1 - probability
    
    return {
        'raw_output': prediction.item(),
        'probability': probability,
        'label': label,
        'confidence': confidence
    }

def test_single_inference(model_path, pkl_path, num_runs=50):
    """
    测试单个10秒片段的推理时间
    """
    # 1. 加载模型
    param = Param()
    model = Model(param)
    
    # 2. 直接加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 尝试加载权重
    try:
        model.load_state_dict(checkpoint)
        print(f"✅ 权重加载成功!")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        # 如果失败，使用非严格模式
        model.load_state_dict(checkpoint, strict=False)
        print("⚠️  使用非严格模式加载权重")
    
    model.eval()
    print(f"✅ 模型加载完成: {model_path}")
    
    # 3. 加载数据
    raw_data, full_data = load_pkl_data(pkl_path)
    print(f"✅ 数据加载完成: {pkl_path}")
    
    # 4. 根据数据类型处理数据
    if hasattr(raw_data, 'shape'):
        print(f"   原始数据形状: {raw_data.shape}")
        
        # 预处理数据 (16, 2000) -> (1, 16, 10, 200)
        if raw_data.shape == (16, 2000):
            processed_data = raw_data.reshape(16, 10, 200)
            processed_data = np.expand_dims(processed_data, axis=0)
        else:
            print(f"⚠️  非常规数据形状，尝试直接使用")
            processed_data = np.expand_dims(raw_data, axis=0)
            
    elif isinstance(raw_data, dict):
        print("❌ 数据为字典格式，需要手动提取脑电数据")
        print(f"   字典键: {list(raw_data.keys())}")
        return
    
    print(f"   模型输入形状: {processed_data.shape}")
    
    # 5. 准备推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.from_numpy(processed_data).float().to(device)
    model = model.to(device)
    print(f"✅ 使用设备: {device}")
    
    # 6. 预热
    print("进行预热运行...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
    
    # 7. 测量推理时间
    print(f"🚀 开始进行 {num_runs} 次推理时间测试...")
    
    timings = []
    final_output = None
    
    with torch.no_grad():
        for i in range(num_runs):
            if device.type == 'cuda':
                # 使用CUDA事件进行更精确的时间测量
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = model(input_tensor)
                end.record()
                torch.cuda.synchronize()
                inference_time = start.elapsed_time(end)
            else:
                start_time = time.perf_counter()
                output = model(input_tensor)
                end_time = time.perf_counter()
                inference_time = (end_time - start_time) * 1000
            
            timings.append(inference_time)
            
            # 保存最后一次的输出用于结果分析
            if i == num_runs - 1:
                final_output = output
            
            if (i + 1) % 10 == 0:
                print(f"   已完成 {i + 1}/{num_runs} 次测试")
    
    # 8. 解释预测结果
    if final_output is not None:
        prediction_result = interpret_prediction(final_output)
        
        print("\n" + "="*60)
        print("🧠 模型预测结果分析")
        print("="*60)
        print(f"原始输出值: {prediction_result['raw_output']:.4f}")
        print(f"预测概率: {prediction_result['probability']:.4f}")
        print(f"分类结果: {prediction_result['label']}")
        print(f"置信度: {prediction_result['confidence']:.4f}")
        
        # 输出预测的置信度级别
        confidence_level = "高" if prediction_result['confidence'] > 0.8 else "中" if prediction_result['confidence'] > 0.6 else "低"
        print(f"置信度级别: {confidence_level}")
        
        # 提供解释性建议
        if prediction_result['label'] == "异常脑电":
            print("💡 建议: 此脑电片段可能显示异常模式，建议进一步检查")
        else:
            print("💡 建议: 此脑电片段显示正常模式")
    
    # 9. 输出时间统计结果
    timings = np.array(timings)
    print("\n" + "="*60)
    print("⏱️  TUAB脑电模型推理时间测试结果")
    print("="*60)
    print(f"数据格式: 16通道 × 2000采样点 (10秒脑电)")
    print(f"测试次数: {num_runs}")
    print(f"平均推理时间: {np.mean(timings):.2f} ± {np.std(timings):.2f} ms")
    print(f"最快时间: {np.min(timings):.2f} ms")
    print(f"最慢时间: {np.max(timings):.2f} ms")
    print(f"推理速度: {1000/np.mean(timings):.1f} 片段/秒")
    print("="*60)
    
    return {
        'mean_time': np.mean(timings),
        'std_time': np.std(timings),
        'min_time': np.min(timings),
        'max_time': np.max(timings),
        'fps': 1000/np.mean(timings),
        'prediction': prediction_result if final_output is not None else None
    }

if __name__ == "__main__":
    # 直接在这里设置你的文件路径
    MODEL_PATH = "./checkpoints/tuab_finetuned.pth"
    PKL_PATH = "./data/processed/tuab/val/example.pkl"
    
    results = test_single_inference(MODEL_PATH, PKL_PATH, num_runs=50)