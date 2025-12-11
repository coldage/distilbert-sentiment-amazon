# main.py
"""
主运行脚本
"""

import os
import sys
import torch
from datetime import datetime

# 确保能导入当前目录的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🧪 DistilBERT 超参数敏感性实验与消融实验")
    print("=" * 60)
    
    # 1. 检查环境
    print("\n1️⃣ 检查环境...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. 设置输出路径
    print("\n2️⃣ 设置输出路径...")
    OUTPUT_BASE = "../../../output/distilBERT/experiments"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_DIR = os.path.join(OUTPUT_BASE, f"experiment_suite_{timestamp}")
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    print(f"✅ 输出目录: {EXPERIMENT_DIR}")
    
    # 3. 导入模块
    print("\n3️⃣ 导入模块...")
    try:
        from experiment_config import ExperimentSuite
        from experiment_manager import ExperimentManager
        from data_manager import DataManager
        
        print("✅ 模块导入成功")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return
    
    # 4. 创建数据管理器
    print("\n4️⃣ 创建数据管理器...")
    data_config = {
        'train_file': './data/train.ft.txt',
        'test_file': './data/test.ft.txt',
        'train_sample_ratio': 0.1,  # 10%数据
        'val_ratio': 0.1,           # 10%验证集
        'batch_size': 16            # 默认批次大小
    }
    
    dm = DataManager(data_config)
    
    # 5. 创建实验配置
    print("\n5️⃣ 创建实验配置...")
    experiments = ExperimentSuite.create_all_experiments()
    
    # 为了测试，只运行前2个实验
    test_experiments = experiments
    print(f"共创建 {len(experiments)} 个实验，测试运行前 {len(test_experiments)} 个")
    
    # 6. 创建实验管理器
    print("\n6️⃣ 创建实验管理器...")
    manager = ExperimentManager(output_dir=EXPERIMENT_DIR)
    manager.add_experiments(test_experiments)
    
    # 7. 运行实验
    print("\n7️⃣ 开始运行实验...")
    print("警告: 这可能需要一些时间")
    
    # 根据GPU内存设置并行数
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory >= 24:
            max_workers = 2
        else:
            max_workers = 1
    else:
        max_workers = 1
    
    print(f"并行运行数: {max_workers}")
    
    try:
        results = manager.run_all_experiments(dm, max_workers=max_workers)
    except Exception as e:
        print(f"❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 8. 导入可视化器
    print("\n8️⃣ 生成可视化图表...")
    try:
        from experiment_visualizer import ExperimentVisualizer
        
        # 加载结果汇总
        summary_path = os.path.join(EXPERIMENT_DIR, "experiment_summary.csv")
        if os.path.exists(summary_path):
            import pandas as pd
            results_df = pd.read_csv(summary_path)
            
            # 创建可视化器
            visualizer = ExperimentVisualizer(results_df, output_dir=EXPERIMENT_DIR)
            
            # 绘制图表
            visualizer.plot_ablation_study(save=True)
            visualizer.plot_hyperparameter_sensitivity(save=True)
            
            print("✅ 图表生成完成")
        else:
            print("⚠️ 未找到结果汇总文件")
    except ImportError:
        print("⚠️ 可视化器未找到，跳过图表生成")
    
    print("\n" + "=" * 60)
    print("🎉 实验运行完成!")
    print("=" * 60)
    print(f"📁 所有结果保存在: {EXPERIMENT_DIR}")

if __name__ == "__main__":
    main()