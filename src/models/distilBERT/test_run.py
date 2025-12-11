# test_run.py
"""
测试脚本 - 快速验证所有模块
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_modules():
    print("🧪 测试模块导入...")
    
    modules = [
        ("data_manager", "DataManager"),
        ("experiment_config", "ExperimentSuite"),
        ("model_factory", "ModelFactory"),
        ("experiment_runner", "ExperimentRunner"),
        ("experiment_manager", "ExperimentManager"),
        ("evaluate", "evaluate")
    ]
    
    for module_name, class_name in modules:
        try:
            if class_name == "evaluate":
                # evaluate是函数，不是类
                __import__(module_name)
                print(f"✅ {module_name}.py")
            else:
                module = __import__(module_name)
                if hasattr(module, class_name):
                    print(f"✅ {module_name}.{class_name}")
                else:
                    print(f"❌ {module_name}.{class_name} - 未找到")
        except ImportError as e:
            print(f"❌ {module_name} - 导入失败: {e}")
    
    print("\n📋 下一步:")
    print("1. 确保数据文件在 ./data/ 目录下")
    print("2. 安装依赖: pip install -r requirements.txt")
    print("3. 运行测试: python main.py")

if __name__ == "__main__":
    test_modules()