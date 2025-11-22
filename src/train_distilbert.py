import os
import rootutils
# 设置根目录位置(通过自动递归往外查找名字为".project-root"的空文件实现)
root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# 设置工作目录到根目录
os.chdir(root_path)
# 使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from data_processing.data_checker import create_tiny_debug_dataset, get_texts_and_labels
from models.distilBERT import create_distilBERT, train_distilBERT


# 数据加载
test_data_path = "data/test.ft.txt.bz2"
train_data_path = "data/train.ft.txt.bz2"
# train_texts, train_labels, _ = get_texts_and_labels(train_data_path, total_lines=3600000)
# test_texts, test_labels, _ = get_texts_and_labels(test_data_path, total_lines=400000)
debug_texts, debug_labels = create_tiny_debug_dataset()


# 模型创建
model = create_distilBERT()

# 模型训练
train_distilBERT(debug_texts, debug_labels, model)



