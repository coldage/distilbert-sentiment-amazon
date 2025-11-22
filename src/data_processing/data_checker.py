import bz2
import math
import os
import rootutils
from tqdm import tqdm

############
## Configs #
############

# # 设置根目录位置(通过自动递归往外查找名字为".project-root"的空文件实现)
# root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# # 设置工作目录到根目录
# os.chdir(root_path)

# # 数据集位置
# test_data_path = "data/test.ft.txt.bz2"
# train_data_path = "data/train.ft.txt.bz2"


###########
## Funcs ##
###########

# 1. 打印path的数据集的前N个条目，默认打印前5个
def print_firstN(path, N=5, encoding='utf-8'):
    with bz2.open(path, 'rt', encoding=encoding) as f:
        for line in f:
            print(line.strip())  # 处理每一行数据s
            N -= 1
            if N == 0: 
                break


# 2. 读取数据，得到标签和文本
# 进度条需要预设total_lines，train是3600000，test是400000
def get_texts_and_labels(path, encoding='utf-8', total_lines=None):
    labels = []
    texts = []
    
    with bz2.open(path, 'rt', encoding=encoding) as f:
        for line in tqdm(f, total=total_lines, desc="加载数据"):
            parts = line.split(' ', 1)
            label = int(parts[0].removeprefix('__label__'))
            text = parts[1].strip()
            labels.append(label)
            texts.append(text)
    
    size = len(labels)
    return texts, labels, size


# 3. 输出数据标签分布情况
def labels_analysis(labels):
    cnt = [0, 0] # 记录正负标签评论的数目
    for label in labels:
        cnt[int(label)-1] += 1 # 标签1表示1-2星的负面评价，标签2表示4-5星的正面评价
    return cnt


# 4. 输出文本长度分布情况，可以绘制分布图像，默认关闭
def texts_analysis(texts, draw_pic=False):
    max_len = 0
    min_len = math.inf
    avg_len = 0
    cnt = 0
    for text in texts:
        length = len(text)
        avg_len += length
        cnt += 1
        if length > max_len: max_len = length
        if length < min_len: min_len = length
    return max_len, min_len, avg_len/cnt


# 5. 手动创建的小数据集，用于调试
def create_tiny_debug_dataset():
    """创建极小的调试数据集"""
    # 简单的文本和标签
    debug_texts = [
        "This movie is great and amazing!",
        "I hate this product, it's terrible.",
        "The book was okay, not too bad.",
        "Excellent service, highly recommended!",
        "Poor quality, very disappointed.",
        "It's a good product for the price.",
        "Worst experience ever, avoid this.",
        "Fantastic! I love it so much.",
        "Mediocre at best, could be better.",
        "Outstanding performance and quality."
    ]
    debug_labels = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # 1: positive, 0: negative
    # print(f"创建微型调试数据集: {len(debug_texts)} 个样本")
    return debug_texts, debug_labels



###########
## Tests ##
###########

# print_firstN(test_data_path, 1)
# texts, labels, size = get_texts_and_labels(test_data_path, total_lines=400000)
# texts, labels, size = get_texts_and_labels(train_data_path, total_lines=3600000)
# cnt = labels_analysis(labels)
# max_len, min_len, avg_len = texts_analysis(texts)
# print(size)
# print(cnt)
# print(max_len, min_len, avg_len)

