import bz2
import math
import os
import rootutils
import random
from tqdm import tqdm

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
            label = int(parts[0].removeprefix('__label__')) # 标签1表示1-2星的负面评价，标签2表示4-5星的正面评价
            label = int(label)-1 # 处理为0为负面，1为正面
            text = parts[1].strip()
            labels.append(label)
            texts.append(text)
    
    size = len(labels)
    return texts, labels, size


# 3. 输出数据标签分布情况
def labels_analysis(labels):
    cnt = [0, 0] # 记录正负标签评论的数目
    for label in labels:
        cnt[label] += 1 
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

# 6. 采样数据集，保持正负样本比例一致
def sample_dataset(texts, labels, sample_ratio=0.1):
    """
    从数据集中采样，保持正负样本比例一致
    :param texts: 原始文本列表
    :param labels: 原始标签列表
    :param sample_ratio: 采样比例，默认0.1
    :return: 采样后的文本列表和标签列表
    """
    # 统计正负样本数量
    cnt = labels_analysis(labels)
    # 计算采样数量
    neg_sample_num = int(cnt[0] * sample_ratio)
    pos_sample_num = int(cnt[1] * sample_ratio)
    # 分开正负样本
    neg_samples = [texts[i] for i in range(len(labels)) if labels[i] == 0]
    pos_samples = [texts[i] for i in range(len(labels)) if labels[i] == 1]
    # 打乱正负样本
    random.shuffle(neg_samples)
    random.shuffle(pos_samples)
    # 采样正负样本
    neg_samples = neg_samples[:neg_sample_num]
    pos_samples = pos_samples[:pos_sample_num]
    # 合并采样结果
    sampled_texts = neg_samples + pos_samples
    # 合并采样标签
    sampled_labels = [0] * neg_sample_num + [1] * pos_sample_num
    # 打乱采样结果
    temp = list(zip(sampled_texts, sampled_labels))
    random.shuffle(temp)
    # 解压缩打乱后的结果
    sampled_texts, sampled_labels = zip(*temp)
    return list(sampled_texts), list(sampled_labels)

# 7.存储处理后的数据集, 格式为ft.bz2
def store_processed_dataset(texts, labels, path):
    """
    存储处理后的数据集到指定路径
    :param texts: 处理后的文本列表
    :param labels: 处理后的标签列表
    :param path: 存储路径
    """
    with bz2.open(path, 'wt', encoding='utf-8') as f:
        for text, label in zip(texts, labels):
            f.write(f"{label}\t{text}\n")
    print(f"数据集已存储到 {path}")

# 8.读取处理后的数据集, 格式为ft.bz2
def read_sampled_dataset(path, encoding='utf-8', total_lines=None):
    """
    从指定路径读取处理后的数据集
    :param path: 数据集路径
    :param encoding: 编码格式，默认utf-8
    :return: 文本列表和标签列表
    """
    texts = []
    labels = []
    size = 0
    with bz2.open(path, 'rt', encoding=encoding) as f:
        for line in tqdm(f, total=total_lines, desc="加载数据"):
            parts = line.split('\t')
            label = int(parts[0])
            text = parts[1].strip()
            labels.append(label)
            texts.append(text)
            size += 1
    return texts, labels, size


###########
## Tests ##
###########
def main():
    # 设置根目录位置(通过自动递归往外查找名字为".project-root"的空文件实现)
    root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    # 设置工作目录到根目录
    os.chdir(root_path)

    # 数据集位置
    test_data_path = "data/test.ft.txt.bz2"
    train_data_path = "data/train.ft.txt.bz2"

    print_firstN(test_data_path, 1)

    # texts, labels, size = get_texts_and_labels(train_data_path, total_lines=3600000)
    # sampled_texts, sampled_labels = sample_dataset(texts, labels, sample_ratio=0.1)
    # store_processed_dataset(sampled_texts, sampled_labels, "data/sampled_train.ft.txt.bz2")

    sampled_texts, sampled_labels, size = read_sampled_dataset("data/sampled_train.ft.txt.bz2", total_lines=360000)
    cnt = labels_analysis(sampled_labels)
    max_len, min_len, avg_len = texts_analysis(sampled_texts)
    print(size)
    print(cnt)
    print(max_len, min_len, avg_len)

if __name__ == "__main__":
    main()
