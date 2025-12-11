import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
import os

from data_processing.data_checker import get_texts_and_labels, read_sampled_dataset
from data_processing.data_loader import SentimentDataset

# 设置随机种子保证可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# 1. 加载数据集（使用IMDB情感分析数据集为例）
print("加载数据集...")
train_texts, train_labels, train_size = read_sampled_dataset("data/sampled_train.ft.txt.bz2", total_lines=360000)
test_texts, test_labels, test_size = get_texts_and_labels("data/test.ft.txt.bz2", total_lines=400000)

# 查看数据集结构
print(f"数据集样例: {train_labels[0]}, {train_texts[0]}")
print(f"训练集大小: {train_size}, 测试集大小: {test_size}")


# 2. 加载模型和分词器
model_name = "distilbert-base-uncased"
print(f"\n加载模型: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # 情感分类：正面/负面
    ignore_mismatched_sizes=True
)


# 3. 数据预处理
print("\n预处理数据...")
max_length = 100

tokenized_train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=max_length)
tokenized_test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=max_length)
tokenized_datasets = {
    "train": tokenized_train_dataset,
    "test": tokenized_test_dataset
}


# 4. 创建数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. 定义评估指标
def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    
    return {
        "accuracy": accuracy,
        "f1": f1
    }

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir="./distilbert-sentiment",  # 输出目录
    overwrite_output_dir=True,  # 覆盖已有输出
    num_train_epochs=3,  # 训练轮数
    per_device_train_batch_size=16,  # 训练批次大小
    per_device_eval_batch_size=16,  # 评估批次大小
    warmup_steps=500,  # 预热步数
    weight_decay=0.01,  # 权重衰减
    logging_dir="./logs",  # 日志目录
    logging_steps=100,  # 每100步记录一次日志
    eval_strategy="epoch",  # 每个epoch评估一次
    save_strategy="epoch",  # 每个epoch保存一次
    learning_rate=2e-5,  # 学习率
    load_best_model_at_end=True,  # 训练结束时加载最佳模型
    metric_for_best_model="accuracy",  # 选择最佳模型的指标
    greater_is_better=True,  # 指标越大越好
    save_total_limit=2,  # 只保存2个最佳模型
    report_to="none",  # 禁用wandb等报告工具
    push_to_hub=False,  # 不推送到Hugging Face Hub
    fp16=torch.cuda.is_available(),  # 如果可用则使用混合精度训练
)

# 7. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. 训练模型
print("\n开始训练...")
train_result = trainer.train()

# 9. 评估模型
print("\n评估模型...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 10. 保存模型
print("\n保存模型...")
trainer.save_model("./distilbert-sentiment-final")
tokenizer.save_pretrained("./distilbert-sentiment-final")

# 11. 测试单个样本
def predict_sentiment(text, model_path="./distilbert-sentiment-final"):
    """预测单个文本的情感"""
    # 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    # 预处理文本
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 获取结果
    sentiment = "正面" if torch.argmax(predictions) == 1 else "负面"
    confidence = torch.max(predictions).item()
    
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "scores": {
            "负面": predictions[0][0].item(),
            "正面": predictions[0][1].item()
        }
    }

# 测试样例
test_texts = [
    "This movie is absolutely fantastic! I loved every minute of it.",
    "The plot was terrible and the acting was even worse.",
    "It was okay, not great but not bad either."
]

print("\n测试预测:")
for text in test_texts:
    result = predict_sentiment(text)
    print(f"文本: {result['text']}")
    print(f"情感: {result['sentiment']} (置信度: {result['confidence']:.2%})")
    print("-" * 50)