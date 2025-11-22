import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data_processing.data_loader import MyTextDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 创建distilBERT
def create_distilBERT():
    # 情感分析任务，实际上是做句子二分类任务，用transformers库构建对应的模型
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2  # 二分类：正面/负面
    )
    return model

# 训练distilBERT
def train_distilBERT(train_texts, train_labels, model, output_dir="output"):
    #  设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 创建checkpoint目录
    # os.makedirs(output_dir, exist_ok=True)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("Tokenizer ready.")
    # 准备数据加载器
    train_dataset = MyTextDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("Data ready.")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    print("Optimizer ready.")
    
    # 训练循环
    model.train()
    print("Training begins.")
    for epoch in range(3):  # 3个epoch
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移到设备（GPU）
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 每50个批次打印一次进度
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # 保存训练结果的checkpoint
        # 需要记录什么指标在这里添加
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(output_dir, f"epoch_{epoch+1}.pt"))
    
    return model