"""
===========================================
PART 2: DISTILBERT MODEL AND TRAINING
===========================================
This module implements DistilBERT model creation and training with:
- Different learning rates for pretrained layers vs classifier head
- LinearScheduleWithWarmup
- Evaluation on test set
- Proper metrics calculation
"""

import os
import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from data_processing.data_loader import MyTextDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils.evaluation import calculate_all_metrics, print_metrics
from tqdm import tqdm
import numpy as np

# 创建distilBERT
def create_distilBERT():
    """
    Create DistilBERT model for sentiment classification.
    DistilBERT: 6-layer Transformer encoder, 12 attention heads, 768 hidden dim, 66M parameters.
    
    Returns:
        DistilBERT model with classification head (768 -> 256 -> 2)
    """
    # 情感分析任务，实际上是做句子二分类任务，用transformers库构建对应的模型
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2  # 二分类：正面/负面
    )
    return model

# 训练distilBERT
# CHANGED: Added proper learning rate scheduling, evaluation, and test set support
def train_distilBERT(train_texts, train_labels, model, test_texts=None, test_labels=None,
                    output_dir="output", batch_size=16, num_epochs=3, max_length=100,
                    pretrained_lr=2e-5, classifier_lr=1e-4, warmup_steps=1000):
    """
    Train DistilBERT model with proper learning rate scheduling and evaluation.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        model: DistilBERT model
        test_texts: Test texts (optional, for evaluation)
        test_labels: Test labels (optional, for evaluation)
        output_dir: Directory to save checkpoints
        batch_size: Batch size (16 or larger, 8 if GPU memory insufficient)
        num_epochs: Number of training epochs
        max_length: Maximum sequence length (100 tokens as per task, or 512 for DistilBERT)
        pretrained_lr: Learning rate for pretrained layers (2e-5)
        classifier_lr: Learning rate for classifier head (1e-4)
        warmup_steps: Number of warmup steps for scheduler (1000)
        
    Returns:
        Trained model
    """
    #  设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 创建checkpoint目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("Tokenizer ready.")
    
    # 准备数据加载器
    train_dataset = MyTextDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Data ready. Training samples: {len(train_texts)}")
    
    # CHANGED: Different learning rates for pretrained layers vs classifier head
    # Separate parameters: pretrained (DistilBERT) vs classifier (new layers)
    pretrained_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name or 'pooler' in name:
            classifier_params.append(param)
        else:
            pretrained_params.append(param)
    
    # CHANGED: AdamW optimizer with different learning rates
    optimizer = AdamW([
        {'params': pretrained_params, 'lr': pretrained_lr},
        {'params': classifier_params, 'lr': classifier_lr}
    ], weight_decay=0.01)
    print(f"Optimizer ready. Pretrained LR: {pretrained_lr}, Classifier LR: {classifier_lr}")
    
    # CHANGED: Linear schedule with warmup
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"Scheduler ready. Warmup steps: {warmup_steps}, Total steps: {total_steps}")
    
    # 训练循环
    model.train()
    print("Training begins.")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
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
            scheduler.step()  # CHANGED: Update learning rate
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # CHANGED: Evaluate on test set if provided
        if test_texts is not None and test_labels is not None:
            print(f"\nEvaluating on test set after epoch {epoch+1}...")
            test_metrics = evaluate_model(model, test_texts, test_labels, tokenizer, 
                                         device, batch_size, max_length)
            print_metrics(test_metrics, f"Epoch {epoch+1} Test Set")

        # 保存训练结果的checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, os.path.join(output_dir, f"distilbert_epoch_{epoch+1}.pt"))
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return model


# ADDED: Evaluation function
def evaluate_model(model, test_texts, test_labels, tokenizer, device, 
                  batch_size=16, max_length=100):
    """
    Evaluate model on test set and return metrics.
    
    Args:
        model: Trained model
        test_texts: Test texts
        test_labels: Test labels
        tokenizer: Tokenizer
        device: Device (CPU/GPU)
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    test_dataset = MyTextDataset(test_texts, test_labels, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions and probabilities
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
            all_labels.extend(labels.cpu().numpy())
    
    model.train()
    
    # Calculate metrics
    metrics = calculate_all_metrics(all_labels, all_predictions, all_probabilities)
    return metrics


# ADDED: Function to load saved model
def load_distilbert_checkpoint(checkpoint_path, device=None):
    """
    Load a saved DistilBERT checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_distilBERT()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model