"""
评估函数模块
保持你原有的评估函数不变，供其他模块调用
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch


def evaluate(model, dataloader, device):
    """
    评估模型，返回多个指标
    
    注意：这个函数需要传入一个dataloader，它的每个batch应该返回一个元组：
        (input_ids, attention_mask, labels)
    或者你可以根据你的MyTextDataset的格式进行调整。
    """
    model.eval()
    predictions, true_labels = [], []
    all_logits = []  # 存储所有logits，用于计算概率
    total_loss = 0
    
    for batch in dataloader:
        # 假设dataloader返回的是元组 (input_ids, attention_mask, labels)
        # 如果你的dataloader返回的是字典，需要相应调整
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(
                b_input_ids, 
                attention_mask=b_input_mask, 
                labels=b_labels
            )
        
        loss = outputs.loss
        logits = outputs.logits
        total_loss += loss.item()
        
        # 保存logits用于计算概率
        logits_np = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        all_logits.append(logits_np)
        true_labels.append(label_ids)
        
        # 用于计算准确率等指标
        predictions.append(logits_np)
    
    # 合并所有结果
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    
    # 计算预测类别
    pred_classes = np.argmax(flat_predictions, axis=1).flatten()
    
    # 计算预测概率（使用softmax）
    # 对于二分类，我们通常使用正类的概率（类别1）
    probs = torch.nn.functional.softmax(torch.tensor(flat_predictions), dim=-1)
    probs_np = probs.numpy()
    
    # 计算准确率
    accuracy = accuracy_score(flat_true_labels, pred_classes)
    
    # 计算宏观F1-score、精确率、召回率
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        flat_true_labels, pred_classes, average='macro'
    )
    
    # 计算加权F1-score（可选）
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        flat_true_labels, pred_classes, average='weighted'
    )
    
    # 计算每个类别的F1-score
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        flat_true_labels, pred_classes, average=None
    )
    
    # 计算ROC-AUC
    # 对于二分类任务，使用正类的概率
    if len(np.unique(flat_true_labels)) == 2:  # 确保是二分类
        # 使用正类（类别1）的概率
        pos_probs = probs_np[:, 1]
        try:
            roc_auc = roc_auc_score(flat_true_labels, pos_probs)
        except:
            # 如果ROC-AUC计算失败（例如所有样本都是同一类别）
            roc_auc = 0.5  # 随机猜测的水平
    else:
        # 如果不是二分类，则计算多分类的ROC-AUC（需要one-hot编码）
        # 这里假设是多分类
        n_classes = len(np.unique(flat_true_labels))
        if n_classes > 2:
            # 将标签转换为one-hot编码
            true_labels_one_hot = label_binarize(flat_true_labels, classes=range(n_classes))
            roc_auc = roc_auc_score(true_labels_one_hot, probs_np, average='macro', multi_class='ovo')
        else:
            roc_auc = 0.5
    
    # 平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 创建返回结果字典
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'class_metrics': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist(),
            'support': support_per_class.tolist()
        },
        'predictions': pred_classes.tolist(),
        'true_labels': flat_true_labels.tolist(),
        'probabilities': probs_np.tolist()
    }
    
    return results


def print_evaluation_results(results):
    """打印评估结果"""
    print("=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"平均损失 (Loss): {results['loss']:.4f}")
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"宏观精确率 (Macro Precision): {results['precision_macro']:.4f}")
    print(f"宏观召回率 (Macro Recall): {results['recall_macro']:.4f}")
    print(f"宏观F1-score (Macro F1): {results['f1_macro']:.4f}")
    print(f"加权F1-score (Weighted F1): {results['f1_weighted']:.4f}")
    
    # 打印每个类别的指标
    print("\n每个类别的详细指标:")
    for i, (precision, recall, f1, support) in enumerate(zip(
        results['class_metrics']['precision'],
        results['class_metrics']['recall'],
        results['class_metrics']['f1'],
        results['class_metrics']['support']
    )):
        print(f"  类别 {i}:")
        print(f"    精确率: {precision:.4f}")
        print(f"    召回率: {recall:.4f}")
        print(f"    F1-score: {f1:.4f}")
        print(f"    样本数: {support}")
    print("=" * 60)


# 可选：添加一个简化版的评估函数，用于快速评估
def evaluate_simple(model, dataloader, device):
    """
    简化的评估函数，只返回准确率和损失
    适用于训练过程中的快速评估
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(
                b_input_ids, 
                attention_mask=b_input_mask, 
                labels=b_labels
            )
        
        loss = outputs.loss
        logits = outputs.logits
        total_loss += loss.item()
        
        # 计算准确率
        _, predicted = torch.max(logits, 1)
        total += b_labels.size(0)
        correct += (predicted == b_labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


if __name__ == "__main__":
    # 示例：测试评估函数
    print("✅ evaluate.py 模块加载成功")
    print("可用函数:")
    print("  1. evaluate(model, dataloader, device) - 完整评估")
    print("  2. evaluate_simple(model, dataloader, device) - 快速评估")
    print("  3. print_evaluation_results(results) - 打印评估结果")