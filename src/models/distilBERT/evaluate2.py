def evaluate(model, dataloader, device):
    """评估模型，返回多个指标"""
    model.eval()
    predictions, true_labels = [], []
    all_logits = []
    total_loss = 0
    
    for batch in dataloader:
        # 假设dataloader返回的是元组 (input_ids, attention_mask, labels)
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
        
        logits_np = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        all_logits.append(logits_np)
        true_labels.append(label_ids)
        predictions.append(logits_np)
    
    # 合并所有结果
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    
    # 计算预测类别
    pred_classes = np.argmax(flat_predictions, axis=1).flatten()
    
    # 计算预测概率
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
    
    # ========== 修改这里：计算ROC曲线数据 ==========
    roc_auc = 0.5
    fpr, tpr, thresholds = None, None, None
    
    # 对于二分类任务，使用正类的概率（类别1）
    if len(np.unique(flat_true_labels)) == 2:
        # 使用正类（类别1）的概率
        pos_probs = probs_np[:, 1]
        try:
            roc_auc = roc_auc_score(flat_true_labels, pos_probs)
            # 计算ROC曲线数据
            fpr, tpr, thresholds = roc_curve(flat_true_labels, pos_probs)
        except:
            # 如果ROC-AUC计算失败（例如所有样本都是同一类别）
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
        'probabilities': probs_np.tolist(),
        # ========== 新增：ROC曲线数据 ==========
        'roc_curve': {
            'fpr': fpr.tolist() if fpr is not None else [],
            'tpr': tpr.tolist() if tpr is not None else [],
            'thresholds': thresholds.tolist() if thresholds is not None else []
        }
    }
    
    return resultsd