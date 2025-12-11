"""
===========================================
PART 2: EVALUATION METRICS
===========================================
This module implements evaluation metrics as specified in the task:
- Accuracy
- Macro-F1 Score
- ROC-AUC
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from typing import Tuple, List


def calculate_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate Accuracy: (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true: True labels (0: negative, 1: positive)
        y_pred: Predicted labels (0: negative, 1: positive)
        
    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def calculate_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate Macro-F1 Score:
    - Compute Precision and Recall for each class
    - Compute F1 for each class: F1 = 2 * (P * R) / (P + R)
    - Macro-F1 = (F1_negative + F1_positive) / 2
    
    Args:
        y_true: True labels (0: negative, 1: positive)
        y_pred: Predicted labels (0: negative, 1: positive)
        
    Returns:
        Macro-F1 score
    """
    return f1_score(y_true, y_pred, average='macro')


def calculate_roc_auc(y_true: List[int], y_pred_proba: List[float]) -> float:
    """
    Calculate ROC-AUC Score:
    - Plot TPR vs FPR curve
    - Calculate area under the curve using trapezoidal integration
    
    Args:
        y_true: True labels (0: negative, 1: positive)
        y_pred_proba: Predicted probabilities for positive class (class 1)
        
    Returns:
        ROC-AUC score
    """
    # ROC-AUC requires probabilities for the positive class
    return roc_auc_score(y_true, y_pred_proba)


def calculate_all_metrics(y_true: List[int], y_pred: List[int], 
                         y_pred_proba: List[float] = None) -> dict:
    """
    Calculate all evaluation metrics at once.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class (optional)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'macro_f1': calculate_macro_f1(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = calculate_roc_auc(y_true, y_pred_proba)
    
    return metrics


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix to add before printing
    """
    print(f"\n{prefix}Evaluation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

