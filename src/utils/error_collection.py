"""
===========================================
PART 5 PREP: ERROR COLLECTION
===========================================
This module provides functions to collect prediction errors for Part 5 (Error Analysis).
This is prep work - the next person can use these functions to analyze errors.

Usage:
    errors = collect_prediction_errors(model, test_texts, test_labels, tokenizer, device)
    save_errors_to_file(errors, "output/errors.json")
"""

import json
import os
import torch
from typing import List, Dict
from data_processing.data_loader import MyTextDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def collect_prediction_errors(model, test_texts: List[str], test_labels: List[int], 
                              tokenizer, device, batch_size=16, max_length=100) -> List[Dict]:
    """
    Collect all prediction errors from the model.
    
    Args:
        model: Trained model
        test_texts: Test texts
        test_labels: True labels
        tokenizer: Tokenizer
        device: Device (CPU/GPU)
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        List of error dictionaries with:
        - text: Original text
        - true_label: True label (0=negative, 1=positive)
        - predicted_label: Predicted label
        - predicted_probability: Probability of predicted class
        - error_type: 'FP' (false positive) or 'FN' (false negative)
    """
    model.eval()
    test_dataset = MyTextDataset(test_texts, test_labels, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    errors = []
    
    with torch.no_grad():
        text_idx = 0
        for batch in tqdm(test_loader, desc="Collecting errors"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Check each prediction in the batch
            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predictions[i].item()
                pred_prob = probabilities[i][predicted_label].item()
                
                # If prediction is wrong, record it
                if true_label != predicted_label:
                    error_type = 'FP' if (true_label == 0 and predicted_label == 1) else 'FN'
                    
                    errors.append({
                        'text': test_texts[text_idx],
                        'true_label': int(true_label),
                        'predicted_label': int(predicted_label),
                        'predicted_probability': float(pred_prob),
                        'error_type': error_type
                    })
                
                text_idx += 1
    
    model.train()
    return errors


def save_errors_to_file(errors: List[Dict], output_path: str):
    """
    Save errors to JSON file.
    
    Args:
        errors: List of error dictionaries
        output_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(errors)} errors to {output_path}")


def categorize_errors(errors: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Basic error categorization (can be extended in Part 5).
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        Dictionary with error categories
    """
    categorized = {
        'false_positives': [e for e in errors if e['error_type'] == 'FP'],
        'false_negatives': [e for e in errors if e['error_type'] == 'FN'],
        'high_confidence_errors': [e for e in errors if e['predicted_probability'] > 0.8],
        'low_confidence_errors': [e for e in errors if e['predicted_probability'] < 0.6]
    }
    
    return categorized


def print_error_summary(errors: List[Dict]):
    """
    Print a summary of collected errors.
    
    Args:
        errors: List of error dictionaries
    """
    total_errors = len(errors)
    fp_count = sum(1 for e in errors if e['error_type'] == 'FP')
    fn_count = sum(1 for e in errors if e['error_type'] == 'FN')
    
    print(f"\n{'='*60}")
    print("ERROR COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total errors: {total_errors}")
    print(f"False Positives (FP): {fp_count}")
    print(f"False Negatives (FN): {fn_count}")
    print(f"{'='*60}\n")

