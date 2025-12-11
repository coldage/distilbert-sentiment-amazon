"""
===========================================
RESULTS VISUALIZATION
===========================================
This module provides visualization tools for model comparison results:
- Generate formatted tables
- Create comparison graphs
- Export to HTML report
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


def load_results(json_path: str) -> List[Dict]:
    """
    Load results from JSON file.
    
    Args:
        json_path: Path to JSON results file
        
    Returns:
        List of result dictionaries
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def create_comparison_table(results: List[Dict], save_path: str = None) -> str:
    """
    Create a formatted comparison table.
    
    Args:
        results: List of result dictionaries
        save_path: Optional path to save table as text file
        
    Returns:
        Formatted table string
    """
    # Header
    table = "\n" + "="*100 + "\n"
    table += "MODEL COMPARISON RESULTS\n"
    table += "="*100 + "\n"
    table += f"{'Model':<20} {'Macro-F1':<12} {'Accuracy':<12} {'ROC-AUC':<12} {'Time (s)':<12} {'Memory (MB)':<15}\n"
    table += "-" * 100 + "\n"
    
    # Data rows
    for result in results:
        table += (f"{result['model_type']:<20} "
                 f"{result['macro_f1']:<12.4f} "
                 f"{result['accuracy']:<12.4f} "
                 f"{result['roc_auc']:<12.4f} "
                 f"{result['training_time_seconds']:<12.2f} "
                 f"{result['memory_usage_mb']:<15.2f}\n")
    
    table += "="*100 + "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table)
        print(f"Table saved to {save_path}")
    
    return table


def create_comparison_graphs(results: List[Dict], output_dir: str = "output"):
    """
    Create comparison graphs for metrics.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save graphs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = [r['model_type'] for r in results]
    
    # Extract metrics
    macro_f1 = [r['macro_f1'] for r in results]
    accuracy = [r['accuracy'] for r in results]
    roc_auc = [r['roc_auc'] for r in results]
    training_time = [r['training_time_seconds'] for r in results]
    memory_usage = [r['memory_usage_mb'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison: Performance, Efficiency, and Resource Usage', fontsize=16, fontweight='bold')
    
    # 1. Macro-F1 Comparison
    axes[0, 0].bar(model_names, macro_f1, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 0].set_title('Macro-F1 Score Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Macro-F1 Score')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(macro_f1):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Accuracy Comparison
    axes[0, 1].bar(model_names, accuracy, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 1].set_title('Accuracy Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(accuracy):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. ROC-AUC Comparison
    axes[0, 2].bar(model_names, roc_auc, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 2].set_title('ROC-AUC Score Comparison', fontweight='bold')
    axes[0, 2].set_ylabel('ROC-AUC Score')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(roc_auc):
        axes[0, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 4. Training Time Comparison
    axes[1, 0].bar(model_names, training_time, color=['#9b59b6', '#f39c12', '#1abc9c'])
    axes[1, 0].set_title('Training Time Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(training_time):
        axes[1, 0].text(i, v + max(training_time)*0.02, f'{v:.2f}s', ha='center', va='bottom')
    
    # 5. Memory Usage Comparison
    if any(m > 0 for m in memory_usage):
        axes[1, 1].bar(model_names, memory_usage, color=['#9b59b6', '#f39c12', '#1abc9c'])
        axes[1, 1].set_title('Memory Usage Comparison', fontweight='bold')
        axes[1, 1].set_ylabel('Memory (MB)')
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(memory_usage):
            if v > 0:
                axes[1, 1].text(i, v + max(memory_usage)*0.02, f'{v:.2f}MB', ha='center', va='bottom')
    else:
        axes[1, 1].text(0.5, 0.5, 'Memory tracking\nnot available\n(CPU mode)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Memory Usage Comparison', fontweight='bold')
    
    # 6. Performance vs Efficiency Trade-off
    axes[1, 2].scatter(training_time, macro_f1, s=200, alpha=0.6, c=['#3498db', '#e74c3c', '#2ecc71'])
    for i, model in enumerate(model_names):
        axes[1, 2].annotate(model, (training_time[i], macro_f1[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    axes[1, 2].set_title('Performance vs Training Time Trade-off', fontweight='bold')
    axes[1, 2].set_xlabel('Training Time (seconds)')
    axes[1, 2].set_ylabel('Macro-F1 Score')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    graph_path = os.path.join(output_dir, "model_comparison_graphs.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"Graphs saved to {graph_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, "model_comparison_graphs.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    
    plt.close()


def create_html_report(results: List[Dict], output_path: str = "output/comparison_report.html"):
    """
    Create an HTML report with results.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save HTML report
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #3498db; color: white; font-weight: bold; }
            tr:hover { background-color: #f5f5f5; }
            .metric { font-weight: bold; color: #2980b9; }
            .best { background-color: #d4edda; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Comparison Report</h1>
            <p>Generated comparison of lightweight pretrained models for sentiment classification.</p>
            
            <h2>Comparison Table</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Macro-F1</th>
                    <th>Accuracy</th>
                    <th>ROC-AUC</th>
                    <th>Training Time (s)</th>
                    <th>Memory Usage (MB)</th>
                </tr>
    """
    
    # Find best values
    best_f1 = max(r['macro_f1'] for r in results)
    best_acc = max(r['accuracy'] for r in results)
    best_auc = max(r['roc_auc'] for r in results)
    best_time = min(r['training_time_seconds'] for r in results)
    
    for result in results:
        f1_class = "best" if result['macro_f1'] == best_f1 else ""
        acc_class = "best" if result['accuracy'] == best_acc else ""
        auc_class = "best" if result['roc_auc'] == best_auc else ""
        time_class = "best" if result['training_time_seconds'] == best_time else ""
        
        html += f"""
                <tr>
                    <td class="metric">{result['model_type']}</td>
                    <td class="{f1_class}">{result['macro_f1']:.4f}</td>
                    <td class="{acc_class}">{result['accuracy']:.4f}</td>
                    <td class="{auc_class}">{result['roc_auc']:.4f}</td>
                    <td class="{time_class}">{result['training_time_seconds']:.2f}</td>
                    <td>{result['memory_usage_mb']:.2f}</td>
                </tr>
        """
    
    html += """
            </table>
            
            <h2>Summary</h2>
            <ul>
    """
    
    # Add summary
    best_f1_model = max(results, key=lambda x: x['macro_f1'])
    best_time_model = min(results, key=lambda x: x['training_time_seconds'])
    
    html += f"""
                <li><strong>Best Macro-F1:</strong> {best_f1_model['model_type']} ({best_f1_model['macro_f1']:.4f})</li>
                <li><strong>Fastest Training:</strong> {best_time_model['model_type']} ({best_time_model['training_time_seconds']:.2f}s)</li>
    """
    
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_path}")


def visualize_results(json_path: str = "output/model_comparison.json", output_dir: str = "output"):
    """
    Main function to create all visualizations from JSON results.
    
    Args:
        json_path: Path to JSON results file
        output_dir: Directory to save outputs
    """
    results = load_results(json_path)
    
    # Create table
    table_path = os.path.join(output_dir, "comparison_table.txt")
    table = create_comparison_table(results, table_path)
    print(table)
    
    # Create graphs
    create_comparison_graphs(results, output_dir)
    
    # Create HTML report
    html_path = os.path.join(output_dir, "comparison_report.html")
    create_html_report(results, html_path)
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    visualize_results()

