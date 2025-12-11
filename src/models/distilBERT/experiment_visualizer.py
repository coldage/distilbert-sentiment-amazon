"""
Visualizer
Plot ablation experiments and hyperparameter sensitivity analysis charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ExperimentVisualizer:
    """Experiment Visualizer"""
    
    def __init__(self, results_df, output_dir="./visualizations"):
        """
        Args:
            results_df: Experiment results DataFrame
            output_dir: Chart output directory
        """
        self.results_df = results_df
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_ablation_study(self, save=True):
        """Plot ablation experiment comparison chart"""
        # Filter ablation experiments
        ablation_df = self.results_df[self.results_df['experiment_name'].str.startswith('abl_')]
        
        if ablation_df.empty:
            print("No ablation experiment data found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Accuracy comparison bar chart
        strategies = ablation_df['experiment_type'].tolist()
        accuracies = ablation_df['test_accuracy'].tolist()
        f1_scores = ablation_df['test_f1_macro'].tolist()
        
        x = np.arange(len(strategies))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Fine-tuning Strategy')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Comparison of Different Fine-tuning Strategies')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(strategies, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
            axes[0, 0].text(i - width/2, acc + 0.01, f'{acc:.3f}', 
                          ha='center', va='bottom', fontsize=9)
            axes[0, 0].text(i + width/2, f1 + 0.01, f'{f1:.3f}', 
                          ha='center', va='bottom', fontsize=9)
        
        # 2. ROC-AUC comparison
        roc_aucs = ablation_df['test_roc_auc'].tolist()
        bars = axes[0, 1].bar(strategies, roc_aucs, alpha=0.8, 
                              color=plt.cm.viridis(np.linspace(0, 1, len(strategies))))
        axes[0, 1].set_xlabel('Fine-tuning Strategy')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].set_title('ROC-AUC Comparison of Different Fine-tuning Strategies')
        axes[0, 1].set_xticklabels(strategies, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, roc_aucs):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Validation loss comparison
        val_losses = ablation_df['best_val_loss'].tolist()
        axes[1, 0].bar(strategies, val_losses, alpha=0.8, color='orange')
        axes[1, 0].set_xlabel('Fine-tuning Strategy')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Best Validation Loss of Different Fine-tuning Strategies')
        axes[1, 0].set_xticklabels(strategies, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Best epoch comparison
        best_epochs = ablation_df['best_epoch'].tolist()
        axes[1, 1].bar(strategies, best_epochs, alpha=0.8, color='green')
        axes[1, 1].set_xlabel('Fine-tuning Strategy')
        axes[1, 1].set_ylabel('Best Epoch')
        axes[1, 1].set_title('Best Training Epochs for Different Fine-tuning Strategies')
        axes[1, 1].set_xticklabels(strategies, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig_path = os.path.join(self.output_dir, "ablation_study_comparison.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"📈 Ablation study chart saved: {fig_path}")
        
        plt.show()
        return fig
    
    def plot_hyperparameter_sensitivity(self, save=True):
        """Plot hyperparameter sensitivity analysis chart"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Learning rate sensitivity
        lr_exps = self.results_df[self.results_df['experiment_name'].str.startswith('lr_')]
        if not lr_exps.empty:
            lr_exps = lr_exps.sort_values('learning_rate')
            axes[0].semilogx(lr_exps['learning_rate'], lr_exps['test_accuracy'], 
                           'o-', linewidth=2, markersize=8, label='Accuracy')
            axes[0].semilogx(lr_exps['learning_rate'], lr_exps['test_f1_macro'], 
                           's-', linewidth=2, markersize=8, label='F1 Score')
            axes[0].set_xlabel('Learning Rate (log scale)')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Learning Rate Sensitivity Analysis')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Mark best point
            best_idx = lr_exps['test_accuracy'].idxmax()
            axes[0].scatter(lr_exps.loc[best_idx, 'learning_rate'], 
                          lr_exps.loc[best_idx, 'test_accuracy'], 
                          color='red', s=200, zorder=5, 
                          label=f'Best: {lr_exps.loc[best_idx, "learning_rate"]:.1e}')
        
        # 2. Batch size sensitivity
        bs_exps = self.results_df[self.results_df['experiment_name'].str.startswith('bs_')]
        if not bs_exps.empty:
            bs_exps = bs_exps.sort_values('batch_size')
            axes[1].plot(bs_exps['batch_size'], bs_exps['test_accuracy'], 
                        'o-', linewidth=2, markersize=8)
            axes[1].set_xlabel('Batch Size')
            axes[1].set_ylabel('Test Accuracy')
            axes[1].set_title('Batch Size Sensitivity Analysis')
            axes[1].grid(True, alpha=0.3)
        
        # 3. All experiments performance ranking
        self.results_df = self.results_df.sort_values('test_accuracy', ascending=False)
        experiments = self.results_df['experiment_name'].tolist()
        accuracies = self.results_df['test_accuracy'].tolist()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
        bars = axes[2].barh(experiments, accuracies, color=colors)
        axes[2].set_xlabel('Test Accuracy')
        axes[2].set_title('Performance Ranking of All Experiments')
        axes[2].grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, acc in zip(bars, accuracies):
            axes[2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{acc:.3f}', va='center')
        
        plt.tight_layout()
        
        if save:
            fig_path = os.path.join(self.output_dir, "hyperparameter_sensitivity.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"📈 Hyperparameter sensitivity chart saved: {fig_path}")
        
        plt.show()
        return fig
    
    def plot_correlation_matrix(self, save=True):
        """Plot correlation matrix heatmap"""
        # Select numeric columns
        numeric_cols = ['learning_rate', 'batch_size', 'epochs', 
                       'test_accuracy', 'test_f1_macro', 'test_roc_auc', 'best_val_loss']
        
        # Filter existing columns
        available_cols = [col for col in numeric_cols if col in self.results_df.columns]
        numeric_df = self.results_df[available_cols]
        
        if len(available_cols) < 2:
            print("Insufficient numeric columns to calculate correlation")
            return
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add text
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black')
        
        # Set axis
        ax.set_xticks(np.arange(len(available_cols)))
        ax.set_yticks(np.arange(len(available_cols)))
        ax.set_xticklabels(available_cols, rotation=45, ha='right')
        ax.set_yticklabels(available_cols)
        
        ax.set_title('Correlation Matrix of Hyperparameters and Performance Metrics')
        
        # Add colorbar
        plt.colorbar(im)
        plt.tight_layout()
        
        if save:
            fig_path = os.path.join(self.output_dir, "correlation_matrix.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"📈 Correlation matrix chart saved: {fig_path}")
        
        plt.show()
        return fig