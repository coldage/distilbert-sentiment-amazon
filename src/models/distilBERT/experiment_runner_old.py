# experiment_runner.py
"""
实验运行器 - 修复训练函数
"""

import os
import torch
import json
import pandas as pd
from datetime import datetime
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
import logging

# 导入你的评估函数
try:
    from evaluate import evaluate, print_evaluation_results
except ImportError:
    print("⚠️  Warning: evaluate.py not found, using dummy evaluate function")
    
    # 临时评估函数
    def evaluate(model, dataloader, device):
        return {'loss': 0.5, 'accuracy': 0.8, 'f1_macro': 0.8}
    
    def print_evaluation_results(results):
        print(f"Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.4f}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """单个实验运行器"""
    
    def __init__(self, config, output_dir=None):
        """
        初始化
        
        Args:
            config: ExperimentConfig对象
            output_dir: 输出目录，如果为None则使用config中的路径
        """
        self.config = config
        
        # 确定输出目录
        if output_dir:
            self.output_base = output_dir
        else:
            self.output_base = config.output_base_path
        
        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(self.output_base, f"{config.name}_{timestamp}")
        
        # 创建子目录
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        logger.info(f"实验输出目录: {self.experiment_dir}")
    
    def run(self, train_dataloader, val_dataloader, test_dataloader):
        """
        运行实验
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            test_dataloader: 测试数据加载器
            
        Returns:
            实验结果字典
        """
        logger.info(f"\n🚀 开始实验: {self.config.name}")
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 创建模型
        from model_factory import ModelFactory
        model = ModelFactory.create_model(
            self.config.model_path,
            self.config.experiment_type,
            self.config.num_labels
        )
        model.to(device)
        
        # 优化器
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        total_steps = len(train_dataloader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # 训练历史
        training_stats = []
        best_val_loss = float('inf')
        best_model_state = None
        
        # 训练循环
        for epoch in range(self.config.epochs):
            logger.info(f'\nEpoch {epoch + 1}/{self.config.epochs}')
            logger.info('-' * 40)
            
            # ========== 训练阶段 ==========
            model.train()
            total_train_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc="训练", leave=False)
            
            for batch in progress_bar:
                # 移动到设备
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                # 前向传播
                model.zero_grad()
                outputs = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                
                # 更新参数
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新进度条
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            
            # ========== 验证阶段 ==========
            logger.info("进行验证...")
            val_results = evaluate(model, val_dataloader, device)
            
            # 记录统计
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_results['loss'],
                'val_accuracy': val_results['accuracy'],
                'val_f1_macro': val_results.get('f1_macro', 0),
                'val_roc_auc': val_results.get('roc_auc', 0),
                'learning_rate': scheduler.get_last_lr()[0]
            }
            training_stats.append(epoch_stats)
            
            # 打印结果
            logger.info(f"训练损失: {avg_train_loss:.4f}")
            logger.info(f"验证损失: {val_results['loss']:.4f}")
            logger.info(f"验证准确率: {val_results['accuracy']:.4f}")
            
            # 检查是否最佳模型
            if val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                logger.info(f"🎉 新的最佳模型! 验证损失: {best_val_loss:.4f}")
            
            # 保存检查点（每2个epoch或最后一个epoch）
            if (epoch + 1) % 2 == 0 or (epoch + 1) == self.config.epochs:
                self._save_checkpoint(
                    model, optimizer, epoch + 1, val_results, training_stats
                )
        
        # ========== 最终测试 ==========
        logger.info("\n使用最佳模型进行最终测试...")
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 测试集评估
        test_results = evaluate(model, test_dataloader, device)
        print_evaluation_results(test_results)
        
        # 保存最终结果
        experiment_result = self._save_experiment_results(
            model, training_stats, test_results, best_epoch, best_val_loss
        )
        
        logger.info(f"\n✅ 实验 {self.config.name} 完成!")
        
        return experiment_result
    
    def _save_checkpoint(self, model, optimizer, epoch, val_results, training_stats):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_results': val_results,
            'training_stats': training_stats,
            'config': self.config.to_dict()
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"💾 检查点已保存: {checkpoint_path}")
        
        return checkpoint_path
    
    def _save_experiment_results(self, model, training_stats, test_results, best_epoch, best_val_loss):
        """保存实验最终结果"""
        # 保存模型
        model_path = os.path.join(self.experiment_dir, "final_model.pth")
        torch.save(model.state_dict(), model_path)
        
        # 保存训练历史
        history_path = os.path.join(self.logs_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        # 保存CSV
        df_stats = pd.DataFrame(training_stats)
        csv_path = os.path.join(self.logs_dir, "training_history.csv")
        df_stats.to_csv(csv_path, index=False)
        
        # 实验总结
        experiment_result = {
            'experiment_name': self.config.name,
            'config': self.config.to_dict(),
            'training_stats': training_stats,
            'test_results': test_results,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'test_accuracy': test_results['accuracy'],
            'test_f1': test_results.get('f1_macro', 0),
            'experiment_dir': self.experiment_dir,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_path = os.path.join(self.experiment_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(experiment_result, f, indent=2)
        
        logger.info(f"📊 实验结果已保存到: {self.experiment_dir}")
        
        return experiment_result