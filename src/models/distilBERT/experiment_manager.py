# experiment_manager.py
"""
实验管理器 - 管理所有实验的运行
"""

import os
import json
import pandas as pd
import concurrent.futures
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, output_dir="./experiment_results"):
        """初始化实验管理器"""
        self.output_dir = output_dir
        self.experiments = []
        self.results = []
        
        # 直接使用传入的目录，不再创建嵌套子目录
        self.main_dir = output_dir
        os.makedirs(self.main_dir, exist_ok=True)
        
        logger.info(f"实验管理器初始化完成")
        logger.info(f"主输出目录: {self.main_dir}")
    
    def add_experiment(self, config):
        """添加实验配置"""
        self.experiments.append(config)
    
    def add_experiments(self, configs):
        """添加多个实验配置"""
        self.experiments.extend(configs)
    
    def run_all_experiments(self, data_manager, max_workers=2):
        """
        运行所有实验
        
        Args:
            data_manager: DataManager实例
            max_workers: 最大并行数
        """
        logger.info(f"开始运行 {len(self.experiments)} 个实验...")
        
        self.results = []
        
        # 使用线程池并行运行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for config in self.experiments:
                # 为每个实验创建运行器
                from experiment_runner import ExperimentRunner
                runner = ExperimentRunner(config, self.main_dir)
                
                # 根据实验的批次大小创建数据加载器
                train_dataloader, val_dataloader, test_dataloader = data_manager.create_dataloaders(
                    batch_size=config.batch_size
                )
                
                # 提交任务
                future = executor.submit(
                    runner.run,
                    train_dataloader,
                    val_dataloader,
                    test_dataloader
                )
                futures[future] = config.name
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                exp_name = futures[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    logger.info(f"✅ 实验 {exp_name} 完成")
                except Exception as e:
                    logger.error(f"❌ 实验 {exp_name} 失败: {str(e)}")
        
        # 保存结果汇总
        self._save_results_summary()
        
        logger.info(f"\n🎉 所有实验完成!")
        logger.info(f"成功: {len(self.results)}/{len(self.experiments)}")
        
        return self.results
    
    def _save_results_summary(self):
        """保存所有实验结果的汇总"""
        if not self.results:
            logger.warning("没有实验结果可保存")
            return
        
        # 提取关键信息
        summary_data = []
        for result in self.results:
            config = result['config']
            summary_data.append({
                'experiment_name': result['experiment_name'],
                'experiment_type': config['experiment_type'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'epochs': config['epochs'],
                'best_epoch': result['best_epoch'],
                'best_val_loss': result['best_val_loss'],
                'test_accuracy': result['test_results']['accuracy'],
                'test_f1_macro': result['test_results'].get('f1_macro', 0),
                'test_roc_auc': result['test_results'].get('roc_auc', 0),
                'experiment_dir': result['experiment_dir']
            })
        
        # 保存为CSV
        df_summary = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(self.main_dir, "experiment_summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        
        # 保存为JSON
        summary_json_path = os.path.join(self.main_dir, "experiment_summary.json")
        with open(summary_json_path, 'w') as f:
            json.dump({
                'total_experiments': len(self.experiments),
                'successful_experiments': len(self.results),
                'results': summary_data
            }, f, indent=2)
        
        logger.info(f"📋 实验结果汇总已保存: {summary_csv_path}")
        
        return df_summary