# experiment_config.py
"""
实验配置定义
"""

class ExperimentConfig:
    """单个实验配置"""
    
    def __init__(self, 
                 name,
                 experiment_type="freeze_backbone",
                 learning_rate=2e-5,
                 batch_size=16,
                 epochs=10,
                 output_base_path="/distilbert-sentiment-amazon/output/distilBERT/experiments"):
        
        self.name = name
        self.experiment_type = experiment_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_base_path = output_base_path
        
        # 固定参数
        self.model_path = '/public/home/hx/NLP/NLP_learn/distilbert_base/distilbert-base-uncased'
        self.num_labels = 2
        self.max_length = 512
        self.weight_decay = 0.01
        self.eps = 1e-8
        self.gradient_clip = 1.0
    
    def to_dict(self):
        """转换为字典"""
        return {
            'name': self.name,
            'experiment_type': self.experiment_type,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'model_path': self.model_path,
            'num_labels': self.num_labels
        }
    
    def __str__(self):
        return (f"Experiment: {self.name}\n"
                f"  Type: {self.experiment_type}\n"
                f"  LR: {self.learning_rate:.1e}\n"
                f"  Batch: {self.batch_size}\n"
                f"  Epochs: {self.epochs}")


class ExperimentSuite:
    """实验套件生成器"""
    
    @staticmethod
    def create_ablation_experiments():
        """创建消融实验"""
        experiments = []
        
        strategies = [
            ("no_finetune", "不微调"),
            ("freeze_backbone", "冻结主体"),
            ("partial_finetune", "部分微调"),
            ("full_finetune", "全微调")
        ]
        
        for strategy, desc in strategies:
            exp = ExperimentConfig(
                name=f"abl_{strategy}",
                experiment_type=strategy,
                learning_rate=2e-5,
                batch_size=16,
                epochs=10
            )
            experiments.append(exp)
        
        return experiments
    
    @staticmethod
    def create_lr_sensitivity_experiments():
        """创建学习率敏感性实验"""
        experiments = []
        
        learning_rates = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
        
        for lr in learning_rates:
            exp = ExperimentConfig(
                name=f"lr_{lr:.1e}",
                experiment_type="full_finetune",
                learning_rate=lr,
                batch_size=16,
                epochs=10
            )
            experiments.append(exp)
        
        return experiments
    
    @staticmethod
    def create_batch_size_experiments():
        """创建批次大小敏感性实验"""
        experiments = []
        
        batch_sizes = [8, 16, 32, 64]
        
        for bs in batch_sizes:
            exp = ExperimentConfig(
                name=f"bs_{bs}",
                experiment_type="full_finetune",
                learning_rate=2e-5,
                batch_size=bs,
                epochs=10
            )
            experiments.append(exp)
        
        return experiments
    
    @staticmethod
    def create_all_experiments():
        """创建所有实验"""
        experiments = []
        
        experiments.extend(ExperimentSuite.create_ablation_experiments())
        experiments.extend(ExperimentSuite.create_lr_sensitivity_experiments())
        experiments.extend(ExperimentSuite.create_batch_size_experiments())
        
        return experiments