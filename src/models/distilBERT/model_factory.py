# model_factory.py
"""
模型工厂 - 基于你的create_distilBERT函数
"""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂"""
    
    @staticmethod
    def create_model(model_path, experiment_type="freeze_backbone", num_labels=2):
        """创建模型"""
        logger.info(f"创建模型 - 策略: {experiment_type}")
        
        # 加载模型
        model = DistilBertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels
        )
        
        # 应用微调策略
        ModelFactory.apply_finetuning_strategy(model, experiment_type)
        
        # 统计参数
        trainable_params, total_params = ModelFactory.count_parameters(model)
        logger.info(f"可训练参数: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        return model
    
    @staticmethod
    def apply_finetuning_strategy(model, strategy):
        """应用微调策略"""
        if strategy == "no_finetune":
            for param in model.parameters():
                param.requires_grad = False
        
        elif strategy == "freeze_backbone":
            for param in model.distilbert.parameters():
                param.requires_grad = False
        
        elif strategy == "partial_finetune":
            # 先全部冻结
            for param in model.parameters():
                param.requires_grad = False
            
            # 解冻最后3层
            num_layers = len(model.distilbert.transformer.layer)
            for i in range(num_layers - 3, num_layers):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = True
            
            # 解冻分类头
            for param in model.pre_classifier.parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif strategy == "full_finetune":
            # 全微调，默认所有参数可训练
            pass
    
    @staticmethod
    def count_parameters(model):
        """统计参数数量"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params
    
    @staticmethod
    def create_tokenizer(model_path):
        """创建分词器"""
        return DistilBertTokenizerFast.from_pretrained(model_path)