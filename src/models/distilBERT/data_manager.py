# data_manager.py
"""
数据管理器 - 加载Amazon评论数据
基于你的数据处理代码重构
"""

import pandas as pd
import numpy as np
import torch
import re
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """数据管理器"""
    
    def __init__(self, config=None):
        """
        初始化数据管理器
        
        Args:
            config: 配置字典，包含：
                - train_file: 训练数据文件路径
                - test_file: 测试数据文件路径
                - train_sample_ratio: 训练集采样比例
                - val_ratio: 验证集比例
                - model_path: 预训练模型路径
                - max_length: 最大文本长度
                - batch_size: 默认批次大小
        """
        self.config = config or {}
        
        # 默认配置
        self.default_config = {
            'train_file': './data/train.ft.txt',
            'test_file': './data/test.ft.txt',
            'train_sample_ratio': 0.1,
            'val_ratio': 0.1,
            'model_path': '/public/home/hx/NLP/NLP_learn/distilbert_base/distilbert-base-uncased',
            'max_length': 512,
            'batch_size': 16,
            'random_state': 42
        }
        
        # 合并配置
        self.config = {**self.default_config, **self.config}
        
        self.tokenizer = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        
        logger.info("数据管理器初始化完成")
    
    @staticmethod
    def read_amazon_file(filepath):
        """读取Amazon评论文件"""
        labels = []
        texts = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # 解析标签和文本
                match = re.match(r'^(__label__[12])\s+(.+)$', line)
                if match:
                    label = match.group(1)
                    text = match.group(2)
                    
                    # 转换标签: __label__1 -> 0, __label__2 -> 1
                    label_num = 0 if label == '__label__1' else 1
                    
                    labels.append(label_num)
                    texts.append(text.lower().strip())
        
        return pd.DataFrame({'label': labels, 'text': texts})
    
    def load_and_prepare_data(self):
        """加载和准备数据"""
        logger.info("加载数据集...")
        
        # 读取数据
        train_df = self.read_amazon_file(self.config['train_file'])
        test_df = self.read_amazon_file(self.config['test_file'])
        
        logger.info(f"原始数据: 训练集={len(train_df)}, 测试集={len(test_df)}")
        
        # 采样
        if self.config['train_sample_ratio'] < 1.0:
            train_df = train_df.sample(
                frac=self.config['train_sample_ratio'], 
                random_state=self.config['random_state']
            )
            logger.info(f"采样后训练集: {len(train_df)}")
        
        # 划分训练集和验证集
        train_df, val_df = train_test_split(
            train_df, 
            test_size=self.config['val_ratio'], 
            random_state=self.config['random_state']
        )
        
        # 提取文本和标签
        train_texts = train_df["text"].tolist()
        train_labels = train_df["label"].tolist()
        val_texts = val_df["text"].tolist()
        val_labels = val_df["label"].tolist()
        test_texts = test_df["text"].tolist()
        test_labels = test_df["label"].tolist()
        
        logger.info(f"数据集统计:")
        logger.info(f"  训练集: {len(train_texts)}")
        logger.info(f"  验证集: {len(val_texts)}")
        logger.info(f"  测试集: {len(test_texts)}")
        
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    
    def setup_tokenizer(self):
        """初始化分词器"""
        logger.info("初始化分词器...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.config['model_path'])
        return self.tokenizer
    
    def create_dataloaders(self, batch_size=None):
        """
        创建数据加载器
        
        Args:
            batch_size: 批次大小，如果为None则使用配置中的值
            
        Returns:
            train_dataloader, val_dataloader, test_dataloader
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        logger.info(f"创建数据加载器，批次大小: {batch_size}")
        
        # 加载数据
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = self.load_and_prepare_data()
        
        # 初始化分词器
        if self.tokenizer is None:
            self.setup_tokenizer()
        
        # 编码文本
        train_encodings = self.tokenizer(
            train_texts, truncation=True, padding=True, max_length=self.config['max_length']
        )
        val_encodings = self.tokenizer(
            val_texts, truncation=True, padding=True, max_length=self.config['max_length']
        )
        test_encodings = self.tokenizer(
            test_texts, truncation=True, padding=True, max_length=self.config['max_length']
        )
        
        # 转换为Tensor
        train_inputs = torch.tensor(train_encodings['input_ids'])
        train_masks = torch.tensor(train_encodings['attention_mask'])
        train_labels_tensor = torch.tensor(train_labels)
        
        val_inputs = torch.tensor(val_encodings['input_ids'])
        val_masks = torch.tensor(val_encodings['attention_mask'])
        val_labels_tensor = torch.tensor(val_labels)
        
        test_inputs = torch.tensor(test_encodings['input_ids'])
        test_masks = torch.tensor(test_encodings['attention_mask'])
        test_labels_tensor = torch.tensor(test_labels)
        
        # 创建数据集
        train_dataset = TensorDataset(train_inputs, train_masks, train_labels_tensor)
        val_dataset = TensorDataset(val_inputs, val_masks, val_labels_tensor)
        test_dataset = TensorDataset(test_inputs, test_masks, test_labels_tensor)
        
        # 创建数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size
        )
        
        self.test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=batch_size
        )
        
        logger.info("✅ 数据加载器创建完成")
        
        return self.train_dataloader, self.val_dataloader, self.test_dataloader
    
    def get_dataloaders(self, batch_size=None):
        """获取数据加载器"""
        if self.train_dataloader is None:
            return self.create_dataloaders(batch_size)
        return self.train_dataloader, self.val_dataloader, self.test_dataloader