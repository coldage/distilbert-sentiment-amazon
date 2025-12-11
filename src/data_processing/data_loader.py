import torch
from torch.utils.data import Dataset

# 数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    # 用分词器去处理texts和labels
    # 可以根据序号获取到单个数据点的信息
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 对文本进行分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long)
        }