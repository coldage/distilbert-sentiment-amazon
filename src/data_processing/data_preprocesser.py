from transformers import AutoTokenizer
import torch

# distilBERT对应的数据预处理函数
# 修改句子长度等，需要在函数体内部更改
# 实际上，这个函数没用到，分词操作在models/distilBERT.py/train_distilBERT里面
# def distilBERT_preprocessor(train_texts, train_labels, test_texts, test_labels):
#     # 创建分词器
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#     max_length = 512 # 最大句子长度

#     # 预处理
#     train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
#     train_inputs = torch.tensor(train_encodings['input_ids'])
#     train_masks = torch.tensor(train_encodings['attention_mask'])
#     train_labels = torch.tensor(train_labels)

#     test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)
#     test_inputs = torch.tensor(test_encodings['input_ids'])
#     test_masks = torch.tensor(test_encodings['attention_mask'])
#     test_labels = torch.tensor(test_labels)

#     return train_inputs, train_masks, train_labels, test_inputs, test_masks, test_labels