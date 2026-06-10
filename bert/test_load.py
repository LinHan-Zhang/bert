import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import config

class ChnSentiDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0), 
            'labels': torch.tensor(label, dtype=torch.long)
        }

print('1. 加载数据...')
df = pd.read_parquet(config.TRAIN_PATH)
print(f'   成功！数据量: {len(df)}')

print('2. 加载 tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, local_files_only=True)
print('   成功')

print('3. 创建 Dataset (只用前100条)...')
dataset = ChnSentiDataset(df.head(100), tokenizer, max_len=256)
print('   成功')

print('4. 创建 DataLoader...')
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
print('   成功')

print('5. 尝试获取第一个 batch...')
batch = next(iter(loader))
print(f'   ✅ 成功！input_ids 形状: {batch["input_ids"].shape}')
print(f'   ✅ attention_mask 形状: {batch["attention_mask"].shape}')
print(f'   ✅ labels 形状: {batch["labels"].shape}')

print('\n🎉 所有测试通过！DataLoader 工作正常！')