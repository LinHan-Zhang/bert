import os
import sys
from unittest.mock import MagicMock

sys.modules['transformers.safetensors_conversion'] = MagicMock()

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import config


class ChnSentiDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        print(f"正在预处理 {len(dataframe)} 条数据...")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = dataframe["label"].tolist()
        
        # 提前 tokenize 所有文本（避免训练时重复计算）
        self.input_ids = []
        self.attention_masks = []
        
        texts = dataframe["text"].tolist()
        for idx, text in enumerate(texts):
            if idx % 1000 == 0:
                print(f"  处理进度: {idx}/{len(texts)}")
            
            encoding = tokenizer(
                str(text),
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            self.input_ids.append(encoding["input_ids"].flatten())
            self.attention_masks.append(encoding["attention_mask"].flatten())
        
        print("预处理完成！")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


if __name__ == "__main__":
    print("1. 正在加载本地 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, local_files_only=True)

    print("2. 正在读取清洗好的 Parquet 数据...")
    train_df = pd.read_parquet(config.TRAIN_PATH)
    print(f"   数据量: {len(train_df)} 条")

    print("3. 正在构建 Dataset 和 DataLoader...")
    train_dataset = ChnSentiDataset(train_df, tokenizer, max_len=256)  # 先用 256，512 太慢
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=0  # 避免 Windows 多进程问题
    )

    print("\n✅ 数据管道构建完成，测试提取一个 Batch：")
    batch = next(iter(train_loader))

    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print("\n数据管道正常工作！")