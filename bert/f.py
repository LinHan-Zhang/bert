import io
import sys

import pandas as pd

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("正在加载训练集与验证集...")
train_df = pd.read_parquet(config.TRAIN_PATH)
val_df = pd.read_parquet(config.VAL_PATH)

# 转化为列表供后续 Tokenizer 使用
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

val_texts = val_df['text'].tolist()
val_labels = val_df['label'].tolist()

print(f"✅ 加载成功！训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}")

