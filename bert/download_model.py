import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config

local_model_path = config.MODEL_PATH

print(f"正在下载模型到: {local_model_path}")

# 下载模型和 tokenizer（需要联网）
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=2)

# 保存到本地
tokenizer.save_pretrained(local_model_path)
model.save_pretrained(local_model_path)

print(f"✅ 模型已保存到: {local_model_path}")
print(f"文件列表: {os.listdir(local_model_path)}")