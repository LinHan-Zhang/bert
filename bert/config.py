import os

# 项目根目录（本文件所在目录），换机器不用改路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "chinese-roberta-wwm-ext")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TRAIN_PATH = os.path.join(DATA_DIR, "train_dataset.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val_dataset.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test_dataset.parquet")

BATCH_SIZE = 32          # RTX 3090 24GB 可用 32；显存不足改回 16
MAX_LEN = 256
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42
NUM_WORKERS = 2        # DataLoader 进程数，Linux GPU 训练推荐 2
