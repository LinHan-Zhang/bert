from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据与预训练模型（与 bert/ 目录对齐，换机器无需改路径）
DEFAULT_DATA_DIR = PROJECT_ROOT / "bert" / "data"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "bert" / "model" / "chinese-roberta-wwm-ext"
DEFAULT_STOPWORDS_PATH = DEFAULT_DATA_DIR / "stopwords_baidu.txt"

LOCAL_DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

LABEL_NAMES = {0: "负面", 1: "正面"}
