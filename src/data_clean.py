from __future__ import annotations

import argparse
import re
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from wordcloud import WordCloud

from project_config import DEFAULT_DATA_DIR, DEFAULT_STOPWORDS_PATH, PROJECT_ROOT
from data_utils import stratified_split


plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def clean_text(text: str) -> str:
    """保留常见中文、英文、数字和基础标点，去除空值和异常字符。"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[^\w\s\u4e00-\u9fa5，。！？、；：,.!?;:（）()《》“”\"'-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_chnsenticorp() -> pd.DataFrame:
    dataset = load_dataset("lansinuote/ChnSentiCorp")
    frames = [pd.DataFrame(dataset[name]) for name in ["train", "validation", "test"]]
    return pd.concat(frames, ignore_index=True)


def balance_classes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    min_class_size = df["label"].value_counts().min()
    return (
        df.groupby("label", group_keys=False)
        .sample(n=min_class_size, random_state=seed)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def make_visualizations(df: pd.DataFrame, output_dir: Path, stopwords_path: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths = df["text"].astype(str).str.len()

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50, color="#4E79A7", edgecolor="white")
    plt.axvline(lengths.mean(), color="#E15759", linestyle="--", label=f"平均长度: {lengths.mean():.1f}")
    plt.title("评论文本长度分布")
    plt.xlabel("字符长度")
    plt.ylabel("样本数")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "text_length_distribution.png", dpi=180)
    plt.close()

    if not stopwords_path or not stopwords_path.exists():
        return

    stop_words = set(stopwords_path.read_text(encoding="utf-8").splitlines())
    font_candidates = [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    font_path = next((p for p in font_candidates if p.exists()), None)
    if font_path is None:
        return

    for label, name in [(0, "negative"), (1, "positive")]:
        words = []
        for text in df[df["label"] == label]["text"]:
            words.extend(word for word in jieba.cut(str(text)) if len(word) > 1 and word not in stop_words)
        if not words:
            continue
        wc = WordCloud(
            font_path=str(font_path),
            background_color="white",
            width=900,
            height=600,
            max_words=120,
            collocations=False,
        )
        wc.generate(" ".join(words)).to_file(output_dir / f"wordcloud_{name}.png")


def main():
    parser = argparse.ArgumentParser(description="清洗 ChnSentiCorp 并生成 Parquet 数据集。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stopwords-path", type=Path, default=DEFAULT_STOPWORDS_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("正在加载 ChnSentiCorp 数据集...")
    df = load_chnsenticorp()
    print(f"原始样本数: {len(df)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"清洗去重后样本数: {len(df)}")

    df = balance_classes(df, seed=args.seed)
    print("均衡后标签分布:")
    print(df["label"].value_counts().sort_index())

    train, val, test = stratified_split(df, seed=args.seed)
    train.to_parquet(output_dir / "train_dataset.parquet", index=False)
    val.to_parquet(output_dir / "val_dataset.parquet", index=False)
    test.to_parquet(output_dir / "test_dataset.parquet", index=False)

    viz_dir = PROJECT_ROOT / "outputs" / "data"
    make_visualizations(train, viz_dir, args.stopwords_path)
    print(f"训练/验证/测试: {len(train)}/{len(val)}/{len(test)}")
    print(f"图表输出目录: {viz_dir}")


if __name__ == "__main__":
    main()
