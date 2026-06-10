from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer=None,
        max_length: int = 256,
        lexicon=None,
        include_lexicon: bool = False,
    ):
        self.texts = [str(text) for text in texts]
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lexicon = lexicon
        self.include_lexicon = include_lexicon

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.tokenizer is None:
            return {"text": text, "label": label}
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": label,
        }
        if self.include_lexicon:
            if self.lexicon is None:
                raise ValueError("A sentiment lexicon is required for lexicon-derived inputs.")
            item["lexicon_features"] = torch.tensor(
                self.lexicon.extract(text), dtype=torch.float32
            )
        return item


def collate_batch(batch):
    """将 batch 中的张量字段正确堆叠。"""
    collated = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }
    if "lexicon_features" in batch[0]:
        collated["lexicon_features"] = torch.stack(
            [item["lexicon_features"] for item in batch]
        )
    return collated


def load_splits(data_dir: Path, max_samples: int | None = None) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    splits = {
        "train": pd.read_parquet(data_dir / "train_dataset.parquet"),
        "val": pd.read_parquet(data_dir / "val_dataset.parquet"),
        "test": pd.read_parquet(data_dir / "test_dataset.parquet"),
    }
    for name, df in splits.items():
        splits[name] = validate_dataframe(df, name)
        if max_samples:
            sample_size = min(max_samples, len(splits[name]))
            splits[name] = splits[name].sample(sample_size, random_state=42).reset_index(drop=True)
    return splits


def validate_dataframe(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    expected = {"text", "label"}
    if set(df.columns) != expected:
        raise ValueError(f"{split_name} columns must be {expected}, got {set(df.columns)}")
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    labels = set(df["label"].unique().tolist())
    if labels - {0, 1}:
        raise ValueError(f"{split_name} labels must be 0/1, got {labels}")
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int = 42):
    train_val, test = train_test_split(
        df, test_size=0.10, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=1 / 9, stratify=train_val["label"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
