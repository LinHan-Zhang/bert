import os
import random
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import config

sys.modules["transformers.safetensors_conversion"] = MagicMock()

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


class ChnSentiDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        print(f"Preprocessing {len(dataframe)} samples...", flush=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = dataframe["label"].tolist()

        self.input_ids = []
        self.attention_masks = []

        texts = dataframe["text"].tolist()
        total_texts = len(texts)
        for idx, text in enumerate(texts, start=1):
            if idx == 1 or idx % 200 == 0 or idx == total_texts:
                print(f"  Preprocess progress: {idx}/{total_texts}", flush=True)

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

        print("Preprocessing complete.", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_batches = len(dataloader)

    print(f"Starting evaluation over {total_batches} batches", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

            if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
                print(
                    f"  Eval batch {batch_idx}/{total_batches} | "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

    avg_loss = total_loss / max(total_batches, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            flush=True,
        )
        return device
    print("Using device: cpu (CUDA not available, check PyTorch GPU install)", flush=True)
    return torch.device("cpu")


def train():
    set_seed(config.SEED)

    model_path = config.MODEL_PATH
    results_dir = config.RESULTS_DIR
    train_path = config.TRAIN_PATH
    val_path = config.VAL_PATH

    batch_size = config.BATCH_SIZE
    max_len = config.MAX_LEN
    num_epochs = config.NUM_EPOCHS
    learning_rate = config.LEARNING_RATE

    os.makedirs(results_dir, exist_ok=True)

    device = get_device()
    use_cuda = device.type == "cuda"

    print("Loading tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        local_files_only=True,
    )
    model.to(device)
    print("Model loaded.", flush=True)

    print("Building dataloaders...", flush=True)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    train_dataset = ChnSentiDataset(train_df, tokenizer, max_len=max_len)
    val_dataset = ChnSentiDataset(val_df, tokenizer, max_len=max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS if use_cuda else 0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS if use_cuda else 0,
        pin_memory=use_cuda,
    )
    print("Dataloaders ready.", flush=True)
    print(
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}",
        flush=True,
    )
    print(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}",
        flush=True,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(results_dir, "best_model")

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}", flush=True)
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_count = 0
        total_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_train_correct += (preds == labels).sum().item()
            total_train_count += labels.size(0)

            if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
                running_loss = total_train_loss / batch_idx
                running_acc = total_train_correct / max(total_train_count, 1)
                print(
                    f"  Train batch {batch_idx}/{total_batches} | "
                    f"loss={loss.item():.4f} avg_loss={running_loss:.4f} "
                    f"avg_acc={running_acc:.4f}",
                    flush=True,
                )

        train_loss = total_train_loss / max(total_batches, 1)
        train_acc = total_train_correct / max(total_train_count, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"Best model saved to: {best_model_path}", flush=True)

    print(f"Training finished. Best val_acc: {best_val_acc:.4f}", flush=True)


if __name__ == "__main__":
    train()
