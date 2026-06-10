from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_utils import SentimentDataset, collate_batch, load_splits
from lexicon import SentimentLexicon
from metrics import compute_metrics, plot_history, save_evaluation
from models import FGM, SLKRoBERTa, TextCNN
from project_config import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, MODEL_OUTPUT_DIR, OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train sentiment analysis models.")
    parser.add_argument("--model", choices=["roberta", "textcnn", "slk_roberta"], default="roberta")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pretrained-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    fgm_group = parser.add_mutually_exclusive_group()
    fgm_group.add_argument("--use-fgm", dest="use_fgm", action="store_true")
    fgm_group.add_argument("--no-fgm", dest="use_fgm", action="store_false")
    parser.set_defaults(use_fgm=None)
    parser.add_argument("--fgm-epsilon", type=float, default=1.0)
    parser.add_argument("--positive-lexicon", type=Path, default=None)
    parser.add_argument("--negative-lexicon", type=Path, default=None)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--multi-pooling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Mean/Max/Self-Attention pooling; disable to use CLS only.",
    )
    parser.add_argument(
        "--gated-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse semantic and lexicon features with a learned gate.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Use a subset for smoke tests.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_textcnn(batch, tokenizer, max_length: int):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "label": labels}


def prepare_batch(batch, device, model_type):
    model_inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["label"].to(device),
    }
    if model_type == "slk_roberta" and "lexicon_features" in batch:
        model_inputs["lexicon_features"] = batch["lexicon_features"].to(device)
    return model_inputs


def train_one_epoch(model, loader, optimizer, scheduler, device, model_type, use_fgm=False, fgm=None):
    model.train()
    losses = []
    for batch in loader:
        model_inputs = prepare_batch(batch, device, model_type)

        optimizer.zero_grad()
        output = model(**model_inputs)
        loss = output["loss"] if isinstance(output, dict) else output.loss
        loss.backward()

        if use_fgm and fgm is not None:
            fgm.attack()
            adv_output = model(**model_inputs)
            adv_loss = adv_output["loss"] if isinstance(adv_output, dict) else adv_output.loss
            adv_loss.backward()
            fgm.restore()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return {"loss": float(np.mean(losses))}


@torch.no_grad()
def evaluate(model, loader, device, model_type):
    model.eval()
    losses, all_labels, all_preds, all_probs = [], [], [], []
    for batch in loader:
        model_inputs = prepare_batch(batch, device, model_type)
        labels = model_inputs["labels"]
        output = model(**model_inputs)
        loss = output["loss"] if isinstance(output, dict) else output.loss
        logits = output["logits"] if isinstance(output, dict) else output.logits
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        losses.append(loss.item())
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = float(np.mean(losses))
    return metrics, all_labels, all_preds, all_probs


def save_model(model, tokenizer, model_dir: Path, model_type: str, lexicon=None):
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir)
    if model_type == "roberta":
        model.save_pretrained(model_dir)
    elif model_type == "slk_roberta":
        torch.save(model.state_dict(), model_dir / "best_model.pt")
        model.config.save_pretrained(model_dir)
        architecture = {
            "model_type": "slk_roberta",
            "lexicon_dim": model.lexicon_dim,
            "fusion_dim": model.fusion_dim,
            "dropout": model.dropout_rate,
            "use_multi_pooling": model.use_multi_pooling,
            "use_gated_fusion": model.use_gated_fusion,
            "cls_token_id": model.cls_token_id,
            "sep_token_id": model.sep_token_id,
            "pad_token_id": model.pad_token_id,
        }
        (model_dir / "architecture.json").write_text(
            json.dumps(architecture, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if lexicon is None:
            raise ValueError("SLK-RoBERTa checkpoints must include their sentiment lexicon.")
        lexicon.save(model_dir / "sentiment_lexicon.json")
    else:
        torch.save(model.state_dict(), model_dir / "best_model.pt")
        config = {
            "model_type": "textcnn",
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
        }
        (model_dir / "model_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_best_model(model_dir: Path, model_type: str, tokenizer, device):
    if model_type == "roberta":
        return BertForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    if model_type == "slk_roberta":
        architecture = json.loads((model_dir / "architecture.json").read_text(encoding="utf-8"))
        config = BertConfig.from_pretrained(str(model_dir))
        model = SLKRoBERTa(
            config=config,
            lexicon_dim=architecture["lexicon_dim"],
            fusion_dim=architecture["fusion_dim"],
            dropout=architecture["dropout"],
            use_multi_pooling=architecture.get("use_multi_pooling", True),
            use_gated_fusion=architecture.get("use_gated_fusion", True),
            cls_token_id=architecture.get("cls_token_id", tokenizer.cls_token_id),
            sep_token_id=architecture.get("sep_token_id", tokenizer.sep_token_id),
            pad_token_id=architecture.get("pad_token_id", tokenizer.pad_token_id),
        )
        model.load_state_dict(
            torch.load(model_dir / "best_model.pt", map_location=device),
            strict=False,
        )
        return model.to(device)
    model = TextCNN(vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    return model


def validate_paths(data_dir: Path, pretrained_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    for name in ("train_dataset.parquet", "val_dataset.parquet", "test_dataset.parquet"):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"缺少数据文件: {path}")
    if not pretrained_dir.exists():
        raise FileNotFoundError(f"预训练模型目录不存在: {pretrained_dir}")
    weights = list(pretrained_dir.glob("*.safetensors")) + list(pretrained_dir.glob("pytorch_model.bin"))
    if not weights:
        raise FileNotFoundError(f"预训练模型目录中未找到权重文件: {pretrained_dir}")


def main():
    args = parse_args()
    if args.use_fgm is None:
        args.use_fgm = args.model == "slk_roberta"
    set_seed(args.seed)
    run_name = args.run_name or f"{args.model}{'_fgm' if args.use_fgm else ''}"
    output_dir = OUTPUT_DIR / run_name
    model_dir = MODEL_OUTPUT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    validate_paths(args.data_dir, args.pretrained_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print(f"Using device: {device}", flush=True)
    print(f"Run: {run_name}", flush=True)

    splits = load_splits(args.data_dir, max_samples=args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(str(args.pretrained_dir))
    lexicon = SentimentLexicon(args.positive_lexicon, args.negative_lexicon)

    if args.model == "roberta":
        model = BertForSequenceClassification.from_pretrained(
            str(args.pretrained_dir), num_labels=2, ignore_mismatched_sizes=True
        )
        train_dataset = SentimentDataset(splits["train"]["text"], splits["train"]["label"], tokenizer, args.max_length)
        val_dataset = SentimentDataset(splits["val"]["text"], splits["val"]["label"], tokenizer, args.max_length)
        test_dataset = SentimentDataset(splits["test"]["text"], splits["test"]["label"], tokenizer, args.max_length)
        collate_fn = collate_batch
    elif args.model == "slk_roberta":
        model = SLKRoBERTa(
            pretrained_dir=args.pretrained_dir,
            lexicon_dim=lexicon.feature_dim,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            use_multi_pooling=args.multi_pooling,
            use_gated_fusion=args.gated_fusion,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        include_lexicon = args.gated_fusion
        train_dataset = SentimentDataset(
            splits["train"]["text"],
            splits["train"]["label"],
            tokenizer,
            args.max_length,
            lexicon,
            include_lexicon,
        )
        val_dataset = SentimentDataset(
            splits["val"]["text"],
            splits["val"]["label"],
            tokenizer,
            args.max_length,
            lexicon,
            include_lexicon,
        )
        test_dataset = SentimentDataset(
            splits["test"]["text"],
            splits["test"]["label"],
            tokenizer,
            args.max_length,
            lexicon,
            include_lexicon,
        )
        collate_fn = collate_batch
    else:
        model = TextCNN(vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id)
        train_dataset = SentimentDataset(splits["train"]["text"], splits["train"]["label"])
        val_dataset = SentimentDataset(splits["val"]["text"], splits["val"]["label"])
        test_dataset = SentimentDataset(splits["test"]["text"], splits["test"]["label"])
        collate_fn = lambda batch: collate_textcnn(batch, tokenizer, args.max_length)

    model.to(device)
    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, collate_fn=collate_fn, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn, **loader_kwargs
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total_steps * 0.1)), num_training_steps=total_steps
    )
    fgm = (
        FGM(model, epsilon=args.fgm_epsilon)
        if args.use_fgm and args.model in {"roberta", "slk_roberta"}
        else None
    )

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.model, args.use_fgm, fgm
        )
        val_metrics, _, _, _ = evaluate(model, val_loader, device, args.model)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)
        print(row)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_model(model, tokenizer, model_dir, args.model, lexicon)

    plot_history(history, output_dir / "history.png")

    model = load_best_model(model_dir, args.model, tokenizer, device)

    test_metrics, labels, preds, probs = evaluate(model, test_loader, device, args.model)
    save_evaluation(output_dir, labels, preds, probs)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("Test metrics:", test_metrics)
    print(f"Saved model: {model_dir}")
    print(f"Saved outputs: {output_dir}")


if __name__ == "__main__":
    main()
