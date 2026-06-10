from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_evaluation(output_dir: Path, labels, preds, probs=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(labels, preds)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = classification_report(
        labels, preds, target_names=["负面", "正面"], digits=4, zero_division=0
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    data = {"label": labels, "prediction": preds}
    if probs is not None:
        data["negative_prob"] = np.asarray(probs)[:, 0]
        data["positive_prob"] = np.asarray(probs)[:, 1]
    pd.DataFrame(data).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


def plot_confusion_matrix(cm, path: Path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["负面", "正面"])
    plt.yticks([0, 1], ["负面", "正面"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_history(history, path: Path):
    df = pd.DataFrame(history)
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.figure(figsize=(8, 5))
    if "train_loss" in df:
        plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    if "val_loss" in df:
        plt.plot(df["epoch"], df["val_loss"], marker="o", label="Val Loss")
    if "val_f1" in df:
        plt.plot(df["epoch"], df["val_f1"], marker="o", label="Val F1")
    plt.xlabel("Epoch")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
