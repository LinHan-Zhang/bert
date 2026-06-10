#!/usr/bin/env python3
"""Generate experiment figures for SLK-RoBERTa report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NOTO_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(NOTO_CJK).exists():
    fm.fontManager.addfont(NOTO_CJK)
    _cjk_name = fm.FontProperties(fname=NOTO_CJK).get_name()
    plt.rcParams["font.sans-serif"] = [_cjk_name, "DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180

COLORS = {
    "primary": "#2E86AB",
    "highlight": "#A23B72",
    "secondary": "#6C757D",
    "accent": "#F18F01",
    "good": "#28A745",
    "phase1": "#7FB3D5",
    "phase2": "#1B4F72",
}


def pct(x: float) -> float:
    return round(x * 100, 2)


def load_f1(rel_path: str) -> float:
    data = json.loads((ROOT / rel_path / "metrics.json").read_text(encoding="utf-8"))
    return pct(data["f1"])


def bar_chart(
    labels,
    values,
    title,
    filename,
    colors=None,
    ylim=(93.5, 96.0),
    annotate_best=True,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors or COLORS["primary"], width=0.62, edgecolor="white")
    if annotate_best:
        best = max(values)
        for bar, val in zip(bars, values):
            if val == best:
                bar.set_color(COLORS["highlight"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test F1 (%)")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def plot_phase1_ablation():
    labels = ["SLK 全开", "w/o 词表", "w/o 多池化", "w/o FGM"]
    values = [
        load_f1("outputs/slk_roberta_fgm"),
        load_f1("outputs/ablation_pool_only"),
        load_f1("outputs/ablation_lexicon_only"),
        load_f1("outputs/ablation_no_fgm"),
    ]
    colors = [COLORS["phase1"], COLORS["highlight"], COLORS["phase1"], COLORS["phase1"]]
    bar_chart(
        labels,
        values,
        "第一阶段消融（epoch=3，默认设置）",
        "fig1_phase1_ablation.png",
        colors=colors,
        annotate_best=False,
    )


def plot_lexicon_evolution():
    labels = ["默认小词表", "DLUT 大词表", "merged 词表", "merged + 微调"]
    values = [
        load_f1("outputs/slk_roberta_fgm"),
        load_f1("outputs/slk_full_big_lexicon"),
        load_f1("outputs/slk_full_merged_lexicon"),
        load_f1("outputs/tune_lr1e5_ep5"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, values, marker="o", linewidth=2.2, color=COLORS["primary"], markersize=8)
    ax.fill_between(range(len(labels)), values, alpha=0.08, color=COLORS["primary"])
    ax.set_ylabel("Test F1 (%)")
    ax.set_title("词表优化与微调过程")
    ax.set_ylim(94.5, 95.6)
    ax.grid(alpha=0.25, linestyle="--")
    for i, val in enumerate(values):
        ax.text(i, val + 0.05, f"{val:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_lexicon_evolution.png", bbox_inches="tight")
    plt.close(fig)


def plot_phase2_ablation():
    labels = ["SLK 全开", "w/o 词表", "w/o 多池化", "w/o FGM"]
    values = [
        load_f1("outputs/tune_lr1e5_ep5"),
        load_f1("outputs/ablation_pool_only_ep5"),
        load_f1("outputs/ablation_lexicon_only_ep5"),
        load_f1("outputs/ablation_no_fgm_ep5"),
    ]
    bar_chart(
        labels,
        values,
        "第二阶段消融（lr=1e-5，同条件对比）",
        "fig3_phase2_ablation.png",
        ylim=(93.8, 95.6),
    )


def plot_main_vs_baselines():
    labels = ["BERT", "RoBERTa", "SLK-RoBERTa"]
    values = [
        load_f1("bert/results/test_eval"),
        load_f1("outputs/roberta_ep5"),
        load_f1("outputs/tune_lr1e5_ep5"),
    ]
    colors = [COLORS["secondary"], COLORS["secondary"], COLORS["highlight"]]
    bar_chart(
        labels,
        values,
        "最终主结果对比",
        "fig4_main_vs_baselines.png",
        colors=colors,
        ylim=(93.5, 96.0),
        annotate_best=False,
    )


def plot_phase_comparison():
    configs = ["SLK 全开", "w/o 词表", "w/o 多池化", "w/o FGM"]
    phase1 = [
        load_f1("outputs/slk_roberta_fgm"),
        load_f1("outputs/ablation_pool_only"),
        load_f1("outputs/ablation_lexicon_only"),
        load_f1("outputs/ablation_no_fgm"),
    ]
    phase2 = [
        load_f1("outputs/tune_lr1e5_ep5"),
        load_f1("outputs/ablation_pool_only_ep5"),
        load_f1("outputs/ablation_lexicon_only_ep5"),
        load_f1("outputs/ablation_no_fgm_ep5"),
    ]
    x = np.arange(len(configs))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - width / 2, phase1, width, label="第一阶段（epoch=3）", color=COLORS["phase1"])
    b2 = ax.bar(x + width / 2, phase2, width, label="第二阶段（lr=1e-5）", color=COLORS["phase2"])
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=12, ha="right")
    ax.set_ylabel("Test F1 (%)")
    ax.set_title("两阶段实验 F1 对比")
    ax.set_ylim(94.0, 95.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_two_phase_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix():
    pred_path = ROOT / "outputs/tune_lr1e5_ep5/test_predictions.csv"
    df = pd.read_csv(pred_path)
    cm = confusion_matrix(df["label"], df["prediction"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["负面", "正面"])
    ax.set_yticks([0, 1], labels=["负面", "正面"])
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("最终模型混淆矩阵（SLK-RoBERTa）")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def main():
    plot_phase1_ablation()
    plot_lexicon_evolution()
    plot_phase2_ablation()
    plot_main_vs_baselines()
    plot_phase_comparison()
    plot_confusion_matrix()
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
