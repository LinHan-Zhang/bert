from __future__ import annotations

import csv
import json
import os
import socket
import sys
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import SentimentPredictor  # noqa: E402

MODEL_DIR = ROOT / "models" / "tune_lr1e5_ep5"
DASHBOARD_DIR = ROOT / "dashboard"
FIGURES_DIR = ROOT / "reports" / "figures"
EXPERIMENT_CSV = ROOT / "reports" / "experiment_results.csv"
BERT_METRICS = ROOT / "bert" / "results" / "test_eval" / "metrics.json"
HISTORY_CSV = ROOT / "outputs" / "tune_lr1e5_ep5" / "history.csv"
TEST_PRED_CSV = ROOT / "outputs" / "tune_lr1e5_ep5" / "test_predictions.csv"
RUN_CONFIG = ROOT / "outputs" / "tune_lr1e5_ep5" / "run_config.json"
DATA_DIR = ROOT / "bert" / "data"
LEXICON_META = ROOT / "data" / "lexicon" / "lexicon_meta.json"

predictor: SentimentPredictor | None = None
recent_predictions: deque[dict[str, Any]] = deque(maxlen=50)
session_stats = {"total": 0, "positive": 0, "negative": 0}


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


def pct(value: float) -> float:
    return round(float(value) * 100, 2)


def active_model_dir() -> Path:
    if (MODEL_DIR / "best_model.pt").exists():
        return MODEL_DIR
    fallback = ROOT / "models" / "slk_roberta_fgm"
    if (fallback / "best_model.pt").exists():
        return fallback
    raise FileNotFoundError("未找到可用的模型权重 best_model.pt")


def load_predictor() -> SentimentPredictor:
    global predictor
    if predictor is None:
        predictor = SentimentPredictor(model_dir=active_model_dir())
    return predictor


def read_experiments() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not EXPERIMENT_CSV.exists():
        return rows
    with EXPERIMENT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "group": row["group"],
                    "run_name": row["run_name"],
                    "model": row["model"],
                    "f1": pct(float(row["f1"])),
                    "accuracy": pct(float(row["accuracy"])),
                    "precision": pct(float(row["precision"])),
                    "recall": pct(float(row["recall"])),
                    "epochs": row.get("epochs", ""),
                    "notes": row["notes"],
                }
            )
    return rows


def read_training_history() -> list[dict[str, Any]]:
    if not HISTORY_CSV.exists():
        return []
    with HISTORY_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_dataset_overview() -> dict[str, Any]:
    splits: dict[str, Any] = {}
    total = 0
    for name, filename in [
        ("train", "train_dataset.parquet"),
        ("val", "val_dataset.parquet"),
        ("test", "test_dataset.parquet"),
    ]:
        path = DATA_DIR / filename
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        pos = int((df["label"] == 1).sum())
        neg = int((df["label"] == 0).sum())
        avg_len = round(float(df["text"].astype(str).str.len().mean()), 1)
        splits[name] = {"total": len(df), "positive": pos, "negative": neg, "avg_length": avg_len}
        total += len(df)

    lexicon = {}
    if LEXICON_META.exists():
        lexicon = json.loads(LEXICON_META.read_text(encoding="utf-8"))

    return {
        "dataset_name": "ChnSentiCorp（酒店评论）",
        "splits": splits,
        "total_samples": total,
        "lexicon": {
            "positive_words": lexicon.get("positive_words", 0),
            "negative_words": lexicon.get("negative_words", 0),
            "source": lexicon.get("source", ""),
        },
    }


def read_confusion() -> dict[str, Any]:
    if not TEST_PRED_CSV.exists():
        return {"matrix": [[0, 0], [0, 0]], "labels": ["负面", "正面"], "accuracy": 0}
    df = pd.read_csv(TEST_PRED_CSV)
    tn = int(((df.label == 0) & (df.prediction == 0)).sum())
    fp = int(((df.label == 0) & (df.prediction == 1)).sum())
    fn = int(((df.label == 1) & (df.prediction == 0)).sum())
    tp = int(((df.label == 1) & (df.prediction == 1)).sum())
    acc = round((tn + tp) / len(df) * 100, 2)
    return {
        "matrix": [[tn, fp], [fn, tp]],
        "labels": ["负面", "正面"],
        "accuracy": acc,
        "errors": fp + fn,
    }


def read_model_info() -> dict[str, Any]:
    model_path = active_model_dir()
    arch = {}
    arch_path = model_path / "architecture.json"
    if arch_path.exists():
        arch = json.loads(arch_path.read_text(encoding="utf-8"))
    config = {}
    if RUN_CONFIG.exists():
        config = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    main = next((x for x in read_experiments() if x["run_name"] == "tune_lr1e5_ep5"), None)
    bert_f1 = None
    if BERT_METRICS.exists():
        bert_f1 = pct(json.loads(BERT_METRICS.read_text(encoding="utf-8"))["f1"])
    return {
        "name": "SLK-RoBERTa",
        "checkpoint": model_path.name,
        "backbone": "chinese-roberta-wwm-ext",
        "modules": ["多粒度池化", "情感词表门控", "FGM 对抗训练"],
        "architecture": arch,
        "training": {
            "lr": config.get("lr", 1e-5),
            "epochs": config.get("epochs", 5),
            "batch_size": config.get("batch_size", 32),
            "max_length": config.get("max_length", 256),
            "dropout": config.get("dropout", 0.2),
        },
        "metrics": main or {},
        "lift_vs_bert": None if bert_f1 is None or not main else round(main["f1"] - bert_f1, 2),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_predictor()
    print(f"Dashboard 模型已加载: {active_model_dir()}")
    yield


app = FastAPI(title="SLK-RoBERTa Dashboard", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/static.html")
def legacy_static() -> RedirectResponse:
    return RedirectResponse(url="/")


@app.get("/api/health")
def health() -> dict[str, Any]:
    model_path = active_model_dir()
    main = next((x for x in read_experiments() if x["run_name"] == "tune_lr1e5_ep5"), None)
    return {
        "status": "ok",
        "model": model_path.name,
        "model_f1": main["f1"] if main else 95.33,
        "experiment_count": len(read_experiments()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/api/dataset/overview")
def dataset_overview() -> dict[str, Any]:
    return read_dataset_overview()


@app.get("/api/analytics/model")
def analytics_model() -> dict[str, Any]:
    return read_model_info()


@app.get("/api/analytics/summary")
def analytics_summary() -> dict[str, Any]:
    experiments = read_experiments()
    main = next((x for x in experiments if x["run_name"] == "tune_lr1e5_ep5"), None)
    if main is None:
        raise HTTPException(status_code=404, detail="未找到主模型实验数据")
    dataset = read_dataset_overview()
    confusion = read_confusion()
    return {
        "kpis": {
            "accuracy": main["accuracy"],
            "precision": main["precision"],
            "recall": main["recall"],
            "f1": main["f1"],
            "total_samples": dataset["total_samples"],
            "test_samples": dataset["splits"].get("test", {}).get("total", 914),
            "experiment_runs": len(experiments),
            "lexicon_words": dataset["lexicon"]["positive_words"] + dataset["lexicon"]["negative_words"],
            "test_errors": confusion["errors"],
        },
        "experiments": experiments,
        "session": session_stats,
        "dataset": dataset,
        "model": read_model_info(),
    }


@app.get("/api/analytics/charts")
def analytics_charts() -> dict[str, Any]:
    experiments = read_experiments()

    def pick(group: str, names: list[str] | None = None) -> list[dict]:
        items = [x for x in experiments if x["group"] == group]
        if names:
            order = {name: idx for idx, name in enumerate(names)}
            items = [x for x in items if x["run_name"] in order]
            items.sort(key=lambda x: order[x["run_name"]])
        return items

    main_compare = pick("main", ["roberta_ep5", "tune_lr1e5_ep5"])
    bert_metrics = None
    if BERT_METRICS.exists():
        bert_metrics = json.loads(BERT_METRICS.read_text(encoding="utf-8"))
        main_compare = [
            {
                "run_name": "bert_test_eval",
                "model": "BERT",
                "f1": pct(bert_metrics["f1"]),
                "accuracy": pct(bert_metrics["accuracy"]),
                "precision": pct(bert_metrics["precision"]),
                "recall": pct(bert_metrics["recall"]),
            },
            {
                "run_name": "roberta_ep5",
                "model": "RoBERTa",
                "f1": next(x["f1"] for x in main_compare if x["run_name"] == "roberta_ep5"),
                "accuracy": next(x["accuracy"] for x in main_compare if x["run_name"] == "roberta_ep5"),
                "precision": next(x["precision"] for x in main_compare if x["run_name"] == "roberta_ep5"),
                "recall": next(x["recall"] for x in main_compare if x["run_name"] == "roberta_ep5"),
            },
            {
                "run_name": "tune_lr1e5_ep5",
                "model": "SLK-RoBERTa",
                "f1": next(x["f1"] for x in main_compare if x["run_name"] == "tune_lr1e5_ep5"),
                "accuracy": next(x["accuracy"] for x in main_compare if x["run_name"] == "tune_lr1e5_ep5"),
                "precision": next(x["precision"] for x in main_compare if x["run_name"] == "tune_lr1e5_ep5"),
                "recall": next(x["recall"] for x in main_compare if x["run_name"] == "tune_lr1e5_ep5"),
            },
        ]

    phase1_runs = pick("ablation3ep")
    phase1_full = next((x for x in read_experiments() if x["run_name"] == "slk_roberta_fgm"), None)
    phase1_lookup = {x["run_name"]: x["f1"] for x in phase1_runs}
    if phase1_full:
        phase1_lookup["slk_roberta_fgm"] = phase1_full["f1"]

    phase2 = pick(
        "main",
        ["tune_lr1e5_ep5", "ablation_pool_only_ep5", "ablation_lexicon_only_ep5", "ablation_no_fgm_ep5"],
    )
    phase2_short = ["SLK 全开", "w/o 词表", "w/o 多池化", "w/o FGM"]
    phase2_keys = [x["model"] for x in phase2]
    phase1_f1 = [
        phase1_lookup.get("slk_roberta_fgm", 94.78),
        phase1_lookup.get("ablation_pool_only", 95.21),
        phase1_lookup.get("ablation_lexicon_only", 95.0),
        phase1_lookup.get("ablation_no_fgm", 94.93),
    ]

    dataset = read_dataset_overview()
    split_labels = []
    split_values = []
    for key in ["train", "val", "test"]:
        if key in dataset["splits"]:
            split_labels.append(key.upper())
            split_values.append(dataset["splits"][key]["total"])

    main_model = next((x for x in experiments if x["run_name"] == "tune_lr1e5_ep5"), None)
    radar = []
    if main_model:
        radar = [
            main_model["accuracy"],
            main_model["precision"],
            main_model["recall"],
            main_model["f1"],
        ]

    prob_bins = {"0-0.2": 0, "0.2-0.5": 0, "0.5-0.8": 0, "0.8-1.0": 0}
    if TEST_PRED_CSV.exists():
        df = pd.read_csv(TEST_PRED_CSV)
        for p in df.apply(lambda r: max(r["negative_prob"], r["positive_prob"]), axis=1):
            if p < 0.2:
                prob_bins["0-0.2"] += 1
            elif p < 0.5:
                prob_bins["0.2-0.5"] += 1
            elif p < 0.8:
                prob_bins["0.5-0.8"] += 1
            else:
                prob_bins["0.8-1.0"] += 1

    return {
        "main_comparison": {
            "labels": [x["model"] for x in main_compare],
            "f1": [x["f1"] for x in main_compare],
            "metrics": main_compare,
        },
        "phase1_ablation": {
            "labels": ["SLK 全开"] + [x["model"] for x in pick("ablation3ep")],
            "f1": [phase1_lookup.get("slk_roberta_fgm", 94.78)]
            + [x["f1"] for x in pick("ablation3ep")],
        },
        "phase2_ablation": {
            "labels": phase2_keys,
            "f1": [x["f1"] for x in phase2],
        },
        "phase_comparison": {
            "labels": phase2_short,
            "phase1": phase1_f1,
            "phase2": [x["f1"] for x in phase2],
        },
        "lexicon_evolution": {
            "labels": [x["model"] for x in pick("lexicon")],
            "f1": [x["f1"] for x in pick("lexicon")],
        },
        "dataset_split": {"labels": split_labels, "values": split_values},
        "radar": radar,
        "confidence_distribution": {
            "labels": list(prob_bins.keys()),
            "values": list(prob_bins.values()),
        },
        "confusion": read_confusion(),
        "training_history": read_training_history(),
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="请输入非空文本")
    try:
        result = load_predictor().predict(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    session_stats["total"] += 1
    if result["label"] == 1:
        session_stats["positive"] += 1
    else:
        session_stats["negative"] += 1

    payload = {
        "text": text,
        "label": result["label_name"],
        "label_id": result["label"],
        "confidence": round(result["confidence"], 4),
        "negative_prob": round(result["negative_prob"], 4),
        "positive_prob": round(result["positive_prob"], 4),
        "semantic_gate": None if result["semantic_gate"] is None else round(result["semantic_gate"], 4),
        "analysis": build_analysis(text, result),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    recent_predictions.appendleft(payload)
    return payload


@app.get("/api/predict/recent")
def recent() -> dict[str, Any]:
    return {"items": list(recent_predictions), "session": session_stats}


def build_analysis(text: str, result: dict[str, Any]) -> str:
    label = result["label_name"]
    conf = result["confidence"]
    pos = result["positive_prob"]
    neg = result["negative_prob"]
    gate = result.get("semantic_gate")
    strength = "较强" if conf >= 0.9 else "中等" if conf >= 0.75 else "偏弱"
    gate_text = f"词表门控 {gate:.3f}，" if gate is not None else ""
    return (
        f"SLK-RoBERTa 判定为{label}情感，置信度 {conf:.1%}（{strength}）。"
        f"正/负概率 {pos:.1%} / {neg:.1%}；{gate_text}文本 {len(text)} 字。"
    )


def find_free_port(start: int = 8080, end: int = 8090) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"端口 {start}-{end} 均被占用")


if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")
if DASHBOARD_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DASHBOARD_DIR), name="assets")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT") or find_free_port())
    print(f"统一 Dashboard 启动: http://0.0.0.0:{port}")
    print("=" * 56)
    print("本机访问:  http://127.0.0.1:{port}".format(port=port))
    print("外网分享（AutoDL）:")
    print(f"  1. 打开 AutoDL 控制台 → 自定义服务")
    print(f"  2. 添加端口 {port}，协议选 http")
    print(f"  3. 复制生成的「公网链接」发给他人即可访问")
    print("=" * 56)
    uvicorn.run(app, host="0.0.0.0", port=port)
