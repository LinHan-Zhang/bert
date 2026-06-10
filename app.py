#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简易 Gradio 入口（可选）。推荐使用统一 Dashboard：python dashboard_app.py"""

import socket
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import SentimentPredictor

MODEL_DIR = Path("models/tune_lr1e5_ep5")
if not (MODEL_DIR / "best_model.pt").exists():
    MODEL_DIR = Path("models/slk_roberta_fgm")

if not (MODEL_DIR / "best_model.pt").exists():
    raise FileNotFoundError(f"模型文件不存在: {MODEL_DIR / 'best_model.pt'}")

print(f"正在加载 SLK-RoBERTa 模型 ({MODEL_DIR.name})...")
predictor = SentimentPredictor(model_dir=MODEL_DIR)
print("模型加载完成！（完整 Dashboard 请运行: python dashboard_app.py）")


def find_free_port(start: int = 7860, end: int = 7870) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"端口 {start}-{end} 均被占用，请先关闭其他 Gradio 进程。")


def predict_sentiment(text: str):
    if not text or not text.strip():
        return "请输入非空评论"
    try:
        result = predictor.predict(text)
        return f"{result['label_name']} （置信度: {result['confidence']:.4f}）"
    except Exception as e:
        return f"预测出错: {str(e)}"


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="例如：这家酒店非常干净，服务态度也很好，推荐！", label="输入中文评论"),
    outputs=gr.Textbox(label="情感分析结果"),
    title="SLK-RoBERTa 情感分析（精简版）",
    description=(
        "完整可视化 Dashboard 请运行 **`python dashboard_app.py`**（端口 8080）。\n"
        "本页为精简 Gradio 推理界面。"
    ),
    examples=[
        ["房间很干净，床也很舒服，下次还会来。"],
        ["服务态度极差，等了两个小时还没上菜，再也不来了。"],
    ],
)

if __name__ == "__main__":
    port = find_free_port()
    if port != 7860:
        print(f"端口 7860 已被占用，改用端口 {port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
