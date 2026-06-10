"""Render 构建阶段：下载推理所需资源（预训练 tokenizer + 可选微调权重）。"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOKENIZER_DIR = ROOT / "bert" / "model" / "chinese-roberta-wwm-ext"
DEFAULT_CHECKPOINT_DIR = ROOT / "models" / "tune_lr1e5_ep5"


def download_tokenizer() -> None:
    if (TOKENIZER_DIR / "vocab.txt").exists() or (TOKENIZER_DIR / "tokenizer.json").exists():
        print(f"Tokenizer 已存在: {TOKENIZER_DIR}")
        return

    from transformers import AutoTokenizer

    print("正在从 HuggingFace 下载 chinese-roberta-wwm-ext tokenizer …")
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"Tokenizer 已保存: {TOKENIZER_DIR}")


def download_checkpoint_from_url(url: str, dest: Path) -> None:
    if (dest / "best_model.pt").exists():
        print(f"微调权重已存在: {dest}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / "_checkpoint.zip"
    print(f"正在下载微调权重: {url}")
    urlretrieve(url, archive)

    with zipfile.ZipFile(archive, "r") as zf:
        names = zf.namelist()
        top_levels = {n.split("/")[0] for n in names if "/" in n}
        if len(top_levels) == 1 and (dest / "best_model.pt").exists() is False:
            inner = top_levels.pop()
            extract_to = dest.parent
            zf.extractall(extract_to)
            inner_path = extract_to / inner
            if inner_path.is_dir() and inner_path != dest:
                if dest.exists():
                    shutil.rmtree(dest)
                inner_path.rename(dest)
        else:
            zf.extractall(dest)

    archive.unlink(missing_ok=True)
    if not (dest / "best_model.pt").exists():
        raise FileNotFoundError(f"下载完成但未找到 best_model.pt: {dest}")
    print(f"微调权重已就绪: {dest}")


def download_checkpoint_from_hf(repo_id: str, dest: Path) -> None:
    if (dest / "best_model.pt").exists():
        print(f"微调权重已存在: {dest}")
        return

    from huggingface_hub import snapshot_download

    print(f"正在从 HuggingFace 下载 checkpoint: {repo_id}")
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(dest))
    if not (dest / "best_model.pt").exists():
        raise FileNotFoundError(f"HuggingFace 仓库中未找到 best_model.pt: {repo_id}")
    print(f"微调权重已就绪: {dest}")


def main() -> None:
    download_tokenizer()

    model_url = os.getenv("MODEL_URL", "").strip()
    model_hf_repo = os.getenv("MODEL_HF_REPO", "").strip()
    checkpoint_dir = Path(os.getenv("MODEL_DIR", str(DEFAULT_CHECKPOINT_DIR)))

    if model_url:
        download_checkpoint_from_url(model_url, checkpoint_dir)
    elif model_hf_repo:
        download_checkpoint_from_hf(model_hf_repo, checkpoint_dir)
    elif (checkpoint_dir / "best_model.pt").exists():
        print(f"使用本地 checkpoint: {checkpoint_dir}")
    else:
        print(
            "警告: 未设置 MODEL_URL / MODEL_HF_REPO，且本地无 best_model.pt。\n"
            "服务可启动，但情感预测需上传权重后再可用。\n"
            "在 Render 环境变量中设置 MODEL_URL（zip 直链）或 MODEL_HF_REPO（HF 仓库名）。"
        )


if __name__ == "__main__":
    main()
