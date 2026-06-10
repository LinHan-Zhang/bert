from __future__ import annotations

from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

from lexicon import SentimentLexicon
from models import SLKRoBERTa
from project_config import DEFAULT_MODEL_DIR, LABEL_NAMES


class SentimentPredictor:
    def __init__(self, model_dir: str | Path, fallback_tokenizer_dir: str | Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_dir = (
            self.model_dir
            if (self.model_dir / "vocab.txt").exists() or (self.model_dir / "tokenizer.json").exists()
            else Path(fallback_tokenizer_dir)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        architecture_path = self.model_dir / "architecture.json"
        self.is_slk = architecture_path.exists()
        if self.is_slk:
            architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
            lexicon_path = self.model_dir / "sentiment_lexicon.json"
            if not lexicon_path.exists():
                raise FileNotFoundError(
                    f"SLK-RoBERTa checkpoint is missing its lexicon: {lexicon_path}"
                )
            self.lexicon = SentimentLexicon.load(lexicon_path)
            config = BertConfig.from_pretrained(str(self.model_dir))
            self.model = SLKRoBERTa(
                config=config,
                lexicon_dim=architecture["lexicon_dim"],
                fusion_dim=architecture["fusion_dim"],
                dropout=architecture["dropout"],
                use_multi_pooling=architecture.get("use_multi_pooling", True),
                use_gated_fusion=architecture.get("use_gated_fusion", True),
                cls_token_id=architecture.get("cls_token_id", self.tokenizer.cls_token_id),
                sep_token_id=architecture.get("sep_token_id", self.tokenizer.sep_token_id),
                pad_token_id=architecture.get("pad_token_id", self.tokenizer.pad_token_id),
            )
            self.model.load_state_dict(
                torch.load(self.model_dir / "best_model.pt", map_location=self.device),
                strict=False,
            )
        else:
            self.lexicon = None
            self.model = BertForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str, max_length: int = 256):
        text = str(text).strip()
        if not text:
            raise ValueError("请输入非空中文评论。")
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if self.is_slk:
            lexicon_features = None
            if self.model.use_gated_fusion:
                lexicon_features = torch.tensor(
                    self.lexicon.extract(text), dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            output = self.model(**encoded, lexicon_features=lexicon_features)
            logits = output["logits"]
            gate_mean = float(output["gate"].mean().item()) if output["gate"] is not None else None
        else:
            logits = self.model(**encoded).logits
            gate_mean = None
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        label = int(probs.argmax())
        return {
            "label": label,
            "label_name": LABEL_NAMES[label],
            "negative_prob": float(probs[0]),
            "positive_prob": float(probs[1]),
            "confidence": float(probs[label]),
            "semantic_gate": gate_mean,
        }
