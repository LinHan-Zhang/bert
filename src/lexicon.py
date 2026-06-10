from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


DEFAULT_POSITIVE_WORDS = {
    "不错": 1.0, "优秀": 1.4, "满意": 1.2, "喜欢": 1.1, "推荐": 1.2,
    "值得": 1.0, "方便": 0.8, "干净": 1.0, "舒适": 1.1, "热情": 1.0,
    "漂亮": 1.0, "惊喜": 1.3, "完美": 1.6, "好用": 1.1, "实用": 0.9,
    "清晰": 0.8, "快速": 0.8, "及时": 0.8, "耐用": 1.0, "便宜": 0.8,
    "划算": 1.1, "愉快": 1.1, "贴心": 1.2, "丰富": 0.8, "稳定": 0.9,
    "流畅": 1.0, "精致": 1.0, "友好": 0.9, "专业": 0.9, "赞": 1.2,
}

DEFAULT_NEGATIVE_WORDS = {
    "不好": 1.0, "糟糕": 1.5, "失望": 1.3, "后悔": 1.2, "差劲": 1.4,
    "难用": 1.1, "麻烦": 0.8, "脏": 1.0, "吵": 0.8, "慢": 0.7,
    "贵": 0.7, "敷衍": 1.2, "生气": 1.2, "破损": 1.3, "卡顿": 1.0,
    "模糊": 0.8, "难吃": 1.2, "异味": 1.1, "简陋": 0.9, "错误": 0.8,
    "骗人": 1.5, "垃圾": 1.6, "退货": 1.2, "投诉": 1.1, "故障": 1.2,
    "发热": 0.8, "下沉": 0.8, "担心": 0.7, "不值": 1.1, "不推荐": 1.3,
}

NEGATIONS = {"不", "没", "没有", "未", "并非", "不是", "毫不", "别", "无"}
DEGREE_WORDS = {
    "极其": 1.8,
    "非常": 1.6,
    "特别": 1.5,
    "太": 1.4,
    "很": 1.3,
    "比较": 1.1,
    "稍微": 0.7,
    "有点": 0.7,
}


def _load_weighted_words(path: Path | None, defaults: dict[str, float]) -> dict[str, float]:
    if path is None:
        return dict(defaults)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {path}")
    words = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        words[parts[0]] = float(parts[1]) if len(parts) > 1 else 1.0
    return words


class SentimentLexicon:
    """Weighted sentiment lexicon used for features and weak supervision."""

    feature_dim = 6

    def __init__(
        self,
        positive_path: Path | None = None,
        negative_path: Path | None = None,
        positive_words: dict[str, float] | None = None,
        negative_words: dict[str, float] | None = None,
        context_window: int = 4,
    ):
        self.positive_words = (
            dict(positive_words)
            if positive_words is not None
            else _load_weighted_words(positive_path, DEFAULT_POSITIVE_WORDS)
        )
        self.negative_words = (
            dict(negative_words)
            if negative_words is not None
            else _load_weighted_words(negative_path, DEFAULT_NEGATIVE_WORDS)
        )
        self.context_window = context_window

    def _sentiment_matches(self, text: str):
        matches = []
        for word, weight in self.positive_words.items():
            start = 0
            while True:
                index = text.find(word, start)
                if index < 0:
                    break
                matches.append((index, index + len(word), float(weight)))
                start = index + len(word)
        for word, weight in self.negative_words.items():
            start = 0
            while True:
                index = text.find(word, start)
                if index < 0:
                    break
                matches.append((index, index + len(word), -float(weight)))
                start = index + len(word)

        selected = []
        occupied = set()
        for start, end, score in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0])):
            positions = set(range(start, end))
            if occupied.isdisjoint(positions):
                selected.append((start, end, score))
                occupied.update(positions)
        return sorted(selected)

    @staticmethod
    def _count_longest_matches(text: str, terms) -> int:
        occupied = set()
        count = 0
        for term in sorted(terms, key=len, reverse=True):
            start = 0
            while True:
                index = text.find(term, start)
                if index < 0:
                    break
                positions = set(range(index, index + len(term)))
                if occupied.isdisjoint(positions):
                    occupied.update(positions)
                    count += 1
                start = index + len(term)
        return count

    def score(self, text: str) -> dict[str, float]:
        text = str(text)
        raw_score = 0.0
        positive_magnitude = 0.0
        negative_magnitude = 0.0
        degree_hits = 0
        negation_hits = 0
        matches = self._sentiment_matches(text)

        for start, _, base_score in matches:
            context = text[max(0, start - self.context_window):start]
            degree = 1.0
            for word, multiplier in DEGREE_WORDS.items():
                if word in context:
                    degree = max(degree, multiplier)
                    degree_hits += context.count(word)
            negations = self._count_longest_matches(context, NEGATIONS)
            negation_hits += negations
            adjusted = base_score * degree * (-1.0 if negations % 2 else 1.0)
            raw_score += adjusted
            if adjusted >= 0:
                positive_magnitude += adjusted
            else:
                negative_magnitude += abs(adjusted)

        normalized_score = math.tanh(raw_score / 3.0)
        return {
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "intensity": (normalized_score + 1.0) / 2.0,
            "positive_magnitude": positive_magnitude,
            "negative_magnitude": negative_magnitude,
            "match_count": float(len(matches)),
            "negation_count": float(negation_hits),
            "degree_count": float(degree_hits),
        }

    def extract(self, text: str) -> np.ndarray:
        result = self.score(text)
        length = max(len(str(text)), 1)
        return np.asarray(
            [
                min(result["positive_magnitude"] / 5.0, 1.0),
                min(result["negative_magnitude"] / 5.0, 1.0),
                result["normalized_score"],
                min(result["match_count"] * 10.0 / length, 1.0),
                min(result["negation_count"] / 3.0, 1.0),
                min(result["degree_count"] / 3.0, 1.0),
            ],
            dtype=np.float32,
        )

    def save(self, path: Path):
        payload = {
            "version": 1,
            "context_window": self.context_window,
            "positive_words": self.positive_words,
            "negative_words": self.negative_words,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            positive_words=payload["positive_words"],
            negative_words=payload["negative_words"],
            context_window=int(payload.get("context_window", 4)),
        )
