"""Build weighted sentiment lexicon files from the public DLUT dictionary."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd

from lexicon import DEFAULT_NEGATIVE_WORDS, DEFAULT_POSITIVE_WORDS
from project_config import DEFAULT_DATA_DIR, PROJECT_ROOT


DEFAULT_DLUT_CSV = PROJECT_ROOT / "data" / "lexicon" / "raw" / "dlut.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "lexicon"


def intensity_to_weight(intensity: float) -> float:
    """Map DLUT intensity (1-9) to lexicon weight (0.7-1.6)."""
    intensity = max(1.0, min(9.0, float(intensity)))
    return round(0.7 + (intensity / 9.0) * 0.9, 2)


def load_dlut_words(csv_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", engine="python")
    df.columns = [str(column).strip() for column in df.columns]

    positive: dict[str, float] = {}
    negative: dict[str, float] = {}

    for _, row in df.iterrows():
        word = str(row["词语"]).strip()
        if not word or word == "nan":
            continue
        polarity = row.get("极性")
        intensity = row.get("强度", 5.0)
        if pd.isna(polarity) or pd.isna(intensity):
            continue

        polarity = int(float(polarity))
        weight = intensity_to_weight(float(intensity))
        if polarity == 1:
            positive[word] = max(positive.get(word, 0.0), weight)
        elif polarity == 2:
            negative[word] = max(negative.get(word, 0.0), weight)

    return positive, negative


def corpus_token_counts(texts) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(jieba.lcut(str(text)))
        counts.update(jieba.lcut(str(text), cut_all=True))
    return counts


def substring_train_counts(texts, candidates) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        text = str(text)
        for word in candidates:
            if word in text:
                counts[word] += 1
    return counts


def select_words(
    words: dict[str, float],
    token_counts: Counter,
    texts=None,
    *,
    min_len: int = 2,
    max_len: int = 6,
    top_k: int = 1200,
    min_hits: int = 1,
    min_intensity: float = 5.0,
) -> dict[str, float]:
    filtered = {
        word: weight
        for word, weight in words.items()
        if min_len <= len(word) <= max_len
    }
    ranked = sorted(
        filtered.items(),
        key=lambda item: (token_counts.get(item[0], 0), item[1], len(item[0])),
        reverse=True,
    )
    selected: dict[str, float] = {}
    for word, weight in ranked:
        if token_counts.get(word, 0) >= min_hits:
            selected[word] = weight
        if len(selected) >= top_k:
            break

    if texts is not None and len(selected) < top_k:
        strong_candidates = {
            word: weight
            for word, weight in filtered.items()
            if weight >= intensity_to_weight(min_intensity) and word not in selected
        }
        strong_candidates = dict(
            sorted(
                strong_candidates.items(),
                key=lambda item: (item[1], len(item[0])),
                reverse=True,
            )[:800]
        )
        substring_hits = substring_train_counts(texts, strong_candidates.keys())
        extras = sorted(
            strong_candidates.items(),
            key=lambda item: (substring_hits.get(item[0], 0), item[1], len(item[0])),
            reverse=True,
        )
        for word, weight in extras:
            if substring_hits.get(word, 0) < max(2, min_hits):
                continue
            selected[word] = weight
            if len(selected) >= top_k:
                break
    return selected


def is_valid_word(word: str, min_len: int, max_len: int) -> bool:
    if not word or len(word) < min_len or len(word) > max_len:
        return False
    if word.isdigit():
        return False
    return bool(re.fullmatch(r"[\u4e00-\u9fa5]+", word))


def mine_train_words(
    train_df: pd.DataFrame,
    *,
    min_len: int = 2,
    max_len: int = 6,
    top_k: int = 300,
    min_hits: int = 5,
    min_log_odds: float = 0.35,
) -> tuple[dict[str, float], dict[str, float]]:
    """Mine polarity-specific words from labeled train texts only."""
    pos_doc_counts: Counter = Counter()
    neg_doc_counts: Counter = Counter()
    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())

    for text, label in zip(train_df["text"], train_df["label"]):
        tokens = set()
        text = str(text)
        for token in jieba.lcut(text):
            if is_valid_word(token, min_len, max_len):
                tokens.add(token)
        for token in jieba.lcut(text, cut_all=True):
            if is_valid_word(token, min_len, max_len):
                tokens.add(token)
        if int(label) == 1:
            pos_doc_counts.update(tokens)
        else:
            neg_doc_counts.update(tokens)

    scored: list[tuple[str, float, float]] = []
    vocabulary = set(pos_doc_counts) | set(neg_doc_counts)
    for word in vocabulary:
        pos_hits = pos_doc_counts[word]
        neg_hits = neg_doc_counts[word]
        if pos_hits + neg_hits < min_hits:
            continue
        pos_ratio = (pos_hits + 1.0) / (n_pos + 2.0)
        neg_ratio = (neg_hits + 1.0) / (n_neg + 2.0)
        log_odds = math.log(pos_ratio / neg_ratio)
        if abs(log_odds) < min_log_odds:
            continue
        weight = round(min(1.5, max(0.9, 0.95 + abs(log_odds) * 0.25)), 2)
        scored.append((word, log_odds, weight))

    scored.sort(key=lambda item: (abs(item[1]), item[2], len(item[0])), reverse=True)

    mined_pos: dict[str, float] = {}
    mined_neg: dict[str, float] = {}
    for word, log_odds, weight in scored:
        if log_odds > 0 and len(mined_pos) < top_k:
            mined_pos[word] = weight
        elif log_odds < 0 and len(mined_neg) < top_k:
            mined_neg[word] = weight
        if len(mined_pos) >= top_k and len(mined_neg) >= top_k:
            break
    return mined_pos, mined_neg


def merge_word_maps(*maps: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for word_map in maps:
        for word, weight in word_map.items():
            merged[word] = max(merged.get(word, 0.0), float(weight))
    return merged


def write_lexicon(path: Path, words: dict[str, float]):
    lines = sorted(words.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    path.write_text(
        "\n".join(f"{word} {weight:.2f}" for word, weight in lines) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build DLUT-based sentiment lexicon files.")
    parser.add_argument("--dlut-csv", type=Path, default=DEFAULT_DLUT_CSV)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_DATA_DIR / "train_dataset.parquet")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pos-file", default="pos.txt")
    parser.add_argument("--neg-file", default="neg.txt")
    parser.add_argument("--top-k", type=int, default=1200, help="Max words per polarity after filtering.")
    parser.add_argument("--min-hits", type=int, default=1, help="Minimum token frequency in train corpus.")
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=6)
    parser.add_argument("--mine-top-k", type=int, default=250, help="Max mined words per polarity from train.")
    parser.add_argument("--mine-min-hits", type=int, default=5, help="Min doc frequency for mined words.")
    parser.add_argument(
        "--mine-min-log-odds",
        type=float,
        default=0.35,
        help="Minimum |log-odds| for mined words.",
    )
    parser.add_argument(
        "--no-mine",
        action="store_true",
        help="Disable train-set word mining; use DLUT + defaults only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positive, negative = load_dlut_words(args.dlut_csv)
    train_df = pd.read_parquet(args.train_path)
    train_texts = train_df["text"].tolist()
    token_counts = corpus_token_counts(train_texts)

    selected_pos = select_words(
        positive,
        token_counts,
        train_texts,
        min_len=args.min_len,
        max_len=args.max_len,
        top_k=args.top_k,
        min_hits=args.min_hits,
    )
    selected_neg = select_words(
        negative,
        token_counts,
        train_texts,
        min_len=args.min_len,
        max_len=args.max_len,
        top_k=args.top_k,
        min_hits=args.min_hits,
    )

    mined_pos: dict[str, float] = {}
    mined_neg: dict[str, float] = {}
    if not args.no_mine:
        mined_pos, mined_neg = mine_train_words(
            train_df,
            min_len=args.min_len,
            max_len=args.max_len,
            top_k=args.mine_top_k,
            min_hits=args.mine_min_hits,
            min_log_odds=args.mine_min_log_odds,
        )

    final_pos = merge_word_maps(DEFAULT_POSITIVE_WORDS, selected_pos, mined_pos)
    final_neg = merge_word_maps(DEFAULT_NEGATIVE_WORDS, selected_neg, mined_neg)

    pos_path = args.output_dir / args.pos_file
    neg_path = args.output_dir / args.neg_file
    write_lexicon(pos_path, final_pos)
    write_lexicon(neg_path, final_neg)

    meta = {
        "source": "大连理工大学情感词汇本体 (DLUT) + train corpus mining",
        "source_url": "https://github.com/yizhanmiao/DLUT-Emotionontology",
        "citation": "徐琳宏, 林鸿飞, 潘宇, 等. 情感词汇本体的构造[J]. 情报学报, 2008, 27(2): 180-185.",
        "positive_words": len(final_pos),
        "negative_words": len(final_neg),
        "dlut_selected_positive": len(selected_pos),
        "dlut_selected_negative": len(selected_neg),
        "mined_positive": len(mined_pos),
        "mined_negative": len(mined_neg),
        "top_k_per_polarity": args.top_k,
        "min_hits": args.min_hits,
        "mine_top_k": args.mine_top_k,
        "mine_min_hits": args.mine_min_hits,
        "mine_min_log_odds": args.mine_min_log_odds,
        "train_corpus_filtered": True,
        "train_mining_enabled": not args.no_mine,
    }
    (args.output_dir / "lexicon_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Positive words: {len(final_pos)} -> {pos_path}")
    print(f"Negative words: {len(final_neg)} -> {neg_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
