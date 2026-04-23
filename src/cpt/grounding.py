from __future__ import annotations

import re
from typing import Any

import numpy as np


def extract_genres_from_text(text: str) -> set[str]:
    match = re.search(r"are:(.*?)(</genres>|<eos>|$)", text)
    if not match:
        return set()
    raw = match.group(1).strip().replace(" ", "")
    if not raw:
        return set()
    return {part for part in raw.split(",") if part}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_genre_prompt(text: str) -> str:
    marker = "are:"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[: idx + len(marker)]


def predict_from_prompt(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int = 30,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
    )
    return tokenizer.decode(out[0])


def mean_jaccard_genres(
    ds_meta_genre: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    n: int = 200,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    idxs = rng.integers(low=0, high=len(ds_meta_genre), size=n)

    scores = []
    empty_pred = 0
    for idx in idxs:
        ids = ds_meta_genre[int(idx)]["input_ids"]
        true_text = tokenizer.decode(ids)
        prompt = build_genre_prompt(true_text)
        pred_text = predict_from_prompt(prompt, model, tokenizer, device)

        true_set = extract_genres_from_text(true_text)
        pred_set = extract_genres_from_text(pred_text)
        if not pred_set:
            empty_pred += 1
        scores.append(jaccard(true_set, pred_set))

    scores_arr = np.array(scores, dtype=np.float32)
    return {
        "n": int(n),
        "mean_jaccard": float(scores_arr.mean()),
        "std_jaccard": float(scores_arr.std()),
        "empty_pred_frac": float(empty_pred / n),
        "scores": scores_arr,
    }
