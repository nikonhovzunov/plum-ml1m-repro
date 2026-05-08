from __future__ import annotations

import math
from collections.abc import Iterable

from .decoding import dedupe_preserve_order


def recall_at_k(candidates: Iterable[int], target: int, k: int) -> float:
    return float(int(int(target) in dedupe_preserve_order(candidates)[: int(k)]))


def ndcg_at_k(candidates: Iterable[int], target: int, k: int) -> float:
    target = int(target)
    for rank, item in enumerate(dedupe_preserve_order(candidates)[: int(k)], start=1):
        if int(item) == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def mrr_at_k(candidates: Iterable[int], target: int, k: int) -> float:
    target = int(target)
    for rank, item in enumerate(dedupe_preserve_order(candidates)[: int(k)], start=1):
        if int(item) == target:
            return 1.0 / rank
    return 0.0


def coverage_at_k(records: Iterable[dict], k: int) -> int:
    recommended: set[int] = set()
    for record in records:
        recommended.update(dedupe_preserve_order(record.get("candidates", []))[: int(k)])
    return len(recommended)


def validate_k_values(k_values: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(k) for k in k_values)
    if not values:
        raise ValueError("k_values must not be empty")
    if any(k <= 0 for k in values):
        raise ValueError("all k_values must be positive")
    if len(set(values)) != len(values):
        raise ValueError("k_values must be unique")
    return values


def evaluate_rankings(records: Iterable[dict], k_values: tuple[int, ...] = (1, 5, 10)) -> dict:
    k_values = validate_k_values(k_values)
    records = list(records)
    metrics: dict[str, float | int] = {"n": len(records)}
    if not records:
        for k in k_values:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
            metrics[f"mrr@{k}"] = 0.0
            metrics[f"coverage@{k}"] = 0
        return metrics

    for k in k_values:
        metrics[f"recall@{k}"] = sum(
            recall_at_k(record.get("candidates", []), record["target_item_idx"], k)
            for record in records
        ) / len(records)
        metrics[f"ndcg@{k}"] = sum(
            ndcg_at_k(record.get("candidates", []), record["target_item_idx"], k)
            for record in records
        ) / len(records)
        metrics[f"mrr@{k}"] = sum(
            mrr_at_k(record.get("candidates", []), record["target_item_idx"], k)
            for record in records
        ) / len(records)
        metrics[f"coverage@{k}"] = coverage_at_k(records, k)
    return metrics
