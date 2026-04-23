from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(candidates: list[int], target: int, k: int) -> float:
    return float(int(int(target) in candidates[:k]))


def ndcg_at_k(candidates: list[int], target: int, k: int) -> float:
    target = int(target)
    for rank, item in enumerate(candidates[:k], start=1):
        if int(item) == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def mrr_at_k(candidates: list[int], target: int, k: int) -> float:
    target = int(target)
    for rank, item in enumerate(candidates[:k], start=1):
        if int(item) == target:
            return 1.0 / rank
    return 0.0


def evaluate_rankings(
    records: Iterable[dict],
    k_values: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    records = list(records)
    if not records:
        return {"n": 0.0}

    metrics: dict[str, float] = {"n": float(len(records))}
    recommended_items: set[int] = set()

    for k in k_values:
        metrics[f"recall@{k}"] = 0.0
        metrics[f"ndcg@{k}"] = 0.0
        metrics[f"mrr@{k}"] = 0.0

    invalid_sid_count = 0
    duplicate_count = 0
    generated_sid_count = 0

    for record in records:
        candidates = [int(item) for item in record.get("candidates", [])]
        target = int(record["target_item_idx"])
        recommended_items.update(candidates)

        generated_sid_count += int(record.get("generated_sid_count", len(candidates)))
        invalid_sid_count += int(record.get("invalid_sid_count", 0))
        duplicate_count += int(record.get("duplicate_count", 0))

        for k in k_values:
            metrics[f"recall@{k}"] += recall_at_k(candidates, target, k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(candidates, target, k)
            metrics[f"mrr@{k}"] += mrr_at_k(candidates, target, k)

    n = float(len(records))
    for key in list(metrics):
        if key != "n":
            metrics[key] /= n

    denom = max(generated_sid_count + invalid_sid_count, 1)
    metrics["invalid_sid_rate"] = invalid_sid_count / denom
    metrics["duplicate_rate"] = duplicate_count / max(generated_sid_count, 1)
    metrics["item_coverage"] = float(len(recommended_items))
    return metrics
