"""Behavior-derived positive pair mining for SID contrastive training."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _rating_weight(rating: float, min_rating: float) -> float:
    if rating < min_rating:
        return 0.0
    return float(rating - min_rating + 1.0)


def build_weighted_cooccurrence_pairs(
    train: pd.DataFrame,
    output_path: str | Path | None = None,
    *,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
    rating_col: str = "rating",
    pos_col: str = "pos",
    timestamp_col: str = "timestamp",
    window_size: int = 20,
    distance_decay: float = 0.85,
    min_rating: float = 4.0,
    top_k_per_item: int = 32,
    min_pair_weight: float = 1e-6,
    use_ppmi: bool = True,
) -> pd.DataFrame:
    """Build strong co-occurrence positives beyond adjacent item pairs.

    This approximates PLUM's co-occurrence alignment signal without injecting a
    mutable collaborative-filtering embedding into the item representation. A
    pair receives higher weight when two positively-rated movies appear in the
    same user's local history window, especially when they are close in time.

    The final score is optionally popularity-debiased using positive PMI and
    then capped to top-K positives per anchor item.
    """

    required = {user_col, item_col, rating_col, pos_col, timestamp_col}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"train is missing columns: {sorted(missing)}")
    if not (0.0 < distance_decay <= 1.0):
        raise ValueError("distance_decay must be in (0, 1].")
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")

    counts: dict[tuple[int, int], float] = defaultdict(float)
    outgoing_mass: dict[int, float] = defaultdict(float)
    n_interactions_used = 0

    sort_cols = [user_col, pos_col, timestamp_col]
    for _, group in train.sort_values(sort_cols).groupby(user_col, sort=False):
        items = group[item_col].to_numpy(np.int64)
        ratings = group[rating_col].to_numpy(np.float32)
        n = len(items)
        if n <= 1:
            continue
        for left in range(n):
            item_i = int(items[left])
            wi = _rating_weight(float(ratings[left]), min_rating)
            if wi <= 0.0:
                continue
            right_stop = min(n, left + window_size + 1)
            for right in range(left + 1, right_stop):
                item_j = int(items[right])
                if item_i == item_j:
                    continue
                wj = _rating_weight(float(ratings[right]), min_rating)
                if wj <= 0.0:
                    continue
                distance = right - left
                pair_weight = (distance_decay ** (distance - 1)) * float(np.sqrt(wi * wj))
                if pair_weight <= min_pair_weight:
                    continue
                counts[(item_i, item_j)] += pair_weight
                counts[(item_j, item_i)] += pair_weight
                outgoing_mass[item_i] += pair_weight
                outgoing_mass[item_j] += pair_weight
                n_interactions_used += 2

    if not counts:
        raise ValueError("No co-occurrence pairs were produced. Check rating/window settings.")

    total_mass = float(sum(counts.values()))
    rows: list[tuple[int, int, float, float, float]] = []
    eps = 1e-12
    for (item_i, item_j), weight in counts.items():
        if use_ppmi:
            expected = outgoing_mass[item_i] * outgoing_mass[item_j]
            pmi = float(np.log((weight * total_mass + eps) / (expected + eps)))
            ppmi = max(0.0, pmi)
            score = ppmi * float(np.log1p(weight))
        else:
            pmi = 0.0
            ppmi = 0.0
            score = float(weight)
        if score <= 0.0:
            continue
        rows.append((item_i, item_j, float(weight), pmi, score))

    pairs = pd.DataFrame(rows, columns=["item_idx", "item_pos", "co_weight", "pmi", "score"])
    if pairs.empty:
        raise ValueError("PPMI filtering removed all pairs. Try use_ppmi=False or lower min_rating.")

    pairs = (
        pairs.sort_values(["item_idx", "score", "co_weight"], ascending=[True, False, False])
        .groupby("item_idx", as_index=False, sort=False)
        .head(top_k_per_item)
        .reset_index(drop=True)
    )
    pairs["rank_for_item"] = pairs.groupby("item_idx").cumcount() + 1
    pairs.attrs["n_interactions_used"] = n_interactions_used
    pairs.attrs["total_mass"] = total_mass

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_parquet(output_path, index=False)

    return pairs
