"""Popularity baseline for MovieLens-1M PLUM-style splits.

This is a lightweight diagnostic baseline, not a benchmark suite. For the
held-out test protocol it ranks items by global frequency in train+val and
filters every item the user watched before the test target.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import pandas as pd


def find_root(start: Path) -> Path:
    root = start.resolve()
    while not (root / "src").exists() and root.parent != root:
        root = root.parent
    return root


ROOT = find_root(Path.cwd())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

evaluate_rankings = importlib.import_module("plum_ml1m.metrics").evaluate_rankings

DEFAULT_OUTPUT = ROOT / "data/processed/artifacts/popularity_baseline_trainval_test_full_seen"
K_VALUES = (1, 5, 10, 20, 50, 100)


def read_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(
        ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/splits/train.parquet")
    parser.add_argument("--val", type=Path, default=ROOT / "data/processed/splits/val.parquet")
    parser.add_argument("--test", type=Path, default=ROOT / "data/processed/splits/test.parquet")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = read_split(args.train)
    val = read_split(args.val)
    test = read_split(args.test)
    train_val = pd.concat([train, val], ignore_index=True)

    popularity = (
        train_val.groupby("item_idx")
        .size()
        .sort_values(ascending=False, kind="mergesort")
        .index.astype(int)
        .tolist()
    )

    seen_by_user = {
        int(user_id): set(int(x) for x in group["item_idx"].tolist())
        for user_id, group in train_val.groupby("user_id", sort=False)
    }

    records = []
    seen_filtered = 0
    for row in test.itertuples(index=False):
        seen = seen_by_user.get(int(row.user_id), set())
        candidates = []
        for item in popularity:
            if item in seen:
                seen_filtered += 1
                continue
            candidates.append(int(item))
            if len(candidates) >= args.top_n:
                break
        records.append(
            {
                "user_id": int(row.user_id),
                "target_item_idx": int(row.item_idx),
                "candidates": candidates,
            }
        )

    metrics = evaluate_rankings(records, K_VALUES)
    metrics.update(
        {
            "baseline": "global_popularity",
            "split": "test",
            "train_source": "train+val",
            "seen_filter_scope": "all prior train+val items for each test user",
            "top_n": int(args.top_n),
            "seconds": time.time() - started,
            "seen_filtered_count": int(seen_filtered),
            "avg_candidates": float(
                sum(len(r["candidates"]) for r in records) / max(len(records), 1)
            ),
        }
    )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(records).to_json(
        args.output_dir / "predictions.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
