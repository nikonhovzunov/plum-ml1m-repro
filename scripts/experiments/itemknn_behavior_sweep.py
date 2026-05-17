"""Behavioral ItemKNN baseline for the MovieLens-1M SID-v2 protocol.

This baseline does not use text embeddings, Semantic IDs, or a language model.
It builds item-item similarities from user behavior and evaluates original
MovieLens item_idx recommendations under the same next-item protocol used by
the generative experiments.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm


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
DEFAULT_TRAIN = ROOT / "data/processed/splits/train.parquet"
DEFAULT_VAL = ROOT / "data/processed/splits/val.parquet"
DEFAULT_TEST = ROOT / "data/processed/splits/test.parquet"
DEFAULT_OUTPUT = ROOT / "reports/itemknn_behavior_v2/validation_w16_all_seen"
K_VALUES = (1, 5, 10, 20, 50, 100)


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def read_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(
        ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
    )


def topk_from_scores(scores: np.ndarray, k: int) -> list[int]:
    finite = np.isfinite(scores)
    n_valid = int(finite.sum())
    if k <= 0 or n_valid == 0:
        return []
    k = min(k, n_valid)
    valid_idx = np.flatnonzero(finite)
    valid_scores = scores[valid_idx]
    if k >= len(valid_idx):
        order = np.argsort(-valid_scores, kind="mergesort")
    else:
        part = np.argpartition(-valid_scores, kth=k - 1)[:k]
        order = part[np.argsort(-valid_scores[part], kind="mergesort")]
    return valid_idx[order].astype(int).tolist()


def recency_weights(length: int, alpha: float) -> np.ndarray:
    offsets = np.arange(length - 1, -1, -1, dtype=np.float32)
    return np.power(float(alpha), offsets).astype(np.float32)


def rating_weights(ratings: np.ndarray, mode: str) -> np.ndarray:
    ratings = np.asarray(ratings, dtype=np.float32)
    if mode == "none":
        return np.ones_like(ratings, dtype=np.float32)
    if mode == "linear":
        return np.clip(ratings / 5.0, 0.0, 1.0).astype(np.float32)
    if mode == "centered_positive":
        return np.clip((ratings - 2.5) / 2.5, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unknown rating mode: {mode}")


def build_examples(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    split: str,
    history_window: int,
    min_history_len: int,
    max_users: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if split == "val":
        context = train
        target = val
    elif split == "test":
        context = pd.concat([train, val], ignore_index=True)
        target = test
    else:
        raise ValueError("split must be val or test")

    context_by_user = {
        int(user_id): group.sort_values(
            ["timestamp", "pos", "item_idx"], kind="mergesort"
        ).to_dict("records")
        for user_id, group in context.groupby("user_id", sort=False)
    }
    target_by_user = {
        int(row.user_id): row._asdict()
        for row in target.sort_values(
            ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
        ).itertuples(index=False)
    }

    user_ids = sorted(target_by_user)
    if max_users is not None and max_users > 0:
        rng = np.random.default_rng(seed)
        user_ids = sorted(rng.choice(user_ids, size=min(max_users, len(user_ids)), replace=False))

    examples: list[dict[str, Any]] = []
    for user_id in user_ids:
        full_history = context_by_user.get(int(user_id), [])
        visible = full_history[-int(history_window) :]
        if len(visible) < int(min_history_len):
            continue
        examples.append(
            {
                "user_id": int(user_id),
                "history_item_idx": [int(row["item_idx"]) for row in visible],
                "history_rating": [float(row["rating"]) for row in visible],
                "all_seen_item_idx": [int(row["item_idx"]) for row in full_history],
                "target_item_idx": int(target_by_user[int(user_id)]["item_idx"]),
            }
        )
    return examples


def build_user_item_matrix(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    weighting: str,
) -> sp.csr_matrix:
    rows = interactions["user_idx"].to_numpy(dtype=np.int64)
    cols = interactions["item_idx"].to_numpy(dtype=np.int64)
    if weighting == "rating":
        data = interactions["rating"].to_numpy(dtype=np.float32)
    else:
        data = np.ones(len(interactions), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)


def bm25_weight_item_user(
    item_user: sp.csr_matrix,
    k1: float = 100.0,
    b: float = 0.8,
) -> sp.csr_matrix:
    matrix = item_user.tocoo(copy=True)
    n_items, n_users = matrix.shape
    user_df = np.bincount(matrix.col, minlength=n_users).astype(np.float32)
    idf = np.log((n_items - user_df + 0.5) / (user_df + 0.5))
    idf = np.maximum(idf, 0.0).astype(np.float32)

    doc_len = np.asarray(item_user.sum(axis=1)).reshape(-1).astype(np.float32)
    avg_doc_len = float(doc_len.mean()) if doc_len.size else 0.0
    norm = k1 * (1.0 - b + b * doc_len / max(avg_doc_len, 1e-12))

    matrix.data = (
        matrix.data
        * (k1 + 1.0)
        / (matrix.data + norm[matrix.row])
        * idf[matrix.col]
    ).astype(np.float32)
    return matrix.tocsr()


def cosine_similarity_from_item_features(item_features: sp.csr_matrix) -> np.ndarray:
    norms = np.sqrt(np.asarray(item_features.multiply(item_features).sum(axis=1)).reshape(-1))
    norms = np.maximum(norms, 1e-12).astype(np.float32)
    sim = (item_features @ item_features.T).toarray().astype(np.float32)
    sim /= norms[:, None]
    sim /= norms[None, :]
    np.fill_diagonal(sim, 0.0)
    return sim


def build_similarity(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    similarity: str,
) -> np.ndarray:
    if similarity == "binary_cosine":
        user_item = build_user_item_matrix(interactions, n_users, n_items, weighting="binary")
        return cosine_similarity_from_item_features(user_item.T.tocsr())
    if similarity == "rating_cosine":
        user_item = build_user_item_matrix(interactions, n_users, n_items, weighting="rating")
        return cosine_similarity_from_item_features(user_item.T.tocsr())
    if similarity == "bm25_cosine":
        user_item = build_user_item_matrix(interactions, n_users, n_items, weighting="binary")
        item_user = user_item.T.tocsr()
        return cosine_similarity_from_item_features(bm25_weight_item_user(item_user))
    raise ValueError(f"Unknown similarity: {similarity}")


def score_candidates(
    sim: np.ndarray,
    history: list[int],
    weights: np.ndarray,
    aggregator: str,
) -> np.ndarray:
    hist = np.asarray(history, dtype=np.int64)
    local = sim[:, hist] * weights.reshape(1, -1)
    if aggregator == "sum":
        return local.sum(axis=1)
    if aggregator == "max":
        return local.max(axis=1)
    if aggregator == "top3":
        n = min(3, local.shape[1])
        part = np.partition(local, kth=local.shape[1] - n, axis=1)[:, -n:]
        return part.sum(axis=1)
    raise ValueError(f"Unknown aggregator: {aggregator}")


def evaluate_variant(
    sim: np.ndarray,
    examples: list[dict[str, Any]],
    alpha: float,
    rating_mode: str,
    aggregator: str,
    top_n: int,
) -> list[dict[str, Any]]:
    records = []
    for example in examples:
        weights = recency_weights(len(example["history_item_idx"]), alpha)
        weights *= rating_weights(np.asarray(example["history_rating"], dtype=np.float32), rating_mode)
        if not np.any(weights > 0):
            weights = np.ones(len(example["history_item_idx"]), dtype=np.float32)
        scores = score_candidates(sim, example["history_item_idx"], weights, aggregator)
        scores[np.asarray(example["all_seen_item_idx"], dtype=np.int64)] = -np.inf
        records.append(
            {
                "user_id": int(example["user_id"]),
                "target_item_idx": int(example["target_item_idx"]),
                "candidates": topk_from_scores(scores, top_n),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-window", type=int, default=16)
    parser.add_argument("--min-history-len", type=int, default=16)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarities", default="binary_cosine,rating_cosine,bm25_cosine")
    parser.add_argument("--alphas", default="1.0,0.98,0.95,0.9,0.85,0.8")
    parser.add_argument("--rating-modes", default="none,linear,centered_positive")
    parser.add_argument("--aggregators", default="sum,max,top3")
    args = parser.parse_args()

    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    similarities = parse_csv_strings(args.similarities)
    alphas = parse_csv_floats(args.alphas)
    rating_modes = parse_csv_strings(args.rating_modes)
    aggregators = parse_csv_strings(args.aggregators)

    train = read_split(args.train)
    val = read_split(args.val)
    test = read_split(args.test)
    train_val = pd.concat([train, val], ignore_index=True)
    sim_source = train if args.split == "val" else train_val
    n_users = int(max(train["user_idx"].max(), val["user_idx"].max(), test["user_idx"].max()) + 1)
    n_items = int(max(train["item_idx"].max(), val["item_idx"].max(), test["item_idx"].max()) + 1)

    examples = build_examples(
        train=train,
        val=val,
        test=test,
        split=args.split,
        history_window=args.history_window,
        min_history_len=args.min_history_len,
        max_users=None if args.max_users <= 0 else args.max_users,
        seed=args.seed,
    )

    run_config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": "behavior_itemknn",
        "split": args.split,
        "similarity_source": "train" if args.split == "val" else "train+val",
        "target_source": args.split,
        "output_dir": str(output_dir),
        "history_window": int(args.history_window),
        "min_history_len": int(args.min_history_len),
        "seen_filter": "all_prior_history",
        "user_count": int(len(examples)),
        "item_count": int(n_items),
        "top_n": int(args.top_n),
        "similarities": similarities,
        "alphas": alphas,
        "rating_modes": rating_modes,
        "aggregators": aggregators,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_records: list[dict[str, Any]] | None = None
    results_jsonl = output_dir / "results.jsonl"
    if results_jsonl.exists():
        results_jsonl.unlink()

    total = len(similarities) * len(alphas) * len(rating_modes) * len(aggregators)
    progress = tqdm(total=total, desc="behavior-itemknn sweep")
    for similarity in similarities:
        sim = build_similarity(sim_source, n_users=n_users, n_items=n_items, similarity=similarity)
        for alpha in alphas:
            for rating_mode in rating_modes:
                for aggregator in aggregators:
                    records = evaluate_variant(
                        sim=sim,
                        examples=examples,
                        alpha=alpha,
                        rating_mode=rating_mode,
                        aggregator=aggregator,
                        top_n=args.top_n,
                    )
                    metrics = evaluate_rankings(records, K_VALUES)
                    metrics.update(
                        {
                            "baseline": "behavior_itemknn",
                            "split": args.split,
                            "similarity": similarity,
                            "similarity_source": run_config["similarity_source"],
                            "alpha": float(alpha),
                            "rating_mode": rating_mode,
                            "aggregator": aggregator,
                            "history_window": int(args.history_window),
                            "seen_filter": "all_prior_history",
                            "top_n": int(args.top_n),
                            "avg_candidates": float(
                                sum(len(r["candidates"]) for r in records) / max(len(records), 1)
                            ),
                            "n": int(len(records)),
                        }
                    )
                    all_results.append(metrics)
                    with results_jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

                    key = (
                        metrics.get("recall@10", 0.0),
                        metrics.get("ndcg@10", 0.0),
                        metrics.get("mrr@10", 0.0),
                        metrics.get("coverage@10", 0),
                    )
                    best_key = (
                        best.get("recall@10", 0.0) if best else -1.0,
                        best.get("ndcg@10", 0.0) if best else -1.0,
                        best.get("mrr@10", 0.0) if best else -1.0,
                        best.get("coverage@10", 0) if best else -1,
                    )
                    if best is None or key > best_key:
                        best = dict(metrics)
                        best_records = records

                    progress.set_postfix(
                        {
                            "sim": similarity,
                            "r10": f"{metrics.get('recall@10', 0.0):.4f}",
                            "best": f"{best.get('recall@10', 0.0):.4f}",
                        }
                    )
                    progress.update(1)
        del sim
    progress.close()

    if best is None or best_records is None:
        raise RuntimeError("No ItemKNN variants were evaluated")

    results_df = pd.DataFrame(all_results).sort_values(
        ["recall@10", "ndcg@10", "mrr@10", "coverage@10"],
        ascending=[False, False, False, False],
    )
    results_df.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "user_id": rec["user_id"],
                "target_item_idx": rec["target_item_idx"],
                **{f"candidate_{i + 1}": item for i, item in enumerate(rec["candidates"])},
            }
            for rec in best_records
        ]
    ).to_parquet(output_dir / "best_predictions.parquet", index=False)

    summary = [
        "# Behavioral ItemKNN Sweep",
        "",
        f"- split: `{args.split}`",
        f"- similarity source: `{run_config['similarity_source']}`",
        f"- users evaluated: `{len(examples)}`",
        f"- history window: `{args.history_window}`",
        "- seen filtering: all prior user history",
        f"- variants: `{total}`",
        f"- seconds: `{time.time() - started:.2f}`",
        "",
        "## Best Variant",
        "",
        "```json",
        json.dumps(best, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top 20",
        "",
        "```csv",
        results_df.head(20).to_csv(index=False).strip(),
        "```",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
