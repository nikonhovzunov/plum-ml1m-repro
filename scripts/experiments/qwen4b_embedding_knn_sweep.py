"""Qwen3-Embedding-4B unweighted concat meta+description KNN baseline for SID-v2.

This script is intentionally non-generative: it uses the same Qwen3-Embedding-4B
item embeddings as the main SID-v2 pipeline, but ranks original MovieLens
item_idx values directly in embedding space. Each item vector is a plain block
concatenation of:

- metadata embedding: title + release year + genres;
- description embedding: movie overview.
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
import torch
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

DEFAULT_EMBEDDINGS = ROOT / "data/processed/item_features/qwen4b_audited_v1_meta_desc_embeddings.npz"
DEFAULT_TRAIN = ROOT / "data/processed/splits/train.parquet"
DEFAULT_VAL = ROOT / "data/processed/splits/val.parquet"
DEFAULT_TEST = ROOT / "data/processed/splits/test.parquet"
DEFAULT_OUTPUT = ROOT / "reports/embedding_knn_qwen4b_v2/concat_meta_description_w16_all_seen"


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def topk_from_scores(scores: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []
    finite = np.isfinite(scores)
    n_valid = int(finite.sum())
    if n_valid == 0:
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


def rating_weights(ratings: np.ndarray, mode: str) -> np.ndarray:
    ratings = np.asarray(ratings, dtype=np.float32)
    if mode == "none":
        return np.ones_like(ratings, dtype=np.float32)
    if mode == "linear":
        return np.clip(ratings / 5.0, 0.0, 1.0).astype(np.float32)
    if mode == "centered_positive":
        return np.clip((ratings - 2.5) / 2.5, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unknown rating_mode: {mode}")


def recency_weights(length: int, alpha: float) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    offsets = np.arange(length - 1, -1, -1, dtype=np.float32)
    return np.power(float(alpha), offsets).astype(np.float32)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(
        ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
    )


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


def make_embedding_variant(
    meta: np.ndarray,
    desc: np.ndarray,
) -> np.ndarray:
    meta = l2_normalize(meta)
    desc = l2_normalize(desc)
    return l2_normalize(np.concatenate([meta, desc], axis=1))


def compute_similarity(embeddings: np.ndarray, device: str) -> np.ndarray:
    if device == "cuda" and torch.cuda.is_available():
        with torch.inference_mode():
            tensor = torch.tensor(embeddings, dtype=torch.float32, device="cuda")
            sim = tensor @ tensor.T
            return sim.cpu().numpy().astype(np.float32)
    return (embeddings @ embeddings.T).astype(np.float32)


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
    k_values: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    max_k = max(k_values)
    records = []
    candidate_lengths = []

    for example in examples:
        history = example["history_item_idx"]
        weights = recency_weights(len(history), alpha)
        weights *= rating_weights(np.asarray(example["history_rating"], dtype=np.float32), rating_mode)
        if not np.any(weights > 0):
            weights = np.ones(len(history), dtype=np.float32)

        scores = score_candidates(sim, history, weights, aggregator)
        scores[np.asarray(example["all_seen_item_idx"], dtype=np.int64)] = -np.inf
        candidates = topk_from_scores(scores, max_k)
        target = int(example["target_item_idx"])
        candidate_lengths.append(len(candidates))

        records.append(
            {
                "user_id": int(example["user_id"]),
                "target_item_idx": target,
                "candidates": candidates,
            }
        )

    metrics: dict[str, Any] = evaluate_rankings(records, k_values=tuple(k_values))
    metrics.update(
        {
            "alpha": float(alpha),
            "rating_mode": rating_mode,
            "aggregator": aggregator,
            "avg_candidates": float(np.mean(candidate_lengths)) if candidate_lengths else 0.0,
        }
    )
    return metrics, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--embedding-path", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-window", type=int, default=16)
    parser.add_argument("--min-history-len", type=int, default=16)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--k-values", default="1,5,10,20,50,100")
    parser.add_argument("--alphas", default="0.8")
    parser.add_argument("--rating-modes", default="none")
    parser.add_argument("--aggregators", default="sum")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    k_values = parse_csv_ints(args.k_values)
    alphas = parse_csv_floats(args.alphas)
    rating_modes = parse_csv_strings(args.rating_modes)
    aggregators = parse_csv_strings(args.aggregators)

    train = load_split(args.train)
    val = load_split(args.val)
    test = load_split(args.test)
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

    bundle = np.load(args.embedding_path)
    item_idx = bundle["item_idx"].astype(np.int64)
    if not np.array_equal(item_idx, np.arange(len(item_idx))):
        raise ValueError("Embedding item_idx must be contiguous and sorted from 0")
    meta = bundle["meta"].astype(np.float32)
    desc = bundle["description"].astype(np.float32)

    run_config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "embedding_path": str(args.embedding_path.resolve()),
        "output_dir": str(output_dir),
        "history_window": int(args.history_window),
        "min_history_len": int(args.min_history_len),
        "seen_filter": "all_prior_history",
        "user_count": int(len(examples)),
        "item_count": int(len(item_idx)),
        "embedding_dim": int(meta.shape[1]),
        "concat_embedding_dim": int(meta.shape[1] + desc.shape[1]),
        "representation": "concat",
        "k_values": k_values,
        "alphas": alphas,
        "rating_modes": rating_modes,
        "aggregators": aggregators,
        "device": device,
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

    total_variants = len(alphas) * len(rating_modes) * len(aggregators)
    progress = tqdm(total=total_variants, desc="embedding-kNN sweep")

    embeddings = make_embedding_variant(meta, desc)
    sim = compute_similarity(embeddings, device=device)
    np.fill_diagonal(sim, 1.0)

    for alpha in alphas:
        for rating_mode in rating_modes:
            for aggregator in aggregators:
                metrics, records = evaluate_variant(
                    sim=sim,
                    examples=examples,
                    alpha=alpha,
                    rating_mode=rating_mode,
                    aggregator=aggregator,
                    k_values=k_values,
                )
                metrics.update(
                    {
                        "split": args.split,
                        "representation": "concat",
                        "history_window": int(args.history_window),
                        "seen_filter": "all_prior_history",
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
                        "r10": f"{metrics.get('recall@10', 0.0):.4f}",
                        "best": f"{best.get('recall@10', 0.0):.4f}",
                    }
                )
                progress.update(1)
    progress.close()

    results_df = pd.DataFrame(all_results).sort_values(
        ["recall@10", "ndcg@10", "mrr@10", "coverage@10"],
        ascending=[False, False, False, False],
    )
    results_df.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if best is None or best_records is None:
        raise RuntimeError("No KNN variants were evaluated")
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

    top_rows = results_df.head(20)
    summary = [
        "# Qwen3-Embedding-4B Concat Meta+Description KNN",
        "",
        f"- split: `{args.split}`",
        f"- users evaluated: `{len(examples)}`",
        f"- history window: `{args.history_window}`",
        "- seen filtering: all prior user history",
        f"- variants: `{total_variants}`",
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
        top_rows.to_csv(index=False).strip(),
        "```",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
