from __future__ import annotations

import json
from ast import literal_eval
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from plum_ml1m.config import load_yaml, validate_config
from plum_ml1m.metrics import evaluate_rankings, validate_k_values
from plum_ml1m.paths import resolve_project_path


class EvaluationReportError(ValueError):
    """Raised when prediction artifacts cannot be evaluated safely."""


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise EvaluationReportError(f"Unsupported table format for {path}")


def _parse_list_like(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list | tuple):
        return [int(x) for x in value]
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = [part.strip() for part in text.split("|") if part.strip()]
        if isinstance(parsed, int | float | str):
            parsed = [parsed]
        return [int(x) for x in parsed]
    return [int(value)]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "user_id": "user_idx",
        "target": "target_item_idx",
        "item_idx": "predicted_item_idx",
        "prediction": "predicted_item_idx",
        "candidate_item_idx": "predicted_item_idx",
        "candidates": "predicted_item_indices",
        "predictions": "predicted_item_indices",
    }
    rename = {column: aliases[column] for column in df.columns if column in aliases}
    return df.rename(columns=rename)


def _build_prediction_groups(df: pd.DataFrame) -> tuple[dict[int, list[int]], dict[int, int]]:
    df = _normalise_columns(df)
    if "user_idx" not in df.columns:
        raise EvaluationReportError("predictions must contain user_idx or user_id")

    prediction_groups: dict[int, list[int]] = {}
    targets: dict[int, int] = {}

    if "predicted_item_indices" in df.columns:
        required = {"user_idx", "predicted_item_indices"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise EvaluationReportError(f"predictions missing required columns: {missing}")
        for row in df.to_dict("records"):
            user = int(row["user_idx"])
            prediction_groups[user] = _parse_list_like(row["predicted_item_indices"])
            if "target_item_idx" in df.columns and not pd.isna(row.get("target_item_idx")):
                targets[user] = int(row["target_item_idx"])
        return prediction_groups, targets

    required = {"user_idx", "predicted_item_idx"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise EvaluationReportError(
            "predictions must contain either predicted_item_indices or "
            f"row-wise predicted_item_idx columns; missing {missing}"
        )

    if "rank" in df.columns:
        df = df.sort_values(["user_idx", "rank"], kind="stable")

    grouped: dict[int, list[int]] = defaultdict(list)
    target_values: dict[int, set[int]] = defaultdict(set)
    for row in df.to_dict("records"):
        user = int(row["user_idx"])
        grouped[user].append(int(row["predicted_item_idx"]))
        if "target_item_idx" in df.columns and not pd.isna(row.get("target_item_idx")):
            target_values[user].add(int(row["target_item_idx"]))

    for user, values in target_values.items():
        if len(values) > 1:
            raise EvaluationReportError(f"user_idx={user} has multiple target_item_idx values")
        targets[user] = next(iter(values))

    return dict(grouped), targets


def _load_targets(path: str | Path | None) -> dict[int, int]:
    if path is None:
        return {}
    df = _normalise_columns(_read_table(path))
    required = {"user_idx", "target_item_idx"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise EvaluationReportError(f"targets missing required columns: {missing}")
    targets: dict[int, int] = {}
    for row in df.to_dict("records"):
        user = int(row["user_idx"])
        target = int(row["target_item_idx"])
        if user in targets and targets[user] != target:
            raise EvaluationReportError(f"targets contain multiple values for user_idx={user}")
        targets[user] = target
    return targets


def _load_seen_history(path: str | Path | None) -> dict[int, set[int]]:
    if path is None:
        return {}
    df = _read_table(path)
    if "user_id" in df.columns and "user_idx" not in df.columns:
        df = df.rename(columns={"user_id": "user_idx"})
    if "user_idx" not in df.columns:
        raise EvaluationReportError("seen history must contain user_idx or user_id")

    if "seen_item_indices" in df.columns:
        return {
            int(row["user_idx"]): set(_parse_list_like(row["seen_item_indices"]))
            for row in df.to_dict("records")
        }

    if "seen_item_idx" in df.columns:
        seen: dict[int, set[int]] = defaultdict(set)
        for row in df.to_dict("records"):
            if not pd.isna(row["seen_item_idx"]):
                seen[int(row["user_idx"])].add(int(row["seen_item_idx"]))
        return dict(seen)

    if "item_idx" in df.columns:
        seen = defaultdict(set)
        for row in df.to_dict("records"):
            if not pd.isna(row["item_idx"]):
                seen[int(row["user_idx"])].add(int(row["item_idx"]))
        return dict(seen)

    raise EvaluationReportError("seen history must contain seen_item_indices, seen_item_idx, or item_idx")


def _filter_candidates(
    raw_predictions: Iterable[int],
    *,
    seen_items: set[int],
    filter_seen: bool,
) -> tuple[list[int], int, int]:
    used: set[int] = set()
    candidates: list[int] = []
    duplicate_count = 0
    seen_filtered_count = 0

    for raw_item in raw_predictions:
        item = int(raw_item)
        if item in used:
            duplicate_count += 1
            continue
        used.add(item)
        if filter_seen and item in seen_items:
            seen_filtered_count += 1
            continue
        candidates.append(item)

    return candidates, duplicate_count, seen_filtered_count


def _split_for_report(config_split: str) -> str:
    return "validation" if config_split == "val" else str(config_split)


def build_evaluation_report(
    *,
    config_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path | None = None,
    targets_path: str | Path | None = None,
    seen_history_path: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = validate_config(config_path)
    raw_config = load_yaml(config_path)
    evaluation = raw_config.get("evaluation", {})
    paths = raw_config.get("paths", {})

    predictions_path = resolve_project_path(predictions_path)
    targets_path = resolve_project_path(targets_path) if targets_path else None
    seen_history_path = resolve_project_path(seen_history_path) if seen_history_path else None

    prediction_groups, embedded_targets = _build_prediction_groups(_read_table(predictions_path))
    explicit_targets = _load_targets(targets_path)
    targets = {**embedded_targets, **explicit_targets}
    seen_history = _load_seen_history(seen_history_path)

    if not targets:
        raise EvaluationReportError(
            "No targets found. Provide target_item_idx in predictions or pass --targets."
        )

    k_values = validate_k_values(tuple(evaluation.get("k_values", [1, 5, 10, 20])))
    filter_seen = bool(evaluation.get("filter_seen", False))
    split_name = str(evaluation.get("split_name", "unknown"))
    report_split = _split_for_report(split_name)

    records: list[dict[str, Any]] = []
    skipped_users = 0
    duplicate_predictions = 0
    seen_items_filtered = 0
    raw_prediction_total = 0
    filtered_prediction_total = 0

    users = sorted(targets)
    for user in users:
        raw_predictions = prediction_groups.get(user, [])
        raw_prediction_total += len(raw_predictions)
        candidates, duplicates, seen_filtered = _filter_candidates(
            raw_predictions,
            seen_items=seen_history.get(user, set()),
            filter_seen=filter_seen,
        )
        duplicate_predictions += duplicates
        seen_items_filtered += seen_filtered
        filtered_prediction_total += len(candidates)
        records.append(
            {
                "user_id": int(user),
                "target_item_idx": int(targets[user]),
                "candidates": candidates,
            }
        )

    for user in prediction_groups:
        if user not in targets:
            skipped_users += 1

    metrics = evaluate_rankings(records, k_values=k_values)
    metrics_block = {
        "recall@1": metrics.get("recall@1"),
        "recall@5": metrics.get("recall@5"),
        "recall@10": metrics.get("recall@10"),
        "recall@20": metrics.get("recall@20"),
        "ndcg@10": metrics.get("ndcg@10"),
        "mrr@10": metrics.get("mrr@10"),
        "coverage@10": metrics.get("coverage@10"),
    }

    num_users = len(users)
    evaluated_users = len(records)
    generated_total = max(raw_prediction_total, 1)
    report = {
        "run_id": run_id or config.get("name", Path(predictions_path).stem),
        "split": report_split,
        "dataset": "MovieLens-1M",
        "config": str(config_path.as_posix()),
        "predictions_path": str(predictions_path.as_posix()),
        "protocol": {
            "id_space": "original_item_idx",
            "seen_filter_scope": evaluation.get("seen_filter_scope", "none"),
            "candidate_universe": evaluation.get(
                "candidate_universe",
                "MovieLens-1M item_idx universe",
            ),
            "k_values": list(k_values),
        },
        "metrics": metrics_block,
        "diagnostics": {
            "num_users": num_users,
            "evaluated_users": evaluated_users,
            "skipped_users": skipped_users,
            "average_predictions_before_filtering": raw_prediction_total / max(num_users, 1),
            "average_predictions_after_filtering": filtered_prediction_total / max(num_users, 1),
            "duplicate_predictions": duplicate_predictions,
            "duplicate_prediction_rate": duplicate_predictions / generated_total,
            "seen_items_filtered": seen_items_filtered,
            "seen_filtered_rate": seen_items_filtered / generated_total,
            "id_space": "original_item_idx",
            "split": report_split,
            "seen_filtering_policy": evaluation.get("seen_filter_scope", "none"),
            "filter_seen": filter_seen,
            "source_split_path": paths.get("split"),
        },
        "artifact_policy": {
            "predictions_committed": False,
            "checkpoints_committed": False,
            "large_artifacts_committed": False,
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report
