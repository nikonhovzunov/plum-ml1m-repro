from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from plum_ml1m.decoding import TrieConstrainedDecoder
from plum_ml1m.eval.runner import EvaluationCase
from plum_ml1m.paths import find_project_root, resolve_project_path


class GenerativeDiagnosticsError(ValueError):
    """Raised when diagnostic artifacts cannot be interpreted safely."""


@dataclass(frozen=True)
class DiagnosticSelection:
    index: int | None = None
    beam_size: int | None = None
    num_return_sequences: int | None = None


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return [dict(item) for item in payload["results"]]
        if isinstance(payload, dict):
            return [payload]
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    raise GenerativeDiagnosticsError(f"Unsupported diagnostic input format: {path}")


def _normalise_split(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return "validation" if text == "val" else text


def _get(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and not pd.isna(record[key]):
            return record[key]
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics and metrics[key] is not None:
                return metrics[key]
    diagnostics = record.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key in keys:
            if key in diagnostics and diagnostics[key] is not None:
                return diagnostics[key]
    decoding = record.get("decoding")
    if isinstance(decoding, dict):
        for key in keys:
            if key in decoding and decoding[key] is not None:
                return decoding[key]
    return None


def _number_or_none(value: Any) -> float | int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value.is_integer():
        return int(value)
    return value


def _int_equals(value: Any, expected: int) -> bool:
    number = _number_or_none(value)
    return number is not None and int(number) == int(expected)


def _bool_or_none(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _select_record(
    records: list[dict[str, Any]],
    selection: DiagnosticSelection,
) -> dict[str, Any]:
    if not records:
        raise GenerativeDiagnosticsError("diagnostic input contains no records")
    if selection.index is not None:
        try:
            return records[int(selection.index)]
        except IndexError as exc:
            raise GenerativeDiagnosticsError(f"record index {selection.index} is out of range") from exc

    filtered = records
    if selection.beam_size is not None:
        filtered = [
            record
            for record in filtered
            if _int_equals(_get(record, "beam_size"), selection.beam_size)
        ]
    if selection.num_return_sequences is not None:
        filtered = [
            record
            for record in filtered
            if _int_equals(_get(record, "num_return_sequences"), selection.num_return_sequences)
        ]

    if len(filtered) != 1:
        raise GenerativeDiagnosticsError(
            "diagnostic input has multiple records; pass --index or "
            "--beam-size/--num-return-sequences"
        )
    return filtered[0]


def _coverage_percent(coverage_at_10: Any, candidate_universe_size: int | None) -> float | None:
    if candidate_universe_size is None or candidate_universe_size <= 0:
        return None
    coverage = _number_or_none(coverage_at_10)
    if coverage is None:
        return None
    return float(coverage) / float(candidate_universe_size)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(find_project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def normalise_generative_diagnostics(
    record: dict[str, Any],
    *,
    run_id: str | None = None,
    split: str | None = None,
    trie_constrained: bool | None = None,
    candidate_universe_size: int | None = None,
) -> dict[str, Any]:
    invalid_sid_rate = _number_or_none(_get(record, "invalid_sid_rate"))
    valid_sid_rate = _number_or_none(_get(record, "valid_sid_rate"))
    if valid_sid_rate is None and invalid_sid_rate is not None:
        valid_sid_rate = 1.0 - float(invalid_sid_rate)

    coverage_at_10 = _number_or_none(_get(record, "coverage@10"))
    coverage_at_10_percent = _number_or_none(_get(record, "coverage@10_percent"))
    if coverage_at_10_percent is None:
        coverage_at_10_percent = _coverage_percent(coverage_at_10, candidate_universe_size)

    return {
        "run_id": run_id or str(_get(record, "run_id", "run_name", "name") or "unknown"),
        "split": _normalise_split(split or _get(record, "split")),
        "decoding": {
            "trie_constrained": (
                trie_constrained
                if trie_constrained is not None
                else _bool_or_none(_get(record, "trie_constrained"))
            ),
            "beam_size": _number_or_none(_get(record, "beam_size")),
            "num_return_sequences": _number_or_none(_get(record, "num_return_sequences")),
            "max_new_tokens": _number_or_none(_get(record, "max_new_tokens")),
        },
        "sid_diagnostics": {
            "invalid_sid_rate": invalid_sid_rate,
            "valid_sid_rate": valid_sid_rate,
            "sid_collision_rate": _number_or_none(_get(record, "sid_collision_rate")),
            "unique_sid_ratio": _number_or_none(_get(record, "unique_sid_ratio")),
        },
        "generation_diagnostics": {
            "duplicate_prediction_rate": _number_or_none(
                _get(record, "duplicate_prediction_rate", "raw_duplicate_valid_rate")
            ),
            "seen_generated_rate": _number_or_none(_get(record, "seen_generated_rate")),
            "avg_raw_valid_candidates": _number_or_none(
                _get(record, "avg_raw_valid_candidates", "avg_raw_valid")
            ),
            "avg_unique_filtered_candidates": _number_or_none(
                _get(record, "avg_unique_filtered_candidates")
            ),
            "avg_predictions_before_filtering": _number_or_none(
                _get(record, "average_predictions_before_filtering")
            ),
            "avg_predictions_after_filtering": _number_or_none(
                _get(record, "average_predictions_after_filtering")
            ),
        },
        "coverage": {
            "coverage@10": coverage_at_10,
            "coverage@10_percent": coverage_at_10_percent,
        },
        "source": {
            "available_fields": sorted(record.keys()),
        },
        "artifact_policy": {
            "large_artifacts_committed": False,
            "prediction_dumps_committed": False,
        },
    }


def build_generative_diagnostics_report(
    *,
    input_path: str | Path,
    output_path: str | Path,
    run_id: str | None = None,
    split: str | None = None,
    trie_constrained: bool | None = None,
    candidate_universe_size: int | None = None,
    selection: DiagnosticSelection | None = None,
) -> dict[str, Any]:
    input_path = resolve_project_path(input_path)
    selection = selection or DiagnosticSelection()
    record = _select_record(_read_records(input_path), selection)
    report = normalise_generative_diagnostics(
        record,
        run_id=run_id,
        split=split,
        trie_constrained=trie_constrained,
        candidate_universe_size=candidate_universe_size,
    )
    report["source"]["input_path"] = _display_path(input_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def compute_generative_diagnostics_from_cases(
    cases: Iterable[EvaluationCase],
    decoder: TrieConstrainedDecoder,
    *,
    run_id: str = "tiny_generative_diagnostics",
    split: str = "validation",
    top_k: int = 10,
    beam_size: int | None = None,
    num_return_sequences: int | None = None,
    max_new_tokens: int | None = None,
    collision_policy: str = "expand",
    candidate_universe_size: int | None = None,
) -> dict[str, Any]:
    cases = list(cases)
    total_sequences = 0
    invalid_sid_sequences = 0
    valid_sid_sequences = 0
    collision_sid_sequences = 0
    sid_outputs: list[tuple[int, ...]] = []
    raw_valid_items_total = 0
    raw_duplicate_valid = 0
    seen_generated = 0
    unique_filtered_total = 0
    insufficient_candidates = 0
    coverage_items: set[int] = set()

    for case in cases:
        raw_items: list[int] = []
        for sequence in case.generated_token_sequences:
            total_sequences += 1
            sid = decoder.decode_token_sequence(sequence)
            if sid is None or not decoder.sid_mapping.has_sid(sid):
                invalid_sid_sequences += 1
                continue
            valid_sid_sequences += 1
            sid_outputs.append(sid)
            resolved_items = decoder.sid_mapping.resolve_sid(sid, policy=collision_policy)
            if len(resolved_items) > 1:
                collision_sid_sequences += 1
            raw_items.extend(int(item) for item in resolved_items)

        raw_valid_items_total += len(raw_items)
        raw_duplicate_valid += max(0, len(raw_items) - len(set(raw_items)))

        seen = {int(item) for item in case.seen_item_idx}
        used: set[int] = set()
        candidates: list[int] = []
        for item in raw_items:
            item = int(item)
            if item in seen:
                seen_generated += 1
                continue
            if item in used:
                continue
            used.add(item)
            candidates.append(item)
            if len(candidates) >= top_k:
                break

        unique_filtered_total += len(candidates)
        coverage_items.update(candidates[:10])
        if len(candidates) < top_k:
            insufficient_candidates += 1

    n_cases = max(len(cases), 1)
    sequence_denom = max(total_sequences, 1)
    valid_sid_denom = max(valid_sid_sequences, 1)
    raw_item_denom = max(raw_valid_items_total, 1)
    coverage_at_10 = len(coverage_items)

    return {
        "run_id": run_id,
        "split": split,
        "decoding": {
            "trie_constrained": True,
            "beam_size": beam_size,
            "num_return_sequences": num_return_sequences,
            "max_new_tokens": max_new_tokens,
        },
        "sid_diagnostics": {
            "invalid_sid_rate": invalid_sid_sequences / sequence_denom,
            "valid_sid_rate": valid_sid_sequences / sequence_denom,
            "sid_collision_rate": collision_sid_sequences / valid_sid_denom,
            "unique_sid_ratio": len(set(sid_outputs)) / valid_sid_denom,
        },
        "generation_diagnostics": {
            "duplicate_prediction_rate": raw_duplicate_valid / raw_item_denom,
            "seen_generated_rate": seen_generated / sequence_denom,
            "avg_raw_valid_candidates": raw_valid_items_total / n_cases,
            "avg_unique_filtered_candidates": unique_filtered_total / n_cases,
            "avg_predictions_before_filtering": raw_valid_items_total / n_cases,
            "avg_predictions_after_filtering": unique_filtered_total / n_cases,
            "insufficient_candidate_users": insufficient_candidates,
        },
        "coverage": {
            "coverage@10": coverage_at_10,
            "coverage@10_percent": _coverage_percent(coverage_at_10, candidate_universe_size),
        },
        "artifact_policy": {
            "large_artifacts_committed": False,
            "prediction_dumps_committed": False,
        },
    }
