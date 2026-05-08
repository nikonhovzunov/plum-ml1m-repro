from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plum_ml1m.decoding import TrieConstrainedDecoder
from plum_ml1m.metrics import evaluate_rankings


@dataclass(frozen=True)
class EvaluationCase:
    user_id: int
    target_item_idx: int
    generated_token_sequences: list[list[int]]
    seen_item_idx: set[int]


def run_sid_evaluation(
    cases: list[EvaluationCase],
    decoder: TrieConstrainedDecoder,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    top_k: int | None = None,
    collision_policy: str = "expand",
) -> dict[str, Any]:
    """Evaluate generated SID-token sequences after canonical decoding.

    The target id space is original `item_idx` after SID-to-item mapping. Trie
    validity is enforced before this function by the generation constraint; this
    CPU-safe runner still treats unknown token sequences as invalid diagnostics.
    """

    top_k = top_k or max(k_values)
    records: list[dict[str, Any]] = []
    invalid_sequences = 0
    duplicate_predictions = 0
    seen_items_filtered = 0

    for case in cases:
        raw_items: list[int] = []
        for sequence in case.generated_token_sequences:
            sid = decoder.decode_token_sequence(sequence)
            if sid is None:
                invalid_sequences += 1
                continue
            raw_items.extend(decoder.sid_mapping.resolve_sid(sid, policy=collision_policy))

        seen = {int(item) for item in case.seen_item_idx}
        used: set[int] = set()
        candidates: list[int] = []
        for item in raw_items:
            item = int(item)
            if item in seen:
                seen_items_filtered += 1
                continue
            if item in used:
                duplicate_predictions += 1
                continue
            used.add(item)
            candidates.append(item)
            if len(candidates) >= top_k:
                break

        records.append(
            {
                "user_id": int(case.user_id),
                "target_item_idx": int(case.target_item_idx),
                "candidates": candidates,
            }
        )

    return {
        "metrics": evaluate_rankings(records, k_values=k_values),
        "diagnostics": {
            "invalid_sid_sequences": invalid_sequences,
            "duplicate_predictions": duplicate_predictions,
            "seen_items_filtered": seen_items_filtered,
        },
        "records": records,
    }
