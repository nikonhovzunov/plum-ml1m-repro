# Metric Snapshot Schema

Each committed metric snapshot is a compact JSON file. It records scalar
metrics and provenance only; it must not include predictions, model weights, or
large generated artifacts.

Required top-level fields:

```json
{
  "run_id": "...",
  "method": "...",
  "method_family": "...",
  "split": "validation|test",
  "dataset": "MovieLens-1M",
  "git_commit": "...",
  "source": {
    "script_or_notebook": "...",
    "config": "...",
    "source_local_path": "...",
    "source_available_in_git": false,
    "note": "..."
  },
  "protocol": {
    "context_source": "train|train+val",
    "seen_filter_scope": "all_prior_user_history|prompt_window|none",
    "target_policy": "...",
    "candidate_universe": "...",
    "id_space": "original_item_idx",
    "k_values": [1, 5, 10, 20]
  },
  "model": {
    "base_model": "...",
    "adapter": "...",
    "sid_protocol": "...",
    "notes": "..."
  },
  "metrics": {
    "recall@1": null,
    "recall@5": null,
    "recall@10": null,
    "recall@20": null,
    "ndcg@10": null,
    "mrr@10": null,
    "coverage@10": null
  },
  "diagnostics": {},
  "artifact_policy": {
    "predictions_committed": false,
    "checkpoints_committed": false,
    "large_artifacts_committed": false
  },
  "limitations": []
}
```

Rules:

- Metrics must be numeric or `null`.
- `split` must be either `validation` or `test`.
- `source_available_in_git` refers to the original metric artifact, not to this
  compact snapshot.
- If a metric was copied from documentation rather than from a committed source
  artifact, the snapshot must say so in `source.note`.
- Large artifacts must never be marked as committed.

## Generative Diagnostic Snapshot Schema

Generative diagnostics explain the retrieval process before and after
SID-to-item mapping. They are compact summaries only; raw generation dumps
remain outside git.

```json
{
  "run_id": "...",
  "split": "validation|test",
  "decoding": {
    "trie_constrained": true,
    "beam_size": 30,
    "num_return_sequences": 30,
    "max_new_tokens": 4
  },
  "sid_diagnostics": {
    "invalid_sid_rate": null,
    "valid_sid_rate": null,
    "sid_collision_rate": null,
    "unique_sid_ratio": null
  },
  "generation_diagnostics": {
    "duplicate_prediction_rate": null,
    "seen_generated_rate": null,
    "avg_raw_valid_candidates": null,
    "avg_unique_filtered_candidates": null,
    "avg_predictions_before_filtering": null,
    "avg_predictions_after_filtering": null
  },
  "coverage": {
    "coverage@10": null,
    "coverage@10_percent": null
  },
  "artifact_policy": {
    "large_artifacts_committed": false,
    "prediction_dumps_committed": false
  }
}
```

Rules:

- Use `null` for unavailable diagnostics.
- `invalid_sid_rate` is measured over generated SID sequences.
- `seen_generated_rate` follows the source artifact definition; in the Qwen
  beam-sweep scripts it is the number of seen generated items divided by the
  number of generated sequences.
- `duplicate_prediction_rate` may be sourced from `raw_duplicate_valid_rate`
  when reading Qwen beam-sweep outputs.
- `coverage@10_percent` is only filled when the item universe size is known.
