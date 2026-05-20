# Metric Snapshot Schema

Each committed metric snapshot is a compact JSON file. It records scalar
metrics and provenance only; it must not include predictions, model weights, or
large generated artifacts.

Common top-level shape:

```json
{
  "run_id": "...",
  "method": "...",
  "method_family": "...",
  "split": "validation|test",
  "dataset": "MovieLens-1M",
  "git_commit": "...",
  "provenance": {
    "source_run_commit": null,
    "snapshot_commit": "...",
    "protocol_commit": "...",
    "source_run_commit_note": "...",
    "source_artifact_available_in_git": false
  },
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
  "sid_protocol": {
    "name": "SID-v2",
    "n_levels": 4,
    "codebook_sizes": [512, 256, 128, 64],
    "collision_policy": "expand"
  },
  "model": {
    "base_model": "...",
    "adapter": "...",
    "sid_protocol": "...",
    "notes": "..."
  },
  "adaptation": {
    "type": "QLoRA32|LoRA16|full_ft|...",
    "cpt": "QLoRA|LoRA|full_ft|none|unknown",
    "sft": "QLoRA|LoRA|full_ft|unknown",
    "base_model": "...",
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
- Top-level `git_commit` is the compact snapshot commit, not necessarily the
  local source run commit.
- `provenance.source_run_commit` may be `null` when the exact local run commit
  was not recorded. In that case `source_run_commit_note` must explain the
  limitation explicitly.
- `provenance.snapshot_commit` identifies the commit where the compact snapshot
  values were first committed.
- `provenance.protocol_commit` identifies the repository protocol/docs revision
  used when the provenance metadata was normalized.
- `source_available_in_git` refers to the original metric artifact, not to this
  compact snapshot.
- `source_artifact_available_in_git` repeats that policy inside the provenance
  block for quick audit checks.
- `sid_protocol` is required for Qwen/SID snapshots. The active SID-v2 rows use
  four levels with `[512, 256, 128, 64]`; SID-depth ablation rows must state
  their ablation codebooks explicitly.
- `adaptation` is required for PEFT/full-FT Qwen snapshots, with separate CPT and
  SFT adaptation types. Missing source details must be written as `unknown`,
  not guessed.
- If a metric was copied from documentation rather than from a committed source
  artifact, the snapshot must say so in `source.note`.
- Large artifacts must never be marked as committed.
- Compact snapshots do not imply that checkpoints, prediction dumps, embeddings,
  adapters, or processed datasets are committed.

## Summary CSV Schema

`summary_test.csv` and `summary_validation.csv` are compact indices over the
JSON metric snapshots. They are intended for quick table rendering only; the
corresponding `*.metrics.json` files remain the canonical scalar metric records.

Required columns:

```text
run_id
method
method_family
split
sid_codebooks
recall@1
recall@5
recall@10
recall@20
ndcg@10
mrr@10
coverage@10
seen_filter_scope
context_source
notes
```

Rules:

- Each `run_id` must resolve to `reports/snapshots/{run_id}.metrics.json`.
- Numeric metric cells must match the corresponding JSON `metrics` values.
- Empty metric cells are allowed only when the JSON metric value is `null`.
- Test summaries must contain only `split=test`; validation summaries must
  contain only `split=validation`.
- Summary rows may shorten prose notes, but must not change scalar metrics.

## Generative Diagnostic Snapshot Schema

Generative diagnostics explain the retrieval process before and after
SID-to-item mapping. They are compact summaries only; raw generation dumps
remain outside git.

```json
{
  "run_id": "...",
  "split": "validation|test",
  "git_commit": "...",
  "provenance": {
    "source_run_commit": null,
    "snapshot_commit": "...",
    "protocol_commit": "...",
    "source_run_commit_note": "...",
    "source_artifact_available_in_git": false
  },
  "sid_protocol": {
    "name": "SID-v2",
    "n_levels": 4,
    "codebook_sizes": [512, 256, 128, 64],
    "collision_policy": "expand"
  },
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
