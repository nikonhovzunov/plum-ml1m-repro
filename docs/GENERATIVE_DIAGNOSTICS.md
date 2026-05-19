# Generative Diagnostics

This document describes compact diagnostics for PLUM-style Semantic-ID
retrieval. These values are meant to explain the generation process behind
Recall/NDCG/MRR/Coverage. They are not new headline metrics.

## What Is Measured

| Field | Meaning |
|---|---|
| `invalid_sid_rate` | Fraction of generated SID sequences that cannot be mapped to a valid item SID. |
| `valid_sid_rate` | Fraction of generated SID sequences that are valid. |
| `sid_collision_rate` | Fraction of valid generated SIDs that expand to more than one item. |
| `unique_sid_ratio` | Unique valid SIDs divided by valid generated SIDs. |
| `duplicate_prediction_rate` | Duplicate valid item generations before final candidate truncation. |
| `seen_generated_rate` | Already watched items generated before seen filtering. |
| `avg_raw_valid_candidates` | Average valid item candidates before seen/dedup filtering. |
| `avg_unique_filtered_candidates` | Average final unique candidates after seen/dedup filtering. |
| `coverage@10` | Number of unique original `item_idx` values recommended in top 10. |

## Available Sources

Current Qwen beam-sweep scripts already compute several diagnostics:

| Source | Available fields |
|---|---|
| `scripts/experiments/qwen3_sft_beam_sweep.py` | `invalid_sid_rate`, `seen_generated_rate`, `raw_duplicate_valid_rate`, `avg_raw_valid`, `avg_unique_filtered_candidates`, `beam_size`, `num_return_sequences`, `coverage@K` |
| `src/plum_ml1m/eval/report.py` | `duplicate_prediction_rate`, `seen_items_filtered`, `avg_predictions_before_filtering`, `avg_predictions_after_filtering`, `coverage@10` from item-level prediction artifacts |
| `src/plum_ml1m/eval/runner.py` | CPU-safe SID decoding diagnostics for tiny fixtures and package tests |

Diagnostics that are not present in a source artifact are left as `null`.
No values should be inferred from memory or filled manually.

## Extraction Command

Use `diagnostics-report` to convert local result JSON/JSONL/CSV files into a
compact committed snapshot:

```bash
python -m plum_ml1m.cli diagnostics-report \
  --input data/processed/artifacts/.../beam_sweep_results.json \
  --output reports/snapshots/qwen3_qlora32_test.diagnostics.json \
  --run-id qwen3_qlora32_test \
  --beam-size 20 \
  --num-return-sequences 20 \
  --trie-constrained
```

If an input file contains multiple rows, select one explicitly with `--index`
or with `--beam-size` and `--num-return-sequences`. This avoids silently
reporting the wrong beam-sweep configuration.

## Compact Diagnostic Table

| Run | SID codebooks | Split | Beam | Return seqs | Invalid SID rate | Seen generated rate | Duplicate valid rate | Avg unique filtered candidates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B LoRA16 | `[512, 256, 128, 64]` | test | 20 | 20 | 0.0000 | 0.2237 | 0.0000 | 15.5262 |
| Qwen3-4B QLoRA32 | `[512, 256, 128, 64]` | test | 20 | 20 | 0.0000 | 0.2113 | 0.0000 | 15.7735 |
| Qwen3-0.6B QLoRA32 | `[512, 256, 128, 64]` | test | 20 | 20 | 0.0000 | 0.2564 | 0.0000 | 14.8728 |
| Qwen3-0.6B full-FT | `[512, 256, 128, 64]` | test | 20 | 20 | 0.0000 | 0.2322 | n/a | 9.5533 |
| Qwen3-0.6B QLoRA32, 3 levels | `[512, 256, 128]` | test | 20 | 20 | 0.0000 | 0.2675 | 0.0000 | 14.6492 |
| Qwen3-4B QLoRA32, 3 levels | `[512, 256, 128]` | test | 20 | 20 | 0.0000 | 0.2353 | 0.0000 | 15.2939 |

The values above are copied from compact snapshots generated from local
`beam_sweep_results.json` or `final_test_metrics.json` files. The raw
beam-sweep artifacts remain outside git.
