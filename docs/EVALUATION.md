# Evaluation Protocol

This project evaluates generative recommendations in the original MovieLens
`item_idx` space after decoding generated Semantic IDs.

## Target

Each evaluation example has one next-item target:

- validation: context is the user's chronological train history, target is the
  validation item;
- test: context is train plus validation history, target is the test item.

The test split is not used for model selection. Validation-selected settings are
frozen before a final test run.

## Decoding

The model generates Semantic ID tokens. Active SID-v2 uses four levels:

```text
<sid_0_*>, <sid_1_*>, <sid_2_*>, <sid_3_*>
```

The active codebook sizes are:

```text
[512, 256, 128, 64]
```

Generation uses trie-constrained decoding, so only SID token sequences present
in the item universe can be produced. If unconstrained diagnostic generation is
used, invalid SID sequences must be counted separately and must not be silently
converted into recommendations.

## SID-to-Item Mapping

Metrics are computed on original `item_idx`, not on SID strings. Generated SIDs
are mapped back to item IDs. If multiple items share one SID, the active
collision policy is `expand`: all items in that collision bucket are considered
in deterministic order, then seen and duplicate items are filtered.

## Seen-Item Filtering

Already-seen items are removed before top-K truncation. The current main
validation/test protocol filters all items watched before the target timestamp.
Older prompt-window-only rows are diagnostic only and must be explicitly marked
as such.

## Duplicate Handling

Duplicate predictions are deduplicated while preserving model order. This means
`[7, 7, 9]` is evaluated as `[7, 9]`.

## Metrics

For one target item per user, Recall@K is equivalent to HitRate@K:

```text
Recall@K(u) = 1[target_u in topK_u]
```

NDCG@K:

```text
NDCG@K(u) = 1 / log2(rank(target_u) + 1), if target appears in top K, else 0
```

MRR@K:

```text
MRR@K(u) = 1 / rank(target_u), if target appears in top K, else 0
```

Coverage@K is the number of unique recommended `item_idx` values across all
users' top-K lists. It is a count, not a percentage.

## Configs

Validation and test configs are intentionally separate:

- `configs/evaluation_val.yaml`
- `configs/evaluation_test.yaml`

`make eval` does not choose a split implicitly. Use `make eval-val` or
`make eval-test`.

## Package-Native Metric Reports

The package includes a CPU-safe command for recomputing item-level metrics from
existing prediction artifacts:

```bash
python -m plum_ml1m.cli eval-report \
  --config configs/evaluation_test.yaml \
  --predictions data/processed/artifacts/evaluation/predictions.parquet \
  --targets data/processed/artifacts/evaluation/targets.parquet \
  --seen-history data/processed/artifacts/evaluation/seen_history.parquet \
  --output reports/snapshots/example_test.metrics.json
```

This command does not train a model, download checkpoints, or run generation.
It only loads ranked item predictions, targets, optional seen histories, and
then computes metrics through the canonical `plum_ml1m.metrics` module.

Supported prediction schemas:

```text
user_idx,target_item_idx,rank,predicted_item_idx
```

or:

```text
user_idx,target_item_idx,predicted_item_indices
```

`predicted_item_indices` may be a JSON/list-like field. The legacy
`candidates` column name is accepted as an alias because several experiment
scripts already export compact item-level predictions with that name. Targets
may also be provided separately with `--targets`. Seen histories may be
supplied as `seen_item_indices`, row-wise `seen_item_idx`, or row-wise
`item_idx`.
Raw SID-token generations are not scored directly by this command; they should
first be decoded to original `item_idx` candidates through the canonical trie
decoder and then exported with one of the item-level schemas above.

The evaluator deduplicates candidates per user, filters seen items before
top-K truncation when the config enables seen filtering, and writes a compact
JSON report with metrics plus diagnostics. Heavy prediction dumps are not
committed; small scalar metric reports may be committed under
`reports/snapshots/`.
