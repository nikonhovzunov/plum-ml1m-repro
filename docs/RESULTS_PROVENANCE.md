# Results Provenance

This document explains where the compact reported metrics come from. It is not
a replacement for full local experiment artifacts; it is a lightweight audit
map for external review.

## Policy

- Prediction dumps, checkpoints, LoRA adapters, embeddings, processed datasets,
  and model weights are not committed.
- Compact scalar metric snapshots are committed under `reports/snapshots/`.
- If a source metric file lives under ignored local paths such as
  `data/processed/` or `reports/<experiment>/`, the snapshot records that path
  but does not commit the source artifact.
- Metrics must not be changed unless the corresponding evaluation is rerun.

## Package-Native Re-Scoring

When local prediction artifacts are available, scalar metrics can be
recomputed without training or generation:

```bash
python -m plum_ml1m.cli eval-report \
  --config configs/evaluation_test.yaml \
  --predictions <local_predictions.parquet_or_csv> \
  --targets <local_targets.parquet_or_csv> \
  --seen-history <local_seen_history.parquet_or_csv> \
  --output reports/snapshots/<run>.metrics.json
```

The command accepts the canonical item-level prediction schema documented in
`docs/EVALUATION.md` and uses `plum_ml1m.metrics` for Recall/NDCG/MRR/Coverage.
It is intended for both baseline and Qwen runs after their prediction artifacts
already exist. It does not commit or recreate large prediction dumps.

## Common Evaluation Protocol

The main reported rows use MovieLens-1M with chronological leave-one-out
splits. Metrics are item-level: generated or ranked candidates are evaluated in
the original `item_idx` id-space. For SID-based Qwen runs, generated Semantic
IDs are constrained by the valid item trie, mapped back to original item IDs,
expanded for SID collisions, deduplicated, and filtered for already watched
items before Recall/NDCG/MRR/Coverage are computed.

The main held-out test table uses all-prior-user-history seen filtering.
Sequence-based methods use a history window of 16 items.

## Held-Out Test Rows

| Snapshot | Method | SID codebooks | Source artifact | Source in git? | Notes |
|---|---|---|---|---|---|
| `popularity_test.metrics.json` | Global Popularity | n/a | `data/processed/artifacts/popularity_baseline_trainval_test_full_seen/metrics.json` | no | Global train+val item frequency baseline. |
| `content_knn_test.metrics.json` | Content KNN with Qwen content | n/a | `reports/embedding_knn_qwen4b_v2/concat_meta_description_w16_all_seen/best.json` | no | Concatenated metadata + overview Qwen content vectors. |
| `itemknn_test.metrics.json` | Behavioral ItemKNN | n/a | `reports/itemknn_behavior_v2/test_w16_all_seen_best_val_selected/best.json` | no | BM25-cosine interaction KNN. |
| `bert4rec_style_multimodal_test.metrics.json` | BERT4Rec-style + Qwen content | n/a | `reports/bert4rec_multimodal/bert4rec_qwen_concat_w16/metrics_test.json` | no | Masked last-position next-item Transformer with item ID plus projected Qwen content vector. |
| `sasrec_multimodal_test.metrics.json` | SASRec-style + Qwen content | n/a | `reports/sasrec_multimodal/sasrec_qwen_concat_w16/metrics_test.json` | no | Causal next-item Transformer with item ID plus projected Qwen content vector. |
| `qwen3_0_6b_fullft_test.metrics.json` | Qwen3-0.6B full-FT | `[512, 256, 128, 64]` | `data/processed/artifacts/sft_qwen3_0_6b_fullft_cpt_fullft_sid_v2_trainval_test_w16_e7_allseen_v1/final_test_metrics.json` | no | SID-v2 trie-constrained generative run; local source records @1/@5/@10 metrics. |
| `qwen3_0_6b_qlora32_test.metrics.json` | Qwen3-0.6B QLoRA32 | `[512, 256, 128, 64]` | `data/processed/artifacts/sft_qwen3_0_6b_qlora32_sid_v2_trainval_test_w16_e1_allseen_v1/test_full_beam20_return20_all_seen_20260519_001408/beam_sweep_results.json` | no | SID-v2 trie-constrained generative run. |
| `qwen3_lora16_test.metrics.json` | Qwen3-4B LoRA16 | `[512, 256, 128, 64]` | `data/processed/artifacts/sft_qwen3_4b_sid_v2_trainval_test_w16_bestepoch_v1/test_full_beam20_return20_strict_full_seen_b5/beam_sweep_results.json` | no | SID-v2 trie-constrained generative run. |
| `qwen3_qlora32_test.metrics.json` | Qwen3-4B QLoRA32 | `[512, 256, 128, 64]` | `data/processed/artifacts/sft_qwen3_4b_qlora32_sid_v2_trainval_test_w16_e3_allseen_v1/test_full_beam20_return20_all_seen_20260515_170416/beam_sweep_results.json` | no | Qwen3-4B QLoRA32 SID-v2 generative run. |

## SID-Depth Ablation Rows

The active SID-v2 assignment has four levels. The 3-level rows below are
separate ablations and should not be read as a silent protocol replacement.

| Snapshot | Method | SID codebooks | Source artifact | Source in git? | Notes |
|---|---|---|---|---|---|
| `qwen3_0_6b_qlora32_3levels_test.metrics.json` | Qwen3-0.6B QLoRA32, 3 SID levels | `[512, 256, 128]` | `data/processed/artifacts/sft_qwen3_0_6b_qlora32_sid_v2_3codebooks_trainval_test_w16_e3_allseen_v1/test_full_beam20_return20_all_seen_20260519_021421/beam_sweep_results.json` | no | 3-level SID-depth ablation. |
| `qwen3_4b_qlora32_3levels_test.metrics.json` | Qwen3-4B QLoRA32, 3 SID levels | `[512, 256, 128]` | `data/processed/artifacts/sft_qwen3_4b_qlora32_sid_v2_3codebooks_trainval_test_w16_e1_allseen_v1/test_beam20_allseen/beam_sweep_results.json` | no | 3-level SID-depth ablation. |

## Validation Rows

Validation rows are kept to document protocol selection and ablations. They are
not held-out test claims.

| Snapshot | Method | SID codebooks | Status |
|---|---|---|---|
| `sid_v1_gpt2_weak_cpt_validation.metrics.json` | GPT-2 S weak-CPT SFT | legacy SID-v1 | Legacy discarded prototype; only documented Recall@10 is preserved. |
| `sid_v2_gpt2_reference_validation.metrics.json` | GPT2-S CPT + SFT | `[512, 256, 128, 64]` | Internal SID-v2 reference. |
| `qwen3_sft_only_validation.metrics.json` | Qwen3-4B SFT-only | `[512, 256, 128, 64]` | No-CPT ablation, validation-only. |
| `qwen3_qlora32_validation.metrics.json` | Qwen3-4B QLoRA32 | `[512, 256, 128, 64]` | Validation-selected generative SID run. |

The no-CPT ablation is validation-only. No held-out test CPT-vs-no-CPT claim is
made from that row unless a separate test artifact is produced later.

## What Is Not Reproducible From Git Alone

A fresh clone can run the lightweight tests, config validation, artifact schema
checks, and smoke test. It cannot reproduce heavy metrics without regenerating
or providing the required local artifacts: MovieLens preprocessing outputs,
movie overviews, Qwen embeddings, SID assignments, CPT/SFT adapters, prediction
dumps, and local evaluation JSON files.

The committed snapshots make the reported scalar metrics auditable, but they do
not contain enough data to re-score predictions.

## Generative Diagnostic Snapshots

Compact diagnostics for Qwen generative runs are stored separately from scalar
metric snapshots:

| Snapshot | Source artifact | Source in git? | Notes |
|---|---|---|---|
| `qwen3_lora16_test.diagnostics.json` | `data/processed/artifacts/sft_qwen3_4b_sid_v2_trainval_test_w16_bestepoch_v1/test_full_beam20_return20_strict_full_seen_b5/beam_sweep_results.json` | no | Beam 20 / return 20, trie-constrained SID decoding. |
| `qwen3_qlora32_test.diagnostics.json` | `data/processed/artifacts/sft_qwen3_4b_qlora32_sid_v2_trainval_test_w16_e3_allseen_v1/test_full_beam20_return20_all_seen_20260515_170416/beam_sweep_results.json` | no | Beam 20 / return 20, trie-constrained SID decoding. |

These files summarize already-existing local beam-sweep outputs. They do not
contain raw generated token sequences or prediction dumps.
