# Results Snapshot

This file records the compact metric snapshot currently reported in the root
README. It is a small provenance document, not a generated prediction dump.
Large artifacts and full prediction files remain outside git.

## Validation

| Run | SID codebooks | Users | Recall@1 | Recall@5 | Recall@10 | NDCG@10 | MRR@10 | Coverage@10 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| SID-v1 GPT-2 weak-CPT SFT | legacy SID-v1 | 6040 | - | - | 0.0253 | - | - | - | discarded first prototype |
| SID-v2 GPT2-S CPT + SFT | `[512, 256, 128, 64]` | 6040 | 0.0336 | 0.1075 | 0.1462 | 0.0836 | 0.0643 | 1185 | internal reference |
| Qwen3-4B SFT-only, no CPT, w16 | `[512, 256, 128, 64]` | 6040 | 0.0366 | 0.1151 | 0.1747 | 0.0954 | 0.0713 | 1470 | full validation, all-prior seen filtering |
| Qwen3-4B CPT-QLoRA32 + SFT-QLoRA32, w16 | `[512, 256, 128, 64]` | 6040 | 0.0646 | 0.1876 | 0.2707 | 0.1538 | 0.1182 | 2173 | full validation, all-prior seen filtering |

The no-CPT ablation is validation-only. There is no held-out test
CPT-vs-no-CPT conclusion in this snapshot.

## Held-Out Test

All rows use all-prior-user-history seen filtering. The Qwen generative rows use
the active train-only CPT checkpoint; train+val refers to the downstream SFT
stage and test-time context source.

| Run | SID codebooks | Users | Recall@1 | Recall@5 | Recall@10 | NDCG@10 | MRR@10 | Coverage@10 | Key settings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Global Popularity | n/a | 6040 | 0.0056 | 0.0202 | 0.0363 | 0.0178 | 0.0123 | 199 | global item frequency from train+val |
| Content KNN | n/a | 6040 | 0.0169 | 0.0414 | 0.0675 | 0.0374 | 0.0284 | 1170 | metadata+overview concat vectors; w16; recency alpha=0.8; rating=none; sum |
| Behavioral ItemKNN BM25-cosine | n/a | 6040 | 0.0427 | 0.1268 | 0.1907 | 0.1056 | 0.0798 | 1788 | w16; alpha=0.8; rating=none; top3 aggregation |
| Qwen3-0.6B full-FT | `[512, 256, 128, 64]` | 6040 | 0.0444 | 0.1437 | 0.2061 | 0.1156 | 0.0878 | 1364 | full-FT CPT + full-FT SFT + SID-v2; w16; all-history filtering |
| Qwen3-0.6B QLoRA32 | `[512, 256, 128, 64]` | 6040 | 0.0613 | 0.1662 | 0.2434 | 0.1395 | 0.1079 | 1776 | CPT-QLoRA32 + SFT-QLoRA32 + SID-v2; w16; all-history filtering |
| Qwen3-4B LoRA16 | `[512, 256, 128, 64]` | 6040 | 0.0626 | 0.1765 | 0.2525 | 0.1451 | 0.1123 | 2108 | CPT-LoRA + SFT-LoRA + SID-v2; w16; all-history filtering |
| Qwen3-4B QLoRA32 | `[512, 256, 128, 64]` | 6040 | 0.0619 | 0.1755 | 0.2550 | 0.1450 | 0.1116 | 2143 | CPT-QLoRA32 + SFT-QLoRA32 + SID-v2; w16; 3 epochs |
| BERT4Rec | n/a | 6040 | 0.0861 | 0.2166 | 0.3066 | 0.1823 | 0.1443 | 2842 | masked last-position next-item Transformer; item ID + projected content vector; w16; 25 epochs |
| SASRec | n/a | 6040 | 0.0844 | 0.2250 | 0.3104 | 0.1835 | 0.1446 | 2455 | causal next-item Transformer; item ID + projected content vector; w16; 3 epochs |

The BERT4Rec and SASRec rows are adapted multimodal sequential
baselines that use item IDs together with projected content vectors. They
are not claimed to be exact canonical reproductions of the original papers. The
PLUM-style Qwen rows are competitive here, but they do not beat the strongest
SASRec/BERT4Rec baselines in the current held-out test table.

## SID Depth Ablation

The active SID-v2 protocol uses four levels with codebook sizes
`[512, 256, 128, 64]`. The 3-level rows below use `[512, 256, 128]` and are
reported only as a depth ablation.

| Run | SID codebooks | Users | Recall@1 | Recall@5 | Recall@10 | NDCG@10 | MRR@10 | Coverage@10 | Key settings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B QLoRA32, 4 levels | `[512, 256, 128, 64]` | 6040 | 0.0613 | 0.1662 | 0.2434 | 0.1395 | 0.1079 | 1776 | active SID-v2 |
| Qwen3-0.6B QLoRA32, 3 levels | `[512, 256, 128]` | 6040 | 0.0623 | 0.1733 | 0.2492 | 0.1434 | 0.1111 | 1843 | SID-depth ablation |
| Qwen3-4B QLoRA32, 4 levels | `[512, 256, 128, 64]` | 6040 | 0.0619 | 0.1755 | 0.2550 | 0.1450 | 0.1116 | 2143 | active SID-v2 |
| Qwen3-4B QLoRA32, 3 levels | `[512, 256, 128]` | 6040 | 0.0616 | 0.1770 | 0.2566 | 0.1459 | 0.1122 | 2073 | SID-depth ablation |

## Baseline and Ablation Scope

| Method | Canonical? | Uses content vectors? | SID codebooks | Split | Seen filtering | Notes |
|---|---:|---:|---|---|---|---|
| Global Popularity | yes | no | n/a | test | all prior | Global train+val item frequency. |
| Content KNN | adapted control | yes | n/a | test | all prior | Non-sequential content similarity over concatenated metadata and overview embeddings. |
| Behavioral ItemKNN BM25-cosine | adapted control | no | n/a | test | all prior | Collaborative item-item similarity over MovieLens interactions. |
| SASRec | adapted | yes | n/a | test | all prior | Local causal Transformer with item IDs plus projected content vectors. |
| BERT4Rec | adapted | yes | n/a | test | all prior | Masked last-position next-item Transformer, not canonical random-mask BERT4Rec pretraining. |
| Qwen3 SID-v2 | adapted PLUM-style | yes via SID/metadata | `[512, 256, 128, 64]` | test | all prior | Generative SID retrieval with trie-constrained decoding. |

ID-only SASRec is a planned baseline to isolate the effect of content
features. It is not reported here because it has not been run under the same
protocol yet.

## Local Sources

The compact numbers above were copied from local JSON reports generated by the
corresponding experiment runs. The source files are intentionally ignored by
git because they live under `data/processed/` or `reports/`.

Representative local paths:

- `data/processed/artifacts/popularity_baseline_trainval_test_full_seen/metrics.json`
- `reports/embedding_knn_qwen4b_v2/concat_meta_description_w16_all_seen/best.json`
- `reports/itemknn_behavior_v2/test_w16_all_seen_best_val_selected/best.json`
- `data/processed/artifacts/sft_qwen3_4b_qlora32_sid_v2_next_watch_w16_allseen_pat3_v1/.../beam_sweep_results.json`
- `data/processed/artifacts/sft_qwen3_4b_qlora32_sid_v2_trainval_test_w16_e3_allseen_v1/.../beam_sweep_results.json`
- `data/processed/artifacts/sft_qwen3_0_6b_qlora32_sid_v2_trainval_test_w16_e1_allseen_v1/.../beam_sweep_results.json`
- `data/processed/artifacts/sft_qwen3_0_6b_fullft_cpt_fullft_sid_v2_trainval_test_w16_e7_allseen_v1/final_test_metrics.json`
- `reports/bert4rec_multimodal/bert4rec_qwen_concat_w16/metrics_test.json`
- `reports/sasrec_multimodal/sasrec_qwen_concat_w16/metrics_test.json`

No metrics in this file should be changed unless the corresponding evaluation
has actually been rerun.
