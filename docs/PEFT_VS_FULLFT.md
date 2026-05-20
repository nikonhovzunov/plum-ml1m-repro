# PEFT vs Full Fine-Tuning in the Qwen3-0.6B Run

This note documents the controlled Qwen3-0.6B comparison used in the current
project report. It is intentionally narrow: it explains one reported
MovieLens-1M experiment and does not claim universal PEFT superiority.

## Motivation

The coursework focuses on LoRA/QLoRA adaptation for a PLUM-style generative
recommendation pipeline. For that reason, the comparison against full
fine-tuning is important: it checks whether QLoRA is only a memory-saving trick
or whether it can also be a competitive adaptation strategy for Semantic-ID
generative retrieval in a low-resource setting.

Qwen3-0.6B is a useful controlled setting because both compared runs use the
same base model size and the same SID-v2 recommendation protocol.

## Experimental Setup

| Component | Qwen3-0.6B QLoRA32 | Qwen3-0.6B full-FT |
|---|---|---|
| Base model | Qwen3-0.6B | Qwen3-0.6B |
| Dataset | MovieLens-1M | MovieLens-1M |
| SID protocol | SID-v2 | SID-v2 |
| SID levels | 4 | 4 |
| SID codebooks | `[512, 256, 128, 64]` | `[512, 256, 128, 64]` |
| History window | 16 | 16 |
| CPT adaptation | QLoRA | full-FT |
| SFT adaptation | QLoRA | full-FT |
| SFT epoch used for test row | 1 | 7 |
| Seen filtering | all-prior-user-history | all-prior-user-history |
| Decoding | trie-constrained, beam 20, return 20 | trie-constrained, beam 20, return 20 |
| Split | held-out test | held-out test |
| Source run commit | not recorded | not recorded |
| Full hyperparameter sweep | not available | not available |

The compact snapshots are:

- `reports/snapshots/qwen3_0_6b_qlora32_test.metrics.json`
- `reports/snapshots/qwen3_0_6b_fullft_test.metrics.json`

The source artifacts are local ignored files under `data/processed/artifacts/`;
model weights, adapters, checkpoints, and prediction dumps are not committed.

## Quality Results

| Model | Adaptation | Recall@10 | NDCG@10 | MRR@10 | Coverage@10 |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | QLoRA32 | 0.2434 | 0.1395 | 0.1079 | 1776 |
| Qwen3-0.6B | full-FT | 0.2061 | 0.1156 | 0.0878 | 1364 |

In the reported controlled Qwen3-0.6B single-run comparison, QLoRA32
outperforms the full-FT baseline under the same evaluation protocol.

## Generative Diagnostics

| Model | Invalid SID rate | Seen generated rate | Duplicate valid rate | Avg unique filtered candidates |
|---|---:|---:|---:|---:|
| Qwen3-0.6B QLoRA32 | 0.0000 | 0.2564 | not recorded | 14.8728 |
| Qwen3-0.6B full-FT | 0.0000 | 0.2322 | not recorded | 9.5533 |

Both rows use trie-constrained decoding, so invalid SID rate is 0. The compact
0.6B snapshots do not record duplicate-valid-rate diagnostics; that value is
therefore left as `not recorded` rather than inferred.

## Interpretation

In the reported controlled Qwen3-0.6B single-run comparison, QLoRA32
outperforms the full-FT baseline under the same evaluation protocol. A
plausible interpretation is that PEFT can act as a regularizer in the small
MovieLens-1M setting and may avoid damaging useful pretrained representations
during adaptation.

This should be read as evidence that PEFT can be competitive, and in this
single run better, under this PLUM-style Semantic-ID protocol. This is not a
universal claim and should be interpreted under the limitations of single-run
evaluation and limited hyperparameter exploration.

## Limitations

- This is a single-run comparison.
- There is no multi-seed confidence interval.
- There is no full hyperparameter sweep for either QLoRA32 or full-FT.
- The full-FT result may be sensitive to learning rate, checkpoint selection,
  batch size, and stopping criteria.
- Some low-level training details are not recorded in the compact snapshots.
- Full training artifacts, checkpoints, adapters, and prediction dumps are not
  committed.
- The result is specific to MovieLens-1M and this SID-v2, window-16,
  trie-constrained, all-prior seen-filtered protocol.
- The result does not imply that PLUM-style Qwen retrieval beats the
  SASRec or BERT4Rec rows in the current held-out test table.

## How To Cite This Result In The Report

В контролируемом сравнении Qwen3-0.6B QLoRA32 и Qwen3-0.6B full-FT при
одинаковом SID-v2 protocol, trie-constrained decoding и all-history seen
filtering QLoRA32 показала более высокие Recall@10, NDCG@10, MRR@10 и
Coverage@10. Это указывает, что PEFT/QLoRA в рассматриваемом low-resource
PLUM-style setup может быть не только способом экономии памяти, но и полезной
регуляризующей стратегией. При этом результат не следует интерпретировать как
универсальное превосходство QLoRA над полным дообучением, поскольку эксперимент
является single-run и не включает полный multi-seed/hyperparameter sweep.
