# Notebook Index

Notebook files are kept as experiment records and launch surfaces for heavy
local runs. The reusable protocol logic lives in `src/plum_ml1m/`; notebooks
should call into that code whenever possible.

## Current Public Path

- `data_prep/`: MovieLens sanity checks, item metadata, and behavior pairs.
- `sid_v2/`: Qwen3-Embedding-4B item embeddings, RQ-VAE SID-v2 training, and SID quality diagnostics.
- `cpt/07_cpt_qwen3_4b_base_sid_v2_qlora32.ipynb`: current Qwen3-4B QLoRA32 CPT notebook.
- `sft/19_sft_qwen3_4b_qlora32_sid_v2_next_watch_w16_allseen_pat3.ipynb`: current validation SFT notebook.
- `sft/21_sft_qwen3_4b_qlora32_sid_v2_trainval_test_w16_e3_allseen.ipynb`: current train+val -> test SFT notebook.
- `baselines/`: multimodal BERT4Rec and SASRec notebooks using the same Qwen3-4B content vectors.

## Historical Material

- `sid_v1_legacy/` documents the removed SID-v1 prototype notebooks and is not the active protocol.
- `cpt/archive/` documents removed obsolete CPT drafts with five-level SID tokens.
- Earlier GPT2 and Qwen3 LoRA notebooks remain only as reference runs for the reported progression. SID-v2 QLoRA32 is the current active generative path.
- Superseded Qwen2.5 notebooks with noisy or broken local outputs were removed from the public tree; compact metric provenance is kept in `docs/RESULTS.md`.
