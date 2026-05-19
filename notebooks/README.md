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

## Current Ablations

- `sid_v2/04_qwen4b_rqvae_sid_v2_3codebook_ablation.ipynb`: 3-level SID ablation.
- `cpt/08_cpt_qwen3_4b_base_sid_v2_3codebook_qlora32_canonical.ipynb`: Qwen3-4B CPT for the 3-level SID ablation.
- `cpt/09_cpt_qwen3_0_6b_base_sid_v2_qlora32.ipynb`: Qwen3-0.6B QLoRA32 CPT with active 4-level SID-v2.
- `cpt/10_cpt_qwen3_0_6b_base_sid_v2_3codebook_qlora32.ipynb`: Qwen3-0.6B QLoRA32 CPT with 3-level SID ablation.
- `cpt/11_cpt_qwen3_0_6b_base_sid_v2_full_finetune.ipynb`: Qwen3-0.6B full fine-tuning CPT.
- `sft/22_sft_qwen3_4b_qlora32_sid_v2_3codebook_next_watch_w16_allseen_pat3.ipynb`: Qwen3-4B 3-level SID validation SFT.
- `sft/23_sft_qwen3_4b_qlora32_sid_v2_3codebook_trainval_test_w16_e1_allseen.ipynb`: Qwen3-4B 3-level SID train+val -> test SFT.
- `sft/24_sft_qwen3_0_6b_qlora32_sid_v2_next_watch_w16_allseen_pat3.ipynb`: Qwen3-0.6B QLoRA32 validation SFT with active 4-level SID-v2.
- `sft/25_sft_qwen3_0_6b_qlora32_sid_v2_3codebook_next_watch_w16_allseen_pat3.ipynb`: Qwen3-0.6B QLoRA32 validation SFT with 3-level SID ablation.
- `sft/26_sft_qwen3_0_6b_qlora32_sid_v2_trainval_test_w16_bestepoch_allseen.ipynb`: Qwen3-0.6B QLoRA32 train+val -> test SFT with active 4-level SID-v2.
- `sft/27_sft_qwen3_0_6b_qlora32_sid_v2_3codebook_trainval_test_w16_bestepoch_allseen.ipynb`: Qwen3-0.6B QLoRA32 train+val -> test SFT with 3-level SID ablation.
- `sft/28_sft_qwen3_0_6b_fullft_cpt_qlora32_sid_v2_next_watch_w16_allseen_pat3.ipynb`: Qwen3-0.6B diagnostic run with QLoRA CPT and full-FT SFT; not used as a headline result.
- `sft/29_sft_qwen3_0_6b_fullft_cpt_fullft_sid_v2_next_watch_w16_allseen_pat3.ipynb`: Qwen3-0.6B full fine-tuning validation SFT.
- `sft/30_sft_qwen3_0_6b_fullft_cpt_fullft_sid_v2_trainval_test_w16_e7_allseen.ipynb`: Qwen3-0.6B full fine-tuning train+val -> test SFT.

## Historical Material

- `sid_v1_legacy/` documents the removed SID-v1 prototype notebooks and is not the active protocol.
- `cpt/archive/` documents removed obsolete CPT drafts with five-level SID tokens.
- Earlier GPT2 and Qwen3 LoRA notebooks remain only as reference runs for the reported progression. The active SID-v2 path is the 4-level Qwen3 setup; 3-level notebooks are ablations.
- Superseded Qwen2.5 notebooks with noisy or broken local outputs were removed from the public tree; compact metric provenance is kept in `docs/RESULTS.md`.
