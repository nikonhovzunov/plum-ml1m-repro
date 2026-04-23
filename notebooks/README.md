# Notebook Map

This directory is organized by research stage.

## Active Path

1. `data_prep/00_sanity.ipynb`
   MovieLens-1M loading, validation, reindexing, chronological train/val/test split.

2. `sid_v2/00_qwen4b_embedding_stage.ipynb`
   `Qwen/Qwen3-Embedding-4B` embeddings for movie metadata and plot descriptions.

3. `sid_v2/02_qwen4b_rqvae_sid_v2.ipynb`
   Current RQ-VAE Semantic ID training.

4. `sid_v2/03_sid_quality_heuristics.ipynb`
   SID interpretability and collision diagnostics.

5. `cpt/03_cpt_qwen2_5_3b_base_sid_v2.ipynb`
   Qwen2.5-3B LoRA CPT on SID behavior and metadata, producing both adapter and merged checkpoint artifacts.

6. `cpt/04_cpt_qwen2_5_3b_grounding_eval.ipynb`
   CPT grounding checks: SID to title/year/genres/descriptions.

7. `sft/04_sft_qwen2_5_3b_sid_v2_next_watch_w12.ipynb`
   Current Qwen2.5-3B SFT-LoRA next-item retrieval experiment on top of the merged CPT checkpoint.

## Legacy and Reporting

- `sid_v1_legacy/` contains the first RQ-VAE/SID notebooks. They are useful for historical comparison but are no longer the active path.
- `cpt/archive/` contains older exploratory GPT-2 CPT notebooks. Outputs were preserved.
- `reporting/` contains executed report-building notebooks.
