# Experiment Scripts

These scripts are heavy local launchers and diagnostics. They are not executed
by CI. Install the package first with:

```bash
python -m pip install -e ".[dev]"
```

## Baselines

- `popularity_baseline.py`: global popularity sanity baseline.
- `qwen4b_embedding_knn_sweep.py`: content KNN over concatenated Qwen3-Embedding-4B metadata + overview embeddings.
- `itemknn_behavior_sweep.py`: behavioral ItemKNN under the same split and seen-filtering protocol.
- `bert4rec_multimodal_train.py`: BERT4Rec with item IDs plus projected Qwen3-4B content vectors.
- `sasrec_multimodal_train.py`: SASRec with item IDs plus projected Qwen3-4B content vectors.

## Generative SID Experiments

- `qwen3_sft_beam_sweep.py`: trie-constrained evaluation for Qwen SFT adapters.
- `run_qwen3_no_cpt_ablation.py`: direct SFT ablation without CPT grounding.
- `run_qwen3_trainval_sft_and_test.py`: legacy Qwen3 LoRA train+val -> test run kept for the earlier all-history row.
- `run_qwen3_qlora32_trainval_e3_allseen_test.py`: current Qwen3 QLoRA32 train+val -> test launcher.
- `run_with_vram_guard.py`: local guard wrapper for long GPU runs.

## SID / RQ-VAE

- `run_qwen4b_rqvae_sid_v2.py`: current SID-v2 RQ-VAE launcher.
- `run_advanced_rqvae_sid_v2.py`: older advanced SID-v2 launcher kept for comparison.
