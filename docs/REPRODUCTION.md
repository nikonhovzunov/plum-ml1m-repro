# Reproduction Protocol

The active protocol is SID-v2:

- MovieLens-1M chronological leave-one-out splits.
- Item text signal: title, year, genres, and audited plot overview.
- Text embeddings: `Qwen/Qwen3-Embedding-4B`, 2560 dimensions.
- SID tokenizer: RQ-VAE with four residual levels and codebook sizes `[1024, 512, 256, 128]`.
- CPT: LoRA continued pre-training on SID-aware item metadata plus train-only behavior.
- SFT: separate LoRA adapter for next watched item prediction.
- Evaluation: trie-constrained SID generation, SID-to-item mapping, seen-item filtering, Recall/NDCG/MRR on original `item_idx`.

The reported metrics in README are existing experiment outputs. They were not recomputed while adding the reproducibility layer.

## Heavy artifacts

Raw data, processed datasets, embeddings, checkpoints, adapters, predictions, and metrics exports are intentionally not tracked by git. See `configs/artifact_manifest.yaml` and `docs/ARTIFACTS.md`.

## Full pipeline shape

```bash
plum-ml1m prepare-data --config configs/prepare_data.yaml
plum-ml1m build-metadata --config configs/metadata.yaml
plum-ml1m build-embeddings --config configs/embeddings.yaml
plum-ml1m train-sid --config configs/rqvae_sid.yaml
plum-ml1m train-cpt --config configs/cpt.yaml
plum-ml1m train-sft --config configs/sft.yaml
plum-ml1m evaluate --config configs/evaluation.yaml
```

By default these commands validate and print a plan. Use explicit execution flags for heavy notebook-backed runs.
