# Repository Audit Summary

This audit focused on engineering and reproducibility quality, not benchmark expansion.

## Findings

- Core SID-v2 protocol is four levels with codebook sizes `[1024, 512, 256, 128]`.
- Several notebooks contained local copies of SID formatting, trie decoding, metric, and path logic.
- Existing `src/sft` utilities already covered parts of mapping, decoding, and metrics, but the public package surface was missing.
- A stale default of five SID levels existed in `src/sft/schema.py`; the active SID-v2 protocol uses four levels.
- Heavy generated artifacts are ignored by git through `data/processed/`, `data/raw/`, and `runs/`.

## Actions Taken

- Added a lightweight package under `src/plum_ml1m`.
- Added tests for SID schema, decoding, metrics, split checks, configs, artifacts, and CLI smoke.
- Added configs, artifact manifest, Makefile, pyproject, CI, and documentation.
- Preserved reported metrics without recomputation.
- Added no external recommender baselines.
