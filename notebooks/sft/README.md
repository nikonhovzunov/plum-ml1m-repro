# SFT notebooks

Supervised fine-tuning notebooks for next-item generative retrieval.

## Active notebook

- `04_sft_qwen2_5_3b_sid_v2_next_watch_w12.ipynb`
  Current Qwen2.5-3B SFT-LoRA run. It uses SID-v2, target-only loss, history window `12`, trie-constrained SID beam decoding, and the merged CPT checkpoint as its base model.

Current full-validation result:

```text
Recall@10 = 0.2318
NDCG@10   = 0.1348
MRR@10    = 0.1051
```

## GPT-2 baselines and smoke tests

- `00_sft_smoke_quickstart.ipynb`
  Small guarded smoke test for the reusable `src/sft` path.

- `01_gpt2_s_weak_cpt_full_pipeline.ipynb`
  Earlier GPT2-S leave-one-out SFT pipeline.

- `03_sft_gpt2_small_sid_v2_next_watch_plum.ipynb`
  GPT2-S SID-v2 baseline used for the current comparison.

- `02_gpt2_medium_plus_plus_plus_sft_monitor.ipynb`
  Older monitored GPT-2 Medium SFT experiment.

The final test split should stay untouched until the validation protocol is frozen.
