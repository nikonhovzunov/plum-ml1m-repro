# CPT notebooks

Continued pre-training notebooks for the PLUM-style SID language-model stage.

## Active notebooks

- `03_cpt_qwen2_5_3b_base_sid_v2.ipynb`
  Current Qwen2.5-3B LoRA CPT run on SID-v2. The corpus mixes train-only behavior windows with movie metadata and short descriptions, saves the CPT adapter, and writes a merged CPT checkpoint for downstream SFT.

- `04_cpt_qwen2_5_3b_grounding_eval.ipynb`
  Grounding evaluation for CPT: SID to title/year/genres and short-description generation.

## Earlier clean GPT-2 path

- `00_cpt_pipeline_quickstart.ipynb`
  Reusable GPT-2 CPT entrypoint built around `src/cpt/`.

- `01_cpt_gpt2_small_sid_v2_plum.ipynb`
- `02_cpt_gpt2_small_sid_v2_curriculum.ipynb`

## Reusable code

The reusable implementation lives in `src/cpt/`:

- `CPTPipeline` prepares tokenizer and behavior/metadata corpora for any supplied SID matrix.
- `train_gpt2_cpt` runs GPT-2-style continued pre-training from saved CPT artifacts.
- `mean_jaccard_genres` provides a lightweight genre-grounding check.

Archived exploratory notebooks live under `archive/`.
