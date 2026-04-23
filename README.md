# PLUM MovieLens-1M Codebase

Codebase for MovieLens-1M generative recommendation experiments inspired by PLUM.

The repository keeps only the reusable code, scripts, and working notebooks. Local datasets, generated artifacts, ad hoc reports, and exported research files are intentionally not tracked.

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- requirements-lock.txt
|-- src/
|   |-- sid/        # item embeddings, SID construction, RQ-VAE training
|   |-- cpt/        # CPT dataset building and training helpers
|   |-- sft/        # SFT examples, decoding, metrics
|   `-- rqvae.py    # legacy RQ-VAE helper used by older notebooks
|-- scripts/
|   |-- run_qwen4b_rqvae_sid_v2.py
|   |-- run_advanced_rqvae_sid_v2.py
|   `-- reporting/
|-- notebooks/
|   |-- data_prep/
|   |-- sid_v1_legacy/
|   |-- sid_v2/
|   |-- cpt/
|   |-- sft/
|   `-- reporting/
|-- research/
|   `-- movie_overviews/
|       |-- notebooks/
|       `-- scripts/
`-- data/           # local only, ignored by git
```

## Data

Place MovieLens-1M files under:

```text
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

Ignored local paths:

```text
data/processed/
runs/
reports/
research/movie_overviews/data/
```

## Minimal Workflow

1. Run `notebooks/data_prep/00_sanity.ipynb` to validate MovieLens-1M and build splits.
2. Build or refresh movie descriptions in `research/movie_overviews/notebooks/`.
3. Run `notebooks/sid_v2/00_qwen4b_embedding_stage.ipynb`.
4. Run `notebooks/sid_v2/02_qwen4b_rqvae_sid_v2.ipynb`.
5. Run `notebooks/cpt/03_cpt_qwen2_5_3b_base_sid_v2.ipynb`.
6. Run `notebooks/sft/04_sft_qwen2_5_3b_sid_v2_next_watch_w12.ipynb`.

## Environment

```powershell
python -m pip install -r requirements.txt
```

Optional local snapshot:

```powershell
python -m pip install -r requirements-lock.txt
```
