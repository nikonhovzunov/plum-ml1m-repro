# PLUM reproduction on MovieLens-1M

This repository is a step-by-step reproduction project for a PLUM-style generative recommendation pipeline on MovieLens-1M.
Current focus: data prep, item features, and co-occurrence neighbor pairs for Semantic ID (SID) training.

## Project structure

- `data/raw/` — original datasets (MovieLens-1M)
- `data/processed/` — processed splits and artifacts (generated locally)
- `notebooks/` — step-by-step notebooks (run in order)
- `src/` — reusable Python modules (will grow as the project evolves)
- `runs/` — experiment outputs (logs, checkpoints; generated locally)
- `reports/` — final report and figures

## Data

**Data is not tracked by git** (see `.gitignore`). You need to download and generate it locally.

Expected raw files (MovieLens-1M) under:
- `data/raw/ml-1m/ratings.dat`
- `data/raw/ml-1m/movies.dat`
- `data/raw/ml-1m/users.dat`

Processed artifacts will be created under:
- `data/processed/splits/` — train/val/test parquet splits
- `data/processed/item_features/` — item feature arrays (e.g., title SVD, year scaling)
- `data/processed/sid_pairs/` — co-occurrence neighbor pairs for SID training

## Notebooks (run in order)

- `00_sanity.ipynb`  
  Load MovieLens-1M, validate schema, build train/val/test splits, and compute a simple popularity baseline.
- `01_item_features.ipynb`  
  Build item features: title TF-IDF → TruncatedSVD (dense embedding) and year normalization (MinMax scaling).  
  (Optional) genres multi-hot encoding.
- `02_sid_pairs.ipynb`  
  Build co-occurrence **neighbor pairs** from train sequences (i_t → i_{t+1}) for SID/contrastive training.

## Repro steps (quick)

1. Download MovieLens-1M and place files into `data/raw/ml-1m/`.
2. Run notebooks in order: `00_sanity` → `01_item_features` → `02_sid_pairs`.
3. Next steps (WIP): train SID model (RQ-VAE style discretization) and a generative retrieval model.

## Notes

- The repository tracks code and notebooks, but **does not upload datasets or large artifacts**.
- All experiments should be reproducible from raw data using the notebooks above.
