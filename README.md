# PLUM reproduction on MovieLens-1M

This repository is a step-by-step reproduction project for a PLUM-style generative recommendation pipeline on MovieLens-1M.

**Current milestone:** a working **RQ-VAE** that discretizes item features into multi-level **Semantic IDs (SIDs)** with:
- **Progressive masking** (stabilizes learning + improves codebook utilization)
- **Co-occurrence contrastive regularization** (uses user behavior pairs to shape the embedding space)
- Full run logging (**configs, metrics, checkpoints**) and comparison utilities


---

## Project structure

- `data/raw/` — original datasets (MovieLens-1M)
- `data/processed/` — processed splits and artifacts (generated locally)
- `notebooks/` — step-by-step notebooks (run in order)
- `src/` — reusable Python modules (model/loss/utilities)
- `runs/` — experiment outputs (configs, metrics, checkpoints; generated locally)
- `reports/` — final report and figures

---

## Data

**Data is not tracked by git** (see `.gitignore`). You need to download and generate it locally.

Expected raw files (MovieLens-1M) under:
- `data/raw/ml-1m/ratings.dat`
- `data/raw/ml-1m/movies.dat`
- `data/raw/ml-1m/users.dat`

Processed artifacts will be created under:
- `data/processed/splits/` — train/val/test parquet splits
- `data/processed/item_features/` — item feature arrays (title SVD, year scaling, genres)
- `data/processed/sid_pairs/` — co-occurrence neighbor pairs (i_t → i_{t+1}) for contrastive training

---

## Notebooks (run in order)

- `00_sanity.ipynb`  
  Load MovieLens-1M, validate schema, build train/val/test splits, and compute a popularity baseline.

- `01_item_features.ipynb`  
  Build item features:
  - title: TF-IDF → TruncatedSVD (dense embedding)
  - year: normalization / scaling
  - genres: multi-hot encoding

- `02_sid_pairs.ipynb`  
  Build co-occurrence **neighbor pairs** from train sequences (i_t → i_{t+1}).  
  Output is used for **co-occurrence contrastive regularization** during RQ-VAE training.

- `03_rqvae_model.ipynb`  
  Train **RQ-VAE** variants:
  - baseline RQ-VAE
  - **RQ-VAE + Progressive Masking**
  - **RQ-VAE + Progressive Masking + Co-occurrence Contrastive Regularization**

  Logs losses + SID health metrics and saves checkpoints to a dedicated run folder in `runs/`.

- `04_eval_runs.ipynb`  
  Load a selected `runs/<run_dir>/` and:
  - plot training curves (train/val)
  - compare multiple runs
  - compute **SID uniqueness** on full item set and per-batch
  - load `checkpoint_best.pt` vs `checkpoint_last.pt` and evaluate under consistent criteria

- `05_compare_runs.ipynb` 
  Side-by-side comparison of saved runs (configs + metrics + final evaluation table).

---

## Model code (`src/`)

Reusable implementation lives in:

- `src/rqvae.py`
  - `RQVAE` (Encoder / MultiResolutionCodebooks / Decoder)
  - `RQVAELoss`
    - recon loss + codebook + commitment
    - co-occurrence contrastive regularization `L_con`

---

## Co-occurrence contrastive regularization (L_con)

We inject a contrastive loss during RQ-VAE training using item pairs mined from user sequences
(i_t → i_{t+1}). This shapes the embedding space so that **frequently co-watched items**
have more compatible representations.

Pairs are generated in `02_sid_pairs.ipynb` and consumed during training in `03_rqvae_model.ipynb`.

---


## Notes

- Repo tracks code and notebooks, but **does not upload datasets or large artifacts**.
- `.idea/` and other IDE files should not be committed (use `.gitignore`).
- All experiments should be reproducible from raw data using the notebooks above.
