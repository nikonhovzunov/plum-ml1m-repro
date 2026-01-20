# PLUM reproduction on MovieLens-1M

This repository is a step-by-step reproduction project for a PLUM-style generative recommendation pipeline on MovieLens-1M.

**Current milestone:** a working baseline **RQ-VAE** that discretizes item features into multi-level **Semantic IDs (SIDs)**, with full run logging (configs, metrics, checkpoints).  
**Next:** progressive masking + co-occurrence (contrastive) loss, then GPT-2 generative retrieval.

---

## Project structure

- `data/raw/` — original datasets (MovieLens-1M)
- `data/processed/` — processed splits and artifacts (generated locally)
- `notebooks/` — step-by-step notebooks (run in order)
- `src/` — reusable Python modules (will grow as the project evolves)
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
  - genres: multi-hot encoding (optional but recommended; improves codebook utilization)

- `02_sid_pairs.ipynb`  
  Build co-occurrence **neighbor pairs** from train sequences (i_t → i_{t+1}) for future SID/contrastive regularization.

- `03_rqvae_model.ipynb`  
  Train baseline **RQ-VAE** to map item features → multi-level discrete codes (**SIDs**).  
  Logs losses + SID health metrics and saves checkpoints to a dedicated run folder in `runs/`.

- `04_eval_runs.ipynb`  
  Load a selected `runs/<run_dir>/` and plot training curves:
  - total/recon/codebook/commit losses (train/val)
  - unique SID counts per level (**U**)
  - perplexity per level (**PPL**)

---

## Experiment logging (`runs/`)

Each training run creates a new folder:

```text
runs/<timestamp>_<variant>/
  config.json
  metrics.json
  checkpoint_last.pt
  checkpoint_best.pt

- `config.json` — model/data hyperparameters for the run
- `metrics.json` — per-epoch metrics:
  - losses: `total`, `recon`, `codebook`, `commit` (train/val)
  - codebook usage: `U` and `PPL` per level (train/val)
- `checkpoint_last.pt` — last epoch checkpoint (model + optimizer)
- `checkpoint_best.pt` — best checkpoint by `val_loss`

Naming convention examples:
- `*_baseline_rqvae_title_year_genres`
- `*_pmask_*`
- `*_cooc_*`
- `*_full_*`

---

## Repro steps (quick)

1. Download MovieLens-1M and place files into `data/raw/ml-1m/`.
2. Run notebooks in order:  
   `00_sanity` → `01_item_features` → `02_sid_pairs` → `03_rqvae_model`
3. Visualize results for a run in: `04_eval_runs`

---

## Interpreting SID metrics (short)

- **U (unique SIDs per level)**: how many different codes are actually used at each level.  
  Higher U usually means the codebook is utilized better (less collapse).

- **PPL (perplexity per level)**: “effective number of codes” used at that level.  
  PPL grows when the distribution of selected codes becomes more diverse.

---

## Roadmap / Next steps

1. **Progressive masking (WIP)**  
   Train with a schedule that gradually activates more SID levels (stabilizes learning and improves factorization).

2. **Co-occurrence / contrastive loss (WIP)**  
   Use `data/processed/sid_pairs/*` to enforce that neighboring items in user sequences have compatible representations/codes.

3. **Generative retrieval (planned)**  
   Train a GPT-2 model to generate item SIDs from user context.

---

## Notes

- The repository tracks code and notebooks, but **does not upload datasets or large artifacts**.
- `.idea/` and other IDE files should not be committed (use `.gitignore`).
- All experiments should be reproducible from raw data using the notebooks above.
