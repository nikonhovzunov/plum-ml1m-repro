# PLUM reproduction on MovieLens-1M

This repository is a step-by-step reproduction project for a PLUM-style generative recommendation pipeline on MovieLens-1M.

**Current milestone:** a working **RQ-VAE** that discretizes item features into multi-level **Semantic IDs (SIDs)** with:
- **Progressive masking** (stabilizes learning + improves codebook utilization)
- **Co-occurrence contrastive regularization** (uses user behavior pairs to shape the embedding space)
- Full run logging (**configs, metrics, checkpoints**) and comparison utilities

**Next:** finalize robust evaluation + pick best checkpoints by target metric, then GPT-2 generative retrieval.

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
  - genres: multi-hot encoding (recommended; improves codebook utilization)

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

- `05_compare_runs.ipynb` *(added)*  
  Side-by-side comparison of saved runs (configs + metrics + final evaluation table).

---

## Model code (`src/`)

Reusable implementation lives in:

- `src/rqvae.py`
  - `RQVAE` (Encoder / MultiResolutionCodebooks / Decoder)
  - `RQVAELoss`
    - recon loss + codebook + commitment
    - co-occurrence contrastive regularization `L_con`

> If importing from notebooks fails (`No module named 'src'`), run notebooks from repository root
or ensure the project root is on `PYTHONPATH`. See “Imports / package setup” below.

---

## Experiment logging (`runs/`)

Each training run creates a new folder:

```text
runs/<timestamp>_<variant>/
  config.json
  metrics.json
  checkpoint_last.pt
  checkpoint_best.pt
```

- `config.json` — model/data hyperparameters for the run
- `metrics.json` — per-epoch metrics:
  - losses: `total`, `recon`, `codebook`, `commit` (+ optional `con`) (train/val)
  - codebook usage: `U` and `PPL` per level (train/val)
- `checkpoint_last.pt` — last epoch checkpoint (model + optimizer)
- `checkpoint_best.pt` — best checkpoint by `val_loss` (default criterion)

Naming convention examples:
- `*_baseline_rqvae_title_year_genres`
- `*_rqvae_*_PROGRESSIVE_MASKING`
- `*_rqvae_*_PROGRESSIVE_MASKING_ContrastiveRegularization`

---

## Repro steps (quick)

1. Download MovieLens-1M and place files into `data/raw/ml-1m/`.
2. Run notebooks in order:  
   `00_sanity` → `01_item_features` → `02_sid_pairs` → `03_rqvae_model`
3. Compare runs and evaluate SIDs in:  
   `04_eval_runs` / `05_compare_runs`

---

## Interpreting SID metrics

### U (unique SIDs per level)
How many different codes are used at each level (per epoch, based on training/val batches).

Higher **U** often means better utilization (less collapse), but it can also increase while reconstruction worsens.

### U% (unique SIDs per batch)
A normalized version of U:

\[
U\_\%(e,l)=\frac{U(e,l)}{B}\cdot 100
\]

Where `B` is batch size (e.g., 512).

Useful for fair comparison across different batch sizes or splits.

### PPL (perplexity per level)
“Effective number of codes” used at that level.  
PPL grows when the distribution of selected codes becomes more diverse.

---

## Co-occurrence contrastive regularization (L_con)

We inject a contrastive loss during RQ-VAE training using item pairs mined from user sequences
(i_t → i_{t+1}). This shapes the embedding space so that **frequently co-watched items**
have more compatible representations.

Pairs are generated in `02_sid_pairs.ipynb` and consumed during training in `03_rqvae_model.ipynb`.

---

## Objective comparison of runs (important)

By default `checkpoint_best.pt` is chosen by **minimum val_loss**.
However, some variants may:
- temporarily achieve low val_loss with **collapsed** code usage,
- later expand the discrete space (higher U/PPL, better SID diversity) with slightly worse loss.

Recommended evaluation protocol:
1. Evaluate the same checkpoint types for every run (e.g., `best` and `last`).
2. Report:
   - full-dataset loss components (recon / codebook / commit / con)
   - full-dataset SID uniqueness: `len(unique(SIDs))/N_items`
   - average per-batch U% across a fixed DataLoader
3. Pick the “best” checkpoint by a composite criterion, e.g.:
   - primary: SID diversity (Uniqueness, U%, PPL)
   - constraint: recon loss not worse than X%
   - tie-breaker: total val_loss

---

## Imports / package setup (Windows + Conda + Jupyter)

If you want `from src.rqvae import RQVAE, RQVAELoss` to work from any notebook:

**Option A (simple):** run notebooks from repo root (so root is in `sys.path`).

**Option B:** add root to `sys.path` at the top of the notebook:

```python
import sys
from pathlib import Path
sys.path.append(str(Path("..").resolve()))
```

**Option C:** install the repo in editable mode (`pip install -e .`) in the same environment
that Jupyter is using.

---

## Roadmap / Next steps

1. **Finalize evaluation tables**
   - compute loss components on full item set for multiple checkpoints
   - aggregate SID diversity metrics into a single comparison DataFrame

2. **Tune contrastive regularization**
   - temperature / normalization
   - contrastive weight `lambda_con`
   - stability checks (avoid NaNs, scaling issues)

3. **Generative retrieval (planned)**
   Train GPT-2 to generate item SIDs from user context.

---

## Notes

- Repo tracks code and notebooks, but **does not upload datasets or large artifacts**.
- `.idea/` and other IDE files should not be committed (use `.gitignore`).
- All experiments should be reproducible from raw data using the notebooks above.
