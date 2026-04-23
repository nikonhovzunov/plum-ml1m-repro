# Movie overview enrichment

This research branch enriches MovieLens-1M `movies.dat` with short movie overviews.

## Outputs

- Notebook: `research/movie_overviews/notebooks/00_build_movie_overviews.ipynb`
- CSV: `research/movie_overviews/data/ml1m_movie_overviews.csv`
- Script: `research/movie_overviews/scripts/build_ml1m_movie_overviews.py`
- Local source cache: `research/movie_overviews/cache/hf_movie_descriptors.parquet`

## CSV Schema

The output CSV has exactly these columns:

- `movie_id`
- `title`
- `year`
- `genres`
- `overview`
- `source`
- `status`

Rows are never dropped. If no confident description is found, `overview` and `source` are empty and `status` is `no_description`.

## Source

The executed run uses `mt0rm0/movie_descriptors` from Hugging Face as the primary external source. That dataset is a CC0 subset of Kaggle's The Movie Dataset and contains TMDb-derived movie overviews.

The script also supports TMDb and OMDb live APIs if `TMDB_API_KEY`, `TMDB_BEARER_TOKEN`, or `OMDB_API_KEY` are present in the environment.

## Run

From the repository root:

```powershell
python .\research\movie_overviews\scripts\build_ml1m_movie_overviews.py `
  --providers hf `
  --output-path .\research\movie_overviews\data\ml1m_movie_overviews.csv `
  --cache-path .\research\movie_overviews\cache\movie_overviews_hf_cache.jsonl
```

To include live API providers when keys are configured:

```powershell
python .\research\movie_overviews\scripts\build_ml1m_movie_overviews.py `
  --providers hf,tmdb,omdb `
  --output-path .\research\movie_overviews\data\ml1m_movie_overviews.csv
```
