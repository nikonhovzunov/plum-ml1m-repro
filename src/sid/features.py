"""Feature builders for PLUM-style Semantic ID experiments.

The canonical SID model should not treat title/year/genres and description as
independent item identities. Instead, each modality is embedded separately and
then fused before RQ-VAE quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBundle:
    """Dense item features aligned by ``item_idx``."""

    item_idx: np.ndarray
    movie_id: np.ndarray
    meta: np.ndarray
    description: np.ndarray
    description_mask: np.ndarray

    @property
    def n_items(self) -> int:
        return int(self.item_idx.shape[0])

    @property
    def modality_dims(self) -> dict[str, int]:
        return {
            "meta": int(self.meta.shape[1]),
            "description": int(self.description.shape[1]),
        }


def _format_genres(genres: object) -> str:
    if isinstance(genres, str):
        if genres.startswith("[") and genres.endswith("]"):
            return genres
        return ", ".join(part for part in genres.split("|") if part)
    if isinstance(genres, Iterable):
        return ", ".join(str(part) for part in genres)
    return ""


def build_item_text_profiles(
    item_meta: pd.DataFrame,
    overviews: pd.DataFrame,
) -> pd.DataFrame:
    """Build stable item text profiles aligned with current ``item_idx``.

    Parameters
    ----------
    item_meta:
        Current project item index table. Must contain ``movie_id`` and
        ``item_idx``; usually ``data/processed/item_features/item_meta.parquet``.
    overviews:
        Movie overview CSV produced in ``research/movie_overviews``.
    """

    required_item = {"movie_id", "item_idx", "title", "years", "genres"}
    required_overview = {"movie_id", "overview", "status"}
    missing_item = required_item - set(item_meta.columns)
    missing_overview = required_overview - set(overviews.columns)
    if missing_item:
        raise ValueError(f"item_meta is missing columns: {sorted(missing_item)}")
    if missing_overview:
        raise ValueError(f"overviews is missing columns: {sorted(missing_overview)}")

    merged = item_meta.merge(
        overviews[["movie_id", "overview", "source", "status"]],
        on="movie_id",
        how="left",
        validate="one_to_one",
    ).sort_values("item_idx")

    if merged["overview"].isna().any():
        missing = merged.loc[merged["overview"].isna(), "movie_id"].head(10).tolist()
        raise ValueError(f"Missing overview rows for movie_id sample: {missing}")

    merged["genres_text"] = merged["genres"].map(_format_genres)
    merged["description_mask"] = merged["overview"].fillna("").str.strip().ne("").astype(np.float32)
    merged["meta_text"] = merged.apply(
        lambda row: (
            f"Movie title: {str(row['title']).strip()}. "
            f"Release year: {int(row['years']) if pd.notna(row['years']) else 'unknown'}. "
            f"Genres: {row['genres_text']}."
        ),
        axis=1,
    )
    merged["description_text"] = merged["overview"].fillna("").map(
        lambda text: f"Plot overview: {str(text).strip()}"
    )
    merged["profile_text"] = merged["meta_text"] + " " + merged["description_text"]

    return merged[
        [
            "movie_id",
            "item_idx",
            "title",
            "years",
            "genres_text",
            "overview",
            "source",
            "status",
            "description_mask",
            "meta_text",
            "description_text",
            "profile_text",
        ]
    ].reset_index(drop=True)


def _encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str | None,
    normalize_embeddings: bool,
    local_files_only: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=device,
        local_files_only=local_files_only,
    )
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32, copy=False)


def encode_text_modalities(
    profiles: pd.DataFrame,
    output_path: str | Path,
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 64,
    device: str | None = None,
    normalize_embeddings: bool = True,
    local_files_only: bool = True,
) -> FeatureBundle:
    """Encode meta and description text and save an ``npz`` feature bundle."""

    required = {"item_idx", "movie_id", "meta_text", "description_text", "description_mask"}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"profiles is missing columns: {sorted(missing)}")

    profiles = profiles.sort_values("item_idx").reset_index(drop=True)
    meta = _encode_texts(
        profiles["meta_text"].tolist(),
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize_embeddings=normalize_embeddings,
        local_files_only=local_files_only,
    )
    description = _encode_texts(
        profiles["description_text"].tolist(),
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize_embeddings=normalize_embeddings,
        local_files_only=local_files_only,
    )

    bundle = FeatureBundle(
        item_idx=profiles["item_idx"].to_numpy(np.int64),
        movie_id=profiles["movie_id"].to_numpy(np.int64),
        meta=meta,
        description=description,
        description_mask=profiles["description_mask"].to_numpy(np.float32).reshape(-1, 1),
    )
    save_feature_bundle(bundle, output_path)
    return bundle


def save_feature_bundle(bundle: FeatureBundle, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        item_idx=bundle.item_idx,
        movie_id=bundle.movie_id,
        meta=bundle.meta,
        description=bundle.description,
        description_mask=bundle.description_mask,
    )


def load_feature_bundle(path: str | Path) -> FeatureBundle:
    data = np.load(Path(path))
    item_idx = data["item_idx"].astype(np.int64)
    description_mask = (
        data["description_mask"].astype(np.float32)
        if "description_mask" in data.files
        else np.ones((item_idx.shape[0], 1), dtype=np.float32)
    )
    return FeatureBundle(
        item_idx=item_idx,
        movie_id=data["movie_id"].astype(np.int64),
        meta=data["meta"].astype(np.float32),
        description=data["description"].astype(np.float32),
        description_mask=description_mask,
    )
