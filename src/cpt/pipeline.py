from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CPTCorpusBuilder, CPTCorpusConfig
from .schema import CPTSchema


@dataclass(frozen=True)
class CPTArtifactPaths:
    processed_dir: Path = Path("data/processed")
    raw_ml1m_dir: Path = Path("data/raw/ml-1m")
    artifacts_dir: Path = Path("data/processed/artifacts")

    @property
    def train_path(self) -> Path:
        return self.processed_dir / "splits" / "train.parquet"

    @property
    def item_meta_path(self) -> Path:
        return self.processed_dir / "item_features" / "item_meta.parquet"

    @property
    def users_path(self) -> Path:
        return self.raw_ml1m_dir / "users.dat"


class CPTPipeline:
    def __init__(
        self,
        paths: CPTArtifactPaths | None = None,
        schema: CPTSchema | None = None,
        corpus_config: CPTCorpusConfig | None = None,
    ):
        self.paths = paths or CPTArtifactPaths()
        self.schema = schema or CPTSchema()
        self.corpus_config = corpus_config or CPTCorpusConfig()
        self.builder = CPTCorpusBuilder(self.schema, self.corpus_config)

    def load_tables(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = pd.read_parquet(self.paths.train_path)
        item_meta = pd.read_parquet(self.paths.item_meta_path)
        users = pd.read_csv(
            self.paths.users_path,
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip"],
            encoding="latin-1",
        )
        return train, users, item_meta

    def prepare_artifacts(
        self,
        sids: np.ndarray,
        run_name: str,
        base_tokenizer: str = "gpt2",
        save: bool = True,
    ) -> dict:
        sids = np.asarray(sids)
        train, users, item_meta = self.load_tables()

        self._validate_sids(sids, item_meta)

        tokenizer = self.builder.build_tokenizer(base_tokenizer, sids=sids, users=users)
        corpora = self.builder.build_all(
            tokenizer=tokenizer,
            train=train,
            users=users,
            item_meta=item_meta,
            sids=sids,
        )

        out_dir = self.paths.artifacts_dir / run_name
        if save:
            self.builder.save_artifacts(out_dir, tokenizer, corpora, sids)

        return {
            "out_dir": out_dir,
            "tokenizer": tokenizer,
            "corpora": corpora,
            "stats": self.corpus_stats(corpora),
        }

    def corpus_stats(self, corpora: dict[str, list[list[int]]]) -> dict[str, dict[str, int | float]]:
        stats = {}
        for name, rows in corpora.items():
            lengths = np.array([len(row) for row in rows], dtype=np.int64)
            stats[name] = {
                "rows": int(len(rows)),
                "min": int(lengths.min()) if len(lengths) else 0,
                "median": float(np.median(lengths)) if len(lengths) else 0.0,
                "p95": float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
                "max": int(lengths.max()) if len(lengths) else 0,
            }
        return stats

    def _validate_sids(self, sids: np.ndarray, item_meta: pd.DataFrame) -> None:
        if sids.ndim != 2:
            raise ValueError(f"sids must be a 2D array, got shape {sids.shape}")
        expected_items = int(item_meta["item_idx"].max()) + 1
        if sids.shape[0] < expected_items:
            raise ValueError(
                f"sids has {sids.shape[0]} rows, but item_meta needs {expected_items}"
            )
