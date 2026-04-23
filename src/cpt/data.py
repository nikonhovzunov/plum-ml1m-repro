from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import CPTSchema


@dataclass(frozen=True)
class CPTCorpusConfig:
    behavior_last_k: int = 125
    max_length: int = 1024
    include_user_features: bool = True
    include_ratings: bool = True
    truncate_long_histories: bool = True


class CPTCorpusBuilder:
    def __init__(
        self,
        schema: CPTSchema | None = None,
        config: CPTCorpusConfig | None = None,
    ):
        self.schema = schema or CPTSchema()
        self.config = config or CPTCorpusConfig()

    def build_tokenizer(self, base_tokenizer: str, sids: np.ndarray, users: pd.DataFrame):
        try:
            from transformers import GPT2TokenizerFast
        except ImportError as exc:
            raise ImportError(
                "transformers is required to build a GPT-2 tokenizer. "
                "Install it in the notebook environment first."
            ) from exc

        tokenizer = GPT2TokenizerFast.from_pretrained(base_tokenizer)
        self.add_tokens(tokenizer, sids=sids, users=users)
        return tokenizer

    def add_tokens(self, tokenizer: Any, sids: np.ndarray, users: pd.DataFrame) -> int:
        tokenizer.add_special_tokens(self.schema.special_tokens)

        extra: set[str] = set(self.schema.schema_tokens)
        extra.update(self.schema.rating_token(rating) for rating in range(1, 6))

        if {"gender", "age", "occupation"}.issubset(users.columns):
            extra.update(f"<gen_{gender}>" for gender in users["gender"].dropna().unique())
            extra.update(f"<age_{int(age)}>" for age in users["age"].dropna().unique())
            extra.update(
                f"<occ_{int(occupation)}>"
                for occupation in users["occupation"].dropna().unique()
            )

        for sid in np.asarray(sids):
            extra.update(self.schema.sid_tokens(sid))

        return tokenizer.add_tokens(sorted(extra))

    def build_all(
        self,
        tokenizer: Any,
        train: pd.DataFrame,
        users: pd.DataFrame,
        item_meta: pd.DataFrame,
        sids: np.ndarray,
    ) -> dict[str, list[list[int]]]:
        return {
            "behavior": self.build_behavior_examples(tokenizer, train, users, sids),
            "meta_title": self.build_title_examples(tokenizer, item_meta, sids),
            "meta_genre": self.build_genre_examples(tokenizer, item_meta, sids),
            "meta_year": self.build_year_examples(tokenizer, item_meta, sids),
        }

    def build_behavior_examples(
        self,
        tokenizer: Any,
        train: pd.DataFrame,
        users: pd.DataFrame,
        sids: np.ndarray,
    ) -> list[list[int]]:
        required = {"user_id", "user_idx", "item_idx", "rating", "timestamp"}
        missing = required - set(train.columns)
        if missing:
            raise ValueError(f"train is missing required columns: {sorted(missing)}")

        users_by_id = users.set_index("user_id", drop=False)
        train = train.sort_values(["user_idx", "timestamp"]).copy()
        examples: list[list[int]] = []

        for user_id, user_group in train.groupby("user_id", sort=False):
            rows = user_group.tail(self.config.behavior_last_k)
            prefix = [self.schema.bos]

            if self.config.include_user_features and user_id in users_by_id.index:
                user_row = users_by_id.loc[user_id]
                prefix.extend([self.schema.user_open])
                prefix.extend(
                    self.schema.user_tokens(
                        user_row["gender"], user_row["age"], user_row["occupation"]
                    )
                )
                prefix.extend([self.schema.user_close])

            prefix.append(self.schema.hist)
            events = [self._behavior_event_tokens(row, sids) for row in rows.itertuples()]
            tokens = self._fit_history(prefix, events)
            tokens.append(self.schema.eos)
            examples.append(tokenizer.convert_tokens_to_ids(tokens))

        return examples

    def build_title_examples(
        self, tokenizer: Any, item_meta: pd.DataFrame, sids: np.ndarray
    ) -> list[list[int]]:
        examples = []
        for row in item_meta.sort_values("item_idx").itertuples(index=False):
            sid_text = "".join(self.schema.sid_tokens(sids[int(row.item_idx)]))
            title = str(row.title).rstrip()
            text = (
                f"{self.schema.bos}{self.schema.title_open}"
                f"Movie {sid_text} has title: {title}"
                f"{self.schema.title_close}{self.schema.eos}"
            )
            examples.append(tokenizer.encode(text, add_special_tokens=False))
        return examples

    def build_genre_examples(
        self, tokenizer: Any, item_meta: pd.DataFrame, sids: np.ndarray
    ) -> list[list[int]]:
        examples = []
        for row in item_meta.sort_values("item_idx").itertuples(index=False):
            sid_text = "".join(self.schema.sid_tokens(sids[int(row.item_idx)]))
            genres = ",".join(str(genre) for genre in row.genres)
            text = (
                f"{self.schema.bos}{self.schema.genres_open}"
                f"The genres in movie {sid_text} are:{genres}"
                f"{self.schema.genres_close}{self.schema.eos}"
            )
            examples.append(tokenizer.encode(text, add_special_tokens=False))
        return examples

    def build_year_examples(
        self, tokenizer: Any, item_meta: pd.DataFrame, sids: np.ndarray
    ) -> list[list[int]]:
        examples = []
        for row in item_meta.sort_values("item_idx").itertuples(index=False):
            sid_text = "".join(self.schema.sid_tokens(sids[int(row.item_idx)]))
            text = (
                f"{self.schema.bos}{self.schema.year_open}"
                f"The movie {sid_text} was released in {int(row.years)}"
                f"{self.schema.year_close}{self.schema.eos}"
            )
            examples.append(tokenizer.encode(text, add_special_tokens=False))
        return examples

    def save_artifacts(
        self,
        out_dir: str | Path,
        tokenizer: Any,
        corpora: dict[str, list[list[int]]],
        sids: np.ndarray,
    ) -> None:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required to save CPT corpora in Hugging Face format. "
                "Install it in the notebook environment first."
            ) from exc

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(out_dir / "tokenizer")

        with (out_dir / "token_spec.json").open("w", encoding="utf-8") as f:
            json.dump(
                self.schema.token_spec(self.config.behavior_last_k),
                f,
                ensure_ascii=False,
                indent=2,
            )

        np.save(out_dir / "SIDs.npy", sids)

        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for name, ids in corpora.items():
            Dataset.from_dict({"input_ids": ids}).save_to_disk(data_dir / name)

    def _behavior_event_tokens(self, row: Any, sids: np.ndarray) -> list[str]:
        tokens = [self.schema.event_open]
        tokens.extend(self.schema.sid_tokens(sids[int(row.item_idx)]))
        if self.config.include_ratings:
            tokens.append(self.schema.rating_token(row.rating))
        tokens.append(self.schema.event_close)
        return tokens

    def _fit_history(self, prefix: list[str], events: list[list[str]]) -> list[str]:
        tokens = list(prefix)
        selected = list(events)

        if self.config.truncate_long_histories:
            while selected:
                candidate = tokens + [tok for event in selected for tok in event]
                if len(candidate) + 1 <= self.config.max_length:
                    return candidate
                selected = selected[1:]

        for event in selected:
            tokens.extend(event)
        return tokens
