from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import SFTSchema


@dataclass(frozen=True)
class SFTDataPaths:
    processed_dir: Path = Path("data/processed")
    raw_ml1m_dir: Path = Path("data/raw/ml-1m")

    @property
    def train_path(self) -> Path:
        return self.processed_dir / "splits" / "train.parquet"

    @property
    def val_path(self) -> Path:
        return self.processed_dir / "splits" / "val.parquet"

    @property
    def test_path(self) -> Path:
        return self.processed_dir / "splits" / "test.parquet"

    @property
    def users_path(self) -> Path:
        return self.raw_ml1m_dir / "users.dat"


@dataclass(frozen=True)
class SFTExampleConfig:
    max_history_events: int = 100
    min_history_events: int = 3
    max_length: int = 1024
    include_user_features: bool = True
    include_ratings: bool = True
    train_examples_per_user: int | None = None
    max_users: int | None = None


class SFTDatasetBuilder:
    def __init__(
        self,
        paths: SFTDataPaths | None = None,
        schema: SFTSchema | None = None,
        config: SFTExampleConfig | None = None,
    ):
        self.paths = paths or SFTDataPaths()
        self.schema = schema or SFTSchema()
        self.config = config or SFTExampleConfig()

    def load_tables(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = pd.read_parquet(self.paths.train_path)
        val = pd.read_parquet(self.paths.val_path)
        test = pd.read_parquet(self.paths.test_path)
        users = pd.read_csv(
            self.paths.users_path,
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip"],
            encoding="latin-1",
        )
        return train, val, test, users

    def build_train_examples(
        self,
        sids: np.ndarray,
        tokenizer: Any,
        train: pd.DataFrame | None = None,
        users: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        if train is None or users is None:
            train, _, _, users = self.load_tables()
        self._validate_inputs(train, sids)

        users_by_id = users.set_index("user_id", drop=False)
        examples: list[dict[str, Any]] = []

        for user_idx, (user_id, group) in enumerate(train.groupby("user_id", sort=False)):
            if self.config.max_users is not None and user_idx >= self.config.max_users:
                break

            events = self._rows_to_events(group, sids)
            positions = range(self.config.min_history_events, len(events))
            if self.config.train_examples_per_user is not None:
                positions = list(positions)[-self.config.train_examples_per_user :]

            user_row = self._get_user_row(users_by_id, user_id)
            for target_pos in positions:
                history = events[:target_pos]
                target = events[target_pos]
                examples.append(
                    self._encode_example(
                        tokenizer=tokenizer,
                        user_id=int(user_id),
                        user_idx=int(target["user_idx"]),
                        split="train",
                        user_row=user_row,
                        history_events=history,
                        target_event=target,
                    )
                )

        return examples

    def build_eval_examples(
        self,
        split: str,
        sids: np.ndarray,
        tokenizer: Any,
        train: pd.DataFrame | None = None,
        val: pd.DataFrame | None = None,
        test: pd.DataFrame | None = None,
        users: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        if split not in {"val", "test"}:
            raise ValueError("split must be 'val' or 'test'")
        if train is None or val is None or test is None or users is None:
            train, val, test, users = self.load_tables()

        self._validate_inputs(train, sids)
        target_df = val if split == "val" else test
        context_df = train if split == "val" else pd.concat([train, val], ignore_index=True)

        users_by_id = users.set_index("user_id", drop=False)
        context_by_user = {
            user_id: self._rows_to_events(group, sids)
            for user_id, group in context_df.groupby("user_id", sort=False)
        }

        examples: list[dict[str, Any]] = []
        for user_idx, row in enumerate(target_df.sort_values(["user_idx", "timestamp"]).itertuples()):
            if self.config.max_users is not None and user_idx >= self.config.max_users:
                break
            target = self._row_to_event(row, sids)
            history = context_by_user.get(row.user_id, [])
            user_row = self._get_user_row(users_by_id, row.user_id)
            examples.append(
                self._encode_example(
                    tokenizer=tokenizer,
                    user_id=int(row.user_id),
                    user_idx=int(row.user_idx),
                    split=split,
                    user_row=user_row,
                    history_events=history,
                    target_event=target,
                )
            )

        return examples

    def _encode_example(
        self,
        tokenizer: Any,
        user_id: int,
        user_idx: int,
        split: str,
        user_row: Any | None,
        history_events: list[dict[str, Any]],
        target_event: dict[str, Any],
    ) -> dict[str, Any]:
        history_events = history_events[-self.config.max_history_events :]
        prefix = self.schema.prompt_prefix_tokens(
            user_row=user_row,
            include_user_features=self.config.include_user_features,
        )
        event_blocks = [
            self.schema.event_tokens(
                event["sid"],
                rating=event.get("rating"),
                include_rating=self.config.include_ratings,
            )
            for event in history_events
        ]
        target_tokens = self.schema.target_tokens(target_event["sid"])

        prompt_tokens = self._fit_prompt(prefix, event_blocks, target_tokens)
        input_tokens = prompt_tokens + target_tokens
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        labels = [-100] * len(prompt_tokens) + input_ids[len(prompt_tokens) :]

        if tokenizer.unk_token_id is not None:
            unknown = [
                token
                for token, token_id in zip(input_tokens, input_ids)
                if token_id == tokenizer.unk_token_id
            ]
            if unknown:
                raise ValueError(f"Tokenizer does not know tokens: {unknown[:10]}")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "user_id": user_id,
            "user_idx": user_idx,
            "split": split,
            "history_item_idx": [int(event["item_idx"]) for event in history_events],
            "target_item_idx": int(target_event["item_idx"]),
            "target_sid": [int(code) for code in target_event["sid"]],
            "prompt_length": len(prompt_tokens),
        }

    def _fit_prompt(
        self,
        prefix: list[str],
        event_blocks: list[list[str]],
        target_tokens: list[str],
    ) -> list[str]:
        selected = list(event_blocks)
        while selected:
            prompt = prefix + [token for event in selected for token in event]
            if len(prompt) + len(target_tokens) <= self.config.max_length:
                return prompt
            selected = selected[1:]

        prompt = list(prefix)
        if len(prompt) + len(target_tokens) > self.config.max_length:
            raise ValueError("Prompt prefix plus target exceeds max_length")
        return prompt

    def _rows_to_events(self, rows: pd.DataFrame, sids: np.ndarray) -> list[dict[str, Any]]:
        sort_cols = [col for col in ["timestamp", "pos", "item_idx"] if col in rows.columns]
        rows = rows.sort_values(sort_cols, kind="mergesort")
        return [self._row_to_event(row, sids) for row in rows.itertuples()]

    def _row_to_event(self, row: Any, sids: np.ndarray) -> dict[str, Any]:
        item_idx = int(row.item_idx)
        return {
            "user_id": int(row.user_id),
            "user_idx": int(row.user_idx),
            "item_idx": item_idx,
            "rating": int(row.rating),
            "timestamp": int(row.timestamp),
            "sid": [int(code) for code in sids[item_idx]],
        }

    def _get_user_row(self, users_by_id: pd.DataFrame, user_id: int) -> Any | None:
        if user_id not in users_by_id.index:
            return None
        return users_by_id.loc[user_id]

    def _validate_inputs(self, interactions: pd.DataFrame, sids: np.ndarray) -> None:
        required = {"user_id", "user_idx", "item_idx", "rating", "timestamp"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"Interactions are missing columns: {sorted(missing)}")
        if np.asarray(sids).ndim != 2:
            raise ValueError(f"sids must be a 2D array, got shape {np.asarray(sids).shape}")
        max_item_idx = int(interactions["item_idx"].max())
        if len(sids) <= max_item_idx:
            raise ValueError(f"sids has {len(sids)} rows, needs item_idx {max_item_idx}")


def examples_to_dataset(examples: Iterable[dict[str, Any]]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("datasets is required to convert SFT examples to Dataset") from exc
    return Dataset.from_list(list(examples))
