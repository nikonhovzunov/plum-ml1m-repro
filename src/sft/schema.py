from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.cpt import CPTSchema


@dataclass(frozen=True)
class SFTSchema:
    cpt: CPTSchema = field(default_factory=CPTSchema)
    n_sid_levels: int = 5

    def user_feature_tokens(self, user_row: Any | None) -> list[str]:
        if user_row is None:
            return []
        return self.cpt.user_tokens(
            gender=user_row["gender"],
            age=int(user_row["age"]),
            occupation=int(user_row["occupation"]),
        )

    def event_tokens(
        self,
        sid: Iterable[int],
        rating: int | float | None = None,
        include_rating: bool = True,
    ) -> list[str]:
        tokens = [self.cpt.event_open]
        tokens.extend(self.cpt.sid_tokens(sid))
        if include_rating and rating is not None:
            tokens.append(self.cpt.rating_token(rating))
        tokens.append(self.cpt.event_close)
        return tokens

    def prompt_prefix_tokens(
        self,
        user_row: Any | None = None,
        include_user_features: bool = True,
    ) -> list[str]:
        tokens = [self.cpt.bos]
        if include_user_features and user_row is not None:
            tokens.extend([self.cpt.user_open])
            tokens.extend(self.user_feature_tokens(user_row))
            tokens.extend([self.cpt.user_close])
        tokens.append(self.cpt.hist)
        return tokens

    def target_tokens(self, sid: Iterable[int]) -> list[str]:
        sid_tokens = self.cpt.sid_tokens(sid)
        if len(sid_tokens) != self.n_sid_levels:
            raise ValueError(
                f"Expected {self.n_sid_levels} SID levels, got {len(sid_tokens)}"
            )
        return [*sid_tokens, self.cpt.eos]
