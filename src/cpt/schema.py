from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class CPTSchema:
    bos: str = "<bos>"
    eos: str = "<eos>"
    pad: str = "<pad>"
    unk: str = "<unk>"
    user_open: str = "<user>"
    user_close: str = "</user>"
    hist: str = "<hist>"
    event_open: str = "<e>"
    event_close: str = "</e>"
    title_open: str = "<title>"
    title_close: str = "</title>"
    year_open: str = "<year>"
    year_close: str = "</year>"
    genres_open: str = "<genres>"
    genres_close: str = "</genres>"
    rating_prefix: str = "<rat_"
    sid_prefix: str = "<sid_"
    extra_schema_tokens: tuple[str, ...] = field(default_factory=tuple)

    @property
    def schema_tokens(self) -> list[str]:
        return [
            self.user_open,
            self.user_close,
            self.hist,
            self.event_open,
            self.event_close,
            self.title_open,
            self.title_close,
            self.year_open,
            self.year_close,
            self.genres_open,
            self.genres_close,
            *self.extra_schema_tokens,
        ]

    @property
    def special_tokens(self) -> dict[str, str]:
        return {
            "bos_token": self.bos,
            "eos_token": self.eos,
            "pad_token": self.pad,
            "unk_token": self.unk,
        }

    def sid_tokens(self, sid: Iterable[int]) -> list[str]:
        return [f"{self.sid_prefix}{level}_{int(code)}>" for level, code in enumerate(sid)]

    def rating_token(self, rating: int | float) -> str:
        return f"{self.rating_prefix}{int(rating)}>"

    def user_tokens(self, gender: str, age: int, occupation: int) -> list[str]:
        return [f"<gen_{gender}>", f"<age_{int(age)}>", f"<occ_{int(occupation)}>"]

    def token_spec(self, behavior_last_k: int) -> dict:
        return {
            "special_tokens": {
                "bos": self.bos,
                "eos": self.eos,
                "pad": self.pad,
                "unk": self.unk,
            },
            "schema_tokens": self.schema_tokens,
            "notes": {
                "behavior_last_k": behavior_last_k,
                "meta_title_template": "Movie <sid...> has title: {title}",
                "meta_genre_template": "The genres in movie <sid...> are: {genres}",
                "meta_year_template": "The movie <sid...> was released in {year}",
            },
        }
