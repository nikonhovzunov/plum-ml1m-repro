from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd

CollisionPolicy = Literal["representative", "expand"]


def normalize_sid(sid: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(code) for code in sid)


@dataclass(frozen=True)
class SIDMapping:
    item_to_sid: dict[int, tuple[int, ...]]
    sid_to_items: dict[tuple[int, ...], list[int]]
    item_popularity: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_sids(
        cls,
        sids: np.ndarray,
        interactions: pd.DataFrame | None = None,
        item_col: str = "item_idx",
    ) -> "SIDMapping":
        sids = np.asarray(sids)
        item_popularity: dict[int, int] = {}
        if interactions is not None and item_col in interactions.columns:
            item_popularity = {
                int(item): int(count)
                for item, count in Counter(interactions[item_col].astype(int)).items()
            }

        item_to_sid: dict[int, tuple[int, ...]] = {}
        sid_to_items: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for item_idx, sid in enumerate(sids):
            sid_tuple = normalize_sid(sid)
            item_to_sid[int(item_idx)] = sid_tuple
            sid_to_items[sid_tuple].append(int(item_idx))

        ranked = {
            sid: sorted(
                items,
                key=lambda item: (-item_popularity.get(item, 0), item),
            )
            for sid, items in sid_to_items.items()
        }
        return cls(
            item_to_sid=item_to_sid,
            sid_to_items=ranked,
            item_popularity=item_popularity,
        )

    @property
    def n_items(self) -> int:
        return len(self.item_to_sid)

    @property
    def n_unique_sids(self) -> int:
        return len(self.sid_to_items)

    @property
    def n_collision_buckets(self) -> int:
        return sum(1 for items in self.sid_to_items.values() if len(items) > 1)

    @property
    def n_collided_items(self) -> int:
        return sum(len(items) for items in self.sid_to_items.values() if len(items) > 1)

    @property
    def n_collision_excess(self) -> int:
        return self.n_items - self.n_unique_sids

    @property
    def uniqueness(self) -> float:
        if not self.item_to_sid:
            return 0.0
        return self.n_unique_sids / self.n_items

    def has_sid(self, sid: Iterable[int]) -> bool:
        return normalize_sid(sid) in self.sid_to_items

    def resolve_sid(
        self,
        sid: Iterable[int],
        policy: CollisionPolicy = "representative",
    ) -> list[int]:
        items = self.sid_to_items.get(normalize_sid(sid), [])
        if policy == "representative":
            return items[:1]
        if policy == "expand":
            return list(items)
        raise ValueError(f"Unknown collision policy: {policy}")

    def sid_candidates_to_items(
        self,
        sid_candidates: Iterable[Iterable[int] | None],
        k: int = 20,
        seen_items: set[int] | None = None,
        policy: CollisionPolicy = "expand",
    ) -> list[int]:
        seen_items = seen_items or set()
        recommendations: list[int] = []
        used: set[int] = set()

        for sid in sid_candidates:
            if sid is None:
                continue
            for item in self.resolve_sid(sid, policy=policy):
                if item in seen_items or item in used:
                    continue
                recommendations.append(item)
                used.add(item)
                if len(recommendations) >= k:
                    return recommendations

        return recommendations
