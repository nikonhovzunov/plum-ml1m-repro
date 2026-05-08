from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .protocol import ACTIVE_SID_PROTOCOL, SIDProtocol

SID_TOKEN_RE = re.compile(r"^<sid_(\d+)_(\d+)>$")
DUP_TOKEN_RE = re.compile(r"^<dup_(\d+)>$")


class SIDTokenError(ValueError):
    """Raised when a SID token cannot be parsed under the active schema."""


def format_sid_token(level: int, code: int) -> str:
    return f"<sid_{int(level)}_{int(code)}>"


def parse_sid_token(token: str) -> tuple[int, int]:
    match = SID_TOKEN_RE.match(str(token))
    if not match:
        raise SIDTokenError(f"Malformed SID token: {token!r}")
    return int(match.group(1)), int(match.group(2))


def format_sid(sid: Iterable[int], protocol: SIDProtocol = ACTIVE_SID_PROTOCOL) -> list[str]:
    values = protocol.validate_sid(tuple(int(x) for x in sid))
    return [format_sid_token(level, code) for level, code in enumerate(values)]


def parse_sid(
    tokens: Iterable[str], protocol: SIDProtocol = ACTIVE_SID_PROTOCOL
) -> tuple[int, ...]:
    values: list[int | None] = [None] * protocol.n_levels
    for token in tokens:
        level, code = parse_sid_token(token)
        if level >= protocol.n_levels:
            raise SIDTokenError(f"SID level {level} is invalid for {protocol.name}")
        if values[level] is not None:
            raise SIDTokenError(f"Duplicate SID level {level} in token sequence")
        values[level] = code
    if any(value is None for value in values):
        missing = [i for i, value in enumerate(values) if value is None]
        raise SIDTokenError(f"Missing SID level(s): {missing}")
    return protocol.validate_sid(tuple(int(value) for value in values))


def try_parse_sid(
    tokens: Iterable[str],
    protocol: SIDProtocol = ACTIVE_SID_PROTOCOL,
) -> tuple[int, ...] | None:
    try:
        return parse_sid(tokens, protocol=protocol)
    except (SIDTokenError, ValueError):
        return None


def is_duplicate_token(token: str) -> bool:
    return DUP_TOKEN_RE.match(str(token)) is not None


@dataclass(frozen=True)
class ItemSIDMapping:
    item_to_sid: dict[int, tuple[int, ...]]
    sid_to_items: dict[tuple[int, ...], list[int]]
    item_popularity: dict[int, int] = field(default_factory=dict)
    protocol: SIDProtocol = ACTIVE_SID_PROTOCOL

    @classmethod
    def from_sids(
        cls,
        sids: np.ndarray,
        interactions: object | None = None,
        item_col: str = "item_idx",
        protocol: SIDProtocol = ACTIVE_SID_PROTOCOL,
    ) -> ItemSIDMapping:
        popularity: dict[int, int] = {}
        if (
            interactions is not None
            and hasattr(interactions, "columns")
            and item_col in interactions.columns
        ):
            popularity = {
                int(item): int(count)
                for item, count in Counter(interactions[item_col].astype(int)).items()
            }

        item_to_sid: dict[int, tuple[int, ...]] = {}
        sid_to_items: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for item_idx, sid in enumerate(np.asarray(sids)):
            sid_tuple = protocol.validate_sid(tuple(int(x) for x in sid))
            item_to_sid[int(item_idx)] = sid_tuple
            sid_to_items[sid_tuple].append(int(item_idx))

        ranked = {
            sid: sorted(items, key=lambda item: (-popularity.get(item, 0), item))
            for sid, items in sid_to_items.items()
        }
        return cls(
            item_to_sid=item_to_sid,
            sid_to_items=ranked,
            item_popularity=popularity,
            protocol=protocol,
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
    def uniqueness(self) -> float:
        return self.n_unique_sids / self.n_items if self.n_items else 0.0

    def has_sid(self, sid: Iterable[int]) -> bool:
        return tuple(int(x) for x in sid) in self.sid_to_items

    def resolve_sid(self, sid: Iterable[int], policy: str | None = None) -> list[int]:
        policy = policy or self.protocol.collision_policy
        items = self.sid_to_items.get(tuple(int(x) for x in sid), [])
        if policy == "expand":
            return list(items)
        if policy == "representative":
            return items[:1]
        raise ValueError(f"Unknown collision policy: {policy}")

    def sid_candidates_to_items(
        self,
        sid_candidates: Iterable[Iterable[int] | None],
        k: int = 10,
        seen_items: set[int] | None = None,
        policy: str | None = None,
    ) -> list[int]:
        seen_items = {int(x) for x in (seen_items or set())}
        recommendations: list[int] = []
        used: set[int] = set()
        for sid in sid_candidates:
            if sid is None:
                continue
            for item in self.resolve_sid(sid, policy=policy):
                item = int(item)
                if item in seen_items or item in used:
                    continue
                recommendations.append(item)
                used.add(item)
                if len(recommendations) >= k:
                    return recommendations
        return recommendations
