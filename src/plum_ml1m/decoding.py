from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .sid import ItemSIDMapping
from .trie import TokenTrie


def dedupe_preserve_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def filter_seen_items(candidates: Iterable[int], seen_items: set[int] | None = None) -> list[int]:
    seen_items = {int(x) for x in (seen_items or set())}
    return [int(item) for item in dedupe_preserve_order(candidates) if int(item) not in seen_items]


@dataclass(frozen=True)
class TrieConstrainedDecoder:
    trie: TokenTrie
    token_ids_to_sid: dict[tuple[int, ...], tuple[int, ...]]
    sid_mapping: ItemSIDMapping

    def decode_token_sequence(self, token_ids: Iterable[int]) -> tuple[int, ...] | None:
        ids = tuple(int(x) for x in token_ids if int(x) != self.trie.eos_id)
        return self.token_ids_to_sid.get(ids)

    def recommend_from_token_sequences(
        self,
        token_sequences: Iterable[Iterable[int]],
        k: int = 10,
        seen_items: set[int] | None = None,
        collision_policy: str = "expand",
    ) -> list[int]:
        sid_candidates = [self.decode_token_sequence(sequence) for sequence in token_sequences]
        return self.sid_mapping.sid_candidates_to_items(
            sid_candidates,
            k=k,
            seen_items=seen_items,
            policy=collision_policy,
        )
