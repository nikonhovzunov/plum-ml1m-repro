from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

TrieNode = dict[int, "TrieNode"]


@dataclass
class TokenTrie:
    eos_id: int
    root: TrieNode = field(default_factory=dict)

    @classmethod
    def from_sequences(cls, sequences: Iterable[Iterable[int]], eos_id: int) -> TokenTrie:
        trie = cls(eos_id=int(eos_id))
        for sequence in sequences:
            trie.insert(sequence)
        return trie

    def insert(self, sequence: Iterable[int]) -> None:
        node = self.root
        for token_id in sequence:
            node = node.setdefault(int(token_id), {})
        node.setdefault(self.eos_id, {})

    def contains(self, sequence: Iterable[int]) -> bool:
        node = self.root
        for token_id in sequence:
            token_id = int(token_id)
            if token_id not in node:
                return False
            node = node[token_id]
        return self.eos_id in node or not node

    def next_tokens(self, prefix: Iterable[int]) -> list[int]:
        node = self.root
        for token_id in prefix:
            token_id = int(token_id)
            if token_id not in node:
                return [self.eos_id]
            node = node[token_id]
        return sorted(node.keys()) if node else [self.eos_id]

    def prefix_allowed_tokens_fn(self, prompt_length: int):
        def allowed_tokens(_batch_id: int, input_ids) -> list[int]:
            generated = input_ids[int(prompt_length) :]
            if hasattr(generated, "tolist"):
                generated = generated.tolist()
            return self.next_tokens(int(token_id) for token_id in generated)

        return allowed_tokens
