from __future__ import annotations

import numpy as np

from .metrics import evaluate_rankings
from .protocol import ACTIVE_SID_PROTOCOL
from .sid import ItemSIDMapping, format_sid, parse_sid
from .trie import TokenTrie


def run_smoke_test() -> dict:
    sid = (3, 12, 7, 1)
    tokens = format_sid(sid)
    assert parse_sid(tokens) == sid

    sids = np.array([[3, 12, 7, 1], [3, 12, 7, 1], [4, 1, 2, 3]], dtype=np.int64)
    mapping = ItemSIDMapping.from_sids(sids, protocol=ACTIVE_SID_PROTOCOL)
    assert mapping.sid_candidates_to_items([sid], seen_items={0}, k=10) == [1]

    trie = TokenTrie.from_sequences([[10, 20, 30], [10, 21, 31]], eos_id=99)
    assert trie.next_tokens([]) == [10]
    assert trie.next_tokens([10]) == [20, 21]
    assert trie.next_tokens([10, 20, 30]) == [99]

    metrics = evaluate_rankings(
        [
            {"target_item_idx": 1, "candidates": [0, 1, 2]},
            {"target_item_idx": 3, "candidates": []},
        ],
        k_values=(1, 3),
    )
    assert metrics["recall@3"] == 0.5
    return {"status": "ok", "sid_levels": ACTIVE_SID_PROTOCOL.n_levels, "metrics": metrics}
