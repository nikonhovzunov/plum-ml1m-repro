import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plum_ml1m.decoding import TrieConstrainedDecoder
from plum_ml1m.eval import EvaluationCase, run_sid_evaluation
from plum_ml1m.sid import ItemSIDMapping
from plum_ml1m.trie import TokenTrie

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiny_eval"


def test_tiny_sid_evaluation_end_to_end():
    assignments = pd.read_csv(FIXTURE_DIR / "sid_assignments.csv")
    sids = assignments[["sid_0", "sid_1", "sid_2", "sid_3"]].to_numpy(dtype=int)
    mapping = ItemSIDMapping.from_sids(np.asarray(sids))

    token_ids_to_sid = {
        (10, 11, 12, 13): (1, 1, 1, 1),
        (20, 21, 22, 23): (2, 2, 2, 2),
        (30, 31, 32, 33): (3, 3, 3, 3),
        (40, 41, 42, 43): (4, 4, 4, 4),
    }
    trie = TokenTrie.from_sequences(token_ids_to_sid.keys(), eos_id=99)
    decoder = TrieConstrainedDecoder(
        trie=trie,
        token_ids_to_sid=token_ids_to_sid,
        sid_mapping=mapping,
    )

    raw_cases = json.loads((FIXTURE_DIR / "generated_sid_sequences.json").read_text())
    cases = [
        EvaluationCase(
            user_id=row["user_id"],
            target_item_idx=row["target_item_idx"],
            generated_token_sequences=row["generated_token_sequences"],
            seen_item_idx=set(row["seen_item_idx"]),
        )
        for row in raw_cases
    ]
    result = run_sid_evaluation(cases, decoder, k_values=(1, 2), top_k=2)
    expected = json.loads((FIXTURE_DIR / "expected_metrics.json").read_text())

    assert result["metrics"]["n"] == expected["n"]
    assert result["metrics"]["recall@1"] == pytest.approx(expected["recall@1"])
    assert result["metrics"]["recall@2"] == pytest.approx(expected["recall@2"])
    assert result["metrics"]["coverage@2"] == expected["coverage@2"]
    assert result["diagnostics"]["invalid_sid_sequences"] == expected["invalid_sid_sequences"]
    assert result["diagnostics"]["seen_items_filtered"] == expected["seen_items_filtered"]
