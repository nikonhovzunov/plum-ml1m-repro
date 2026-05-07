import pytest

from plum_ml1m.metrics import coverage_at_k, evaluate_rankings, mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_ndcg_mrr_formulas():
    candidates = [5, 7, 9]
    assert recall_at_k(candidates, target=7, k=1) == 0.0
    assert recall_at_k(candidates, target=7, k=2) == 1.0
    assert ndcg_at_k(candidates, target=7, k=3) == pytest.approx(1.0 / 1.5849625007)
    assert mrr_at_k(candidates, target=7, k=3) == 0.5


def test_duplicate_and_empty_predictions():
    assert recall_at_k([1, 1, 2], target=2, k=2) == 1.0
    assert recall_at_k([], target=2, k=10) == 0.0
    assert ndcg_at_k([], target=2, k=10) == 0.0
    assert mrr_at_k([], target=2, k=10) == 0.0


def test_evaluate_rankings_and_coverage():
    records = [
        {"target_item_idx": 2, "candidates": [1, 2, 3]},
        {"target_item_idx": 4, "candidates": [4, 2]},
        {"target_item_idx": 9, "candidates": []},
    ]
    metrics = evaluate_rankings(records, k_values=(1, 2, 3))
    assert metrics["n"] == 3
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@2"] == pytest.approx(2 / 3)
    assert metrics["coverage@3"] == 4
    assert coverage_at_k(records, 2) == 3
