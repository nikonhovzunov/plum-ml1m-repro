"""Legacy SFT metric imports.

Not part of the canonical reproduction path. Kept so older notebooks and scripts
can still import the names while using the authoritative metric implementation.
"""

from __future__ import annotations

from plum_ml1m.metrics import evaluate_rankings, mrr_at_k, ndcg_at_k, recall_at_k

__all__ = ["evaluate_rankings", "mrr_at_k", "ndcg_at_k", "recall_at_k"]
