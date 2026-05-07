# Evaluation

Evaluation is done in the original item-id space after SID decoding.

## Candidate generation

1. Build a prompt from the user's history window.
2. Generate SID token sequences with beam search.
3. Restrict generation with a trie containing valid item SID sequences.
4. Parse generated SIDs.
5. Map each SID to one or more item indices.
6. Expand SID collisions according to the active `expand` policy.
7. Remove already seen items.
8. Keep the first `K` unique item recommendations.

## Metrics

For a user with target item `y` and recommendation list `R`:

```text
Recall@K = 1[y in R[:K]]
MRR@K    = 1 / rank(y) if y appears in R[:K], else 0
NDCG@K   = 1 / log2(rank(y) + 1) if y appears in R[:K], else 0
Coverage@K = number of unique recommended items across all users
```

Metrics are averaged over users for Recall, MRR, and NDCG. Coverage is reported as a count of unique recommended items, not a percentage.

## Invalid SIDs

The active comparison tables omit invalid SID rate because trie-constrained decoding rules out invalid item SID sequences by construction.
