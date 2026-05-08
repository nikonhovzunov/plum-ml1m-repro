# Legacy Code

Canonical package namespace:

```text
src/plum_ml1m/
```

Legacy experimental modules are kept under:

```text
src/plum_ml1m/legacy/
```

These modules preserve older CPT, SFT, and RQ-VAE research code. They are not the
canonical source of truth for SID schema, metric formulas, or artifact/config
validation.

## Canonical Sources

- SID protocol: `src/plum_ml1m/protocol.py`
- SID token formatting and item mapping: `src/plum_ml1m/sid.py`
- Trie constraints: `src/plum_ml1m/trie.py`
- Decoding and seen filtering: `src/plum_ml1m/decoding.py`
- Ranking metrics: `src/plum_ml1m/metrics.py`
- Config validation: `src/plum_ml1m/config.py`
- Artifact manifest validation: `src/plum_ml1m/artifacts.py`

## Experimental Scripts

Ad-hoc historical launchers live in:

```text
scripts/experiments/
```

They are kept for research history. Current reproducibility checks should go
through the package CLI and Makefile targets.

Historical SID-v1 notebooks under `notebooks/sid_v1_legacy/` may contain
five-level prototype SIDs. That is archival material only and is not the active
SID-v2 protocol.
