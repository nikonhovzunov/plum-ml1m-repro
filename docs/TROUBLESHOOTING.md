# Troubleshooting

## Config validation fails

Run:

```bash
plum-ml1m validate-config --config-dir configs
```

Every config must include `name`, `stage`, and `paths`.

## Smoke test fails

Run:

```bash
python -m pytest tests/test_cli_smoke.py -q
```

The smoke test exercises only lightweight SID, trie, mapping, and metric logic.

## Heavy artifacts are missing

This is expected on a fresh clone. The repository does not track raw data, processed datasets, embeddings, or model checkpoints. See `docs/ARTIFACTS.md`.

## Notebook and CLI results differ

Treat `src/plum_ml1m` and `configs/` as the source of truth for protocol definitions. If a notebook changes SID schema, split logic, decoding, or metrics, document the change before reporting new metrics.
