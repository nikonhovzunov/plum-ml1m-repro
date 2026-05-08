# Environment

The install contract is `pyproject.toml` plus `requirements.txt`.

The file below is a historical local environment snapshot only:

```text
docs/environment-snapshot-2026-04-21.txt
```

It is not a lockfile and should not be used as the primary installation source.
It captures one local workspace state and may include unrelated packages.

## Lightweight Development

```bash
python -m pip install -e ".[dev]"
make lint
make test
make config-check
make artifacts-check-schema
make smoke-test
```

These commands are CPU-safe and should not download large model weights.

## Heavy Experiments

Full embedding, RQ-VAE, CPT, SFT, and generation runs require local data and
model artifacts. They are intentionally opt-in and should not run in CI.
