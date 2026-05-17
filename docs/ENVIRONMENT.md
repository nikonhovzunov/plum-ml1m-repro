# Environment

The install contract is `pyproject.toml` plus `requirements.txt`.

Historical local environment dumps are not kept as install contracts. If one is
created for debugging, keep it outside the repository or clearly mark it as a
non-reproducible workspace snapshot.

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

```bash
python -m pip install -e ".[dev,modeling,notebooks]"
```

`ipywidgets` is intentionally not part of the default notebook extra because it
can trigger long-path installation failures on some Windows setups. Install it
manually only if interactive notebook widgets are needed.
