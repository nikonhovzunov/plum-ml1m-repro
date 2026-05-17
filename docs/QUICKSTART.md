# Quickstart

This repository is a PLUM-style MovieLens-1M reproduction/adaptation. The light commands below do not train large models and are intended for review, CI, and local sanity checks.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Validate the lightweight project surface

```bash
make test
make lint
make config-check
make artifacts-check-schema
make smoke-test
```

Without `make`, use the direct commands:

```bash
python -m pytest -m "not gpu and not slow"
python -m ruff check src tests scripts
python -m compileall -q src tests scripts
python -m plum_ml1m.cli validate-config --config-dir configs
python -m plum_ml1m.cli validate-artifacts --manifest configs/artifact_manifest.yaml --mode schema
python -m plum_ml1m.cli smoke-test
```

## Inspect executable plans

The pipeline commands validate configs and print an execution plan by default. Heavy execution is opt-in.

```bash
make prepare-data
make embeddings
make train-sid
make train-cpt
make train-sft
make eval-val
make eval-test
```

To execute a heavy notebook-backed step, pass `--execute --notebook ...` explicitly through the CLI. CI never does this.
