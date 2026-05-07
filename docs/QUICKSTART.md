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
make artifacts-check
make smoke-test
```

## Inspect executable plans

The pipeline commands validate configs and print an execution plan by default. Heavy execution is opt-in.

```bash
make prepare-data
make embeddings
make train-sid
make train-cpt
make train-sft
make eval
```

To execute a heavy notebook-backed step, pass `--execute --notebook ...` explicitly through the CLI. CI never does this.
