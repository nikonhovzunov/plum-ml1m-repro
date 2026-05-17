# Reproduction

This repository is a PLUM-style MovieLens-1M reproduction/adaptation. The code
path is designed to make the current protocol inspectable and testable; it does
not claim SOTA. Baseline rows are included only as protocol anchors under the
same MovieLens-1M split, not as a broad recommender benchmark suite.

## Lightweight Checks

```bash
python -m pip install -e ".[dev]"
make lint
make test
make config-check
make artifacts-check-schema
make smoke-test
```

If `make` is unavailable, use:

```bash
python -m ruff check src tests scripts
python -m compileall -q src tests scripts
python -m pytest -m "not gpu and not slow"
python -m plum_ml1m.cli validate-config --config-dir configs
python -m plum_ml1m.cli validate-artifacts --manifest configs/artifact_manifest.yaml --mode schema
python -m plum_ml1m.cli smoke-test
```

These checks validate package imports, SID schema, trie decoding, ranking
metrics, config consistency, artifact manifest schema, and a tiny end-to-end
evaluation fixture.

## Full Pipeline Stages

The heavy pipeline is local-artifact dependent:

```bash
python -m pip install -e ".[dev,modeling,notebooks]"
```

```bash
make prepare-data
make embeddings
make train-sid
make train-cpt
make train-sft
make eval-val
make eval-test
```

Training and evaluation commands are planning/validation wrappers unless called
with explicit execution options in the CLI. Heavy notebooks and experiment
scripts are not run by CI.

## Split Discipline

- validation: model selection only;
- test: one final evaluation after selecting settings;
- test metrics must not be edited into reports unless an actual evaluation was
  run.

Validation and test evaluation configs are separate:

```text
configs/evaluation_val.yaml
configs/evaluation_test.yaml
```
