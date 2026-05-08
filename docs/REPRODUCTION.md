# Reproduction

This repository is a PLUM-style MovieLens-1M reproduction/adaptation. The code
path is designed to make the current protocol inspectable and testable; it does
not claim SOTA and does not include external recommender baselines.

## Lightweight Checks

```bash
python -m pip install -e ".[dev]"
make lint
make test
make config-check
make artifacts-check-schema
make smoke-test
```

These checks validate package imports, SID schema, trie decoding, ranking
metrics, config consistency, artifact manifest schema, and a tiny end-to-end
evaluation fixture.

## Full Pipeline Stages

The heavy pipeline is local-artifact dependent:

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
