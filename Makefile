PYTHON ?= python

.PHONY: install test lint format-check config-check artifacts-check-schema artifacts-check-local \
	smoke-test prepare-data embeddings train-sid train-cpt train-sft eval eval-val eval-test \
	clean-generated

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -m "not gpu and not slow"

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m compileall -q src tests

format-check:
	$(PYTHON) -m ruff format --check src tests scripts

config-check:
	$(PYTHON) -m plum_ml1m.cli validate-config --config-dir configs

artifacts-check-schema:
	$(PYTHON) -m plum_ml1m.cli validate-artifacts --manifest configs/artifact_manifest.yaml --mode schema

artifacts-check-local:
	$(PYTHON) -m plum_ml1m.cli validate-artifacts --manifest configs/artifact_manifest.yaml --mode local --root .

smoke-test:
	$(PYTHON) -m plum_ml1m.cli smoke-test

prepare-data:
	$(PYTHON) -m plum_ml1m.cli prepare-data --config configs/prepare_data.yaml

embeddings:
	$(PYTHON) -m plum_ml1m.cli build-embeddings --config configs/embeddings.yaml

train-sid:
	$(PYTHON) -m plum_ml1m.cli train-sid --config configs/rqvae_sid.yaml

train-cpt:
	$(PYTHON) -m plum_ml1m.cli train-cpt --config configs/cpt.yaml

train-sft:
	$(PYTHON) -m plum_ml1m.cli train-sft --config configs/sft.yaml

eval:
	@echo "Choose an explicit split: make eval-val or make eval-test"
	@exit 2

eval-val:
	$(PYTHON) -m plum_ml1m.cli evaluate --config configs/evaluation_val.yaml

eval-test:
	$(PYTHON) -m plum_ml1m.cli evaluate --config configs/evaluation_test.yaml

clean-generated:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache')]]"
