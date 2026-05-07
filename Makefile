PYTHON ?= python

.PHONY: install test lint prepare-data embeddings train-sid train-cpt train-sft eval smoke-test config-check artifacts-check

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m compileall -q src tests

config-check:
	$(PYTHON) -m plum_ml1m.cli validate-config --config-dir configs

artifacts-check:
	$(PYTHON) -m plum_ml1m.cli validate-artifacts --manifest configs/artifact_manifest.yaml

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
	$(PYTHON) -m plum_ml1m.cli evaluate --config configs/evaluation.yaml

smoke-test:
	$(PYTHON) -m plum_ml1m.cli smoke-test
