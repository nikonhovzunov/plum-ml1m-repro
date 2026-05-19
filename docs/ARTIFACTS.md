# Artifact Contract

The repository does not commit large generated artifacts: raw MovieLens files,
processed datasets, embeddings, checkpoints, LoRA adapters, prediction dumps,
and full generated reports are local outputs. Compact scalar metric snapshots
under `reports/snapshots/` are committed for auditability.

The contract lives in:

```text
configs/artifact_manifest.yaml
```

Each artifact record declares:

- `name`
- `stage`
- `kind`
- `path`
- `required_for`
- `produced_by`
- `consumed_by`
- `safe_to_commit`
- `exists_required_for_ci`
- `expected_type`
- `expected_size_bytes`
- `sha256`
- `schema`
- `notes`

## Validation Modes

Schema-only validation is CI-safe and does not require local data:

```bash
make artifacts-check-schema
```

It validates manifest shape, required fields, artifact kinds, expected type
values, checksum format, and schema-field types. It does not touch local heavy
files.

Local validation checks the actual files available on the current machine:

```bash
make artifacts-check-local
```

Local mode verifies:

- file or directory existence;
- `expected_type` (`file`, `dir`, or `file_or_dir`);
- exact `expected_size_bytes` when declared;
- `sha256` for file artifacts when declared;
- required files inside directory artifacts when `schema.files` is declared;
- CSV/Parquet columns when `schema.columns` is declared.

For CSV files the validator reads only the header. For Parquet files it reads
the schema metadata through `pyarrow`, not the full table. If `pyarrow` is not
available, Parquet column validation fails with an explicit error.

Local validation is expected to fail on a fresh clone until the heavy pipeline is
run or the artifacts are provided by the maintainer. The missing-artifact list
is therefore a local reproduction checklist, not a CI failure target.

## Required Categories

The manifest separates:

- required input artifacts;
- generated output artifacts;
- optional diagnostics.

Generated artifacts are intentionally marked `safe_to_commit: false`.

## Why Large Artifacts Stay Out of Git

The manifest is the contract for reproducing or locating generated files. Git
stores only code, configs, docs, tiny fixtures, and compact scalar snapshots.
Large local files such as embeddings, checkpoints, adapters, processed
datasets, and prediction dumps remain ignored to keep the repository reviewable
and cloneable.
