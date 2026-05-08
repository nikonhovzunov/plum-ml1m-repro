# Artifact Contract

The repository does not commit large generated artifacts: raw MovieLens files,
processed datasets, embeddings, checkpoints, LoRA adapters, prediction dumps,
and generated metric reports are local outputs.

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

Local validation checks whether artifacts exist and whether checksums match when
they are declared:

```bash
make artifacts-check-local
```

Local validation is expected to fail on a fresh clone until the heavy pipeline is
run or the artifacts are provided by the maintainer.

## Required Categories

The manifest separates:

- required input artifacts;
- generated output artifacts;
- optional diagnostics.

Generated artifacts are intentionally marked `safe_to_commit: false`.
