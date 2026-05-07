# Artifacts

The tracked repository contains code, configs, notebooks, and documentation. Heavy generated artifacts are intentionally ignored by git.

The artifact requirements are listed in:

```text
configs/artifact_manifest.yaml
```

Required artifact classes:

- raw MovieLens data;
- enriched metadata;
- movie overviews;
- embeddings;
- SID checkpoint;
- SID assignment table;
- CPT LoRA adapter;
- SFT LoRA adapter;
- predictions;
- final metrics report.

Use:

```bash
plum-ml1m validate-artifacts --manifest configs/artifact_manifest.yaml
```

This validates the manifest schema and required artifact types. It does not require the large local files to exist.
