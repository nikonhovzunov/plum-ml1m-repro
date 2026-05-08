# Maintenance Audit

Current audit focus:

- one canonical package namespace: `plum_ml1m`;
- no root-level `src` package;
- active SID-v2 protocol has four levels and codebook sizes
  `[1024, 512, 256, 128]`;
- old CPT/SFT/SID research modules are explicit legacy modules;
- validation and test evaluation configs are separate;
- artifact manifest has schema-only and local validation modes;
- tiny CPU-only evaluation fixture covers SID decoding, collision expansion,
  seen filtering, and metric aggregation.

Known intentional non-goals:

- no external recommender baselines are added;
- no SOTA claim is made;
- large generated artifacts are not committed;
- root `README.md` is frozen for this maintenance pass.
