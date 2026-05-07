# Known Limitations

- This repository is a PLUM-style reproduction/adaptation, not a SOTA claim.
- Public sequential recommendation baselines are intentionally not implemented here.
- Reported metrics are preserved from existing experiment outputs unless evaluation is explicitly rerun.
- Heavy training still lives primarily in notebooks and local artifact directories; the new CLI provides validated entry points and safe execution plans.
- Full reproduction requires MovieLens-1M and local model/artifact generation.
- External movie overview enrichment may depend on source availability and manual audit quality.
