# Reports

This folder stores local project notes, figures, diagnostics, and experiment exports.

Markdown reports are intentionally ignored by git, except `README.md`. This keeps draft reports and supervisor-facing writeups local while allowing the repository-level README to stay clean and stable.

## Layout

- `status/`  
  Project status snapshots and broad experiment summaries.

- `methodology/`  
  Methodological notes and step-by-step protocol drafts.

- `sid/`  
  RQ-VAE, Semantic ID, SID quality, and architecture notes.

- `cpt/`  
  Continued pre-training notes, model inventory, and grounding evaluation artifacts.

- `sft/`  
  Supervised fine-tuning reports, retrieval metrics, and validation figures.

## Tracking Policy

Tracked:

- `reports/README.md`

Ignored:

- `reports/*.md`
- `reports/**/*.md`

Figures, CSV summaries, and JSON metrics are not ignored by the report-specific rule.
