# `exploratory/` — pre-paper EDA on the raw SuperCon dump

Exploratory phase that preceded the leakage paper; kept for provenance, not used by any number in
the manuscript. Brought in from the `referee-response` branch (2026-08-28), which was the first
time this material was placed under version control.

- `notebooks/EDA.ipynb` — deduplication and quality checks on the raw SuperCon dump, RF/XGB
  feature importances, and a **temporal validation** by publication-year cut-off (1995/2005/2015)
  that does not appear in the paper (`notebooks/model_summary.json`).
- `notebooks/supercon_primary*.parquet` — the dump after parsing / deduplication / quality filter.
- `notebooks/*.tsv`, `quality/*.csv` — duplicate samples, outlier context, suspect rows, quality
  report, feature-importance and feature-family tables (the `quality/` copies differ from the
  `notebooks/` ones and are the later versions).
- `supercon_report_pdf.py` — generic tabular-report generator used on those files.
- `pyproject.toml` — the project-level environment of that phase (`uv`), independent of
  `requirements.txt`.

**Not included:** `data/primary.tsv`, the raw SuperCon (NIMS) dump the EDA starts from. It is
kept out of the public repository pending a check of its redistribution terms; the curated
`data/supercon_stanev.csv` used by the paper is the released list of Stanev et al. and is included.
