# Publishing this repository

This folder is a complete, self-contained research repository. To publish it:

## Option A — GitHub (recommended, free, all venues accept it)

```bash
cd superconductor-tc-leakage-eval
git init
git add -A
git commit -m "Initial release: leakage-aware Tc evaluation"

# create an EMPTY repo named 'superconductor-tc-leakage-eval' on github.com first
# (github.com/new — do NOT add a README/license, this repo has them), then:
git branch -M main
git remote add origin https://github.com/Freakscode/superconductor-tc-leakage-eval.git
git push -u origin main
```

Your paper's Data-availability URL then becomes:
    https://github.com/Freakscode/superconductor-tc-leakage-eval

## Option B — Zenodo (adds a citable DOI; do this too, before submission)

1. Push to GitHub as above.
2. Go to zenodo.org → log in with GitHub → "GitHub" tab → flip the switch ON
   for `superconductor-tc-leakage-eval`.
3. On GitHub, create a release (tag `v1.0`). Zenodo mints a DOI automatically.
4. Put the DOI badge in README and cite the DOI in the paper.

A DOI is what makes the archive permanent and citable — reviewers like to see it.

## Note on data
- `data/datasetB_featurized.npz` (Dataset B, 2.6 MB) is included — Dataset B is
  fully reproducible offline from it.
- Dataset A raw files are NOT committed (they're ~28 MB and belong to UCI).
  `code/get_datasetA.py` downloads them from the canonical UCI archive.
