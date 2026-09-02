# Setting up and publishing this repository

## 1. Environment (exact versions)

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # == pins: numpy 2.3.3, pandas 2.3.2, scipy 1.16.2, scikit-learn 1.7.2,
                                       #          xgboost 3.0.5, matminer 0.9.2, pymatgen 2025.6.14, matplotlib 3.10.6
python code/get_datasetA.py            # Dataset A raw files (~28 MB, not committed) -> data/
python code/v2/verify_checksums.py     # all four data files must report OK
```

## 2. Repository root

Every script in `code/v2/` locates the repository as `Path(__file__).resolve().parents[2]`.
If you run from a copy or a symlinked layout, set the override explicitly:

```bash
export SUPERCON_ROOT=/absolute/path/to/superconductor-tc-leakage-eval
```

`code/v2/*.py --dry-run` prints the resolved `ROOT`, `DATA`, `OUT` and flags missing inputs.

## 3. Verification commands

| what | command |
|---|---|
| data checksums | `python code/v2/verify_checksums.py` |
| fold indices | `python code/v2/make_splits.py --verify` |
| figures | `python code/v2/make_figures.py` (writes `figures/figure1..4.png`; add `--submit DIR` to copy them to the manuscript folder) |
| manuscript numbers | `python code/v2/verify_manuscript.py --tex /path/to/paper1_manuscript.tex` |

## 4. Publishing

```bash
cd superconductor-tc-leakage-eval
git add -A && git commit -m "v2: code/v2, factorize fix, pinned requirements, checksums, fold indices"
git push origin main
git tag v2.0 && git push origin v2.0      # Zenodo mints the DOI from the release
```

Zenodo: zenodo.org → GitHub tab → switch ON `superconductor-tc-leakage-eval` → create the
GitHub release `v2.0` → put the DOI in README and in the paper's Data availability.

## 5. Note on data
- `data/datasetB_featurized.npz` (Dataset B, 2.6 MB) is included; Dataset B is fully
  reproducible offline from it. Its `groups` field is the reference family coding (contiguous
  codes, factorize after the Tc>0 filter — see "Changes in v2" in README).
- Dataset A raw files are NOT committed (~28 MB, UCI); `code/get_datasetA.py` downloads them
  and `data/CHECKSUMS.sha256` fixes the expected hashes.
- `results_v2/datasetB_featurized_live.npz` (8 MB) is not committed; it equals the shipped
  `.npz` at float32 precision (see README) and `code/v2/_capture_live_featurization.py`
  recreates it.
