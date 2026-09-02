# splits/ — the cross-validation partitions as data

`datasetA_folds.csv` (21 263 rows, 3 365 families) and `datasetB_folds.csv` (12 440 rows,
3 063 families) list, for every row of the two datasets, its chemical family and the fold it
falls in under every partition used in the paper:

| column | meaning |
|---|---|
| `row_index` | 0-based row of `data/train.csv` (A) / of `data/datasetB_featurized.npz` (B, = Tc>0 rows of `supercon_stanev.csv` in file order) |
| `family_id` | chemical-family code = constituent-element set, `pandas.factorize`; A: `tc_pipeline.chemical_families(unique_m.csv)`, B: `groups` field of the `.npz` |
| `family_key` | the element set the code stands for (`"|".join(sorted(elements))`) |
| `fold_deterministic` | `GroupKFold(5)` without shuffle — the published protocol (Tables 1, 3, 4; figures) |
| `fold_seed0..4` | `GroupKFold(5, shuffle=True, random_state=s)` — Table 2 v2 (independent grouped partitions) |
| `random_fold_seed0..4` | `KFold(5, shuffle=True, random_state=s)` — the random arm |
| `fold_deterministic_v1_prefilter_codes` | **B only.** `GroupKFold(5)` on the *gapped* family codes that v1's live `datasetB_pipeline.py` produced (factorize before the Tc>0 filter). Same family partition, different fold assignment: 8 937 of 12 440 rows sit in a different fold than under `fold_deterministic` (1 721–1 897 per fold). Kept only so v1 numbers stay reproducible; it is **not** the reference. |

## Why indices and not just the procedure

`GroupKFold` assigns whole families to folds greedily by family size and breaks ties between
equally sized families **by the order of the integer codes**. The codes come from
`pandas.factorize`, i.e. from the order of first appearance in the data file. Two runs that
agree on which rows share a family but number the families differently get different folds.
The Dataset-B v1 bug is exactly this (see `results_v2/fix_factorize_verificacion.json`), so
the partition is versioned here as data. scikit-learn 1.7.2, pandas
2.3.2, numpy 2.3.3; the CSV header lines (`#`) repeat this.

## Commands

```bash
python code/v2/make_splits.py            # regenerate both files (≈5 s; needs data/train.csv, unique_m.csv)
python code/v2/make_splits.py --verify   # regenerate in memory and assert identity with the files
```

Read with `pandas.read_csv("splits/datasetB_folds.csv", comment="#")`.
