# Leakage-aware evaluation of composition-based superconductor $T_c$ models

**Repository:** https://github.com/Freakscode/superconductor-tc-leakage-eval

Code, versioned cross-validation partitions, featurized data, measurement tables and the
figure-generation script for:

> **Random splits inflate the accuracy of composition-based superconductor $T_c$
> models: a leakage-aware evaluation replicated across two datasets**
> Gabriel Jaime Cardona Osorio, Institución Universitaria Pascual Bravo, Medellín, Colombia.
> Submitted to *Machine Learning: Science and Technology* (IOP Publishing). **Manuscript v2.**

---

## The result in one paragraph

Composition-based ML models for the superconducting critical temperature are routinely
evaluated with **random** train/test splits. Because these datasets are built from **doping
series** — families of near-identical compounds sharing the same constituent elements and
differing only in stoichiometry — a random split scatters family members across both
partitions and scores the model on chemical near-twins of its own training data. Enforcing a
**chemical-family-aware** split (GroupKFold on the set of constituent elements) raises the MAE
of XGBoost and random forest by **52–61 %** (5-seed means over independent grouped
partitions; e.g. XGBoost 5.25→8.24 K on Dataset A, 3.87→6.23 K on Dataset B) and lowers
$R^2$ by **0.071–0.081**, replicated across two datasets and two model families. A
**featureless family-mean null model** reaches $R^2$ = 0.80 (A) / 0.76 (B) under random
splitting but collapses to $R^2\approx0$ under family-aware splitting — most of the headline
accuracy is memorization of family structure (ICC(1) = 0.85 in both datasets). The models
nonetheless retain **87 % (A) / 85 % (B) top-100 screening precision** under the family-aware
split (the null ranker falls to the ~10 % base rate), so they remain useful ranking tools.

## Repository layout

```
superconductor-tc-leakage-eval/
├── code/
│   ├── tc_pipeline.py            # Dataset A (Hamidieh/UCI): models, chemical_families(), CV helpers
│   ├── datasetB_pipeline.py      # Dataset B (Stanev/SuperCon): Magpie featurization + CV  (v2: factorize fix)
│   ├── null_model_analysis.py    # featureless null model, top-100 comparator, seed intervals (v1)
│   ├── get_datasetA.py           # downloads the UCI raw data (not redistributed here)
│   └── v2/                       # every script behind the v2 measurements, tables and figures
│       ├── run_v2_measurements.py      # nulo_presencia.csv, design_effect.csv, seed_runs_top100.csv, alignment check
│       ├── run_tabla2_shuffle.py       # 60 CV runs (3 schemes x 2 datasets x 2 models x 5 seeds) -> tabla2_runs_por_semilla.csv
│       ├── agg_tabla2.py               # aggregates the 60 runs into tabla2_shuffle.csv & co.
│       ├── build_comparacion.py        # tabla2_comparacion.md (v2 vs v1 Table 2)
│       ├── run_diagnostics_v2.py       # nn_distance.csv, family_concentration.csv, reproduction_check.csv (~10 s)
│       ├── make_figures.py             # regenerates figures/figure1..4.png from the CSVs (+ fidelity asserts)
│       ├── verify_manuscript.py        # checks every number in the .tex against its CSV
│       ├── make_splits.py              # writes / verifies splits/*.csv
│       ├── verify_checksums.py         # checks data/ against data/CHECKSUMS.sha256
│       ├── _top100_harness.py          # the single top-100 precision definition (library)
│       └── _capture_live_featurization.py  # dumps a live matminer featurization for cross-checks
├── data/
│   ├── CHECKSUMS.sha256          # SHA-256 of the four data files + provenance notes
│   ├── supercon_stanev.csv       # public SuperCon formula+Tc list (Dataset B source, 16 414 rows)
│   └── datasetB_featurized.npz   # Dataset B: X (12 440 x 132 Magpie, float32), y, family codes `groups`
│       (train.csv / unique_m.csv for Dataset A are fetched by code/get_datasetA.py; hashes in CHECKSUMS.sha256)
├── splits/
│   ├── datasetA_folds.csv        # row -> family, deterministic fold, GroupKFold shuffle seeds 0-4, KFold seeds 0-4
│   ├── datasetB_folds.csv        # idem (+ the fold assignment v1's live script actually used)
│   └── README.md                 # why the indices are published as data
├── results/                      # v1 tables (kept for the record; Dataset-B rows: see "Changes in v2")
├── results_v2/                   # v2 measurements: every CSV/JSON carries a `procedencia` column
│   ├── nulo_presencia.csv                # Table 1: full features / element-presence null / family-mean null
│   ├── tabla2_shuffle.csv                # Table 2: 5-seed means, independent grouped partitions
│   ├── tabla2_runs_por_semilla.csv       # the 60 individual CV runs behind Table 2
│   ├── tabla2_noshuffle_control.csv      # same, GroupKFold without shuffle (published protocol)
│   ├── design_effect.csv                 # Table 3: ICC(1), eta2, design effect, effective N
│   ├── top100_unificado.csv              # Table 4: top-100 screening precision, canonical threshold
│   ├── nn_distance.csv, family_concentration.csv, reproduction_check.csv   # §3 diagnostics
│   ├── verificacion_alineacion_B.json, fix_factorize_verificacion.json     # Dataset-B alignment / factorize fix evidence
│   ├── fig3d_nn_source_verificacion.json # why Fig 3(d) Dataset B reads 11x, not the 8.9x of the 2026-08-21 draft PNG
│   ├── tabla2_comparacion.md, efecto_vs_ruido.md, figuras_captions.md, P1_diagnostico.md
│   └── ... (32 files)
├── figures/
│   ├── figure1.png … figure4.png # regenerated by code/v2/make_figures.py (figure4_ceiling.png = copy of figure4.png)
│   └── v1/                       # the three v1 PNGs, kept for the record
├── requirements.txt              # exact pins (==) of the environment used for every v2 number
└── LICENSE  (MIT)
```

## Environment

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5, matminer 0.9.2, ...
python code/get_datasetA.py            # Dataset A raw files (~28 MB) -> data/train.csv, data/unique_m.csv
```

All scripts under `code/v2/` resolve the repository root from their own location
(`Path(__file__).resolve().parents[2]`); set **`SUPERCON_ROOT=/path/to/repo`** to override
(e.g. when running from a copy). They can be run from any working directory and each accepts
`--help` and `--dry-run` (resolve paths, check inputs, compute nothing).

## Verification commands

```bash
# (i)   data integrity — the four data files against data/CHECKSUMS.sha256
python code/v2/verify_checksums.py                    # or: cd data && shasum -a 256 -c CHECKSUMS.sha256

# (ii)  cross-validation partitions — regenerate in memory and compare with splits/*.csv
python code/v2/make_splits.py --verify                # ~5 s

# (iii) figures — rebuild figures/figure1..4.png from results_v2/*.csv and data/ (fidelity asserts included)
python code/v2/make_figures.py                        # ~10 s-2 min (refits XGBoost, GroupKFold(5), Dataset A); also drops PNG copies in results_v2/
python code/v2/make_figures.py --submit /path/to/manuscript_dir    # also copy the PNGs there

# (iv)  manuscript numbers — every figure quoted in the .tex against its CSV of provenance
python code/v2/verify_manuscript.py --tex /path/to/paper1_manuscript.tex
#       (default: ../paper1_submission/paper1_manuscript.tex relative to the repository, if present)
```

## Regenerating the measurements

```bash
python code/v2/run_diagnostics_v2.py                  # ~10 s
python code/v2/run_v2_measurements.py                 # tens of minutes (matminer re-featurization + CV, both datasets)
NW=10 python code/v2/run_tabla2_shuffle.py --paso1    # 60 CV runs, ~45 min on 10 threads
python code/v2/agg_tabla2.py && python code/v2/build_comparacion.py
```
`top100_unificado.csv` and `top100_factorial.csv` were produced by short driver cells that
call `code/v2/_top100_harness.py` (the harness is versioned; the drivers were interactive —
their `procedencia` column names the exact function, seed and hyperparameter source).

### Key headline numbers

Table 2 protocol: 5 seeds, `KFold(5, shuffle=True, random_state=s)` vs `GroupKFold(5, shuffle=True,
random_state=s)`, model seed = partition seed; mean ± std (ddof=0). Source: `results_v2/tabla2_shuffle.csv`.

| Dataset | Model | Random MAE (K) | Family MAE (K) | MAE inflation | Random $R^2$ | Family $R^2$ |
|---|---|---|---|---|---|---|
| A | XGBoost | 5.25 ± 0.01 | 8.24 ± 0.08 | +56.9 ± 1.6 % | 0.928 | 0.856 |
| A | Random Forest | 5.41 ± 0.01 | 8.62 ± 0.07 | +59.5 ± 1.4 % | 0.924 | 0.848 |
| B | XGBoost | 3.87 ± 0.02 | 6.23 ± 0.05 | +60.8 ± 0.5 % | 0.930 | 0.849 |
| B | Random Forest | 4.32 ± 0.01 | 6.58 ± 0.02 | +52.5 ± 0.6 % | 0.917 | 0.835 |

Null models (deterministic `GroupKFold(5)`, seed 42; `results_v2/nulo_presencia.csv`):
family-mean null $R^2$ = 0.801 (A) / 0.764 (B) under random CV, −0.007 / −0.003 under family
CV. Top-100 screening precision under family CV (`results_v2/top100_unificado.csv`):
87.0 ± 11.0 % (A), 85.2 ± 5.9 % (B); 98.4 % / 96.0 % under random CV.

## Changes in v2

- **Bug fix — `code/datasetB_pipeline.py`, order of `factorize` and the Tc>0 filter.** v1
  factorized the chemical-family key over all 16 406 featurized rows (4 191 families) and
  applied the Tc>0 mask afterwards, leaving the 3 063 surviving families with non-contiguous
  codes (0…4189). `GroupKFold` breaks ties between equally sized families by code order, so
  those codes yield a *different* fold assignment than the contiguous codes stored in
  `data/datasetB_featurized.npz` — 8 937 of 12 440 rows (1 721–1 897 per fold) land in a
  different fold, although the family partition itself is identical. **The Dataset-B numbers
  of v1 that were produced by running `datasetB_pipeline.py` live were therefore computed
  on this defective partition**, not on the one the distributed checkpoint encodes. v2
  factorizes after the filter; the live codes are now `np.array_equal` to the `.npz`
  `groups` and `GroupKFold(5)` gives identical test indices
  (`results_v2/fix_factorize_verificacion.json`). The v1 assignment is kept as the column
  `fold_deterministic_v1_prefilter_codes` of `splits/datasetB_folds.csv` for the record.
- **Figure 3(d), Dataset B, corrected.** The same numbering bug reached one figure panel: the
  2026-08-21 draft of `make_figures.py` took the Dataset-B family codes from a live
  re-featurization file (pre-filter, gapped codes). `GroupShuffleSplit(random_state=42)`
  therefore held out a *different* set of families (2 845 instead of 2 477 test rows) and the
  panel read "8.9× farther" (grouped 1-NN median 1.238 K-space units) while the manuscript
  text and `results_v2/nn_distance.csv` — both computed with the `.npz` codes — give 0.140 vs
  1.584, a factor of 11.3. `figures/figure3.png` in this repository is regenerated with the
  reference codes (11×, n = 2 477); the script now asserts the drawn medians against
  `nn_distance.csv` (85 fidelity checks). Evidence: `results_v2/fig3d_nn_source_verificacion.json`.
  Figures 1, 2 and 4 are byte-identical to the manuscript PNGs.
- **Partitions are data.** `splits/datasetA_folds.csv` and `splits/datasetB_folds.csv` list
  the fold of every row under every partition used; `make_splits.py --verify` checks them.
- **All v2 scripts are in the repository** (`code/v2/`), with the repository root resolved
  from the script location or `SUPERCON_ROOT`; no absolute paths. v1 shipped **no**
  figure-generation code — the three v1 PNGs could not be regenerated from the repository;
  `code/v2/make_figures.py` now rebuilds all four figures and asserts every drawn value
  against its CSV.
- **`requirements.txt` pins exact versions (`==`)** read from the environment that produced
  every v2 number (v1 used `>=` ranges while stating the versions were fixed).
- **`data/CHECKSUMS.sha256`** with the SHA-256 of the four data files and provenance notes
  (`verify_checksums.py`). The retrieval date of `supercon_stanev.csv` was not recorded in
  v1; the hash is the reference.
- **Numbers.** Table 2 now uses five *independent* grouped partitions (`GroupKFold(5,
  shuffle=True)`) instead of one fixed partition with five model seeds; top-100 precision uses
  a single canonical threshold definition (90th percentile of the *training fold*); new
  Table 3 (ICC / design effect) and Figure 4; element-presence null added. The v1 tables in
  `results/` are kept unchanged for the record.
- `results_v2/datasetB_featurized_live.npz` (8 MB, a float64 re-featurization used for
  cross-checks) is **not** committed: its post-mask `X` equals `data/datasetB_featurized.npz`
  at float32 precision (`np.array_equal` after casting) and `y` is identical; only the family
  codes follow the v1 (pre-mask) convention. `_capture_live_featurization.py` recreates it.

## Reproducibility notes

- **Grouping is deterministic.** Chemical-family ids are generated with `pandas.factorize` on
  the sorted element-set string, **never** Python's per-process-randomized `hash()`; because
  `GroupKFold` tie-breaking depends on the code *order*, the indices are versioned in `splits/`.
- **Fixed random seeds.** Every model is fitted with a fixed `random_state`. The Dataset A
  pipeline fits with `n_jobs=-1`; the Dataset B pipeline runs matminer featurization and CV
  single-threaded (`n_jobs=1`) to avoid matminer/loky deadlocks under restricted semaphore
  limits.
- **Version sensitivity.** The v1 Dataset-B top-100 value (81.4 %) does not reproduce with
  xgboost 3.0.5 (v2: 85.2 %); see `results_v2/P1_diagnostico.md`. Pin the versions in
  `requirements.txt` to reproduce v2.
- **Datasets are public.** Dataset A is UCI ML Repository #464 (Hamidieh 2018); Dataset B is
  the SuperCon formula list released with Stanev et al. 2018, re-featurized with matminer's
  Magpie descriptors. A and B both derive from SuperCon and are not independent in origin.

## Data sources & citation

- **Dataset A:** Hamidieh K (2018). *Comput. Mater. Sci.* **154** 346–354.
  UCI ML Repository, Superconductivity Data Set (#464).
- **Dataset B:** Stanev V et al. (2018). *npj Comput. Mater.* **4** 29.

If you use this code or the leakage-aware protocol, please cite the accompanying paper:

```bibtex
@article{cardona2026leakage,
  title   = {Random splits inflate the accuracy of composition-based
             superconductor $T_c$ models: a leakage-aware evaluation
             replicated across two datasets},
  author  = {Cardona Osorio, Gabriel Jaime},
  journal = {Machine Learning: Science and Technology},
  year    = {2026},
  note    = {Submitted}
}
```

## Beyond the paper: `code/referee/` and `exploratory/`

Two bodies of work that share this repository's data and family grouping but are **not** inputs to
any number in the manuscript:

- **`code/referee/`** — conformal prediction intervals for $T_c$ under family-held-out deployment
  (split / CQR / normalized conformal; heteroskedasticity vs. exchangeability decomposition; paired
  family bootstrap). Outputs in `results_referee/` and `results_seed/`. Re-run and verified
  identical on 2026-09-02. See `code/referee/README.md`.
- **`exploratory/`** — the pre-paper EDA on the raw SuperCon dump, including a temporal validation
  by publication year that the paper does not use. See `exploratory/README.md`.

Both were merged from the `referee-response` branch, whose history is preserved.

## License

MIT — see [LICENSE](LICENSE). Dataset A and Dataset B remain under their original
terms at the sources cited above.

---
*ORCID: [0009-0003-3743-8559](https://orcid.org/0009-0003-3743-8559)*
