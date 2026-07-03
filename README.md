# Leakage-aware evaluation of composition-based superconductor $T_c$ models

**Repository:** https://github.com/Freakscode/superconductor-tc-leakage-eval

Code, deterministic splits, featurized data, and figure-generation scripts for:

> **Random splits inflate the accuracy of composition-based superconductor $T_c$
> models: a leakage-aware evaluation replicated across two datasets**
> Gabriel Jaime Cardona Osorio, Institución Universitaria Pascual Bravo, Medellín, Colombia.
> Submitted to *Machine Learning: Science and Technology* (IOP Publishing).

---

## The result in one paragraph

Composition-based ML models for the superconducting critical temperature are
routinely evaluated with **random** train/test splits. Because these datasets are
built from **doping series** — families of near-identical compounds sharing the same
constituent elements and differing only in stoichiometry — a random split scatters
family members across both partitions and scores the model on chemical near-twins of
its own training data. Enforcing a **chemical-family-aware** split (GroupKFold on the
set of constituent elements) raises XGBoost MAE by **~61%** (5.2→8.4 K on Dataset A;
3.9→6.3 K on Dataset B) and lowers $R^2$ by **~0.08**, replicated across two
independent datasets and two model families. A **featureless family-mean null model**
recovers 82–86% of the random-split $R^2$ but collapses to $R^2\approx0$ under
family-aware splitting — most of the headline accuracy is memorization of family
structure. The models nonetheless retain **81–90% top-100 screening precision** under
the family-aware split (where the null ranker falls to the ~10% base rate), so they
remain useful ranking tools.

## Repository layout

```
superconductor-tc-leakage-eval/
├── code/
│   ├── tc_pipeline.py            # Dataset A (Hamidieh/UCI) — full random-vs-family pipeline
│   ├── datasetB_pipeline.py      # Dataset B (Stanev/SuperCon) — Magpie featurization + CV
│   ├── null_model_analysis.py    # featureless null model, top-100 comparator, seed intervals
│   └── get_datasetA.py           # downloads the UCI raw data (not redistributed here)
├── data/
│   ├── supercon_stanev.csv       # public SuperCon formula+Tc list (Dataset B source)
│   └── datasetB_featurized.npz   # Dataset B: X(12440x132 Magpie), y, deterministic groups
├── results/
│   ├── consolidated_results.csv          # Table 1 (both models, both datasets, both splits)
│   ├── experiment_summary.json           # Dataset A metrics + diagnostics
│   ├── datasetB_summary.json             # Dataset B metrics + diagnostics
│   ├── baseline_decomposition.{csv,json} # null-model R² decomposition (Table 1 italic rows)
│   ├── repeated_seed_intervals.csv       # 5-seed MAE inflation ± std (Table 2)
│   ├── gap_intervals.csv                 # gap summary with intervals
│   ├── top100_null_comparator.csv        # null vs model top-100 precision (Figure 2b)
│   └── intervals.json, top100_null.json  # consolidated machine-readable results
├── figures/
│   ├── figure1_anatomy.png               # Fig 1: anatomy of the leakage
│   ├── figure2_null_decomposition.png    # Fig 2: null-model decomposition
│   └── figure3_summary.png               # Fig 3: cross-dataset summary
├── requirements.txt
└── LICENSE  (MIT)
```

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# --- Dataset B (Stanev): fully self-contained, uses the shipped featurized checkpoint ---
cd code
python null_model_analysis.py --dataset B --featurized ../data/datasetB_featurized.npz

# --- Dataset A (Hamidieh/UCI): fetch raw data first (~28 MB, not redistributed) ---
python get_datasetA.py
python null_model_analysis.py --dataset A --uci-dir ../data/
python tc_pipeline.py          # full random-vs-family CV, both models
```

### Key headline numbers (5-fold CV)

| Dataset | Model | Random MAE | Family MAE | MAE inflation | Random $R^2$ | Family $R^2$ |
|---|---|---|---|---|---|---|
| A | XGBoost | 5.22 | 8.39 | +61% | 0.929 | 0.850 |
| A | Random Forest | 5.40 | 8.68 | +61% | 0.923 | 0.845 |
| B | XGBoost | 3.90 | 6.28 | +61% | 0.929 | 0.846 |
| B | Random Forest | 4.33 | 6.52 | +51% | 0.916 | 0.838 |
| A | *Family-mean null* | 9.38 | 29.38 | — | *0.802* | *−0.004* |
| B | *Family-mean null* | 8.18 | 22.68 | — | *0.767* | *−0.004* |

## Reproducibility notes

- **Grouping is deterministic.** Chemical-family group ids are generated with
  `pandas.factorize` on the sorted element-set string, **never** Python's
  per-process-randomized `hash()`. Splits are identical across runs and machines.
- **Fixed random seeds.** Every model is fitted with a fixed `random_state`, so the
  tree ensembles are reproducible independent of thread count. The Dataset A pipeline
  (`tc_pipeline.py`) fits with `n_jobs=-1` for speed; the Dataset B pipeline
  (`datasetB_pipeline.py`) additionally runs matminer featurization and CV
  single-threaded (`n_jobs=1`) to avoid matminer/loky deadlocks under restricted
  semaphore limits. Neither choice affects the reported numbers.
- **Datasets are public.** Dataset A is UCI ML Repository #464 (Hamidieh 2018);
  Dataset B is the SuperCon formula list released with Stanev et al. 2018, re-featurized
  from scratch with matminer's Magpie descriptors.

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

## License

MIT — see [LICENSE](LICENSE). Dataset A and Dataset B remain under their original
terms at the sources cited above.

---
*ORCID: [0009-0003-3743-8559](https://orcid.org/0009-0003-3743-8559)*
