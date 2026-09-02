# `code/referee/` — conformal-interval re-analysis under family-held-out deployment

Post-submission analysis brought in from the `referee-response` branch (2026-08-28). It is **not
part of the leakage paper's results**; it reuses this repository's data, chemical-family grouping
and featurizations to study how conformal prediction intervals for $T_c$ behave when the test
compounds come from unseen chemical families ("deploy" regime) rather than from a random split
("iid"). Model: XGBoost (600 trees, depth 8), nominal coverage $1-\alpha = 0.90$, seeds 0–4.

Scripts resolve the repository root from their own location (`SUPERCON_ROOT` overrides) and read
`data/`; they write to `results_referee/` and `results_seed/`. Run from anywhere:

```bash
python code/referee/referee_reanalysis.py   # -> results_referee/methods_summary.json, methods_rows.csv, deploy_materials.csv  (~4.5 min)
python code/referee/referee_inference.py    # -> results_referee/inference.json, paired_A.csv, paired_B.csv                   (~2 min)
python code/referee/seed_conformal3.py      # -> results_seed/variants_conditional.{csv,json}                                  (~2 min)
python code/referee/seed_diagA.py           # -> results_seed/datasetA_*.csv, datasetA_drivers.json                            (~30 s)
```

| script | status | what it measures |
|---|---|---|
| `referee_reanalysis.py` | current | R1 decomposes the high-$T_c$ coverage deficit into heteroskedasticity vs. exchangeability failure (two oracles); R2 compares split / CQR / normalized conformal under iid, shift and deploy regimes on both datasets |
| `referee_inference.py` | current | I1 paired family-bootstrap of normalized-CV vs. split coverage; I2 cluster-robust test of the near/far coverage gap within high-$T_c$, with and without adjusting for predicted $T_c$ |
| `seed_conformal3.py` | current | 3 methods × 3 regimes × 5 seeds with conditional coverage (Dataset B); supersedes `seed_conformal.py` and `seed_conformal2.py` |
| `seed_diagA.py` | current | Dataset A replication of the distance/coverage diagnostics |
| `seed_conformal.py`, `seed_conformal2.py`, `seed_diag.py`, `seed_diag2.py` | superseded | earlier iterations, kept because their outputs are in `results_seed/` |

**Reproduction check (2026-09-02).** The four current scripts were re-run from this layout in the
pinned environment (`requirements.txt`); all 13 files they produce are numerically identical to the
versions committed on the branch (`inference.json` to $1.5\times10^{-15}$, everything else exact).
The 12 files written by the superseded scripts were not re-run.

**What the numbers say** (`methods_summary.json`, `inference.json`; nominal 0.90):

- Split conformal covers high-$T_c$ compounds at 0.84 (A) / 0.72 (B) under iid and at **0.61 /
  0.51 under deploy**. Even the global oracle (quantile from the test residuals, so marginal
  coverage is nominal by construction) leaves high-$T_c$ at 0.82 / 0.67: part of the deficit is
  heteroskedasticity that no constant-width band can fix, not only exchangeability failure.
- Normalized conformal with cross-validated scale recovers 0.80 / 0.76 under deploy. The paired
  family bootstrap puts its high-$T_c$ gain over split at +0.20 [0.13, 0.26] on A, with **no change
  in overall coverage** (−0.01 [−0.04, 0.02]): the method redistributes coverage toward high $T_c$
  by widening those intervals 1.56× and narrowing low-$T_c$ ones 0.88×.
- The "Simpson reversal" (within high-$T_c$, compounds far from training data violate more) is
  present marginally (far − near coverage −0.19 [−0.29, −0.10]; unadjusted logit coefficient
  −0.50 [−0.66, −0.35]) but **does not survive adjustment for predicted $T_c$**: the adjusted
  coefficient is −0.01 [−0.18, 0.15]. Distance to the training set is a proxy for predicted $T_c$
  here, not an independent driver of interval failure. This is the branch's own result; the
  branch's commit message does not draw this conclusion, so it is recorded here.
