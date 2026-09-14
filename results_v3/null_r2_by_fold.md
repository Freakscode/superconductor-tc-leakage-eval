# R² del nulo agrupado: por qué es ligeramente negativo

Procedencia: `code/v3/null_r2_v3.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5 | input results_v3/oof_v3.csv.gz (FamilyMeanNull, grouped = GroupKFold(5) det.; under GroupKFold every test family is unseen so the prediction is the TRAIN-fold mean) | R2 = 1 - SSE/SST, SST around the pooled OOF mean`

## Dataset A

R² pooled del nulo agrupado = -0.00684. Descomposición exacta: R² = 1 − (Σ_f SSW_f + Σ_f n_f(ȳ_f − c_f)²)/SST = 1 − (24644786 + 473802)/24948009.
Si cada constante c_f coincidiera con la media del test del fold, R² sería 1 − ΣSSW/SST = +0.01215 (SST incluye la varianza ENTRE folds de las medias de test, 303223). El término de desajuste Σ n_f(ȳ_f − c_f)² = 473802 es lo que empuja el R² por debajo de cero.

| fold | n_test | c_f (media train) | ȳ_f (media test) | ȳ_f − c_f (K) | n_f(ȳ_f−c_f)² | R²_f (ref. media test) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4253 | 33.839 | 36.751 | +2.912 | 36059 | -0.00694 |
| 1 | 4253 | 35.800 | 28.906 | -6.894 | 202112 | -0.05539 |
| 2 | 4253 | 33.125 | 39.606 | +6.481 | 178616 | -0.02777 |
| 3 | 4252 | 35.125 | 31.607 | -3.517 | 52606 | -0.01135 |
| 4 | 4252 | 34.218 | 35.236 | +1.018 | 4408 | -0.00093 |

## Dataset B

R² pooled del nulo agrupado = -0.00338. Descomposición exacta: R² = 1 − (Σ_f SSW_f + Σ_f n_f(ȳ_f − c_f)²)/SST = 1 − (9962199 + 93968)/10022341.
Si cada constante c_f coincidiera con la media del test del fold, R² sería 1 − ΣSSW/SST = +0.00600 (SST incluye la varianza ENTRE folds de las medias de test, 60141). El término de desajuste Σ n_f(ȳ_f − c_f)² = 93968 es lo que empuja el R² por debajo de cero.

| fold | n_test | c_f (media train) | ȳ_f (media test) | ȳ_f − c_f (K) | n_f(ȳ_f−c_f)² | R²_f (ref. media test) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2468 | 23.769 | 22.060 | -1.709 | 7205 | -0.00412 |
| 1 | 2467 | 24.310 | 19.893 | -4.417 | 48124 | -0.02850 |
| 2 | 2467 | 23.276 | 24.029 | +0.753 | 1397 | -0.00063 |
| 3 | 2467 | 22.775 | 26.036 | +3.261 | 26236 | -0.01219 |
| 4 | 2467 | 23.004 | 25.116 | +2.112 | 11006 | -0.00513 |

