# Control: retirar familiares vs retirar filas al azar (H10)

`code/v3/remove_relatives_control.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5 | XGBoost = make_models(42)['XGBoost'] of tc_pipeline (A) / datasetB_pipeline (B), paper descriptors, n_jobs=8 | D1: test = KFold(5, shuffle, rs=42) fold, pool = other folds | D2: test = half of each GroupKFold(5) fold (per-family halves, default_rng(0), singletons -> test), pool = other folds + other half | (b) pool minus relatives (same family as a test row); (c) pool minus |relatives| random rows, reps default_rng(0..4) | top100: K=100, threshold p90(pool a), >= | intervals: paired bootstrap of whole families (n_boot=2000, default_rng(0)), conditional on predictions`


## Dataset A — D1_test_fold_aleatorio (unión de los tests: 21263 filas, 3365 familias; retirado en b: 78.2 % del pool)

| condición | n_train medio | MAE [IC fam] | RMSE | R² [IC fam] | top-100 media±sd |
|---|---|---|---|---|---|
| a_pool_con_familiares | 17010 | 5.230 [4.876, 5.594] | 9.173 | 0.9283 [0.9181, 0.9374] | 0.984±0.012 |
| b_pool_sin_familiares | 3707 | 10.135 [9.225, 11.064] | 15.723 | 0.7893 [0.7526, 0.8163] | 0.934±0.027 |
| c_pool_sin_n_filas_al_azar_media_reps | 3707 | 6.806 [6.391, 7.227] | 11.590 | 0.8855 [0.8709, 0.8984] | 0.958±0.003 |

| diferencia | ΔMAE [IC fam] | ΔRMSE | ΔR² [IC fam] | red. SSE [IC fam] |
|---|---|---|---|---|
| diff_b_minus_a | +4.905 [+4.220, +5.627] | +6.550 | -0.1390 [-0.1704, -0.1145] | -193.8 % [-236.9, -157.5] |
| diff_c_minus_a_media_reps | +1.576 [+1.451, +1.708] | +2.417 | -0.0428 [-0.0494, -0.0372] | -59.6 % [-68.4, -52.2] |
| diff_b_minus_c_media_reps | +3.329 [+2.636, +4.059] | +4.133 | -0.0962 [-0.1260, -0.0715] | -84.1 % [-110.3, -60.7] |

Fracción de la brecha (b−a) en MAE atribuible a los familiares y no al tamaño: 67.9 %.


## Dataset A — D2_test_mitad_fold_agrupado (unión de los tests: 11673 filas, 3365 familias; retirado en b: 10.1 % del pool)

| condición | n_train medio | MAE [IC fam] | RMSE | R² [IC fam] | top-100 media±sd |
|---|---|---|---|---|---|
| a_pool_con_familiares | 18928 | 5.569 [5.215, 5.913] | 9.722 | 0.9190 [0.9063, 0.9288] | 0.902±0.097 |
| b_pool_sin_familiares | 17010 | 8.145 [7.481, 8.873] | 12.906 | 0.8573 [0.8342, 0.8760] | 0.812±0.147 |
| c_pool_sin_n_filas_al_azar_media_reps | 17010 | 5.636 [5.278, 5.989] | 9.867 | 0.9166 [0.9040, 0.9264] | 0.889±0.009 |

| diferencia | ΔMAE [IC fam] | ΔRMSE | ΔR² [IC fam] | red. SSE [IC fam] |
|---|---|---|---|---|
| diff_b_minus_a | +2.576 [+2.083, +3.137] | +3.183 | -0.0617 [-0.0811, -0.0464] | -76.2 % [-101.8, -54.8] |
| diff_c_minus_a_media_reps | +0.067 [+0.012, +0.122] | +0.145 | -0.0024 [-0.0044, -0.0006] | -3.0 % [-5.4, -0.6] |
| diff_b_minus_c_media_reps | +2.509 [+2.026, +3.052] | +3.039 | -0.0593 [-0.0786, -0.0444] | -71.1 % [-95.5, -50.9] |

Fracción de la brecha (b−a) en MAE atribuible a los familiares y no al tamaño: 97.4 %.


## Dataset B — D1_test_fold_aleatorio (unión de los tests: 12336 filas, 3046 familias; retirado en b: 66.4 % del pool)

| condición | n_train medio | MAE [IC fam] | RMSE | R² [IC fam] | top-100 media±sd |
|---|---|---|---|---|---|
| a_pool_con_familiares | 9869 | 3.779 [3.487, 4.087] | 7.439 | 0.9319 [0.9226, 0.9397] | 0.960±0.021 |
| b_pool_sin_familiares | 3319 | 7.145 [6.416, 8.007] | 12.916 | 0.7947 [0.7506, 0.8265] | 0.716±0.092 |
| c_pool_sin_n_filas_al_azar_media_reps | 3319 | 5.039 [4.661, 5.444] | 9.485 | 0.8892 [0.8751, 0.9009] | 0.912±0.007 |

| diferencia | ΔMAE [IC fam] | ΔRMSE | ΔR² [IC fam] | red. SSE [IC fam] |
|---|---|---|---|---|
| diff_b_minus_a | +3.366 [+2.855, +3.986] | +5.476 | -0.1372 [-0.1782, -0.1086] | -201.4 % [-255.9, -158.9] |
| diff_c_minus_a_media_reps | +1.260 [+1.131, +1.394] | +2.046 | -0.0426 [-0.0508, -0.0359] | -62.6 % [-74.3, -52.5] |
| diff_b_minus_c_media_reps | +2.106 [+1.636, +2.688] | +3.431 | -0.0946 [-0.1346, -0.0666] | -85.5 % [-120.0, -59.5] |

Fracción de la brecha (b−a) en MAE atribuible a los familiares y no al tamaño: 62.6 %.


## Dataset B — D2_test_mitad_fold_agrupado (unión de los tests: 7171 filas, 3046 familias; retirado en b: 9.5 % del pool)

| condición | n_train medio | MAE [IC fam] | RMSE | R² [IC fam] | top-100 media±sd |
|---|---|---|---|---|---|
| a_pool_con_familiares | 10902 | 4.126 [3.815, 4.444] | 7.971 | 0.9190 [0.9089, 0.9287] | 0.840±0.094 |
| b_pool_sin_familiares | 9869 | 5.947 [5.392, 6.564] | 10.858 | 0.8497 [0.8245, 0.8712] | 0.756±0.101 |
| c_pool_sin_n_filas_al_azar_media_reps | 9869 | 4.216 [3.895, 4.546] | 8.210 | 0.9140 [0.9033, 0.9244] | 0.851±0.006 |

| diferencia | ΔMAE [IC fam] | ΔRMSE | ΔR² [IC fam] | red. SSE [IC fam] |
|---|---|---|---|---|
| diff_b_minus_a | +1.821 [+1.489, +2.218] | +2.888 | -0.0693 [-0.0897, -0.0520] | -85.6 % [-111.4, -64.3] |
| diff_c_minus_a_media_reps | +0.090 [+0.020, +0.166] | +0.240 | -0.0050 [-0.0094, -0.0012] | -6.1 % [-11.8, -1.4] |
| diff_b_minus_c_media_reps | +1.731 [+1.411, +2.127] | +2.648 | -0.0644 [-0.0853, -0.0476] | -74.9 % [-99.5, -55.3] |

Fracción de la brecha (b−a) en MAE atribuible a los familiares y no al tamaño: 95.1 %.
