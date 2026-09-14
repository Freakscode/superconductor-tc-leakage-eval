# H07 - Inferencia pareada por familias (v3)

Procedencia: `code/v3/paired_family_bootstrap.py | input oof_v3.csv.gz (code/v3/oof_v3.py, seed 42) | family bootstrap = resample the k families with replacement, n_boot=2000, numpy default_rng(0), SAME draw for both protocols (paired); row bootstrap = resample rows, same n_boot/seed, unpaired (v2 scheme) and paired | percentile CI 2.5/97.5 | intervals CONDITIONAL on the fitted OOF predictions (no re-training) | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5`

Intervalos percentil 95 %, **condicionales a las predicciones OOF ya ajustadas** (semilla 42; no incluyen la variabilidad de reentrenar). Bootstrap de familias = remuestreo con reemplazo de las k familias completas, MISMAS familias en ambos brazos (pareado). Bootstrap por filas = esquema de results_v2/bootstrap_intervalos.csv (filas independientes en cada brazo, no pareado) y su variante pareada. Umbral alto-Tc no interviene aquí.

## Contraste random - grouped, bootstrap pareado por familias

| Dataset | Modelo | MAE random | MAE grouped | Inflación % [IC] | ΔMAE K [IC] | ΔR² [IC] | MAE macro random | MAE macro grouped | Inflación macro % [IC] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | FamilyMeanNull | 9.40 [8.59, 10.22] | 29.41 [27.62, 31.26] | 212.8 [189.7, 238.5] | 20.01 [18.43, 21.62] | 0.807 [0.781, 0.837] | 14.30 [13.77, 14.85] | 28.89 [28.43, 29.36] | 102.0 [95.3, 108.8] |
| A | RandomForest | 5.38 [5.02, 5.77] | 8.69 [7.91, 9.53] | 61.4 [52.2, 71.5] | 3.31 [2.76, 3.93] | 0.079 [0.061, 0.099] | 5.33 [5.06, 5.60] | 6.69 [6.39, 6.99] | 25.5 [23.2, 28.0] |
| A | XGBoost | 5.23 [4.88, 5.59] | 8.31 [7.57, 9.07] | 58.9 [50.4, 68.6] | 3.08 [2.58, 3.65] | 0.074 [0.058, 0.094] | 5.07 [4.82, 5.32] | 6.43 [6.16, 6.72] | 27.0 [23.8, 30.5] |
| B | FamilyMeanNull | 8.07 [7.42, 8.72] | 22.49 [21.20, 23.86] | 178.8 [161.3, 197.3] | 14.42 [13.37, 15.52] | 0.772 [0.743, 0.801] | 11.93 [11.45, 12.40] | 20.80 [20.34, 21.29] | 74.3 [68.7, 80.3] |
| B | RandomForest | 4.22 [3.91, 4.57] | 6.46 [5.83, 7.19] | 53.0 [44.5, 62.8] | 2.24 [1.84, 2.72] | 0.079 [0.061, 0.105] | 3.95 [3.73, 4.18] | 4.81 [4.56, 5.07] | 21.9 [18.7, 25.2] |
| B | XGBoost | 3.78 [3.49, 4.09] | 6.18 [5.58, 6.88] | 63.7 [54.6, 74.3] | 2.41 [2.01, 2.88] | 0.084 [0.066, 0.109] | 3.50 [3.31, 3.70] | 4.53 [4.29, 4.78] | 29.4 [25.4, 33.8] |

## Anchura del intervalo: familias (pareado) frente a filas

| Dataset | Modelo | Estadístico | IC familias | IC filas no pareado (v2) | IC filas pareado | anchura fam/filas-no-par | anchura fam/filas-par | IC v2 publicado |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | FamilyMeanNull | MAE_grouped | [27.62, 31.26] | [29.18, 29.66] | [29.17, 29.65] | 7.54 | 7.58 | - |
| A | FamilyMeanNull | MAE_random | [8.59, 10.22] | [9.25, 9.56] | [9.24, 9.56] | 5.19 | 5.08 | - |
| A | FamilyMeanNull | R2_grouped | [-0.025, -0.003] | [-0.008, -0.006] | [-0.008, -0.006] | 14.85 | 14.77 | - |
| A | FamilyMeanNull | R2_random | [0.773, 0.823] | [0.794, 0.808] | [0.793, 0.807] | 3.59 | 3.64 | - |
| A | FamilyMeanNull | dMAE_K | [18.43, 21.62] | [19.72, 20.30] | [19.73, 20.28] | 5.53 | 5.76 | - |
| A | FamilyMeanNull | dR2 | [0.781, 0.837] | [0.800, 0.814] | [0.800, 0.814] | 4.02 | 3.98 | - |
| A | FamilyMeanNull | inflacion_pct | [189.67, 238.46] | [207.04, 218.98] | [207.07, 218.40] | 4.09 | 4.31 | - |
| A | RandomForest | MAE_grouped | [7.91, 9.53] | [8.55, 8.82] | [8.55, 8.83] | 5.95 | 5.91 | - |
| A | RandomForest | MAE_random | [5.02, 5.77] | [5.27, 5.48] | [5.27, 5.49] | 3.55 | 3.49 | - |
| A | RandomForest | R2_grouped | [0.820, 0.866] | [0.840, 0.851] | [0.840, 0.851] | 4.37 | 4.22 | - |
| A | RandomForest | R2_random | [0.914, 0.933] | [0.920, 0.928] | [0.920, 0.928] | 2.32 | 2.31 | - |
| A | RandomForest | dMAE_K | [2.76, 3.93] | [3.14, 3.48] | [3.22, 3.40] | 3.44 | 6.26 | - |
| A | RandomForest | dR2 | [0.061, 0.099] | [0.072, 0.085] | [0.075, 0.082] | 2.80 | 5.63 | - |
| A | RandomForest | inflacion_pct | [52.20, 71.53] | [57.50, 65.71] | [59.33, 63.72] | 2.35 | 4.41 | - |
| A | XGBoost | MAE_grouped | [7.57, 9.07] | [8.18, 8.45] | [8.18, 8.45] | 5.61 | 5.52 | [8.18, 8.44] |
| A | XGBoost | MAE_random | [4.88, 5.59] | [5.13, 5.33] | [5.13, 5.33] | 3.55 | 3.56 | [5.13, 5.33] |
| A | XGBoost | R2_grouped | [0.830, 0.875] | [0.849, 0.860] | [0.849, 0.860] | 4.18 | 4.15 | - |
| A | XGBoost | R2_random | [0.918, 0.937] | [0.924, 0.932] | [0.924, 0.932] | 2.22 | 2.26 | - |
| A | XGBoost | dMAE_K | [2.58, 3.65] | [2.92, 3.25] | [2.98, 3.19] | 3.24 | 5.19 | - |
| A | XGBoost | dR2 | [0.058, 0.094] | [0.067, 0.080] | [0.070, 0.078] | 2.66 | 4.49 | - |
| A | XGBoost | inflacion_pct | [50.36, 68.65] | [55.02, 62.95] | [56.68, 61.43] | 2.31 | 3.85 | [55.07, 62.97] |
| B | FamilyMeanNull | MAE_grouped | [21.20, 23.86] | [22.18, 22.78] | [22.18, 22.79] | 4.37 | 4.38 | - |
| B | FamilyMeanNull | MAE_random | [7.42, 8.72] | [7.88, 8.26] | [7.89, 8.26] | 3.43 | 3.48 | - |
| B | FamilyMeanNull | R2_grouped | [-0.014, -0.001] | [-0.004, -0.003] | [-0.004, -0.003] | 8.75 | 9.04 | - |
| B | FamilyMeanNull | R2_random | [0.739, 0.794] | [0.756, 0.781] | [0.755, 0.780] | 2.23 | 2.22 | - |
| B | FamilyMeanNull | dMAE_K | [13.37, 15.52] | [14.07, 14.76] | [14.10, 14.73] | 3.08 | 3.41 | - |
| B | FamilyMeanNull | dR2 | [0.743, 0.801] | [0.759, 0.784] | [0.759, 0.784] | 2.34 | 2.27 | - |
| B | FamilyMeanNull | inflacion_pct | [161.27, 197.27] | [171.23, 186.37] | [172.10, 184.97] | 2.38 | 2.80 | - |
| B | RandomForest | MAE_grouped | [5.83, 7.19] | [6.30, 6.63] | [6.30, 6.64] | 4.17 | 4.08 | - |
| B | RandomForest | MAE_random | [3.91, 4.57] | [4.11, 4.35] | [4.11, 4.35] | 2.79 | 2.77 | - |
| B | RandomForest | R2_grouped | [0.809, 0.860] | [0.829, 0.847] | [0.829, 0.847] | 3.03 | 3.01 | - |
| B | RandomForest | R2_random | [0.907, 0.926] | [0.912, 0.924] | [0.911, 0.924] | 1.58 | 1.54 | - |
| B | RandomForest | dMAE_K | [1.84, 2.72] | [2.03, 2.44] | [2.13, 2.35] | 2.16 | 4.00 | - |
| B | RandomForest | dR2 | [0.061, 0.105] | [0.069, 0.090] | [0.073, 0.086] | 2.17 | 3.53 | - |
| B | RandomForest | inflacion_pct | [44.55, 62.82] | [47.15, 58.77] | [50.16, 56.16] | 1.57 | 3.05 | - |
| B | XGBoost | MAE_grouped | [5.58, 6.88] | [6.02, 6.34] | [6.03, 6.34] | 4.06 | 4.14 | [6.01, 6.33] |
| B | XGBoost | MAE_random | [3.49, 4.09] | [3.67, 3.90] | [3.67, 3.89] | 2.69 | 2.69 | [3.79, 4.03] |
| B | XGBoost | R2_grouped | [0.820, 0.869] | [0.839, 0.856] | [0.839, 0.856] | 2.91 | 2.90 | - |
| B | XGBoost | R2_random | [0.923, 0.940] | [0.927, 0.937] | [0.927, 0.937] | 1.67 | 1.64 | - |
| B | XGBoost | dMAE_K | [2.01, 2.88] | [2.20, 2.60] | [2.29, 2.53] | 2.20 | 3.58 | - |
| B | XGBoost | dR2 | [0.066, 0.109] | [0.074, 0.094] | [0.078, 0.091] | 2.17 | 3.21 | - |
| B | XGBoost | inflacion_pct | [54.57, 74.33] | [57.14, 70.08] | [60.08, 67.50] | 1.53 | 2.66 | [51.39, 63.93] |
