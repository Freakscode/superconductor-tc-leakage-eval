# Validación temporal (experimento 7, v3)

Procedencia: `{"script": "code/v3/temporal_validation.py", "python": "3.13.15", "numpy": "2.3.3", "pandas": "2.3.2", "sklearn": "1.7.2", "xgboost": "3.0.5", "matminer": "0.9.2", "pymatgen": "2025.6.14", "input": "data/temporal_corpus_v3.npz", "modelos": "code/datasetB_pipeline.make_models(42), n_jobs=8", "corte": "train year_first<T, test year_first>=T", "umbral_top100": "percentil 90 de y del entrenamiento, operador >=", "bootstrap": "familias completas, B=2000, semilla 0, pareado; condicional a las predicciones OOF/temporales", "protocolos": "L0 KFold(5,shuffle,rs=42); L1==L0 (corpus deduplicado); L2 GroupKFold(5) familia; L3 GroupKFold(5) anfitrion (fraccion>=0.05); L4 GroupKFold(5) sobre KMeans(k=50, rs=0) de fracciones estandarizadas del train", "segundos_total": 2194.6, "corte_principal": 2010, "meta": {"report_level_available": true, "principal_T1995": {"n_train": 6281, "n_test": 6262, "thr_p90_train": 76.5, "frac_familia_nueva_test": 0.7425742574257426, "fam_train": 1822, "fam_test": 1734, "segundos": 237.5}, "principal_T2005": {"n_train": 9254, "n_test": 3289, "thr_p90_train": 80.0, "frac_familia_nueva_test": 0.8364244451200973, "fam_train": 2375, "fam_test": 1092, "segundos": 232.6}, "principal_T2010": {"n_train": 10215, "n_test": 2328, "thr_p90_train": 79.5, "frac_familia_nueva_test": 0.7633161512027491, "fam_train": 2642, "fam_test": 798, "segundos": 218.3}, "principal_T2015": {"n_train": 11734, "n_test": 809, "thr_p90_train": 77.0, "frac_familia_nueva_test": 0.7181705809641533`

Pregunta: ¿qué protocolo retrospectivo (aleatorio / composición / sistema elemental / anfitrión / LOCO) anticipa mejor el rendimiento real sobre superconductores reportados por primera vez DESPUÉS de un corte temporal?

Diseño: corpus `data/temporal_corpus_v3.csv` (una fila por composición normalizada, Tc = mediana de reportes, año = primer reporte). Para cada corte T: (a) modelo entrenado en todo lo anterior a T (año < T), congelado, evaluado en lo posterior (año ≥ T); (b) sólo con lo anterior a T, predicciones OOF 5-fold bajo L0–L4; (c) estimación (b) − real (a). Umbral alto-Tc = percentil 90 de la Tc de entrenamiento, operador ≥. Intervalos: bootstrap de familias completas (B=2000, semilla 0), pareado; condicionales a las predicciones ya ajustadas. L1 (GroupKFold por composición) coincide con L0 en este corpus deduplicado salvo por la partición.

## 1. Rendimiento temporal real (modelo congelado) — XGBoost

| T | n_train | n_test | % fam. nueva | estrato | n | fam. | MAE [IC fam.] | RMSE | R² [IC fam.] | Spearman | top-100 [IC] | prevalencia ≥p90 |
|---:|---:|---:|---:|---|---:|---:|---|---|---|---|---|---:|
| 1995 | 6281 | 6262 | 74.3 | todo | 6262 | 1734 | 10.71 [10.03, 11.35] | 16.86 | 0.65 [0.59, 0.70] | 0.73 | 0.90 [0.79, 1.00] | 0.092 |
| 1995 | 6281 | 6262 | 74.3 | familia_nueva | 4650 | 1445 | 10.96 [10.24, 11.70] | 17.34 | 0.52 [0.44, 0.59] | 0.60 | 0.76 [0.63, 0.88] | 0.055 |
| 1995 | 6281 | 6262 | 74.3 | familia_vista | 1612 | 289 | 9.99 [8.75, 11.22] | 15.39 | 0.78 [0.71, 0.83] | 0.89 | 0.88 [0.74, 0.98] | 0.202 |
| 1995 | 6281 | 6262 | 74.3 | composicion_vista_reportes | 1776 | 282 | 2.91 [2.14, 3.52] | 6.24 | 0.97 [0.94, 0.98] | 0.98 | 0.98 [0.88, 1.00] | 0.218 |
| 2005 | 9254 | 3289 | 83.6 | todo | 3289 | 1092 | 8.72 [7.93, 9.53] | 15.22 | 0.31 [0.14, 0.46] | 0.57 | 0.47 [0.29, 0.63] | 0.018 |
| 2005 | 9254 | 3289 | 83.6 | familia_nueva | 2751 | 892 | 9.51 [8.64, 10.45] | 16.23 | 0.10 [-0.12, 0.27] | 0.47 | 0.23 [0.10, 0.38] | 0.011 |
| 2005 | 9254 | 3289 | 83.6 | familia_vista | 538 | 200 | 4.71 [3.69, 5.84] | 8.30 | 0.87 [0.83, 0.91] | 0.89 | 0.30 [0.11, 0.53] | 0.056 |
| 2005 | 9254 | 3289 | 83.6 | composicion_vista_reportes | 892 | 270 | 2.09 [1.60, 2.67] | 6.49 | 0.90 [0.80, 0.98] | 0.94 | 0.19 [0.03, 0.46] | 0.021 |
| 2010 | 10215 | 2328 | 76.3 | todo | 2328 | 798 | 7.43 [6.67, 8.27] | 13.17 | 0.33 [0.18, 0.47] | 0.61 | 0.17 [0.07, 0.29] | 0.009 |
| 2010 | 10215 | 2328 | 76.3 | familia_nueva | 1777 | 625 | 8.47 [7.51, 9.50] | 14.64 | 0.19 [0.01, 0.34] | 0.55 | 0.11 [0.04, 0.20] | 0.009 |
| 2010 | 10215 | 2328 | 76.3 | familia_vista | 551 | 173 | 4.07 [3.35, 4.79] | 6.45 | 0.83 [0.71, 0.89] | 0.84 | 0.06 [0.00, 0.14] | 0.011 |
| 2010 | 10215 | 2328 | 76.3 | composicion_vista_reportes | 800 | 258 | 2.39 [1.62, 3.33] | 7.90 | 0.80 [0.62, 0.95] | 0.95 | 0.06 [0.00, 0.17] | 0.009 |
| 2015 | 11734 | 809 | 71.8 | todo | 809 | 351 | 3.87 [2.98, 4.87] | 12.72 | 0.49 [0.32, 0.75] | 0.78 | 0.08 [0.03, 0.14] | 0.015 |
| 2015 | 11734 | 809 | 71.8 | familia_nueva | 581 | 246 | 4.29 [3.15, 5.65] | 14.53 | 0.40 [0.24, 0.72] | 0.72 | 0.07 [0.02, 0.14] | 0.017 |
| 2015 | 11734 | 809 | 71.8 | familia_vista | 228 | 105 | 2.81 [2.03, 3.76] | 5.99 | 0.84 [0.80, 0.88] | 0.89 | 0.02 [0.00, 0.05] | 0.009 |
| 2015 | 11734 | 809 | 71.8 | composicion_vista_reportes | 441 | 183 | 2.01 [1.16, 3.20] | 9.20 | 0.77 [0.75, 0.94] | 0.91 | 0.03 [0.00, 0.08] | 0.007 |

`composicion_vista_reportes` = re-mediciones posteriores a T de composiciones ya presentes en el entrenamiento (nivel reporte; la predicción es la del modelo para una composición de su propio entrenamiento). Para RandomForest véase el CSV.

## 2. Estimación retrospectiva por protocolo vs. real — XGBoost

### T = 1995  (n_train = 6281, n_test = 6262, 74.3 % del test en familias nuevas)

| protocolo | % OOF fam. nueva | MAE est. [IC] | MAE real | Δ MAE [IC] | R² est. [IC] | R² real | Δ R² [IC] | top-100 est. | top-100 real | Δ top-100 [IC] | Spearman est. / real |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| L0 aleatorio (KFold) | 16.0 | 3.36 [2.99, 3.77] | 10.71 | -7.34 [-8.09, -6.61] | 0.934 [0.922, 0.944] | 0.650 | 0.284 [0.234, 0.347] | 0.92 | 0.90 | 0.02 [-0.10, 0.15] | 0.95 / 0.73 |
| L0 aleatorio (KFold) — reponderado a la mezcla del test | 16.0 | 3.28 [2.99, 3.59] | 10.71 | -7.42 [-8.13, -6.73] | 0.905 [0.885, 0.922] | 0.650 | 0.255 [0.202, 0.320] | 0.91 | 0.90 | 0.01 [-0.15, 0.15] | 0.88 / 0.73 |
| L1 composición | 16.0 | 3.24 [2.88, 3.62] | 10.71 | -7.47 [-8.23, -6.75] | 0.937 [0.925, 0.948] | 0.650 | 0.288 [0.237, 0.350] | 0.96 | 0.90 | 0.06 [-0.06, 0.17] | 0.95 / 0.73 |
| L1 composición — reponderado a la mezcla del test | 16.0 | 3.24 [2.96, 3.54] | 10.71 | -7.47 [-8.17, -6.77] | 0.911 [0.890, 0.928] | 0.650 | 0.261 [0.208, 0.326] | 0.93 | 0.90 | 0.03 [-0.13, 0.15] | 0.88 / 0.73 |
| L2 sistema elemental | 100.0 | 5.26 [4.50, 6.12] | 10.71 | -5.45 [-6.48, -4.40] | 0.866 [0.832, 0.891] | 0.650 | 0.217 [0.156, 0.284] | 0.84 | 0.90 | -0.06 [-0.24, 0.10] | 0.91 / 0.73 |
| L3 anfitrión (frac. ≥ 0,05) | 59.4 | 6.08 [5.30, 6.95] | 10.71 | -4.63 [-5.66, -3.57] | 0.825 [0.791, 0.855] | 0.650 | 0.175 [0.116, 0.243] | 0.73 | 0.90 | -0.17 [-0.37, 0.01] | 0.90 / 0.73 |
| L3 anfitrión (frac. ≥ 0,05) — reponderado a la mezcla del test | 59.4 | 6.03 [5.24, 6.90] | 10.71 | -4.68 [-5.71, -3.61] | 0.819 [0.786, 0.849] | 0.650 | 0.169 [0.110, 0.234] | 0.67 | 0.90 | -0.23 [-0.40, -0.01] | 0.89 / 0.73 |
| L4 LOCO (KMeans k=50) | 78.4 | 10.08 [8.74, 11.50] | 10.71 | -0.63 [-2.15, 0.93] | 0.596 [0.552, 0.638] | 0.650 | -0.053 [-0.120, 0.019] | 0.82 | 0.90 | -0.08 [-0.27, 0.08] | 0.78 / 0.73 |
| L4 LOCO (KMeans k=50) — reponderado a la mezcla del test | 78.4 | 9.81 [8.50, 11.21] | 10.71 | -0.89 [-2.39, 0.64] | 0.601 [0.557, 0.643] | 0.650 | -0.048 [-0.115, 0.024] | 0.83 | 0.90 | -0.07 [-0.27, 0.08] | 0.78 / 0.73 |

### T = 2005  (n_train = 9254, n_test = 3289, 83.6 % del test en familias nuevas)

| protocolo | % OOF fam. nueva | MAE est. [IC] | MAE real | Δ MAE [IC] | R² est. [IC] | R² real | Δ R² [IC] | top-100 est. | top-100 real | Δ top-100 [IC] | Spearman est. / real |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| L0 aleatorio (KFold) | 14.3 | 4.32 [3.96, 4.68] | 8.72 | -4.41 [-5.29, -3.55] | 0.912 [0.886, 0.930] | 0.313 | 0.599 [0.454, 0.776] | 0.99 | 0.47 | 0.52 [0.34, 0.70] | 0.95 / 0.57 |
| L0 aleatorio (KFold) — reponderado a la mezcla del test | 14.3 | 4.59 [4.08, 5.17] | 8.72 | -4.14 [-5.12, -3.21] | 0.781 [0.648, 0.884] | 0.313 | 0.468 [0.274, 0.696] | 0.99 | 0.47 | 0.52 [0.35, 0.70] | 0.88 / 0.57 |
| L1 composición | 13.7 | 4.23 [3.85, 4.59] | 8.72 | -4.49 [-5.37, -3.66] | 0.915 [0.889, 0.933] | 0.313 | 0.602 [0.458, 0.779] | 0.98 | 0.47 | 0.51 [0.34, 0.70] | 0.96 / 0.57 |
| L1 composición — reponderado a la mezcla del test | 13.7 | 4.44 [3.95, 5.03] | 8.72 | -4.29 [-5.27, -3.35] | 0.796 [0.652, 0.900] | 0.313 | 0.483 [0.281, 0.710] | 0.94 | 0.47 | 0.47 [0.29, 0.68] | 0.88 / 0.57 |
| L2 sistema elemental | 100.0 | 7.01 [6.21, 7.89] | 8.72 | -1.71 [-2.84, -0.54] | 0.818 [0.778, 0.852] | 0.313 | 0.504 [0.357, 0.685] | 0.90 | 0.47 | 0.43 [0.23, 0.64] | 0.92 / 0.57 |
| L3 anfitrión (frac. ≥ 0,05) | 55.6 | 8.10 [7.28, 8.97] | 8.72 | -0.63 [-1.78, 0.59] | 0.765 [0.732, 0.796] | 0.313 | 0.452 [0.305, 0.631] | 0.78 | 0.47 | 0.31 [0.10, 0.55] | 0.91 / 0.57 |
| L3 anfitrión (frac. ≥ 0,05) — reponderado a la mezcla del test | 55.6 | 8.31 [7.38, 9.33] | 8.72 | -0.42 [-1.68, 0.88] | 0.736 [0.699, 0.773] | 0.313 | 0.423 [0.276, 0.604] | 0.77 | 0.47 | 0.30 [0.03, 0.56] | 0.90 / 0.57 |
| L4 LOCO (KMeans k=50) | 77.8 | 11.31 [10.29, 12.32] | 8.72 | 2.58 [1.31, 3.86] | 0.618 [0.572, 0.658] | 0.313 | 0.305 [0.158, 0.488] | 0.38 | 0.47 | -0.09 [-0.32, 0.19] | 0.85 / 0.57 |
| L4 LOCO (KMeans k=50) — reponderado a la mezcla del test | 77.8 | 11.62 [10.54, 12.67] | 8.72 | 2.90 [1.60, 4.18] | 0.610 [0.562, 0.651] | 0.313 | 0.297 [0.149, 0.480] | 0.38 | 0.47 | -0.09 [-0.32, 0.19] | 0.85 / 0.57 |

### T = 2010  (n_train = 10215, n_test = 2328, 76.3 % del test en familias nuevas)

| protocolo | % OOF fam. nueva | MAE est. [IC] | MAE real | Δ MAE [IC] | R² est. [IC] | R² real | Δ R² [IC] | top-100 est. | top-100 real | Δ top-100 [IC] | Spearman est. / real |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| L0 aleatorio (KFold) | 14.0 | 4.34 [3.96, 4.70] | 7.43 | -3.10 [-4.03, -2.25] | 0.910 [0.889, 0.927] | 0.328 | 0.582 [0.440, 0.727] | 0.99 | 0.17 | 0.82 [0.68, 0.91] | 0.95 / 0.61 |
| L0 aleatorio (KFold) — reponderado a la mezcla del test | 14.0 | 4.60 [4.18, 5.09] | 7.43 | -2.84 [-3.75, -1.92] | 0.813 [0.714, 0.889] | 0.328 | 0.485 [0.316, 0.659] | 0.94 | 0.17 | 0.77 [0.60, 0.91] | 0.88 / 0.61 |
| L1 composición | 13.2 | 4.18 [3.84, 4.51] | 7.43 | -3.25 [-4.16, -2.42] | 0.912 [0.888, 0.930] | 0.328 | 0.584 [0.441, 0.730] | 0.97 | 0.17 | 0.80 [0.68, 0.91] | 0.95 / 0.61 |
| L1 composición — reponderado a la mezcla del test | 13.2 | 4.53 [4.07, 5.06] | 7.43 | -2.90 [-3.85, -1.97] | 0.783 [0.657, 0.886] | 0.328 | 0.455 [0.272, 0.648] | 0.88 | 0.17 | 0.71 [0.51, 0.88] | 0.88 / 0.61 |
| L2 sistema elemental | 100.0 | 6.76 [6.06, 7.55] | 7.43 | -0.67 [-1.83, 0.43] | 0.827 [0.790, 0.857] | 0.328 | 0.499 [0.353, 0.647] | 0.92 | 0.17 | 0.75 [0.59, 0.89] | 0.93 / 0.61 |
| L3 anfitrión (frac. ≥ 0,05) | 59.3 | 7.53 [6.81, 8.25] | 7.43 | 0.10 [-1.00, 1.12] | 0.799 [0.769, 0.826] | 0.328 | 0.471 [0.326, 0.616] | 0.87 | 0.17 | 0.70 [0.51, 0.85] | 0.91 / 0.61 |
| L3 anfitrión (frac. ≥ 0,05) — reponderado a la mezcla del test | 59.3 | 7.68 [6.95, 8.40] | 7.43 | 0.24 [-0.89, 1.26] | 0.782 [0.749, 0.812] | 0.328 | 0.454 [0.309, 0.600] | 0.86 | 0.17 | 0.69 [0.49, 0.83] | 0.91 / 0.61 |
| L4 LOCO (KMeans k=50) | 80.4 | 11.35 [10.34, 12.34] | 7.43 | 3.92 [2.55, 5.17] | 0.604 [0.567, 0.638] | 0.328 | 0.276 [0.133, 0.427] | 0.54 | 0.17 | 0.37 [0.14, 0.58] | 0.83 / 0.61 |
| L4 LOCO (KMeans k=50) — reponderado a la mezcla del test | 80.4 | 11.09 [10.10, 12.06] | 7.43 | 3.66 [2.30, 4.88] | 0.610 [0.573, 0.643] | 0.328 | 0.282 [0.140, 0.432] | 0.53 | 0.17 | 0.36 [0.15, 0.58] | 0.83 / 0.61 |

### T = 2015  (n_train = 11734, n_test = 809, 71.8 % del test en familias nuevas)

| protocolo | % OOF fam. nueva | MAE est. [IC] | MAE real | Δ MAE [IC] | R² est. [IC] | R² real | Δ R² [IC] | top-100 est. | top-100 real | Δ top-100 [IC] | Spearman est. / real |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| L0 aleatorio (KFold) | 14.0 | 4.18 [3.88, 4.48] | 3.87 | 0.31 [-0.71, 1.25] | 0.910 [0.886, 0.927] | 0.489 | 0.421 [0.152, 0.599] | 0.98 | 0.08 | 0.90 [0.82, 0.96] | 0.95 / 0.78 |
| L0 aleatorio (KFold) — reponderado a la mezcla del test | 14.0 | 4.55 [4.14, 5.00] | 3.87 | 0.67 [-0.40, 1.68] | 0.801 [0.683, 0.882] | 0.489 | 0.311 [0.023, 0.515] | 0.94 | 0.08 | 0.86 [0.73, 0.96] | 0.89 / 0.78 |
| L1 composición | 12.8 | 4.06 [3.76, 4.35] | 3.87 | 0.18 [-0.85, 1.13] | 0.913 [0.889, 0.929] | 0.489 | 0.423 [0.155, 0.601] | 0.98 | 0.08 | 0.90 [0.83, 0.96] | 0.95 / 0.78 |
| L1 composición — reponderado a la mezcla del test | 12.8 | 4.50 [4.09, 4.96] | 3.87 | 0.63 [-0.46, 1.64] | 0.803 [0.684, 0.886] | 0.489 | 0.314 [0.024, 0.517] | 0.94 | 0.08 | 0.86 [0.72, 0.96] | 0.89 / 0.78 |
| L2 sistema elemental | 100.0 | 6.49 [5.91, 7.14] | 3.87 | 2.62 [1.50, 3.76] | 0.825 [0.790, 0.852] | 0.489 | 0.336 [0.070, 0.512] | 0.92 | 0.08 | 0.84 [0.75, 0.94] | 0.92 / 0.78 |
| L3 anfitrión (frac. ≥ 0,05) | 57.1 | 7.44 [6.82, 8.09] | 3.87 | 3.57 [2.44, 4.69] | 0.786 [0.753, 0.812] | 0.489 | 0.296 [0.031, 0.475] | 0.81 | 0.08 | 0.73 [0.50, 0.89] | 0.91 / 0.78 |
| L3 anfitrión (frac. ≥ 0,05) — reponderado a la mezcla del test | 57.1 | 7.50 [6.88, 8.16] | 3.87 | 3.62 [2.49, 4.75] | 0.770 [0.733, 0.799] | 0.489 | 0.281 [0.015, 0.457] | 0.77 | 0.08 | 0.69 [0.43, 0.88] | 0.90 / 0.78 |
| L4 LOCO (KMeans k=50) | 79.2 | 10.89 [9.91, 11.85] | 3.87 | 7.01 [5.63, 8.34] | 0.602 [0.567, 0.632] | 0.489 | 0.112 [-0.157, 0.286] | 0.36 | 0.08 | 0.28 [0.10, 0.54] | 0.82 / 0.78 |
| L4 LOCO (KMeans k=50) — reponderado a la mezcla del test | 79.2 | 10.53 [9.56, 11.51] | 3.87 | 6.66 [5.28, 7.94] | 0.609 [0.576, 0.639] | 0.489 | 0.120 [-0.150, 0.294] | 0.37 | 0.08 | 0.29 [0.11, 0.54] | 0.83 / 0.78 |

## 3. Ranking de protocolos por |estimación − real| (estimación 'todo')

| T | modelo | MAE | R² | top-100 | Spearman |
|---:|---|---|---|---|---|
| 1995 | RandomForest | L4 > L3 > L2 > L0 > L1 | L4 > L3 > L2 > L0 > L1 | L0 > L1 > L2 > L4 > L3 | L4 > L3 > L2 > L1 > L0 |
| 1995 | XGBoost | L4 > L3 > L2 > L0 > L1 | L4 > L3 > L2 > L0 > L1 | L0 > L1 > L2 > L4 > L3 | L4 > L3 > L2 > L0 > L1 |
| 2005 | RandomForest | L3 > L2 > L4 > L0 > L1 | L4 > L3 > L2 > L0 > L1 | L3 > L4 > L2 > L0 > L1 | L4 > L3 > L2 > L0 > L1 |
| 2005 | XGBoost | L3 > L2 > L4 > L0 > L1 | L4 > L3 > L2 > L0 > L1 | L4 > L3 > L2 > L1 > L0 | L4 > L3 > L2 > L0 > L1 |
| 2010 | RandomForest | L2 > L3 > L0 > L1 > L4 | L4 > L3 > L2 > L0 > L1 | L3 > L4 > L2 > L0 > L1 | L4 > L3 > L2 > L0 > L1 |
| 2010 | XGBoost | L3 > L2 > L0 > L1 > L4 | L4 > L3 > L2 > L0 > L1 | L4 > L3 > L2 > L1 > L0 | L4 > L3 > L2 > L0 > L1 |
| 2015 | RandomForest | L1 > L0 > L2 > L3 > L4 | L4 > L3 > L2 > L0 > L1 | L4 > L3 > L2 > L0 > L1 | L4 > L3 > L2 > L0 > L1 |
| 2015 | XGBoost | L1 > L0 > L2 > L3 > L4 | L4 > L3 > L2 > L0 > L1 | L4 > L3 > L2 > L0 > L1 | L4 > L3 > L2 > L0 > L1 |

(el primero es el protocolo cuya estimación retrospectiva queda más cerca del rendimiento temporal real)

## 4. Robustez

### 4.1 Semillas (L0: KFold random_state 42/0/1/2; L4: KMeans random_state 0/1/2/3) — XGBoost

| T | protocolo | semilla | MAE | R² | top-100 |
|---:|---|---|---|---|---|
| 1995 | L0_aleatorio | base | 3.36 | 0.934 | 0.92 |
| 1995 | L0_aleatorio | 0 | 3.40 | 0.931 | 0.96 |
| 1995 | L0_aleatorio | 1 | 3.35 | 0.935 | 0.95 |
| 1995 | L0_aleatorio | 2 | 3.33 | 0.936 | 0.97 |
| 1995 | L4_loco_cluster | base | 10.08 | 0.596 | 0.82 |
| 1995 | L4_loco_cluster | 1 | 9.67 | 0.623 | 0.66 |
| 1995 | L4_loco_cluster | 2 | 10.60 | 0.533 | 0.87 |
| 1995 | L4_loco_cluster | 3 | 9.50 | 0.629 | 0.90 |
| 2005 | L0_aleatorio | base | 4.32 | 0.912 | 0.99 |
| 2005 | L0_aleatorio | 0 | 4.38 | 0.908 | 0.95 |
| 2005 | L0_aleatorio | 1 | 4.37 | 0.910 | 0.99 |
| 2005 | L0_aleatorio | 2 | 4.34 | 0.911 | 0.99 |
| 2005 | L4_loco_cluster | base | 11.31 | 0.618 | 0.38 |
| 2005 | L4_loco_cluster | 1 | 11.54 | 0.603 | 0.39 |
| 2005 | L4_loco_cluster | 2 | 11.56 | 0.611 | 0.34 |
| 2005 | L4_loco_cluster | 3 | 11.43 | 0.627 | 0.34 |
| 2010 | L0_aleatorio | base | 4.34 | 0.910 | 0.99 |
| 2010 | L0_aleatorio | 0 | 4.30 | 0.907 | 0.96 |
| 2010 | L0_aleatorio | 1 | 4.32 | 0.910 | 0.99 |
| 2010 | L0_aleatorio | 2 | 4.35 | 0.905 | 0.95 |
| 2010 | L4_loco_cluster | base | 11.35 | 0.604 | 0.54 |
| 2010 | L4_loco_cluster | 1 | 11.57 | 0.566 | 0.28 |
| 2010 | L4_loco_cluster | 2 | 11.98 | 0.570 | 0.65 |
| 2010 | L4_loco_cluster | 3 | 11.51 | 0.594 | 0.63 |
| 2015 | L0_aleatorio | base | 4.18 | 0.910 | 0.98 |
| 2015 | L0_aleatorio | 0 | 4.16 | 0.909 | 0.97 |
| 2015 | L0_aleatorio | 1 | 4.17 | 0.911 | 0.98 |
| 2015 | L0_aleatorio | 2 | 4.18 | 0.909 | 0.99 |
| 2015 | L4_loco_cluster | base | 10.89 | 0.602 | 0.36 |
| 2015 | L4_loco_cluster | 1 | 11.15 | 0.579 | 0.54 |
| 2015 | L4_loco_cluster | 2 | 10.96 | 0.613 | 0.82 |
| 2015 | L4_loco_cluster | 3 | 10.98 | 0.610 | 0.72 |

### 4.2 Tc = máximo por composición (en lugar de la mediana), corte principal — XGBoost; n_train = 10215, n_test = 2328

| protocolo | MAE est. | MAE real | Δ MAE [IC] | R² est. | R² real | Δ R² [IC] | top-100 est. | top-100 real |
|---|---|---|---|---|---|---|---|---|
| L0 aleatorio (KFold) | 4.50 | 7.64 | -3.14 [-4.08, -2.28] | 0.902 | 0.352 | 0.550 [0.393, 0.708] | 0.97 | 0.17 |
| L1 composición | 4.34 | 7.64 | -3.31 [-4.22, -2.46] | 0.902 | 0.352 | 0.550 [0.394, 0.709] | 0.96 | 0.17 |
| L2 sistema elemental | 6.94 | 7.64 | -0.70 [-1.83, 0.36] | 0.813 | 0.352 | 0.461 [0.304, 0.621] | 0.90 | 0.17 |
| L3 anfitrión (frac. ≥ 0,05) | 7.74 | 7.64 | 0.10 [-1.02, 1.14] | 0.792 | 0.352 | 0.440 [0.279, 0.600] | 0.82 | 0.17 |
| L4 LOCO (KMeans k=50) | 11.72 | 7.64 | 4.08 [2.70, 5.38] | 0.585 | 0.352 | 0.232 [0.073, 0.394] | 0.53 | 0.17 |

### 4.3 Excluyendo composiciones con año de regla de respaldo (anio_ambiguo), corte principal — XGBoost; n_train = 9689, n_test = 2284

| protocolo | MAE est. | MAE real | Δ MAE [IC] | R² est. | R² real | Δ R² [IC] | top-100 est. | top-100 real |
|---|---|---|---|---|---|---|---|---|
| L0 aleatorio (KFold) | 4.49 | 9.13 | -4.64 [-6.14, -3.32] | 0.905 | -0.202 | 1.108 [0.801, 1.514] | 0.97 | 0.15 |
| L1 composición | 4.34 | 9.13 | -4.79 [-6.28, -3.45] | 0.911 | -0.202 | 1.114 [0.807, 1.515] | 0.97 | 0.15 |
| L2 sistema elemental | 6.88 | 9.13 | -2.25 [-3.89, -0.80] | 0.827 | -0.202 | 1.030 [0.727, 1.440] | 0.91 | 0.15 |
| L3 anfitrión (frac. ≥ 0,05) | 8.65 | 9.13 | -0.48 [-2.20, 1.11] | 0.753 | -0.202 | 0.955 [0.651, 1.371] | 0.80 | 0.15 |
| L4 LOCO (KMeans k=50) | 11.63 | 9.13 | 2.50 [0.65, 4.18] | 0.615 | -0.202 | 0.818 [0.508, 1.224] | 0.44 | 0.15 |
