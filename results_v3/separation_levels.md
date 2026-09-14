# Niveles de separación (H09, exp. 3)

`code/v3/separation_levels.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5 | XGBoost = make_models(42)['XGBoost'] of tc_pipeline (A) / datasetB_pipeline (B), paper descriptors, n_jobs=8 | L0 KFold(5, shuffle, rs=42); L1 GroupKFold(5) on normalised composition (fractions rounded 1e-3); L2 GroupKFold(5) on element set (== splits fold_deterministic); L3 GroupKFold(5) on host = elements with fraction >= thr (0.05; sensitivity 0.02, 0.10); L4 GroupKFold(5) on KMeans(50, random_state=0, n_init=10) clusters of standardised R1 | NN: 1-NN euclidean in R2 standardised with StandardScaler fitted on the TRAIN fold (nn_*_trainscaler) or on all rows (nn_*_globalscaler, as run_diagnostics_v2.py) | exact_*: share of test rows whose key (R2 vector bytes / composition 1e-3 / element set / host 0.05) occurs in train | top100: K=100, p90(train), >= | intervals vs L0: paired bootstrap of whole element-system families (n_boot=2000, default_rng(0)), conditional on OOF predictions`


## Dataset A (n=21263, k=3365 sistemas elementales)

| nivel | grupos | test medio | test con vector R2 idéntico en train | misma composición 1e-3 | mismo sistema | mismo anfitrión 0.05 | mediana d(1-NN) scaler-train | scaler-global | MAE [IC fam] | R² [IC fam] | top-100 | ΔMAE vs L0 [IC fam] | ΔR² vs L0 [IC fam] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0_aleatorio_filas | 21263 | 4253 (4252–4253) | 37.0 % | 40.3 % | 93.3 % | 94.9 % | 0.057 | 0.057 | 5.230 [4.876, 5.594] | 0.9283 [0.9181, 0.9374] | 0.984±0.012 | — | — |
| L1_composicion_1e-3 | 14612 | 4253 (4252–4253) | 0.0 % | 0.0 % | 88.1 % | 92.2 % | 0.183 | 0.184 | 5.466 [5.090, 5.849] | 0.9233 [0.9114, 0.9342] | 0.974±0.015 | +0.236 [+0.121, +0.359] | -0.0049 [-0.0105, -0.0003] |
| L2_sistema_elemental | 3365 | 4253 (4252–4253) | 0.0 % | 0.0 % | 0.0 % | 73.7 % | 1.454 | 1.456 | 8.313 [7.573, 9.073] | 0.8546 [0.8297, 0.8752] | 0.870±0.110 | +3.083 [+2.583, +3.653] | -0.0737 [-0.0941, -0.0578] |
| L3_anfitrion_0.05 | 2371 | 4253 (4252–4253) | 0.0 % | 0.0 % | 50.9 % | 0.0 % | 1.291 | 1.287 | 8.950 [8.279, 9.632] | 0.8356 [0.8113, 0.8557] | 0.840±0.198 | +3.720 [+3.246, +4.192] | -0.0927 [-0.1119, -0.0775] |
| L3_anfitrion_0.02 | 2963 | 4253 (4252–4253) | 0.0 % | 0.0 % | 46.9 % | 71.0 % | 1.225 | 1.222 | 8.470 [7.786, 9.167] | 0.8548 [0.8373, 0.8708] | 0.836±0.155 | +3.239 [+2.813, +3.702] | -0.0735 [-0.0856, -0.0622] |
| L3_anfitrion_0.10 | 1791 | 4253 (4252–4253) | 0.0 % | 0.0 % | 54.9 % | 57.1 % | 1.584 | 1.614 | 10.733 [9.829, 11.803] | 0.7767 [0.7466, 0.8045] | 0.834±0.228 | +5.503 [+4.788, +6.354] | -0.1515 [-0.1757, -0.1288] |
| L4_LOCO_kmeans50 | 50 | 4253 (4242–4258) | 0.0 % | 0.0 % | 12.6 % | 46.1 % | 2.080 | 2.061 | 10.876 [9.937, 11.892] | 0.7725 [0.7364, 0.8023] | 0.698±0.224 | +5.645 [+4.944, +6.465] | -0.1558 [-0.1881, -0.1291] |

## Dataset B (n=12336, k=3046 sistemas elementales)

| nivel | grupos | test medio | test con vector R2 idéntico en train | misma composición 1e-3 | mismo sistema | mismo anfitrión 0.05 | mediana d(1-NN) scaler-train | scaler-global | MAE [IC fam] | R² [IC fam] | top-100 | ΔMAE vs L0 [IC fam] | ΔR² vs L0 [IC fam] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0_aleatorio_filas | 12336 | 2467 (2467–2468) | 4.9 % | 10.0 % | 87.4 % | 90.4 % | 0.144 | 0.144 | 3.779 [3.487, 4.087] | 0.9319 [0.9226, 0.9397] | 0.960±0.021 | — | — |
| L1_composicion_1e-3 | 11529 | 2467 (2467–2468) | 0.0 % | 0.0 % | 86.1 % | 89.6 % | 0.176 | 0.176 | 3.861 [3.556, 4.190] | 0.9310 [0.9195, 0.9398] | 0.942±0.017 | +0.082 [+0.016, +0.147] | -0.0008 [-0.0061, +0.0034] |
| L2_sistema_elemental | 3046 | 2467 (2467–2468) | 0.0 % | 0.0 % | 0.0 % | 62.6 % | 1.516 | 1.519 | 6.185 [5.579, 6.877] | 0.8477 [0.8198, 0.8690] | 0.824±0.076 | +2.406 [+2.013, +2.882] | -0.0842 [-0.1086, -0.0659] |
| L3_anfitrion_0.05 | 2215 | 2467 (2467–2468) | 0.0 % | 0.0 % | 41.8 % | 0.0 % | 1.353 | 1.351 | 7.127 [6.462, 7.808] | 0.7985 [0.7725, 0.8203] | 0.686±0.141 | +3.348 [+2.912, +3.827] | -0.1334 [-0.1570, -0.1126] |
| L3_anfitrion_0.02 | 2699 | 2467 (2467–2468) | 0.0 % | 0.0 % | 39.5 % | 59.8 % | 1.183 | 1.183 | 6.637 [6.000, 7.311] | 0.8318 [0.8095, 0.8515] | 0.770±0.136 | +2.858 [+2.447, +3.284] | -0.1001 [-0.1184, -0.0836] |
| L3_anfitrion_0.10 | 1751 | 2467 (2467–2468) | 0.0 % | 0.0 % | 39.1 % | 44.8 % | 1.723 | 1.720 | 7.917 [7.239, 8.653] | 0.7789 [0.7485, 0.8043] | 0.644±0.168 | +4.138 [+3.671, +4.633] | -0.1530 [-0.1785, -0.1307] |
| L4_LOCO_kmeans50 | 50 | 2467 (2463–2477) | 0.0 % | 0.0 % | 13.2 % | 32.4 % | 3.015 | 2.941 | 9.083 [8.262, 9.979] | 0.7175 [0.6762, 0.7520] | 0.524±0.229 | +5.304 [+4.669, +5.989] | -0.2144 [-0.2534, -0.1803] |

## Diagnóstico de vecinos (Fig. 1(d)/3(d)) con scaler ajustado solo en train

| dataset | mediana random (global → train) | mediana agrupado (global → train) | razón (global → train) | dif. rel. agrupado | dif. rel. random |
|---|---|---|---|---|---|
| A | 0.0542 → 0.0542 | 1.2727 → 1.2565 | 23.5 → 23.2 | -1.27 % | +0.04 % |
| B | 0.1481 → 0.1487 | 1.5098 → 1.5052 | 10.2 → 10.1 | -0.30 % | +0.42 % |
| B_v2_npz_comparacion | 0.1398 → 0.1401 | 1.5841 → 1.5755 | 11.3 → 11.2 | -0.55 % | +0.22 % |

## Notas
- Cada nivel corresponde a una población objetivo distinta (nuevas medidas de composiciones conocidas → L0/L1; nuevas estequiometrías de sistemas conocidos → L1; sistemas inéditos → L2; anfitriones inéditos → L3; regiones composicionales inéditas → L4); el criterio más estricto no es 'el correcto' por defecto.
- L3: anfitrión = elementos con fracción molar ≥ 0.05 (los dopantes por debajo se ignoran); sensibilidad con 0.02 y 0.10. L4: KMeans(50, semilla 0) sobre R1 estandarizado (sin etiquetas).
- Intervalos: bootstrap pareado de sistemas elementales completos (2 000, semilla 0), condicional a las predicciones OOF; el emparejamiento con L0 usa las mismas familias remuestreadas en ambos niveles.
- Distancias 1-NN en el espacio R2 estandarizado con scaler ajustado SOLO en el fold de entrenamiento; la columna *globalscaler* reproduce el procedimiento de run_diagnostics_v2.py (scaler sobre todo X) para medir la diferencia.
