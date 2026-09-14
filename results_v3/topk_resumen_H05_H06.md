# H05/H06 — Selección top-K corregida: prevalencia real, empates, métricas de selección y negativos medidos

Todos los números proceden de results_v3/topk_prevalence.csv, topk_ties.csv, topk_metrics.csv, topk_metrics_diffs.csv, topk_with_negatives.csv y topk_negatives_summary.json (scripts code/v3/topk_*.py). Operador del umbral: positivo := y >= p90(y del fold de entrenamiento). IC 95 % = bootstrap de familias (2 000 remuestreos, semilla 0), pareado entre condiciones, condicional a las predicciones OOF ya ajustadas (semilla 42, una partición por protocolo).

## 1. Prevalencia real de la clase alto-Tc por fold (esperanza del selector aleatorio)

| Dataset | Protocolo | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | media | mín–máx |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | grouped | 13.80 | 3.62 | 18.22 | 7.48 | 8.51 | 10.33 | 3.62–18.22 |
| A | random | 9.62 | 10.37 | 8.68 | 10.68 | 11.01 | 10.07 | 8.68–11.01 |
| Bv3 | grouped | 8.10 | 6.28 | 11.55 | 13.50 | 11.39 | 10.17 | 6.28–13.50 |
| Bv3 | random | 9.08 | 11.07 | 10.21 | 10.58 | 9.24 | 10.04 | 9.08–11.07 |
| Bv2 | grouped | 9.24 | 7.32 | 17.08 | 7.07 | 10.29 | 10.20 | 7.07–17.08 |
| Bv2 | random | 10.29 | 11.17 | 10.49 | 9.81 | 8.68 | 10.09 | 8.68–11.17 |

El informe adversarial midió A agrupado 13,80/3,62/18,22/7,48/8,51 % y B v2 agrupado 9,24/7,32/17,08/7,07/10,29 %: reproducidos exactamente. B v3 (checkpoint auditado, otra partición de familias) da 8.10/6.28/11.55/13.50/11.39 %. La fracción del fold de ENTRENAMIENTO con y >= p90 tampoco es exactamente 10 % (empates en Tc): 10.00–10.34 % en el conjunto de folds; hasta 16 filas de test coinciden exactamente con el umbral en un fold, por lo que '>' y '>=' difieren.

## 2. Empates (K = 100)

- **Nulo media-de-familia, agrupado, A**: predicción constante por fold (n_distinct_pred = 1), todo el pool empatado (grupo de empate = 4252–4253 filas). Selección publicada np.argsort(pred)[-100:] = últimas 100 filas del fold: media 11.6 % (por fold 31/0/15/8/4). Permutando sólo el orden de las filas (200 veces): la media de folds recorre 7.4–14.4 %. Precisión esperada fraccional = prevalencia del fold: media 10.33 % (200 desempates aleatorios: 10.37 %).
- **Nulo media-de-familia, agrupado, B v3**: predicción constante por fold (n_distinct_pred = 1), todo el pool empatado (grupo de empate = 2467–2468 filas). Selección publicada np.argsort(pred)[-100:] = últimas 100 filas del fold: media 10.6 % (por fold 3/10/11/14/15). Permutando sólo el orden de las filas (200 veces): la media de folds recorre 7.0–13.8 %. Precisión esperada fraccional = prevalencia del fold: media 10.17 % (200 desempates aleatorios: 10.16 %).
- **Nulo media-de-familia, agrupado, B v2**: predicción constante por fold (n_distinct_pred = 1), todo el pool empatado (grupo de empate = 2488–2488 filas). Selección publicada np.argsort(pred)[-100:] = últimas 100 filas del fold: media 9.6 % (por fold 8/6/18/10/6). Permutando sólo el orden de las filas (200 veces): la media de folds recorre 6.4–14.8 %. Precisión esperada fraccional = prevalencia del fold: media 10.20 % (200 desempates aleatorios: 10.15 %).
- **Modelo de presencia (R0), agrupado, A**: empata dentro de familia; grupo de empate en el corte de 3–123 filas; por orden de filas la precisión de un fold varía entre 55 y 100 %; media con la selección argsort del código 84.2 % → fraccional 84.29 %.
- **Modelo de presencia (R0), agrupado, B v3**: empata dentro de familia; grupo de empate en el corte de 7–97 filas; por orden de filas la precisión de un fold varía entre 50 y 83 %; media con la selección argsort del código 68.2 % → fraccional 68.01 %.
- **XGBoost, agrupado, A**: grupo de empate máximo en el corte 21; argsort 87.0 % → fraccional 86.97 %.
- **RandomForest, agrupado, A**: grupo de empate máximo en el corte 16; argsort 85.0 % → fraccional 85.50 %.
- **XGBoost, agrupado, B v3**: grupo de empate máximo en el corte 1; argsort 82.4 % → fraccional 82.40 %.
- **RandomForest, agrupado, B v3**: grupo de empate máximo en el corte 1; argsort 81.0 % → fraccional 81.00 %.

## 3. Métricas de selección, K = 100, media de 5 folds [IC fam 95 %]

| Dataset | Ranker | Split | Prevalencia % | Precisión@100 % | IC filas | Recuperación@100 % | Enriquecimiento | Familias distintas | Máx. cuota familia % | Precisión con <=3/familia % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | null | random | 10.07 | 95.8 [84.3, 97.9] | [93.2, 97.3] | 22.5 [16.2, 30.7] | 9.56 [7.39, 12.35] | 26.8 | 26.2 | 85.9 |
| A | null | grouped | 10.33 | 10.3 [7.7, 12.9] | [9.9, 10.7] | 2.4 [2.2, 2.6] | 1.00 [1.00, 1.00] | 72.8 | 9.4 | 8.8 |
| A | presence | random | 10.07 | 95.5 [85.0, 97.8] | [93.3, 97.2] | 22.4 [16.3, 30.9] | 9.53 [7.37, 12.44] | 26.4 | 26.2 | 86.1 |
| A | presence | grouped | 10.33 | 84.3 [67.2, 89.4] | [79.9, 87.3] | 25.5 [19.6, 32.5] | 10.83 [8.47, 13.29] | 14.2 | 44.0 | 58.0 |
| A | XGBoost | random | 10.07 | 98.4 [93.6, 100.0] | [97.0, 99.4] | 23.1 [17.0, 32.4] | 9.83 [7.68, 12.96] | 33.6 | 20.4 | 94.6 |
| A | XGBoost | grouped | 10.33 | 87.0 [77.2, 92.0] | [83.7, 89.9] | 25.2 [21.1, 34.1] | 10.71 [8.91, 13.83] | 19.0 | 41.6 | 74.2 |
| A | RandomForest | random | 10.07 | 98.4 [92.5, 99.8] | [96.8, 99.4] | 23.1 [16.9, 32.2] | 9.84 [7.68, 12.93] | 30.8 | 21.2 | 93.9 |
| A | RandomForest | grouped | 10.33 | 85.5 [74.4, 90.7] | [81.8, 88.4] | 24.3 [20.1, 33.0] | 10.36 [8.55, 13.38] | 17.8 | 36.2 | 72.4 |
| Bv3 | null | random | 10.04 | 80.7 [72.7, 87.3] | [76.8, 84.0] | 32.7 [26.2, 41.4] | 8.08 [6.63, 9.89] | 43.6 | 12.6 | 75.9 |
| Bv3 | null | grouped | 10.17 | 10.2 [8.1, 12.2] | [9.7, 10.7] | 4.1 [3.8, 4.4] | 1.00 [1.00, 1.00] | 79.8 | 5.4 | 9.4 |
| Bv3 | presence | random | 10.04 | 79.7 [73.7, 87.7] | [76.0, 83.8] | 32.3 [26.3, 41.8] | 7.97 [6.66, 9.99] | 46.8 | 15.0 | 77.7 |
| Bv3 | presence | grouped | 10.17 | 68.0 [56.2, 76.6] | [63.2, 71.9] | 28.8 [23.1, 37.7] | 7.11 [5.81, 8.87] | 21.0 | 21.4 | 57.0 |
| Bv3 | XGBoost | random | 10.04 | 96.0 [93.0, 98.0] | [94.0, 97.4] | 39.0 [31.3, 50.2] | 9.61 [7.96, 11.98] | 51.8 | 7.4 | 94.8 |
| Bv3 | XGBoost | grouped | 10.17 | 82.4 [75.1, 88.8] | [79.0, 85.8] | 34.8 [29.7, 45.6] | 8.57 [7.55, 10.75] | 26.8 | 25.0 | 71.4 |
| Bv3 | RandomForest | random | 10.04 | 95.4 [92.0, 97.6] | [93.2, 97.2] | 38.7 [31.0, 49.7] | 9.55 [7.93, 11.87] | 51.8 | 8.0 | 94.4 |
| Bv3 | RandomForest | grouped | 10.17 | 81.0 [73.6, 88.4] | [77.6, 84.9] | 34.1 [29.4, 45.1] | 8.42 [7.43, 10.60] | 26.0 | 24.2 | 70.8 |

Diferencias pareadas (puntos porcentuales de precisión@100, bootstrap de familias pareado):

| Dataset | Comparación | Δ pp | IC 95 % | fracción bootstrap > 0 |
|---|---|---:|---:|---:|
| A | XGBoost random - grouped | +11.4 | [+6.9, +19.8] | 1.000 |
| A | RandomForest random - grouped | +12.9 | [+8.0, +22.7] | 1.000 |
| A | presence random - grouped | +11.2 | [+5.8, +25.5] | 1.000 |
| A | XGBoost - null (grouped) | +76.6 | [+67.6, +81.7] | 1.000 |
| A | XGBoost - presence (grouped) | +2.7 | [-0.8, +14.2] | 0.955 |
| A | RandomForest - presence (grouped) | +1.2 | [-3.3, +12.1] | 0.855 |
| A | XGBoost - RandomForest (grouped) | +1.5 | [-1.1, +5.6] | 0.880 |
| A | presence - null (grouped) | +74.0 | [+57.7, +79.1] | 1.000 |
| Bv3 | XGBoost random - grouped | +13.6 | [+7.8, +19.6] | 1.000 |
| Bv3 | RandomForest random - grouped | +14.4 | [+7.6, +20.8] | 1.000 |
| Bv3 | presence random - grouped | +11.7 | [+6.3, +21.5] | 1.000 |
| Bv3 | XGBoost - null (grouped) | +72.2 | [+65.6, +78.2] | 1.000 |
| Bv3 | XGBoost - presence (grouped) | +14.4 | [+6.7, +24.5] | 1.000 |
| Bv3 | RandomForest - presence (grouped) | +13.0 | [+5.6, +23.9] | 0.999 |
| Bv3 | XGBoost - RandomForest (grouped) | +1.4 | [-2.8, +4.3] | 0.675 |
| Bv3 | presence - null (grouped) | +57.8 | [+46.5, +66.0] | 1.000 |

Concentración por familia (XGBoost agrupado, K = 100, por fold): familias distintas A: 17/31/7/16/24; Bv3: 34/20/22/30/28; cuota máxima de una familia A: 36/17/75/61/19 %; Bv3: 14/30/27/25/29 %.

Tabla completa por K en topk_metrics.md.

## 4. Negativos medidos (H05) — Dataset B v3 ∪ entradas Tc <= 0

- Origen: 3966 filas con Tc <= 0 en supercon_stanev.csv; todas Tc == 0 exactamente (0 negativas, 0 NaN). Reglas de exclusión de audit_B.py aplicadas: 41 filas caen (elemento_sin_coeficiente_multielemental 22, OY_oxigeno_indeterminado 14, O_final_sin_coeficiente 4, coeficiente_extremo_error_tipografico 1); 0 con NaN Magpie; 3925 negativos featurizados.
- 2328 negativos pertenecen a familias (conjunto de elementos) presentes en B v3 y 1597 a 1113 familias nuevas; 34 tienen la misma composición normalizada que un superconductor de B v3 (etiquetas contradictorias del corpus); 93 son duplicados entre negativos. Elementos sin columna de presencia en B v3: Cm, Po.
- Interpretación de la etiqueta: Las 3 966 filas tienen Tc == 0 exactamente (ninguna negativa, ninguna NaN). El fichero de Stanev (columnas name, Tc) no registra la temperatura minima explorada ni distingue 'medido no superconductor hasta T_min', 'Tc no reportada' y 'medicion censurada'; la literatura que reutiliza el mismo fichero describe estas ~4 000 entradas como 'sin Tc reportada o con ausencia de superconductividad reportada (Tc = 0 K)' (J. Chem. Inf. Model. 2024, doi:10.1021/acs.jcim.4c01137). Por tanto: 'negativo' aqui = 'registrado como Tc = 0 en SuperCon', un negativo censurado a una temperatura desconocida; no puede afirmarse que sea no superconductor a toda temperatura. Los negativos cuya composicion normalizada coincide con un superconductor de B v3 (conflicto_composicion_con_positivo) son etiquetas contradictorias del propio corpus y se cuentan aparte.

| Modelo (entrenado sólo con positivos) | Split | Negativos en el pool % | Precisión@100 sólo positivos % | Precisión@100 con negativos % [IC fam] | Negativos esperados en top-100 [IC fam] | Intermedios (0 < Tc < umbral) en top-100 | Mediana Tc predicha negativos / positivos (K) | Negativos con predicción >= umbral % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | random | 24.1 | 96.0 | 90.4 [86.0, 94.6] | 6.0 [2.8, 9.0] | 3.6 | 7.8 / 10.4 | 1.32 |
| RandomForest | random | 24.1 | 95.4 | 90.8 [86.6, 94.4] | 4.8 [2.4, 7.8] | 4.4 | 8.6 / 10.7 | 1.15 |
| presence | random | 24.1 | 79.7 | 74.5 [66.0, 82.4] | 10.3 [6.5, 14.0] | 15.2 | 7.6 / 10.6 | 2.34 |
| XGBoost | grouped | 24.1 | 82.4 | 78.2 [70.0, 84.8] | 6.2 [3.8, 8.8] | 15.6 | 8.7 / 10.6 | 1.40 |
| RandomForest | grouped | 24.1 | 81.0 | 77.4 [68.6, 84.8] | 6.0 [3.6, 8.4] | 16.6 | 9.7 / 10.8 | 1.07 |
| presence | grouped | 24.1 | 68.0 | 61.1 [50.6, 71.2] | 10.2 [6.4, 14.0] | 28.6 | 9.3 / 10.7 | 2.96 |

Por K (XGBoost agrupado): K=10: precisión 94.0 → 94.0 %, negativos 0.0; K=25: precisión 93.6 → 89.6 %, negativos 1.0; K=50: precisión 87.6 → 83.9 %, negativos 2.7; K=100: precisión 82.4 → 78.2 %, negativos 6.2; K=200: precisión 74.1 → 71.1 %, negativos 12.6.

Ejemplos de negativos (Tc = 0 registrada) que entran en el top-100 agrupado de XGBoost (desempate estable), ordenados por Tc predicha: Hg1Ba2Ca4Cu5O12.32 (107 K); Y1Ba2Cu2.97Al0.03O7 (91 K); Y1Ba2Cu2.985Al0.015O7 (91 K); Y1Ba2Cu3O6.975 (91 K); Tm1Ba2Cu3O1 (90 K); Bi1.6Pb0.35V0.05Sr2Ca2Cu3O9.92 (90 K); Hg0.5Pb0.5Sr3Ba1Cu3C1O7 (89 K, familia nueva); Bi2Sr2Ca1Cu2O8.255 (88 K). Varios son estequiometrías de cupratos conocidos (p. ej. Y1Ba2Cu3O6.975, Tm1Ba2Cu3O1) cuya Tc = 0 es más probablemente un error de captura o una fase no superconductora por O que un no superconductor de la familia; el fichero no permite decidirlo.

## 5. Frases para el manuscrito v3 (sustituyen a 'more than four out of five attempts' y al 'base rate 10 % by construction')

EN (sección 2.6, definición): "We define the high-Tc class as test compounds whose measured Tc is at or above (>=) the 90th percentile of Tc in the training fold. Because families differ in their Tc distribution, the prevalence of that class in the held-out pool is not 10 %: it ranges from 3.6 % to 18.2 % across the five family-held-out folds of Dataset A (mean 10.3 %) and from 6.3 % to 13.5 % on Dataset B (mean 10.2 %). We therefore report precision@K together with the fold prevalence (the expected precision of a random selector) and their ratio, the enrichment. Ties in the predicted Tc are resolved by the expected precision under a uniformly random ordering of the tied compounds; the family-mean null under grouped validation predicts a constant per fold, so its precision equals the fold prevalence by construction (10.3 % on A, 10.2 % on B; the 11.6 % and 9.6 % of the previous version were artefacts of the row order)."

EN (sección 3.6, resultado): "Under family-held-out validation the full model attains a retrospective precision@100 of 87.0 % (95 % family-bootstrap CI 77.2–92.0) on Dataset A and 82.4 % (75.1–88.8) on Dataset B, against fold prevalences of 10.3 % and 10.2 % (enrichment 10.7 and 8.6; recall@100 25 % and 35 % of the high-Tc compounds in the pool). This is retrospective precision among corpus materials whose families were held out of training: it measures how well the ranker recovers already-reported high-Tc compounds of unseen element sets, not the probability that a synthesis succeeds. The 100 selections are not 100 independent discoveries either: they come from 19 (A) and 27 (B) distinct families on average, a single family supplies up to 75 % of a fold's top-100, and limiting the selection to three compounds per family lowers precision@100 to 74.2 % and 71.4 %."

EN (sección 3.6, negativos): "Dataset B, like every SuperCon-derived regression set, contains only materials reported as superconducting. When the 3925 SuperCon entries recorded as Tc = 0 are returned to the held-out pools (24 % of the pool), the regressor trained on positives only places 6.2 of them (3.8–8.8) in each top-100 and precision@100 falls to 78.2 % (70.0–84.8); a further 16 selections are superconductors below the high-Tc threshold. The source records a single value, Tc = 0, without the lowest temperature explored, so measured non-superconductivity, absence of a report and censoring cannot be separated; these entries are censored negatives of unknown floor, and a discovery-oriented evaluation would need a candidate universe with interpretable negatives."

EN (contribución de los descriptores, sustituye '7.7 ± 1.0 points'): "With the same family multiset resampled in both arms, the descriptors add +2.7 points (-0.8 to +14.2) over element presence on Dataset A and +14.4 points (+6.7 to +24.5) on Dataset B at K = 100; on Dataset A the interval includes zero for this partition, so the descriptor contribution to screening is established on B and only suggestive on A."

EN (random vs grouped): "Random splitting overstates precision@100 by +11.4 points (+6.9 to +19.8) on A and +13.6 points (+7.8 to +19.6) on B (paired family bootstrap)."

Tabla 5 (valores nuevos, %, media de folds, fraccional bajo empates; entre corchetes IC fam 95 %):

| Predictor | Split | A | B (v3) |
|---|---|---:|---:|
| Family-mean null | Random | 95.8 [84.3, 97.9] | 80.7 [72.7, 87.3] |
| Family-mean null | Chemical-family | 10.3 [7.7, 12.9] | 10.2 [8.1, 12.2] |
| Element presence | Random | 95.5 [85.0, 97.8] | 79.7 [73.7, 87.7] |
| Element presence | Chemical-family | 84.3 [67.2, 89.4] | 68.0 [56.2, 76.6] |
| Full model (XGBoost) | Random | 98.4 [93.6, 100.0] | 96.0 [93.0, 98.0] |
| Full model (XGBoost) | Chemical-family | 87.0 [77.2, 92.0] | 82.4 [75.1, 88.8] |
| Full model (RandomForest) | Random | 98.4 [92.5, 99.8] | 95.4 [92.0, 97.6] |
| Full model (RandomForest) | Chemical-family | 85.5 [74.4, 90.7] | 81.0 [73.6, 88.4] |
| Prevalencia del pool (media; mín–máx) | Random | 10.1 (8.7–11.0) | 10.0 (9.1–11.1) |
| Prevalencia del pool (media; mín–máx) | Chemical-family | 10.3 (3.6–18.2) | 10.2 (6.3–13.5) |

---
procedencia: code/v3/topk_metrics.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5, matminer 0.9.2, pymatgen 2025.6.14 | input topk_oof_predictions.csv.gz (seed 42; random = KFold(5, shuffle, rs=42); grouped = GroupKFold(5) deterministic); positive := y >= np.percentile(y_train, 90); precision/recall/enrichment fractional under ties (topk_common.expected_hits); n_families/max_share/budget = mean over 200 random tie-breaks (RandomState(0)); budget = at most 3 rows per family; CI = percentile bootstrap of families of the test pools, 2000 resamples, RandomState(0), same family multiset for all conditions of a dataset (paired); CI_row = i.i.d. row bootstrap within each pool; all intervals conditional on the fitted OOF predictions; aggregate = mean over 5 folds || code/v3/topk_negatives.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5, matminer 0.9.2, pymatgen 2025.6.14 | negatives = rows of data/supercon_stanev.csv with Tc <= 0, audited with the generic rules of code/v3/audit_B.py, Magpie ElementProperty('magpie') as datasetB_pipeline.featurize(); positives = data/datasetB_featurized_v3.npz; folds = splits/datasetB_folds_v3.csv fold_deterministic (grouped; negatives follow their family, new families balanced deterministically) / KFold(5, shuffle, rs=42) on positives + RandomState(42) blocks for negatives (random); models make_models(42) of code/datasetB_pipeline.py fitted on POSITIVES ONLY (n_jobs=8); positive := y >= p90(y_train); expected counts fractional under ties; CI = family bootstrap of the extended pool (2000, RandomState(0)), conditional on fitted predictions || code/v3/topk_prevalence_ties.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5, matminer 0.9.2, pymatgen 2025.6.14 | seed 42; random = KFold(5, shuffle=True, random_state=42); grouped = GroupKFold(5) deterministic (== splits fold_deterministic); family = constituent-element set; threshold = np.percentile(y[train fold], 90), positive := y >= thr; n_jobs=8 | K=100; legacy = np.argsort(pred)[-K:] as in tc_pipeline.topk_precision / datasetB_pipeline (ties resolved by row position); rowperm = same selection after 200 random permutations of the test rows (RandomState(0)); fractional = expected precision under a uniformly random order of the tied rows (topk_common.expected_hits); random_tiebreak = mean of 200 explicit random tie-breaks (RandomState(0))