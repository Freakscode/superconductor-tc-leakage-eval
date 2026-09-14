# Inferencia v3 — H07 (pareado por familias), H08 (Tabla 3), H01 (N efectivo), H11 (dispersión), R² del nulo, sensibilidad A

Entorno: python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5. Datos canónicos: A = data/train.csv + unique_m.csv (N 21 263, k 3 365); B v3 = data/datasetB_featurized_v3.npz (N 12 336, k 3 046). Semilla 42; random = KFold(5, shuffle, rs=42); grouped = GroupKFold(5) determinista (verificado contra splits/). Todos los intervalos: bootstrap PAREADO de familias completas, 2 000 remuestreos, default_rng(0), percentil 2.5/97.5, **condicionales a las predicciones OOF ya ajustadas** (no incluyen la variabilidad de reentrenar; esa la cubren las 5 semillas de la Tabla 2). Umbral alto-Tc = p90 del fold de entrenamiento, operador >= (no interviene en estos resultados).

## 0. Verificación de las predicciones OOF (results_v3/oof_v3.csv.gz, oof_v3_check.csv)

- 10 de 10 métricas con referencia coinciden a < 1e-6 (de hecho ≤ 1e-15): A XGBoost y nulo vs results_v2/nulo_presencia.csv; B v3 XGBoost, RandomForest y nulo vs results_v3/tabla1_B_antes_despues.csv. n_jobs=8 no altera ningún valor.
- A RandomForest (sin referencia v2 a 6 decimales): MAE 5.383 / 8.691 frente a v1 (3 decimales) 5.396 / 8.683; results_v2/consolidated_results.csv no existe (la ruta de la tarea); la referencia usada es results/consolidated_results.csv (v1).
- Cruce con results_v3/oof_predictions.csv.gz (otra línea): A idéntico (XGB 0.0, RF ≤ 9e-14); su B es el checkpoint v2 (N 12 440), no comparable fila a fila con v3.

## 1. H07 — bootstrap pareado por familias (results_v3/paired_family_bootstrap.csv/.md, family_losses_v3.csv)

| Dataset | Modelo | MAE random | MAE grouped | Inflación % | ΔMAE (K) | ΔR² | MAE macro random | MAE macro grouped | Inflación macro % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | XGBoost | 5.23 [4.88, 5.59] | 8.31 [7.57, 9.07] | 58.9 [50.4, 68.6] | 3.08 [2.58, 3.65] | 0.074 [0.058, 0.094] | 5.07 [4.82, 5.32] | 6.43 [6.16, 6.72] | 27.0 [23.8, 30.5] |
| A | RandomForest | 5.38 [5.02, 5.77] | 8.69 [7.91, 9.53] | 61.4 [52.2, 71.5] | 3.31 [2.76, 3.93] | 0.079 [0.061, 0.099] | 5.33 [5.06, 5.60] | 6.69 [6.39, 6.99] | 25.5 [23.2, 28.0] |
| A | FamilyMeanNull | 9.40 [8.59, 10.22] | 29.41 [27.62, 31.26] | 212.8 [189.7, 238.5] | 20.01 [18.43, 21.62] | 0.807 [0.781, 0.837] | 14.30 [13.77, 14.85] | 28.89 [28.43, 29.36] | 102.0 [95.3, 108.8] |
| B | XGBoost | 3.78 [3.49, 4.09] | 6.18 [5.58, 6.88] | 63.7 [54.6, 74.3] | 2.41 [2.01, 2.88] | 0.084 [0.066, 0.109] | 3.50 [3.31, 3.70] | 4.53 [4.29, 4.78] | 29.4 [25.4, 33.8] |
| B | RandomForest | 4.22 [3.91, 4.57] | 6.46 [5.83, 7.19] | 53.0 [44.5, 62.8] | 2.24 [1.84, 2.72] | 0.079 [0.061, 0.105] | 3.95 [3.73, 4.18] | 4.81 [4.56, 5.07] | 21.9 [18.7, 25.2] |
| B | FamilyMeanNull | 8.07 [7.42, 8.72] | 22.49 [21.20, 23.86] | 178.8 [161.3, 197.3] | 14.42 [13.37, 15.52] | 0.772 [0.743, 0.801] | 11.93 [11.45, 12.40] | 20.80 [20.34, 21.29] | 74.3 [68.7, 80.3] |

Cociente de anchuras IC familias (pareado) / IC filas no pareado (esquema v2) y / IC filas pareado:

| Dataset | Modelo | Estadístico | familias | filas no pareado (v2) | filas pareado | fam/filas-no-par | fam/filas-par |
|---|---|---|---:|---:|---:|---:|---:|
| A | XGBoost | inflacion_pct | [50.4, 68.6] | [55.0, 62.9] | [56.7, 61.4] | 2.31 | 3.85 |
| A | XGBoost | dMAE_K | [2.58, 3.65] | [2.92, 3.25] | [2.98, 3.19] | 3.24 | 5.19 |
| A | XGBoost | dR2 | [0.058, 0.094] | [0.067, 0.080] | [0.070, 0.078] | 2.66 | 4.49 |
| A | XGBoost | MAE_random | [4.88, 5.59] | [5.13, 5.33] | [5.13, 5.33] | 3.55 | 3.56 |
| A | XGBoost | MAE_grouped | [7.57, 9.07] | [8.18, 8.45] | [8.18, 8.45] | 5.61 | 5.52 |
| A | RandomForest | inflacion_pct | [52.2, 71.5] | [57.5, 65.7] | [59.3, 63.7] | 2.35 | 4.41 |
| A | RandomForest | dMAE_K | [2.76, 3.93] | [3.14, 3.48] | [3.22, 3.40] | 3.44 | 6.26 |
| A | RandomForest | dR2 | [0.061, 0.099] | [0.072, 0.085] | [0.075, 0.082] | 2.80 | 5.63 |
| A | RandomForest | MAE_random | [5.02, 5.77] | [5.27, 5.48] | [5.27, 5.49] | 3.55 | 3.49 |
| A | RandomForest | MAE_grouped | [7.91, 9.53] | [8.55, 8.82] | [8.55, 8.83] | 5.95 | 5.91 |
| B | XGBoost | inflacion_pct | [54.6, 74.3] | [57.1, 70.1] | [60.1, 67.5] | 1.53 | 2.66 |
| B | XGBoost | dMAE_K | [2.01, 2.88] | [2.20, 2.60] | [2.29, 2.53] | 2.20 | 3.58 |
| B | XGBoost | dR2 | [0.066, 0.109] | [0.074, 0.094] | [0.078, 0.091] | 2.17 | 3.21 |
| B | XGBoost | MAE_random | [3.49, 4.09] | [3.67, 3.90] | [3.67, 3.89] | 2.69 | 2.69 |
| B | XGBoost | MAE_grouped | [5.58, 6.88] | [6.02, 6.34] | [6.03, 6.34] | 4.06 | 4.14 |
| B | RandomForest | inflacion_pct | [44.5, 62.8] | [47.2, 58.8] | [50.2, 56.2] | 1.57 | 3.05 |
| B | RandomForest | dMAE_K | [1.84, 2.72] | [2.03, 2.44] | [2.13, 2.35] | 2.16 | 4.00 |
| B | RandomForest | dR2 | [0.061, 0.105] | [0.069, 0.090] | [0.073, 0.086] | 2.17 | 3.53 |
| B | RandomForest | MAE_random | [3.91, 4.57] | [4.11, 4.35] | [4.11, 4.35] | 2.79 | 2.77 |
| B | RandomForest | MAE_grouped | [5.83, 7.19] | [6.30, 6.63] | [6.30, 6.64] | 4.17 | 4.08 |

Lectura: el IC por familias de la inflación (XGBoost) es 2.3× (A) y 1.5× (B) más ancho que el publicado por filas; el límite inferior sigue lejos de cero (50.4 % en A, 54.6 % en B). El IC por filas no pareado de v2 se reproduce dentro del error Monte Carlo ([55.02, 62.95] vs publicado [55.07, 62.97] en A; B ya no comparable por el cambio de checkpoint). La inflación MACRO (media de MAE por familia) es 27.0 % (A) y 29.4 % (B): menos de la mitad de la inflación pooled. La brecha pooled está dominada por las familias grandes (que pesan por número de filas); para la familia típica la brecha existe pero es menor. Es un resultado nuevo que el manuscrito debe reportar.

## 2. H08 — Tabla 3 en dos bloques (results_v3/tabla3_v3.csv/.md, tabla3_v3_folds.csv, oof_tabla3_v3.csv.gz)

| Bloque | Dataset | Comparación | MAE random | MAE grouped | Inflación % [IC fam.] | ΔR² [IC] | Inflación macro % [IC] | valor v2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| a_fixed | A | publicada: misma config en ambos brazos | 5.230 | 8.313 | 58.9 [50.4, 68.6] | 0.074 [0.058, 0.094] | 27.0 [23.8, 30.5] | 58.9 |
| a_fixed | A | modal v2 (d10,lr0.05,mcw10): misma config en ambos brazos | 5.108 | 8.346 | 63.4 [54.1, 73.6] | 0.076 [0.060, 0.096] | 29.7 [26.4, 33.1] | 63.4 |
| b_nested | A | anidado en ambos brazos | 5.055 | 8.370 | 65.6 [56.2, 75.8] | 0.076 [0.059, 0.095] | 30.4 [27.0, 34.0] | MAE anidado grouped v2 8.370 |
| b_nested_sens | A | anidado en ambos brazos, interno del brazo aleatorio SIN barajar (sensibilidad) | 5.156 | 8.370 | 62.3 [53.2, 72.5] | 0.076 [0.060, 0.096] | 28.3 [25.2, 31.6] | MAE anidado grouped v2 8.370 |
| mixto_v2 | A | MIXTO (v2): anidado agrupado / modal fija aleatoria | 5.108 | 8.370 | 63.9 [54.5, 74.2] | 0.077 [0.060, 0.096] | 30.1 [26.8, 33.4] | 63.9 |
| mixto_v2 | A | MIXTO (v2): anidado agrupado / publicada aleatoria | 5.230 | 8.370 | 60.0 [50.9, 70.5] | 0.075 [0.059, 0.095] | 25.7 [22.6, 29.2] | 60.0 |
| a_fixed | B | publicada: misma config en ambos brazos | 3.779 | 6.185 | 63.7 [54.6, 74.3] | 0.084 [0.066, 0.109] | 29.4 [25.4, 33.8] | 57.6 |
| a_fixed | B | modal v2 (d10,lr0.05,mcw10): misma config en ambos brazos | 3.954 | 6.229 | 57.6 [49.2, 67.1] | 0.083 [0.066, 0.104] | 24.7 [21.0, 28.7] | 52.4 |
| b_nested | B | anidado en ambos brazos | 3.776 | 6.209 | 64.4 [55.4, 74.8] | 0.084 [0.066, 0.107] | 30.4 [26.4, 34.7] | MAE anidado grouped v2 6.226 |
| b_nested_sens | B | anidado en ambos brazos, interno del brazo aleatorio SIN barajar (sensibilidad) | 3.778 | 6.209 | 64.3 [55.2, 74.9] | 0.084 [0.066, 0.107] | 30.2 [26.1, 34.4] | MAE anidado grouped v2 6.226 |
| mixto_v2 | B | MIXTO (v2): anidado agrupado / modal fija aleatoria | 3.954 | 6.209 | 57.0 [48.2, 67.3] | 0.082 [0.065, 0.106] | 24.6 [20.8, 28.7] | 53.1 |
| mixto_v2 | B | MIXTO (v2): anidado agrupado / publicada aleatoria | 3.779 | 6.209 | 64.3 [55.3, 74.7] | 0.084 [0.067, 0.108] | 30.2 [26.2, 34.6] | 59.1 |

- Brazo agrupado anidado de A reproduce v2 exactamente (MAE 8.3698; ganadoras por fold y MAE externo por fold idénticos, tabla3_v3_folds.csv). B v3 rehecho: 6.2085 (v2, checkpoint v2: 6.226).
- Anidado en AMBOS brazos: inflación 65.6 % [56.2, 75.8] en A y 64.4 % [55.4, 74.8] en B. Ganadoras random: d10,lr0.05,mcw1;d10,lr0.05,mcw1;d10,lr0.05,mcw1;d10,lr0.05,mcw1;d10,lr0.05,mcw1 (A); d10,lr0.1,mcw1;d10,lr0.05,mcw1;d10,lr0.1,mcw1;d10,lr0.05,mcw1;d10,lr0.05,mcw1 (B). La región de complejidad seleccionada es la misma en los dos brazos (depth 8–10, lr 0.05–0.1).
- Sensibilidad: con el CV interno del brazo aleatorio SIN barajar (KFold(3) por bloques contiguos), el MAE interno medio sube a 8.66 K en A (frente a 5.44 K barajado) porque las filas de train.csv están ordenadas y los bloques contiguos se comportan casi como particiones por familia; el resultado externo apenas cambia (inflación 62.3 % en A, 64.3 % en B).
- La comparación 'mixta' de v2 (63.9 % A / 53.1 % B) mezcla un brazo anidado con una configuración modal elegida a posteriori; no es 'mismos hiperparámetros'. Los valores correctos para la frase 'mismos hiperparámetros en ambos brazos' son los del bloque (a): publicada 58.9 / 63.7 %, modal 63.4 / 57.6 % (B en v3).

## 3. H01 — efecto de diseño para un estimador definido (results_v3/design_effect_v3.csv, design_effect_v3.md)

| Dataset | ICC(1) | m̄ | m0 | m_w | N_eff (i) media por filas | N_eff (ii) media de medias | N_eff v2 [1+ρ(m0−1)] | N_eff (i) sin familia mayor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.8541 | 6.319 | 6.300 | 69.63 | 357 | 3621 | 3847 | 509 |
| B v3 | 0.8463 | 4.050 | 4.045 | 17.84 | 809 | 3249 | 3448 | 853 |
| B v2 | 0.8457 | 4.061 | 4.057 | 18.09 | 805 | 3268 | 3470 | 848 |

ICC y m0 de v2 se reproducen (0.854112 / 0.845661; 3847 / 3470); m_w y N_eff(i) reproducen el informe (69.63 / 18.09; 356.6 / 805.0). El párrafo de reemplazo está en design_effect_v3.md.

## 4. H11 — dispersión entre réplicas de vector de features (results_v3/replicate_dispersion.csv/.md)

- 8511 filas en 2418 grupos; desviación absoluta media a la MEDIA 3.4913 K (v2 3.4913), a la MEDIANA 3.2229 K (informe 3.2229). Medianas: 1.163 / 0.800 K.
- Baseline de réplica excluida (mediana de las OTRAS réplicas): MAE 4.7184 K (media de las otras: 4.8486 K). XGBoost random sobre esas filas: 4.8311 K (v2 4.8311); XGBoost grouped: 7.8077 K; RF random 4.7187 K.
- Lectura: 3.49 K no es un mínimo (la mediana da 3.22 K) y ninguno de los dos es una cota del MAE poblacional (usan la propia etiqueta). Un predictor sin features que copia la mediana de las otras réplicas iguala o supera al XGBoost aleatorio en esas filas (4.72 vs 4.83 K): sobre las filas duplicadas el modelo aleatorio no hace más que recuperar la etiqueta de las réplicas.

## 5. R² del nulo agrupado — Dataset A (results_v3/null_r2_by_fold.csv/.md)

R² pooled = -0.00684 = 1 − (ΣSSW 24644786 + Σ n_f(ȳ_f − c_f)² 473802) / SST 24948009. Si cada constante coincidiera con la media del test de su fold, R² = +0.0122 (SST contiene la varianza entre medias de fold, 303223). Por fold, ȳ_f − c_f va de -6.89 a +6.48 K; R²_f (ref. media del test) entre -0.0554 y -0.0009.

## 5. R² del nulo agrupado — Dataset B (results_v3/null_r2_by_fold.csv/.md)

R² pooled = -0.00338 = 1 − (ΣSSW 9962199 + Σ n_f(ȳ_f − c_f)² 93968) / SST 10022341. Si cada constante coincidiera con la media del test de su fold, R² = +0.0060 (SST contiene la varianza entre medias de fold, 60141). Por fold, ȳ_f − c_f va de -4.42 a +3.26 K; R²_f (ref. media del test) entre -0.0285 y -0.0006.

## 6. Sensibilidad de la Tabla 1 (A) a las 1 927 filas con O=1 espurio (results_v3/tabla1_A_sin_O1.csv, _inflacion.csv, .md)

| Variante | Modelo | MAE random | MAE grouped | Inflación % [IC] | ΔR² | Tabla 1 completa (MAE r/g, inflación, ΔR²) |
|---|---|---:|---:|---:|---:|---|
| refold | FamilyMeanNull | 8.773 | 27.137 | 209.3 [185.4, 235.8] | 0.8003 | 9.404 / 29.413, 212.8 %, 0.8074 |
| refold | XGBoost | 4.789 | 7.753 | 61.9 [52.8, 72.1] | 0.0779 | 5.230 / 8.313, 58.9 %, 0.0737 |
| refold | RandomForest | 4.962 | 7.833 | 57.9 [48.3, 69.3] | 0.0743 | 5.383 / 8.691, 61.4 %, 0.0786 |
| same_folds | FamilyMeanNull | 8.851 | 27.171 | 207.0 [184.0, 233.3] | 0.8001 | 9.404 / 29.413, 212.8 %, 0.8074 |
| same_folds | XGBoost | 4.806 | 7.809 | 62.5 [51.5, 75.0] | 0.0815 | 5.230 / 8.313, 58.9 %, 0.0737 |
| same_folds | RandomForest | 4.944 | 7.977 | 61.3 [51.1, 73.3] | 0.0788 | 5.383 / 8.691, 61.4 %, 0.0786 |

Lectura: excluir el 9.1 % de filas baja el MAE ≈ 0.44 K (random) y 0.56 K (grouped) en XGBoost (son filas difíciles: el modelo completo comete MAE 9.44 / 13.74 K sobre ellas); la inflación pasa de 58.9 % a 61.9 % y ΔR² de 0.074 a 0.078. La conclusión no cambia de signo ni de magnitud; los descriptores de esas filas siguen siendo los publicados (no se corrigen).

