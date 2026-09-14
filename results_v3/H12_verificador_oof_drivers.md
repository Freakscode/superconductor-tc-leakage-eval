# H12 — verificador por celda, test de mutación, predicciones OOF y drivers versionados

Procedencia global: entorno `supercon-repro` (Python 3.13, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2,
xgboost 3.0.5, matminer 0.9.2); repo `superconductor-tc-leakage-eval` en `main` = `12ca34b` (v2.0);
manuscrito v5 = `paper1_manuscript.tex` de la raíz del repo. Todos los scripts nuevos están en `code/v3/`,
resuelven rutas desde `Path(__file__)` (override `SUPERCON_ROOT`, salidas `SUPERCON_RESULTS_V3`) y se
ejecutaron desde `/tmp`. Cada cifra de este documento sale de una celda ejecutada; se redondea solo aquí.

## 1. Reproducción del hallazgo H12 sobre el verificador v2 (medido)

`code/v2/verify_manuscript.py --tex paper1_manuscript.tex` → `comprobaciones OK: 117 / DISCREPANCIAS: 0`, exit 0.
Con la celda `Published configuration & 5.23 & 8.31` mutada a `99.99`: **117 OK / 0 discrepancias, exit 0**
(confirmado). Sin argumentos desde `/tmp` funciona aquí sólo porque existe la carpeta hermana
`../paper1_submission/`; en un checkout limpio falla (busca fuera del repo).

## 2. Verificador v3 (`code/v3/verify_manuscript.py` + `code/v3/manuscript_numbers.json`)

* Gramática mínima del .tex: entornos `tabular` → filas → celdas (`\multicolumn` expandido, `\toprule/\midrule/
  \bottomrule/\cmidrule` tolerados); en prosa `\num`, `\SI`, `\SIrange`, `\numrange`, números en `$…$` y
  versiones `~x.y.z`, cada uno con línea, sección más cercana y ±40 caracteres de contexto.
* Manifiesto de **490 entradas** (generado por `code/v3/build_manifest.py`): 175 celdas de las 5 tablas
  (`tab:main`, `tab:robust`, `tab:tuning`, `tab:deff`, `tab:screen`) y 315 anclas de prosa. Cada entrada
  se comprueba en SU ubicación (tabla/fila/columna/parte, o ancla de contexto única) contra su fuente:
  CSV/JSON de `results_v2/` (clave + campo + agregación), expresiones sobre varias fuentes (diferencias,
  cocientes, rangos), hiperparámetros leídos con `make_models(42)` de `code/tc_pipeline.py` y
  `code/datasetB_pipeline.py`, versiones leídas de `requirements.txt`, y 25 constantes declaradas con razón.
* Resultado sobre el v5 (medido, `results_v3/verify_report_v5.json`): **463 OK, 25 CONST, 2 DISCREPANCIAS,
  0 NO ENCONTRADA; cobertura 509/509 = 100 % de las cifras numéricas del .tex** (definición de cifra: celda
  numérica de tabla, número dentro de macro siunitx, número en modo matemático, cadena de versión;
  enteros sueltos en texto corrido como "top-100", "90th" o "five" quedan fuera por diseño). Exit 1.
* Sin argumentos desde `/tmp` usa `<repo>/paper1_manuscript.tex` (el .tex de la raíz), como pide H12.

### Las dos discrepancias son reales (medidas, no las suaviza el verificador)

| id | .tex | fuente | valor |
|---|---|---|---|
| `gap_dR2_A` (l.355, §3.2) | "lowers $R^2$ by \num{0.073}" | `nulo_presencia.csv` R2 random − grouped, A | 0.928277 − 0.854601 = **0.0737 → 0.074** (0.073 sale de restar los valores ya redondeados 0.928 − 0.855) |
| `grid_grp_range_d6plus` (l.465, §3.4) | "within \SI{0.2}{\kelvin} once depth is at least 6" | `hp_grid_ambos.csv`, rango de MAE_grouped con depth∈{6,8,10} | A: 8.528 − 8.306 = 0.22 K ✓; **B: 6.473 − 6.170 = 0.30 K → 0.3 K** |

### Cifras cubiertas sólo por declaración (25 CONST, 27 apariciones): base rate 10 % (×5), umbral Tc≤0 / Tc>0 (×4),
R²≈0 / R²=0 de un predictor constante (×3), identidades algebraicas 1, −1 (×4), rango de literatura
R²≈0.93 / 5 K / R²≳0.92 / 6 K (×4), scikit-learn 1.6 (hecho de librería), nivel 95 % del bootstrap, clip 3 de
la figura 1(d), umbral 77 K, primera semilla 0 (×2). Están listadas con su razón en el manifiesto.

### Nota de procedencia (12 entradas OK con `note`): las filas **Random forest** de `tab:main` no tienen
generador v2; su única fuente versionada es la tabla v1 `results/consolidated_results.csv`, contra la que
coinciden. Ver §4: una corrida fresca no las reproduce.

## 3. Test de mutación (`code/v3/test_verify_mutation.py`, semilla 0, `results_v3/verify_mutation_test.json`)

12 mutaciones aleatorias (último dígito +1) entre las entradas OK + los dos casos de la revisión
(5.23→99.99 y 5.23→8.31, valor que sí existe en otra celda de la misma tabla): **v3 detecta 14/14**
(exit 1 y la entrada mutada en DISCREPANCIA/NO ENCONTRADA); **v2 detecta 2/14** (sólo cuando los dígitos
mutados no aparecen en ningún otro sitio del .tex). El .tex sin mutar produce exactamente la baseline
conocida {`gap_dR2_A`, `grid_grp_range_d6plus`}; el test exige `baseline == KNOWN_BASELINE` (quedará vacía
cuando el v3 corrija esas dos frases).

## 4. Predicciones fuera de muestra (`code/v3/dump_oof_predictions.py` → `results_v3/oof_predictions.csv.gz`, 159 692 filas)

Columnas: dataset, model, protocol, seed, row_index, family_id, fold, y_true, y_pred, threshold_p90_train,
config_json, procedencia. Protocolos: `random` = KFold(5, shuffle, rs=42), `grouped` = GroupKFold(5)
determinista, más el control `grouped_v1_prefilter_partition` (B). Folds verificados contra `splits/*.csv`:
`grouped` == `fold_deterministic` (A y B); la semilla 42 del brazo random **no está en splits/** (guarda
semillas 0–4), así que se verificó KFold(rs=0) == `random_fold_seed0` como control de identidad del splitter.

MAE pooled recalculado desde el archivo (`results_v3/oof_summary.csv`):

| dataset | modelo | protocolo | MAE v3 | RMSE v3 | R² v3 | MAE v2 (`nulo_presencia`) | MAE v1 (`results/consolidated_results.csv`) | tex (tab:main) |
|---|---|---|---|---|---|---|---|---|
| A | XGBoost | random | 5.230280 | 9.173 | 0.9283 | 5.230280 (Δ=0) | 5.222 | 5.23 ✓ |
| A | XGBoost | grouped | 8.313166 | 13.061 | 0.8546 | 8.313166 (Δ=0) | 8.394 | 8.31 ✓ |
| A | RandomForest | random | 5.383020 | 9.441 | 0.9240 | — | 5.396 | **5.40 / 9.47 / 0.923 ✗** |
| A | RandomForest | grouped | 8.690711 | 13.466 | 0.8455 | — | 8.683 | **8.68** / 13.47 / 0.845 |
| B | XGBoost | random | 3.914646 | 7.680 | 0.9284 | 3.914646 (Δ=0) | 3.896 | 3.91 ✓ |
| B | XGBoost | grouped | 6.169775 | 10.988 | 0.8534 | 6.169775 (Δ=0) | 6.278 | 6.17 ✓ |
| B | RandomForest | random | 4.322637 | 8.303 | 0.9163 | — | 4.328 | **4.33 / 8.31** / 0.916 |
| B | RandomForest | grouped | 6.481624 | 11.444 | 0.8410 | — | 6.517 | **6.52 / 11.56 / 0.838 ✗** |
| B | XGBoost | grouped, partición v1 | 6.184156 | 11.033 | 0.8522 | — | 6.278 | — |
| B | RandomForest | grouped, partición v1 | 6.483831 | 11.497 | 0.8395 | — | 6.517 | — |

* XGBoost: idéntico bit a bit a v2 en A y B (los 4 MAE, y también umbral/top-100 por fold, ver §6).
* **Random forest: 9 de las 12 celdas de `tab:main` no coinciden a la precisión impresa** con la corrida
  fresca del pipeline versionado en el entorno fijado; las celdas del .tex son los valores v1
  (`consolidated_results.csv`, 3 decimales). El README ya declara que los números B de v1 salieron de la
  partición defectuosa (factorize pre-filtro); pero el control sobre esa partición tampoco reproduce v1
  (RF B grouped 6.484 vs 6.517; XGB 6.184 vs 6.278), luego la diferencia v1↔v3 no es sólo la partición:
  v1 corrió con otras versiones (requirements v1 con `>=`) y/o con la featurización en vivo (float64) en
  lugar del checkpoint float32. Conclusión cualitativa intacta (inflación RF: A 61.4 %, B 49.9 % con v3).
* Contradice la frase del manuscrito v5 (§4.4): "every Dataset-B number in this version was computed on the
  checkpoint partition" — las 6 celdas RF de B no lo fueron (son v1).

## 5. Bootstrap (`code/v3/bootstrap_intervals.py` → `results_v3/bootstrap_intervalos_v3.csv`)

Bootstrap de FILAS de |y−ŷ| pooled, 2000 réplicas, `default_rng(0)`, percentiles 2.5/97.5, XGBoost semilla 42:

* Estimadores puntuales (MAE random/grouped, inflación, MAE por fold min/max) **iguales a
  `results_v2/bootstrap_intervalos.csv` a 1e-9** (las predicciones OOF son las mismas).
* Extremos de los intervalos: el flujo RNG original no es recuperable (se probaron `default_rng`/`RandomState`
  con semillas 0 y 42, apareado y no apareado; ninguno reproduce a <1e-6). El esquema **no apareado**
  (remuestreos independientes para cada brazo) reproduce el CSV v2 dentro del error Monte Carlo:
  |z| ≤ 1.83 en los 12 extremos (máximo en MAE_random_lo de B; sd MC de los extremos de inflación ≈ 0.11–0.17 pp; 20 semillas).
* El esquema **apareado** (mismas filas en ambos brazos, el adecuado para un contraste dentro de fila) da
  intervalos de inflación mucho más estrechos: A **56.7–61.4 %** (publicado 55.1–63.0), B **54.1–61.2 %**
  (publicado 51.4–63.9); los extremos publicados quedan a |z| = 18–34 del apareado. Los intervalos de MAE
  por brazo son iguales en ambos esquemas. El manuscrito describe "2000 resamples of the test rows" sin decir
  que los dos brazos se remuestrean independientemente.
* La columna `seed=42` del CSV v2 es la semilla del modelo, no la del bootstrap.

## 6. Otros resultados declarados interactivos en README/`procedencia` y su estado

| archivo v2 | procedencia declarada | usado por el manuscrito | driver v3 |
|---|---|---|---|
| `bootstrap_intervalos.csv` | ninguna (sin script) | sí (§3.2) | `code/v3/bootstrap_intervals.py` (reproduce puntuales exactos, CI dentro de MC) |
| `top100_unificado.csv` | harness versionado, driver interactivo | sí (tab:screen, fig. 2) | `code/v3/top100_from_oof.py`: XGBoost reproducido exactamente (0.870±0.110 A, 0.852±0.059 B, 0.984, 0.960) |
| `top100_factorial.csv` | "celda" | sí ("0.4 points", §2.6 y §4.4) | `code/v3/top100_from_oof.py`: 0.852 − 0.848 = 0.4 pp reproducido |
| `fix_factorize_verificacion.json` | "verification cell" | sí (16414, 8, 3966, 8937, 1721–1897) | `code/v3/factorize_check.py`: 15/15 campos iguales |
| "64 % above 77 K" ({Ba,Cu,O,Y}) | sin archivo de resultados | sí (§3.4) | `code/v3/data_facts.py`: 63.61 % (Tc>77), 64.17 % (Tc≥77) → "64" ✓ |
| `top100_diagnostico_datos.json` | "cell" | no | sin driver (no alimenta el manuscrito) |
| `tabla2_dispersion_entre_folds.csv`, `tabla2_fold_mae_por_corrida.csv` | "celda" | no directamente (tabla 2 usa `tabla2_shuffle.csv`, con driver) | sin driver |
| `hp_sensitivity_B_gap.csv`, `replicacion_diff.csv`, `tabla2_control_reproduccion.csv`, `tabla2_crecimiento_desviacion.csv`, `tabla2_inflacion_por_semilla.csv` | sin columna `procedencia` | no | sin driver |
| filas Random forest de `tab:main` | sólo v1 | sí | `code/v3/dump_oof_predictions.py` (valores nuevos, no coinciden con el .tex; §4) |

## 7. Archivos

* `code/v3/verify_manuscript.py`, `code/v3/manuscript_numbers.json`, `code/v3/build_manifest.py`,
  `code/v3/test_verify_mutation.py`, `code/v3/dump_oof_predictions.py`, `code/v3/bootstrap_intervals.py`,
  `code/v3/top100_from_oof.py`, `code/v3/factorize_check.py`, `code/v3/data_facts.py`.
* `results_v3/verify_report_v5.json`, `results_v3/verify_mutation_test.json`, `results_v3/oof_predictions.csv.gz`,
  `results_v3/oof_summary.csv`, `results_v3/bootstrap_intervalos_v3.csv`, `results_v3/top100_from_oof.csv`,
  `results_v3/fix_factorize_verificacion_v3.json`, `results_v3/data_facts.csv`.
