# M4 — ¿El ~58 % de inflación es propiedad del dataset o de la afinación bajo CV aleatoria?

Procedencia: `results_v2/run_M4_m2.py` (modos `grid`, `nested`, `strat`, `merge`), entorno `supercon-repro`
(sklearn 1.7.2, xgboost 3.0.5, n_jobs=8, tree_method=hist). Modelo: XGBoost. Configuración publicada importada
de `make_models(seed)["XGBoost"]` (A: `code/tc_pipeline.py`, n=250/d8/lr0.07/ss0.9/cs0.7; B:
`code/datasetB_pipeline.py`, n=300/d10/lr0.05/ss0.8/cs0.8). Grupos = conjunto de elementos vía `pandas.factorize`.
Tablas: `hp_grid_ambos.csv`, `nested_cv_inflacion.csv`, `nested_cv_folds.csv`, `nested_cv_inner_grid.csv`,
`stratified_group_check.csv`. Logs crudos en `_m4_raw/`.

## 1. Protocolo

- Rejilla: max_depth ∈ {4,6,8,10} × learning_rate ∈ {0.05,0.1} × min_child_weight ∈ {1,10}; n_estimators=300;
  subsample/colsample_bytree de la config publicada de cada dataset. 16 configuraciones + la publicada.
- `grid`: cada configuración evaluada con KFold(5, shuffle, rs=42) (aleatoria) y GroupKFold(5) determinista
  (agrupada); MAE pooled out-of-fold; semilla 42.
- `nested`: externo GroupKFold(5) determinista; interno GroupKFold(3) sobre el train del fold externo; se elige
  la configuración con menor MAE interna pooled; se reajusta en el train externo y se predice el test externo.
  MAE agrupada anidada = pooled sobre los 5 tests externos.
- Coste real: A 222 s (grid) + 242 s (nested); B ~260 s + 227 s. No hubo que reducir la rejilla.

## 2. Resultados (K; redondeo solo aquí)

| | A | B |
|---|---|---|
| MAE agrupada **anidada** (hiperparámetros afinados sin fuga) | **8.370** | **6.226** |
| Configs ganadoras por fold externo | d10/lr0.05/mcw10 ×4, d8/lr0.05/mcw10 ×1 | d8/lr0.05/mcw1, d10/lr0.1/mcw1, d8/lr0.1/mcw1, d10/lr0.05/mcw10 ×2 |
| Config modal | d10/lr0.05/mcw10 (4/5) | d10/lr0.05/mcw10 (2/5; el resto 1/5 cada una) |
| MAE aleatoria, config **publicada** | 5.230 | 3.915 |
| MAE aleatoria, config **modal** | 5.108 | 4.066 |
| MAE agrupada GKF(5), config publicada | 8.313 | 6.170 |
| MAE agrupada GKF(5), config modal | 8.346 | 6.198 |
| Inflación publicada, misma config (grid) | 58.9 % | 57.6 % |
| Inflación anidada / aleatoria publicada | 60.0 % | 59.1 % |
| **Inflación anidada / aleatoria modal (comparación limpia)** | **63.9 %** | **53.1 %** |
| Rango de inflación en la rejilla (16 configs) | 28.5 – 68.7 % (mediana 55.1) | 22.6 – 60.7 % (mediana 46.4) |
| Rango de MAE agrupada en la rejilla | 8.306 – 9.194 (≥d6: 8.306 – 8.528) | 6.170 – 7.036 (≥d6: 6.170 – 6.473) |
| Rango de MAE aleatoria en la rejilla | 5.050 – 7.130 | 3.869 – 5.738 |
| Dispersión entre semillas del paper (tabla2_shuffle, GKF shuffle, 5 semillas) | 56.9 ± 1.6 % (54.4 – 59.3) | 60.8 ± 0.5 % (60.3 – 61.5) |

**Cuál es la comparación limpia y por qué.** El numerador anidado (8.370 / 6.226) se obtiene con
hiperparámetros elegidos sin ver las familias de test. Dividirlo por la MAE aleatoria de la configuración
*publicada* mezcla dos efectos (cambio de protocolo + cambio de hiperparámetros: la publicada fue afinada bajo
CV aleatoria). Dividirlo por la MAE aleatoria de la *misma* configuración que gana la CV anidada (la modal)
deja un solo factor variable — el protocolo de partición — y es por tanto la comparación limpia:
**63.9 % (A) y 53.1 % (B)**. Caveat: en B la ganadora modal solo gana 2/5 folds, así que el numerador anidado
mezcla cuatro configuraciones; la fila de la rejilla con la modal en ambos brazos (52.4 %) coincide con el
valor anidado dentro de 0.7 puntos, así que la mezcla no altera la conclusión. Una comparación aún más simétrica
(CV anidada aleatoria vs. CV anidada agrupada) no se corrió; la rejilla completa la aproxima porque la MAE
aleatoria de la mejor config por CV aleatoria (A: d10/lr0.1/mcw1 = 5.050; B: d10/lr0.1/mcw1 = 3.869) es la cota
inferior del denominador y daría 65.7 % (A) y 60.9 % (B).

## 3. Veredicto

1. **La MAE agrupada es casi insensible a los hiperparámetros; la MAE aleatoria no lo es.** Para profundidad ≥ 6
   la MAE agrupada se mueve en una banda de 0.22 K (A) / 0.30 K (B), mientras la aleatoria cae de 5.9 → 5.05 K
   (A) y de 4.6 → 3.87 K (B) al profundizar los árboles. Por eso la inflación *sube* con la complejidad
   (A: 29 % a d4 → 69 % a d10; B: 23 % → 61 %): la profundidad extra solo compra ajuste a las réplicas
   intra-familia que la partición aleatoria deja pasar, no generalización a familias nuevas. Esto generaliza a A
   lo que la auditoría midió solo en B con 4 configuraciones.
2. **Afinar sin fuga no cambia la conclusión.** La CV agrupada anidada elige la misma región de complejidad
   (d8–d10) que la afinación publicada, y su MAE agrupada (8.370 / 6.226) es 0.06 K *peor* que la de la config
   publicada evaluada de forma agrupada (8.313 / 6.170): no hay una configuración "honesta" que cierre la brecha.
   La selección interna es además plana (diferencia entre la mejor y la 5ª mejor config interna: 0.06–0.11 K en
   A, 0.04–0.12 K en B), lo que explica la heterogeneidad de ganadoras en B.
3. **¿Propiedad del dataset o de la afinación?** El *orden de magnitud* (50–65 %) es del dataset: cualquier
   configuración razonable (d ≥ 8, la región que gana en ambas afinaciones) da 55–69 % en A y 46–61 % en B, y la
   afinación sin fuga da 60.0 % / 59.1 % contra el denominador publicado, indistinguible del ~58 % del
   manuscrito. El *valor exacto*, en cambio, sí depende de la configuración más que del ruido de semilla: la
   comparación limpia da 63.9 % en A (+7 puntos sobre la media de tabla2, fuera del rango 54.4–59.3 de 5
   semillas) y 53.1 % en B (−7.7 puntos, fuera de 60.3–61.5). La diferencia supera el ruido entre semillas
   (σ ≈ 0.5–1.6 puntos) en ambos datasets y en direcciones opuestas. Conclusión para el manuscrito: reportar la
   inflación como una banda (~50–65 % para las configuraciones competitivas de ambos datasets) y decir
   explícitamente que la cifra depende de la profundidad porque solo el brazo aleatorio responde a ella; el
   ~58 % no es un artefacto de haber afinado bajo CV aleatoria.
4. Corrección a la auditoría: la config publicada **no** tiene el mayor gap de la rejilla; en A la superan las
   cuatro configs d10 (63–69 %) y en B d10/lr0.1 (58–61 %). El "máximo" de la auditoría era solo el máximo de
   sus 4 configuraciones.

## 4. m2 — Robustez a la estratificación (`stratified_group_check.csv`)

Config publicada, semillas s ∈ {0,1,2}; StratifiedGroupKFold(5, shuffle, rs=s) sobre 5 bins de cuantiles de Tc
vs GroupKFold(5, shuffle, rs=s) vs KFold(5, shuffle, rs=s). Inflación respecto al KFold de la misma semilla.
Familia mayor de A = Ba|Cu|O|Y (720 filas, 3.4 % del dataset, 63.6 % con Tc > 77 K, 11.8 % de todas las filas
con Tc > 77 K). Familia mayor de B: 131 filas (1.05 %), 13.7 % con Tc > 77 K.

| dataset | splitter | MAE (s=0/1/2) | inflación % | frac Tc>77 por fold (min–max, s=0/1/2) | fold de la familia mayor: rango de MAE (1 = peor de 5) |
|---|---|---|---|---|---|
| A | GroupKFold | 8.243 / 8.119 / 8.262 | 57.2 / 54.4 / 57.2 | 0.126–0.237 / 0.134–0.240 / 0.144–0.254 | 2 / 1 / 1 |
| A | StratifiedGroupKFold | 8.491 / 8.270 / 8.401 | 61.9 / 57.2 / 59.8 | 0.135–0.258 / 0.153–0.248 / 0.105–0.252 | 1 / 4 / 1 |
| B | GroupKFold | 6.179 / 6.255 / 6.205 | 60.3 / 61.3 / 60.8 | 0.070–0.126 / 0.071–0.146 / 0.070–0.149 | 3 / 4 / 1 |
| B | StratifiedGroupKFold | 6.175 / 6.117 / 6.303 | 60.2 / 57.7 / 63.3 | 0.072–0.120 / 0.066–0.118 / 0.075–0.149 | 1 / 1 / 1 |
| A / B | KFold (referencia) | 5.245–5.260 / 3.855–3.879 | 0 | 0.172–0.193 / 0.088–0.111 | — |

**Veredicto m2 (una línea):** el 52–61 % no depende de dónde caiga YBCO: estratificar por bins de Tc deja la
inflación en 57–62 % (A) y 58–63 % (B) y no homogeneiza la fracción de Tc > 77 K por fold (A: rango 0.10–0.26
con y sin estratificación) porque la familia mayor es un bloque indivisible de 720 filas mayoritariamente
> 77 K; lo que sí cambia con la posición de YBCO es la MAE *del fold* que la recibe (suele ser el peor: 4 de 6
particiones en A, 7.8–9.8 K), no la MAE pooled.

Contradicción registrada: se esperaba que StratifiedGroupKFold homogeneizara la fracción de alto Tc; no lo
hace en A (el rango por fold incluso se ensancha en 2 de 3 semillas) y la MAE agrupada estratificada es
sistemáticamente ~0.15–0.25 K *mayor* que la GroupKFold de la misma semilla en A.
