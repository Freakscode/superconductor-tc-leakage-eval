# Efecto vs. ruido entre semillas — nulo de presencia de elementos (v2)

Procedencia de cada número de este documento: `results_v2/run_v2_measurements.py`,
hiperparámetros importados con `importlib` desde `code/tc_pipeline.py` (Dataset A:
XGB n_est=250, depth=8, lr=0.07, sub=0.9, col=0.7, gamma=0, alpha=0.01) y
`code/datasetB_pipeline.py` (Dataset B: XGB n_est=300, depth=10, lr=0.05, sub=0.8,
col=0.8). Agrupamiento = conjunto de elementos constituyentes vía `pandas.factorize`.
Métrica: precisión top-100, umbral = percentil 90 del fold de **entrenamiento**,
promediada sobre 5 folds. Semillas = [42, 0, 1, 2, 3], usadas a la vez para el modelo
y para la asignación de folds (`GroupKFold(5, shuffle=True, random_state=seed)`).
Veredicto emitido por `effect_vs_seed_noise` de la skill *measure-before-you-claim*.

---

## Dataset A — veredicto literal

> **`exceeds seed noise - reportable`**

| cantidad | valor |
|---|---|
| `label` | Dataset A: top-100 agrupado, presencia de elementos vs features completas |
| `baseline_mean` (features completas) | 90.88 % |
| `variant_mean` (sólo presencia) | 83.16 % |
| `effect` | -7.72 pts |
| `seed_sd` | 3.49 pts |
| `effect_over_noise` | **2.21×** |
| `paired.mean_paired_diff` | -7.72 pts |
| `paired.sd_paired_diff` | 1.04 pts |
| `paired.n_seeds_favouring_variant` | 0 de 5 |

Números por semilla (top-100 %, GroupKFold-5 con folds remezclados):

| semilla | features completas (81) | sólo presencia (86) | brecha |
|---|---|---|---|
| 42 | 88.60 | 80.60 | 8.00 |
| 0 | 92.40 | 86.20 | 6.20 |
| 1 | 95.60 | 88.40 | 7.20 |
| 2 | 88.80 | 80.40 | 8.40 |
| 3 | 89.00 | 80.20 | 8.80 |
| **media** | **90.88** | **83.16** | **7.72** |

## Dataset B — veredicto literal

> **`exceeds seed noise - reportable`**

| cantidad | valor |
|---|---|
| `label` | Dataset B: top-100 agrupado, presencia de elementos vs features completas |
| `baseline_mean` (features completas) | 84.60 % |
| `variant_mean` (sólo presencia) | 70.24 % |
| `effect` | -14.36 pts |
| `seed_sd` | 1.46 pts |
| `effect_over_noise` | **9.85×** |
| `paired.mean_paired_diff` | -14.36 pts |
| `paired.sd_paired_diff` | 2.65 pts |
| `paired.n_seeds_favouring_variant` | 0 de 5 |

| semilla | features completas (132) | sólo presencia (83) | brecha |
|---|---|---|---|
| 42 | 83.40 | 71.40 | 12.00 |
| 0 | 83.80 | 70.40 | 13.40 |
| 1 | 84.60 | 71.60 | 13.00 |
| 2 | 86.00 | 67.20 | 18.80 |
| 3 | 85.20 | 70.60 | 14.60 |
| **media** | **84.60** | **70.24** | **14.36** |

## Contraste A vs B — la brecha de B es mayor que la de A (misma prueba, sobre las brechas apareadas)

> **`exceeds seed noise - reportable`**

| cantidad | valor |
|---|---|
| brecha media en A | 7.72 pts |
| brecha media en B | 14.36 pts |
| diferencia (B − A) | 6.64 pts |
| `seed_sd` | 2.01 pts |
| `effect_over_noise` | **3.30×** |
| semillas en que B > A | 5 de 5 |

Los rangos no se solapan: brechas de A ∈ [6.2, 8.8] pts,
brechas de B ∈ [12.0, 18.8] pts
(separación mínima 3.2 pts).

---

## Qué se puede afirmar

1. **En ambos datasets la brecha en ranking es real.** Quitar toda la información
   estadística y de Magpie, dejando sólo indicadores binarios de presencia de elemento,
   cuesta 7.7 ± 1.0 pts de precisión
   top-100 en A (2.2× el ruido entre semillas) y
   14.4 ± 2.7 pts en B
   (9.9× el ruido). En las 5 semillas el modelo completo gana
   en los dos datasets (0 y
   0 semillas favorecen al nulo, de 5).
2. **La brecha de B es aproximadamente el doble de la de A, y eso también supera el ruido:**
   6.6 pts de diferencia, 3.3× la
   dispersión entre semillas, con rangos disjuntos. La afirmación cualitativa del manuscrito
   —"la presencia de elementos basta mucho más en A que en B"— **se sostiene**.
3. **El nulo de presencia queda muy por encima del nulo de media-de-familia bajo CV agrupada.**
   Con familias retenidas, la media de familia colapsa a
   11.6 % (A) y 9.6 % (B) de precisión
   top-100 —no puede consultar familias no vistas— mientras la presencia de elementos
   mantiene 84.2 % y 66.8 %. La identidad química
   *composicional* generaliza fuera de familia; la *tabla de consulta* no.

## Qué NO se puede afirmar

1. **No se puede afirmar "~2 puntos" para A.** La corrida preliminar reportó ~2 pts porque
   usó los folds deterministas de `GroupKFold(5)` sin remezclar: ahí la brecha es
   2.8 pts, pero esa cifra está a **z = -4.8** de la
   media de las 5 semillas (7.7 ± 1.0 pts) y **fuera** del rango observado
   [6.2, 8.8]. Es un artefacto de una única partición
   afortunada, no una propiedad del dataset. La cifra reportable para A es
   **7.7 pts**, no 2.
2. **Tampoco se puede afirmar "18 puntos" para B como número central.** Los folds
   deterministas dan 18.4 pts, dentro del rango de semillas
   [12.0, 18.8] (z = 1.5) pero en su
   extremo alto. La cifra reportable es **14.4 ± 2.7 pts**.
3. **No se puede afirmar que "la presencia de elementos casi empata al modelo completo" en A.**
   Con 7.7 pts de brecha a 2.2× el ruido, la
   diferencia es reportable, no un empate. La formulación correcta es que la brecha en A es
   *aproximadamente la mitad* de la de B.
4. **η² no es un R² de validación cruzada.** Es un techo **ORÁCULO** calculado con todos los
   datos (0.8769 en A, 0.8835 en B); no es
   alcanzable por un modelo que no ve la familia de prueba. No debe compararse con las
   cifras de CV como si fuera un competidor.
5. **No se midió** si la brecha proviene de la estequiometría (fracciones) o de los
   descriptores elementales *per se*: el nulo de presencia elimina ambas cosas a la vez.
   Separarlas requeriría un tercer nulo (presencia + fracciones, sin Magpie), que no se corrió.

## Frase lista para el manuscrito

> Bajo validación cruzada agrupada por familia química, un XGBoost entrenado únicamente sobre
> indicadores binarios de presencia de elemento alcanza
> 83.2 % de precisión top-100 en el Dataset A y
> 70.2 % en el Dataset B, frente a
> 90.9 % y 84.6 % del modelo con las features
> completas del paper (medias sobre 5 semillas). La penalización por descartar toda la
> información estadística y de Magpie es de 7.7 ± 1.0
> puntos en A (2.2× la dispersión entre semillas) y
> 14.4 ± 2.7 puntos en B
> (9.9×); la diferencia entre ambas brechas es de
> 6.6 puntos (3.3× el ruido, rangos disjuntos).
> Es decir: en ambos conjuntos la mera lista de elementos presentes recupera la mayor parte
> del poder de ranking, y lo hace en mayor medida en el Dataset A que en el B.
