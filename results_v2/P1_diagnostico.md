# P1 — Diagnóstico de la discrepancia del top-100 (Dataset B: 81.4 % vs 84.8 %)

**Fecha:** 2026-08-21 · **Entorno:** `supercon-repro` (python 3.13, xgboost 3.0.5,
scikit-learn 1.7.2, numpy 2.3.3, matminer 0.9.2, pymatgen 2025.6.14, arm64-Darwin)
**Datos de respaldo:** `top100_unificado.csv`, `top100_factorial.csv`,
`top100_diagnostico_datos.json`, `live_run_datasetB/datasetB_summary.json`

---

## Resumen ejecutivo

La hipótesis de trabajo era que los 3.4 puntos entre 81.4 % y 84.8 % se explicaban por
dos entradas: **(a)** la definición del umbral de alto-Tc y **(b)** el conjunto
featurizado. **La medición no la respalda.**

| Factor | Efecto medido sobre el top-100 agrupado |
|---|---|
| (a) Umbral: fold de entrenamiento vs dataset completo | **0.4 puntos** (85.2 % → 84.8 %) |
| (b) Featurización: matminer en vivo vs `.npz` | **0.0 puntos exactos** |
| (c) *Etiquetado de grupos* (factor no previsto) | **1.6 puntos** (85.2 % → 83.6 %) |
| (d) *StandardScaler* (factor no previsto) | **1.4 puntos** (85.2 % → 83.8 %) |
| **Residuo sin explicar frente a 81.4 %** | **≈ 2.0–3.8 puntos** |

Los dos factores del enunciado suman **0.4 puntos de 3.4**. Aparecieron dos factores
ocultos más (grupos y scaler) que pesan más que el umbral. Y aun combinándolos todos,
**ninguna de las ocho variantes medidas reproduce el 81.4 % publicado**: el rango
completo observado es **83.4 %–85.2 %**. El valor 81.4 % no es reproducible en esta
máquina con este código.

---

## 1. La ruta de matminer en vivo, ejecutada por primera vez

`code/datasetB_pipeline.py --supercon data/supercon_stanev.csv` completo:

- **Tiempo total: 463 s** (7 min 43 s) de pared.
- **La featurización fue sólo 10.8 s** de esos 463 s — featurizó 16 406 de 16 414
  fórmulas (8 fallos, descartadas por el `except` del script). El comentario del
  docstring («la featurización es monohilo», presentada como el cuello de botella)
  es engañoso: el 97.7 % del tiempo se lo llevan los cuatro CV completos
  (RandomForest + XGBoost × random + grouped), no matminer.
- Resultado: `topk100.grouped_mean = 0.836 ± 0.068`, `random_mean = 0.960`.

### ¿Reproduce el 81.4 ± 7.3 publicado?
**No.** La ruta en vivo da **83.6 % ± 6.8**, es decir **2.2 puntos por encima** del
valor del abstract, con desviación entre folds compatible pero no igual. Tampoco
reproduce el 84.8 %. Cae *entre* los dos valores publicados, lo que descarta la
lectura simple de «81.4 = ruta en vivo, 84.8 = ruta del `.npz`».

---

## 2. El factorial 2×2 (mismos folds, misma semilla, un solo fit por fold)

Para que la comparación sea estructuralmente limpia, cada fold se ajusta **una vez** y
el modelo ajustado se evalúa con **ambos** umbrales: el umbral entra sólo en la
evaluación, nunca en el ajuste, así que la afirmación «mismo modelo, mismos folds» no
depende de la determinancia de XGBoost.

| Umbral | Featurización | top-100 medio | sd entre folds |
|---|---|---|---|
| fold de entrenamiento | matminer en vivo | **85.2 %** | 5.9 |
| dataset completo | matminer en vivo | **84.8 %** | 5.3 |
| fold de entrenamiento | `.npz` distribuido | **85.2 %** | 5.9 |
| dataset completo | `.npz` distribuido | **84.8 %** | 5.3 |

Lectura del factorial:

- **La featurización no mueve nada.** Las filas 1–3 y 2–4 son idénticas hasta el
  último decimal. El efecto principal de (b) es **exactamente 0.000**.
- **El umbral mueve 0.4 puntos**, no 3.4. Y su dirección es la contraria a la
  intuición de «filtrar información infla el número»: el umbral del dataset completo
  da un resultado *más bajo* (84.8 %) que el del fold de entrenamiento (85.2 %). La
  razón es aritmética: el p90 global de Dataset B es 77.0 K, mientras que los p90 por
  fold de entrenamiento son 77.5 / 78.0 / 73.5 / 78.2 / 77.0 K — en 4 de 5 folds el
  umbral local es *más exigente*, pero en el fold 3 baja a 73.5 K y eso regala aciertos.
  El promedio de los dos efectos casi se cancela.
- **No hay interacción.** Umbral × featurización = 0.

---

## 3. Los dos factores que el enunciado no contemplaba

El 2×2 dejaba 3.0 puntos sin explicar, así que medí los dos factores restantes que
separan `datasetB_pipeline.topk` de `null_model_analysis.top100_null_vs_model`. Ambos
están en `top100_factorial.csv` marcados como `CONTROL:`.

### (c) El etiquetado de grupos cambia los folds — 1.6 puntos

`datasetB_pipeline.featurize()` aplica `pandas.factorize` **antes** de la máscara
`Tc>0`. El resultado son etiquetas **con huecos**, en el rango 0..4189 para 3 063
familias reales. El `.npz` distribuido las trae recontadas de forma **contigua**,
0..3062. Las familias son las mismas — verifiqué que la *partición* es idéntica
(`pandas.factorize(gL) == gn` elemento a elemento) — pero `GroupKFold` asigna grupos
por tamaño decreciente y **rompe empates por orden de etiqueta**. Con miles de familias
de tamaño 1–2, el empate es la regla, no la excepción:

- **~1 700–1 900 de las 2 488 filas de test cambian de fold** (por fold:
  1 897 / 1 753 / 1 779 / 1 721 / 1 787).
- Las dos particiones no son ni la misma ni una permutación una de la otra.
- Efecto sobre el top-100 (umbral de fold de entrenamiento, sin scaler):
  **85.2 % con etiquetas contiguas → 83.6 % con etiquetas con huecos.**

Esto es un hallazgo con consecuencias más allá del top-100: **cualquier número de
Dataset B obtenido corriendo `datasetB_pipeline.py` no es comparable fold-a-fold con
cualquier número obtenido desde el `.npz`**, aunque las familias y las features sean
las mismas. El `.npz` no es un simple caché de la ruta en vivo.

### (d) El `StandardScaler` de `null_model_analysis` — 1.4 puntos

`null_model_analysis.top100_null_vs_model` estandariza X por fold; `datasetB_pipeline.topk`
no. XGBoost es invariante a escala en los *cortes* que elige, pero no en el orden exacto
de empates ni en el submuestreo de columnas, así que el ranking cambia:
**85.2 % sin scaler → 83.8 % con scaler** (etiquetas contiguas, umbral de fold).

O sea: los tres factores medibles (umbral, grupos, scaler) llevan el número de 85.2 %
hasta **83.4 %** en el peor caso combinado. Siguen faltando **2 puntos** para el 81.4 %.

---

## 4. ¿Son idénticas la featurización en vivo y el `.npz`?

**Sí, a precisión float32.** Esto es lo que mata la hipótesis (b):

| Comprobación | Resultado |
|---|---|
| Filas × columnas | 12 440 × 132 en ambos — igual |
| Orden de filas | idéntico (`y` exactamente igual, `np.array_equal`) |
| Partición de familias | idéntica |
| `live.astype(float32) == npz` bit a bit | **True** |
| `np.allclose(live, npz, rtol=1e-5)` | **True** |
| Diferencia absoluta máxima | 1.22 × 10⁻⁴ |
| Diferencia **relativa** máxima | **5.9 × 10⁻⁸** |
| Columnas con dif. relativa > 1e-5 | **0 de 132** |

Las 592 814 celdas con diferencia distinta de cero (de 1 642 080) y las 2 columnas /
542 filas con diferencia absoluta > 1e-4 son **puro redondeo de almacenamiento**: el
`.npz` guarda float32, matminer produce float64. Las columnas afectadas son las de
magnitud grande (~10³–10⁴, tipo `MagpieData range MeltingT`), donde 1e-4 absoluto es
1e-8 relativo. **No hay ninguna diferencia real de contenido.** El `.npz` es un caché
fiel de las features; su única infidelidad es el etiquetado de grupos (§3c).

---

## 5. ¿De dónde sale entonces el 81.4 %? Lo que descarté

- **No es no-determinismo.** Dos corridas idénticas dan el mismo vector de folds
  bit a bit; y `n_jobs` ∈ {1, 2, 4, 16} da exactamente el mismo resultado, así que el
  `n_jobs=-1` de `tc_pipeline` frente al `n_jobs=1` de `datasetB_pipeline` no explica nada.
- **No es el desempate del `argsort`.** `argsort(yp)[-100:]` y `argsort(-yp)[:100]`
  seleccionan el mismo conjunto. Hay empates —2 322 predicciones únicas en 2 488 filas del
  fold 0, o sea 166 valores repetidos— pero ninguno cae en la frontera del top-100, así que
  las dos formas de `argsort` coinciden. Comprobado en el fold 0 únicamente.
- **No es la agrupación de Dataset A por regex vs columnas de elemento.**
  `null_model_analysis.element_set_groups` sobre `unique_m["material"]` y
  `tc_pipeline.chemical_families` dan **la misma partición** (3 365 familias, idénticas).
- **Sensibilidad a la semilla: insuficiente.** Con semillas {0,1,2,7,42,123}, el
  top-100 agrupado de B recorre 83.6–86.6 % (sd 0.97) y el de A 86.0–87.4 % (sd 0.44).
  El 81.4 % queda **fuera** de ese rango por más de 2 puntos.
- **Reconstrucción aritmética.** De `mean=0.814, std=0.0728286…` con K=100 se deduce
  Σfolds = 407 y Σfolds² = 33 395; hay 24 vectores enteros compatibles (p. ej.
  [67, 83, 85, 86, 86]). **Ninguno** coincide con el medido [74, 85, 87, 90, 90]. El
  fold peor publicado es 67–76 %; el peor medido es 74 %.

**Conclusión sobre el origen:** el 81.4 % (y el 89.8 % de Dataset A, que tampoco
reproduzco: mido **87.0 %** con folds [88, 70, 100, 97, 80] frente a los publicados
[77, 84, 100, 100, 88]) proceden de un entorno de librerías distinto del actual —
casi con seguridad otra versión mayor de XGBoost, cuyo cambio de algoritmo de
construcción de árboles altera el *ranking* de predicciones aun con semilla fija.
`requirements.txt` sólo pide `xgboost>=2.0`, y aquí corre 3.0.5. **La discrepancia
81.4 vs 84.8 no es una diferencia de definición: es una diferencia de entorno de
ejecución que quedó registrada en dos artefactos generados en momentos distintos.**
Nótese la asimetría: el **84.8 %** de `top100_null_comparator.csv` lo reproduzco hoy
**exactamente** (umbral de dataset completo, etiquetas contiguas, sin scaler), y al
llamar directamente a `null_model_analysis.top100_null_vs_model` obtengo 84.6 % con los
hiperparámetros de A y 83.4 % con los de B — todo dentro de ±1.4 puntos. En cambio el
**81.4 %** de `datasetB_summary.json` no se reproduce por **ninguna** de las ocho rutas
medidas. Es decir: el artefacto del `.npz` sigue siendo reproducible y el de la ruta en
vivo no, lo que sitúa el 81.4 % como el valor obsoleto de los dos.

---

## 6. Definición canónica y recálculo

**Definición canónica fijada:** umbral de alto-Tc = **percentil 90 de `y` en el fold de
ENTRENAMIENTO** (`np.percentile(y[tr], 90)`). Es la única que no filtra información del
conjunto de test hacia el criterio de evaluación: con el p90 global, el umbral que juzga
al fold *k* se calculó usando los propios materiales del fold *k*. La diferencia numérica
es pequeña (0.4 puntos), pero la propiedad metodológica es categórica, y es exactamente
la clase de fuga que este manuscrito denuncia. Se fija además: **sin `StandardScaler`**
(XGBoost no lo necesita), **etiquetas de grupo contiguas** (factorize *después* de
cualquier filtrado de filas) y **K = 100**.

| Dataset | Split | top-100 medio | sd entre folds | Folds |
|---|---|---|---|---|
| A (Hamidieh/UCI, N=21 263) | random | **98.4 %** | 1.2 | 99, 99, 97, 97, 100 |
| A (Hamidieh/UCI, N=21 263) | agrupado | **87.0 %** | 11.0 | 88, 70, 100, 97, 80 |
| B (Stanev/SuperCon, N=12 440) | random | **96.0 %** | 0.6 | 97, 96, 96, 95, 96 |
| B (Stanev/SuperCon, N=12 440) | agrupado | **85.2 %** | 5.9 | 90, 74, 90, 87, 85 |

**Verificación de la implementación** (una referencia distinta por dataset, porque las dos
funciones del repo usan hiperparámetros distintos):

- **Dataset A:** mi harness reproduce `tc_pipeline.topk_precision` **bit a bit** en los 5
  folds agrupados ([88, 70, 100, 97, 80]).
- **Dataset B:** `tc_pipeline.topk_precision` **no** sirve como referencia aquí (usa
  n_estimators=250, los de A). La referencia válida es el `topk` interno de
  `datasetB_pipeline.run`, cuya salida en vivo reproduzco dígito a dígito
  (`grouped_mean = 0.8360000000000001`, `std = 0.068`, con las etiquetas con huecos del script).

Es decir, el cambio de números frente a lo publicado no viene de mi implementación.

---

## 7. Qué debe ir al manuscrito v2

1. **Sustituir los dos valores por uno solo.** Dataset B, XGBoost, split familiar:
   **85.2 % ± 5.9** (media ± sd entre los 5 folds de `GroupKFold(5)`, umbral = p90 del
   fold de entrenamiento). Elimina el 81.4 % del abstract y §3.6, y el 84.8 % de §3.7 y
   Figura 2b. Dataset A, split familiar: **87.0 % ± 11.0** (reemplaza el 89.8 %).
2. **Reportar también el par random** para que la caída sea legible con la misma
   definición: A 98.4 % → 87.0 %; B 96.0 % → 85.2 %.
3. **Declarar la definición explícitamente** en la leyenda de la Figura 2b y en §3.6:
   «umbral de alto-Tc = percentil 90 del Tc del fold de entrenamiento; K=100;
   sin estandarización de features». Sin esa frase el número no es reproducible.
4. **Añadir la nota de entorno**: los valores del v1 se generaron con una versión de
   XGBoost distinta; fijar `xgboost==3.0.5` en `requirements.txt` en lugar de
   `>=2.0`, y declarar que el top-100 tiene una sensibilidad a la semilla de ±1 punto
   (rango 83.6–86.6 % en 6 semillas para B).
5. **Corregir el bug de agrupamiento en `datasetB_pipeline.featurize()`**: mover el
   `pandas.factorize` a *después* de la máscara `Tc>0`, o re-factorizar tras el filtrado.
   Tal como está, los folds del script no coinciden con los del `.npz` distribuido
   (~1 800 de 2 488 filas por fold), lo que hace que las dos rutas del propio repo no
   sean comparables entre sí. **Esto afecta a todos los números de Dataset B del
   manuscrito, no sólo al top-100.**
6. **Matizar la afirmación sobre el coste de la featurización**: son 11 s, no el cuello
   de botella; el coste de `datasetB_pipeline.py` son los 463 s de validación cruzada.
