# Tabla 2 (robustez) — recalculada con particiones agrupadas realmente independientes

Generado por `run_tabla2_shuffle.py` + `agg_tabla2.py` + `build_comparacion.py`. Las tablas de dispersión entre folds (§3) provienen de `tabla2_dispersion_entre_folds.csv` y `tabla2_fold_mae_por_corrida.csv`, escritas por una celda interactiva sobre las salidas de `agg_tabla2.py` y no por ninguno de los tres scripts nombrados. Todas las cifras salen de las 60 corridas de validacion cruzada registradas en `tabla2_runs_por_semilla.csv`; se redondea solo al imprimir.

## 1. El problema, medido

La leyenda publicada de la Tabla 2 dice que ambos esquemas se repitieron *"over five independent fold assignments (seeds 0-4)"*. Para el brazo agrupado eso es falso: `code/tc_pipeline.py` y `code/datasetB_pipeline.py` construyen el brazo agrupado con `GroupKFold(5)`, sin `shuffle`, que es determinista. La semilla solo entraba por `make_models(seed)`, es decir por el `random_state` del **modelo**.

Verificacion directa (`tabla2_determinismo_groupkfold.json`):

- **Dataset A** (N=21263, 3365 familias): cinco invocaciones de `GroupKFold(5)` devuelven folds **identicos** indice por indice (`true`). Con `GroupKFold(5, shuffle=True, random_state=s)`, s=0..4, se obtienen **5 particiones distintas de 5**: el Jaccard entre los conjuntos de familias del primer fold cae a 0.104-0.122 (media 0.112).
- **Dataset B** (N=12440, 3063 familias): cinco invocaciones de `GroupKFold(5)` devuelven folds **identicos** indice por indice (`true`). Con `GroupKFold(5, shuffle=True, random_state=s)`, s=0..4, se obtienen **5 particiones distintas de 5**: el Jaccard entre los conjuntos de familias del primer fold cae a 0.091-0.143 (media 0.110).

El Jaccard medio observado (0.112 en A, 0.110 en B) coincide con el valor esperado para dos subconjuntos independientes de 1/5 de las familias, (1/25)/(9/25) = 1/9 = 0.111. Es decir: `shuffle=True` aleatoriza de verdad la asignacion familia -> fold, no la perturba levemente.

Efecto colateral que conviene declarar: sin `shuffle`, `GroupKFold` balancea los folds por tamano ([4253, 4253, 4253, 4252, 4252] muestras en A); con `shuffle=True` el balanceo se pierde (p.ej. semilla 0 en A: [4816, 4325, 4424, 3998, 3700]). Ninguno de los dos esquemas filtra familias: las familias compartidas entre train y test son 0 en las 40 corridas agrupadas.

## 2. Tabla 2 publicada vs. Tabla 2 v2

Brazo aleatorio en ambas: `KFold(5, shuffle=True, random_state=s)`. Brazo agrupado: `GroupKFold(5)` en la publicada (particion fija), `GroupKFold(5, shuffle=True, random_state=s)` en la v2 (particion variable). s = 0..4 en ambos casos; desviacion = `np.std(ddof=0)`; la inflacion se calcula emparejada por semilla y luego se promedia.

| Dataset / modelo | MAE aleatorio (K) | MAE agrupado (K) | Inflacion MAE (%) | dR2 |
|---|---|---|---|---|
| **A / XGB** — publicada | 5.25 ± 0.02 | 8.34 ± 0.10 | 59 ± 2 | -0.078 (sin ±) |
| A / XGB — **v2 (shuffle)** | 5.25 ± 0.01 | 8.24 ± 0.07 | 56.9 ± 1.6 | -0.071 ± 0.003 |
| **A / RF** — publicada | 5.41 ± 0.01 | 8.69 ± 0.07 | 61 ± 1 | -0.080 (sin ±) |
| A / RF — **v2 (shuffle)** | 5.41 ± 0.01 | 8.62 ± 0.07 | 59.5 ± 1.4 | -0.076 ± 0.002 |
| **B / XGB** — publicada | 4.03 ± 0.01 | 6.20 ± 0.05 | 54 ± 2 | -0.075 (sin ±) |
| B / XGB — **v2 (shuffle)** | 3.87 ± 0.02 | 6.23 ± 0.04 | 60.8 ± 0.5 | -0.081 ± 0.002 |
| **B / RF** — publicada | 4.46 ± 0.01 | 6.48 ± 0.03 | 45 ± 1 | -0.067 (sin ±) |
| B / RF — **v2 (shuffle)** | 4.32 ± 0.01 | 6.58 ± 0.02 | 52.5 ± 0.6 | -0.081 ± 0.002 |

Fuentes: `tabla2_shuffle.csv` (v2), `results/repeated_seed_intervals.csv` del repo (publicada). El dR2 publicado se toma de las columnas R2 de ese mismo archivo (R2_grouped_mean − R2_random_mean).

## 3. Cuanto cambia la desviacion — y por que no cambia como se esperaba

**Se esperaba que la desviacion del brazo agrupado creciera al dejar variar la particion. Medida, no crece.** La desviacion del MAE agrupado con particion variable es 0.018-0.075 K, o sea 0.61-0.94x la publicada (0.014-0.073 K en la reproduccion del protocolo publicado). Sigue en el mismo orden de magnitud, 0.02-0.08 K.

| Dataset / modelo | σ(MAE agr.) publicada | σ reproducida sin shuffle | σ v2 con shuffle | factor v2 / sin shuffle |
|---|---|---|---|---|
| A / XGB | 0.10 | 0.073 | 0.075 | 1.03x |
| A / RF | 0.07 | 0.016 | 0.066 | 4.07x |
| B / XGB | 0.05 | 0.029 | 0.045 | 1.56x |
| B / RF | 0.03 | 0.014 | 0.018 | 1.35x |

El aumento es de 1.03x a 4.07x (mediana 1.45x) frente al mismo codigo sin shuffle, y en las cuatro combinaciones queda *por debajo* de la desviacion que reporta la tabla publicada. En la inflacion el patron es mixto: A/XGB 1.21x, A/RF 2.47x, B/XGB 0.76x, B/RF 0.94x.

La razon es metodologica, no un error de la correccion: **el MAE out-of-fold agrupado se calcula sobre las N muestras del dataset completo**, con lo que redibujar la asignacion familia -> fold reordena que muestra cae en que fold pero deja casi intacto el promedio global. La particion sí importa mucho, pero su efecto vive *entre folds*, no entre semillas:

| Dataset / modelo | σ entre folds, sin shuffle (K) | σ entre folds, con shuffle (K) | σ del MAE agrupado entre semillas (K) |
|---|---|---|---|
| A / XGB | 0.512 | 0.730 | 0.075 |
| A / RF | 0.566 | 0.876 | 0.066 |
| B / XGB | 0.344 | 0.586 | 0.045 |
| B / RF | 0.447 | 0.603 | 0.018 |

Ahi sí crece al liberar la particion: 1.35x-1.70x. Y es 10-33 veces mas grande que la desviacion entre semillas. El MAE de un fold agrupado individual va de 4.90 K a 10.55 K. Esa es la cifra que un lector debe ver si quiere saber cuanto depende el resultado de que familias quedan fuera; la barra ± de la Tabla 2, con cualquiera de los dos protocolos, no lo dice.

## 4. Reproduccion del protocolo publicado (control)

Corriendo `GroupKFold(5)` sin shuffle con `make_models(s)`, s=0..4 — el protocolo publicado tal cual:

| Dataset / modelo | MAE agr. publ. | MAE agr. repro. | MAE aleat. publ. | MAE aleat. repro. | Infl. publ. (%) | Infl. repro. (%) |
|---|---|---|---|---|---|---|
| A / XGB | 8.34 | 8.43 | 5.25 | 5.25 | 59 | 60.5 |
| A / RF | 8.69 | 8.69 | 5.41 | 5.41 | 61 | 60.7 |
| B / XGB | 6.20 | 6.19 | 4.03 | 3.87 | 54 | 59.7 |
| B / RF | 6.48 | 6.47 | 4.46 | 4.32 | 45 | 49.9 |

Dataset A reproduce dentro de 0.09 K. **Dataset B no reproduce el brazo aleatorio**: [4.46, 4.03] K publicados frente a [4.32, 3.87] K reproducidos a partir de `data/datasetB_featurized.npz` con `make_models()` de `code/datasetB_pipeline.py`. El brazo agrupado de B sí reproduce (≤0.03 K). No hay en el arbol ningun script que genere `results/repeated_seed_intervals.csv`, asi que no se puede auditar de donde salieron los 4.03/4.46 K. Por eso la inflacion publicada de B (54% y 45%) sube a 60% y 50% con el *mismo* protocolo publicado: el desplazamiento de B viene del brazo aleatorio, no del shuffle.

## 5. Rango global de inflacion del MAE (para re-anclar el abstract)

- Medias por combinacion (v2, shuffle): A/RandomForest 59.5%, A/XGBoost 56.9%, B/RandomForest 52.5%, B/XGBoost 60.8%.
- **Rango de las medias: 52.5%-60.8%.**
- Rango sobre las 20 corridas individuales: 51.9%-61.5%.
- Rango publicado (abstract actual): 45%-61%.
- Rango con el protocolo publicado reproducido aqui: 49.9%-60.7%.

El extremo inferior sube de 45% a 52%: el 45% publicado era B/RF, y su brazo aleatorio es precisamente el que no reproduce. El extremo superior se mantiene en ~61%. **El abstract del v2 debe decir 52%-61%, no 45%-61%.**

## 6. Que poner en el v2

Leyenda de la Tabla 2 (reemplazo literal propuesto):

> Values are mean ± s.d. over five independent fold assignments (seeds 0-4): both arms use `shuffle=True` with `random_state` = seed, so the seed re-draws the partition — `KFold(5, shuffle=True, random_state=s)` for the random arm and `GroupKFold(5, shuffle=True, random_state=s)` (scikit-learn >= 1.4) for the chemical-family arm — and it also sets the model's `random_state`. No chemical family is shared between training and test folds in the grouped arm. The quoted s.d. is the spread of the pooled out-of-fold metric across seeds and is therefore small by construction (<= 0.075 K); the spread that reflects *which* families are held out is the between-fold spread, 0.59-0.88 K, with individual grouped folds ranging from 4.90 K to 10.55 K MAE.

Nota obligatoria de errata / metodo, porque la version publicada del codigo no hacia esto:

> In the originally released code the grouped arm used `GroupKFold(5)` without shuffling, which is deterministic; the seed varied only the model's `random_state`, not the partition. The grouped-arm s.d. reported there (0.03-0.10 K) therefore measured model variance, not partition variance. Re-running with `shuffle=True` shifts the grouped MAE by at most 0.19 K across the four dataset x model combinations and leaves every qualitative conclusion unchanged.

Detalle que un revisor va a mirar: el desplazamiento del MAE agrupado al activar `shuffle` (A/XGB -0.19 K, A/RF -0.06 K, B/XGB +0.04 K, B/RF +0.11 K) equivale a 0.96x-6.20x la barra ± de esa misma celda: la excede en 2 de las 4 combinaciones (A/XGB 2.6x, B/RF 6.2x) y la iguala en las otras dos. Es coherente con lo anterior: la barra ± no cubre la variacion por particion, y por eso no debe leerse como intervalo de confianza del MAE agrupado.

Ademas: el brazo aleatorio de Dataset B en la Tabla 2 publicada (4.03 / 4.46 K) no se reproduce desde el codigo archivado (3.87 / 4.32 K). Hay que corregir esos dos valores y, con ellos, las inflaciones de B.

## Archivos

- `tabla2_shuffle.csv` — Tabla 2 v2 (entregable principal)
- `tabla2_noshuffle_control.csv` — control: protocolo publicado reproducido
- `tabla2_runs_por_semilla.csv` — las 60 corridas crudas
- `tabla2_determinismo_groupkfold.json` — evidencia del paso 1
- `tabla2_crecimiento_desviacion.csv` — factores de crecimiento de σ
- `tabla2_control_reproduccion.csv` — publicada vs reproducida
- `tabla2_dispersion_entre_folds.csv` — σ entre folds por esquema
- `tabla2_fold_mae_por_corrida.csv` — MAE por fold, corrida a corrida
- `tabla2_inflacion_por_semilla.csv` — inflacion emparejada, semilla a semilla
- `tabla2_rango_inflacion.json` — rango global de inflacion
