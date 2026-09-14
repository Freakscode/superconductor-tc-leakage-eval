# H11 - Dispersión de Tc entre filas con vector de features idéntico (Dataset A)

Procedencia: `code/v3/replicate_dispersion.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5 | Dataset A data/train.csv; replicate group = identical 81-feature vector (pandas.util.hash_pandas_object of the feature columns, as code/v2/run_twins_noise.py), groups with >= 2 rows | model errors from results_v3/oof_v3.csv.gz (code/v3/oof_v3.py, seed 42)`

Sobre las 8511 filas (2418 grupos) de A cuyas 81 features coinciden exactamente con al menos otra fila:

- desviación absoluta media a la MEDIA del grupo: 3.4913 K (el '3.49 K' de v2);
- desviación absoluta media a la MEDIANA del grupo: 3.2229 K (la constante que minimiza el error absoluto);
- ambas usan la propia etiqueta de la fila. Baseline de réplica excluida (mediana de las OTRAS réplicas): MAE = 4.7184 K; con la media de las otras: 4.8486 K.
- MAE del modelo XGBoost (KFold aleatorio, semilla 42) sobre esas filas: 4.8311 K; sobre el resto: 5.4967 K; GroupKFold sobre esas filas: 7.8077 K.

## Redacción propuesta (sustituye 'piso de ruido de 3.49 K')

> Entre las 8511 filas de A que comparten exactamente el mismo vector de 81 descriptores (2418 grupos), la desviación absoluta media de Tc respecto a la mediana de su grupo es 3.22 K (respecto a la media, 3.49 K). Llamamos a esta cantidad *dispersión observada no explicada por la representación*: es la variación de Tc que ningún modelo determinista de estos descriptores puede reproducir en esas filas, pero no es una cota inferior del MAE poblacional ni un error de medición puro, porque se calcula con todas las etiquetas del grupo y puede reflejar estructura, presión, preparación o condiciones no registradas. Un predictor de réplica excluida (mediana de las demás réplicas del grupo) alcanza un MAE de 4.72 K en esas filas, frente a 4.83 K del XGBoost con partición aleatoria.
