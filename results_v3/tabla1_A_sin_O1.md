# Sensibilidad de la Tabla 1 (A) a las 1 927 filas con O=1 espurio

Procedencia: `code/v3/tabla1_A_sensitivity.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5 | Dataset A minus the rows whose unique_m.material ends in 'O' without digit (regex O$ after stripping) AND O == 1 in the element columns | models make_models(42) n_jobs=8; FamilyMeanNull = train-fold family mean | refold: KFold(5, shuffle, rs=42) / GroupKFold(5) on the subset; same_folds: splits/datasetA_folds.csv labels restricted | inflation CI: paired family bootstrap 2000, default_rng(0), conditional on OOF`

Filas excluidas: 1927 (9.06 %); N restante 19336, familias 3171 (de 3365). Los descriptores NO se corrigen; solo se excluyen filas.

| Variante | Modelo | MAE random | MAE grouped | Inflación % [IC fam.] | R² random | R² grouped | ΔR² | Tabla 1 completa: MAE r / g, inflación %, ΔR² |
|---|---|---:|---:|---:|---:|---:|---:|---|
| refold | FamilyMeanNull | 8.773 | 27.137 | 209.3 [185.4, 235.8] | 0.7989 | -0.0014 | 0.8003 | 9.404 / 29.413, 212.8, 0.8074 |
| refold | XGBoost | 4.789 | 7.753 | 61.9 [52.8, 72.1] | 0.9291 | 0.8511 | 0.0779 | 5.230 / 8.313, 58.9, 0.0737 |
| refold | RandomForest | 4.962 | 7.833 | 57.9 [48.3, 69.3] | 0.9243 | 0.8501 | 0.0743 | 5.383 / 8.691, 61.4, 0.0786 |
| same_folds | FamilyMeanNull | 8.851 | 27.171 | 207.0 [184.0, 233.3] | 0.7954 | -0.0046 | 0.8001 | 9.404 / 29.413, 212.8, 0.8074 |
| same_folds | XGBoost | 4.806 | 7.809 | 62.5 [51.5, 75.0] | 0.9276 | 0.8461 | 0.0815 | 5.230 / 8.313, 58.9, 0.0737 |
| same_folds | RandomForest | 4.944 | 7.977 | 61.3 [51.1, 73.3] | 0.9249 | 0.8461 | 0.0788 | 5.383 / 8.691, 61.4, 0.0786 |

same_folds/random usa random_fold_seed0 de splits/ (rs=42 no está en el archivo); la variante comparable con la Tabla 1 (rs=42) es refold.

