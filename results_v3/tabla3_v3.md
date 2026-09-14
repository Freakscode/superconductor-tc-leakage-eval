# Tabla 3 (v3) - inflación del MAE bajo configuración fija y bajo procedimiento anidado (XGBoost)

Procedencia: `code/v3/nested_both_arms.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5 | XGBoost tree_method=hist n_jobs=8 seed 42 | grid = 16 configs of code/v2/run_M4_m2.py (max_depth {4,6,8,10} x lr {0.05,0.1} x min_child_weight {1,10}, n_estimators=300, subsample/colsample_bytree of the published config) | grouped nested: outer GroupKFold(5) det., inner GroupKFold(3) on outer train, winner = min pooled inner MAE | random nested: outer KFold(5, shuffle, rs=42), inner KFold(3, shuffle=True, rs=42); sensitivity nested_random_noshuffle: inner KFold(3) without shuffle | fixed: published = make_models(42)['XGBoost']; modal = modal winner of the v2 grouped nested search | family bootstrap paired, 2000 resamples, default_rng(0), conditional on OOF predictions | B = data/datasetB_featurized_v3.npz (v3)`

Intervalos: bootstrap pareado de familias completas (2000, semilla 0), percentil 95 %, condicionales a las predicciones OOF. Inflación = MAE_grouped / MAE_random − 1. Dataset B = checkpoint v3 (N = 12 336, k = 3 046); los valores v2 de referencia son del checkpoint v2 (N = 12 440).

## (a) Configuración fija en ambos brazos (solo cambia el protocolo de partición)

| Dataset | Configuración | MAE random | MAE grouped | Inflación % [IC fam.] | ΔR² [IC] | Inflación macro % [IC] | v2 (fija) |
|---|---|---:|---:|---:|---:|---:|---:|
| A | publicada: misma config en ambos brazos | 5.230 | 8.313 | 58.9 [50.4, 68.6] | 0.074 [0.058, 0.094] | 27.0 [23.8, 30.5] | 58.9 |
| A | modal v2 (d10,lr0.05,mcw10): misma config en ambos brazos | 5.108 | 8.346 | 63.4 [54.1, 73.6] | 0.076 [0.060, 0.096] | 29.7 [26.4, 33.1] | 63.4 |
| B | publicada: misma config en ambos brazos | 3.779 | 6.185 | 63.7 [54.6, 74.3] | 0.084 [0.066, 0.109] | 29.4 [25.4, 33.8] | 57.6 |
| B | modal v2 (d10,lr0.05,mcw10): misma config en ambos brazos | 3.954 | 6.229 | 57.6 [49.2, 67.1] | 0.083 [0.066, 0.104] | 24.7 [21.0, 28.7] | 52.4 |

## (b) Procedimiento anidado en AMBOS brazos (selección interna de 16 configuraciones dentro del train externo)

| Dataset | MAE anidado random | MAE anidado grouped | Inflación % [IC fam.] | ΔR² [IC] | Inflación macro % [IC] | ganadoras grouped | ganadoras random | v2 MAE anidado grouped |
|---|---:|---:|---:|---:|---:|---|---|---:|
| A | 5.055 | 8.370 | 65.6 [56.2, 75.8] | 0.076 [0.059, 0.095] | 30.4 [27.0, 34.0] | d10,lr0.05,mcw10;d10,lr0.05,mcw10;d10,lr0.05,mcw10;d10,lr0.05,mcw10;d8,lr0.05,mcw10 | d10,lr0.05,mcw1;d10,lr0.05,mcw1;d10,lr0.05,mcw1;d10,lr0.05,mcw1;d10,lr0.05,mcw1 | 8.370 |
| A (sens.: interno sin barajar) | 5.156 | 8.370 | 62.3 [53.2, 72.5] | 0.076 [0.060, 0.096] | 28.3 [25.2, 31.6] | d10,lr0.05,mcw10;d10,lr0.05,mcw10;d10,lr0.05,mcw10;d10,lr0.05,mcw10;d8,lr0.05,mcw10 | d10,lr0.05,mcw10;d10,lr0.05,mcw10;d10,lr0.05,mcw10;d8,lr0.05,mcw10;d10,lr0.05,mcw10 | 8.370 |
| B | 3.776 | 6.209 | 64.4 [55.4, 74.8] | 0.084 [0.066, 0.107] | 30.4 [26.4, 34.7] | d10,lr0.05,mcw10;d10,lr0.05,mcw1;d8,lr0.05,mcw1;d8,lr0.1,mcw1;d10,lr0.05,mcw1 | d10,lr0.1,mcw1;d10,lr0.05,mcw1;d10,lr0.1,mcw1;d10,lr0.05,mcw1;d10,lr0.05,mcw1 | 6.226 |
| B (sens.: interno sin barajar) | 3.778 | 6.209 | 64.3 [55.2, 74.9] | 0.084 [0.066, 0.107] | 30.2 [26.1, 34.4] | d10,lr0.05,mcw10;d10,lr0.05,mcw1;d8,lr0.05,mcw1;d8,lr0.1,mcw1;d10,lr0.05,mcw1 | d10,lr0.05,mcw1;d10,lr0.05,mcw1;d10,lr0.1,mcw1;d10,lr0.05,mcw1;d10,lr0.1,mcw1 | 6.226 |

## Referencia: la comparación MIXTA publicada en v2 (no es 'mismos hiperparámetros')

| Dataset | Comparación | MAE random (fija) | MAE grouped (anidado) | Inflación % [IC fam.] | v2 |
|---|---|---:|---:|---:|---:|
| A | MIXTO (v2): anidado agrupado / modal fija aleatoria | 5.108 | 8.370 | 63.9 [54.5, 74.2] | 63.9 |
| A | MIXTO (v2): anidado agrupado / publicada aleatoria | 5.230 | 8.370 | 60.0 [50.9, 70.5] | 60.0 |
| B | MIXTO (v2): anidado agrupado / modal fija aleatoria | 3.954 | 6.209 | 57.0 [48.2, 67.3] | 53.1 |
| B | MIXTO (v2): anidado agrupado / publicada aleatoria | 3.779 | 6.209 | 64.3 [55.3, 74.7] | 59.1 |
