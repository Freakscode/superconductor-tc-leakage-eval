# Experimento 7 — validación temporal: hallazgos y respuesta a la pregunta

Procedencia: `code/v3/temporal_corpus.py` → `data/temporal_corpus_v3.{csv,npz}`; `code/v3/temporal_validation.py` →
`results_v3/temporal_validation{.csv,_comparacion.csv,_predicciones.csv.gz,_meta.json}`; `code/v3/temporal_report.py` →
`temporal_validation.md`, `_tabla.tex`, `_fig.{png,pdf}`, `_resumen.json`. Entorno supercon-repro (numpy 2.3.3, pandas 2.3.2,
scikit-learn 1.7.2, xgboost 3.0.5, matminer 0.9.2, pymatgen 2025.6.14), n_jobs=8. Tiempos: corpus 13 s (featurización 10.5 s
de 12 710 composiciones); validación completa 2 200 s (4 cortes × (2 ajustes temporales + 5 protocolos × 5 folds × 2 modelos)
+ 3 semillas extra para L0/L4 en los 4 cortes + 2 sensibilidades en el corte principal ≈ 560 ajustes).
Todos los intervalos son bootstrap de familias completas (B = 2000, semilla 0), pareado, condicional a las predicciones OOF /
temporales ya ajustadas. Umbral alto-Tc = percentil 90 de la Tc del entrenamiento, operador ≥. Corte: train = año de primer
reporte < T; test = año ≥ T. Corte principal T = 2010 (regla: el más tardío con ≥ 1 500 filas de test y ≥ 60 % de familias nuevas;
2005 y 1995 también cumplen y se reportan).

## Pregunta
¿Qué protocolo retrospectivo (L0 aleatorio / L1 composición / L2 sistema elemental / L3 anfitrión / L4 LOCO por clúster) anticipa
mejor el rendimiento real sobre superconductores reportados por primera vez DESPUÉS de un corte temporal?

## Respuesta (medida)
**Ninguno lo reproduce.** La validación aleatoria es el peor proxy para R² en los cuatro cortes (sobreestima +0.28 / +0.60 /
+0.58 / +0.42 en 1995 / 2005 / 2010 / 2015; todos los IC excluyen 0) y para MAE en 1995 y 2005 (subestima 7.3 y 4.4 K); en 2010 el peor proxy de MAE es L4 (+3.9 K frente a −3.1 K de L0), y en 2015 (MAE) y 1995 (top-100) L0 es el más cercano. Los protocolos
agrupados quedan sistemáticamente más cerca: el error medio absoluto de la estimación de R² sobre los cuatro cortes cae de forma
monótona con la fuerza del agrupamiento (L0 0.47 → L2 0.39 → L3 0.35 → L4 0.19), y en MAE el más cercano es L3 (2.2 K; L0 3.8 K;
L4 3.5 K). Pero ningún protocolo, aplicado a los datos disponibles en su momento, habría anticipado el R² real a ±0.1 en más de un
corte: L4 lo acierta en 1995 (ΔR² = −0.05 [−0.12, 0.02]) y 2015 (+0.11 [−0.16, 0.29]) y lo sobreestima en 0.28–0.31 en 2005–2010,
además de ser pesimista en MAE después de 2005 (+2.6 / +3.9 / +7.0 K). La dirección del error residual es estable (aleatorio
optimista; LOCO pesimista en MAE). **El hallazgo favorece al protocolo agrupado sólo como mejor aproximación, no como estimador
insesgado del rendimiento prospectivo.**

## Cifras clave (XGBoost; RF en el CSV, misma ordenación)
| T | n_train → n_test | % fam. nuevas | MAE real [IC] | R² real [IC] | top-100 real (prevalencia ≥ p90) | ΔMAE L0 / L2 / L3 / L4 | ΔR² L0 / L2 / L3 / L4 |
|---:|---|---:|---|---|---|---|---|
| 1995 | 6 281 → 6 262 | 74.3 | 10.71 [10.03, 11.35] | 0.650 [0.59, 0.70] | 0.90 (9.2 %) | −7.34 / −5.45 / −4.63 / −0.63* | +0.28 / +0.22 / +0.18 / −0.05* |
| 2005 | 9 254 → 3 289 | 83.6 | 8.72 [7.93, 9.53] | 0.313 [0.14, 0.46] | 0.47 (1.8 %) | −4.41 / −1.71 / −0.63* / +2.58 | +0.60 / +0.50 / +0.45 / +0.31 |
| 2010 | 10 215 → 2 328 | 76.3 | 7.43 [6.67, 8.27] | 0.328 [0.18, 0.47] | 0.17 (0.9 %) | −3.10 / −0.67* / +0.10* / +3.92 | +0.58 / +0.50 / +0.47 / +0.28 |
| 2015 | 11 734 → 809 | 71.8 | 3.87 [2.98, 4.87] | 0.489 [0.32, 0.75] | 0.08 (1.5 %) | +0.31* / +2.62 / +3.57 / +7.01 | +0.42 / +0.34 / +0.30 / +0.11* |

\* el IC del 95 % de la diferencia incluye 0. L1 ≡ L0 salvo partición (corpus deduplicado): ΔMAE/ΔR² difieren de L0 en < 0.2 K / 0.005.

Estratos del test (T = 2010, XGB): familias nuevas (n = 1 777, 625 familias) MAE 8.47 [7.51, 9.50], R² 0.19 [0.01, 0.34];
familias vistas / composición nueva (n = 551) MAE 4.07 [3.35, 4.79], R² 0.83 [0.71, 0.89]; re-mediciones posteriores de
composiciones del entrenamiento (800 reportes) MAE 2.39 [1.62, 3.33], R² 0.80 [0.62, 0.95]. Es decir, el régimen que la
validación aleatoria mide (R² ≈ 0.91) se parece al de las composiciones ya conocidas, no al de lo que se descubre después.

Reponderar las filas OOF de L0 a la mezcla nuevas/vistas del test NO recupera el valor real (R² 0.81 vs 0.33 en 2010; 0.78 vs 0.31
en 2005): las familias "nuevas" dentro de una partición aleatoria no son representativas de las que se descubren después.

## Advertencias que deben ir al manuscrito
1. **top-100 no es transferible como precisión entre épocas.** La prevalencia de Tc ≥ p90(train) entre las composiciones
   posteriores cae de 9.2 % (1995) a 0.9–1.8 % (2005–2015); la precisión real cae de 0.90 a 0.08–0.47 mientras todos los
   protocolos estiman 0.36–0.99. Como lift sobre la prevalencia, el ranking prospectivo es fuerte (26× en 2005, 18× en 2010,
   5× en 2015). Spearman real 0.57–0.78 vs 0.95 retrospectivo (L0).
2. **El R² temporal es frágil.** Excluir las 526 composiciones del entrenamiento (5 %) cuya fecha procede de una regla de respaldo
   (mayoritariamente aleaciones y fases de Chevrel de 1960–1980 con cita sin paréntesis) desplaza las predicciones XGB de la familia
   Bi–S–O (no vista antes de 2010) de ≈ 15 K a ≈ 60 K y lleva el R² real de 0.33 a −0.20 (MAE 7.4 → 9.1 K), mientras las
   estimaciones retrospectivas L0/L4 cambian < 0.3 K y < 0.02 (L3: 1.1 K, 0.05). La extrapolación a familias nuevas depende de
   pocos vecinos composicionales; la métrica más estable es el MAE.
3. **Semillas.** L0 (KFold rs 42/0/1/2): R² 0.905–0.936 en todos los cortes. L4 (KMeans rs 0/1/2/3): R² 0.53–0.63, MAE 9.5–12.0 K;
   la precisión top-100 de L4 oscila 0.28–0.65 (2010) y 0.36–0.82 (2015): 100 ítems son pocos para esa métrica.
4. **Tc = máximo por composición** (T = 2010): real MAE 7.64, R² 0.352; ΔMAE L0 −3.14, L2 −0.70*, L3 +0.10*, L4 +4.08; ΔR² L0
   +0.55, L4 +0.23. Misma ordenación y conclusiones.
5. **Fechas.** La fecha es la del reporte en SuperCon (no la del descubrimiento); 23 % de las composiciones (2 862 de 12 543) tienen varios reportes
   fechados (rango mediano 4 años) y se usa el primero. 570 de 12 543 composiciones tienen fecha de regla de respaldo.
6. **La validación temporal del EDA** (`notebooks/model_summary.json`, N = 15 816) se reproduce exactamente pero usa fórmulas con la
   estequiometría de O inventada (sufijos `-Y`/`+z` borrados; 1 859 materiales), colapsa 1 270 filas no parseables en un solo
   material, incluye Tc ≤ 0 y no aplica la auditoría química; sus R² (0.71/0.56/0.56) no son comparables y no se reutilizan.

## Contradicciones / matices respecto al informe y al manuscrito
- Informe (viabilidad): "55–73 % del test en familias inexistentes" → medido 72–84 % tras la auditoría química y con corte < T / ≥ T
  (el corpus auditado tiene 12 543 composiciones frente a las 13 449 sin auditar; 3 267 familias vs 3 564).
- Informe: "16–28 % son composiciones ya vistas" → a nivel de composición deduplicada es 0 por construcción; a nivel de reporte,
  las re-mediciones de composiciones conocidas son 800 de 3 847 reportes posteriores a 2010 (21 %), coherente.
- Manuscrito v5 (Scope): "Family-held-out validation is a better proxy for discovery than a random split" → confirmado en dirección
  (más cerca en todas las métricas salvo top-100 en 1995 y MAE en 2015), pero debe matizarse: tampoco reproduce el rendimiento
  prospectivo; la frase "the decisive test is prospective" queda ahora respondida con cifras.
- Tarea: "L1 por composición normalizada" → en un corpus con una fila por composición L1 ≡ L0; la distinción L0/L1 sólo es
  informativa en Dataset B (con duplicados), no aquí.
