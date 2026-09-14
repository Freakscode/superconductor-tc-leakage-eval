# H01 - Efecto de diseño para un estimador definido (v3)

Procedencia: `code/v3/design_effect_v3.py | python 3.13.15, numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, xgboost 3.0.5 | ICC(1) one-way ANOVA (Searle m0); model = random intercept, independent families, common marginal variance, intra-family correlation rho; deff = Var(estimator)/(sigma^2/N), N_eff = N/deff; family = constituent-element set; A = data/unique_m.csv, B_v2 = data/datasetB_featurized.npz, B_v3 = data/datasetB_featurized_v3.npz | *_v2_ref = results_v2/design_effect.csv`

## Tabla

| Dataset | N | k | ICC(1) | m̄ | m0 (Searle) | m_w = Σn²/N | N_eff (i) media por filas | N_eff (ii) media de medias | N_eff con 1+ρ(m0−1) (v2) | N_eff (i) sin familia mayor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 21263 | 3365 | 0.8541 | 6.319 | 6.300 | 69.633 | 356.6 (deff 59.62) | 3621 (deff 5.87) | 3847 (deff 5.53) | 508.6 (N=20543) |
| B_v2 | 12440 | 3063 | 0.8457 | 4.061 | 4.057 | 18.091 | 805.0 (deff 15.45) | 3268 (deff 3.81) | 3470 (deff 3.59) | 847.7 (N=12309) |
| B_v3 | 12336 | 3046 | 0.8463 | 4.050 | 4.045 | 17.841 | 808.8 (deff 15.25) | 3249 (deff 3.80) | 3448 (deff 3.58) | 853.0 (N=12205) |

Comprobaciones: ICC y m0 de A y B_v2 coinciden con results_v2/design_effect.csv (ICC 0.854112 / 0.845661; N_eff Searle 3847.2 / 3470.0) y con el informe (m_w 69.633 / 18.091; N_eff media por filas 356.6 / 805.0).

Fórmulas (modelo de intercepto aleatorio, familias independientes, varianza marginal común σ², correlación intrafamilia ρ):
- (i) media ponderada por filas: Var = σ²/N² [(1−ρ)N + ρΣn_i²] ⇒ deff = 1 + ρ(m_w − 1), N_eff = N/deff; límite ρ→1: N²/Σn_i² (= k solo si las familias tienen igual tamaño).
- (ii) media de medias de familia (peso 1/n_i por fila): Var = σ²/k² Σ_i[ρ + (1−ρ)/n_i] ⇒ deff = (N/k²)Σ_i[ρ + (1−ρ)/n_i]; límite ρ→1: N_eff → k.
- pesos generales w_i por fila: Var = σ² Σ_i n_i w_i² [(1−ρ) + ρ n_i] / (Σ_i n_i w_i)².
- La fórmula de v2, 1 + ρ(m0 − 1), no es la varianza de ningún estimador de la media bajo este modelo; su valor (3847 en A) queda cerca del estimador (ii) (3621) por accidente numérico (k grande), no por equivalencia.

Sensibilidad:
- sin la familia mayor (A: n=720, 3.4 % de las filas): N_eff (i) pasa de 357 a 509; en B_v3 (n=131) de 809 a 853. El estimador (i) es muy sensible a la familia dominante; el (ii) apenas (3621 → 3610).
- pesos 1/√n_i: N_eff 2444 (A), 2521 (B_v3), intermedio entre (i) y (ii).

## Párrafo de reemplazo (§3.5 / Tabla 4 / Fig. 4b; retirar del abstract y de la conclusión la 'información equivalente a k familias')

> Treating family identity as a clustering variable, the one-way ANOVA intraclass correlation of $T_c$ is 0.854 (A) and 0.846 (B), and family identity alone accounts in-sample for 87.7 % and 88.4 % of the variance of $T_c$ ($\eta^2$). We report these as descriptions of how concentrated the target is within families. A design effect, by contrast, is only defined for a specified estimator. Under a random-intercept model with independent families, common marginal variance and intra-family correlation $\rho$, the variance of the row-weighted mean of $T_c$ is $\sigma^2/N^2\,[(1-\rho)N + \rho\sum_i n_i^2]$, so its design effect is $1 + \rho(m_w - 1)$ with $m_w = \sum_i n_i^2/N$ (= 69.6 in A, 17.8 in B), not Searle's $m_0$ (6.30, 4.05); the corresponding effective sample size is 357 (A) and 809 (B), and it rises to 509 and 853 when the largest family is removed ($m_w$ falls from 69.6 to 46.8 and from 17.8 to 16.6), i.e. the row-weighted estimator is dominated by the largest families. For the mean of family means (rows weighted $1/n_i$) the design effect is $(N/k^2)\sum_i[\rho + (1-\rho)/n_i]$ and the effective size is 3621 and 3249, close to $k$. These numbers describe the precision of one estimator of the mean $T_c$ under one model; they are not the effective sample size of the learning problem, nor of the MAE of a predictor, and we no longer claim that the datasets carry 'the information of $k$ independent observations'.

Frases a retirar: abstract (l. 56-60 'effective sample size (3847 and 3470) within 14 % of the number of chemical families'), §3.5 (l. 544-550 'carry the statistical information of 3847 independent observations'), Fig. 4b leyenda (l. 612-616), §4.1 (l. 680-682), conclusión (l. 772-775). Ecuación (2): sustituir $m$ por $m_w$ y declarar el estimador.
