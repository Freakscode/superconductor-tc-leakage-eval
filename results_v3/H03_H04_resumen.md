# H03 + H04: auditoria quimica de B, checkpoint v3 y solapamiento A/B — resumen ejecutado

Todos los numeros salen de code/v3/{audit_B,regenerate_B_v3,overlap_AB}.py (supercon-repro: Python 3.13, numpy 2.3.3, pandas 2.3.2, scipy 1.16.2, scikit-learn 1.7.2, xgboost 3.0.5, matminer 0.9.2, pymatgen 2025.6.14). 'Medido' salvo indicacion.

## H03 — auditoria semantica (data/audit/semantic_audit_B.csv)

- Fundamento (3DSC, Sommer et al. 2023, Sci. Data 10:816, p. 3, texto obtenido del PDF): la Y final de `Bi4.4Sr3.6Ca2Cu4OY` es una cantidad desconocida de oxigeno, no itrio; SuperCon exige un coeficiente explicito por elemento, incluso 1. 3DSC excluyo 128 entradas por formulas invalidas de este tipo; en el origen completo (16 414 filas) contamos 148 formulas con algun simbolo sin coeficiente (89 `OY`), 107 con Tc>0.
- Base v2: 12 440 filas parseables con Tc>0. Decision final por fila: excluir 104, corregir 3, mantener el resto. v3: N = 12336, k = 3046 (v2: 12 440 / 3 063).
- Excluidas (104): 75 `…OY` (54 con familia v2 erronea por Y espurio, 21 con itrio real ademas del marcador; en todas la estequiometria de O es indeterminada); 7 con O final sin coeficiente (p.ej. `Y1Ba2Cu3F0.825O`, `Bi2Sr2Ca1Cu2O`); 19 con otro elemento sin coeficiente en formula multielemental (p.ej. `BaKBi1O3`, `Ru0.97Co0.03Sr2Gd1CuO8`); 2 errores de captura no recuperables (`Bi1.6Pb0.4Sr2Cu3Ca2O1013`, `Y1Ba2Cu3O6050`); 1 artefacto de fusion (`D0.11Nb1D0.13Nb1`, duplica `D0.13Nb1`, Tc identica).
- Corregidas (3, familia inalterada): `Pr185Ce0.15Cu1O4`→`Pr1.85Ce0.15Cu1O4`; `La1.85Sr0.15Cu98Ni0.02O4`→`La1.85Sr0.15Cu0.98Ni0.02O4`; `Rb3C60C60`→`Rb3C60` (balance de sitio + serie documentada en el propio origen).
- Mantenidas con marca: 25 isotopos D/T (pymatgen→H; Magpie identico al hidruro, verificado), 2 coeficientes cero (extremo de serie; Magpie identico sin el elemento, verificado), 257 aleaciones en % atomico y 70 fulleridos/clatratos (coef>50 legitimo), 111 elementos repetidos por notacion de sitio, 2 filas con Am, 760 filas en 363 grupos duplicados por composicion normalizada (1e-6; 1 463/658 a 1e-3) — misma familia, no cruzan folds agrupados.
- 8 formulas no parseables: 4 recuperables en principio (`Y2C2Br0.5!1.5` con '!'=I; tres `B1Sr2Ca(n-1)CunO2N+3` con O=2n+3) y 4 no (O indeterminado '=z'/'+2', coeficiente negativo). v3 NO las anade.
- Efecto sobre familias: 0 filas retenidas cambian de familia; 17 familias desaparecen (13 contienen Y; 12 existian SOLO por filas OY), 0 nuevas, 0 fusiones. La particion GroupKFold(5) determinista cambia para 9453 de 12 336 filas comunes (desempate por orden de codigos), asi que la comparacion v2/v3 de Tabla 1 incluye ruido de particion.

## Tabla 1 de B antes/despues (results_v3/tabla1_B_antes_despues.csv; KFold(5, shuffle, rs=42) / GroupKFold(5); make_models(42))

| Modelo | Split | v2 MAE & RMSE & R2 | v3 MAE & RMSE & R2 |
|---|---|---|---|
| Family-mean null | random | 8.19 & 13.94 & 0.764 | 8.07 & 13.71 & 0.769 |
| Family-mean null | grouped | 22.67 & 28.74 & -0.003 | 22.49 & 28.55 & -0.003 |
| XGBoost | random | 3.91 & 7.68 & 0.928 | 3.78 & 7.44 & 0.932 |
| XGBoost | grouped | 6.17 & 10.99 & 0.853 | 6.18 & 11.12 & 0.848 |
| RandomForest | random | 4.32 & 8.30 & 0.916 | 4.22 & 8.18 & 0.918 |
| RandomForest | grouped | 6.48 & 11.44 & 0.841 | 6.46 & 11.46 & 0.838 |

Inflacion MAE agrupado/aleatorio: v2 RandomForest: 49.9%; v2 XGBoost: 57.6%; v3 RandomForest: 53.0%; v3 XGBoost: 63.7%. v2 reproduce el manuscrito (XGB 3.91/7.68/0.928 y 6.17/10.99/0.853; nulo 8.19/13.94/0.764 con rs=42). RF agrupado medido 6.48/11.44/0.841 frente a 6.52/11.56/0.838 en el manuscrito (discrepancia menor, ver contradicciones).

## H04 — solapamiento A/B (results_v3/overlap_AB.csv)

| criterio | B v2: filas (frac) / familias con alguna fila | B v3: filas (frac) / familias |
|---|---|---|
| (i) cadena de formula identica | 11835 (95.1%) / 2887 de 3063 | 11810 (95.7%) / 2882 de 3046 |
| (ii) composicion normalizada 1e-3 | 11929 (95.9%) / 2888 de 3063 | 11903 (96.5%) / 2883 de 3046 |
| (iii) composicion normalizada + |dTc|<=0.1 K | 10932 (87.9%) / 2736 de 3063 | 10907 (88.4%) / 2731 de 3046 |
| (iv) sistema elemental (familia) | 11980 (96.3%) / 2891 de 3063 | 11920 (96.6%) / 2883 de 3046 |
| (iv-inv) filas de A cuya familia esta en B | 19746 (92.9%) / 2891 de 3365 | 19728 (92.8%) / 2883 de 3365 |

Los tres numeros del informe (2 891/3 063 familias = 94,4 %; 96,3 % de filas por sistema; 11 835/12 440 cadenas = 95,1 %) se reproducen exactamente sobre v2. No hay metadatos de publicacion en data/, asi que la interseccion por publicacion no es computable.

## Complementos (v3): B_out = 416 filas / 163 familias (3.4 % de B; Tc media 5.7 K, mediana 2.1 K, 89 familias de 1 fila). A_out = 1535 filas / 482 familias (7.2 % de A; Tc media 64.3 K, 222 familias de 1 fila; 683 filas con O final sin coeficiente).

Evaluacion cruzada en espacio Magpie comun (results_v3/overlap_complement_eval.csv; IC 95 % por bootstrap de 2 000 remuestreos de familias):

| entrenado en | evaluado en | n filas / fam | MAE [IC] | R2 [IC] |
|---|---|---|---|---|
| B (Magpie, todas las filas) | A_out (familia NO en B) | 1535 / 482 | 17.68 [16.50, 18.84] | 0.500 [0.413, 0.570] |
| B (Magpie, todas las filas) | A_in (familia SI en B) | 19728 / 2883 | 3.96 [3.41, 4.52] | 0.923 [0.913, 0.932] |
| B (Magpie, todas las filas) | A completo | 21263 / 3365 | 4.95 [4.50, 5.43] | 0.900 [0.890, 0.910] |
| A (Magpie, todas las filas) | B_out (familia NO en A) | 416 / 163 | 3.73 [2.91, 4.63] | 0.615 [0.415, 0.694] |
| A (Magpie, todas las filas) | B_in (familia SI en A) | 11920 / 2883 | 2.54 [2.38, 2.71] | 0.980 [0.976, 0.983] |
| A (Magpie, todas las filas) | B completo | 12336 / 3046 | 2.58 [2.42, 2.75] | 0.978 [0.974, 0.981] |
| B (Magpie, GroupKFold(5) interno) | B out-of-fold agrupado | 12336 / 3046 | 6.18 [5.58, 6.88] | 0.848 [0.820, 0.869] |
| B (Magpie, GroupKFold(5) interno) | B_out filas, predichas out-of-fold agrupado | 416 / 163 | 2.32 [1.58, 3.18] | 0.787 [0.702, 0.882] |
| A (Magpie, GroupKFold(5) interno) | A out-of-fold agrupado | 21263 / 3365 | 7.78 [7.22, 8.32] | 0.863 [0.844, 0.879] |
| A (Magpie, GroupKFold(5) interno) | A_out filas, predichas out-of-fold agrupado | 1535 / 482 | 12.98 [12.00, 13.99] | 0.705 [0.640, 0.760] |

Lectura: (1) el modelo entrenado en A predice B_in con R2 = 0,98 — no es generalizacion, es que el 95 % de las cadenas de B estan en A; (2) sobre los complementos reales el rendimiento cae a R2 0,50 (B→A_out) y 0,61 (A→B_out), con intervalos anchos; (3) los complementos son pequenos y NO representativos (A_out es casi todo cuprato de alta Tc, media 64 K; B_out es casi todo convencional de baja Tc, media 5,7 K), y el 44 % de A_out son formulas con O sin coeficiente (O=1 en unique_m), asi que el numero B→A_out es provisional. Un test temporal o externo sigue siendo necesario.

## Hallazgo colateral (fuera del alcance pedido, medido)

El mismo defecto de nomenclatura existe en Dataset A: 1927 de 21 263 filas (9.1 %) terminan en O sin coeficiente (p.ej. `Y1Ba2Cu3O`, Tc 90 K) y unique_m.csv registra O = 1 en TODAS ellas (1663 con Tc>30 K, 405 familias). La familia no cambia, pero los 81 descriptores de Hamidieh de esas filas se calcularon con una estequiometria de O falsa. Ademas 6 filas con el token `IN` (indio) estan parseadas como I+N (familia espuria Ce|Cu|I|N|Nd|O). Requiere una auditoria de A analoga (no realizada aqui).

## Frase para el manuscrito v3

"A semantic audit of the Dataset-B formulas (Supplementary Table SX; 3DSC, Sommer et al. 2023, p. 3) excluded 104 of 12 440 rows whose stoichiometry is indeterminate under the SuperCon nomenclature (75 trailing 'OY' = unknown oxygen, not yttrium; 29 other count-less elements or capture errors) and corrected three typographical formulas, giving N = 12 336 and k = 3 046 families; 17 families, 12 of which existed only through spurious-yttrium 'OY' rows, disappear. Re-running Table 1 on the corrected checkpoint changes every entry by <= 0.19 K in MAE (<= 0.14 K for the two models), <= 0.24 K in RMSE and <= 0.006 in R2, and the grouped/random MAE inflation is unchanged or larger (XGBoost 58 % -> 64 %; random forest 50 % -> 53 %). Datasets A and B are two curations of the same SuperCon source: 95.7 % of B's formula strings, 96.5 % of its normalised compositions and 96.6 % of its rows' elemental systems (2 883 of 3 046 families) also occur in A, so the agreement between them is robustness across related curations, not independent replication. On the small non-overlapping complements (416 B rows in 163 families absent from A; 1 535 A rows in 482 families absent from B) cross-trained XGBoost models reach R2 = 0.61 [0.42, 0.69] and 0.50 [0.41, 0.57] (family bootstrap), versus 0.98 on the overlapping part."
