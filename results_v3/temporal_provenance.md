# Procedencia del corpus temporal v3 (experimento 7)

Procedencia: `{"script": "code/v3/temporal_corpus.py", "python": "3.13.15", "numpy": "2.3.3", "pandas": "2.3.2", "pymatgen": "2025.6.14", "matminer": "0.9.2", "input": "SUPERCON_DATA_RAW (SuperCon/NIMS primary.tsv, no redistribuido)", "anio": "primer (19xx|20xx) entre parentesis del campo journal; fallbacks arXiv YYMM y anio suelto; ventana [1911,2021]; fecha de una composicion = MIN sobre sus reportes", "composicion": "pymatgen fractional_composition redondeada a 1e-3 sobre la columna element", "tc": "mediana de los reportes Tc>0 de la composicion (Tc_max para sensibilidad)", "exclusiones": "reglas de code/v3/audit_B.py (H03) aplicadas a element; Tc<=0 fuera", "familia": "conjunto de elementos (pymatgen, D/T->H, coef 0 omitido); codigo = pandas.factorize de la clave ordenada sobre el corpus final", "featurizacion": "matminer ElementProperty.from_preset('magpie'), 132 columnas, receta de code/datasetB_pipeline.py::featurize()", "segundos_featurizacion": 10.5, "n_composiciones": 12543}`

## 1. Origen y campo de fecha

- Dump SuperCon/NIMS `primary.tsv`: 26321 filas (num, name, element, str3, utc, tc, journal). Unidad distinta de K: 0 filas; Tc<=0: 89; Tc no numerica: 0.
- El unico campo con fecha es `journal` (cadena bibliografica libre). La fecha es la del REPORTE, no la del descubrimiento.
- La columna `element` es la formula exacta; `name` (notacion de sustitucion) no se usa.

## 2. Reglas de extraccion del anio (orden de aplicacion; gana la primera)

| regla | patron | ejemplo | filas (todas) |
|---|---|---|---:|
| R1 paren | `\((19xx|20xx)\)` | `Phys.Rev.B,55(1997)11832` | 24989 |
| R2 arxiv_yymm | `cond-mat/YYMM.nnnn`, `arXiv:YYMM.nnnn` | `cond-mat/1408.2625` -> 2014 | 75 |
| R3 arxiv_old | `cond-mat/YYMMnnn` | `cond-mat/0305503` -> 2003 | 12 |
| R4 bare | 19xx/20xx aislado (no pegado a letras/digitos) | `Springer, 1981, p=288` | 1159 |
| R4 ambiguo | >1 candidato distinto en R4 | -- | 0 |
| fuera_de_rango | anio fuera de [1911,2021] | -- | 0 |
| sin_anio | ninguna regla | `Unpublished results`, `Phys.Lett.A,38()323`, `(19879)`, `(2995)` | 86 |

- Cobertura: 99.67 % de las filas con anio; rango 1913-2021.
- Ambiguedades resueltas: numeros de pagina tipo `L2080` (JJAP Lett.) NO son anios (exigimos no adyacencia a letras/digitos); `cond-mat/0808.2026:PRB079212509` da 2008 por R2 y no 2026 por R4 (R2 precede a R4); `Science, 2000(1978)535` da 1978 (R1 precede a R4).
- `anio_ambiguo` = composicion cuya fecha de primer reporte procede de R2-R4 (regla de respaldo): 570 de 12543 composiciones; se usa en la sensibilidad.

## 3. Exclusiones quimicas (mismas reglas que `code/v3/audit_B.py`, aplicadas a `element`, filas Tc>0)

| categoria | filas Tc>0 | decision |
|---|---:|---|
| ok | 20851 | mantener |
| no_parseada_pymatgen | 4346 | excluir |
| elemento_sin_coeficiente_multielemental | 937 | excluir |
| OY_oxigeno_indeterminado | 84 | excluir |
| O_final_sin_coeficiente | 8 | excluir |
| correccion_documentada | 3 | corregir (3 lecturas tipograficas documentadas) |
| coeficiente_extremo_error_tipografico | 2 | excluir |
| elemento_repetido_artefacto | 1 | excluir |

- Filas Tc>0: 26232; conservadas: 20854 (79.5 %); de ellas con anio: 20771.
- `no_parseada_pymatgen` incluye sobre todo oxigeno indeterminado (`Oz`, `O4-Y`, `O7+d`, `O6+x`): la auditoria H03 no inventa cantidades.
- Composiciones sin descriptores Magpie finitos (simbolos que no son elementos: `M`, `Ls`, `Gs`): 167 (ejemplos: Gd0.09Ls0.91Os2, Gs1, M0.5Ta1S2, M0.125Nb1S2, M0.125Ta1S2).

## 4. Deduplicacion por composicion

- Clave = composicion fraccional redondeada a 1e-3. Composiciones con algun reporte: 12761; con anio: 12710; corpus final: **12543 composiciones, 3267 familias**.
- Fecha = PRIMER reporte (min). 2862 composiciones tienen >1 reporte fechado; entre ellas el rango de anios tiene mediana 4, p75 9, p90 19 (74 % con rango > 0). Usar el primer reporte evita que una composicion antigua re-medida parezca nueva.
- Tc = MEDIANA de los reportes Tc>0 (Tc_max para sensibilidad): coincide con el maximo en 82.8 % de las composiciones; diferencia media 0.42 K.

## 5. Composicion del futuro por corte (nivel composicion; train = anio_primer_reporte < T, test >= T)

| T | n_train | n_test | familias train | familias test | % test en familia NUEVA | reportes posteriores de composicion ya vista / reportes posteriores |
|---:|---:|---:|---:|---:|---:|---:|
| 1995 | 6281 | 6262 | 1822 | 1734 | 74.3 | 1776 / 10770 |
| 2000 | 8061 | 4482 | 2113 | 1406 | 81.0 | 1115 / 7826 |
| 2005 | 9254 | 3289 | 2375 | 1092 | 83.6 | 892 / 5613 |
| 2010 | 10215 | 2328 | 2642 | 798 | 76.3 | 800 / 3847 |
| 2015 | 11734 | 809 | 3021 | 351 | 71.8 | 441 / 1464 |

Tras deduplicar, toda composicion del test es nueva por construccion; la categoria 'composicion vista' solo existe al nivel de reporte (re-mediciones posteriores de composiciones conocidas) y se evalua aparte en `temporal_validation.py`.

## 6. Enlace con los datasets del articulo

- Dataset B v3: 12274 de 12336 filas (99.5 %) y 11473 de 11535 composiciones unicas reciben anio de primer reporte.
- Dataset A (unique_m.csv): 17355 de 21263 filas (81.6 %) y 11448 de 14617 composiciones unicas. La cobertura menor de A se debe a las filas con O sin coeficiente (O=1 en unique_m.csv) y a la normalizacion de Hamidieh.

## 7. Auditoria de la validacion temporal del EDA (`notebooks/EDA.ipynb`, `model_summary.json`)

- Reproducido EXACTAMENTE: N = 15816 materiales (reportado 15816); corte 1995: 8353/7463 (reportado 8353/7463); corte 2005: 12339/3477 (reportado 12339/3477); corte 2015: 15133/683 (reportado 15133/683).
- Diferencias con el corpus v3 que explican 15 816 vs 12543:
  1. El EDA 'limpia' la formula borrando sufijos `-Y`, `+z`, `-d`... (`O7-Y` -> `O7`): inventa la estequiometria de oxigeno que la auditoria H03 excluye; 1859 de sus materiales proceden de formulas alteradas.
  2. Las formulas que siguen sin parsear comparten UN material_id (md5('|')): 1270 filas colapsadas en 1 'material' con Tc mediana.
  3. Incluye Tc<=0 (69 materiales) y no aplica ninguna exclusion quimica (OY, sin coeficiente, >150 atomos).
  4. La identidad de material es `reduced_formula|chemsys` (pymatgen), no la composicion a 1e-3; el anio es el minimo (igual que aqui); train = anio <= corte y test = anio > corte (aqui < T / >= T).
  5. Descriptores propios (bag-of-elements + 8 estadisticos mendeleev), no los 132 Magpie; RF 500 arboles / XGB 600 sin los hiperparametros del articulo.
  Materiales del EDA con Tc>0, parseables y sin formula alterada: 13891 (orden de magnitud del corpus v3).
- Sus cifras (RF: R2 0.71 / 0.56 / 0.56 en 1995 / 2005 / 2015) son por tanto de un corpus distinto, sin intervalos ni estratos, y no se reutilizan.
