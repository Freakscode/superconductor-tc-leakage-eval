# Auditoria semantica de Dataset B (H03) -- resumen

Procedencia: {"script": "code/v3/audit_B.py", "python": "3.13.15", "pandas": "2.3.2", "numpy": "2.3.3", "pymatgen": "2025.6.14", "matminer": "0.9.2", "input": "data/supercon_stanev.csv", "definicion_fila": "fila_origen = fila 0-based de supercon_stanev.csv; checkpoint_row = posicion entre filas parseables con Tc>0 (orden de origen)"}

- Filas del origen: 16414; parseables con Tc>0 (base v2): 12440; no parseables con Tc>0: 8.
- Filas de la base v2 con al menos una marca: 1252.
- Decision final por fila (base v2): corregir=3, excluir=104, mantener=1145; filas en v3 = 12336 (de ellas 3 corregidas).
- Duplicados por composicion normalizada: 760 filas en 363 grupos (1e-6); 1463 filas en 658 grupos (1e-3). Se mantienen.

| categoria | filas | mantener | corregir | excluir |
|---|---:|---:|---:|---:|
| OY_oxigeno_indeterminado | 75 | 0 | 0 | 75 |
| O_final_sin_coeficiente | 7 | 0 | 0 | 7 |
| coeficiente_alto_cluster_o_celda | 70 | 70 | 0 | 0 |
| coeficiente_alto_porcentaje_atomico | 257 | 257 | 0 | 0 |
| coeficiente_cero | 2 | 2 | 0 | 0 |
| coeficiente_extremo_error_tipografico | 5 | 0 | 3 | 2 |
| duplicado_composicion_normalizada | 760 | 760 | 0 | 0 |
| elemento_raro_Am_Cm_Po | 2 | 2 | 0 | 0 |
| elemento_repetido | 1 | 0 | 1 | 0 |
| elemento_repetido_artefacto | 1 | 0 | 0 | 1 |
| elemento_repetido_notacion_por_sitio | 111 | 111 | 0 | 0 |
| elemento_sin_coeficiente_monoelemental | 2 | 2 | 0 | 0 |
| elemento_sin_coeficiente_multielemental | 19 | 0 | 0 | 19 |
| isotopo_D_T | 25 | 25 | 0 | 0 |
| no_parseada_pymatgen | 8 | 0 | 0 | 8 |

## Fundamento de la lectura 'OY'

3DSC, Sommer et al. 2023, Sci. Data 10:816, doi:10.1038/s41597-023-02721-y, p. 3: 'in the formula Bi4.4Sr3.6Ca2Cu4OY the Y was considered to represent yttrium, but it actually represented an unknown quantity of oxygen (the SuperCon database strictly follows a nomenclature, where each element has a count, even if the count is 1)'

## Correcciones aplicadas (familia inalterada en las tres)

- `Pr185Ce0.15Cu1O4` -> `Pr1.85Ce0.15Cu1O4`: Pr185 es Pr1.85 con el punto perdido: Pr(2-x)Ce(x)CuO4 con x=0.15 (1.85+0.15=2, sitio de tierra rara completo); el propio origen contiene Pr1.85Ce0.15Cu1O4.065/4.161/4.175 con Tc 16-19 K, coherentes con Tc=20 K. Fuente: Takagi, Uchida & Tokura, Phys. Rev. Lett. 62, 1197 (1989): Pr2-xCexCuO4, x~0.15, Tc~20 K.
- `La1.85Sr0.15Cu98Ni0.02O4` -> `La1.85Sr0.15Cu0.98Ni0.02O4`: Cu98 es Cu0.98 con el punto perdido: La1.85Sr0.15Cu(1-x)Ni(x)O4 con x=0.02 (0.98+0.02=1, sitio de Cu completo); el origen contiene la serie x=0.01,0.03,0.04,0.05 con Tc 30.7/14.0/3.2/2.1 K, y Tc=21.96 K cae entre x=0.01 y x=0.03. Fuente: Xiao et al., Phys. Rev. B 42, 8752 (1990): Ni substitution in La1.85Sr0.15CuO4.
- `Rb3C60C60` -> `Rb3C60`: token C60 duplicado (parseado como Rb3C120, fraccion molar distorsionada); Rb3C60 es el fullerido canonico y su Tc documentada (29-29.5 K) coincide con la Tc de la fila (29.0 K). Fuente: Rosseinsky et al., Phys. Rev. Lett. 66, 2830 (1991): Rb3C60, Tc = 28-29.5 K.

## Filas no parseables (excluidas ya en v1/v2)

- fila 2095 `Eu1.45Pr0.05Ce0.5Sr2Cu2Nb1O10=z` (Tc=21.4): pymatgen: =z is an invalid formula!. no recuperable sin inventar una cantidad (oxigeno indeterminado '=z'/'+2', o coeficiente negativo)
- fila 4554 `Sm1Ba-1Cu3O6.94` (Tc=86.3): pymatgen: Amounts in Composition cannot be negative!. no recuperable sin inventar una cantidad (oxigeno indeterminado '=z'/'+2', o coeficiente negativo)
- fila 9806 `Y2C2Br0.5!1.5` (Tc=11.6): pymatgen: !1.5 is an invalid formula!. recuperable en principio: '!' es con alta probabilidad 'I' (Y2C2Br0.5I1.5, carburo-haluro Y2C2(Br,I)2, Tc 10-12 K; Simon et al. 1991); NO se recupera en v3 para no anadir filas
- fila 9848 `Hg0.3Pb0.7Sr1.75La0.25Cu1O4+2` (Tc=30.0): pymatgen: +2 is an invalid formula!. no recuperable sin inventar una cantidad (oxigeno indeterminado '=z'/'+2', o coeficiente negativo)
- fila 13547 `Hg1Sr2Ho0.333Ce0.667Cu2O6=z` (Tc=72.0): pymatgen: =z is an invalid formula!. no recuperable sin inventar una cantidad (oxigeno indeterminado '=z'/'+2', o coeficiente negativo)
- fila 14018 `B1Sr2Ca3Cu4O2N+3` (Tc=110.0): pymatgen: +3 is an invalid formula!. recuperable en principio: 'O2N+3' = O(2n+3) con n = nº de Cu (fases BSr2Ca(n-1)CunO(2n+3), Ihara et al.); NO se recupera en v3
- fila 14504 `B1Sr2Ca4Cu5O2N+3` (Tc=85.0): pymatgen: +3 is an invalid formula!. idem O(2n+3), n=5; NO se recupera en v3
- fila 16346 `B1Sr2Ca2Cu3O2N+3` (Tc=75.0): pymatgen: +3 is an invalid formula!. idem O(2n+3), n=3; NO se recupera en v3
