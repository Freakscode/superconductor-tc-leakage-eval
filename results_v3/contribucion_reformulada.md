# Contribución reformulada (H02, H09) — insumos para el manuscrito v3

Procedencia: `results_v3/antecedentes.md`, `results_v3/antecedentes_verificacion.json` (citas literales verificadas), `results_v3/h09_ybco_family_check.json` (medición), `results_v3/refs_crossref_v3.json` (bibliografía desde CrossRef). Nada de lo que sigue afirma un resultado experimental nuevo: los números citados son los del manuscrito v5 o mediciones de este paso; donde un experimento 1–7 aún no se ha ejecutado se indica como *pendiente*.

## (a) Párrafo de contribuciones nuevo (≤ 4 viñetas; no reclama el agrupamiento)

> Grouping superconductors by their constituent-element set for train/test separation is established practice: Sommer *et al.* use it as the default protocol of the 3DSC benchmark (with the same Ba–Cu–O–Y example), and PhaseSelect takes the element set itself as the unit of analysis. What has not been measured is how much of the headline accuracy that practice removes, for which population, and which retrospective protocol best anticipates performance on chemistries not yet in the corpus. Our contributions are:
>
> 1. **A controlled measurement of the random-versus-grouped gap** on identical rows, descriptors, models and test budgets, replicated across two independently curated SuperCon-derived datasets and two featurizations, with family-paired uncertainty rather than row-resampled intervals.
> 2. **A ladder of separation levels and of nulls.** Separation by normalised composition, by element system and by host/cluster (LOCO) is compared at matched test-set size; family-mean and element-presence nulls bound how much of the conventional $R^2$ is available from a family label alone, and a presence → fractions → properties ablation isolates what each representation adds.
> 3. **A characterisation of the corpus as clustered data** — family concentration and intraclass correlation reported as descriptive quantities, with an explicitly defined estimator for any effective-sample-size statement — together with an audit of the chemical unit of evaluation (formula parsing, duplication, cross-dataset overlap).
> 4. **Row-level, verifiable partitions and a per-claim verifier**, so that the corrected protocol and every number in the paper can be regenerated with one command.

(Si el experimento 7 se completa, la viñeta 2 debe añadir “and by date”; si no, no prometerlo.)

## (b) Pregunta científica y sub-preguntas mapeadas a los experimentos 1–7 del informe

**Pregunta:** ¿qué evaluación retrospectiva predice mejor el rendimiento en superconductores de sistemas o fechas aún no observados, y qué información predictiva sobrevive cuando se controlan duplicación, dependencia familiar y selección del corpus?

| # | Sub-pregunta | Experimento del informe | Antecedente más cercano y qué deja abierto |
|---|---|---|---|
| Q1 | ¿Es fiable la unidad química sobre la que se evalúa (fórmula → sistema elemental)? ¿Cambian las métricas al corregir fórmulas inválidas (“…OY”) y duplicados? | 1. Auditoría semántica y normalización, antes/después | 3DSC p. 3 detecta las fórmulas “OY” y las excluye (128 entradas) pero no mide su efecto sobre métricas; Hamidieh sólo elimina filas repetidas exactas. |
| Q2 | ¿Persiste la brecha aleatorio→agrupado cuando la incertidumbre respeta la dependencia familiar, y para qué población (micro vs macro por familia)? | 2. Inferencia pareada por familias y métricas macro | Ninguno de los antecedentes reporta la brecha; 3DSC sólo reporta el protocolo agrupado. |
| Q3 | ¿Cuánto de la brecha procede de duplicados, de vecinos estequiométricos y de cambio de dominio químico? | 3. Niveles de separación con presupuestos comparables (aleatorio, composición, sistema, anfitrión/LOCO) | Meredig 2018 (LOCO por clusters), Roberts 2017 (bloquear puede sobreestimar el error interpolativo), Stanev 2018 Fig. 4b–e (macro-familias). Nadie compara los niveles a presupuesto igual en superconductores. |
| Q4 | ¿Qué información añade cada representación: presencia de elementos → fracciones → propiedades elementales, frente a vecino más próximo? | 4. Ablación presencia → fracciones → propiedades; vecino próximo y modelo composicional fuerte | PhaseSelect opera sólo con el conjunto de elementos (descarta fracciones); Meredig introduce el benchmark de vecino más próximo. |
| Q5 | ¿La concordancia entre Datasets A y B trasciende sus registros compartidos? | 5. Solapamiento A/B y evaluación en materiales/sistemas no compartidos | 3DSC y Stanev parten del mismo SuperCon; ningún trabajo cuantifica la dependencia entre “réplicas”. |
| Q6 | ¿Cuánto rinde un presupuesto de búsqueda definido cuando hay negativos medidos, empates resueltos y diversidad de familias? | 6. Selección con negativos, empates y diversidad | Stanev 2018 y PhaseSelect clasifican Tc>10 K pero evalúan con particiones aleatorias / por phase field, sin presupuesto K ni empates explícitos. |
| Q7 | ¿Qué protocolo retrospectivo (aleatorio, composición, sistema, LOCO) anticipa mejor el error y el ranking bajo un corte temporal verificable? | 7. Validación temporal con procedencia y modelo congelado (si viable) | Sheridan 2013 responde esto para QSAR (time-split ≈ prospectivo; aleatorio optimista; leave-class-out pesimista). Sin equivalente en superconductores. |

## (c) Reemplazo de “uno-a-uno con serie de dopaje” → “agrupación operativa por sistema elemental” (H09)

**Texto v5 (Introducción, contribución 1):** “A domain-appropriate grouping variable: the constituent-element set, which corresponds one-to-one with the doping series that generate the redundancy, and which can be verified by inspection rather than assumed from an unsupervised clustering.”

**Texto v5 (§2.2):** “This is the grouping that corresponds to the doping series responsible for the redundancy, and unlike an unsupervised clustering it is inspectable: a reader can verify family membership from the formula.”

**Propuesta v3 (§2.2, sustituye ambas frases):**

> We group compounds by their *element system* — the set of constituent elements irrespective of stoichiometry — following the protocol of the 3DSC benchmark~\cite{sommer2023}. This is an operational grouping, not a one-to-one map onto doping series. It is coarser than a series in one direction: YBa$_2$Cu$_3$O$_{7-\delta}$, Y$_2$Ba$_4$Cu$_7$O$_{15-\delta}$ and YBa$_2$Cu$_4$O$_8$ share the system \{Y, Ba, Cu, O\} yet are distinct phases with distinct structures~\cite{gupta1993}; in Dataset~A the single group \{Ba, Cu, O, Y\} holds 720 entries and 184 distinct formulas, with Cu/Y ratios of 3 (596 rows), 3.5 (38) and 4 (51) and $\Tc$ from 4.3 to 130~K. It is finer than a series in the other direction: introducing any new dopant (Ca in Y$_{1-x}$Ca$_x$Ba$_2$Cu$_3$O$_7$) creates a new group even though the host lattice is unchanged. What the grouping does guarantee is that no two compounds differing only in stoichiometry fall on opposite sides of a split, and that membership can be verified from the formula. Section~\ref{sec:results-robust} compares it with a finer level (normalised composition) and a coarser one (host/cluster), because each corresponds to a different use case rather than to a single “correct” criterion~\cite{roberts2017}.

Cifras de la propuesta: medidas en `h09_ybco_family_check.json` (Dataset B: 99 filas, Cu/Y = 3: 55, 3,5: 20, 4: 9; Tc 4,3–100,1 K). La existencia de las fases 123/247/124 como estructuras distintas es conocimiento general; Gupta & Gupta 1993 sólo aporta la comparación explícita 123 vs 124 (título verificado en CrossRef, texto cerrado).

## (d) Borrador de §4.2 “Relation to prior work” (inglés, formato del .tex actual)

```latex
\subsection{Relation to prior work}

That extrapolative validation is harder than interpolative validation is established across
fields. Meredig \etal~\cite{meredig2018} introduced leave-one-cluster-out cross-validation for
materials and showed, on a superconductor benchmark, that model performance depends on the
sampling of the training data and on the degree of extrapolation; Xiong \etal~\cite{xiong2020}
reached the same conclusion with forward cross-validation along the property axis, again including
$\Tc$. Outside materials, Roberts \etal~\cite{roberts2017} showed that block cross-validation is
almost always preferable to random cross-validation when the goal is prediction to new data, while
warning that blocking can itself induce extrapolation and overestimate interpolation error, and
Sheridan~\cite{sheridan2013} showed for QSAR that time-split validation tracks prospective
performance better than either random selection (too optimistic) or leave-class-out selection (too
pessimistic). Kapoor and Narayanan~\cite{kapoor2023} place these observations within a broader
pattern of leakage-driven irreproducibility, and the Matbench efforts~\cite{dunn2020,riebesell2024}
separate regression accuracy from discovery utility.

For superconductors specifically, grouping by constituent-element set is already in use. Sommer
\etal~\cite{sommer2023} evaluate every model on the 3DSC dataset with train/test splits grouped by
``chemical system'', defined as the set of all elements including dopants (their example is
Ba--Cu--O--Y for YBa$_2$Cu$_3$O$_7$), and use five-fold cross-validation grouped by chemical system
within superconductor classes; they also flag the SuperCon formulas whose trailing ``Y'' denotes an
unknown oxygen content rather than yttrium, an issue we audit in section~\ref{sec:families}.
PhaseSelect~\cite{vasylenko2023} goes further and takes the element set (``phase field'') as the
unit of learning, collapsing each field to its maximum reported $\Tc$. Stanev
\etal~\cite{stanev2018}, whose cleaned SuperCon extract is our Dataset~B, showed that models
trained on one macro-family (low-$\Tc$, cuprate, iron-based) have no predictive power on the others,
but evaluated their headline models with random 85/15 splits, as did Hamidieh~\cite{hamidieh2018}
with repeated random 2/3--1/3 splits on the data that form our Dataset~A.

None of these works measures the quantity this paper is about. Sommer \etal\ use the grouped
protocol without comparing it to a random one; Hamidieh and Stanev \etal\ report only the random
one; PhaseSelect discards within-system stoichiometry by construction; Meredig \etal\ and Xiong
\etal\ use clusters or property ranks that cannot be verified from a formula and do not control the
test budget across protocols. Our contribution is therefore not the grouping but the measurement:
the size of the random-to-grouped gap on identical rows, models and test budgets, replicated across
two datasets and two featurizations, with family-paired uncertainty; the comparison of separation
levels at matched budgets; the nulls that bound how much of the conventional $\Rsq$ a family label
alone supplies; and the audit of the chemical unit of evaluation. The grouping itself is
operational rather than physical (section~\ref{sec:families}); which level of separation is
``right'' depends on the intended use, which is why we report several rather than one.
```

Notas para el editor del .tex:
- `\cite{sommer2023}` debe entrar también en la Introducción (contribución 1) y en §2.2; `\cite{gupta1993}` y `\cite{roberts2017}` en §2.2 (texto (c)).
- Si el experimento 3 (niveles) o el 7 (temporal) no se ejecutan, eliminar la frase correspondiente del último párrafo: no prometer lo no medido.
- La afirmación sobre Meredig (“clusters … cannot be verified from a formula”) se apoya en el resumen (LOCO CV) y en fuentes secundarias que describen clusters k-means; el texto completo no fue accesible desde este entorno. Confirmar con el PDF antes de enviar.

### \bibitem nuevas (generadas desde CrossRef por `code/v3/fetch_refs_crossref.py`; listas para pegar en `thebibliography`)

Las claves `hamidieh2018`, `stanev2018`, `meredig2018` ya existen en el .tex y coinciden campo a campo con CrossRef (comprobado en `bibitems_v3.tex`); no duplicar.

```latex
\bibitem{sommer2023}
Sommer T, Willa R, Schmalian J and Friederich P 2023 3DSC - a dataset of superconductors including crystal structures
\textit{Sci. Data} \textbf{10} 816
\url{https://doi.org/10.1038/s41597-023-02721-y}

\bibitem{vasylenko2023}
Vasylenko A, Antypov D, Gusev V V, Gaultois M W, Dyer M S and Rosseinsky M J 2023 Element selection for functional materials discovery by integrated machine learning of elemental contributions to properties
\textit{npj Comput. Mater.} \textbf{9} 164
\url{https://doi.org/10.1038/s41524-023-01072-x}

\bibitem{xiong2020}
Xiong Z, Cui Y, Liu Z, Zhao Y, Hu M and Hu J 2020 Evaluating explorative prediction power of machine learning algorithms for materials discovery using $k$-fold forward cross-validation
\textit{Comput. Mater. Sci.} \textbf{171} 109203
\url{https://doi.org/10.1016/j.commatsci.2019.109203}

\bibitem{sheridan2013}
Sheridan R P 2013 Time-split cross-validation as a method for estimating the goodness of prospective prediction
\textit{J. Chem. Inf. Model.} \textbf{53} 783--90
\url{https://doi.org/10.1021/ci400084k}

\bibitem{roberts2017}
Roberts D R, Bahn V, Ciuti S, Boyce M S, Elith J, Guillera-Arroita G, Hauenstein S, Lahoz-Monfort J J, Schröder B, Thuiller W, Warton D I, Wintle B A, Hartig F and Dormann C F 2017 Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure
\textit{Ecography} \textbf{40} 913--29
\url{https://doi.org/10.1111/ecog.02881}

\bibitem{gupta1993}
Gupta R P and Gupta M 1993 A comparative study of the pressure-induced charge redistribution in YBa$_{2}$Cu$_{3}$O$_{7}$ and YBa$_{2}$Cu$_{4}$O$_{8}$
\textit{Physica C} \textbf{204} 349--58
\url{https://doi.org/10.1016/0921-4534(93)91019-r}
```
