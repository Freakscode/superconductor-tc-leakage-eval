"""topk_report.py -- assemble <RESULTS_V3>/topk_resumen_H05_H06.md from the topk_*.csv/json outputs
(numbers are read from the files, never typed), including the replacement sentences for the manuscript.
Usage: python code/v3/topk_report.py   (after topk_prevalence_ties.py, topk_metrics.py, topk_negatives.py)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import topk_common as tc  # noqa: E402

R = tc.RES_V3
prev = tc.read_csv(R / "topk_prevalence.csv"); ties = tc.read_csv(R / "topk_ties.csv")
met = tc.read_csv(R / "topk_metrics.csv"); dif = tc.read_csv(R / "topk_metrics_diffs.csv")
neg = tc.read_csv(R / "topk_with_negatives.csv"); nsum = json.load(open(R / "topk_negatives_summary.json"))
agg = met[met.fold == "mean_5folds"]; nagg = neg[neg.fold == "mean_5folds"]


def m(ds, model, proto, K=100):
    return agg[(agg.dataset == ds) & (agg.model == model) & (agg.protocol == proto) & (agg.K == K)].iloc[0]


def d(ds, comp, K=100):
    return dif[(dif.dataset == ds) & (dif.comparison == comp) & (dif.K == K)].iloc[0]


def n(model, proto, K=100):
    return nagg[(nagg.model == model) & (nagg.protocol == proto) & (nagg.K == K)].iloc[0]


def prev_range(ds, proto):
    p = prev[(prev.dataset == ds) & (prev.protocol == proto)].prevalence_test_ge_pct
    return p.min(), p.max(), p.mean()


def tie_row(ds, model, proto):
    t = ties[(ties.dataset == ds) & (ties.model == model) & (ties.protocol == proto)]
    return t


L = []
L.append("# H05/H06 — Selección top-K corregida: prevalencia real, empates, métricas de selección y negativos medidos\n")
L.append("Todos los números proceden de results_v3/topk_prevalence.csv, topk_ties.csv, topk_metrics.csv, topk_metrics_diffs.csv, "
         "topk_with_negatives.csv y topk_negatives_summary.json (scripts code/v3/topk_*.py). Operador del umbral: positivo := y >= p90(y del fold "
         "de entrenamiento). IC 95 % = bootstrap de familias (2 000 remuestreos, semilla 0), pareado entre condiciones, condicional a las "
         "predicciones OOF ya ajustadas (semilla 42, una partición por protocolo).\n")

L.append("## 1. Prevalencia real de la clase alto-Tc por fold (esperanza del selector aleatorio)\n")
L.append("| Dataset | Protocolo | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | media | mín–máx |"); L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for ds in ("A", "Bv3", "Bv2"):
    for proto in ("grouped", "random"):
        p = prev[(prev.dataset == ds) & (prev.protocol == proto)].sort_values("fold").prevalence_test_ge_pct.values
        L.append(f"| {ds} | {proto} | " + " | ".join(f"{v:.2f}" for v in p) + f" | {p.mean():.2f} | {p.min():.2f}–{p.max():.2f} |")
L.append("\nEl informe adversarial midió A agrupado 13,80/3,62/18,22/7,48/8,51 % y B v2 agrupado 9,24/7,32/17,08/7,07/10,29 %: reproducidos exactamente. "
         "B v3 (checkpoint auditado, otra partición de familias) da " + "/".join(f"{v:.2f}" for v in prev[(prev.dataset == 'Bv3') & (prev.protocol == 'grouped')].sort_values('fold').prevalence_test_ge_pct.values) + " %. "
         "La fracción del fold de ENTRENAMIENTO con y >= p90 tampoco es exactamente 10 % (empates en Tc): "
         f"{prev.train_frac_ge_pct.min():.2f}–{prev.train_frac_ge_pct.max():.2f} % en el conjunto de folds; hasta {int(prev.n_test_rows_equal_thr.max())} filas de test coinciden exactamente con el umbral en un fold, por lo que '>' y '>=' difieren.\n")

L.append("## 2. Empates (K = 100)\n")
for ds, lab in (("A", "A"), ("Bv3", "B v3"), ("Bv2", "B v2")):
    t = tie_row(ds, "null", "grouped")
    L.append(f"- **Nulo media-de-familia, agrupado, {lab}**: predicción constante por fold (n_distinct_pred = {int(t.n_distinct_pred.max())}), todo el pool empatado "
             f"(grupo de empate = {int(t.tie_group_size_at_cut.min())}–{int(t.tie_group_size_at_cut.max())} filas). Selección publicada np.argsort(pred)[-100:] = últimas 100 filas del fold: "
             f"media {t.precision100_legacy_argsort_pct.mean():.1f} % (por fold {'/'.join(f'{v:.0f}' for v in t.precision100_legacy_argsort_pct)}). "
             f"Permutando sólo el orden de las filas (200 veces): la media de folds recorre {t.fold_mean_rowperm_min_pct.iloc[0]:.1f}–{t.fold_mean_rowperm_max_pct.iloc[0]:.1f} %. "
             f"Precisión esperada fraccional = prevalencia del fold: media {t.precision100_fractional_expected_pct.mean():.2f} % (200 desempates aleatorios: {t.precision100_random_tiebreak_mean_pct.mean():.2f} %).")
for ds, lab in (("A", "A"), ("Bv3", "B v3")):
    t = tie_row(ds, "presence", "grouped")
    L.append(f"- **Modelo de presencia (R0), agrupado, {lab}**: empata dentro de familia; grupo de empate en el corte de {int(t.tie_group_size_at_cut.min())}–{int(t.tie_group_size_at_cut.max())} filas; "
             f"por orden de filas la precisión de un fold varía entre {t.precision100_rowperm_min_pct.min():.0f} y {t.precision100_rowperm_max_pct.max():.0f} %; "
             f"media con la selección argsort del código {t.precision100_legacy_argsort_pct.mean():.1f} % → fraccional {t.precision100_fractional_expected_pct.mean():.2f} %.")
for ds, lab in (("A", "A"), ("Bv3", "B v3")):
    for model in ("XGBoost", "RandomForest"):
        t = tie_row(ds, model, "grouped")
        L.append(f"- **{model}, agrupado, {lab}**: grupo de empate máximo en el corte {int(t.tie_group_size_at_cut.max())}; argsort {t.precision100_legacy_argsort_pct.mean():.1f} % → fraccional {t.precision100_fractional_expected_pct.mean():.2f} %.")

L.append("\n## 3. Métricas de selección, K = 100, media de 5 folds [IC fam 95 %]\n")
L.append("| Dataset | Ranker | Split | Prevalencia % | Precisión@100 % | IC filas | Recuperación@100 % | Enriquecimiento | Familias distintas | Máx. cuota familia % | Precisión con <=3/familia % |")
L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for ds in ("A", "Bv3"):
    for model in ("null", "presence", "XGBoost", "RandomForest"):
        for proto in ("random", "grouped"):
            r = m(ds, model, proto)
            L.append(f"| {ds} | {model} | {proto} | {r.prevalence_pct:.2f} | {r.precision_pct:.1f} [{r.precision_ci_fam_lo_pct:.1f}, {r.precision_ci_fam_hi_pct:.1f}] | [{r.precision_ci_row_lo_pct:.1f}, {r.precision_ci_row_hi_pct:.1f}] "
                     f"| {r.recall_pct:.1f} [{r.recall_ci_fam_lo_pct:.1f}, {r.recall_ci_fam_hi_pct:.1f}] | {r.enrichment:.2f} [{r.enrichment_ci_fam_lo:.2f}, {r.enrichment_ci_fam_hi:.2f}] | {r.n_families_topK:.1f} | {r.max_family_share_pct:.1f} | {r.budget_precision_pct:.1f} |")
L.append("\nDiferencias pareadas (puntos porcentuales de precisión@100, bootstrap de familias pareado):\n")
L.append("| Dataset | Comparación | Δ pp | IC 95 % | fracción bootstrap > 0 |"); L.append("|---|---|---:|---:|---:|")
for ds in ("A", "Bv3"):
    for comp in dif.comparison.unique():
        r = d(ds, comp)
        L.append(f"| {ds} | {comp} | {r.d_precision_pp:+.1f} | [{r.d_precision_ci_lo_pp:+.1f}, {r.d_precision_ci_hi_pp:+.1f}] | {r.frac_boot_d_precision_gt0:.3f} |")
pf = met[(met.fold != "mean_5folds") & (met.K == 100) & (met.protocol == "grouped") & (met.model == "XGBoost")]
L.append("\nConcentración por familia (XGBoost agrupado, K = 100, por fold): familias distintas "
         + "; ".join(f"{ds}: {'/'.join(f'{v:.0f}' for v in pf[pf.dataset == ds].n_families_topK)}" for ds in ("A", "Bv3"))
         + "; cuota máxima de una familia " + "; ".join(f"{ds}: {'/'.join(f'{v:.0f}' for v in pf[pf.dataset == ds].max_family_share_pct)} %" for ds in ("A", "Bv3")) + ".\n")
L.append("Tabla completa por K en topk_metrics.md.\n")

L.append("## 4. Negativos medidos (H05) — Dataset B v3 ∪ entradas Tc <= 0\n")
ex = nsum["exclusiones_por_categoria"]
L.append(f"- Origen: {nsum['n_negativos_origen']} filas con Tc <= 0 en supercon_stanev.csv; todas Tc == 0 exactamente ({nsum['n_Tc_negativo']} negativas, {nsum['n_Tc_nan']} NaN). "
         f"Reglas de exclusión de audit_B.py aplicadas: {sum(ex.values())} filas caen ({', '.join(f'{k} {v}' for k, v in ex.items())}); "
         f"{nsum['n_nan_magpie_excluidos']} con NaN Magpie; {nsum['n_negativos_featurizados']} negativos featurizados.")
L.append(f"- {nsum['n_familia_existente_en_Bv3']} negativos pertenecen a familias (conjunto de elementos) presentes en B v3 y {nsum['n_familia_nueva']} a {nsum['n_familias_nuevas']} familias nuevas; "
         f"{nsum['n_conflicto_composicion_con_positivo']} tienen la misma composición normalizada que un superconductor de B v3 (etiquetas contradictorias del corpus); "
         f"{nsum['n_duplicados_entre_negativos']} son duplicados entre negativos. Elementos sin columna de presencia en B v3: {', '.join(nsum['elementos_de_negativos_ausentes_en_Bv3'])}.")
L.append("- Interpretación de la etiqueta: " + nsum["interpretacion_de_las_etiquetas"])
L.append("\n| Modelo (entrenado sólo con positivos) | Split | Negativos en el pool % | Precisión@100 sólo positivos % | Precisión@100 con negativos % [IC fam] | Negativos esperados en top-100 [IC fam] | Intermedios (0 < Tc < umbral) en top-100 | Mediana Tc predicha negativos / positivos (K) | Negativos con predicción >= umbral % |")
L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for proto in ("random", "grouped"):
    for model in ("XGBoost", "RandomForest", "presence"):
        r = n(model, proto)
        L.append(f"| {model} | {proto} | {r.frac_neg_pool_pct:.1f} | {r.precision_pos_only_pct:.1f} | {r.precision_with_neg_pct:.1f} [{r.precision_with_neg_ci_fam_lo_pct:.1f}, {r.precision_with_neg_ci_fam_hi_pct:.1f}] "
                 f"| {r.expected_neg_in_topK:.1f} [{r.expected_neg_in_topK_ci_fam_lo:.1f}, {r.expected_neg_in_topK_ci_fam_hi:.1f}] | {r.expected_intermediate_in_topK:.1f} | {r.neg_pred_median:.1f} / {r.pos_pred_median:.1f} | {r.neg_pred_ge_thr_pct:.2f} |")
L.append("\nPor K (XGBoost agrupado): " + "; ".join(f"K={K}: precisión {n('XGBoost','grouped',K).precision_pos_only_pct:.1f} → {n('XGBoost','grouped',K).precision_with_neg_pct:.1f} %, negativos {n('XGBoost','grouped',K).expected_neg_in_topK:.1f}" for K in tc.KS) + ".")
L.append("\nEjemplos de negativos (Tc = 0 registrada) que entran en el top-100 agrupado de XGBoost (desempate estable), ordenados por Tc predicha: "
         + "; ".join(f"{e['formula']} ({e['pred_Tc']:.0f} K{', familia nueva' if not e['family_in_Bv3'] else ''})" for e in nsum["negativos_en_top100_XGBoost_grouped_ejemplos"][:8]) + ". "
         "Varios son estequiometrías de cupratos conocidos (p. ej. Y1Ba2Cu3O6.975, Tm1Ba2Cu3O1) cuya Tc = 0 es más probablemente un error de captura o una fase no superconductora por O que un no superconductor de la familia; el fichero no permite decidirlo.\n")

L.append("## 5. Frases para el manuscrito v3 (sustituyen a 'more than four out of five attempts' y al 'base rate 10 % by construction')\n")
a = m("A", "XGBoost", "grouped"); b = m("Bv3", "XGBoost", "grouped"); an = m("A", "null", "grouped"); bn = m("Bv3", "null", "grouped")
ap = m("A", "presence", "grouped"); bp = m("Bv3", "presence", "grouped"); ar = m("A", "RandomForest", "grouped"); br = m("Bv3", "RandomForest", "grouped")
pa = prev_range("A", "grouped"); pb = prev_range("Bv3", "grouped")
nx = n("XGBoost", "grouped"); dxa = d("A", "XGBoost - presence (grouped)"); dxb = d("Bv3", "XGBoost - presence (grouped)")
dra = d("A", "XGBoost random - grouped"); drb = d("Bv3", "XGBoost random - grouped")
L.append("EN (sección 2.6, definición): \"We define the high-Tc class as test compounds whose measured Tc is at or above (>=) the 90th percentile of Tc in the "
         "training fold. Because families differ in their Tc distribution, the prevalence of that class in the held-out pool is not 10 %: it ranges from "
         f"{pa[0]:.1f} % to {pa[1]:.1f} % across the five family-held-out folds of Dataset A (mean {pa[2]:.1f} %) and from {pb[0]:.1f} % to {pb[1]:.1f} % on Dataset B "
         f"(mean {pb[2]:.1f} %). We therefore report precision@K together with the fold prevalence (the expected precision of a random selector) and their ratio, "
         "the enrichment. Ties in the predicted Tc are resolved by the expected precision under a uniformly random ordering of the tied compounds; the "
         "family-mean null under grouped validation predicts a constant per fold, so its precision equals the fold prevalence by construction "
         f"({an.precision_pct:.1f} % on A, {bn.precision_pct:.1f} % on B; the {11.6:.1f} % and 9.6 % of the previous version were artefacts of the row order).\"\n")
L.append("EN (sección 3.6, resultado): \"Under family-held-out validation the full model attains a retrospective precision@100 of "
         f"{a.precision_pct:.1f} % (95 % family-bootstrap CI {a.precision_ci_fam_lo_pct:.1f}–{a.precision_ci_fam_hi_pct:.1f}) on Dataset A and {b.precision_pct:.1f} % "
         f"({b.precision_ci_fam_lo_pct:.1f}–{b.precision_ci_fam_hi_pct:.1f}) on Dataset B, against fold prevalences of {pa[2]:.1f} % and {pb[2]:.1f} % (enrichment "
         f"{a.enrichment:.1f} and {b.enrichment:.1f}; recall@100 {a.recall_pct:.0f} % and {b.recall_pct:.0f} % of the high-Tc compounds in the pool). This is retrospective "
         "precision among corpus materials whose families were held out of training: it measures how well the ranker recovers already-reported high-Tc compounds "
         "of unseen element sets, not the probability that a synthesis succeeds. The 100 selections are not 100 independent discoveries either: they come from "
         f"{a.n_families_topK:.0f} (A) and {b.n_families_topK:.0f} (B) distinct families on average, a single family supplies up to {pf[pf.dataset=='A'].max_family_share_pct.max():.0f} % of a fold's top-100, "
         f"and limiting the selection to three compounds per family lowers precision@100 to {a.budget_precision_pct:.1f} % and {b.budget_precision_pct:.1f} %.\"\n")
L.append("EN (sección 3.6, negativos): \"Dataset B, like every SuperCon-derived regression set, contains only materials reported as superconducting. When the "
         f"{nsum['n_negativos_featurizados']} SuperCon entries recorded as Tc = 0 are returned to the held-out pools ({nx.frac_neg_pool_pct:.0f} % of the pool), the regressor trained on "
         f"positives only places {nx.expected_neg_in_topK:.1f} of them ({nx.expected_neg_in_topK_ci_fam_lo:.1f}–{nx.expected_neg_in_topK_ci_fam_hi:.1f}) in each top-100 and "
         f"precision@100 falls to {nx.precision_with_neg_pct:.1f} % ({nx.precision_with_neg_ci_fam_lo_pct:.1f}–{nx.precision_with_neg_ci_fam_hi_pct:.1f}); a further "
         f"{nx.expected_intermediate_in_topK:.0f} selections are superconductors below the high-Tc threshold. The source records a single value, Tc = 0, without the lowest "
         "temperature explored, so measured non-superconductivity, absence of a report and censoring cannot be separated; these entries are censored negatives "
         "of unknown floor, and a discovery-oriented evaluation would need a candidate universe with interpretable negatives.\"\n")
L.append("EN (contribución de los descriptores, sustituye '7.7 ± 1.0 points'): \"With the same family multiset resampled in both arms, the descriptors add "
         f"{dxa.d_precision_pp:+.1f} points ({dxa.d_precision_ci_lo_pp:+.1f} to {dxa.d_precision_ci_hi_pp:+.1f}) over element presence on Dataset A and {dxb.d_precision_pp:+.1f} points "
         f"({dxb.d_precision_ci_lo_pp:+.1f} to {dxb.d_precision_ci_hi_pp:+.1f}) on Dataset B at K = 100; on Dataset A the interval includes zero for this partition, so the "
         "descriptor contribution to screening is established on B and only suggestive on A.\"\n")
L.append("EN (random vs grouped): \"Random splitting overstates precision@100 by "
         f"{dra.d_precision_pp:+.1f} points ({dra.d_precision_ci_lo_pp:+.1f} to {dra.d_precision_ci_hi_pp:+.1f}) on A and {drb.d_precision_pp:+.1f} points ({drb.d_precision_ci_lo_pp:+.1f} to {drb.d_precision_ci_hi_pp:+.1f}) on B "
         "(paired family bootstrap).\"\n")
L.append("Tabla 5 (valores nuevos, %, media de folds, fraccional bajo empates; entre corchetes IC fam 95 %):\n")
L.append("| Predictor | Split | A | B (v3) |"); L.append("|---|---|---:|---:|")
for model, lab in (("null", "Family-mean null"), ("presence", "Element presence"), ("XGBoost", "Full model (XGBoost)"), ("RandomForest", "Full model (RandomForest)")):
    for proto, plab in (("random", "Random"), ("grouped", "Chemical-family")):
        ra, rb = m("A", model, proto), m("Bv3", model, proto)
        L.append(f"| {lab} | {plab} | {ra.precision_pct:.1f} [{ra.precision_ci_fam_lo_pct:.1f}, {ra.precision_ci_fam_hi_pct:.1f}] | {rb.precision_pct:.1f} [{rb.precision_ci_fam_lo_pct:.1f}, {rb.precision_ci_fam_hi_pct:.1f}] |")
L.append(f"| Prevalencia del pool (media; mín–máx) | Random | {prev_range('A','random')[2]:.1f} ({prev_range('A','random')[0]:.1f}–{prev_range('A','random')[1]:.1f}) | {prev_range('Bv3','random')[2]:.1f} ({prev_range('Bv3','random')[0]:.1f}–{prev_range('Bv3','random')[1]:.1f}) |")
L.append(f"| Prevalencia del pool (media; mín–máx) | Chemical-family | {pa[2]:.1f} ({pa[0]:.1f}–{pa[1]:.1f}) | {pb[2]:.1f} ({pb[0]:.1f}–{pb[1]:.1f}) |")
L.append("\n---\nprocedencia: " + met.procedencia.iloc[0] + " || " + neg.procedencia.iloc[0] + " || " + ties.procedencia.iloc[0])
(R / "topk_resumen_H05_H06.md").write_text("\n".join(L), encoding="utf-8")
print(f"wrote {R / 'topk_resumen_H05_H06.md'} ({len(L)} blocks)")
