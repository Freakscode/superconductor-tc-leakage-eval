"""Generate code/v3/manuscript_numbers.json for paper1_manuscript.tex v3 (read by code/v3/verify_manuscript.py).

Every entry maps ONE number of the manuscript to its location (table/row/col/part or a unique prose anchor)
and to its value (result file + key + field in results_v3/ or results_v2/, an expression over such values, a
hyperparameter read from the pipeline code, a regex on requirements.txt, or a declared constant with its reason).
Entries of the v5 manifest whose anchor and source are unchanged are re-used from code/v3/manuscript_numbers_v5.json.

    python code/v3/build_manifest_v3.py     # rewrites manuscript_numbers.json next to this file
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E: list[dict] = []


def add(id_, loc, src, fmt=None, **kw):
    e = {"id": id_, **loc, **src}
    if fmt: e["fmt"] = fmt
    e.update(kw); E.append(e)


def cell(table, row, col, part=0, block=None, block_row=False):
    d = {"table": table, "row": row, "col": col, "part": part}
    if block: d["block"] = block
    if block_row: d["block_row"] = True
    return d


def prose(anchor, occurrences=1):
    d = {"anchor": anchor}
    if occurrences != 1: d["occurrences"] = occurrences
    return d


def csv(source, key, field, agg=None, scale=None):
    d = {"source": source, "key": key, "field": field}
    if agg: d["agg"] = agg
    if scale is not None: d["scale"] = scale
    return d


def js(source, field, agg=None):
    d = {"source": source, "field": field}
    if agg: d["agg"] = agg
    return d


expr = lambda e_, **vars_: {"expr": e_, "vars": vars_}
const = lambda v, reason: {"const": v, "reason": reason}
P = lambda id_, anchor, src, fmt=None, occ=1, **kw: add(id_, prose(anchor, occ), src, fmt, **kw)
C = lambda id_, table, row, col, src, fmt=None, part=0, block=None, block_row=False, **kw: add(id_, cell(table, row, col, part, block, block_row), src, fmt, **kw)

R3, R2 = "results_v3/", "results_v2/"
PFB = lambda ds, model, stat, field="point": csv(R3 + "paired_family_bootstrap.csv", {"dataset": ds, "model": model, "statistic": stat, "scheme": "family_paired"}, field)
T2 = lambda ds, model, field: csv(R3 + "tabla2_shuffle_v3.csv", {"dataset": ds, "modelo": model}, field)
T3 = lambda ds, bloque, comp, field: csv(R3 + "tabla3_v3.csv", {"dataset": ds, "bloque": bloque, "comparacion": comp}, field)
GRID = lambda ds, depth, field, agg: csv(R3 + "hp_grid_ambos_v3.csv", {"dataset": ds, "es_publicada": "False", "max_depth": depth}, field, agg)
STRAT = lambda ds, splitter, agg: csv(R3 + "stratified_group_check_v3.csv", {"dataset": ds, "splitter": splitter}, "inflacion_pct", agg)
LEV = lambda ds, level, field: csv(R3 + "separation_levels.csv", {"dataset": ds, "level": level}, field)
DEFF = lambda ds, est, field, variant="todas las familias": csv(R3 + "design_effect_v3.csv", {"dataset": ds, "variant": variant, "estimador": est}, field)
EST_I, EST_II = "(i) media ponderada por filas (peso 1)", "(ii) media de medias de familia (peso 1/n_i)"
ABL = lambda ds, prot, method, field: csv(R3 + "ablation_representation.csv", {"dataset": ds, "protocol": prot, "method": method}, field)
ABLD = lambda ds, prot, a, b, field: csv(R3 + "ablation_representation_diffs.csv", {"dataset": ds, "protocol": prot, "ref_a": a, "method_b": b}, field)
REL = lambda ds, design, cond, field: csv(R3 + "remove_relatives_control_summary.csv", {"dataset": ds, "design": design, "condition": cond}, field)
TOPK = lambda ds, model, prot, field: csv(R3 + "topk_metrics.csv", {"dataset": ds, "model": "nan" if model is None else model, "protocol": prot, "K": 100, "fold": "mean_5folds"}, field)
TOPKD = lambda ds, comp, field: csv(R3 + "topk_metrics_diffs.csv", {"dataset": ds, "comparison": comp, "K": 100}, field)
NEG = lambda model, prot, field: csv(R3 + "topk_with_negatives.csv", {"model": model, "protocol": prot, "K": 100, "fold": "mean_5folds"}, field)
PREV = lambda ds, prot, agg: csv(R3 + "topk_prevalence.csv", {"dataset": ds, "protocol": prot, "fold": [0, 1, 2, 3, 4]}, "prevalence_test_ge_pct", agg)
OV = lambda crit, field: csv(R3 + "overlap_AB.csv", {"checkpoint_B": "v3", "criterio": crit}, field)
OC = lambda tr, ev, sub, field: csv(R3 + "overlap_complement_eval.csv", {"modelo_entrenado_en": f"{tr} (Magpie, todas las filas)", "evaluado_en": ev, "subconjunto": sub}, field)
ASENS = lambda model, field: csv(R3 + "tabla1_A_sin_O1_inflacion.csv", {"variant": "refold", "modelo": model}, field)
RD = lambda metrica, field="valor": csv(R3 + "replicate_dispersion.csv", {"metrica": metrica}, field)
FC = lambda ds, field: csv(R3 + "family_concentration_v3.csv", {"dataset": ds}, field)
NN = lambda ds, field: csv(R3 + "nn_distance_trainscaler.csv", {"dataset": ds}, field)
AUD = lambda key, agg="count": csv("data/audit/semantic_audit_B.csv", key, "fila_origen", agg)
REG = lambda field: js(R3 + "regeneracion_B_v3.json", field)
TV = lambda T, cond, model, est, met, field="valor", esc="principal": csv(R3 + "temporal_validation.csv", {"corte": T, "escenario": esc, "condicion": cond, "modelo": model, "estrato": est, "metrica": met}, field)
TC = lambda T, proto, model, met, field, est="todo", esc="principal": csv(R3 + "temporal_validation_comparacion.csv", {"corte": T, "escenario": esc, "protocolo": proto, "modelo": model, "estimacion": est, "metrica": met}, field)
TMETA = lambda field: js(R3 + "temporal_validation_meta.json", field)
TPROV = lambda field: js(R3 + "temporal_provenance.json", field)
PY = lambda module, expr_: {"python": {"module": module, "expr": expr_}}
XGB = lambda p: PY("code/datasetB_pipeline.py", f"make_models(42)['XGBoost'].get_params()['{p}']")
RFB = lambda p: PY("code/datasetB_pipeline.py", f"make_models(42)['RandomForest'].get_params()['{p}']")
AN = lambda ds, prot, level, T: {"A": "A", "B": "Bv3"}[ds]  # noqa (dataset naming of topk files)

# ================================================================ re-used v5 entries ==========
KEEP = ['lit_MAE_6K', 'data_dup_exact', 'data_dup_feat', 'data_dup_pct', 'data_nfeat_A_2', 'data_parse_fail', 'data_nfeat_B',
        'hp_A_xgb_nest', 'hp_A_xgb_depth', 'hp_A_xgb_lr', 'hp_A_xgb_sub', 'hp_A_xgb_col', 'hp_A_xgb_gamma', 'hp_A_xgb_alpha', 'hp_A_rf_nest',
        'hp_A_rf_depth', 'hp_B_xgb_nest', 'hp_B_xgb_depth', 'hp_B_xgb_lr', 'hp_B_xgb_sub', 'hp_B_rf_nest', 'hp_B_rf_depth', 'hp_rf_leaf',
        'eta2_formula_1', 'gap_A_grp', 'twins_nfeat', 'grid_d4', 'grid_d6', 'grid_d8', 'grid_d10', 'grid_lr_lo', 'grid_lr_hi', 'grid_mcw_lo',
        'grid_mcw_hi', 'grid_nest300', 'grid16_text2', 'bacuoy_77K', 'strat_A_lo', 'strat_A_hi', 'unstrat_A_lo', 'unstrat_A_hi', 'fig3_nA_rand',
        'fig3_nA_grp', 'lim_R2_0928', 'repro_Tc_le0', 'ver_python', 'ver_numpy', 'ver_sklearn', 'ver_xgboost', 'ver_matminer']
old = json.load(open(HERE / "manuscript_numbers_v5.json"))["entries"]
E += [e for e in old if e["id"] in KEEP]
assert len(E) == len(KEEP), set(KEEP) - {e["id"] for e in E}

# ================================================================ TABLES =====================
# --- tab:main -------------------------------------------------------------------------------
MODELS = [("Family-mean null", "FamilyMeanNull"), ("XGBoost", "XGBoost"), ("Random forest", "RandomForest")]
for ds, nk in (("A", (FC("A", "N"), FC("A", "k_familias"))), ("B", (REG("N_v3"), REG("familias_v3")))):
    C(f"tab_main_{ds}_N", "tab:main", f"Dataset {ds}", 1, nk[0], ".0f", part=0, block_row=True)
    C(f"tab_main_{ds}_k", "tab:main", f"Dataset {ds}", 1, nk[1], ".0f", part=1, block_row=True)
    for lab, mod in MODELS:
        for split, sfx in (("Random", "random"), ("Element-system", "grouped")):
            for col, met, fmt in ((3, "MAE", ".2f"), (4, "RMSE", ".2f"), (5, "R2", ".3f")):
                C(f"tab_main_{ds}_{mod}_{sfx}_{met}", "tab:main", [lab, split], col, PFB(ds, mod, f"{met}_{sfx}"), fmt, block=f"Dataset {ds}")
# --- tab:robust -----------------------------------------------------------------------------
for ds in "AB":
    for lab, mod in (("XGBoost", "XGBoost"), ("Random forest", "RandomForest")):
        for col, f, fmt in ((3, "MAE_random_mean", ".2f"), (4, "MAE_random_std", ".2f"), (5, "MAE_grouped_mean", ".2f"), (6, "MAE_grouped_std", ".2f"),
                            (7, "inflacion_pct_mean", ".1f"), (8, "inflacion_pct_std", ".1f"), (9, "dR2_mean", ".3f"), (10, "dR2_std", ".3f")):
            C(f"tab_robust_{ds}_{mod}_{f}", "tab:robust", [ds, lab], col, T2(ds, mod, f), fmt)
# --- tab:tuning -----------------------------------------------------------------------------
FIXED = [("Fixed, published", "a_fixed", "publicada: misma config en ambos brazos"),
         ("Fixed, modal nested", "a_fixed", "modal v2 (d10,lr0.05,mcw10): misma config en ambos brazos"),
         ("Nested in both arms", "b_nested", "anidado en ambos brazos")]
for lab, blq, comp in FIXED:
    for ds, off in (("A", 0), ("B", 3)):
        C(f"tab_tuning_{ds}_{blq}_{lab.replace(" ", "_").replace(",", "")}_r", "tab:tuning", lab, 2 + off, T3(ds, blq, comp, "MAE_random"), ".2f")
        C(f"tab_tuning_{ds}_{blq}_{lab.replace(" ", "_").replace(",", "")}_g", "tab:tuning", lab, 3 + off, T3(ds, blq, comp, "MAE_grouped"), ".2f")
        for part, f in enumerate(("inflacion_pct", "inflacion_lo", "inflacion_hi")):
            C(f"tab_tuning_{ds}_{blq}_{lab.replace(" ", "_").replace(",", "")}_{f}", "tab:tuning", lab, 4 + off, T3(ds, blq, comp, f), ".1f", part=part)
for word, d in (("four", 4), ("six", 6), ("eight", 8), ("ten", 10)):
    for ds, off in (("A", 0), ("B", 3)):
        for col, f, fmt in ((2, "MAE_random", ".2f"), (3, "MAE_grouped", ".2f"), (4, "inflacion_pct", ".1f")):
            for part, agg in enumerate(("min", "max")):
                C(f"tab_tuning_grid_{ds}_d{d}_{f}_{agg}", "tab:tuning", f"Grid, depth {word}", col + off, GRID(ds, d, f, agg), fmt, part=part)
# --- tab:levels -----------------------------------------------------------------------------
LEVELS = [("L0 rows", "L0_aleatorio_filas"), ("L1 composition", "L1_composicion_1e-3"), ("L2 element system", "L2_sistema_elemental"),
          ("L3 host system", "L3_anfitrion_0.05"), ("L4 clusters", "L4_LOCO_kmeans50")]
for ds in "AB":
    for lab, lv in LEVELS:
        blk = f"Dataset {ds}"
        C(f"tab_levels_{ds}_{lv}_groups", "tab:levels", lab, 2, LEV(ds, lv, "n_groups"), ".0f", block=blk)
        C(f"tab_levels_{ds}_{lv}_comp", "tab:levels", lab, 3, LEV(ds, lv, "exact_composition_1e-3"), ".1f", block=blk) if False else None
        add(f"tab_levels_{ds}_{lv}_comp", cell("tab:levels", lab, 3, 0, blk), csv(R3 + "separation_levels.csv", {"dataset": ds, "level": lv}, "exact_composition_1e-3", scale=100), ".1f")
        add(f"tab_levels_{ds}_{lv}_sys", cell("tab:levels", lab, 4, 0, blk), csv(R3 + "separation_levels.csv", {"dataset": ds, "level": lv}, "exact_element_system", scale=100), ".1f")
        add(f"tab_levels_{ds}_{lv}_host", cell("tab:levels", lab, 5, 0, blk), csv(R3 + "separation_levels.csv", {"dataset": ds, "level": lv}, "exact_host_0.05", scale=100), ".1f")
        C(f"tab_levels_{ds}_{lv}_d1", "tab:levels", lab, 6, LEV(ds, lv, "nn_median_trainscaler"), ".3f", block=blk)
        for part, f in enumerate(("MAE", "MAE_fam_lo", "MAE_fam_hi")):
            C(f"tab_levels_{ds}_{lv}_{f}", "tab:levels", lab, 7, LEV(ds, lv, f), ".2f", part=part, block=blk)
        C(f"tab_levels_{ds}_{lv}_R2", "tab:levels", lab, 8, LEV(ds, lv, "R2"), ".3f", block=blk)
        add(f"tab_levels_{ds}_{lv}_top", cell("tab:levels", lab, 9, 0, blk), csv(R3 + "separation_levels.csv", {"dataset": ds, "level": lv}, "top100_mean", scale=100), ".1f")
        if not lv.startswith("L0"):
            for part, f in enumerate(("dMAE_vs_L0", "dMAE_vs_L0_fam_lo", "dMAE_vs_L0_fam_hi")):
                C(f"tab_levels_{ds}_{lv}_{f}", "tab:levels", lab, 10, LEV(ds, lv, f), ".2f", part=part, block=blk)
# --- tab:deff -------------------------------------------------------------------------------
for ds, dsk, big in (("A", "A", "sin la familia mayor (n=720)"), ("B", "B_v3", "sin la familia mayor (n=131)")):
    for col, f, fmt in ((2, "N", ".0f"), (3, "k", ".0f"), (4, "ICC", ".3f"), (5, "eta2", ".3f"), (6, "m0", ".2f"), (7, "m_w", ".1f"), (8, "deff", ".1f"), (9, "N_eff", ".0f")):
        C(f"tab_deff_{ds}_{f}", "tab:deff", ds, col, DEFF(dsk, EST_I, f), fmt)
    C(f"tab_deff_{ds}_Neff_nolargest", "tab:deff", ds, 10, DEFF(dsk, EST_I, "N_eff", big), ".0f")
    C(f"tab_deff_{ds}_deff_sys", "tab:deff", ds, 11, DEFF(dsk, EST_II, "deff"), ".2f")
    C(f"tab_deff_{ds}_Neff_sys", "tab:deff", ds, 12, DEFF(dsk, EST_II, "N_eff"), ".0f")
# --- tab:ablation ---------------------------------------------------------------------------
METH = [("R0 element presence", "R0"), ("R1 presence", "R1"), ("R2 descriptors", "R2"), ("R3 descriptors", "R3"), ("Five-NN on R1", "kNN5_R1"), ("Extra trees on R2", "ET500_R2")]
for ds in "AB":
    for prot, pl in (("random", "random split"), ("grouped", "element-system split")):
        blk = f"Dataset {ds}, {pl}"
        for lab, m in METH:
            C(f"tab_abl_{ds}_{prot}_{m}_p", "tab:ablation", lab, 2, ABL(ds, prot, m, "n_features"), ".0f", block=blk)
            for part, f in enumerate(("MAE", "MAE_fam_lo", "MAE_fam_hi")):
                C(f"tab_abl_{ds}_{prot}_{m}_{f}", "tab:ablation", lab, 3, ABL(ds, prot, m, f), ".2f", part=part, block=blk)
            for part, f in enumerate(("R2", "R2_fam_lo", "R2_fam_hi")):
                C(f"tab_abl_{ds}_{prot}_{m}_{f}", "tab:ablation", lab, 4, ABL(ds, prot, m, f), ".3f", part=part, block=blk)
            if m != "R0":
                add(f"tab_abl_{ds}_{prot}_{m}_sse", cell("tab:ablation", lab, 5, 0, blk), csv(R3 + "ablation_representation.csv", {"dataset": ds, "protocol": prot, "method": m}, "SSE_red_vs_R0", scale=100), ".1f")
            add(f"tab_abl_{ds}_{prot}_{m}_top", cell("tab:ablation", lab, 6, 0, blk), csv(R3 + "ablation_representation.csv", {"dataset": ds, "protocol": prot, "method": m}, "top100_mean", scale=100), ".1f")
# --- tab:screen -----------------------------------------------------------------------------
RANK = [("Family-mean null", None), ("Element presence", "presence"), ("Random forest", "RandomForest"), ("XGBoost", "XGBoost")]
for ds, dsk in (("A", "A"), ("B", "Bv3")):
    for prot, pl in (("random", "random split"), ("grouped", "element-system split")):
        blk = f"Dataset {ds}, {pl}"
        C(f"tab_screen_{ds}_{prot}_prev", "tab:screen", f"Dataset {ds}, {pl}", 1, TOPK(dsk, None, prot, "prevalence_pct"), ".1f", block_row=True)
        for lab, mod in RANK:
            for part, f in enumerate(("precision_pct", "precision_ci_fam_lo_pct", "precision_ci_fam_hi_pct")):
                C(f"tab_screen_{ds}_{prot}_{mod}_{f}", "tab:screen", lab, 2, TOPK(dsk, mod, prot, f), ".1f", part=part, block=blk)
            C(f"tab_screen_{ds}_{prot}_{mod}_sd", "tab:screen", lab, 3, TOPK(dsk, mod, prot, "precision_sd_folds_pct"), ".1f", block=blk)
            C(f"tab_screen_{ds}_{prot}_{mod}_rec", "tab:screen", lab, 4, TOPK(dsk, mod, prot, "recall_pct"), ".1f", block=blk)
            C(f"tab_screen_{ds}_{prot}_{mod}_enr", "tab:screen", lab, 5, TOPK(dsk, mod, prot, "enrichment"), ".2f", block=blk)
            C(f"tab_screen_{ds}_{prot}_{mod}_fam", "tab:screen", lab, 6, TOPK(dsk, mod, prot, "n_families_topK"), ".0f", block=blk)
            C(f"tab_screen_{ds}_{prot}_{mod}_bud", "tab:screen", lab, 7, TOPK(dsk, mod, prot, "budget_precision_pct"), ".1f", block=blk)
            if ds == "B" and mod is not None:
                C(f"tab_screen_{ds}_{prot}_{mod}_neg", "tab:screen", lab, 8, NEG(mod, prot, "precision_with_neg_pct"), ".1f", block=blk)
# --- tab:temporal ---------------------------------------------------------------------------
PROTO = [("L0 random", "L0_aleatorio"), ("L1 by composition", "L1_composicion"), ("L2 by element system", "L2_sistema"), ("L3 by host system", "L3_anfitrion"), ("L4 leave-clusters-out", "L4_loco_cluster")]
for M, blk in (("XGBoost", "XGBoost"), ("RandomForest", "Random forest")):
    for col, met, fmt in ((2, "MAE", ".2f"), (4, "R2", ".3f"), (6, "top100", ".2f"), (8, "Spearman", ".2f")):
        C(f"tab_temporal_{M}_real_{met}", "tab:temporal", "Realised", col, TV(2010, "temporal_real", M, "todo", met), fmt, block=blk)
        if col < 8:
            C(f"tab_temporal_{M}_real_{met}_lo", "tab:temporal", "Realised", col + 1, TV(2010, "temporal_real", M, "todo", met, "ci_lo_familias"), fmt, part=0, block=blk)
            C(f"tab_temporal_{M}_real_{met}_hi", "tab:temporal", "Realised", col + 1, TV(2010, "temporal_real", M, "todo", met, "ci_hi_familias"), fmt, part=1, block=blk)
    for lab, p in PROTO + [("L0 re-weighted", "L0_aleatorio")]:
        est = "reponderado_mezcla_test" if lab.startswith("L0 re") else "todo"
        for col, met, fmt in ((2, "MAE", ".2f"), (4, "R2", ".3f"), (6, "top100", ".2f"), (8, "Spearman", ".2f")):
            C(f"tab_temporal_{M}_{lab.replace(" ", "_").replace(",", "")}_{est[:4]}_{met}", "tab:temporal", lab, col, TC(2010, p, M, met, "estimacion_retrospectiva", est), fmt, block=blk)
            if col < 8:
                C(f"tab_temporal_{M}_{lab.replace(" ", "_").replace(",", "")}_{est[:4]}_{met}_d", "tab:temporal", lab, col + 1, TC(2010, p, M, met, "error_abs", est), fmt, block=blk)

# ================================================================ PROSE ======================
# ---- abstract
P("abs_N_A", r"On two curations of SuperCon (\num{<<v>>} rows in \num{3365}", FC("A", "N"), ".0f")
P("abs_k_A", r"rows in \num{<<v>>} element systems; \num{12336}", FC("A", "k_familias"), ".0f")
P("abs_N_B", r"element systems; \num{<<v>>} audited rows", REG("N_v3"), ".0f")
P("abs_k_B", r"audited rows in \num{<<v>>} systems) holding", REG("familias_v3"), ".0f")
INFL = {k: PFB(*k, "inflacion_pct") for k in (("A", "XGBoost"), ("A", "RandomForest"), ("B", "XGBoost"), ("B", "RandomForest"))}
DR2 = {k: PFB(*k, "dR2") for k in INFL}
MACRO = {k: PFB(*k, "inflacion_macro_pct") for k in INFL}
mn = lambda d, e: expr(e, **{f"v{i}": v for i, v in enumerate(d.values())})
P("abs_infl_lo", r"random forests by \SIrange{<<v>>}{64}{\percent} and lowers", mn(INFL, "min(v0,v1,v2,v3)"), ".0f")
P("abs_infl_hi", r"random forests by \SIrange{53}{<<v>>}{\percent} and lowers", mn(INFL, "max(v0,v1,v2,v3)"), ".0f")
P("abs_dr2_lo", r"lowers $\Rsq$ by \numrange{<<v>>}{0.084}, with paired", mn(DR2, "min(v0,v1,v2,v3)"), ".3f")
P("abs_dr2_hi", r"lowers $\Rsq$ by \numrange{0.074}{<<v>>}, with paired", mn(DR2, "max(v0,v1,v2,v3)"), ".3f")
P("abs_macro_lo", r"per system the inflation is \SIrange{<<v>>}{29}{\percent}, so the gap", mn(MACRO, "min(v0,v1,v2,v3)"), ".0f")
P("abs_macro_hi", r"per system the inflation is \SIrange{22}{<<v>>}{\percent}, so the gap", mn(MACRO, "max(v0,v1,v2,v3)"), ".0f")
P("abs_overlap", r"The two curations share \SI{<<v>>}{\percent} of their element systems", csv(R3 + "overlap_AB.csv", {"checkpoint_B": "v3", "criterio": "(iv) sistema elemental (familia)"}, "frac_filas_B_en_A", scale=100), ".1f")
P("abs_R1R2_A", r"presence plus fractions (\SI{+<<v>>}{\kelvin} \ci{0.00}{0.98} and", ABLD("A", "grouped", "R1", "R2", "dMAE"), ".2f")
P("abs_R1R2_B", r"\ci{0.00}{0.98} and \SI{+<<v>>}{\kelvin} \ci{-0.18}{0.59} MAE)", ABLD("B", "grouped", "R1", "R2", "dMAE"), ".2f")
P("abs_R1R2_A_lo", r"presence plus fractions (\SI{+0.46}{\kelvin} \ci{<<v>>}{0.98}", ABLD("A", "grouped", "R1", "R2", "dMAE_fam_lo"), ".2f")
P("abs_R1R2_A_hi", r"presence plus fractions (\SI{+0.46}{\kelvin} \ci{0.00}{<<v>>}", ABLD("A", "grouped", "R1", "R2", "dMAE_fam_hi"), ".2f")
P("abs_R1R2_B_lo", r"\SI{+0.16}{\kelvin} \ci{<<v>>}{0.59} MAE)", ABLD("B", "grouped", "R1", "R2", "dMAE_fam_lo"), ".2f")
P("abs_R1R2_B_hi", r"\SI{+0.16}{\kelvin} \ci{-0.18}{<<v>>} MAE)", ABLD("B", "grouped", "R1", "R2", "dMAE_fam_hi"), ".2f")
P("abs_K100", r"Retrospective top-\num{<<v>>} precision among corpus", csv(R3 + "topk_metrics.csv", {"dataset": "A", "model": "XGBoost", "protocol": "grouped", "K": 100, "fold": "mean_5folds"}, "K"), ".0f")
P("abs_top_A", r"were held out stays at \SI{<<v>>}{\percent} and \SI{82.4}", TOPK("A", "XGBoost", "grouped", "precision_pct"), ".1f")
P("abs_top_B", r"stays at \SI{87.0}{\percent} and \SI{<<v>>}{\percent} (\SI{78.2}", TOPK("Bv3", "XGBoost", "grouped", "precision_pct"), ".1f")
P("abs_top_Bneg", r"(\SI{<<v>>}{\percent} once measured non-superconductors", NEG("XGBoost", "grouped", "precision_with_neg_pct"), ".1f")
P("abs_prev_lo", r"against fold prevalences of \SIrange{<<v>>}{18.2}{\percent}", PREV("A", "grouped", "min"), ".1f")
P("abs_prev_hi", r"against fold prevalences of \SIrange{3.6}{<<v>>}{\percent}", PREV("A", "grouped", "max"), ".1f")
P("abs_ncomp", r"on a corpus of \num{<<v>>} compositions dated by first report", TPROV("procedencia.n_composiciones"), ".0f")
P("abs_L0_R2", r"pre-2010 data estimates $\Rsq = \num{<<v>>}$ where the realised", TC(2010, "L0_aleatorio", "XGBoost", "R2", "estimacion_retrospectiva"), ".2f")
P("abs_real_R2", r"the realised value on later compositions is \num{<<v>>};", TV(2010, "temporal_real", "XGBoost", "todo", "R2"), ".2f")
P("abs_within01", r"but none is within \num{<<v>>} at more than one cut", const(0.1, "tolerance chosen to summarise table tab:temporal / results_v3/temporal_validation_comparacion.csv"), ".1f")
# ---- introduction
P("intro_R2_092", r"report $\Rsq \gtrsim \num{<<v>>}$ and MAE below", const(0.92, "literature range quoted in the introduction (no result file)"), ".2f")
# ---- 2.1 datasets
P("data_N_A", r"Data Set~<<*>>: \num{<<v>>} rows with \num{81}", FC("A", "N"), ".0f")
P("data_nfeat_A_1", r"\num{21263} rows with \num{<<v>>} statistical features", csv(R2 + "nulo_presencia.csv", {"dataset": "A", "representacion": "features_completas_paper", "split": "aleatorio_KFold5"}, "n_features"), ".0f")
P("data_B_source", r"(\num{<<v>>} entries). Dropping the \num{3966}", csv(R3 + "overlap_AB.csv", {"checkpoint_B": "v3", "criterio": "(iv) sistema elemental (familia)"}, "N_B", scale=None) if False else js(R3 + "semantic_audit_B_decisions.json", "meta.n_source"), ".0f")
P("data_B_tcle0", r"Dropping the \num{<<v>>} entries with $\Tc \le 0$ and \num{8}", expr("a - b - c", a=js(R3 + "semantic_audit_B_decisions.json", "meta.n_source"), b=js(R3 + "semantic_audit_B_decisions.json", "meta.n_parsed_Tc_gt0"), c=js(R2 + "fix_factorize_verificacion.json", "n_parse_failures_pymatgen")), ".0f")
P("data_tcle0_const", r"entries with $\Tc \le <<v>>$ and \num{8} whose formula", const(0, "definition of the Tc>0 filter"), ".0f")
P("data_B_v2N", r"cannot parse leaves \num{<<v>>} rows (the checkpoint", REG("N_v2"), ".0f")
P("data_excl", r"excludes \num{<<v>>} of these and corrects \num{3}, giving", REG("filas_excluidas"), ".0f")
P("data_corr", r"of these and corrects \num{<<v>>}, giving the audited", REG("filas_corregidas"), ".0f")
P("data_N_B", r"used throughout: \num{<<v>>} rows in \num{3046} element systems, featurized", REG("N_v3"), ".0f")
P("data_k_B", r"throughout: \num{12336} rows in \num{<<v>>} element systems, featurized", REG("familias_v3"), ".0f")
# ---- 2.2 audit
P("aud_rows", r"assigns each of the \num{<<v>>} rows a category and a decision", js(R3 + "semantic_audit_B_decisions.json", "meta.n_parsed_Tc_gt0"), ".0f")
P("aud_excl", r"We exclude \num{<<v>>} rows: \num{75} whose oxygen", REG("filas_excluidas"), ".0f")
P("aud_OY", r"We exclude \num{104} rows: \num{<<v>>} whose oxygen", AUD({"decision_final_fila": "excluir", "categoria": "OY_oxigeno_indeterminado"}), ".0f")
P("aud_OY_spur", r"\texttt{OY} marker (\num{<<v>>} of them had been assigned", REG("familias_OY_espurias_filas") if False else const(54, "results_v3/H03_H04_resumen.md: 54 OY rows carried a spurious yttrium family (count in data/audit/semantic_audit_B.csv, column interpretacion)"), ".0f")
P("aud_Ofinal", r"yttrium-containing system), \num{<<v>>} ending in \texttt{O} without", AUD({"decision_final_fila": "excluir", "categoria": "O_final_sin_coeficiente"}), ".0f")
P("aud_multi", r"without a coefficient, \num{<<v>>} with another element lacking", AUD({"decision_final_fila": "excluir", "categoria": "elemento_sin_coeficiente_multielemental"}), ".0f")
P("aud_typo", r"multi-element formula, \num{<<v>>} unrecoverable typographical", AUD({"decision_final_fila": "excluir", "categoria": "coeficiente_extremo_error_tipografico"}), ".0f")
P("aud_artefact", r"typographical coefficients and \num{<<v>>} artefact of string", AUD({"decision_final_fila": "excluir", "categoria": "elemento_repetido_artefacto"}), ".0f")
P("aud_corr3", r"concatenation. We correct \num{<<v>>} formulas with a misplaced", REG("filas_corregidas"), ".0f")
P("aud_dup760", r"repeated elements and the \num{<<v>>} rows that duplicate another", js(R3 + "semantic_audit_B_decisions.json", "meta.n_dup_rows_1e6"), ".0f")
P("aud_fam17", r"No retained row changes system; \num{<<v>>} systems disappear", expr("a - b", a=REG("familias_v2"), b=REG("familias_v3")), ".0f")
P("aud_fam12", r"systems disappear, \num{<<v>>} of which existed only through", const(12, "results_v3/H03_H04_resumen.md: 12 of the 17 vanished systems existed only through OY rows"), ".0f")
P("aud_part9453", r"changes for \num{<<v>>} of the \num{12336} retained rows", REG("filas_retenidas_que_cambian_de_fold_GroupKFold") if False else const(9453, "results_v3/H03_H04_resumen.md / regeneracion_B_v3.json: rows whose deterministic GroupKFold fold changes between v2 and v3"), ".0f")
P("aud_part_N", r"changes for \num{9453} of the \num{<<v>>} retained rows", REG("N_v3"), ".0f")
P("aud_A_1927", r"composition table: \num{<<v>>} formulas (\SI{9.1}{\percent}", js(R3 + "A_nomenclatura_check.json", "filas_O_final_sin_coeficiente"), ".0f")
P("aud_A_pct", r"\num{1927} formulas (\SI{<<v>>}{\percent} of the rows) end in", expr("100*a/b", a=js(R3 + "A_nomenclatura_check.json", "filas_O_final_sin_coeficiente"), b=FC("A", "N")), ".1f")
# ---- 2.3 overlap
P("ov_str", r"nearly complete: \SI{<<v>>}{\percent} of the audited Dataset~B rows have an identical formula", csv(R3 + "overlap_AB.csv", {"checkpoint_B": "v3", "criterio": "(i) cadena de formula identica"}, "frac_filas_B_en_A", scale=100), ".1f")
P("ov_comp", r"in Dataset~A, \SI{<<v>>}{\percent} an identical normalised composition", csv(R3 + "overlap_AB.csv", {"checkpoint_B": "v3", "criterio": "(ii) composicion normalizada 1e-3"}, "frac_filas_B_en_A", scale=100), ".1f")
P("ov_sys", r"normalised composition, and \SI{<<v>>}{\percent} belong to an element system present", csv(R3 + "overlap_AB.csv", {"checkpoint_B": "v3", "criterio": "(iv) sistema elemental (familia)"}, "frac_filas_B_en_A", scale=100), ".1f")
P("ov_2883", r"present in~A (\num{<<v>>} of the \num{3046} systems", OV("(iv) sistema elemental (familia)", "familias_B_con_alguna_fila_en_A"), ".0f")
P("ov_kB", r"present in~A (\num{2883} of the \num{<<v>>} systems", OV("(iv) sistema elemental (familia)", "k_B"), ".0f")
P("ov_inv", r"conversely \SI{<<v>>}{\percent} of the Dataset~A rows belong", csv(R3 + "overlap_AB.csv", {"checkpoint_B": "v3", "criterio": "(iv-inv) filas de A cuya familia esta en B"}, "frac_filas_B_en_A", scale=100), ".1f")
OCT = R3 + "overlap_complementos_tamanos.json"
P("ov_Bout_rows", r"unrepresentative: \num{<<v>>} Dataset~B rows in \num{163}", js(OCT, "B_out_filas"), ".0f")
P("ov_Bout_fam", r"Dataset~B rows in \num{<<v>>} systems absent from~A (mean", js(OCT, "B_out_familias"), ".0f")
P("ov_Bout_tc", r"systems absent from~A (mean $\Tc$ \SI{<<v>>}{\kelvin}) and", js(OCT, "B_out_Tc_media"), ".1f")
P("ov_Aout_rows", r"\SI{5.7}{\kelvin}) and \num{<<v>>} Dataset~A rows in \num{482}", js(OCT, "A_out_filas"), ".0f")
P("ov_Aout_fam", r"Dataset~A rows in \num{<<v>>} systems absent from~B", js(OCT, "A_out_familias"), ".0f")
P("ov_Aout_tc", r"systems absent from~B (mean $\Tc$ \SI{<<v>>}{\kelvin}, \num{683}", js(OCT, "A_out_Tc_media"), ".1f")
P("ov_Aout_O", r"\SI{64.3}{\kelvin}, \num{<<v>>} of them with the coefficient-less", js(OCT, "A_out_filas_con_O_final_sin_coeficiente"), ".0f")
# ---- 2.4 families
Y = R3 + "h09_ybco_family_check.json"
P("fam_720", r"holds \num{<<v>>} Dataset~A rows in \num{184} distinct", js(Y, "datasetA.family_rows"), ".0f")
P("fam_184", r"Dataset~A rows in \num{<<v>>} distinct formulas", js(Y, "datasetA.family_distinct_formulas"), ".0f")
P("fam_tcmin", r"distinct formulas with $\Tc$ from \SI{<<v>>}{\kelvin} to \SI{130}", js(Y, "datasetA.family_Tc_min_K"), ".1f")
P("fam_tcmax", r"from \SI{4.3}{\kelvin} to \SI{<<v>>}{\kelvin}, and it merges", js(Y, "datasetA.family_Tc_max_K"), ".0f")
P("fam_123", r"phases (\num{<<v>>}, \num{38} and \num{51} rows", js(Y, "datasetA.rows_CuY_eq_3_(123-type)"), ".0f")
P("fam_247", r"phases (\num{596}, \num{<<v>>} and \num{51} rows", const(38, "results_v3/h09_ybco_family_check.json datasetA.rows_CuY_eq_3.5_(247-type) (key contains a dot, not addressable by json path)"), ".0f")
P("fam_124", r"(\num{596}, \num{38} and \num{<<v>>} rows with Cu:Y", js(Y, "datasetA.rows_CuY_eq_4_(124-type)"), ".0f")
P("fam_ratio3", r"rows with Cu:Y ratios of \num{<<v>>}, \num{3.5} and", const(3, "Cu:Y stoichiometric ratio of the 123 phase (definition)"), ".0f")
P("fam_ratio35", r"Cu:Y ratios of \num{3}, \num{<<v>>} and \num{4})", const(3.5, "Cu:Y stoichiometric ratio of the 247 phase (definition)"), ".1f")
P("fam_ratio4", r"ratios of \num{3}, \num{3.5} and \num{<<v>>})~<<*>>. Holding out a system", const(4, "Cu:Y stoichiometric ratio of the 124 phase (definition)"), ".0f")
P("fam_A_k", r"Dataset~A has \num{<<v>>} systems for \num{21263} rows", FC("A", "k_familias"), ".0f")
P("fam_A_N", r"has \num{3365} systems for \num{<<v>>} rows, \SI{79.5}", FC("A", "N"), ".0f")
P("fam_A_ge5", r"for \num{21263} rows, \SI{<<v>>}{\percent} of the rows in systems of five", FC("A", "pct_en_familias_ge5"), ".1f")
P("fam_A_top1", r"the three largest holding \num{<<v>>}, \num{358} and", csv(R3 + "family_concentration_v3.csv", {"dataset": "A"}, "top3_tam;0"), ".0f")
P("fam_A_top2", r"three largest holding \num{720}, \num{<<v>>} and \num{323}", csv(R3 + "family_concentration_v3.csv", {"dataset": "A"}, "top3_tam;1"), ".0f")
P("fam_A_top3", r"holding \num{720}, \num{358} and \num{<<v>>} rows, and", csv(R3 + "family_concentration_v3.csv", {"dataset": "A"}, "top3_tam;2"), ".0f")
P("fam_A_single", r"\num{323} rows, and \num{<<v>>} singletons. Dataset~B", FC("A", "n_singleton"), ".0f")
P("fam_B_k", r"singletons. Dataset~B has \num{<<v>>} systems for \num{12336}", FC("B", "k_familias"), ".0f")
P("fam_B_N", r"Dataset~B has \num{3046} systems for \num{<<v>>} rows, \SI{66.5}", FC("B", "N"), ".0f")
P("fam_B_ge5", r"for \num{12336} rows, \SI{<<v>>}{\percent} in systems of five or more", FC("B", "pct_en_familias_ge5"), ".1f")
P("fam_B_max", r"a largest system of \num{<<v>>} rows and \num{1316}", FC("B", "tam_familia_max"), ".0f")
P("fam_B_single", r"largest system of \num{131} rows and \num{<<v>>} singletons", FC("B", "n_singleton"), ".0f")
# ---- 2.5 protocols
P("prot_seed0", r"shuffled splitters over seeds \numrange{<<v>>}{4} (\texttt{GroupKFold", csv(R3 + "tabla2_runs_por_semilla_v3.csv", {"dataset": "B"}, "seed", "min"), ".0f")
P("prot_seed4", r"shuffled splitters over seeds \numrange{0}{<<v>>} (\texttt{GroupKFold", csv(R3 + "tabla2_runs_por_semilla_v3.csv", {"dataset": "B"}, "seed", "max"), ".0f")
P("prot_skl16", r"available since scikit-learn~<<ver>>), so the grouped", const("1.6", "scikit-learn release that introduced GroupKFold(shuffle=True)"))
P("prot_1e3", r"(fractions rounded to $<<v>>^{-3}$, so replicate", const(10, "rounding base of the L1 composition key (10^-3)"), ".0f")
P("prot_host005", r"molar fraction at least \num{<<v>>} (dopants below", const(0.05, "L3 host threshold, definition (results_v3/separation_levels.csv level L3_anfitrion_0.05)"), ".2f")
P("prot_host002", r"threshold are ignored; \num{<<v>>} and \num{0.10} as sensitivity", const(0.02, "L3 sensitivity threshold (level L3_anfitrion_0.02)"), ".2f")
P("prot_host010", r"ignored; \num{0.02} and \num{<<v>>} as sensitivity", const(0.10, "L3 sensitivity threshold (level L3_anfitrion_0.10)"), ".2f")
# ---- 2.6 models
P("hp_B_rf_leaf2", r"\texttt{min\_samples\_split} \num{<<v>>}. Both configurations", expr("a if a == b else float('nan')", a=PY("code/tc_pipeline.py", "make_models(42)['RandomForest'].get_params()['min_samples_split']"), b=RFB("min_samples_split")), ".0f")
P("hp_grid16", r"arms and over a \num{<<v>>}-point grid. Regression", csv(R3 + "hp_grid_ambos_v3.csv", {"dataset": "B", "es_publicada": "False"}, "config", "count"), ".0f")
# ---- 2.7 representations
P("rep_R0_A", r"element presence (\num{<<v>>} indicators for~A, \num{83}", ABL("A", "random", "R0", "n_features"), ".0f")
P("rep_R0_B", r"(\num{86} indicators for~A, \num{<<v>>} for~B)", ABL("B", "random", "R0", "n_features"), ".0f")
P("rep_R1_A", r"presence plus molar fractions (\num{<<v>>} and \num{166} columns)", ABL("A", "random", "R1", "n_features"), ".0f")
P("rep_R1_B", r"molar fractions (\num{172} and \num{<<v>>} columns)", ABL("B", "random", "R1", "n_features"), ".0f")
P("rep_R2_A", r"the published descriptors (\num{<<v>>} and \num{132}), and", ABL("A", "random", "R2", "n_features"), ".0f")
P("rep_R2_B", r"published descriptors (\num{81} and \num{<<v>>}), and", ABL("B", "random", "R2", "n_features"), ".0f")
P("rep_R3_A", r"descriptors plus fractions (\num{<<v>>} and \num{298})", ABL("A", "random", "R3", "n_features"), ".0f")
P("rep_R3_B", r"descriptors plus fractions (\num{253} and \num{<<v>>})", ABL("B", "random", "R3", "n_features"), ".0f")
P("rep_ET500", r"extra-trees ensemble of \num{<<v>>} trees on R2", const(500, "n_estimators of ET500_R2 in code/v3/ablation_representation.py"), ".0f")
P("rep_sse1", r"sum of squared errors, $<<v>> - \mathrm{SSE}_b/\mathrm{SSE}_a$)", const(1, "algebraic identity"), ".0f")
# ---- 2.8 clustering formulas (constants of the algebra)
P("cl_var1", r"is $\sigma^2 N^{-2}\,[(<<v>>-\rho)N + \rho\sum_i n_i^2]$", const(1, "algebraic identity"), ".0f")
P("cl_deff1", r"\mathrm{deff}_{\mathrm{rows}} = <<v>> + \rho\,(m_w - 1)", const(1, "algebraic identity"), ".0f")
P("cl_deff1b", r"\rho\,(m_w - <<v>>), \qquad m_w", const(1, "algebraic identity"), ".0f")
P("cl_inv", r"(rows weighted $<<v>>/n_i$)", const(1, "algebraic identity"), ".0f")
P("cl_sys1", r"\sum_i[\rho + (<<v>>-\rho)/n_i]$. Neither equals", const(1, "algebraic identity"), ".0f")
P("cl_m0a", r"Neither equals $<<v>> + \rho(m_0 - 1)$, which the previous", const(1, "algebraic identity"), ".0f")
P("cl_m0b", r"Neither equals $1 + \rho(m_0 - <<v>>)$, which the previous", const(1, "algebraic identity"), ".0f")
P("cl_icc_m0", r"m_0 = \frac{N - \sum_i n_i^2 / N}{k - <<v>>}", const(1, "Searle m0 definition"), ".0f")
P("cl_eta1", r"is $\eta^2 = <<v>> - \mathrm{SS_{within}}", const(1, "algebraic identity"), ".0f")
P("cl_icc_1", r"(m_0 - <<v>>)\,\mathrm{MS_W}}", const(1, "ICC(1) definition"), ".0f")
# ---- 2.9 selection
P("sel_10a", r"close to, but not equal to, \SI{<<v>>}{\percent}: under the element-system", const(10, "nominal base rate implied by a 90th-percentile threshold"), ".0f")
P("sel_10b", r"that fold's prevalence---not \SI{<<v>>}{\percent}. For", const(10, "nominal base rate implied by a 90th-percentile threshold"), ".0f")
P("sel_prevA_lo", r"prevalence ranges from \SI{<<v>>}{\percent} to \SI{18.2}{\percent} across", PREV("A", "grouped", "min"), ".1f")
P("sel_prevA_hi", r"ranges from \SI{3.6}{\percent} to \SI{<<v>>}{\percent} across the five", PREV("A", "grouped", "max"), ".1f")
P("sel_prevB_lo", r"folds and from \SI{<<v>>}{\percent} to \SI{13.5}{\percent} in Dataset~B", PREV("Bv3", "grouped", "min"), ".1f")
P("sel_prevB_hi", r"from \SI{6.3}{\percent} to \SI{<<v>>}{\percent} in Dataset~B", PREV("Bv3", "grouped", "max"), ".1f")
P("sel_K", r"For $K = \num{<<v>>}$ we report precision", const(100, "K of the selection metric (definition)"), ".0f")
NEGJ = R3 + "topk_negatives_summary.json"
P("sel_neg3925", r"the precision when the \num{<<v>>} measured non-superconductors of the Stanev list", NEG("XGBoost", "grouped", "n_neg_pool"), ".0f")
P("sel_neg_tc0", r"Stanev list ($\Tc \le <<v>>$, \SI{24.1}", const(0, "definition of a measured non-superconductor"), ".0f")
P("sel_neg_pct", r"($\Tc \le 0$, \SI{<<v>>}{\percent} of the enlarged pool)", NEG("XGBoost", "grouped", "frac_neg_pool_pct"), ".1f")
# ---- 2.10 inference
P("inf_95", r"All intervals in this paper are \SI{<<v>>}{\percent} percentile", const(95, "confidence level (percentile 2.5/97.5)"), ".0f")
P("inf_2000", r"whole element systems} (\num{<<v>>} resamples, seed~0)", PFB("A", "XGBoost", "MAE_random", "n_boot"), ".0f")
WR = lambda ds, m, s: csv(R3 + "paired_family_bootstrap.csv", {"dataset": ds, "model": m, "statistic": s, "scheme": "family_paired"}, "width_ratio_family_over_row_unpaired")
P("inf_ratio_lo", r"which the previous version used, are \numrange{<<v>>}{5.6} times narrower for the same statistics", expr("min(w0,w1,w2,w3,w4,w5,w6,w7)", **{f"w{i}": WR(ds, "XGBoost", st) for i, (ds, st) in enumerate([(d, t) for d in "AB" for t in ("MAE_random", "MAE_grouped", "inflacion_pct", "dR2")])}), ".1f")
P("inf_ratio_hi", r"previous version used, are \numrange{1.5}{<<v>>} times narrower for the same statistics", expr("max(w0,w1,w2,w3,w4,w5,w6,w7)", **{f"w{i}": WR(ds, "XGBoost", st) for i, (ds, st) in enumerate([(d, t) for d in "AB" for t in ("MAE_random", "MAE_grouped", "inflacion_pct", "dR2")])}), ".1f")
# ---- 2.11 NN
P("nn_80", r"under a random $<<v>>/20$ hold-out and a grouped", const(80, "80/20 hold-out (definition)"), ".0f")
P("nn_20", r"under a random $80/<<v>>$ hold-out and a grouped", const(20, "80/20 hold-out (definition)"), ".0f")
# ---- 2.12 temporal corpus
P("tc_raw", r"SuperCon export (\num{<<v>>} reports; not redistributed", TPROV("n_raw"), ".0f")
P("tc_year", r"bibliographic field (\SI{<<v>>}{\percent} of reports; the year", TPROV("year_extractable_pct_all"), ".1f")
P("tc_tc0", r"formulas, reports with $\Tc \le <<v>>$ were dropped, and the \num{20771}", const(0, "definition of the Tc>0 filter"), ".0f")
P("tc_reports", r"were dropped, and the \num{<<v>>} dated reports were collapsed", TPROV("n_reports_kept_with_year"), ".0f")
P("tc_ncomp", r"reports were collapsed to \num{<<v>>} normalised compositions in \num{3267}", TPROV("procedencia.n_composiciones"), ".0f")
P("tc_nsys", r"normalised compositions in \num{<<v>>} element systems, each dated", TMETA("meta.n_familias_corpus") if False else const(3267, "results_v3/temporal_validation.md / temporal_corpus.py: element systems of the dated corpus"), ".0f")
# ---- 3.1 repro
P("rep_A_mae", r"gives MAE $=\SI{<<v>>}{\kelvin}$ and $\Rsq = \num{0.928}$ for XGBoost on Dataset~A", PFB("A", "XGBoost", "MAE_random"), ".2f")
P("rep_A_r2", r"and $\Rsq = \num{<<v>>}$ for XGBoost on Dataset~A", PFB("A", "XGBoost", "R2_random"), ".3f")
P("rep_B_mae", r"and MAE $=\SI{<<v>>}{\kelvin}$ with $\Rsq = \num{0.932}$ on Dataset~B", PFB("B", "XGBoost", "MAE_random"), ".2f")
P("rep_B_r2", r"with $\Rsq = \num{<<v>>}$ on Dataset~B (table", PFB("B", "XGBoost", "R2_random"), ".3f")
T1B = lambda ck, mod, split, field: csv(R3 + "tabla1_B_antes_despues.csv", {"checkpoint": ck, "modelo": mod, "split": split, "random_state": 42}, field)
T1I = lambda ck, mod, field: csv(R3 + "tabla1_B_inflacion_antes_despues.csv", {"checkpoint": ck, "modelo": mod}, field)
P("rep_v2N", r"previous checkpoint (\num{<<v>>} rows) the random-split", REG("N_v2"), ".0f")
P("rep_v2_rand", r"XGBoost MAE falls from \SI{<<v>>}{\kelvin} to \SI{3.78}{\kelvin} and the grouped", T1B("v2", "XGBoost", "random", "MAE"), ".2f")
P("rep_v3_rand", r"MAE falls from \SI{3.91}{\kelvin} to \SI{<<v>>}{\kelvin} and the grouped MAE", T1B("v3", "XGBoost", "random", "MAE"), ".2f")
P("rep_v2_grp", r"grouped MAE is unchanged (\SI{<<v>>}{\kelvin} to \SI{6.18}", T1B("v2", "XGBoost", "grouped", "MAE"), ".2f")
P("rep_v3_grp", r"is unchanged (\SI{6.17}{\kelvin} to \SI{<<v>>}{\kelvin}), so the XGBoost", T1B("v3", "XGBoost", "grouped", "MAE"), ".2f")
P("rep_v2_infl", r"XGBoost inflation rises from \SI{<<v>>}{\percent} to \SI{63.7}", expr("100*(a-1)", a=T1I("v2", "XGBoost", "inflacion_MAE_ratio")), ".1f")
P("rep_v3_infl", r"inflation rises from \SI{57.6}{\percent} to \SI{<<v>>}{\percent}; for the random forest", expr("100*(a-1)", a=T1I("v3", "XGBoost", "inflacion_MAE_ratio")), ".1f")
P("rep_v2_infl_rf", r"random forest it rises from \SI{<<v>>}{\percent} to \SI{53.0}", expr("100*(a-1)", a=T1I("v2", "RandomForest", "inflacion_MAE_ratio")), ".1f")
P("rep_v3_infl_rf", r"it rises from \SI{49.9}{\percent} to \SI{<<v>>}{\percent}. Part", expr("100*(a-1)", a=T1I("v3", "RandomForest", "inflacion_MAE_ratio")), ".1f")
# table 1 caption
NR = lambda ds, agg: expr(f"{agg}(abs(a),abs(b),abs(c),abs(d),abs(f))", **{k: csv(R3 + "null_r2_by_fold.csv", {"dataset": ds, "fold": str(i)}, "diff_K") for k, i in zip("abcdf", range(5))})
P("cap1_diff_lo", r"differs from the test-fold mean by \SIrange{<<v>>}{4.4}{\kelvin} across folds", NR("B", "min"), ".1f")
P("cap1_diff_hi", r"differs from the test-fold mean by \SIrange{0.8}{<<v>>}{\kelvin} across folds", NR("B", "max"), ".1f")
P("cap1_seed42", r"cross-validation (seed \num{<<v>>}; deterministic", csv(R3 + "tabla1_B_antes_despues.csv", {"checkpoint": "v3", "modelo": "XGBoost", "split": "random"}, "random_state"), ".0f")
# fig 1 caption
P("fig1_n", r"element-system size ($n=\num{<<v>>}$ rows in $k=\num{3365}$ systems", FC("A", "N"), ".0f")
P("fig1_k", r"rows in $k=\num{<<v>>}$ systems; the largest", FC("A", "k_familias"), ".0f")
P("fig1_720", r"$\{$Ba, Cu, O, Y$\}$, holds \num{<<v>>} rows). (b)", FC("A", "tam_familia_max"), ".0f")
P("fig1_seeds0", r"five shuffled partitions (seeds \numrange{<<v>>}{4}), error bars", csv(R3 + "tabla2_runs_por_semilla_v3.csv", {"dataset": "B"}, "seed", "min"), ".0f")
P("fig1_seeds4", r"five shuffled partitions (seeds \numrange{0}{<<v>>}), error bars", csv(R3 + "tabla2_runs_por_semilla_v3.csv", {"dataset": "B"}, "seed", "max"), ".0f")
P("fig1_mae", r"deterministic partition (MAE $=\SI{<<v>>}{\kelvin}$, $\Rsq=\num{0.85}$", PFB("A", "XGBoost", "MAE_grouped"), ".1f")
P("fig1_r2", r"(MAE $=\SI{8.3}{\kelvin}$, $\Rsq=\num{<<v>>}$, both", PFB("A", "XGBoost", "R2_grouped"), ".2f")
P("fig1_nrand", r"hold-out per scheme ($n=\num{<<v>>}$ random and \num{4344} grouped", NN("A", "n_test_random"), ".0f")
P("fig1_ngrp", r"random and \num{<<v>>} grouped test rows; clipped", NN("A", "n_test_grouped"), ".0f")
P("fig1_clip", r"clipped at \num{<<v>>} for display). The median", const(3, "display clip of the histogram in code/v3/make_figures_v3.py"), ".0f")
P("fig1_ratio", r"The median test row is \num{<<v>>} times farther from its nearest", NN("A", "ratio_trainscaler"), ".0f")
# ---- 3.2 gap
P("gap_A_r", r"raises XGBoost's pooled MAE from \SI{<<v>>}{\kelvin} to \SI{8.31}", PFB("A", "XGBoost", "MAE_random"), ".2f")
P("gap_A_dmae", r"on Dataset~A ($\Delta$MAE $=\SI{<<v>>}{\kelvin}$ \ci{2.58}{3.65}, inflation", PFB("A", "XGBoost", "dMAE_K"), ".2f")
P("gap_A_dmae_lo", r"($\Delta$MAE $=\SI{3.08}{\kelvin}$ \ci{<<v>>}{3.65}, inflation", PFB("A", "XGBoost", "dMAE_K", "lo"), ".2f")
P("gap_A_dmae_hi", r"($\Delta$MAE $=\SI{3.08}{\kelvin}$ \ci{2.58}{<<v>>}, inflation", PFB("A", "XGBoost", "dMAE_K", "hi"), ".2f")
P("gap_A_infl", r"\ci{2.58}{3.65}, inflation \SI{<<v>>}{\percent} \ci{50.4}{68.6}) and from", PFB("A", "XGBoost", "inflacion_pct"), ".1f")
P("gap_A_infl_lo", r"inflation \SI{58.9}{\percent} \ci{<<v>>}{68.6}) and from", PFB("A", "XGBoost", "inflacion_pct", "lo"), ".1f")
P("gap_A_infl_hi", r"inflation \SI{58.9}{\percent} \ci{50.4}{<<v>>}) and from", PFB("A", "XGBoost", "inflacion_pct", "hi"), ".1f")
P("gap_B_r", r"and from \SI{<<v>>}{\kelvin} to \SI{6.18}{\kelvin} on Dataset~B (\SI{2.41}", PFB("B", "XGBoost", "MAE_random"), ".2f")
P("gap_B_g", r"from \SI{3.78}{\kelvin} to \SI{<<v>>}{\kelvin} on Dataset~B (\SI{2.41}", PFB("B", "XGBoost", "MAE_grouped"), ".2f")
P("gap_B_dmae", r"on Dataset~B (\SI{<<v>>}{\kelvin} \ci{2.01}{2.88}; \SI{63.7}", PFB("B", "XGBoost", "dMAE_K"), ".2f")
P("gap_B_dmae_lo", r"on Dataset~B (\SI{2.41}{\kelvin} \ci{<<v>>}{2.88};", PFB("B", "XGBoost", "dMAE_K", "lo"), ".2f")
P("gap_B_dmae_hi", r"on Dataset~B (\SI{2.41}{\kelvin} \ci{2.01}{<<v>>};", PFB("B", "XGBoost", "dMAE_K", "hi"), ".2f")
P("gap_B_infl", r"\ci{2.01}{2.88}; \SI{<<v>>}{\percent} \ci{54.6}{74.3}), and lowers", PFB("B", "XGBoost", "inflacion_pct"), ".1f")
P("gap_B_infl_lo", r"\SI{63.7}{\percent} \ci{<<v>>}{74.3}), and lowers", PFB("B", "XGBoost", "inflacion_pct", "lo"), ".1f")
P("gap_B_infl_hi", r"\SI{63.7}{\percent} \ci{54.6}{<<v>>}), and lowers", PFB("B", "XGBoost", "inflacion_pct", "hi"), ".1f")
P("gap_A_dr2", r"and lowers $\Rsq$ by \num{<<v>>} \ci{0.058}{0.094} and \num{0.084}", PFB("A", "XGBoost", "dR2"), ".3f")
P("gap_A_dr2_lo", r"lowers $\Rsq$ by \num{0.074} \ci{<<v>>}{0.094}", PFB("A", "XGBoost", "dR2", "lo"), ".3f")
P("gap_A_dr2_hi", r"lowers $\Rsq$ by \num{0.074} \ci{0.058}{<<v>>}", PFB("A", "XGBoost", "dR2", "hi"), ".3f")
P("gap_B_dr2", r"\ci{0.058}{0.094} and \num{<<v>>} \ci{0.066}{0.109}. The random forest", PFB("B", "XGBoost", "dR2"), ".3f")
P("gap_B_dr2_lo", r"and \num{0.084} \ci{<<v>>}{0.109}. The random forest", PFB("B", "XGBoost", "dR2", "lo"), ".3f")
P("gap_B_dr2_hi", r"and \num{0.084} \ci{0.066}{<<v>>}. The random forest", PFB("B", "XGBoost", "dR2", "hi"), ".3f")
P("gap_rf_A", r"behaves alike (\SI{<<v>>}{\percent} \ci{52.2}{71.5} and \SI{53.0}", PFB("A", "RandomForest", "inflacion_pct"), ".1f")
P("gap_rf_A_lo", r"behaves alike (\SI{61.4}{\percent} \ci{<<v>>}{71.5} and", PFB("A", "RandomForest", "inflacion_pct", "lo"), ".1f")
P("gap_rf_A_hi", r"behaves alike (\SI{61.4}{\percent} \ci{52.2}{<<v>>} and", PFB("A", "RandomForest", "inflacion_pct", "hi"), ".1f")
P("gap_rf_B", r"\ci{52.2}{71.5} and \SI{<<v>>}{\percent} \ci{44.5}{62.8}). All", PFB("B", "RandomForest", "inflacion_pct"), ".1f")
P("gap_rf_B_lo", r"and \SI{53.0}{\percent} \ci{<<v>>}{62.8}). All", PFB("B", "RandomForest", "inflacion_pct", "lo"), ".1f")
P("gap_rf_B_hi", r"and \SI{53.0}{\percent} \ci{44.5}{<<v>>}). All", PFB("B", "RandomForest", "inflacion_pct", "hi"), ".1f")
P("mac_A_r", r"macro MAE of XGBoost rises from \SI{<<v>>}{\kelvin} to \SI{6.43}", PFB("A", "XGBoost", "MAE_macro_random"), ".2f")
P("mac_A_g", r"rises from \SI{5.07}{\kelvin} to \SI{<<v>>}{\kelvin} on~A (\SI{27.0}", PFB("A", "XGBoost", "MAE_macro_grouped"), ".2f")
P("mac_A_infl", r"on~A (\SI{<<v>>}{\percent} \ci{23.8}{30.5}) and from \SI{3.50}", PFB("A", "XGBoost", "inflacion_macro_pct"), ".1f")
P("mac_A_lo", r"on~A (\SI{27.0}{\percent} \ci{<<v>>}{30.5}) and", PFB("A", "XGBoost", "inflacion_macro_pct", "lo"), ".1f")
P("mac_A_hi", r"on~A (\SI{27.0}{\percent} \ci{23.8}{<<v>>}) and", PFB("A", "XGBoost", "inflacion_macro_pct", "hi"), ".1f")
P("mac_B_r", r"\ci{23.8}{30.5}) and from \SI{<<v>>}{\kelvin} to \SI{4.53}{\kelvin} on~B", PFB("B", "XGBoost", "MAE_macro_random"), ".2f")
P("mac_B_g", r"from \SI{3.50}{\kelvin} to \SI{<<v>>}{\kelvin} on~B (\SI{29.4}", PFB("B", "XGBoost", "MAE_macro_grouped"), ".2f")
P("mac_B_infl", r"on~B (\SI{<<v>>}{\percent} \ci{25.4}{33.8}); for the random forest", PFB("B", "XGBoost", "inflacion_macro_pct"), ".1f")
P("mac_B_lo", r"on~B (\SI{29.4}{\percent} \ci{<<v>>}{33.8}); for", PFB("B", "XGBoost", "inflacion_macro_pct", "lo"), ".1f")
P("mac_B_hi", r"on~B (\SI{29.4}{\percent} \ci{25.4}{<<v>>}); for", PFB("B", "XGBoost", "inflacion_macro_pct", "hi"), ".1f")
P("mac_rf_A", r"for the random forest \SI{<<v>>}{\percent} and \SI{21.9}{\percent}. The pooled", PFB("A", "RandomForest", "inflacion_macro_pct"), ".1f")
P("mac_rf_B", r"for the random forest \SI{25.5}{\percent} and \SI{<<v>>}{\percent}. The pooled", PFB("B", "RandomForest", "inflacion_macro_pct"), ".1f")
P("gap_share966", r"since they share \SI{<<v>>}{\percent} of their systems (section", csv(R3 + "overlap_AB.csv", {"checkpoint_B": "v3", "criterio": "(iv) sistema elemental (familia)"}, "frac_filas_B_en_A", scale=100), ".1f")
P("nn_A_r", r"closest training row is \num{<<v>>} in Dataset~A; under a grouped", NN("A", "nn_median_random_trainscaler"), ".3f")
P("nn_A_g", r"hold-out of the same size it is \num{<<v>>}, a factor of \num{23}", NN("A", "nn_median_grouped_trainscaler"), ".2f")
P("nn_A_ratio", r"it is \num{1.26}, a factor of \num{<<v>>}. For Dataset~B", NN("A", "ratio_trainscaler"), ".0f")
P("nn_B_r", r"For Dataset~B the medians are \num{<<v>>} and \num{1.51}, a", NN("B", "nn_median_random_trainscaler"), ".3f")
P("nn_B_g", r"the medians are \num{0.149} and \num{<<v>>}, a factor of \num{10}", NN("B", "nn_median_grouped_trainscaler"), ".2f")
P("nn_B_ratio", r"and \num{1.51}, a factor of \num{<<v>>}. Counting directly", NN("B", "ratio_trainscaler"), ".0f")
P("twins_pct", r"Counting directly, \SI{<<v>>}{\percent} of Dataset~A test rows under five-fold", csv(R3 + "replicate_dispersion.csv", {"metrica": "v2_frac_test_con_gemelo_exacto_en_train_random_KFold5_rs42"}, "valor", scale=100), ".1f")
P("twins_pct0", r"under system-aware splitting the fraction is \SI{<<v>>}{\percent}. For more", csv(R3 + "replicate_dispersion.csv", {"metrica": "v2_frac_test_con_gemelo_exacto_en_train_GroupKFold5_familias"}, "valor", scale=100), ".1f")
P("disp_8511", r"Among the \num{<<v>>} Dataset~A rows that share their feature vector", RD("n_filas_con_vector_repetido"), ".0f")
P("disp_2418", r"with at least one other row (\num{<<v>>} groups), the mean absolute", RD("n_grupos"), ".0f")
P("disp_322", r"from the group median is \SI{<<v>>}{\kelvin} (\SI{3.49}{\kelvin} from the group mean)", RD("desv_abs_media_a_MEDIANA_del_grupo_K"), ".2f")
P("disp_349", r"group median is \SI{3.22}{\kelvin} (\SI{<<v>>}{\kelvin} from the group mean)", RD("desv_abs_media_a_MEDIA_del_grupo_K"), ".2f")
P("disp_472", r"replicates of the group) reaches MAE \SI{<<v>>}{\kelvin} on these rows, against \SI{4.83}", RD("baseline_replica_excluida_MEDIANA_otras_MAE_K"), ".2f")
P("disp_483", r"on these rows, against \SI{<<v>>}{\kelvin} for XGBoost under random splitting and", RD("MAE_modelo_XGBoost_random_filas_con_vector_repetido_K"), ".2f")
P("disp_781", r"under random splitting and \SI{<<v>>}{\kelvin} under grouped splitting", RD("MAE_modelo_XGBoost_grouped_filas_con_vector_repetido_K"), ".2f")
P("sens_1927", r"Removing the \num{<<v>>} Dataset~A rows with coefficient-less oxygen", js(R3 + "A_nomenclatura_check.json", "filas_O_final_sin_coeficiente"), ".0f")
P("sens_N", r"coefficient-less oxygen (section~<<*>>; \num{<<v>>} rows and \num{3171} systems remain", ASENS("XGBoost", "N"), ".0f")
P("sens_k", r"\num{19336} rows and \num{<<v>>} systems remain, folds redrawn", const(3171, "results_v3/tabla1_A_sin_O1.md: families remaining after removing the 1 927 O=1 rows"), ".0f")
P("sens_xgb_infl", r"leaves the XGBoost inflation at \SI{<<v>>}{\percent} \ci{52.8}{72.1} and", ASENS("XGBoost", "inflacion_pct"), ".1f")
P("sens_xgb_lo", r"inflation at \SI{61.9}{\percent} \ci{<<v>>}{72.1} and", ASENS("XGBoost", "inflacion_lo"), ".1f")
P("sens_xgb_hi", r"inflation at \SI{61.9}{\percent} \ci{52.8}{<<v>>} and", ASENS("XGBoost", "inflacion_hi"), ".1f")
P("sens_xgb_dr2", r"and the $\Rsq$ drop at \num{<<v>>}, against \SI{58.9}", ASENS("XGBoost", "dR2"), ".3f")
P("sens_full_infl", r"against \SI{<<v>>}{\percent} and \num{0.074} on the full table", ASENS("XGBoost", "inflacion_full_pct"), ".1f")
P("sens_full_dr2", r"against \SI{58.9}{\percent} and \num{<<v>>} on the full table", ASENS("XGBoost", "dR2_full"), ".3f")
P("sens_rf_full", r"random-forest inflation moves from \SI{<<v>>}{\percent} to \SI{57.9}", ASENS("RandomForest", "inflacion_full_pct"), ".1f")
P("sens_rf", r"moves from \SI{61.4}{\percent} to \SI{<<v>>}{\percent} \ci{48.3}{69.3}", ASENS("RandomForest", "inflacion_pct"), ".1f")
P("sens_rf_lo", r"to \SI{57.9}{\percent} \ci{<<v>>}{69.3}", ASENS("RandomForest", "inflacion_lo"), ".1f")
P("sens_rf_hi", r"to \SI{57.9}{\percent} \ci{48.3}{<<v>>}", ASENS("RandomForest", "inflacion_hi"), ".1f")
P("oc_A_in", r"Dataset~A rows whose system \emph{is} in~B with $\Rsq = \num{<<v>>}$ and the \num{1535}", OC("B", "A", "A_in (familia SI en B)", "R2"), ".3f")
P("oc_A_out_n", r"and the \num{<<v>>} rows whose system is not with", OC("B", "A", "A_out (familia NO en B)", "n_filas"), ".0f")
P("oc_A_out_r2", r"whose system is not with $\Rsq = \num{<<v>>}$ \ci{0.413}{0.570}", OC("B", "A", "A_out (familia NO en B)", "R2"), ".3f")
P("oc_A_out_lo", r"with $\Rsq = \num{0.500}$ \ci{<<v>>}{0.570}", OC("B", "A", "A_out (familia NO en B)", "R2_lo95"), ".3f")
P("oc_A_out_hi", r"with $\Rsq = \num{0.500}$ \ci{0.413}{<<v>>}", OC("B", "A", "A_out (familia NO en B)", "R2_hi95"), ".3f")
P("oc_A_out_mae", r"\ci{0.413}{0.570} (MAE \SI{<<v>>}{\kelvin}); a model trained on all of~A", OC("B", "A", "A_out (familia NO en B)", "MAE"), ".2f")
P("oc_B_out_n", r"predicts the \num{<<v>>} Dataset-B rows in systems absent from~A", OC("A", "B", "B_out (familia NO en A)", "n_filas"), ".0f")
P("oc_B_out_r2", r"systems absent from~A with $\Rsq = \num{<<v>>}$ \ci{0.415}{0.694}", OC("A", "B", "B_out (familia NO en A)", "R2"), ".3f")
P("oc_B_out_lo", r"absent from~A with $\Rsq = \num{0.615}$ \ci{<<v>>}{0.694}", OC("A", "B", "B_out (familia NO en A)", "R2_lo95"), ".3f")
P("oc_B_out_hi", r"absent from~A with $\Rsq = \num{0.615}$ \ci{0.415}{<<v>>}", OC("A", "B", "B_out (familia NO en A)", "R2_hi95"), ".3f")
P("oc_B_out_mae", r"\ci{0.415}{0.694} (MAE \SI{<<v>>}{\kelvin}) and the rest with", OC("A", "B", "B_out (familia NO en A)", "MAE"), ".2f")
P("oc_B_in", r"and the rest with $\Rsq = \num{<<v>>}$. The", OC("A", "B", "B_in (familia SI en A)", "R2"), ".3f")
# ---- 3.3 robust
I5 = lambda agg: expr(f"{agg}(a,b,c,d)", a=T2("A", "XGBoost", "inflacion_pct_mean"), b=T2("A", "RandomForest", "inflacion_pct_mean"), c=T2("B", "XGBoost", "inflacion_pct_mean"), d=T2("B", "RandomForest", "inflacion_pct_mean"))
D5 = lambda agg: expr(f"{agg}(a,b,c,d)", a=T2("A", "XGBoost", "dR2_mean"), b=T2("A", "RandomForest", "dR2_mean"), c=T2("B", "XGBoost", "dR2_mean"), d=T2("B", "RandomForest", "dR2_mean"))
S5 = expr("max(a,b,c,d)", a=T2("A", "XGBoost", "inflacion_pct_std"), b=T2("A", "RandomForest", "inflacion_pct_std"), c=T2("B", "XGBoost", "inflacion_pct_std"), d=T2("B", "RandomForest", "inflacion_pct_std"))
P("rob_lo", r"gives MAE inflations from \SI{<<v>>}{\percent} to \SI{60.5}{\percent} across", I5("min"), ".1f")
P("rob_hi", r"inflations from \SI{51.0}{\percent} to \SI{<<v>>}{\percent} across", I5("max"), ".1f")
P("rob_dr2_a", r"combinations and $\Delta\Rsq$ from \num{<<v>>} to \num{-0.081}, with", D5("max"), ".3f")
P("rob_dr2_b", r"$\Delta\Rsq$ from \num{-0.071} to \num{<<v>>}, with", D5("min"), ".3f")
P("rob_sd", r"standard deviations of at most \num{<<v>>} percentage points", S5, ".1f")
P("cap2_seeds0", r"shuffled partitions (seeds \numrange{<<v>>}{4}; both arms", csv(R3 + "tabla2_runs_por_semilla_v3.csv", {"dataset": "B"}, "seed", "min"), ".0f")
P("cap2_seeds4", r"shuffled partitions (seeds \numrange{0}{<<v>>}; both arms", csv(R3 + "tabla2_runs_por_semilla_v3.csv", {"dataset": "B"}, "seed", "max"), ".0f")
# ---- 3.4 tuning
P("tun_grid16", r"a nested search over a \num{<<v>>}-point XGBoost", csv(R3 + "hp_grid_ambos_v3.csv", {"dataset": "B", "es_publicada": "False"}, "config", "count"), ".0f")
NEST = lambda ds, f: T3(ds, "b_nested", "anidado en ambos brazos", f)
P("tun_nest_A", r"arms the inflation is \SI{<<v>>}{\percent} \ci{56.2}{75.8} on~A", NEST("A", "inflacion_pct"), ".1f")
P("tun_nest_A_lo", r"\SI{65.6}{\percent} \ci{<<v>>}{75.8} on~A", NEST("A", "inflacion_lo"), ".1f")
P("tun_nest_A_hi", r"\SI{65.6}{\percent} \ci{56.2}{<<v>>} on~A", NEST("A", "inflacion_hi"), ".1f")
P("tun_nest_B", r"on~A and \SI{<<v>>}{\percent} \ci{55.4}{74.8} on~B, indistinguishable", NEST("B", "inflacion_pct"), ".1f")
P("tun_nest_B_lo", r"and \SI{64.4}{\percent} \ci{<<v>>}{74.8} on~B", NEST("B", "inflacion_lo"), ".1f")
P("tun_nest_B_hi", r"and \SI{64.4}{\percent} \ci{55.4}{<<v>>} on~B", NEST("B", "inflacion_hi"), ".1f")
P("cap3_16", r"on the outer training fold. \emph{Grid}: the \num{<<v>>} configurations each scored", csv(R3 + "hp_grid_ambos_v3.csv", {"dataset": "B", "es_publicada": "False"}, "config", "count"), ".0f")
P("cap3_seed", r"(shuffle, seed <<v>>) and \texttt{GroupKFold(5)}", const(42, "KFold random_state of the grid (results_v3/hp_grid_ambos_v3.csv procedencia)"), ".0f")
GA = lambda d, f, agg: csv(R3 + "hp_grid_ambos_v3.csv", {"dataset": "A", "es_publicada": "False", "max_depth": d}, f, agg)
GAll = lambda ds, f, agg, key=None: csv(R3 + "hp_grid_ambos_v3.csv", {"dataset": ds, "es_publicada": "False", **(key or {})}, f, agg)
P("grid_A_grp_range", r"the grouped MAE of Dataset~A moves within \SI{<<v>>}{\kelvin}, and within", expr("a-b", a=GAll("A", "MAE_grouped", "max"), b=GAll("A", "MAE_grouped", "min")), ".1f")
P("grid_A_grp_range6", r"and within \SI{<<v>>}{\kelvin} once depth is at least", expr("a-b", a=GAll("A", "MAE_grouped", "max", {"max_depth": [6, 8, 10]}), b=GAll("A", "MAE_grouped", "min", {"max_depth": [6, 8, 10]})), ".1f")
P("grid_A_r4lo", r"falls monotonically with depth, from \SIrange<<*>>{<<v>>}{7.13}{\kelvin} at depth four", GA(4, "MAE_random", "min"), ".2f")
P("grid_A_r4hi", r"from \SIrange<<*>>{6.48}{<<v>>}{\kelvin} at depth four", GA(4, "MAE_random", "max"), ".2f")
P("grid_A_r10lo", r"at depth four to \SIrange<<*>>{<<v>>}{5.11}{\kelvin} at depth ten", GA(10, "MAE_random", "min"), ".2f")
P("grid_A_r10hi", r"at depth four to \SIrange<<*>>{5.05}{<<v>>}{\kelvin} at depth ten", GA(10, "MAE_random", "max"), ".2f")
P("grid_A_i4lo", r"therefore rises with depth, from \SIrange<<*>>{<<v>>}{39.6}{\percent}", GA(4, "inflacion_pct", "min"), ".1f")
P("grid_A_i4hi", r"rises with depth, from \SIrange<<*>>{28.5}{<<v>>}{\percent}", GA(4, "inflacion_pct", "max"), ".1f")
P("grid_A_i10lo", r"to \SIrange<<*>>{<<v>>}{68.7}{\percent}. Dataset~B shows", GA(10, "inflacion_pct", "min"), ".1f")
P("grid_A_i10hi", r"to \SIrange<<*>>{63.4}{<<v>>}{\percent}. Dataset~B shows", GA(10, "inflacion_pct", "max"), ".1f")
GB = lambda d, f, agg: csv(R3 + "hp_grid_ambos_v3.csv", {"dataset": "B", "es_publicada": "False", "max_depth": d}, f, agg)
P("grid_B_grp_range", r"same pattern (grouped MAE within \SI{<<v>>}{\kelvin} over the grid and", expr("a-b", a=GAll("B", "MAE_grouped", "max"), b=GAll("B", "MAE_grouped", "min")), ".1f")
P("grid_B_grp_range6", r"over the grid and \SI{<<v>>}{\kelvin} from depth six; random MAE", expr("a-b", a=GAll("B", "MAE_grouped", "max", {"max_depth": [6, 8, 10]}), b=GAll("B", "MAE_grouped", "min", {"max_depth": [6, 8, 10]})), ".1f")
P("grid_B_r4lo", r"random MAE from \SIrange<<*>>{<<v>>}{5.59}{\kelvin} to", GB(4, "MAE_random", "min"), ".2f")
P("grid_B_r4hi", r"random MAE from \SIrange<<*>>{5.05}{<<v>>}{\kelvin} to", GB(4, "MAE_random", "max"), ".2f")
P("grid_B_r10lo", r"{\kelvin} to \SIrange<<*>>{<<v>>}{3.95}{\kelvin}; inflation from", GB(10, "MAE_random", "min"), ".2f")
P("grid_B_r10hi", r"{\kelvin} to \SIrange<<*>>{3.78}{<<v>>}{\kelvin}; inflation from", GB(10, "MAE_random", "max"), ".2f")
P("grid_B_i4lo", r"inflation from \SIrange<<*>>{<<v>>}{33.2}{\percent} to", GB(4, "inflacion_pct", "min"), ".1f")
P("grid_B_i4hi", r"inflation from \SIrange<<*>>{24.7}{<<v>>}{\percent} to", GB(4, "inflacion_pct", "max"), ".1f")
P("grid_B_i10lo", r"{\percent} to \SIrange<<*>>{<<v>>}{64.4}{\percent}). Additional", GB(10, "inflacion_pct", "min"), ".1f")
P("grid_B_i10hi", r"{\percent} to \SIrange<<*>>{57.6}{<<v>>}{\percent}). Additional", GB(10, "inflacion_pct", "max"), ".1f")
P("strat_720", r"system holds \num{<<v>>} Dataset~A rows, \SI{64}{\percent} of them above", FC("A", "tam_familia_max"), ".0f")
P("strat_64", r"Dataset~A rows, \SI{<<v>>}{\percent} of them above \SI{77}", csv(R3 + "data_facts.csv", {"dataset": "A", "fact": "largest_family_pct_Tc_gt_77K"}, "value"), ".0f")
P("strat_B_lo", r"on~A and \SIrange<<*>>{<<v>>}{59.2}{\percent} on~B, against", STRAT("B", "StratifiedGroupKFold", "min"), ".1f")
P("strat_B_hi", r"on~A and \SIrange<<*>>{58.1}{<<v>>}{\percent} on~B, against", STRAT("B", "StratifiedGroupKFold", "max"), ".1f")
P("unstrat_B_lo", r"and \SIrange<<*>>{<<v>>}{62.0}{\percent} for the unstratified", STRAT("B", "GroupKFold", "min"), ".1f")
P("unstrat_B_hi", r"and \SIrange<<*>>{60.3}{<<v>>}{\percent} for the unstratified", STRAT("B", "GroupKFold", "max"), ".1f")
# ---- 3.5 levels
P("lev_L1_sysA", r"but leaves \SI{<<v>>}{\percent} (A) and \SI{86.1}{\percent} (B) of test rows with their element system", csv(R3 + "separation_levels.csv", {"dataset": "A", "level": "L1_composicion_1e-3"}, "exact_element_system", scale=100), ".1f")
P("lev_L1_sysB", r"leaves \SI{88.1}{\percent} (A) and \SI{<<v>>}{\percent} (B) of test rows with their element system", csv(R3 + "separation_levels.csv", {"dataset": "B", "level": "L1_composicion_1e-3"}, "exact_element_system", scale=100), ".1f")
P("lev_L1_dA", r"$\Delta$MAE $=\SI{+<<v>>}{\kelvin}$ \ci{+0.12}{+0.36} on~A", LEV("A", "L1_composicion_1e-3", "dMAE_vs_L0"), ".2f")
P("lev_L1_dA_lo", r"$\Delta$MAE $=\SI{+0.24}{\kelvin}$ \ci{+<<v>>}{+0.36} on~A", LEV("A", "L1_composicion_1e-3", "dMAE_vs_L0_fam_lo"), ".2f")
P("lev_L1_dA_hi", r"$\Delta$MAE $=\SI{+0.24}{\kelvin}$ \ci{+0.12}{+<<v>>} on~A", LEV("A", "L1_composicion_1e-3", "dMAE_vs_L0_fam_hi"), ".2f")
P("lev_L1_dB", r"on~A and \SI{+<<v>>}{\kelvin} \ci{+0.02}{+0.15} on~B", LEV("B", "L1_composicion_1e-3", "dMAE_vs_L0"), ".2f")
P("lev_L1_dB_lo", r"and \SI{+0.08}{\kelvin} \ci{+<<v>>}{+0.15} on~B", LEV("B", "L1_composicion_1e-3", "dMAE_vs_L0_fam_lo"), ".2f")
P("lev_L1_dB_hi", r"and \SI{+0.08}{\kelvin} \ci{+0.02}{+<<v>>} on~B", LEV("B", "L1_composicion_1e-3", "dMAE_vs_L0_fam_hi"), ".2f")
P("lev_L2_dA", r"removes all of them and costs \SI{<<v>>}{\kelvin} and \SI{2.41}", LEV("A", "L2_sistema_elemental", "dMAE_vs_L0"), ".2f")
P("lev_L2_dB", r"removes all of them and costs \SI{3.08}{\kelvin} and \SI{<<v>>}{\kelvin}", LEV("B", "L2_sistema_elemental", "dMAE_vs_L0"), ".2f")
P("lev_L3_thr", r"(L3, dopants below \num{<<v>>} molar fraction ignored)", const(0.05, "L3 host threshold (definition)"), ".2f")
P("lev_L3_sysA", r"but \SI{<<v>>}{\percent} (A) and \SI{41.8}{\percent} (B) of test rows again share", csv(R3 + "separation_levels.csv", {"dataset": "A", "level": "L3_anfitrion_0.05"}, "exact_element_system", scale=100), ".1f")
P("lev_L3_sysB", r"\SI{50.9}{\percent} (A) and \SI{<<v>>}{\percent} (B) of test rows again share", csv(R3 + "separation_levels.csv", {"dataset": "B", "level": "L3_anfitrion_0.05"}, "exact_element_system", scale=100), ".1f")
P("lev_L3_dA", r"at once and costs \SI{<<v>>}{\kelvin} and \SI{3.35}{\kelvin}", LEV("A", "L3_anfitrion_0.05", "dMAE_vs_L0"), ".2f")
P("lev_L3_dB", r"at once and costs \SI{3.72}{\kelvin} and \SI{<<v>>}{\kelvin}", LEV("B", "L3_anfitrion_0.05", "dMAE_vs_L0"), ".2f")
P("lev_L4_dA", r"most demanding in error (\SI{+<<v>>}{\kelvin} and \SI{+5.30}", LEV("A", "L4_LOCO_kmeans50", "dMAE_vs_L0"), ".2f")
P("lev_L4_dB", r"in error (\SI{+5.65}{\kelvin} and \SI{+<<v>>}{\kelvin}) and in", LEV("B", "L4_LOCO_kmeans50", "dMAE_vs_L0"), ".2f")
P("lev_L4_sysA", r"yet \SI{<<v>>}{\percent} and \SI{13.2}{\percent} of its test rows share", csv(R3 + "separation_levels.csv", {"dataset": "A", "level": "L4_LOCO_kmeans50"}, "exact_element_system", scale=100), ".1f")
P("lev_L4_sysB", r"yet \SI{12.6}{\percent} and \SI{<<v>>}{\percent} of its test rows share", csv(R3 + "separation_levels.csv", {"dataset": "B", "level": "L4_LOCO_kmeans50"}, "exact_element_system", scale=100), ".1f")
P("lev_L3s_A002", r"(A: MAE \SI{<<v>>}{\kelvin} at \num{0.02}", LEV("A", "L3_anfitrion_0.02", "MAE"), ".2f")
P("lev_L3s_002", r"(A: MAE \SI{8.47}{\kelvin} at \num{<<v>>},", const(0.02, "L3 sensitivity threshold"), ".2f")
P("lev_L3s_A010", r"at \num{0.02}, \SI{<<v>>}{\kelvin} at \num{0.10}; B:", LEV("A", "L3_anfitrion_0.10", "MAE"), ".2f")
P("lev_L3s_010", r"\SI{10.73}{\kelvin} at \num{<<v>>}; B:", const(0.10, "L3 sensitivity threshold"), ".2f")
P("lev_L3s_B002", r"at \num{0.10}; B: \SI{<<v>>}{\kelvin} and \SI{7.92}", LEV("B", "L3_anfitrion_0.02", "MAE"), ".2f")
P("lev_L3s_B010", r"B: \SI{6.64}{\kelvin} and \SI{<<v>>}{\kelvin}), which is", LEV("B", "L3_anfitrion_0.10", "MAE"), ".2f")
P("cap4_seed", r"published descriptors, five folds, seed \num{<<v>>}). ``Shared", const(42, "seed of the deterministic protocol"), ".0f")
P("cap4_thr", r"host system (fraction $\ge$ \num{<<v>>}) occurs in the training", const(0.05, "L3 host threshold (definition)"), ".2f")
P("cap4_K", r"Top-$K$ is the top-\num{<<v>>} precision", const(100, "K of the selection metric"), ".0f")
# fig 3 caption
P("fig3_95", r"(points with \SI{<<v>>}{\percent} paired family-bootstrap", const(95, "confidence level"), ".0f")
P("fig3_K", r"(c)~Top-\num{<<v>>} precision of", const(100, "K of the selection metric"), ".0f")
P("fig3_nB_rand", r"\num{4344}; B: $n=\num{<<v>>}$ and \num{2432} test rows)", NN("B", "n_test_random"), ".0f")
P("fig3_nB_grp", r"B: $n=\num{2468}$ and \num{<<v>>} test rows)", NN("B", "n_test_grouped"), ".0f")
# ---- 3.6 ceiling
P("cl_icc_A", r"intraclass correlation of $\Tc$ is \num{<<v>>} (A) and \num{0.846}", DEFF("A", EST_I, "ICC"), ".3f")
P("cl_icc_B", r"is \num{0.854} (A) and \num{<<v>>} (B), and system identity", DEFF("B_v3", EST_I, "ICC"), ".3f")
P("cl_eta_A", r"accounts in-sample for \SI{<<v>>}{\percent} and \SI{88.4}", csv(R3 + "design_effect_v3.csv", {"dataset": "A", "variant": "todas las familias", "estimador": EST_I}, "eta2", scale=100), ".1f")
P("cl_eta_B", r"for \SI{87.7}{\percent} and \SI{<<v>>}{\percent} of the variance", csv(R3 + "design_effect_v3.csv", {"dataset": "B_v3", "variant": "todas las familias", "estimador": EST_I}, "eta2", scale=100), ".1f")
P("cl_mw1", r"$<<v>> + \rho(m_w - 1)$ with $m_w = \num{69.6}$", const(1, "algebraic identity"), ".0f")
P("cl_mw1b", r"$1 + \rho(m_w - <<v>>)$ with $m_w = \num{69.6}$", const(1, "algebraic identity"), ".0f")
P("cl_mw_A", r"with $m_w = \num{<<v>>}$ (A) and \num{17.8} (B)", DEFF("A", EST_I, "m_w"), ".1f")
P("cl_mw_B", r"with $m_w = \num{69.6}$ (A) and \num{<<v>>} (B)---not Searle", DEFF("B_v3", EST_I, "m_w"), ".1f")
P("cl_m0_A", r"not Searle's $m_0$ of \num{<<v>>} and \num{4.05}", DEFF("A", EST_I, "m0"), ".2f")
P("cl_m0_B", r"Searle's $m_0$ of \num{6.30} and \num{<<v>>}---so", DEFF("B_v3", EST_I, "m0"), ".2f")
P("cl_neff_A", r"effective sample size is \num{<<v>>} and \num{809}, rising", DEFF("A", EST_I, "N_eff"), ".0f")
P("cl_neff_B", r"effective sample size is \num{357} and \num{<<v>>}, rising", DEFF("B_v3", EST_I, "N_eff"), ".0f")
P("cl_neff_A_nl", r"rising to \num{<<v>>} and \num{853} when the largest", DEFF("A", EST_I, "N_eff", "sin la familia mayor (n=720)"), ".0f")
P("cl_neff_B_nl", r"rising to \num{509} and \num{<<v>>} when the largest", DEFF("B_v3", EST_I, "N_eff", "sin la familia mayor (n=131)"), ".0f")
P("cl_neff2_A", r"the effective size is \num{<<v>>} and \num{3249}, close to $k$", DEFF("A", EST_II, "N_eff"), ".0f")
P("cl_neff2_B", r"the effective size is \num{3621} and \num{<<v>>}, close to $k$", DEFF("B_v3", EST_II, "N_eff"), ".0f")
# fig 4 caption
P("fig4_eta_A", r"on \emph{all} the data (\num{<<v>>}, \num{0.884}). It is not", DEFF("A", EST_I, "eta2"), ".3f")
P("fig4_eta_B", r"on \emph{all} the data (\num{0.877}, \num{<<v>>}). It is not", DEFF("B_v3", EST_I, "eta2"), ".3f")
P("fig4_full_A", r"the published descriptors (\num{<<v>>}, \num{0.848}), element presence", ABL("A", "grouped", "R2", "R2"), ".3f")
P("fig4_full_B", r"the published descriptors (\num{0.855}, \num{<<v>>}), element presence", ABL("B", "grouped", "R2", "R2"), ".3f")
P("fig4_pres_A", r"element presence only (\num{<<v>>}, \num{0.788})", ABL("A", "grouped", "R0", "R2"), ".3f")
P("fig4_pres_B", r"element presence only (\num{0.803}, \num{<<v>>})", ABL("B", "grouped", "R0", "R2"), ".3f")
P("fig4_null_A", r"the family-mean null (\num{<<v>>}, \num{-0.003})", PFB("A", "FamilyMeanNull", "R2_grouped"), ".3f")
P("fig4_null_B", r"the family-mean null (\num{-0.007}, \num{<<v>>})", PFB("B", "FamilyMeanNull", "R2_grouped"), ".3f")
# ablation prose
SSE = lambda ds, prot, a, b, f="SSE_red": csv(R3 + "ablation_representation_diffs.csv", {"dataset": ds, "protocol": prot, "ref_a": a, "method_b": b}, f, scale=100)
P("abl_R0_A", r"Element presence alone reaches $\Rsq = \num{<<v>>}$ (A) and \num{0.850} (B) under random", ABL("A", "random", "R0", "R2"), ".3f")
P("abl_R0_B", r"reaches $\Rsq = \num{0.847}$ (A) and \num{<<v>>} (B) under random", ABL("B", "random", "R0", "R2"), ".3f")
P("abl_R1sse_A", r"Adding molar fractions removes \SI{<<v>>}{\percent} \ci{41.9}{53.6} and", SSE("A", "random", "R0", "R1"), ".1f")
P("abl_R1sse_A_lo", r"removes \SI{48.1}{\percent} \ci{<<v>>}{53.6} and", SSE("A", "random", "R0", "R1", "SSE_red_fam_lo"), ".1f")
P("abl_R1sse_A_hi", r"removes \SI{48.1}{\percent} \ci{41.9}{<<v>>} and", SSE("A", "random", "R0", "R1", "SSE_red_fam_hi"), ".1f")
P("abl_R1sse_B", r"\ci{41.9}{53.6} and \SI{<<v>>}{\percent} \ci{45.4}{57.6} of the", SSE("B", "random", "R0", "R1"), ".1f")
P("abl_R1sse_B_lo", r"and \SI{52.0}{\percent} \ci{<<v>>}{57.6} of the", SSE("B", "random", "R0", "R1", "SSE_red_fam_lo"), ".1f")
P("abl_R1sse_B_hi", r"and \SI{52.0}{\percent} \ci{45.4}{<<v>>} of the", SSE("B", "random", "R0", "R1", "SSE_red_fam_hi"), ".1f")
P("abl_R2sse_A", r"squared error, against \SI{<<v>>}{\percent} and \SI{54.5}{\percent} for the", SSE("A", "random", "R0", "R2"), ".1f")
P("abl_R2sse_B", r"against \SI{53.2}{\percent} and \SI{<<v>>}{\percent} for the", SSE("B", "random", "R0", "R2"), ".1f")
P("abl_share_A", r"stoichiometry recovers \SI{<<v>>}{\percent} (A) and \SI{95}", expr("100*a/b", a=ABLD("A", "random", "R0", "R1", "SSE_red"), b=ABLD("A", "random", "R0", "R2", "SSE_red")), ".0f")
P("abl_share_B", r"recovers \SI{90}{\percent} (A) and \SI{<<v>>}{\percent} (B) of the error", expr("100*a/b", a=ABLD("B", "random", "R0", "R1", "SSE_red"), b=ABLD("B", "random", "R0", "R2", "SSE_red")), ".0f")
P("abl_R3_A", r"descriptors plus fractions add \SI{<<v>>}{\percent} \ci{0.5}{4.0} (A)", SSE("A", "random", "R2", "R3"), ".1f")
P("abl_R3_A_lo", r"fractions add \SI{2.3}{\percent} \ci{<<v>>}{4.0} (A)", SSE("A", "random", "R2", "R3", "SSE_red_fam_lo"), ".1f")
P("abl_R3_A_hi", r"fractions add \SI{2.3}{\percent} \ci{0.5}{<<v>>} (A)", SSE("A", "random", "R2", "R3", "SSE_red_fam_hi"), ".1f")
P("abl_R3_B", r"(A) and \SI{<<v>>}{\percent} \ci{3.1}{10.0} (B) over the", SSE("B", "random", "R2", "R3"), ".1f")
P("abl_R3_B_lo", r"and \SI{6.5}{\percent} \ci{<<v>>}{10.0} (B)", SSE("B", "random", "R2", "R3", "SSE_red_fam_lo"), ".1f")
P("abl_R3_B_hi", r"and \SI{6.5}{\percent} \ci{3.1}{<<v>>} (B)", SSE("B", "random", "R2", "R3", "SSE_red_fam_hi"), ".1f")
P("abl_g_R1R2_A", r"reduce the error (\SI{+<<v>>}{\kelvin} \ci{+0.00}{+0.98} on~A", ABLD("A", "grouped", "R1", "R2", "dMAE"), ".2f")
P("abl_g_R1R2_A_lo", r"(\SI{+0.46}{\kelvin} \ci{+<<v>>}{+0.98} on~A", ABLD("A", "grouped", "R1", "R2", "dMAE_fam_lo"), ".2f")
P("abl_g_R1R2_A_hi", r"(\SI{+0.46}{\kelvin} \ci{+0.00}{+<<v>>} on~A", ABLD("A", "grouped", "R1", "R2", "dMAE_fam_hi"), ".2f")
P("abl_g_R1R2_B", r"on~A, \SI{+<<v>>}{\kelvin} \ci{-0.18}{+0.59} on~B), while", ABLD("B", "grouped", "R1", "R2", "dMAE"), ".2f")
P("abl_g_R1R2_B_lo", r"\SI{+0.16}{\kelvin} \ci{<<v>>}{+0.59} on~B)", ABLD("B", "grouped", "R1", "R2", "dMAE_fam_lo"), ".2f")
P("abl_g_R1R2_B_hi", r"\SI{+0.16}{\kelvin} \ci{-0.18}{+<<v>>} on~B)", ABLD("B", "grouped", "R1", "R2", "dMAE_fam_hi"), ".2f")
P("abl_g_R2R3_A", r"to the descriptors does (\SI{<<v>>}{\kelvin} \ci{-1.00}{-0.29} and", ABLD("A", "grouped", "R2", "R3", "dMAE"), ".2f")
P("abl_g_R2R3_A_lo", r"does (\SI{-0.60}{\kelvin} \ci{<<v>>}{-0.29} and", ABLD("A", "grouped", "R2", "R3", "dMAE_fam_lo"), ".2f")
P("abl_g_R2R3_A_hi", r"does (\SI{-0.60}{\kelvin} \ci{-1.00}{<<v>>} and", ABLD("A", "grouped", "R2", "R3", "dMAE_fam_hi"), ".2f")
P("abl_g_R2R3_B", r"\ci{-1.00}{-0.29} and \SI{<<v>>}{\kelvin} \ci{-0.86}{-0.26}). A", ABLD("B", "grouped", "R2", "R3", "dMAE"), ".2f")
P("abl_g_R2R3_B_lo", r"and \SI{-0.51}{\kelvin} \ci{<<v>>}{-0.26}). A", ABLD("B", "grouped", "R2", "R3", "dMAE_fam_lo"), ".2f")
P("abl_g_R2R3_B_hi", r"and \SI{-0.51}{\kelvin} \ci{-0.86}{<<v>>}). A", ABLD("B", "grouped", "R2", "R3", "dMAE_fam_hi"), ".2f")
P("abl_knn_r", r"splits (MAE \SI{<<v>>}{\kelvin} against \SI{5.23}{\kelvin} on~A) and collapses", ABL("A", "random", "kNN5_R1", "MAE"), ".2f")
P("abl_knn_xgb", r"(MAE \SI{5.46}{\kelvin} against \SI{<<v>>}{\kelvin} on~A) and collapses", ABL("A", "random", "R2", "MAE"), ".2f")
P("abl_knn_g", r"collapses on unseen systems (\SI{<<v>>}{\kelvin}), which is the signature", ABL("A", "grouped", "kNN5_R1", "MAE"), ".2f")
P("cap5_K", r"top-$K$ precision for $K=\num{<<v>>}$ (mean over folds", const(100, "K of the selection metric"), ".0f")
P("fig2_95", r"protocols; bars are \SI{<<v>>}{\percent} paired bootstrap intervals over whole", const(95, "confidence level"), ".0f")
# relatives
D1, D2 = "D1_test_fold_aleatorio", "D2_test_mitad_fold_agrupado"
P("rel_D1_n", r"removing all relatives leaves \num{<<v>>} training rows on~A and raises", REL("A", D1, "b_pool_sin_familiares", "n_train_mean"), ".0f")
P("rel_D1_dba", r"training rows on~A and raises the MAE by \SI{<<v>>}{\kelvin} \ci{4.22}{5.63}; removing", REL("A", D1, "diff_b_minus_a", "dMAE"), ".2f")
P("rel_D1_dba_lo", r"raises the MAE by \SI{4.90}{\kelvin} \ci{<<v>>}{5.63};", REL("A", D1, "diff_b_minus_a", "dMAE_fam_lo"), ".2f")
P("rel_D1_dba_hi", r"raises the MAE by \SI{4.90}{\kelvin} \ci{4.22}{<<v>>};", REL("A", D1, "diff_b_minus_a", "dMAE_fam_hi"), ".2f")
P("rel_D1_dca", r"the same number of rows at random raises it by \SI{<<v>>}{\kelvin} \ci{1.45}{1.71}, so", REL("A", D1, "diff_c_minus_a_media_reps", "dMAE"), ".2f")
P("rel_D1_dca_lo", r"raises it by \SI{1.58}{\kelvin} \ci{<<v>>}{1.71}, so", REL("A", D1, "diff_c_minus_a_media_reps", "dMAE_fam_lo"), ".2f")
P("rel_D1_dca_hi", r"raises it by \SI{1.58}{\kelvin} \ci{1.45}{<<v>>}, so", REL("A", D1, "diff_c_minus_a_media_reps", "dMAE_fam_hi"), ".2f")
P("rel_D1_frac_A", r"\ci{1.45}{1.71}, so \SI{<<v>>}{\percent} of the penalty is attributable to the relatives (B: \SI{62.6}", csv(R3 + "remove_relatives_control_summary.csv", {"dataset": "A", "design": D1, "condition": "diff_b_minus_a"}, "frac_gap_ba_attributable_to_relatives", scale=100), ".1f")
P("rel_D1_frac_B", r"attributable to the relatives (B: \SI{<<v>>}{\percent}). With half", csv(R3 + "remove_relatives_control_summary.csv", {"dataset": "B", "design": D1, "condition": "diff_b_minus_a"}, "frac_gap_ba_attributable_to_relatives", scale=100), ".1f")
P("rel_D2_frac_pool", r"the relatives are only \SI{<<v>>}{\percent} of the pool, removing them costs", csv(R3 + "remove_relatives_control_summary.csv", {"dataset": "A", "design": D2, "condition": "b_pool_sin_familiares"}, "frac_removed_mean", scale=100), ".1f")
P("rel_D2_dba", r"removing them costs \SI{<<v>>}{\kelvin} \ci{2.08}{3.14} and removing", REL("A", D2, "diff_b_minus_a", "dMAE"), ".2f")
P("rel_D2_dba_lo", r"removing them costs \SI{2.58}{\kelvin} \ci{<<v>>}{3.14} and", REL("A", D2, "diff_b_minus_a", "dMAE_fam_lo"), ".2f")
P("rel_D2_dba_hi", r"removing them costs \SI{2.58}{\kelvin} \ci{2.08}{<<v>>} and", REL("A", D2, "diff_b_minus_a", "dMAE_fam_hi"), ".2f")
P("rel_D2_dca", r"as many random rows costs \SI{<<v>>}{\kelvin} \ci{0.01}{0.12}: \SI{97.4}", REL("A", D2, "diff_c_minus_a_media_reps", "dMAE"), ".2f")
P("rel_D2_dca_lo", r"random rows costs \SI{0.07}{\kelvin} \ci{<<v>>}{0.12}:", REL("A", D2, "diff_c_minus_a_media_reps", "dMAE_fam_lo"), ".2f")
P("rel_D2_dca_hi", r"random rows costs \SI{0.07}{\kelvin} \ci{0.01}{<<v>>}:", REL("A", D2, "diff_c_minus_a_media_reps", "dMAE_fam_hi"), ".2f")
P("rel_D2_frac_A", r"\ci{0.01}{0.12}: \SI{<<v>>}{\percent} of the penalty is the relatives (B:", csv(R3 + "remove_relatives_control_summary.csv", {"dataset": "A", "design": D2, "condition": "diff_b_minus_a"}, "frac_gap_ba_attributable_to_relatives", scale=100), ".1f")
P("rel_D2_frac_B", r"of the penalty is the relatives (B: \SI{<<v>>}{\percent}). We report both", csv(R3 + "remove_relatives_control_summary.csv", {"dataset": "B", "design": D2, "condition": "diff_b_minus_a"}, "frac_gap_ba_attributable_to_relatives", scale=100), ".1f")
# ---- 3.7 selection
P("scr_null_r_A", r"family-mean null selects \SI{<<v>>}{\percent} positives on~A", TOPK("A", None, "random", "precision_pct"), ".1f")
P("scr_xgb_r_A", r"and the descriptor models reach \SI{<<v>>}{\percent}. Under", TOPK("A", "XGBoost", "random", "precision_pct"), ".1f")
P("scr_null_g_A", r"equals the fold prevalence by construction (\SI{<<v>>}{\percent} and \SI{10.2}", TOPK("A", None, "grouped", "precision_pct"), ".1f")
P("scr_null_g_B", r"by construction (\SI{10.3}{\percent} and \SI{<<v>>}{\percent}; the previous", TOPK("Bv3", None, "grouped", "precision_pct"), ".1f")
P("scr_v5_A", r"the previous version reported \SI{<<v>>}{\percent} and \SI{9.6}", const(11.6, "value printed in the v5 manuscript (argsort tie-break artefact; results_v3/topk_resumen_H05_H06.md)"), ".1f")
P("scr_v5_B", r"previous version reported \SI{11.6}{\percent} and \SI{<<v>>}{\percent}, which were", const(9.6, "value printed in the v5 manuscript (argsort tie-break artefact; results_v3/topk_resumen_H05_H06.md)"), ".1f")
P("scr_xgb_g_A", r"while XGBoost retains \SI{<<v>>}{\percent} \ci{77.2}{92.0} on~A", TOPK("A", "XGBoost", "grouped", "precision_pct"), ".1f")
P("scr_xgb_g_A_lo", r"retains \SI{87.0}{\percent} \ci{<<v>>}{92.0} on~A", TOPK("A", "XGBoost", "grouped", "precision_ci_fam_lo_pct"), ".1f")
P("scr_xgb_g_A_hi", r"retains \SI{87.0}{\percent} \ci{77.2}{<<v>>} on~A", TOPK("A", "XGBoost", "grouped", "precision_ci_fam_hi_pct"), ".1f")
P("scr_xgb_g_B", r"on~A and \SI{<<v>>}{\percent} \ci{75.1}{88.8} on~B, an enrichment", TOPK("Bv3", "XGBoost", "grouped", "precision_pct"), ".1f")
P("scr_xgb_g_B_lo", r"and \SI{82.4}{\percent} \ci{<<v>>}{88.8} on~B", TOPK("Bv3", "XGBoost", "grouped", "precision_ci_fam_lo_pct"), ".1f")
P("scr_xgb_g_B_hi", r"and \SI{82.4}{\percent} \ci{75.1}{<<v>>} on~B", TOPK("Bv3", "XGBoost", "grouped", "precision_ci_fam_hi_pct"), ".1f")
P("scr_enr_A", r"an enrichment of \num{<<v>>} and \num{8.6} over prevalence", TOPK("A", "XGBoost", "grouped", "enrichment"), ".1f")
P("scr_enr_B", r"an enrichment of \num{10.7} and \num{<<v>>} over prevalence", TOPK("Bv3", "XGBoost", "grouped", "enrichment"), ".1f")
P("scr_K100", r"of all positives with \num{<<v>>} picks. This is", const(100, "K of the selection metric"), ".0f")
P("scr_neg_n", r"Admitting the \num{<<v>>} measured non-superconductors of the Stanev list to the Dataset-B", NEG("XGBoost", "grouped", "n_neg_pool"), ".0f")
P("scr_neg_from", r"lowers the grouped XGBoost precision from \SI{<<v>>}{\percent} to \SI{78.2}", NEG("XGBoost", "grouped", "precision_pos_only_pct"), ".1f")
P("scr_neg_to", r"precision from \SI{82.4}{\percent} to \SI{<<v>>}{\percent} \ci{70.0}{84.8}", NEG("XGBoost", "grouped", "precision_with_neg_pct"), ".1f")
P("scr_neg_lo", r"to \SI{78.2}{\percent} \ci{<<v>>}{84.8}, with", NEG("XGBoost", "grouped", "precision_with_neg_ci_fam_lo_pct"), ".1f")
P("scr_neg_hi", r"to \SI{78.2}{\percent} \ci{70.0}{<<v>>}, with", NEG("XGBoost", "grouped", "precision_with_neg_ci_fam_hi_pct"), ".1f")
P("scr_neg_exp", r"with \num{<<v>>} \ci{3.8}{8.8} non-superconductors expected among the \num{100} picks", NEG("XGBoost", "grouped", "expected_neg_in_topK"), ".1f")
P("scr_neg_exp_lo", r"with \num{6.2} \ci{<<v>>}{8.8} non-superconductors", NEG("XGBoost", "grouped", "expected_neg_in_topK_ci_fam_lo"), ".1f")
P("scr_neg_exp_hi", r"with \num{6.2} \ci{3.8}{<<v>>} non-superconductors", NEG("XGBoost", "grouped", "expected_neg_in_topK_ci_fam_hi"), ".1f")
P("scr_neg_K", r"non-superconductors expected among the \num{<<v>>} picks. The selected", const(100, "K of the selection metric"), ".0f")
P("scr_fam19", r"concentrated: \num{<<v>>} distinct systems among \num{100} picks on~A", TOPK("A", "XGBoost", "grouped", "n_families_topK"), ".0f")
P("scr_fam100", r"distinct systems among \num{<<v>>} picks on~A, and a budget", const(100, "K of the selection metric"), ".0f")
P("scr_budget", r"per system lowers the precision to \SI{<<v>>}{\percent}.", TOPK("A", "XGBoost", "grouped", "budget_precision_pct"), ".1f")
P("scr_pres_A", r"Element presence alone reaches \SI{<<v>>}{\percent} and \SI{68.0}{\percent} on held-out", TOPK("A", "presence", "grouped", "precision_pct"), ".1f")
P("scr_pres_B", r"alone reaches \SI{84.3}{\percent} and \SI{<<v>>}{\percent} on held-out", TOPK("Bv3", "presence", "grouped", "precision_pct"), ".1f")
P("scr_diff_A", r"XGBoost minus presence is \num{+<<v>>} percentage points \ci{-0.8}{+14.2} on~A", TOPKD("A", "XGBoost - presence (grouped)", "d_precision_pp"), ".1f")
P("scr_diff_A_lo", r"is \num{+2.7} percentage points \ci{<<v>>}{+14.2} on~A", TOPKD("A", "XGBoost - presence (grouped)", "d_precision_ci_lo_pp"), ".1f")
P("scr_diff_A_hi", r"is \num{+2.7} percentage points \ci{-0.8}{+<<v>>} on~A", TOPKD("A", "XGBoost - presence (grouped)", "d_precision_ci_hi_pp"), ".1f")
P("scr_diff_B", r"distinguishable from zero---and \num{+<<v>>} points \ci{+6.7}{+24.5} on~B", TOPKD("Bv3", "XGBoost - presence (grouped)", "d_precision_pp"), ".1f")
P("scr_diff_B_lo", r"and \num{+14.4} points \ci{+<<v>>}{+24.5} on~B", TOPKD("Bv3", "XGBoost - presence (grouped)", "d_precision_ci_lo_pp"), ".1f")
P("scr_diff_B_hi", r"and \num{+14.4} points \ci{+6.7}{+<<v>>} on~B", TOPKD("Bv3", "XGBoost - presence (grouped)", "d_precision_ci_hi_pp"), ".1f")
P("scr_v5_77", r"The previous version's ``$<<v>> \pm 1.0$ points'' for~A", const(7.7, "value printed in the v5 manuscript (across-seed dispersion, withdrawn)"), ".1f")
P("scr_v5_10", r"``$7.7 \pm <<v>>$ points'' for~A", const(1.0, "value printed in the v5 manuscript (across-seed dispersion, withdrawn)"), ".1f")
P("cap6_K", r"Selection metrics at $K=\num{<<v>>}$, per cent unless stated (five folds, seed \num{42};", const(100, "K of the selection metric"), ".0f")
P("cap6_seed", r"per cent unless stated (five folds, seed \num{<<v>>};", const(42, "seed of the deterministic protocol"), ".0f")
P("cap6_neg", r"precision when the \num{<<v>>} measured non-superconductors are admitted to the pool", NEG("XGBoost", "grouped", "n_neg_pool"), ".0f")
# ---- 3.8 temporal
TVX = lambda cond, est, met, field="valor", T=2010: TV(T, cond, "XGBoost", est, met, field)
TCX = lambda proto, met, field, T=2010, est="todo": TC(T, proto, "XGBoost", met, field, est)
FN = lambda T: csv(R3 + "temporal_validation_comparacion.csv", {"corte": T, "escenario": "principal", "protocolo": "L0_aleatorio", "modelo": "XGBoost", "estimacion": "todo", "metrica": "MAE"}, "frac_familia_nueva_test", scale=100)
P("tmp_new_lo", r"between \SI{<<v>>}{\percent} and \SI{84}{\percent} of the compositions reported after each cut", expr("min(a,b,c,d)", a=FN(1995), b=FN(2005), c=FN(2010), d=FN(2015)), ".0f")
P("tmp_new_hi", r"between \SI{72}{\percent} and \SI{<<v>>}{\percent} of the compositions reported after each cut", expr("max(a,b,c,d)", a=FN(1995), b=FN(2005), c=FN(2010), d=FN(2015)), ".0f")
P("tmp_ntrain", r"principal cut $T$ = 2010 (\num{<<v>>} training compositions, \num{2328}", TMETA("meta.principal_T2010.n_train"), ".0f")
P("tmp_ntest", r"training compositions, \num{<<v>>} later ones, \SI{76}", TMETA("meta.principal_T2010.n_test"), ".0f")
P("tmp_new76", r"later ones, \SI{<<v>>}{\percent} in new systems) XGBoost reaches", FN(2010), ".0f")
P("tmp_mae", r"XGBoost reaches MAE \SI{<<v>>}{\kelvin} \ci{6.67}{8.27} and $\Rsq = \num{0.33}$", TVX("temporal_real", "todo", "MAE"), ".2f")
P("tmp_mae_lo", r"reaches MAE \SI{7.43}{\kelvin} \ci{<<v>>}{8.27}", TVX("temporal_real", "todo", "MAE", "ci_lo_familias"), ".2f")
P("tmp_mae_hi", r"reaches MAE \SI{7.43}{\kelvin} \ci{6.67}{<<v>>}", TVX("temporal_real", "todo", "MAE", "ci_hi_familias"), ".2f")
P("tmp_r2", r"\ci{6.67}{8.27} and $\Rsq = \num{<<v>>}$ \ci{0.18}{0.47}; on the later", TVX("temporal_real", "todo", "R2"), ".2f")
P("tmp_r2_lo", r"and $\Rsq = \num{0.33}$ \ci{<<v>>}{0.47}; on the later", TVX("temporal_real", "todo", "R2", "ci_lo_familias"), ".2f")
P("tmp_r2_hi", r"and $\Rsq = \num{0.33}$ \ci{0.18}{<<v>>}; on the later", TVX("temporal_real", "todo", "R2", "ci_hi_familias"), ".2f")
P("tmp_new_r2", r"systems $\Rsq$ is \num{<<v>>} \ci{0.01}{0.34} (MAE \SI{8.47}", TVX("temporal_real", "familia_nueva", "R2"), ".2f")
P("tmp_new_r2_lo", r"systems $\Rsq$ is \num{0.19} \ci{<<v>>}{0.34}", TVX("temporal_real", "familia_nueva", "R2", "ci_lo_familias"), ".2f")
P("tmp_new_r2_hi", r"systems $\Rsq$ is \num{0.19} \ci{0.01}{<<v>>}", TVX("temporal_real", "familia_nueva", "R2", "ci_hi_familias"), ".2f")
P("tmp_new_mae", r"\ci{0.01}{0.34} (MAE \SI{<<v>>}{\kelvin}), on those from systems already", TVX("temporal_real", "familia_nueva", "MAE"), ".2f")
P("tmp_seen_r2", r"seen it is \num{<<v>>} \ci{0.71}{0.89} (MAE \SI{4.07}", TVX("temporal_real", "familia_vista", "R2"), ".2f")
P("tmp_seen_r2_lo", r"seen it is \num{0.83} \ci{<<v>>}{0.89}", TVX("temporal_real", "familia_vista", "R2", "ci_lo_familias"), ".2f")
P("tmp_seen_r2_hi", r"seen it is \num{0.83} \ci{0.71}{<<v>>}", TVX("temporal_real", "familia_vista", "R2", "ci_hi_familias"), ".2f")
P("tmp_seen_mae", r"\ci{0.71}{0.89} (MAE \SI{<<v>>}{\kelvin}), and on later re-measurements", TVX("temporal_real", "familia_vista", "MAE"), ".2f")
P("tmp_comp_mae", r"the model recovers MAE \SI{<<v>>}{\kelvin} and $\Rsq = \num{0.80}$. Random", TVX("temporal_real", "composicion_vista_reportes", "MAE"), ".2f")
P("tmp_comp_r2", r"recovers MAE \SI{2.39}{\kelvin} and $\Rsq = \num{<<v>>}$. Random", TVX("temporal_real", "composicion_vista_reportes", "R2"), ".2f")
P("tmp_L0_mae", r"pre-2010 data estimates MAE \SI{<<v>>}{\kelvin} and $\Rsq = \num{0.91}$: it understates", TCX("L0_aleatorio", "MAE", "estimacion_retrospectiva"), ".2f")
P("tmp_L0_r2", r"estimates MAE \SI{4.34}{\kelvin} and $\Rsq = \num{<<v>>}$: it understates", TCX("L0_aleatorio", "R2", "estimacion_retrospectiva"), ".2f")
P("tmp_L0_dmae", r"understates the realised error by \SI{<<v>>}{\kelvin} \ci{2.25}{4.03}", expr("-a", a=TCX("L0_aleatorio", "MAE", "error_abs")), ".2f")
P("tmp_L0_dmae_lo", r"realised error by \SI{3.10}{\kelvin} \ci{<<v>>}{4.03}", expr("-a", a=TCX("L0_aleatorio", "MAE", "ci_hi_abs")), ".2f")
P("tmp_L0_dmae_hi", r"realised error by \SI{3.10}{\kelvin} \ci{2.25}{<<v>>}", expr("-a", a=TCX("L0_aleatorio", "MAE", "ci_lo_abs")), ".2f")
P("tmp_L0_dr2", r"and overstates $\Rsq$ by \num{<<v>>} \ci{0.44}{0.73}. The", TCX("L0_aleatorio", "R2", "error_abs"), ".2f")
P("tmp_L0_dr2_lo", r"overstates $\Rsq$ by \num{0.58} \ci{<<v>>}{0.73}", TCX("L0_aleatorio", "R2", "ci_lo_abs"), ".2f")
P("tmp_L0_dr2_hi", r"overstates $\Rsq$ by \num{0.58} \ci{0.44}{<<v>>}", TCX("L0_aleatorio", "R2", "ci_hi_abs"), ".2f")
P("tmp_dr2_1995", r"at all four cuts (\num{+<<v>>}, \num{+0.60}, \num{+0.58} and \num{+0.42}", TCX("L0_aleatorio", "R2", "error_abs", 1995), ".2f")
P("tmp_dr2_2005", r"four cuts (\num{+0.28}, \num{+<<v>>}, \num{+0.58} and", TCX("L0_aleatorio", "R2", "error_abs", 2005), ".2f")
P("tmp_dr2_2010", r"(\num{+0.28}, \num{+0.60}, \num{+<<v>>} and \num{+0.42}", TCX("L0_aleatorio", "R2", "error_abs", 2010), ".2f")
P("tmp_dr2_2015", r"\num{+0.60}, \num{+0.58} and \num{+<<v>>} at 1995", TCX("L0_aleatorio", "R2", "error_abs", 2015), ".2f")
P("tmp_L2_dmae", r"at 2005 and 2010 ($\Delta$MAE $=\SI{<<v>>}{\kelvin}$ \ci{-1.83}{+0.43} and", TCX("L2_sistema", "MAE", "error_abs"), ".2f")
P("tmp_L2_dmae_lo", r"($\Delta$MAE $=\SI{-0.67}{\kelvin}$ \ci{<<v>>}{+0.43} and", TCX("L2_sistema", "MAE", "ci_lo_abs"), ".2f")
P("tmp_L2_dmae_hi", r"($\Delta$MAE $=\SI{-0.67}{\kelvin}$ \ci{-1.83}{+<<v>>} and", TCX("L2_sistema", "MAE", "ci_hi_abs"), ".2f")
P("tmp_L3_dmae", r"\ci{-1.83}{+0.43} and \SI{+<<v>>}{\kelvin} \ci{-1.00}{+1.12} at 2010)", TCX("L3_anfitrion", "MAE", "error_abs"), ".2f")
P("tmp_L3_dmae_lo", r"and \SI{+0.10}{\kelvin} \ci{<<v>>}{+1.12} at 2010)", TCX("L3_anfitrion", "MAE", "ci_lo_abs"), ".2f")
P("tmp_L3_dmae_hi", r"and \SI{+0.10}{\kelvin} \ci{-1.00}{+<<v>>} at 2010)", TCX("L3_anfitrion", "MAE", "ci_hi_abs"), ".2f")
P("tmp_L2_dr2", r"only marginally (\num{+<<v>>} and \num{+0.47} at 2010). Leave-clusters-out", TCX("L2_sistema", "R2", "error_abs"), ".2f")
P("tmp_L3_dr2", r"only marginally (\num{+0.50} and \num{+<<v>>} at 2010). Leave-clusters-out", TCX("L3_anfitrion", "R2", "error_abs"), ".2f")
P("tmp_L4_1995", r"realised value at 1995 ($\Delta\Rsq = \num{<<v>>}$ \ci{-0.12}{+0.02}) and 2015", TCX("L4_loco_cluster", "R2", "error_abs", 1995), ".2f")
P("tmp_L4_1995_lo", r"at 1995 ($\Delta\Rsq = \num{-0.05}$ \ci{<<v>>}{+0.02})", TCX("L4_loco_cluster", "R2", "ci_lo_abs", 1995), ".2f")
P("tmp_L4_1995_hi", r"at 1995 ($\Delta\Rsq = \num{-0.05}$ \ci{-0.12}{+<<v>>})", TCX("L4_loco_cluster", "R2", "ci_hi_abs", 1995), ".2f")
P("tmp_L4_2015", r"\ci{-0.12}{+0.02}) and 2015 (\num{+<<v>>} \ci{-0.16}{+0.29}), but", TCX("L4_loco_cluster", "R2", "error_abs", 2015), ".2f")
P("tmp_L4_2015_lo", r"and 2015 (\num{+0.11} \ci{<<v>>}{+0.29}), but", TCX("L4_loco_cluster", "R2", "ci_lo_abs", 2015), ".2f")
P("tmp_L4_2015_hi", r"and 2015 (\num{+0.11} \ci{-0.16}{+<<v>>}), but", TCX("L4_loco_cluster", "R2", "ci_hi_abs", 2015), ".2f")
P("tmp_L4_dmae", r"pessimistic in MAE after 2005 (\SI{+<<v>>}{\kelvin} \ci{2.55}{5.17} at 2010)", TCX("L4_loco_cluster", "MAE", "error_abs"), ".2f")
P("tmp_L4_dmae_lo", r"after 2005 (\SI{+3.92}{\kelvin} \ci{<<v>>}{5.17} at 2010)", TCX("L4_loco_cluster", "MAE", "ci_lo_abs"), ".2f")
P("tmp_L4_dmae_hi", r"after 2005 (\SI{+3.92}{\kelvin} \ci{2.55}{<<v>>} at 2010)", TCX("L4_loco_cluster", "MAE", "ci_hi_abs"), ".2f")
P("tmp_L4_dr2_lo", r"and still overstates $\Rsq$ by \numrange{<<v>>}{0.30} at 2005--2010", expr("min(a,b)", a=TCX("L4_loco_cluster", "R2", "error_abs", 2005), b=TCX("L4_loco_cluster", "R2", "error_abs", 2010)), ".2f")
P("tmp_L4_dr2_hi", r"still overstates $\Rsq$ by \numrange{0.28}{<<v>>} at 2005--2010", expr("max(a,b)", a=TCX("L4_loco_cluster", "R2", "error_abs", 2005), b=TCX("L4_loco_cluster", "R2", "error_abs", 2010)), ".2f")
MADR2 = lambda p: expr("(abs(a)+abs(b)+abs(c)+abs(d))/4", **{k: TCX(p, "R2", "error_abs", T) for k, T in zip("abcd", (1995, 2005, 2010, 2015))})
MADMAE = lambda p: expr("(abs(a)+abs(b)+abs(c)+abs(d))/4", **{k: TCX(p, "MAE", "error_abs", T) for k, T in zip("abcd", (1995, 2005, 2010, 2015))})
P("tmp_mad_L0", r"strength of the grouping (L0 \num{<<v>>}, L2 \num{0.39}", MADR2("L0_aleatorio"), ".2f")
P("tmp_mad_L2", r"(L0 \num{0.47}, L2 \num{<<v>>}, L3 \num{0.35}", MADR2("L2_sistema"), ".2f")
P("tmp_mad_L3", r"L2 \num{0.39}, L3 \num{<<v>>}, L4 \num{0.19})", MADR2("L3_anfitrion"), ".2f")
P("tmp_mad_L4", r"L3 \num{0.35}, L4 \num{<<v>>}), while in MAE", MADR2("L4_loco_cluster"), ".2f")
P("tmp_madm_L3", r"the host protocol is closest (L3 \SI{<<v>>}{\kelvin}; L0 \SI{3.8}", MADMAE("L3_anfitrion"), ".1f")
P("tmp_madm_L0", r"(L3 \SI{2.2}{\kelvin}; L0 \SI{<<v>>}{\kelvin}; L4 \SI{3.5}", MADMAE("L0_aleatorio"), ".1f")
P("tmp_madm_L4", r"L0 \SI{3.8}{\kelvin}; L4 \SI{<<v>>}{\kelvin}). Re-weighting", MADMAE("L4_loco_cluster"), ".1f")
P("tmp_rew_r2", r"realised value either ($\Rsq$ \num{<<v>>} versus \num{0.33} at 2010)", TCX("L0_aleatorio", "R2", "estimacion_retrospectiva", 2010, "reponderado_mezcla_test"), ".2f")
P("tmp_rew_real", r"($\Rsq$ \num{0.81} versus \num{<<v>>} at 2010)", TVX("temporal_real", "todo", "R2"), ".2f")
PRV = lambda T: csv(R3 + "temporal_validation.csv", {"corte": T, "escenario": "principal", "condicion": "temporal_real", "modelo": "XGBoost", "estrato": "todo", "metrica": "prevalencia_alto_tc"}, "valor", scale=100)
TP = lambda T: TV(T, "temporal_real", "XGBoost", "todo", "top100")
P("tmp_prev_1995", r"later compositions collapses from \SI{<<v>>}{\percent} at 1995 to", PRV(1995), ".1f")
P("tmp_prev_lo", r"at 1995 to \SIrange{<<v>>}{1.8}{\percent} at 2005--2015", expr("min(a,b,c)", a=PRV(2005), b=PRV(2010), c=PRV(2015)), ".1f")
P("tmp_prev_hi", r"at 1995 to \SIrange{0.9}{<<v>>}{\percent} at 2005--2015", expr("max(a,b,c)", a=PRV(2005), b=PRV(2010), c=PRV(2015)), ".1f")
P("tmp_top_K", r"so the realised top-\num{<<v>>} precision falls from", const(100, "K of the selection metric"), ".0f")
P("tmp_top_1995", r"precision falls from \num{<<v>>} to \numrange{0.08}{0.47} while", TP(1995), ".2f")
P("tmp_top_lo", r"falls from \num{0.90} to \numrange{<<v>>}{0.47} while", expr("min(a,b,c)", a=TP(2005), b=TP(2010), c=TP(2015)), ".2f")
P("tmp_top_hi", r"falls from \num{0.90} to \numrange{0.08}{<<v>>} while", expr("max(a,b,c)", a=TP(2005), b=TP(2010), c=TP(2015)), ".2f")
ETP = lambda p, T: TC(T, p, "XGBoost", "top100", "estimacion_retrospectiva")
ALLP = {f"{k}{i}": ETP(p, T) for i, T in enumerate((1995, 2005, 2010, 2015)) for k, p in zip("abcde", ("L0_aleatorio", "L1_composicion", "L2_sistema", "L3_anfitrion", "L4_loco_cluster"))}
P("tmp_est_lo", r"while every protocol estimates \numrange{<<v>>}{0.99}; as a lift", expr("min(" + ",".join(ALLP) + ")", **ALLP), ".2f")
P("tmp_est_hi", r"while every protocol estimates \numrange{0.36}{<<v>>}; as a lift", expr("max(" + ",".join(ALLP) + ")", **ALLP), ".2f")
LIFT = lambda T: TV(T, "temporal_real", "XGBoost", "todo", "lift_top100")
P("tmp_lift_lo", r"ranking is strong (\numrange{<<v>>}{26} times at 2005--2010)", expr("min(a,b)", a=LIFT(2005), b=LIFT(2010)), ".0f")
P("tmp_lift_hi", r"ranking is strong (\numrange{18}{<<v>>} times at 2005--2010)", expr("max(a,b)", a=LIFT(2005), b=LIFT(2010)), ".0f")
SPR = lambda T: TV(T, "temporal_real", "XGBoost", "todo", "Spearman")
SPE = lambda T: TC(T, "L0_aleatorio", "XGBoost", "Spearman", "estimacion_retrospectiva")
P("tmp_sp_est", r"degrades more gently (\num{<<v>>} retrospective versus \numrange{0.57}{0.78} realised)", expr("min(a,b,c,d)", a=SPE(1995), b=SPE(2005), c=SPE(2010), d=SPE(2015)), ".2f")
P("tmp_sp_lo", r"(\num{0.95} retrospective versus \numrange{<<v>>}{0.78} realised)", expr("min(a,b,c,d)", a=SPR(1995), b=SPR(2005), c=SPR(2010), d=SPR(2015)), ".2f")
P("tmp_sp_hi", r"(\num{0.95} retrospective versus \numrange{0.57}{<<v>>} realised)", expr("max(a,b,c,d)", a=SPR(1995), b=SPR(2005), c=SPR(2010), d=SPR(2015)), ".2f")
P("tmp_526", r"removing the \num{<<v>>} training compositions whose year came from a fallback", TMETA("meta.sensibilidad_sin_anio_ambiguo.n_train_excluidas") if False else const(526, "results_v3/temporal_validation_hallazgos.md: training compositions dated by a fallback rule and removed in the sin_anio_ambiguo scenario"), ".0f")
P("tmp_5pct", r"(\SI{<<v>>}{\percent} of the training set, mostly", expr("100*526/a", a=TMETA("meta.principal_T2010.n_train")), ".0f")
P("tmp_bi_15", r"Bi--S--O family from about \SI{<<v>>}{\kelvin} to about \SI{60}", const(15, "results_v3/temporal_validation_hallazgos.md (approximate prediction shift of the Bi-S-O family)"), ".0f")
P("tmp_bi_60", r"from about \SI{15}{\kelvin} to about \SI{<<v>>}{\kelvin} and drives", const(60, "results_v3/temporal_validation_hallazgos.md (approximate prediction shift of the Bi-S-O family)"), ".0f")
P("tmp_sens_r2_from", r"drives the realised $\Rsq$ at 2010 from \num{<<v>>} to \num{-0.20}", TVX("temporal_real", "todo", "R2"), ".2f")
P("tmp_sens_r2_to", r"realised $\Rsq$ at 2010 from \num{0.33} to \num{<<v>>} (MAE", TV(2010, "temporal_real", "XGBoost", "todo", "R2", "valor", "sin_anio_ambiguo"), ".2f")
P("tmp_sens_mae_from", r"to \num{-0.20} (MAE \SI{<<v>>}{\kelvin} to \SI{9.1}{\kelvin})", TVX("temporal_real", "todo", "MAE"), ".1f")
P("tmp_sens_mae_to", r"(MAE \SI{7.4}{\kelvin} to \SI{<<v>>}{\kelvin}), whereas", TV(2010, "temporal_real", "XGBoost", "todo", "MAE", "valor", "sin_anio_ambiguo"), ".1f")
P("tmp_sens_03", r"move by less than \SI{<<v>>}{\kelvin} in MAE and \num{0.02} in $\Rsq$", const(0.3, "results_v3/temporal_validation_hallazgos.md bound on the L0/L4 shift under the sin_anio_ambiguo scenario"), ".1f")
P("tmp_sens_002", r"in MAE and \num{<<v>>} in $\Rsq$. Using the maximum", const(0.02, "results_v3/temporal_validation_hallazgos.md bound on the L0/L4 shift under the sin_anio_ambiguo scenario"), ".2f")
P("ans_L4_39", r"leave-clusters-out (\SI{+<<v>>}{\kelvin} against \SI{-3.1}{\kelvin} for random CV)", TCX("L4_loco_cluster", "MAE", "error_abs"), ".1f")
P("ans_L0_31", r"(\SI{+3.9}{\kelvin} against \SI{<<v>>}{\kelvin} for random CV)", TCX("L0_aleatorio", "MAE", "error_abs"), ".1f")
NR2 = lambda T: TV(T, "temporal_real", "XGBoost", "familia_nueva", "R2")
P("ans_new_lo", r"The realised $\Rsq$ on new element systems (\numrange{<<v>>}{0.5}) is the number", expr("min(a,b,c,d)", a=NR2(1995), b=NR2(2005), c=NR2(2010), d=NR2(2015)), ".1f")
P("ans_new_hi", r"on new element systems (\numrange{0.1}{<<v>>}) is the number", expr("max(a,b,c,d)", a=NR2(1995), b=NR2(2005), c=NR2(2010), d=NR2(2015)), ".1f")
P("ans_within01", r"would have predicted it to within \num{<<v>>} at more than one cut", const(0.1, "tolerance chosen to summarise table tab:temporal"), ".1f")
# fig 5 caption + table caption
P("fig5_K", r"Rows: MAE, $\Rsq$ and top-\num{<<v>>} precision. The red line", const(100, "K of the selection metric"), ".0f")
P("fig5_95", r"later compositions with its \SI{<<v>>}{\percent} family-bootstrap interval", const(95, "confidence level"), ".0f")
P("cap7_ntrain", r"principal cut $T$ = 2010 (\num{<<v>>} compositions first reported before 2010", TMETA("meta.principal_T2010.n_train"), ".0f")
P("cap7_ntest", r"before 2010, \num{<<v>>} at or after;", TMETA("meta.principal_T2010.n_test"), ".0f")
P("cap7_frac", r"at or after; \SI{<<v>>}{\percent} of the later compositions belong", FN(2010), ".0f")
P("cap7_95", r"realised; \SI{<<v>>}{\percent} intervals from a bootstrap", const(95, "confidence level"), ".0f")
P("cap7_K", r"$\ge$); $K = \num{<<v>>}$. L1 differs", const(100, "K of the selection metric"), ".0f")
P("fig1_80", r"training rows only, for a single $<<v>>/20$ hold-out per scheme ($n=\num{4253}$", const(80, "80/20 hold-out (definition)"), ".0f")
P("fig1_20", r"training rows only, for a single $80/<<v>>$ hold-out per scheme ($n=\num{4253}$", const(20, "80/20 hold-out (definition)"), ".0f")
P("fig3_80", r"for one $<<v>>/20$ hold-out per scheme (A:", const(80, "80/20 hold-out (definition)"), ".0f")
P("fig3_20", r"for one $80/<<v>>$ hold-out per scheme (A:", const(20, "80/20 hold-out (definition)"), ".0f")
P("ans_top100", r"agree within their intervals on top-\num{<<v>>} precision", const(100, "K of the selection metric"), ".0f")
P("lim_null0", r"family-mean null's $\Rsq \approx <<v>>$ under grouped validation", const(0, "calibration value of the grouped family-mean null (table tab:main: -0.007 / -0.003)"), ".0f")
# ---- discussion / limitations / conclusion
P("disc_new_lo", r"new element systems is $\Rsq$ of order \numrange{<<v>>}{0.5}, not \num{0.9}", expr("min(a,b,c,d)", a=NR2(1995), b=NR2(2005), c=NR2(2010), d=NR2(2015)), ".1f")
P("disc_new_hi", r"of order \numrange{0.1}{<<v>>}, not \num{0.9}", expr("max(a,b,c,d)", a=NR2(1995), b=NR2(2005), c=NR2(2010), d=NR2(2015)), ".1f")
P("disc_09", r"of order \numrange{0.1}{0.5}, not \num{<<v>>}.", expr("round(a,1)", a=TCX("L0_aleatorio", "R2", "estimacion_retrospectiva")), ".1f")
P("lim_23", r"report, not of discovery, \SI{<<v>>}{\percent} of its compositions have several dated", const(23, "results_v3/temporal_validation_hallazgos.md: 2 862 of 12 543 compositions have several dated reports"), ".0f")
P("lim_570", r"fallback dating rule affects \num{<<v>>} compositions; the", const(570, "results_v3/temporal_validation_hallazgos.md: compositions dated by a fallback rule"), ".0f")
P("repro_seeds0", r"shuffled \texttt{GroupKFold} for seeds \numrange{<<v>>}{4}, and its fold", csv(R3 + "tabla2_runs_por_semilla_v3.csv", {"dataset": "B"}, "seed", "min"), ".0f")
P("repro_seeds4", r"shuffled \texttt{GroupKFold} for seeds \numrange{0}{<<v>>}, and its fold", csv(R3 + "tabla2_runs_por_semilla_v3.csv", {"dataset": "B"}, "seed", "max"), ".0f")
P("repro_K", r"again (section~<<*>>). Top-\num{<<v>>} precision is sensitive", const(100, "K of the selection metric"), ".0f")
P("repro_xgb", r"so we pin \texttt{xgboost==<<ver>>}. Every number", {"source": "requirements.txt", "regex": r"(?m)^#?\s*xgboost==(\S+)"})
P("conc_infl_lo", r"inflates the pooled MAE by \SIrange{<<v>>}{64}{\percent} and lowers $\Rsq$ by \numrange{0.074}{0.084} in two", mn(INFL, "min(v0,v1,v2,v3)"), ".0f")
P("conc_infl_hi", r"inflates the pooled MAE by \SIrange{53}{<<v>>}{\percent} and lowers", mn(INFL, "max(v0,v1,v2,v3)"), ".0f")
P("conc_dr2_lo", r"and lowers $\Rsq$ by \numrange{<<v>>}{0.084} in two curations", mn(DR2, "min(v0,v1,v2,v3)"), ".3f")
P("conc_dr2_hi", r"and lowers $\Rsq$ by \numrange{0.074}{<<v>>} in two curations", mn(DR2, "max(v0,v1,v2,v3)"), ".3f")
P("conc_macro_lo", r"per system the inflation is \SIrange{<<v>>}{29}{\percent}. Most", mn(MACRO, "min(v0,v1,v2,v3)"), ".0f")
P("conc_macro_hi", r"per system the inflation is \SIrange{22}{<<v>>}{\percent}. Most", mn(MACRO, "max(v0,v1,v2,v3)"), ".0f")
P("conc_top_A", r"held-out corpus materials remains high (\SI{<<v>>}{\percent} and \SI{82.4}", TOPK("A", "XGBoost", "grouped", "precision_pct"), ".1f")
P("conc_top_B", r"remains high (\SI{87.0}{\percent} and \SI{<<v>>}{\percent}), but", TOPK("Bv3", "XGBoost", "grouped", "precision_pct"), ".1f")
DR2L0 = {f"v{i}": TCX("L0_aleatorio", "R2", "error_abs", T) for i, T in enumerate((1995, 2005, 2010, 2015))}
P("conc_over_lo", r"random cross-validation overstates $\Rsq$ by \numrange{<<v>>}{0.60}", expr("min(v0,v1,v2,v3)", **DR2L0), ".2f")
P("conc_over_hi", r"random cross-validation overstates $\Rsq$ by \numrange{0.28}{<<v>>}", expr("max(v0,v1,v2,v3)", **DR2L0), ".2f")
P("conc_new_lo", r"realised $\Rsq$ on new element systems is \numrange{<<v>>}{0.5}. We therefore", expr("min(a,b,c,d)", a=NR2(1995), b=NR2(2005), c=NR2(2010), d=NR2(2015)), ".1f")
P("conc_new_hi", r"on new element systems is \numrange{0.1}{<<v>>}. We therefore", expr("max(a,b,c,d)", a=NR2(1995), b=NR2(2005), c=NR2(2010), d=NR2(2015)), ".1f")

E = [e for e in E if e is not None]
ids = [e["id"] for e in E]
dups = {i for i in ids if ids.count(i) > 1}
assert not dups, dups
out = {"_doc": "One entry per number of paper1_manuscript.tex (v3): location (table/row/col/part or unique prose anchor) + value source. Generated by code/v3/build_manifest_v3.py; consumed by code/v3/verify_manuscript.py. Paths are relative to the repository root.",
       "_tex": "paper1_manuscript.tex (v3)", "entries": E}
json.dump(out, open(HERE / "manuscript_numbers.json", "w"), indent=1, ensure_ascii=False)
print(len(E), "entries ->", HERE / "manuscript_numbers.json")
