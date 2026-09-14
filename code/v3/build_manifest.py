"""Generate code/v3/manuscript_numbers.json — the manifest read by code/v3/verify_manuscript.py.

The manifest is data; this builder only exists so that the ~400 entries are written once with
helper functions instead of by hand.  Every entry maps ONE number of paper1_manuscript.tex to
its semantic location (table/row/col/part, or a unique prose anchor) and to the value it must
equal (result file + key + field, an expression over such values, a hyperparameter read from
the pipeline code, a regex on requirements.txt, or a declared constant with its reason).

    python code/v3/build_manifest.py        # rewrites manuscript_numbers.json next to this file
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E: list[dict] = []
_ids: set[str] = set()


def add(id_: str, loc: dict, src: dict, fmt: str | None = None, **kw):
    assert id_ not in _ids, f"duplicate id {id_}"; _ids.add(id_)
    e = {"id": id_, **loc, **src}
    if fmt: e["fmt"] = fmt
    e.update(kw); E.append(e)


# ---------------------------------------------------------------- location helpers -----------
def cell(table, row, col, part=0, block=None, block_row=False):
    d = {"table": table, "row": row, "col": col, "part": part}
    if block: d["block"] = block
    if block_row: d["block_row"] = True
    return d


def prose(anchor, occurrences=1):
    d = {"anchor": anchor}
    if occurrences != 1: d["occurrences"] = occurrences
    return d


# ---------------------------------------------------------------- source helpers -------------
def csv(source, key, field, agg=None, scale=None):
    d = {"source": source, "key": key, "field": field}
    if agg: d["agg"] = agg
    if scale is not None: d["scale"] = scale
    return d


NP = lambda ds, rep, split, field, scale=None: csv("results_v2/nulo_presencia.csv", {"dataset": ds, "representacion": rep, "split": split}, field, scale=scale)
FEAT, PRES, NULL = "features_completas_paper", "presencia_elementos", "nulo_media_familia"
RND, GRP = "aleatorio_KFold5", "GroupKFold5_familias"
DE = lambda ds, field, scale=None: csv("results_v2/design_effect.csv", {"dataset": ds}, field, scale=scale)
T2 = lambda ds, modelo, field: csv("results_v2/tabla2_shuffle.csv", {"dataset": ds, "modelo": modelo}, field)
T2all = lambda field, agg: csv("results_v2/tabla2_shuffle.csv", {}, field, agg=agg)
NN = lambda ds, field: csv("results_v2/nn_distance.csv", {"dataset": ds}, field)
FC = lambda ds, field: csv("results_v2/family_concentration.csv", {"dataset": ds}, field)
RC = lambda ds, field: csv("results_v2/reproduction_check.csv", {"dataset": ds}, field)
BS = lambda ds, field: csv("results_v2/bootstrap_intervalos.csv", {"dataset": ds}, field)
NC = lambda ds, field: csv("results_v2/nested_cv_inflacion.csv", {"dataset": ds}, field)
HG = lambda ds, depth, field, agg: csv("results_v2/hp_grid_ambos.csv", {"dataset": ds, "es_publicada": "False", "max_depth": depth}, field, agg=agg)
HGall = lambda ds, field, agg: csv("results_v2/hp_grid_ambos.csv", {"dataset": ds, "es_publicada": "False"}, field, agg=agg)
SG = lambda ds, splitter, agg: csv("results_v2/stratified_group_check.csv", {"dataset": ds, "splitter": splitter}, "inflacion_pct", agg=agg)
TW = lambda metrica, field="valor", split=None, scale=None: csv("results_v2/twins_noise_A.csv", {"metrica": metrica, **({"split": split} if split else {})}, field, scale=scale)
FX = lambda field, agg=None: {"source": "results_v2/fix_factorize_verificacion.json", "field": field, **({"agg": agg} if agg else {})}
EV = lambda field: {"source": "results_v2/efecto_vs_ruido.json", "field": field}
V1 = lambda ds, model, split, field: csv("results/consolidated_results.csv", {"dataset": {"A": "A (Hamidieh/UCI, N=21,263)", "B": "B (Stanev/SuperCon, N=12,440)"}[ds], "model": model, "split": split}, field)
PY = lambda module, expr: {"python": {"module": module, "expr": expr}}
REQ = lambda pkg: {"source": "requirements.txt", "regex": rf"(?m)^#?\s*{pkg}==(\S+)"}
expr = lambda e, **vars_: {"expr": e, "vars": vars_}
const = lambda v, reason: {"const": v, "reason": reason}

XGA = lambda p: PY("code/tc_pipeline.py", f"make_models(42)['XGBoost'].get_params()['{p}']")
RFA = lambda p: PY("code/tc_pipeline.py", f"make_models(42)['RandomForest'].get_params()['{p}']")
XGB = lambda p: PY("code/datasetB_pipeline.py", f"make_models(42)['XGBoost'].get_params()['{p}']")
RFB = lambda p: PY("code/datasetB_pipeline.py", f"make_models(42)['RandomForest'].get_params()['{p}']")

# ======================================================================= TABLES =============
# --- tab:main --------------------------------------------------------------------------------
for ds in "AB":
    blk = f"Dataset {ds}"
    add(f"tab_main_{ds}_N", cell("tab:main", blk, 1, 0, block_row=True), DE(ds, "N"), ".0f")
    add(f"tab_main_{ds}_k", cell("tab:main", blk, 1, 1, block_row=True), DE(ds, "k"), ".0f")
    for rep, rowname in ((NULL, "Family-mean null"), (FEAT, "XGBoost")):
        for split, sp in ((RND, "Random"), (GRP, "Chemical-family")):
            for col, field, fmt in ((3, "MAE", ".2f"), (4, "RMSE", ".2f"), (5, "R2", ".3f")):
                add(f"tab_main_{ds}_{rowname.split()[0]}_{sp}_{field}", cell("tab:main", [rowname, sp], col, block=blk),
                    NP(ds, rep, split, field), fmt)
    # Random-forest rows: no v2 generator exists; the only versioned source is the v1 table
    # results/consolidated_results.csv (3 decimals).  Dataset-B v1 rows were computed on the
    # pre-filter-factorize partition (README "Changes in v2") -> see results_v3/oof_summary.csv.
    for split, sp in (("Random", "Random"), ("Chemical-family", "Chemical-family")):
        for col, field, fmt in ((3, "MAE_K", ".2f"), (4, "RMSE_K", ".2f"), (5, "R2", ".3f")):
            add(f"tab_main_{ds}_RF_{sp}_{field}", cell("tab:main", ["Random forest", sp], col, block=blk),
                V1(ds, "RandomForest", split, field), fmt,
                note="fuente = tabla v1 (results/consolidated_results.csv), sin generador v2; una corrida fresca del pipeline versionado "
                     "en el entorno fijado (code/v3/dump_oof_predictions.py -> results_v3/oof_summary.csv) difiere en la 2a decimal del MAE en las 4 filas RF")

# --- tab:robust ------------------------------------------------------------------------------
for ds in "AB":
    for modelo, rowname in (("XGBoost", "XGBoost"), ("RandomForest", "Random forest")):
        for col, field, fmt in ((3, "MAE_random_mean", ".2f"), (4, "MAE_random_std", ".2f"), (5, "MAE_grouped_mean", ".2f"),
                                (6, "MAE_grouped_std", ".2f"), (7, "inflacion_pct_mean", ".1f"), (8, "inflacion_pct_std", ".1f"),
                                (9, "dR2_mean", ".3f"), (10, "dR2_std", ".3f")):
            add(f"tab_robust_{ds}_{modelo}_{field}", cell("tab:robust", [ds, rowname], col), T2(ds, modelo, field), fmt)

# --- tab:tuning ------------------------------------------------------------------------------
for ds, c0 in (("A", 2), ("B", 5)):
    for i, field in enumerate(("MAE_random_publicada", "MAE_grouped_publicada_GKF5", "inflacion_publicada_misma_config_pct")):
        add(f"tab_tuning_{ds}_pub_{field}", cell("tab:tuning", "Published configuration", c0 + i), NC(ds, field), ".2f" if i < 2 else ".1f")
    for i, field in enumerate(("MAE_random_modal", "MAE_grouped_anidada", "inflacion_anidada_vs_random_modal_pct")):
        add(f"tab_tuning_{ds}_nested_{field}", cell("tab:tuning", "Nested grouped selection", c0 + i), NC(ds, field), ".2f" if i < 2 else ".1f")
    for depth in (4, 6, 8, 10):
        row = f"Grid, depth {depth} \\(range\\)"
        if ds == "A":
            add(f"tab_tuning_depthlabel_{depth}", cell("tab:tuning", row, 1), HG("A", depth, "max_depth", "first"), ".0f")
        for i, (field, fmt) in enumerate((("MAE_random", ".2f"), ("MAE_grouped", ".2f"), ("inflacion_pct", ".1f"))):
            add(f"tab_tuning_{ds}_d{depth}_{field}_lo", cell("tab:tuning", row, c0 + i, 0), HG(ds, depth, field, "min"), fmt)
            add(f"tab_tuning_{ds}_d{depth}_{field}_hi", cell("tab:tuning", row, c0 + i, 1), HG(ds, depth, field, "max"), fmt)

# --- tab:deff --------------------------------------------------------------------------------
for ds in "AB":
    for col, field, fmt in ((2, "N", ".0f"), (3, "k", ".0f"), (4, "ICC", ".3f"), (5, "eta2", ".3f"), (6, "deff_mbar", ".2f"),
                            (7, "deff_searle", ".2f"), (8, "N_eff_mbar", ".0f"), (9, "N_eff_searle", ".0f"), (10, "razon_contra_k_searle", ".2f")):
        add(f"tab_deff_{ds}_{field}", cell("tab:deff", f"^{ds}$", col), DE(ds, field), fmt)

# --- tab:screen ------------------------------------------------------------------------------
for rep, rowname in ((NULL, "Family-mean null"), (PRES, "Element presence"), (FEAT, "Full model")):
    for split, sp in ((RND, "Random"), (GRP, "Chemical-family")):
        for ds, c0 in (("A", 3), ("B", 5)):
            add(f"tab_screen_{ds}_{rowname.split()[0]}_{sp}_mean", cell("tab:screen", [rowname, sp], c0), NP(ds, rep, split, "top100_mean", scale=100), ".1f")
            add(f"tab_screen_{ds}_{rowname.split()[0]}_{sp}_std", cell("tab:screen", [rowname, sp], c0 + 1), NP(ds, rep, split, "top100_std", scale=100), ".1f")

# ======================================================================= PROSE ==============
P = lambda id_, anchor, src, fmt=None, occ=1, **kw: add(id_, prose(anchor, occ), src, fmt, **kw)

# --- headline numbers repeated in abstract / figures / discussion / conclusion ---------------
P("infl_range_lo", r"\SIrange{<<v>>}{61}{\percent}", T2all("inflacion_pct_mean", "min"), ".0f", occ=5)
P("infl_range_hi", r"\SIrange{52}{<<v>>}{\percent}", T2all("inflacion_pct_mean", "max"), ".0f", occ=5)
P("dR2_range_lo", r"\numrange{<<v>>}{0.081}", expr("-x", x=T2all("dR2_mean", "max")), ".3f", occ=3)
P("dR2_range_hi", r"\numrange{0.071}{<<v>>}", expr("-x", x=T2all("dR2_mean", "min")), ".3f", occ=3)
ICC_BOTH = expr("a if format(a, '.2f') == format(b, '.2f') else float('nan')", a=DE("A", "ICC"), b=DE("B", "ICC"))
WITHIN_K = expr("max((a - 1) * 100, (b - 1) * 100)", a=DE("A", "razon_contra_k_searle"), b=DE("B", "razon_contra_k_searle"))
P("abs_N_A", r"(\num{<<v>>} and \num{12440} compounds) and two featurizations", DE("A", "N"), ".0f")
P("abs_N_B", r"(\num{21263} and \num{<<v>>} compounds) and two featurizations", DE("B", "N"), ".0f")
P("abs_ICC", r"within a family is \num{<<v>>} in both datasets", ICC_BOTH, ".2f")
P("abs_deff_lo", r"design effect of \numrange{<<v>>}{5.5}", DE("B", "deff_searle"), ".1f")
P("abs_deff_hi", r"design effect of \numrange{3.6}{<<v>>}", DE("A", "deff_searle"), ".1f")
P("abs_Neff_A", r"effective sample size (\num{<<v>>} and \num{3470}) within", DE("A", "N_eff_searle"), ".0f")
P("abs_Neff_B", r"effective sample size (\num{3847} and \num{<<v>>}) within", DE("B", "N_eff_searle"), ".0f")
P("abs_within14", r"\num{3470}) within \SI{<<v>>}{\percent} of the number of", WITHIN_K, ".0f")
P("eta2_pct_A", r"\SI{<<v>>}{\percent} and \SI{88.4}{\percent} of the variance in $\Tc$", DE("A", "eta2", scale=100), ".1f", occ=2)
P("eta2_pct_B", r"\SI{87.7}{\percent} and \SI{<<v>>}{\percent} of the variance in $\Tc$", DE("B", "eta2", scale=100), ".1f", occ=2)
P("top100_A_vs_base", r"\SI{<<v>>}{\percent} and \SI{85.2}{\percent} against a \SI{10}{\percent} base rate", NP("A", FEAT, GRP, "top100_mean", 100), ".1f", occ=2)
P("top100_B_vs_base", r"\SI{87.0}{\percent} and \SI{<<v>>}{\percent} against a \SI{10}{\percent} base rate", NP("B", FEAT, GRP, "top100_mean", 100), ".1f", occ=2)
BASE10 = const(10, "base rate = 100 - 90th percentile threshold (protocol definition, section 2.6)")
P("base_rate_abs_concl", r"\SI{87.0}{\percent} and \SI{85.2}{\percent} against a \SI{<<v>>}{\percent} base rate", BASE10, ".0f", occ=2)
P("abs_null_A", r"collapses to chance (\SI{<<v>>}{\percent} and \SI{9.6}{\percent})", NP("A", NULL, GRP, "top100_mean", 100), ".1f")
P("abs_null_B", r"collapses to chance (\SI{11.6}{\percent} and \SI{<<v>>}{\percent})", NP("B", NULL, GRP, "top100_mean", 100), ".1f")
P("lit_R2_093", r"routinely report $\Rsq \approx <<v>>$", const(0.93, "literature range quoted in the abstract (no result file)"))
P("lit_MAE_5K", r"MAE $\approx \SI{<<v>>}{\kelvin}$ for superconducting", const(5, "literature range quoted in the abstract (no result file)"))
P("lit_R2_092", r"report $\Rsq \gtrsim <<v>>$ and MAE below", const(0.92, "literature range quoted in the introduction (no result file)"))
P("lit_MAE_6K", r"MAE below \SI{<<v>>}{\kelvin}, and deep", const(6, "literature range quoted in the introduction (no result file)"))

# --- 2.1 Datasets ---------------------------------------------------------------------------
P("data_N_A", r": \num{<<v>>} compounds with \num{81} statistical features", DE("A", "N"), ".0f")
P("data_nfeat_A", r"compounds with \num{<<v>>} statistical features", NP("A", FEAT, RND, "n_features"), ".0f")
P("data_dup_exact", r"It contains \num{<<v>>} rows that are exact duplicates", TW("filas_duplicadas_features_y_Tc"), ".0f")
P("data_dup_feat", r"and \num{<<v>>} rows (\SI{28.7}{\percent}) whose", TW("filas_con_features_duplicadas"), ".0f")
P("data_dup_pct", r"(\SI{<<v>>}{\percent}) whose \num{81} features", TW("pct_filas_features_duplicadas"), ".1f")
P("data_nfeat_A_2", r"whose \num{<<v>>} features coincide exactly", NP("A", FEAT, RND, "n_features"), ".0f")
P("data_stanev_rows", r"which has \num{<<v>>} entries. We drop", FX("n_rows_stanev_csv"), ".0f")
P("data_dropped_Tc0", r"We drop the \num{<<v>>} entries with $\Tc \le 0$", expr("n - f - k", n=FX("n_rows_stanev_csv"), f=FX("n_parse_failures_pymatgen"), k=FX("n_rows_Tc_gt0")), ".0f")
P("data_Tc_le0_def", r"entries with $\Tc \le <<v>>$ (non-superconducting", const(0, "definition of the Tc>0 filter"), ".0f")
P("data_parse_fail", r"and \num{<<v>>} whose formula \texttt{pymatgen} cannot parse", FX("n_parse_failures_pymatgen"), ".0f")
P("data_N_B", r"cannot parse, yielding \num{<<v>>} compounds", FX("n_rows_Tc_gt0"), ".0f")
P("data_nfeat_B", r"with the \num{<<v>>} Magpie descriptors", NP("B", FEAT, RND, "n_features"), ".0f")

# --- 2.2 Chemical families ------------------------------------------------------------------
P("fam_k_A", r"Dataset~A contains \num{<<v>>} families for \num{21263} compounds", FC("A", "k_familias"), ".0f")
P("fam_N_A", r"families for \num{<<v>>} compounds, with \SI{79.5}{\percent}", FC("A", "N"), ".0f")
P("fam_pct5_A", r"with \SI{<<v>>}{\percent} of all materials in families of five", FC("A", "pct_en_familias_ge5"), ".1f")
P("fam_top1_A", r"largest families hold \num{<<v>>}, \num{358} and \num{323}", csv("results_v2/family_concentration.csv", {"dataset": "A"}, "top3_tam;0"), ".0f")
P("fam_top2_A", r"hold \num{720}, \num{<<v>>} and \num{323} compounds", csv("results_v2/family_concentration.csv", {"dataset": "A"}, "top3_tam;1"), ".0f")
P("fam_top3_A", r"\num{358} and \num{<<v>>} compounds. Dataset~B", csv("results_v2/family_concentration.csv", {"dataset": "A"}, "top3_tam;2"), ".0f")
P("fam_k_B", r"Dataset~B contains \num{<<v>>} families for \num{12440}", FC("B", "k_familias"), ".0f")
P("fam_N_B", r"families for \num{<<v>>} compounds, with \SI{66.7}{\percent}", FC("B", "N"), ".0f")
P("fam_pct5_B", r"with \SI{<<v>>}{\percent} in families of five or more and", FC("B", "pct_en_familias_ge5"), ".1f")
P("fam_max_B", r"a largest family of \num{<<v>>}.", FC("B", "tam_familia_max"), ".0f")
P("fam_single_A", r"singletons (\num{<<v>>} and \num{1322} respectively)", FC("A", "n_singleton"), ".0f")
P("fam_single_B", r"singletons (\num{1169} and \num{<<v>>} respectively)", FC("B", "n_singleton"), ".0f")
P("fam_Tc0_mask", r"before a $\Tc><<v>>$ mask leaves gaps", const(0, "definition of the Tc>0 filter"), ".0f")

# --- 2.3 protocols, 2.4 models ---------------------------------------------------------------
P("sklearn_16", r"available since scikit-learn~<<v>>, so that", const(1.6, "library fact: GroupKFold(shuffle=) exists since scikit-learn 1.6 (release notes), no result file"))
P("hp_A_xgb_nest", r"with \num{<<v>>} estimators, maximum depth \num{8}", XGA("n_estimators"), ".0f")
P("hp_A_xgb_depth", r"estimators, maximum depth \num{<<v>>}, learning rate \num{0.07}", XGA("max_depth"), ".0f")
P("hp_A_xgb_lr", r"learning rate \num{<<v>>}, subsample \num{0.9}", XGA("learning_rate"), ".2f")
P("hp_A_xgb_sub", r"subsample \num{<<v>>}, column subsample \num{0.7}", XGA("subsample"), ".1f")
P("hp_A_xgb_col", r"column subsample \num{<<v>>}, \texttt{gamma}", XGA("colsample_bytree"), ".1f")
P("hp_A_xgb_gamma", r"\texttt{gamma} \num{<<v>>},", XGA("gamma"), ".0f")
P("hp_A_xgb_alpha", r"\texttt{reg\_alpha} \num{<<v>>};", XGA("reg_alpha"), ".2f")
P("hp_A_rf_nest", r"random forest with \num{<<v>>} trees, maximum depth \num{25}", RFA("n_estimators"), ".0f")
P("hp_A_rf_depth", r"trees, maximum depth \num{<<v>>} and \texttt{sqrt}", RFA("max_depth"), ".0f")
P("hp_B_xgb_nest", r"Dataset~B: XGBoost with \num{<<v>>} estimators", XGB("n_estimators"), ".0f")
P("hp_B_xgb_depth", r"estimators, maximum depth \num{<<v>>}, learning rate \num{0.05}", XGB("max_depth"), ".0f")
P("hp_B_xgb_lr", r"learning rate \num{<<v>>}, subsample and column subsample", XGB("learning_rate"), ".2f")
P("hp_B_xgb_sub", r"subsample and column subsample \num{<<v>>}, \texttt{hist}", expr("a if a == b else float('nan')", a=XGB("subsample"), b=XGB("colsample_bytree")), ".1f")
P("hp_B_rf_nest", r"random forest with \num{<<v>>} trees and maximum depth \num{30}", RFB("n_estimators"), ".0f")
P("hp_B_rf_depth", r"trees and maximum depth \num{<<v>>}, without feature", RFB("max_depth"), ".0f")
P("hp_rf_leaf", r"\texttt{min\_samples\_leaf} \num{<<v>>} and", expr("a if a == b else float('nan')", a=RFA("min_samples_leaf"), b=RFB("min_samples_leaf")), ".0f")
P("hp_rf_split", r"\texttt{min\_samples\_split} \num{<<v>>}. The two", expr("a if a == b else float('nan')", a=RFA("min_samples_split"), b=RFB("min_samples_split")), ".0f")
GRID_N = expr("a if a == b else float('nan')", a=HGall("A", "config", "count"), b=HGall("B", "config", "count"))
P("grid16_methods", r"cross-validation and over a \num{<<v>>}-point grid", GRID_N, ".0f")

# --- 2.5 nulls, 2.6 screening, 3.1 reproduction -------------------------------------------------
P("pres_nfeat_A", r"(\num{<<v>>} indicators for Dataset~A, \num{83} for Dataset~B)", NP("A", PRES, RND, "n_features"), ".0f")
P("pres_nfeat_B", r"(\num{86} indicators for Dataset~A, \num{<<v>>} for Dataset~B)", NP("B", PRES, RND, "n_features"), ".0f")
P("eta2_formula_1", r"$\eta^2 = <<v>> - \mathrm{SS_{within}}", const(1, "algebraic identity in equation text"), ".0f")
K100 = PY("code/tc_pipeline.py", "topk_precision.__defaults__[0]")
P("top100_K_def", r"the fraction of the \num{<<v>>} highest-predicted", K100, ".0f")
P("base_rate_def", r"so the base rate is \SI{<<v>>}{\percent} by construction", BASE10, ".0f")
THR04 = expr("100 * (a - b)", a=csv("results_v2/top100_factorial.csv", {"celda": "train_fold|npz_distribuido"}, "top100_mean"),
             b=csv("results_v2/top100_factorial.csv", {"celda": "full|npz_distribuido"}, "top100_mean"))
P("thr_04_methods", r"grouped figure by \num{<<v>>} points, but it would", THR04, ".1f")
P("thr_04_repro", r"definitions differ by \num{<<v>>} points on Dataset~B", THR04, ".1f")
TEST80 = expr("100 * (1 - float(t))", t={"source": "code/tc_pipeline.py", "regex": r"test_size=(0\.\d+)"})
TEST20 = expr("100 * float(t)", t={"source": "code/tc_pipeline.py", "regex": r"test_size=(0\.\d+)"})
P("holdout80_nn", r"under a random $<<v>>/20$ hold-out and under a grouped", TEST80, ".0f")
P("holdout20_nn", r"under a random $80/<<v>>$ hold-out and under a grouped", TEST20, ".0f")
P("holdout80_repro", r"A single random $<<v>>/20$ split gives", TEST80, ".0f")
P("holdout20_repro", r"A single random $80/<<v>>$ split gives", TEST20, ".0f")
P("holdout80_figs", r"for a single $<<v>>/20$ hold-out per scheme", TEST80, ".0f", occ=2)
P("holdout20_figs", r"for a single $80/<<v>>$ hold-out per scheme", TEST20, ".0f", occ=2)
P("repro_MAE_A", r"split gives MAE $=\SI{<<v>>}{\kelvin}$ and $\Rsq = 0.932$", RC("A", "MAE"), ".2f")
P("repro_R2_A", r"and $\Rsq = <<v>>$ on Dataset~A, and MAE", RC("A", "R2"), ".3f")
P("repro_MAE_B", r"and MAE $=\SI{<<v>>}{\kelvin}$ with $\Rsq = 0.934$", RC("B", "MAE"), ".2f")
P("repro_R2_B", r"with $\Rsq = <<v>>$ on Dataset~B. These", RC("B", "R2"), ".3f")

# --- figure 1 caption -------------------------------------------------------------------------
P("fig1_n", r"($n=\num{<<v>>}$ materials in $k=\num{3365}$ families", DE("A", "N"), ".0f")
P("fig1_k", r"in $k=\num{<<v>>}$ families; the largest", DE("A", "k"), ".0f")
P("fig1_720", r"holds \num{<<v>>} doping variants", FC("A", "tam_familia_max"), ".0f")
P("fig1_npred", r"partition ($n=\num{<<v>>}$ predictions", DE("A", "N"), ".0f")
P("fig1_MAE83", r"family; MAE $=\SI{<<v>>}{\kelvin}$, $\Rsq=0.85$", NP("A", FEAT, GRP, "MAE"), ".1f")
P("fig1_R2085", r"$\Rsq=<<v>>$, both cross-validated", NP("A", FEAT, GRP, "R2"), ".2f")
P("fig1_ntest_rand", r"per scheme ($n=\num{<<v>>}$ random and \num{4344} family-split", NN("A", "n_test_random"), ".0f")
P("fig1_ntest_grp", r"random and \num{<<v>>} family-split test materials", NN("A", "n_test_grouped"), ".0f")
P("fig1_clip3", r"distances clipped at \num{<<v>>} for display", const(3, "display clip of figure 1(d) (plotting choice in code/v2/make_figures.py)"), ".0f")
P("fig1_ratio23", r"median test material is \num{<<v>>} times farther", NN("A", "ratio"), ".0f")

# --- 3.2 gap ----------------------------------------------------------------------------------
P("gap_A_rand", r"raises XGBoost's MAE from \SI{<<v>>}{\kelvin} to \SI{8.31}{\kelvin} on Dataset~A", NP("A", FEAT, RND, "MAE"), ".2f")
P("gap_A_grp", r"from \SI{5.23}{\kelvin} to \SI{<<v>>}{\kelvin} on Dataset~A", NP("A", FEAT, GRP, "MAE"), ".2f")
P("gap_B_rand", r"and from \SI{<<v>>}{\kelvin} to \SI{6.17}{\kelvin} on Dataset~B", NP("B", FEAT, RND, "MAE"), ".2f")
P("gap_B_grp", r"from \SI{3.91}{\kelvin} to \SI{<<v>>}{\kelvin} on Dataset~B", NP("B", FEAT, GRP, "MAE"), ".2f")
dR2 = lambda ds: expr("r - g", r=NP(ds, FEAT, RND, "R2"), g=NP(ds, FEAT, GRP, "R2"))
P("gap_dR2_A", r"lowers $\Rsq$ by \num{<<v>>} and \num{0.075}", dR2("A"), ".3f")
P("gap_dR2_B", r"lowers $\Rsq$ by \num{0.073} and \num{<<v>>}", dR2("B"), ".3f")
P("nn_A_rand", r"closest training compound is \num{<<v>>} in Dataset~A", NN("A", "nn_median_random"), ".3f")
P("nn_A_grp", r"of the same size it is \num{<<v>>}, a factor of", NN("A", "nn_median_grouped"), ".3f")
P("nn_A_ratio", r"\num{1.273}, a factor of \num{<<v>>}.", NN("A", "ratio"), ".0f")
P("nn_B_rand", r"the medians are \num{<<v>>} and \num{1.584}", NN("B", "nn_median_random"), ".3f")
P("nn_B_grp", r"the medians are \num{0.140} and \num{<<v>>}", NN("B", "nn_median_grouped"), ".3f")
P("nn_B_ratio", r"\num{1.584}, a factor of \num{<<v>>}.", NN("B", "ratio"), ".0f")
P("twins_frac", r"Counting directly, \SI{<<v>> \pm 1.1}{\percent}", TW("frac_test_con_gemelo_exacto_en_train", split="random_KFold5_rs42", scale=100), ".1f")
P("twins_frac_sd", r"Counting directly, \SI{37.0 \pm <<v>>}{\percent}", TW("frac_test_con_gemelo_exacto_en_train", "sd_entre_folds", split="random_KFold5_rs42", scale=100), ".1f")
P("twins_nfeat", r"an \emph{identical} \num{<<v>>}-feature vector", NP("A", FEAT, RND, "n_features"), ".0f")
P("twins_frac_grp", r"the fraction is \SI{<<v>>}{\percent}. The random protocol", TW("frac_test_con_gemelo_exacto_en_train", split="GroupKFold5_familias", scale=100), ".1f")
P("twins_n8511", r"among the \num{<<v>>} Dataset~A rows that share", TW("n_filas_con_gemelo"), ".0f")
P("twins_floor", r"within-group mean is \SI{<<v>>}{\kelvin}, and XGBoost", TW("piso_ruido_MAE_solo_filas_con_gemelo_K"), ".2f")
P("twins_model", r"random splitting reaches \SI{<<v>>}{\kelvin} on those same rows", TW("MAE_random_modelo_solo_filas_con_gemelo_K"), ".2f")
P("twins_40pct", r"only on that \SI{<<v>>}{\percent} of the data", expr("100 * n / N", n=TW("n_filas_con_gemelo"), N=TW("n_filas_con_gemelo", "n")), ".0f")
P("boot_n", r"(\num{<<v>>} resamples of the test rows)", BS("A", "n_boot"), ".0f")
P("boot_ci95", r"gives \SI{<<v>>}{\percent} intervals for the MAE inflation", const(95, "confidence level of the percentile bootstrap (2.5/97.5 percentiles; results_v2/bootstrap_intervalos.csv stores lo/hi only)"), ".0f")
P("boot_infl_A_lo", r"\SIrange<<*>>{<<v>>}{63.0}{\percent} on Dataset~A", BS("A", "inflacion_lo"), ".1f")
P("boot_infl_A_hi", r"{55.1}{<<v>>}{\percent} on Dataset~A", BS("A", "inflacion_hi"), ".1f")
P("boot_infl_B_lo", r"\SIrange<<*>>{<<v>>}{63.9}{\percent} on Dataset~B", BS("B", "inflacion_lo"), ".1f")
P("boot_infl_B_hi", r"{51.4}{<<v>>}{\percent} on Dataset~B", BS("B", "inflacion_hi"), ".1f")
P("boot_seed", r"for XGBoost at seed \num{<<v>>};", BS("A", "seed"), ".0f")
P("boot_rand_A_lo", r"(\SIrange<<*>>{<<v>>}{5.33}{\kelvin} versus", BS("A", "MAE_random_lo"), ".2f")
P("boot_rand_A_hi", r"{5.13}{<<v>>}{\kelvin} versus", BS("A", "MAE_random_hi"), ".2f")
P("boot_grp_A_lo", r"versus \SIrange<<*>>{<<v>>}{8.44}{\kelvin} on~A", BS("A", "MAE_grouped_lo"), ".2f")
P("boot_grp_A_hi", r"{8.18}{<<v>>}{\kelvin} on~A", BS("A", "MAE_grouped_hi"), ".2f")
P("boot_rand_B_lo", r"on~A, \SIrange<<*>>{<<v>>}{4.03}{\kelvin} versus", BS("B", "MAE_random_lo"), ".2f")
P("boot_rand_B_hi", r"{3.79}{<<v>>}{\kelvin} versus", BS("B", "MAE_random_hi"), ".2f")
P("boot_grp_B_lo", r"versus \SIrange<<*>>{<<v>>}{6.33}{\kelvin} on~B", BS("B", "MAE_grouped_lo"), ".2f")
P("boot_grp_B_hi", r"{6.01}{<<v>>}{\kelvin} on~B", BS("B", "MAE_grouped_hi"), ".2f")

# --- 3.3 robustness (table caption + text) ------------------------------------------------------
SEED0 = const(0, "first seed of the 0..4 protocol (n_semillas = 5 in results_v2/tabla2_shuffle.csv)")
SEED4 = expr("n - 1", n=T2("A", "XGBoost", "n_semillas"))
P("robust_seeds_lo", r"shuffled splitters with seeds \numrange{<<v>>}{4}, so the grouped", SEED0, ".0f")
P("robust_seeds_hi", r"shuffled splitters with seeds \numrange{0}{<<v>>}, so the grouped", SEED4, ".0f")
P("robust_infl_min", r"MAE inflation ranges from \SI{<<v>>}{\percent} to \SI{60.8}{\percent} across", T2all("inflacion_pct_mean", "min"), ".1f")
P("robust_infl_max", r"from \SI{52.5}{\percent} to \SI{<<v>>}{\percent} across the four", T2all("inflacion_pct_mean", "max"), ".1f")
P("robust_dR2_hi", r"$\Delta\Rsq$ from \num{<<v>>} to \num{-0.081} (figure", T2all("dR2_mean", "max"), ".3f")
P("robust_dR2_lo", r"from \num{-0.071} to \num{<<v>>} (figure", T2all("dR2_mean", "min"), ".3f")
P("robust_std_max", r"at most \num{<<v>>} percentage points of inflation", T2all("inflacion_pct_std", "max"), ".1f")

# --- figure 2 caption ---------------------------------------------------------------------------
P("fig2_nA", r"(Dataset~A: $n=\num{<<v>>}$, $k=\num{3365}$; Dataset~B", DE("A", "N"), ".0f")
P("fig2_kA", r"$n=\num{21263}$, $k=\num{<<v>>}$; Dataset~B", DE("A", "k"), ".0f")
P("fig2_nB", r"Dataset~B: $n=\num{<<v>>}$, $k=\num{3063}$). Under", DE("B", "N"), ".0f")
P("fig2_kB", r"Dataset~B: $n=\num{12440}$, $k=\num{<<v>>}$). Under", DE("B", "k"), ".0f")
NULL_SHARE = lambda ds: expr("100 * a / b", a=NP(ds, NULL, RND, "R2"), b=NP(ds, FEAT, RND, "R2"))
P("fig2_share_A", r"the null alone reaches \SI{<<v>>}{\percent} (A) and \SI{82}{\percent} (B)", NULL_SHARE("A"), ".0f")
P("fig2_share_B", r"\SI{86}{\percent} (A) and \SI{<<v>>}{\percent} (B) of the full model", NULL_SHARE("B"), ".0f")
R2ZERO = const(0, "R^2 of a constant predictor equal to the test mean (see review H-otros: not 'by definition' in general)")
P("fig2_R2approx0", r"it collapses to $\Rsq \approx <<v>>$ ($\num{-0.007}$", R2ZERO, ".0f")
P("fig2_nullR2_A", r"($\num{<<v>>}$ and $\num{-0.003}$, drawn as a baseline", NP("A", NULL, GRP, "R2"), ".3f")
P("fig2_nullR2_B", r"($\num{-0.007}$ and $\num{<<v>>}$, drawn as a baseline", NP("B", NULL, GRP, "R2"), ".3f")
P("fig2_null_rand_A", r"as well as the model (\SI{<<v>>}{\percent} versus \SI{98.4}{\percent} on A)", NP("A", NULL, RND, "top100_mean", 100), ".1f")
P("fig2_full_rand_A", r"(\SI{95.8}{\percent} versus \SI{<<v>>}{\percent} on A)", NP("A", FEAT, RND, "top100_mean", 100), ".1f")
P("fig2_null_grp_A", r"it falls to \SI{<<v>>}{\percent} (A) and \SI{9.6}{\percent} (B), at or near", NP("A", NULL, GRP, "top100_mean", 100), ".1f")
P("fig2_null_grp_B", r"\SI{11.6}{\percent} (A) and \SI{<<v>>}{\percent} (B), at or near", NP("B", NULL, GRP, "top100_mean", 100), ".1f")
P("fig2_base10", r"at or near the \SI{<<v>>}{\percent} base rate, while XGBoost", BASE10, ".0f")
P("fig2_full_A", r"base rate, while XGBoost retains \SI{<<v>>}{\percent} and \SI{85.2}{\percent}.", NP("A", FEAT, GRP, "top100_mean", 100), ".1f")
P("fig2_full_B", r"while XGBoost retains \SI{87.0}{\percent} and \SI{<<v>>}{\percent}.", NP("B", FEAT, GRP, "top100_mean", 100), ".1f")

# --- 3.4 tuning (text + table caption) ----------------------------------------------------------
P("grid16_text1", r"folds, a \num{<<v>>}-point XGBoost grid", GRID_N, ".0f")
P("grid_d4", r"(maximum depth $\{<<v>>,6,8,10\}$", HGall("A", "max_depth", "min"), ".0f")
P("grid_d6", r"(maximum depth $\{4,<<v>>,8,10\}$", HG("A", 6, "max_depth", "first"), ".0f")
P("grid_d8", r"(maximum depth $\{4,6,<<v>>,10\}$", HG("A", 8, "max_depth", "first"), ".0f")
P("grid_d10", r"(maximum depth $\{4,6,8,<<v>>\}$", HGall("A", "max_depth", "max"), ".0f")
P("grid_lr_lo", r"learning rate $\{<<v>>,0.1\}$", HGall("A", "learning_rate", "min"), ".2f")
P("grid_lr_hi", r"learning rate $\{0.05,<<v>>\}$", HGall("A", "learning_rate", "max"), ".1f")
P("grid_mcw_lo", r"minimum child weight $\{<<v>>,10\}$", HGall("A", "min_child_weight", "min"), ".0f")
P("grid_mcw_hi", r"minimum child weight $\{1,<<v>>\}$", HGall("A", "min_child_weight", "max"), ".0f")
P("grid_nest300", r"$\{1,10\}$; \num{<<v>>} estimators)", HGall("A", "n_estimators", "first"), ".0f")
DEPTHS_WON = lambda agg: expr(f"{agg}(int(m) for s in (a, b) for m in re.findall(r'd(\\d+)', s))", a=NC("A", "configs_ganadoras_por_fold"), b=NC("B", "configs_ganadoras_por_fold"))
P("nested_depth_lo", r"(depth \numrange{<<v>>}{10} in every outer fold", DEPTHS_WON("min"), ".0f")
P("nested_depth_hi", r"(depth \numrange{8}{<<v>>} in every outer fold", DEPTHS_WON("max"), ".0f")
P("nested_MAE_A", r"returns a grouped MAE of \SI{<<v>>}{\kelvin} (A) and \SI{6.23}{\kelvin} (B)", NC("A", "MAE_grouped_anidada"), ".2f")
P("nested_MAE_B", r"\SI{8.37}{\kelvin} (A) and \SI{<<v>>}{\kelvin} (B), \SI{0.06}", NC("B", "MAE_grouped_anidada"), ".2f")
P("nested_worse", r"(B), \SI{<<v>>}{\kelvin} \emph{worse} than the published", expr("max(a - pa, b - pb)", a=NC("A", "MAE_grouped_anidada"), pa=NC("A", "MAE_grouped_publicada_GKF5"), b=NC("B", "MAE_grouped_anidada"), pb=NC("B", "MAE_grouped_publicada_GKF5")), ".2f")
P("nested_infl_A", r"gives an inflation of \SI{<<v>>}{\percent} on~A and \SI{53.1}{\percent} on~B", NC("A", "inflacion_anidada_vs_random_modal_pct"), ".1f")
P("nested_infl_B", r"\SI{63.9}{\percent} on~A and \SI{<<v>>}{\percent} on~B. What the grid", NC("B", "inflacion_anidada_vs_random_modal_pct"), ".1f")
P("grid16_text2", r"Across all \num{<<v>>} configurations the grouped MAE", GRID_N, ".0f")
GRANGE = lambda ds, depth_min=None: expr("hi - lo", hi=(HGall(ds, "MAE_grouped", "max") if depth_min is None else csv("results_v2/hp_grid_ambos.csv", {"dataset": ds, "es_publicada": "False", "max_depth": depth_min}, "MAE_grouped", agg="max")),
                                       lo=(HGall(ds, "MAE_grouped", "min") if depth_min is None else csv("results_v2/hp_grid_ambos.csv", {"dataset": ds, "es_publicada": "False", "max_depth": depth_min}, "MAE_grouped", agg="min")))
P("grid_grp_range_A", r"grouped MAE moves within \SI{<<v>>}{\kelvin} on A and \SI{0.9}{\kelvin} on B", GRANGE("A"), ".1f")
P("grid_grp_range_B", r"\SI{0.9}{\kelvin} on A and \SI{<<v>>}{\kelvin} on B, and within", GRANGE("B"), ".1f")
P("grid_grp_range_d6plus", r"and within \SI{<<v>>}{\kelvin} once depth is at least \num{6}",
  expr("max(ahi - alo, bhi - blo)", ahi=csv("results_v2/hp_grid_ambos.csv", {"dataset": "A", "es_publicada": "False", "max_depth": [6, 8, 10]}, "MAE_grouped", agg="max"),
       alo=csv("results_v2/hp_grid_ambos.csv", {"dataset": "A", "es_publicada": "False", "max_depth": [6, 8, 10]}, "MAE_grouped", agg="min"),
       bhi=csv("results_v2/hp_grid_ambos.csv", {"dataset": "B", "es_publicada": "False", "max_depth": [6, 8, 10]}, "MAE_grouped", agg="max"),
       blo=csv("results_v2/hp_grid_ambos.csv", {"dataset": "B", "es_publicada": "False", "max_depth": [6, 8, 10]}, "MAE_grouped", agg="min")), ".1f")
P("grid_depth6_thr", r"once depth is at least \num{<<v>>}; the random-split", HG("A", 6, "max_depth", "first"), ".0f")
P("grid_rand_A_d4_lo", r"on~A from \SIrange<<*>>{<<v>>}{7.1}{\kelvin} at depth \num{4} to", HG("A", 4, "MAE_random", "min"), ".1f")
P("grid_rand_A_d4_hi", r"{6.5}{<<v>>}{\kelvin} at depth \num{4} to", HG("A", 4, "MAE_random", "max"), ".1f")
P("grid_rand_A_d4_lbl", r"{6.5}{7.1}{\kelvin} at depth \num{<<v>>} to", HG("A", 4, "max_depth", "first"), ".0f")
P("grid_rand_A_d10_lo", r"to \SIrange<<*>>{<<v>>}{5.11}{\kelvin} at depth \num{10}, on~B", HG("A", 10, "MAE_random", "min"), ".2f")
P("grid_rand_A_d10_hi", r"{5.05}{<<v>>}{\kelvin} at depth \num{10}, on~B", HG("A", 10, "MAE_random", "max"), ".2f")
P("grid_rand_A_d10_lbl", r"{5.05}{5.11}{\kelvin} at depth \num{<<v>>}, on~B from", HG("A", 10, "max_depth", "first"), ".0f")
P("grid_rand_B_d4_lo", r"on~B from \SIrange<<*>>{<<v>>}{5.7}{\kelvin} to", HG("B", 4, "MAE_random", "min"), ".1f")
P("grid_rand_B_d4_hi", r"{5.1}{<<v>>}{\kelvin} to \SIrange", HG("B", 4, "MAE_random", "max"), ".1f")
P("grid_rand_B_d10_lo", r"to \SIrange<<*>>{<<v>>}{4.07}{\kelvin}. The inflation", HG("B", 10, "MAE_random", "min"), ".2f")
P("grid_rand_B_d10_hi", r"{3.87}{<<v>>}{\kelvin}. The inflation", HG("B", 10, "MAE_random", "max"), ".2f")
P("grid_infl_A_d4_lo", r"on~A from \SIrange<<*>>{<<v>>}{39.6}{\percent} at depth \num{4} to", HG("A", 4, "inflacion_pct", "min"), ".1f")
P("grid_infl_A_d4_hi", r"{28.5}{<<v>>}{\percent} at depth \num{4} to", HG("A", 4, "inflacion_pct", "max"), ".1f")
P("grid_infl_A_d4_lbl", r"{28.5}{39.6}{\percent} at depth \num{<<v>>} to", HG("A", 4, "max_depth", "first"), ".0f")
P("grid_infl_A_d10_lo", r"to \SIrange<<*>>{<<v>>}{68.7}{\percent} at depth \num{10}, on~B", HG("A", 10, "inflacion_pct", "min"), ".1f")
P("grid_infl_A_d10_hi", r"{63.4}{<<v>>}{\percent} at depth \num{10}, on~B from", HG("A", 10, "inflacion_pct", "max"), ".1f")
P("grid_infl_A_d10_lbl", r"{63.4}{68.7}{\percent} at depth \num{<<v>>}, on~B from", HG("A", 10, "max_depth", "first"), ".0f")
P("grid_infl_B_d4_lo", r"on~B from \SIrange<<*>>{<<v>>}{32.5}{\percent} to", HG("B", 4, "inflacion_pct", "min"), ".1f")
P("grid_infl_B_d4_hi", r"{22.6}{<<v>>}{\percent} to \SIrange", HG("B", 4, "inflacion_pct", "max"), ".1f")
P("grid_infl_B_d10_lo", r"to \SIrange<<*>>{<<v>>}{60.7}{\percent}. Additional", HG("B", 10, "inflacion_pct", "min"), ".1f")
P("grid_infl_B_d10_hi", r"{52.4}{<<v>>}{\percent}. Additional", HG("B", 10, "inflacion_pct", "max"), ".1f")
P("infl_approx58", r"$\sim$\SI{<<v>>}{\percent} quoted elsewhere", expr("(a + b) / 2", a=NC("A", "inflacion_publicada_misma_config_pct"), b=NC("B", "inflacion_publicada_misma_config_pct")), ".0f")
P("competitive_depth8", r"competitive region (depth $\geq <<v>>$) it lies between", HG("A", 8, "max_depth", "first"), ".0f")
D8 = lambda ds, agg: csv("results_v2/hp_grid_ambos.csv", {"dataset": ds, "es_publicada": "False", "max_depth": [8, 10]}, "inflacion_pct", agg=agg)
P("competitive_infl_lo", r"it lies between \SI{<<v>>}{\percent} and \SI{68.7}{\percent}, and it is not", expr("min(a, b)", a=D8("A", "min"), b=D8("B", "min")), ".1f")
P("competitive_infl_hi", r"between \SI{46.5}{\percent} and \SI{<<v>>}{\percent}, and it is not", expr("max(a, b)", a=D8("A", "max"), b=D8("B", "max")), ".1f")
P("tuning_cap_grid16a", r"\emph{Grid}: the same \num{<<v>>} configurations each scored", GRID_N, ".0f")
P("tuning_cap_grid16b", r"ranges are over the \num{<<v>>} points. Inflation", GRID_N, ".0f")
P("bacuoy_720", r"family holds \num{<<v>>} Dataset~A compounds, \SI{64}", FC("A", "tam_familia_max"), ".0f")
P("bacuoy_pct77", r"Dataset~A compounds, \SI{<<v>>}{\percent} of them above \SI{77}{\kelvin}", PY("code/v3/data_facts.py", "largest_family_pct_above('A', 77.0)"), ".0f")
P("bacuoy_77K", r"of them above \SI{<<v>>}{\kelvin}, and as an indivisible", const(77, "chosen high-Tc threshold (liquid-nitrogen temperature), same cut as frac_Tc_gt77 in results_v2/stratified_group_check.csv"), ".0f")
P("strat_A_lo", r"leaves the inflation at \SIrange<<*>>{<<v>>}{61.9}{\percent} on~A", SG("A", "StratifiedGroupKFold", "min"), ".1f")
P("strat_A_hi", r"{57.2}{<<v>>}{\percent} on~A and", SG("A", "StratifiedGroupKFold", "max"), ".1f")
P("strat_B_lo", r"on~A and \SIrange<<*>>{<<v>>}{63.3}{\percent} on~B, against", SG("B", "StratifiedGroupKFold", "min"), ".1f")
P("strat_B_hi", r"{57.7}{<<v>>}{\percent} on~B, against", SG("B", "StratifiedGroupKFold", "max"), ".1f")
P("unstrat_A_lo", r"against \SIrange<<*>>{<<v>>}{57.2}{\percent} and", SG("A", "GroupKFold", "min"), ".1f")
P("unstrat_A_hi", r"{54.4}{<<v>>}{\percent} and \SIrange", SG("A", "GroupKFold", "max"), ".1f")
P("unstrat_B_lo", r"and \SIrange<<*>>{<<v>>}{61.3}{\percent} for the unstratified", SG("B", "GroupKFold", "min"), ".1f")
P("unstrat_B_hi", r"{60.3}{<<v>>}{\percent} for the unstratified", SG("B", "GroupKFold", "max"), ".1f")

# --- 3.5 ceiling -----------------------------------------------------------------------------------
P("ceil_ICC", r"intraclass correlation is \num{<<v>>} in both datasets: knowing", ICC_BOTH, ".2f")
P("ceil_N_A", r"Dataset~A's \num{<<v>>} compounds carry the statistical information", DE("A", "N"), ".0f")
P("ceil_Neff_A", r"statistical information of \num{<<v>>} independent observations against", DE("A", "N_eff_searle"), ".0f")
P("ceil_k_A", r"independent observations against \num{<<v>>} families, a ratio of \num{1.14}", DE("A", "k"), ".0f")
P("ceil_ratio_A", r"a ratio of \num{<<v>>}; Dataset~B's", DE("A", "razon_contra_k_searle"), ".2f")
P("ceil_N_B", r"Dataset~B's \num{<<v>>} carry \num{3470} against", DE("B", "N"), ".0f")
P("ceil_Neff_B", r"carry \num{<<v>>} against \num{3063} families", DE("B", "N_eff_searle"), ".0f")
P("ceil_k_B", r"against \num{<<v>>} families, a ratio of \num{1.13}", DE("B", "k"), ".0f")
P("ceil_ratio_B", r"a ratio of \num{<<v>>}. This is the expected limit", DE("B", "razon_contra_k_searle"), ".2f")
P("ceil_ICC_to_1", r"as $\ICC \to <<v>>$, $\mathrm{deff} \to m$", const(1, "algebraic limit in the text"), ".0f")
P("ceil_ICC_085", r"At $\ICC = <<v>>$ both datasets sit within", ICC_BOTH, ".2f")
P("ceil_within14", r"both datasets sit within \SI{<<v>>}{\percent} of it", WITHIN_K, ".0f")
P("ceil_mbar_vs_m0", r"changes $\Neff$ by less than \SI{<<v>>}{\percent} in both datasets",
  expr("max(100 * abs(a1 - a2) / a1, 100 * abs(b1 - b2) / b1)", a1=DE("A", "N_eff_searle"), a2=DE("A", "N_eff_mbar"), b1=DE("B", "N_eff_searle"), b2=DE("B", "N_eff_mbar")), ".1f")
P("ceil_eta2_A", r"accounts for $\eta^2 = \num{<<v>>}$ of the variance in $\Tc$ on Dataset~A", DE("A", "eta2"), ".3f")
P("ceil_eta2_B", r"on Dataset~A and \num{<<v>>} on Dataset~B. This is not a score", DE("B", "eta2"), ".3f")
P("pres_R2_rand_A", r"are present reach $\Rsq = <<v>>$ and \num{0.846} under a random split", NP("A", PRES, RND, "R2"), ".3f")
P("pres_R2_rand_B", r"$\Rsq = 0.847$ and \num{<<v>>} under a random split, against", NP("B", PRES, RND, "R2"), ".3f")
P("full_R2_rand_0928", r"under a random split, against \num{<<v>>} for the full featurization---about", NP("A", FEAT, RND, "R2"), ".3f")
P("pres_share_91", r"featurization---about \SI{<<v>>}{\percent} of the headline", expr("min(100 * a / fa, 100 * b / fb)", a=NP("A", PRES, RND, "R2"), fa=NP("A", FEAT, RND, "R2"), b=NP("B", PRES, RND, "R2"), fb=NP("B", FEAT, RND, "R2")), ".0f")
DADV = lambda ds, split: expr("f - p", f=NP(ds, FEAT, split, "R2"), p=NP(ds, PRES, split, "R2"))
P("adv_A_rand", r"in $\Rsq$ it narrows, from \num{<<v>>} to \num{0.052} on Dataset~A", DADV("A", RND), ".3f")
P("adv_A_grp", r"from \num{0.081} to \num{<<v>>} on Dataset~A (\num{0.803}", DADV("A", GRP), ".3f")
P("pres_R2_grp_A", r"on Dataset~A (\num{<<v>>} versus \num{0.855} grouped)", NP("A", PRES, GRP, "R2"), ".3f")
P("full_R2_grp_A", r"(\num{0.803} versus \num{<<v>>} grouped) and from", NP("A", FEAT, GRP, "R2"), ".3f")
P("adv_B_rand", r"grouped) and from \num{<<v>>} to \num{0.074} on Dataset~B", DADV("B", RND), ".3f")
P("adv_B_grp", r"from \num{0.082} to \num{<<v>>} on Dataset~B (\num{0.779}", DADV("B", GRP), ".3f")
P("pres_R2_grp_B", r"on Dataset~B (\num{<<v>>} versus \num{0.853}). Under both", NP("B", PRES, GRP, "R2"), ".3f")
P("full_R2_grp_B", r"(\num{0.779} versus \num{<<v>>}). Under both", NP("B", FEAT, GRP, "R2"), ".3f")

# --- figure 3 caption -----------------------------------------------------------------------------
P("fig3_nA_rand", r"per scheme (A: $n=\num{<<v>>}$ and \num{4344}; B:", NN("A", "n_test_random"), ".0f")
P("fig3_nA_grp", r"(A: $n=\num{4253}$ and \num{<<v>>}; B:", NN("A", "n_test_grouped"), ".0f")
P("fig3_nB_rand", r"B: $n=\num{<<v>>}$ and \num{2477} test materials)", NN("B", "n_test_random"), ".0f")
P("fig3_nB_grp", r"B: $n=\num{2488}$ and \num{<<v>>} test materials)", NN("B", "n_test_grouped"), ".0f")

# --- figure 4 caption -----------------------------------------------------------------------------
P("fig4_eta2_A", r"on \emph{all} the data (\num{<<v>>}, \num{0.884}). It is not", DE("A", "eta2"), ".3f")
P("fig4_eta2_B", r"on \emph{all} the data (\num{0.877}, \num{<<v>>}). It is not", DE("B", "eta2"), ".3f")
P("fig4_full_A", r"the paper's features (\num{<<v>>}, \num{0.853}), a null", NP("A", FEAT, GRP, "R2"), ".3f")
P("fig4_full_B", r"the paper's features (\num{0.855}, \num{<<v>>}), a null", NP("B", FEAT, GRP, "R2"), ".3f")
P("fig4_pres_A", r"element-presence indicators (\num{<<v>>}, \num{0.779}) and the family-mean", NP("A", PRES, GRP, "R2"), ".3f")
P("fig4_pres_B", r"element-presence indicators (\num{0.803}, \num{<<v>>}) and the family-mean", NP("B", PRES, GRP, "R2"), ".3f")
P("fig4_null_A", r"family-mean null ($\num{<<v>>}$, $\num{-0.003}$). The brackets", NP("A", NULL, GRP, "R2"), ".3f")
P("fig4_null_B", r"family-mean null ($\num{-0.007}$, $\num{<<v>>}$). The brackets", NP("B", NULL, GRP, "R2"), ".3f")
P("fig4_adv_A", r"which is \num{<<v>>} on Dataset~A and \num{0.074} on Dataset~B. (b)", DADV("A", GRP), ".3f")
P("fig4_adv_B", r"which is \num{0.052} on Dataset~A and \num{<<v>>} on Dataset~B. (b)", DADV("B", GRP), ".3f")
P("fig4_deff_formula_1", r"$N$ divided by $<<v>> + (m_0-1)\ICC$", const(1, "deff formula, equation (2)"), ".0f")
P("fig4_deff_formula_m1", r"$1 + (m_0<<v>>)\ICC$ with Searle's", const(-1, "deff formula, equation (2)"), ".0f")
P("fig4_ICC", r"With $\ICC=\num{<<v>>}$ in both datasets the design effect", ICC_BOTH, ".2f")
P("fig4_deff_A", r"the design effect is \num{<<v>>} (A) and \num{3.6} (B)", DE("A", "deff_searle"), ".1f")
P("fig4_deff_B", r"design effect is \num{5.5} (A) and \num{<<v>>} (B), so", DE("B", "deff_searle"), ".1f")
P("fig4_N_A", r"(B), so \num{<<v>>} materials carry about as much", DE("A", "N"), ".0f")
P("fig4_Neff_A", r"independent information as \num{<<v>>}, close to the \num{3365} families", DE("A", "N_eff_searle"), ".0f")
P("fig4_k_A", r"close to the \num{<<v>>} families.", DE("A", "k"), ".0f")

# --- 3.6 screening (table caption + text) ----------------------------------------------------------
P("screen_cap_base10", r"so the base rate is \SI{<<v>>}{\percent}. Values are means", BASE10, ".0f")
P("screen_full_A_text", r"the full model retains \SI{<<v>>}{\percent} precision on Dataset~A", NP("A", FEAT, GRP, "top100_mean", 100), ".1f")
P("screen_full_B_text", r"precision on Dataset~A and \SI{<<v>>}{\percent} on Dataset~B against a", NP("B", FEAT, GRP, "top100_mean", 100), ".1f")
P("screen_base10_text", r"on Dataset~B against a \SI{<<v>>}{\percent} base rate. Screening", BASE10, ".0f")
P("screen_K100_text", r"synthesises the top-ranked \num{<<v>>} candidates", K100, ".0f")
P("screen_null_rand_A", r"under a random split (\SI{<<v>>}{\percent} and \SI{80.2}{\percent}), collapses", NP("A", NULL, RND, "top100_mean", 100), ".1f")
P("screen_null_rand_B", r"(\SI{95.8}{\percent} and \SI{<<v>>}{\percent}), collapses to", NP("B", NULL, RND, "top100_mean", 100), ".1f")
P("screen_null_grp_A", r"collapses to \SI{<<v>>}{\percent} and \SI{9.6}{\percent} under grouped", NP("A", NULL, GRP, "top100_mean", 100), ".1f")
P("screen_null_grp_B", r"collapses to \SI{11.6}{\percent} and \SI{<<v>>}{\percent} under grouped", NP("B", NULL, GRP, "top100_mean", 100), ".1f")
P("screen_pres_grp_A", r"ranking power (\SI{<<v>>}{\percent} and \SI{66.8}{\percent})", NP("A", PRES, GRP, "top100_mean", 100), ".1f")
P("screen_pres_grp_B", r"ranking power (\SI{84.2}{\percent} and \SI{<<v>>}{\percent})", NP("B", PRES, GRP, "top100_mean", 100), ".1f")
P("seed_gap_A", r"discarding them costs $<<v>> \pm 1.0$ points on Dataset~A", expr("abs(x)", x=EV("A.paired.mean_paired_diff")), ".1f")
P("seed_gap_A_sd", r"costs $7.7 \pm <<v>>$ points on Dataset~A", EV("A.paired.sd_paired_diff"), ".1f")
P("seed_gap_A_ratio", r"on Dataset~A ($<<v>>\times$ the across-seed dispersion)", EV("A.effect_over_noise"), ".1f")
P("seed_gap_B", r"dispersion) and $<<v>> \pm 2.7$ points on Dataset~B", expr("abs(x)", x=EV("B.paired.mean_paired_diff")), ".1f")
P("seed_gap_B_sd", r"and $14.4 \pm <<v>>$ points on Dataset~B", EV("B.paired.sd_paired_diff"), ".1f")
P("seed_gap_B_ratio", r"points on Dataset~B ($<<v>>\times$), with the full model", EV("B.effect_over_noise"), ".1f")
P("seed_gap_BA", r"a difference of \num{<<v>>} points at $3.3\times$", EV("cross_B_vs_A.effect"), ".1f")
P("seed_gap_BA_ratio", r"points at $<<v>>\times$ the across-seed dispersion with disjoint", EV("cross_B_vs_A.effect_over_noise"), ".1f")
P("conc_pct5_A_disc", r"of the two (\SI{<<v>>}{\percent} versus \SI{66.7}{\percent} of compounds", FC("A", "pct_en_familias_ge5"), ".1f")
P("conc_pct5_B_disc", r"(\SI{79.5}{\percent} versus \SI{<<v>>}{\percent} of compounds", FC("B", "pct_en_familias_ge5"), ".1f")

# --- 4 discussion, limitations, reproducibility -------------------------------------------------------
P("disc_ICC", r"MAE inflation, the $\ICC$ of \num{<<v>>}, the effective", ICC_BOTH, ".2f")
P("disc_within14", r"that lands within \SI{<<v>>}{\percent} of the family count", WITHIN_K, ".0f")
P("lim_R2approx0", r"collapse to $\Rsq \approx <<v>>$ under grouped validation is a tautology", R2ZERO, ".0f")
P("lim_R2eq0", r"a constant predictor has $\Rsq = <<v>>$ by definition", R2ZERO, ".0f")
P("lim_null_A_080", r"the same lookup reaches \num{<<v>>} and \num{0.76} without", NP("A", NULL, RND, "R2"), ".2f")
P("lim_null_B_076", r"reaches \num{0.80} and \num{<<v>>} without using", NP("B", NULL, RND, "R2"), ".2f")
P("main_cap_null_A", r"the same lookup reaches $\Rsq = <<v>>$ and $0.76$.", NP("A", NULL, RND, "R2"), ".2f")
P("main_cap_null_B", r"reaches $\Rsq = 0.80$ and $<<v>>$.", NP("B", NULL, RND, "R2"), ".2f")
P("lim_R2_0928", r"the random-split $\Rsq$ of \num{<<v>>} exceeds it", NP("A", FEAT, RND, "R2"), ".3f")
P("repro_seeds_lo", r"with shuffling for seeds \numrange{<<v>>}{4}, and its fold", SEED0, ".0f")
P("repro_seeds_hi", r"with shuffling for seeds \numrange{0}{<<v>>}, and its fold", SEED4, ".0f")
P("repro_rows_changing", r"changes the fold of \num{<<v>>} of the \num{12440} Dataset~B rows", FX("rows_changing_fold_pre_vs_npz"), ".0f")
P("repro_N_B", r"of the \num{<<v>>} Dataset~B rows (\numrange{1721}", FX("n_rows_Tc_gt0"), ".0f")
P("repro_perfold_lo", r"rows (\numrange{<<v>>}{1897} per fold)", FX("rows_changing_fold_pre_vs_npz_per_fold", "min"), ".0f")
P("repro_perfold_hi", r"rows (\numrange{1721}{<<v>>} per fold)", FX("rows_changing_fold_pre_vs_npz_per_fold", "max"), ".0f")
P("repro_Tc_le0", r"before removing the $\Tc \le <<v>>$ rows", const(0, "definition of the Tc>0 filter"), ".0f")
P("ver_python", r"produced with Python~<<ver>>, numpy", REQ("python"))
P("ver_numpy", r"numpy~<<ver>>, scikit-learn", REQ("numpy"))
P("ver_sklearn", r"scikit-learn~<<ver>>, xgboost", REQ("scikit-learn"))
P("ver_xgboost", r"xgboost~<<ver>> and matminer", REQ("xgboost"))
P("ver_matminer", r"and matminer~<<ver>>.", REQ("matminer"))

# --- 5 conclusion --------------------------------------------------------------------------------------
P("concl_ICC", r"intraclass correlation of \num{<<v>>}, these datasets carry", ICC_BOTH, ".2f")
P("concl_Neff_A", r"chemical families---\num{<<v>>} and \num{3470} effective observations", DE("A", "N_eff_searle"), ".0f")
P("concl_Neff_B", r"---\num{3847} and \num{<<v>>} effective observations", DE("B", "N_eff_searle"), ".0f")
P("concl_N_A", r"effective observations for \num{<<v>>} and \num{12440} compounds", DE("A", "N"), ".0f")
P("concl_N_B", r"observations for \num{21263} and \num{<<v>>} compounds---and", DE("B", "N"), ".0f")
P("concl_pres_A", r"reaches $\Rsq = \num{<<v>>}$ and \num{0.846} under the conventional protocol", NP("A", PRES, RND, "R2"), ".3f")
P("concl_pres_B", r"$\Rsq = \num{0.847}$ and \num{<<v>>} under the conventional protocol", NP("B", PRES, RND, "R2"), ".3f")
P("concl_full_0928", r"the conventional protocol, against \num{<<v>>} for the full featurization. The practical", NP("A", FEAT, RND, "R2"), ".3f")
P("concl_MAE_83", r"for Dataset~A, MAE $\approx \SI{<<v>>}{\kelvin}$ rather than", NP("A", FEAT, GRP, "MAE"), ".1f")
P("concl_MAE_52", r"rather than \SI{<<v>>}{\kelvin}. Obtaining it costs", NP("A", FEAT, RND, "MAE"), ".1f")

manifest = {
    "_doc": "One entry per number of paper1_manuscript.tex: location (table/row/col/part or unique prose anchor) + value source. "
            "Generated by code/v3/build_manifest.py; consumed by code/v3/verify_manuscript.py. Paths are relative to the repository root.",
    "_tex": "paper1_manuscript.tex (v5, commit 12ca34b)",
    "entries": E,
}
out = HERE / "manuscript_numbers.json"
out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
print(f"{len(E)} entries -> {out}")
