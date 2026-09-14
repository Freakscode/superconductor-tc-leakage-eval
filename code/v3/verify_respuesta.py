"""Comprueba las cifras de REVISION_RESPUESTA.md contra los CSV de results_v3.

Uso: python verify_respuesta.py   (desde cualquier directorio)
Sale con 1 si alguna cifra del documento no coincide con su fuente.
"""
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "results_v3", HERE.parent.parent / "results_v3"):
    if (cand / "design_effect_v3.csv").exists():
        RV3 = cand
        break
else:
    sys.exit("no encuentro results_v3/design_effect_v3.csv; ejecuta desde el repo o desde results_v3/")
ok, bad = [], []


def chk(label, doc, got, tol=6e-3):
    if got is None:
        bad.append(f"{label}: fuente no encontrada (doc={doc})")
    elif abs(float(doc) - float(got)) <= tol:
        ok.append(label)
    else:
        bad.append(f"{label}: doc={doc} fuente={float(got):.4f}")


def one(df, **kw):
    m = pd.Series(True, index=df.index)
    for c, v in kw.items():
        col = df[c].astype(str)
        m &= col.str.contains(v, case=False, regex=True) if isinstance(v, str) else (df[c] == v)
    return df[m].iloc[0] if m.any() else None


# ---- H01 efecto de diseño ----
d = pd.read_csv(RV3 / "design_effect_v3.csv")
base = d[d.variant.str.startswith("todas")]
for ds, mw, ne_row, ne_mean, ne_nolarge, ne_sqrt in (
    ("A", 69.633, 357, 3621, 509, 2444),
    ("B_v3", 17.841, 809, 3249, 853, 2521),
):
    r = one(base, dataset=ds, estimador="media ponderada por filas")
    chk(f"H01 {ds} m_w", mw, r.m_w, 1e-2)
    chk(f"H01 {ds} N_eff filas", ne_row, round(r.N_eff), 0.51)
    chk(f"H01 {ds} N_eff medias", ne_mean, round(one(base, dataset=ds, estimador="media de medias").N_eff), 0.51)
    nl = d[d.variant.str.startswith("sin la familia")]
    chk(f"H01 {ds} N_eff sin mayor", ne_nolarge, round(one(nl, dataset=ds, estimador="media ponderada por filas").N_eff), 0.51)
    chk(f"H01 {ds} N_eff 1/sqrt(n)", ne_sqrt, round(one(base, dataset=ds, estimador="1/sqrt").N_eff), 0.51)
chk("H01 A ICC", 0.8541, base[base.dataset == "A"].ICC.iloc[0], 5e-5)
chk("H01 B ICC", 0.8463, base[base.dataset == "B_v3"].ICC.iloc[0], 5e-5)
chk("H01 A m0", 6.300, base[base.dataset == "A"].m0.iloc[0], 1e-3)
chk("H01 A m_bar", 6.319, base[base.dataset == "A"].m_bar.iloc[0], 1e-3)
chk("H01 formula v5 A", 3847, round(one(base, dataset="A", estimador="formula del manuscrito").N_eff), 0.51)
chk("H01 formula v5 B", 3470, round(one(base, dataset="B_v2", estimador="formula del manuscrito").N_eff), 0.51)

# ---- H07 bootstrap pareado por familias ----
p = pd.read_csv(RV3 / "paired_family_bootstrap.csv")
pf = p[(p.model == "XGBoost") & (p.scheme.str.contains("family", case=False))]
for ds, pt, lo, hi, macro, mlo, mhi, dmae, dlo, dhi, dr2 in (
    ("A", 58.9, 50.4, 68.6, 27.0, 23.8, 30.5, 3.08, 2.58, 3.65, 0.074),
    ("B", 63.7, 54.6, 74.3, 29.4, 25.4, 33.8, 2.41, 2.01, 2.88, 0.084),
):
    q = pf[pf.dataset == ds]
    r = one(q, statistic="^inflacion_pct$")
    chk(f"H07 {ds} inflación", pt, r.point, 0.05)
    chk(f"H07 {ds} IC inf", lo, r.lo, 0.05)
    chk(f"H07 {ds} IC sup", hi, r.hi, 0.05)
    m = one(q, statistic="^inflacion_macro_pct$")
    chk(f"H07 {ds} macro", macro, m.point, 0.05)
    chk(f"H07 {ds} macro IC inf", mlo, m.lo, 0.05)
    chk(f"H07 {ds} macro IC sup", mhi, m.hi, 0.05)
    dm = one(q, statistic="^dMAE_K$")
    chk(f"H07 {ds} dMAE", dmae, dm.point, 0.01)
    chk(f"H07 {ds} dMAE IC inf", dlo, dm.lo, 0.01)
    chk(f"H07 {ds} dMAE IC sup", dhi, dm.hi, 0.01)
    chk(f"H07 {ds} dR2", dr2, one(q, statistic="^dR2$").point, 1e-3)
    chk(f"H07 {ds} MAE macro random", {"A": 5.07, "B": 3.50}[ds], one(q, statistic="^MAE_macro_random$").point, 6e-3)
    chk(f"H07 {ds} MAE macro grouped", {"A": 6.43, "B": 4.53}[ds], one(q, statistic="^MAE_macro_grouped$").point, 6e-3)

# ---- H08 Tabla 3 ----
t = pd.read_csv(RV3 / "tabla3_v3.csv")
for ds, nested, pub, modal, mae_r, mae_g in (
    ("A", 65.6, 58.9, 63.4, 5.055, 8.370),
    ("B", 64.4, 63.7, 57.6, 3.776, 6.209),
):
    q = t[t.dataset == ds]
    nq = one(q, bloque="^b_nested$")
    chk(f"H08 {ds} anidado", nested, nq.inflacion_pct, 0.05)
    chk(f"H08 {ds} anidado MAE random", mae_r, nq.MAE_random, 1e-3)
    chk(f"H08 {ds} anidado MAE grouped", mae_g, nq.MAE_grouped, 1e-3)
    chk(f"H08 {ds} publicada", pub, one(q, comparacion="^publicada").inflacion_pct, 0.05)
    chk(f"H08 {ds} modal", modal, one(q, comparacion="^modal").inflacion_pct, 0.05)
chk("H08 A mixto v2 (63.9)", 63.9, one(t[t.dataset == "A"], comparacion="MIXTO.*modal").inflacion_pct, 0.05)

# ---- H10 ablación ----
a = pd.read_csv(RV3 / "ablation_representation.csv")
g = a[a.protocol == "grouped"]
for ds, mae_r2, mae_r1 in (("A", 8.313, 7.854), ("B", 6.185, 6.027)):
    chk(f"H10 {ds} R2 MAE agrupado", mae_r2, one(g[g.dataset == ds], method="^R2$").MAE, 0.01)
    chk(f"H10 {ds} R1 MAE agrupado", mae_r1, one(g[g.dataset == ds], method="^R1$").MAE, 0.01)
dif = pd.read_csv(RV3 / "ablation_representation_diffs.csv")
sse_col = [c for c in dif.columns if "sse" in c.lower() and ("red" in c.lower() or "pct" in c.lower())][0]
dmae_col = [c for c in dif.columns if c.lower().startswith("dmae") and "lo" not in c and "hi" not in c][0]
for ds, sse01, sse12, sse02, dmae_g in (("A", 48.1, 9.8, 53.2, 0.459), ("B", 52.0, 5.2, 54.5, 0.158)):
    dr = dif[(dif.dataset == ds) & (dif.protocol == "random")]
    for lbl, ra, mb, val in ((f"H10 {ds} SSE R0->R1", "R0", "R1", sse01),
                             (f"H10 {ds} SSE R1->R2", "R1", "R2", sse12),
                             (f"H10 {ds} SSE R0->R2", "R0", "R2", sse02)):
        row = dr[(dr.ref_a == ra) & (dr.method_b == mb)]
        # SSE_red se almacena como fracción; el documento la cita en porcentaje
        chk(lbl, val, 100 * row[sse_col].iloc[0] if len(row) else None, 0.15)
    gg = dif[(dif.dataset == ds) & (dif.protocol == "grouped") & (dif.ref_a == "R1") & (dif.method_b == "R2")]
    chk(f"H10 {ds} dMAE R1->R2 agrupado", dmae_g, gg[dmae_col].iloc[0] if len(gg) else None, 0.01)

# ---- H05/H06 selección ----
k = pd.read_csv(RV3 / "topk_metrics.csv")
sel = k[(k.K == 100) & (k.protocol == "grouped") & (k.model == "XGBoost") & (k.fold == "mean_5folds")]
for ds, prec, enr, nfam in (("A", 87.0, 10.71, 19.0), ("Bv3", 82.4, 8.57, 26.8)):
    s = sel[sel.dataset == ds].iloc[0]
    chk(f"H05 {ds} P@100", prec, s.precision_pct, 0.06)
    chk(f"H05 {ds} enriquecimiento", enr, s.enrichment, 0.06)
    chk(f"H05 {ds} familias en top-100", nfam, s.n_families_topK, 0.06)
    chk(f"H05 {ds} presupuesto <=3/familia", {"A": 74.2, "Bv3": 71.4}[ds], s.budget_precision_pct, 0.06)
prev = pd.read_csv(RV3 / "topk_prevalence.csv")
pa = prev[(prev.dataset == "A") & (prev.protocol == "grouped")]
chk("H06 A prevalencia min", 3.62, pa.prevalence_test_ge_pct.min(), 0.01)
chk("H06 A prevalencia max", 18.22, pa.prevalence_test_ge_pct.max(), 0.01)
chk("H06 A prevalencia media", 10.33, pa.prevalence_test_ge_pct.mean(), 0.01)
pb = prev[(prev.dataset == "Bv3") & (prev.protocol == "grouped")]
chk("H06 B v3 prevalencia min", 6.28, pb.prevalence_test_ge_pct.min(), 0.01)
chk("H06 B v3 prevalencia max", 13.50, pb.prevalence_test_ge_pct.max(), 0.01)
chk("H06 B v3 prevalencia media", 10.17, pb.prevalence_test_ge_pct.mean(), 0.01)
neg = pd.read_csv(RV3 / "topk_with_negatives.csv")
nn = neg[(neg.K == 100) & (neg.protocol == "grouped") & (neg.model == "XGBoost") & (neg.fold == "mean_5folds")].iloc[0]
chk("H05 B P@100 solo positivos", 82.4, nn.precision_pos_only_pct, 0.06)
chk("H05 B P@100 con negativos", 78.2, nn.precision_with_neg_pct, 0.06)
chk("H05 B IC inf con negativos", 70.0, nn.precision_with_neg_ci_fam_lo_pct, 0.06)
chk("H05 B IC sup con negativos", 84.8, nn.precision_with_neg_ci_fam_hi_pct, 0.06)
chk("H05 B negativos por top-100", 6.2, nn.expected_neg_in_topK, 0.06)
chk("H05 B negativos totales (5 folds)", 31, nn.expected_neg_in_topK_sum_5folds, 0.51)
chk("H05 B negativos en el pool", 3925, neg.n_neg_pool.max(), 0.51)

# ---- H11 dispersión de réplicas ----
rv = pd.read_csv(RV3 / "replicate_dispersion.csv").set_index("metrica").valor
chk("H11 n filas", 8511, rv["n_filas_con_vector_repetido"], 0.51)
chk("H11 n grupos", 2418, rv["n_grupos"], 0.51)
chk("H11 dev mediana", 3.2229, rv["desv_abs_media_a_MEDIANA_del_grupo_K"], 1e-3)
chk("H11 dev media", 3.4913, rv["desv_abs_media_a_MEDIA_del_grupo_K"], 1e-3)
chk("H11 baseline réplica excluida", 4.72, rv["baseline_replica_excluida_MEDIANA_otras_MAE_K"], 6e-3)
chk("H11 XGB random en esas filas", 4.83, rv["MAE_modelo_XGBoost_random_filas_con_vector_repetido_K"], 6e-3)

# ---- Experimento 7 ----
tv = pd.read_csv(RV3 / "temporal_validation.csv")
p10 = tv[(tv.corte == 2010) & (tv.escenario == "principal") & (tv.modelo == "XGBoost") & (tv.estrato == "todo")]
val = lambda cond, met: (p10[(p10.condicion == cond) & (p10.metrica == met)].valor.iloc[0]
                         if len(p10[(p10.condicion == cond) & (p10.metrica == met)]) else None)
chk("Exp7 MAE real 2010", 7.43, val("temporal_real", "MAE"), 6e-3)
chk("Exp7 R2 real 2010", 0.33, val("temporal_real", "R2"), 6e-3)
chk("Exp7 MAE L0 2010", 4.34, val("L0_aleatorio", "MAE"), 6e-3)
chk("Exp7 R2 L0 2010", 0.91, val("L0_aleatorio", "R2"), 6e-3)
nuevas = tv[(tv.corte == 2010) & (tv.escenario == "principal") & (tv.modelo == "XGBoost")
            & (tv.condicion == "temporal_real") & (tv.estrato == "familia_nueva")]
chk("Exp7 R2 familias nuevas", 0.19, nuevas[nuevas.metrica == "R2"].valor.iloc[0] if len(nuevas) else None, 6e-3)
vistas = tv[(tv.corte == 2010) & (tv.escenario == "principal") & (tv.modelo == "XGBoost")
            & (tv.condicion == "temporal_real") & (tv.estrato == "familia_vista")]
chk("Exp7 R2 familias vistas", 0.83, vistas[vistas.metrica == "R2"].valor.iloc[0] if len(vistas) else None, 6e-3)

print(f"comprobaciones OK: {len(ok)}")
print(f"DISCREPANCIAS: {len(bad)}")
for b in bad:
    print("  ✗", b)
sys.exit(1 if bad else 0)
