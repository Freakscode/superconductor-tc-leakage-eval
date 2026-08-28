"""Comprueba que cada cifra del .tex coincide con su CSV de results_v2."""
import re, sys, pandas as pd
TEX = open("paper1_submission/paper1_manuscript.tex").read()
np_ = lambda d, c: d[c].astype(float)
ok, bad = [], []
def chk(label, tex_val, src_val, tol=0.006):
    hit = abs(float(tex_val) - float(src_val)) <= tol
    (ok if hit else bad).append(f"{label}: tex={tex_val} src={src_val:.4f}")
def intex(s):  # ¿aparece literalmente en el .tex?
    return s in TEX

npr = pd.read_csv("results_v2/nulo_presencia.csv")
de  = pd.read_csv("results_v2/design_effect.csv")
t2  = pd.read_csv("results_v2/tabla2_shuffle.csv")
t1  = pd.read_csv("results_v2/top100_unificado.csv")
nn  = pd.read_csv("results_v2/nn_distance.csv")
fc  = pd.read_csv("results_v2/family_concentration.csv")
rc  = pd.read_csv("results_v2/reproduction_check.csv")

# --- Tabla 1 del manuscrito (tab:main) contra nulo_presencia.csv ---
rep = {"features_completas_paper":"XGBoost", "nulo_media_familia":"Family-mean null"}
for _, r in npr[npr.representacion.isin(rep)].iterrows():
    sp = "Random" if "aleatorio" in r.split else "Chemical-family"
    for col, val in [("MAE", r.MAE), ("RMSE", r.RMSE), ("R2", r.R2)]:
        s = f"{val:.2f}" if col != "R2" else f"{val:.3f}"
        chk(f"tab:main {r.dataset} {rep[r.representacion]} {sp} {col}={s}", s, val, 0.006)
        if not intex(s): bad.append(f"AUSENTE del tex: {r.dataset} {rep[r.representacion]} {sp} {col}={s}")

# --- Tabla 2 (tab:robust) ---
for _, r in t2.iterrows():
    for s, v in [(f"{r.MAE_random_mean:.2f}", r.MAE_random_mean), (f"{r.MAE_grouped_mean:.2f}", r.MAE_grouped_mean),
                 (f"{r.inflacion_pct_mean:.1f}", r.inflacion_pct_mean), (f"{r.dR2_mean:.3f}", r.dR2_mean)]:
        if not intex(s.lstrip("-")): bad.append(f"tab:robust {r.dataset}/{r.modelo}: {s} no está en el tex")
        else: ok.append(f"tab:robust {r.dataset}/{r.modelo} {s}")

# --- Tabla 3 (tab:deff) ---
for _, r in de.iterrows():
    for s in [f"{r.ICC:.3f}", f"{r.eta2:.3f}", f"{r.deff_searle:.2f}", f"{int(round(r.N_eff_searle))}",
              f"{r.razon_contra_k_searle:.2f}", str(int(r.N)), str(int(r.k))]:
        if not intex(s): bad.append(f"tab:deff {r.dataset}: {s} no está en el tex")
        else: ok.append(f"tab:deff {r.dataset} {s}")

# --- Tabla 4 (tab:screen) y top100 ---
for _, r in t1.iterrows():
    s = f"{100*r.top100_mean:.1f}"
    if not intex(s): bad.append(f"top100 {r.dataset} {r.split}: {s} no está en el tex")
    else: ok.append(f"top100 {r.dataset} {r.split} {s}")

# --- diagnósticos ---
for _, r in nn.iterrows():
    for s in [f"{r.nn_median_random:.3f}", f"{r.nn_median_grouped:.3f}"]:
        if not intex(s): bad.append(f"nn {r.dataset}: {s} no está en el tex")
for _, r in fc.iterrows():
    if not intex(f"{r.pct_en_familias_ge5:.1f}"): bad.append(f"conc {r.dataset}: {r.pct_en_familias_ge5:.1f} ausente")
for _, r in rc.iterrows():
    for s in [f"{r.MAE:.2f}", f"{r.R2:.3f}"]:
        if not intex(s): bad.append(f"repro {r.dataset}: {s} no está en el tex")

print(f"comprobaciones OK: {len(ok)}")
print(f"DISCREPANCIAS: {len(bad)}")
for b in bad: print("  ✗", b)
