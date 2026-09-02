"""Comprueba que cada cifra del .tex coincide con su CSV de results_v2."""
import argparse, os, re, sys, pandas as pd
from pathlib import Path
# --- repository root (portable) -------------------------------------------------------
# ROOT = the repository root. Override with the SUPERCON_ROOT environment variable;
# by default it is resolved from this file's location: code/v2/<script>.py -> parents[2].
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
REPO = ROOT                       # the repository itself (code/, data/, results/, figures/)
CODE = ROOT / "code"              # tc_pipeline.py, datasetB_pipeline.py (hyperparameters live here)
DATA = ROOT / "data"
OUT  = ROOT / "results_v2"        # v2 measurements (inside the repository)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--tex", type=Path, default=None,
                help="manuscript .tex (default: ROOT.parent/paper1_submission/paper1_manuscript.tex if it exists)")
ap.add_argument("--dry-run", action="store_true", help="resolve paths, check inputs exist, exit")
args = ap.parse_args()
TEX_PATH = args.tex or (ROOT.parent / "paper1_submission" / "paper1_manuscript.tex")
_CSV = ["nulo_presencia.csv", "design_effect.csv", "tabla2_shuffle.csv", "top100_unificado.csv",
        "nn_distance.csv", "family_concentration.csv", "reproduction_check.csv"]
if args.dry_run:
    print(f"ROOT={ROOT}\nOUT={OUT}\nTEX={TEX_PATH}")
    for p in [TEX_PATH] + [OUT / f for f in _CSV]:
        print(f"  {'ok ' if p.exists() else 'MISSING'} {p}")
    sys.exit(0)
if not TEX_PATH.exists():
    sys.exit(f"manuscript not found: {TEX_PATH}  (pass --tex PATH)")
TEX = open(TEX_PATH).read()
np_ = lambda d, c: d[c].astype(float)
ok, bad = [], []
def chk(label, tex_val, src_val, tol=0.006):
    hit = abs(float(tex_val) - float(src_val)) <= tol
    (ok if hit else bad).append(f"{label}: tex={tex_val} src={src_val:.4f}")
def intex(s):  # ¿aparece literalmente en el .tex?
    return s in TEX

npr = pd.read_csv(OUT / "nulo_presencia.csv")
de  = pd.read_csv(OUT / "design_effect.csv")
t2  = pd.read_csv(OUT / "tabla2_shuffle.csv")
t1  = pd.read_csv(OUT / "top100_unificado.csv")
nn  = pd.read_csv(OUT / "nn_distance.csv")
fc  = pd.read_csv(OUT / "family_concentration.csv")
rc  = pd.read_csv(OUT / "reproduction_check.csv")

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

print(f"tex: {TEX_PATH}")

# ---------------------------------------------------------------------------------------------
# v2.1 — cifras añadidas tras la auditoría de replicabilidad (twins/noise, bootstrap, CV anidada,
#        rejilla de hiperparámetros, estratificación, corrección de factorize, filtros de datos)
# ---------------------------------------------------------------------------------------------
import json as _json
def need(label, s):
    # un rango "a}{b" (\SIrange/\numrange) también vale escrito "a--b" (celda de tabla)
    hit = intex(s) or ("}{" in s and intex(s.replace("}{", "--")))
    (ok if hit else bad).append(f"{label}: {s}" + ("" if hit else " NO está en el tex"))

_tw = pd.read_csv(OUT / "twins_noise_A.csv")
tw = _tw.drop_duplicates("metrica", keep="first").set_index("metrica")["valor"]   # primera fila = split random
fr = _tw[_tw.metrica=="frac_test_con_gemelo_exacto_en_train"].set_index("split")
need("twins frac random", f"{100*fr.loc['random_KFold5_rs42','valor']:.1f} \\pm {100*fr.loc['random_KFold5_rs42','sd_entre_folds']:.1f}")
need("twins frac grouped", f"{100*fr.loc['GroupKFold5_familias','valor']:.1f}}}{{\\percent")
need("twins n filas", str(int(tw["n_filas_con_gemelo"])))
need("twins piso", f"{tw['piso_ruido_MAE_solo_filas_con_gemelo_K']:.2f}")
need("twins MAE modelo", f"{tw['MAE_random_modelo_solo_filas_con_gemelo_K']:.2f}")
need("dup features", str(int(tw["filas_con_features_duplicadas"])))
need("dup pct", f"{tw['pct_filas_features_duplicadas']:.1f}")
need("dup exactos", str(int(tw["filas_duplicadas_features_y_Tc"])))

bs = pd.read_csv(OUT / "bootstrap_intervalos.csv").set_index("dataset")
for ds in ("A","B"):
    r = bs.loc[ds]
    for s_ in (f"{r.inflacion_lo:.1f}}}{{{r.inflacion_hi:.1f}", f"{r.MAE_random_lo:.2f}}}{{{r.MAE_random_hi:.2f}", f"{r.MAE_grouped_lo:.2f}}}{{{r.MAE_grouped_hi:.2f}"):
        need(f"bootstrap {ds}", s_)

nc = pd.read_csv(OUT / "nested_cv_inflacion.csv").set_index("dataset")
for ds in ("A","B"):
    r = nc.loc[ds]
    for s_ in (f"{r.MAE_grouped_anidada:.2f}", f"{r.inflacion_anidada_vs_random_modal_pct:.1f}", f"{r.MAE_random_modal:.2f}",
               f"{r.MAE_random_publicada:.2f}", f"{r.MAE_grouped_publicada_GKF5:.2f}", f"{r.inflacion_publicada_misma_config_pct:.1f}"):
        need(f"nested {ds}", s_)

hg = pd.read_csv(OUT / "hp_grid_ambos.csv"); hg = hg[hg.es_publicada != True] if "es_publicada" in hg else hg
for ds, g in hg.groupby("dataset"):
    for d, h in g.groupby("max_depth"):
        for col, fmt in (("inflacion_pct","{:.1f}"),("MAE_random","{:.2f}"),("MAE_grouped","{:.2f}")):
            need(f"grid {ds} d{d} {col}", fmt.format(h[col].min()) + "}{" + fmt.format(h[col].max()))

sg = pd.read_csv(OUT / "stratified_group_check.csv")
for ds, g in sg.groupby("dataset"):
    for sp in ("GroupKFold","StratifiedGroupKFold"):
        v = g[g.splitter==sp].inflacion_pct
        need(f"strat {ds} {sp}", f"{v.min():.1f}}}{{{v.max():.1f}")

fx = _json.load(open(OUT / "fix_factorize_verificacion.json"))
for k in ("rows_changing_fold_pre_vs_npz","n_rows_stanev_csv","n_parse_failures_pymatgen"):
    need(f"factorize {k}", str(fx[k]))
pf = fx["rows_changing_fold_pre_vs_npz_per_fold"]; need("factorize per-fold", f"{min(pf)}}}{{{max(pf)}")
need("filtro Tc<=0", str(fx["n_rows_stanev_csv"] - fx["n_parse_failures_pymatgen"] - fx["n_rows_Tc_gt0"]))

print(f"comprobaciones OK: {len(ok)}")
print(f"DISCREPANCIAS: {len(bad)}")
for b in bad: print("  ✗", b)
sys.exit(1 if bad else 0)
