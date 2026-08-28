"""
run_v2_measurements.py -- v2 measurements: element-presence null, family-identity
ceiling / design effect, and effect-vs-seed-noise check, on Datasets A and B.

Hyperparameters are IMPORTED from the published repo scripts (never rewritten):
  Dataset A -> tc_pipeline.make_models(seed)["XGBoost"]
  Dataset B -> datasetB_pipeline.make_models(seed)["XGBoost"]
Grouping is ALWAYS the constituent-element set via pandas.factorize:
  Dataset A -> tc_pipeline.chemical_families(unique_m)
  Dataset B -> 'groups' field of datasetB_featurized.npz (verified == factorize of
               "|".join(sorted(elems)) on the re-featurized, Tc>0-filtered rows)

Outputs (results_v2/): nulo_presencia.csv, design_effect.csv, efecto_vs_ruido.md,
                       verificacion_alineacion_B.json, seed_runs_top100.csv
Nothing is written inside the repo working tree.
"""
from __future__ import annotations
import sys, os, json, time, importlib.util
sys.dont_write_bytecode = True          # keep the clean repo tree free of __pycache__
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold

REPO = "/Users/freakscode/Proyectos 2025/Proyectos Académicos/Superconductors/superconductor-tc-leakage-eval"
OUT  = "/Users/freakscode/Proyectos 2025/Proyectos Académicos/Superconductors/results_v2"
SCRIPT = "results_v2/run_v2_measurements.py"
PRIMARY_SEED, SEEDS = 42, [42, 0, 1, 2, 3]
K_TOP = 100

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

tcp = load_mod("tc_pipeline",       f"{REPO}/code/tc_pipeline.py")
dsB = load_mod("datasetB_pipeline", f"{REPO}/code/datasetB_pipeline.py")

# --------------------------------------------------------------- metric helpers
def metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    return dict(MAE=float(np.mean(np.abs(yt - yp))),
                RMSE=float(np.sqrt(np.mean((yt - yp) ** 2))),
                R2=float(1.0 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()))

def topk_from_preds(y_tr, y_te, yp_te, K=K_TOP):
    """Identical definition to tc_pipeline.topk_precision / datasetB_pipeline's inner
    topk: threshold = 90th percentile of the TRAINING fold, take the K highest
    predicted in the test pool, report the fraction truly above threshold."""
    thr = np.percentile(y_tr, 90)
    return float((y_te[np.argsort(yp_te)[-K:]] >= thr).mean())

def make_splitter(scheme, seed, shuffle_groups):
    if scheme == "random":
        return KFold(5, shuffle=True, random_state=seed)
    if shuffle_groups:
        return GroupKFold(5, shuffle=True, random_state=seed)
    return GroupKFold(5)

def iter_folds(splitter, X, y, groups, scheme):
    return splitter.split(X, y) if scheme == "random" else splitter.split(X, y, groups)

# --------------------------------------------------------------- one evaluation
def evaluate(model_factory, X, y, groups, scheme, seed, shuffle_groups=False,
             rep_is_null_lookup=False):
    """Pooled out-of-fold MAE/RMSE/R2 + per-fold top-K precision. One fit per fold
    serves both. rep_is_null_lookup -> featureless family-mean predictor."""
    spl = make_splitter(scheme, seed, shuffle_groups)
    yp_oof = np.empty(len(y), dtype=float); topks = []
    for tr, te in iter_folds(spl, X, y, groups, scheme):
        if rep_is_null_lookup:
            gmean = y[tr].mean()
            fam = pd.Series(y[tr]).groupby(groups[tr]).mean().to_dict()
            yp = np.array([fam.get(groups[i], gmean) for i in te], dtype=float)
        else:
            m = model_factory(seed); m.fit(X[tr], y[tr]); yp = m.predict(X[te])
        yp_oof[te] = yp
        topks.append(topk_from_preds(y[tr], y[te], yp))
    out = metrics(y, yp_oof)
    out.update(top100_mean=float(np.mean(topks)), top100_std=float(np.std(topks)),
               top100_folds=[float(t) for t in topks])
    return out

# =============================================================== DATA
t_start = time.time()
featA = pd.read_csv(f"{REPO}/data/train.csv"); uniqA = pd.read_csv(f"{REPO}/data/unique_m.csv")
assert np.allclose(featA["critical_temp"], uniqA["critical_temp"]), "A not row-aligned"
XA_full = featA.drop(columns=["critical_temp"]).values.astype(np.float64)
yA = featA["critical_temp"].values.astype(np.float64)
gA = tcp.chemical_families(uniqA)
elem_colsA = [c for c in uniqA.columns if c not in ("critical_temp", "material")]
XA_pres = (uniqA[elem_colsA].values > 0).astype(np.float64)

z = np.load(f"{REPO}/data/datasetB_featurized.npz", allow_pickle=True)
XB_full, yB, gB = z["X"].astype(np.float64), z["y"].astype(np.float64), z["groups"]

# --- Dataset B: rebuild formulas with the repo's own featurizer and VERIFY alignment
XB_re, yB_re, gB_re, formulas_all = dsB.featurize(f"{REPO}/data/supercon_stanev.csv")
m_pos = yB_re > 0
XB_re_f, yB_re_f, gB_re_f, formulas = XB_re[m_pos], yB_re[m_pos], gB_re[m_pos], formulas_all[m_pos]

from pymatgen.core import Composition
elem_sets = [sorted(e.symbol for e in Composition(f).elements) for f in formulas]
elem_colsB = sorted({e for s in elem_sets for e in s})
jdx = {e: j for j, e in enumerate(elem_colsB)}
XB_pres = np.zeros((len(formulas), len(elem_colsB)), dtype=np.float64)
for i, s in enumerate(elem_sets):
    for e in s: XB_pres[i, jdx[e]] = 1.0
key_pres = pd.factorize(pd.Series(["|".join(s) for s in elem_sets]))[0]

def same_partition(a, b):
    ca, cb = pd.factorize(np.asarray(a))[0], pd.factorize(np.asarray(b))[0]
    df = pd.DataFrame({"a": ca, "b": cb})
    return (bool(df.groupby("a")["b"].nunique().max() == 1 and
                 df.groupby("b")["a"].nunique().max() == 1),
            int(df.groupby("a")["b"].nunique().max()), int(df.groupby("b")["a"].nunique().max()))

align = dict(
    n_rows_after_Tc_gt0_filter=int(len(yB_re_f)), n_rows_npz=int(len(yB)), expected=12440,
    n_rows_ok=bool(len(yB_re_f) == 12440 == len(yB)),
    y_allclose=bool(np.allclose(yB_re_f, yB)), y_array_equal=bool(np.array_equal(yB_re_f, yB)),
    max_abs_dy=float(np.abs(yB_re_f - yB).max()),
    X_allclose=bool(np.allclose(XB_re_f, XB_full, equal_nan=True)),
    max_abs_dX=float(np.abs(XB_re_f - XB_full).max()),
    groups_codes_array_equal_prefilter_factorize=bool(np.array_equal(gB_re_f, gB)),
    groups_codes_array_equal_postfilter_factorize=bool(np.array_equal(key_pres, gB)),
    groups_same_partition_vs_prefilter=same_partition(gB, gB_re_f),
    groups_same_partition_vs_postfilter=same_partition(gB, key_pres),
    n_families_npz=int(len(np.unique(gB))), n_families_rebuilt=int(len(np.unique(key_pres))),
    nota=("Los codigos enteros de 'groups' del npz corresponden a un factorize APLICADO "
          "DESPUES del filtro Tc>0; un factorize aplicado ANTES del filtro produce "
          "etiquetas distintas pero la PARTICION es identica (1-a-1 en ambos sentidos). "
          "y y X coinciden exactamente (X a precision float32). Alineacion CONFIRMADA."),
    procedencia=f"{SCRIPT} | datasetB_pipeline.featurize(supercon_stanev.csv) vs datasetB_featurized.npz",
)
assert align["n_rows_ok"] and align["y_allclose"] and align["X_allclose"] \
       and align["groups_same_partition_vs_prefilter"][0], f"DATASET B NOT ALIGNED: {align}"
json.dump(align, open(f"{OUT}/verificacion_alineacion_B.json", "w"), indent=2, ensure_ascii=False)

# cross-check that our top-K definition reproduces the repo's function on Dataset A
_tr, _te = next(GroupKFold(5).split(XA_full, yA, gA))
_m = tcp.make_models(PRIMARY_SEED)["XGBoost"]; _m.fit(XA_full[_tr], yA[_tr])
_ours = topk_from_preds(yA[_tr], yA[_te], _m.predict(XA_full[_te]))
_theirs = tcp.topk_precision(XA_full, yA, _tr, _te, K=K_TOP, seed=PRIMARY_SEED)
assert abs(_ours - _theirs) < 1e-12, f"topk definition mismatch {_ours} vs {_theirs}"
print(f"[check] topk def matches tc_pipeline.topk_precision: {_ours:.4f} == {_theirs:.4f}", flush=True)

DATASETS = {
    "A": dict(y=yA, groups=gA, factory=lambda s: tcp.make_models(s)["XGBoost"],
              reps={"features_completas_paper": (XA_full, "81 estadisticas Hamidieh (train.csv)"),
                    "presencia_elementos":      (XA_pres, f"{XA_pres.shape[1]} indicadores binarios (unique_m>0)")},
              hp="tc_pipeline.make_models: XGB n_est=250 depth=8 lr=0.07 sub=0.9 col=0.7 gamma=0 alpha=0.01",
              grp="tc_pipeline.chemical_families(unique_m.csv)"),
    "B": dict(y=yB, groups=gB, factory=lambda s: dsB.make_models(s)["XGBoost"],
              reps={"features_completas_paper": (XB_full, "132 Magpie matminer (npz checkpoint)"),
                    "presencia_elementos":      (XB_pres, f"{XB_pres.shape[1]} indicadores binarios (pymatgen Composition)")},
              hp="datasetB_pipeline.make_models: XGB n_est=300 depth=10 lr=0.05 sub=0.8 col=0.8",
              grp="npz 'groups' (== factorize de '|'.join(sorted(elems)), particion verificada)"),
}

# =============================================================== PASO A
rows = []
for ds, cfg in DATASETS.items():
    y, g, fac = cfg["y"], cfg["groups"], cfg["factory"]
    for rep, (X, desc) in cfg["reps"].items():
        for scheme, scheme_lbl in (("random", "aleatorio_KFold5"), ("grouped", "GroupKFold5_familias")):
            t0 = time.time()
            r = evaluate(fac, X, y, g, scheme, PRIMARY_SEED)
            rows.append(dict(dataset=ds, representacion=rep, split=scheme_lbl,
                MAE=r["MAE"], RMSE=r["RMSE"], R2=r["R2"],
                top100_mean=r["top100_mean"], top100_std=r["top100_std"],
                n_features=int(X.shape[1]), N=int(len(y)), k_familias=int(len(np.unique(g))),
                top100_folds=";".join(f"{t:.4f}" for t in r["top100_folds"]),
                procedencia=(f"script={SCRIPT} | {cfg['hp']} | seed={PRIMARY_SEED} | "
                             f"rep={desc} | grupos={cfg['grp']} | "
                             f"top100: K=100, umbral=p90(y_fold_entrenamiento), "
                             f"OOF pooled MAE/RMSE/R2 | split={('KFold(5,shuffle,rs=42)' if scheme=='random' else 'GroupKFold(5) determinista')}")))
            print(f"[A] {ds}/{rep}/{scheme}: MAE={r['MAE']:.3f} R2={r['R2']:.4f} "
                  f"top100={100*r['top100_mean']:.1f}+-{100*r['top100_std']:.1f} ({time.time()-t0:.0f}s)", flush=True)
        # featureless family-mean null (no fit) for context, same folds
    for scheme, scheme_lbl in (("random", "aleatorio_KFold5"), ("grouped", "GroupKFold5_familias")):
        r = evaluate(None, cfg["reps"]["features_completas_paper"][0], y, g, scheme,
                     PRIMARY_SEED, rep_is_null_lookup=True)
        rows.append(dict(dataset=ds, representacion="nulo_media_familia", split=scheme_lbl,
            MAE=r["MAE"], RMSE=r["RMSE"], R2=r["R2"], top100_mean=r["top100_mean"],
            top100_std=r["top100_std"], n_features=0, N=int(len(y)), k_familias=int(len(np.unique(g))),
            top100_folds=";".join(f"{t:.4f}" for t in r["top100_folds"]),
            procedencia=(f"script={SCRIPT} | predictor SIN features: media de Tc de la familia en "
                         f"el fold de entrenamiento (media global si familia no vista) | seed={PRIMARY_SEED} | "
                         f"grupos={cfg['grp']} | top100: K=100, umbral=p90(y_fold_entrenamiento)")))
        print(f"[A] {ds}/family_mean_null/{scheme}: MAE={r['MAE']:.3f} R2={r['R2']:.4f} "
              f"top100={100*r['top100_mean']:.1f}", flush=True)

pd.DataFrame(rows).to_csv(f"{OUT}/nulo_presencia.csv", index=False)
print("wrote nulo_presencia.csv", flush=True)

# =============================================================== PASO B
def design_effect(y, groups):
    y = np.asarray(y, float); codes, uniq = pd.factorize(np.asarray(groups))
    N, k = len(y), len(uniq)
    ni = np.bincount(codes, minlength=k).astype(float)
    gm = np.bincount(codes, weights=y, minlength=k) / ni
    grand = y.mean()
    SS_total  = float(((y - grand) ** 2).sum())
    SS_within = float(((y - gm[codes]) ** 2).sum())
    SS_between = float((ni * (gm - grand) ** 2).sum())
    MSB, MSW = SS_between / (k - 1), SS_within / (N - k)
    m0 = (N - (ni ** 2).sum() / N) / (k - 1)                 # Searle
    ICC = (MSB - MSW) / (MSB + (m0 - 1) * MSW)
    m_bar = N / k
    d_m, d_s = 1 + (m_bar - 1) * ICC, 1 + (m0 - 1) * ICC
    return dict(N=N, k=k, m_bar=m_bar, m0_searle=m0, ICC=ICC,
                eta2=1.0 - SS_within / SS_total, MSB=MSB, MSW=MSW,
                SS_total=SS_total, SS_within=SS_within, SS_between=SS_between,
                deff_mbar=d_m, deff_searle=d_s, N_eff_mbar=N / d_m, N_eff_searle=N / d_s,
                razon_contra_k=(N / d_s) / k, razon_contra_k_mbar=(N / d_m) / k,
                razon_contra_k_searle=(N / d_s) / k,
                max_family_size=int(ni.max()), n_singleton_families=int((ni == 1).sum()),
                pct_rows_in_largest_family=float(100 * ni.max() / N))

de_rows = []
for ds, cfg in DATASETS.items():
    d = design_effect(cfg["y"], cfg["groups"])
    d.update(dataset=ds, eta2_tipo="ORACULO (todos los datos, NO es un R2 de CV)",
             procedencia=(f"script={SCRIPT} | ANOVA de efectos aleatorios de una via, clusters "
                          f"desiguales | m0 de Searle=(N-sum(ni^2)/N)/(k-1) | "
                          f"ICC=(MSB-MSW)/(MSB+(m0-1)MSW) | eta2=1-SSW/SStot con media de Tc por "
                          f"familia (oraculo) | deff=1+(m-1)ICC con m in (m_bar, m0) | "
                          f"razon_contra_k usa la variante Searle | grupos={cfg['grp']}"))
    de_rows.append(d)
    print(f"[B] {ds}: N={d['N']} k={d['k']} ICC={d['ICC']:.4f} eta2={d['eta2']:.4f} "
          f"deff=({d['deff_mbar']:.3f},{d['deff_searle']:.3f}) "
          f"Neff=({d['N_eff_mbar']:.0f},{d['N_eff_searle']:.0f})", flush=True)

cols = ["dataset","N","k","m_bar","m0_searle","ICC","eta2","deff_mbar","deff_searle",
        "N_eff_mbar","N_eff_searle","razon_contra_k","razon_contra_k_mbar","razon_contra_k_searle",
        "MSB","MSW","SS_total","SS_within","SS_between","max_family_size",
        "n_singleton_families","pct_rows_in_largest_family","eta2_tipo","procedencia"]
pd.DataFrame(de_rows)[cols].to_csv(f"{OUT}/design_effect.csv", index=False)
print("wrote design_effect.csv", flush=True)

# =============================================================== PASO C
seed_rows, seed_store = [], {}
for ds, cfg in DATASETS.items():
    y, g, fac = cfg["y"], cfg["groups"], cfg["factory"]
    for rep in ("features_completas_paper", "presencia_elementos"):
        X = cfg["reps"][rep][0]; vals = []
        for s in SEEDS:
            r = evaluate(fac, X, y, g, "grouped", s, shuffle_groups=True)
            vals.append(100 * r["top100_mean"])
            seed_rows.append(dict(dataset=ds, representacion=rep, seed=s,
                top100_pct=100 * r["top100_mean"], top100_std_folds_pct=100 * r["top100_std"],
                MAE=r["MAE"], R2=r["R2"],
                top100_folds=";".join(f"{t:.4f}" for t in r["top100_folds"]),
                procedencia=(f"script={SCRIPT} | {cfg['hp']} | seed={s} (semilla del modelo Y de la "
                             f"asignacion de folds) | split=GroupKFold(5,shuffle=True,random_state=seed) | "
                             f"top100: K=100, umbral=p90(y_fold_entrenamiento)")))
            print(f"[C] {ds}/{rep}/seed={s}: top100={100*r['top100_mean']:.2f}", flush=True)
        seed_store[(ds, rep)] = vals

pd.DataFrame(seed_rows).to_csv(f"{OUT}/seed_runs_top100.csv", index=False)
json.dump({f"{d}|{r}": v for (d, r), v in seed_store.items()},
          open(f"{OUT}/seed_store.json", "w"), indent=2)
print(f"wrote seed_runs_top100.csv | TOTAL wall {time.time()-t_start:.0f}s", flush=True)
