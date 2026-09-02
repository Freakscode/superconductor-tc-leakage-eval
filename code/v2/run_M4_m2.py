"""
run_M4_m2.py -- M4 (nested grouped-CV hyperparameter tuning: is the ~58 % inflation a
property of the dataset or of tuning under random CV?) and m2 (StratifiedGroupKFold
robustness check) on Datasets A and B, XGBoost only.

Hyperparameters of the PUBLISHED configuration are IMPORTED from the repo scripts
(tc_pipeline.make_models / datasetB_pipeline.make_models), never rewritten. The grid
only varies max_depth / learning_rate / min_child_weight with n_estimators=300 and the
subsample / colsample_bytree of the published configuration of each dataset.
Grouping is ALWAYS the constituent-element set via pandas.factorize
(A: tc_pipeline.chemical_families(unique_m); B: 'groups' field of the .npz).

Usage:  python run_M4_m2.py <A|B> <grid|nested|strat>     -> results_v2/_m4_raw/<dataset>_<mode>.json
        python run_M4_m2.py merge                        -> the CSV deliverables
Outputs go to <repo>/results_v2/ (ROOT = repo root, or SUPERCON_ROOT). Usage: run_M4_m2.py {A,B} {grid,nested,strat,merge}
"""
from __future__ import annotations
import os
from pathlib import Path
import sys, os, json, time, importlib.util, itertools
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import sklearn, xgboost
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from xgboost import XGBRegressor

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
REPO = ROOT
OUT  = ROOT / "results_v2"
RAW  = OUT / "_m4_raw"; os.makedirs(RAW, exist_ok=True)
SCRIPT = "code/v2/run_M4_m2.py"
N_JOBS = 8
VERS = f"sklearn {sklearn.__version__}, xgboost {xgboost.__version__}"

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

tcp = load_mod("tc_pipeline",       f"{REPO}/code/tc_pipeline.py")
dsB = load_mod("datasetB_pipeline", f"{REPO}/code/datasetB_pipeline.py")

def load_dataset(ds):
    if ds == "A":
        feat = pd.read_csv(f"{REPO}/data/train.csv"); uniq = pd.read_csv(f"{REPO}/data/unique_m.csv")
        assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "A not row-aligned"
        X = feat.drop(columns=["critical_temp"]).values.astype(np.float64)
        y = feat["critical_temp"].values.astype(np.float64)
        g = tcp.chemical_families(uniq)
        fam_names = pd.Series(uniq["material"].values)
        return X, y, g, tcp.make_models, uniq
    d = np.load(f"{REPO}/data/datasetB_featurized.npz")
    X, y, g = d["X"].astype(np.float32), d["y"].astype(np.float64), d["groups"]
    assert len(X) == 12440 and (y > 0).all()
    return X, y, g, dsB.make_models, None

def published_params(make_models, seed):
    """Exact published XGBoost hyperparameters, read from the repo factory (never retyped)."""
    return make_models(seed)["XGBoost"].get_params()

def grid_configs(pub):
    cfgs = []
    for md, lr, mcw in itertools.product([4, 6, 8, 10], [0.05, 0.1], [1, 10]):
        cfgs.append(dict(max_depth=md, learning_rate=lr, min_child_weight=mcw, n_estimators=300,
                         subsample=pub["subsample"], colsample_bytree=pub["colsample_bytree"]))
    return cfgs

def cfg_label(c):
    return f"d{c['max_depth']},lr{c['learning_rate']},mcw{c['min_child_weight']}"

def make_xgb(cfg, seed):
    return XGBRegressor(**cfg, n_jobs=N_JOBS, random_state=seed, tree_method="hist")

def oof_mae(model_fn, X, y, splits):
    yp = np.empty(len(y), float); fold_mae = []
    for tr, te in splits:
        m = model_fn(); m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te])
        fold_mae.append(float(np.mean(np.abs(y[te] - yp[te]))))
    return float(np.mean(np.abs(y - yp))), fold_mae, yp

def r2(yt, yp): return float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())

# =================================================================== modes
def run_grid(ds, seed=42):
    X, y, g, mm, _ = load_dataset(ds); pub = published_params(mm, seed)
    rand_splits = list(KFold(5, shuffle=True, random_state=seed).split(X))
    grp_splits  = list(GroupKFold(5).split(X, y, g))
    rows = []
    # published configuration first (exact repo factory, n_estimators may differ from 300)
    todo = [("publicada", None)] + [(cfg_label(c), c) for c in grid_configs(pub)]
    for i, (lab, cfg) in enumerate(todo):
        fn = (lambda: mm(seed)["XGBoost"].set_params(n_jobs=N_JOBS)) if cfg is None else (lambda: make_xgb(cfg, seed))
        t = time.time()
        mr, fr, ypr = oof_mae(fn, X, y, rand_splits)
        mg, fg, ypg = oof_mae(fn, X, y, grp_splits)
        p = pub if cfg is None else cfg
        rows.append(dict(dataset=ds, config=lab, es_publicada=cfg is None,
                         max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                         min_child_weight=p.get("min_child_weight") or 1, n_estimators=p["n_estimators"],
                         subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                         MAE_random=mr, MAE_grouped=mg, R2_random=r2(y, ypr), R2_grouped=r2(y, ypg),
                         gap_K=mg - mr, inflacion_pct=100 * (mg / mr - 1),
                         MAE_random_folds=";".join(f"{v:.4f}" for v in fr),
                         MAE_grouped_folds=";".join(f"{v:.4f}" for v in fg)))
        print(f"[grid {ds}] {i+1}/{len(todo)} {lab}: rand {mr:.3f} grp {mg:.3f} ({time.time()-t:.0f}s)", flush=True)
    json.dump(rows, open(f"{RAW}/{ds}_grid.json", "w"), indent=1)

def run_nested(ds, seed=42):
    X, y, g, mm, _ = load_dataset(ds); pub = published_params(mm, seed)
    cfgs = grid_configs(pub); labels = [cfg_label(c) for c in cfgs]
    outer = list(GroupKFold(5).split(X, y, g))
    yp_oof = np.empty(len(y), float); folds = []; inner_tab = []
    for k, (tr, te) in enumerate(outer):
        t = time.time()
        inner = list(GroupKFold(3).split(X[tr], y[tr], g[tr]))
        inner_mae = np.zeros(len(cfgs))
        for j, c in enumerate(cfgs):
            m_in, f_in, _ = oof_mae(lambda: make_xgb(c, seed), X[tr], y[tr], inner)
            inner_mae[j] = m_in          # pooled OOF MAE over the 3 inner folds
            inner_tab.append(dict(dataset=ds, outer_fold=k, config=labels[j], MAE_interna=m_in,
                                  MAE_interna_folds=";".join(f"{v:.4f}" for v in f_in)))
        j_best = int(np.argmin(inner_mae))
        m = make_xgb(cfgs[j_best], seed); m.fit(X[tr], y[tr]); yp_oof[te] = m.predict(X[te])
        mae_te = float(np.mean(np.abs(y[te] - yp_oof[te])))
        # for reference: what the published config does on the same outer fold
        mp = mm(seed)["XGBoost"].set_params(n_jobs=N_JOBS); mp.fit(X[tr], y[tr])
        mae_pub = float(np.mean(np.abs(y[te] - mp.predict(X[te]))))
        folds.append(dict(dataset=ds, outer_fold=k, n_test=len(te), n_familias_test=len(set(g[te])),
                          config_ganadora=labels[j_best], MAE_interna_ganadora=float(inner_mae[j_best]),
                          MAE_externa_ganadora=mae_te, MAE_externa_publicada=mae_pub,
                          ranking_interno=";".join(f"{labels[i]}={inner_mae[i]:.3f}" for i in np.argsort(inner_mae)[:5])))
        print(f"[nested {ds}] outer {k}: best {labels[j_best]} inner {inner_mae[j_best]:.3f} outer {mae_te:.3f} pub {mae_pub:.3f} ({time.time()-t:.0f}s)", flush=True)
    res = dict(folds=folds, inner=inner_tab, MAE_nested_pooled=float(np.mean(np.abs(y - yp_oof))),
               R2_nested=r2(y, yp_oof))
    json.dump(res, open(f"{RAW}/{ds}_nested.json", "w"), indent=1)

def run_strat(ds, seeds=(0, 1, 2), n_bins=5):
    X, y, g, mm, uniq = load_dataset(ds)
    # 5 quantile bins of Tc for stratification; largest family for the "where does it fall" question
    ybin = pd.qcut(y, n_bins, labels=False, duplicates="drop")
    fam_sizes = pd.Series(g).value_counts(); big = int(fam_sizes.index[0])
    big_name = None
    if uniq is not None:
        ec = [c for c in uniq.columns if c not in ("critical_temp", "material")]
        big_name = "|".join(sorted(e for e in ec if (uniq.loc[g == big, ec] > 0).all()[e]))
    rows = []
    for s in seeds:
        splitters = {
            "KFold":                (KFold(5, shuffle=True, random_state=s), lambda sp: sp.split(X)),
            "GroupKFold":           (GroupKFold(5, shuffle=True, random_state=s), lambda sp: sp.split(X, y, g)),
            "StratifiedGroupKFold": (StratifiedGroupKFold(5, shuffle=True, random_state=s), lambda sp: sp.split(X, ybin, g)),
        }
        for name, (sp, itf) in splitters.items():
            t = time.time(); splits = list(itf(sp))
            if name != "KFold":
                for tr, te in splits: assert not (set(g[tr]) & set(g[te]))
            mae, fmae, yp = oof_mae(lambda: mm(s)["XGBoost"].set_params(n_jobs=N_JOBS), X, y, splits)
            frac77 = [float((y[te] > 77).mean()) for _, te in splits]
            fold_sizes = [len(te) for _, te in splits]
            big_fold = [k for k, (_, te) in enumerate(splits) if big in set(g[te])]
            big_fold = big_fold[0] if len(big_fold) == 1 else ("varios" if name == "KFold" else None)
            rows.append(dict(dataset=ds, splitter=name, seed=s, MAE=mae, R2=r2(y, yp),
                             MAE_folds=";".join(f"{v:.4f}" for v in fmae),
                             frac_Tc_gt77_min=min(frac77), frac_Tc_gt77_max=max(frac77),
                             frac_Tc_gt77_rango=max(frac77) - min(frac77),
                             frac_Tc_gt77_folds=";".join(f"{v:.4f}" for v in frac77),
                             n_test_min=min(fold_sizes), n_test_max=max(fold_sizes),
                             familia_mayor_id=big, familia_mayor_elems=big_name, familia_mayor_n=int(fam_sizes.iloc[0]),
                             fold_familia_mayor=big_fold,
                             MAE_fold_familia_mayor=(fmae[big_fold] if isinstance(big_fold, int) else None)))
            print(f"[strat {ds}] s={s} {name}: MAE {mae:.3f} frac77 {min(frac77):.3f}-{max(frac77):.3f} bigfold {big_fold} ({time.time()-t:.0f}s)", flush=True)
    json.dump(rows, open(f"{RAW}/{ds}_strat.json", "w"), indent=1)

# =================================================================== merge
def merge():
    prov_base = (f"{SCRIPT} | modelo XGBoost; config publicada = make_models(seed)['XGBoost'] importada de "
                 f"code/tc_pipeline.py (A) / code/datasetB_pipeline.py (B); grupos = familia quimica "
                 f"(conjunto de elementos) via pandas.factorize; {VERS}; n_jobs={N_JOBS}, tree_method=hist")
    grid = pd.DataFrame(sum((json.load(open(f"{RAW}/{d}_grid.json")) for d in "AB"), []))
    grid["procedencia"] = (prov_base + " | run_grid: rejilla 16 configs (max_depth x lr x min_child_weight, n_estimators=300, "
                           "subsample/colsample de make_models) + config publicada; random = KFold(5, shuffle=True, rs=42); "
                           "grouped = GroupKFold(5) determinista; MAE pooled out-of-fold; seed=42")
    grid.to_csv(f"{OUT}/hp_grid_ambos.csv", index=False)

    fold_rows, summ = [], []
    tab2 = pd.read_csv(f"{OUT}/tabla2_shuffle.csv")
    for d in "AB":
        nest = json.load(open(f"{RAW}/{d}_nested.json"))
        fold_rows += nest["folds"]
        gd = grid[grid.dataset == d].set_index("config")
        wins = pd.Series([f["config_ganadora"] for f in nest["folds"]]).value_counts()
        if wins.iloc[0] > 1 and (wins == wins.iloc[0]).sum() == 1:
            modal, modal_rule = wins.index[0], f"gana {wins.iloc[0]}/5 folds externos"
        else:
            # tie -> median config of the grid by inner-MAE rank across folds
            inner = pd.DataFrame(nest["inner"])
            rk = inner.groupby("outer_fold")["MAE_interna"].rank().groupby(inner["config"]).mean()
            modal, modal_rule = rk.idxmin(), f"empate ({dict(wins)}); se toma la config con mejor rango interno medio"
        pub_row = gd.loc["publicada"]; mod_row = gd.loc[modal]
        mae_nested = nest["MAE_nested_pooled"]
        t2 = tab2[(tab2.dataset == d) & (tab2.modelo == "XGBoost")].iloc[0]
        summ.append(dict(
            dataset=d, n=len(gd) and int({"A": 21263, "B": 12440}[d]),
            MAE_grouped_anidada=mae_nested, R2_grouped_anidada=nest["R2_nested"],
            config_ganadora_modal=modal, regla_modal=modal_rule,
            configs_ganadoras_por_fold=";".join(f["config_ganadora"] for f in nest["folds"]),
            MAE_random_publicada=pub_row.MAE_random, MAE_grouped_publicada_GKF5=pub_row.MAE_grouped,
            inflacion_publicada_misma_config_pct=pub_row.inflacion_pct,
            MAE_random_modal=mod_row.MAE_random, MAE_grouped_modal_GKF5=mod_row.MAE_grouped,
            inflacion_modal_misma_config_pct=mod_row.inflacion_pct,
            inflacion_anidada_vs_random_publicada_pct=100 * (mae_nested / pub_row.MAE_random - 1),
            inflacion_anidada_vs_random_modal_pct=100 * (mae_nested / mod_row.MAE_random - 1),
            comparacion_limpia="inflacion_anidada_vs_random_modal_pct (mismos hiperparametros en ambos brazos: "
                               "solo cambia el protocolo de particion)",
            grid_MAE_grouped_min=gd.drop("publicada").MAE_grouped.min(),
            grid_config_MAE_grouped_min=gd.drop("publicada").MAE_grouped.idxmin(),
            grid_inflacion_min_pct=gd.drop("publicada").inflacion_pct.min(),
            grid_inflacion_max_pct=gd.drop("publicada").inflacion_pct.max(),
            grid_inflacion_mediana_pct=gd.drop("publicada").inflacion_pct.median(),
            tabla2_inflacion_media_pct=t2.inflacion_pct_mean, tabla2_inflacion_std_pct=t2.inflacion_pct_std,
            tabla2_inflacion_min_pct=t2.inflacion_pct_min, tabla2_inflacion_max_pct=t2.inflacion_pct_max,
            procedencia=prov_base + " | run_nested: externo GroupKFold(5) determinista, interno GroupKFold(3) sobre el "
                        "train externo, seleccion por MAE interna pooled sobre 16 configs; MAE_grouped_anidada = pooled "
                        "out-of-fold externo; MAE_random_* de hp_grid_ambos.csv (KFold(5, shuffle=True, rs=42)); "
                        "tabla2_* de tabla2_shuffle.csv (5 semillas, GroupKFold shuffle=True); seed=42"))
    fd = pd.DataFrame(fold_rows); fd["procedencia"] = summ[0]["procedencia"]
    fd.to_csv(f"{OUT}/nested_cv_folds.csv", index=False)
    pd.DataFrame(summ).to_csv(f"{OUT}/nested_cv_inflacion.csv", index=False)
    pd.DataFrame(sum((json.load(open(f"{RAW}/{d}_nested.json"))["inner"] for d in "AB"), [])) \
        .assign(procedencia=summ[0]["procedencia"]).to_csv(f"{OUT}/nested_cv_inner_grid.csv", index=False)

    st = pd.DataFrame(sum((json.load(open(f"{RAW}/{d}_strat.json")) for d in "AB"), []))
    kf = st[st.splitter == "KFold"].set_index(["dataset", "seed"]).MAE
    st["MAE_KFold_misma_semilla"] = [kf.loc[(d, s)] for d, s in zip(st.dataset, st.seed)]
    st["inflacion_pct"] = 100 * (st.MAE / st.MAE_KFold_misma_semilla - 1)
    st["procedencia"] = (prov_base + " | run_strat: config publicada make_models(s); StratifiedGroupKFold(5, shuffle=True, "
                         "random_state=s) estratificando por 5 bins de cuantiles de Tc (pd.qcut) vs GroupKFold(5, shuffle=True, "
                         "random_state=s) vs KFold(5, shuffle=True, random_state=s), s in {0,1,2}; MAE pooled out-of-fold; "
                         "inflacion respecto al KFold de la misma semilla; familia mayor = grupo con mas filas")
    st.to_csv(f"{OUT}/stratified_group_check.csv", index=False)
    print(pd.DataFrame(summ).T.to_string()); print(st[["dataset", "splitter", "seed", "MAE", "inflacion_pct",
          "frac_Tc_gt77_min", "frac_Tc_gt77_max", "fold_familia_mayor", "MAE_fold_familia_mayor"]].to_string())

if __name__ == "__main__":
    if sys.argv[1] == "merge": merge()
    else:
        ds, mode = sys.argv[1], sys.argv[2]
        t0 = time.time(); {"grid": run_grid, "nested": run_nested, "strat": run_strat}[mode](ds)
        print(f"done {ds} {mode} in {time.time()-t0:.0f}s")
