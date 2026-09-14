"""H08 - Table 3 rebuilt in two correctly labelled blocks, XGBoost, Datasets A and B v3.

Block (a) "fixed configuration in BOTH arms": the published configuration (make_models(42)['XGBoost'])
    and the modal configuration of the v2 nested search (results_v2/nested_cv_inflacion.csv), each run
    under random KFold(5, rs=42) and deterministic GroupKFold(5). Only the partition protocol changes.
Block (b) "nested procedure in BOTH arms": outer fold -> inner 3-fold search over the 16-point grid of
    code/v2/run_M4_m2.py (max_depth x learning_rate x min_child_weight, n_estimators=300,
    subsample/colsample of the published config), winner refit on the outer train.
      grouped arm: outer GroupKFold(5), inner GroupKFold(3)      (v2 procedure, re-run on B v3; A re-run and
                   compared with results_v2/_m4_raw/A_nested.json / nested_cv_inflacion.csv)
      random  arm: outer KFold(5, shuffle=True, rs=42), inner KFold(3, shuffle=True, rs=42) on the outer train.
                   Sensitivity: inner KFold(3) WITHOUT shuffle (contiguous blocks; in A the rows of train.csv are
                   ordered so that contiguous blocks behave quasi-grouped: inner MAE ~8.7 K vs outer 5.0 K).
    inflation_nested = MAE(nested grouped) / MAE(nested random) - 1.
For reference only, the MIXED comparison published in v2 (nested grouped vs fixed modal random) is also
listed and labelled as mixed.
Intervals: paired bootstrap of whole families (2000, seed 0) on the OOF predictions, conditional on the
fitted predictions (code/v3/h07_common.py).

Outputs: results_v3/oof_tabla3_v3.csv.gz (all OOF predictions incl. winning config per fold),
         results_v3/tabla3_v3_folds.csv (winners and per-fold MAE), results_v3/tabla3_v3.csv, tabla3_v3.md.

    python code/v3/nested_both_arms.py [--datasets A B] [--skip-nested]
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBRegressor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h07_common import (ROOT, RES_V2, RES_V3, SEED, N_JOBS, VERS, load_dataset, check_splits, metrics, with_n_jobs,
                        load_mod, FamilyStats, paired_family_bootstrap, ci)

PROC = (f"code/v3/nested_both_arms.py | {VERS} | XGBoost tree_method=hist n_jobs={N_JOBS} seed {SEED} | grid = 16 configs of "
        f"code/v2/run_M4_m2.py (max_depth {{4,6,8,10}} x lr {{0.05,0.1}} x min_child_weight {{1,10}}, n_estimators=300, "
        f"subsample/colsample_bytree of the published config) | grouped nested: outer GroupKFold(5) det., inner GroupKFold(3) on "
        f"outer train, winner = min pooled inner MAE | random nested: outer KFold(5, shuffle, rs=42), inner KFold(3, shuffle=True, rs=42); "
        f"sensitivity nested_random_noshuffle: inner KFold(3) without shuffle | "
        f"fixed: published = make_models(42)['XGBoost']; modal = modal winner of the v2 grouped nested search | family bootstrap "
        f"paired, 2000 resamples, default_rng(0), conditional on OOF predictions | B = data/datasetB_featurized_v3.npz (v3)")


def grid_configs(pub):
    cfgs = []
    for md, lr, mcw in itertools.product([4, 6, 8, 10], [0.05, 0.1], [1, 10]):
        cfgs.append(dict(max_depth=md, learning_rate=lr, min_child_weight=mcw, n_estimators=300,
                         subsample=pub["subsample"], colsample_bytree=pub["colsample_bytree"]))
    return cfgs


def cfg_label(c): return f"d{c['max_depth']},lr{c['learning_rate']},mcw{c['min_child_weight']}"


def make_xgb(cfg): return XGBRegressor(**cfg, n_jobs=N_JOBS, random_state=SEED, tree_method="hist")


def oof_fixed(model_fn, X, y, splits):
    yp = np.empty(len(y)); fold = np.empty(len(y), int)
    for k, (tr, te) in enumerate(splits):
        m = model_fn(); m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te]); fold[te] = k
    return yp, fold


def nested(X, y, g, cfgs, labels, outer, inner_fn, tag, ds):
    yp = np.empty(len(y)); fold = np.empty(len(y), int); win = np.empty(len(y), dtype=object); folds = []
    for k, (tr, te) in enumerate(outer):
        t = time.time(); inner = inner_fn(tr)
        inner_mae = np.zeros(len(cfgs))
        for j, c in enumerate(cfgs):
            ypi, _ = oof_fixed(lambda: make_xgb(c), X[tr], y[tr], inner)
            inner_mae[j] = np.mean(np.abs(y[tr] - ypi))
        jb = int(np.argmin(inner_mae))
        m = make_xgb(cfgs[jb]); m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te]); fold[te] = k; win[te] = labels[jb]
        folds.append(dict(dataset=ds, arm=tag, outer_fold=k, n_test=len(te), n_familias_test=len(set(g[te])),
                          config_ganadora=labels[jb], MAE_interna_ganadora=float(inner_mae[jb]),
                          MAE_externa=float(np.mean(np.abs(y[te] - yp[te]))),
                          ranking_interno=";".join(f"{labels[i]}={inner_mae[i]:.3f}" for i in np.argsort(inner_mae)[:5])))
        print(f"  [{ds} {tag}] outer {k}: best {labels[jb]} inner {inner_mae[jb]:.3f} outer {folds[-1]['MAE_externa']:.3f} ({time.time()-t:.0f}s)", flush=True)
    return yp, fold, win, folds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["A", "B"])
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--skip-nested", action="store_true", help="only block (a)")
    a = ap.parse_args(); RES_V3.mkdir(parents=True, exist_ok=True)
    m4 = load_mod("run_M4_m2", str(ROOT / "code" / "v2" / "run_M4_m2.py"))          # grid identity check only
    v2 = pd.read_csv(RES_V2 / "nested_cv_inflacion.csv").set_index("dataset")
    v2grid = pd.read_csv(RES_V2 / "hp_grid_ambos.csv")
    v2folds = pd.read_csv(RES_V2 / "nested_cv_folds.csv")

    oof_frames, fold_rows, tab_rows = [], [], []
    for ds in a.datasets:
        X, y, g, make_models, meta = load_dataset(ds); check_splits(ds, X, y, g, meta["splits"])
        pub = make_models(SEED)["XGBoost"].get_params()
        cfgs = grid_configs(pub); labels = [cfg_label(c) for c in cfgs]
        assert [cfg_label(c) for c in m4.grid_configs(pub)] == labels and m4.grid_configs(pub) == cfgs, "grid differs from run_M4_m2"
        rand = list(KFold(5, shuffle=True, random_state=SEED).split(X)); grp = list(GroupKFold(5).split(X, y, g))
        modal_v2 = v2.loc[ds, "config_ganadora_modal"]
        cfg_modal = cfgs[labels.index(modal_v2)]
        preds = {}
        # ---------------- block (a): fixed configs in both arms
        for name, fn, cfgtxt in (("publicada", lambda: with_n_jobs(make_models(SEED)["XGBoost"]), json.dumps({k: pub[k] for k in ("max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "gamma", "reg_alpha")}, default=str)),
                                 (f"modal_v2({modal_v2})", lambda: make_xgb(cfg_modal), json.dumps(cfg_modal))):
            for proto, splits in (("random", rand), ("grouped", grp)):
                t = time.time(); yp, fold = oof_fixed(fn, X, y, splits); preds[f"fixed_{name}_{proto}"] = yp
                oof_frames.append(pd.DataFrame(dict(dataset=ds, block="a_fixed", arm=name, protocol=proto, row_index=np.arange(len(y)), family_id=g,
                                                    fold=fold, y_true=y, y_pred=yp, config=cfgtxt)))
                mm = metrics(y, yp)
                print(f"  [{ds} fixed {name} {proto}] MAE={mm['MAE']:.4f} R2={mm['R2']:.4f} ({time.time()-t:.0f}s)", flush=True)
        # ---------------- block (b): nested in both arms
        if not a.skip_nested:
            ypg, fg, wg, rows_g = nested(X, y, g, cfgs, labels, grp, lambda tr: list(GroupKFold(3).split(X[tr], y[tr], g[tr])), "nested_grouped", ds)
            ypr, fr, wr, rows_r = nested(X, y, g, cfgs, labels, rand, lambda tr: list(KFold(3, shuffle=True, random_state=SEED).split(X[tr])), "nested_random", ds)
            ypn, fn_, wn, rows_n = nested(X, y, g, cfgs, labels, rand, lambda tr: list(KFold(3).split(X[tr])), "nested_random_noshuffle", ds)
            preds["nested_grouped"] = ypg; preds["nested_random"] = ypr; preds["nested_random_noshuffle"] = ypn; fold_rows += rows_g + rows_r + rows_n
            oof_frames.append(pd.DataFrame(dict(dataset=ds, block="b_nested", arm="nested", protocol="grouped", row_index=np.arange(len(y)), family_id=g, fold=fg, y_true=y, y_pred=ypg, config=wg)))
            oof_frames.append(pd.DataFrame(dict(dataset=ds, block="b_nested", arm="nested", protocol="random", row_index=np.arange(len(y)), family_id=g, fold=fr, y_true=y, y_pred=ypr, config=wr)))
            oof_frames.append(pd.DataFrame(dict(dataset=ds, block="b_nested_sens", arm="nested_inner_noshuffle", protocol="random", row_index=np.arange(len(y)), family_id=g, fold=fn_, y_true=y, y_pred=ypn, config=wn)))
        # ---------------- paired family bootstrap of every contrast
        fs = FamilyStats(y, g, preds); one = np.ones(fs.k)
        B = paired_family_bootstrap(fs, preds.keys(), a.n_boot, a.boot_seed)
        contrasts = [("a_fixed", "publicada: misma config en ambos brazos", f"fixed_publicada_random", f"fixed_publicada_grouped"),
                     ("a_fixed", f"modal v2 ({modal_v2}): misma config en ambos brazos", f"fixed_modal_v2({modal_v2})_random", f"fixed_modal_v2({modal_v2})_grouped")]
        if not a.skip_nested:
            contrasts += [("b_nested", "anidado en ambos brazos", "nested_random", "nested_grouped"),
                          ("b_nested_sens", "anidado en ambos brazos, interno del brazo aleatorio SIN barajar (sensibilidad)", "nested_random_noshuffle", "nested_grouped"),
                          ("mixto_v2", f"MIXTO (v2): anidado agrupado / modal fija aleatoria", f"fixed_modal_v2({modal_v2})_random", "nested_grouped"),
                          ("mixto_v2", f"MIXTO (v2): anidado agrupado / publicada aleatoria", "fixed_publicada_random", "nested_grouped")]
        for block, label, cr, cg in contrasts:
            Pr, Pg = fs.stats(one, cr), fs.stats(one, cg)
            infl = 100 * (B[f"MAE_{cg}"] / B[f"MAE_{cr}"] - 1); dr2 = B[f"R2_{cr}"] - B[f"R2_{cg}"]
            infl_macro = 100 * (B[f"MAE_macro_{cg}"] / B[f"MAE_macro_{cr}"] - 1)
            row = dict(dataset=ds, N=len(y), k=fs.k, bloque=block, comparacion=label, brazo_random=cr, brazo_grouped=cg,
                       MAE_random=Pr["MAE"], MAE_grouped=Pg["MAE"], R2_random=Pr["R2"], R2_grouped=Pg["R2"],
                       inflacion_pct=100 * (Pg["MAE"] / Pr["MAE"] - 1), inflacion_lo=ci(infl)[0], inflacion_hi=ci(infl)[1],
                       dR2=Pr["R2"] - Pg["R2"], dR2_lo=ci(dr2)[0], dR2_hi=ci(dr2)[1],
                       MAE_macro_random=Pr["MAE_macro"], MAE_macro_grouped=Pg["MAE_macro"],
                       inflacion_macro_pct=100 * (Pg["MAE_macro"] / Pr["MAE_macro"] - 1), inflacion_macro_lo=ci(infl_macro)[0], inflacion_macro_hi=ci(infl_macro)[1])
            if not a.skip_nested:
                row["configs_ganadoras_grouped"] = ";".join(r["config_ganadora"] for r in rows_g)
                row["configs_ganadoras_random"] = ";".join(r["config_ganadora"] for r in (rows_n if "SIN barajar" in label else rows_r))
                row["MAE_interna_media_random"] = float(np.mean([r["MAE_interna_ganadora"] for r in (rows_n if "SIN barajar" in label else rows_r)]))
                row["MAE_interna_media_grouped"] = float(np.mean([r["MAE_interna_ganadora"] for r in rows_g]))
            # v2 references
            if block == "a_fixed":
                key = "publicada" if "publicada" in label else modal_v2
                r2 = v2grid[(v2grid.dataset == ds) & (v2grid.config == key)].iloc[0]
                row.update(v2_MAE_random=r2.MAE_random, v2_MAE_grouped=r2.MAE_grouped, v2_inflacion_pct=r2.inflacion_pct)
            elif block in ("b_nested", "b_nested_sens"):
                row.update(v2_MAE_grouped_anidada=v2.loc[ds, "MAE_grouped_anidada"], v2_configs_ganadoras=v2.loc[ds, "configs_ganadoras_por_fold"])
            else:
                row.update(v2_inflacion_pct=v2.loc[ds, "inflacion_anidada_vs_random_modal_pct" if "modal" in label else "inflacion_anidada_vs_random_publicada_pct"])
            row["v2_checkpoint_B"] = "v2 (N=12440)" if ds == "B" else "-"
            tab_rows.append(row)
            print(f"  [{ds}] {label}: MAE_r={Pr['MAE']:.4f} MAE_g={Pg['MAE']:.4f} infl={row['inflacion_pct']:.2f}% [{row['inflacion_lo']:.2f},{row['inflacion_hi']:.2f}]", flush=True)

    O = pd.concat(oof_frames, ignore_index=True); O["procedencia"] = PROC
    O.to_csv(RES_V3 / "oof_tabla3_v3.csv.gz", index=False, compression="gzip")
    F = pd.DataFrame(fold_rows)
    if len(F):
        # A: compare with v2 fold winners / MAE (A data and folds unchanged)
        vA = v2folds[v2folds.dataset == "A"].set_index("outer_fold")
        for i, r in F.iterrows():
            if r.dataset == "A" and r.arm == "nested_grouped":
                F.loc[i, "v2_config_ganadora"] = vA.loc[r.outer_fold, "config_ganadora"]; F.loc[i, "v2_MAE_externa"] = vA.loc[r.outer_fold, "MAE_externa_ganadora"]
        F["procedencia"] = PROC; F.to_csv(RES_V3 / "tabla3_v3_folds.csv", index=False)
    T = pd.DataFrame(tab_rows); T["procedencia"] = PROC + " | v2_* = results_v2/hp_grid_ambos.csv, nested_cv_inflacion.csv (B on the v2 checkpoint)"
    T.to_csv(RES_V3 / "tabla3_v3.csv", index=False)

    L = ["# Tabla 3 (v3) - inflación del MAE bajo configuración fija y bajo procedimiento anidado (XGBoost)", "",
         f"Procedencia: `{PROC}`", "",
         "Intervalos: bootstrap pareado de familias completas (2000, semilla 0), percentil 95 %, condicionales a las predicciones OOF. "
         "Inflación = MAE_grouped / MAE_random − 1. Dataset B = checkpoint v3 (N = 12 336, k = 3 046); los valores v2 de referencia son del checkpoint v2 (N = 12 440).", "",
         "## (a) Configuración fija en ambos brazos (solo cambia el protocolo de partición)", "",
         "| Dataset | Configuración | MAE random | MAE grouped | Inflación % [IC fam.] | ΔR² [IC] | Inflación macro % [IC] | v2 (fija) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in T[T.bloque == "a_fixed"].iterrows():
        L.append(f"| {r.dataset} | {r.comparacion} | {r.MAE_random:.3f} | {r.MAE_grouped:.3f} | {r.inflacion_pct:.1f} [{r.inflacion_lo:.1f}, {r.inflacion_hi:.1f}] | {r.dR2:.3f} [{r.dR2_lo:.3f}, {r.dR2_hi:.3f}] | {r.inflacion_macro_pct:.1f} [{r.inflacion_macro_lo:.1f}, {r.inflacion_macro_hi:.1f}] | {r.v2_inflacion_pct:.1f} |")
    if not a.skip_nested:
        L += ["", "## (b) Procedimiento anidado en AMBOS brazos (selección interna de 16 configuraciones dentro del train externo)", "",
              "| Dataset | MAE anidado random | MAE anidado grouped | Inflación % [IC fam.] | ΔR² [IC] | Inflación macro % [IC] | ganadoras grouped | ganadoras random | v2 MAE anidado grouped |", "|---|---:|---:|---:|---:|---:|---|---|---:|"]
        for _, r in T[T.bloque.isin(["b_nested", "b_nested_sens"])].iterrows():
            L.append(f"| {r.dataset}{' (sens.: interno sin barajar)' if r.bloque == 'b_nested_sens' else ''} | {r.MAE_random:.3f} | {r.MAE_grouped:.3f} | {r.inflacion_pct:.1f} [{r.inflacion_lo:.1f}, {r.inflacion_hi:.1f}] | {r.dR2:.3f} [{r.dR2_lo:.3f}, {r.dR2_hi:.3f}] | {r.inflacion_macro_pct:.1f} [{r.inflacion_macro_lo:.1f}, {r.inflacion_macro_hi:.1f}] | {r.configs_ganadoras_grouped} | {r.configs_ganadoras_random} | {r.v2_MAE_grouped_anidada:.3f} |")
        L += ["", "## Referencia: la comparación MIXTA publicada en v2 (no es 'mismos hiperparámetros')", "",
              "| Dataset | Comparación | MAE random (fija) | MAE grouped (anidado) | Inflación % [IC fam.] | v2 |", "|---|---|---:|---:|---:|---:|"]
        for _, r in T[T.bloque == "mixto_v2"].iterrows():
            L.append(f"| {r.dataset} | {r.comparacion} | {r.MAE_random:.3f} | {r.MAE_grouped:.3f} | {r.inflacion_pct:.1f} [{r.inflacion_lo:.1f}, {r.inflacion_hi:.1f}] | {r.v2_inflacion_pct:.1f} |")
    (RES_V3 / "tabla3_v3.md").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {RES_V3 / 'tabla3_v3.csv'}, tabla3_v3.md, tabla3_v3_folds.csv, oof_tabla3_v3.csv.gz")


if __name__ == "__main__":
    main()
