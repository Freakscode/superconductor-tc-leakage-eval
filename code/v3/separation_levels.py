"""Separation levels with a comparable budget (H09, experiment 3) + nearest-neighbour diagnostic with a
scaler fitted on the TRAINING fold only.

Same XGBoost (make_models(42)["XGBoost"] of each pipeline, paper descriptors R2, n_jobs=8), 5 folds,
seed 42, for every level of train/test separation:
    L0  random rows                       KFold(5, shuffle=True, random_state=42)
    L1  normalised composition            group = sorted (element, round(fraction, 3)); GroupKFold(5)
    L2  element system (the paper)        group = constituent-element set; GroupKFold(5) == splits/*fold_deterministic
    L3  host                              group = set of elements with molar fraction >= 0.05 (dopants below the
                                          threshold ignored); sensitivity 0.02 and 0.10; GroupKFold(5)
    L4  LOCO by cluster                   KMeans(k=50, random_state=0, n_init=10) on R1 = [presence, fractions]
                                          standardised over the whole dataset (unsupervised, no labels); GroupKFold(5)
Per level and fold: number of groups, test size, share of test rows with an EXACT neighbour in train (identical
R2 feature vector; identical normalised composition; same element system; same host), median 1-NN Euclidean
distance in R2 space standardised with a StandardScaler fitted on the TRAIN fold only (and, for comparison,
fitted on all rows as in code/v2/run_diagnostics_v2.py), MAE/RMSE/R2 pooled and top-100 per fold (threshold
p90 of train, >=). Paired bootstrap of whole element-system families (2 000, default_rng(0)) for the
differences vs L0, conditional on the fitted OOF predictions.

nn_distance_trainscaler.csv repeats figure 1(d)/3(d) (80/20 random hold-out with RandomState(42).shuffle vs
GroupShuffleSplit(test_size=0.2, random_state=42)) with the scaler fitted on train only next to the global
scaler, for A, B v3 and (for comparison with results_v2/nn_distance.csv) B v2.

Outputs (results_v3/): separation_levels.csv, separation_levels_folds.csv, separation_levels.md,
nn_distance_trainscaler.csv.
    python code/v3/separation_levels.py [--datasets A B]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h09h10_common as C
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def keys_composition(d, ndig=3):
    return pd.factorize(np.array(["|".join(f"{e}:{round(x, ndig):.3f}" for e, x in zip(d.elements, row) if x > 0) for row in d.F]))[0]


def keys_host(d, thr):
    return pd.factorize(np.array(["|".join(e for e, x in zip(d.elements, row) if x >= thr) for row in d.F]))[0]


def keys_cluster(d, k=50, seed=0):
    Z = StandardScaler().fit_transform(d.representation("R1"))
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)


def nn_median(X, tr, te, scaler_on):
    sc = StandardScaler().fit(X[tr] if scaler_on == "train" else X)
    A, B = sc.transform(X[tr]), sc.transform(X[te])
    dist = NearestNeighbors(n_neighbors=1).fit(A).kneighbors(B)[0].ravel()
    return dist


def exact_frac(keys, tr, te):
    return float(np.isin(keys[te], keys[tr]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["A", "B"])
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--out", type=Path, default=C.RES_V3)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    proc = (f"code/v3/separation_levels.py | {C.versions()} | XGBoost = make_models({C.SEED})['XGBoost'] of tc_pipeline (A) / datasetB_pipeline (B), paper descriptors, n_jobs={C.N_JOBS} | "
            f"L0 KFold(5, shuffle, rs={C.SEED}); L1 GroupKFold(5) on normalised composition (fractions rounded 1e-3); L2 GroupKFold(5) on element set (== splits fold_deterministic); "
            f"L3 GroupKFold(5) on host = elements with fraction >= thr (0.05; sensitivity 0.02, 0.10); L4 GroupKFold(5) on KMeans(50, random_state=0, n_init=10) clusters of standardised R1 | "
            f"NN: 1-NN euclidean in R2 standardised with StandardScaler fitted on the TRAIN fold (nn_*_trainscaler) or on all rows (nn_*_globalscaler, as run_diagnostics_v2.py) | "
            f"exact_*: share of test rows whose key (R2 vector bytes / composition 1e-3 / element set / host 0.05) occurs in train | top100: K=100, p90(train), >= | "
            f"intervals vs L0: paired bootstrap of whole element-system families (n_boot={args.n_boot}, default_rng({C.BOOT_SEED})), conditional on OOF predictions")
    rows, frows, nnrows, md = [], [], [], ["# Niveles de separación (H09, exp. 3)\n", f"`{proc}`\n"]
    for ds in args.datasets:
        d = C.load(ds); T0 = time.time()
        keys = {"R2_exact": pd.factorize(np.array([r.tobytes() for r in np.ascontiguousarray(d.X2)], dtype=object))[0],
                "composition_1e-3": keys_composition(d), "element_system": d.groups, "host_0.05": keys_host(d, 0.05)}
        t = time.time(); clusters = keys_cluster(d); print(f"[{ds}] kmeans {time.time()-t:.0f}s", flush=True)
        levels = {"L0_aleatorio_filas": (C.random_folds(d), np.arange(d.n)),
                  "L1_composicion_1e-3": (list(GroupKFold(5).split(d.X2, d.y, keys["composition_1e-3"])), keys["composition_1e-3"]),
                  "L2_sistema_elemental": (C.grouped_folds(d), d.groups),
                  "L3_anfitrion_0.05": (list(GroupKFold(5).split(d.X2, d.y, keys["host_0.05"])), keys["host_0.05"]),
                  "L3_anfitrion_0.02": (list(GroupKFold(5).split(d.X2, d.y, keys_host(d, 0.02))), keys_host(d, 0.02)),
                  "L3_anfitrion_0.10": (list(GroupKFold(5).split(d.X2, d.y, keys_host(d, 0.10))), keys_host(d, 0.10)),
                  "L4_LOCO_kmeans50": (list(GroupKFold(5).split(d.X2, d.y, clusters)), clusters)}
        preds, tops, info = {}, {}, {}
        for lvl, (sp, gk) in levels.items():
            yp, tp, sec = C.oof(lambda: d.xgb(), d.X2, d.y, sp); preds[lvl], tops[lvl] = yp, tp
            dtr, dgl, ex = [], [], {k: [] for k in keys}
            for f, (tr, te) in enumerate(sp):
                a, b = nn_median(d.X2, tr, te, "train"), nn_median(d.X2, tr, te, "global"); dtr.append(a); dgl.append(b)
                fr = {k: exact_frac(v, tr, te) for k, v in keys.items()}
                for k in keys: ex[k].append(fr[k])
                m = C.metrics(d.y[te], yp[te])
                frows.append({"dataset": ds, "level": lvl, "fold": f, "n_groups_total": len(np.unique(gk)), "n_groups_test": len(np.unique(gk[te])),
                              "n_train": len(tr), "n_test": len(te), **{f"exact_{k}": fr[k] for k in keys},
                              "nn_median_trainscaler": float(np.median(a)), "nn_median_globalscaler": float(np.median(b)),
                              "nn_p25_trainscaler": float(np.percentile(a, 25)), "nn_p75_trainscaler": float(np.percentile(a, 75)),
                              **{k: m[k] for k in ("MAE", "RMSE", "R2")}, "top100": tp[f], "procedencia": proc})
            dtr, dgl = np.concatenate(dtr), np.concatenate(dgl); m = C.metrics(d.y, yp)
            info[lvl] = {"dataset": ds, "level": lvl, "n_groups": len(np.unique(gk)), "n_test_mean": float(np.mean([len(te) for _, te in sp])),
                         "n_test_min": min(len(te) for _, te in sp), "n_test_max": max(len(te) for _, te in sp),
                         **{f"exact_{k}": float(np.mean(ex[k])) for k in keys},
                         "nn_median_trainscaler": float(np.median(dtr)), "nn_p25_trainscaler": float(np.percentile(dtr, 25)), "nn_p75_trainscaler": float(np.percentile(dtr, 75)),
                         "nn_median_globalscaler": float(np.median(dgl)), "nn_median_rel_diff_train_vs_global": float(np.median(dtr) / np.median(dgl) - 1),
                         **{k: m[k] for k in ("MAE", "RMSE", "R2", "SSE")}, "top100_mean": float(np.mean(tp)), "top100_sd": float(np.std(tp)),
                         "top100_folds": ";".join(f"{x:.2f}" for x in tp), "fit_seconds": round(sec, 1)}
            print(f"  {ds} {lvl:22s} groups={info[lvl]['n_groups']:6d} test~{info[lvl]['n_test_mean']:.0f} exactR2={info[lvl]['exact_R2_exact']:.3f} "
                  f"nn_tr={info[lvl]['nn_median_trainscaler']:.3f} nn_gl={info[lvl]['nn_median_globalscaler']:.3f} MAE={m['MAE']:.3f} R2={m['R2']:.4f} top100={np.mean(tp):.3f} ({sec:.0f}s)", flush=True)
        fam, pairs = C.family_bootstrap(d.y, preds, d.groups, args.n_boot, C.BOOT_SEED)
        for lvl in levels:
            r = dict(info[lvl]); r.update({f"{k}_fam_{s}": fam[lvl][f"{k}_{s}"] for k in ("MAE", "R2") for s in ("lo", "hi")})
            if lvl != "L0_aleatorio_filas":
                p = pairs[("L0_aleatorio_filas", lvl)]; m0 = info["L0_aleatorio_filas"]
                r.update(dMAE_vs_L0=r["MAE"] - m0["MAE"], dMAE_vs_L0_fam_lo=p["dMAE_lo"], dMAE_vs_L0_fam_hi=p["dMAE_hi"],
                         dR2_vs_L0=r["R2"] - m0["R2"], dR2_vs_L0_fam_lo=p["dR2_lo"], dR2_vs_L0_fam_hi=p["dR2_hi"],
                         SSE_change_vs_L0=r["SSE"] / m0["SSE"] - 1, SSE_change_vs_L0_fam_lo=-p["SSE_red_hi"], SSE_change_vs_L0_fam_hi=-p["SSE_red_lo"],
                         dtop100_vs_L0=r["top100_mean"] - m0["top100_mean"])
            r["n_boot"], r["boot_seed"], r["procedencia"] = args.n_boot, C.BOOT_SEED, proc; rows.append(r)
        md.append(f"\n## Dataset {ds} (n={d.n}, k={d.k} sistemas elementales)\n")
        md.append("| nivel | grupos | test medio | test con vector R2 idéntico en train | misma composición 1e-3 | mismo sistema | mismo anfitrión 0.05 | mediana d(1-NN) scaler-train | scaler-global | MAE [IC fam] | R² [IC fam] | top-100 | ΔMAE vs L0 [IC fam] | ΔR² vs L0 [IC fam] |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows[-len(levels):]:
            dm = f"{r['dMAE_vs_L0']:+.3f} [{r['dMAE_vs_L0_fam_lo']:+.3f}, {r['dMAE_vs_L0_fam_hi']:+.3f}]" if "dMAE_vs_L0" in r else "—"
            dr = f"{r['dR2_vs_L0']:+.4f} [{r['dR2_vs_L0_fam_lo']:+.4f}, {r['dR2_vs_L0_fam_hi']:+.4f}]" if "dR2_vs_L0" in r else "—"
            md.append(f"| {r['level']} | {r['n_groups']} | {r['n_test_mean']:.0f} ({r['n_test_min']}–{r['n_test_max']}) | {100*r['exact_R2_exact']:.1f} % | {100*r['exact_composition_1e-3']:.1f} % | "
                      f"{100*r['exact_element_system']:.1f} % | {100*r['exact_host_0.05']:.1f} % | {r['nn_median_trainscaler']:.3f} | {r['nn_median_globalscaler']:.3f} | "
                      f"{r['MAE']:.3f} [{r['MAE_fam_lo']:.3f}, {r['MAE_fam_hi']:.3f}] | {r['R2']:.4f} [{r['R2_fam_lo']:.4f}, {r['R2_fam_hi']:.4f}] | {r['top100_mean']:.3f}±{r['top100_sd']:.3f} | {dm} | {dr} |")
        # --- figure 1(d)/3(d) hold-out diagnostic with train-only scaler
        for tag, X, g in ([(ds, d.X2, d.groups)] + ([("B_v2_npz_comparacion", *(lambda z: (z["X"].astype(float), z["groups"]))(np.load(C.DATA / "datasetB_featurized.npz")))] if ds == "B" else [])):
            ri = np.arange(len(X)); np.random.RandomState(C.SEED).shuffle(ri); cut = int(0.8 * len(X))
            g_tr, g_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=C.SEED).split(X, np.zeros(len(X)), g))
            row = {"dataset": tag, "n_test_random": len(X) - cut, "n_test_grouped": len(g_te)}
            for sc in ("global", "train"):
                dr_, dg_ = nn_median(X, ri[:cut], ri[cut:], sc), nn_median(X, g_tr, g_te, sc)
                row.update({f"nn_median_random_{sc}scaler": float(np.median(dr_)), f"nn_median_grouped_{sc}scaler": float(np.median(dg_)),
                            f"ratio_{sc}scaler": float(np.median(dg_) / np.median(dr_)),
                            f"nn_p25_random_{sc}scaler": float(np.percentile(dr_, 25)), f"nn_p75_random_{sc}scaler": float(np.percentile(dr_, 75)),
                            f"nn_p25_grouped_{sc}scaler": float(np.percentile(dg_, 25)), f"nn_p75_grouped_{sc}scaler": float(np.percentile(dg_, 75)),
                            f"frac_zero_random_{sc}scaler": float((dr_ == 0).mean()), f"frac_zero_grouped_{sc}scaler": float((dg_ == 0).mean())})
            row["rel_diff_grouped_median_train_vs_global"] = row["nn_median_grouped_trainscaler"] / row["nn_median_grouped_globalscaler"] - 1
            row["rel_diff_random_median_train_vs_global"] = row["nn_median_random_trainscaler"] / row["nn_median_random_globalscaler"] - 1
            row["procedencia"] = proc; nnrows.append(row)
            print(f"  NN hold-out {tag}: random {row['nn_median_random_globalscaler']:.4f}->{row['nn_median_random_trainscaler']:.4f} | grouped {row['nn_median_grouped_globalscaler']:.4f}->{row['nn_median_grouped_trainscaler']:.4f} | ratio {row['ratio_globalscaler']:.2f}->{row['ratio_trainscaler']:.2f}", flush=True)
        print(f"[{ds}] total {time.time()-T0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(args.out / "separation_levels.csv", index=False)
    pd.DataFrame(frows).to_csv(args.out / "separation_levels_folds.csv", index=False)
    pd.DataFrame(nnrows).to_csv(args.out / "nn_distance_trainscaler.csv", index=False)
    md.append("\n## Diagnóstico de vecinos (Fig. 1(d)/3(d)) con scaler ajustado solo en train\n")
    md.append("| dataset | mediana random (global → train) | mediana agrupado (global → train) | razón (global → train) | dif. rel. agrupado | dif. rel. random |\n|---|---|---|---|---|---|")
    for r in nnrows:
        md.append(f"| {r['dataset']} | {r['nn_median_random_globalscaler']:.4f} → {r['nn_median_random_trainscaler']:.4f} | {r['nn_median_grouped_globalscaler']:.4f} → {r['nn_median_grouped_trainscaler']:.4f} | "
                  f"{r['ratio_globalscaler']:.1f} → {r['ratio_trainscaler']:.1f} | {100*r['rel_diff_grouped_median_train_vs_global']:+.2f} % | {100*r['rel_diff_random_median_train_vs_global']:+.2f} % |")
    md.append("\n## Notas\n- Cada nivel corresponde a una población objetivo distinta (nuevas medidas de composiciones conocidas → L0/L1; nuevas estequiometrías de sistemas conocidos → L1; sistemas inéditos → L2; anfitriones inéditos → L3; regiones composicionales inéditas → L4); el criterio más estricto no es 'el correcto' por defecto.\n"
              "- L3: anfitrión = elementos con fracción molar ≥ 0.05 (los dopantes por debajo se ignoran); sensibilidad con 0.02 y 0.10. L4: KMeans(50, semilla 0) sobre R1 estandarizado (sin etiquetas).\n"
              "- Intervalos: bootstrap pareado de sistemas elementales completos (2 000, semilla 0), condicional a las predicciones OOF; el emparejamiento con L0 usa las mismas familias remuestreadas en ambos niveles.\n"
              "- Distancias 1-NN en el espacio R2 estandarizado con scaler ajustado SOLO en el fold de entrenamiento; la columna *globalscaler* reproduce el procedimiento de run_diagnostics_v2.py (scaler sobre todo X) para medir la diferencia.\n")
    (args.out / "separation_levels.md").write_text("\n".join(md))
    print(f"wrote {args.out / 'separation_levels.csv'}, _folds.csv, .md, nn_distance_trainscaler.csv")


if __name__ == "__main__":
    main()
