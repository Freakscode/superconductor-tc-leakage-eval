"""Control "remove the test's relatives vs remove the same number of random rows" (H10).

With the TEST FIXED, the same XGBoost (make_models(42)["XGBoost"] of each pipeline, paper descriptors R2,
n_jobs=8) is trained on
    (a) the full training pool, which contains relatives of the test (rows of the same chemical family);
    (b) that pool minus every relative of the test  -> a family-disjoint (grouped) training set;
    (c) that pool minus the SAME NUMBER of rows drawn at random (5 repetitions, default_rng(rep)).
(b) - (c) isolates the effect of the relatives from the effect of the smaller training set.

Two designs, because the literal one removes most of the training pool:
    D1  test = each fold of the random protocol KFold(5, shuffle=True, random_state=42); pool (a) = the other
        four folds (= the random protocol's training set). Relatives are ~78 % (A) / ~66 % (B) of the pool.
    D2  test = half of each GroupKFold(5) fold (within every family of the fold, rows are split in two halves
        with default_rng(0); singleton families go to the test half); pool (a) = the other four folds + the
        other half (the relatives); (b) = the other four folds (= grouped protocol); (c) = (a) minus |relatives|
        random rows. Relatives are ~10 % of the pool, so (c) is a mild size perturbation.
Top-100 (K=100, operator >=) uses the p90 threshold of pool (a) for the three conditions of a fold so that
the three models are scored against the same target set. Pooled metrics over the union of the test sets;
paired bootstrap of whole families (2 000, default_rng(0)) for (b)-(a), (c)-(a), (b)-(c), conditional on
the fitted predictions; for (c) the 5 repetitions are bootstrapped separately and their interval bounds
averaged (declared approximation).

Outputs (results_v3/): remove_relatives_control.csv (per fold x condition x rep),
remove_relatives_control_summary.csv, remove_relatives_control.md.
    python code/v3/remove_relatives_control.py [--datasets A B] [--reps 5]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h09h10_common as C


def fit_eval(d, tr, te, thr):
    m = d.xgb(); t0 = time.time(); m.fit(d.X2[tr], d.y[tr]); yp = m.predict(d.X2[te])
    met = C.metrics(d.y[te], yp)
    top = float((d.y[te][np.argsort(yp)[-100:]] >= thr).mean())
    return yp, met, top, time.time() - t0


def halves(d, te, seed=0):
    """Split the rows of a grouped fold within each family into a test half and a relatives half."""
    rng = np.random.default_rng(seed); T, R = [], []
    for g in np.unique(d.groups[te]):
        idx = te[d.groups[te] == g]; idx = rng.permutation(idx); h = (len(idx) + 1) // 2
        T.extend(idx[:h]); R.extend(idx[h:])
    return np.array(sorted(T)), np.array(sorted(R), dtype=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["A", "B"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--out", type=Path, default=C.RES_V3)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    proc = (f"code/v3/remove_relatives_control.py | {C.versions()} | XGBoost = make_models({C.SEED})['XGBoost'] of tc_pipeline (A) / datasetB_pipeline (B), "
            f"paper descriptors, n_jobs={C.N_JOBS} | D1: test = KFold(5, shuffle, rs={C.SEED}) fold, pool = other folds | D2: test = half of each GroupKFold(5) fold "
            f"(per-family halves, default_rng(0), singletons -> test), pool = other folds + other half | (b) pool minus relatives (same family as a test row); "
            f"(c) pool minus |relatives| random rows, reps default_rng(0..{args.reps-1}) | top100: K=100, threshold p90(pool a), >= | "
            f"intervals: paired bootstrap of whole families (n_boot={args.n_boot}, default_rng({C.BOOT_SEED})), conditional on predictions")
    rows, srows, md = [], [], ["# Control: retirar familiares vs retirar filas al azar (H10)\n", f"`{proc}`\n"]
    for ds in args.datasets:
        d = C.load(ds); T0 = time.time()
        designs = {"D1_test_fold_aleatorio": [(tr, te, None) for tr, te in C.random_folds(d)],
                   "D2_test_mitad_fold_agrupado": [(tr,) + halves(d, te) for tr, te in C.grouped_folds(d)]}
        for design, folds in designs.items():
            preds = {"a": np.full(d.n, np.nan), "b": np.full(d.n, np.nan), **{f"c{r}": np.full(d.n, np.nan) for r in range(args.reps)}}
            test_mask = np.zeros(d.n, bool)
            for f, item in enumerate(folds):
                if design.startswith("D1"):
                    tr, te, _ = item; pool = tr
                    rel = np.isin(d.groups[pool], np.unique(d.groups[te]))
                else:
                    tr, te, R = item; pool = np.sort(np.concatenate([tr, R]))
                    rel = np.isin(pool, R)
                n_rel = int(rel.sum()); thr = float(np.percentile(d.y[pool], 90)); test_mask[te] = True
                conds = {"a_pool_con_familiares": pool, "b_pool_sin_familiares": pool[~rel]}
                for r in range(args.reps):
                    drop = np.random.default_rng(r).choice(len(pool), n_rel, replace=False)
                    conds[f"c{r}_pool_sin_n_filas_al_azar"] = np.delete(pool, drop)
                for cond, trn in conds.items():
                    key = cond.split("_")[0]
                    if key == "b": assert not (set(d.groups[trn]) & set(d.groups[te])), "family leak in (b)"
                    yp, met, top, sec = fit_eval(d, trn, te, thr); preds[key][te] = yp
                    n_rel_left = int(np.isin(d.groups[trn], np.unique(d.groups[te])).sum()) if design.startswith("D1") else int(np.isin(trn, R).sum())
                    rows.append({"dataset": ds, "design": design, "fold": f, "condition": cond, "n_train": len(trn), "n_test": len(te),
                                 "n_removed": len(pool) - len(trn), "frac_removed": (len(pool) - len(trn)) / len(pool), "relatives_left_in_train": n_rel_left,
                                 "thr_p90_pool_a": thr, **met, "top100": top, "fit_seconds": round(sec, 1), "procedencia": proc})
                    print(f"  {ds} {design[:2]} f{f} {cond:28s} n_train={len(trn):6d} MAE={met['MAE']:.3f} R2={met['R2']:.4f} top100={top:.2f} ({sec:.0f}s)", flush=True)
            # pooled over the union of test sets + paired family bootstrap
            te_all = np.where(test_mask)[0]; y = d.y[te_all]; g = d.groups[te_all]
            P = {k: v[te_all] for k, v in preds.items()}; assert not any(np.isnan(v).any() for v in P.values())
            fam, pairs = C.family_bootstrap(y, P, g, args.n_boot, C.BOOT_SEED)
            met = {k: C.metrics(y, v) for k, v in P.items()}
            R = pd.DataFrame([r for r in rows if r["dataset"] == ds and r["design"] == design])
            def cond_summary(key, label):
                sub = R[R.condition.str.startswith(key)]
                return {"dataset": ds, "design": design, "condition": label, "n_test_pooled": len(te_all), "k_families_test": len(np.unique(g)),
                        "n_train_mean": float(sub.n_train.mean()), "frac_removed_mean": float(sub.frac_removed.mean()),
                        **{m: met[key][m] for m in ("MAE", "RMSE", "R2")}, "MAE_fam_lo": fam[key]["MAE_lo"], "MAE_fam_hi": fam[key]["MAE_hi"],
                        "R2_fam_lo": fam[key]["R2_lo"], "R2_fam_hi": fam[key]["R2_hi"], "top100_mean": float(sub.top100.mean()), "top100_sd": float(sub.top100.std(ddof=0)),
                        "procedencia": proc}
            s_a, s_b = cond_summary("a", "a_pool_con_familiares"), cond_summary("b", "b_pool_sin_familiares")
            cs = [cond_summary(f"c{r}", f"c{r}") for r in range(args.reps)]
            s_c = {**cs[0], "condition": "c_pool_sin_n_filas_al_azar_media_reps"}
            for m in ("MAE", "RMSE", "R2", "MAE_fam_lo", "MAE_fam_hi", "R2_fam_lo", "R2_fam_hi", "top100_mean", "n_train_mean"):
                s_c[m] = float(np.mean([c[m] for c in cs]))
            s_c["top100_sd"] = float(np.std([c["top100_mean"] for c in cs]))
            def diff(a, b_keys, label):
                ds_ = [pairs[(a, b)] for b in b_keys]; ma = met[a]; mb = {m: np.mean([met[b][m] for b in b_keys]) for m in ("MAE", "RMSE", "R2")}
                return {"dataset": ds, "design": design, "condition": label, "n_test_pooled": len(te_all), "k_families_test": len(np.unique(g)),
                        "dMAE": mb["MAE"] - ma["MAE"], "dMAE_fam_lo": np.mean([x["dMAE_lo"] for x in ds_]), "dMAE_fam_hi": np.mean([x["dMAE_hi"] for x in ds_]),
                        "dRMSE": mb["RMSE"] - ma["RMSE"], "dR2": mb["R2"] - ma["R2"], "dR2_fam_lo": np.mean([x["dR2_lo"] for x in ds_]), "dR2_fam_hi": np.mean([x["dR2_hi"] for x in ds_]),
                        "SSE_red": float(np.mean([1 - C.metrics(y, P[b])["SSE"] / C.metrics(y, P[a])["SSE"] for b in b_keys])),
                        "SSE_red_fam_lo": np.mean([x["SSE_red_lo"] for x in ds_]), "SSE_red_fam_hi": np.mean([x["SSE_red_hi"] for x in ds_]), "procedencia": proc}
            ck = [f"c{r}" for r in range(args.reps)]
            d_ba, d_ca, d_bc = diff("a", ["b"], "diff_b_minus_a"), diff("a", ck, "diff_c_minus_a_media_reps"), diff("c0", ["b"], "diff_b_minus_c_rep0")
            d_bc_all = [pairs[(c, "b")] for c in ck]
            d_bc.update(condition="diff_b_minus_c_media_reps", dMAE=float(np.mean([met["b"]["MAE"] - met[c]["MAE"] for c in ck])),
                        dMAE_fam_lo=np.mean([x["dMAE_lo"] for x in d_bc_all]), dMAE_fam_hi=np.mean([x["dMAE_hi"] for x in d_bc_all]),
                        dRMSE=float(np.mean([met["b"]["RMSE"] - met[c]["RMSE"] for c in ck])), dR2=float(np.mean([met["b"]["R2"] - met[c]["R2"] for c in ck])),
                        dR2_fam_lo=np.mean([x["dR2_lo"] for x in d_bc_all]), dR2_fam_hi=np.mean([x["dR2_hi"] for x in d_bc_all]),
                        SSE_red=float(np.mean([1 - C.metrics(y, P["b"])["SSE"] / C.metrics(y, P[c])["SSE"] for c in ck])),
                        SSE_red_fam_lo=np.mean([x["SSE_red_lo"] for x in d_bc_all]), SSE_red_fam_hi=np.mean([x["SSE_red_hi"] for x in d_bc_all]))
            frac_rel = (d_ba["dMAE"] - d_ca["dMAE"]) / d_ba["dMAE"] if d_ba["dMAE"] else np.nan
            for s in (s_a, s_b, s_c, d_ba, d_ca, d_bc): s["frac_gap_ba_attributable_to_relatives"] = frac_rel; srows.append(s)
            md.append(f"\n## Dataset {ds} — {design} (unión de los tests: {len(te_all)} filas, {len(np.unique(g))} familias; retirado en b: {100*s_b['frac_removed_mean']:.1f} % del pool)\n")
            md.append("| condición | n_train medio | MAE [IC fam] | RMSE | R² [IC fam] | top-100 media±sd |\n|---|---|---|---|---|---|")
            for s in (s_a, s_b, s_c):
                md.append(f"| {s['condition']} | {s['n_train_mean']:.0f} | {s['MAE']:.3f} [{s['MAE_fam_lo']:.3f}, {s['MAE_fam_hi']:.3f}] | {s['RMSE']:.3f} | {s['R2']:.4f} [{s['R2_fam_lo']:.4f}, {s['R2_fam_hi']:.4f}] | {s['top100_mean']:.3f}±{s['top100_sd']:.3f} |")
            md.append("\n| diferencia | ΔMAE [IC fam] | ΔRMSE | ΔR² [IC fam] | red. SSE [IC fam] |\n|---|---|---|---|---|")
            for s in (d_ba, d_ca, d_bc):
                md.append(f"| {s['condition']} | {s['dMAE']:+.3f} [{s['dMAE_fam_lo']:+.3f}, {s['dMAE_fam_hi']:+.3f}] | {s['dRMSE']:+.3f} | {s['dR2']:+.4f} [{s['dR2_fam_lo']:+.4f}, {s['dR2_fam_hi']:+.4f}] | {100*s['SSE_red']:+.1f} % [{100*s['SSE_red_fam_lo']:+.1f}, {100*s['SSE_red_fam_hi']:+.1f}] |")
            md.append(f"\nFracción de la brecha (b−a) en MAE atribuible a los familiares y no al tamaño: {100*frac_rel:.1f} %.\n")
        print(f"[{ds}] total {time.time()-T0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(args.out / "remove_relatives_control.csv", index=False)
    pd.DataFrame(srows).to_csv(args.out / "remove_relatives_control_summary.csv", index=False)
    (args.out / "remove_relatives_control.md").write_text("\n".join(md))
    print(f"wrote {args.out / 'remove_relatives_control.csv'}, _summary.csv, .md")


if __name__ == "__main__":
    main()
