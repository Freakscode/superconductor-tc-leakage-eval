"""Representation ablation with identical folds and identical model budget (H10).

Representations (same XGBoost = make_models(42)["XGBoost"] of each pipeline, n_jobs=8; NOT re-tuned per
representation: the question is what each representation adds under the paper's configuration, and
re-tuning would confound representation with budget; the tree booster is not scale-sensitive, so the
binary/fraction inputs need no rescaling):
    R0  element presence (0/1)                      A: 86 columns (unique_m.csv > 0)   B: 83 (pymatgen)
    R1  presence + molar fractions (0 if absent)     A: unique_m amounts / row sum       B: Composition.fractional_composition
    R2  the paper's descriptors                      A: 81 Hamidieh statistics           B: 132 Magpie
    R3  R1 + R2 concatenated
References (pre-specified before measuring):
    kNN5_R1   KNeighborsRegressor(k=5, Euclidean on R1 as is, weights = 1/d; exact duplicates get all the weight)
    ET500_R2  ExtraTreesRegressor(n_estimators=500, default max_features=1.0, random_state=42) on R2
Protocols: random = KFold(5, shuffle=True, random_state=42); grouped = GroupKFold(5) deterministic
(== splits/*fold_deterministic). Metrics: pooled out-of-fold MAE/RMSE/R2/SSE; top-100 precision per fold
(threshold = p90 of the TRAIN fold, operator >=), mean/sd over the 5 folds.
Inference: paired bootstrap of whole FAMILIES (2 000 resamples, default_rng(0)), same resample for every
representation; the row-level paired bootstrap is reported next to it. Intervals are conditional on the
fitted OOF predictions. Differences are reported as losses (dMAE, dRMSE, dR2) and as the relative
reduction of the residual sum of squares 1 - SSE_b/SSE_a; NO ratios of R2.

Outputs (results_v3/): ablation_representation.csv, ablation_representation_diffs.csv,
ablation_representation.md, ablation_oof.csv.gz.
    python code/v3/ablation_representation.py [--datasets A B] [--n-boot 2000]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h09h10_common as C
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor

METHODS = ["R0", "R1", "R2", "R3", "kNN5_R1", "ET500_R2"]
PAIRS = [("R0", "R1"), ("R1", "R2"), ("R2", "R3"), ("R0", "R2"), ("R0", "R3"), ("R1", "R3"),
         ("R1", "kNN5_R1"), ("R2", "kNN5_R1"), ("R0", "kNN5_R1"), ("R3", "kNN5_R1"),
         ("R2", "ET500_R2"), ("R0", "ET500_R2"), ("R1", "ET500_R2"), ("R3", "ET500_R2"), ("kNN5_R1", "ET500_R2")]


def model_and_X(d: C.Dataset, method):
    if method.startswith("R"):
        return (lambda: d.xgb()), d.representation(method)
    if method == "kNN5_R1":
        return (lambda: KNeighborsRegressor(n_neighbors=5, weights="distance", metric="euclidean", n_jobs=C.N_JOBS)), d.representation("R1")
    if method == "ET500_R2":
        return (lambda: ExtraTreesRegressor(n_estimators=500, random_state=C.SEED, n_jobs=C.N_JOBS)), d.representation("R2")
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["A", "B"])
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--out", type=Path, default=C.RES_V3)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    proc = (f"code/v3/ablation_representation.py | {C.versions()} | XGBoost = make_models({C.SEED})['XGBoost'] of tc_pipeline (A) / "
            f"datasetB_pipeline (B), n_jobs={C.N_JOBS}, hyperparameters NOT re-tuned per representation | kNN5_R1 = KNeighborsRegressor(5, "
            f"weights=1/d, euclidean on R1) | ET500_R2 = ExtraTreesRegressor(500, random_state={C.SEED}) on R2 | random = KFold(5, shuffle, rs={C.SEED}); "
            f"grouped = GroupKFold(5) deterministic == splits/*fold_deterministic | top100: K=100, threshold p90(y train fold), operator >= | "
            f"intervals: paired bootstrap of whole families (n_boot={args.n_boot}, default_rng({C.BOOT_SEED})), percentile 2.5/97.5, conditional on OOF predictions; "
            f"row-level paired bootstrap reported as *_row_* | A fractions from unique_m.csv amounts (O=1 defect of 1 927 rows kept, results_v3/A_nomenclatura_check.json); "
            f"B fractions from the formula field of datasetB_featurized_v3.npz via pymatgen")
    rows, drows, oof_frames, md = [], [], [], ["# Ablación de representación (H10)\n", f"`{proc}`\n"]
    for ds in args.datasets:
        d = C.load(ds); T0 = time.time()
        splits = {"random": C.random_folds(d), "grouped": C.grouped_folds(d)}
        print(f"[{ds}] n={d.n} k={d.k} | R0 {d.P.shape[1]} | R1 {d.representation('R1').shape[1]} | R2 {d.X2.shape[1]} | R3 {d.representation('R3').shape[1]}", flush=True)
        for proto, sp in splits.items():
            preds, tops, secs, nfeat = {}, {}, {}, {}
            for mth in METHODS:
                fn, X = model_and_X(d, mth)
                yp, tp, sec = C.oof(fn, X, d.y, sp)
                preds[mth], tops[mth], secs[mth], nfeat[mth] = yp, tp, sec, X.shape[1]
                m = C.metrics(d.y, yp)
                print(f"  {ds} {proto:8s} {mth:9s} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} R2={m['R2']:.4f} top100={np.mean(tp):.3f} ({sec:.0f}s)", flush=True)
                oof_frames.append(pd.DataFrame({"dataset": ds, "protocol": proto, "method": mth, "row_index": np.arange(d.n),
                                                "family_id": d.groups, "fold": C.fold_labels(sp, d.n), "y_true": d.y, "y_pred": yp}))
            fam, fpairs = C.family_bootstrap(d.y, preds, d.groups, args.n_boot, C.BOOT_SEED)
            rowb, rpairs = C.family_bootstrap(d.y, preds, np.arange(d.n), args.n_boot, C.BOOT_SEED)
            for mth in METHODS:
                m = C.metrics(d.y, preds[mth])
                rows.append({"dataset": ds, "protocol": proto, "method": mth, "n_features": nfeat[mth], "n": d.n, "k_families": d.k,
                             **{k: m[k] for k in ("MAE", "RMSE", "R2", "SSE")},
                             "SSE_red_vs_R0": 1 - m["SSE"] / C.metrics(d.y, preds["R0"])["SSE"],
                             **{f"{k}_fam_{s}": fam[mth][f"{k}_{s}"] for k in ("MAE", "RMSE", "R2") for s in ("lo", "hi")},
                             **{f"{k}_row_{s}": rowb[mth][f"{k}_{s}"] for k in ("MAE", "R2") for s in ("lo", "hi")},
                             "top100_mean": float(np.mean(tops[mth])), "top100_sd": float(np.std(tops[mth])),
                             "top100_folds": ";".join(f"{t:.2f}" for t in tops[mth]), "fit_seconds": round(secs[mth], 1),
                             "n_boot": args.n_boot, "boot_seed": C.BOOT_SEED, "procedencia": proc})
            for a, b in PAIRS:
                ma, mb = C.metrics(d.y, preds[a]), C.metrics(d.y, preds[b]); fp, rp = fpairs[(a, b)], rpairs[(a, b)]
                td = np.array(tops[b]) - np.array(tops[a])
                drows.append({"dataset": ds, "protocol": proto, "ref_a": a, "method_b": b,
                              "dMAE": mb["MAE"] - ma["MAE"], "dMAE_fam_lo": fp["dMAE_lo"], "dMAE_fam_hi": fp["dMAE_hi"], "dMAE_row_lo": rp["dMAE_lo"], "dMAE_row_hi": rp["dMAE_hi"],
                              "dRMSE": mb["RMSE"] - ma["RMSE"], "dRMSE_fam_lo": fp["dRMSE_lo"], "dRMSE_fam_hi": fp["dRMSE_hi"],
                              "dR2": mb["R2"] - ma["R2"], "dR2_fam_lo": fp["dR2_lo"], "dR2_fam_hi": fp["dR2_hi"], "dR2_row_lo": rp["dR2_lo"], "dR2_row_hi": rp["dR2_hi"],
                              "SSE_red": 1 - mb["SSE"] / ma["SSE"], "SSE_red_fam_lo": fp["SSE_red_lo"], "SSE_red_fam_hi": fp["SSE_red_hi"],
                              "SSE_red_row_lo": rp["SSE_red_lo"], "SSE_red_row_hi": rp["SSE_red_hi"],
                              "dtop100_mean": float(td.mean()), "dtop100_min": float(td.min()), "dtop100_max": float(td.max()),
                              "n_boot": args.n_boot, "boot_seed": C.BOOT_SEED, "procedencia": proc})
            # markdown
            md.append(f"\n## Dataset {ds} — protocolo {proto} (n={d.n}, k={d.k} familias)\n")
            md.append("| método | p | MAE [IC fam] | RMSE | R² [IC fam] | red. SSE vs R0 | top-100 media±sd | s |\n|---|---|---|---|---|---|---|---|")
            for r in rows[-len(METHODS):]:
                md.append(f"| {r['method']} | {r['n_features']} | {r['MAE']:.3f} [{r['MAE_fam_lo']:.3f}, {r['MAE_fam_hi']:.3f}] | {r['RMSE']:.3f} | "
                          f"{r['R2']:.4f} [{r['R2_fam_lo']:.4f}, {r['R2_fam_hi']:.4f}] | {100*r['SSE_red_vs_R0']:+.1f} % | "
                          f"{r['top100_mean']:.3f}±{r['top100_sd']:.3f} | {r['fit_seconds']:.0f} |")
            md.append("\nDiferencias b − a (bootstrap pareado por familias; entre paréntesis por filas):\n")
            md.append("| a → b | ΔMAE [IC fam] (IC filas) | ΔRMSE [IC fam] | ΔR² [IC fam] (IC filas) | red. SSE 1−SSE_b/SSE_a [IC fam] | Δtop-100 media (min, max) |\n|---|---|---|---|---|---|")
            for r in drows[-len(PAIRS):]:
                md.append(f"| {r['ref_a']} → {r['method_b']} | {r['dMAE']:+.3f} [{r['dMAE_fam_lo']:+.3f}, {r['dMAE_fam_hi']:+.3f}] ({r['dMAE_row_lo']:+.3f}, {r['dMAE_row_hi']:+.3f}) | "
                          f"{r['dRMSE']:+.3f} [{r['dRMSE_fam_lo']:+.3f}, {r['dRMSE_fam_hi']:+.3f}] | {r['dR2']:+.4f} [{r['dR2_fam_lo']:+.4f}, {r['dR2_fam_hi']:+.4f}] ({r['dR2_row_lo']:+.4f}, {r['dR2_row_hi']:+.4f}) | "
                          f"{100*r['SSE_red']:+.1f} % [{100*r['SSE_red_fam_lo']:+.1f}, {100*r['SSE_red_fam_hi']:+.1f}] | {r['dtop100_mean']:+.3f} ({r['dtop100_min']:+.2f}, {r['dtop100_max']:+.2f}) |")
        print(f"[{ds}] total {time.time()-T0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(args.out / "ablation_representation.csv", index=False)
    pd.DataFrame(drows).to_csv(args.out / "ablation_representation_diffs.csv", index=False)
    oof = pd.concat(oof_frames, ignore_index=True); oof["procedencia"] = proc
    oof.to_csv(args.out / "ablation_oof.csv.gz", index=False, compression="gzip")
    md.append("\n## Notas\n- Umbral alto-Tc = percentil 90 del fold de ENTRENAMIENTO, operador `>=`; top-100 = precisión entre los 100 de mayor predicción del fold de test.\n"
              "- Intervalos: bootstrap pareado de familias completas (unidad de dependencia), 2 000 remuestreos, semilla 0, condicional a las predicciones OOF ya ajustadas; el intervalo por filas se da solo para mostrar la diferencia.\n"
              "- El XGBoost no se re-afina por representación (misma configuración = mismo presupuesto); la comparación mide lo que cada representación aporta bajo la configuración publicada.\n"
              "- La reducción relativa del error cuadrático residual respecto a R0 es la magnitud comparable entre representaciones; no se reportan razones R²/R².\n")
    (args.out / "ablation_representation.md").write_text("\n".join(md))
    print(f"wrote {args.out / 'ablation_representation.csv'}, _diffs.csv, .md, ablation_oof.csv.gz")


if __name__ == "__main__":
    main()
