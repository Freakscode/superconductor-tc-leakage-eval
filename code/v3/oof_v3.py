"""Out-of-fold predictions on the v3 canonical data (Dataset A; Dataset B v3 checkpoint) - the input
of every paired-by-family interval of the revision (H07).

For each dataset x model in {XGBoost, RandomForest, FamilyMeanNull} x protocol in {random, grouped}
(seed 42; random = KFold(5, shuffle=True, random_state=42); grouped = GroupKFold(5) deterministic,
checked against splits/), writes results_v3/oof_v3.csv.gz with
    dataset, model, protocol, row_index, family_id, fold, y_true, y_pred, thr_p90_train, procedencia
thr_p90_train = np.percentile(y[train fold], 90); high-Tc operator used elsewhere is `>=`.

Checks (results_v3/oof_v3_check.csv): pooled MAE/RMSE/R2 vs results_v3/tabla1_B_antes_despues.csv
(B v3, n_jobs=1 runs) and, for A, vs results_v2/nulo_presencia.csv (XGBoost, null) and
results/consolidated_results.csv (v1, 3 decimals). Models are fit with n_jobs=8; the check shows
whether the thread count changes any pooled metric (it should not for RF; XGBoost hist is verified).

    python code/v3/oof_v3.py [--datasets A B] [--models XGBoost RandomForest FamilyMeanNull]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h07_common import (ROOT, RES_V2, RES_V3, SEED, N_JOBS, VERS, load_dataset, protocols, fold_labels,
                       check_splits, family_mean_null, metrics, with_n_jobs)

PROCEDENCIA = (f"code/v3/oof_v3.py | {VERS} | seed {SEED}; random = KFold(5, shuffle=True, random_state={SEED}); "
               f"grouped = GroupKFold(5) deterministic (== splits/*fold_deterministic); family = constituent-element set "
               f"(A: tc_pipeline.chemical_families(unique_m.csv); B v3: `groups` of data/datasetB_featurized_v3.npz); "
               f"models = make_models({SEED}) of code/tc_pipeline.py (A) / code/datasetB_pipeline.py (B) with n_jobs={N_JOBS}; "
               f"FamilyMeanNull = mean Tc of the family in the TRAIN fold, global train mean if unseen; "
               f"thr_p90_train = np.percentile(y[train fold], 90), high-Tc operator >=")


def run_one(X, y, groups, make_models, model, cv, g):
    yp = np.full(len(y), np.nan); thr = np.full(len(y), np.nan); folds = np.full(len(y), -1)
    it = cv.split(X, y, g) if g is not None else cv.split(X, y)
    for k, (tr, te) in enumerate(it):
        if model == "FamilyMeanNull":
            yp[te] = family_mean_null(y[tr], groups[tr], groups[te])
        else:
            m = with_n_jobs(make_models(SEED)[model]); m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te])
        thr[te] = np.percentile(y[tr], 90); folds[te] = k
    assert not np.isnan(yp).any()
    return yp, thr, folds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["A", "B"])
    ap.add_argument("--models", nargs="+", default=["XGBoost", "RandomForest", "FamilyMeanNull"])
    ap.add_argument("--out", type=Path, default=RES_V3)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    # reference values --------------------------------------------------------------------------
    refB = pd.read_csv(RES_V3 / "tabla1_B_antes_despues.csv"); refB = refB[refB.checkpoint == "v3"]
    refB_name = {"XGBoost": "XGBoost", "RandomForest": "RandomForest", "FamilyMeanNull": "Family-mean null"}
    refA2 = pd.read_csv(RES_V2 / "nulo_presencia.csv"); refA2 = refA2[refA2.dataset == "A"]
    refA2_name = {"XGBoost": "features_completas_paper", "FamilyMeanNull": "nulo_media_familia"}
    refA2_split = {"random": "aleatorio_KFold5", "grouped": "GroupKFold5_familias"}
    refv1 = pd.read_csv(ROOT / "results" / "consolidated_results.csv")
    refv1["ds"] = refv1.dataset.str[0]; refv1["protocol"] = refv1.split.map({"Random": "random", "Chemical-family": "grouped"})

    frames, checks = [], []
    for ds in args.datasets:
        X, y, groups, make_models, meta = load_dataset(ds)
        check_splits(ds, X, y, groups, meta["splits"])
        print(f"[{ds}] n={len(y)} k={len(np.unique(groups))} folds OK vs {meta['splits'].name}", flush=True)
        for model in args.models:
            for proto, (cv, g) in protocols(groups).items():
                t0 = time.time()
                yp, thr, folds = run_one(X, y, groups, make_models, model, cv, g)
                frames.append(pd.DataFrame({"dataset": ds, "model": model, "protocol": proto, "row_index": np.arange(len(y)),
                                            "family_id": groups, "fold": folds, "y_true": y, "y_pred": yp,
                                            "thr_p90_train": thr, "procedencia": PROCEDENCIA}))
                m = metrics(y, yp)
                row = {"dataset": ds, "model": model, "protocol": proto, "n": len(y), "k": len(np.unique(groups)),
                       "n_jobs": N_JOBS, **m, "seconds": round(time.time() - t0, 1)}
                if ds == "B":
                    r = refB[(refB.modelo == refB_name[model]) & (refB.split == proto) & (refB.random_state == 42)]
                    if len(r) == 1:
                        row.update(ref="results_v3/tabla1_B_antes_despues.csv (v3, n_jobs=1)", MAE_ref=r.MAE.iloc[0], RMSE_ref=r.RMSE.iloc[0], R2_ref=r.R2.iloc[0])
                if ds == "A":
                    r = refA2[(refA2.representacion == refA2_name.get(model, "?")) & (refA2.split == refA2_split[proto])]
                    if len(r) == 1:
                        row.update(ref="results_v2/nulo_presencia.csv", MAE_ref=r.MAE.iloc[0], RMSE_ref=r.RMSE.iloc[0], R2_ref=r.R2.iloc[0])
                    r1 = refv1[(refv1.ds == "A") & (refv1.model == model) & (refv1.protocol == proto)]
                    if len(r1) == 1:
                        row.update(MAE_v1_3dp=r1.MAE_K.iloc[0], R2_v1=r1.R2.iloc[0])
                if "MAE_ref" in row:
                    row["dMAE_vs_ref"] = row["MAE"] - row["MAE_ref"]; row["dR2_vs_ref"] = row["R2"] - row["R2_ref"]
                    row["match_1e-6"] = bool(abs(row["dMAE_vs_ref"]) < 1e-6 and abs(row["dR2_vs_ref"]) < 1e-6)
                checks.append(row)
                print(f"  {ds} {model:14s} {proto:8s} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}"
                      + (f"  ref MAE={row['MAE_ref']:.4f} d={row['dMAE_vs_ref']:+.2e}" if "MAE_ref" in row else "")
                      + f" ({row['seconds']} s)", flush=True)
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(args.out / "oof_v3.csv.gz", index=False, compression="gzip")
    C = pd.DataFrame(checks); C["procedencia"] = PROCEDENCIA
    C.to_csv(args.out / "oof_v3_check.csv", index=False)
    print(C.drop(columns="procedencia").to_string())
    print(f"wrote {args.out / 'oof_v3.csv.gz'} ({len(oof)} rows) and oof_v3_check.csv")


if __name__ == "__main__":
    main()
