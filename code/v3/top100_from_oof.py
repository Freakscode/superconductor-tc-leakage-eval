"""Top-100 screening precision recomputed from the released out-of-fold predictions (v3).

results_v2/top100_unificado.csv and results_v2/top100_factorial.csv were written by interactive
driver cells (README).  For the FULL model (XGBoost, seed 42) every quantity they contain is a
deterministic function of results_v3/oof_predictions.csv.gz, so this script recomputes, per
(dataset, model, protocol) and per fold:

    top100 precision, threshold = 90th percentile of the TRAINING fold   (the paper's definition;
                                   stored per row as threshold_p90_train)
    top100 precision, threshold = 90th percentile of the WHOLE dataset   (the leaky alternative,
                                   quoted as differing by 0.4 points on Dataset B)

with the same rule as tc_pipeline.topk_precision: y_true[argsort(y_pred)[-100:]] >= thr.
Writes results_v3/top100_from_oof.csv and prints the comparison with the v2 files.

    python code/v3/top100_from_oof.py
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
RES_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, default=RES_V3 / "oof_predictions.csv.gz")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--out", type=Path, default=RES_V3 / "top100_from_oof.csv")
    args = ap.parse_args()
    oof = pd.read_csv(args.oof)
    rows = []
    for (ds, model, proto, seed), g in oof.groupby(["dataset", "model", "protocol", "seed"]):
        y_all = g.sort_values("row_index").y_true.values; thr_full = np.percentile(y_all, 90)
        pf_train, pf_full = [], []
        for k, f in g.groupby("fold"):
            top = f.y_true.values[np.argsort(f.y_pred.values)[-args.K:]]
            thr_tr = f.threshold_p90_train.iloc[0]; assert (f.threshold_p90_train == thr_tr).all()
            pf_train.append(float((top >= thr_tr).mean())); pf_full.append(float((top >= thr_full).mean()))
        rows.append({"dataset": ds, "model": model, "protocol": proto, "seed": seed, "K": args.K,
                     "top100_trainfold_mean": np.mean(pf_train), "top100_trainfold_std": np.std(pf_train),
                     "top100_trainfold_folds": ";".join(f"{v:.4f}" for v in pf_train),
                     "top100_fulldata_mean": np.mean(pf_full), "top100_fulldata_std": np.std(pf_full),
                     "top100_fulldata_folds": ";".join(f"{v:.4f}" for v in pf_full),
                     "diff_trainfold_minus_fulldata_pts": 100 * (np.mean(pf_train) - np.mean(pf_full)),
                     "procedencia": "code/v3/top100_from_oof.py | from results_v3/oof_predictions.csv.gz (code/v3/dump_oof_predictions.py); "
                                    "rule = tc_pipeline.topk_precision (y_true[argsort(y_pred)[-K:]] >= thr); train-fold threshold = threshold_p90_train column; "
                                    "full-data threshold = np.percentile(y, 90) over all rows"})
    out = pd.DataFrame(rows)
    # --- compare with v2 -----------------------------------------------------------------------------
    v2 = pd.read_csv(ROOT / "results_v2" / "nulo_presencia.csv"); v2 = v2[v2.representacion == "features_completas_paper"]
    v2["protocol"] = v2.split.map({"aleatorio_KFold5": "random", "GroupKFold5_familias": "grouped"})
    for i, r in out.iterrows():
        m = v2[(v2.dataset == r.dataset) & (v2.protocol == r.protocol)]
        if r.model == "XGBoost" and len(m) == 1:
            out.loc[i, "v2_top100_mean"] = m.top100_mean.iloc[0]; out.loc[i, "v2_top100_std"] = m.top100_std.iloc[0]
            out.loc[i, "equals_v2"] = bool(abs(r.top100_trainfold_mean - m.top100_mean.iloc[0]) < 1e-9 and abs(r.top100_trainfold_std - m.top100_std.iloc[0]) < 1e-6)
    tf = pd.read_csv(ROOT / "results_v2" / "top100_factorial.csv")
    v2_full = tf[tf.celda == "full|npz_distribuido"].top100_mean.iloc[0]
    sel = (out.dataset == "B") & (out.model == "XGBoost") & (out.protocol == "grouped")
    out.loc[sel, "v2_factorial_fulldata_mean"] = v2_full
    args.out.parent.mkdir(parents=True, exist_ok=True); out.to_csv(args.out, index=False)
    print(out.drop(columns=["procedencia", "top100_trainfold_folds", "top100_fulldata_folds"]).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
