"""Seed experiment 6: does the Simpson reversal replicate on Dataset A?

Replication across two independent datasets and featurizations is the signature
of Paper 1; any Paper 2 finding must carry over or be reported as dataset-specific.

Dataset A = Hamidieh/UCI (N=21,263, 81 statistical features), grouped by the set
of constituent elements exactly as in tc_pipeline.py (pandas.factorize on the
sorted element tuple, so group ids are deterministic).

Reproduces on Dataset A the two findings from Dataset B:
  1. Deployment-regime coverage deficit (train+calib random, test = unseen families)
  2. Simpson reversal: nn_dist is NEGATIVELY associated with interval violation
     marginally, but POSITIVELY associated within the high-predicted-Tc subset.

Outputs: results_seed/datasetA_drivers.{csv,json}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ALPHA = 0.10
SEEDS = (0, 1, 2, 3, 4)
XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9,
           colsample_bytree=0.9, tree_method="hist", n_jobs=1, random_state=42)
N_BINS = 5
HI_PRED = 40.0

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))  # repo root
DATA = ROOT / "data"
OUT = ROOT / "results_seed"


def conformal_q(residuals, alpha):
    n = len(residuals)
    return float(np.sort(residuals)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])


def load_dataset_a():
    train = pd.read_csv(DATA / "train.csv")
    uniq = pd.read_csv(DATA / "unique_m.csv")
    y = train["critical_temp"].to_numpy(float)
    X = train.drop(columns=["critical_temp"]).to_numpy(np.float32)

    elem_cols = [c for c in uniq.columns if c not in ("critical_temp", "material")]
    present = uniq[elem_cols].to_numpy() > 0
    keys = ["-".join(sorted(np.asarray(elem_cols)[row])) for row in present]
    groups = pd.factorize(pd.Series(keys))[0]
    assert len(groups) == len(y) == X.shape[0]
    return X, y, groups


def main() -> None:
    X, y, groups = load_dataset_a()
    print("Dataset A:", X.shape, "| families:", len(np.unique(groups)), flush=True)

    cov_rows, corr_rows, cond_rows, bin_rows = [], [], [], []

    for seed in SEEDS:
        rest, test = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed)
                          .split(X, y, groups))
        tr, cal = train_test_split(rest, test_size=0.30, random_state=seed + 100)

        model = XGBRegressor(**XGB).fit(X[tr], y[tr])
        pred = model.predict(X[test])
        r_cal = np.abs(y[cal] - model.predict(X[cal]))
        q = conformal_q(r_cal, ALPHA)
        inside = ((y[test] >= pred - q) & (y[test] <= pred + q)).astype(float)
        violated = 1.0 - inside

        sc = StandardScaler().fit(X[tr])
        nn_dist = NearestNeighbors(n_neighbors=1).fit(sc.transform(X[tr])) \
            .kneighbors(sc.transform(X[test]))[0].ravel()

        hiT = y[test] >= 77
        fam = groups[test]
        fam_cov = np.asarray([inside[fam == f].mean() for f in np.unique(fam)
                              if (fam == f).sum() >= 5], float)
        cov_rows.append(dict(
            seed=seed, coverage=round(float(inside.mean()), 4),
            coverage_hiTc=round(float(inside[hiT].mean()), 4),
            width_mean=round(float(2 * q), 3),
            mae=round(float(np.mean(np.abs(y[test] - pred))), 3),
            fam_cov_p10=round(float(np.percentile(fam_cov, 10)), 4) if len(fam_cov) else np.nan,
            frac_families_undercovered=round(float(np.mean(fam_cov < 0.80)), 4) if len(fam_cov) else np.nan))

        rho, _ = spearmanr(nn_dist, violated)
        corr_rows.append(dict(seed=seed, variable="nn_dist",
                              spearman_rho_vs_violation=round(float(rho), 4)))
        rho_p, _ = spearmanr(pred, violated)
        corr_rows.append(dict(seed=seed, variable="pred_tc",
                              spearman_rho_vs_violation=round(float(rho_p), 4)))

        edges = np.quantile(nn_dist, np.linspace(0, 1, N_BINS + 1))
        b = np.clip(np.digitize(nn_dist, edges[1:-1]), 0, N_BINS - 1)
        for k in range(N_BINS):
            m = b == k
            if m.sum():
                bin_rows.append(dict(seed=seed, bin=k,
                                     d_median=round(float(np.median(nn_dist[m])), 4),
                                     coverage=round(float(inside[m].mean()), 4),
                                     mae=round(float(np.mean(np.abs(y[test][m] - pred[m]))), 3),
                                     frac_hiTc=round(float(hiT[m].mean()), 4), n=int(m.sum())))

        hp = pred >= HI_PRED
        if hp.sum() > 200:
            v = nn_dist[hp]
            e3 = np.quantile(v, np.linspace(0, 1, 4))
            b3 = np.clip(np.digitize(v, e3[1:-1]), 0, 2)
            for k in range(3):
                m = b3 == k
                if m.sum():
                    cond_rows.append(dict(seed=seed, tercile=k,
                                          d_median=round(float(np.median(v[m])), 4),
                                          coverage=round(float(inside[hp][m].mean()), 4),
                                          n=int(m.sum())))
        print("seed", seed, "cov", round(float(inside.mean()), 3),
              "hiTc", round(float(inside[hiT].mean()), 3), flush=True)

    OUT.mkdir(exist_ok=True)
    for name, rows in (("datasetA_cov", cov_rows), ("datasetA_corr", corr_rows),
                       ("datasetA_bins", bin_rows), ("datasetA_cond", cond_rows)):
        keys = list(rows[0].keys())
        with (OUT / f"{name}.csv").open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r[k]) for k in keys) + "\n")

    def mean_of(rows, field, **filt):
        sel = [r for r in rows if all(r[k] == v for k, v in filt.items())]
        return round(float(np.nanmean([r[field] for r in sel])), 4) if sel else None

    summary = {
        "deployment_coverage": {f: mean_of(cov_rows, f) for f in
                                ("coverage", "coverage_hiTc", "width_mean", "mae",
                                 "fam_cov_p10", "frac_families_undercovered")},
        "coverage_std": round(float(np.std([r["coverage"] for r in cov_rows])), 4),
        "coverage_hiTc_std": round(float(np.std([r["coverage_hiTc"] for r in cov_rows])), 4),
        "spearman_vs_violation": {
            v: mean_of(corr_rows, "spearman_rho_vs_violation", variable=v)
            for v in ("nn_dist", "pred_tc")},
        "coverage_by_nn_dist_quintile": {
            str(k): {f: mean_of(bin_rows, f, bin=k)
                     for f in ("d_median", "coverage", "mae", "frac_hiTc")}
            for k in range(N_BINS)},
        "nn_dist_within_highTc": {
            str(k): {f: mean_of(cond_rows, f, tercile=k) for f in ("d_median", "coverage", "n")}
            for k in range(3)},
    }
    (OUT / "datasetA_drivers.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
