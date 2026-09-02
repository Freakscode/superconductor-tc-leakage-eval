"""Seed experiment 5: what actually predicts the coverage deficit?

seed_diag.py produced an inverted result: coverage RISES with nearest-neighbour
distance (0.64 in the nearest bin to 0.93 in the farthest) while MAE FALLS
(10.9 -> 3.5 K). The confound is visible in the same table: the fraction of
Tc >= 77 K materials falls from 0.25 to 0.01 across those bins. Densely sampled
cuprate families sit close to training data AND carry the large Tc values and
large errors; distant materials are mostly low-Tc compounds that are trivially
predicted near zero.

So Paper 1's NN-distance diagnostic, which explained MAE inflation BETWEEN split
schemes, does not locate coverage failure WITHIN the deployment regime. This
asks what does.

Candidate conditioning variables, all computable at prediction time without y:
  nn_dist    - standardized distance to nearest training material (the Paper 1 quantity)
  pred_tc    - the model's own point prediction
  ens_std    - disagreement across a 10-member bagged ensemble
  local_res  - mean residual of the k=25 nearest CALIBRATION materials

For each: coverage of the nominal-90% interval by quintile, and Spearman
correlation between the variable and per-material interval violation. Then a
conditional test: does nn_dist retain any signal once pred_tc is controlled
(coverage by nn_dist quintile WITHIN the high-Tc subset)?

Outputs: results_seed/diag_drivers.{csv,json}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ALPHA = 0.10
SEEDS = (0, 1, 2, 3, 4)
XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9,
           colsample_bytree=0.9, tree_method="hist", n_jobs=1, random_state=42)
N_ENS = 10
K_LOCAL = 25
N_BINS = 5

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))  # repo root
NPZ = ROOT / "data" / "datasetB_featurized.npz"
OUT = ROOT / "results_seed"


def conformal_q(residuals, alpha):
    n = len(residuals)
    return float(np.sort(residuals)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    X, y, groups = z["X"], z["y"], z["groups"]

    bin_rows, corr_rows, cond_rows = [], [], []

    for seed in SEEDS:
        rest, test = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed)
                          .split(X, y, groups))
        tr, cal = train_test_split(rest, test_size=0.30, random_state=seed + 100)

        model = XGBRegressor(**XGB).fit(X[tr], y[tr])
        pred = model.predict(X[test])
        r_cal = np.abs(y[cal] - model.predict(X[cal]))
        q = conformal_q(r_cal, ALPHA)
        violated = (np.abs(y[test] - pred) > q).astype(float)
        inside = 1.0 - violated

        sc = StandardScaler().fit(X[tr])
        Xtr_s, Xte_s, Xcal_s = sc.transform(X[tr]), sc.transform(X[test]), sc.transform(X[cal])

        nn1 = NearestNeighbors(n_neighbors=1).fit(Xtr_s)
        nn_dist = nn1.kneighbors(Xte_s)[0].ravel()

        ens = []
        rs = np.random.RandomState(seed)
        for j in range(N_ENS):
            idx = rs.choice(len(tr), size=len(tr), replace=True)
            mj = XGBRegressor(**{**XGB, "n_estimators": 200, "random_state": 1000 + j})
            mj.fit(X[tr][idx], y[tr][idx])
            ens.append(mj.predict(X[test]))
        ens_std = np.std(np.vstack(ens), axis=0)

        nnk = NearestNeighbors(n_neighbors=min(K_LOCAL, len(cal))).fit(Xcal_s)
        local_res = r_cal[nnk.kneighbors(Xte_s)[1]].mean(axis=1)

        variables = {"nn_dist": nn_dist, "pred_tc": pred,
                     "ens_std": ens_std, "local_res": local_res}

        for name, v in variables.items():
            rho, p = spearmanr(v, violated)
            corr_rows.append(dict(seed=seed, variable=name,
                                  spearman_rho_vs_violation=round(float(rho), 4),
                                  p_value=float(p)))
            edges = np.quantile(v, np.linspace(0, 1, N_BINS + 1))
            b = np.clip(np.digitize(v, edges[1:-1]), 0, N_BINS - 1)
            for k in range(N_BINS):
                m = b == k
                if m.sum() == 0:
                    continue
                bin_rows.append(dict(seed=seed, variable=name, bin=k,
                                     v_median=round(float(np.median(v[m])), 4),
                                     coverage=round(float(inside[m].mean()), 4),
                                     mae=round(float(np.mean(np.abs(y[test][m] - pred[m]))), 3),
                                     frac_hiTc=round(float((y[test][m] >= 77).mean()), 4),
                                     n=int(m.sum())))

        # conditional test: nn_dist within the high-predicted-Tc subset only
        hi = pred >= 40.0
        if hi.sum() > 200:
            v = nn_dist[hi]
            edges = np.quantile(v, np.linspace(0, 1, 4))
            b = np.clip(np.digitize(v, edges[1:-1]), 0, 2)
            for k in range(3):
                m = b == k
                if m.sum() == 0:
                    continue
                cond_rows.append(dict(seed=seed, tercile=k,
                                      d_median=round(float(np.median(v[m])), 4),
                                      coverage=round(float(inside[hi][m].mean()), 4),
                                      n=int(m.sum())))
        print("seed", seed, "done", flush=True)

    OUT.mkdir(exist_ok=True)
    for name, rows in (("diag_drivers_bins", bin_rows), ("diag_drivers_corr", corr_rows),
                       ("diag_drivers_cond", cond_rows)):
        keys = list(rows[0].keys())
        with (OUT / f"{name}.csv").open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r[k]) for k in keys) + "\n")

    summary = {"spearman_vs_violation": {}, "coverage_by_quintile": {}, "nn_dist_within_highTc": {}}
    for name in ("nn_dist", "pred_tc", "ens_std", "local_res"):
        sel = [r for r in corr_rows if r["variable"] == name]
        summary["spearman_vs_violation"][name] = dict(
            mean=round(float(np.mean([r["spearman_rho_vs_violation"] for r in sel])), 4),
            std=round(float(np.std([r["spearman_rho_vs_violation"] for r in sel])), 4))
        summary["coverage_by_quintile"][name] = {
            str(k): {f: round(float(np.mean([r[f] for r in bin_rows
                                             if r["variable"] == name and r["bin"] == k])), 4)
                     for f in ("v_median", "coverage", "mae", "frac_hiTc")}
            for k in range(N_BINS)}
    for k in range(3):
        sel = [r for r in cond_rows if r["tercile"] == k]
        if sel:
            summary["nn_dist_within_highTc"][str(k)] = dict(
                d_median=round(float(np.mean([r["d_median"] for r in sel])), 4),
                coverage=round(float(np.mean([r["coverage"] for r in sel])), 4),
                n=round(float(np.mean([r["n"] for r in sel])), 1))
    (OUT / "diag_drivers.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
