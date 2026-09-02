"""Seed experiment 4: is the coverage deficit PREDICTABLE before you see the truth?

Paper 1 introduced a nearest-neighbour (NN) distance diagnostic to explain why
random splits inflate MAE. This asks whether the same quantity, computed at
prediction time with no access to y, forecasts WHERE conformal intervals stop
covering. If it does, the practitioner gets an actionable instrument (an abstention
/ widening rule) instead of a warning, which is the difference between a paper that
reports a problem and one that hands over a fix.

Deployment regime only (train+calib random, test = unseen chemical families).

Part A - Is coverage a function of NN distance?
  Bin test materials by standardized distance to their nearest TRAINING material;
  report empirical coverage of the nominal-90% split-conformal interval per bin.

Part B - Does a distance-aware repair work?
  distance-Mondrian : conformal calibrated separately within each NN-distance bin
                      (bins defined on calibration data, applied to test)
  Compared against plain split-conformal and against a selective-prediction rule
  that abstains on the most distant q% of test materials.

Outputs: results_seed/diag_distance.{csv,json}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ALPHA = 0.10
SEEDS = (0, 1, 2, 3, 4)
XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9,
           colsample_bytree=0.9, tree_method="hist", n_jobs=1, random_state=42)
N_BINS = 5
ABSTAIN_FRACS = (0.0, 0.10, 0.20, 0.30, 0.50)

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))  # repo root
NPZ = ROOT / "data" / "datasetB_featurized.npz"
OUT = ROOT / "results_seed"


def conformal_q(residuals: np.ndarray, alpha: float) -> float:
    n = len(residuals)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(residuals)[k - 1])


def nn_distance(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Standardized Euclidean distance from each query row to nearest reference row."""
    sc = StandardScaler().fit(reference)
    nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(reference))
    d, _ = nn.kneighbors(sc.transform(query))
    return d.ravel()


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    X, y, groups = z["X"], z["y"], z["groups"]

    bin_rows, repair_rows, abstain_rows = [], [], []

    for seed in SEEDS:
        rest, test = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed)
                          .split(X, y, groups))
        tr, cal = train_test_split(rest, test_size=0.30, random_state=seed + 100)

        model = XGBRegressor(**XGB).fit(X[tr], y[tr])
        pred = model.predict(X[test])
        r_cal = np.abs(y[cal] - model.predict(X[cal]))

        d_cal = nn_distance(X[tr], X[cal])
        d_test = nn_distance(X[tr], X[test])

        # ---------- Part A: coverage as a function of NN distance ----------
        q_plain = conformal_q(r_cal, ALPHA)
        inside_plain = (y[test] >= pred - q_plain) & (y[test] <= pred + q_plain)
        edges = np.quantile(d_test, np.linspace(0, 1, N_BINS + 1))
        edges[-1] += 1e-9
        b_test = np.clip(np.digitize(d_test, edges[1:-1]), 0, N_BINS - 1)
        for b in range(N_BINS):
            m = b_test == b
            if m.sum() == 0:
                continue
            bin_rows.append(dict(
                seed=seed, bin=b,
                d_median=round(float(np.median(d_test[m])), 4),
                n=int(m.sum()),
                coverage=round(float(inside_plain[m].mean()), 4),
                mae=round(float(np.mean(np.abs(y[test][m] - pred[m]))), 3),
                frac_hiTc=round(float((y[test][m] >= 77).mean()), 4),
            ))

        # ---------- Part B1: distance-Mondrian conformal ----------
        cal_edges = np.quantile(d_cal, np.linspace(0, 1, N_BINS + 1))
        cal_edges[0], cal_edges[-1] = -np.inf, np.inf
        b_cal_m = np.digitize(d_cal, cal_edges[1:-1])
        b_test_m = np.digitize(d_test, cal_edges[1:-1])
        lo, hi = np.empty_like(pred), np.empty_like(pred)
        for b in np.unique(b_test_m):
            mc, mt = b_cal_m == b, b_test_m == b
            qb = conformal_q(r_cal[mc], ALPHA) if mc.sum() >= 50 else q_plain
            lo[mt], hi[mt] = pred[mt] - qb, pred[mt] + qb
        for name, (l_, h_) in {"split": (pred - q_plain, pred + q_plain),
                               "distance-mondrian": (lo, hi)}.items():
            ins = (y[test] >= l_) & (y[test] <= h_)
            hiT = y[test] >= 77
            repair_rows.append(dict(
                seed=seed, method=name,
                coverage=round(float(ins.mean()), 4),
                coverage_hiTc=round(float(ins[hiT].mean()), 4),
                width_mean=round(float(np.mean(h_ - l_)), 3),
            ))

        # ---------- Part B2: selective prediction (abstain on most distant) ----------
        order = np.argsort(d_test)
        for frac in ABSTAIN_FRACS:
            keep = order[: int(round(len(order) * (1 - frac)))]
            ins = inside_plain[keep]
            hiT = y[test][keep] >= 77
            abstain_rows.append(dict(
                seed=seed, abstain_frac=frac, n_kept=int(len(keep)),
                coverage=round(float(ins.mean()), 4),
                coverage_hiTc=round(float(ins[hiT].mean()), 4) if hiT.any() else float("nan"),
                n_hiTc_kept=int(hiT.sum()),
                mae=round(float(np.mean(np.abs(y[test][keep] - pred[keep]))), 3),
            ))
        print("seed", seed, "done", flush=True)

    OUT.mkdir(exist_ok=True)
    for name, rows in (("diag_bins", bin_rows), ("diag_repair", repair_rows),
                       ("diag_abstain", abstain_rows)):
        keys = list(rows[0].keys())
        with (OUT / f"{name}.csv").open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r[k]) for k in keys) + "\n")

    def agg(rows, by, fields):
        out = {}
        for key in sorted({r[by] for r in rows}):
            sel = [r for r in rows if r[by] == key]
            out[str(key)] = {f: round(float(np.nanmean([r[f] for r in sel])), 4)
                             for f in fields}
            out[str(key)]["n_seeds"] = len(sel)
        return out

    summary = {
        "coverage_by_distance_bin": agg(bin_rows, "bin",
                                        ["d_median", "coverage", "mae", "n", "frac_hiTc"]),
        "repair": agg(repair_rows, "method", ["coverage", "coverage_hiTc", "width_mean"]),
        "selective_prediction": agg(abstain_rows, "abstain_frac",
                                    ["coverage", "coverage_hiTc", "n_kept",
                                     "n_hiTc_kept", "mae"]),
    }
    (OUT / "diag_distance.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
