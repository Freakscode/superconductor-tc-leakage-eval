"""Seed experiment: does split-conformal UQ hold up under chemical-family shift?

Compares empirical coverage / width of 90% split-conformal prediction intervals
for XGBoost Tc regression on Dataset B (Stanev/SuperCon, 132 Magpie features)
under two calibration/test regimes:

  IID   : calibration and test drawn randomly (exchangeable) -> textbook case
  SHIFT : calibration and test are disjoint chemical families (GroupShuffleSplit)

Also evaluates two candidate repairs:
  - Mondrian / group-size-conditional conformal (bin by family size)
  - Normalized (locally-adaptive) conformal, scaling residuals by a learned
    difficulty estimate sigma(x) fit on the calibration residuals.

Outputs: results/seed_conformal.csv, results/seed_conformal.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from xgboost import XGBRegressor

ALPHA = 0.10  # target 90% coverage
SEEDS = (0, 1, 2, 3, 4)
XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9,
           colsample_bytree=0.9, tree_method="hist", n_jobs=1, random_state=42)

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))  # repo root
NPZ = ROOT / "data" / "datasetB_featurized.npz"
OUT = ROOT / "results_seed"


def conformal_q(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of calibration residuals."""
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(residuals)[k - 1])


def evaluate(y_true, lo, hi):
    cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
    return cov, float(np.mean(hi - lo)), float(np.median(hi - lo))


def run_regime(X, y, groups, regime: str, seed: int) -> list[dict]:
    """Split into train / calib / test under one regime and score three conformal variants."""
    if regime == "shift":
        outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        rest_idx, test_idx = next(outer.split(X, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed + 100)
        tr_idx, cal_idx = next(inner.split(X[rest_idx], y[rest_idx], groups[rest_idx]))
    else:
        outer = ShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        rest_idx, test_idx = next(outer.split(X))
        inner = ShuffleSplit(n_splits=1, test_size=0.30, random_state=seed + 100)
        tr_idx, cal_idx = next(inner.split(X[rest_idx]))
    tr_idx, cal_idx = rest_idx[tr_idx], rest_idx[cal_idx]

    model = XGBRegressor(**XGB).fit(X[tr_idx], y[tr_idx])
    r_cal = np.abs(y[cal_idx] - model.predict(X[cal_idx]))
    pred_test = model.predict(X[test_idx])
    mae = float(np.mean(np.abs(y[test_idx] - pred_test)))

    rows = []

    # --- 1. plain split conformal -------------------------------------------
    q = conformal_q(r_cal, ALPHA)
    cov, w_mean, w_med = evaluate(y[test_idx], pred_test - q, pred_test + q)
    rows.append(dict(method="split-conformal", coverage=cov, width_mean=w_mean,
                     width_median=w_med, mae=mae))

    # --- 2. normalized (locally adaptive) conformal --------------------------
    sig_model = XGBRegressor(**{**XGB, "n_estimators": 300}).fit(
        X[tr_idx], np.abs(y[tr_idx] - model.predict(X[tr_idx])) + 1e-6)
    s_cal = np.clip(sig_model.predict(X[cal_idx]), 1e-3, None)
    s_test = np.clip(sig_model.predict(X[test_idx]), 1e-3, None)
    q_n = conformal_q(r_cal / s_cal, ALPHA)
    cov, w_mean, w_med = evaluate(y[test_idx], pred_test - q_n * s_test,
                                  pred_test + q_n * s_test)
    rows.append(dict(method="normalized-conformal", coverage=cov, width_mean=w_mean,
                     width_median=w_med, mae=mae))

    # --- 3. Mondrian conformal, binned by family size ------------------------
    gsize = np.bincount(groups)[groups]
    edges = np.array([0, 1, 3, 10, np.inf])
    bin_cal = np.digitize(gsize[cal_idx], edges[1:-1])
    bin_test = np.digitize(gsize[test_idx], edges[1:-1])
    lo = np.empty_like(pred_test)
    hi = np.empty_like(pred_test)
    q_global = q
    for b in np.unique(bin_test):
        m_cal, m_test = bin_cal == b, bin_test == b
        qb = conformal_q(r_cal[m_cal], ALPHA) if m_cal.sum() >= 50 else q_global
        lo[m_test] = pred_test[m_test] - qb
        hi[m_test] = pred_test[m_test] + qb
    cov, w_mean, w_med = evaluate(y[test_idx], lo, hi)
    rows.append(dict(method="mondrian-conformal", coverage=cov, width_mean=w_mean,
                     width_median=w_med, mae=mae))

    for r in rows:
        r.update(regime=regime, seed=seed, n_train=len(tr_idx),
                 n_calib=len(cal_idx), n_test=len(test_idx))
    return rows


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    X, y, groups = z["X"], z["y"], z["groups"]

    records = [r for regime in ("iid", "shift") for seed in SEEDS
               for r in run_regime(X, y, groups, regime, seed)]

    OUT.mkdir(exist_ok=True)
    keys = ["regime", "method", "seed", "coverage", "width_mean", "width_median",
            "mae", "n_train", "n_calib", "n_test"]
    with (OUT / "seed_conformal.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in records:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    summary = {}
    for regime in ("iid", "shift"):
        for method in ("split-conformal", "normalized-conformal", "mondrian-conformal"):
            sel = [r for r in records if r["regime"] == regime and r["method"] == method]
            summary[f"{regime}|{method}"] = dict(
                coverage_mean=round(float(np.mean([r["coverage"] for r in sel])), 4),
                coverage_std=round(float(np.std([r["coverage"] for r in sel])), 4),
                width_mean=round(float(np.mean([r["width_mean"] for r in sel])), 3),
                mae_mean=round(float(np.mean([r["mae"] for r in sel])), 3),
            )
    (OUT / "seed_conformal.json").write_text(
        json.dumps({"target_coverage": 1 - ALPHA, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
