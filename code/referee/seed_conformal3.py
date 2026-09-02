"""Seed experiment 3: do the corrective conformal variants repair CONDITIONAL coverage?

Supersedes seed_conformal.py and seed_conformal2.py, which each measured half of
what the claim needs:
  - seed_conformal.py  tested 3 methods but only MARGINAL coverage, and only in
    the iid / shift regimes (never 'deploy').
  - seed_conformal2.py tested conditional coverage but only for plain
    split-conformal.

Here: 3 methods x 3 regimes x 5 seeds, with conditional coverage throughout, so
that any claim about whether the textbook fixes repair the deployment failure
rests on an actual measurement.

Methods : split-conformal, normalized (locally adaptive), Mondrian (by family size)
Regimes : iid    - train/calib/test all random
          shift  - train/calib/test all family-disjoint
          deploy - train+calib random, test = unseen families  (the realistic case)

Conditional metrics: coverage on Tc >= 77 K (screening region), per-family
coverage 10th percentile, and fraction of families covered below 80%.

Outputs: results_seed/variants_conditional.{csv,json}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, train_test_split
from xgboost import XGBRegressor

ALPHA = 0.10
SEEDS = (0, 1, 2, 3, 4)
XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9,
           colsample_bytree=0.9, tree_method="hist", n_jobs=1, random_state=42)
REGIMES = ("iid", "shift", "deploy")
METHODS = ("split", "normalized", "mondrian")

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))  # repo root
NPZ = ROOT / "data" / "datasetB_featurized.npz"
OUT = ROOT / "results_seed"


def conformal_q(residuals: np.ndarray, alpha: float) -> float:
    n = len(residuals)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(residuals)[k - 1])


def make_split(X, y, groups, regime: str, seed: int):
    if regime == "iid":
        rest, test = next(ShuffleSplit(1, test_size=0.30, random_state=seed).split(X))
        tr, cal = next(ShuffleSplit(1, test_size=0.30, random_state=seed + 100).split(X[rest]))
        return rest[tr], rest[cal], test
    if regime == "shift":
        rest, test = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed)
                          .split(X, y, groups))
        tr, cal = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed + 100)
                       .split(X[rest], y[rest], groups[rest]))
        return rest[tr], rest[cal], test
    if regime == "deploy":
        rest, test = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed)
                          .split(X, y, groups))
        tr, cal = train_test_split(rest, test_size=0.30, random_state=seed + 100)
        return tr, cal, test
    raise ValueError(regime)


def intervals(method, X, y, groups, tr, cal, test, model, pred_test):
    """Return (lo, hi) for one conformal variant."""
    r_cal = np.abs(y[cal] - model.predict(X[cal]))

    if method == "split":
        q = conformal_q(r_cal, ALPHA)
        return pred_test - q, pred_test + q

    if method == "normalized":
        sig = XGBRegressor(**{**XGB, "n_estimators": 300}).fit(
            X[tr], np.abs(y[tr] - model.predict(X[tr])) + 1e-6)
        s_cal = np.clip(sig.predict(X[cal]), 1e-3, None)
        s_test = np.clip(sig.predict(X[test]), 1e-3, None)
        q = conformal_q(r_cal / s_cal, ALPHA)
        return pred_test - q * s_test, pred_test + q * s_test

    if method == "mondrian":
        gsize = np.bincount(groups)[groups]
        edges = np.array([1, 3, 10])
        b_cal, b_test = np.digitize(gsize[cal], edges), np.digitize(gsize[test], edges)
        q_global = conformal_q(r_cal, ALPHA)
        lo, hi = np.empty_like(pred_test), np.empty_like(pred_test)
        for b in np.unique(b_test):
            m_cal, m_test = b_cal == b, b_test == b
            qb = conformal_q(r_cal[m_cal], ALPHA) if m_cal.sum() >= 50 else q_global
            lo[m_test], hi[m_test] = pred_test[m_test] - qb, pred_test[m_test] + qb
        return lo, hi

    raise ValueError(method)


def conditional_stats(y_true, lo, hi, fam) -> dict:
    inside = (y_true >= lo) & (y_true <= hi)
    covs = np.asarray([inside[fam == f].mean() for f in np.unique(fam)
                       if (fam == f).sum() >= 5], dtype=float)
    hi_mask = y_true >= 77.0
    return dict(
        coverage=float(inside.mean()),
        width_mean=float(np.mean(hi - lo)),
        coverage_hiTc=float(inside[hi_mask].mean()) if hi_mask.any() else float("nan"),
        n_hiTc=int(hi_mask.sum()),
        n_families_eval=int(len(covs)),
        fam_cov_p10=float(np.percentile(covs, 10)) if len(covs) else float("nan"),
        frac_families_undercovered=float(np.mean(covs < 0.80)) if len(covs) else float("nan"),
    )


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    X, y, groups = z["X"], z["y"], z["groups"]

    records = []
    for regime in REGIMES:
        for seed in SEEDS:
            tr, cal, test = make_split(X, y, groups, regime, seed)
            model = XGBRegressor(**XGB).fit(X[tr], y[tr])
            pred_test = model.predict(X[test])
            mae = float(np.mean(np.abs(y[test] - pred_test)))
            for method in METHODS:
                lo, hi = intervals(method, X, y, groups, tr, cal, test, model, pred_test)
                rec = dict(regime=regime, method=method, seed=seed, mae=round(mae, 3))
                rec.update(conditional_stats(y[test], lo, hi, groups[test]))
                records.append(rec)
                print(f"{regime:7s} {method:11s} s{seed} "
                      f"cov {rec['coverage']:.3f} hiTc {rec['coverage_hiTc']:.3f} "
                      f"p10 {rec['fam_cov_p10']:.3f} w {rec['width_mean']:.1f}", flush=True)

    OUT.mkdir(exist_ok=True)
    keys = list(records[0].keys())
    with (OUT / "variants_conditional.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in records:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    summary = {}
    for regime in REGIMES:
        for method in METHODS:
            sel = [r for r in records if r["regime"] == regime and r["method"] == method]
            summary[f"{regime}|{method}"] = {
                k: round(float(np.mean([r[k] for r in sel])), 4)
                for k in ("coverage", "coverage_hiTc", "width_mean", "mae",
                          "fam_cov_p10", "frac_families_undercovered")
            }
            summary[f"{regime}|{method}"]["coverage_std"] = round(
                float(np.std([r["coverage"] for r in sel])), 4)
            summary[f"{regime}|{method}"]["coverage_hiTc_std"] = round(
                float(np.std([r["coverage_hiTc"] for r in sel])), 4)

    # width change of each variant relative to split-conformal, per regime
    deltas = {}
    for regime in REGIMES:
        base = summary[f"{regime}|split"]["width_mean"]
        for method in ("normalized", "mondrian"):
            wm = summary[f"{regime}|{method}"]["width_mean"]
            deltas[f"{regime}|{method}"] = round(100.0 * (wm - base) / base, 2)

    (OUT / "variants_conditional.json").write_text(json.dumps(
        {"target_coverage": 1 - ALPHA, "summary": summary,
         "width_change_pct_vs_split": deltas}, indent=2))
    print(json.dumps({"summary": summary, "width_change_pct_vs_split": deltas}, indent=2))


if __name__ == "__main__":
    main()
