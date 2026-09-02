"""Seed experiment 2: the deployment regime for conformal Tc intervals.

Regime "deploy" is the one that matters for discovery and that regime 1 did not
test: the practitioner calibrates on data exchangeable with their TRAINING pool
(random calibration split) and then predicts on genuinely NEW chemical families.
Split-conformal's exchangeability assumption is violated exactly here.

Three regimes, 90% target coverage, XGBoost on Dataset B (132 Magpie features):

  iid      train / calib / test all random                       (textbook)
  shift    train / calib / test all family-disjoint              (regime 1)
  deploy   train + calib random, test = unseen families          (realistic)

For each we report marginal coverage plus CONDITIONAL coverage: per-family
coverage spread, and coverage restricted to the high-Tc region (>= 77 K) that
screening actually targets.

Outputs: results_seed/deploy_conformal.{csv,json}
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

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))  # repo root
NPZ = ROOT / "data" / "datasetB_featurized.npz"
OUT = ROOT / "results_seed"


def conformal_q(residuals: np.ndarray, alpha: float) -> float:
    n = len(residuals)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(residuals)[k - 1])


def make_split(X, y, groups, regime: str, seed: int):
    """Return (train_idx, calib_idx, test_idx) for one regime."""
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
        # test = unseen families; train and calib are a RANDOM split of the rest
        rest, test = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed)
                          .split(X, y, groups))
        tr, cal = train_test_split(rest, test_size=0.30, random_state=seed + 100)
        return tr, cal, test
    raise ValueError(regime)


def conditional_stats(y_true, lo, hi, fam, seed_regime: dict) -> dict:
    inside = (y_true >= lo) & (y_true <= hi)
    out = dict(seed_regime)
    out["coverage"] = float(inside.mean())
    out["width_mean"] = float(np.mean(hi - lo))

    # per-family conditional coverage (families with >= 5 test members)
    covs = [inside[fam == f].mean() for f in np.unique(fam) if (fam == f).sum() >= 5]
    covs = np.asarray(covs, dtype=float)
    out["n_families_eval"] = int(len(covs))
    out["fam_cov_median"] = float(np.median(covs)) if len(covs) else float("nan")
    out["fam_cov_p10"] = float(np.percentile(covs, 10)) if len(covs) else float("nan")
    out["frac_families_undercovered"] = float(np.mean(covs < 0.80)) if len(covs) else float("nan")

    hi_mask = y_true >= 77.0
    out["coverage_hiTc"] = float(inside[hi_mask].mean()) if hi_mask.any() else float("nan")
    out["n_hiTc"] = int(hi_mask.sum())
    return out


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    X, y, groups = z["X"], z["y"], z["groups"]

    records = []
    for regime in ("iid", "shift", "deploy"):
        for seed in SEEDS:
            tr, cal, test = make_split(X, y, groups, regime, seed)
            model = XGBRegressor(**XGB).fit(X[tr], y[tr])
            q = conformal_q(np.abs(y[cal] - model.predict(X[cal])), ALPHA)
            pred = model.predict(X[test])
            rec = conditional_stats(
                y[test], pred - q, pred + q, groups[test],
                dict(regime=regime, seed=seed, q=round(q, 3),
                     mae=round(float(np.mean(np.abs(y[test] - pred))), 3)))
            records.append(rec)
            print(regime, seed, "cov", round(rec["coverage"], 3),
                  "hiTc", round(rec["coverage_hiTc"], 3), "q", round(q, 2), flush=True)

    OUT.mkdir(exist_ok=True)
    keys = list(records[0].keys())
    with (OUT / "deploy_conformal.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in records:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    summary = {}
    for regime in ("iid", "shift", "deploy"):
        sel = [r for r in records if r["regime"] == regime]
        summary[regime] = {
            k: round(float(np.mean([r[k] for r in sel])), 4)
            for k in ("coverage", "coverage_hiTc", "width_mean", "q", "mae",
                      "fam_cov_median", "fam_cov_p10", "frac_families_undercovered")
        }
        summary[regime]["coverage_std"] = round(
            float(np.std([r["coverage"] for r in sel])), 4)
    (OUT / "deploy_conformal.json").write_text(
        json.dumps({"target_coverage": 1 - ALPHA, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
