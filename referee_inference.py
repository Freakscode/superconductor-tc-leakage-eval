"""Cluster-robust inference for the referee re-analysis.

Two questions the seed-level summaries cannot answer:

I1. Is the norm_cv improvement over split real once the correlated unit
    (chemical family) is respected? Compares them with a PAIRED family
    bootstrap - both methods scored on the same resampled families, so the
    family-composition noise that dominates the marginal CIs cancels.

I2. Does the Simpson reversal survive cluster-robust inference and a proper
    control for the confounder? The original conditioned by splitting on
    predicted Tc and re-binning distance, which controls coarsely. Here:
      (a) family-bootstrap CI on the near-vs-far coverage gap within high-Tc
      (b) logistic regression of violation on nn_dist with pred_tc as a
          covariate, refit on family-bootstrap resamples

Reads results_referee/{methods_rows,deploy_materials}.csv (produced by
referee_reanalysis.py) plus a re-run of the two interval methods to get
per-material inside/outside flags for the paired test.

Outputs: results_referee/inference.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split
from xgboost import XGBRegressor

ALPHA = 0.10
SEEDS = (0, 1, 2, 3, 4)
HI_TC = 77.0
N_BOOT = 2000
XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9,
           colsample_bytree=0.9, tree_method="hist", n_jobs=4, random_state=42)
XGB_AUX = {**XGB, "n_estimators": 300}

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "superconductor-tc-leakage-eval" / "data"
OUT = ROOT / "results_referee"


def conformal_q(s, alpha=ALPHA):
    n = len(s)
    return float(np.sort(s)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])


def load(ds):
    if ds == "B":
        z = np.load(DATA / "datasetB_featurized.npz", allow_pickle=True)
        return z["X"], z["y"], z["groups"]
    train = pd.read_csv(DATA / "train.csv")
    uniq = pd.read_csv(DATA / "unique_m.csv")
    y = train["critical_temp"].to_numpy(float)
    X = train.drop(columns=["critical_temp"]).to_numpy(np.float32)
    ecols = [c for c in uniq.columns if c not in ("critical_temp", "material")]
    keys = ["-".join(sorted(np.asarray(ecols)[r])) for r in uniq[ecols].to_numpy() > 0]
    return X, y, pd.factorize(pd.Series(keys))[0]


def paired_records(ds):
    """Per-material inside flags for split and norm_cv, deployment regime."""
    X, y, g = load(ds)
    frames = []
    for seed in SEEDS:
        rest, test = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(X, y, g))
        tr, cal = train_test_split(rest, test_size=0.30, random_state=seed + 100)
        model = XGBRegressor(**XGB).fit(X[tr], y[tr])
        pred = model.predict(X[test])
        r_cal = np.abs(y[cal] - model.predict(X[cal]))

        q = conformal_q(r_cal)
        in_split = (np.abs(y[test] - pred) <= q)

        oof = np.empty(len(tr))
        for a, b in KFold(3, shuffle=True, random_state=0).split(tr):
            fm = XGBRegressor(**XGB_AUX).fit(X[tr[a]], y[tr[a]])
            oof[b] = np.abs(y[tr[b]] - fm.predict(X[tr[b]]))
        sig = XGBRegressor(**XGB_AUX).fit(X[tr], oof + 1e-6)
        s_cal = np.clip(sig.predict(X[cal]), 1e-3, None)
        s_te = np.clip(sig.predict(X[test]), 1e-3, None)
        qn = conformal_q(r_cal / s_cal)
        in_norm = (np.abs(y[test] - pred) <= qn * s_te)

        frames.append(pd.DataFrame(dict(
            seed=seed, group=g[test], y=y[test], pred=pred,
            in_split=in_split.astype(int), in_norm=in_norm.astype(int),
            w_split=2 * q, w_norm=2 * qn * s_te)))
        print(f"  {ds} seed{seed} paired records done", flush=True)
    return pd.concat(frames, ignore_index=True)


def family_bootstrap(df, stat_fn, n_boot=N_BOOT, seed=0):
    """Percentile CI of stat_fn(resampled df), resampling families with replacement."""
    fams = df["group"].to_numpy()
    ufam = np.unique(fams)
    idx_by_fam = {f: np.flatnonzero(fams == f) for f in ufam}
    rs = np.random.RandomState(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        draw = rs.choice(ufam, size=len(ufam), replace=True)
        idx = np.concatenate([idx_by_fam[f] for f in draw])
        vals[b] = stat_fn(df.iloc[idx])
    vals = vals[np.isfinite(vals)]
    return dict(mean=float(vals.mean()),
                ci=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                p_two_sided_gt0=float(2 * min((vals <= 0).mean(), (vals >= 0).mean())))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    result = {}

    for ds in ("B", "A"):
        rec = paired_records(ds)
        rec.to_csv(OUT / f"paired_{ds}.csv", index=False)
        hi = rec[rec.y >= HI_TC]

        # --- I1: paired family bootstrap, norm_cv minus split ---
        result[f"{ds}|paired_hiTc_norm_minus_split"] = family_bootstrap(
            hi, lambda d: d.in_norm.mean() - d.in_split.mean())
        result[f"{ds}|paired_all_norm_minus_split"] = family_bootstrap(
            rec, lambda d: d.in_norm.mean() - d.in_split.mean())
        result[f"{ds}|width_hiTc_norm_over_split"] = float(
            hi.w_norm.mean() / hi.w_split.mean())
        result[f"{ds}|width_lowTc_norm_over_split"] = float(
            rec[rec.y < HI_TC].w_norm.mean() / rec[rec.y < HI_TC].w_split.mean())

        # --- I2: Simpson reversal, cluster-robust ---
        mats = pd.read_csv(OUT / "deploy_materials.csv")
        m = mats[mats.dataset == ds].copy()
        m_hi = m[m.pred >= 40.0].copy()
        if len(m_hi) > 300:
            terc = m_hi.groupby("seed")["nn_dist"].transform(
                lambda s: pd.qcut(s, 3, labels=False, duplicates="drop"))
            m_hi = m_hi.assign(terc=terc)
            gap = lambda d: (1 - d[d.terc == 2].violated.mean()) - (1 - d[d.terc == 0].violated.mean())
            result[f"{ds}|simpson_far_minus_near_hiTc"] = family_bootstrap(m_hi, gap)

        # logistic: violation ~ nn_dist + pred, family bootstrap on the nn_dist coef
        def coef(d):
            from sklearn.linear_model import LogisticRegression
            Z = np.column_stack([d.nn_dist.to_numpy(), d.pred.to_numpy()])
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
            v = d.violated.to_numpy()
            if len(np.unique(v)) < 2:
                return np.nan
            return float(LogisticRegression(max_iter=400).fit(Z, v).coef_[0][0])
        result[f"{ds}|logit_nn_coef_adjusted_for_pred"] = family_bootstrap(
            m, coef, n_boot=300)

        def coef_unadj(d):
            from sklearn.linear_model import LogisticRegression
            Z = d.nn_dist.to_numpy().reshape(-1, 1)
            Z = (Z - Z.mean()) / (Z.std() + 1e-12)
            v = d.violated.to_numpy()
            if len(np.unique(v)) < 2:
                return np.nan
            return float(LogisticRegression(max_iter=400).fit(Z, v).coef_[0][0])
        result[f"{ds}|logit_nn_coef_unadjusted"] = family_bootstrap(
            m, coef_unadj, n_boot=300)
        print(f"{ds} inference done", flush=True)

    (OUT / "inference.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
