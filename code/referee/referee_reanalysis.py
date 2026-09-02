"""Referee re-analysis of the Paper 2 seed experiments.

Audits three things the original seed scripts did not separate:

R1. DECOMPOSITION. The original framing attributes the low high-Tc coverage
    (0.505) to distribution shift. But the iid regime already shows 0.723
    high-Tc coverage with ZERO shift, so part of the deficit is heteroskedasticity
    interacting with a CONSTANT-WIDTH interval, not exchangeability failure.
    Two oracles separate them:
      oracle_global : q from the TEST residuals. Marginal coverage is exactly
                      nominal by construction; whatever high-Tc deficit remains
                      is pure heteroskedasticity, unfixable by recalibration
                      of a constant-width band.
      oracle_cond   : separate q for the high-Tc and low-Tc test subsets. Shows
                      what a perfectly conditional method could achieve.

R2. REFUTATION TEST on "the textbook fixes fail". The original normalized
    conformal fit sigma(x) on IN-SAMPLE training residuals of a depth-8,
    600-tree XGBoost. Those residuals are memorization artefacts, near zero and
    uninformative about difficulty, so the variant was crippled by
    implementation, not by the problem. Three honest difficulty estimates are
    compared against it, plus conformalized quantile regression (CQR) - the
    standard adaptive-interval method the proposal never tested.

R3. CLUSTER-ROBUST UNCERTAINTY. Test materials are clustered by chemical
    family and coverage is near-constant within family, so the naive binomial
    SE (0.006 at n~3900) is far too small. Coverage CIs come from a bootstrap
    that resamples FAMILIES, not materials.

Also fixes a leak: the original Mondrian binning computed family size with
np.bincount over ALL rows including test.

Outputs: results_referee/methods_summary.json, methods_rows.csv,
         deploy_materials_{A,B}.csv (per-material records for the Simpson audit)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ALPHA = 0.10
SEEDS = (0, 1, 2, 3, 4)
HI_TC = 77.0
N_BOOT = 400
XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9,
           colsample_bytree=0.9, tree_method="hist", n_jobs=4, random_state=42)
XGB_AUX = {**XGB, "n_estimators": 300}

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))  # repo root
DATA = ROOT / "data"
OUT = ROOT / "results_referee"


def conformal_q(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    return float(np.sort(scores)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])


def load_b():
    z = np.load(DATA / "datasetB_featurized.npz", allow_pickle=True)
    return z["X"], z["y"], z["groups"]


def load_a():
    train = pd.read_csv(DATA / "train.csv")
    uniq = pd.read_csv(DATA / "unique_m.csv")
    y = train["critical_temp"].to_numpy(float)
    X = train.drop(columns=["critical_temp"]).to_numpy(np.float32)
    ecols = [c for c in uniq.columns if c not in ("critical_temp", "material")]
    present = uniq[ecols].to_numpy() > 0
    keys = ["-".join(sorted(np.asarray(ecols)[row])) for row in present]
    return X, y, pd.factorize(pd.Series(keys))[0]


def make_split(X, y, groups, regime, seed):
    if regime == "iid":
        from sklearn.model_selection import ShuffleSplit
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


def cluster_bootstrap_ci(inside, fam, n_boot=N_BOOT, seed=0, subset=None):
    """Percentile CI for coverage, resampling FAMILIES (the correlated unit)."""
    if subset is not None:
        inside, fam = inside[subset], fam[subset]
    if len(inside) == 0:
        return (float("nan"), float("nan"))
    ufam = np.unique(fam)
    by_fam = {f: inside[fam == f] for f in ufam}
    rs = np.random.RandomState(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        draw = rs.choice(ufam, size=len(ufam), replace=True)
        stats[b] = np.concatenate([by_fam[f] for f in draw]).mean()
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def build_intervals(X, y, groups, tr, cal, test, methods):
    """Fit the point model once, then produce (lo, hi) for each requested method."""
    model = XGBRegressor(**XGB).fit(X[tr], y[tr])
    pred = model.predict(X[test])
    r_cal = np.abs(y[cal] - model.predict(X[cal]))
    r_test = np.abs(y[test] - pred)
    out = {}

    if "split" in methods:
        q = conformal_q(r_cal)
        out["split"] = (pred - q, pred + q)

    # --- oracles: bound what recalibration alone can achieve ---
    if "oracle_global" in methods:
        q = conformal_q(r_test)
        out["oracle_global"] = (pred - q, pred + q)
    if "oracle_cond" in methods:
        lo, hi = np.empty_like(pred), np.empty_like(pred)
        m = y[test] >= HI_TC
        for mask in (m, ~m):
            if mask.sum() >= 20:
                q = conformal_q(r_test[mask])
            else:
                q = conformal_q(r_test)
            lo[mask], hi[mask] = pred[mask] - q, pred[mask] + q
        out["oracle_cond"] = (lo, hi)

    # --- normalized conformal, three difficulty estimates ---
    def normalized(sig_tr_idx, sig_targets, q_idx, name):
        sig = XGBRegressor(**XGB_AUX).fit(X[sig_tr_idx], sig_targets + 1e-6)
        s_q = np.clip(sig.predict(X[q_idx]), 1e-3, None)
        s_te = np.clip(sig.predict(X[test]), 1e-3, None)
        qn = conformal_q(np.abs(y[q_idx] - model.predict(X[q_idx])) / s_q)
        out[name] = (pred - qn * s_te, pred + qn * s_te)

    if "norm_insample" in methods:  # reproduces the original flaw
        normalized(tr, np.abs(y[tr] - model.predict(X[tr])), cal, "norm_insample")
    if "norm_cv" in methods:        # honest: out-of-fold residuals
        oof = np.empty(len(tr))
        for a, b in KFold(3, shuffle=True, random_state=0).split(tr):
            fm = XGBRegressor(**XGB_AUX).fit(X[tr[a]], y[tr[a]])
            oof[b] = np.abs(y[tr[b]] - fm.predict(X[tr[b]]))
        normalized(tr, oof, cal, "norm_cv")
    if "norm_cal" in methods:       # honest: sigma from half the calibration set
        ca, cb = train_test_split(cal, test_size=0.5, random_state=7)
        normalized(ca, np.abs(y[ca] - model.predict(X[ca])), cb, "norm_cal")

    if "mondrian_fixed" in methods:
        # family size from TRAIN+CAL only - the original used all rows incl. test
        seen = np.concatenate([tr, cal])
        counts = pd.Series(groups[seen]).value_counts()
        gs = lambda idx: np.array([counts.get(g, 0) for g in groups[idx]])
        edges = np.array([1, 3, 10])
        b_cal, b_te = np.digitize(gs(cal), edges), np.digitize(gs(test), edges)
        qg = conformal_q(r_cal)
        lo, hi = np.empty_like(pred), np.empty_like(pred)
        for b in np.unique(b_te):
            mc, mt = b_cal == b, b_te == b
            qb = conformal_q(r_cal[mc]) if mc.sum() >= 50 else qg
            lo[mt], hi[mt] = pred[mt] - qb, pred[mt] + qb
        out["mondrian_fixed"] = (lo, hi)

    if "cqr" in methods:
        # conformalized quantile regression - the standard ADAPTIVE-width method
        qlo = XGBRegressor(**{**XGB_AUX, "objective": "reg:quantileerror",
                              "quantile_alpha": ALPHA / 2}).fit(X[tr], y[tr])
        qhi = XGBRegressor(**{**XGB_AUX, "objective": "reg:quantileerror",
                              "quantile_alpha": 1 - ALPHA / 2}).fit(X[tr], y[tr])
        lo_cal, hi_cal = qlo.predict(X[cal]), qhi.predict(X[cal])
        score = np.maximum(lo_cal - y[cal], y[cal] - hi_cal)
        qc = conformal_q(score)
        out["cqr"] = (qlo.predict(X[test]) - qc, qhi.predict(X[test]) + qc)

    return out, pred, model


def run(name, X, y, groups, regimes, methods, want_materials=False):
    rows, materials = [], []
    for regime in regimes:
        for seed in SEEDS:
            tr, cal, test = make_split(X, y, groups, regime, seed)
            iv, pred, model = build_intervals(X, y, groups, tr, cal, test, methods)
            fam, ytest = groups[test], y[test]
            hi_mask = ytest >= HI_TC

            for meth, (lo, hi) in iv.items():
                inside = (ytest >= lo) & (ytest <= hi)
                ci = cluster_bootstrap_ci(inside, fam, seed=seed)
                ci_hi = cluster_bootstrap_ci(inside, fam, seed=seed, subset=hi_mask)
                rows.append(dict(
                    dataset=name, regime=regime, method=meth, seed=seed,
                    coverage=round(float(inside.mean()), 4),
                    cov_lo=round(ci[0], 4), cov_hi=round(ci[1], 4),
                    coverage_hiTc=round(float(inside[hi_mask].mean()), 4) if hi_mask.any() else np.nan,
                    cov_hiTc_lo=round(ci_hi[0], 4), cov_hiTc_hi=round(ci_hi[1], 4),
                    width_mean=round(float(np.mean(hi - lo)), 3),
                    width_hiTc=round(float(np.mean((hi - lo)[hi_mask])), 3) if hi_mask.any() else np.nan,
                    n_test=int(len(test)), n_hiTc=int(hi_mask.sum()),
                    n_families=int(len(np.unique(fam))),
                    mae=round(float(np.mean(np.abs(ytest - pred))), 3)))

            if want_materials and regime == "deploy":
                sc = StandardScaler().fit(X[tr])
                d = NearestNeighbors(n_neighbors=1).fit(sc.transform(X[tr])) \
                    .kneighbors(sc.transform(X[test]))[0].ravel()
                lo, hi = iv["split"]
                materials.append(pd.DataFrame(dict(
                    dataset=name, seed=seed, group=fam, y=ytest, pred=pred,
                    nn_dist=d, violated=(~((ytest >= lo) & (ytest <= hi))).astype(int))))
            print(f"  {name} {regime} seed{seed} done", flush=True)
    return rows, materials


def main() -> None:
    OUT.mkdir(exist_ok=True)
    all_rows, all_mat = [], []

    Xb, yb, gb = load_b()
    print("Dataset B", Xb.shape, "families", len(np.unique(gb)), flush=True)
    r, m = run("B", Xb, yb, gb,
               regimes=("iid", "shift", "deploy"),
               methods=("split", "oracle_global", "oracle_cond", "norm_insample",
                        "norm_cv", "norm_cal", "mondrian_fixed", "cqr"),
               want_materials=True)
    all_rows += r; all_mat += m

    Xa, ya, ga = load_a()
    print("Dataset A", Xa.shape, "families", len(np.unique(ga)), flush=True)
    r, m = run("A", Xa, ya, ga,
               regimes=("iid", "deploy"),
               methods=("split", "oracle_global", "oracle_cond", "norm_cv", "cqr"),
               want_materials=True)
    all_rows += r; all_mat += m

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "methods_rows.csv", index=False)
    pd.concat(all_mat, ignore_index=True).to_csv(OUT / "deploy_materials.csv", index=False)

    summary = {}
    for (ds, rg, me), g in df.groupby(["dataset", "regime", "method"]):
        summary[f"{ds}|{rg}|{me}"] = dict(
            coverage=round(float(g.coverage.mean()), 4),
            coverage_sd=round(float(g.coverage.std(ddof=1)), 4),
            coverage_ci=[round(float(g.cov_lo.mean()), 4), round(float(g.cov_hi.mean()), 4)],
            coverage_hiTc=round(float(g.coverage_hiTc.mean()), 4),
            coverage_hiTc_sd=round(float(g.coverage_hiTc.std(ddof=1)), 4),
            coverage_hiTc_ci=[round(float(g.cov_hiTc_lo.mean()), 4),
                              round(float(g.cov_hiTc_hi.mean()), 4)],
            width_mean=round(float(g.width_mean.mean()), 3),
            width_hiTc=round(float(g.width_hiTc.mean()), 3),
            mae=round(float(g.mae.mean()), 3))
    (OUT / "methods_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
