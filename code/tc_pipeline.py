"""
tc_pipeline.py — Reproducible Tc-prediction pipeline with chemical-leakage-aware validation.

Companion code for: "Machine Learning applied to Energy Superconducting Materials:
An Ontological Approach" (Cardona Osorio) — extends the Random-Forest/XGBoost Tc module
(ontology stage 1, 'Molecular & Compositional Design') with group-aware validation that
answers specific objective OE4 ('validate ... using performance metrics').

Key idea: the UCI/Hamidieh SuperCon-derived dataset is dominated by doping series
(near-duplicate compositions). A random train/test split leaks these near-twins across
the split and INFLATES R2. Holding out whole chemical FAMILIES (materials sharing the
same element set) measures honest generalization to new chemistries.

Data: UCI Superconductivity Data Set #464
  train.csv     -> 21,263 x 81 compositional features + critical_temp  (Dataset A)
  unique_m.csv  -> per-element fractions + chemical formula (row-aligned to train.csv)

Usage:
  python tc_pipeline.py --data_dir superconduct_data --out results/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, KFold, GroupKFold,
                                     GroupShuffleSplit)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor


# ----- model factory (hyperparameters from the IEEE paper's GridSearchCV optima) -----
def make_models(seed: int = 42) -> dict:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=25, max_features="sqrt",
            min_samples_leaf=3, min_samples_split=5, n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(
            n_estimators=250, max_depth=8, learning_rate=0.07, subsample=0.9,
            colsample_bytree=0.7, gamma=0, reg_alpha=0.01, n_jobs=-1,
            random_state=seed, tree_method="hist"),
    }


def metrics(yt, yp) -> dict:
    return dict(MAE=float(mean_absolute_error(yt, yp)),
                RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
                R2=float(r2_score(yt, yp)))


def chemical_families(unique_m: pd.DataFrame) -> np.ndarray:
    """Assign each material a group id = frozenset of its constituent elements.
    Materials in the same doping series (same elements, different fractions) share a group."""
    elem_cols = [c for c in unique_m.columns if c not in ("critical_temp", "material")]
    present = unique_m[elem_cols] > 0
    fams = present.apply(lambda r: frozenset(e for e in elem_cols if r[e]), axis=1)
    # Deterministic integer ids via factorize (NOT hash(): str hashing is per-process
    # randomized, which would make group splits non-reproducible across runs).
    key = fams.apply(lambda f: "|".join(sorted(f)))
    return pd.factorize(key)[0]


def cv_predict_manual(model_fn, X, y, cv, groups=None) -> np.ndarray:
    """Out-of-fold predictions without joblib's process executor (sandbox-safe)."""
    yp = np.empty_like(y, dtype=float)
    splitter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for tr, te in splitter:
        m = model_fn(); m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te])
    return yp


def topk_precision(X, y, tr, te, K=100, seed=42) -> float:
    """Of the K materials with highest PREDICTED Tc in the test pool, what fraction
    are truly in the top decile (high-Tc) — i.e. screening usefulness, not pointwise error."""
    m = make_models(seed)["XGBoost"]; m.fit(X[tr], y[tr])
    yp = m.predict(X[te]); thr = np.percentile(y[tr], 90)
    return float((y[te][np.argsort(yp)[-K:]] >= thr).mean())


def run(data_dir: str, out: str, seed: int = 42) -> dict:
    data_dir, out = Path(data_dir), Path(out); out.mkdir(parents=True, exist_ok=True)
    feat = pd.read_csv(data_dir / "train.csv")
    uniq = pd.read_csv(data_dir / "unique_m.csv")
    assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "files not row-aligned"

    X = feat.drop(columns=["critical_temp"]).values
    y = feat["critical_temp"].values
    groups = chemical_families(uniq)

    res = {"n_materials": int(len(X)), "n_families": int(len(set(groups))), "single": {}, "cv": {}}

    # single split: random vs grouped
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    g_tr, g_te = next(gss.split(X, y, groups))
    assert not (set(groups[g_tr]) & set(groups[g_te])), "family overlap!"
    for name, m in make_models(seed).items():
        m.fit(Xtr, ytr); res["single"][f"random_{name}"] = metrics(yte, m.predict(Xte))
        m2 = make_models(seed)[name]; m2.fit(X[g_tr], y[g_tr])
        res["single"][f"grouped_{name}"] = metrics(y[g_te], m2.predict(X[g_te]))

    # 5-fold CV: random vs grouped
    for name in ("RandomForest", "XGBoost"):
        for scheme, cv, grp in [("random", KFold(5, shuffle=True, random_state=seed), None),
                                ("grouped", GroupKFold(5), groups)]:
            yp = cv_predict_manual(lambda n=name: make_models(seed)[n], X, y, cv, grp)
            res["cv"][f"{scheme}_{name}"] = metrics(y, yp)

    # compositional distance + screening precision
    Xs = StandardScaler().fit_transform(X)
    nn = lambda a, b: NearestNeighbors(n_neighbors=1).fit(Xs[a]).kneighbors(Xs[b])[0].ravel()
    ri = np.arange(len(X)); np.random.RandomState(seed).shuffle(ri); cut = int(0.8 * len(X))
    res["nn_dist_median"] = {"random": float(np.median(nn(ri[:cut], ri[cut:]))),
                             "grouped": float(np.median(nn(g_tr, g_te)))}
    # Screening precision is sensitive to WHICH families are held out, so report the
    # distribution across the 5 grouped folds (mean +/- std), not a single split.
    rand_p, grp_p = [], []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        rand_p.append(topk_precision(X, y, tr, te, seed=seed))
    for tr, te in GroupKFold(5).split(X, y, groups):
        grp_p.append(topk_precision(X, y, tr, te, seed=seed))
    res["topk100_precision"] = {
        "random_mean": float(np.mean(rand_p)), "random_std": float(np.std(rand_p)),
        "grouped_mean": float(np.mean(grp_p)), "grouped_std": float(np.std(grp_p)),
        "grouped_folds": [float(p) for p in grp_p]}

    json.dump(res, open(out / "experiment_summary.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="superconduct_data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    r = run(**vars(ap.parse_args()))
    print(json.dumps(r, indent=2))
