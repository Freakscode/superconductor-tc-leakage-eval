"""
datasetB_pipeline.py — Reproduce paper "Dataset B" (Stanev/Magpie) and run the
chemical-leakage-aware validation, mirroring tc_pipeline.py for Dataset A.

Dataset B source: Stanev et al. SuperCon list (github.com/vstanev1/Supercon),
Magpie-featurized with matminer. The paper's high R2 (0.931) is reproduced on the
Tc>0 SUPERCONDUCTOR subset with the paper's Dataset-B hyperparameters (Table IV);
the full set including Tc=0 non-superconductors is a different, harder task.

Single-threaded throughout: matminer's multiprocessing pool and sklearn's joblib
loky backend both deadlock under restricted semaphore limits — featurize and CV in
plain loops (n_jobs=1).

Usage:
  python datasetB_pipeline.py --supercon supercon_stanev.csv --out resultsB/
Requires: matminer, pymatgen, xgboost, scikit-learn.
"""
from __future__ import annotations
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.model_selection import (train_test_split, KFold, GroupKFold, GroupShuffleSplit)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor


def make_models(seed=42):
    # Paper Table IV (Dataset B) optima.
    return {
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=30,
            min_samples_leaf=3, min_samples_split=5, n_jobs=1, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=seed, tree_method="hist"),
    }

def metrics(yt, yp):
    return dict(MAE=float(mean_absolute_error(yt, yp)),
                RMSE=float(np.sqrt(mean_squared_error(yt, yp))), R2=float(r2_score(yt, yp)))

def cv_predict_manual(model_fn, X, y, cv, groups=None):
    yp = np.empty_like(y, dtype=float)
    sp = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for tr, te in sp:
        m = model_fn(); m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te])
    return yp


def family_codes(formulas) -> np.ndarray:
    """Chemical-family id = constituent-element set (same rule as tc_pipeline.chemical_families).
    Integer codes come from pandas.factorize over the sorted "|"-joined element symbols, so
    they are deterministic across processes (no str hashing)."""
    key = pd.Series(["|".join(sorted(e.symbol for e in Composition(f).elements)) for f in formulas])
    return pd.factorize(key)[0]


def featurize(supercon_csv: str, superconductors_only: bool = True):
    """Magpie-featurize Stanev formulas, single-threaded. Returns X, y, groups, formulas.

    superconductors_only=True keeps the Tc>0 rows (the paper's Dataset B basis, 12 440 rows)
    and computes the family codes ON THE FILTERED rows.

    v2 FIX (factorize-after-filter). In v1 this function factorized the family key over ALL
    featurized rows (16 406, 4 191 families) and run() applied the Tc>0 mask afterwards,
    leaving 3 063 surviving families with non-contiguous codes (gaps in 0..4190). The
    partition (which rows share a family) is unaffected, but GroupKFold assigns groups to
    folds by sorted group size with ties broken by code order, so the gapped codes produce a
    DIFFERENT fold assignment than the contiguous codes stored as `groups` in
    data/datasetB_featurized.npz (~1 800 of 2 488 rows per test fold change fold). Every v1
    Dataset-B number produced by running this script live therefore came from a different
    5-fold partition than the distributed checkpoint. Factorizing after the mask yields codes
    that are np.array_equal to the .npz `groups` and identical GroupKFold(5) test indices
    (see results_v2/fix_factorize_verificacion.json and splits/datasetB_folds.csv).
    """
    sc = pd.read_csv(supercon_csv)                       # columns: name, Tc
    ep = ElementProperty.from_preset("magpie"); labels = ep.feature_labels()
    feats, tcs, formulas = [], [], []
    t0 = time.time()
    for name, tc in zip(sc["name"], sc["Tc"]):
        try:
            f = ep.featurize(Composition(name)); feats.append(f); tcs.append(tc); formulas.append(name)
        except Exception:
            pass
    print(f"featurized {len(feats)}/{len(sc)} formulas in {time.time()-t0:.0f}s")
    B = pd.DataFrame(feats, columns=labels); B["critical_temp"] = tcs; B["formula"] = formulas
    B = B.dropna().reset_index(drop=True)
    if superconductors_only:                              # filter FIRST ...
        B = B[B["critical_temp"] > 0].reset_index(drop=True)
    X = B[labels].values; y = B["critical_temp"].values
    groups = family_codes(B["formula"].values)            # ... THEN factorize (contiguous codes)
    return X, y, groups, B["formula"].values


def run(supercon: str, out: str, seed: int = 42, superconductors_only: bool = True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # The Tc>0 mask is applied INSIDE featurize() so that family codes are factorized on the
    # filtered rows (v2 fix; see featurize docstring). Masking here after factorize would
    # reproduce the v1 fold-assignment bug.
    X, y, groups, formulas = featurize(supercon, superconductors_only=superconductors_only)
    res = {"n_featurized": int(len(X)), "n_features": int(X.shape[1])}
    if superconductors_only:                              # reproduce the paper's R2~0.93 basis
        res["basis"] = "Tc>0 superconductors"
    res["n_used"] = int(len(X)); res["n_families"] = int(len(set(groups)))

    res["cv"] = {}
    for name in ("RandomForest", "XGBoost"):
        for scheme, cv, grp in [("random", KFold(5, shuffle=True, random_state=seed), None),
                                ("grouped", GroupKFold(5), groups)]:
            yp = cv_predict_manual(lambda n=name: make_models(seed)[n], X, y, cv, grp)
            res["cv"][f"{scheme}_{name}"] = metrics(y, yp)

    # distance + screening
    Xs = StandardScaler().fit_transform(X)
    nn = lambda a, b: NearestNeighbors(n_neighbors=1).fit(Xs[a]).kneighbors(Xs[b])[0].ravel()
    ri = np.arange(len(X)); np.random.RandomState(seed).shuffle(ri); cut = int(0.8*len(X))
    g_tr, g_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, groups))
    res["nn_dist_median"] = {"random": float(np.median(nn(ri[:cut], ri[cut:]))),
                             "grouped": float(np.median(nn(g_tr, g_te)))}
    def topk(tr, te, K=100):
        m = make_models(seed)["XGBoost"]; m.fit(X[tr], y[tr]); yp = m.predict(X[te])
        return float((y[te][np.argsort(yp)[-K:]] >= np.percentile(y[tr], 90)).mean())
    rp = [topk(tr, te) for tr, te in KFold(5, shuffle=True, random_state=seed).split(X)]
    gp = [topk(tr, te) for tr, te in GroupKFold(5).split(X, y, groups)]
    res["topk100"] = {"random_mean": float(np.mean(rp)), "grouped_mean": float(np.mean(gp)),
                      "grouped_std": float(np.std(gp))}
    json.dump(res, open(out / "datasetB_summary.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--supercon", default="supercon_stanev.csv")
    ap.add_argument("--out", default="resultsB")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all_materials", action="store_true",
                    help="include Tc=0 non-superconductors (harder task, lower R2)")
    a = ap.parse_args()
    print(json.dumps(run(a.supercon, a.out, a.seed, not a.all_materials), indent=2))
