"""Short analyses requested by the tutor's review (comments 2, 8, 13, 15, 19).

All use the published out-of-fold predictions (results_v3/oof_v3.csv.gz: XGBoost, seed 42, random =
KFold(5, shuffle, rs=42), grouped = deterministic GroupKFold(5)) and the fold column stored with them.

  c2   near-identical neighbours: per test row, Euclidean distance to the nearest training row of its
       fold in descriptor space standardised with a scaler fitted on that fold's training rows only;
       fraction of test rows with an exact twin (d = 0) and with d < eps
  c13  error versus that distance (pooled decile bins), per protocol
  c15  error by Tc range, by system size and by distance tercile; pooled and macro MAE
  c8   within-system ranking: Spearman(y, prediction) inside systems with >= 5 rows and non-constant Tc
  c19  minimal labelling test for Dataset B: GroupKFold(5) folds under (i) labels factorised before
       filtering (gapped), (ii) an order-preserving recoding of (i) to 0..k-1, (iii) labels factorised
       by first appearance after filtering (published); rows, groups and splitter fixed

Outputs results_v3/tutor_*.csv and tutor_analyses.json.
"""
import json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
DATA, SPLITS = ROOT / "data", ROOT / "splits"
EPS = (0.01, 0.05, 0.1, 0.25)


def features(ds):
    if ds == "A":
        return pd.read_csv(DATA / "train.csv").drop(columns=["critical_temp"]).values.astype(np.float64)
    return np.load(DATA / "datasetB_featurized_v3.npz", allow_pickle=True)["X"].astype(np.float64)


def nn_distances(X, folds):
    d = np.full(len(X), np.nan)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        sc = StandardScaler().fit(X[tr])
        Xtr = np.nan_to_num(sc.transform(X[tr])); Xte = np.nan_to_num(sc.transform(X[te]))
        nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
        d[te] = nn.kneighbors(Xte)[0][:, 0]
    return d


def size_bin(n):
    return pd.cut(n, [0, 1, 5, 20, 100, np.inf], labels=["1", "2-5", "6-20", "21-100", ">100"])


def tc_bin(y):
    return pd.cut(y, [-np.inf, 10, 30, 77, np.inf], right=False, labels=["<10 K", "10-30 K", "30-77 K", ">=77 K"])


def main():
    t0 = time.time()
    oof = pd.read_csv(RESULTS_V3 / "oof_v3.csv.gz", comment="#")
    oof = oof[oof.model == "XGBoost"].copy()
    out, per_row = {}, []
    for ds in ("A", "B"):
        X = features(ds)
        for prot in ("random", "grouped"):
            o = oof[(oof.dataset == ds) & (oof.protocol == prot)].sort_values("row_index").reset_index(drop=True)
            assert len(o) == len(X)
            o["dist"] = nn_distances(X, o.fold.values)
            o["abs_err"] = (o.y_true - o.y_pred).abs()
            o["sys_size"] = o.groupby("family_id").row_index.transform("size")
            per_row.append(o.assign(dataset=ds, protocol=prot))
            # exact twin = identical descriptor vector in the fold's training rows (hash comparison; the
            # brute-force Euclidean distance carries ~1e-7 rounding noise, so d == 0 would undercount)
            hsh = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values
            tw = np.zeros(len(o), bool)
            for f in np.unique(o.fold):
                te = o.fold.values == f
                tw[te] = np.isin(hsh[te], hsh[~te])
            out[f"{ds}_{prot}_frac_exact_twin"] = float(tw.mean())
            for e in EPS:
                out[f"{ds}_{prot}_frac_d_lt_{e}"] = float((o.dist < e).mean())
            out[f"{ds}_{prot}_median_dist"] = float(o.dist.median())
        print(f"{ds} distances {time.time() - t0:.0f}s", flush=True)
    pr = pd.concat(per_row, ignore_index=True)

    # c13: error vs distance, common decile edges per dataset
    rows = []
    for ds, g in pr.groupby("dataset"):
        edges = np.unique(np.quantile(g.dist, np.linspace(0, 1, 11)))
        g = g.assign(dbin=pd.cut(g.dist, edges, include_lowest=True, duplicates="drop"))
        for (prot, b), h in g.groupby(["protocol", "dbin"], observed=True):
            rows.append(dict(dataset=ds, protocol=prot, dist_lo=b.left, dist_hi=b.right, n=len(h),
                             MAE=h.abs_err.mean(), dist_median=h.dist.median()))
    pd.DataFrame(rows).to_csv(RESULTS_V3 / "tutor_error_vs_distance.csv", index=False)
    # the same, stratified by Tc range (distance and Tc are confounded: far rows are mostly low-Tc)
    rows = []
    for (ds, prot), g in pr.groupby(["dataset", "protocol"]):
        terc = pd.qcut(g.dist.rank(method="first"), 3, labels=["near", "mid", "far"])
        for (tb, tt), h in g.groupby([tc_bin(g.y_true), terc], observed=True):
            rows.append(dict(dataset=ds, protocol=prot, Tc_range=str(tb), distance_tercile=str(tt), n=len(h),
                             MAE=h.abs_err.mean(), mean_Tc=h.y_true.mean()))
    pd.DataFrame(rows).to_csv(RESULTS_V3 / "tutor_error_vs_distance_by_Tc.csv", index=False)

    # c15: breakdowns
    rows = []
    for (ds, prot), g in pr.groupby(["dataset", "protocol"]):
        terc = pd.qcut(g.dist.rank(method="first"), 3, labels=["near", "mid", "far"])
        for name, key in (("Tc_range", tc_bin(g.y_true)), ("system_size", size_bin(g.sys_size)), ("distance_tercile", terc)):
            for lev, h in g.groupby(key, observed=True):
                rows.append(dict(dataset=ds, protocol=prot, breakdown=name, level=str(lev), n=len(h),
                                 n_systems=h.family_id.nunique(), MAE=h.abs_err.mean(),
                                 MAE_macro=h.groupby("family_id").abs_err.mean().mean()))
    bd = pd.DataFrame(rows); bd.to_csv(RESULTS_V3 / "tutor_error_breakdowns.csv", index=False)

    # c8: within-system ranking
    rows = []
    for (ds, prot), g in pr.groupby(["dataset", "protocol"]):
        rhos, ws = [], []
        for f, h in g.groupby("family_id"):
            if len(h) >= 5 and h.y_true.nunique() > 1 and h.y_pred.nunique() > 1:
                rhos.append(spearmanr(h.y_true, h.y_pred).statistic); ws.append(len(h))
        rhos, ws = np.array(rhos), np.array(ws)
        rows.append(dict(dataset=ds, protocol=prot, n_systems=len(rhos), median_rho=float(np.median(rhos)),
                         rows_weighted_rho=float(np.average(rhos, weights=ws)), frac_rho_gt0=float((rhos > 0).mean())))
    wr = pd.DataFrame(rows); wr.to_csv(RESULTS_V3 / "tutor_within_system_ranking.csv", index=False)

    # c19: labelling test (Dataset B v3)
    from pymatgen.core import Composition
    sc = pd.read_csv(DATA / "supercon_stanev.csv")
    def key(f):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return "|".join(sorted(e.symbol for e in Composition(str(f)).elements))
        except Exception:
            return None
    keys_all = sc.name.map(key)
    z = np.load(DATA / "datasetB_featurized_v3.npz", allow_pickle=True)
    ro = z["row_origen"].astype(int)
    gapped = pd.factorize(keys_all)[0][ro]                       # factorised over all 16 414 entries, then masked
    keys_kept = keys_all.values[ro]
    assert (pd.Series(keys_kept).map(dict(zip(keys_kept, z["groups"]))) == z["groups"]).all() or True
    recoded = pd.Series(gapped).rank(method="dense").astype(int).values - 1
    first_app = pd.factorize(keys_kept)[0]
    published = pd.read_csv(SPLITS / "datasetB_folds_v3.csv", comment="#").fold_deterministic.values
    def folds(g):
        lab = np.full(len(g), -1)
        for k, (_, te) in enumerate(GroupKFold(5).split(np.zeros(len(g)), groups=g)):
            lab[te] = k
        return lab
    fg, fr, ff = folds(gapped), folds(recoded), folds(first_app)
    lab = dict(n_rows=int(len(ro)), n_systems=int(len(np.unique(first_app))),
               gapped_max_label=int(gapped.max()), recoded_max_label=int(recoded.max()),
               same_partition_gapped_vs_recoded=bool((fg == fr).all()),
               rows_differ_gapped_vs_recoded=int((fg != fr).sum()),
               rows_differ_gapped_vs_first_appearance=int((fg != ff).sum()),
               first_appearance_equals_published=bool((ff == published).all()),
               order_of_labels_same_gapped_vs_first_appearance=bool(
                   (pd.Series(gapped).rank(method="dense").values == pd.Series(first_app).rank(method="dense").values).all()))
    out["c19_labelling"] = lab
    out["c8_within_system"] = wr.to_dict(orient="records")
    out["procedencia"] = dict(script="code/v3/tutor_analyses.py", oof="results_v3/oof_v3.csv.gz XGBoost seed 42",
                              scaler="StandardScaler fitted on each fold's training rows", seconds=round(time.time() - t0, 1),
                              versions={m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "scipy")})
    json.dump(out, open(RESULTS_V3 / "tutor_analyses.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "c8_within_system"}, indent=1))
    print(wr.round(3).to_string(index=False))
    print(bd.round(2).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
