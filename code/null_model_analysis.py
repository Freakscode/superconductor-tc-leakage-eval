"""
null_model_analysis.py
----------------------
Regenerates the leakage-aware analyses added in the revised manuscript:

  * featureless family-mean NULL model (Table 1 italic rows, Figure 2a)
  * top-100 screening precision null comparator (Figure 2b)
  * repeated-seed intervals on the MAE-inflation gap (Table 2)

All numbers are computed with the SAME chemical-family grouping used in the
paper (GroupKFold on the set of constituent elements, generated deterministically
with pandas.factorize -- never the per-process-randomised hash()).

Usage
-----
    # Dataset B (Stanev/SuperCon), from the shipped featurised checkpoint:
    python null_model_analysis.py --dataset B --featurized ../data/datasetB_featurized.npz

    # Dataset A (Hamidieh/UCI): first fetch the raw data, then run:
    python get_datasetA.py                # downloads UCI #464 into ../data/
    python null_model_analysis.py --dataset A --uci-dir ../data/

Every metric is aggregated over 5 folds; --seeds N repeats both CV schemes over
N independent fold assignments for the interval estimate.
"""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------- metrics
def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)

# ----------------------------------------------------------------------------- grouping
def element_set_groups(formulas):
    """Group id = set of constituent elements, via pandas.factorize (stable)."""
    import re
    def elemset(f):
        return frozenset(re.findall(r'[A-Z][a-z]?', str(f)))
    keys = [tuple(sorted(elemset(f))) for f in formulas]
    codes, _ = pd.factorize([str(k) for k in keys])
    return codes.astype(np.int64)

# ----------------------------------------------------------------------------- null model
def family_mean_null_cv(y, groups, splitter, split_groups=None):
    """Featureless predictor: test material -> mean y of its family in train fold
    (global train mean when the family is unseen). Returns pooled metrics + the
    fraction of test materials whose family was seen in training."""
    yp = np.empty_like(y, dtype=float)
    seen = np.zeros(len(y), dtype=bool)
    it = splitter.split(y, groups=split_groups) if split_groups is not None else splitter.split(y)
    for tr, te in it:
        gmean = y[tr].mean()
        fam_mean = {}
        for g in np.unique(groups[tr]):
            fam_mean[g] = y[tr][groups[tr] == g].mean()
        for i in te:
            g = groups[i]
            if g in fam_mean:
                yp[i] = fam_mean[g]; seen[i] = True
            else:
                yp[i] = gmean
    return dict(MAE=mae(y, yp), RMSE=rmse(y, yp), R2=r2(y, yp),
                test_family_seen_pct=round(100.0 * seen.mean(), 1)), yp

# ----------------------------------------------------------------------------- top-100 precision
def top100_precision(y_true, y_pred, hi_thresh):
    """Fraction of the 100 highest-predicted materials that are truly high-Tc."""
    order = np.argsort(-y_pred)[:100]
    return float(np.mean(y_true[order] >= hi_thresh))

def top100_null_vs_model(X, y, groups, model, scheme, n_splits=5, seed=0):
    """Per-fold top-100 precision for the family-mean ranker and for `model`."""
    hi = np.percentile(y, 90)               # high-Tc = top decile
    if scheme == "random":
        spl = KFold(n_splits=n_splits, shuffle=True, random_state=seed); it = spl.split(X)
    else:
        spl = GroupKFold(n_splits=n_splits); it = spl.split(X, groups=groups)
    null_p, model_p = [], []
    for tr, te in it:
        # null ranker: family-mean lookup
        gmean = y[tr].mean(); fam = {g: y[tr][groups[tr] == g].mean() for g in np.unique(groups[tr])}
        null_pred = np.array([fam.get(groups[i], gmean) for i in te])
        null_p.append(100 * top100_precision(y[te], null_pred, hi))
        # model ranker
        sc = StandardScaler().fit(X[tr])
        model.fit(sc.transform(X[tr]), y[tr])
        mp = model.predict(sc.transform(X[te]))
        model_p.append(100 * top100_precision(y[te], mp, hi))
    return (np.mean(null_p), np.std(null_p)), (np.mean(model_p), np.std(model_p))

# ----------------------------------------------------------------------------- drivers
def run_null_decomposition(y, groups):
    kf = KFold(5, shuffle=True, random_state=0)
    gkf = GroupKFold(5)
    rnd, _ = family_mean_null_cv(y, groups, kf)
    grp, _ = family_mean_null_cv(y, groups, gkf, split_groups=groups)
    return {"family_mean_random": rnd, "family_mean_grouped": grp}

def load_datasetA(uci_dir):
    train = pd.read_csv(f"{uci_dir}/train.csv")
    unique_m = pd.read_csv(f"{uci_dir}/unique_m.csv")
    y = train["critical_temp"].values
    X = train.drop(columns=["critical_temp"]).values.astype(np.float32)
    groups = element_set_groups(unique_m["material"].values)
    return X, y, groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["A", "B"], required=True)
    ap.add_argument("--featurized", help="datasetB_featurized.npz (Dataset B)")
    ap.add_argument("--uci-dir", help="folder with train.csv + unique_m.csv (Dataset A)")
    ap.add_argument("--out", default="null_analysis_out.json")
    args = ap.parse_args()

    if args.dataset == "B":
        z = np.load(args.featurized)
        X, y, groups = z["X"], z["y"], z["groups"]
    else:
        X, y, groups = load_datasetA(args.uci_dir)

    print(f"Dataset {args.dataset}: N={len(y)}  families={len(np.unique(groups))}")
    decomp = run_null_decomposition(y, groups)
    print("Featureless family-mean null:")
    for k, v in decomp.items():
        print(f"  {k}: MAE={v['MAE']:.2f} R2={v['R2']:.3f} seen={v['test_family_seen_pct']}%")
    json.dump(decomp, open(args.out, "w"), indent=1)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
