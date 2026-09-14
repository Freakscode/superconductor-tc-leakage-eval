"""topk_prevalence_ties.py -- H06: real per-fold prevalence of the high-Tc class and tie handling in top-K.

Step 1  Out-of-fold predictions for every ranker of the null ladder, both protocols, Dataset A and
        Dataset B v3 (plus B v2 for reproduction of the report's figures):
            null      family-mean lookup (global train mean if the family is unseen)
            presence  XGBoost (make_models(42), same hyperparameters) on binary element indicators
            XGBoost / RandomForest  make_models(42) on the paper's features
        -> <RESULTS_V3>/topk_oof_predictions.csv.gz   (cache used by topk_metrics.py / topk_negatives.py)
        Cross-check: A and Bv2 XGBoost/RandomForest OOF vs results_v3/oof_predictions.csv.gz.
Step 2  Prevalence per fold: fraction of the TEST pool with y >= p90(y_train) (and with >, and the
        training-fold fraction itself) -> <RESULTS_V3>/topk_prevalence.csv
Step 3  Ties at K=100: for each ranker/protocol/fold, number of distinct predictions, size of the tie
        group at the cut, the published selection (np.argsort(pred)[-K:]), its sensitivity to the ROW
        ORDER (200 permutations, seed 0), the fractional expected precision, and the mean over 200
        random tie-breaks (seed 0) -> <RESULTS_V3>/topk_ties.csv

Usage: python code/v3/topk_prevalence_ties.py [--skip-fit]  (--skip-fit reuses the cache)
Paths resolve from this file; SUPERCON_ROOT / SUPERCON_RESULTS_V3 override.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import topk_common as tc  # noqa: E402

SCRIPT = "code/v3/topk_prevalence_ties.py"
N_PERM = 200
K_TIES = 100


def procedencia(extra: str = "") -> str:
    return (f"{SCRIPT} | {tc.versions()} | seed {tc.SEED}; random = KFold(5, shuffle=True, random_state=42); "
            f"grouped = GroupKFold(5) deterministic (== splits fold_deterministic); family = constituent-element set; "
            f"threshold = np.percentile(y[train fold], 90), positive := y >= thr; n_jobs={tc.N_JOBS}"
            + (f" | {extra}" if extra else ""))


# --------------------------------------------------------------------------------------- step 1
def compute_oof(datasets: dict) -> pd.DataFrame:
    rows = []
    for dname, d in datasets.items():
        y, g = d["y"], d["groups"]
        folds = tc.protocol_folds(d)
        rankers = {"null": None, "presence": ("XGBoost", d["X_pres"]),
                   "XGBoost": ("XGBoost", d["X"]), "RandomForest": ("RandomForest", d["X"])}
        if dname == "Bv2":
            rankers = {"null": None, "XGBoost": ("XGBoost", d["X"])}
        for proto, lab in folds.items():
            for rname, spec in rankers.items():
                t0 = time.time(); yp = np.full(len(y), np.nan); thr_row = np.full(len(y), np.nan)
                for k in range(5):
                    tr, te = np.where(lab != k)[0], np.where(lab == k)[0]
                    thr = float(np.percentile(y[tr], 90))
                    if spec is None:
                        yp[te] = tc.family_mean_null(y, g, tr, te)
                    else:
                        mname, X = spec
                        yp[te] = tc.fit_predict(d["make_models"](tc.SEED)[mname], X[tr], y[tr], X[te])
                    thr_row[te] = thr
                assert not np.isnan(yp).any()
                rows.append(pd.DataFrame(dict(dataset=dname, model=rname, protocol=proto, row_index=np.arange(len(y)),
                                              family_id=g, fold=lab, y_true=y, y_pred=yp, threshold_p90_train=thr_row)))
                print(f"[oof] {dname}/{rname}/{proto}: {time.time()-t0:.0f}s", flush=True)
    oof = pd.concat(rows, ignore_index=True)
    oof["procedencia"] = procedencia("presence = XGBoost make_models(42) on binary element indicators "
                                     "(A: unique_m>0; B: pymatgen Composition)")
    return oof


def crosscheck_oof(oof: pd.DataFrame) -> dict:
    ref_path = tc.RES_V3 / "oof_predictions.csv.gz"
    out = {}
    if not ref_path.exists():
        return {"nota": "results_v3/oof_predictions.csv.gz no encontrado"}
    ref = tc.read_csv(ref_path)
    for (ds_new, ds_ref) in (("A", "A"), ("Bv2", "B")):
        for model in ("XGBoost", "RandomForest"):
            for proto in ("random", "grouped"):
                a = oof[(oof.dataset == ds_new) & (oof.model == model) & (oof.protocol == proto)].sort_values("row_index")
                b = ref[(ref.dataset == ds_ref) & (ref.model == model) & (ref.protocol == proto)].sort_values("row_index")
                if len(a) == 0 or len(b) == 0:
                    continue
                assert np.array_equal(a.fold.values, b.fold.values) and np.allclose(a.y_true.values, b.y_true.values)
                assert np.allclose(a.threshold_p90_train.values, b.threshold_p90_train.values)
                dif = np.abs(a.y_pred.values - b.y_pred.values)
                # does the legacy top-100 selection change?
                ch = 0
                for k in range(5):
                    m = a.fold.values == k
                    s1 = set(a.row_index.values[m][tc.legacy_topk_idx(a.y_pred.values[m], 100)])
                    s2 = set(b.row_index.values[m][tc.legacy_topk_idx(b.y_pred.values[m], 100)])
                    ch += len(s1 ^ s2) // 2
                out[f"{ds_new}/{model}/{proto}"] = dict(max_abs_diff=float(dif.max()), mean_abs_diff=float(dif.mean()),
                                                        top100_rows_changed_total_5folds=int(ch),
                                                        ref_n_jobs="tc_pipeline n_jobs=-1 (A) / datasetB_pipeline n_jobs=1 (B)")
    return out


# --------------------------------------------------------------------------------------- step 2
def prevalence(oof: pd.DataFrame, datasets: dict) -> pd.DataFrame:
    rows = []
    base = oof[oof.model == "null"]        # y/fold/thr identical across rankers
    for (ds, proto), sub in base.groupby(["dataset", "protocol"], sort=False):
        y_all = datasets[ds]["y"]
        for k in sorted(sub.fold.unique()):
            te = sub[sub.fold == k]; thr = float(te.threshold_p90_train.iloc[0])
            tr_mask = (sub.fold != k).values; ytr = sub.y_true.values[tr_mask]
            assert abs(np.percentile(ytr, 90) - thr) < 1e-9
            rows.append(dict(dataset=ds, protocol=proto, fold=k, n_train=int(tr_mask.sum()), n_test=int(len(te)),
                             threshold_p90_train=thr,
                             n_pos_test_ge=int((te.y_true >= thr).sum()), prevalence_test_ge_pct=100 * float((te.y_true >= thr).mean()),
                             prevalence_test_gt_pct=100 * float((te.y_true > thr).mean()),
                             train_frac_ge_pct=100 * float((ytr >= thr).mean()), train_frac_gt_pct=100 * float((ytr > thr).mean()),
                             n_families_test=int(te.family_id.nunique()),
                             n_test_rows_equal_thr=int((te.y_true == thr).sum())))
    df = pd.DataFrame(rows)
    df["procedencia"] = procedencia("prevalence_test_ge = fraction of the TEST fold with y >= p90(train) = expectation of "
                                    "a uniformly random selector's precision@K in that fold; train_frac shows that ties "
                                    "in Tc keep even the training fraction from being exactly 10 %")
    return df


# --------------------------------------------------------------------------------------- step 3
def ties(oof: pd.DataFrame) -> pd.DataFrame:
    rng_perm = np.random.RandomState(0); rng_tb = np.random.RandomState(0)
    rows = []
    for (ds, model, proto), sub in oof.groupby(["dataset", "model", "protocol"], sort=False):
        per_fold_perm = []   # 5 x N_PERM legacy precision after row permutation
        for k in sorted(sub.fold.unique()):
            te = sub[sub.fold == k]; yp = te.y_pred.values; yt = te.y_true.values; thr = te.threshold_p90_train.iloc[0]
            hit = (yt >= thr).astype(float); n = len(yt)
            # tie structure at the cut
            order = np.argsort(-yp, kind="stable"); ps = yp[order]; g = tc.tie_groups(ps)
            T = np.bincount(g); cumT = np.cumsum(T); idx = np.searchsorted(cumT, K_TIES, side="right")
            cut_group_size = int(T[idx]) if idx < len(T) else 0
            cut_needed = int(K_TIES - (cumT[idx] - T[idx])) if idx < len(T) else 0
            legacy = float(hit[tc.legacy_topk_idx(yp, K_TIES)].mean())
            legacy_desc = float(hit[np.argsort(-yp)[:K_TIES]].mean())
            perms = np.empty(N_PERM)
            for r in range(N_PERM):
                pi = rng_perm.permutation(n)
                perms[r] = hit[pi][tc.legacy_topk_idx(yp[pi], K_TIES)].mean()
            per_fold_perm.append(perms)
            frac = tc.expected_hits(yp, hit, (K_TIES,))[K_TIES] / K_TIES
            tb = np.empty(N_PERM)
            for r in range(N_PERM):
                tb[r] = hit[tc.random_tiebreak_order(yp, rng_tb)[:K_TIES]].mean()
            rows.append(dict(dataset=ds, model=model, protocol=proto, fold=k, n_test=n, threshold_p90_train=float(thr),
                             prevalence_test_ge_pct=100 * hit.mean(), n_distinct_pred=int(len(T)),
                             tie_group_size_at_cut=cut_group_size, slots_needed_from_cut_group=cut_needed,
                             cut_falls_inside_tie=bool(0 < cut_needed < cut_group_size),
                             precision100_legacy_argsort_pct=100 * legacy, precision100_argsort_desc_pct=100 * legacy_desc,
                             precision100_rowperm_mean_pct=100 * perms.mean(), precision100_rowperm_min_pct=100 * perms.min(),
                             precision100_rowperm_max_pct=100 * perms.max(),
                             precision100_fractional_expected_pct=100 * frac,
                             precision100_random_tiebreak_mean_pct=100 * tb.mean(), precision100_random_tiebreak_sd_pct=100 * tb.std(ddof=1)))
        # 5-fold mean under each row permutation (the report's "6.2-14.2 %" statistic)
        P = np.vstack(per_fold_perm)      # 5 x N_PERM (permutation r is not shared across folds; fine: independent order)
        fm = P.mean(axis=0)
        for r in rows[-5:]:
            r["fold_mean_rowperm_min_pct"] = 100 * float(fm.min()); r["fold_mean_rowperm_max_pct"] = 100 * float(fm.max())
            r["fold_mean_rowperm_mean_pct"] = 100 * float(fm.mean())
    df = pd.DataFrame(rows)
    df["procedencia"] = procedencia(f"K={K_TIES}; legacy = np.argsort(pred)[-K:] as in tc_pipeline.topk_precision / datasetB_pipeline "
                                    f"(ties resolved by row position); rowperm = same selection after {N_PERM} random permutations of "
                                    f"the test rows (RandomState(0)); fractional = expected precision under a uniformly random order "
                                    f"of the tied rows (topk_common.expected_hits); random_tiebreak = mean of {N_PERM} explicit random "
                                    f"tie-breaks (RandomState(0))")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-fit", action="store_true"); a = ap.parse_args()
    tc.RES_V3.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    datasets = {"A": tc.load_A(), "Bv3": tc.load_B("v3"), "Bv2": tc.load_B("v2")}
    for d in datasets.values():
        print(f"[data] {d['name']}: N={len(d['y'])} k={len(np.unique(d['groups']))} presence_cols={d['n_pres']}", flush=True)
    if a.skip_fit and tc.OOF_CACHE.exists():
        oof = tc.read_csv(tc.OOF_CACHE)
    else:
        oof = compute_oof(datasets)
        oof.to_csv(tc.OOF_CACHE, index=False, compression="gzip")
        print(f"[write] {tc.OOF_CACHE} ({len(oof)} rows)", flush=True)
    chk = crosscheck_oof(oof)
    json.dump(dict(crosscheck_vs_results_v3_oof_predictions=chk, procedencia=procedencia()),
              open(tc.RES_V3 / "topk_oof_crosscheck.json", "w"), indent=1)
    print(json.dumps(chk, indent=1), flush=True)
    prev = prevalence(oof, datasets); prev.to_csv(tc.RES_V3 / "topk_prevalence.csv", index=False)
    print(prev.drop(columns="procedencia").to_string(), flush=True)
    tie = ties(oof); tie.to_csv(tc.RES_V3 / "topk_ties.csv", index=False)
    cols = ["dataset", "model", "protocol", "fold", "n_distinct_pred", "tie_group_size_at_cut", "precision100_legacy_argsort_pct",
            "precision100_rowperm_min_pct", "precision100_rowperm_max_pct", "precision100_fractional_expected_pct",
            "precision100_random_tiebreak_mean_pct", "fold_mean_rowperm_min_pct", "fold_mean_rowperm_max_pct"]
    print(tie[cols].round(2).to_string(), flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
