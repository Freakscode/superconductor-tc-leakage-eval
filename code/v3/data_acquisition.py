"""Data-acquisition experiment (tutor comment 23): depth versus breadth at a fixed record budget.

Question: to improve transfer to element systems never seen in training, is it better to measure more
compositions of systems already represented, or to add new systems?

Design (both datasets, XGBoost with the paper's fixed configuration):
  * external test   = 20 % of element systems, all their rows, held out; 3 independent draws (seed t)
  * training pool   = the remaining systems
  * budget N        = 500, 1000, 2000 training rows
  * arms            = per-system cap c in {1, 2, 4, 8, 32, all}: visit pool systems in a random order and
                      take at most c random rows from each until N rows are collected (c=1 maximises the
                      number of systems, c=all takes whole systems); plus 'random_rows' (N rows drawn
                      uniformly, the natural sampling of the corpus)
  * replicates      = 10 per (test draw, budget); the system order and within-system row order are shared
                      by all arms of a replicate, so arm differences are paired
Metrics on the fixed external test: pooled MAE, macro MAE (mean of per-system MAE), R^2, and selection
with threshold t = 90th percentile of the TRAINING sample's Tc (operator >=): prevalence pi, precision@100,
enrichment = P@100/pi, distinct systems among the top 100.
Intervals for paired differences are percentiles over the 30 (test draw x replicate) pairs: they describe
variation over training samples and test draws, conditional on this corpus.

Output: results_v3/data_acquisition.csv (one row per fit) and data_acquisition_summary.csv.
"""
import importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
CODE, DATA, SPLITS = ROOT / "code", ROOT / "data", ROOT / "splits"
BUDGETS = (500, 1000, 2000)
CAPS = (1, 2, 4, 8, 32, None)
N_TEST_DRAWS, N_REPS, TEST_FRAC, K = 3, 10, 0.20, 100


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def load_dataset(ds):
    if ds == "A":
        tr = pd.read_csv(DATA / "train.csv")
        X = tr.drop(columns=["critical_temp"]).values.astype(np.float32); y = tr.critical_temp.values
        fam = pd.read_csv(SPLITS / "datasetA_folds.csv", comment="#").family_id.values
        mk = load_mod("tc_pipeline", CODE / "tc_pipeline.py").make_models
    else:
        z = np.load(DATA / "datasetB_featurized_v3.npz", allow_pickle=True)
        X, y, fam = z["X"].astype(np.float32), z["y"], z["groups"]
        mk = load_mod("datasetB_pipeline", CODE / "datasetB_pipeline.py").make_models
    return X, y, fam, mk


def sample_rows(order, rows_by_sys, cap, N):
    out = []
    for s in order:
        r = rows_by_sys[s]
        take = r if cap is None else r[:cap]
        need = N - len(out)
        out.extend(take[:need])
        if len(out) >= N:
            break
    return np.asarray(out)


def evaluate(y_te, p, fam_te, thr):
    pos = y_te >= thr
    top = np.argsort(-p, kind="stable")[:K]
    prev = float(pos.mean())
    prec = float(pos[top].mean())
    d = pd.DataFrame({"f": fam_te, "e": np.abs(y_te - p)})
    return dict(MAE=float(mean_absolute_error(y_te, p)), MAE_macro=float(d.groupby("f").e.mean().mean()),
                R2=float(r2_score(y_te, p)), thr_K=float(thr), prevalence=prev, P_at_100=prec,
                enrichment=prec / prev if prev > 0 else np.nan, systems_top100=int(len(set(fam_te[top]))))


def main():
    rows = []
    t0 = time.time()
    for ds in ("A", "B"):
        X, y, fam, mk = load_dataset(ds)
        systems = np.unique(fam)
        for t in range(N_TEST_DRAWS):
            rng_t = np.random.default_rng(1000 + t)
            test_sys = set(rng_t.choice(systems, size=int(round(TEST_FRAC * len(systems))), replace=False))
            te = np.flatnonzero(np.isin(fam, list(test_sys)))
            pool = np.flatnonzero(~np.isin(fam, list(test_sys)))
            pool_sys = np.unique(fam[pool])
            for r in range(N_REPS):
                rng = np.random.default_rng(10_000 * t + r)
                rows_by_sys = {s: rng.permutation(pool[fam[pool] == s]) for s in pool_sys}
                order = rng.permutation(pool_sys)
                rand_rows = rng.permutation(pool)
                for N in BUDGETS:
                    arms = [(f"cap{c}" if c else "cap_all", sample_rows(order, rows_by_sys, c, N)) for c in CAPS]
                    arms.append(("random_rows", rand_rows[:N]))
                    for arm, idx in arms:
                        assert len(idx) == N, (ds, arm, N, len(idx))
                        m = mk(42)["XGBoost"]; m.set_params(n_jobs=4)
                        m.fit(X[idx], y[idx]); p = m.predict(X[te])
                        thr = np.percentile(y[idx], 90)
                        res = evaluate(y[te], p, fam[te], thr)
                        # fixed thresholds, identical for every arm: the training-sample p90 differs
                        # between arms (they sample different Tc distributions), so selection metrics
                        # are compared across arms at 77 K and at the p90 of the whole training pool
                        for tag, tfix in (("77K", 77.0), ("poolp90", float(np.percentile(y[pool], 90)))):
                            e = evaluate(y[te], p, fam[te], tfix)
                            res.update({f"{k}_{tag}": e[k] for k in ("thr_K", "prevalence", "P_at_100", "enrichment", "systems_top100")})
                        rows.append(dict(dataset=ds, test_draw=t, rep=r, budget=N, arm=arm,
                                         n_systems_train=int(len(np.unique(fam[idx]))),
                                         rows_per_system=float(N / len(np.unique(fam[idx]))),
                                         mean_Tc_train=float(y[idx].mean()), n_test=int(len(te)),
                                         n_test_systems=len(test_sys), **res))
                print(f"{ds} test{t} rep{r} done {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    prov = dict(script="code/v3/data_acquisition.py", budgets=BUDGETS, caps=[c or "all" for c in CAPS],
                n_test_draws=N_TEST_DRAWS, n_reps=N_REPS, test_frac=TEST_FRAC, model="XGBoost make_models(42)",
                threshold="p90 of training sample, >=", seconds=round(time.time() - t0, 1),
                versions={m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "xgboost")})
    df["procedencia"] = json.dumps(prov)
    RESULTS_V3.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_V3 / "data_acquisition.csv", index=False)

    # summary + paired differences versus cap_all (depth) within each (dataset, test_draw, rep, budget)
    met = ["MAE", "MAE_macro", "R2", "P_at_100", "enrichment", "systems_top100", "n_systems_train", "prevalence",
           "P_at_100_77K", "enrichment_77K", "prevalence_77K", "P_at_100_poolp90", "enrichment_poolp90",
           "systems_top100_poolp90", "rows_per_system", "mean_Tc_train"]
    summ = []
    for (ds, N, arm), g in df.groupby(["dataset", "budget", "arm"]):
        ref = df[(df.dataset == ds) & (df.budget == N) & (df.arm == "cap_all")].set_index(["test_draw", "rep"])
        gg = g.set_index(["test_draw", "rep"])
        rec = dict(dataset=ds, budget=N, arm=arm, n_fits=len(g))
        for mname in met:
            rec[f"{mname}_mean"] = gg[mname].mean()
            d = (gg[mname] - ref.loc[gg.index, mname]).values
            rec[f"d_{mname}_vs_cap_all_mean"] = d.mean()
            rec[f"d_{mname}_vs_cap_all_lo"] = np.percentile(d, 2.5)
            rec[f"d_{mname}_vs_cap_all_hi"] = np.percentile(d, 97.5)
        summ.append(rec)
    s = pd.DataFrame(summ); s["procedencia"] = json.dumps(prov)
    s.to_csv(RESULTS_V3 / "data_acquisition_summary.csv", index=False)
    print(f"wrote {len(df)} fits, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
