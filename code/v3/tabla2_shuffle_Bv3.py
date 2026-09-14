"""tabla2_shuffle_Bv3.py -- Table 2 (seed robustness) for Dataset B on the AUDITED v3 checkpoint.

Same design as code/v2/run_tabla2_shuffle.py (which produced results_v2/tabla2_runs_por_semilla.csv on
the v2 checkpoint, N = 12 440): for seeds s = 0..4, random arm = KFold(5, shuffle=True, random_state=s),
grouped arm = GroupKFold(5, shuffle=True, random_state=s); model = datasetB_pipeline.make_models(s)[model].
Only the input changes: data/datasetB_featurized_v3.npz (N = 12 336, k = 3 046).
Dataset A is unchanged between v2 and v3, so its rows are taken from results_v2 and merged here.

Outputs (<RESULTS_V3>): tabla2_runs_por_semilla_v3.csv (per run), tabla2_shuffle_v3.csv (aggregate, same columns
as results_v2/tabla2_shuffle.csv: A rows copied from v2, B rows recomputed).
    python code/v3/tabla2_shuffle_Bv3.py            # ~10-20 min with NW=8 threads
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3_common as C  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold

SEEDS = [0, 1, 2, 3, 4]
OUT = C.RES_V3


def one(task, X, y, g, make_models):
    model, s = task
    t0 = time.time()
    rows = []
    for scheme in ("random_shuffle", "grouped_shuffle"):
        cv = KFold(5, shuffle=True, random_state=s) if scheme == "random_shuffle" else GroupKFold(5, shuffle=True, random_state=s)
        yp = np.empty_like(y, dtype=float); fold_mae, leak = [], []
        for tr, te in (cv.split(X, y, g) if scheme.startswith("grouped") else cv.split(X, y)):
            m = make_models(s)[model]; m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te])
            fold_mae.append(float(mean_absolute_error(y[te], yp[te]))); leak.append(int(len(set(g[tr]) & set(g[te]))))
        rows.append(dict(dataset="B", modelo=model, scheme=scheme, seed=int(s), MAE=float(mean_absolute_error(y, yp)),
                         R2=float(r2_score(y, yp)), n=int(len(y)), n_familias=int(len(np.unique(g))),
                         fold_MAE=json.dumps([round(v, 6) for v in fold_mae]), familias_compartidas_tr_te=int(sum(leak)),
                         segundos=round(time.time() - t0, 1), checkpoint="v3"))
    return rows


def main():
    X, y, g, make_models, meta = C.load_dataset("B")
    assert len(y) == 12336 and len(np.unique(g)) == 3046, (len(y), len(np.unique(g)))
    tasks = [(mo, s) for mo in ("RandomForest", "XGBoost") for s in SEEDS]
    rows = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get("NW", "8"))) as ex:
        for i, rr in enumerate(ex.map(lambda t: one(t, X, y, g, make_models), tasks), 1):
            rows += rr; print(f"[{i}/{len(tasks)}] {rr[0]['modelo']} s{rr[0]['seed']} MAE r/g = {rr[0]['MAE']:.4f}/{rr[1]['MAE']:.4f} ({rr[1]['segundos']}s)", flush=True)
    runs = pd.DataFrame(rows)
    runs["procedencia"] = f"code/v3/tabla2_shuffle_Bv3.py | {C.VERS} | data/datasetB_featurized_v3.npz | random = KFold(5, shuffle, rs=s); grouped = GroupKFold(5, shuffle=True, random_state=s); model = datasetB_pipeline.make_models(s); s = 0..4"
    runs.to_csv(OUT / "tabla2_runs_por_semilla_v3.csv", index=False)
    # aggregate, same columns as results_v2/tabla2_shuffle.csv (std ddof=0, inflation paired by seed)
    v2 = pd.read_csv(C.RES_V2 / "tabla2_shuffle.csv")
    agg = []
    for mo in ("RandomForest", "XGBoost"):
        r = runs[(runs.modelo == mo) & (runs.scheme == "random_shuffle")].sort_values("seed")
        q = runs[(runs.modelo == mo) & (runs.scheme == "grouped_shuffle")].sort_values("seed")
        assert (r.seed.values == q.seed.values).all()
        infl = 100 * (q.MAE.values / r.MAE.values - 1); dr2 = q.R2.values - r.R2.values
        agg.append(dict(dataset="B", modelo=mo, MAE_random_mean=r.MAE.mean(), MAE_random_std=r.MAE.std(ddof=0),
                        MAE_grouped_mean=q.MAE.mean(), MAE_grouped_std=q.MAE.std(ddof=0),
                        inflacion_pct_mean=infl.mean(), inflacion_pct_std=infl.std(ddof=0), dR2_mean=dr2.mean(), dR2_std=dr2.std(ddof=0),
                        n_semillas=5, procedencia=runs.procedencia.iloc[0], R2_random_mean=r.R2.mean(), R2_random_std=r.R2.std(ddof=0),
                        R2_grouped_mean=q.R2.mean(), R2_grouped_std=q.R2.std(ddof=0), MAE_grouped_min=q.MAE.min(), MAE_grouped_max=q.MAE.max(),
                        inflacion_pct_min=infl.min(), inflacion_pct_max=infl.max(), n=int(len(y)), n_familias=int(len(np.unique(g))),
                        v2_MAE_random_mean=float(v2[(v2.dataset == "B") & (v2.modelo == mo)].MAE_random_mean.iloc[0]),
                        v2_MAE_grouped_mean=float(v2[(v2.dataset == "B") & (v2.modelo == mo)].MAE_grouped_mean.iloc[0]),
                        v2_inflacion_pct_mean=float(v2[(v2.dataset == "B") & (v2.modelo == mo)].inflacion_pct_mean.iloc[0])))
    a_rows = v2[v2.dataset == "A"].copy()
    for c in ("v2_MAE_random_mean", "v2_MAE_grouped_mean", "v2_inflacion_pct_mean"): a_rows[c] = np.nan
    out = pd.concat([a_rows, pd.DataFrame(agg)], ignore_index=True)
    out.to_csv(OUT / "tabla2_shuffle_v3.csv", index=False)
    print(out[["dataset", "modelo", "MAE_random_mean", "MAE_grouped_mean", "inflacion_pct_mean", "inflacion_pct_std", "dR2_mean", "inflacion_pct_min", "inflacion_pct_max"]].to_string())


if __name__ == "__main__":
    main()
