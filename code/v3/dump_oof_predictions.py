"""Out-of-fold predictions, row by row, for every headline run of the paper (v3, H12).

For Dataset A (data/train.csv + unique_m.csv, family = tc_pipeline.chemical_families) and
Dataset B (data/datasetB_featurized.npz, family = `groups` field), for XGBoost and RandomForest
(make_models(seed) of each pipeline, seed 42), under
    random  : KFold(5, shuffle=True, random_state=42)
    grouped : GroupKFold(5)  (deterministic, the published protocol)
writes results_v3/oof_predictions.csv.gz with one row per (dataset, model, protocol, row):
    dataset, model, protocol, seed, row_index, family_id, fold, y_true, y_pred,
    threshold_p90_train (90th percentile of y in the TRAINING fold), config_json, procedencia
and results_v3/oof_summary.csv with the pooled MAE/RMSE/R2 recomputed from that file, next to
the v2 (results_v2/nulo_presencia.csv) and v1 (results/consolidated_results.csv) values.

Fold assignment is cross-checked against splits/*.csv (fold_deterministic for the grouped arm;
the random arm at seed 42 is NOT in splits/, which stores random seeds 0-4 only, so KFold(rs=0)
is checked against random_fold_seed0 as a splitter-identity control and the rs=42 labels are
written here).

Usage:  python code/v3/dump_oof_predictions.py [--datasets A B] [--models XGBoost RandomForest]
        [--out DIR]     (default DIR = <ROOT>/../results_v3, override with SUPERCON_RESULTS_V3)
"""
from __future__ import annotations
import argparse, importlib.util, json, os, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import sklearn, xgboost
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
CODE, DATA, SPLITS = ROOT / "code", ROOT / "data", ROOT / "splits"
RES_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
SEED = 42
PROCEDENCIA = (f"code/v3/dump_oof_predictions.py | python {platform.python_version()}, numpy {np.__version__}, "
               f"pandas {pd.__version__}, scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__} | "
               f"seed {SEED}; random = KFold(5, shuffle=True, random_state={SEED}); grouped = GroupKFold(5) "
               f"deterministic; family = constituent-element set (A: tc_pipeline.chemical_families(unique_m.csv); "
               f"B: `groups` of data/datasetB_featurized.npz = pandas.factorize AFTER the Tc>0 filter); "
               f"models = make_models({SEED}) of code/tc_pipeline.py (A) / code/datasetB_pipeline.py (B); "
               f"threshold_p90_train = np.percentile(y[train fold], 90)")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def load_dataset(ds: str):
    if ds == "A":
        tcp = load_mod("tc_pipeline", str(CODE / "tc_pipeline.py"))
        feat = pd.read_csv(DATA / "train.csv"); uniq = pd.read_csv(DATA / "unique_m.csv")
        assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "train.csv / unique_m.csv misaligned"
        X = feat.drop(columns=["critical_temp"]).values.astype(float)
        y = feat["critical_temp"].values.astype(float)
        groups = tcp.chemical_families(uniq)
        return X, y, groups, tcp.make_models
    if ds == "B":
        # datasetB_pipeline imports pymatgen/matminer; make_models itself needs neither, but the
        # module import does. Keep the exact hyperparameters of the pipeline by importing it.
        dbp = load_mod("datasetB_pipeline", str(CODE / "datasetB_pipeline.py"))
        z = np.load(DATA / "datasetB_featurized.npz", allow_pickle=True)
        return z["X"], z["y"].astype(float), z["groups"].astype(int), dbp.make_models
    raise ValueError(ds)


def fold_labels(splitter, X, y, groups=None):
    lab = np.full(len(y), -1, dtype=int)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for k, (_, te) in enumerate(it):
        lab[te] = k
    assert (lab >= 0).all()
    return lab


def config_of(model) -> str:
    p = model.get_params()
    if type(model).__name__ == "XGBRegressor":            # drop the None defaults, keep what the pipeline set
        p = {k: v for k, v in p.items() if v is not None}
    return json.dumps({"class": type(model).__name__, "params": p}, sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["A", "B"])
    ap.add_argument("--models", nargs="+", default=["XGBoost", "RandomForest"])
    ap.add_argument("--out", type=Path, default=RES_V3)
    ap.add_argument("--no-v1-control", action="store_true", help="skip the Dataset-B v1-partition control run")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames, summary = [], []
    for ds in args.datasets:
        X, y, groups, make_models = load_dataset(ds)
        sp = pd.read_csv(SPLITS / f"dataset{ds}_folds.csv", comment="#")
        assert len(sp) == len(y) and np.array_equal(sp["row_index"].values, np.arange(len(y)))
        assert np.array_equal(sp["family_id"].values, groups), f"{ds}: family_id of splits != recomputed groups"
        protocols = {"random": (KFold(5, shuffle=True, random_state=SEED), None),
                     "grouped": (GroupKFold(5), groups)}
        folds = {name: fold_labels(cv, X, y, g) for name, (cv, g) in protocols.items()}
        # --- cross-check against the released partitions -------------------------------------
        assert np.array_equal(folds["grouped"], sp["fold_deterministic"].values), f"{ds}: GroupKFold(5) != splits/fold_deterministic"
        ctrl = fold_labels(KFold(5, shuffle=True, random_state=0), X, y)
        assert np.array_equal(ctrl, sp["random_fold_seed0"].values), f"{ds}: KFold(rs=0) != splits/random_fold_seed0"
        print(f"[{ds}] n={len(y)} k={len(np.unique(groups))}  folds OK vs splits/ (grouped=fold_deterministic; KFold rs=0 control)")
        for model in args.models:
            for proto, (cv, g) in protocols.items():
                t0 = time.time()
                yp = np.full(len(y), np.nan); thr = np.full(len(y), np.nan)
                it = cv.split(X, y, g) if g is not None else cv.split(X, y)
                cfg = None
                for k, (tr, te) in enumerate(it):
                    assert (folds[proto][te] == k).all()
                    m = make_models(SEED)[model]; cfg = cfg or config_of(m)
                    m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te]); thr[te] = np.percentile(y[tr], 90)
                assert not np.isnan(yp).any()
                df = pd.DataFrame({"dataset": ds, "model": model, "protocol": proto, "seed": SEED,
                                   "row_index": np.arange(len(y)), "family_id": groups, "fold": folds[proto],
                                   "y_true": y, "y_pred": yp, "threshold_p90_train": thr,
                                   "config_json": cfg, "procedencia": PROCEDENCIA})
                frames.append(df)
                summary.append({"dataset": ds, "model": model, "protocol": proto, "seed": SEED, "n": len(y),
                                "MAE": mean_absolute_error(y, yp), "RMSE": float(np.sqrt(mean_squared_error(y, yp))),
                                "R2": r2_score(y, yp), "seconds": round(time.time() - t0, 1)})
                print(f"  {ds} {model:12s} {proto:8s} MAE={summary[-1]['MAE']:.4f} R2={summary[-1]['R2']:.4f} ({summary[-1]['seconds']} s)", flush=True)
    # --- control: Dataset B on the v1 (pre-filter factorize) partition -----------------------------
    # splits/datasetB_folds.csv keeps the fold assignment that v1's live datasetB_pipeline.py used
    # (fold_deterministic_v1_prefilter_codes).  Scoring the same models on it tells whether a
    # published Dataset-B number came from that defective partition (README "Changes in v2").
    if "B" in args.datasets and not args.no_v1_control:
        X, y, groups, make_models = load_dataset("B")
        sp = pd.read_csv(SPLITS / "datasetB_folds.csv", comment="#")
        v1f = sp["fold_deterministic_v1_prefilter_codes"].values
        for model in args.models:
            t0 = time.time(); yp = np.full(len(y), np.nan); thr = np.full(len(y), np.nan); cfg = None
            for k in range(5):
                te = np.where(v1f == k)[0]; tr = np.where(v1f != k)[0]
                m = make_models(SEED)[model]; cfg = cfg or config_of(m)
                m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te]); thr[te] = np.percentile(y[tr], 90)
            frames.append(pd.DataFrame({"dataset": "B", "model": model, "protocol": "grouped_v1_prefilter_partition", "seed": SEED,
                                        "row_index": np.arange(len(y)), "family_id": groups, "fold": v1f, "y_true": y, "y_pred": yp,
                                        "threshold_p90_train": thr, "config_json": cfg, "procedencia": PROCEDENCIA}))
            summary.append({"dataset": "B", "model": model, "protocol": "grouped_v1_prefilter_partition", "seed": SEED, "n": len(y),
                            "MAE": mean_absolute_error(y, yp), "RMSE": float(np.sqrt(mean_squared_error(y, yp))),
                            "R2": r2_score(y, yp), "seconds": round(time.time() - t0, 1)})
            print(f"  B {model:12s} v1-part  MAE={summary[-1]['MAE']:.4f} R2={summary[-1]['R2']:.4f} ({summary[-1]['seconds']} s)", flush=True)
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(args.out / "oof_predictions.csv.gz", index=False, compression="gzip")
    # --- compare with v2 (nulo_presencia.csv), v1 (results/consolidated_results.csv) and the .tex ----
    v2 = pd.read_csv(ROOT / "results_v2" / "nulo_presencia.csv")
    v2 = v2[v2.representacion == "features_completas_paper"]
    v2["protocol"] = v2.split.map({"aleatorio_KFold5": "random", "GroupKFold5_familias": "grouped"})
    v1 = pd.read_csv(ROOT / "results" / "consolidated_results.csv")
    v1["dataset"] = v1.dataset.str[0]; v1["protocol"] = v1.split.map({"Random": "random", "Chemical-family": "grouped"})
    v1 = pd.concat([v1, v1[v1.protocol == "grouped"].assign(protocol="grouped_v1_prefilter_partition")])   # compare the control with v1 too
    S = pd.DataFrame(summary)
    for i, r in S.iterrows():
        a = v2[(v2.dataset == r.dataset) & (v2.protocol == r.protocol)]
        if r.model == "XGBoost" and len(a) == 1:            # nulo_presencia stores XGBoost only
            S.loc[i, "MAE_v2"] = a.MAE.iloc[0]; S.loc[i, "R2_v2"] = a.R2.iloc[0]
        b = v1[(v1.dataset == r.dataset) & (v1.protocol == r.protocol) & (v1.model == r.model)]
        if len(b) == 1:
            S.loc[i, "MAE_v1"] = b.MAE_K.iloc[0]; S.loc[i, "R2_v1"] = b.R2.iloc[0]
    S["dMAE_vs_v2"] = S.MAE - S.MAE_v2; S["dMAE_vs_v1"] = S.MAE - S.MAE_v1
    try:                                                   # tab:main of the manuscript, as typed
        sys.path.insert(0, str(Path(__file__).resolve().parent)); import verify_manuscript as vm
        text = vm.strip_comments((ROOT / "paper1_manuscript.tex").read_text())
        tab = [t for t in vm.parse_tables(text) if t["label"] == "tab:main"][0]
        rowname = {"XGBoost": "XGBoost", "RandomForest": "Random forest"}; spname = {"random": "Random", "grouped": "Chemical-family"}
        for i, r in S.iterrows():
            hits = [x for x in tab["rows"] if x["clean"][:2] == [rowname[r.model], spname.get(r.protocol, "?")] and f"Dataset {r.dataset}" in x["block"]]
            if len(hits) == 1:
                S.loc[i, "MAE_tex"], S.loc[i, "RMSE_tex"], S.loc[i, "R2_tex"] = [float(v) for v in hits[0]["clean"][2:5]]
        S["MAE_2dp_matches_tex"] = S.apply(lambda r: (format(r.MAE, ".2f") == format(r.MAE_tex, ".2f")) if pd.notna(r.get("MAE_tex")) else None, axis=1)
    except Exception as e:                                 # noqa
        print("tab:main comparison skipped:", e)
    S["procedencia"] = PROCEDENCIA + " | v2 = results_v2/nulo_presencia.csv (features_completas_paper, XGBoost only); v1 = results/consolidated_results.csv (3 decimals; Dataset B v1 ran on the pre-filter-factorize partition, README 'Changes in v2')"
    S.to_csv(args.out / "oof_summary.csv", index=False)
    print(S.drop(columns="procedencia").to_string())
    print(f"wrote {args.out / 'oof_predictions.csv.gz'} ({len(oof)} rows) and oof_summary.csv")


if __name__ == "__main__":
    main()
