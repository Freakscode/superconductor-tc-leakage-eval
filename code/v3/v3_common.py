"""v3_common.py — shared helpers for code/v3/oof_v3.py (H07 paired-by-family intervals).

NOTE (2026-09-14 00:24): this file was accidentally overwritten by a concurrent v3 task (the H09/H10
helpers now live in code/v3/h09h10_common.py). It was restored from the API that oof_v3.py imports:
    ROOT, RES_V2, RES_V3, SEED, N_JOBS, VERS, load_dataset, protocols, fold_labels, check_splits,
    family_mean_null, metrics, with_n_jobs
If the original author has a different version, theirs should replace this one.

Canonical inputs (v3): A = data/train.csv + data/unique_m.csv, family = tc_pipeline.chemical_families,
folds = splits/datasetA_folds.csv; B = data/datasetB_featurized_v3.npz (`groups`), folds =
splits/datasetB_folds_v3.csv. Models = make_models(seed) of code/tc_pipeline.py (A) /
code/datasetB_pipeline.py (B); with_n_jobs only changes the thread count.
"""
from __future__ import annotations
import importlib.util, os, platform, sys
from pathlib import Path
import numpy as np, pandas as pd
import sklearn, xgboost
from sklearn.model_selection import KFold, GroupKFold

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
CODE, DATA, SPLITS = ROOT / "code", ROOT / "data", ROOT / "splits"
RES_V2 = ROOT / "results_v2"
RES_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
SEED, N_JOBS = 42, 8
VERS = (f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
        f"scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def load_dataset(ds: str):
    """Returns X, y, groups, make_models, meta (meta['splits'] = Path of the folds csv, meta['formula'] for B)."""
    if ds == "A":
        tcp = load_mod("tc_pipeline", str(CODE / "tc_pipeline.py"))
        feat = pd.read_csv(DATA / "train.csv"); uniq = pd.read_csv(DATA / "unique_m.csv")
        assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "train.csv / unique_m.csv misaligned"
        X = feat.drop(columns=["critical_temp"]).values.astype(float)
        y = feat["critical_temp"].values.astype(float)
        groups = tcp.chemical_families(uniq)
        return X, y, groups, tcp.make_models, {"splits": SPLITS / "datasetA_folds.csv", "material": uniq["material"].values}
    if ds == "B":
        dbp = load_mod("datasetB_pipeline", str(CODE / "datasetB_pipeline.py"))
        z = np.load(DATA / "datasetB_featurized_v3.npz", allow_pickle=True)
        return (z["X"].astype(float), z["y"].astype(float), z["groups"], dbp.make_models,
                {"splits": SPLITS / "datasetB_folds_v3.csv", "formula": z["formula"]})
    raise ValueError(ds)


def protocols(groups, seed=SEED):
    return {"random": (KFold(5, shuffle=True, random_state=seed), None), "grouped": (GroupKFold(5), groups)}


def fold_labels(splitter, n, groups=None):
    lab = np.full(n, -1); dummy = np.zeros((n, 1))
    for f, (_, te) in enumerate(splitter.split(dummy, None, groups)): lab[te] = f
    return lab


def check_splits(ds, X, y, groups, splits_path):
    """GroupKFold(5) == fold_deterministic; KFold(5, shuffle, rs=0) == random_fold_seed0 (splitter-identity control)."""
    s = pd.read_csv(splits_path, comment="#")
    assert len(s) == len(y) and np.array_equal(s.family_id.values, groups), f"{ds}: family codes != splits"
    assert np.array_equal(fold_labels(GroupKFold(5), len(y), groups), s.fold_deterministic.values), f"{ds}: GroupKFold != splits"
    assert np.array_equal(fold_labels(KFold(5, shuffle=True, random_state=0), len(y)), s.random_fold_seed0.values), f"{ds}: KFold rs0 != splits"
    return True


def family_mean_null(y_tr, g_tr, g_te):
    """Mean Tc of the family in the TRAIN fold; global train mean for unseen families."""
    means = pd.Series(y_tr).groupby(g_tr).mean()
    return pd.Series(g_te).map(means).fillna(float(np.mean(y_tr))).values


def metrics(y, yp):
    e = y - yp
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def with_n_jobs(model, n_jobs=N_JOBS):
    model.set_params(n_jobs=n_jobs); return model
