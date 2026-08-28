"""
_top100_harness.py — one place where the top-100 screening-precision definition lives,
so every number in results_v2/ is produced by the same measured code path.

Definitions under test (the manuscript's two values differ in these axes):
  threshold    : "train_fold" -> np.percentile(y[tr], 90)   [datasetB_pipeline.topk]
                 "full"       -> np.percentile(y, 90)       [null_model_analysis.top100_null_vs_model]
  featurization: "live"       -> matminer ElementProperty magpie, run now
                 "npz"        -> data/datasetB_featurized.npz shipped checkpoint
  scaler       : False        -> raw X                      [datasetB_pipeline.topk]
                 True         -> StandardScaler fit on train fold [null_model_analysis]

Hyperparameters are ALWAYS taken from the repo scripts via importlib (never rewritten):
  Dataset A -> tc_pipeline.make_models(seed)["XGBoost"]
  Dataset B -> datasetB_pipeline.make_models(seed)["XGBoost"]
"""
from __future__ import annotations
import importlib.util, sys
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
REPO = "/Users/freakscode/Proyectos 2025/Proyectos Académicos/Superconductors/superconductor-tc-leakage-eval"


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def xgb_factory(dataset: str, seed: int = 42):
    """Return a zero-arg factory producing a FRESH XGBRegressor with the repo's
    per-dataset hyperparameters. dataset in {"A","B"}."""
    if dataset == "A":
        mod = load_mod("tcp_h", f"{REPO}/code/tc_pipeline.py")
    elif dataset == "B":
        mod = load_mod("dsB_h", f"{REPO}/code/datasetB_pipeline.py")
    else:
        raise ValueError(dataset)
    return lambda: mod.make_models(seed)["XGBoost"]


def folds_for(scheme: str, X, y, groups, seed: int = 42):
    """Materialise fold index pairs. grouped -> GroupKFold(5) (deterministic, seedless);
    random -> KFold(5, shuffle=True, random_state=seed) as in both repo scripts."""
    if scheme == "grouped":
        return [(tr, te) for tr, te in GroupKFold(5).split(X, y, groups)]
    if scheme == "random":
        return [(tr, te) for tr, te in KFold(5, shuffle=True, random_state=seed).split(X)]
    raise ValueError(scheme)


def top100_folds(X, y, model_factory, folds, threshold: str, K: int = 100,
                 scaler: bool = False):
    """Per-fold top-K screening precision (fraction, 0-1). Returns list of floats.

    Selection is the K highest PREDICTED Tc in the held-out pool; a hit is a test
    material whose TRUE Tc is at or above the high-Tc threshold.
    """
    if threshold == "full":
        thr_full = float(np.percentile(y, 90))
    out = []
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        if scaler:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = model_factory()
        m.fit(Xtr, y[tr])
        yp = m.predict(Xte)
        thr = thr_full if threshold == "full" else float(np.percentile(y[tr], 90))
        sel = np.argsort(yp)[-K:]
        out.append(float((y[te][sel] >= thr).mean()))
    return out


def top100_folds_both_thresholds(X, y, model_factory, folds, K: int = 100,
                                 scaler: bool = False):
    """Fit ONCE per fold, evaluate both threshold definitions on the same fitted
    model. The threshold enters only the evaluation, never the fit, so this is
    exactly equivalent to two separate runs of top100_folds() while making the
    'same model, same folds' claim structural rather than dependent on
    XGBoost determinism. Returns dict threshold -> list of per-fold precisions.
    """
    thr_full = float(np.percentile(y, 90))
    out = {"train_fold": [], "full": []}
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        if scaler:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = model_factory()
        m.fit(Xtr, y[tr])
        yp = m.predict(Xte)
        sel = np.argsort(yp)[-K:]
        hit = y[te][sel]
        out["train_fold"].append(float((hit >= float(np.percentile(y[tr], 90))).mean()))
        out["full"].append(float((hit >= thr_full).mean()))
    return out


def summarize(folds_prec):
    a = np.asarray(folds_prec, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std()), folds=[float(v) for v in a])
