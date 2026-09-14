"""h09h10_common.py — shared loaders, representations and family-bootstrap helpers for the v3 scripts
ablation_representation.py, remove_relatives_control.py and separation_levels.py (H09/H10).

Canonical inputs (v3):
  A : data/train.csv (81 features, y = critical_temp) + data/unique_m.csv (element amounts, row-aligned);
      family = tc_pipeline.chemical_families(unique_m); folds = splits/datasetA_folds.csv.
  B : data/datasetB_featurized_v3.npz (X 132 Magpie, y, groups, formula); folds = splits/datasetB_folds_v3.csv.
Models: make_models(seed)["XGBoost"] of code/tc_pipeline.py (A) / code/datasetB_pipeline.py (B) with
n_jobs overridden (threads only; no hyperparameter change).
Inference: the unit of dependence is the chemical FAMILY. `family_bootstrap` resamples whole families
(default 2 000 resamples, numpy default_rng(0)); when several prediction vectors are passed the SAME
family resample is applied to all of them (paired). Intervals are conditional on the fitted
out-of-fold predictions (the models are not refitted inside the bootstrap).
"""
from __future__ import annotations
import importlib.util, os, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import sklearn, xgboost
from sklearn.model_selection import KFold, GroupKFold

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
CODE, DATA, SPLITS = ROOT / "code", ROOT / "data", ROOT / "splits"
RES_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
SEED, N_JOBS, N_BOOT, BOOT_SEED = 42, 8, 2000, 0


def versions() -> str:
    return (f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
            f"scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- datasets
class Dataset:
    """X2 = paper descriptors; P = element presence (0/1); F = molar fractions (0 if absent);
    elements = column order of P/F; groups = chemical family (element set); folds = splits/*.csv."""
    def __init__(self, name, X2, y, groups, P, F, elements, folds, make_models, extra=None):
        self.name, self.X2, self.y, self.groups = name, X2, y, groups
        self.P, self.F, self.elements, self.folds, self.make_models = P, F, elements, folds, make_models
        self.extra = extra or {}
        self.n, self.k = len(y), len(np.unique(groups))

    def xgb(self, seed=SEED, n_jobs=N_JOBS):
        m = self.make_models(seed)["XGBoost"]; m.set_params(n_jobs=n_jobs); return m

    def representation(self, code):
        return {"R0": self.P, "R1": np.hstack([self.P, self.F]), "R2": self.X2,
                "R3": np.hstack([self.P, self.F, self.X2])}[code]


def load_A() -> Dataset:
    tcp = load_mod("tc_pipeline", str(CODE / "tc_pipeline.py"))
    feat = pd.read_csv(DATA / "train.csv"); uniq = pd.read_csv(DATA / "unique_m.csv")
    assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "train.csv / unique_m.csv misaligned"
    X2 = feat.drop(columns=["critical_temp"]).values.astype(float)
    y = feat["critical_temp"].values.astype(float)
    groups = tcp.chemical_families(uniq)
    elem_cols = [c for c in uniq.columns if c not in ("critical_temp", "material")]
    amounts = uniq[elem_cols].values.astype(float)
    P = (amounts > 0).astype(float)
    tot = amounts.sum(1, keepdims=True); assert (tot > 0).all()
    F = amounts / tot                                   # molar fractions as recorded (O=1 defect kept, see A_nomenclatura_check.json)
    folds = pd.read_csv(SPLITS / "datasetA_folds.csv", comment="#")
    assert len(folds) == len(y) and np.array_equal(folds.family_id.values, groups)
    return Dataset("A", X2, y, groups, P, F, elem_cols, folds, tcp.make_models,
                   extra={"material": uniq["material"].values})


def load_B() -> Dataset:
    dbp = load_mod("datasetB_pipeline", str(CODE / "datasetB_pipeline.py"))
    from pymatgen.core import Composition
    z = np.load(DATA / "datasetB_featurized_v3.npz", allow_pickle=True)
    X2, y, groups, formula = z["X"].astype(float), z["y"].astype(float), z["groups"], z["formula"]
    comps = [Composition(f).fractional_composition for f in formula]
    elements = sorted({e.symbol for c in comps for e in c.elements})
    idx = {e: i for i, e in enumerate(elements)}
    F = np.zeros((len(y), len(elements)))
    for r, c in enumerate(comps):
        for e, a in c.get_el_amt_dict().items(): F[r, idx[e]] = a
    assert np.allclose(F.sum(1), 1.0)
    P = (F > 0).astype(float)
    folds = pd.read_csv(SPLITS / "datasetB_folds_v3.csv", comment="#")
    assert len(folds) == len(y) and np.array_equal(folds.family_id.values, groups)
    return Dataset("B", X2, y, groups, P, F, elements, folds, dbp.make_models, extra={"formula": formula})


def load(ds: str) -> Dataset:
    return load_A() if ds == "A" else load_B()


# ----------------------------------------------------------------------------- folds
def grouped_folds(d: Dataset):
    """GroupKFold(5) deterministic on the family codes; verified against splits/*.csv fold_deterministic."""
    sp = list(GroupKFold(5).split(d.X2, d.y, d.groups))
    lab = np.empty(d.n, int)
    for f, (_, te) in enumerate(sp): lab[te] = f
    assert np.array_equal(lab, d.folds.fold_deterministic.values), "GroupKFold(5) != splits fold_deterministic"
    return sp


def random_folds(d: Dataset, seed=SEED):
    """KFold(5, shuffle=True, random_state=42) — the paper's random protocol (seed 42 is not stored in splits/;
    splits/random_fold_seed0 is used as a splitter-identity control)."""
    lab0 = np.empty(d.n, int)
    for f, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(d.X2)): lab0[te] = f
    assert np.array_equal(lab0, d.folds.random_fold_seed0.values), "KFold(rs=0) != splits random_fold_seed0"
    return list(KFold(5, shuffle=True, random_state=seed).split(d.X2))


def fold_labels(splits, n):
    lab = np.empty(n, int)
    for f, (_, te) in enumerate(splits): lab[te] = f
    return lab


# ----------------------------------------------------------------------------- fitting
def top100(y_tr, y_te, yp_te, K=100):
    """Precision@100: share of the K highest-PREDICTED test rows whose true Tc >= p90 of the TRAIN fold."""
    thr = np.percentile(y_tr, 90)
    return float((y_te[np.argsort(yp_te)[-K:]] >= thr).mean())


def oof(model_fn, X, y, splits):
    """Out-of-fold predictions + per-fold top-100 + fit time (s)."""
    yp = np.full(len(y), np.nan); tops = []; t0 = time.time()
    for tr, te in splits:
        m = model_fn(); m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te]); tops.append(top100(y[tr], y[te], yp[te]))
    assert not np.isnan(yp).any()
    return yp, tops, time.time() - t0


def metrics(y, yp):
    e = y - yp
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), SSE=float((e ** 2).sum()))


# ----------------------------------------------------------------------------- family bootstrap
def _sums(y, preds, unit):
    """Per-unit sufficient statistics: n, sum|e|, sum e^2 (per method), sum y, sum y^2."""
    codes, k = pd.factorize(unit)[0], len(np.unique(unit))
    cols = [np.ones(len(y)), y, y ** 2]
    for yp in preds: e = y - yp; cols += [np.abs(e), e ** 2]
    M = np.column_stack(cols)
    S = np.zeros((k, M.shape[1]))
    np.add.at(S, codes, M)
    return S


def family_bootstrap(y, preds: dict, unit, n_boot=N_BOOT, seed=BOOT_SEED, ref=None):
    """Bootstrap of whole UNITS (families or rows) with replacement; the same resample is applied to every
    prediction vector in `preds` (paired). Returns dict name -> {MAE, RMSE, R2 (+_lo/_hi)}, and for every
    ordered pair (a, b) with a != b the paired differences b - a of MAE, RMSE, R2 and the relative SSE
    reduction 1 - SSE_b/SSE_a (all percentile 2.5/97.5). Conditional on the fitted predictions."""
    names = list(preds); S = _sums(y, [preds[n] for n in names], unit); k = S.shape[0]
    rng = np.random.default_rng(seed)
    agg = np.empty((n_boot, S.shape[1]))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k); agg[b] = w @ S
    n, sy, sy2 = agg[:, 0], agg[:, 1], agg[:, 2]
    sst = sy2 - sy ** 2 / n
    stat = {}
    for i, nm in enumerate(names):
        sae, sse = agg[:, 3 + 2 * i], agg[:, 4 + 2 * i]
        stat[nm] = dict(MAE=sae / n, RMSE=np.sqrt(sse / n), R2=1 - sse / sst, SSE=sse)
    out = {}
    for nm in names:
        out[nm] = {m: float(np.mean(stat[nm][m])) for m in ("MAE", "RMSE", "R2")}
        for m in ("MAE", "RMSE", "R2"):
            lo, hi = np.percentile(stat[nm][m], [2.5, 97.5]); out[nm][f"{m}_lo"], out[nm][f"{m}_hi"] = float(lo), float(hi)
    pairs = {}
    for a in names:
        for b in names:
            if a == b: continue
            d = {}
            for m in ("MAE", "RMSE", "R2"):
                diff = stat[b][m] - stat[a][m]; lo, hi = np.percentile(diff, [2.5, 97.5])
                d[f"d{m}_lo"], d[f"d{m}_hi"] = float(lo), float(hi)
            red = 1 - stat[b]["SSE"] / stat[a]["SSE"]; lo, hi = np.percentile(red, [2.5, 97.5])
            d["SSE_red_lo"], d["SSE_red_hi"] = float(lo), float(hi)
            pairs[(a, b)] = d
    return out, pairs
