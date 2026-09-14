"""Shared helpers for the v3 INFERENCE scripts (H01, H07, H08, H11; owner: oof_v3.py and friends).

Kept under a task-specific name so it does not collide with other v3 helper modules (v3_common.py was
overwritten by a concurrent task on 2026-09-14). Contents:
  * dataset loaders (A; B v3 = canonical for this phase; B_v2 only for comparison),
  * the two partition protocols of the paper (random KFold rs=42; deterministic GroupKFold(5)),
  * the family-mean null predictor (same definition as code/v2/run_v2_measurements.py),
  * per-family sufficient statistics + a paired bootstrap of WHOLE FAMILIES for the contrast
    random - grouped of pooled MAE / RMSE / R2 and of the macro (per-family) MAE.

Paths: ROOT = repo root (from __file__), override with SUPERCON_ROOT. Results of this phase go
to RES_V3 = ROOT/../results_v3 (override SUPERCON_RESULTS_V3).
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
SEED = 42
N_JOBS = 8
VERS = (f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
        f"scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def load_dataset(ds: str):
    """Returns X, y, groups, make_models, meta for ds in {'A', 'B', 'B_v3', 'B_v2'}. 'B' == B v3."""
    if ds == "A":
        tcp = load_mod("tc_pipeline", str(CODE / "tc_pipeline.py"))
        feat = pd.read_csv(DATA / "train.csv"); uniq = pd.read_csv(DATA / "unique_m.csv")
        assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "train.csv / unique_m.csv misaligned"
        X = feat.drop(columns=["critical_temp"]).values.astype(float)
        y = feat["critical_temp"].values.astype(float)
        groups = tcp.chemical_families(uniq)
        return X, y, groups, tcp.make_models, {"unique_m": uniq, "splits": SPLITS / "datasetA_folds.csv",
                                                "source": "data/train.csv + data/unique_m.csv"}
    if ds in ("B", "B_v3", "B_v2"):
        dbp = load_mod("datasetB_pipeline", str(CODE / "datasetB_pipeline.py"))
        fn = "datasetB_featurized_v3.npz" if ds != "B_v2" else "datasetB_featurized.npz"
        z = np.load(DATA / fn, allow_pickle=True)
        meta = {"splits": SPLITS / ("datasetB_folds_v3.csv" if ds != "B_v2" else "datasetB_folds.csv"),
                "source": f"data/{fn}"}
        for k in ("formula", "row_origen"):
            if k in z.files: meta[k] = z[k]
        return z["X"].astype(float), z["y"].astype(float), z["groups"].astype(int), dbp.make_models, meta
    raise ValueError(f"unknown dataset {ds!r}")


def with_n_jobs(model, n_jobs=N_JOBS):
    """Thread count only (bit-identical results for RF; XGBoost hist checked against v2 in oof_v3.py)."""
    return model.set_params(n_jobs=n_jobs)


def protocols(groups, seed=SEED):
    return {"random": (KFold(5, shuffle=True, random_state=seed), None),
            "grouped": (GroupKFold(5), groups)}


def fold_labels(cv, X, y, groups=None):
    lab = np.full(len(y), -1, dtype=int)
    it = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for k, (_, te) in enumerate(it):
        lab[te] = k
    assert (lab >= 0).all()
    return lab


def check_splits(ds, X, y, groups, splits_csv, seed=SEED):
    """GroupKFold(5) == fold_deterministic of splits/; KFold(rs=0) == random_fold_seed0 (splitter identity)."""
    sp = pd.read_csv(splits_csv, comment="#")
    assert len(sp) == len(y) and np.array_equal(sp["row_index"].values, np.arange(len(y))), f"{ds}: splits length/row_index"
    assert np.array_equal(sp["family_id"].values, groups), f"{ds}: family_id of splits != groups"
    assert np.array_equal(fold_labels(GroupKFold(5), X, y, groups), sp["fold_deterministic"].values), f"{ds}: GroupKFold(5) != fold_deterministic"
    assert np.array_equal(fold_labels(KFold(5, shuffle=True, random_state=0), X, y), sp["random_fold_seed0"].values), f"{ds}: KFold(rs=0) != random_fold_seed0"
    return sp


def family_mean_null(y_tr, g_tr, g_te):
    """Featureless predictor: mean Tc of the family in the TRAINING fold; global train mean if unseen
    (code/v2/run_v2_measurements.py::evaluate, rep_is_null_lookup=True)."""
    gmean = y_tr.mean()
    fam = pd.Series(y_tr).groupby(g_tr).mean().to_dict()
    return np.array([fam.get(g, gmean) for g in g_te], dtype=float)


def metrics(yt, yp):
    e = yt - yp
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                R2=float(1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum()))


# ----------------------------------------------------------------------------- family bootstrap
class FamilyStats:
    """Per-family sufficient statistics for prediction vectors on the SAME rows.
    Pooled MAE/RMSE/R2 and macro MAE of a resample are functions of per-family sums, so a bootstrap
    draw is a bincount of family indices - no row-level work per replicate."""

    def __init__(self, y, groups, preds: dict):
        codes, uniq = pd.factorize(np.asarray(groups))
        self.k = len(uniq); self.codes = codes; self.family_ids = uniq
        self.n = np.bincount(codes, minlength=self.k).astype(float)
        self.sy = np.bincount(codes, weights=y, minlength=self.k)
        self.sy2 = np.bincount(codes, weights=y ** 2, minlength=self.k)
        self.sabs, self.ssq = {}, {}
        for name, yp in preds.items():
            e = y - yp
            self.sabs[name] = np.bincount(codes, weights=np.abs(e), minlength=self.k)
            self.ssq[name] = np.bincount(codes, weights=e ** 2, minlength=self.k)

    def stats(self, w, name):
        """w = multiplicity of each family in the resample (all ones = original sample)."""
        N = (w * self.n).sum()
        mae = (w * self.sabs[name]).sum() / N
        sse = (w * self.ssq[name]).sum()
        sst = (w * self.sy2).sum() - (w * self.sy).sum() ** 2 / N
        macro = (w * (self.sabs[name] / self.n)).sum() / w.sum()
        return dict(MAE=mae, RMSE=np.sqrt(sse / N), R2=1 - sse / sst, MAE_macro=macro)


def family_draws(k, n_boot, seed):
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(0, k, k), minlength=k).astype(float) for _ in range(n_boot)]


def paired_family_bootstrap(fs: FamilyStats, conds, n_boot=2000, seed=0):
    """Resample families with replacement (same draw for all conditions). Returns {stat_cond: array}."""
    conds = list(conds)
    out = {f"{m}_{c}": np.empty(n_boot) for c in conds for m in ("MAE", "RMSE", "R2", "MAE_macro")}
    for i, w in enumerate(family_draws(fs.k, n_boot, seed)):
        for c in conds:
            for m, v in fs.stats(w, c).items(): out[f"{m}_{c}"][i] = v
    return out


def paired_row_bootstrap(y, pa, pb, n_boot=2000, seed=0, paired=True):
    """Row bootstrap (the v2 scheme), paired or with independent draws per arm."""
    rng = np.random.default_rng(seed); n = len(y)
    ea, eb = np.abs(y - pa), np.abs(y - pb); qa, qb = (y - pa) ** 2, (y - pb) ** 2
    out = {f"{m}_{c}": np.empty(n_boot) for c in ("a", "b") for m in ("MAE", "RMSE", "R2")}
    for i in range(n_boot):
        ia = rng.integers(0, n, n); ib = ia if paired else rng.integers(0, n, n)
        for c, idx, e, q in (("a", ia, ea, qa), ("b", ib, eb, qb)):
            sst = ((y[idx] - y[idx].mean()) ** 2).sum()
            out[f"MAE_{c}"][i] = e[idx].mean(); out[f"RMSE_{c}"][i] = np.sqrt(q[idx].mean()); out[f"R2_{c}"][i] = 1 - q[idx].sum() / sst
    return out


def ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))
