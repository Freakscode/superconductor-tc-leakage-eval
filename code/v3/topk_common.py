"""topk_common.py -- shared loaders and tie-aware top-K selection metrics (v3, H05/H06).

Used by code/v3/topk_prevalence_ties.py, topk_metrics.py, topk_negatives.py.

Conventions fixed here (declared in every output file's `procedencia`):
  * high-Tc threshold = np.percentile(y[TRAIN fold], 90); a test row is a "positive" (hit) when
    y >= thr  (operator >=, the one used by tc_pipeline.topk_precision / datasetB_pipeline.topk);
  * random protocol = KFold(5, shuffle=True, random_state=42); grouped = GroupKFold(5) deterministic,
    verified against splits/*.csv (fold_deterministic);
  * family = constituent-element set (A: tc_pipeline.chemical_families(unique_m); B: `groups` of the
    npz checkpoint);
  * expected precision@K under ties (H06): rows sorted by predicted Tc descending; tie groups are
    consumed whole while they fit in K; when the cut falls inside a tie group of total weight t with
    positive weight h and r = K - (weight already taken) slots left, the group contributes r*h/t
    expected hits (= expectation over a uniformly random ordering of the tied rows). Row weights
    (bootstrap multiplicities) generalise t, h, r to weighted counts.
"""
from __future__ import annotations
import importlib.util, os, sys, platform
from pathlib import Path
import numpy as np, pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
CODE, DATA, SPLITS = ROOT / "code", ROOT / "data", ROOT / "splits"
RES_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
SEED = 42
N_JOBS = 8
KS = (10, 25, 50, 100, 200)
OOF_CACHE = RES_V3 / "topk_oof_predictions.csv.gz"


def read_csv(path, **kw):
    """pandas maps the literal string "null" to NaN by default; our ranker is called "null"."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kw)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def versions() -> str:
    import sklearn, xgboost
    from importlib.metadata import version as _v
    return (f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
            f"scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}, matminer {_v('matminer')}, "
            f"pymatgen {_v('pymatgen')}")


# ------------------------------------------------------------------------------------ data
def element_indicators(formulas) -> tuple[np.ndarray, list[str]]:
    """Binary element-presence matrix (the v2 'presence' representation, pymatgen Composition)."""
    from pymatgen.core import Composition
    sets = [sorted(e.symbol for e in Composition(f).elements) for f in formulas]
    elems = sorted({e for s in sets for e in s}); jdx = {e: j for j, e in enumerate(elems)}
    X = np.zeros((len(formulas), len(elems)), dtype=np.float64)
    for i, s in enumerate(sets):
        for e in s:
            X[i, jdx[e]] = 1.0
    return X, elems


def load_A() -> dict:
    tcp = load_mod("tc_pipeline", str(CODE / "tc_pipeline.py"))
    feat = pd.read_csv(DATA / "train.csv"); uniq = pd.read_csv(DATA / "unique_m.csv")
    assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "train.csv / unique_m.csv misaligned"
    X = feat.drop(columns=["critical_temp"]).values.astype(np.float64)
    y = feat["critical_temp"].values.astype(np.float64)
    groups = tcp.chemical_families(uniq)
    elem_cols = [c for c in uniq.columns if c not in ("critical_temp", "material")]
    Xp = (uniq[elem_cols].values > 0).astype(np.float64)
    folds = pd.read_csv(SPLITS / "datasetA_folds.csv", comment="#")
    return dict(name="A", X=X, y=y, groups=groups, X_pres=Xp, folds=folds, make_models=tcp.make_models,
                formula=uniq["material"].values.astype(str), n_pres=len(elem_cols))


def load_B(version: str = "v3") -> dict:
    dsB = load_mod("datasetB_pipeline", str(CODE / "datasetB_pipeline.py"))
    npz = DATA / ("datasetB_featurized_v3.npz" if version == "v3" else "datasetB_featurized.npz")
    z = np.load(npz, allow_pickle=True)
    X = z["X"].astype(np.float64); y = z["y"].astype(np.float64); groups = z["groups"].astype(np.int64)
    if "formula" in z.files:
        formula = z["formula"].astype(str)
    else:   # v2 checkpoint stores no formula: rebuild the alignment from the source (parseable & Tc>0)
        from pymatgen.core import Composition
        sc = pd.read_csv(DATA / "supercon_stanev.csv"); keep = []
        for name, tc in zip(sc["name"], sc["Tc"]):
            try:
                Composition(name)
            except Exception:
                continue
            if tc > 0:
                keep.append(name)
        formula = np.array(keep, dtype=str); assert len(formula) == len(y), "v2 alignment failed"
    Xp, elems = element_indicators(formula)
    folds = pd.read_csv(SPLITS / ("datasetB_folds_v3.csv" if version == "v3" else "datasetB_folds.csv"), comment="#")
    return dict(name=f"B{version}", X=X, y=y, groups=groups, X_pres=Xp, folds=folds, make_models=dsB.make_models,
                formula=formula, n_pres=len(elems), row_origen=(z["row_origen"] if "row_origen" in z.files else None))


def fold_labels(splitter, n, groups=None) -> np.ndarray:
    lab = np.full(n, -1, dtype=int); X0 = np.zeros((n, 1))
    it = splitter.split(X0, None, groups) if groups is not None else splitter.split(X0)
    for k, (_, te) in enumerate(it):
        lab[te] = k
    assert (lab >= 0).all()
    return lab


def protocol_folds(d: dict) -> dict[str, np.ndarray]:
    """Fold labels for both protocols, cross-checked against splits/*.csv."""
    from sklearn.model_selection import KFold, GroupKFold
    n = len(d["y"])
    grouped = fold_labels(GroupKFold(5), n, d["groups"])
    assert np.array_equal(grouped, d["folds"]["fold_deterministic"].values), f"{d['name']}: GroupKFold(5) != splits fold_deterministic"
    ctrl = fold_labels(KFold(5, shuffle=True, random_state=0), n)
    assert np.array_equal(ctrl, d["folds"]["random_fold_seed0"].values), f"{d['name']}: KFold(rs=0) != splits random_fold_seed0"
    random = fold_labels(KFold(5, shuffle=True, random_state=SEED), n)
    return {"random": random, "grouped": grouped}


def family_mean_null(y, groups, tr, te) -> np.ndarray:
    gmean = y[tr].mean(); fam = pd.Series(y[tr]).groupby(groups[tr]).mean().to_dict()
    return np.array([fam.get(g, gmean) for g in groups[te]], dtype=float)


def fit_predict(model, Xtr, ytr, Xte) -> np.ndarray:
    try:
        model.set_params(n_jobs=N_JOBS)
    except Exception:
        pass
    model.fit(Xtr, ytr); return model.predict(Xte)


# ------------------------------------------------------------------------------------ selection metrics
def tie_groups(pred_sorted: np.ndarray) -> np.ndarray:
    """Group id (0..G-1) of each row of a DESCENDING-sorted prediction vector; equal preds share an id."""
    return np.concatenate([[0], np.cumsum(np.diff(pred_sorted) != 0)])


def expected_hits(pred, hit, Ks, w=None):
    """Expected number of positives among the top-K predicted rows, fractional under ties (see module
    docstring). Returns dict K -> expected hits. `w` = non-negative row weights (default 1)."""
    pred = np.asarray(pred, float); hit = np.asarray(hit, float)
    w = np.ones_like(pred) if w is None else np.asarray(w, float)
    keep = w > 0
    pred, hit, w = pred[keep], hit[keep], w[keep]
    order = np.argsort(-pred, kind="stable")
    p, h, ww = pred[order], hit[order], w[order]
    g = tie_groups(p)
    T = np.bincount(g, weights=ww); H = np.bincount(g, weights=ww * h)
    cumT = np.cumsum(T); prevT = cumT - T
    out = {}
    for K in Ks:
        idx = np.searchsorted(cumT, K, side="right")          # first group NOT fully inside K
        hits = H[:idx].sum()
        if idx < len(T):
            r = K - prevT[idx]
            hits += r * H[idx] / T[idx]
        out[K] = float(hits)
    return out


def random_tiebreak_order(pred, rng) -> np.ndarray:
    """Descending order of pred with ties broken uniformly at random."""
    return np.lexsort((rng.random(len(pred)), -np.asarray(pred, float)))


def legacy_topk_idx(pred, K) -> np.ndarray:
    """The selection used by the published code: np.argsort(pred)[-K:] (ties -> position order)."""
    return np.argsort(np.asarray(pred, float))[-K:]


def budget_selection(order, fam, K, max_per_family=3) -> np.ndarray:
    """Walk the ranking `order`; take a row only if its family has < max_per_family rows already taken."""
    taken = {}; sel = []
    for i in order:
        f = fam[i]
        if taken.get(f, 0) < max_per_family:
            taken[f] = taken.get(f, 0) + 1; sel.append(i)
            if len(sel) == K:
                break
    return np.array(sel, dtype=int)
