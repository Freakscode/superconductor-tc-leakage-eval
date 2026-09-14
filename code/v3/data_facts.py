"""Facts quoted in the manuscript that are computed directly from the raw data files and had
no result-file driver in v2 (e.g. "the {Ba,Cu,O,Y} family holds 720 compounds, 64 % of them
above 77 K").  Functions are called live by code/v3/verify_manuscript.py (manifest entries of
type `python`); `python code/v3/data_facts.py` writes them to results_v3/data_facts.csv with a
`procedencia` column.

Requires data/train.csv + data/unique_m.csv (code/get_datasetA.py) for Dataset A and
data/datasetB_featurized.npz for Dataset B.
"""
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
DATA, CODE = ROOT / "data", ROOT / "code"
RES_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
_cache: dict = {}


def load(ds: str):
    """-> (y, groups, family_key) for dataset 'A' or 'B' (family = constituent-element set)."""
    if ds in _cache: return _cache[ds]
    if ds == "A":
        spec = importlib.util.spec_from_file_location("tc_pipeline", CODE / "tc_pipeline.py")
        tcp = importlib.util.module_from_spec(spec); sys.modules["tc_pipeline"] = tcp; spec.loader.exec_module(tcp)
        feat = pd.read_csv(DATA / "train.csv"); uniq = pd.read_csv(DATA / "unique_m.csv")
        assert np.allclose(feat["critical_temp"], uniq["critical_temp"])
        y = feat["critical_temp"].values.astype(float); groups = tcp.chemical_families(uniq)
        elem = [c for c in uniq.columns if c not in ("critical_temp", "material")]
        key = (uniq[elem] > 0).apply(lambda r: "|".join(sorted(e for e in elem if r[e])), axis=1).values
    elif ds == "B":
        z = np.load(DATA / "datasetB_featurized.npz", allow_pickle=True)
        y, groups = z["y"].astype(float), z["groups"].astype(int); key = None
    else:
        raise ValueError(ds)
    _cache[ds] = (y, groups, key)
    return _cache[ds]


def largest_family(ds: str):
    y, groups, key = load(ds)
    code = int(np.bincount(groups).argmax()); m = groups == code
    return {"family_id": code, "family_key": (key[m][0] if key is not None else None), "n": int(m.sum()), "y": y[m]}


def largest_family_pct_above(ds: str, thr: float) -> float:
    """Percentage of the largest family's rows with Tc > thr (strict)."""
    f = largest_family(ds)
    return float(100 * (f["y"] > thr).mean())


def largest_family_size(ds: str) -> int:
    return largest_family(ds)["n"]


if __name__ == "__main__":
    RES_V3.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in ("A", "B"):
        try:
            f = largest_family(ds)
        except FileNotFoundError as e:
            print(f"[{ds}] skipped: {e}"); continue
        for thr in (77.0,):
            rows.append({"dataset": ds, "fact": f"largest_family_pct_Tc_gt_{thr:g}K", "value": largest_family_pct_above(ds, thr),
                         "family_id": f["family_id"], "family_key": f["family_key"], "n_family": f["n"],
                         "procedencia": "code/v3/data_facts.py | family = constituent-element set (A: tc_pipeline.chemical_families(unique_m.csv); B: npz groups); strict Tc > thr"})
            rows.append({"dataset": ds, "fact": f"largest_family_pct_Tc_ge_{thr:g}K", "value": float(100 * (f["y"] >= thr).mean()),
                         "family_id": f["family_id"], "family_key": f["family_key"], "n_family": f["n"],
                         "procedencia": "code/v3/data_facts.py | same, Tc >= thr"})
    out = pd.DataFrame(rows); out.to_csv(RES_V3 / "data_facts.csv", index=False)
    print(out.drop(columns="procedencia").to_string()); print(f"wrote {RES_V3 / 'data_facts.csv'}")
