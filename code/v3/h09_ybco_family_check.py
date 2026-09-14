#!/usr/bin/env python
"""H09 check: the constituent-element set {Ba,Cu,O,Y} is an *operational* grouping,
not a one-to-one doping series. Measures, in Dataset A (unique_m.csv) and Dataset B
(supercon_stanev.csv, Tc>0, pymatgen-parsable), how many rows the Y-Ba-Cu-O family
holds, how many distinct formulas, the Tc range, and the distribution of the Cu/Y
ratio -- which separates the 123 (YBa2Cu3O7-d), 124 (YBa2Cu4O8) and 247
(Y2Ba4Cu7O15-d) phases that share one element set but have different structures.

Output: <SUPERCON_ROOT>/results_v3/h09_ybco_family_check.json  (with procedencia).
Runs from any cwd; paths resolved from __file__ (override with SUPERCON_ROOT).
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("SUPERCON_ROOT", REPO.parent))
OUT = ROOT / "results_v3"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "code"))

import tc_pipeline as tp                      # noqa: E402  (repo module)
from pymatgen.core import Composition         # noqa: E402
import pymatgen.core                          # noqa: E402

FAMILY = "Ba|Cu|O|Y"


def ratio_counts(r: pd.Series, n: int = 8) -> dict:
    vc = r.round(2).value_counts().head(n)
    return {f"{k:.2f}": int(v) for k, v in vc.items()}


def main() -> int:
    # ---- Dataset A -------------------------------------------------------
    um = pd.read_csv(REPO / "data" / "unique_m.csv")
    fam_ids = tp.chemical_families(um)               # same function as the paper
    elem_cols = [c for c in um.columns if c not in ("critical_temp", "material")]
    present = um[elem_cols] > 0
    key = present.apply(lambda r: "|".join(sorted(e for e in elem_cols if r[e])), axis=1)
    a = um[key == FAMILY]
    assert (a["Y"] > 0).all() and (a["Cu"] > 0).all()
    assert len(set(fam_ids[key == FAMILY])) == 1, "family id must be unique for one element set"
    cu_y_A = a["Cu"] / a["Y"]

    # ---- Dataset B -------------------------------------------------------
    B = pd.read_csv(REPO / "data" / "supercon_stanev.csv")
    B = B[B.Tc > 0].reset_index(drop=True)

    def elset(f):
        try:
            return "|".join(sorted(str(e) for e in Composition(f).elements))
        except Exception:
            return None

    keyB = B["name"].map(elset)
    n_parse_fail = int(keyB.isna().sum())
    B = B[~keyB.isna()].reset_index(drop=True)
    keyB = keyB.dropna().reset_index(drop=True)
    assert len(B) == 12440, f"expected 12440 Dataset-B rows after Tc>0 and parse filter, got {len(B)}"
    b = B[keyB == FAMILY]

    def cu_over_y(f):
        c = Composition(f)
        return c["Cu"] / c["Y"]

    cu_y_B = b["name"].map(cu_over_y)

    res = {
        "family": FAMILY,
        "datasetA": {
            "rows_total": int(len(um)),
            "family_rows": int(len(a)),
            "family_distinct_formulas": int(a["material"].nunique()),
            "family_Tc_min_K": float(a.critical_temp.min()),
            "family_Tc_max_K": float(a.critical_temp.max()),
            "family_Tc_sd_K": float(a.critical_temp.std()),
            "CuY_ratio_counts": ratio_counts(cu_y_A),
            "rows_CuY_eq_3_(123-type)": int((cu_y_A.round(2) == 3.0).sum()),
            "rows_CuY_eq_3.5_(247-type)": int((cu_y_A.round(2) == 3.5).sum()),
            "rows_CuY_eq_4_(124-type)": int((cu_y_A.round(2) == 4.0).sum()),
            "example_formulas_CuY_4": a.loc[cu_y_A.round(2) == 4.0, "material"].drop_duplicates().head(5).tolist(),
        },
        "datasetB": {
            "rows_Tc_pos_before_parse_filter": int(len(B) + n_parse_fail),
            "rows_parse_failures_dropped": n_parse_fail,
            "rows_total": int(len(B)),
            "family_rows": int(len(b)),
            "family_distinct_formulas": int(b["name"].nunique()),
            "family_Tc_min_K": float(b.Tc.min()),
            "family_Tc_max_K": float(b.Tc.max()),
            "CuY_ratio_counts": ratio_counts(cu_y_B),
            "rows_CuY_eq_3_(123-type)": int((cu_y_B.round(2) == 3.0).sum()),
            "rows_CuY_eq_3.5_(247-type)": int((cu_y_B.round(2) == 3.5).sum()),
            "rows_CuY_eq_4_(124-type)": int((cu_y_B.round(2) == 4.0).sum()),
            "example_formulas_CuY_4": b.loc[cu_y_B.round(2) == 4.0, "name"].drop_duplicates().head(5).tolist(),
        },
        "interpretacion": (
            "Un unico conjunto de elementos {Ba,Cu,O,Y} agrupa fases con estequiometria Cu/Y = 3, 3.5 y 4 "
            "(estructuras 123, 247 y 124), con Tc entre ~4 K y >100 K. El grupo es una agrupacion operativa por "
            "sistema elemental, no una serie de dopaje uno-a-uno."
        ),
        "procedencia": {
            "script": "code/v3/h09_ybco_family_check.py",
            "definicion_familia": "tc_pipeline.chemical_families (A); pymatgen Composition + factorize tras Tc>0 (B)",
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "pymatgen": pymatgen.core.__version__,
            "semilla": "no aplica (conteo determinista)",
            "datos": ["data/unique_m.csv", "data/supercon_stanev.csv"],
        },
    }
    (OUT / "h09_ybco_family_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res["datasetA"].items() if "example" not in k}, indent=0))
    print(json.dumps({k: v for k, v in res["datasetB"].items() if "example" not in k}, indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
