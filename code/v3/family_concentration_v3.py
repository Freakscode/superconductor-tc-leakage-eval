"""family_concentration_v3.py -- family-size concentration of both datasets (B on the audited v3 checkpoint).
Same definitions as results_v2/family_concentration.csv (code/v2/run_diagnostics_v2.py); A is unchanged, B is v3.
    python code/v3/family_concentration_v3.py   -> <RESULTS_V3>/family_concentration_v3.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import v3_common as C  # noqa: E402

rows = []
for ds in ("A", "B"):
    X, y, g, _, meta = C.load_dataset(ds)
    sizes = pd.Series(g).value_counts().values
    rows.append(dict(dataset=ds, checkpoint="unchanged" if ds == "A" else "v3", N=int(len(y)), k_familias=int(len(sizes)),
                     tam_medio=float(sizes.mean()), pct_en_familias_ge5=float(100 * sizes[sizes >= 5].sum() / sizes.sum()),
                     pct_en_familias_ge2=float(100 * sizes[sizes >= 2].sum() / sizes.sum()),
                     n_familias_ge5=int((sizes >= 5).sum()), n_singleton=int((sizes == 1).sum()), pct_singleton_filas=float(100 * (sizes == 1).sum() / sizes.sum()),
                     tam_familia_max=int(sizes.max()), top3_tam=";".join(str(int(s)) for s in sorted(sizes, reverse=True)[:3]),
                     procedencia=f"code/v3/family_concentration_v3.py | {C.VERS} | family = constituent-element set (v3_common.load_dataset groups)"))
out = pd.DataFrame(rows); out.to_csv(C.RES_V3 / "family_concentration_v3.csv", index=False)
v2 = pd.read_csv(C.RES_V2 / "family_concentration.csv")
assert abs(float(out[out.dataset == "A"].pct_en_familias_ge5.iloc[0]) - float(v2[v2.dataset == "A"].pct_en_familias_ge5.iloc[0])) < 1e-9
print(out.drop(columns="procedencia").to_string(index=False))
