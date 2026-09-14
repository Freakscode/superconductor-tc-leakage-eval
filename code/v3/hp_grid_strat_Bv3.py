"""hp_grid_strat_Bv3.py -- Table 3 grid rows and the StratifiedGroupKFold check for Dataset B on the AUDITED v3 checkpoint.

Re-runs code/v2/run_M4_m2.py's `grid` and `strat` modes with the only change being the input
(data/datasetB_featurized_v3.npz, N = 12 336, k = 3 046, groups = npz 'groups'); the grid, seeds, splitters and
metrics are imported from that script so the design is identical. Dataset A is unchanged (rows copied from v2).
Outputs (<RESULTS_V3>): hp_grid_ambos_v3.csv (A from v2 + B v3), stratified_group_check_v3.csv (A from v2 + B v3).
    python code/v3/hp_grid_strat_Bv3.py
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import v3_common as C  # noqa: E402

spec = importlib.util.spec_from_file_location("run_M4_m2", HERE.parent / "v2" / "run_M4_m2.py")
M4 = importlib.util.module_from_spec(spec); sys.modules["run_M4_m2"] = M4; spec.loader.exec_module(M4)
X, y, g, make_models, meta = C.load_dataset("B")
assert len(y) == 12336 and len(np.unique(g)) == 3046
M4.load_dataset = lambda ds: (X.astype(np.float32), y, g, make_models, None)   # only the input changes
M4.RAW = C.RES_V3 / "_m4_raw_v3"; os.makedirs(M4.RAW, exist_ok=True)
PROV = f"code/v3/hp_grid_strat_Bv3.py (re-run of code/v2/run_M4_m2.py grid/strat on data/datasetB_featurized_v3.npz) | {C.VERS}"

t0 = time.time(); M4.run_grid("B"); print(f"grid done {time.time()-t0:.0f}s", flush=True)
t0 = time.time(); M4.run_strat("B"); print(f"strat done {time.time()-t0:.0f}s", flush=True)

grid = pd.DataFrame(json.load(open(M4.RAW / "B_grid.json"))); grid["procedencia"] = PROV
v2g = pd.read_csv(C.RES_V2 / "hp_grid_ambos.csv")
pd.concat([v2g[v2g.dataset == "A"], grid], ignore_index=True).to_csv(C.RES_V3 / "hp_grid_ambos_v3.csv", index=False)
st = pd.DataFrame(json.load(open(M4.RAW / "B_strat.json")))
kf = st[st.splitter == "KFold"].set_index("seed")["MAE"]
st["MAE_KFold_misma_semilla"] = st["seed"].map(kf); st["inflacion_pct"] = 100 * (st["MAE"] / st["MAE_KFold_misma_semilla"] - 1); st["procedencia"] = PROV
v2s = pd.read_csv(C.RES_V2 / "stratified_group_check.csv")
pd.concat([v2s[v2s.dataset == "A"], st], ignore_index=True).to_csv(C.RES_V3 / "stratified_group_check_v3.csv", index=False)
pub = grid[grid.es_publicada]; assert abs(float(pub.MAE_random.iloc[0]) - 3.779) < 0.01 and abs(float(pub.MAE_grouped.iloc[0]) - 6.185) < 0.01, pub[["MAE_random", "MAE_grouped"]]
print(grid[["config", "MAE_random", "MAE_grouped", "inflacion_pct"]].round(3).to_string(index=False))
print(st[["splitter", "seed", "MAE", "inflacion_pct"]].round(2).to_string(index=False))
