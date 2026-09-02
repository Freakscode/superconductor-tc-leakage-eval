"""Capture datasetB_pipeline.featurize() output verbatim so the factorial can
reuse the live-matminer feature matrix without re-running featurization.
Saves BOTH the pre-mask arrays and the post-mask (Tc>0) arrays."""
import sys, os, time, importlib.util, json
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
# --- repository root (portable) -------------------------------------------------------
# ROOT = the repository root. Override with the SUPERCON_ROOT environment variable;
# by default it is resolved from this file's location: code/v2/<script>.py -> parents[2].
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
REPO = ROOT                       # the repository itself (code/, data/, results/, figures/)
CODE = ROOT / "code"              # tc_pipeline.py, datasetB_pipeline.py (hyperparameters live here)
DATA = ROOT / "data"
OUT  = ROOT / "results_v2"        # v2 measurements (inside the repository)

if "--help" in sys.argv or "-h" in sys.argv or "--dry-run" in sys.argv:
    print(__doc__); print(f"ROOT={ROOT}\nCODE={CODE}\nDATA={DATA}\nOUT={OUT}")
    for p in (CODE / "datasetB_pipeline.py", DATA / "supercon_stanev.csv"):
        print(f"  {'ok ' if p.exists() else 'MISSING'} {p}")
    sys.exit(0)

spec = importlib.util.spec_from_file_location("dsB", str(CODE / "datasetB_pipeline.py"))
dsB = importlib.util.module_from_spec(spec); spec.loader.exec_module(dsB)

t0 = time.time()
# superconductors_only=False: keep the PRE-mask arrays (this file captures both). NOTE: with
# False the family codes are factorized over all 16 406 rows (v1 convention); the post-mask
# codes saved below therefore have gaps and are NOT array_equal to data/datasetB_featurized.npz
# (same partition, different labels). featurize() with the default True gives the npz codes.
X, y, groups, formulas = dsB.featurize(str(DATA / "supercon_stanev.csv"), superconductors_only=False)
t_feat = time.time() - t0

m = y > 0                                   # exactly what run() does next
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / "datasetB_featurized_live.npz",
                    X_all=X, y_all=y, groups_all=groups, formulas_all=formulas,
                    X=X[m], y=y[m], groups=groups[m], formulas=formulas[m])
meta = dict(featurize_seconds=t_feat, n_featurized=int(len(X)), n_features=int(X.shape[1]),
            n_used_Tcgt0=int(m.sum()), n_families_premask=int(len(set(groups))),
            n_families_postmask=int(len(set(groups[m]))),
            X_dtype=str(X.dtype), y_dtype=str(y.dtype),
            procedencia="datasetB_pipeline.featurize() via importlib, matminer ElementProperty magpie, live")
json.dump(meta, open(OUT / "live_featurization_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
