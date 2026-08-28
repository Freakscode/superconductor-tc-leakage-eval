"""Capture datasetB_pipeline.featurize() output verbatim so the factorial can
reuse the live-matminer feature matrix without re-running featurization.
Saves BOTH the pre-mask arrays and the post-mask (Tc>0) arrays."""
import sys, time, importlib.util, json
import numpy as np
sys.dont_write_bytecode = True
REPO = "/Users/freakscode/Proyectos 2025/Proyectos Académicos/Superconductors/superconductor-tc-leakage-eval"
OUT  = "/Users/freakscode/Proyectos 2025/Proyectos Académicos/Superconductors/results_v2"

spec = importlib.util.spec_from_file_location("dsB", f"{REPO}/code/datasetB_pipeline.py")
dsB = importlib.util.module_from_spec(spec); spec.loader.exec_module(dsB)

t0 = time.time()
X, y, groups, formulas = dsB.featurize(f"{REPO}/data/supercon_stanev.csv")
t_feat = time.time() - t0

m = y > 0                                   # exactly what run() does next
np.savez_compressed(f"{OUT}/datasetB_featurized_live.npz",
                    X_all=X, y_all=y, groups_all=groups, formulas_all=formulas,
                    X=X[m], y=y[m], groups=groups[m], formulas=formulas[m])
meta = dict(featurize_seconds=t_feat, n_featurized=int(len(X)), n_features=int(X.shape[1]),
            n_used_Tcgt0=int(m.sum()), n_families_premask=int(len(set(groups))),
            n_families_postmask=int(len(set(groups[m]))),
            X_dtype=str(X.dtype), y_dtype=str(y.dtype),
            procedencia="datasetB_pipeline.featurize() via importlib, matminer ElementProperty magpie, live")
json.dump(meta, open(f"{OUT}/live_featurization_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
