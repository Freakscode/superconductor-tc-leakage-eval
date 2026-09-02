"""Diagnósticos restantes para el v2: distancia NN, concentración familiar, check de reproducción."""
import json, os, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- repository root (portable) -------------------------------------------------------
# ROOT = the repository root. Override with the SUPERCON_ROOT environment variable;
# by default it is resolved from this file's location: code/v2/<script>.py -> parents[2].
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
REPO = ROOT                       # the repository itself (code/, data/, results/, figures/)
CODE = ROOT / "code"              # tc_pipeline.py, datasetB_pipeline.py (hyperparameters live here)
DATA = ROOT / "data"
OUT  = ROOT / "results_v2"        # v2 measurements (inside the repository)

if "--help" in sys.argv or "-h" in sys.argv or "--dry-run" in sys.argv:
    print(__doc__); print(f"ROOT={ROOT}\nDATA={DATA}\nOUT={OUT}")
    for p in (CODE / "tc_pipeline.py", CODE / "datasetB_pipeline.py", DATA / "train.csv",
              DATA / "unique_m.csv", DATA / "datasetB_featurized.npz"):
        print(f"  {'ok ' if p.exists() else 'MISSING'} {p}")
    print("usage: python run_diagnostics_v2.py [--dry-run]  (~8 s; writes nn_distance.csv, "
          "family_concentration.csv, reproduction_check.csv into results_v2/)")
    sys.exit(0)
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
tcp = load("tcp", str(CODE / "tc_pipeline.py"))
dbp = load("dbp", str(CODE / "datasetB_pipeline.py"))

feat = pd.read_csv(DATA / "train.csv"); uniq = pd.read_csv(DATA / "unique_m.csv")
XA = feat.drop(columns=["critical_temp"]).values; yA = feat["critical_temp"].values
gA = tcp.chemical_families(uniq)
z = np.load(DATA / "datasetB_featurized.npz"); XB, yB, gB = z["X"], z["y"], z["groups"]

SEED = 42
rows_nn, rows_conc, rows_rep = [], [], []
for ds, X, y, g, mk in [("A", XA, yA, gA, lambda: tcp.make_models(SEED)["XGBoost"]),
                        ("B", XB, yB, gB, lambda: dbp.make_models(SEED)["XGBoost"])]:
    Xs = StandardScaler().fit_transform(X)
    nn = lambda a, b: NearestNeighbors(n_neighbors=1).fit(Xs[a]).kneighbors(Xs[b])[0].ravel()
    ri = np.arange(len(X)); np.random.RandomState(SEED).shuffle(ri); cut = int(0.8*len(X))
    g_tr, g_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, g))
    assert not (set(g[g_tr]) & set(g[g_te])), "fuga de familias"
    dr, dg = nn(ri[:cut], ri[cut:]), nn(g_tr, g_te)
    rows_nn.append(dict(dataset=ds, nn_median_random=float(np.median(dr)), nn_median_grouped=float(np.median(dg)),
        ratio=float(np.median(dg)/np.median(dr)), nn_p25_random=float(np.percentile(dr,25)),
        nn_p75_random=float(np.percentile(dr,75)), nn_p25_grouped=float(np.percentile(dg,25)),
        nn_p75_grouped=float(np.percentile(dg,75)), n_test_random=len(dr), n_test_grouped=len(dg),
        procedencia=f"run_diagnostics_v2.py | StandardScaler global; 1-NN euclidiana; hold-out 80/20 "
                    f"(random_state={SEED}) vs GroupShuffleSplit(test_size=0.2, random_state={SEED})"))
    s = pd.Series(g).value_counts()
    rows_conc.append(dict(dataset=ds, N=len(y), k_familias=int(s.size), tam_medio=float(len(y)/s.size),
        pct_en_familias_ge5=float(100*s[s>=5].sum()/len(y)), n_familias_ge5=int((s>=5).sum()),
        n_singleton=int((s==1).sum()), tam_familia_max=int(s.max()),
        top3_tam=";".join(str(int(v)) for v in s.head(3).values),
        procedencia="run_diagnostics_v2.py | familias = conjunto de elementos vía pandas.factorize"))
    # check de reproducción: split único 80/20 aleatorio, hiperparámetros de la fuente
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    m = mk(); m.fit(Xtr, ytr); p = m.predict(Xte)
    rows_rep.append(dict(dataset=ds, MAE=float(mean_absolute_error(yte,p)),
        RMSE=float(np.sqrt(mean_squared_error(yte,p))), R2=float(r2_score(yte,p)),
        n_train=len(ytr), n_test=len(yte),
        procedencia=f"run_diagnostics_v2.py | split único 80/20 random_state={SEED}; "
                    f"XGBoost de {'tc_pipeline' if ds=='A' else 'datasetB_pipeline'}.make_models"))

for rows, f in [(rows_nn,"nn_distance.csv"), (rows_conc,"family_concentration.csv"), (rows_rep,"reproduction_check.csv")]:
    OUT.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(OUT / f, index=False); print(f"→ {OUT / f}")
print(pd.DataFrame(rows_nn)[["dataset","nn_median_random","nn_median_grouped","ratio"]].to_string(index=False))
print(pd.DataFrame(rows_conc)[["dataset","N","k_familias","pct_en_familias_ge5","top3_tam"]].to_string(index=False))
print(pd.DataFrame(rows_rep)[["dataset","MAE","RMSE","R2"]].to_string(index=False))
