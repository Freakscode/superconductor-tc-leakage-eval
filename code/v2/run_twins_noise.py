"""Gemelos exactos de features test↔train y piso de ruido de etiqueta (auditoría V2/V3). Dataset A."""
import os, sys, importlib.util, numpy as np, pandas as pd, sklearn
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
REPO = ROOT; OUT = ROOT/"results_v2"
spec = importlib.util.spec_from_file_location("tcp", REPO/"code/tc_pipeline.py"); tcp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tcp)
tr = pd.read_csv(REPO/"data/train.csv"); um = pd.read_csv(REPO/"data/unique_m.csv")
feat = [c for c in tr.columns if c != "critical_temp"]; y = tr.critical_temp.values; g = tcp.chemical_families(um)
key = pd.util.hash_pandas_object(tr[feat], index=False).values
def twin(splitter, groups=None):
    fr = []
    for a, b in splitter.split(tr[feat].values, y, groups):
        s = set(key[a]); fr.append(np.mean([k in s for k in key[b]]))
    return np.mean(fr), np.std(fr)
rk, rks = twin(KFold(5, shuffle=True, random_state=42)); gk, gks = twin(GroupKFold(5), g)
d = pd.DataFrame({"k": key, "tc": y}); grp = d.groupby("k").tc
multi = grp.filter(lambda s: len(s) > 1); dev = (multi - d.loc[multi.index, "k"].map(grp.mean())).abs()
prov = f"script=results_v2/run_twins_noise.py | sklearn {sklearn.__version__} | KFold(5,shuffle,rs=42) vs GroupKFold(5) familias | gemelo = hash idéntico de las 81 features"
# MAE del modelo publicado (XGBoost, KFold aleatorio rs=42) restringida a las filas con gemelo, para comparar con el piso
yp = np.empty_like(y, dtype=float)
for a, b in KFold(5, shuffle=True, random_state=42).split(tr[feat].values):
    m = tcp.make_models(42)["XGBoost"]; m.fit(tr[feat].values[a], y[a]); yp[b] = m.predict(tr[feat].values[b])
err = np.abs(y - yp); idx = multi.index.values
mae_twin, mae_rest = err[idx].mean(), np.delete(err, idx).mean()
rows = [dict(dataset="A", metrica="frac_test_con_gemelo_exacto_en_train", split="random_KFold5_rs42", valor=rk, sd_entre_folds=rks, n=len(y), procedencia=prov),
        dict(dataset="A", metrica="frac_test_con_gemelo_exacto_en_train", split="GroupKFold5_familias", valor=gk, sd_entre_folds=gks, n=len(y), procedencia=prov),
        dict(dataset="A", metrica="filas_con_features_duplicadas", split="-", valor=int(tr.duplicated(subset=feat).sum()), sd_entre_folds=np.nan, n=len(y), procedencia=prov),
        dict(dataset="A", metrica="filas_duplicadas_features_y_Tc", split="-", valor=int(tr.duplicated().sum()), sd_entre_folds=np.nan, n=len(y), procedencia=prov),
        dict(dataset="A", metrica="n_filas_con_gemelo", split="-", valor=len(multi), sd_entre_folds=np.nan, n=len(y), procedencia=prov),
        dict(dataset="A", metrica="piso_ruido_MAE_solo_filas_con_gemelo_K", split="-", valor=dev.mean(), sd_entre_folds=np.nan, n=len(multi), procedencia=prov+" | media |Tc - media del grupo de gemelos|"),
        dict(dataset="A", metrica="piso_ruido_mediana_solo_filas_con_gemelo_K", split="-", valor=dev.median(), sd_entre_folds=np.nan, n=len(multi), procedencia=prov),
        dict(dataset="A", metrica="pct_filas_features_duplicadas", split="-", valor=100*tr.duplicated(subset=feat).sum()/len(y), sd_entre_folds=np.nan, n=len(y), procedencia=prov),
        dict(dataset="A", metrica="MAE_random_modelo_solo_filas_con_gemelo_K", split="random_KFold5_rs42", valor=mae_twin, sd_entre_folds=np.nan, n=len(multi), procedencia=prov+" | XGBoost make_models(42)"),
        dict(dataset="A", metrica="MAE_random_modelo_filas_sin_gemelo_K", split="random_KFold5_rs42", valor=mae_rest, sd_entre_folds=np.nan, n=len(y)-len(multi), procedencia=prov+" | XGBoost make_models(42)"),
        dict(dataset="A", metrica="cota_inferior_MAE_global_aportada_K", split="-", valor=len(multi)/len(y)*dev.mean(), sd_entre_folds=np.nan, n=len(y), procedencia=prov+" | = frac_gemelos * piso")]
pd.DataFrame(rows).to_csv(OUT/"twins_noise_A.csv", index=False)
print(pd.DataFrame(rows)[["metrica","split","valor","n"]].to_string(index=False))
