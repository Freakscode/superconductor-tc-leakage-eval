"""H11 - dispersion of Tc among rows of Dataset A that share an IDENTICAL 81-feature vector.

The manuscript called the mean |Tc - group mean| over these rows (3.49 K) a "noise floor" of the MAE.
It is not a lower bound: the constant minimising the absolute error is the group MEDIAN, and neither
statistic is a population irreducible error (all labels of the group are used; same representation does
not mean same experiment). This script measures, on the same rows (feature-vector groups of size >= 2,
identified exactly as in code/v2/run_twins_noise.py via hash_pandas_object of the 81 columns):
  * mean |Tc - group mean|      (v2 value 3.4913 K)        * mean |Tc - group median|   (expected 3.2229 K)
  * medians of both deviations, and the same per group size class,
  * leave-one-replicate-out baseline: prediction = MEDIAN (and mean) of the OTHER rows of the group,
    which does not use the row's own label (an honest featureless predictor for a replicated vector),
  * MAE of the Table-1 models (XGBoost / RandomForest, random KFold rs=42 and GroupKFold) restricted to
    these rows, from results_v3/oof_v3.csv.gz, and on the complementary rows.
Output: results_v3/replicate_dispersion.csv (+ .md with the replacement wording).

    python code/v3/replicate_dispersion.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h07_common import DATA, RES_V2, RES_V3, VERS, load_dataset

PROC = (f"code/v3/replicate_dispersion.py | {VERS} | Dataset A data/train.csv; replicate group = identical 81-feature vector "
        f"(pandas.util.hash_pandas_object of the feature columns, as code/v2/run_twins_noise.py), groups with >= 2 rows | "
        f"model errors from results_v3/oof_v3.csv.gz (code/v3/oof_v3.py, seed 42)")


def main():
    X, y, g, _, meta = load_dataset("A")
    tr = pd.read_csv(DATA / "train.csv"); feat = [c for c in tr.columns if c != "critical_temp"]
    key = pd.util.hash_pandas_object(tr[feat], index=False).values
    d = pd.DataFrame({"k": key, "tc": y, "fam": g})
    size = d.groupby("k").tc.transform("size")
    multi = d[size >= 2].copy(); idx = multi.index.values
    assert len(multi) == 8511, len(multi)
    grp = multi.groupby("k").tc
    multi["gmean"] = grp.transform("mean"); multi["gmed"] = grp.transform("median"); multi["n"] = grp.transform("size")
    multi["gsum"] = grp.transform("sum")
    multi["loo_mean"] = (multi.gsum - multi.tc) / (multi.n - 1)
    # leave-one-out median of the OTHER replicates
    loo_med = np.empty(len(multi))
    for kk, s in grp:
        v = s.values; pos = multi.index.get_indexer(s.index)
        for j in range(len(v)):
            loo_med[pos[j]] = np.median(np.delete(v, j))
    multi["loo_med"] = loo_med
    dev_mean, dev_med = (multi.tc - multi.gmean).abs(), (multi.tc - multi.gmed).abs()
    dev_loo_mean, dev_loo_med = (multi.tc - multi.loo_mean).abs(), (multi.tc - multi.loo_med).abs()
    n_groups = multi.k.nunique()
    rows = [
        dict(metrica="n_filas_con_vector_repetido", valor=len(multi), n=len(multi), nota="grupos de >= 2 filas con las 81 features idénticas"),
        dict(metrica="n_grupos", valor=n_groups, n=len(multi), nota=""),
        dict(metrica="n_grupos_size2", valor=int((multi.groupby('k').size() == 2).sum()), n=len(multi), nota="en grupos de 2 la mediana y la media coinciden"),
        dict(metrica="pct_filas_en_grupos_size2", valor=100 * float((multi.n == 2).mean()), n=len(multi), nota=""),
        dict(metrica="desv_abs_media_a_MEDIA_del_grupo_K", valor=dev_mean.mean(), n=len(multi), nota="valor v2 'piso' 3.4913; usa la propia etiqueta"),
        dict(metrica="desv_abs_media_a_MEDIANA_del_grupo_K", valor=dev_med.mean(), n=len(multi), nota="constante que minimiza el error absoluto; usa la propia etiqueta"),
        dict(metrica="mediana_desv_abs_a_media_K", valor=dev_mean.median(), n=len(multi), nota="v2 1.1633"),
        dict(metrica="mediana_desv_abs_a_mediana_K", valor=dev_med.median(), n=len(multi), nota=""),
        dict(metrica="pct_filas_desv_a_mediana_igual_0", valor=100 * float((dev_med == 0).mean()), n=len(multi), nota=""),
        dict(metrica="baseline_replica_excluida_MEDIANA_otras_MAE_K", valor=dev_loo_med.mean(), n=len(multi), nota="prediccion = mediana de las OTRAS replicas del grupo (no usa la etiqueta propia)"),
        dict(metrica="baseline_replica_excluida_MEDIA_otras_MAE_K", valor=dev_loo_mean.mean(), n=len(multi), nota="prediccion = media de las OTRAS replicas"),
        dict(metrica="baseline_replica_excluida_MEDIANA_otras_mediana_K", valor=dev_loo_med.median(), n=len(multi), nota=""),
    ]
    # by group size class
    cls = pd.cut(multi.n, [1, 2, 3, 5, 10, 10 ** 6], labels=["2", "3", "4-5", "6-10", ">10"])
    for c, s in multi.groupby(cls, observed=True):
        rows.append(dict(metrica=f"desv_abs_a_mediana_K__size_{c}", valor=(s.tc - s.gmed).abs().mean(), n=len(s), nota="por clase de tamaño de grupo"))
        rows.append(dict(metrica=f"replica_excluida_mediana_MAE_K__size_{c}", valor=(s.tc - s.loo_med).abs().mean(), n=len(s), nota="por clase de tamaño de grupo"))
    # Table-1 models restricted to these rows
    oof = pd.read_csv(RES_V3 / "oof_v3.csv.gz"); oof = oof[oof.dataset == "A"]
    mask = np.zeros(len(y), bool); mask[idx] = True
    for (model, proto), o in oof.groupby(["model", "protocol"]):
        o = o.sort_values("row_index"); e = np.abs(o.y_true.values - o.y_pred.values)
        rows.append(dict(metrica=f"MAE_modelo_{model}_{proto}_filas_con_vector_repetido_K", valor=e[mask].mean(), n=int(mask.sum()), nota="v2: XGBoost random 4.8311"))
        rows.append(dict(metrica=f"MAE_modelo_{model}_{proto}_filas_sin_vector_repetido_K", valor=e[~mask].mean(), n=int((~mask).sum()), nota="v2: XGBoost random 5.4967"))
    # exact twin present in the train fold (random) / never (grouped), from v2 twins file for reference
    tw = pd.read_csv(RES_V2 / "twins_noise_A.csv")
    for _, r in tw[tw.metrica == "frac_test_con_gemelo_exacto_en_train"].iterrows():
        rows.append(dict(metrica=f"v2_frac_test_con_gemelo_exacto_en_train_{r.split}", valor=r.valor, n=int(r.n), nota="results_v2/twins_noise_A.csv"))
    D = pd.DataFrame(rows); D["dataset"] = "A"; D["procedencia"] = PROC
    RES_V3.mkdir(parents=True, exist_ok=True); D.to_csv(RES_V3 / "replicate_dispersion.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
    print(D[["metrica", "valor", "n"]].to_string())
    v = D.set_index("metrica").valor
    L = ["# H11 - Dispersión de Tc entre filas con vector de features idéntico (Dataset A)", "", f"Procedencia: `{PROC}`", "",
         f"Sobre las {int(v['n_filas_con_vector_repetido'])} filas ({int(v['n_grupos'])} grupos) de A cuyas 81 features coinciden exactamente con al menos otra fila:", "",
         f"- desviación absoluta media a la MEDIA del grupo: {v['desv_abs_media_a_MEDIA_del_grupo_K']:.4f} K (el '3.49 K' de v2);",
         f"- desviación absoluta media a la MEDIANA del grupo: {v['desv_abs_media_a_MEDIANA_del_grupo_K']:.4f} K (la constante que minimiza el error absoluto);",
         f"- ambas usan la propia etiqueta de la fila. Baseline de réplica excluida (mediana de las OTRAS réplicas): MAE = {v['baseline_replica_excluida_MEDIANA_otras_MAE_K']:.4f} K; con la media de las otras: {v['baseline_replica_excluida_MEDIA_otras_MAE_K']:.4f} K.",
         f"- MAE del modelo XGBoost (KFold aleatorio, semilla 42) sobre esas filas: {v['MAE_modelo_XGBoost_random_filas_con_vector_repetido_K']:.4f} K; sobre el resto: {v['MAE_modelo_XGBoost_random_filas_sin_vector_repetido_K']:.4f} K; "
         f"GroupKFold sobre esas filas: {v['MAE_modelo_XGBoost_grouped_filas_con_vector_repetido_K']:.4f} K.", "",
         "## Redacción propuesta (sustituye 'piso de ruido de 3.49 K')", "",
         f"> Entre las {int(v['n_filas_con_vector_repetido'])} filas de A que comparten exactamente el mismo vector de 81 descriptores ({int(v['n_grupos'])} grupos), la desviación absoluta media de Tc respecto a la mediana de su grupo es {v['desv_abs_media_a_MEDIANA_del_grupo_K']:.2f} K "
         f"(respecto a la media, {v['desv_abs_media_a_MEDIA_del_grupo_K']:.2f} K). Llamamos a esta cantidad *dispersión observada no explicada por la representación*: es la variación de Tc que ningún modelo determinista de estos descriptores puede reproducir en esas filas, "
         f"pero no es una cota inferior del MAE poblacional ni un error de medición puro, porque se calcula con todas las etiquetas del grupo y puede reflejar estructura, presión, preparación o condiciones no registradas. "
         f"Un predictor de réplica excluida (mediana de las demás réplicas del grupo) alcanza un MAE de {v['baseline_replica_excluida_MEDIANA_otras_MAE_K']:.2f} K en esas filas, frente a {v['MAE_modelo_XGBoost_random_filas_con_vector_repetido_K']:.2f} K del XGBoost con partición aleatoria."]
    (RES_V3 / "replicate_dispersion.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-3:])); print(f"wrote {RES_V3 / 'replicate_dispersion.csv'} and .md")


if __name__ == "__main__":
    main()
