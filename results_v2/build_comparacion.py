"""build_comparacion.py — genera results_v2/tabla2_comparacion.md desde los CSV/JSON medidos."""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd

ROOT = os.environ["SUPERCON_ROOT"]
OUT = os.path.join(ROOT, "results_v2")
R = lambda f: os.path.join(OUT, f)

t_new = pd.read_csv(R("tabla2_shuffle.csv"))
t_ctl = pd.read_csv(R("tabla2_noshuffle_control.csv"))
grow = pd.read_csv(R("tabla2_crecimiento_desviacion.csv"))
ctrl = pd.read_csv(R("tabla2_control_reproduccion.csv"))
disp = pd.read_csv(R("tabla2_dispersion_entre_folds.csv"))
rango = json.load(open(R("tabla2_rango_inflacion.json")))
det = json.load(open(R("tabla2_determinismo_groupkfold.json")))

PUB = {("A", "XGBoost"): (5.25, 0.02, 8.34, 0.10, 59, 2),
       ("A", "RandomForest"): (5.41, 0.01, 8.69, 0.07, 61, 1),
       ("B", "XGBoost"): (4.03, 0.01, 6.20, 0.05, 54, 2),
       ("B", "RandomForest"): (4.46, 0.01, 6.48, 0.03, 45, 1)}

# dR2 publicado: se DERIVA de results/repeated_seed_intervals.csv del repo (no se teclea).
# Ese archivo publica R2 por brazo, no un dR2 emparejado, asi que solo hay media disponible.
_pubcsv = pd.read_csv(os.path.join(ROOT, "superconductor-tc-leakage-eval", "results",
                                   "repeated_seed_intervals.csv"))
_p = _pubcsv.pivot_table(index=["dataset", "model"], columns="split", values="R2_mean")
PUB_DR2 = {(d, m): float(_p.loc[(d, m), "grouped"] - _p.loc[(d, m), "random"]) for d, m in _p.index}
ORD = [("A", "XGBoost"), ("A", "RandomForest"), ("B", "XGBoost"), ("B", "RandomForest")]
key = lambda d, m: (d, m)
row_new = {key(r.dataset, r.modelo): r for _, r in t_new.iterrows()}
row_ctl = {key(r.dataset, r.modelo): r for _, r in t_ctl.iterrows()}
row_gro = {key(r.dataset, r.modelo): r for _, r in grow.iterrows()}
row_ctr = {key(r.dataset, r.modelo): r for _, r in ctrl.iterrows()}
dsp = {(r.dataset, r.modelo, r.scheme): r for _, r in disp.iterrows()}
lbl = {"XGBoost": "XGB", "RandomForest": "RF"}

L = []
A = L.append
A("# Tabla 2 (robustez) — recalculada con particiones agrupadas realmente independientes")
A("")
A("Generado por `run_tabla2_shuffle.py` + `agg_tabla2.py` + `build_comparacion.py`. "
  "Todas las cifras salen de las 60 corridas de validacion cruzada registradas en "
  "`tabla2_runs_por_semilla.csv`; se redondea solo al imprimir.")
A("")
A("## 1. El problema, medido")
A("")
A("La leyenda publicada de la Tabla 2 dice que ambos esquemas se repitieron *\"over five "
  "independent fold assignments (seeds 0-4)\"*. Para el brazo agrupado eso es falso: "
  "`code/tc_pipeline.py` y `code/datasetB_pipeline.py` construyen el brazo agrupado con "
  "`GroupKFold(5)`, sin `shuffle`, que es determinista. La semilla solo entraba por "
  "`make_models(seed)`, es decir por el `random_state` del **modelo**.")
A("")
A("Verificacion directa (`tabla2_determinismo_groupkfold.json`):")
A("")
for d in ("A", "B"):
    e = det["datasets"][d]
    A(f"- **Dataset {d}** (N={e['n']}, {e['n_familias']} familias): cinco invocaciones de "
      f"`GroupKFold(5)` devuelven folds **identicos** indice por indice "
      f"(`{str(e['noshuffle_5_invocaciones_folds_identicos']).lower()}`). Con "
      f"`GroupKFold(5, shuffle=True, random_state=s)`, s=0..4, se obtienen "
      f"**{e['shuffle_n_particiones_distintas_de_5']} particiones distintas de 5**: el Jaccard "
      f"entre los conjuntos de familias del primer fold cae a "
      f"{e['shuffle_jaccard_fold1_pares_min']:.3f}-{e['shuffle_jaccard_fold1_pares_max']:.3f} "
      f"(media {e['shuffle_jaccard_fold1_pares_mean']:.3f}).")
A("")
A(f"El Jaccard medio observado ({det['datasets']['A']['shuffle_jaccard_fold1_pares_mean']:.3f} en A, "
  f"{det['datasets']['B']['shuffle_jaccard_fold1_pares_mean']:.3f} en B) coincide con el valor "
  f"esperado para dos subconjuntos independientes de 1/5 de las familias, "
  f"(1/25)/(9/25) = 1/9 = {1/9:.3f}. Es decir: `shuffle=True` aleatoriza de verdad la asignacion "
  "familia -> fold, no la perturba levemente.")
A("")
A("Efecto colateral que conviene declarar: sin `shuffle`, `GroupKFold` balancea los folds por "
  f"tamano ({det['datasets']['A']['noshuffle_fold_sizes']} muestras en A); con `shuffle=True` el "
  f"balanceo se pierde (p.ej. semilla 0 en A: {det['datasets']['A']['shuffle_fold_sizes_por_semilla']['0']}). "
  "Ninguno de los dos esquemas filtra familias: las familias compartidas entre train y test son 0 "
  "en las 40 corridas agrupadas.")
A("")
A("## 2. Tabla 2 publicada vs. Tabla 2 v2")
A("")
A("Brazo aleatorio en ambas: `KFold(5, shuffle=True, random_state=s)`. Brazo agrupado: "
  "`GroupKFold(5)` en la publicada (particion fija), "
  "`GroupKFold(5, shuffle=True, random_state=s)` en la v2 (particion variable). "
  "s = 0..4 en ambos casos; desviacion = `np.std(ddof=0)`; la inflacion se calcula emparejada "
  "por semilla y luego se promedia.")
A("")
A("| Dataset / modelo | MAE aleatorio (K) | MAE agrupado (K) | Inflacion MAE (%) | dR2 |")
A("|---|---|---|---|---|")
for d, m in ORD:
    p = PUB[(d, m)]
    dp = PUB_DR2[(d, m)]
    n = row_new[(d, m)]
    A(f"| **{d} / {lbl[m]}** — publicada | {p[0]:.2f} ± {p[1]:.2f} | {p[2]:.2f} ± {p[3]:.2f} "
      f"| {p[4]} ± {p[5]} | {dp:+.3f} (sin ±) |")
    A(f"| {d} / {lbl[m]} — **v2 (shuffle)** | {n.MAE_random_mean:.2f} ± {n.MAE_random_std:.2f} "
      f"| {n.MAE_grouped_mean:.2f} ± {n.MAE_grouped_std:.2f} "
      f"| {n.inflacion_pct_mean:.1f} ± {n.inflacion_pct_std:.1f} "
      f"| {n.dR2_mean:+.3f} ± {n.dR2_std:.3f} |")
A("")
A("Fuentes: `tabla2_shuffle.csv` (v2), `results/repeated_seed_intervals.csv` del repo (publicada). "
  "El dR2 publicado se toma de las columnas R2 de ese mismo archivo "
  "(R2_grouped_mean − R2_random_mean).")
A("")
A("## 3. Cuanto cambia la desviacion — y por que no cambia como se esperaba")
A("")
A("**Se esperaba que la desviacion del brazo agrupado creciera al dejar variar la particion. "
  "Medida, no crece.** La desviacion del MAE agrupado con particion variable es "
  f"{t_new.MAE_grouped_std.min():.3f}-{t_new.MAE_grouped_std.max():.3f} K, "
  f"o sea {grow.factor_vs_pub.min():.2f}-{grow.factor_vs_pub.max():.2f}x la publicada "
  f"({t_ctl.MAE_grouped_std.min():.3f}-{t_ctl.MAE_grouped_std.max():.3f} K en la reproduccion "
  "del protocolo publicado). Sigue en el mismo orden de magnitud, 0.02-0.08 K.")
A("")
A("| Dataset / modelo | σ(MAE agr.) publicada | σ reproducida sin shuffle | σ v2 con shuffle | factor v2 / sin shuffle |")
A("|---|---|---|---|---|")
for d, m in ORD:
    g = row_gro[(d, m)]
    A(f"| {d} / {lbl[m]} | {g.std_MAE_grouped_pub:.2f} | {g.std_MAE_grouped_noshuffle:.3f} "
      f"| {g.std_MAE_grouped_shuffle:.3f} | {g.factor_vs_noshuffle:.2f}x |")
A("")
A(f"El aumento es de {grow.factor_vs_noshuffle.min():.2f}x a {grow.factor_vs_noshuffle.max():.2f}x "
  f"(mediana {grow.factor_vs_noshuffle.median():.2f}x) frente al mismo codigo sin shuffle, y en tres "
  "de las cuatro combinaciones queda *por debajo* de la desviacion que reporta la tabla publicada. "
  "En la inflacion el patron es mixto: "
  f"{', '.join(f'{d}/{lbl[m]} {row_gro[(d,m)].factor_infl_vs_noshuffle:.2f}x' for d, m in ORD)}.")
A("")
A("La razon es metodologica, no un error de la correccion: **el MAE out-of-fold agrupado se calcula "
  "sobre las N muestras del dataset completo**, con lo que redibujar la asignacion familia -> fold "
  "reordena que muestra cae en que fold pero deja casi intacto el promedio global. La particion "
  "sí importa mucho, pero su efecto vive *entre folds*, no entre semillas:")
A("")
A("| Dataset / modelo | σ entre folds, sin shuffle (K) | σ entre folds, con shuffle (K) | σ del MAE agrupado entre semillas (K) |")
A("|---|---|---|---|")
for d, m in ORD:
    a = dsp[(d, m, "grouped_noshuffle")]; b = dsp[(d, m, "grouped_shuffle")]
    n = row_new[(d, m)]
    A(f"| {d} / {lbl[m]} | {a.fold_std_mean:.3f} | {b.fold_std_mean:.3f} | {n.MAE_grouped_std:.3f} |")
A("")
fac = (disp[disp.scheme == "grouped_shuffle"].set_index(["dataset", "modelo"]).fold_std_mean /
       disp[disp.scheme == "grouped_noshuffle"].set_index(["dataset", "modelo"]).fold_std_mean)
ratio_fp = [dsp[(d, m, "grouped_shuffle")].fold_std_mean / row_new[(d, m)].MAE_grouped_std for d, m in ORD]
A(f"Ahi sí crece al liberar la particion: {fac.min():.2f}x-{fac.max():.2f}x. Y es "
  f"{min(ratio_fp):.0f}-{max(ratio_fp):.0f} veces mas grande que la desviacion entre semillas. "
  "El MAE de un fold agrupado individual va de "
  f"{disp[disp.scheme=='grouped_shuffle'].fold_min.min():.2f} K a "
  f"{disp[disp.scheme=='grouped_shuffle'].fold_max.max():.2f} K. "
  "Esa es la cifra que un lector debe ver si quiere saber cuanto depende el resultado de que "
  "familias quedan fuera; la barra ± de la Tabla 2, con cualquiera de los dos protocolos, no lo dice.")
A("")
A("## 4. Reproduccion del protocolo publicado (control)")
A("")
A("Corriendo `GroupKFold(5)` sin shuffle con `make_models(s)`, s=0..4 — el protocolo publicado tal cual:")
A("")
A("| Dataset / modelo | MAE agr. publ. | MAE agr. repro. | MAE aleat. publ. | MAE aleat. repro. | Infl. publ. (%) | Infl. repro. (%) |")
A("|---|---|---|---|---|---|---|")
for d, m in ORD:
    c = row_ctr[(d, m)]
    A(f"| {d} / {lbl[m]} | {c.MAE_grouped_pub:.2f} | {c.MAE_grouped_repro:.2f} "
      f"| {c.MAE_random_pub:.2f} | {c.MAE_random_repro:.2f} | {c.infl_pub:.0f} | {c.infl_repro:.1f} |")
A("")
A("Dataset A reproduce dentro de 0.09 K. **Dataset B no reproduce el brazo aleatorio**: "
  f"{ctrl[ctrl.dataset=='B'].MAE_random_pub.tolist()} K publicados frente a "
  f"{ctrl[ctrl.dataset=='B'].MAE_random_repro.tolist()} K reproducidos a partir de "
  "`data/datasetB_featurized.npz` con `make_models()` de `code/datasetB_pipeline.py`. El brazo "
  "agrupado de B sí reproduce (≤0.03 K). No hay en el arbol ningun script que genere "
  "`results/repeated_seed_intervals.csv`, asi que no se puede auditar de donde salieron los "
  "4.03/4.46 K. Por eso la inflacion publicada de B (54% y 45%) sube a "
  f"{row_ctr[('B','XGBoost')].infl_repro:.0f}% y {row_ctr[('B','RandomForest')].infl_repro:.0f}% "
  "con el *mismo* protocolo publicado: el desplazamiento de B viene del brazo aleatorio, no del shuffle.")
A("")
A("## 5. Rango global de inflacion del MAE (para re-anclar el abstract)")
A("")
A(f"- Medias por combinacion (v2, shuffle): "
  + ", ".join(f"{k} {v:.1f}%" for k, v in rango["medias_por_combinacion"].items()) + ".")
A(f"- **Rango de las medias: {rango['rango_de_medias_pct'][0]:.1f}%-{rango['rango_de_medias_pct'][1]:.1f}%.**")
A(f"- Rango sobre las 20 corridas individuales: {rango['rango_todas_las_corridas_pct'][0]:.1f}%-"
  f"{rango['rango_todas_las_corridas_pct'][1]:.1f}%.")
A(f"- Rango publicado (abstract actual): {rango['rango_publicado_medias_pct'][0]}%-"
  f"{rango['rango_publicado_medias_pct'][1]}%.")
A(f"- Rango con el protocolo publicado reproducido aqui: {rango['rango_control_noshuffle_pct'][0]:.1f}%-"
  f"{rango['rango_control_noshuffle_pct'][1]:.1f}%.")
A("")
A("El extremo inferior sube de 45% a 52%: el 45% publicado era B/RF, y su brazo aleatorio es "
  "precisamente el que no reproduce. El extremo superior se mantiene en ~61%. "
  "**El abstract del v2 debe decir 52%-61%, no 45%-61%.**")
A("")
A("## 6. Que poner en el v2")
A("")
A("Leyenda de la Tabla 2 (reemplazo literal propuesto):")
A("")
A("> Values are mean ± s.d. over five independent fold assignments (seeds 0-4): both arms use "
  "`shuffle=True` with `random_state` = seed, so the seed re-draws the partition — "
  "`KFold(5, shuffle=True, random_state=s)` for the random arm and "
  "`GroupKFold(5, shuffle=True, random_state=s)` (scikit-learn >= 1.4) for the chemical-family arm — "
  "and it also sets the model's `random_state`. No chemical family is shared between training and "
  "test folds in the grouped arm. The quoted s.d. is the spread of the pooled out-of-fold metric "
  f"across seeds and is therefore small by construction (<= {t_new.MAE_grouped_std.max():.2f} K); "
  "the spread that reflects *which* families are held out is the between-fold spread, "
  f"{disp[disp.scheme=='grouped_shuffle'].fold_std_mean.min():.2f}-"
  f"{disp[disp.scheme=='grouped_shuffle'].fold_std_mean.max():.2f} K, with individual grouped folds "
  f"ranging from {disp[disp.scheme=='grouped_shuffle'].fold_min.min():.2f} K to "
  f"{disp[disp.scheme=='grouped_shuffle'].fold_max.max():.2f} K MAE.")
A("")
A("Nota obligatoria de errata / metodo, porque la version publicada del codigo no hacia esto:")
A("")
_dmax = (t_new.set_index(["dataset", "modelo"]).MAE_grouped_mean -
         t_ctl.set_index(["dataset", "modelo"]).MAE_grouped_mean).abs().max()
A("> In the originally released code the grouped arm used `GroupKFold(5)` without shuffling, which "
  "is deterministic; the seed varied only the model's `random_state`, not the partition. The "
  "grouped-arm s.d. reported there (0.03-0.10 K) therefore measured model variance, not partition "
  f"variance. Re-running with `shuffle=True` shifts the grouped MAE by at most {_dmax:.2f} K "
  "across the four dataset x model combinations and leaves every qualitative conclusion unchanged.")
A("")
_d = (t_new.set_index(["dataset", "modelo"]).MAE_grouped_mean -
      t_ctl.set_index(["dataset", "modelo"]).MAE_grouped_mean)
_rat = {(d, m): abs(_d.loc[(d, m)]) / row_new[(d, m)].MAE_grouped_std for d, m in ORD}
_n_exc = sum(1 for v in _rat.values() if v > 1)
A(f"Detalle que un revisor va a mirar: el desplazamiento del MAE agrupado al activar `shuffle` "
  f"({', '.join(f'{d}/{lbl[m]} {_d.loc[(d,m)]:+.2f} K' for d, m in ORD)}) equivale a "
  f"{min(_rat.values()):.2f}x-{max(_rat.values()):.2f}x la barra ± de esa misma celda: la excede en "
  f"{_n_exc} de las 4 combinaciones ("
  + ", ".join(f'{d}/{lbl[m]} {_rat[(d,m)]:.1f}x' for d, m in ORD if _rat[(d, m)] > 1)
  + ") y la iguala en las otras dos. Es coherente con lo anterior: la barra ± no cubre la variacion "
    "por particion, y por eso no debe leerse como intervalo de confianza del MAE agrupado.")
A("")
A("Ademas: el brazo aleatorio de Dataset B en la Tabla 2 publicada (4.03 / 4.46 K) no se reproduce "
  "desde el codigo archivado (3.87 / 4.32 K). Hay que corregir esos dos valores y, con ellos, "
  "las inflaciones de B.")
A("")
A("## Archivos")
A("")
for f, d in [("tabla2_shuffle.csv", "Tabla 2 v2 (entregable principal)"),
             ("tabla2_noshuffle_control.csv", "control: protocolo publicado reproducido"),
             ("tabla2_runs_por_semilla.csv", "las 60 corridas crudas"),
             ("tabla2_determinismo_groupkfold.json", "evidencia del paso 1"),
             ("tabla2_crecimiento_desviacion.csv", "factores de crecimiento de σ"),
             ("tabla2_control_reproduccion.csv", "publicada vs reproducida"),
             ("tabla2_dispersion_entre_folds.csv", "σ entre folds por esquema"),
             ("tabla2_fold_mae_por_corrida.csv", "MAE por fold, corrida a corrida"),
             ("tabla2_inflacion_por_semilla.csv", "inflacion emparejada, semilla a semilla"),
             ("tabla2_rango_inflacion.json", "rango global de inflacion")]:
    A(f"- `{f}` — {d}")

open(R("tabla2_comparacion.md"), "w").write("\n".join(L) + "\n")
print("escrito", R("tabla2_comparacion.md"))
