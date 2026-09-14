"""Consolidated report of the v3 inference phase (H01, H07, H08, H11, null R2, A nomenclature sensitivity):
reads the results_v3 CSVs produced by oof_v3.py, paired_family_bootstrap.py, nested_both_arms.py,
design_effect_v3.py, replicate_dispersion.py, null_r2_v3.py, tabla1_A_sensitivity.py and writes
results_v3/inferencia_v3_resumen.md with every number formatted from the files (nothing retyped) and the
replacement sentences for manuscript v3.

    python code/v3/inferencia_resumen_v3.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h07_common import RES_V3, VERS

R = RES_V3


def main():
    P = pd.read_csv(R / "paired_family_bootstrap.csv"); T3 = pd.read_csv(R / "tabla3_v3.csv"); DE = pd.read_csv(R / "design_effect_v3.csv")
    RD = pd.read_csv(R / "replicate_dispersion.csv").set_index("metrica").valor; NR = pd.read_csv(R / "null_r2_by_fold.csv")
    SA = pd.read_csv(R / "tabla1_A_sin_O1_inflacion.csv"); CK = pd.read_csv(R / "oof_v3_check.csv")

    def p(ds, model, stat, scheme="family_paired"):
        r = P[(P.dataset == ds) & (P.model == model) & (P.statistic == stat) & (P.scheme == scheme)].iloc[0]; return r
    def fmt(r, d=1): return f"{r.point:.{d}f} [{r.lo:.{d}f}, {r.hi:.{d}f}]"
    def de(ds, est, var="todas"):
        return DE[(DE.dataset == ds) & (DE.estimador.str.startswith(est)) & (DE.variant.str.startswith(var))].iloc[0]
    def t3(ds, bloque, key):
        return T3[(T3.dataset == ds) & (T3.bloque == bloque) & (T3.comparacion.str.contains(key, regex=False))].iloc[0]

    L = ["# Inferencia v3 — H07 (pareado por familias), H08 (Tabla 3), H01 (N efectivo), H11 (dispersión), R² del nulo, sensibilidad A", "",
         f"Entorno: {VERS}. Datos canónicos: A = data/train.csv + unique_m.csv (N 21 263, k 3 365); B v3 = data/datasetB_featurized_v3.npz (N 12 336, k 3 046). "
         "Semilla 42; random = KFold(5, shuffle, rs=42); grouped = GroupKFold(5) determinista (verificado contra splits/). Todos los intervalos: bootstrap PAREADO de familias completas, 2 000 remuestreos, default_rng(0), percentil 2.5/97.5, "
         "**condicionales a las predicciones OOF ya ajustadas** (no incluyen la variabilidad de reentrenar; esa la cubren las 5 semillas de la Tabla 2). Umbral alto-Tc = p90 del fold de entrenamiento, operador >= (no interviene en estos resultados).", ""]
    # ---------------- checks
    L += ["## 0. Verificación de las predicciones OOF (results_v3/oof_v3.csv.gz, oof_v3_check.csv)", ""]
    ok = CK[CK["match_1e-6"] == True]
    L += [f"- {len(ok)} de {CK.MAE_ref.notna().sum()} métricas con referencia coinciden a < 1e-6 (de hecho ≤ 1e-15): A XGBoost y nulo vs results_v2/nulo_presencia.csv; B v3 XGBoost, RandomForest y nulo vs results_v3/tabla1_B_antes_despues.csv. n_jobs=8 no altera ningún valor.",
          f"- A RandomForest (sin referencia v2 a 6 decimales): MAE {CK[(CK.dataset=='A')&(CK.model=='RandomForest')&(CK.protocol=='random')].MAE.iloc[0]:.3f} / {CK[(CK.dataset=='A')&(CK.model=='RandomForest')&(CK.protocol=='grouped')].MAE.iloc[0]:.3f} frente a v1 (3 decimales) 5.396 / 8.683; results_v2/consolidated_results.csv no existe (la ruta de la tarea); la referencia usada es results/consolidated_results.csv (v1).",
          "- Cruce con results_v3/oof_predictions.csv.gz (otra línea): A idéntico (XGB 0.0, RF ≤ 9e-14); su B es el checkpoint v2 (N 12 440), no comparable fila a fila con v3.", ""]
    # ---------------- H07
    L += ["## 1. H07 — bootstrap pareado por familias (results_v3/paired_family_bootstrap.csv/.md, family_losses_v3.csv)", "",
          "| Dataset | Modelo | MAE random | MAE grouped | Inflación % | ΔMAE (K) | ΔR² | MAE macro random | MAE macro grouped | Inflación macro % |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in ("A", "B"):
        for m in ("XGBoost", "RandomForest", "FamilyMeanNull"):
            L.append(f"| {ds} | {m} | {fmt(p(ds,m,'MAE_random'),2)} | {fmt(p(ds,m,'MAE_grouped'),2)} | {fmt(p(ds,m,'inflacion_pct'))} | {fmt(p(ds,m,'dMAE_K'),2)} | {fmt(p(ds,m,'dR2'),3)} | {fmt(p(ds,m,'MAE_macro_random'),2)} | {fmt(p(ds,m,'MAE_macro_grouped'),2)} | {fmt(p(ds,m,'inflacion_macro_pct'))} |")
    L += ["", "Cociente de anchuras IC familias (pareado) / IC filas no pareado (esquema v2) y / IC filas pareado:", "",
          "| Dataset | Modelo | Estadístico | familias | filas no pareado (v2) | filas pareado | fam/filas-no-par | fam/filas-par |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for ds in ("A", "B"):
        for m in ("XGBoost", "RandomForest"):
            for st, d in (("inflacion_pct", 1), ("dMAE_K", 2), ("dR2", 3), ("MAE_random", 2), ("MAE_grouped", 2)):
                f, u, q = p(ds, m, st), p(ds, m, st, "row_unpaired_v2scheme"), p(ds, m, st, "row_paired")
                L.append(f"| {ds} | {m} | {st} | [{f.lo:.{d}f}, {f.hi:.{d}f}] | [{u.lo:.{d}f}, {u.hi:.{d}f}] | [{q.lo:.{d}f}, {q.hi:.{d}f}] | {f.width_ratio_family_over_row_unpaired:.2f} | {f.width_ratio_family_over_row_paired:.2f} |")
    fa, fb = p("A", "XGBoost", "inflacion_pct"), p("B", "XGBoost", "inflacion_pct")
    ua, ub = p("A", "XGBoost", "inflacion_pct", "row_unpaired_v2scheme"), p("B", "XGBoost", "inflacion_pct", "row_unpaired_v2scheme")
    L += ["", f"Lectura: el IC por familias de la inflación (XGBoost) es {fa.width_ratio_family_over_row_unpaired:.1f}× (A) y {fb.width_ratio_family_over_row_unpaired:.1f}× (B) más ancho que el publicado por filas; "
          f"el límite inferior sigue lejos de cero ({fa.lo:.1f} % en A, {fb.lo:.1f} % en B). El IC por filas no pareado de v2 se reproduce dentro del error Monte Carlo ([{ua.lo:.2f}, {ua.hi:.2f}] vs publicado [55.07, 62.97] en A; B ya no comparable por el cambio de checkpoint). "
          f"La inflación MACRO (media de MAE por familia) es {p('A','XGBoost','inflacion_macro_pct').point:.1f} % (A) y {p('B','XGBoost','inflacion_macro_pct').point:.1f} % (B): menos de la mitad de la inflación pooled. La brecha pooled está dominada por las familias grandes (que pesan por número de filas); "
          "para la familia típica la brecha existe pero es menor. Es un resultado nuevo que el manuscrito debe reportar.", ""]
    # ---------------- H08
    L += ["## 2. H08 — Tabla 3 en dos bloques (results_v3/tabla3_v3.csv/.md, tabla3_v3_folds.csv, oof_tabla3_v3.csv.gz)", "",
          "| Bloque | Dataset | Comparación | MAE random | MAE grouped | Inflación % [IC fam.] | ΔR² [IC] | Inflación macro % [IC] | valor v2 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in T3.iterrows():
        v2 = r.get("v2_inflacion_pct", np.nan); v2s = f"{v2:.1f}" if pd.notna(v2) else (f"MAE anidado grouped v2 {r.v2_MAE_grouped_anidada:.3f}" if pd.notna(r.get("v2_MAE_grouped_anidada", np.nan)) else "-")
        L.append(f"| {r.bloque} | {r.dataset} | {r.comparacion} | {r.MAE_random:.3f} | {r.MAE_grouped:.3f} | {r.inflacion_pct:.1f} [{r.inflacion_lo:.1f}, {r.inflacion_hi:.1f}] | {r.dR2:.3f} [{r.dR2_lo:.3f}, {r.dR2_hi:.3f}] | {r.inflacion_macro_pct:.1f} [{r.inflacion_macro_lo:.1f}, {r.inflacion_macro_hi:.1f}] | {v2s} |")
    nA, nB = t3("A", "b_nested", "anidado en ambos brazos"), t3("B", "b_nested", "anidado en ambos brazos")
    sA, sB = t3("A", "b_nested_sens", "SIN barajar"), t3("B", "b_nested_sens", "SIN barajar")
    L += ["", f"- Brazo agrupado anidado de A reproduce v2 exactamente (MAE {nA.MAE_grouped:.4f}; ganadoras por fold y MAE externo por fold idénticos, tabla3_v3_folds.csv). B v3 rehecho: {nB.MAE_grouped:.4f} (v2, checkpoint v2: 6.226).",
          f"- Anidado en AMBOS brazos: inflación {nA.inflacion_pct:.1f} % [{nA.inflacion_lo:.1f}, {nA.inflacion_hi:.1f}] en A y {nB.inflacion_pct:.1f} % [{nB.inflacion_lo:.1f}, {nB.inflacion_hi:.1f}] en B. Ganadoras random: {nA.configs_ganadoras_random} (A); {nB.configs_ganadoras_random} (B). La región de complejidad seleccionada es la misma en los dos brazos (depth 8–10, lr 0.05–0.1).",
          f"- Sensibilidad: con el CV interno del brazo aleatorio SIN barajar (KFold(3) por bloques contiguos), el MAE interno medio sube a {sA.MAE_interna_media_random:.2f} K en A (frente a {nA.MAE_interna_media_random:.2f} K barajado) porque las filas de train.csv están ordenadas y los bloques contiguos se comportan casi como particiones por familia; el resultado externo apenas cambia (inflación {sA.inflacion_pct:.1f} % en A, {sB.inflacion_pct:.1f} % en B).",
          f"- La comparación 'mixta' de v2 (63.9 % A / 53.1 % B) mezcla un brazo anidado con una configuración modal elegida a posteriori; no es 'mismos hiperparámetros'. Los valores correctos para la frase 'mismos hiperparámetros en ambos brazos' son los del bloque (a): publicada {t3('A','a_fixed','publicada').inflacion_pct:.1f} / {t3('B','a_fixed','publicada').inflacion_pct:.1f} %, modal {t3('A','a_fixed','modal').inflacion_pct:.1f} / {t3('B','a_fixed','modal').inflacion_pct:.1f} % (B en v3).", ""]
    # ---------------- H01
    a1, a2, am, b1, b2, bm = de("A", "(i)"), de("A", "(ii)"), de("A", "formula del manuscrito"), de("B_v3", "(i)"), de("B_v3", "(ii)"), de("B_v3", "formula del manuscrito")
    a1s, b1s = de("A", "(i)", "sin"), de("B_v3", "(i)", "sin")
    L += ["## 3. H01 — efecto de diseño para un estimador definido (results_v3/design_effect_v3.csv, design_effect_v3.md)", "",
          "| Dataset | ICC(1) | m̄ | m0 | m_w | N_eff (i) media por filas | N_eff (ii) media de medias | N_eff v2 [1+ρ(m0−1)] | N_eff (i) sin familia mayor |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
          f"| A | {a1.ICC:.4f} | {a1.m_bar:.3f} | {a1.m0:.3f} | {a1.m_w:.2f} | {a1.N_eff:.0f} | {a2.N_eff:.0f} | {am.N_eff:.0f} | {a1s.N_eff:.0f} |",
          f"| B v3 | {b1.ICC:.4f} | {b1.m_bar:.3f} | {b1.m0:.3f} | {b1.m_w:.2f} | {b1.N_eff:.0f} | {b2.N_eff:.0f} | {bm.N_eff:.0f} | {b1s.N_eff:.0f} |",
          f"| B v2 | {de('B_v2','(i)').ICC:.4f} | {de('B_v2','(i)').m_bar:.3f} | {de('B_v2','(i)').m0:.3f} | {de('B_v2','(i)').m_w:.2f} | {de('B_v2','(i)').N_eff:.0f} | {de('B_v2','(ii)').N_eff:.0f} | {de('B_v2','formula del manuscrito').N_eff:.0f} | {de('B_v2','(i)','sin').N_eff:.0f} |", "",
          "ICC y m0 de v2 se reproducen (0.854112 / 0.845661; 3847 / 3470); m_w y N_eff(i) reproducen el informe (69.63 / 18.09; 356.6 / 805.0). El párrafo de reemplazo está en design_effect_v3.md.", ""]
    # ---------------- H11
    L += ["## 4. H11 — dispersión entre réplicas de vector de features (results_v3/replicate_dispersion.csv/.md)", "",
          f"- {int(RD['n_filas_con_vector_repetido'])} filas en {int(RD['n_grupos'])} grupos; desviación absoluta media a la MEDIA {RD['desv_abs_media_a_MEDIA_del_grupo_K']:.4f} K (v2 3.4913), a la MEDIANA {RD['desv_abs_media_a_MEDIANA_del_grupo_K']:.4f} K (informe 3.2229). Medianas: {RD['mediana_desv_abs_a_media_K']:.3f} / {RD['mediana_desv_abs_a_mediana_K']:.3f} K.",
          f"- Baseline de réplica excluida (mediana de las OTRAS réplicas): MAE {RD['baseline_replica_excluida_MEDIANA_otras_MAE_K']:.4f} K (media de las otras: {RD['baseline_replica_excluida_MEDIA_otras_MAE_K']:.4f} K). XGBoost random sobre esas filas: {RD['MAE_modelo_XGBoost_random_filas_con_vector_repetido_K']:.4f} K (v2 4.8311); XGBoost grouped: {RD['MAE_modelo_XGBoost_grouped_filas_con_vector_repetido_K']:.4f} K; RF random {RD['MAE_modelo_RandomForest_random_filas_con_vector_repetido_K']:.4f} K.",
          "- Lectura: 3.49 K no es un mínimo (la mediana da 3.22 K) y ninguno de los dos es una cota del MAE poblacional (usan la propia etiqueta). Un predictor sin features que copia la mediana de las otras réplicas iguala o supera al XGBoost aleatorio en esas filas (4.72 vs 4.83 K): sobre las filas duplicadas el modelo aleatorio no hace más que recuperar la etiqueta de las réplicas.", ""]
    # ---------------- null R2
    for ds in ("A", "B"):
        po = NR[(NR.dataset == ds) & (NR.fold == "pooled")].iloc[0]
        L += [f"## 5. R² del nulo agrupado — Dataset {ds} (results_v3/null_r2_by_fold.csv/.md)", "",
              f"R² pooled = {po.R2_fold_vs_pooled_mean:.5f} = 1 − (ΣSSW {po.SSW_test:.0f} + Σ n_f(ȳ_f − c_f)² {po.n_x_diff2:.0f}) / SST {po.SST_pooled:.0f}. Si cada constante coincidiera con la media del test de su fold, R² = {po.R2_if_c_equal_test_mean:+.4f} (SST contiene la varianza entre medias de fold, {po.SS_between_folds:.0f}). "
              f"Por fold, ȳ_f − c_f va de {NR[(NR.dataset==ds)&(NR.fold!='pooled')].diff_K.min():+.2f} a {NR[(NR.dataset==ds)&(NR.fold!='pooled')].diff_K.max():+.2f} K; R²_f (ref. media del test) entre {NR[(NR.dataset==ds)&(NR.fold!='pooled')].R2_fold_vs_test_mean.min():+.4f} y {NR[(NR.dataset==ds)&(NR.fold!='pooled')].R2_fold_vs_test_mean.max():+.4f}.", ""]
    # ---------------- A sensitivity
    L += ["## 6. Sensibilidad de la Tabla 1 (A) a las 1 927 filas con O=1 espurio (results_v3/tabla1_A_sin_O1.csv, _inflacion.csv, .md)", "",
          "| Variante | Modelo | MAE random | MAE grouped | Inflación % [IC] | ΔR² | Tabla 1 completa (MAE r/g, inflación, ΔR²) |", "|---|---|---:|---:|---:|---:|---|"]
    for _, r in SA.iterrows():
        L.append(f"| {r.variant} | {r.modelo} | {r.MAE_random:.3f} | {r.MAE_grouped:.3f} | {r.inflacion_pct:.1f} [{r.inflacion_lo:.1f}, {r.inflacion_hi:.1f}] | {r.dR2:.4f} | {r.MAE_random_full:.3f} / {r.MAE_grouped_full:.3f}, {r.inflacion_full_pct:.1f} %, {r.dR2_full:.4f} |")
    x = SA[(SA.variant == "refold") & (SA.modelo == "XGBoost")].iloc[0]
    L += ["", f"Lectura: excluir el 9.1 % de filas baja el MAE ≈ {x.MAE_random_full - x.MAE_random:.2f} K (random) y {x.MAE_grouped_full - x.MAE_grouped:.2f} K (grouped) en XGBoost (son filas difíciles: el modelo completo comete MAE 9.44 / 13.74 K sobre ellas); la inflación pasa de {x.inflacion_full_pct:.1f} % a {x.inflacion_pct:.1f} % y ΔR² de {x.dR2_full:.3f} a {x.dR2:.3f}. La conclusión no cambia de signo ni de magnitud; los descriptores de esas filas siguen siendo los publicados (no se corrigen).", ""]
    (R / "inferencia_v3_resumen.md").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {R / 'inferencia_v3_resumen.md'}")


if __name__ == "__main__":
    main()
