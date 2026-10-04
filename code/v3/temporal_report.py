"""temporal_report.py -- render the experiment-7 tables/figure from the CSVs of temporal_validation.py
(no model refit). Inputs <RESULTS_V3>/temporal_validation.csv, temporal_validation_comparacion.csv.
Outputs <RESULTS_V3>/temporal_validation.md, temporal_validation_tabla.tex (booktabs + siunitx),
        temporal_validation_fig.png/.pdf, temporal_validation_resumen.json (numbers quoted in the text).
Usage:  python code/v3/temporal_report.py [--principal 2010] [--modelo XGBoost]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
PROTO_LABEL = {"L0_aleatorio": "L0 aleatorio (KFold)", "L1_composicion": "L1 composición", "L2_sistema": "L2 sistema elemental",
               "L3_anfitrion": "L3 anfitrión (frac. ≥ 0,05)", "L4_loco_cluster": "L4 LOCO (KMeans k=50)"}
PROTO_TEX = {"L0_aleatorio": "L0 random $K$-fold", "L1_composicion": "L1 grouped by composition", "L2_sistema": "L2 grouped by elemental system",
             "L3_anfitrion": "L3 grouped by host elements", "L4_loco_cluster": "L4 leave-clusters-out ($k=50$)"}   # manuscript is in English
PROTOS = list(PROTO_LABEL)


def fmt(v, d=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def V(r, a="valor"):
    return float(getattr(r, a)) if r is not None else np.nan


def cell(v, lo, hi, d=2):
    return f"{fmt(v, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--principal", type=int, default=2010)
    ap.add_argument("--modelo", default="XGBoost")
    ap.add_argument("--lang", default="es", choices=("es", "en"), help="language of the FIGURE labels (tables/markdown unchanged)")
    ap.add_argument("--fig-only", action="store_true", help="only redraw the figure (do not rewrite md/tex/json)")
    a = ap.parse_args(argv)
    EN = a.lang == "en"
    df = pd.read_csv(RESULTS_V3 / "temporal_validation.csv")
    cmp = pd.read_csv(RESULTS_V3 / "temporal_validation_comparacion.csv")
    prov = json.loads(df.procedencia.iloc[0])
    P = a.principal; M = a.modelo
    cuts = sorted(df.corte.unique())
    base = df[df.escenario == "principal"]

    def get(cond, model, stratum, metric, T, esc="principal"):
        r = df[(df.corte == T) & (df.escenario == esc) & (df.condicion == cond) & (df.modelo == model) & (df.estrato == stratum) & (df.metrica == metric)]
        return r.iloc[0] if len(r) else None

    def cget(proto, model, est, metric, T, esc="principal"):
        r = cmp[(cmp.corte == T) & (cmp.escenario == esc) & (cmp.protocolo == proto) & (cmp.modelo == model) & (cmp.estimacion == est) & (cmp.metrica == metric)]
        return r.iloc[0] if len(r) else None

    # ---------------------------------------------------------------- summary numbers
    resumen = dict(procedencia=prov, corte_principal=P, modelo=M, cortes={})
    for T in cuts:
        d = {}
        for st in ("todo", "familia_nueva", "familia_vista", "composicion_vista_reportes"):
            r = {k: get("temporal_real", M, st, k, T) for k in ("MAE", "RMSE", "R2", "Spearman", "top100", "prevalencia_alto_tc")}
            if r["MAE"] is None: continue
            d[st] = {k: dict(valor=float(v.valor), ci=[float(v.ci_lo_familias), float(v.ci_hi_familias)], n=int(v.n), n_familias=int(v.n_familias))
                     for k, v in r.items() if v is not None}
        d["protocolos"] = {}
        for p in PROTOS:
            for est in ("todo", "reponderado_mezcla_test"):
                rows = {k: cget(p, M, est, k, T) for k in ("MAE", "RMSE", "R2", "Spearman", "top100")}
                if rows["MAE"] is None: continue
                d["protocolos"][f"{p}|{est}"] = {k: dict(estimacion=float(v.estimacion_retrospectiva), real=float(v.temporal_real), error_abs=float(v.error_abs),
                                                          ci_abs=[float(v.ci_lo_abs), float(v.ci_hi_abs)], error_rel=float(v.error_rel),
                                                          ci_rel=[float(v.ci_lo_rel), float(v.ci_hi_rel)]) for k, v in rows.items() if v is not None}
        resumen["cortes"][str(T)] = d
    # ranking of protocols by |error| in MAE and R2 at each cut (both models)
    rank = {}
    for T in cuts:
        for model in sorted(df.modelo.unique()):
            for metric in ("MAE", "R2", "top100", "Spearman"):
                sub = cmp[(cmp.corte == T) & (cmp.escenario == "principal") & (cmp.modelo == model) & (cmp.estimacion == "todo") & (cmp.metrica == metric) & cmp.protocolo.isin(PROTOS)]
                if len(sub):
                    o = sub.assign(a=sub.error_abs.abs()).sort_values("a")
                    rank[f"{T}|{model}|{metric}"] = o.protocolo.tolist()
    resumen["ranking_por_error_absoluto"] = rank
    # which protocol's CI covers the real value
    cover = {}
    for T in cuts:
        for model in sorted(df.modelo.unique()):
            for metric in ("MAE", "R2", "top100"):
                sub = cmp[(cmp.corte == T) & (cmp.escenario == "principal") & (cmp.modelo == model) & (cmp.metrica == metric) & cmp.protocolo.isin(PROTOS)]
                cover[f"{T}|{model}|{metric}"] = {f"{r.protocolo}|{r.estimacion}": bool(r.ci_lo_abs <= 0 <= r.ci_hi_abs) for r in sub.itertuples()}
    resumen["intervalo_error_cubre_cero"] = cover
    if not a.fig_only: json.dump(resumen, open(RESULTS_V3 / "temporal_validation_resumen.json", "w"), indent=1, ensure_ascii=False)

    # ---------------------------------------------------------------- markdown
    L = ["# Validación temporal (experimento 7, v3)", "", f"Procedencia: `{json.dumps(prov, ensure_ascii=False)[:1500]}`", "",
         "Pregunta: ¿qué protocolo retrospectivo (aleatorio / composición / sistema elemental / anfitrión / LOCO) anticipa mejor el "
         "rendimiento real sobre superconductores reportados por primera vez DESPUÉS de un corte temporal?", "",
         "Diseño: corpus `data/temporal_corpus_v3.csv` (una fila por composición normalizada, Tc = mediana de reportes, año = primer reporte). "
         "Para cada corte T: (a) modelo entrenado en todo lo anterior a T (año < T), congelado, evaluado en lo posterior (año ≥ T); "
         "(b) sólo con lo anterior a T, predicciones OOF 5-fold bajo L0–L4; (c) estimación (b) − real (a). Umbral alto-Tc = percentil 90 de la "
         "Tc de entrenamiento, operador ≥. Intervalos: bootstrap de familias completas (B=2000, semilla 0), pareado; condicionales a las "
         "predicciones ya ajustadas. L1 (GroupKFold por composición) coincide con L0 en este corpus deduplicado salvo por la partición.", ""]
    L += ["## 1. Rendimiento temporal real (modelo congelado) — " + M, "",
          "| T | n_train | n_test | % fam. nueva | estrato | n | fam. | MAE [IC fam.] | RMSE | R² [IC fam.] | Spearman | top-100 [IC] | prevalencia ≥p90 |",
          "|---:|---:|---:|---:|---|---:|---:|---|---|---|---|---|---:|"]
    for T in cuts:
        meta = prov["meta"].get(f"principal_T{T}", {})
        for st in ("todo", "familia_nueva", "familia_vista", "composicion_vista_reportes"):
            r = {k: get("temporal_real", M, st, k, T) for k in ("MAE", "RMSE", "R2", "Spearman", "top100", "prevalencia_alto_tc")}
            if r["MAE"] is None: continue
            L.append(f"| {T} | {meta.get('n_train','')} | {meta.get('n_test','')} | {100*meta.get('frac_familia_nueva_test',np.nan):.1f} | {st} | {int(r['MAE'].n)} | {int(r['MAE'].n_familias)} | "
                     f"{cell(r['MAE'].valor, r['MAE'].ci_lo_familias, r['MAE'].ci_hi_familias)} | {fmt(r['RMSE'].valor)} | "
                     f"{cell(r['R2'].valor, r['R2'].ci_lo_familias, r['R2'].ci_hi_familias)} | {fmt(r['Spearman'].valor)} | "
                     f"{cell(r['top100'].valor, r['top100'].ci_lo_familias, r['top100'].ci_hi_familias)} | {fmt(V(r['prevalencia_alto_tc']), 3)} |")
    L += ["", "`composicion_vista_reportes` = re-mediciones posteriores a T de composiciones ya presentes en el entrenamiento (nivel reporte; la "
          "predicción es la del modelo para una composición de su propio entrenamiento). Para RandomForest véase el CSV.", ""]
    # protocols table per cut
    L += ["## 2. Estimación retrospectiva por protocolo vs. real — " + M, ""]
    for T in cuts:
        rt = get("temporal_real", M, "todo", "MAE", T)
        if rt is None: continue
        L += [f"### T = {T}  (n_train = {int(rt.n) if False else prov['meta'][f'principal_T{T}']['n_train']}, n_test = {int(rt.n)}, "
              f"{100*prov['meta'][f'principal_T{T}']['frac_familia_nueva_test']:.1f} % del test en familias nuevas)", "",
              "| protocolo | % OOF fam. nueva | MAE est. [IC] | MAE real | Δ MAE [IC] | R² est. [IC] | R² real | Δ R² [IC] | top-100 est. | top-100 real | Δ top-100 [IC] | Spearman est. / real |",
              "|---|---:|---|---|---|---|---|---|---|---|---|---|"]
        for p in PROTOS:
            for est in ("todo", "reponderado_mezcla_test"):
                c = {k: cget(p, M, est, k, T) for k in ("MAE", "R2", "top100", "Spearman")}
                if c["MAE"] is None: continue
                e = {k: get(p, M, est, k, T) for k in ("MAE", "R2", "top100")}
                name = PROTO_LABEL[p] + (" — reponderado a la mezcla del test" if est != "todo" else "")
                L.append(f"| {name} | {100*c['MAE'].frac_familia_nueva_oof:.1f} | {cell(e['MAE'].valor, e['MAE'].ci_lo_familias, e['MAE'].ci_hi_familias)} | {fmt(c['MAE'].temporal_real)} | "
                         f"{cell(c['MAE'].error_abs, c['MAE'].ci_lo_abs, c['MAE'].ci_hi_abs)} | {cell(e['R2'].valor, e['R2'].ci_lo_familias, e['R2'].ci_hi_familias, 3)} | {fmt(c['R2'].temporal_real, 3)} | "
                         f"{cell(c['R2'].error_abs, c['R2'].ci_lo_abs, c['R2'].ci_hi_abs, 3)} | {fmt(e['top100'].valor)} | {fmt(c['top100'].temporal_real)} | "
                         f"{cell(c['top100'].error_abs, c['top100'].ci_lo_abs, c['top100'].ci_hi_abs)} | {fmt(c['Spearman'].estimacion_retrospectiva)} / {fmt(c['Spearman'].temporal_real)} |")
        L.append("")
    # ranking summary
    L += ["## 3. Ranking de protocolos por |estimación − real| (estimación 'todo')", "",
          "| T | modelo | MAE | R² | top-100 | Spearman |", "|---:|---|---|---|---|---|"]
    for T in cuts:
        for model in sorted(df.modelo.unique()):
            row = [rank.get(f"{T}|{model}|{m}", []) for m in ("MAE", "R2", "top100", "Spearman")]
            L.append(f"| {T} | {model} | " + " | ".join(" > ".join(x.split('_')[0] for x in r) for r in row) + " |")
    L += ["", "(el primero es el protocolo cuya estimación retrospectiva queda más cerca del rendimiento temporal real)", ""]
    # robustness
    L += ["## 4. Robustez", ""]
    seeds = df[df.condicion.str.contains("_seed") & (df.estrato == "todo") & (df.modelo == M) & df.metrica.isin(["MAE", "R2", "top100"])]
    if len(seeds):
        L += ["### 4.1 Semillas (L0: KFold random_state 42/0/1/2; L4: KMeans random_state 0/1/2/3) — " + M, "",
              "| T | protocolo | semilla | MAE | R² | top-100 |", "|---:|---|---|---|---|---|"]
        for T in cuts:
            for p in ("L0_aleatorio", "L4_loco_cluster"):
                basev = {m: get(p, M, "todo", m, T) for m in ("MAE", "R2", "top100")}
                if basev["MAE"] is not None:
                    L.append(f"| {T} | {p} | base | {fmt(basev['MAE'].valor)} | {fmt(basev['R2'].valor, 3)} | {fmt(basev['top100'].valor)} |")
                sub = seeds[(seeds.corte == T) & seeds.condicion.str.startswith(p)]
                for cond in sorted(sub.condicion.unique()):
                    v = {m: sub[(sub.condicion == cond) & (sub.metrica == m)].valor.iloc[0] for m in ("MAE", "R2", "top100")}
                    L.append(f"| {T} | {p} | {cond.split('_seed')[1]} | {fmt(v['MAE'])} | {fmt(v['R2'], 3)} | {fmt(v['top100'])} |")
        L.append("")
    for esc, title in (("tc_max", "### 4.2 Tc = máximo por composición (en lugar de la mediana), corte principal"),
                       ("sin_anio_ambiguo", "### 4.3 Excluyendo composiciones con año de regla de respaldo (anio_ambiguo), corte principal")):
        sub = cmp[(cmp.escenario == esc) & (cmp.modelo == M) & (cmp.estimacion == "todo")]
        if not len(sub): continue
        meta = prov["meta"].get(f"{esc}_T{P}") or prov["meta"].get({"sin_anio_ambiguo": f"sin_ambiguos_T{P}"}.get(esc, ""), {})
        L += [title + f" — {M}; n_train = {meta.get('n_train')}, n_test = {meta.get('n_test')}", "",
              "| protocolo | MAE est. | MAE real | Δ MAE [IC] | R² est. | R² real | Δ R² [IC] | top-100 est. | top-100 real |", "|---|---|---|---|---|---|---|---|---|"]
        for p in PROTOS:
            c = {k: cget(p, M, "todo", k, P, esc) for k in ("MAE", "R2", "top100")}
            if c["MAE"] is None: continue
            L.append(f"| {PROTO_LABEL[p]} | {fmt(c['MAE'].estimacion_retrospectiva)} | {fmt(c['MAE'].temporal_real)} | {cell(c['MAE'].error_abs, c['MAE'].ci_lo_abs, c['MAE'].ci_hi_abs)} | "
                     f"{fmt(c['R2'].estimacion_retrospectiva, 3)} | {fmt(c['R2'].temporal_real, 3)} | {cell(c['R2'].error_abs, c['R2'].ci_lo_abs, c['R2'].ci_hi_abs, 3)} | "
                     f"{fmt(c['top100'].estimacion_retrospectiva)} | {fmt(c['top100'].temporal_real)} |")
        L.append("")
    if not a.fig_only: (RESULTS_V3 / "temporal_validation.md").write_text("\n".join(L))

    # ---------------------------------------------------------------- LaTeX table (principal cut, both models)
    meta = prov["meta"][f"principal_T{P}"]
    tex = [f"% generated by code/v3/temporal_report.py -- experiment 7 (temporal validation), principal cut T={P}",
           "% requires \\usepackage{booktabs,siunitx,multirow}",
           "\\begin{table*}[t]", "\\centering",
           f"\\caption{{Retrospective protocol estimates versus realised performance on the temporal split. SuperCon corpus dated by first report "
           f"(\\num{{{meta['n_train']}}} compositions first reported before {P}, \\num{{{meta['n_test']}}} at or after; "
           f"{100*meta['frac_familia_nueva_test']:.0f}\\,\\% of the later compositions belong to elemental systems absent before {P}). "
           f"Estimate = 5-fold out-of-fold predictions using only the pre-{P} data; realised = one model fitted on all pre-{P} data and evaluated on the later compositions. "
           "$\\Delta$ = estimate $-$ realised; 95\\,\\% intervals from a bootstrap of whole chemical families (B = 2000), conditional on the fitted predictions. "
           "High-$T_c$ threshold = 90th percentile of the training $T_c$ (operator $\\geq$). L1 differs from L0 only in the partition (the corpus holds one row per composition). "
           "In the \\emph{Realised} row the $\\Delta$ columns give the 95\\,\\% interval of the realised value.}",
           f"\\label{{tab:temporal}}", "\\small", "\\sisetup{retain-explicit-plus}",
           "\\begin{tabular}{ll S[table-format=2.2] S[table-format=-2.2] S[table-format=1.3] S[table-format=-1.3] S[table-format=1.2] S[table-format=-1.2] S[table-format=1.2]}",
           "\\toprule",
           "Model & Protocol & {MAE (K)} & {$\\Delta$MAE} & {$R^2$} & {$\\Delta R^2$} & {top-100} & {$\\Delta$top-100} & {Spearman} \\\\",
           "\\midrule"]
    for model in ("XGBoost", "RandomForest"):
        rt = {k: get("temporal_real", model, "todo", k, P) for k in ("MAE", "R2", "top100", "Spearman")}
        if rt["MAE"] is None: continue
        tex.append(f"\\multirow{{{len(PROTOS)+2}}}{{*}}{{{model}}} & \\textbf{{Realised (train $<$ {P}, test $\\geq$ {P})}} & \\bfseries {rt['MAE'].valor:.2f} & {{[{rt['MAE'].ci_lo_familias:.2f}, {rt['MAE'].ci_hi_familias:.2f}]}} & "
                   f"\\bfseries {rt['R2'].valor:.3f} & {{[{rt['R2'].ci_lo_familias:.3f}, {rt['R2'].ci_hi_familias:.3f}]}} & \\bfseries {rt['top100'].valor:.2f} & "
                   f"{{[{rt['top100'].ci_lo_familias:.2f}, {rt['top100'].ci_hi_familias:.2f}]}} & {rt['Spearman'].valor:.2f} \\\\")
        for p in PROTOS:
            c = {k: cget(p, model, "todo", k, P) for k in ("MAE", "R2", "top100", "Spearman")}
            if c["MAE"] is None: continue
            tex.append(f" & {PROTO_TEX[p]} & {c['MAE'].estimacion_retrospectiva:.2f} & {c['MAE'].error_abs:+.2f} & {c['R2'].estimacion_retrospectiva:.3f} & "
                       f"{c['R2'].error_abs:+.3f} & {c['top100'].estimacion_retrospectiva:.2f} & {c['top100'].error_abs:+.2f} & {c['Spearman'].estimacion_retrospectiva:.2f} \\\\")
        c = {k: cget("L0_aleatorio", model, "reponderado_mezcla_test", k, P) for k in ("MAE", "R2", "top100", "Spearman")}
        if c["MAE"] is not None:
            tex.append(f" & L0 re-weighted to the test mix of new/seen systems & {c['MAE'].estimacion_retrospectiva:.2f} & {c['MAE'].error_abs:+.2f} & {c['R2'].estimacion_retrospectiva:.3f} & "
                       f"{c['R2'].error_abs:+.3f} & {c['top100'].estimacion_retrospectiva:.2f} & {c['top100'].error_abs:+.2f} & {c['Spearman'].estimacion_retrospectiva:.2f} \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex += ["\\end{tabular}", "\\end{table*}"]
    if not a.fig_only: (RESULTS_V3 / "temporal_validation_tabla.tex").write_text("\n".join(tex) + "\n")

    # ---------------------------------------------------------------- figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    metrics = [("MAE", "MAE (K)  ↓"), ("R2", "R²  ↑"), ("top100", "Top-100 precision  ↑" if EN else "Precisión top-100  ↑")]
    fig, axes = plt.subplots(len(metrics), len(cuts), figsize=(2.0 * len(cuts) + 0.8, 1.55 * len(metrics) + 0.6), sharex=True, sharey="row",
                             gridspec_kw=dict(wspace=0.08, hspace=0.18))
    axes = np.array(axes).reshape(len(metrics), len(cuts))
    col_proto, col_real, col_rf = "#1f5fa8", "#c0392b", "#7f8c8d"
    xs = np.arange(len(PROTOS))
    for j, T in enumerate(cuts):
        for i, (m, lab) in enumerate(metrics):
            ax = axes[i, j]
            rt = get("temporal_real", M, "todo", m, T)
            if rt is not None:
                ax.axhspan(rt.ci_lo_familias, rt.ci_hi_familias, color=col_real, alpha=0.15, lw=0)
                ax.axhline(rt.valor, color=col_real, lw=1.2)
            vals = [get(p, M, "todo", m, T) for p in PROTOS]
            v = np.array([r.valor if r is not None else np.nan for r in vals]); lo = np.array([r.ci_lo_familias if r is not None else np.nan for r in vals]); hi = np.array([r.ci_hi_familias if r is not None else np.nan for r in vals])
            ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="o", color=col_proto, ms=3.5, capsize=1.5, lw=0.9)
            rw = get("L0_aleatorio", M, "reponderado_mezcla_test", m, T)
            if rw is not None:
                ax.plot([0], [rw.valor], marker="D", ms=3.2, mfc="white", mec=col_proto, ls="none")
            other = "RandomForest" if M == "XGBoost" else "XGBoost"
            vo = [get(p, other, "todo", m, T) for p in PROTOS]
            ax.plot(xs + 0.18, [r.valor if r is not None else np.nan for r in vo], marker="s", ms=2.6, color=col_rf, ls="none", alpha=0.8)
            rto = get("temporal_real", other, "todo", m, T)
            if rto is not None: ax.plot([len(PROTOS) - 0.55], [rto.valor], marker="s", ms=2.6, color=col_rf, ls="none", alpha=0.8)
            ax.set_xticks(xs); ax.set_xticklabels([p.split("_")[0] for p in PROTOS])
            ax.margins(x=0.08, y=0.08)
            if j == 0: ax.set_ylabel(lab)
            if i == 0:
                meta = prov["meta"][f"principal_T{T}"]
                ax.set_title((f"cut {T}: {meta['n_train']/1000:.1f}k → {meta['n_test']/1000:.1f}k\n{100*meta['frac_familia_nueva_test']:.0f}% in new systems" if EN else
                              f"corte {T}: {meta['n_train']/1000:.1f}k → {meta['n_test']/1000:.1f}k\n{100*meta['frac_familia_nueva_test']:.0f} % en familias nuevas"), loc="left")
    from matplotlib.lines import Line2D
    if EN:
        handles = [Line2D([], [], color=col_real, lw=1.2, label=f"realised performance on the temporal split ({M}); band = 95% family-bootstrap interval"),
                   Line2D([], [], marker="o", color=col_proto, ls="none", ms=3.5, label=f"retrospective out-of-fold estimate of each protocol ({M})"),
                   Line2D([], [], marker="D", mfc="white", mec=col_proto, ls="none", ms=3.2, label="L0 re-weighted to the new/seen-system mix of the later compositions"),
                   Line2D([], [], marker="s", color=col_rf, ls="none", ms=2.6, label="RandomForest (protocols; realised value at the right)")]
    else:
        handles = [Line2D([], [], color=col_real, lw=1.2, label=f"rendimiento temporal real ({M}); banda = IC 95 % familias"),
               Line2D([], [], marker="o", color=col_proto, ls="none", ms=3.5, label=f"estimación retrospectiva OOF por protocolo ({M})"),
               Line2D([], [], marker="D", mfc="white", mec=col_proto, ls="none", ms=3.2, label="L0 reponderado a la mezcla nuevas/vistas del test"),
               Line2D([], [], marker="s", color=col_rf, ls="none", ms=2.6, label="RandomForest (protocolos y real, a la derecha)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(("Random validation (L0/L1) overstates R² on superconductors reported after the cut at every cut;\n"
                  "grouped protocols (L2–L4) come closer but none anticipates it stably") if EN else
                 ("La validación aleatoria (L0/L1) sobreestima el R² sobre superconductores posteriores al corte en todos los cortes;\n"
                 "los protocolos agrupados (L2–L4) se acercan más pero ninguno lo anticipa de forma estable"), x=0.01, y=1.02, ha="left", fontsize=8)
    stem = "temporal_validation_fig" + ("_en" if EN else "")
    fig.savefig(RESULTS_V3 / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS_V3 / f"{stem}.pdf", bbox_inches="tight")
    if EN:
        import shutil; shutil.copyfile(RESULTS_V3 / f"{stem}.png", ROOT / "figure5.png"); shutil.copyfile(RESULTS_V3 / f"{stem}.pdf", ROOT / "figure5.pdf")
        print(f"[save] {ROOT / 'figure5.png'} (+ .pdf)")
    print(json.dumps(dict(md=str(RESULTS_V3 / "temporal_validation.md"), tex=str(RESULTS_V3 / "temporal_validation_tabla.tex"),
                          fig=str(RESULTS_V3 / "temporal_validation_fig.png"), ranking=rank), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
