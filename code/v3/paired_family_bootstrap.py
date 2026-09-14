"""H07 - paired bootstrap of WHOLE FAMILIES for the contrast random - grouped (Table 1 models).

Input: results_v3/oof_v3.csv.gz (code/v3/oof_v3.py). For every dataset x model with both protocols:
  * family bootstrap, PAIRED: 2000 resamples (seed 0) of the k families with replacement; the same
    resampled families are used for the random and the grouped predictions; statistics recomputed
    per resample: pooled MAE, RMSE, R2 (each arm), inflation = 100*(MAE_g/MAE_r - 1), dR2 = R2_r - R2_g,
    macro MAE (mean of per-family MAE) and its contrast (ratio and difference).
  * row bootstrap, unpaired (the v2 scheme of results_v2/bootstrap_intervalos.csv) and paired, same
    n_boot/seed, for the width comparison (ratio family / row).
Intervals are percentile 2.5 / 97.5 and are CONDITIONAL on the out-of-fold predictions already fitted
(seed 42): they carry the sampling variability of the evaluated materials, not the variability of
re-training (which the five seeds of Table 2 address separately).

Outputs: results_v3/paired_family_bootstrap.csv (one row per dataset x model x statistic x scheme),
         results_v3/paired_family_bootstrap.md,
         results_v3/family_losses_v3.csv (per-family n, MAE_random, MAE_grouped for both models).

    python code/v3/paired_family_bootstrap.py [--n-boot 2000] [--seed 0]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h07_common import RES_V2, RES_V3, VERS, FamilyStats, paired_family_bootstrap, paired_row_bootstrap, ci


def summarize(name, point, draws, scheme, ds, model, k, n, nb, seed):
    lo, hi = ci(draws)
    return dict(dataset=ds, model=model, statistic=name, scheme=scheme, point=point, lo=lo, hi=hi, width=hi - lo,
                boot_se=float(np.std(draws, ddof=1)), n_rows=n, k_families=k, n_boot=nb, boot_seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, default=RES_V3 / "oof_v3.csv.gz")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RES_V3)
    a = ap.parse_args()
    oof = pd.read_csv(a.oof)
    v2 = pd.read_csv(RES_V2 / "bootstrap_intervalos.csv").set_index("dataset")
    proc = (f"code/v3/paired_family_bootstrap.py | input {a.oof.name} (code/v3/oof_v3.py, seed 42) | family bootstrap = "
            f"resample the k families with replacement, n_boot={a.n_boot}, numpy default_rng({a.seed}), SAME draw for both "
            f"protocols (paired); row bootstrap = resample rows, same n_boot/seed, unpaired (v2 scheme) and paired | percentile CI "
            f"2.5/97.5 | intervals CONDITIONAL on the fitted OOF predictions (no re-training) | {VERS}")
    rows, fam_rows = [], []
    for (ds, model), g in oof.groupby(["dataset", "model"]):
        r = g[g.protocol == "random"].sort_values("row_index"); q = g[g.protocol == "grouped"].sort_values("row_index")
        if len(r) == 0 or len(q) == 0: continue
        assert np.array_equal(r.row_index.values, q.row_index.values) and np.array_equal(r.family_id.values, q.family_id.values)
        y, gr = r.y_true.values, r.family_id.values
        pr, pg = r.y_pred.values, q.y_pred.values
        fs = FamilyStats(y, gr, {"random": pr, "grouped": pg})
        one = np.ones(fs.k)
        P = {c: fs.stats(one, c) for c in ("random", "grouped")}
        # per-family losses (for the .csv of family losses)
        fam_rows.append(pd.DataFrame({"dataset": ds, "model": model, "family_id": fs.family_ids, "n": fs.n.astype(int),
                                      "MAE_random": fs.sabs["random"] / fs.n, "MAE_grouped": fs.sabs["grouped"] / fs.n,
                                      "mean_Tc": fs.sy / fs.n}))
        # ---- family bootstrap, paired
        F = paired_family_bootstrap(fs, ("random", "grouped"), a.n_boot, a.seed)
        stats = {}
        for c in ("random", "grouped"):
            for m in ("MAE", "RMSE", "R2", "MAE_macro"):
                stats[f"{m}_{c}"] = (P[c][m], F[f"{m}_{c}"])
        stats["inflacion_pct"] = (100 * (P["grouped"]["MAE"] / P["random"]["MAE"] - 1), 100 * (F["MAE_grouped"] / F["MAE_random"] - 1))
        stats["dMAE_K"] = (P["grouped"]["MAE"] - P["random"]["MAE"], F["MAE_grouped"] - F["MAE_random"])
        stats["dRMSE_K"] = (P["grouped"]["RMSE"] - P["random"]["RMSE"], F["RMSE_grouped"] - F["RMSE_random"])
        stats["dR2"] = (P["random"]["R2"] - P["grouped"]["R2"], F["R2_random"] - F["R2_grouped"])
        stats["inflacion_macro_pct"] = (100 * (P["grouped"]["MAE_macro"] / P["random"]["MAE_macro"] - 1), 100 * (F["MAE_macro_grouped"] / F["MAE_macro_random"] - 1))
        stats["dMAE_macro_K"] = (P["grouped"]["MAE_macro"] - P["random"]["MAE_macro"], F["MAE_macro_grouped"] - F["MAE_macro_random"])
        for name, (pt, dr) in stats.items():
            rows.append(summarize(name, pt, dr, "family_paired", ds, model, fs.k, len(y), a.n_boot, a.seed))
        # ---- row bootstrap, unpaired (v2 scheme) and paired
        for scheme, paired in (("row_unpaired_v2scheme", False), ("row_paired", True)):
            R = paired_row_bootstrap(y, pr, pg, a.n_boot, a.seed, paired)
            rs = {"MAE_random": (P["random"]["MAE"], R["MAE_a"]), "MAE_grouped": (P["grouped"]["MAE"], R["MAE_b"]),
                  "RMSE_random": (P["random"]["RMSE"], R["RMSE_a"]), "RMSE_grouped": (P["grouped"]["RMSE"], R["RMSE_b"]),
                  "R2_random": (P["random"]["R2"], R["R2_a"]), "R2_grouped": (P["grouped"]["R2"], R["R2_b"]),
                  "inflacion_pct": (stats["inflacion_pct"][0], 100 * (R["MAE_b"] / R["MAE_a"] - 1)),
                  "dMAE_K": (stats["dMAE_K"][0], R["MAE_b"] - R["MAE_a"]), "dRMSE_K": (stats["dRMSE_K"][0], R["RMSE_b"] - R["RMSE_a"]),
                  "dR2": (stats["dR2"][0], R["R2_a"] - R["R2_b"])}
            for name, (pt, dr) in rs.items():
                rows.append(summarize(name, pt, dr, scheme, ds, model, fs.k, len(y), a.n_boot, a.seed))
        print(f"{ds} {model:14s} MAE_r={P['random']['MAE']:.4f} MAE_g={P['grouped']['MAE']:.4f} infl={stats['inflacion_pct'][0]:.2f}% "
              f"fam-CI [{ci(stats['inflacion_pct'][1])[0]:.2f},{ci(stats['inflacion_pct'][1])[1]:.2f}]  macro_r={P['random']['MAE_macro']:.3f} macro_g={P['grouped']['MAE_macro']:.3f}", flush=True)

    D = pd.DataFrame(rows)
    # width ratios family_paired / row_*
    piv = D.pivot_table(index=["dataset", "model", "statistic"], columns="scheme", values="width")
    D = D.merge(piv.rename(columns=lambda c: f"width_{c}").reset_index(), on=["dataset", "model", "statistic"], how="left")
    D["width_ratio_family_over_row_unpaired"] = D["width_family_paired"] / D["width_row_unpaired_v2scheme"]
    D["width_ratio_family_over_row_paired"] = D["width_family_paired"] / D["width_row_paired"]
    # v2 published intervals (XGBoost, rows unpaired, B v2 checkpoint) for reference
    v2map = {"MAE_random": ("MAE_random_lo", "MAE_random_hi"), "MAE_grouped": ("MAE_grouped_lo", "MAE_grouped_hi"), "inflacion_pct": ("inflacion_lo", "inflacion_hi")}
    for i, r in D.iterrows():
        if r.model == "XGBoost" and r.statistic in v2map and r.dataset in v2.index:
            lo, hi = v2map[r.statistic]
            D.loc[i, "v2_lo"], D.loc[i, "v2_hi"] = v2.loc[r.dataset, lo], v2.loc[r.dataset, hi]
            D.loc[i, "v2_width"] = v2.loc[r.dataset, hi] - v2.loc[r.dataset, lo]
    D["procedencia"] = proc + " | v2_* = results_v2/bootstrap_intervalos.csv (rows, unpaired, XGBoost seed 42, B = v2 checkpoint N=12440)"
    a.out.mkdir(parents=True, exist_ok=True)
    D.to_csv(a.out / "paired_family_bootstrap.csv", index=False)
    FL = pd.concat(fam_rows, ignore_index=True); FL["procedencia"] = proc; FL.to_csv(a.out / "family_losses_v3.csv", index=False)

    # ---------------------------------------------------------------- markdown
    L = ["# H07 - Inferencia pareada por familias (v3)", "",
         f"Procedencia: `{proc}`", "",
         "Intervalos percentil 95 %, **condicionales a las predicciones OOF ya ajustadas** (semilla 42; no incluyen la variabilidad de reentrenar). "
         "Bootstrap de familias = remuestreo con reemplazo de las k familias completas, MISMAS familias en ambos brazos (pareado). "
         "Bootstrap por filas = esquema de results_v2/bootstrap_intervalos.csv (filas independientes en cada brazo, no pareado) y su variante pareada. "
         "Umbral alto-Tc no interviene aquí.", ""]
    fam = D[D.scheme == "family_paired"]
    L += ["## Contraste random - grouped, bootstrap pareado por familias", "",
          "| Dataset | Modelo | MAE random | MAE grouped | Inflación % [IC] | ΔMAE K [IC] | ΔR² [IC] | MAE macro random | MAE macro grouped | Inflación macro % [IC] |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (ds, model), g in fam.groupby(["dataset", "model"]):
        s = g.set_index("statistic")
        f = lambda n, d=2: f"{s.loc[n,'point']:.{d}f} [{s.loc[n,'lo']:.{d}f}, {s.loc[n,'hi']:.{d}f}]"
        L.append(f"| {ds} | {model} | {f('MAE_random')} | {f('MAE_grouped')} | {f('inflacion_pct',1)} | {f('dMAE_K')} | {f('dR2',3)} | {f('MAE_macro_random')} | {f('MAE_macro_grouped')} | {f('inflacion_macro_pct',1)} |")
    L += ["", "## Anchura del intervalo: familias (pareado) frente a filas", "",
          "| Dataset | Modelo | Estadístico | IC familias | IC filas no pareado (v2) | IC filas pareado | anchura fam/filas-no-par | anchura fam/filas-par | IC v2 publicado |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for (ds, model, st), g in D.groupby(["dataset", "model", "statistic"]):
        if st not in ("MAE_random", "MAE_grouped", "inflacion_pct", "dMAE_K", "dR2", "R2_random", "R2_grouped"): continue
        s = g.set_index("scheme")
        if not {"family_paired", "row_unpaired_v2scheme", "row_paired"} <= set(s.index): continue
        d = 3 if st.startswith(("R2", "dR2")) else 2
        f = lambda sc: f"[{s.loc[sc,'lo']:.{d}f}, {s.loc[sc,'hi']:.{d}f}]"
        v2s = f"[{s.loc['family_paired','v2_lo']:.{d}f}, {s.loc['family_paired','v2_hi']:.{d}f}]" if pd.notna(s.loc["family_paired"].get("v2_lo", np.nan)) else "-"
        L.append(f"| {ds} | {model} | {st} | {f('family_paired')} | {f('row_unpaired_v2scheme')} | {f('row_paired')} | "
                 f"{s.loc['family_paired','width_ratio_family_over_row_unpaired']:.2f} | {s.loc['family_paired','width_ratio_family_over_row_paired']:.2f} | {v2s} |")
    (a.out / "paired_family_bootstrap.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))
    print(f"wrote {a.out / 'paired_family_bootstrap.csv'} ({len(D)} rows), .md, family_losses_v3.csv")


if __name__ == "__main__":
    main()
