"""
agg_tabla2.py — Agrega las 60 corridas de run_tabla2_shuffle.py en la Tabla 2 v2
y la compara contra la Tabla 2 publicada.

Convenciones (fijadas aqui y documentadas en la columna 'procedencia'):
  * inflacion_pct se calcula EMPAREJADA por semilla:
        infl(s) = 100 * (MAE_grouped(s) - MAE_random(s)) / MAE_random(s)
    y luego se toma media +/- desviacion sobre s. No se calcula a partir de las
    medias, porque ambos brazos comparten la semilla y el emparejamiento importa.
  * dR2(s) = R2_grouped(s) - R2_random(s)   (SIGNADO; negativo = el agrupado cae)
  * desviacion = np.std con ddof=0 (poblacional), igual que el codigo publicado
    (results/repeated_seed_intervals.csv), para que las cifras sean comparables.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- repository root (portable) -------------------------------------------------------
# ROOT = the repository root. Override with the SUPERCON_ROOT environment variable;
# by default it is resolved from this file's location: code/v2/<script>.py -> parents[2].
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
REPO = ROOT                       # the repository itself (code/, data/, results/, figures/)
CODE = ROOT / "code"              # tc_pipeline.py, datasetB_pipeline.py (hyperparameters live here)
DATA = ROOT / "data"
OUT  = ROOT / "results_v2"        # v2 measurements (inside the repository)
SEEDS = [0, 1, 2, 3, 4]

# Tabla 2 publicada (results/repeated_seed_intervals.csv del repo, GroupKFold(5) SIN shuffle)
PUB = {
    ("A", "XGBoost"):      dict(MAE_random_mean=5.25, MAE_random_std=0.02, MAE_grouped_mean=8.34,
                                MAE_grouped_std=0.10, R2_random_mean=0.928, R2_random_std=0.001,
                                R2_grouped_mean=0.850, R2_grouped_std=0.004, inflacion_pct=59, inflacion_pct_std=2),
    ("A", "RandomForest"): dict(MAE_random_mean=5.41, MAE_random_std=0.01, MAE_grouped_mean=8.69,
                                MAE_grouped_std=0.07, R2_random_mean=0.924, R2_random_std=0.001,
                                R2_grouped_mean=0.844, R2_grouped_std=0.003, inflacion_pct=61, inflacion_pct_std=1),
    ("B", "XGBoost"):      dict(MAE_random_mean=4.03, MAE_random_std=0.01, MAE_grouped_mean=6.20,
                                MAE_grouped_std=0.05, R2_random_mean=0.929, R2_random_std=0.001,
                                R2_grouped_mean=0.854, R2_grouped_std=0.003, inflacion_pct=54, inflacion_pct_std=2),
    ("B", "RandomForest"): dict(MAE_random_mean=4.46, MAE_random_std=0.01, MAE_grouped_mean=6.48,
                                MAE_grouped_std=0.03, R2_random_mean=0.916, R2_random_std=0.000,
                                R2_grouped_mean=0.849, R2_grouped_std=0.002, inflacion_pct=45, inflacion_pct_std=1),
}


def agg(runs, grouped_scheme):
    """Agrega por (dataset, modelo) usando 'grouped_scheme' como brazo agrupado."""
    rows = []
    for (ds, mo), _ in runs.groupby(["dataset", "modelo"]):
        r = runs[(runs.dataset == ds) & (runs.modelo == mo) & (runs.scheme == "random_shuffle")] \
            .set_index("seed").sort_index()
        g = runs[(runs.dataset == ds) & (runs.modelo == mo) & (runs.scheme == grouped_scheme)] \
            .set_index("seed").sort_index()
        assert list(r.index) == SEEDS and list(g.index) == SEEDS, f"faltan semillas {ds}/{mo}"
        infl = 100.0 * (g.MAE.values - r.MAE.values) / r.MAE.values   # emparejada por semilla
        dr2 = g.R2.values - r.R2.values                                # signado, negativo = cae
        rows.append(dict(
            dataset=ds, modelo=mo,
            MAE_random_mean=float(r.MAE.mean()), MAE_random_std=float(r.MAE.std(ddof=0)),
            MAE_grouped_mean=float(g.MAE.mean()), MAE_grouped_std=float(g.MAE.std(ddof=0)),
            R2_random_mean=float(r.R2.mean()), R2_random_std=float(r.R2.std(ddof=0)),
            R2_grouped_mean=float(g.R2.mean()), R2_grouped_std=float(g.R2.std(ddof=0)),
            inflacion_pct_mean=float(np.mean(infl)), inflacion_pct_std=float(np.std(infl, ddof=0)),
            dR2_mean=float(np.mean(dr2)), dR2_std=float(np.std(dr2, ddof=0)),
            n_semillas=len(SEEDS),
            MAE_grouped_min=float(g.MAE.min()), MAE_grouped_max=float(g.MAE.max()),
            inflacion_pct_min=float(np.min(infl)), inflacion_pct_max=float(np.max(infl)),
            n=int(r.n.iloc[0]), n_familias=int(r.n_familias.iloc[0]),
        ))
    return pd.DataFrame(rows).sort_values(["dataset", "modelo"]).reset_index(drop=True)


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="resolve paths, check that the input CSV exists, exit")
    return ap.parse_args()


def main():
    args = _cli()
    src = OUT / "tabla2_runs_por_semilla.csv"
    if args.dry_run:
        print(f"ROOT={ROOT}\nOUT={OUT}\n  {'ok ' if src.exists() else 'MISSING'} {src}"); return
    runs = pd.read_csv(src)
    assert len(runs) == 60, f"esperaba 60 corridas, hay {len(runs)}"
    assert (runs.familias_compartidas_tr_te[runs.scheme.str.startswith("grouped")] == 0).all(), \
        "fuga de familias en un brazo agrupado"

    import sklearn
    proc_new = (f"run_tabla2_shuffle.py + agg_tabla2.py | brazo agrupado = "
                f"GroupKFold(5, shuffle=True, random_state=s), s=0..4 (sklearn {sklearn.__version__}); "
                f"brazo aleatorio = KFold(5, shuffle=True, random_state=s); "
                f"modelo = make_models(s) importado de code/tc_pipeline.py (A) y code/datasetB_pipeline.py (B); "
                f"grupos = familia quimica (conjunto de elementos) via pandas.factorize; "
                f"inflacion emparejada por semilla; dR2 = R2_grouped - R2_random (signado); std ddof=0")
    proc_ctrl = proc_new.replace("GroupKFold(5, shuffle=True, random_state=s), s=0..4",
                                 "GroupKFold(5) SIN shuffle (particion FIJA; s solo cambia el modelo) "
                                 "= protocolo PUBLICADO, control de reproduccion")

    t_new = agg(runs, "grouped_shuffle");    t_new["procedencia"] = proc_new
    t_ctl = agg(runs, "grouped_noshuffle");  t_ctl["procedencia"] = proc_ctrl

    cols = ["dataset", "modelo", "MAE_random_mean", "MAE_random_std", "MAE_grouped_mean",
            "MAE_grouped_std", "inflacion_pct_mean", "inflacion_pct_std", "dR2_mean", "dR2_std",
            "n_semillas", "procedencia"]
    extra = ["R2_random_mean", "R2_random_std", "R2_grouped_mean", "R2_grouped_std",
             "MAE_grouped_min", "MAE_grouped_max", "inflacion_pct_min", "inflacion_pct_max",
             "n", "n_familias"]
    t_new[cols + extra].to_csv(os.path.join(OUT, "tabla2_shuffle.csv"), index=False)
    t_ctl[cols + extra].to_csv(os.path.join(OUT, "tabla2_noshuffle_control.csv"), index=False)

    # ---- control de reproduccion contra la tabla publicada ----
    ctrl = []
    for _, row in t_ctl.iterrows():
        p = PUB[(row.dataset, row.modelo)]
        ctrl.append(dict(dataset=row.dataset, modelo=row.modelo,
                         MAE_grouped_pub=p["MAE_grouped_mean"], MAE_grouped_repro=round(row.MAE_grouped_mean, 2),
                         MAE_grouped_std_pub=p["MAE_grouped_std"], MAE_grouped_std_repro=round(row.MAE_grouped_std, 2),
                         MAE_random_pub=p["MAE_random_mean"], MAE_random_repro=round(row.MAE_random_mean, 2),
                         infl_pub=p["inflacion_pct"], infl_repro=round(row.inflacion_pct_mean, 1)))
    pd.DataFrame(ctrl).to_csv(os.path.join(OUT, "tabla2_control_reproduccion.csv"), index=False)

    # ---- factores de crecimiento de la desviacion ----
    rat = []
    for _, n in t_new.iterrows():
        c = t_ctl[(t_ctl.dataset == n.dataset) & (t_ctl.modelo == n.modelo)].iloc[0]
        p = PUB[(n.dataset, n.modelo)]
        rat.append(dict(dataset=n.dataset, modelo=n.modelo,
                        std_MAE_grouped_pub=p["MAE_grouped_std"],
                        std_MAE_grouped_noshuffle=c.MAE_grouped_std,
                        std_MAE_grouped_shuffle=n.MAE_grouped_std,
                        factor_vs_noshuffle=n.MAE_grouped_std / c.MAE_grouped_std,
                        factor_vs_pub=n.MAE_grouped_std / p["MAE_grouped_std"],
                        std_infl_pub=p["inflacion_pct_std"], std_infl_noshuffle=c.inflacion_pct_std,
                        std_infl_shuffle=n.inflacion_pct_std,
                        factor_infl_vs_noshuffle=n.inflacion_pct_std / c.inflacion_pct_std,
                        std_dR2_noshuffle=abs(c.dR2_std), std_dR2_shuffle=abs(n.dR2_std),
                        factor_dR2=abs(n.dR2_std) / abs(c.dR2_std)))
    pd.DataFrame(rat).to_csv(os.path.join(OUT, "tabla2_crecimiento_desviacion.csv"), index=False)

    # ---- rango global de inflacion sobre las 4 combinaciones ----
    per_seed_infl = []
    for _, row in runs[runs.scheme == "grouped_shuffle"].iterrows():
        r = runs[(runs.dataset == row.dataset) & (runs.modelo == row.modelo) &
                 (runs.scheme == "random_shuffle") & (runs.seed == row.seed)].iloc[0]
        per_seed_infl.append(dict(dataset=row.dataset, modelo=row.modelo, seed=int(row.seed),
                                 inflacion_pct=100.0 * (row.MAE - r.MAE) / r.MAE))
    ps = pd.DataFrame(per_seed_infl)
    ps.to_csv(os.path.join(OUT, "tabla2_inflacion_por_semilla.csv"), index=False)
    rango = dict(
        definicion=("rango de la inflacion de MAE sobre las 4 combinaciones dataset x modelo, "
                    "brazo agrupado con GroupKFold shuffle=True"),
        medias_por_combinacion={f"{r.dataset}/{r.modelo}": round(r.inflacion_pct_mean, 1)
                                for _, r in t_new.iterrows()},
        rango_de_medias_pct=[round(t_new.inflacion_pct_mean.min(), 1), round(t_new.inflacion_pct_mean.max(), 1)],
        rango_todas_las_corridas_pct=[round(ps.inflacion_pct.min(), 1), round(ps.inflacion_pct.max(), 1)],
        rango_publicado_medias_pct=[min(p["inflacion_pct"] for p in PUB.values()),
                                    max(p["inflacion_pct"] for p in PUB.values())],
        rango_control_noshuffle_pct=[round(t_ctl.inflacion_pct_mean.min(), 1),
                                     round(t_ctl.inflacion_pct_mean.max(), 1)],
        procedencia=proc_new)
    json.dump(rango, open(os.path.join(OUT, "tabla2_rango_inflacion.json"), "w"), indent=2)

    pd.set_option("display.width", 200)
    print(t_new[["dataset", "modelo", "MAE_random_mean", "MAE_random_std", "MAE_grouped_mean",
                 "MAE_grouped_std", "inflacion_pct_mean", "inflacion_pct_std", "dR2_mean", "dR2_std"]]
          .round(4).to_string(index=False))
    print()
    print(pd.DataFrame(ctrl).to_string(index=False))
    print()
    print(pd.DataFrame(rat).round(3).to_string(index=False))
    print()
    print(json.dumps(rango, indent=1))


if __name__ == "__main__":
    main()
