"""Sensitivity of Table 1 (Dataset A) to the nomenclature defect: 1 927 rows whose formula in
unique_m.csv ends in 'O' without a coefficient and whose element columns record O = 1
(results_v3/A_nomenclatura_check.json; recomputed here from unique_m.material).

The 81 Hamidieh descriptors of those rows are computed with that O = 1 and are NOT corrected (they are the
published descriptors); the rows are only EXCLUDED and Table 1 (family-mean null, XGBoost, RandomForest x
random / grouped) is recomputed on the remaining 19 336 rows in two ways:
  refold      : new KFold(5, shuffle=True, rs=42) and GroupKFold(5) on the subset (the honest protocol on the
                reduced corpus; the partition necessarily changes);
  same_folds  : the ORIGINAL fold labels of splits/datasetA_folds.csv restricted to the subset (train and
                test rows removed from each original fold) - isolates the effect of the rows from the effect of
                re-partitioning.
Family ids are unchanged (element set is unaffected by the missing O coefficient). Intervals of the
inflation: paired family bootstrap (2000, seed 0). Output: results_v3/tabla1_A_sin_O1.csv (+ .md).

    python code/v3/tabla1_A_sensitivity.py
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h07_common import (RES_V3, SEED, N_JOBS, VERS, load_dataset, family_mean_null, metrics, with_n_jobs,
                        FamilyStats, paired_family_bootstrap, ci)

PROC = (f"code/v3/tabla1_A_sensitivity.py | {VERS} | Dataset A minus the rows whose unique_m.material ends in 'O' without digit "
        f"(regex O$ after stripping) AND O == 1 in the element columns | models make_models({SEED}) n_jobs={N_JOBS}; FamilyMeanNull = "
        f"train-fold family mean | refold: KFold(5, shuffle, rs=42) / GroupKFold(5) on the subset; same_folds: splits/datasetA_folds.csv labels "
        f"restricted | inflation CI: paired family bootstrap 2000, default_rng(0), conditional on OOF")


def run(X, y, g, make_models, splits, models):
    preds = {}
    for model in models:
        yp = np.full(len(y), np.nan)
        for tr, te in splits:
            if model == "FamilyMeanNull":
                yp[te] = family_mean_null(y[tr], g[tr], g[te])
            else:
                m = with_n_jobs(make_models(SEED)[model]); m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te])
        assert not np.isnan(yp).any(); preds[model] = yp
    return preds


def main():
    X, y, g, make_models, meta = load_dataset("A")
    um = meta["unique_m"]
    mat = um["material"].astype(str).str.strip()
    ends_O = mat.str.endswith("O")            # last token is a bare 'O' (no coefficient after it)
    o1 = (um["O"] == 1)
    bad = (ends_O & o1).values
    chk = json.load(open(RES_V3 / "A_nomenclatura_check.json"))
    print(f"rows ending in O w/o coefficient: {int(ends_O.sum())}; of them O==1: {int(bad.sum())}; A_nomenclatura_check: {chk['filas_O_final_sin_coeficiente']}", flush=True)
    assert int(bad.sum()) == chk["de_ellas_con_O_igual_1_en_unique_m"] == 1927, int(bad.sum())
    keep = ~bad; Xs, ys, gs = X[keep], y[keep], g[keep]
    sp = pd.read_csv(meta["splits"], comment="#")
    models = ["FamilyMeanNull", "XGBoost", "RandomForest"]
    # full-data reference from oof_v3
    oof = pd.read_csv(RES_V3 / "oof_v3.csv.gz"); oof = oof[oof.dataset == "A"]
    ref = {(m, p): o.sort_values("row_index").y_pred.values for (m, p), o in oof.groupby(["model", "protocol"])}
    rows, tab = [], []
    variants = {
        "refold": {"random": list(KFold(5, shuffle=True, random_state=SEED).split(Xs)), "grouped": list(GroupKFold(5).split(Xs, ys, gs))},
        "same_folds": {p: [(np.where(sp[c].values[keep] != f)[0], np.where(sp[c].values[keep] == f)[0]) for f in range(5)]
                       for p, c in (("random", "random_fold_seed0"), ("grouped", "fold_deterministic"))},
    }
    # NOTE same_folds/random uses random_fold_seed0 because splits/ stores random seeds 0-4 only (rs=42 is not in the file);
    # the refold variant is the one comparable with Table 1 (rs=42).
    for variant, prot in variants.items():
        preds = {}
        for proto, splits in prot.items():
            for tr, te in splits: assert not (set(gs[tr]) & set(gs[te])) or proto == "random"
            t = time.time(); P = run(Xs, ys, gs, make_models, splits, models)
            for m, yp in P.items(): preds[f"{m}_{proto}"] = yp
            print(f"[{variant} {proto}] " + " ".join(f"{m}={metrics(ys, P[m])['MAE']:.3f}" for m in models) + f" ({time.time()-t:.0f}s)", flush=True)
        fs = FamilyStats(ys, gs, preds); one = np.ones(fs.k)
        B = paired_family_bootstrap(fs, preds.keys(), 2000, 0)
        for m in models:
            for proto in ("random", "grouped"):
                mm = metrics(ys, preds[f"{m}_{proto}"])
                full = metrics(y, ref[(m, proto)]) if (m, proto) in ref else {}
                # full-data model evaluated on the kept rows only (same predictions, subset of rows)
                sub = metrics(y[keep], ref[(m, proto)][keep]) if (m, proto) in ref else {}
                rows.append(dict(variant=variant, modelo=m, split=proto, N=len(ys), k=len(np.unique(gs)), **mm,
                                 MAE_full_Tabla1=full.get("MAE"), R2_full_Tabla1=full.get("R2"),
                                 MAE_full_model_on_kept_rows=sub.get("MAE"), R2_full_model_on_kept_rows=sub.get("R2"),
                                 MAE_full_model_on_excluded_rows=metrics(y[bad], ref[(m, proto)][bad])["MAE"] if (m, proto) in ref else None))
            infl = 100 * (B[f"MAE_{m}_grouped"] / B[f"MAE_{m}_random"] - 1)
            Pr, Pg = fs.stats(one, f"{m}_random"), fs.stats(one, f"{m}_grouped")
            fr = metrics(y, ref[(m, "random")]); fg = metrics(y, ref[(m, "grouped")])
            tab.append(dict(variant=variant, modelo=m, N=len(ys), MAE_random=Pr["MAE"], MAE_grouped=Pg["MAE"], inflacion_pct=100 * (Pg["MAE"] / Pr["MAE"] - 1),
                            inflacion_lo=ci(infl)[0], inflacion_hi=ci(infl)[1], R2_random=Pr["R2"], R2_grouped=Pg["R2"], dR2=Pr["R2"] - Pg["R2"],
                            MAE_random_full=fr["MAE"], MAE_grouped_full=fg["MAE"], inflacion_full_pct=100 * (fg["MAE"] / fr["MAE"] - 1),
                            R2_random_full=fr["R2"], R2_grouped_full=fg["R2"], dR2_full=fr["R2"] - fg["R2"]))
    D = pd.DataFrame(rows); D["procedencia"] = PROC; D.to_csv(RES_V3 / "tabla1_A_sin_O1.csv", index=False)
    T = pd.DataFrame(tab); T["procedencia"] = PROC; T.to_csv(RES_V3 / "tabla1_A_sin_O1_inflacion.csv", index=False)
    pd.set_option("display.width", 250)
    print(D.drop(columns="procedencia").to_string()); print(T.drop(columns="procedencia").to_string())
    L = ["# Sensibilidad de la Tabla 1 (A) a las 1 927 filas con O=1 espurio", "", f"Procedencia: `{PROC}`", "",
         f"Filas excluidas: {int(bad.sum())} ({100*bad.mean():.2f} %); N restante {len(ys)}, familias {len(np.unique(gs))} (de {len(np.unique(g))}). Los descriptores NO se corrigen; solo se excluyen filas.", "",
         "| Variante | Modelo | MAE random | MAE grouped | Inflación % [IC fam.] | R² random | R² grouped | ΔR² | Tabla 1 completa: MAE r / g, inflación %, ΔR² |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in T.iterrows():
        L.append(f"| {r.variant} | {r.modelo} | {r.MAE_random:.3f} | {r.MAE_grouped:.3f} | {r.inflacion_pct:.1f} [{r.inflacion_lo:.1f}, {r.inflacion_hi:.1f}] | {r.R2_random:.4f} | {r.R2_grouped:.4f} | {r.dR2:.4f} | {r.MAE_random_full:.3f} / {r.MAE_grouped_full:.3f}, {r.inflacion_full_pct:.1f}, {r.dR2_full:.4f} |")
    L += ["", "same_folds/random usa random_fold_seed0 de splits/ (rs=42 no está en el archivo); la variante comparable con la Tabla 1 (rs=42) es refold.", ""]
    (RES_V3 / "tabla1_A_sin_O1.md").write_text("\n".join(L) + "\n")
    print(f"wrote {RES_V3 / 'tabla1_A_sin_O1.csv'}, tabla1_A_sin_O1_inflacion.csv, .md")


if __name__ == "__main__":
    main()
