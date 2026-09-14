"""topk_negatives.py -- H05: measured negatives (Tc <= 0) re-enter the screening pool of Dataset B.

data/supercon_stanev.csv has 3 966 rows with Tc <= 0 that datasetB_pipeline.featurize() drops (Tc>0 mask).
This script
  1. audits those rows with the SAME generic exclusion rules as code/v3/audit_B.py (unparseable formula,
     trailing 'OY', trailing 'O' without coefficient, multi-element symbol without coefficient, >150 atoms /
     coefficient > 500 typo; documented CORRECTIONS applied), Magpie-featurises the survivors with the recipe
     of datasetB_pipeline.featurize() and flags (a) negatives whose family (element set) exists in B v3,
     (b) negatives whose normalised composition (1e-6) coincides with a B v3 superconductor (label conflict);
  2. builds B_neg = B v3 (12 336 positives, y>0) + negatives; assigns negatives to the v3 folds:
       grouped : same fold as their family (fold_deterministic of splits/datasetB_folds_v3.csv); NEW families
                 are spread over the 5 folds by a deterministic balanced rule (largest family first -> fold with
                 fewest negatives so far);
       random  : RandomState(42) permutation into 5 near-equal blocks;
  3. re-fits the paper's regressors (XGBoost, RandomForest, presence-XGBoost; make_models(42)) on the
     POSITIVE-ONLY training folds (exactly as in the paper; the positive OOF predictions are asserted equal to
     topk_oof_predictions.csv.gz) and ranks the extended test pool: how many negatives enter the top-K, and
     what is precision@K once the pool contains materials measured as non-superconducting;
  4. reports what the source allows: every negative is recorded as exactly Tc = 0; the file has no field for
     the lowest temperature explored, so "measured non-superconductor down to T_min", "not reported" and
     "censored" cannot be separated (documented, not inferred).

Outputs: <RESULTS_V3>/topk_with_negatives.csv, topk_negatives_audit.csv, topk_negatives_summary.json
Usage:   python code/v3/topk_negatives.py
"""
from __future__ import annotations
import json, re, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import topk_common as tc  # noqa: E402

SCRIPT = "code/v3/topk_negatives.py"
KS = list(tc.KS)
N_BOOT, BOOT_SEED = 2000, 0
MODELS = ("XGBoost", "RandomForest", "presence")


def procedencia() -> str:
    return (f"{SCRIPT} | {tc.versions()} | negatives = rows of data/supercon_stanev.csv with Tc <= 0, audited with the "
            f"generic rules of code/v3/audit_B.py, Magpie ElementProperty('magpie') as datasetB_pipeline.featurize(); "
            f"positives = data/datasetB_featurized_v3.npz; folds = splits/datasetB_folds_v3.csv fold_deterministic (grouped; "
            f"negatives follow their family, new families balanced deterministically) / KFold(5, shuffle, rs=42) on positives "
            f"+ RandomState(42) blocks for negatives (random); models make_models(42) of code/datasetB_pipeline.py fitted on "
            f"POSITIVES ONLY (n_jobs={tc.N_JOBS}); positive := y >= p90(y_train); expected counts fractional under ties; "
            f"CI = family bootstrap of the extended pool ({N_BOOT}, RandomState({BOOT_SEED})), conditional on fitted predictions")


def audit_negatives(sc: pd.DataFrame, audit_mod) -> pd.DataFrame:
    from pymatgen.core import Composition
    rows = []
    for i, (name, tc_) in enumerate(zip(sc["name"], sc["Tc"])):
        if tc_ > 0:
            continue
        comp, err = audit_mod.parse(name)
        cat, dec, f3 = "ok", "mantener", name
        if comp is None:
            cat, dec = "no_parseada_pymatgen", "excluir"
        else:
            cl = audit_mod.countless(name); cf = audit_mod.coefs(name)
            els = sorted(e.symbol for e in comp.elements); n_at = comp.num_atoms
            mx = max([v for _, v in cf], default=np.nan)
            if name.endswith("OY"):
                cat, dec = "OY_oxigeno_indeterminado", "excluir"
            elif cl and len(els) > 1:
                cat, dec = ("O_final_sin_coeficiente" if (cl == ["O"] and name.endswith("O")) else "elemento_sin_coeficiente_multielemental"), "excluir"
            elif cl:
                cat = "elemento_sin_coeficiente_monoelemental"
            if dec == "mantener" and mx > 50:
                if name in audit_mod.CORRECTIONS:
                    cat, dec, f3 = "coeficiente_extremo_error_tipografico", "corregir", audit_mod.CORRECTIONS[name][0]
                elif n_at > 150 or mx > 500:
                    cat, dec = "coeficiente_extremo_error_tipografico", "excluir"
                else:
                    cat = "coeficiente_alto_legitimo"
            if dec == "mantener" and audit_mod.repeated_symbols(name):
                cat = "elemento_repetido_notacion_por_sitio" if cat == "ok" else cat
        rows.append(dict(fila_origen=i, formula_original=name, Tc=float(tc_), categoria=cat, decision=dec, formula_v3=f3))
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty
    audit_mod = tc.load_mod("audit_B", str(tc.CODE / "v3" / "audit_B.py"))
    sc = pd.read_csv(tc.DATA / "supercon_stanev.csv")
    B = tc.load_B("v3"); yB, gB, XB = B["y"], B["groups"], B["X"]
    folds_pos = tc.protocol_folds(B)
    fam_key3 = B["folds"]["family_key"].values.astype(str)
    key2code = dict(zip(fam_key3, gB)); assert len(set(key2code.values())) == len(np.unique(gB))
    fam_fold = dict(zip(gB, folds_pos["grouped"]))                       # family -> grouped fold (whole family)
    oof = tc.read_csv(tc.OOF_CACHE); oof = oof[oof.dataset == "Bv3"]

    # ---------------------------------------------------------------- 1. audit + featurise
    neg = audit_negatives(sc, audit_mod)
    n_neg_src = int((sc.Tc <= 0).sum()); assert len(neg) == n_neg_src
    summary = dict(n_negativos_origen=n_neg_src, n_Tc_exactamente_0=int((sc.Tc == 0).sum()), n_Tc_negativo=int((sc.Tc < 0).sum()),
                   n_Tc_nan=int(sc.Tc.isna().sum()), columnas_origen=list(sc.columns),
                   exclusiones_por_categoria=neg[neg.decision == "excluir"].categoria.value_counts().to_dict(),
                   categorias_mantenidas=neg[neg.decision != "excluir"].categoria.value_counts().to_dict())
    keep = neg[neg.decision != "excluir"].copy()
    ep = ElementProperty.from_preset("magpie"); labels = ep.feature_labels()
    feats = [ep.featurize(Composition(f)) for f in keep.formula_v3]
    F = pd.DataFrame(feats, columns=labels)
    nan_mask = F.isna().any(axis=1).values
    summary["n_nan_magpie_excluidos"] = int(nan_mask.sum())
    keep = keep[~nan_mask].reset_index(drop=True); Xn = F[~nan_mask].values.astype(np.float64)
    z = np.load(tc.DATA / "datasetB_featurized_v3.npz", allow_pickle=True)
    assert list(z["feature_labels"].astype(str)) == list(labels), "Magpie label order differs from the v3 checkpoint"
    keep["family_key"] = ["|".join(sorted(e.symbol for e in Composition(f).elements)) for f in keep.formula_v3]
    keep["family_in_Bv3"] = keep.family_key.map(lambda k: k in key2code)
    # label conflicts: same normalised composition as a v3 superconductor
    pos_keys = {}
    for f, y_ in zip(B["formula"], yB):
        pos_keys.setdefault(audit_mod.norm_key(Composition(f), 6), []).append(float(y_))
    nk = [audit_mod.norm_key(Composition(f), 6) for f in keep.formula_v3]
    keep["conflicto_composicion_con_positivo"] = [k in pos_keys for k in nk]
    keep["Tc_positivo_conflicto_max"] = [max(pos_keys[k]) if k in pos_keys else np.nan for k in nk]
    dup_neg = pd.Series(nk).duplicated(keep=False).values
    keep["duplicado_entre_negativos"] = dup_neg
    summary.update(n_negativos_featurizados=int(len(keep)), n_familia_existente_en_Bv3=int(keep.family_in_Bv3.sum()),
                   n_familia_nueva=int((~keep.family_in_Bv3).sum()), n_familias_nuevas=int(keep.loc[~keep.family_in_Bv3, "family_key"].nunique()),
                   n_conflicto_composicion_con_positivo=int(keep.conflicto_composicion_con_positivo.sum()),
                   n_duplicados_entre_negativos=int(dup_neg.sum()))
    # family codes for negatives
    new_keys = sorted(keep.loc[~keep.family_in_Bv3, "family_key"].unique())
    k3 = int(gB.max()) + 1
    code_of = dict(key2code); code_of.update({k: k3 + j for j, k in enumerate(new_keys)})
    gN = keep.family_key.map(code_of).values.astype(np.int64)
    # ---------------------------------------------------------------- 2. fold assignment
    fold_neg = {}
    lab = np.full(len(keep), -1)
    for i, (g, ex) in enumerate(zip(gN, keep.family_in_Bv3)):
        if ex:
            lab[i] = fam_fold[g]
    load = np.bincount(lab[lab >= 0], minlength=5).astype(float)
    sizes = keep.loc[~keep.family_in_Bv3].groupby("family_key").size().sort_values(ascending=False)
    for k_, (key, sz) in enumerate(zip(sizes.index, sizes.values)):
        f = int(np.argmin(load)); load[f] += sz
        lab[(keep.family_key.values == key)] = f
    assert (lab >= 0).all(); fold_neg["grouped"] = lab
    perm = np.random.RandomState(tc.SEED).permutation(len(keep)); lab_r = np.empty(len(keep), dtype=int)
    for f, blk in enumerate(np.array_split(perm, 5)):
        lab_r[blk] = f
    fold_neg["random"] = lab_r
    keep["fold_grouped"] = fold_neg["grouped"]; keep["fold_random"] = fold_neg["random"]
    # under the grouped protocol a negative in an existing family is scored by the model that never saw that family
    summary["negativos_por_fold_grouped"] = np.bincount(fold_neg["grouped"], minlength=5).tolist()
    summary["negativos_por_fold_random"] = np.bincount(fold_neg["random"], minlength=5).tolist()

    # ---------------------------------------------------------------- 3. refit on positives only, score the extended pool
    Xp_pos, elems = tc.element_indicators(B["formula"])
    # presence matrix for negatives on the SAME element columns (unseen elements -> no column; count them)
    jdx = {e: j for j, e in enumerate(elems)}; Xp_neg = np.zeros((len(keep), len(elems))); unseen_el = set()
    for i, f in enumerate(keep.formula_v3):
        for e in Composition(f).elements:
            if e.symbol in jdx:
                Xp_neg[i, jdx[e.symbol]] = 1.0
            else:
                unseen_el.add(e.symbol)
    summary["elementos_de_negativos_ausentes_en_Bv3"] = sorted(unseen_el)
    rows = []; preds_neg = {}
    for proto in ("random", "grouped"):
        for model in MODELS:
            ref = oof[(oof.model == model) & (oof.protocol == proto)].sort_values("row_index")
            pn = np.full(len(keep), np.nan)
            for k in range(5):
                tr = np.where(folds_pos[proto] != k)[0]; te = np.where(folds_pos[proto] == k)[0]
                if model == "presence":
                    m = B["make_models"](tc.SEED)["XGBoost"]; Xtr, Xte, Xng = Xp_pos[tr], Xp_pos[te], Xp_neg[fold_neg[proto] == k]
                else:
                    m = B["make_models"](tc.SEED)[model]; Xtr, Xte, Xng = XB[tr], XB[te], Xn[fold_neg[proto] == k]
                yp_pos = tc.fit_predict(m, Xtr, yB[tr], Xte)
                assert np.allclose(yp_pos, ref.y_pred.values[te], atol=1e-6), f"positive OOF not reproduced: {model}/{proto}/fold{k}"
                pn[fold_neg[proto] == k] = m.predict(Xng)
            preds_neg[(model, proto)] = pn
            keep[f"pred_{model}_{proto}"] = pn
            print(f"[fit] {model}/{proto} done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- 4. metrics on the extended pool
    rng = np.random.RandomState(BOOT_SEED)
    k_all = k3 + len(new_keys)
    fam_counts = rng.multinomial(k_all, np.full(k_all, 1.0 / k_all), size=N_BOOT)
    for proto in ("random", "grouped"):
        for model in MODELS:
            ref = oof[(oof.model == model) & (oof.protocol == proto)].sort_values("row_index")
            boot = np.zeros((N_BOOT, len(KS), 2))     # precision_with_neg, n_neg_topK (fold means)
            fold_rows = []
            for k in range(5):
                te = np.where(folds_pos[proto] == k)[0]; ng = np.where(fold_neg[proto] == k)[0]
                thr = float(ref.threshold_p90_train.values[te][0])
                pred = np.concatenate([ref.y_pred.values[te], preds_neg[(model, proto)][ng]])
                y = np.concatenate([yB[te], np.zeros(len(ng))])
                is_neg = np.concatenate([np.zeros(len(te)), np.ones(len(ng))])
                is_conf = np.concatenate([np.zeros(len(te)), keep.conflicto_composicion_con_positivo.values[ng].astype(float)])
                fam = np.concatenate([gB[te], gN[ng]])
                hit = (y >= thr).astype(float) * (1 - is_neg)
                eh = tc.expected_hits(pred, hit, KS); en = tc.expected_hits(pred, is_neg, KS); ec = tc.expected_hits(pred, is_conf, KS)
                eh_pos_only = tc.expected_hits(ref.y_pred.values[te], (yB[te] >= thr).astype(float), KS)
                n_pool = len(te) + len(ng)
                for K in KS:
                    fold_rows.append(dict(dataset="Bv3+neg", model=model, protocol=proto, K=K, fold=k, n_pos_pool=len(te), n_neg_pool=len(ng),
                                          frac_neg_pool_pct=100 * len(ng) / n_pool, threshold_p90_train=thr, prevalence_hi_pool_pct=100 * hit.sum() / n_pool,
                                          precision_pos_only_pct=100 * eh_pos_only[K] / K, precision_with_neg_pct=100 * eh[K] / K,
                                          expected_neg_in_topK=en[K], expected_conflict_neg_in_topK=ec[K],
                                          expected_intermediate_in_topK=K - eh[K] - en[K],
                                          neg_pred_median=float(np.median(pred[len(te):])), pos_pred_median=float(np.median(pred[:len(te)])),
                                          neg_pred_ge_thr_pct=100 * float((pred[len(te):] >= thr).mean()),
                                          pos_hi_pred_ge_thr_pct=100 * float((pred[:len(te)][yB[te] >= thr] >= thr).mean()),
                                          neg_true_pred_mae=float(np.abs(pred[len(te):]).mean())))
                for b in range(N_BOOT):
                    w = fam_counts[b][fam]
                    ehb = tc.expected_hits(pred, hit, KS, w); enb = tc.expected_hits(pred, is_neg, KS, w)
                    tot = w.sum()
                    for j, K in enumerate(KS):
                        boot[b, j, 0] += ehb[K] / min(K, tot) / 5; boot[b, j, 1] += enb[K] / 5
            fr = pd.DataFrame(fold_rows)
            for j, K in enumerate(KS):
                m = fr[fr.K == K]
                rows.append(dict(dataset="Bv3+neg", model=model, protocol=proto, K=K, fold="mean_5folds", n_pos_pool=int(m.n_pos_pool.sum()), n_neg_pool=int(m.n_neg_pool.sum()),
                                 frac_neg_pool_pct=m.frac_neg_pool_pct.mean(), threshold_p90_train=np.nan, prevalence_hi_pool_pct=m.prevalence_hi_pool_pct.mean(),
                                 precision_pos_only_pct=m.precision_pos_only_pct.mean(), precision_with_neg_pct=m.precision_with_neg_pct.mean(),
                                 precision_with_neg_ci_fam_lo_pct=100 * np.percentile(boot[:, j, 0], 2.5), precision_with_neg_ci_fam_hi_pct=100 * np.percentile(boot[:, j, 0], 97.5),
                                 expected_neg_in_topK=m.expected_neg_in_topK.mean(), expected_neg_in_topK_ci_fam_lo=np.percentile(boot[:, j, 1], 2.5),
                                 expected_neg_in_topK_ci_fam_hi=np.percentile(boot[:, j, 1], 97.5), expected_neg_in_topK_sum_5folds=m.expected_neg_in_topK.sum(),
                                 expected_conflict_neg_in_topK=m.expected_conflict_neg_in_topK.mean(), expected_intermediate_in_topK=m.expected_intermediate_in_topK.mean(),
                                 neg_pred_median=m.neg_pred_median.mean(), pos_pred_median=m.pos_pred_median.mean(), neg_pred_ge_thr_pct=m.neg_pred_ge_thr_pct.mean(),
                                 pos_hi_pred_ge_thr_pct=m.pos_hi_pred_ge_thr_pct.mean(), neg_true_pred_mae=m.neg_true_pred_mae.mean()))
            rows.extend(fr.to_dict("records"))
            r100 = fr[fr.K == 100]
            print(f"[{proto}/{model}] P@100 pos-only {r100.precision_pos_only_pct.mean():.1f}% -> with negatives {r100.precision_with_neg_pct.mean():.1f}%; "
                  f"negatives in top-100 per fold {r100.expected_neg_in_topK.round(1).tolist()} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows); out["procedencia"] = procedencia()
    out.to_csv(tc.RES_V3 / "topk_with_negatives.csv", index=False)
    keep["procedencia"] = procedencia()
    neg_full = neg.merge(keep.drop(columns=["formula_original", "Tc", "categoria", "decision", "formula_v3"]), on="fila_origen", how="left")
    neg_full.to_csv(tc.RES_V3 / "topk_negatives_audit.csv", index=False)
    # which negatives enter the grouped XGBoost top-100 most often (deterministic tie-break: stable sort)
    top_examples = []
    for k in range(5):
        te = np.where(folds_pos["grouped"] == k)[0]; ng = np.where(fold_neg["grouped"] == k)[0]
        pred = np.concatenate([oof[(oof.model == "XGBoost") & (oof.protocol == "grouped")].sort_values("row_index").y_pred.values[te], preds_neg[("XGBoost", "grouped")][ng]])
        order = np.argsort(-pred, kind="stable")[:100]
        for o in order:
            if o >= len(te):
                r = keep.iloc[ng[o - len(te)]]
                top_examples.append(dict(fold=k, formula=r.formula_v3, family_key=r.family_key, pred_Tc=float(pred[o]), family_in_Bv3=bool(r.family_in_Bv3),
                                         conflicto=bool(r.conflicto_composicion_con_positivo), Tc_pos_conflicto=(None if np.isnan(r.Tc_positivo_conflicto_max) else float(r.Tc_positivo_conflicto_max))))
    summary["negativos_en_top100_XGBoost_grouped_ejemplos"] = sorted(top_examples, key=lambda d: -d["pred_Tc"])[:25]
    summary["n_negativos_en_top100_XGBoost_grouped_total_5folds_desempate_estable"] = len(top_examples)
    summary["interpretacion_de_las_etiquetas"] = (
        "Las 3 966 filas tienen Tc == 0 exactamente (ninguna negativa, ninguna NaN). El fichero de Stanev (columnas name, Tc) no "
        "registra la temperatura minima explorada ni distingue 'medido no superconductor hasta T_min', 'Tc no reportada' y "
        "'medicion censurada'; la literatura que reutiliza el mismo fichero describe estas ~4 000 entradas como 'sin Tc reportada "
        "o con ausencia de superconductividad reportada (Tc = 0 K)' (J. Chem. Inf. Model. 2024, doi:10.1021/acs.jcim.4c01137). Por tanto: "
        "'negativo' aqui = 'registrado como Tc = 0 en SuperCon', un negativo censurado a una temperatura desconocida; no puede "
        "afirmarse que sea no superconductor a toda temperatura. Los negativos cuya composicion normalizada coincide con un "
        "superconductor de B v3 (conflicto_composicion_con_positivo) son etiquetas contradictorias del propio corpus y se cuentan aparte.")
    summary["procedencia"] = procedencia(); summary["segundos"] = round(time.time() - t0)
    json.dump(summary, open(tc.RES_V3 / "topk_negatives_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("negativos_en_top100_XGBoost_grouped_ejemplos", "interpretacion_de_las_etiquetas", "procedencia")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
