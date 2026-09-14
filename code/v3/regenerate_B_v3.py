"""regenerate_B_v3.py -- apply the semantic-audit decisions (H03) to Dataset B and measure the effect.

Stage 1 (default)   data/audit/semantic_audit_B.csv  ->  data/datasetB_featurized_v3.npz
                                                      +  splits/datasetB_folds_v3.csv
                                                      +  <RESULTS_V3>/regeneracion_B_v3.json
  * rows with decision_final_fila == "excluir" are dropped; "corregir" rows use formula_v3;
  * Magpie featurisation = the SAME recipe as code/datasetB_pipeline.py::featurize()
    (matminer ElementProperty.from_preset("magpie"), pymatgen Composition, dropna, Tc>0 mask
    BEFORE pandas.factorize of the sorted "|"-joined element key);
  * X is stored as float32 like the v2 checkpoint; extra fields: formula, row_origen (0-based row
    of data/supercon_stanev.csv), checkpoint_row_v2 (row of data/datasetB_featurized.npz).
  * splits: GroupKFold(5) deterministic, GroupKFold(5, shuffle=True, random_state=0..4),
    KFold(5, shuffle=True, random_state=0..4) -- same columns/header convention as
    splits/datasetB_folds.csv (code/v2/make_splits.py), plus checkpoint_row_v2.

Stage 2 (--tabla1)  Table-1 sensitivity, v2 checkpoint vs v3 checkpoint, same seeds:
    family-mean null, XGBoost, RandomForest (datasetB_pipeline.make_models(42)) x
    random KFold(5, shuffle=True, random_state=42) / GroupKFold(5)  ->  <RESULTS_V3>/tabla1_B_antes_despues.csv
    (the manuscript's null row used KFold random_state=0; that variant is reported too, flagged).

Usage:  python code/v3/regenerate_B_v3.py [--tabla1] [--only-tabla1]
Paths resolve from this file (ROOT = repo root); SUPERCON_ROOT / SUPERCON_RESULTS_V3 override.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time, warnings
from importlib.metadata import version as _v
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
CODE, DATA, SPLITS = ROOT / "code", ROOT / "data", ROOT / "splits"
RESULTS_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
SEEDS = [0, 1, 2, 3, 4]
NPZ_V2, NPZ_V3 = DATA / "datasetB_featurized.npz", DATA / "datasetB_featurized_v3.npz"
FOLDS_V2, FOLDS_V3 = SPLITS / "datasetB_folds.csv", SPLITS / "datasetB_folds_v3.csv"


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def procedencia(extra=None):
    import sklearn, xgboost
    p = dict(script="code/v3/regenerate_B_v3.py", python=sys.version.split()[0], numpy=np.__version__,
             pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
             matminer=_v("matminer"), pymatgen=_v("pymatgen"),
             familia="conjunto de elementos constituyentes (pymatgen Composition.elements, D/T->H, coef 0 omitido), "
                     "codigo = pandas.factorize de la clave '|'.join(sorted(symbols)) DESPUES del filtro Tc>0",
             featurizacion="matminer ElementProperty.from_preset('magpie'), 132 columnas, misma receta que "
                           "code/datasetB_pipeline.py::featurize()")
    if extra: p.update(extra)
    return p


def fold_labels(splitter, n, groups=None):
    lab = np.full(n, -1, dtype=int); X = np.zeros((n, 1))
    it = splitter.split(X, None, groups) if groups is not None else splitter.split(X)
    for k, (_, te) in enumerate(it):
        lab[te] = k
    assert (lab >= 0).all()
    return lab


# ------------------------------------------------------------------------------ stage 1
def regenerate() -> dict:
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty
    audit = pd.read_csv(DATA / "audit" / "semantic_audit_B.csv")
    dec = (audit[audit.checkpoint_row >= 0].drop_duplicates("checkpoint_row")
           .set_index("checkpoint_row")[["decision_final_fila", "formula_v3"]])
    sc = pd.read_csv(DATA / "supercon_stanev.csv")
    z2 = np.load(NPZ_V2, allow_pickle=True); X2, y2, g2 = z2["X"], z2["y"], z2["groups"]
    ep = ElementProperty.from_preset("magpie"); labels = ep.feature_labels()

    # walk the source in order, reproduce v2 alignment (parseable & Tc>0), apply decisions
    ck = -1; rows = []
    for i, (name, tc) in enumerate(zip(sc["name"], sc["Tc"])):
        try:
            Composition(name)
        except Exception:
            continue
        if not tc > 0:
            continue
        ck += 1
        d, f3 = ("mantener", name) if ck not in dec.index else (dec.loc[ck, "decision_final_fila"], dec.loc[ck, "formula_v3"])
        if not isinstance(f3, str) or not f3:
            f3 = name
        rows.append(dict(checkpoint_row_v2=ck, row_origen=i, formula_v2=name, formula_v3=f3, Tc=tc, decision=d))
    df = pd.DataFrame(rows)
    assert len(df) == len(y2) == 12440 and np.array_equal(df.Tc.values, y2), "v2 alignment broken"
    key2 = np.array(["|".join(sorted(e.symbol for e in Composition(f).elements)) for f in df.formula_v2])
    assert np.array_equal(pd.factorize(key2)[0], g2), "v2 family codes not reproduced"

    keep = df[df.decision != "excluir"].reset_index(drop=True)
    t0 = time.time(); feats = []
    for f in keep.formula_v3:
        feats.append(ep.featurize(Composition(f)))
    B = pd.DataFrame(feats, columns=labels)
    nan_rows = int(B.isna().any(axis=1).sum())
    assert nan_rows == 0, f"{nan_rows} rows with NaN Magpie features"
    key3 = np.array(["|".join(sorted(e.symbol for e in Composition(f).elements)) for f in keep.formula_v3])
    groups3 = pd.factorize(key3)[0]
    X3 = B[labels].values.astype(np.float32); y3 = keep.Tc.values.astype(np.float64)

    # consistency: unchanged rows must reproduce the v2 features (float32 tolerance)
    unchanged = (keep.decision == "mantener").values
    diff = np.abs(X3[unchanged].astype(np.float64) - X2[keep.checkpoint_row_v2.values[unchanged]].astype(np.float64))
    rel = diff / (np.abs(X2[keep.checkpoint_row_v2.values[unchanged]].astype(np.float64)) + 1e-6)
    same_feats = bool((rel.max(axis=1) < 1e-4).all())

    np.savez_compressed(NPZ_V3, X=X3, y=y3, groups=groups3.astype(np.int64),
                        formula=keep.formula_v3.values.astype(str), row_origen=keep.row_origen.values.astype(np.int64),
                        checkpoint_row_v2=keep.checkpoint_row_v2.values.astype(np.int64),
                        feature_labels=np.array(labels, dtype=str))

    # ---- family bookkeeping
    fam2 = pd.Series(key2); fam3 = pd.Series(key3)
    fams2, fams3 = set(fam2), set(fam3)
    key2_kept = key2[keep.checkpoint_row_v2.values]
    rows_changed_family = int((key2_kept != key3).sum())
    fam_disappeared = sorted(fams2 - fams3)
    fam_new = sorted(fams3 - fams2)
    excluded = df[df.decision == "excluir"]
    excl_keys = key2[excluded.checkpoint_row_v2.values]
    # families that existed ONLY because of excluded rows (spurious-Y families among them)
    only_excluded = [k for k in fam_disappeared]
    spurious_Y = [k for k in only_excluded if "Y" in k.split("|")]
    # counterfactual: OY rows without real Y -> family with Y removed; does it exist elsewhere in v3?
    oy = excluded[excluded.formula_v2.str.endswith("OY")]
    cf = []
    for f, k in zip(oy.formula_v2, key2[oy.checkpoint_row_v2.values]):
        import re
        real_y = bool(re.search(r"Y\d", f[:-2]))
        k_true = k if real_y else "|".join(e for e in k.split("|") if e != "Y")
        cf.append(dict(formula=f, familia_v2=k, familia_quimica_correcta=k_true, itrio_real=real_y,
                       familia_correcta_existe_en_v3=k_true in fams3))
    cf = pd.DataFrame(cf)

    # ---- splits v3 (same convention as code/v2/make_splits.py)
    n = len(keep)
    out = pd.DataFrame({"row_index": np.arange(n), "family_id": groups3, "family_key": key3,
                        "fold_deterministic": fold_labels(GroupKFold(5), n, groups3)})
    for s in SEEDS:
        out[f"fold_seed{s}"] = fold_labels(GroupKFold(5, shuffle=True, random_state=s), n, groups3)
    for s in SEEDS:
        out[f"random_fold_seed{s}"] = fold_labels(KFold(5, shuffle=True, random_state=s), n)
    out["checkpoint_row_v2"] = keep.checkpoint_row_v2.values
    for c in ["fold_deterministic"] + [f"fold_seed{s}" for s in SEEDS]:
        assert out.groupby("family_id")[c].nunique().max() == 1, c
    import sklearn
    hdr = (f"# dataset B v3: {n} rows, {len(fams3)} chemical families. data/datasetB_featurized_v3.npz rows "
           f"(Tc>0 subset of data/supercon_stanev.csv minus the {len(excluded)} rows excluded by data/audit/semantic_audit_B.csv, "
           f"{int((keep.decision == 'corregir').sum())} formulas corrected); family = `groups` field.\n"
           f"# fold_deterministic = GroupKFold(5) (no shuffle, published protocol); fold_seedS = GroupKFold(5, shuffle=True,\n"
           f"#   random_state=S); random_fold_seedS = KFold(5, shuffle=True, random_state=S). scikit-learn {sklearn.__version__},\n"
           f"#   pandas {pd.__version__}, numpy {np.__version__}. Generated by code/v3/regenerate_B_v3.py.\n"
           f"# GroupKFold breaks size ties by the ORDER of the integer family codes (pandas.factorize order of first\n"
           f"#   appearance), so the fold assignment depends on the code order, not only on the family partition.\n"
           f"#   These indices, not just the procedure, are the reference. checkpoint_row_v2 maps each row to\n"
           f"#   data/datasetB_featurized.npz / splits/datasetB_folds.csv. Read with pandas.read_csv(..., comment='#').\n")
    SPLITS.mkdir(parents=True, exist_ok=True)
    with open(FOLDS_V3, "w") as f:
        f.write(hdr); out.to_csv(f, index=False)

    # fold overlap between v2 and v3 deterministic partitions (on the common rows)
    f2 = pd.read_csv(FOLDS_V2, comment="#")
    common_v2_fold = f2.set_index("row_index").loc[keep.checkpoint_row_v2.values, "fold_deterministic"].values
    rows_change_fold = int((common_v2_fold != out.fold_deterministic.values).sum())

    rep = dict(procedencia=procedencia(dict(featurize_seconds=round(time.time() - t0, 1))),
               N_v2=int(len(df)), N_v3=int(n), filas_excluidas=int(len(excluded)),
               filas_corregidas=int((keep.decision == "corregir").sum()),
               familias_v2=len(fams2), familias_v3=len(fams3),
               filas_retenidas_que_cambian_de_familia=rows_changed_family,
               familias_que_desaparecen=len(fam_disappeared), familias_nuevas=len(fam_new),
               familias_fusionadas="0 (ninguna fila retenida cambia de clave; la exclusion no fusiona familias)",
               familias_desaparecidas_con_Y=len(spurious_Y),
               familias_desaparecidas_lista=fam_disappeared,
               OY_filas=int(len(oy)), OY_con_itrio_real=int(cf.itrio_real.sum()),
               OY_familia_v2_erronea=int((~cf.itrio_real).sum()),
               OY_familia_correcta_existe_en_v3=int(cf.familia_correcta_existe_en_v3.sum()),
               OY_familias_v2_erroneas_distintas=int(cf.loc[~cf.itrio_real, "familia_v2"].nunique()),
               OY_familias_v2_erroneas_exclusivas=int(sum(1 for k in set(cf.loc[~cf.itrio_real, "familia_v2"]) if k in fam_disappeared)),
               features_filas_no_modificadas_reproducen_v2=same_feats,
               filas_comunes_que_cambian_de_fold_deterministico_v2_vs_v3=rows_change_fold,
               filas_excluidas_por_familia_size=excluded.assign(k=excl_keys).groupby("k").size().describe().to_dict())
    RESULTS_V3.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(RESULTS_V3 / "regeneracion_B_v3.json", "w"), indent=1, ensure_ascii=False)
    cf.to_csv(RESULTS_V3 / "OY_familias_contrafactual.csv", index=False)
    return rep


# ------------------------------------------------------------------------------ stage 2
def family_mean_null(y, groups, cv, split_groups=None):
    yp = np.empty_like(y, dtype=float)
    it = cv.split(np.zeros((len(y), 1)), None, split_groups) if split_groups is not None else cv.split(np.zeros((len(y), 1)))
    for tr, te in it:
        gm = y[tr].mean(); fm = pd.Series(y[tr]).groupby(groups[tr]).mean().to_dict()
        yp[te] = [fm.get(g, gm) for g in groups[te]]
    return yp


def metrics(yt, yp):
    return dict(MAE=float(mean_absolute_error(yt, yp)), RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
                R2=float(r2_score(yt, yp)))


def tabla1(seed=42) -> pd.DataFrame:
    dbp = load_mod("datasetB_pipeline", str(CODE / "datasetB_pipeline.py"))
    recs = []
    for ver, path in [("v2", NPZ_V2), ("v3", NPZ_V3)]:
        z = np.load(path, allow_pickle=True); X, y, g = z["X"], z["y"], z["groups"]
        schemes = [("random", KFold(5, shuffle=True, random_state=seed), None), ("grouped", GroupKFold(5), g)]
        for scheme, cv, grp in schemes:
            yp = family_mean_null(y, g, cv, grp)
            recs.append(dict(checkpoint=ver, N=len(y), k=int(len(np.unique(g))), modelo="Family-mean null",
                             split=scheme, random_state=seed, **metrics(y, yp)))
        yp = family_mean_null(y, g, KFold(5, shuffle=True, random_state=0))
        recs.append(dict(checkpoint=ver, N=len(y), k=int(len(np.unique(g))), modelo="Family-mean null (KFold rs=0, como el manuscrito)",
                         split="random", random_state=0, **metrics(y, yp)))
        for name in ("XGBoost", "RandomForest"):
            for scheme, cv, grp in schemes:
                t0 = time.time()
                yp = dbp.cv_predict_manual(lambda n=name: dbp.make_models(seed)[n], X, y, cv, grp)
                recs.append(dict(checkpoint=ver, N=len(y), k=int(len(np.unique(g))), modelo=name, split=scheme,
                                 random_state=seed, **metrics(y, yp), fit_seconds=round(time.time() - t0, 1)))
                print(f"[{ver}] {name}/{scheme}: {recs[-1]['MAE']:.3f} {recs[-1]['R2']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(recs)
    # inflation = grouped MAE / random MAE (and R2 drop) per model & checkpoint
    infl = []
    for (ver, m), g_ in df[df.random_state == seed].groupby(["checkpoint", "modelo"]):
        r = g_[g_.split == "random"].iloc[0]; q = g_[g_.split == "grouped"].iloc[0]
        infl.append(dict(checkpoint=ver, modelo=m, MAE_random=r.MAE, MAE_grouped=q.MAE,
                         inflacion_MAE_ratio=q.MAE / r.MAE, inflacion_MAE_abs=q.MAE - r.MAE,
                         R2_random=r.R2, R2_grouped=q.R2, caida_R2=r.R2 - q.R2))
    infl = pd.DataFrame(infl)
    df["procedencia"] = json.dumps(procedencia(dict(models="datasetB_pipeline.make_models(42)",
                                                    cv="KFold(5, shuffle=True, random_state=42) / GroupKFold(5)")))
    df.to_csv(RESULTS_V3 / "tabla1_B_antes_despues.csv", index=False)
    infl["procedencia"] = df["procedencia"].iloc[0]
    infl.to_csv(RESULTS_V3 / "tabla1_B_inflacion_antes_despues.csv", index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabla1", action="store_true", help="also run the Table-1 sensitivity (slow, ~15-25 min single-thread)")
    ap.add_argument("--only-tabla1", action="store_true")
    a = ap.parse_args()
    if not a.only_tabla1:
        rep = regenerate()
        print(json.dumps({k: v for k, v in rep.items() if k not in ("familias_desaparecidas_lista", "procedencia")}, indent=1, ensure_ascii=False))
    if a.tabla1 or a.only_tabla1:
        df = tabla1()
        print(df.drop(columns=["procedencia"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
