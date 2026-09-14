"""overlap_AB.py -- how much of Dataset B is already in Dataset A (H04), and what happens off the overlap.

Part 1  intersection matrix B∩A (rows and families) for four definitions of "same":
          (i)   identical formula string            (B.formula == A.material)
          (ii)  normalised composition               (molar fractions rounded to 1e-3, sorted "El0.xxx|..." key)
          (iii) normalised composition AND |Tc_B - Tc_A| <= 0.1 K for at least one A row with that key
          (iv)  elemental system (chemical family = constituent-element set)
        computed for the v2 checkpoint (N=12440) and the v3 checkpoint (N=12336)  ->  <RESULTS_V3>/overlap_AB.csv
        (A has no publication metadata in data/, so the "same publication" intersection asked for in the
        review cannot be computed here.)

Part 2  complements:  B_out = rows of B whose family is NOT in A;  A_out = rows of A whose family is NOT in B.
        A and B have different descriptor sets (81 Hamidieh vs 132 Magpie), so the cross-evaluation is done in a
        COMMON Magpie space: unique_m.material is Magpie-featurised with the same matminer recipe as B.
          * XGBoost(datasetB_pipeline.make_models(42))  trained on ALL of B (Magpie)  -> evaluated on A_out, A_in
          * XGBoost(tc_pipeline.make_models(42))        trained on ALL of A (Magpie)  -> evaluated on B_out, B_in
        MAE / RMSE / R2 with 95 % percentile intervals from a bootstrap over FAMILIES (2000 resamples, seed 0):
        families are resampled with replacement and all their rows enter the metric.
        Reference rows: the same models evaluated in-domain by GroupKFold(5) on their own data (Magpie space),
        so the complement numbers can be compared with the paper's grouped protocol.
        -> <RESULTS_V3>/overlap_complement_eval.csv

Usage:  python code/v3/overlap_AB.py [--skip-eval] [--boot 2000]
Paths resolve from this file (ROOT = repo root); SUPERCON_ROOT / SUPERCON_RESULTS_V3 override.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, re, sys, time, warnings
from importlib.metadata import version as _v
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
CODE, DATA = ROOT / "code", ROOT / "data"
RESULTS_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
A_MAGPIE = DATA / "datasetA_magpie_v3.npz"


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def procedencia(extra=None):
    import sklearn, xgboost
    p = dict(script="code/v3/overlap_AB.py", python=sys.version.split()[0], numpy=np.__version__,
             pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
             matminer=_v("matminer"), pymatgen=_v("pymatgen"),
             A="data/unique_m.csv (21263 rows, material + element amounts + critical_temp)",
             B_v2="data/datasetB_featurized.npz + formulas re-derived from data/supercon_stanev.csv (parseable, Tc>0)",
             B_v3="data/datasetB_featurized_v3.npz (formula field)",
             familia="conjunto de elementos constituyentes ('|'.join(sorted(symbols)))",
             composicion_normalizada="fracciones molares (pymatgen Composition.fractional_composition) redondeadas a 1e-3")
    if extra: p.update(extra)
    return p


def trailing_O_no_count(f: str) -> bool:
    """SuperCon nomenclature violation: formula ends in 'O' with no coefficient (e.g. 'Y1Ba2Cu3O')."""
    t = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", f)
    return bool(t) and t[-1][0] == "O" and t[-1][1] == ""


def fam_key(comp):
    return "|".join(sorted(e.symbol for e in comp.elements))


def norm_key(comp, nd=3):
    return "|".join(f"{k}{v:.{nd}f}" for k, v in sorted(comp.fractional_composition.get_el_amt_dict().items()))


def load_B(version: str):
    from pymatgen.core import Composition
    if version == "v3":
        z = np.load(DATA / "datasetB_featurized_v3.npz", allow_pickle=True)
        formulas = z["formula"].astype(str); X, y, g = z["X"], z["y"], z["groups"]
    else:
        z = np.load(DATA / "datasetB_featurized.npz", allow_pickle=True); X, y, g = z["X"], z["y"], z["groups"]
        sc = pd.read_csv(DATA / "supercon_stanev.csv"); formulas = []
        for name, tc in zip(sc["name"], sc["Tc"]):
            try:
                Composition(name)
            except Exception:
                continue
            if tc > 0:
                formulas.append(name)
        formulas = np.array(formulas); assert len(formulas) == len(y)
    comps = [Composition(f) for f in formulas]
    df = pd.DataFrame(dict(formula=formulas, Tc=y, fam=[fam_key(c) for c in comps], nk=[norm_key(c) for c in comps]))
    assert np.array_equal(pd.factorize(df.fam)[0], g), "family codes != checkpoint groups"
    return df, X


def load_A():
    from pymatgen.core import Composition
    uniq = pd.read_csv(DATA / "unique_m.csv")
    comps = [Composition(m) for m in uniq.material]
    df = pd.DataFrame(dict(formula=uniq.material.values, Tc=uniq.critical_temp.values,
                           fam=[fam_key(c) for c in comps], nk=[norm_key(c) for c in comps]))
    return df, comps


# ------------------------------------------------------------------------------ part 1
def intersections(B: pd.DataFrame, A: pd.DataFrame, tag: str) -> list[dict]:
    setA_str, setA_nk, setA_fam = set(A.formula), set(A.nk), set(A.fam)
    tcA = A.groupby("nk").Tc.apply(lambda s: np.sort(s.values)).to_dict()
    m_str = B.formula.isin(setA_str).values
    m_nk = B.nk.isin(setA_nk).values
    m_tc = np.array([nk in tcA and np.min(np.abs(tcA[nk] - tc)) <= 0.1 + 1e-9 for nk, tc in zip(B.nk, B.Tc)])
    m_fam = B.fam.isin(setA_fam).values
    fams_B = B.fam.unique(); nfB = len(fams_B)
    fam_in_A = np.isin(fams_B, list(setA_fam))
    # family-level for the finer criteria: family "matched" if at least one of its rows matches
    rows = []
    for crit, m in [("(i) cadena de formula identica", m_str), ("(ii) composicion normalizada 1e-3", m_nk),
                    ("(iii) composicion normalizada + |dTc|<=0.1 K", m_tc), ("(iv) sistema elemental (familia)", m_fam)]:
        fam_any = B.assign(m=m).groupby("fam").m.any()
        fam_all = B.assign(m=m).groupby("fam").m.all()
        rows.append(dict(checkpoint_B=tag, criterio=crit, N_B=len(B), filas_B_en_A=int(m.sum()),
                         frac_filas_B_en_A=float(m.mean()), k_B=nfB,
                         familias_B_con_alguna_fila_en_A=int(fam_any.sum()), frac_familias_B_alguna=float(fam_any.mean()),
                         familias_B_con_todas_sus_filas_en_A=int(fam_all.sum()), frac_familias_B_todas=float(fam_all.mean())))
    # the reverse direction for the family criterion (needed for A_out)
    setB_fam = set(B.fam)
    a_in = A.fam.isin(setB_fam).values
    rows.append(dict(checkpoint_B=tag, criterio="(iv-inv) filas de A cuya familia esta en B", N_B=len(A),
                     filas_B_en_A=int(a_in.sum()), frac_filas_B_en_A=float(a_in.mean()), k_B=A.fam.nunique(),
                     familias_B_con_alguna_fila_en_A=int(A.fam[a_in].nunique()),
                     frac_familias_B_alguna=float(A.fam[a_in].nunique() / A.fam.nunique()),
                     familias_B_con_todas_sus_filas_en_A=int(A.fam[a_in].nunique()),
                     frac_familias_B_todas=float(A.fam[a_in].nunique() / A.fam.nunique())))
    return rows


# ------------------------------------------------------------------------------ part 2
def featurize_A(comps) -> np.ndarray:
    if A_MAGPIE.exists():
        z = np.load(A_MAGPIE); return z["X"]
    from matminer.featurizers.composition import ElementProperty
    ep = ElementProperty.from_preset("magpie"); t0 = time.time()
    X = np.array([ep.featurize(c) for c in comps], dtype=np.float64)
    assert np.isfinite(X).all(), "NaN in A Magpie features"
    np.savez_compressed(A_MAGPIE, X=X.astype(np.float32), feature_labels=np.array(ep.feature_labels(), dtype=str),
                        procedencia=json.dumps(procedencia(dict(featurize_seconds=round(time.time() - t0, 1)))))
    return X.astype(np.float32)


def metrics(yt, yp):
    return dict(MAE=float(mean_absolute_error(yt, yp)), RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
                R2=float(r2_score(yt, yp)) if len(yt) > 1 else float("nan"))


def boot_families(y, yp, fam, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(fam)
    idx_by_fam = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    out = {"MAE": [], "RMSE": [], "R2": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_fam[p] for p in pick])
        m = metrics(y[idx], yp[idx])
        for k in out: out[k].append(m[k])
    return {f"{k}_lo95": float(np.nanpercentile(v, 2.5)) for k, v in out.items()} | \
           {f"{k}_hi95": float(np.nanpercentile(v, 97.5)) for k, v in out.items()}


def evaluate(B, XB, A, XA, n_boot, seed=42) -> list[dict]:
    dbp = load_mod("datasetB_pipeline", str(CODE / "datasetB_pipeline.py"))
    tcp = load_mod("tc_pipeline", str(CODE / "tc_pipeline.py"))
    famA, famB = set(A.fam), set(B.fam)
    B_out = ~B.fam.isin(famA).values; A_out = ~A.fam.isin(famB).values
    rows = []

    def rec(train, test, subset, y, yp, fam, note):
        r = dict(modelo_entrenado_en=train, evaluado_en=test, subconjunto=subset, n_filas=int(len(y)),
                 n_familias=int(pd.Series(fam).nunique()), **metrics(y, yp), nota=note)
        if len(y) > 1:
            r.update(boot_families(y, yp, fam, n_boot))
        return r

    for tag, mk, Xtr, ytr, Xte, yte, famte, mask_out, tr_name, te_name in [
            ("B->A", dbp.make_models, XB, B.Tc.values, XA, A.Tc.values, A.fam.values, A_out, "B (Magpie, todas las filas)", "A"),
            ("A->B", tcp.make_models, XA, A.Tc.values, XB, B.Tc.values, B.fam.values, B_out, "A (Magpie, todas las filas)", "B")]:
        t0 = time.time(); m = mk(seed)["XGBoost"]; m.fit(Xtr, ytr); yp = m.predict(Xte)
        print(f"[{tag}] fit {time.time()-t0:.0f}s", flush=True)
        rows.append(rec(tr_name, te_name, f"{te_name}_out (familia NO en {tr_name[0]})", yte[mask_out], yp[mask_out], famte[mask_out],
                        "complemento sin solapamiento de familias; unica evaluacion realmente externa"))
        rows.append(rec(tr_name, te_name, f"{te_name}_in (familia SI en {tr_name[0]})", yte[~mask_out], yp[~mask_out], famte[~mask_out],
                        "la familia esta en el entrenamiento: NO es una evaluacion externa, referencia de contraste"))
        rows.append(rec(tr_name, te_name, f"{te_name} completo", yte, yp, famte, "mezcla in/out"))
    # in-domain grouped references in the same Magpie space
    for name, mk, X, df in [("B", dbp.make_models, XB, B), ("A", tcp.make_models, XA, A)]:
        g = pd.factorize(df.fam)[0]; yp = np.empty(len(df))
        for tr, te in GroupKFold(5).split(X, df.Tc.values, g):
            m = mk(seed)["XGBoost"]; m.fit(X[tr], df.Tc.values[tr]); yp[te] = m.predict(X[te])
        rows.append(rec(f"{name} (Magpie, GroupKFold(5) interno)", name, f"{name} out-of-fold agrupado", df.Tc.values, yp, df.fam.values,
                        "referencia: protocolo agrupado del articulo, en espacio Magpie"))
        # and restricted to the same families as the complement, for apples-to-apples
        mask = A_out if name == "A" else B_out
        rows.append(rec(f"{name} (Magpie, GroupKFold(5) interno)", name, f"{name}_out filas, predichas out-of-fold agrupado",
                        df.Tc.values[mask], yp[mask], df.fam.values[mask],
                        "mismas filas que el complemento, pero predichas por el modelo del propio dataset (GroupKFold)"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-eval", action="store_true"); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    RESULTS_V3.mkdir(parents=True, exist_ok=True)
    A, compsA = load_A()
    rows = []
    for ver in ("v2", "v3"):
        B, XB = load_B(ver); rows += intersections(B, A, ver)
    ov = pd.DataFrame(rows); ov["procedencia"] = json.dumps(procedencia())
    ov.to_csv(RESULTS_V3 / "overlap_AB.csv", index=False)
    print(ov.drop(columns=["procedencia"]).to_string())
    # A-side data-quality note: does A carry the same defective SuperCon formulas?
    a_oy = int(A.formula.str.endswith("OY").sum())
    a_countless = int(A.formula.apply(lambda f: any(n == "" for _, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", f))).sum())
    print(json.dumps(dict(A_formulas_OY=a_oy, A_formulas_con_elemento_sin_coeficiente=a_countless,
                          A_material_duplicado_cadena=int(A.formula.duplicated().sum())), indent=1))
    # Dataset-A side of H03: the same SuperCon nomenclature defect exists in unique_m.csv
    uniq = pd.read_csv(DATA / "unique_m.csv")
    mO = uniq.material.apply(trailing_O_no_count).values
    m_in = uniq.material.str.contains(r"IN\d").values   # 'IN' (indium mis-cased) parsed as I + N
    chkA = dict(procedencia=procedencia(), filas_A=int(len(uniq)),
                filas_O_final_sin_coeficiente=int(mO.sum()), frac=float(mO.mean()),
                de_ellas_con_O_igual_1_en_unique_m=int((uniq.O.values[mO] == 1).sum()),
                de_ellas_Tc_mayor_30K=int((uniq.critical_temp.values[mO] > 30).sum()),
                familias_afectadas=int(A.fam[mO].nunique()),
                ejemplos=uniq.material[mO].head(8).tolist(),
                filas_token_IN_parseado_como_I_N=int(m_in.sum()), ejemplos_IN=uniq.material[m_in].head(3).tolist(),
                nota="en A la estequiometria de O de estas filas es O=1 en las columnas elementales (y por tanto en los 81 "
                     "descriptores de Hamidieh); la familia (conjunto de elementos) no se ve afectada salvo en las filas 'IN'")
    json.dump(chkA, open(RESULTS_V3 / "A_nomenclatura_check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in chkA.items() if k != "procedencia"}, indent=1, ensure_ascii=False))
    B, XB = load_B("v3")
    famA, famB = set(A.fam), set(B.fam)
    bo, ao = ~B.fam.isin(famA).values, ~A.fam.isin(famB).values

    def stats(df, m, tag):
        fs = df.fam[m].value_counts()
        return {f"{tag}_filas": int(m.sum()), f"{tag}_familias": int(df.fam[m].nunique()),
                f"{tag}_frac_filas": float(m.mean()), f"{tag}_Tc_media": float(df.Tc[m].mean()),
                f"{tag}_Tc_sd": float(df.Tc[m].std()), f"{tag}_Tc_mediana": float(df.Tc[m].median()),
                f"{tag}_familias_de_1_fila": int((fs == 1).sum()), f"{tag}_familia_max": int(fs.max()),
                f"{tag}_filas_con_O_final_sin_coeficiente": int(df.formula[m].apply(trailing_O_no_count).sum())}
    sizes = dict(**stats(B, bo, "B_out"), **stats(A, ao, "A_out"),
                 B_Tc_media=float(B.Tc.mean()), B_Tc_sd=float(B.Tc.std()), A_Tc_media=float(A.Tc.mean()), A_Tc_sd=float(A.Tc.std()),
                 A_formulas_OY=a_oy, A_formulas_con_elemento_sin_coeficiente=a_countless,
                 A_formulas_con_O_final_sin_coeficiente=int(A.formula.apply(trailing_O_no_count).sum()),
                 A_filas_O_final_sin_coeficiente_con_O_igual_1_en_unique_m="ver results_v3/A_nomenclatura_check.json",
                 A_material_duplicado_cadena=int(A.formula.duplicated().sum()),
                 nota="A_out/B_out definidos sobre B v3; complementos por FAMILIA (sistema elemental) no presente en el otro dataset")
    json.dump(dict(procedencia=procedencia(), **sizes), open(RESULTS_V3 / "overlap_complementos_tamanos.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(sizes, indent=1, ensure_ascii=False))
    if a.skip_eval:
        return 0
    XA = featurize_A(compsA)
    ev = pd.DataFrame(evaluate(B, XB, A, XA, a.boot))
    ev["procedencia"] = json.dumps(procedencia(dict(bootstrap=f"{a.boot} remuestreos de familias, seed 0, percentiles 2.5/97.5",
                                                     modelos="XGBoost de datasetB_pipeline.make_models(42) (B) y tc_pipeline.make_models(42) (A)",
                                                     **sizes)))
    ev.to_csv(RESULTS_V3 / "overlap_complement_eval.csv", index=False)
    print(ev.drop(columns=["procedencia", "nota"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
