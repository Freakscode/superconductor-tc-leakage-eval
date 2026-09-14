"""temporal_corpus.py -- build the TEMPORAL CORPUS (experiment 7, v3) from the SuperCon/NIMS dump.

Input  (NOT in the repo; redistribution terms unverified)
    $SUPERCON_DATA_RAW  default <ROOT>/../data/primary.tsv   sep='\\t', skiprows=2,
                        header num,name,element,str3,utc,tc,journal
    `element` = exact formula (SuperCon nomenclature: every element carries a count);
    `name`    = common formula with substitution notation, NOT parseable -> never used.
Outputs (derived data only: no journal strings, no per-report dump)
    data/temporal_corpus_v3.csv   one row per normalised composition:
        comp_key (fractional composition rounded 1e-3, '|'-joined), formula (representative =
        `element` string of the earliest report, formula_v3 after the audit corrections),
        Tc_median, Tc_max, Tc_min, year_first, year_last, n_reportes, n_reportes_con_anio,
        familia (sorted element set), familia_code, year_rule_first (regex rule that produced
        year_first), anio_ambiguo (year_first from a fallback rule, see below)
    data/temporal_corpus_v3.npz   X (132 Magpie float32, recipe = datasetB_pipeline.featurize),
        y (= Tc_median), y_max, groups (= familia_code), year (= year_first), comp_key, formula,
        anio_ambiguo, feature_labels
    <RESULTS_V3>/temporal_provenance.json / .md   extraction rules, counts, exclusions, EDA audit,
        linkage with Dataset A / Dataset B v3

Year-extraction rules (applied to `journal`, in this order; the FIRST that fires wins)
    R1 paren      \\((19xx|20xx)\\)                    'Phys.Rev.B,55(1997)11832'  -> 1997
    R2 arxiv_yymm (cond-mat/|arXiv:)YYMM.nnnn        'cond-mat/1408.2625'         -> 2014
    R3 arxiv_old  cond-mat/YYMMnnn                    'cond-mat/0305503'           -> 2003
    R4 bare       standalone 19xx|20xx not adjacent to letters/digits (page numbers such as
                  'L2080' are therefore NOT years); several distinct candidates -> ambiguous
    none          'Unpublished results', 'Phys.Lett.A,38()323', typos '(19879)', '(2995)' ...
    Plausibility window [1911, 2021] (Onnes 1911; newest parenthesised year in the dump = 2021).
    A year outside the window is set to missing and counted as 'fuera_de_rango'.
    anio_ambiguo = year_first produced by R2-R4 (fallback rules) -- used for the sensitivity run.

Chemical exclusions = the SAME rules as code/v3/audit_B.py (H03), applied to `element`:
    no_parseada_pymatgen (incl. 'Oz', 'O4-Y', 'O7+d' = indeterminate oxygen), OY_oxigeno_indeterminado,
    O_final_sin_coeficiente, elemento_sin_coeficiente_multielemental, coeficiente_extremo (>150 atoms
    or coefficient >500), elemento_repetido_artefacto; the three documented CORRECTIONS are applied
    when the exact string occurs. Tc<=0 rows are dropped (superconductors only, as Dataset B).
Deduplication: normalised composition (fractions rounded to 1e-3) -> year_first = MIN year over the
    reports with a year (a composition with reports but none dated is dropped); Tc = MEDIAN over
    all its Tc>0 reports (Tc_max kept for the sensitivity run).

Usage:  python code/v3/temporal_corpus.py         (paths from this file; SUPERCON_ROOT,
        SUPERCON_RESULTS_V3, SUPERCON_DATA_RAW override)
"""
from __future__ import annotations
import hashlib, json, os, re, sys, time, warnings
from collections import Counter
from importlib.metadata import version as _v
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Composition

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RESULTS_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
RAW = Path(os.environ.get("SUPERCON_DATA_RAW", ROOT.parent / "data" / "primary.tsv"))
YEAR_MIN, YEAR_MAX = 1911, 2021
TOK = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
CORRECTIONS = {"Pr185Ce0.15Cu1O4": "Pr1.85Ce0.15Cu1O4",
               "La1.85Sr0.15Cu98Ni0.02O4": "La1.85Sr0.15Cu0.98Ni0.02O4",
               "Rb3C60C60": "Rb3C60"}


def procedencia(extra=None):
    p = dict(script="code/v3/temporal_corpus.py", python=sys.version.split()[0], numpy=np.__version__,
             pandas=pd.__version__, pymatgen=_v("pymatgen"), matminer=_v("matminer"),
             input="SUPERCON_DATA_RAW (SuperCon/NIMS primary.tsv, no redistribuido)",
             anio="primer (19xx|20xx) entre parentesis del campo journal; fallbacks arXiv YYMM y anio suelto; "
                  f"ventana [{YEAR_MIN},{YEAR_MAX}]; fecha de una composicion = MIN sobre sus reportes",
             composicion="pymatgen fractional_composition redondeada a 1e-3 sobre la columna element",
             tc="mediana de los reportes Tc>0 de la composicion (Tc_max para sensibilidad)",
             exclusiones="reglas de code/v3/audit_B.py (H03) aplicadas a element; Tc<=0 fuera",
             familia="conjunto de elementos (pymatgen, D/T->H, coef 0 omitido); codigo = pandas.factorize "
                     "de la clave ordenada sobre el corpus final",
             featurizacion="matminer ElementProperty.from_preset('magpie'), 132 columnas, receta de "
                           "code/datasetB_pipeline.py::featurize()")
    if extra: p.update(extra)
    return p


# ------------------------------------------------------------------------------ year rules
def extract_year(j):
    """-> (year|None, rule). Rules R1..R4 documented in the module docstring."""
    if not isinstance(j, str) or not j.strip():
        return None, "vacio"
    m = re.search(r"\(((?:19|20)\d{2})\)", j)
    if m:
        return int(m.group(1)), "paren"
    m = re.search(r"(?:cond-mat/|arxiv:)\s*(\d{2})(\d{2})\.\d{4}", j, flags=re.I)
    if m and 1 <= int(m.group(2)) <= 12:
        return 2000 + int(m.group(1)), "arxiv_yymm"
    m = re.search(r"cond-mat/(\d{2})(\d{2})\d{3}(?!\d)", j, flags=re.I)
    if m and 1 <= int(m.group(2)) <= 12:
        yy = int(m.group(1))
        return (1900 if yy >= 92 else 2000) + yy, "arxiv_old"
    cands = sorted(set(re.findall(r"(?<![A-Za-z0-9])((?:19|20)\d{2})(?![A-Za-z0-9])", j)))
    if len(cands) == 1:
        return int(cands[0]), "bare"
    if len(cands) > 1:
        return int(cands[0]), "bare_ambiguo"
    return None, "sin_anio"


# ------------------------------------------------------------------------------ chemistry
def classify(f):
    """audit_B.py rules on one `element` string -> (decision, categoria, formula_v3)."""
    f = f.strip()
    if f in CORRECTIONS:
        return "corregir", "correccion_documentada", CORRECTIONS[f]
    try:
        comp = Composition(f)
        if comp.num_atoms <= 0 or len(comp.elements) == 0:
            raise ValueError("vacia")
    except Exception:
        return "excluir", "no_parseada_pymatgen", ""
    toks = TOK.findall(f)
    cl = [s for s, n in toks if n == ""]
    cf = [(s, float(n)) for s, n in toks if n != ""]
    els = sorted(e.symbol for e in comp.elements)
    if f.endswith("OY"):
        return "excluir", "OY_oxigeno_indeterminado", ""
    if cl and len(els) > 1:
        if cl == ["O"] and f.endswith("O"):
            return "excluir", "O_final_sin_coeficiente", ""
        return "excluir", "elemento_sin_coeficiente_multielemental", ""
    mx = max([v for _, v in cf], default=0.0)
    if mx > 50 and (comp.num_atoms > 150 or mx > 500):
        return "excluir", "coeficiente_extremo_error_tipografico", ""
    if f == "D0.11Nb1D0.13Nb1":
        return "excluir", "elemento_repetido_artefacto", ""
    return "mantener", "ok", f


def comp_key(formula: str, nd: int = 3) -> str:
    fc = Composition(formula).fractional_composition.get_el_amt_dict()
    return "|".join(f"{k}{v:.{nd}f}" for k, v in sorted(fc.items()))


def family_key(formula: str) -> str:
    return "|".join(sorted(e.symbol for e in Composition(formula).elements))


# ------------------------------------------------------------------------------ EDA audit
_ZVAR = re.compile(r'([A-Z][a-z]?\d*(?:\.\d+)?)\s*[\+\-](?:z|Z|δ|x|y)')
def eda_clean_formula(s):                       # verbatim logic of notebooks/EDA.ipynb cell 20
    x = s.strip().replace(" ", "")
    x = re.sub(r'-[A-Za-z0-9]+$', '', x)
    return _ZVAR.sub(r'\1', x)

def eda_extract_year(j):                        # verbatim logic of notebooks/EDA.ipynb cell 21
    if j is None: return None
    m = re.search(r'\((\d{4})\)', j)
    if m:
        y = int(m.group(1)); return y if 1900 <= y <= 2100 else None
    m = re.search(r'\b(19|20)\d{2}\b', j)
    if m:
        y = int(m.group(0)); return y if 1900 <= y <= 2100 else None
    return None

def audit_eda(raw: pd.DataFrame) -> dict:
    """Reproduce the EDA's 'Vista A' construction (material_id = md5(reduced_formula|chemsys) over
    the cleaned `element` string, Tc median, year = min) to explain its N = 15 816."""
    f = raw.element.where(raw.element.str.len() > 0, raw.name).map(eda_clean_formula)
    tc = pd.to_numeric(raw.tc, errors="coerce")
    def rf(x):
        try: return Composition(x).reduced_formula
        except Exception: return None
    def cs(x):
        try: return Composition(x).chemical_system
        except Exception: return None
    red, chem = f.map(rf), f.map(cs)
    mid = [hashlib.md5(f"{r or ''}|{c or ''}".encode()).hexdigest()[:12] for r, c in zip(red, chem)]
    d = pd.DataFrame(dict(mid=mid, tc=tc, year=raw.journal.map(eda_extract_year), parsed=red.notna(),
                          cleaned_changed=(f != raw.element.str.strip().str.replace(" ", ""))))
    d = d[d.tc.notna()]
    g = d.groupby("mid").agg(tc=("tc", "median"), year=("year", "min"), n=("tc", "size"),
                             parsed=("parsed", "first"), changed=("cleaned_changed", "any"))
    with_year = g[g.year.notna()]
    out = dict(eda_rows_with_tc=int(len(d)), eda_materials=int(len(g)),
               eda_materials_with_year=int(len(with_year)),
               eda_materials_unparsed_bucket=int((~g.parsed).sum()),
               eda_unparsed_bucket_rows=int(g.loc[~g.parsed, "n"].sum()) if (~g.parsed).any() else 0,
               eda_materials_tc_le0=int((with_year.tc <= 0).sum()),
               eda_materials_formula_altered_by_cleaning=int(with_year.changed.sum()),
               eda_materials_tc_gt0_unaltered=int(((with_year.tc > 0) & (~with_year.changed) & with_year.parsed).sum()),
               eda_splits={str(c): dict(n_train_le=int((with_year.year <= c).sum()), n_test_gt=int((with_year.year > c).sum()))
                           for c in (1995, 2005, 2015)},
               eda_reported=dict(n_total=15816, splits={"1995": (8353, 7463), "2005": (12339, 3477), "2015": (15133, 683)}),
               nota="material_id = md5(reduced_formula|chemsys) sobre la formula 'limpiada' (sufijos -Y/+z eliminados = "
                    "estequiometria de O inventada); las formulas no parseables comparten UN material_id ('|'); "
                    "incluye Tc=0; Tc = mediana; year = min; train = year<=corte, test = year>corte; sin GroupKFold.")
    return out


# ------------------------------------------------------------------------------ main build
def build() -> dict:
    t0 = time.time()
    raw = pd.read_csv(RAW, sep="\t", skiprows=2, dtype=str, keep_default_na=False)
    assert list(raw.columns) == ["num", "name", "element", "str3", "utc", "tc", "journal"], raw.columns
    n_raw = len(raw)
    raw["tc_f"] = pd.to_numeric(raw.tc, errors="coerce")
    yr = raw.journal.map(extract_year)
    raw["year"] = yr.map(lambda t: t[0]).astype("float")
    raw["year_rule"] = yr.map(lambda t: t[1])
    out_of_range = raw.year.notna() & ((raw.year < YEAR_MIN) | (raw.year > YEAR_MAX))
    raw.loc[out_of_range, "year_rule"] = "fuera_de_rango"
    raw.loc[out_of_range, "year"] = np.nan
    counts_year = dict(Counter(raw.year_rule))
    unit_not_K = int((raw.utc.str.strip().str.upper() != "K").sum())

    cls = raw.element.map(classify)
    raw["decision"] = cls.map(lambda t: t[0]); raw["categoria"] = cls.map(lambda t: t[1])
    raw["formula_v3"] = cls.map(lambda t: t[2])
    tc_pos = raw.tc_f > 0
    counts_chem_tcpos = dict(Counter(raw.loc[tc_pos, "categoria"]))
    keep = tc_pos & (raw.decision != "excluir")
    k = raw[keep].copy()
    k["comp_key"] = k.formula_v3.map(comp_key)
    k["familia"] = k.formula_v3.map(family_key)
    n_reports_kept = int(len(k))
    n_reports_kept_with_year = int(k.year.notna().sum())

    # per-composition aggregation
    k = k.reset_index().rename(columns={"index": "raw_row"})
    k_year = k[k.year.notna()].sort_values(["year", "raw_row"])
    first = k_year.drop_duplicates("comp_key", keep="first").set_index("comp_key")
    agg = k.groupby("comp_key").agg(Tc_median=("tc_f", "median"), Tc_max=("tc_f", "max"), Tc_min=("tc_f", "min"),
                                    n_reportes=("tc_f", "size"), n_reportes_con_anio=("year", "count"),
                                    year_last=("year", "max"), familia=("familia", "first"))
    agg = agg.join(first[["formula_v3", "year", "year_rule"]].rename(
        columns={"formula_v3": "formula", "year": "year_first", "year_rule": "year_rule_first"}), how="inner")
    n_comp_total = int(k.comp_key.nunique())
    agg["anio_ambiguo"] = (agg.year_rule_first != "paren").astype(int)
    agg["year_first"] = agg.year_first.astype(int); agg["year_last"] = agg.year_last.astype(int)
    agg = agg.reset_index().sort_values(["year_first", "comp_key"]).reset_index(drop=True)
    agg["familia_code"] = pd.factorize(agg.familia)[0]
    # featurise (one call per composition; representative formula)
    from matminer.featurizers.composition import ElementProperty
    ep = ElementProperty.from_preset("magpie"); labels = ep.feature_labels()
    t1 = time.time()
    def feat(f):                                   # same try/except policy as datasetB_pipeline.featurize
        try:
            return ep.featurize(Composition(f))
        except Exception:
            return [np.nan] * len(labels)
    X = np.array([feat(f) for f in agg.formula], dtype=np.float32)
    t_feat = time.time() - t1
    finite = np.isfinite(X).all(axis=1)
    n_dropna = int((~finite).sum())
    dropped_formulas = agg.formula[~finite].tolist()[:10]
    agg = agg[finite].reset_index(drop=True); X = X[finite]
    agg["familia_code"] = pd.factorize(agg.familia)[0]        # contiguous after dropna
    multi = agg[agg.n_reportes_con_anio > 1]
    span = (multi.year_last - multi.year_first)

    # linkage with Dataset A and Dataset B v3
    link = {}
    try:
        zB = np.load(DATA / "datasetB_featurized_v3.npz", allow_pickle=True)
        kB = pd.Series([comp_key(f) for f in zB["formula"]])
        ymap = agg.set_index("comp_key").year_first
        link["B_v3_rows"] = int(len(kB)); link["B_v3_rows_with_year"] = int(kB.isin(ymap.index).sum())
        link["B_v3_unique_comp"] = int(kB.nunique()); link["B_v3_unique_comp_with_year"] = int(kB.drop_duplicates().isin(ymap.index).sum())
    except Exception as e:
        link["B_v3_error"] = str(e)[:100]
    try:
        um = pd.read_csv(DATA / "unique_m.csv")
        els = [c for c in um.columns if c not in ("critical_temp", "material")]
        def keyA(row):
            d = {e: float(v) for e, v in zip(els, row) if v > 0}
            s = sum(d.values())
            return "|".join(f"{e}{v/s:.3f}" for e, v in sorted(d.items()))
        kA = pd.Series([keyA(r) for r in um[els].values])
        link["A_rows"] = int(len(kA)); link["A_rows_with_year"] = int(kA.isin(ymap.index).sum())
        link["A_unique_comp"] = int(kA.nunique()); link["A_unique_comp_with_year"] = int(kA.drop_duplicates().isin(ymap.index).sum())
    except Exception as e:
        link["A_error"] = str(e)[:100]

    # write outputs
    DATA.mkdir(exist_ok=True); RESULTS_V3.mkdir(parents=True, exist_ok=True)
    prov = procedencia(dict(segundos_featurizacion=round(t_feat, 1), n_composiciones=int(len(agg))))
    cols = ["comp_key", "formula", "Tc_median", "Tc_max", "Tc_min", "year_first", "year_last", "n_reportes",
            "n_reportes_con_anio", "familia", "familia_code", "year_rule_first", "anio_ambiguo"]
    with open(DATA / "temporal_corpus_v3.csv", "w") as fh:      # read with pandas.read_csv(..., comment='#')
        fh.write("# procedencia: " + json.dumps(prov, ensure_ascii=False) + "\n")
        agg[cols].to_csv(fh, index=False)
    np.savez_compressed(DATA / "temporal_corpus_v3.npz", X=X, y=agg.Tc_median.values.astype(float),
                        y_max=agg.Tc_max.values.astype(float), groups=agg.familia_code.values.astype(int),
                        year=agg.year_first.values.astype(int), comp_key=agg.comp_key.values.astype(str),
                        formula=agg.formula.values.astype(str), anio_ambiguo=agg.anio_ambiguo.values.astype(int),
                        n_reportes=agg.n_reportes.values.astype(int),
                        feature_labels=np.array(labels), procedencia=json.dumps(prov, ensure_ascii=False))

    # per-cut composition of the future (composition level)
    cuts = {}
    for T in (1995, 2000, 2005, 2010, 2015):
        tr, te = agg[agg.year_first < T], agg[agg.year_first >= T]
        fam_tr = set(tr.familia)
        new_fam = ~te.familia.isin(fam_tr)
        cuts[str(T)] = dict(n_train=int(len(tr)), n_test=int(len(te)), fam_train=int(tr.familia.nunique()),
                            fam_test=int(te.familia.nunique()), frac_test_familia_nueva=round(float(new_fam.mean()), 4),
                            frac_test_familia_vista=round(float(1 - new_fam.mean()), 4),
                            fam_test_nuevas=int(te.familia[new_fam].nunique()))
    # report-level: how many later reports concern compositions already known before T
    for T in cuts:
        Tn = int(T)
        kk = k_year.join(agg.set_index("comp_key").year_first, on="comp_key", how="inner")
        later = kk[(kk.year >= Tn) & (kk.year_first < Tn)]
        cuts[T]["reportes_posteriores_de_composicion_vista"] = int(len(later))
        cuts[T]["reportes_posteriores_total"] = int((kk.year >= Tn).sum())

    summary = dict(procedencia=prov, n_raw=n_raw, unit_not_K_rows=unit_not_K,
                   tc_le0_rows=int((raw.tc_f <= 0).sum()), tc_nan_rows=int(raw.tc_f.isna().sum()),
                   year_rule_counts_all=counts_year,
                   year_rule_counts_tc_gt0=dict(Counter(raw.loc[tc_pos, "year_rule"])),
                   year_extractable_pct_all=round(100 * raw.year.notna().mean(), 2),
                   year_min=int(raw.year.min()), year_max=int(raw.year.max()),
                   out_of_range_examples=raw.loc[out_of_range, "journal"].drop_duplicates().head(5).tolist(),
                   chem_categoria_tc_gt0=counts_chem_tcpos,
                   n_reports_tc_gt0=int(tc_pos.sum()), n_reports_kept=n_reports_kept,
                   n_reports_kept_with_year=n_reports_kept_with_year,
                   n_compositions_kept=n_comp_total, n_compositions_with_year=int(len(first)),
                   n_compositions_dropna_magpie=n_dropna, dropna_magpie_examples=dropped_formulas, n_compositions_final=int(len(agg)),
                   n_families_final=int(agg.familia.nunique()),
                   n_compositions_multi_report=int((agg.n_reportes > 1).sum()),
                   n_compositions_multi_dated_report=int(len(multi)),
                   span_years_multi=dict(median=float(span.median()), p75=float(span.quantile(.75)),
                                         p90=float(span.quantile(.9)), frac_gt0=float((span > 0).mean())),
                   tc_median_vs_max=dict(frac_equal=float((agg.Tc_median == agg.Tc_max).mean()),
                                         mean_abs_diff_K=float((agg.Tc_max - agg.Tc_median).mean()),
                                         p90_abs_diff_K=float((agg.Tc_max - agg.Tc_median).quantile(.9))),
                   anio_ambiguo_compositions=int(agg.anio_ambiguo.sum()),
                   year_rule_first_counts=dict(Counter(agg.year_rule_first)),
                   cuts=cuts, linkage=link, eda_audit=audit_eda(raw),
                   segundos_total=round(time.time() - t0, 1))
    json.dump(summary, open(RESULTS_V3 / "temporal_provenance.json", "w"), indent=1, ensure_ascii=False)
    (RESULTS_V3 / "temporal_provenance.md").write_text(provenance_md(summary))
    return summary


def provenance_md(s: dict) -> str:
    y, c, e, l = s["year_rule_counts_all"], s["chem_categoria_tc_gt0"], s["eda_audit"], s["linkage"]
    pct = lambda a, b: f"{100*a/b:.1f} %"
    out = ["# Procedencia del corpus temporal v3 (experimento 7)", "",
           f"Procedencia: `{json.dumps(s['procedencia'], ensure_ascii=False)}`", "",
           "## 1. Origen y campo de fecha", "",
           f"- Dump SuperCon/NIMS `primary.tsv`: {s['n_raw']} filas (num, name, element, str3, utc, tc, journal). "
           f"Unidad distinta de K: {s['unit_not_K_rows']} filas; Tc<=0: {s['tc_le0_rows']}; Tc no numerica: {s['tc_nan_rows']}.",
           "- El unico campo con fecha es `journal` (cadena bibliografica libre). La fecha es la del REPORTE, no la del descubrimiento.",
           "- La columna `element` es la formula exacta; `name` (notacion de sustitucion) no se usa.", "",
           "## 2. Reglas de extraccion del anio (orden de aplicacion; gana la primera)", "",
           "| regla | patron | ejemplo | filas (todas) |", "|---|---|---|---:|",
           f"| R1 paren | `\\((19xx|20xx)\\)` | `Phys.Rev.B,55(1997)11832` | {y.get('paren',0)} |",
           f"| R2 arxiv_yymm | `cond-mat/YYMM.nnnn`, `arXiv:YYMM.nnnn` | `cond-mat/1408.2625` -> 2014 | {y.get('arxiv_yymm',0)} |",
           f"| R3 arxiv_old | `cond-mat/YYMMnnn` | `cond-mat/0305503` -> 2003 | {y.get('arxiv_old',0)} |",
           f"| R4 bare | 19xx/20xx aislado (no pegado a letras/digitos) | `Springer, 1981, p=288` | {y.get('bare',0)} |",
           f"| R4 ambiguo | >1 candidato distinto en R4 | -- | {y.get('bare_ambiguo',0)} |",
           f"| fuera_de_rango | anio fuera de [{YEAR_MIN},{YEAR_MAX}] | -- | {y.get('fuera_de_rango',0)} |",
           f"| sin_anio | ninguna regla | `Unpublished results`, `Phys.Lett.A,38()323`, `(19879)`, `(2995)` | {y.get('sin_anio',0)} |", "",
           f"- Cobertura: {s['year_extractable_pct_all']} % de las filas con anio; rango {s['year_min']}-{s['year_max']}.",
           "- Ambiguedades resueltas: numeros de pagina tipo `L2080` (JJAP Lett.) NO son anios (exigimos no adyacencia a letras/digitos); "
           "`cond-mat/0808.2026:PRB079212509` da 2008 por R2 y no 2026 por R4 (R2 precede a R4); `Science, 2000(1978)535` da 1978 (R1 precede a R4).",
           "- `anio_ambiguo` = composicion cuya fecha de primer reporte procede de R2-R4 (regla de respaldo): "
           f"{s['anio_ambiguo_compositions']} de {s['n_compositions_final']} composiciones; se usa en la sensibilidad.", "",
           "## 3. Exclusiones quimicas (mismas reglas que `code/v3/audit_B.py`, aplicadas a `element`, filas Tc>0)", "",
           "| categoria | filas Tc>0 | decision |", "|---|---:|---|"]
    dec = {"ok": "mantener", "correccion_documentada": "corregir (3 lecturas tipograficas documentadas)"}
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        out.append(f"| {k} | {v} | {dec.get(k, 'excluir')} |")
    out += ["", f"- Filas Tc>0: {s['n_reports_tc_gt0']}; conservadas: {s['n_reports_kept']} ({pct(s['n_reports_kept'], s['n_reports_tc_gt0'])}); "
            f"de ellas con anio: {s['n_reports_kept_with_year']}.",
            "- `no_parseada_pymatgen` incluye sobre todo oxigeno indeterminado (`Oz`, `O4-Y`, `O7+d`, `O6+x`): la auditoria H03 no inventa cantidades.",
            f"- Composiciones sin descriptores Magpie finitos (simbolos que no son elementos: `M`, `Ls`, `Gs`): {s['n_compositions_dropna_magpie']} "
            f"(ejemplos: {', '.join(s['dropna_magpie_examples'][:5])}).", "",
            "## 4. Deduplicacion por composicion", "",
            f"- Clave = composicion fraccional redondeada a 1e-3. Composiciones con algun reporte: {s['n_compositions_kept']}; con anio: "
            f"{s['n_compositions_with_year']}; corpus final: **{s['n_compositions_final']} composiciones, {s['n_families_final']} familias**.",
            f"- Fecha = PRIMER reporte (min). {s['n_compositions_multi_dated_report']} composiciones tienen >1 reporte fechado; entre ellas el rango "
            f"de anios tiene mediana {s['span_years_multi']['median']:.0f}, p75 {s['span_years_multi']['p75']:.0f}, p90 {s['span_years_multi']['p90']:.0f} "
            f"({100*s['span_years_multi']['frac_gt0']:.0f} % con rango > 0). Usar el primer reporte evita que una composicion antigua re-medida parezca nueva.",
            f"- Tc = MEDIANA de los reportes Tc>0 (Tc_max para sensibilidad): coincide con el maximo en {100*s['tc_median_vs_max']['frac_equal']:.1f} % de las "
            f"composiciones; diferencia media {s['tc_median_vs_max']['mean_abs_diff_K']:.2f} K.", "",
            "## 5. Composicion del futuro por corte (nivel composicion; train = anio_primer_reporte < T, test >= T)", "",
            "| T | n_train | n_test | familias train | familias test | % test en familia NUEVA | reportes posteriores de composicion ya vista / reportes posteriores |",
            "|---:|---:|---:|---:|---:|---:|---:|"]
    for T, d in s["cuts"].items():
        out.append(f"| {T} | {d['n_train']} | {d['n_test']} | {d['fam_train']} | {d['fam_test']} | {100*d['frac_test_familia_nueva']:.1f} | "
                   f"{d['reportes_posteriores_de_composicion_vista']} / {d['reportes_posteriores_total']} |")
    out += ["", "Tras deduplicar, toda composicion del test es nueva por construccion; la categoria 'composicion vista' solo existe al nivel de "
            "reporte (re-mediciones posteriores de composiciones conocidas) y se evalua aparte en `temporal_validation.py`.", "",
            "## 6. Enlace con los datasets del articulo", "",
            f"- Dataset B v3: {l.get('B_v3_rows_with_year')} de {l.get('B_v3_rows')} filas ({pct(l.get('B_v3_rows_with_year',0), l.get('B_v3_rows',1))}) y "
            f"{l.get('B_v3_unique_comp_with_year')} de {l.get('B_v3_unique_comp')} composiciones unicas reciben anio de primer reporte.",
            f"- Dataset A (unique_m.csv): {l.get('A_rows_with_year')} de {l.get('A_rows')} filas ({pct(l.get('A_rows_with_year',0), l.get('A_rows',1))}) y "
            f"{l.get('A_unique_comp_with_year')} de {l.get('A_unique_comp')} composiciones unicas. La cobertura menor de A se debe a las filas con O sin "
            "coeficiente (O=1 en unique_m.csv) y a la normalizacion de Hamidieh.", "",
            "## 7. Auditoria de la validacion temporal del EDA (`notebooks/EDA.ipynb`, `model_summary.json`)", "",
            f"- Reproducido EXACTAMENTE: N = {e['eda_materials_with_year']} materiales (reportado {e['eda_reported']['n_total']}); "
            + "; ".join(f"corte {T}: {d['n_train_le']}/{d['n_test_gt']} (reportado {e['eda_reported']['splits'][T][0]}/{e['eda_reported']['splits'][T][1]})"
                        for T, d in e["eda_splits"].items()) + ".",
            "- Diferencias con el corpus v3 que explican 15 816 vs " + str(s["n_compositions_final"]) + ":",
            f"  1. El EDA 'limpia' la formula borrando sufijos `-Y`, `+z`, `-d`... (`O7-Y` -> `O7`): inventa la estequiometria de oxigeno que la auditoria H03 "
            f"excluye; {e['eda_materials_formula_altered_by_cleaning']} de sus materiales proceden de formulas alteradas.",
            f"  2. Las formulas que siguen sin parsear comparten UN material_id (md5('|')): {e['eda_unparsed_bucket_rows']} filas colapsadas en 1 'material' "
            "con Tc mediana.",
            f"  3. Incluye Tc<=0 ({e['eda_materials_tc_le0']} materiales) y no aplica ninguna exclusion quimica (OY, sin coeficiente, >150 atomos).",
            "  4. La identidad de material es `reduced_formula|chemsys` (pymatgen), no la composicion a 1e-3; el anio es el minimo (igual que aqui); "
            "train = anio <= corte y test = anio > corte (aqui < T / >= T).",
            "  5. Descriptores propios (bag-of-elements + 8 estadisticos mendeleev), no los 132 Magpie; RF 500 arboles / XGB 600 sin los hiperparametros del articulo.",
            f"  Materiales del EDA con Tc>0, parseables y sin formula alterada: {e['eda_materials_tc_gt0_unaltered']} (orden de magnitud del corpus v3).",
            "- Sus cifras (RF: R2 0.71 / 0.56 / 0.56 en 1995 / 2005 / 2015) son por tanto de un corpus distinto, sin intervalos ni estratos, y no se reutilizan.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    s = build()
    print(json.dumps({k: v for k, v in s.items() if k not in ("procedencia", "eda_audit", "cuts", "linkage")},
                     indent=1, ensure_ascii=False))
    print(json.dumps(dict(cuts=s["cuts"], linkage=s["linkage"], eda_audit=s["eda_audit"]), indent=1, ensure_ascii=False))
