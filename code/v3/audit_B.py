"""audit_B.py -- semantic (chemical) audit of the Dataset-B source formulas (H03).

Input   data/supercon_stanev.csv   (Stanev et al. SuperCon list; columns name, Tc)
Output  data/audit/semantic_audit_B.csv        one line per (row, category); a row may carry
                                               several categories; `decision` is the FINAL
                                               decision for the row (excluir > corregir > mantener)
        <RESULTS_V3>/semantic_audit_B_summary.md   counts per category and per decision
        <RESULTS_V3>/semantic_audit_B_decisions.json   row-level decision map used by
                                               regenerate_B_v3.py (checkpoint_row -> decision,
                                               formula_v3)

Row identifiers
  fila_origen     0-based row of data/supercon_stanev.csv (header excluded)
  checkpoint_row  0-based row of data/datasetB_featurized.npz = position among the rows that
                  pymatgen parses AND have Tc>0, in source order (v2 alignment verified by
                  results_v2/verificacion_alineacion_B.json); -1 for the 8 unparsed rows.

Chemical basis of the rules (see `fuente` column)
  * SuperCon nomenclature: every element carries an explicit count, even 1 (3DSC, Sommer et al.
    2023, Sci. Data 10:816, p. 3). A symbol without count is therefore a defective formula; the
    canonical case is the trailing "OY" = oxygen of unknown quantity, NOT yttrium (same page).
  * We never invent an oxygen (or any other) quantity. Defective formulas whose stoichiometry is
    indeterminate are EXCLUDED; formulas are CORRECTED only when a single typographical reading
    is fixed by charge/site balance and matches the documented compound.
  * pymatgen maps the isotopes D/T to H and drops zero-amount elements; Magpie descriptors are
    then identical to the hydride / to the formula without the absent element (verified live).

Usage:  python code/v3/audit_B.py            (paths resolved from this file; SUPERCON_ROOT overrides)
"""
from __future__ import annotations
import json, os, re, sys, warnings
from collections import Counter
from importlib.metadata import version as _v
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Composition

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
AUDIT_DIR = DATA / "audit"
RESULTS_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
SRC_3DSC = ("3DSC, Sommer et al. 2023, Sci. Data 10:816, doi:10.1038/s41597-023-02721-y, p. 3: "
            "'in the formula Bi4.4Sr3.6Ca2Cu4OY the Y was considered to represent yttrium, but it "
            "actually represented an unknown quantity of oxygen (the SuperCon database strictly "
            "follows a nomenclature, where each element has a count, even if the count is 1)'")
TOK = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

PROCEDENCIA = dict(script="code/v3/audit_B.py", python=sys.version.split()[0],
                   pandas=pd.__version__, numpy=np.__version__, pymatgen=_v("pymatgen"),
                   matminer=_v("matminer"), input="data/supercon_stanev.csv",
                   definicion_fila="fila_origen = fila 0-based de supercon_stanev.csv; checkpoint_row = "
                                   "posicion entre filas parseables con Tc>0 (orden de origen)")


def parse(name):
    try:
        return Composition(name), ""
    except Exception as e:
        return None, str(e)[:100]


def countless(f):
    return [s for s, n in TOK.findall(f) if n == ""]


def coefs(f):
    return [(s, float(n)) for s, n in TOK.findall(f) if n != ""]


def repeated_symbols(f):
    c = Counter(s for s, _ in TOK.findall(f))
    return sorted(s for s, k in c.items() if k > 1)


def norm_key(comp: Composition, nd: int) -> str:
    fc = comp.fractional_composition.get_el_amt_dict()
    return "|".join(f"{k}{v:.{nd}f}" for k, v in sorted(fc.items()))


# ------------------------------------------------------------------ documented corrections
# Only typographical readings fixed by site balance AND matching a documented compound/series
# already present in the same source. Family (element set) is unchanged in every case.
CORRECTIONS = {
    "Pr185Ce0.15Cu1O4": ("Pr1.85Ce0.15Cu1O4",
        "Pr185 es Pr1.85 con el punto perdido: Pr(2-x)Ce(x)CuO4 con x=0.15 (1.85+0.15=2, sitio de tierra "
        "rara completo); el propio origen contiene Pr1.85Ce0.15Cu1O4.065/4.161/4.175 con Tc 16-19 K, "
        "coherentes con Tc=20 K",
        "Takagi, Uchida & Tokura, Phys. Rev. Lett. 62, 1197 (1989): Pr2-xCexCuO4, x~0.15, Tc~20 K"),
    "La1.85Sr0.15Cu98Ni0.02O4": ("La1.85Sr0.15Cu0.98Ni0.02O4",
        "Cu98 es Cu0.98 con el punto perdido: La1.85Sr0.15Cu(1-x)Ni(x)O4 con x=0.02 (0.98+0.02=1, sitio de Cu "
        "completo); el origen contiene la serie x=0.01,0.03,0.04,0.05 con Tc 30.7/14.0/3.2/2.1 K, y "
        "Tc=21.96 K cae entre x=0.01 y x=0.03",
        "Xiao et al., Phys. Rev. B 42, 8752 (1990): Ni substitution in La1.85Sr0.15CuO4"),
    "Rb3C60C60": ("Rb3C60",
        "token C60 duplicado (parseado como Rb3C120, fraccion molar distorsionada); Rb3C60 es el fullerido "
        "canonico y su Tc documentada (29-29.5 K) coincide con la Tc de la fila (29.0 K)",
        "Rosseinsky et al., Phys. Rev. Lett. 66, 2830 (1991): Rb3C60, Tc = 28-29.5 K"),
}

RECOVERABLE_UNPARSED = {
    "Y2C2Br0.5!1.5": "recuperable en principio: '!' es con alta probabilidad 'I' (Y2C2Br0.5I1.5, carburo-haluro "
                     "Y2C2(Br,I)2, Tc 10-12 K; Simon et al. 1991); NO se recupera en v3 para no anadir filas",
    "B1Sr2Ca3Cu4O2N+3": "recuperable en principio: 'O2N+3' = O(2n+3) con n = nº de Cu (fases BSr2Ca(n-1)CunO(2n+3), "
                        "Ihara et al.); NO se recupera en v3",
    "B1Sr2Ca4Cu5O2N+3": "idem O(2n+3), n=5; NO se recupera en v3",
    "B1Sr2Ca2Cu3O2N+3": "idem O(2n+3), n=3; NO se recupera en v3",
}


def audit(sc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    lines = []          # one dict per (row, category)
    final = {}          # checkpoint_row -> dict(decision, formula_v3)
    ck = -1
    parsed_rows = []
    for i, (name, tc) in enumerate(zip(sc["name"], sc["Tc"])):
        comp, err = parse(name)
        if comp is None:
            if tc > 0:
                rec = RECOVERABLE_UNPARSED.get(name, "no recuperable sin inventar una cantidad (oxigeno indeterminado "
                                               "'=z'/'+2', o coeficiente negativo)")
                lines.append(dict(fila_origen=i, checkpoint_row=-1, formula_original=name, Tc=tc,
                                  categoria="no_parseada_pymatgen", interpretacion=f"pymatgen: {err}. {rec}",
                                  decision="excluir", formula_corregida="",
                                  motivo="ya excluida por el parser en v1/v2; v3 no anade filas al checkpoint",
                                  fuente="pymatgen Composition; " + SRC_3DSC if "OY" in name else "pymatgen Composition"))
            continue
        if not tc > 0:
            continue
        ck += 1
        parsed_rows.append((ck, i, name, tc, comp))

    # second pass over the N=12440 parsed Tc>0 rows (needs the whole set for duplicates)
    keys3 = {}
    keys6 = {}
    for ck, i, name, tc, comp in parsed_rows:
        keys3.setdefault(norm_key(comp, 3), []).append(ck)
        keys6.setdefault(norm_key(comp, 6), []).append(ck)
    by_ck = {ck: (i, name, tc, comp) for ck, i, name, tc, comp in parsed_rows}

    def add(ck, categoria, interpretacion, decision, motivo, fuente, formula_corregida=""):
        i, name, tc, comp = by_ck[ck]
        lines.append(dict(fila_origen=i, checkpoint_row=ck, formula_original=name, Tc=tc, categoria=categoria,
                          interpretacion=interpretacion, decision=decision, formula_corregida=formula_corregida,
                          motivo=motivo, fuente=fuente))
        rank = {"excluir": 2, "corregir": 1, "mantener": 0}
        cur = final.get(ck)
        if cur is None or rank[decision] > rank[cur["decision"]]:
            final[ck] = dict(decision=decision, formula_v3=formula_corregida or name, categoria=categoria)
        elif cur is not None and rank[decision] == rank[cur["decision"]] and decision == "corregir":
            final[ck]["formula_v3"] = formula_corregida

    for ck, i, name, tc, comp in parsed_rows:
        els = sorted(e.symbol for e in comp.elements)
        cl = countless(name)
        cf = coefs(name)
        rep = repeated_symbols(name)
        n_at = comp.num_atoms
        mx = max([v for _, v in cf], default=np.nan)
        # (a) trailing OY
        if name.endswith("OY"):
            real_y = bool(re.search(r"Y\d", name[:-2]))
            add(ck, "OY_oxigeno_indeterminado",
                "la Y final no es itrio sino una cantidad indeterminada de oxigeno; " +
                ("la formula contiene ademas itrio real con coeficiente (la familia v2 es correcta en cuanto a "
                 "elementos) pero la estequiometria de O es desconocida" if real_y else
                 "la familia v2 contiene Y erroneamente y la estequiometria de O es desconocida"),
                "excluir", "sin la cantidad de O no puede calcularse la featurizacion Magpie (fracciones molares); "
                "no se inventa una cantidad; opcion por defecto de la tarea = excluir", SRC_3DSC)
        elif cl:
            if len(els) == 1:
                add(ck, "elemento_sin_coeficiente_monoelemental",
                    f"simbolo sin coeficiente ({','.join(cl)}) en formula monoelemental: la fraccion molar es 1 "
                    "sea cual sea el coeficiente", "mantener",
                    "la composicion normalizada no depende del coeficiente ausente", SRC_3DSC)
            elif cl == ["O"] and name.endswith("O"):
                add(ck, "O_final_sin_coeficiente",
                    "oxigeno final sin coeficiente: la nomenclatura SuperCon exige coeficiente explicito; leerlo como "
                    "O1 da una estequiometria imposible para un cuprato (p.ej. Y1Ba2Cu3F0.825O)",
                    "excluir", "cantidad de O indeterminada, mismo fundamento que OY; no se inventa", SRC_3DSC)
            else:
                add(ck, "elemento_sin_coeficiente_multielemental",
                    f"simbolo(s) sin coeficiente {cl}: no es distinguible 'coeficiente 1 implicito' de 'digito "
                    "perdido' (p.ej. BaKBi1O3 vs Ba(1-x)K(x)BiO3; Ru0.97Co0.03Sr2Gd1CuO8 vs Ru-1212 con Cu2)",
                    "excluir", "estequiometria indeterminada bajo la nomenclatura SuperCon (3DSC excluyo esta clase); "
                    "la familia (conjunto de elementos) no cambia, solo se pierde la fila", SRC_3DSC)
        # (d) zero coefficients
        zeros = [s for s, v in cf if v == 0]
        if zeros:
            add(ck, "coeficiente_cero",
                f"elemento(s) {zeros} con coeficiente 0 = extremo de una serie de sustitucion; pymatgen omite el "
                "elemento y la featurizacion Magpie es identica a la formula sin el (verificado)",
                "mantener", "composicion y familia correctas sin el elemento ausente", "pymatgen Composition (verificacion live)")
        # (d) extreme coefficients
        if mx > 50:
            if name in CORRECTIONS:
                new, why, src = CORRECTIONS[name]
                add(ck, "coeficiente_extremo_error_tipografico", why, "corregir",
                    "lectura unica fijada por balance de sitio y compuesto documentado presente en el mismo origen",
                    src, new)
            elif n_at > 150 or (mx > 500):
                add(ck, "coeficiente_extremo_error_tipografico",
                    f"coeficiente {mx:g} (total {n_at:g} atomos) es un error de captura (p.ej. O1013, O6050); la "
                    "cantidad correcta de O no es recuperable sin inventarla",
                    "excluir", ">150 atomos por formula (criterio 3DSC) y cantidad de O indeterminada",
                    "3DSC p. 3 (exclusion de formulas con >150 atomos); regla 'no inventar O'")
            elif 99.0 <= n_at <= 101.0 and all(v >= 0.5 or s in ("Fe", "Co", "Ni", "O") for s, v in cf):
                add(ck, "coeficiente_alto_porcentaje_atomico",
                    f"aleacion en porcentaje atomico (suma {n_at:g}); la composicion normalizada es correcta",
                    "mantener", "notacion legitima", "SuperCon/Stanev (aleaciones en at.%)")
            else:
                add(ck, "coeficiente_alto_cluster_o_celda",
                    f"coeficiente {mx:g} legitimo: fullerido (C60/C70), boro beta-romboedrico (B105), clatrato o "
                    "mayenita, o aleacion en at.% con dopante <0.5",
                    "mantener", "estequiometria quimicamente legitima", "quimica conocida del compuesto")
        # (d) repeated element tokens
        if rep:
            if name in CORRECTIONS:
                new, why, src = CORRECTIONS[name]
                add(ck, "elemento_repetido", why, "corregir",
                    "lectura unica fijada por Tc documentada", src, new)
            elif name == "D0.11Nb1D0.13Nb1":
                add(ck, "elemento_repetido_artefacto",
                    f"simbolo(s) repetido(s) {rep}: dos entradas concatenadas ('D0.11Nb1' y 'D0.13Nb1'); pymatgen suma "
                    "los coeficientes (H0.24Nb2). El origen ya contiene D0.13Nb1 con Tc identica (9.12 K)",
                    "excluir", "artefacto de fusion; duplica una fila existente", "inspeccion del origen")
            else:
                add(ck, "elemento_repetido_notacion_por_sitio",
                    f"simbolo(s) repetido(s) {rep}: notacion por sitio cristalografico (p.ej. Pb0.5Cu0.5 en el sitio de "
                    "Pb y Cu1 en el plano); pymatgen suma los coeficientes y la composicion total es correcta",
                    "mantener", "notacion legitima de SuperCon; composicion y familia correctas",
                    "SuperCon/Stanev (notacion por sitio)")
        # (c) isotopes
        if re.search(r"(?<![A-Za-z])[DT](?![a-z])", name):
            add(ck, "isotopo_D_T",
                "deuterio/tritio: pymatgen los mapea a H; familia y descriptores Magpie identicos al hidruro "
                "(verificado: D2S1 == H2S1 en las 132 columnas). Se pierde la informacion isotopica (efecto isotopico "
                "inverso en PdH/PdD no representable)",
                "mantener", "sin fundamento para excluir; la familia H|M es quimicamente correcta", "pymatgen Composition (verificacion live)")
        # rare heavy elements
        if set(els) & {"Am", "Cm", "Po"}:
            add(ck, "elemento_raro_Am_Cm_Po",
                "3DSC excluyo Am/Cm/Po por rareza; aqui los descriptores Magpie son finitos", "mantener",
                "no hay error quimico; se mantiene para no alterar la base N", "3DSC p. 3")
        # (e) duplicates after normalisation
        k6 = norm_key(comp, 6)
        if len(keys6[k6]) > 1:
            others = [by_ck[c][1] for c in keys6[k6] if c != ck]
            tcs = [by_ck[c][2] for c in keys6[k6]]
            add(ck, "duplicado_composicion_normalizada",
                f"misma composicion normalizada (1e-6) que {others[:3]}; Tc del grupo {min(tcs):.2f}-{max(tcs):.2f} K",
                "mantener", "misma familia -> nunca cruzan folds agrupados; Stanev solo promedio duplicados por cadena; "
                "no hay fundamento quimico para excluir una u otra", "3DSC p. 3 (21 materiales con orden distinto)")

    df = pd.DataFrame(lines)
    df["decision_final_fila"] = df["checkpoint_row"].map(lambda c: final.get(c, {}).get("decision", "excluir"))
    df["formula_v3"] = df["checkpoint_row"].map(lambda c: final.get(c, {}).get("formula_v3", ""))
    df = df.sort_values(["checkpoint_row", "categoria"]).reset_index(drop=True)
    meta = dict(n_source=int(len(sc)), n_parsed_Tc_gt0=len(parsed_rows),
                n_dup_groups_1e3=sum(1 for v in keys3.values() if len(v) > 1),
                n_dup_rows_1e3=sum(len(v) for v in keys3.values() if len(v) > 1),
                n_dup_groups_1e6=sum(1 for v in keys6.values() if len(v) > 1),
                n_dup_rows_1e6=sum(len(v) for v in keys6.values() if len(v) > 1))
    return df, final, meta


def summary_md(df, final, meta) -> str:
    rows_flagged = df[df.checkpoint_row >= 0].checkpoint_row.nunique()
    dec = Counter(v["decision"] for v in final.values())
    n_keep = meta["n_parsed_Tc_gt0"] - dec.get("excluir", 0)
    out = ["# Auditoria semantica de Dataset B (H03) -- resumen", "",
           f"Procedencia: {json.dumps(PROCEDENCIA, ensure_ascii=False)}", "",
           f"- Filas del origen: {meta['n_source']}; parseables con Tc>0 (base v2): {meta['n_parsed_Tc_gt0']}; "
           f"no parseables con Tc>0: {int((df.checkpoint_row < 0).sum())}.",
           f"- Filas de la base v2 con al menos una marca: {rows_flagged}.",
           f"- Decision final por fila (base v2): " + ", ".join(f"{k}={v}" for k, v in sorted(dec.items())) +
           f"; filas en v3 = {n_keep} (de ellas {dec.get('corregir', 0)} corregidas).",
           f"- Duplicados por composicion normalizada: {meta['n_dup_rows_1e6']} filas en {meta['n_dup_groups_1e6']} grupos "
           f"(1e-6); {meta['n_dup_rows_1e3']} filas en {meta['n_dup_groups_1e3']} grupos (1e-3). Se mantienen.", "",
           "| categoria | filas | mantener | corregir | excluir |", "|---|---:|---:|---:|---:|"]
    for cat, g in df.groupby("categoria"):
        c = Counter(g["decision"])
        out.append(f"| {cat} | {len(g)} | {c.get('mantener', 0)} | {c.get('corregir', 0)} | {c.get('excluir', 0)} |")
    out += ["", "## Fundamento de la lectura 'OY'", "", SRC_3DSC, "",
            "## Correcciones aplicadas (familia inalterada en las tres)", ""]
    for k, (new, why, src) in CORRECTIONS.items():
        out.append(f"- `{k}` -> `{new}`: {why}. Fuente: {src}.")
    out += ["", "## Filas no parseables (excluidas ya en v1/v2)", ""]
    for _, r in df[df.checkpoint_row < 0].iterrows():
        out.append(f"- fila {r.fila_origen} `{r.formula_original}` (Tc={r.Tc}): {r.interpretacion}")
    return "\n".join(out) + "\n"


def main() -> int:
    sc = pd.read_csv(DATA / "supercon_stanev.csv")
    df, final, meta = audit(sc)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True); RESULTS_V3.mkdir(parents=True, exist_ok=True)
    df["procedencia"] = json.dumps(PROCEDENCIA, ensure_ascii=False)
    cols = ["fila_origen", "checkpoint_row", "formula_original", "Tc", "categoria", "interpretacion", "decision",
            "formula_corregida", "motivo", "fuente", "decision_final_fila", "formula_v3", "procedencia"]
    df[cols].to_csv(AUDIT_DIR / "semantic_audit_B.csv", index=False)
    json.dump(dict(procedencia=PROCEDENCIA, meta=meta,
                   decisions={str(k): v for k, v in sorted(final.items())}),
              open(RESULTS_V3 / "semantic_audit_B_decisions.json", "w"), indent=1, ensure_ascii=False)
    (RESULTS_V3 / "semantic_audit_B_summary.md").write_text(summary_md(df, final, meta))
    print(json.dumps(dict(meta=meta, decisions=Counter(v["decision"] for v in final.values()),
                          categorias=df.categoria.value_counts().to_dict()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
