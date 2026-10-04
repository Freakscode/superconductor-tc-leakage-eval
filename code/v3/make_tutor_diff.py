"""Marked-up version v5 -> v3.1 for the tutor, with one note per review comment.

1. Copies paper1_manuscript.tex (v3.1, clean) to an annotated copy and inserts \revnote{N}{text} after
   an anchor phrase for each of the 23 comments of the tutor's review (Paper_Gabriel.pdf).
2. Runs latexdiff between paper1_manuscript_v5_backup.tex (the version the tutor read) and the
   annotated copy, and compiles the result twice with pdflatex.
The clean manuscript is never modified. Output: <outdir>/paper1_diff_v5_v31.{tex,pdf}

Usage: python code/v3/make_tutor_diff.py OUTDIR
"""
import os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TB = Path(os.environ.get("TEXBIN", "/usr/local/texlive/2025/bin/universal-darwin"))
TAG = "avasquee@unal.edu.co"
NOTES = [  # (comment, anchor in v3.1 text, note)
    (1, "once entries recorded as non-superconducting are admitted to the candidate pool)",
     "Resumen acotado: retirados el tamaño efectivo, la fracción de memorización y la promesa de síntesis; "
     "la frase temporal da ahora MAE y R²; se añade el resultado de adquisición (c. 23)."),
    (2, "and the wrong one when the target is a new system.",
     "«every test row» sustituido por fracciones medidas (gemelo exacto 37.0 % en A, 4.9 % en B); "
     "se reconoce cuándo la partición aleatoria es la medida adecuada."),
    (3, "grouping, not a one-to-one map onto doping series", "Ya corregido en la versión anterior: definición operativa y contraejemplo {Y, Ba, Cu, O}."),
    (4, "\\label{sec:overlap}", "Ya corregido: solapamiento cuantificado de cuatro formas; retirada la «replicación independiente»."),
    (5, "so the target throughout is the $\\Tc$ recorded in the database.",
     "Nuevo: tabla de flujo de filtrado (Tabla 1) y declaración del criterio de Tc. La agregación por "
     "composición (mediana, con el máximo como sensibilidad) solo afecta al corpus temporal."),
    (6, "was a different curation of SuperCon.",
     "Origen identificado: manuscrito de congreso previo del autor (GridSearchCV, CV aleatoria de 3 pliegues, "
     "split 80/20, −MSE). Las rejillas no se registraron; se declara y se remite a la selección anidada."),
    (7, "is not a well-posed quantity.",
     "Ya corregido: N efectivo solo por estimador, con su fórmula; retirada la equivalencia con el número de familias."),
    (8, "not its ability to order the system's variants.",
     "Nuevo: ordenamiento dentro de sistemas (Spearman mediano 0.39 en A y 0.56 en B con sistemas retenidos)."),
    (9, "\\label{sec:results-ceiling}", "c. 9 ya corregido: η² como descripción, sin inferir fuga; oráculo separado de las cifras de CV. c. 17 ya corregido: escalera presencia → fracciones → descriptores con los mismos pliegues."),
    (10, "the intervals of section~\\ref{sec:results-gap} cover both values.",
     "c. 10 ya corregido: R² del nulo medido (−0.007 / −0.003) y explicado en la leyenda de la Tabla 1."),
    (11, "such entries expected among the \\num{100} picks.",
     "c. 11: t_f en kelvin por pliegue añadido a la leyenda de la tabla de selección. c. 12 ya corregido "
     "(empates fraccionales). c. 16: «measured non-superconductors» sustituido; SuperCon no distingue "
     "no superconductor, no reportado y censurado."),
    (13, "The nearest-neighbour diagnostic is consistent with this reading",
     "«confirms the mechanism» y «of the same size» retirados; nuevo análisis de error frente a distancia, "
     "estratificado por Tc."),
    (14, "\\label{sec:results-robust}",
     "c. 14 y c. 18 por confirmar: cada tabla declara si la inflación es cociente de MAE agregados; los "
     "valores por semilla están en results_v3."),
    (15, "which is why the per-system inflation is smaller than the pooled one",
     "Nuevo: errores por rango de Tc, tamaño de sistema y distancia."),
    (19, "The partition depends on the order of the labels, not on their gaps;",
     "Nuevo: prueba mínima de etiquetado (con huecos, recodificada, por primera aparición)."),
    (20, "attribute no procedural detail to them beyond that.",
     "No abordado del todo: texto completo de Meredig 2018 no obtenido. Añadidos Konno, Li, MatFold, Durdy, Murdock y Kauwe."),
    (21, "\\section{Conclusion}", "Reescrita en la versión anterior; añadida la regla de adquisición."),
    (22, "so it is not a prospective test.",
     "«forward-in-time» → «retrospective temporal split». El test principal ya excluía composiciones vistas; "
     "sensibilidad a la datación medida en los cuatro cortes (Limitaciones)."),
    (23, "\\label{sec:acquisition}", "Nuevo experimento: presupuesto fijo, amplitud frente a profundidad (Figura 6)."),
]
MACRO = (r"\newcommand{\revnote}[2]{{\color{blue!65!black}\sffamily\footnotesize"
         r" [@" + TAG.replace("_", r"\_") + r" --- comentario~#1: #2]}}")


def latex_escape(t):
    for a, b in (("\\", r"\textbackslash{}"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("&", r"\&"),
                 ("{", r"\{"), ("}", r"\}"), ("η²", r"$\eta^2$"), ("R²", r"$R^2$"), ("→", r"$\to$"),
                 ("−", r"$-$"), ("t\\_f", r"$t_f$")):
        t = t.replace(a, b)
    return t


def ws(s):
    return r"\s+".join(re.escape(t) for t in s.split())


def main(outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    tex = (ROOT / "paper1_manuscript.tex").read_text()
    for n, anchor, note in NOTES:
        ms = list(re.finditer(ws(anchor), tex)); assert len(ms) >= 1, (n, anchor)
        m = ms[0]
        note = latex_escape(note)
        tex = tex[:m.end()] + f" \\revnote{{{n}}}{{{note}}}" + tex[m.end():]
    tex = tex.replace("\\begin{document}", MACRO + "\n\\begin{document}", 1)
    ann = out / "paper1_v31_annotated.tex"; ann.write_text(tex)
    old = out / "paper1_v5.tex"; shutil.copy(ROOT / "paper1_manuscript_v5_backup.tex", old)
    diff = out / "paper1_diff_v5_v31.tex"
    r = subprocess.run([str(TB / "latexdiff"), "--type=CFONT", "--math-markup=0",
                        "--append-safecmd=revnote", r"--config=PICTUREENV=(?:picture|DIFnomarkup|tabular)[\w\d*@]*", "--exclude-textcmd=section,subsection",
                        str(old), str(ann)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    diff.write_text(r.stdout)
    for f in ["iopjournal.cls", "orcid.pdf"] + [p.name for p in ROOT.glob("figure*.png")]:
        shutil.copy(ROOT / f, out / f)
    for _ in range(2):
        c = subprocess.run([str(TB / "pdflatex"), "-interaction=nonstopmode", diff.name], cwd=out, capture_output=True, text=True, timeout=240)
    log = (out / "paper1_diff_v5_v31.log").read_text(errors="replace")
    errs = [l for l in log.splitlines() if l.startswith("!")]
    print("notas:", len(NOTES), "| errores LaTeX:", len(errs), errs[:5])
    print(re.findall(r"Output written on .*", log))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
