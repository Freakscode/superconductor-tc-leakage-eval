"""Shorter abstract and named AI-use disclosure (v3.1, author's request of 2026-10-04).

Replaces the abstract. Every number of the new abstract takes the value specification of the manifest
entry that checked the same number in the old abstract (matched by value), with a new anchor; the old
abstract entries are removed from both manifests. Rewrites the AI-use paragraph of the
Acknowledgements to name the model.
"""
import json, re, sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper1_manuscript.tex"
MAIN, M31 = ROOT / "code/v3/manuscript_numbers.json", ROOT / "code/v3/manuscript_numbers_v31.json"

NEW = r"""Composition-based models of the superconducting critical temperature are usually validated by
splitting SuperCon-derived datasets row by row, although those datasets are built from doping series
of a few thousand element systems. We ask which retrospective evaluation best anticipates performance
on unseen element systems or later dates. On two curations of SuperCon that share
\SI{96.6}{\percent} of their systems (\num{21263} and \num{12336} rows), holding out whole element
systems raises the pooled MAE of tree ensembles by \SIrange{53}{64}{\percent}, with paired
family-bootstrap intervals that exclude zero; per system the inflation is \SIrange{22}{29}{\percent},
so the gap concentrates in the large systems. Element identity explains most of the conventional
$\Rsq$, and on held-out systems the elemental-property descriptors do not beat element presence plus
stoichiometric fractions. Retrospective top-\num{100} precision on held-out systems stays at
\SI{87.0}{\percent} and \SI{82.4}{\percent}. On \num{12543} compositions dated by first report,
random cross-validation on pre-2010 data estimates MAE \SI{4.34}{\kelvin} and $\Rsq = \num{0.91}$,
against \SI{7.43}{\kelvin} and \num{0.33} realised on later compositions; grouped protocols come
closer, but none is within \num{0.1} of the realised $\Rsq$ at more than one cut. At a fixed budget,
one training record per system instead of whole systems lowers the per-system error on unseen systems
by \SI{2.9}{\kelvin} and \SI{1.8}{\kelvin} in the two curations. Row-level partitions, the audited data and a
per-claim verifier are released."""

AI_OLD = r"\textit{Use of generative AI.} A large-language-model research assistant \textbf{[author to state model and version]} was used to write and run analysis code,"
AI_NEW = (r"\textit{Use of generative AI.} Claude Opus 5.5 (Anthropic) was used in the execution of the "
          r"algorithms---writing and running the analysis code---and in the search for and verification of "
          r"sources of information. It was also used to")
AI_OLD2 = r"design and execute the experiments of"
AI_NEW2 = r"design and execute the experiments of"


def main():
    tex = TEX.read_text()
    a0, a1 = tex.index("\\begin{abstract}") + len("\\begin{abstract}"), tex.index("\\end{abstract}")
    old_abs = re.sub(r"\s+", " ", tex[a0:a1])
    spec = importlib.util.spec_from_file_location("vm", ROOT / "code/v3/verify_manuscript.py")
    vm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)
    mains, m31 = json.load(open(MAIN)), json.load(open(M31))

    def in_old_abstract(e):
        if "anchor" not in e: return False
        pat = re.escape(re.sub(r"\s+", " ", e["anchor"]).strip()).replace(re.escape("<<v>>"), vm.NUM).replace(re.escape("<<n>>"), vm.NUM).replace(re.escape("<<*>>"), r".{0,60}?")
        return re.search(pat, old_abs) is not None
    pool = []
    for m in (mains, m31):
        keep = []
        for e in m["entries"]:
            if e.get("table"): keep.append(e); continue
            (pool if in_old_abstract(e) else keep).append(e)
        m["entries"] = keep
    # value -> spec (first match wins)
    by_val = {}
    for e in pool:
        v, _ = vm.resolve(e)
        try: key = format(float(v), e.get("fmt") or ".6g")
        except (TypeError, ValueError): continue
        by_val.setdefault(key, e)
    tex = tex[:a0] + "\n" + NEW + "\n" + tex[a1:]
    flat = re.sub(r"\s+", " ", NEW)
    body = re.sub(r"\s+", " ", tex.split("\\begin{document}")[1].split("\\begin{thebibliography}")[0])
    toks = list(re.finditer(r"\\(?:SI|num|SIrange|numrange)\{([^}]*)\}(?:\{([^}]*)\})?|\$\\Rsq = \\num\{([^}]*)\}\$", flat))
    new = []
    for m in re.finditer(r"(?:\\num|\\SI|\\SIrange|\\numrange)\{", flat):
        pass
    # enumerate every number inside siunitx macros of the new abstract
    for m in re.finditer(r"\\(SI|num)\{(-?[\d.]+)\}|\\(SIrange|numrange)\{(-?[\d.]+)\}\{(-?[\d.]+)\}", flat):
        vals = [(m.start(2), m.group(2))] if m.group(2) else [(m.start(4), m.group(4)), (m.start(5), m.group(5))]
        for pos, txt in vals:
            fmt = f".{len(txt.split('.')[1])}f" if "." in txt else ".0f"
            src = by_val.get(format(float(txt), fmt))
            if src is None:
                if txt in ("100", "2010", "0.1"):
                    src = {"const": float(txt), "reason": "quoted parameter of the analysis (K of top-K, temporal cut, R2 tolerance)"}
                else:
                    raise KeyError(f"no source for abstract number {txt}")
            for back in (30, 50, 80, 120, 170):
                b0 = flat.rfind(" ", 0, max(0, pos - back)) + 1
                anchor = flat[b0:pos] + "<<v>>" + flat[pos + len(txt):pos + len(txt) + 12]
                pat = re.escape(anchor).replace(re.escape("<<v>>"), vm.NUM)
                if len(re.findall(pat, body)) == 1: break
            else: raise AssertionError(txt)
            e = {k: v for k, v in src.items() if k not in ("id", "anchor", "occurrences", "note")}
            e.update(id=f"abs31_{len(new):02d}", anchor=anchor, fmt=fmt)
            new.append(e)
    m31["entries"] += new
    assert AI_OLD in tex, "AI paragraph not found"
    tex = tex.replace(AI_OLD, AI_NEW)
    TEX.write_text(tex)
    json.dump(mains, open(MAIN, "w"), indent=1, ensure_ascii=False)
    json.dump(m31, open(M31, "w"), indent=1, ensure_ascii=False)
    print(f"old abstract entries removed: {len(pool)} | new abstract entries: {len(new)}")


if __name__ == "__main__":
    sys.exit(main())
