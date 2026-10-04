"""Verify every number of the manuscript IN ITS OWN LOCATION against its result file (v3, H12).

Why a v3.  code/v2/verify_manuscript.py checks that each expected figure appears *somewhere*
in the .tex (substring search over the whole file).  A table cell can therefore be corrupted
without the check noticing, as long as the same digits survive elsewhere (the 5.23 -> 99.99
mutation of the adversarial review passes v2 with "117 OK / 0 discrepancias").

What this script does instead
  1. Parses the .tex with a minimal grammar:
       * every `tabular` inside a `table` environment -> rows -> cells (\\multicolumn expanded,
         \\toprule/\\midrule/\\bottomrule/\\cmidrule tolerated, {...} braces stripped);
       * in the prose (everything outside tabulars, after \\begin{document}, before the
         bibliography) the siunitx macros \\num{}, \\SI{}{}, \\SIrange{}{}{}, \\numrange{}{}, bare
         numbers inside $...$ math and `~x.y.z` version strings, each with line number, the
         nearest \\section/\\subsection and +-40 characters of context.
  2. Loads a MANIFEST (code/v3/manuscript_numbers.json) with one entry per figure:
       cell  : {"table": "tab:tuning", "row": "Published configuration", "col": 2, "part": 0,
                "source": "results_v2/nested_cv_inflacion.csv", "key": {"dataset": "A"},
                "field": "MAE_random_publicada", "fmt": ".2f"}
       prose : {"anchor": "raises XGBoost's MAE from \\SI{<<v>>}{\\kelvin}", "source": ..., ...}
                (`anchor` is literal text with <<v>> = the checked number, <<n>> = any number,
                 <<ver>> = a dotted version string, <<*>> = up to 60 arbitrary characters; it must match exactly `occurrences`
                 times (default 1) in the whitespace-normalised prose).
     Value sources: CSV (key/field/agg/scale), JSON (dotted path), a regex on a text file
     (requirements.txt), an expression over several such values (`expr` + `vars`), a Python
     expression evaluated in a repository module (`python`: hyperparameters straight from
     make_models), or a declared constant (`const` + `reason`, reported separately as NOT
     checked against any result file).
  3. Checks EACH entry at ITS location: OK / DISCREPANCIA / NO ENCONTRADA (with line,
     section, context).  Exit status 1 if anything is not OK.
  4. Reports coverage: every numeric token found in the .tex that no manifest entry claims.

Runs with no arguments from a clean checkout (default .tex = <repo root>/paper1_manuscript.tex).
Options: --tex PATH  --manifest PATH  --report PATH.json  --quiet  --min-coverage F (fail below F)
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, re, sys
from pathlib import Path

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
HERE = Path(__file__).resolve().parent
DEFAULT_TEX = ROOT / "paper1_manuscript.tex"
DEFAULT_MANIFEST = HERE / "manuscript_numbers.json"

NUM = r"-?\d+(?:\.\d+)?"
# a number that is not glued to a word, a subscript/superscript or another number
NUM_TOKEN = re.compile(r"(?<![\w.^_\\-])-?\d+(?:\.\d+)?(?![\w.])")


# ============================================================================ .tex parsing ====
def strip_comments(tex: str) -> str:
    """Blank out %-comments (keeping line structure and offsets: replaced by spaces)."""
    out = []
    for line in tex.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        out.append(line if m is None else line[:m.start()] + " " * (len(line) - m.start()))
    return "\n".join(out)


def balanced(s: str, i: int) -> int:
    """s[i] == '{'; return index just past the matching '}'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"): depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0: return j + 1
    raise ValueError("unbalanced braces")


def line_of(text: str, off: int) -> int:
    return text.count("\n", 0, off) + 1


def section_index(text: str):
    secs = [(m.start(), m.group(2).strip()) for m in re.finditer(r"\\(section|subsection)\*?\{([^}]*)\}", text)]
    def nearest(off):
        name = "(preamble/abstract)"
        for pos, nm in secs:
            if pos <= off: name = nm
            else: break
        return name
    return nearest


def clean_cell(cell: str) -> str:
    """Reduce a cell / label to plain text with numbers exposed and sub/superscripts removed."""
    s = cell
    s = re.sub(r"\\(?:SIrange|numrange)(?:\[[^\]]*\])?\{([^}]*)\}\{([^}]*)\}(?:\{[^}]*\})?", r" \1 -- \2 ", s)
    s = re.sub(r"\\SI(?:\[[^\]]*\])?\{([^}]*)\}\{[^}]*\}", r" \1 ", s)
    s = re.sub(r"\\num(?:\[[^\]]*\])?\{([^}]*)\}", r" \1 ", s)
    s = re.sub(r"[_^](\{[^}]*\}|\\?\w+|.)", " ", s)              # subscripts / superscripts
    s = re.sub(r"\\(?:textit|textbf|emph|texttt|mathrm)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)                            # remaining macros (\Rsq, \pm, \% ...)
    s = s.replace("{", " ").replace("}", " ").replace("$", " ").replace("~", " ").replace("--", " -- ")
    return re.sub(r"\s+", " ", s).strip()


def numbers_in(cleaned: str):
    return [m.group(0) for m in NUM_TOKEN.finditer(cleaned)]


def raw_number_spans(raw_cell: str):
    """Spans of the numeric tokens inside the RAW cell text (same order as numbers_in(clean_cell))."""
    s = raw_cell.replace("--", "  ")
    s = re.sub(r"[_^](\{[^}]*\}|\\?\w+|.)", lambda m: " " * len(m.group(0)), s)
    s = re.sub(r"\\(?:multicolumn|cmidrule)\{\d+\}", lambda m: " " * len(m.group(0)), s)
    return [(m.start(), m.end(), m.group(0)) for m in NUM_TOKEN.finditer(s)]


def parse_tables(text: str):
    """-> list of {label, line, rows:[{cells:[str], clean:[str], line, block}]} for each tabular."""
    tables = []
    for tm in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", text, re.S):
        body = tm.group(1); base = tm.start(1)
        lab = re.search(r"\\label\{(tab:[^}]*)\}", body)
        label = lab.group(1) if lab else f"table@line{line_of(text, tm.start())}"
        for bt in re.finditer(r"\\begin\{tabular\}", body):
            spec_end = balanced(body, body.index("{", bt.end()))
            end = body.index("\\end{tabular}", spec_end)
            tab_body = body[spec_end:end]; tab_base = base + spec_end
            rows, block, pos = [], "", 0
            for raw in re.split(r"\\\\", tab_body):
                seg_off = tab_base + pos; pos += len(raw) + 2
                blank_ = lambda m: " " * len(m.group(0))            # mask, keep offsets
                r = re.sub(r"\\(?:toprule|midrule|bottomrule)", blank_, raw)
                r = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", blank_, r)
                if not r.strip(): continue
                k = len(r) - len(r.lstrip()); row_off = seg_off + k
                cells, cell_offs, start = [], [], 0
                for m in list(re.finditer(r"(?<!\\)&", r)) + [None]:
                    cend = m.start() if m else len(r)
                    seg = r[start:cend]; lead = len(seg) - len(seg.lstrip()); c = seg.strip()
                    if c.startswith("\\multicolumn"):          # \multicolumn{n}{fmt}{content} -> n cells
                        i1 = c.index("{"); j1 = balanced(c, i1); n = int(c[i1 + 1:j1 - 1])
                        i2 = c.index("{", j1); j2 = balanced(c, i2)
                        i3 = c.index("{", j2); j3 = balanced(c, i3)
                        content = c[i3 + 1:j3 - 1] + c[j3:]; lead2 = len(content) - len(content.lstrip())
                        cells += [content.strip()] + [""] * (n - 1)
                        cell_offs += [seg_off + start + lead + i3 + 1 + lead2] + [None] * (n - 1)
                    else:
                        cells.append(c); cell_offs.append(seg_off + start + lead)
                    start = cend + 1
                clean = [clean_cell(c) for c in cells]
                is_block = bool(re.match(r"\s*\\multicolumn", r.strip())) and sum(1 for c in cells if c) == 1
                if is_block: block = clean[0]
                rows.append({"cells": cells, "cell_offs": cell_offs, "clean": clean, "line": line_of(text, row_off),
                             "block": block, "is_block": is_block})
            tables.append({"label": label, "line": line_of(text, tm.start()), "rows": rows, "span": (tm.start(), tm.end()),
                           "tabular_span": (base + bt.start(), base + end + len("\\end{tabular}"))})
    return tables


def prose_text(text: str, tables):
    """Blank out tables, preamble and bibliography; return text with offsets preserved."""
    t = list(text)
    def blank(a, b):
        for i in range(a, b):
            if t[i] != "\n": t[i] = " "
    for tb in tables: blank(*tb["tabular_span"])          # captions stay: they are prose
    bd = text.find("\\begin{document}"); blank(0, bd if bd >= 0 else 0)
    bib = text.find("\\begin{thebibliography}")
    if bib >= 0: blank(bib, len(text))
    s = "".join(t)
    for pat in (r"\\includegraphics\[[^\]]*\]\{[^}]*\}", r"\\(?:label|ref|cite|url|setlength|orcid|email)\{[^}]*\}",
                r"\\setlength\{[^}]*\}\{[^}]*\}", r"\\begin\{[^}]*\}", r"\\end\{[^}]*\}"):
        s = re.sub(pat, lambda m: " " * len(m.group(0)), s)
    return s


def prose_tokens(text: str, prose: str, nearest):
    """Numeric tokens in the prose with their absolute spans."""
    toks, taken = [], []
    def add(kind, val, a, b):
        toks.append({"kind": kind, "value": val, "start": a, "end": b, "line": line_of(text, a),
                     "section": nearest(a), "context": re.sub(r"\s+", " ", text[max(0, a - 40):b + 40])})
        taken.append((a, b))
    for m in re.finditer(r"\\(SIrange|numrange)(?:\[[^\]]*\])?\{([^}]*)\}\{([^}]*)\}", prose):
        for g in (2, 3):
            for n in NUM_TOKEN.finditer(m.group(g)):
                add(m.group(1), n.group(0), m.start(g) + n.start(), m.start(g) + n.end())
    for m in re.finditer(r"\\(SI|num)(?:\[[^\]]*\])?\{([^}]*)\}", prose):
        for n in NUM_TOKEN.finditer(m.group(2)):
            add(m.group(1), n.group(0), m.start(2) + n.start(), m.start(2) + n.end())
    macro_spans = [(m.start(), m.end()) for m in re.finditer(r"\\(?:SIrange|numrange|SI|num)(?:\[[^\]]*\])?\{[^}]*\}(?:\{[^}]*\})*", prose)]
    def in_macro(a): return any(x <= a < y for x, y in macro_spans)
    for m in re.finditer(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", prose, re.S):
        inner = re.sub(r"[_^](\{[^}]*\}|\\?\w+|.)", lambda k: " " * len(k.group(0)), m.group(1))
        for n in NUM_TOKEN.finditer(inner):
            a = m.start(1) + n.start()
            if not in_macro(a): add("math", n.group(0), a, m.start(1) + n.end())
    for m in re.finditer(r"~(\d+\.\d+(?:\.\d+)?)", prose):
        add("version", m.group(1), m.start(1), m.end(1))
    toks.sort(key=lambda t: t["start"])
    return toks


def normalise(prose: str):
    """Whitespace-normalised prose + map normalised index -> original offset."""
    out, idx, i, n = [], [], 0, len(prose)
    while i < n:
        if prose[i].isspace():
            j = i
            while j < n and prose[j].isspace(): j += 1
            out.append(" "); idx.append(i); i = j
        else:
            out.append(prose[i]); idx.append(i); i += 1
    idx.append(n)
    return "".join(out), idx


# ============================================================================ value sources ===
_CACHE: dict = {}


def load_csv(path: Path):
    import pandas as pd
    key = ("csv", str(path))
    if key not in _CACHE: _CACHE[key] = pd.read_csv(path, comment=None)
    return _CACHE[key]


def json_path(obj, dotted: str):
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", dotted):
        obj = obj[int(part[1:-1])] if part.startswith("[") else obj[part]
    return obj


def load_module(relpath: str):
    key = ("mod", relpath)
    if key not in _CACHE:
        p = ROOT / relpath
        spec = importlib.util.spec_from_file_location(p.stem, p)
        m = importlib.util.module_from_spec(spec); sys.modules[p.stem] = m; spec.loader.exec_module(m)
        _CACHE[key] = m
    return _CACHE[key]


def resolve(spec: dict):
    """-> (value, description).  value is float, int or str."""
    if "const" in spec:
        return spec["const"], f"const ({spec.get('reason', 'sin razón')})"
    if "expr" in spec:
        env = {"min": min, "max": max, "abs": abs, "round": round, "math": math, "len": len, "sum": sum, "float": float, "int": int,
               "str": str, "re": re, "format": format, "sorted": sorted, "set": set}
        for name, sub in spec.get("vars", {}).items():
            env[name], _ = resolve(sub)
        return eval(spec["expr"], {"__builtins__": {}, **env}), f"expr {spec['expr']}"
    if "python" in spec:
        mod = load_module(spec["python"]["module"])
        return eval(spec["python"]["expr"], {"__builtins__": {}}, vars(mod)), f"{spec['python']['module']}: {spec['python']['expr']}"
    src = ROOT / spec["source"]
    if not src.exists(): raise FileNotFoundError(str(src))
    if "regex" in spec:
        m = re.search(spec["regex"], src.read_text())
        if not m: raise KeyError(f"regex {spec['regex']!r} not found in {spec['source']}")
        return m.group(1), f"{spec['source']} /{spec['regex']}/"
    if src.suffix == ".json":
        v = json_path(json.load(open(src)), spec["field"])
        if spec.get("agg"): v = {"min": min, "max": max, "len": len, "sum": sum}[spec["agg"]](v)
        return v, f"{spec['source']}:{spec['field']}"
    df = load_csv(src)
    sub = df
    for k, want in spec.get("key", {}).items():
        col = sub[k].astype(str)
        sub = sub[col.isin([str(w) for w in want])] if isinstance(want, list) else sub[col == str(want)]
    if len(sub) == 0: raise KeyError(f"key {spec.get('key')} selects no row of {spec['source']}")
    field = spec["field"]
    if ";" in field or "|" in field:                       # "col;i" -> i-th element of a ';'-separated cell
        col, i = re.split(r"[;|]", field); vals = [float(x) for x in str(sub[col].iloc[0]).split(";")]
        v = vals[int(i)]
    else:
        col = sub[field]
        agg = spec.get("agg")
        if agg == "count":
            return len(sub), f"{spec['source']} {spec.get('key', '')} count"
        if agg is None:
            if len(sub) != 1: raise KeyError(f"key {spec.get('key')} selects {len(sub)} rows of {spec['source']} (need agg)")
            v = col.iloc[0]
        else:
            v = {"min": col.min(), "max": col.max(), "mean": col.mean(), "sum": col.sum(), "count": len(col),
                 "std": col.std(ddof=0), "first": col.iloc[0]}[agg]
    if isinstance(v, str):
        try: v = float(v)
        except ValueError: pass
    else:
        v = float(v) * float(spec.get("scale", 1)) + float(spec.get("offset", 0))
    return v, f"{spec['source']} {spec.get('key', '')} {field}{' ' + spec['agg'] if spec.get('agg') else ''}"


def compare(tex_str: str, value, fmt: str | None, tol: float | None):
    """-> (status, expected_str).  status in OK / OK~ (rounding edge) / DISCREPANCIA."""
    tex_str = tex_str.replace("\u2212", "-")
    if isinstance(value, str) and not re.fullmatch(NUM, value or ""):
        return ("OK" if tex_str == value else "DISCREPANCIA"), value
    v = float(value)
    exp = format(v, fmt) if fmt else (str(int(round(v))) if float(v).is_integer() else repr(v))
    if tex_str == exp or tex_str.lstrip("+") == exp: return "OK", exp
    try: t = float(tex_str)
    except ValueError: return "DISCREPANCIA", exp
    d = len(tex_str.split(".")[1]) if "." in tex_str else 0
    if tol is None: tol = 0.5 * 10 ** (-d) + 1e-9
    if abs(t - v) <= tol: return ("OK" if fmt is None else "OK~"), exp
    return "DISCREPANCIA", exp


# ============================================================================ verification ===
def find_row(table, entry):
    rows = [r for r in table["rows"] if not r["is_block"] or entry.get("block_row")]
    if entry.get("block"):
        rows = [r for r in rows if re.search(entry["block"], r["block"] or "")]
    want = entry["row"] if isinstance(entry["row"], list) else [entry["row"]]
    hits = []
    for r in rows:
        if len(r["clean"]) < len(want): continue
        if all(re.fullmatch(w, r["clean"][i]) or re.search(w, r["clean"][i]) for i, w in enumerate(want)):
            hits.append(r)
    return hits


def verify(tex_path: Path, manifest_path: Path):
    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    nearest = section_index(text)
    tables = parse_tables(text)
    prose = prose_text(text, tables)
    ptoks = prose_tokens(text, prose, nearest)
    norm, idx = normalise(prose)
    manifest = json.load(open(manifest_path))
    entries = list(manifest["entries"])
    # additions written by later amendment scripts (e.g. manuscript_numbers_v31.json) sit next to
    # the main manifest and are always checked together with it
    # (only *_v3<digit>.json: manuscript_numbers_v5.json is the archived manifest of the v5 text)
    for extra in sorted(Path(manifest_path).parent.glob(Path(manifest_path).stem + "_v3[0-9].json")):
        entries += json.load(open(extra))["entries"]
    results, claimed_cells, claimed_prose = [], set(), set()
    tbl = {t["label"]: t for t in tables}

    for e in entries:
        res = {"id": e.get("id", "?"), "status": None, "where": "", "tex": None, "expected": None, "source": "",
               "kind": "const" if "const" in e else "checked", "note": e.get("note")}
        try:
            value, desc = resolve(e)
        except Exception as ex:                    # noqa
            value, desc = None, f"FUENTE NO RESUELTA: {type(ex).__name__}: {ex}"
        res["source"] = desc
        if "table" in e:
            t = tbl.get(e["table"])
            if t is None:
                res.update(status="NO ENCONTRADA", where=f"tabla {e['table']} no existe"); results.append(res); continue
            hits = find_row(t, e)
            if len(hits) != 1:
                res.update(status="NO ENCONTRADA", where=f"{e['table']} fila {e['row']!r} bloque {e.get('block')!r}: {len(hits)} coincidencias"); results.append(res); continue
            row = hits[0]; col = int(e["col"]) - 1
            if col >= len(row["clean"]):
                res.update(status="NO ENCONTRADA", where=f"{e['table']} fila {e['row']!r}: sin columna {e['col']}"); results.append(res); continue
            nums = numbers_in(row["clean"][col]); part = int(e.get("part", 0))
            res["where"] = f"{e['table']} l.{row['line']} fila {row['clean'][0][:40]!r} col {e['col']} part {part}"
            if part >= len(nums):
                res.update(status="NO ENCONTRADA", where=res["where"] + f" (celda {row['cells'][col]!r} sin número #{part})"); results.append(res); continue
            claimed_cells.add((e["table"], row["line"], col, part))
            res["tex"] = nums[part]
            rs = raw_number_spans(row["cells"][col])
            if len(rs) == len(nums) and row["cell_offs"][col] is not None:
                res["span"] = [[row["cell_offs"][col] + rs[part][0], row["cell_offs"][col] + rs[part][1]]]
        else:
            pat = re.escape(re.sub(r"\s+", " ", e["anchor"]).strip())
            pat = (pat.replace(re.escape("<<v>>"), f"(?P<v>{NUM})").replace(re.escape("<<ver>>"), r"(?P<v>\d+(?:\.\d+)+)")
                      .replace(re.escape("<<n>>"), NUM).replace(re.escape("<<*>>"), r".{0,60}?"))
            ms = list(re.finditer(pat, norm))
            need = int(e.get("occurrences", 1))
            if len(ms) != need:
                res.update(status="NO ENCONTRADA", where=f"ancla {e['anchor'][:70]!r}: {len(ms)} coincidencias (esperadas {need})"); results.append(res); continue
            vals, wheres, spans = [], [], []
            for m in ms:
                a, b = idx[m.start("v")], idx[m.end("v") - 1] + 1; spans.append([a, b])
                for tk in ptoks:
                    if tk["start"] <= a and b <= tk["end"]: claimed_prose.add((tk["start"], tk["end"]))
                vals.append(m.group("v")); wheres.append(f"l.{line_of(text, a)} [{nearest(a)}] …{re.sub(r'[ ]+', ' ', text[max(0, a - 40):b + 40]).strip()}…")
            res["tex"] = vals if need > 1 else vals[0]; res["where"] = " | ".join(wheres); res["span"] = spans
        if value is None:
            res["status"] = "NO ENCONTRADA"; results.append(res); continue
        if "const" in e:
            sts = [compare(v, value, e.get("fmt"), e.get("tol"))[0] for v in (res["tex"] if isinstance(res["tex"], list) else [res["tex"]])]
            res["status"] = "CONST" if all(s.startswith("OK") for s in sts) else "DISCREPANCIA"
            res["expected"] = str(value); results.append(res); continue
        sts = [compare(v, value, e.get("fmt"), e.get("tol")) for v in (res["tex"] if isinstance(res["tex"], list) else [res["tex"]])]
        res["expected"] = sts[0][1]
        res["status"] = "DISCREPANCIA" if any(s[0] == "DISCREPANCIA" for s in sts) else ("OK~" if any(s[0] == "OK~" for s in sts) else "OK")
        results.append(res)

    # ---- coverage -------------------------------------------------------------------------------
    universe, uncovered = [], []
    for t in tables:
        for r in t["rows"]:
            for ci, c in enumerate(r["clean"]):
                for pi, n in enumerate(numbers_in(c)):
                    universe.append(1)
                    if (t["label"], r["line"], ci, pi) not in claimed_cells:
                        uncovered.append(f"{t['label']} l.{r['line']} fila {r['clean'][0][:35]!r} col {ci + 1} part {pi}: {n}")
    for tk in ptoks:
        universe.append(1)
        if (tk["start"], tk["end"]) not in claimed_prose:
            uncovered.append(f"l.{tk['line']} [{tk['section']}] {tk['kind']} {tk['value']}: …{tk['context'].strip()}…")
    cov = 1 - len(uncovered) / max(1, len(universe))
    return {"tex": str(tex_path), "manifest": str(manifest_path), "results": results, "n_tokens": len(universe),
            "n_uncovered": len(uncovered), "coverage": cov, "uncovered": uncovered, "tables": [t["label"] for t in tables]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--report", type=Path, default=None, help="write the full JSON report here")
    ap.add_argument("--min-coverage", type=float, default=None, help="exit 1 if coverage below this fraction")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if not args.tex.exists(): sys.exit(f"manuscript not found: {args.tex}  (pass --tex PATH)")
    if not args.manifest.exists(): sys.exit(f"manifest not found: {args.manifest}")
    rep = verify(args.tex, args.manifest)
    R = rep["results"]
    counts = {s: sum(1 for r in R if r["status"] == s) for s in ("OK", "OK~", "CONST", "DISCREPANCIA", "NO ENCONTRADA")}
    print(f"tex: {rep['tex']}\nmanifest: {rep['manifest']}\ntablas: {', '.join(rep['tables'])}")
    print(f"entradas del manifiesto: {len(R)}  ->  OK {counts['OK']}  OK~(borde de redondeo) {counts['OK~']}  "
          f"CONST(no verificada contra fuente) {counts['CONST']}  DISCREPANCIA {counts['DISCREPANCIA']}  NO ENCONTRADA {counts['NO ENCONTRADA']}")
    for r in R:
        if r["status"] in ("DISCREPANCIA", "NO ENCONTRADA") or (r["status"] == "OK~" and not args.quiet):
            print(f"  ✗ [{r['status']}] {r['id']}: tex={r['tex']} esperado={r['expected']}  @ {r['where']}  <- {r['source']}")
    noted = [r for r in R if r.get("note")]
    if noted and not args.quiet:
        print(f"entradas con nota de procedencia ({len(noted)}):")
        for r in noted: print(f"  ! [{r['status']}] {r['id']}: {r['note']}")
    print(f"cobertura: {rep['n_tokens'] - rep['n_uncovered']}/{rep['n_tokens']} cifras numéricas del .tex reclamadas por el manifiesto = {100 * rep['coverage']:.1f} %")
    if not args.quiet:
        for u in rep["uncovered"]: print("  · sin manifiesto:", u)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(args.report, "w"), indent=1, ensure_ascii=False, default=str)
        print(f"informe: {args.report}")
    fail = counts["DISCREPANCIA"] + counts["NO ENCONTRADA"] > 0
    if args.min_coverage is not None and rep["coverage"] < args.min_coverage: fail = True
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
