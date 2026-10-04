"""Tutor comment 5: data-flow table (row counts at each filtering step) and the Tc-criterion statement.

Inserts table tab:flow before the semantic-audit subsection and appends its cell entries to
code/v3/manuscript_numbers_v31.json. Each cell reuses the value specification that the main manifest
already uses for the same count in the prose (copied from the entry named in `_from`).
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX, M31, MAIN = ROOT / "paper1_manuscript.tex", ROOT / "code/v3/manuscript_numbers_v31.json", ROOT / "code/v3/manuscript_numbers.json"
ROWS = [  # (first cell, rows-entry id in main manifest, systems-entry id or None)
    ("Dataset A as distributed by UCI", "abs_N_A", "abs_k_A"),
    ("Dataset B: Stanev list of SuperCon entries", "data_B_source", None),
    ("B: formula not parseable", "data_parse_fail", None),
    ("B: recorded Tc of zero or below", "data_B_tcle0", None),
    ("B: retained before the audit", "data_B_v2N", None),
    ("B: excluded by the semantic audit", "data_excl", None),
    ("Dataset B, audited checkpoint", "abs_N_B", "abs_k_B"),
    ("Temporal corpus: primary SuperCon reports", "tc_raw", None),
    ("Temporal: dated reports retained", "tc_reports", None),
    ("Temporal: compositions, one row each", "abs_ncomp", None),
]


def main():
    tex = TEX.read_text(); assert "tab:flow" not in tex
    main_e = {e["id"]: e for e in json.load(open(MAIN))["entries"]}
    sys.path.insert(0, str(ROOT / "code/v3"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("vm", ROOT / "code/v3/verify_manuscript.py")
    vm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)
    lines, entries = [], []
    for i, (label, rid, sid) in enumerate(ROWS):
        cells = [label]
        for col, eid in ((2, rid), (3, sid)):
            if eid is None:
                cells.append("{---}"); continue
            src = {k: v for k, v in main_e[eid].items() if k not in ("id", "anchor", "occurrences", "note")}
            v, _ = vm.resolve(src)
            cells.append(f"\\num{{{format(float(v), '.0f')}}}")
            entries.append({"id": f"v31_flow_{i}_{col}", "table": "tab:flow", "row": label, "col": col, "part": 0,
                            **src, "fmt": ".0f", "note": f"value specification copied from main-manifest entry {eid}"})
        lines.append(" & ".join(cells) + " \\\\")
        if label.startswith("Dataset A") or label.startswith("Dataset B, audited"):
            lines.append("\\midrule")
    table = ("\\begin{table}[t]\n\\caption{\\label{tab:flow}Row counts at each filtering step. Systems are element "
             "systems (identical constituent-element sets). The temporal corpus collapses the dated reports of one "
             "composition into a single row (section~\\ref{sec:temporal-data}).}\n\\centering\\small\n"
             "\\begin{tabular}{@{}lrr@{}}\n\\toprule\nStep & Rows & Systems \\\\\n\\midrule\n" + "\n".join(lines) +
             "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    text = ("Table~\\ref{tab:flow} gives the number of rows at each filtering step. None of the files used here "
            "records the criterion behind a $\\Tc$ (resistive onset, zero resistance or magnetic transition) or the "
            "measurement conditions, so the target throughout is the $\\Tc$ recorded in the database. Reports of "
            "one composition under different criteria enter Dataset~B as separate rows; in the temporal corpus "
            "they are collapsed to the median $\\Tc$ of the reports, with the maximum as a sensitivity check "
            "(section~\\ref{sec:temporal}).\n\n")
    m = re.search(r"\\subsection\{Semantic audit of the corpus\}", tex); assert m
    tex = tex[:m.start()] + text + table + tex[m.start():]
    TEX.write_text(tex)
    m31 = json.load(open(M31)); m31["entries"] += entries; json.dump(m31, open(M31, "w"), indent=1, ensure_ascii=False)
    print(f"tab:flow inserted, {len(entries)} cell entries")


if __name__ == "__main__":
    sys.exit(main())
