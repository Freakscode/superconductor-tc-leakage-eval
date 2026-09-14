#!/usr/bin/env python
"""Resolve the DOIs of the new prior-work references via CrossRef and emit
IOP-style \\bibitem entries matching the style of paper1_manuscript.tex.

Every bibliographic field (authors, year, title, journal, volume, pages/article
number) is taken verbatim from the CrossRef record; nothing is typed by hand.

Outputs (in <SUPERCON_ROOT>/results_v3/):
  refs_crossref_v3.json  -- raw normalised CrossRef fields + procedencia
  bibitems_v3.tex        -- \\bibitem blocks ready to paste into thebibliography

Usage:  python fetch_refs_crossref.py            (paths resolved from __file__)
        SUPERCON_ROOT=/path/to/Superconductors python fetch_refs_crossref.py
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import os
import re
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]            # superconductor-tc-leakage-eval
ROOT = Path(os.environ.get("SUPERCON_ROOT", REPO.parent))  # .../Superconductors
OUT = ROOT / "results_v3"
OUT.mkdir(parents=True, exist_ok=True)

# key -> DOI. Keys follow the manuscript's convention (firstauthorYEAR).
DOIS = {
    "sommer2023":    "10.1038/s41597-023-02721-y",
    "meredig2018":   "10.1039/C8ME00012C",
    "vasylenko2023": "10.1038/s41524-023-01072-x",
    "stanev2018":    "10.1038/s41524-018-0085-8",
    "hamidieh2018":  "10.1016/j.commatsci.2018.07.052",
    "xiong2020":     "10.1016/j.commatsci.2019.109203",
    "sheridan2013":  "10.1021/ci400084k",
    "roberts2017":   "10.1111/ecog.02881",
    "gupta1993":     "10.1016/0921-4534(93)91019-R",
}

# Journal abbreviations in the manuscript's house style (IOP). Only used for
# display; the full CrossRef container-title is kept in the JSON.
JOURNAL_ABBREV = {
    "Scientific Data": "Sci. Data",
    "Molecular Systems Design & Engineering": "Mol. Syst. Des. Eng.",
    "npj Computational Materials": "npj Comput. Mater.",
    "Computational Materials Science": "Comput. Mater. Sci.",
    "Journal of Chemical Information and Modeling": "J. Chem. Inf. Model.",
    "Ecography": "Ecography",
    "Physica C: Superconductivity": "Physica C",
}


def _clean_title(t: str, key: str = "") -> str:
    """Typographic normalisation only -- no words are added or removed."""
    t = re.sub(r"<mml:math.*?</mml:math>", "$k$", t, flags=re.S)   # Xiong: MathML k
    t = re.sub(r"\$k\$\s+-", "$k$-", t)
    t = re.sub(r"\s+", " ", t).strip().rstrip(".")
    t = html.unescape(t).replace("‐", "-")                           # &amp;, U+2010 hyphen
    if key == "gupta1993":                                           # YBa2Cu3O7 -> YBa$_2$Cu$_3$O$_7$
        t = re.sub(r"([A-Z][a-z]?)(\d+)", r"\1$_{\2}$", t)
    if key == "sheridan2013":                                        # CrossRef stores Title Case; house style is sentence case
        t = t[0] + t[1:].lower()
    return t


def _initials(given: str) -> str:
    parts = re.split(r"[\s\-]+", given.replace(".", " ").strip())
    return " ".join(p[0] for p in parts if p)


def _author_str(authors: list[dict]) -> str:
    names = []
    for a in authors:
        fam = a.get("family", "").replace("‐", "-")
        giv = a.get("given", "")
        names.append(f"{fam} {_initials(giv)}".strip())
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _pages(m: dict) -> str:
    page = m.get("page")
    if page and "-" in page:
        a, b = page.split("-", 1)
        # IOP style compresses the end page: 346-354 -> 346--54
        if len(b) == len(a):
            k = 0
            while k < len(a) - 1 and a[k] == b[k]:
                k += 1
            b = b[k:]
        return f"{a}--{b}"
    if page:
        return page
    return m.get("article-number", "")


def fetch(doi: str) -> dict:
    r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=60)
    r.raise_for_status()
    return r.json()["message"]


def main() -> int:
    records, bib = {}, []
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    for key, doi in DOIS.items():
        m = fetch(doi)
        issued = (m.get("published-print") or m.get("published-online") or m["issued"])["date-parts"][0]
        journal = html.unescape((m.get("container-title") or [""])[0])
        rec = {
            "doi": m["DOI"],
            "title": _clean_title(m["title"][0], key),
            "authors": [{"family": a.get("family", ""), "given": a.get("given", "")} for a in m.get("author", [])],
            "journal": journal,
            "journal_abbrev": JOURNAL_ABBREV.get(journal, journal),
            "volume": m.get("volume"),
            "issue": m.get("issue"),
            "page": m.get("page"),
            "article_number": m.get("article-number"),
            "year": issued[0],
            "publisher": m.get("publisher"),
            "procedencia": {"fuente": "api.crossref.org/works/<doi>", "consultado_utc": stamp,
                            "script": "code/v3/fetch_refs_crossref.py"},
        }
        records[key] = rec
        entry = (f"\\bibitem{{{key}}}\n"
                 f"{_author_str(m.get('author', []))} {rec['year']} {rec['title']}\n"
                 f"\\textit{{{rec['journal_abbrev']}}} \\textbf{{{rec['volume']}}} {_pages(m)}\n"
                 f"\\url{{https://doi.org/{rec['doi']}}}\n")
        bib.append(entry)
        print(f"{key:14s} {rec['year']} {rec['journal_abbrev']} {rec['volume']} {_pages(m)}")

    (OUT / "refs_crossref_v3.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))
    header = ("% Generated by code/v3/fetch_refs_crossref.py from CrossRef records; do not hand-edit.\n"
              f"% Consulted {stamp}. Keys already present in paper1_manuscript.tex (hamidieh2018,\n"
              "% stanev2018, meredig2018) are regenerated here for cross-checking only.\n\n")
    (OUT / "bibitems_v3.tex").write_text(header + "\n".join(bib))
    print("wrote", OUT / "refs_crossref_v3.json", "and", OUT / "bibitems_v3.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
