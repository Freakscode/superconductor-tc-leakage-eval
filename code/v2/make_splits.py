"""make_splits.py -- materialise the cross-validation partitions used in the paper as DATA.

Writes  splits/datasetA_folds.csv  and  splits/datasetB_folds.csv  with columns
    row_index            0-based row of data/train.csv (A) / of data/datasetB_featurized.npz (B)
    family_id            chemical-family code (constituent-element set, pandas.factorize):
                         A: tc_pipeline.chemical_families(unique_m.csv);  B: `groups` of the .npz
    family_key           the "|"-joined sorted element symbols the code stands for
    fold_deterministic   GroupKFold(5) without shuffle          (the published protocol)
    fold_seed0..4        GroupKFold(5, shuffle=True, random_state=s), s = 0..4   (Table 2 v2)
    random_fold_seed0..4 KFold(5, shuffle=True, random_state=s), s = 0..4       (random arm)
    fold_deterministic_v1_prefilter_codes   (B only) GroupKFold(5) on the v1 gapped codes
                         (factorize BEFORE the Tc>0 filter) = the partition v1's live script used

WHY PUBLISH INDICES AND NOT ONLY THE PROCEDURE
GroupKFold assigns whole families to folds greedily by family size; ties between equally
sized families are broken by the ORDER OF THE INTEGER CODES. The codes come from
pandas.factorize, i.e. from the order in which families first appear in the data file.
Two runs that build the same partition of rows into families but number the families
differently (e.g. factorize before vs. after the Tc>0 filter of Dataset B -- the v1 bug)
get DIFFERENT fold assignments (8 937 of 12 440 Dataset-B rows change fold). The fold
indices are therefore data, not procedure, and are versioned here so that any result in
the paper can be reproduced on exactly the same held-out rows.

Usage
    python code/v2/make_splits.py            # (re)generate splits/*.csv
    python code/v2/make_splits.py --verify   # regenerate in memory and compare with the files
    python code/v2/make_splits.py --dry-run  # resolve paths, check inputs, exit
Requires data/train.csv + data/unique_m.csv (code/get_datasetA.py), data/datasetB_featurized.npz,
data/supercon_stanev.csv and pymatgen (for the Dataset-B family_key strings).
"""
from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
CODE, DATA, SPLITS = ROOT / "code", ROOT / "data", ROOT / "splits"
SEEDS = [0, 1, 2, 3, 4]
FILES = {"A": SPLITS / "datasetA_folds.csv", "B": SPLITS / "datasetB_folds.csv"}


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def fold_labels(splitter, n, groups=None):
    lab = np.full(n, -1, dtype=int)
    X = np.zeros((n, 1))
    it = splitter.split(X, None, groups) if groups is not None else splitter.split(X)
    for k, (_, te) in enumerate(it):
        lab[te] = k
    assert (lab >= 0).all()
    return lab


def build(ds: str) -> tuple[pd.DataFrame, dict]:
    if ds == "A":
        tcp = load_mod("tc_pipeline", str(CODE / "tc_pipeline.py"))
        feat = pd.read_csv(DATA / "train.csv"); uniq = pd.read_csv(DATA / "unique_m.csv")
        assert len(feat) == len(uniq) == 21263, (len(feat), len(uniq))
        assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "train/unique_m not row-aligned"
        groups = tcp.chemical_families(uniq)
        elem_cols = [c for c in uniq.columns if c not in ("critical_temp", "material")]
        pres = uniq[elem_cols].values > 0
        key = np.array(["|".join(sorted(e for e, p in zip(elem_cols, row) if p)) for row in pres])
        src = "data/train.csv rows; family = tc_pipeline.chemical_families(data/unique_m.csv)"
    else:
        z = np.load(DATA / "datasetB_featurized.npz", allow_pickle=True)
        groups = z["groups"]; y = z["y"]
        assert len(groups) == 12440, len(groups)
        from pymatgen.core import Composition
        sc = pd.read_csv(DATA / "supercon_stanev.csv")
        rows = []
        for name, tc in zip(sc["name"], sc["Tc"]):
            try:
                rows.append(("|".join(sorted(e.symbol for e in Composition(name).elements)), tc))
            except Exception:
                pass
        df_all = pd.DataFrame(rows, columns=["key", "Tc"])
        # v1 convention (bug): factorize over ALL parsed rows, THEN drop Tc<=0 -> gapped codes
        v1_codes = pd.factorize(df_all["key"])[0][(df_all["Tc"] > 0).values]
        df = df_all[df_all["Tc"] > 0].reset_index(drop=True)
        assert len(df) == len(groups) and np.array_equal(df["Tc"].values, y), "stanev.csv / npz misaligned"
        assert np.array_equal(pd.factorize(df["key"])[0], groups), "family_key factorize != npz groups"
        key = df["key"].values
        src = "data/datasetB_featurized.npz rows (Tc>0 subset of data/supercon_stanev.csv); family = `groups` field"
    assert np.array_equal(pd.factorize(pd.Series(key))[0], groups), "key order != code order"
    n = len(groups)
    out = pd.DataFrame({"row_index": np.arange(n), "family_id": groups, "family_key": key,
                        "fold_deterministic": fold_labels(GroupKFold(5), n, groups)})
    for s in SEEDS:
        out[f"fold_seed{s}"] = fold_labels(GroupKFold(5, shuffle=True, random_state=s), n, groups)
    for s in SEEDS:
        out[f"random_fold_seed{s}"] = fold_labels(KFold(5, shuffle=True, random_state=s), n)
    if ds == "B":
        # the fold assignment v1 actually used when running datasetB_pipeline.py live (same family
        # partition, gapped codes 0..4189 -> different GroupKFold tie-breaking). Kept so v1 numbers
        # remain reproducible; NOT the reference partition.
        out["fold_deterministic_v1_prefilter_codes"] = fold_labels(GroupKFold(5), n, v1_codes)
        assert out.groupby("family_id")["fold_deterministic_v1_prefilter_codes"].nunique().max() == 1
    # no family may straddle folds in any grouped column
    for c in ["fold_deterministic"] + [f"fold_seed{s}" for s in SEEDS]:
        assert out.groupby("family_id")[c].nunique().max() == 1, c
    import sklearn
    meta = dict(n_rows=n, n_families=int(len(np.unique(groups))), source=src,
                sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__)
    return out, meta


def header(ds, meta) -> str:
    return (f"# dataset {ds}: {meta['n_rows']} rows, {meta['n_families']} chemical families. {meta['source']}.\n"
            f"# fold_deterministic = GroupKFold(5) (no shuffle, published protocol); fold_seedS = GroupKFold(5, shuffle=True,\n"
            f"#   random_state=S); random_fold_seedS = KFold(5, shuffle=True, random_state=S). scikit-learn {meta['sklearn']},\n"
            f"#   pandas {meta['pandas']}, numpy {meta['numpy']}. Generated by code/v2/make_splits.py.\n"
            f"# GroupKFold breaks size ties by the ORDER of the integer family codes (pandas.factorize order of first\n"
            f"#   appearance), so the fold assignment depends on the code order, not only on the family partition.\n"
            f"#   These indices, not just the procedure, are the reference. Read with pandas.read_csv(..., comment='#').\n")


def write(ds, df, meta):
    SPLITS.mkdir(parents=True, exist_ok=True)
    with open(FILES[ds], "w") as f:
        f.write(header(ds, meta)); df.to_csv(f, index=False)
    print(f"wrote {FILES[ds]}  ({len(df)} rows)")


def verify(ds, df) -> bool:
    p = FILES[ds]
    if not p.exists():
        print(f"MISSING {p}"); return False
    old = pd.read_csv(p, comment="#")
    same = list(old.columns) == list(df.columns) and len(old) == len(df) and all(
        np.array_equal(old[c].values, df[c].values) for c in df.columns)
    print(f"{'OK      ' if same else 'MISMATCH'} {p}")
    return same


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="regenerate in memory and compare with splits/*.csv")
    ap.add_argument("--dry-run", action="store_true", help="resolve paths, check inputs, exit")
    ap.add_argument("--only", choices=["A", "B"], default=None)
    a = ap.parse_args()
    inputs = [CODE / "tc_pipeline.py", DATA / "train.csv", DATA / "unique_m.csv",
              DATA / "datasetB_featurized.npz", DATA / "supercon_stanev.csv"]
    if a.dry_run:
        print(f"ROOT={ROOT}\nSPLITS={SPLITS}")
        for p in inputs + list(FILES.values()):
            print(f"  {'ok ' if p.exists() else 'MISSING'} {p}")
        return 0
    ok = True
    for ds in ([a.only] if a.only else ["A", "B"]):
        df, meta = build(ds)
        if a.verify:
            ok &= verify(ds, df)
        else:
            write(ds, df, meta)
    if a.verify:
        print("splits verified: identical" if ok else "splits DIFFER from regeneration")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
