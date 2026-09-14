"""Versioned driver for results_v2/fix_factorize_verificacion.json (written in v2 by an interactive
"verification cell").  Recomputes, from data/supercon_stanev.csv, data/datasetB_featurized.npz and
splits/datasetB_folds.csv, the Dataset-B row-filter and fold-assignment facts quoted in the
manuscript (sections 2.1 and 4.4): 16 414 entries, 8 unparseable formulas, 3 966 with Tc <= 0,
12 440 kept; 8 937 rows (1 721-1 897 per fold) whose deterministic GroupKFold(5) fold differs
between the v1 (pre-filter) and v2 (post-filter) family codes.

Writes results_v3/fix_factorize_verificacion_v3.json and prints the comparison with the v2 file.

    python code/v3/factorize_check.py
"""
from __future__ import annotations
import json, os, platform, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
DATA, SPLITS = ROOT / "data", ROOT / "splits"
RES_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))


def family_key(formula: str):
    from pymatgen.core import Composition
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "|".join(sorted(e.symbol for e in Composition(formula).elements))
    except Exception:            # noqa  (the 8 unparseable formulas)
        return None


def fold_labels(groups):
    lab = np.full(len(groups), -1)
    for k, (_, te) in enumerate(GroupKFold(5).split(np.zeros((len(groups), 1)), groups=groups)): lab[te] = k
    return lab


def main():
    sc = pd.read_csv(DATA / "supercon_stanev.csv")
    keys = [family_key(f) for f in sc["name"]]
    ok = np.array([k is not None for k in keys])
    df = pd.DataFrame({"key": keys, "Tc": sc["Tc"].values})[ok].reset_index(drop=True)
    mask = (df["Tc"] > 0).values
    codes_pre = pd.factorize(df["key"])[0]                     # v1: factorize over all featurized rows ...
    codes_pre_masked = codes_pre[mask]                          # ... then apply the Tc>0 mask (gapped codes)
    codes_post = pd.factorize(df["key"][mask])[0]               # v2: factorize AFTER the filter (contiguous)
    z = np.load(DATA / "datasetB_featurized.npz", allow_pickle=True); groups_npz = z["groups"].astype(int)
    sp = pd.read_csv(SPLITS / "datasetB_folds.csv", comment="#")
    f_pre, f_post = fold_labels(codes_pre_masked), fold_labels(codes_post)
    out = {
        "n_rows_stanev_csv": int(len(sc)), "n_parse_failures_pymatgen": int((~ok).sum()), "n_rows_Tc_le0": int((~mask).sum()),
        "n_rows_Tc_gt0": int(mask.sum()), "n_rows_npz": int(len(groups_npz)), "y_match_npz": bool(np.allclose(df["Tc"][mask].values, z["y"])),
        "n_families_prefilter": int(codes_pre.max() + 1), "prefilter_code_range_after_mask": [int(codes_pre_masked.min()), int(codes_pre_masked.max())],
        "n_families_postfilter": int(codes_post.max() + 1), "postfilter_code_range": [int(codes_post.min()), int(codes_post.max())],
        "codes_post_array_equal_npz_groups": bool(np.array_equal(codes_post, groups_npz)),
        "codes_pre_array_equal_npz_groups": bool(np.array_equal(codes_pre_masked, groups_npz)),
        "same_family_partition_pre_vs_post": bool(pd.Series(codes_pre_masked).groupby(codes_post).nunique().max() == 1),
        "groupkfold_test_idx_post_equal_npz": bool(np.array_equal(f_post, fold_labels(groups_npz))),
        "groupkfold_post_equal_splits_fold_deterministic": bool(np.array_equal(f_post, sp["fold_deterministic"].values)),
        "groupkfold_pre_equal_splits_v1_column": bool(np.array_equal(f_pre, sp["fold_deterministic_v1_prefilter_codes"].values)),
        "rows_changing_fold_pre_vs_npz": int((f_pre != f_post).sum()),
        "rows_changing_fold_pre_vs_npz_per_fold": [int(((f_pre != f_post) & (f_post == k)).sum()) for k in range(5)],
        "fold_sizes_npz": [int((f_post == k).sum()) for k in range(5)],
        "procedencia": f"code/v3/factorize_check.py | pymatgen Composition -> sorted element symbols; pandas.factorize; sklearn GroupKFold(5) | "
                       f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}",
    }
    v2 = json.load(open(ROOT / "results_v2" / "fix_factorize_verificacion.json"))
    out["comparison_with_v2"] = {k: {"v3": out[k], "v2": v2.get(k), "equal": out[k] == v2.get(k)} for k in out if k in v2 and k != "procedencia"}
    RES_V3.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES_V3 / "fix_factorize_verificacion_v3.json", "w"), indent=1)
    for k, v in out["comparison_with_v2"].items(): print(f"{k:45s} v3={v['v3']!s:30s} v2={v['v2']!s:30s} {'OK' if v['equal'] else 'DIFF'}")
    print("all equal:", all(v["equal"] for v in out["comparison_with_v2"].values()))


if __name__ == "__main__":
    main()
