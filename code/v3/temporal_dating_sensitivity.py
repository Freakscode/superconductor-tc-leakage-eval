"""Sensitivity of the temporal validation to fallback-dated compositions, at every cut.

temporal_validation.py runs the "sin_anio_ambiguo" scenario (compositions whose first-report year
came from a fallback dating rule removed) only at the principal cut. This driver reuses its own
run_cut() to run both scenarios at all four cuts in one process, checks that the principal scenario
reproduces results_v3/temporal_validation.csv, and writes
results_v3/temporal_dating_sensitivity.csv. It does not modify any existing result file.

Usage: python code/v3/temporal_dating_sensitivity.py [--cuts 1995 2005 2010 2015]
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, nargs="+", default=[1995, 2005, 2010, 2015])
    a = ap.parse_args(argv)
    tv = load("temporal_validation", HERE / "temporal_validation.py")
    dbp = tv.load_mod("datasetB_pipeline", tv.CODE / "datasetB_pipeline.py")
    z = np.load(tv.DATA / "temporal_corpus_v3.npz", allow_pickle=True)
    X, y, fam, year, ck, amb = (z["X"].astype(np.float32), z["y"], z["groups"], z["year"],
                                z["comp_key"].astype(str), z["anio_ambiguo"].astype(bool))
    keep = ~amb
    rows, cmp_rows, preds, meta = [], [], [], {"n_ambiguous_total": int(amb.sum())}
    t0 = time.time()
    for T in a.cuts:
        tv.YEAR, tv.FAM = year, fam
        meta[f"principal_T{T}"] = tv.run_cut(T, X, y, fam, ck, dbp, "principal", rows, cmp_rows, preds)
        tv.YEAR, tv.FAM = year[keep], fam[keep]
        meta[f"sin_anio_ambiguo_T{T}"] = tv.run_cut(T, X[keep], y[keep], fam[keep], ck[keep], dbp,
                                                     "sin_anio_ambiguo", rows, cmp_rows, preds)
        meta[f"n_ambiguous_train_T{T}"] = int((amb & (year < T)).sum())
        meta[f"n_ambiguous_test_T{T}"] = int((amb & (year >= T)).sum())
        print(f"cut {T} done, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(subset=["corte", "escenario", "condicion", "modelo", "estrato", "metrica"])

    # the principal scenario must reproduce the canonical file exactly
    canon = pd.read_csv(tv.RESULTS_V3 / "temporal_validation.csv")
    key = ["corte", "escenario", "condicion", "modelo", "estrato", "metrica"]
    m = df[df.escenario == "principal"].merge(canon[canon.escenario == "principal"], on=key, suffixes=("", "_canon"))
    max_dev = float((m.valor - m.valor_canon).abs().max())
    meta["principal_vs_canonical"] = {"n_matched": int(len(m)), "max_abs_dev": max_dev}
    assert len(m) > 0 and max_dev < 1e-9, f"principal scenario does not reproduce canonical ({len(m)} rows, dev {max_dev})"

    prov = tv.procedencia(dict(script="code/v3/temporal_dating_sensitivity.py", segundos_total=round(time.time() - t0, 1), meta=meta))
    df["procedencia"] = json.dumps(prov, ensure_ascii=False)
    out = tv.RESULTS_V3 / "temporal_dating_sensitivity.csv"
    df.to_csv(out, index=False)
    print(json.dumps(meta, indent=1))
    print(f"wrote {out} ({len(df)} rows), total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
