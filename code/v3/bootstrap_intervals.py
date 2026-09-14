"""Versioned driver for the bootstrap intervals of section 3.2 (results_v2/bootstrap_intervalos.csv
had no generator in the repository: it was produced in an interactive cell).

Reads results_v3/oof_predictions.csv.gz (code/v3/dump_oof_predictions.py) and, for every
(dataset, model) with both protocols at the same seed, bootstraps the pooled out-of-fold
absolute errors over ROWS (n_boot resamples, percentile 2.5/97.5) in two schemes:

  unpaired : independent row resamples for the random arm and for the grouped arm; the
             inflation = MAE_grouped*/MAE_random* - 1 is formed from two independent
             resamples.  This is the scheme that reproduces results_v2/bootstrap_intervalos.csv
             within Monte-Carlo error (its exact RNG stream is not recoverable).
  paired   : ONE row resample applied to both arms (the two predictions of a row travel
             together).  This is the appropriate scheme for a within-row contrast and gives a
             much narrower inflation interval; it is reported alongside.

Writes results_v3/bootstrap_intervalos_v3.csv (one row per dataset x model x scheme) with the
v2 values, the difference, and the Monte-Carlo standard deviation of each endpoint estimated
from `--mc-reps` extra seeds, so that "reproduces within MC error" is a measured statement.

    python code/v3/bootstrap_intervals.py [--n-boot 2000] [--seed 0] [--mc-reps 20]
"""
from __future__ import annotations
import argparse, os, platform
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
RES_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
COLS = ["MAE_random_lo", "MAE_random_hi", "MAE_grouped_lo", "MAE_grouped_hi", "inflacion_lo", "inflacion_hi"]


def boot(er, eg, seed, n_boot, paired):
    rng = np.random.default_rng(seed); n = len(er)
    mr, mg = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n); mr[b] = er[i].mean()
        j = i if paired else rng.integers(0, n, n); mg[b] = eg[j].mean()
    inf = 100 * (mg / mr - 1)
    return np.r_[np.percentile(mr, [2.5, 97.5]), np.percentile(mg, [2.5, 97.5]), np.percentile(inf, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, default=RES_V3 / "oof_predictions.csv.gz")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc-reps", type=int, default=20, help="extra seeds used to estimate the Monte-Carlo sd of each endpoint")
    ap.add_argument("--out", type=Path, default=RES_V3 / "bootstrap_intervalos_v3.csv")
    args = ap.parse_args()

    oof = pd.read_csv(args.oof)
    v2 = pd.read_csv(ROOT / "results_v2" / "bootstrap_intervalos.csv").set_index("dataset")
    proc = (f"code/v3/bootstrap_intervals.py | input {args.oof.name} (code/v3/dump_oof_predictions.py) | bootstrap of ROWS of the pooled "
            f"out-of-fold |y - yhat|, n_boot={args.n_boot}, percentile CI 2.5/97.5, numpy.random.default_rng({args.seed}).integers; "
            f"MC sd from seeds {args.seed + 1}..{args.seed + args.mc_reps} | python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}")
    rows = []
    for (ds, model, seed), g in oof.groupby(["dataset", "model", "seed"]):
        r = g[g.protocol == "random"].sort_values("row_index"); q = g[g.protocol == "grouped"].sort_values("row_index")
        if len(r) == 0 or len(q) == 0: continue
        assert np.array_equal(r.row_index.values, q.row_index.values) and np.array_equal(r.y_true.values, q.y_true.values)
        er, eg = np.abs(r.y_true.values - r.y_pred.values), np.abs(q.y_true.values - q.y_pred.values)
        fold_mae = pd.Series(eg).groupby(q.fold.values).mean()
        for scheme, paired in (("unpaired", False), ("paired", True)):
            ci = boot(er, eg, args.seed, args.n_boot, paired)
            mc = np.array([boot(er, eg, s, args.n_boot, paired) for s in range(args.seed + 1, args.seed + 1 + args.mc_reps)]) if args.mc_reps else None
            row = {"dataset": ds, "model": model, "model_seed": seed, "scheme": scheme, "n": len(er), "n_boot": args.n_boot, "boot_seed": args.seed,
                   "MAE_random": er.mean(), "MAE_grouped": eg.mean(), "inflacion_pct": 100 * (eg.mean() / er.mean() - 1),
                   "fold_grouped_min": fold_mae.min(), "fold_grouped_max": fold_mae.max()}
            row.update(dict(zip(COLS, ci)))
            if mc is not None:
                row.update({f"{c}_mc_sd": s for c, s in zip(COLS, mc.std(0))})
            if model == "XGBoost" and ds in v2.index:               # the v2 file is XGBoost, seed 42
                for c in COLS:
                    row[f"{c}_v2"] = v2.loc[ds, c]; row[f"{c}_diff_vs_v2"] = ci[COLS.index(c)] - v2.loc[ds, c]
                    if mc is not None: row[f"{c}_z_vs_v2"] = (v2.loc[ds, c] - mc.mean(0)[COLS.index(c)]) / mc.std(0)[COLS.index(c)]
                row["point_estimates_equal_v2"] = bool(np.allclose([er.mean(), eg.mean(), fold_mae.min(), fold_mae.max()],
                                                                    v2.loc[ds, ["MAE_random", "MAE_grouped", "fold_grouped_min", "fold_grouped_max"]].values.astype(float), atol=1e-9))
            row["procedencia"] = proc
            rows.append(row)
            print(f"{ds} {model:12s} {scheme:8s} MAE_r [{ci[0]:.3f},{ci[1]:.3f}] MAE_g [{ci[2]:.3f},{ci[3]:.3f}] infl [{ci[4]:.2f},{ci[5]:.2f}]"
                  + (f"  | v2 infl [{v2.loc[ds,'inflacion_lo']:.2f},{v2.loc[ds,'inflacion_hi']:.2f}]" if model == "XGBoost" and ds in v2.index else ""), flush=True)
    out = pd.DataFrame(rows); args.out.parent.mkdir(parents=True, exist_ok=True); out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
