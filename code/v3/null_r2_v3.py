"""Why the grouped family-mean null has R2 slightly BELOW zero (-0.007 in A, -0.003 in B): not "by definition".

Under GroupKFold every test family is unseen, so the null predicts the TRAIN-fold mean c_f in fold f.
R2 = 1 - SSE/SST with SST computed around the mean of the pooled OOF target. Per fold,
    SSE_f = sum_{i in f} (y_i - c_f)^2 = SSW_f + n_f (ybar_f - c_f)^2 ,   SSW_f = sum (y_i - ybar_f)^2
so R2 is exactly zero only when each c_f equals the test-fold mean ybar_f; the (deterministic) mismatch
between train and test means is what makes it negative. This script measures, per fold and pooled,
c_f, ybar_f, n_f (ybar_f - c_f)^2, the per-fold R2 (test mean as reference: R2_f = -n_f(ybar_f-c_f)^2/SSW_f)
and the exact decomposition of the pooled R2 for Datasets A and B v3 (and the row-mean null for reference).
Input: results_v3/oof_v3.csv.gz (FamilyMeanNull, grouped). Output: results_v3/null_r2_by_fold.csv (+ .md).

    python code/v3/null_r2_v3.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h07_common import RES_V3, VERS

PROC = (f"code/v3/null_r2_v3.py | {VERS} | input results_v3/oof_v3.csv.gz (FamilyMeanNull, grouped = GroupKFold(5) det.; "
        f"under GroupKFold every test family is unseen so the prediction is the TRAIN-fold mean) | R2 = 1 - SSE/SST, SST around the pooled OOF mean")


def main():
    oof = pd.read_csv(RES_V3 / "oof_v3.csv.gz")
    rows, L = [], ["# R² del nulo agrupado: por qué es ligeramente negativo", "", f"Procedencia: `{PROC}`", ""]
    for ds in ("A", "B"):
        o = oof[(oof.dataset == ds) & (oof.model == "FamilyMeanNull") & (oof.protocol == "grouped")].sort_values("row_index")
        y, yp, fold = o.y_true.values, o.y_pred.values, o.fold.values
        ybar = y.mean(); sst = ((y - ybar) ** 2).sum(); sse = ((y - yp) ** 2).sum()
        r2 = 1 - sse / sst
        ssw_tot, shift_tot, between_tot = 0.0, 0.0, 0.0
        for f in range(5):
            m = fold == f; yf = y[m]; cf = np.unique(yp[m])
            assert len(cf) == 1, f"{ds} fold {f}: the grouped null is not constant within the fold ({len(cf)} values)"
            cf = cf[0]; nf = m.sum(); ybf = yf.mean(); ssw = ((yf - ybf) ** 2).sum(); shift = nf * (ybf - cf) ** 2
            ssw_tot += ssw; shift_tot += shift; between_tot += nf * (ybf - ybar) ** 2
            rows.append(dict(dataset=ds, fold=f, n_test=int(nf), c_train_mean=cf, ybar_test=ybf, diff_K=ybf - cf,
                             SSW_test=ssw, n_x_diff2=shift, SSE_fold=ssw + shift, R2_fold_vs_test_mean=-shift / ssw,
                             R2_fold_vs_pooled_mean=1 - (ssw + shift) / ((yf - ybar) ** 2).sum()))
        assert abs((ssw_tot + shift_tot) - sse) < 1e-6 and abs((ssw_tot + between_tot) - sst) < 1e-6
        rows.append(dict(dataset=ds, fold="pooled", n_test=len(y), c_train_mean=np.nan, ybar_test=ybar, diff_K=np.nan,
                         SSW_test=ssw_tot, n_x_diff2=shift_tot, SSE_fold=sse, R2_fold_vs_test_mean=np.nan, R2_fold_vs_pooled_mean=r2,
                         SST_pooled=sst, SS_between_folds=between_tot, R2_if_c_equal_test_mean=1 - ssw_tot / sst,
                         R2_decomposition=f"1 - (SSW {ssw_tot:.1f} + shift {shift_tot:.1f}) / SST {sst:.1f}"))
        L += [f"## Dataset {ds}", "", f"R² pooled del nulo agrupado = {r2:.5f}. Descomposición exacta: R² = 1 − (Σ_f SSW_f + Σ_f n_f(ȳ_f − c_f)²)/SST = 1 − ({ssw_tot:.0f} + {shift_tot:.0f})/{sst:.0f}.",
              f"Si cada constante c_f coincidiera con la media del test del fold, R² sería 1 − ΣSSW/SST = {1 - ssw_tot / sst:+.5f} (SST incluye la varianza ENTRE folds de las medias de test, {between_tot:.0f}). "
              f"El término de desajuste Σ n_f(ȳ_f − c_f)² = {shift_tot:.0f} es lo que empuja el R² por debajo de cero.", "",
              "| fold | n_test | c_f (media train) | ȳ_f (media test) | ȳ_f − c_f (K) | n_f(ȳ_f−c_f)² | R²_f (ref. media test) |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in [x for x in rows if x["dataset"] == ds and x["fold"] != "pooled"]:
            L.append(f"| {r['fold']} | {r['n_test']} | {r['c_train_mean']:.3f} | {r['ybar_test']:.3f} | {r['diff_K']:+.3f} | {r['n_x_diff2']:.0f} | {r['R2_fold_vs_test_mean']:+.5f} |")
        L.append("")
    D = pd.DataFrame(rows); D["procedencia"] = PROC; D.to_csv(RES_V3 / "null_r2_by_fold.csv", index=False)
    (RES_V3 / "null_r2_by_fold.md").write_text("\n".join(L) + "\n")
    pd.set_option("display.width", 250); print(D.drop(columns=["procedencia", "R2_decomposition"], errors="ignore").to_string())
    print(f"wrote {RES_V3 / 'null_r2_by_fold.csv'} and .md")


if __name__ == "__main__":
    main()
