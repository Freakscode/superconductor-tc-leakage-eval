"""H01 - design effect for an EXPLICIT estimator, not a universal "effective N".

For Dataset A, Dataset B v2 (comparison) and Dataset B v3 (canonical) computes
  ICC(1)  one-way ANOVA intraclass correlation of Tc, rho = (MSB - MSW) / (MSB + (m0 - 1) MSW),
  m_bar   mean family size N/k,
  m0      Searle's size  (N - sum n_i^2 / N) / (k - 1),
  m_w     sum n_i^2 / N  (the size that enters the variance of the ROW-weighted mean),
and, under the random-intercept model (independent families, common marginal variance sigma^2,
intra-family correlation rho), the design effect deff = Var(estimator) / (sigma^2/N) and N_eff = N/deff
for two estimators of the population mean of Tc:
  (i)  row-weighted mean  (every row weight 1):     deff = 1 + rho (m_w - 1)
  (ii) mean of family means (row weight 1/n_i):     deff = (N/k^2) sum_i [rho + (1 - rho)/n_i]
General weights w_i per row of family i: Var = sigma^2 sum_i n_i w_i^2 [(1-rho) + rho n_i] / (sum_i n_i w_i)^2.
Also the manuscript's formula deff = 1 + rho (m0 - 1) (labelled as such) and sensitivities: without the
largest family; weights 1/n_i (== estimator ii); weights 1/sqrt(n_i).

The row of the .csv is (dataset, estimator/variant). Nothing here is the effective N of the LEARNING
problem or of the MAE: it is the variance inflation of one estimator of a mean under a model.

    python code/v3/design_effect_v3.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h07_common import RES_V2, RES_V3, VERS, load_dataset

PROC = (f"code/v3/design_effect_v3.py | {VERS} | ICC(1) one-way ANOVA (Searle m0); model = random intercept, independent "
        f"families, common marginal variance, intra-family correlation rho; deff = Var(estimator)/(sigma^2/N), N_eff = N/deff; "
        f"family = constituent-element set; A = data/unique_m.csv, B_v2 = data/datasetB_featurized.npz, B_v3 = data/datasetB_featurized_v3.npz")


def anova(y, g):
    codes, uniq = pd.factorize(np.asarray(g)); k = len(uniq); N = len(y)
    n = np.bincount(codes, minlength=k).astype(float)
    fam_mean = np.bincount(codes, weights=y, minlength=k) / n
    grand = y.mean()
    ssb = (n * (fam_mean - grand) ** 2).sum(); ssw = ((y - fam_mean[codes]) ** 2).sum(); sst = ((y - grand) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    m0 = (N - (n ** 2).sum() / N) / (k - 1)
    rho = (msb - msw) / (msb + (m0 - 1) * msw)
    return dict(N=N, k=k, n=n, m_bar=N / k, m0=m0, m_w=(n ** 2).sum() / N, ICC=rho, eta2=ssb / sst, MSB=msb, MSW=msw,
                max_family=int(n.max()), pct_rows_largest=100 * n.max() / N, n_singletons=int((n == 1).sum()))


def deff_weighted(n, rho, w):
    """w = weight of each ROW of family i (one value per family). Var / (sigma^2/N)."""
    N = n.sum()
    var = (n * w ** 2 * ((1 - rho) + rho * n)).sum() / (n * w).sum() ** 2   # in units of sigma^2
    return var * N


def rows_for(ds, y, g, label):
    A = anova(y, g); n, rho = A["n"], A["ICC"]
    base = {k: v for k, v in A.items() if k != "n"}; base.update(dataset=ds, variant=label)
    out = []
    ests = {
        "(i) media ponderada por filas (peso 1)": (deff_weighted(n, rho, np.ones_like(n)), "1 + rho (m_w - 1)"),
        "(ii) media de medias de familia (peso 1/n_i)": (deff_weighted(n, rho, 1 / n), "(N/k^2) sum_i [rho + (1-rho)/n_i]"),
        "peso 1/sqrt(n_i)": (deff_weighted(n, rho, 1 / np.sqrt(n)), "general weighted formula"),
        "formula del manuscrito 1 + rho (m0 - 1) [no corresponde a un estimador definido]": (1 + rho * (A["m0"] - 1), "1 + rho (m0 - 1)"),
        "formula 1 + rho (m_bar - 1) [no corresponde a un estimador definido]": (1 + rho * (A["m_bar"] - 1), "1 + rho (m_bar - 1)"),
    }
    # closed-form checks
    assert abs(ests["(i) media ponderada por filas (peso 1)"][0] - (1 + rho * (A["m_w"] - 1))) < 1e-9
    assert abs(ests["(ii) media de medias de familia (peso 1/n_i)"][0] - (A["N"] / A["k"] ** 2) * (rho + (1 - rho) / n).sum()) < 1e-9
    for est, (deff, formula) in ests.items():
        out.append({**base, "estimador": est, "formula": formula, "deff": deff, "N_eff": A["N"] / deff,
                    "N_eff_over_k": A["N"] / deff / A["k"], "N_eff_limit_rho1": (A["N"] ** 2 / (n ** 2).sum()) if est.startswith("(i)") else (A["k"] if est.startswith("(ii)") else np.nan)})
    return out


def main():
    rows = []
    for ds in ("A", "B_v2", "B_v3"):
        X, y, g, _, meta = load_dataset(ds)
        rows += rows_for(ds, y, g, "todas las familias")
        # sensitivity: drop the largest family
        codes, uniq = pd.factorize(g); n = np.bincount(codes); big = np.argmax(n)
        keep = codes != big
        rows += rows_for(ds, y[keep], g[keep], f"sin la familia mayor (n={n[big]})")
        print(f"{ds}: N={len(y)} k={len(uniq)} largest family n={n[big]}", flush=True)
    D = pd.DataFrame(rows)
    # reference: results_v2/design_effect.csv (A and B v2)
    ref = pd.read_csv(RES_V2 / "design_effect.csv").set_index("dataset")
    for i, r in D.iterrows():
        key = {"A": "A", "B_v2": "B"}.get(r.dataset)
        if key and r.variant == "todas las familias":
            D.loc[i, "ICC_v2_ref"] = ref.loc[key, "ICC"]; D.loc[i, "m0_v2_ref"] = ref.loc[key, "m0_searle"]; D.loc[i, "N_eff_searle_v2_ref"] = ref.loc[key, "N_eff_searle"]
    D["procedencia"] = PROC + " | *_v2_ref = results_v2/design_effect.csv"
    RES_V3.mkdir(parents=True, exist_ok=True)
    D.to_csv(RES_V3 / "design_effect_v3.csv", index=False)
    cols = ["dataset", "variant", "N", "k", "ICC", "m_bar", "m0", "m_w", "estimador", "deff", "N_eff", "N_eff_over_k", "N_eff_limit_rho1"]
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(D[cols].to_string())
    print(f"wrote {RES_V3 / 'design_effect_v3.csv'}")


if __name__ == "__main__":
    main()
