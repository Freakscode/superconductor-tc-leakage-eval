"""topk_metrics.py -- H05/H06: selection metrics per fold and aggregated, K in {10,25,50,100,200}.

Reads <RESULTS_V3>/topk_oof_predictions.csv.gz (written by topk_prevalence_ties.py) for Dataset A and
Dataset B v3, rankers null / presence / XGBoost / RandomForest, protocols random / grouped, and computes
per fold (test pool = the held-out fold, positives := y >= p90(y_train)):
    precision@K      expected fraction of positives among the top-K predicted (fractional under ties)
    recall@K         expected hits / number of positives in the pool
    enrichment@K     precision@K / prevalence of positives in the pool (= 1 for a random selector)
    n_families@K     distinct families among the top-K (mean over 200 random tie-breaks, seed 0)
    max_family_share max share of one family in the top-K (idem)
    budget precision top-K with at most 3 rows per family (idem), and its number of families
Aggregation = mean over the 5 folds (the manuscript's convention). Intervals: percentile bootstrap of
FAMILIES (2000 resamples, RandomState(0)); the same family multiset is used for every ranker/protocol of a
dataset, so differences between conditions are PAIRED. Row-level bootstrap (i.i.d. rows within each pool)
is reported alongside for precision@K to show the width difference. All intervals are conditional on the
out-of-fold predictions already fitted (they do not re-fit models).

Outputs: <RESULTS_V3>/topk_metrics.csv, topk_metrics_diffs.csv, topk_metrics.md
Usage:   python code/v3/topk_metrics.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import topk_common as tc  # noqa: E402

SCRIPT = "code/v3/topk_metrics.py"
N_BOOT, BOOT_SEED, N_TB, TB_SEED, BUDGET = 2000, 0, 200, 0, 3
DATASETS = ("A", "Bv3")
MODELS = ("null", "presence", "XGBoost", "RandomForest")
PROTOS = ("random", "grouped")
KS = list(tc.KS)


def procedencia() -> str:
    return (f"{SCRIPT} | {tc.versions()} | input topk_oof_predictions.csv.gz (seed 42; random = KFold(5, shuffle, rs=42); "
            f"grouped = GroupKFold(5) deterministic); positive := y >= np.percentile(y_train, 90); precision/recall/"
            f"enrichment fractional under ties (topk_common.expected_hits); n_families/max_share/budget = mean over {N_TB} "
            f"random tie-breaks (RandomState({TB_SEED})); budget = at most {BUDGET} rows per family; CI = percentile "
            f"bootstrap of families of the test pools, {N_BOOT} resamples, RandomState({BOOT_SEED}), same family multiset "
            f"for all conditions of a dataset (paired); CI_row = i.i.d. row bootstrap within each pool; all intervals "
            f"conditional on the fitted OOF predictions; aggregate = mean over 5 folds")


class Pool:
    """A held-out fold pool with its ranking pre-sorted (descending prediction, stable)."""
    def __init__(self, pred, hit, fam):
        self.n = len(pred); self.pred = np.asarray(pred, float); self.hit = np.asarray(hit, float); self.fam = np.asarray(fam)
        self.order = np.argsort(-self.pred, kind="stable")
        self.g = tc.tie_groups(self.pred[self.order]); self.h = self.hit[self.order]; self.G = int(self.g.max()) + 1

    def expected_hits(self, w=None):
        ww = np.ones(self.n) if w is None else np.asarray(w, float)[self.order]
        T = np.bincount(self.g, weights=ww, minlength=self.G); H = np.bincount(self.g, weights=ww * self.h, minlength=self.G)
        cumT = np.cumsum(T); prevT = cumT - T; tot = cumT[-1]
        out = np.empty(len(KS)); denom = np.empty(len(KS))
        for j, K in enumerate(KS):
            idx = np.searchsorted(cumT, K, side="right"); hits = H[:idx].sum()
            if idx < self.G:
                hits += (K - prevT[idx]) * H[idx] / T[idx]
            out[j] = hits; denom[j] = min(K, tot)
        return out, denom, tot, (ww * self.h).sum()

    def tiebreak_metrics(self, rng):
        """n_families, max_family_share, budget precision, budget n_families: means over N_TB random tie-breaks."""
        acc = np.zeros((len(KS), 4))
        for _ in range(N_TB):
            order = tc.random_tiebreak_order(self.pred, rng)
            for j, K in enumerate(KS):
                top = order[:K]; fams = self.fam[top]
                _, cnt = np.unique(fams, return_counts=True)
                sel = tc.budget_selection(order, self.fam, K, BUDGET)
                acc[j] += [len(cnt), cnt.max() / K, self.hit[sel].mean(), len(np.unique(self.fam[sel]))]
        return acc / N_TB


def main() -> int:
    t0 = time.time()
    oof = tc.read_csv(tc.OOF_CACHE)
    rows, diffs, md = [], [], []
    for ds in DATASETS:
        sub_ds = oof[oof.dataset == ds]
        fam_all = np.sort(sub_ds[(sub_ds.model == "null") & (sub_ds.protocol == "grouped")].sort_values("row_index").family_id.values)
        k_fam = int(fam_all.max()) + 1
        n_rows = len(fam_all)
        rng_boot = np.random.RandomState(BOOT_SEED); rng_row = np.random.RandomState(BOOT_SEED); rng_tb = np.random.RandomState(TB_SEED)
        # family multisets, shared across all conditions of this dataset (paired)
        fam_counts = rng_boot.multinomial(k_fam, np.full(k_fam, 1.0 / k_fam), size=N_BOOT)       # N_BOOT x k
        pools = {}
        for model in MODELS:
            for proto in PROTOS:
                s = sub_ds[(sub_ds.model == model) & (sub_ds.protocol == proto)]
                pools[(model, proto)] = [Pool(f.y_pred.values, (f.y_true.values >= f.threshold_p90_train.values).astype(float), f.family_id.values)
                                         for _, f in sorted(s.groupby("fold"), key=lambda kv: kv[0])]
        # row-bootstrap weights per pool are drawn per (proto) pool structure -> per fold sizes; draw once per proto
        row_w = {}
        for proto in PROTOS:
            sizes = [p.n for p in pools[("null", proto)]]
            row_w[proto] = [rng_row.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT) for n in sizes]
        boot_store = {}
        for (model, proto), pl in pools.items():
            # ---- point estimates per fold
            fold_stats = []
            for k, p in enumerate(pl):
                hits, denom, tot, npos = p.expected_hits()
                prev = npos / tot
                tb = p.tiebreak_metrics(rng_tb)
                for j, K in enumerate(KS):
                    fold_stats.append(dict(dataset=ds, model=model, protocol=proto, K=K, fold=k, n_test=p.n, n_pos=int(npos),
                                           prevalence_pct=100 * prev, precision_pct=100 * hits[j] / denom[j], recall_pct=100 * hits[j] / npos,
                                           enrichment=(hits[j] / denom[j]) / prev, n_families_topK=tb[j, 0], max_family_share_pct=100 * tb[j, 1],
                                           budget_precision_pct=100 * tb[j, 2], budget_n_families=tb[j, 3]))
            fs = pd.DataFrame(fold_stats)
            # ---- bootstrap (family, paired) and (row)
            fam_prec = np.empty((N_BOOT, len(KS))); fam_rec = np.empty_like(fam_prec); fam_enr = np.empty_like(fam_prec)
            row_prec = np.empty_like(fam_prec)
            for b in range(N_BOOT):
                acc_p = np.zeros(len(KS)); acc_r = np.zeros(len(KS)); acc_e = np.zeros(len(KS)); acc_rp = np.zeros(len(KS))
                for k, p in enumerate(pl):
                    w = fam_counts[b][p.fam]
                    hits, denom, tot, npos = p.expected_hits(w)
                    if tot <= 0 or npos <= 0:
                        pr = hits / np.maximum(denom, 1); acc_p += pr; acc_r += 0; acc_e += 0
                    else:
                        pr = hits / denom; acc_p += pr; acc_r += hits / npos; acc_e += pr / (npos / tot)
                    hits2, denom2, _, _ = p.expected_hits(row_w[proto][k][b]); acc_rp += hits2 / denom2
                fam_prec[b] = acc_p / 5; fam_rec[b] = acc_r / 5; fam_enr[b] = acc_e / 5; row_prec[b] = acc_rp / 5
            boot_store[(model, proto)] = dict(prec=fam_prec, rec=fam_rec, enr=fam_enr)
            for j, K in enumerate(KS):
                m = fs[fs.K == K]
                rows.append(dict(dataset=ds, model=model, protocol=proto, K=K, fold="mean_5folds", n_test=int(m.n_test.sum()), n_pos=int(m.n_pos.sum()),
                                 prevalence_pct=m.prevalence_pct.mean(), precision_pct=m.precision_pct.mean(), recall_pct=m.recall_pct.mean(),
                                 enrichment=m.enrichment.mean(), n_families_topK=m.n_families_topK.mean(), max_family_share_pct=m.max_family_share_pct.mean(),
                                 budget_precision_pct=m.budget_precision_pct.mean(), budget_n_families=m.budget_n_families.mean(),
                                 precision_sd_folds_pct=m.precision_pct.std(ddof=0),
                                 precision_ci_fam_lo_pct=100 * np.percentile(fam_prec[:, j], 2.5), precision_ci_fam_hi_pct=100 * np.percentile(fam_prec[:, j], 97.5),
                                 precision_ci_row_lo_pct=100 * np.percentile(row_prec[:, j], 2.5), precision_ci_row_hi_pct=100 * np.percentile(row_prec[:, j], 97.5),
                                 recall_ci_fam_lo_pct=100 * np.percentile(fam_rec[:, j], 2.5), recall_ci_fam_hi_pct=100 * np.percentile(fam_rec[:, j], 97.5),
                                 enrichment_ci_fam_lo=np.percentile(fam_enr[:, j], 2.5), enrichment_ci_fam_hi=np.percentile(fam_enr[:, j], 97.5)))
            rows.extend(fs.to_dict("records"))
            print(f"[{ds}] {model}/{proto}: P@100={fs[fs.K==100].precision_pct.mean():.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        # ---- paired differences (percentage points / enrichment ratio)
        pairs = [(("XGBoost", "random"), ("XGBoost", "grouped"), "XGBoost random - grouped"),
                 (("RandomForest", "random"), ("RandomForest", "grouped"), "RandomForest random - grouped"),
                 (("presence", "random"), ("presence", "grouped"), "presence random - grouped"),
                 (("XGBoost", "grouped"), ("null", "grouped"), "XGBoost - null (grouped)"),
                 (("XGBoost", "grouped"), ("presence", "grouped"), "XGBoost - presence (grouped)"),
                 (("RandomForest", "grouped"), ("presence", "grouped"), "RandomForest - presence (grouped)"),
                 (("XGBoost", "grouped"), ("RandomForest", "grouped"), "XGBoost - RandomForest (grouped)"),
                 (("presence", "grouped"), ("null", "grouped"), "presence - null (grouped)")]
        point = pd.DataFrame([r for r in rows if r["dataset"] == ds and r["fold"] == "mean_5folds"]).set_index(["model", "protocol", "K"])
        for a, b, label in pairs:
            for j, K in enumerate(KS):
                dp = boot_store[a]["prec"][:, j] - boot_store[b]["prec"][:, j]
                de = boot_store[a]["enr"][:, j] - boot_store[b]["enr"][:, j]
                dr = boot_store[a]["rec"][:, j] - boot_store[b]["rec"][:, j]
                diffs.append(dict(dataset=ds, comparison=label, K=K,
                                  d_precision_pp=point.loc[a + (K,), "precision_pct"] - point.loc[b + (K,), "precision_pct"],
                                  d_precision_ci_lo_pp=100 * np.percentile(dp, 2.5), d_precision_ci_hi_pp=100 * np.percentile(dp, 97.5),
                                  frac_boot_d_precision_gt0=float((dp > 0).mean()),
                                  d_recall_pp=point.loc[a + (K,), "recall_pct"] - point.loc[b + (K,), "recall_pct"],
                                  d_recall_ci_lo_pp=100 * np.percentile(dr, 2.5), d_recall_ci_hi_pp=100 * np.percentile(dr, 97.5),
                                  d_enrichment=point.loc[a + (K,), "enrichment"] - point.loc[b + (K,), "enrichment"],
                                  d_enrichment_ci_lo=np.percentile(de, 2.5), d_enrichment_ci_hi=np.percentile(de, 97.5)))
    out = pd.DataFrame(rows); out["procedencia"] = procedencia()
    out.to_csv(tc.RES_V3 / "topk_metrics.csv", index=False)
    dd = pd.DataFrame(diffs); dd["procedencia"] = procedencia()
    dd.to_csv(tc.RES_V3 / "topk_metrics_diffs.csv", index=False)

    # ---------------------------------------------------------------- markdown summary
    md.append("# Selección top-K corregida (H05, H06) — Dataset A y Dataset B v3\n")
    md.append("Positivo := y >= p90(y del fold de entrenamiento). Precisión/recuperación/enriquecimiento con empates resueltos por "
              "esperanza fraccional; nº de familias y presupuesto (<= 3 filas por familia) = media de 200 desempates aleatorios. "
              "IC 95 % = bootstrap de familias del pool de test (2 000 remuestreos, semilla 0, pareado entre condiciones), "
              "condicional a las predicciones OOF ya ajustadas; IC_filas = bootstrap i.i.d. de filas, sólo para comparar anchuras. "
              "Agregado = media de los 5 folds.\n")
    agg = out[out.fold == "mean_5folds"]
    for ds in DATASETS:
        md.append(f"\n## {ds}: K = 100\n")
        md.append("| Ranker | Split | Prevalencia % | Precisión@100 % [IC fam] | [IC filas] | Recuperación@100 % [IC fam] | Enriquecimiento [IC fam] | Familias en top-100 | Máx. cuota familia % | Precisión presupuesto % |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for model in MODELS:
            for proto in PROTOS:
                r = agg[(agg.dataset == ds) & (agg.model == model) & (agg.protocol == proto) & (agg.K == 100)].iloc[0]
                md.append(f"| {model} | {proto} | {r.prevalence_pct:.2f} | {r.precision_pct:.1f} [{r.precision_ci_fam_lo_pct:.1f}, {r.precision_ci_fam_hi_pct:.1f}] "
                          f"| [{r.precision_ci_row_lo_pct:.1f}, {r.precision_ci_row_hi_pct:.1f}] | {r.recall_pct:.1f} [{r.recall_ci_fam_lo_pct:.1f}, {r.recall_ci_fam_hi_pct:.1f}] "
                          f"| {r.enrichment:.2f} [{r.enrichment_ci_fam_lo:.2f}, {r.enrichment_ci_fam_hi:.2f}] | {r.n_families_topK:.1f} | {r.max_family_share_pct:.1f} | {r.budget_precision_pct:.1f} |")
        md.append(f"\n### {ds}: precisión@K (%) por K, media de folds [IC fam]\n")
        md.append("| Ranker | Split | " + " | ".join(f"K={K}" for K in KS) + " |"); md.append("|---|---|" + "---:|" * len(KS))
        for model in MODELS:
            for proto in PROTOS:
                cells = []
                for K in KS:
                    r = agg[(agg.dataset == ds) & (agg.model == model) & (agg.protocol == proto) & (agg.K == K)].iloc[0]
                    cells.append(f"{r.precision_pct:.1f} [{r.precision_ci_fam_lo_pct:.1f}, {r.precision_ci_fam_hi_pct:.1f}]")
                md.append(f"| {model} | {proto} | " + " | ".join(cells) + " |")
        md.append(f"\n### {ds}: enriquecimiento@K (precisión / prevalencia del fold), media de folds [IC fam]\n")
        md.append("| Ranker | Split | " + " | ".join(f"K={K}" for K in KS) + " |"); md.append("|---|---|" + "---:|" * len(KS))
        for model in MODELS:
            for proto in PROTOS:
                cells = []
                for K in KS:
                    r = agg[(agg.dataset == ds) & (agg.model == model) & (agg.protocol == proto) & (agg.K == K)].iloc[0]
                    cells.append(f"{r.enrichment:.2f} [{r.enrichment_ci_fam_lo:.2f}, {r.enrichment_ci_fam_hi:.2f}]")
                md.append(f"| {model} | {proto} | " + " | ".join(cells) + " |")
        md.append(f"\n### {ds}: diferencias pareadas (puntos porcentuales de precisión@K) [IC fam pareado]\n")
        md.append("| Comparación | " + " | ".join(f"K={K}" for K in KS) + " |"); md.append("|---|" + "---:|" * len(KS))
        for label in dd[dd.dataset == ds].comparison.unique():
            cells = []
            for K in KS:
                r = dd[(dd.dataset == ds) & (dd.comparison == label) & (dd.K == K)].iloc[0]
                cells.append(f"{r.d_precision_pp:+.1f} [{r.d_precision_ci_lo_pp:+.1f}, {r.d_precision_ci_hi_pp:+.1f}]")
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append(f"\n### {ds}: por fold, K = 100, protocolo agrupado\n")
        md.append("| Ranker | fold | n_test | positivos | prevalencia % | precisión % | recuperación % | enriquecimiento | familias top-100 | máx. cuota % | presupuesto % |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        pf = out[(out.dataset == ds) & (out.fold != "mean_5folds") & (out.K == 100) & (out.protocol == "grouped")]
        for model in MODELS:
            for _, r in pf[pf.model == model].iterrows():
                md.append(f"| {model} | {r.fold} | {r.n_test} | {r.n_pos} | {r.prevalence_pct:.2f} | {r.precision_pct:.1f} | {r.recall_pct:.1f} | {r.enrichment:.2f} | {r.n_families_topK:.1f} | {r.max_family_share_pct:.1f} | {r.budget_precision_pct:.1f} |")
    md.append(f"\n---\nprocedencia: {procedencia()}\n")
    (tc.RES_V3 / "topk_metrics.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[done] {time.time()-t0:.0f}s -> topk_metrics.csv / topk_metrics_diffs.csv / topk_metrics.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
