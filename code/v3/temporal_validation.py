"""temporal_validation.py -- experiment 7 (v3): which retrospective protocol anticipates the REAL
performance on superconductors first reported AFTER a temporal cut?

Input   data/temporal_corpus_v3.npz  (code/v3/temporal_corpus.py; one row per normalised composition,
        X = 132 Magpie, y = median Tc, y_max, groups = family code, year = first report, anio_ambiguo)
        $SUPERCON_DATA_RAW (optional, not redistributed) -- only for the report-level stratum (iii)
        'composicion vista' = later reports (year >= T) of compositions first reported before T.
Design  For each cut T in CUTS (train = year_first < T, test = year_first >= T):
  (a) REAL temporal performance: model fitted on ALL compositions before T (frozen), evaluated on the
      compositions first reported at/after T. Strata of the test: familia_nueva (element set unseen
      before T) / familia_vista (element set seen, composition new -- every composition is new by
      construction after deduplication). Stratum (iii) composicion_vista is measured at REPORT level.
  (b) RETROSPECTIVE estimates using ONLY the data before T, 5-fold OOF predictions under
      L0 aleatorio      KFold(5, shuffle, random_state=42)
      L1 composicion    GroupKFold by normalised composition == L0 on this deduplicated corpus
                        (reported for completeness; identical partition is NOT re-fitted)
      L2 sistema        GroupKFold(5) deterministic by element set (family)
      L3 anfitrion      GroupKFold(5) deterministic by host element set (fraction >= 0.05)
      L4 LOCO cluster   KMeans(k=50, random_state=0, n_init=10) on standardised element fractions of the
                        TRAIN rows; clusters used as groups in GroupKFold(5) (each fold holds out ~10
                        clusters)
      OOF rows are stratified as familia_nueva / familia_vista relative to the training folds.
  (c) COMPARISON: estimate(b) - real(a) in MAE, RMSE, R2, top-100 precision, Spearman; absolute and
      relative; plus the RE-WEIGHTED estimate (OOF rows weighted so that the familia_nueva/vista mix
      equals the temporal test's mix) for each protocol.
Inference  Bootstrap of complete FAMILIES (B=2000, seed 0), conditional on the fitted OOF / temporal
      predictions. Train-side draws are shared by all protocols (paired); test-side draws are shared by
      all models/strata (paired). estimate-real intervals combine the (independent) train and test draws
      elementwise. Row-level bootstrap reported alongside. Spearman intervals use weighted Pearson on
      the ranks of the original sample (approximation).
Metrics  MAE, RMSE, R2, Spearman(y, yhat), top100 = fraction of the 100 highest predictions whose true
      Tc >= the 90th percentile of the TRAINING targets (operator >=; for OOF the percentile of the
      fold's training part, evaluated per fold and then pooled with the per-fold thresholds),
      prevalencia_alto_tc = fraction of the evaluated rows with Tc >= that threshold (base rate),
      lift_top100 = top100 / prevalencia.
Robustness  L0/L4 with seeds 0,1,2; Tc = max instead of median; excluding compositions whose first
      year comes from a fallback rule (anio_ambiguo).
Models   datasetB_pipeline.make_models(42) (XGBoost, RandomForest) with n_jobs=8 (predictions verified
      identical to n_jobs=1 for XGBoost hist; RF trees are seeded individually).

Usage:  python code/v3/temporal_validation.py [--cuts 1995 2005 2010 2015] [--principal 2010]
        [--no-robustez] [--out-prefix temporal_validation]
Outputs <RESULTS_V3>/temporal_validation.csv (long table), temporal_validation_comparacion.csv,
        temporal_validation.md, temporal_validation_predicciones.csv.gz, temporal_validation_tabla.tex
"""
from __future__ import annotations
import argparse, importlib.util, json, os, re, sys, time, warnings
from importlib.metadata import version as _v
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
CODE, DATA = ROOT / "code", ROOT / "data"
RESULTS_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))
RAW = Path(os.environ.get("SUPERCON_DATA_RAW", ROOT.parent / "data" / "primary.tsv"))
N_JOBS, B_BOOT, BOOT_SEED, K_CLUSTERS, HOST_MIN = 8, 2000, 0, 50, 0.05
MODELS = ("XGBoost", "RandomForest")
PROTOCOLS = ("L0_aleatorio", "L1_composicion", "L2_sistema", "L3_anfitrion", "L4_loco_cluster")
ROBUST_PROTOCOLS = ("L0_aleatorio", "L4_loco_cluster")
METRICS = ("MAE", "RMSE", "R2", "Spearman", "top100", "prevalencia_alto_tc", "lift_top100")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def procedencia(extra=None):
    import sklearn, xgboost
    p = dict(script="code/v3/temporal_validation.py", python=sys.version.split()[0], numpy=np.__version__,
             pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
             matminer=_v("matminer"), pymatgen=_v("pymatgen"), input="data/temporal_corpus_v3.npz",
             modelos="code/datasetB_pipeline.make_models(42), n_jobs=8", corte="train year_first<T, test year_first>=T",
             umbral_top100="percentil 90 de y del entrenamiento, operador >=",
             bootstrap=f"familias completas, B={B_BOOT}, semilla {BOOT_SEED}, pareado; condicional a las predicciones OOF/temporales",
             protocolos="L0 KFold(5,shuffle,rs=42); L1==L0 (corpus deduplicado); L2 GroupKFold(5) familia; "
                        f"L3 GroupKFold(5) anfitrion (fraccion>={HOST_MIN}); L4 GroupKFold(5) sobre KMeans(k={K_CLUSTERS}, rs=0) "
                        "de fracciones estandarizadas del train")
    if extra: p.update(extra)
    return p


# ------------------------------------------------------------------------------ helpers
def fractions_matrix(comp_keys):
    els = sorted({re.match(r"[A-Z][a-z]?", t).group(0) for k in comp_keys for t in k.split("|")})
    idx = {e: i for i, e in enumerate(els)}
    F = np.zeros((len(comp_keys), len(els)), dtype=float)
    for r, k in enumerate(comp_keys):
        for t in k.split("|"):
            m = re.match(r"([A-Z][a-z]?)([\d.]+)", t); F[r, idx[m.group(1)]] = float(m.group(2))
    return F, els


def host_key(comp_key, thr=HOST_MIN):
    return "|".join(sorted(m.group(1) for t in comp_key.split("|")
                           for m in [re.match(r"([A-Z][a-z]?)([\d.]+)", t)] if float(m.group(2)) >= thr))


def top100_mask(yhat, K=100):
    m = np.zeros(len(yhat), bool); m[np.argsort(-yhat, kind="stable")[:K]] = True; return m


def point_metrics(y, yhat, thr):
    e = yhat - y
    out = dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
               R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 else np.nan,
               Spearman=float(spearmanr(y, yhat).statistic) if len(y) > 2 else np.nan)
    hit = (y >= thr)                       # thr scalar, or per-row threshold (OOF: fold-specific)
    out["top100"] = float(hit[top100_mask(yhat)].mean()) if len(y) >= 100 else np.nan
    out["prevalencia_alto_tc"] = float(hit.mean())
    out["lift_top100"] = out["top100"] / out["prevalencia_alto_tc"] if out["prevalencia_alto_tc"] > 0 else np.nan
    return out


class Boot:
    """Family bootstrap weights shared across conditions (paired). W[b, family] = multinomial count."""
    def __init__(self, fam_codes, B=B_BOOT, seed=BOOT_SEED):
        self.fam = np.asarray(fam_codes); self.u, self.inv = np.unique(self.fam, return_inverse=True)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(len(self.u), np.full(len(self.u), 1 / len(self.u)), size=B).astype(float)

    def row_weights(self, rows):        # rows: index array into the corpus subset this Boot was built on
        return self.W[:, self.inv[rows]]                       # (B, n)


def weighted_metrics(y, yhat, thr, W):
    """Vectorised metrics under weight matrix W (B, n). thr scalar or per-row array."""
    e = yhat - y; sw = W.sum(1)
    mae = (W * np.abs(e)).sum(1) / sw
    rmse = np.sqrt((W * e ** 2).sum(1) / sw)
    ybar = (W * y).sum(1) / sw
    r2 = 1 - (W * e ** 2).sum(1) / (W * (y[None, :] - ybar[:, None]) ** 2).sum(1)
    ry, rp = rankdata(y), rankdata(yhat)
    my, mp = (W * ry).sum(1) / sw, (W * rp).sum(1) / sw
    cov = (W * (ry[None] - my[:, None]) * (rp[None] - mp[:, None])).sum(1)
    sp = cov / np.sqrt((W * (ry[None] - my[:, None]) ** 2).sum(1) * (W * (rp[None] - mp[:, None]) ** 2).sum(1))
    hit = (y >= thr).astype(float)
    order = np.argsort(-yhat, kind="stable")
    Wo = W[:, order]; cw = np.cumsum(Wo, axis=1)
    take = np.clip(100 - (cw - Wo), 0, None); take = np.minimum(take, Wo)   # weight of each row inside the top-100 multiset
    top = (take * hit[order][None]).sum(1) / np.maximum(take.sum(1), 1e-12)
    prev = (W * hit).sum(1) / sw
    return dict(MAE=mae, RMSE=rmse, R2=r2, Spearman=sp, top100=top, prevalencia_alto_tc=prev,
                lift_top100=np.where(prev > 0, top / np.maximum(prev, 1e-12), np.nan))


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


# ------------------------------------------------------------------------------ core per cut
def run_cut(T, X, y, fam, comp_keys, dbp, tag, rows_out, cmp_out, pred_out, seeds_L0L4=((42, 0),),
            report_level=None, verbose=True, robust_only=False):
    """robust_only=True: skip (a) and the comparison, run only L0/L4 for every seed in seeds_L0L4
    (the base seeds (42, 0) are then NOT expected in seeds_L0L4)."""
    t0 = time.time()
    tr = np.where(YEAR < T)[0]; te = np.where(YEAR >= T)[0]
    Xtr, ytr, ftr = X[tr], y[tr], fam[tr]; Xte, yte, fte = X[te], y[te], fam[te]
    thr = float(np.percentile(ytr, 90))
    fam_train = set(ftr.tolist()); new_fam_te = ~np.isin(fte, list(fam_train))
    strata_te = np.where(new_fam_te, "familia_nueva", "familia_vista")
    w_mix = {s: float((strata_te == s).mean()) for s in ("familia_nueva", "familia_vista")}
    boot_te, boot_tr = Boot(fte), Boot(ftr)
    rowboot_te, rowboot_tr = Boot(np.arange(len(te))), Boot(np.arange(len(tr)))
    real = {}     # (model, stratum) -> dict metric -> (point, boot array)

    def emit(cond, model, stratum, yy, yp, th, fb, rb, rows_idx, n_fam, extra=None):
        pm = point_metrics(yy, yp, th)
        Wf, Wr = fb.row_weights(rows_idx), rb.row_weights(rows_idx)
        bf, br = weighted_metrics(yy, yp, th, Wf), weighted_metrics(yy, yp, th, Wr)
        for k in METRICS:
            lo, hi = ci(bf[k]); lor, hir = ci(br[k])
            rows_out.append(dict(corte=T, escenario=tag, condicion=cond, modelo=model, estrato=stratum, n=len(yy),
                                 n_familias=n_fam, metrica=k, valor=pm[k], ci_lo_familias=lo, ci_hi_familias=hi,
                                 ci_lo_filas=lor, ci_hi_filas=hir, **(extra or {})))
        return pm, bf

    # (a) real temporal
    for name in (() if robust_only else MODELS):
        m = dbp.make_models(42)[name]; m.set_params(n_jobs=N_JOBS); t1 = time.time()
        m.fit(Xtr, ytr); yp = m.predict(Xte); fit_s = time.time() - t1
        for stratum, mask in (("todo", np.ones(len(te), bool)), ("familia_nueva", new_fam_te), ("familia_vista", ~new_fam_te)):
            idx = np.where(mask)[0]
            if len(idx) < 30: continue
            pm, bf = emit("temporal_real", name, stratum, yte[idx], yp[idx], thr, boot_te, rowboot_te, idx,
                          len(set(fte[idx].tolist())), dict(segundos_ajuste=round(fit_s, 1)))
            real[(name, stratum)] = (pm, bf)
        pred_out.append(pd.DataFrame(dict(corte=T, escenario=tag, condicion="temporal_real", modelo=name,
                                          comp_key=comp_keys[te], y=yte, yhat=yp, estrato=strata_te)))
        # stratum (iii): later reports of compositions already known before T (report level)
        if report_level is not None:
            rl = report_level[(report_level.year >= T) & (report_level.year_first < T)]
            pos = pd.Series(np.arange(len(tr)), index=comp_keys[tr])
            rl = rl[rl.comp_key.isin(pos.index)]
            if len(rl) >= 30:
                ptr = m.predict(Xtr[pos[rl.comp_key].values])
                yy = rl.tc.values.astype(float); fb = Boot(rl.fam.values); rb = Boot(np.arange(len(rl)))
                emit("temporal_real", name, "composicion_vista_reportes", yy, ptr, thr, fb, rb, np.arange(len(rl)),
                     rl.fam.nunique(), dict(nota="nivel reporte; prediccion del modelo para una composicion de su propio entrenamiento"))

    # (b) retrospective protocols on the training part only
    ck_tr = comp_keys[tr]
    Ftr, _ = fractions_matrix(ck_tr)
    host = pd.factorize(pd.Series([host_key(k) for k in ck_tr]))[0]
    fam_tr_codes = pd.factorize(pd.Series(ftr))[0]
    comp_codes = pd.factorize(pd.Series(ck_tr))[0]
    Fstd = StandardScaler().fit_transform(Ftr)
    for seed_kf, seed_km in seeds_L0L4:
        km = KMeans(n_clusters=K_CLUSTERS, random_state=seed_km, n_init=10).fit(Fstd)
        splits = {"L0_aleatorio": list(KFold(5, shuffle=True, random_state=seed_kf).split(Xtr)),
                  "L1_composicion": list(GroupKFold(5).split(Xtr, ytr, comp_codes)),
                  "L2_sistema": list(GroupKFold(5).split(Xtr, ytr, fam_tr_codes)),
                  "L3_anfitrion": list(GroupKFold(5).split(Xtr, ytr, host)),
                  "L4_loco_cluster": list(GroupKFold(5).split(Xtr, ytr, km.labels_))}
        for proto, sp in splits.items():
            if robust_only and proto not in ROBUST_PROTOCOLS:
                continue
            ptag = proto if not robust_only else f"{proto}_seed{seed_kf if proto.startswith('L0') else seed_km}"
            oof = {n: np.empty(len(tr)) for n in MODELS}; thr_row = np.empty(len(tr)); strat = np.empty(len(tr), dtype=object)
            t1 = time.time()
            for a, b in sp:
                thr_row[b] = np.percentile(ytr[a], 90)
                seen = set(ftr[a].tolist()); strat[b] = np.where(np.isin(ftr[b], list(seen)), "familia_vista", "familia_nueva")
                for n in MODELS:
                    mm = dbp.make_models(42)[n]; mm.set_params(n_jobs=N_JOBS); mm.fit(Xtr[a], ytr[a]); oof[n][b] = mm.predict(Xtr[b])
            fit_s = time.time() - t1
            mix = {s: float((strat == s).mean()) for s in ("familia_nueva", "familia_vista")}
            for n in MODELS:
                est = {}
                for stratum, mask in (("todo", np.ones(len(tr), bool)), ("familia_nueva", strat == "familia_nueva"), ("familia_vista", strat == "familia_vista")):
                    idx = np.where(mask)[0]
                    if len(idx) < 30: continue
                    pm, bf = emit(ptag, n, stratum, ytr[idx], oof[n][idx], thr_row[idx], boot_tr, rowboot_tr, idx,
                                  len(set(ftr[idx].tolist())), dict(segundos_ajuste=round(fit_s, 1), frac_familia_nueva_oof=mix["familia_nueva"]))
                    est[stratum] = (pm, bf)
                # re-weighted estimate: OOF rows weighted so the stratum mix equals the temporal test mix
                wrow = np.array([w_mix[s] / max(mix[s], 1e-12) for s in strat]) if all(mix[s] > 0 for s in w_mix) else None
                if wrow is not None:
                    Wf = boot_tr.row_weights(np.arange(len(tr))) * wrow[None]; Wr = rowboot_tr.row_weights(np.arange(len(tr))) * wrow[None]
                    pmw = weighted_metrics(ytr, oof[n], thr_row, wrow[None])
                    bfw, brw = weighted_metrics(ytr, oof[n], thr_row, Wf), weighted_metrics(ytr, oof[n], thr_row, Wr)
                    for k in METRICS:
                        lo, hi = ci(bfw[k]); lor, hir = ci(brw[k])
                        rows_out.append(dict(corte=T, escenario=tag, condicion=ptag, modelo=n, estrato="reponderado_mezcla_test", n=len(tr),
                                             n_familias=len(fam_train), metrica=k, valor=float(pmw[k][0]), ci_lo_familias=lo, ci_hi_familias=hi,
                                             ci_lo_filas=lor, ci_hi_filas=hir, frac_familia_nueva_oof=mix["familia_nueva"],
                                             frac_familia_nueva_test=w_mix["familia_nueva"]))
                    est["reponderado_mezcla_test"] = ({k: float(pmw[k][0]) for k in METRICS}, bfw)
                # (c) comparison with the real temporal metric ("todo")
                if (n, "todo") in real:
                    pr, br_ = real[(n, "todo")]
                    for stratum in ("todo", "reponderado_mezcla_test"):
                        if stratum not in est: continue
                        pe, be = est[stratum]
                        for k in METRICS:
                            d = pe[k] - pr[k]; db = be[k] - br_[k]
                            rel = d / abs(pr[k]) if pr[k] not in (0, np.nan) else np.nan
                            lo, hi = ci(db); rlo, rhi = ci(db / np.abs(br_[k]))
                            cmp_out.append(dict(corte=T, escenario=tag, protocolo=ptag, modelo=n, estimacion=stratum, metrica=k,
                                                estimacion_retrospectiva=pe[k], temporal_real=pr[k], error_abs=d, ci_lo_abs=lo, ci_hi_abs=hi,
                                                error_rel=rel, ci_lo_rel=rlo, ci_hi_rel=rhi, n_train=len(tr), n_test=len(te),
                                                frac_familia_nueva_test=w_mix["familia_nueva"], frac_familia_nueva_oof=mix["familia_nueva"]))
                pred_out.append(pd.DataFrame(dict(corte=T, escenario=tag, condicion=ptag, modelo=n, comp_key=ck_tr, y=ytr, yhat=oof[n], estrato=strat)))
            if verbose: print(f"  [{tag} T={T}] {ptag}: {fit_s:.0f}s", flush=True)
    if verbose: print(f"[{tag} T={T}] n_train={len(tr)} n_test={len(te)} new_fam={w_mix['familia_nueva']:.3f} total {time.time()-t0:.0f}s", flush=True)
    return dict(n_train=int(len(tr)), n_test=int(len(te)), thr_p90_train=thr, frac_familia_nueva_test=w_mix["familia_nueva"],
                fam_train=len(fam_train), fam_test=int(len(set(fte.tolist()))), segundos=round(time.time() - t0, 1))


def report_level_table(comp_keys, year_first_by_key, corpus_mod):
    """Later reports of known compositions (needs the raw dump). Returns DataFrame(comp_key, tc, year, year_first, fam) or None."""
    if not RAW.exists(): return None
    raw = pd.read_csv(RAW, sep="\t", skiprows=2, dtype=str, keep_default_na=False)
    raw["tc"] = pd.to_numeric(raw.tc, errors="coerce")
    yr = raw.journal.map(corpus_mod.extract_year); raw["year"] = yr.map(lambda t: t[0]).astype(float)
    raw.loc[(raw.year < corpus_mod.YEAR_MIN) | (raw.year > corpus_mod.YEAR_MAX), "year"] = np.nan
    cls = raw.element.map(corpus_mod.classify)
    raw["dec"] = cls.map(lambda t: t[0]); raw["f3"] = cls.map(lambda t: t[2])
    k = raw[(raw.tc > 0) & (raw.dec != "excluir") & raw.year.notna()].copy()
    k["comp_key"] = k.f3.map(corpus_mod.comp_key)
    k = k[k.comp_key.isin(year_first_by_key.index)]
    k["year_first"] = year_first_by_key[k.comp_key].values
    fam_by_key = pd.Series(FAM, index=comp_keys); k["fam"] = fam_by_key[k.comp_key].values
    return k[["comp_key", "tc", "year", "year_first", "fam"]].reset_index(drop=True)


# ------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, nargs="+", default=[1995, 2005, 2010, 2015])
    ap.add_argument("--principal", type=int, default=2010)
    ap.add_argument("--no-robustez", action="store_true")
    ap.add_argument("--out-prefix", default="temporal_validation")
    a = ap.parse_args(argv)
    global YEAR, FAM
    dbp = load_mod("datasetB_pipeline", CODE / "datasetB_pipeline.py")
    corpus_mod = load_mod("temporal_corpus", CODE / "v3" / "temporal_corpus.py")
    z = np.load(DATA / "temporal_corpus_v3.npz", allow_pickle=True)
    X, y, ymax, fam, year, ck, amb = (z["X"].astype(np.float32), z["y"], z["y_max"], z["groups"], z["year"],
                                      z["comp_key"].astype(str), z["anio_ambiguo"].astype(bool))
    YEAR, FAM = year, fam
    rows, cmp_rows, preds, meta = [], [], [], {}
    yf = pd.Series(year, index=ck)
    rl = report_level_table(ck, yf, corpus_mod)
    meta["report_level_available"] = rl is not None
    T0 = time.time()
    for T in a.cuts:
        meta[f"principal_T{T}"] = run_cut(T, X, y, fam, ck, dbp, "principal", rows, cmp_rows, preds, report_level=rl)
    if not a.no_robustez:
        # (i) seeds for L0 / L4 (all cuts)
        for T in a.cuts:
            run_cut(T, X, y, fam, ck, dbp, "principal", rows, cmp_rows, preds,
                    seeds_L0L4=((0, 1), (1, 2), (2, 3)), report_level=None, verbose=True, robust_only=True)
        # (ii) Tc = max per composition (principal cut)
        meta[f"tc_max_T{a.principal}"] = run_cut(a.principal, X, ymax, fam, ck, dbp, "tc_max", rows, cmp_rows, preds)
        # (iii) exclude ambiguous-year compositions (principal cut)
        keep = ~amb
        YEAR, FAM = year[keep], fam[keep]
        meta[f"sin_anio_ambiguo_T{a.principal}"] = run_cut(a.principal, X[keep], y[keep], fam[keep], ck[keep], dbp, "sin_anio_ambiguo", rows, cmp_rows, preds)
        YEAR, FAM = year, fam
    # the seed re-runs also re-emit the seed-42/0 rows -> drop exact duplicates
    df = pd.DataFrame(rows).drop_duplicates(subset=["corte", "escenario", "condicion", "modelo", "estrato", "metrica"])
    cmp = pd.DataFrame(cmp_rows).drop_duplicates(subset=["corte", "escenario", "protocolo", "modelo", "estimacion", "metrica"])
    pr = pd.concat(preds, ignore_index=True).drop_duplicates(subset=["corte", "escenario", "condicion", "modelo", "comp_key"])
    pr = pr[~pr.condicion.str.contains("_seed")]          # seed re-runs: metrics only (size)
    prov = procedencia(dict(segundos_total=round(time.time() - T0, 1), corte_principal=a.principal, meta=meta))
    RESULTS_V3.mkdir(parents=True, exist_ok=True)
    for d, name in ((df, f"{a.out_prefix}.csv"), (cmp, f"{a.out_prefix}_comparacion.csv")):
        d = d.copy(); d["procedencia"] = json.dumps(prov, ensure_ascii=False); d.to_csv(RESULTS_V3 / name, index=False)
    with open(RESULTS_V3 / f"{a.out_prefix}_predicciones.csv", "w") as fh:
        fh.write("# procedencia: " + json.dumps(prov, ensure_ascii=False) + "\n"); pr.to_csv(fh, index=False)
    import gzip, shutil
    with open(RESULTS_V3 / f"{a.out_prefix}_predicciones.csv", "rb") as fi, gzip.open(RESULTS_V3 / f"{a.out_prefix}_predicciones.csv.gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(RESULTS_V3 / f"{a.out_prefix}_predicciones.csv")
    json.dump(prov, open(RESULTS_V3 / f"{a.out_prefix}_meta.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, indent=1)); print(f"total {time.time()-T0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
