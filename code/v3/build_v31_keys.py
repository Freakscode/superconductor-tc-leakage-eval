"""Flat table of every number added to the manuscript in v3.1 (tutor review + data-acquisition
experiment), computed from the result files so that the per-cell verifier can check each one.

Reads results_v3/{tutor_analyses.json, tutor_error_vs_distance.csv, tutor_error_vs_distance_by_Tc.csv,
tutor_error_breakdowns.csv, tutor_within_system_ranking.csv, topk_prevalence.csv,
temporal_dating_sensitivity.csv, data_acquisition.csv, data_acquisition_key.json}
and writes results_v3/v31_key.json (flat: name -> value).
"""
import json, os, sys
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
RV3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT.parent / "results_v3"))


def main():
    k = {}
    ta = json.load(open(RV3 / "tutor_analyses.json"))
    for ds in "AB":
        for prot in ("random", "grouped"):
            k[f"twin_{ds}_{prot}_pct"] = 100 * ta[f"{ds}_{prot}_frac_exact_twin"]
            k[f"dlt01_{ds}_{prot}_pct"] = 100 * ta[f"{ds}_{prot}_frac_d_lt_0.1"]
    k["lab_rows_differ_gapped_first"] = ta["c19_labelling"]["rows_differ_gapped_vs_first_appearance"]
    k["lab_rows_differ_gapped_recoded"] = ta["c19_labelling"]["rows_differ_gapped_vs_recoded"]
    k["lab_n_rows"] = ta["c19_labelling"]["n_rows"]

    wr = pd.read_csv(RV3 / "tutor_within_system_ranking.csv")
    for r in wr.itertuples():
        k[f"wrho_{r.dataset}_{r.protocol}_median"] = r.median_rho
        k[f"wrho_{r.dataset}_{r.protocol}_fracpos_pct"] = 100 * r.frac_rho_gt0
        k[f"wrho_{r.dataset}_{r.protocol}_nsys"] = r.n_systems

    ev = pd.read_csv(RV3 / "tutor_error_vs_distance.csv")
    for (ds, prot), g in ev.groupby(["dataset", "protocol"]):
        g = g.sort_values("dist_lo")
        k[f"evd_{ds}_{prot}_max"] = g.MAE.max(); k[f"evd_{ds}_{prot}_min"] = g.MAE.min()
        k[f"evd_{ds}_{prot}_farthest"] = g.MAE.iloc[-1]
    et = pd.read_csv(RV3 / "tutor_error_vs_distance_by_Tc.csv")
    for r in et[et.protocol == "grouped"].itertuples():
        if r.Tc_range == ">=77 K":
            k[f"evdtc_{r.dataset}_77_{r.distance_tercile}"] = r.MAE
    bd = pd.read_csv(RV3 / "tutor_error_breakdowns.csv")
    for r in bd[bd.protocol == "grouped"].itertuples():
        lev = {"<10 K": "lt10", "10-30 K": "10_30", "30-77 K": "30_77", ">=77 K": "ge77"}.get(r.level, r.level).replace("-", "_").replace(">", "gt")
        k[f"bd_{r.dataset}_{r.breakdown}_{lev}"] = r.MAE
        if r.breakdown == "distance_tercile":
            k[f"bd_{r.dataset}_dist_{r.level}_n"] = r.n
    # mean Tc of the near / far distance terciles (grouped): weighted over Tc ranges
    for ds in "AB":
        g = et[(et.protocol == "grouped") & (et.dataset == ds)]
        for t in ("near", "far"):
            h = g[g.distance_tercile == t]; k[f"terc_meanTc_{ds}_{t}"] = float((h.mean_Tc * h.n).sum() / h.n.sum())

    tp = pd.read_csv(RV3 / "topk_prevalence.csv")
    thr = [c for c in tp.columns if "thr" in c.lower() or "umbral" in c.lower()][0]
    for ds, lab in (("A", "A"), ("Bv3", "B")):
        q = tp[(tp.dataset == ds) & (tp.protocol == "grouped")][thr]
        k[f"thr_{lab}_min"], k[f"thr_{lab}_max"] = float(q.min()), float(q.max())

    s = pd.read_csv(RV3 / "temporal_dating_sensitivity.csv")
    q = s[(s.modelo == "XGBoost") & (s.estrato == "todo") & (s.condicion == "temporal_real")]
    p = q.pivot_table(index=["corte", "metrica"], columns="escenario", values="valor")
    for (c, m), r in p.iterrows():
        if m in ("MAE", "R2"):
            k[f"dating_d{m}_{c}"] = float(r.sin_anio_ambiguo - r.principal)

    # dating sensitivity of each protocol's retrospective estimate (MAE), and 2010 headline values
    e = s[(s.modelo == "XGBoost") & (s.estrato == "todo") & (s.metrica == "MAE")]
    pe = e.pivot_table(index=["corte", "condicion"], columns="escenario", values="valor")
    for (c, cond), r in pe.iterrows():
        k[f"dating_est_dMAE_{cond}_{c}"] = float(r.sin_anio_ambiguo - r.principal)
    tv = pd.read_csv(RV3 / "temporal_validation.csv")
    tq = tv[(tv.corte == 2010) & (tv.escenario == "principal") & (tv.modelo == "XGBoost") & (tv.estrato == "todo")]
    for cond in ("temporal_real", "L0_aleatorio"):
        for m in ("MAE", "R2"):
            k[f"t2010_{cond}_{m}"] = float(tq[(tq.condicion == cond) & (tq.metrica == m)].valor.iloc[0])

    dk = json.load(open(RV3 / "data_acquisition_key.json"))
    for name, v in dk.items():
        if isinstance(v, dict):
            for f in ("mean", "lo", "hi"):
                k[f"acq_{name}_{f}"] = v[f]
        elif isinstance(v, (int, float)):
            k[f"acq_{name}"] = v
        elif isinstance(v, list):
            k[f"acq_{name}_min"], k[f"acq_{name}_max"] = min(v), max(v)
    d = pd.read_csv(RV3 / "data_acquisition.csv")
    for ds in "AB":
        for arm in ("cap1", "cap_all", "random_rows"):
            for N in (500, 2000):
                g = d[(d.dataset == ds) & (d.arm == arm) & (d.budget == N)]
                k[f"acq_{ds}_{arm}_N{N}_P77"] = 100 * g.P_at_100_77K.mean()
                k[f"acq_{ds}_{arm}_N{N}_meanTc"] = g.mean_Tc_train.mean()
                k[f"acq_{ds}_{arm}_N{N}_nsys"] = g.n_systems_train.mean()
        k[f"acq_{ds}_prev77_pct"] = 100 * d[d.dataset == ds].prevalence_77K.mean()
    k["procedencia"] = "code/v3/build_v31_keys.py; sources listed in the module docstring"
    json.dump(k, open(RV3 / "v31_key.json", "w"), indent=1)
    for n, v in k.items():
        if n != "procedencia":
            print(f"{n:48} {v:.4f}" if isinstance(v, float) else f"{n:48} {v}")


if __name__ == "__main__":
    sys.exit(main())
