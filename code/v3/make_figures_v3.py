"""make_figures_v3.py -- regenerate figure1..4.png of paper1_manuscript.tex (v3) from the v3 result files.

Adapted from code/v2/make_figures.py (same palette, style helpers, overlap check and fidelity asserts);
what changed for v3 (adversarial revision H01/H07/H09/H10):
  * every Dataset-B quantity comes from the AUDITED checkpoint data/datasetB_featurized_v3.npz (N 12 336);
  * intervals are paired bootstraps of whole chemical families (results_v3/paired_family_bootstrap.csv);
  * figure 2 no longer stacks R^2 as an additive decomposition: it draws DIFFERENCES of losses
    (Delta MAE and SSE reduction) between representations, with family-paired intervals
    (results_v3/ablation_representation_diffs.csv);
  * figure 3(b) shows pooled versus macro (mean-of-family-means) MAE inflation instead of the seed means;
  * figure 4(b) shows the effective sample size under the two explicitly defined estimators of
    results_v3/design_effect_v3.csv instead of the single Searle-m0 number;
  * the nearest-neighbour panels (1d, 3d) standardise with a scaler fitted on the TRAINING rows only and
    are asserted against results_v3/nn_distance_trainscaler.csv.
Every drawn number is asserted against its CSV (verify()); the geometric text-overlap check must be CLEAN.

    python code/v3/make_figures_v3.py [--submit ../paper1_submission] [--no-verify]
    SUPERCON_ROOT=/path/to/repo SUPERCON_RESULTS_V3=/path/to/results_v3  (defaults: repo of this file, <repo>/results_v3)
"""
from __future__ import annotations
import argparse, importlib.util, os, shutil, sys, time
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
CODE, DATA = ROOT / "code", ROOT / "data"
RES_V2 = ROOT / "results_v2"
RES_V3 = Path(os.environ.get("SUPERCON_RESULTS_V3", ROOT / "results_v3"))
FIG = ROOT / "figures"
SUBMIT = FIG
SEED, DPI = 42, 300

C_RANDOM, C_FAMILY, C_GAIN, C_PRESENCE, C_ORACLE = "#9E9E9E", "#C2452D", "#2C6FAD", "#E08214", "#3F3F3F"
C_TEXT, C_META = "#333333", "#777777"
C_A, C_B = "#2C6FAD", "#7B4EA3"   # dataset hue (fig 2 / 4b)


def _shade(hex_color, lighten=0.0):
    rgb = np.array(mpl.colors.to_rgb(hex_color)); return tuple(rgb + (1.0 - rgb) * float(lighten))


def apply_figure_style(*, sizes=(8, 7, 6)):
    base, ann, tick = sizes
    mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": DPI, "savefig.bbox": "tight", "font.size": base,
                         "axes.titlesize": base, "axes.labelsize": base, "legend.fontsize": ann, "xtick.labelsize": tick,
                         "ytick.labelsize": tick, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation": "left", "axes.titleweight": "normal", "axes.labelcolor": C_TEXT,
                         "text.color": C_TEXT, "xtick.color": C_TEXT, "ytick.color": C_TEXT, "axes.edgecolor": "#555555",
                         "axes.linewidth": 0.8, "legend.frameon": False, "axes.grid": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})


def panel_letter(ax, letter, dx=-0.18, dy=1.02):
    return ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold", fontsize=mpl.rcParams["font.size"] + 2, va="bottom", ha="left")


# ------------------------------------------------------------------------------ loaders --------
def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def load_csvs():
    csv = {"pfb": pd.read_csv(RES_V3 / "paired_family_bootstrap.csv"),
           "abl": pd.read_csv(RES_V3 / "ablation_representation.csv"),
           "abld": pd.read_csv(RES_V3 / "ablation_representation_diffs.csv"),
           "deff": pd.read_csv(RES_V3 / "design_effect_v3.csv"),
           "topk": pd.read_csv(RES_V3 / "topk_metrics.csv"),
           "nn": pd.read_csv(RES_V3 / "nn_distance_trainscaler.csv").set_index("dataset"),
           "shuffleA": pd.read_csv(RES_V2 / "tabla2_shuffle.csv"),
           "perseedA": pd.read_csv(RES_V2 / "tabla2_runs_por_semilla.csv"),
           "oof": pd.read_csv(RES_V3 / "oof_summary.csv"),
           "t1B": pd.read_csv(RES_V3 / "tabla1_B_antes_despues.csv")}
    csv["pfb"] = csv["pfb"][csv["pfb"].scheme == "family_paired"]
    assert len(csv["abl"]) == 24 and len(csv["abld"]) == 60
    return csv


def pfb(csv, ds, model, stat):
    r = csv["pfb"][(csv["pfb"].dataset == ds) & (csv["pfb"].model == model) & (csv["pfb"].statistic == stat)]
    assert len(r) == 1, (ds, model, stat, len(r)); return r.iloc[0]


def abl(csv, ds, prot, method):
    d = csv["abl"]; r = d[(d.dataset == ds) & (d.protocol == prot) & (d.method == method)]
    assert len(r) == 1; return r.iloc[0]


def abld(csv, ds, prot, a, b):
    d = csv["abld"]; r = d[(d.dataset == ds) & (d.protocol == prot) & (d.ref_a == a) & (d.method_b == b)]
    assert len(r) == 1, (ds, prot, a, b); return r.iloc[0]


def deff(csv, ds, est_prefix, variant="todas las familias"):
    d = csv["deff"]; r = d[(d.dataset == ds) & (d.variant == variant) & d.estimador.str.startswith(est_prefix)]
    assert len(r) == 1, (ds, est_prefix, len(r)); return r.iloc[0]


def topk(csv, ds, model, prot, K=100):
    d = csv["topk"]; mod = d.model.isna() if model is None else (d.model == model)
    r = d[(d.dataset == ds) & mod & (d.protocol == prot) & (d.K == K) & (d.fold == "mean_5folds")]
    assert len(r) == 1, (ds, model, prot, len(r)); return r.iloc[0]


def shufA(csv, model):
    d = csv["shuffleA"]; r = d[(d.dataset == "A") & (d.modelo == model)]; assert len(r) == 1; return r.iloc[0]


# ------------------------------------------------------------------------------ recompute ------
def recompute(verbose=True):
    """Family sizes (A), pooled OOF parity (A, XGBoost, GroupKFold(5)), and 1-NN distances (A, B v3) with the
    scaler fitted on the training rows only (H09: no test information enters the standardisation)."""
    t0 = time.time()
    tcp = _import("tc_pipeline", CODE / "tc_pipeline.py")
    feat = pd.read_csv(DATA / "train.csv"); uniq = pd.read_csv(DATA / "unique_m.csv")
    assert np.allclose(feat["critical_temp"], uniq["critical_temp"])
    XA = feat.drop(columns=["critical_temp"]).values; yA = feat["critical_temp"].values; gA = tcp.chemical_families(uniq)
    elem_cols = [c for c in uniq.columns if c not in ("critical_temp", "material")]
    fam_key = (uniq[elem_cols] > 0).apply(lambda r: "|".join(sorted(e for e in elem_cols if r[e])), axis=1)
    vc = fam_key.value_counts(); sizes_A = vc.values.astype(int); top_fams = [(k, int(v)) for k, v in vc.head(3).items()]
    zB = np.load(DATA / "datasetB_featurized_v3.npz", allow_pickle=True)
    XB, yB, gB = zB["X"].astype(float), zB["y"].astype(float), zB["groups"]
    assert len(yB) == 12336 and len(np.unique(gB)) == 3046
    ypA = tcp.cv_predict_manual(lambda: tcp.make_models(SEED)["XGBoost"], XA, yA, GroupKFold(5), gA)
    parity = dict(y=yA, yp=ypA, mae=float(mean_absolute_error(yA, ypA)), r2=float(r2_score(yA, ypA)), n=int(len(yA)))

    def nn_dists(X, groups):
        ri = np.arange(len(X)); np.random.RandomState(SEED).shuffle(ri); cut = int(0.8 * len(X))
        g_tr, g_te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, np.zeros(len(X)), groups))
        assert not (set(groups[g_tr]) & set(groups[g_te]))
        def near(tr, te):
            sc = StandardScaler().fit(X[tr])                     # TRAIN-only scaler
            return NearestNeighbors(n_neighbors=1).fit(sc.transform(X[tr])).kneighbors(sc.transform(X[te]))[0].ravel()
        return near(ri[:cut], ri[cut:]), near(g_tr, g_te)
    nn = {"A": nn_dists(XA, gA), "B": nn_dists(XB, gB)}
    out = dict(sizes_A=sizes_A, top_fams_A=top_fams, N_A=int(len(yA)), k_A=int(len(sizes_A)), parity=parity, nn=nn,
               N_B=int(len(yB)), k_B=int(len(np.unique(gB))), seconds=time.time() - t0)
    if verbose:
        print(f"[recompute] A N={out['N_A']} k={out['k_A']} largest={sizes_A.max()}; parity MAE={parity['mae']:.4f} R2={parity['r2']:.4f}")
        for nm, (r_, g_) in nn.items():
            print(f"[recompute] NN {nm}: random {np.median(r_):.4f} grouped {np.median(g_):.4f} ratio {np.median(g_)/np.median(r_):.2f} n={len(r_)}/{len(g_)}")
        print(f"[recompute] {out['seconds']:.0f}s")
    return out


# ------------------------------------------------------------------------------ overlap check --
def _offview(fig):
    dead = set()
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if not (lo - 1e-9 <= loc <= hi + 1e-9): dead.add(tick.label1); dead.add(tick.label2)
    return dead


def check_overlaps(fig, label=""):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); dead = _offview(fig)
    texts = [(t, mpl.text.Text.get_window_extent(t, r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible() and t not in dead]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    ticklabels = {ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both")) for ax in fig.axes}
    f = []
    for i, (ta, ba) in enumerate(texts):
        for tb, bb in texts[i + 1:]:
            if ba.overlaps(bb): f.append(("text/text", ta.get_text()[:38], tb.get_text()[:38]))
    for t, bt in texts:
        for s, bs in spines:
            if bt.overlaps(bs) and t not in ticklabels.get(s.axes, ()): f.append(("text/spine", t.get_text()[:38], s.spine_type))
    fb = fig.bbox
    for t, bt in texts:
        if not (bt.x0 >= fb.x0 - 1 and bt.y0 >= fb.y0 - 1 and bt.x1 <= fb.x1 + 1 and bt.y1 <= fb.y1 + 1): f.append(("outside-canvas", t.get_text()[:38], ""))
    print(f"[overlap] {label}: {'CLEAN' if not f else str(len(f)) + ' FINDINGS'}")
    for x in f: print("   ", x)
    return f


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True); p = FIG / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    shutil.copyfile(p, ROOT / name)                                  # manuscript reads figures from the repo root
    if SUBMIT != FIG: SUBMIT.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, SUBMIT / name)
    print(f"[save] {p} (+ {ROOT / name}{' + ' + str(SUBMIT / name) if SUBMIT != FIG else ''})"); return p


def ci_err(v, lo, hi):
    return np.array([[v - lo], [hi - v]])


# ================================================================================ FIGURE 1 =====
def figure1(csv, rc):
    D = {}
    fig = plt.figure(figsize=(7.4, 5.9))
    gs = fig.add_gridspec(2, 2, hspace=0.66, wspace=0.36, left=0.115, right=0.975, top=0.875, bottom=0.15)
    axa, axb, axc, axd = (fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1))
    fig.suptitle(f"Anatomy of chemical data leakage in composition-based $T_c$ prediction (Dataset A, $N$ = {rc['N_A']:,})", fontsize=9, y=0.975)
    # (a) family sizes
    sizes = rc["sizes_A"]; rank = np.arange(1, len(sizes) + 1)
    D["a_share_ge2"] = float(sizes[sizes >= 2].sum() / sizes.sum()); D["a_k"] = int(len(sizes)); D["a_max"] = int(sizes.max())
    axa.fill_between(rank, 1, sizes, step="post", color="#D8D8D8", edgecolor="#606060", linewidth=0.6)
    axa.set_yscale("log"); axa.set_ylim(0.8, 1900); axa.set_yticks([1, 10, 100, 1000]); axa.set_yticklabels(["1", "10", "100", "1k"])
    axa.set_xlim(-60, len(sizes) + 120); axa.set_xlabel("Element system (rank by size)"); axa.set_ylabel("Materials in system")
    axa.set_title(f"{D['a_share_ge2']*100:.0f}% of materials share their\nelement set with another material", fontsize=8)
    block = "three largest systems\n" + "\n".join("{" + ", ".join(k.split("|")) + f"}}: {c}" for k, c in rc["top_fams_A"])
    axa.annotate(block, xy=(rank[1], sizes[1]), xytext=(len(sizes) * 0.20, 300), fontsize=6, color=C_TEXT, va="center", ha="left",
                 linespacing=1.45, arrowprops=dict(arrowstyle="-", lw=0.5, color=C_META, shrinkA=2, shrinkB=1.5))
    # (b) MAE random vs grouped, 5 seeds (Dataset A unchanged from v2)
    models = ["RandomForest", "XGBoost"]; x = np.arange(2); w = 0.34
    for j, (scheme, col, lab, key) in enumerate([("random", C_RANDOM, "Random split", "MAE_random"), ("grouped", C_FAMILY, "Element-system split", "MAE_grouped")]):
        vals = [shufA(csv, m)[f"{key}_mean"] for m in models]; errs = [shufA(csv, m)[f"{key}_std"] for m in models]
        D[f"b_{scheme}"] = [float(v) for v in vals]
        axb.bar(x + (j - 0.5) * w, vals, w, yerr=errs, capsize=2.4, color=col, edgecolor="none", label=lab, error_kw=dict(lw=0.7, ecolor="#444444"))
        pr = csv["perseedA"]; arm = "random_shuffle" if scheme == "random" else "grouped_shuffle"
        for i, m in enumerate(models):
            pts = pr[(pr.dataset == "A") & (pr.modelo == m) & (pr.scheme == arm)]["MAE"].values
            axb.scatter(np.full(len(pts), x[i] + (j - 0.5) * w), pts, s=5, facecolor="white", edgecolor="#333333", linewidth=0.5, zorder=5)
        for i, v in enumerate(vals): axb.text(x[i] + (j - 0.5) * w, v + errs[i] + 0.28, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    infl = [shufA(csv, m)["inflacion_pct_mean"] for m in models]; D["b_inflation"] = [float(v) for v in infl]
    axb.set_xticks(x); axb.set_xticklabels(models); axb.set_ylim(0, 11.0); axb.set_ylabel("Test MAE (K) · 5-fold CV")
    axb.set_title(f"Holding out element systems: error rises {min(infl):.0f}\u2013{max(infl):.0f}%", fontsize=8)
    axb.text(0.985, 0.965, "lower = better", transform=axb.transAxes, ha="right", va="top", fontsize=6, color=C_META)
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.135), ncols=2, fontsize=6.5, handlelength=1.1, borderpad=0.1, columnspacing=1.4, handletextpad=0.5)
    # (c) parity
    p = rc["parity"]; D["c_mae"], D["c_r2"], D["c_n"] = p["mae"], p["r2"], p["n"]; lim = 180
    axc.scatter(p["y"], p["yp"], s=3.2, alpha=0.10, color=C_FAMILY, edgecolor="none", rasterized=True)
    axc.plot([0, lim], [0, lim], ls="--", lw=0.8, color="#333333", label="perfect")
    axc.set_xlim(-6, lim + 6); axc.set_ylim(-6, lim + 6); axc.set_xticks([0, 40, 80, 120, 160]); axc.set_yticks([0, 40, 80, 120, 160]); axc.set_aspect("equal")
    axc.set_xlabel("Actual $T_c$ (K)"); axc.set_ylabel("Predicted $T_c$ (K)"); axc.set_title("XGBoost under element-system hold-out", fontsize=8)
    axc.text(0.035, 0.95, f"held-out systems, pooled out-of-fold\nMAE = {p['mae']:.1f} K   $R^2$ = {p['r2']:.2f}", transform=axc.transAxes, fontsize=6.2, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", lw=0.5))
    axc.legend(loc="lower right", fontsize=6.5, handlelength=1.4, borderpad=0.1)
    # (d) NN distances (train-only scaler)
    dr, dg = rc["nn"]["A"]; mr, mg = float(np.median(dr)), float(np.median(dg))
    D["d_median_random"], D["d_median_family"], D["d_ratio"] = mr, mg, mg / mr; clip = 3.0; bins = np.linspace(0, clip, 46)
    axd.hist(np.clip(dr, 0, clip), bins=bins, color=C_RANDOM, alpha=0.85, label="Random split", edgecolor="none")
    axd.hist(np.clip(dg, 0, clip), bins=bins, color=C_FAMILY, alpha=0.62, label="Element-system split", edgecolor="none")
    axd.axvline(mr, ls="--", lw=1.0, color="#4A4A4A"); axd.axvline(mg, ls="--", lw=1.0, color=C_FAMILY)
    axd.set_xlim(-0.08, clip + 0.08); axd.set_ylim(0, 2450)
    axd.annotate(f"median {mr:.3f}", xy=(mr, 1320), xytext=(mr + 0.28, 1620), fontsize=6.5, color="#4A4A4A", va="center", ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=C_META, shrinkA=0, shrinkB=1))
    axd.text(mg + 0.07, 1100, f"median {mg:.2f}", fontsize=6.5, color=C_FAMILY, va="center", ha="left")
    axd.set_xlabel("Distance to nearest training material\n(standardised on training rows; clipped at 3)"); axd.set_ylabel("Test materials")
    axd.set_title(f"Random splits test on near-twins ({mg/mr:.0f}\u00d7 closer)", fontsize=8)
    axd.legend(loc="upper right", fontsize=6.5, handlelength=1.1, borderpad=0.1, labelspacing=0.25)
    for ax, L in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")): panel_letter(ax, L, dx=-0.20 if ax in (axa, axc) else -0.155, dy=1.12)
    return fig, D


# ================================================================================ FIGURE 2 =====
STEPS = [("R0", "R1", "add fractions\nto presence"), ("R1", "R2", "replace by\ndescriptors"), ("R2", "R3", "add fractions\nto descriptors")]
CONDS = [("A", "random"), ("A", "grouped"), ("B", "random"), ("B", "grouped")]


def figure2(csv, rc):
    """Representation ablation as loss DIFFERENCES (no stacked R^2). (a) Delta MAE per step, (b) SSE reduction vs presence."""
    D = {}
    fig = plt.figure(figsize=(7.4, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.38, left=0.10, right=0.985, top=0.74, bottom=0.33)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    fig.suptitle("What each representation adds: most of the descriptor gain over element presence is stoichiometry, and on\n"
                 "held-out element systems the descriptors alone do not beat presence + fractions", fontsize=8.2, y=1.0, x=0.01, ha="left")
    cols = {("A", "random"): _shade(C_A, 0.55), ("A", "grouped"): C_A, ("B", "random"): _shade(C_B, 0.55), ("B", "grouped"): C_B}
    marks = {"random": "o", "grouped": "s"}
    xs = np.arange(len(STEPS)); off = np.linspace(-0.27, 0.27, len(CONDS))
    for k, (ds, prot) in enumerate(CONDS):
        for i, (a, b, _) in enumerate(STEPS):
            r = abld(csv, ds, prot, a, b); D[f"a_{ds}_{prot}_{a}{b}"] = float(r.dMAE)
            axa.errorbar(xs[i] + off[k], r.dMAE, yerr=ci_err(r.dMAE, r.dMAE_fam_lo, r.dMAE_fam_hi), fmt=marks[prot], ms=4.2, color=cols[(ds, prot)],
                         mec=cols[(ds, prot)], mfc=cols[(ds, prot)] if prot == "grouped" else "white", capsize=1.8, lw=0.9, zorder=3)
    axa.axhline(0, lw=0.7, color="#777777", zorder=1)
    axa.set_xticks(xs); axa.set_xticklabels([s[2] for s in STEPS], fontsize=6.5)
    axa.set_ylabel("\u0394MAE of the step (K)   \u00b7   negative = better"); axa.set_xlim(-0.6, len(STEPS) - 0.4); axa.margins(y=0.08)
    axa.set_title("Fractions cut the error; descriptors add little\nonce fractions are present", fontsize=8)
    # (b) SSE reduction vs R0
    reps = [("R1", "+ fractions"), ("R2", "descriptors"), ("R3", "descriptors\n+ fractions")]; xb = np.arange(len(reps))
    for k, (ds, prot) in enumerate(CONDS):
        for i, (b, _) in enumerate(reps):
            r = abld(csv, ds, prot, "R0", b); D[f"b_{ds}_{prot}_{b}"] = float(r.SSE_red)
            axb.errorbar(xb[i] + off[k], 100 * r.SSE_red, yerr=100 * ci_err(r.SSE_red, r.SSE_red_fam_lo, r.SSE_red_fam_hi), fmt=marks[prot], ms=4.2,
                         color=cols[(ds, prot)], mec=cols[(ds, prot)], mfc=cols[(ds, prot)] if prot == "grouped" else "white", capsize=1.8, lw=0.9, zorder=3)
    axb.axhline(0, lw=0.7, color="#777777", zorder=1)
    axb.set_xticks(xb); axb.set_xticklabels([r[1] for r in reps], fontsize=6.5); axb.set_xlim(-0.6, len(reps) - 0.4); axb.margins(y=0.08)
    axb.set_ylabel("SSE reduction vs element presence (%)\nhigher = better"); axb.set_title("Every representation removes less error\non held-out systems than on random splits", fontsize=8)
    handles = [Line2D([], [], marker="o", mfc="white", mec=C_A, color=C_A, ls="none", ms=4.2, label="A · random split"),
               Line2D([], [], marker="s", color=C_A, ls="none", ms=4.2, label="A · element-system split"),
               Line2D([], [], marker="o", mfc="white", mec=C_B, color=C_B, ls="none", ms=4.2, label="B · random split"),
               Line2D([], [], marker="s", color=C_B, ls="none", ms=4.2, label="B · element-system split")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 0.0), handlelength=1.0, columnspacing=1.4)
    fig.text(0.5, 0.125, "bars: 95% paired bootstrap of whole element systems (2 000 resamples), conditional on the fitted out-of-fold predictions", fontsize=6, color=C_META, ha="center", va="top")
    for ax, L in ((axa, "a"), (axb, "b")): panel_letter(ax, L, dx=-0.22 if ax is axa else -0.30, dy=1.12)
    return fig, D


# ================================================================================ FIGURE 3 =====
def figure3(csv, rc):
    D = {}
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.42, left=0.145, right=0.965, top=0.875, bottom=0.095)
    axa, axb, axc, axd = (fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1))
    fig.suptitle("The random-to-grouped gap in two curations of SuperCon: pooled and family-averaged error,\nscreening precision and neighbour distance", fontsize=8.2, y=0.99, x=0.01, ha="left")
    rows = [("A", "XGBoost"), ("A", "RandomForest"), ("B", "XGBoost"), ("B", "RandomForest")]
    ylab = [f"{d} · {m}" for d, m in rows]; ypos = np.arange(len(rows))[::-1]
    # (a) pooled MAE dumbbells with family CIs
    for yy, (ds, m) in zip(ypos, rows):
        r0, r1, inf = pfb(csv, ds, m, "MAE_random"), pfb(csv, ds, m, "MAE_grouped"), pfb(csv, ds, m, "inflacion_pct")
        D[f"a_random_{ds}_{m}"], D[f"a_grouped_{ds}_{m}"], D[f"a_infl_{ds}_{m}"] = float(r0.point), float(r1.point), float(inf.point)
        axa.plot([r0.point, r1.point], [yy, yy], lw=2.4, color="#C9C9C9", zorder=1, solid_capstyle="round")
        for r, col in ((r0, C_RANDOM), (r1, C_FAMILY)):
            axa.errorbar(r.point, yy, xerr=ci_err(r.point, r.lo, r.hi), fmt="o", ms=5.5, color=col, mec="white", mew=0.6, capsize=1.6, lw=0.8, zorder=3)
        axa.text(r1.hi + 0.12, yy, f"+{inf.point:.0f}%", va="center", ha="left", fontsize=6.6, color=C_FAMILY, fontweight="bold")
    axa.set_yticks(ypos); axa.set_yticklabels(ylab, fontsize=6.6); axa.set_ylim(-0.55, len(rows) - 0.02); axa.set_xlim(3.0, 11.6)
    axa.set_xlabel("Pooled MAE (K)   ·   lower = better"); axa.set_title("Pooled error rises 53\u201364% on unseen systems", fontsize=8)
    axa.legend(handles=[Line2D([], [], marker="o", color=C_RANDOM, ls="none", ms=5.5, label="Random split"), Line2D([], [], marker="o", color=C_FAMILY, ls="none", ms=5.5, label="Element-system split")],
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=6.3, handlelength=1.0, borderpad=0.1, columnspacing=1.0)
    # (b) pooled vs macro inflation
    for yy, (ds, m) in zip(ypos, rows):
        ip, im = pfb(csv, ds, m, "inflacion_pct"), pfb(csv, ds, m, "inflacion_macro_pct")
        D[f"b_pooled_{ds}_{m}"], D[f"b_macro_{ds}_{m}"] = float(ip.point), float(im.point)
        axb.errorbar(ip.point, yy + 0.16, xerr=ci_err(ip.point, ip.lo, ip.hi), fmt="o", ms=5, color=C_FAMILY, capsize=1.6, lw=0.8, zorder=3)
        axb.errorbar(im.point, yy - 0.16, xerr=ci_err(im.point, im.lo, im.hi), fmt="D", ms=4.2, color=_shade(C_FAMILY, 0.45), mec=C_FAMILY, capsize=1.6, lw=0.8, zorder=3)
    axb.set_yticks(ypos); axb.set_yticklabels(ylab, fontsize=6.6); axb.set_ylim(-0.55, len(rows) - 0.02); axb.set_xlim(10, 82)
    axb.set_xlabel("MAE inflation, grouped vs random (%)"); axb.set_title("Averaged per system the gap halves:\nit concentrates in the large systems", fontsize=8)
    axb.legend(handles=[Line2D([], [], marker="o", color=C_FAMILY, ls="none", ms=5, label="pooled over rows"), Line2D([], [], marker="D", color=_shade(C_FAMILY, 0.45), mec=C_FAMILY, ls="none", ms=4.2, label="mean of per-system MAEs (macro)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=6.3, handlelength=1.0, borderpad=0.1, columnspacing=1.0)
    # (c) top-100 precision (K=100, fractional under ties) with family CIs
    rankers = [(None, "Family-mean null", C_RANDOM), ("presence", "Element presence", C_PRESENCE), ("XGBoost", "XGBoost", C_GAIN)]
    w = 0.26; xs = np.arange(2)
    for di, ds in enumerate(["A", "Bv3"]):
        for j, (mod, lab, col) in enumerate(rankers):
            for prot, hatch, alpha in (("random", "///", 0.45), ("grouped", None, 1.0)):
                r = topk(csv, ds, mod, prot); D[f"c_{ds}_{mod}_{prot}"] = float(r.precision_pct)
                xx = di + (j - 1) * w + (-0.06 if prot == "random" else 0.06)
                axc.bar(xx, r.precision_pct, 0.11, color=col, alpha=alpha, hatch=hatch, edgecolor="white" if hatch else "none", lw=0.3, zorder=2, label=lab if (di == 0 and prot == "grouped") else None)
                axc.errorbar(xx, r.precision_pct, yerr=ci_err(r.precision_pct, r.precision_ci_fam_lo_pct, r.precision_ci_fam_hi_pct), fmt="none", ecolor="#333333", capsize=1.4, lw=0.6, zorder=4)
        prev = topk(csv, ds, None, "grouped").prevalence_pct; D[f"c_prev_{ds}"] = float(prev)
        axc.plot([di - 0.45, di + 0.45], [prev, prev], ls="--", lw=0.8, color="#333333", zorder=1)
    axc.text(0.99, 0.985, "dashed: fold prevalence of the high-$T_c$ class\n(expected precision of a random selector)", fontsize=6, color="#333333", ha="right", va="top", transform=axc.transAxes)
    axc.set_xticks(xs); axc.set_xticklabels(["Dataset A\n(Hamidieh)", "Dataset B\n(Stanev, audited)"], fontsize=6.8); axc.set_xlim(-0.55, 1.55)
    axc.set_ylim(0, 128); axc.set_yticks([0, 20, 40, 60, 80, 100]); axc.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)])
    axc.set_ylabel("Top-100 precision (fractional under ties)"); axc.set_title("Ranking skill survives on held-out systems;\nunder random splits even the null ranker excels", fontsize=8)
    h = [Patch(facecolor=c, label=l) for _, l, c in rankers] + [Patch(facecolor="#BBBBBB", hatch="///", edgecolor="white", label="random split (hatched)"), Patch(facecolor="#BBBBBB", label="element-system split")]
    axc.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3, fontsize=6.0, handlelength=1.1, borderpad=0.1, columnspacing=0.9, handletextpad=0.4)
    # (d) NN distances
    for di, ds in enumerate("AB"):
        dr, dg = rc["nn"][ds]; mr, mg = float(np.median(dr)), float(np.median(dg))
        D[f"d_random_{ds}"], D[f"d_family_{ds}"], D[f"d_ratio_{ds}"] = mr, mg, mg / mr
        for j, (vals, col) in enumerate([(dr, C_RANDOM), (dg, C_FAMILY)]):
            bp = axd.boxplot([vals], positions=[di + (j - 0.5) * 0.36], widths=0.30, showfliers=False, patch_artist=True, whis=(5, 95))
            bp["boxes"][0].set(facecolor=col, edgecolor="#3A3A3A", linewidth=0.7)
            for el in ("whiskers", "caps"):
                for a in bp[el]: a.set(color="#3A3A3A", linewidth=0.7)
            bp["medians"][0].set(color="black", linewidth=1.3)
        axd.text(di, 3.28, f"{mg/mr:.0f}\u00d7 farther", ha="center", va="bottom", fontsize=6.8, color=C_FAMILY, fontweight="bold")
    axd.set_xticks(xs); axd.set_xticklabels(["Dataset A\n(Hamidieh)", "Dataset B\n(Stanev, audited)"], fontsize=6.8); axd.set_xlim(-0.5, 1.5); axd.set_ylim(-0.12, 4.45)
    axd.set_ylabel("Distance to nearest training neighbour\n(standardised on training rows)"); axd.set_title("Why: random splits test on near-duplicate twins", fontsize=8)
    axd.legend(handles=[Patch(facecolor=C_RANDOM, edgecolor="#3A3A3A", lw=0.7, label="Random split"), Patch(facecolor=C_FAMILY, edgecolor="#3A3A3A", lw=0.7, label="Element-system split")],
               loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2, fontsize=6.3, handlelength=1.1, borderpad=0.1, columnspacing=1.1, handletextpad=0.4)
    axd.text(0.015, 0.985, "box: IQR, whiskers 5th\u201395th pct", transform=axd.transAxes, fontsize=6, color=C_META, va="top", ha="left")
    for ax, L in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")): panel_letter(ax, L, dx=-0.34 if ax in (axa, axb) else -0.26, dy=1.06)
    return fig, D


# ================================================================================ FIGURE 4 =====
def figure4(csv, rc):
    """(a) in-sample oracle eta2 (hatched) next to three CROSS-VALIDATED R^2 (grouped): full descriptors, element presence,
    family-mean null. NO ratio oracle/CV is formed anywhere. (b) N_eff under two explicitly defined estimators."""
    D = {}
    fig = plt.figure(figsize=(7.4, 3.8))
    gs = fig.add_gridspec(1, 2, wspace=0.40, left=0.08, right=0.985, top=0.78, bottom=0.33, width_ratios=[1.15, 1])
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    fig.suptitle("The corpus is clustered: an element-system label explains most of the in-sample variance, and the effective\nsample size depends on which estimator the design effect refers to", fontsize=8.2, y=1.0, x=0.01, ha="left")
    slots = [0, 1.15, 2.15, 3.15]; w = 0.34
    labels = ["oracle $\\eta^2$\n(in-sample,\nnot CV)", "full\ndescriptors", "element\npresence", "family-mean\nnull"]
    for di, (ds, dsk, col) in enumerate([("A", "A", _shade(C_GAIN, 0.45)), ("B", "B_v3", C_GAIN)]):
        eta = float(deff(csv, dsk, "(i)").eta2)
        full = float(abl(csv, ds, "grouped", "R2").R2); pres = float(abl(csv, ds, "grouped", "R0").R2)
        null = float(pfb(csv, ds, "FamilyMeanNull", "R2_grouped").point)
        D[f"a_eta_{ds}"], D[f"a_full_{ds}"], D[f"a_presence_{ds}"], D[f"a_fammean_{ds}"] = eta, full, pres, null
        xx = np.array(slots) + (di - 0.5) * w
        axa.bar(xx[0], eta, w, color="white", edgecolor=C_ORACLE, hatch="////", lw=0.8, zorder=2)
        axa.bar(xx[1:3], [full, pres], w, color=col, edgecolor="none", zorder=2)
        if null > 0.02: axa.bar(xx[3], null, w, color=col, edgecolor="none", zorder=2)
        else: axa.plot([xx[3] - w / 2, xx[3] + w / 2], [0, 0], lw=1.8, color=col, solid_capstyle="butt", zorder=4)
        for x_, v in zip(xx, (eta, full, pres)): axa.text(x_, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3)
        axa.text(xx[3], 0.03 + 0.09 * di, f"{null:.3f}", ha="center", va="bottom", fontsize=6.3, color=col if di else "#666666")
        d = abld(csv, ds, "grouped", "R0", "R2"); D[f"a_diff_{ds}"] = float(d.dR2)
    axa.axhline(0, lw=0.7, color="#777777", zorder=3); axa.axvline(0.575, ls=":", lw=0.8, color=C_META)
    axa.set_xticks(slots); axa.set_xticklabels(labels, fontsize=6.3); axa.set_ylim(-0.06, 1.12); axa.set_ylabel("$R^2$"); axa.set_xlim(-0.55, 3.7)
    axa.set_title("Grouped-CV $R^2$ versus nulls; the oracle is no competitor", fontsize=8)
    axa.text(0.575 + 0.05, 1.04, "cross-validated, element-system hold-out \u2192", fontsize=6, color=C_META, ha="left", va="bottom")
    axa.legend(handles=[Patch(facecolor=_shade(C_GAIN, 0.45), label="Dataset A"), Patch(facecolor=C_GAIN, label="Dataset B (audited)"),
                        Patch(facecolor="white", edgecolor=C_ORACLE, hatch="////", label="in-sample oracle (family means on all data)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.30), ncols=3, fontsize=6.2, handlelength=1.1, borderpad=0.1, columnspacing=1.0)
    # (b) N_eff by estimator (log axis, points, no bars)
    ests = [("(i)", "row-weighted\nmean"), ("(ii)", "mean of\nsystem means")]
    yb = np.array([1.0, 0.0])
    for di, (ds, dsk, col, mk) in enumerate([("A", "A", _shade(C_GAIN, 0.45), "o"), ("B", "B_v3", C_GAIN, "s")]):
        r = deff(csv, dsk, "(i)"); N, k = int(r.N), int(r.k); D[f"b_N_{ds}"], D[f"b_k_{ds}"] = N, k
        for yy, (pre, _) in zip(yb, ests):
            ne = float(deff(csv, dsk, pre).N_eff); D[f"b_neff_{pre}_{ds}"] = ne
            axb.plot([ne], [yy + (0.12 if di == 0 else -0.12)], marker=mk, ms=6, color=col, ls="none", zorder=3)
            axb.text(ne * 1.13, yy + (0.12 if di == 0 else -0.12), f"{ne:,.0f}", fontsize=6.3, ha="left", va="center", color=C_TEXT)
        axb.axvline(k, color=col, lw=1.0, ls="--", zorder=1); axb.axvline(N, color=col, lw=1.0, ls="-", alpha=0.5, zorder=1)
        bb = dict(fc="white", ec="none", pad=0.6)
        axb.text(k, 1.55 + 0.28 * di, f"{ds}: $k$ = {k:,}", fontsize=6.2, color=C_TEXT, ha="center", va="bottom", bbox=bb, zorder=5)
        axb.text(N, 1.55 + 0.28 * di, f"{ds}: $N$ = {N:,}", fontsize=6.2, color=C_TEXT, ha="center", va="bottom", bbox=bb, zorder=5)
    axb.set_xscale("log"); axb.set_xlim(250, 40000); axb.set_xticks([300, 1000, 3000, 10000, 30000]); axb.set_xticklabels(["300", "1k", "3k", "10k", "30k"])
    axb.set_yticks(yb); axb.set_yticklabels([e[1] for e in ests], fontsize=6.3); axb.set_ylim(-0.55, 2.25)
    axb.set_xlabel("$N_{\\mathrm{eff}} = N/\\mathrm{deff}$ of the estimator of mean $T_c$")
    axb.set_title("Same ICC, two estimators: $N_{\\mathrm{eff}}$ differs tenfold", fontsize=8)
    axb.legend(handles=[Line2D([], [], marker="o", color=_shade(C_GAIN, 0.45), ls="none", ms=6, label="Dataset A"), Line2D([], [], marker="s", color=C_GAIN, ls="none", ms=6, label="Dataset B (audited)"),
                        Line2D([], [], color="#555555", ls="--", label="$k$ systems"), Line2D([], [], color="#555555", ls="-", alpha=0.5, label="$N$ rows")],
               loc="upper center", bbox_to_anchor=(0.5, -0.30), ncols=2, fontsize=6.2, handlelength=1.3, borderpad=0.1, columnspacing=1.0)
    for ax, L in ((axa, "a"), (axb, "b")): panel_letter(ax, L, dx=-0.10 if ax is axa else -0.27, dy=1.08)
    return fig, D


# ================================================================================ verify ======
def verify(csv, rc, D1, D2, D3, D4, tol=5e-9):
    fails = []
    def chk(label, got, want, t=tol):
        if abs(float(got) - float(want)) > t: fails.append((label, got, want)); print(f"  MISMATCH {label}: drawn {got} vs csv {want}")
    o = csv["oof"]; r = o[(o.dataset == "A") & (o.model == "XGBoost") & (o.protocol == "grouped")].iloc[0]
    chk("fig1c MAE", D1["c_mae"], r.MAE, 1e-6); chk("fig1c R2", D1["c_r2"], r.R2, 1e-6)
    chk("fig1a k", D1["a_k"], rc["k_A"]); chk("fig1a k=3365", D1["a_k"], 3365); chk("fig1a max", D1["a_max"], 720)
    for ds in "AB":
        n = csv["nn"].loc[ds]
        chk(f"nn {ds} random", D3[f"d_random_{ds}"], n.nn_median_random_trainscaler, 2e-4); chk(f"nn {ds} grouped", D3[f"d_family_{ds}"], n.nn_median_grouped_trainscaler, 2e-4)
        chk(f"nn {ds} n_random", len(rc["nn"][ds][0]), n.n_test_random); chk(f"nn {ds} n_grouped", len(rc["nn"][ds][1]), n.n_test_grouped)
    chk("fig1d==fig3d A", D1["d_median_family"], D3["d_family_A"])
    for ds in "AB":
        for m in ("XGBoost", "RandomForest"):
            chk(f"fig3a {ds} {m}", D3[f"a_grouped_{ds}_{m}"], pfb(csv, ds, m, "MAE_grouped").point); chk(f"fig3b macro {ds} {m}", D3[f"b_macro_{ds}_{m}"], pfb(csv, ds, m, "inflacion_macro_pct").point)
    for ds in "AB":
        assert D4[f"a_full_{ds}"] > D4[f"a_presence_{ds}"] > D4[f"a_fammean_{ds}"], "ordering full > presence > null broken"
        chk(f"fig4 diff {ds}", D4[f"a_full_{ds}"] - D4[f"a_presence_{ds}"], D4[f"a_diff_{ds}"], 1e-6)
        chk(f"fig4b Neff(i) {ds}", D4[f"b_neff_(i)_{ds}"], deff(csv, "A" if ds == "A" else "B_v3", "(i)").N_eff)
    chk("fig4b N_B v3", D4["b_N_B"], 12336); chk("fig4b k_B v3", D4["b_k_B"], 3046)
    for (ds, prot) in CONDS:
        for a, b, _ in STEPS: chk(f"fig2a {ds} {prot} {a}{b}", D2[f"a_{ds}_{prot}_{a}{b}"], abld(csv, ds, prot, a, b).dMAE)
    import inspect
    src = inspect.getsource(figure4)
    assert "eta / " not in src and "/ eta" not in src, "figure 4 must not form an oracle/CV ratio"
    assert not fails, f"{len(fails)} drawn values disagree with the canonical CSVs"
    print(f"[verify] {len(D1) + len(D2) + len(D3) + len(D4)} drawn quantities; all asserted values agree with results_v3/results_v2 CSVs")


def main():
    global SUBMIT
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", type=Path, default=None); ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    if a.submit: SUBMIT = a.submit.resolve()
    apply_figure_style()
    csv = load_csvs(); rc = recompute()
    figs = {}
    fig1, D1 = figure1(csv, rc); figs["figure1.png"] = fig1
    fig2, D2 = figure2(csv, rc); figs["figure2.png"] = fig2
    fig3, D3 = figure3(csv, rc); figs["figure3.png"] = fig3
    fig4, D4 = figure4(csv, rc); figs["figure4.png"] = fig4
    n_over = 0
    for name, fig in figs.items():
        save(fig, name); n_over += len(check_overlaps(fig, name))
    if not a.no_verify: verify(csv, rc, D1, D2, D3, D4)
    assert n_over == 0, f"geometric check found {n_over} overlapping/clipped labels"
    print("[done]")


if __name__ == "__main__":
    main()
