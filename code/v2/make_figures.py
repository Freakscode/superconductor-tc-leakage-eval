#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — Regenerate figures 1-4 of the manuscript
"Leakage-aware evaluation of composition-based Tc prediction" (v2).

WHY THIS FILE EXISTS
--------------------
The v1 repository shipped figures/figure1_anatomy.png, figure2_null_decomposition.png
and figure3_summary.png but no plotting code: not a single matplotlib call under code/.
A reviewer could not regenerate them, and several numbers they displayed have since been
re-measured (e.g. Dataset B grouped top-100 is 85.2%, not the 81.4% of the v1 draft).
This script is the single, deterministic entry point that rebuilds all four figures.

WHERE THE NUMBERS COME FROM
---------------------------
Every summary number drawn is READ from the canonical measurement CSVs in results_v2/
(measured by run_v2_measurements.py / run_tabla2_shuffle.py / agg_tabla2.py):

  top100_unificado.csv   top-100 screening precision, canonical threshold definition
  nulo_presencia.csv     MAE/RMSE/R2/top-100 for full features, element-presence null
                         and family-mean null; both datasets, both CV schemes
  design_effect.csv      oracle eta2, ICC(1), design effect, effective N
  tabla2_shuffle.csv     5-seed means with genuinely independent grouped partitions
  tabla2_runs_por_semilla.csv   per-seed MAE/R2 (raw points overlaid on Fig 1b)

Only the panels that CANNOT come from those CSVs are recomputed here from data/ in the
repository, using the published hyperparameters imported from code/ (never retyped):
  Fig 1a  family-size distribution           (grouping = tc_pipeline.chemical_families)
  Fig 1c  parity plot, pooled out-of-fold    (XGBoost, GroupKFold(5), seed 42)
  Fig 1d / Fig 3d  nearest-neighbour distance in standardized feature space

Nothing is typed by hand. The FIDELITY ASSERTS at the bottom re-open every CSV and check
each drawn value against its source, so the script fails loudly if a figure and the
measurements ever drift apart.

DETERMINISM
-----------
Single seed (42) for every recomputation; GroupKFold(5) without shuffle for the
deterministic partition; no parallel non-determinism (XGBoost tree_method="hist",
fixed n_jobs). Runs end-to-end unattended.

USAGE
-----
    python make_figures.py                       # writes figures/figure1..4.png (+ copies in results_v2/)
    python make_figures.py --submit ../paper1_submission   # ALSO write the PNGs to that folder
    python make_figures.py --dry-run             # resolve paths / check inputs, draw nothing
    SUPERCON_ROOT=/path/to/repo python make_figures.py     # explicit repository root

Environment: supercon-repro (numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2,
xgboost 3.0.5, matminer 0.9.2, matplotlib 3.10.6).
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
# --- repository root (portable) -------------------------------------------------------
# ROOT = the repository root. Override with the SUPERCON_ROOT environment variable;
# by default it is resolved from this file's location: code/v2/<script>.py -> parents[2].
ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
REPO = ROOT                       # the repository itself (code/, data/, results/, figures/)
CODE = ROOT / "code"              # tc_pipeline.py, datasetB_pipeline.py (hyperparameters live here)
DATA = ROOT / "data"
OUT  = ROOT / "results_v2"        # v2 measurements (inside the repository)
HERE = OUT                        # CSVs of provenance are read from here; PNG copies land here too
FIG  = ROOT / "figures"           # primary PNG output (figures/figure1..4.png)
SUBMIT = FIG                      # may be overridden with --submit DIR (e.g. the manuscript folder)
SEED = 42
DPI = 300

# --------------------------------------------------------------------------------------
# Palette — one colour per ENTITY, threaded across all four figures (figure-style §4.1)
# --------------------------------------------------------------------------------------
C_RANDOM = "#9E9E9E"   # random split / nominal count  (neutral, never the focal series)
C_FAMILY = "#C2452D"   # chemical-family split         (focal, vermillion; CVD-safe vs grey)
C_GAIN = "#2C6FAD"     # gain from features + model / full-feature CV R2
C_PRESENCE = "#E08214" # element-presence null
C_ORACLE = "#3F3F3F"   # oracle eta2 ceiling (hatched everywhere — NOT a CV quantity)
C_TEXT = "#333333"
C_META = "#777777"


def _shade(hex_color, lighten=0.0):
    """Sample within a hue family: lighten=0 is the base colour, 1.0 is white (§4.3)."""
    rgb = np.array(mpl.colors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * float(lighten))


# --------------------------------------------------------------------------------------
# figure-style helpers. Loaded from the skill when available (this script is normally run
# inside the session where skill('figure-style') is active); otherwise the local fallbacks
# below reproduce the same behaviour so the script runs standalone for a reviewer.
# --------------------------------------------------------------------------------------
def _resolve_style_helpers():
    g = globals()
    have = {}
    for nm in ("apply_figure_style", "panel_letter", "panel_crops"):
        have[nm] = g.get(nm) or g.get("figure_style__" + nm)
    if all(have.values()):
        return have["apply_figure_style"], have["panel_letter"], have["panel_crops"]

    def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
        base, ann, tick = sizes
        mpl.rcParams.update({
            "figure.dpi": 110, "savefig.dpi": DPI, "savefig.bbox": "tight",
            "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
            "legend.fontsize": ann, "xtick.labelsize": tick, "ytick.labelsize": tick,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.titlelocation": "left", "axes.titleweight": "normal",
            "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
            "xtick.color": C_TEXT, "ytick.color": C_TEXT,
            "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
            "legend.frameon": False, "axes.grid": grid,
            "figure.facecolor": "white", "savefig.facecolor": "white",
        })

    def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
        t = letter.lower() if case == "lower" else letter.upper()
        return ax.text(dx, dy, t, transform=ax.transAxes, fontweight="bold",
                       fontsize=(fontsize or mpl.rcParams["font.size"] + 2),
                       va="bottom", ha="left")

    def panel_crops(fig, dpi=None, pad_px=6, bbox_inches=None, pad_inches=None):
        dpi = dpi or fig.get_dpi()
        W, H = fig.get_size_inches() * dpi
        out = {}
        for i, ax in enumerate(fig.axes):
            bb = ax.get_tightbbox(fig.canvas.get_renderer())
            out[chr(ord("a") + i)] = (max(0, bb.x0 - pad_px), max(0, H - bb.y1 - pad_px),
                                      min(W, bb.x1 + pad_px), min(H, H - bb.y0 + pad_px))
        return out

    return apply_figure_style, panel_letter, panel_crops


apply_figure_style, panel_letter, panel_crops = _resolve_style_helpers()


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def _import_from_path(name: str, path: Path):
    """Import the paper's pipeline modules so hyperparameters are NEVER retyped here."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_csvs() -> dict:
    """The canonical v2 measurements. Single source of truth for every summary number."""
    csv = {
        "top100": pd.read_csv(HERE / "top100_unificado.csv"),
        "nulo": pd.read_csv(HERE / "nulo_presencia.csv"),
        "design": pd.read_csv(HERE / "design_effect.csv"),
        "shuffle": pd.read_csv(HERE / "tabla2_shuffle.csv"),
        "perseed": pd.read_csv(HERE / "tabla2_runs_por_semilla.csv"),
    }
    assert len(csv["top100"]) == 4 and len(csv["design"]) == 2 and len(csv["shuffle"]) == 4
    assert len(csv["nulo"]) == 12
    return csv


def nulo(csv, dataset, rep, split):
    """One row of nulo_presencia.csv. rep in {features_completas_paper, presencia_elementos,
    nulo_media_familia}; split in {aleatorio_KFold5, GroupKFold5_familias}."""
    d = csv["nulo"]
    r = d[(d.dataset == dataset) & (d.representacion == rep) & (d.split == split)]
    assert len(r) == 1, (dataset, rep, split, len(r))
    return r.iloc[0]


def top100(csv, dataset, split):
    d = csv["top100"]
    r = d[(d.dataset == f"Dataset {dataset}") & (d.split == split)]
    assert len(r) == 1, (dataset, split, len(r))
    return r.iloc[0]


def shuf(csv, dataset, model):
    d = csv["shuffle"]
    r = d[(d.dataset == dataset) & (d.modelo == model)]
    assert len(r) == 1
    return r.iloc[0]


def design(csv, dataset):
    r = csv["design"][csv["design"].dataset == dataset]
    assert len(r) == 1
    return r.iloc[0]


# --------------------------------------------------------------------------------------
# Recomputation of the three panels that are not in any CSV
# --------------------------------------------------------------------------------------
def recompute(verbose=True) -> dict:
    """Rebuild the raw-data quantities: family-size distribution (A), pooled out-of-fold
    parity predictions (A), and nearest-training-neighbour distances (A and B).

    Dataset A grouping and hyperparameters come from code/tc_pipeline.py; Dataset B
    features come from the Magpie checkpoint and its hyperparameters from
    code/datasetB_pipeline.py. Nothing here is a hand-entered constant."""
    t0 = time.time()
    tcp = _import_from_path("tc_pipeline", CODE / "tc_pipeline.py")
    dbp = _import_from_path("datasetB_pipeline", CODE / "datasetB_pipeline.py")

    # ---- Dataset A: 81 Hamidieh statistics + element-set families
    feat = pd.read_csv(DATA / "train.csv")
    uniq = pd.read_csv(DATA / "unique_m.csv")
    assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "files not row-aligned"
    XA = feat.drop(columns=["critical_temp"]).values
    yA = feat["critical_temp"].values
    gA = tcp.chemical_families(uniq)

    # family sizes, descending, plus readable names for the three largest
    elem_cols = [c for c in uniq.columns if c not in ("critical_temp", "material")]
    fam_key = (uniq[elem_cols] > 0).apply(
        lambda r: "|".join(sorted(e for e in elem_cols if r[e])), axis=1)
    vc = fam_key.value_counts()
    sizes_A = vc.values.astype(int)
    top_fams = [(k, int(v)) for k, v in vc.head(3).items()]

    # ---- Dataset B: Magpie checkpoint from the repository (float32, bitwise as shipped)
    npzB = np.load(DATA / "datasetB_featurized.npz", allow_pickle=True)
    XB, yB, gB_npz = npzB["X"], npzB["y"], npzB["groups"]

    # Dataset B nearest-neighbour source. The family codes are ALWAYS the contiguous `groups` of
    # the shipped npz (factorize AFTER the Tc>0 filter; the reference partition, see
    # splits/datasetB_folds.csv). v2-draft versions of this script took the codes from the live
    # file (`groups_all[m]`, factorized BEFORE the filter -> gapped codes). Both code sets define
    # the SAME family partition, but GroupShuffleSplit(random_state=42) holds out DIFFERENT
    # families for the two numberings: measured 2026-09-02 (code/v2, sklearn 1.7.2) --
    #   gapped codes (v1 convention):  grouped median 1.2385, 2 845 test rows, ratio 8.9x
    #   npz codes (reference):         grouped median 1.5841, 2 477 test rows, ratio 11.3x
    # and float32 vs float64 features change the medians only in the 8th decimal. The manuscript
    # text and results_v2/nn_distance.csv use the npz codes (1.584 / 0.140 -> 11.3x); the
    # figure-3(d) panel of the 2026-08-21 draft PNG showed 8.9x because of the gapped codes. An
    # earlier comment here claimed the two labellings were equivalent for this panel; it was
    # wrong and is corrected. The live file, when present, only supplies float64 features.
    live = HERE / "datasetB_featurized_live.npz"
    if live.exists():
        L = np.load(live, allow_pickle=True)
        assert np.array_equal(L["y"], yB), "live file not row-aligned with the npz"
        XB_nn, yB_nn, gB_nn = L["X"], L["y"], gB_npz
        nn_source_B = "datasetB_featurized_live.npz features (float64) + npz `groups` (reference codes)"
    else:
        XB_nn, yB_nn, gB_nn = XB, yB, gB_npz
        nn_source_B = "data/datasetB_featurized.npz (float32 features, reference `groups`)"

    # ---- Fig 1c: pooled out-of-fold predictions, XGBoost, deterministic GroupKFold(5)
    ypA = tcp.cv_predict_manual(lambda: tcp.make_models(SEED)["XGBoost"],
                                XA, yA, GroupKFold(5), gA)
    parity = dict(y=yA, yp=ypA,
                  mae=float(mean_absolute_error(yA, ypA)),
                  r2=float(r2_score(yA, ypA)), n=int(len(yA)))

    # ---- Fig 1d / 3d: distance to nearest TRAINING material in standardized feature space
    def nn_dists(X, y, groups):
        Xs = StandardScaler().fit_transform(X)
        near = lambda a, b: NearestNeighbors(n_neighbors=1).fit(Xs[a]).kneighbors(Xs[b])[0].ravel()
        ri = np.arange(len(X))
        np.random.RandomState(SEED).shuffle(ri)
        cut = int(0.8 * len(X))
        g_tr, g_te = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                            random_state=SEED).split(X, y, groups))
        assert not (set(groups[g_tr]) & set(groups[g_te])), "family overlap in grouped hold-out"
        return near(ri[:cut], ri[cut:]), near(g_tr, g_te)

    dA_r, dA_g = nn_dists(XA, yA, gA)
    dB_r, dB_g = nn_dists(XB_nn, yB_nn, gB_nn)

    out = {
        "sizes_A": sizes_A, "top_fams_A": top_fams,
        "N_A": int(len(yA)), "k_A": int(len(sizes_A)),
        "parity": parity,
        "nn": {"A": (dA_r, dA_g), "B": (dB_r, dB_g)},
        "nn_source_B": nn_source_B,
        "N_B": int(len(yB)), "k_B": int(len(set(gB_npz.tolist()))),
        "seconds": time.time() - t0,
    }
    if verbose:
        print(f"[recompute] A: N={out['N_A']} k={out['k_A']} largest={sizes_A.max()} "
              f"singletons={int((sizes_A == 1).sum())}")
        print(f"[recompute] parity (A, XGBoost, GroupKFold(5) pooled OOF): "
              f"MAE={parity['mae']:.4f} R2={parity['r2']:.4f}")
        for nm, (r_, g_) in out["nn"].items():
            print(f"[recompute] NN {nm}: median random={np.median(r_):.4f} "
                  f"family={np.median(g_):.4f} ratio={np.median(g_)/np.median(r_):.2f}x")
        print(f"[recompute] Dataset B NN source: {nn_source_B}")
        print(f"[recompute] {out['seconds']:.0f}s")
    return out


# --------------------------------------------------------------------------------------
# §9.1 geometric check — no text/text or text/spine overlaps, everything inside the canvas
# --------------------------------------------------------------------------------------
def _offview_ticklabels(fig):
    """Tick labels whose tick sits outside its axis view interval. Matplotlib keeps these
    as live Text objects but never renders them, so they are not real overlap findings."""
    dead = set()
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())),
                               (ax.yaxis, sorted(ax.get_ylim()))):
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if not (lo - 1e-9 <= loc <= hi + 1e-9):
                    dead.add(tick.label1)
                    dead.add(tick.label2)
    return dead


def _text_box(t, r):
    """Extent of the glyphs only. Annotation.get_window_extent unions in the arrow patch,
    so a label with a long leader would report a false overlap; a leader crossing a label
    is a §9.2 perceptual question, not a §9.1 text-box collision."""
    return mpl.text.Text.get_window_extent(t, r)


def check_overlaps(fig, label=""):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dead = _offview_ticklabels(fig)
    texts = [(t, _text_box(t, r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in dead]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes
              for s in ax.spines.values() if s.get_visible()]
    ticklabels = {ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"))
                  for ax in fig.axes}
    findings = []
    for i, (ta, ba) in enumerate(texts):
        for tb, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                findings.append(("text/text", ta.get_text()[:38], tb.get_text()[:38]))
    for t, bt in texts:
        for s, bs in spines:
            if bt.overlaps(bs) and t not in ticklabels.get(s.axes, ()):
                findings.append(("text/spine", t.get_text()[:38], s.spine_type))
    fb = fig.bbox
    for t, bt in texts:
        if not (bt.x0 >= fb.x0 - 1 and bt.y0 >= fb.y0 - 1
                and bt.x1 <= fb.x1 + 1 and bt.y1 <= fb.y1 + 1):
            findings.append(("outside-canvas", t.get_text()[:38], ""))
    print(f"[overlap] {label}: {'CLEAN' if not findings else str(len(findings)) + ' FINDINGS'}")
    for f in findings:
        print("   ", f)
    return findings


def saved_panel_crops(fig, save_dpi=DPI, pad_inches=None, pad_px=10):
    """Per-panel crop boxes in PIXEL coordinates OF THE SAVED PNG.

    panel_crops() assumes the PNG spans the full figure; these figures are written with
    bbox_inches="tight", which re-origins and rescales the canvas, so panel boxes must be
    mapped through the tight bbox (in inches) and the save dpi or the crops land on the
    wrong panel."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = mpl.rcParams["savefig.pad_inches"] if pad_inches is None else pad_inches
    tb = fig.get_tightbbox(r)                       # inches
    x0i, y0i, x1i, y1i = tb.x0 - pad, tb.y0 - pad, tb.x1 + pad, tb.y1 + pad
    W = (x1i - x0i) * save_dpi
    H = (y1i - y0i) * save_dpi
    out = {}
    for i, ax in enumerate(fig.axes):
        bb = ax.get_tightbbox(r)                    # display px at fig.dpi
        ax0, ax1 = bb.x0 / fig.dpi, bb.x1 / fig.dpi
        ay0, ay1 = bb.y0 / fig.dpi, bb.y1 / fig.dpi
        left = (ax0 - x0i) * save_dpi - pad_px
        right = (ax1 - x0i) * save_dpi + pad_px
        top = (y1i - ay1) * save_dpi - pad_px       # PNG y grows downward
        bot = (y1i - ay0) * save_dpi + pad_px
        out[chr(ord("a") + i)] = (max(0, round(left)), max(0, round(top)),
                                  min(round(W), round(right)), min(round(H), round(bot)))
    return out


def save(fig, name, drawn_note=""):
    """Write the PNG to figures/ (FIG), keep a copy in results_v2/ (HERE) and, if --submit was
    given, a third copy in that folder (SUBMIT)."""
    FIG.mkdir(parents=True, exist_ok=True); HERE.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    shutil.copyfile(p, HERE / name)
    if SUBMIT != FIG:
        SUBMIT.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, SUBMIT / name)
    print(f"[save] {p}  (+ copy in results_v2/{' + ' + str(SUBMIT) if SUBMIT != FIG else ''}) {drawn_note}")
    return p


# ======================================================================================
# FIGURE 1 — Anatomy of chemical data leakage (Dataset A)
#   (a) family-size distribution      recomputed from data/
#   (b) MAE random vs family by model READ from tabla2_shuffle.csv (5-seed means)
#   (c) parity under family hold-out  recomputed, verified against nulo_presencia.csv
#   (d) nearest-neighbour distance    recomputed from data/
# ======================================================================================
def figure1(csv, rc) -> dict:
    drawn = {}
    fig = plt.figure(figsize=(7.4, 5.9))
    gs = fig.add_gridspec(2, 2, hspace=0.66, wspace=0.36,
                          left=0.115, right=0.975, top=0.875, bottom=0.125)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    N_A, k_A = rc["N_A"], rc["k_A"]
    fig.suptitle(f"Anatomy of chemical data leakage in composition-based "
                 f"$T_c$ prediction (Dataset A, $N$ = {N_A:,})",
                 fontsize=9, y=0.975, x=0.5, ha="center")

    # ---------------- (a) family-size distribution -------------------------------------
    sizes = rc["sizes_A"]
    rank = np.arange(1, len(sizes) + 1)
    share_ge2 = sizes[sizes >= 2].sum() / sizes.sum()
    drawn["a_share_ge2"] = float(share_ge2)
    drawn["a_k"] = int(len(sizes))
    drawn["a_max"] = int(sizes.max())

    axa.fill_between(rank, 1, sizes, step="post", color="#D8D8D8",
                     edgecolor="#606060", linewidth=0.6)
    axa.set_yscale("log")
    axa.set_ylim(0.8, 1900)
    axa.set_yticks([1, 10, 100, 1000])
    axa.set_yticklabels(["1", "10", "100", "1k"])
    axa.set_xlim(-60, len(sizes) + 120)
    axa.set_xlabel("Chemical family (rank by size)")
    axa.set_ylabel("Materials in family")
    share_ge5 = sizes[sizes >= 5].sum() / sizes.sum()
    drawn["a_share_ge5"] = float(share_ge5)
    axa.set_title(f"{share_ge2*100:.0f}% of materials share their\n"
                  f"element set with another material", fontsize=8)
    # name the three largest families in one block with a single leader to the head of the
    # distribution: three separate leaders would converge on the same few pixels and cross (§6.9)
    block = "three largest families\n" + "\n".join(
        "{" + ", ".join(k.split("|")) + f"}}: {c}" for k, c in rc["top_fams_A"])
    axa.annotate(block, xy=(rank[1], sizes[1]), xytext=(len(sizes) * 0.20, 300),
                 fontsize=6, color=C_TEXT, va="center", ha="left", linespacing=1.45,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=C_META,
                                 shrinkA=2, shrinkB=1.5))

    # ---------------- (b) MAE random vs family, per model ------------------------------
    models = ["RandomForest", "XGBoost"]
    x = np.arange(len(models))
    w = 0.34
    for j, (scheme, col, lab, key) in enumerate([
            ("random", C_RANDOM, "Random split", "MAE_random"),
            ("grouped", C_FAMILY, "Chemical-family split", "MAE_grouped")]):
        vals = [shuf(csv, "A", m)[f"{key}_mean"] for m in models]
        errs = [shuf(csv, "A", m)[f"{key}_std"] for m in models]
        drawn[f"b_{scheme}"] = [float(v) for v in vals]
        axb.bar(x + (j - 0.5) * w, vals, w, yerr=errs, capsize=2.4,
                color=col, edgecolor="none", label=lab,
                error_kw=dict(lw=0.7, ecolor="#444444"))
        # raw per-seed points behind the mean (§6.1)
        pr = csv["perseed"]
        arm = "random_shuffle" if scheme == "random" else "grouped_shuffle"
        for i, m in enumerate(models):
            pts = pr[(pr.dataset == "A") & (pr.modelo == m) & (pr.scheme == arm)]["MAE"].values
            axb.scatter(np.full(len(pts), x[i] + (j - 0.5) * w), pts, s=5,
                        facecolor="white", edgecolor="#333333", linewidth=0.5, zorder=5)
        for i, v in enumerate(vals):
            axb.text(x[i] + (j - 0.5) * w, v + errs[i] + 0.28, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7)

    infl = [shuf(csv, "A", m)["inflacion_pct_mean"] for m in models]
    drawn["b_inflation"] = [float(v) for v in infl]
    axb.set_xticks(x)
    axb.set_xticklabels(models)
    axb.set_ylim(0, 11.0)
    axb.set_ylabel("Test MAE (K) · 5-fold CV")
    axb.set_title(f"Family-aware validation: error rises "
                  f"{min(infl):.0f}\u2013{max(infl):.0f}%", fontsize=8)
    axb.text(0.985, 0.965, "lower = better", transform=axb.transAxes,
             ha="right", va="top", fontsize=6, color=C_META)
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.135), ncols=2, fontsize=6.5,
               handlelength=1.1, borderpad=0.1, columnspacing=1.4, handletextpad=0.5)
    # ---------------- (c) parity plot under family hold-out ----------------------------
    p = rc["parity"]
    drawn["c_mae"], drawn["c_r2"], drawn["c_n"] = p["mae"], p["r2"], p["n"]
    lim = 180
    axc.scatter(p["y"], p["yp"], s=3.2, alpha=0.10, color=C_FAMILY,
                edgecolor="none", rasterized=True)
    axc.plot([0, lim], [0, lim], ls="--", lw=0.8, color="#333333", label="perfect")
    axc.set_xlim(-6, lim + 6)
    axc.set_ylim(-6, lim + 6)
    axc.set_xticks([0, 40, 80, 120, 160])
    axc.set_yticks([0, 40, 80, 120, 160])
    axc.set_aspect("equal")
    axc.set_xlabel("Actual $T_c$ (K)")
    axc.set_ylabel("Predicted $T_c$ (K)")
    axc.set_title("XGBoost under chemical-family hold-out", fontsize=8)
    axc.text(0.035, 0.95, f"held-out families, pooled out-of-fold\n"
                          f"MAE = {p['mae']:.1f} K   $R^2$ = {p['r2']:.2f}",
             transform=axc.transAxes, fontsize=6.2, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", lw=0.5))
    axc.legend(loc="lower right", fontsize=6.5, handlelength=1.4, borderpad=0.1)

    # ---------------- (d) nearest-neighbour distance histogram -------------------------
    dr, dg = rc["nn"]["A"]
    mr, mg = float(np.median(dr)), float(np.median(dg))
    drawn["d_median_random"], drawn["d_median_family"] = mr, mg
    drawn["d_ratio"] = mg / mr
    clip = 3.0
    bins = np.linspace(0, clip, 46)
    axd.hist(np.clip(dr, 0, clip), bins=bins, color=C_RANDOM, alpha=0.85,
             label="Random split", edgecolor="none")
    axd.hist(np.clip(dg, 0, clip), bins=bins, color=C_FAMILY, alpha=0.62,
             label="Chemical-family split", edgecolor="none")
    axd.axvline(mr, ls="--", lw=1.0, color="#4A4A4A")
    axd.axvline(mg, ls="--", lw=1.0, color=C_FAMILY)
    axd.set_xlim(-0.08, clip + 0.08)
    axd.set_ylim(0, 2450)
    axd.annotate(f"median {mr:.3f}", xy=(mr, 1320), xytext=(mr + 0.28, 1620),
                 fontsize=6.5, color="#4A4A4A", va="center", ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=C_META, shrinkA=0, shrinkB=1))
    axd.text(mg + 0.07, 1100, f"median {mg:.2f}", fontsize=6.5, color=C_FAMILY,
             va="center", ha="left")
    axd.set_xlabel("Distance to nearest training material\n"
                   "(standardized feature space; clipped at 3)")
    axd.set_ylabel("Test materials")
    axd.set_title(f"Random splits test on near-twins ({mg/mr:.0f}\u00d7 closer)", fontsize=8)
    axd.legend(loc="upper right", fontsize=6.5, handlelength=1.1, borderpad=0.1,
               labelspacing=0.25)

    for ax, L in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
        panel_letter(ax, L, dx=-0.20 if ax in (axa, axc) else -0.155, dy=1.12)

    return fig, drawn


# ======================================================================================
# FIGURE 2 — Null-model decomposition
#   (a) R2 of the featureless family-mean null vs the full model  -> nulo_presencia.csv
#   (b) top-100 precision of the null ranker vs XGBoost           -> nulo_presencia.csv
#                                                                    + top100_unificado.csv
# ======================================================================================
def figure2(csv, rc) -> dict:
    drawn = {}
    fig = plt.figure(figsize=(7.4, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.085, right=0.985, top=0.83, bottom=0.275)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    fig.suptitle("Null-model decomposition: a featureless family-mean predictor separates "
                 "memorization from genuine skill", fontsize=8.5, y=0.99)

    schemes = [("aleatorio_KFold5", "random"), ("GroupKFold5_familias", "family-\naware")]
    slots, labels, dsets = [], [], []
    for di, ds in enumerate("AB"):
        for si, (skey, slab) in enumerate(schemes):
            slots.append(di * 2.55 + si)
            labels.append(slab)
            dsets.append((ds, skey))

    # ---------------- (a) how much of the headline R2 is memorization? -----------------
    for xpos, (ds, skey) in zip(slots, dsets):
        full = nulo(csv, ds, "features_completas_paper", skey).R2
        null = nulo(csv, ds, "nulo_media_familia", skey).R2
        drawn[f"a_full_{ds}_{skey}"] = float(full)
        drawn[f"a_null_{ds}_{skey}"] = float(null)
        floor = max(null, 0.0)                      # a negative R2 has no stackable height
        axa.bar(xpos, floor, 0.74, color=C_RANDOM, edgecolor="none", zorder=2)
        axa.bar(xpos, full - floor, 0.74, bottom=floor, color=C_GAIN,
                edgecolor="none", zorder=2)
        axa.text(xpos, full + 0.022, f"{full:.2f}", ha="center", va="bottom", fontsize=7)
        if null > 0.05:
            axa.text(xpos, floor / 2, f"{null/full*100:.0f}%\nmemo-\nrizable", ha="center",
                     va="center", fontsize=6.0, color="white", fontweight="bold",
                     linespacing=1.25)
            drawn[f"a_memorizable_{ds}_{skey}"] = float(null / full * 100)
        else:
            # zero-valued segment gets a visible stub at the baseline (§6.1)
            axa.plot([xpos - 0.37, xpos + 0.37], [0, 0], lw=1.8, color=C_RANDOM,
                     solid_capstyle="butt", zorder=4)
            axa.text(xpos, 0.055, f"{null:+.3f}", ha="center", va="bottom", fontsize=6.4,
                     color="white", fontweight="bold")

    axa.axhline(0, lw=0.7, color="#777777", zorder=3)
    axa.set_xticks(slots)
    axa.set_xticklabels(labels, fontsize=6.5)
    axa.set_ylim(-0.07, 1.06)
    axa.set_ylabel("$R^2$ (5-fold CV)")
    axa.set_title("How much of the headline $R^2$ is memorization?", fontsize=8)
    for di, ds in enumerate("AB"):
        axa.text(di * 2.55 + 0.5, -0.175, f"Dataset {ds}", ha="center", va="top", fontsize=7,
                 transform=axa.get_xaxis_transform())
    axa.legend(handles=[Patch(facecolor=C_RANDOM, label="Featureless family-mean floor"),
                        Patch(facecolor=C_GAIN, label="Gain from features + model")],
               loc="upper center", bbox_to_anchor=(0.5, -0.235), ncols=1, fontsize=6.5,
               handlelength=1.1, borderpad=0.1, labelspacing=0.3)

    # ---------------- (b) screening skill: null ranker vs XGBoost ----------------------
    w = 0.35
    for xpos, (ds, skey) in zip(slots, dsets):
        null_p = nulo(csv, ds, "nulo_media_familia", skey).top100_mean * 100
        xgb_p = top100(csv, ds, "random" if skey == "aleatorio_KFold5" else "grouped").top100_mean * 100
        drawn[f"b_null_{ds}_{skey}"] = float(null_p)
        drawn[f"b_xgb_{ds}_{skey}"] = float(xgb_p)
        axb.bar(xpos - w / 2, null_p, w, color=C_RANDOM, edgecolor="none", zorder=2)
        axb.bar(xpos + w / 2, xgb_p, w, color=C_FAMILY, edgecolor="none", zorder=2)
        axb.text(xpos - w / 2, null_p + 1.6, f"{null_p:.0f}", ha="center", va="bottom",
                 fontsize=6.6, color="#595959")
        axb.text(xpos + w / 2, xgb_p + 1.6, f"{xgb_p:.0f}", ha="center", va="bottom",
                 fontsize=6.6, color=C_FAMILY)

    axb.axhline(10, ls="--", lw=0.8, color="#333333", zorder=1)
    axb.set_xlim(-0.62, slots[-1] + 0.62)
    axb.text((slots[1] + slots[2]) / 2, 14.5, "10% base rate", fontsize=6.4,
             ha="center", va="bottom", color="#333333")
    axb.set_xticks(slots)
    axb.set_xticklabels(labels, fontsize=6.5)
    axb.set_ylim(0, 112)
    axb.set_ylabel("Top-100 screening precision (%)")
    axb.set_title("Screening skill is real only under family-aware eval", fontsize=8)
    for di, ds in enumerate("AB"):
        axb.text(di * 2.55 + 0.5, -0.175, f"Dataset {ds}", ha="center", va="top", fontsize=7,
                 transform=axb.get_xaxis_transform())
    axb.legend(handles=[Patch(facecolor=C_RANDOM, label="Featureless family-mean ranker"),
                        Patch(facecolor=C_FAMILY, label="XGBoost")],
               loc="upper center", bbox_to_anchor=(0.5, -0.235), ncols=1, fontsize=6.5,
               handlelength=1.1, borderpad=0.1, labelspacing=0.3)

    for ax, L in ((axa, "a"), (axb, "b")):
        panel_letter(ax, L, dx=-0.135, dy=1.045)
    return fig, drawn


# ======================================================================================
# FIGURE 3 — Cross-dataset summary
#   (a) MAE dumbbells   -> tabla2_shuffle.csv (5-seed means, independent partitions)
#   (b) R2 dumbbells    -> tabla2_shuffle.csv
#   (c) top-100         -> top100_unificado.csv (per-fold points overlaid)
#   (d) NN distance     -> recomputed from data/
# ======================================================================================
def figure3(csv, rc) -> dict:
    drawn = {}
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.42,
                          left=0.145, right=0.965, top=0.855, bottom=0.095)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    fig.suptitle("Leakage-aware evaluation of $T_c$ prediction: the chemical-data-leakage "
                 "gap replicates on two independent datasets", fontsize=8.2, y=0.985)

    fig.text(0.5, 0.925, "(a, b) mean over 5 independent family partitions   ·   "
             "(c) points are the 5 folds of one partition",
             fontsize=6.2, color=C_META, ha="center", va="bottom")

    rows = [("A", "XGBoost"), ("A", "RandomForest"), ("B", "XGBoost"), ("B", "RandomForest")]
    ylab = [f"{d} · {m}" for d, m in rows]
    ypos = np.arange(len(rows))[::-1]

    # ---------------- (a) MAE dumbbells -----------------------------------------------
    for yy, (ds, mdl) in zip(ypos, rows):
        s = shuf(csv, ds, mdl)
        drawn[f"a_random_{ds}_{mdl}"] = float(s.MAE_random_mean)
        drawn[f"a_grouped_{ds}_{mdl}"] = float(s.MAE_grouped_mean)
        drawn[f"a_infl_{ds}_{mdl}"] = float(s.inflacion_pct_mean)
        axa.plot([s.MAE_random_mean, s.MAE_grouped_mean], [yy, yy], lw=2.4,
                 color="#C9C9C9", zorder=1, solid_capstyle="round")
        axa.scatter(s.MAE_random_mean, yy, s=42, color=C_RANDOM, zorder=3,
                    edgecolor="white", linewidth=0.6)
        axa.scatter(s.MAE_grouped_mean, yy, s=42, color=C_FAMILY, zorder=3,
                    edgecolor="white", linewidth=0.6)
        axa.text(s.MAE_grouped_mean + 0.16, yy, f"+{s.inflacion_pct_mean:.0f}%",
                 va="center", ha="left", fontsize=6.6, color=C_FAMILY, fontweight="bold")
    axa.set_yticks(ypos)
    axa.set_yticklabels(ylab, fontsize=6.6)
    axa.set_ylim(-0.55, len(rows) - 0.02)
    axa.set_xlim(3.3, 10.9)
    axa.set_xlabel("MAE (K)   ·   lower = better")
    axa.set_title("Error rises 52\u201361% on unseen chemistries", fontsize=8)
    s_top = shuf(csv, *rows[0])
    axa.annotate("random split", xy=(s_top.MAE_random_mean, ypos[0]),
                 xytext=(s_top.MAE_random_mean - 0.15, ypos[0] + 0.52), fontsize=6.4,
                 color="#5A5A5A", ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=C_META, shrinkA=1, shrinkB=3))
    axa.annotate("chemical-family split", xy=(s_top.MAE_grouped_mean, ypos[0]),
                 xytext=(s_top.MAE_grouped_mean + 0.15, ypos[0] + 0.52), fontsize=6.4,
                 color=C_FAMILY, ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=C_META, shrinkA=1, shrinkB=3))

    # ---------------- (b) R2 dumbbells ------------------------------------------------
    for yy, (ds, mdl) in zip(ypos, rows):
        s = shuf(csv, ds, mdl)
        drawn[f"b_random_{ds}_{mdl}"] = float(s.R2_random_mean)
        drawn[f"b_grouped_{ds}_{mdl}"] = float(s.R2_grouped_mean)
        drawn[f"b_dR2_{ds}_{mdl}"] = float(s.dR2_mean)
        axb.plot([s.R2_random_mean, s.R2_grouped_mean], [yy, yy], lw=2.4,
                 color="#C9C9C9", zorder=1, solid_capstyle="round")
        axb.scatter(s.R2_random_mean, yy, s=42, color=C_RANDOM, zorder=3,
                    edgecolor="white", linewidth=0.6)
        axb.scatter(s.R2_grouped_mean, yy, s=42, color=C_FAMILY, zorder=3,
                    edgecolor="white", linewidth=0.6)
        axb.text(s.R2_grouped_mean - 0.0035, yy, f"{s.dR2_mean:+.3f}".replace("-", "\u2212"),
                 va="center", ha="right", fontsize=6.6, color=C_FAMILY, fontweight="bold")
    axb.set_yticks(ypos)
    axb.set_yticklabels(ylab, fontsize=6.6)
    axb.set_ylim(-0.55, len(rows) - 0.02)
    axb.set_xlim(0.800, 0.945)
    axb.set_xticks([0.82, 0.86, 0.90, 0.94])
    axb.set_xlabel("$R^2$   ·   higher = better")
    axb.set_title("$R^2$ drops \u22480.08 on both datasets", fontsize=8)

    # ---------------- (c) top-100 screening precision ---------------------------------
    xs = [0, 1]
    w = 0.36
    for di, ds in enumerate("AB"):
        for j, (split, col, lab) in enumerate([("random", C_RANDOM, "Random split"),
                                               ("grouped", C_FAMILY, "Chemical-family split")]):
            r = top100(csv, ds, split)
            v, e = r.top100_mean * 100, r.top100_std * 100
            drawn[f"c_{split}_{ds}"] = float(v)
            axc.bar(di + (j - 0.5) * w, v, w, yerr=e, capsize=2.6, color=col,
                    edgecolor="none", error_kw=dict(lw=0.8, ecolor="#3A3A3A"),
                    label=lab if di == 0 else None, zorder=2)
            folds = [float(x) * 100 for x in ast.literal_eval(r.folds)]
            axc.scatter(np.full(len(folds), di + (j - 0.5) * w), folds, s=7,
                        facecolor="white", edgecolor="#333333", linewidth=0.55, zorder=5)
        rg = top100(csv, ds, "grouped")
        axc.text(di + 0.5 * w, rg.top100_mean * 100 + rg.top100_std * 100 + 2.6,
                 f"{rg.top100_mean*100:.0f}%", ha="center", va="bottom", fontsize=7,
                 color=C_FAMILY, fontweight="bold")
    axc.set_xticks(xs)
    axc.set_xticklabels(["Dataset A\n(Hamidieh)", "Dataset B\n(Stanev)"], fontsize=6.8)
    axc.set_ylim(0, 118)
    axc.set_yticks([0, 20, 40, 60, 80, 100])
    axc.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    axc.set_ylabel("Top-100 screening precision")
    axc.set_title("Still a reliable screening filter under family-aware CV", fontsize=8)

    axc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2, fontsize=6.3,
               handlelength=1.1, borderpad=0.1, columnspacing=1.1, handletextpad=0.4)

    # ---------------- (d) nearest-neighbour distance ----------------------------------
    for di, ds in enumerate("AB"):
        dr, dg = rc["nn"][ds]
        mr, mg = float(np.median(dr)), float(np.median(dg))
        drawn[f"d_random_{ds}"], drawn[f"d_family_{ds}"] = mr, mg
        drawn[f"d_ratio_{ds}"] = mg / mr
        for j, (vals, col) in enumerate([(dr, C_RANDOM), (dg, C_FAMILY)]):
            bp = axd.boxplot([vals], positions=[di + (j - 0.5) * 0.36], widths=0.30,
                             showfliers=False, patch_artist=True, whis=(5, 95))
            bp["boxes"][0].set(facecolor=col, edgecolor="#3A3A3A", linewidth=0.7)
            for el in ("whiskers", "caps"):
                for a in bp[el]:
                    a.set(color="#3A3A3A", linewidth=0.7)
            bp["medians"][0].set(color="black", linewidth=1.3)
        axd.text(di, 3.28, f"{mg/mr:.0f}\u00d7 farther" if mg / mr >= 10
                 else f"{mg/mr:.1f}\u00d7 farther",
                 ha="center", va="bottom", fontsize=6.8, color=C_FAMILY, fontweight="bold")
    axd.set_xticks(xs)
    axd.set_xticklabels(["Dataset A\n(Hamidieh)", "Dataset B\n(Stanev)"], fontsize=6.8)
    axd.set_xlim(-0.5, 1.5)
    axd.set_ylim(-0.12, 4.45)
    axd.set_ylabel("Distance to nearest training\nneighbour (std. feat. space)")
    axd.set_title("Why: random splits test on near-duplicate 'twins'", fontsize=8)
    axd.legend(handles=[Patch(facecolor=C_RANDOM, edgecolor="#3A3A3A", lw=0.7,
                              label="Random split"),
                        Patch(facecolor=C_FAMILY, edgecolor="#3A3A3A", lw=0.7,
                              label="Chemical-family split")],
               loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2, fontsize=6.3,
               handlelength=1.1, borderpad=0.1, columnspacing=1.1, handletextpad=0.4)
    axd.text(0.015, 0.985, "box: IQR, whiskers 5th\u201395th pct", transform=axd.transAxes,
             fontsize=6, color=C_META, va="top", ha="left")

    for ax, L in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
        panel_letter(ax, L, dx=-0.34 if ax in (axa, axb) else -0.215, dy=1.06)
    return fig, drawn


# ======================================================================================
# FIGURE 4 (NEW in v2) — What chemical-family identity accounts for
#   (a) variance decomposition: the in-sample ORACLE eta2 (hatched, non-CV) shown beside
#       three cross-validated quantities — full features, element-presence null,
#       family-mean null
#   (b) the design effect: nominal N vs effective N vs the number of families k
#
# CRITICAL (figure-style §1.2): eta2 is computed on ALL the data with the true per-family
# mean of Tc. It is an IN-SAMPLE ORACLE, not a cross-validated score, so it is NOT a
# visual peer of the CV bars: it is drawn hatched, in its own axis band, and named as
# oracle in the tick label and in the legend.
#
# NO RATIO MAY BE FORMED BETWEEN THE ORACLE BAR AND ANY CV BAR. Dividing the CV R2 (0.855)
# by the oracle eta2 (0.877) to claim the features "recover 97% of the ceiling" mixes an
# out-of-sample score with an in-sample one; the two are incommensurable and
# efecto_vs_ruido.md rules that comparison out explicitly. The v1 draft of this figure did
# it in the title and in a panel annotation; both were removed. The only differences drawn
# here are CV-minus-CV (full features vs element presence, both GroupKFold(5)).
# ======================================================================================
def figure4(csv, rc) -> dict:
    drawn = {}
    fig = plt.figure(figsize=(7.4, 4.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.46, 1.0], wspace=0.30,
                          left=0.095, right=0.985, top=0.79, bottom=0.295)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # Title states two magnitudes, each in its own frame of reference, and forms no ratio
    # between them. The eta2 percentage is the oracle quantity expressed as a share of total
    # variance (its own definition), NOT a fraction of anything cross-validated.
    eta_pct = [design(csv, ds).eta2 * 100 for ds in "AB"]
    eta_txt = (f"~{round(np.mean(eta_pct)):.0f}%" if round(eta_pct[0]) == round(eta_pct[1])
               else f"{min(eta_pct):.0f}\u2013{max(eta_pct):.0f}%")
    fig.suptitle(f"Family identity alone explains {eta_txt} of $T_c$ variance in-sample; "
                 f"effective $N$ falls to the number of families", fontsize=8.5, y=0.985)

    # ---------------- (a) variance decomposition --------------------------------------
    bars = [
        ("oracle", "Family identity\n(oracle $\\eta^2$)", C_ORACLE, "///"),
        ("full", "Paper\nfeatures", C_GAIN, None),
        ("presence", "Element\npresence only", C_PRESENCE, None),
        ("fammean", "Family-mean\nnull", C_RANDOM, None),
    ]
    w = 0.36
    for bi, (kind, lab, col, hatch) in enumerate(bars):
        for di, ds in enumerate("AB"):
            if kind == "oracle":
                v = design(csv, ds).eta2
            elif kind == "full":
                v = nulo(csv, ds, "features_completas_paper", "GroupKFold5_familias").R2
            elif kind == "presence":
                v = nulo(csv, ds, "presencia_elementos", "GroupKFold5_familias").R2
            else:
                v = nulo(csv, ds, "nulo_media_familia", "GroupKFold5_familias").R2
            drawn[f"a_{kind}_{ds}"] = float(v)
            xx = bi + (di - 0.5) * w
            # §4.3 hierarchical categories: the representation picks the hue, the dataset
            # samples within it (A lighter, B darker) -- one hue family per comparator.
            face = _shade(col, 0.42 if di == 0 else 0.0)
            axa.bar(xx, v, w, color=face, edgecolor="white" if hatch else "none",
                    hatch=hatch, linewidth=0.8 if hatch else 0, zorder=2)
            if v > 0.05:
                axa.text(xx, v + 0.022, f"{v:.3f}", ha="center", va="bottom", fontsize=6.4)
            else:
                # zero-height segment gets a visible stub at the baseline (§6.1); the two
                # near-zero values are named once together to keep the labels legible
                axa.plot([xx - w / 2, xx + w / 2], [0, 0], lw=1.6, color=col,
                         solid_capstyle="butt", zorder=4)

    fm_a = drawn["a_fammean_A"]
    fm_b = drawn["a_fammean_B"]
    axa.text(len(bars) - 1.02, -0.045,
             f"A {fm_a:+.3f}\nB {fm_b:+.3f}".replace("-", "\u2212"),
             ha="center", va="top", fontsize=6.3, color="#5A5A5A", linespacing=1.35)

    # The one legitimate difference in this panel: full features minus element presence.
    # Both arms are grouped-CV R2 on the same folds, so subtracting them is meaningful --
    # unlike any comparison against the in-sample oracle bar.
    for ds in "AB":
        drawn[f"a_gain_over_presence_{ds}"] = float(drawn[f"a_full_{ds}"]
                                                    - drawn[f"a_presence_{ds}"])

    axa.axhline(0, lw=0.7, color="#777777", zorder=3)
    # nested category axis: minor ticks name the dataset per bar, major ticks the comparator
    axa.set_xticks([bi + (di - 0.5) * w for bi in range(len(bars)) for di in (0, 1)],
                   minor=True)
    axa.set_xticklabels(["A", "B"] * len(bars), minor=True, fontsize=6.2)
    axa.tick_params(axis="x", which="minor", length=0, pad=1.5)
    axa.set_xticks(range(len(bars)))
    axa.set_xticklabels([b[1] for b in bars], fontsize=6.4)
    axa.tick_params(axis="x", which="major", length=0, pad=11)
    axa.set_xlim(-0.62, len(bars) - 0.42)
    axa.set_ylim(-0.20, 1.12)
    axa.set_ylabel("Variance in $T_c$ explained")
    axa.set_title("Hatched = in-sample oracle; solid = 5-fold grouped CV", fontsize=8,
                  pad=22)
    axa.text(0.905, 0.60, "higher\n= better", transform=axa.transAxes, ha="center",
             va="center", fontsize=6, color=C_META, linespacing=1.4)
    axa.legend(handles=[
        Patch(facecolor=_shade(C_ORACLE, 0.42), hatch="///", edgecolor="white",
              label="hatched = ORACLE $\\eta^2$ on all data (in-sample); "
                    "not comparable to a CV score"),
        Patch(facecolor="white", edgecolor="none",
              label="A = Dataset A ($N$=21,263, $k$=3,365)   ·   "
                    "B = Dataset B ($N$=12,440, $k$=3,063)")],
        loc="upper center", bbox_to_anchor=(0.47, -0.235), ncols=1, fontsize=6.2,
        handlelength=1.1, borderpad=0.1, labelspacing=0.35)

    # Bracket the only like-for-like difference: features vs element presence, both grouped
    # CV on the same folds. One bracket PER DATASET, anchored on that dataset's own two
    # bars, so it is unambiguous which pair each value belongs to.
    for di, (ds, y_br) in enumerate([("A", 0.928), ("B", 1.010)]):
        x0 = 1 + (di - 0.5) * w
        x1 = 2 + (di - 0.5) * w
        axa.plot([x0, x0, x1, x1], [y_br - 0.026, y_br, y_br, y_br - 0.026],
                 lw=0.6, color="#5A5A5A", clip_on=False, zorder=5)
        axa.text((x0 + x1) / 2, y_br + 0.010,
                 f"{ds}: +{drawn[f'a_gain_over_presence_{ds}']:.3f}",
                 ha="center", va="bottom", fontsize=6.2, color=C_TEXT, zorder=5)
    axa.text(0.5, 1.045, "features \u2212 element presence, both grouped CV",
             transform=axa.transAxes, ha="center", va="bottom", fontsize=6.2, color=C_META)

    # ---------------- (b) the design effect: N is not the sample size ------------------
    for di, ds in enumerate("AB"):
        d = design(csv, ds)
        drawn[f"b_N_{ds}"] = int(d.N)
        drawn[f"b_Neff_{ds}"] = float(d.N_eff_searle)
        drawn[f"b_k_{ds}"] = int(d.k)
        drawn[f"b_deff_{ds}"] = float(d.deff_searle)
        drawn[f"b_ICC_{ds}"] = float(d.ICC)
        base = di * 1.0
        for j, (val, col, hatch) in enumerate([
                (d.N, C_RANDOM, None), (d.N_eff_searle, C_GAIN, None), (d.k, C_FAMILY, "\\\\\\")]):
            xx = base + (j - 1) * 0.27
            axb.bar(xx, val, 0.25, color=col, edgecolor="white" if hatch else "none",
                    hatch=hatch, linewidth=0.8 if hatch else 0, zorder=2)
            txt = f"{val/1000:.1f}k" if val >= 1000 else f"{val:.0f}"
            axb.text(xx, val * 1.05, txt, ha="center", va="bottom", fontsize=6.4,
                     color=C_TEXT)
        # the design effect IS the drop from nominal to effective N: draw it as that drop
        axb.annotate("", xy=(base, d.N_eff_searle * 1.48), xytext=(base, d.N * 0.985),
                     arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#4A4A4A",
                                     shrinkA=0, shrinkB=0, mutation_scale=7))
        axb.text(base + 0.045, (d.N + d.N_eff_searle * 1.48) / 2,
                 f"\u00f7 {d.deff_searle:.1f}\ndesign\neffect", ha="left", va="center",
                 fontsize=6.2, color=C_TEXT, linespacing=1.35)

    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Dataset A", "Dataset B"], fontsize=7)
    axb.set_xlim(-0.55, 1.55)
    axb.set_ylim(0, 25800)
    axb.set_yticks([0, 5000, 10000, 15000, 20000])
    axb.set_yticklabels(["0", "5k", "10k", "15k", "20k"])
    axb.set_ylabel("Number of observations")
    axb.text(0.985, 0.975, f"ICC(1) = {design(csv, 'A').ICC:.2f} (A), "
             f"{design(csv, 'B').ICC:.2f} (B)", transform=axb.transAxes, ha="right",
             va="top", fontsize=6, color=C_META)
    axb.set_title("Effective $N$ \u2248 number of families", fontsize=8)
    axb.legend(handles=[Patch(facecolor=C_RANDOM, label="Nominal $N$ (materials)"),
                        Patch(facecolor=C_GAIN, label="Effective $N$ ($N/$design effect)"),
                        Patch(facecolor=C_FAMILY, hatch="\\\\\\", edgecolor="white",
                              label="$k$ (chemical families)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=1, fontsize=6.2,
               handlelength=1.1, borderpad=0.1, labelspacing=0.3)

    for ax, L in ((axa, "a"), (axb, "b")):
        panel_letter(ax, L, dx=-0.115 if ax is axa else -0.20, dy=1.05)
    return fig, drawn


# ======================================================================================
# FIDELITY ASSERTS
# Re-open every canonical CSV and check each DRAWN value against its source. This is the
# guard that keeps the figures and the measurements from silently drifting apart: if a CSV
# is re-measured and a figure is not regenerated (or vice versa), this fails.
# ======================================================================================
def verify(csv, rc, D1, D2, D3, D4, tol=5e-9):
    """Returns (n_checks, failures). Every check names the CSV column it came from."""
    checks = []   # (label, drawn, expected)

    def chk(label, got, want):
        checks.append((label, float(got), float(want)))

    # ---- Figure 1 -------------------------------------------------------------------
    dA = design(csv, "A")
    chk("F1a k (families) vs design_effect.csv:k", D1["a_k"], dA.k)
    chk("F1a largest family vs design_effect.csv:max_family_size", D1["a_max"], dA.max_family_size)
    for i, m in enumerate(["RandomForest", "XGBoost"]):
        s = shuf(csv, "A", m)
        chk(f"F1b MAE random {m} vs tabla2_shuffle.csv", D1["b_random"][i], s.MAE_random_mean)
        chk(f"F1b MAE grouped {m} vs tabla2_shuffle.csv", D1["b_grouped"][i], s.MAE_grouped_mean)
        chk(f"F1b inflation {m} vs tabla2_shuffle.csv", D1["b_inflation"][i], s.inflacion_pct_mean)
    rowA = nulo(csv, "A", "features_completas_paper", "GroupKFold5_familias")
    chk("F1c parity MAE vs nulo_presencia.csv:MAE", D1["c_mae"], rowA.MAE)
    chk("F1c parity R2 vs nulo_presencia.csv:R2", D1["c_r2"], rowA.R2)
    chk("F1c parity n vs nulo_presencia.csv:N", D1["c_n"], rowA.N)
    chk("F1d median random == F3d median random (A)", D1["d_median_random"], D3["d_random_A"])
    chk("F1d median family == F3d median family (A)", D1["d_median_family"], D3["d_family_A"])

    # ---- Figure 2 -------------------------------------------------------------------
    for ds in "AB":
        for skey in ("aleatorio_KFold5", "GroupKFold5_familias"):
            chk(f"F2a full R2 {ds}/{skey} vs nulo_presencia.csv",
                D2[f"a_full_{ds}_{skey}"],
                nulo(csv, ds, "features_completas_paper", skey).R2)
            chk(f"F2a family-mean null R2 {ds}/{skey} vs nulo_presencia.csv",
                D2[f"a_null_{ds}_{skey}"], nulo(csv, ds, "nulo_media_familia", skey).R2)
            chk(f"F2b null top-100 {ds}/{skey} vs nulo_presencia.csv",
                D2[f"b_null_{ds}_{skey}"],
                nulo(csv, ds, "nulo_media_familia", skey).top100_mean * 100)
            split = "random" if skey == "aleatorio_KFold5" else "grouped"
            chk(f"F2b XGBoost top-100 {ds}/{split} vs top100_unificado.csv",
                D2[f"b_xgb_{ds}_{skey}"], top100(csv, ds, split).top100_mean * 100)

    # ---- Figure 3 -------------------------------------------------------------------
    for ds in "AB":
        for mdl in ("XGBoost", "RandomForest"):
            s = shuf(csv, ds, mdl)
            chk(f"F3a MAE random {ds}/{mdl} vs tabla2_shuffle.csv",
                D3[f"a_random_{ds}_{mdl}"], s.MAE_random_mean)
            chk(f"F3a MAE grouped {ds}/{mdl} vs tabla2_shuffle.csv",
                D3[f"a_grouped_{ds}_{mdl}"], s.MAE_grouped_mean)
            chk(f"F3a inflation {ds}/{mdl} vs tabla2_shuffle.csv",
                D3[f"a_infl_{ds}_{mdl}"], s.inflacion_pct_mean)
            chk(f"F3b R2 random {ds}/{mdl} vs tabla2_shuffle.csv",
                D3[f"b_random_{ds}_{mdl}"], s.R2_random_mean)
            chk(f"F3b R2 grouped {ds}/{mdl} vs tabla2_shuffle.csv",
                D3[f"b_grouped_{ds}_{mdl}"], s.R2_grouped_mean)
            chk(f"F3b dR2 {ds}/{mdl} vs tabla2_shuffle.csv", D3[f"b_dR2_{ds}_{mdl}"], s.dR2_mean)
        for split in ("random", "grouped"):
            chk(f"F3c top-100 {ds}/{split} vs top100_unificado.csv",
                D3[f"c_{split}_{ds}"], top100(csv, ds, split).top100_mean * 100)

    # ---- Figure 4 -------------------------------------------------------------------
    for ds in "AB":
        d = design(csv, ds)
        chk(f"F4a oracle eta2 {ds} vs design_effect.csv:eta2", D4[f"a_oracle_{ds}"], d.eta2)
        chk(f"F4a full-feature CV R2 {ds} vs nulo_presencia.csv", D4[f"a_full_{ds}"],
            nulo(csv, ds, "features_completas_paper", "GroupKFold5_familias").R2)
        chk(f"F4a element-presence CV R2 {ds} vs nulo_presencia.csv", D4[f"a_presence_{ds}"],
            nulo(csv, ds, "presencia_elementos", "GroupKFold5_familias").R2)
        chk(f"F4a family-mean CV R2 {ds} vs nulo_presencia.csv", D4[f"a_fammean_{ds}"],
            nulo(csv, ds, "nulo_media_familia", "GroupKFold5_familias").R2)
        chk(f"F4b nominal N {ds} vs design_effect.csv:N", D4[f"b_N_{ds}"], d.N)
        chk(f"F4b effective N {ds} vs design_effect.csv:N_eff_searle",
            D4[f"b_Neff_{ds}"], d.N_eff_searle)
        chk(f"F4b k {ds} vs design_effect.csv:k", D4[f"b_k_{ds}"], d.k)
        chk(f"F4b design effect {ds} vs design_effect.csv:deff_searle",
            D4[f"b_deff_{ds}"], d.deff_searle)
        chk(f"F4b ICC {ds} vs design_effect.csv:ICC", D4[f"b_ICC_{ds}"], d.ICC)
        # the ONE legitimate difference in panel (a): both arms are grouped-CV R2
        chk(f"F4a features\u2212presence gain {ds} (CV\u2212CV) vs nulo_presencia.csv",
            D4[f"a_gain_over_presence_{ds}"],
            nulo(csv, ds, "features_completas_paper", "GroupKFold5_familias").R2
            - nulo(csv, ds, "presencia_elementos", "GroupKFold5_familias").R2)
        # The CV bars must be ordered full > presence > family-mean for the panel's reading
        # to hold. NOTE: this deliberately does NOT constrain the oracle bar against them --
        # the oracle is in-sample and is not a competitor to any CV bar (§1.2).
        assert D4[f"a_full_{ds}"] > D4[f"a_presence_{ds}"] > D4[f"a_fammean_{ds}"], \
            f"CV ordering violated for Dataset {ds}"

    # No oracle-to-CV ratio may be drawn or stored: that is the comparison
    # efecto_vs_ruido.md rules out, and it was removed from this figure after review.
    forbidden = [k for k in D4 if "pct_of_ceiling" in k or "frac_of_oracle" in k]
    assert not forbidden, (f"figure 4 must not form an oracle/CV ratio; found {forbidden}. "
                           "eta2 is in-sample and R2 is out-of-sample.")

    # ---- cross-figure consistency (§1.7 one number per claim) -----------------------
    chk("F1b vs F3a: A/XGBoost grouped MAE identical",
        D1["b_grouped"][1], D3["a_grouped_A_XGBoost"])
    chk("F2b vs F3c: A grouped top-100 identical",
        D2["b_xgb_A_GroupKFold5_familias"], D3["c_grouped_A"])
    chk("F2b vs F3c: B grouped top-100 identical",
        D2["b_xgb_B_GroupKFold5_familias"], D3["c_grouped_B"])
    chk("F2a vs F4a: A full-feature grouped R2 identical",
        D2["a_full_A_GroupKFold5_familias"], D4["a_full_A"])

    # ---- Fig 1d / 3d nearest-neighbour medians vs results_v2/nn_distance.csv (added 2026-09-02
    # after the 8.9x / 11.3x Dataset-B discrepancy; tolerance 1e-5 because the figure may use
    # float64 features while run_diagnostics_v2.py uses the float32 npz).
    nn_csv = pd.read_csv(HERE / "nn_distance.csv").set_index("dataset")
    nn_checks = []
    for ds in "AB":
        for arm, col in (("random", "nn_median_random"), ("family", "nn_median_grouped")):
            got, want = float(D3[f"d_{arm}_{ds}"]), float(nn_csv.loc[ds, col])
            nn_checks.append((f"F3d median {arm} {ds} vs nn_distance.csv:{col}", got, want))

    fails = [(lab, g, w) for lab, g, w in checks if abs(g - w) > tol] + \
            [(lab, g, w) for lab, g, w in nn_checks if abs(g - w) > 1e-5]
    checks = checks + nn_checks
    print(f"\n[verify] {len(checks)} fidelity checks against the canonical CSVs: "
          f"{len(checks) - len(fails)} passed, {len(fails)} failed")
    for lab, g, w in fails:
        print(f"   FAIL {lab}: drawn={g!r} csv={w!r} (delta={g - w:.3e})")
    assert not fails, f"{len(fails)} drawn values disagree with the canonical CSVs"
    return len(checks), fails


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-verify", action="store_true", help="skip the fidelity asserts")
    ap.add_argument("--submit", type=Path, default=None,
                    help="additionally copy the PNGs to this folder (e.g. the manuscript directory)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve paths and check that every input exists; draw nothing")
    args = ap.parse_args()
    global SUBMIT
    if args.submit is not None:
        SUBMIT = args.submit.resolve()
    inputs = [HERE / f for f in ("top100_unificado.csv", "nulo_presencia.csv", "design_effect.csv",
                                 "tabla2_shuffle.csv", "tabla2_runs_por_semilla.csv")]
    inputs += [DATA / f for f in ("train.csv", "unique_m.csv", "datasetB_featurized.npz")]
    inputs += [CODE / f for f in ("tc_pipeline.py", "datasetB_pipeline.py")]
    missing = [p for p in inputs if not p.exists()]
    if args.dry_run:
        print(f"ROOT={ROOT}\nDATA={DATA}\nCODE={CODE}\nOUT={OUT}\nFIG={FIG}\nSUBMIT={SUBMIT}")
        for p in inputs:
            print(f"  {'ok ' if p.exists() else 'MISSING'} {p}")
        return 1 if missing else 0
    assert not missing, f"missing inputs: {missing}"

    apply_figure_style(frame="open", sizes=(8, 7, 6))
    mpl.rcParams["savefig.dpi"] = DPI

    csv = load_csvs()
    rc = recompute()

    figs = {}
    fig1, D1 = figure1(csv, rc); figs["figure1.png"] = fig1
    fig2, D2 = figure2(csv, rc); figs["figure2.png"] = fig2
    fig3, D3 = figure3(csv, rc); figs["figure3.png"] = fig3
    fig4, D4 = figure4(csv, rc); figs["figure4.png"] = fig4

    findings = {}
    for name, fig in figs.items():
        findings[name] = check_overlaps(fig, name)
        save(fig, name)
    # figure 4 also keeps its descriptive filename, which the manuscript references
    shutil.copyfile(FIG / "figure4.png", FIG / "figure4_ceiling.png")
    shutil.copyfile(FIG / "figure4.png", HERE / "figure4_ceiling.png")
    if SUBMIT != FIG:
        shutil.copyfile(FIG / "figure4.png", SUBMIT / "figure4_ceiling.png")
    print(f"[save] {FIG / 'figure4_ceiling.png'}  (+ copy in results_v2/)")

    n_overlap = sum(len(v) for v in findings.values())
    print(f"\n[overlap] total findings across 4 figures: {n_overlap}")
    assert n_overlap == 0, f"geometric check found {n_overlap} overlapping/clipped labels"

    if not args.no_verify:
        n, fails = verify(csv, rc, D1, D2, D3, D4)
        print(f"[verify] OK — all {n} drawn values match their CSV of provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
