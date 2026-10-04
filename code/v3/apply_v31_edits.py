"""Apply the v3.1 amendments (tutor review + data-acquisition experiment) to paper1_manuscript.tex.

Every number inserted is written as a placeholder and rendered from results_v3/v31_key.json:
    {{k:NAME|FMT}}        value of NAME, formatted with FMT
    {{k:NAME|FMT|SCALE}}  value * SCALE (e.g. -1 to quote a reduction as a positive number)
    {{c:VALUE|FMT}}       declared constant (design parameter); recorded with a reason
For each placeholder a manifest entry with a unique prose anchor is generated and written to
code/v3/manuscript_numbers_v31.json, which verify_manuscript.py loads in addition to the main
manifest. Run once on the v3 text; it refuses to run twice (asserts every old string is present).
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper1_manuscript.tex"
KEY = json.load(open(ROOT.parent / "results_v3" / "v31_key.json"))
CONST_REASON = "design parameter of the v3.1 analyses (code/v3/data_acquisition.py, code/v3/tutor_analyses.py)"
PH = re.compile(r"\{\{([kc]):([^|}]+)\|([^|}]+)(?:\|([^}]+))?\}\}")
pending = []      # (marker, entry)


def render(s):
    def sub(m):
        kind, name, fmt, scale = m.group(1), m.group(2), m.group(3), m.group(4)
        if kind == "k":
            v = KEY[name] * (float(scale) if scale else 1.0)
            e = {"source": "results_v3/v31_key.json", "field": name, "fmt": fmt}
            if scale: e["scale"] = float(scale)
        else:
            v = float(name); e = {"const": v, "reason": CONST_REASON, "fmt": fmt}
        txt = format(v, fmt)
        if txt in ("-0.00", "-0.0"): txt = txt[1:]
        marker = f"\x00{len(pending)}\x00"
        pending.append((marker, txt, e))
        return marker
    return PH.sub(sub, s)


def ws(s):
    return r"\s+".join(re.escape(t) for t in s.split())


def rep(tex, old, new, count=1):
    n = len(re.findall(ws(old), tex))
    assert n == count, f"{n} matches (expected {count}) for: {old[:70]}"
    return re.sub(ws(old), lambda _m: render(new), tex)


def insert_after(tex, anchor, new):
    m = re.search(ws(anchor), tex); assert m, anchor[:60]
    return tex[:m.end()] + render(new) + tex[m.end():]


def insert_before(tex, anchor, new):
    m = re.search(ws(anchor), tex); assert m, anchor[:60]
    return tex[:m.start()] + render(new) + tex[m.start():]


def main():
    tex = TEX.read_text()
    assert "sec:acquisition" not in tex, "v3.1 edits already applied"

    # ---- comment 16: SuperCon does not distinguish measured non-superconductors -----------------
    tex = rep(tex, "once measured non-superconductors are admitted to the candidate pool)",
              "once entries recorded as non-superconducting are admitted to the candidate pool)")
    tex = rep(tex, "measured non-superconductors of the Stanev list", "entries recorded as non-superconducting in the Stanev list", count=2)
    tex = rep(tex, "measured non-superconductors are admitted to the pool.",
              "entries recorded as non-superconducting are admitted to the pool; the selection threshold "
              "$t_f$ ranges over \\SIrange{{{k:thr_A_min|.1f}}}{{{k:thr_A_max|.1f}}}{\\kelvin} (A) and "
              "\\SIrange{{{k:thr_B_min|.1f}}}{{{k:thr_B_max|.1f}}}{\\kelvin} (B) across folds.")
    tex = rep(tex, "non-superconductors expected among", "such entries expected among")
    tex = insert_after(tex, "such entries expected among the \\num{100} picks.",
                       " SuperCon does not distinguish a measured absence of superconductivity from an unreported or "
                       "censored measurement, so these entries are recorded negatives, not confirmed non-superconductors.")
    # ---- abstract: temporal sentence (comments 1, 22) and acquisition result (comment 23) --------
    tex = rep(tex, "no retrospective protocol reproduces forward-in-time performance: random cross-validation on "
                   "pre-2010 data estimates $\\Rsq = \\num{0.91}$ where the realised value on later compositions is "
                   "\\num{0.33};",
              "no cross-validation protocol reproduces the performance realised on a retrospective temporal split: "
              "random cross-validation on pre-2010 data estimates MAE \\SI{{{k:t2010_L0_aleatorio_MAE|.2f}}}{\\kelvin} "
              "and $\\Rsq = \\num{0.91}$ where the realised values on later compositions are "
              "\\SI{{{k:t2010_temporal_real_MAE|.2f}}}{\\kelvin} and \\num{0.33};")
    tex = rep(tex, "within \\num{0.1} at more than one cut. Row-level partitions",
              "within \\num{0.1} at more than one cut. At a fixed budget of training records, spreading them over "
              "more element systems lowers the error on unseen systems: one record per system instead of whole "
              "systems lowers the per-system MAE by \\SI{{{k:acq_A_N500_cap1_minus_capall_MAE_macro_mean|.1f|-1}}}{\\kelvin} "
              "(A) and \\SI{{{k:acq_B_N500_cap1_minus_capall_MAE_macro_mean|.1f|-1}}}{\\kelvin} (B) at "
              "\\num{{{c:500|.0f}}} records, without improving top-\\num{{{c:100|.0f}}} selection. Row-level partitions")
    # ---- comment 2: "every test row" -> measured fractions --------------------------------------
    tex = rep(tex, "When such a dataset is split at random, near-identical variants of every test row sit in the training "
                   "set, and the metric measures interpolation within a series the model has already seen.",
              "When such a dataset is split at random, \\SI{{{k:twin_A_random_pct|.1f}}}{\\percent} of the Dataset~A "
              "test rows have a training row with an identical descriptor vector "
              "(\\SI{{{k:twin_B_random_pct|.1f}}}{\\percent} in Dataset~B), and \\SI{{{k:dlt01_A_random_pct|.0f}}}{\\percent} "
              "and \\SI{{{k:dlt01_B_random_pct|.0f}}}{\\percent} have one within \\num{{{c:0.1|.1f}}} standardised units "
              "(section~\\ref{sec:results-gap}); the metric then largely measures interpolation within series the model "
              "has already seen. That is the right measure when the target is a new member of a known system, and the "
              "wrong one when the target is a new system.")
    # ---- comment 6: provenance of the hyperparameters --------------------------------------------
    tex = rep(tex, "with hyperparameters fixed per dataset from prior grid searches and held constant across protocols.",
              "with hyperparameters fixed per dataset and held constant across protocols. They come from the author's "
              "earlier, unpublished conference manuscript on the same task, which selected them with "
              "\\texttt{GridSearchCV} by three-fold \\emph{random} cross-validation on a random 80/20 training split, "
              "scored by negative mean squared error; its search grids were not recorded, and its Dataset~B was a "
              "different curation of SuperCon. Because that search used row-level folds, "
              "section~\\ref{sec:results-tuning} re-selects the hyperparameters by nested search inside each protocol.")
    # ---- comment 13: nearest-neighbour wording ----------------------------------------------------
    tex = rep(tex, "To confirm that the protocols differ in the way the construction predicts, we measure",
              "To describe how far each protocol places test rows from the training data, we measure")
    tex = rep(tex, "and a grouped hold-out of the same size.",
              "and a grouped hold-out with the same nominal 80/20 fraction (the realised test sizes differ because "
              "systems are indivisible). Using the five-fold out-of-fold predictions, with the scaler refitted on each "
              "training fold, we also relate the absolute error to this distance, to the recorded $\\Tc$ and to the "
              "size of the test row's system.")
    tex = rep(tex, "The nearest-neighbour diagnostic confirms the mechanism",
              "The nearest-neighbour diagnostic is consistent with this reading")
    tex = rep(tex, "under a grouped hold-out of the same size it is", "under the grouped hold-out it is")
    # ---- comments 8, 13, 15: error vs distance, Tc range, system size; within-system ranking ------
    m = re.search(ws("For more than a third of its test set"), tex); assert m
    end = tex.find("\n\n", m.end())
    para = (
        "\n\nDistance alone does not order the error, because it is confounded with $\\Tc$. Under random splitting "
        "the MAE stays within \\SIrange{{{k:evd_A_random_min|.1f}}}{{{k:evd_A_random_max|.1f}}}{\\kelvin} across "
        "distance deciles on Dataset~A and \\SIrange{{{k:evd_B_random_min|.1f}}}{{{k:evd_B_random_max|.1f}}}{\\kelvin} "
        "on Dataset~B. Under grouped splitting it peaks at \\SI{{{k:evd_A_grouped_max|.1f}}}{\\kelvin} (A) and "
        "\\SI{{{k:evd_B_grouped_max|.1f}}}{\\kelvin} (B) at intermediate distances and falls to "
        "\\SI{{{k:evd_A_grouped_farthest|.1f}}}{\\kelvin} and \\SI{{{k:evd_B_grouped_farthest|.1f}}}{\\kelvin} in the "
        "farthest decile, whose rows are mostly low-$\\Tc$ materials: the mean $\\Tc$ of the nearest third of the "
        "grouped test rows of Dataset~A is \\SI{{{k:terc_meanTc_A_near|.1f}}}{\\kelvin}, that of the farthest third "
        "\\SI{{{k:terc_meanTc_A_far|.1f}}}{\\kelvin}. Among high-$\\Tc$ rows (77~K and above), where the screening "
        "question lives, the error does grow with distance: from \\SI{{{k:evdtc_A_77_near|.1f}}}{\\kelvin} to "
        "\\SI{{{k:evdtc_A_77_mid|.1f}}}{\\kelvin} to \\SI{{{k:evdtc_A_77_far|.1f}}}{\\kelvin} across distance "
        "terciles on A, and from \\SI{{{k:evdtc_B_77_near|.1f}}}{\\kelvin} to \\SI{{{k:evdtc_B_77_mid|.1f}}}{\\kelvin} "
        "to \\SI{{{k:evdtc_B_77_far|.1f}}}{\\kelvin} on B. By $\\Tc$ range the grouped MAE rises from "
        "\\SI{{{k:bd_A_Tc_range_lt10|.1f}}}{\\kelvin} below 10~K to \\SI{{{k:bd_A_Tc_range_ge77|.1f}}}{\\kelvin} at "
        "77~K and above on A (\\SI{{{k:bd_B_Tc_range_lt10|.1f}}}{\\kelvin} to \\SI{{{k:bd_B_Tc_range_ge77|.1f}}}{\\kelvin} "
        "on B), and by system size from \\SI{{{k:bd_A_system_size_1|.1f}}}{\\kelvin} for single-row systems to "
        "\\SI{{{k:bd_A_system_size_gt100|.1f}}}{\\kelvin} for systems of more than one hundred rows "
        "(\\SI{{{k:bd_B_system_size_1|.1f}}}{\\kelvin} to \\SI{{{k:bd_B_system_size_gt100|.1f}}}{\\kelvin} on B). The "
        "error of the grouped protocol is concentrated in the large, high-$\\Tc$ systems, which is why the "
        "per-system inflation is smaller than the pooled one (section~\\ref{sec:results-gap}). Within held-out systems "
        "the model still orders the members: among systems with at least five test rows, the median Spearman "
        "correlation between predicted and recorded $\\Tc$ is \\num{{{k:wrho_A_grouped_median|.2f}}} on A and "
        "\\num{{{k:wrho_B_grouped_median|.2f}}} on B under grouped splitting, against "
        "\\num{{{k:wrho_A_random_median|.2f}}} and \\num{{{k:wrho_B_random_median|.2f}}} under random splitting, and "
        "it is positive in \\SI{{{k:wrho_A_grouped_fracpos_pct|.0f}}}{\\percent} and "
        "\\SI{{{k:wrho_B_grouped_fracpos_pct|.0f}}}{\\percent} of the held-out systems. What holding out a system "
        "removes is the model's knowledge of its $\\Tc$ level, not its ability to order the system's variants.")
    tex = tex[:end] + render(para) + tex[end:]
    # ---- comment 19: minimal labelling test --------------------------------------------------------
    tex = rep(tex, "so a comparison of results between the two checkpoints includes partition noise;",
              "so a comparison of results between the two checkpoints includes partition noise. A minimal test "
              "isolates the cause: on the same rows, labels factorised before filtering (with gaps) and the same labels "
              "recoded to consecutive integers in the same order give identical folds "
              "(\\num{{{k:lab_rows_differ_gapped_recoded|.0f}}} rows differ), whereas factorising by first appearance "
              "after filtering, which yields the published labels, reassigns \\num{{{k:lab_rows_differ_gapped_first|.0f}}} "
              "rows. The partition depends on the order of the labels, not on their gaps;")
    # ---- comment 22: naming of the temporal test, and its overlap with the past ------------------
    tex = tex.replace("forward-in-time", "retrospective temporal")
    tex = rep(tex, "reported at or after $T$?",
              "reported at or after $T$? We call this a retrospective temporal split: it uses only reports already "
              "in the database, so it is not a prospective test.")
    tex = rep(tex, "and $\\Rsq = \\num{0.80}$. Random five-fold CV on the same pre-2010 data",
              "and $\\Rsq = \\num{0.80}$. Those re-measurements are scored separately: the principal test set contains "
              "no composition present in the training set, because the corpus holds one row per composition, dated by "
              "its first report. Random five-fold CV on the same pre-2010 data")
    tex = rep(tex, "whereas the random-CV and leave-clusters-out estimates move by less than",
              "whereas at that cut the random-CV and leave-clusters-out estimates move by less than")
    tex = insert_after(tex, "leave-clusters-out estimates move by less than \\SI{0.3}{\\kelvin} in MAE and \\num{0.02} in $\\Rsq$",
                       " and the host-grouped estimate by \\SI{{{k:dating_est_dMAE_L3_anfitrion_2010|.2f}}}{\\kelvin} in MAE")
    tex = rep(tex, "The evaluation that comes closest to a prospective test---training",
              "The retrospective temporal split---training")
    tex = rep(tex, "is anticipated by none of the retrospective protocols", "is anticipated by none of the cross-validation protocols")
    # ---- limitations: dating sensitivity measured at the four cuts ---------------------------------
    tex = rep(tex, "the sensitivity to those rows is large for $\\Rsq$ and small for MAE.",
              "removing those rows changes the realised MAE by \\SI{{{k:dating_dMAE_1995|+.2f}}}{\\kelvin}, "
              "\\SI{{{k:dating_dMAE_2005|+.2f}}}{\\kelvin}, \\SI{{{k:dating_dMAE_2010|+.2f}}}{\\kelvin} and "
              "\\SI{{{k:dating_dMAE_2015|+.2f}}}{\\kelvin} and the realised $\\Rsq$ by \\num{{{k:dating_dR2_1995|+.2f}}}, "
              "\\num{{{k:dating_dR2_2005|+.2f}}}, \\num{{{k:dating_dR2_2010|+.2f}}} and \\num{{{k:dating_dR2_2015|+.2f}}} at "
              "the 1995, 2005, 2010 and 2015 cuts, so the realised figures at 2005 and 2010 depend on dating, while the "
              "protocol closest to the realised value is the same in both scenarios at every cut and for both metrics "
              "(\\texttt{code/v3/temporal\\_dating\\_sensitivity.py}).")
    # ---- contributions: new item -----------------------------------------------------------------
    tex = rep(tex, "can be regenerated with one command. \\end{enumerate}",
              "can be regenerated with one command.\n\\item A fixed-budget acquisition experiment showing that, for "
              "transfer to unseen element systems, the number of distinct systems in the training set matters more "
              "than the number of records per system (section~\\ref{sec:acquisition}).\n\\end{enumerate}")
    # ---- comment 23: data-acquisition subsection + figure ------------------------------------------
    acq = r"""
\subsection{Spending a fixed budget: more systems or more variants}
\label{sec:acquisition}

The previous sections diagnose; this one asks what a group that wants better predictions on new
chemistries should measure. We hold out \SI{{{c:20|.0f}}}{\percent} of the element systems, with all their rows, as a
fixed external test (three independent draws; Dataset~A: \num{{{k:acq_A_n_test_systems|.0f}}} systems and
\numrange{{{k:acq_A_n_test_rows_by_draw_min|.0f}}}{{{k:acq_A_n_test_rows_by_draw_max|.0f}}} rows; Dataset~B:
\num{{{k:acq_B_n_test_systems|.0f}}} systems and
\numrange{{{k:acq_B_n_test_rows_by_draw_min|.0f}}}{{{k:acq_B_n_test_rows_by_draw_max|.0f}}} rows). From the remaining
systems we draw training sets of \num{{{c:500|.0f}}}, \num{{{c:1000|.0f}}} and \num{{{c:2000|.0f}}} records under a
per-system cap: the systems are visited in random order and at most $c$ records are taken from each, for
$c$ equal to one, two, four, eight, thirty-two and unlimited (whole systems), and, as a reference, records drawn
uniformly at random, which follow the natural sampling of the corpus. Each design is repeated ten times
per test draw with shared system and record orders, so the designs are paired; intervals are the
2.5th--97.5th percentiles over the thirty (draw, replicate) pairs and describe variation over training samples
and test draws, conditional on this corpus. The model is the XGBoost configuration of
section~\ref{sec:models}.

At every budget and in both datasets, spreading the budget over more systems lowers the error on the
unseen systems (figure~\ref{fig:acquisition}). One record per system instead of whole systems lowers the
per-system MAE by \SI{{{k:acq_A_N500_cap1_minus_capall_MAE_macro_mean|.2f|-1}}}{\kelvin}
\ci{{{k:acq_A_N500_cap1_minus_capall_MAE_macro_hi|.2f|-1}}}{{{k:acq_A_N500_cap1_minus_capall_MAE_macro_lo|.2f|-1}}} at
\num{{{c:500|.0f}}} records and by \SI{{{k:acq_A_N2000_cap1_minus_capall_MAE_macro_mean|.2f|-1}}}{\kelvin}
\ci{{{k:acq_A_N2000_cap1_minus_capall_MAE_macro_hi|.2f|-1}}}{{{k:acq_A_N2000_cap1_minus_capall_MAE_macro_lo|.2f|-1}}} at
\num{{{c:2000|.0f}}} on Dataset~A, and by \SI{{{k:acq_B_N500_cap1_minus_capall_MAE_macro_mean|.2f|-1}}}{\kelvin}
\ci{{{k:acq_B_N500_cap1_minus_capall_MAE_macro_hi|.2f|-1}}}{{{k:acq_B_N500_cap1_minus_capall_MAE_macro_lo|.2f|-1}}} and
\SI{{{k:acq_B_N2000_cap1_minus_capall_MAE_macro_mean|.2f|-1}}}{\kelvin}
\ci{{{k:acq_B_N2000_cap1_minus_capall_MAE_macro_hi|.2f|-1}}}{{{k:acq_B_N2000_cap1_minus_capall_MAE_macro_lo|.2f|-1}}} on
Dataset~B; the pooled MAE falls by \SI{{{k:acq_A_N500_cap1_minus_capall_MAE_mean|.2f|-1}}}{\kelvin} and
\SI{{{k:acq_A_N2000_cap1_minus_capall_MAE_mean|.2f|-1}}}{\kelvin} (A) and by
\SI{{{k:acq_B_N500_cap1_minus_capall_MAE_mean|.2f|-1}}}{\kelvin} and
\SI{{{k:acq_B_N2000_cap1_minus_capall_MAE_mean|.2f|-1}}}{\kelvin} (B), with every interval excluding zero.
Across the six capped designs and three budgets the error depends on the number of distinct systems far
more than on the number of records per system: in a regression on both, each doubling of the number of
systems lowers the per-system MAE by \SI{{{k:acq_A_MAE_macro_per_doubling_systems_K|.2f|-1}}}{\kelvin} (A) and
\SI{{{k:acq_B_MAE_macro_per_doubling_systems_K|.2f|-1}}}{\kelvin} (B), each doubling of the records per system by
\SI{{{k:acq_A_MAE_macro_per_doubling_rows_per_system_K|.2f|-1}}}{\kelvin} and
\SI{{{k:acq_B_MAE_macro_per_doubling_rows_per_system_K|.2f|-1}}}{\kelvin}. In budget terms, \num{{{c:500|.0f}}}
records spread one per system do as well as \num{{{c:2000|.0f}}} records in whole systems (difference
\SI{{{k:acq_A_cap1_N500_minus_capall_N2000_MAE_macro_mean|+.2f}}}{\kelvin}
\ci{{{k:acq_A_cap1_N500_minus_capall_N2000_MAE_macro_lo|.2f}}}{{{k:acq_A_cap1_N500_minus_capall_N2000_MAE_macro_hi|.2f}}}
on A and \SI{{{k:acq_B_cap1_N500_minus_capall_N2000_MAE_macro_mean|+.2f}}}{\kelvin}
\ci{{{k:acq_B_cap1_N500_minus_capall_N2000_MAE_macro_lo|.2f}}}{{{k:acq_B_cap1_N500_minus_capall_N2000_MAE_macro_hi|.2f}}}
on B). Records drawn at random, which concentrate on the large systems, fall between the two extremes.

Selection behaves differently. At a fixed threshold of 77~K, reached by
\SI{{{k:acq_A_prev77_pct|.1f}}}{\percent} (A) and \SI{{{k:acq_B_prev77_pct|.1f}}}{\percent} (B) of the external test rows,
top-\num{{{c:100|.0f}}} precision does not separate the designs: one record per system against whole systems gives
\num{{{k:acq_A_N500_cap1_minus_capall_P_at_100_77K_mean|+.1f|100}}} points
\ci{{{k:acq_A_N500_cap1_minus_capall_P_at_100_77K_lo|.1f|100}}}{{{k:acq_A_N500_cap1_minus_capall_P_at_100_77K_hi|.1f|100}}}
at \num{{{c:500|.0f}}} records on A and \num{{{k:acq_B_N2000_cap1_minus_capall_P_at_100_77K_mean|+.1f|100}}} points
\ci{{{k:acq_B_N2000_cap1_minus_capall_P_at_100_77K_lo|.1f|100}}}{{{k:acq_B_N2000_cap1_minus_capall_P_at_100_77K_hi|.1f|100}}}
at \num{{{c:2000|.0f}}} on B, and every such interval includes zero. Random records give the highest precision
(\SI{{{k:acq_A_random_rows_N2000_P77|.1f}}}{\percent} on A and \SI{{{k:acq_B_random_rows_N2000_P77|.1f}}}{\percent} on B at
\num{{{c:2000|.0f}}} records) because they reproduce the corpus' over-representation of high-$\Tc$ systems: their mean
training $\Tc$ is \SI{{{k:acq_A_random_rows_N500_meanTc|.1f}}}{\kelvin} on A against
\SI{{{k:acq_A_cap1_N500_meanTc|.1f}}}{\kelvin} for one-per-system sampling. Breadth buys accuracy across chemistries;
at these budgets it does not buy a better ranking of the highest-$\Tc$ candidates, which depends on how
many high-$\Tc$ systems the training set contains.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figure6}
\caption{\label{fig:acquisition}\textbf{At a fixed budget of training records, spreading them over more element
systems lowers the error on unseen systems; selection at 77~K does not separate the designs.} Rows:
Dataset~A (top) and Dataset~B (bottom). (a,\,d)~Per-system (macro) MAE on a fixed external test of held-out
systems against the training budget, for per-system caps from one record per system to whole systems and
for records drawn at random (dashed); bands: 2.5th--97.5th percentiles over thirty (test draw, replicate)
pairs for the two extreme designs. (b,\,e)~The same MAE against the number of distinct systems in the
training set; marker shape gives the budget. (c,\,f)~Top-100 precision at a fixed 77~K threshold. XGBoost
with the configuration of section~\ref{sec:models}; \texttt{code/v3/data\_acquisition.py}.}
\end{figure}
"""
    tex = insert_before(tex, "\\section{Discussion}", acq + "\n")
    # ---- discussion and conclusion -----------------------------------------------------------------
    tex = insert_before(tex, "\\subsection{Relation to prior work}",
        "The acquisition experiment turns the diagnosis into a rule a laboratory can use. If the aim is "
        "accuracy on chemistries not yet in the database, new element systems are worth more than further "
        "variants of known ones, by a margin that a fourfold larger budget spent on whole systems does not "
        "recover. If the aim is to rank the highest-$\\Tc$ candidates, breadth alone does not help at these "
        "budgets, and the composition of the training set by $\\Tc$ matters more.\n\n")
    tex = rep(tex, "treating its figure as a proxy with a known direction of error.",
              "treating its figure as a proxy with a known direction of error. For groups deciding what to measure, "
              "the acquisition experiment gives a concrete rule: at a fixed budget, new element systems lower the "
              "error on unseen chemistries more than new variants of known ones, although they do not by themselves "
              "improve the ranking of the highest-$\\Tc$ candidates.")
    # ---- prior work (Konno temporal separation, Li 2023, MatFold, LOCO-CV, Murdock, Kauwe) ---------
    tex = insert_before(tex, "What this paper adds is the quantification of the gap",
        "Konno \\etal~\\cite{konno2021} already separated SuperCon temporally, training only on data added "
        "before 2010 to evaluate a superconductor classifier on later materials, and argued that temporally "
        "separated splits should replace random ones; our temporal test differs in scoring every retrospective "
        "protocol against the realised performance rather than reporting a single split. Li "
        "\\etal~\\cite{li2023} showed that models trained on the 2018 Materials Project degrade on compounds "
        "added by 2021, and that adding a small, deliberately chosen fraction of the new data recovers much of the "
        "accuracy; our acquisition experiment asks the related question of how a fixed budget should be spread "
        "across element systems. MatFold~\\cite{witman2025} and kernelised leave-one-cluster-out "
        "validation~\\cite{durdy2022} provide general tools for the grouped and cluster splits used here. Murdock "
        "\\etal~\\cite{murdock2020} asked whether domain-knowledge featurizations are necessary and Kauwe "
        "\\etal~\\cite{kauwe2020} whether models can find materials with extraordinary property values; our "
        "representation ladder and selection metrics bear on the same questions, which we cite by their framing "
        "only, as we have not compared results quantitatively. ")
    bib = r"""
\bibitem{li2023}
Li K, DeCost B, Choudhary K, Greenwood M and Hattrick-Simpers J 2023 A critical examination of
robustness and generalizability of machine learning prediction of materials properties
\textit{npj Comput. Mater.} \textbf{9} 55
\url{https://doi.org/10.1038/s41524-023-01012-9}

\bibitem{witman2025}
Witman M D and Schindler P 2025 MatFold: systematic insights into materials discovery models'
performance through standardized cross-validation protocols \textit{Digit. Discov.} \textbf{4} 625--35
\url{https://doi.org/10.1039/d4dd00250d}

\bibitem{durdy2022}
Durdy S, Gaultois M W, Gusev V V, Bollegala D and Rosseinsky M J 2022 Random projections and
kernelised leave one cluster out cross validation: universal baselines and evaluation tools for
supervised machine learning of material properties \textit{Digit. Discov.} \textbf{1} 763--78
\url{https://doi.org/10.1039/d2dd00039c}

\bibitem{murdock2020}
Murdock R J, Kauwe S K, Wang A Y-T and Sparks T D 2020 Is domain knowledge necessary for machine
learning materials properties? \textit{Integr. Mater. Manuf. Innov.} \textbf{9} 221--7
\url{https://doi.org/10.1007/s40192-020-00179-z}

\bibitem{kauwe2020}
Kauwe S K, Graser J, Murdock R and Sparks T D 2020 Can machine learning find extraordinary
materials? \textit{Comput. Mater. Sci.} \textbf{174} 109498
\url{https://doi.org/10.1016/j.commatsci.2019.109498}
"""
    tex = insert_before(tex, "\\end{thebibliography}", bib.lstrip("\n") + "\n")
    # ---- acknowledgements and AI-use disclosure (IOP policy) ---------------------------------------
    tex = insert_before(tex, "\\section*{Data availability}",
        "\\section*{Acknowledgements}\n\nThe author thanks his thesis tutor for a detailed review of a previous "
        "version of this manuscript. \\textbf{[Author to complete: funding statement and competing interests.]}\n\n"
        "\\textit{Use of generative AI.} A large-language-model research assistant "
        "\\textbf{[author to state model and version]} was used to write and run analysis code, design and "
        "execute the experiments of sections~\\ref{sec:audit}--\\ref{sec:acquisition}, render the figures, "
        "verify bibliographic records and draft the text; an independent AI-assisted adversarial review was used "
        "to audit a previous version. All outputs were checked against the released result files by the "
        "per-claim verifier and by the author, who takes full responsibility for the content.\n\n")

    # ---- resolve markers and build manifest entries -------------------------------------------------
    entries = []
    for marker, txt, e in pending:
        pos = tex.find(marker); assert pos >= 0, marker
        entries.append((pos, marker, txt, e))
    for pos, marker, txt, e in entries:
        tex = tex.replace(marker, txt, 1) if False else tex
    # replace all markers now, then build anchors on the final text
    final = tex
    for marker, txt, e in pending:
        final = final.replace(marker, txt, 1)
    norm = lambda s: re.sub(r"\s+", " ", s)
    flat = norm(final)
    out = []
    for i, (marker, txt, e) in enumerate(pending):
        # locate this placeholder in a version where only it is a marker
        t = tex
        for j, (mk, tx2, _) in enumerate(pending):
            if j != i: t = t.replace(mk, tx2, 1)
        t = norm(t); p = t.find(marker)
        for back in (40, 60, 90, 130, 180):
            a0 = t.rfind(" ", 0, max(0, p - back)) + 1
            anchor = t[a0:p] + "<<v>>" + t[p + len(marker):p + len(marker) + 14]
            lit = anchor.replace("<<v>>", txt)
            if flat.count(lit) == 1:
                break
        else:
            raise AssertionError(f"no unique anchor for {e}")
        out.append({"id": f"v31_{i:03d}_" + (e.get("field") or f"const_{txt}"), "anchor": anchor, **e})
    TEX.write_text(final)
    json.dump({"_doc": "v3.1 additions (code/v3/apply_v31_edits.py); loaded by verify_manuscript.py with the main manifest",
               "entries": out}, open(ROOT / "code/v3/manuscript_numbers_v31.json", "w"), indent=1, ensure_ascii=False)
    print(f"edits applied; {len(out)} manifest entries")


if __name__ == "__main__":
    sys.exit(main())
