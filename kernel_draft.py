"""Helpers for the claim-discipline skill: measure before you write.

Each function exists because a claim in this project was written before it was
measured, and turned out to be wrong. See SKILL.md for the incidents.
"""

import numpy as np


def coverage_report(y_true, lo, hi, groups=None, hi_threshold=None, min_group=5):
    """Marginal AND conditional coverage of prediction intervals in one call.

    Reporting only marginal coverage is the failure mode this guards against:
    a method can hit its nominal rate overall while badly under-covering the
    subpopulation the work is actually about.

    Returns a dict with marginal coverage, coverage above `hi_threshold`
    (the decision-relevant region), and the per-group coverage distribution
    (median, 10th percentile, fraction of groups below 80%).
    """
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    inside = (y_true >= lo) & (y_true <= hi)

    out = {
        "n": int(len(y_true)),
        "coverage": float(inside.mean()),
        "width_mean": float(np.mean(hi - lo)),
        "width_median": float(np.median(hi - lo)),
    }

    if hi_threshold is not None:
        mask = y_true >= hi_threshold
        out["n_above_threshold"] = int(mask.sum())
        out["coverage_above_threshold"] = float(inside[mask].mean()) if mask.any() else float("nan")

    if groups is not None:
        groups = np.asarray(groups)
        per = np.asarray([inside[groups == g].mean() for g in np.unique(groups)
                          if (groups == g).sum() >= min_group], dtype=float)
        out["n_groups_evaluated"] = int(len(per))
        if len(per):
            out["group_coverage_median"] = float(np.median(per))
            out["group_coverage_p10"] = float(np.percentile(per, 10))
            out["frac_groups_under_80"] = float(np.mean(per < 0.80))
    return out


def effect_vs_seed_noise(baseline_by_seed, variant_by_seed, label="variant"):
    """Is a claimed improvement larger than the run-to-run dispersion?

    Pass the per-seed metric values for each arm (same metric, paired by seed
    where possible). An effect smaller than the seed standard deviation is not
    an effect you can write a sentence about.

    Returns effect size, seed dispersion, their ratio, and a verdict string.
    """
    base = np.asarray(baseline_by_seed, dtype=float)
    var = np.asarray(variant_by_seed, dtype=float)
    effect = float(var.mean() - base.mean())
    noise = float(np.sqrt(0.5 * (base.std(ddof=1) ** 2 + var.std(ddof=1) ** 2)))
    ratio = float(abs(effect) / noise) if noise > 0 else float("inf")

    if ratio < 1.0:
        verdict = "indistinguishable from seed noise - do not claim an effect"
    elif ratio < 2.0:
        verdict = "comparable to seed noise - report with the dispersion, hedge the claim"
    else:
        verdict = "exceeds seed noise - reportable"

    paired = None
    if len(base) == len(var):
        diff = var - base
        paired = {"mean_paired_diff": float(diff.mean()),
                  "sd_paired_diff": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
                  "n_seeds_favouring_variant": int((diff > 0).sum())}

    return {"label": label, "baseline_mean": float(base.mean()),
            "variant_mean": float(var.mean()), "effect": effect,
            "seed_sd": noise, "effect_over_noise": ratio,
            "verdict": verdict, "paired": paired}


def confound_scan(value, outcome, stratifier, n_strata=3, strata_labels=None):
    """Does an association survive conditioning, or is it a Simpson reversal?

    `value` is the candidate explanatory variable, `outcome` the thing you claim
    it explains (e.g. a 0/1 failure indicator), `stratifier` the variable you
    suspect is the real driver. Reports the marginal Spearman correlation and
    the within-stratum correlations after binning on `stratifier`.

    A sign flip between marginal and conditional is a Simpson reversal: the
    marginal number is an artefact of how the strata are mixed, and any
    mechanism sentence built on it is wrong.
    """
    from scipy.stats import spearmanr

    value = np.asarray(value, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    stratifier = np.asarray(stratifier, dtype=float)

    rho_marg, p_marg = spearmanr(value, outcome)
    edges = np.quantile(stratifier, np.linspace(0, 1, n_strata + 1))
    bins = np.clip(np.digitize(stratifier, edges[1:-1]), 0, n_strata - 1)

    within = []
    for k in range(n_strata):
        mask = bins == k
        if mask.sum() < 20:
            continue
        rho_k, p_k = spearmanr(value[mask], outcome[mask])
        within.append({
            "stratum": strata_labels[k] if strata_labels is not None else k,
            "n": int(mask.sum()),
            "stratifier_median": float(np.median(stratifier[mask])),
            "value_median": float(np.median(value[mask])),
            "outcome_mean": float(outcome[mask].mean()),
            "spearman": float(rho_k),
            "p_value": float(p_k),
        })

    rhos = [w["spearman"] for w in within]
    reversal = bool(rhos) and all(np.sign(r) == -np.sign(rho_marg) for r in rhos if r != 0)
    weakened = bool(rhos) and abs(np.mean(rhos)) < 0.5 * abs(rho_marg)

    if reversal:
        verdict = "SIMPSON REVERSAL - marginal sign is an artefact of the strata mix"
    elif weakened:
        verdict = "association mostly explained by the stratifier - do not claim it as a mechanism"
    else:
        verdict = "association survives conditioning"

    return {"spearman_marginal": float(rho_marg), "p_marginal": float(p_marg),
            "within_strata": within, "sign_reversal": reversal,
            "verdict": verdict}
