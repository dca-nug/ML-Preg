"""
=============================================================================
 ML09 -- MANUSCRIPT TABLES AND FIGURES
=============================================================================

Reads the fitted models from ML02 to ML07 and produces everything that goes
into the manuscript. Fits nothing; every number here comes from predictions
already stored.

TABLES
------
    Table 1   six learners compared, per window and outcome
    Table 2   XGBoost variants: main, SMOTE, multilabel, subgroup, external
    Table 3   incremental value of the diagnoses recorded during pregnancy:
              the during-pregnancy model minus the before-pregnancy model,
              with paired bootstrap intervals
    Table 4   selected hyperparameters
    Table 5   both classification cut-offs side by side

All five are written twice: once per table, and once into
`manuscript_tables.xlsx`, which holds every sheet in one workbook. Each table
has a Formatted sheet, where an estimate and its interval sit in one cell
ready to paste, and a Numeric sheet with the components in separate columns.

FIGURES
-------
    forest plot of AUC with intervals, both windows side by side
    ROC, precision-recall, calibration, and decision curves, six panels per
    figure, one figure per metric per model set and per variant

INCREMENTAL VALUE
-----------------
Table 3 is the analysis the manuscript rests on, and it is the slowest thing
here. The two windows share a test split, so the difference is bootstrapped
in pairs: one resample of row indices, both models scored on the same rows,
the difference taken within the resample. Pairing removes the between-model
correlation from the interval, which an unpaired comparison would leave in
and report as uncertainty. Where the two windows do not share rows the code
falls back to resampling each independently and records that in the `paired`
column.

The calibration slope is fitted by damped iteratively reweighted least
squares rather than a polynomial fit to the calibration curve, which is what
the slope means. It returns NaN when the fit does not converge, which happens
for placental abruption and for pre-pregnancy postpartum haemorrhage: with
666 and 8,162 events the predicted probabilities occupy too narrow a range
to identify a slope. Those cells read NA and are not filled in. When the
point estimate is not estimable the bootstrap for that cell is skipped
entirely, since resampling a quantity that does not converge produces
nothing but runtime.

WHAT IS AND IS NOT COMPARABLE
-----------------------------
The external series is a different cohort with a different prevalence. Its
AUC sits in the same table as the internal one because reviewers expect to
see them together, not because the two are on the same scale. The comparison
that holds is between the two windows within a cohort.

INPUT
-----
    output_ML02 .. output_ML07      whichever exist; missing ones are skipped

OUTPUT (in OUT_DIR)
-------------------
    manuscript_tables.xlsx          every table, one workbook
    table1_model_comparison.xlsx
    table2_xgb_variants.xlsx
    table3_incremental_value.xlsx
    table4_hyperparameters.xlsx
    table5_threshold_comparison.xlsx
    fig_forest_auc.png
    fig_<set>_<metric>_<window>.png
    fig_window_comparison_<variant>_<metric>.png
    benchmark_ML09.csv
    environment_ML09.txt

USAGE
-----
    python ML09_extract_results.py                 everything
    python ML09_extract_results.py --tables        tables only
    python ML09_extract_results.py --figures       figures only
    python ML09_extract_results.py --increment     Table 3 only
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

from ML01_prepare_analytic_set import OUTCOME_LABELS, OUTCOMES
from ml_utils import threshold_metrics

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

DIRS = {"main": "./output_ML02", "comparison": "./output_ML03",
        "smote": "./output_ML04", "multilabel": "./output_ML05",
        "subgroup": "./output_ML06", "external": "./output_ML07"}
OUT_DIR = r"./output_ML09"

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
N_CAL_BINS = 10
DCA_THRESHOLDS = np.round(np.arange(0.01, 0.5001, 0.01), 4)
WINDOWS = ["pre", "during"]

# Display names for the two exposure windows. "pre" and "during" stay as the
# internal keys and in every filename; only what a reader sees changes here.
#
# The pairing matters. "Pre-pregnancy" against "antenatal" mixes two
# vocabularies, and "prenatal" against "antenatal" is not a pairing at all -
# the two words are synonyms, both meaning before birth. "Before" against
# "during" turns on one word and needs no obstetric vocabulary to read.
#
# Whatever is chosen here should match the Methods section.
WINDOW_LABELS = {"pre": "Before pregnancy", "during": "During pregnancy"}

METRIC_KEYS = [("AUC", "auc"), ("AUPRC", "auprc"), ("Accuracy", "acc"),
               ("F1", "f1"), ("Sensitivity", "sens"), ("Specificity", "spec"),
               ("PPV", "ppv"), ("NPV", "npv")]

# Columns kept in the Formatted sheets of Tables 1 and 2. The default is
# everything; drop entries here to narrow a table for the manuscript without
# touching the code that builds it. The Numeric sheets are unaffected and
# always carry every column.
#
# Two groupings are worth knowing when trimming. AUC, AUPRC, Brier, and the
# calibration slope do not depend on a cut-off, so they compare models
# directly. Accuracy, F1, sensitivity, specificity, PPV, and NPV are read at
# a cut-off that differs between models, so a difference between two models
# in those six columns is partly a difference in where their operating points
# fell, not only in how well they rank patients. Keep the cut-off columns if
# the six are kept; drop all nine together if the table is only there to
# justify the choice of learner.
FORMATTED_COLUMNS = [
    "AUC", "AUPRC",                                   # threshold-free
    "Brier", "Calibration slope",                     # threshold-free
    "Accuracy", "F1", "Sensitivity", "Specificity", "PPV", "NPV",
    "Threshold", "Alert rate (test)", "Threshold source",
]

# Mark the reported cut-off on each ROC curve. Only meaningful while the
# threshold-dependent metrics are being reported.
SHOW_OPERATING_POINTS = True

# Build Table 6. Redundant if the threshold columns are dropped above.
BUILD_OPERATING_POINT_TABLE = True

# label, pre file, during file, model key, kind, directory
MODEL_COMPARISON = [
    ("Logistic Regression", "pre_lr_results.pkl", "during_lr_results.pkl",
     "LogisticRegression", "flat", DIRS["comparison"]),
    ("ElasticNet", "pre_elasticnet_results.pkl", "during_elasticnet_results.pkl",
     "ElasticNet", "flat", DIRS["comparison"]),
    ("Random Forest", "pre_rf_results.pkl", "during_rf_results.pkl",
     "RandomForest", "flat", DIRS["comparison"]),
    ("XGBoost", "pre_xgb_results.pkl", "during_xgb_results.pkl",
     "XGBoost", "flat", DIRS["main"]),
    ("CatBoost", "pre_catboost_results.pkl", "during_catboost_results.pkl",
     "CatBoost", "flat", DIRS["comparison"]),
    ("LightGBM", "pre_lgbm_results.pkl", "during_lgbm_results.pkl",
     "LightGBM", "flat", DIRS["comparison"]),
]

XGB_VARIANTS = [
    ("XGBoost (main)", "pre_xgb_results.pkl", "during_xgb_results.pkl",
     "XGBoost", "flat", DIRS["main"]),
    ("XGBoost + SMOTE", "pre_xgb_smote_results.pkl",
     "during_xgb_smote_results.pkl", "XGBoost", "flat", DIRS["smote"]),
    ("XGBoost (multilabel)", "pre_xgb_multi_results.pkl",
     "during_xgb_multi_results.pkl", "XGBoost", "flat", DIRS["multilabel"]),
    ("XGBoost (subgroup)", "pre_xgb_subgroup_results.pkl",
     "during_xgb_subgroup_results.pkl", "XGBoost", "subgroup", DIRS["subgroup"]),
    ("External (TMUCRD)", "extval_pre_results.pkl", "extval_during_results.pkl",
     "XGBoost", "flat", DIRS["external"]),
]

FIGURE_LABELS = {
    "c_abortive": "Abortive outcome", "c_preecl": "Preeclampsia",
    "c_preterm": "Preterm birth", "c_prom": "PROM",
    "c_abrupt": "Placental abruption", "c_pph": "Postpartum haemorrhage",
}

PRE_COLOUR, DURING_COLOUR = "#1F77B4", "#D62728"


# =============================================================================
# LOADING
# =============================================================================


def load_pickle(directory: str, filename: str):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        print(f"  [skip] not found: {path}")
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def expand_series(label, pre_file, during_file, key, kind, directory):
    """Return [(series label, {'pre': {outcome: result}, 'during': ...})].

    A subgroup file holds one results object per stratum, so it expands to one
    series per stratum. Everything else expands to a single series.
    """
    pre_obj = load_pickle(directory, pre_file)
    during_obj = load_pickle(directory, during_file)
    if pre_obj is None and during_obj is None:
        return []

    if kind == "flat":
        pre = {oc: pre_obj[oc][key] for oc in OUTCOMES
               if pre_obj and oc in pre_obj} if pre_obj else {}
        during = {oc: during_obj[oc][key] for oc in OUTCOMES
                  if during_obj and oc in during_obj} if during_obj else {}
        return [(label, {"pre": pre, "during": during})]

    strata = sorted(set((pre_obj or {}).keys()) | set((during_obj or {}).keys()))
    series = []
    for stratum in strata:
        pre = {oc: pre_obj[stratum][oc][key] for oc in OUTCOMES
               if pre_obj and stratum in pre_obj
               and oc in pre_obj[stratum]} if pre_obj else {}
        during = {oc: during_obj[stratum][oc][key] for oc in OUTCOMES
                  if during_obj and stratum in during_obj
                  and oc in during_obj[stratum]} if during_obj else {}
        series.append((f"{label} [{stratum}]", {"pre": pre, "during": during}))
    return series


def collect(registry):
    series = []
    for row in registry:
        series += expand_series(*row)
    return series


# =============================================================================
# METRICS
# =============================================================================


def calibration_slope(y, p) -> float:
    """Slope of a logistic regression of the outcome on the predicted logit.

    Fitted by damped IRLS with a step-halving line search on the deviance,
    which reaches the same maximum-likelihood estimate as a GLM fit while
    converging on outcomes where an undamped fit diverges. Returns NaN on a
    singular design or genuine non-convergence, never a substitute value.
    """
    y = np.asarray(y, float)
    eps = 1e-12
    p = np.asarray(p, float)
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    return _slope_from_logit(y, lp)


def _slope_from_logit(y, lp, cap=1e3) -> float:
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)

    def deviance(b):
        eta = np.clip(X @ b, -30, 30)
        mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        return -2.0 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

    current = deviance(beta)
    for _ in range(30):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-10, None)
        z = eta + (y - mu) / W
        XtW = X.T * W
        try:
            step = np.linalg.solve(XtW @ X, XtW @ z) - beta
        except np.linalg.LinAlgError:
            return np.nan
        factor, accepted = 1.0, False
        for _ in range(12):
            candidate = beta + factor * step
            value = deviance(candidate)
            if np.isfinite(value) and value <= current + 1e-8:
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            break
        converged = np.max(np.abs(factor * step)) < 1e-9
        beta, current = candidate, value
        if converged:
            break
    slope = float(beta[1])
    return slope if np.isfinite(slope) and abs(slope) <= cap else np.nan


def net_benefit(y, p, threshold: float) -> float:
    y = np.asarray(y)
    flagged = p >= threshold
    n = len(y)
    tp = np.sum(flagged & (y == 1))
    fp = np.sum(flagged & (y == 0))
    return tp / n - fp / n * (threshold / (1 - threshold))


def net_benefit_treat_all(y, threshold: float) -> float:
    prevalence = np.mean(y)
    return prevalence - (1 - prevalence) * (threshold / (1 - threshold))


def paired_delta_bootstrap(y_pre, p_pre, y_during, p_during,
                           n=N_BOOTSTRAP, seed=RANDOM_SEED, do_slope=True):
    """Bootstrap the antenatal-minus-pre-pregnancy difference.

    When both windows were scored on the same test rows the resample is a
    single draw of row indices applied to both, so the two models are always
    compared on identical patients and the correlation between them drops out
    of the interval. Otherwise each window is resampled independently, which
    gives a wider and less powerful interval; `paired` records which was used.
    """
    y_pre = np.asarray(y_pre)
    y_during = np.asarray(y_during)
    paired = len(y_pre) == len(y_during) and np.array_equal(y_pre, y_during)

    eps = 1e-12
    lp_pre = np.log(np.clip(p_pre, eps, 1 - eps) / np.clip(1 - p_pre, eps, 1 - eps))
    lp_during = np.log(np.clip(p_during, eps, 1 - eps)
                       / np.clip(1 - p_during, eps, 1 - eps))

    rng = np.random.RandomState(seed)
    delta_auc, delta_brier, delta_slope = [], [], []
    slope_pre, slope_during = [], []

    for _ in range(n):
        if paired:
            idx = rng.randint(0, len(y_during), len(y_during))
            yb = y_during[idx]
            if len(np.unique(yb)) < 2:
                continue
            delta_auc.append(roc_auc_score(yb, p_during[idx])
                             - roc_auc_score(yb, p_pre[idx]))
            delta_brier.append(brier_score_loss(yb, p_during[idx])
                               - brier_score_loss(yb, p_pre[idx]))
            if do_slope:
                s_pre = _slope_from_logit(yb, lp_pre[idx])
                s_during = _slope_from_logit(yb, lp_during[idx])
        else:
            i_pre = rng.randint(0, len(y_pre), len(y_pre))
            i_during = rng.randint(0, len(y_during), len(y_during))
            if (len(np.unique(y_pre[i_pre])) < 2
                    or len(np.unique(y_during[i_during])) < 2):
                continue
            delta_auc.append(roc_auc_score(y_during[i_during], p_during[i_during])
                             - roc_auc_score(y_pre[i_pre], p_pre[i_pre]))
            delta_brier.append(
                brier_score_loss(y_during[i_during], p_during[i_during])
                - brier_score_loss(y_pre[i_pre], p_pre[i_pre]))
            if do_slope:
                s_pre = _slope_from_logit(y_pre[i_pre], lp_pre[i_pre])
                s_during = _slope_from_logit(y_during[i_during], lp_during[i_during])
        if do_slope:
            delta_slope.append(s_during - s_pre)
            slope_pre.append(s_pre)
            slope_during.append(s_during)

    def interval(values):
        finite = np.array([v for v in values if np.isfinite(v)])
        if not len(finite):
            return (np.nan, np.nan)
        return (float(np.percentile(finite, 2.5)),
                float(np.percentile(finite, 97.5)))

    return {"paired": paired, "delta_auc": interval(delta_auc),
            "delta_brier": interval(delta_brier),
            "delta_slope": interval(delta_slope),
            "slope_pre": interval(slope_pre),
            "slope_during": interval(slope_during)}


def fmt(value, lower=None, upper=None, decimals=3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if lower is None or not np.isfinite(lower):
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}f} ({lower:.{decimals}f}-{upper:.{decimals}f})"


# =============================================================================
# TABLES
# =============================================================================


def _metric_rows(series, label_column: str):
    formatted, numeric = [], []
    for window in WINDOWS:
        for series_label, data in series:
            for outcome in OUTCOMES:
                if outcome not in data[window]:
                    continue
                result = data[window][outcome]
                slope = calibration_slope(result["y_test"], result["probas"])

                base = {"Window": WINDOW_LABELS[window], label_column: series_label,
                        "Outcome": OUTCOME_LABELS[outcome]}
                fmt_row, num_row = dict(base), dict(base)

                for name, key in METRIC_KEYS:
                    if key not in result:
                        fmt_row[name], num_row[name] = "NA", np.nan
                        continue
                    num_row[name] = round(result[key], 4)
                    ci = result.get("ci", {})
                    if key in ci:
                        lower, upper = ci[key]["lower"], ci[key]["upper"]
                        num_row[f"{name}_lower"] = round(lower, 4)
                        num_row[f"{name}_upper"] = round(upper, 4)
                        fmt_row[name] = fmt(result[key], lower, upper)
                    else:
                        fmt_row[name] = fmt(result[key])

                brier = result.get("brier", np.nan)
                num_row["Brier"] = round(brier, 4) if brier == brier else np.nan
                ci = result.get("ci", {})
                if "brier" in ci:
                    num_row["Brier_lower"] = round(ci["brier"]["lower"], 4)
                    num_row["Brier_upper"] = round(ci["brier"]["upper"], 4)
                    fmt_row["Brier"] = fmt(brier, ci["brier"]["lower"],
                                           ci["brier"]["upper"], 4)
                else:
                    fmt_row["Brier"] = fmt(brier, decimals=4)

                num_row["Calibration slope"] = (round(slope, 3)
                                                if np.isfinite(slope) else np.nan)
                fmt_row["Calibration slope"] = fmt(slope)
                # The six threshold-dependent metrics above are read at a
                # cut-off that differs between models. Reporting them without
                # the cut-off invites the reader to attribute to the model a
                # difference that belongs to the operating point.
                # The cut-off and the share of the cohort it flags stay out
                # of the Formatted sheet, which is already nine metrics wide.
                # They have their own table and their own marker on the ROC
                # panels; the Numeric sheet keeps them for reference.
                threshold = result.get("threshold", np.nan)
                num_row["Threshold"] = (round(threshold, 4)
                                        if threshold == threshold else np.nan)
                fmt_row["Threshold"] = fmt(threshold, decimals=4)
                alert = alert_rate_on_test(result)
                num_row["Alert rate (test)"] = round(alert, 4)
                fmt_row["Alert rate (test)"] = f"{alert * 100:.1f}%"
                source = result.get("threshold_source", "")
                num_row["Threshold source"] = source
                fmt_row["Threshold source"] = source

                formatted.append(fmt_row)
                numeric.append(num_row)

    formatted = pd.DataFrame(formatted)
    # Identity columns always stay; the rest follow FORMATTED_COLUMNS, in the
    # order given there.
    identity = ["Window", label_column, "Outcome"]
    keep = identity + [c for c in FORMATTED_COLUMNS if c in formatted.columns]
    return formatted[keep], pd.DataFrame(numeric)


def alert_rate_on_test(result) -> float:
    """Share of the test split flagged positive at the reported cut-off.

    The alert rate stored during training is the share flagged in the
    training split, since that is where the cut-off was read. Every metric in
    these tables is computed on the test split, so the figure reported beside
    them is recomputed there.
    """
    p = np.asarray(result["probas"])
    return float(np.mean(p >= result["threshold"]))


def table6_operating_points():
    """Cut-offs and alert rates, models across the columns, one cell each.

    Six learners with overlapping AUC intervals are six points on nearly the
    same curve. This table is what separates them, and it is small enough to
    sit beside the metric table instead of widening it.
    """
    if not BUILD_OPERATING_POINT_TABLE:
        return {}
    print("\nTable 6 -- operating points")
    series = collect(MODEL_COMPARISON) + [
        s for s in collect(XGB_VARIANTS) if "main" not in s[0]]
    if not series:
        return {}
    rows = []
    for window in WINDOWS:
        for outcome in OUTCOMES:
            row = {"Window": WINDOW_LABELS[window],
                   "Outcome": OUTCOME_LABELS[outcome]}
            for label, data in series:
                if outcome not in data[window]:
                    row[label] = "NA"
                    continue
                result = data[window][outcome]
                row[label] = (f"{result['threshold']:.4f} "
                              f"({alert_rate_on_test(result) * 100:.1f}%)")
            rows.append(row)
    frame = pd.DataFrame(rows)
    note = pd.DataFrame([{"Window": "Each cell: probability cut-off "
                          "(share of the test split flagged positive). "
                          "Cut-offs were derived from out-of-fold predictions "
                          "in the training split."}])
    return {"T6_OperatingPoints": pd.concat([frame, note], ignore_index=True)}


def table1_model_comparison():
    print("\nTable 1 -- model comparison")
    series = collect(MODEL_COMPARISON)
    if not series:
        return {}
    formatted, numeric = _metric_rows(series, "Model")
    return {"T1_Formatted": formatted, "T1_Numeric": numeric}


def table2_xgb_variants():
    print("\nTable 2 -- XGBoost variants")
    series = collect(XGB_VARIANTS)
    if not series:
        return {}
    formatted, numeric = _metric_rows(series, "Variant")
    return {"T2_Formatted": formatted, "T2_Numeric": numeric}


def _increment_task(series_label, outcome, pre, during):
    auc_pre, auc_during = pre["auc"], during["auc"]
    brier_pre, brier_during = pre["brier"], during["brier"]
    slope_pre = calibration_slope(pre["y_test"], pre["probas"])
    slope_during = calibration_slope(during["y_test"], during["probas"])

    do_slope = np.isfinite(slope_pre) and np.isfinite(slope_during)
    boot = paired_delta_bootstrap(pre["y_test"], pre["probas"],
                                  during["y_test"], during["probas"],
                                  do_slope=do_slope)
    significant = boot["delta_auc"][0] > 0 or boot["delta_auc"][1] < 0

    def stored(result, key):
        ci = result.get("ci", {})
        return ((ci[key]["lower"], ci[key]["upper"]) if key in ci
                else (np.nan, np.nan))

    auc_pre_ci, auc_during_ci = stored(pre, "auc"), stored(during, "auc")
    brier_pre_ci, brier_during_ci = stored(pre, "brier"), stored(during, "brier")

    return {
        "Variant": series_label, "Outcome": OUTCOME_LABELS[outcome],
        "paired": boot["paired"],
        "AUC_pre": round(auc_pre, 3),
        "AUC_pre_lower": round(auc_pre_ci[0], 3),
        "AUC_pre_upper": round(auc_pre_ci[1], 3),
        "AUC_during": round(auc_during, 3),
        "AUC_during_lower": round(auc_during_ci[0], 3),
        "AUC_during_upper": round(auc_during_ci[1], 3),
        "delta_AUC": round(auc_during - auc_pre, 4),
        "delta_AUC_lower": round(boot["delta_auc"][0], 4),
        "delta_AUC_upper": round(boot["delta_auc"][1], 4),
        "delta_AUC_excludes_zero": bool(significant),
        "Brier_pre": round(brier_pre, 4),
        "Brier_pre_lower": round(brier_pre_ci[0], 4),
        "Brier_pre_upper": round(brier_pre_ci[1], 4),
        "Brier_during": round(brier_during, 4),
        "Brier_during_lower": round(brier_during_ci[0], 4),
        "Brier_during_upper": round(brier_during_ci[1], 4),
        "delta_Brier": round(brier_during - brier_pre, 4),
        "delta_Brier_lower": round(boot["delta_brier"][0], 4),
        "delta_Brier_upper": round(boot["delta_brier"][1], 4),
        "Slope_pre": round(slope_pre, 3) if np.isfinite(slope_pre) else np.nan,
        "Slope_pre_lower": round(boot["slope_pre"][0], 3),
        "Slope_pre_upper": round(boot["slope_pre"][1], 3),
        "Slope_during": round(slope_during, 3) if np.isfinite(slope_during) else np.nan,
        "Slope_during_lower": round(boot["slope_during"][0], 3),
        "Slope_during_upper": round(boot["slope_during"][1], 3),
        "delta_Slope": (round(slope_during - slope_pre, 4)
                        if do_slope else np.nan),
        "delta_Slope_lower": round(boot["delta_slope"][0], 3),
        "delta_Slope_upper": round(boot["delta_slope"][1], 3),
    }


def table3_incremental_value(n_jobs=-1):
    print("\nTable 3 -- incremental value of antenatal diagnoses")
    series = collect(XGB_VARIANTS)
    jobs = [(label, outcome, data["pre"][outcome], data["during"][outcome])
            for label, data in series for outcome in OUTCOMES
            if outcome in data["pre"] and outcome in data["during"]]
    if not jobs:
        print("  nothing to compare")
        return {}
    print(f"  {len(jobs)} variant-by-outcome comparisons, "
          f"{N_BOOTSTRAP} paired resamples each")
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_increment_task)(*job) for job in jobs)
    numeric = pd.DataFrame(rows)

    formatted = pd.DataFrame({
        "Variant": numeric["Variant"], "Outcome": numeric["Outcome"],
        f"AUC ({WINDOW_LABELS['pre'].lower()})": [
            fmt(r.AUC_pre, r.AUC_pre_lower, r.AUC_pre_upper)
            for r in numeric.itertuples()],
        f"AUC ({WINDOW_LABELS['during'].lower()})": [
            fmt(r.AUC_during, r.AUC_during_lower, r.AUC_during_upper)
            for r in numeric.itertuples()],
        "Delta AUC (95% CI)": [
            fmt(r.delta_AUC, r.delta_AUC_lower, r.delta_AUC_upper)
            for r in numeric.itertuples()],
        "CI excludes zero": numeric["delta_AUC_excludes_zero"],
        f"Brier ({WINDOW_LABELS['pre'].lower()})": [
            fmt(r.Brier_pre, r.Brier_pre_lower, r.Brier_pre_upper, 4)
            for r in numeric.itertuples()],
        f"Brier ({WINDOW_LABELS['during'].lower()})": [
            fmt(r.Brier_during, r.Brier_during_lower, r.Brier_during_upper, 4)
            for r in numeric.itertuples()],
        "Delta Brier (95% CI)": [
            fmt(r.delta_Brier, r.delta_Brier_lower, r.delta_Brier_upper, 4)
            for r in numeric.itertuples()],
        f"Calibration slope ({WINDOW_LABELS['pre'].lower()})": [
            fmt(r.Slope_pre, r.Slope_pre_lower, r.Slope_pre_upper)
            for r in numeric.itertuples()],
        f"Calibration slope ({WINDOW_LABELS['during'].lower()})": [
            fmt(r.Slope_during, r.Slope_during_lower, r.Slope_during_upper)
            for r in numeric.itertuples()],
        "Delta calibration slope (95% CI)": [
            fmt(r.delta_Slope, r.delta_Slope_lower, r.delta_Slope_upper)
            for r in numeric.itertuples()],
    })
    return {"T3_Formatted": formatted, "T3_Numeric": numeric}


def table4_hyperparameters():
    print("\nTable 4 -- hyperparameters")
    rows = []
    for window, filename in [("pre", "pre_xgb_results.pkl"),
                             ("during", "during_xgb_results.pkl")]:
        obj = load_pickle(DIRS["main"], filename)
        if obj is None:
            continue
        for outcome in OUTCOMES:
            if outcome not in obj:
                continue
            entry = obj[outcome]["XGBoost"]
            rows.append({"Window": WINDOW_LABELS[window],
                         "Outcome": OUTCOME_LABELS[outcome],
                         "Features": len(entry.get("feat_names", [])),
                         "Training episodes": entry.get("n_train"),
                         "Test episodes": entry.get("n_test"),
                         **entry.get("best_params", {})})
    return {"T4_Hyperparameters": pd.DataFrame(rows)} if rows else {}


def table5_threshold_comparison():
    """Both cut-offs side by side, from the diagnostics files each script writes.

    The gap between them is the optimism that would have entered the reported
    sensitivity and specificity had the cut-off been chosen on the test split.
    """
    print("\nTable 5 -- cut-off comparison")
    frames = []
    for directory in DIRS.values():
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if filename.startswith("threshold_diagnostics") and filename.endswith(".csv"):
                frame = pd.read_csv(os.path.join(directory, filename))
                frame.insert(0, "source", os.path.basename(directory))
                frames.append(frame)
    if not frames:
        print("  no threshold diagnostics found")
        return {}
    combined = pd.concat(frames, ignore_index=True)
    if {"sens_train_oof", "sens_test_reopt"} <= set(combined.columns):
        combined["sens_optimism"] = (combined["sens_test_reopt"]
                                     - combined["sens_train_oof"]).round(4)
    return {"T5_Thresholds": combined}


# =============================================================================
# FIGURES
# =============================================================================


def _panel_grid():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    return fig, axes.ravel()


def _draw_curve(ax, metric, y, p, result, colour=None, name=""):
    if metric == "roc":
        fpr, tpr, _ = roc_curve(y, p)
        line, = ax.plot(fpr, tpr, lw=1.4, color=colour,
                        label=f"{name} (AUC {result['auc']:.3f})")
        # Where the reported cut-off sits on the curve. Six models with
        # overlapping curves can still report very different sensitivity, and
        # the marker shows that the difference is the operating point rather
        # than the model.
        if SHOW_OPERATING_POINTS and "threshold" in result:
            at = threshold_metrics(np.asarray(y), np.asarray(p),
                                   result["threshold"])
            ax.plot(1 - at["spec"], at["sens"], marker="o", ms=6,
                    color=line.get_color(), markeredgecolor="white",
                    markeredgewidth=0.8, linestyle="none")
    elif metric == "pr":
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, lw=1.4, color=colour,
                label=f"{name} (AUPRC {result['auprc']:.3f})")
    elif metric == "cal":
        try:
            observed, predicted = calibration_curve(
                y, p, n_bins=N_CAL_BINS, strategy="quantile")
            slope = calibration_slope(y, p)
            suffix = (f"slope {slope:.2f}" if np.isfinite(slope) else "slope NA")
            ax.plot(predicted, observed, marker="o", ms=3.5, lw=1.2,
                    color=colour, label=f"{name} ({suffix})")
        except Exception:
            pass
    elif metric == "dca":
        benefit = [net_benefit(y, p, t) for t in DCA_THRESHOLDS]
        ax.plot(DCA_THRESHOLDS, benefit, lw=1.4, color=colour, label=name)
        return float(np.max(benefit))
    return 0.0


def _finish_panel(ax, metric, outcome, reference_y, dca_max, probability_sets):
    if metric == "roc":
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("Sensitivity")
    elif metric == "pr":
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    elif metric == "cal":
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=0.9,
                label="Perfect calibration")
        # Predicted probabilities for a rare outcome occupy a small part of
        # the unit square; plotting the whole of it compresses every curve
        # into the corner.
        limit = max([np.percentile(p, 99) for p in probability_sets] + [0.05])
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction")
    elif metric == "dca":
        if reference_y is not None:
            ax.plot(DCA_THRESHOLDS,
                    [net_benefit_treat_all(reference_y, t) for t in DCA_THRESHOLDS],
                    "k--", lw=0.9, label="Treat all")
            ax.axhline(0, color="gray", lw=0.9, label="Treat none")
        # The treat-all curve falls steeply below zero for a rare outcome. Left
        # unclipped it sets the axis range and flattens the model curves into a
        # line, which is what the figure exists to show.
        top = max(dca_max, 1e-3) * 1.15
        ax.set_ylim(-0.12 * top, top)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
    ax.set_title(FIGURE_LABELS[outcome], fontsize=11)
    ax.legend(fontsize=6.5, loc="best")


def figure_set(set_name: str, registry, window: str) -> None:
    series = collect(registry)
    if not series:
        return
    titles = {"roc": "ROC curves", "pr": "Precision-recall curves",
              "cal": "Calibration", "dca": "Decision curve analysis"}

    for metric in ("roc", "pr", "cal", "dca"):
        fig, axes = _panel_grid()
        for ax, outcome in zip(axes, OUTCOMES):
            dca_max, reference_y, probability_sets = 0.0, None, []
            for series_label, data in series:
                if outcome not in data[window]:
                    continue
                result = data[window][outcome]
                y = np.asarray(result["y_test"])
                p = np.asarray(result["probas"])
                if len(np.unique(y)) < 2:
                    continue
                if reference_y is None:
                    reference_y = y
                probability_sets.append(p)
                dca_max = max(dca_max,
                              _draw_curve(ax, metric, y, p, result,
                                          name=series_label))
            _finish_panel(ax, metric, outcome, reference_y, dca_max,
                          probability_sets or [np.array([0.05])])
        fig.suptitle(f"{set_name} -- {titles[metric]} -- "
                     f"{WINDOW_LABELS[window].lower()} window", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(OUT_DIR, f"fig_{set_name}_{metric}_{window}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  -> {path}")


def figure_window_comparison(variant_label: str, prefix: str, pre, during) -> None:
    titles = {"roc": "ROC curves", "pr": "Precision-recall curves",
              "cal": "Calibration", "dca": "Decision curve analysis"}
    for metric in ("roc", "pr", "cal", "dca"):
        fig, axes = _panel_grid()
        for ax, outcome in zip(axes, OUTCOMES):
            dca_max, reference_y, probability_sets = 0.0, None, []
            for data, colour, name in [
                    (pre, PRE_COLOUR, WINDOW_LABELS["pre"]),
                    (during, DURING_COLOUR, WINDOW_LABELS["during"])]:
                if outcome not in data:
                    continue
                result = data[outcome]
                y = np.asarray(result["y_test"])
                p = np.asarray(result["probas"])
                if len(np.unique(y)) < 2:
                    continue
                if reference_y is None:
                    reference_y = y
                probability_sets.append(p)
                dca_max = max(dca_max,
                              _draw_curve(ax, metric, y, p, result, colour, name))
            _finish_panel(ax, metric, outcome, reference_y, dca_max,
                          probability_sets or [np.array([0.05])])
        fig.suptitle(f"{titles[metric]} -- {WINDOW_LABELS['pre'].lower()} "
                     f"against {WINDOW_LABELS['during'].lower()} "
                     f"({variant_label})", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(OUT_DIR, f"fig_window_comparison_{prefix}_{metric}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  -> {path}")


def figure_forest() -> None:
    pre = load_pickle(DIRS["main"], "pre_xgb_results.pkl")
    during = load_pickle(DIRS["main"], "during_xgb_results.pkl")
    if pre is None or during is None:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ticks, labels, position = [], [], 0
    for outcome in OUTCOMES:
        for name, obj, colour, offset in [
                (WINDOW_LABELS["pre"], pre, PRE_COLOUR, 0.16),
                (WINDOW_LABELS["during"], during, DURING_COLOUR, -0.16)]:
            if outcome not in obj:
                continue
            result = obj[outcome]["XGBoost"]
            auc = result["auc"]
            lower = result["ci"]["auc"]["lower"]
            upper = result["ci"]["auc"]["upper"]
            ax.errorbar(auc, position + offset,
                        xerr=[[auc - lower], [upper - auc]], fmt="o", ms=6,
                        color=colour, capsize=3, lw=1.6,
                        label=name if outcome == OUTCOMES[0] else "")
            ax.annotate(f"{auc:.3f} [{lower:.3f}-{upper:.3f}]",
                        (auc, position + offset), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)
        ticks.append(position)
        labels.append(FIGURE_LABELS[outcome])
        position += 1
    ax.axvline(0.5, color="gray", ls="--", lw=0.9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUC (95% CI)")
    ax.legend(loc="lower right", frameon=True)
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_forest_auc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


# =============================================================================
# MAIN
# =============================================================================


def write_workbook(sheets: dict, path: str) -> None:
    if not sheets:
        return
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"  XLSX -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tables and figures.")
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--increment", action="store_true")
    args = parser.parse_args()
    everything = not (args.tables or args.figures or args.increment)

    os.makedirs(OUT_DIR, exist_ok=True)

    from ml_utils import Bench, environment_report, warn_if_no_psutil
    print("=" * 74)
    print(" ML09 -- MANUSCRIPT TABLES AND FIGURES")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML09.txt")))
    warn_if_no_psutil()

    bench = Bench("ML09, extraction")
    workbook = {}

    if everything or args.tables:
        with bench.stage("Tables 1, 2, 4, 5", "pandas"):
            for builder, filename in [
                    (table1_model_comparison, "table1_model_comparison.xlsx"),
                    (table2_xgb_variants, "table2_xgb_variants.xlsx"),
                    (table4_hyperparameters, "table4_hyperparameters.xlsx"),
                    (table6_operating_points, "table6_operating_points.xlsx"),
                    (table5_threshold_comparison,
                     "table5_threshold_comparison.xlsx")]:
                sheets = builder()
                workbook.update(sheets)
                write_workbook(sheets, os.path.join(OUT_DIR, filename))

    if everything or args.figures:
        with bench.stage("Figures", "matplotlib"):
            figure_forest()
            for window in WINDOWS:
                figure_set("model_comparison", MODEL_COMPARISON, window)
                figure_set("xgb_variants", XGB_VARIANTS[:-1], window)
                figure_set("main_and_external",
                           [XGB_VARIANTS[0], XGB_VARIANTS[-1]], window)
            for row in XGB_VARIANTS:
                for label, data in expand_series(*row):
                    prefix = (label.lower().replace(" ", "_")
                              .replace("(", "").replace(")", "")
                              .replace("+", "").replace("[", "")
                              .replace("]", "").replace("__", "_"))
                    figure_window_comparison(label, prefix,
                                             data["pre"], data["during"])

    if everything or args.increment:
        with bench.stage("Table 3, paired bootstrap", "joblib"):
            sheets = table3_incremental_value()
            workbook.update(sheets)
            write_workbook(sheets,
                           os.path.join(OUT_DIR, "table3_incremental_value.xlsx"))

    if workbook:
        write_workbook(workbook, os.path.join(OUT_DIR, "manuscript_tables.xlsx"))

    bench.finalise(os.path.join(OUT_DIR, "benchmark_ML09.csv"))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()