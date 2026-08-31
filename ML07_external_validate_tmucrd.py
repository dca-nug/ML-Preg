"""
=============================================================================
 ML07 -- EXTERNAL VALIDATION ON TMUCRD
=============================================================================

Applies the BPJS-derived models to the Taipei Medical University Clinical
Research Database without refitting. The stored model, the stored scaler, and
the stored cut-off are used as they are; nothing is tuned on the external
cohort. Re-optimising anything on TMUCRD would make this an assessment of
whether a model can be fitted there, which is a different question.

WHAT IS HELD FIXED AND WHAT IS NOT
----------------------------------
    model, scaler      loaded from the ML02 outputs, applied unchanged
    feature order      taken from the stored `feat_names`, not from TMUCRD
    cut-off            the `train_oof` cut-off derived in BPJS, transported

A second cut-off, re-optimised on TMUCRD, is computed and reported beside the
transported one. It is not a validation result. It is the ceiling the
transported cut-off is compared against, and the gap between them is the cost
of not being able to recalibrate the operating point in a new setting.

FEATURES THAT DO NOT EXIST IN THE EXTERNAL COHORT
-------------------------------------------------
    subsid    absent from TMUCRD. Fixed at 0.
    dom       present but not comparable. TMUCRD is a single metropolitan
              hospital system, so the Java-Bali versus outer-islands contrast
              has no analogue. Fixed at 0.
    anything else absent is filled with 0, which asserts the diagnosis was
              not recorded.

That last assertion is the weak point of the whole exercise, and it is not
symmetric: a diagnosis absent from TMUCRD because the coding system differs
is indistinguishable from a diagnosis absent because the patient did not have
it. The number of filled features and the share of the model's SHAP weight
they carry are both reported per outcome, so a validation result resting
mostly on zero-filled inputs is identifiable rather than reported as if it
were a like-for-like comparison.

A sensitivity check refits nothing but re-predicts with subsid and dom set to
1 instead of 0, to show how much of the external result depends on the two
constants chosen above.

INPUT
-----
    tmucrd_pregnancy_by_episode.csv      from step 10 of the cohort pipeline
    pre_xgb_results.pkl                  from ML02
    during_xgb_results.pkl

OUTPUT (in OUT_DIR)
-------------------
    extval_metrics.xlsx                  external metrics per window, outcome
    extval_comparison.xlsx               internal, external, and difference
    extval_calibration.xlsx              calibration bin data
    extval_coverage.xlsx                 features filled with zero, per model
    extval_invariance.xlsx               subsid and dom set to 0 against 1
    extval_pre_results.pkl               predictions and metrics
    extval_during_results.pkl
    extval_calibration_<window>.png
    extval_auc_comparison.png
    benchmark_ML07.csv
    environment_ML07.txt

USAGE
-----
    python ML07_external_validate_tmucrd.py
=============================================================================
"""

from __future__ import annotations

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

import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

from ML01_prepare_analytic_set import OUTCOME_LABELS, OUTCOMES, add_par_risk
from ml_core import THRESHOLD_NAMES
from ml_utils import (Bench, bootstrap_metrics, environment_report, fmt_ci,
                      save_pickle, threshold_metrics, warn_if_no_psutil,
                      youden_threshold)

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

TMUCRD_PATH = r"./tmucrd_pregnancy_by_episode.csv"
BPJS_DIR = r"./output_ML02"
OUT_DIR = r"./output_ML07"

MODEL_KEY = "XGBoost"
N_BOOTSTRAP = 1000
N_CAL_BINS = 10

# Applied to every model. See the header for why each is held at a constant.
FIXED_VALUES = {"subsid": 0, "dom": 0}

WINDOWS = {"pre": "pre_xgb_results.pkl", "during": "during_xgb_results.pkl"}


# =============================================================================
# FEATURE MATRIX
# =============================================================================


def build_external_matrix(df: pd.DataFrame, feat_names: list[str],
                          overrides: dict) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the external feature matrix in the stored feature order.

    Column order follows `feat_names` rather than the external file, because
    a tree ensemble applied to a permuted matrix produces predictions without
    complaining.
    """
    columns, filled = {}, []
    n = len(df)
    for feature in feat_names:
        if feature in overrides:
            columns[feature] = np.full(n, overrides[feature], dtype=float)
        elif feature in df.columns:
            columns[feature] = df[feature].values.astype(float)
        else:
            columns[feature] = np.zeros(n, dtype=float)
            filled.append(feature)
    return pd.DataFrame(columns, columns=feat_names), filled


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray):
    """Cox calibration: slope from logit(p) as the sole predictor, intercept
    from an offset model with the slope constrained to one.

    Returns NaN where the fit does not converge, which happens when predicted
    probabilities occupy a narrow range. That is reported rather than
    replaced with a value.
    """
    eps = 1e-12
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    try:
        slope = sm.GLM(y, sm.add_constant(lp),
                       family=sm.families.Binomial()).fit().params[1]
    except Exception:
        slope = np.nan
    try:
        intercept = sm.GLM(y, np.ones((len(y), 1)),
                           family=sm.families.Binomial(),
                           offset=lp).fit().params[0]
    except Exception:
        intercept = np.nan
    return float(intercept), float(slope)


# =============================================================================
# ONE WINDOW
# =============================================================================


def validate_window(bpjs: dict, df: pd.DataFrame, window: str):
    print(f"\n{'#' * 74}\n#  EXTERNAL VALIDATION -- {window.upper()}"
          f"   (BPJS model applied to TMUCRD)\n{'#' * 74}")

    rows, comparison, calibration, coverage, invariance = [], [], [], [], []
    results, curves = {}, {}

    for outcome in OUTCOMES:
        label = OUTCOME_LABELS[outcome]
        if outcome not in df.columns:
            print(f"  {label}: outcome absent from TMUCRD, skipped")
            continue

        entry = bpjs[outcome][MODEL_KEY]
        model, scaler = entry["model"], entry["scaler"]
        feat_names = entry["feat_names"]

        y = df[outcome].values.astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            print(f"  {label}: no variation in the external outcome, skipped")
            continue

        X, filled = build_external_matrix(df, feat_names, FIXED_VALUES)
        Xt = (pd.DataFrame(scaler.transform(X), columns=feat_names)
              if scaler is not None else X)
        p = model.predict_proba(Xt)[:, 1]

        thresholds = {
            "bpjs_transported": float(entry["threshold"]),
            "tmucrd_reoptimised": youden_threshold(y, p),
        }

        ranking = {"auc": roc_auc_score(y, p),
                   "auprc": average_precision_score(y, p),
                   "brier": brier_score_loss(y, p)}
        at_cutoff = {name: threshold_metrics(y, p, thr)
                     for name, thr in thresholds.items()}
        ci = bootstrap_metrics(y, p, thresholds, n_bootstraps=N_BOOTSTRAP)
        intercept, slope = calibration_intercept_slope(y, p)
        primary = at_cutoff["bpjs_transported"]

        print(f"\n  {label}")
        print(f"      episodes {len(y):,}   events {int(y.sum()):,} "
              f"({y.mean() * 100:.2f}%)   features filled with zero "
              f"{len(filled)}/{len(feat_names)}")
        print(f"      AUC {ranking['auc']:.3f} "
              f"[{ci['bpjs_transported']['auc']['lower']:.3f}-"
              f"{ci['bpjs_transported']['auc']['upper']:.3f}]   "
              f"AUPRC {ranking['auprc']:.3f}   Brier {ranking['brier']:.4f}")
        print(f"      calibration intercept {intercept:+.3f}   "
              f"slope {slope:.3f}   Sens {primary['sens']:.3f}   "
              f"Spec {primary['spec']:.3f}")

        rows.append({
            "Window": window, "Outcome": label,
            "n": len(y), "events": int(y.sum()),
            "prevalence": round(float(y.mean()), 6),
            "AUC": round(ranking["auc"], 3),
            "AUC_lower": round(ci["bpjs_transported"]["auc"]["lower"], 3),
            "AUC_upper": round(ci["bpjs_transported"]["auc"]["upper"], 3),
            "AUPRC": round(ranking["auprc"], 3),
            "Brier": round(ranking["brier"], 4),
            "Calibration_intercept": round(intercept, 3),
            "Calibration_slope": round(slope, 3),
            "Threshold_transported": round(thresholds["bpjs_transported"], 4),
            "Threshold_reoptimised": round(thresholds["tmucrd_reoptimised"], 4),
            **{f"{m.capitalize()}_transported": round(primary[m], 3)
               for m in ("acc", "f1", "sens", "spec", "ppv", "npv")},
            **{f"{m.capitalize()}_reoptimised":
               round(at_cutoff["tmucrd_reoptimised"][m], 3)
               for m in ("acc", "f1", "sens", "spec", "ppv", "npv")},
        })

        # internal against external, same model, same cut-off definition
        comparison.append({
            "Window": window, "Outcome": label,
            "AUC_internal": round(entry["auc"], 3),
            "AUC_external": round(ranking["auc"], 3),
            "AUC_difference": round(ranking["auc"] - entry["auc"], 3),
            "AUPRC_internal": round(entry["auprc"], 3),
            "AUPRC_external": round(ranking["auprc"], 3),
            "AUPRC_difference": round(ranking["auprc"] - entry["auprc"], 3),
            "Brier_internal": round(entry["brier"], 4),
            "Brier_external": round(ranking["brier"], 4),
            "Prevalence_internal": round(float(entry["y_test"].mean()), 6),
            "Prevalence_external": round(float(y.mean()), 6),
            "Sens_internal": round(entry["sens"], 3),
            "Sens_external": round(primary["sens"], 3),
            "Spec_internal": round(entry["spec"], 3),
            "Spec_external": round(primary["spec"], 3),
        })

        coverage.append({
            "Window": window, "Outcome": label,
            "n_features": len(feat_names),
            "n_present": len(feat_names) - len(filled) - len(FIXED_VALUES),
            "n_fixed": len(FIXED_VALUES),
            "n_filled_zero": len(filled),
            "share_filled_zero": round(len(filled) / len(feat_names), 4),
            "filled_features": ";".join(filled),
        })

        try:
            frac_pos, mean_pred = calibration_curve(
                y, p, n_bins=N_CAL_BINS, strategy="quantile")
            for observed, predicted in zip(frac_pos, mean_pred):
                calibration.append({"Window": window, "Outcome": label,
                                    "mean_predicted": round(predicted, 5),
                                    "observed_fraction": round(observed, 5)})
            curves[outcome] = (mean_pred, frac_pos)
        except Exception as exc:
            print(f"      calibration curve not estimable: {exc}")

        # how much of the result rests on the two constants
        X1, _ = build_external_matrix(df, feat_names,
                                      {k: 1 for k in FIXED_VALUES})
        X1t = (pd.DataFrame(scaler.transform(X1), columns=feat_names)
               if scaler is not None else X1)
        p1 = model.predict_proba(X1t)[:, 1]
        invariance.append({
            "Window": window, "Outcome": label,
            "AUC_fixed_0": round(ranking["auc"], 3),
            "AUC_fixed_1": round(roc_auc_score(y, p1), 3),
            "AUC_difference": round(roc_auc_score(y, p1) - ranking["auc"], 4),
            "mean_absolute_probability_shift": round(float(np.mean(np.abs(p1 - p))), 5),
            "max_absolute_probability_shift": round(float(np.max(np.abs(p1 - p))), 5),
        })

        results[outcome] = {MODEL_KEY: {
            "probas": p, "y_test": y, "feat_names": feat_names,
            "auc": ranking["auc"], "auprc": ranking["auprc"],
            "brier": ranking["brier"],
            **{m: primary[m] for m in ("acc", "f1", "sens", "spec", "ppv", "npv")},
            "threshold": thresholds["bpjs_transported"],
            "ci": ci["bpjs_transported"],
            "thresholds": thresholds,
            "metrics_by_threshold": at_cutoff, "ci_by_threshold": ci,
            "calib_intercept": intercept, "calib_slope": slope,
            "filled_features": filled,
        }}

    if curves:
        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        for outcome, (predicted, observed) in curves.items():
            plt.plot(predicted, observed, marker="o",
                     label=OUTCOME_LABELS[outcome])
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed fraction")
        plt.title(f"External calibration, {window} window")
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"extval_calibration_{window}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\n      figure -> {path}")

    return rows, comparison, calibration, coverage, invariance, results


def plot_auc_comparison(comparison: list[dict], path: str) -> None:
    frame = pd.DataFrame(comparison)
    if frame.empty:
        return
    windows = frame["Window"].unique()
    fig, axes = plt.subplots(1, len(windows), figsize=(6 * len(windows), 5),
                             squeeze=False)
    for ax, window in zip(axes[0], windows):
        subset = frame[frame["Window"] == window]
        x = np.arange(len(subset))
        ax.bar(x - 0.2, subset["AUC_internal"], 0.4, label="BPJS (internal)")
        ax.bar(x + 0.2, subset["AUC_external"], 0.4, label="TMUCRD (external)")
        ax.axhline(0.5, color="grey", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(subset["Outcome"], rotation=30, ha="right")
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUC")
        ax.set_title(f"{window} window")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  figure -> {path}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML07 -- EXTERNAL VALIDATION ON TMUCRD")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML07.txt")))
    warn_if_no_psutil()

    bench = Bench("ML07, external validation")

    with bench.stage("Load external cohort", "pandas") as b:
        if not os.path.exists(TMUCRD_PATH):
            raise SystemExit(f"[FATAL] not found: {os.path.abspath(TMUCRD_PATH)}")
        df = add_par_risk(pd.read_csv(TMUCRD_PATH, low_memory=False))
        print(f"    {len(df):,} episodes x {len(df.columns)} columns")
        present = [oc for oc in OUTCOMES if oc in df.columns]
        for outcome in present:
            print(f"    {OUTCOME_LABELS[outcome]:<13}: "
                  f"{int(df[outcome].sum()):>7,} "
                  f"({df[outcome].mean() * 100:5.2f}%)")
        missing = [oc for oc in OUTCOMES if oc not in df.columns]
        if missing:
            print(f"    outcomes absent from TMUCRD: {missing}")
        b["rows_out"] = len(df)

    all_rows, all_comparison, all_calibration = [], [], []
    all_coverage, all_invariance = [], []

    for window, filename in WINDOWS.items():
        path = os.path.join(BPJS_DIR, filename)
        if not os.path.exists(path):
            raise SystemExit(f"[FATAL] model file not found: {path}. Run ML02 first.")
        with open(path, "rb") as handle:
            bpjs = pickle.load(handle)

        with bench.stage(f"Validate ({window})", "sklearn", rows_in=len(df)) as b:
            rows, comparison, calibration, coverage, invariance, results = \
                validate_window(bpjs, df, window)
            all_rows += rows
            all_comparison += comparison
            all_calibration += calibration
            all_coverage += coverage
            all_invariance += invariance
            save_pickle(results,
                        os.path.join(OUT_DIR, f"extval_{window}_results.pkl"))
            b["rows_out"] = len(df)

    pd.DataFrame(all_rows).to_excel(
        os.path.join(OUT_DIR, "extval_metrics.xlsx"), index=False)
    pd.DataFrame(all_comparison).to_excel(
        os.path.join(OUT_DIR, "extval_comparison.xlsx"), index=False)
    pd.DataFrame(all_calibration).to_excel(
        os.path.join(OUT_DIR, "extval_calibration.xlsx"), index=False)
    pd.DataFrame(all_coverage).to_excel(
        os.path.join(OUT_DIR, "extval_coverage.xlsx"), index=False)
    pd.DataFrame(all_invariance).to_excel(
        os.path.join(OUT_DIR, "extval_invariance.xlsx"), index=False)

    plot_auc_comparison(all_comparison,
                        os.path.join(OUT_DIR, "extval_auc_comparison.png"))

    if all_coverage:
        worst = max(all_coverage, key=lambda c: c["share_filled_zero"])
        print(f"\n  largest share of features filled with zero: "
              f"{worst['share_filled_zero'] * 100:.1f}% "
              f"({worst['Window']}, {worst['Outcome']})")
        if worst["share_filled_zero"] > 0.10:
            print("  [note] more than a tenth of the feature set is absent "
                  "from the external cohort for at least one model. State "
                  "this in the manuscript; the external estimate is partly a "
                  "measure of how the model behaves on inputs it was not "
                  "given.")

    bench.finalise(os.path.join(OUT_DIR, "benchmark_ML07.csv"), rows_out=len(df))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
