"""
=============================================================================
 ML02 -- PRIMARY MODEL: XGBoost, PRE-PREGNANCY AND ANTENATAL WINDOWS
=============================================================================

Fits the models the manuscript reports. One binary classifier per outcome per
exposure window: six outcomes, two windows, twelve models. Everything except
the feature set is held identical between the two windows, so the difference
in performance between them is attributable to the added antenatal diagnoses
rather than to a difference in how the models were built.

The fitting and scoring procedure lives in `ml_core.fit_evaluate` and is
shared with ML03 to ML06. This script supplies the estimator, the grid, and
the SHAP step, which is specific to the primary model.

THE CLASSIFICATION CUT-OFF
--------------------------
The analysis notebooks selected the cut-off by maximising the Youden index on
the test set, then reported sensitivity, specificity, F1, PPV and NPV at that
cut-off on the same test set. A cut-off is a parameter; choosing it on the
data used to evaluate it makes those five figures optimistic. Two cut-offs
are therefore reported:

    train_oof     derived from out-of-fold predictions within the training
                  split and transported to the reported probability scale by
                  its predicted-positive rate. Primary, and the cut-off
                  transported to the external cohort in ML07.

    test_reopt    Youden recomputed on the test split, as the notebooks did.
                  Reported alongside as an upper bound; the gap between the
                  two is how much of the operating-point performance comes
                  from having chosen the operating point on the test data.

AUC, AUPRC, Brier, and everything downstream of them - incremental
discrimination, calibration, decision curves - do not depend on a cut-off and
are unaffected by the change.

    `use_label_encoder` is no longer passed to XGBClassifier. It was removed
    from the library in version 2.0 and ignored with a warning in 3.x, so
    dropping it changes the warning output and nothing else.

INPUT
-----
    pregnancy by episode cleaned.csv     read through ML01

OUTPUT (in OUT_DIR)
-------------------
    pre_xgb_results.pkl                  models, predictions, metrics
    during_xgb_results.pkl
    pre_xgb_scalers.pkl                  fitted StandardScaler per outcome
    during_xgb_scalers.pkl
    pre_xgb_shap_values.pkl              mean absolute and raw SHAP
    during_xgb_shap_values.pkl
    pre_xgb_metrics.xlsx                 Summary, Numeric, By_threshold
    during_xgb_metrics.xlsx
    pre_xgb_shap_top20.xlsx
    during_xgb_shap_top20.xlsx
    {prefix}_shap_<outcome>.png
    threshold_diagnostics.csv            both cut-offs, alert rates, rho
    benchmark_ML02.csv
    environment_ML02.txt

PKL filenames and key names are unchanged from the notebooks, so ML10 reads
them without modification.

USAGE
-----
    python ML02_train_xgb_main.py
=============================================================================
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ML01_prepare_analytic_set import (OUTCOME_LABELS, OUTCOMES, WINDOWS,
                                       build_matrix, load_analytic_set)
from ml_core import diagnostic_row, fit_evaluate, metrics_sheets, write_sheets
from ml_utils import (Bench, environment_report, save_pickle,
                      warn_if_no_psutil)

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR = r"./output_ML02"

RANDOM_STATE = 123
TEST_SIZE = 0.30
CV_FOLDS = 5
N_BOOTSTRAP = 1000
N_JOBS = -1
TOP_N_SHAP = 20

# Rows sampled from the test split for the SHAP explainer. TreeExplainer is
# exact but linear in rows; the mean absolute value is stable well below the
# full split. None uses every test row.
SHAP_SAMPLE = None

PRIMARY_THRESHOLD = "train_oof"

PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

MODEL_KEY = "XGBoost"
PREFIX = {"pre": "pre_xgb", "during": "during_xgb"}

# =============================================================================
# SHAP
# =============================================================================


def unwrap_calibrated(model):
    """Return the booster underneath a CalibratedClassifierCV.

    The attribute was renamed from `base_estimator` to `estimator` in
    scikit-learn 1.2, so both are tried. Only the first of the calibrated
    members is explained, which is what the notebooks did; averaging over all
    five would change the reported importances.
    """
    if isinstance(model, CalibratedClassifierCV):
        member = model.calibrated_classifiers_[0]
        return getattr(member, "estimator", None) or member.base_estimator
    return model


def compute_shap(model, X_scaled, feat_names):
    try:
        values = shap.TreeExplainer(unwrap_calibrated(model)).shap_values(X_scaled)
        if isinstance(values, list):
            values = values[-1]
    except Exception as exc:
        print(f"      [warn] SHAP unavailable: {exc}")
        return None, None
    series = (pd.Series(np.abs(values).mean(axis=0), index=feat_names)
              .sort_values(ascending=False))
    return series, values


def plot_shap_bar(series, outcome_label, top_n, path):
    top = series.head(top_n)
    n = len(top)
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.40)))
    values, labels = top.values[::-1], top.index[::-1]
    ax.barh(range(n), values, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"{outcome_label}  ·  XGBoost\nTop {top_n} features",
                 fontsize=12, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    span = values.max() if values.max() > 0 else 1
    for i, value in enumerate(values):
        ax.text(value + span * 0.01, i, f"{value:.4f}", va="center",
                fontsize=8, color="#333333")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def shap_for_result(result, X, prefix, outcome):
    """Recreate the scaled test matrix from the stored split and explain it."""
    X_te = X.loc[result["_test_index"], result["feat_names"]]
    scaler = result["scaler"]
    Xte = scaler.transform(X_te) if scaler is not None else X_te.values
    if SHAP_SAMPLE is not None and len(Xte) > SHAP_SAMPLE:
        idx = np.random.RandomState(42).choice(len(Xte), SHAP_SAMPLE,
                                               replace=False)
        Xte = Xte[idx]
    return compute_shap(result["model"], Xte, result["feat_names"])


# =============================================================================
# ONE WINDOW
# =============================================================================


def run_window(df: pd.DataFrame, window: str) -> list[dict]:
    prefix = PREFIX[window]
    print(f"\n{'#' * 74}\n#  WINDOW = {window.upper()}   prefix = {prefix}\n{'#' * 74}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    results, scalers, shap_means, shap_raw, diagnostics = {}, {}, {}, {}, []

    for outcome in OUTCOMES:
        label = OUTCOME_LABELS[outcome]
        print(f"\n  {'-' * 66}\n  {window.upper()}  |  {label}  ({outcome})")

        X, y, feat_names = build_matrix(df, window, outcome)
        print(f"      episodes {len(X):,}   positives {int(y.sum()):,} "
              f"({y.mean() * 100:.2f}%)   features {len(feat_names)}")

        estimator = XGBClassifier(eval_metric="logloss",
                                  random_state=RANDOM_STATE, verbosity=0)
        result = fit_evaluate(
            estimator, PARAM_GRID, X, y, model_name=MODEL_KEY, cv=cv,
            needs_scaling=True, calibrate=True, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
            n_bootstrap=N_BOOTSTRAP, primary_threshold=PRIMARY_THRESHOLD,
            calibration_folds=CV_FOLDS)

        results[outcome] = {MODEL_KEY: result}
        scalers[outcome] = result["scaler"]
        diagnostics.append(diagnostic_row(result, window=window, outcome=label,
                                          model=MODEL_KEY))

        series, raw = shap_for_result(result, X, prefix, outcome)
        if series is not None:
            shap_means[outcome] = series
            shap_raw[outcome] = {MODEL_KEY: raw}
            plot_shap_bar(series, label, TOP_N_SHAP,
                          os.path.join(OUT_DIR, f"{prefix}_shap_{outcome}.png"))

    save_pickle(results, os.path.join(OUT_DIR, f"{prefix}_results.pkl"))
    save_pickle(scalers, os.path.join(OUT_DIR, f"{prefix}_scalers.pkl"))
    save_pickle({"mean_abs": {o: {MODEL_KEY: s} for o, s in shap_means.items()},
                 "raw": shap_raw},
                os.path.join(OUT_DIR, f"{prefix}_shap_values.pkl"))

    write_sheets(metrics_sheets(results, OUTCOMES, OUTCOME_LABELS, MODEL_KEY),
                 os.path.join(OUT_DIR, f"{prefix}_metrics.xlsx"))

    path = os.path.join(OUT_DIR, f"{prefix}_shap_top{TOP_N_SHAP}.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for outcome, series in shap_means.items():
            top = series.head(TOP_N_SHAP).reset_index()
            top.columns = ["Feature", "Mean |SHAP|"]
            top.insert(0, "Rank", range(1, len(top) + 1))
            top.to_excel(writer, sheet_name=OUTCOME_LABELS[outcome][:28],
                         index=False)
    print(f"      SHAP XLSX -> {path}")

    return diagnostics


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML02 -- XGBoost, PRE-PREGNANCY AND ANTENATAL WINDOWS")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML02.txt")))
    warn_if_no_psutil()

    bench = Bench("ML02, XGBoost main")

    with bench.stage("Load analytic set", "pandas") as b:
        df = load_analytic_set()
        b["rows_out"] = len(df)

    diagnostics = []
    for window in WINDOWS:
        with bench.stage(f"Train and evaluate ({window})", "xgboost",
                         rows_in=len(df)) as b:
            diagnostics += run_window(df, window)
            b["rows_out"] = len(df)

    path = os.path.join(OUT_DIR, "threshold_diagnostics.csv")
    pd.DataFrame(diagnostics).to_csv(path, index=False)
    print(f"\n  threshold diagnostics -> {path}")

    # The transport carries a quantile from the out-of-fold scale to the
    # fitted one. Rho says how closely the two scales rank the same episodes.
    rho_min = min(d["spearman_rho"] for d in diagnostics)
    print(f"  lowest rank agreement across models: rho = {rho_min:.4f}")
    if rho_min < 0:
        print("  [WARNING] at least one model has a negative rank agreement. "
              "Platt scaling fits a negative slope when the base model does "
              "not discriminate on the calibration folds, which inverts the "
              "ordering. Check the AUC for that outcome: if it is at or "
              "below 0.5 the model has no signal, and neither cut-off nor "
              "any metric derived from one is interpretable.")
    elif rho_min < 0.95:
        print("  [note] the fitted and out-of-fold scales disagree more than "
              "expected for at least one model. The cut-off is still derived "
              "without reference to the test split. Describe it as the "
              "operating point at the out-of-fold predicted-positive rate, "
              "not as the Youden optimum of the calibrated scale, and report "
              "rho alongside.")

    bench.finalise(os.path.join(OUT_DIR, "benchmark_ML02.csv"), rows_out=len(df))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
