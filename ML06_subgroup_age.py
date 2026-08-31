"""
=============================================================================
 ML06 -- SUBGROUP ANALYSIS BY MATERNAL AGE
=============================================================================

Refits the primary models separately within the two maternal age strata, to
establish whether discrimination holds in the group where the outcomes are
more common.

    age_risk = 0    20 to 35 years
    age_risk = 1    under 20 or over 35 years

Twenty-four models: two windows, two strata, six outcomes.

WHAT SEPARATE MODELS DO AND DO NOT SHOW
---------------------------------------
Each stratum is split, tuned, and evaluated on its own, so a difference in
AUC between strata is not a test of interaction. It is two independent
estimates in two populations with different outcome prevalence and different
sample size, and AUC is not comparable across populations whose case mix
differs. The stratum-specific confidence intervals will overlap for most
outcomes, and where they do not, the difference is still confounded with
prevalence.

What the analysis does support is the weaker claim the manuscript makes: that
the models do not fail in the smaller stratum. `age_risk` is excluded as a
predictor within each stratum, since it is constant there.

The higher-risk stratum is about 18 percent of the cohort, so for placental
abruption it holds roughly 120 events in total and about 36 in the test
split. Metrics for that combination will be unstable, and the bootstrap
interval will be wide. The realised number of resamples that contained both
classes is reported, so a metric computed from a degenerate resample set is
identifiable rather than silently reported.

INPUT
-----
    pregnancy by episode cleaned.csv     read through ML01

OUTPUT (in OUT_DIR)
-------------------
    pre_xgb_subgroup_results.pkl         {age_risk_0: {...}, age_risk_1: {...}}
    during_xgb_subgroup_results.pkl
    pre_xgb_subgroup_metrics.xlsx        one sheet group per stratum
    during_xgb_subgroup_metrics.xlsx
    subgroup_sizes.csv                   n and events per stratum and outcome
    threshold_diagnostics_subgroup.csv
    benchmark_ML06.csv
    environment_ML06.txt

USAGE
-----
    python ML06_subgroup_age.py
=============================================================================
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ML01_prepare_analytic_set import (OUTCOME_LABELS, OUTCOMES, WINDOWS,
                                       build_matrix, load_analytic_set)
from ml_core import diagnostic_row, fit_evaluate, metrics_sheets
from ml_utils import (Bench, environment_report, save_pickle,
                      warn_if_no_psutil)

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR = r"./output_ML06"

RANDOM_STATE = 123
TEST_SIZE = 0.30
CV_FOLDS = 5
N_BOOTSTRAP = 1000
N_JOBS = -1
PRIMARY_THRESHOLD = "train_oof"

AGE_RISK_LEVELS = [0, 1]
AGE_RISK_LABELS = {0: "20-35 years", 1: "under 20 or over 35 years"}

# A stratum with fewer events than this for a given outcome is skipped rather
# than fitted. Below roughly one event per feature the split can leave a fold
# with no positive case, and the reported metric would describe the failure
# rather than the model.
MIN_EVENTS = 50

PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

MODEL_KEY = "XGBoost"
PREFIX = {"pre": "pre_xgb_subgroup", "during": "during_xgb_subgroup"}


# =============================================================================
# ONE WINDOW
# =============================================================================


def run_window(df: pd.DataFrame, window: str) -> tuple[dict, list, list]:
    prefix = PREFIX[window]
    print(f"\n{'#' * 74}\n#  WINDOW = {window.upper()}   prefix = {prefix}\n{'#' * 74}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    by_stratum, diagnostics, sizes = {}, [], []

    for level in AGE_RISK_LEVELS:
        stratum = df[df["age_risk"] == level]
        key = f"age_risk_{level}"
        print(f"\n  {'=' * 66}\n  stratum {key}  ({AGE_RISK_LABELS[level]})   "
              f"{len(stratum):,} episodes\n  {'=' * 66}")

        results = {}
        for outcome in OUTCOMES:
            label = OUTCOME_LABELS[outcome]
            X, y, feat_names = build_matrix(stratum, window, outcome)

            # age_risk is constant within the stratum and carries no
            # information there.
            if "age_risk" in X.columns:
                X = X.drop(columns=["age_risk"])
                feat_names = list(X.columns)

            events = int(y.sum())
            sizes.append({"window": window, "stratum": key,
                          "stratum_label": AGE_RISK_LABELS[level],
                          "outcome": label, "n": len(y), "events": events,
                          "prevalence": round(y.mean(), 6),
                          "fitted": events >= MIN_EVENTS})

            print(f"\n  {'-' * 62}\n  {key}  |  {label}   "
                  f"{len(y):,} episodes, {events:,} events "
                  f"({y.mean() * 100:.2f}%)")

            if events < MIN_EVENTS:
                print(f"      skipped: fewer than {MIN_EVENTS} events")
                continue

            estimator = XGBClassifier(eval_metric="logloss",
                                      random_state=RANDOM_STATE, verbosity=0)
            result = fit_evaluate(
                estimator, PARAM_GRID, X, y, model_name=MODEL_KEY, cv=cv,
                needs_scaling=True, calibrate=True, test_size=TEST_SIZE,
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
                n_bootstrap=N_BOOTSTRAP, primary_threshold=PRIMARY_THRESHOLD,
                calibration_folds=CV_FOLDS)

            results[outcome] = {MODEL_KEY: result}
            diagnostics.append(diagnostic_row(
                result, window=window, stratum=key, outcome=label,
                model=MODEL_KEY))

        by_stratum[key] = results

    save_pickle(by_stratum, os.path.join(OUT_DIR, f"{prefix}_results.pkl"))

    path = os.path.join(OUT_DIR, f"{prefix}_metrics.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for key, results in by_stratum.items():
            fitted = [oc for oc in OUTCOMES if oc in results]
            if not fitted:
                continue
            sheets = metrics_sheets(results, fitted, OUTCOME_LABELS, MODEL_KEY)
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=f"{key}_{name}"[:31],
                               index=False)
    print(f"      XLSX -> {path}")

    return by_stratum, diagnostics, sizes


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML06 -- SUBGROUP ANALYSIS BY MATERNAL AGE")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML06.txt")))
    warn_if_no_psutil()

    bench = Bench("ML06, age subgroups")

    with bench.stage("Load analytic set", "pandas") as b:
        df = load_analytic_set()
        b["rows_out"] = len(df)

    diagnostics, sizes = [], []
    for window in WINDOWS:
        with bench.stage(f"Train and evaluate ({window})", "xgboost",
                         rows_in=len(df)) as b:
            _, diag, size = run_window(df, window)
            diagnostics += diag
            sizes += size
            b["rows_out"] = len(df)

    pd.DataFrame(sizes).to_csv(os.path.join(OUT_DIR, "subgroup_sizes.csv"),
                               index=False)
    pd.DataFrame(diagnostics).to_csv(
        os.path.join(OUT_DIR, "threshold_diagnostics_subgroup.csv"),
        index=False)

    skipped = [s for s in sizes if not s["fitted"]]
    if skipped:
        print(f"\n  {len(skipped)} stratum-outcome combinations were not "
              f"fitted (fewer than {MIN_EVENTS} events):")
        for s in skipped:
            print(f"      {s['window']:<7} {s['stratum']:<11} "
                  f"{s['outcome']:<13} {s['events']:>5} events")

    bench.finalise(os.path.join(OUT_DIR, "benchmark_ML06.csv"), rows_out=len(df))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
