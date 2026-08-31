"""
=============================================================================
 ML04 -- SENSITIVITY ANALYSIS: SMOTE
=============================================================================

Repeats the primary XGBoost models with synthetic minority oversampling
applied to the training split, to establish whether the incremental value of
antenatal diagnoses depends on how class imbalance is handled. Outcome
prevalence ranges from 11.5 percent to 0.12 percent, so the question is
reasonable to ask; it is asked as a sensitivity analysis rather than as the
primary specification because oversampling distorts the predicted
probabilities and the manuscript reports calibration.

    SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42)

Verbatim from the analysis notebooks: the minority class is oversampled to 30
percent of the majority, not to parity.

WHERE THE RESAMPLING HAPPENS
----------------------------
Two placement errors are easy to make with SMOTE, and the analysis notebooks
made both.

The first is the fold boundary. The notebooks applied SMOTE once to the whole
training split and then ran the grid search over folds of the resampled data.
A synthetic point is interpolated between real minority neighbours, so a
point derived in part from an observation held out in a given fold appears in
the folds used to predict it, and hyperparameter selection is optimistic.

The second is the distribution the calibrator is fitted on, and it matters
more here because the manuscript reports calibration. Platt scaling fitted on
resampled data learns to map scores onto the resampled prevalence. At
sampling_strategy=0.3 the minority class is 23 percent of the resampled
training set; for placental abruption the true prevalence is 0.12 percent.
The resulting probabilities are inflated by roughly two orders of magnitude,
and the Brier score and calibration slope describe that error rather than
anything about the model.

Both are avoided by putting the resampler inside the estimator:

    GridSearchCV(Pipeline([SMOTE, XGBoost])).fit(X_train, y_train)
        the grid search resamples within each fold

    CalibratedClassifierCV(selected_pipeline, cv=5).fit(X_train, y_train)
        the base model is refitted with SMOTE inside it, and the sigmoid is
        fitted on held-out folds whose prevalence is the real one

The training split is never resampled outside a fold.

    SMOTE_INSIDE_FOLDS = True    the above. Default.
    SMOTE_INSIDE_FOLDS = False   reproduces the notebooks: SMOTE applied once
                                 to the training split, grid search and
                                 calibration both fitted on resampled data.
                                 Retained so the published specification can
                                 be regenerated, not because it is defensible.

The two settings select different hyperparameters and produce different
numbers. The default is the correct placement.

The cut-off derivation is unaffected by the switch: out-of-fold predictions
always resample inside the fold, because a cut-off read off synthetic class
balance would not describe the population the model is applied to. See
`ml_core._oof_estimator`.

WHAT THIS ANALYSIS IS FOR
-------------------------
Oversampling does not improve the ranking of patients by risk, and it moves
predicted probabilities away from observed frequencies. The expected result
is therefore an AUC close to the primary analysis with a worse Brier score
and a calibration slope further from one. That is a finding worth reporting
in a paper whose claims rest on calibration and net benefit, not a failed
experiment. The literature on imbalance corrections in clinical prediction
reaches the same conclusion; van den Goorbergh and colleagues (JAMIA, 2022)
is the reference usually cited for it, though the citation should be checked
before it goes into the manuscript.

INPUT
-----
    pregnancy by episode cleaned.csv     read through ML01

OUTPUT (in OUT_DIR)
-------------------
    pre_xgb_smote_results.pkl
    during_xgb_smote_results.pkl
    pre_xgb_smote_metrics.xlsx           Summary, Numeric, By_threshold
    during_xgb_smote_metrics.xlsx
    threshold_diagnostics_smote.csv
    benchmark_ML04.csv
    environment_ML04.txt

USAGE
-----
    python ML04_train_xgb_smote.py
=============================================================================
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
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

OUT_DIR = r"./output_ML04"

RANDOM_STATE = 123
TEST_SIZE = 0.30
CV_FOLDS = 5
N_BOOTSTRAP = 1000
N_JOBS = -1
PRIMARY_THRESHOLD = "train_oof"

SMOTE_CONFIG = {"sampling_strategy": 0.3, "k_neighbors": 5,
                "random_state": 42}

# See the header. True is the correct placement; False reproduces the
# published specification, in which the grid search and the calibrator were
# both fitted on resampled data.
SMOTE_INSIDE_FOLDS = True

PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

MODEL_KEY = "XGBoost"
PREFIX = {"pre": "pre_xgb_smote", "during": "during_xgb_smote"}


# =============================================================================
# ONE WINDOW
# =============================================================================


def run_window(df: pd.DataFrame, window: str) -> list[dict]:
    prefix = PREFIX[window]
    print(f"\n{'#' * 74}\n#  WINDOW = {window.upper()}   prefix = {prefix}")
    print(f"#  SMOTE {SMOTE_CONFIG}   inside folds: {SMOTE_INSIDE_FOLDS}")
    print("#" * 74)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    results, diagnostics = {}, []

    for outcome in OUTCOMES:
        label = OUTCOME_LABELS[outcome]
        print(f"\n  {'-' * 66}\n  {window.upper()}  |  {label}  ({outcome})")

        X, y, feat_names = build_matrix(df, window, outcome)
        print(f"      episodes {len(X):,}   positives {int(y.sum()):,} "
              f"({y.mean() * 100:.2f}%)   features {len(feat_names)}")

        estimator = XGBClassifier(eval_metric="logloss",
                                  random_state=RANDOM_STATE, verbosity=0)
        grid = PARAM_GRID
        if SMOTE_INSIDE_FOLDS:
            from imblearn.pipeline import Pipeline as ImbPipeline
            estimator = ImbPipeline([("resample", SMOTE(**SMOTE_CONFIG)),
                                     ("clf", estimator)])
            grid = {f"clf__{k}": v for k, v in PARAM_GRID.items()}
            resampler = None          # already inside the estimator
        else:
            resampler = SMOTE(**SMOTE_CONFIG)

        result = fit_evaluate(
            estimator, grid, X, y, model_name=MODEL_KEY, cv=cv,
            needs_scaling=True, calibrate=True, resampler=resampler,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, n_jobs=N_JOBS,
            n_bootstrap=N_BOOTSTRAP, primary_threshold=PRIMARY_THRESHOLD,
            calibration_folds=CV_FOLDS)

        result["smote_config"] = dict(SMOTE_CONFIG)
        result["smote_inside_folds"] = SMOTE_INSIDE_FOLDS
        results[outcome] = {MODEL_KEY: result}
        diagnostics.append(diagnostic_row(result, window=window, outcome=label,
                                          model=f"{MODEL_KEY}+SMOTE"))

    save_pickle(results, os.path.join(OUT_DIR, f"{prefix}_results.pkl"))
    write_sheets(metrics_sheets(results, OUTCOMES, OUTCOME_LABELS, MODEL_KEY),
                 os.path.join(OUT_DIR, f"{prefix}_metrics.xlsx"))
    return diagnostics


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML04 -- SENSITIVITY ANALYSIS: SMOTE")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML04.txt")))
    warn_if_no_psutil()

    bench = Bench("ML04, XGBoost + SMOTE")

    with bench.stage("Load analytic set", "pandas") as b:
        df = load_analytic_set()
        b["rows_out"] = len(df)

    diagnostics = []
    for window in WINDOWS:
        with bench.stage(f"Train and evaluate ({window})", "xgboost + SMOTE",
                         rows_in=len(df)) as b:
            diagnostics += run_window(df, window)
            b["rows_out"] = len(df)

    pd.DataFrame(diagnostics).to_csv(
        os.path.join(OUT_DIR, "threshold_diagnostics_smote.csv"), index=False)

    bench.finalise(os.path.join(OUT_DIR, "benchmark_ML04.csv"), rows_out=len(df))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
