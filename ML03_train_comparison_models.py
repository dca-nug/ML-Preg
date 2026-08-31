"""
=============================================================================
 ML03 -- COMPARISON LEARNERS
=============================================================================

Fits the five learners the primary model is compared against, over the same
outcomes, the same windows, the same split, and the same folds. The comparison
is only informative if nothing else differs, so everything except the
estimator, its grid, and two per-learner switches comes from `ml_core`.

    learner            standardised   calibrated
    -----------------------------------------------
    LogisticRegression      yes           no
    ElasticNet              yes           no
    RandomForest            no            yes
    LightGBM                no            yes
    CatBoost                no            yes

Standardisation is applied to the two penalised linear models, whose
coefficients are scale-dependent, and withheld from the tree ensembles, which
are invariant to monotone rescaling of a feature. Platt scaling is applied to
the ensembles and withheld from the linear models, which produce calibrated
probabilities by construction under a correctly specified model. Both
conventions are carried over from the analysis notebooks unchanged.

These five scripts were previously five files differing in about a dozen
lines. They are one file with a command-line switch, because five copies of a
procedure drift apart and a correction applied to four of them is worse than
no correction.

INPUT
-----
    pregnancy by episode cleaned.csv     read through ML01

OUTPUT (in OUT_DIR, one set per learner)
----------------------------------------
    pre_<key>_results.pkl                models, predictions, metrics
    during_<key>_results.pkl
    pre_<key>_metrics.xlsx               Summary, Numeric, By_threshold
    during_<key>_metrics.xlsx
    threshold_diagnostics_<key>.csv
    benchmark_ML03_<key>.csv
    environment_ML03.txt

where <key> is lr, elasticnet, rf, lgbm, or catboost. Filenames match the
notebooks so ML10 reads them unchanged.

USAGE
-----
    python ML03_train_comparison_models.py                 all five
    python ML03_train_comparison_models.py --model lgbm    one of them
    python ML03_train_comparison_models.py --model lr rf

Fitting all five over both windows and six outcomes is the longest single job
in the repository. Running them one at a time is the practical way to use it.
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold

from ML01_prepare_analytic_set import (OUTCOME_LABELS, OUTCOMES, WINDOWS,
                                       build_matrix, load_analytic_set)
from ml_core import diagnostic_row, fit_evaluate, metrics_sheets, write_sheets
from ml_utils import (Bench, environment_report, save_pickle,
                      warn_if_no_psutil)

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR = r"./output_ML03"

RANDOM_STATE = 123
TEST_SIZE = 0.30
CV_FOLDS = 5
N_BOOTSTRAP = 1000
N_JOBS = -1
PRIMARY_THRESHOLD = "train_oof"


def _logistic():
    from sklearn.linear_model import LogisticRegression
    return (LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,
                               solver="lbfgs"),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]})


def _elasticnet():
    from sklearn.linear_model import LogisticRegression
    return (LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,
                               solver="saga", penalty="elasticnet"),
            {"C": [0.001, 0.01, 0.1, 1], "l1_ratio": [0.1, 0.5, 0.9]})


def _random_forest():
    from sklearn.ensemble import RandomForestClassifier
    return (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20],
             "min_samples_split": [2, 5]})


def _lightgbm():
    from lightgbm import LGBMClassifier
    return (LGBMClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS,
                           verbose=-1),
            {"n_estimators": [100, 200], "max_depth": [3, 5, 7],
             "learning_rate": [0.01, 0.1], "num_leaves": [31, 63]})


def _catboost():
    from catboost import CatBoostClassifier
    return (CatBoostClassifier(random_state=RANDOM_STATE, verbose=0,
                               allow_writing_files=False),
            {"iterations": [100, 200], "depth": [4, 6, 8],
             "learning_rate": [0.01, 0.1]})


# key -> (display name, builder, needs_scaling, calibrate, filename stem)
LEARNERS = {
    "lr":         ("LogisticRegression", _logistic,      True,  False, "lr"),
    "elasticnet": ("ElasticNet",         _elasticnet,    True,  False, "elasticnet"),
    "rf":         ("RandomForest",       _random_forest, False, True,  "rf"),
    "lgbm":       ("LightGBM",           _lightgbm,      False, True,  "lgbm"),
    "catboost":   ("CatBoost",           _catboost,      False, True,  "catboost"),
}


# =============================================================================
# ONE LEARNER
# =============================================================================


def run_learner(df: pd.DataFrame, key: str) -> None:
    model_key, builder, needs_scaling, calibrate, stem = LEARNERS[key]

    print(f"\n{'=' * 74}\n  {model_key}   "
          f"(standardised: {needs_scaling}, calibrated: {calibrate})\n{'=' * 74}")

    bench = Bench(f"ML03, {model_key}")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    diagnostics = []

    for window in WINDOWS:
        prefix = f"{window}_{stem}"
        with bench.stage(f"Train and evaluate ({window})", model_key,
                         rows_in=len(df)) as b:
            print(f"\n{'#' * 74}\n#  WINDOW = {window.upper()}   "
                  f"prefix = {prefix}\n{'#' * 74}")

            results = {}
            for outcome in OUTCOMES:
                label = OUTCOME_LABELS[outcome]
                print(f"\n  {'-' * 66}\n  {window.upper()}  |  {label}")

                X, y, feat_names = build_matrix(df, window, outcome)
                print(f"      episodes {len(X):,}   positives {int(y.sum()):,} "
                      f"({y.mean() * 100:.2f}%)   features {len(feat_names)}")

                estimator, grid = builder()
                result = fit_evaluate(
                    estimator, grid, X, y, model_name=model_key, cv=cv,
                    needs_scaling=needs_scaling, calibrate=calibrate,
                    test_size=TEST_SIZE, random_state=RANDOM_STATE,
                    n_jobs=N_JOBS, n_bootstrap=N_BOOTSTRAP,
                    primary_threshold=PRIMARY_THRESHOLD,
                    calibration_folds=CV_FOLDS)

                results[outcome] = {model_key: result}
                diagnostics.append(diagnostic_row(
                    result, window=window, outcome=label, model=model_key))

            save_pickle(results, os.path.join(OUT_DIR, f"{prefix}_results.pkl"))
            write_sheets(
                metrics_sheets(results, OUTCOMES, OUTCOME_LABELS, model_key),
                os.path.join(OUT_DIR, f"{prefix}_metrics.xlsx"))
            b["rows_out"] = len(df)

    pd.DataFrame(diagnostics).to_csv(
        os.path.join(OUT_DIR, f"threshold_diagnostics_{stem}.csv"), index=False)
    bench.finalise(os.path.join(OUT_DIR, f"benchmark_ML03_{stem}.csv"),
                   rows_out=len(df))


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the comparison learners.")
    parser.add_argument("--model", nargs="+", choices=sorted(LEARNERS),
                        default=sorted(LEARNERS),
                        help="which learners to fit (default: all five)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML03 -- COMPARISON LEARNERS")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML03.txt")))
    warn_if_no_psutil()

    df = load_analytic_set()
    for key in args.model:
        run_learner(df, key)

    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
