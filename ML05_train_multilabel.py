"""
=============================================================================
 ML05 -- SENSITIVITY ANALYSIS: MULTILABEL SPECIFICATION
=============================================================================

Refits the six outcomes as a multilabel problem, with one shared feature
matrix per window instead of an outcome-specific one, to establish whether
the incremental value of antenatal diagnoses depends on modelling the
outcomes separately.

WHAT THIS SPECIFICATION ACTUALLY CHANGES
----------------------------------------
`MultiOutputClassifier` fits one independent binary classifier per target.
The six models share a feature matrix and a training split; they do not share
parameters, and no target is used as an input to another. The specification
therefore does not model dependence between outcomes.

This matters for how the analysis is described. A manuscript sentence of the
form "multilabel models that simultaneously predicted all six complications
to assess whether capturing inter-outcome dependencies would alter the
results" does not describe what this script does. Two ways to make the two
agree:

    describe what is fitted   the multilabel arm tests sensitivity to the
                              per-outcome feature exclusions and to the
                              stratification variable, not to inter-outcome
                              dependence. No code change; the Methods
                              sentence is rewritten.

    fit what is described     `ClassifierChain` feeds each predicted outcome
                              forward as a feature for the next, which does
                              model dependence. This is a different analysis:
                              the result depends on the chain order, and a
                              predicted outcome used as an input is not
                              available at prediction time in the way a
                              diagnosis is. Set MODEL_FORM = "chain".

The default is "multioutput", which reproduces the analysis notebooks.

WHAT DIFFERS FROM THE PRIMARY ANALYSIS
--------------------------------------
    feature matrix    shared across the six outcomes, so the per-outcome
                      exclusions in `IGNORED_COLS_MAP` do not apply. In the
                      antenatal window the shared exclusion list removes the
                      same columns for every outcome.

    stratification    the split is stratified on preeclampsia alone, since a
                      single split has to serve six targets. Rare outcomes
                      are therefore not balanced across the split.

    grid              larger than the primary grid, verbatim from the
                      notebooks: 48 candidate configurations.

Everything else - folds, calibration, cut-offs, bootstrap - matches ML02.

INPUT
-----
    pregnancy by episode cleaned.csv     read through ML01

OUTPUT (in OUT_DIR)
-------------------
    pre_xgb_multi_results.pkl
    during_xgb_multi_results.pkl
    pre_xgb_multi_scaler.pkl
    during_xgb_multi_scaler.pkl
    pre_xgb_multi_metrics.xlsx           Summary, Numeric, By_threshold
    during_xgb_multi_metrics.xlsx
    multilabel_features.csv              the shared feature list per window
    threshold_diagnostics_multi.csv
    benchmark_ML05.csv
    environment_ML05.txt

USAGE
-----
    python ML05_train_multilabel.py
=============================================================================
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ML01_prepare_analytic_set import (DEMOGRAPHIC_FEATURES, DROP_ALWAYS,
                                       OUTCOME_LABELS, OUTCOMES, WINDOWS,
                                       IGNORED_COLS_MAP, load_analytic_set)
from ml_core import THRESHOLD_NAMES, metrics_sheets, write_sheets
from ml_utils import (Bench, bootstrap_metrics, environment_report,
                      rank_agreement, save_pickle, threshold_metrics,
                      transport_threshold, warn_if_no_psutil,
                      youden_threshold)

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR = r"./output_ML05"

RANDOM_STATE = 123
TEST_SIZE = 0.30
CV_FOLDS = 5
N_BOOTSTRAP = 1000
N_JOBS = -1
PRIMARY_THRESHOLD = "train_oof"

# "multioutput" reproduces the notebooks. "chain" models dependence between
# outcomes; see the header before switching.
MODEL_FORM = "multioutput"

# The split has to serve six targets, so it is stratified on one of them.
STRATIFY_ON = "c_preecl"

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

MODEL_KEY = "XGBoost"
PREFIX = {"pre": "pre_xgb_multi", "during": "during_xgb_multi"}


# =============================================================================
# SHARED FEATURE SET
# =============================================================================


def shared_features(df: pd.DataFrame, window: str) -> list[str]:
    """One feature list per window, shared by all six targets.

    The per-outcome exclusions do not apply, so the antenatal list is the
    union of what every outcome would have excluded. Reusing
    `IGNORED_COLS_MAP` rather than restating the columns keeps the two
    specifications tied to one source.
    """
    if window == "pre":
        drop = set(DROP_ALWAYS)
        drop |= {c for c in df.columns if c.startswith(("a_", "c_"))}
    else:
        drop = set(DROP_ALWAYS)
        for cols in IGNORED_COLS_MAP.values():
            drop |= set(cols)
        drop |= {c for c in df.columns if c.startswith("a_")}
        drop |= set(OUTCOMES)

    feats = [c for c in df.columns if c not in drop]

    unexpected = [c for c in feats if not c.startswith(("b_", "c_"))
                  and c not in DEMOGRAPHIC_FEATURES]
    if unexpected:
        raise SystemExit(f"[FATAL] {window}: columns with no rule: {unexpected}")
    leaked = [c for c in feats if c.startswith("a_") or c in OUTCOMES]
    if leaked:
        raise SystemExit(f"[FATAL] {window}: leakage: {leaked}")
    if window == "pre" and any(c.startswith("c_") for c in feats):
        raise SystemExit(f"[FATAL] pre: antenatal leakage")
    return feats


# =============================================================================
# ONE WINDOW
# =============================================================================


def build_multilabel():
    base = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE,
                         verbosity=0)
    if MODEL_FORM == "chain":
        model = ClassifierChain(base, order="random", random_state=RANDOM_STATE)
        grid = {f"base_estimator__{k}": v for k, v in PARAM_GRID.items()}
    else:
        model = MultiOutputClassifier(base)
        grid = {f"estimator__{k}": v for k, v in PARAM_GRID.items()}
    return model, grid


def run_window(df: pd.DataFrame, window: str, cv) -> tuple[dict, list, list]:
    prefix = PREFIX[window]
    print(f"\n{'#' * 74}\n#  WINDOW = {window.upper()}   prefix = {prefix}   "
          f"form = {MODEL_FORM}\n{'#' * 74}")

    feat_names = shared_features(df, window)
    X = df[feat_names]
    Y = df[OUTCOMES]
    print(f"  shared feature matrix: {len(feat_names)} features")

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=Y[STRATIFY_ON])

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)

    model, grid = build_multilabel()
    print(f"  grid search over {np.prod([len(v) for v in PARAM_GRID.values()])} "
          f"configurations, {CV_FOLDS} folds")
    search = GridSearchCV(model, grid, cv=CV_FOLDS,
                          scoring="roc_auc_ovr_weighted", n_jobs=N_JOBS,
                          refit=True)
    search.fit(Xtr, Y_tr)
    best = search.best_estimator_
    print(f"  best params {search.best_params_}")
    print(f"  best CV score {search.best_score_:.4f}")

    results, diagnostics = {}, []

    for idx, outcome in enumerate(OUTCOMES):
        label = OUTCOME_LABELS[outcome]
        print(f"\n  {'-' * 66}\n  {window.upper()}  |  {label}")

        y_tr = Y_tr.iloc[:, idx].values
        y_te = Y_te.iloc[:, idx].values

        # `MultiOutputClassifier` and `ClassifierChain` both hold one fitted
        # estimator per target. The cut-off is derived per target from
        # out-of-fold predictions of that estimator, which matches how the
        # per-outcome models in ML02 derive theirs.
        member = clone(best.estimators_[idx])
        oof_raw = cross_val_predict(member, Xtr, y_tr, cv=cv,
                                    method="predict_proba",
                                    n_jobs=N_JOBS)[:, 1]
        thr_raw = youden_threshold(y_tr, oof_raw)

        calibrated = CalibratedClassifierCV(best.estimators_[idx],
                                            method="sigmoid", cv=CV_FOLDS)
        calibrated.fit(Xtr, y_tr)

        p_tr = calibrated.predict_proba(Xtr)[:, 1]
        p_te = calibrated.predict_proba(Xte)[:, 1]

        thr_train_oof, alert_rate = transport_threshold(oof_raw, thr_raw, p_tr)
        thr_test_reopt = youden_threshold(y_te, p_te)
        thresholds = {"train_oof": thr_train_oof, "test_reopt": thr_test_reopt}
        rho = rank_agreement(oof_raw, p_tr)

        ranking = {"auc": roc_auc_score(y_te, p_te),
                   "auprc": average_precision_score(y_te, p_te),
                   "brier": brier_score_loss(y_te, p_te)}
        at_cutoff = {name: threshold_metrics(y_te, p_te, thr)
                     for name, thr in thresholds.items()}
        ci = bootstrap_metrics(y_te, p_te, thresholds, n_bootstraps=N_BOOTSTRAP)
        primary = at_cutoff[PRIMARY_THRESHOLD]

        print(f"      AUC {ranking['auc']:.3f} "
              f"[{ci[PRIMARY_THRESHOLD]['auc']['lower']:.3f}-"
              f"{ci[PRIMARY_THRESHOLD]['auc']['upper']:.3f}]   "
              f"AUPRC {ranking['auprc']:.3f}   Brier {ranking['brier']:.4f}   "
              f"Sens {primary['sens']:.3f}   rho {rho:.4f}")

        results[outcome] = {MODEL_KEY: {
            "model": calibrated, "scaler": scaler, "feat_names": feat_names,
            "best_params": search.best_params_,
            "probas": p_te, "y_test": y_te,
            "auc": ranking["auc"], "auprc": ranking["auprc"],
            "brier": ranking["brier"],
            "f1": primary["f1"], "acc": primary["acc"], "sens": primary["sens"],
            "spec": primary["spec"], "ppv": primary["ppv"], "npv": primary["npv"],
            "threshold": thresholds[PRIMARY_THRESHOLD],
            "ci": ci[PRIMARY_THRESHOLD],
            "model_name": MODEL_KEY, "model_form": MODEL_FORM,
            "threshold_source": PRIMARY_THRESHOLD,
            "thresholds": thresholds,
            "metrics_by_threshold": at_cutoff, "ci_by_threshold": ci,
            "threshold_raw_scale": thr_raw, "alert_rate": alert_rate,
            "rank_agreement_rho": rho,
            "n_train": len(y_tr), "n_test": len(y_te),
        }}

        row = {"window": window, "outcome": label, "model": "XGBoost multilabel",
               "threshold_raw_oof": round(thr_raw, 6),
               "alert_rate": round(alert_rate, 6),
               "threshold_train_oof": round(thr_train_oof, 6),
               "threshold_test_reopt": round(thr_test_reopt, 6),
               "spearman_rho": round(rho, 6),
               "n_bootstrap_used": ci[PRIMARY_THRESHOLD]["n_resamples"]}
        for metric in ("sens", "spec", "ppv", "f1"):
            for name in THRESHOLD_NAMES:
                row[f"{metric}_{name}"] = round(at_cutoff[name][metric], 4)
        diagnostics.append(row)

    save_pickle(results, os.path.join(OUT_DIR, f"{prefix}_results.pkl"))
    save_pickle(scaler, os.path.join(OUT_DIR, f"{prefix}_scaler.pkl"))
    write_sheets(metrics_sheets(results, OUTCOMES, OUTCOME_LABELS, MODEL_KEY),
                 os.path.join(OUT_DIR, f"{prefix}_metrics.xlsx"))

    features = [{"window": window, "feature": f} for f in feat_names]
    return results, diagnostics, features


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML05 -- SENSITIVITY ANALYSIS: MULTILABEL SPECIFICATION")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML05.txt")))
    warn_if_no_psutil()

    if MODEL_FORM not in ("multioutput", "chain"):
        raise SystemExit(f"[FATAL] unknown MODEL_FORM: {MODEL_FORM}")

    bench = Bench("ML05, multilabel")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    with bench.stage("Load analytic set", "pandas") as b:
        df = load_analytic_set()
        b["rows_out"] = len(df)

    diagnostics, features = [], []
    for window in WINDOWS:
        with bench.stage(f"Train and evaluate ({window})", "xgboost multilabel",
                         rows_in=len(df)) as b:
            _, diag, feats = run_window(df, window, cv)
            diagnostics += diag
            features += feats
            b["rows_out"] = len(df)

    pd.DataFrame(features).to_csv(
        os.path.join(OUT_DIR, "multilabel_features.csv"), index=False)
    pd.DataFrame(diagnostics).to_csv(
        os.path.join(OUT_DIR, "threshold_diagnostics_multi.csv"), index=False)

    bench.finalise(os.path.join(OUT_DIR, "benchmark_ML05.csv"), rows_out=len(df))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
