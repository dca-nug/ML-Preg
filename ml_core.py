"""
=============================================================================
 SHARED TRAINING AND EVALUATION ROUTINE
=============================================================================

One definition of how a model is fitted and scored, imported by ML02 through
ML06. The alternative, which the analysis notebooks used, is a copy of the
procedure in each script; five copies drift, and a correction applied to four
of them is worse than no correction at all.

Everything that differs between scripts is a parameter: the estimator, the
grid, whether the features are standardised, whether the probabilities are
calibrated, and whether the training split is resampled. Everything that does
not differ - the split, the folds, the cut-off derivation, the bootstrap - is
fixed here.

THE PROCEDURE
-------------
    1  stratified split, seed and proportion fixed by the caller
    2  StandardScaler fitted on the training split, if the learner needs it
    3  resampling of the training split, if the caller supplies a resampler
    4  grid search, stratified k-fold, AUC
    5  out-of-fold predictions from the selected configuration
    6  Platt scaling on the training split, if the caller asks for it
    7  cut-off derived from step 5, transported to the reported scale
    8  metrics on the test split, percentile bootstrap CIs

WHICH DATA EACH STEP SEES
-------------------------
Steps 4 and 6 see the resampled training split when a resampler is supplied.
Step 5 does not: out-of-fold predictions are taken on the original training
split, with resampling applied inside each fold, because a cut-off derived
from synthetic majority-minority balance would not describe the population
the model is applied to. Steps 7 and 8 see original data only.

The test split is untouched until step 8. Nothing before it - not the scaler,
not the grid search, not the cut-off - is fitted on data the model is scored
against.
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, cross_val_predict,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

from ml_utils import (bootstrap_metrics, rank_agreement, threshold_metrics,
                      transport_threshold, youden_threshold)

THRESHOLD_NAMES = ("train_oof", "test_reopt")


def _oof_estimator(estimator, resampler):
    """The object whose out-of-fold predictions define the cut-off.

    With a resampler this is an imbalanced-learn pipeline, so that resampling
    happens inside each fold rather than once over the whole training split.
    Resampling before the fold split would place synthetic points derived
    from a held-out observation into the fold used to predict it.
    """
    if resampler is None:
        return clone(estimator)
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([("resample", clone(resampler)),
                        ("clf", clone(estimator))])


def fit_evaluate(estimator, param_grid, X, y, *, model_name, cv,
                 needs_scaling=True, calibrate=True, resampler=None,
                 test_size=0.30, random_state=123, n_jobs=-1,
                 n_bootstrap=1000, primary_threshold="train_oof",
                 calibration_folds=5, verbose=True) -> dict:
    """Fit one binary classifier and score it on a held-out split.

    Returns a dictionary whose top-level keys reproduce the schema written by
    the analysis notebooks, so that the extraction scripts read it unchanged,
    with the additional cut-off information nested beneath.
    """
    feat_names = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    y_tr_arr, y_te_arr = y_tr.values, y_te.values

    if needs_scaling:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_tr)
        Xte = scaler.transform(X_te)
    else:
        scaler = None
        Xtr, Xte = X_tr.values, X_te.values

    if resampler is not None:
        Xfit, yfit = clone(resampler).fit_resample(Xtr, y_tr_arr)
        if verbose:
            print(f"      resampled training split {len(y_tr_arr):,} -> "
                  f"{len(yfit):,}  (events {int(y_tr_arr.sum()):,} -> "
                  f"{int(yfit.sum()):,})")
    else:
        Xfit, yfit = Xtr, y_tr_arr

    search = GridSearchCV(estimator, param_grid, cv=cv, scoring="roc_auc",
                          n_jobs=n_jobs, refit=True)
    search.fit(Xfit, yfit)
    best = search.best_estimator_
    if verbose:
        print(f"      best params {search.best_params_}")

    # --- cut-off, derived without touching the test split --------------------
    oof_raw = cross_val_predict(_oof_estimator(best, resampler), Xtr, y_tr_arr,
                                cv=cv, method="predict_proba",
                                n_jobs=n_jobs)[:, 1]
    thr_raw = youden_threshold(y_tr_arr, oof_raw)

    if calibrate:
        final = CalibratedClassifierCV(best, method="sigmoid",
                                       cv=calibration_folds)
        final.fit(Xfit, yfit)
    else:
        final = best

    p_tr = final.predict_proba(Xtr)[:, 1]
    p_te = final.predict_proba(Xte)[:, 1]

    thr_train_oof, alert_rate = transport_threshold(oof_raw, thr_raw, p_tr)
    thr_test_reopt = youden_threshold(y_te_arr, p_te)
    thresholds = {"train_oof": thr_train_oof, "test_reopt": thr_test_reopt}

    # Rank agreement between the two scales the transport bridges: the
    # out-of-fold scores the cut-off is read from, and the fitted
    # probabilities its quantile is applied to. This is the assumption the
    # transport makes, stated as a number.
    #
    # It is deliberately not computed against `best.predict_proba(Xtr)`. That
    # vector is in-sample for a model that may have memorised the training
    # split, so a low correlation would measure overfitting rather than
    # anything about the transport.
    rho = rank_agreement(oof_raw, p_tr)

    if verbose:
        print(f"      cut-off train_oof {thr_train_oof:.4f} "
              f"(alert rate {alert_rate * 100:.2f}%)   "
              f"test_reopt {thr_test_reopt:.4f}   rho {rho:.4f}")

    # --- evaluation ----------------------------------------------------------
    ranking = {"auc": roc_auc_score(y_te_arr, p_te),
               "auprc": average_precision_score(y_te_arr, p_te),
               "brier": brier_score_loss(y_te_arr, p_te)}
    at_cutoff = {name: threshold_metrics(y_te_arr, p_te, thr)
                 for name, thr in thresholds.items()}
    ci = bootstrap_metrics(y_te_arr, p_te, thresholds, n_bootstraps=n_bootstrap)
    primary = at_cutoff[primary_threshold]

    if verbose:
        lo = ci[primary_threshold]["auc"]["lower"]
        hi = ci[primary_threshold]["auc"]["upper"]
        print(f"      AUC {ranking['auc']:.3f} [{lo:.3f}-{hi:.3f}]   "
              f"AUPRC {ranking['auprc']:.3f}   Brier {ranking['brier']:.4f}   "
              f"Sens {primary['sens']:.3f}   Spec {primary['spec']:.3f}")

    return {
        "model": final,
        "scaler": scaler,
        "feat_names": feat_names,
        "best_params": search.best_params_,
        "probas": p_te,
        "y_test": y_te_arr,
        "auc": ranking["auc"],
        "auprc": ranking["auprc"],
        "brier": ranking["brier"],
        "f1": primary["f1"],
        "acc": primary["acc"],
        "sens": primary["sens"],
        "spec": primary["spec"],
        "ppv": primary["ppv"],
        "npv": primary["npv"],
        "threshold": thresholds[primary_threshold],
        "ci": ci[primary_threshold],
        "model_name": model_name,
        "threshold_source": primary_threshold,
        "thresholds": thresholds,
        "metrics_by_threshold": at_cutoff,
        "ci_by_threshold": ci,
        "threshold_raw_scale": thr_raw,
        "alert_rate": alert_rate,
        "rank_agreement_rho": rho,
        "n_train": len(y_tr_arr),
        "n_test": len(y_te_arr),
        "_train_index": X_tr.index.values,
        "_test_index": X_te.index.values,
    }


def diagnostic_row(result: dict, **context) -> dict:
    """One row of `threshold_diagnostics.csv`."""
    row = dict(context)
    at = result["metrics_by_threshold"]
    row.update({
        "threshold_raw_oof": round(result["threshold_raw_scale"], 6),
        "alert_rate": round(result["alert_rate"], 6),
        "threshold_train_oof": round(result["thresholds"]["train_oof"], 6),
        "threshold_test_reopt": round(result["thresholds"]["test_reopt"], 6),
        "spearman_rho": round(result["rank_agreement_rho"], 6),
        "n_bootstrap_used": result["ci"]["n_resamples"],
    })
    for metric in ("sens", "spec", "ppv", "f1"):
        for name in THRESHOLD_NAMES:
            row[f"{metric}_{name}"] = round(at[name][metric], 4)
    return row


METRICS_ORDER = [
    ("AUC", "auc"), ("AUPRC", "auprc"), ("Brier", "brier"),
    ("F1 Score", "f1"), ("Accuracy", "acc"), ("Sensitivity", "sens"),
    ("Specificity", "spec"), ("PPV", "ppv"), ("NPV", "npv"),
]


def metrics_sheets(results: dict, outcomes, outcome_labels,
                   model_key: str) -> dict[str, pd.DataFrame]:
    """Summary, Numeric, and By_threshold sheets from a results object.

    `results` is {outcome: {model_key: result}}, the schema the notebooks
    wrote and the extraction scripts read.
    """
    from ml_utils import fmt_ci

    summary, numeric, by_threshold = [], [], []
    for outcome in outcomes:
        res = results[outcome][model_key]
        label = outcome_labels[outcome]

        row = {"Outcome": label, "Model": model_key}
        for name, key in METRICS_ORDER:
            row[name] = fmt_ci(res[key], res["ci"], key)
        row["Threshold"] = round(res["threshold"], 4)
        row["Threshold source"] = res["threshold_source"]
        summary.append(row)

        row = {"Outcome": label, "Model": model_key}
        for name, key in METRICS_ORDER:
            row[name] = round(res[key], 4)
            row[f"{name}_lower"] = round(res["ci"][key]["lower"], 4)
            row[f"{name}_upper"] = round(res["ci"][key]["upper"], 4)
        row["Threshold"] = round(res["threshold"], 4)
        numeric.append(row)

        for name in THRESHOLD_NAMES:
            row = {"Outcome": label, "Model": model_key,
                   "Threshold source": name,
                   "Threshold": round(res["thresholds"][name], 4)}
            for label_m, key in METRICS_ORDER:
                value = (res[key] if key in ("auc", "auprc", "brier")
                         else res["metrics_by_threshold"][name][key])
                row[label_m] = fmt_ci(value, res["ci_by_threshold"][name], key)
            by_threshold.append(row)

    return {"Summary": pd.DataFrame(summary),
            "Numeric": pd.DataFrame(numeric),
            "By_threshold": pd.DataFrame(by_threshold)}


def write_sheets(sheets: dict[str, pd.DataFrame], path: str) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    print(f"      XLSX -> {path}")
