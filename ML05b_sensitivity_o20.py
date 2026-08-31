"""
=============================================================================
 ML05b -- SENSITIVITY ANALYSIS: ANTENATAL O20 REMOVED
=============================================================================

Refits the antenatal-window models with `c_earlyhemo` (ICD-10 O20, haemorrhage
in early pregnancy) removed from the feature set, leaves everything else
identical to ML02, and reports the incremental value both ways here. ML09 is
not involved and does not need to know this arm exists.

WHY THIS ARM EXISTS
-------------------
O20 covers threatened abortion and early-pregnancy bleeding. It is a clinical
antecedent of pregnancy loss rather than a restatement of it: most episodes
coded O20 do not end in loss, and the code is recorded while the pregnancy is
ongoing, so it is available at the time a prediction would be made. It is
therefore retained in the primary analysis.

It is also the antenatal diagnosis lying closest to the abortive outcome in
both time and clinical meaning, and abortive outcome carries the largest
incremental value in the primary analysis. A reader is entitled to ask how
much of that gain rests on this one code. `IGNORED_COLS_MAP` already removes
`c_anh` (O46, antenatal haemorrhage), which is the later-gestation analogue of
O20, so the asymmetry needs an answer rather than an assertion.

This arm supplies the answer as a number. It does not establish that retaining
O20 was wrong. It separates the contribution of the proximal code from that of
the rest of the antenatal feature set.

WHAT IS REPORTED
----------------
Per outcome, on the ML02 test split:

    AUC before            from ML02, unchanged
    AUC during            from ML02, with O20 available
    AUC during, withheld  fitted here
    delta, primary        during minus before
    delta, withheld       during-withheld minus before
    attributable to O20   the difference between the two deltas

All three deltas come from one set of paired resamples, so the models are
always scored on the same rows and the difference between the two deltas is
interpretable. The last line answers the reviewer's question: if it is a small
share of the primary delta, the incremental value does not rest on O20.

WHAT IS HELD FIXED
------------------
    split             same `test_size` and `random_state` as ML02, and `y` is
                      unchanged, so the stratified split is the same one. ML02
                      does not persist its test index, so the run verifies the
                      split by comparing the stored `y_test` vectors element
                      for element and stops if they differ.

    hyperparameters   read from the ML02 antenatal results and passed as a
                      one-point grid, so the search is not repeated. See
                      REUSE_ML02_PARAMS for what each choice measures.

    everything else   folds, scaling, calibration, cut-off rule, bootstrap.

    pre-pregnancy     not refitted and not copied. No `c_` column enters the
                      pre-pregnancy window, so removing one cannot change that
                      model. Its stored predictions are read from ML02 for the
                      paired comparison.

`b_earlyhemo` is a different variable: O20 recorded before the onset of the
index pregnancy, that is, in an earlier gestation. It is history, not a
concurrent finding, and is left in place.

INPUT
-----
    pregnancy by episode cleaned.csv     read through ML01
    output_ML02/pre_xgb_results.pkl      predictions, for the paired delta
    output_ML02/during_xgb_results.pkl   predictions and hyperparameters

OUTPUT (in OUT_DIR)
-------------------
    o20_incremental_value.csv        the comparison table, both deltas
    o20_exclusion_report.csv         what was removed, AUC either way
    during_xgb_noO20_results.pkl     the refitted models
    during_xgb_noO20_scalers.pkl
    during_xgb_noO20_metrics.xlsx    Summary, Numeric, By_threshold
    threshold_diagnostics_noO20.csv
    benchmark_ML05b.csv
    environment_ML05b.txt

USAGE
-----
    python ML05b_sensitivity_o20.py
=============================================================================
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ML01_prepare_analytic_set import (OUTCOME_LABELS, OUTCOMES,
                                       build_matrix, load_analytic_set)
from ml_core import diagnostic_row, fit_evaluate, metrics_sheets, write_sheets
from ml_utils import (Bench, environment_report, save_pickle,
                      warn_if_no_psutil)

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

ML02_DIR = r"./output_ML02"
OUT_DIR = r"./output_ML05b"

# Copied from ML02 rather than imported, so that editing ML02 does not
# silently change what this arm holds fixed. The split assertion below fails
# loudly if the two drift apart.
RANDOM_STATE = 123
TEST_SIZE = 0.30
CV_FOLDS = 5
N_BOOTSTRAP = 1000
N_JOBS = -1
PRIMARY_THRESHOLD = "train_oof"

# Bootstrap for the comparison table. Separate from N_BOOTSTRAP, which is
# passed to `fit_evaluate` for the per-model intervals.
N_BOOTSTRAP_DELTA = 1000
DELTA_SEED = 42

# Below this many positives in the test split, a delta interval is not
# reported. The point estimate still is.
MIN_POSITIVES = 25

MODEL_KEY = "XGBoost"
PREFIX = "during_xgb_noO20"

# The columns withheld in this arm. `c_earlyhemo` is O20 in the antenatal
# window; the mapping is set in step 4 of the cohort pipeline
# (`pregnancy_conditions['earlyhemo'] = ['O20']`).
WITHHELD_COLUMNS = ["c_earlyhemo"]

# Outcomes refitted. O20 is retained for all six in the primary analysis, so
# all six are refitted by default. Narrow to ["c_abortive"] to cut runtime to
# roughly a sixth.
SENSITIVITY_OUTCOMES = list(OUTCOMES)

# True  : hyperparameters fixed at the ML02 values, so the only difference
#         between the two runs is the withheld column. This isolates the
#         contribution of O20.
# False : the full grid is searched again, so the model may adapt to the
#         reduced feature set. This measures what the antenatal window is
#         worth without O20 available at all, which is the more favourable
#         comparison and the slower one.
REUSE_ML02_PARAMS = True

PARAM_GRID_FULL = {
    "n_estimators": [100],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


# =============================================================================
# READING ML02
# =============================================================================


def read_pickle(path: str):
    if not os.path.exists(path):
        sys.exit(f"[FATAL] not found: {os.path.abspath(path)}\n"
                 "        Run ML02 before this arm.")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def ml02_result(window: str, outcome: str) -> dict | None:
    filename = {"pre": "pre_xgb_results.pkl",
                "during": "during_xgb_results.pkl"}[window]
    store = read_pickle(os.path.join(ML02_DIR, filename))
    if outcome not in store:
        return None
    return store[outcome].get(MODEL_KEY)


def ml02_best_params(outcome: str) -> dict:
    result = ml02_result("during", outcome)
    if result is None:
        sys.exit(f"[FATAL] outcome {outcome} absent from the ML02 antenatal "
                 "results. Run ML02, or set REUSE_ML02_PARAMS = False.")
    params = result.get("best_params")
    if not params:
        sys.exit(f"[FATAL] no best_params stored for {outcome}")
    # `fit_evaluate` expects a grid. A one-point grid fits the model the
    # primary analysis selected, without repeating the search.
    return {k: [v] for k, v in params.items()}


# =============================================================================
# CHECKS
# =============================================================================


def resolve_withheld(df: pd.DataFrame) -> list[str]:
    """Confirm the withheld columns exist before anything is fitted.

    A silent no-op is the failure mode this guards against: if the column were
    named differently upstream, the run would complete, reproduce ML02
    exactly, and be reported as evidence that O20 does not matter.
    """
    missing = [c for c in WITHHELD_COLUMNS if c not in df.columns]
    if missing:
        available = sorted(c for c in df.columns if c.startswith("c_"))
        sys.exit(
            f"[FATAL] withheld column(s) absent from the input: {missing}\n"
            f"        antenatal columns present: {available}\n"
            "        Set WITHHELD_COLUMNS to the name this cohort uses for "
            "O20. Do not proceed with an empty exclusion."
        )
    return list(WITHHELD_COLUMNS)


def check_alignment(result_here, pre_result, during_result, outcome: str):
    """Stop unless the three models are scored on the same episodes, in order.

    ML02 does not persist `_test_index`: `fit_evaluate` returns it and ML02
    uses it for SHAP within the run, but it does not survive pickling. The
    check is therefore made on `y_test`, which is stored.

    If the three runs drew the same split in the same order, their `y_test`
    vectors are identical element for element. Across a test split of this
    size, a permutation or a different draw that reproduced the vector exactly
    is not a realistic possibility, so equality is treated as establishing
    both the split and the row order. This is what the paired bootstrap below
    needs: it indexes the three probability vectors with one set of row
    positions.
    """
    stored = {"pre": pre_result, "during": during_result, "noO20": result_here}

    absent = [name for name, res in stored.items() if res.get("y_test") is None]
    if absent:
        sys.exit(f"[FATAL] {outcome}: no stored y_test for {absent}. The "
                 "three models cannot be aligned.")

    lengths = {name: len(res["y_test"]) for name, res in stored.items()}
    lengths.update({f"probas_{name}": len(res["probas"])
                    for name, res in stored.items()})
    if len(set(lengths.values())) != 1:
        sys.exit(f"[FATAL] {outcome}: stored vectors are not the same length: "
                 f"{lengths}")

    reference = np.asarray(stored["noO20"]["y_test"])
    for name in ("pre", "during"):
        other = np.asarray(stored[name]["y_test"])
        if not np.array_equal(reference, other):
            mismatch = int((reference != other).sum())
            sys.exit(
                f"[FATAL] {outcome}: the {name} test outcome vector differs "
                f"from this run in {mismatch:,} of {len(reference):,} rows. "
                "The two arms are not scored on the same episodes, so a "
                "difference in AUC is not attributable to the withheld "
                "column. Check that RANDOM_STATE and TEST_SIZE match ML02 "
                "and that the cohort file has not been rebuilt since ML02 "
                "was run."
            )

    # If a future ML02 does persist the index, use it as a second check.
    indices = {name: res.get("_test_index") for name, res in stored.items()}
    if all(v is not None for v in indices.values()):
        reference_index = np.asarray(indices["noO20"])
        for name in ("pre", "during"):
            if not np.array_equal(np.asarray(indices[name]), reference_index):
                sys.exit(f"[FATAL] {outcome}: {name} test index differs from "
                         "this run despite matching outcome vectors.")


# =============================================================================
# PAIRED BOOTSTRAP
# =============================================================================


def paired_deltas(y, p_pre, p_during, p_withheld, n_bootstrap: int, seed: int):
    """Both incremental values and their difference, from one set of resamples.

    All three models are scored on the same resampled rows within each draw,
    so the correlation between them is removed from the intervals rather than
    reported as uncertainty. Drawing the two deltas from the same resamples is
    what makes their difference interpretable.
    """
    y = np.asarray(y)
    p_pre = np.asarray(p_pre, float)
    p_during = np.asarray(p_during, float)
    p_withheld = np.asarray(p_withheld, float)

    blank = (float("nan"),) * 3
    if len(np.unique(y)) < 2:
        return {"primary": blank, "withheld": blank, "attributable": blank}

    auc_pre = roc_auc_score(y, p_pre)
    point = {"primary": roc_auc_score(y, p_during) - auc_pre,
             "withheld": roc_auc_score(y, p_withheld) - auc_pre}
    point["attributable"] = point["primary"] - point["withheld"]

    if int(y.sum()) < MIN_POSITIVES:
        return {k: (v, float("nan"), float("nan")) for k, v in point.items()}

    rng = np.random.RandomState(seed)
    draws = {"primary": [], "withheld": [], "attributable": []}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y), len(y))
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        base = roc_auc_score(yb, p_pre[idx])
        d_primary = roc_auc_score(yb, p_during[idx]) - base
        d_withheld = roc_auc_score(yb, p_withheld[idx]) - base
        draws["primary"].append(d_primary)
        draws["withheld"].append(d_withheld)
        draws["attributable"].append(d_primary - d_withheld)

    out = {}
    for key, value in point.items():
        sample = draws[key]
        if not sample:
            out[key] = (value, float("nan"), float("nan"))
        else:
            out[key] = (value,
                        float(np.percentile(sample, 2.5)),
                        float(np.percentile(sample, 97.5)))
    return out


def fmt(triple) -> str:
    point, lower, upper = triple
    if not np.isfinite(lower) or not np.isfinite(upper):
        return f"{point:+.4f} (CI not reported)"
    return f"{point:+.4f} ({lower:+.4f} to {upper:+.4f})"


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML05b -- SENSITIVITY ANALYSIS: ANTENATAL O20 REMOVED")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML05b.txt")))
    warn_if_no_psutil()

    bench = Bench("ML05b, O20 withheld")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    with bench.stage("Load analytic set", "pandas") as b:
        df = load_analytic_set()
        b["rows_out"] = len(df)

    withheld = resolve_withheld(df)
    print(f"\n  withholding from the antenatal window: {withheld}")
    for col in withheld:
        print(f"    {col}: present in {int(df[col].sum()):,} episodes "
              f"({df[col].mean() * 100:.2f}%)")

    results, scalers, diagnostics = {}, {}, []
    report, incremental = [], []

    with bench.stage("Train and evaluate (during, O20 withheld)", "xgboost",
                     rows_in=len(df)) as b:
        for outcome in SENSITIVITY_OUTCOMES:
            label = OUTCOME_LABELS[outcome]
            print(f"\n  {'-' * 66}\n  DURING (O20 withheld)  |  {label}  "
                  f"({outcome})")

            X, y, feat_names = build_matrix(df, "during", outcome)
            keep = [f for f in feat_names if f not in withheld]
            removed = [f for f in feat_names if f in withheld]
            if not removed:
                print(f"      [skip] {withheld} already excluded upstream for "
                      f"{label}; this arm would duplicate ML02")
                continue

            X = X[keep]
            print(f"      episodes {len(X):,}   positives {int(y.sum()):,} "
                  f"({y.mean() * 100:.2f}%)   features {len(keep)} "
                  f"(was {len(feat_names)})")

            grid = (ml02_best_params(outcome) if REUSE_ML02_PARAMS
                    else PARAM_GRID_FULL)
            if REUSE_ML02_PARAMS:
                shown = {k: v[0] for k, v in grid.items()}
                print(f"      hyperparameters fixed at ML02 values: {shown}")

            estimator = XGBClassifier(eval_metric="logloss",
                                      random_state=RANDOM_STATE, verbosity=0)
            result = fit_evaluate(
                estimator, grid, X, y, model_name=MODEL_KEY, cv=cv,
                needs_scaling=True, calibrate=True, test_size=TEST_SIZE,
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
                n_bootstrap=N_BOOTSTRAP, primary_threshold=PRIMARY_THRESHOLD,
                calibration_folds=CV_FOLDS)

            pre_result = ml02_result("pre", outcome)
            during_result = ml02_result("during", outcome)
            if pre_result is None or during_result is None:
                sys.exit(f"[FATAL] {outcome} missing from the ML02 results; "
                         "the paired comparison cannot be formed.")
            check_alignment(result, pre_result, during_result, outcome)

            results[outcome] = {MODEL_KEY: result}
            scalers[outcome] = result["scaler"]
            diagnostics.append(diagnostic_row(result, window="during",
                                              outcome=label, model=MODEL_KEY))

            y_test = np.asarray(pre_result["y_test"])
            deltas = paired_deltas(
                y_test, pre_result["probas"], during_result["probas"],
                result["probas"], N_BOOTSTRAP_DELTA, DELTA_SEED)

            auc_pre = pre_result["auc"]
            auc_during = during_result["auc"]
            auc_withheld = result["auc"]

            print(f"      AUC before {auc_pre:.4f}   during {auc_during:.4f}   "
                  f"during without O20 {auc_withheld:.4f}")
            print(f"      delta, primary        {fmt(deltas['primary'])}")
            print(f"      delta, O20 withheld   {fmt(deltas['withheld'])}")
            print(f"      attributable to O20   {fmt(deltas['attributable'])}")

            primary_point = deltas["primary"][0]
            share = (deltas["attributable"][0] / primary_point * 100
                     if np.isfinite(primary_point) and primary_point != 0
                     else float("nan"))

            report.append({
                "outcome": label,
                "withheld": ", ".join(removed),
                "n_features_ML02": len(feat_names),
                "n_features_noO20": len(keep),
                "auc_during_ML02": round(auc_during, 6),
                "auc_during_noO20": round(auc_withheld, 6),
                "auc_difference": round(auc_withheld - auc_during, 6),
                "hyperparameters": ("fixed at ML02" if REUSE_ML02_PARAMS
                                    else "re-searched"),
            })

            row = {"outcome": label,
                   "n_test": len(y_test),
                   "positives_test": int(y_test.sum()),
                   "auc_before": round(auc_pre, 6),
                   "auc_during": round(auc_during, 6),
                   "auc_during_noO20": round(auc_withheld, 6)}
            for key in ("primary", "withheld", "attributable"):
                point, lower, upper = deltas[key]
                row[f"delta_{key}"] = round(point, 6)
                row[f"delta_{key}_lower"] = round(lower, 6)
                row[f"delta_{key}_upper"] = round(upper, 6)
                row[f"delta_{key}_formatted"] = fmt(deltas[key])
            row["pct_of_primary_delta_from_O20"] = (
                round(share, 2) if np.isfinite(share) else np.nan)
            incremental.append(row)

        b["rows_out"] = len(df)

    if not results:
        sys.exit("[FATAL] no outcome was refitted; nothing to write")

    save_pickle(results, os.path.join(OUT_DIR, f"{PREFIX}_results.pkl"))
    save_pickle(scalers, os.path.join(OUT_DIR, f"{PREFIX}_scalers.pkl"))
    write_sheets(
        metrics_sheets(results, list(results.keys()), OUTCOME_LABELS, MODEL_KEY),
        os.path.join(OUT_DIR, f"{PREFIX}_metrics.xlsx"))

    pd.DataFrame(diagnostics).to_csv(
        os.path.join(OUT_DIR, "threshold_diagnostics_noO20.csv"), index=False)
    pd.DataFrame(report).to_csv(
        os.path.join(OUT_DIR, "o20_exclusion_report.csv"), index=False)

    incremental_df = pd.DataFrame(incremental)
    path = os.path.join(OUT_DIR, "o20_incremental_value.csv")
    incremental_df.to_csv(path, index=False)

    print(f"\n{'=' * 74}")
    print(" INCREMENTAL VALUE WITH AND WITHOUT O20")
    print("=" * 74)
    display = incremental_df[[
        "outcome", "auc_before", "auc_during", "auc_during_noO20",
        "delta_primary_formatted", "delta_withheld_formatted",
        "delta_attributable_formatted", "pct_of_primary_delta_from_O20",
    ]].rename(columns={
        "delta_primary_formatted": "delta (primary)",
        "delta_withheld_formatted": "delta (O20 withheld)",
        "delta_attributable_formatted": "attributable to O20",
        "pct_of_primary_delta_from_O20": "% of delta",
    })
    print(display.to_string(index=False))
    print(f"\n  written to {path}")

    bench.finalise(os.path.join(OUT_DIR, "benchmark_ML05b.csv"),
                   rows_out=len(df))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()