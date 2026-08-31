"""
=============================================================================
 ML05c -- WITHIN-WOMAN OVERLAP BETWEEN THE TRAINING AND TEST SPLITS
=============================================================================

Recomputes discrimination on the subset of test episodes contributed by women
who appear nowhere in the training split, and compares it with the full test
split. Fits nothing. Every number comes from predictions already stored by
ML02.

THE QUESTION
------------
A woman may contribute more than one pregnancy episode, and the split is drawn
over episodes rather than over women. Two episodes from the same woman are
different pregnancies with different outcomes, which is why they are modelled
as separate observations. The features, however, are not independent: chronic
conditions, region, and premium category repeat across her episodes, and much
of the pre-pregnancy diagnosis history is shared. A model with several hundred
binary indicators can key on that recurring pattern. Where a woman sits on
both sides of the split, part of the test performance may come from
recognising her rather than from generalising to someone new.

Whether this matters is an empirical question, not a matter of principle. This
script answers it by holding out the affected episodes and recomputing.

    full test split      what ML02 reports
    clean subset         test episodes whose woman contributes no training
                         episode

If AUC and the incremental value hold on the clean subset, episode-level
splitting is defensible and one sentence in the Methods records the check. If
they fall, the split has to be redrawn over women, which does require
refitting.

WHAT THE COMPARISON DOES NOT CONTROL
------------------------------------
The clean subset is not a random sample of the test split. Women contributing
one observed episode are enriched in it, and gravidity is associated with both
the features and the outcomes, so a difference between the two estimates
carries case-mix as well as overlap. The subset is also smaller, so its
interval is wider. The comparison is read as a check for a large discrepancy,
not as an unbiased estimate of the overlap effect. The script reports the
gravidity composition of both sets so the case-mix shift is visible rather
than assumed away.

HOW THE SPLIT IS RECOVERED
--------------------------
`fit_evaluate` returns `_test_index`, but it does not survive pickling, so the
ML02 results do not carry it. Where it is present it is used directly. Where
it is not, the split is redrawn with the same seed, size, and stratification,
and the reconstruction is checked against the `y_test` ML02 stored before any
number is computed from it. A mismatch stops the run.

The training split is the complement of the test index within the analytic
set, since the split is a partition. Mapping both to `PSTV01` gives the set of
women on each side without refitting anything.

INPUT
-----
    pregnancy by episode cleaned.csv     read through ML01, for PSTV01
    output_ML02/pre_xgb_results.pkl
    output_ML02/during_xgb_results.pkl

OUTPUT (in OUT_DIR)
-------------------
    within_woman_overlap.csv             episode and woman counts per model
    within_woman_auc_comparison.csv      AUC on the full split and the subset
    within_woman_incremental.csv         delta AUC on the full split and subset
    within_woman_gravidity.csv           case-mix of both sets
    environment_ML05c.txt

USAGE
-----
    python ML05c_within_woman_check.py
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
from sklearn.model_selection import train_test_split

from ML01_prepare_analytic_set import (OUTCOME_LABELS, OUTCOMES,
                                       load_analytic_set)
from ml_utils import environment_report

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

ML02_DIR = r"./output_ML02"
OUT_DIR = r"./output_ML05c"

MODEL_KEY = "XGBoost"
WINDOW_FILES = {"pre": "pre_xgb_results.pkl",
                "during": "during_xgb_results.pkl"}

# Column identifying the woman. Set in step 4 of the cohort pipeline and
# dropped from the feature matrix by ML01's DROP_ALWAYS.
WOMAN_ID = "PSTV01"

# Must match ML02. Used to redraw the test split when the stored results do
# not carry the index; the reconstruction is verified against the stored
# outcome vector before it is used.
RANDOM_STATE = 123
TEST_SIZE = 0.30

N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# Minimum clean-subset positives below which an interval is not reported. With
# fewer events the bootstrap distribution is driven by a handful of episodes
# and an interval would suggest a precision the subset does not have.
MIN_POSITIVES = 25


# =============================================================================
# LOADING
# =============================================================================


def read_pickle(directory: str, filename: str):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        sys.exit(f"[FATAL] not found: {os.path.abspath(path)}\n"
                 "        Run ML02 before this check.")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def stored_result(store, outcome: str) -> dict | None:
    if outcome not in store:
        return None
    return store[outcome].get(MODEL_KEY)


def reconstruct_split(df: pd.DataFrame, outcome: str, stored_y) -> pd.Index:
    """Redraw the ML02 test split and verify it against the stored outcomes.

    ML02 does not persist `_test_index`: `fit_evaluate` returns it and ML02
    uses it for SHAP within the run, but it does not survive pickling. Mapping
    test episodes to women therefore requires the split to be redrawn here.

    `train_test_split` is deterministic in the number of rows, the stratifying
    labels, and the seed, none of which have changed, so splitting the row
    index reproduces the partition ML02 drew over the feature matrix. That is
    an argument, not a guarantee, so it is checked: the outcome vector of the
    reconstructed test split must equal the `y_test` ML02 stored, element for
    element. Across a split of this size an incorrect reconstruction that
    reproduced the vector exactly is not a realistic possibility.

    A mismatch stops the run. The alternative would be to report a within-woman
    comparison built on the wrong episodes, which is worse than reporting
    nothing.
    """
    _, test_index = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df[outcome])

    stored_y = np.asarray(stored_y)
    if len(test_index) != len(stored_y):
        sys.exit(
            f"[FATAL] {outcome}: reconstructed test split has "
            f"{len(test_index):,} rows, ML02 stored {len(stored_y):,}. "
            "TEST_SIZE or the cohort file does not match the ML02 run."
        )

    rebuilt_y = df.loc[test_index, outcome].values
    if not np.array_equal(rebuilt_y, stored_y):
        mismatch = int((rebuilt_y != stored_y).sum())
        sys.exit(
            f"[FATAL] {outcome}: the reconstructed test split does not match "
            f"the one ML02 used ({mismatch:,} of {len(stored_y):,} outcomes "
            "differ).\n"
            "        The split cannot be recovered from the stored results, "
            "so test episodes cannot be mapped to women and this check "
            "cannot be run.\n"
            "        Fix upstream instead of adjusting anything here: have "
            "`fit_evaluate` keep `_test_index` in the saved result, rerun "
            "ML02, and this script will use the stored index directly. "
            "Check first that RANDOM_STATE and TEST_SIZE match ML02, that "
            "the split is stratified on the outcome, and that the cohort "
            "file has not been rebuilt since ML02 was run."
        )
    return test_index


def split_for(df: pd.DataFrame, results: dict, outcome: str) -> pd.Index:
    """Test index for one outcome, from the stored index where it exists."""
    stored = {w: r.get("_test_index") for w, r in results.items()}
    if all(v is not None for v in stored.values()):
        reference = np.asarray(stored["pre"])
        if not np.array_equal(reference, np.asarray(stored["during"])):
            sys.exit(f"[FATAL] {outcome}: the two windows do not share a test "
                     "split, so a paired difference is not defined.")
        return pd.Index(reference)

    y_pre = np.asarray(results["pre"]["y_test"])
    y_during = np.asarray(results["during"]["y_test"])
    if not np.array_equal(y_pre, y_during):
        sys.exit(f"[FATAL] {outcome}: the two windows do not share a test "
                 "split, so a paired difference is not defined.")
    return reconstruct_split(df, outcome, y_pre)


# =============================================================================
# METRICS
# =============================================================================


def auc_with_ci(y, p, n_bootstrap: int, seed: int):
    """AUC and a percentile interval, or NaN where it is not defined."""
    y = np.asarray(y)
    p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")

    point = roc_auc_score(y, p)
    if int(y.sum()) < MIN_POSITIVES:
        return point, float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], p[idx]))
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_delta(y, p_pre, p_during, n_bootstrap: int, seed: int):
    """Difference in AUC between the two windows, bootstrapped in pairs.

    Both windows are scored on the same resampled rows and the difference is
    taken within the resample, so the correlation between the two models is
    removed from the interval rather than reported as uncertainty.
    """
    y = np.asarray(y)
    p_pre = np.asarray(p_pre, float)
    p_during = np.asarray(p_during, float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")

    point = roc_auc_score(y, p_during) - roc_auc_score(y, p_pre)
    if int(y.sum()) < MIN_POSITIVES:
        return point, float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], p_during[idx])
                     - roc_auc_score(y[idx], p_pre[idx]))
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML05c -- WITHIN-WOMAN OVERLAP BETWEEN TRAINING AND TEST SPLITS")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML05c.txt")))

    df = load_analytic_set(verbose=False)
    if WOMAN_ID not in df.columns:
        sys.exit(f"[FATAL] {WOMAN_ID} absent from the analytic set. This "
                 "check needs the woman identifier, which ML01 drops from "
                 "the feature matrix but leaves in the loaded frame.")

    women = df[WOMAN_ID]
    episodes_per_woman = women.value_counts()
    repeat_women = int((episodes_per_woman > 1).sum())
    print(f"\n  {len(df):,} episodes contributed by "
          f"{episodes_per_woman.size:,} women")
    print(f"  {repeat_women:,} women contribute more than one episode "
          f"({repeat_women / episodes_per_woman.size * 100:.2f}%)")

    stores = {w: read_pickle(ML02_DIR, f) for w, f in WINDOW_FILES.items()}

    overlap_rows, auc_rows, delta_rows, gravidity_rows = [], [], [], []

    for outcome in OUTCOMES:
        label = OUTCOME_LABELS[outcome]
        print(f"\n  {'-' * 66}\n  {label}")

        results = {w: stored_result(stores[w], outcome) for w in WINDOW_FILES}
        if any(r is None for r in results.values()):
            print("      [skip] not present in both windows")
            continue

        test_index = split_for(df, results, outcome)
        train_index = df.index.difference(test_index)

        train_women = set(women.loc[train_index].unique())
        test_women = women.loc[test_index]
        clean_mask = ~test_women.isin(train_women).values

        n_test = len(test_index)
        n_clean = int(clean_mask.sum())
        n_overlap_women = len(set(test_women.unique()) & train_women)

        print(f"      test episodes {n_test:,}   "
              f"clean {n_clean:,} ({n_clean / n_test * 100:.1f}%)   "
              f"women on both sides {n_overlap_women:,}")

        overlap_rows.append({
            "outcome": label,
            "n_test_episodes": n_test,
            "n_test_women": int(test_women.nunique()),
            "n_women_in_both_splits": n_overlap_women,
            "n_clean_episodes": n_clean,
            "pct_clean_episodes": round(n_clean / n_test * 100, 2),
        })

        # Gravidity composition, so the case-mix shift between the full split
        # and the clean subset is visible.
        if "par_risk" in df.columns:
            par = df.loc[test_index, "par_risk"].values
            for name, mask in (("full test split", np.ones(n_test, bool)),
                               ("clean subset", clean_mask)):
                if mask.sum() == 0:
                    continue
                subset = par[mask]
                gravidity_rows.append({
                    "outcome": label, "set": name, "n": int(mask.sum()),
                    "pct_first_pregnancy": round((subset == 0).mean() * 100, 2),
                    "pct_second_to_fourth": round((subset == 1).mean() * 100, 2),
                    "pct_fifth_or_later": round((subset == 2).mean() * 100, 2),
                })

        y = np.asarray(results["pre"]["y_test"])
        probas = {w: np.asarray(results[w]["probas"], float)
                  for w in WINDOW_FILES}

        for w in WINDOW_FILES:
            if len(probas[w]) != n_test or len(y) != n_test:
                sys.exit(
                    f"[FATAL] {outcome}/{w}: stored predictions have "
                    f"{len(probas[w])} rows and the test index has {n_test}. "
                    "The two are not aligned and the subset cannot be taken."
                )

        if n_clean == 0:
            print("      [skip] no test episode is free of training overlap")
            continue

        for w in WINDOW_FILES:
            full = auc_with_ci(y, probas[w], N_BOOTSTRAP, RANDOM_SEED)
            clean = auc_with_ci(y[clean_mask], probas[w][clean_mask],
                                N_BOOTSTRAP, RANDOM_SEED)
            auc_rows.append({
                "outcome": label, "window": w,
                "n_full": n_test, "positives_full": int(y.sum()),
                "auc_full": round(full[0], 6),
                "auc_full_lower": round(full[1], 6),
                "auc_full_upper": round(full[2], 6),
                "n_clean": n_clean,
                "positives_clean": int(y[clean_mask].sum()),
                "auc_clean": round(clean[0], 6),
                "auc_clean_lower": round(clean[1], 6),
                "auc_clean_upper": round(clean[2], 6),
                "auc_difference": round(clean[0] - full[0], 6),
            })
            print(f"      {w:<7} AUC full {full[0]:.4f}   "
                  f"clean {clean[0]:.4f}   "
                  f"difference {clean[0] - full[0]:+.4f}")

        full_delta = paired_delta(y, probas["pre"], probas["during"],
                                  N_BOOTSTRAP, RANDOM_SEED)
        clean_delta = paired_delta(y[clean_mask], probas["pre"][clean_mask],
                                   probas["during"][clean_mask],
                                   N_BOOTSTRAP, RANDOM_SEED)
        delta_rows.append({
            "outcome": label,
            "n_full": n_test, "n_clean": n_clean,
            "delta_auc_full": round(full_delta[0], 6),
            "delta_auc_full_lower": round(full_delta[1], 6),
            "delta_auc_full_upper": round(full_delta[2], 6),
            "delta_auc_clean": round(clean_delta[0], 6),
            "delta_auc_clean_lower": round(clean_delta[1], 6),
            "delta_auc_clean_upper": round(clean_delta[2], 6),
            "delta_difference": round(clean_delta[0] - full_delta[0], 6),
        })
        print(f"      delta   full {full_delta[0]:+.4f}   "
              f"clean {clean_delta[0]:+.4f}   "
              f"difference {clean_delta[0] - full_delta[0]:+.4f}")

    for rows, name in ((overlap_rows, "within_woman_overlap.csv"),
                       (auc_rows, "within_woman_auc_comparison.csv"),
                       (delta_rows, "within_woman_incremental.csv"),
                       (gravidity_rows, "within_woman_gravidity.csv")):
        if rows:
            path = os.path.join(OUT_DIR, name)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"\n  {name} -> {path}")

    if delta_rows:
        print("\n  incremental value, full test split against clean subset")
        print(pd.DataFrame(delta_rows)[
            ["outcome", "n_clean", "delta_auc_full", "delta_auc_clean",
             "delta_difference"]].to_string(index=False))

    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()