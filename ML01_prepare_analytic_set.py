"""
=============================================================================
 ML01 -- ANALYTIC SET PREPARATION AND FEATURE SELECTION
=============================================================================

Loads the pregnancy-level dataset produced by the cohort-reconstruction
pipeline, derives the one engineered variable the models expect, and defines
which columns are eligible as predictors in each of the two exposure windows.

Every training script imports this module rather than repeating the logic, so
there is exactly one definition of the feature set in the repository. Running
this file directly performs the checks and writes the feature manifest without
training anything, which is the fastest way to confirm that a rebuilt cohort
is still compatible with the models.

WINDOWS
-------
    pre       diagnoses recorded before estimated pregnancy onset. All
              antenatal (c_) and post-delivery (a_) flags are removed, so the
              model sees only pre-pregnancy history and demographics.

    during    diagnoses recorded during pregnancy. Post-delivery flags are
              removed. Antenatal flags that record the outcome itself, the
              mode or course of delivery, or a competing terminal event are
              removed per outcome through `IGNORED_COLS_MAP`; antenatal flags
              that represent a diagnosis available before the outcome is
              known are retained.

`IGNORED_COLS_MAP` is reproduced column for column from the analysis
notebooks. It is written out per outcome rather than collapsed into one
shared list even though the six lists are currently identical in content,
because each entry encodes a clinical judgement about one outcome and the
lists are not interchangeable in principle.

WHY THE SELECTION IS GUARDED
----------------------------
The notebooks built the feature matrix by subtraction: take every column and
remove the named ones. Under that rule the feature set is never stated
anywhere, it is whatever the input file happens to contain minus a list. A
column added upstream becomes a predictor with no message in the log. The
episode-level file carries `pregnancy_start` and `pregnancy_end`, and the
second of those is the date of the outcome.

The subtraction is therefore kept exactly as it was, and three checks are
added after it:

    1. Every retained column matches `b_*`, `c_*`, or a named demographic.
    2. Every retained column is numeric.
    3. No `a_*` column survives, and in the `pre` window no `c_*` column does.

A column the pipeline has not been told about halts the run instead of
entering the model. The checks cannot change the feature set: they either
pass, leaving it as the subtraction produced it, or they stop execution.

INPUT
-----
    pregnancy by episode cleaned.csv     from step 5 of the cohort pipeline

OUTPUT (when run directly)
--------------------------
    feature_manifest.csv                 window x outcome x feature
    feature_counts.csv                   feature count per window and outcome
    cohort_fingerprint.csv               row count, columns, outcome prevalence
    environment_ML01.txt                 library versions and machine spec

USAGE
-----
    python ML01_prepare_analytic_set.py
=============================================================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml_utils import environment_report

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = r"./pregnancy by episode cleaned.csv"
OUT_DIR = r"./output_ML01"

OUTCOMES = ["c_abortive", "c_preecl", "c_preterm", "c_prom", "c_abrupt", "c_pph"]
OUTCOME_LABELS = {
    "c_abortive": "Abortive",
    "c_preecl": "Preeclampsia",
    "c_preterm": "Preterm",
    "c_prom": "PROM",
    "c_abrupt": "Abruption",
    "c_pph": "PPH",
}

WINDOWS = ["pre", "during"]

# Demographic predictors. `par_risk` is engineered below; the other three are
# read from the file.
DEMOGRAPHIC_FEATURES = ["age_risk", "dom", "subsid", "par_risk"]

# Identifiers, raw variables superseded by a derived form, and episode
# metadata. `age` and `n_preg` are dropped because `age_risk` and `par_risk`
# replace them; `pregnancy_start` and `pregnancy_end` are dates, and the
# second is the date of the outcome.
DROP_ALWAYS = ["PSTV01", "age", "n_preg", "ref_year",
               "pregnancy_start", "pregnancy_end", "b_normal"]

# =============================================================================
# IGNORED COLUMNS (during window) -- verbatim from During_All_comparison.ipynb
# =============================================================================

_DURING_IGNORED = [
    "PSTV01", "age", "n_preg", "b_normal", "c_abortive", "c_normal", "c_instrum",
    "c_caesar", "c_assisted", "c_preterm", "c_prom", "c_retained", "c_abrupt",
    "c_pph", "c_obstrau", "c_laceration", "c_umbilical", "c_distress", "c_iph",
    "c_ecl", "c_preecl", "c_malpres", "c_obspelvic", "c_long", "c_abnforce",
    "c_fail", "ref_year", "c_malpresent", "c_abnamnio", "c_prolong", "c_disprop",
    "c_previa", "c_multiple", "c_anh", "c_placental", "c_polyhydra",
]

IGNORED_COLS_MAP = {outcome: list(_DURING_IGNORED) for outcome in OUTCOMES}

# `c_abnorpelv` (O34, abnormality of pelvic organs) is absent from every list
# above and is therefore retained as a predictor for all six outcomes. This is
# deliberate: it is a diagnosis recorded during pregnancy, not a description
# of the delivery or of the outcome.

# =============================================================================
# LOADING
# =============================================================================


def add_par_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Gravidity band: 0 = first pregnancy, 1 = second to fourth, 2 = fifth or
    later. Identical definition in both analysis notebooks."""
    if "n_preg" not in df.columns:
        sys.exit("[FATAL] n_preg is required to derive par_risk")
    cond = [(df["n_preg"] == 1),
            (df["n_preg"] >= 2) & (df["n_preg"] <= 4),
            (df["n_preg"] > 4)]
    df["par_risk"] = np.select(cond, [0, 1, 2])
    return df


def load_analytic_set(path: str = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    """Read the episode-level file, derive `par_risk`, and check the schema.

    `age_risk` is read from the file and never recomputed here. It is derived
    in step 4 of the cohort pipeline from the aggregated maternal age, so
    re-deriving it downstream would risk two definitions of the same
    indicator diverging without either being obviously wrong.
    """
    if not os.path.exists(path):
        sys.exit(f"[FATAL] input not found: {os.path.abspath(path)}")

    df = pd.read_csv(path, low_memory=False)
    df = add_par_risk(df)

    required = OUTCOMES + ["age_risk", "dom", "subsid", "n_preg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[FATAL] required columns absent: {missing}")

    for col in OUTCOMES:
        vals = set(pd.unique(df[col].dropna()))
        if not vals <= {0, 1}:
            sys.exit(f"[FATAL] outcome {col} is not binary: {sorted(vals)[:8]}")

    na = [c for c in DEMOGRAPHIC_FEATURES if df[c].isna().any()]
    if na:
        # The models have no imputation step. A missing demographic would
        # propagate to a NaN feature and, depending on the learner, either
        # raise or be silently treated as a category.
        sys.exit(f"[FATAL] missing values in demographic features: {na}. "
                 "Resolve upstream in the cohort pipeline.")

    if verbose:
        print(f"    {len(df):,} episodes x {len(df.columns)} columns")
        print(f"    at-risk maternal age : {df['age_risk'].mean() * 100:5.2f}%")
        for oc in OUTCOMES:
            print(f"    {OUTCOME_LABELS[oc]:<13}: {int(df[oc].sum()):>8,} "
                  f"({df[oc].mean() * 100:5.2f}%)")
    return df


# =============================================================================
# FEATURE SELECTION
# =============================================================================


def dropped_columns(df: pd.DataFrame, window: str, outcome: str) -> list[str]:
    """The subtraction, unchanged from the notebooks."""
    if window == "pre":
        drop = list(DROP_ALWAYS)
        drop += [c for c in df.columns if c.startswith(("a_", "c_"))]
    elif window == "during":
        drop = list(set(IGNORED_COLS_MAP[outcome]) | set(DROP_ALWAYS))
        drop += [c for c in df.columns if c.startswith("a_")]
    else:
        raise ValueError(f"unknown window: {window}")
    return drop


def feature_columns(df: pd.DataFrame, window: str, outcome: str) -> list[str]:
    """Apply the subtraction, then verify what survived it.

    Column order follows the input file so that a stored `feat_names` list can
    be reproduced exactly. Any deviation raises rather than being repaired,
    because a silently repaired feature matrix is the failure this function
    exists to prevent.
    """
    drop = set(dropped_columns(df, window, outcome))
    feats = [c for c in df.columns if c not in drop]

    unexpected = [c for c in feats
                  if not c.startswith(("b_", "c_"))
                  and c not in DEMOGRAPHIC_FEATURES]
    if unexpected:
        sys.exit(
            f"[FATAL] {window}/{outcome}: columns present in the input that "
            f"the pipeline has no rule for: {unexpected}\n"
            "        Add them to DROP_ALWAYS if they are metadata, or to "
            "DEMOGRAPHIC_FEATURES if they are predictors. They are not "
            "admitted by default."
        )

    non_numeric = [c for c in feats
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        sys.exit(f"[FATAL] {window}/{outcome}: non-numeric features: {non_numeric}")

    leaked = [c for c in feats if c.startswith("a_")]
    if leaked:
        sys.exit(f"[FATAL] {window}/{outcome}: post-delivery leakage: {leaked}")

    if window == "pre":
        antenatal = [c for c in feats if c.startswith("c_")]
        if antenatal:
            sys.exit(f"[FATAL] pre/{outcome}: antenatal leakage: {antenatal}")

    if outcome in feats:
        sys.exit(f"[FATAL] {window}/{outcome}: the outcome is in its own feature set")

    if not feats:
        sys.exit(f"[FATAL] {window}/{outcome}: no features survived selection")

    return feats


def build_matrix(df: pd.DataFrame, window: str, outcome: str):
    """Return (X, y, feat_names) for one window and outcome."""
    feats = feature_columns(df, window, outcome)
    return df[feats], df[outcome], feats


# =============================================================================
# MAIN -- checks and manifest only, no training
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML01 -- ANALYTIC SET PREPARATION")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML01.txt")))
    print()

    df = load_analytic_set()

    manifest, counts = [], []
    for window in WINDOWS:
        for outcome in OUTCOMES:
            feats = feature_columns(df, window, outcome)
            counts.append({
                "window": window,
                "outcome": OUTCOME_LABELS[outcome],
                "n_features": len(feats),
                "n_demographic": sum(f in DEMOGRAPHIC_FEATURES for f in feats),
                "n_prepregnancy": sum(f.startswith("b_") for f in feats),
                "n_antenatal": sum(f.startswith("c_") for f in feats),
            })
            manifest += [{"window": window,
                          "outcome": OUTCOME_LABELS[outcome],
                          "feature": f} for f in feats]

    pd.DataFrame(manifest).to_csv(
        os.path.join(OUT_DIR, "feature_manifest.csv"), index=False)
    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(os.path.join(OUT_DIR, "feature_counts.csv"), index=False)

    # The fingerprint is what a later run is compared against when the cohort
    # is rebuilt: it states, in one file, the dataset the models were fitted on.
    fingerprint = [
        {"item": "n_episodes", "value": len(df)},
        {"item": "n_columns", "value": len(df.columns)},
        {"item": "age_risk_prevalence", "value": round(df["age_risk"].mean(), 6)},
    ]
    fingerprint += [{"item": f"prevalence_{oc}", "value": round(df[oc].mean(), 6)}
                    for oc in OUTCOMES]
    fingerprint += [{"item": f"positives_{oc}", "value": int(df[oc].sum())}
                    for oc in OUTCOMES]
    pd.DataFrame(fingerprint).to_csv(
        os.path.join(OUT_DIR, "cohort_fingerprint.csv"), index=False)

    print("\n  feature counts")
    print(counts_df.to_string(index=False))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
