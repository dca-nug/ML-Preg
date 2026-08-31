"""
=============================================================================
 ML08 -- POPULATION CHARACTERISTICS
=============================================================================

Builds the descriptive tables that open the Results section. Reads the
analytic set through ML01 and fits nothing.

    Complications         prevalence of each of the six outcomes, and how
                          often they occur together

Percentages use two decimal places in every table. Denominators are column
totals: a percentage in the case column is a share of cases, so each variable
sums to 100 down a column, not across a row.
    Demographics          whole cohort, age as mean and SD, the rest as n (%)
    By_complication       case against control for each outcome, with a test
    Effect_size           Cohen's d for age, Cramer's V for the categorical
                          variables

ON THE p-VALUES
---------------
With 551,488 episodes, a difference of no clinical consequence will have a
p below 0.001. The p-values are reported because reviewers expect them in a
table of this kind, and the effect sizes are reported because they are what
the table is actually for. Where the two disagree, the effect size is the
one to read: a Cramer's V of 0.02 across half a million episodes describes a
difference that exists and does not matter.

The co-occurrence counts exclude abortive outcome, since a pregnancy that
ends before twenty weeks cannot also end in preterm birth, PROM, abruption,
or postpartum haemorrhage. Counting it among concurrent conditions would
create a category that cannot occur.

INPUT
-----
    pregnancy by episode cleaned.csv     read through ML01

OUTPUT (in OUT_DIR)
-------------------
    population_characteristics.xlsx      four sheets, as above
    benchmark_ML08.csv
    environment_ML08.txt

USAGE
-----
    python ML08_population_characteristics.py
=============================================================================
"""

from __future__ import annotations

import itertools
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import chi2_contingency, ttest_ind

from ML01_prepare_analytic_set import OUTCOMES, load_analytic_set
from ml_utils import Bench, environment_report, warn_if_no_psutil

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR = r"./output_ML08"

OUTCOME_NAMES = {
    "c_abortive": "Abortive outcome",
    "c_preecl": "Preeclampsia",
    "c_preterm": "Preterm birth",
    "c_prom": "Premature rupture of membranes",
    "c_abrupt": "Placental abruption",
    "c_pph": "Postpartum haemorrhage",
}

# Order the complication table follows in the manuscript.
COMPLICATION_ORDER = ["c_abortive", "c_preecl", "c_prom", "c_preterm",
                      "c_abrupt", "c_pph"]

# Abortive outcome is excluded: it precludes the other five.
CO_OCCURRING = ["c_preecl", "c_prom", "c_preterm", "c_abrupt", "c_pph"]

SHORT_NAME = {"c_preecl": "preeclampsia", "c_prom": "PROM",
              "c_preterm": "preterm birth", "c_abrupt": "placental abruption",
              "c_pph": "PPH"}

# variable, display label, type
#
# `n_preg` and `par_risk` both appear. The banded form is what the models use;
# the ungrouped form shows how the bands were populated, which matters here
# because the highest band is nearly empty (see the note below).
VARIABLES = [
    ("age", "Age", "continuous"),
    ("age_risk", "Maternal age risk", "categorical"),
    ("dom", "Region of residence", "categorical"),
    ("subsid", "Membership status", "categorical"),
    ("n_preg", "Gravidity", "categorical"),
    ("par_risk", "Pregnancy order at index event", "categorical"),
    ("ref_year", "Year of pregnancy", "categorical"),
]

# Gravidity counts pregnancies observed inside the study window, not lifetime
# gravidity: a pregnancy before 2015 is not in the data and cannot be
# counted. Episodes at gravidity five and above are therefore far rarer than
# they are in the population, and the ungrouped distribution makes that
# visible in a way the three bands do not.

# Levels above this are pooled into one row, since a level holding a handful
# of episodes supports no comparison and widens every table by a column.
# None disables pooling.
POOL_ABOVE = {"n_preg": 7}

# Percentages are reported to two decimal places throughout. With 551,488
# episodes the second decimal is supported by the data - the standard error
# of a proportion near 18 percent is about 0.05 percentage points - and the
# complication table needs it: placental abruption is 0.12 percent, which one
# decimal would round to 0.1 and make indistinguishable from 0.14.
PERCENT_DECIMALS = 2

LEVEL_LABELS = {
    "age_risk": {0: "20-35 years", 1: "under 20 or over 35 years"},
    "par_risk": {0: "first pregnancy", 1: "second to fourth", 2: "fifth or later"},
    "dom": {0: "Java-Bali", 1: "outer islands"},
    "subsid": {0: "non-subsidised", 1: "subsidised"},
}


# =============================================================================
# EFFECT SIZES
# =============================================================================


def level_label(column: str, level) -> str:
    """Display name for one level, whole numbers rendered without a decimal."""
    named = LEVEL_LABELS.get(column, {})
    if level in named:
        return named[level]
    if isinstance(level, (int, np.integer)) or (
            isinstance(level, float) and float(level).is_integer()):
        return str(int(level))
    return str(level)


def pool_high_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the tail of a count variable into a single top level.

    Applied to a copy: the pooled column is used for the descriptive tables
    only and never reaches a model.
    """
    df = df.copy()
    for column, ceiling in POOL_ABOVE.items():
        if column not in df.columns or ceiling is None:
            continue
        above = int((df[column] > ceiling).sum())
        if above:
            print(f"    {column}: {above:,} episodes above {ceiling} "
                  f"pooled into '{ceiling} or more'")
            LEVEL_LABELS.setdefault(column, {})[ceiling] = f"{ceiling} or more"
            df[column] = df[column].clip(upper=ceiling)
    return df


def available_variables(df: pd.DataFrame):
    """Drop any characteristic whose column is not in the file, with a note."""
    present, absent = [], []
    for column, label, kind in VARIABLES:
        (present if column in df.columns else absent).append(
            (column, label, kind))
    for column, label, _ in absent:
        print(f"    [note] {label} ({column}) not in the file, omitted")
    return present


def percent(count: int, denominator: int) -> str:
    """Count with its percentage, e.g. "44,760 (8.12%)"."""
    if denominator <= 0:
        return f"{count:,} (NA)"
    share = count / denominator * 100
    return f"{count:,} ({share:.{PERCENT_DECIMALS}f}%)"


def cohens_d(case: pd.Series, control: pd.Series) -> float:
    n1, n2 = len(case), len(control)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = case.std(ddof=1), control.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (case.mean() - control.mean()) / pooled if pooled > 0 else np.nan


def cramers_v(variable: pd.Series, outcome: pd.Series) -> float:
    table = pd.crosstab(variable, outcome)
    if min(table.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    k = min(table.shape) - 1
    return np.sqrt(chi2 / (table.values.sum() * k)) if k > 0 else np.nan


# =============================================================================
# TABLES
# =============================================================================


def table_complications(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    def fmt(count: int) -> str:
        # A co-occurring pair can be rarer than the smallest value two
        # decimals can express. Rounding those to 0.00% would read as absent.
        share = count / total * 100
        floor = 10 ** -PERCENT_DECIMALS
        return (f"{count:,} (<{floor:.{PERCENT_DECIMALS}f}%)"
                if 0 < count and share < floor
                else percent(count, total))

    rows = [{"Complication": "Pregnancy complications", "n (%)": ""}]
    for outcome in COMPLICATION_ORDER:
        rows.append({"Complication": "    " + OUTCOME_NAMES[outcome],
                     "n (%)": fmt(int(df[outcome].sum()))})

    concurrent = df[CO_OCCURRING].sum(axis=1)
    rows.append({"Complication": "Multiple complications", "n (%)": ""})
    rows.append({"Complication": "    Two concurrent conditions", "n (%)":
                 fmt(int((concurrent == 2).sum()))})

    pairs = df[concurrent == 2]
    counts = []
    for a, b in itertools.combinations(CO_OCCURRING, 2):
        n = int(((pairs[a] == 1) & (pairs[b] == 1)).sum())
        if n > 0:
            counts.append((f"{SHORT_NAME[a]} and {SHORT_NAME[b]}", n))
    for label, n in sorted(counts, key=lambda item: -item[1]):
        rows.append({"Complication": "        " + label, "n (%)": fmt(n)})

    for k, name in [(3, "Three concurrent conditions"),
                    (4, "Four concurrent conditions"),
                    (5, "Five concurrent conditions")]:
        n = int((concurrent == k).sum())
        if k <= 4 or n > 0:
            rows.append({"Complication": "    " + name, "n (%)": fmt(n)})

    return pd.DataFrame(rows)


def table_demographics(df: pd.DataFrame, variables) -> pd.DataFrame:
    total = len(df)
    rows = [{"Characteristic": "N", "Value": f"{total:,}"}]
    for column, label, kind in variables:
        if kind == "continuous":
            rows.append({"Characteristic": f"{label}, mean (SD)",
                         "Value": f"{df[column].mean():.2f} "
                                  f"({df[column].std(ddof=1):.2f})"})
        else:
            for level in sorted(df[column].dropna().unique()):
                n = int((df[column] == level).sum())
                rows.append({"Characteristic":
                             f"{label}: {level_label(column, level)}",
                             "Value": percent(n, total)})
    return pd.DataFrame(rows)


def table_by_complication(df: pd.DataFrame, variables) -> pd.DataFrame:
    index = []
    for column, label, kind in variables:
        if kind == "continuous":
            index.append((column, f"{label}, mean (SD)", None))
        else:
            for level in sorted(df[column].dropna().unique()):
                index.append((column,
                              f"{label}: {level_label(column, level)}, n (%)",
                              level))

    kinds = {column: kind for column, _, kind in variables}
    table = {("", "Characteristic"): [row[1] for row in index]}

    for outcome in OUTCOMES:
        case = df[df[outcome] == 1]
        control = df[df[outcome] == 0]

        p_values = {}
        for column, _, kind in variables:
            try:
                if kind == "continuous":
                    p_values[column] = ttest_ind(case[column], control[column],
                                                 equal_var=False).pvalue
                else:
                    p_values[column] = chi2_contingency(
                        pd.crosstab(df[column], df[outcome]),
                        correction=False)[1]
            except Exception:
                p_values[column] = np.nan

        case_column, control_column, p_column, seen = [], [], [], set()
        for column, _, level in index:
            if kinds[column] == "continuous":
                case_column.append(f"{case[column].mean():.2f} "
                                   f"({case[column].std(ddof=1):.2f})")
                control_column.append(f"{control[column].mean():.2f} "
                                      f"({control[column].std(ddof=1):.2f})")
            else:
                n_case = int((case[column] == level).sum())
                n_control = int((control[column] == level).sum())
                case_column.append(percent(n_case, len(case)))
                control_column.append(percent(n_control, len(control)))
            if column not in seen:
                p = p_values[column]
                p_column.append("<0.001" if p == p and p < 0.001
                                else (f"{p:.3f}" if p == p else "NA"))
                seen.add(column)
            else:
                p_column.append("")

        name = OUTCOME_NAMES[outcome]
        table[(name, f"Case (n={len(case):,})")] = case_column
        table[(name, f"Control (n={len(control):,})")] = control_column
        table[(name, "p")] = p_column

    return pd.DataFrame(table)


def table_effect_size(df: pd.DataFrame, variables) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for column, label, kind in variables:
        marker = "+" if kind == "continuous" else "*"
        row = {"Characteristic": label + marker}
        for outcome in OUTCOMES:
            if kind == "continuous":
                value = cohens_d(df[df[outcome] == 1][column],
                                 df[df[outcome] == 0][column])
            else:
                value = cramers_v(df[column], df[outcome])
            row[OUTCOME_NAMES[outcome]] = round(value, 3) if value == value else np.nan
        rows.append(row)

    footnotes = pd.DataFrame([
        {"Characteristic": "+ Cohen's d: negligible <0.2, small 0.2-0.5, "
                           "medium 0.5-0.8, large >=0.8"},
        {"Characteristic": "* Cramer's V: negligible <0.1, small 0.1-0.3, "
                           "medium 0.3-0.5, large >=0.5"},
    ])
    return pd.DataFrame(rows), footnotes


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" ML08 -- POPULATION CHARACTERISTICS")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_ML08.txt")))
    warn_if_no_psutil()

    bench = Bench("ML08, population characteristics")

    with bench.stage("Load analytic set", "pandas") as b:
        df = load_analytic_set()
        b["rows_out"] = len(df)

    with bench.stage("Build tables", "pandas", rows_in=len(df)) as b:
        df = pool_high_levels(df)
        variables = available_variables(df)
        complications = table_complications(df)
        demographics = table_demographics(df, variables)
        by_complication = table_by_complication(df, variables)
        effect_size, footnotes = table_effect_size(df, variables)
        b["rows_out"] = len(df)

    path = os.path.join(OUT_DIR, "population_characteristics.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        complications.to_excel(writer, sheet_name="Complications", index=False)
        demographics.to_excel(writer, sheet_name="Demographics", index=False)
        by_complication.to_excel(writer, sheet_name="By_complication")
        effect_size.to_excel(writer, sheet_name="Effect_size", index=False)
        footnotes.to_excel(writer, sheet_name="Effect_size", index=False,
                           header=False, startrow=len(effect_size) + 2)
    print(f"\n  XLSX -> {path}")

    print("\n  complication frequencies")
    print(complications.to_string(index=False))
    print("\n  effect sizes")
    print(effect_size.to_string(index=False))

    bench.finalise(os.path.join(OUT_DIR, "benchmark_ML08.csv"), rows_out=len(df))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()