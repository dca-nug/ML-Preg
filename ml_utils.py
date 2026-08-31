"""
=============================================================================
 SHARED UTILITIES
=============================================================================

Instrumentation, metric computation, and threshold derivation used by every
script in this repository. Nothing here is specific to one model or one
outcome; anything that is belongs in the script that owns it.

This module is deliberately self-contained. The cohort-reconstruction
repository has an equivalent `pipeline_utils.py`, and the benchmarking API is
kept identical so that the two sets of benchmark files can be read with the
same code, but neither repository imports the other.

CONTENTS
--------
    Bench                    per-stage wall-clock time and peak memory
    environment_report       library versions and machine specification
    warn_if_no_psutil        memory figures are absent without psutil

    threshold_metrics        confusion-matrix metrics at a fixed cut-off
    bootstrap_metrics        percentile CIs, one or more cut-offs per run
    youden_threshold         argmax(TPR - FPR) on a score vector
    transport_threshold      operating point moved between score scales
    fmt_ci                   "0.812 (0.804-0.820)" for the summary sheets
=============================================================================
"""

from __future__ import annotations

import os
import platform
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:                                   # pragma: no cover
    _HAS_PSUTIL = False


# =============================================================================
# INSTRUMENTATION
# =============================================================================


def warn_if_no_psutil() -> None:
    if not _HAS_PSUTIL:
        print("    [note] psutil is not installed; peak memory will be blank. "
              "pip install psutil")


def _rss_mb() -> float:
    if not _HAS_PSUTIL:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


class Bench:
    """Records one row per stage: label, engine, rows in/out, seconds, peak MB.

    Used as::

        bench = Bench("ML02, XGBoost main")
        with bench.stage("Train", "xgboost", rows_in=len(X)) as b:
            ...
            b["rows_out"] = len(X)
        bench.finalise("benchmark_ML02.csv", rows_out=len(X))

    Peak memory is the maximum resident set size observed at the start and end
    of the stage. It is a floor, not a true peak: a short-lived allocation
    inside the stage is invisible. Sampling continuously would need a second
    thread, which is not worth the complexity for a reported figure that only
    needs to establish the order of magnitude.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.rows: list[dict] = []
        self.t0 = time.time()

    @contextmanager
    def stage(self, label: str, engine: str = "", rows_in: int | None = None):
        row = {"stage": label, "engine": engine, "rows_in": rows_in,
               "rows_out": None, "seconds": None, "peak_mb": None}
        print(f"\n  [{len(self.rows) + 1}] {label}")
        t, m0 = time.time(), _rss_mb()
        try:
            yield row
        finally:
            row["seconds"] = round(time.time() - t, 2)
            row["peak_mb"] = round(max(m0, _rss_mb()), 1)
            self.rows.append(row)
            print(f"      done in {row['seconds']:,.2f} s"
                  + (f", peak {row['peak_mb']:,.0f} MB"
                     if row["peak_mb"] == row["peak_mb"] else ""))

    def finalise(self, path: str, rows_out: int | None = None) -> None:
        total = round(time.time() - self.t0, 2)
        self.rows.append({"stage": "TOTAL", "engine": self.name,
                          "rows_in": None, "rows_out": rows_out,
                          "seconds": total, "peak_mb": round(_rss_mb(), 1)})
        pd.DataFrame(self.rows).to_csv(path, index=False)
        print(f"\n  total {total:,.2f} s   benchmark -> {path}")


def environment_report(path: str) -> str:
    """Write and return the library versions and machine specification.

    Recorded because a run is only reproducible against a stated environment,
    and because XGBoost and scikit-learn have both changed default behaviour
    between minor versions in ways that move the third decimal of an AUC.
    """
    lines = [
        f"generated        : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"python           : {sys.version.split()[0]}",
        f"platform         : {platform.platform()}",
        f"processor        : {platform.processor() or 'unknown'}",
        f"logical cores    : {os.cpu_count()}",
    ]
    if _HAS_PSUTIL:
        lines.append(f"total memory     : "
                     f"{psutil.virtual_memory().total / 1024 ** 3:.1f} GB")

    for mod in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
                "catboost", "imblearn", "shap", "statsmodels", "matplotlib"):
        try:
            m = __import__(mod)
            lines.append(f"{mod:<17}: {getattr(m, '__version__', 'unknown')}")
        except ImportError:
            lines.append(f"{mod:<17}: not installed")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


# =============================================================================
# METRICS
# =============================================================================


def threshold_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                      threshold: float) -> dict:
    """Accuracy, F1, sensitivity, specificity, PPV, NPV at one cut-off."""
    preds = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "acc":  accuracy_score(y_true, preds),
        "f1":   f1_score(y_true, preds, zero_division=0),
        "sens": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "spec": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "ppv":  tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "npv":  tn / (tn + fn) if (tn + fn) > 0 else 0.0,
    }


RANKING_KEYS = ["auc", "auprc", "brier"]
CUTOFF_KEYS = ["acc", "f1", "sens", "spec", "ppv", "npv"]


def bootstrap_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                      thresholds: dict[str, float], n_bootstraps: int = 1000,
                      ci: float = 0.95, seed: int = 42) -> dict[str, dict]:
    """Percentile bootstrap CIs, evaluated at every cut-off in one pass.

    `thresholds` maps a name to a cut-off. The return value maps the same
    names to the usual metric dictionaries. Ranking metrics do not depend on
    the cut-off, so they are computed once per resample and copied into every
    entry rather than recomputed; with 1,000 resamples over ~165,000 rows the
    saving is not cosmetic.

    Resampling is on the row index, so a resample can contain no positive
    case for a rare outcome. Those resamples are skipped rather than scored,
    which makes the effective number of resamples slightly smaller than
    requested for the rarest outcomes. The realised count is returned under
    the key `n_resamples` so the shortfall is visible.
    """
    rng = np.random.RandomState(seed)
    acc = {name: {k: [] for k in RANKING_KEYS + CUTOFF_KEYS}
           for name in thresholds}
    n_used = 0

    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        n_used += 1
        ranking = {
            "auc":   roc_auc_score(yt, yp),
            "auprc": average_precision_score(yt, yp),
            "brier": brier_score_loss(yt, yp),
        }
        for name, thr in thresholds.items():
            for k, v in ranking.items():
                acc[name][k].append(v)
            for k, v in threshold_metrics(yt, yp, thr).items():
                acc[name][k].append(v)

    alpha = (1 - ci) / 2
    out = {}
    for name, store in acc.items():
        out[name] = {k: {"lower": float(np.percentile(v, alpha * 100)),
                         "upper": float(np.percentile(v, (1 - alpha) * 100))}
                     for k, v in store.items() if v}
        out[name]["n_resamples"] = n_used
    return out


def fmt_ci(value: float, ci_dict: dict, key: str, dec: int = 3) -> str:
    f = f"{{:.{dec}f}}"
    return (f"{f.format(value)} "
            f"({f.format(ci_dict[key]['lower'])}-{f.format(ci_dict[key]['upper'])})")


# =============================================================================
# THRESHOLDS
# =============================================================================


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """The cut-off maximising sensitivity + specificity - 1.

    Youden weights a missed case and a false alarm equally. For outcomes with
    a prevalence below one percent that is not a clinically meaningful trade-
    off, and the resulting positive predictive value will be very low. The
    cut-off is reported because it is conventional, not because it is
    recommended; threshold-free measures carry the substantive claims.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[np.argmax(tpr - fpr)])


def transport_threshold(source_scores: np.ndarray, source_threshold: float,
                        target_scores: np.ndarray) -> tuple[float, float]:
    """Move an operating point from one score scale to another by rank.

    WHY THIS IS NEEDED
    ------------------
    The cut-off is chosen on out-of-fold predictions from the uncalibrated
    model, because obtaining out-of-fold predictions from the calibrated
    classifier would require nesting a five-fold calibration inside a five-
    fold split: twenty-five fits per outcome per period instead of five.
    Predictions are then reported on the calibrated scale, so the cut-off has
    to move between two scales.

    WHAT IS TRANSPORTED
    -------------------
    Not the number itself, but the predicted-positive rate it implies. If the
    Youden cut-off flags the top 12.4 percent of out-of-fold scores, the
    transported cut-off is the 87.6th percentile of the calibrated training
    probabilities. This holds under any strictly increasing relabelling of
    the scores, so it does not require the calibrated probability to be a
    monotone function of the raw score - which it is not exactly, because
    `CalibratedClassifierCV(cv=k)` averages k separately calibrated models
    fitted on different folds.

    Averaging does perturb the ranking slightly. The size of that perturbation
    is measured, not assumed: callers should report the Spearman correlation
    between the two scales (see `rank_agreement`).

    Returns (target_threshold, predicted_positive_rate).
    """
    rate = float(np.mean(source_scores >= source_threshold))
    rate = min(max(rate, 1e-9), 1.0)
    target = float(np.quantile(target_scores, 1.0 - rate))
    return target, rate


def rank_agreement(a: np.ndarray, b: np.ndarray, sample: int = 50_000,
                   seed: int = 42) -> float:
    """Spearman correlation between two score vectors over the same rows.

    Reported as the diagnostic supporting `transport_threshold`. Computed on a
    subsample because the rank transform is O(n log n) and the estimate is
    already stable well below the full training size.
    """
    from scipy.stats import spearmanr
    n = len(a)
    if n > sample:
        idx = np.random.RandomState(seed).choice(n, sample, replace=False)
        a, b = a[idx], b[idx]
    rho = spearmanr(a, b).statistic
    return float(rho)


def save_pickle(obj, path: str) -> None:
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    size = os.path.getsize(path) / 1024 ** 2
    print(f"      PKL  -> {path}  ({size:,.1f} MB)")
