# Incremental value of antenatal diagnoses for outcome-specific prediction of pregnancy complications

Analysis code for a study of whether diagnoses recorded during pregnancy add
predictive value over pre-pregnancy history alone, in six complications:
abortive outcome, preeclampsia, preterm birth, premature rupture of membranes,
placental abruption, and postpartum haemorrhage.

Models are fitted on pregnancy episodes reconstructed from Indonesian national
health insurance (BPJS Kesehatan) claims and validated without refitting on the
Taipei Medical University Clinical Research Database (TMUCRD).

<!-- TODO: badges once the DOI exists
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

---

## What this repository is and is not

It contains the code that produced every table and figure in the manuscript.
It does not contain the data. Neither source dataset can be redistributed. See
[Data availability](#data-availability).

Nothing here reconstructs the cohort. The episode-level file this code reads is
the output of a separate pipeline, at
[dca-nug/BPJS-Preg](https://github.com/dca-nug/BPJS-Preg)
<!-- TODO: add its Zenodo concept DOI -->.

---

## Installation

```bash
git clone https://github.com/dca-nug/ML-Preg.git
cd ML-Preg
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or later. The reported run used Python 3.11.4 on Windows 10.

To reproduce the reported numbers rather than run the code on a current stack,
install the pinned environment instead:

```bash
pip install -r requirements-lock.txt
```

---

## Input files

Both are placed in the repository root. Neither is included.

| File | Rows | Produced by |
|---|---|---|
| `pregnancy by episode cleaned.csv` | 551,488 episodes | step 5 of the cohort pipeline |
| `tmucrd_pregnancy_by_episode.csv` | external cohort | TMUCRD extraction, same schema |

`ML01_prepare_analytic_set.py` halts with a fatal error if the schema does not
match. Run it first on any rebuilt cohort. It trains nothing and takes seconds.

---

## Running the analysis

Scripts are run in order from the repository root. Each writes to its own
`output_ML0X/` directory and reads only from directories written by an earlier
step.

```bash
python ML01_prepare_analytic_set.py        # schema checks, feature manifest
python ML02_train_xgb_main.py              # primary models, SHAP
python ML03_train_comparison_models.py     # five comparison learners
python ML04_train_xgb_smote.py             # sensitivity: SMOTE
python ML05_train_multilabel.py            # sensitivity: multilabel
python ML06_subgroup_age.py                # subgroup: maternal age
python ML07_external_validate_tmucrd.py    # external validation
python ML08_population_characteristics.py  # descriptive tables
python ML09_extract_results.py             # manuscript tables and figures
```

ML03 is the longest job. It accepts a switch so the five learners can be run
separately:

```bash
python ML03_train_comparison_models.py --model lgbm
```

Two further scripts are self-contained and are not read by ML09. They answer
specific methodological questions and are not part of the reported analysis:

```bash
python ML05b_sensitivity_o20.py            # antenatal O20 withheld
python ML05c_within_woman_check.py         # test episodes from unseen women
```

### Dependencies between steps

| Script | Reads |
|---|---|
| ML01 | the episode-level CSV |
| ML02–ML06, ML08 | the episode-level CSV, through ML01 |
| ML05b, ML05c | `output_ML02/`, plus the CSV |
| ML07 | `output_ML02/`, plus the TMUCRD CSV |
| ML09 | `output_ML02/` through `output_ML07/` |

ML09 fits nothing. Every number it writes comes from predictions already stored.

---

## Shared modules

| Module | Contents |
|---|---|
| `ML01_prepare_analytic_set.py` | the one definition of the feature set, imported by every training script |
| `ml_core.py` | the one definition of how a model is fitted and scored |
| `ml_utils.py` | benchmarking, metrics, threshold derivation |

The feature set is built by subtraction and then verified. A column present in
the input that the pipeline has no rule for stops the run instead of entering
the model.

---

## Two classification cut-offs

Every metric that depends on an operating point is reported at both.

- `train_oof` is derived from out-of-fold predictions within the training split
  and transported to the reported probability scale by its predicted-positive
  rate. This is the primary cut-off and the one carried to TMUCRD.
- `test_reopt` is the Youden optimum recomputed on the test split. It is
  reported as an upper bound. The gap between the two is how much of the
  operating-point performance comes from choosing the operating point on the
  data used to evaluate it.

AUC, AUPRC, Brier, and everything downstream of them do not depend on a
cut-off. Table 5 of `output_ML09/` quantifies the gap.

---

## Reproducing the reported run

`requirements.txt` states floors. `requirements-lock.txt` states the versions
the reported numbers were produced with, transcribed from
`output_ML02/environment_ML02.txt`:

    Python 3.11.4, numpy 1.26.4, pandas 2.3.0, scipy 1.16.1,
    scikit-learn 1.7.1, xgboost 3.0.4, lightgbm 4.6.0, catboost 1.2.10,
    imbalanced-learn 0.14.0, shap 0.48.0, statsmodels 0.14.5,
    matplotlib 3.10.6

XGBoost and scikit-learn have both changed default behaviour between minor
versions in ways that move the third decimal of an AUC, so the two files are
not interchangeable.

Every script writes its own `output_ML0X/environment_ML0X.txt` at the start of
the run, before anything is fitted.

`environment_report` does not record `openpyxl`, `joblib`, or `psutil`, so
those three are given as floors in the lock file rather than as transcriptions.

Random seeds are fixed in each script (`RANDOM_STATE = 123` for splits and
learners, `seed = 42` for the bootstrap). Results are reproducible on a fixed
environment. They are not guaranteed to be bit-identical across library
versions or across the number of threads XGBoost is given.

---

## Runtime

Wall-clock time and peak memory per stage are written to
`output_ML0X/benchmark_ML0X.csv` on every run. Peak memory is the maximum
resident set size observed at the start and end of a stage, so it is a floor
rather than a true peak.

<!-- TODO: fill from your own benchmark_*.csv files
| Script | Wall clock | Peak memory |
|---|---|---|
-->

---

## Data availability

The BPJS Kesehatan claims data are not publicly available. They were obtained
under a data-use agreement with BPJS Kesehatan and their use is governed by
Indonesian Law No. 17 of 2023 on Health. Requests go to BPJS Kesehatan.

TMUCRD is held by Taipei Medical University and requires institutional approval
for access. Requests go to the TMU Clinical Data Center.

Neither dataset is redistributed here, and neither can be reconstructed from
this repository.

<!-- TODO: IRB / ethics approval numbers -->
Ethics approval: `TODO`.

---

## Citation

If you use this code, cite the archived release and the article.

> Nugroho DCA, Muhtar MS, Triastuti IA, Hsu JC, Lee YCG, Su ECY.
> Incremental Value of Antenatal Diagnoses for Outcome-Specific Prediction of
> Pregnancy Complications: A Nationwide Claims Analysis.

Machine-readable metadata is in `CITATION.cff`.


---

## License

MIT, see [LICENSE](LICENSE). The license covers the code in this repository.
It does not extend to either dataset.
