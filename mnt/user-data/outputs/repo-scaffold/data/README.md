# Data

No data file is distributed with this repository.

The scripts read two files, placed in the repository root rather than in this
directory, because that is where the configured paths point:

    ./pregnancy by episode cleaned.csv
    ./tmucrd_pregnancy_by_episode.csv

## Expected schema

`ML01_prepare_analytic_set.py` states the requirement and halts if it is not
met. It trains nothing, so it is the cheapest way to check a rebuilt cohort.

Required columns:

- the six outcomes: `c_abortive`, `c_preecl`, `c_preterm`, `c_prom`,
  `c_abrupt`, `c_pph`, each coded 0 or 1
- `age_risk`, `dom`, `subsid`, `n_preg`, with no missing values
- pre-pregnancy diagnosis flags, prefixed `b_`
- antenatal diagnosis flags, prefixed `c_`
- post-delivery diagnosis flags, prefixed `a_`, which are always dropped

Any column that is not a `b_`, `c_`, or `a_` flag and is not one of the named
demographic or metadata columns stops the run. It is not admitted as a
predictor by default.

`par_risk` is derived from `n_preg` inside ML01. `age_risk` is read from the
file and never recomputed, because it is derived in step 4 of the cohort
pipeline and two definitions of the same indicator would diverge without
either being obviously wrong.

## Access

The BPJS Kesehatan claims data were obtained under a data-use agreement.
Their use is governed by Indonesian Law No. 17 of 2023 on Health. Requests go
to BPJS Kesehatan.

TMUCRD is held by Taipei Medical University and requires institutional approval.
Requests go to the TMU Clinical Data Center.
