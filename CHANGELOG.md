# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
read here as: the major version changes when a reported number would change.

## [1.0.0] — TODO date

First archived release. Corresponds to the submitted manuscript.

### Added
- `ML01`–`ML09`, the analysis pipeline.
- `ML05b`, `ML05c`, self-contained checks that are not read by `ML09`.
- `ml_core.py`, one definition of how a model is fitted and scored, shared by
  `ML02`–`ML06`.
- `ml_utils.py`, benchmarking, metrics, and threshold derivation.

### Changed from the analysis notebooks
These are the differences between this repository and the notebooks the work
was originally done in. Each changes a reported number.

- The classification cut-off is derived from out-of-fold predictions within
  the training split, not from the Youden index on the test split. The
  test-reoptimised cut-off is reported alongside as an upper bound.
- SMOTE is applied inside each cross-validation fold, and the calibrator is
  fitted on data at the original prevalence rather than on resampled data.
  `SMOTE_INSIDE_FOLDS = False` reproduces the notebook placement.
- The feature set is verified after the subtraction that builds it. An
  unrecognised column halts the run instead of entering the model.
- `use_label_encoder` is no longer passed to `XGBClassifier`. This changes the
  warning output and nothing else.
