# Changelog

## Unreleased

- A learned forecast abstains with `feature_name_too_long` instead of raising `ValidationError`
  when an `untrained-feature.` or `clipped.` issue for a long symptom name would exceed 96
  characters, and `fit` raises `InputError("feature_name_too_long")` instead of `ValidationError`
  when a `missing.` term a candidate would use is that long (see `docs/TRAINING.md`). Forecasts and
  fits that succeed today are unchanged, and so are the errors that a fit raised before reaching
  that term.

- `from_v1` and `migrate-v1` report an invalid `imported_at` as `invalid_import_time`, the code the
  FHIR adapter uses, instead of raising `ValidationError` with the supplied value
  (`schema_validation_failed` in the CLI). It is now checked even when there are no events, after
  the request itself, so a migration without events no longer accepts an invalid time.

- `from_v1` and `migrate-v1` report a deleted v1 event without `kind` as `schema_validation_failed`
  with its position, as for an event that is not deleted, instead of raising `KeyError`.

- `CSVAdapter` reports a default `kind` that is not hashable, such as a list, dict or set, as
  `schema_validation_failed` with the row number, as it already did for other values of the wrong
  type, instead of raising `TypeError`.

- `scripts/check_artifacts.py` runs Python and Node.js by absolute paths. When either was found
  through a relative `PATH` entry, the script failed with `FileNotFoundError` once its subprocesses
  ran in the temporary workspace.

- `scripts/check_artifacts.py` and `scripts/check_repository.py` exit with an error when assertions
  are disabled (`python -O` or `PYTHONOPTIMIZE`). Their checks are mostly `assert` statements, so
  they used to report passes without running them. `check_artifacts.py` also runs its
  subprocesses without `PYTHONOPTIMIZE`, which `python -E` would otherwise ignore only in itself.

- `scripts/check_repository.py` reads Markdown and `pyproject.toml` as UTF-8, so it no longer fails
  under a locale whose default encoding is not UTF-8, such as cp1252 on Windows. CI now runs it.

- CI pins `actions/checkout` v7.0.1 (`3d3c42e`) and `actions/setup-python` v7.1.0 (`6d4fd22`), which
  target Node.js 24. The v4 and v5 pins target the deprecated Node.js 20, so every job warned that
  they were being forced to run on Node.js 24. The 0.4.0 test report and
  `evidence/actions-pins.json` still describe the run with the old pins.

## 0.4.0 (2026-11-15)

- Compatibility: a forecast from `MultiFactorPipeline.run` is `learning-1.1` when the subject's
  snapshot is valid and its visible events include an input the spec never reads; a bleeding or
  check-in report is always such an input. Fit requests and models with `unit_rules` are
  `learning-1.1` too. Readers that accept only `learning-1.0`, such as 0.3.0, reject these
  documents; all other learning documents are still `learning-1.0`, and the event wire version is
  still 2.0.

- CI installs the test-only `reference` extra (NumPy), fails when any test is skipped
  (`--fail-on-skip`) and runs every check even after an earlier one fails.

- `landmark_attempts` counts, for each elapsed day, the landmark queries a live scheduler would have
  made, how many were eligible, each rejection code, and the outcome code of each attempt made
  before the labels cutoff.

- `dataset_inventory` counts what each subject's snapshot at a labels cutoff holds (onsets, intervals,
  chains, availability, measurement streams and rejected samples, unused inputs) before fitting.

- Rejected labels say which outcome was observed, checked in the order of the table in
  `docs/TRAINING.md`. `right-censored-no-known-outcome` and `unconfirmed-label-continuity` are gone;
  such cases get the first matching code, such as `outcome-right-censored` (an explicit no-onset
  report reaches the landmark day), `outcome-not-observed`, `outcome-interval-censored`,
  `label-excluded-from-history` or a conflict code. Reports that bear on the outcome follow the
  `make_landmark` rules; a continuity claim that names the wrong onset or skips the next one, or a
  second onset on the anchor date, is rejected, and some such cases were labelled exact before. Bad `cohort` or `data_kind` fails the call.

- `label_landmark` and `build_cases` compare `labels_cutoff` as an instant, not as text. A cutoff
  equal to or earlier than the landmark cutoff now fails with `label_cutoff_not_later`; depending
  on its spelling it could be recorded as `right-censored-no-known-outcome`, `incomplete_day`,
  `anchor-revised-after-prediction` or `event_after_query_date`. A later cutoff written with a
  negative offset is no longer rejected. A cutoff that is not a valid instant in 1900-01-01 to
  2200-12-31 raises `invalid_labels_cutoff`, which `build_cases` raises once even when there are
  no queries. A valid cutoff whose local day at a query's offset is after 2200-12-31 gives
  `label_date_outside_domain`, which `build_cases` records for that case.

- `MultiFactorPipeline.run` names the visible input kinds the model's spec never reads in
  `LearnedPrediction.unused_inputs` (names only). Such forecasts are `learning-1.1`.

- `LandmarkSpec.unit_rules` declares unit conversions for new measurements. The rules are stored
  and hashed with the model, so training and every forecast convert the same way. Documents that
  use them are `learning-1.1`; documents without them are unchanged `learning-1.0`.
  `make_landmark` reports landmark validation failures as `InputError`. A spec whose derived feature
  names would exceed 96 characters (metric names over 77, symptom names over 81) is rejected with
  `feature_name_too_long`; such a spec could not produce a valid landmark. A metric listed in `metrics`
  that has a unit rule is limited to 59 characters, so that its forecast issue names stay valid.

- Published JSON Schemas state tuple limits as `minItems`/`maxItems`. They used `minLength`/
  `maxLength`, which JSON Schema applies only to strings. Runtime validation is unchanged.

- `scripts/check_artifacts.py` resolves its temporary directory, so it works behind a symlink
  (macOS `/var`), with Windows short names and with `TMPDIR=.` on Python 3.11, and it exits non-zero
  when a check is skipped.

## 0.3.0 — 2026-09-17

- Add causal fixed-landmark construction, explicit later labels and rejected-case accounting.
- Add exact regularized residual training, held-out blending/selection and separate conformal calibration.
- Add typed JSON model artifacts, prediction explanations and held-out evaluation with abstentions.
- Add fit/forecast/evaluate CLI commands, eleven schemas and optional model-bound HTTP forecast route.
- Check direct landmark scope as well as raw pipeline scope; certify exact ridge normal equations.
- Add synthetic signal/noise/drift stress cases without claiming clinical performance.
- Put the actual implemented formulas and applicability conditions in README.
- Preserve event wire version 2.0; learning artifacts use learning-1.0.


## 0.2.0 — 2026-09-17

Replaced the website-centered JavaScript deliverable with an embeddable Python SDK.
Added typed event contracts, JSON/JSONL/CSV/records/FHIR-subset import, exact unit
normalization, versioned knowledge-time snapshots, injectable feature/predictor
protocols, JSON CLI, optional OpenAPI HTTP adapter, packaged schemas, and wheel/sdist.
Connected normalized measurements to a caller-parameterized duration research model.
Removed the frontend and browser requirement. Preserved 0.1.0 model behavior for
explicitly equivalent requests and kept its source only as a regression reference.
Contract 2.0 is not wire compatible with 1; an explicit migration adapter is included.

This is an alpha engineering handoff, not a clinical release or a PyPI publication.
