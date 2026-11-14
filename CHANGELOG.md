# Changelog

## Unreleased

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
