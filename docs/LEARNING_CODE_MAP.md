# Learning implementation map

- `learning/contracts.py`: immutable specification, landmark, labelled-case,
  fit/artifact/result models; no clinical-validation flag can be set by an example.
- `learning/landmarks.py`: original-cutoff feature snapshot; separately available
  outcome label; rejected/censored-case accounting.
- `learning/exact.py`: median, exact quantization rounding, rational elimination,
  ridge normal-equation certificate, conformal rank and infinity.
- `learning/fitting.py`: train-only transform, predeclared candidate families,
  tuning blend/selection, frozen calibration and deterministic model digest.
- `learning/prediction.py`: shared scope checks, fixed-landmark and data-availability
  checks, exact contributions, point rounding and interval output.
- `learning/evaluation.py`: test attempts, abstentions, errors, coverage, widths;
  no refitting and no test labels passed into prediction.
- `learning/survival.py`: exact daily forecast from an elapsed day with its residual mass kept,
  conditioning on a later day, pooled and shrunk life-table hazards.
- `learning/scoring.py`: exact probability of a labelled outcome's bounds under a daily forecast,
  and its log score as a float; nothing calls it yet.
- `learning/life_table.py`: pooled life table from complete cycle lengths and a forecast that
  shrinks it toward the person's clean landmark history; a comparator, exact arithmetic.
- `cli.py`: fit / forecast / evaluate from versioned JSON.
- `api.py`: model optionally bound at startup, same pipeline; no train endpoint.
- `tests/test_learning.py`: learning, leakage, mathematical, schema and interface
  regressions. Original tests still cover adapters and state-duration calculations.

README specifies the equations. This map uses function/file identities rather
than line numbers that would become stale after a formatter run.
