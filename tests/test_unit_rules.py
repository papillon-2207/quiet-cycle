"""Unit rules declared in LandmarkSpec travel with the model from import to prediction.

The respiratory-rate cohort is synthetic: it shows the pipeline can learn a supplied relation,
not that respiratory rate predicts menstrual onset in people.
"""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction

import pytest
from pydantic import ValidationError

from quietcycle import CycleContext, CycleDataset, CyclePipeline, InputError, Measurement, NoOnset, Onset, Query, Rational
from quietcycle.adapters import CSVAdapter, from_records, to_csv
from quietcycle.dates import add_days
from quietcycle.learning import (
    FitRequest, FittedModel, LandmarkSpec, MultiFactorPipeline, UnitRuleSpec, build_cases, fit, make_landmark,
    predict_landmark,
)
from quietcycle.serialization import canonical_json, loads


def rule(unit: str, scale: Fraction, canonical: str = "/min", metric: str = "respiratory_rate") -> UnitRuleSpec:
    return UnitRuleSpec(metric=metric, unit=unit, canonical_unit=canonical, scale=Rational.from_fraction(scale))


PER_MINUTE, PER_HOUR = rule("/min", Fraction(1)), rule("/h", Fraction(1, 60))


def spec(*rules: UnitRuleSpec) -> LandmarkSpec:
    return LandmarkSpec(metrics=("respiratory_rate",), symptoms=(), unit_rules=rules or (PER_MINUTE, PER_HOUR))


def subject(name: str, anchor: str, index: int, *, unit: str = "/min") -> tuple[CycleDataset, Query]:
    base, signal = 28 + index % 4, index % 5 - 2
    events: list = []
    for j in range(5):
        day = add_days(anchor, -(4 - j) * base)
        events.append(Onset(subject_id=name, id=f"o{j}", date=day, recorded_at=day + "T08:00:00Z", utc_offset_minutes=0,
                            certainty="confirmed", previous_onset_id=f"o{j - 1}" if j else None,
                            continuity="confirmed" if j else "unknown"))
    as_of = add_days(anchor, 20)
    events.append(NoOnset(subject_id=name, id="n", anchor_id="o4", through=add_days(as_of, -1),
                          recorded_at=as_of + "T08:00:00Z", utc_offset_minutes=0))
    for elapsed in range(10, 20):
        day = add_days(anchor, elapsed)
        per_minute = 15 + (signal if elapsed >= 17 else 0)
        events.append(Measurement(subject_id=name, id=f"r{elapsed}", date=day, recorded_at=day + "T08:00:00Z",
                                  measured_at=day + "T07:00:00Z", utc_offset_minutes=0, metric="respiratory_rate",
                                  value=str(per_minute * 60 if unit == "/h" else per_minute), unit=unit))
    end = add_days(anchor, base + 2 * signal)
    events.append(Onset(subject_id=name, id="o5", date=end, recorded_at=end + "T08:00:00Z", utc_offset_minutes=0,
                        certainty="confirmed", previous_onset_id="o4", continuity="confirmed"))
    query = Query(subject_id=name, as_of=as_of, cutoff=as_of + "T12:00:00Z", utc_offset_minutes=0,
                  context=CycleContext(mode="natural-cycle"))
    return CycleDataset(events=tuple(events)), query


@pytest.fixture(scope="module")
def trained():
    groups, inputs = [], []
    for k, (phase, count, anchor, labels) in enumerate((
            ("train", 12, "2023-06-01", "2023-09-01T00:00:00Z"), ("tune", 6, "2024-01-01", "2024-04-01T00:00:00Z"),
            ("calibration", 6, "2024-07-01", "2024-10-01T00:00:00Z"), ("test", 4, "2025-01-01", "2025-04-01T00:00:00Z"))):
        cases = []
        for i in range(count):
            dataset, query = subject(f"{phase}-{i}", anchor, i + 7 * k, unit="/h" if i % 2 else "/min")
            dataset = CSVAdapter().loads(to_csv(dataset))
            built = build_cases(dataset, (query,), spec=spec(), labels_cutoff=labels, data_kind="synthetic", cohort="rr")
            assert not built.rejected
            cases.extend(built.cases)
            inputs.append((dataset, query))
        groups.append(tuple(cases))
    request = FitRequest(schema_version="learning-1.1", spec=spec(), train=groups[0], tune=groups[1], calibration=groups[2])
    return request, fit(request), groups[3], inputs[-4:]


def test_declared_unit_reaches_training_json_and_prediction(trained, tmp_path):
    request, model, test, inputs = trained
    assert model.schema_version == "learning-1.1"
    assert "measurement.respiratory_rate.delta" in model.selected.terms
    restored = FittedModel.model_validate(loads(canonical_json(model)))
    assert restored == model
    result = MultiFactorPipeline(restored).run(*inputs[0])
    assert result.status == "estimate"
    assert result == predict_landmark(model, test[0].landmark)
    paths = {name: tmp_path / f"{name}.json" for name in ("fit", "model", "request")}
    paths["fit"].write_text(canonical_json(request))
    paths["request"].write_text(canonical_json({"dataset": inputs[0][0], "query": inputs[0][1]}))
    subprocess.run([sys.executable, "-m", "quietcycle", "fit", str(paths["fit"]), "-o", str(paths["model"])], check=True)
    raw = subprocess.check_output([sys.executable, "-m", "quietcycle", "forecast", str(paths["request"]),
                                   "--model", str(paths["model"])], text=True)
    assert json.loads(raw) == result.model_dump(mode="json")


def test_rows_and_csv_and_both_units_give_the_same_features():
    hourly, query = subject("p", "2025-01-01", 3, unit="/h")
    minute, _ = subject("p", "2025-01-01", 3, unit="/min")
    landmarks = [make_landmark(d, query, spec()) for d in (
        hourly, minute, CSVAdapter().loads(to_csv(hourly)),
        from_records([e.model_dump(mode="json") for e in hourly.events]))]
    assert all(lm.values == landmarks[0].values for lm in landmarks)
    # index 3: signal +1, so the last three completed days read 16 breaths per minute.
    assert next(v for v in landmarks[0].values if v.name == "measurement.respiratory_rate.recent").value.as_fraction() == 16


def test_undeclared_unit_abstains_and_baseline_still_fails_closed(trained):
    _, model, _, _ = trained
    dataset, query = subject("p", "2025-01-01", 3, unit="1/s")
    result = MultiFactorPipeline(model).run(dataset, query)
    assert (result.status, result.reason) == ("abstain", "unsupported_metric_or_unit")
    with pytest.raises(InputError, match="unsupported_metric_or_unit"):
        CyclePipeline().run(*subject("p", "2025-01-01", 3))


def test_other_rules_produce_another_spec(trained):
    _, model, _, _ = trained
    dataset, query = subject("p", "2025-01-01", 3)
    other = make_landmark(dataset, query, spec(PER_MINUTE))
    assert predict_landmark(model, other).reason == "landmark-spec-mismatch"


def test_rule_identical_to_a_builtin_is_accepted_as_a_no_op():
    builtin = rule("/min", Fraction(1), metric="resting_heart_rate")
    with_builtin = LandmarkSpec(metrics=("respiratory_rate", "resting_heart_rate"), symptoms=(),
                                unit_rules=(PER_MINUTE, PER_HOUR, builtin))
    without = LandmarkSpec(metrics=("respiratory_rate", "resting_heart_rate"), symptoms=(), unit_rules=(PER_MINUTE, PER_HOUR))
    dataset, query = subject("p", "2025-01-01", 3)
    assert make_landmark(dataset, query, with_builtin).values == make_landmark(dataset, query, without).values


@pytest.mark.parametrize("build,code", [
    (lambda: spec(PER_MINUTE, rule("/s", Fraction(1), canonical="/s")), "metric_has_two_canonical_units"),
    (lambda: spec(PER_MINUTE, PER_MINUTE), "duplicate_unit_rule"),
    (lambda: spec(rule("1/s", Fraction(59), metric="resting_heart_rate")), "unit_rule_override_forbidden"),
    (lambda: rule("/h", Fraction(0)), "nonpositive_unit_scale"),
    (lambda: rule("/h", Fraction(1, 2 ** 64)), "unit_rule_precision_budget"),
    (lambda: UnitRuleSpec(metric="respiratory_rate", unit="/h", canonical_unit="/min", scale=Rational.from_fraction(Fraction(1)),
                          offset=Rational.from_fraction(Fraction(-(2 ** 64)))), "unit_rule_precision_budget"),
    (lambda: rule("/min", Fraction(2)), "inconsistent_identity_unit_rule"),
    (lambda: UnitRuleSpec(metric="respiratory_rate", unit="/min", canonical_unit="/min", scale=Rational.from_fraction(Fraction(1)),
                          offset=Rational.from_fraction(Fraction(1))), "inconsistent_identity_unit_rule"),
    (lambda: spec(PER_MINUTE, UnitRuleSpec(metric="body_temperature", unit="[degF]", canonical_unit="Cel",
                                           scale=Rational.from_fraction(Fraction(5, 9)),
                                           offset=Rational.from_fraction(Fraction(-160, 9) + Fraction(1, 10 ** 9)))),
     "unit_rule_override_forbidden"),
    (lambda: spec(PER_MINUTE, rule("/min", Fraction(1), canonical="bpm", metric="resting_heart_rate")),
     "unit_rule_override_forbidden"),
    (lambda: LandmarkSpec(metrics=("wrist_temperature",), symptoms=(),
                          unit_rules=(rule("mK", Fraction(1, 1000), canonical="K", metric="wrist_temperature"),)),
     "metric_has_two_canonical_units"),
    (lambda: LandmarkSpec(metrics=(), symptoms=(), unit_rules=(PER_MINUTE, rule("/s", Fraction(1), canonical="/s"))),
     "metric_has_two_canonical_units"),
    (lambda: spec(*(rule(f"u{i}", Fraction(1, i + 2)) for i in range(33))), "at most 32 items"),
])
def test_invalid_unit_rules(build, code):
    with pytest.raises(ValidationError, match=code):
        build()


def test_derived_feature_names_fit_the_identifier_limit():
    # "measurement.<metric>.recent" and "symptom.<symptom>.impact" are the longest names, at most 96.
    LandmarkSpec(metrics=("m" * 77,), symptoms=("s" * 81,))
    for metrics, symptoms in ((("m" * 78,), ()), ((), ("s" * 82,))):
        with pytest.raises(ValidationError, match="feature_name_too_long"):
            LandmarkSpec(metrics=metrics, symptoms=symptoms)


def test_a_declared_metric_name_fits_its_forecast_issues():
    # A metric with a rule can carry data, so "untrained-feature.measurement.<metric>.recent" must fit
    # in 96 characters: at most 59 for the metric name.
    def declared(name: str) -> LandmarkSpec:
        return LandmarkSpec(metrics=(name,), symptoms=(), unit_rules=(rule("/min", Fraction(1), metric=name),))
    declared("m" * 59)
    with pytest.raises(ValidationError, match="feature_name_too_long"):
        declared("m" * 60)


def test_64_bit_scale_and_offset_are_allowed():
    edge = Fraction(2 ** 64 - 1)
    assert UnitRuleSpec(metric="m", unit="a", canonical_unit="b", scale=Rational.from_fraction(1 / edge),
                        offset=Rational.from_fraction(-edge)).scale.as_fraction() == 1 / edge


def test_scale_and_offset_convert_exactly():
    # canonical = value / 2 + 3, and the last three completed days read 16: 16 / 2 + 3 = 11.
    custom = UnitRuleSpec(metric="respiratory_rate", unit="half", canonical_unit="/min",
                          scale=Rational.from_fraction(Fraction(1, 2)), offset=Rational.from_fraction(Fraction(3)))
    dataset, query = subject("p", "2025-01-01", 3, unit="half")
    landmark = make_landmark(dataset, query, spec(PER_MINUTE, custom))
    assert next(v for v in landmark.values if v.name == "measurement.respiratory_rate.recent").value.as_fraction() == 11


def test_contract_version_follows_unit_rules(trained):
    request, model, _, _ = trained
    with pytest.raises(ValidationError, match="contract_version_mismatch"):
        FitRequest.model_validate({**request.model_dump(mode="python"), "schema_version": "learning-1.0"})
    with pytest.raises(ValidationError, match="contract_version_mismatch"):
        FittedModel.model_validate({**model.model_dump(mode="python"), "schema_version": "learning-1.0"})
    legacy = {**request.model_dump(mode="python"), "spec": LandmarkSpec().model_dump(mode="python")}
    with pytest.raises(ValidationError, match="contract_version_mismatch"):
        FitRequest.model_validate(legacy)
    assert "unit_rules" not in canonical_json(LandmarkSpec())


def test_feature_precision_overflow_is_a_recorded_rejection():
    # Four coprime 63-bit scales: recent days pair u0/u1, reference days pair u2/u3, so the delta's
    # denominator needs about 293 bits, over the landmark's 256-bit budget.
    primes = (2 ** 63 - 25, 2 ** 63 - 165, 2 ** 63 - 259, 2 ** 63 - 301)
    wide = spec(PER_MINUTE, *(rule(f"u{i}", Fraction(1, p)) for i, p in enumerate(primes)))
    dataset, query = subject("p", "2025-01-01", 3)
    events = [e for e in dataset.events if e.kind != "measurement"]
    for elapsed in range(10, 20):
        day = add_days("2025-01-01", elapsed)
        for unit in (("u0", "u1") if elapsed >= 17 else ("u2", "u3")):
            events.append(Measurement(subject_id="p", id=f"{unit}-{elapsed}", date=day, recorded_at=day + "T08:00:00Z",
                                      measured_at=day + "T07:00:00Z", utc_offset_minutes=0, metric="respiratory_rate",
                                      value="0.000000000007", unit=unit))
    data = CycleDataset(events=tuple(events))
    with pytest.raises(InputError, match="feature_precision_budget") as raised:
        make_landmark(data, query, wide)
    # The Pydantic error echoes input values, so it must not stay attached to the InputError.
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
    built = build_cases(data, (query,), spec=wide, labels_cutoff="2025-04-01T00:00:00Z", data_kind="synthetic", cohort="c")
    assert built.rejected == (("p", query.as_of, "feature_precision_budget"),)
