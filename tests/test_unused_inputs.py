"""Learned forecasts name the visible inputs that the model's spec never reads."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from quietcycle import Bleeding, Checkin, CycleContext, CycleDataset, Measurement, Symptom, Tombstone
from quietcycle.learning import LearnedPrediction, MultiFactorPipeline, fit
from quietcycle.serialization import canonical_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from synthetic_cohort import cohort  # noqa: E402 (local example fixture path above)


@pytest.fixture(scope="module")
def trained():
    request, test, inputs, rejected = cohort(train_n=12, tune_n=6, calibration_n=6, test_n=2)
    assert not rejected
    return fit(request), inputs[0]


def extras(subject: str) -> tuple:
    day, at = "2025-01-15", "2025-01-15T08:00:00Z"
    common = dict(subject_id=subject, date=day, recorded_at=at, utc_offset_minutes=0)
    return (
        Measurement(id="steps-1", metric="steps", value="9123", unit="{count}", measured_at=at, **common),
        Symptom(id="mood-1", symptom="low_mood", state="present", impact=2, **common),
        Bleeding(id="bleed-1", state="spotting", **common),
        Checkin(id="check-1", state="none", **common),
    )


def test_unused_inputs_are_named_without_values(trained):
    model, (data, query) = trained
    plain = MultiFactorPipeline(model).run(data, query)
    result = MultiFactorPipeline(model).run(CycleDataset(events=data.events + extras(query.subject_id)), query)
    assert result.unused_inputs == ("bleeding", "checkin", "measurement.steps", "symptom.low_mood")
    assert result.schema_version == "learning-1.1"
    assert "9123" not in canonical_json(result)
    assert (result.point_date, result.raw_cycle_length, result.intervals) == (plain.point_date, plain.raw_cycle_length, plain.intervals)


def test_without_unused_inputs_the_forecast_is_learning_1_0(trained):
    model, (data, query) = trained
    result = MultiFactorPipeline(model).run(data, query)
    assert result.schema_version == "learning-1.0"
    assert "unused_inputs" not in canonical_json(result)


def test_abstentions_also_name_unused_inputs(trained):
    model, (data, query) = trained
    paused = query.model_copy(update={"context": CycleContext(mode="natural-cycle", paused=True)})
    result = MultiFactorPipeline(model).run(CycleDataset(events=data.events + extras(query.subject_id)), paused)
    assert (result.status, result.reason) == ("abstain", "paused")
    assert result.unused_inputs == ("bleeding", "checkin", "measurement.steps", "symptom.low_mood")


def test_only_visible_inputs_are_named(trained):
    model, (data, query) = trained
    late = Checkin(subject_id=query.subject_id, id="late", date=query.as_of, recorded_at="2025-03-01T08:00:00Z",
                   utc_offset_minutes=0, state="marked")
    deleted = (Bleeding(subject_id=query.subject_id, id="gone", date="2025-01-15", recorded_at="2025-01-15T08:00:00Z",
                        utc_offset_minutes=0, state="flow"),
               Tombstone(subject_id=query.subject_id, id="gone", revision=2, target_kind="bleeding",
                         recorded_at="2025-01-16T08:00:00Z", utc_offset_minutes=0))
    result = MultiFactorPipeline(model).run(CycleDataset(events=data.events + (late,) + deleted), query)
    assert result.unused_inputs == () and result.schema_version == "learning-1.0"


@pytest.mark.parametrize("version,unused", [("learning-1.1", ()), ("learning-1.0", ("bleeding",))])
def test_version_follows_unused_inputs(trained, version, unused):
    model, (data, query) = trained
    raw = MultiFactorPipeline(model).run(data, query).model_dump(mode="python")
    with pytest.raises(ValidationError, match="contract_version_mismatch"):
        LearnedPrediction.model_validate({**raw, "schema_version": version, "unused_inputs": unused})


def test_long_valid_names_do_not_break_the_forecast(trained):
    model, (data, query) = trained
    long_symptom = Symptom(subject_id=query.subject_id, id="long-s", date="2025-01-15", recorded_at="2025-01-15T08:00:00Z",
                           utc_offset_minutes=0, symptom="s" * 96, state="present")
    plain = MultiFactorPipeline(model).run(data, query)
    result = MultiFactorPipeline(model).run(CycleDataset(events=data.events + (long_symptom,)), query)
    assert (result.status, result.point_date, result.unused_inputs) == ("estimate", plain.point_date, ("symptom." + "s" * 96,))


def test_unlisted_measurements_are_still_unit_checked(trained):
    model, (data, query) = trained
    odd = Measurement(subject_id=query.subject_id, id="odd", date="2025-01-15", recorded_at="2025-01-15T08:00:00Z",
                      measured_at="2025-01-15T07:00:00Z", utc_offset_minutes=0, metric="m" * 96, value="1", unit="{score}")
    result = MultiFactorPipeline(model).run(CycleDataset(events=data.events + (odd,)), query)
    assert (result.status, result.reason, result.unused_inputs) == ("abstain", "unsupported_metric_or_unit", ("measurement." + "m" * 96,))


def conflicting(data, query):
    first = Bleeding(subject_id=query.subject_id, id="dup", date="2025-01-15", recorded_at="2025-01-15T08:00:00Z",
                     utc_offset_minutes=0, state="flow")
    return CycleDataset(events=data.events + (first, first.model_copy(update={"state": "spotting"})))


def test_an_invalid_snapshot_abstains_without_a_list(trained):
    model, (data, query) = trained
    result = MultiFactorPipeline(model).run(conflicting(data, query), query)
    assert (result.status, result.reason, result.schema_version, result.issues) == (
        "abstain", "conflicting_revision", "learning-1.0", ())


def test_an_invalid_snapshot_leaves_another_abstention_unchanged(trained):
    # A paused forecast reports "paused" as before; the snapshot error is not added.
    model, (data, query) = trained
    paused = query.model_copy(update={"context": CycleContext(mode="natural-cycle", paused=True)})
    result = MultiFactorPipeline(model).run(conflicting(data, query), paused)
    assert (result.reason, result.schema_version, result.unused_inputs, result.issues) == (
        "paused", "learning-1.0", (), ())


def test_published_schema_rejects_duplicate_unused_inputs(trained):
    jsonschema = pytest.importorskip("jsonschema")
    model, (data, query) = trained
    raw = MultiFactorPipeline(model).run(data, query).model_dump(mode="json")
    document = {**raw, "schema_version": "learning-1.1", "unused_inputs": ["bleeding", "bleeding"]}
    assert not jsonschema.Draft202012Validator(LearnedPrediction.model_json_schema()).is_valid(document)
    assert jsonschema.Draft202012Validator(LearnedPrediction.model_json_schema()).is_valid(
        {**document, "unused_inputs": ["bleeding"]})


@pytest.mark.parametrize("unused", [("checkin", "bleeding"), ("bleeding", "bleeding"), ("note.x",)])
def test_unused_inputs_must_be_sorted_unique_kind_names(trained, unused):
    model, (data, query) = trained
    raw = MultiFactorPipeline(model).run(data, query).model_dump(mode="python")
    with pytest.raises(ValidationError):
        LearnedPrediction.model_validate({**raw, "schema_version": "learning-1.1", "unused_inputs": unused})
