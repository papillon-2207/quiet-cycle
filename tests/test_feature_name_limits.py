"""Names built from long symptom names or ridge penalties must not exceed the 96-character limit."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from quietcycle import CycleDataset, InputError, Symptom
from quietcycle.dates import add_days, day_difference
from quietcycle.learning import Landmark, LandmarkSpec, MultiFactorPipeline, fit, make_landmark, predict_landmark

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import synthetic_cohort as sc  # noqa: E402 (local example fixture path above)

SUBJECT_FIXTURE = sc.subject_fixture


def long_symptom(subject: str, as_of: str, name: str, impact: int | None, state: str = "present") -> Symptom:
    day = add_days(as_of, -2)
    return Symptom(subject_id=subject, id="long", date=day, recorded_at=day + "T08:00:00Z", utc_offset_minutes=0,
                   symptom=name, state=state, impact=impact)


def cohort(monkeypatch, name: str, training_impact=None):
    """The synthetic cohort with one more symptom in the spec; training_impact(index) adds it to subjects."""
    spec = LandmarkSpec(symptoms=LandmarkSpec().symptoms + (name,))
    def fixture(subject, anchor, index, *, regime="signal"):
        dataset, query = SUBJECT_FIXTURE(subject, anchor, index, regime=regime)
        if training_impact is None or (state := training_impact(index)) is None:
            return dataset, query
        event = long_symptom(subject, query.as_of, name, *state)
        return CycleDataset(events=dataset.events + (event,)), query

    monkeypatch.setattr(sc, "LandmarkSpec", lambda: spec)
    monkeypatch.setattr(sc, "subject_fixture", fixture)
    request, test, inputs, rejected = sc.cohort(train_n=24, tune_n=12, calibration_n=12, test_n=6)
    assert not rejected
    return request, inputs


def with_long_symptom(inputs, name: str, impact: int):
    dataset, query = inputs[0]
    events = tuple(e for e in dataset.events if e.id != "long") + (long_symptom(query.subject_id, query.as_of, name, impact),)
    return CycleDataset(events=events), query


def test_an_untrained_symptom_whose_issue_name_is_too_long_abstains(monkeypatch):
    # "untrained-feature.symptom.<64 characters>.impact" is 97 characters.
    name = "s" * 64
    request, inputs = cohort(monkeypatch, name)
    model = fit(request)
    assert MultiFactorPipeline(model).run(*inputs[0]).status == "estimate"
    dataset, query = with_long_symptom(inputs, name, 2)
    result = MultiFactorPipeline(model).run(dataset, query)
    assert (result.status, result.reason) == ("abstain", "feature_name_too_long")
    assert result.issues == make_landmark(dataset, query, model.spec).issues
    shorter = "s" * 63
    request, inputs = cohort(monkeypatch, shorter)
    result = MultiFactorPipeline(fit(request)).run(*with_long_symptom(inputs, shorter, 2))
    assert result.status == "estimate"
    assert "untrained-feature.symptom." + shorter + ".impact" in result.issues


def test_a_clipped_feature_whose_issue_name_is_too_long_abstains(monkeypatch):
    # "clipped.symptom.<74 characters>.impact" is 97 characters; the symptom varies in training.
    for size, status in ((73, "estimate"), (74, "abstain")):
        name = "s" * size
        request, inputs = cohort(monkeypatch, name, lambda index: (1 if index == 0 else 0,))
        result = MultiFactorPipeline(fit(request)).run(*with_long_symptom(inputs, name, 3))
        assert result.status == status
        if status == "abstain":
            assert result.reason == "feature_name_too_long"
        else:
            assert "clipped.symptom." + name + ".impact" in result.issues


def shifted(landmark: Landmark, days: int) -> Landmark:
    data = landmark.model_dump(mode="json")
    data["anchor_date"] = add_days(data["anchor_date"], days)
    as_of = add_days(data["query"]["as_of"], days)
    data["query"]["as_of"] = as_of
    data["query"]["cutoff"] = as_of + data["query"]["cutoff"][10:]
    return Landmark.model_validate(data)


def test_a_date_outside_the_domain_still_takes_precedence(monkeypatch):
    # Both date-domain checks come before the name check: one for the point date, one for an
    # interval end. Either abstention must stay date-domain-limit.
    name = "s" * 64
    request, inputs = cohort(monkeypatch, name)
    model = fit(request)
    plain = predict_landmark(model, make_landmark(*inputs[0], model.spec))
    radius = plain.intervals[-1].radius_days
    assert plain.status == "estimate" and radius
    long = make_landmark(*with_long_symptom(inputs, name, 2), model.spec)
    to_last_day = day_difference("2200-12-31", plain.point_date)
    for days in (to_last_day + 1, to_last_day, to_last_day - radius + 1):
        result = predict_landmark(model, shifted(long, days))
        assert (result.status, result.reason) == ("abstain", "date-domain-limit"), days
    result = predict_landmark(model, shifted(long, to_last_day - radius))
    assert (result.status, result.reason) == ("abstain", "feature_name_too_long")


@pytest.mark.parametrize("size, pattern, fits", [(81, "never", True), (81, "always", True), (81, "half", False),
                                                 (73, "half", True), (74, "half", False),
                                                 (75, "reported", True), (76, "reported", False)])
def test_fit_rejects_a_missing_indicator_name_that_is_too_long(monkeypatch, size, pattern, fits):
    # "missing.symptom.<74 characters>.impact" is 97 characters. It becomes a term only when the
    # impact is missing for some training subjects and recorded for others. Likewise
    # "missing.symptom.<76 characters>.rate" when the symptom is reported, even without an impact,
    # for some training subjects and not for others.
    name = "s" * size

    def training(index):
        if pattern == "never" or pattern in ("half", "reported") and index % 2 == 0:
            return None
        if pattern == "reported":
            return (None, "present")
        return (2, "present") if index % 4 in (1, 2) else (None, "absent")

    request, _ = cohort(monkeypatch, name, training)
    if fits:
        assert max(len(t) for c in fit(request).candidates for t in c.terms) <= 96
    else:
        with pytest.raises(InputError) as caught:
            fit(request)
        assert caught.value.code == "feature_name_too_long"


@pytest.mark.parametrize("digits, fits", [(81, True), (82, False)])
def test_fit_rejects_a_ridge_penalty_too_long_for_a_candidate_name(digits, fits):
    # "history-ridge-<82 digits>" is 96 characters, but "symptoms-ridge-<82 digits>" is 97.
    request, _, _, _ = sc.cohort(train_n=24, tune_n=12, calibration_n=12, test_n=6)
    penalty = 10 ** (digits - 1)
    request = type(request).model_validate({**dict(request), "ridge_penalties": (1, penalty)})
    if fits:
        assert fit(request).candidates[-1].name == f"combined-ridge-{penalty}"
    else:
        with pytest.raises(InputError) as caught:
            fit(request)
        assert caught.value.code == "ridge_penalty_too_large"
        assert caught.value.__cause__ is None and caught.value.__context__ is None
        assert str(penalty) not in str(caught.value)
        # An error that fit raised before reaching the candidate keeps its precedence.
        with pytest.raises(InputError) as caught:
            fit(type(request).model_validate({**dict(request), "train": request.train + request.train[:1]}))
        assert caught.value.code == "duplicate-cycle-across-or-within-partitions"
