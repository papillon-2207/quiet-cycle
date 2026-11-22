"""Versioned contracts for a fitted, fixed-landmark residual model.

Artifacts contain fitted statistics, not executable pickles. They remain sensitive:
subject hashes and small-sample coefficients are not anonymization.
"""
from __future__ import annotations

from datetime import datetime
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, Field, StrictBool, StrictInt, StringConstraints, model_validator

from ..dates import day_difference
from ..models import CivilDate, Identifier, Instant, Model, Query, Rational, Unit, _tuple, array_bounds
from ..units import DEFAULT_RULES

Hash = Annotated[str, StringConstraints(strict=True, pattern=r"^[a-f0-9]{64}$")]
Names = Annotated[tuple[Identifier, ...], BeforeValidator(_tuple)]


def _zero() -> Rational:
    return Rational(numerator="0", denominator="1")


class UnitRuleSpec(Model):
    """Exact conversion declared with the spec: canonical value = value * scale + offset."""
    metric: Identifier
    unit: Unit
    canonical_unit: Unit
    scale: Rational
    offset: Rational = Field(default_factory=_zero)

    @model_validator(mode="after")
    def valid(self) -> Self:
        if self.scale.as_fraction() <= 0:
            raise ValueError("nonpositive_unit_scale")
        # Otherwise two samples labelled with the same canonical unit could mean different things.
        if self.unit == self.canonical_unit and (self.scale.as_fraction(), self.offset.as_fraction()) != (Fraction(1), Fraction(0)):
            raise ValueError("inconsistent_identity_unit_rule")
        # Bounded so that feature arithmetic over a few samples stays within exact-number budgets.
        if any(max(abs(f.numerator).bit_length(), f.denominator.bit_length()) > 64
               for f in (self.scale.as_fraction(), self.offset.as_fraction())):
            raise ValueError("unit_rule_precision_budget")
        return self

    def same_as_builtin(self) -> bool:
        builtin = DEFAULT_RULES.get((self.metric, self.unit))
        return builtin is not None and (builtin.canonical_unit, builtin.scale, builtin.offset) == (
            self.canonical_unit, self.scale.as_fraction(), self.offset.as_fraction())


def _empty(value: tuple[object, ...]) -> bool:
    return value == ()


class LandmarkSpec(Model):
    # Elapsed days, zero at onset. An artifact only applies at this landmark.
    elapsed_day: Annotated[StrictInt, Field(ge=1, le=180)] = 20
    min_history: Annotated[StrictInt, Field(ge=1, le=30)] = 3
    max_history: Annotated[StrictInt, Field(ge=1, le=30)] = 12
    recent_days: Annotated[StrictInt, Field(ge=1, le=14)] = 3
    reference_days: Annotated[StrictInt, Field(ge=1, le=30)] = 7
    minimum_measurement_days: Annotated[StrictInt, Field(ge=1, le=14)] = 2
    metrics: Names = ("wrist_temperature", "resting_heart_rate", "hrv_rmssd", "sleep_duration")
    symptoms: Names = ("headache", "fatigue", "abdominal_pain", "gastrointestinal")
    # learning-1.1 only. Omitted when empty, so learning-1.0 specs keep their bytes and digests.
    unit_rules: Annotated[tuple[UnitRuleSpec, ...], BeforeValidator(_tuple),
                          Field(max_length=32, json_schema_extra=array_bounds)] = Field(default=(), exclude_if=_empty)

    @model_validator(mode="after")
    def valid(self) -> Self:
        keys = [(r.metric, r.unit) for r in self.unit_rules]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate_unit_rule")
        if any(key in DEFAULT_RULES and not r.same_as_builtin() for key, r in zip(keys, self.unit_rules)):
            raise ValueError("unit_rule_override_forbidden")
        for metric in sorted({r.metric for r in self.unit_rules}):
            units = {rule.canonical_unit for (name, _), rule in DEFAULT_RULES.items() if name == metric}
            units |= {r.canonical_unit for r in self.unit_rules if r.metric == metric}
            if len(units) > 1:
                raise ValueError("metric_has_two_canonical_units")
        if self.min_history > self.max_history:
            raise ValueError("invalid_history_window")
        if self.recent_days + self.reference_days > self.elapsed_day:
            raise ValueError("landmark_before_observation_windows")
        if self.minimum_measurement_days > min(self.recent_days, self.reference_days):
            raise ValueError("measurement_days_exceed_window")
        if len(self.metrics) > 8 or len(self.symptoms) > 8:
            raise ValueError("feature_budget")
        if len(set(self.metrics)) != len(self.metrics) or len(set(self.symptoms)) != len(self.symptoms):
            raise ValueError("duplicate_feature")
        # A derived name over the identifier limit would make every landmark of this spec invalid.
        if any(len(name) > 96 for name in expected_feature_names(self)):
            raise ValueError("feature_name_too_long")
        # A declared metric can carry data, so its longest forecast issue must be an identifier too.
        declared = {r.metric for r in self.unit_rules}
        if any(len(f"untrained-feature.measurement.{m}.recent") > 96 for m in self.metrics if m in declared):
            raise ValueError("feature_name_too_long")
        return self


def expected_feature_names(spec: LandmarkSpec) -> tuple[str, ...]:
    names = ["history.median", "history.last", "history.mad", "history.change", "history.count"]
    names.extend(f"measurement.{metric}.{stat}" for metric in spec.metrics for stat in ("recent", "delta"))
    names.extend(f"symptom.{symptom}.{stat}" for symptom in spec.symptoms for stat in ("rate", "impact"))
    return tuple(sorted(names))


class FeatureValue(Model):
    name: Identifier
    value: Rational | None


Values = Annotated[tuple[FeatureValue, ...], BeforeValidator(_tuple), Field(max_length=64, json_schema_extra=array_bounds)]


class Landmark(Model):
    subject_id: Identifier
    anchor_id: Identifier
    anchor_date: CivilDate
    query: Query
    spec_sha256: Hash
    snapshot_sha256: Hash
    baseline_length: Rational
    values: Values
    issues: Names = ()

    @model_validator(mode="after")
    def valid(self) -> Self:
        if self.subject_id != self.query.subject_id:
            raise ValueError("subject_mismatch")
        if self.anchor_date >= self.query.as_of:
            raise ValueError("invalid_anchor")
        names = [x.name for x in self.values]
        if names != sorted(set(names)):
            raise ValueError("feature_order")
        for item in self.values:
            if item.value is not None:
                f = item.value.as_fraction()
                if max(abs(f.numerator).bit_length(), f.denominator.bit_length()) > 256:
                    raise ValueError("feature_precision_budget")
        if self.baseline_length.as_fraction() <= 0:
            raise ValueError("invalid_baseline")
        historical = next((v.value for v in self.values if v.name == "history.median"), None)
        if historical is None or historical != self.baseline_length:
            raise ValueError("baseline_history_mismatch")
        return self


class LabelledCase(Model):
    landmark: Landmark
    outcome_date: CivilDate
    outcome_known_at: Instant
    # This describes supplied data, not a claim made by the library.
    data_kind: Literal["synthetic", "observational"]
    cohort: Identifier

    @model_validator(mode="after")
    def valid(self) -> Self:
        q = self.landmark.query
        if self.outcome_date < q.as_of or self.outcome_known_at <= q.cutoff:
            raise ValueError("outcome_not_future_to_prediction")
        # The onset's date is in its own offset, which a case does not keep. Reject only a date that
        # had not begun anywhere, even at +14:00, when the outcome became known.
        if datetime.fromisoformat(self.outcome_known_at) < datetime.fromisoformat(self.outcome_date + "T00:00:00+14:00"):
            raise ValueError("label_available_before_event")
        return self

    @property
    def length(self) -> int:
        return day_difference(self.outcome_date, self.landmark.anchor_date)


Cases = Annotated[tuple[LabelledCase, ...], BeforeValidator(_tuple), Field(max_length=5000, json_schema_extra=array_bounds)]


LearningVersion = Literal["learning-1.0", "learning-1.1"]


def _check_version(schema_version: str, spec: LandmarkSpec) -> None:
    # CONTRACT.md: a contract change needs a new schema version. Unit rules are that change.
    if (schema_version == "learning-1.1") != bool(spec.unit_rules):
        raise ValueError("contract_version_mismatch")


class FitRequest(Model):
    schema_version: LearningVersion = "learning-1.0"
    spec: LandmarkSpec = Field(default_factory=LandmarkSpec)
    train: Cases
    tune: Cases
    calibration: Cases
    ridge_penalties: Annotated[tuple[StrictInt, ...], BeforeValidator(_tuple)] = (1, 10, 100)
    split_mode: Literal["new-subject", "forward-personal"] = "new-subject"
    # Clipping is a numerical robustness choice, not a physiological normal range.
    clipping: Annotated[StrictInt, Field(ge=1, le=100)] = 8
    quantization: Annotated[StrictInt, Field(ge=1, le=1000000)] = 1000

    @model_validator(mode="after")
    def valid(self) -> Self:
        _check_version(self.schema_version, self.spec)
        if not 1 <= len(self.ridge_penalties) <= 8 or any(x <= 0 for x in self.ridge_penalties):
            raise ValueError("positive_ridge_penalties_required")
        if tuple(sorted(set(self.ridge_penalties))) != self.ridge_penalties:
            raise ValueError("penalties_must_be_sorted_unique")
        if len(self.train) < 8 or len(self.tune) < 4 or len(self.calibration) < 1:
            raise ValueError("insufficient_development_cases")
        return self


class ScaleColumn(Model):
    name: Identifier
    center: Rational
    scale: Rational
    observed_in_train: StrictBool

    @model_validator(mode="after")
    def valid(self) -> Self:
        if self.scale.as_fraction() <= 0:
            raise ValueError("nonpositive_scale")
        return self


class LinearCandidate(Model):
    name: Identifier
    penalty: Annotated[StrictInt, Field(ge=0)]
    terms: Names
    coefficients: Annotated[tuple[Rational, ...], BeforeValidator(_tuple), Field(max_length=128, json_schema_extra=array_bounds)]
    blend: Rational
    tune_mse: Rational

    @model_validator(mode="after")
    def valid(self) -> Self:
        if len(self.coefficients) != len(self.terms) or len(set(self.terms)) != len(self.terms):
            raise ValueError("coefficient_shape")
        if not 0 <= self.blend.as_fraction() <= 1 or self.tune_mse.as_fraction() < 0:
            raise ValueError("invalid_candidate")
        return self


class FittedModel(Model):
    schema_version: LearningVersion = "learning-1.0"
    algorithm: Literal["exact-landmark-ridge-ensemble-v1"] = "exact-landmark-ridge-ensemble-v1"
    spec: LandmarkSpec
    columns: Annotated[tuple[ScaleColumn, ...], BeforeValidator(_tuple), Field(max_length=64, json_schema_extra=array_bounds)]
    clipping: Annotated[StrictInt, Field(ge=1, le=100)]
    quantization: Annotated[StrictInt, Field(ge=1, le=1000000)]
    selected: LinearCandidate
    candidates: Annotated[tuple[LinearCandidate, ...], BeforeValidator(_tuple), Field(max_length=33, json_schema_extra=array_bounds)]
    calibration_scores: Annotated[tuple[StrictInt, ...], BeforeValidator(_tuple)]
    train_count: Annotated[StrictInt, Field(ge=8)]
    tune_count: Annotated[StrictInt, Field(ge=4)]
    calibration_count: Annotated[StrictInt, Field(ge=1)]
    train_subject_hashes: Annotated[tuple[Hash, ...], BeforeValidator(_tuple)]
    tune_subject_hashes: Annotated[tuple[Hash, ...], BeforeValidator(_tuple)]
    calibration_subject_hashes: Annotated[tuple[Hash, ...], BeforeValidator(_tuple)]
    split_mode: Literal["new-subject", "forward-personal"]
    fitted_at: Instant
    available_at: Instant
    training_digest: Hash
    data_kind: Literal["synthetic", "observational"]
    cohorts: Names
    clinical_validation: Literal["not-established"] = "not-established"

    @model_validator(mode="after")
    def valid(self) -> Self:
        _check_version(self.schema_version, self.spec)
        if self.selected not in self.candidates:
            raise ValueError("selected_candidate_missing")
        if self.available_at < self.fitted_at:
            raise ValueError("model_time_order")
        names = [c.name for c in self.columns]
        if tuple(names) != expected_feature_names(self.spec):
            raise ValueError("column_order_or_schema")
        all_terms = set(names) | {"missing." + n for n in names} | {"intercept"}
        if "measurement.wrist_temperature.delta" in names and "measurement.resting_heart_rate.delta" in names:
            all_terms.add("interaction.temperature_heart")
        if "symptom.fatigue.rate" in names and "measurement.sleep_duration.delta" in names:
            all_terms.add("interaction.fatigue_sleep")
        if len({c.name for c in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate_candidate")
        for candidate in self.candidates:
            if any(t not in all_terms for t in candidate.terms):
                raise ValueError("unknown_model_term")
            if candidate.name == "baseline":
                if candidate.terms or candidate.penalty or candidate.blend.as_fraction():
                    raise ValueError("invalid_baseline_candidate")
            elif not candidate.terms or candidate.terms[0] != "intercept" or candidate.penalty <= 0:
                raise ValueError("invalid_ridge_candidate")
        if len(self.calibration_scores) != self.calibration_count:
            raise ValueError("calibration_count_mismatch")
        if list(self.calibration_scores) != sorted(self.calibration_scores) or any(x < 0 for x in self.calibration_scores):
            raise ValueError("invalid_calibration_scores")
        for group in (self.train_subject_hashes, self.tune_subject_hashes, self.calibration_subject_hashes):
            if tuple(sorted(set(group))) != group:
                raise ValueError("subject_hash_order")
        if self.split_mode == "new-subject":
            groups = [set(self.train_subject_hashes), set(self.tune_subject_hashes), set(self.calibration_subject_hashes)]
            if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
                raise ValueError("subject_overlap")
            if len(self.calibration_subject_hashes) != self.calibration_count:
                raise ValueError("calibration_subject_not_independent_unit")
        return self


CoverageAssumption = Literal["exchangeable-new-subjects-required", "dependent-time-series-no-coverage-guarantee"]


class CalibratedInterval(Model):
    nominal_coverage_ppm: Annotated[StrictInt, Field(gt=0, lt=1000000)]
    start: CivilDate | None
    end: CivilDate | None
    radius_days: Annotated[StrictInt, Field(ge=0)] | None
    unbounded: StrictBool
    assumption: CoverageAssumption

    @model_validator(mode="after")
    def valid(self) -> Self:
        if self.unbounded:
            if any(v is not None for v in (self.start, self.end, self.radius_days)):
                raise ValueError("unbounded_interval_has_bounds")
        elif self.start is None or self.end is None or self.radius_days is None or self.start > self.end:
            raise ValueError("invalid_interval")
        return self


class Contribution(Model):
    term: Identifier
    days: Rational


# A kind prefix plus any caller-supplied metric or symptom identifier.
UnusedInput = Annotated[str, StringConstraints(
    strict=True, pattern=r"^(?:(?:measurement|symptom)\.[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}|bleeding|checkin)$")]


class LearnedPrediction(Model):
    schema_version: LearningVersion = "learning-1.0"
    model_sha256: Hash
    subject_id: Identifier
    as_of: CivilDate
    cutoff: Instant
    status: Literal["estimate", "abstain"]
    reason: Identifier
    point_date: CivilDate | None = None
    point_cycle_length: Annotated[StrictInt, Field(ge=1)] | None = None
    raw_cycle_length: Rational | None = None
    baseline_cycle_length: Rational | None = None
    intervals: Annotated[tuple[CalibratedInterval, ...], BeforeValidator(_tuple)] = ()
    contributions: Annotated[tuple[Contribution, ...], BeforeValidator(_tuple)] = ()
    issues: Names = ()
    clinical_validation: Literal["not-established"] = "not-established"
    # Visible input kinds the spec never reads (learning-1.1). Set only by MultiFactorPipeline.run,
    # which sees raw events; predict_landmark and evaluate never have them.
    # Sorted and unique; a schema can state uniqueness but not the order.
    unused_inputs: Annotated[tuple[UnusedInput, ...], BeforeValidator(_tuple)] = Field(
        default=(), exclude_if=_empty, json_schema_extra={"uniqueItems": True})

    @model_validator(mode="after")
    def valid(self) -> Self:
        if (self.schema_version == "learning-1.1") != bool(self.unused_inputs):
            raise ValueError("contract_version_mismatch")
        if list(self.unused_inputs) != sorted(set(self.unused_inputs)):
            raise ValueError("unused_inputs_order")
        if self.status == "abstain":
            if any(v is not None for v in (self.point_date, self.point_cycle_length, self.raw_cycle_length, self.baseline_cycle_length)) or self.intervals or self.contributions:
                raise ValueError("abstention_has_estimate")
        else:
            if any(v is None for v in (self.point_date, self.point_cycle_length, self.raw_cycle_length, self.baseline_cycle_length)):
                raise ValueError("estimate_missing_fields")
            if self.point_date is not None and self.point_date < self.as_of:
                raise ValueError("past_prediction")
        return self
