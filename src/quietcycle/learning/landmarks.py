"""Build causal features from the existing typed event pipeline.

Labels are selected in a separate, explicit later snapshot. Whole-cycle summaries
and post-onset backfills never enter an earlier feature snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from ..dates import add_days, day_difference, instant, local_day
from ..errors import InputError
from ..features import DefaultFeatureBuilder
from ..ledger import snapshot
from ..models import Bleeding, Checkin, CycleDataset, Event, Identifier, Measurement, NoOnset, Onset, Query, Rational, Symptom
from ..units import UnitRegistry, UnitRule
from ..serialization import canonical_json, parse_model
from .contracts import FeatureValue, LabelledCase, Landmark, LandmarkSpec
from .exact import median


def spec_digest(spec: LandmarkSpec) -> str:
    return sha256(canonical_json(spec).encode()).hexdigest()


def unused_inputs(spec: LandmarkSpec, events: tuple[Event, ...]) -> tuple[str, ...]:
    """Names of visible input kinds that the spec never reads; never values or counts."""
    names: set[str] = set()
    for event in events:
        if isinstance(event, Measurement) and event.metric not in spec.metrics:
            names.add("measurement." + event.metric)
        elif isinstance(event, Symptom) and event.symptom not in spec.symptoms:
            names.add("symptom." + event.symptom)
        elif isinstance(event, (Bleeding, Checkin)):
            names.add(event.kind)
    return tuple(sorted(names))


def _check_reports(onsets: list[Onset], reports: list[NoOnset]) -> None:
    """A no-onset report must start at a confirmed onset and end before the next one."""
    by_id = {o.id: o for o in onsets}
    for report in reports:
        origin = by_id.get(report.anchor_id)
        if origin is None or origin.certainty != "confirmed" or report.through < origin.date:
            raise InputError("invalid-no-onset-report")
        if any(origin.date < o.date <= report.through for o in onsets):
            raise InputError("conflicting-no-onset-report")


def _check_case_metadata(data_kind: str, cohort: str) -> None:
    if data_kind not in ("synthetic", "observational"):
        raise InputError("invalid_data_kind")
    try:
        TypeAdapter(Identifier).validate_python(cohort)
    except ValidationError:
        raise InputError("invalid_cohort") from None


def _input_error(exc: ValidationError, fallback: str) -> InputError:
    """Never let a Pydantic error, which echoes values, escape; keep a single validator's code."""
    errors = exc.errors()
    cause = errors[0].get("ctx", {}).get("error") if len(errors) == 1 else None
    return InputError(str(cause) if isinstance(cause, ValueError) else fallback)


def make_landmark(dataset: CycleDataset, query: Query, spec: LandmarkSpec) -> Landmark:
    dataset = parse_model(CycleDataset, dataset)
    query = parse_model(Query, query)
    spec = parse_model(LandmarkSpec, spec)
    if query.context.paused:
        raise InputError("paused")
    if query.context.mode != "natural-cycle":
        raise InputError("outside-declared-scope")
    selected = snapshot(dataset, query)
    # Declared rules travel with the spec; a rule equal to a built-in adds nothing.
    registry = UnitRegistry({(r.metric, r.unit): UnitRule(r.canonical_unit, r.scale.as_fraction(), r.offset.as_fraction())
                             for r in spec.unit_rules if not r.same_as_builtin()})
    features = DefaultFeatureBuilder(registry).build(selected, query)
    if features.conflicting_onsets or not features.onsets:
        raise InputError("missing-or-conflicting-onsets")
    anchor = features.onsets[-1]
    if anchor.certainty != "confirmed" or anchor.exclude_from_history:
        raise InputError("uncertain-or-excluded-anchor")
    if query.context.changed_since and anchor.date < query.context.changed_since:
        raise InputError("context-changed-within-current-cycle")
    if day_difference(query.as_of, anchor.date) != spec.elapsed_day:
        raise InputError("landmark-day-mismatch")
    _check_reports(list(features.onsets), list(features.no_onset))
    reports = [r.through for r in features.no_onset if r.anchor_id == anchor.id]
    if not reports or max(reports) != add_days(query.as_of, -1):
        raise InputError("onset-status-unconfirmed")
    # A suffix only: never skip an unknown interval and call the remainder contiguous.
    recent = [i for i in features.intervals if not i.changed][-spec.max_history:]
    if len(recent) < spec.min_history or any(not i.known or i.excluded or i.changed for i in recent):
        raise InputError("insufficient-clean-history")
    lengths = [Fraction(i.length) for i in recent]
    center = median(lengths)
    values: dict[str, Fraction | None] = {
        "history.median": center,
        "history.last": lengths[-1],
        "history.mad": median([abs(x - center) for x in lengths]),
        "history.change": lengths[-1] - lengths[-2] if len(lengths) > 1 else Fraction(0),
        "history.count": Fraction(len(lengths)),
    }
    issues: list[str] = []
    # Use completed dates only; partial-day sampling cannot be mistaken for a daily summary.
    end = add_days(query.as_of, -1)
    recent_start = add_days(query.as_of, -spec.recent_days)
    reference_start = add_days(recent_start, -spec.reference_days)
    for metric in spec.metrics:
        selected_m = [m for m in features.measurements if m.metric == metric and reference_start <= m.date <= end and m.state == "measured"]
        streams = {(m.source, m.method, m.body_site) for m in selected_m}
        rvalue = delta = None
        if len(streams) > 1:
            issues.append("mixed-stream." + metric)
        else:
            days: dict[str, list[Fraction]] = {}
            for m in selected_m:
                if m.value is not None:
                    days.setdefault(m.date, []).append(m.value.as_fraction())
            daily = {day: median(vs) for day, vs in days.items()}
            r = [v for day, v in daily.items() if recent_start <= day <= end]
            b = [v for day, v in daily.items() if reference_start <= day < recent_start]
            if len(r) >= spec.minimum_measurement_days:
                rvalue = median(r)
            if rvalue is not None and len(b) >= spec.minimum_measurement_days:
                delta = rvalue - median(b)
        values[f"measurement.{metric}.recent"] = rvalue
        values[f"measurement.{metric}.delta"] = delta
    for symptom in spec.symptoms:
        days_s: dict[str, list[Symptom]] = {}
        for e in selected.events:
            if isinstance(e, Symptom) and e.symptom == symptom and recent_start <= e.date <= end:
                days_s.setdefault(e.date, []).append(e)
        usable: list[int] = []
        impacts: list[Fraction] = []
        for reports_s in days_s.values():
            states = {r.state for r in reports_s}
            if len(states) != 1:
                issues.append("conflicting-symptom." + symptom)
                continue
            state = reports_s[0].state
            if state in ("present", "absent"):
                usable.append(int(state == "present"))
                if state == "absent":
                    impacts.append(Fraction(0))
                elif len({r.impact for r in reports_s}) == 1 and reports_s[0].impact is not None:
                    impacts.append(Fraction(reports_s[0].impact))
        # A denominator is recorded known days, not every day in the window.
        values[f"symptom.{symptom}.rate"] = Fraction(sum(usable), len(usable)) if usable else None
        values[f"symptom.{symptom}.impact"] = sum(impacts, Fraction(0)) / len(impacts) if impacts else None
    try:
        return Landmark(subject_id=query.subject_id, anchor_id=anchor.id, anchor_date=anchor.date,
                        query=query, spec_sha256=spec_digest(spec), snapshot_sha256=selected.sha256,
                        baseline_length=Rational.from_fraction(center),
                        values=tuple(FeatureValue(name=n, value=None if v is None else Rational.from_fraction(v)) for n, v in sorted(values.items())),
                        issues=tuple(sorted(set(issues))))
    except ValidationError as exc:
        raise _input_error(exc, "invalid_landmark") from None


def _labels_cutoff(labels_cutoff: str) -> str:
    try:
        return instant(labels_cutoff)
    except ValueError:
        raise InputError("invalid_labels_cutoff") from None


def _classify_outcome(dataset: CycleDataset, *, query: Query, anchor_id: str, anchor_date: str,
                      labels_cutoff: str) -> tuple[str, Onset | None, str | None]:
    """("exact", next onset, when both were known) or (code, None, None), from reports visible at
    labels_cutoff only. No report is never read as no onset."""
    # Normalized instants share one fixed-width UTC spelling, so text order is time order.
    labels_cutoff = _labels_cutoff(labels_cutoff)
    if labels_cutoff <= query.cutoff:
        raise InputError("label_cutoff_not_later")
    try:
        label_day = local_day(labels_cutoff, query.utc_offset_minutes)
    except ValueError:
        raise InputError("label_date_outside_domain") from None
    label_q = Query(subject_id=query.subject_id, as_of=label_day,
                    cutoff=labels_cutoff, utc_offset_minutes=query.utc_offset_minutes, context=query.context)
    events = snapshot(dataset, label_q).events
    onsets = sorted((e for e in events if isinstance(e, Onset)), key=lambda o: (o.date, o.id))
    anchor = next((o for o in onsets if o.id == anchor_id), None)
    if (anchor is None or anchor.date != anchor_date or anchor.certainty != "confirmed" or anchor.exclude_from_history
            or [o.date for o in onsets].count(anchor_date) != 1):
        return "anchor-revised-after-prediction", None, None
    later = [o for o in onsets if o.date > anchor.date]
    reports = [e for e in events if isinstance(e, NoOnset)]
    covered = max((r.through for r in reports if r.anchor_id == anchor.id), default=None)
    if later:
        end = later[0]
        if sum(o.date == end.date for o in onsets) != 1:
            return "conflicting-label", None, None
        if end.date < query.as_of:
            return "retrospective-label-before-landmark", None, None
    # Only a report that could cover a day from the anchor to the next onset bears on this outcome.
    by_id, horizon = {o.id: o for o in onsets}, later[0].date if later else "9999-12-31"
    relevant = [r for r in reports if r.through >= anchor.date and (
        by_id[r.anchor_id].date < horizon if r.anchor_id in by_id else r.through <= horizon)]
    try:
        _check_reports(onsets, relevant)
    except InputError as exc:
        return exc.code, None, None
    if later:
        if (end.continuity == "confirmed" and end.previous_onset_id != anchor.id) or any(
                o.previous_onset_id == anchor.id and o.continuity == "confirmed" for o in later[1:]):
            return "conflicting-label-continuity", None, None
        if max(anchor.available_at, end.available_at) <= query.cutoff:
            return "outcome-already-known-at-landmark", None, None
        if end.exclude_from_history:
            return "label-excluded-from-history", None, None
        if end.certainty == "confirmed":
            # A confirmed link to another onset was rejected above, so this link is to the anchor.
            if end.continuity == "confirmed":
                return "exact", end, max(anchor.available_at, end.available_at)
            # A real onset, but an unrecorded one may lie between the last report and it.
            return "outcome-interval-censored", None, None
        if any(o.certainty == "confirmed" for o in later[1:]):
            # The doubtful onset may not be real, but a confirmed one bounds the outcome.
            return "outcome-interval-censored", None, None
    # No usable next onset: only an explicit report after the landmark censors the outcome.
    if covered is not None and covered >= query.as_of:
        return "outcome-right-censored", None, None
    return "outcome-not-observed", None, None


def label_landmark(dataset: CycleDataset, landmark: Landmark, *, labels_cutoff: str,
                   data_kind: Literal["synthetic", "observational"], cohort: str) -> LabelledCase:
    dataset = parse_model(CycleDataset, dataset)
    landmark = parse_model(Landmark, landmark)
    _labels_cutoff(labels_cutoff)  # a bad cutoff is reported before bad case metadata
    _check_case_metadata(data_kind, cohort)
    code, end, known_at = _classify_outcome(dataset, query=landmark.query, anchor_id=landmark.anchor_id,
                                            anchor_date=landmark.anchor_date, labels_cutoff=labels_cutoff)
    if end is None or known_at is None:
        raise InputError(code)
    try:
        return LabelledCase(landmark=landmark, outcome_date=end.date, outcome_known_at=known_at,
                            data_kind=data_kind, cohort=cohort)
    except ValidationError as exc:
        raise _input_error(exc, "invalid_labelled_case") from None


@dataclass(frozen=True)
class CaseBuildResult:
    cases: tuple[LabelledCase, ...]
    rejected: tuple[tuple[str, str, str], ...]


def build_cases(dataset: CycleDataset, queries: tuple[Query, ...], *, spec: LandmarkSpec,
                labels_cutoff: str, data_kind: Literal["synthetic", "observational"], cohort: str) -> CaseBuildResult:
    """Account for every requested landmark, including censored and invalid ones."""
    # A malformed cutoff is a configuration error for the whole call, not a per-case rejection.
    labels_cutoff = _labels_cutoff(labels_cutoff)
    _check_case_metadata(data_kind, cohort)  # and so are invalid case labels
    cases: list[LabelledCase] = []
    rejected: list[tuple[str, str, str]] = []
    for query in queries:
        try:
            landmark = make_landmark(dataset, query, spec)
            cases.append(label_landmark(dataset, landmark, labels_cutoff=labels_cutoff, data_kind=data_kind, cohort=cohort))
        except InputError as exc:
            rejected.append((query.subject_id, query.as_of, exc.code))
    return CaseBuildResult(tuple(cases), tuple(rejected))
