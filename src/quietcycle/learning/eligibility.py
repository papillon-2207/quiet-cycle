"""Count what a dataset can support before any model is trained. Counts only; no identifiers or values."""
from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from ..dates import add_days, instant, local_day
from ..errors import InputError
from ..features import DefaultFeatureBuilder
from ..ledger import Snapshot, snapshot
from ..models import CycleContext, CycleDataset, Event, Measurement, Onset, Query
from ..serialization import parse_model
from .contracts import LandmarkSpec
from .landmarks import _classify_outcome, _input_error, _unit_registry, make_landmark, unused_inputs


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _labels_day(cutoff: str, utc_offset_minutes: int) -> str:
    try:
        return local_day(cutoff, utc_offset_minutes)
    except ValueError as exc:
        raise InputError("invalid_offset" if str(exc) == "invalid_offset" else "label_date_outside_domain") from None


def dataset_inventory(dataset: CycleDataset, *, spec: LandmarkSpec, labels_cutoff: str,
                      utc_offset_minutes: int) -> dict[str, Any]:
    """Each subject is read from its own snapshot at labels_cutoff, so revisions count once."""
    dataset = parse_model(CycleDataset, dataset)
    spec = parse_model(LandmarkSpec, spec)
    try:
        cutoff = instant(labels_cutoff)
    except ValueError:
        raise InputError("invalid_labels_cutoff") from None
    as_of = _labels_day(cutoff, utc_offset_minutes)
    by_subject: dict[str, list[Event]] = {}
    for event in dataset.events:
        by_subject.setdefault(event.subject_id, []).append(event)
    errors: dict[str, int] = {}
    longest_runs: dict[str, int] = {}
    onsets = {"confirmed": 0, "uncertain": 0, "excluded": 0}
    intervals = {"confirmed_consecutive": 0, "unconfirmed": 0, "excluded": 0, "same_date": 0}
    availability = {"import_time": 0, "recorded_time_only": 0}
    streams: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    subjects = visible = chains = conflicting = 0
    unused: set[str] = set()
    registry = _unit_registry(spec)
    builder = DefaultFeatureBuilder()
    for subject in sorted(by_subject):
        query = Query(subject_id=subject, as_of=as_of, cutoff=cutoff, utc_offset_minutes=utc_offset_minutes,
                      context=CycleContext(mode="unknown"))
        try:
            # The events were validated as part of the dataset; this only groups them.
            events = snapshot(CycleDataset.model_construct(events=tuple(by_subject[subject])), query).events
        except InputError as exc:
            subjects += 1
            _increment(errors, exc.code)  # only the first error found
            continue
        if not events:  # nothing about this subject was known at the cutoff
            continue
        subjects += 1
        visible += len(events)
        unused.update(unused_inputs(spec, events))
        # Reuse the feature builder's interval rule; measurements are left out so that an
        # unknown unit cannot remove the subject from the onset counts.
        history = builder.build(Snapshot(tuple(e for e in events if not isinstance(e, Measurement)), ""), query)
        for o in history.onsets:
            onsets["excluded" if o.exclude_from_history else o.certainty] += 1
        # The learner rejects any landmark whose snapshot holds two onsets on one date.
        conflicting += history.conflicting_onsets
        run = longest = 0
        for interval in history.intervals:
            if interval.length > 0 and interval.known and not interval.excluded:
                intervals["confirmed_consecutive"] += 1
                run += 1
                chains += run == 1
                longest = max(longest, run)
            else:
                intervals["same_date" if interval.length == 0 else "excluded" if interval.excluded else "unconfirmed"] += 1
                run = 0
        _increment(longest_runs, str(longest))
        for event in events:
            availability["import_time" if event.available_at > event.recorded_at else "recorded_time_only"] += 1
            if isinstance(event, Measurement):
                key = (event.metric, event.unit, event.provenance.source, event.method or "", event.body_site or "")
                stream = streams.setdefault(key, {"measured": 0, "missing": 0, "rejected": {}})
                stream[event.state] += 1
                try:
                    registry.normalize(event)
                except InputError as exc:
                    _increment(stream["rejected"], exc.code)
    return {
        "scope": "data-inventory-not-model-performance",
        "labels_cutoff": cutoff,
        "subjects": subjects,
        "subject_errors": dict(sorted(errors.items())),
        "visible_events": visible,
        "onsets": onsets,
        "intervals": intervals,
        "subjects_with_conflicting_onsets": conflicting,
        "confirmed_chains": chains,
        "subjects_by_longest_confirmed_run": dict(sorted(longest_runs.items(), key=lambda item: int(item[0]))),
        "availability": availability,
        "measurement_streams": [{"metric": m, "unit": u, "source": s, "method": me or None, "body_site": b or None,
                                 **counts, "rejected": dict(sorted(counts["rejected"].items()))}
                                for (m, u, s, me, b), counts in sorted(streams.items())],
        "unused_inputs": sorted(unused),
        "vendor_derived": "not-recorded-in-data",
    }


def _anchors(events: list[Event], date: str, known: str) -> list[str]:
    """Onsets whose latest version known by then is an onset on date, as snapshot selects versions."""
    latest: dict[str, list[Event]] = {}
    for event in events:
        if event.available_at <= known:
            best = latest.get(event.id)
            if best is None or event.revision > best[0].revision:
                latest[event.id] = [event]
            elif event.revision == best[0].revision:
                best.append(event)
    return sorted(i for i, versions in latest.items() if any(isinstance(v, Onset) and v.date == date for v in versions))


def landmark_attempts(dataset: CycleDataset, *, spec: LandmarkSpec, elapsed_days: tuple[int, ...], labels_cutoff: str,
                      query_time: str, utc_offset_minutes: int) -> dict[str, Any]:
    """Every landmark query a live scheduler would have made, per elapsed day, with its result and outcome."""
    dataset = parse_model(CycleDataset, dataset)
    spec = parse_model(LandmarkSpec, spec)
    if (not isinstance(elapsed_days, (tuple, list)) or not elapsed_days or any(type(e) is not int for e in elapsed_days)
            or list(elapsed_days) != sorted(set(elapsed_days))):
        raise InputError("invalid_elapsed_days")
    try:
        cutoff = instant(labels_cutoff)
    except ValueError:
        raise InputError("invalid_labels_cutoff") from None
    _labels_day(cutoff, utc_offset_minutes)
    if not isinstance(query_time, str) or re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", query_time) is None:
        raise InputError("invalid_query_time")
    zone = f"{'-' if utc_offset_minutes < 0 else '+'}{abs(utc_offset_minutes) // 60:02d}:{abs(utc_offset_minutes) % 60:02d}"
    by_subject: dict[str, list[Event]] = {}
    for event in dataset.events:
        by_subject.setdefault(event.subject_id, []).append(event)
    days: list[dict[str, Any]] = []
    for elapsed in elapsed_days:
        try:
            day_spec = LandmarkSpec.model_validate({**spec.model_dump(mode="python"), "elapsed_day": elapsed})
        except ValidationError as exc:
            days.append({"elapsed_day": elapsed, "not_applicable": _input_error(exc, "invalid_landmark_spec").code})
            continue
        eligible = late = 0
        rejected: dict[str, int] = {}
        outcomes: dict[str, int] = {}
        for subject in sorted(by_subject):
            events = by_subject[subject]
            # The events were validated as part of the dataset; this only groups them.
            data = CycleDataset.model_construct(events=tuple(events))
            for date in sorted({e.date for e in events if isinstance(e, Onset) and e.available_at <= cutoff}):
                try:
                    as_of = add_days(date, elapsed)
                except ValueError:
                    continue  # no query can be made on a day past the date domain
                try:
                    query_cutoff = instant(f"{as_of}T{query_time}:00{zone}")
                except ValueError:
                    query_cutoff = cutoff  # its UTC instant is past the domain, so after the labels cutoff
                # Normalized instants share one fixed-width UTC spelling, so text order is time order.
                anchors = _anchors(events, date, min(query_cutoff, cutoff))
                if not anchors:
                    continue
                if query_cutoff >= cutoff:
                    late += 1
                    continue
                query = Query(subject_id=subject, as_of=as_of, cutoff=query_cutoff, utc_offset_minutes=utc_offset_minutes,
                              context=CycleContext(mode="natural-cycle"))
                try:
                    make_landmark(data, query, day_spec)
                    eligible += 1
                except InputError as exc:
                    _increment(rejected, exc.code)
                if len(anchors) > 1:
                    _increment(outcomes, "anchor-not-unique")
                    continue
                try:
                    outcome = _classify_outcome(data, query=query, anchor_id=anchors[0], anchor_date=date,
                                                labels_cutoff=cutoff)[0]
                except InputError as exc:
                    outcome = exc.code
                _increment(outcomes, outcome)
        days.append({"elapsed_day": elapsed, "attempts": eligible + sum(rejected.values()) + late, "eligible": eligible,
                     "rejected": dict(sorted(rejected.items())), "after_labels_cutoff": late,
                     "outcomes": dict(sorted(outcomes.items()))})
    return {"scope": "landmark-attempts-not-model-performance", "labels_cutoff": cutoff, "query_time": query_time,
            "days": days}
