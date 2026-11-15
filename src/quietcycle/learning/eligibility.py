"""Count what a dataset can support before any model is trained. Counts only; no identifiers or values."""
from __future__ import annotations

from typing import Any

from ..dates import instant, local_day
from ..errors import InputError
from ..features import DefaultFeatureBuilder
from ..ledger import Snapshot, snapshot
from ..models import CycleContext, CycleDataset, Event, Measurement, Query
from ..serialization import parse_model
from .contracts import LandmarkSpec
from .landmarks import _unit_registry, unused_inputs


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def dataset_inventory(dataset: CycleDataset, *, spec: LandmarkSpec, labels_cutoff: str,
                      utc_offset_minutes: int) -> dict[str, Any]:
    """Each subject is read from its own snapshot at labels_cutoff, so revisions count once."""
    dataset = parse_model(CycleDataset, dataset)
    spec = parse_model(LandmarkSpec, spec)
    try:
        cutoff = instant(labels_cutoff)
    except ValueError:
        raise InputError("invalid_labels_cutoff") from None
    try:
        as_of = local_day(cutoff, utc_offset_minutes)
    except ValueError as exc:
        raise InputError("invalid_offset" if str(exc) == "invalid_offset" else "label_date_outside_domain") from None
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
