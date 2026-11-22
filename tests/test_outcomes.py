"""Outcome codes after a landmark, from explicit reports only. Every date is derived by hand.

Fixture: confirmed linked onsets o0..o4 every 28 days, o4 = 2025-01-01. The landmark is elapsed
day 20: as_of 2025-01-21, cutoff 12:00Z, with a no-onset report for o4 through 2025-01-20.
Labels are read at 2025-04-01 unless a test says otherwise.
"""
from __future__ import annotations

from typing import Self

import pytest
from pydantic import ValidationError, model_validator

from quietcycle import CycleContext, CycleDataset, InputError, NoOnset, Onset, Query, Tombstone
from quietcycle.dates import add_days
from quietcycle.learning import LabelledCase, LandmarkSpec, build_cases, label_landmark, landmarks, make_landmark

AS_OF, LABELS = "2025-01-21", "2025-04-01T00:00:00Z"


def onset(i: int, day: str, *, previous: str | None = "o4", recorded: str | None = None, **changes) -> Onset:
    data = dict(subject_id="p", id=f"o{i}", date=day, recorded_at=recorded or day + "T08:00:00Z", utc_offset_minutes=0,
                certainty="confirmed", previous_onset_id=previous, continuity="confirmed" if previous else "unknown")
    data.update(changes)
    return Onset(**data)


def report(name: str, through: str, *, recorded: str | None = None, anchor: str = "o4") -> NoOnset:
    return NoOnset(subject_id="p", id=name, anchor_id=anchor, through=through,
                   recorded_at=recorded or add_days(through, 1) + "T08:00:00Z", utc_offset_minutes=0)


def history() -> list:
    events: list = [onset(j, add_days("2025-01-01", -(4 - j) * 28), previous=f"o{j - 1}" if j else None) for j in range(5)]
    return events + [report("landmark-report", "2025-01-20")]


def query(offset: int = 0) -> Query:
    return Query(subject_id="p", as_of=AS_OF, cutoff="2025-01-21T12:00:00Z", utc_offset_minutes=offset,
                 context=CycleContext(mode="natural-cycle"))


def code(*later, labels: str = LABELS) -> str:
    built = build_cases(CycleDataset(events=tuple(history()) + later), (query(),), spec=LandmarkSpec(),
                        labels_cutoff=labels, data_kind="synthetic", cohort="c")
    return "exact" if built.cases else built.rejected[0][2]


def known_at_landmark(*later) -> str:
    # The landmark is built without the later events, as if they were recorded just after its cutoff.
    landmark = make_landmark(CycleDataset(events=tuple(history())), query(), LandmarkSpec())
    try:
        label_landmark(CycleDataset(events=tuple(history()) + later), landmark, labels_cutoff=LABELS,
                       data_kind="synthetic", cohort="c")
    except InputError as exc:
        return exc.code
    return "exact"


@pytest.mark.parametrize("later,expected", [
    ((onset(5, "2025-01-30"),), "exact"),
    ((onset(5, "2025-01-30"), onset(6, "2025-01-30", previous="o4")), "conflicting-label"),
    # Row 2 before row 3: two onsets on 01-15, recorded after the landmark.
    ((onset(5, "2025-01-15", recorded="2025-02-01T08:00:00Z"), onset(6, "2025-01-15", recorded="2025-02-01T08:00:00Z")),
     "conflicting-label"),
    # Row 2 before row 4.
    ((onset(5, "2025-01-30"), onset(6, "2025-01-30"), report("past", "2025-02-02")), "conflicting-label"),
    ((onset(5, "2025-01-30"), report("past-the-onset", "2025-02-02")), "conflicting-no-onset-report"),
    # Row 4 boundary: a report through the day before the onset is fine; through the onset is not.
    ((onset(5, "2025-01-30"), report("up-to", "2025-01-29")), "exact"),
    ((onset(5, "2025-01-30"), report("onto", "2025-01-30")), "conflicting-no-onset-report"),
    # A report from an earlier onset that reaches the anchor is checked: o3's report runs past o4.
    ((onset(5, "2025-01-30"), report("o3-late", "2025-02-10", anchor="o3")), "conflicting-no-onset-report"),
    ((report("o3-late", "2025-02-10", anchor="o3"),), "conflicting-no-onset-report"),
    ((report("unknown-anchor", "2025-02-10", anchor="zz"),), "invalid-no-onset-report"),
    ((onset(5, "2025-01-30"), report("unknown-anchor", "2025-01-25", anchor="zz")), "invalid-no-onset-report"),
    # The window starts at the anchor day: o3's report through it contradicts o4, while o2's late
    # report about its own cycle cannot touch this outcome. An unknown origin ending on o5 counts.
    ((onset(5, "2025-01-30"), report("o3x", "2025-01-01", anchor="o3", recorded="2025-02-05T08:00:00Z")),
     "conflicting-no-onset-report"),
    ((onset(5, "2025-01-30"), report("o2-late", "2024-12-04", anchor="o2", recorded="2025-02-01T08:00:00Z")), "exact"),
    ((onset(5, "2025-01-30"), report("zz", "2025-01-30", anchor="zz")), "invalid-no-onset-report"),
    # Reports about later cycles cannot change this outcome: an unknown origin ending after o5,
    # o5's own report reaching o6, and a report for o6 after o6 was made uncertain.
    ((onset(5, "2025-01-30"), report("unknown-anchor", "2025-02-10", anchor="zz")), "exact"),
    ((onset(5, "2025-01-30"), onset(6, "2025-02-27", previous="o5"), report("o5-late", "2025-02-27", anchor="o5")), "exact"),
    ((onset(5, "2025-01-30"), onset(6, "2025-02-27", previous="o5", certainty="uncertain"),
      report("o6-report", "2025-03-10", anchor="o6")), "exact"),
    # Row 1: an anchor made uncertain or excluded after the prediction, or a second onset on its date.
    ((onset(5, "2025-01-30"), onset(9, "2025-01-01", previous="o3", recorded="2025-02-01T08:00:00Z",
                                    certainty="uncertain", exclude_from_history=True)), "anchor-revised-after-prediction"),
    # The anchor's own later revision on the same date is not a second onset.
    ((onset(5, "2025-01-30"), onset(4, "2025-01-01", previous="o3", revision=2, recorded="2025-02-01T08:00:00Z")), "exact"),
    ((onset(5, "2025-01-30"), onset(4, "2025-01-01", previous="o3", revision=2, recorded="2025-02-01T08:00:00Z",
                                    certainty="uncertain")), "anchor-revised-after-prediction"),
    ((onset(5, "2025-01-30"), onset(4, "2025-01-01", previous="o3", revision=2, recorded="2025-02-01T08:00:00Z",
                                    exclude_from_history=True)), "anchor-revised-after-prediction"),
    # Row 4 before row 5.
    ((onset(5, "2025-01-30", previous="o3"), report("past", "2025-02-02")), "conflicting-no-onset-report"),
    ((onset(5, "2025-01-30", previous="o3"),), "conflicting-label-continuity"),
    ((onset(5, "2025-01-30", previous=None), onset(6, "2025-02-27")), "conflicting-label-continuity"),
    # Only a confirmed link to the anchor counts as skipping o5; a link from o6 to o3 is not checked.
    ((onset(5, "2025-01-30"), onset(6, "2025-02-27", continuity="unknown")), "exact"),
    ((onset(5, "2025-01-30"), onset(6, "2025-02-27", previous="o3")), "exact"),
    ((onset(5, "2025-01-30", exclude_from_history=True),), "label-excluded-from-history"),
    ((onset(5, "2025-02-27", previous=None),), "outcome-interval-censored"),
    # An uncertain next onset with a confirmed one after it: the outcome is no later than 02-27.
    ((onset(5, "2025-01-30", certainty="uncertain"), onset(6, "2025-02-27", previous="o5")), "outcome-interval-censored"),
    # An excluded onset is still an onset, so it bounds the outcome too.
    ((onset(5, "2025-01-30", certainty="uncertain"), onset(6, "2025-02-27", previous="o5", exclude_from_history=True)),
     "outcome-interval-censored"),
    # Linking o6 straight to the anchor skips o5, which is rejected even though o5 is doubtful.
    ((onset(5, "2025-01-30", certainty="uncertain"), onset(6, "2025-02-27")), "conflicting-label-continuity"),
    ((report("through-as-of", AS_OF),), "outcome-right-censored"),
    ((report("follow-up", "2025-02-10"),), "outcome-right-censored"),
    ((onset(5, "2025-01-30", certainty="uncertain"), report("before-doubt", "2025-01-25")), "outcome-right-censored"),
    ((), "outcome-not-observed"),
    ((onset(5, "2025-01-30", certainty="uncertain"),), "outcome-not-observed"),
])
def test_each_outcome_row(later, expected):
    assert code(*later) == expected


@pytest.mark.parametrize("later,expected", [
    ((onset(5, AS_OF),), "outcome-already-known-at-landmark"),
    # Row 5 before row 6, and row 6 before row 7.
    ((onset(5, AS_OF, previous="o3"),), "conflicting-label-continuity"),
    ((onset(5, AS_OF, exclude_from_history=True),), "outcome-already-known-at-landmark"),
])
def test_rows_for_an_outcome_known_at_the_landmark(later, expected):
    assert known_at_landmark(*later) == expected


@pytest.mark.parametrize("changes,code", [({"cohort": "bad cohort"}, "invalid_cohort"),
                                          ({"data_kind": "clinical"}, "invalid_data_kind")])
def test_invalid_case_metadata_fails_the_whole_call(changes, code):
    arguments = {"spec": LandmarkSpec(), "labels_cutoff": LABELS, "data_kind": "synthetic", "cohort": "c", **changes}
    for later in ((onset(5, "2025-01-30"),), ()):
        data = CycleDataset(events=tuple(history()) + later)
        with pytest.raises(InputError, match=code):
            build_cases(data, (query(),), **arguments)
        with pytest.raises(InputError, match=code):
            label_landmark(data, make_landmark(data, query(), LandmarkSpec()), **{k: v for k, v in arguments.items() if k != "spec"})


def test_an_invalid_labels_cutoff_is_reported_before_case_metadata():
    data = CycleDataset(events=tuple(history()) + (onset(5, "2025-01-30"),))
    arguments = {"labels_cutoff": "bad", "data_kind": "synthetic", "cohort": "bad cohort"}
    with pytest.raises(InputError, match="invalid_labels_cutoff"):
        build_cases(data, (query(),), spec=LandmarkSpec(), **arguments)
    with pytest.raises(InputError, match="invalid_labels_cutoff"):
        label_landmark(data, make_landmark(data, query(), LandmarkSpec()), **arguments)


@pytest.mark.parametrize("extra,code", [
    # o3 is made uncertain, so its report has no confirmed origin; without the report the history fails.
    ((onset(3, add_days("2025-01-01", -28), previous="o2", revision=2, recorded="2025-01-02T08:00:00Z",
            certainty="uncertain"), report("o3-report", "2024-12-20", anchor="o3")), "invalid-no-onset-report"),
    ((onset(3, add_days("2025-01-01", -28), previous="o2", revision=2, recorded="2025-01-02T08:00:00Z",
            certainty="uncertain"),), "insufficient-clean-history"),
    ((report("before-origin", "2024-12-01", anchor="o3"),), "invalid-no-onset-report"),
])
def test_landmark_report_rules(extra, code):
    with pytest.raises(InputError, match=code):
        make_landmark(CycleDataset(events=tuple(history()) + extra), query(), LandmarkSpec())


def test_a_report_through_its_own_onset_day_is_valid():
    data = CycleDataset(events=tuple(history()) + (report("same-day", "2024-12-04", anchor="o3"),))
    assert make_landmark(data, query(), LandmarkSpec()).anchor_id == "o4"


def test_reports_after_the_labels_cutoff_are_invisible():
    late = report("late", "2025-02-10", recorded="2025-04-02T08:00:00Z")
    assert code(late) == "outcome-not-observed"
    assert code(late, labels="2025-04-03T00:00:00Z") == "outcome-right-censored"


def test_an_outcome_already_visible_at_the_cutoff_is_rejected():
    # The landmark is built without o5; o5 (dated as_of, uncertain) was recorded before the cutoff.
    landmark = make_landmark(CycleDataset(events=tuple(history())), query(), LandmarkSpec())
    known = onset(5, AS_OF, certainty="uncertain", recorded=AS_OF + "T08:00:00Z")
    with pytest.raises(InputError, match="outcome-already-known-at-landmark"):
        label_landmark(CycleDataset(events=tuple(history()) + (known,)), landmark, labels_cutoff=LABELS,
                       data_kind="synthetic", cohort="c")


@pytest.mark.parametrize("recorded, offset, query_offset", [
    # Recorded on its own date, which at the query's offset is still the day before.
    ("2025-01-30T07:59:00+08:00", 480, 0),  # after travel east
    ("2025-01-30T00:30:00+02:00", 120, 60),  # after a one-hour change, as with summer time
    ("2025-01-30T08:00:00Z", 0, -720),
])
def test_a_label_keeps_the_offset_of_its_onset(recorded, offset, query_offset):
    later = onset(5, "2025-01-30", recorded=recorded, utc_offset_minutes=offset)
    built = build_cases(CycleDataset(events=tuple(history()) + (later,)), (query(query_offset),), spec=LandmarkSpec(),
                        labels_cutoff=LABELS, data_kind="synthetic", cohort="c")
    assert not built.rejected and [c.outcome_date for c in built.cases] == ["2025-01-30"]


def test_labels_read_each_event_on_its_own_date():
    # At labels cutoffs that are still 2025-01-29 at the query's offset 0, an onset dated 2025-01-30
    # at +08:00 is visible. It is the outcome, or a later onset that does not change the outcome.
    def labelled(*later, labels):
        built = build_cases(CycleDataset(events=tuple(history()) + later), (query(),), spec=LandmarkSpec(),
                            labels_cutoff=labels, data_kind="synthetic", cohort="c")
        return [c.outcome_date for c in built.cases], built.rejected
    east = onset(5, "2025-01-30", recorded="2025-01-30T07:59:00+08:00", utc_offset_minutes=480)
    assert labelled(east, labels="2025-01-29T23:59:30Z") == (["2025-01-30"], ())
    next_one = onset(6, "2025-02-27", previous="o5", recorded="2025-02-27T07:00:00+08:00", utc_offset_minutes=480)
    assert labelled(onset(5, "2025-01-30"), next_one, labels="2025-02-26T23:30:00Z") == (["2025-01-30"], ())


def test_labels_read_an_event_recorded_at_the_largest_offset():
    # Recorded at 00:00+14:00 on its date; 30 seconds later it is still the day before at +13:59.
    late = onset(5, "2025-01-30", recorded="2025-01-30T00:00:00+14:00", utc_offset_minutes=840)
    built = build_cases(CycleDataset(events=tuple(history()) + (late,)), (query(),), spec=LandmarkSpec(),
                        labels_cutoff="2025-01-29T10:00:30Z", data_kind="synthetic", cohort="c")
    assert [c.outcome_date for c in built.cases] == ["2025-01-30"] and not built.rejected


@pytest.mark.parametrize("query_offset, labels", [(0, "2200-12-31T12:00:00Z"), (-720, "2200-12-31T11:00:00Z"),
                                                  (60, "2200-12-31T23:30:00Z")])
def test_labels_read_near_the_end_of_the_date_domain(query_offset, labels):
    # Past 2200-12-31T10:00Z, +14:00 is in 2201, so the labels are read on the UTC day, 2200-12-31,
    # where an onset recorded on that date at offset 0 is visible.
    events: list = [onset(j, add_days("2200-11-01", -(4 - j) * 28), previous=f"o{j - 1}" if j else None) for j in range(5)]
    events += [report("landmark-report", "2200-11-20", anchor="o4"), onset(5, "2200-11-30"),
               onset(6, "2200-12-31", previous="o5", recorded="2200-12-31T09:00:00Z")]
    cutoff = instant_at("2200-11-21T12:00:00", query_offset)
    late = Query(subject_id="p", as_of="2200-11-21", cutoff=cutoff, utc_offset_minutes=query_offset,
                 context=CycleContext(mode="natural-cycle"))
    built = build_cases(CycleDataset(events=tuple(events)), (late,), spec=LandmarkSpec(),
                        labels_cutoff=labels, data_kind="synthetic", cohort="c")
    assert [c.outcome_date for c in built.cases] == ["2200-11-30"] and not built.rejected


def instant_at(local: str, offset: int) -> str:
    sign = "-" if offset < 0 else "+"
    return f"{local}{sign}{abs(offset) // 60:02d}:{abs(offset) % 60:02d}"


def test_a_case_known_before_its_outcome_date_began_anywhere_is_invalid():
    case = dict(landmark=make_landmark(CycleDataset(events=tuple(history())), query(), LandmarkSpec()),
                outcome_date="2025-01-30", data_kind="synthetic", cohort="c")
    # 2025-01-30 begins first at +14:00, the largest offset: at 2025-01-29T10:00Z.
    assert LabelledCase(**case, outcome_known_at="2025-01-29T10:00:00Z").outcome_known_at == "2025-01-29T10:00:00.000Z"
    with pytest.raises(ValidationError, match="label_available_before_event"):
        LabelledCase(**case, outcome_known_at="2025-01-29T09:59:59.999Z")


def test_an_outcome_known_at_the_cutoff_is_the_version_its_snapshot_held():
    seen = onset(5, AS_OF, certainty="uncertain", recorded=AS_OF + "T08:00:00Z")
    revised = onset(5, AS_OF, revision=2, recorded="2025-02-01T08:00:00Z")
    # o5 was visible before the cutoff with the date it still has; its revision 2 came later.
    assert known_at_landmark(seen, revised) == "outcome-already-known-at-landmark"
    # So was it when the anchor gets a later revision that keeps its date.
    anchor = onset(4, "2025-01-01", previous="o3", revision=2, recorded="2025-02-01T08:00:00Z")
    assert known_at_landmark(seen, revised, anchor) == "outcome-already-known-at-landmark"
    # At the cutoff the snapshot held another date, or a deletion, or another subject's o5.
    moved = onset(5, "2025-01-15", revision=2, recorded=AS_OF + "T09:00:00Z")
    deleted = Tombstone(subject_id="p", id="o5", revision=2, target_kind="onset", recorded_at=AS_OF + "T09:00:00Z",
                        utc_offset_minutes=0)
    again = onset(5, AS_OF, revision=3, recorded="2025-02-01T08:00:00Z")
    assert known_at_landmark(seen, moved, again) == "exact"
    assert known_at_landmark(seen, deleted, again) == "exact"
    other = onset(5, AS_OF, recorded=AS_OF + "T08:00:00Z", subject_id="q", previous=None)
    assert known_at_landmark(other, revised) == "exact"


def test_label_validation_failures_are_input_errors(monkeypatch):
    # label_landmark settles every case that LabelledCase would reject before building it, so a
    # stricter case class stands in for a check that it could reach.
    class Stricter(LabelledCase):
        @model_validator(mode="after")
        def stricter(self) -> Self:
            raise ValueError("label_available_before_event")

    monkeypatch.setattr(landmarks, "LabelledCase", Stricter)
    data = CycleDataset(events=tuple(history()) + (onset(5, "2025-01-30"),))
    landmark = make_landmark(data, query(), LandmarkSpec())
    with pytest.raises(InputError, match="label_available_before_event") as raised:
        label_landmark(data, landmark, labels_cutoff=LABELS, data_kind="synthetic", cohort="c")
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
