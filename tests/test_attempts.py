"""Landmark attempts per elapsed day: every query a live scheduler would have made.

Hand-derived. Labels cutoff 2025-06-01T00:00Z, queries at 12:00 local, offset 0 unless stated.
Subject p: confirmed linked onsets p0..p4 on 01-01, 01-29, 02-26, 03-26, 04-23 (recorded 08:00Z on
the day), and a no-onset report for p3 through 04-14, recorded 04-15 08:00Z.
  Elapsed 20: queries 01-21, 02-18, 03-18, 04-15, 05-13. Only 04-15 has a report through the day
  before and three known intervals, so it is eligible; the others are onset-status-unconfirmed.
  Outcomes: the next onset is linked and unknown at each query, so exact, except 05-13 (no later
  onset, no report): outcome-not-observed.
  Elapsed 40: queries 02-10, 03-10, 04-07, 05-05 each already see the next onset, so
  landmark-day-mismatch and retrospective-label-before-landmark. 06-02 is after the labels cutoff.
  Elapsed 5: the 3 + 7 observation days do not fit, so the day is not applicable.
"""
from __future__ import annotations

import pytest

from quietcycle import Bleeding, CycleDataset, InputError, NoOnset, Onset, Tombstone
from quietcycle.dates import add_days
from quietcycle.learning import LandmarkSpec, landmark_attempts

LABELS = "2025-06-01T00:00:00Z"


def onset(subject: str, i: int, day: str, *, recorded: str | None = None, linked: bool = True, **changes) -> Onset:
    data = dict(subject_id=subject, id=f"{subject}{i}", date=day, recorded_at=recorded or day + "T08:00:00Z",
                utc_offset_minutes=0, certainty="confirmed", previous_onset_id=f"{subject}{i - 1}" if i and linked else None,
                continuity="confirmed" if i and linked else "unknown")
    data.update(changes)
    return Onset(**data)


def attempts(*events, elapsed_days=(20,), **arguments) -> list[dict]:
    arguments = {"spec": LandmarkSpec(), "labels_cutoff": LABELS, "query_time": "12:00", "utc_offset_minutes": 0,
                 **arguments}
    return landmark_attempts(CycleDataset(events=events), elapsed_days=elapsed_days, **arguments)["days"]


def subject_p() -> list:
    events: list = [onset("p", k, add_days("2025-01-01", 28 * k)) for k in range(5)]
    events.append(NoOnset(subject_id="p", id="p-report", anchor_id="p3", through="2025-04-14",
                          recorded_at="2025-04-15T08:00:00Z", utc_offset_minutes=0))
    return events


def test_each_elapsed_day_counts_attempts_codes_and_outcomes():
    report = landmark_attempts(CycleDataset(events=tuple(subject_p())), spec=LandmarkSpec(), elapsed_days=(5, 20, 40),
                               labels_cutoff=LABELS, query_time="12:00", utc_offset_minutes=0)
    assert report == {
        "scope": "landmark-attempts-not-model-performance",
        "labels_cutoff": "2025-06-01T00:00:00.000Z",
        "query_time": "12:00",
        "days": [
            {"elapsed_day": 5, "not_applicable": "landmark_before_observation_windows"},
            {"elapsed_day": 20, "attempts": 5, "eligible": 1, "rejected": {"onset-status-unconfirmed": 4},
             "after_labels_cutoff": 0, "outcomes": {"exact": 4, "outcome-not-observed": 1}},
            {"elapsed_day": 40, "attempts": 5, "eligible": 0, "rejected": {"landmark-day-mismatch": 4},
             "after_labels_cutoff": 1, "outcomes": {"retrospective-label-before-landmark": 4}},
        ],
    }


def test_attempts_follow_what_was_known_at_each_query():
    # q0 was moved from 01-01 to 01-02 on 01-10, before either query, then deleted on 02-01, after them:
    # 01-21 is not an attempt, 01-22 is, and its outcome sees the deletion.
    # s0 was backfilled on 03-01, after its 01-21 query, so that query never happened.
    # The versions are listed newest first; a dataset need not be in revision order.
    events = (onset("q", 0, "2025-01-02", revision=2, recorded="2025-01-10T08:00:00Z"),
              onset("q", 0, "2025-01-01"),
              Tombstone(subject_id="q", id="q0", revision=3, recorded_at="2025-02-01T08:00:00Z",
                        utc_offset_minutes=0, target_kind="onset"),
              onset("s", 0, "2025-01-01", recorded="2025-03-01T08:00:00Z"))
    assert attempts(*events) == [{"elapsed_day": 20, "attempts": 1, "eligible": 0,
                                  "rejected": {"onset-status-unconfirmed": 1}, "after_labels_cutoff": 0,
                                  "outcomes": {"anchor-revised-after-prediction": 1}}]


def test_each_attempt_is_counted_once_with_its_own_code():
    # u: two onsets on 01-01 are one attempt, rejected, with no single anchor for an outcome.
    # v: a +14:00 bleeding report dated 01-22 is visible at the 01-21 query, so only that day fails;
    #    the 02-18 query (anchor v1) is a normal attempt.
    # w: w0 moved from 05-20 to 05-21 before the labels cutoff; both queries are after it, and only
    #    the current date counts.
    # x: x0 moved from 05-20 to 05-21 on 06-05, after the labels cutoff, which cannot be used: the
    #    06-09 query counts as after the labels cutoff.
    # y: two different versions of one report, recorded after the 01-21 query, make the labels-time
    #    snapshot invalid, so the outcome is that error.
    events = (onset("u", 0, "2025-01-01"), onset("u", 1, "2025-01-01", linked=False),
              onset("v", 0, "2025-01-01"), onset("v", 1, "2025-01-29"),
              Bleeding(subject_id="v", id="v-bleed", date="2025-01-22", recorded_at="2025-01-21T11:00:00Z",
                       utc_offset_minutes=840, state="flow"),
              onset("w", 0, "2025-05-20"), onset("w", 0, "2025-05-21", revision=2, recorded="2025-05-25T08:00:00Z"),
              onset("x", 0, "2025-05-20"), onset("x", 0, "2025-05-21", revision=2, recorded="2025-06-05T08:00:00Z"),
              onset("y", 0, "2025-01-01"),
              *(Bleeding(subject_id="y", id="y-bleed", date="2025-02-01", recorded_at=f"2025-02-01T0{h}:00:00Z",
                         utc_offset_minutes=0, state=state) for h, state in ((8, "flow"), (9, "spotting"))))
    assert attempts(*events) == [{
        "elapsed_day": 20, "attempts": 6, "eligible": 0,
        "rejected": {"event_after_query_date": 1, "missing-or-conflicting-onsets": 1, "onset-status-unconfirmed": 2},
        "after_labels_cutoff": 2,
        "outcomes": {"anchor-not-unique": 1, "conflicting_revision": 1, "exact": 1, "outcome-not-observed": 1}}]


def test_query_time_and_offset_set_each_query_cutoff():
    # At 07:00 local the 04-15 query cannot see the report recorded at 08:00Z, so it is not eligible.
    # At +02:00, 09:00 local is 07:00Z: the same.
    # At +05:30, 13:00 local is 07:30Z, so the minutes of the offset matter.
    for arguments in ({"query_time": "07:00"}, {"query_time": "09:00", "utc_offset_minutes": 120},
                      {"query_time": "13:00", "utc_offset_minutes": 330}):
        (day,) = attempts(*subject_p(), **arguments)
        assert (day["eligible"], day["rejected"]) == (0, {"onset-status-unconfirmed": 5})


def test_boundaries_of_what_a_query_can_see():
    # Recorded exactly at the 01-21 12:00Z query, so that query sees it.
    (day,) = attempts(onset("z", 0, "2025-01-01", recorded="2025-01-21T12:00:00Z"))
    assert (day["attempts"], day["rejected"]) == (1, {"onset-status-unconfirmed": 1})
    # A labels cutoff equal to the 05-13 query's cutoff puts that query after it.
    (day,) = attempts(*subject_p(), labels_cutoff="2025-05-13T12:00:00Z")
    assert (day["attempts"], day["eligible"], day["after_labels_cutoff"], day["outcomes"]) == (5, 1, 1, {"exact": 4})
    # A query day past 2200-12-31 cannot exist, so it is no attempt at all.
    (day,) = attempts(onset("e", 0, "2200-12-20"), labels_cutoff="2200-12-31T00:00:00Z")
    assert (day["attempts"], day["after_labels_cutoff"]) == (0, 0)
    # On 2200-12-31, 12:00 at -14:00 is 2201-01-01 02:00Z, past the domain: after any labels cutoff.
    (day,) = attempts(onset("e", 0, "2200-12-11"), labels_cutoff="2200-12-31T00:00:00Z", utc_offset_minutes=-840)
    assert (day["attempts"], day["after_labels_cutoff"]) == (1, 1)
    # 12:00 at -14:00 is 02:00Z the next day, so the 04-15 query sees the report; at +14:00 it is
    # 22:00Z the day before, so no query sees one.
    (west,) = attempts(*subject_p(), utc_offset_minutes=-840)
    (east,) = attempts(*subject_p(), utc_offset_minutes=840)
    assert (west["eligible"], west["rejected"]) == (1, {"onset-status-unconfirmed": 4})
    assert (east["eligible"], east["rejected"]) == (0, {"onset-status-unconfirmed": 5})


def test_conflicting_versions_make_an_attempt_on_each_date_they_name():
    # Two different versions at revision 1: snapshot cannot choose, so both dates are tried and fail.
    versions = [onset("c", 0, "2025-01-01"), onset("c", 0, "2025-01-02")]
    for events in (versions, versions[::-1]):
        assert attempts(*events) == [{"elapsed_day": 20, "attempts": 2, "eligible": 0,
                                      "rejected": {"conflicting_revision": 2}, "after_labels_cutoff": 0,
                                      "outcomes": {"conflicting_revision": 2}}]


def test_counts_add_up():
    for day in attempts(*subject_p(), elapsed_days=(20, 40)):
        assert day["attempts"] == day["eligible"] + sum(day["rejected"].values()) + day["after_labels_cutoff"]
        assert sum(day["outcomes"].values()) == day["eligible"] + sum(day["rejected"].values())


@pytest.mark.parametrize("changes,code", [
    ({"elapsed_days": ()}, "invalid_elapsed_days"),
    ({"elapsed_days": (20, 20)}, "invalid_elapsed_days"),
    ({"elapsed_days": (40, 20)}, "invalid_elapsed_days"),
    ({"elapsed_days": (True,)}, "invalid_elapsed_days"),
    ({"query_time": "24:00"}, "invalid_query_time"),
    ({"query_time": "12:00:00"}, "invalid_query_time"),
    ({"utc_offset_minutes": 900}, "invalid_offset"),
    ({"labels_cutoff": "2025-06-01"}, "invalid_labels_cutoff"),
    ({"labels_cutoff": "2200-12-31T20:00:00Z", "utc_offset_minutes": 840}, "label_date_outside_domain"),
])
def test_invalid_arguments_fail_before_any_attempt(changes, code):
    arguments = {"spec": LandmarkSpec(), "elapsed_days": (20,), "labels_cutoff": LABELS, "query_time": "12:00",
                 "utc_offset_minutes": 0, **changes}
    with pytest.raises(InputError, match=code):
        landmark_attempts(CycleDataset(events=tuple(subject_p())), **arguments)
