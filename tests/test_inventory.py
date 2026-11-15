"""Dataset inventory: what a dataset can support, counted before any model is trained.

Hand-derived fixture, all times UTC, labels cutoff 2024-06-30:
  a: onsets every 28 days from 2024-01-01 (a0..a5), confirmed and linked: 5 confirmed intervals in
     one chain; one no-onset report; one bleeding report. 8 visible events.
  b: b0 2024-02-01 confirmed; b1 2024-03-01 confirmed but not linked; b2 2024-03-30 uncertain:
     both intervals unconfirmed. Wrist temperature from "ring": one measured, one explicitly
     missing, both imported. One more reading recorded after the cutoff is invisible. A vendor
     score with a unit that has no rule must not drop b from the counts. 6 visible.
  c: two events share an id and revision with different content, so its snapshot is invalid.
  d: one confirmed onset marked excluded from history. 1 visible.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from quietcycle import Bleeding, CycleDataset, InputError, Measurement, NoOnset, Onset, Symptom, Tombstone
from quietcycle.dates import add_days
from quietcycle.learning import LandmarkSpec, UnitRuleSpec, dataset_inventory
from quietcycle.models import Rational
from quietcycle.serialization import canonical_json


def onset(subject: str, i: int, day: str, *, linked: bool = True, **changes) -> Onset:
    data = dict(subject_id=subject, id=f"{subject}{i}", date=day, recorded_at=day + "T08:00:00Z", utc_offset_minutes=0,
                certainty="confirmed", previous_onset_id=f"{subject}{i - 1}" if i and linked else None,
                continuity="confirmed" if i and linked else "unknown")
    data.update(changes)
    return Onset(**data)


def wrist(event_id: str, day: str, recorded: str, **changes) -> Measurement:
    data = dict(subject_id="b", id=event_id, date=day, recorded_at=recorded, measured_at=day + "T07:00:00Z",
                utc_offset_minutes=0, metric="wrist_temperature", value="35.9", unit="Cel",
                provenance={"source": "ring", "imported_at": add_days(day, 1) + "T00:00:00Z"})
    data.update(changes)
    return Measurement(**data)


def dataset() -> CycleDataset:
    events: list = [onset("a", k, add_days("2024-01-01", 28 * k)) for k in range(6)]
    events += [NoOnset(subject_id="a", id="a-report", anchor_id="a4", through="2024-05-11",
                       recorded_at="2024-05-12T08:00:00Z", utc_offset_minutes=0),
               Bleeding(subject_id="a", id="a-bleed", date="2024-05-21", recorded_at="2024-05-21T08:00:00Z",
                        utc_offset_minutes=0, state="flow")]
    events += [onset("b", 0, "2024-02-01"), onset("b", 1, "2024-03-01", linked=False),
               onset("b", 2, "2024-03-30", certainty="uncertain"),
               wrist("t1", "2024-03-10", "2024-03-10T08:00:00Z"),
               wrist("t2", "2024-03-12", "2024-03-12T08:00:00Z", state="missing", value=None, missing_reason="not-worn"),
               wrist("t3", "2024-07-01", "2024-07-02T08:00:00Z", provenance={"source": "watch"}),
               wrist("v1", "2024-03-15", "2024-03-15T08:00:00Z", metric="vendor_score", value="3", unit="{score}",
                     provenance={"source": "manual"})]
    events += [onset("c", 0, "2024-02-01"), onset("c", 0, "2024-02-02")]
    events += [onset("d", 0, "2024-02-01", exclude_from_history=True)]
    return CycleDataset(events=tuple(events))


def test_inventory_counts_every_item_by_hand():
    report = dataset_inventory(dataset(), spec=LandmarkSpec(), labels_cutoff="2024-06-30T00:00:00Z", utc_offset_minutes=0)
    assert report == {
        "scope": "data-inventory-not-model-performance",
        "labels_cutoff": "2024-06-30T00:00:00.000Z",
        "subjects": 4,
        "subject_errors": {"conflicting_revision": 1},
        "visible_events": 15,
        "onsets": {"confirmed": 8, "uncertain": 1, "excluded": 1},
        "intervals": {"confirmed_consecutive": 5, "unconfirmed": 2, "excluded": 0, "same_date": 0},
        "subjects_with_conflicting_onsets": 0,
        "confirmed_chains": 1,
        "subjects_by_longest_confirmed_run": {"0": 2, "5": 1},
        "availability": {"import_time": 2, "recorded_time_only": 13},
        "measurement_streams": [
            {"metric": "vendor_score", "unit": "{score}", "source": "manual", "method": None, "body_site": None,
             "measured": 1, "missing": 0, "rejected": {"unsupported_metric_or_unit": 1}},
            {"metric": "wrist_temperature", "unit": "Cel", "source": "ring", "method": None, "body_site": None,
             "measured": 1, "missing": 1, "rejected": {}}],
        "unused_inputs": ["bleeding", "measurement.vendor_score"],
        "vendor_derived": "not-recorded-in-data",
    }


def inventory(*events) -> dict:
    return dataset_inventory(CycleDataset(events=events), spec=LandmarkSpec(), labels_cutoff="2024-06-30T00:00:00Z",
                             utc_offset_minutes=0)


def every_28_days(subject: str, n: int, **per_index) -> list:
    return [onset(subject, k, add_days("2024-01-01", 28 * k), **per_index.get(str(k), {})) for k in range(n)]


def test_a_subject_known_only_after_the_cutoff_is_not_counted():
    late = onset("e", 0, "2024-06-01", recorded_at="2024-07-05T08:00:00Z")
    report = dataset_inventory(CycleDataset(events=dataset().events + (late,)), spec=LandmarkSpec(),
                               labels_cutoff="2024-06-30T00:00:00Z", utc_offset_minutes=0)
    assert (report["subjects"], report["subjects_by_longest_confirmed_run"]) == (4, {"0": 2, "5": 1})


def test_runs_break_at_unconfirmed_and_excluded_intervals():
    # f: f4 is not linked, so runs of 3 (f0..f3) and 2 (f4..f6); the longest comes first.
    # x: x3 is excluded (and not linked), so x2-x3 and x3-x4 are excluded and the runs are 2 (x0..x2)
    #    and 2 (x4..x6).
    report = inventory(*every_28_days("f", 7, **{"4": {"linked": False}}),
                       *every_28_days("x", 7, **{"3": {"exclude_from_history": True, "linked": False}}))
    assert report["onsets"] == {"confirmed": 13, "uncertain": 0, "excluded": 1}
    assert report["intervals"] == {"confirmed_consecutive": 9, "unconfirmed": 1, "excluded": 2, "same_date": 0}
    assert (report["confirmed_chains"], report["subjects_by_longest_confirmed_run"]) == (4, {"2": 1, "3": 1})


def test_two_onsets_on_one_date_break_the_chain():
    # g9 shares g2's date (02-26) and claims to follow it. Sorted by date and id: g0, g1, g2, g9, g3.
    # g2-g9 is linked but has length 0, and g9-g3 is unconfirmed because g3 links to g2. make_landmark
    # rejects any landmark whose snapshot holds both g2 and g9; landmarks before 02-26 are unaffected.
    g = every_28_days("g", 4) + [onset("g", 9, add_days("2024-01-01", 56), previous_onset_id="g2")]
    report = inventory(*g)
    assert report["onsets"]["confirmed"] == 5 and report["subjects_with_conflicting_onsets"] == 1
    assert report["intervals"] == {"confirmed_consecutive": 2, "unconfirmed": 1, "excluded": 0, "same_date": 1}
    assert (report["confirmed_chains"], report["subjects_by_longest_confirmed_run"]) == (1, {"2": 1})


def test_a_same_date_pair_is_same_date_even_when_excluded():
    pair = [onset("h", 0, "2024-01-01"), onset("h", 1, "2024-01-01", exclude_from_history=True)]
    assert inventory(*pair)["intervals"] == {"confirmed_consecutive": 0, "unconfirmed": 0, "excluded": 0, "same_date": 1}


def test_onset_precedence_and_subjects_without_onsets():
    doubtful = onset("k", 0, "2024-02-01", certainty="uncertain", exclude_from_history=True)
    only_bleeding = Bleeding(subject_id="m", id="m-bleed", date="2024-02-01", recorded_at="2024-02-01T08:00:00Z",
                             utc_offset_minutes=0, state="flow")
    report = inventory(doubtful, only_bleeding)
    assert report["onsets"] == {"confirmed": 0, "uncertain": 0, "excluded": 1}
    assert (report["subjects"], report["subjects_by_longest_confirmed_run"]) == (2, {"0": 2})


def test_unused_inputs_come_from_the_snapshot():
    late = Symptom(subject_id="a", id="acne", date="2024-06-29", recorded_at="2024-07-01T08:00:00Z", utc_offset_minutes=0,
                   symptom="acne", state="present")
    assert inventory(onset("a", 0, "2024-06-01"), late)["unused_inputs"] == []


def test_revisions_and_deletions_count_once():
    # h1 is revised to 2024-01-30, and h3 is deleted: h0, h1 (revision 2) and h2 remain.
    h = every_28_days("h", 4)
    h[2] = onset("h", 2, "2024-02-27")
    revised = onset("h", 1, "2024-01-30", revision=2, recorded_at="2024-02-01T08:00:00Z")
    deleted = Tombstone(subject_id="h", id="h3", revision=2, recorded_at="2024-04-01T08:00:00Z",
                        utc_offset_minutes=0, target_kind="onset")
    report = inventory(*h, revised, deleted)
    assert (report["visible_events"], report["onsets"]["confirmed"], report["intervals"]["confirmed_consecutive"]) == (3, 3, 2)


def test_streams_are_split_by_method_and_body_site():
    events = [wrist("w1", "2024-03-10", "2024-03-10T08:00:00Z", method="skin"),
              wrist("w2", "2024-03-11", "2024-03-11T08:00:00Z", method="core"),
              wrist("w3", "2024-03-12", "2024-03-12T08:00:00Z", method="skin", body_site="left-wrist")]
    streams = inventory(*events)["measurement_streams"]
    assert [(s["method"], s["body_site"], s["measured"]) for s in streams] == [
        ("core", None, 1), ("skin", None, 1), ("skin", "left-wrist", 1)]


def test_streams_count_samples_the_learner_would_reject():
    # The learner normalizes every sample with the spec's rules; one rejected sample blocks a landmark.
    score = wrist("v1", "2024-03-15", "2024-03-15T08:00:00Z", metric="vendor_score", value="3", unit="{score}")
    milli = wrist("w1", "2024-03-16", "2024-03-16T08:00:00Z", unit="mK", value="309000")
    steps = wrist("s1", "2024-03-17", "2024-03-17T08:00:00Z", metric="steps", unit="{count}", value="1.5")
    # A missing sample is normalized too, so its unit must still have a rule.
    gap = wrist("g1", "2024-03-18", "2024-03-18T08:00:00Z", unit="mK", state="missing", value=None, missing_reason="not-worn")
    rule = UnitRuleSpec(metric="vendor_score", unit="{score}", canonical_unit="{score}",
                        scale=Rational.from_fraction(Fraction(1)))
    data = CycleDataset(events=(score, milli, steps, gap))
    rejected = {}
    for spec in (LandmarkSpec(), LandmarkSpec(unit_rules=(rule,))):
        report = dataset_inventory(data, spec=spec, labels_cutoff="2024-06-30T00:00:00Z", utc_offset_minutes=0)
        rejected[len(spec.unit_rules)] = [(s["metric"], s["rejected"]) for s in report["measurement_streams"]]
    assert rejected[0] == [("steps", {"fractional_step_count": 1}), ("vendor_score", {"unsupported_metric_or_unit": 1}),
                           ("wrist_temperature", {"unsupported_metric_or_unit": 2})]
    assert rejected[1][1] == ("vendor_score", {})


def test_an_import_time_before_recording_does_not_set_availability():
    early = wrist("w1", "2024-03-10", "2024-03-10T08:00:00Z", provenance={"source": "ring", "imported_at": "2024-03-10T07:00:00Z"})
    later = wrist("w2", "2024-03-11", "2024-03-11T08:00:00Z")
    assert inventory(early, later)["availability"] == {"import_time": 1, "recorded_time_only": 1}


def test_inventory_has_no_identifiers_or_values():
    text = canonical_json(dataset_inventory(dataset(), spec=LandmarkSpec(), labels_cutoff="2024-06-30T00:00:00Z",
                                            utc_offset_minutes=0))
    assert not any(token in text for token in ('"a0"', '"b1"', "35.9", "t1", "a-report"))


def test_inventory_offset_sets_the_local_day():
    # Cutoff 2024-06-29 12:00Z is local 2024-06-29 at offset 0 but 2024-06-30 02:00 at +14:00.
    # The report, recorded 2024-06-29 11:00Z at +14:00, is dated 2024-06-30 local.
    late = Bleeding(subject_id="a", id="late", date="2024-06-30", recorded_at="2024-06-29T11:00:00Z",
                    utc_offset_minutes=840, state="none")
    data = CycleDataset(events=dataset().events + (late,))
    at_utc = dataset_inventory(data, spec=LandmarkSpec(), labels_cutoff="2024-06-29T12:00:00Z", utc_offset_minutes=0)
    at_local = dataset_inventory(data, spec=LandmarkSpec(), labels_cutoff="2024-06-29T12:00:00Z", utc_offset_minutes=840)
    assert at_utc["subject_errors"] == {"conflicting_revision": 1, "event_after_query_date": 1}
    assert at_local["subject_errors"] == {"conflicting_revision": 1}
    assert at_local["visible_events"] == 16


@pytest.mark.parametrize("changes,code", [
    ({"labels_cutoff": "2024-06-30"}, "invalid_labels_cutoff"),
    ({"utc_offset_minutes": 900}, "invalid_offset"),
    ({"labels_cutoff": "2200-12-31T23:00:00Z", "utc_offset_minutes": 840}, "label_date_outside_domain"),
])
def test_inventory_rejects_invalid_arguments(changes, code):
    arguments = {"spec": LandmarkSpec(), "labels_cutoff": "2024-06-30T00:00:00Z", "utc_offset_minutes": 0, **changes}
    with pytest.raises(InputError, match=code):
        dataset_inventory(dataset(), **arguments)
