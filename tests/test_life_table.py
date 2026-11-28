"""Life-table baselines. Fixture: confirmed linked onsets o0..o4 every 28 days, o4 = 2025-01-01; the
landmark is elapsed day 20 (as_of 2025-01-21) with a no-onset report through 2025-01-20."""
from fractions import Fraction

import pytest

from quietcycle import CycleContext, CycleDataset, InputError, NoOnset, Onset, Query
from quietcycle.dates import add_days
from quietcycle.learning import LandmarkSpec
from quietcycle.learning.life_table import LifeTableModel, fit_life_table, forecast_life_table, history_lengths
from quietcycle.learning.survival import ObservedCycle, from_hazards, pooled_hazards, shrunk_hazards


def dataset(*later) -> CycleDataset:
    onsets = [Onset(subject_id="p", id=f"o{j}", date=add_days("2025-01-01", -(4 - j) * 28), recorded_at=add_days("2025-01-01", -(4 - j) * 28) + "T08:00:00Z",
                    utc_offset_minutes=0, certainty="confirmed", previous_onset_id=f"o{j - 1}" if j else None,
                    continuity="confirmed" if j else "unknown") for j in range(5)]
    report = NoOnset(subject_id="p", id="r", anchor_id="o4", through="2025-01-20", recorded_at="2025-01-21T08:00:00Z", utc_offset_minutes=0)
    return CycleDataset(events=tuple(onsets) + (report,) + later)


QUERY = Query(subject_id="p", as_of="2025-01-21", cutoff="2025-01-21T12:00:00Z", utc_offset_minutes=0,
              context=CycleContext(mode="natural-cycle"))
MODEL = fit_life_table((28, 28, 30, 30), spec=LandmarkSpec(), horizon=30, strength=Fraction(1))


def test_history_is_the_clean_history_a_landmark_uses():
    assert history_lengths(dataset(), QUERY, LandmarkSpec()) == (28, 28, 28, 28)
    with pytest.raises(InputError, match="landmark-day-mismatch"):
        history_lengths(dataset(), QUERY, LandmarkSpec(elapsed_day=21))


def test_the_pooled_hazards_come_from_complete_cycles():
    cycles = tuple(ObservedCycle(entry_day=1, last_day=n, onset=True) for n in (28, 28, 30, 30))
    assert tuple(h.as_fraction() for h in MODEL.hazards) == pooled_hazards(cycles, first_day=1, last_day=30)
    # Day 28: 4 at risk, 2 onsets; day 30: 2 and 2.
    assert MODEL.hazards[27].as_fraction() == Fraction(1, 2) and MODEL.hazards[29].as_fraction() == Fraction(5, 6)
    with pytest.raises(InputError, match="empty_risk_set"):
        fit_life_table((28, 30), spec=LandmarkSpec(), horizon=31, strength=Fraction(1))


def test_a_forecast_shrinks_the_pooled_hazards_toward_the_person():
    forecast = forecast_life_table(MODEL, dataset(), QUERY)
    population = tuple(h.as_fraction() for h in MODEL.hazards)
    personal = tuple(ObservedCycle(entry_day=1, last_day=28, onset=True) for _ in range(4))
    assert forecast == from_hazards(20, shrunk_hazards(population, personal, first_day=1, strength=Fraction(1))[19:])
    # Day 20: pooled (0 + 1/2) / (4 + 1) = 1/10, personal (1/10 + 0) / (1 + 4) = 1/50.
    assert forecast.points[0].cycle_length == 20 and forecast.points[0].mass.as_fraction() == Fraction(1, 50)


def test_later_records_do_not_change_a_forecast():
    late = Onset(subject_id="p", id="o5", date="2025-01-25", recorded_at="2025-01-26T08:00:00Z", utc_offset_minutes=0,
                 certainty="confirmed", previous_onset_id="o4", continuity="confirmed")
    assert forecast_life_table(MODEL, dataset(late), QUERY) == forecast_life_table(MODEL, dataset(), QUERY)


def test_the_model_has_its_own_identity_and_survives_json():
    assert (MODEL.model_id, MODEL.backend) == ("life-table-shrinkage-v1", "exact-fraction")
    assert LifeTableModel.model_validate_json(MODEL.model_dump_json()) == MODEL


@pytest.mark.parametrize("lengths,code", [((), "invalid_lengths"), ((28, 0), "invalid_lengths"), ((28, True), "invalid_lengths")])
def test_invalid_training_lengths(lengths, code):
    with pytest.raises(InputError, match=code):
        fit_life_table(lengths, spec=LandmarkSpec(), horizon=20, strength=Fraction(1))


def test_the_horizon_reaches_the_landmark_day():
    with pytest.raises(InputError, match="horizon_before_landmark"):
        fit_life_table((28, 30), spec=LandmarkSpec(), horizon=19, strength=Fraction(1))


def test_a_cycle_longer_than_the_horizon_is_at_risk_through_it_without_an_onset():
    long = fit_life_table((28, 1000), spec=LandmarkSpec(), horizon=30, strength=Fraction(1))
    # Day 30: only the 1000-day cycle is at risk, and it has no onset that day.
    assert long.hazards[29].as_fraction() == Fraction(1, 4)
    onsets = tuple(Onset(subject_id="q", id=f"o{j}", date=add_days("2020-01-01", 1500 * j), recorded_at=add_days("2020-01-01", 1500 * j) + "T08:00:00Z",
                         utc_offset_minutes=0, certainty="confirmed", previous_onset_id=f"o{j - 1}" if j else None,
                         continuity="confirmed" if j else "unknown") for j in range(4))
    anchor = add_days("2020-01-01", 4500)
    data = CycleDataset(events=onsets + (NoOnset(subject_id="q", id="r", anchor_id="o3", through=add_days(anchor, 19),
                                                  recorded_at=add_days(anchor, 20) + "T08:00:00Z", utc_offset_minutes=0),))
    query = Query(subject_id="q", as_of=add_days(anchor, 20), cutoff=add_days(anchor, 20) + "T12:00:00Z", utc_offset_minutes=0,
                  context=CycleContext(mode="natural-cycle"))
    # A 1500-day history: no ValidationError carrying the value, and the cycles count as censored at day 30.
    assert history_lengths(data, query, LandmarkSpec()) == (1500, 1500, 1500)
    forecast = forecast_life_table(long, data, query)
    personal = tuple(ObservedCycle(entry_day=1, last_day=30, onset=False) for _ in range(3))
    assert forecast == from_hazards(20, shrunk_hazards(tuple(h.as_fraction() for h in long.hazards), personal,
                                                      first_day=1, strength=Fraction(1))[19:])


def test_an_inconsistent_model_is_refused():
    good = MODEL.model_dump(mode="json")
    short = dict(good, hazards=good["hazards"][:5])
    certain = dict(good, hazards=[{"numerator": "1", "denominator": "1"}] + good["hazards"][1:])
    impossible = dict(good, hazards=[{"numerator": "0", "denominator": "1"}] + good["hazards"][1:])
    weak = dict(good, strength={"numerator": "-1", "denominator": "1"})
    for bad, code in ((short, "hazards_before_landmark"), (certain, "hazard_not_strictly_inside"),
                      (impossible, "hazard_not_strictly_inside"), (weak, "invalid_strength")):
        with pytest.raises(ValueError, match=code):
            LifeTableModel.model_validate(bad)


def test_the_training_digest_covers_everything_that_shapes_the_model():
    def digest(lengths=(28, 30), spec=LandmarkSpec(), strength=Fraction(1), horizon=28):
        return fit_life_table(lengths, spec=spec, horizon=horizon, strength=strength).training_sha256
    assert digest() == digest(lengths=(30, 28))
    assert len({digest(), digest(lengths=(28, 31)), digest(spec=LandmarkSpec(max_history=11)), digest(strength=Fraction(2)),
                digest(horizon=27)}) == 5
    assert fit_life_table((28, 30, 30), spec=LandmarkSpec(), horizon=28, strength=Fraction(1)).train_cycles == 3


@pytest.mark.parametrize("arguments,code", [({"strength": Fraction(0)}, "invalid_strength"), ({"strength": 1}, "invalid_strength"),
                                            ({"horizon": 20.0}, "invalid_horizon"), ({"horizon": True}, "invalid_horizon")])
def test_invalid_fit_arguments(arguments, code):
    with pytest.raises(InputError, match=code):
        fit_life_table((28, 30), **{"spec": LandmarkSpec(), "horizon": 28, "strength": Fraction(1), **arguments})
