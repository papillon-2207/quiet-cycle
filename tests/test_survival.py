import random
from fractions import Fraction

import pytest
from pydantic import ValidationError

from quietcycle.errors import InputError
from quietcycle.learning import survival
from quietcycle.learning.survival import (
    DailyForecast, ObservedCycle, condition, from_hazards, pooled_hazards, shrunk_hazards,
)


def masses(forecast):
    return [(p.cycle_length, p.mass.as_fraction()) for p in forecast.points], forecast.residual_mass.as_fraction()


def test_today_has_its_own_hazard_and_the_tail_keeps_the_rest():
    forecast = from_hazards(20, (Fraction(1, 2), Fraction(1, 3)))
    assert masses(forecast) == ([(20, Fraction(1, 2)), (21, Fraction(1, 6))], Fraction(1, 3))


def test_conditioning_on_a_later_day_equals_starting_from_it():
    hazards = tuple(Fraction((7 * i) % 11 + 1, 13) for i in range(15))
    forecast = from_hazards(20, hazards)
    for later in range(20, 36):
        assert condition(forecast, later) == from_hazards(later, hazards[later - 20:])
    assert masses(condition(forecast, 35)) == ([], Fraction(1))


def test_mass_must_be_conserved_and_consecutive_from_the_elapsed_day():
    good = from_hazards(5, (Fraction(1, 4),)).model_dump()
    short = dict(good, residual_mass={"numerator": "1", "denominator": "2"})
    gap = dict(good, points=[dict(good["points"][0], cycle_length=6)])
    for bad, code in ((short, "mass_not_conserved"), (gap, "nonconsecutive_days")):
        with pytest.raises(ValidationError, match=code):
            DailyForecast.model_validate(bad)


def test_no_conditioning_past_exhausted_support_or_into_the_past():
    certain = from_hazards(10, (Fraction(1),))
    with pytest.raises(InputError, match="support_exhausted"):
        condition(certain, 11)
    with pytest.raises(InputError, match="condition_before_forecast"):
        condition(certain, 9)


def test_hazards_must_be_fractions_in_the_unit_interval_given_as_a_tuple():
    for hazards in ((Fraction(-1, 3),), (Fraction(4, 3),), (1,), (0.5,), (h for h in (Fraction(1, 2),))):
        with pytest.raises(InputError, match="invalid_hazard"):
            from_hazards(1, hazards)
    with pytest.raises(InputError, match="invalid_hazard"):
        shrunk_hazards((0.5,), (), first_day=1, strength=Fraction(1))


def test_days_outside_the_supported_range_are_input_errors():
    half = (Fraction(1, 2),)
    for call in (lambda: from_hazards(0, half), lambda: from_hazards(True, half), lambda: from_hazards(995, half * 10),
                 lambda: from_hazards(1001, ()), lambda: condition(from_hazards(1, half), 1001),
                 lambda: pooled_hazards((), first_day=0, last_day=1),
                 lambda: shrunk_hazards(half, (), first_day=-5, strength=Fraction(1)),
                 lambda: shrunk_hazards((), (), first_day=1001, strength=Fraction(1))):
        with pytest.raises(InputError, match="invalid_day"):
            call()
    with pytest.raises(InputError, match="empty_window"):
        pooled_hazards((), first_day=2, last_day=1)


def test_unvalidated_cycles_and_forecasts_are_input_errors():
    bad = ObservedCycle.model_construct(entry_day=0, last_day=5000, onset="no")
    with pytest.raises(InputError, match="invalid_cycle"):
        pooled_hazards((bad,), first_day=1, last_day=3)
    with pytest.raises(InputError, match="invalid_cycle"):
        shrunk_hazards((Fraction(1, 2),), (bad,), first_day=1, strength=Fraction(1))
    for forecast in (None, DailyForecast.model_construct(elapsed_day=5, points=(), residual_mass=None)):
        with pytest.raises(InputError, match="invalid_forecast"):
            condition(forecast, 5)


def test_pooled_hazards_count_censored_days_at_risk_but_not_as_onsets():
    cycles = (ObservedCycle(entry_day=1, last_day=3, onset=True),
              ObservedCycle(entry_day=1, last_day=2, onset=False),
              ObservedCycle(entry_day=2, last_day=3, onset=True))
    # Jeffreys Beta(1/2, 1/2) posterior mean (D + 1/2) / (N + 1) for each day.
    assert pooled_hazards(cycles, first_day=1, last_day=3) == (Fraction(1, 6), Fraction(1, 8), Fraction(5, 6))
    with pytest.raises(InputError, match="empty_risk_set"):
        pooled_hazards(cycles, first_day=1, last_day=4)


def test_a_cycle_ends_on_or_after_the_day_it_enters_the_risk_set():
    with pytest.raises(ValidationError, match="cycle_ends_before_entry"):
        ObservedCycle(entry_day=5, last_day=4, onset=True)


def test_shrinkage_keeps_the_population_where_a_person_has_no_days_at_risk():
    population = (Fraction(1, 10), Fraction(1, 5), Fraction(1, 2))
    personal = (ObservedCycle(entry_day=1, last_day=2, onset=True), ObservedCycle(entry_day=2, last_day=2, onset=False))
    shrunk = shrunk_hazards(population, personal, first_day=1, strength=Fraction(3))
    # (c mu + d) / (c + n): day 1 one cycle survived; day 2 one onset and one censored; day 3 not at risk.
    assert shrunk == (Fraction(3, 40), Fraction(8, 25), Fraction(1, 2))
    assert shrunk_hazards(population, (), first_day=1, strength=Fraction(3)) == population
    assert all(0 < h < 1 for h in shrunk_hazards(population, personal, first_day=1, strength=Fraction(1, 10**9)))
    with pytest.raises(InputError, match="invalid_strength"):
        shrunk_hazards(population, personal, first_day=1, strength=Fraction(0))


def test_an_unstorable_forecast_fails_before_the_products_grow(monkeypatch):
    rng = random.Random(7)
    hazards = tuple(Fraction(rng.randrange(1, 2**512), 2**512 + rng.randrange(2**512)) for _ in range(1000))

    def never(*args):
        raise AssertionError("multiplied every day before failing")
    monkeypatch.setattr(survival, "_forecast", never)
    with pytest.raises(InputError, match="fraction_precision_budget"):
        from_hazards(1, hazards)


def test_checking_conservation_stops_when_the_running_sum_outgrows_the_budget():
    # Distinct 6799-bit denominators: each mass is storable, their exact sum is not.
    rng = random.Random(11)
    points = [{"cycle_length": 1 + i, "mass": {"numerator": "1", "denominator": str(2**6798 + 2 * rng.randrange(2**6797) + 1)}}
              for i in range(30)]
    with pytest.raises(ValidationError, match="fraction_precision_budget"):
        DailyForecast.model_validate({"elapsed_day": 1, "points": points, "residual_mass": {"numerator": "0", "denominator": "1"}})


def test_life_tables_count_only_the_days_inside_their_window():
    cycles = (ObservedCycle(entry_day=1, last_day=2, onset=True),    # before the window
              ObservedCycle(entry_day=1, last_day=4, onset=True),    # crosses its start, onset on day 4
              ObservedCycle(entry_day=6, last_day=8, onset=False),   # after it
              ObservedCycle(entry_day=2, last_day=9, onset=True),    # onset past its end
              ObservedCycle(entry_day=4, last_day=5, onset=True),    # onset on day 5
              ObservedCycle(entry_day=3, last_day=3, onset=True))    # onset on its first day
    # Day 3: 3 at risk, 1 onset; day 4: 3 and 1; day 5: 2 and 1.
    assert pooled_hazards(cycles, first_day=3, last_day=5) == (Fraction(3, 8), Fraction(3, 8), Fraction(1, 2))
    population = (Fraction(1, 10), Fraction(1, 5), Fraction(1, 2))
    assert shrunk_hazards(population, cycles, first_day=3, strength=Fraction(2)) == (Fraction(6, 25), Fraction(7, 25), Fraction(1, 2))
