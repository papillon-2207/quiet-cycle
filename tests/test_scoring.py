import math
from fractions import Fraction

import pytest

from quietcycle.errors import InputError
from quietcycle.learning.scoring import log_score, outcome_probability
from quietcycle.learning.survival import DailyForecast, from_hazards

# P(L = 20) = 1/2, P(L = 21) = 1/6, residual P(L > 21) = 1/3.
FORECAST = from_hazards(20, (Fraction(1, 2), Fraction(1, 3)))


@pytest.mark.parametrize("lower,upper,expected", [
    (20, 20, Fraction(1, 2)), (21, 21, Fraction(1, 6)),        # exact
    (21, None, Fraction(1, 2)), (22, None, Fraction(1, 3)),    # right-censored: the residual counts
    (20, 21, Fraction(2, 3)), (5, 21, Fraction(2, 3)),         # interval; the forecast holds no mass before day 20
    (5, None, Fraction(1)),
])
def test_the_probability_of_what_the_labels_allow(lower, upper, expected):
    assert outcome_probability(FORECAST, lower, upper) == expected


@pytest.mark.parametrize("lower,upper", [(22, 22), (21, 22), (23, None)])
def test_bounds_that_split_the_residual_are_refused(lower, upper):
    # The residual covers every day after 21 and no onset at all, so no part of it belongs to one day.
    with pytest.raises(InputError, match="outcome_beyond_forecast"):
        outcome_probability(FORECAST, lower, upper)


def test_a_forecast_without_residual_mass_scores_any_later_outcome():
    certain = from_hazards(20, (Fraction(1, 2), Fraction(1)))
    assert [outcome_probability(certain, *b) for b in ((20, 30), (23, None), (22, 22))] == [1, 0, 0]
    empty = from_hazards(20, ())
    assert outcome_probability(empty, 20, None) == 1
    with pytest.raises(InputError, match="outcome_beyond_forecast"):
        outcome_probability(empty, 20, 20)


def test_an_outcome_before_the_forecast_day_contradicts_it():
    with pytest.raises(InputError, match="outcome_before_forecast"):
        outcome_probability(FORECAST, 5, 19)


def test_an_unvalidated_forecast_is_refused():
    with pytest.raises(InputError, match="invalid_forecast"):
        outcome_probability(DailyForecast.model_construct(elapsed_day=20, points=(), residual_mass=None), 20, None)


@pytest.mark.parametrize("lower,upper", [(0, None), (21, 20), (True, None), (20, 20.0), (1002, None), (1001, 1001), (20, 1001)])
def test_invalid_bounds(lower, upper):
    with pytest.raises(InputError, match="invalid_bounds"):
        outcome_probability(FORECAST, lower, upper)


def test_log_score_is_minus_the_log_of_that_probability():
    assert math.isclose(log_score(FORECAST, 21, 21), math.log(6), rel_tol=1e-15)
    assert math.isclose(log_score(FORECAST, 22, None), math.log(3), rel_tol=1e-15)
    assert log_score(from_hazards(20, (Fraction(1),)), 21, None) == math.inf


@pytest.mark.parametrize("exponent", [60, 200, 1000])
def test_a_probability_near_one_keeps_its_small_score(exponent):
    # -log(1 - x) is about x for tiny x; subtracting two large logarithms would give 0.
    near_one = from_hazards(1, (Fraction(1, 2**exponent),))
    assert math.isclose(log_score(near_one, 2, None), 2.0**-exponent, rel_tol=1e-12)


def test_the_residual_of_a_thousand_day_forecast_scores_a_cycle_longer_than_that():
    long = from_hazards(1, (Fraction(1, 2),) * 1000)
    assert outcome_probability(long, 1001, None) == Fraction(1, 2**1000)


def test_a_subnormal_probability_keeps_its_score():
    subnormal = from_hazards(1, (1 - Fraction(1, 3 * 2**1072),))
    assert math.isclose(log_score(subnormal, 2, None), 1072 * math.log(2) + math.log(3), rel_tol=1e-12)


def test_a_tiny_probability_still_has_a_finite_score():
    tiny = from_hazards(1, (Fraction(1, 2**2000),))
    assert math.isclose(log_score(tiny, 1, 1), 2000 * math.log(2), rel_tol=1e-12)


def test_the_true_forecast_scores_better_than_a_swapped_one():
    other = from_hazards(20, (Fraction(1, 3), Fraction(1, 2)))
    outcomes = [((20, 20), Fraction(1, 2)), ((21, 21), Fraction(1, 6)), ((22, None), Fraction(1, 3))]

    def expected(forecast):
        return math.fsum(float(q) * log_score(forecast, *bounds) for bounds, q in outcomes)
    assert expected(FORECAST) < expected(other)
