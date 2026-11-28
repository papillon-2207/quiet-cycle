"""Score a daily forecast against what the labels allow: a lower and an optional upper cycle length.

outcome_probability is exact. log_score is minus its natural logarithm, a float: the logarithm of a
rational number other than 1 is irrational, so the score leaves the exact backend here. Its last
bits can differ across platforms, as the C library's log can.

The censored log score assumes that when the person stopped reporting, for a right-censored outcome,
or when they looked, for an interval-censored one, does not depend on L; it is proper under that
assumption, which the data cannot check. A forecast whose residual mass would have to be divided
among days cannot be scored, so an evaluation must fix the last day for every model, require
forecasts to cover it, and count refused cases rather than drop them.
"""
from __future__ import annotations

import math
from fractions import Fraction

from ..errors import InputError
from .survival import MAX_DAY, DailyForecast, _valid


def _valid_day(value: object) -> bool:
    return type(value) is int and 1 <= value <= MAX_DAY


def outcome_probability(forecast: DailyForecast, lower: int, upper: int | None) -> Fraction:
    """P(lower <= L <= upper) under the forecast; upper None means no upper bound, so the residual
    counts. Bounds that would divide a nonzero residual mass among days are refused."""
    forecast = _valid(DailyForecast, forecast, "invalid_forecast")
    # Without an upper bound, lower = MAX_DAY + 1 asks for L > MAX_DAY, which a residual can answer.
    if (not (_valid_day(lower) or (upper is None and lower == MAX_DAY + 1 and type(lower) is int))
            or (upper is not None and (not _valid_day(upper) or upper < lower))):
        raise InputError("invalid_bounds")
    if upper is not None and upper < forecast.elapsed_day:
        raise InputError("outcome_before_forecast")
    residual = forecast.residual_mass.as_fraction()
    last = forecast.elapsed_day + len(forecast.points) - 1
    if residual and (lower - 1 if upper is None else upper) > last:
        raise InputError("outcome_beyond_forecast")
    inside = sum((p.mass.as_fraction() for p in forecast.points
                  if lower <= p.cycle_length and (upper is None or p.cycle_length <= upper)), Fraction())
    return inside + (residual if upper is None else 0)


def log_score(forecast: DailyForecast, lower: int, upper: int | None) -> float:
    """-log P(lower <= L <= upper); infinite when the forecast gave the outcome no mass."""
    probability = outcome_probability(forecast, lower, upper)
    if probability == 0:
        return math.inf
    if probability > Fraction(1, 2):
        return -math.log1p(-float(1 - probability))  # 1 - p is exact, so a tiny score stays accurate
    if probability >= Fraction(2) ** -1022:
        return -math.log(float(probability))
    # Below the smallest normal float: the logarithms of the integers stay finite.
    return math.log(probability.denominator) - math.log(probability.numerator)
