"""Exact daily forecasts of the cycle length L from elapsed day d on, and life-table hazards.

A forecast holds P(L = k | L >= d) for k = d, d+1, ..., K and keeps P(L > K | L >= d) as residual
mass, which covers both a later onset and none at all. It is never renormalized away. Today, k = d,
can hold the onset, so its mass is the hazard h(d) itself.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import accumulate
from typing import Annotated, Literal, Self, TypeVar

from pydantic import BeforeValidator, Field, StrictBool, StrictInt, ValidationError, model_validator

from ..errors import InputError
from ..models import Model, Probability, _tuple, array_bounds

MAX_DAY = 1000
PRECISION_BITS = 6800  # the limit of Rational.from_fraction


def _bits(value: Fraction) -> int:
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())
M = TypeVar("M", bound=Model)
Day = Annotated[StrictInt, Field(ge=1, le=MAX_DAY)]


class DayMass(Model):
    cycle_length: Day
    mass: Probability


class DailyForecast(Model):
    elapsed_day: Day
    points: Annotated[tuple[DayMass, ...], BeforeValidator(_tuple), Field(max_length=1000, json_schema_extra=array_bounds)]
    residual_mass: Probability
    residual_meaning: Literal["after-horizon-or-never-no-renormalization"] = "after-horizon-or-never-no-renormalization"

    @model_validator(mode="after")
    def conservation(self) -> Self:
        if [p.cycle_length for p in self.points] != list(range(self.elapsed_day, self.elapsed_day + len(self.points))):
            raise ValueError("nonconsecutive_days")
        total = self.residual_mass.as_fraction()
        for point in self.points:
            total += point.mass.as_fraction()
            # A sum of conditioned masses needs at most twice the storage budget; an arbitrary list of
            # storable masses can need far more, and each addition then gets slower.
            if _bits(total) > 2 * PRECISION_BITS + 8:
                raise ValueError("fraction_precision_budget")
        if total != 1:
            raise ValueError("mass_not_conserved")
        return self


class ObservedCycle(Model):
    """At risk from entry_day through last_day; onset=True means L = last_day, otherwise an explicit
    report says there was no onset through last_day, so L > last_day."""
    entry_day: Day
    last_day: Day
    onset: StrictBool

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.last_day < self.entry_day:
            raise ValueError("cycle_ends_before_entry")
        return self


def _days(first: object, last: object) -> None:
    if type(first) is not int or type(last) is not int or not (1 <= first <= MAX_DAY and first - 1 <= last <= MAX_DAY):
        raise InputError("invalid_day")


def _valid(kind: type[M], value: object, code: str) -> M:
    # Revalidate, so that an instance from model_construct cannot carry unchecked values.
    if type(value) is not kind:
        raise InputError(code)
    try:
        return kind.model_validate(value)
    except ValidationError:
        raise InputError(code) from None


def _hazards(hazards: object) -> tuple[Fraction, ...]:
    if type(hazards) is not tuple or any(type(h) is not Fraction or not 0 <= h <= 1 for h in hazards):
        raise InputError("invalid_hazard")
    return hazards


def _cycles(cycles: object) -> tuple[ObservedCycle, ...]:
    if type(cycles) is not tuple:
        raise InputError("invalid_cycle")
    return tuple(_valid(ObservedCycle, c, "invalid_cycle") for c in cycles)


def _forecast(elapsed_day: int, masses: list[Fraction], tail: Fraction) -> DailyForecast:
    return DailyForecast(elapsed_day=elapsed_day, residual_mass=Probability.from_fraction(tail),
                         points=tuple(DayMass(cycle_length=elapsed_day + i, mass=Probability.from_fraction(m))
                                      for i, m in enumerate(masses)))


def from_hazards(elapsed_day: int, hazards: tuple[Fraction, ...]) -> DailyForecast:
    """hazards[i] = h(elapsed_day + i) = P(L = elapsed_day + i | L >= elapsed_day + i)."""
    hazards = _hazards(hazards)
    _days(elapsed_day, elapsed_day + len(hazards) - 1 if type(elapsed_day) is int else None)
    surviving, masses = Fraction(1), []
    for hazard in hazards:
        masses.append(surviving * hazard)
        surviving *= 1 - hazard
        # Reject once the survival product passes the budget, before the products grow further, even
        # though a later factor could have reduced it again. So from_hazards can reject a suffix of
        # hazards whose forecast condition() can still produce.
        if _bits(surviving) > PRECISION_BITS:
            raise InputError("fraction_precision_budget")
    return _forecast(elapsed_day, masses, surviving)


def condition(forecast: DailyForecast, elapsed_day: int) -> DailyForecast:
    """The same forecast given L >= elapsed_day. A no-onset report through day t gives L >= t + 1."""
    forecast = _valid(DailyForecast, forecast, "invalid_forecast")
    _days(elapsed_day, elapsed_day)
    if elapsed_day < forecast.elapsed_day:
        raise InputError("condition_before_forecast")
    kept = [p.mass.as_fraction() for p in forecast.points if p.cycle_length >= elapsed_day]
    tail = forecast.residual_mass.as_fraction()
    remaining = sum(kept, Fraction()) + tail
    if remaining == 0:
        raise InputError("support_exhausted")
    return _forecast(elapsed_day, [m / remaining for m in kept], tail / remaining)


def _risk_counts(cycles: tuple[ObservedCycle, ...], first_day: int, last_day: int) -> tuple[list[int], list[int]]:
    """For each day from first_day to last_day: cycles at risk, and cycles with their onset that day."""
    changes, onsets = [0] * (last_day - first_day + 2), [0] * (last_day - first_day + 1)
    for c in cycles:
        lower, upper = max(c.entry_day, first_day), min(c.last_day, last_day)
        if lower <= upper:
            changes[lower - first_day] += 1
            changes[upper - first_day + 1] -= 1
        if c.onset and first_day <= c.last_day <= last_day:
            onsets[c.last_day - first_day] += 1
    return list(accumulate(changes))[:-1], onsets


def pooled_hazards(cycles: tuple[ObservedCycle, ...], *, first_day: int, last_day: int) -> tuple[Fraction, ...]:
    """Population life table: for each day k, the posterior mean (D_k + 1/2) / (N_k + 1) under a
    Jeffreys Beta(1/2, 1/2) prior, where N_k cycles were at risk on day k and D_k had their onset."""
    cycles = _cycles(cycles)
    _days(first_day, last_day)
    if first_day > last_day:
        raise InputError("empty_window")
    at_risk, onsets = _risk_counts(cycles, first_day, last_day)
    if 0 in at_risk:
        raise InputError("empty_risk_set")
    return tuple((d + Fraction(1, 2)) / (n + 1) for n, d in zip(at_risk, onsets))


def shrunk_hazards(population: tuple[Fraction, ...], personal: tuple[ObservedCycle, ...], *,
                   first_day: int, strength: Fraction) -> tuple[Fraction, ...]:
    """Personal hazards (c mu_k + d_k) / (c + n_k): the population hazard mu_k counts as c cycles
    at risk. A day on which the person was never at risk keeps the population hazard exactly."""
    population, personal = _hazards(population), _cycles(personal)
    _days(first_day, first_day + len(population) - 1 if type(first_day) is int else None)
    if type(strength) is not Fraction or strength <= 0:
        raise InputError("invalid_strength")
    at_risk, onsets = _risk_counts(personal, first_day, first_day + len(population) - 1)
    return tuple((strength * mu + d) / (strength + n) for mu, n, d in zip(population, at_risk, onsets))
