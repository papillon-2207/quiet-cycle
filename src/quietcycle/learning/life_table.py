"""Life-table baselines for the event-time comparison, in exact arithmetic.

The pooled hazards come from complete cycle lengths; a forecast shrinks them toward the person's
clean history, as a landmark selects it, and starts at the landmark day. They are comparators, not
a trained model of anything but cycle length.

The lengths carry no dates, so the caller must take them from data available before any forecast
cutoff it will be compared at. Leaving out cycles still open at that time makes long cycles look
rarer than they are; a cycle longer than the horizon counts as at risk through it, with no onset.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, Field, StrictInt, model_validator

from ..errors import InputError
from ..features import DefaultFeatureBuilder
from ..ledger import snapshot
from ..models import CycleDataset, Model, Probability, Query, Rational, _tuple, array_bounds
from ..serialization import canonical_json, parse_model
from .contracts import LandmarkSpec
from .landmarks import _recent, _unit_registry, make_landmark
from .survival import MAX_DAY, DailyForecast, ObservedCycle, from_hazards, pooled_hazards, shrunk_hazards


class LifeTableModel(Model):
    model_id: Literal["life-table-shrinkage-v1"] = "life-table-shrinkage-v1"
    backend: Literal["exact-fraction"] = "exact-fraction"
    spec: LandmarkSpec
    hazards: Annotated[tuple[Probability, ...], BeforeValidator(_tuple), Field(min_length=1, max_length=MAX_DAY, json_schema_extra=array_bounds)]
    strength: Rational
    train_cycles: Annotated[StrictInt, Field(ge=1)]
    training_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len(self.hazards) < self.spec.elapsed_day:
            raise ValueError("hazards_before_landmark")
        if any(not 0 < h.as_fraction() < 1 for h in self.hazards):
            raise ValueError("hazard_not_strictly_inside")
        if self.strength.as_fraction() <= 0:
            raise ValueError("invalid_strength")
        return self


def _cycles(lengths: tuple[int, ...], horizon: int) -> tuple[ObservedCycle, ...]:
    # Past the horizon only "no onset yet" matters, so a longer cycle is censored at it.
    return tuple(ObservedCycle(entry_day=1, last_day=min(n, horizon), onset=n <= horizon) for n in lengths)


def history_lengths(dataset: CycleDataset, query: Query, spec: LandmarkSpec) -> tuple[int, ...]:
    """The clean history a landmark at this query uses. A query make_landmark refuses is refused with
    the same code."""
    make_landmark(dataset, query, spec)
    dataset, query, spec = parse_model(CycleDataset, dataset), parse_model(Query, query), parse_model(LandmarkSpec, spec)
    features = DefaultFeatureBuilder(_unit_registry(spec)).build(snapshot(dataset, query), query)
    return tuple(i.length for i in _recent(features, spec))


def fit_life_table(lengths: tuple[int, ...], *, spec: LandmarkSpec, horizon: int, strength: Fraction) -> LifeTableModel:
    """Pooled hazards for days 1 to horizon from complete cycle lengths. A horizon past the longest
    length leaves a day with no cycle at risk and fails."""
    if type(lengths) is not tuple or not lengths or any(type(n) is not int or not 1 <= n <= MAX_DAY for n in lengths):
        raise InputError("invalid_lengths")
    if type(strength) is not Fraction or strength <= 0:
        raise InputError("invalid_strength")
    spec = parse_model(LandmarkSpec, spec)
    if type(horizon) is not int or not 1 <= horizon <= MAX_DAY:
        raise InputError("invalid_horizon")
    if horizon < spec.elapsed_day:
        raise InputError("horizon_before_landmark")
    hazards = pooled_hazards(_cycles(lengths, horizon), first_day=1, last_day=horizon)
    digest = canonical_json({"spec": spec, "strength": Rational.from_fraction(strength), "lengths": sorted(lengths), "horizon": horizon})
    return LifeTableModel(spec=spec, hazards=tuple(Probability.from_fraction(h) for h in hazards),
                          strength=Rational.from_fraction(strength), train_cycles=len(lengths),
                          training_sha256=sha256(digest.encode()).hexdigest())


def forecast_life_table(model: LifeTableModel, dataset: CycleDataset, query: Query) -> DailyForecast:
    """P(L = k | L >= d) from the landmark day d on, with the person's clean history shrinking the
    pooled hazards; days past the model's horizon are the residual mass."""
    model = parse_model(LifeTableModel, model)
    personal = _cycles(history_lengths(dataset, query, model.spec), len(model.hazards))
    shrunk = shrunk_hazards(tuple(h.as_fraction() for h in model.hazards), personal, first_day=1,
                            strength=model.strength.as_fraction())
    elapsed = model.spec.elapsed_day
    return from_hazards(elapsed, shrunk[elapsed - 1:])
