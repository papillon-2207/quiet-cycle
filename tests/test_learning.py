from __future__ import annotations

import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from quietcycle import CycleDataset, InputError, Measurement, Symptom
from quietcycle.dates import add_days
from quietcycle.learning import (
    FitRequest, FittedModel, MultiFactorPipeline, build_cases, evaluate, fit,
    label_landmark, make_landmark, model_digest, predict_landmark,
)
from quietcycle.learning.exact import conformal_radius, ridge, round_day, solve
from quietcycle.learning.fitting import raw_length
from quietcycle.serialization import canonical_json, loads

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from synthetic_cohort import cohort, subject_fixture  # noqa: E402 (local example fixture path above)


@pytest.fixture(scope="module")
def experiment():
    request, test, inputs, rejected = cohort(train_n=24, tune_n=12, calibration_n=12, test_n=6)
    assert not rejected
    return request, fit(request), test, inputs


def changed(model, **kwargs):
    data = model.model_dump(mode="python")
    data.update(kwargs)
    return type(model).model_validate(data)


def test_complete_raw_pipeline(experiment):
    request, model, test, inputs = experiment
    output = MultiFactorPipeline(model).run(*inputs[0])
    assert output.status == "estimate"
    assert output == predict_landmark(model, test[0].landmark)
    assert output.point_date >= test[0].landmark.query.as_of
    assert output.clinical_validation == "not-established"
    assert model.selected.name != "baseline"
    assert sum((c.days.as_fraction() for c in output.contributions), output.baseline_cycle_length.as_fraction()) == output.raw_cycle_length.as_fraction()


def test_learning_1_0_identity_is_frozen(experiment):
    # learning-1.0 digests of 0.3.0 output for the default spec, new-subject split and this
    # fixture. They do not cover every code path.
    import hashlib
    request, model, test, inputs = experiment
    def sha(value):
        return hashlib.sha256(canonical_json(value).encode()).hexdigest()
    assert model_digest(model) == "43ae2e61580b90ea3fe8b2806407ea803a53de85c2d127681f5af817c3500a04"
    assert sha(request) == "597fe20d30814e297f1a4a81be82a824d789c261e85bdea4b943f01886fc97da"
    assert sha(test[0].landmark) == "7cd87bc40289daa449d441a61724c9d2918b87ba1d609bcf467a1745724ea1df"
    assert sha(MultiFactorPipeline(model).run(*inputs[0])) == "abbd71fea7d5584b7febbde093a1e1c93951d98c7dbb4eec58b88964c2da93f8"
    assert sha(evaluate(model, test)) == "0f5f24604ed92138931e281e9cdab6fbc12107bc045ad79336cc10a2c7717174"


def test_baseline_tune_error_uses_the_survival_floor(experiment):
    # No onset through yesterday means the cycle lasts at least elapsed_day days, so a personal
    # median below the landmark is floored before scoring. The fixture alone never binds the floor.
    from quietcycle import Rational
    req, _, _, _ = experiment
    ten = Rational.from_fraction(Fraction(10))
    def shortened(case):
        values = tuple(changed(v, value=ten) if v.name == "history.median" else v for v in case.landmark.values)
        return changed(case, landmark=changed(case.landmark, baseline_length=ten, values=values))
    tune = tuple(shortened(c) for c in req.tune)
    model = fit(changed(req, tune=tune))
    # Tune lengths 34 27 31 35 26 32 37 28 33 31 28 33, all scored against the floor 20:
    # (196+49+121+225+36+144+289+64+169+121+64+169) / 12 = 1647/12.
    assert [c.length for c in tune] == [34, 27, 31, 35, 26, 32, 37, 28, 33, 31, 28, 33]
    assert model.candidates[0].name == "baseline"
    assert model.candidates[0].tune_mse.as_fraction() == Fraction(549, 4)


def test_forward_personal_intervals_carry_no_coverage_claim(experiment):
    req, _, test, _ = experiment
    model = fit(changed(req, split_mode="forward-personal"))
    result = predict_landmark(model, test[0].landmark)
    assert result.status == "estimate"
    assert {i.assumption for i in result.intervals} == {"dependent-time-series-no-coverage-guarantee"}


def test_fit_is_order_invariant(experiment):
    request, model, _, _ = experiment
    reordered = changed(request, train=tuple(reversed(request.train)), tune=tuple(reversed(request.tune)), calibration=tuple(reversed(request.calibration)))
    assert canonical_json(fit(reordered)) == canonical_json(model)


def test_json_roundtrip(experiment):
    _, model, test, _ = experiment
    restored = FittedModel.model_validate(loads(canonical_json(model)))
    assert model_digest(restored) == model_digest(model)
    assert predict_landmark(restored, test[0].landmark) == predict_landmark(model, test[0].landmark)


def test_calibration_cannot_change_selected_coefficients(experiment):
    req, model, _, _ = experiment
    later = tuple(changed(c, outcome_date=add_days(c.outcome_date, 2), outcome_known_at=add_days(c.outcome_date, 3)+"T08:00:00Z") for c in req.calibration)
    refit = fit(changed(req, calibration=later))
    assert refit.selected == model.selected
    assert refit.columns == model.columns
    assert refit.calibration_scores != model.calibration_scores


def test_test_labels_are_not_an_input_to_prediction(experiment):
    _, model, test, _ = experiment
    case = test[0]
    later = changed(case, outcome_date=add_days(case.outcome_date, 7), outcome_known_at=add_days(case.outcome_date, 8)+"T08:00:00Z")
    assert predict_landmark(model, case.landmark) == predict_landmark(model, later.landmark)


def test_no_backdated_model(experiment):
    req, model, _, _ = experiment
    assert predict_landmark(model, req.calibration[0].landmark).reason == "model-not-available-at-cutoff"


def test_subject_overlap_rejected(experiment):
    req, _, _, _ = experiment
    original = req.tune[0]
    q = changed(original.landmark.query, subject_id=req.train[0].landmark.subject_id)
    lm = changed(original.landmark, subject_id=q.subject_id, query=q, anchor_id="different-anchor")
    modified = changed(original, landmark=lm)
    with pytest.raises(InputError, match="subject-overlap"):
        fit(changed(req, tune=(modified,) + req.tune[1:]))


def test_duplicate_case_rejected(experiment):
    req, _, _, _ = experiment
    with pytest.raises(InputError, match="duplicate-cycle"):
        fit(changed(req, train=req.train + (req.train[0],)))


def test_temporal_boundary_rejected(experiment):
    req, _, _, _ = experiment
    c = changed(req.train[-1], outcome_known_at="2024-02-01T00:00:00Z")
    with pytest.raises(InputError, match="partition-temporal-leakage"):
        fit(changed(req, train=req.train[:-1] + (c,)))


def test_synthetic_real_mix_rejected(experiment):
    req, _, _, _ = experiment
    c = changed(req.train[0], data_kind="observational")
    with pytest.raises(InputError, match="mixed-synthetic"):
        fit(changed(req, train=(c,) + req.train[1:]))


def test_spec_mismatch_rejected(experiment):
    req, _, _, _ = experiment
    with pytest.raises(InputError, match="landmark-spec"):
        fit(changed(req, spec=changed(req.spec, elapsed_day=21)))


def test_insufficient_calibration_is_unbounded(experiment):
    req, _, test, _ = experiment
    model = fit(changed(req, calibration=(req.calibration[0],)))
    result = predict_landmark(model, test[0].landmark)
    assert all(i.unbounded and i.radius_days is None and i.start is None and i.end is None for i in result.intervals)


def test_missing_day_not_no_onset(experiment):
    _, model, _, inputs = experiment
    data, q = inputs[0]
    reduced = CycleDataset(events=tuple(e for e in data.events if e.kind != "no-onset"))
    assert MultiFactorPipeline(model).run(reduced, q).reason == "onset-status-unconfirmed"


def test_no_future_backfill_leakage(experiment):
    _, model, _, inputs = experiment
    data, q = inputs[0]
    old = next(e for e in data.events if isinstance(e, Measurement) and e.metric == "wrist_temperature")
    later = changed(old, revision=2, value="100", recorded_at="2025-03-01T08:00:00Z")
    new = CycleDataset(events=data.events + (later,))
    assert MultiFactorPipeline(model).run(data, q) == MultiFactorPipeline(model).run(new, q)


def test_reversed_events_identical(experiment):
    _, model, _, inputs = experiment
    data, q = inputs[0]
    assert MultiFactorPipeline(model).run(data, q) == MultiFactorPipeline(model).run(CycleDataset(events=tuple(reversed(data.events))), q)


def test_mixed_stream_not_merged(experiment):
    req, _, _, inputs = experiment
    data, q = inputs[0]
    old = next(e for e in data.events if isinstance(e, Measurement) and e.metric == "wrist_temperature")
    duplicate = changed(old, id="other-device", provenance={"source": "other-device"})
    lm = make_landmark(CycleDataset(events=data.events+(duplicate,)), q, req.spec)
    assert "mixed-stream.wrist_temperature" in lm.issues
    assert all(v.value is None for v in lm.values if v.name.startswith("measurement.wrist_temperature"))


def test_missing_symptom_not_zero(experiment):
    req, _, _, inputs = experiment
    data, q = inputs[0]
    reduced = CycleDataset(events=tuple(e for e in data.events if not (isinstance(e, Symptom) and e.symptom == "fatigue")))
    lm = make_landmark(reduced, q, req.spec)
    assert next(v for v in lm.values if v.name == "symptom.fatigue.rate").value is None


def test_same_day_measurements_excluded(experiment):
    req, _, _, inputs = experiment
    data, q = inputs[0]
    m = Measurement(subject_id=q.subject_id, id="today", date=q.as_of, recorded_at=q.cutoff,
                    measured_at=q.as_of+"T07:00:00Z", utc_offset_minutes=0, metric="wrist_temperature", value="90", unit="Cel")
    a = make_landmark(data, q, req.spec)
    b = make_landmark(CycleDataset(events=data.events+(m,)), q, req.spec)
    assert a.values == b.values


def test_censored_record_is_accounted(experiment):
    req, _, _, inputs = experiment
    data, q = inputs[0]
    no_end = CycleDataset(events=tuple(e for e in data.events if e.id != "o5"))
    result = build_cases(no_end, (q,), spec=req.spec, labels_cutoff="2025-04-01T00:00:00Z", data_kind="synthetic", cohort="test")
    assert not result.cases
    # No report after the landmark: not observed. Absence of reports is not right-censoring.
    assert result.rejected[0][2] == "outcome-not-observed"


@pytest.mark.parametrize("labels_cutoff", [
    "2025-01-21T12:00:00Z",       # the landmark cutoff itself, spelled without milliseconds
    "2025-01-21T20:00:00+09:00",  # 11:00Z, an hour before the landmark cutoff
])
def test_labels_cutoff_must_be_a_later_instant(experiment, labels_cutoff):
    req, _, _, inputs = experiment
    data, q = inputs[0]
    assert q.cutoff == "2025-01-21T12:00:00.000Z"
    landmark = make_landmark(data, q, req.spec)
    with pytest.raises(InputError, match="label_cutoff_not_later"):
        label_landmark(data, landmark, labels_cutoff=labels_cutoff, data_kind="synthetic", cohort="test")
    # Each query has its own cutoff, so build_cases records this per case.
    built = build_cases(data, (q,), spec=req.spec, labels_cutoff=labels_cutoff, data_kind="synthetic", cohort="test")
    assert built.rejected == ((q.subject_id, q.as_of, "label_cutoff_not_later"),)


def test_a_later_instant_written_with_a_negative_offset_is_later(experiment):
    # 2025-01-21T11:00:00-05:00 is 16:00Z, after the 12:00Z landmark cutoff, though its text sorts first.
    req, _, _, inputs = experiment
    data, q = inputs[0]
    landmark = make_landmark(data, q, req.spec)
    outcome_known = next(e for e in data.events if e.id == "o5").available_at
    assert outcome_known > "2025-01-21T16:00:00.000Z"
    # Accepted as later; nothing has been reported yet at 16:00Z, so the outcome is not observed.
    with pytest.raises(InputError, match="outcome-not-observed"):
        label_landmark(data, landmark, labels_cutoff="2025-01-21T11:00:00-05:00", data_kind="synthetic", cohort="test")


def test_labels_day_past_the_date_domain_is_the_utc_day(experiment):
    # At +14:00, 2200-12-31T10:00Z is local 2201-01-01, outside the domain; the UTC day 2200-12-31 is used.
    from quietcycle import Query
    req, _, _, _ = experiment
    data, q = subject_fixture("late", "2200-11-20", 3)
    late = Query.model_validate({**q.model_dump(), "cutoff": "2200-12-10T23:00:00+14:00", "utc_offset_minutes": 840})
    landmark = make_landmark(data, late, req.spec)
    case = label_landmark(data, landmark, labels_cutoff="2200-12-31T10:00:00Z", data_kind="synthetic", cohort="test")
    assert case.outcome_date == "2200-12-18"


def test_malformed_labels_cutoff_is_an_input_error(experiment):
    req, _, _, inputs = experiment
    data, q = inputs[0]
    landmark = make_landmark(data, q, req.spec)
    with pytest.raises(InputError, match="invalid_labels_cutoff"):
        label_landmark(data, landmark, labels_cutoff="2025-04-01", data_kind="synthetic", cohort="test")
    with pytest.raises(InputError, match="invalid_labels_cutoff"):
        build_cases(data, (q,), spec=req.spec, labels_cutoff="2025-04-01", data_kind="synthetic", cohort="test")


def test_revised_anchor_not_relabelled(experiment):
    req, _, _, inputs = experiment
    data, q = inputs[0]
    lm = make_landmark(data, q, req.spec)
    old = next(e for e in data.events if e.id == "o4")
    new = changed(old, revision=2, date=add_days(old.date, 1), recorded_at="2025-03-01T08:00:00Z")
    with pytest.raises(InputError, match="anchor-revised"):
        label_landmark(CycleDataset(events=data.events+(new,)), lm, labels_cutoff="2025-04-01T00:00:00Z", data_kind="synthetic", cohort="test")


@pytest.mark.parametrize("field,value", [("paused", True), ("mode", "other"), ("mode", "unknown"), ("changed_since", "2025-01-15")])
def test_context_restrictions(experiment, field, value):
    _, model, _, inputs = experiment
    data, q = inputs[0]
    changed_q = changed(q, context=changed(q.context, **{field:value}))
    assert MultiFactorPipeline(model).run(data, changed_q).status == "abstain"


def test_evaluation_keeps_attempts(experiment):
    req, model, test, _ = experiment
    report = evaluate(model, test + (req.calibration[0],))
    assert report["attempted"] == len(test)+1
    assert report["estimated"] == len(test)
    assert report["abstained"] == 1
    assert report["clinical_validation"] == "not-established"


def test_prediction_changes_with_observed_signal(experiment):
    req, model, _, inputs = experiment
    data, q = inputs[0]
    modified = []
    recent_start = add_days(q.as_of, -3)
    for e in data.events:
        if isinstance(e, Measurement) and e.metric == "wrist_temperature" and e.date >= recent_start:
            e = changed(e, value="37")
        modified.append(e)
    a = make_landmark(data, q, req.spec)
    b = make_landmark(CycleDataset(events=tuple(modified)), q, req.spec)
    assert raw_length(model, a)[0] != raw_length(model, b)[0]


@pytest.mark.parametrize("v,expected", [(Fraction(1,2),1),(Fraction(-1,2),0),(Fraction(3,2),2),(Fraction(19,10),2),(Fraction(14,10),1)])
def test_rounding(v, expected):
    assert round_day(v) == expected


@pytest.mark.parametrize("n,p,expected", [(0,900000,None),(1,900000,None),(8,900000,None),(9,900000,8),(4,800000,3)])
def test_conformal_rank(n,p,expected):
    assert conformal_radius(tuple(range(n)),p) == expected


def test_rank_coverage_enumerated():
    hits = total = 0
    for permutation in itertools.permutations(range(5)):
        radius = conformal_radius(permutation[:4],800000)
        hits += int(radius is None or permutation[4] <= radius)
        total += 1
    assert Fraction(hits,total) == Fraction(4,5)


@pytest.mark.parametrize("penalty",[1,3,100])
def test_ridge_normal_equation_exact(penalty):
    rng = random.Random(437)
    for _ in range(20):
        rows = [tuple([Fraction(1)] + [Fraction(rng.randrange(-7,8),rng.randrange(1,4)) for _ in range(3)]) for _ in range(8)]
        outcomes = [Fraction(rng.randrange(-10,11),3) for _ in rows]
        beta = ridge(rows,outcomes,penalty)
        for j in range(4):
            gradient = sum((x[j]*(sum((x[k]*beta[k] for k in range(4)),Fraction())-y) for x,y in zip(rows,outcomes)),Fraction())
            gradient += penalty*beta[j] if j else 0
            assert gradient == 0


def test_against_independent_numpy_solution():
    np = pytest.importorskip("numpy")
    rows = [(Fraction(1),Fraction(i),Fraction(i%3)) for i in range(8)]
    y = [Fraction(i*i%7) for i in range(8)]
    exact = ridge(rows,y,3)
    a = np.array(rows,dtype=float)
    b = np.array(y,dtype=float)
    ref = np.linalg.solve(a.T@a+np.diag([0,3,3]),a.T@b)
    assert np.max(np.abs(ref-np.array(exact,dtype=float))) < 1e-12


@pytest.mark.parametrize("bad", [0,-1,True,1.0])
def test_ridge_invalid_penalty(bad):
    with pytest.raises(InputError):
        ridge([(Fraction(1),)], [Fraction(2)], bad)


def test_collinear_features_still_solve():
    beta = ridge([(Fraction(1),Fraction(2),Fraction(2))]*8,[Fraction(3)]*8,1)
    assert beta == (Fraction(3),Fraction(0),Fraction(0))


def test_singular_unregularized_solver_rejected():
    with pytest.raises(InputError,match="singular"):
        solve([[Fraction(0)]],[Fraction(1)])


def test_no_float_fallback():
    with pytest.raises(InputError,match="fraction_matrix"):
        ridge([(Fraction(1),2.0)],[Fraction(3)],1)


def test_dependency_free_learning_import():
    import subprocess
    code = "import sys; before=set(sys.modules); import quietcycle.learning; assert not any(n in set(sys.modules)-before for n in ['numpy','scipy','sklearn','torch','fastapi'])"
    subprocess.run([sys.executable,"-c",code],check=True)

@pytest.mark.parametrize("context,reason", [({"mode":"natural-cycle","paused":True},"paused"),
    ({"mode":"other"},"outside-declared-scope"), ({"mode":"unknown"},"outside-declared-scope"),
    ({"mode":"natural-cycle","changed_since":"2025-01-02"},"context-changed-within-current-cycle")])
def test_direct_landmark_cannot_bypass_scope(experiment, context, reason):
    _, model, test, _ = experiment
    from quietcycle import CycleContext
    lm = test[0].landmark
    altered = changed(lm, query=changed(lm.query, context=CycleContext(**context)))
    assert predict_landmark(model, altered).reason == reason


def test_fit_direct_cases_cannot_bypass_scope(experiment):
    req, _, _, _ = experiment
    from quietcycle import CycleContext
    case = req.train[0]
    altered = changed(case, landmark=changed(case.landmark,
        query=changed(case.landmark.query, context=CycleContext(mode="other"))))
    with pytest.raises(InputError, match="outside-scope"):
        fit(changed(req, train=(altered,)+req.train[1:]))


def test_learned_http_same_as_sdk(experiment):
    from fastapi.testclient import TestClient
    from quietcycle import PipelineRequest
    from quietcycle.api import create_app
    _, model, _, inputs = experiment
    data, query = inputs[0]
    body = PipelineRequest(dataset=data,query=query)
    app = create_app(model=model)
    with TestClient(app) as client:
        response = client.post("/v3/forecast",content=canonical_json(body), headers={"Content-Type":"application/json"})
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert response.json() == MultiFactorPipeline(model).run(data,query).model_dump(mode="json")
        assert "/v3/forecast" in client.get("/openapi.json").json()["paths"]
        assert client.post("/v3/forecast",content='{"dataset":{},"dataset":{}}',headers={"Content-Type":"application/json"}).status_code == 422
        assert client.get("/docs").status_code == 404
    assert "/v3/forecast" not in create_app().openapi()["paths"]


def test_learned_cli_roundtrip(experiment, tmp_path):
    import subprocess
    from quietcycle import PipelineRequest
    req, model, test, inputs = experiment
    paths = {n: tmp_path/(n+".json") for n in ("fit", "model", "request", "cases")}
    paths["fit"].write_text(canonical_json(req))
    paths["request"].write_text(canonical_json(PipelineRequest(dataset=inputs[0][0],query=inputs[0][1])))
    paths["cases"].write_text(canonical_json(test))
    subprocess.run([sys.executable,"-m","quietcycle","fit",str(paths["fit"]),"-o",str(paths["model"])],check=True,capture_output=True)
    restored = FittedModel.model_validate(loads(paths["model"].read_text()))
    assert restored == model
    raw = subprocess.check_output([sys.executable,"-m","quietcycle","forecast",str(paths["request"]),"--model",str(paths["model"])],text=True)
    assert json.loads(raw) == MultiFactorPipeline(model).run(*inputs[0]).model_dump(mode="json")
    raw = subprocess.check_output([sys.executable,"-m","quietcycle","evaluate",str(paths["cases"]),"--model",str(paths["model"])],text=True)
    assert json.loads(raw) == json.loads(canonical_json(evaluate(model,test)))


def test_learning_schema_contracts(experiment):
    from jsonschema import Draft202012Validator
    from quietcycle.learning import LearnedPrediction, LabelledCase, Landmark
    request, model, test, _ = experiment
    for value, cls in [(request,FitRequest),(model,FittedModel),(test[0],LabelledCase),
                        (test[0].landmark,Landmark),(predict_landmark(model,test[0].landmark),LearnedPrediction)]:
        schema = cls.model_json_schema()
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value.model_dump(mode="json"))


def test_model_notebook_no_hidden_clock_or_random(experiment, tmp_path):
    import os
    import subprocess
    _, model, test, _ = experiment
    path = tmp_path/"model.json"
    path.write_text(canonical_json(model))
    lm = tmp_path/"landmark.json"
    lm.write_text(canonical_json(test[0].landmark))
    code = '''import sys
from quietcycle.learning import FittedModel, Landmark, predict_landmark
from quietcycle.serialization import canonical_json,loads
from pathlib import Path
model=FittedModel.model_validate(loads(Path(sys.argv[1]).read_text()))
lm=Landmark.model_validate(loads(Path(sys.argv[2]).read_text()))
print(canonical_json(predict_landmark(model,lm)))
'''
    outputs = [subprocess.check_output([sys.executable,"-c",code,str(path),str(lm)],env={**os.environ,"PYTHONHASHSEED":seed}) for seed in ("0","7","123456")]
    assert outputs[0] == outputs[1] == outputs[2]


def test_late_discovered_outcome_before_landmark_is_rejected(experiment):
    req, _, _, inputs = experiment
    data, query = inputs[0]
    modified = []
    for event in data.events:
        if event.id == "o5":
            event = changed(event, date=add_days(query.as_of,-1), recorded_at="2025-03-01T08:00:00Z")
        modified.append(event)
    result = build_cases(CycleDataset(events=tuple(modified)),(query,),spec=req.spec,
        labels_cutoff="2025-04-01T00:00:00Z",data_kind="synthetic",cohort="test")
    assert not result.cases
    assert result.rejected[0][2] == "retrospective-label-before-landmark"


def test_blend_matches_exact_tune_equation(experiment):
    from quietcycle.learning.fitting import vector
    request, model, _, _ = experiment
    candidate = model.selected
    corrections = []
    residuals = []
    for case in request.tune:
        x,_ = vector(case.landmark,model.columns,model.clipping,model.quantization)
        corrections.append(sum((c.as_fraction()*x[t] for t,c in zip(candidate.terms,candidate.coefficients)),Fraction()))
        residuals.append(Fraction(case.length)-case.landmark.baseline_length.as_fraction())
    denominator = sum((v*v for v in corrections),Fraction())
    expected = min(Fraction(1),max(Fraction(),sum((g*r for g,r in zip(corrections,residuals)),Fraction())/denominator)) if denominator else Fraction()
    assert candidate.blend.as_fraction() == expected
