import json
from fractions import Fraction
from pathlib import Path

import pytest

from quietcycle import CycleDataset, CyclePipeline, Measurement, Query
from quietcycle.adapters import CSVAdapter, FHIRObservationAdapter, from_json, from_jsonl, from_records, from_v1, to_csv, to_jsonl
from quietcycle.errors import InputError
from quietcycle.serialization import canonical_json
from quietcycle.units import UnitRegistry, UnitRule

ROOT = Path(__file__).resolve().parents[1]


def measured(**changes):
    data = dict(subject_id="demo", id="m", date="2025-07-01", recorded_at="2025-07-01T12:00:00Z", utc_offset_minutes=0,
                metric="body_temperature", value="98.6", unit="[degF]", measured_at="2025-07-01T11:00:00Z")
    data.update(changes)
    return Measurement(**data)


def test_all_file_formats_are_equivalent(sample):
    pipeline = CyclePipeline()
    expected = canonical_json(pipeline.run(sample.dataset, sample.query))
    options = [from_json(canonical_json(sample.dataset)), from_jsonl(to_jsonl(sample.dataset)),
               CSVAdapter().loads(to_csv(sample.dataset)),
               from_records([e.model_dump(mode="json") for e in sample.dataset.events])]
    for dataset in options:
        assert canonical_json(pipeline.run(dataset, sample.query)) == expected


def test_csv_alias_and_defaults(sample):
    data = to_csv(sample.dataset).replace("subject_id,id,", "person,event_id,", 1)
    parsed = CSVAdapter(column_map={"person": "subject_id", "event_id": "id"}).loads(data)
    assert parsed == sample.dataset


@pytest.mark.parametrize("text", ["", "id,id\na,b\n", "unknown\na\n", "id,kind\na,onset,extra\n", 'id,kind\n"unclosed,onset\n'])
def test_invalid_csv(text):
    with pytest.raises(InputError):
        CSVAdapter().loads(text)


def test_csv_no_bool_guessing(sample):
    text = to_csv(sample.dataset)
    with pytest.raises(InputError, match="csv_boolean_required"):
        CSVAdapter().loads(text.replace(",false,", ",0,", 1))


def test_mixed_measurement_csv_roundtrip(sample):
    dataset = sample.dataset.merge(CycleDataset(events=(measured(),)))
    assert CSVAdapter().loads(to_csv(dataset)) == dataset


def test_measurements_converted_exactly(sample):
    event = measured()
    result = CyclePipeline().run(sample.dataset.merge(CycleDataset(events=(event,))), sample.query)
    assert result.measurements[0].value.as_fraction() == 37
    assert result.measurements[0].unit == "Cel"
    assert result.measurements[0].used_by_default_model is False
    assert result.forecast == CyclePipeline().run(sample.dataset, sample.query).forecast


@pytest.mark.parametrize("metric,value,unit,expected", [
    ("heart_rate", "1.25", "1/s", Fraction(75)),
    ("hrv_rmssd", "0.045", "s", Fraction(45)),
    ("sleep_duration", "7.5", "h", Fraction(27000)),
    ("body_temperature", "310.15", "K", Fraction(37)),
    ("body_temperature", "98.6", "[degF]", Fraction(37)),
    ("body_temperature", "32", "[degF]", Fraction(0)),
    ("skin_temperature", "33.125", "Cel", Fraction(265, 8)),
])
def test_unit_cases(metric, value, unit, expected):
    assert UnitRegistry().normalize(measured(metric=metric, value=value, unit=unit)).value.as_fraction() == expected


def test_unknown_units_and_metric_identity():
    for metric, unit in [("body_temperature", "/min"), ("unknown", "Cel")]:
        with pytest.raises(InputError):
            UnitRegistry().normalize(measured(metric=metric, unit=unit))
    assert UnitRegistry().normalize(measured(metric="skin_temperature")).metric != "body_temperature"
    with pytest.raises(InputError):
        UnitRegistry().normalize(measured(metric="steps", value="1.5", unit="{count}"))


def test_registry_extension_is_local():
    registry = UnitRegistry({("vendor.example", "1"): UnitRule("1", Fraction(2))})
    event = measured(metric="vendor.example", value="5", unit="1")
    assert registry.normalize(event).value.as_fraction() == 10
    with pytest.raises(InputError):
        UnitRegistry().normalize(event)
    with pytest.raises(InputError):
        UnitRegistry({("body_temperature", "Cel"): UnitRule("Cel")})


def fhir_observation(**changes):
    data = {
        "resourceType": "Observation", "id": "o1", "status": "final",
        "subject": {"reference": "Patient/p1"}, "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
        "effectiveDateTime": "2025-07-01T10:00:00Z", "issued": "2025-07-01T11:00:00Z",
        "valueQuantity": {"value": 72, "system": "http://unitsofmeasure.org", "code": "/min"},
    }
    data.update(changes)
    return data


def fhir_adapter():
    return FHIRObservationAdapter(subject_map={"Patient/p1": "demo"}, imported_at="2025-07-02T00:00:00Z", utc_offset_minutes=0)


def test_fhir_import_does_not_invent_onset(sample):
    dataset = fhir_adapter().loads(json.dumps(fhir_observation()))
    assert dataset.events[0].kind == "measurement"
    result = CyclePipeline().run(sample.dataset.merge(dataset), sample.query)
    assert result.measurements[0].value.as_fraction() == 72
    assert result.forecast == CyclePipeline().run(sample.dataset, sample.query).forecast


def test_fhir_decimal_and_import_time(sample):
    item = fhir_observation(valueQuantity={"value": 72.25, "system": "http://unitsofmeasure.org", "code": "/min"})
    data = fhir_adapter().loads(json.dumps(item))
    assert data.events[0].value == "72.25"
    assert data.events[0].available_at == "2025-07-02T00:00:00.000Z"
    before = Query.model_validate({**sample.query.model_dump(), "as_of": "2025-07-01", "cutoff": "2025-07-01T23:00:00Z"})
    assert CyclePipeline().run(data, before).observation_count == 0


@pytest.mark.parametrize("changes", [
    {"status": "preliminary"}, {"subject": {"reference": "Patient/other"}},
    {"component": []}, {"effectivePeriod": {}}, {"valueString": "high"},
    {"modifierExtension": []}, {"valueQuantity": {"value": 1, "code": "/min"}},
    {"code": {"coding": [{"system": "http://loinc.org", "code": "unknown"}]}},
])
def test_fhir_unsupported_is_explicit(changes):
    with pytest.raises(InputError):
        fhir_adapter().from_mapping(fhir_observation(**changes))


@pytest.mark.parametrize("coding", [
    [{"system": "http://loinc.org", "code": ["8867-4"]}],
    [{"system": {"nested": 1}, "code": "8867-4"}, {"system": "http://loinc.org", "code": "8867-4"}],
])
def test_fhir_unhashable_coding_is_rejected(coding):
    with pytest.raises(InputError, match="invalid_fhir_observation") as caught:
        fhir_adapter().from_mapping({"resourceType": "Bundle", "type": "collection",
                                     "entry": [{"resource": fhir_observation(code={"coding": coding})}]})
    assert caught.value.row == 1


@pytest.mark.parametrize("value", ["absent", None])
def test_fhir_effective_time_is_required(value):
    item = fhir_observation()
    if value == "absent":
        del item["effectiveDateTime"]
    else:
        item["effectiveDateTime"] = value
    with pytest.raises(InputError, match="invalid_fhir_observation"):
        fhir_adapter().from_mapping(item)


def test_fhir_non_mapping_coding_entries_are_skipped():
    coding = ["not-a-coding", {"system": "http://loinc.org", "code": "8867-4"}]
    assert fhir_adapter().from_mapping(fhir_observation(code={"coding": coding})).events[0].metric == "heart_rate"


def test_fhir_two_mapped_body_sites_are_ambiguous():
    site = {"coding": [{"system": "urn:site", "code": "a"}, {"system": "urn:site", "code": "b"}]}
    adapter = FHIRObservationAdapter(subject_map={"Patient/p1": "demo"}, imported_at="2025-07-02T00:00:00Z",
                                     utc_offset_minutes=0, body_site_map={("urn:site", "a"): "wrist", ("urn:site", "b"): "finger"})
    with pytest.raises(InputError, match="unmapped_fhir_bodysite"):
        adapter.from_mapping(fhir_observation(bodySite=site))


def test_csv_missing_measurement_keeps_its_null_value():
    text = to_csv(CycleDataset(events=(measured(state="missing", value=None, missing_reason="not-worn"),)))
    event = CSVAdapter().loads(text).events[0]
    assert (event.state, event.value, event.missing_reason) == ("missing", None, "not-worn")


def test_fhir_two_mapped_codes_are_ambiguous():
    coding = [{"system": "http://loinc.org", "code": "8867-4"}, {"system": "http://loinc.org", "code": "8310-5"}]
    with pytest.raises(InputError, match="unmapped_or_ambiguous_fhir_code"):
        fhir_adapter().from_mapping(fhir_observation(code={"coding": coding}))


def test_fhir_missing_value_not_zero():
    item = fhir_observation()
    del item["valueQuantity"]
    item["dataAbsentReason"] = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/data-absent-reason", "code": "not-asked"}]}
    data = fhir_adapter().from_mapping(item)
    assert data.events[0].state == "missing"
    assert data.events[0].value is None


def test_fhir_bundle_and_pagination():
    bundle = {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": fhir_observation()}]}
    assert len(fhir_adapter().from_mapping(bundle).events) == 1
    bundle["link"] = [{"relation": "next", "url": "https://unused.invalid"}]
    with pytest.raises(InputError, match="incomplete_fhir_pagination"):
        fhir_adapter().from_mapping(bundle)


def test_legacy_migration(sample):
    old = json.loads((ROOT / "examples/legacy-v1.json").read_text())
    assert from_v1(old, subject_id="demo") == sample
    imported = from_v1(old, subject_id="demo", imported_at="2025-07-03T00:00:00Z")
    assert CyclePipeline().run(imported.dataset, imported.query).observation_count == 0


def test_fhir_sources_and_site_mapping():
    from pathlib import Path
    import json
    raw = json.loads((Path(__file__).resolve().parents[1] / "examples/fhir-observation.json").read_text(), parse_float=str)
    raw["bodySite"] = {"coding": [{"system": "urn:example", "code": "oral"}]}
    base = dict(subject_map={"Patient/example": "demo"}, imported_at="2025-07-02T12:00:00Z", utc_offset_minutes=0)
    with pytest.raises(InputError, match="unmapped_fhir_bodysite"):
        FHIRObservationAdapter(**base).from_mapping(raw)
    a = FHIRObservationAdapter(**base, source_id="server-a", body_site_map={("urn:example", "oral"): "oral"}).from_mapping(raw)
    b = FHIRObservationAdapter(**base, source_id="server-b", body_site_map={("urn:example", "oral"): "oral"}).from_mapping(raw)
    assert a.events[0].body_site == "oral"
    assert a.events[0].id != b.events[0].id
    assert len(a.merge(b).events) == 2


@pytest.mark.parametrize("kind", [["onset"], {"kind": "onset"}, {"onset"}])
def test_csv_kind_that_is_not_text_is_a_row_error(kind):
    # A default kind of the wrong type reached a dictionary lookup and raised TypeError.
    lines = (ROOT / "examples/events.csv").read_text(encoding="utf-8").splitlines()
    headers = lines[0].split(",")
    without_kind = "\n".join(",".join(v for h, v in zip(headers, line.split(",")) if h != "kind") for line in lines) + "\n"
    with pytest.raises(InputError) as caught:
        CSVAdapter(defaults={"kind": kind}).loads(without_kind)
    assert (caught.value.code, caught.value.row) == ("schema_validation_failed", 2)
    # With a kind column the default applies only to a row whose kind cell is blank.
    second = lines[2].split(",")
    second[headers.index("kind")] = ""
    blank = "\n".join([*lines[:2], ",".join(second), *lines[3:]]) + "\n"
    with pytest.raises(InputError) as caught:
        CSVAdapter(defaults={"kind": kind}).loads(blank)
    assert (caught.value.code, caught.value.row) == ("schema_validation_failed", 3)
    assert len(CSVAdapter(defaults={"kind": kind}).loads("\n".join(lines) + "\n").events) == len(lines) - 1
@pytest.mark.parametrize("position", [0, 2])
def test_legacy_deleted_event_without_kind_is_a_row_error(position, tmp_path, capsys):
    # A deleted v1 event without kind raised KeyError; an undeleted one was already a row error.
    from quietcycle.cli import main
    old = json.loads((ROOT / "examples/legacy-v1.json").read_text())
    event = old["events"][position]
    del event["kind"]
    event["deleted"] = True
    with pytest.raises(InputError) as caught:
        from_v1(old, subject_id="demo")
    assert (caught.value.code, caught.value.row) == ("schema_validation_failed", position + 1)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    assert main(["migrate-v1", str(path), "--subject-id", "demo"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "schema_validation_failed"
