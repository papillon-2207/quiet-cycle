"""Focused mutations, one per protected behavior; not an exhaustive mutation-coverage claim."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    ("missing-becomes-negative", "predictors.py", "        if past:\n", "        if False:\n", "tests/test_pipeline.py::test_missing_is_not_no_onset"),
    ("future-report-leak", "ledger.py", "event.subject_id != query.subject_id or event.available_at > query.cutoff", "event.subject_id != query.subject_id", "tests/test_pipeline.py::test_late_revision_is_invisible_in_replay"),
    ("wrong-fahrenheit-scale", "units.py", "Fraction(5, 9)", "Fraction(4, 9)", "tests/test_adapters.py::test_unit_cases"),
    ("csv-boolean-inversion", "adapters/csv.py", 'record[name] = text_value == "true"', 'record[name] = text_value == "false"', "tests/test_adapters.py::test_all_file_formats_are_equivalent"),
    ("duplicate-json-last-wins", "serialization.py", "        if key in out:\n", "        if False:\n", "tests/test_models.py::test_wire_parser_rejects_ambiguous_json"),
    ("fhir-source-collision", "adapters/fhir.py", 'self.source_id + "\\0" + reference', 'reference', "tests/test_adapters.py::test_fhir_sources_and_site_mapping"),
    ("drop-residual-mass", "research/duration.py", "Probability.from_fraction(sum(mass.values(), Fraction()))", "Probability.from_fraction(Fraction())", "tests/test_research.py::test_duration_simple_and_residual"),
    ("kernel-flat-not-triangular", "predictors.py", "bandwidth + 1 - abs(delta)", "bandwidth + 1", "tests/test_reference.py::test_1400_kernel_and_allocation_cases"),

    ("conformal-missing-infinity-rank", "learning/exact.py", "(len(scores) + 1) * coverage_ppm", "len(scores) * coverage_ppm", "tests/test_learning.py::test_conformal_rank"),
    ("model-backdated", "learning/prediction.py", "if q.cutoff <= model.available_at:", "if False:", "tests/test_learning.py::test_no_backdated_model"),
    ("direct-scope-bypass", "learning/prediction.py", "if q.context.mode != \"natural-cycle\":", "if False:", "tests/test_learning.py::test_direct_landmark_cannot_bypass_scope"),
    ("round-ties-wrong", "learning/exact.py", "shifted = value + Fraction(1, 2)", "shifted = value", "tests/test_learning.py::test_rounding"),
    ("regularize-intercept", "learning/exact.py", "penalty if i == j and i else 0", "penalty if i == j else 0", "tests/test_learning.py::test_ridge_normal_equation_exact"),
    ("schema-array-string-keywords", "models.py", "            schema[array_name] = schema.pop(name)\n", "            pass\n", "tests/test_interfaces.py::test_array_bounds_use_array_keywords"),
    ("schema-array-bound-dropped", "models.py", "            schema[array_name] = schema.pop(name)\n", "            schema.pop(name)\n", "tests/test_interfaces.py::test_each_tuple_limit_is_published"),
    ("declared-unit-rules-ignored", "learning/landmarks.py", "DefaultFeatureBuilder(registry)", "DefaultFeatureBuilder()", "tests/test_unit_rules.py::test_rows_and_csv_and_both_units_give_the_same_features"),
    ("builtin-rule-not-skipped", "learning/landmarks.py", "for r in spec.unit_rules if not r.same_as_builtin()})", "for r in spec.unit_rules})", "tests/test_unit_rules.py::test_rule_identical_to_a_builtin_is_accepted_as_a_no_op"),
    ("two-canonical-units-allowed", "learning/contracts.py", "            if len(units) > 1:\n", "            if False:\n", "tests/test_unit_rules.py::test_invalid_unit_rules"),
    ("contract-version-unchecked", "learning/contracts.py", "    if (schema_version == \"learning-1.1\") != bool(spec.unit_rules):\n", "    if False:\n", "tests/test_unit_rules.py::test_contract_version_follows_unit_rules"),
    ("identity-rule-unchecked", "learning/contracts.py", "        if self.unit == self.canonical_unit and", "        if False and", "tests/test_unit_rules.py::test_invalid_unit_rules"),
    ("unit-offset-dropped", "learning/landmarks.py", "r.scale.as_fraction(), r.offset.as_fraction())\n", "r.scale.as_fraction(), Fraction(0))\n", "tests/test_unit_rules.py::test_scale_and_offset_convert_exactly"),
    ("landmark-validation-escapes", "learning/landmarks.py", "    except ValidationError as exc:\n", "    except KeyError as exc:\n", "tests/test_unit_rules.py::test_feature_precision_overflow_is_a_recorded_rejection"),
    ("validation-error-chained", "learning/landmarks.py", "else \"invalid_landmark\") from None", "else \"invalid_landmark\") from exc", "tests/test_unit_rules.py::test_feature_precision_overflow_is_a_recorded_rejection"),
    ("long-feature-names-accepted", "learning/contracts.py", "        if any(len(name) > 96 for name in expected_feature_names(self)):\n", "        if False:\n", "tests/test_unit_rules.py::test_derived_feature_names_fit_the_identifier_limit"),
    ("declared-metric-name-unbounded", "learning/contracts.py", "        if any(len(f\"untrained-feature.measurement.{m}.recent\") > 96 for m in self.metrics if m in declared):\n", "        if False:\n", "tests/test_unit_rules.py::test_a_declared_metric_name_fits_its_forecast_issues"),
    ("unused-inputs-dropped", "learning/prediction.py", "        if not names:\n            return result\n", "        return result\n", "tests/test_unused_inputs.py::test_unused_inputs_are_named_without_values"),
    ("unused-symptoms-ignored", "learning/landmarks.py", "        elif isinstance(event, Symptom) and event.symptom not in spec.symptoms:\n", "        elif False:\n", "tests/test_unused_inputs.py::test_unused_inputs_are_named_without_values"),
    ("snapshot-error-escapes", "learning/prediction.py", "        except InputError:\n            # Without a valid snapshot", "        except KeyError:\n            # Without a valid snapshot", "tests/test_unused_inputs.py::test_an_invalid_snapshot_abstains_without_a_list"),
    ("unused-uniqueness-unpublished", "learning/contracts.py", "json_schema_extra={\"uniqueItems\": True}", "json_schema_extra={}", "tests/test_unused_inputs.py::test_published_schema_rejects_duplicate_unused_inputs"),
    ("unused-order-unchecked", "learning/contracts.py", "        if list(self.unused_inputs) != sorted(set(self.unused_inputs)):\n", "        if False:\n", "tests/test_unused_inputs.py::test_unused_inputs_must_be_sorted_unique_kind_names"),
    ("prediction-version-unchecked", "learning/contracts.py", "        if (self.schema_version == \"learning-1.1\") != bool(self.unused_inputs):\n", "        if False:\n", "tests/test_unused_inputs.py::test_version_follows_unused_inputs"),
    ("training-time-leak", "learning/fitting.py", "if max(c.outcome_known_at for c in left) >= min(c.landmark.query.cutoff for c in right):", "if False:", "tests/test_learning.py::test_temporal_boundary_rejected"),
]


def main() -> None:
    results = []
    with tempfile.TemporaryDirectory(prefix="quiet-cycle-mutations-") as temporary:
        work = Path(temporary)
        for name in ("src", "tests", "examples", "scripts", "contracts"):
            shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        shutil.copy2(ROOT / "pyproject.toml", work / "pyproject.toml")
        env = {**os.environ, "PYTHONPATH": str(work / "src")}
        # A test that already fails unmutated would count every mutation against it as killed.
        baseline = subprocess.run([sys.executable, "-m", "pytest", "-q", *sorted({m[4] for m in MUTATIONS})], cwd=work,
            env=env, text=True, encoding="utf-8", capture_output=True, timeout=120)
        if baseline.returncode != 0:
            print(baseline.stdout, baseline.stderr, file=sys.stderr)
            raise SystemExit("Mutation targets fail without any mutation")
        for name, filename, old, new, test in MUTATIONS:
            path = work / "src/quietcycle" / filename
            original = path.read_text()
            assert original.count(old) == 1, name
            path.write_text(original.replace(old, new))
            # Force fresh bytecode even on filesystems with coarse mtimes.
            for cache in (work / "src").rglob("__pycache__"):
                shutil.rmtree(cache)
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", test], cwd=work,
                env=env, text=True, encoding="utf-8", capture_output=True, timeout=30)
            path.write_text(original)
            killed = result.returncode == 1 and "FAILED " in result.stdout and "ERROR collecting" not in result.stdout
            results.append({"mutation": name, "test": test, "killed": killed,
                            "exit_code": result.returncode, "summary": result.stdout.splitlines()[-1:]})
            if not killed:
                print(result.stdout, result.stderr, file=sys.stderr)
    report = {"selected_mutations": len(results), "killed": sum(x["killed"] for x in results),
              "scope": "targeted-regressions-not-exhaustive-mutation-coverage", "results": results}
    print(json.dumps(report, indent=2))
    if report["killed"] != len(results):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
