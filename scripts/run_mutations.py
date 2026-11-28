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
    ("declared-unit-rules-ignored", "learning/landmarks.py", "DefaultFeatureBuilder(_unit_registry(spec))", "DefaultFeatureBuilder()", "tests/test_unit_rules.py::test_rows_and_csv_and_both_units_give_the_same_features"),
    ("builtin-rule-not-skipped", "learning/landmarks.py", "for r in spec.unit_rules if not r.same_as_builtin()})", "for r in spec.unit_rules})", "tests/test_unit_rules.py::test_rule_identical_to_a_builtin_is_accepted_as_a_no_op"),
    ("two-canonical-units-allowed", "learning/contracts.py", "            if len(units) > 1:\n", "            if False:\n", "tests/test_unit_rules.py::test_invalid_unit_rules"),
    ("contract-version-unchecked", "learning/contracts.py", "    if (schema_version == \"learning-1.1\") != bool(spec.unit_rules):\n", "    if False:\n", "tests/test_unit_rules.py::test_contract_version_follows_unit_rules"),
    ("identity-rule-unchecked", "learning/contracts.py", "        if self.unit == self.canonical_unit and", "        if False and", "tests/test_unit_rules.py::test_invalid_unit_rules"),
    ("unit-offset-dropped", "learning/landmarks.py", "r.scale.as_fraction(), r.offset.as_fraction())\n", "r.scale.as_fraction(), Fraction(0))\n", "tests/test_unit_rules.py::test_scale_and_offset_convert_exactly"),
    ("landmark-validation-escapes", "learning/landmarks.py", "    except ValidationError as exc:\n        raise _input_error(exc, \"invalid_landmark\")", "    except KeyError as exc:\n        raise _input_error(exc, \"invalid_landmark\")", "tests/test_unit_rules.py::test_feature_precision_overflow_is_a_recorded_rejection"),
    ("validation-error-chained", "learning/landmarks.py", "_input_error(exc, \"invalid_landmark\") from None", "_input_error(exc, \"invalid_landmark\") from exc", "tests/test_unit_rules.py::test_feature_precision_overflow_is_a_recorded_rejection"),
    ("long-feature-names-accepted", "learning/contracts.py", "        if any(len(name) > 96 for name in expected_feature_names(self)):\n", "        if False:\n", "tests/test_unit_rules.py::test_derived_feature_names_fit_the_identifier_limit"),
    ("declared-metric-name-unbounded", "learning/contracts.py", "        if any(len(f\"untrained-feature.measurement.{m}.recent\") > 96 for m in self.metrics if m in declared):\n", "        if False:\n", "tests/test_unit_rules.py::test_a_declared_metric_name_fits_its_forecast_issues"),
    ("unused-inputs-dropped", "learning/prediction.py", "        if not names:\n            return result\n", "        return result\n", "tests/test_unused_inputs.py::test_unused_inputs_are_named_without_values"),
    ("unused-symptoms-ignored", "learning/landmarks.py", "        elif isinstance(event, Symptom) and event.symptom not in spec.symptoms:\n", "        elif False:\n", "tests/test_unused_inputs.py::test_unused_inputs_are_named_without_values"),
    ("snapshot-error-escapes", "learning/prediction.py", "        except InputError:\n            # Without a valid snapshot", "        except KeyError:\n            # Without a valid snapshot", "tests/test_unused_inputs.py::test_an_invalid_snapshot_abstains_without_a_list"),
    ("unused-uniqueness-unpublished", "learning/contracts.py", "json_schema_extra={\"uniqueItems\": True}", "json_schema_extra={}", "tests/test_unused_inputs.py::test_published_schema_rejects_duplicate_unused_inputs"),
    ("unused-order-unchecked", "learning/contracts.py", "        if list(self.unused_inputs) != sorted(set(self.unused_inputs)):\n", "        if False:\n", "tests/test_unused_inputs.py::test_unused_inputs_must_be_sorted_unique_kind_names"),
    ("prediction-version-unchecked", "learning/contracts.py", "        if (self.schema_version == \"learning-1.1\") != bool(self.unused_inputs):\n", "        if False:\n", "tests/test_unused_inputs.py::test_version_follows_unused_inputs"),
    ("labels-cutoff-compared-as-text", "learning/landmarks.py", "    labels_cutoff = _labels_cutoff(labels_cutoff)\n    if labels_cutoff <= query.cutoff:", "    raw = labels_cutoff\n    labels_cutoff = _labels_cutoff(labels_cutoff)\n    if labels_cutoff <= query.cutoff or raw <= query.cutoff:", "tests/test_learning.py::test_a_later_instant_written_with_a_negative_offset_is_later"),
    ("labels-cutoff-checked-per-case-only", "learning/landmarks.py", "    # A malformed cutoff is a configuration error for the whole call, not a per-case rejection.\n    labels_cutoff = _labels_cutoff(labels_cutoff)\n", "", "tests/test_learning.py::test_malformed_labels_cutoff_is_an_input_error"),
    ("report-past-onset-accepted", "learning/landmarks.py", "        if any(origin.date < o.date <= report.through for o in onsets):\n", "        if False:\n", "tests/test_outcomes.py::test_each_outcome_row"),
    ("skipped-link-accepted", "learning/landmarks.py", "o.previous_onset_id == anchor.id and o.continuity == \"confirmed\" for o in later[1:]", "False for o in later[1:]", "tests/test_outcomes.py::test_each_outcome_row"),
    ("excluded-onset-labelled", "learning/landmarks.py", "        if end.exclude_from_history:\n", "        if False:\n", "tests/test_outcomes.py::test_each_outcome_row"),
    ("unlinked-onset-not-interval", "learning/landmarks.py", "            # A real onset, but an unrecorded one may lie between the last report and it.\n            return \"outcome-interval-censored\", None, None, (shortest, day_difference(end.date, anchor.date))\n", "            # A real onset, but an unrecorded one may lie between the last report and it.\n            return \"outcome-not-observed\", None, None, None\n", "tests/test_outcomes.py::test_each_outcome_row"),
    ("censoring-boundary-shifted", "learning/landmarks.py", "    if covered is not None and covered >= query.as_of:\n", "    if covered is not None and covered > query.as_of:\n", "tests/test_outcomes.py::test_each_outcome_row"),
    ("report-boundary-exclusive", "learning/landmarks.py", "origin.date < o.date <= report.through for o in onsets", "origin.date < o.date < report.through for o in onsets", "tests/test_outcomes.py::test_each_outcome_row"),
    ("label-reports-unchecked", "learning/landmarks.py", "    try:\n        _check_reports(onsets, relevant)\n    except InputError as exc:\n        return exc.code, None, None, None\n", "", "tests/test_outcomes.py::test_each_outcome_row"),
    ("uncertain-next-not-bounded", "learning/landmarks.py", "        if bound is not None:\n", "        if False:\n", "tests/test_outcomes.py::test_each_outcome_row"),
    ("skip-ignores-continuity", "learning/landmarks.py", "o.previous_onset_id == anchor.id and o.continuity == \"confirmed\" for o in later[1:]", "o.previous_onset_id == anchor.id for o in later[1:]", "tests/test_outcomes.py::test_each_outcome_row"),
    ("known-checked-after-exclusion", "learning/landmarks.py", "        if _held_at(dataset, anchor, query.cutoff) and _held_at(dataset, end, query.cutoff):\n            return \"outcome-already-known-at-landmark\", None, None, None\n        if end.exclude_from_history:\n            return \"label-excluded-from-history\", None, None, None\n", "        if end.exclude_from_history:\n            return \"label-excluded-from-history\", None, None, None\n        if _held_at(dataset, anchor, query.cutoff) and _held_at(dataset, end, query.cutoff):\n            return \"outcome-already-known-at-landmark\", None, None, None\n", "tests/test_outcomes.py::test_rows_for_an_outcome_known_at_the_landmark"),
    ("case-metadata-per-case", "learning/landmarks.py", "    _check_case_metadata(data_kind, cohort)  # and so are invalid case labels\n", "", "tests/test_outcomes.py::test_invalid_case_metadata_fails_the_whole_call"),
    ("csv-unhashable-kind-escapes", "adapters/csv.py", "                except TypeError:  # an unhashable", "                except KeyError:  # an unhashable", "tests/test_adapters.py::test_csv_kind_that_is_not_text_is_a_row_error"),
    ("legacy-deleted-kind-required", "adapters/legacy.py", 'record["target_kind"] = old.get("kind")', 'record["target_kind"] = old["kind"]', "tests/test_adapters.py::test_legacy_deleted_event_without_kind_is_a_row_error"),
    ("legacy-import-time-escapes", "adapters/legacy.py", "    except ValidationError:\n        raise InputError(\"invalid_import_time\")", "    except KeyError:\n        raise InputError(\"invalid_import_time\")", "tests/test_adapters.py::test_legacy_invalid_imported_at_is_an_input_error"),
    ("legacy-empty-import-time-unchecked", "adapters/legacy.py", "    if not records:\n        _provenance(imported_at)", "    if False:\n        _provenance(imported_at)", "tests/test_adapters.py::test_legacy_invalid_imported_at_is_an_input_error"),
    ("long-issue-names-predicted", "learning/prediction.py", "    if any(len(name) > 96 for name in names):\n", "    if False:\n", "tests/test_feature_name_limits.py::test_an_untrained_symptom_whose_issue_name_is_too_long_abstains"),
    ("long-term-names-fitted", "learning/fitting.py", "            if any(len(t) > 96 for t in terms):\n", "            if False:\n", "tests/test_feature_name_limits.py::test_fit_rejects_a_missing_indicator_name_that_is_too_long"),
    ("later-cycle-reports-checked", "learning/landmarks.py", "        by_id[r.anchor_id].date < horizon if r.anchor_id in by_id else r.through <= horizon)]", "        True)]", "tests/test_outcomes.py::test_each_outcome_row"),
    ("unknown-origin-report-ignored", "learning/landmarks.py", "if r.anchor_id in by_id else r.through <= horizon)]", "if r.anchor_id in by_id else False)]", "tests/test_outcomes.py::test_each_outcome_row"),
    ("uncertain-origin-accepted", "learning/landmarks.py", "        if origin is None or origin.certainty != \"confirmed\" or report.through < origin.date:\n", "        if origin is None or report.through < origin.date:\n", "tests/test_outcomes.py::test_landmark_report_rules"),
    ("report-before-origin-accepted", "learning/landmarks.py", "        if origin is None or origin.certainty != \"confirmed\" or report.through < origin.date:\n", "        if origin is None or origin.certainty != \"confirmed\":\n", "tests/test_outcomes.py::test_landmark_report_rules"),
    ("same-day-report-invalid", "learning/landmarks.py", "or report.through < origin.date:\n", "or report.through <= origin.date:\n", "tests/test_outcomes.py::test_a_report_through_its_own_onset_day_is_valid"),
    ("metadata-before-cutoff", "learning/landmarks.py", "    _labels_cutoff(labels_cutoff)  # a bad cutoff is reported before bad case metadata\n", "", "tests/test_outcomes.py::test_an_invalid_labels_cutoff_is_reported_before_case_metadata"),
    ("window-floor-dropped", "learning/landmarks.py", "    relevant = [r for r in reports if r.through >= anchor.date and (", "    relevant = [r for r in reports if (", "tests/test_outcomes.py::test_each_outcome_row"),
    ("anchor-date-duplicate-ignored", "learning/landmarks.py", "            or [o.date for o in onsets].count(anchor_date) != 1):\n", "            or False):\n", "tests/test_outcomes.py::test_each_outcome_row"),
    ("anchor-exclusion-ignored", "learning/landmarks.py", " or anchor.exclude_from_history\n", "\n", "tests/test_outcomes.py::test_each_outcome_row"),
    ("label-validation-escapes", "learning/landmarks.py", "    except ValidationError as exc:\n        raise _input_error(exc, \"invalid_labelled_case\")", "    except KeyError as exc:\n        raise _input_error(exc, \"invalid_labelled_case\")", "tests/test_outcomes.py::test_label_validation_failures_are_input_errors"),
    ("label-error-chained", "learning/landmarks.py", "_input_error(exc, \"invalid_labelled_case\") from None", "_input_error(exc, \"invalid_labelled_case\") from exc", "tests/test_outcomes.py::test_label_validation_failures_are_input_errors"),
    ("inventory-reads-measurement-units", "learning/eligibility.py", "Snapshot(tuple(e for e in events if not isinstance(e, Measurement)), \"\")", "Snapshot(events, \"\")", "tests/test_inventory.py::test_inventory_counts_every_item_by_hand"),
    ("inventory-ignores-offset", "learning/eligibility.py", "    as_of = _labels_day(cutoff, utc_offset_minutes)\n", "    as_of = _labels_day(cutoff, 0)\n", "tests/test_inventory.py::test_inventory_offset_sets_the_local_day"),
    ("inventory-hides-exclusion", "learning/eligibility.py", "onsets[\"excluded\" if o.exclude_from_history else o.certainty] += 1", "onsets[o.certainty] += 1", "tests/test_inventory.py::test_inventory_counts_every_item_by_hand"),
    ("inventory-drops-bad-subjects", "learning/eligibility.py", "            _increment(errors, exc.code)  # only the first error found\n", "            pass\n", "tests/test_inventory.py::test_inventory_counts_every_item_by_hand"),
    ("inventory-counts-unknown-subjects", "learning/eligibility.py", "        if not events:  # nothing about this subject was known at the cutoff\n", "        if False:\n", "tests/test_inventory.py::test_a_subject_known_only_after_the_cutoff_is_not_counted"),
    ("inventory-run-not-reset", "learning/eligibility.py", "                run = 0\n", "                pass\n", "tests/test_inventory.py::test_runs_break_at_unconfirmed_and_excluded_intervals"),
    ("inventory-longest-is-last", "learning/eligibility.py", "                longest = max(longest, run)\n", "                longest = run\n", "tests/test_inventory.py::test_runs_break_at_unconfirmed_and_excluded_intervals"),
    ("inventory-excluded-interval-confirmed", "learning/eligibility.py", "interval.length > 0 and interval.known and not interval.excluded:", "interval.length > 0 and interval.known:", "tests/test_inventory.py::test_runs_break_at_unconfirmed_and_excluded_intervals"),
    ("inventory-any-import-time", "learning/eligibility.py", "\"import_time\" if event.available_at > event.recorded_at else", "\"import_time\" if event.provenance.imported_at else", "tests/test_inventory.py::test_an_import_time_before_recording_does_not_set_availability"),
    ("inventory-stream-ignores-method", "learning/eligibility.py", "event.provenance.source, event.method or \"\", event.body_site", "event.provenance.source, \"\", event.body_site", "tests/test_inventory.py::test_streams_are_split_by_method_and_body_site"),
    ("inventory-domain-error-as-offset", "learning/eligibility.py", "\"invalid_offset\" if str(exc) == \"invalid_offset\" else \"label_date_outside_domain\"", "\"invalid_offset\"", "tests/test_inventory.py::test_inventory_rejects_invalid_arguments"),
    ("inventory-same-date-confirmed", "learning/eligibility.py", "            if interval.length > 0 and interval.known and not interval.excluded:\n", "            if interval.known and not interval.excluded:\n", "tests/test_inventory.py::test_two_onsets_on_one_date_break_the_chain"),
    ("inventory-exclusion-after-continuity", "learning/eligibility.py", "\"same_date\" if interval.length == 0 else \"excluded\" if interval.excluded else \"unconfirmed\"", "\"same_date\" if interval.length == 0 else \"unconfirmed\" if not interval.known else \"excluded\"", "tests/test_inventory.py::test_runs_break_at_unconfirmed_and_excluded_intervals"),
    ("inventory-rejections-uncounted", "learning/eligibility.py", "                    _increment(stream[\"rejected\"], exc.code)\n", "                    pass\n", "tests/test_inventory.py::test_streams_count_samples_the_learner_would_reject"),
    ("inventory-registry-ignores-spec", "learning/eligibility.py", "    registry = _unit_registry(spec)\n", "    registry = _unit_registry(LandmarkSpec())\n", "tests/test_inventory.py::test_streams_count_samples_the_learner_would_reject"),
    ("inventory-unused-from-all-events", "learning/eligibility.py", "        unused.update(unused_inputs(spec, events))\n", "        unused.update(unused_inputs(spec, by_subject[subject]))\n", "tests/test_inventory.py::test_unused_inputs_come_from_the_snapshot"),
    ("inventory-onset-exclusion-after-certainty", "learning/eligibility.py", "onsets[\"excluded\" if o.exclude_from_history else o.certainty] += 1", "onsets[o.certainty if o.certainty == \"uncertain\" else \"excluded\" if o.exclude_from_history else o.certainty] += 1", "tests/test_inventory.py::test_onset_precedence_and_subjects_without_onsets"),
    ("inventory-exclusion-before-same-date", "learning/eligibility.py", "\"same_date\" if interval.length == 0 else \"excluded\" if interval.excluded else \"unconfirmed\"", "\"excluded\" if interval.excluded else \"same_date\" if interval.length == 0 else \"unconfirmed\"", "tests/test_inventory.py::test_a_same_date_pair_is_same_date_even_when_excluded"),
    ("inventory-missing-samples-unchecked", "learning/eligibility.py", "                try:\n                    registry.normalize(event)\n", "                try:\n                    if event.state == \"measured\":\n                        registry.normalize(event)\n", "tests/test_inventory.py::test_streams_count_samples_the_learner_would_reject"),
    ("attempts-use-hindsight", "learning/eligibility.py", "anchors = _anchors(events, date, min(query_cutoff, cutoff))", "anchors = _anchors(events, date, cutoff)", "tests/test_attempts.py::test_attempts_follow_what_was_known_at_each_query"),
    ("late-attempts-use-later-data", "learning/eligibility.py", "anchors = _anchors(events, date, min(query_cutoff, cutoff))", "anchors = _anchors(events, date, query_cutoff)", "tests/test_attempts.py::test_each_attempt_is_counted_once_with_its_own_code"),
    ("superseded-version-attempted", "learning/eligibility.py", "            if best is None or event.revision > best[0].revision:\n", "            if True:\n", "tests/test_attempts.py::test_attempts_follow_what_was_known_at_each_query"),
    ("late-attempts-dropped", "learning/eligibility.py", "                    late += 1\n", "                    pass\n", "tests/test_attempts.py::test_each_elapsed_day_counts_attempts_codes_and_outcomes"),
    ("ambiguous-anchor-classified", "learning/eligibility.py", "                if len(anchors) > 1:\n", "                if False:\n", "tests/test_attempts.py::test_each_attempt_is_counted_once_with_its_own_code"),
    ("outcome-error-escapes", "learning/eligibility.py", "                except InputError as exc:\n                    outcome = exc.code\n", "                except KeyError as exc:\n                    outcome = str(exc)\n", "tests/test_attempts.py::test_each_attempt_is_counted_once_with_its_own_code"),
    ("query-time-ignored", "learning/eligibility.py", "query_cutoff = instant(f\"{as_of}T{query_time}:00{zone}\")", "query_cutoff = instant(f\"{as_of}T12:00:00{zone}\")", "tests/test_attempts.py::test_query_time_and_offset_set_each_query_cutoff"),
    ("query-offset-ignored", "learning/eligibility.py", "query_cutoff = instant(f\"{as_of}T{query_time}:00{zone}\")", "query_cutoff = instant(f\"{as_of}T{query_time}:00Z\")", "tests/test_attempts.py::test_query_time_and_offset_set_each_query_cutoff"),
    ("elapsed-order-unchecked", "learning/eligibility.py", "            or list(elapsed_days) != sorted(set(elapsed_days))):\n", "            or False):\n", "tests/test_attempts.py::test_invalid_arguments_fail_before_any_attempt"),
    ("day-spec-not-revalidated", "learning/eligibility.py", "            day_spec = LandmarkSpec.model_validate({**spec.model_dump(mode=\"python\"), \"elapsed_day\": elapsed})\n", "            day_spec = spec.model_copy(update={\"elapsed_day\": elapsed})\n", "tests/test_attempts.py::test_each_elapsed_day_counts_attempts_codes_and_outcomes"),
    ("recorded-at-query-invisible", "learning/eligibility.py", "        if event.available_at <= known:\n", "        if event.available_at < known:\n", "tests/test_attempts.py::test_boundaries_of_what_a_query_can_see"),
    ("query-at-labels-cutoff-not-late", "learning/eligibility.py", "                if query_cutoff >= cutoff:\n", "                if query_cutoff > cutoff:\n", "tests/test_attempts.py::test_boundaries_of_what_a_query_can_see"),
    ("offset-minutes-dropped", "learning/eligibility.py", "{abs(utc_offset_minutes) % 60:02d}", "00", "tests/test_attempts.py::test_query_time_and_offset_set_each_query_cutoff"),
    ("conflicting-version-first-wins", "learning/eligibility.py", "            elif event.revision == best[0].revision:\n                best.append(event)\n", "", "tests/test_attempts.py::test_conflicting_versions_make_an_attempt_on_each_date_they_name"),
    ("conflicting-version-last-only", "learning/eligibility.py", "any(isinstance(v, Onset) and v.date == date for v in versions)", "isinstance(versions[-1], Onset) and versions[-1].date == date", "tests/test_attempts.py::test_conflicting_versions_make_an_attempt_on_each_date_they_name"),
    ("query-day-past-domain-counted", "learning/eligibility.py", "                    continue  # no query can be made on a day past the date domain\n", "                    as_of, query_cutoff = \"\", cutoff\n", "tests/test_attempts.py::test_boundaries_of_what_a_query_can_see"),
    ("labels-day-unchecked", "learning/eligibility.py", "    _labels_day(cutoff, utc_offset_minutes)\n", "", "tests/test_attempts.py::test_invalid_arguments_fail_before_any_attempt"),
    ("overflowing-query-dropped", "learning/eligibility.py", "                    query_cutoff = cutoff  # its UTC instant is past the domain, so after the labels cutoff\n", "                    continue\n", "tests/test_attempts.py::test_boundaries_of_what_a_query_can_see"),
    ("negative-offset-sign-lost", "learning/eligibility.py", "{abs(utc_offset_minutes) // 60:02d}", "{utc_offset_minutes // 60:02d}", "tests/test_attempts.py::test_boundaries_of_what_a_query_can_see"),
    ("training-time-leak", "learning/fitting.py", "if max(c.outcome_known_at for c in left) >= min(c.landmark.query.cutoff for c in right):", "if False:", "tests/test_learning.py::test_temporal_boundary_rejected"),
    ("cli-format-option-ignored", "cli.py", "getattr(args, name) is not None and args.format != fmt", "False", "tests/test_interfaces.py::test_cli_rejects_an_option_its_format_would_ignore"),
    ("cli-fhir-source-default-lost", "cli.py", 'source_id="fhir-r4" if args.source_id is None else args.source_id', "source_id=args.source_id", "tests/test_interfaces.py::test_cli_fhir_import_matches_the_adapter"),
    ("ridge-name-limit-unchecked", "learning/fitting.py", "            if len(name) > 96:", "            if False:", "tests/test_feature_name_limits.py::test_fit_rejects_a_ridge_penalty_too_long_for_a_candidate_name"),
    ("label-day-in-utc", "learning/contracts.py", 'self.outcome_date + "T00:00:00+14:00"', 'self.outcome_date + "T00:00:00+00:00"', "tests/test_outcomes.py::test_a_label_keeps_the_offset_of_its_onset"),
    ("label-day-start-excluded", "learning/contracts.py", "datetime.fromisoformat(self.outcome_known_at) < ", "datetime.fromisoformat(self.outcome_known_at) <= ", "tests/test_outcomes.py::test_a_case_known_before_its_outcome_date_began_anywhere_is_invalid"),
    ("labels-day-at-query-offset", "learning/landmarks.py", "offset, label_day = 840, local_day(labels_cutoff, 840)", "offset, label_day = query.utc_offset_minutes, local_day(labels_cutoff, query.utc_offset_minutes)", "tests/test_outcomes.py::test_labels_read_each_event_on_its_own_date"),
    ("labels-day-short-of-the-largest-offset", "learning/landmarks.py", "offset, label_day = 840, local_day(labels_cutoff, 840)", "offset, label_day = 839, local_day(labels_cutoff, 839)", "tests/test_outcomes.py::test_labels_read_an_event_recorded_at_the_largest_offset"),
    ("labels-domain-fallback-to-query", "learning/landmarks.py", "offset, label_day = 0, local_day(labels_cutoff, 0)", "offset, label_day = query.utc_offset_minutes, local_day(labels_cutoff, query.utc_offset_minutes)", "tests/test_outcomes.py::test_labels_read_near_the_end_of_the_date_domain"),
    ("labels-domain-error-escapes", "learning/landmarks.py", "    except ValueError:\n        offset, label_day = 0,", "    except KeyError:\n        offset, label_day = 0,", "tests/test_learning.py::test_labels_day_past_the_date_domain_is_the_utc_day"),
    ("known-outcome-newest-version", "learning/landmarks.py", "_held_at(dataset, anchor, query.cutoff) and _held_at(dataset, end, query.cutoff)", "max(anchor.available_at, end.available_at) <= query.cutoff", "tests/test_attempts.py::test_a_later_revision_does_not_hide_an_outcome_known_at_a_query"),
    ("known-outcome-any-date", "learning/landmarks.py", "isinstance(held, Onset) and held.date == onset.date", "isinstance(held, Onset)", "tests/test_outcomes.py::test_an_outcome_known_at_the_cutoff_is_the_version_its_snapshot_held"),
    ("known-outcome-any-subject", "learning/landmarks.py", "if e.subject_id == onset.subject_id and e.id == onset.id and e.available_at <= cutoff]", "if e.id == onset.id and e.available_at <= cutoff]", "tests/test_outcomes.py::test_an_outcome_known_at_the_cutoff_is_the_version_its_snapshot_held"),
    ("known-outcome-first-version", "learning/landmarks.py", "held = max(versions, key=lambda e: e.revision, default=None)", "held = min(versions, key=lambda e: e.revision, default=None)", "tests/test_outcomes.py::test_an_outcome_known_at_the_cutoff_is_the_version_its_snapshot_held"),
    ("known-outcome-anchor-newest", "learning/landmarks.py", "_held_at(dataset, anchor, query.cutoff) and _held_at(dataset, end, query.cutoff)", "anchor.available_at <= query.cutoff and _held_at(dataset, end, query.cutoff)", "tests/test_outcomes.py::test_an_outcome_known_at_the_cutoff_is_the_version_its_snapshot_held"),
    ("censored-bound-ignores-report", "learning/landmarks.py", "shortest = day_difference(covered, anchor.date) + 1 if covered is not None else 1", "shortest = day_difference(query.as_of, anchor.date)", "tests/test_outcomes.py::test_bounds_of_each_usable_outcome"),
    ("interval-bound-from-doubtful-onset", "learning/landmarks.py", "(shortest, day_difference(bound.date, anchor.date))", "(shortest, day_difference(later[0].date, anchor.date))", "tests/test_outcomes.py::test_bounds_of_each_usable_outcome"),
    ("report-before-anchor-bounds", "learning/landmarks.py", " and r.through >= anchor.date), default=None)", "), default=None)", "tests/test_outcomes.py::test_bounds_of_each_usable_outcome"),
    ("today-hazard-skipped", "learning/survival.py", "        masses.append(surviving * hazard)\n        surviving *= 1 - hazard\n", "        surviving *= 1 - hazard\n        masses.append(surviving * hazard)\n", "tests/test_survival.py::test_today_has_its_own_hazard_and_the_tail_keeps_the_rest"),
    ("condition-drops-tail", "learning/survival.py", "    remaining = sum(kept, Fraction()) + tail\n", "    remaining = sum(kept, Fraction())\n", "tests/test_survival.py::test_conditioning_on_a_later_day_equals_starting_from_it"),
    ("conservation-unchecked", "learning/survival.py", "        if total != 1:\n            raise ValueError(\"mass_not_conserved\")\n", "        if False:\n            raise ValueError(\"mass_not_conserved\")\n", "tests/test_survival.py::test_mass_must_be_conserved_and_consecutive_from_the_elapsed_day"),
    ("condition-into-the-past", "learning/survival.py", "elapsed_day < forecast.elapsed_day:", "elapsed_day < 0:", "tests/test_survival.py::test_no_conditioning_past_exhausted_support_or_into_the_past"),
    ("jeffreys-prior-dropped", "learning/survival.py", "(d + Fraction(1, 2)) / (n + 1)", "Fraction(d, n)", "tests/test_survival.py::test_pooled_hazards_count_censored_days_at_risk_but_not_as_onsets"),
    ("shrinkage-strength-ignored", "learning/survival.py", "(strength * mu + d) / (strength + n)", "(mu + d) / (1 + n)", "tests/test_survival.py::test_shrinkage_keeps_the_population_where_a_person_has_no_days_at_risk"),
    ("hazard-type-unchecked", "learning/survival.py", "any(type(h) is not Fraction or not 0 <= h <= 1 for h in hazards)", "any(not 0 <= h <= 1 for h in hazards)", "tests/test_survival.py::test_hazards_must_be_fractions_in_the_unit_interval_given_as_a_tuple"),
    ("day-range-unchecked", "learning/survival.py", "not (1 <= first <= MAX_DAY and first - 1 <= last <= MAX_DAY):", "False:", "tests/test_survival.py::test_days_outside_the_supported_range_are_input_errors"),
    ("empty-window-allowed", "learning/survival.py", "    if first_day > last_day:\n        raise InputError(\"empty_window\")\n", "", "tests/test_survival.py::test_days_outside_the_supported_range_are_input_errors"),
    ("cycle-not-revalidated", "learning/survival.py", "    return tuple(_valid(ObservedCycle, c, \"invalid_cycle\") for c in cycles)\n", "    return cycles\n", "tests/test_survival.py::test_unvalidated_cycles_and_forecasts_are_input_errors"),
    ("precision-checked-late", "learning/survival.py", "        if _bits(surviving) > PRECISION_BITS:\n", "        if False:\n", "tests/test_survival.py::test_an_unstorable_forecast_fails_before_the_products_grow"),
    ("running-sum-unbounded", "learning/survival.py", "            if _bits(total) > 2 * PRECISION_BITS + 8:\n", "            if False:\n", "tests/test_survival.py::test_checking_conservation_stops_when_the_running_sum_outgrows_the_budget"),
    ("censored-day-counted-as-onset", "learning/survival.py", "        if c.onset and first_day <= c.last_day <= last_day:\n", "        if first_day <= c.last_day <= last_day:\n", "tests/test_survival.py::test_pooled_hazards_count_censored_days_at_risk_but_not_as_onsets"),
    ("entry-day-ignored", "learning/survival.py", "lower, upper = max(c.entry_day, first_day), min(c.last_day, last_day)", "lower, upper = first_day, min(c.last_day, last_day)", "tests/test_survival.py::test_shrinkage_keeps_the_population_where_a_person_has_no_days_at_risk"),
    ("risk-window-not-clipped", "learning/survival.py", "lower, upper = max(c.entry_day, first_day), min(c.last_day, last_day)", "lower, upper = c.entry_day, min(c.last_day, last_day)", "tests/test_survival.py::test_life_tables_count_only_the_days_inside_their_window"),
    ("shrunk-window-from-day-one", "learning/survival.py", "_risk_counts(personal, first_day, first_day + len(population) - 1)", "_risk_counts(personal, 1, len(population))", "tests/test_survival.py::test_life_tables_count_only_the_days_inside_their_window"),
    ("onset-on-first-window-day-dropped", "learning/survival.py", "if c.onset and first_day <= c.last_day <= last_day:", "if c.onset and first_day < c.last_day <= last_day:", "tests/test_survival.py::test_life_tables_count_only_the_days_inside_their_window"),
    ("onset-after-window-counted", "learning/survival.py", "        if c.onset and first_day <= c.last_day <= last_day:\n            onsets[c.last_day - first_day] += 1\n", "        if c.onset and first_day <= c.last_day:\n            onsets[min(c.last_day, last_day) - first_day] += 1\n", "tests/test_survival.py::test_life_tables_count_only_the_days_inside_their_window"),
    ("censored-residual-dropped", "learning/scoring.py", "    return inside + (residual if upper is None else 0)\n", "    return inside\n", "tests/test_scoring.py::test_the_probability_of_what_the_labels_allow"),
    ("residual-split-allowed", "learning/scoring.py", "    if residual and (lower - 1 if upper is None else upper) > last:\n", "    if False:\n", "tests/test_scoring.py::test_bounds_that_split_the_residual_are_refused"),
    ("outcome-lower-bound-ignored", "learning/scoring.py", "if lower <= p.cycle_length and (upper", "if (upper", "tests/test_scoring.py::test_the_probability_of_what_the_labels_allow"),
    ("log-score-through-float", "learning/scoring.py", "    return math.log(probability.denominator) - math.log(probability.numerator)\n", "    return -math.log(float(probability))\n", "tests/test_scoring.py::test_a_tiny_probability_still_has_a_finite_score"),
    ("zero-residual-refused", "learning/scoring.py", "    if residual and (lower - 1", "    if (lower - 1", "tests/test_scoring.py::test_a_forecast_without_residual_mass_scores_any_later_outcome"),
    ("outcome-before-forecast-scored", "learning/scoring.py", "    if upper is not None and upper < forecast.elapsed_day:\n", "    if False:\n", "tests/test_scoring.py::test_an_outcome_before_the_forecast_day_contradicts_it"),
    ("near-one-by-subtraction", "learning/scoring.py", "        return -math.log1p(-float(1 - probability))", "        return math.log(probability.denominator) - math.log(probability.numerator)", "tests/test_scoring.py::test_a_probability_near_one_keeps_its_small_score"),
    ("subnormal-through-integer-logs", "learning/scoring.py", "    if probability >= Fraction(2) ** -1022:\n", "    if probability >= Fraction(2) ** -1074:\n", "tests/test_scoring.py::test_a_subnormal_probability_keeps_its_score"),
    ("life-table-personal-ignored", "learning/life_table.py", "    personal = _cycles(history_lengths(dataset, query, model.spec), len(model.hazards))\n", "    personal = ()\n", "tests/test_life_table.py::test_a_forecast_shrinks_the_pooled_hazards_toward_the_person"),
    ("life-table-starts-a-day-late", "learning/life_table.py", "    return from_hazards(elapsed, shrunk[elapsed - 1:])\n", "    return from_hazards(elapsed, shrunk[elapsed:])\n", "tests/test_life_table.py::test_a_forecast_shrinks_the_pooled_hazards_toward_the_person"),
    ("life-table-horizon-unchecked", "learning/life_table.py", "    if horizon < spec.elapsed_day:\n", "    if False:\n", "tests/test_life_table.py::test_the_horizon_reaches_the_landmark_day"),
    ("long-cycle-counted-as-onset", "learning/life_table.py", "onset=n <= horizon)", "onset=True)", "tests/test_life_table.py::test_a_cycle_longer_than_the_horizon_is_at_risk_through_it_without_an_onset"),
    ("digest-without-spec", "learning/life_table.py", "{\"spec\": spec, \"strength\"", "{\"strength\"", "tests/test_life_table.py::test_the_training_digest_covers_everything_that_shapes_the_model"),
    ("life-table-model-unchecked", "learning/life_table.py", "        if any(not 0 < h.as_fraction() < 1 for h in self.hazards):\n", "        if False:\n", "tests/test_life_table.py::test_an_inconsistent_model_is_refused"),
    ("digest-without-horizon", "learning/life_table.py", ", \"horizon\": horizon})", "})", "tests/test_life_table.py::test_the_training_digest_covers_everything_that_shapes_the_model"),
    ("zero-hazard-allowed", "learning/life_table.py", "if any(not 0 < h.as_fraction() < 1 for h in self.hazards):", "if any(not 0 <= h.as_fraction() < 1 for h in self.hazards):", "tests/test_life_table.py::test_an_inconsistent_model_is_refused"),
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
            original = path.read_text(encoding="utf-8")
            assert original.count(old) == 1, name
            path.write_text(original.replace(old, new), encoding="utf-8")
            # Force fresh bytecode even on filesystems with coarse mtimes.
            for cache in (work / "src").rglob("__pycache__"):
                shutil.rmtree(cache)
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", test], cwd=work,
                env=env, text=True, encoding="utf-8", capture_output=True, timeout=30)
            path.write_text(original, encoding="utf-8")
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
