import json

import pytest

from pareto.config import SETTINGS
from pareto.contracts import EstimationResult
from pareto.spec import SUPPORTED_ESTIMATORS, Specification
from scripts.run_flip_spike import (
    _dataset_decision,
    _overall_decision,
    _spec_count_summary,
    analysis_config,
    build_spike_specs,
    dataset_report,
    render_markdown,
    required_columns,
    validate_required_columns,
)


def _result(
    *,
    status: str = "ok",
    coefficient: float | None = 1.0,
) -> EstimationResult:
    return EstimationResult(
        spec_id="s1",
        estimator="OLS",
        coefficient=coefficient,
        status=status,
        error="failed" if status != "ok" else None,
    )


def _axis_metrics(
    *,
    sign_comparable_pairs: int,
    sign_flip_count: int,
    sign_flip_rate: float | None,
    mean_abs_delta: float,
) -> dict[str, float | int | None]:
    return {
        "n_pairs": sign_comparable_pairs,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": mean_abs_delta,
        "sign_comparable_pairs": sign_comparable_pairs,
        "sign_flip_count": sign_flip_count,
        "sign_flip_rate": sign_flip_rate,
        "significance_comparable_pairs": 0,
        "significance_flip_count": 0,
        "significance_flip_rate": None,
    }


def _diagnosis(
    *,
    control_flips: int = 0,
    estimator_flips: int = 0,
    comparable_pairs: int = 1,
) -> dict[str, object]:
    control_rate = control_flips / comparable_pairs if comparable_pairs else None
    estimator_rate = estimator_flips / comparable_pairs if comparable_pairs else None
    return {
        "matched_pairs": {
            "control_set": _axis_metrics(
                sign_comparable_pairs=comparable_pairs,
                sign_flip_count=control_flips,
                sign_flip_rate=control_rate,
                mean_abs_delta=2.0,
            ),
            "estimator": _axis_metrics(
                sign_comparable_pairs=comparable_pairs,
                sign_flip_count=estimator_flips,
                sign_flip_rate=estimator_rate,
                mean_abs_delta=1.0,
            ),
        },
        "anova_partial_r2": {"control_set": 0.2, "estimator": 0.1},
        "warnings": [],
    }


def test_flip_spike_reads_required_config_fields():
    config = analysis_config("divorce")

    assert config["dataset"] == "divorce"
    assert config["outcome"] == "asmrs"
    assert config["treatment"] == "post"
    assert config["unit_col"] == "stfips"
    assert config["time_col"] == "year"
    assert config["controls"] == ("pcinc", "asmrh", "cases")
    assert config["weight_col"] == "weight"


@pytest.mark.parametrize("dataset", ["divorce", "castle"])
def test_flip_spike_builds_deterministic_8_specs(dataset):
    config = analysis_config(dataset)

    first = build_spike_specs(dataset, config)
    second = build_spike_specs(dataset, config)

    assert len(first) == 8
    assert [spec.model_dump() for spec in first] == [spec.model_dump() for spec in second]


@pytest.mark.parametrize("dataset", ["divorce", "castle"])
def test_flip_spike_spec_ids_are_unique_and_stable(dataset):
    config = analysis_config(dataset)
    specs = build_spike_specs(dataset, config)

    assert [spec.spec_id for spec in specs] == [f"{dataset}_{i:02d}" for i in range(8)]
    assert len({spec.spec_id for spec in specs}) == len(specs)


@pytest.mark.parametrize("dataset", ["divorce", "castle"])
def test_flip_spike_specs_are_specifications_with_supported_estimators(dataset):
    config = analysis_config(dataset)
    specs = build_spike_specs(dataset, config)

    assert all(isinstance(spec, Specification) for spec in specs)
    assert {spec.estimator for spec in specs} == set(SUPPORTED_ESTIMATORS)


@pytest.mark.parametrize("dataset", ["divorce", "castle"])
def test_flip_spike_specs_stay_under_hard_cap(dataset):
    config = analysis_config(dataset)
    specs = build_spike_specs(dataset, config)

    assert len(specs) <= SETTINGS.max_specifications


def test_flip_spike_report_is_json_serializable_without_running_estimators():
    config = analysis_config("castle")
    specs = build_spike_specs("castle", config)
    report = dataset_report("castle", config, specs, results=None, dry_run=True)

    json.dumps(report)


def test_flip_spike_dataset_report_requires_results_when_not_dry_run():
    config = analysis_config("castle")
    specs = build_spike_specs("castle", config)

    with pytest.raises(ValueError, match="results are required when dry_run is False"):
        dataset_report("castle", config, specs, results=None, dry_run=False)


def test_flip_spike_missing_required_columns_error_is_clear():
    config = analysis_config("divorce")
    columns = set(required_columns(config))
    columns.remove(config["outcome"])

    with pytest.raises(ValueError, match="divorce: missing required columns"):
        validate_required_columns("divorce", config, columns)


@pytest.mark.parametrize("dataset", ["divorce", "castle"])
def test_flip_spike_does_not_generate_sample_filter(dataset):
    config = analysis_config(dataset)
    specs = build_spike_specs(dataset, config)

    assert {spec.sample_filter for spec in specs} == {None}


@pytest.mark.parametrize(
    "results",
    [
        [_result(status="failed", coefficient=None)],
        [_result(status="ok", coefficient=None)],
    ],
)
def test_flip_spike_dataset_decision_is_inconclusive_without_ok_coefficients(results):
    decision = _dataset_decision(results, _diagnosis(control_flips=1))

    assert decision["status"] == "INCONCLUSIVE"


def test_flip_spike_dataset_decision_is_inconclusive_without_comparable_pairs():
    decision = _dataset_decision([_result()], _diagnosis(comparable_pairs=0))

    assert decision["status"] == "INCONCLUSIVE"


def test_flip_spike_dataset_decision_is_no_go_without_sign_flips():
    decision = _dataset_decision([_result()], _diagnosis(control_flips=0, estimator_flips=0))

    assert decision["status"] == "NO-GO"


def test_flip_spike_dataset_decision_reports_estimator_flips_when_not_dominant():
    decision = _dataset_decision([_result()], _diagnosis(control_flips=2, estimator_flips=1))

    assert decision["status"] == "GO"
    assert decision["descriptive_dominant_sign_axis"] == "control_set"
    assert "with estimator flips also observed" in decision["reason"]
    assert "not estimator flip" not in decision["reason"]


def test_flip_spike_dataset_decision_reports_no_estimator_flip_when_absent():
    decision = _dataset_decision([_result()], _diagnosis(control_flips=1, estimator_flips=0))

    assert decision["status"] == "GO"
    assert decision["descriptive_dominant_sign_axis"] == "control_set"
    assert "no estimator flip was observed" in decision["reason"]


def test_flip_spike_overall_decision_prioritizes_inconclusive():
    reports = [
        {"dataset": "divorce", "decision": {"status": "GO"}},
        {"dataset": "castle", "decision": {"status": "INCONCLUSIVE"}},
    ]

    decision = _overall_decision(reports)

    assert decision["status"] == "INCONCLUSIVE"


def test_flip_spike_overall_decision_go_when_all_conclusive_and_any_go():
    reports = [
        {"dataset": "divorce", "decision": {"status": "GO"}},
        {"dataset": "castle", "decision": {"status": "NO-GO"}},
    ]

    decision = _overall_decision(reports)

    assert decision["status"] == "GO"


def test_flip_spike_overall_decision_no_go_when_no_dataset_goes():
    reports = [
        {"dataset": "divorce", "decision": {"status": "NO-GO"}},
        {"dataset": "castle", "decision": {"status": "NO-GO"}},
    ]

    decision = _overall_decision(reports)

    assert decision["status"] == "NO-GO"


@pytest.mark.parametrize("dataset", ["divorce", "castle"])
def test_flip_spike_overall_no_go_reason_only_names_analyzed_dataset(dataset):
    decision = _overall_decision([{"dataset": dataset, "decision": {"status": "NO-GO"}}])

    assert decision["status"] == "NO-GO"
    assert dataset in decision["reason"]
    other_dataset = "castle" if dataset == "divorce" else "divorce"
    assert other_dataset not in decision["reason"]


def test_flip_spike_overall_no_go_reason_names_both_analyzed_datasets():
    decision = _overall_decision(
        [
            {"dataset": "divorce", "decision": {"status": "NO-GO"}},
            {"dataset": "castle", "decision": {"status": "NO-GO"}},
        ]
    )

    assert decision["status"] == "NO-GO"
    assert "divorce" in decision["reason"]
    assert "castle" in decision["reason"]


def test_flip_spike_spec_count_summary_is_derived_from_dataset_reports():
    summary = _spec_count_summary(
        [
            {"dataset": "divorce", "spec_count": 3},
            {"dataset": "castle", "spec_count": 5},
        ]
    )

    assert summary == {
        "specs_per_dataset": None,
        "spec_counts_by_dataset": {"divorce": 3, "castle": 5},
    }


def test_flip_spike_markdown_uses_report_spec_counts():
    markdown = render_markdown(
        {
            "spec_matrix": {
                "estimators": ["OLS", "TWFE"],
                "specs_per_dataset": None,
                "spec_counts_by_dataset": {"divorce": 3, "castle": 5},
            },
            "decision_rules": [],
            "limitations": [],
            "overall_decision": {"status": "INCONCLUSIVE", "reason": "test"},
            "datasets": [],
        }
    )

    assert "- Specs per dataset: divorce=3; castle=5" in markdown


def test_flip_spike_markdown_documents_go_and_descriptive_axis_semantics():
    markdown = render_markdown(
        {
            "spec_matrix": {
                "estimators": ["OLS", "TWFE"],
                "specs_per_dataset": 2,
                "spec_counts_by_dataset": {"test": 2},
            },
            "decision_rules": [
                "GO validates readable axis fragility in the current axis set; it does "
                "not by itself validate a canonical estimator-only or staggered-estimator story.",
                "Descriptive dominant sign axis ranks axes by sign_flip_count, then "
                "sign_flip_rate, then mean_abs_delta; this is not formal statistical dominance.",
            ],
            "limitations": [],
            "overall_decision": {"status": "INCONCLUSIVE", "reason": "test"},
            "datasets": [],
        }
    )

    assert "GO validates readable axis fragility" in markdown
    assert "canonical estimator-only or staggered-estimator story" in markdown
    assert "not formal statistical dominance" in markdown
