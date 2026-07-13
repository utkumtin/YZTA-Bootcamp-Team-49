"""Run the S2-15 divorce/castle flip spike.

This is intentionally a thin spike harness around the committed analysis core:
Specification, run_specs, summarize, and diagnose_axes. It does not reimplement
sign, significance, matched-pair, or ANOVA attribution logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct script execution needs the repo root on sys.path before Pareto imports.
from pareto.analysis.runner import run_specs  # noqa: E402
from pareto.analysis.variance import diagnose_axes, summarize  # noqa: E402
from pareto.config import SETTINGS  # noqa: E402
from pareto.contracts import EstimationResult  # noqa: E402
from pareto.spec import SUPPORTED_ESTIMATORS, Specification  # noqa: E402

DATASETS = ("divorce", "castle")
TREATMENT_DUMMY_LOGICAL_NAMES = ("post", "treated")


def _require_key(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context}: missing required key {key!r}")
    return mapping[key]


def _one_source(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    sources = _require_key(config, "sources", dataset)
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"{dataset}: sources must contain one configured source")
    if len(sources) != 1:
        raise ValueError(f"{dataset}: expected exactly one source, found {list(sources)}")
    return next(iter(sources.values()))


def _column(columns: dict[str, str], logical_name: str, dataset: str) -> str:
    if logical_name not in columns:
        raise ValueError(f"{dataset}: missing logical column mapping {logical_name!r}")
    return columns[logical_name]


def _treatment_column(columns: dict[str, str], dataset: str) -> str:
    matches = [name for name in TREATMENT_DUMMY_LOGICAL_NAMES if name in columns]
    if len(matches) != 1:
        raise ValueError(
            f"{dataset}: expected exactly one treatment dummy among "
            f"{TREATMENT_DUMMY_LOGICAL_NAMES}, found {matches}"
        )
    return columns[matches[0]]


def load_dataset_config(dataset: str, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")
    path = repo_root / "data" / dataset / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{dataset}: config not found at data/{dataset}/config.yaml")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{dataset}: config must be a YAML mapping")
    return loaded


def analysis_config(dataset: str, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    config = load_dataset_config(dataset, repo_root=repo_root)
    source = _one_source(config, dataset)
    columns = _require_key(source, "columns", dataset)
    panel = _require_key(config, "panel", dataset)
    if not isinstance(columns, dict):
        raise ValueError(f"{dataset}: source columns must be a mapping")
    if not isinstance(panel, dict):
        raise ValueError(f"{dataset}: panel must be a mapping")

    outcomes = _require_key(panel, "outcome", dataset)
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise ValueError(f"{dataset}: panel.outcome must contain exactly one outcome")

    covariates = _require_key(panel, "covariates", dataset)
    if not isinstance(covariates, list):
        raise ValueError(f"{dataset}: panel.covariates must be a list")

    weight_logical = _require_key(panel, "weight", dataset)
    file_name = _require_key(source, "file", dataset)

    resolved = {
        "dataset": dataset,
        "config_path": f"data/{dataset}/config.yaml",
        "data_path": f"data/{dataset}/{file_name}",
        "unit_col": _column(columns, _require_key(panel, "unit", dataset), dataset),
        "time_col": _column(columns, _require_key(panel, "time", dataset), dataset),
        "outcome": _column(columns, outcomes[0], dataset),
        "treatment": _treatment_column(columns, dataset),
        "controls": tuple(_column(columns, control, dataset) for control in covariates),
        "weight_col": _column(columns, weight_logical, dataset),
        "cluster_by": _column(columns, _require_key(panel, "unit", dataset), dataset),
    }
    return resolved


def required_columns(config: dict[str, Any]) -> tuple[str, ...]:
    ordered = [
        config["unit_col"],
        config["time_col"],
        config["outcome"],
        config["treatment"],
        config["cluster_by"],
        *config["controls"],
        config["weight_col"],
    ]
    return tuple(dict.fromkeys(ordered))


def validate_required_columns(dataset: str, config: dict[str, Any], columns: Iterable[str]) -> None:
    available = set(columns)
    missing = [column for column in required_columns(config) if column not in available]
    if missing:
        raise ValueError(f"{dataset}: missing required columns: {missing}")


def load_dataset_frame(
    dataset: str,
    config: dict[str, Any],
    repo_root: Path = REPO_ROOT,
) -> pd.DataFrame:
    path = repo_root / config["data_path"]
    if not path.exists():
        raise FileNotFoundError(f"{dataset}: dataset CSV not found at {config['data_path']}")
    df = pd.read_csv(path)
    validate_required_columns(dataset, config, df.columns)
    return df


def build_spike_specs(dataset: str, config: dict[str, Any]) -> list[Specification]:
    control_levels = ((), config["controls"])
    weight_levels = (None, config["weight_col"])
    specs: list[Specification] = []

    for controls in control_levels:
        for weight_col in weight_levels:
            for estimator in SUPPORTED_ESTIMATORS:
                spec_id = f"{dataset}_{len(specs):02d}"
                specs.append(
                    Specification(
                        spec_id=spec_id,
                        outcome=config["outcome"],
                        treatment=config["treatment"],
                        controls=tuple(controls),
                        unit_fe=config["unit_col"] if estimator == "TWFE" else None,
                        time_fe=config["time_col"] if estimator == "TWFE" else None,
                        cluster_by=config["cluster_by"],
                        estimator=estimator,
                        sample_filter=None,
                        include_never_treated=True,
                        weight_col=weight_col,
                    )
                )

    if len(specs) > SETTINGS.max_specifications:
        raise ValueError(
            f"{dataset}: generated {len(specs)} specs, above cap {SETTINGS.max_specifications}"
        )
    return specs


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def spec_to_report(spec: Specification) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "estimator": spec.estimator,
        "controls": list(spec.controls),
        "weight_col": spec.weight_col,
        "sample_filter": spec.sample_filter,
        "cluster_by": spec.cluster_by,
        "unit_fe": spec.unit_fe,
        "time_fe": spec.time_fe,
    }


def _failed_specs(results: Sequence[EstimationResult]) -> list[dict[str, Any]]:
    return [
        {"spec_id": result.spec_id, "estimator": result.estimator, "error": result.error}
        for result in results
        if result.status != "ok"
    ]


def _failure_reasons(results: Sequence[EstimationResult]) -> dict[str, int]:
    counter = Counter(result.error or "unknown" for result in results if result.status != "ok")
    return dict(sorted(counter.items()))


def _coefficient_range(results: Sequence[EstimationResult]) -> dict[str, float | None]:
    coefs = [
        float(result.coefficient)
        for result in results
        if result.status == "ok" and result.coefficient is not None
    ]
    return {
        "min": round(min(coefs), 6) if coefs else None,
        "max": round(max(coefs), 6) if coefs else None,
    }


def _descriptive_dominant_sign_axis(
    matched_pairs: dict[str, dict[str, Any]],
) -> str | None:
    """Rank axes descriptively by sign flips, rate, then effect-size movement.

    This is a readability heuristic for the spike report, not a formal statistical
    dominance test. Matched-pair counts and rates still come from diagnose_axes.
    """
    candidates = [
        (axis, metrics)
        for axis, metrics in matched_pairs.items()
        if metrics["sign_flip_count"] > 0
    ]
    if not candidates:
        return None
    axis, _metrics = max(
        candidates,
        key=lambda item: (
            item[1]["sign_flip_count"],
            item[1]["sign_flip_rate"] or 0.0,
            item[1]["mean_abs_delta"] or 0.0,
        ),
    )
    return axis


def _dominant_partial_r2_axis(partial_r2: dict[str, float | None]) -> str | None:
    candidates = [(axis, value) for axis, value in partial_r2.items() if value is not None]
    if not candidates:
        return None
    axis, value = max(candidates, key=lambda item: item[1] or 0.0)
    if value is None or value <= 0:
        return None
    return axis


def _dataset_decision(
    results: Sequence[EstimationResult],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    ok = [result for result in results if result.status == "ok" and result.coefficient is not None]
    matched_pairs = diagnosis["matched_pairs"]
    comparable_pairs = sum(
        metrics["sign_comparable_pairs"] for metrics in matched_pairs.values()
    )
    sign_flips = sum(metrics["sign_flip_count"] for metrics in matched_pairs.values())
    dominant_sign_axis = _descriptive_dominant_sign_axis(matched_pairs)
    dominant_partial_r2_axis = _dominant_partial_r2_axis(diagnosis["anova_partial_r2"])
    estimator_sign_flips = matched_pairs.get("estimator", {}).get("sign_flip_count", 0)

    if not ok:
        status = "INCONCLUSIVE"
        reason = "all specifications failed or returned missing coefficients"
    elif comparable_pairs == 0:
        status = "INCONCLUSIVE"
        reason = "no sign-comparable matched pairs"
    elif sign_flips > 0 and dominant_sign_axis is not None:
        status = "GO"
        if dominant_sign_axis == "estimator":
            reason = "readable estimator-dominant sign fragility"
        elif estimator_sign_flips > 0:
            reason = (
                f"readable axis fragility; {dominant_sign_axis} is dominant, "
                "with estimator flips also observed"
            )
        else:
            reason = (
                f"readable axis fragility; {dominant_sign_axis} is dominant "
                "and no estimator flip was observed"
            )
    else:
        status = "NO-GO"
        reason = "successful comparable specs, but no readable sign flip"

    return {
        "status": status,
        "reason": reason,
        "sign_comparable_pairs": comparable_pairs,
        "sign_flip_count": sign_flips,
        "descriptive_dominant_sign_axis": dominant_sign_axis,
        "dominant_partial_r2_axis": dominant_partial_r2_axis,
    }


def dataset_report(
    dataset: str,
    config: dict[str, Any],
    specs: Sequence[Specification],
    results: Sequence[EstimationResult] | None,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    base = {
        "dataset": dataset,
        "config_path": config["config_path"],
        "data_path": config["data_path"],
        "required_columns": list(required_columns(config)),
        "spec_count": len(specs),
        "specs": [spec_to_report(spec) for spec in specs],
        "limitations": [
            "sample axis not included: no canonical estimator-applied sample filter is defined",
            "pre_period_window not included: current estimators do not apply it",
            "include_never_treated not included as an axis: current estimators do not apply it",
            "canonical staggered estimators are not currently supported; only OLS and TWFE run",
        ],
    }

    if dry_run:
        return {
            **base,
            "dry_run": True,
            "n_total": len(specs),
            "n_ok": None,
            "n_failed": None,
            "coefficient_range": {"min": None, "max": None},
            "summary": None,
            "axis_diagnosis": None,
            "failed_specs": [],
            "failure_reasons": {},
            "decision": {
                "status": "INCONCLUSIVE",
                "reason": "dry-run validates config and spec matrix only",
                "sign_comparable_pairs": 0,
                "sign_flip_count": 0,
                "descriptive_dominant_sign_axis": None,
                "dominant_partial_r2_axis": None,
            },
        }

    if results is None:
        raise ValueError("results are required when dry_run is False")
    summary = summarize(list(results))
    diagnosis = diagnose_axes(list(results), list(specs))
    decision = _dataset_decision(results, diagnosis)

    return _jsonable(
        {
            **base,
            "dry_run": False,
            "n_total": len(results),
            "n_ok": summary["n_ok"],
            "n_failed": summary["n_failed"],
            "coefficient_range": _coefficient_range(results),
            "summary": dict(summary),
            "axis_diagnosis": diagnosis,
            "failed_specs": _failed_specs(results),
            "failure_reasons": _failure_reasons(results),
            "decision": decision,
        }
    )


def _overall_decision(dataset_reports: Sequence[dict[str, Any]]) -> dict[str, str]:
    dataset_names = _format_dataset_names(report["dataset"] for report in dataset_reports)
    statuses = [report["decision"]["status"] for report in dataset_reports]
    if "INCONCLUSIVE" in statuses:
        return {
            "status": "INCONCLUSIVE",
            "reason": (
                f"at least one analyzed dataset ({dataset_names}) lacks enough "
                "successful comparable results"
            ),
        }
    if "GO" in statuses:
        return {
            "status": "GO",
            "reason": (
                f"all analyzed datasets ({dataset_names}) are conclusive and at least "
                "one has readable sign fragility"
            ),
        }
    return {
        "status": "NO-GO",
        "reason": (
            f"no readable sign flip in analyzed dataset(s): {dataset_names}; "
            "backstop axis expansion required"
        ),
    }


def _format_dataset_names(dataset_names: Iterable[Any]) -> str:
    names = [str(name) for name in dataset_names]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _spec_count_summary(dataset_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = {
        str(report["dataset"]): int(report["spec_count"]) for report in dataset_reports
    }
    unique_counts = set(by_dataset.values())
    return {
        "specs_per_dataset": unique_counts.pop() if len(unique_counts) == 1 else None,
        "spec_counts_by_dataset": by_dataset,
    }


def run_dataset(dataset: str, *, dry_run: bool, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    config = analysis_config(dataset, repo_root=repo_root)
    df = load_dataset_frame(dataset, config, repo_root=repo_root)
    specs = build_spike_specs(dataset, config)
    results = None if dry_run else run_specs(df, specs)
    return dataset_report(dataset, config, specs, results, dry_run=dry_run)


def run_spike(dataset: str, *, dry_run: bool, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    datasets = DATASETS if dataset == "all" else (dataset,)
    reports = [run_dataset(name, dry_run=dry_run, repo_root=repo_root) for name in datasets]
    report = {
        "spike": "S2-15 Flip spike",
        "dry_run": dry_run,
        "spec_matrix": {
            "controls": ["none", "configured control set"],
            "weighting": ["unweighted", "configured weight column"],
            "estimators": list(SUPPORTED_ESTIMATORS),
            "sample_axis": "not included",
            **_spec_count_summary(reports),
        },
        "decision_rules": [
            "GO: successful comparable matched-pair results show at least one "
            "positive-negative sign flip with readable axis attribution.",
            "GO validates readable axis fragility in the current axis set; it does "
            "not by itself validate a canonical estimator-only or staggered-estimator story.",
            "NO-GO: successful comparable results exist for both datasets, but neither "
            "has a readable sign flip; backstop axis expansion required.",
            "INCONCLUSIVE: estimator/spec failures or no sign-comparable matched pairs "
            "prevent a reliable product decision.",
            "Descriptive dominant sign axis ranks axes by sign_flip_count, then "
            "sign_flip_rate, then mean_abs_delta; this is not formal statistical dominance.",
        ],
        "limitations": [
            "sample axis excluded because config and current estimators do not define "
            "a canonical applied filter",
            "pre_period_window and include_never_treated excluded because current "
            "estimators do not apply them",
            "canonical staggered estimators are not supported in the committed core; "
            "only OLS and TWFE are run",
        ],
        "datasets": reports,
    }
    report["overall_decision"] = _overall_decision(reports)
    return _jsonable(report)


def _format_rate(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    shared_spec_count = report["spec_matrix"].get("specs_per_dataset")
    spec_counts_by_dataset = report["spec_matrix"].get("spec_counts_by_dataset", {})
    if shared_spec_count is not None:
        spec_count_line = f"- Specs per dataset: {shared_spec_count}"
    else:
        formatted_counts = "; ".join(
            f"{dataset}={count}" for dataset, count in spec_counts_by_dataset.items()
        )
        spec_count_line = f"- Specs per dataset: {formatted_counts}"

    lines = [
        "# S2-15 Flip Spike",
        "",
        "## Spec Matrix",
        "",
        "- Controls: none; configured control set",
        "- Weighting: unweighted; configured weight column",
        f"- Estimators: {', '.join(report['spec_matrix']['estimators'])}",
        "- Sample axis: not included",
        spec_count_line,
        "",
        "## Decision Rules",
        "",
    ]
    lines.extend(f"- {rule}" for rule in report["decision_rules"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(
        [
            "",
            "## Overall Decision",
            "",
            f"**{report['overall_decision']['status']}**: {report['overall_decision']['reason']}",
            "",
        ]
    )

    for dataset in report["datasets"]:
        decision = dataset["decision"]
        summary = dataset["summary"] or {}
        matched = (dataset["axis_diagnosis"] or {}).get("matched_pairs", {})
        lines.extend(
            [
                f"## {dataset['dataset'].title()}",
                "",
                f"- Decision: **{decision['status']}** - {decision['reason']}",
                f"- Data: `{dataset['data_path']}`",
                f"- Specs: {dataset['spec_count']}",
                f"- Successful specs: {dataset['n_ok']}",
                f"- Failed specs: {dataset['n_failed']}",
                f"- Band: {summary.get('band')}",
                f"- Sign agreement: {_format_rate(summary.get('sign_agreement'))}",
                "- Coefficient range: "
                f"{dataset['coefficient_range']['min']} to "
                f"{dataset['coefficient_range']['max']}",
                "- Descriptive dominant sign axis: "
                f"{decision['descriptive_dominant_sign_axis']}",
                f"- Dominant partial-R2 axis: {decision['dominant_partial_r2_axis']}",
                "",
                "### Matched-Pair Axis Attribution",
                "",
                "| Axis | Pairs | Sign-comparable | Sign flips | Sign flip rate | "
                "Significance-comparable | Significance flips | Significance flip rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for axis, metrics in matched.items():
            lines.append(
                "| "
                f"{axis} | {metrics['n_pairs']} | {metrics['sign_comparable_pairs']} | "
                f"{metrics['sign_flip_count']} | {_format_rate(metrics['sign_flip_rate'])} | "
                f"{metrics['significance_comparable_pairs']} | "
                f"{metrics['significance_flip_count']} | "
                f"{_format_rate(metrics['significance_flip_rate'])} |"
            )

        lines.extend(["", "### ANOVA Partial-R2", ""])
        partial_r2 = (dataset["axis_diagnosis"] or {}).get("anova_partial_r2", {})
        if partial_r2:
            for axis, value in partial_r2.items():
                lines.append(f"- {axis}: {_format_rate(value)}")
        else:
            lines.append("- Not run in dry-run mode.")

        if dataset["failed_specs"]:
            lines.extend(["", "### Failed Specs", ""])
            for failure in dataset["failed_specs"]:
                lines.append(
                    f"- {failure['spec_id']} ({failure['estimator']}): {failure['error']}"
                )
        else:
            lines.extend(["", "### Failed Specs", "", "- None"])

        lines.extend(["", "### Spec List", ""])
        for spec in dataset["specs"]:
            lines.append(
                "- "
                f"{spec['spec_id']}: estimator={spec['estimator']}, "
                f"controls={spec['controls']}, weight={spec['weight_col']}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json":
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        out_path.write_text(render_markdown(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pareto S2-15 flip spike")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", help="Write Markdown by default, or JSON when suffix is .json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_spike(args.dataset, dry_run=args.dry_run)
    if args.out:
        write_report(report, Path(args.out))
    else:
        print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
