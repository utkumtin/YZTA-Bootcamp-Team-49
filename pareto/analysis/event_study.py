"""Deterministic baseline event-study / pre-trend diagnostic helper.

This module is intentionally independent from the multiverse estimator runner. It
builds a small TWFE event-study check that the variance panel can consume as a
plain JSON-serializable dict. The language is diagnostic: coefficients are
parallel-trend evidence, not proof.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd


DEFAULT_REFERENCE_PERIOD = -1
DEFAULT_EVENT_TIME_WINDOW = (-4, 4)


def estimate_pretrend_event_study(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    unit_col: str,
    time_col: str,
    cohort_col: str,
    never_treated_col: str,
    controls: Sequence[str] = (),
    weight_col: str | None = None,
    treated_cohorts: Sequence[int | float | str] | None = None,
    event_time_window: tuple[int, int] = DEFAULT_EVENT_TIME_WINDOW,
    reference_period: int = DEFAULT_REFERENCE_PERIOD,
) -> dict[str, Any]:
    """Estimate a TWFE event-study series for a committed-cohort diagnostic check.

    ``treated_cohorts=None`` is a broad fallback that treats all non-missing
    cohorts as treated. Medicaid committed baseline callers should pass an
    explicit cohort such as ``treated_cohorts=(2014,)``. This helper produces a
    diagnostic series for parallel-trend evidence, not proof.

    ``treated_cohorts`` lets committed-baseline calls keep one cohort as treated
    while retaining never-treated controls and excluding other/not-yet-treated
    cohorts. The reference period is omitted from dummy construction, so every
    displayed coefficient is relative to it. Column names should be pyfixest
    formula-safe.
    """
    controls = tuple(dict.fromkeys(controls))
    treated_cohort_keys = _cohort_keys(treated_cohorts)
    base = _base_result(
        event_time_window=event_time_window,
        reference_period=reference_period,
        treated_cohorts=treated_cohort_keys,
    )

    window_min, window_max = event_time_window
    if window_min > window_max:
        return _failed_result(base, "event_time_window min must be <= max.")
    if not window_min <= reference_period <= window_max:
        return _failed_result(base, "reference_period must be inside event_time_window.")

    dummy_times = [t for t in range(window_min, window_max + 1) if t != reference_period]
    dummy_cols = [_event_dummy_name(t) for t in dummy_times]
    conflicts = sorted((set(df.columns) | set(controls)) & set(dummy_cols))
    if conflicts:
        return _failed_result(
            base,
            "Generated event-study dummy columns conflict with existing columns "
            f"or controls: {conflicts}",
        )

    required = [outcome_col, unit_col, time_col, cohort_col, never_treated_col, *controls]
    if weight_col is not None:
        required.append(weight_col)
    missing = [col for col in dict.fromkeys(required) if col not in df.columns]
    if missing:
        return _failed_result(base, f"Missing required columns: {missing}")

    work_result = _build_working_frame(
        df,
        outcome_col=outcome_col,
        unit_col=unit_col,
        time_col=time_col,
        cohort_col=cohort_col,
        never_treated_col=never_treated_col,
        controls=controls,
        weight_col=weight_col,
        treated_cohort_keys=treated_cohort_keys,
        event_time_window=event_time_window,
        reference_period=reference_period,
    )
    if isinstance(work_result, str):
        return _failed_result(base, work_result)
    work, used_cohort_keys, bin_counts, warnings = work_result
    if treated_cohorts is None:
        warnings.insert(
            0,
            "treated_cohorts not provided; using all non-missing cohorts as treated. "
            "For committed-baseline checks, pass an explicit treated cohort.",
        )
    base["treated_cohorts"] = _jsonable(used_cohort_keys)
    base["warnings"].extend(warnings)
    base["n_obs"] = int(len(work))

    rhs = " + ".join([*dummy_cols, *controls])
    if not rhs:
        return _failed_result(base, "No event-time dummies or controls available for model.")

    formula = f"{outcome_col} ~ {rhs} | {unit_col} + {time_col}"
    kwargs: dict[str, Any] = {"vcov": {"CRV1": unit_col}}
    if weight_col is not None:
        kwargs["weights"] = weight_col

    try:
        fit = _fit_model(formula, work, kwargs)
    except Exception as exc:  # noqa: BLE001
        return _failed_result(base, f"Model fit failed: {exc}")

    series: list[dict[str, Any]] = []
    for event_time, dummy_col in zip(dummy_times, dummy_cols, strict=True):
        point, warning = _series_point(fit, dummy_col, event_time, bin_counts[event_time])
        if warning is not None:
            base["warnings"].append(warning)
        series.append(point)

    base["status"] = "ok"
    base["series"] = series
    return _jsonable(base)


def _base_result(
    *,
    event_time_window: tuple[int, int],
    reference_period: int,
    treated_cohorts: list[Any],
) -> dict[str, Any]:
    return {
        "status": "failed",
        "estimator": "TWFE",
        "reference_period": int(reference_period),
        "event_time_window": {
            "min": int(event_time_window[0]),
            "max": int(event_time_window[1]),
        },
        "treated_cohorts": _jsonable(treated_cohorts),
        "series": [],
        "n_obs": 0,
        "warnings": [],
        "error": None,
    }


def _failed_result(base: dict[str, Any], error: str) -> dict[str, Any]:
    out = {**base, "status": "failed", "error": error}
    return _jsonable(out)


def _cohort_key(value: Any) -> int | float | str | None:
    if _is_missing(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not _is_missing(numeric):
        as_float = float(numeric)
        if as_float.is_integer():
            return int(as_float)
        return as_float
    return str(value)


def _cohort_keys(values: Sequence[int | float | str] | None) -> list[Any]:
    if values is None:
        return []
    keys = [_cohort_key(value) for value in values]
    return sorted({key for key in keys if key is not None}, key=lambda item: str(item))


def _normalize_never_treated(value: Any) -> bool | None:
    if _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _build_working_frame(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    unit_col: str,
    time_col: str,
    cohort_col: str,
    never_treated_col: str,
    controls: Sequence[str],
    weight_col: str | None,
    treated_cohort_keys: list[Any],
    event_time_window: tuple[int, int],
    reference_period: int,
) -> tuple[pd.DataFrame, list[Any], dict[int, int], list[str]] | str:
    work = df.copy()
    normalized_never_treated = work[never_treated_col].map(_normalize_never_treated)
    if bool(normalized_never_treated.isna().any()):
        return "never_treated_col contains unrecognized values."
    work["_pareto_never_treated"] = normalized_never_treated.astype(bool)
    if not bool(work["_pareto_never_treated"].any()):
        return "No never-treated control observations remain after normalization."
    work["_pareto_cohort_key"] = work[cohort_col].map(_cohort_key)

    if treated_cohort_keys:
        raw_treated_mask = work["_pareto_cohort_key"].isin(treated_cohort_keys)
    else:
        raw_treated_mask = work["_pareto_cohort_key"].notna()

    # Never-treated units are the control group for this committed-baseline check.
    # Other cohorts are excluded when treated_cohorts pins a committed cohort.
    keep_mask = raw_treated_mask | work["_pareto_never_treated"]
    work = work.loc[keep_mask].copy()
    if work.empty:
        return "No treated or never-treated observations remain after cohort filtering."

    selected_treated = (
        work["_pareto_cohort_key"].isin(treated_cohort_keys)
        if treated_cohort_keys
        else work["_pareto_cohort_key"].notna()
    )
    used_cohort_keys = sorted(
        {key for key in work.loc[selected_treated, "_pareto_cohort_key"]},
        key=lambda item: str(item),
    )
    if not used_cohort_keys:
        return "No treated cohort observations remain after cohort filtering."

    try:
        work["_pareto_time"] = pd.to_numeric(work[time_col], errors="raise")
        work["_pareto_cohort"] = pd.to_numeric(work[cohort_col], errors="coerce")
    except Exception as exc:  # noqa: BLE001
        return f"time_col and cohort_col must be numeric or year-like: {exc}"

    selected_treated = work["_pareto_cohort_key"].isin(used_cohort_keys)
    if bool(work.loc[selected_treated, "_pareto_cohort"].isna().any()):
        return "treated cohort values must be numeric/year-like for event-time construction."
    work["_pareto_event_time"] = pd.NA
    work.loc[selected_treated, "_pareto_event_time"] = (
        work.loc[selected_treated, "_pareto_time"]
        - work.loc[selected_treated, "_pareto_cohort"]
    )
    work["_pareto_event_time"] = pd.to_numeric(work["_pareto_event_time"], errors="coerce")

    window_min, window_max = event_time_window
    in_window_treated = selected_treated & work["_pareto_event_time"].between(
        window_min,
        window_max,
    )
    non_reference_pre_period = (
        in_window_treated
        & (work["_pareto_event_time"] < 0)
        & (work["_pareto_event_time"] != reference_period)
    )
    if not bool(non_reference_pre_period.any()):
        return "No non-reference pre-period event-time observations available for diagnostic check."
    if not bool((selected_treated & (work["_pareto_event_time"] == reference_period)).any()):
        return "No observations found for the reference_period."

    dummy_times = [t for t in range(window_min, window_max + 1) if t != reference_period]
    bin_counts: dict[int, int] = {}
    for event_time in dummy_times:
        dummy_col = _event_dummy_name(event_time)
        bin_mask = selected_treated & (work["_pareto_event_time"] == event_time)
        # The reference period is the omitted category, so no dummy is generated.
        # Never-treated controls stay in sample with all event-time dummies set to 0.
        # Committed-baseline mode excludes other treated cohorts when pinned above.
        work[dummy_col] = bin_mask.astype(float)
        bin_counts[event_time] = int(bin_mask.sum())

    if weight_col is not None:
        numeric_weights = pd.to_numeric(work[weight_col], errors="coerce")
        if bool(numeric_weights.isna().any()) or bool((numeric_weights <= 0).any()):
            return "weight_col must contain positive numeric weights."
        work[weight_col] = numeric_weights

    fit_cols = [
        outcome_col,
        unit_col,
        time_col,
        *controls,
        *([weight_col] if weight_col is not None else []),
        *[_event_dummy_name(t) for t in dummy_times],
    ]
    before = len(work)
    work = work.dropna(subset=fit_cols)
    warnings: list[str] = []
    if len(work) < before:
        warnings.append(f"{before - len(work)} rows dropped due to missing model inputs.")
    if work.empty:
        return "No observations remain after dropping missing model inputs."

    bin_counts = {
        event_time: int(
            (
                work["_pareto_cohort_key"].isin(used_cohort_keys)
                & (work["_pareto_event_time"] == event_time)
            ).sum()
        )
        for event_time in dummy_times
    }

    return work, used_cohort_keys, bin_counts, warnings


def _event_dummy_name(event_time: int) -> str:
    if event_time < 0:
        return f"event_m{abs(event_time)}"
    if event_time > 0:
        return f"event_p{event_time}"
    return "event_0"


def _fit_model(formula: str, data: pd.DataFrame, kwargs: dict[str, Any]):  # noqa: ANN201
    import pyfixest as pf

    return pf.feols(formula, data=data, **kwargs)


def _series_point(
    fit: Any,
    dummy_col: str,
    event_time: int,
    n_obs: int,
) -> tuple[dict[str, Any], str | None]:
    coef = fit.coef()
    if dummy_col not in coef.index:
        return _empty_point(event_time, n_obs), f"{dummy_col} dropped or omitted by TWFE fit."

    try:
        ci = fit.confint().loc[dummy_col]
        pvalues = fit.pvalue()
        point = {
            "event_time": int(event_time),
            "coefficient": _number_or_none(coef[dummy_col]),
            "ci_low": _number_or_none(ci.iloc[0]),
            "ci_high": _number_or_none(ci.iloc[1]),
            "p_value": _number_or_none(pvalues[dummy_col]),
            "n_obs": int(n_obs),
        }
    except Exception as exc:  # noqa: BLE001
        return _empty_point(event_time, n_obs), f"{dummy_col} extraction failed: {exc}"
    return point, None


def _empty_point(event_time: int, n_obs: int) -> dict[str, Any]:
    return {
        "event_time": int(event_time),
        "coefficient": None,
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "n_obs": int(n_obs),
    }


def _number_or_none(value: Any) -> float | None:
    if _is_missing(value):
        return None
    as_float = float(value)
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return as_float


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if _is_missing(value):
        return None
    if isinstance(value, bool | str | int | float):
        return value
    return str(value)
