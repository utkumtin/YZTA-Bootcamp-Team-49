from __future__ import annotations

import pandas as pd


def infer_column(df: pd.DataFrame, *candidates: str) -> str | None:
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def event_study_cache_key(
    *,
    results_path: str,
    df: pd.DataFrame,
    payload: dict[str, object],
    cohort_col: str,
    never_treated_col: str,
    treated_cohorts: tuple[object, ...],
) -> tuple[object, ...]:
    controls = tuple(payload.get("controls", ()))
    return (
        results_path,
        tuple(str(col) for col in df.columns),
        df.shape,
        payload.get("outcome_col"),
        payload.get("unit_col"),
        payload.get("time_col"),
        cohort_col,
        never_treated_col,
        treated_cohorts,
        controls,
    )
