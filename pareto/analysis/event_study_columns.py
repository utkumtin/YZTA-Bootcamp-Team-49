from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict, cast

import pandas as pd


class EventStudySkippedPayload(TypedDict):
    status: Literal["skipped"]
    reason: str


class EventStudyPayload(TypedDict):
    """Column choices needed to run the panel's event-study diagnostic."""

    status: Literal["pending_columns"]
    outcome_col: str
    unit_col: str
    time_col: str
    cohort_col: str | None
    never_treated_col: str | None
    controls: tuple[str, ...]
    column_options: list[str]
    sources: dict[str, str]


EventStudyPayloadResult: TypeAlias = EventStudyPayload | EventStudySkippedPayload


def infer_column(df: pd.DataFrame, *candidates: str) -> str | None:
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return cast(str, lowered[candidate.lower()])
    return None


def event_study_cache_key(
    *,
    results_path: str,
    df: pd.DataFrame,
    payload: EventStudyPayload,
    cohort_col: str,
    never_treated_col: str,
    treated_cohorts: tuple[object, ...],
) -> tuple[object, ...]:
    return (
        results_path,
        tuple(str(col) for col in df.columns),
        df.shape,
        payload["outcome_col"],
        payload["unit_col"],
        payload["time_col"],
        cohort_col,
        never_treated_col,
        treated_cohorts,
        payload["controls"],
    )
