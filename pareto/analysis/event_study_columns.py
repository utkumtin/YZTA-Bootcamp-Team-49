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


def build_event_study_payload(
    df: pd.DataFrame | None,
    *,
    estimand: object | None,
    analysis_state: dict | None,
) -> EventStudyPayloadResult | None:
    """Pre-trend event-study için gerekli kolon eşlemesini hesaplar.

    NEDEN buraya taşındı (#51/12): eskiden `3_variance_panel.py` içinde
    `st.session_state`'i doğrudan okuyan bir yardımcıydı — Streamlit'ten
    bağımsız test edilemiyordu ve sayfa akışının ortasına gömülüydü. Artık
    saf bir fonksiyon: girdileri parametre olarak alır, session_state'e
    dokunmaz. Sayfa yalnızca session_state'ten okuyup bu fonksiyona geçirir.
    """
    if df is None:
        return None

    state = analysis_state or {}

    outcome_col = None
    outcome_source = "unknown"
    if estimand is not None:
        outcome_col = getattr(getattr(estimand, "estimand", None), "outcome", None)
        if outcome_col:
            outcome_source = "estimand"

    if not outcome_col:
        outcome_col = infer_column(df, "outcome", "y", "dependent")
        outcome_source = "inferred" if outcome_col else "unknown"

    unit_col = state.get("unit_col")
    unit_source = "analysis_state" if unit_col else "inferred"
    if not unit_col:
        unit_col = infer_column(df, "unit", "unit_id", "id")

    time_col = state.get("time_col")
    time_source = "analysis_state" if time_col else "inferred"
    if not time_col:
        time_col = infer_column(df, "year", "time", "date", "period")

    cohort_col = infer_column(df, "cohort", "cohort_id", "treatment_time", "group")
    cohort_source = "inferred" if cohort_col else "unknown"

    never_treated_col = infer_column(df, "never_treated", "never_treat", "untreated", "control")
    never_treated_source = "inferred" if never_treated_col else "unknown"

    controls = tuple(str(control) for control in (state.get("controls") or []))

    if not outcome_col or not unit_col or not time_col:
        # Bu metin panelde doğrudan kullanıcıya basılıyor; arayüzün geri kalanı
        # gibi Türkçe olmalı.
        return {
            "status": "skipped",
            "reason": "Pre-trend teşhisi için gerekli kolonlar eksik.",
        }

    return {
        "status": "pending_columns",
        "outcome_col": outcome_col,
        "unit_col": unit_col,
        "time_col": time_col,
        "cohort_col": cohort_col,
        "never_treated_col": never_treated_col,
        "controls": controls,
        "column_options": [str(col) for col in df.columns],
        "sources": {
            "outcome_col": outcome_source,
            "unit_col": unit_source,
            "time_col": time_source,
            "cohort_col": cohort_source,
            "never_treated_col": never_treated_source,
        },
    }
