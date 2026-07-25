from __future__ import annotations

import pandas as pd

from pareto.analysis.event_study_columns import (
    EventStudyPayload,
    event_study_cache_key,
    infer_column,
)


def test_infer_column_ignores_unrelated_columns() -> None:
    df = pd.DataFrame(columns=["state_id", "year", "uninsured_rate"])
    assert infer_column(df, "cohort", "cohort_id", "group") is None


def test_event_study_cache_key_uses_typed_controls() -> None:
    df = pd.DataFrame(columns=["outcome", "unit", "year"])
    payload: EventStudyPayload = {
        "status": "pending_columns",
        "outcome_col": "outcome",
        "unit_col": "unit",
        "time_col": "year",
        "cohort_col": None,
        "never_treated_col": None,
        "controls": ("income",),
        "column_options": ["outcome", "unit", "year"],
        "sources": {},
    }
    key = event_study_cache_key(
        results_path="runs/example/results.json",
        df=df,
        payload=payload,
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
    )
    assert key[-1] == ("income",)
