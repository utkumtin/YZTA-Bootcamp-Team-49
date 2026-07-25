from __future__ import annotations
from pareto.analysis.event_study_columns import build_event_study_payload

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



def test_build_event_study_payload_skips_without_dataframe() -> None:
    assert build_event_study_payload(None, estimand=None, analysis_state=None) is None


def test_build_event_study_payload_reports_missing_required_columns() -> None:
    df = pd.DataFrame({"state": ["a"], "year": [2014], "uninsured_rate": [10.0]})
    payload = build_event_study_payload(
        df,
        estimand=None,
        analysis_state={"unit_col": "state", "time_col": "year", "controls": ["pop"]},
    )
    assert payload["status"] == "skipped"
    assert "required columns" in payload["reason"]


def test_build_event_study_payload_uses_estimand_outcome_first() -> None:
    class _FakeEstimand:
        outcome = "uninsured_rate"

    class _FakeFrozen:
        estimand = _FakeEstimand()

    df = pd.DataFrame({"state": ["a"], "year": [2014], "uninsured_rate": [10.0]})
    payload = build_event_study_payload(
        df,
        estimand=_FakeFrozen(),
        analysis_state={"unit_col": "state", "time_col": "year", "controls": []},
    )
    assert payload["status"] == "pending_columns"
    assert payload["outcome_col"] == "uninsured_rate"
    assert payload["sources"]["outcome_col"] == "estimand"