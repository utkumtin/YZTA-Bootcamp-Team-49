from __future__ import annotations

import pandas as pd

from pareto.analysis.event_study_columns import infer_column


def test_infer_column_ignores_unrelated_columns() -> None:
    df = pd.DataFrame(columns=["state_id", "year", "uninsured_rate"])
    assert infer_column(df, "cohort", "cohort_id", "group") is None
