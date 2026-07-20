import json
from typing import Any

import pandas as pd
import pytest

import pareto.analysis.event_study as event_study
from pareto.analysis.event_study import estimate_pretrend_event_study


def _panel(
    *,
    years: range = range(2011, 2019),
    never_values: tuple[Any, Any] = (True, False),
) -> pd.DataFrame:
    rows = []
    cohorts = [2014] * 8 + [2015] * 4 + [pd.NA] * 8
    for unit, cohort in enumerate(cohorts):
        never = never_values[0] if pd.isna(cohort) else never_values[1]
        unit_fe = unit * 0.2
        for year in years:
            year_fe = (year - 2011) * 0.1
            event_time = None if pd.isna(cohort) else year - int(cohort)
            effect = 0.0
            if event_time is not None and event_time >= 0:
                effect = 0.5 + 0.1 * event_time
            y = unit_fe + year_fe + effect
            rows.append(
                {
                    "unit": f"u{unit:02d}",
                    "year": year,
                    "cohort": cohort,
                    "never_treated": never,
                    "y": y,
                    "x": unit % 3,
                    "weight": 1.0 + (unit % 5),
                }
            )
    return pd.DataFrame(rows)


def _estimate(df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    pytest.importorskip("pyfixest")
    event_time_window = kwargs.pop("event_time_window", (-3, 3))
    reference_period = kwargs.pop("reference_period", -1)
    controls = kwargs.pop("controls", ())
    return estimate_pretrend_event_study(
        df,
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        controls=controls,
        event_time_window=event_time_window,
        reference_period=reference_period,
        **kwargs,
    )


def test_event_study_synthetic_panel_ok_and_json_serializable():
    out = _estimate(_panel(), treated_cohorts=(2014,))

    assert out["status"] == "ok"
    assert out["estimator"] == "TWFE"
    assert out["n_obs"] == 128
    json.dumps(out)


def test_event_study_reference_period_is_omitted_from_series():
    out = _estimate(_panel(), treated_cohorts=(2014,))

    assert -1 not in {point["event_time"] for point in out["series"]}


def test_event_study_pre_period_coefficients_are_produced():
    out = _estimate(_panel(), treated_cohorts=(2014,), event_time_window=(-3, 4))
    pre_points = [point for point in out["series"] if point["event_time"] < 0]
    event_zero = next(point for point in out["series"] if point["event_time"] == 0)

    assert {point["event_time"] for point in pre_points} == {-3, -2}
    assert all(point["coefficient"] is not None for point in pre_points)
    assert all(abs(point["coefficient"]) < 0.25 for point in pre_points)
    assert event_zero["coefficient"] > 0.25


def test_event_study_committed_cohort_keeps_never_controls_and_excludes_other_cohorts():
    out = _estimate(_panel(), treated_cohorts=(2014,))

    assert out["treated_cohorts"] == [2014]
    assert out["n_obs"] == 128  # 8 treated 2014 units + 8 never-treated units, 8 years each


def test_event_study_all_treated_cohorts_used_when_not_pinned():
    out = _estimate(_panel())

    assert out["treated_cohorts"] == [2014, 2015]
    assert out["n_obs"] == 160


def test_event_study_missing_required_column_returns_failed_output():
    out = estimate_pretrend_event_study(
        _panel().drop(columns=["cohort"]),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
    )

    assert out["status"] == "failed"
    assert "Missing required columns" in out["error"]
    assert "cohort" in out["error"]


def test_event_study_no_pre_period_returns_failed_output():
    out = estimate_pretrend_event_study(
        _panel(years=range(2014, 2018)),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
        event_time_window=(-3, 3),
        reference_period=-1,
    )

    assert out["status"] == "failed"
    assert "No non-reference pre-period" in out["error"]


def test_event_study_only_reference_pre_period_returns_failed_output():
    out = estimate_pretrend_event_study(
        _panel(years=range(2013, 2018)),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
        event_time_window=(-3, 3),
        reference_period=-1,
    )

    assert out["status"] == "failed"
    assert "No non-reference pre-period" in out["error"]


def test_event_study_reference_period_outside_window_returns_failed_output():
    # This post-only window intentionally validates fail-loud reference-period handling.
    out = estimate_pretrend_event_study(
        _panel(),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        event_time_window=(0, 3),
        reference_period=-1,
    )

    assert out["status"] == "failed"
    assert "reference_period" in out["error"]


def test_event_study_missing_reference_observation_returns_failed_output():
    df = _panel()
    df = df[df["year"] != 2013]

    out = estimate_pretrend_event_study(
        df,
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
        event_time_window=(-3, 3),
        reference_period=-1,
    )

    assert out["status"] == "failed"
    assert "reference_period" in out["error"]


def test_event_study_non_numeric_treated_cohort_returns_failed_output():
    df = _panel()
    df.loc[df["cohort"] == 2014, "cohort"] = "2014Q1"

    out = estimate_pretrend_event_study(
        df,
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=("2014Q1",),
        event_time_window=(-3, 3),
        reference_period=-1,
    )

    assert out["status"] == "failed"
    assert "treated cohort values must be numeric/year-like" in out["error"]


def test_event_study_unrecognized_never_treated_values_return_failed_output():
    out = estimate_pretrend_event_study(
        _panel(never_values=("evet", "hayir")),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
    )

    assert out["status"] == "failed"
    assert out["error"] == "never_treated_col contains unrecognized values."


def test_event_study_requires_never_treated_controls():
    out = estimate_pretrend_event_study(
        _panel(never_values=(False, False)),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
    )

    assert out["status"] == "failed"
    assert out["error"] == "No never-treated control observations remain after normalization."


def test_event_study_existing_dummy_column_conflict_returns_failed_output():
    df = _panel()
    df["event_m2"] = 1.0

    out = estimate_pretrend_event_study(
        df,
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
        event_time_window=(-2, 0),
        reference_period=-1,
    )

    assert out["status"] == "failed"
    assert "Generated event-study dummy columns conflict" in out["error"]
    assert "event_m2" in out["error"]


def test_event_study_control_dummy_name_conflict_returns_failed_output():
    out = estimate_pretrend_event_study(
        _panel(),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        controls=("event_m2",),
        treated_cohorts=(2014,),
        event_time_window=(-2, 0),
        reference_period=-1,
    )

    assert out["status"] == "failed"
    assert "Generated event-study dummy columns conflict" in out["error"]
    assert "event_m2" in out["error"]


class _FakeFit:
    def coef(self) -> pd.Series:
        return pd.Series({"event_m2": -0.1, "event_0": 0.2})

    def confint(self) -> pd.DataFrame:
        return pd.DataFrame({0: [-0.2, 0.1], 1: [0.0, 0.3]}, index=["event_m2", "event_0"])

    def pvalue(self) -> pd.Series:
        return pd.Series({"event_m2": 0.2, "event_0": 0.01})


def test_event_study_extracts_available_event_bins(monkeypatch):
    monkeypatch.setattr(event_study, "_fit_model", lambda *_args, **_kwargs: _FakeFit())

    out = estimate_pretrend_event_study(
        _panel(),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
        event_time_window=(-2, 0),
        reference_period=-1,
    )

    dropped = next(point for point in out["series"] if point["event_time"] == 0)
    assert out["status"] == "ok"
    assert dropped["coefficient"] == 0.2
    omitted = next(point for point in out["series"] if point["event_time"] == -2)
    assert omitted["coefficient"] == -0.1
    assert all(point["event_time"] != -1 for point in out["series"])


def test_event_study_warns_when_treated_cohorts_not_provided(monkeypatch):
    monkeypatch.setattr(event_study, "_fit_model", lambda *_args, **_kwargs: _FakeFit())

    out = estimate_pretrend_event_study(
        _panel(),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        event_time_window=(-2, 0),
        reference_period=-1,
    )

    assert out["status"] == "ok"
    assert any("treated_cohorts not provided" in warning for warning in out["warnings"])


def test_event_study_no_treated_cohorts_warning_when_explicit(monkeypatch):
    monkeypatch.setattr(event_study, "_fit_model", lambda *_args, **_kwargs: _FakeFit())

    out = estimate_pretrend_event_study(
        _panel(),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
        event_time_window=(-2, 0),
        reference_period=-1,
    )

    assert out["status"] == "ok"
    assert not any("treated_cohorts not provided" in warning for warning in out["warnings"])


def test_event_study_deduplicates_controls_before_formula(monkeypatch):
    captured: dict[str, str] = {}

    def capture_fit(formula: str, *_args: Any, **_kwargs: Any) -> _FakeFit:
        captured["formula"] = formula
        return _FakeFit()

    monkeypatch.setattr(event_study, "_fit_model", capture_fit)

    out = estimate_pretrend_event_study(
        _panel(),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        controls=("x", "x"),
        treated_cohorts=(2014,),
        event_time_window=(-2, 0),
        reference_period=-1,
    )

    assert out["status"] == "ok"
    assert captured["formula"].count("x") == 1


def test_event_study_omitted_bin_none_metrics_and_warning(monkeypatch):
    monkeypatch.setattr(event_study, "_fit_model", lambda *_args, **_kwargs: _FakeFit())

    out = estimate_pretrend_event_study(
        _panel(),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
        event_time_window=(-3, 0),
        reference_period=-1,
    )

    omitted = next(point for point in out["series"] if point["event_time"] == -3)
    assert omitted["coefficient"] is None
    assert omitted["ci_low"] is None
    assert any("dropped or omitted" in warning for warning in out["warnings"])


def test_event_study_weight_col_runs():
    out = _estimate(_panel(), treated_cohorts=(2014,), weight_col="weight")

    assert out["status"] == "ok"
    assert out["series"]


def test_event_study_zero_weight_returns_failed_output():
    df = _panel()
    df.loc[df.index[0], "weight"] = 0.0

    out = estimate_pretrend_event_study(
        df,
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        weight_col="weight",
        treated_cohorts=(2014,),
    )

    assert out["status"] == "failed"
    assert out["error"] == "weight_col must contain positive numeric weights."


def test_event_study_negative_weight_returns_failed_output():
    df = _panel()
    df.loc[df.index[0], "weight"] = -1.0

    out = estimate_pretrend_event_study(
        df,
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        weight_col="weight",
        treated_cohorts=(2014,),
    )

    assert out["status"] == "failed"
    assert out["error"] == "weight_col must contain positive numeric weights."


def test_event_study_non_numeric_weight_returns_failed_output():
    df = _panel()
    df["weight"] = df["weight"].astype(object)
    df.loc[df.index[0], "weight"] = "bad"

    out = estimate_pretrend_event_study(
        df,
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        weight_col="weight",
        treated_cohorts=(2014,),
    )

    assert out["status"] == "failed"
    assert out["error"] == "weight_col must contain positive numeric weights."


def test_event_study_missing_weight_returns_failed_output():
    df = _panel()
    df.loc[df.index[0], "weight"] = pd.NA

    out = estimate_pretrend_event_study(
        df,
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        weight_col="weight",
        treated_cohorts=(2014,),
    )

    assert out["status"] == "failed"
    assert out["error"] == "weight_col must contain positive numeric weights."


def test_event_study_normalizes_non_bool_never_treated_values():
    out = _estimate(_panel(never_values=("true", "false")), treated_cohorts=(2014,))

    assert out["status"] == "ok"
    assert out["n_obs"] == 128


def test_event_study_output_has_no_pandas_or_numpy_types():
    out = _estimate(_panel(), treated_cohorts=(2014,))

    def walk(value: Any) -> None:
        assert not value.__class__.__module__.startswith(("numpy", "pandas"))
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(out)
    json.dumps(out)


def test_event_study_model_fit_exception_returns_failed_output(monkeypatch):
    def raise_fit(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(event_study, "_fit_model", raise_fit)

    out = estimate_pretrend_event_study(
        _panel(),
        outcome_col="y",
        unit_col="unit",
        time_col="year",
        cohort_col="cohort",
        never_treated_col="never_treated",
        treated_cohorts=(2014,),
        event_time_window=(-2, 0),
        reference_period=-1,
    )

    assert out["status"] == "failed"
    assert "boom" in out["error"]


def test_event_study_window_is_inclusive_and_excludes_outside_times():
    out = _estimate(_panel(), treated_cohorts=(2014,), event_time_window=(-2, 1))

    assert [point["event_time"] for point in out["series"]] == [-2, 0, 1]
