"""pyfixest OLS/TWFE estimates against the committed R/fixest reference."""

from pathlib import Path

import pandas as pd
import pytest

from pareto.analysis.estimators import estimate_one
from pareto.cleaning.merge import build_panel, load_dataset_config
from pareto.spec import Specification

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDICAID_DIR = REPO_ROOT / "data" / "medicaid"
CDC_RAW = MEDICAID_DIR / "raw" / "cdc_wonder_mortality_2009_2019.tsv"
REFERENCE_CSV = REPO_ROOT / "tests" / "fixtures" / "r_reference_medicaid.csv"

# Estimates and confidence intervals should agree closely across fixest implementations.
ESTIMATE_TOLERANCE = {"rel": 1e-5, "abs": 1e-6}
# Tail probabilities amplify small numerical differences, so p-values get a modestly looser bound.
P_VALUE_TOLERANCE = {"rel": 1e-4, "abs": 1e-8}

REFERENCE_ROWS = pd.read_csv(REFERENCE_CSV).to_dict(orient="records")


def _optional_text(value: object) -> str | None:
    return None if pd.isna(value) else str(value)


@pytest.fixture(scope="module")
def medicaid_committed_sample() -> pd.DataFrame:
    if not CDC_RAW.exists():
        pytest.skip("Medicaid CDC raw fixture not available; skipping R-reference parity check")
    pytest.importorskip("pyfixest")

    config = load_dataset_config(MEDICAID_DIR)
    committed_cohorts = config["treatment"]["committed_baseline"]["cohorts"]
    assert committed_cohorts == [2014], "R reference fixture expects the 2014 committed cohort"

    df = build_panel(MEDICAID_DIR).df
    cohort = committed_cohorts[0]
    in_committed_cohort = df["treatment_cohort"].eq(cohort).fillna(False)
    never_treated = df["never_treated"].fillna(False).astype(bool)
    sample = df[in_committed_cohort | never_treated].copy()
    sample["treated_post"] = (
        sample["treatment_cohort"].eq(cohort).fillna(False) & (sample["year"] >= cohort)
    ).astype(int)
    return sample


@pytest.mark.parametrize("reference", REFERENCE_ROWS, ids=lambda row: row["spec_id"])
def test_ols_twfe_match_r_fixest_reference(
    medicaid_committed_sample: pd.DataFrame, reference: dict[str, object]
) -> None:
    controls_text = _optional_text(reference["controls"])
    controls = tuple(controls_text.split(";")) if controls_text else ()
    estimator = str(reference["estimator"]).upper()
    spec = Specification(
        spec_id=str(reference["spec_id"]),
        outcome=str(reference["outcome"]),
        treatment="treated_post",
        controls=controls,
        unit_fe="county_fips" if estimator == "TWFE" else None,
        time_fe="year" if estimator == "TWFE" else None,
        cluster_by=str(reference["cluster_by"]),
        estimator=estimator,
        weight_col=_optional_text(reference["weight_col"]),
    )

    result = estimate_one(spec, medicaid_committed_sample)

    assert result.status == "ok", result.error
    assert result.n_obs == int(reference["n_complete_cases"])
    for field in ("coefficient", "std_error", "ci_low", "ci_high"):
        assert getattr(result, field) == pytest.approx(
            float(reference[field]), **ESTIMATE_TOLERANCE
        )
    assert result.p_value == pytest.approx(
        float(reference["p_value"]), **P_VALUE_TOLERANCE
    )
