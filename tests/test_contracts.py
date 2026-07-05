import pandas as pd
import pytest

from pareto.contracts import CleanPanel, PanelManifest, Tier


def test_validate_fail_loud_on_missing_column():
    # NEDEN: cleaning→analysis dikişi sessizce yanlış kolonla ilerlemesin.
    df = pd.DataFrame({"y": [1, 2]})
    panel = CleanPanel(df=df, manifest=PanelManifest(outcome_cols=("y",), unit_col="missing"))
    with pytest.raises(ValueError):
        panel.validate_contract()


def test_tier1_panel_did_when_panel_and_treatment_present():
    df = pd.DataFrame({"y": [1, 2], "u": [1, 2], "t": [1, 2], "d": [0, 1]})
    panel = CleanPanel(
        df=df,
        manifest=PanelManifest(outcome_cols=("y",), unit_col="u", time_col="t", treatment_col="d"),
    )
    assert panel.validate_contract() is Tier.TIER1_PANEL_DID


def test_tier2_cross_ols_when_no_panel_structure():
    # NEDEN: yapı yoksa nazikçe OLS-multiverse'e degrade + "ilişkisel, nedensel değil".
    df = pd.DataFrame({"y": [1, 2], "x": [3, 4]})
    panel = CleanPanel(df=df, manifest=PanelManifest(outcome_cols=("y",), covariate_cols=("x",)))
    assert panel.validate_contract() is Tier.TIER2_CROSS_OLS
