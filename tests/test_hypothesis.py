import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from pareto.analysis.hypothesis import (
    SocraticDeclaration,
    TACProposal,
    elicit_estimand,
    freeze_estimand,
    validate_estimand_spec_mapping,
)
from pareto.llm.router import use_test_model
from pareto.spec import Specification


def _proposal_args() -> dict[str, object]:
    return {
        "estimand_type": "ATT",
        "treatment": "Medicaid expansion adoption",
        "treatment_coding": "expanded",
        "outcome": "uninsured_rate",
        "outcome_unit": "percentage points",
        "population": "US states in the clean panel",
        "time_scope": "2010-2020",
        "expected_sign": "negative",
        "identification_assumption": "parallel_trends",
        "h0": "ATT = 0",
        "h1": "ATT < 0",
        "implied_result_translation": "Expansion states should see lower uninsured rates.",
        "confirmation_question": "Should I freeze expanded -> uninsured_rate as a negative ATT?",
        "needs_clarification": False,
        "clarification_question": None,
    }


def test_testmodel_dialogue_freezes_expected_estimand_fields():
    declaration = SocraticDeclaration(
        conceptual_treatment="Medicaid expansion",
        conceptual_outcome="uninsured rate",
        expected_sign="negative",
    )

    with use_test_model(TestModel(custom_output_args=_proposal_args())):
        frozen = elicit_estimand(
            "Estimate whether Medicaid expansion lowered uninsured rates.",
            ["state", "year", "expanded", "uninsured_rate"],
            declaration,
            approved=True,
        )

    assert frozen.estimand.treatment == "Medicaid expansion adoption"
    assert frozen.estimand.treatment_coding == "expanded"
    assert frozen.estimand.outcome == "uninsured_rate"
    assert frozen.estimand.expected_sign == "negative"
    assert frozen.freeze_hash == frozen.estimand.freeze_hash()


def test_freeze_estimand_fails_when_agent_is_uncertain():
    proposal = TACProposal(
        **{
            **_proposal_args(),
            "needs_clarification": True,
            "clarification_question": "Which treatment column encodes expansion?",
        }
    )

    with pytest.raises(ValueError, match="targeted confirmation needed"):
        freeze_estimand(proposal, approved=True)


def test_freeze_estimand_requires_user_approval():
    proposal = TACProposal(**_proposal_args())

    with pytest.raises(ValueError, match="user approval is required"):
        freeze_estimand(proposal, approved=False)


def test_validate_estimand_spec_mapping_accepts_clean_twfe_mapping():
    proposal = TACProposal(**_proposal_args())
    frozen = freeze_estimand(proposal, approved=True)
    spec = Specification(
        spec_id="s",
        outcome="uninsured_rate",
        treatment="expanded",
        cluster_by="state",
        estimator="TWFE",
        unit_fe="state",
        time_fe="year",
    )

    validate_estimand_spec_mapping(
        frozen,
        spec,
        available_columns=["state", "year", "expanded", "uninsured_rate"],
    )


def test_validate_estimand_spec_mapping_fails_loud_on_dirty_mapping():
    proposal = TACProposal(**_proposal_args())
    frozen = freeze_estimand(proposal, approved=True)
    spec = Specification(
        spec_id="s",
        outcome="mortality_rate",
        treatment="expanded",
        cluster_by="state",
        estimator="TWFE",
        unit_fe="state",
        time_fe="year",
    )

    with pytest.raises(ValueError, match="Estimand -> Specification mapping error"):
        validate_estimand_spec_mapping(
            frozen,
            spec,
            available_columns=["state", "year", "expanded", "mortality_rate"],
        )


def test_socratic_declaration_requires_expected_sign_before_agent():
    with pytest.raises(ValidationError):
        SocraticDeclaration(
            conceptual_treatment="Medicaid expansion",
            conceptual_outcome="uninsured rate",
            expected_sign="lower",
        )
