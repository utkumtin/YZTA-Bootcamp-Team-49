"""Context & Hypothesis - estimand-first.

The estimand is a typed theoretical object, not free-form LLM text. The flow is:
Socratic user declaration -> TAC technical proposal -> targeted confirmation ->
frozen estimand with a deterministic hash before any specification menu is built.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel

from ..config import ModelRole
from ..llm.router import build_agent
from ..spec import Specification


class Estimand(BaseModel):
    """Theoretical estimand, including the identifying assumption."""

    model_config = {"frozen": True}

    estimand_type: Literal["ATT", "ATE", "LATE"] = "ATT"
    treatment: str  # conceptual treatment + coding
    treatment_coding: str
    outcome: str  # outcome column
    outcome_unit: str
    population: str
    time_scope: str
    expected_sign: Literal["positive", "negative", "ambiguous"]
    identification_assumption: str = "parallel_trends"
    h0: str
    h1: str

    def freeze_hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class SocraticDeclaration(BaseModel):
    """The user's pre-agent conceptual declaration."""

    conceptual_treatment: str
    conceptual_outcome: str
    expected_sign: Literal["positive", "negative", "ambiguous"]


class TACProposal(BaseModel):
    """Technical formalization plus a plain-language back-translation."""

    estimand_type: Literal["ATT", "ATE", "LATE"] = "ATT"
    treatment: str
    treatment_coding: str
    outcome: str
    outcome_unit: str
    population: str
    time_scope: str
    expected_sign: Literal["positive", "negative", "ambiguous"]
    identification_assumption: str = "parallel_trends"
    h0: str
    h1: str
    implied_result_translation: str
    confirmation_question: str
    needs_clarification: bool = False
    clarification_question: str | None = None

    def to_estimand(self) -> Estimand:
        return Estimand(
            estimand_type=self.estimand_type,
            treatment=self.treatment,
            treatment_coding=self.treatment_coding,
            outcome=self.outcome,
            outcome_unit=self.outcome_unit,
            population=self.population,
            time_scope=self.time_scope,
            expected_sign=self.expected_sign,
            identification_assumption=self.identification_assumption,
            h0=self.h0,
            h1=self.h1,
        )


class FrozenEstimand(BaseModel):
    """Approved estimand artifact: immutable body plus deterministic hash."""

    model_config = {"frozen": True}

    estimand: Estimand
    freeze_hash: str


def draft_tac_proposal(
    *,
    research_story: str,
    available_columns: list[str],
    declaration: SocraticDeclaration,
) -> TACProposal:
    """Create the TAC proposal after the Socratic declaration exists."""
    if not available_columns:
        raise ValueError("TAC proposal needs at least one available column.")

    prompt = (
        "Research story:\n"
        f"{research_story}\n\n"
        "Available columns:\n"
        f"{available_columns}\n\n"
        "User Socratic declaration made before seeing the agent:\n"
        f"{declaration.model_dump()}\n\n"
        "Return a TACProposal. Map conceptual treatment/outcome to explicit columns, "
        "translate the implied result back to plain language, and ask one targeted "
        "confirmation question. If the mapping is uncertain, set needs_clarification=true "
        "and provide clarification_question."
    )
    agent = build_agent(
        ModelRole.JUDGE,
        system_prompt=(
            "You formalize causal estimands. Be conservative: never invent unavailable "
            "columns, preserve the user's expected_sign, and fail by asking a targeted "
            "clarification when treatment/outcome mapping is unclear."
        ),
        output_type=TACProposal,
    )
    return agent.run_sync(prompt).output


def freeze_estimand(proposal: TACProposal, *, approved: bool) -> FrozenEstimand:
    """Freeze the estimand only after explicit user approval."""
    if proposal.needs_clarification:
        question = proposal.clarification_question or proposal.confirmation_question
        raise ValueError(f"Estimand cannot be frozen; targeted confirmation needed: {question}")
    if not approved:
        raise ValueError(
            f"Estimand was not frozen; user approval is required: {proposal.confirmation_question}"
        )

    estimand = proposal.to_estimand()
    return FrozenEstimand(estimand=estimand, freeze_hash=estimand.freeze_hash())


def validate_estimand_spec_mapping(
    frozen: FrozenEstimand,
    spec: Specification,
    *,
    available_columns: list[str] | None = None,
) -> None:
    """Fail loud unless the frozen estimand maps cleanly to the specification."""
    estimand = frozen.estimand
    errors: list[str] = []

    if spec.treatment != estimand.treatment_coding:
        errors.append(
            f"spec.treatment={spec.treatment!r} does not match "
            f"estimand.treatment_coding={estimand.treatment_coding!r}"
        )
    if spec.outcome != estimand.outcome:
        errors.append(
            f"spec.outcome={spec.outcome!r} does not match estimand.outcome={estimand.outcome!r}"
        )

    if available_columns is not None:
        cols = set(available_columns)
        missing = [c for c in (estimand.treatment_coding, estimand.outcome) if c not in cols]
        if missing:
            errors.append(f"estimand columns are missing from available_columns: {missing}")

    if estimand.identification_assumption == "parallel_trends" and spec.estimator == "OLS":
        errors.append("parallel_trends estimands cannot cleanly map to an OLS specification")

    if errors:
        raise ValueError("Estimand -> Specification mapping error: " + "; ".join(errors))


def elicit_estimand(
    research_story: str,
    available_columns: list[str],
    declaration: SocraticDeclaration,
    *,
    approved: bool,
) -> FrozenEstimand:
    """Run Socratic + TAC elicitation and freeze the estimand on approval."""
    proposal = draft_tac_proposal(
        research_story=research_story,
        available_columns=available_columns,
        declaration=declaration,
    )
    return freeze_estimand(proposal, approved=approved)
